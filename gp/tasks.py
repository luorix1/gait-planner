"""The two SPRINT tasks, both on SPRINT's imitation recipe (``sprint.recipes.imitation.Imitation``):

- :class:`TerrainImitation`, the full-pose DeepMimic baseline: the recipe's reward, observation and falls, on each
  clip's own terrain (a SPRINT profile, or a tile of one shared heightfield for ground a profile cannot describe:
  turns, platforms, gaps), with a 2D height scan in the observation and resets raised by the clip's per-frame lift
  (no contact sphere starts inside the terrain).
- :class:`PlanTracking`, the planner's task: the actor sees its body, the height scan and the plan (per foot:
  down or not, the foothold, time to touchdown and lift-off; the centre-of-mass target; and the pelvis orientation,
  as the recipe's root orientation error) at the recipe's lookahead taps, never the joint angles; the critic also
  reads the full reference (privileged). Reward: matching the planned contacts and footholds, staying in the
  centre-of-mass band and the pelvis-orientation band, the frozen style prior's score, minus the recipe's effort
  terms. The pelvis orientation fixes where the hips are, so with the footholds it leaves each leg little more than
  its knee's swivel and the ankle (user 2026-11-15: too many free degrees of freedom without it). Falls stay the recipe's (pelvis 0.3 m from the clip's or tilted 60 deg from it).

Free clips (gp.free: generated courses, no reference motion; ``D["free"]``) start at their first frame only. In
:class:`PlanTracking` they fall by losing the plan's centre of mass (gp.free.FREE_FALL_M) or tilting 60 deg, and the
critic reads a flag that the episode has no clip. The baseline has nothing to imitate on them: give it none.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sprint.recipes.imitation import _POLAR, Imitation, ImitationConfig, _heading, _qconj, _qmul, _qrot, _rotvec

from . import data as gpdata
from .free import FREE_FALL_M
from .smp import Prior, StyleReward, W
from .terrain import Ground

#: Height scan: terrain height relative to the root at these offsets in the root's heading frame (m): forward x lateral,
#: forward-major. Lateral offsets are symmetric, so the left/right mirror swaps them.
SCAN_FORWARD = tuple(np.round(np.arange(-0.4, 1.21, 0.1), 2))
SCAN_LATERAL = (-0.2, 0.0, 0.2)


def recipe_config(**kw: Any) -> ImitationConfig:
    """SPRINT's recommended imitation recipe without EMG (40 Hz, 75 physics steps, implicit normal contact,
    SCONE motor delays, metabolic + cubed activation + compartment balance effort)."""
    c = ImitationConfig.recommended(emg=False)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class TerrainImitation(Imitation):
    """The full-pose baseline on each clip's own terrain. ``D``: :func:`gp.data.load`."""

    def __init__(self, D: dict[str, Any], config: ImitationConfig | None = None, *, starts: list[Any] | None = None,
                 seed: int = 0) -> None:
        super().__init__(gpdata.motions(D), config or recipe_config(), starts=starts, seed=seed)
        self.D = D

    def bind(self, model: Any) -> None:
        super().bind(model)
        if self.symmetric:
            # the left/right mirror PPO's symmetry loss uses: SPRINT's axis-based coord_signs gets Rajagopal's hip
            # adduction, hip rotation and knee wrong (gp.data.coord_mirror_signs); rebuild the blocks with FK signs
            self.csign = gpdata.coord_mirror_signs(model)
            self.blocks = self._blocks(model)
            self._obs_dim = sum(b[1] for b in self.blocks)
        nf, nl = len(SCAN_FORWARD), len(SCAN_LATERAL)
        perm = [f * nl + (nl - 1 - k) for f in range(nf) for k in range(nl)]
        self.blocks.append(("height_scan", nf * nl, perm, [1.0] * (nf * nl)))
        self._obs_dim += nf * nl
        bn = list(model.body_names)
        self.key = [bn.index(b) for b in gpdata.KEY_BODIES]
        h = model.host
        sph = self.L.contact_spheres
        self.sph_body = h["geom_body"][sph].astype(np.int64)
        self.sph_pos = np.asarray(h["geom_pos"][sph], np.float32)
        self.sph_r = np.asarray(h["geom_size"][sph][:, 0], np.float32)
        self.sph_side = np.array([gpdata.SIDES.index(bn[b][-1]) for b in self.sph_body])
        self.feet_side = np.array([gpdata.SIDES.index(bn[b][-1]) for b in self.L.feet])

    def setup(self, env: Any) -> None:
        # ponytail: the recipe's setup runs forward kinematics on every reference frame in one Data (one world per
        # frame: ~18 GB for v1's 1.5M frames on a GPU); chunk it for the duration. Upstream fix: chunk in SPRINT.
        import sprint.recipes.motion as motion
        whole = motion.forward_kinematics_np

        def chunked(model: Any, qpos: np.ndarray, n: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
            parts = [whole(model, qpos[i:i + n]) for i in range(0, len(qpos), n)]
            return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

        motion.forward_kinematics_np = chunked
        try:
            super().setup(env)
        finally:
            motion.forward_kinematics_np = whole
        D, dev = self.D, self.dev
        env.data.set_terrains([np.c_[x[:n], y[:n]] for x, y, n in zip(D["profile_x"], D["profile_y"], D["profile_n"],
                                                                       strict=True)])
        # clips on the shared heightfield (D["hf"]: gp.terrain.atlas; D["terrain_kind"] 1, D["hf_offset"] their tile)
        from sprint import _backend as B
        from sprint.terrain import HFIELD
        self.G = Ground.of(D, dev)
        self.on_hf, self.hf = self.G.on_hf, self.G.hf
        if self.hf is not None:
            hf = D["hf"]
            env.data.set_heightfield(hf["heights"], hf["cell"], hf["origin"], hf["block"])
            self.hf_off = self.G.off
            self._offset = B.to_torch(env.data.terrain_offset)
        n = len(self.on_hf)
        self.terrain_of_clip = torch.where(self.on_hf, torch.full((n,), HFIELD, device=dev),
                                           torch.arange(n, device=dev)).to(torch.int32)
        self.lift = torch.as_tensor(D["lift"], dtype=torch.float32, device=dev)
        self.free_clip = torch.as_tensor(np.asarray(D.get("free", np.zeros(len(self.motions), bool))), device=dev)
        g = torch.as_tensor([(f, k) for f in SCAN_FORWARD for k in SCAN_LATERAL], dtype=torch.float32, device=dev)
        self.scan_f, self.scan_l = g[:, 0], g[:, 1]
        self.key_b = torch.as_tensor(self.key, device=dev)
        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
        self.p_contact, self.p_foothold, self.p_com = T(D["plan_contact"], torch.bool), T(D["foothold"]), T(D["com"])
        self.sph_b, self.sph_p, self.sph_rad = T(self.sph_body, torch.long), T(self.sph_pos), T(self.sph_r)
        side = T(self.sph_side, torch.long)
        self.sph_w = torch.stack([(side == i).float() / (side == i).sum() for i in range(2)])     # [2, S] means
        fs = T(self.feet_side, torch.long)
        self.feet_w = torch.stack([(fs == i).float() for i in range(2)])                          # [2, feet] sums

    def ground(self, x: Any, z: Any) -> Any:
        """Terrain height under ``(x, z)`` ``[N, S]``, each world on its clip's terrain: its profile (which ignores z),
        or its tile of the shared heightfield."""
        return self.G(self.clip, x, z)

    def reset(self, s: Any, worlds: Any) -> None:
        super().reset(s, worlds)
        idx = worlds.nonzero().squeeze(1)
        if idx.numel():
            fr = idx[self.free_clip[self.clip[idx]]]
            if fr.numel():                             # free clips start where they are placed: frame 0, no noise
                self.frame[fr] = self.clip_start[self.clip[fr]]
                self.start_frame[fr] = 0
                s.qpos[fr], s.qvel[fr] = self.ref["qpos"][self.frame[fr]], self.ref["qvel"][self.frame[fr]]
            s.qpos[idx, self.L.vertical] += self.lift[self.frame[idx]]
            if self.hf is not None:                    # each heightfield clip stands on its own tile
                self._offset[idx] = self.hf_off[self.clip[idx]]

    def record(self, s: Any) -> Any:
        """This step's style record (:func:`gp.data.record_layout`), as :func:`gp.data.style_records` from data."""
        p, q = s.body_pos[:, self.root], s.body_quat[:, self.root]
        return torch.cat([p, q, s.qpos[:, self.tracked], s.body_pos[:, self.key_b].flatten(1),
                          self.ground(p[:, :1], p[:, 2:3])], 1)

    # -- the simulation's side of the plan (both tasks: the planner's reward, everyone's evaluation) ----------------
    def com(self, s: Any) -> Any:
        """Whole-body centre of mass ``[N, 3]``."""
        return (s.body_com_pos() * s.body_mass[:, None]).sum(1) / s.total_mass

    def sphere_bottoms(self, s: Any) -> Any:
        """Each foot contact sphere's lowest point ``[N, S, 3]``."""
        b = self.sph_b
        ctr = s.body_pos[:, b] + _qrot(s.body_quat[:, b], self.sph_p.expand(self.n, -1, -1))
        ctr[..., 1] -= self.sph_rad
        return ctr

    def foot_points(self, s: Any) -> Any:
        """Per side (l, r), the mean of its contact spheres' lowest points ``[N, 2, 3]``, as the plans' footholds."""
        return torch.einsum("ks,nsc->nkc", self.sph_w, self.sphere_bottoms(s))

    def foot_force(self, s: Any) -> Any:
        """Per side (l, r), the vertical ground force on its foot bodies ``[N, 2]`` (N)."""
        return torch.einsum("kf,nf->nk", self.feet_w, s.contact_force()[:, self.feet, 1])

    def observation(self, s: Any) -> Any:
        root = s.body_com_pos()[:, self.root]
        M = _heading(s.body_quat[:, self.root])                       # rows: forward, up, lateral (world)
        x = root[:, :1] + self.scan_f * M[:, 0, :1] + self.scan_l * M[:, 2, :1]
        z = root[:, 2:3] + self.scan_f * M[:, 0, 2:3] + self.scan_l * M[:, 2, 2:3]
        return torch.cat([super().observation(s), self.ground(x, z) - root[:, 1:2]], 1)


@dataclass
class PlanConfig:
    """The plan-tracking reward. Positive terms sum to 1 (style half of it, as MimicKit's SMP tasks)."""
    w_contact: float = 0.15
    w_foothold: float = 0.15
    tol_foothold: float = 0.10             # m, horizontal
    w_com: float = 0.10
    com_band: tuple[float, float, float] = (0.10, 0.05, 0.05)   # m half-widths: forward, up, lateral (heading frame)
    tol_com: float = 0.05                  # m beyond the band
    w_orient: float = 0.10                 # the pelvis orientation (the root's)
    orient_band: float = 0.10              # rad (6 deg) of rotation from the planned orientation, free
    tol_orient: float = 0.10               # rad beyond the band
    w_style: float = 0.50
    contact_bw: float = 0.05               # a foot is down above this fraction of body weight
    prior: str = ""                        # the style prior's file (train_prior.py)


class PlanTracking(TerrainImitation):
    """Track the plan; see the module docstring."""

    #: The reference itself: the critic's alone (the actor reads the plan, not the pose).
    CRITIC_ONLY = (*Imitation.CRITIC_ONLY, "root_position_error", "joint_angle_error",
                   "joint_velocity_error", "root_velocity_error", "feet_position_error", "free_episode")

    def __init__(self, D: dict[str, Any], config: ImitationConfig | None = None, plan: PlanConfig | None = None, *,
                 starts: list[Any] | None = None, seed: int = 0) -> None:
        super().__init__(D, config, starts=starts, seed=seed)
        self.plan = plan or PlanConfig()
        self.name = "plan_tracking"

    def bind(self, model: Any) -> None:
        super().bind(model)
        foot = [1.0, *_POLAR, 1.0, 1.0]                 # down, foothold (3), touchdown, lift-off
        for t in self.taps:
            self.blocks += [(f"plan_feet[{t}]", 12, [*range(6, 12), *range(6)], foot * 2),
                            (f"plan_com[{t}]", 3, [0, 1, 2], list(_POLAR))]
            self._obs_dim += 15
        self.blocks.append(("free_episode", 1, [0], [1.0]))
        self._obs_dim += 1

    def setup(self, env: Any) -> None:
        super().setup(env)
        D, dev = self.D, self.dev
        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
        self.p_td, self.p_lo, self.p_record = T(D["t_touchdown"]), T(D["t_liftoff"]), T(D["record"])
        self.band = T(self.plan.com_band)
        if not self.plan.prior:
            raise ValueError("PlanConfig.prior: the style prior's file (train_prior.py)")
        self.style = StyleReward(Prior.load(self.plan.prior, dev))
        self.ring = torch.zeros(self.n, W + 1, self.p_record.shape[1], device=dev)
        self.lost = torch.zeros(self.n, dtype=torch.long, device=dev)

    # -- hooks ---------------------------------------------------------------------------------------------------
    def reset(self, s: Any, worlds: Any) -> None:
        super().reset(s, worlds)
        idx = worlds.nonzero().squeeze(1)
        if idx.numel():                     # the window so far: the clip's own frames, clamped at its start
            self.lost[idx] = 0
            f = self.frame[idx][:, None] + torch.arange(-W, 1, device=self.dev)
            f = torch.maximum(f, self.clip_start[self.clip[idx]][:, None])
            self.ring[idx] = self.p_record[f]

    def observation(self, s: Any) -> Any:
        obs = super().observation(s)
        rq = s.body_quat[:, self.root]
        root = s.body_com_pos()[:, self.root]
        com = self.com(s)
        M = _heading(rq)
        last = self.clip_last[self.clip] + self.taps[-1]
        parts = [obs]
        for t in self.taps:
            f = torch.minimum(self.frame + t, last)
            fh = torch.einsum("nij,nkj->nki", M, self.p_foothold[f] - root[:, None])
            feet = torch.cat([self.p_contact[f].float()[..., None], fh, self.p_td[f][..., None], self.p_lo[f][..., None]], -1)
            parts += [feet.flatten(1), torch.einsum("nij,nj->ni", M, self.p_com[f] - com)]
        parts.append(self.free_clip[self.clip].float()[:, None])
        return torch.nan_to_num(torch.cat(parts, 1), 0.0, 0.0, 0.0)

    def reward(self, s: Any) -> Any:
        super().reward(s)                   # advances the frame; falls and the imitation metrics (logged, not paid)
        c, pc, f = self.config, self.plan, self.frame
        free = self.free_clip[self.clip]
        if free.any():                      # no clip pelvis to stay near: a free episode falls by losing the plan
            lost = (self.com(s) - self.p_com[f])[:, [0, 2]].norm(dim=1) > FREE_FALL_M
            self.lost = torch.where(lost, self.lost + 1, 0)
            self._fell = torch.where(free, (self.lost >= c.fall_steps) | (self.tilted >= c.fall_steps), self._fell)
        down_ref = self.p_contact[f]                                                          # [N, 2]
        down = self.foot_force(s) > pc.contact_bw * self.bw
        r_contact = (down == down_ref).float().mean(1)
        d = (self.foot_points(s) - self.p_foothold[f])[..., [0, 2]].norm(dim=-1)              # [N, 2]
        hit = torch.exp(-0.5 * (d / pc.tol_foothold).pow(2))
        n_ref = down_ref.float().sum(1)
        r_foothold = torch.where(n_ref > 0, (hit * down_ref).sum(1) / n_ref.clamp_min(1), torch.ones_like(n_ref))
        M = _heading(s.body_quat[:, self.root])
        e = torch.einsum("nij,nj->ni", M, self.com(s) - self.p_com[f]).abs()
        r_com = torch.exp(-0.5 * ((e - self.band).clamp_min(0) / pc.tol_com).pow(2).sum(1))
        eo = _rotvec(_qmul(_qconj(s.body_quat[:, self.root]), self.ref["root_quat"][f])).norm(dim=1)     # rad
        r_orient = torch.exp(-0.5 * ((eo - pc.orient_band).clamp_min(0) / pc.tol_orient).pow(2))
        self.ring = torch.cat([self.ring[:, 1:], self.record(s)[:, None]], 1)
        r_style, sds = self.style(self.ring)
        metabolic = s.metabolic_power.sum(1) / s.total_mass
        a3 = s.activation.clamp_min(0).pow(3)
        effort = c.w_metabolic * metabolic + c.w_activation_cubed * a3.mean(1) + c.w_activation_cubed_sum * a3.sum(1)
        if c.w_balance and self.balance_groups:
            ag = s.activation[:, self.bal_idx]
            k = self.bal_mask.sum(1)
            mean = (ag * self.bal_mask).sum(2) / k
            var = ((ag - mean[..., None]).pow(2) * self.bal_mask).sum(2) / k
            effort = effort + c.w_balance * (var / (mean.pow(2) + c.balance_eps ** 2)).sum(1)
        plan = pc.w_contact * r_contact + pc.w_foothold * r_foothold + pc.w_com * r_com + pc.w_orient * r_orient
        self.metrics.update(plan=plan, r_contact=r_contact, r_foothold=r_foothold, r_com=r_com, r_orient=r_orient,
                            orient_error_deg=torch.rad2deg(eo), r_style=r_style,
                            sds_loss=sds, effort=effort, foothold_error=d.mean(1), com_error=e.norm(dim=1))
        return plan + pc.w_style * r_style - effort

    def plan_dict(self) -> dict[str, Any]:
        return asdict(self.plan)
