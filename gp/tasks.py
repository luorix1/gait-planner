"""The two SPRINT tasks, both on SPRINT's imitation recipe (``sprint.recipes.imitation.Imitation``):

- :class:`TerrainImitation`, the full-pose DeepMimic baseline: the recipe's reward, observation and falls, on each
  clip's own terrain profile, with a height scan in the observation and resets raised by the clip's per-frame
  lift (no contact sphere starts inside the terrain).
- :class:`PlanTracking`, the planner's task: the actor sees its body, the height scan and the plan (per foot:
  down or not, the foothold, time to touchdown and lift-off; and the centre-of-mass target) at the recipe's
  lookahead taps, never the full pose; the critic also reads the full reference (privileged). Reward: matching
  the planned contacts and footholds, staying in the centre-of-mass band, the frozen style prior's score, minus
  the recipe's effort terms. Falls stay the recipe's (pelvis 0.3 m from the clip's or tilted 60 deg from it).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sprint.recipes.imitation import _POLAR, Imitation, ImitationConfig, _qrot

from . import data as gpdata
from .smp import Prior, StyleReward, W
from .terrain import height

#: Height scan: terrain height under these offsets ahead of the root (m, along x: every v1 terrain runs along x).
SCAN = tuple(np.round(np.arange(-0.4, 1.21, 0.1), 2))


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
        n = len(SCAN)
        self.blocks.append(("height_scan", n, list(range(n)), [1.0] * n))
        self._obs_dim += n
        self.key = [list(model.body_names).index(b) for b in gpdata.KEY_BODIES]

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
        self.terrain_of_clip = torch.arange(len(self.motions), dtype=torch.int32, device=dev)
        self.tX = torch.as_tensor(D["profile_x"], dtype=torch.float32, device=dev)
        self.tY = torch.as_tensor(D["profile_y"], dtype=torch.float32, device=dev)
        self.lift = torch.as_tensor(D["lift"], dtype=torch.float32, device=dev)
        self.scan = torch.as_tensor(SCAN, dtype=torch.float32, device=dev)
        self.key_b = torch.as_tensor(self.key, device=dev)

    def ground(self, x: Any) -> Any:
        """Terrain height under ``x`` ``[N, S]``, each world on its clip's profile."""
        return height(self.tX, self.tY, self.clip, x)

    def reset(self, s: Any, worlds: Any) -> None:
        super().reset(s, worlds)
        idx = worlds.nonzero().squeeze(1)
        if idx.numel():
            s.qpos[idx, self.L.vertical] += self.lift[self.frame[idx]]

    def record(self, s: Any) -> Any:
        """This step's style record (:func:`gp.data.record_layout`), as :func:`gp.data.style_records` from data."""
        p, q = s.body_pos[:, self.root], s.body_quat[:, self.root]
        return torch.cat([p, q, s.qpos[:, self.tracked], s.body_pos[:, self.key_b].flatten(1), self.ground(p[:, :1])], 1)

    def observation(self, s: Any) -> Any:
        root = s.body_com_pos()[:, self.root]
        scan = self.ground(root[:, :1] + self.scan) - root[:, 1:2]
        return torch.cat([super().observation(s), scan], 1)


@dataclass
class PlanConfig:
    """The plan-tracking reward. Positive terms sum to 1 (style half of it, as MimicKit's SMP tasks)."""
    w_contact: float = 0.15
    w_foothold: float = 0.15
    tol_foothold: float = 0.10             # m, horizontal
    w_com: float = 0.20
    com_band: tuple[float, float, float] = (0.10, 0.05, 0.05)   # m half-widths: forward, up, lateral (heading frame)
    tol_com: float = 0.05                  # m beyond the band
    w_style: float = 0.50
    contact_bw: float = 0.05               # a foot is down above this fraction of body weight
    prior: str = ""                        # the style prior's file (train_prior.py)


class PlanTracking(TerrainImitation):
    """Track the plan; see the module docstring."""

    #: The reference itself: the critic's alone (the actor reads the plan, not the pose).
    CRITIC_ONLY = (*Imitation.CRITIC_ONLY, "root_position_error", "root_orientation_error", "joint_angle_error",
                   "joint_velocity_error", "root_velocity_error", "feet_position_error")

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
        bn = list(model.body_names)
        h = model.host
        sph = self.L.contact_spheres
        self.sph_body = h["geom_body"][sph].astype(np.int64)
        self.sph_pos = np.asarray(h["geom_pos"][sph], np.float32)
        self.sph_r = np.asarray(h["geom_size"][sph][:, 0], np.float32)
        self.sph_side = np.array([gpdata.SIDES.index(bn[b][-1]) for b in self.sph_body])
        self.feet_side = np.array([gpdata.SIDES.index(bn[b][-1]) for b in self.L.feet])

    def setup(self, env: Any) -> None:
        super().setup(env)
        D, dev = self.D, self.dev
        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
        self.p_contact, self.p_foothold = T(D["plan_contact"], torch.bool), T(D["foothold"])
        self.p_td, self.p_lo, self.p_com = T(D["t_touchdown"]), T(D["t_liftoff"]), T(D["com"])
        self.p_record = T(D["record"])
        self.sph_b, self.sph_p, self.sph_rad = T(self.sph_body, torch.long), T(self.sph_pos), T(self.sph_r)
        side = T(self.sph_side, torch.long)
        self.sph_w = torch.stack([(side == i).float() / (side == i).sum() for i in range(2)])     # [2, S] means
        fs = T(self.feet_side, torch.long)
        self.feet_w = torch.stack([(fs == i).float() for i in range(2)])                          # [2, feet] sums
        self.band = T(self.plan.com_band)
        if not self.plan.prior:
            raise ValueError("PlanConfig.prior: the style prior's file (train_prior.py)")
        self.style = StyleReward(Prior.load(self.plan.prior, dev))
        self.ring = torch.zeros(self.n, W + 1, self.p_record.shape[1], device=dev)

    # -- the simulation's side of the plan -----------------------------------------------------------------------
    def _com(self, s: Any) -> Any:
        return (s.body_com_pos() * s.body_mass[:, None]).sum(1) / s.total_mass

    def _foot_points(self, s: Any) -> Any:
        """Per side, the mean of its contact spheres' lowest points ``[N, 2, 3]`` (as the plans' footholds)."""
        b = self.sph_b
        ctr = s.body_pos[:, b] + _qrot(s.body_quat[:, b], self.sph_p.expand(self.n, -1, -1))
        ctr[..., 1] -= self.sph_rad
        return torch.einsum("ks,nsc->nkc", self.sph_w, ctr)

    # -- hooks ---------------------------------------------------------------------------------------------------
    def reset(self, s: Any, worlds: Any) -> None:
        super().reset(s, worlds)
        idx = worlds.nonzero().squeeze(1)
        if idx.numel():                     # the window so far: the clip's own frames, clamped at its start
            f = self.frame[idx][:, None] + torch.arange(-W, 1, device=self.dev)
            f = torch.maximum(f, self.clip_start[self.clip[idx]][:, None])
            self.ring[idx] = self.p_record[f]

    def observation(self, s: Any) -> Any:
        obs = super().observation(s)
        rq = s.body_quat[:, self.root]
        root = s.body_com_pos()[:, self.root]
        com = self._com(s)
        from sprint.recipes.imitation import _heading
        M = _heading(rq)
        last = self.clip_last[self.clip] + self.taps[-1]
        parts = [obs]
        for t in self.taps:
            f = torch.minimum(self.frame + t, last)
            fh = torch.einsum("nij,nkj->nki", M, self.p_foothold[f] - root[:, None])
            feet = torch.cat([self.p_contact[f].float()[..., None], fh, self.p_td[f][..., None], self.p_lo[f][..., None]], -1)
            parts += [feet.flatten(1), torch.einsum("nij,nj->ni", M, self.p_com[f] - com)]
        return torch.nan_to_num(torch.cat(parts, 1), 0.0, 0.0, 0.0)

    def reward(self, s: Any) -> Any:
        super().reward(s)                   # advances the frame; falls and the imitation metrics (logged, not paid)
        c, pc, f = self.config, self.plan, self.frame
        from sprint.recipes.imitation import _heading
        down_ref = self.p_contact[f]                                                          # [N, 2]
        fy = torch.einsum("kf,nf->nk", self.feet_w, s.contact_force()[:, self.feet, 1])
        down = fy > pc.contact_bw * self.bw
        r_contact = (down == down_ref).float().mean(1)
        d = (self._foot_points(s) - self.p_foothold[f])[..., [0, 2]].norm(dim=-1)              # [N, 2]
        hit = torch.exp(-0.5 * (d / pc.tol_foothold).pow(2))
        n_ref = down_ref.float().sum(1)
        r_foothold = torch.where(n_ref > 0, (hit * down_ref).sum(1) / n_ref.clamp_min(1), torch.ones_like(n_ref))
        M = _heading(s.body_quat[:, self.root])
        e = torch.einsum("nij,nj->ni", M, self._com(s) - self.p_com[f]).abs()
        r_com = torch.exp(-0.5 * ((e - self.band).clamp_min(0) / pc.tol_com).pow(2).sum(1))
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
        plan = pc.w_contact * r_contact + pc.w_foothold * r_foothold + pc.w_com * r_com
        self.metrics.update(plan=plan, r_contact=r_contact, r_foothold=r_foothold, r_com=r_com, r_style=r_style,
                            sds_loss=sds, effort=effort, foothold_error=d.mean(1), com_error=e.norm(dim=1))
        return plan + pc.w_style * r_style - effort

    def plan_dict(self) -> dict[str, Any]:
        return asdict(self.plan)
