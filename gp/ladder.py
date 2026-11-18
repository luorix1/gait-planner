"""Rungs L0 and L1 of the footstep planner ladder (docs/plan_page/planner-ladder.html), build B1: Watch Your Step's task
(gp.wys) on our muscle body, with walking learned from human clips (AB01 only, stage 1). L1 = the paper (blind
guesser, two 7 x 7 patches); L0 = the whole 33 x 21 map in the actor's input instead, no guesser.

Changes from gp.wys (plan v2 and the ladder page):
- **Policy inputs** add each muscle's activation, fiber length and speed (per optimal fiber length), tendon force (per
  max isometric force) and the excitation it receives after the motor delay: the imitation recipe's muscle state.
  The critic reads them too.
- **Reward** = w_task x the paper's speed and turn-rate tracking only (their weights x ``task_scale``, so the two top
  out at 1 like the style reward) + w_style x the AMP style reward (gp.amp: GAIT_amp's judge, on AB01 pose pairs
  25 ms apart) - the recipe's effort, as our other muscle studies use it (metabolic power 0.02 per W/kg, summed cubed
  activation 0.02, compartment balance 0.02). As GAIT_amp: no fall penalty, no other robot terms.
- **Episodes start** from a random frame of a level AB01 clip (walking, start-stop, standing) on the world's tile,
  with the clip's own velocities; the command starts at that frame's forward speed and heading.
- **40 Hz control** (the clips' rate; the paper's is 50 Hz), commands forward only (0 to the fastest walking clip,
  1.5 m/s) with heading turns, stage-1 terrain (rough, slopes, stepping stones), no helper pull by default, and a
  fall when the pelvis drops below ``fall_height`` over the ground (the paper's 0.7 rad tilt misses a collapse)
  or a thigh or shank touches the ground.

The style judge's input, one pair per control step (a policy step, or two consecutive clip frames): each frame's
pelvis height over the ground under it, the up axis in the pelvis frame, the 17 joint angles, and both calcanei
relative to the pelvis in its heading frame; then the pelvis displacement in the first frame's heading frame and the
heading change, both per second. Poses only, through one function on both sides, so a feature cannot mean one thing
for the clips and another for the simulation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import data as gpdata
from .amp import AMP, AMPConfig
from .wys import WatchYourStep, WysConfig, _heading, _irot, _wrap
from .wys_terrain import Terrain

STAGE1 = ("random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv", "stepping_stones")
MUSCLE_BLOCKS = ("activation", "fiber_length", "fiber_velocity", "muscle_force", "received")


@dataclass
class LadderConfig(WysConfig):
    dt: float = 0.025                     # 40 Hz, the clips' rate
    lin_x: tuple = (0.0, 1.5)             # forward only, up to AB01's fastest walking clip (1.53 m/s)
    lin_y: tuple = (0.0, 0.0)
    assist_offset: float = 0.0            # no helper pull (episodes start mid-stride from clip frames)
    perception: str = "patches"           # "patches" (L1) | "map" (L0)
    types: tuple = STAGE1                 # terrain columns worlds are spread over
    w_task: float = 0.5
    w_style: float = 0.5
    task_scale: float = 0.5               # the paper's weights x this: the two tracking terms sum to at most 1
    #: Of the paper's terms only speed and turn-rate tracking are paid (user 2026-11-18, as GAIT_amp: task + style +
    #: effort, a fall ends the episode with no penalty). Dropped: the standing bonus (tracking a zero command already
    #: pays for standing still), upright, vertical speed, roll and pitch rates, joint acceleration, excitation change and
    #: leg contact (the style judge and the effort cover posture, smoothness and cost; a leg on the ground is a fall),
    #: the joint-torque penalty (the effort prices the muscles) and the hip/waist deviation terms (they cost AB01's own
    #: walking 0.55 + 0.31 rad per step). The dropped terms are still computed and logged.
    w: dict = field(default_factory=lambda: {k: v if k.startswith("track_") else 0.0 for k, v in WysConfig().w.items()})
    fall_on_leg_contact: bool = True      # a thigh or shank on the ground ends the episode (kneeling, sitting, a trip)
    w_metabolic: float = 0.02             # the imitation recipe's effort (ImitationConfig.recommended)
    w_activation_cubed_sum: float = 0.02
    w_balance: float = 0.02
    balance_eps: float = 0.01
    fall_height: float = 0.5              # m, pelvis over the ground under it
    start_modes: tuple = ("walk", "start_stop", "stand")      # level clips only
    style_modes: tuple = ("walk", "start_stop", "stand")      # all grades
    amp: AMPConfig = field(default_factory=AMPConfig)


def _yaw(psi: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(psi)
    return torch.stack([z, torch.sin(psi / 2), z, torch.cos(psi / 2)], -1)


class Ladder(WatchYourStep):
    name = "ladder"

    def __init__(self, terrain: Terrain, D: dict[str, Any], config: LadderConfig | None = None, *, seed: int = 0,
                 log_dir: Any = None, foot_model: Any = None) -> None:
        super().__init__(terrain, config or LadderConfig(), seed=seed, log_dir=log_dir, foot_model=foot_model)
        self.D = D

    # -- layout ---------------------------------------------------------------------------------------------------
    def layout(self) -> list[tuple]:
        body, scan, hmap = self.body_blocks, self.scan_block, self.map_block
        mus = [(n, self.nmus, self.mperm, [1.0] * self.nmus) for n in MUSCLE_BLOCKS]
        seen = scan if self.c.perception == "patches" else hmap
        return [*body, *mus, seen] + [("critic_" + b[0], *b[1:]) for b in [*body, *mus, hmap, scan]]

    def pair_mirror(self) -> tuple[np.ndarray, np.ndarray]:
        """``(perm, sign)`` mirroring one style-judge pair: ``x[perm] * sign``."""
        nj = self.nj
        perm = [0, 1, 2, 3] + [4 + j for j in self.jperm]
        sign = [1.0, 1.0, 1.0, -1.0] + list(self.jsign)
        o = 4 + nj
        perm += [o + 3, o + 4, o + 5, o, o + 1, o + 2]
        sign += [1.0, 1.0, -1.0] * 2
        d = len(perm)
        P = perm + [d + k for k in perm] + [2 * d, 2 * d + 1, 2 * d + 2, 2 * d + 3]
        S = sign + sign + [1.0, 1.0, -1.0, -1.0]
        return np.array(P, np.int64), np.array(S, np.float32)

    # -- setup ----------------------------------------------------------------------------------------------------
    def setup(self, env: Any) -> None:
        super().setup(env)
        c, dev = self.c, self.dev
        allowed = torch.as_tensor([k for k, t in enumerate(self.T.types) if t in c.types], device=dev)
        self.col = allowed[torch.div(torch.arange(self.n, device=dev) * len(allowed), self.n, rounding_mode="floor")]
        self.e_y = torch.tensor([0.0, 1.0, 0.0], device=dev)
        self._expert(env.model)
        self._balance_tables(list(env.model.muscle_names))
        self.start_pair = torch.zeros(self.n, dtype=torch.long, device=dev)
        self.amp_buf = torch.zeros(c.steps_per_iter, self.n, self.amp.dim, device=dev)
        self.amp_ok = torch.zeros(c.steps_per_iter, self.n, dtype=torch.bool, device=dev)
        self.amp_stats: dict[str, float] = {}

    def _expert(self, model: Any) -> None:
        """The human pairs (every clip of ``style_modes``) and the start frames (the level clips of ``start_modes``)."""
        from sprint.recipes.motion import forward_kinematics_np
        c, D, dev = self.c, self.D, self.dev
        assert [str(n) for n in D["coord_names"]] == list(self.coords), "clip coordinates differ from the model's"
        clips = np.flatnonzero(np.isin(D["mode"], c.style_modes))
        rows = [np.arange(D["start"][k], D["start"][k] + D["frames"][k]) for k in clips]
        q = np.concatenate([D["qpos"][r] for r in rows]).astype(np.float32)
        fk = [forward_kinematics_np(model, q[a:a + 50_000]) for a in range(0, len(q), 50_000)]
        xp, xq = np.concatenate([f[0] for f in fk]), np.concatenate([f[1] for f in fk])
        n = np.array([len(r) for r in rows])
        first = np.r_[0, np.cumsum(n)[:-1]]
        gy = np.concatenate([gpdata.ground_under(D, k, xp[a:a + m, self.root, 0], xp[a:a + m, self.root, 2])
                             for k, a, m in zip(clips, first, n, strict=True)])
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
        self.ex_q, self.ex_qd = T(q), T(np.concatenate([D["qvel"][r] for r in rows]))
        self.ex_gy, self.ex_lift = T(gy), T(np.concatenate([D["lift"][r] for r in rows]))
        f, p, psi = self.pose(self.ex_q, T(xp), T(xq), self.ex_gy)
        a = torch.as_tensor(np.concatenate([np.arange(s, s + m - 1) for s, m in zip(first, n, strict=True)]), device=dev)
        pairs = self.pair((f[a], p[a], psi[a]), (f[a + 1], p[a + 1], psi[a + 1]))
        pstart = np.r_[0, np.cumsum(n - 1)]
        self.ex_frame, self.ex_psi = a, psi[a]
        self.ex_vf = pairs[:, 2 * f.shape[1]]                                   # forward speed, heading frame
        level = np.isin(D["mode"][clips], c.start_modes) & (D["terrain_class"][clips] == "level")
        self.st_start = torch.as_tensor(pstart[:-1][level], device=dev)
        self.st_count = torch.as_tensor(np.diff(pstart)[level], device=dev)
        self.amp = AMP(c.amp, pairs, pstart, self.pair_mirror(), device=dev)
        print(f"[ladder] style judge: {len(clips)} clips, {len(pairs)} pairs of {pairs.shape[1]} numbers; "
              f"starts from {int(level.sum())} level clips", flush=True)

    def _balance_tables(self, mn: list[str]) -> None:
        """Muscles split into compartments, as the imitation recipe groups them for its balance cost."""
        groups: dict[str, list[int]] = {}
        for i, nm in enumerate(mn):
            side = nm[-2:] if nm[-2:].lower() in ("_r", "_l") else ""
            base = nm[: len(nm) - len(side)] if side else nm
            key = ("vasti" if re.match(r"vas_?(int|lat|med)", base) else
                   "triceps_surae" if re.match(r"(soleus|gas_?(med|lat)|(med|lat)_gas)$", base) else
                   re.sub(r"\d+$", "", base) if re.search(r"\d$", base) else "")
            if key:
                groups.setdefault(key + side, []).append(i)
        G = [g for g in groups.values() if len(g) > 1]
        K = max(len(g) for g in G)
        self.bal_idx = torch.as_tensor([g + [g[0]] * (K - len(g)) for g in G], device=self.dev)
        self.bal_mask = torch.as_tensor([[1.0] * len(g) + [0.0] * (K - len(g)) for g in G], device=self.dev)

    # -- the style judge's input ----------------------------------------------------------------------------------
    def pose(self, qpos: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor,
             gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(one frame's pose features, root position, heading) from ``qpos [B, nv]``, body origins and orientations
        ``[B, nb, 3|4]`` and the ground under the pelvis ``[B]``."""
        r = self.root
        q = quat[:, r]
        psi = _heading(q)
        up = _irot(q, self.e_y.expand(len(q), 3))
        feet = _irot(_yaw(psi)[:, None].expand(-1, 2, -1), pos[:, self.feet] - pos[:, r:r + 1])
        # the root position is cloned: a view of the state would move with the next physics step
        return (torch.cat([(pos[:, r, 1] - gy)[:, None], up, qpos[:, self.jidx], feet.flatten(1)], 1),
                pos[:, r].clone(), psi)

    def pair(self, a: tuple, b: tuple) -> torch.Tensor:
        (f0, p0, psi0), (f1, p1, psi1) = a, b
        dp = _irot(_yaw(psi0), p1 - p0) / self.c.dt
        return torch.cat([f0, f1, dp, (_wrap(psi1 - psi0) / self.c.dt)[:, None]], 1)

    def _pelvis_ground(self, s: Any) -> torch.Tensor:
        p = s.body_pos[:, self.root]
        return self.ground(p[:, 0:1], p[:, 2:3])[:, 0]

    # -- hooks ----------------------------------------------------------------------------------------------------
    def observation(self, s: Any) -> torch.Tensor:
        c = self.c
        b = self.body_state(s)
        hmap, scan_a, scan_c = self.scans(s, b)
        noisy, clean = self.body_obs(b)
        mus = [s.activation, s.fiber_length / s.optimal_fiber_length, s.fiber_velocity / s.optimal_fiber_length,
               s.muscle_force / s.max_isometric_force, s.received]
        if c.perception == "patches":
            seen = scan_a
        else:                                   # the whole map, with the patches' drift and noise
            seen = (hmap + self.z_drift + self._sym(hmap.shape, c.scan_noise)).clamp(-1, 1)
        return torch.cat([*noisy, *mus, seen, *clean, *mus, hmap, scan_c], 1).nan_to_num(0.0, 0.0, 0.0)

    def before_step(self, s: Any) -> None:
        super().before_step(s)
        self._prev = self.pose(s.qpos, s.body_pos, s.body_quat, self._pelvis_ground(s))

    def reward(self, s: Any) -> torch.Tensor:
        c = self.c
        b = self.body_state(s)
        terms = self.terms(s, b)
        gy = self._pelvis_ground(s)
        low = (s.body_pos[:, self.root, 1] - gy) < c.fall_height
        self._fell = self._fell | low
        if c.fall_on_leg_contact:
            self._fell = self._fell | (terms["undesired_contacts"] > 0)
        terms["termination"] = self._fell.float()
        task = c.task_scale * sum(c.w[k] * v for k, v in terms.items())
        pair = self.pair(self._prev, self.pose(s.qpos, s.body_pos, s.body_quat, gy)).nan_to_num(0.0, 0.0, 0.0)
        style = self.amp.reward(pair)
        metabolic = s.metabolic_power.sum(1) / s.total_mass
        a3 = s.activation.clamp_min(0).pow(3).sum(1)
        ag = s.activation[:, self.bal_idx]                                              # [N, G, K]
        k = self.bal_mask.sum(1)
        mean = (ag * self.bal_mask).sum(2) / k
        var = ((ag - mean[..., None]).pow(2) * self.bal_mask).sum(2) / k
        balance = (var / (mean.pow(2) + c.balance_eps ** 2)).sum(1)
        effort = c.w_metabolic * metabolic + c.w_activation_cubed_sum * a3 + c.w_balance * balance
        r = c.w_task * task + c.w_style * style - effort
        learn = c.training and c.amp.train_disc
        if learn:
            t = self.steps % c.steps_per_iter
            self.amp_buf[t], self.amp_ok[t] = pair, s.diverged == 0
        self.advance(s, b, terms)
        if learn and self.steps % c.steps_per_iter == 0:
            self.amp_stats = self.amp.train(self.amp_buf[self.amp_ok])
        self.metrics.update(r_task=task, r_style=style, metabolic=metabolic, activation_cubed_sum=a3, balance=balance,
                            effort=effort, fell_low=low.float(),
                            **{k_: torch.tensor(v) for k_, v in self.amp_stats.items()})
        return r

    def _fpm_step(self, s: Any, b: dict[str, torch.Tensor]) -> None:
        if self.c.perception == "patches":                  # L0 has no guesser to refit
            super()._fpm_step(s, b)

    def place(self, s: Any, i: torch.Tensor) -> None:
        """A random frame of a level start clip, its velocities kept, its pelvis over a random point of the tile's
        centre (+-0.5 m) at the clip's own height over its ground (plus its reset lift), on the highest ground under
        the feet's reach."""
        c, k = self.c, len(i)
        sc = torch.randint(len(self.st_start), (k,), device=self.dev, generator=self.gen)
        j = self.st_start[sc] + (torch.rand(k, device=self.dev, generator=self.gen) * self.st_count[sc]).long()
        f = self.ex_frame[j]
        q, qd = self.ex_q[f].clone(), self.ex_qd[f].clone()
        dx, dz = self._sym((k,), c.reset_pos), self._sym((k,), c.reset_pos)
        p = self.pel
        q[:, p["ty"]] += self.ex_lift[f] - self.ex_gy[f] + self.ground_max(i, dx, dz) + 0.01
        q[:, p["tx"]], q[:, p["tz"]] = dx, dz
        self.nominal[i] = self.q0[self.jidx] + self._sym((k, self.nj), c.nominal_jitter)
        s.qpos[i], s.qvel[i] = q, qd
        self.prev_qvel[i] = qd
        self.prev_action[i] = 0.0
        self.start_pair[i] = j

    def reset(self, s: Any, worlds: torch.Tensor) -> None:
        super().reset(s, worlds)
        i = worlds.nonzero().squeeze(1)
        if len(i) == 0 or not self.c.training:          # evaluation keeps the command it was given
            return
        j = self.start_pair[i]
        v = self.ex_vf[j].clamp(*self.c.lin_x)
        self.cmd[i, 0], self.cmd[i, 1] = v, 0.0
        self.heading_target[i] = self.ex_psi[j]
        self.standing[i] = v < self.c.stand_threshold
        self.turning[i] = False
