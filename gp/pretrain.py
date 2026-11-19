"""Stage 0 of B1: imitation pretraining for the ladder's policy. The policy keeps L1's inputs (body state, command,
muscle states, two patches) and learns to walk by tracking AB01's level clips on flat ground with SPRINT's imitation
reward (``ImitationConfig.recommended``'s terms, weights and tolerances, STC's effort). Stage 1 (gp.ladder) then starts
from it (``train_wys.py --init-from``).

The reference is in the reward only, never in the policy's input:
- the command follows the clip: its forward speed averaged over ``speed_window`` frames each side, and its heading
  (the yaw-rate command steers toward it, as in the ladder);
- the reference clock follows the policy's gait: each step it advances one frame plus the best of ``-warp..+warp``
  frames by joint angles, and stays within ``warp_max`` frames of real time. A slightly different cadence is not
  punished, and stalling the reference is bounded.
- Positions are compared relative to the pelvis in its heading frame (feet, landmarks) and as height over the ground
  (pelvis), not in the world: with no reference in the input a world position drifts.

Why (2026-11-18): from L1's untrained start (every muscle at 0.05) no uniform drive keeps this body up (0.05 -> 0.8 s,
0.5 -> 0.25 s), so a survival-driven reward pushes the extensors down from the first update. Imitation pays for the
right joints and pelvis height every step, which is how STC trains this exact model (Rajagopal_lower_collision_v3).
Not ported from STC: the EMG term (no EMG in data/ab01), excitation priming at reset, error-weighted starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .amp import AMPConfig
from .ladder import Ladder, LadderConfig, _yaw
from .wys import _heading, _irot, _rot


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion product, (x, y, z, w)."""
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat([aw * bv + bw * av + torch.cross(av, bv, dim=-1), aw * bw - (av * bv).sum(-1, keepdim=True)], -1)


def _qconj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], -1)


def _rotvec(q: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis x angle) of a unit quaternion, the short way round."""
    q = torch.where(q[..., 3:] < 0, -q, q)
    v, w = q[..., :3], q[..., 3:]
    n = v.norm(dim=-1, keepdim=True)
    return v * torch.where(n > 1e-8, 2.0 * torch.atan2(n, w) / n.clamp_min(1e-8), torch.full_like(n, 2.0))


@dataclass
class PretrainConfig(LadderConfig):
    w_style: float = 0.0
    amp: AMPConfig = field(default_factory=lambda: AMPConfig(train_disc=False))
    #: Effort: SPRINT's recommended imitation effort, STC's weights on this body (0.02 each).
    w_metabolic: float = 0.02
    w_activation_cubed_sum: float = 0.02
    w_balance: float = 0.02
    episode_s: float = 10.0               # STC: 400 steps
    resample_s: float = 1e9               # the command follows the clip, never redrawn
    standing_frac: float = 0.0
    turning_frac: float = 0.0
    disturbance_n: float = 0.0            # no pushes, friction spread or constant push while learning to walk
    push_s: float = 1e9
    friction_scale: tuple = (1.0, 1.0)
    fall_on_leg_contact: bool = False
    # tracking terms (ImitationConfig.recommended): weights sum to 1
    w_joint: float = 0.30
    tol_joint: float = 0.15               # rad
    w_velocity: float = 0.15
    tol_vel_joint: float = 1.5            # rad/s
    tol_vel_root_ang: float = 1.0         # rad/s, the pelvis rotation coordinates
    tol_vel_root_lin: float = 0.5         # m/s, heading frame
    w_height: float = 0.05                # the recipe's root position, vertical only (height over the ground)
    tol_height: float = 0.10              # m
    w_orientation: float = 0.10
    tol_orientation: float = 0.20         # rad
    w_feet: float = 0.20
    tol_feet: float = 0.10                # m, calcaneus COMs relative to the pelvis
    w_landmarks: float = 0.15
    tol_landmarks: float = 0.10           # m, COMs relative to the pelvis
    landmarks: tuple = ("femur_l", "femur_r", "tibia_l", "tibia_r", "toes_l", "toes_r", "torso")
    w_torso: float = 0.05
    tol_torso: float = 0.10               # rad
    warp: int = 2                         # frames the reference clock may slip or gain per step
    warp_max: int = 10                    # frames (0.25 s) it may stray from real time
    speed_window: int = 20                # frames each side for the command's speed (1 s window)
    fall_height_error: float = 0.15       # m, pelvis height off the reference's (STC cfg_4)
    fall_angle_deg: float = 60.0          # pelvis orientation off the reference's
    fall_steps: int = 3                   # in a row (STC)


class Pretrain(Ladder):
    name = "pretrain"

    def setup(self, env: Any) -> None:
        super().setup(env)
        c, dev = self.c, self.dev
        bn = list(env.model.body_names)
        self.torso = bn.index("torso")
        self.lm = [bn.index(b) for b in c.landmarks]
        com_local = torch.as_tensor(np.asarray(env.model.host["body_com"])[: len(bn)], dtype=torch.float32, device=dev)
        self.ex_com = self.ex_xp + _rot(self.ex_xq, com_local.expand(len(self.ex_xp), -1, -1))
        # the command's speed per frame: the pair's forward speed (a clip's last frame repeats its last pair),
        # averaged over +-speed_window frames inside the clip
        F = len(self.ex_q)
        sp = torch.zeros(F, device=dev)
        sp[self.ex_frame] = self.ex_vf
        last = self.ex_last_f == torch.arange(F, device=dev)
        sp[last] = sp[last.nonzero().squeeze(1) - 1]
        cs = torch.cat([torch.zeros(1, device=dev), sp.cumsum(0)])
        i = torch.arange(F, device=dev)
        lo = torch.maximum(self.ex_first_f, i - c.speed_window)
        hi = torch.minimum(self.ex_last_f, i + c.speed_window)
        self.ex_speed_f = (cs[hi + 1] - cs[lo]) / (hi - lo + 1).float()
        n = self.n
        self.ref_f = torch.zeros(n, dtype=torch.long, device=dev)
        self.ref_start = torch.zeros(n, dtype=torch.long, device=dev)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)
        self.far = torch.zeros(n, dtype=torch.long, device=dev)
        self.tilted = torch.zeros(n, dtype=torch.long, device=dev)
        self._ended = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, s: Any, worlds: torch.Tensor) -> None:
        super().reset(s, worlds)
        i = worlds.nonzero().squeeze(1)
        if len(i) == 0:
            return
        f = self.ex_frame[self.start_pair[i]]
        self.ref_f[i], self.ref_start[i], self.t[i] = f, f, 0
        self.far[i], self.tilted[i], self._ended[i] = 0, 0, False

    def before_step(self, s: Any) -> None:
        f = self.ref_f
        self.heading_target[:] = self.ex_psi_f[f]
        self.cmd[:, 0], self.cmd[:, 1] = self.ex_speed_f[f], 0.0
        self.standing[:] = False
        super().before_step(s)

    def _warp(self, s: Any) -> torch.Tensor:
        """The next reference frame: one frame on, plus the best of -warp..+warp by joint angles, inside the clip and
        within ``warp_max`` frames of real time."""
        c = self.c
        cand = (self.ref_f + 1)[:, None] + torch.arange(-c.warp, c.warp + 1, device=self.dev)
        wall = self.ref_start + self.t
        hi = torch.minimum(self.ex_last_f[self.ref_f], wall + c.warp_max)
        lo = torch.minimum(torch.maximum(self.ex_first_f[self.ref_f], wall - c.warp_max), hi)
        cand = torch.maximum(torch.minimum(cand, hi[:, None]), lo[:, None])
        d = (s.qpos[:, None, self.jidx] - self.ex_q[cand][..., self.jidx]).pow(2).mean(-1)
        return cand.gather(1, d.argmin(1, keepdim=True)).squeeze(1)

    def track(self, s: Any, f: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """The tracking terms against reference frames ``f``, the pelvis height error and the pelvis orientation error
        angle."""
        c, r, p = self.c, self.root, self.pel
        rot, tr = [p["tilt"], p["list"], p["rotation"]], [p["tx"], p["ty"], p["tz"]]
        ej = s.qpos[:, self.jidx] - self.ex_q[f][:, self.jidx]
        psi_s, psi_r = _heading(s.body_quat[:, r]), self.ex_psi_f[f]
        ys, yr = _yaw(psi_s), _yaw(psi_r)
        dv = torch.cat([(s.qvel[:, self.jidx] - self.ex_qd[f][:, self.jidx]) / c.tol_vel_joint,
                        (s.qvel[:, rot] - self.ex_qd[f][:, rot]) / c.tol_vel_root_ang,
                        (_irot(ys, s.qvel[:, tr]) - _irot(yr, self.ex_qd[f][:, tr])) / c.tol_vel_root_lin], 1)
        eh = (s.body_pos[:, r, 1] - self._pelvis_ground(s)) - (self.ex_xp[f, r, 1] - self.ex_gy[f])
        eo = _rotvec(_qmul(_qconj(s.body_quat[:, r]), self.ex_xq[f, r]))
        et = _rotvec(_qmul(_qconj(s.body_quat[:, self.torso]), self.ex_xq[f, self.torso]))
        com = s.body_com_pos()

        def rel(x: torch.Tensor, root: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
            return _irot(yaw[:, None].expand(-1, x.shape[1], -1), x - root[:, None])

        ef = rel(com[:, self.feet], s.body_pos[:, r], ys) - rel(self.ex_com[f][:, self.feet], self.ex_xp[f, r], yr)
        el = rel(com[:, self.lm], s.body_pos[:, r], ys) - rel(self.ex_com[f][:, self.lm], self.ex_xp[f, r], yr)
        terms = dict(joint=torch.exp(-0.5 * (ej / c.tol_joint).pow(2).mean(1)),
                     velocity=torch.exp(-0.5 * dv.pow(2).mean(1)),
                     height=torch.exp(-0.5 * (eh / c.tol_height).pow(2)),
                     orientation=torch.exp(-0.5 * (eo / c.tol_orientation).pow(2).sum(1)),
                     feet=torch.exp(-0.5 * (ef / c.tol_feet).pow(2).sum(2).mean(1)),
                     landmarks=torch.exp(-0.5 * (el / c.tol_landmarks).pow(2).sum(2).mean(1)),
                     torso=torch.exp(-0.5 * (et / c.tol_torso).pow(2).sum(1)))
        return terms, eh, eo.norm(dim=1)

    def reward(self, s: Any) -> torch.Tensor:
        c = self.c
        b = self.body_state(s)
        terms = self.terms(s, b)                  # the paper's terms, logged only
        self.t += 1
        self.ref_f = self._warp(s)
        tr, eh, angle = self.track(s, self.ref_f)
        imitation = (c.w_joint * tr["joint"] + c.w_velocity * tr["velocity"] + c.w_height * tr["height"]
                     + c.w_orientation * tr["orientation"] + c.w_feet * tr["feet"] + c.w_landmarks * tr["landmarks"]
                     + c.w_torso * tr["torso"])
        effort, metabolic, a3, balance = self.effort(s)
        self.far = torch.where(eh.abs() > c.fall_height_error, self.far + 1, 0)
        self.tilted = torch.where(angle > np.radians(c.fall_angle_deg), self.tilted + 1, 0)
        low = (s.body_pos[:, self.root, 1] - self._pelvis_ground(s)) < c.fall_height
        self._fell = (self.far >= c.fall_steps) | (self.tilted >= c.fall_steps) | low
        self._ended = self.ref_f >= self.ex_last_f[self.ref_f]
        terms["termination"] = self._fell.float()
        self.advance(s, b, terms)
        ej = s.qpos[:, self.jidx] - self.ex_q[self.ref_f][:, self.jidx]
        self.metrics.update(imitation=imitation, metabolic=metabolic, activation_cubed_sum=a3, balance=balance,
                            effort=effort, fell_low=low.float(), height_error=eh.abs(),
                            joint_error_deg=torch.rad2deg(ej.pow(2).mean(1).sqrt()),
                            warp_offset=(self.ref_f - self.ref_start - self.t).float(),
                            **{f"i_{k}": v for k, v in tr.items()})
        return imitation - effort

    def done(self, s: Any) -> tuple[torch.Tensor, torch.Tensor]:
        fell, out = super().done(s)
        return fell, out | (self._ended & ~fell)
