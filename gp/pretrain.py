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
from .terrain import hf_height
from .wys import _heading, _irot, _rot
from .wys_terrain import COLS


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
    #: Ramp strips: (grade in degrees, columns); the last columns of the map. Ramp clips start on the strip of their
    #: grade (|grade| within ``ramp_match`` degrees), everything else on flat columns.
    ramps: tuple = ((5.0, 2), (10.0, 2))
    ramp_match: float = 1.5
    #: Share of the worlds on ramps (split evenly over the grades that have clips; none without ramp clips).
    ramp_share: float = 0.3
    #: The imitation teacher (to distil into the L1 student): the policy also sees the clip, per tap (control steps
    #: ahead) its joint angles, pelvis height, pelvis orientation, feet relative to the pelvis, pelvis velocity and
    #: pelvis position, each as the error from the body now. Teacher runs set ``warp`` 0: the clip plays at its rate.
    teacher: bool = False
    taps: tuple = (1, 2, 3, 5, 8, 14, 24, 40)

    def ramp_columns(self) -> dict[int, float]:
        """Column -> grade for the ramp strips."""
        cols, c = {}, COLS
        for g, k in reversed(self.ramps):
            for _ in range(k):
                c -= 1
                cols[c] = g
        return cols


class Pretrain(Ladder):
    name = "pretrain"

    def layout(self) -> list[tuple]:
        base = super().layout()
        if not self.c.teacher:
            return base
        k = next(i for i, b in enumerate(base) if b[0].startswith("critic_"))
        nj, polar, axial = self.nj, [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]
        ref = []
        for t in self.c.taps:
            ref += [(f"ref_joint[{t}]", nj, list(self.jperm), list(self.jsign)), (f"ref_height[{t}]", 1, [0], [1.0]),
                    (f"ref_orientation[{t}]", 3, [0, 1, 2], axial), (f"ref_feet[{t}]", 6, [3, 4, 5, 0, 1, 2], polar * 2),
                    (f"ref_velocity[{t}]", 3, [0, 1, 2], polar), (f"ref_position[{t}]", 3, [0, 1, 2], polar)]
        return base[:k] + ref + base[k:] + [("critic_" + b[0], *b[1:]) for b in ref]

    def setup(self, env: Any) -> None:
        super().setup(env)
        c, dev = self.c, self.dev
        self.pos_off = torch.zeros(self.n, 3, device=dev)
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
        self._ramp_setup()
        n = self.n
        self.ref_f = torch.zeros(n, dtype=torch.long, device=dev)
        self.ref_start = torch.zeros(n, dtype=torch.long, device=dev)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)
        self.far = torch.zeros(n, dtype=torch.long, device=dev)
        self.tilted = torch.zeros(n, dtype=torch.long, device=dev)
        self._ended = torch.zeros(n, dtype=torch.bool, device=dev)

    def _ramp_setup(self) -> None:
        """Start pools by ground grade (level, then each ramp grade) and every world's column and pool."""
        c, D, dev = self.c, self.D, self.dev
        grades = [0.0] + [g for g, _ in c.ramps]
        pools: list[list[int]] = [[] for _ in grades]
        for clip, p0, p1 in zip(self.ex_clips, self.ex_pstart[:-1], self.ex_pstart[1:], strict=True):
            if D["mode"][clip] not in c.start_modes:
                continue
            g = abs(float(D["grade_deg"][clip])) if str(D["terrain_class"][clip]) != "level" else 0.0
            near = [abs(g - q) for q in grades]
            if min(near) <= c.ramp_match:
                pools[int(np.argmin(near))] += range(int(p0), int(p1))
        self.pools = [torch.as_tensor(p, dtype=torch.long, device=dev) for p in pools]
        ramp_cols = c.ramp_columns()
        flat_cols = [k for k in range(COLS) if k not in ramp_cols]
        live = [gi for gi in range(1, len(grades)) if len(self.pools[gi])]
        n = self.n
        n_ramp = int(round(n * c.ramp_share)) if live else 0
        share = [n - n_ramp] + [0] * (len(grades) - 1)
        for k, gi in enumerate(live):
            share[gi] = n_ramp // len(live) + (k < n_ramp % len(live))
        wg, col = [], []
        for gi, m in enumerate(share):
            cols = flat_cols if gi == 0 else [k for k, g in ramp_cols.items() if g == grades[gi]]
            wg += [gi] * m
            col += [cols[r % len(cols)] for r in range(m)]
        self.wgrade = torch.as_tensor(wg, device=dev)
        self.col = torch.as_tensor(col, device=dev)
        print(f"[pretrain] start windows by grade {dict(zip(grades, [len(p) for p in self.pools], strict=True))} pairs; "
              f"worlds {dict(zip(grades, share, strict=True))}", flush=True)

    def place(self, s: Any, i: torch.Tensor) -> None:
        """A random frame from the start pool of the world's ground grade, its velocities kept, its pelvis over a
        random point of the tile's centre (+-0.5 m) at the clip's own height over its ground (plus its reset lift), on
        the ground straight under the pelvis (flat or a ramp: no steps to clear)."""
        c, k = self.c, len(i)
        j = torch.empty(k, dtype=torch.long, device=self.dev)
        g = self.wgrade[i]
        for gi, pool in enumerate(self.pools):
            m = g == gi
            if bool(m.any()):
                j[m] = pool[torch.randint(len(pool), (int(m.sum()),), device=self.dev, generator=self.gen)]
        f = self.ex_frame[j]
        q, qd = self.ex_q[f].clone(), self.ex_qd[f].clone()
        dx, dz = self._sym((k,), c.reset_pos), self._sym((k,), c.reset_pos)
        off = self._offset[i]
        under = hf_height(self.hf_h, self.hf_b, self.T.cell, self.T.origin, dx[:, None] + off[:, :1], dz[:, None] + off[:, 1:])[:, 0]
        p = self.pel
        q[:, p["ty"]] += self.ex_lift[f] - self.ex_gy[f] + under + 0.01
        q[:, p["tx"]], q[:, p["tz"]] = dx, dz
        self.nominal[i] = self.q0[self.jidx] + self._sym((k, self.nj), c.nominal_jitter)
        s.qpos[i], s.qvel[i] = q, qd
        self.prev_qvel[i] = qd
        self.prev_action[i] = 0.0
        self.start_pair[i] = j

    def reset(self, s: Any, worlds: torch.Tensor) -> None:
        super().reset(s, worlds)
        i = worlds.nonzero().squeeze(1)
        if len(i) == 0:
            return
        f = self.ex_frame[self.start_pair[i]]
        self.ref_f[i], self.ref_start[i], self.t[i] = f, f, 0
        tr = [self.pel["tx"], self.pel["ty"], self.pel["tz"]]
        self.pos_off[i] = s.qpos[i][:, tr] - self.ex_q[f][:, tr]          # the reset's placement
        self.far[i], self.tilted[i], self._ended[i] = 0, 0, False

    def observation(self, s: Any) -> torch.Tensor:
        obs = super().observation(s)
        if not self.c.teacher:
            return obs
        n = sum(b[1] for b in super().layout() if not b[0].startswith("critic_"))
        ref = self._ref_features(s).nan_to_num(0.0, 0.0, 0.0)
        return torch.cat([obs[:, :n], ref, obs[:, n:], ref], 1)

    def _ref_features(self, s: Any) -> torch.Tensor:
        """Per tap, the clip ``t`` control steps ahead as errors from the body now (the teacher's view of the clip)."""
        r, p = self.root, self.pel
        tr = [p["tx"], p["ty"], p["tz"]]
        q = s.body_quat[:, r]
        ys = _yaw(_heading(q))
        com = s.body_com_pos()

        def rel(x: torch.Tensor, root: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
            return _irot(yaw[:, None].expand(-1, x.shape[1], -1), x - root[:, None])

        h_s = s.body_pos[:, r, 1] - self._pelvis_ground(s)
        feet_s = rel(com[:, self.feet], s.body_pos[:, r], ys).flatten(1)
        v_s = _irot(ys, s.qvel[:, tr])
        out = []
        for t in self.c.taps:
            f = torch.minimum(self.ref_f + t, self.ex_last_f[self.ref_f])
            yr = _yaw(self.ex_psi_f[f])
            out += [self.ex_q[f][:, self.jidx] - s.qpos[:, self.jidx],
                    ((self.ex_xp[f, r, 1] - self.ex_gy[f]) - h_s)[:, None],
                    _rotvec(_qmul(_qconj(q), self.ex_xq[f, r])),
                    rel(self.ex_com[f][:, self.feet], self.ex_xp[f, r], yr).flatten(1) - feet_s,
                    _irot(yr, self.ex_qd[f][:, tr]) - v_s,
                    _irot(ys, self.ex_q[f][:, tr] + self.pos_off - s.qpos[:, tr])]
        return torch.cat(out, 1)

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

    def clip_inputs(self, a: int, b: int) -> torch.Tensor:
        """The estimator's input (``system_state``) at reference frames ``a..b-1`` of one clip, from the clip itself:
        pelvis COM velocity and angular velocity (central differences) in the pelvis frame, gravity there, the command
        (the clip's speed, a zero turn rate: stage 0 steers toward the clip's own heading), joints from the standing
        pose and joint speeds."""
        c, dt, r = self.c, self.c.dt, self.root
        qr = self.ex_xq[a:b, r].cpu()
        v = torch.gradient(self.ex_com[a:b, r].cpu(), spacing=dt, dim=0)[0]
        dq = _qmul(qr[2:], _qconj(qr[:-2]))
        w = _rotvec(torch.cat([dq[:1], dq, dq[-1:]])) / (2 * dt)
        cmd = torch.zeros(b - a, 3)
        cmd[:, 0] = self.ex_speed_f[a:b].cpu()
        jidx = self.jidx.cpu()
        return torch.cat([_irot(qr, v), _irot(qr, w), _irot(qr, torch.tensor([0.0, -1.0, 0.0]).expand(b - a, 3)),
                          cmd[:, list(c.command_dims)], self.ex_q[a:b].cpu()[:, jidx] - self.q0[self.jidx].cpu(),
                          self.ex_qd[a:b].cpu()[:, jidx]], 1)

    def clip_estimator_windows(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """The foot placement estimator's training set from the clips, per clip window: :meth:`clip_inputs` and the
        label (each foot's next strictly-future landing, (forward, lateral) in the pelvis frame, as the simulation's
        labels), frames whose next landing lies past the window's end dropped."""
        from . import fpm as FPM
        D, r = self.D, self.root
        rows = np.concatenate([np.arange(D["start"][k], D["start"][k] + D["frames"][k]) for k in self.ex_clips])
        contact = torch.as_tensor(D["plan_contact"][rows])
        xp, xq = self.ex_xp.cpu(), self.ex_xq.cpu()
        first, last = self.ex_first_f.cpu(), self.ex_last_f.cpu()
        out = []
        for a in torch.unique(first).tolist():
            b = int(last[a]) + 1
            land, ok = FPM.next_landings(contact[a:b], xp[a:b][:, self.feet])
            y = _irot(xq[a:b, r][:, None].expand(-1, 2, -1), land - xp[a:b, r][:, None])[..., [0, 2]].flatten(1)
            keep = ok.all(1)
            out.append((self.clip_inputs(a, b)[keep], y[keep]))
        return out

    def clip_estimator_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`clip_estimator_windows`, every window together."""
        w = self.clip_estimator_windows()
        return torch.cat([x for x, _ in w]), torch.cat([y for _, y in w])

    def done(self, s: Any) -> tuple[torch.Tensor, torch.Tensor]:
        fell, out = super().done(s)
        return fell, out | (self._ended & ~fell)
