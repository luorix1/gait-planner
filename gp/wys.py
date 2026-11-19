"""Watch Your Step (Heyrman et al. 2026, arXiv 2610.29612) on our Rajagopal muscle body in SPRINT: the authors'
environment (github.com/leggedrobotics/foothold-prediction-model, fpm/env_cfg.py) term by term, with the G1 swapped
for our body. The plan page lists what changes and why; in short:

- **Command** (their TurnInPlaceVelocityCommand): forward U(-1, 1), lateral U(-0.5, 0.5) m/s, a heading U(-pi, pi)
  tracked at yaw rate clip(0.5 * error, +-1) rad/s; 2 % standing (all zero), 2 % turning in place (no linear part);
  redrawn every 10 s and at reset.
- **Actor sees** (with their noise): root linear velocity (+-0.15, spikes), angular velocity x 0.2 (+-0.2, spikes),
  gravity in the root frame (+-0.05), the command, joint angles from nominal (+-0.01), joint speeds x 0.05 (+-1.5),
  the last action (90 excitations), and the foothold scan: per foot, its predicted landing (forward, lateral) and the
  7 x 7 heights around it, from a 1.6 x 1.0 m, 5 cm map under the torso (heights = torso height - ground - 0.5); the
  patches drift +-5 cm (xy, per episode) and +-5 cm (z), plus +-0.1 noise, clipped to +-1.
- **Critic** also sees the same terms without noise, the whole 33 x 21 map, and the scan without noise or z drift (the
  authors' critic shares the actor's xy drift through module state; so does ours).
- **Foothold predictor** (gp.fpm): blind, from the clean 46-number body state; outputs pinned at (0, -+0.1) m for the
  first 500 iterations, then refit from scratch on 500k fresh samples every 500 iterations (24 steps per iteration).
- **Reward** x dt (IsaacLab scales every term by the step): velocity and yaw-rate tracking, standing still at the
  nominal pose, upright posture, and the penalties of their table; "torques" are the muscles' joint torques, contacts
  off the feet are thigh and shank spheres.
- **Episodes**: 20 s, fall at a tilt over 0.7 rad (-20). Reset on the world's tile: root +-0.5 m, any yaw, random
  root velocity; joints at nominal.
- **Randomisation**: friction (scale 0.5-2 on the model's 0.9/0.6, i.e. dynamic 0.3-1.2), nominal pose +-0.01 rad,
  a constant +-7 N force on the pelvis per episode, a velocity push every 4 s. Not ported: body mass and torso CoM
  (need per-world bodies), PD gains and joint friction (no PD).
- **Curricula**: a world moves up a level after walking over 4 m, down after under half its commanded distance
  (IsaacLab terrain_levels_vel); an upward pull of 0.5 x weight x (1 - mean level / 4) on the torso while training.

Axes: SPRINT is y-up; the root frame is (forward x, up y, right z). The command's lateral speed is the authors'
(positive = left); the reward compares it with -v_z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sprint.env import TorchTask
from sprint.recipes.imitation import mirrored

from . import data as gpdata
from . import fpm as FPM
from .terrain import hf_height
from .wys_terrain import ROWS, Terrain


@dataclass
class WysConfig:
    dt: float = 0.02                     # 50 Hz
    episode_s: float = 20.0
    lin_x: tuple = (-1.0, 1.0)
    lin_y: tuple = (-0.5, 0.5)
    ang_z: tuple = (-1.0, 1.0)
    heading_stiffness: float = 0.5
    resample_s: float = 10.0
    standing_frac: float = 0.02
    turning_frac: float = 0.02
    # map and patches
    scan_res: float = 0.05
    scan_size: tuple = (1.6, 1.0)
    patch: int = 7
    xy_drift: float = 0.05
    z_drift: float = 0.05
    scan_noise: float = 0.1
    bootstrap: tuple = ((0.0, -0.1), (0.0, 0.1))    # (forward, lateral right) per foot l, r: the authors' (0, +-0.1)
    # predictor schedule (PPO iterations of `steps_per_iter` steps)
    steps_per_iter: int = 24
    bootstrap_iters: int = 500
    refit_iters: int = 500
    refit_samples: int = 500_000
    # reward weights (their RewardsCfg)
    w: dict = field(default_factory=lambda: dict(
        track_lin_vel_xy=1.0, track_ang_vel_z=1.0, stand_still=2.5, lin_vel_z=-0.5, ang_vel_xy=-0.05,
        joint_torques=-1.0e-5, joint_acc=-2.5e-8, action_rate=-0.01, undesired_contacts=-1.0,
        joint_deviation_legs=-1.0, joint_deviation_waist=-1.0, upright=0.25, termination=-20.0))
    track_std2: float = 0.25
    stand_threshold: float = 0.1
    stand_std: float = 2.0
    upright_frac: float = 0.95           # target = this x our standing pelvis-over-foot height (G1: 0.72 m)
    upright_std: float = 0.06
    contact_threshold: float = 1.0       # N
    action_rate_max: float = 5.0
    fall_tilt: float = 0.7               # rad
    # events
    friction_scale: tuple = (0.5, 2.0)
    nominal_jitter: float = 0.01
    disturbance_n: float = 7.0
    push_s: float = 4.0
    push_lin: tuple = (0.5, 0.5, 0.2)    # forward, lateral, up (m/s)
    push_ang: tuple = (0.52, 0.52, 0.78)  # roll, pitch, yaw (rad/s)
    reset_pos: float = 0.5
    reset_lin: tuple = (0.5, 0.3, 0.2)
    reset_ang: tuple = (0.5, 0.5, 0.2)
    # curricula
    max_init_level: int = 2
    assist_offset: float = 0.5
    assist_cutoff: float = 4.0
    # obs noise (their PolicyCfg)
    n_lin: float = 0.15
    n_ang: float = 0.2
    n_grav: float = 0.05
    n_q: float = 0.01
    n_qd: float = 1.5
    spike_p: float = 1e-4
    spike_lin: float = 5.0
    spike_ang: float = 10.0
    training: bool = True                # False: no assist, pushes, disturbance or predictor refits (evaluation)
    eval_assist: float = 0.0             # with training False: a fixed helper pull (x weight), to watch a run as it trains
    #: evaluation: the tiles (row, col) worlds take in turn at each reset, in place of the curriculum's
    eval_tiles: tuple = ()


def _rot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def _irot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _rot(torch.cat([-q[..., :3], q[..., 3:]], -1), v)


def _heading(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.atan2(-2 * (x * z - w * y), 1 - 2 * (y * y + z * z))


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class WatchYourStep(TorchTask):
    name = "watch_your_step"

    def __init__(self, terrain: Terrain, config: WysConfig | None = None, *, seed: int = 0,
                 log_dir: str | Path | None = None, foot_model: str | Path | None = None) -> None:
        self.T, self.c, self.seed = terrain, config or WysConfig(), seed
        self.log_dir = Path(log_dir) if log_dir else None
        self.foot_model_path = Path(foot_model) if foot_model else None
        self.metrics: dict[str, Any] = {}

    # -- layout -------------------------------------------------------------------------------------------------------
    def bind(self, model: Any) -> None:
        c = self.c
        cn, bn, mn = list(model.coord_names), list(model.body_names), list(model.muscle_names)
        self.coords = cn
        self.joints = [k for k, n in enumerate(cn) if not n.startswith("pelvis_")]
        jn = [cn[k] for k in self.joints]
        self.nj, self.nmus = len(self.joints), int(model.nmus)
        self.root, self.torso = bn.index("pelvis"), bn.index("torso")
        self.feet = [bn.index("calcn_l"), bn.index("calcn_r")]
        self.toes = [bn.index("toes_l"), bn.index("toes_r")]
        self.legs = [bn.index(b) for b in ("femur_l", "femur_r", "tibia_l", "tibia_r")]
        self.pel = {n: cn.index(f"pelvis_{n}") for n in ("tilt", "list", "rotation", "tx", "ty", "tz")}
        self.dev_legs = [jn.index(n) for n in ("hip_adduction_l", "hip_adduction_r", "hip_rotation_l", "hip_rotation_r")]
        self.dev_waist = [jn.index(n) for n in ("lumbar_extension", "lumbar_bending", "lumbar_rotation")]
        csign = gpdata.coord_mirror_signs(model)
        jperm = [jn.index(mirrored(n)) for n in jn]
        jsign = [float(csign[self.joints[k]]) for k in range(self.nj)]
        mperm = [mn.index(mirrored(n)) for n in mn]
        self.nx, self.ny = (int(round(s / c.scan_res)) + 1 for s in c.scan_size)       # 33 forward x 21 lateral
        p = c.patch
        ppatch = [ix * p + (p - 1 - iy) for ix in range(p) for iy in range(p)]
        foot = lambda o: [o + 0, o + 1] + [o + 2 + k for k in ppatch]                  # noqa: E731
        n1 = 2 + p * p
        scan_perm = foot(n1) + foot(0)
        scan_sign = ([1.0, -1.0] + [1.0] * p * p) * 2
        map_perm = [ix * self.ny + (self.ny - 1 - iy) for ix in range(self.nx) for iy in range(self.ny)]
        polar, axial = ([0, 1, 2], [1.0, 1.0, -1.0]), ([0, 1, 2], [-1.0, -1.0, 1.0])
        body = [("lin_vel", 3, *polar), ("ang_vel", 3, *axial), ("gravity", 3, *polar),
                ("command", 3, [0, 1, 2], [1.0, -1.0, -1.0]), ("joint_pos", self.nj, jperm, jsign),
                ("joint_vel", self.nj, jperm, jsign), ("last_action", self.nmus, mperm, [1.0] * self.nmus)]
        self.body_blocks = body
        self.scan_block = ("foothold_scan", 2 * n1, scan_perm, scan_sign)
        self.map_block = ("height_scan", self.nx * self.ny, map_perm, [1.0] * len(map_perm))
        self.mperm, self.jperm, self.jsign = mperm, jperm, jsign
        self.state_dim = 12 + 2 * self.nj                                             # the predictor's input
        self.blocks = self.layout()
        self._obs_dim = sum(b[1] for b in self.blocks)
        self.n_actor = sum(b[1] for b in self.blocks if not b[0].startswith("critic_"))

    def layout(self) -> list[tuple]:
        """Observation blocks ``(name, size, mirror permutation, sign)``: the actor's, then the critic's (``critic_``)."""
        body, scan, hmap = self.body_blocks, self.scan_block, self.map_block
        return ([*body, scan] + [("critic_" + b[0], *b[1:]) for b in body]
                + [("critic_" + hmap[0], *hmap[1:]), ("critic_" + scan[0], *scan[1:])])

    def actor_columns(self) -> np.ndarray:
        return np.arange(self.n_actor, dtype=np.int64)

    def mirror_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        perm, sign, base = [], [], 0
        for _, size, p, s in self.blocks:
            perm += [base + i for i in p]
            sign += s
            base += size
        return np.array(perm, np.int64), np.array(sign, np.float32), np.array(self.mperm, np.int64)

    @property
    def symmetric(self) -> bool:
        return True

    def describe(self) -> str:
        out, o = [f"observation: {self._obs_dim} numbers ({self.n_actor} actor)"], 0
        for name, size, _, _ in self.blocks:
            out.append(f"  {o:4d}-{o + size - 1:4d}  {name} ({size})")
            o += size
        return "\n".join(out)

    # -- setup ------------------------------------------------------------------------------------------------------
    def setup(self, env: Any) -> None:
        from sprint import _backend as B
        from sprint.terrain import HFIELD
        c, n = self.c, env.num_worlds
        self._offset = B.to_torch(env.data.terrain_offset)
        dev = self._offset.device
        self.n, self.nw, self.dev = n, int(env.data.nworld), dev
        self.gen = torch.Generator(device=dev).manual_seed(self.seed)
        T = self.T
        env.data.set_heightfield(T.heights, T.cell, T.origin, T.block)
        B.to_torch(env.data.terrain_id)[:n] = HFIELD
        self.hf_h = torch.as_tensor(T.heights, device=dev)
        self.hf_b = torch.as_tensor(T.block, device=dev)
        self.centres = torch.as_tensor(T.centres, dtype=torch.float32, device=dev)            # [ROWS, COLS, 2]
        cols = self.centres.shape[1]
        self.col = torch.div(torch.arange(n, device=dev) * cols, n, rounding_mode="floor").clamp(max=cols - 1)
        self.level = torch.randint(0, c.max_init_level + 1, (n,), device=dev, generator=self.gen)
        h = env.model.host
        q0 = torch.as_tensor(np.asarray(h["coord_default"][: len(self.coords)], np.float32), device=dev)
        self.q0 = q0
        self.nominal = q0[self.joints].expand(n, -1).clone()
        self.jidx = torch.as_tensor(self.joints, device=dev)
        # muscle -> joint torque map (SPRINT's dof_mus_adr / dof_mus / dof_arm_slot, as its own kernel sums it)
        adr, mus, slot = (np.asarray(h[k]) for k in ("dof_mus_adr", "dof_mus", "dof_arm_slot"))
        dof_of = np.repeat(np.arange(len(adr) - 1), np.diff(adr))
        self.t_dof = torch.as_tensor(dof_of, device=dev)
        self.t_mus = torch.as_tensor(mus[: len(dof_of)], device=dev, dtype=torch.long)
        self.t_slot = torch.as_tensor(slot[: len(dof_of)], device=dev, dtype=torch.long)
        self._arm = B.to_torch(env.data.moment_arm)
        # standing pelvis-over-foot height for the upright term
        s = env.state
        from sprint.dynamics import forward_kinematics
        qp = s.qpos.clone()
        s.qpos[:] = q0
        forward_kinematics(env.model, env.data)
        stand = float((s.body_pos[0, self.root, 1] - s.body_pos[0, self.feet, 1].min()).item())
        s.qpos[:] = qp
        self.upright_target = c.upright_frac * stand
        self.pelvis_ty0 = float(q0[self.pel["ty"]])
        self.mass = float(s.body_mass.reshape(-1, s.body_mass.shape[-1])[0].sum())
        # per-world state
        z = lambda *k: torch.zeros(n, *k, device=dev)   # noqa: E731
        self.cmd, self.heading_target = z(3), z()
        self.standing, self.turning = torch.zeros(n, dtype=torch.bool, device=dev), torch.zeros(n, dtype=torch.bool, device=dev)
        self.cmd_timer, self.push_timer = z(), z()
        self.xy_drift, self.z_drift = z(2), z(1)
        self.disturb = z(3)
        self.prev_action, self.prev_qvel = z(self.nmus), z(len(self.coords))
        self.friction = self._uniform((n,), c.friction_scale)
        env.data.set_friction_scale(self.friction.cpu().numpy(), worlds=np.arange(n))
        g = torch.arange(-(self.nx // 2), self.nx // 2 + 1, device=dev) * c.scan_res
        lat = torch.arange(-(self.ny // 2), self.ny // 2 + 1, device=dev) * c.scan_res
        self.grid_f, self.grid_l = g.repeat_interleave(self.ny), lat.repeat(self.nx)          # forward-major
        k = c.patch // 2
        off = torch.arange(-k, k + 1, device=dev)
        self.off_x, self.off_y = off.repeat_interleave(c.patch), off.repeat(c.patch)
        # predictor
        self.fpm = FPM.NormalizedFootPredictor(FPM.FootPredictor(self.state_dim, 4),
                                               torch.zeros(self.state_dim), torch.ones(self.state_dim)).to(dev)
        self.fpm_ready = False
        if self.foot_model_path is not None:
            self.fpm.load_state_dict(torch.load(self.foot_model_path, map_location=dev))
            self.fpm_ready = True
        self.steps, self._eval_next = 0, 0
        self.buf: dict[str, list] | None = None
        self.record_steps = math.ceil(c.refit_samples / n)
        self.fpm_val = float("nan")
        # perturbation slots: 0 = the helper pull on the torso, 1 = the constant push on the pelvis
        from sprint.dynamics.perturb import Perturbation
        env.perturb([Perturbation("torso.com", duration=1e9), Perturbation("pelvis.com", duration=1e9)])
        self._pert = B.to_torch(env.data.pert_force).view(2, self.nw, 3)

    def _uniform(self, shape: tuple, r: tuple) -> torch.Tensor:
        return r[0] + (r[1] - r[0]) * torch.rand(shape, device=self.dev, generator=self.gen)

    def _sym(self, shape: tuple, a: float | tuple) -> torch.Tensor:
        a = torch.as_tensor(a, dtype=torch.float32, device=self.dev)
        return (2 * torch.rand(shape, device=self.dev, generator=self.gen) - 1) * a

    # -- geometry ------------------------------------------------------------------------------------------------------
    def ground(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Terrain height under world ``(x, z)`` ``[N, K]`` (each world on its tile)."""
        o = self._offset[: self.n]
        return hf_height(self.hf_h, self.hf_b, self.T.cell, self.T.origin, x + o[:, :1], z + o[:, 1:])

    def body_state(self, s: Any) -> dict[str, torch.Tensor]:
        q = s.body_quat[:, self.root]
        vel = s.body_com_vel()[:, self.root]
        w = s.body_vel[:, self.root, :3]
        qj, qdj = s.qpos[:, self.jidx], s.qvel[:, self.jidx]
        return dict(quat=q, lin=_irot(q, vel), ang=_irot(q, w),
                    grav=_irot(q, torch.tensor([0.0, -1.0, 0.0], device=self.dev).expand(self.n, 3)),
                    qrel=qj - self.nominal, qd=qdj)

    def system_state(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        """The predictor's input: clean and unscaled (their SystemStateCfg)."""
        return torch.cat([b["lin"], b["ang"], b["grav"], self.cmd, b["qrel"], b["qd"]], 1)

    def predict(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[N, 2, 2]`` predicted landing per foot (l, r): (forward, lateral right) in the root frame."""
        if self.fpm_ready and not self._bootstrapping():
            with torch.no_grad():
                return self.fpm(self.system_state(b)).view(self.n, 2, 2)
        return torch.tensor(self.c.bootstrap, device=self.dev).expand(self.n, 2, 2)

    def _bootstrapping(self) -> bool:
        return self.c.training and self.steps <= self.c.bootstrap_iters * self.c.steps_per_iter

    def scans(self, s: Any, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(map [N, 33*21], actor scan [N, 102], critic scan [N, 102]): their height_scan and foothold_scan."""
        c = self.c
        tp, tq = s.body_pos[:, self.torso], s.body_quat[:, self.torso]
        psi = _heading(tq)
        cs, sn = psi.cos()[:, None], psi.sin()[:, None]
        # heading frame: forward = (cos, -sin) in (x, z), right = (sin, cos)
        gx = tp[:, :1] + self.grid_f * cs + self.grid_l * sn
        gz = tp[:, 2:3] - self.grid_f * sn + self.grid_l * cs
        hmap = tp[:, 1:2] - self.ground(gx, gz) - 0.5                                    # [N, nx*ny]
        pred = self.predict(b)                                                          # [N, 2, 2]
        pb = torch.stack([pred[..., 0], torch.zeros_like(pred[..., 0]), pred[..., 1]], -1)
        pw = s.body_pos[:, self.root, None] + _rot(b["quat"][:, None].expand(-1, 2, -1), pb) - tp[:, None]
        f = pw[..., 0] * cs + pw[..., 2] * (-sn)                                         # into the torso heading frame
        lat = pw[..., 0] * sn + pw[..., 2] * cs
        f, lat = f + self.xy_drift[:, :1], lat + self.xy_drift[:, 1:]
        ax = torch.round((f + c.scan_size[0] / 2) / c.scan_res).long()
        ay = torch.round((lat + c.scan_size[1] / 2) / c.scan_res).long()
        sx = (ax[..., None] + self.off_x).clamp(0, self.nx - 1)
        sy = (ay[..., None] + self.off_y).clamp(0, self.ny - 1)
        patch = torch.gather(hmap[:, None].expand(-1, 2, -1), 2, sx * self.ny + sy)        # [N, 2, 49]
        critic = torch.cat([pred, patch], 2).flatten(1)
        actor = torch.cat([pred, patch + self.z_drift[:, None]], 2).flatten(1)
        actor = actor + self._sym(actor.shape, c.scan_noise)
        return hmap.clamp(-1, 1), actor.clamp(-1, 1), critic.clamp(-1, 1)

    # -- hooks -----------------------------------------------------------------------------------------------------
    def observation(self, s: Any) -> torch.Tensor:
        b = self.body_state(s)
        hmap, scan_a, scan_c = self.scans(s, b)
        noisy, clean = self.body_obs(b)
        return torch.cat([*noisy, scan_a, *clean, hmap, scan_c], 1).nan_to_num(0.0, 0.0, 0.0)

    def body_obs(self, b: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """The body blocks, with the actor's noise and without (the critic's). Noise is in physical units, added
        before the observation scales (0.2 on the turn rate, 0.05 on joint speeds), as IsaacLab and legged_gym do.
        Before 2026-11-18 it was added after them: 5x and 20x the intended noise (joint speeds +-30 rad/s)."""
        c = self.c
        sp = lambda x, p, amp: x + amp * (torch.rand(x.shape, device=self.dev, generator=self.gen) < p) * (2 * torch.rand(x.shape, device=self.dev, generator=self.gen) - 1)  # noqa: E731
        noisy = [sp(b["lin"] + self._sym(b["lin"].shape, c.n_lin), c.spike_p, c.spike_lin),
                 0.2 * sp(b["ang"] + self._sym(b["ang"].shape, c.n_ang), c.spike_p, c.spike_ang),
                 b["grav"] + self._sym(b["grav"].shape, c.n_grav), self.cmd,
                 b["qrel"] + self._sym(b["qrel"].shape, c.n_q), 0.05 * (b["qd"] + self._sym(b["qd"].shape, c.n_qd)),
                 self.prev_action.clamp(-10, 10)]
        clean = [b["lin"], 0.2 * b["ang"], b["grav"], self.cmd, b["qrel"], 0.05 * b["qd"], self.prev_action.clamp(-10, 10)]
        return noisy, clean

    def before_step(self, s: Any) -> None:
        c = self.c
        # command: redraw on the timer, then heading control and the standing / turning overrides
        due = self.cmd_timer <= 0
        if bool(due.any()):
            self._resample(due.nonzero().squeeze(1))
        self.cmd_timer -= c.dt
        psi = _heading(s.body_quat[:, self.root])
        self.cmd[:, 2] = (c.heading_stiffness * _wrap(self.heading_target - psi)).clamp(*c.ang_z)
        self.cmd[self.standing] = 0.0
        self.cmd[self.turning, :2] = 0.0
        if not c.training and c.eval_assist:
            self._pert[0, : self.n] = torch.tensor([0.0, c.eval_assist * self.mass * 9.80665, 0.0], device=self.dev)
        if c.training:
            # helper pull (world up) and the constant pelvis push (body frame, so turned with the pelvis)
            alpha = max(0.0, min(c.assist_offset, (1 - c.assist_offset) * (1 - float(self.level.float().mean()) / c.assist_cutoff)))
            self._pert[0, : self.n] = torch.tensor([0.0, alpha * self.mass * 9.80665, 0.0], device=self.dev)
            self._pert[1, : self.n] = _rot(s.body_quat[:, self.root], self.disturb)
            self.metrics["assist"] = torch.tensor(alpha)
            # velocity push every 4 s
            self.push_timer -= c.dt
            push = self.push_timer <= 0
            if bool(push.any()):
                i = push.nonzero().squeeze(1)
                k = len(i)
                lin = self._sym((k, 3), c.push_lin)
                ang = self._sym((k, 3), c.push_ang)
                p = self.pel
                s.qvel[i, p["tx"]] += lin[:, 0] * torch.cos(psi[i]) + lin[:, 1] * torch.sin(psi[i])
                s.qvel[i, p["tz"]] += -lin[:, 0] * torch.sin(psi[i]) + lin[:, 1] * torch.cos(psi[i])
                s.qvel[i, p["ty"]] += lin[:, 2]
                s.qvel[i, p["list"]] += ang[:, 0]
                s.qvel[i, p["tilt"]] += ang[:, 1]
                s.qvel[i, p["rotation"]] += ang[:, 2]
                self.push_timer[i] = c.push_s

    def _resample(self, i: torch.Tensor) -> None:
        c, k = self.c, len(i)
        self.cmd[i, 0] = self._uniform((k,), c.lin_x)
        self.cmd[i, 1] = self._uniform((k,), c.lin_y)
        self.cmd[i, 2] = self._uniform((k,), c.ang_z)
        self.heading_target[i] = self._uniform((k,), (-math.pi, math.pi))
        self.standing[i] = torch.rand(k, device=self.dev, generator=self.gen) <= c.standing_frac
        self.turning[i] = torch.rand(k, device=self.dev, generator=self.gen) <= c.turning_frac
        self.cmd_timer[i] = c.resample_s

    def joint_torques(self, s: Any) -> torch.Tensor:
        arm = self._arm.view(-1, self.nw)[self.t_slot][:, : self.n].T                   # [N, S]
        tau = torch.zeros(self.n, len(self.coords), device=self.dev)
        tau.index_add_(1, self.t_dof, s.muscle_force[:, self.t_mus] * arm)
        return tau[:, self.jidx]

    def reward(self, s: Any) -> torch.Tensor:
        b = self.body_state(s)
        terms = self.terms(s, b)
        r = sum(self.c.w[k] * v for k, v in terms.items()) * self.c.dt
        self.advance(s, b, terms)
        return r

    def terms(self, s: Any, b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Their reward terms, unweighted; sets the fall flag (``termination``)."""
        c = self.c
        act = s.excitation
        v_left = -b["lin"][:, 2]
        e_lin = (self.cmd[:, 0] - b["lin"][:, 0]) ** 2 + (self.cmd[:, 1] - v_left) ** 2
        e_ang = (self.cmd[:, 2] - b["ang"][:, 1]) ** 2
        standing = (self.cmd[:, :2].norm(dim=1) < c.stand_threshold) & (self.cmd[:, 2].abs() < c.stand_threshold)
        foot_y = s.body_pos[:, self.feet, 1].min(1).values
        deficit = (self.upright_target - (s.body_pos[:, self.root, 1] - foot_y)).clamp(min=0)
        qacc = (s.qvel[:, self.jidx] - self.prev_qvel[:, self.jidx]) / c.dt
        fc = s.contact_force()
        terms = dict(
            track_lin_vel_xy=torch.exp(-e_lin / c.track_std2),
            track_ang_vel_z=torch.exp(-e_ang / c.track_std2),
            stand_still=torch.exp(-b["qrel"].abs().sum(1) / c.stand_std) * standing,
            lin_vel_z=b["lin"][:, 1] ** 2,
            ang_vel_xy=b["ang"][:, 0] ** 2 + b["ang"][:, 2] ** 2,
            joint_torques=(self.joint_torques(s) ** 2).sum(1),
            joint_acc=(qacc ** 2).sum(1),
            action_rate=((act - self.prev_action).nan_to_num(c.action_rate_max).clamp(-c.action_rate_max, c.action_rate_max) ** 2).sum(1),
            undesired_contacts=(fc[:, self.legs].norm(dim=-1) > c.contact_threshold).float().sum(1),
            joint_deviation_legs=b["qrel"][:, self.dev_legs].abs().sum(1),
            joint_deviation_waist=b["qrel"][:, self.dev_waist].abs().sum(1),
            upright=torch.exp(-deficit ** 2 / c.upright_std ** 2),
        )
        self._fell = self._tilt(b) > c.fall_tilt
        terms["termination"] = self._fell.float()
        return terms

    def advance(self, s: Any, b: dict[str, torch.Tensor], terms: dict[str, torch.Tensor]) -> None:
        """After the reward: log the terms, keep this step's action and speeds, count the step, feed the predictor."""
        for k, v in terms.items():
            self.metrics[f"r_{k}"] = v
        self.metrics["level"] = self.level.float()
        self.metrics["fpm_val_loss"] = torch.tensor(self.fpm_val)
        self.prev_action = s.excitation.clone()
        self.prev_qvel = s.qvel.clone()
        self.steps += 1
        if self.c.training:
            self._fpm_step(s, b)

    def _tilt(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.acos((-b["grav"][:, 1]).clamp(-1, 1))

    def done(self, s: Any) -> tuple[torch.Tensor, torch.Tensor]:
        fell = self._fell | (s.diverged != 0)
        return fell, s.episode_step >= int(round(self.c.episode_s / self.c.dt))

    def reset(self, s: Any, worlds: torch.Tensor) -> None:
        c = self.c
        i = worlds.nonzero().squeeze(1)
        k = len(i)
        if k == 0:
            return
        # terrain curriculum from the episode that just ended (IsaacLab terrain_levels_vel), then the new tile
        if c.eval_tiles:
            t = torch.as_tensor(c.eval_tiles, device=self.dev)
            pick = (self._eval_next + torch.arange(k, device=self.dev)) % len(t)
            self._eval_next = int(self._eval_next + k)
            self.level[i], self.col[i] = t[pick, 0], t[pick, 1]
        elif self.steps > 0:
            dist = s.body_pos[i, self.root][:, [0, 2]].norm(dim=1).nan_to_num(0.0)
            up = dist > 4.0
            down = (dist < self.cmd[i, :2].norm(dim=1) * c.episode_s * 0.5) & ~up
            lv = self.level[i] + up.long() - down.long()
            hi = torch.randint(0, ROWS, (k,), device=self.dev, generator=self.gen)
            self.level[i] = torch.where(lv >= ROWS, hi, lv.clamp(min=0))
        self._offset[i] = self.centres[self.level[i], self.col[i]]
        self.place(s, i)
        self._resample(i)
        self.push_timer[i] = c.push_s
        self.xy_drift[i] = self._sym((k, 2), c.xy_drift)
        self.z_drift[i] = self._sym((k, 1), c.z_drift)
        self.disturb[i] = self._sym((k, 3), c.disturbance_n) if c.training else 0.0

    def ground_max(self, i: torch.Tensor, dx: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        """The highest ground in a 0.6 m square around ``(dx, dz)`` on worlds ``i``'s tiles (feet's reach)."""
        o = torch.linspace(-0.3, 0.3, 7, device=self.dev)
        px = (dx[:, None] + o.repeat_interleave(7)[None])
        pz = (dz[:, None] + o.repeat(7)[None])
        off = self._offset[i]
        return hf_height(self.hf_h, self.hf_b, self.T.cell, self.T.origin, px + off[:, :1], pz + off[:, 1:]).max(1).values

    def place(self, s: Any, i: torch.Tensor) -> None:
        """The reset pose on worlds ``i``: nominal joints, root on the tile +-0.5 m, any yaw, random root velocity."""
        c, k = self.c, len(i)
        q = self.q0.expand(k, -1).clone()
        self.nominal[i] = self.q0[self.jidx] + self._sym((k, self.nj), c.nominal_jitter)
        q[:, self.jidx] = self.nominal[i]
        dx, dz = self._sym((k,), c.reset_pos), self._sym((k,), c.reset_pos)
        yaw = self._sym((k,), math.pi)
        p = self.pel
        q[:, p["tx"]], q[:, p["tz"]], q[:, p["rotation"]] = dx, dz, yaw
        # stand on the highest ground under the feet's reach, so no sphere starts inside it
        q[:, p["ty"]] = self.pelvis_ty0 + self.ground_max(i, dx, dz) + 0.02
        qd = torch.zeros_like(q)
        lin, ang = self._sym((k, 3), c.reset_lin), self._sym((k, 3), c.reset_ang)
        qd[:, p["tx"]], qd[:, p["tz"]], qd[:, p["ty"]] = lin[:, 0], lin[:, 1], lin[:, 2]
        qd[:, p["list"]], qd[:, p["tilt"]], qd[:, p["rotation"]] = ang[:, 0], ang[:, 1], ang[:, 2]
        s.qpos[i], s.qvel[i] = q, qd
        self.prev_qvel[i] = qd
        self.prev_action[i] = 0.0

    # -- the predictor's refits ---------------------------------------------------------------------------------------
    def _fpm_step(self, s: Any, b: dict[str, torch.Tensor]) -> None:
        c = self.c
        boot = c.bootstrap_iters * c.steps_per_iter
        if self.steps <= boot:
            return
        phase = (self.steps - boot) % (c.refit_iters * c.steps_per_iter)
        if phase < self.record_steps:
            if self.buf is None:
                self.buf = {"state": [], "contact": [], "root_pos": [], "root_quat": [], "feet": []}
            fc = s.contact_force()
            down = torch.stack([(fc[:, self.feet[j]] + fc[:, self.toes[j]]).norm(dim=-1) > c.contact_threshold
                                for j in range(2)], 1)
            for k_, v in (("state", self.system_state(b)), ("contact", down), ("root_pos", s.body_pos[:, self.root]),
                          ("root_quat", b["quat"]), ("feet", s.body_pos[:, self.feet])):
                self.buf[k_].append(v.detach().clone())
        elif phase == self.record_steps and self.buf is not None:
            self._refit()

    def _refit(self) -> None:
        H = {k: torch.stack(v, 1) for k, v in self.buf.items()}                         # [N, T, ...]
        self.buf = None
        x = H["state"][:, :-1].reshape(-1, self.state_dim)
        y = FPM.foothold_labels(H["contact"][:, 1:], H["root_pos"][:, 1:], H["root_quat"][:, 1:], H["feet"][:, 1:])
        it = self.steps // self.c.steps_per_iter
        losses: list[str] = []
        model = FPM.fit(x, y.reshape(x.shape[0], -1), self.dev, seed=it, log=losses.append)
        self.fpm.load_state_dict(model.state_dict())
        self.fpm_ready = True
        self.fpm_val = float(losses[-1].split("val")[-1]) if losses else float("nan")
        print(f"[FPM] iteration {it}: refit on {tuple(x.shape)}; {losses[-1] if losses else ''}", flush=True)
        if self.log_dir is not None:
            torch.save(self.fpm.state_dict(), self.log_dir / f"foot_model_{it}.pt")

    def carried_state(self, env: Any) -> dict[str, Any]:
        return {}
