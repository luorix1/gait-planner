"""The planner (plan doc, step 3): a diffusion model of plan windows, conditioned on the ground ahead and a goal.

Like PARC's generator (Xu et al. 2025: motion diffusion conditioned on a local heightmap and a target), but it
generates the plan, not the pose. A plan frame (20 Hz) is each foot's down flag (+-1) and the xz of its current or next
foothold, the centre of mass, and the pelvis orientation as its heading and its tilt from that heading (a rotation
vector; ``CHANNELS``). A window is ``HIST`` frames of the plan so far (given,
clean) and ``FUT`` frames to generate, all in the heading frame of the last given frame (the anchor): positions
relative to its centre of mass, heading relative to its heading. Conditions:
- **ground**: terrain height on a forward x lateral grid (``PATCH_F`` x ``PATCH_L``) in the anchor frame, relative to
  the anchor's centre-of-mass height, read from the clip's own terrain;
- **goal**: the centre of mass's xz ``FUT`` frames on, in the anchor frame (direction and speed).

Network: the style prior's DiT blocks (gp.smp._Block) over [ground tokens (two stride-2 convolutions), plan tokens],
the diffusion step and goal through adaLN; x0 prediction, MSE on the generated frames, the prior's 50-step
squared-cosine schedule, deterministic DDIM sampling over every step, EMA weights. Training mirrors half of each batch
left to right (:func:`mirror`).

:func:`to_plan` decodes a 20 Hz plan to the tasks' 40 Hz plan (gp.data.plan_targets: a contact's foothold is the mean
of its frames' generated footholds, its height the ground's), and :func:`plan_course` rolls the planner out along
generated courses from a standing start: generate ``FUT`` frames, keep ``KEEP``, re-anchor, repeat.

The history is the plan's own past (at deployment too: the previous plan, not the measured body).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sprint.recipes.imitation import _qconj, _qmul, _rotvec

from . import data as gpdata
from .smp import _alphas_cumprod, _Block, _timestep_embedding
from .terrain import Ground, atlas, hf_height

STRIDE = 2                                   # data frames (40 Hz) per plan frame
HZ = gpdata.HZ / STRIDE
HIST, FUT, KEEP = 8, 24, 10                  # 0.4 s given, 1.2 s generated, 0.5 s kept per rollout step
PATCH_F = tuple(np.round(np.arange(-0.6, 3.01, 0.1), 2))
PATCH_L = tuple(np.round(np.arange(-0.8, 0.81, 0.1), 2))
CHANNELS = ("down_l", "down_r", "hold_l_f", "hold_l_lat", "hold_r_f", "hold_r_lat", "com_f", "com_y", "com_lat",
            "cos_dpsi", "sin_dpsi", "tilt_x", "tilt_y", "tilt_z")
C = len(CHANNELS)
GOAL = [6, 8]                                # the goal is the last generated frame's com_f, com_lat
MIRROR_PERM = [1, 0, 4, 5, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
MIRROR_SIGN = [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0]   # tilt: a pseudovector
T_DIFF = 50
MIN_STD = 0.05
#: Full-pose windows (``Windows(pose=True)``) append the pelvis origin (anchor frame, relative to the anchor's centre
#: of mass) and every non-pelvis coordinate to the plan channels; the plan channels stay first, so a full-pose
#: window's ``[..., :C]`` is a plan window.
POSE_COORDS = ("hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "hip_flexion_r", "hip_adduction_r",
               "hip_rotation_r", "lumbar_extension", "lumbar_bending", "lumbar_rotation", "knee_angle_l", "knee_angle_r",
               "ankle_angle_l", "ankle_angle_r", "subtalar_angle_l", "subtalar_angle_r", "mtp_angle_l", "mtp_angle_r")
POSE_CHANNELS = (*CHANNELS, "pelvis_f", "pelvis_y", "pelvis_lat", *POSE_COORDS)


# --- plan frames ---------------------------------------------------------------------------------------------------

def heading(quat: torch.Tensor) -> torch.Tensor:
    """Heading (rotation about +y) of a body's x axis, from quaternions ``[..., 4]`` (x, y, z, w)."""
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(-2 * (x * z - w * y), 1 - 2 * (y * y + z * z))


def yaw_quat(psi: torch.Tensor) -> torch.Tensor:
    """Quaternions (x, y, z, w) of heading ``psi`` (rotation about +y)."""
    z = torch.zeros_like(psi)
    return torch.stack([z, torch.sin(psi / 2), z, torch.cos(psi / 2)], -1)


def qexp(v: torch.Tensor) -> torch.Tensor:
    """Quaternions of rotation vectors ``[..., 3]`` (the inverse of SPRINT's ``_rotvec``)."""
    a = v.norm(dim=-1, keepdim=True)
    k = torch.where(a > 1e-6, torch.sin(a / 2) / a.clamp_min(1e-6), 0.5 - a * a / 48)
    return torch.cat([v * k, torch.cos(a / 2)], -1)


def tilt(quat: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """A body's orientation ``quat`` less its heading ``psi``, as a rotation vector: ``quat = yaw(psi) * exp(tilt)``."""
    return _rotvec(_qmul(_qconj(yaw_quat(psi)), quat))


def to_frame(xz: torch.Tensor, origin: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """World (x, z) ``[B, ..., 2]`` -> (forward, lateral) in the heading frame at ``origin`` ``[B, 2]``, ``psi`` ``[B]``
    (forward = (cos psi, -sin psi); lateral = the body's right, +z at psi = 0)."""
    v = (xz.shape[0], *([1] * (xz.dim() - 2)))
    c, s = psi.cos().view(v), psi.sin().view(v)
    d = xz - origin.view(*v, 2)
    return torch.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], -1)


def from_frame(fl: torch.Tensor, origin: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_frame`."""
    v = (fl.shape[0], *([1] * (fl.dim() - 2)))
    c, s = psi.cos().view(v), psi.sin().view(v)
    return origin.view(*v, 2) + torch.stack([fl[..., 0] * c + fl[..., 1] * s, -fl[..., 0] * s + fl[..., 1] * c], -1)


def features(down: torch.Tensor, hold: torch.Tensor, com: torch.Tensor, psi: torch.Tensor, quat: torch.Tensor,
             a_com: torch.Tensor, a_psi: torch.Tensor, pelvis: torch.Tensor | None = None,
             joints: torch.Tensor | None = None) -> torch.Tensor:
    """World plan frames (down ``[B, K, 2]``, foothold xz ``[B, K, 2, 2]``, com ``[B, K, 3]``, heading ``[B, K]``,
    pelvis orientation ``[B, K, 4]``) -> ``[B, K, C]`` in the frame of anchor (``a_com`` ``[B, 3]``, ``a_psi`` ``[B]``).
    With the pelvis origin ``[B, K, 3]`` and joint coordinates ``[B, K, len(POSE_COORDS)]``: the full-pose channels."""
    o = a_com[:, [0, 2]]
    cxz = to_frame(com[..., [0, 2]], o, a_psi)
    dpsi = psi - a_psi[:, None]
    x = torch.cat([down.float() * 2 - 1, to_frame(hold, o, a_psi).flatten(2), cxz[..., :1],
                   (com[..., 1] - a_com[:, 1:2])[..., None], cxz[..., 1:], dpsi.cos()[..., None],
                   dpsi.sin()[..., None], tilt(quat, psi)], -1)
    if pelvis is None:
        return x
    pxz = to_frame(pelvis[..., [0, 2]], o, a_psi)
    return torch.cat([x, pxz[..., :1], (pelvis[..., 1] - a_com[:, 1:2])[..., None], pxz[..., 1:], joints], -1)


def world(x: torch.Tensor, a_com: torch.Tensor, a_psi: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Inverse of :func:`features`: ``[B, K, C]`` -> (down, foothold xz, com, heading, pelvis orientation), the down
    flag thresholded at 0."""
    o = a_com[:, [0, 2]]
    cxz = from_frame(x[..., GOAL], o, a_psi)
    com = torch.stack([cxz[..., 0], x[..., 7] + a_com[:, 1:2], cxz[..., 1]], -1)
    psi = a_psi[:, None] + torch.atan2(x[..., 10], x[..., 9])
    return (x[..., :2] > 0, from_frame(x[..., 2:6].unflatten(-1, (2, 2)), o, a_psi), com, psi,
            _qmul(yaw_quat(psi), qexp(x[..., 11:14])))


def patch(G: Ground, clip: torch.Tensor, a_com: torch.Tensor, a_psi: torch.Tensor) -> torch.Tensor:
    """Ground heights ``[B, len(PATCH_F), len(PATCH_L)]`` around each anchor, relative to its centre-of-mass height."""
    g = torch.tensor([(f, lat) for f in PATCH_F for lat in PATCH_L], dtype=torch.float32, device=a_com.device)
    xz = from_frame(g.expand(len(clip), -1, -1), a_com[:, [0, 2]], a_psi)
    return (G(clip, xz[..., 0], xz[..., 1]) - a_com[:, 1:2]).view(len(clip), len(PATCH_F), len(PATCH_L))


def mirror(x: torch.Tensor, p: torch.Tensor, perm: Any = MIRROR_PERM, sign: Any = MIRROR_SIGN
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """The same windows reflected left to right: feet swapped, lateral coordinates and heading changes negated, the
    patch's lateral axis reversed (``PATCH_L`` is symmetric). Full-pose windows pass :func:`pose_mirror`'s."""
    return x[..., perm] * x.new_tensor(sign), p.flip(-1)


def pose_mirror(coord_names: list[str]) -> tuple[list[int], list[float]]:
    """``mirror``'s permutation and signs for full-pose windows: the plan's, the pelvis's lateral offset negated, and
    each joint coordinate from its partner with gp.data.coord_mirror_signs' sign (forward kinematics, not the axis
    rule, which gets Rajagopal's hips and knee wrong)."""
    from sprint.recipes.imitation import mirrored
    signs = gpdata.coord_mirror_signs(gpdata.load_model())
    names = list(coord_names)
    perm = [*MIRROR_PERM, C, C + 1, C + 2] + [C + 3 + POSE_COORDS.index(mirrored(n)) for n in POSE_COORDS]
    sign = [*MIRROR_SIGN, 1.0, 1.0, -1.0] + [float(signs[names.index(n)]) for n in POSE_COORDS]
    return perm, sign


class Windows:
    """Plan windows of a loaded dataset (gp.data.load), on ``device``. ``anchors``: every frame with a full window in
    its clip; ``self(a)`` -> (windows ``[B, HIST + FUT, C]``, patches, goals ``[B, 2]``) at anchor frames ``a``.
    ``pose``: full-pose windows (``POSE_CHANNELS``). ``balance``: batches draw anchors by gp.data.clip_weights(D,
    "sqrt") (each source by the square root of its length, every recording equal within it), not uniformly."""

    def __init__(self, D: dict[str, Any], device: Any = "cpu", pose: bool = False, balance: bool = False) -> None:
        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
        self.down = T(D["plan_contact"], torch.bool)
        self.hold = T(D["foothold"])[..., [0, 2]]
        self.com = T(D["com"])
        q = T(D["record"])[:, gpdata.record_layout(len(D["tracked"]))["root_quat"]]
        self.quat = q / q.norm(dim=-1, keepdim=True)
        self.psi = heading(self.quat)
        self.clip = T(np.repeat(np.arange(len(D["frames"])), D["frames"]), torch.long)
        self.G = Ground.of(D, device)
        self.anchors = T(np.concatenate([np.arange(s + STRIDE * (HIST - 1), s + n - STRIDE * FUT)
                                         for s, n in zip(D["start"], D["frames"], strict=True)]), torch.long)
        self.k = STRIDE * torch.arange(-HIST + 1, FUT + 1, device=device)
        self.pose = pose
        self.perm, self.sign = (MIRROR_PERM, MIRROR_SIGN)
        if pose:
            rec = T(D["record"])
            self.pelvis = rec[:, gpdata.record_layout(len(D["tracked"]))["root_pos"]]
            self.joints = T(D["qpos"])[:, [list(D["coord_names"]).index(n) for n in POSE_COORDS]]
            self.perm, self.sign = pose_mirror(D["coord_names"])
        self.c = len(self.sign)
        self.w = None
        if balance:                              # clip weight spread over the clip's anchors
            n_a = np.maximum(np.asarray(D["frames"]) - STRIDE * (HIST - 1 + FUT), 0)
            cw = gpdata.clip_weights(D, "sqrt")
            self.w = T(np.repeat(np.divide(cw, n_a, out=np.zeros_like(cw), where=n_a > 0), n_a))

    def __call__(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = a[:, None] + self.k
        extra = (self.pelvis[k], self.joints[k]) if self.pose else ()
        x = features(self.down[k], self.hold[k], self.com[k], self.psi[k], self.quat[k], self.com[a], self.psi[a], *extra)
        return x, patch(self.G, self.clip[a], self.com[a], self.psi[a]), x[:, -1, GOAL]

    def batch(self, n: int, gen: torch.Generator | None = None, mirrored: bool = True
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``n`` random windows; with ``mirrored`` a random half reflected left to right."""
        dev = self.anchors.device
        i = (torch.randint(len(self.anchors), (n,), device=dev, generator=gen) if self.w is None
             else torch.multinomial(self.w, n, replacement=True, generator=gen))
        x, p, _ = self(self.anchors[i])
        if mirrored:
            flip = torch.rand(n, device=dev, generator=gen) < 0.5
            x[flip], p[flip] = mirror(x[flip], p[flip], self.perm, self.sign)
        return x, p, x[:, -1, GOAL]


# --- the network ---------------------------------------------------------------------------------------------------

class PlanNet(nn.Module):
    """x0-predicting DiT over [ground tokens, plan tokens]; the diffusion step and the goal condition through adaLN."""

    def __init__(self, layers: int = 4, heads: int = 4, head_dim: int = 64, c: int = C) -> None:
        super().__init__()
        D = heads * head_dim
        self.D = D
        self.inp = nn.Linear(c + 1, D)                                # + the "given" flag
        self.pe = nn.Parameter(torch.randn(1, HIST + FUT, D) * 0.02)
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.SiLU(), nn.Conv2d(32, D, 3, 2, 1))
        n_tok = self.conv(torch.zeros(1, 1, len(PATCH_F), len(PATCH_L))).flatten(2).shape[2]
        self.gpe = nn.Parameter(torch.randn(1, n_tok, D) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        self.goal = nn.Sequential(nn.Linear(2, D), nn.SiLU(), nn.Linear(D, D))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        self.blocks = nn.ModuleList(_Block(D, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, c)
        given = torch.zeros(1, HIST + FUT, 1)
        given[:, :HIST] = 1.0
        self.register_buffer("given", given)

    def forward(self, x: torch.Tensor, t: torch.Tensor, p: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        h = self.inp(torch.cat([x, self.given.expand(len(x), -1, -1)], -1)) + self.pe
        g = self.conv(p[:, None]).flatten(2).transpose(1, 2) + self.gpe
        t6 = self.ada(self.t_mlp(_timestep_embedding(t)) + self.goal(goal)).view(-1, 6, self.D)
        z = torch.cat([g, h], 1)
        for b in self.blocks:
            z = b(z, t6)
        return self.out(self.norm(z[:, g.shape[1]:]))


class Planner(nn.Module):
    """The denoiser, its EMA copy (what samples), the schedule and the normalisers. Public methods take and return
    unnormalised windows, patches and goals."""

    def __init__(self, T: int = T_DIFF, layers: int = 4, heads: int = 4, head_dim: int = 64, c: int = C) -> None:
        super().__init__()
        self.cfg = dict(T=T, layers=layers, heads=heads, head_dim=head_dim, c=c)
        self.c = c
        self.net = PlanNet(layers, heads, head_dim, c)
        self.ema = torch.optim.swa_utils.AveragedModel(self.net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995))
        self.register_buffer("ab", _alphas_cumprod(T))
        for k, n in (("x", c), ("p", 1), ("g", 2)):
            self.register_buffer(f"{k}_mean", torch.zeros(n))
            self.register_buffer(f"{k}_std", torch.ones(n))

    def set_normaliser(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor) -> None:
        for k, v in (("x", x.reshape(-1, self.c)), ("p", p.reshape(-1, 1)), ("g", g)):
            getattr(self, f"{k}_mean").copy_(v.mean(0))
            getattr(self, f"{k}_std").copy_(v.std(0).clamp_min(MIN_STD))

    def _n(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (x - self.x_mean) / self.x_std, (p - self.p_mean) / self.p_std, (g - self.g_mean) / self.g_std

    def loss(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """MSE of the predicted clean future at a uniform diffusion step, the history given clean."""
        x, p, g = self._n(x, p, g)
        t = torch.randint(0, self.cfg["T"], (len(x),), device=x.device)
        a = self.ab[t][:, None, None]
        xt = torch.cat([x[:, :HIST], a.sqrt() * x[:, HIST:] + (1 - a).sqrt() * torch.randn_like(x[:, HIST:])], 1)
        return F.mse_loss(self.net(xt, t, p, g)[:, HIST:], x[:, HIST:])

    @torch.no_grad()
    def generate(self, hist: torch.Tensor, p: torch.Tensor, g: torch.Tensor, gen: torch.Generator | None = None,
                 ema: bool = True, stride: int = 1) -> torch.Tensor:
        """The future ``[B, FUT, C]`` after ``hist`` ``[B, HIST, C]``: deterministic DDIM over every ``stride``-th
        diffusion step (1 = all)."""
        pad = hist.new_zeros(len(hist), FUT, self.c)
        h, p, g = self._n(torch.cat([hist, pad], 1), p, g)
        h = h[:, :HIST]
        net = self.ema if ema else self.net
        x = torch.randn(len(h), FUT, self.c, device=h.device, generator=gen)
        ts = list(range(self.cfg["T"] - 1, -1, -stride))
        for t, t_prev in zip(ts, [*ts[1:], -1], strict=True):
            x0 = net(torch.cat([h, x], 1), torch.full((len(h),), t, device=h.device), p, g)[:, HIST:]
            if t_prev < 0:
                x = x0
                break
            eps = (x - self.ab[t].sqrt() * x0) / (1 - self.ab[t]).sqrt()
            x = self.ab[t_prev].sqrt() * x0 + (1 - self.ab[t_prev]).sqrt() * eps
        return x * self.x_std + self.x_mean

    def save(self, path: str | Path, **extra: Any) -> None:
        torch.save(dict(cfg=self.cfg, state=self.state_dict(), **extra), path)

    @classmethod
    def load(cls, path: str | Path, device: Any = "cpu") -> Planner:
        ck = torch.load(path, map_location=device, weights_only=False)
        p = cls(**ck["cfg"]).to(device)
        p.load_state_dict(ck["state"])
        return p.eval().requires_grad_(False)


# --- plans -----------------------------------------------------------------------------------------------------

def to_plan(down: np.ndarray, hold: np.ndarray, com: np.ndarray, psi: np.ndarray, quat: np.ndarray,
            ground: Any) -> dict[str, Any]:
    """A 20 Hz world plan (down ``[K, 2]``, foothold xz ``[K, 2, 2]``, com ``[K, 3]``, heading ``[K]``, pelvis
    orientation ``[K, 4]``) -> the tasks' 40 Hz plan: gp.free.stand_in_plan's keys and ``pelvis_quat`` ``[T, 4]``
    (slerped). ``ground(x, z)``: terrain height (numpy)."""
    from scipy.spatial.transform import Rotation, Slerp
    K = len(down)
    n = STRIDE * (K - 1) + 1
    i, t = np.arange(n) // STRIDE, np.arange(n) / STRIDE
    c, hz = down[i], hold[i]
    feet = np.stack([hz[..., 0], ground(hz[..., 0], hz[..., 1]), hz[..., 1]], -1)
    hold3, td, lo = zip(*(gpdata.plan_targets(c[:, k], feet[:, k]) for k in (0, 1)), strict=True)
    com40 = np.stack([np.interp(t, np.arange(K), com[:, j]) for j in range(3)], 1)
    s_com = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(com40[:, [0, 2]], axis=0), axis=1))]
    return dict(plan_contact=c, foothold=np.stack(hold3, 1), t_touchdown=np.stack(td, 1), t_liftoff=np.stack(lo, 1),
                com=com40, s_com=s_com, heading=np.interp(t, np.arange(K), np.unwrap(psi)),
                pelvis_quat=Slerp(np.arange(K), Rotation.from_quat(quat))(t).as_quat())


def plan_course(planner: Planner, tiles: list[dict[str, Any]], D: dict[str, Any], stand: int, *, speed: float = 1.2,
                s0: float = 1.0, end_pad: float = 1.0, ramp_s: float = 1.5, seed: int = 0, device: Any = "cpu"
                ) -> list[dict[str, Any]]:
    """Roll the planner out along course tiles (gp.terrain_gen.course_tile), one plan per tile (:func:`to_plan`).

    Start: clip ``stand`` of ``D`` (level standing)'s first window's history, placed at arc length ``s0`` heading
    along the course. Each step the goal is the centreline ``v * FUT / HZ`` m past the nearest point (``v`` ramps to
    ``speed`` over ``ramp_s``), clamped ``end_pad`` before the end, so the plan stops there; the rollout runs until
    every course has had time to finish plus 3 s."""
    B = len(tiles)
    A = atlas(tiles, tiles[0]["cell"])
    G = Ground(np.tile([-1.0, 1.0], (B, 1)), np.zeros((B, 2)), np.ones(B, bool), A, A["offsets"], device)
    W = Windows(gpdata.take(D, [stand]), device)
    a = W.anchors[:1]
    hist = W(a)[0][:, :HIST].expand(B, -1, -1)
    above = float(W.com[a, 1] - W.G(W.clip[a], W.com[a, :1], W.com[a, 2:3])[0, 0])   # standing com over its ground
    paths = [T["path"] for T in tiles]
    s_a = np.full(B, s0)
    s_end = np.array([p["s"][-1] - end_pad for p in paths])
    a_xz = np.array([[np.interp(s0, p["s"], p["x"]), np.interp(s0, p["s"], p["z"])] for p in paths])
    a_psi = torch.tensor([np.interp(s0, p["s"], np.unwrap(p["heading"])) for p in paths], dtype=torch.float32, device=device)
    clip = torch.arange(B, device=device)
    xz_t = torch.as_tensor(a_xz, dtype=torch.float32, device=device)
    a_com = torch.cat([xz_t[:, :1], G(clip, xz_t[:, :1], xz_t[:, 1:]) + above, xz_t[:, 1:]], 1)
    seq = [world(hist, a_com, a_psi)]
    gen = torch.Generator(device=device).manual_seed(seed)
    steps = int(np.ceil(((s_end - s0).max() / speed + ramp_s + 3.0) * HZ / KEEP))
    for k in range(steps):
        v = speed * min(1.0, k * KEEP / HZ / ramp_s + 0.25)
        goal_w = np.zeros((B, 2))
        ac = a_com.cpu().numpy()
        for b, p in enumerate(paths):           # the nearest centreline point, never behind the last one by much
            win = np.flatnonzero((p["s"] >= s_a[b] - 0.5) & (p["s"] <= s_a[b] + 2.0))
            s_a[b] = p["s"][win[np.argmin((p["x"][win] - ac[b, 0]) ** 2 + (p["z"][win] - ac[b, 2]) ** 2)]]
            sg = min(s_a[b] + v * FUT / HZ, s_end[b])
            goal_w[b] = np.interp(sg, p["s"], p["x"]), np.interp(sg, p["s"], p["z"])
        goal = to_frame(torch.as_tensor(goal_w, dtype=torch.float32, device=device), a_com[:, [0, 2]], a_psi)
        fut = planner.generate(hist, patch(G, clip, a_com, a_psi), goal, gen)[:, :KEEP]
        seq.append(world(fut, a_com, a_psi))
        a_com, a_psi = seq[-1][2][:, -1], seq[-1][3][:, -1]
        recent = [torch.cat([s[j] for s in seq], 1)[:, -HIST:] for j in range(5)]
        hist = features(*recent, a_com, a_psi)
    full = [torch.cat([s[j] for s in seq], 1).cpu().numpy() for j in range(5)]
    return [to_plan(*(f[b] for f in full),
                    ground=lambda x, z, T=T: hf_height(T["heights"], T["block"], T["cell"], T["origin"], x, z))
            for b, T in enumerate(tiles)]


# --- in the loop ---------------------------------------------------------------------------------------------------

LIFT = 0.009                # the foot point's height over the ground in contact (AB01 median), for generated footholds
GOAL_FRAMES = STRIDE * FUT  # the goal: where the centre of mass is this many 40 Hz frames on


def plan_obs(task: Any, s: Any, contact: torch.Tensor, hold: torch.Tensor, td: torch.Tensor, lo: torch.Tensor,
             com: torch.Tensor) -> torch.Tensor:
    """The student's plan inputs ``[N, 15 x taps]`` from a plan at its taps (contact ``[N, K, 2]``, foothold
    ``[N, K, 2, 3]``, td, lo ``[N, K, 2]``, com ``[N, K, 3]``, world frame): per tap and foot down, foothold from the
    pelvis in its heading frame, time to touchdown and to lift-off; the centre of mass from the body's."""
    from sprint.recipes.imitation import _heading
    M = _heading(s.body_quat[:, task.root])
    fh = torch.einsum("nij,nkfj->nkfi", M, hold - s.body_com_pos()[:, task.root][:, None, None])
    dc = torch.einsum("nij,nkj->nki", M, com - task.com(s)[:, None])
    feet = torch.cat([contact.float()[..., None], fh, td[..., None], lo[..., None]], -1).flatten(2)   # [N, K, 12]
    return torch.nan_to_num(torch.cat([feet, dc], -1).flatten(1), 0.0, 0.0, 0.0)


def _next(flag: torch.Tensor) -> torch.Tensor:
    """Index of the first frame at or after each frame where ``flag`` [N, K, 2] holds; K where none does."""
    K = flag.shape[1]
    i = torch.arange(K, device=flag.device)[None, :, None].expand_as(flag)
    pos = torch.where(flag, i, K)
    return pos.flip(1).cummin(1).values.flip(1)


class Replanner:
    """The planner in the loop of a recipe environment (gp.tasks.TerrainImitation): each world follows a 40 Hz plan
    fixed in the world, ``STRIDE * FUT + 1`` frames from its last generation, and replans every ``every`` control
    steps and at each touchdown (``min_gap`` steps apart at least).

    A generation sees the body's measured last 0.4 s (down = vertical foot force over 5 % of body weight; a down
    foot's foothold = its point, a swinging foot's = the plan's; centre of mass; pelvis), the ground around it (the
    task's) and a goal: the clip's own centre-of-mass travel over the next 1.2 s in the clip's heading frame (an
    intent, ``go this far``, never a correction toward the clip). A new episode's history is its clip's past (the
    initial condition the recipe starts the body in). ``self(s)``, once per control step: the student's plan inputs
    (:func:`plan_obs`), as gp distill's ``--plan`` reads the clip's."""

    def __init__(self, planner: Planner, task: Any, D: dict[str, Any], every: int = 8, min_gap: int = 2,
                 stride: int = 5, seed: int = 0) -> None:
        self.P, self.task, self.every, self.min_gap, self.stride = planner, task, every, min_gap, stride
        dev = self.dev = task.dev
        self.W = Windows(D, dev)                         # the clip tables (history at reset, goals)
        n, L, R = task.n, STRIDE * FUT + 1, STRIDE * (HIST - 1) + 1
        z = lambda *k, dt=torch.float32: torch.zeros(n, *k, device=dev, dtype=dt)  # noqa: E731
        self.contact, self.hold, self.td, self.lo, self.com = z(L, 2, dt=torch.bool), z(L, 2, 3), z(L, 2), z(L, 2), z(L, 3)
        self.p = torch.zeros(n, dtype=torch.long, device=dev)          # "now" in the plan
        self.since = torch.zeros(n, dtype=torch.long, device=dev)      # steps since the last generation
        self.h_down, self.h_hold, self.h_com = z(R, 2, dt=torch.bool), z(R, 2, 2), z(R, 3)   # measured, 40 Hz
        self.h_psi, self.h_quat = z(R), z(R, 4)
        self.taps = torch.as_tensor(task.taps, device=dev)
        self.g = torch.tensor([(f, lat) for f in PATCH_F for lat in PATCH_L], dtype=torch.float32, device=dev)
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.replans = 0

    def _clip_history(self, w: torch.Tensor) -> None:
        """Worlds ``w`` (indices): their clip's frames before the episode's first (held at the clip start), so the
        measured first frame completes a 0.4 s history."""
        t, W = self.task, self.W
        R = self.h_down.shape[1]
        k = (t.frame[w, None] + torch.arange(-R, 0, device=self.dev)).clamp(min=t.clip_start[t.clip[w]][:, None])
        self.h_down[w], self.h_hold[w], self.h_com[w] = W.down[k], W.hold[k], W.com[k]
        self.h_psi[w], self.h_quat[w] = W.psi[k], W.quat[k]
        h = self.hold[w]                                                # the swinging foot's foothold until the first plan
        h[:, 0, :, 0], h[:, 0, :, 2] = W.hold[t.frame[w]][..., 0], W.hold[t.frame[w]][..., 1]
        self.hold[w] = h

    def _measure(self, s: Any) -> None:
        t, n = self.task, self.task.n
        bw = 9.81 * float(torch.as_tensor(s.total_mass).flatten()[0])
        down = t.foot_force(s) > 0.05 * bw
        i = torch.arange(n, device=self.dev)
        hold = torch.where(down[..., None], t.foot_points(s)[..., [0, 2]], self.hold[i, self.p][..., [0, 2]])
        q = s.body_quat[:, t.root]
        for h, v in ((self.h_down, down), (self.h_hold, hold), (self.h_com, t.com(s)), (self.h_psi, heading(q)),
                     (self.h_quat, q)):
            h.copy_(torch.cat([h[:, 1:], v[:, None]], 1))

    def _goal(self, w: torch.Tensor) -> torch.Tensor:
        """Worlds ``w``'s goal: the clip's centre-of-mass travel over the next 1.2 s in the clip's heading frame."""
        t, W = self.task, self.W
        f = t.frame[w]
        k = torch.minimum(f + GOAL_FRAMES, t.clip_last[t.clip[w]] + t.taps[-1])
        return to_frame(W.com[k][:, [0, 2]], W.com[f][:, [0, 2]], W.psi[f])

    @torch.no_grad()
    def _replan(self, w: torch.Tensor) -> None:
        t, dev = self.task, self.dev
        sl = slice(None, None, STRIDE)
        a_com, a_psi = self.h_com[w, -1], self.h_psi[w, -1]
        hist = features(self.h_down[w, sl], self.h_hold[w, sl], self.h_com[w, sl], self.h_psi[w, sl],
                        self.h_quat[w, sl], a_com, a_psi)
        xz = from_frame(self.g.expand(len(w), -1, -1), a_com[:, [0, 2]], a_psi)
        p = (t.G(t.clip[w], xz[..., 0], xz[..., 1]) - a_com[:, 1:2]).view(len(w), len(PATCH_F), len(PATCH_L))
        fut = self.P.generate(hist, p, self._goal(w), self.gen, stride=self.stride)
        down, hxz, com, _, _ = world(fut, a_com, a_psi)                                # [B, FUT, ...] at 20 Hz
        down = torch.cat([self.h_down[w, -1:], down], 1)                                # now, then the future
        hxz = torch.cat([self.h_hold[w, -1:], hxz], 1)
        com = torch.cat([a_com[:, None], com], 1)
        L = self.contact.shape[1]
        j = torch.arange(L, device=dev) // STRIDE
        frac = (torch.arange(L, device=dev) % STRIDE).float() / STRIDE
        c40, h40 = down[:, j], hxz[:, j]
        com40 = com[:, j] + frac[None, :, None] * (com[:, (j + 1).clamp(max=FUT)] - com[:, j])
        y = t.G(t.clip[w], h40[..., 0].flatten(1), h40[..., 1].flatten(1)).view(len(w), L, 2) + LIFT
        cap, i = gpdata.PLAN_CAP_S, torch.arange(L, device=dev)[None, :, None]
        nd, nu = _next(c40), _next(~c40)
        self.td[w] = torch.where(nd < L, (nd - i).float() / gpdata.HZ, cap).clamp(max=cap)
        self.lo[w] = torch.where(c40, torch.where(nu < L, (nu - i).float() / gpdata.HZ, cap).clamp(max=cap), 0.0)
        self.contact[w], self.com[w] = c40, com40
        self.hold[w] = torch.stack([h40[..., 0], y, h40[..., 1]], -1)
        self.p[w], self.since[w] = 0, 0
        self.replans += len(w)

    def __call__(self, s: Any) -> torch.Tensor:
        fresh = s.episode_step == 0
        down_before = self.h_down[:, -1].clone()
        self.p = torch.where(fresh, 0, (self.p + 1).clamp(max=self.contact.shape[1] - 1))
        self.since += 1
        if bool(fresh.any()):
            self._clip_history(fresh.nonzero().squeeze(1))
        self._measure(s)
        touchdown = (self.h_down[:, -1] & ~down_before).any(1) & ~fresh
        due = fresh | (self.since >= self.every) | (touchdown & (self.since >= self.min_gap))
        if bool(due.any()):
            self._replan(due.nonzero().squeeze(1))
        n = self.task.n
        k = (self.p[:, None] + self.taps).clamp(max=self.contact.shape[1] - 1)
        i = torch.arange(n, device=self.dev)[:, None]
        return plan_obs(self.task, s, self.contact[i, k], self.hold[i, k], self.td[i, k], self.lo[i, k], self.com[i, k])


class LazyReplanner(Replanner):
    """:class:`Replanner` with event triggers (gp.lazy) in place of "every ``every`` steps and at every touchdown":
    a new plan only when the clip's intent changes (``GOAL``), the body leaves the plan (``DRIFT``), a foot lands off
    its planned foothold or touches down where the plan has it swinging (``STEP``), or the student's farthest
    lookahead tap would run past the plan's end (``LOW``). ``self.log`` counts generations by reason.

    The lookahead limits how lazy this can be: the student reads the plan up to ``task.taps[-1]`` frames ahead (40,
    1 s) and a plan holds ``STRIDE * FUT`` (48, 1.2 s), so ``LOW`` still fires 0.2 s after each generation, exactly
    today's 5 Hz. The events only remove the replans at uneventful touchdowns. Playing a plan for longer needs a longer
    plan (e.g. ``FUT`` = 2 s plays 1 s) or a shorter lookahead. Terrain is the task's static ground here, so
    ``TERRAIN`` is not checked."""

    def __init__(self, planner: Planner, task: Any, D: dict[str, Any], triggers: Any = None, stride: int = 5,
                 seed: int = 0) -> None:
        from . import lazy
        super().__init__(planner, task, D, every=1 << 30, min_gap=0, stride=stride, seed=seed)
        self.lazy, self.T, self.log = lazy, triggers or lazy.Triggers(), lazy.Log()
        n = task.n
        self.goal0 = torch.zeros(n, 2, device=self.dev)                               # the goal at the last generation
        self.unplanned = torch.zeros(n, 2, dtype=torch.long, device=self.dev)         # steps down while planned swinging

    def _replan(self, w: torch.Tensor) -> None:
        super()._replan(w)
        self.goal0[w] = self._goal(w)

    def __call__(self, s: Any) -> torch.Tensor:
        lz, T, hz = self.lazy, self.T, gpdata.HZ
        fresh = s.episode_step == 0
        down_before = self.h_down[:, -1].clone()
        L = self.contact.shape[1]
        self.p = torch.where(fresh, 0, (self.p + 1).clamp(max=L - 1))
        self.since += 1
        if bool(fresh.any()):
            self._clip_history(fresh.nonzero().squeeze(1))
        self._measure(s)
        n = self.task.n
        i = torch.arange(n, device=self.dev)
        down = self.h_down[:, -1]
        touchdown = down & ~down_before & ~fresh[:, None]
        self.unplanned = torch.where(down & ~self.contact[i, self.p], self.unplanned + 1, 0)
        due, why = lz.decide(
            T, hz, since=self.since, fresh=fresh, min_left=1,
            left=(L - 1 - self.p) - int(self.taps[-1]),               # frames before the farthest tap runs off the plan
            goal=lz.goal_shift(self._goal(i), self.goal0),
            drift_m=lz.drift(self.h_com[:, -1, [0, 2]], self.com[i, self.p][:, [0, 2]]),
            step=lz.step_miss(touchdown, self.h_hold[:, -1], self.hold[i, self.p][..., [0, 2]]),
            early=(self.unplanned >= T.frames(T.early_s, hz)).any(1))
        if bool(due.any()):
            self.log.add(why)
            self._replan(due.nonzero().squeeze(1))
            self.unplanned[due] = 0
        k = (self.p[:, None] + self.taps).clamp(max=L - 1)
        j = i[:, None]
        return plan_obs(self.task, s, self.contact[j, k], self.hold[j, k], self.td[j, k], self.lo[j, k], self.com[j, k])
