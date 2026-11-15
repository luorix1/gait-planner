"""The planner (plan doc, step 3): a diffusion model of plan windows, conditioned on the ground ahead and a goal.

Like PARC's generator (Xu et al. 2025: motion diffusion conditioned on a local heightmap and a target), but it
generates the plan, not the pose. A plan frame (20 Hz) is each foot's down flag (+-1) and the xz of its current or next
foothold, the centre of mass, and the heading (``CHANNELS``). A window is ``HIST`` frames of the plan so far (given,
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

from . import data as gpdata
from .smp import _alphas_cumprod, _Block, _timestep_embedding
from .terrain import Ground, atlas, hf_height

STRIDE = 2                                   # data frames (40 Hz) per plan frame
HZ = gpdata.HZ / STRIDE
HIST, FUT, KEEP = 8, 24, 10                  # 0.4 s given, 1.2 s generated, 0.5 s kept per rollout step
PATCH_F = tuple(np.round(np.arange(-0.6, 3.01, 0.1), 2))
PATCH_L = tuple(np.round(np.arange(-0.8, 0.81, 0.1), 2))
CHANNELS = ("down_l", "down_r", "hold_l_f", "hold_l_lat", "hold_r_f", "hold_r_lat", "com_f", "com_y", "com_lat",
            "cos_dpsi", "sin_dpsi")
C = len(CHANNELS)
GOAL = [6, 8]                                # the goal is the last generated frame's com_f, com_lat
MIRROR_PERM = [1, 0, 4, 5, 2, 3, 6, 7, 8, 9, 10]
MIRROR_SIGN = [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
T_DIFF = 50
MIN_STD = 0.05


# --- plan frames ---------------------------------------------------------------------------------------------------

def heading(quat: torch.Tensor) -> torch.Tensor:
    """Heading (rotation about +y) of a body's x axis, from quaternions ``[..., 4]`` (x, y, z, w)."""
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(-2 * (x * z - w * y), 1 - 2 * (y * y + z * z))


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


def features(down: torch.Tensor, hold: torch.Tensor, com: torch.Tensor, psi: torch.Tensor, a_com: torch.Tensor,
             a_psi: torch.Tensor) -> torch.Tensor:
    """World plan frames (down ``[B, K, 2]``, foothold xz ``[B, K, 2, 2]``, com ``[B, K, 3]``, heading ``[B, K]``) ->
    ``[B, K, C]`` in the frame of anchor (``a_com`` ``[B, 3]``, ``a_psi`` ``[B]``)."""
    o = a_com[:, [0, 2]]
    cxz = to_frame(com[..., [0, 2]], o, a_psi)
    dpsi = psi - a_psi[:, None]
    return torch.cat([down.float() * 2 - 1, to_frame(hold, o, a_psi).flatten(2), cxz[..., :1],
                      (com[..., 1] - a_com[:, 1:2])[..., None], cxz[..., 1:], dpsi.cos()[..., None],
                      dpsi.sin()[..., None]], -1)


def world(x: torch.Tensor, a_com: torch.Tensor, a_psi: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Inverse of :func:`features`: ``[B, K, C]`` -> (down, foothold xz, com, heading), the down flag thresholded at 0."""
    o = a_com[:, [0, 2]]
    cxz = from_frame(x[..., GOAL], o, a_psi)
    com = torch.stack([cxz[..., 0], x[..., 7] + a_com[:, 1:2], cxz[..., 1]], -1)
    return (x[..., :2] > 0, from_frame(x[..., 2:6].unflatten(-1, (2, 2)), o, a_psi), com,
            a_psi[:, None] + torch.atan2(x[..., 10], x[..., 9]))


def patch(G: Ground, clip: torch.Tensor, a_com: torch.Tensor, a_psi: torch.Tensor) -> torch.Tensor:
    """Ground heights ``[B, len(PATCH_F), len(PATCH_L)]`` around each anchor, relative to its centre-of-mass height."""
    g = torch.tensor([(f, lat) for f in PATCH_F for lat in PATCH_L], dtype=torch.float32, device=a_com.device)
    xz = from_frame(g.expand(len(clip), -1, -1), a_com[:, [0, 2]], a_psi)
    return (G(clip, xz[..., 0], xz[..., 1]) - a_com[:, 1:2]).view(len(clip), len(PATCH_F), len(PATCH_L))


def mirror(x: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The same windows reflected left to right: feet swapped, lateral coordinates and heading changes negated, the
    patch's lateral axis reversed (``PATCH_L`` is symmetric)."""
    return x[..., MIRROR_PERM] * x.new_tensor(MIRROR_SIGN), p.flip(-1)


class Windows:
    """Plan windows of a loaded dataset (gp.data.load), on ``device``. ``anchors``: every frame with a full window in
    its clip; ``self(a)`` -> (windows ``[B, HIST + FUT, C]``, patches, goals ``[B, 2]``) at anchor frames ``a``."""

    def __init__(self, D: dict[str, Any], device: Any = "cpu") -> None:
        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
        self.down = T(D["plan_contact"], torch.bool)
        self.hold = T(D["foothold"])[..., [0, 2]]
        self.com = T(D["com"])
        self.psi = heading(T(D["record"])[:, gpdata.record_layout(len(D["tracked"]))["root_quat"]])
        self.clip = T(np.repeat(np.arange(len(D["frames"])), D["frames"]), torch.long)
        self.G = Ground.of(D, device)
        self.anchors = T(np.concatenate([np.arange(s + STRIDE * (HIST - 1), s + n - STRIDE * FUT)
                                         for s, n in zip(D["start"], D["frames"], strict=True)]), torch.long)
        self.k = STRIDE * torch.arange(-HIST + 1, FUT + 1, device=device)

    def __call__(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = a[:, None] + self.k
        x = features(self.down[k], self.hold[k], self.com[k], self.psi[k], self.com[a], self.psi[a])
        return x, patch(self.G, self.clip[a], self.com[a], self.psi[a]), x[:, -1, GOAL]

    def batch(self, n: int, gen: torch.Generator | None = None, mirrored: bool = True
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``n`` random windows; with ``mirrored`` a random half reflected left to right."""
        dev = self.anchors.device
        x, p, _ = self(self.anchors[torch.randint(len(self.anchors), (n,), device=dev, generator=gen)])
        if mirrored:
            flip = torch.rand(n, device=dev, generator=gen) < 0.5
            x[flip], p[flip] = mirror(x[flip], p[flip])
        return x, p, x[:, -1, GOAL]


# --- the network ---------------------------------------------------------------------------------------------------

class PlanNet(nn.Module):
    """x0-predicting DiT over [ground tokens, plan tokens]; the diffusion step and the goal condition through adaLN."""

    def __init__(self, layers: int = 4, heads: int = 4, head_dim: int = 64) -> None:
        super().__init__()
        D = heads * head_dim
        self.D = D
        self.inp = nn.Linear(C + 1, D)                                # + the "given" flag
        self.pe = nn.Parameter(torch.randn(1, HIST + FUT, D) * 0.02)
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.SiLU(), nn.Conv2d(32, D, 3, 2, 1))
        n_tok = self.conv(torch.zeros(1, 1, len(PATCH_F), len(PATCH_L))).flatten(2).shape[2]
        self.gpe = nn.Parameter(torch.randn(1, n_tok, D) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        self.goal = nn.Sequential(nn.Linear(2, D), nn.SiLU(), nn.Linear(D, D))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        self.blocks = nn.ModuleList(_Block(D, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, C)
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

    def __init__(self, T: int = T_DIFF, layers: int = 4, heads: int = 4, head_dim: int = 64) -> None:
        super().__init__()
        self.cfg = dict(T=T, layers=layers, heads=heads, head_dim=head_dim)
        self.net = PlanNet(layers, heads, head_dim)
        self.ema = torch.optim.swa_utils.AveragedModel(self.net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995))
        self.register_buffer("ab", _alphas_cumprod(T))
        for k, n in (("x", C), ("p", 1), ("g", 2)):
            self.register_buffer(f"{k}_mean", torch.zeros(n))
            self.register_buffer(f"{k}_std", torch.ones(n))

    def set_normaliser(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor) -> None:
        for k, v in (("x", x.reshape(-1, C)), ("p", p.reshape(-1, 1)), ("g", g)):
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
                 ema: bool = True) -> torch.Tensor:
        """The future ``[B, FUT, C]`` after ``hist`` ``[B, HIST, C]``: deterministic DDIM over every diffusion step."""
        pad = hist.new_zeros(len(hist), FUT, C)
        h, p, g = self._n(torch.cat([hist, pad], 1), p, g)
        h = h[:, :HIST]
        net = self.ema if ema else self.net
        x = torch.randn(len(h), FUT, C, device=h.device, generator=gen)
        for t in range(self.cfg["T"] - 1, -1, -1):
            x0 = net(torch.cat([h, x], 1), torch.full((len(h),), t, device=h.device), p, g)[:, HIST:]
            if t == 0:
                x = x0
                break
            eps = (x - self.ab[t].sqrt() * x0) / (1 - self.ab[t]).sqrt()
            x = self.ab[t - 1].sqrt() * x0 + (1 - self.ab[t - 1]).sqrt() * eps
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

def to_plan(down: np.ndarray, hold: np.ndarray, com: np.ndarray, psi: np.ndarray, ground: Any) -> dict[str, Any]:
    """A 20 Hz world plan (down ``[K, 2]``, foothold xz ``[K, 2, 2]``, com ``[K, 3]``, heading ``[K]``) -> the tasks'
    40 Hz plan with gp.free.stand_in_plan's keys. ``ground(x, z)``: terrain height (numpy)."""
    K = len(down)
    n = STRIDE * (K - 1) + 1
    i, t = np.arange(n) // STRIDE, np.arange(n) / STRIDE
    c, hz = down[i], hold[i]
    feet = np.stack([hz[..., 0], ground(hz[..., 0], hz[..., 1]), hz[..., 1]], -1)
    hold3, td, lo = zip(*(gpdata.plan_targets(c[:, k], feet[:, k]) for k in (0, 1)), strict=True)
    com40 = np.stack([np.interp(t, np.arange(K), com[:, j]) for j in range(3)], 1)
    s_com = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(com40[:, [0, 2]], axis=0), axis=1))]
    return dict(plan_contact=c, foothold=np.stack(hold3, 1), t_touchdown=np.stack(td, 1), t_liftoff=np.stack(lo, 1),
                com=com40, s_com=s_com, heading=np.interp(t, np.arange(K), np.unwrap(psi)))


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
        recent = [torch.cat([s[j] for s in seq], 1)[:, -HIST:] for j in range(4)]
        hist = features(*recent, a_com, a_psi)
    full = [torch.cat([s[j] for s in seq], 1).cpu().numpy() for j in range(4)]
    return [to_plan(*(f[b] for f in full),
                    ground=lambda x, z, T=T: hf_height(T["heights"], T["block"], T["cell"], T["origin"], x, z))
            for b, T in enumerate(tiles)]
