"""The full-pose planner, version 2 (train_planner.py --pose2): gp.planner's diffusion model with four changes from
Kimodo (Rempe et al. 2026) and MotionBricks (2026), after the first full-pose planner slid its stance feet 3-5x real
and floated when chained:

1. **Heights tied to the terrain.** Every height (pelvis, each foot point) and the ground patch are relative to the
   ground under the anchor's pelvis, so nothing drifts when plans chain: the old planner measured heights from the
   anchor's centre of mass, a body quantity, and floated. (Not from the ground directly under each point: that jumps
   at every step or box edge, and decoding it teleported the pelvis 55 cm onto a box.) The pelvis centre of mass
   replaces the whole-body centre of mass everywhere (anchor origin, goal, channels; user 2026-11-20).
2. **Foot points as outputs** (heel and ball of each foot, with their own contact flags), tied to the joint angles by a
   forward-kinematics consistency loss through gp.footfk.FootFK, and contact losses: a point the data has on the
   ground keeps the data's height and moves as the data's does. The consistency loss has its own weight (``fk_w``):
   it is ~200x stiffer than the denoising loss, and at 0.3 it destabilised training (v5, 2026-11-21).
   A **terrain penetration** loss (``pen_w``) keeps every contact sphere, from the joints, out of the ground patch:
   v5 crept into a 50 cm box without jumping, then snapped the pelvis up once the history put its feet inside.
   Every geometric loss is Huber past 2 cm and weighted by the window's signal share (alpha-bar of its noise level):
   squared, at every noise level, a few platform/box windows at high noise carried ~40 % of it and v6 diverged.
3. **Classifier-free guidance on the goal**: the goal is dropped (a learned "no goal" token) ``GOAL_DROP`` of the time
   in training; sampling mixes the two predictions, ``x = x_none + w (x_goal - x_none)``.
4. **A smoothed anchor heading**: the circular mean of the pelvis heading over the last ``SMOOTH_S`` (causal), not the
   pelvis yaw at one frame, which swings with every step.

Version 10 options (roadmap P1; all off by default, so v9 checkpoints load and sample as before):
- ``segments``: 9 segment poses (gp.footfk.SEGMENTS: origin + 6D rotation in the anchor frame) appended as channels,
  computed from the joints by gp.footfk.SegFK, so the joints stay the truth; consistency loss under ``fk_w``.
- ``hist_drop``: the history replaced by a learned "no history" token in that share of windows; sampling blends the
  predictions with and without it (``BLEND``, PARC's s = 0.65), so a memorised continuation cannot overrule the terrain.
- ``hist_noise_mode="body"``: history noise as a smooth whole-body offset (joints, pelvis height and tilt) re-run
  through FK, so a noisy history is still one body (white per-channel noise made joints and foot points disagree).
- ``goal_h``: the goal at a random horizon (``GOAL_H0``..FUT frames, given as an input) with ``GOAL_NOISE`` m of noise;
  v9's exact 1.2 s pelvis goal plus the history was nearly a fingerprint of a memorised clip.
- ``patch_res``: terrain patch cell size (v9 0.1 m; 0.05 m keeps a 22 cm tread at 4 cells, three stride-2 convolutions
  keep ~50 tokens). ``contact_surface``: a contact point's height over the terrain under it (exact heightfield) against
  the data's, not its absolute height. With a batch context (Windows.batch), penetration is checked on the exact
  heightfield, not the lowest of 4 patch cells. ``rand_terrain``: that share of windows on a random terrain patch
  (boxes, a staircase) with the penetration loss only.
- ``dropout`` (attention + residual), ``sample_steps`` (DDIM steps when sampling; v9 used all 50).

A frame (20 Hz, ``CHANNELS``): 4 contact flags (heel/ball, l/r, +-1), 2 footholds (forward, lateral), the pelvis
centre of mass (forward, lateral, height), the pelvis heading (cos, sin of its change from the anchor's)
and tilt (rotation vector), 17 joint coordinates, and 4 foot points (forward, lateral, height). Windows,
network, schedule and sampling otherwise as gp.planner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sprint.recipes.imitation import _qmul

from . import data as gpdata
from .footfk import LEG, SEGMENTS, FootFK, SegFK
from .planner import (
    FUT,
    HIST,
    MIN_STD,
    PATCH_F,
    PATCH_L,
    POSE_COORDS,
    STRIDE,
    T_DIFF,
    from_frame,
    heading,
    qexp,
    tilt,
    to_frame,
    yaw_quat,
)
from .smp import _alphas_cumprod, _Block, _timestep_embedding
from .terrain import Ground

SMOOTH_S = 1.0
GOAL_DROP = 0.1
CONTACT_W = 0.1             # weight of each geometric loss
GEO_SCALE = 0.02            # m: a 2 cm miss costs 1 (before the weight)
PEN_TOL = 0.01              # m of a sphere under the patch's ground that costs nothing (the data sinks too)
POINTS = ("heel_l", "ball_l", "heel_r", "ball_r")          # the data's contact flag order
CHANNELS = ("c_heel_l", "c_ball_l", "c_heel_r", "c_ball_r", "hold_l_f", "hold_l_lat", "hold_r_f", "hold_r_lat",
            "pel_f", "pel_lat", "pel_h", "cos_dpsi", "sin_dpsi", "tilt_x", "tilt_y", "tilt_z", *POSE_COORDS,
            *(f"{p}_{a}" for p in POINTS for a in ("f", "lat", "h")))
C = len(CHANNELS)
J0 = CHANNELS.index(POSE_COORDS[0])
P0 = CHANNELS.index("heel_l_f")
GOAL = [8, 9]
LEG_COLS = [J0 + POSE_COORDS.index(n) for n in LEG]
NSEG = len(SEGMENTS)
SEG_C = 9 * NSEG                      # per segment: origin (f, lat, h) + rotation columns 1, 2 in (f, up, lat)
SEG_W = 0.5                           # the segment block's weight in the denoising loss (the 45 base channels: 1)
ROT_SCALE = 0.035                     # a rotation-column miss of 0.035 (~2 deg) costs 1, as 2 cm does for a point
BLEND = 0.65                          # sampling: x = x_no_history + BLEND (x_history - x_no_history)
GOAL_H0, GOAL_NOISE = 8, 0.05         # random goal horizon from 0.4 s; goal noise (m)
Q_NOISE, H_NOISE, TILT_NOISE = 0.05, 0.01, 0.03   # body-mode history noise at full strength: rad, m, rad
SEG_FK = gpdata.ROOT / "data" / "seg_fk.pt"


def grid(res: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The terrain patch's (forward, lateral) sample points at cell size ``res`` (0.1 = gp.planner's PATCH_F/L)."""
    return (tuple(np.round(np.arange(-0.6, 3.0 + 1e-6, res), 3).tolist()),
            tuple(np.round(np.arange(-0.8, 0.8 + 1e-6, res), 3).tolist()))


DEFAULT_GRID = (PATCH_F, PATCH_L)


def _rot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(u.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(u.expand_as(t), t, dim=-1)


def mirror_spec(coord_names: list[str]) -> tuple[list[int], list[float]]:
    """``(perm, sign)``: a window reflected left to right is ``x[..., perm] * sign``."""
    from sprint.recipes.imitation import mirrored
    signs = gpdata.coord_mirror_signs(gpdata.load_model())
    names = list(coord_names)
    perm = [2, 3, 0, 1, 6, 7, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15]
    sign = [1.0] * 4 + [1.0, -1.0, 1.0, -1.0] + [1.0, -1.0, 1.0] + [1.0, -1.0] + [-1.0, -1.0, 1.0]
    perm += [J0 + POSE_COORDS.index(mirrored(n)) for n in POSE_COORDS]
    sign += [float(signs[names.index(n)]) for n in POSE_COORDS]
    for i in (2, 3, 0, 1):                                   # foot points: swap sides, negate lateral
        perm += [P0 + 3 * i, P0 + 3 * i + 1, P0 + 3 * i + 2]
        sign += [1.0, -1.0, 1.0]
    return perm, sign


def smoothed_heading(psi: torch.Tensor, start: np.ndarray, frames: np.ndarray, n: int) -> torch.Tensor:
    """Per frame, the circular mean of ``psi`` over that frame and the ``n - 1`` before it, within its clip."""
    out = torch.empty_like(psi)
    for s, f in zip(start.tolist(), frames.tolist(), strict=True):
        cs = torch.cumsum(torch.stack([psi[s:s + f].cos(), psi[s:s + f].sin()], -1), 0)
        lag = torch.cat([cs.new_zeros(n, 2), cs[:-n] if f > n else cs.new_zeros(0, 2)])[:f]
        m = cs - lag
        out[s:s + f] = torch.atan2(m[:, 1], m[:, 0])
    return out


class Frames:
    """World quantities per frame of a loaded dataset, the ingredients of :func:`features`."""

    def __init__(self, D: dict[str, Any], device: Any, fk: FootFK) -> None:
        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
        model = gpdata.load_model()
        lay = gpdata.record_layout(len(D["tracked"]))
        rec = T(D["record"])
        self.quat = rec[:, lay["root_quat"]] / rec[:, lay["root_quat"]].norm(dim=-1, keepdim=True)
        self.bcom = T(np.asarray(model.host["body_com"]).reshape(-1, 3)[0])
        self.pcom = rec[:, lay["root_pos"]] + _rot(self.quat, self.bcom.expand(len(rec), 3))
        self.psi = heading(self.quat)
        self.psi_s = smoothed_heading(self.psi, np.asarray(D["start"]), np.asarray(D["frames"]),
                                      int(round(SMOOTH_S * gpdata.HZ)))
        self.contact = T(D["contact"], torch.bool)
        self.hold = T(D["foothold"])[..., [0, 2]]
        names = list(D["coord_names"])
        q = T(D["qpos"])
        self.joints = q[:, [names.index(n) for n in POSE_COORDS]]
        self.clip = T(np.repeat(np.arange(len(D["frames"])), D["frames"]), torch.long)
        self.G = Ground.of(D, device)
        with torch.no_grad():
            pts = torch.cat([foot_points(fk, q[i:i + 200_000][:, [names.index(n) for n in LEG]])
                             for i in range(0, len(q), 200_000)])
        self.points = rec[:, None, lay["root_pos"]] + _rot(self.quat[:, None], pts)      # [N, 4, 3] world
        self.g_pcom = self.ground(self.clip, self.pcom)                 # the anchor's ground, when it anchors

    def ground(self, clip: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Terrain height under points ``p`` ``[N, 3]`` of clips ``clip`` ``[N]``."""
        out = torch.empty(len(p), device=p.device)
        for i in range(0, len(p), 500_000):                 # ponytail: chunked; Ground wants [N, S] per clip row
            j = slice(i, i + 500_000)
            out[j] = self.G(clip[j], p[j, None, 0], p[j, None, 2])[:, 0]
        return out


def foot_points(fk: FootFK, q_leg: torch.Tensor) -> torch.Tensor:
    """Leg coordinates ``[..., 14]`` -> heel and ball centres ``[..., 4, 3]`` in the pelvis frame (``POINTS``: the
    heel sphere; the ball = the mean of the two forefoot spheres on the calcaneus)."""
    return pick_points(fk, fk(q_leg))


def pick_points(fk: FootFK, s: torch.Tensor) -> torch.Tensor:
    """All contact spheres ``[..., S, 3]`` -> the ``POINTS`` ``[..., 4, 3]``."""
    h_l, l_l, m_l, h_r, l_r, m_r = (fk.index[k] for k in ("heel_l", "toe_lat_l", "toe_med_l", "heel_r", "toe_lat_r",
                                                          "toe_med_r"))
    return torch.stack([s[..., h_l, :], (s[..., l_l, :] + s[..., m_l, :]) / 2,
                        s[..., h_r, :], (s[..., l_r, :] + s[..., m_r, :]) / 2], -2)


def features(fr: Frames, k: torch.Tensor, o: torch.Tensor, a_psi: torch.Tensor, g0: torch.Tensor) -> torch.Tensor:
    """Frames ``k`` ``[B, K]`` -> ``[B, K, C]`` in the frame of the anchor (pelvis com xz ``o`` ``[B, 2]``, smoothed
    heading ``a_psi`` ``[B]``, heights over the ground under it ``g0`` ``[B]``)."""
    pc = fr.pcom[k]
    pxz = to_frame(pc[..., [0, 2]], o, a_psi)
    dpsi = fr.psi[k] - a_psi[:, None]
    pts = fr.points[k]
    pt = torch.cat([to_frame(pts[..., [0, 2]], o, a_psi), (pts[..., 1] - g0[:, None, None])[..., None]], -1).flatten(-2)
    return torch.cat([fr.contact[k].float() * 2 - 1, to_frame(fr.hold[k], o, a_psi).flatten(-2), pxz,
                      (pc[..., 1] - g0[:, None])[..., None], dpsi.cos()[..., None], dpsi.sin()[..., None],
                      tilt(fr.quat[k], fr.psi[k]), fr.joints[k], pt], -1)


def patch(G: Ground, clip: torch.Tensor, o: torch.Tensor, a_psi: torch.Tensor, g0: torch.Tensor,
          grid: tuple = DEFAULT_GRID) -> torch.Tensor:
    """Ground heights ``[B, len(grid[0]), len(grid[1])]`` around each anchor, relative to the ground under it."""
    pf, pl = grid
    g = torch.tensor([(f, lat) for f in pf for lat in pl], dtype=torch.float32, device=o.device)
    xz = from_frame(g.expand(len(clip), -1, -1), o, a_psi)
    return (G(clip, xz[..., 0], xz[..., 1]) - g0[:, None]).view(len(clip), len(pf), len(pl))


class Windows:
    """gp.planner.Windows for the version-2 frames: ``batch(n)`` -> (windows ``[n, HIST + FUT, C]``, patches,
    goals ``[n, 2]``)."""

    def __init__(self, D: dict[str, Any], device: Any, fk: FootFK, balance: bool = False, terrain: bool = False,
                 grid: tuple = DEFAULT_GRID, sfk: SegFK | None = None) -> None:
        self.fr = Frames(D, device, fk)
        self.grid, self.sfk = grid, sfk
        self.anchors = torch.as_tensor(np.concatenate([np.arange(s + STRIDE * (HIST - 1), s + n - STRIDE * FUT)
                                                       for s, n in zip(D["start"], D["frames"], strict=True)]),
                                       dtype=torch.long, device=device)
        self.k = STRIDE * torch.arange(-HIST + 1, FUT + 1, device=device)
        self.perm, self.sign = mirror_spec(D["coord_names"])
        self.w = None
        if balance:
            n_a = np.maximum(np.asarray(D["frames"]) - STRIDE * (HIST - 1 + FUT), 0)
            cw = gpdata.clip_weights(D, "sqrt", terrain=terrain)
            self.w = torch.as_tensor(np.repeat(np.divide(cw, n_a, out=np.zeros_like(cw), where=n_a > 0), n_a),
                                     dtype=torch.float32, device=device)

    def anchor(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fr = self.fr
        return fr.pcom[a][:, [0, 2]], fr.psi_s[a], fr.g_pcom[a]

    def __call__(self, a: torch.Tensor, segs: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o, psi, g0 = self.anchor(a)
        x = features(self.fr, a[:, None] + self.k, o, psi, g0)
        if segs and self.sfk is not None:
            x = torch.cat([x, seg_channels(self.sfk, x)], -1)
        return x, patch(self.fr.G, self.fr.clip[a], o, psi, g0, self.grid), x[:, -1, GOAL]

    def batch(self, n: int, gen: torch.Generator | None = None, mirrored: bool = True
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """(windows, patches, goals, context): the context (ground, clip, anchor, mirrored) lets the loss look up the
        exact heightfield."""
        dev = self.anchors.device
        i = (torch.randint(len(self.anchors), (n,), device=dev, generator=gen) if self.w is None
             else torch.multinomial(self.w, n, replacement=True, generator=gen))
        a = self.anchors[i]
        x, p, _ = self(a, segs=False)
        flip = torch.zeros(n, dtype=torch.bool, device=dev)
        if mirrored:
            flip = torch.rand(n, device=dev, generator=gen) < 0.5
            x[flip] = x[flip][..., self.perm] * x.new_tensor(self.sign)
            p[flip] = p[flip].flip(-1)
        if self.sfk is not None:                       # from the (mirrored) joints, so no segment mirror table
            x = torch.cat([x, seg_channels(self.sfk, x)], -1)
        o, psi, g0 = self.anchor(a)
        return x, p, x[:, -1, GOAL], dict(G=self.fr.G, clip=self.fr.clip[a], o=o, psi=psi, g0=g0, flip=flip)


class Net(nn.Module):
    """gp.planner.PlanNet with a learned "no goal" embedding for classifier-free guidance."""

    def __init__(self, layers: int, heads: int, head_dim: int, qk_norm: bool = False, qk_gain: bool = True, c: int = C,
                 n_goal: int = 2, grid: tuple = DEFAULT_GRID, dropout: float = 0.0, hist_null: bool = False) -> None:
        super().__init__()
        D = heads * head_dim
        self.D = D
        self.inp = nn.Linear(c + 1, D)
        self.pe = nn.Parameter(torch.randn(1, HIST + FUT, D) * 0.02)
        fine = len(grid[0]) > 40                         # a 5 cm patch: one more stride-2 convolution, ~50 tokens again
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.SiLU(),
                                  *([nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU()] if fine else []),
                                  nn.Conv2d(64 if fine else 32, D, 3, 2, 1))
        n_tok = self.conv(torch.zeros(1, 1, len(grid[0]), len(grid[1]))).flatten(2).shape[2]
        self.gpe = nn.Parameter(torch.randn(1, n_tok, D) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        self.goal = nn.Sequential(nn.Linear(n_goal, D), nn.SiLU(), nn.Linear(D, D))
        self.null = nn.Parameter(torch.zeros(1, D))
        self.null_hist = nn.Parameter(torch.zeros(1, 1, D)) if hist_null else None
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        self.blocks = nn.ModuleList(_Block(D, heads, qk_norm, qk_gain, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, c)
        given = torch.zeros(1, HIST + FUT, 1)
        given[:, :HIST] = 1.0
        self.register_buffer("given", given)

    def forward(self, x: torch.Tensor, t: torch.Tensor, p: torch.Tensor, goal: torch.Tensor,
                keep: torch.Tensor, keep_h: torch.Tensor | None = None) -> torch.Tensor:
        h = self.inp(torch.cat([x, self.given.expand(len(x), -1, -1)], -1))
        if keep_h is not None and self.null_hist is not None:          # "no history": the given frames' tokens replaced
            h = torch.cat([torch.where(keep_h[:, None, None], h[:, :HIST], self.null_hist.expand(len(x), HIST, -1)),
                           h[:, HIST:]], 1)
        h = h + self.pe
        g = self.conv(p[:, None]).flatten(2).transpose(1, 2) + self.gpe
        ge = torch.where(keep[:, None], self.goal(goal), self.null.expand(len(x), -1))
        t6 = self.ada(self.t_mlp(_timestep_embedding(t)) + ge).view(-1, 6, self.D)
        z = torch.cat([g, h], 1)
        for b in self.blocks:
            z = b(z, t6)
        return self.out(self.norm(z[:, g.shape[1]:]))


def geometry(fk: FootFK, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(foot points from the channels, foot points from the joints) ``[B, K, 4, 3]`` and every contact sphere from the
    joints ``[B, K, S, 3]``, all in the anchor frame with heights over the anchor's ground (x forward, y up, z
    lateral). ``x`` unnormalised."""
    ch = x[..., P0:P0 + 12].unflatten(-1, (4, 3))[..., [0, 2, 1]]       # (f, lat, h) -> (f, h, lat)
    psi = torch.atan2(x[..., 12], x[..., 11])
    quat = _qmul(yaw_quat(psi), qexp(x[..., 13:16]))
    pc = x[..., [8, 10, 9]]
    origin = pc - _rot(quat, fk.bcom.expand_as(pc))
    sph = origin[..., None, :] + _rot(quat[..., None, :], fk(x[..., LEG_COLS]))
    return ch, pick_points(fk, sph), sph


def robust(sq: torch.Tensor) -> torch.Tensor:
    """A squared error in ``GEO_SCALE`` units, linear past 1 (Huber): a few far-off points cannot take the gradient."""
    return torch.where(sq < 1, sq, 2 * sq.clamp_min(1).sqrt() - 1)


def ground_under(p: torch.Tensor, f: torch.Tensor, lat: torch.Tensor, grid: tuple = DEFAULT_GRID
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """The lowest of the four patch heights around each anchor-frame point (``f``, ``lat`` ``[B, ...]``; patch ``p``
    ``[B, len(PATCH_F), len(PATCH_L)]``) and whether it is on the patch. The lowest, not interpolated: interpolation
    smears a box edge over a cell and would put a real foot on the floor beside the box "inside" it."""
    pf, pl = grid
    nf, nl = len(pf), len(pl)
    fi, li = (f - pf[0]) / (pf[1] - pf[0]), (lat - pl[0]) / (pl[1] - pl[0])
    on = (fi >= 0) & (fi <= nf - 1) & (li >= 0) & (li <= nl - 1)
    i, j = fi.floor().clamp(0, nf - 2).long(), li.floor().clamp(0, nl - 2).long()
    flat = p.flatten(1)

    def at(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.gather(flat, 1, (a * nl + b).flatten(1)).view_as(f)
    return torch.minimum(torch.minimum(at(i, j), at(i + 1, j)), torch.minimum(at(i, j + 1), at(i + 1, j + 1))), on


def patch_ground(p: torch.Tensor, f: torch.Tensor, lat: torch.Tensor, grid: tuple) -> torch.Tensor:
    """The nearest patch cell's height under anchor-frame points (exact for :func:`random_terrain`'s cell boxes)."""
    nf, nl = len(grid[0]), len(grid[1])
    fi = ((f - grid[0][0]) / (grid[0][1] - grid[0][0])).round().clamp(0, nf - 1).long()
    li = ((lat - grid[1][0]) / (grid[1][1] - grid[1][0])).round().clamp(0, nl - 1).long()
    return torch.gather(p.flatten(1), 1, (fi * nl + li).flatten(1)).view_as(f)


def exact_ground(ctx: dict[str, Any], f: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    """The window's own terrain (Windows.batch context) under anchor-frame points ``[B, ...]``, over the anchor's
    ground; a mirrored window's lateral is flipped back first."""
    flip = ctx["flip"].view(-1, *[1] * (lat.dim() - 1))
    fl = torch.stack([f, torch.where(flip, -lat, lat)], -1).flatten(1, -2)          # [B, N, 2]
    xz = from_frame(fl, ctx["o"], ctx["psi"])
    return (ctx["G"](ctx["clip"], xz[..., 0], xz[..., 1]) - ctx["g0"][:, None]).view_as(f)


def random_terrain(p: torch.Tensor, grid: tuple, boxes: int = 4) -> torch.Tensor:
    """Patches ``[n, F, L]`` with up to ``boxes`` random boxes (-0.3..0.6 m) and, a third of the time, a staircase
    (rise 0.1-0.2 m up or down, tread 0.25-0.4 m), all at least 0.3 m ahead of the anchor so the body's own ground
    stays (PARC's random-terrain steps, with the penetration loss only)."""
    n, dev = len(p), p.device
    F_ = torch.as_tensor(grid[0], dtype=p.dtype, device=dev)[None, :, None]
    L_ = torch.as_tensor(grid[1], dtype=p.dtype, device=dev)[None, None, :]
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, 1, 1, device=dev)  # noqa: E731
    out = p.clone()
    stairs = torch.rand(n, 1, 1, device=dev) < 1 / 3
    f0, rise, tread = u(0.3, 1.5), u(0.1, 0.2) * torch.where(torch.rand(n, 1, 1, device=dev) < 0.5, 1.0, -1.0), u(0.25, 0.4)
    step = rise * torch.floor((F_ - f0) / tread + 1).clamp_min(0)
    out = torch.where(stairs & (F_ >= f0), step.expand_as(out), out)
    for _ in range(boxes):
        f0, df, l0, dl, h = u(0.3, 2.6), u(0.3, 1.5), u(-0.8, 0.6), u(0.3, 1.6), u(-0.3, 0.6)
        m = (torch.rand(n, 1, 1, device=dev) < 0.6) & (F_ >= f0) & (F_ <= f0 + df) & (L_ >= l0) & (L_ <= l0 + dl)
        out = torch.where(m, h.expand_as(out), out)
    return out


def seg_channels(sfk: SegFK, x: torch.Tensor) -> torch.Tensor:
    """Unnormalised frames ``[..., >= C]`` (anchor frame) -> the ``SEG_C`` segment channels from their joints and
    pelvis: per segment origin (f, lat, h over the anchor's ground) and rotation columns 1, 2 in (f, up, lat)."""
    psi = torch.atan2(x[..., 12], x[..., 11])
    quat = _qmul(yaw_quat(psi), qexp(x[..., 13:16]))
    pc = x[..., [8, 10, 9]]
    origin = pc - _rot(quat, sfk.bcom.expand_as(pc))
    pos, rot6 = sfk(x[..., J0:J0 + len(POSE_COORDS)])
    q9 = quat[..., None, :]
    pw = origin[..., None, :] + _rot(q9, pos)
    return torch.cat([pw[..., [0, 2, 1]], _rot(q9, rot6[..., :3]), _rot(q9, rot6[..., 3:])], -1).flatten(-2)


class PosePlanner(nn.Module):
    """Denoiser, EMA copy, schedule, normalisers and the frozen foot stand-in; methods take unnormalised tensors."""

    def __init__(self, T: int = T_DIFF, layers: int = 6, heads: int = 8, head_dim: int = 64,
                 contact_w: float = CONTACT_W, goal_drop: float = GOAL_DROP, hist_noise: float = 0.0,
                 fk_w: float | None = None, pen_w: float = 0.0, qk_norm: bool = False, qk_gain: bool = True,
                 segments: bool = False, hist_drop: float = 0.0, hist_noise_mode: str = "white", goal_h: bool = False,
                 patch_res: float = 0.1, contact_surface: bool = False, rand_terrain: float = 0.0, dropout: float = 0.0,
                 sample_steps: int | None = None) -> None:
        super().__init__()
        self.cfg = dict(T=T, layers=layers, heads=heads, head_dim=head_dim, contact_w=contact_w, goal_drop=goal_drop,
                        hist_noise=hist_noise, fk_w=contact_w if fk_w is None else fk_w, pen_w=pen_w, qk_norm=qk_norm,
                        qk_gain=qk_gain, segments=segments, hist_drop=hist_drop, hist_noise_mode=hist_noise_mode,
                        goal_h=goal_h, patch_res=patch_res, contact_surface=contact_surface, rand_terrain=rand_terrain,
                        dropout=dropout, sample_steps=sample_steps or T)
        self.c = C + (SEG_C if segments else 0)
        self.grid = DEFAULT_GRID if patch_res == 0.1 else grid(patch_res)
        self.net = Net(layers, heads, head_dim, qk_norm, qk_gain, self.c, 3 if goal_h else 2, self.grid, dropout,
                       hist_drop > 0)
        self.ema = torch.optim.swa_utils.AveragedModel(self.net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995))
        self.register_buffer("ab", _alphas_cumprod(T))
        for k, n in (("x", self.c), ("p", 1), ("g", 2)):
            self.register_buffer(f"{k}_mean", torch.zeros(n))
            self.register_buffer(f"{k}_std", torch.ones(n))
        self.fk: FootFK | None = None
        self.sfk: SegFK | None = None

    def attach(self, fk: FootFK) -> PosePlanner:
        """The frozen foot stand-in (not saved: gp.footfk.fit reloads it), with the pelvis com offset; with segments,
        also the frozen segment stand-in (gp.footfk.fit_seg's cached fit)."""
        from .footfk import fit_seg
        model = gpdata.load_model()
        fk.bcom = torch.as_tensor(np.asarray(model.host["body_com"]).reshape(-1, 3)[0], dtype=torch.float32,
                                  device=fk.radius.device)
        self.fk = fk
        if self.cfg.get("segments"):
            self.sfk = fit_seg(None, fk.radius.device, SEG_FK)
            self.sfk.bcom = fk.bcom
        return self

    def set_normaliser(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor, ctx: Any = None) -> None:
        for k, v in (("x", x.reshape(-1, self.c)), ("p", p.reshape(-1, 1)), ("g", g)):
            getattr(self, f"{k}_mean").copy_(v.mean(0))
            getattr(self, f"{k}_std").copy_(v.std(0).clamp_min(MIN_STD))

    def _n(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (x - self.x_mean) / self.x_std, (p - self.p_mean) / self.p_std, (g - self.g_mean) / self.g_std

    def body_frames(self, xh: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
        """History frames ``[B, H, c]`` (unnormalised) with a smooth whole-body offset of strength ``sd`` ``[B]``:
        joints offset and drifting, pelvis height and tilt offset, foot points and segments re-run through FK."""
        B, H = xh.shape[:2]
        dev = xh.device
        tau = torch.linspace(-0.5, 0.5, H, device=dev)[None, :, None]
        nj = len(POSE_COORDS)
        x = xh[..., :C].clone()
        x[..., J0:J0 + nj] += sd[:, None, None] * Q_NOISE * (torch.randn(B, 1, nj, device=dev) + torch.randn(B, 1, nj, device=dev) * tau)
        x[..., 10] += sd[:, None] * H_NOISE * torch.randn(B, 1, device=dev)
        x[..., 13:16] += sd[:, None, None] * TILT_NOISE * torch.randn(B, 1, 3, device=dev)
        x[..., P0:P0 + 12] = geometry(self.fk, x)[1][..., [0, 2, 1]].flatten(-2)
        return torch.cat([x, seg_channels(self.sfk, x)], -1) if self.sfk is not None else x

    def loss(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor, ctx: dict[str, Any] | None = None) -> torch.Tensor:
        """Denoising MSE on the generated frames + the weighted geometric losses (``self.parts`` keeps each). ``ctx``
        (Windows.batch) enables the exact-heightfield terms and random-terrain windows."""
        cfg = self.cfg
        B, dev = len(x), x.device
        keep_h = torch.rand(B, device=dev) >= cfg.get("hist_drop", 0.0)
        rt = (torch.rand(B, device=dev) < cfg.get("rand_terrain", 0.0)) if ctx is not None else torch.zeros(B, dtype=torch.bool, device=dev)
        if bool(rt.any()):
            p = p.clone()
            p[rt] = random_terrain(p[rt], self.grid)
        if cfg.get("goal_h"):                     # the goal at a random horizon, with noise; the horizon is an input
            hz = torch.randint(GOAL_H0, FUT + 1, (B,), device=dev)
            g = x[torch.arange(B, device=dev), HIST - 1 + hz][:, GOAL] + GOAL_NOISE * torch.randn(B, 2, device=dev)
        xh = x[:, :HIST]
        if cfg.get("hist_noise") and cfg.get("hist_noise_mode") == "body":
            xh = self.body_frames(xh, torch.rand(B, device=dev) * cfg["hist_noise"])
        xn, pn, gn = self._n(x, p, g)
        given = (xh - self.x_mean) / self.x_std
        if cfg.get("hist_noise") and cfg.get("hist_noise_mode", "white") == "white":   # v9
            sd = torch.rand(B, 1, 1, device=dev) * cfg["hist_noise"]
            given = given + sd * torch.randn_like(given)
        if cfg.get("goal_h"):
            gn = torch.cat([gn, (hz.float() / FUT)[:, None]], -1)
        t = torch.randint(0, cfg["T"], (B,), device=dev)
        a = self.ab[t][:, None, None]
        xt = torch.cat([given, a.sqrt() * xn[:, HIST:] + (1 - a).sqrt() * torch.randn_like(xn[:, HIST:])], 1)
        keep = torch.rand(B, device=dev) >= cfg["goal_drop"]
        pred = self.net(xt, t, pn, gn, keep, keep_h)[:, HIST:]
        d = ~rt                                                         # windows on their own terrain
        nd = d.sum().clamp_min(1)
        err = (pred - xn[:, HIST:]).square()
        mse = (err[..., :C].mean((1, 2)) * d).sum() / nd
        if self.c > C:
            mse = (mse + SEG_W * (err[..., C:].mean((1, 2)) * d).sum() / nd) / (1 + SEG_W)
        self.parts = {"mse": float(mse.detach())}
        if not (cfg["contact_w"] or cfg["fk_w"] or cfg["pen_w"]) or self.fk is None:
            return mse
        pu = (pred * self.x_std + self.x_mean).float()
        ru = x[:, HIST - 1:].float()                                      # the last given frame + the future
        pu = torch.cat([ru[:, :1], pu], 1)                                # so the first step is scored too
        surface = bool(cfg.get("contact_surface")) and ctx is not None
        with torch.autocast(x.device.type, enabled=False):                # millimetres need fp32, not bf16
            ch_p, kin_p, sph_p = geometry(self.fk, pu)
            ch_r, _, _ = geometry(self.fk, ru)
            if ctx is not None:
                ex = exact_ground(ctx, sph_p[..., 0], sph_p[..., 2])
                ground = torch.where(rt[:, None, None], patch_ground(p.float(), sph_p[..., 0], sph_p[..., 2], self.grid), ex)
                on_p = torch.ones_like(ground, dtype=torch.bool)
            else:
                ground, on_p = ground_under(p.float(), sph_p[..., 0], sph_p[..., 2], self.grid)
            if surface:
                gp_ = exact_ground(ctx, ch_p[..., 0], ch_p[..., 2])
                gr_ = exact_ground(ctx, ch_r[..., 0], ch_r[..., 2])
            seg_l = None
            if self.sfk is not None:
                sk = seg_channels(self.sfk, pu).unflatten(-1, (NSEG, 9))
                sp = pu[..., C:].unflatten(-1, (NSEG, 9))
        # each window's geometry counts by its signal share: at high noise x0 is a blur over modes (on the platform
        # yet or not), the data's one geometry is not its target, and those few windows dominated the gradient (v6)
        sig = self.ab[t][:, None, None]
        fk_l = (robust(((ch_p - kin_p)[:, 1:] / GEO_SCALE).square().sum(-1)) * sig).mean()
        if self.sfk is not None:
            seg_l = ((robust(((sp[..., :3] - sk[..., :3]) / GEO_SCALE).square().sum(-1))
                      + robust(((sp[..., 3:] - sk[..., 3:]) / ROT_SCALE).square().sum(-1))) * sig)[:, 1:].mean()
            fk_l = fk_l + seg_l
        on = (ru[..., :4] > 0) & d[:, None, None]                          # [B, K, 4]
        dh = (ch_p[..., 1] - gp_) - (ch_r[..., 1] - gr_) if surface else ch_p[..., 1] - ch_r[..., 1]
        h_l = (robust((dh / GEO_SCALE).square()) * on * sig)[:, 1:].sum() / on[:, 1:].sum().clamp_min(1)
        both = on[:, 1:] & on[:, :-1]
        both[:, 0] &= keep_h[:, None]                                     # no history seen: no seam to keep
        step = ((ch_p[:, 1:] - ch_p[:, :-1]) - (ch_r[:, 1:] - ch_r[:, :-1])) / GEO_SCALE
        v_l = (robust(step.square().sum(-1)) * both * sig).sum() / both.sum().clamp_min(1)
        assert ground.shape == sph_p.shape[:-1], (ground.shape, sph_p.shape)
        under = robust((torch.relu(ground - (sph_p[..., 1] - self.fk.radius) - PEN_TOL) / GEO_SCALE).square())
        pen_l = (under * on_p * sig)[:, 1:].sum() / on_p[:, 1:].sum().clamp_min(1)
        self.parts.update(fk=float(fk_l.detach()), contact_h=float(h_l.detach()), contact_v=float(v_l.detach()),
                          pen=float(pen_l.detach()))
        if seg_l is not None:
            self.parts["seg"] = float(seg_l.detach())
        return mse + cfg["contact_w"] * (h_l + v_l) + cfg["fk_w"] * fk_l + cfg["pen_w"] * pen_l

    @torch.no_grad()
    def generate(self, hist: torch.Tensor, p: torch.Tensor, g: torch.Tensor, gen: torch.Generator | None = None,
                 ema: bool = True, stride: int | None = None, w: float = 1.0, s: float | None = None) -> torch.Tensor:
        """The future ``[B, FUT, c]`` after ``hist`` ``[B, HIST, c]``: DDIM over every ``stride``-th step (default: the
        model's ``sample_steps``). Goal guidance ``w`` (1 = off: no unconditional pass; w = 2 worsened feet in v4) and,
        for a model trained with ``hist_drop``, the history blend ``s`` (default ``BLEND``). ``g`` ``[B, 2]`` (1.2 s
        ahead) or, for ``goal_h`` models, ``[B, 3]`` with the horizon as a share of 1.2 s. With segments, the
        segment channels of the result are recomputed from its joints (the joints are the truth)."""
        cfg = self.cfg
        if stride is None:
            stride = max(1, cfg["T"] // cfg.get("sample_steps", cfg["T"]))
        if s is None:
            s = BLEND if cfg.get("hist_drop") else 1.0
        B, dev = len(hist), hist.device
        pad = hist.new_zeros(B, FUT, self.c)
        h, pn, gn = self._n(torch.cat([hist, pad], 1), p, g[..., :2])
        if cfg.get("goal_h"):
            gn = torch.cat([gn, g[..., 2:3] if g.shape[-1] > 2 else torch.ones(B, 1, device=dev)], -1)
        h = h[:, :HIST]
        net = self.ema if ema else self.net
        passes = [(True, True)] + ([(True, False)] if s != 1 else []) + ([(False, True)] if w != 1 else [])
        n = len(passes)
        keep = torch.cat([torch.full((B,), kg, device=dev) for kg, _ in passes])
        keep_h = torch.cat([torch.full((B,), kh, device=dev) for _, kh in passes])
        x = torch.randn(B, FUT, self.c, device=dev, generator=gen)
        ts = list(range(cfg["T"] - 1, -1, -stride))
        for t, t_prev in zip(ts, [*ts[1:], -1], strict=True):
            tt = torch.full((n * B,), t, device=dev)
            out = net(torch.cat([h, x], 1).repeat(n, 1, 1), tt, pn.repeat(n, 1, 1), gn.repeat(n, 1), keep, keep_h)[:, HIST:]
            parts = dict(zip(passes, out.split(B), strict=True))
            x0 = parts[(True, True)]
            if s != 1:
                x0 = parts[(True, False)] + s * (x0 - parts[(True, False)])
            if w != 1:
                x0 = x0 + (w - 1) * (parts[(True, True)] - parts[(False, True)])
            if t_prev < 0:
                x = x0
                break
            eps = (x - self.ab[t].sqrt() * x0) / (1 - self.ab[t]).sqrt()
            x = self.ab[t_prev].sqrt() * x0 + (1 - self.ab[t_prev]).sqrt() * eps
        x = x * self.x_std + self.x_mean
        if self.sfk is not None:
            x = torch.cat([x[..., :C], seg_channels(self.sfk, x)], -1)
        return x

    def with_segments(self, x: torch.Tensor) -> torch.Tensor:
        """Base frames ``[..., C]`` -> the model's frames (segment channels from the joints appended if it has them)."""
        return torch.cat([x, seg_channels(self.sfk, x)], -1) if self.sfk is not None else x

    def save(self, path: str | Path, **extra: Any) -> None:
        state = {k: v for k, v in self.state_dict().items() if not k.startswith(("fk.", "sfk."))}  # stand-ins reload
        torch.save(dict(cfg=self.cfg, state=state, version=2, **extra), path)

    @classmethod
    def load(cls, path: str | Path, device: Any = "cpu") -> PosePlanner:
        ck = torch.load(path, map_location=device, weights_only=False)
        p = cls(**ck["cfg"]).to(device)
        p.load_state_dict({k: v for k, v in ck["state"].items() if not k.startswith(("fk.", "sfk."))})
        return p.eval().requires_grad_(False)
