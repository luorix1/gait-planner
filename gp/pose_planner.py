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
3. **Classifier-free guidance on the goal**: the goal is dropped (a learned "no goal" token) ``GOAL_DROP`` of the time
   in training; sampling mixes the two predictions, ``x = x_none + w (x_goal - x_none)``.
4. **A smoothed anchor heading**: the circular mean of the pelvis heading over the last ``SMOOTH_S`` (causal), not the
   pelvis yaw at one frame, which swings with every step.

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
import torch.nn.functional as F
from sprint.recipes.imitation import _qmul

from . import data as gpdata
from .footfk import LEG, FootFK
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
GUIDANCE = 2.0
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


def patch(G: Ground, clip: torch.Tensor, o: torch.Tensor, a_psi: torch.Tensor, g0: torch.Tensor) -> torch.Tensor:
    """Ground heights ``[B, len(PATCH_F), len(PATCH_L)]`` around each anchor, relative to the ground under it."""
    g = torch.tensor([(f, lat) for f in PATCH_F for lat in PATCH_L], dtype=torch.float32, device=o.device)
    xz = from_frame(g.expand(len(clip), -1, -1), o, a_psi)
    return (G(clip, xz[..., 0], xz[..., 1]) - g0[:, None]).view(len(clip), len(PATCH_F), len(PATCH_L))


class Windows:
    """gp.planner.Windows for the version-2 frames: ``batch(n)`` -> (windows ``[n, HIST + FUT, C]``, patches,
    goals ``[n, 2]``)."""

    def __init__(self, D: dict[str, Any], device: Any, fk: FootFK, balance: bool = False, terrain: bool = False) -> None:
        self.fr = Frames(D, device, fk)
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

    def __call__(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o, psi, g0 = self.anchor(a)
        x = features(self.fr, a[:, None] + self.k, o, psi, g0)
        return x, patch(self.fr.G, self.fr.clip[a], o, psi, g0), x[:, -1, GOAL]

    def batch(self, n: int, gen: torch.Generator | None = None, mirrored: bool = True
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = self.anchors.device
        i = (torch.randint(len(self.anchors), (n,), device=dev, generator=gen) if self.w is None
             else torch.multinomial(self.w, n, replacement=True, generator=gen))
        x, p, _ = self(self.anchors[i])
        if mirrored:
            flip = torch.rand(n, device=dev, generator=gen) < 0.5
            x[flip] = x[flip][..., self.perm] * x.new_tensor(self.sign)
            p[flip] = p[flip].flip(-1)
        return x, p, x[:, -1, GOAL]


class Net(nn.Module):
    """gp.planner.PlanNet with a learned "no goal" embedding for classifier-free guidance."""

    def __init__(self, layers: int, heads: int, head_dim: int) -> None:
        super().__init__()
        D = heads * head_dim
        self.D = D
        self.inp = nn.Linear(C + 1, D)
        self.pe = nn.Parameter(torch.randn(1, HIST + FUT, D) * 0.02)
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.SiLU(), nn.Conv2d(32, D, 3, 2, 1))
        n_tok = self.conv(torch.zeros(1, 1, len(PATCH_F), len(PATCH_L))).flatten(2).shape[2]
        self.gpe = nn.Parameter(torch.randn(1, n_tok, D) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        self.goal = nn.Sequential(nn.Linear(2, D), nn.SiLU(), nn.Linear(D, D))
        self.null = nn.Parameter(torch.zeros(1, D))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        self.blocks = nn.ModuleList(_Block(D, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, C)
        given = torch.zeros(1, HIST + FUT, 1)
        given[:, :HIST] = 1.0
        self.register_buffer("given", given)

    def forward(self, x: torch.Tensor, t: torch.Tensor, p: torch.Tensor, goal: torch.Tensor,
                keep: torch.Tensor) -> torch.Tensor:
        h = self.inp(torch.cat([x, self.given.expand(len(x), -1, -1)], -1)) + self.pe
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


def ground_under(p: torch.Tensor, f: torch.Tensor, lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The lowest of the four patch heights around each anchor-frame point (``f``, ``lat`` ``[B, ...]``; patch ``p``
    ``[B, len(PATCH_F), len(PATCH_L)]``) and whether it is on the patch. The lowest, not interpolated: interpolation
    smears a box edge over a cell and would put a real foot on the floor beside the box "inside" it."""
    nf, nl = len(PATCH_F), len(PATCH_L)
    fi, li = (f - PATCH_F[0]) / (PATCH_F[1] - PATCH_F[0]), (lat - PATCH_L[0]) / (PATCH_L[1] - PATCH_L[0])
    on = (fi >= 0) & (fi <= nf - 1) & (li >= 0) & (li <= nl - 1)
    i, j = fi.floor().clamp(0, nf - 2).long(), li.floor().clamp(0, nl - 2).long()
    flat = p.flatten(1)

    def at(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.gather(flat, 1, (a * nl + b).flatten(1)).view_as(f)
    return torch.minimum(torch.minimum(at(i, j), at(i + 1, j)), torch.minimum(at(i, j + 1), at(i + 1, j + 1))), on


class PosePlanner(nn.Module):
    """Denoiser, EMA copy, schedule, normalisers and the frozen foot stand-in; methods take unnormalised tensors."""

    def __init__(self, T: int = T_DIFF, layers: int = 6, heads: int = 8, head_dim: int = 64,
                 contact_w: float = CONTACT_W, goal_drop: float = GOAL_DROP, hist_noise: float = 0.0,
                 fk_w: float | None = None, pen_w: float = 0.0) -> None:
        super().__init__()
        self.cfg = dict(T=T, layers=layers, heads=heads, head_dim=head_dim, contact_w=contact_w, goal_drop=goal_drop,
                        hist_noise=hist_noise, fk_w=contact_w if fk_w is None else fk_w, pen_w=pen_w)
        self.c = C
        self.net = Net(layers, heads, head_dim)
        self.ema = torch.optim.swa_utils.AveragedModel(self.net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995))
        self.register_buffer("ab", _alphas_cumprod(T))
        for k, n in (("x", C), ("p", 1), ("g", 2)):
            self.register_buffer(f"{k}_mean", torch.zeros(n))
            self.register_buffer(f"{k}_std", torch.ones(n))
        self.fk: FootFK | None = None

    def attach(self, fk: FootFK) -> PosePlanner:
        """The frozen foot stand-in (not saved: gp.footfk.fit reloads it), with the pelvis com offset."""
        model = gpdata.load_model()
        fk.bcom = torch.as_tensor(np.asarray(model.host["body_com"]).reshape(-1, 3)[0], dtype=torch.float32,
                                  device=fk.radius.device)
        self.fk = fk
        return self

    def set_normaliser(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor) -> None:
        for k, v in (("x", x.reshape(-1, C)), ("p", p.reshape(-1, 1)), ("g", g)):
            getattr(self, f"{k}_mean").copy_(v.mean(0))
            getattr(self, f"{k}_std").copy_(v.std(0).clamp_min(MIN_STD))

    def _n(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (x - self.x_mean) / self.x_std, (p - self.p_mean) / self.p_std, (g - self.g_mean) / self.g_std

    def loss(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Denoising MSE on the generated frames + the weighted geometric losses (``self.parts`` keeps each)."""
        xn, pn, gn = self._n(x, p, g)
        B, dev = len(x), x.device
        t = torch.randint(0, self.cfg["T"], (B,), device=dev)
        a = self.ab[t][:, None, None]
        given = xn[:, :HIST]
        if self.cfg.get("hist_noise"):        # a history that is not quite the data's (the simulated body's, later)
            sd = torch.rand(B, 1, 1, device=dev) * self.cfg["hist_noise"]
            given = given + sd * torch.randn_like(given)
        xt = torch.cat([given, a.sqrt() * xn[:, HIST:] + (1 - a).sqrt() * torch.randn_like(xn[:, HIST:])], 1)
        keep = torch.rand(B, device=dev) >= self.cfg["goal_drop"]
        pred = self.net(xt, t, pn, gn, keep)[:, HIST:]
        mse = F.mse_loss(pred, xn[:, HIST:])
        self.parts = {"mse": float(mse.detach())}
        if not (self.cfg["contact_w"] or self.cfg["fk_w"] or self.cfg["pen_w"]) or self.fk is None:
            return mse
        pu = (pred * self.x_std + self.x_mean).float()
        ru = x[:, HIST - 1:].float()                                      # the last given frame + the future
        pu = torch.cat([ru[:, :1], pu], 1)                                # so the first step is scored too
        with torch.autocast(x.device.type, enabled=False):                # millimetres need fp32, not bf16
            ch_p, kin_p, sph_p = geometry(self.fk, pu)
            ch_r, _, _ = geometry(self.fk, ru)
            ground, on_p = ground_under(p.float(), sph_p[..., 0], sph_p[..., 2])
        fk_l = ((ch_p - kin_p)[:, 1:] / GEO_SCALE).square().sum(-1).mean()
        on = ru[..., :4] > 0                                              # [B, K, 4]
        h_l = (((ch_p - ch_r)[..., 1] / GEO_SCALE).square() * on)[:, 1:].sum() / on[:, 1:].sum().clamp_min(1)
        both = on[:, 1:] & on[:, :-1]
        step = ((ch_p[:, 1:] - ch_p[:, :-1]) - (ch_r[:, 1:] - ch_r[:, :-1])) / GEO_SCALE
        v_l = (step.square().sum(-1) * both).sum() / both.sum().clamp_min(1)
        under = (torch.relu(ground - (sph_p[..., 1] - self.fk.radius) - PEN_TOL) / GEO_SCALE).square() * on_p
        pen_l = under[:, 1:].sum() / on_p[:, 1:].sum().clamp_min(1)
        self.parts.update(fk=float(fk_l.detach()), contact_h=float(h_l.detach()), contact_v=float(v_l.detach()),
                          pen=float(pen_l.detach()))
        return mse + self.cfg["contact_w"] * (h_l + v_l) + self.cfg["fk_w"] * fk_l + self.cfg["pen_w"] * pen_l

    @torch.no_grad()
    def generate(self, hist: torch.Tensor, p: torch.Tensor, g: torch.Tensor, gen: torch.Generator | None = None,
                 ema: bool = True, stride: int = 1, w: float = GUIDANCE) -> torch.Tensor:
        """The future ``[B, FUT, C]`` after ``hist`` ``[B, HIST, C]``: DDIM over every ``stride``-th step, guided
        with weight ``w`` (1 = the plain conditional model)."""
        pad = hist.new_zeros(len(hist), FUT, C)
        h, pn, gn = self._n(torch.cat([hist, pad], 1), p, g)
        h = h[:, :HIST]
        net = self.ema if ema else self.net
        B = len(h)
        keep = torch.cat([torch.ones(B, dtype=torch.bool, device=h.device), torch.zeros(B, dtype=torch.bool, device=h.device)])
        x = torch.randn(B, FUT, C, device=h.device, generator=gen)
        ts = list(range(self.cfg["T"] - 1, -1, -stride))
        for t, t_prev in zip(ts, [*ts[1:], -1], strict=True):
            tt = torch.full((2 * B,), t, device=h.device)
            out = net(torch.cat([h, x], 1).repeat(2, 1, 1), tt, pn.repeat(2, 1, 1), gn.repeat(2, 1), keep)[:, HIST:]
            x0 = out[B:] + w * (out[:B] - out[B:])
            if t_prev < 0:
                x = x0
                break
            eps = (x - self.ab[t].sqrt() * x0) / (1 - self.ab[t]).sqrt()
            x = self.ab[t_prev].sqrt() * x0 + (1 - self.ab[t_prev]).sqrt() * eps
        return x * self.x_std + self.x_mean

    def save(self, path: str | Path, **extra: Any) -> None:
        state = {k: v for k, v in self.state_dict().items() if not k.startswith("fk.")}   # gp.footfk.fit reloads it
        torch.save(dict(cfg=self.cfg, state=state, version=2, **extra), path)

    @classmethod
    def load(cls, path: str | Path, device: Any = "cpu") -> PosePlanner:
        ck = torch.load(path, map_location=device, weights_only=False)
        p = cls(**ck["cfg"]).to(device)
        p.load_state_dict({k: v for k, v in ck["state"].items() if not k.startswith("fk.")})
        return p.eval().requires_grad_(False)
