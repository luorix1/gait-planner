"""Foot contact losses for the full-pose planner (train_planner.py --contact).

The planner generates the pelvis and joint angles separately, and nothing tied them together where the feet are:
generated motion slid stance feet 3-5x faster than real data and, chained, floated. These losses put the feet back
in: the 18 contact spheres' positions through a differentiable stand-in for forward kinematics.

- :class:`FootFK`: an MLP from the 14 leg coordinates to every contact sphere's centre in the pelvis frame, fitted
  once to SPRINT's own forward kinematics (real poses plus noisy ones, so generated poses stay in range) and frozen.
  The Rajagopal knee moves on splines (coupled translations), so a learned map is shorter than porting the joints.
- :func:`contact_loss`: in the anchor frame, (a) every sphere's position against the real window's, and (b) for the
  spheres that are really on the ground in two consecutive frames (the foot down and the sphere within 1 cm of that
  foot's lowest), the predicted displacement against the real one: a stance foot stays where the data's stays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sprint.recipes.imitation import _qmul

from . import data as gpdata

LEG = ("hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l", "subtalar_angle_l",
       "mtp_angle_l", "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
       "subtalar_angle_r", "mtp_angle_r")
POS_SCALE = 0.10            # m: a 10 cm sphere error costs 1
STEP_SCALE = 0.02           # m per 20 Hz frame (0.4 m/s): a stance sphere drifting this much costs 1
GROUND_TOL = 0.01           # m over the foot's lowest sphere still counts as on the ground


class FootFK(nn.Module):
    """Leg coordinates ``[..., 14]`` (``LEG``) -> contact sphere centres ``[..., S, 3]`` in the pelvis frame: one
    MLP per leg, each from that leg's 7 coordinates to its own spheres."""

    def __init__(self, side: Any, width: int = 512) -> None:
        super().__init__()
        side = torch.as_tensor(np.asarray(side), dtype=torch.long)
        self.n = len(side)
        self.register_buffer("radius", torch.zeros(self.n))
        self.register_buffer("side", side)                                   # 0 = l, 1 = r
        self.nets = nn.ModuleList(
            nn.Sequential(nn.Linear(14, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width),
                          nn.SiLU(), nn.Linear(width, 3 * int((side == s).sum()))) for s in (0, 1))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        out = q.new_zeros(*q.shape[:-1], self.n, 3)
        for s, net in enumerate(self.nets):
            qs = q[..., 7 * s:7 * s + 7]
            out[..., self.side == s, :] = net(torch.cat([qs.sin(), qs.cos()], -1)).unflatten(-1, (-1, 3)).to(out.dtype)
        return out


def sphere_centres(model: Any, q: np.ndarray, names: list[str]) -> np.ndarray:
    """SPRINT's forward kinematics with the pelvis at the origin, unrotated: sphere centres ``[N, S, 3]``."""
    from sprint.recipes.motion import _rotate, forward_kinematics_np, model_layout
    L, h = model_layout(model), model.host
    sph = L.contact_spheres
    body = h["geom_body"][sph].astype(int)
    full = np.zeros((len(q), len(names)), np.float32)
    full[:, [names.index(n) for n in LEG]] = q
    xp, xq = forward_kinematics_np(model, full)
    return (xp[:, body] + _rotate(xq[:, body], np.broadcast_to(h["geom_pos"][sph], xp[:, body].shape))).astype(np.float32)


def fit(D: dict[str, Any], device: Any, path: Path, poses: int = 400_000, iters: int = 30_000) -> FootFK:
    """Fit (or load from ``path``) the stand-in on ``poses`` leg poses from ``D``, half with 0.15 rad of noise."""
    from sprint.recipes.motion import model_layout
    model = gpdata.load_model()
    L, h = model_layout(model), model.host
    sph = L.contact_spheres
    fk = FootFK([0 if model.body_names[int(b)].endswith("_l") else 1 for b in h["geom_body"][sph]]).to(device)
    fk.radius.copy_(torch.as_tensor(h["geom_size"][sph][:, 0], dtype=torch.float32))
    fk.index = {str(model.geom_names[int(g)]): i for i, g in enumerate(sph)}      # sphere name -> output row
    if path.exists():
        fk.load_state_dict(torch.load(path, map_location=device))
        return fk.eval().requires_grad_(False)
    names = list(D["coord_names"])
    rng = np.random.default_rng(0)
    q = np.asarray(D["qpos"])[rng.integers(0, len(D["qpos"]), poses)][:, [names.index(n) for n in LEG]]
    q[poses // 2:] += rng.normal(0.0, 0.15, q[poses // 2:].shape)
    cache = path.with_suffix(".samples.npz")                # the FK samples, so a refit skips the slow part
    if cache.exists():
        q, y = np.load(cache)["q"], np.load(cache)["y"]
    else:
        y = np.concatenate([sphere_centres(model, q[i:i + 50_000], names) for i in range(0, poses, 50_000)])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, q=q, y=y)
    Q, Y = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (q, y))
    n_val = 20_000
    opt = torch.optim.AdamW(fk.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(iters):
        i = torch.randint(n_val, len(Q), (8192,), device=device, generator=gen)
        loss = (fk(Q[i]) - Y[i]).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        err = (fk(Q[:n_val]) - Y[:n_val]).norm(dim=-1)
    print(f"FootFK: held-out sphere error median {1000 * err.median():.1f} mm, p99 {1000 * err.quantile(0.99):.1f} mm",
          flush=True)
    if float(err.quantile(0.99)) > 0.01:
        raise RuntimeError("FootFK is not accurate enough (p99 > 1 cm)")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fk.state_dict(), path)
    return fk.eval().requires_grad_(False)


def spheres(fk: FootFK, x: torch.Tensor, C: int, leg_cols: list[int]) -> torch.Tensor:
    """Full-pose windows ``[B, K, c]`` (unnormalised) -> sphere centres ``[B, K, S, 3]`` in the anchor frame (x
    forward, y up, z lateral, origin at the anchor's centre of mass)."""
    from .planner import qexp, yaw_quat
    psi = torch.atan2(x[..., 10], x[..., 9])
    quat = _qmul(yaw_quat(psi), qexp(x[..., 11:14]))
    local = fk(x[..., leg_cols])
    u, w = quat[..., None, :3], quat[..., None, 3:]
    t = 2.0 * torch.cross(u.expand_as(local), local, dim=-1)
    return x[..., None, C:C + 3] + local + w * t + torch.cross(u.expand_as(t), t, dim=-1)


def contact_loss(fk: FootFK, pred: torch.Tensor, real: torch.Tensor, C: int, leg_cols: list[int]
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """(sphere position loss, stance drift loss) of predicted future frames ``pred`` against the real ``real``
    (both ``[B, K, c]``, unnormalised)."""
    sp, sr = spheres(fk, pred, C, leg_cols), spheres(fk, real, C, leg_cols)
    pos = ((sp - sr) / POS_SCALE).square().sum(-1).mean()
    bottom = sr[..., 1] - fk.radius                                          # [B, K, S]
    down = real[..., :2] > 0                                                 # [B, K, 2] l, r
    low = torch.stack([bottom[..., fk.side == s].amin(-1) for s in (0, 1)], -1)
    on = down[..., fk.side] & (bottom - low[..., fk.side] < GROUND_TOL)      # [B, K, S]
    both = on[:, 1:] & on[:, :-1]
    step = ((sp[:, 1:] - sp[:, :-1]) - (sr[:, 1:] - sr[:, :-1])) / STEP_SCALE
    drift = (step.square().sum(-1) * both).sum() / both.sum().clamp_min(1)
    return pos, drift


if __name__ == "__main__":                       # fit and check on the active dataset
    D = gpdata.load("train")
    fit(D, "cuda" if torch.cuda.is_available() else "cpu", gpdata.ROOT / "data" / "foot_fk.pt")
