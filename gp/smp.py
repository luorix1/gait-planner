"""Score-matching motion prior (SMP: Mu et al. 2025, arXiv 2512.03028), ported from MimicKit (Apache-2.0,
``mimickit/learning/tinymdm`` and ``smp_agent.py``, commit fbe1a36) to plain torch.

A small diffusion transformer (TinyMDM's DiT: 2 blocks, 4 heads x 64, adaLN-single timestep conditioning, a 50-step
squared-cosine schedule, epsilon prediction, L1 loss, EMA weights) learns the distribution of ``W``-frame windows of
motion features from real clips. As a reward it scores a window by score distillation: noise the window at a few
diffusion steps and measure how well the frozen prior predicts the noise (ESM-SDS loss). The reward is
``exp(-scale * mean_k(loss_k / mean|loss_k|))``, the mean taken over the policy's own samples (MimicKit's
``DiffNormalizer``), frozen after ``norm_samples``.

Same as MimicKit: the architecture, schedule, loss, the diffusion steps scored (22, 15, 8), the scale (6) and the
per-frame features relative to the window's last frame (AMP's ``compute_disc_obs`` with ``global_obs`` off and
``root_height_obs`` on, no joint velocities). Changed:
- **Features** are this body's: root position relative to the last frame's root in its heading frame, with the
  height above the terrain under it (not above a flat floor); root orientation (tangent and normal); the tracked
  coordinates as angles, not 6D rotations (OpenSim builds the 3-DOF hips and lumbar from three hinge coordinates
  each, in a fixed order; their middle axes stay far from 90 deg, so the angles never wrap or lock; knee, ankle,
  subtalar and mtp are 1-DOF); five key bodies (heels, toes, torso) relative to
  their frame's root; root linear and angular velocity, from differences of the window's own frames (so the data
  and the simulation compute them identically).
- **Window**: 13 frames at 40 Hz (0.325 s; MimicKit: 10 at 30 Hz).
- **Sampling** (train_prior.py): every frame equally likely by default; MimicKit draws a clip by its dataset weight,
  then a time in it. ``--balance`` gives every recording the same weight.
- **Normaliser floor**: per-feature std floored at 0.05, not 0.2 (heights and positions here are in metres and
  vary by a few cm).
- **Mirrored windows** (:func:`mirror_records`): half of every training batch is reflected left to right, so the
  prior sees each motion and its mirror image and has no side preference.
- **Conditional prior** (``n_cond`` > 0, ``train_prior.py --cond``; the paper's style-label prior): the clip's
  situation as the course states it (:func:`conditions`: ground grade, speed, terrain class, mode) is embedded and added
  to the timestep embedding, as MimicKit's class labels are; 10 % of training labels are dropped to a learned null
  embedding, so ``c=None`` scores with the unconditional branch. Scoring uses guidance scale 1 (the paper's setting):
  the conditional prediction alone.
- Not ported: classifier-free guidance at scales other than 1, generative state init (GSI; this body resets
  from clip states), DDPM sampling. ``diffusers`` is not needed: with epsilon prediction and no sample clipping,
  MimicKit's DDIM round trip returns the network's own epsilon.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sprint.recipes.imitation import _heading, _qconj, _qmul, _qrot, _rotvec

from .data import KEY_BODIES, record_layout

W = 13
T_DIFF = 50
SDS_STEPS = (22, 15, 8)
SDS_SCALE = 6.0
MIN_STD = 0.05
TERRAIN_CLASSES = ("level", "ramp_up", "ramp_down", "ramp_stand", "obstacles", "stairs_up", "stairs_down")
MODES = ("walk", "run", "start_stop", "stand")
N_COND = 2 + len(TERRAIN_CLASSES) + len(MODES)
LABEL_DROP = 0.1


def conditions(D: dict[str, Any], clips: Any = None) -> np.ndarray:
    """Per clip ``[n, N_COND]``, what a conditional prior is told: ground grade / 10 deg, speed / 2 m/s, terrain
    class and mode one-hot (an unknown class or mode = all zeros)."""
    c = np.arange(len(D["name"])) if clips is None else np.asarray(clips)
    return np.c_[np.asarray(D["grade_deg"], float)[c] / 10, np.asarray(D["speed_mps"], float)[c] / 2,
                 np.asarray(D["terrain_class"])[c, None] == np.array(TERRAIN_CLASSES),
                 np.asarray(D["mode"])[c, None] == np.array(MODES)].astype(np.float32)


def window_features(rec: torch.Tensor, nt: int, dt: float) -> torch.Tensor:
    """Style records ``[B, W + 1, D]`` (oldest first; :func:`gp.data.record_layout`) -> features ``[B, W, C]``."""
    lay = record_layout(nt)
    p, q = rec[..., lay["root_pos"]], rec[..., lay["root_quat"]]
    q = q / q.norm(dim=-1, keepdim=True)
    B, W1 = rec.shape[:2]
    kp = rec[..., lay["key_pos"]].reshape(B, W1, -1, 3)
    M = _heading(q[:, -1])                                            # world -> the last frame's heading frame
    rel = torch.einsum("bij,bwj->bwi", M, p - p[:, -1:])
    root = torch.stack([rel[..., 0], p[..., 1] - rec[..., lay["ground"]][..., 0], rel[..., 2]], -1)
    ex, ey = torch.zeros_like(p), torch.zeros_like(p)
    ex[..., 0], ey[..., 1] = 1.0, 1.0
    tan = torch.einsum("bij,bwj->bwi", M, _qrot(q, ex))
    nrm = torch.einsum("bij,bwj->bwi", M, _qrot(q, ey))
    key = torch.einsum("bij,bwkj->bwki", M, kp - p[..., None, :]).flatten(2)
    vel = (rel[:, 1:] - rel[:, :-1]) / dt
    omega = torch.einsum("bij,bwj->bwi", M, _rotvec(_qmul(q[:, 1:], _qconj(q[:, :-1])))) / dt
    pose = torch.cat([root, tan, nrm, rec[..., lay["joints"]], key], -1)[:, 1:]
    return torch.cat([pose, vel, omega], -1)


def mirror_maps(model: Any, tracked: list[str]) -> dict[str, Any]:
    """What the sagittal reflection does to a style record: the tracked coordinates' partner and sign
    (:func:`gp.data.coord_mirror_signs`, from forward kinematics) and the key bodies' partners."""
    from sprint.recipes.imitation import mirrored

    from .data import coord_mirror_signs
    names = list(model.coord_names)
    cs = coord_mirror_signs(model)
    return dict(joint_perm=[tracked.index(mirrored(n)) for n in tracked],
                joint_sign=[float(cs[names.index(n)]) for n in tracked],
                key_perm=[list(KEY_BODIES).index(mirrored(b)) for b in KEY_BODIES])


def mirror_records(rec: torch.Tensor, maps: dict[str, Any]) -> torch.Tensor:
    """Style records ``[..., D]`` of the same motion reflected left to right (world z -> -z; the terrain profiles do not
    vary along z): root and key-body z negated, the root quaternion reflected (x, y, z, w) -> (-x, -y, z, w), left and
    right coordinates and key bodies swapped."""
    nt = len(maps["joint_perm"])
    lay = record_layout(nt)
    out = rec.clone()
    out[..., 2] = -rec[..., 2]
    q = lay["root_quat"].start
    out[..., q:q + 2] = -rec[..., q:q + 2]
    sign = torch.as_tensor(maps["joint_sign"], dtype=rec.dtype, device=rec.device)
    out[..., lay["joints"]] = rec[..., lay["joints"]][..., maps["joint_perm"]] * sign
    kp = rec[..., lay["key_pos"]].unflatten(-1, (-1, 3))[..., maps["key_perm"], :].clone()
    kp[..., 2] = -kp[..., 2]
    out[..., lay["key_pos"]] = kp.flatten(-2)
    return out


def channels(nt: int, n_key: int = 5) -> int:
    return 3 + 6 + nt + 3 * n_key + 6


# --- the network (TinyMDM's DiT) --------------------------------------------------------------------------

def _timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """diffusers ``Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)``."""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freq[None]
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


class _Block(nn.Module):
    """StableMotionDiTBlock: adaLN-modulated self-attention and a SwiGLU feed-forward, each gated."""

    def __init__(self, dim: int, heads: int, qk_norm: bool = False) -> None:
        super().__init__()
        self.heads = heads
        # RMS-normalised queries and keys bound the attention logits: without it the full-pose planner's adaLN shift
        # grew 1 -> 37 and its logits 371 -> 347,000 within 14k iterations, and training broke (2026-11-22)
        self.qn = nn.RMSNorm(dim // heads) if qk_norm else None
        self.kn = nn.RMSNorm(dim // heads) if qk_norm else None
        self.n1 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        self.ff_in = nn.Linear(dim, 8 * dim)
        self.ff_out = nn.Linear(4 * dim, dim)
        self.table = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, t6: torch.Tensor) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2 = (self.table + t6).unbind(1)
        h = self.n1(x) * (1 + sc1[:, None]) + sh1[:, None]
        B, S, D = h.shape
        q, k, v = self.qkv(h).view(B, S, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if self.qn is not None:
            q, k = self.qn(q), self.kn(k)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, D)
        x = x + g1[:, None] * self.out(a)
        h = self.n2(x) * (1 + sc2[:, None]) + sh2[:, None]
        u, g = self.ff_in(h).chunk(2, -1)
        return x + g2[:, None] * self.ff_out(u * F.silu(g))


class DiT(nn.Module):
    def __init__(self, channels: int, layers: int = 2, heads: int = 4, head_dim: int = 64, max_len: int = 32,
                 n_cond: int = 0) -> None:
        super().__init__()
        D = heads * head_dim
        self.D = D
        self.pre = nn.Linear(channels, channels, bias=False)          # MimicKit's residual 1x1 conv
        self.proj_in = nn.Linear(channels, D, bias=False)
        self.t_mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        if n_cond:                                                    # labels -> added to the timestep embedding
            self.c_emb = nn.Linear(n_cond, D)
            self.c_null = nn.Parameter(torch.randn(D) * 0.02)
        self.blocks = nn.ModuleList(_Block(D, heads) for _ in range(layers))
        self.proj_out = nn.Linear(D, channels, bias=False)
        self.post = nn.Linear(channels, channels, bias=False)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))
        pe = torch.zeros(1, max_len, D)
        pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None,
                drop: torch.Tensor | None = None) -> torch.Tensor:
        """``c`` ``[B, n_cond]`` labels (None = the null label); ``drop`` ``[B]`` bool: these rows take the null label."""
        x = x + self.pre(x)
        h = self.proj_in(x) + self.pe[:, : x.shape[1]]
        temb = self.t_mlp(_timestep_embedding(t))
        if hasattr(self, "c_emb"):
            null = self.c_null.expand_as(temb)
            e = null if c is None else self.c_emb(c)
            temb = temb + (e if drop is None else torch.where(drop[:, None], null, e))
        t6 = self.ada(temb).view(-1, 6, self.D)
        for b in self.blocks:
            h = b(h, t6)
        y = self.proj_out(h)
        return y + self.post(y)


def _alphas_cumprod(T: int) -> torch.Tensor:
    """diffusers' ``squaredcos_cap_v2`` schedule."""
    f = lambda s: math.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    betas = torch.tensor([min(1 - f((i + 1) / T) / f(i / T), 0.999) for i in range(T)], dtype=torch.float64)
    return torch.cumprod(1 - betas, 0).float()


class Prior(nn.Module):
    """The denoiser, its EMA copy (what scores), the schedule and the feature normaliser."""

    def __init__(self, nt: int, dt: float, window: int = W, T: int = T_DIFF, layers: int = 2, heads: int = 4,
                 head_dim: int = 64, n_cond: int = 0) -> None:
        super().__init__()
        self.cfg = dict(nt=nt, dt=dt, window=window, T=T, layers=layers, heads=heads, head_dim=head_dim, n_cond=n_cond)
        c = channels(nt)
        self.net = DiT(c, layers, heads, head_dim, n_cond=n_cond)
        self.ema = torch.optim.swa_utils.AveragedModel(self.net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995))
        self.register_buffer("ab", _alphas_cumprod(T))
        self.register_buffer("mean", torch.zeros(c))
        self.register_buffer("std", torch.ones(c))

    def set_normaliser(self, x: torch.Tensor, min_std: float = MIN_STD) -> None:
        x = x.reshape(-1, x.shape[-1])
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(min_std))

    def features(self, rec: torch.Tensor) -> torch.Tensor:
        return (window_features(rec, self.cfg["nt"], self.cfg["dt"]) - self.mean) / self.std

    def noised(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(x)
        a = self.ab[t][:, None, None]
        return a.sqrt() * x + (1 - a).sqrt() * eps, eps

    def labels(self, D: dict[str, Any], clips: Any) -> torch.Tensor | None:
        """The labels of clips ``clips`` of ``D`` for a conditional prior; None for an unconditional one."""
        return torch.as_tensor(conditions(D, clips), device=self.ab.device) if self.cfg.get("n_cond") else None

    def loss(self, x: torch.Tensor, c: torch.Tensor | None = None, drop: float = LABEL_DROP) -> torch.Tensor:
        """Training loss on normalised windows: L1 between predicted and true noise, uniform diffusion step; with
        labels ``c``, a ``drop`` share of them replaced by the null label."""
        t = torch.randint(0, self.cfg["T"], (x.shape[0],), device=x.device)
        xt, eps = self.noised(x, t)
        d = None if c is None else torch.rand(x.shape[0], device=x.device) < drop
        return F.l1_loss(self.net(xt, t, c, d), eps)

    @torch.no_grad()
    def sds(self, x: torch.Tensor, steps: tuple[int, ...] = SDS_STEPS, c: torch.Tensor | None = None) -> torch.Tensor:
        """ESM-SDS loss ``[B, len(steps)]`` of normalised windows, with the EMA weights (labels ``c``: conditional)."""
        out = []
        for k in steps:
            t = torch.full((x.shape[0],), k, dtype=torch.long, device=x.device)
            xt, eps = self.noised(x, t)
            out.append((self.ema(xt, t, c) - eps).pow(2).mean((1, 2)))
        return torch.stack(out, 1)

    def save(self, path: str | Path, **extra: Any) -> None:
        torch.save(dict(cfg=self.cfg, state=self.state_dict(), **extra), path)

    @classmethod
    def load(cls, path: str | Path, device: Any = "cpu") -> Prior:
        ck = torch.load(path, map_location=device, weights_only=False)
        p = cls(**ck["cfg"]).to(device)
        p.load_state_dict(ck["state"])
        return p.eval().requires_grad_(False)


class StyleReward:
    """The SMP reward of a batch of style-record windows: ``exp(-scale * mean_k(loss_k / mean|loss_k|))``."""

    def __init__(self, prior: Prior, scale: float = SDS_SCALE, steps: tuple[int, ...] = SDS_STEPS,
                 norm_samples: float = 1e8) -> None:
        self.prior, self.scale, self.steps, self.norm_samples = prior, scale, steps, norm_samples
        dev = prior.ab.device
        self.mean_abs = torch.ones(len(steps), device=dev)
        self.count = 0

    @torch.no_grad()
    def __call__(self, rec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (reward ``[B]``, mean SDS loss ``[B]``). A window with a non-finite record (a diverged world, which SPRINT
        resets after the reward and pays nothing) scores 0 and stays out of the normaliser: one NaN there made every
        later reward NaN (the first planner run, 2026-11-15)."""
        loss = self.prior.sds(self.prior.features(rec), self.steps)
        ok = torch.isfinite(loss).all(1)
        loss = torch.where(ok[:, None], loss, torch.zeros_like(loss))
        r = torch.where(ok, torch.exp(-self.scale * (loss / self.mean_abs.clamp_min(1e-4)).mean(1)), torch.zeros_like(ok, dtype=loss.dtype))
        if self.count < self.norm_samples:                       # after scoring, as MimicKit (it updates per iteration)
            self.mean_abs = (self.count * self.mean_abs + loss.abs().sum(0)) / (self.count + ok.sum())
            self.count += loss.shape[0]                          # host-side count (no sync); diverged rows are rare
        return r, loss.mean(1)

    def state_dict(self) -> dict[str, Any]:
        return dict(mean_abs=self.mean_abs.cpu(), count=self.count)

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.mean_abs, self.count = d["mean_abs"].to(self.mean_abs.device), int(d["count"])
