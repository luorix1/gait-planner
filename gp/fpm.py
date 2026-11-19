"""The foothold prediction model of Heyrman et al. 2026 ("Watch Your Step", arXiv 2610.29612), ported from the authors'
code (github.com/leggedrobotics/foothold-prediction-model, fpm/foot_pred/{model,labels,trainer}.py; MIT licence,
Copyright (c) 2026 ETH Zurich) without changes to what it computes. Only the frame convention differs: SPRINT is y-up,
so a foothold's planar position is its (forward, lateral) = (x, z) in the root frame, where the authors' z-up code
takes (x, y).

- :class:`FootPredictor`: MLP, hidden (128, 128), each layer Linear -> LayerNorm -> LeakyReLU(0.1), orthogonal weights,
  zero biases; :class:`NormalizedFootPredictor` stores the input mean and std of the fit's training split.
- :func:`foothold_labels`: per foot, the next landing strictly in the future (in contact: the landing after the coming
  lift-off; in the air: the landing it is about to make), in the root frame at each step. A window whose label would
  lie past a reset (the root jumps over 0.5 m in one step) takes the post-reset foot position, as the authors' code does.
- :func:`fit`: Huber (delta 0.01), Adam 5e-3, batch 1024, 25 epochs, random 80/20 split, from scratch every time.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

HIDDEN = (128, 128)
LR, BATCH, EPOCHS, DELTA = 5e-3, 1024, 25, 0.01


class FootPredictor(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_dims: tuple[int, ...] = HIDDEN) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = input_size
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.LayerNorm(h), nn.LeakyReLU(0.1)]
            last = h
        layers.append(nn.Linear(last, output_size))
        self.net = nn.Sequential(*layers)
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class NormalizedFootPredictor(nn.Module):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("in_mean", mean.reshape(-1))
        self.register_buffer("in_std", std.reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.in_mean) / self.in_std)


def _quat_inv_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the inverse of unit quaternions ``q`` (x, y, z, w)."""
    u, w = -q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def future_contacts(contact: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor, foot_pos: torch.Tensor,
                    reset: torch.Tensor) -> torch.Tensor:
    """Per step and foot, the position of its next strictly-future landing in the root frame.

    ``contact`` ``[E, T, F]`` (bool or 0/1), ``root_pos`` ``[E, T, 3]``, ``root_quat`` ``[E, T, 4]`` (x, y, z, w),
    ``foot_pos`` ``[E, T, F, 3]`` world, ``reset`` ``[E, T]`` (a reset happens between step t and t+1)."""
    E, T, F = contact.shape
    out = torch.zeros(E, T, F, 3, device=contact.device)
    is_reset = reset > 0.5
    for i in range(F):
        pos = foot_pos[:, :, i]
        latch_next, latch_future = pos[:, -1].clone(), pos[:, -1].clone()
        was_in_air = torch.zeros(E, dtype=torch.bool, device=contact.device)
        for t in range(T - 1, -1, -1):
            r = is_reset[:, t]
            latch_next[r], latch_future[r], was_in_air[r] = pos[r, t], pos[r, t], False
            down = contact[:, t, i] > 0.5
            lift = down & was_in_air
            latch_future[lift] = latch_next[lift]
            latch_next[down] = pos[down, t]
            out[:, t, i] = torch.where(down[:, None], latch_future, latch_next)
            was_in_air = ~down
    return _quat_inv_rotate(root_quat[:, :, None], out - root_pos[:, :, None])


def next_landings(contact: torch.Tensor, foot_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One continuous recording (a clip): ``contact`` ``[T, F]``, ``foot_pos`` ``[T, F, 3]`` world -> each step's next
    strictly-future landing per foot ``[T, F, 3]`` (the same definition as :func:`future_contacts`: in contact, the
    landing after the coming lift-off; in the air, the landing it is about to make) and ``valid`` ``[T, F]``: False
    where that landing lies past the end of the recording (``future_contacts`` labels those with the last position)."""
    T, F = contact.shape
    out = torch.zeros(T, F, 3, dtype=foot_pos.dtype)
    valid = torch.zeros(T, F, dtype=torch.bool)
    for i in range(F):
        nxt = fut = None
        in_air_after = False
        for t in range(T - 1, -1, -1):
            down = bool(contact[t, i])
            if down and in_air_after:               # the last stance step before a lift-off
                fut = nxt
            if down:
                nxt = foot_pos[t, i]
            lab = fut if down else nxt
            if lab is not None:
                out[t, i], valid[t, i] = lab, True
            in_air_after = not down
    return out, valid


def detect_resets(root_pos: torch.Tensor, jump: float = 0.5) -> torch.Tensor:
    """``[E, T]``: 1 where the root's horizontal position jumps more than ``jump`` m to the next step."""
    xz = root_pos[..., [0, 2]]
    r = torch.zeros(xz.shape[:2], device=xz.device)
    r[:, :-1] = ((xz[:, 1:] - xz[:, :-1]).norm(dim=-1) > jump).float()
    return r


def foothold_labels(contact: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor,
                    foot_pos: torch.Tensor) -> torch.Tensor:
    """``[E, T, 2F]``: each foot's next landing, (forward, lateral) in the root frame. The authors drop the first
    recorded step (its contact flags predate the step); callers pass histories already aligned."""
    p = future_contacts(contact, root_pos, root_quat, foot_pos, detect_resets(root_pos))
    return p[..., [0, 2]].flatten(2)


def fit(x: torch.Tensor, y: torch.Tensor, device: Any, seed: int = 0, log: Any = print) -> NormalizedFootPredictor:
    """A fresh predictor fitted to ``x`` ``[S, D]`` -> ``y`` ``[S, 2F]`` (the authors' ``fit_foot_predictor``). Safe to
    call from inside a rollout's ``no_grad``/``inference_mode``, as the authors' is."""
    with torch.inference_mode(False), torch.enable_grad():
        return _fit(x.clone(), y.clone(), device, seed, log)


def _fit(x: torch.Tensor, y: torch.Tensor, device: Any, seed: int, log: Any) -> NormalizedFootPredictor:
    g = torch.Generator().manual_seed(seed)
    x, y = x.float().cpu(), y.float().cpu()
    keep = torch.isfinite(x).all(1) & torch.isfinite(y).all(1)
    x, y = x[keep], y[keep]
    idx = torch.randperm(len(x), generator=g)
    n_val = int(0.2 * len(x))
    va, tr = idx[:n_val], idx[n_val:]
    mean, std = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    core = FootPredictor(x.shape[1], y.shape[1])
    model = NormalizedFootPredictor(core, mean, std).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.HuberLoss(delta=DELTA)
    xt, yt, xv, yv = x[tr].to(device), y[tr].to(device), x[va].to(device), y[va].to(device)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xt), generator=g).to(device)
        tl = 0.0
        for k in range(0, len(xt), BATCH):
            b = perm[k:k + BATCH]
            loss = crit(model(xt[b]), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            vl = float(sum(crit(model(xv[k:k + BATCH]), yv[k:k + BATCH]) * len(xv[k:k + BATCH])
                           for k in range(0, len(xv), BATCH))) / max(len(xv), 1)
        if epoch in (0, EPOCHS - 1):
            log(f"[FPM] epoch {epoch + 1}/{EPOCHS}  train {tl / len(xt):.5f}  val {vl:.5f}")
    return model.eval().requires_grad_(False)
