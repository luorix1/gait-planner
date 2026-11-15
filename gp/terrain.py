"""Seed-library terrain -> SPRINT terrain profiles, and height lookups on them (numpy for data, torch for tasks).

A profile is an (x, y) polyline in the model's frame (OpenSim: x forward, y up), extruded along z, as
``sprint.terrain.pack`` takes it: its first and last segments are rays. The seed library turned every ramp, stair and
sloped floor so its gradient runs along +x, so one profile per clip is exact.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def profile(T: dict[str, Any]) -> np.ndarray | None:
    """A clip's terrain description (seed library ``terrain`` JSON, canonical units) -> SPRINT knots ``[K, 2]``,
    scaled to the model; None for heightfield-only terrain (PARC obstacles)."""
    s = T["scale"]
    if T["type"] == "plane":
        if abs(T.get("b") or 0.0) > 1e-9:
            raise ValueError("plane slopes sideways; the seed library should have turned it onto +x")
        P = [[-50.0, -50 * T["a"] + T["c"] * s], [50.0, 50 * T["a"] + T["c"] * s]]
    elif T["type"] == "ramp":
        m, s0, z0 = T["slope"], T["s0"], T["z0"]
        b, zt = T.get("intercept", z0 - m * s0), T.get("z_top", z0 + m * (T["s1"] - s0))
        st = (zt - b) / m
        P = [[s0 - 50, z0], [s0, z0]] + ([[s0, m * s0 + b]] if m * s0 + b > z0 + 1e-6 else []) + [[st, zt], [st + 50, zt]]
        P = [[x * s, z * s] for x, z in P]
    elif T["type"] == "stairs":
        e, L = T["edges"], T["levels"]
        P = [[e[0] - 50, L[0]]]
        for k, x in enumerate(e):
            P += [[x, L[k]], [x, L[k + 1]]]
        P += [[e[-1] + 50, L[-1]]]
        P = [[x * s, z * s] for x, z in P]
    else:
        return None
    P = np.array(P, np.float64)
    u = T.get("u", [1.0, 0.0])
    if abs(u[1]) > 1e-6:
        raise ValueError(f"terrain runs along {u}, not x")
    return P[::-1] * [-1, 1] if u[0] < 0 else P


def pad(profiles: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Profiles -> ``(X, Y)`` ``[P, K]``, each padded by continuing its last segment, so a batched lookup sees the
    same end rays SPRINT does."""
    K = max(len(p) for p in profiles)
    X, Y = np.zeros((len(profiles), K)), np.zeros((len(profiles), K))
    for i, p in enumerate(profiles):
        n = len(p)
        d = (p[-1] - p[-2]) / np.linalg.norm(p[-1] - p[-2])
        ext = p[-1] + np.arange(1, K - n + 1)[:, None] * 1e3 * d
        X[i], Y[i] = np.r_[p[:, 0], ext[:, 0]], np.r_[p[:, 1], ext[:, 1]]
    return X, Y


def height(X: Any, Y: Any, pid: Any, x: Any) -> Any:
    """Surface height under ``x`` ``[N, S]`` on profile ``pid[n]`` of the padded ``(X, Y)``; numpy or torch."""
    if isinstance(x, np.ndarray):
        xs, ys = X[pid], Y[pid]
        i = np.clip(np.stack([np.searchsorted(r, v, side="right") for r, v in zip(xs, x, strict=True)]), 1, xs.shape[1] - 1)
        g = lambda a, j: np.take_along_axis(a, j, 1)  # noqa: E731
    else:
        import torch
        xs, ys = X[pid], Y[pid]
        i = torch.searchsorted(xs.contiguous(), x.contiguous(), right=True).clamp(1, xs.shape[1] - 1)
        g = lambda a, j: a.gather(1, j)  # noqa: E731
    x0, x1, y0, y1 = g(xs, i - 1), g(xs, i), g(ys, i - 1), g(ys, i)
    dx = x1 - x0
    t = (x - x0) / (dx + (dx == 0))
    return y0 + t * (y1 - y0)
