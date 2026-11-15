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


# --- heightfields ------------------------------------------------------------------------------------------------
# SPRINT's shared heightfield (Data.set_heightfield): heights[j, i] at map (x0 + i*cell, z0 + j*cell), an optional block
# top per cell (NaN = triangulated), and world w reading map point (x, z) + terrain_offset[w]. Clips whose ground is not a
# profile (turns, platforms, gaps) stand on their own tile of one such map.

def hf_height(heights: Any, block: Any, cell: float, origin: Any, x: Any, z: Any) -> Any:
    """Surface height at map ``(x, z)`` (any matching shapes; numpy or torch), the surface SPRINT queries and collides
    against: a block cell's top where ``block`` is finite, else the cell's triangle (split along (i, j) -> (i+1, j+1),
    ``u >= v`` the lower-right one); beyond the grid the border extends."""
    tch = not isinstance(x, np.ndarray)
    if tch:
        import torch
        floor, clip, where, finite = torch.floor, torch.clamp, torch.where, torch.isfinite
    else:
        floor, clip, where, finite = np.floor, np.clip, np.where, np.isfinite
    nz, nx = heights.shape
    fx, fz = (x - origin[0]) / cell, (z - origin[1]) / cell
    i, j = clip(floor(fx), 0, nx - 2), clip(floor(fz), 0, nz - 2)
    u, v = clip(fx - i, 0.0, 1.0), clip(fz - j, 0.0, 1.0)
    i, j = (i.long(), j.long()) if tch else (i.astype(int), j.astype(int))
    h00, h10, h01, h11 = heights[j, i], heights[j, i + 1], heights[j + 1, i], heights[j + 1, i + 1]
    tri = where(u >= v, h00 + u * (h10 - h00) + v * (h11 - h10), h00 + v * (h01 - h00) + u * (h11 - h01))
    if block is None:
        return tri
    b = block[j, i]
    return where(finite(b), b, tri)


class Ground:
    """Terrain height ``[N, S]`` under ``(x, z)`` ``[N, S]`` for clips ``clip`` ``[N]`` (torch): each clip on its
    profile (padded ``X``, ``Y``: :func:`pad`) or, where ``on_hf``, on its tile of heightfield ``hf`` (:func:`atlas`)
    at ``offset``. :meth:`of` reads a loaded dataset (gp.data.load, gp.free.merge)."""

    def __init__(self, X: Any, Y: Any, on_hf: Any, hf: dict[str, Any] | None = None, offset: Any = None,
                 device: Any = "cpu") -> None:
        import torch
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
        self.X, self.Y = T(X), T(Y)
        self.on_hf = torch.as_tensor(np.asarray(on_hf, bool), device=device)
        self.hf: dict[str, Any] | None = None
        if bool(self.on_hf.any()):
            assert hf is not None and offset is not None
            self.hf = dict(heights=T(hf["heights"]), block=T(hf["block"]), cell=float(hf["cell"]),
                           origin=tuple(float(v) for v in hf["origin"]))
            self.off = T(offset)

    @classmethod
    def of(cls, D: dict[str, Any], device: Any = "cpu") -> Ground:
        kind = np.asarray(D.get("terrain_kind", np.zeros(len(D["name"]), int)))
        return cls(D["profile_x"], D["profile_y"], kind == 1, D.get("hf"), D.get("hf_offset"), device)

    def __call__(self, clip: Any, x: Any, z: Any) -> Any:
        import torch
        h = height(self.X, self.Y, clip, x)
        if self.hf is None:
            return h
        f, off = self.hf, self.off[clip]
        hh = hf_height(f["heights"], f["block"], f["cell"], f["origin"], x + off[:, :1], z + off[:, 1:2])
        return torch.where(self.on_hf[clip][:, None], hh, h)


def atlas(tiles: list[dict[str, Any]], cell: float, gap: int = 4) -> dict[str, Any]:
    """Lay heightfield tiles side by side in one map. Each tile: ``heights`` ``[nz, nx]``, optional ``block``
    ``[nz-1, nx-1]`` (NaN = triangulated), ``origin`` (its own (x, z) of heights[0, 0]), all at ``cell``. Tiles go in rows
    of about sqrt(n), ``gap`` cells apart, each padded into the gap with its own edge heights. Returns ``heights``,
    ``block``, ``origin``, ``cell`` for ``Data.set_heightfield`` and ``offsets`` ``[n, 2]``: a world on tile k reads map
    (x, z) + offsets[k], i.e. it stands on the tile at the tile's own coordinates."""
    n = len(tiles)
    cols = max(1, int(np.ceil(np.sqrt(n))))
    rows = [list(range(r * cols, min(n, (r + 1) * cols))) for r in range(int(np.ceil(n / cols)))]
    row_h = [max(tiles[k]["heights"].shape[0] for k in r) for r in rows]
    width = max(sum(tiles[k]["heights"].shape[1] + gap for k in r) for r in rows) + gap
    H = np.zeros((sum(h + gap for h in row_h) + gap, width))
    B = np.full((H.shape[0] - 1, H.shape[1] - 1), np.nan)
    offsets = np.zeros((n, 2))
    g, j0 = gap // 2, gap
    for r, row in enumerate(rows):
        i0 = gap
        for k in row:
            t = tiles[k]
            nz, nx = t["heights"].shape
            H[j0 - g:j0 + nz + g, i0 - g:i0 + nx + g] = np.pad(t["heights"], g, mode="edge")
            if t.get("block") is not None:
                B[j0:j0 + nz - 1, i0:i0 + nx - 1] = t["block"]
            offsets[k] = (i0 * cell - t["origin"][0], j0 * cell - t["origin"][1])
            i0 += nx + gap
        j0 += row_h[r] + gap
    return dict(heights=H, block=B, origin=(0.0, 0.0), offsets=offsets, cell=cell)


def tile_from_profile(P: np.ndarray, x_range: tuple[float, float], z_range: tuple[float, float],
                      cell: float) -> dict[str, Any]:
    """A heightfield tile of profile ``P`` extruded along z over the given ranges: block cells at the profile's height at
    each cell's centre (vertical walls where it steps), so a stair stays a stair. For tests and for profile-only clips
    that share a map with heightfield ones."""
    xs = np.arange(x_range[0], x_range[1] + cell / 2, cell)
    zs = np.arange(z_range[0], z_range[1] + cell / 2, cell)
    X1, Y1 = pad([P])
    hx = height(X1, Y1, np.zeros(1, int), xs[None])[0]
    hc = height(X1, Y1, np.zeros(1, int), (xs[:-1] + cell / 2)[None])[0]
    return dict(heights=np.tile(hx, (len(zs), 1)), block=np.tile(hc, (len(zs) - 1, 1)), origin=(xs[0], zs[0]))


def tile_from_boxes(hf: np.ndarray, hf_min: Any, dx: float) -> dict[str, Any]:
    """A heightfield tile of a box grid as the seed library exports PARC terrain: canonical z-up cell ``hf[i, j]`` sits at
    ``hf_min + (i, j) * dx`` and is flat over +-dx/2 (nearest-cell lookup), and the model frame has x = canonical x,
    z = -canonical y. Every cell is a block, so walls stay vertical; vertices take their highest neighbour."""
    hf = np.asarray(hf, np.float64)
    B = hf[:, ::-1].T                                     # [nz, nx]: rows along model z (ascending), columns along x
    P = np.pad(B, 1, mode="edge")
    H = np.maximum.reduce([P[:-1, :-1], P[1:, :-1], P[:-1, 1:], P[1:, 1:]])
    origin = (float(hf_min[0]) - dx / 2, -(float(hf_min[1]) + (hf.shape[1] - 0.5) * dx))
    return dict(heights=H, block=B, origin=origin, cell=float(dx))
