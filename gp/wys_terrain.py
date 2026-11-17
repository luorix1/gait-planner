"""The training terrain of Heyrman et al. 2026 (Watch Your Step; their ``TRAINING_TERRAINS_CFG``) as one SPRINT
heightfield: 10 levels x 20 columns of 8 x 8 m tiles with a 20 m flat border, columns shared by type in the paper's
proportions, difficulty (row + U(0, 1)) / 10 per tile, each tile seeded as their ``SeededTerrainGenerator`` seeds it.

Map axes: the authors' (x, y) on the ground are SPRINT's (x, z); heights are y. A tile's (row, col) sits at map
x = border + row * 8, z = border + col * 8. Worlds stand on a tile through ``Data.terrain_offset`` = the tile's centre,
so a body near the world origin is at the tile's centre.

Each type follows the generating function (sources: the authors' fpm/terrains/{mesh,hf}_terrains.py and IsaacLab's
trimesh/height_field terrains):
- pyramid stairs / inverted: rings of width U(0.25, 0.30) m inside a 1 m border, each a step higher (lower) toward a
  2 m central platform; step height 0.10 -> 0.25 m with difficulty.
- boxes (random grid): 0.45 m cells with flat tops at U(-h, h), h 0.05 -> 0.30 m; a 2 m central platform at +h; the
  0.35 m left over at the edge is border at 0.
- random rough: heights drawn from {0.02, 0.04, ..., 0.10} m on a 0.1 m grid inside a 0.25 m border (difficulty is
  ignored, as in IsaacLab).
- pyramid slope / inverted: height = slope * 4 m * xx * yy (xx, yy rising linearly to 1 at the centre), clipped at the
  2 m platform's corner height; slope 0 -> 0.55.
- flat.
- stepping stones: square stones 0.5 -> 0.3 m wide, 0.1 -> 0.2 m apart, tops within +-(-0.05 -> 0.05) m, holes
  0.3 -> 3.0 m deep, a 2 m platform at the centre (the authors' 0.1 m grid, upsampled).
- pit: a 3.5 m centre square sunk (or, half the time, raised) by 0.1 -> 0.4 (0.1 -> 0.6) m, then rings of 0.6 m
  ground and gaps 0.1 -> 0.4 m wide, 1 m deep, inside a 0.25 m border.

Stepped tiles are SPRINT block cells (flat tops, vertical walls); slopes and rough ground are triangulated.
Heights are quantised to the authors' 5 mm vertical step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

CELL = 0.05                  # m, the heightfield cell (stepping stones and pits are built on the authors' 0.1 m grid)
TILE = 8.0
ROWS, COLS = 10, 20
BORDER = 20.0
VSTEP = 0.005                # m, IsaacLab's vertical_scale
SEED = 0
#: (name, proportion), in the authors' order
TYPES = (("pyramid_stairs", 0.15), ("pyramid_stairs_inv", 0.15), ("boxes", 0.15), ("random_rough", 0.05),
         ("hf_pyramid_slope", 0.075), ("hf_pyramid_slope_inv", 0.05), ("flat", 0.025), ("stepping_stones", 0.3),
         ("pit", 0.05))
N_PX = int(round(TILE / CELL))           # cells per tile side


def column_types() -> list[str]:
    """The type of each of the 20 columns, as IsaacLab assigns them from the proportions."""
    p = np.array([q for _, q in TYPES], np.float64)
    p /= p.sum()
    c = np.cumsum(p)
    return [TYPES[int(np.min(np.where(k / COLS + 0.001 < c)[0]))][0] for k in range(COLS)]


def _q(h: np.ndarray) -> np.ndarray:
    return np.rint(h / VSTEP) * VSTEP


def _centres(n: int = N_PX) -> np.ndarray:
    return (np.arange(n) + 0.5) * CELL


# -- tile functions: (difficulty, rng) -> (cells [nx, ny] or None, nodes [nx+1, ny+1] or None), authors' (x, y) order


def pyramid_stairs(d: float, rng: np.random.Generator, inverted: bool) -> tuple[np.ndarray, None]:
    h = 0.10 + d * (0.25 - 0.10)
    w = rng.uniform(0.25, 0.30)
    border, platform = 1.0, 2.0
    n = int((TILE - 2 * border - platform) // (2 * w) + 1)
    x = _centres()
    e = np.minimum(x - border, TILE - border - x)                       # inset along one axis
    inset = np.minimum(e[:, None], e[None, :])
    k = np.floor(inset / w)
    top = np.where(k < n, k + 1, n + 1) * h
    top = np.where(inset < 0, 0.0, -top if inverted else top)
    return _q(top), None


def boxes(d: float, rng: np.random.Generator) -> tuple[np.ndarray, None]:
    gw, platform = 0.45, 2.0
    gh = 0.05 + d * (0.30 - 0.05)
    nb = int(TILE / gw)
    border = TILE - nb * gw
    noise = rng.uniform(-gh, gh, (nb, nb))
    x = _centres()
    idx = np.floor((x - border / 2) / gw).astype(int)
    inside = (idx >= 0) & (idx < nb)
    top = np.zeros((N_PX, N_PX))
    ii = np.clip(idx, 0, nb - 1)
    top = np.where(inside[:, None] & inside[None, :], noise[ii[:, None], ii[None, :]], 0.0)
    c = (np.abs(x - TILE / 2) < platform / 2)
    top[np.ix_(c, c)] = gh
    return top, None


def random_rough(d: float, rng: np.random.Generator) -> tuple[None, np.ndarray]:
    border = 0.25
    vals = np.arange(0.02, 0.10 + 1e-9, 0.02)
    m = int(round((TILE - 2 * border) / 0.1)) + 1
    coarse = rng.choice(vals, size=(m, m))
    xs = np.arange(N_PX + 1) * CELL
    u = np.clip((xs - border) / 0.1, 0, m - 1)
    i0 = np.clip(np.floor(u).astype(int), 0, m - 2)
    f = u - i0
    rows = coarse[i0] * (1 - f)[:, None] + coarse[i0 + 1] * f[:, None]
    nodes = rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :]
    inside = (xs >= border) & (xs <= TILE - border)
    return None, _q(np.where(inside[:, None] & inside[None, :], nodes, 0.0))


def pyramid_slope(d: float, rng: np.random.Generator, inverted: bool) -> tuple[None, np.ndarray]:
    border, platform = 0.25, 2.0
    slope = (0.0 + d * 0.55) * (-1 if inverted else 1)
    inner = TILE - 2 * border
    xs = np.arange(N_PX + 1) * CELL
    t = np.clip((xs - border) / inner, 0, 1)
    rise = 1 - np.abs(2 * t - 1)                                      # 0 at the inner edge, 1 at the centre
    hmax = slope * inner / 2
    raw = hmax * rise[:, None] * rise[None, :]
    corner = 1 - (platform / 2) / (inner / 2)
    zpf = hmax * corner * corner
    raw = np.clip(raw, min(0, zpf), max(0, zpf))
    inside = (xs >= border) & (xs <= TILE - border)
    return None, _q(np.where(inside[:, None] & inside[None, :], raw, 0.0))


def flat(d: float, rng: np.random.Generator) -> tuple[np.ndarray, None]:
    return np.zeros((N_PX, N_PX)), None


def stepping_stones(d: float, rng: np.random.Generator) -> tuple[np.ndarray, None]:
    """The authors' ``stepping_stones_terrain`` on their 0.1 m grid (np.random, seeded per tile), upsampled 2x."""
    hs = 0.1
    lerp = lambda r: r[0] + d * (r[1] - r[0])  # noqa: E731
    n = int(TILE / hs)
    stone, gap = int(lerp((0.5, 0.3)) / hs), int(lerp((0.1, 0.2)) / hs)
    max_h, holes = int(lerp((-0.05, 0.05)) / VSTEP), int(lerp((-0.3, -3.0)) / VSTEP)
    heights = np.arange(-max_h, max_h + 1) if max_h > 0 else np.array([0])
    hf = np.full((n, n), holes)
    start_y = 0
    while start_y < n:
        stop_y = min(n, start_y + stone)
        start_x = np.random.randint(0, stone)
        stop_x = max(0, start_x - gap)
        if stop_x > 0:
            hf[0:stop_x, start_y:stop_y] = np.random.choice(heights)
        while start_x < n:
            stop_x = min(n, start_x + stone)
            hf[start_x:stop_x, start_y:stop_y] = np.random.choice(heights)
            start_x += stone + gap
        start_y += stone + gap
    pw = int(2.0 / hs)
    hf[(n - pw) // 2:(n + pw) // 2, (n - pw) // 2:(n + pw) // 2] = 0
    return np.kron(hf * VSTEP, np.ones((2, 2))), None


def pit(d: float, rng: np.random.Generator) -> tuple[np.ndarray, None]:
    """The authors' ``pit_gap_terrain`` (0.1 m grid inside a 0.25 m border), upsampled 2x."""
    hs, border = 0.1, 0.25
    sign = -1 if random.randint(0, 1) == 0 else 1
    rng_off = (0.1, 0.4) if sign < 0 else (0.1, 0.6)
    depth = rng_off[0] + d * (rng_off[1] - rng_off[0])
    gap_w = 0.1 + d * (0.4 - 0.1)
    inner = int(round((TILE - 2 * border) / hs))
    depth_px, gap_depth_px = int(abs(depth) / VSTEP), int(1.0 / VSTEP)
    gap_px, ground_px, half_pit = max(int(gap_w / hs), 0), max(int(0.6 / hs), 1), max(int(0.5 * 3.5 / hs), 1)
    dx = np.abs(np.arange(inner) - inner // 2)
    cheb = np.maximum(dx[:, None], dx[None, :])
    hf = np.zeros((inner, inner))
    if gap_px > 0:
        dd = cheb - half_pit
        hf[(dd >= 0) & (np.mod(dd, ground_px + gap_px) >= ground_px)] = -gap_depth_px
    hf[cheb < half_pit] = sign * depth_px
    full = np.zeros((int(TILE / hs), int(TILE / hs)))
    b = int(round(border / hs))
    full[b:b + inner, b:b + inner] = hf
    return np.kron(full * VSTEP, np.ones((2, 2))), None


FUNCS = {
    "pyramid_stairs": lambda d, r: pyramid_stairs(d, r, False),
    "pyramid_stairs_inv": lambda d, r: pyramid_stairs(d, r, True),
    "boxes": boxes, "random_rough": random_rough,
    "hf_pyramid_slope": lambda d, r: pyramid_slope(d, r, False),
    "hf_pyramid_slope_inv": lambda d, r: pyramid_slope(d, r, True),
    "flat": flat, "stepping_stones": stepping_stones, "pit": pit,
}


@dataclass
class Terrain:
    heights: np.ndarray          # [nz + 1, nx + 1] node heights (SPRINT: heights[j, i] at (x0 + i*cell, z0 + j*cell))
    block: np.ndarray            # [nz, nx] flat-top cell heights, NaN = triangulated
    cell: float
    origin: tuple[float, float]
    centres: np.ndarray          # [ROWS, COLS, 2] tile centres (map x, z)
    types: list[str]             # per column
    difficulty: np.ndarray       # [ROWS, COLS]


def build(seed: int = SEED) -> Terrain:
    """The whole map, deterministic in ``seed`` (each tile seeded as the authors' generator seeds it)."""
    types = column_types()
    nb = int(round(BORDER / CELL))
    nx, nz = 2 * nb + ROWS * N_PX, 2 * nb + COLS * N_PX
    heights = np.zeros((nz + 1, nx + 1), np.float32)
    block = np.full((nz, nx), np.nan, np.float32)
    diff = np.zeros((ROWS, COLS))
    top_rng = np.random.default_rng(seed)
    for c in range(COLS):
        for r in range(ROWS):
            d = (r + top_rng.uniform()) / ROWS
            diff[r, c] = d
            tile_seed = (int(seed) * 6364136223846793005 + c * 1000003 + r * 10007) % (2 ** 32)
            np.random.seed(tile_seed)
            random.seed(tile_seed)
            cells, nodes = FUNCS[types[c]](d, np.random.default_rng(tile_seed))
            i0, j0 = nb + r * N_PX, nb + c * N_PX               # authors' x -> map x (i), y -> map z (j)
            if cells is not None:
                block[j0:j0 + N_PX, i0:i0 + N_PX] = cells.T
                pad = np.pad(cells, 1, mode="edge")
                nd = np.maximum.reduce([pad[:-1, :-1], pad[1:, :-1], pad[:-1, 1:], pad[1:, 1:]])
                heights[j0:j0 + N_PX + 1, i0:i0 + N_PX + 1] = nd.T
            else:
                heights[j0:j0 + N_PX + 1, i0:i0 + N_PX + 1] = nodes.T
    xs = BORDER + (np.arange(ROWS) + 0.5) * TILE
    zs = BORDER + (np.arange(COLS) + 0.5) * TILE
    centres = np.stack(np.meshgrid(xs, zs, indexing="ij"), -1)
    return Terrain(heights, block, CELL, (0.0, 0.0), centres, types, diff)


def summary(T: Terrain) -> dict[str, Any]:
    """Per type: columns and the height range of its tiles at the lowest and highest level."""
    out: dict[str, Any] = {}
    nb = int(round(BORDER / CELL))
    for c, t in enumerate(T.types):
        e = out.setdefault(t, {"cols": 0})
        e["cols"] += 1
        for r, key in ((0, "level0"), (ROWS - 1, "level9")):
            j0, i0 = nb + c * N_PX, nb + r * N_PX
            b = T.block[j0:j0 + N_PX, i0:i0 + N_PX]
            h = np.where(np.isfinite(b), b, T.heights[j0:j0 + N_PX, i0:i0 + N_PX])
            e[key] = (round(float(h.min()), 3), round(float(h.max()), 3))
    return out
