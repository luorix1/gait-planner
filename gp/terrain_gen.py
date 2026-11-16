"""Random terrain courses that people could build and walk, as SPRINT terrain profiles.

A course runs along +x as a chain of segments: level runs, ramps and stair flights. The rules:

- it starts and ends level, with room to start and to stop;
- every ramp and every flight is entered and left through a level landing: no ramp straight into a ramp or into
  stairs, no flight straight into another, and no up-to-down reversal without level ground between;
- a ramp or a flight goes one way, with one grade, or one rise and tread, throughout;
- the course stays within ``max_height`` of its start.

Sizes default to the training split's 5th-95th percentiles (data/v1): ramp grades 5-12.5 deg over 4.5-6 m of run,
stair rises 0.10-0.175 m, treads 0.25-0.37 m, flights of 3-8 steps; landings of 1.5-4 m. ``TerrainSpec.held_out()``
samples the held-out range instead (ramps 16-18 deg, rises 0.175-0.19 m), for tests on ground the training never had.

Use: ``env.data.set_terrains([course_profile(sample_course(rng)) for _ in range(n)])``, then each world's
``terrain_id``. Not used by v1 training: each training episode replays a clip on that clip's own terrain, because
the clip's plan belongs to it. These courses are for evaluating on new ground and for the planner's loop (plan doc,
G3), which proposes plans on terrain the data never had.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TerrainSpec:
    grade_deg: tuple[float, float] = (5.0, 12.5)
    ramp_run_m: tuple[float, float] = (4.5, 6.0)
    rise_m: tuple[float, float] = (0.10, 0.175)
    tread_m: tuple[float, float] = (0.25, 0.37)
    steps: tuple[int, int] = (3, 8)
    landing_m: tuple[float, float] = (1.5, 4.0)
    start_m: float = 4.0                    # level run before the first feature (a start, a few strides)
    end_m: float = 4.0                      # level run after the last one (room to stop)
    features: tuple[int, int] = (1, 4)      # ramps and flights per course
    p_ramp: float = 0.5                     # the rest are stair flights
    max_height_m: float = 2.5

    @classmethod
    def held_out(cls) -> TerrainSpec:
        """Ramps and stairs steeper and taller than any training clip (the test_terrain split's range)."""
        return replace(cls(), grade_deg=(16.0, 18.0), rise_m=(0.175, 0.19))


def sample_course(rng: np.random.Generator, spec: TerrainSpec = TerrainSpec()) -> list[dict[str, Any]]:
    """A course: ``[{"kind": "level", "run": m} | {"kind": "ramp", "run": m, "grade_deg": deg, "dir": +-1}
    | {"kind": "stairs", "steps": n, "rise": m, "tread": m, "dir": +-1}, ...]``, level first and last and between
    every two features."""
    segs: list[dict[str, Any]] = [dict(kind="level", run=spec.start_m)]
    h = 0.0
    n = int(rng.integers(spec.features[0], spec.features[1] + 1))
    for k in range(n):
        if rng.random() < spec.p_ramp:
            f: dict[str, Any] = dict(kind="ramp", run=float(rng.uniform(*spec.ramp_run_m)),
                                     grade_deg=float(rng.uniform(*spec.grade_deg)))
            dh = f["run"] * math.tan(math.radians(f["grade_deg"]))
        else:
            f = dict(kind="stairs", steps=int(rng.integers(spec.steps[0], spec.steps[1] + 1)),
                     rise=float(rng.uniform(*spec.rise_m)), tread=float(rng.uniform(*spec.tread_m)))
            dh = f["steps"] * f["rise"]
        ways = [d for d in (1, -1) if abs(h + d * dh) <= spec.max_height_m]
        if not ways:                         # this one would leave the height band: a level run instead
            segs.append(dict(kind="level", run=float(rng.uniform(*spec.landing_m))))
            continue
        f["dir"] = int(rng.choice(ways))
        h += f["dir"] * dh
        segs.append(f)
        segs.append(dict(kind="level", run=spec.end_m if k == n - 1 else float(rng.uniform(*spec.landing_m))))
    if segs[-1]["kind"] != "level" or segs[-1]["run"] < spec.end_m:
        segs.append(dict(kind="level", run=spec.end_m))
    return _merge_level(segs)


def _merge_level(segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in segs:
        if out and s["kind"] == "level" and out[-1]["kind"] == "level":
            out[-1] = dict(kind="level", run=out[-1]["run"] + s["run"])
        else:
            out.append(dict(s))
    return out


def course_profile(segs: list[dict[str, Any]], x0: float = -2.0) -> np.ndarray:
    """SPRINT profile knots ``[K, 2]`` (x forward, y up) of a course starting at ``x0``, height 0. Its end
    segments are level, so the rays SPRINT extends them with stay level."""
    x, y = x0, 0.0
    pts = [(x, y)]
    for s in segs:
        if s["kind"] == "level":
            x += s["run"]
            pts.append((x, y))
        elif s["kind"] == "ramp":
            x += s["run"]
            y += s["dir"] * s["run"] * math.tan(math.radians(s["grade_deg"]))
            pts.append((x, y))
        else:
            for _ in range(s["steps"]):
                y += s["dir"] * s["rise"]
                pts.append((x, y))                     # the riser
                x += s["tread"]
                pts.append((x, y))                     # the tread
    return np.array(pts, np.float64)


def check_course(segs: list[dict[str, Any]], spec: TerrainSpec = TerrainSpec()) -> None:
    """Raise ``ValueError`` where a course breaks the module's rules."""
    if segs[0]["kind"] != "level" or segs[0]["run"] < spec.start_m - 1e-9:
        raise ValueError("a course starts with a level run")
    if segs[-1]["kind"] != "level" or segs[-1]["run"] < spec.end_m - 1e-9:
        raise ValueError("a course ends with a level run")
    h = 0.0
    for a, b in zip(segs[:-1], segs[1:], strict=True):
        if a["kind"] != "level" and b["kind"] != "level":
            raise ValueError(f"{a['kind']} straight into {b['kind']}: a level landing must come between")
    for s in segs[1:-1]:
        if s["kind"] == "level" and s["run"] < spec.landing_m[0] - 1e-9:
            raise ValueError(f"landing of {s['run']:.2f} m, under {spec.landing_m[0]} m")
    for s in segs:
        if s["kind"] == "level":
            continue
        if s["kind"] == "ramp":
            if not spec.grade_deg[0] <= s["grade_deg"] <= spec.grade_deg[1]:
                raise ValueError(f"ramp grade {s['grade_deg']:.1f} deg outside {spec.grade_deg}")
            h += s["dir"] * s["run"] * math.tan(math.radians(s["grade_deg"]))
        else:
            if not spec.rise_m[0] <= s["rise"] <= spec.rise_m[1] or not spec.tread_m[0] <= s["tread"] <= spec.tread_m[1]:
                raise ValueError(f"stairs rise {s['rise']:.3f} / tread {s['tread']:.3f} m outside the spec")
            h += s["dir"] * s["steps"] * s["rise"]
        if abs(h) > spec.max_height_m + 1e-9:
            raise ValueError(f"course leaves +-{spec.max_height_m} m ({h:.2f} m)")


# ---------------------------------------------------------------------------------------------------------------------
# 2D courses on the heightfield: landings with 90 deg turns, flights that turn on a landing, gaps and platforms besides
# ramps and straight flights.
#
# Every part of a course runs along a grid axis (headings 0, 90, 180, 270 deg): a heightfield's block cells are
# axis-aligned squares, so a riser or a wall at any other angle comes out as a sawtooth. The ground is painted as boxes:
# each straight stretch is a rectangle ``causeway_m`` wide with its height profile along it, each turn a square landing
# as wide as the causeway (the walker turns there, as on PARC's stairs), a gap a pit 1 m wider than the causeway on each
# side; elsewhere the ground is flat at the course's lowest landing. Stairs, gaps, platforms and landings are block cells
# (vertical walls), ramps triangulated (smooth slope). A level landing comes between every two features, a landing turn
# counts as one, and a course whose centreline comes back near itself is drawn again.
#
# ``difficulty`` (0-1) sets how far each size may go: a size is drawn between its gentle end and
# gentle + difficulty * (hard - gentle).
#
# The defaults stay inside what the seed data (data/v2 train) holds, for the proof of concept: level ground, landing
# turns (AB01's walks turn round), ramps up to its 10 deg, and gaps one ordinary step clears (AB01's median level step,
# 0.58 m, less a 0.25 m foot). No stairs (5 PARC clips) and no platforms (PARC's are all 0.58 m, jumped onto at a run)
# until data with them arrives; ``CourseSpec.full()`` has them.

KINDS_2D = ("turn", "ramp", "stairs", "turning_stairs", "gap", "platform")
TURN_RADIUS_M = 0.6          # the centreline's arc inside a landing turn (the ground there is the landing's square)


@dataclass(frozen=True)
class CourseSpec:
    """Each pair is (gentle, hard); difficulty 0 keeps every size at its gentle end."""
    grade_deg: tuple[float, float] = (3.0, 10.0)
    ramp_run_m: tuple[float, float] = (2.0, 5.0)
    rise_m: tuple[float, float] = (0.10, 0.175)
    tread_m: tuple[float, float] = (0.37, 0.25)
    steps: tuple[int, int] = (3, 8)
    gap_m: tuple[float, float] = (0.10, 0.35)
    platform_m: tuple[float, float] = (0.10, 0.60)
    platform_top_m: tuple[float, float] = (2.0, 1.2)
    landing_m: tuple[float, float] = (3.0, 2.0)
    features: tuple[int, int] = (1, 3)
    start_m: float = 3.0
    end_m: float = 3.0
    causeway_m: float = 2.4              # width of every stretch, and the side of a landing turn
    pit_m: float = 1.0                   # gap depth
    max_height_m: float = 2.5
    kinds: tuple[str, ...] = ("turn", "ramp", "gap")

    @classmethod
    def full(cls) -> CourseSpec:
        """Every kind, steeper ramps, wider gaps, more features: for when stairs and jump data are in the seed."""
        return replace(cls(), grade_deg=(3.0, 12.5), gap_m=(0.15, 0.8), landing_m=(2.0, 1.5), features=(2, 6),
                       kinds=KINDS_2D)


def _size(rng: np.random.Generator, pair: tuple[float, float], d: float) -> float:
    a, b = pair[0], pair[0] + d * (pair[1] - pair[0])
    return float(rng.uniform(min(a, b), max(a, b)))


def _turn_length(spec: CourseSpec) -> float:
    hw, r = spec.causeway_m / 2, TURN_RADIUS_M
    return 2 * (hw - r) + r * math.pi / 2


def sample_course2d(rng: np.random.Generator, difficulty: float = 0.5, spec: CourseSpec = CourseSpec(),
                    tries: int = 50) -> list[dict[str, Any]]:
    """Segments ``{"kind", "length", "turn" (rad, +-pi/2 on a landing turn, + = left), "h0", ...}``: level first and
    last, and a level run or a landing turn between any two features."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    for _ in range(tries):
        segs: list[dict[str, Any]] = [dict(kind="level", length=spec.start_m, turn=0.0)]
        h = 0.0
        n = int(rng.integers(spec.features[0], spec.features[0] + round(d * (spec.features[1] - spec.features[0])) + 1))
        for k in range(n):
            kind = str(rng.choice(spec.kinds))
            side = float(rng.choice((-1.0, 1.0)))
            if kind == "turn":
                new = [dict(kind="turn", length=_turn_length(spec), turn=side * math.pi / 2)]
                dh = 0.0
            elif kind == "ramp":
                new = [dict(kind="ramp", length=_size(rng, spec.ramp_run_m, 1.0), grade_deg=_size(rng, spec.grade_deg, d),
                            turn=0.0)]
                dh = new[0]["length"] * math.tan(math.radians(new[0]["grade_deg"]))
            elif kind in ("stairs", "turning_stairs"):
                hi = spec.steps[0] + round(d * (spec.steps[1] - spec.steps[0]))
                rise, tread = _size(rng, spec.rise_m, d), _size(rng, spec.tread_m, d)
                fl = lambda m, r=rise, t=tread: dict(kind="stairs", steps=m, rise=r, tread=t, length=m * t, turn=0.0)  # noqa: E731
                new = [fl(int(rng.integers(spec.steps[0], hi + 1)))]
                if kind == "turning_stairs":    # a flight, a landing where the walker turns, a second flight the same way
                    new += [dict(kind="turn", length=_turn_length(spec), turn=side * math.pi / 2),
                            fl(int(rng.integers(spec.steps[0], hi + 1)))]
                dh = sum(f["steps"] for f in new if f["kind"] == "stairs") * rise
            elif kind == "gap":
                new, dh = [dict(kind="gap", length=_size(rng, spec.gap_m, d), turn=0.0)], 0.0
            else:
                new = [dict(kind="platform", height=_size(rng, spec.platform_m, d),
                            length=_size(rng, spec.platform_top_m, d), turn=0.0)]
                dh = 0.0
            if dh:
                ways = [w for w in (1, -1) if abs(h + w * dh) <= spec.max_height_m]
                if not ways:
                    continue
                way = int(rng.choice(ways))
                for f in new:
                    if f["kind"] in ("stairs", "ramp"):
                        f["dir"] = way
            for f in new:
                f["h0"] = h
                if f["kind"] == "stairs":
                    h += f["dir"] * f["steps"] * f["rise"]
                elif f["kind"] == "ramp":
                    h += f["dir"] * f["length"] * math.tan(math.radians(f["grade_deg"]))
                segs.append(f)
            segs.append(dict(kind="level", length=spec.end_m if k == n - 1 else _size(rng, spec.landing_m, d),
                             turn=0.0, h0=h))
        if segs[-1]["kind"] != "level":
            segs.append(dict(kind="level", length=spec.end_m, turn=0.0, h0=h))
        segs[0]["h0"] = 0.0
        if not _crosses(centreline(segs, 0.05), spec.causeway_m + 1.0):
            return segs
    raise RuntimeError("no course kept clear of itself; lower the difficulty or the feature count")


def _height(s: dict[str, Any], u: np.ndarray) -> np.ndarray:
    """Profile height along a segment, ``u`` metres from its start."""
    h0, k = s["h0"], s["kind"]
    if k == "ramp":
        return h0 + s["dir"] * np.clip(u, 0, s["length"]) * math.tan(math.radians(s["grade_deg"]))
    if k == "stairs":
        return h0 + s["dir"] * s["rise"] * np.clip(np.floor(u / s["tread"]) + 1, 1, s["steps"])
    if k == "gap":
        return np.full_like(u, h0 - CourseSpec.pit_m)
    if k == "platform":
        return np.full_like(u, h0 + s["height"])
    return np.full_like(u, h0)


def _pieces(s: dict[str, Any]) -> list[tuple[float, float]]:
    """A segment's centreline as (length, turn) pieces: straight, or straight-arc-straight inside a landing turn."""
    if s["kind"] != "turn":
        return [(s["length"], 0.0)]
    r = TURN_RADIUS_M
    st = (s["length"] - r * math.pi / 2) / 2
    return [(st, 0.0), (r * math.pi / 2, s["turn"]), (st, 0.0)]


def _poses(segs: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    """Each segment's start (x, z, heading)."""
    out, x, z, psi = [], 0.0, 0.0, 0.0
    for s in segs:
        out.append((x, z, psi))
        if s["kind"] == "turn":
            hw = s["length"] / 2 - TURN_RADIUS_M * math.pi / 4 + TURN_RADIUS_M     # the landing's half side
            cx, cz = x + hw * math.cos(psi), z - hw * math.sin(psi)
            psi += s["turn"]
            x, z = cx + hw * math.cos(psi), cz - hw * math.sin(psi)
        else:
            x, z = x + s["length"] * math.cos(psi), z - s["length"] * math.sin(psi)
    return out


def centreline(segs: list[dict[str, Any]], ds: float) -> dict[str, np.ndarray]:
    """Points every ``ds`` m along the course: ``s``, ``x``, ``z``, ``heading`` (rad; forward = (cos, -sin) in x, z),
    profile ``h``, segment index ``seg`` and whether the ground there is a block cell (``block``)."""
    out: dict[str, list] = {k: [] for k in ("s", "x", "z", "heading", "h", "seg", "block")}
    s0 = 0.0
    for i, (s, (x, z, psi)) in enumerate(zip(segs, _poses(segs), strict=True)):
        u0 = 0.0
        for length, turn in _pieces(s):
            n = max(1, int(math.ceil(length / ds)))
            step = length / n
            psis = psi + turn / n * (np.arange(n) + 0.5)
            xs = x + np.cumsum(step * np.cos(psis)) - 0.5 * step * np.cos(psis)
            zs = z - np.cumsum(step * np.sin(psis)) + 0.5 * step * np.sin(psis)
            u = u0 + (np.arange(n) + 0.5) * step
            for k, v in (("s", s0 + u), ("x", xs), ("z", zs), ("heading", psis), ("h", _height(s, u)),
                         ("seg", np.full(n, i)), ("block", np.full(n, s["kind"] != "ramp"))):
                out[k].append(v)
            x, z = xs[-1] + 0.5 * step * np.cos(psis[-1]), zs[-1] - 0.5 * step * np.sin(psis[-1])
            psi, u0 = psi + turn, u0 + length
        s0 += s["length"]
    return {k: np.concatenate(v) for k, v in out.items()}


def _crosses(c: dict[str, np.ndarray], clear: float) -> bool:
    """Does the centreline pass within ``clear`` m of a part of itself more than 3 * clear m away along it?"""
    from scipy.spatial import cKDTree
    pts = np.c_[c["x"], c["z"]]
    for i, j in cKDTree(pts).query_pairs(clear):
        if abs(c["s"][i] - c["s"][j]) > 3 * clear:
            return True
    return False


def course_tile(segs: list[dict[str, Any]], cell: float = 0.04, margin: float = 1.0,
                spec: CourseSpec = CourseSpec()) -> dict[str, Any]:
    """A heightfield tile of a course (``heights``, ``block``, ``origin``, ``cell``, as gp.terrain.atlas takes) plus its
    ``path`` (gp.terrain_gen.centreline every 5 cm) and ``segments``. The course starts at (0, 0) heading +x."""
    c = centreline(segs, cell / 2)
    hw = spec.causeway_m / 2
    pad = hw + 1.0 + margin
    x0, z0 = c["x"].min() - pad, c["z"].min() - pad
    nx = int(math.ceil((c["x"].max() + pad - x0) / cell)) + 1
    nz = int(math.ceil((c["z"].max() + pad - z0) / cell)) + 1
    low = float(min(s["h0"] for s in segs if s["kind"] != "gap"))
    vx, vz = np.meshgrid(x0 + np.arange(nx) * cell, z0 + np.arange(nz) * cell)
    cx, cz = vx[:-1, :-1] + cell / 2, vz[:-1, :-1] + cell / 2
    H, Hc, blk = np.full(vx.shape, low), np.full(cx.shape, low), np.ones(cx.shape, bool)
    for s, (px, pz, psi) in zip(segs, _poses(segs), strict=True):
        e, n = (math.cos(psi), -math.sin(psi)), (-math.sin(psi), -math.cos(psi))
        for X, Z, out in ((vx, vz, H), (cx, cz, Hc)):
            u = (X - px) * e[0] + (Z - pz) * e[1]
            v = (X - px) * n[0] + (Z - pz) * n[1]
            if s["kind"] == "turn":                    # the landing's square, centred hw ahead of the entry
                m = (np.abs(u - hw) <= hw) & (np.abs(v) <= hw)
            else:
                m = (u >= 0) & (u < s["length"]) & (np.abs(v) <= hw + (1.0 if s["kind"] == "gap" else 0.0))
            out[m] = _height(s, u[m])
            if out is Hc:
                blk[m] = s["kind"] != "ramp"
    sub = slice(None, None, max(1, int(round(0.05 / (cell / 2)))))
    return dict(heights=H, block=np.where(blk, Hc, np.nan), origin=(float(x0), float(z0)), cell=float(cell),
                path={k_: v[sub] for k_, v in c.items()}, segments=segs)
