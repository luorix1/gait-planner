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
