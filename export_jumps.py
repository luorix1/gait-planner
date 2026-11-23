"""Jumps that need preparation, with and without the skill layer (gp.takeoff): a full-pose planner (train_planner.py
--pose2) rolled out open loop (export_pose2.lazy_rollout, event-triggered replanning) on straight courses with a gap
(0.3 m: a step; 0.6, 1.0, 1.4 m: run-ups) or a platform (0.2, 0.4, 0.6 m: crouches), each after 6 m of level run-up.

Goals, two ways, everything else the same:
- ``constant``: the course speed throughout (scoreboard.py's courses): the planner must find the run-up or the crouch
  on its own within its 1.2 s and 3 m;
- ``takeoff``: gp.takeoff.TakeoffGoals: the run-up planned once when the feature comes into view (8 m), carried by the
  goals, re-solved on events; with a ``goal_h`` planner the landing point at its own time.

Per course (summary.json, printed), from the pelvis centre of mass along the route (kinematic proxies, not physics):
the skill's targets (takeoff speed, crouch) and what the rollout did: speed over the 0.2 s before the takeoff point,
the pelvis's lowest height over its ground in the 0.6 s before it against its standing height (the crouch), whether it
got across (0.5 m past the landing) or fell (pelvis under the takeoff ground + 0.4 m over a gap), and replans/s.

Run: GP_DATASET=v4 .venv/bin/python export_jumps.py runs/pose2_planner_v12 [--checkpoint planner_best.pt]
Out: <run>/jumps.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_pose2 import NS, circmean, lazy_rollout, start
from gp import data, footfk, lazy
from gp import takeoff as tk
from gp.planner import FUT, HIST, HZ, STRIDE, to_frame
from gp.pose_planner import PosePlanner, Windows
from gp.terrain import Ground, atlas, pad
from gp.terrain_gen import CourseSpec, course_tile
from scoreboard import START_X, place

SPEED = 1.2


def jump_courses() -> list[tuple[str, str, float, list[dict]]]:
    """(name, kind, size, gp.terrain_gen segments): 6 m of run-up, the feature, 6 m after."""
    lv = lambda m, h=0.0: dict(kind="level", length=m, turn=0.0, h0=h)     # noqa: E731
    out = [(f"gap {100 * w:.0f} cm", "gap", w, [lv(6.0), dict(kind="gap", length=w, turn=0.0, h0=0.0), lv(6.0)])
           for w in (0.3, 0.6, 1.0, 1.4)]
    out += [(f"box {100 * h:.0f} cm", "up", h, [lv(6.0), dict(kind="platform", height=h, length=6.0, turn=0.0, h0=0.0)])
            for h in (0.2, 0.4, 0.6)]
    return out


def route_s(p: dict[str, np.ndarray], xz: np.ndarray) -> np.ndarray:
    """Route position of each point ``xz`` ``[T, 2]`` (nearest route point; the courses are straight)."""
    d = (p["x"][None] - xz[:, :1]) ** 2 + (p["z"][None] - xz[:, 1:]) ** 2
    return p["s"][d.argmin(1)]


def measure(com: np.ndarray, p: dict[str, np.ndarray], skill: tk.Skill | None, kind: str) -> dict:
    """What the rollout did against the first skill (``com`` ``[T, 3]`` pelvis com at 20 Hz)."""
    s = route_s(p, com[:, [0, 2]])
    above = com[:, 1] - np.interp(s, p["s"], p["h"])
    stand = float(np.median(above[: int(HZ)]))
    v = np.r_[0.0, np.diff(s)] * HZ
    out = dict(stand_m=round(stand, 3), max_s=round(float(s.max()), 2))
    if skill is None or skill.kind not in ("gap", "up"):
        return dict(out, skill=None if skill is None else skill.kind, note=None if skill is None else skill.note)
    f = skill.feature
    i = int(np.argmax(s >= skill.s_takeoff)) if (s >= skill.s_takeoff).any() else None
    out.update(skill=skill.kind, v_takeoff_target=round(skill.v_takeoff, 2), crouch_target_m=round(skill.crouch_m, 3),
               s_takeoff=round(skill.s_takeoff, 2))
    if i is not None:
        w0, c0 = max(0, i - int(0.2 * HZ)), max(0, i - int(0.6 * HZ))
        out.update(v_takeoff=round(float(v[w0:i + 1].mean()), 2), crouch_m=round(stand - float(above[c0:i + 1].min()), 3))
    out["across"] = bool(s.max() >= skill.s_land + 0.5)
    if kind == "gap":
        over = (s > f.s_near) & (s < f.s_far)
        out["fell"] = bool((com[over, 1] < 0.4).any())             # takeoff ground at 0: the pelvis in the pit
    else:
        out["on_top"] = bool(out["across"] and abs(above[-1] - stand) < 0.15)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="planner_best.pt")
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--hold", type=int, default=50)
    ap.add_argument("--xfade", type=int, default=0)
    ap.add_argument("--hold-smooth", type=float, default=0.0)
    a = ap.parse_args()
    run, dev = Path(a.run), "cuda" if torch.cuda.is_available() else "cpu"
    D = data.load(("test_trial", "test_terrain"))
    fk = footfk.fit(D, dev, data.ROOT / "data" / "foot_fk.pt")
    P = PosePlanner.load(run / a.checkpoint, dev).attach(fk)
    W = Windows(D, dev, fk)
    W.starts, W.lengths = D["start"], D["frames"]

    courses = jump_courses()
    tiles = [course_tile(segs, spec=CourseSpec.full()) for *_, segs in courses]
    hf = atlas(tiles, tiles[0]["cell"])
    n = len(tiles)
    X, Y = pad([np.array([[-1e3, 0.0], [1e3, 0.0]])] * n)
    G = Ground(X, Y, np.ones(n, bool), hf, hf["offsets"], dev)
    gid = torch.arange(n, device=dev)
    paths = [t["path"] for t in tiles]

    # every course starts from the longest held-out AB01 level walk, placed at START_X heading +x (as scoreboard.py)
    room = np.asarray(D["frames"]) - STRIDE * (HIST + FUT)
    walk = [k for k in range(len(D["name"])) if str(D["mode"][k]) == "walk" and str(D["terrain_class"][k]) == "level"
            and str(D["source"][k]) == "ab01" and room[k] > 0]
    k_walk = max(walk, key=lambda k: float(D["frames"][k]))
    a0 = torch.full((n,), int(D["start"][k_walk]) + STRIDE * (HIST - 1), device=dev)
    s, ph = start(W, a0)
    o, a_psi = s["pcom"][:, -1, [0, 2]], circmean(ph[:, -NS:])
    g0 = W.fr.G(W.fr.clip[a0], s["pcom"][:, -1:, 0], s["pcom"][:, -1:, 2])[:, 0]
    xz = torch.tensor([[START_X, 0.0]], device=dev).expand(n, 2)
    zero = torch.zeros(n, device=dev)
    c_start = place(s, ph, o, a_psi, g0, xz, zero, G(gid, xz[:, :1], xz[:, 1:])[:, 0])

    def constant(f: int, o: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        ahead = torch.stack([o[:, 0] + SPEED * FUT / HZ, o[:, 1] * 0], -1)           # straight courses along +x
        return to_frame(ahead[:, None], o, psi)[:, 0]

    n_frames = int(round(a.seconds * HZ))
    T = lazy.Triggers()
    rows = []
    for mode in ("constant", "takeoff"):
        gen = torch.Generator(device=dev).manual_seed(0)
        goals = (tk.TakeoffGoals(paths, [SPEED] * n, [START_X] * n, HZ, goal_h=bool(P.cfg.get("goal_h")))
                 if mode == "takeoff" else constant)
        r, st, seam = lazy_rollout(P, G, gid, *c_start, goals, n_frames, T, a.guidance, gen, a.hold, xfade=a.xfade,
                                   hold_smooth=a.hold_smooth)
        for b, (name, kind, _size, _) in enumerate(courses):
            p = paths[b]
            first = (goals.history[b][0] if mode == "takeoff" else None)
            sk = tk.solve(tk.scan(p["s"][p["s"] >= START_X], p["h"][p["s"] >= START_X]), START_X, SPEED, SPEED,
                          float(p["s"][-1]))                    # the targets, the same for both modes
            row = dict(mode=mode, course=name, **measure(r["com"][b], p, sk, kind),
                       replans_per_s=round(float(seam[b].sum() / 2 / a.seconds), 2))
            if mode == "takeoff":
                row["skill_history"] = goals.history[b]
                row["first_skill"] = first
            rows.append(row)
            print(row, flush=True)
        print(mode, "replanning", st, *(["skill re-solves", goals.log] if mode == "takeoff" else []), flush=True)
    (run / "jumps.json").write_text(json.dumps(rows, indent=1))
    print(f"-> {run / 'jumps.json'}")


if __name__ == "__main__":
    main()
