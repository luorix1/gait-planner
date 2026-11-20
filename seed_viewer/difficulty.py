"""Rough difficulty labels for the PARC clips (easy / medium / hard, with a one-line reason), shown by the seed viewer.

Measured on the canonical clip (PARC FK keypoints, z up, the fitted heightfield) and the model export (knee angle):
- level changes: the ground under the feet at each stance (a foot point within 6 cm of the terrain under it; PARC's
  own contact flags miss many stances), a change of more than 0.15 m between stances, and the pelvis's horizontal
  speed when the last foot leaves the lower or upper level;
- cuts: the largest change in running direction within 0.5 s while faster than 2.5 m/s;
- top speed (0.2 s smoothed), longest flight (no foot near the ground), deepest knee (the model allows 140 deg).

Hard if any: knee past 140 deg; a level change of 0.5 m or more taken at 4 m/s or faster; a level change of 0.9 m or
more; top speed 5.5 m/s or more; a cut of 90 deg or more at speed; flight longer than 0.6 s. Medium if any: a level
change of 0.3 m or more; top speed 4 m/s or more; a cut of 45 deg or more at speed; flight longer than 0.35 s.
Otherwise easy. The thresholds are a judgement, not a measurement of what the body can do.

Run:  cd ~/Jinwoo/gait_planner && .venv/bin/python seed_viewer/difficulty.py
Out:  seed_library/data/viewer_difficulty.json  {clip name: [level, reason]}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gp import data  # noqa: E402

LIB = data.SEED                                            # seed_library/data/osim_export
CANON = LIB.parent / "canonical"
OUT = LIB.parent / "viewer_difficulty.json"


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(n) / n
    return np.stack([np.convolve(np.pad(c, (n // 2, n - 1 - n // 2), mode="edge"), k, "valid") for c in x.T], 1)


def measure(name: str, file: str) -> dict:
    z = np.load(CANON / file, allow_pickle=True)
    fps, root, kp, hf, mn, dx = int(z["fps"]), z["root"], z["kp"], z["hf"], z["hf_min"], float(z["dx"])
    vel = np.gradient(smooth(root[:, :2], max(1, round(0.2 * fps))), axis=0) * fps
    speed = np.linalg.norm(vel, axis=1)
    feet = kp[:, 4:10]                                     # ankles, heels, toes [T, 6, 3]
    i = np.clip(np.round((feet[..., 0] - mn[0]) / dx).astype(int), 0, hf.shape[0] - 1)
    j = np.clip(np.round((feet[..., 1] - mn[1]) / dx).astype(int), 0, hf.shape[1] - 1)
    gap = feet[..., 2] - hf[i, j]
    down = (gap < 0.06).any(1)
    ground = np.where(down, hf[i, j][np.arange(len(gap)), gap.argmin(1)], np.nan)
    # level changes between consecutive stances
    changes, lvl, lt = [], None, None
    for t in np.flatnonzero(down):
        g = float(ground[t])
        if lvl is not None and abs(g - lvl) > 0.15:
            changes.append((g - lvl, float(speed[lt])))    # height, speed as the last foot left the old level
        if lvl is None or abs(g - lvl) > 0.15:
            lvl = g
        lt = t
    # flight
    d = np.diff(np.r_[1, down.astype(int), 1])
    flight = float(((np.flatnonzero(d == 1) - np.flatnonzero(d == -1)).max(initial=0)) / fps)
    # cuts: direction change within 0.5 s while fast
    head = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    w = max(1, round(0.5 * fps))
    cut = 0.0
    for t in range(len(head) - w):
        if speed[t] > 2.5 and speed[t + w] > 2.5:
            cut = max(cut, abs(np.degrees(head[t + w] - head[t])))
    knee = None
    ex = LIB / file
    if ex.exists():
        e = np.load(ex, allow_pickle=True)
        names = [str(n) for n in e["coord_names"]]
        knee = float(np.degrees(np.abs(e["q"][:, [names.index("knee_angle_r"), names.index("knee_angle_l")]]).max()))
    return dict(changes=changes, top=float(speed.max()), cut=cut, flight=flight, knee=knee)


def label(m: dict) -> tuple[str, str]:
    ch = [c for c in m["changes"] if abs(c[0]) >= 0.3]
    ups, downs = [c for c in ch if c[0] > 0], [c for c in ch if c[0] < 0]
    steps = ", ".join(f"{d} {abs(c[0]):.1f} m at {c[1]:.1f} m/s" for d, cs in (("up", ups), ("down", downs)) if cs
                      for c in [max(cs, key=lambda c: c[1])])           # the fastest of each direction
    hard, med = [], []
    if m["knee"] is not None and m["knee"] > 140:
        hard.append(f"knee {m['knee']:.0f} deg > model's 140")
    if any(abs(c[0]) >= 0.5 and c[1] >= 4.0 for c in ch) or any(abs(c[0]) >= 0.9 for c in ch):
        hard.append(steps)
    if m["top"] >= 5.5:
        hard.append(f"top speed {m['top']:.1f} m/s")
    if m["cut"] >= 90:
        hard.append(f"cuts {m['cut']:.0f} deg in 0.5 s at speed")
    if m["flight"] > 0.6:
        hard.append(f"flight {m['flight']:.2f} s")
    if hard:
        return "hard", "; ".join(hard)
    if steps:
        med.append(steps)
    if m["top"] >= 4.0:
        med.append(f"top speed {m['top']:.1f} m/s")
    if m["cut"] >= 45:
        med.append(f"turns {m['cut']:.0f} deg in 0.5 s at speed")
    if m["flight"] > 0.35:
        med.append(f"flight {m['flight']:.2f} s")
    if med:
        return "medium", "; ".join(med)
    return "easy", f"{'no step over 0.3 m' if not ch else steps}, top speed {m['top']:.1f} m/s, turns at most {m['cut']:.0f} deg in 0.5 s"


def main() -> None:
    man = json.loads((LIB / "manifest_parc.json").read_text())
    out = {}
    for r in man:
        file = r.get("file") or f"parc/{r['name']}.npz"
        if not (CANON / file).exists():
            continue
        lv, why = label(measure(r["name"], file))
        out[r["name"]] = [lv, why]
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    n = {k: sum(v[0] == k for v in out.values()) for k in ("easy", "medium", "hard")}
    print(f"{OUT}: {len(out)} clips, {n}")


if __name__ == "__main__":
    main()
