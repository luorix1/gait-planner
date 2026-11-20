"""Rough difficulty labels (easy / medium / hard, with a one-line reason) for the PARC, AB01 and Bones SEED clips,
shown by the seed viewer.

Measured on the canonical clip (FK keypoints, z up, the clip's terrain as its heightfield) and the model export (knee):
- level changes: the ground under the feet at each stance (a foot point within 6 cm of the terrain under it; PARC's
  own contact flags miss many stances), a change of more than 0.15 m between stances, and the pelvis's horizontal
  speed when the last foot leaves the lower or upper level;
- cuts: the largest change in running direction within 0.5 s while faster than 2.5 m/s;
- top speed (0.2 s smoothed), longest flight (no foot near the ground), deepest knee (the model allows 140 deg);
- jump height: the pelvis's highest rise above its usual height over the higher of the ground the feet last stood on
  and the ground they land on next;
- facing from the hip line: spin (facing change within 0.5 s) and travel backward or sideways (above 0.8 m/s);
- the clip's terrain: stairs (rise per step) and ramps (grade).

Hard if any: knee past 140 deg; a level change of 0.5 m or more taken at 4 m/s or faster; a level change of 0.9 m or
more; top speed 5.5 m/s or more; a cut of 90 deg or more at speed; flight longer than 0.6 s; a jump of 0.35 m or more.
Medium if any: a level change of 0.3 m or more; top speed 4 m/s or more; a cut of 45 deg or more at speed; flight
longer than 0.35 s; a jump of 0.2 m or more; a spin of 90 deg or more within 0.5 s (180: hard); moving backward
or sideways at 1.5 m/s or more; stairs with a rise of 0.2 m or more or taken at 2 m/s or more; a ramp of
10 deg or more. Otherwise easy. The thresholds are a judgement, not a measurement of what the body can do.

Run:  cd ~/Jinwoo/gait_planner && .venv/bin/python seed_viewer/difficulty.py      (~14k clips, a few minutes)
Out:  seed_library/data/viewer_difficulty.json  {clip name: [level, reason]}
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
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


def measure(file: str) -> dict:
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
    fast = (speed[:-w] > 2.5) & (speed[w:] > 2.5)
    cut = float(np.degrees(np.abs(head[w:] - head[:-w])[fast]).max(initial=0.0))
    # jump height over the higher of the ground the feet last stood on and the ground they land on next
    t_ = np.arange(len(ground))
    prev = np.maximum.accumulate(np.where(down, t_, 0))
    nxt = np.minimum.accumulate(np.where(down, t_, len(t_) - 1)[::-1])[::-1]
    base = np.fmax(ground[prev], ground[nxt]) if down.any() else np.zeros(len(t_))
    h = root[:, 2] - np.nan_to_num(base)
    jump = float(h.max() - np.median(h[down])) if down.any() else 0.0
    # facing (from the hip line), spin within 0.5 s, and moving backward or sideways
    right = smooth(kp[:, 1, :2] - kp[:, 0, :2], max(1, round(0.1 * fps)))
    yaw = np.unwrap(np.arctan2(right[:, 0], -right[:, 1]))          # forward = z x right
    spin = float(np.degrees(np.abs(yaw[w:] - yaw[:-w])).max(initial=0.0))
    rel = np.degrees(np.abs((np.arctan2(vel[:, 1], vel[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi))
    moving = speed > 0.8
    back = float(speed[moving & (rel > 135)].max(initial=0.0))
    side = float(speed[moving & (rel > 60) & (rel < 120)].max(initial=0.0))
    ter = json.loads(str(z["terrain"]))
    knee = None
    ex = LIB / file
    if ex.exists():
        e = np.load(ex, allow_pickle=True)
        names = [str(n) for n in e["coord_names"]]
        knee = float(np.degrees(np.abs(e["q"][:, [names.index("knee_angle_r"), names.index("knee_angle_l")]]).max()))
    return dict(changes=changes, top=float(speed.max()), cut=cut, flight=flight, knee=knee, jump=jump, ter=ter, spin=spin, back=back, side=side,
                stair_speed=float(np.median(speed)))


def label(m: dict) -> tuple[str, str]:
    ch = [c for c in m["changes"] if abs(c[0]) >= 0.3]
    ups, downs = [c for c in ch if c[0] > 0], [c for c in ch if c[0] < 0]
    steps = ", ".join(f"{d} {abs(c[0]):.1f} m at {c[1]:.1f} m/s" for d, cs in (("up", ups), ("down", downs)) if cs
                      for c in [max(cs, key=lambda c: c[1])])           # the fastest of each direction
    hard, med = [], []
    if m["knee"] is not None and m["knee"] >= 141:
        hard.append(f"knee {m['knee']:.0f} deg > model's 140")
    if any(abs(c[0]) >= 0.5 and c[1] >= 4.0 for c in ch) or any(abs(c[0]) >= 0.9 for c in ch):
        hard.append(steps)
    if m["top"] >= 5.5:
        hard.append(f"top speed {m['top']:.1f} m/s")
    if m["cut"] >= 90:
        hard.append(f"cuts {m['cut']:.0f} deg in 0.5 s at speed")
    if m["flight"] > 0.6:
        hard.append(f"flight {m['flight']:.2f} s")
    if m["jump"] >= 0.35:
        hard.append(f"jumps {m['jump']:.2f} m high")
    if m["spin"] >= 180:
        hard.append(f"spins {m['spin']:.0f} deg in 0.5 s")
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
    if m["jump"] >= 0.2:
        med.append(f"jumps {m['jump']:.2f} m high")
    if m["spin"] >= 90:
        med.append(f"spins {m['spin']:.0f} deg in 0.5 s")
    if m["back"] >= 1.5:
        med.append(f"moves backward at {m['back']:.1f} m/s")
    if m["side"] >= 1.5:
        med.append(f"moves sideways at {m['side']:.1f} m/s")
    t = m["ter"]
    if t.get("type") == "stairs" and t.get("n_steps", 0) > 1 and (t["rise_m"] >= 0.2 or m["stair_speed"] >= 2.0):
        med.append(f"stairs, {t['rise_m'] * 100:.0f} cm rise at {m['stair_speed']:.1f} m/s")
    if t.get("type") == "plane" and abs(t.get("grade_deg", 0)) >= 10:
        med.append(f"ramp {t['grade_deg']:.0f} deg")
    if med:
        return "medium", "; ".join(med)
    t, where = m["ter"], ""
    if t.get("type") == "stairs" and t.get("n_steps", 0) > 1:
        where = f"stairs {t['rise_m'] * 100:.0f} cm, "
    elif t.get("type") == "plane" and abs(t.get("grade_deg", 0)) >= 2:
        where = f"ramp {t['grade_deg']:.0f} deg, "
    return "easy", f"{where}top speed {m['top']:.1f} m/s, spins at most {m['spin']:.0f} deg in 0.5 s"


def one(r: dict) -> tuple[str, list] | None:
    file = r.get("file") or f"{r['name'].split('_')[0]}/{r['name']}.npz"
    if not (CANON / file).exists():
        return None
    try:
        return r["name"], list(label(measure(file)))
    except Exception as e:  # noqa: BLE001 - one odd clip should not stop the rest
        return r["name"], ["?", f"not measured: {type(e).__name__}: {e}"]


def main() -> None:
    man = [r for pat in ("manifest_parc.json", "manifest_ab01.json", "manifest_seed_bones_*.json")
           for f in sorted(LIB.glob(pat)) for r in json.loads(f.read_text())]
    with Pool() as pool:
        out = dict(x for x in pool.imap_unordered(one, man, chunksize=32) if x)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    for src in ("parc", "ab01", "seed"):
        n = {k: sum(v[0] == k for n_, v in out.items() if n_.startswith(src)) for k in ("easy", "medium", "hard", "?")}
        print(f"{src}: {n}")
    print(f"-> {OUT}: {len(out)} clips")


if __name__ == "__main__":
    main()
