"""SPRINT Studio folders for checking the terrain the style prior's clips stand on: one per representative training clip
(by default the median-speed clip of each source and terrain class), each with two episodes:

- ``ep000_initial_state``: the model exactly as training resets it at the clip's first frame (the reset lift
  included) on the clip's SPRINT terrain (profile, or for PARC its own heightfield tile alone: the shared map would
  show neighbouring clips' tiles within the export's 3 m crop), with the clip's own pose as the ghost;
- ``ep001_reference_replay``: the clip's coordinates played kinematically on that terrain, frame by frame (no physics:
  the motion the style prior trains on, where the feet should meet the ground throughout).

Courses (gp.terrain_gen.sample_course2d): ``--courses N --difficulty D [--seed S]`` writes N generated courses instead,
each with ``ep000_initial_state`` (a real standing frame at the course start) and ``ep001_course_walkthrough`` (that
pose carried along the centreline, a view of the ground only: no walking, no physics).

Run: .venv/bin/python export_studio.py [--out DIR] [clip names ...]
     .venv/bin/python export_studio.py --courses 3 --difficulty 0.8
Out: one folder per clip in SPRINT Studio's results folder (or --out), already in its browser
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sprint.dynamics.forward import forward_kinematics
from sprint.export import FrameSource, export_rollout
from sprint.export import format as fmt
from sprint.export.meshes import bake

from gp import data
from gp.terrain import atlas, tile_from_boxes
from gp.terrain_gen import course_tile, sample_course2d
from train import build


def picks(D: dict) -> list[int]:
    """The median-speed clip of each (source, terrain class)."""
    out = []
    for key in sorted({(s, c) for s, c in zip(D["source"], D["terrain_class"], strict=True)}):
        idx = [i for i in range(len(D["name"])) if (D["source"][i], D["terrain_class"][i]) == key]
        out.append(sorted(idx, key=lambda i: D["speed_mps"][i])[len(idx) // 2])
    return out


def own_tile(Di: dict) -> dict:
    """A heightfield clip on a map of its own tile only, built as gp.data.build builds it."""
    name = str(Di["name"][0])
    r = next(r for f in data.SEED.glob("manifest_*.json") for r in json.loads(f.read_text()) if r["name"] == name)
    d = np.load(data.SEED / r["file"])
    cn = list(d["coord_names"])
    t = tile_from_boxes(d["hf"], d["hf_min"], float(d["dx"]))
    x0, z0 = float(d["q"][0, cn.index("pelvis_tx")]), float(d["q"][0, cn.index("pelvis_tz")])
    t["origin"] = (t["origin"][0] - x0, t["origin"][1] - z0)
    A = atlas([t], t["cell"])
    return dict(Di, hf=A, hf_offset=A["offsets"])


def export_clip(D: dict, i: int, root: Path, meshes: dict) -> Path:
    Di = data.take(D, [i])
    if int(Di["terrain_kind"][0]) == 1:
        Di = own_tile(Di)
    name = str(Di["name"][0])
    env, task = build("baseline", Di, 1, 0, "cpu", starts=[(0, 0)])
    src = FrameSource(env, meshes=meshes)                      # ghost: the task's reference pose (the clip's own)
    out = root / f"{time.strftime('%y%m%d.%H%M%S')}.gait_planner.{name}"
    export_rollout(out, env, None, n_steps=1, sim={"model": str(data.MODEL)}, src=src, log=lambda *_: None)
    for f in out.glob("ep*.json.gz"):                          # export_rollout's one physics step: not wanted here
        f.unlink()
    env.reset()                                                # the training reset: clip frame 0 plus its lift
    frames = [src.frame(0.0)]
    fmt.write_episode(out, "ep000_initial_state.json.gz", src.meta(1, frames), frames)
    replay = FrameSource(env, meshes=meshes, ghost=False)
    q, qd = torch.as_tensor(Di["qpos"]), torch.as_tensor(Di["qvel"])
    frames = []
    for k in range(len(q)):
        env.state.qpos[0], env.state.qvel[0] = q[k], qd[k]
        forward_kinematics(env.model, env.data)
        frames.append(replay.frame(k * env.dt_ctrl))
    fmt.write_episode(out, "ep001_reference_replay.json.gz", replay.meta(len(frames), frames), frames)
    lift = float(Di["lift"][0])
    print(f"{name:55s} {Di['terrain_class'][0]:12s} {len(q) / data.HZ:4.1f} s  reset lift {100 * lift:.2f} cm -> {out}")
    return out


def export_course(D: dict, stand: int, segs: list, root: Path, meshes: dict, name: str) -> Path:
    T = course_tile(segs)
    A = atlas([T], T["cell"])
    Di = dict(data.take(D, [stand]), terrain_kind=np.array([1]), hf=A, hf_offset=A["offsets"])
    env, task = build("baseline", Di, 1, 0, "cpu", starts=[(0, 0)])
    out = root / f"{time.strftime('%y%m%d.%H%M%S')}.gait_planner.{name}"
    export_rollout(out, env, None, n_steps=1, sim={"model": str(data.MODEL)}, src=FrameSource(env, meshes=meshes),
                   log=lambda *_: None)
    for f in out.glob("ep*.json"):
        f.unlink()
    for f in out.glob("ep*.json.gz"):
        f.unlink()
    names, p = Di["coord_names"], T["path"]
    q0 = torch.as_tensor(Di["qpos"][0]).clone()
    # carried along the path at the highest ground within 0.3 m, so the pose floats over gaps instead of sinking
    w = max(1, int(round(0.3 / float(p["s"][1] - p["s"][0]))))
    hmax = np.array([p["h"][max(0, k - w):k + w + 1].max() for k in range(len(p["h"]))])
    src = FrameSource(env, meshes=meshes, ghost=False)

    def place(k: int) -> None:
        q = q0.clone()
        q[names.index("pelvis_tx")], q[names.index("pelvis_tz")] = float(p["x"][k]), float(p["z"][k])
        q[names.index("pelvis_ty")] += float(hmax[k])
        q[names.index("pelvis_rotation")] = float(p["heading"][k])
        env.state.qpos[0], env.state.qvel[0] = q, 0.0
        forward_kinematics(env.model, env.data)

    k0 = int(np.searchsorted(p["s"], 1.0))
    place(k0)
    frames = [src.frame(0.0)]
    fmt.write_episode(out, "ep000_initial_state.json.gz", src.meta(1, frames), frames)
    ks = range(k0, len(p["s"]), max(1, int(round(1.2 / data.HZ / float(p["s"][1] - p["s"][0])))))
    frames = []
    for n, k in enumerate(ks):
        place(k)
        frames.append(src.frame(n / data.HZ))
    fmt.write_episode(out, "ep001_course_walkthrough.json.gz", src.meta(len(frames), frames), frames)
    feats = [(s["kind"], {k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()
                          if k not in ("kind", "h0", "turn", "length")} | ({"deg": round(np.degrees(s["turn"]))} if s["turn"] else {}))
             for s in segs if s["kind"] != "level"]
    print(f"{name}: {p['s'][-1]:.1f} m, grid {T['heights'].shape}, {feats} -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--courses", type=int, default=0)
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    D = data.load("train")
    if a.courses:
        root = Path(a.out) if a.out else fmt.results_dir()
        meshes = bake(str(data.MODEL))
        stand = next(i for i in range(len(D["name"])) if D["mode"][i] == "stand" and D["terrain_class"][i] == "level")
        rng = np.random.default_rng(a.seed)
        for k in range(a.courses):
            export_course(D, stand, sample_course2d(rng, a.difficulty), root, meshes, f"course_d{a.difficulty:.1f}_s{a.seed}_{k}")
        return
    idx = [list(D["name"]).index(n) for n in a.clips] if a.clips else picks(D)
    root = Path(a.out) if a.out else fmt.results_dir()
    root.mkdir(parents=True, exist_ok=True)
    meshes = bake(str(data.MODEL))
    for i in idx:
        export_clip(D, i, root, meshes)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
