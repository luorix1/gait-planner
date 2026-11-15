"""SPRINT Studio folders for checking the terrain the style prior's clips stand on: one per representative training clip
(by default the median-speed clip of each source and terrain class), each with two episodes:

- ``ep000_initial_state``: the model exactly as training resets it at the clip's first frame (the reset lift
  included) on the clip's SPRINT terrain profile, with the clip's own pose as the ghost;
- ``ep001_reference_replay``: the clip's coordinates played kinematically on that terrain, frame by frame (no physics:
  the motion the style prior trains on, where the feet should meet the ground throughout).

Run: .venv/bin/python export_studio.py [--out DIR] [clip names ...]
Out: one folder per clip in SPRINT Studio's results folder (or --out), already in its browser
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sprint.dynamics.forward import forward_kinematics
from sprint.export import FrameSource, export_rollout
from sprint.export import format as fmt
from sprint.export.meshes import bake

from gp import data
from train import build


def picks(D: dict) -> list[int]:
    """The median-speed clip of each (source, terrain class)."""
    out = []
    for key in sorted({(s, c) for s, c in zip(D["source"], D["terrain_class"], strict=True)}):
        idx = [i for i in range(len(D["name"])) if (D["source"][i], D["terrain_class"][i]) == key]
        out.append(sorted(idx, key=lambda i: D["speed_mps"][i])[len(idx) // 2])
    return out


def export_clip(D: dict, i: int, root: Path, meshes: dict) -> Path:
    Di = data.take(D, [i])
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = data.load("train")
    idx = [list(D["name"]).index(n) for n in a.clips] if a.clips else picks(D)
    root = Path(a.out) if a.out else fmt.results_dir()
    root.mkdir(parents=True, exist_ok=True)
    meshes = bake(str(data.MODEL))
    for i in idx:
        export_clip(D, i, root, meshes)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
