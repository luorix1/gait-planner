"""A SPRINT Studio folder from a recipe teacher run (train.py --task baseline --dataset ...): the teacher alone,
deterministic, one episode from the first frame of each picked window. Windows are picked per mode and terrain class
(walk/level, walk/ramp_up, start_stop/level, ...): ``--per-group`` of them, spread over the group's speeds.

Run: .venv/bin/python export_recipe.py runs/recipe_teacher_ab01 [--per-group 2] [--to-studio]
Out: runs/<run>/studio_teacher/ (with --to-studio also copied into SPRINT Studio's results folder)
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sprint.export import FrameSource, export_rollout
from sprint.export import format as fmt
from sprint.export.meshes import bake
from sprint.recipes.ppo import PPO
from sprint.recipes.study import default_ppo

from gp import data
from train import build, load_clips


def pick(D: dict, per_group: int) -> list[int]:
    """``per_group`` windows of each (mode, terrain class), at evenly spaced speed ranks."""
    out: list[int] = []
    for g in sorted({(str(m), str(t)) for m, t in zip(D["mode"], D["terrain_class"], strict=True)}):
        idx = [k for k in range(len(D["name"])) if (str(D["mode"][k]), str(D["terrain_class"][k])) == g]
        idx.sort(key=lambda k: float(D["speed_mps"][k]))
        out += [idx[int(round(r))] for r in np.linspace(0, len(idx) - 1, min(per_group, len(idx)))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--per-group", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--to-studio", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run = Path(a.run)
    args = json.loads((run / "record.json").read_text())["args"]
    D = load_clips(args["dataset"], args.get("trials", ""), args.get("modes", ""))
    chosen = pick(D, a.per_group)
    D = data.take(D, chosen)
    n = len(D["name"])
    env, task = build("baseline", D, 1, 0, a.device, starts=[(k, 0) for k in range(n)])
    out = run / "studio_teacher"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(run / "checkpoint.pt", out / "teacher.pt")         # a snapshot: a running teacher rewrites its checkpoint
    cfg = default_ppo()
    agent = PPO(env, replace(cfg, symmetry_coef=cfg.symmetry_coef if task.symmetric else 0.0), device=a.device)
    agent.load(str(out / "teacher.pt"))
    net = agent.net.eval()
    policy = torch.no_grad()(lambda obs: net.actor(obs, squashed=True))
    shutil.copy2(data.MODEL, out / "model.osim")
    export_rollout(out, env, policy, n_steps=int(round(a.seconds * data.HZ)), episodes=n,
                   config=dict(run=str(run), policy=f"recipe teacher at {agent.env_steps / 1e6:.0f}M steps",
                               clips=[f"{c} ({m}, {t}, {s:.2f} m/s)" for c, m, t, s in
                                      zip(D["name"], D["mode"], D["terrain_class"], D["speed_mps"], strict=True)]),
                   sim={"model": str((out / "model.osim").resolve())}, src=FrameSource(env, meshes=bake(data.MODEL)))
    (out / "teacher.pt").unlink()
    print(f"-> {out}  ({n} episodes at {agent.env_steps / 1e6:.0f}M steps)")
    if a.to_studio:
        dest = fmt.results_dir() / f"{run.name}.{out.name}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
        print(f"-> {dest}")


if __name__ == "__main__":
    main()
