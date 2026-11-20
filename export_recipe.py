"""A SPRINT Studio folder from a recipe teacher run (train.py --task baseline --dataset ...): the teacher alone,
deterministic, one episode from the first frame of each picked window. Windows are picked per mode and terrain class
(walk/level, walk/ramp_up, start_stop/level, ...): ``--per-group`` of them, spread over the group's speeds.

Run: .venv/bin/python export_recipe.py runs/recipe_teacher_ab01 [--per-group 2] [--to-studio]
Out: runs/<run>/studio_teacher/ (with --to-studio also copied into SPRINT Studio's results folder)

Every window (--per-group 0), in N parallel CPU processes, then one folder:
  for k in 0..N-1: export_recipe.py RUN --per-group 0 --shard k/N --device cpu &   (-> studio_teacher_all.part<k>)
  export_recipe.py RUN --per-group 0 --merge N [--to-studio]                         (-> studio_teacher_all)
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
    ap.add_argument("--per-group", type=int, default=2, help="0 = every window")
    ap.add_argument("--shard", default="", help="K/N: only windows K, K+N, ... (folder <out>.part<K>)")
    ap.add_argument("--merge", type=int, default=0, help="join the N shard folders into one, episodes in window order")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--to-studio", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run = Path(a.run)
    args = json.loads((run / "record.json").read_text())["args"]
    D = load_clips(args["dataset"], args.get("trials", ""), args.get("modes", ""))
    chosen = pick(D, a.per_group) if a.per_group else list(range(len(D["name"])))
    name = "studio_teacher" if a.per_group else "studio_teacher_all"
    clips = [f"{c} ({m}, {t}, {s:.2f} m/s)" for c, m, t, s in zip(*(np.asarray(D[k])[chosen] for k in
                                                                   ("name", "mode", "terrain_class", "speed_mps")), strict=True)]
    if a.merge:
        return merge(run / name, a.merge, clips, a.to_studio)
    if a.shard:
        k, N = map(int, a.shard.split("/"))
        chosen, clips, name = chosen[k::N], clips[k::N], f"{name}.part{k}"
    D = data.take(D, chosen)
    n = len(D["name"])
    env, task = build("baseline", D, 1, 0, a.device, starts=[(k, 0) for k in range(n)])
    out = run / name
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
                               clips=clips),
                   sim={"model": str((out / "model.osim").resolve())}, src=FrameSource(env, meshes=bake(data.MODEL)))
    (out / "teacher.pt").unlink()
    print(f"-> {out}  ({n} episodes at {agent.env_steps / 1e6:.0f}M steps)")
    if a.to_studio:
        to_studio(out, run)


def to_studio(out: Path, run: Path) -> None:
    dest = fmt.results_dir() / f"{run.name}.{out.name}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(out, dest)
    print(f"-> {dest}")


def merge(out: Path, N: int, clips: list[str], studio: bool) -> None:
    """Shard folders ``<out>.part<k>`` -> ``out``: episode j of shard k becomes episode k + j N (window order)."""
    parts = [out.with_name(f"{out.name}.part{k}") for k in range(N)]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    for f in ("model.json.gz", "model.osim"):
        shutil.copy2(parts[0] / f, out / f)
    cfg = json.loads((parts[0] / "config.sprint").read_text())
    cfg["clips"] = clips
    (out / "config.sprint").write_text(json.dumps(cfg, indent=1))
    n = 0
    for k, p in enumerate(parts):
        for f in sorted(p.glob("ep*.json.gz")):
            j = int(f.name[2:5])
            shutil.copy2(f, out / f"ep{k + j * N:03d}{f.name[5:]}")
            n += 1
    assert n == len(clips), f"{n} episodes for {len(clips)} windows"
    print(f"-> {out}  ({n} episodes)")
    if studio:
        to_studio(out, out.parent)


if __name__ == "__main__":
    main()
