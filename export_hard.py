"""A SPRINT Studio folder from a hard-clip check (imitate_clip.py, scripts/hard/run.sh): its policy.pt, deterministic,
from the clip's first frame with the recipe's reset noise (as eval.json's episodes), on the clip's own terrain, for the
clip's playable length. The folder carries the model, so it opens anywhere.

Run: .venv/bin/python export_hard.py runs/hard/<clip> [more runs ...] [--episodes 3] [--to-studio]
Out: runs/hard/<clip>/studio/ (with --to-studio also copied into SPRINT Studio's results folder)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from sprint.export import FrameSource, export_rollout
from sprint.export import format as fmt
from sprint.export.meshes import bake
from sprint.recipes.flashsac import load_policy

from gp import data
from imitate_clip import FromStart, make_env


def export(run: Path, episodes: int, device: str, to_studio: bool) -> None:
    args = json.loads((run / "record.json").read_text())["args"]
    clip = args["clips"][0]
    full = data.load("train", src=Path(args["data"]))
    D = data.take(full, [list(full["name"]).index(clip)])
    env, task = make_env(D, 1, 0, device, FromStart, starts=[(0, 0)], metabolic=args.get("metabolic"))
    act = load_policy(str(run / "policy.pt"), env)
    out = run / "studio"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    shutil.copy2(data.MODEL, out / "model.osim")
    ev = json.loads((run / "eval.json").read_text())["clips"][clip] if (run / "eval.json").exists() else {}
    export_rollout(out, env, lambda obs: act(env.torch(obs)), n_steps=int(task.clip_span[0]), episodes=episodes,
                   config=dict(run=str(run), clip=clip, policy="policy.pt (best periodic selection), deterministic",
                               eval={k: v for k, v in ev.items() if k != "episode_steps"}),
                   sim={"model": str((out / "model.osim").resolve())}, src=FrameSource(env, meshes=bake(data.MODEL)))
    print(f"-> {out}")
    if to_studio:
        dest = fmt.results_dir() / f"hard.{clip}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
        print(f"-> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--to-studio", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    for r in a.runs:
        export(Path(r), a.episodes, a.device, a.to_studio)


if __name__ == "__main__":
    main()
