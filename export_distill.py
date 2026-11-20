"""A SPRINT Studio folder from a distill.py run: the student alone, deterministic, in the teacher's environment (the
recipe's), one episode from the first frame of each window it was distilled on, with the same input noise it was
trained with. The folder carries the model, so it opens anywhere.

Run: .venv/bin/python export_distill.py runs/distill_mc021 [--per-group 2] [--to-studio]
Out: runs/<run>/studio_student/ (with --to-studio also copied into SPRINT Studio's results folder)
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import torch
from sprint.export import FrameSource, export_rollout
from sprint.export import format as fmt
from sprint.export.meshes import bake
from sprint.recipes.study import default_ppo

from distill import StudentView, student_loc, student_net
from export_recipe import pick
from gp import data
from gp.pretrain import PretrainConfig
from train import build, load_clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--per-group", type=int, default=0, help="only this many windows per mode and terrain class, spread "
                                                            "over their speeds (export_recipe.pick); 0 = every window")
    ap.add_argument("--to-studio", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run = Path(a.run)
    args = json.loads((run / "record.json").read_text())["args"]
    D = load_clips(args["dataset"], args.get("trials", ""), args.get("modes", "walk,start_stop,stand"))
    if a.per_group:
        D = data.take(D, pick(D, a.per_group))
    n = len(D["name"])
    env, task = build("baseline", D, 1, 0, a.device, starts=[(k, 0) for k in range(n)])
    view = StudentView(None, D, replace(PretrainConfig(), teacher=False, warp=0, bootstrap_iters=0, refit=False),
                       seed=0, foot_model=args["foot_model"])
    view.bind(env.model)
    view.attach(env, task)
    obs_dim = sum(b[1] for b in view.blocks)
    net = student_net(view, env, replace(default_ppo(), rollout_steps=view.c.steps_per_iter), obs_dim).to(view.dev)
    net.load_state_dict(torch.load(run / "checkpoint.pt", map_location=view.dev, weights_only=False)["net"])
    clips = [f"{c} ({m}, {t}, {v:.2f} m/s)" for c, m, t, v in zip(D["name"], D["mode"], D["terrain_class"], D["speed_mps"], strict=True)]
    net.eval()

    @torch.no_grad()
    def policy(_obs: torch.Tensor) -> torch.Tensor:
        s = env.state
        view.prev_action = s.excitation.clone()
        view.new_episode(s.episode_step == 0)          # a fresh episode: new drifts, no last action
        return net._squash(student_loc(net, view.observe(s)[:, :view.n_actor]))

    out = run / "studio_student"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(data.MODEL, out / "model.osim")
    export_rollout(out, env, policy, n_steps=int(round(a.seconds * data.HZ)), episodes=n,
                   config=dict(run=str(run), policy="distilled student (L1 inputs)", clips=clips),
                   sim={"model": str((out / "model.osim").resolve())}, src=FrameSource(env, meshes=bake(data.MODEL)))
    print(f"-> {out}  ({n} episodes)")
    if a.to_studio:
        dest = fmt.results_dir() / f"{run.name}.{out.name}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
        print(f"-> {dest}")


if __name__ == "__main__":
    main()
