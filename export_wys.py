"""A SPRINT Studio folder from a Watch Your Step run (train_wys.py): one 10 s episode per terrain type at level 5,
walking forward at 0.8 m/s, with the run's policy and its latest foothold predictor, training aids off (no helper pull,
pushes or constant push; the authors' play settings). The folder carries the model, so it opens anywhere.

Run: .venv/bin/python export_wys.py runs/wys_v1 [--level 5] [--speed 0.8] [--seconds 10] [--assist 0.5] [--to-studio]
Out: runs/<run>/studio_<iteration>_L<level>[_pull<assist>]/ (with --to-studio also copied into SPRINT Studio's results folder)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import torch
from sprint.export import export_rollout
from sprint.export import format as fmt
from sprint.export.meshes import bake
from sprint.export.rollout import FrameSource
from sprint.recipes.ppo import ActorCritic, PPOConfig, env_action_bounds

from gp import data
from gp.wys import WysConfig
from train_wys import build


def policy(env, path: Path):
    """The trained actor, deterministic (as evaluate.py loads it)."""
    dev = "cuda" if env.device.is_cuda else "cpu"
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = PPOConfig(**{k: tuple(v) if k == "hidden" else v for k, v in ck["cfg"].items()})
    lo, hi = env_action_bounds(env, cfg.excitation_min)
    net = ActorCritic(env.obs_dim, env.action_dim, cfg, action_low=lo, action_high=hi,
                      actor_cols=ck["net"].get("actor_cols")).to(dev)
    missing, unexpected = net.load_state_dict(ck["net"], strict=False)
    if [k for k in missing if not k.startswith("vf.")] or unexpected:
        raise ValueError(f"{path} does not fit this environment")
    net.eval()
    return torch.no_grad()(lambda obs: net.actor(env.torch(obs), squashed=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--speed", type=float, default=0.8)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--assist", type=float, default=0.0, help="keep a helper pull of this x weight on (training's is 0.5 at level 0)")
    ap.add_argument("--to-studio", action="store_true", help="also copy the folder into SPRINT Studio's results folder")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run = Path(a.run)
    models = sorted(run.glob("foot_model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    from gp import wys_terrain as WT
    types = WT.column_types()
    tiles = tuple((a.level, types.index(t)) for t in dict.fromkeys(types))           # first column of each type
    cfg = replace(WysConfig(), training=False, eval_assist=a.assist, eval_tiles=tiles, lin_x=(a.speed, a.speed), lin_y=(0.0, 0.0),
                  ang_z=(0.0, 0.0), heading_stiffness=0.0, standing_frac=0.0, turning_frac=0.0,
                  resample_s=a.seconds + 1, xy_drift=0.0, z_drift=0.0, scan_noise=0.0,
                  n_lin=0.0, n_ang=0.0, n_grav=0.0, n_q=0.0, n_qd=0.0, spike_p=0.0)
    env, task = build(1, 0, a.device, None, cfg, foot_model=str(models[-1]) if models else None)
    it = json.loads((run / "record.json").read_text()).get("iterations")
    state = run / "task_state.pt"
    if state.exists():
        it = torch.load(state, weights_only=False)["steps"] // task.c.steps_per_iter
    out = run / f"studio_{it}_L{a.level}" if not a.assist else run / f"studio_{it}_L{a.level}_pull{a.assist:g}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(data.MODEL, out / "model.osim")        # the folder carries its model; meshes baked from beside the source
    n = int(round(a.seconds / cfg.dt))
    export_rollout(out, env, policy(env, run / "policy.pt"), n_steps=n, episodes=len(tiles),
                   config=dict(run=str(run), iteration=it, foot_model=models[-1].name if models else "bootstrap",
                               tiles={f"ep{k:03d}": types[c] for k, (_, c) in enumerate(tiles)}, level=a.level, assist=a.assist,
                               speed_mps=a.speed),
                   sim={"model": str((out / "model.osim").resolve())}, src=FrameSource(env, meshes=bake(data.MODEL)))
    print(f"-> {out}  ({len(tiles)} episodes: {', '.join(dict.fromkeys(types))})")
    if a.to_studio:
        dest = fmt.results_dir() / f"{run.name}.{out.name}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
        print(f"-> {dest}")


if __name__ == "__main__":
    main()
