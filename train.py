"""Train a muscle policy on the train split with SPRINT's PPO (the imitation recipe's default learner settings).

- ``--task baseline``: full-pose DeepMimic on each clip's terrain (gp.tasks.TerrainImitation).
- ``--task planner``: plan tracking with the style prior (gp.tasks.PlanTracking); needs ``--prior``.

Both use the same body, clips, physics (40 Hz control, 75 physics steps, implicit contact), motor delays, effort
terms and learner, so they compare at matched compute (``--hours``).

Run (GPU, detached; a fresh --out per run):
  setsid nohup .venv/bin/python train.py --task planner --prior runs/prior_v1/prior.pt \
      --out runs/planner_v1 --hours 4 > runs/planner_v1.log 2>&1 < /dev/null &
Out: <out>/record.json, progress.csv, checkpoint.pt (every 10 min and at the end; the planner's also holds the
style reward's normaliser), policy.pt (inference weights).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import sprint
import torch
from sprint.recipes.ppo import PPO
from sprint.recipes.study import default_ppo

from gp import data
from gp.tasks import PlanConfig, PlanTracking, TerrainImitation, recipe_config


def build(task_name: str, D: dict, worlds: int, seed: int, device: str, prior: str = "", starts: list | None = None):
    model = data.load_model(device)
    c = recipe_config()
    task = (PlanTracking(D, c, PlanConfig(prior=prior), starts=starts, seed=seed) if task_name == "planner"
            else TerrainImitation(D, c, starts=starts, seed=seed))
    opts = sprint.StepOptions(dt_ctrl=1.0 / data.HZ, n_sub=c.physics_steps, implicit_friction=True,
                              implicit_normal=c.implicit_normal, strict=False)
    env = sprint.VecEnv(model, worlds, task=task, seed=seed, step_options=opts, device=device, delays=c.delays)
    return env, task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("baseline", "planner"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--prior", default="")
    ap.add_argument("--worlds", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True)                       # a fresh folder per run, never overwrite one
    torch.manual_seed(a.seed)
    D = data.load("train")
    env, task = build(a.task, D, a.worlds, a.seed, a.device, a.prior)
    cfg = default_ppo()
    cfg = replace(cfg, symmetry_coef=cfg.symmetry_coef if task.symmetric else 0.0)
    agent = PPO(env, cfg, device="cuda" if env.device.is_cuda else "cpu")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parent, capture_output=True,
                                text=True).stdout.strip())
    record = dict(args=vars(a), gait_planner_git=git, dirty=dirty, sprint=sprint.__version__, torch=torch.__version__,
                  split=json.loads((data.OUT / "split.json").read_text()), clips=len(D["name"]),
                  imitation=task.config.to_dict(), plan=asdict(task.plan) if a.task == "planner" else None,
                  ppo={k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
                  specific_tension_pa=data.SPECIFIC_TENSION, symmetric=task.symmetric, obs_dim=task.obs_dim, actor_inputs=len(task.actor_columns()),
                  started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    print(task.describe(), flush=True)
    rows: list[dict] = []
    f = (out / "progress.csv").open("w", buffering=1)
    w: csv.DictWriter | None = None
    start = time.time()
    deadline = start + a.hours * 3600

    def log(s: dict) -> None:
        nonlocal w
        s = dict(s, seconds=round(time.time() - start))
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(s), extrasaction="ignore")
            w.writeheader()
        w.writerow(s)
        rows.append(s)
        if len(rows) % 20 == 1:
            keys = [k for k in ("task/imitation", "task/plan", "task/r_style", "task/joint_error_deg", "episode_length",
                                "terminated_frac", "steps_per_s") if k in s]
            print(f"{s['env_steps'] / 1e6:7.1f}M  {s['seconds']:6d} s  " + "  ".join(f"{k} {s[k]:.3g}" for k in keys), flush=True)

    def save() -> None:
        agent.save(str(out / "checkpoint.pt"))
        agent.save_inference(str(out / "policy.pt"))
        if a.task == "planner":
            torch.save(task.style.state_dict(), out / "style_norm.pt")

    last = time.time()
    while time.time() < deadline:
        agent.train(10, on_log=log, deadline=deadline)
        if time.time() - last > 600:
            save()
            last = time.time()
    save()
    record.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), env_steps=int(agent.env_steps),
                  train_seconds=round(time.time() - start), final=rows[-1] if rows else None)
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))


if __name__ == "__main__":
    main()
