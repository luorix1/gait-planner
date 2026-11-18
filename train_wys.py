"""Train the Watch Your Step replication (gp.wys) with SPRINT's PPO: our learner as tuned for the muscle body
(sprint.recipes.study.default_ppo), with 24 steps per iteration so the predictor's 500-iteration schedule matches the
paper's. Physics: our body as the imitation recipe runs it (implicit contact, SCONE motor delays), at the paper's
50 Hz control with 60 physics steps (3 kHz).

``--task ladder`` trains a rung of the footstep planner ladder instead (gp.ladder, build B1: muscle inputs, AMP
style reward from AB01 clips, clip-frame starts, the recipe's effort terms, 40 Hz with 75 physics steps):
``--perception patches`` is L1 (the paper's blind guesser and patches), ``--perception map`` is L0 (the whole map).

Run (GPU, detached):
  setsid nohup .venv/bin/python train_wys.py --out runs/wys_v1 --hours 12 > runs/wys_v1.log 2>&1 < /dev/null &
  setsid nohup .venv/bin/python train_wys.py --task ladder --perception patches --out runs/b1_L1 --hours 12 \
      > runs/b1_L1.log 2>&1 < /dev/null &
An --out that holds a checkpoint.pt is continued (the PPO state, the terrain levels, the predictor and its schedule),
so a job cut off by a cluster's time limit resumes when submitted again.
Out: <out>/record.json, progress.csv, tb/, checkpoint.pt and policy.pt (every 10 min and at the end),
foot_model_<iteration>.pt (each refit), task_state.pt (terrain levels, the current predictor, the style judge).
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
from torch.utils.tensorboard import SummaryWriter

from gp import data
from gp import wys_terrain as WT
from gp.ladder import Ladder, LadderConfig
from gp.tasks import recipe_config
from gp.wys import WatchYourStep, WysConfig


def build(worlds: int, seed: int, device: str, out: Path | None = None, config: WysConfig | None = None,
          foot_model: str | None = None, kind: str = "wys", dataset: str = "ab01", delays: bool = True):
    """``kind`` "wys" (gp.wys) or "ladder" (gp.ladder, on ``dataset``'s training clips; ``config`` a LadderConfig)."""
    model = data.load_model(device)
    c = recipe_config()
    if kind == "ladder":
        cfg = config or LadderConfig()
        D = data.load("train", src=data.ROOT / "data" / dataset)
        task: WatchYourStep = Ladder(WT.build(), D, cfg, seed=seed, log_dir=out, foot_model=foot_model)
    else:
        cfg = config or WysConfig()
        task = WatchYourStep(WT.build(), cfg, seed=seed, log_dir=out, foot_model=foot_model)
    opts = sprint.StepOptions(dt_ctrl=cfg.dt, n_sub=int(round(cfg.dt * 3000)), implicit_friction=True,
                              implicit_normal=c.implicit_normal, strict=False, average_fext=True)
    env = sprint.VecEnv(model, worlds, task=task, seed=seed, step_options=opts, device=device,
                        delays=c.delays if delays else None, perturb_slots=2)
    return env, task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--worlds", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", choices=("wys", "ladder"), default="wys")
    ap.add_argument("--perception", choices=("patches", "map"), default="patches", help="ladder: L1 patches, L0 map")
    ap.add_argument("--dataset", default="ab01", help="ladder: the clips (data/<dataset>)")
    ap.add_argument("--no-delays", action="store_true", help="no motor delays (default: the recipe's SCONE delays)")
    a = ap.parse_args()
    out = Path(a.out)
    resume = (out / "checkpoint.pt").exists()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    config = LadderConfig(perception=a.perception) if a.task == "ladder" else None
    env, task = build(a.worlds, a.seed, a.device, out, config, kind=a.task, dataset=a.dataset, delays=not a.no_delays)
    cfg = replace(default_ppo(), rollout_steps=task.c.steps_per_iter)
    agent = PPO(env, cfg, device="cuda" if env.device.is_cuda else "cpu")
    if resume:
        agent.load(str(out / "checkpoint.pt"))
        ts = torch.load(out / "task_state.pt", map_location=task.dev, weights_only=False)
        task.level[:] = ts["level"].to(task.dev)
        task.steps, task.fpm_ready = int(ts["steps"]), bool(ts["fpm_ready"])
        task.fpm.load_state_dict(ts["fpm"])
        if "amp" in ts:
            task.amp.load_state_dict(ts["amp"])
        print(f"resumed {out} at iteration {task.steps // task.c.steps_per_iter}, {agent.env_steps / 1e6:.0f}M steps", flush=True)
    here = Path(__file__).parent
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True).stdout.strip())
    record = dict(args=vars(a), gait_planner_git=git, dirty=dirty, sprint=sprint.__version__, torch=torch.__version__,
                  wys=asdict(task.c), ppo={k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
                  terrain=dict(types=task.T.types, cell=task.T.cell), obs_dim=task.obs_dim,
                  actor_inputs=int(task.n_actor), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if resume and (out / "record.json").exists():
        first = json.loads((out / "record.json").read_text())
        first.setdefault("resumes", []).append(record)
        record = first
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    print(task.describe(), flush=True)
    new_csv = not (out / "progress.csv").exists()
    f = (out / "progress.csv").open("a", buffering=1)
    tb = SummaryWriter(out / "tb", flush_secs=30)
    w: csv.DictWriter | None = None
    start = time.time()
    deadline = start + a.hours * 3600
    rows: list[dict] = []

    def log(s: dict) -> None:
        nonlocal w
        s = dict(s, seconds=round(time.time() - start), iteration=task.steps // task.c.steps_per_iter)
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(s), extrasaction="ignore")
            if new_csv:
                w.writeheader()
        w.writerow(s)
        rows.append(s)
        for k, v in s.items():
            if isinstance(v, (int, float)) and k != "env_steps":
                tb.add_scalar(k, v, s["env_steps"])
        if len(rows) % 20 == 1:
            keys = [k for k in ("task/level", "task/r_track_lin_vel_xy", "task/r_style", "task/metabolic",
                                "task/amp_accuracy", "task/assist", "episode_length",
                                "terminated_frac", "steps_per_s") if k in s]
            print(f"it {s['iteration']:6d}  {s['env_steps'] / 1e6:7.1f}M  {s['seconds']:6d} s  "
                  + "  ".join(f"{k} {s[k]:.3g}" for k in keys), flush=True)

    def save() -> None:
        agent.save(str(out / "checkpoint.pt"))
        agent.save_inference(str(out / "policy.pt"))
        ts = dict(level=task.level.cpu(), steps=task.steps, fpm=task.fpm.state_dict(), fpm_ready=task.fpm_ready)
        if isinstance(task, Ladder):
            ts["amp"] = task.amp.state_dict()
        torch.save(ts, out / "task_state.pt")

    last = time.time()
    while time.time() < deadline:
        agent.train(10, on_log=log, deadline=deadline)
        if time.time() - last > 600:
            save()
            last = time.time()
    save()
    tb.close()
    record.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), env_steps=int(agent.env_steps),
                  iterations=task.steps // task.c.steps_per_iter, train_seconds=round(time.time() - start),
                  final=rows[-1] if rows else None)
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))


if __name__ == "__main__":
    main()
