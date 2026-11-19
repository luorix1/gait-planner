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
import math
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import sprint
import torch
from sprint.recipes.ppo import PPO
from sprint.recipes.study import default_ppo
from torch.utils.tensorboard import SummaryWriter

from gp import data
from gp import wys_terrain as WT
from gp.ladder import Ladder, LadderConfig
from gp.pretrain import Pretrain, PretrainConfig
from gp.tasks import recipe_config
from gp.wys import WatchYourStep, WysConfig

#: --gaitamp-conditions (ladder, diagnostic): GAIT_amp's walking runs' training conditions on our body and learner,
#: to split body + learner from conditions (b1_L1 parked its leg extensors; GAIT_amp's walkers did not). Plus flat
#: ground and no motor delays (build's ``flat``, ``delays``). Rewards stay ours (effort at its share).
GAITAMP = dict(n_lin=0.0, n_ang=0.0, n_grav=0.0, n_q=0.0, n_qd=0.0, spike_p=0.0, scan_noise=0.0, xy_drift=0.0, z_drift=0.0,
               disturbance_n=0.0, friction_scale=(1.0, 1.0), push_s=1e9,           # no pushes, friction spread, noise
               fall_tilt=1.05, fall_on_leg_contact=False, fall_height=0.6,         # their 60 deg, 0.6 m pelvis
               start_modes=("walk",), style_modes=("walk",),                       # walking clips only (their judge's)
               start_speed_in_range=True,                                          # start mid-stride at 0.5-1.5 m/s
               lin_x=(0.5, 1.5), resample_s=1e9, standing_frac=0.0, turning_frac=0.0)  # one straight speed per episode


class FloorPPO(PPO):
    """SPRINT's PPO plus a floor under each muscle's mean drive: ``floor_coef x (pre-squash floor - mean)^2`` wherever
    the actor's deterministic excitation is below ``floor``, added through the symmetry-loss hook (one call per actor
    minibatch; the logged ``symmetry`` includes it). Under tanh a mean at excitation 0.02 moves 15-40x less per update
    than one mid-range, and the b1 runs parked their leg extensors there for good (11-18); the floor keeps every muscle
    where learning can still raise it. 0 = off."""

    floor: float = 0.0
    floor_coef: float = 1.0

    def _symmetry_loss(self, obs: torch.Tensor) -> torch.Tensor:
        loss = super()._symmetry_loss(obs)
        if self.floor <= 0.0:
            return loss
        lo = self.cfg.excitation_min
        pre = math.atanh(2.0 * (self.floor - lo) / (1.0 - lo) - 1.0)
        below = (pre - self.net.actor(obs)).clamp(min=0.0).pow(2).mean()
        return loss + self.floor_coef / self.cfg.symmetry_coef * below


def build(worlds: int, seed: int, device: str, out: Path | None = None, config: WysConfig | None = None,
          foot_model: str | None = None, kind: str = "wys", dataset: str = "ab01", delays: bool = True, flat: bool = False,
          trials: tuple = ()):
    """``kind`` "wys" (gp.wys), "ladder" (gp.ladder, on ``dataset``'s training clips; ``config`` a LadderConfig) or
    "pretrain" (gp.pretrain, stage 0: imitation of the clips, always on levelled tiles); ``flat``: every tile levelled;
    ``trials``: only these trials' windows (e.g. "ab01_mc021"). Clips come from every split."""
    flat = flat or kind == "pretrain"
    T = WT.build()
    if flat:
        T.heights[:] = 0.0
        T.block[:] = np.nan
    if kind == "pretrain":
        WT.ramp_strips(T, (config or PretrainConfig()).ramp_columns())
    model = data.load_model(device)
    c = recipe_config()
    if kind in ("ladder", "pretrain"):
        cfg = config or (PretrainConfig() if kind == "pretrain" else LadderConfig())
        # every split (user 11-18: data is short, the final training uses all clips)
        D = data.load(("train", "test_trial", "test_terrain"), src=data.ROOT / "data" / dataset)
        if trials:
            D = data.take(D, [k for k, n in enumerate(D["name"]) if any(str(n).startswith(t + "_") for t in trials)])
        cls = Pretrain if kind == "pretrain" else Ladder
        task: WatchYourStep = cls(T, D, cfg, seed=seed, log_dir=out, foot_model=foot_model)
    else:
        cfg = config or WysConfig()
        task = WatchYourStep(T, cfg, seed=seed, log_dir=out, foot_model=foot_model)
    opts = sprint.StepOptions(dt_ctrl=cfg.dt, n_sub=int(round(cfg.dt * 3000)), implicit_friction=True,
                              implicit_normal=c.implicit_normal, strict=False, average_fext=True)
    env = sprint.VecEnv(model, worlds, task=task, seed=seed, step_options=opts, device=device,
                        delays=c.delays if delays else None, perturb_slots=2)
    return env, task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--steps", type=float, default=float("inf"), help="also stop at this many environment steps (total, resumes included)")
    ap.add_argument("--worlds", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", choices=("wys", "ladder", "pretrain"), default="wys")
    ap.add_argument("--perception", choices=("patches", "map"), default="patches", help="ladder: L1 patches, L0 map")
    ap.add_argument("--dataset", default="ab01", help="ladder: the clips (data/<dataset>)")
    ap.add_argument("--no-delays", action="store_true", help="no motor delays (default: the recipe's SCONE delays)")
    ap.add_argument("--gaitamp-conditions", action="store_true", help="ladder diagnostic: GAIT_amp's training conditions (GAITAMP)")
    ap.add_argument("--no-policy-command", action="store_true", help="ladder L0.5: the policy does not see the command")
    ap.add_argument("--trials", default="", help="ladder/pretrain: only these trials' windows, comma-separated (e.g. ab01_mc021)")
    ap.add_argument("--init-from", help="ladder: start the policy, critic and guesser from this run (e.g. a pretrain run)")
    ap.add_argument("--drive-floor", type=float, default=0.05, help="lowest mean excitation per muscle (FloorPPO; 0 = off)")
    a = ap.parse_args()
    out = Path(a.out)
    resume = (out / "checkpoint.pt").exists()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    config = ({"ladder": LadderConfig, "pretrain": PretrainConfig}[a.task](perception=a.perception, policy_command=not a.no_policy_command)
              if a.task in ("ladder", "pretrain") else None)
    if a.gaitamp_conditions:
        config = replace(config, **GAITAMP)
    env, task = build(a.worlds, a.seed, a.device, out, config, kind=a.task, dataset=a.dataset,
                      delays=not (a.no_delays or a.gaitamp_conditions), flat=a.gaitamp_conditions,
                      trials=tuple(t for t in a.trials.split(",") if t))
    cfg = replace(default_ppo(), rollout_steps=task.c.steps_per_iter)
    agent = FloorPPO(env, cfg, device="cuda" if env.device.is_cuda else "cpu")
    agent.floor = a.drive_floor
    if resume:
        agent.load(str(out / "checkpoint.pt"))
        ts = torch.load(out / "task_state.pt", map_location=task.dev, weights_only=False)
        task.level[:] = ts["level"].to(task.dev)
        task.steps, task.fpm_ready = int(ts["steps"]), bool(ts["fpm_ready"])
        task.fpm.load_state_dict(ts["fpm"])
        if "amp" in ts:
            task.amp.load_state_dict(ts["amp"])
        print(f"resumed {out} at iteration {task.steps // task.c.steps_per_iter}, {agent.env_steps / 1e6:.0f}M steps", flush=True)
    elif a.init_from:                       # weights only: a fresh optimiser and step count for the new reward
        src = Path(a.init_from)
        agent.net.load_state_dict(torch.load(src / "checkpoint.pt", map_location=task.dev, weights_only=False)["net"])
        ts = torch.load(src / "task_state.pt", map_location=task.dev, weights_only=False)
        task.fpm.load_state_dict(ts["fpm"])
        task.fpm_ready = bool(ts["fpm_ready"])
        print(f"initialised from {src}", flush=True)
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
            keys = [k for k in ("task/level", "task/imitation", "task/joint_error_deg", "task/r_speed", "task/r_heading", "task/r_track_lin_vel_xy", "task/r_style", "task/metabolic",
                                "task/amp_accuracy", "task/effort", "task/assist", "episode_length",
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
    while time.time() < deadline and agent.env_steps < a.steps:
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
