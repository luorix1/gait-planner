"""Can the training body reproduce a hard clip at all? Single-clip imitation with SPRINT's FlashSAC on
gp.tasks.TerrainImitation (the clip's own terrain, the recipe's reward, physics and motor delays), wired as
``sprint.recipes.study.Study.imitate`` wires it: the recipe's FlashSACConfig, ``policy.pt`` the best of the periodic
selections (every 5M steps, each clip from two fixed starts). Then an evaluation of that policy from each clip's first
frame: ``--episodes`` deterministic episodes per clip, each with the recipe's reset noise (0.02 rad, 0.1 rad/s).

eval.json, per clip: episode length against the playable length (the clip minus the recipe's 1 s lookahead), the
share that fell, joint-angle error (deg, RMS over the tracked rotations) and pelvis distance from the reference (m),
averaged over each episode's steps; the pelvis's highest rise above its start and the flight (both feet under 5 % of
body weight; the clip's own heel/toe flags for the reference), longest and total, against the clip's over the same
frames.

Learner and effort as STC's Rajagopal running test (STC configs/rajagopal/imitation/flash_sac/test_running.yaml, the
same model and recipe family): 150M frames, 2048 worlds, 4M replay, batch 8192, 1024 transitions per update, learning
from 400k transitions, metabolic weight 0.002 (STC found 0.02 pushes running toward a knee-dominant strategy; a
feasibility check should not be held back by effort). Not ported from STC: per-joint weights (ankle 3, MTP 0.25) and
error-weighted episode starts (the recipe has neither).

Run (GPU, detached; a fresh --out per run; clips from a dataset built by ``gp.data.build("hard", clips=[...])``,
e.g. scripts/hard/build.sh; scripts/hard/run.sh runs one clip):
  setsid nohup .venv/bin/python imitate_clip.py parc_platform_platform_005_w0 --frames 30e6 \
      --out runs/hard_clips/parc_platform_platform_005_w0 > runs/hard_clips/platform.log 2>&1 < /dev/null &
Out: <out>/record.json, progress.csv, tb/ (TensorBoard), checkpoint.pt (every 10 min and at the end), policy.pt,
eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import sprint
import torch
from sprint.recipes.flashsac import FlashSAC, FlashSACConfig, load_policy
from sprint.recipes.study import _Best, _selection_starts
from torch.utils.tensorboard import SummaryWriter

from gp import data
from gp.tasks import TerrainImitation, recipe_config

ROOT = Path(__file__).parent
AIR_BW = 0.05                     # a foot is down above this fraction of body weight (as PlanConfig.contact_bw)


class FromStart(TerrainImitation):
    """Fixed starts (``starts``) with the recipe's reset noise back on, so repeated episodes from one frame differ."""

    def reset(self, s: Any, worlds: Any) -> None:
        super().reset(s, worlds)
        idx = worlds.nonzero().squeeze(1)
        if idx.numel():
            sh, c = (idx.numel(), len(self.L.tracked)), self.config
            s.qpos[idx[:, None], self.tracked] += c.noise_position * torch.randn(sh, device=self.dev, generator=self.gen)
            s.qvel[idx[:, None], self.tracked] += c.noise_velocity * torch.randn(sh, device=self.dev, generator=self.gen)


def make_env(D: dict, worlds: int, seed: int, device: str, cls: type = TerrainImitation,
             starts: list | None = None, metabolic: float | None = None) -> tuple[Any, Any]:
    """train.build's env (same body, recipe, step options) for any TerrainImitation subclass and fixed starts;
    ``metabolic`` replaces the recipe's metabolic weight."""
    c = recipe_config() if metabolic is None else recipe_config(w_metabolic=metabolic)
    task = cls(D, c, starts=starts, seed=seed)
    opts = sprint.StepOptions(dt_ctrl=1.0 / data.HZ, n_sub=c.physics_steps, implicit_friction=True,
                              implicit_normal=c.implicit_normal, strict=False, average_fext=True)
    env = sprint.VecEnv(data.load_model(device), worlds, task=task, seed=seed, step_options=opts, device=device,
                        delays=c.delays)
    return env, task


def runs(flags: Any) -> tuple[int, int]:
    """Longest run of True and the count of True, ``flags`` ``[T]`` (numpy bool)."""
    d = np.diff(np.r_[0, flags.astype(np.int8), 0])
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max(initial=0)), int(flags.sum())


def evaluate(D: dict, policy: Path, episodes: int, seed: int, device: str) -> dict:
    n = len(D["name"])
    env, task = make_env(D, episodes * n, seed, device, FromStart, starts=[(k, 0) for k in range(n) for _ in range(episodes)])
    env.auto_reset = False
    act = load_policy(str(policy), env)
    obs = env.reset()
    s, N, dev = env.state, env.num_worlds, task.dev
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    fell, steps = torch.zeros_like(alive), torch.zeros(N, device=dev)
    jerr, perr = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    y0 = s.body_pos[:, task.root, 1].clone()
    ymax = y0.clone()
    air = []
    for _ in range(task.config.max_steps):
        _, _, term, trunc, _ = env.step(act(env.torch(obs)))
        obs = env.obs
        term, trunc = env.torch(term).bool(), env.torch(trunc).bool()
        a = alive.float()
        jerr += task.metrics["joint_error_deg"] * a
        perr += task.metrics["pelvis_distance"] * a
        ymax = torch.where(alive, torch.maximum(ymax, s.body_pos[:, task.root, 1]), ymax)
        air.append(((task.foot_force(s) <= AIR_BW * task.bw).all(1) & alive).cpu().numpy())
        steps += a
        fell |= alive & term
        alive &= ~(term | trunc)
        if not bool(alive.any()):
            break
    air_np, st = np.stack(air, 1), steps.cpu().numpy()
    dt, out = 1.0 / data.HZ, {}
    rl = data.record_layout(len(D["tracked"]))
    for k in range(n):
        w = slice(k * episodes, (k + 1) * episodes)
        span = int(task.clip_span[k])                                  # playable steps from frame 0
        f0 = int(D["start"][k])
        ref_y = D["record"][f0:f0 + span + 1, rl["root_pos"]][:, 1]
        ref_long, ref_tot = runs(~D["plan_contact"][f0 + 1:f0 + span + 1].any(1))
        sim = [runs(air_np[i]) for i in range(w.start, w.stop)]
        out[str(D["name"][k])] = dict(
            episodes=episodes, clip_s=round(int(D["frames"][k]) * dt, 3), playable_s=round(span * dt, 3),
            episode_s=round(float(st[w].mean()) * dt, 3), episode_frac=round(float(st[w].mean()) / span, 3),
            fell_frac=round(float(fell[w].float().mean()), 3),
            joint_error_deg=round(float((jerr[w] / steps[w].clamp_min(1)).mean()), 2),
            pelvis_error_m=round(float((perr[w] / steps[w].clamp_min(1)).mean()), 3),
            pelvis_rise_m=round(float((ymax[w] - y0[w]).mean()), 3), ref_pelvis_rise_m=round(float(ref_y.max() - ref_y[0]), 3),
            flight_longest_s=round(float(np.mean([a for a, _ in sim])) * dt, 3), ref_flight_longest_s=round(ref_long * dt, 3),
            flight_total_s=round(float(np.mean([b for _, b in sim])) * dt, 3), ref_flight_total_s=round(ref_tot * dt, 3),
            episode_steps=[int(x) for x in st[w]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=float, default=150e6, help="env steps; also the FlashSAC schedule's horizon")
    ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--data", default=str(ROOT / "data/hard"))
    ap.add_argument("--worlds", type=int, default=2048)
    ap.add_argument("--capacity", type=int, default=4_000_000, help="replay transitions (STC test_running: 4M)")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--transitions-per-update", type=int, default=1024)
    ap.add_argument("--learning-starts", type=int, default=400_000)
    ap.add_argument("--metabolic", type=float, default=0.002, help="metabolic weight (recipe 0.02)")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="a small learner and frequent selections (CPU plumbing test)")
    a = ap.parse_args()
    if not (a.frames or a.minutes):
        ap.error("give --frames and/or --minutes")
    out = Path(a.out)
    out.mkdir(parents=True)                       # a fresh folder per run, never overwrite one
    torch.manual_seed(a.seed)
    full = data.load("train", src=Path(a.data))
    names = list(full["name"])
    D = data.take(full, [names.index(c) for c in a.clips])
    env, task = make_env(D, a.worlds, a.seed, a.device, metabolic=a.metabolic)
    cfg = replace(FlashSACConfig(), batch_size=a.batch, transitions_per_update=a.transitions_per_update,
                  learning_starts=a.learning_starts)
    if a.frames:
        cfg = replace(cfg, total_frames=int(a.frames))
    if a.capacity:
        cfg = replace(cfg, capacity=a.capacity)
    if a.smoke:
        cfg = replace(cfg, capacity=min(cfg.capacity, 100_000), batch_size=256, learning_starts=2048,
                      actor_hidden_dim=128, critic_hidden_dim=128, compile_mode=None)
    dev = "cuda" if env.device.is_cuda else "cpu"
    agent = FlashSAC(env, cfg, device=dev)
    sel_env = make_env(D, 2 * len(D["name"]), a.seed, a.device, starts=_selection_starts(task.motions, task.config),
                       metabolic=a.metabolic)[0]
    best = _Best(out, sel_env, every=16 * a.worlds * cfg.rollout_steps if a.smoke else 5_000_000)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    record: dict[str, Any] = dict(
        args=vars(a), gait_planner_git=git, dirty=dirty, sprint=sprint.__version__, torch=torch.__version__,
        clips=list(a.clips), frames=[int(f) for f in D["frames"]], imitation=task.config.to_dict(),
        flashsac={k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
        replay_gb=round(agent.replay.memory_bytes() / 1e9, 2), obs_dim=task.obs_dim, action_dim=env.action_dim,
        specific_tension_pa=data.SPECIFIC_TENSION, symmetric=task.symmetric, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    print(task.describe(), f"\nreplay {record['replay_gb']} GB, obs {task.obs_dim}, actions {env.action_dim}", flush=True)
    f = (out / "progress.csv").open("w", buffering=1)
    tb = SummaryWriter(out / "tb", flush_secs=30)
    w: csv.DictWriter | None = None
    start = time.time()
    deadline = start + a.minutes * 60 if a.minutes else float("inf")
    stop = cfg.total_frames if a.frames else float("inf")
    shown = 0.0

    def log(s: dict) -> None:
        nonlocal w, shown
        s = dict(s, seconds=round(time.time() - start))
        for k, v in s.items():
            if isinstance(v, (int, float)) and k != "env_steps":
                tb.add_scalar(k, v, s["env_steps"])
        if not s.get("updates"):                  # warm-up rows lack the learner's columns
            return
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(s), extrasaction="ignore")
            w.writeheader()
        w.writerow(s)
        if time.time() - shown > 60:
            shown = time.time()
            keys = [k for k in ("task/imitation", "task/joint_error_deg", "episode_length", "terminated_frac",
                                "steps_per_s") if k in s]
            print(f"{s['env_steps'] / 1e6:7.1f}M  {s['seconds']:6d} s  " + "  ".join(f"{k} {s[k]:.3g}" for k in keys),
                  flush=True)

    last = time.time()
    while time.time() < deadline and agent.env_steps < stop:
        agent.train(10 if not a.smoke else 1, on_log=log, deadline=deadline)
        if agent.env_steps >= best.next:
            best(agent)
        if time.time() - last > 600:
            agent.save(str(out / "checkpoint.pt"))
            last = time.time()
    agent.save(str(out / "checkpoint.pt"))
    best(agent)                                   # the final policy competes too
    tb.close()
    kept = best.record
    record.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), env_steps=int(agent.env_steps),
                  train_seconds=round(time.time() - start), best=kept)
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    del env, task, sel_env, agent, best           # free the replay before the evaluation's env
    torch.cuda.empty_cache()
    ev = evaluate(D, out / "policy.pt", a.episodes, a.seed + 1, a.device)
    (out / "eval.json").write_text(json.dumps(dict(policy="policy.pt", best=kept, clips=ev), indent=1))
    print(json.dumps({k: {kk: v for kk, v in e.items() if kk != "episode_steps"} for k, e in ev.items()}, indent=1))


if __name__ == "__main__":
    main()
