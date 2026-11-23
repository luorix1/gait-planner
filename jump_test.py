"""How high can the training body jump? A reference-free maximal vertical jump, to tell a weak body from a policy that
does not try (2026-11-22: SEED box jumps, jumps over obstacles and broad jumps fail in imitate_clip.py with the push-off
extensors far from saturated and take-off at ~1.3 m/s where the clips need ~2.7-3.0).

Task (MaxJump): gp.tasks.TerrainImitation on one AB01 standing clip (ab01_mc050_w0) for its start states, physics,
motor delays and observation layout, with the reward replaced: each step pays the whole-body centre of mass's new
height record above the episode's first step, in cm, so an episode's return is its apex rise. No effort cost and no
tracking (a capacity test). An episode ends after 2 s, or if the pelvis drops under half its start height or tilts
past 60 deg. The policy kept is the one with the highest mean apex over fixed deterministic starts (imitate_clip's
selection favours not falling, which a jump that lands badly would lose). Learner: imitate_clip's FlashSAC.

Compare with countermovement jumps WITHOUT arm swing (hands on hips): the model has no arms.

eval.json: per deterministic episode the apex rise of the centre of mass (cm), its take-off vertical speed (m/s,
highest before the apex), the longest flight (both feet under 5 % of body weight) and the peak activation of the
main extensors over the episode; then their means.

Run (GPU, detached; a fresh --out per run):
  GP_DATASET=v5 setsid nohup .venv/bin/python jump_test.py --out runs/jump_test_v1 > runs/jump_test_v1.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import sprint
import torch
from sprint.recipes.flashsac import FlashSAC, FlashSACConfig, load_policy
from sprint.recipes.imitation import _qrot

from gp import data
from gp.tasks import TerrainImitation
from imitate_clip import AIR_BW, make_env, runs

ROOT = Path(__file__).parent
CLIP = "ab01_mc050_w0"
MAX_STEPS = 80                                  # 2 s at 40 Hz
EXTENSORS = ("vaslat", "vasmed", "vasint", "recfem", "soleus", "gasmed", "gaslat", "glmax1", "glmax2", "glmax3",
             "semimem", "bflh", "addmagIsch")


class MaxJump(TerrainImitation):
    """Reward = the centre of mass's new height record (cm); falls = pelvis under half its start height or tilted
    past 60 deg; 2 s episodes. Everything else (starts, physics, observation) is the recipe's."""

    def __init__(self, D: dict[str, Any], config: Any = None, **kw: Any) -> None:
        super().__init__(D, config, **kw)
        m = data.load_model()
        nb = len(m.body_names)
        self._mass = torch.as_tensor(np.asarray(m.host["body_mass"][:nb], np.float32))
        self._bcom = torch.as_tensor(np.asarray(m.host["body_com"], np.float32).reshape(-1, 3)[:nb])

    def setup(self, env: Any) -> None:
        super().setup(env)
        n = env.num_worlds
        self._mass, self._bcom = self._mass.to(self.dev), self._bcom.to(self.dev)
        self.com0, self.best, self.pel0 = (torch.zeros(n, device=self.dev) for _ in range(3))
        self.fresh = torch.ones(n, dtype=torch.bool, device=self.dev)

    def com(self, s: Any) -> torch.Tensor:
        p = s.body_pos + _qrot(s.body_quat, self._bcom.expand_as(s.body_pos))
        return (p * self._mass[:, None]).sum(1) / self._mass.sum()

    def reset(self, s: Any, worlds: Any) -> None:
        super().reset(s, worlds)
        self.fresh |= worlds.bool()

    def reward(self, s: Any) -> Any:
        super().reward(s)                                  # the recipe's bookkeeping (reference frame, sensing)
        y = self.com(s)[:, 1]
        pel = s.body_pos[:, self.root, 1]
        f = self.fresh
        self.com0, self.best, self.pel0 = torch.where(f, y, self.com0), torch.where(f, y, self.best), torch.where(f, pel, self.pel0)
        self.fresh = torch.zeros_like(f)
        gain = (y - self.best).clamp_min(0)
        self.best = torch.maximum(self.best, y)
        ey = torch.zeros_like(s.body_pos[:, self.root])
        ey[:, 1] = 1.0
        up = _qrot(s.body_quat[:, self.root], ey)[:, 1]
        self._fell = (pel < 0.5 * self.pel0) | (up < math.cos(math.radians(60.0)))
        self._out_of_time = (s.episode_step >= MAX_STEPS) | (self.frame >= self.clip_last[self.clip])
        self.metrics = dict(self.metrics, apex_cm=100 * (self.best - self.com0))
        return 100 * gain


def jump_env(D: dict, worlds: int, seed: int, device: str, starts: list | None = None) -> tuple[Any, Any]:
    env, task = make_env(D, worlds, seed, device, MaxJump, starts=starts, metabolic=0.0)
    c = task.config
    c.w_activation_cubed = c.w_activation_cubed_sum = c.w_balance = c.w_emg = 0.0
    c.max_steps = MAX_STEPS
    return env, task


def evaluate(D: dict, policy: Path, episodes: int, seed: int, device: str) -> dict[str, Any]:
    """Deterministic episodes from evenly spaced frames of the standing clip."""
    span = int(D["frames"][0]) - MAX_STEPS - 41
    env, task = jump_env(D, episodes, seed, device, starts=[(0, int(f)) for f in np.linspace(0, span, episodes)])
    env.auto_reset = False
    act = load_policy(str(policy), env)
    obs = env.reset()
    s, dev = env.state, task.dev
    alive = torch.ones(episodes, dtype=torch.bool, device=dev)
    ys, air, acts = [], [], []
    names = [str(n) for n in getattr(data.load_model(), "muscle_names", [])]
    ext = [i for i, n in enumerate(names) if n.rsplit("_", 1)[0] in EXTENSORS]
    for _ in range(MAX_STEPS):
        _, _, term, trunc, _ = env.step(act(env.torch(obs)))
        obs = env.obs
        ys.append(torch.where(alive, task.com(s)[:, 1], torch.nan).cpu().numpy())
        air.append(((task.foot_force(s) <= AIR_BW * task.bw).all(1) & alive).cpu().numpy())
        if ext:
            acts.append(torch.where(alive[:, None], s.activation[:, ext], 0.0).cpu().numpy())
        alive &= ~(env.torch(term).bool() | env.torch(trunc).bool())
        if not bool(alive.any()):
            break
    Y = np.stack(ys, 1)                                       # [E, T], NaN after an episode ends
    vy = np.diff(Y, axis=1) * data.HZ
    rows = []
    for e in range(episodes):
        y = Y[e][~np.isnan(Y[e])]
        k = int(np.argmax(y))
        v = vy[e][:max(k, 1)]
        row = dict(apex_cm=round(100 * float(y[k] - y[0]), 1),
                   takeoff_vy_mps=round(float(np.nanmax(v)) if np.isfinite(v).any() else 0.0, 2),
                   flight_s=round(runs(np.stack(air, 1)[e])[0] / data.HZ, 3))
        if ext:
            A = np.stack(acts, 1)[e]
            row["peak_activation"] = {names[i]: round(float(A[:, j].max()), 2) for j, i in enumerate(ext)}
        rows.append(row)
    mean = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("apex_cm", "takeoff_vy_mps", "flight_s")}
    return dict(mean=mean, best_apex_cm=max(r["apex_cm"] for r in rows), episodes=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=float, default=60e6)
    ap.add_argument("--worlds", type=int, default=2048)
    ap.add_argument("--capacity", type=int, default=2_000_000, help="replay transitions (2M ~6 GB at this obs size)")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--select-every", type=float, default=5e6)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="a small learner and frequent selections (plumbing test)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True)                                   # a fresh folder per run
    torch.manual_seed(a.seed)
    full = data.load(("test_trial", "test_terrain"))
    D = data.take(full, [list(full["name"]).index(CLIP)])
    env, task = jump_env(D, a.worlds, a.seed, a.device)
    cfg = replace(FlashSACConfig(), batch_size=a.batch, total_frames=int(a.frames), capacity=a.capacity)
    if a.smoke:
        cfg = replace(cfg, capacity=min(cfg.capacity, 100_000), batch_size=256, learning_starts=2048,
                      actor_hidden_dim=128, critic_hidden_dim=128, compile_mode=None)
    agent = FlashSAC(env, cfg, device="cuda" if env.device.is_cuda else "cpu")
    record: dict[str, Any] = dict(args=vars(a), clip=CLIP, sprint=sprint.__version__, obs_dim=task.obs_dim,
                                  action_dim=env.action_dim, replay_gb=round(agent.replay.memory_bytes() / 1e9, 2),
                                  started=time.strftime("%Y-%m-%d %H:%M:%S"), selections=[])
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"MaxJump on {CLIP}: obs {task.obs_dim}, actions {env.action_dim}, replay {record['replay_gb']} GB", flush=True)
    every = 20 * a.worlds if a.smoke else int(a.select_every)
    nxt, best, start, shown = every, -math.inf, time.time(), 0.0

    def log(s: dict) -> None:
        nonlocal shown
        if time.time() - shown > 60 and s.get("updates"):
            shown = time.time()
            keys = [k for k in ("task/apex_cm", "episode_length", "terminated_frac", "steps_per_s") if k in s]
            print(f"{s['env_steps'] / 1e6:7.1f}M  {time.time() - start:6.0f} s  " + "  ".join(f"{k} {s[k]:.3g}" for k in keys), flush=True)

    while agent.env_steps < cfg.total_frames:
        agent.train(1 if a.smoke else 10, on_log=log)
        if agent.env_steps >= nxt or agent.env_steps >= cfg.total_frames:
            nxt += every
            cand = out / "policy.candidate.pt"
            agent.save_inference(str(cand))
            ev = evaluate(D, cand, a.episodes, a.seed + 1, a.device)
            record["selections"].append(dict(env_steps=int(agent.env_steps), **ev["mean"]))
            print(f"  selection at {agent.env_steps / 1e6:.1f}M: {ev['mean']}", flush=True)
            if ev["mean"]["apex_cm"] > best:
                best = ev["mean"]["apex_cm"]
                cand.replace(out / "policy.pt")
            (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    agent.save(str(out / "checkpoint.pt"))
    del env, task, agent
    torch.cuda.empty_cache()
    ev = evaluate(D, out / "policy.pt", a.episodes, a.seed + 2, a.device)
    record.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), best_selection_apex_cm=best)
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    (out / "eval.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev["mean"]), "best episode apex", ev["best_apex_cm"], "cm", flush=True)


if __name__ == "__main__":
    main()
