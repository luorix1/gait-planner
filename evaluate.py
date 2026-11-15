"""Evaluate a trained policy on held-out clips: one deterministic episode per clip from its first frame.

Per episode: whether it completed (the clip's end or 10 s without falling), the fraction of the clip survived, and
the task's metrics averaged over the episode (the planner's contact, foothold, centre-of-mass and style terms; the
recipe's joint error for both). With ``--prior`` (default: the planner's own) both tasks are also scored by the
same style prior (SDS loss of the simulated motion, from its 14th step), so the baseline and the planner compare.

Run: .venv/bin/python evaluate.py runs/planner_v1 [--prior runs/prior_v1/prior.pt]
Out: <run>/eval.json, and a table per split and terrain class
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sprint.recipes.ppo import ActorCritic, PPOConfig, env_action_bounds

from gp import data
from gp.smp import Prior, W
from train import build


def policy(env, path: Path):
    """The trained actor, deterministic (``sprint.recipes.run.Run.policy`` without the recurrent branch)."""
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


def run_split(run: Path, rec: dict, split: str, per_class: int, scorer: Prior | None, device: str) -> list[dict]:
    full = data.load(split)
    rng = np.random.default_rng(0)
    cls = np.array(full["terrain_class"])
    pick = np.concatenate([rng.permutation(np.flatnonzero(cls == c))[:per_class] for c in sorted(set(cls))])
    D = data.take(full, np.sort(pick))
    n = len(D["name"])
    task_name = rec["args"]["task"]
    env, task = build(task_name, D, n, 0, device, rec["args"].get("prior", ""), starts=[(i, 0) for i in range(n)])
    if task_name == "planner":
        task.style.load_state_dict(torch.load(run / "style_norm.pt"))
        task.style.norm_samples = 0                     # frozen at the training run's normaliser
    act = policy(env, run / "policy.pt")
    obs = env.reset()
    dev = task.dev
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    steps = torch.zeros(n, device=dev)
    sums: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(n, device=dev))
    ring = torch.zeros(n, W + 1, D["record"].shape[1], device=dev)
    sds, sds_n = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    possible = torch.minimum(task.clip_span.float(), torch.full_like(task.clip_span.float(), task.config.max_steps))
    for t in range(task.config.max_steps + 1):
        obs, _, term, trunc, _ = env.step(act(obs))
        live = ~done
        steps += live.float()
        for k, v in task.metrics.items():
            sums[k] += torch.where(live, v.float(), torch.zeros_like(v.float()))
        if scorer is not None:
            ring = torch.cat([ring[:, 1:], task.record(env.state)[:, None]], 1)
            if t >= W:
                loss = scorer.sds(scorer.features(ring)).mean(1)
                ok = live & (env.torch(term) == 0)
                sds += torch.where(ok, loss, torch.zeros_like(loss))
                sds_n += ok.float()
        term, trunc = env.torch(term).bool(), env.torch(trunc).bool()
        fell |= live & term
        done |= term | trunc
        if bool(done.all()):
            break
    out = []
    for i in range(n):
        row = dict(clip=str(D["name"][i]), terrain_class=str(D["terrain_class"][i]), subject=str(D["subject"][i]),
                   completed=not bool(fell[i]), survived=float(steps[i] / possible[i].clamp_min(1)),
                   **{k: float(v[i] / steps[i].clamp_min(1)) for k, v in sums.items()})
        if scorer is not None:
            row["style_sds_loss"] = float(sds[i] / sds_n[i]) if sds_n[i] > 0 else float("nan")
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--prior", default="")
    ap.add_argument("--splits", nargs="+", default=["test_subject", "test_terrain"])
    ap.add_argument("--per-class", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run = Path(a.run)
    rec = json.loads((run / "record.json").read_text())
    prior = a.prior or rec["args"].get("prior", "")
    scorer = Prior.load(prior, a.device) if prior else None
    report = {}
    for split in a.splits:
        rows = run_split(run, rec, split, a.per_class, scorer, a.device)
        report[split] = rows
        keys = ["completed", "survived"] + [k for k in ("joint_error_deg", "r_contact", "foothold_error", "com_error",
                                                        "r_style", "style_sds_loss") if k in rows[0]]
        print(f"\n{split}\n{'class':12s} {'n':>3s} " + " ".join(f"{k:>15s}" for k in keys))
        for c in sorted({r["terrain_class"] for r in rows}):
            rs = [r for r in rows if r["terrain_class"] == c]
            print(f"{c:12s} {len(rs):3d} " + " ".join(f"{np.nanmean([r[k] for r in rs]):15.3f}" for k in keys))
    report["prior"] = prior
    (run / "eval.json").write_text(json.dumps(report, indent=1))
    print("->", run / "eval.json")


if __name__ == "__main__":
    main()
