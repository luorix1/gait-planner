"""A distill.py student alone from the first frame of every window it was distilled on (one world each, deterministic,
one episode): falls, the share of the window played and joint error, by mode and terrain class. With ``--planner``
a --plan student reads that planner's plans in the loop (gp.planner.Replanner) instead of the clip's.

Run (GPU): .venv/bin/python eval_windows.py runs/distill_ab01_plan [--planner runs/planner_ab01/planner.pt]
Out: printed table, and <run>/eval_windows[_planner].json
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import replace
from pathlib import Path

import torch
from sprint.recipes.study import default_ppo

from distill import StudentView, student_inputs, student_loc, student_net
from gp.pretrain import PretrainConfig
from train import build, load_clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--planner", default="")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run, dev = Path(a.run), a.device
    args = json.loads((run / "record.json").read_text())["args"]
    D = load_clips(args["dataset"], args.get("trials", ""), args["modes"])
    n = len(D["name"])
    env, task = build("baseline", D, n, 0, dev, starts=[(k, 0) for k in range(n)])
    view = StudentView(None, D, replace(PretrainConfig(), teacher=False, warp=0, bootstrap_iters=0, refit=False),
                       seed=0, foot_model=args["foot_model"])
    view.bind(env.model)
    view.attach(env, task)
    extra, full = student_inputs(env, task, D, args.get("privileged", False), args.get("plan", False), a.planner)
    obs_dim = sum(b[1] for b in view.blocks) + extra
    net = student_net(view, env, replace(default_ppo(), rollout_steps=view.c.steps_per_iter), obs_dim, extra).to(dev)
    net.load_state_dict(torch.load(run / "checkpoint.pt", map_location=dev, weights_only=False)["net"])
    net.eval()
    obs = env.torch(env.reset())
    view.new_episode(torch.ones(n, dtype=torch.bool, device=dev))
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fell = torch.zeros_like(alive)
    steps, err = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    span = (task.clip_last - task.clip_start).float()[task.clip]
    with torch.no_grad():
        for _ in range(a.steps):
            x = full(view.observe(env.state), obs)
            obs, _, term, trunc, _ = env.step(net._squash(student_loc(net, x[:, net.actor_cols])))
            obs = env.torch(obs)
            view.prev_action = env.state.excitation.clone()
            err += task.metrics["joint_error_deg"] * alive
            steps += alive
            done = (env.torch(term).bool() | env.torch(trunc).bool()) & alive
            fell |= done & task._fell
            alive &= ~done
            view.new_episode(done)
            if not alive.any():
                break
    played = (steps / span.clamp(min=1)).clamp(max=1).cpu().numpy()
    fell_np, err_np = fell.cpu().numpy(), (err / steps.clamp(min=1)).cpu().numpy()
    g: dict[str, list[int]] = collections.defaultdict(list)
    for k in range(n):
        g[f"{D['mode'][k]} {D['terrain_class'][k]}"].append(k)
        g["ALL"].append(k)
    res = {key: dict(n=len(i), fell=float(fell_np[i].mean()), played=float(played[i].mean()),
                     joint_error_deg=float(err_np[i].mean())) for key, i in sorted(g.items())}
    print(f"{'group':24s} {'n':>4s} {'fell':>6s} {'played':>7s} {'joint err':>9s}")
    for key, r in res.items():
        print(f"{key:24s} {r['n']:4d} {100 * r['fell']:5.0f}% {100 * r['played']:6.0f}% {r['joint_error_deg']:8.1f}°")
    out = run / ("eval_windows_planner.json" if a.planner else "eval_windows.json")
    out.write_text(json.dumps(dict(planner=a.planner, groups=res,
                                   windows={str(D["name"][k]): dict(fell=bool(fell_np[k]), played=float(played[k]))
                                            for k in range(n)}), indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
