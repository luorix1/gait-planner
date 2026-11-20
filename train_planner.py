"""Train the planner (gp/planner.py) on the train split's plans: AdamW, batch 512, gradient clip 1.0, EMA 0.995 every 10
steps after 5k (copied before), normalisers from 20k windows, half of every batch mirrored left to right. Logs the
held-out split's loss beside the training loss.

With --dataset ab01: every window of the walk/start-stop/stand modes, every 10th trial held out.

Run (GPU, detached; a fresh --out per run):
  setsid nohup .venv/bin/python train_planner.py --out runs/planner_net_v1 > runs/planner_net_v1.log 2>&1 < /dev/null &
Out: <out>/planner.pt (latest; overwritten every 2k iterations), <out>/progress.csv, <out>/tb (TensorBoard: tensorboard --logdir runs)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from gp import data
from gp.planner import Planner, Windows
from train import load_clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dataset", default="", help="train on data/<dataset> (all splits, train.py's load_clips) with "
                    "every --holdout-th trial held out for the held-out loss; default: data.OUT's train split")
    ap.add_argument("--modes", default="walk,start_stop,stand")
    ap.add_argument("--holdout", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True)                      # a fresh folder per run, never overwrite one
    torch.manual_seed(0)
    if a.dataset:
        D = load_clips(a.dataset, "", a.modes)
        trial = [str(n).rsplit("_", 1)[0] for n in D["name"]]
        held = set(sorted(set(trial))[a.holdout - 1::a.holdout])
        W = Windows(data.take(D, [k for k, t in enumerate(trial) if t not in held]), a.device)
        Wt = Windows(data.take(D, [k for k, t in enumerate(trial) if t in held]), a.device)
        (out / "held_out_trials.txt").write_text("\n".join(sorted(held)) + "\n")
    else:
        W = Windows(data.load("train"), a.device)
        Wt = Windows(data.load(("test_trial", "test_terrain")), a.device)
    print(f"{len(W.anchors)} training windows, {len(Wt.anchors)} held out", flush=True)
    planner = Planner(layers=a.layers, heads=a.heads, head_dim=a.head_dim).to(a.device)
    planner.set_normaliser(*W.batch(20_000))
    opt = torch.optim.AdamW(planner.net.parameters(), lr=a.lr)
    gen = torch.Generator(device=a.device).manual_seed(1)
    log = csv.writer((out / "progress.csv").open("w", buffering=1))
    log.writerow(["iter", "seconds", "train_loss", "held_out_loss"])
    tb = SummaryWriter(out / "tb", flush_secs=30)
    t0, run = time.time(), 0.0
    for it in range(1, a.iters + 1):
        loss = planner.loss(*W.batch(a.batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(planner.net.parameters(), 1.0)
        opt.step()
        if it <= 5_000:
            planner.ema.module.load_state_dict(planner.net.state_dict())
        elif it % 10 == 0:
            planner.ema.update_parameters(planner.net)
        run = 0.99 * run + 0.01 * float(loss.detach()) if it > 1 else float(loss.detach())
        if it % 100 == 0:
            tb.add_scalar("loss/train", run, it)
        if it % 500 == 0 or it == a.iters:
            with torch.no_grad():
                held = float(planner.loss(*Wt.batch(2048, gen, mirrored=False)))
            log.writerow([it, round(time.time() - t0), round(run, 5), round(held, 5)])
            tb.add_scalar("loss/held_out", held, it)
            print(f"{it:>7d}  {time.time() - t0:7.0f} s  train {run:.4f}  held-out {held:.4f}", flush=True)
        if it % 2000 == 0 or it == a.iters:
            planner.save(out / "planner.pt", iters=it, data=a.dataset or str(data.OUT), mirrored=True)
    tb.close()


if __name__ == "__main__":
    main()
