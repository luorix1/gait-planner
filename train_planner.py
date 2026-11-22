"""Train the planner (gp/planner.py) on the train split's plans: AdamW, batch 512, gradient clip 1.0, EMA 0.995 every 10
steps after 5k (copied before), normalisers from 20k windows, half of every batch mirrored left to right. Logs the
held-out split's loss beside the training loss.

With --dataset ab01: every window of the walk/start-stop/stand modes, every 10th trial held out.

Run (GPU, detached; a fresh --out per run):
  setsid nohup .venv/bin/python train_planner.py --out runs/planner_net_v1 > runs/planner_net_v1.log 2>&1 < /dev/null &
Full-pose planner (--pose): plan + pelvis + every joint coordinate (gp.planner.POSE_CHANNELS); Bones SEED needs
--seed-cleared. Licence-clean baseline: GP_DATASET=v3 ... --pose --balance.

Out: <out>/planner.pt (latest; overwritten every 2k iterations), <out>/planner_best.pt (lowest held-out loss), <out>/progress.csv, <out>/tb (TensorBoard: tensorboard --logdir runs)
With --score-every N (--pose2): every N iterations <out>/ckpt_<it>.pt + a scoreboard.py row in <out>/scores.jsonl; at the end
<out>/planner_score_best.pt = the checkpoint with the best mean rank over the scoreboard's balanced metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
    ap.add_argument("--pose", action="store_true", help="full-pose planner: plan + pelvis + every joint coordinate")
    ap.add_argument("--pose2", action="store_true", help="full-pose planner v2 (gp.pose_planner): absolute heights, "
                    "foot points + contact losses, goal guidance, smoothed heading")
    ap.add_argument("--contact-w", type=float, default=0.1, help="--pose2: weight of each geometric loss")
    ap.add_argument("--fk-w", type=float, default=None, help="--pose2: weight of the FK-consistency loss "
                    "(default: --contact-w; it is ~200x stiffer than the denoising loss, 0.3 destabilised v5)")
    ap.add_argument("--pen-w", type=float, default=0.0, help="--pose2: weight of the terrain penetration loss")
    ap.add_argument("--qk-norm", action="store_true", help="--pose2: RMS-normalised attention queries/keys with no "
                    "learnable gain (bounds the logits; v5-v7 broke on attention logit growth, v8's gain let them creep)")
    ap.add_argument("--cosine", action="store_true", help="cosine learning-rate decay to 0 over --iters")
    ap.add_argument("--hist-noise", type=float, default=0.0, help="--pose2: history noise, per window uniform in "
                    "[0, this] normalised units (0 = off)")
    ap.add_argument("--goal-drop", type=float, default=0.1, help="--pose2: share of training windows without the goal")
    ap.add_argument("--balance-terrain", action="store_true", help="--pose2 with --balance: each terrain class by the "
                    "square root of its length too")
    ap.add_argument("--balance", action="store_true", help="batches by gp.data.clip_weights(D, 'sqrt'), not uniform")
    ap.add_argument("--seed-cleared", action="store_true", help="allow Bones SEED clips in a --pose planner "
                    "(its licence forbids motion generators unless cleared)")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                    help="bfloat16 autocast for the network (2x faster on a 4090; the weights and EMA stay fp32)")
    ap.add_argument("--segments", action="store_true", help="--pose2 v10: 9 segment poses as channels (gp.footfk.SegFK)")
    ap.add_argument("--hist-drop", type=float, default=0.0, help="--pose2 v10: share of windows with the history hidden; "
                    "sampling blends with/without it")
    ap.add_argument("--hist-noise-mode", choices=("white", "body"), default="white",
                    help="--pose2: per-channel noise (v9) or a smooth whole-body offset re-run through FK (v10)")
    ap.add_argument("--goal-horizon", action="store_true", help="--pose2 v10: goal at a random 0.4-1.2 s horizon + noise")
    ap.add_argument("--patch-res", type=float, default=0.1, help="--pose2: terrain patch cell (m); v10 0.05")
    ap.add_argument("--contact-surface", action="store_true", help="--pose2 v10: contact height over the exact terrain")
    ap.add_argument("--rand-terrain", type=float, default=0.0, help="--pose2 v10: share of windows on random terrain, "
                    "penetration loss only")
    ap.add_argument("--dropout", type=float, default=0.0, help="--pose2: attention + residual dropout")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (torch default 0.01)")
    ap.add_argument("--sample-steps", type=int, default=None, help="--pose2: DDIM steps when sampling (default all 50)")
    ap.add_argument("--score-every", type=int, default=0, help="--pose2: roll out the P0 scoreboard every N iterations "
                    "(keeps ckpt_<it>.pt; 0 = off)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True)                      # a fresh folder per run, never overwrite one
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16 and a.device.startswith("cuda"))
    if a.dataset:
        D = load_clips(a.dataset, "", a.modes)
        trial = [str(n).rsplit("_", 1)[0] for n in D["name"]]
        held = set(sorted(set(trial))[a.holdout - 1::a.holdout])
        Dtr, Dte = (data.take(D, [k for k, t in enumerate(trial) if (t in held) == h]) for h in (False, True))
        (out / "held_out_trials.txt").write_text("\n".join(sorted(held)) + "\n")
    else:
        Dtr, Dte = data.load("train"), data.load(("test_trial", "test_terrain"))
    if (a.pose or a.pose2) and not a.seed_cleared and "seed_bones" in set(map(str, Dtr["source"])):
        raise SystemExit("Bones SEED in a motion generator needs --seed-cleared (licence section 6)")
    if a.pose2:
        from gp import footfk
        from gp.pose_planner import PosePlanner
        from gp.pose_planner import Windows as Windows2
        fk = footfk.fit(Dtr, a.device, data.ROOT / "data" / "foot_fk.pt")
        if a.segments:
            from gp.pose_planner import SEG_FK
            footfk.fit_seg(Dtr, a.device, SEG_FK)
        planner = PosePlanner(layers=a.layers, heads=a.heads, head_dim=a.head_dim, contact_w=a.contact_w,
                              goal_drop=a.goal_drop, hist_noise=a.hist_noise, fk_w=a.fk_w, pen_w=a.pen_w,
                              qk_norm=a.qk_norm, qk_gain=False, segments=a.segments, hist_drop=a.hist_drop,
                              hist_noise_mode=a.hist_noise_mode, goal_h=a.goal_horizon, patch_res=a.patch_res,
                              contact_surface=a.contact_surface, rand_terrain=a.rand_terrain, dropout=a.dropout,
                              sample_steps=a.sample_steps).to(a.device).attach(fk)
        W = Windows2(Dtr, a.device, fk, a.balance, a.balance_terrain, planner.grid, planner.sfk)
        Wt = Windows2(Dte, a.device, fk, grid=planner.grid, sfk=planner.sfk)
    else:
        W, Wt = Windows(Dtr, a.device, a.pose, a.balance), Windows(Dte, a.device, a.pose)
    print(f"{len(W.anchors)} training windows, {len(Wt.anchors)} held out", flush=True)
    if not a.pose2:
        planner = Planner(layers=a.layers, heads=a.heads, head_dim=a.head_dim, c=W.c).to(a.device)
    planner.set_normaliser(*W.batch(20_000))
    board = None
    if a.score_every and a.pose2:
        from scoreboard import Scoreboard
        board, rows = Scoreboard(Dte, fk, a.device), []
    opt = torch.optim.AdamW(planner.net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters) if a.cosine else None
    gen = torch.Generator(device=a.device).manual_seed(1)
    log = csv.writer((out / "progress.csv").open("w", buffering=1))
    log.writerow(["iter", "seconds", "train_loss", "held_out_loss"])
    tb = SummaryWriter(out / "tb", flush_secs=30)
    t0, run, best = time.time(), 0.0, float("inf")
    for it in range(1, a.iters + 1):
        with amp:
            loss = planner.loss(*W.batch(a.batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(planner.net.parameters(), 1.0)
        opt.step()
        if sched:
            sched.step()
        if it <= 5_000:
            planner.ema.module.load_state_dict(planner.net.state_dict())
        elif it % 10 == 0:
            planner.ema.update_parameters(planner.net)
        run = 0.99 * run + 0.01 * float(loss.detach()) if it > 1 else float(loss.detach())
        if it % 100 == 0:
            tb.add_scalar("loss/train", run, it)
            for k, v in getattr(planner, "parts", {}).items():
                tb.add_scalar(f"parts/{k}", v, it)
        if it % 500 == 0 or it == a.iters:
            with torch.no_grad(), amp:
                held = float(planner.loss(*Wt.batch(2048, gen, mirrored=False)))
            log.writerow([it, round(time.time() - t0), round(run, 5), round(held, 5)])
            tb.add_scalar("loss/held_out", held, it)
            parts = "  ".join(f"{k} {v:.4f}" for k, v in getattr(planner, "parts", {}).items())
            print(f"{it:>7d}  {time.time() - t0:7.0f} s  train {run:.4f}  held-out {held:.4f}  {parts}", flush=True)
            if held < best:
                best = held
                planner.save(out / "planner_best.pt", iters=it, held_out=held, data=a.dataset or str(data.OUT),
                             mirrored=True, pose=a.pose, pose2=a.pose2)
        if board is not None and (it % a.score_every == 0 or it == a.iters):
            name = f"ckpt_{it:06d}.pt"
            planner.save(out / name, iters=it, data=a.dataset or str(data.OUT), mirrored=True, pose=a.pose, pose2=a.pose2)
            torch.cuda.empty_cache()
            try:
                r = dict(board(planner), checkpoint=name, iters=it)
            except torch.OutOfMemoryError:          # a shared GPU: skip this pass rather than lose the run
                r = None
                print(f"{it:>7d}  scoreboard skipped (out of GPU memory)", flush=True)
            torch.cuda.empty_cache()
            if r is not None:
                rows.append(r)
                with (out / "scores.jsonl").open("a") as f:
                    f.write(json.dumps(r) + "\n")
                for k, v in r["balanced"].items():
                    tb.add_scalar(f"score/{k}", v, it)
                print(f"{it:>7d}  scoreboard {r['balanced']}  ({r['rollout_s']} s)", flush=True)
        if it % 2000 == 0 or it == a.iters:
            planner.save(out / "planner.pt", iters=it, data=a.dataset or str(data.OUT), mirrored=True, pose=a.pose, pose2=a.pose2)
    if board is not None and rows:
        from scoreboard import select
        best = select(rows)
        shutil.copy2(out / best["checkpoint"], out / "planner_score_best.pt")
        print(f"scoreboard best by mean rank: {best['checkpoint']} -> planner_score_best.pt", flush=True)
    tb.close()


if __name__ == "__main__":
    main()
