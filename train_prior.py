"""Train the style prior (SMP, gp/smp.py) on the train split's real clips, with MimicKit's settings: AdamW 2e-4,
batch 512, gradient clip 1.0, 200k iterations, EMA 0.995 every 10 steps after 5k (copied before), normaliser from
20k windows. Half of every batch is mirrored left to right (gp.smp.mirror_records). Logs the held-out subjects' loss
beside the training loss.

Subsets (the gate's checks, check_gate.py): ``--fold K/N`` keeps the motion groups (gp.data.motion_group) whose crc32 is
K mod N, so a clip's score can come from a prior that never saw it; ``--max-grade G`` drops ramp clips steeper than G deg;
``--drop-grade LO,HI`` drops clips with LO < |grade| <= HI (the interpolation test: 2.5,7.5 leaves 0 and 10 deg).

``--cond``: the conditional prior (gp.smp.conditions: grade, speed, terrain class, mode per clip; 10 % of labels
dropped to the null label).

``--balance``: every recording (gp.data.motion_group) is equally likely in a batch, however long it is, instead of every
frame. By frames, AB01's 8 long start-stop trials are 40 % of the data and PARC's 5 gap jumps 1 %; by recording 6 % and
4 %.

Run: .venv/bin/python train_prior.py --out runs/prior_v1 [--balance]
Out: <out>/prior.pt (latest; overwritten every 2k iterations), <out>/progress.csv, <out>/tb (TensorBoard: tensorboard --logdir runs)
"""

from __future__ import annotations

import argparse
import csv
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from gp import data
from gp.smp import N_COND, Prior, W, conditions, mirror_maps, mirror_records


def windows(D: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Style records on the device, and every frame that ends a full window (``W + 1`` records of one clip)."""
    rec = torch.as_tensor(D["record"], device=device)
    ends = np.concatenate([np.arange(s + W, s + n) for s, n in zip(D["start"], D["frames"], strict=True)])
    return rec, torch.as_tensor(ends, device=device)


def recording_weights(D: dict, device: str) -> torch.Tensor:
    """A sampling weight per window end (as :func:`windows` lists them): each recording's ends sum to 1."""
    g = np.concatenate([[data.motion_group(str(c))] * max(0, n - W) for c, n in zip(D["name"], D["frames"], strict=True)])
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    return torch.as_tensor(1.0 / cnt[inv], device=device, dtype=torch.float32)


def frame_labels(D: dict, device: str) -> torch.Tensor:
    """gp.smp.conditions per frame (each frame takes its clip's labels), indexed like the records."""
    return torch.as_tensor(np.repeat(conditions(D), D["frames"], 0), device=device)


def batch(rec: torch.Tensor, ends: torch.Tensor, n: int, gen: torch.Generator | None = None,
          mirror: dict | None = None, weights: torch.Tensor | None = None, lab: torch.Tensor | None = None) -> Any:
    """``n`` windows ``[n, W + 1, D]``, uniform over ends or drawn by ``weights``; with ``mirror``
    (gp.smp.mirror_maps) a random half reflected left to right; with ``lab`` (:func:`frame_labels`), (windows, their
    labels) (a mirrored window keeps its labels)."""
    k = (torch.multinomial(weights, n, replacement=True, generator=gen) if weights is not None
         else torch.randint(len(ends), (n,), device=ends.device, generator=gen))
    i = ends[k]
    w = rec[i[:, None] + torch.arange(-W, 1, device=ends.device)]
    if mirror is not None:
        flip = torch.rand(n, device=w.device, generator=gen) < 0.5
        w[flip] = mirror_records(w[flip], mirror)
    return w if lab is None else (w, lab[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--src", default=str(data.OUT))
    ap.add_argument("--fold", default="", help="K/N: train on the K-th of N motion-group folds")
    ap.add_argument("--max-grade", type=float, default=None, help="drop clips on ground steeper than this (deg)")
    ap.add_argument("--balance", action="store_true", help="every recording equally likely, however long")
    ap.add_argument("--drop-grade", default="", help="LO,HI: drop clips with LO < |grade| <= HI deg")
    ap.add_argument("--cond", action="store_true", help="conditional prior (gp.smp.conditions)")
    ap.add_argument("--min-std", type=float, default=None, help="normaliser std floor (default gp.smp.MIN_STD; MimicKit 0.2)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True)                      # a fresh folder per run, never overwrite one
    torch.manual_seed(0)
    D, Dt = data.load("train", Path(a.src)), data.load(data.HELD_OUT, Path(a.src))
    keep = np.ones(len(D["name"]), bool)
    if a.fold:
        k, n = (int(v) for v in a.fold.split("/"))
        keep &= np.array([zlib.crc32(data.motion_group(str(x)).encode()) % n == k for x in D["name"]])
    if a.max_grade is not None:
        keep &= np.abs(np.asarray(D["grade_deg"])) <= a.max_grade
    if a.drop_grade:
        lo, hi = (float(v) for v in a.drop_grade.split(","))
        g = np.abs(np.asarray(D["grade_deg"], float))
        keep &= ~((g > lo) & (g <= hi))
    D = data.take(D, np.flatnonzero(keep))
    print(f"{int(keep.sum())} of {len(keep)} training clips", flush=True)
    rec, ends = windows(D, a.device)
    rec_t, ends_t = windows(Dt, a.device)
    wts = recording_weights(D, a.device) if a.balance else None
    lab, lab_t = (frame_labels(D, a.device), frame_labels(Dt, a.device)) if a.cond else (None, None)
    prior = Prior(nt=len(D["tracked"]), dt=1.0 / D["hz"], n_cond=N_COND if a.cond else 0).to(a.device)
    maps = mirror_maps(data.load_model(), D["tracked"])
    from gp.smp import MIN_STD, window_features
    min_std = MIN_STD if a.min_std is None else a.min_std
    prior.set_normaliser(window_features(batch(rec, ends, 20_000, mirror=maps, weights=wts), prior.cfg["nt"], prior.cfg["dt"]),
                         min_std)
    opt = torch.optim.AdamW(prior.net.parameters(), lr=a.lr)
    gen = torch.Generator(device=a.device).manual_seed(1)
    log = csv.writer((out / "progress.csv").open("w", buffering=1))
    log.writerow(["iter", "seconds", "train_loss", "held_out_loss"])
    tb = SummaryWriter(out / "tb", flush_secs=30)
    t0, run = time.time(), 0.0
    for it in range(1, a.iters + 1):
        b = batch(rec, ends, a.batch, mirror=maps, weights=wts, lab=lab)
        loss = prior.loss(prior.features(b[0]), b[1]) if a.cond else prior.loss(prior.features(b))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.net.parameters(), 1.0)
        opt.step()
        if it <= 5_000:
            prior.ema.module.load_state_dict(prior.net.state_dict())
        elif it % 10 == 0:
            prior.ema.update_parameters(prior.net)
        run = 0.99 * run + 0.01 * float(loss.detach()) if it > 1 else float(loss.detach())
        if it % 100 == 0:
            tb.add_scalar("loss/train", run, it)
        if it % 1000 == 0 or it == a.iters:
            with torch.no_grad():
                bt = batch(rec_t, ends_t, 4096, gen, lab=lab_t)
                held = float(prior.loss(prior.features(bt[0]), bt[1], 0.0) if a.cond else prior.loss(prior.features(bt)))
            log.writerow([it, round(time.time() - t0), round(run, 5), round(held, 5)])
            tb.add_scalar("loss/held_out", held, it)
            print(f"{it:>7d}  {time.time() - t0:7.0f} s  train {run:.4f}  held-out {held:.4f}", flush=True)
        if it % 2000 == 0 or it == a.iters:
            prior.save(out / "prior.pt", iters=it, data=str(a.src), mirrored=True, fold=a.fold, max_grade=a.max_grade,
                       balance=a.balance, drop_grade=a.drop_grade, cond=a.cond, min_std=min_std)
    tb.close()


if __name__ == "__main__":
    main()
