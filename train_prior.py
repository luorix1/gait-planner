"""Train the style prior (SMP, gp/smp.py) on the train split's real clips, with MimicKit's settings: AdamW 2e-4,
batch 512, gradient clip 1.0, 200k iterations, EMA 0.995 every 10 steps after 5k (copied before), normaliser from
20k windows. Half of every batch is mirrored left to right (gp.smp.mirror_records). Logs the held-out subjects' loss
beside the training loss.

Run: .venv/bin/python train_prior.py --out runs/prior_v1
Out: <out>/prior.pt (latest; overwritten every 2k iterations), <out>/progress.csv, <out>/tb (TensorBoard: tensorboard --logdir runs)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from gp import data
from gp.smp import Prior, W, mirror_maps, mirror_records


def windows(D: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Style records on the device, and every frame that ends a full window (``W + 1`` records of one clip)."""
    rec = torch.as_tensor(D["record"], device=device)
    ends = np.concatenate([np.arange(s + W, s + n) for s, n in zip(D["start"], D["frames"], strict=True)])
    return rec, torch.as_tensor(ends, device=device)


def batch(rec: torch.Tensor, ends: torch.Tensor, n: int, gen: torch.Generator | None = None,
          mirror: dict | None = None) -> torch.Tensor:
    """``n`` windows ``[n, W + 1, D]``; with ``mirror`` (gp.smp.mirror_maps) a random half reflected left to right."""
    i = ends[torch.randint(len(ends), (n,), device=ends.device, generator=gen)]
    w = rec[i[:, None] + torch.arange(-W, 1, device=ends.device)]
    if mirror is not None:
        flip = torch.rand(n, device=w.device, generator=gen) < 0.5
        w[flip] = mirror_records(w[flip], mirror)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--src", default=str(data.OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True)                      # a fresh folder per run, never overwrite one
    torch.manual_seed(0)
    D, Dt = data.load("train", Path(a.src)), data.load(data.HELD_OUT, Path(a.src))
    rec, ends = windows(D, a.device)
    rec_t, ends_t = windows(Dt, a.device)
    prior = Prior(nt=len(D["tracked"]), dt=1.0 / D["hz"]).to(a.device)
    maps = mirror_maps(data.load_model(), D["tracked"])
    from gp.smp import window_features
    prior.set_normaliser(window_features(batch(rec, ends, 20_000, mirror=maps), prior.cfg["nt"], prior.cfg["dt"]))
    opt = torch.optim.AdamW(prior.net.parameters(), lr=a.lr)
    gen = torch.Generator(device=a.device).manual_seed(1)
    log = csv.writer((out / "progress.csv").open("w", buffering=1))
    log.writerow(["iter", "seconds", "train_loss", "held_out_loss"])
    tb = SummaryWriter(out / "tb", flush_secs=30)
    t0, run = time.time(), 0.0
    for it in range(1, a.iters + 1):
        loss = prior.loss(prior.features(batch(rec, ends, a.batch, mirror=maps)))
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
                held = float(prior.loss(prior.features(batch(rec_t, ends_t, 4096, gen))))
            log.writerow([it, round(time.time() - t0), round(run, 5), round(held, 5)])
            tb.add_scalar("loss/held_out", held, it)
            print(f"{it:>7d}  {time.time() - t0:7.0f} s  train {run:.4f}  held-out {held:.4f}", flush=True)
        if it % 2000 == 0 or it == a.iters:
            prior.save(out / "prior.pt", iters=it, data=str(a.src), mirrored=True)
    tb.close()


if __name__ == "__main__":
    main()
