"""The foot placement estimator fitted to human footsteps: AB01's clips (every split, walking, start-stop and standing,
level and ramps) labelled exactly as the simulation labels the policy's own landings (gp.pretrain
``clip_estimator_data``), with Watch Your Step's model and fit (gp.fpm). Reports the landing error on whole clips held
out of the fit, then fits the saved model on every clip.

Run: .venv/bin/python fit_fpm.py --out models/fpm_ab01.pt
Use: train_wys.py --foot-model models/fpm_ab01.pt (a fixed estimator: no bootstrap, no refits)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gp import fpm as FPM
from gp.pretrain import PretrainConfig
from train_wys import build


def err_cm(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        d = (model(x).view(-1, 2, 2) - y.view(-1, 2, 2)).norm(dim=-1)
    return {"median_cm": float(d.median() * 100), "mean_cm": float(d.mean() * 100), "p90_cm": float(d.quantile(0.9) * 100)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="ab01")
    a = ap.parse_args()
    env, task = build(1, 0, "cpu", None, PretrainConfig(), kind="pretrain", dataset=a.dataset)
    per = task.clip_estimator_windows()
    X, Y = torch.cat([x for x, _ in per]), torch.cat([y for _, y in per])
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(per), generator=g).tolist()
    n_val = len(per) // 5
    val, tr = order[:n_val], order[n_val:]
    xt, yt = torch.cat([per[i][0] for i in tr]), torch.cat([per[i][1] for i in tr])
    xv, yv = torch.cat([per[i][0] for i in val]), torch.cat([per[i][1] for i in val])
    held = FPM.fit(xt, yt, "cpu", seed=0)
    fixed = torch.tensor(PretrainConfig().bootstrap).view(1, 4)
    report = {"samples": len(X), "windows": len(per), "held_out_windows": n_val,
              "held_out": err_cm(held, xv, yv),
              "fixed_offsets_baseline": err_cm(lambda x: fixed.expand(len(x), 4), xv, yv)}
    model = FPM.fit(X, Y, "cpu", seed=0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
