"""Evaluate a trained policy on held-out clips: one deterministic episode per clip from its first frame.

Per episode: whether it completed (the clip's end or 10 s without falling), the fraction of the clip survived, and
the task's metrics averaged over the episode (the planner's contact, foothold, centre-of-mass and style terms; the
recipe's joint error for both). With ``--prior`` (default: the planner's own) both tasks are also scored by the
same style prior (SDS loss of the simulated motion, from its 14th step), so the baseline and the planner compare.

Each episode also runs through the acceptance filter (gp/accept.py, thresholds from ``data/<dataset>/accept.json``): the
six checks, accepted or not, the numbers behind each, and G1's share of the right-stance moment and force curves
inside the real band. The style check needs ``accept.json`` calibrated with the same prior. The activations of the
muscles Camargo's EMG measured are saved per right stride (``emg_strides.npz``) for ``emg_report.py``.

Run: .venv/bin/python evaluate.py runs/planner_v1 [--prior runs/prior_v1/prior.pt]
Out: <run>/eval.json, <run>/emg_strides.npz, and a table per split and terrain class
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sprint.recipes.ppo import ActorCritic, PPOConfig, env_action_bounds

from camargo_emg import CHANNELS, MUSCLES
from gp import accept, data, emg
from gp.smp import Prior, W
from gp.tasks import PlanConfig
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


def run_split(run: Path, rec: dict, split: str, per_class: int, scorer: Prior | None, device: str,
              thresholds: dict | None) -> tuple[list[dict], list[tuple[np.ndarray, str]]]:
    full = data.load(split)
    rng = np.random.default_rng(0)
    cls = np.array(full["terrain_class"])
    pick = np.concatenate([rng.permutation(np.flatnonzero(cls == c))[:per_class] for c in sorted(set(cls))])
    D = data.take(full, np.sort(pick))
    n = len(D["name"])
    task_name = rec["args"]["task"]
    env, task = build(task_name, D, n, 0, device, rec["args"].get("prior", ""), starts=[(i, 0) for i in range(n)],
                      plan=rec.get("plan"))
    env.auto_reset = False          # a world's last step is read after the step, so it must not be reset to its start
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
    # the acceptance filter's trace, per step (gp.accept.check)
    from sprint import _backend as B
    from sprint.recipes.imitation import _heading
    m, s = env.model, env.state
    nv, bw, mass = int(m.nv), float(env.total_mass) * 9.80665, float(env.total_mass)
    jcol = torch.as_tensor([list(m.coord_names).index(j) for j in accept.JOINTS], device=dev)
    lim = m.host["limit_dof"][: int((m.host["limit_dof"] >= 0).sum())].astype(int)
    lim_dof = torch.as_tensor(lim, device=dev)
    lim_lo = torch.as_tensor(m.host["limit_lo"][: len(lim)], dtype=torch.float32, device=dev) + accept.LIMIT_MARGIN
    lim_hi = torch.as_tensor(m.host["limit_hi"][: len(lim)], dtype=torch.float32, device=dev) - accept.LIMIT_MARGIN
    band = torch.as_tensor(task.plan.com_band if task_name == "planner" else PlanConfig().com_band, device=dev)
    mn = list(m.muscle_names)
    emg_w = torch.zeros(m.nmus, len(CHANNELS), device=dev)               # channel = mean of its muscles' activations
    for k, ch in enumerate(CHANNELS):
        for mus in MUSCLES[ch]:
            emg_w[mn.index(mus), k] = 1.0 / len(MUSCLES[ch])
    trace: dict[str, list] = defaultdict(list)
    path_len, metab_j = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    last_root = s.body_com_pos()[:, task.root, [0, 2]].clone()
    for t in range(task.config.max_steps + 1):
        obs, _, term, trunc, _ = env.step(act(obs))
        live = ~done
        steps += live.float()
        for k, v in task.metrics.items():
            sums[k] += torch.where(live, v.float(), torch.zeros_like(v.float()))
        loss = torch.full((n,), float("nan"), device=dev)
        if scorer is not None:
            ring = torch.cat([ring[:, 1:], task.record(env.state)[:, None]], 1)
            if t >= W:
                loss = scorer.sds(scorer.features(ring)).mean(1)
                ok = live & (env.torch(term) == 0)
                sds += torch.where(ok, loss, torch.zeros_like(loss))
                sds_n += ok.float()
        f = task.frame
        sph = task.sphere_bottoms(s)
        q = s.qpos
        root = s.body_com_pos()[:, task.root]
        M = _heading(s.body_quat[:, task.root])
        com_err = torch.einsum("nij,nj->ni", M, task.com(s) - task.p_com[f]).abs()
        step = dict(sds=loss, plan_contact=task.p_contact[f], foothold_ref=task.p_foothold[f],
                    com_out=(com_err > band).any(1), fv=task.foot_force(s) / bw, foot=task.foot_points(s),
                    moment=B.to_torch(env.data.qfrc_total)[: nv * n].view(nv, n).T[:, jcol] / mass,
                    act_sat=s.activation > accept.SAT,
                    at_limit=((q[:, lim_dof] < lim_lo) | (q[:, lim_dof] > lim_hi)).any(1),
                    sphere=sph, touching=sph[..., 1] - task.ground(sph[..., 0], sph[..., 2]) < accept.TOUCH_M,
                    emg_act=s.activation @ emg_w)
        for k, v in step.items():
            trace[k].append(v.cpu())
        path_len += torch.where(live, (root[:, [0, 2]] - last_root).norm(dim=1), torch.zeros_like(path_len))
        metab_j += torch.where(live, s.metabolic_power.sum(1) * env.dt_ctrl, torch.zeros_like(metab_j))
        last_root = root[:, [0, 2]].clone()
        term, trunc = env.torch(term).bool(), env.torch(trunc).bool()
        fell |= live & term
        done |= term | trunc
        if bool(done.all()):
            break
    out, strides = [], []
    for i in range(n):
        row = dict(clip=str(D["name"][i]), terrain_class=str(D["terrain_class"][i]), subject=str(D["subject"][i]),
                   mode=str(D["mode"][i]),
                   completed=not bool(fell[i]), survived=float(steps[i] / possible[i].clamp_min(1)),
                   **{k: float(v[i] / steps[i].clamp_min(1)) for k, v in sums.items()})
        if scorer is not None:
            row["style_sds_loss"] = float(sds[i] / sds_n[i]) if sds_n[i] > 0 else float("nan")
        k = int(steps[i])
        tr = {key: torch.stack(v)[:k, i].numpy() for key, v in trace.items()}
        tr.update(fell=bool(fell[i]), metabolic_j_per_kg_m=float(metab_j[i] / mass / path_len[i].clamp_min(1e-3)))
        row.update(accept.check(tr, accept.thresholds(thresholds, split, row["terrain_class"], row["mode"],
                                                      float(D["speed_mps"][i]))))
        pr = emg.stride_profiles(tr["fv"][:, 1] > accept.DOWN_BW, tr["emg_act"])
        row["emg_strides"] = len(pr)
        if row["mode"] != "run":                         # Camargo's EMG reference is all walking
            strides += [(x, row["terrain_class"]) for x in pr]
        out.append(row)
    return out, strides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--prior", default="")
    ap.add_argument("--splits", nargs="+", default=[data.HELD_OUT, "test_terrain"])
    ap.add_argument("--per-class", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run = Path(a.run)
    rec = json.loads((run / "record.json").read_text())
    prior = a.prior or rec["args"].get("prior", "")
    scorer = Prior.load(prior, a.device) if prior else None
    acc_path = data.OUT / "accept.json"
    thresholds = json.loads(acc_path.read_text()) if acc_path.exists() else None
    if thresholds and (not prior or Path(thresholds.get("prior") or "").resolve() != Path(prior).resolve()):
        print(f"style check off: {acc_path} was calibrated with prior {thresholds.get('prior')}, not {prior or 'none'}")
        thresholds["style"] = {}
    report = {}
    prof, prof_split, prof_cls = [], [], []
    for split in a.splits:
        rows, strides = run_split(run, rec, split, a.per_class, scorer, a.device, thresholds)
        prof += [x for x, _ in strides]
        prof_split += [split] * len(strides)
        prof_cls += [c for _, c in strides]
        report[split] = rows
        for r in rows:                                   # G1 reads the curves' mean share in band
            ib = [v for v in (r.get("in_band") or {}).values() if v is not None]
            r["in_band_mean"] = float(np.mean(ib)) if ib else float("nan")
        keys = ["completed", "survived", "accepted", "in_band_mean"] + [
            k for k in ("joint_error_deg", "r_contact", "foothold_error", "com_error", "r_style", "style_sds_loss")
            if k in rows[0]]
        print(f"\n{split}\n{'class':12s} {'n':>3s} " + " ".join(f"{k:>15s}" for k in keys))
        for c in sorted({r["terrain_class"] for r in rows}):
            rs = [r for r in rows if r["terrain_class"] == c]
            print(f"{c:12s} {len(rs):3d} " + " ".join(f"{np.nanmean([r[k] for r in rs]):15.3f}" for k in keys))
    report.update(prior=prior, accept_json=str(acc_path) if thresholds else None)
    np.savez(run / "emg_strides.npz", profiles=np.array(prof, np.float32).reshape(-1, len(CHANNELS), 101),
             split=np.array(prof_split), terrain_class=np.array(prof_cls), channels=np.array(CHANNELS))
    (run / "eval.json").write_text(json.dumps(report, indent=1))
    print("->", run / "eval.json")


if __name__ == "__main__":
    main()
