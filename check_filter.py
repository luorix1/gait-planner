"""The acceptance filter's exam, as check_gate.py is the style gate's: per kind of motion, does each check keep real
held-out motion and reject bad copies of it? Real clips become check traces (gp.accept.real_trace: contact spheres by
forward kinematics, the clip's contact labels as the feet's load, its inverse-dynamics kinetics where it has them) and
go through gp.accept.check, the same function the simulation's episodes go through.

Real held-out clips and the limits they meet:
- the active dataset's held-out splits (``data.HELD_OUT``, test_terrain): the contact check against
  ``data/<dataset>/accept.json`` (its training clips); AB01's treadmill walking also against Camargo's kinetic limits
  there (a subject Camargo never had);
- Camargo's held-out subjects and terrain (v1 test_subject, test_terrain): kinetics and contact against limits rebuilt
  by gp.accept's recipe from Camargo's training subjects only (``N_CAMARGO`` training clips for contact);
- AB01's running, one trial held out at a time: the running limits (speed trend included) rebuilt from the other
  trials and read at the held-out clip's speed. The held-out splits hold no running with kinetics, and the contact
  limits saw these clips, so only GRF and joints are scored.
Bad copies of the same clips: ``foot_skate`` (check_prior.skate: the whole body slid at 0.5 m/s) against the contact
check; ``kinetics_x1.3`` (forces and moments x1.3) against the GRF and joint checks; ``knee_lock`` (each knee held at
its extension limit while its foot is down) against the contact check's joint-limit part.

Per group (split, terrain class / mode, source): the share of real clips passing each check (``contact`` split into its
``slide`` and ``limits`` parts) and of bad copies rejected. Target (``TARGET``): at least 90 % each, in every group, for
each rate over ``MIN_JUDGE`` clips or more (kinetic rates: clips with a valid stance; smaller samples are shown, not
judged). ``real_by_subject`` says whether a group's failures are one atypical subject or everyone a little.

Run: GP_DATASET=v3 CUDA_VISIBLE_DEVICES= .venv/bin/python check_filter.py   (after python -m gp.accept; CPU, ~3 min)
Out: data/<dataset>/filter_exam.json and a table
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np

from check_prior import skate
from gp import accept, data

TARGET = 0.90
MIN_JUDGE = int(round(1 / (1 - TARGET)))      # a group needs this many clips to show a 10 % failure rate
N_CAMARGO = 1500                              # Camargo training clips for its contact limits
KIN_SCALE = 1.3
COPIES = {"foot_skate": ("contact",), "kinetics_x1.3": ("grf", "joints", "kinetics"), "knee_lock": ("limits",)}
CHECKS = ("grf", "joints", "contact", "slide", "limits")


def knee_lock(q: np.ndarray, down: np.ndarray) -> np.ndarray:
    """``q`` with each knee at its extension limit while its foot is down."""
    b = accept._body()
    names, lim = list(b["model"].coord_names), list(b["lim"])
    q = q.copy()
    for k, s in enumerate(data.SIDES):
        j = names.index(f"knee_angle_{s}")
        q[down[:, k], j] = b["lo"][lim.index(j)]
    return q


def verdict(o: dict[str, Any], thr: dict[str, Any]) -> dict[str, bool | None]:
    """The checks this exam scores, from one gp.accept.check result: GRF, joints, contact and contact's two parts."""
    has = thr.get("slide_m") is not None
    return dict(grf=o["grf"], joints=o["joints"], contact=o["contact"],
                slide=bool(o["worst_slide_m"] <= thr["slide_m"]) if has else None,
                limits=bool(o["at_limit_frac"] <= thr["at_limit_frac"]) if has else None,
                kinetics=None if o["grf"] is None else bool(o["grf"] and o["joints"]))


def score(r: dict[str, Any], thr: dict[str, Any], copies: bool = True) -> dict[str, Any]:
    """One real clip (gp.accept.real_clips) and its bad copies against ``thr`` -> per kind, :func:`verdict`."""
    t = accept.real_trace(r["q"], r["down"], r["fv"], r["m"])
    o = accept.check(t, thr)
    out: dict[str, Any] = {"real": verdict(o, thr), "subject": r["subject"]}
    if o["kinetic_stances"]:
        out["kinetics_x1.3"] = verdict(accept.check(dict(t, fv=t["fv"] * KIN_SCALE, moment=t["moment"] * KIN_SCALE), thr), thr)
    if copies:
        names = list(accept._body()["model"].coord_names)
        sk = skate(r["q"], names, np.arange(len(r["q"])) / data.HZ)
        out["foot_skate"] = verdict(accept.check(accept.real_trace(sk, r["down"]), thr), thr)
        out["knee_lock"] = verdict(accept.check(accept.real_trace(knee_lock(r["q"], r["down"]), r["down"]), thr), thr)
    return out


def group(r: dict[str, Any], split: str | None = None) -> str:
    return f"{split or r['split']}: {r['cls']}/{r['mode']} ({r['source']})"


def main() -> None:
    acc_path = data.OUT / "accept.json"
    acc = json.loads(acc_path.read_text())
    assert "contact" in acc, f"{acc_path} predates the contact limits: run python -m gp.accept"
    res: dict[str, list] = defaultdict(list)
    # 1. the active dataset's held-out clips, against its accept.json
    kf = "kinetics_ab01.npz" if (data.OUT / "kinetics_ab01.npz").exists() else None
    for r in accept.real_clips(data.OUT, kf, (data.HELD_OUT, "test_terrain"), source=None):
        res[group(r)].append(score(r, accept.thresholds(acc, r["split"], r["cls"], r["mode"], r["speed"])))
    print(f"scored {data.DATASET} held-out", flush=True)
    # 2. Camargo's held-out subjects and terrain, against limits from its training subjects
    cam = accept.real_clips(splits=("train", "test_subject", "test_terrain"))
    train = [r for r in cam if r["split"] == "train"]
    rng = np.random.default_rng(0)
    acc_c = dict(kinetics={c: accept._limits([r for r in train if r["cls"] == c]) for c in sorted({r["cls"] for r in train})},
                 contact=accept.contact_limits([train[i] for i in rng.choice(len(train), min(N_CAMARGO, len(train)), replace=False)]),
                 style={})
    for r in cam:
        if r["split"] != "train":
            res[group(r)].append(score(r, accept.thresholds(acc_c, r["split"], r["cls"], r["mode"], r["speed"])))
    print("scored Camargo held-out", flush=True)
    # 3. AB01 running, one trial held out at a time
    if kf:
        run = accept.real_clips(data.OUT, kf, source="ab01", mode="run")
        top = float(np.load(data.OUT / "clips.npz")["speed_mps"][np.load(data.OUT / "clips.npz")["mode"] == "run"].max())
        for g in sorted({data.motion_group(r["name"]) for r in run}):
            rest = [r for r in run if data.motion_group(r["name"]) != g]
            lim = accept._limits(rest, accept.RUN_STANCE_S)
            acc_r = dict(kinetics={}, kinetics_run=dict(lim, by_speed=accept._by_speed(rest, lim, top)), style={})
            for r in run:
                if data.motion_group(r["name"]) == g:
                    thr = accept.thresholds(acc_r, "train", r["cls"], "run", r["speed"])
                    res["train, one trial held out: running (ab01)"].append(score(r, thr, copies=False))
        print("scored AB01 running", flush=True)
    # per group: real pass rates, bad-copy rejection rates
    out: dict[str, Any] = dict(dataset=data.DATASET, accept_json=str(acc_path), target=TARGET, min_judge=MIN_JUDGE,
                               kin_scale=KIN_SCALE, groups={})
    rate = lambda xs, want: float(np.mean([x is want for x in xs])) if xs else None  # noqa: E731
    for g in sorted(res):
        rows = res[g]
        row: dict[str, Any] = dict(clips=len(rows), real={c: rate([x["real"][c] for x in rows if x["real"][c] is not None], True)
                                                          for c in CHECKS})
        row["n_kinetic"] = sum(x["real"]["grf"] is not None for x in rows)
        subj = sorted({x["subject"] for x in rows})
        if len(subj) > 1:                                # who fails: one atypical subject or everyone a little
            row["real_by_subject"] = {s: {c: rate([x["real"][c] for x in rows if x["subject"] == s and x["real"][c] is not None],
                                                  True) for c in CHECKS} for s in subj}
        row["rejected"] = {k: {c: rate([x[k][c] for x in rows if k in x and x[k][c] is not None], False) for c in cs}
                           for k, cs in COPIES.items()}
        # judged: rates over MIN_JUDGE clips or more (kinetic rates count the clips with a valid stance)
        n = {c: row["n_kinetic"] if c in ("grf", "joints", "kinetics") else len(rows) for c in (*CHECKS, "kinetics")}
        vals = {f"real {c}": v for c, v in row["real"].items() if v is not None and n[c] >= MIN_JUDGE}
        vals |= {f"{k} {c}": v for k, d in row["rejected"].items() for c, v in d.items() if v is not None and n[c] >= MIN_JUDGE}
        row["below_target"] = sorted(k for k, v in vals.items() if v < TARGET)
        row["pass"] = not row["below_target"] if vals else None
        out["groups"][g] = row
    out["exam_pass"] = all(r["pass"] is not False for r in out["groups"].values())
    f = lambda v: "     -" if v is None else f"{v:6.2f}"  # noqa: E731
    print(f"\n{'group':56s} {'n':>4s} {'kin':>4s} | real pass: {'grf':>6s} {'joints':>6s} {'contct':>6s} {'slide':>6s} "
          f"{'limits':>6s} | rejected: skate  x1.3 grf joints either  knee lock")
    for g, r in out["groups"].items():
        print(f"{g:56s} {r['clips']:4d} {r['n_kinetic']:4d} | {'':10s} " + " ".join(f(r["real"][c]) for c in CHECKS)
              + f" | {'':9s} {f(r['rejected']['foot_skate']['contact'])} {f(r['rejected']['kinetics_x1.3']['grf'])} "
              f"{f(r['rejected']['kinetics_x1.3']['joints'])} {f(r['rejected']['kinetics_x1.3']['kinetics'])} "
              f"{f(r['rejected']['knee_lock']['limits'])}"
              + {True: "", False: "  FAIL", None: "  (small)"}[r["pass"]])
    dest = data.OUT / "filter_exam.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nexam {'PASS' if out['exam_pass'] else 'FAIL'} -> {dest}")


if __name__ == "__main__":
    main()
