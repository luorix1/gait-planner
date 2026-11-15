"""AB01 inverse dynamics on Rajagopal v3: the running kinetic ranges the acceptance filter lacks (Camargo is all
walking, so running skipped the ground-force and joint-load checks).

For each AB01 treadmill clip of the active dataset (gp.data; locomotion on the instrumented split-belt treadmill),
the seed library's Rajagopal v3 kinematics plus the two belt plates (``data/treadmill/processed/mot``, 1 kHz, on the
mocap clock like the IK; read only) carried through the transforms the kinematics went through. The clips are the
``_og_`` trials (the belt's travel added to the pelvis), so the belt's travel, per IK frame the ``_og_`` IK's pelvis
translation minus the ``_tm_`` IK's, is added to the centres of pressure; then lab -> canonical -> the export's turn
and leg-length scale (camargo_id.plate_transform); forces scaled by model mass / subject mass (83.6 kg, user
2026-10-30). Then camargo_id.net_moments: each loaded plate to the nearer foot, OpenSim ID on the force-free model.

The plates run about 14 ms behind the motion capture (found 2026-11-15: total vertical force against the whole-body
centre of mass's vertical acceleration peaks there at r 0.99 in every trial, against 0.96 at zero lag; at 2.4 m/s the
lag alone puts the centre of pressure 3 cm ahead of the foot). Each trial's lag is estimated that way (:func:`plate_lag`)
and taken off the plates' clock.

Validity per leg, per frame: a foot that is down carries exactly one loaded plate, and that plate carries no other
foot (a crossover puts both feet on one belt); a foot in the air is valid. Walking clips are solved too, as a check
of the pipeline against Camargo's level walking; only running feeds the filter (Camargo stays the walking reference).

Run (OpenSim 4.6): ~/miniconda3/envs/bolt/bin/python ab01_id.py [--jobs 4]
Out: data/<dataset>/kinetics_ab01.npz, per frame of the dataset (NaN outside AB01 treadmill clips and on invalid
legs): moment_nm_per_kg, grf_bw (l, r); kinetics_ab01.json: per mode and class, clips, valid share, and the 95th
percentile of per-clip peaks beside Camargo's level walking.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import camargo_id as ci
from gp import data

SESSION = Path.home() / "Jinwoo/ab01_ilseung"          # read only
IK = SESSION / "opensim/H2190/ik"
PLATES = SESSION / "data/treadmill/processed/mot"
SUBJECT_KG = 83.6
G = 9.80665


def read_mot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.strip() == "endheader")
    return lines[k + 1].split("\t"), np.array([[float(x) for x in ln.split()] for ln in lines[k + 2:] if ln.strip()])


def plate_lag(trial: str, ft: np.ndarray, fy: np.ndarray, span: float = 0.05) -> tuple[float, float]:
    """How late the plates run behind the motion capture (s) and the correlation there: the total vertical force
    ``fy`` (times ``ft``) against the whole-body centre of mass's vertical acceleration (H2190 BodyKinematics of the
    ``_tm_`` IK), both low-passed at 12 Hz zero-phase; by Newton they move together."""
    from scipy.signal import butter, filtfilt
    c, k = read_mot(SESSION / "opensim/H2190/body_kinematics" / f"{trial}_tm_marker_BodyKinematics_pos_global.sto")
    t, y = k[:, 0], k[:, c.index("center_of_mass_Y")]
    lp = lambda x, fs: filtfilt(*butter(2, 12.0 / (fs / 2)), x)  # noqa: E731
    a = np.gradient(np.gradient(lp(y, 1 / np.median(np.diff(t))), t), t)
    f = lp(fy, 1 / np.median(np.diff(ft)))
    inner = slice(int(span * 100) + 5, -int(span * 100) - 5)
    lags = np.arange(-span, span + 1e-9, 0.001)
    r = np.array([np.corrcoef(np.interp(t[inner] + L, ft, f), a[inner])[0, 1] for L in lags])
    return float(lags[r.argmax()]), float(r.max())


def clips() -> list[str]:
    """The active dataset's AB01 treadmill locomotion clips."""
    c = np.load(data.OUT / "clips.npz")
    out = []
    for n, s, mode in zip(c["name"], c["source"], c["mode"], strict=True):
        if s == "ab01" and mode in ("walk", "run"):
            m = json.loads(str(np.load(ci.SEED / "canonical" / "ab01" / f"{n}.npz")["meta"]))
            if m["kind"] == "treadmill":
                out.append(str(n))
    return out


def solve_clip(name: str, tmp: Path) -> dict | None:
    exp = np.load(ci.SEED / "osim_export" / "ab01" / f"{name}.npz")
    T = json.loads(str(exp["terrain"]))
    m = json.loads(str(np.load(ci.SEED / "canonical" / "ab01" / f"{name}.npz")["meta"]))
    trial = m["original"].removesuffix("_og_marker.mot")
    q, names = exp["q"].astype(float), list(exp["coord_names"])
    co, og = read_mot(IK / f"{trial}_og_marker.mot")
    ct, tm = read_mot(IK / f"{trial}_tm_marker.mot")
    if co != ct or len(og) != len(tm):
        return None
    t = og[0, 0] + (int(round(m["window_s"][0] * ci.FPS)) + np.arange(len(q))) / ci.FPS
    pc, fp = read_mot(PLATES / f"{trial}_grf_processed.mot")
    ft, fp = fp[:, 0], fp[:, 1:]
    if len(pc) != 19 or t[-1] > ft[-1]:
        return None
    lag, r_lag = plate_lag(trial, ft, fp[:, 1] + fp[:, 10])
    ft = ft - lag                                                  # onto the motion capture's clock
    tr = [co.index(f"pelvis_t{a}") for a in "xyz"]
    belt = og[:, tr] - tm[:, tr]                                   # the belt's travel, lab frame, per IK frame
    shift = np.stack([np.interp(ft, og[:, 0], belt[:, k]) for k in range(3)], 1)
    model, state, m_model = ci.force_free_model(str(ci.MODEL))   # fresh per clip: the ID tool keeps hold of it
    F, P, _ = ci.carry_plates(ft, fp, t, ci.plate_transform(T.get("aligned_yaw_rad", 0.0)), T["scale"],
                              m_model / SUBJECT_KG, cop_shift=shift)
    r = ci.net_moments(model, state, m_model, q, names, t, F, P, tmp, name)
    ct_ = exp["contact"].astype(bool)
    down = np.stack([ct_[:, 2] | ct_[:, 3], ct_[:, 0] | ct_[:, 1]], 1)            # r, l
    n_on = np.stack([(r["side"] == i).sum(1) for i in range(2)], 1)              # loaded plates given to each foot
    valid = ~down | (n_on == 1)
    valid[down.all(1) & (n_on.max(1) > 1)] = False               # both feet down, one holding both plates: one belt
    d = np.linalg.norm(P[..., [0, 2]][:, :, None] - r["foot"][:, None, :, [0, 2]], axis=-1)    # [T, P, 2]
    near = np.stack([np.where(r["side"] == i, d[..., i], np.inf).min(1) for i in range(2)], 1)
    dist = np.where(np.isfinite(near), near, np.nan)              # centre of pressure to its foot: a transform check
    return dict(moment=r["moment"], grf_bw=r["grf"] / (m_model * G), valid=valid, coord_names=names,
                cop_foot_m=dist, lag_s=lag, lag_r=r_lag, mode=m["mode"], cls=m["terrain_class"], speed=m["speed_mps"], grade=m["grade_deg"])


def _worker(names: list[str]) -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for n in names:
            try:
                r = solve_clip(n, Path(td))
            except Exception as e:  # noqa: BLE001  a clip whose trial lacks a file
                print(f"  {n}: {e!r}"[:200], flush=True)
                r = None
            if r is not None:
                out[n] = r
    return out


def main(jobs: int) -> None:
    from multiprocessing import Pool
    names = clips()
    chunks = [names[i::jobs] for i in range(jobs)]
    res: dict = {}
    with Pool(jobs) as pool:
        for r in pool.imap_unordered(_worker, chunks):
            res.update(r)
    print(f"{len(res)} of {len(names)} AB01 treadmill clips solved", flush=True)
    c = np.load(data.OUT / "clips.npz")
    F, names_d = int(c["frames"].sum()), list(c["coord_names"])
    moment = np.full((F, len(names_d)), np.nan, np.float32)
    grf = np.full((F, 2, 3), np.nan, np.float32)
    summary: dict = {}
    peaks: dict = {}
    jn = [f"{j}_{s}" for j in ci.SAGITTAL for s in ("l", "r")]
    for i, n in enumerate(c["name"]):
        r = res.get(str(n))
        if r is None:
            continue
        s, k = int(c["start"][i]), int(c["frames"][i])
        src, dst = np.arange(len(r["valid"])) / ci.FPS, np.arange(k) / data.HZ
        near = np.clip(np.rint(dst * ci.FPS).astype(int), 0, len(src) - 1)
        ok_r, ok_l = r["valid"][near, 0], r["valid"][near, 1]
        mask = {"r": ok_r, "l": ok_l, "": ok_r & ok_l}
        cols = [r["coord_names"].index(x) for x in names_d]
        v = np.stack([np.interp(dst, src, r["moment"][:, cols][:, j]) for j in range(len(names_d))], 1)
        for j, x in enumerate(names_d):
            v[~mask[ci.side_of(x)], j] = np.nan
        moment[s:s + k] = v
        g = np.stack([np.stack([np.interp(dst, src, r["grf_bw"][:, f, a]) for a in range(3)], 1) for f in range(2)], 1)
        g[~ok_r, 0], g[~ok_l, 1] = np.nan, np.nan
        grf[s:s + k] = g[:, ::-1]                       # stored (l, r) like the plans
        key = f"{r['mode']}/{r['cls']}"
        sm = summary.setdefault(key, dict(clips=0, frames=0, valid_r=0, valid_l=0, cop_to_foot_med_m=[], plate_lag_ms=[]))
        sm["clips"] += 1
        sm["frames"] += k
        sm["valid_r"] += int(ok_r.sum())
        sm["valid_l"] += int(ok_l.sum())
        sm["cop_to_foot_med_m"].append(float(np.nanmedian(r["cop_foot_m"])))
        sm["plate_lag_ms"].append(1000 * r["lag_s"])
        pk = peaks.setdefault(key, {"grf": []} | {j: [] for j in jn})
        pk["grf"].append(float(np.nanmax(g[..., 1])) if np.isfinite(g[..., 1]).any() else np.nan)
        for j in jn:
            col = v[:, names_d.index(j)]
            pk[j].append(float(np.nanmax(np.abs(col))) if np.isfinite(col).any() else np.nan)
    acc = data.OUT / "accept.json"                                # Camargo's limits (gp.accept)
    ref = json.loads(acc.read_text()).get("kinetics", {}).get("level") if acc.exists() else None
    for key, sm in summary.items():
        sm["valid_share"] = round((sm.pop("valid_r") + sm.pop("valid_l")) / (2 * sm["frames"]), 3)
        sm["cop_to_foot_med_m"] = round(float(np.median(sm["cop_to_foot_med_m"])), 3)
        sm["plate_lag_ms"] = [round(float(np.min(sm["plate_lag_ms"])), 1), round(float(np.max(sm["plate_lag_ms"])), 1)]
        pk = peaks[key]
        sm["p95_of_clip_peaks"] = dict(grf_bw=round(float(np.nanpercentile(pk["grf"], 95)), 2),
                                       **{j: round(float(np.nanpercentile(pk[f"{j}_l"] + pk[f"{j}_r"], 95)), 2)
                                          for j in ci.SAGITTAL})
    out = dict(subject_kg=SUBJECT_KG, clips_solved=len(res), by_mode_class=summary,
               camargo_level_walking=dict(grf_bw=ref["grf_peak_bw"], **ref["joint_peak_nm_per_kg"]) if ref else None)
    np.savez(data.OUT / "kinetics_ab01.npz", moment_nm_per_kg=moment, grf_bw=grf, coord_names=np.array(names_d))
    (data.OUT / "kinetics_ab01.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4)
