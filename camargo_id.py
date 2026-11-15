"""Camargo 2021 inverse dynamics on Rajagopal v3: the human kinetic ranges the acceptance filter calibrates against
(the plan doc's open item: Camargo's own ID was solved on its own models).

For each v1 Camargo clip, the seed library's Rajagopal v3 kinematics (the same motion the policies learn from) plus
Camargo's force plates carried through the same transforms the kinematics went through: lab (OpenSim, y up) ->
canonical (z up) -> the turn that put the terrain on +x -> the scale to the model's leg length; forces scaled by
model mass / subject mass, so body-weight-normalised forces and Nm/kg moments are kept. Each loaded plate goes to
the nearer foot. Forces are low-passed at 6 Hz like the coordinates (OpenSim's ID tool filters those). The plates'
moment columns are not used: they are not free moments at the centre of pressure (tens of N m about the
horizontal axes), and the vertical free moment barely moves the joint moments reported here. Every force element
is removed from the model, so the moments are net joint moments, comparable with Camargo's own.

Validity is per leg: a leg's joint moments depend only on what is distal to it, its own motion and its own foot's
force. A leg is valid where Camargo's own ID for it is (they blank each leg whose contact missed the plates; 64 % of
frames per leg against 36 % for both at once) and its foot, when down, carries a plate. Trunk and pelvis rows need
both legs. The result is resampled onto v1's 40 Hz frames.

Run, two stages (different interpreters):
  1. extract plates and Camargo's ID from the archives (mat-io):
     cd ~/Jinwoo/gait_planner && VIRTUAL_ENV= uv run --no-project --python 3.11 --with mat-io --with pandas \
         python camargo_id.py extract
  2. solve (OpenSim 4.6): ~/miniconda3/envs/bolt/bin/python camargo_id.py solve [--jobs 4]
Out: data/camargo_raw/<subject>/<mode>__<trial>.npz (stage 1); data/v1/kinetics.npz and data/v1/kinetics.json
(stage 2: per v1 frame, joint moments in N m/kg and per-foot ground force in body weights, NaN where invalid;
Camargo's own moments beside them, knee and MTP sign-flipped to Rajagopal's flexion-positive convention; a per-class
comparison).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "camargo_raw"
V1 = ROOT / "data" / "v1"
SEED = Path.home() / "Jinwoo/seed_library/data"
CAMARGO = Path.home() / "Jinwoo/datasets/camargo2021"
MODEL = Path.home() / "Jinwoo/neural_test/models/Rajagopal_lower_collision_v3.osim"
FPS = 30.0
CUTOFF_HZ = 6.0
MIN_N = 20.0
C = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])     # lab (OpenSim, y up) -> canonical (z up), as the adapter
SAGITTAL = ("hip_flexion", "knee_angle", "ankle_angle")


def side_of(coord: str) -> str:
    """'r' or 'l' for a leg coordinate (``knee_angle_r``, ``knee_angle_r_beta``), '' for pelvis and lumbar."""
    parts = coord.split("_")
    return "r" if "r" in parts[1:] else "l" if "l" in parts[1:] else ""


def v1_camargo_clips() -> list[str]:
    c = np.load(V1 / "clips.npz")
    return [str(n) for n, s in zip(c["name"], c["source"], strict=True) if s == "camargo2021"]


def canonical_meta(name: str) -> dict:
    return json.loads(str(np.load(SEED / "canonical" / "camargo" / f"{name}.npz")["meta"]))


# --- stage 1 ---------------------------------------------------------------------------------------------------
def extract() -> None:
    import matio
    info = matio.load_from_mat(str(CAMARGO / "SubjectInfo.mat"))["data"]
    mass = {str(np.ravel(s)[0]): float(w) for s, w in zip(info["Subject"], info["Weight"], strict=True)}
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "subject_mass_kg.json").write_text(json.dumps(mass, indent=1))
    trials = sorted({(canonical_meta(n)["subject"].removeprefix("camargo_"), canonical_meta(n)["trial"]) for n in v1_camargo_clips()})
    zips = [zipfile.ZipFile(z) for z in sorted(CAMARGO.glob("Subjects_Part*.zip"))]
    members = {m: z for z in zips for m in z.namelist() if m.endswith(".mat") and ("/fp/" in m or "/id/" in m)}
    done = 0
    for subj, trial in trials:
        mode, tname = trial.split("/")
        out = RAW / subj / f"{mode}__{tname}.npz"
        if out.exists():
            continue
        found = {}
        for kind in ("fp", "id"):
            hit = [m for m in members if m.startswith(f"{subj}/") and f"/{mode}/{kind}/{tname}.mat" in m]
            if len(hit) != 1:
                print(f"  {subj} {trial}: {len(hit)} {kind} files", flush=True)
                break
            with members[hit[0]].open(hit[0]) as f:
                tmp = RAW / f"_tmp_{kind}.mat"
                tmp.write_bytes(f.read())
            found[kind] = matio.load_from_mat(str(tmp))["data"]
            tmp.unlink()
        if len(found) < 2:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        fp, idt = found["fp"], found["id"]
        np.savez_compressed(out, fp_time=fp["Header"].to_numpy(float), fp=fp.drop(columns="Header").to_numpy(float),
                            fp_cols=np.array(fp.columns[1:], dtype=str), id_time=idt["Header"].to_numpy(float),
                            id=idt.drop(columns="Header").to_numpy(float), id_cols=np.array(idt.columns[1:], dtype=str))
        done += 1
        if done % 200 == 0:
            print(f"{done} trials", flush=True)
    print(f"extracted {done} new trials of {len(trials)}")


# --- stage 2 ---------------------------------------------------------------------------------------------------
def _lowpass(x: np.ndarray, dt: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    b, a = butter(2, CUTOFF_HZ / (0.5 / dt))
    return filtfilt(b, a, x, axis=0)


def clip_times(name: str) -> np.ndarray:
    """Lab time of each of the clip's 30 Hz frames (sources/camargo.py: the trial's active span +-1 s, windowed)."""
    m = canonical_meta(name)
    subj = m["subject"].removeprefix("camargo_")
    mode, trial = m["trial"].split("/")
    d = np.load(SEED / "staging" / "camargo" / subj / f"{mode}__{trial}.npz", allow_pickle=False)
    lab = d["labels"]
    if len(lab):
        act = np.flatnonzero(lab != "idle")
        t0 = d["label_time"][act[0]] - 1.0
    else:
        t0 = d["time"][0]
    start = max(t0, d["time"][0])
    a, b = (int(round(w * FPS)) for w in m["window_s"])
    return start + np.arange(a, b) / FPS


def plate_transform(yaw: float) -> np.ndarray:
    """Lab (OpenSim, y up) -> the export's OpenSim frame, before the leg-length scale: canonical, the export's turn
    about +z (seedlib.prep), back to OpenSim axes."""
    ca, sa = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
    to_osim = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])      # canonical (x, y, z) -> OpenSim (x, z, -y)
    return to_osim @ Rz @ C


def carry_plates(ft: np.ndarray, fp: np.ndarray, t: np.ndarray, M: np.ndarray, scale: float, mass_ratio: float,
                 cop_shift: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plates (``fp`` ``[N, 9 P]``: per plate force, centre of pressure, moment, lab frame, at times ``ft``) -> each
    plate's force and centre of pressure at times ``t`` in the export's frame (``M``, ``scale``; forces times
    ``mass_ratio``): ``[T, P, 3]`` each (NaN centre of pressure where the plate carries under ``MIN_N``), and each
    plate's footprint ``[P, 4]`` (x min, x max, z min, z max): the extent of its loaded centres of pressure.
    ``cop_shift`` ``[N, 3]`` (lab) is added to the centres of pressure first (a treadmill belt's travel).

    Below ``MIN_N`` a plate reads noise and its centre of pressure is meaningless (an unloaded default), so the force
    is zeroed there before the 6 Hz low-pass, and the centre of pressure is held at the nearest loaded sample: the
    force the zero-phase filter spreads a few ms past a touchdown or a lift-off acts at the heel-strike or toe-off
    point. (Without this, 2-3 frames at each heel strike read hip moments of -6 to -8 N m/kg; fixed 2026-11-14.)"""
    n_plate = fp.shape[1] // 9
    dt = float(np.median(np.diff(ft)))
    F_all, P_all, boxes = [], [], []
    for p in range(n_plate):
        Fr, P = fp[:, 9 * p: 9 * p + 3].copy(), fp[:, 9 * p + 3: 9 * p + 6].copy()
        on = Fr[:, 1] > MIN_N
        Fr[~on] = 0.0
        box = np.full(4, np.nan)
        if on.any():
            k = np.arange(len(on))
            prev = np.maximum.accumulate(np.where(on, k, -1))
            nxt = np.minimum.accumulate(np.where(on, k, len(on))[::-1])[::-1]
            prev, nxt = np.where(prev < 0, nxt, prev), np.where(nxt >= len(on), prev, nxt)
            P = P[np.where(k - prev <= nxt - k, prev, nxt)]
            if cop_shift is not None:
                P = P + cop_shift
            xz = (P[on] @ M.T * scale)[:, [0, 2]]
            box = np.array([xz[:, 0].min(), xz[:, 0].max(), xz[:, 1].min(), xz[:, 1].max()])
        boxes.append(box)
        F = _lowpass(Fr, dt)
        Fi = np.stack([np.interp(t, ft, F[:, k]) for k in range(3)], 1)
        Pi = np.stack([np.interp(t, ft, P[:, k]) for k in range(3)], 1)
        Pi[Fi[:, 1] < MIN_N] = np.nan
        F_all.append(Fi @ M.T * mass_ratio)
        P_all.append(Pi @ M.T * scale)
    return np.stack(F_all, 1), np.stack(P_all, 1), np.array(boxes)


def plates_on_clip(name: str, t: np.ndarray, scale: float, yaw: float,
                   mass_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A Camargo clip's plates at times ``t`` (:func:`carry_plates`)."""
    m = canonical_meta(name)
    raw = np.load(RAW / m["subject"].removeprefix("camargo_") / f"{m['trial'].replace('/', '__')}.npz", allow_pickle=True)
    return carry_plates(raw["fp_time"], raw["fp"], t, plate_transform(yaw), scale, mass_ratio)


def force_free_model(model_path: str) -> tuple:
    """(model, state, mass) with every force element removed: inverse dynamics then gives net joint moments."""
    import opensim as osim
    model = osim.Model(model_path)
    model.updForceSet().clearAndDestroy()      # net joint moments: no muscles, contacts or limit forces act
    state = model.initSystem()
    return model, state, sum(model.getBodySet().get(i).getMass() for i in range(model.getBodySet().getSize()))


def net_moments(model, state, m_model: float, q: np.ndarray, names: list[str], t: np.ndarray, F: np.ndarray,
                P: np.ndarray, tmp: Path, name: str) -> dict:
    """OpenSim inverse dynamics of coordinates ``q`` ``[T, n]`` (``names``) at times ``t`` under plates ``F``, ``P``
    ``[T, P, 3]`` (:func:`carry_plates`), each loaded plate given to the nearer foot (horizontal distance to the
    midpoint of its calcaneus and toes origins) and applied to that calcaneus. -> ``moment`` ``[T, n]`` (N m/kg),
    per foot (r, l) ``grf`` ``[T, 2, 3]`` (N) and ``foot`` positions, and per plate ``side`` ``[T, P]`` (0 r, 1 l,
    -1 unloaded) and ``on``."""
    import opensim as osim
    cs = model.getCoordinateSet()
    cols = [(cs.get(i), names.index(cs.get(i).getName())) for i in range(cs.getSize())]
    bodies = {s: (model.getBodySet().get(f"calcn_{s}"), model.getBodySet().get(f"toes_{s}")) for s in ("r", "l")}
    foot = np.zeros((len(q), 2, 3))
    for k in range(len(q)):
        for c, j in cols:
            c.setValue(state, float(q[k, j]), False)
        model.realizePosition(state)
        for i, s in enumerate(("r", "l")):
            a, b = (np.array([v.get(n) for n in range(3)]) for v in (bb.getPositionInGround(state) for bb in bodies[s]))
            foot[k, i] = (a + b) / 2
    # assign each loaded plate to the nearer foot (horizontal distance), combine per foot
    on = np.isfinite(P[..., 0])                                                       # [T, P]
    d = np.linalg.norm((P[:, :, None, [0, 2]] - foot[:, None, :, [0, 2]]), axis=-1)   # [T, P, 2]
    side = np.where(on, np.nanargmin(np.where(np.isfinite(d), d, np.inf), axis=2), -1)
    grf, cop = np.zeros((len(q), 2, 3)), np.zeros((len(q), 2, 3))
    for i in range(2):
        w = (side == i)
        fy = np.where(w, F[..., 1], 0.0)
        grf[:, i] = np.where(w[..., None], F, 0.0).sum(1)
        cop[:, i] = np.nansum(np.where(w[..., None], P * fy[..., None], 0.0), 1) / np.maximum(fy.sum(1), 1e-9)[:, None]
        cop[fy.sum(1) <= 0, i] = foot[fy.sum(1) <= 0, i]
    # OpenSim ID on the force-free model, external loads on the calcanei
    stem = tmp / name
    head = ["time"] + [f"{s}_ground_{k}{a}" for s in ("r", "l") for k in ("force_v", "force_p", "torque_") for a in "xyz"]
    rows = np.c_[t - t[0], np.concatenate([np.concatenate([grf[:, i], cop[:, i], np.zeros((len(t), 3))], 1) for i in range(2)], 1)]
    _write_mot(stem.with_suffix(".grf.mot"), head, rows, "grf")
    qh = ["time"] + [c.getName() for c, _ in cols]
    qr = np.c_[t - t[0], np.stack([np.degrees(q[:, j]) if c.getMotionType() == osim.Coordinate.Rotational else q[:, j]
                                   for c, j in cols], 1)]
    _write_mot(stem.with_suffix(".q.mot"), qh, qr, "coordinates", degrees=True)
    loads = osim.ExternalLoads()
    for s in ("r", "l"):
        ef = osim.ExternalForce()
        ef.setName(f"grf_{s}")
        ef.set_applied_to_body(f"calcn_{s}")
        ef.set_force_expressed_in_body("ground")
        ef.set_point_expressed_in_body("ground")
        ef.set_force_identifier(f"{s}_ground_force_v")
        ef.set_point_identifier(f"{s}_ground_force_p")
        ef.set_torque_identifier(f"{s}_ground_torque_")
        ef.set_data_source_name(str(stem.with_suffix(".grf.mot")))
        loads.cloneAndAppend(ef)
    loads.setDataFileName(str(stem.with_suffix(".grf.mot")))
    loads.printToXML(str(stem.with_suffix(".loads.xml")))
    tool = osim.InverseDynamicsTool()
    tool.setModel(model)
    tool.setCoordinatesFileName(str(stem.with_suffix(".q.mot")))
    tool.setLowpassCutoffFrequency(CUTOFF_HZ)
    tool.setExternalLoadsFileName(str(stem.with_suffix(".loads.xml")))
    tool.setStartTime(0.0)
    tool.setEndTime(float(t[-1] - t[0]))
    tool.setResultsDir(str(tmp))
    tool.setOutputGenForceFileName(f"{name}.id.sto")
    tool.run()
    sto = osim.TimeSeriesTable(str(tmp / f"{name}.id.sto"))
    lab = list(sto.getColumnLabels())
    out = np.full((len(t), len(names)), np.nan)
    tt = np.array(sto.getIndependentColumn())
    for j, n in enumerate(names):
        key = next((c for c in lab if c in (f"{n}_moment", f"{n}_force")), None)
        if key is not None:
            out[:, j] = np.interp(t - t[0], tt, sto.getDependentColumn(key).to_numpy()) / m_model
    for f in stem.parent.glob(f"{name}.*"):
        f.unlink()
    return dict(moment=out, grf=grf, foot=foot, side=side, on=on)


def solve_clip(name: str, model_path: str, mass: dict, tmp: Path) -> dict | None:
    exp = np.load(SEED / "osim_export" / "camargo" / f"{name}.npz")
    T = json.loads(str(exp["terrain"]))
    m = canonical_meta(name)
    subj = m["subject"].removeprefix("camargo_")
    t = clip_times(name)
    q, names = exp["q"].astype(float), list(exp["coord_names"])
    if len(t) != len(q):
        return None
    model, state, m_model = force_free_model(model_path)
    F, P, box = plates_on_clip(name, t, T["scale"], T.get("aligned_yaw_rad", 0.0), m_model / mass[subj])
    r = net_moments(model, state, m_model, q, names, t, F, P, tmp, name)
    out, grf, foot, on = r["moment"], r["grf"], r["foot"], r["on"]
    # validity per leg (see the module docstring)
    raw = np.load(RAW / subj / f"{m['trial'].replace('/', '__')}.npz", allow_pickle=True)    # our own files
    ic = [str(x) for x in raw["id_cols"]]
    theirs = np.stack([np.interp(t, raw["id_time"], raw["id"][:, ic.index(f"{n}_moment")]) if f"{n}_moment" in ic
                       else np.full(len(t), np.nan) for n in names], 1) / mass[subj]
    for n in names:                     # their knee and MTP count extension positive (gait2392); Rajagopal flexion
        if n.startswith(("knee_angle", "mtp_angle")):
            theirs[:, names.index(n)] *= -1
    ct = exp["contact"].astype(bool)
    down = np.stack([ct[:, 2] | ct[:, 3], ct[:, 0] | ct[:, 1]], 1)                    # r, l (contact: heel_l, toe_l, heel_r, toe_r)
    carried = grf[..., 1] > MIN_N
    valid = np.stack([np.isfinite(theirs[:, [names.index(f"{j}_{s}") for j in SAGITTAL]]).all(1)
                      & (carried[:, i] | ~down[:, i]) for i, s in enumerate(("r", "l"))], 1)   # [T, 2]: r, l
    # one plate under both feet cannot say how the weight splits: both legs invalid while both are down on it
    pad = 0.15                      # m around the footprint: the loaded centres of pressure under-cover a plate
    fx, fz = foot[:, None, :, 0], foot[:, None, :, 2]                                  # [T, 1, 2]
    inside = ((fx >= box[None, :, None, 0] - pad) & (fx <= box[None, :, None, 1] + pad)
              & (fz >= box[None, :, None, 2] - pad) & (fz <= box[None, :, None, 3] + pad))    # [T, P, 2]
    shared = (on & inside.all(2)).any(1) & down.all(1)
    valid &= ~shared[:, None]
    g = 9.80665
    return dict(moment=out, theirs=theirs, grf_bw=grf / (m_model * g), valid=valid, coord_names=names)


def _write_mot(path: Path, head: list[str], rows: np.ndarray, name: str, degrees: bool = False) -> None:
    with path.open("w") as f:
        f.write(f"{name}\nversion=1\nnRows={len(rows)}\nnColumns={len(head)}\ninDegrees={'yes' if degrees else 'no'}\nendheader\n")
        f.write("\t".join(head) + "\n")
        np.savetxt(f, rows, delimiter="\t", fmt="%.8g")


def _worker(args: tuple[list[str], str]) -> dict:
    import tempfile
    names, model_path = args
    mass = json.loads((RAW / "subject_mass_kg.json").read_text())
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for n in names:
            try:
                r = solve_clip(n, model_path, mass, Path(td))
            except Exception as e:  # a clip whose trial lacks plates or ID
                print(f"  {n}: {e!r}"[:200], flush=True)
                r = None
            if r is not None:
                out[n] = r
    return out


def solve(jobs: int) -> None:
    from multiprocessing import Pool
    names = [n for n in v1_camargo_clips()
             if (RAW / canonical_meta(n)["subject"].removeprefix("camargo_") / f"{canonical_meta(n)['trial'].replace('/', '__')}.npz").exists()]
    chunks = [(names[i::jobs * 8], str(MODEL)) for i in range(jobs * 8)]
    res: dict = {}
    with Pool(jobs) as pool:
        for k, r in enumerate(pool.imap_unordered(_worker, chunks)):
            res.update(r)
            print(f"{k + 1}/{len(chunks)} chunks, {len(res)} clips", flush=True)
    # onto v1's 40 Hz frames
    c = np.load(V1 / "clips.npz")
    F, names_v1 = int(c["frames"].sum()), list(c["coord_names"])
    moment, theirs = np.full((F, len(names_v1)), np.nan, np.float32), np.full((F, len(names_v1)), np.nan, np.float32)
    grf = np.full((F, 2, 3), np.nan, np.float32)
    summary: dict = {}
    for i, n in enumerate(c["name"]):
        r = res.get(str(n))
        if r is None:
            continue
        s, k = int(c["start"][i]), int(c["frames"][i])
        src = np.arange(len(r["valid"])) / FPS
        dst = np.arange(k) / 40.0
        near = np.clip(np.rint(dst * FPS).astype(int), 0, len(src) - 1)
        ok_r, ok_l = r["valid"][near, 0], r["valid"][near, 1]
        mask = {"r": ok_r, "l": ok_l, "": ok_r & ok_l}
        cols = [r["coord_names"].index(x) for x in names_v1]
        for out, a in ((moment, r["moment"][:, cols]), (theirs, r["theirs"][:, cols])):
            v = np.stack([np.interp(dst, src, a[:, j]) for j in range(a.shape[1])], 1)
            for j, x in enumerate(names_v1):
                v[~mask[side_of(x)], j] = np.nan
            out[s:s + k] = v
        g = np.stack([np.stack([np.interp(dst, src, r["grf_bw"][:, f, a]) for a in range(3)], 1) for f in range(2)], 1)
        g[~ok_r, 0], g[~ok_l, 1] = np.nan, np.nan
        grf[s:s + k] = g[:, ::-1]                       # stored (l, r) like the plans
        cls = str(c["terrain_class"][i])
        sm = summary.setdefault(cls, dict(clips=0, valid_frames_r=0, valid_frames_l=0, frames=0))
        sm["clips"] += 1
        sm["valid_frames_r"] += int(ok_r.sum())
        sm["valid_frames_l"] += int(ok_l.sum())
        sm["frames"] += k
    # Rajagopal v3 vs Camargo's own, on valid frames: peaks per class (N m/kg) and the frame-wise RMS difference
    for cls, sm in summary.items():
        rows = np.concatenate([np.arange(c["start"][i], c["start"][i] + c["frames"][i])
                               for i in range(len(c["name"])) if str(c["terrain_class"][i]) == cls])
        for n in ("hip_flexion_r", "knee_angle_r", "ankle_angle_r"):
            j = names_v1.index(n)
            a, b = moment[rows, j], theirs[rows, j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.any():
                sm[n] = dict(ours_p2_p98=[round(float(x), 2) for x in np.percentile(a[ok], [2, 98])],
                             theirs_p2_p98=[round(float(x), 2) for x in np.percentile(b[ok], [2, 98])],
                             rms_diff=round(float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))), 3),
                             r=round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3))
    np.savez(V1 / "kinetics.npz", moment_nm_per_kg=moment, camargo_moment_nm_per_kg=theirs, grf_bw=grf,
             coord_names=np.array(names_v1))
    (V1 / "kinetics.json").write_text(json.dumps(dict(clips_solved=len(res), by_class=summary), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["extract"]:
        extract()
    elif sys.argv[1:2] == ["solve"]:
        solve(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4)
    else:
        sys.exit(__doc__)
