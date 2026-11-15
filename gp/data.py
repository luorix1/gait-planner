"""The v1 training data, built once from the seed library's Rajagopal v3 export.

- **Clips**: AB01 and Camargo 2021 (PARC obstacle clips are dropped in v1: their terrain is heightfield-only).
- **Split** (``split`` per clip): ``test_subject`` = every clip of three Camargo subjects; ``test_terrain`` = the
  steepest ramps (|grade| >= 16 deg, Camargo's 18 deg) and the tallest stairs (Camargo stair_4, 178 mm) of every
  other subject, for the held-out-terrain tests; ``train`` = the rest.
- **Dropped**: clips with a joint speed over 20 rad/s anywhere (``MAX_JOINT_SPEED``; source IK glitches).
- **Motion**: coordinates low-passed at 6 Hz and differentiated as SPRINT's ``load_motion`` does, resampled from
  30 Hz to the 40 Hz control step SPRINT's imitation recipe is tuned at. Each clip is moved so it starts at
  x = z = 0, its terrain profile with it.
- **Plans** (``plans.npz``): per frame and foot (l, r) whether it is down (heel or toe flag), the foothold of its
  current or next contact (mean foot point over that contact), the time to its next touchdown and to its lift-off
  (capped at 1 s), and the whole-body centre of mass.
- **Reset lift** (``clips.npz`` ``lift``): per frame, how far to raise the root so no foot contact sphere starts
  inside the terrain (push-off frames dip a few cm; see the seed library README), plus 1 mm.
- **Style records** (``plans.npz`` ``record``): the per-frame quantities the style prior's windows are built from
  (:func:`gp.smp.window_features`), from the same forward kinematics the simulation uses.

Run (CPU, ~10 min): ``.venv/bin/python -m gp.data``
Out: ``data/v1/clips.npz``, ``data/v1/plans.npz``, ``data/v1/split.json``
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .terrain import height, pad, profile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v1"
SEED = Path.home() / "Jinwoo/seed_library/data/osim_export"
#: Byte-identical to the model the seed library's IK used (STC/assets/models/gait/Rajagopal_lower_collision_v3.osim).
MODEL = Path.home() / "Jinwoo/neural_test/models/Rajagopal_lower_collision_v3.osim"
HZ = 40.0
SRC_HZ = 30.0
CUTOFF_HZ = 6.0
PLAN_CAP_S = 1.0
#: A clip whose low-passed joint speed passes this anywhere is dropped (source IK glitches: 7 Camargo clips at
#: 25-132 rad/s; real gait peaks near 10, the 99.9th percentile of every joint here is under 8).
MAX_JOINT_SPEED = 20.0
KEY_BODIES = ("calcn_l", "calcn_r", "toes_l", "toes_r", "torso")
SIDES = ("l", "r")
#: Drawn once: np.random.default_rng(2026).choice(sorted Camargo subjects, 3, replace=False).
TEST_SUBJECTS = ("camargo_AB06", "camargo_AB09", "camargo_AB24")


def split_of(r: dict[str, Any]) -> str:
    if r["subject"] in TEST_SUBJECTS:
        return "test_subject"
    if r["terrain_class"].startswith("ramp") and abs(r.get("grade_deg") or 0.0) >= 16.0:
        return "test_terrain"
    if r["source"] == "camargo2021" and "_stair_4_" in r["name"]:
        return "test_terrain"
    return "train"


#: Pa. Rajagopal's F0 = PCSA x 60 N/cm^2, so the metabolic model's muscle masses (F0 / tension x density x l0)
#: need 0.6 MPa; SPRINT's default, OpenSim's 0.25 MPa, makes every muscle 2.4x heavy and heat 2.4x high. Needs
#: the gait-planner-engine SPRINT (setup_env.sh). Metabolics only: forces and the compile hash are unchanged.
SPECIFIC_TENSION = 0.6e6


def load_model(device: Any = "cpu") -> Any:
    """The body as SPRINT's recipe loads it (friction transition 0.01 m/s; its 17 joint limits are its own),
    with the specific tension its muscle forces were built on."""
    from sprint.recipes.study import load_model as _load
    return _load(MODEL, device, specific_tension=SPECIFIC_TENSION)[0]


def record_layout(nt: int) -> dict[str, slice]:
    """Columns of a style record: root (pelvis origin) position and quaternion (x, y, z, w), the tracked
    coordinates, the key bodies' origins, and the terrain height under the root."""
    k = 7 + nt
    return dict(root_pos=slice(0, 3), root_quat=slice(3, 7), joints=slice(7, k), key_pos=slice(k, k + 3 * len(KEY_BODIES)),
                ground=slice(k + 3 * len(KEY_BODIES), k + 3 * len(KEY_BODIES) + 1))


def style_records(xp: np.ndarray, xq: np.ndarray, qpos: np.ndarray, tracked: np.ndarray, key: list[int],
                  P: np.ndarray) -> np.ndarray:
    """Style records ``[T, D]`` from forward kinematics (body origins ``xp``, quaternions ``xq``) on profile ``P``."""
    X1, Y1 = pad([P])
    ground = height(X1, Y1, np.zeros(1, int), xp[None, :, 0, 0])[0]
    return np.concatenate([xp[:, 0], xq[:, 0], qpos[:, tracked], xp[:, key].reshape(len(xp), -1), ground[:, None]], 1)


def _intervals(c: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True as (first, last) frame pairs."""
    d = np.diff(np.r_[0, c.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1, strict=True))


def plan_targets(c: np.ndarray, foot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One foot: contact flags ``[T]`` and foot point ``[T, 3]`` -> foothold of the current or next contact
    ``[T, 3]``, time to touchdown and to lift-off ``[T]`` (s, capped)."""
    T = len(c)
    hold, td, lo = foot.copy(), np.full(T, PLAN_CAP_S), np.zeros(T)
    iv = _intervals(c)
    prev_end = -1
    for a, b in iv:
        fh = foot[a:b + 1].mean(0)
        hold[prev_end + 1:b + 1] = fh                    # the swing before it and the contact itself
        td[prev_end + 1:a] = np.minimum((a - np.arange(prev_end + 1, a)) / HZ, PLAN_CAP_S)
        td[a:b + 1] = 0.0
        lo[a:b + 1] = np.minimum((b + 1 - np.arange(a, b + 1)) / HZ, PLAN_CAP_S) if b < T - 1 else PLAN_CAP_S
        prev_end = b
    if iv and prev_end < T - 1:                          # swing after the last contact: keep its foothold
        hold[prev_end + 1:] = hold[prev_end]
    return hold, td, lo


def sphere_gap(P: np.ndarray, ctr: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Signed distance ``[T, S]`` from spheres (centres ``[T, S, 2]`` in (x, y), radii ``[S]``) to the profile ``P``
    (end segments as rays), minus the radius: negative = inside the terrain, as SPRINT's profile contact sees it."""
    P = P.copy()
    for i, j in ((0, 1), (-1, -2)):
        v = P[i] - P[j]
        P[i] = P[j] + v / np.linalg.norm(v) * 1e4
    a, ab = P[:-1], P[1:] - P[:-1]
    t = np.clip(((ctr[..., None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    dist = np.linalg.norm(ctr[..., None, :] - (a + t[..., None] * ab), axis=-1).min(-1)
    X1, Y1 = pad([P])
    inside = ctr[..., 1] < height(X1, Y1, np.zeros(1, int), ctr[..., 0].reshape(1, -1))[0].reshape(ctr.shape[:2])
    return np.where(inside, -dist, dist) - r


def build(limit: int | None = None, out: Path = OUT) -> None:
    """``limit``: the first clips only (smoke tests), written to ``out``."""
    from sprint.recipes.motion import _lowpass, _rotate, forward_kinematics_np, model_layout

    t0 = time.time()
    model = load_model()
    L = model_layout(model)
    names, h = list(model.coord_names), model.host
    nb = int(model.nbody)
    tx, tz = names.index("pelvis_tx"), names.index("pelvis_tz")
    key = [list(model.body_names).index(b) for b in KEY_BODIES]
    mass = np.asarray(h["body_mass"][:nb], np.float64)
    bcom = np.asarray(h["body_com"], np.float64).reshape(-1, 3)[:nb]
    sph = L.contact_spheres
    sph_body = h["geom_body"][sph].astype(int)
    sph_side = np.array([model.body_names[b][-1] for b in sph_body])
    rad = np.asarray(h["geom_size"][sph][:, 0], np.float64)

    rows = [r for f in sorted(SEED.glob("manifest_*.json")) for r in json.loads(f.read_text())
            if r["exported"] and r["source"] != "parc_seed" and not r.get("uses_hands")][:limit]
    meta: dict[str, list[Any]] = {k: [] for k in ("name", "subject", "source", "terrain_class", "mode", "split",
                                                  "grade_deg", "stair_rise_m", "speed_mps", "frames")}
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ("qpos", "qvel", "contact", "lift", "plan_contact", "foothold",
                                                           "t_touchdown", "t_liftoff", "com", "record")}
    profiles, glitched = [], []
    rot = [k for k, c in enumerate(names) if not c.startswith("pelvis_t")]
    for n, r in enumerate(rows):
        d = np.load(SEED / r["file"])
        cn = list(d["coord_names"])
        q = d["q"][:, [cn.index(c) for c in names]].astype(np.float64)
        P = profile(json.loads(str(d["terrain"])))
        if P is None:
            raise ValueError(f"{r['name']}: no profile terrain")
        x0, z0 = q[0, tx], q[0, tz]
        q[:, tx] -= x0
        q[:, tz] -= z0
        P = P - [x0, 0.0]
        # SPRINT's load_motion: low-pass, central differences, low-pass again; then onto the control grid
        src_dt = 1.0 / SRC_HZ
        ts = np.arange(len(q)) * src_dt
        qf = _lowpass(q, src_dt, CUTOFF_HZ, 4)
        qd = _lowpass(np.gradient(qf, ts, axis=0, edge_order=2), src_dt, CUTOFF_HZ, 4)
        grid = np.arange(int(np.floor(ts[-1] * HZ + 1e-9)) + 1) / HZ
        qpos = np.stack([np.interp(grid, ts, qf[:, k]) for k in range(len(names))], 1)
        qvel = np.stack([np.interp(grid, ts, qd[:, k]) for k in range(len(names))], 1)
        if np.abs(qvel[:, rot]).max() > MAX_JOINT_SPEED:
            glitched.append(r["name"])
            continue
        contact = d["contact"].astype(bool)[np.clip(np.rint(grid * SRC_HZ).astype(int), 0, len(q) - 1)]
        # forward kinematics: centre of mass, foot points, style records
        xp, xq = forward_kinematics_np(model, qpos.astype(np.float32))
        xp, xq = xp.astype(np.float64), xq.astype(np.float64)
        com = ((xp + _rotate(xq, np.broadcast_to(bcom, xp.shape))) * mass[None, :, None]).sum(1) / mass.sum()
        ctr = xp[:, sph_body] + _rotate(xq[:, sph_body], np.broadcast_to(h["geom_pos"][sph], xp[:, sph_body].shape))
        bottom = ctr - np.stack([np.zeros_like(rad), rad, np.zeros_like(rad)], -1)
        pc = np.stack([contact[:, 0] | contact[:, 1], contact[:, 2] | contact[:, 3]], 1)
        hold, td, lo = zip(*(plan_targets(pc[:, i], bottom[:, sph_side == s].mean(1)) for i, s in enumerate(SIDES)),
                           strict=True)
        lift = np.maximum(0.0, -sphere_gap(P, ctr[..., :2], rad).min(1)) + 0.001
        rec = style_records(xp, xq, qpos, L.tracked, key, P)
        for k, v in (("qpos", qpos), ("qvel", qvel), ("contact", contact), ("lift", lift), ("plan_contact", pc),
                     ("foothold", np.stack(hold, 1)), ("t_touchdown", np.stack(td, 1)), ("t_liftoff", np.stack(lo, 1)),
                     ("com", com), ("record", rec)):
            arrays[k].append(v.astype(bool) if v.dtype == bool else v.astype(np.float32))
        profiles.append(P)
        horiz = qvel[:, [tx, tz]]
        for k, v in (("name", r["name"]), ("subject", r["subject"]), ("source", r["source"]),
                     ("terrain_class", r["terrain_class"]), ("mode", r["mode"]), ("split", split_of(r)),
                     ("grade_deg", float(r.get("grade_deg") or 0.0)), ("stair_rise_m", float(r.get("stair_rise_m") or 0.0)),
                     ("speed_mps", float(np.median(np.linalg.norm(horiz, axis=1)))), ("frames", len(grid))):
            meta[k].append(v)
        if n % 500 == 0:
            print(f"{n}/{len(rows)} clips, {time.time() - t0:.0f} s", flush=True)
    X, Y = pad(profiles)
    frames = np.array(meta["frames"])
    out.mkdir(parents=True, exist_ok=True)
    cat = {k: np.concatenate(v) for k, v in arrays.items()}
    np.savez(out / "clips.npz", coord_names=np.array(names), start=np.r_[0, np.cumsum(frames)[:-1]],
             profile_x=X, profile_y=Y, profile_n=np.array([len(P) for P in profiles]), qpos=cat["qpos"],
             qvel=cat["qvel"], contact=cat["contact"], lift=cat["lift"],
             hz=HZ, **{k: np.array(v) for k, v in meta.items()})
    np.savez(out / "plans.npz", contact=cat["plan_contact"], foothold=cat["foothold"], t_touchdown=cat["t_touchdown"],
             t_liftoff=cat["t_liftoff"], com=cat["com"], record=cat["record"], key_bodies=np.array(KEY_BODIES),
             tracked=np.array([names[k] for k in L.tracked]))
    split = np.array(meta["split"])
    summary = {s: dict(clips=int((split == s).sum()), hours=round(float(frames[split == s].sum()) / HZ / 3600, 2))
               for s in ("train", "test_subject", "test_terrain")}
    summary.update(dropped_joint_speed=glitched, test_subjects=list(TEST_SUBJECTS), held_out_terrain="ramps |grade| >= 16 deg, Camargo stair_4",
                   hz=HZ, frames=int(frames.sum()), built=time.strftime("%Y-%m-%d %H:%M"))
    (out / "split.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), f"\n{time.time() - t0:.0f} s")


def load(split: str | tuple[str, ...] = "train", src: Path = OUT) -> dict[str, Any]:
    """The clips of ``split`` (one or several), every per-frame array cut to them and re-offset."""
    c, p = np.load(src / "clips.npz"), np.load(src / "plans.npz")
    keep = np.flatnonzero(np.isin(c["split"], [split] if isinstance(split, str) else list(split)))
    frames = c["frames"][keep]
    rows = np.concatenate([np.arange(s, s + n) for s, n in zip(c["start"][keep], frames, strict=True)])
    D: dict[str, Any] = {k: c[k][keep] for k in ("name", "subject", "source", "terrain_class", "mode", "split",
                                                 "grade_deg", "stair_rise_m", "speed_mps", "frames",
                                                 "profile_x", "profile_y", "profile_n")}
    D.update({k: c[k][rows] for k in ("qpos", "qvel", "contact", "lift")})
    D.update({k: p[k][rows] for k in ("foothold", "t_touchdown", "t_liftoff", "com", "record")})
    D.update(plan_contact=p["contact"][rows], start=np.r_[0, np.cumsum(frames)[:-1]], coord_names=list(c["coord_names"]),
             hz=float(c["hz"]), key_bodies=list(p["key_bodies"]), tracked=list(p["tracked"]))
    return D


PER_CLIP = ("name", "subject", "source", "terrain_class", "mode", "split", "grade_deg", "stair_rise_m", "speed_mps",
            "frames", "profile_x", "profile_y", "profile_n")
PER_FRAME = ("qpos", "qvel", "contact", "lift", "foothold", "t_touchdown", "t_liftoff", "com", "record", "plan_contact")


def take(D: dict[str, Any], clips: Any) -> dict[str, Any]:
    """Some clips of a loaded ``D`` (indices into its clips), re-offset."""
    clips = np.asarray(clips)
    rows = np.concatenate([np.arange(D["start"][i], D["start"][i] + D["frames"][i]) for i in clips])
    out = dict(D, **{k: D[k][clips] for k in PER_CLIP}, **{k: D[k][rows] for k in PER_FRAME})
    out["start"] = np.r_[0, np.cumsum(out["frames"])[:-1]]
    return out


def motions(D: dict[str, Any]) -> list[Any]:
    """The clips as SPRINT :class:`~sprint.recipes.motion.Motion` s (level floor as far as SPRINT's recipe knows:
    the tasks set each clip's own terrain)."""
    from sprint.recipes.motion import Motion
    out = []
    for i, (s, n) in enumerate(zip(D["start"], D["frames"], strict=True)):
        out.append(Motion(name=str(D["name"][i]), path=Path(str(D["name"][i])), sha256="", dt=1.0 / D["hz"], t0=0.0,
                          qpos=np.ascontiguousarray(D["qpos"][s:s + n]), qvel=np.ascontiguousarray(D["qvel"][s:s + n]),
                          source_rate_hz=SRC_HZ, noise_deg={}, speed=float(D["speed_mps"][i]), ground_offset=0.0,
                          grade_deg=0.0))
    return out


if __name__ == "__main__":
    build()
