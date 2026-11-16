"""The training data, built once from the seed library's Rajagopal v3 export. Datasets (``DATASETS``):

- **v2** (the default, ``data/v2``, user 2026-11-15): AB01 as below plus PARC's capture (running, platforms, gaps, a
  few stairs; one performer, already passed through PARC's physics tracker), so the loop starts with every skill it
  is meant to combine. PARC terrain is a box grid, built into one shared heightfield (``hf_*`` in ``clips.npz``,
  ``terrain_kind`` 1). PARC clips that use the hands and those named in ``EXCLUDE`` are left out. PARC is split by motion (a motion's tracked, untracked, straightened and terrain-jittered copies
  together, by its last ``<word>_<NNN>`` tag): ``crc32(tag) % 10 == 0`` goes to ``test_trial``.
- **ab01** (``data/ab01``): our AB01 pilot alone, one subject: treadmill standing and locomotion at -15 to
  15 deg (the belt travel integrated, so the lab frame climbs or descends) and the 12 overground walks with turns.
  Split by trial: ``test_terrain`` = every +-15 deg trial (locomotion and standing); ``test_trial`` = trial 4 of each
  overground speed and the 1.2 m/s treadmill trials at 0, +5 and -10 deg; ``train`` = the rest.
- **v1** (``data/v1``): AB01 and Camargo 2021, split by held-out Camargo subjects and terrain. Camargo is set aside
  since 2026-11-14 (its IK welds the lumbar, so the trunk never moves, and some stair clips' heels never reach the
  ground); kept for reproduction and for the Camargo kinetics and EMG built on it.
- **hard** (``data/hard``, 2026-11-16): only the clips named (``build("hard", clips=[...])``) from AB01, PARC and
  Bones SEED, for single-clip feasibility runs (``imitate_clip.py``); all ``train``. SEED is confidential: its take
  names live in the git-ignored ``runs/hard_clips_chain.sh``, and ``data/`` stays on this machine.

Both:
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

Kinetic, GRF and EMG references stay Camargo's (``REF`` = ``data/v1``; user 2026-11-15: the most nominal human data).

Run (CPU, ~1 min): ``.venv/bin/python -m gp.data [v2|ab01|v1]``
Out: ``data/<dataset>/clips.npz``, ``plans.npz``, ``split.json``
"""

from __future__ import annotations

import json
import os
import re
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .terrain import atlas, height, hf_height, pad, profile, tile_from_boxes

ROOT = Path(__file__).resolve().parents[1]
#: The active dataset (data/ab01, v1, v2 and v3 are kept for reproduction); GP_DATASET picks another per process.
DATASET = os.environ.get("GP_DATASET", "v4")
OUT = ROOT / "data" / DATASET
#: Camargo's inverse dynamics, plates and EMG (built on v1): the human reference ranges for every dataset.
REF = ROOT / "data" / "v1"
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
#: Clips left out by name (substring): ``_mdm_`` likely PARC generator output; ``mid_climbing`` hand-assisted (PARC
#: labels a hand down in 3-5 % of frames, under the seed library's 5 % cut); ``parc_dm_platform_024_dm`` swing toe
#: 7.5 cm into a platform face for 0.33 s (its untracked twin is clean). User 2026-11-15.
EXCLUDE = ("_mdm_", "mid_climbing", "parc_dm_platform_024_dm")
#: PARC's physics-tracker replays: its dm_motion_recorder appends ``_dm`` to the motion PARC's torque-driven humanoid
#: produced while tracking a clip, so these are simulation, not capture. In v2 they were 16.0 of PARC's 17.9 min (all
#: running, jumps, stairs and walking, half the platforms). Datasets with ``no_replays`` leave them out (user
#: 2026-11-16); PARC's untracked platform clips stay.
SIM_REPLAY = re.compile(r"(^|_)dm(_|$)")
KEY_BODIES = ("calcn_l", "calcn_r", "toes_l", "toes_r", "torso")
SIDES = ("l", "r")
#: Drawn once: np.random.default_rng(2026).choice(sorted Camargo subjects, 3, replace=False).
TEST_SUBJECTS = ("camargo_AB06", "camargo_AB09", "camargo_AB24")


#: AB01, by trial number (mc_NNN; MC_TRIAL_INDEX.md): every +-15 deg trial is held-out terrain; trial 4 of each overground
#: speed and the 1.2 m/s treadmill trials at 0, +5 and -10 deg are held-out trials.
AB01_TEST_TERRAIN = {0, 6, 7, 8, 9, 35, 36, 37}
AB01_TEST_TRIAL = {12, 21, 28, 50, 54, 58}


def split_ab01(r: dict[str, Any]) -> str:
    mc = int(r["name"].split("_mc")[1][:3])
    return "test_terrain" if mc in AB01_TEST_TERRAIN else "test_trial" if mc in AB01_TEST_TRIAL else "train"


def split_v2(r: dict[str, Any]) -> str:
    if r["source"] == "ab01":
        return split_ab01(r)
    # PARC: one motion's copies together (``_dm`` tracked and untracked versions, ``_straight``, ``_wN`` terrain
    # jitter) by its last ``<word>_<NNN>`` tag; distinct motions sharing a tag just stay together
    tag = re.findall(r"([a-z]+_\d{3})", r["name"])[-1]
    return "test_trial" if zlib.crc32(tag.encode()) % 10 == 0 else "train"


def clip_weights(D: dict[str, Any], sources: str = "recordings", lengths: Any = None) -> np.ndarray:
    """A sampling weight per clip, summing to 1: every recording (:func:`motion_group`) gets the same total, split among
    its clips by length (``lengths``, default the clips' frames); with ``sources="sqrt"`` each source (dataset) first
    gets a share by the square root of its total length, split equally among its recordings (Bones SEED's 13,000
    takes would otherwise drown AB01 and PARC). The style prior's batches and the tasks' episode starts both use it."""
    ln = np.asarray(D["frames"] if lengths is None else lengths, float)
    _, inv = np.unique([motion_group(str(n)) for n in D["name"]], return_inverse=True)
    tot = np.bincount(inv, ln)[inv]
    w = np.divide(ln, tot, out=np.zeros_like(ln), where=tot > 0)
    if sources == "sqrt":
        src = np.asarray(D["source"]).astype(str)
        for name in np.unique(src):
            m = src == name
            if w[m].sum() > 0:
                w[m] *= np.sqrt(ln[m].sum()) / w[m].sum()
    return w / w.sum()


def split_v4(r: dict[str, Any]) -> str:
    """AB01 and PARC as v2; Bones SEED by actor (a tenth of the actors held out, as unseen people)."""
    if r["source"] == "seed_bones":
        return "test_trial" if zlib.crc32(r["subject"].encode()) % 10 == 0 else "train"
    return split_v2(r)


def motion_group(name: str) -> str:
    """The recording a clip comes from: an AB01 trial (``ab01_mc019``), a Bones SEED take (the name without its window,
    ``seed_<take>__A<actor>``), or a PARC motion with all its copies (its last ``<word>_<NNN>`` tag, as
    :func:`split_v2`). Folds of the data keep a group together."""
    if name.startswith("ab01_"):
        return name.split("_w")[0]
    if name.startswith("seed_"):
        return name.rsplit("_w", 1)[0]
    tags = re.findall(r"([a-z]+_\d{3})", name)
    return tags[-1] if tags else name


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


DATASETS: dict[str, dict[str, Any]] = {
    "v4": dict(sources=("ab01", "parc_seed", "seed_bones"), split=split_v4, splits=("train", "test_trial", "test_terrain"),
               no_replays=True, held_out="as v3; Bones SEED: actors with crc32(subject) % 10 == 0 (unseen people)"),
    "v3": dict(sources=("ab01", "parc_seed"), split=split_v2, splits=("train", "test_trial", "test_terrain"), no_replays=True,
               held_out="as v2, without PARC's physics-tracker replays (SIM_REPLAY)"),
    "v2": dict(sources=("ab01", "parc_seed"), split=split_v2, splits=("train", "test_trial", "test_terrain"),
               held_out="AB01 as ab01; PARC test_trial: motions with crc32(<word>_<NNN> tag) % 10 == 0"),
    "ab01": dict(sources=("ab01",), split=split_ab01, splits=("train", "test_trial", "test_terrain"),
                 held_out="test_terrain: AB01 +-15 deg trials; test_trial: overground trial 4 per speed, treadmill "
                          "1.2 m/s at 0/+5/-10 deg"),
    "v1": dict(sources=("ab01", "camargo2021"), split=split_of, splits=("train", "test_subject", "test_terrain"),
               held_out="test_subject: Camargo AB06/AB09/AB24; test_terrain: ramps |grade| >= 16 deg, Camargo stair_4"),
    "hard": dict(sources=("ab01", "parc_seed", "seed_bones"), clips_required=True, split=lambda r: "train",
                 splits=("train",), held_out="none: single-clip feasibility runs"),
}
#: The active dataset's held-out-motion split (the one the prior's gate and the evaluation use).
HELD_OUT = DATASETS[DATASET]["splits"][1]


def load_model(device: Any = "cpu") -> Any:
    """The body as SPRINT's recipe loads it (friction transition 0.01 m/s; its 17 joint limits are its own),
    with the specific tension its muscle forces were built on."""
    from sprint.recipes.study import load_model as _load
    return _load(MODEL, device, specific_tension=SPECIFIC_TENSION)[0]


def coord_mirror_signs(model: Any, poses: int = 16) -> np.ndarray:
    """Per coordinate, the sign ``s`` with mirror(q)[k] = s[k] * q[partner(k)], found by forward kinematics: the
    mirrored pose must put every body origin at the reflection (z -> -z) of its partner's. SPRINT's ``coord_signs``
    reads each axis in its own joint frame, which gets Rajagopal's hips (identical axes, mirrored joint frames) and its
    coupled knee (one coordinate on three axes) wrong: hip adduction, hip rotation and knee come out -1, and that
    "mirror" puts bodies up to 0.78 m off. Starts from coord_signs and flips a pair wherever that lowers the error;
    coordinates that move no body origin (MTP) keep coord_signs' value. Raises if no choice mirrors the model."""
    from sprint.recipes.imitation import coord_signs, mirrored
    from sprint.recipes.motion import forward_kinematics_np, model_layout
    L = model_layout(model)
    names, bodies = list(model.coord_names), list(model.body_names)
    signs = coord_signs(names, L.axes, L.is_rot).astype(np.float64)
    part = [names.index(mirrored(n)) for n in names]
    q = np.random.default_rng(0).uniform(-0.3, 0.3, (poses, len(names)))
    ref = forward_kinematics_np(model, q.astype(np.float32))[0][:, [bodies.index(mirrored(b)) for b in bodies]] * [1, 1, -1]
    # summed, not max: with several pairs wrong, fixing one leaves the max to the others and the search would stall
    err = lambda s: float(np.abs(forward_kinematics_np(model, (q[:, part] * s).astype(np.float32))[0] - ref).sum())  # noqa: E731
    best = err(signs)
    for _ in range(2):
        for k in range(len(names)):
            if part[k] < k:
                continue
            trial = signs.copy()
            trial[[k, part[k]]] *= -1
            e = err(trial)
            if e < best - 1e-9:
                signs, best = trial, e
    worst = float(np.abs(forward_kinematics_np(model, (q[:, part] * signs).astype(np.float32))[0] - ref).max())
    if worst > 1e-4:
        raise ValueError(f"{model.name}: no coordinate signs mirror the model (body error {worst:.3f} m)")
    return signs.astype(np.float32)


def record_layout(nt: int) -> dict[str, slice]:
    """Columns of a style record: root (pelvis origin) position and quaternion (x, y, z, w), the tracked
    coordinates, the key bodies' origins, and the terrain height under the root."""
    k = 7 + nt
    return dict(root_pos=slice(0, 3), root_quat=slice(3, 7), joints=slice(7, k), key_pos=slice(k, k + 3 * len(KEY_BODIES)),
                ground=slice(k + 3 * len(KEY_BODIES), k + 3 * len(KEY_BODIES) + 1))


def style_records(xp: np.ndarray, xq: np.ndarray, qpos: np.ndarray, tracked: np.ndarray, key: list[int],
                  ground: np.ndarray) -> np.ndarray:
    """Style records ``[T, D]`` from forward kinematics (body origins ``xp``, quaternions ``xq``) and the terrain
    height under the root ``ground`` ``[T]``."""
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


def ground_under(D: dict[str, Any], c: int, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Terrain height at ``(x, z)`` for clip ``c`` of a loaded ``D``: its profile, or its tile of the shared heightfield."""
    if D["terrain_kind"][c] == 1:
        hf, off = D["hf"], D["hf_offset"][c]
        return hf_height(hf["heights"], hf["block"], hf["cell"], hf["origin"], x + off[0], z + off[1])
    n = D["profile_n"][c]
    X1, Y1 = pad([np.c_[D["profile_x"][c][:n], D["profile_y"][c][:n]]])
    return height(X1, Y1, np.zeros(1, int), np.asarray(x, np.float64).reshape(1, -1))[0].reshape(np.shape(x))


def heightfield_gap(hf: dict[str, Any], offset: Any, ctr: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Clearance ``[T, S]`` of spheres (centres ``[T, S, 3]`` in a clip's frame, radii ``[S]``) above that clip's tile of
    the shared heightfield (``hf``: gp.terrain.atlas, ``offset`` the tile's), measured straight down: negative =
    inside. The heightfield counterpart of :func:`sphere_gap`, for the reset lift of heightfield clips."""
    from .terrain import hf_height
    h = hf_height(np.asarray(hf["heights"]), np.asarray(hf["block"]), hf["cell"], hf["origin"],
                  ctr[..., 0] + offset[0], ctr[..., 2] + offset[1])
    return ctr[..., 1] - r - h


def excluded(name: str, dataset: str = DATASET) -> str:
    """Why ``dataset`` leaves a clip out by its name ('' if it does not): an EXCLUDE entry, or a PARC tracker replay."""
    hit = next((x for x in EXCLUDE if x in name), "")
    if not hit and DATASETS[dataset].get("no_replays") and name.startswith("parc_") and SIM_REPLAY.search(name):
        hit = "PARC tracker replay (_dm)"
    return hit


def resample(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SPRINT's load_motion on a clip's coordinates ``[T, n]`` at SRC_HZ: low-pass, central differences, low-pass
    again, then onto the control grid. Returns qpos and qvel ``[F, n]`` and the grid times ``[F]``."""
    from sprint.recipes.motion import _lowpass
    src_dt = 1.0 / SRC_HZ
    ts = np.arange(len(q)) * src_dt
    qf = _lowpass(q, src_dt, CUTOFF_HZ, 4)
    qd = _lowpass(np.gradient(qf, ts, axis=0, edge_order=2), src_dt, CUTOFF_HZ, 4)
    grid = np.arange(int(np.floor(ts[-1] * HZ + 1e-9)) + 1) / HZ
    qpos = np.stack([np.interp(grid, ts, qf[:, k]) for k in range(q.shape[1])], 1)
    qvel = np.stack([np.interp(grid, ts, qd[:, k]) for k in range(q.shape[1])], 1)
    return qpos, qvel, grid


def build(dataset: str = DATASET, limit: int | None = None, out: Path | None = None,
          clips: Any = None) -> None:
    """``limit``: the first clips only (smoke tests), written to ``out``. ``clips``: only these names (``hard`` needs
    them; the other datasets default to all)."""
    from sprint.recipes.motion import _rotate, forward_kinematics_np, model_layout

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

    ds = DATASETS[dataset]
    out = out or ROOT / "data" / dataset
    keep = clips
    if ds.get("clips_required") and not keep:
        raise ValueError(f"{dataset}: name its clips, build({dataset!r}, clips=[...])")
    rows = [r for f in sorted(SEED.glob("manifest_*.json")) for r in json.loads(f.read_text())
            if r["exported"] and r["source"] in ds["sources"] and not r.get("uses_hands")
            and not excluded(r["name"], dataset) and (keep is None or r["name"] in keep)][:limit]
    meta: dict[str, list[Any]] = {k: [] for k in ("name", "subject", "source", "terrain_class", "mode", "split",
                                                  "grade_deg", "stair_rise_m", "speed_mps", "frames")}
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ("qpos", "qvel", "contact", "lift", "plan_contact", "foothold",
                                                           "t_touchdown", "t_liftoff", "com", "record")}
    profiles, tiles, glitched = [], [], []
    rot = [k for k, c in enumerate(names) if not c.startswith("pelvis_t")]
    for n, r in enumerate(rows):
        d = np.load(SEED / r["file"])
        cn = list(d["coord_names"])
        q = d["q"][:, [cn.index(c) for c in names]].astype(np.float64)
        T = json.loads(str(d["terrain"]))
        P, tile = profile(T), None
        if P is None:                                    # PARC box grid -> a tile of the shared heightfield
            if T.get("type") != "heightfield":
                raise ValueError(f"{r['name']}: terrain {T.get('type')}")
            tile, P = tile_from_boxes(d["hf"], d["hf_min"], float(d["dx"])), np.array([[-1.0, 0.0], [1.0, 0.0]])
        x0, z0 = q[0, tx], q[0, tz]
        q[:, tx] -= x0
        q[:, tz] -= z0
        P = P - [x0, 0.0]
        if tile is not None:
            tile["origin"] = (tile["origin"][0] - x0, tile["origin"][1] - z0)
        qpos, qvel, grid = resample(q)
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
        if tile is None:
            X1, Y1 = pad([P])
            ground = height(X1, Y1, np.zeros(1, int), xp[None, :, 0, 0])[0]
            gap = sphere_gap(P, ctr[..., :2], rad)
        else:
            ground = hf_height(tile["heights"], tile["block"], tile["cell"], tile["origin"], xp[:, 0, 0], xp[:, 0, 2])
            gap = heightfield_gap(tile, (0.0, 0.0), ctr, rad)
        lift = np.maximum(0.0, -gap.min(1)) + 0.001
        rec = style_records(xp, xq, qpos, L.tracked, key, ground)
        for k, v in (("qpos", qpos), ("qvel", qvel), ("contact", contact), ("lift", lift), ("plan_contact", pc),
                     ("foothold", np.stack(hold, 1)), ("t_touchdown", np.stack(td, 1)), ("t_liftoff", np.stack(lo, 1)),
                     ("com", com), ("record", rec)):
            arrays[k].append(v.astype(bool) if v.dtype == bool else v.astype(np.float32))
        profiles.append(P)
        tiles.append(tile)
        horiz = qvel[:, [tx, tz]]
        for k, v in (("name", r["name"]), ("subject", r["subject"]), ("source", r["source"]),
                     ("terrain_class", r["terrain_class"]), ("mode", r["mode"]), ("split", ds["split"](r)),
                     ("grade_deg", float(r.get("grade_deg") or 0.0)), ("stair_rise_m", float(r.get("stair_rise_m") or 0.0)),
                     ("speed_mps", float(np.median(np.linalg.norm(horiz, axis=1)))), ("frames", len(grid))):
            meta[k].append(v)
        if n % 500 == 0:
            print(f"{n}/{len(rows)} clips, {time.time() - t0:.0f} s", flush=True)
    X, Y = pad(profiles)
    frames = np.array(meta["frames"])
    kind = np.array([t is not None for t in tiles], int)
    hf_offset, hf = np.zeros((len(tiles), 2)), {}
    if kind.any():
        on = [t for t in tiles if t is not None]
        cell = on[0]["cell"]
        if not all(np.isclose(t["cell"], cell) for t in on):
            raise ValueError("heightfield clips with different cell sizes cannot share one map")
        A = atlas(on, cell)
        hf_offset[kind == 1] = A["offsets"]
        hf = dict(hf_heights=A["heights"], hf_block=A["block"], hf_origin=np.array(A["origin"]), hf_cell=cell)
    out.mkdir(parents=True, exist_ok=True)
    cat = {k: np.concatenate(v) for k, v in arrays.items()}
    np.savez(out / "clips.npz", coord_names=np.array(names), start=np.r_[0, np.cumsum(frames)[:-1]],
             profile_x=X, profile_y=Y, profile_n=np.array([len(P) for P in profiles]), qpos=cat["qpos"],
             qvel=cat["qvel"], contact=cat["contact"], lift=cat["lift"], terrain_kind=kind, hf_offset=hf_offset,
             hz=HZ, **hf, **{k: np.array(v) for k, v in meta.items()})
    np.savez(out / "plans.npz", contact=cat["plan_contact"], foothold=cat["foothold"], t_touchdown=cat["t_touchdown"],
             t_liftoff=cat["t_liftoff"], com=cat["com"], record=cat["record"], key_bodies=np.array(KEY_BODIES),
             tracked=np.array([names[k] for k in L.tracked]))
    split = np.array(meta["split"])
    summary = {s: dict(clips=int((split == s).sum()), hours=round(float(frames[split == s].sum()) / HZ / 3600, 2))
               for s in ds["splits"]}
    summary.update(dataset=dataset, sources=list(ds["sources"]), held_out=ds["held_out"], dropped_joint_speed=glitched,
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
    D.update(terrain_kind=c["terrain_kind"][keep] if "terrain_kind" in c else np.zeros(len(keep), int),
             hf_offset=c["hf_offset"][keep] if "hf_offset" in c else np.zeros((len(keep), 2)))
    if "hf_heights" in c:
        D["hf"] = dict(heights=c["hf_heights"], block=c["hf_block"], cell=float(c["hf_cell"]),
                       origin=tuple(float(v) for v in c["hf_origin"]))
    return D


PER_CLIP = ("name", "subject", "source", "terrain_class", "mode", "split", "grade_deg", "stair_rise_m", "speed_mps",
            "frames", "profile_x", "profile_y", "profile_n")
PER_FRAME = ("qpos", "qvel", "contact", "lift", "foothold", "t_touchdown", "t_liftoff", "com", "record", "plan_contact")


def take(D: dict[str, Any], clips: Any) -> dict[str, Any]:
    """Some clips of a loaded ``D`` (indices into its clips), re-offset."""
    clips = np.asarray(clips)
    rows = np.concatenate([np.arange(D["start"][i], D["start"][i] + D["frames"][i]) for i in clips])
    out = dict(D, **{k: D[k][clips] for k in PER_CLIP}, **{k: D[k][rows] for k in PER_FRAME},
               **{k: np.asarray(D[k])[clips] for k in ("terrain_kind", "hf_offset") if k in D})
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
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else DATASET)
