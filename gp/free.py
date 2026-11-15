"""Episodes without a reference clip: generated courses (gp.terrain_gen) as stand-in clips.

A course becomes a clip the tasks can run like any other: it always starts from a real standing frame placed at the
course start, its plan comes from a planner, and its "reference" is only that standing pose carried along the plan's
centre of mass (heading along the course), so the recipe's root errors tell the critic where the plan expects the
body. The tasks treat these clips differently in three ways (``D["free"]``): an episode starts at frame 0 only, a fall
is losing the plan's centre of mass by ``FREE_FALL_M`` (or tilting 60 deg) instead of leaving the clip's pelvis, and
the critic reads a flag saying the episode has no clip.

Plans come from the learned planner (gp.planner.plan_course) or from :func:`stand_in_plan`: footholds on every stair tread,
evenly spaced steps (at most ``step`` m) elsewhere, never on a riser's or a pit's edge, alternating feet 2 x
``half_width`` apart, and a fixed walking rhythm (``t_step`` per step, ``dbl`` of double support). It is a stand-in,
not a model of human footstep choice.

:func:`merge` puts free clips beside a loaded dataset in one dict with one shared heightfield at the courses' cell (the
dataset's heightfield clips cut to their root path +- 1.5 m and refined to that cell).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import data as gpdata
from .terrain import atlas, hf_height

FREE_FALL_M = 0.5            # a free episode falls once its centre of mass is this far from the plan's (horizontal)
REFINE = 8                   # course cell = the dataset's heightfield cell / REFINE (0.0485 m for v2's PARC tiles)


def _path_at(p: dict[str, np.ndarray], s: np.ndarray) -> dict[str, np.ndarray]:
    """Centreline position, heading and profile height at arc lengths ``s`` (nearest sample for the height)."""
    k = np.clip(np.searchsorted(p["s"], s), 0, len(p["s"]) - 1)
    return dict(x=np.interp(s, p["s"], p["x"]), z=np.interp(s, p["s"], p["z"]),
                heading=np.interp(s, p["s"], np.unwrap(p["heading"])), h=p["h"][k])


def stand_in_plan(T: dict[str, Any], com_height: float, *, step: float = 0.6, t_step: float = 0.55, dbl: float = 0.12,
                  half_width: float = 0.08, s0: float = 1.0, end_pad: float = 1.0) -> dict[str, Any]:
    """A plan on course tile ``T`` (gp.terrain_gen.course_tile) at gp.data.HZ: per frame ``plan_contact`` [T, 2],
    ``foothold`` [T, 2, 3], ``t_touchdown``, ``t_liftoff`` [T, 2], ``com`` [T, 3], and the centre of mass's arc length
    ``s_com`` and ``heading`` [T]; ``steps`` = the footholds' arc lengths. Feet: index 0 left, 1 right."""
    p, segs = T["path"], T["segments"]
    fixed, forbid, u0 = [], [], 0.0
    for sg in segs:
        a, b = u0, u0 + sg["length"]
        if sg["kind"] == "stairs":
            fixed += [a + (i + 0.5) * sg["tread"] for i in range(sg["steps"])]
            forbid.append((a - 0.12, b + 0.12))
        elif sg["kind"] == "gap":
            forbid.append((a - 0.15, b + 0.15))
        elif sg["kind"] == "platform":
            forbid += [(a - 0.12, a + 0.12), (b - 0.12, b + 0.12)]
        u0 = b
    s_end = float(p["s"][-1]) - end_pad
    anchors = sorted({s0, s_end, *[f for f in fixed if s0 < f < s_end]})
    steps = [s0]
    for a, b in zip(anchors[:-1], anchors[1:], strict=True):
        n = int(np.ceil((b - a) / step - 1e-9))
        for k in range(1, n):
            x = a + k * (b - a) / n
            for lo, hi in forbid:                       # off a riser's or a pit's edge, to the nearer side
                if lo < x < hi and not any(abs(x - f) < 1e-9 for f in fixed):
                    x = lo if x - lo < hi - x else hi
            if x - steps[-1] > 0.2 and b - x > 0.2:
                steps.append(x)
        steps.append(b)
    st = np.array(steps)                               # st[0]: both feet at the start; st[1:]: one step each
    K = len(st) - 1
    t1 = 0.8                                            # standing still for the first 0.8 s
    td = np.r_[t1 - t_step, t1 + t_step * np.arange(K + 1)]   # td[k]: touchdown of step k (k = 0 is the stance start)
    n_frames = int(np.ceil((td[-1] + 1.0) * gpdata.HZ)) + 1
    t = np.arange(n_frames) / gpdata.HZ
    side = np.r_[-1, (np.arange(1, K + 2) % 2)]         # step k >= 1: right (1) on odd k, left (0) on even k
    st_full = np.r_[st, st[-1]]                         # step K + 1: the other foot joins the last foothold
    P = _path_at(p, st_full)
    left = np.stack([-np.sin(P["heading"]), -np.cos(P["heading"])], 1)          # the left of the heading, in (x, z)
    contact = np.zeros((n_frames, 2), bool)
    foot = np.zeros((n_frames, 2, 3))
    for sd in (0, 1):
        ks = [0] + [k for k in range(1, K + 2) if side[k] == sd]
        for j, k in enumerate(ks):
            down = td[k] if k else 0.0
            nxt = ks[j + 1] if j + 1 < len(ks) else None
            up = td[nxt - 1] + dbl if nxt is not None else np.inf    # lifts once the other foot has landed
            if k == 0 and sd == 1:
                up = td[1] - (t_step - dbl)                          # the right foot steps first
            m = (t >= down) & (t < up)
            off = (1 if sd == 0 else -1) * half_width * left[k]
            contact[m, sd] = True
            foot[m, sd] = (P["x"][k] + off[0], P["h"][k], P["z"][k] + off[1])
    hold, ttd, tlo = zip(*(gpdata.plan_targets(contact[:, sd], foot[:, sd]) for sd in (0, 1)), strict=True)
    # centre of mass: midway between the last two footholds at each touchdown, standing still at both ends
    tk = np.r_[0.0, td[0], td[1:]]
    sk = np.r_[st[0], st[0], (st_full[:-1] + st_full[1:]) / 2]
    sk[-1] = st_full[-1]
    s_com = np.interp(t, tk, sk)
    C = _path_at(p, s_com)
    w = max(1, int(round(0.3 / float(p["s"][1] - p["s"][0]))))   # over a pit the body rides the ground beside it
    hmax = np.array([p["h"][max(0, k - w):k + w + 1].max() for k in range(len(p["h"]))])
    com = np.stack([C["x"], np.interp(s_com, p["s"], hmax) + com_height, C["z"]], 1)
    return dict(plan_contact=contact, foothold=np.stack(hold, 1), t_touchdown=np.stack(ttd, 1),
                t_liftoff=np.stack(tlo, 1), com=com, s_com=s_com, heading=C["heading"], steps=st_full)


def course_clips(tiles: list[dict[str, Any]], D: dict[str, Any], stand: int, plans: list[dict[str, Any]] | None = None,
                 names: list[str] | None = None) -> dict[str, Any]:
    """Free clips (gp.data.load's layout, plus ``free`` and the ``tiles``) for course tiles, starting from clip
    ``stand`` of ``D``'s first frame (a level standing clip). ``plans``: one per tile (default :func:`stand_in_plan`)."""
    from sprint.recipes.motion import _rotate, forward_kinematics_np, model_layout
    model = gpdata.load_model()
    L = model_layout(model)
    h = model.host
    names_c = list(D["coord_names"])
    ix = {c: names_c.index(c) for c in ("pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt", "pelvis_list",
                                        "pelvis_rotation")}
    q0 = np.asarray(D["qpos"][D["start"][stand]], np.float64)
    com0 = np.asarray(D["com"][D["start"][stand]], np.float64)
    to_pelvis = np.array([q0[ix["pelvis_tx"]] - com0[0], q0[ix["pelvis_tz"]] - com0[2]])   # heading 0, standing
    key = [list(model.body_names).index(b) for b in gpdata.KEY_BODIES]
    sph = L.contact_spheres
    sph_body, rad = h["geom_body"][sph].astype(int), np.asarray(h["geom_size"][sph][:, 0], np.float64)
    out: dict[str, list] = {k: [] for k in (*gpdata.PER_CLIP, *gpdata.PER_FRAME)}
    for i, T in enumerate(tiles):
        P = plans[i] if plans else stand_in_plan(T, com_height=float(com0[1]))
        n = len(P["s_com"])
        ps = P["heading"]
        q = np.repeat(q0[None], n, 0)
        c, s_ = np.cos(ps), np.sin(ps)
        q[:, ix["pelvis_tx"]] = P["com"][:, 0] + c * to_pelvis[0] + s_ * to_pelvis[1]
        q[:, ix["pelvis_tz"]] = P["com"][:, 2] - s_ * to_pelvis[0] + c * to_pelvis[1]
        q[:, ix["pelvis_ty"]] = q0[ix["pelvis_ty"]] + P["com"][:, 1] - com0[1]
        q[:, ix["pelvis_rotation"]] = ps
        if "pelvis_quat" in P:                         # a planner's plan: its pelvis orientation (the joint is Z-X-Y)
            from scipy.spatial.transform import Rotation
            e = Rotation.from_quat(P["pelvis_quat"]).as_euler("ZXY")
            q[:, ix["pelvis_tilt"]], q[:, ix["pelvis_list"]], q[:, ix["pelvis_rotation"]] = e[:, 0], e[:, 1], np.unwrap(e[:, 2])
        qd = np.gradient(q, 1.0 / gpdata.HZ, axis=0)
        xp, xq = (a.astype(np.float64) for a in forward_kinematics_np(model, q.astype(np.float32)))
        ctr = xp[:, sph_body] + _rotate(xq[:, sph_body], np.broadcast_to(h["geom_pos"][sph], xp[:, sph_body].shape))
        lift = np.maximum(0.0, -gpdata.heightfield_gap(T, (0.0, 0.0), ctr, rad).min(1)) + 0.001
        ground = hf_height(T["heights"], T["block"], T["cell"], T["origin"], xp[:, 0, 0], xp[:, 0, 2])
        rec = gpdata.style_records(xp, xq, q, L.tracked, key, ground)
        pc = P["plan_contact"]
        for k, v in (("qpos", q), ("qvel", qd), ("contact", np.repeat(pc, 2, 1)), ("lift", lift),
                     ("plan_contact", pc), ("foothold", P["foothold"]), ("t_touchdown", P["t_touchdown"]),
                     ("t_liftoff", P["t_liftoff"]), ("com", P["com"]), ("record", rec)):
            out[k].append(v.astype(bool) if v.dtype == bool else v.astype(np.float32))
        kinds = "+".join(s["kind"] for s in T["segments"] if s["kind"] != "level") or "level"
        for k, v in (("name", names[i] if names else f"course_{i}_{kinds}"), ("subject", "none"), ("source", "course"),
                     ("terrain_class", "course"), ("mode", "walk"), ("split", "free"), ("grade_deg", 0.0),
                     ("stair_rise_m", 0.0), ("speed_mps", float(P["s_com"][-1] / (n / gpdata.HZ))), ("frames", n)):
            out[k].append(v)
    F: dict[str, Any] = {k: (np.concatenate(v) if k in gpdata.PER_FRAME else np.array(v))
                         for k, v in out.items() if k not in ("profile_x", "profile_y", "profile_n")}
    F.update(start=np.r_[0, np.cumsum(F["frames"])[:-1]], coord_names=names_c, hz=D["hz"], key_bodies=D["key_bodies"],
             tracked=D["tracked"], tiles=tiles, free=np.ones(len(tiles), bool))
    return F


def crop_refine(A: dict[str, Any], offset: Any, xs: np.ndarray, zs: np.ndarray, margin: float, f: int) -> dict[str, Any]:
    """Clip-frame tile: atlas ``A`` around map points (xs, zs) + offset +- margin, every cell split into f x f."""
    cell = A["cell"]
    x0, z0 = A["origin"]
    mx, mz = xs + offset[0], zs + offset[1]
    i0 = int(np.clip(np.floor((mx.min() - margin - x0) / cell), 0, A["heights"].shape[1] - 2))
    i1 = int(np.clip(np.ceil((mx.max() + margin - x0) / cell), i0 + 1, A["heights"].shape[1] - 1))
    j0 = int(np.clip(np.floor((mz.min() - margin - z0) / cell), 0, A["heights"].shape[0] - 2))
    j1 = int(np.clip(np.ceil((mz.max() + margin - z0) / cell), j0 + 1, A["heights"].shape[0] - 1))
    B = np.kron(A["block"][j0:j1, i0:i1], np.ones((f, f)))
    H = A["heights"][j0:j1 + 1, i0:i1 + 1]
    Hf = H[np.minimum(np.arange((j1 - j0) * f + 1) // f, j1 - j0)][:, np.minimum(np.arange((i1 - i0) * f + 1) // f, i1 - i0)]
    return dict(heights=Hf, block=B, origin=(x0 + i0 * cell - offset[0], z0 + j0 * cell - offset[1]), cell=cell / f)


def merge(D: dict[str, Any], F: dict[str, Any], margin: float = 1.5) -> dict[str, Any]:
    """``D`` (gp.data.load) and free clips ``F`` (:func:`course_clips`, its tiles at ``D``'s cell / REFINE) in one dict."""
    tiles = list(F["tiles"])
    cell = tiles[0]["cell"]
    kind = np.asarray(D.get("terrain_kind", np.zeros(len(D["name"]), int)))
    real = []
    if kind.any():
        f = int(round(D["hf"]["cell"] / cell))
        if abs(D["hf"]["cell"] / f - cell) > 1e-6:
            raise ValueError(f"courses must use the dataset's cell / {REFINE}: {D['hf']['cell'] / REFINE:.5f} m")
        tx, tz = list(D["coord_names"]).index("pelvis_tx"), list(D["coord_names"]).index("pelvis_tz")
        for c in np.flatnonzero(kind == 1):
            q = D["qpos"][D["start"][c]:D["start"][c] + D["frames"][c]]
            real.append(crop_refine(D["hf"], D["hf_offset"][c], q[:, tx], q[:, tz], margin, f))
    A = atlas(real + tiles, cell)
    off = np.zeros((len(D["name"]) + len(tiles), 2))
    off[np.flatnonzero(kind == 1)] = A["offsets"][:len(real)]
    off[len(D["name"]):] = A["offsets"][len(real):]
    K = D["profile_x"].shape[1]
    M: dict[str, Any] = {k: np.concatenate([np.asarray(D[k]), np.asarray(F[k])]) for k in (*gpdata.PER_CLIP, *gpdata.PER_FRAME)
                         if k not in ("profile_x", "profile_y", "profile_n")}
    nf = len(tiles)
    M.update(profile_x=np.concatenate([D["profile_x"], np.tile(np.arange(K) - 1.0, (nf, 1))]),
             profile_y=np.concatenate([D["profile_y"], np.zeros((nf, K))]),
             profile_n=np.concatenate([D["profile_n"], np.full(nf, K)]),
             start=np.r_[0, np.cumsum(M["frames"])[:-1]], coord_names=D["coord_names"], hz=D["hz"],
             key_bodies=D["key_bodies"], tracked=D["tracked"], terrain_kind=np.r_[kind, np.ones(nf, int)], hf_offset=off,
             hf=A, free=np.r_[np.zeros(len(D["name"]), bool), np.ones(nf, bool)])
    return M
