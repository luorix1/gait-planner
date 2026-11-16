"""The acceptance filter (plan doc, gate G1): six checks an episode must pass to join the planner's dataset. Kinetic
thresholds are Camargo's per terrain class (all its subjects, ``data.REF``; user 2026-11-15: the most nominal human
data); the style threshold comes from the active dataset's real clips of the same split and class; the contact
thresholds from its training clips of the same mode. Camargo is all walking, so running takes AB01's treadmill running
instead (``ab01_id.py``: 15 clips at 2.0-2.7 m/s, level and +-5 deg, one subject; one ``kinetics_run`` entry, used for
running on level ground and ramps). A class neither has (PARC's obstacles: jumps, drops, platforms; generated courses)
leaves the GRF and joint checks pending.

**Like with like.** Every kinetic measure is per stance and per leg on both sides (a stance: a foot down for 3+ frames;
real clips: only stances whose every frame is valid), and an episode's value is the median over its stances, so a
10 s episode with 16 stances compares with real clips of 1-5 valid stances without the max-over-stances inflation
(about 11 % on AB01 walking). The threshold is the 95th percentile of that per-clip median over real clips.

**Running limits rise with speed.** The dataset's running is faster than AB01's (PARC: median 3.4 m/s, up to 4.5), and
AB01's peaks rise with speed (per m/s: GRF +0.22 BW, matching ~0.2 in Nilsson & Thorstensson 1989; ankle +0.45, hip
+0.87 N m/kg; knee flat, as Schache et al. 2011 found for knee stance work). So each running limit is a function of the
episode's target speed: the larger of AB01's pooled 95th percentile and the 95 % prediction bound of a straight line
through AB01's level clips' values against speed (``RUN_PI``), both of the same per-clip median. The bound widens with
distance from AB01's speeds, which is the headroom for extrapolating from one subject at 2.0-2.7 m/s. Tabulated every
0.25 m/s up to the dataset's fastest running clip; faster than that, the fastest's limits hold.

**Contact.** The slide of a loaded foot: per foot and stance (simulation: vertical force above ``DOWN_BW``; real clips:
their contact labels), the median horizontal speed of the foot's slowest contact sphere (the point it rolls over: heel,
then toes, so heel-to-toe roll and a pivot on the forefoot read as still; heights are not used, so IK dips do not
matter) times the stance's duration; an episode's value is its worst stance. On real clips this is ~1 cm in a typical
walking stance; brief label flickers and treadmill-belt drift set the tail. Time at joint limits: the share of frames
with any limited coordinate within 1 deg of (or past) its limit. Both thresholds are the 95th percentile over the
active dataset's training clips of the episode's mode and class (``walk/level``), else of its mode, else of all clips,
whichever first has ``100 / (100 - PCT_HI)`` clips or more. ``check_filter.py`` is the exam: real held-out clips pass, bad copies fail.

| Check | Rejects when | Threshold from |
|---|---|---|
| Style | a 2 s window's mean SDS loss is above the real 95th percentile (its score below the 5th) | the prior on real windows |
| Plan followed | a planned contact (3+ frames) with under half its frames in contact; a stance more than 15 cm from its planned foothold; the centre of mass out of its band over 10 % of steps | fixed |
| Ground reaction | median stance peak vertical force or loading rate above the real 95th percentile | Camargo's plates (running: AB01's treadmill) |
| Joint loads | median stance peak net moment at a hip, knee or ankle above the real 95th percentile | Camargo (running: AB01) ID on Rajagopal v3 |
| Muscle effort | a muscle above 0.95 activation over 10 % of any 1 s window | fixed |
| Contact | the worst stance's slide, or the time at joint limits, above the real 95th percentile | the dataset's training clips, per mode |

An episode that fell is rejected before any check, and it is accepted only when every check passed: a check without a
threshold is ``None`` (pending, listed in ``unchecked``) and blocks acceptance. Both sides are processed alike: 40 Hz,
forces and moments low-passed at 6 Hz (Camargo's plates and ID coordinates were; a real trace says ``filtered``).

Departures from the plan doc's drawing:
- **Joint loads are net joint moments**, not knee and hip reaction forces: SPRINT's reaction pass refuses wrapped
  muscle paths (Rajagopal v3 wraps), and real reaction forces would need muscle-force estimates. Camargo's ID gives
  net moments; the simulation's are ``qfrc_total``, what SPRINT's exporter reports as the net joint moment.
- **Metabolic cost per metre is reported, not gated**: the v1 datasets hold no human metabolic data.

For G1's curve comparison: per class, the real 5th-95th percentile band of the right leg's hip, knee and ankle
moments and vertical force over the right stance (touchdown to lift-off, 101 points; whole strides are rarely all on
plates; stances of 0.3-1.5 s, running 0.12-0.45 s); an episode's score is the fraction of its stances' points inside
the band.

Run: .venv/bin/python -m gp.accept [--prior runs/prior_v1/prior.pt]  (without a prior: no style threshold yet)
Out: data/<dataset>/accept.json: ``kinetics`` per class, ``kinetics_run``, ``contact`` per mode, ``style`` per split
     and class
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from typing import Any

import numpy as np

from . import data

SPLITS = (data.HELD_OUT, "test_terrain")
JOINTS = tuple(f"{j}_{s}" for j in ("hip_flexion", "knee_angle", "ankle_angle") for s in ("l", "r"))
CURVES = ("hip_flexion_r", "knee_angle_r", "ankle_angle_r", "grf_r")
CUTOFF_HZ = 6.0
DOWN_BW = 0.05
STYLE_WIN = int(2 * data.HZ)
SAT, SAT_FRAC, SAT_WIN = 0.95, 0.10, int(data.HZ)
FOOTHOLD_MISS_M, CONTACT_HALF, COM_OUT_FRAC = 0.15, 0.5, 0.10
LIMIT_MARGIN = math.radians(1.0)
J3 = ("hip_flexion", "knee_angle", "ankle_angle")
MEASURES = ("grf_peak_bw", "grf_loading_rate_bw_s", *J3)
STANCE_S = (0.3, 1.5)
RUN_STANCE_S = (0.12, 0.45)
RUN_CLASSES = ("level", "ramp_up", "ramp_down")     # AB01 ran on these; not PARC's obstacles
RUN_PI = 0.95                                        # one-sided prediction bound of the speed trend (running limits)
PCT_HI, PCT_LO = 95, 5
MIN_CLIPS = int(round(100 / (100 - PCT_HI)))        # fewest clips a percentile limit of its own is taken from


# --- shared signal processing ------------------------------------------------------------------------------------
def lowpass(x: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    if len(x) < 16:
        return x
    b, a = butter(2, CUTOFF_HZ / (0.5 * data.HZ))
    return filtfilt(b, a, x, axis=0)


def runs(c: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    return [(a, b) for a, b in data._intervals(c) if b - a + 1 >= min_len]


def stance_peaks(fv: np.ndarray, m: np.ndarray, down: np.ndarray) -> np.ndarray:
    """``[n, 5]``: per stance of either leg (``down`` 3+ frames) whose every frame is valid, its peak vertical force,
    peak loading rate (BW/s; the steepest rise in its first half) and that leg's peak |moment| at hip, knee and ankle."""
    out = []
    for k, s in enumerate(data.SIDES):
        leg = np.c_[fv[:, k], m[:, [JOINTS.index(f"{j}_{s}") for j in J3]]]
        for a, b in runs(down[:, k], 3):
            seg = leg[a:b + 1]
            if np.isfinite(seg).all():
                rise = seg[: max(2, len(seg) // 2), 0]
                out.append([seg[:, 0].max(), np.diff(rise).max() * data.HZ, *np.abs(seg[:, 1:]).max(0)])
    return np.array(out, float).reshape(-1, len(MEASURES))


def kinetics(fv: np.ndarray, m: np.ndarray, down: np.ndarray) -> dict[str, Any]:
    """One episode's (or real clip's) kinetic measures: the median over its stances of :func:`stance_peaks`."""
    p = stance_peaks(fv, m, down)
    return dict(zip(MEASURES, (np.median(p, 0) if len(p) else np.full(len(MEASURES), np.nan)).tolist(), strict=True),
                stances=len(p))


def slide(sphere: np.ndarray, side: np.ndarray, down: np.ndarray) -> float:
    """The worst loaded stance's slide (m): per foot and stance (``down`` 3+ frames), the median horizontal speed of the
    foot's slowest contact sphere (``sphere`` ``[T, S, 3]`` bottoms, ``side`` ``[S]`` 0 left / 1 right) times the
    stance's duration."""
    v = np.linalg.norm(np.diff(np.asarray(sphere)[..., [0, 2]], axis=0), axis=-1) * data.HZ      # [T - 1, S]: t -> t + 1
    worst = 0.0
    for k in range(2):
        slow = v[:, np.asarray(side) == k].min(1)
        for a, b in runs(down[:, k], 3):
            worst = max(worst, float(np.median(slow[a:b])) * (b - a + 1) / data.HZ)
    return worst


@functools.lru_cache(maxsize=1)
def _body() -> dict[str, Any]:
    """The model's contact spheres (body, offset, radius, side) and joint limits, for traces of real clips."""
    from sprint.recipes.motion import model_layout
    model = data.load_model()
    h, bn = model.host, list(model.body_names)
    sph = model_layout(model).contact_spheres
    body = h["geom_body"][sph].astype(int)
    n = int((h["limit_dof"] >= 0).sum())
    return dict(model=model, body=body, pos=np.asarray(h["geom_pos"][sph]), rad=np.asarray(h["geom_size"][sph][:, 0]),
                side=np.array([data.SIDES.index(bn[b][-1]) for b in body]), lim=h["limit_dof"][:n].astype(int),
                lo=np.asarray(h["limit_lo"][:n]), hi=np.asarray(h["limit_hi"][:n]))


def at_limit(q: np.ndarray) -> np.ndarray:
    """Per frame: is any limited coordinate within ``LIMIT_MARGIN`` of its limit, or past it (as evaluate.py's trace)?"""
    b = _body()
    x = np.asarray(q)[:, b["lim"]]
    return ((x < b["lo"] + LIMIT_MARGIN) | (x > b["hi"] - LIMIT_MARGIN)).any(1)


def real_trace(q: np.ndarray, down: np.ndarray, fv: np.ndarray | None = None, m: np.ndarray | None = None) -> dict[str, Any]:
    """A real clip (coordinates ``q``, contact labels ``down`` ``[T, 2]``) as :func:`check`'s trace: contact sphere
    bottoms by forward kinematics, the labels as the feet's load, time at joint limits, and its kinetics (``fv``
    ``[T, 2]`` BW and ``m`` ``[T, 6]`` N m/kg for ``JOINTS``, already low-passed, NaN where invalid; none: NaN). The
    terms only a simulation has are neutral: no style score, the plan its own contacts and feet, no saturation."""
    from sprint.recipes.motion import _rotate, forward_kinematics_np
    b = _body()
    n = len(q)
    xp, xq = forward_kinematics_np(b["model"], np.asarray(q, np.float32))
    sph = (xp[:, b["body"]] + _rotate(xq[:, b["body"]], np.broadcast_to(b["pos"], xp[:, b["body"]].shape))).astype(float)
    sph[..., 1] -= b["rad"]
    foot = np.stack([sph[:, b["side"] == k].mean(1) for k in range(2)], 1)
    return dict(sds=np.full(n, np.nan), plan_contact=down, down=down, foothold_ref=foot, foot=foot,
                com_out=np.zeros(n, bool), fv=np.full((n, 2), np.nan) if fv is None else fv,
                moment=np.full((n, len(JOINTS)), np.nan) if m is None else m, filtered=True,
                act_sat=np.zeros((n, 1), bool), at_limit=at_limit(q), sphere=sph, sphere_side=b["side"], fell=False,
                metabolic_j_per_kg_m=float("nan"))


def stances(down_r: np.ndarray, span: tuple[float, float] = STANCE_S) -> list[tuple[int, int]]:
    """Right stances, touchdown to lift-off, of plausible length ``span`` (s; one cut short by the clip's ends is
    dropped)."""
    return [(a, b) for a, b in runs(down_r, 3) if a > 0 and b < len(down_r) - 1 and span[0] <= (b - a) / data.HZ <= span[1]]


def resample(x: np.ndarray, a: int, b: int) -> np.ndarray:
    return np.interp(np.linspace(a, b, 101), np.arange(len(x)), x)


# --- calibration on real clips -----------------------------------------------------------------------------------
def real_clips(src: Any = None, kin_file: str | None = "kinetics.npz", splits: tuple[str, ...] | None = None,
               source: str | None = "camargo2021", mode: str | None = None) -> list[dict[str, Any]]:
    """Per real clip of dataset ``src`` (default ``data.REF``, Camargo) in ``splits``, of ``source`` and ``mode`` (None:
    any): its labels, coordinates ``q``, both feet's contacts and kinetics from ``kin_file`` (``fv``, ``m``: NaN where
    that leg is invalid, validity is per leg, see camargo_id.py and ab01_id.py; all NaN without a file)."""
    src = src or data.REF
    c = dict(np.load(src / "clips.npz"))
    down = np.load(src / "plans.npz")["contact"]
    K = dict(np.load(src / kin_file)) if kin_file else None
    keep = np.ones(len(c["name"]), bool)
    for key, want in (("split", splits), ("source", source and [source]), ("mode", mode and [mode])):
        if want:
            keep &= np.isin(c[key], list(want))
    jcol = [list(K["coord_names"]).index(j) for j in JOINTS] if K else []
    out = []
    for i in np.flatnonzero(keep):
        s, n = int(c["start"][i]), int(c["frames"][i])
        out.append(dict(name=str(c["name"][i]), cls=str(c["terrain_class"][i]), mode=str(c["mode"][i]),
                        split=str(c["split"][i]), source=str(c["source"][i]), subject=str(c["subject"][i]),
                        speed=float(c["speed_mps"][i]), q=c["qpos"][s:s + n], down=down[s:s + n],
                        fv=K["grf_bw"][s:s + n][..., 1].astype(float) if K else np.full((n, 2), np.nan),
                        m=K["moment_nm_per_kg"][s:s + n][:, jcol].astype(float) if K else np.full((n, len(JOINTS)), np.nan)))
    return out


def _by_speed(rs: list[dict[str, Any]], base: dict[str, Any], top: float) -> dict[str, Any]:
    """Running limits against target speed (module docstring): per measure, at speeds 2.0 to ``top`` every 0.25 m/s,
    the larger of ``base``'s limit and the ``RUN_PI`` prediction bound of a line through the level clips' values (the
    same per-clip median as ``base``)."""
    from scipy import stats
    lv = [(r["speed"], k) for r in rs if r["cls"] == "level" for k in [kinetics(r["fv"], r["m"], r["down"])] if k["stances"]]
    x = np.array([s for s, _ in lv])
    v = np.unique(np.r_[np.arange(2.0, top, 0.25), top]).round(3)
    t = stats.t.ppf(RUN_PI, len(x) - 2)
    sxx = ((x - x.mean()) ** 2).sum()

    def bound(key: str, floor: float) -> list[float]:
        y = np.array([k[key] for _, k in lv])
        slope, icpt = np.polyfit(x, y, 1)
        se = np.sqrt(((y - slope * x - icpt) ** 2).sum() / (len(x) - 2))
        pi = slope * v + icpt + t * se * np.sqrt(1 + 1 / len(x) + (v - x.mean()) ** 2 / sxx)
        return np.maximum(pi, floor).round(3).tolist()

    return dict(speed_mps=v.tolist(), grf_peak_bw=bound("grf_peak_bw", base["grf_peak_bw"]),
                grf_loading_rate_bw_s=bound("grf_loading_rate_bw_s", base["grf_loading_rate_bw_s"]),
                joint_peak_nm_per_kg={j: bound(j, base["joint_peak_nm_per_kg"][j]) for j in J3})


def _limits(rs: list[dict[str, Any]], span: tuple[float, float] = STANCE_S) -> dict[str, Any]:
    """The kinetic limits of a set of real clips: 95th percentiles of their per-clip :func:`kinetics` (clips with a
    valid stance), and the G1 bands."""
    k = [x for r in rs for x in [kinetics(r["fv"], r["m"], r["down"])] if x["stances"]]
    pct = lambda key: float(np.percentile([x[key] for x in k], PCT_HI))  # noqa: E731
    cur = np.array([x for r in rs for x in _curves(r["m"], r["fv"], r["down"][:, 1], span)])
    return dict(clips=len(k), stances=sum(x["stances"] for x in k), grf_peak_bw=pct("grf_peak_bw"),
                grf_loading_rate_bw_s=pct("grf_loading_rate_bw_s"), joint_peak_nm_per_kg={j: pct(j) for j in J3},
                band={c: [np.percentile(cur[:, i], PCT_LO, 0).round(4).tolist(),
                          np.percentile(cur[:, i], PCT_HI, 0).round(4).tolist()] for i, c in enumerate(CURVES)}
                if len(cur) >= 10 else None, stance_s=list(span))


def contact_limits(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """The 95th percentiles of real clips' worst :func:`slide` and time at joint limits, per ``mode/class``, per mode
    and over ``all``, each where it has ``MIN_CLIPS`` clips or more (``thresholds`` takes the most specific)."""
    v = []
    for r in rs:
        t = real_trace(r["q"], r["down"])
        v.append((slide(t["sphere"], t["sphere_side"], t["down"]), float(t["at_limit"].mean())))
    v = np.array(v).reshape(-1, 2)
    keys = [("all", r["mode"], f"{r['mode']}/{r['cls']}") for r in rs]
    out = {}
    for g in sorted({k for ks in keys for k in ks}, key=lambda k: (k != "all", k)):
        sel = np.array([g in ks for ks in keys])
        if sel.sum() >= MIN_CLIPS:
            out[g] = dict(clips=int(sel.sum()), slide_m=float(np.percentile(v[sel, 0], PCT_HI)),
                          at_limit_frac=float(np.percentile(v[sel, 1], PCT_HI)))
    return out


def _curves(m: np.ndarray, fv: np.ndarray, down_r: np.ndarray, span: tuple[float, float] = STANCE_S) -> list[np.ndarray]:
    """``[4, 101]`` per right stance whose every frame is finite."""
    cols = [JOINTS.index(c) for c in CURVES[:3]]
    out = []
    for a, b in stances(down_r, span):
        seg = np.c_[m[a:b + 1, cols], fv[a:b + 1, 1]]
        if np.isfinite(seg).all():
            out.append(np.stack([resample(seg[:, k], 0, b - a) for k in range(4)]))
    return out


def _style_losses(prior_path: str, split: str, per_class: int = 400) -> dict[str, np.ndarray]:
    """Per class, the mean SDS loss of real 2 s windows (``per_class`` windows, 13-frame windows scored per step)."""
    import torch

    from .smp import Prior, W
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prior = Prior.load(prior_path, dev)
    D = data.load(split)
    rec = torch.as_tensor(D["record"], device=dev)
    rng = np.random.default_rng(0)
    out: dict[str, list[float]] = {}
    for cls in sorted(set(D["terrain_class"])):
        idx = np.flatnonzero(np.array(D["terrain_class"]) == cls)
        starts = [(int(D["start"][i]) + W, int(D["start"][i] + D["frames"][i]) - STYLE_WIN) for i in idx]
        starts = [(a, b) for a, b in starts if b > a]
        for k in rng.integers(len(starts), size=per_class):
            a, b = starts[k]
            t0 = int(rng.integers(a, b))
            ends = torch.arange(t0, t0 + STYLE_WIN, device=dev)
            win = rec[ends[:, None] + torch.arange(-W, 1, device=dev)]
            torch.manual_seed(0)
            out.setdefault(cls, []).append(float(prior.sds(prior.features(win)).mean()))
    return {k: np.array(v) for k, v in out.items()}


def calibrate(prior: str | None = None) -> dict[str, Any]:
    res: dict[str, Any] = dict(prior=prior, percentiles=[PCT_LO, PCT_HI], joints=JOINTS, curves=CURVES,
                               kinetics_from=str(data.REF), statistic="median over an episode's stances of each "
                               "stance's peak (per leg); limits: 95th percentile over real clips", kinetics={}, style={})
    clips = real_clips()                                                # Camargo, every subject
    for cls in sorted({r["cls"] for r in clips}):
        res["kinetics"][cls] = _limits([r for r in clips if r["cls"] == cls])
    if (data.OUT / "kinetics_ab01.npz").exists():                       # ab01_id.py
        run = real_clips(data.OUT, "kinetics_ab01.npz", source="ab01", mode="run")
        res["kinetics_run"] = dict(_limits(run, RUN_STANCE_S), source="AB01 treadmill running, ab01_id.py",
                                   classes=list(RUN_CLASSES))
        c = np.load(data.OUT / "clips.npz")
        top = float(c["speed_mps"][c["mode"] == "run"].max())          # the dataset's fastest running clip
        res["kinetics_run"]["by_speed"] = _by_speed(run, res["kinetics_run"], top)
    res["contact"] = contact_limits(real_clips(data.OUT, None, ("train",), source=None))
    for split in SPLITS:
        style = _style_losses(prior, split) if prior else {}
        res["style"][split] = {cls: float(np.percentile(v, PCT_HI)) for cls, v in style.items()}
    (data.OUT / "accept.json").write_text(json.dumps(res, indent=1))
    return res


def thresholds(acc: dict[str, Any] | None, split: str, cls: str, mode: str,
               speed: float | None = None) -> dict[str, Any] | None:
    """One episode's thresholds from ``accept.json``: Camargo's kinetics for its class (none when Camargo lacks the
    class), or AB01's running kinetics when running on level ground or a ramp (none on obstacles), at the episode's
    target ``speed`` (m/s; the reference clip's or the plan's) when given; the contact limits of its mode and class (or
    of its mode, or of all modes); and the style threshold of its split and class."""
    if not acc:
        return None
    kin = acc["kinetics"].get(cls) if mode != "run" else acc.get("kinetics_run") if cls in RUN_CLASSES else None
    if kin and mode == "run" and speed is not None and "by_speed" in kin:
        b = kin["by_speed"]
        at = lambda y: float(np.interp(speed, b["speed_mps"], y))  # noqa: E731
        kin = dict(kin, grf_peak_bw=at(b["grf_peak_bw"]), grf_loading_rate_bw_s=at(b["grf_loading_rate_bw_s"]),
                   joint_peak_nm_per_kg={j: at(y) for j, y in b["joint_peak_nm_per_kg"].items()})
    con = acc.get("contact", {})
    con = con.get(f"{mode}/{cls}") or con.get(mode) or con.get("all", {})
    return dict(kin or {}, style_loss=acc["style"].get(split, {}).get(cls), slide_m=con.get("slide_m"),
                at_limit_frac=con.get("at_limit_frac"))


# --- the filter --------------------------------------------------------------------------------------------------
def check(tr: dict[str, Any], thr: dict[str, Any] | None) -> dict[str, Any]:
    """One episode's trace (``evaluate.py``: per step ``sds``, ``plan_contact``, ``foothold_ref``, ``com_out``,
    ``fv``, ``foot``, ``moment``, ``act_sat``, ``at_limit``, ``sphere``; and ``sphere_side``, ``fell``,
    ``metabolic_j_per_kg_m``; a real clip's (:func:`real_trace`) also ``down``, its contact labels, and ``filtered``)
    against its thresholds -> pass per check (None: no threshold yet), accepted, the numbers behind them, and the G1
    in-band fraction per curve."""
    down = np.asarray(tr["down"]) if "down" in tr else tr["fv"] > DOWN_BW
    fv, m = (tr["fv"], tr["moment"]) if tr.get("filtered") else (lowpass(tr["fv"]), lowpass(tr["moment"]))
    out: dict[str, Any] = dict(fell=bool(tr["fell"]))
    thr = thr or {}
    # style
    sds = tr["sds"][np.isfinite(tr["sds"])]
    if len(sds):
        k = min(STYLE_WIN, len(sds))
        worst = float(np.convolve(sds, np.ones(k) / k, "valid").max())
    else:
        worst = float("nan")
    lim = thr.get("style_loss")
    out["style_worst_loss"] = worst
    out["style"] = None if lim is None or not np.isfinite(worst) else bool(worst <= lim)
    # plan followed
    missed_contact = sum(down[a:b + 1, s].mean() < CONTACT_HALF for s in range(2) for a, b in runs(tr["plan_contact"][:, s], 3))
    off = [float(np.linalg.norm((tr["foot"][a:b + 1, s].mean(0) - tr["foothold_ref"][(a + b) // 2, s])[[0, 2]]))
           for s in range(2) for a, b in runs(down[:, s], 3) if tr["plan_contact"][(a + b) // 2, s]]
    com_out = float(tr["com_out"].mean())
    out.update(missed_contacts=int(missed_contact), worst_foothold_m=max(off, default=0.0), com_out_frac=com_out,
               plan=bool(missed_contact == 0 and max(off, default=0.0) <= FOOTHOLD_MISS_M and com_out <= COM_OUT_FRAC))
    # ground reaction, joint loads: medians over the episode's stances of each stance's peaks
    kin = kinetics(fv, m, down)
    jp = {j: kin[j] for j in J3}
    out.update(grf_peak_bw=kin["grf_peak_bw"], grf_loading_rate_bw_s=kin["grf_loading_rate_bw_s"],
               joint_peak_nm_per_kg=jp, kinetic_stances=kin["stances"])
    has = thr.get("grf_peak_bw") is not None and kin["stances"] > 0
    out["grf"] = None if not has else bool(kin["grf_peak_bw"] <= thr["grf_peak_bw"]
                                           and kin["grf_loading_rate_bw_s"] <= thr["grf_loading_rate_bw_s"])
    out["joints"] = None if not has else bool(all(jp[j] <= thr["joint_peak_nm_per_kg"][j] for j in J3))
    # muscle effort
    sat = tr["act_sat"].astype(float)
    k = min(SAT_WIN, len(sat))
    worst_sat = float(np.stack([np.convolve(sat[:, i], np.ones(k) / k, "valid") for i in range(sat.shape[1])]).max())
    out.update(worst_saturation_frac=worst_sat, metabolic_j_per_kg_m=tr["metabolic_j_per_kg_m"], effort=bool(worst_sat <= SAT_FRAC))
    # contact: the worst loaded stance's slide; time at joint limits
    sl, at_lim = slide(tr["sphere"], tr["sphere_side"], down), float(np.mean(tr["at_limit"]))
    has = thr.get("slide_m") is not None and thr.get("at_limit_frac") is not None
    out.update(worst_slide_m=sl, at_limit_frac=at_lim,
               contact=None if not has else bool(sl <= thr["slide_m"] and at_lim <= thr["at_limit_frac"]))
    checks = ("style", "plan", "grf", "joints", "effort", "contact")
    out["accepted"] = bool(not out["fell"] and all(out[c] is True for c in checks))
    out["unchecked"] = [c for c in checks if out[c] is None]          # pending: no threshold yet (style before the prior)
    out["failed"] = [c for c in checks if out[c] is False] + (["fell"] if out["fell"] else [])
    # G1: share of each curve inside the real band
    band = thr.get("band")
    if band:
        cols = [JOINTS.index(c) for c in CURVES[:3]]
        frac: dict[str, list[float]] = {c: [] for c in CURVES}
        for a, b in stances(down[:, 1], tuple(thr.get("stance_s", STANCE_S))):
            seg = np.c_[m[a:b + 1, cols], fv[a:b + 1, 1]]
            for k, c in enumerate(CURVES):
                y = resample(seg[:, k], 0, b - a)
                lo, hi = np.array(band[c][0]), np.array(band[c][1])
                frac[c].append(float(((y >= lo) & (y <= hi)).mean()))
        out["in_band"] = {c: float(np.mean(v)) if v else None for c, v in frac.items()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=None)
    r = calibrate(ap.parse_args().prior)
    for cls, t in r["kinetics"].items():
        print(f"Camargo {cls:12s} clips {t['clips']:4d} stances {t['stances']:5d}  GRF {t['grf_peak_bw']:.2f} BW "
              f"{t['grf_loading_rate_bw_s']:.1f} BW/s  moments " + " ".join(f"{k.split('_')[0]} {v:.2f}" for k, v in
                                                                     t["joint_peak_nm_per_kg"].items()))
    if "kinetics_run" in r:
        t = r["kinetics_run"]
        print(f"AB01 running        clips {t['clips']:4d} stances {t['stances']:5d}  GRF {t['grf_peak_bw']:.2f} BW "
              f"{t['grf_loading_rate_bw_s']:.1f} BW/s  moments " + " ".join(f"{k.split('_')[0]} {v:.2f}" for k, v in
                                                                     t["joint_peak_nm_per_kg"].items()))
        b = t["by_speed"]
        for k, v in enumerate(b["speed_mps"]):
            print(f"  running at {v:4.2f} m/s: GRF {b['grf_peak_bw'][k]:.2f} BW {b['grf_loading_rate_bw_s'][k]:.1f} BW/s"
                  "  moments " + " ".join(f"{j.split('_')[0]} {y[k]:.2f}" for j, y in b["joint_peak_nm_per_kg"].items()))
    for mode, t in r["contact"].items():
        print(f"contact {mode:18s} clips {t['clips']:4d}  slide {100 * t['slide_m']:.1f} cm  at limits {t['at_limit_frac']:.1%}")
    for split, st in r["style"].items():
        print(f"style {split:12s} " + "  ".join(f"{c} {v:.3f}" for c, v in st.items()))
    print("->", data.OUT / "accept.json")
