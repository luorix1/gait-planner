"""The acceptance filter (plan doc, gate G1): six checks an episode must pass to join the planner's dataset. Kinetic
thresholds are Camargo's per terrain class (all its subjects, ``data.REF``; user 2026-11-15: the most nominal human
data); the style threshold comes from the active dataset's real clips of the same split and class. Camargo is all
walking, so running takes AB01's treadmill running instead (``ab01_id.py``: 15 clips at 2.0-2.7 m/s, level and
+-5 deg, one subject; one ``kinetics_run`` entry, used for running on level ground and ramps). A class neither has
(PARC's obstacles: jumps, drops, platforms) leaves the GRF and joint checks unchecked.

**Running limits rise with speed.** The dataset's running is faster than AB01's (PARC: median 3.4 m/s, up to 4.5), and
AB01's peaks rise with speed (per m/s: GRF +0.22 BW, matching ~0.2 in Nilsson & Thorstensson 1989; ankle +0.45, hip
+0.87 N m/kg; knee flat, as Schache et al. 2011 found for knee stance work). So each running limit is a function of the
episode's target speed: the larger of AB01's pooled 95th percentile and the 95 % prediction bound of a straight line
through AB01's level clips' peaks against speed (``RUN_PI``). The bound widens with distance from AB01's speeds, which
is the headroom for extrapolating from one subject at 2.0-2.7 m/s. Tabulated every 0.25 m/s up to the dataset's
fastest running clip; faster than that, the fastest's limits hold.

| Check | Rejects when | Threshold from |
|---|---|---|
| Style | a 2 s window's mean SDS loss is above the real 95th percentile (its score below the 5th) | the prior on real windows |
| Plan followed | a planned contact (3+ frames) with under half its frames in contact; a stance more than 15 cm from its planned foothold; the centre of mass out of its band over 10 % of steps | fixed |
| Ground reaction | peak vertical force or peak loading rate above the real 95th percentile | Camargo's plates (running: AB01's treadmill) |
| Joint loads | peak net moment at a hip, knee or ankle above the real 95th percentile | Camargo (running: AB01) ID on Rajagopal v3 |
| Muscle effort | a muscle above 0.95 activation over 10 % of any 1 s window | fixed |
| Contact | a touching contact sphere sliding over 2 cm, or a coordinate within 1 deg of a limit over 5 % of steps | fixed |

An episode that fell is rejected before any check. Both sides are processed alike: 40 Hz, forces and moments
low-passed at 6 Hz (Camargo's plates and ID coordinates were), a foot down above 5 % of body weight.

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
Out: data/<dataset>/accept.json: ``kinetics`` per class, ``style`` per split and class
"""

from __future__ import annotations

import argparse
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
SLIP_M, TOUCH_M = 0.02, 0.005
LIMIT_MARGIN, LIMIT_FRAC = math.radians(1.0), 0.05
STANCE_S = (0.3, 1.5)
RUN_STANCE_S = (0.12, 0.45)
RUN_CLASSES = ("level", "ramp_up", "ramp_down")     # AB01 ran on these; not PARC's obstacles
RUN_PI = 0.95                                        # one-sided prediction bound of the speed trend (running limits)
PCT_HI, PCT_LO = 95, 5


# --- shared signal processing ------------------------------------------------------------------------------------
def lowpass(x: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    if len(x) < 16:
        return x
    b, a = butter(2, CUTOFF_HZ / (0.5 * data.HZ))
    return filtfilt(b, a, x, axis=0)


def runs(c: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    return [(a, b) for a, b in data._intervals(c) if b - a + 1 >= min_len]


def grf_measures(fv: np.ndarray, down: np.ndarray) -> tuple[float, float]:
    """Peak vertical force and peak loading rate (BW, BW/s; the steepest rise in a stance's first half), both feet."""
    peak, lr = float(np.nanmax(fv)) if np.isfinite(fv).any() else float("nan"), float("nan")
    for k in range(fv.shape[1]):
        for a, b in runs(down[:, k], 3):
            seg = fv[a:a + max(2, (b - a + 1) // 2), k]
            if np.isfinite(seg).all() and len(seg) > 1:
                lr = float(np.nanmax([lr, np.diff(seg).max() * data.HZ]))
    return peak, lr


def stances(down_r: np.ndarray, span: tuple[float, float] = STANCE_S) -> list[tuple[int, int]]:
    """Right stances, touchdown to lift-off, of plausible length ``span`` (s; one cut short by the clip's ends is
    dropped)."""
    return [(a, b) for a, b in runs(down_r, 3) if a > 0 and b < len(down_r) - 1 and span[0] <= (b - a) / data.HZ <= span[1]]


def resample(x: np.ndarray, a: int, b: int) -> np.ndarray:
    return np.interp(np.linspace(a, b, 101), np.arange(len(x)), x)


# --- calibration on real clips -----------------------------------------------------------------------------------
def _real_clips(src: Any = None, kinetics: str = "kinetics.npz", source: str = "camargo2021",
                mode: str | None = None) -> list[dict[str, Any]]:
    """Per real clip of ``source`` (and ``mode``) in dataset ``src`` (default ``data.REF``, Camargo): its class,
    kinetics from ``kinetics`` (NaN where that leg is invalid: validity is per leg, see camargo_id.py and ab01_id.py)
    and both feet's contacts."""
    src = src or data.REF
    c = np.load(src / "clips.npz")
    K = np.load(src / kinetics)
    p = np.load(src / "plans.npz")
    names = list(K["coord_names"])
    jcol = [names.index(j) for j in JOINTS]
    out = []
    for i in np.flatnonzero((c["source"] == source) & ((c["mode"] == mode) if mode else True)):
        s, n = int(c["start"][i]), int(c["frames"][i])
        m = K["moment_nm_per_kg"][s:s + n][:, jcol].astype(float)
        fv = K["grf_bw"][s:s + n][..., 1].astype(float)            # (l, r)
        if np.isfinite(m).sum(0).min() < data.HZ:      # under 1 s of valid frames for some joint
            continue
        out.append(dict(cls=str(c["terrain_class"][i]), m=m, fv=fv, down=p["contact"][s:s + n], start=s, frames=n,
                        speed=float(c["speed_mps"][i])))
    return out


def _peaks(r: dict[str, Any]) -> dict[str, float]:
    """One clip's filter measures: GRF peak and loading rate, and each joint's peak moment over both legs."""
    g = grf_measures(r["fv"], r["down"])
    pk = np.nanmax(np.abs(r["m"]), 0)
    return dict(grf_peak_bw=g[0], grf_loading_rate_bw_s=g[1],
                **{j: float(max(pk[JOINTS.index(f"{j}_l")], pk[JOINTS.index(f"{j}_r")]))
                   for j in ("hip_flexion", "knee_angle", "ankle_angle")})


def _by_speed(rs: list[dict[str, Any]], base: dict[str, Any], top: float) -> dict[str, Any]:
    """Running limits against target speed (module docstring): per measure, at speeds 2.0 to ``top`` every 0.25 m/s,
    the larger of ``base``'s limit and the ``RUN_PI`` prediction bound of a line through the level clips' peaks."""
    from scipy import stats
    lv = [r for r in rs if r["cls"] == "level"]
    x = np.array([r["speed"] for r in lv])
    v = np.unique(np.r_[np.arange(2.0, top, 0.25), top]).round(3)
    pk = [_peaks(r) for r in lv]
    t = stats.t.ppf(RUN_PI, len(x) - 2)
    sxx = ((x - x.mean()) ** 2).sum()

    def bound(key: str, floor: float) -> list[float]:
        y = np.array([q[key] for q in pk])
        slope, icpt = np.polyfit(x, y, 1)
        se = np.sqrt(((y - slope * x - icpt) ** 2).sum() / (len(x) - 2))
        pi = slope * v + icpt + t * se * np.sqrt(1 + 1 / len(x) + (v - x.mean()) ** 2 / sxx)
        return np.maximum(pi, floor).round(3).tolist()

    return dict(speed_mps=v.tolist(), grf_peak_bw=bound("grf_peak_bw", base["grf_peak_bw"]),
                grf_loading_rate_bw_s=bound("grf_loading_rate_bw_s", base["grf_loading_rate_bw_s"]),
                joint_peak_nm_per_kg={j: bound(j, base["joint_peak_nm_per_kg"][j])
                                      for j in ("hip_flexion", "knee_angle", "ankle_angle")})


def _limits(rs: list[dict[str, Any]], span: tuple[float, float] = STANCE_S) -> dict[str, Any]:
    """The kinetic limits of a set of real clips: 95th percentiles of their per-clip peaks, and the G1 bands."""
    peaks = np.array([np.nanmax(np.abs(r["m"]), 0) for r in rs])                             # [clips, 6], per leg
    g = np.array([grf_measures(r["fv"], r["down"]) for r in rs])
    cur = np.array([x for r in rs for x in _curves(r["m"], r["fv"], r["down"][:, 1], span)])
    joint_peak = {j: float(np.nanpercentile(peaks[:, [JOINTS.index(f"{j}_l"), JOINTS.index(f"{j}_r")]], PCT_HI))
                  for j in ("hip_flexion", "knee_angle", "ankle_angle")}
    return dict(clips=len(rs), stances=len(cur), grf_peak_bw=float(np.nanpercentile(g[:, 0], PCT_HI)),
                grf_loading_rate_bw_s=float(np.nanpercentile(g[:, 1], PCT_HI)), joint_peak_nm_per_kg=joint_peak,
                band={c: [np.percentile(cur[:, k], PCT_LO, 0).round(4).tolist(),
                          np.percentile(cur[:, k], PCT_HI, 0).round(4).tolist()] for k, c in enumerate(CURVES)}
                if len(cur) >= 10 else None, stance_s=list(span))


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
                               kinetics_from=str(data.REF), kinetics={}, style={})
    clips = _real_clips()
    for cls in sorted({r["cls"] for r in clips}):
        res["kinetics"][cls] = _limits([r for r in clips if r["cls"] == cls])
    if (data.OUT / "kinetics_ab01.npz").exists():                       # ab01_id.py
        run = _real_clips(data.OUT, "kinetics_ab01.npz", "ab01", "run")
        res["kinetics_run"] = dict(_limits(run, RUN_STANCE_S), source="AB01 treadmill running, ab01_id.py",
                                   classes=list(RUN_CLASSES))
        c = np.load(data.OUT / "clips.npz")
        top = float(c["speed_mps"][c["mode"] == "run"].max())          # the dataset's fastest running clip
        res["kinetics_run"]["by_speed"] = _by_speed(run, res["kinetics_run"], top)
    for split in SPLITS:
        style = _style_losses(prior, split) if prior else {}
        res["style"][split] = {cls: float(np.percentile(v, PCT_HI)) for cls, v in style.items()}
    (data.OUT / "accept.json").write_text(json.dumps(res, indent=1))
    return res


def thresholds(acc: dict[str, Any] | None, split: str, cls: str, mode: str,
               speed: float | None = None) -> dict[str, Any] | None:
    """One episode's thresholds from ``accept.json``: Camargo's kinetics for its class (none when Camargo lacks the
    class), or AB01's running kinetics when running on level ground or a ramp (none on obstacles), at the episode's
    target ``speed`` (m/s; the reference clip's or the plan's) when given, and the style threshold of its split and
    class."""
    if not acc:
        return None
    kin = acc["kinetics"].get(cls) if mode != "run" else acc.get("kinetics_run") if cls in RUN_CLASSES else None
    if kin and mode == "run" and speed is not None and "by_speed" in kin:
        b = kin["by_speed"]
        at = lambda y: float(np.interp(speed, b["speed_mps"], y))  # noqa: E731
        kin = dict(kin, grf_peak_bw=at(b["grf_peak_bw"]), grf_loading_rate_bw_s=at(b["grf_loading_rate_bw_s"]),
                   joint_peak_nm_per_kg={j: at(y) for j, y in b["joint_peak_nm_per_kg"].items()})
    return dict(kin or {}, style_loss=acc["style"].get(split, {}).get(cls))


# --- the filter --------------------------------------------------------------------------------------------------
def check(tr: dict[str, Any], thr: dict[str, Any] | None) -> dict[str, Any]:
    """One episode's trace (``evaluate.py``: per step ``sds``, ``plan_contact``, ``foothold_ref``, ``com_out``,
    ``fv``, ``foot``, ``moment``, ``act_sat``, ``at_limit``, ``sphere``, ``touching``; and ``fell``,
    ``metabolic_j_per_kg_m``) against its class's thresholds -> pass per check, accepted, the numbers behind them,
    and the G1 in-band fraction per curve."""
    fv = lowpass(tr["fv"])
    down = tr["fv"] > DOWN_BW
    m = lowpass(tr["moment"])
    out: dict[str, Any] = dict(fell=bool(tr["fell"]))
    # style
    sds = tr["sds"][np.isfinite(tr["sds"])]
    if len(sds):
        k = min(STYLE_WIN, len(sds))
        worst = float(np.convolve(sds, np.ones(k) / k, "valid").max())
    else:
        worst = float("nan")
    lim = thr.get("style_loss") if thr else None
    out["style_worst_loss"] = worst
    out["style"] = None if lim is None or not np.isfinite(worst) else bool(worst <= lim)
    # plan followed
    missed_contact = sum(down[a:b + 1, s].mean() < CONTACT_HALF for s in range(2) for a, b in runs(tr["plan_contact"][:, s], 3))
    off = [float(np.linalg.norm((tr["foot"][a:b + 1, s].mean(0) - tr["foothold_ref"][(a + b) // 2, s])[[0, 2]]))
           for s in range(2) for a, b in runs(down[:, s], 3) if tr["plan_contact"][(a + b) // 2, s]]
    com_out = float(tr["com_out"].mean())
    out.update(missed_contacts=int(missed_contact), worst_foothold_m=max(off, default=0.0), com_out_frac=com_out,
               plan=bool(missed_contact == 0 and max(off, default=0.0) <= FOOTHOLD_MISS_M and com_out <= COM_OUT_FRAC))
    # ground reaction, joint loads
    peak, lr = grf_measures(fv, down)
    jp = {j: float(np.abs(m[:, [JOINTS.index(f"{j}_l"), JOINTS.index(f"{j}_r")]]).max())
          for j in ("hip_flexion", "knee_angle", "ankle_angle")}
    out.update(grf_peak_bw=peak, grf_loading_rate_bw_s=lr, joint_peak_nm_per_kg=jp)
    kin = bool(thr and "grf_peak_bw" in thr)
    out["grf"] = None if not kin else bool(peak <= thr["grf_peak_bw"] and lr <= thr["grf_loading_rate_bw_s"])
    out["joints"] = None if not kin else bool(all(jp[j] <= thr["joint_peak_nm_per_kg"][j] for j in jp))
    # muscle effort
    sat = tr["act_sat"].astype(float)
    k = min(SAT_WIN, len(sat))
    worst_sat = float(np.stack([np.convolve(sat[:, i], np.ones(k) / k, "valid") for i in range(sat.shape[1])]).max())
    out.update(worst_saturation_frac=worst_sat, metabolic_j_per_kg_m=tr["metabolic_j_per_kg_m"], effort=bool(worst_sat <= SAT_FRAC))
    # contact: slip of each touching sphere over its touching run; time at joint limits
    slip = 0.0
    for i in range(tr["touching"].shape[1]):
        for a, b in runs(tr["touching"][:, i], 3):
            d = np.linalg.norm((tr["sphere"][a:b + 1, i] - tr["sphere"][a, i])[:, [0, 2]], axis=1)
            slip = max(slip, float(d.max()))
    at_lim = float(tr["at_limit"].mean())
    out.update(worst_slip_m=slip, at_limit_frac=at_lim, contact=bool(slip <= SLIP_M and at_lim <= LIMIT_FRAC))
    checks = ("style", "plan", "grf", "joints", "effort", "contact")
    out["accepted"] = bool(not out["fell"] and all(out[c] is not False for c in checks))
    out["unchecked"] = [c for c in checks if out[c] is None]          # no threshold yet (style before the prior)
    out["failed"] = [c for c in checks if out[c] is False] + (["fell"] if out["fell"] else [])
    # G1: share of each curve inside the real band
    band = thr.get("band") if thr else None
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
    for split, st in r["style"].items():
        print(f"style {split:12s} " + "  ".join(f"{c} {v:.3f}" for c, v in st.items()))
    print("->", data.OUT / "accept.json")
