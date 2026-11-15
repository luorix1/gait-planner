"""EMG timing: Camargo's right-leg EMG over the right stride per terrain class (data/v1/emg.npz, camargo_emg.py), and the
simulation's activations of the same muscles on the same footing.

- **Stride profiles**: touchdown to the next touchdown of the right foot (strides of 0.6-2.0 s), 101 points. Real strides
  are whole (EMG needs no force plates); the real foot contact is the clip's contact flag, the simulation's is the right
  foot carrying 5 % of body weight (gp.accept).
- **Amplitude**: real envelopes are normalised per subject and channel (by that subject's 95th percentile over all its v1
  frames), simulated activations per run, class and muscle (95th percentile over the class's strides).
- **On or off**: a muscle is on above 25 % of the way from its 10th to its 95th percentile, on each side (relative, as
  SPRINT's ``emg_relative``: an absolute cut reads low-amplitude channels as always off).
- **Scores** per class and muscle: Pearson r between the simulated and real mean profiles, and on/off agreement (share of
  the 101 points where the simulation's majority on/off matches the real one). Rectus femoris is reported, not scored.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from . import accept, data

STRIDE_S = (0.6, 2.0)
ON_FRAC = 0.25
P_LO, P_HI = 10, 95


def load_emg() -> tuple[np.ndarray, list[str], dict[str, list[str]], tuple[str, ...]]:
    z = np.load(data.OUT / "emg.npz")
    return z["envelope"], list(z["channels"]), json.loads(str(z["muscles"])), tuple(z["unreliable"])


def strides(down_r: np.ndarray) -> list[tuple[int, int]]:
    """Right strides, touchdown to the next touchdown, of plausible length (a contact under way at the first frame is
    not a touchdown)."""
    td = [a for a, _ in accept.runs(down_r, 3) if a > 0]
    return [(a, b) for a, b in zip(td[:-1], td[1:], strict=True) if STRIDE_S[0] <= (b - a) / data.HZ <= STRIDE_S[1]]


def stride_profiles(down_r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``[strides, C, 101]`` of signals ``x`` ``[T, C]`` over every right stride whose frames are all finite."""
    out = [np.stack([accept.resample(x[a:b + 1, k], 0, b - a) for k in range(x.shape[1])])
           for a, b in strides(down_r) if np.isfinite(x[a:b + 1]).all()]
    return np.array(out).reshape(-1, x.shape[1], 101)


def reference(split: str) -> dict[str, dict[str, Any]]:
    """Per class: the real mean profile and on-probability ``[C, 101]`` (normalised per subject and channel) and the
    stride count, from the Camargo clips of ``split``."""
    c, p = np.load(data.OUT / "clips.npz"), np.load(data.OUT / "plans.npz")
    env, channels, _, _ = load_emg()
    subj = np.repeat(c["subject"], c["frames"])
    norm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in np.unique(c["subject"][c["source"] == "camargo2021"]):
        e = env[subj == s]
        norm[str(s)] = (np.nanpercentile(e, P_LO, 0), np.nanpercentile(e, P_HI, 0))
    prof: dict[str, list[np.ndarray]] = {}
    on: dict[str, list[np.ndarray]] = {}
    for i in np.flatnonzero((c["split"] == split) & (c["source"] == "camargo2021")):
        s, n = int(c["start"][i]), int(c["frames"][i])
        lo, hi = norm[str(c["subject"][i])]
        pr = stride_profiles(p["contact"][s:s + n, 1], env[s:s + n])
        if not len(pr):
            continue
        cls = str(c["terrain_class"][i])
        prof.setdefault(cls, []).append(pr / hi[None, :, None])
        on.setdefault(cls, []).append(pr > (lo + ON_FRAC * (hi - lo))[None, :, None])
    return {cls: dict(channels=channels, mean=np.concatenate(prof[cls]).mean(0), on_prob=np.concatenate(on[cls]).mean(0),
                      strides=int(sum(len(x) for x in prof[cls]))) for cls in prof}


def summarise(profiles: np.ndarray) -> dict[str, Any]:
    """Simulated stride profiles ``[strides, C, 101]`` of one class -> mean profile and on-probability, normalised as the
    module docstring says."""
    lo, hi = np.percentile(profiles, P_LO, axis=(0, 2)), np.percentile(profiles, P_HI, axis=(0, 2))
    hi = np.maximum(hi, 1e-6)
    return dict(mean=(profiles / hi[None, :, None]).mean(0),
                on_prob=(profiles > (lo + ON_FRAC * (hi - lo))[None, :, None]).mean(0), strides=len(profiles))


def compare(ref: dict[str, Any], sim: dict[str, Any], unreliable: tuple[str, ...] = ()) -> dict[str, Any]:
    """Per channel: Pearson r of the mean profiles and on/off agreement; and their means over the scored channels."""
    out: dict[str, Any] = {}
    for k, ch in enumerate(ref["channels"]):
        a, b = sim["mean"][k], ref["mean"][k]
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")
        agree = float(((sim["on_prob"][k] >= 0.5) == (ref["on_prob"][k] >= 0.5)).mean())
        out[ch] = dict(r=r, on_off_agreement=agree, scored=ch not in unreliable)
    scored = [v for v in out.values() if v["scored"]]
    out["mean_r"] = float(np.nanmean([v["r"] for v in scored]))
    out["mean_on_off_agreement"] = float(np.mean([v["on_off_agreement"] for v in scored]))
    return out
