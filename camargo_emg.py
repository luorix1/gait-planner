"""Camargo 2021 EMG envelopes on v1's 40 Hz frames: the human muscle timing the evaluation compares the policies' activations
with (plan doc, claims: EMG timing on held-out subjects).

Camargo recorded 11 surface EMG channels on the right side at 1 kHz. Each is band-passed 20-400 Hz, full-wave rectified and
low-passed at 6 Hz (all 4th-order, zero-phase), then sampled at the clip's v1 frame times (the same clock as the kinematics:
camargo_id.clip_times). Envelopes are stored raw (no amplitude normalisation; gp.emg normalises per subject and channel).

Channel -> Rajagopal v3 muscle(s), in ``MUSCLES``: gluteus medius is the mean of the model's three compartments; rectus femoris
is reported but kept out of summary scores (surface rectus femoris reads mostly vasti crosstalk outside its swing burst).

Run: cd ~/Jinwoo/gait_planner && VIRTUAL_ENV= uv run --no-project --python 3.11 --with mat-io --with pandas --with scipy \\
         python camargo_emg.py
Out: data/v1/emg.npz (envelope [frames, channels], NaN outside Camargo clips; channels; frames per clip found)
"""

from __future__ import annotations

import json
import zipfile

import numpy as np

import camargo_id as ci

CHANNELS = ("gastrocmed", "tibialisanterior", "soleus", "vastusmedialis", "vastuslateralis", "rectusfemoris", "bicepsfemoris",
            "semitendinosus", "gracilis", "gluteusmedius", "rightexternaloblique")
MUSCLES = {"gastrocmed": ["gasmed_r"], "tibialisanterior": ["tibant_r"], "soleus": ["soleus_r"], "vastusmedialis": ["vasmed_r"],
           "vastuslateralis": ["vaslat_r"], "rectusfemoris": ["recfem_r"], "bicepsfemoris": ["bflh_r"],
           "semitendinosus": ["semiten_r"], "gracilis": ["grac_r"], "gluteusmedius": ["glmed1_r", "glmed2_r", "glmed3_r"],
           "rightexternaloblique": ["ext_obl_r"]}
UNRELIABLE = ("rectusfemoris",)
HZ = 40.0


def envelope(x: np.ndarray, dt: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [20.0, 400.0], btype="band", fs=1.0 / dt)
    x = np.abs(filtfilt(b, a, x - x.mean(0), axis=0))
    b, a = butter(4, 6.0, fs=1.0 / dt)
    return filtfilt(b, a, x, axis=0)


def main() -> None:
    import matio
    c = np.load(ci.V1 / "clips.npz")
    names = [str(n) for n in c["name"]]
    env = np.full((int(c["frames"].sum()), len(CHANNELS)), np.nan, np.float32)
    zips = [zipfile.ZipFile(z) for z in sorted(ci.CAMARGO.glob("Subjects_Part*.zip"))]
    members = {m: z for z in zips for m in z.namelist() if m.endswith(".mat") and "/emg/" in m}
    by_trial: dict[tuple[str, str], list[int]] = {}
    for i, n in enumerate(names):
        if c["source"][i] == "camargo2021":
            m = ci.canonical_meta(n)
            by_trial.setdefault((m["subject"].removeprefix("camargo_"), m["trial"]), []).append(i)
    done = missing = 0
    tmp = ci.RAW / "_tmp_emg.mat"
    for (subj, trial), clips in sorted(by_trial.items()):
        mode, tname = trial.split("/")
        hit = [m for m in members if m.startswith(f"{subj}/") and m.endswith(f"/{mode}/emg/{tname}.mat")]
        if len(hit) != 1:
            missing += len(clips)
            continue
        with members[hit[0]].open(hit[0]) as f:
            tmp.write_bytes(f.read())
        t = matio.load_from_mat(str(tmp))["data"]
        time = t["Header"].to_numpy(float)
        e = envelope(t[list(CHANNELS)].to_numpy(float), float(np.median(np.diff(time))))
        for i in clips:
            t30 = ci.clip_times(names[i])
            nf = int(c["frames"][i])
            tv = t30[0] + np.arange(nf) / HZ
            s = int(c["start"][i])
            env[s:s + nf] = np.stack([np.interp(tv, time, e[:, k], left=np.nan, right=np.nan) for k in range(len(CHANNELS))], 1)
            done += 1
        if done % 500 < len(clips):
            print(f"{done} clips", flush=True)
    tmp.unlink(missing_ok=True)
    np.savez(ci.V1 / "emg.npz", envelope=env, channels=np.array(CHANNELS), muscles=json.dumps(MUSCLES),
             unreliable=np.array(UNRELIABLE))
    print(f"envelopes for {done} clips; {missing} clips had no EMG file")


if __name__ == "__main__":
    main()
