# gait_planner

A muscle policy that walks on ramps and stairs from a sparse plan: per foot, when it is down and where, and a
centre-of-mass band. The joints are free. A frozen motion-style prior (SMP) scores how human the motion looks, and
human kinetic ranges come from Camargo 2021 inverse dynamics. Plan:
https://claude.ai/artifact/Mv4JnEM6LDP3fDAusVAAan

Started 2026-11-14. Data: the seed library's Rajagopal v3 export (`~/Jinwoo/seed_library`).

## Environments

- `.venv/` (`./setup_env.sh`): SPRINT from branch `gait-planner-engine` of `~/Jinwoo/GAIT` (v1.0.0 = a44107e plus
  the per-model specific tension, 42c0606), installed from its pinned worktree `~/Jinwoo/worktrees/sprint-gait-planner`,
  with torch and the rest of `requirements.lock`. Every `python` below is `.venv/bin/python`.
- `~/miniconda3/envs/bolt/bin/python` (OpenSim 4.6): `camargo_id.py solve`.
- `uv run --with mat-io --with pandas`: `camargo_id.py extract` (reads Camargo's MATLAB tables).

## Order of work (GPU steps are yours to launch; a fresh `--out` per run)

| Step | Command | Device |
|---|---|---|
| 0. data (done) | `python -m gp.data` -> `data/v1/` | CPU, 2 min |
| 0b. kinetics (done) | `camargo_id.py extract`, then `camargo_id.py solve` -> `data/v1/kinetics.npz` | CPU |
| 0c. filter thresholds (kinetic part done) | `python -m gp.accept` -> `data/v1/accept.json` | CPU |
| 0d. EMG envelopes (done) | `camargo_emg.py` (uv, mat-io) -> `data/v1/emg.npz` | CPU |
| 1. style prior | `python train_prior.py --out runs/prior_v1` (200k iterations) | GPU |
| 2. gate G0 | `python check_prior.py runs/prior_v1/prior.pt` -> `g0.json` | GPU or CPU |
| 3. baseline | `python train.py --task baseline --out runs/baseline_v1 --hours H` | GPU |
| 4. planner | `python train.py --task planner --prior runs/prior_v1/prior.pt --out runs/planner_v1 --hours H` | GPU |
| 2b. style threshold | `python -m gp.accept --prior runs/prior_v1/prior.pt` (adds the style check) | GPU or CPU |
| 5. evaluate | `python evaluate.py runs/<run> --prior runs/prior_v1/prior.pt` -> `eval.json` (with the filter) | GPU or CPU |
| 6. gate G1 | `python g1.py runs/planner_v1 runs/baseline_v1` -> `g1.json` | CPU |
| 7. EMG timing | `python emg_report.py runs/planner_v1 runs/baseline_v1` -> `emg_timing.json`, `emg_<class>.png` | CPU |

Launch training detached (`setsid nohup ... > runs/<name>.log 2>&1 < /dev/null &`), as the docstrings show.
Baseline and planner share body, clips, physics, delays, effort and learner, so equal `--hours` is matched compute.

Checks: `python -m pytest tests -q` (CPU, ~40 s). Terrain lookups match SPRINT; both tasks step with finite rewards
and consistent observation layouts; the simulation's style record equals the data's; the actor never reads the full
pose; the prior trains; PPO runs an iteration.

## Layout

| Path | What |
|---|---|
| `gp/data.py` | builds `data/v1` from the seed library: split, 40 Hz clips, terrain profiles, reset lifts, plans, style records |
| `gp/terrain.py` | seed-library terrain -> SPRINT profiles; height lookups (numpy, torch) |
| `gp/smp.py` | the style prior: TinyMDM's DiT, diffusion schedule, window features, SDS reward (port of MimicKit) |
| `gp/tasks.py` | `TerrainImitation` (full-pose baseline) and `PlanTracking` (the planner's task), both on SPRINT's imitation recipe |
| `train_prior.py`, `check_prior.py` | train the prior; gate G0 |
| `train.py`, `evaluate.py` | train a policy (PPO); evaluate it on held-out clips |
| `camargo_id.py` | Camargo inverse dynamics on Rajagopal v3 |
| `gp/accept.py`, `g1.py` | the acceptance filter (six checks, thresholds from real clips); gate G1 |
| `camargo_emg.py`, `gp/emg.py`, `emg_report.py` | Camargo EMG envelopes; EMG timing of policies against held-out subjects |
| `gp/terrain_gen.py` | random terrain courses (level landings between every ramp and flight) for held-out tests and the loop |
| `tests/test_smoke.py` | the CPU checks |

## Data v1 (`data/v1/split.json`)

- AB01 + Camargo 2021: 6,533 clips, 13.8 h at 40 Hz. PARC obstacle clips are out of v1 (heightfield-only terrain).
- **Split:** `train` 4,944 clips (10.5 h); `test_subject` 3 Camargo subjects (AB06, AB09, AB24; drawn with seed
  2026), all their clips; `test_terrain` Camargo's 18 deg ramps and 178 mm stairs (stair_4) for everyone else.
  The held-out terrain is the steepest and tallest, so the terrain test is extrapolation.
- **Dropped:** 11 clips whose joint speed passes 20 rad/s anywhere (source IK spikes up to 132 rad/s).
- **Plans** (`plans.npz`): per frame and foot, down or not, the foothold of its current or next contact, time to
  touchdown and lift-off (capped at 1 s), and the whole-body centre of mass. Copied from the clips (v1).
- **Reset lift:** per frame, the root raise that takes every contact sphere out of the terrain (signed distance to
  the profile, as SPRINT's contact sees it): median 0.3 cm, p99 3 cm, max 9.6 cm.

## Choices (defaults; each is one constant)

- **Control at 40 Hz**, 75 physics steps (3 kHz), implicit contact, SCONE motor delays: SPRINT's recommended
  imitation recipe without EMG. The clips are resampled from 30 Hz.
- **Style prior:** MimicKit's settings (2 DiT blocks, T = 50, diffusion steps 22/15/8, scale 6, AdamW 2e-4, batch
  512, 200k iterations, EMA 0.995). Window 13 frames (0.325 s, MimicKit's 10 at 30 Hz), scored every step. Features
  are this body's own (see `gp/smp.py`): root height above the terrain under it, not above a flat floor. Not
  mirror-augmented.
- **Planner reward:** contact 0.15, foothold 0.15 (10 cm), centre-of-mass band 0.20 (+-10 cm forward, +-5 cm up and
  sideways, 5 cm tolerance beyond), style 0.50 (MimicKit's 50/50), minus the recipe's effort (metabolic 0.02, summed
  cubed activation 0.02, compartment balance 0.02); the same effort in the baseline.
- **Metabolic cost at 0.6 MPa** (`gp.data.SPECIFIC_TENSION`): Rajagopal's F0 = PCSA x 60 N/cm^2, and SPRINT's default
  0.25 MPa makes every muscle 2.4x heavy, so heat (charged per kg) read 2.4x high. From the same states the cost falls
  to 0.53-0.60x (9.4 -> 5.2 W/kg at 10 % excitation everywhere). Forces are unchanged (tested).
- **Observation:** the actor reads its body, a 2D height scan (17 points from -0.4 to 1.2 m ahead x 3 at -0.2, 0,
  0.2 m to the side, in its heading frame) and the plan at the
  recipe's lookahead taps (0-40 steps = 1 s). The critic also reads the full reference.
- **Falls:** the recipe's (pelvis over 0.3 m from the clip's, or tilted over 60 deg, for 3 steps).
- **G0 thresholds** (`check_prior.py`): real beats each degraded copy in >= 95 % of pairs; with the threshold at
  the degraded copies' 95th percentile, no held-out subject over 10 % below it. That second reading is mine.

## Kinetics (`data/v1/kinetics.json`)

Camargo's force plates through the same transforms as the kinematics, onto Rajagopal v3, OpenSim ID with every
force element removed. All 6,242 Camargo clips solve; 32-40 % of frames are valid (every foot that is down is on a
plate, and Camargo's own ID exists there). Against Camargo's own ID (their knee sign flipped), right leg, valid frames:

| Class | Hip r | Knee r | Ankle r | Knee moment p2-p98, ours (theirs), N m/kg |
|---|---|---|---|---|
| level | 0.961 | 0.958 | 0.968 | -0.85 to 0.42 (-0.80 to 0.46) |
| ramp_up | 0.987 | 0.959 | 0.961 | -0.93 to 0.58 (-0.90 to 0.60) |
| ramp_down | 0.978 | 0.956 | 0.878 | -1.73 to 0.29 (-1.75 to 0.30) |
| stairs_up | 0.993 | 0.996 | 0.997 | -1.14 to 0.35 (-1.13 to 0.37) |
| stairs_down | 0.982 | 0.995 | 0.995 | -1.40 to 0.23 (-1.41 to 0.24) |

## Acceptance filter and G1 (`gp/accept.py`, `g1.py`)

The plan doc's six checks, each threshold the 5th or 95th percentile of the same measure on the real clips of the
episode's split and terrain class: style (2 s windows), plan followed, ground reaction (peak force, loading rate),
joint loads, muscle effort (saturation), contact (sliding spheres, joint limits). A fallen episode is rejected first.
`evaluate.py` runs every episode through it. Two departures from the doc's drawing:
- joint loads are **net joint moments**, not knee and hip reaction forces: SPRINT's reaction pass refuses wrapped
  muscle paths, which Rajagopal v3 has, and real reaction forces would need muscle-force estimates;
- metabolic cost per metre is **reported, not gated**: there is no human metabolic data in v1.

G1's curve check compares the right stance (touchdown to lift-off), not the whole stride: a whole stride is almost
never on plates throughout.

## EMG timing (`camargo_emg.py`, `gp/emg.py`, `emg_report.py`)

Camargo's 11 right-side EMG channels (band-pass 20-400 Hz, rectified, 6 Hz low-pass) on v1's frames, mapped to
Rajagopal muscles (gluteus medius = its three compartments). Real and simulated signals become right-stride profiles
(touchdown to touchdown, 101 points), normalised per subject and channel (real) or per run, class and muscle
(simulation), with on/off relative to each muscle's own 10th-95th percentile range. Scores per class and muscle:
Pearson r of the mean profiles and on/off agreement; rectus femoris is shown, not scored (its surface signal is
mostly vasti crosstalk outside its swing burst). The ceiling, training subjects' real EMG against the held-out
subjects', is r 0.87-0.92 and agreement 0.86-0.89 by class.

## Heightfield terrain (`gp/terrain.py`, `gp/tasks.py`)

A clip's ground is either a SPRINT profile or a tile of one shared heightfield (`Data.set_heightfield`), for ground a
profile cannot describe: turns, winding stairs, platforms, gaps. `gp.terrain.atlas` packs tiles into one map and
returns each tile's offset; a clip on it has `terrain_kind` 1 and `hf_offset` its tile's, and the tasks set the
world's `terrain_id` to SPRINT's HFIELD and its `terrain_offset` at every reset. `gp.terrain.hf_height` (numpy and
torch) is SPRINT's surface exactly, block cells included; `gp.data.heightfield_gap` gives the reset lift on a tile.
One env mixes profile and heightfield clips. Tested: a stairs clip on a 2 cm tile of its own profile has the same
style record and, but within one cell of a riser, the same height scan.

## Random terrain (`gp/terrain_gen.py`)

Courses along +x of level runs, ramps and stair flights, as SPRINT profiles: level at both ends, a level landing
between every two features (no ramp into a ramp, no up-to-down reversal without level ground), one grade or one
rise and tread per feature, within +-2.5 m of the start. Sizes default to the training split's 5th-95th
percentiles (ramps 5-12.5 deg over 4.5-6 m, rises 0.10-0.175 m, treads 0.25-0.37 m, 3-8 steps);
`TerrainSpec.held_out()` samples the held-out range. Not used by v1 training (each episode replays a clip on its own
terrain, because the plan belongs to it): it is for evaluating on new ground and for the planner's loop (G3).

## Known limits

- Plans are copied from real clips; footholds are not predicted (plan doc, open question 2).
- v1's data is all profiles. Heightfield clips (the PARC obstacles, generated 2D courses) run in the tasks but are not
  built into the data yet.
- `gp/tasks.py` patches SPRINT's reference kinematics to run in chunks for the duration of setup (1.5M frames in one
  Data would take ~18 GB of GPU memory). Upstream fix: chunk in SPRINT.
