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
| 1. style prior | `python train_prior.py --out runs/prior_v1` (200k iterations) | GPU |
| 2. gate G0 | `python check_prior.py runs/prior_v1/prior.pt` -> `g0.json` | GPU or CPU |
| 3. baseline | `python train.py --task baseline --out runs/baseline_v1 --hours H` | GPU |
| 4. planner | `python train.py --task planner --prior runs/prior_v1/prior.pt --out runs/planner_v1 --hours H` | GPU |
| 5. evaluate | `python evaluate.py runs/<run> --prior runs/prior_v1/prior.pt` -> `eval.json` | GPU or CPU |

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
- **Observation:** the actor reads its body, a height scan (17 points, -0.4 to 1.2 m along x) and the plan at the
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

Not yet built: the acceptance filter that uses these ranges (phase G1).

## Known limits

- Plans are copied from real clips; footholds are not predicted (plan doc, open question 2).
- Terrain profiles run along x; every v1 terrain does, the PARC obstacles do not.
- `gp/tasks.py` patches SPRINT's reference kinematics to run in chunks for the duration of setup (1.5M frames in one
  Data would take ~18 GB of GPU memory). Upstream fix: chunk in SPRINT.
