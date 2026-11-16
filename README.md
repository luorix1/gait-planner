# gait_planner

A muscle policy that walks on ramps and stairs from a sparse plan: per foot, when it is down and where, a
centre-of-mass band and the pelvis orientation. The joints are free. A frozen motion-style prior (SMP) scores how human
the motion looks, and human kinetic ranges come from Camargo 2021 (walking) and AB01 (running) inverse dynamics. Plan:
https://claude.ai/artifact/JEJTXyCWPxEBVHdF3iomaD

Started 2026-11-14. Data: the seed library's Rajagopal v3 export (`~/Jinwoo/seed_library`).

## Environments

- `.venv/` (`./setup_env.sh`): SPRINT from branch `gait-planner-engine` of `~/Jinwoo/GAIT` (v1.0.0 = a44107e plus
  the per-model specific tension, 42c0606), installed from its pinned worktree `~/Jinwoo/worktrees/sprint-gait-planner`,
  with torch and the rest of `requirements.lock`. Every `python` below is `.venv/bin/python`.
- `~/miniconda3/envs/bolt/bin/python` (OpenSim 4.6): `camargo_id.py solve`, `ab01_id.py`.
- `uv run --with mat-io --with pandas`: `camargo_id.py extract` (reads Camargo's MATLAB tables).

## Order of work (GPU steps are yours to launch; a fresh `--out` per run)

| Step | Command | Device |
|---|---|---|
| 0. data (done) | `python -m gp.data` -> `data/<dataset>/` (the active one, `gp.data.DATASET`; `GP_DATASET=v3` or an argument picks another) | CPU, 15 s |
| 0b. Camargo kinetics (done) | `camargo_id.py extract`, then `camargo_id.py solve` -> `data/v1/kinetics.npz` | CPU |
| 0c. filter thresholds (done) | `python -m gp.accept` -> `data/<dataset>/accept.json` (Camargo kinetics per class) | CPU |
| 0d. Camargo EMG (done) | `camargo_emg.py` (uv, mat-io) -> `data/v1/emg.npz` | CPU |
| 0e. AB01 running kinetics (done) | `ab01_id.py` (bolt env) -> `data/<dataset>/kinetics_ab01.npz`, then `python -m gp.accept` | CPU, 2 min |
| 1. style prior | `python train_prior.py --out runs/prior_v1` (200k iterations) | GPU |
| 2. gate G0 | `python check_prior.py runs/prior_v1/prior.pt` -> `g0.json` | GPU or CPU |
| 3. baseline | `python train.py --task baseline --out runs/baseline_v1 --hours H` | GPU |
| 4. planner | `python train.py --task planner --prior runs/prior_v1/prior.pt --out runs/planner_v1 --hours H` | GPU |
| 2b. style threshold | `python -m gp.accept --prior runs/prior_v1/prior.pt` (adds the style check) | GPU or CPU |
| 5. evaluate | `python evaluate.py runs/<run> --prior runs/prior_v1/prior.pt` -> `eval.json` (with the filter) | GPU or CPU |
| 6. gate G1 | `python g1.py runs/planner_v1 runs/baseline_v1` -> `g1.json` | CPU |
| 7. EMG timing | `python emg_report.py runs/planner_v1 runs/baseline_v1` -> `emg_timing.json`, `emg_<class>.png` | CPU |
| 8. planner network | `python train_planner.py --out runs/planner_net_v1` (200k iterations) | GPU |
| 9. check it | `python check_planner.py runs/planner_net_v1/planner.pt` -> `check/check.json`, `check/course_*.png` | GPU or CPU |

Launch training detached (`setsid nohup ... > runs/<name>.log 2>&1 < /dev/null &`), as the docstrings show.
Baseline and planner share body, clips, physics, delays, effort and learner, so equal `--hours` is matched compute.

Checks: `python -m pytest tests -q` (CPU, ~2 min). Terrain lookups match SPRINT; both tasks step with finite rewards
and consistent observation layouts; the simulation's style record equals the data's; the actor never reads the full
pose; the prior trains; PPO runs an iteration.

## Layout

| Path | What |
|---|---|
| `gp/data.py` | builds `data/<dataset>` from the seed library: split, 40 Hz clips, terrain (profiles, PARC heightfield tiles), reset lifts, plans, style records |
| `gp/terrain.py` | seed-library terrain -> SPRINT profiles; height lookups (numpy, torch) |
| `gp/smp.py` | the style prior: TinyMDM's DiT, diffusion schedule, window features, SDS reward (port of MimicKit) |
| `gp/tasks.py` | `TerrainImitation` (full-pose baseline) and `PlanTracking` (the planner's task), both on SPRINT's imitation recipe |
| `train_prior.py`, `check_prior.py` | train the prior; gate G0 |
| `train.py`, `evaluate.py` | train a policy (PPO); evaluate it on held-out clips |
| `camargo_id.py`, `ab01_id.py` | inverse dynamics on Rajagopal v3: Camargo (walking references), AB01 treadmill (running) |
| `gp/accept.py`, `g1.py` | the acceptance filter (six checks, thresholds from real clips); gate G1 |
| `camargo_emg.py`, `gp/emg.py`, `emg_report.py` | Camargo EMG envelopes; EMG timing of policies against held-out subjects |
| `gp/terrain_gen.py` | random terrain courses (level landings between every ramp and flight) for held-out tests and the loop |
| `gp/free.py` | episodes without a clip: generated courses as free clips; the stand-in plan |
| `gp/planner.py`, `train_planner.py`, `check_planner.py` | the planner network (diffusion over plan windows); train it; score it before physics |
| `seed_viewer/` | local debug viewer of the whole seed library (clip on its terrain as SPRINT poses it, QA numbers, traces, flags); 127.0.0.1 only |
| `tests/test_smoke.py` | the CPU checks |

## Data (`data/<dataset>/split.json`)

**v3 (2026-11-16; active once today's v2 prior exams finish).** v2 without PARC's physics-tracker replays
(`gp.data.SIM_REPLAY`): PARC's `dm_motion_recorder` appends `_dm` to the motion its torque-driven humanoid produced
while tracking a clip, and those were 16.0 of v2's 17.9 PARC minutes (all running, gap jumps, jump strides, stairs and
walking, half the platforms). Only PARC's 31 untracked platform clips stay (1.9 min). 322 clips, 48 min: `train` 243
(36 min), `test_trial` 57, `test_terrain` 22. Real jumps and sprints are to come from Bones SEED instead.

**v2 (2026-11-15).** AB01 (below) plus PARC's clips, so the loop starts with every skill it is
meant to combine: 498 clips, 1.07 h. PARC adds 207 clips (18 min, one performer; 89 % of it PARC's physics-tracker
replays, which v3 drops): level running (3.3 m/s median), platforms, gap jumps, 5 stair clips and 2 walks. Its terrain is a box grid
(0.39 m cells), built into one shared heightfield; each clip stands on its own tile. Left out (`gp.data.EXCLUDE`):
clips that use the hands, 3 `_mdm_` clips (likely PARC generator output), the 5 `mid_climbing` clips (PARC labels a
hand down in 3-5 % of frames) and `parc_dm_platform_024_dm` (swing toe 7.5 cm into a platform face). PARC is split by
motion (tracked, untracked, straightened and jittered copies together): `train` 406 clips (51 min), `test_trial` 70
(10 min), `test_terrain` 22 (AB01's). Kept: `stairs_001`'s swing toe enters a riser by 3.5 cm for 0.15 s, and
landing toes sink 5-6 cm for a few frames in the jumps. Our toes reach past PARC's foot box, so toes that pass close
to a face in PARC's data poke into it here.

**ab01 (`data/ab01`).** Our AB01 pilot alone: 291 clips, 46 min at 40 Hz, one subject.
- Treadmill standing (7 grades) and locomotion (-15 to 15 deg, 0.4-2.8 m/s; the belt travel integrated, so the lab frame
  climbs or descends), and all 12 overground walks with turns (mc_047-058, 188 windows, 30 min).
- **Split by trial** (one subject, so there are no held-out subjects): `train` 213 clips (34 min); `test_trial` 56
  (9 min): trial 4 of each overground speed and the 1.2 m/s treadmill trials at 0, +5 and -10 deg; `test_terrain` 22
  (4 min): every +-15 deg trial. `data.HELD_OUT` names the held-out-motion split the gate and the evaluation use.
- AB01 has no stairs and 2.5 min of treadmill running (1.9-2.7 m/s).

**v1 (`data/v1`), set aside.** AB01 + Camargo 2021: 6,533 clips, 13.8 h; split by three held-out Camargo subjects and
the steepest/tallest Camargo terrain. Camargo is out for now: its IK welds the lumbar (the trunk never moves) and some
stair clips' heels never reach the ground (unclear whether theirs or our processing). The Camargo kinetics and EMG below
are built on v1 and stay the human references for every dataset (user 2026-11-15: the most nominal human data).

Both:
- **Dropped:** clips whose joint speed passes 20 rad/s anywhere (source IK spikes; 1 AB01 window, 10 Camargo clips).
- **Plans** (`plans.npz`): per frame and foot, down or not, the foothold of its current or next contact, time to
  touchdown and lift-off (capped at 1 s), and the whole-body centre of mass. Copied from the clips.
- **Reset lift:** per frame, the root raise that takes every contact sphere out of the terrain (signed distance to
  the profile, as SPRINT's contact sees it).

## Choices (defaults; each is one constant)

- **Control at 40 Hz**, 75 physics steps (3 kHz), implicit contact, SCONE motor delays: SPRINT's recommended
  imitation recipe without EMG. The clips are resampled from 30 Hz.
- **Style prior:** MimicKit's settings (2 DiT blocks, T = 50, diffusion steps 22/15/8, scale 6, AdamW 2e-4, batch
  512, 200k iterations, EMA 0.995). Window 13 frames (0.325 s, MimicKit's 10 at 30 Hz), scored every step. Features
  are this body's own (see `gp/smp.py`): root height above the terrain under it, not above a flat floor. Half of
  every batch is mirrored left to right.
- **Planner reward:** contact 0.15, foothold 0.15 (10 cm), centre-of-mass band 0.10 (+-10 cm forward, +-5 cm up and
  sideways, 5 cm tolerance beyond), pelvis orientation 0.10 (6 deg free, 6 deg tolerance beyond), style 0.50
  (MimicKit's 50/50), minus the recipe's effort (metabolic 0.02, summed cubed activation 0.02, compartment balance
  0.02); the same effort in the baseline. The actor sees the planned pelvis orientation as the recipe's root
  orientation error at each lookahead tap (user 2026-11-15: with the footholds it pins where the hips are, leaving
  each leg little more than its knee's swivel and the ankle; without it the plan left too many free DoF).
- **Metabolic cost at 0.6 MPa** (`gp.data.SPECIFIC_TENSION`): Rajagopal's F0 = PCSA x 60 N/cm^2, and SPRINT's default
  0.25 MPa makes every muscle 2.4x heavy, so heat (charged per kg) read 2.4x high. From the same states the cost falls
  to 0.53-0.60x (9.4 -> 5.2 W/kg at 10 % excitation everywhere). Forces are unchanged (tested).
- **Observation:** the actor reads its body, the plan at the recipe's lookahead taps (0-40 steps = 1 s) and the
  terrain only around where the plan puts each foot down next (`gp.tasks.PATCH`: 7 x 7 heights 5 cm apart, a 0.35 m
  square in its heading frame, shifted up to 5 cm per episode), as Heyrman et al. 2026 (arXiv 2610.29612) found
  patches at the coming footholds learn faster and step better on sparse ground than a whole map or patches at the
  current feet. The plan already says where the feet go, so no foothold predictor is needed. The critic also reads the
  2D strip scan (17 points from -0.4 to 1.2 m ahead x 3 at -0.2, 0, 0.2 m to the side) and the full reference;
  `PlanConfig(actor_strip=True)` gives the actor the strip too (the A/B test). The baseline keeps the strip scan.
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

### AB01 treadmill (`data/v2/kinetics_ab01.json`)

Camargo is all walking; AB01 ran 2.5 min on the instrumented split-belt treadmill (2.0-2.8 m/s, level and +-5 deg).
`ab01_id.py` solves every AB01 treadmill clip the same way (`camargo_id.carry_plates` / `net_moments`), with the belt's
travel added to the centres of pressure (the clips are `_og_`: belt-integrated) and forces scaled by model mass (75.3 kg) /
83.6 kg. **The plates run 12-19 ms behind the motion capture**: total vertical force against the whole-body centre of
mass's vertical acceleration correlates at r 0.99 at that lag and 0.96 at zero; uncorrected, the centre of pressure
sat ahead of the foot in proportion to speed (+3.2 cm of it at 2.4 m/s). Each trial's lag is estimated and removed.
After that the centre of pressure sits 1.3 cm from the foot's line sideways and 2-5 cm ahead of its midpoint under load
(walking, running), and the belts carry 1.02 body weights on average. 89 clips, 97-100 % of leg-frames valid.

95th percentile of per-clip peaks (GRF in BW, moments in N m/kg; the filter's numbers):

| | GRF | hip | knee | ankle |
|---|---|---|---|---|
| AB01 walking, level (12 clips, 0.4-1.6 m/s) | 1.27 | 0.83 | 1.04 | 1.66 |
| Camargo walking, level (1,765 clips) | 1.50 | 1.40 | 1.53 | 1.92 |
| AB01 running (15 clips, 2.0-2.8 m/s) | 2.36 | 2.46 | 2.21 | 2.35 |

AB01 walking falls inside Camargo's limits (the pipeline check); only running feeds the filter.

**Running limits rise with speed.** The dataset's running is faster than AB01's (PARC: median 3.4 m/s in train, 4.0 held
out, up to 4.5), and AB01's peaks rise with speed (per m/s: GRF +0.22 BW, as ~0.2 in Nilsson & Thorstensson 1989; ankle
+0.45, hip +0.87 N m/kg; knee flat, as Schache et al. 2011 found for knee stance work). Each running limit is the larger
of the pooled 95th percentile above and the 95 % prediction bound of a line through AB01's level clips' peaks against
speed, at the episode's target speed (the reference clip's or the plan's, so running faster than asked earns nothing);
the bound widens away from AB01's speeds, which is the headroom for extrapolating. Tabulated to the fastest running
clip (4.5 m/s); faster holds that limit.

| Target speed | GRF (BW) | loading (BW/s) | hip | knee | ankle |
|---|---|---|---|---|---|
| 2.0 m/s | 2.36 | 33.5 | 2.46 | 2.39 | 2.35 |
| 3.5 m/s | 2.51 | 46.2 | 3.87 | 2.56 | 2.83 |
| 4.5 m/s | 2.75 | 58.4 | 5.13 | 2.83 | 3.38 |

## Acceptance filter and G1 (`gp/accept.py`, `g1.py`)

The plan doc's six checks, each threshold the 5th or 95th percentile of the same measure on real clips: kinetics on
all 22 Camargo subjects per terrain class, style on the active dataset's clips of the episode's split and class. Camargo
is all walking, so running on level ground and ramps takes AB01's running limits instead (one entry, `kinetics_run`;
running stances 0.12-0.45 s for the G1 curves), PARC's obstacles (jumps, drops, platforms) leave the GRF and joint
checks unchecked, and G1 and the EMG report use walking episodes only. The checks: style (2 s windows), plan followed, ground reaction (peak force, loading rate),
joint loads, muscle effort (saturation), contact (sliding spheres, joint limits). A fallen episode is rejected first.
`evaluate.py` runs every episode through it. Two departures from the doc's drawing:
- joint loads are **net joint moments**, not knee and hip reaction forces: SPRINT's reaction pass refuses wrapped
  muscle paths, which Rajagopal v3 has, and real reaction forces would need muscle-force estimates;
- metabolic cost per metre is **reported, not gated**: there is no human metabolic data in the references.

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

## 2D courses (`gp/terrain_gen.py`: `sample_course2d`, `course_tile`)

Courses for the loop and for tests on new ground, on the heightfield, from (0, 0) heading +x: ramps, straight flights,
turning flights (a flight, a square landing where the walker turns 90 deg, a second flight the same way, as on PARC's
stairs), landing turns, gaps (1 m pits) and platforms; a level landing or a landing turn between every two features,
level at both ends, within +-2.5 m of the start height. Every part runs along a grid axis: block cells are axis-aligned
squares, so a riser or wall at any other angle is a sawtooth (winding stairs were dropped for that reason). The ground is
painted as boxes, each stretch a 2.4 m-wide rectangle and each turn a 2.4 m square landing; elsewhere it is flat at the
lowest landing, and pits reach 1 m past the causeway on each side. Ramps are triangulated, everything else block cells.
`difficulty` (0-1) sets how far each size may go from its gentle end towards its hard end (`CourseSpec`: rises
0.10-0.175 m, treads 0.37-0.25 m, grades 3-12.5 deg, gaps 0.15-0.8 m, platforms 0.1-0.6 m). A course whose centreline
comes back within 3.4 m of itself is drawn again. `export_studio.py --courses N --difficulty D [--seed S]` writes them for
SPRINT Studio (a standing pose at the start and carried along the course; no physics).

## Episodes without a clip (`gp/free.py`)

A generated course becomes a free clip the tasks run beside real ones: it starts from a real standing frame at the
course start (frame 0 only, no noise); its "reference" is that pose carried along the plan's centre of mass, so the
critic still sees where the plan expects the body; in `PlanTracking` it falls by losing the plan's centre of mass by
0.5 m or tilting 60 deg, and the critic (not the actor) reads a flag that the episode has no clip. `merge` puts free
clips beside a loaded dataset with one heightfield at the courses' cell (v2's PARC cell / 8 = 4.85 cm; the dataset's
heightfield clips cut to their path +- 1.5 m and refined). Until the planner exists, `stand_in_plan` makes the plans:
footholds on every tread, even steps of at most 0.6 m elsewhere, never on a riser's or pit's edge, feet 16 cm apart,
0.55 s per step with 0.12 s of double support. It is a stand-in, not a model of footstep choice; `gp.planner.plan_course`
makes the learned planner's plans in the same format (`course_clips(..., plans=...)`).

## The planner network (`gp/planner.py`)

A diffusion model of plan windows, like PARC's generator but over the plan instead of the pose. Plan frames at 20 Hz:
each foot's down flag and the xz of its current or next foothold, the centre of mass, the pelvis orientation (heading,
and its tilt from that heading as a rotation vector). A window is 0.4 s
given (the plan so far) and 1.2 s generated, in the heading frame of the last given frame. Conditions: the ground on a
3.6 m x 1.6 m grid ahead (0.1 m spacing, relative to the centre of mass's height) and the goal (where the centre of
mass is at the window's end). Network: the style prior's DiT blocks over 50 ground tokens and 32 plan tokens, 4 layers
x 256, x0 prediction, 50-step DDIM; half of each batch mirrored. Trained on every window of the train split's plans
(96k). Decoding: a contact's foothold is the mean of its frames' generated footholds, its height the ground's; times to
touchdown and lift-off from the contacts (`gp.data.plan_targets`). Rollout on a course: generate 1.2 s, keep 0.5 s,
re-anchor; the goal is the centreline `speed x 1.2 s` ahead, clamped before the end so the plan stops.
`check_planner.py` scores held-out windows (contacts, next touchdown's time and place, centre of mass) and course
rollouts (footholds on the course, in gaps, across edges; reaching the end) beside the stand-in plan.

## Known limits

- The muscle policy still trains on plans copied from real clips; the planner network is trained and checked on its
  own (above), and `train.py` does not run free episodes yet: the loop's driver comes next.
- Jumps, drops and platforms (PARC's obstacles) have no kinetic reference; running's is one subject on a treadmill.
- `gp/tasks.py` patches SPRINT's reference kinematics to run in chunks for the duration of setup (1.5M frames in one
  Data would take ~18 GB of GPU memory). Upstream fix: chunk in SPRINT.
