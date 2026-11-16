"""CPU smoke checks: terrain lookups match SPRINT, both tasks step with consistent observations and finite rewards,
the prior trains and scores, and PPO runs an iteration. Run: .venv/bin/python -m pytest tests -q
(needs the active dataset: python -m gp.data)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gp import data


@pytest.fixture(scope="module")
def D() -> dict:
    full = data.load("train")
    cls = np.array(full["terrain_class"])
    pick = [int(np.flatnonzero(cls == c)[0]) for c in sorted(set(cls))]          # one clip of each terrain class
    return data.take(full, pick)


def test_terrain_matches_sprint(D: dict) -> None:
    from sprint.terrain import height as sprint_height

    from gp.terrain import height
    rng = np.random.default_rng(0)
    for i in range(len(D["name"])):
        P = np.c_[D["profile_x"][i][: D["profile_n"][i]], D["profile_y"][i][: D["profile_n"][i]]]
        x = rng.uniform(P[0, 0] - 5, P[-1, 0] + 5, 200)
        x = x[np.min(np.abs(x[:, None] - P[None, :, 0]), 1) > 1e-4]          # not exactly on a riser
        ref = np.array([sprint_height(P, v) for v in x])
        ours = height(D["profile_x"], D["profile_y"], np.full(1, i), x[None])[0]
        tours = height(torch.as_tensor(D["profile_x"]), torch.as_tensor(D["profile_y"]), torch.full((1,), i),
                       torch.as_tensor(x[None]))[0].numpy()
        assert np.abs(ours - ref).max() < 1e-6 and np.abs(tours - ref).max() < 1e-6, D["name"][i]


def _prior(D: dict, tmp_path) -> str:
    from gp.smp import Prior, window_features
    from train_prior import batch, recording_weights, windows
    rec, ends = windows(D, "cpu")
    w = recording_weights(D, "cpu")                        # balanced: each recording's windows sum to 1
    assert len(w) == len(ends) and abs(float(w.sum()) - len({data.motion_group(str(c)) for c in D["name"]})) < 1e-3
    assert batch(rec, ends, 8, weights=w).shape == batch(rec, ends, 8).shape
    p = Prior(nt=len(D["tracked"]), dt=1.0 / D["hz"])
    p.set_normaliser(window_features(batch(rec, ends, 512), p.cfg["nt"], p.cfg["dt"]))
    opt = torch.optim.AdamW(p.net.parameters(), lr=2e-4)
    first = None
    for _ in range(30):
        loss = p.loss(p.features(batch(rec, ends, 64)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or float(loss.detach())
    assert float(loss.detach()) < first
    p.ema.module.load_state_dict(p.net.state_dict())
    path = str(tmp_path / "prior.pt")
    p.save(path)
    return path


def _env(task, worlds: int = 6):
    import sprint

    from gp.tasks import recipe_config
    model = data.load_model()
    c = task.config
    opts = sprint.StepOptions(dt_ctrl=1.0 / 40, n_sub=c.physics_steps, implicit_friction=True,
                              implicit_normal=c.implicit_normal, strict=False)
    assert recipe_config().physics_steps == 75
    return sprint.VecEnv(model, worlds, task=task, seed=0, step_options=opts, device="cpu", delays=c.delays)


def _roll(env, steps: int = 12) -> list:
    obs = env.reset()
    rewards = []
    for _ in range(steps):
        a = torch.full((env.num_worlds, env.action_dim), 0.05)
        obs, r, *_ = env.step(a)
        rewards.append(env.torch(r) if hasattr(env, "torch") else torch.as_tensor(r))
    o = env.torch(obs) if hasattr(env, "torch") else torch.as_tensor(obs)
    assert o.shape[1] == env.task.obs_dim and torch.isfinite(o).all()
    assert all(torch.isfinite(x).all() for x in rewards)
    return rewards


def test_conditional_prior(D: dict, tmp_path) -> None:
    from gp.smp import N_COND, Prior, conditions, window_features
    from train_prior import batch, frame_labels, windows
    rec, ends = windows(D, "cpu")
    lab = frame_labels(D, "cpu")
    assert lab.shape == (len(rec), N_COND) and torch.equal(lab[D["start"]], torch.as_tensor(conditions(D)))
    w, c = batch(rec, ends, 64, lab=lab)
    assert c.shape == (64, N_COND)
    p = Prior(nt=len(D["tracked"]), dt=1.0 / D["hz"], n_cond=N_COND)
    p.set_normaliser(window_features(batch(rec, ends, 512), p.cfg["nt"], p.cfg["dt"]))
    opt = torch.optim.AdamW(p.net.parameters(), lr=2e-4)
    first = None
    for _ in range(30):
        w, c = batch(rec, ends, 64, lab=lab)
        loss = p.loss(p.features(w), c)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or float(loss.detach())
    assert float(loss.detach()) < first
    p.ema.module.load_state_dict(p.net.state_dict())
    p.save(tmp_path / "cond.pt")
    q = Prior.load(tmp_path / "cond.pt")
    x = q.features(w)
    s_c, s_null = q.sds(x, c=c), q.sds(x)                   # labelled and null-label scores differ, both finite
    assert torch.isfinite(s_c).all() and torch.isfinite(s_null).all() and not torch.allclose(s_c, s_null)
    assert q.labels(D, [0]).shape == (1, N_COND) and Prior(nt=3, dt=0.025).labels(D, [0]) is None


def test_terrain_imitation_steps(D: dict) -> None:
    from gp.tasks import TerrainImitation
    env = _env(TerrainImitation(D))
    _roll(env)
    perm, sign, act = env.task.mirror_maps()
    assert len(perm) == env.task.obs_dim and sorted(act.tolist()) == list(range(env.action_dim))
    assert np.array_equal(env.task.csign, data.coord_mirror_signs(env.model))     # FK signs, not SPRINT's axis rule


def test_plan_tracking_steps(D: dict, tmp_path) -> None:
    from gp.tasks import PlanConfig, PlanTracking
    task = PlanTracking(D, plan=PlanConfig(prior=_prior(D, tmp_path)))
    env = _env(task)
    _roll(env)
    cols = task.actor_columns()
    offsets = np.cumsum([0] + [b[1] for b in task.blocks])[:-1]
    actor_blocks = {b[0].split("[")[0] for b, o in zip(task.blocks, offsets, strict=True) if o in set(cols.tolist())}
    assert "joint_angle_error" not in actor_blocks and "plan_feet" in actor_blocks
    assert "foothold_patch" in actor_blocks and "height_scan" not in actor_blocks         # the strip is the critic's
    assert "root_orientation_error" in actor_blocks and "root_position_error" not in actor_blocks   # pelvis orientation
    from gp.tasks import PATCH
    i = [b[0] for b in task.blocks].index("foothold_patch")
    perm = np.array(task.blocks[i][2])
    assert (perm[perm] == np.arange(len(perm))).all()                                     # the mirror is an involution
    task.patch_shift.zero_()                    # without the jitter, each patch's centre is the ground at the landing
    k, o, s = PATCH, int(offsets[i]), env.state
    patch = task.observation(s)[:, o:o + 2 * k * k].view(-1, 2, k, k)
    hold = task.next_footholds()
    want = task.ground(hold[..., 0], hold[..., 2]) - s.body_com_pos()[:, task.root, 1:2]
    assert torch.allclose(patch[:, :, k // 2, k // 2], want, atol=1e-5)
    assert "height_scan" not in PlanTracking(D, plan=PlanConfig(prior="x", actor_strip=True)).CRITIC_ONLY
    pc = task.plan
    assert abs(pc.w_contact + pc.w_foothold + pc.w_com + pc.w_orient + pc.w_style - 1.0) < 1e-9
    for k in ("r_contact", "r_foothold", "r_com", "r_orient", "r_style"):
        v = task.metrics[k]
        assert torch.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), k


def test_ppo_iteration(D: dict, tmp_path) -> None:
    from dataclasses import replace

    from sprint.recipes.ppo import PPO
    from sprint.recipes.study import default_ppo

    from gp.tasks import PlanConfig, PlanTracking
    env = _env(PlanTracking(D, plan=PlanConfig(prior=_prior(D, tmp_path))), worlds=8)
    agent = PPO(env, replace(default_ppo(), hidden=(64, 64), minibatches=2, rollout_steps=4), device="cpu")
    agent.train(1)
    assert agent.env_steps > 0


def test_sim_record_matches_data(D: dict) -> None:
    """The simulation's style record at a reset equals the data's, but for the reset lift on the root height."""
    from gp.data import record_layout
    from gp.tasks import TerrainImitation
    starts = [(i, 10) for i in range(len(D["name"]))]
    env = _env(TerrainImitation(D, starts=starts), worlds=len(starts))
    env.reset()
    task = env.task
    sim = task.record(env.state).cpu().numpy()
    f = (D["start"] + 10).astype(int)
    ref = D["record"][f].copy()
    ref[:, 1] += D["lift"][f]
    lay = record_layout(len(D["tracked"]))
    q_s, q_r = sim[:, lay["root_quat"]], ref[:, lay["root_quat"]]
    assert np.abs(np.abs((q_s * q_r).sum(1)) - 1).max() < 1e-5
    for k in ("root_pos", "joints", "ground"):
        assert np.abs(sim[:, lay[k]] - ref[:, lay[k]]).max() < 1e-4, k
    kp = np.abs(sim[:, lay["key_pos"]] - ref[:, lay["key_pos"]]).reshape(len(f), -1, 3)
    assert kp[..., [0, 2]].max() < 1e-4 and np.abs(kp[..., 1] - D["lift"][f][:, None]).max() < 1e-4


def test_specific_tension(D: dict) -> None:
    """Rajagopal's muscle masses at 0.6 MPa: from the same state, the metabolic cost drops (heat is charged per kg of
    muscle, work is not) and no force changes."""
    from gp.tasks import TerrainImitation
    starts = [(i, 20) for i in range(len(D["name"]))]
    out = {}
    for pa in (0.25e6, data.SPECIFIC_TENSION):
        env = _env(TerrainImitation(D, starts=starts), worlds=len(starts))
        assert np.allclose(env.model.host["mus_tension"], data.SPECIFIC_TENSION)
        env.model.set_specific_tension(pa)
        env.reset()
        env.step(torch.full((env.num_worlds, env.action_dim), 0.2))
        s = env.state
        out[pa] = (s.metabolic_power.sum(1).cpu().numpy(), s.muscle_force.cpu().numpy())
    (m25, f25), (m60, f60) = out[0.25e6], out[data.SPECIFIC_TENSION]
    assert np.array_equal(f25, f60)
    assert ((m60 < 0.9 * m25) & (m60 > m25 / 2.4)).all(), (m25, m60)


def test_accept_check() -> None:
    """The acceptance filter passes a clean episode and rejects a sliding foot, a missed contact and a fall."""
    from gp import accept
    T, S = 120, 18
    foot = np.zeros((T, 2, 3))
    clean = dict(sds=np.full(T, 0.2), plan_contact=np.ones((T, 2), bool), foothold_ref=foot.copy(), foot=foot.copy(),
                 com_out=np.zeros(T, bool), fv=np.full((T, 2), 0.5), moment=np.full((T, 6), 0.3),
                 act_sat=np.zeros((T, 90), bool), at_limit=np.zeros(T, bool), sphere=np.zeros((T, S, 3)),
                 touching=np.ones((T, S), bool), fell=False, metabolic_j_per_kg_m=3.0)
    thr = dict(grf_peak_bw=1.5, grf_loading_rate_bw_s=15.0, style_loss=1.0, band=None,
               joint_peak_nm_per_kg=dict(hip_flexion=1.4, knee_angle=1.5, ankle_angle=2.0))
    r = accept.check(clean, thr)
    assert r["accepted"] and not r["failed"] and not r["unchecked"], r
    slide = dict(clean, sphere=clean["sphere"] + np.linspace(0, 0.03, T)[:, None, None] * [1, 0, 0])
    assert accept.check(slide, thr)["failed"] == ["contact"]
    missed = dict(clean, fv=np.where(np.arange(T)[:, None] < 80, 0.0, 0.5) * [1, 1])
    assert "plan" in accept.check(missed, thr)["failed"]
    assert accept.check(dict(clean, fell=True), thr)["failed"] == ["fell"]
    assert accept.check(clean, dict(thr, style_loss=None))["unchecked"] == ["style"]
    acc = dict(kinetics=dict(level={k: v for k, v in thr.items() if k != "style_loss"}), style=dict(test_trial=dict(level=1.0)))
    assert accept.thresholds(acc, "test_trial", "level", "walk") == thr                # Camargo kinetics + our style
    assert accept.check(clean, accept.thresholds(acc, "test_trial", "level", "run"))["unchecked"] == ["grf", "joints"]
    assert accept.check(clean, accept.thresholds(acc, "test_trial", "obstacles", "run"))["unchecked"] == [
        "style", "grf", "joints"]
    # running takes AB01's running limits on level ground and ramps, still none on obstacles
    run_thr = dict(grf_peak_bw=0.4, grf_loading_rate_bw_s=40.0, band=None, stance_s=list(accept.RUN_STANCE_S),
                   joint_peak_nm_per_kg=dict(hip_flexion=2.5, knee_angle=2.2, ankle_angle=2.4))
    acc["kinetics_run"] = run_thr
    t = accept.thresholds(acc, "test_trial", "ramp_up", "run")
    assert t["grf_peak_bw"] == 0.4 and t["style_loss"] is None
    assert accept.check(clean, accept.thresholds(acc, "test_trial", "level", "run"))["failed"] == ["grf"]   # 0.5 > 0.4
    assert accept.thresholds(acc, "test_trial", "level", "walk")["grf_peak_bw"] == 1.5
    assert "grf_peak_bw" not in accept.thresholds(acc, "test_trial", "obstacles", "run")
    acc["kinetics_run"] = dict(run_thr, by_speed=dict(speed_mps=[2.0, 4.0], grf_peak_bw=[0.4, 0.8],
                                                      grf_loading_rate_bw_s=[40.0, 60.0],
                                                      joint_peak_nm_per_kg=dict(hip_flexion=[2.5, 4.5],
                                                                                knee_angle=[2.2, 2.4],
                                                                                ankle_angle=[2.4, 3.2])))
    fast = accept.thresholds(acc, "test_trial", "level", "run", speed=3.0)               # limits rise with speed
    assert abs(fast["grf_peak_bw"] - 0.6) < 1e-9 and abs(fast["joint_peak_nm_per_kg"]["hip_flexion"] - 3.5) < 1e-9
    assert accept.thresholds(acc, "test_trial", "level", "run", speed=9.0)["grf_peak_bw"] == 0.8   # held past the top
    assert accept.thresholds(acc, "test_trial", "level", "run")["grf_peak_bw"] == 0.4             # no speed: pooled
    assert accept.thresholds(acc, "test_trial", "level", "walk", speed=3.0)["grf_peak_bw"] == 1.5  # walking unaffected
    run = np.r_[np.zeros(5), np.ones(10), np.zeros(10), np.ones(10), np.zeros(5)].astype(bool)   # 0.25 s stances
    assert len(accept.stances(run, accept.RUN_STANCE_S)) == 2 and not accept.stances(run)


def test_mirror_records(D: dict) -> None:
    """A mirrored style record equals the style record of the mirrored pose (coordinates reflected with SPRINT's
    mirror signs, then forward kinematics); mirroring twice is the identity."""
    from sprint.recipes.imitation import mirrored
    from sprint.recipes.motion import forward_kinematics_np, model_layout

    from gp.data import coord_mirror_signs, record_layout, style_records
    from gp.smp import mirror_maps, mirror_records
    model = data.load_model()
    L, names = model_layout(model), list(model.coord_names)
    cs = coord_mirror_signs(model)
    legs = {n: float(cs[names.index(n)]) for n in names if n.endswith("_r")}
    assert all(v == 1.0 for v in legs.values()), legs          # paired leg coordinates mirror with +1 on Rajagopal v3
    maps = mirror_maps(model, D["tracked"])
    for i in range(len(D["name"])):
        rows = np.arange(int(D["start"][i]) + 5, int(D["start"][i]) + 45, 5)
        q = D["qpos"][rows].astype(np.float64)
        qm = q[:, [names.index(mirrored(n)) for n in names]] * cs
        xp, xq = forward_kinematics_np(model, qm.astype(np.float32))
        key = [list(model.body_names).index(b) for b in data.KEY_BODIES]
        # the mirrored pose stands on the mirrored terrain: the original ground at the reflected root (z -> -z)
        ground = data.ground_under(D, i, xp[:, 0, 0].astype(np.float64), -xp[:, 0, 2].astype(np.float64))
        ref = style_records(xp.astype(np.float64), xq.astype(np.float64), qm, L.tracked, key, ground)
        rec = torch.as_tensor(D["record"][rows])
        got = mirror_records(rec, maps).numpy()
        lay = record_layout(len(D["tracked"]))
        assert np.abs(np.abs((got[:, lay["root_quat"]] * ref[:, lay["root_quat"]]).sum(1)) - 1).max() < 1e-5
        for k in ("root_pos", "joints", "key_pos", "ground"):
            assert np.abs(got[:, lay[k]] - ref[:, lay[k]]).max() < 1e-4, (D["name"][i], k)
        assert torch.allclose(mirror_records(mirror_records(rec, maps), maps), rec)


def test_terrain_courses() -> None:
    """Random courses keep the rules (level start, end and landings; sizes in range; height band), SPRINT accepts
    their profiles, and SPRINT reads them level at both ends; a ramp straight into a ramp is refused."""
    from sprint.terrain import height, pack

    from gp.terrain_gen import TerrainSpec, check_course, course_profile, sample_course
    rng = np.random.default_rng(0)
    for spec in (TerrainSpec(), TerrainSpec.held_out()):
        courses = [sample_course(rng, spec) for _ in range(300)]
        for c in courses:
            check_course(c, spec)
        assert {s["kind"] for c in courses for s in c} == {"level", "ramp", "stairs"}
        assert {s["dir"] for c in courses for s in c if "dir" in s} == {1, -1}
        profs = [course_profile(c) for c in courses]
        pack(profs)
        for P in profs[:50]:
            assert height(P, P[0, 0] - 5.0) == 0.0 and abs(height(P, P[-1, 0] + 5.0) - P[-1, 1]) < 1e-9
    bad = [dict(kind="level", run=4.0), dict(kind="ramp", run=5.0, grade_deg=8.0, dir=1),
           dict(kind="ramp", run=5.0, grade_deg=8.0, dir=-1), dict(kind="level", run=4.0)]
    with pytest.raises(ValueError, match="straight into"):
        check_course(bad)


def test_emg_scores() -> None:
    """Stride profiles cut touchdown to touchdown; a profile set scored against itself reaches r 1 and full agreement;
    rectus femoris is left out of the summary."""
    from gp import emg
    T = 400
    down = (np.arange(T) % 44) < 26                                   # 1.1 s strides, 60 % stance
    x = np.stack([np.sin(np.arange(T) / 44 * 2 * np.pi + k) + 1.5 for k in range(3)], 1)
    pr = emg.stride_profiles(down, x)
    assert pr.shape == (8, 3, 101)
    s = emg.summarise(pr)
    ref = dict(channels=["soleus", "rectusfemoris", "gracilis"], **s)
    out = emg.compare(ref, s, unreliable=("rectusfemoris",))
    assert abs(out["soleus"]["r"] - 1) < 1e-9 and out["soleus"]["on_off_agreement"] == 1.0
    assert not out["rectusfemoris"]["scored"] and abs(out["mean_r"] - 1) < 1e-9


def test_heightfield_lookup_and_atlas() -> None:
    """gp.terrain.hf_height (numpy and torch) is SPRINT's heightfield surface, blocks included; tiles packed by atlas()
    read back their own heights through their offsets."""
    from sprint.terrain import hf_height as sprint_hf

    from gp.terrain import atlas, hf_height
    rng = np.random.default_rng(0)
    tiles = []
    for nz, nx in ((30, 50), (18, 70), (40, 25)):
        h = rng.normal(0, 0.05, (nz, nx))
        b = np.where(rng.random((nz - 1, nx - 1)) < 0.3, rng.normal(0.2, 0.1, (nz - 1, nx - 1)), np.nan)
        tiles.append(dict(heights=h, block=b, origin=(rng.uniform(-3, 3), rng.uniform(-3, 3))))
    cell = 0.05
    t0 = tiles[0]
    x = rng.uniform(t0["origin"][0] - 0.5, t0["origin"][0] + 50 * cell + 0.5, 500)
    z = rng.uniform(t0["origin"][1] - 0.5, t0["origin"][1] + 30 * cell + 0.5, 500)
    ref = np.array([sprint_hf(t0["heights"], cell, t0["origin"], a, c, block=t0["block"]) for a, c in zip(x, z, strict=True)])
    assert np.abs(hf_height(t0["heights"], t0["block"], cell, t0["origin"], x, z) - ref).max() < 1e-9
    tt = hf_height(torch.as_tensor(t0["heights"]), torch.as_tensor(t0["block"]), cell, t0["origin"],
                   torch.as_tensor(x), torch.as_tensor(z)).numpy()
    assert np.abs(tt - ref).max() < 1e-9
    A = atlas(tiles, cell)
    for k, t in enumerate(tiles):
        nz, nx = t["heights"].shape
        x = rng.uniform(t["origin"][0], t["origin"][0] + (nx - 1) * cell, 300)
        z = rng.uniform(t["origin"][1], t["origin"][1] + (nz - 1) * cell, 300)
        own = hf_height(t["heights"], t["block"], cell, t["origin"], x, z)
        via = hf_height(A["heights"], A["block"], cell, A["origin"], x + A["offsets"][k, 0], z + A["offsets"][k, 1])
        assert np.abs(own - via).max() < 1e-9, k


def test_clip_on_heightfield(D: dict) -> None:
    """A clip on stairs (a level clip given a stairs profile), once as that profile and once on a heightfield tile of
    it, has the same style record and height scan; a profile clip and a heightfield clip share one env and step; the
    scan's mirror swaps left and right."""
    from sprint.terrain import HFIELD
    from sprint.terrain import stairs as sprint_stairs

    from gp.data import heightfield_gap
    from gp.tasks import SCAN_LATERAL, TerrainImitation
    from gp.terrain import atlas, pad, tile_from_profile
    cls = list(D["terrain_class"])
    i = cls.index("level")
    two = data.take(D, [i, i])
    s, n = int(D["start"][i]), int(D["frames"][i])
    P = np.asarray(sprint_stairs(x0=float(D["qpos"][s + 10, 3]) + 0.4, n=6, run=0.30, rise=0.17,
                                 y0=float(D["profile_y"][i][0])), np.float64)
    X2, Y2 = pad([P, P])                                           # both clips stand on these stairs, as profiles
    two.update(profile_x=X2, profile_y=Y2, profile_n=np.array([len(P), len(P)]))
    xr = (D["qpos"][s:s + n, 3].min() - 3, D["qpos"][s:s + n, 3].max() + 3)
    zr = (D["qpos"][s:s + n, 5].min() - 3, D["qpos"][s:s + n, 5].max() + 3)
    A = atlas([tile_from_profile(P, xr, zr, 0.02)], 0.02)
    mixed = dict(two, terrain_kind=np.array([0, 1]), hf_offset=np.array([[0.0, 0.0], A["offsets"][0]]), hf=A)
    env_p = _env(TerrainImitation(two, starts=[(0, 10), (1, 10)]), worlds=2)
    env_h = _env(TerrainImitation(mixed, starts=[(0, 10), (1, 10)]), worlds=2)
    op, oh = env_p.torch(env_p.reset()), env_h.torch(env_h.reset())
    tid = env_h.data.terrain_id.numpy()[:2]
    assert tid[0] >= 0 and tid[1] == HFIELD
    rp, rh = env_p.task.record(env_p.state), env_h.task.record(env_h.state)
    assert torch.allclose(rp, rh, atol=1e-5)                                       # same pose, same ground under it
    scan = slice(env_h.task.obs_dim - len(env_h.task.scan_f), env_h.task.obs_dim)
    d = (op[:, scan] - oh[:, scan]).abs()
    assert float(d.median()) < 1e-4 and float((d < 0.01).float().mean()) > 0.95   # risers within one 2 cm cell
    for _ in range(5):
        env_h.step(torch.full((2, env_h.action_dim), 0.05))
    assert torch.isfinite(env_h.torch(env_h.obs)).all()
    perm, _, _ = env_h.task.mirror_maps()
    blk = perm[scan]
    assert np.array_equal(blk[blk - scan.start] , np.arange(scan.start, scan.stop)) and len(SCAN_LATERAL) == 3
    ctr = np.zeros((1, 1, 3))
    ctr[0, 0] = [xr[0] + 1.0, 0.05, zr[0] + 1.0]
    g = heightfield_gap(A, A["offsets"][0], ctr, np.array([0.03]))
    assert abs(float(g[0, 0]) - (0.02 - P[0, 1])) < 1e-6                                   # flat start of the tile


def test_parc_tile_matches_boxes() -> None:
    """A PARC clip's box grid as a heightfield tile reads the same heights as the seed library's nearest-cell lookup
    (canonical (x, y) = model (x, -z)), away from cell edges."""
    import json

    from gp.terrain import hf_height, tile_from_boxes
    r = next(r for r in json.loads((data.SEED / "manifest_parc.json").read_text())
             if r["exported"] and r["terrain_class"] == "obstacles")
    d = np.load(data.SEED / r["file"])
    hf, lo, dx = d["hf"].astype(np.float64), d["hf_min"].astype(np.float64), float(d["dx"])
    assert np.ptp(hf) > 0.3                                                   # a clip with real platforms
    t = tile_from_boxes(hf, lo, dx)
    rng = np.random.default_rng(0)
    ij = rng.uniform(0, 1, (2000, 2)) * (np.array(hf.shape) - 1)
    ij = ij[np.abs(ij - np.round(ij)).max(1) < 0.45]                          # not on a cell edge
    xy = lo + ij * dx
    ref = hf[tuple(np.round(ij).astype(int).T)]
    got = hf_height(t["heights"], t["block"], t["cell"], t["origin"], xy[:, 0], -xy[:, 1])
    assert np.abs(got - ref).max() < 1e-9


def test_courses_2d() -> None:
    """2D courses: level ends, a landing (or a landing turn) between features, 90 deg turns only, every size inside its
    difficulty's range, and the tile's surface under the centreline equal to the profile (but within a cell of a wall)."""
    from gp.terrain import hf_height
    from gp.terrain_gen import CourseSpec, course_tile, sample_course2d
    spec, rng = CourseSpec.full(), np.random.default_rng(0)
    assert {s["kind"] for k in range(20) for s in sample_course2d(np.random.default_rng(k), 1.0)} == {"level", *CourseSpec().kinds}
    seen = set()
    for d in (0.0, 0.5, 1.0):
        for _ in range(6):
            segs = sample_course2d(rng, d, spec)
            assert segs[0]["kind"] == segs[-1]["kind"] == "level"
            assert all({"level", "turn"} & {a["kind"], b["kind"]} for a, b in zip(segs[:-1], segs[1:], strict=True))
            assert all(abs(abs(s["turn"]) - np.pi / 2) < 1e-9 for s in segs if s["turn"])     # grid-aligned turns only
            for s in segs:
                seen.add(s["kind"])
                if "rise" in s:
                    assert spec.rise_m[0] - 1e-9 <= s["rise"] <= spec.rise_m[0] + d * (spec.rise_m[1] - spec.rise_m[0]) + 1e-9
                if s["kind"] == "gap":
                    assert s["length"] <= spec.gap_m[0] + d * (spec.gap_m[1] - spec.gap_m[0]) + 1e-9
            T = course_tile(segs)
            p = T["path"]
            err = np.abs(hf_height(T["heights"], T["block"], T["cell"], T["origin"], p["x"], p["z"]) - p["h"])
            assert np.mean(err < 0.005) > 0.97 and err.max() <= spec.pit_m + 0.2     # ramps interpolate to < 2 mm
    assert seen == {"level", "turn", "ramp", "stairs", "gap", "platform"}


def test_free_episodes(tmp_path) -> None:
    """Generated courses run beside real clips (gp.free): they start at frame 0 standing on the course, the critic
    sees the free flag and the actor does not, rewards are finite, and leaving the plan's centre of mass is a fall."""
    from gp.free import FREE_FALL_M, REFINE, course_clips, merge, stand_in_plan
    from gp.tasks import PlanConfig, PlanTracking
    from gp.terrain_gen import course_tile, sample_course2d
    full = data.load("train")
    cls = np.array(full["terrain_class"])
    stand = next(i for i in range(len(cls)) if full["mode"][i] == "stand" and cls[i] == "level")
    hfc = int(np.flatnonzero(np.asarray(full["terrain_kind"]) == 1)[0])
    Ds = data.take(full, [stand, hfc])
    tiles = [course_tile(sample_course2d(np.random.default_rng(k), 0.8), cell=full["hf"]["cell"] / REFINE) for k in (1, 2)]
    for T in tiles:                                        # the stand-in plan walks: no flight, feet alternate
        P = stand_in_plan(T, 0.98)
        assert P["plan_contact"].any(1).all() and 0.15 < P["plan_contact"].all(1).mean() < 0.4
    M = merge(Ds, course_clips(tiles, Ds, 0))
    task = PlanTracking(M, plan=PlanConfig(prior=_prior(Ds, tmp_path)), starts=[(2, 0), (3, 0), (1, 10)])
    env = _env(task, worlds=3)
    env.reset()
    assert (task.frame[:2] == task.clip_start[task.clip[:2]]).all()
    tx, tz = M["coord_names"].index("pelvis_tx"), M["coord_names"].index("pelvis_tz")
    q = env.state.qpos
    assert torch.allclose(task.ground(q[:, tx:tx + 1], q[:, tz:tz + 1])[:2], torch.zeros(2, 1), atol=1e-6)
    offsets = np.cumsum([0] + [b[1] for b in task.blocks])[:-1]
    flag = int(offsets[[b[0] for b in task.blocks].index("free_episode")])
    assert flag not in set(task.actor_columns().tolist())
    env.state.qpos[0, tx] += FREE_FALL_M + 0.5               # world 0 jumps away from its plan; world 1 stays
    for _ in range(task.config.fall_steps):
        obs, r, *_ = env.step(torch.full((3, env.action_dim), 0.05))
        assert torch.isfinite(torch.as_tensor(r)).all()
    assert bool(task._fell[0]) and not bool(task._fell[1])
    assert torch.as_tensor(obs)[:, flag].tolist() == [1.0, 1.0, 0.0]


def test_planner(D: dict) -> None:
    """The planner (gp.planner): windows round-trip to the world plan, mirror twice is the identity, the loss trains,
    a real clip's 20 Hz plan decodes back to its 40 Hz plan, and a rollout on a course makes a free clip."""
    from gp.free import REFINE, course_clips, stand_in_plan
    from gp.planner import GOAL, HIST, STRIDE, Planner, Windows, features, mirror, plan_course, to_plan, world
    from gp.terrain_gen import course_tile, sample_course2d
    W = Windows(D)
    a = W.anchors[torch.randperm(len(W.anchors), generator=torch.Generator().manual_seed(0))[:64]]
    x, p, g = W(a)
    assert torch.equal(g, x[:, -1, GOAL]) and x[:, HIST - 1, 6:9].abs().max() == 0
    k = a[:, None] + W.k
    d, h, c, ps, qu = world(x, W.com[a], W.psi[a])
    assert torch.equal(d, W.down[k]) and (h - W.hold[k]).abs().max() < 1e-5 and (c - W.com[k]).abs().max() < 1e-5
    assert (1 - (qu * W.quat[k]).sum(-1).abs()).max() < 1e-5                       # pelvis orientation round-trips
    assert (features(d, h, c, ps, qu, W.com[a], W.psi[a]) - x).abs().max() < 1e-4
    xm, pm = mirror(*mirror(x, p))
    assert torch.equal(xm, x) and torch.equal(pm, p)
    torch.manual_seed(0)
    P = Planner(T=5, layers=1, heads=2, head_dim=16)
    P.set_normaliser(*W.batch(2000))
    opt = torch.optim.AdamW(P.net.parameters(), lr=1e-3)
    first = None
    for _ in range(30):
        loss = P.loss(x, p, g)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < first
    c0 = 0                                                  # decode a real clip's plan from every other frame
    s, n = D["start"][c0], D["frames"][c0]
    f = np.arange(s, s + n, STRIDE)
    Pl = to_plan(D["plan_contact"][f], D["foothold"][f][..., [0, 2]], D["com"][f], W.psi[f].numpy(), W.quat[f].numpy(),
                 lambda x_, z_: data.ground_under(D, c0, x_, z_))
    q40 = W.quat[s:s + len(Pl["pelvis_quat"])].numpy()
    assert (1 - np.abs((Pl["pelvis_quat"] * q40).sum(-1))).max() < 1e-3           # slerped between plan frames
    m = len(Pl["plan_contact"])
    assert (Pl["plan_contact"] == D["plan_contact"][s:s + m]).mean() > 0.97
    assert np.abs(Pl["foothold"][..., [0, 2]] - D["foothold"][s:s + m][..., [0, 2]]).max() < 1e-4
    full = data.load("train")
    stand = next(i for i in range(len(full["name"])) if full["mode"][i] == "stand" and full["terrain_class"][i] == "level")
    T = course_tile(sample_course2d(np.random.default_rng(3), 0.5), cell=full["hf"]["cell"] / REFINE)
    plan = plan_course(P, [T], full, stand)[0]
    assert set(plan) == set(stand_in_plan(T, 0.98)) - {"steps"} | {"pelvis_quat"} and len(plan["com"]) == len(plan["plan_contact"])
    F = course_clips([T], data.take(full, [stand]), 0, plans=[plan])
    assert F["frames"][0] == len(plan["com"]) and np.isfinite(F["qpos"]).all()
    # the free clip's pelvis follows the planned orientation
    rl = data.record_layout
    rq = F["record"][:, rl(len(F["tracked"]))["root_quat"]]
    assert (1 - np.abs((rq * plan["pelvis_quat"]).sum(-1))).max() < 1e-4


def test_diverged_world(D: dict, tmp_path) -> None:
    """A world whose state goes non-finite (SPRINT resets it after the task's reward and observation) must not crash
    the task: the heightfield lookup once cast NaN to an index and took the GPU down 5 min into the first runs."""
    from gp.tasks import PlanConfig, PlanTracking, TerrainImitation
    from gp.terrain import hf_height
    H, B = np.zeros((3, 3)), np.full((2, 2), np.nan)
    for x in (np.array([np.nan, np.inf, 0.5]), torch.tensor([np.nan, np.inf, 0.5])):
        t = (lambda a: a) if isinstance(x, np.ndarray) else torch.as_tensor  # noqa: E731
        h = hf_height(t(H), t(B), 1.0, (0.0, 0.0), x, x * 0)
        assert np.isfinite(np.asarray(h)).all()
    full = data.load("train")
    hfc = int(np.flatnonzero(np.asarray(full["terrain_kind"]) == 1)[0])
    Ds = data.take(full, [hfc, int(np.flatnonzero(np.asarray(full["terrain_kind"]) == 0)[0])])
    for task in (TerrainImitation(Ds, starts=[(0, 10), (1, 10)]),
                 PlanTracking(Ds, plan=PlanConfig(prior=_prior(Ds, tmp_path)), starts=[(0, 10), (1, 10)])):
        env = _env(task, worlds=2)
        env.reset()
        env.state.qpos[0] = float("nan")
        for _ in range(3):
            obs, r, *_ = env.step(torch.full((2, env.action_dim), 0.05))
        assert torch.isfinite(torch.as_tensor(r)).all() and torch.isfinite(torch.as_tensor(obs)[1]).all()
        if isinstance(task, PlanTracking):                   # the style normaliser stays clean (the NaN-forever bug)
            assert torch.isfinite(task.style.mean_abs).all() and torch.isfinite(task.metrics["r_style"]).all()
