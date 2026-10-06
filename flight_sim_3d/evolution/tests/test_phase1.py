"""Phase-1 shared profiles (genome/ALIGNMENT.md): ramp shape, climb-rate feed-forward, comfort, log0 genes,
configurable roll gains, loading from Flight Dynamics' jsbsim_root, and agreement with genome/'s simulator."""
import hashlib
import json
import math
import os

import numpy as np
import pytest

from evolution import batch, cache, genome, sim, trajectory, validate_traj

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TEAM = os.path.dirname(PKG)
FD_ROOT = os.path.join(TEAM, "flight-dynamics", "jsbsim_root")
GENOME = os.path.join(TEAM, "genome")
G = 32.174
need_fd = pytest.mark.skipif(not os.path.isdir(FD_ROOT), reason="flight-dynamics/jsbsim_root not present")


def phase1_cfg():
    with open(os.path.join(PKG, "configs", "phase1.json")) as f:
        return batch.resolve_config(json.load(f), "phase1")


def phase1_profile(name):
    return sim.Profile.from_dict(next(a for a in phase1_cfg()["aircraft"] if a["name"] == name)["resolved_profile"])


def sample(sc, t_end, dt=sim.DT):
    t = np.arange(0.0, t_end + 1e-9, dt)
    ref = np.array([sc.target(x)[0] for x in t])
    rate = np.array([sc.target_rate(x) for x in t])
    tstep = np.array([sc.target(x)[1] for x in t])
    cmd = np.array([sc.target_cmd(x)[0] for x in t])
    return t, ref, rate, tstep, cmd


# ------------------------------------------------------------------ ramp shape
def test_ramp_shape_trapezoid_600fpm_01g():
    sc = sim.Scenario(seed=0, steps=[(0.0, 4000.0), (5.0, 4200.0), (50.0, 4000.0)], ramp_fpm=600.0, ramp_accel_g=0.1)
    t, ref, rate, tstep, cmd = sample(sc, 90.0)
    a, vmax = 0.1 * G, 10.0
    # limits: climb rate <= 600 fpm, |accel| <= 0.1 g; reference and rate are continuous (no jump at the command)
    assert np.max(np.abs(rate)) <= vmax + 1e-12
    assert np.max(np.abs(np.diff(rate))) <= a * sim.DT + 1e-9
    assert np.max(np.abs(np.diff(ref))) <= vmax * sim.DT + 1e-9
    # numerical slope of the reference == reported rate (midpoint rule: exact inside a phase, <= a*DT/2 across a corner)
    slope_err = np.abs(np.diff(ref) / sim.DT - 0.5 * (rate[1:] + rate[:-1]))
    assert np.max(slope_err) <= a * sim.DT / 2 and np.median(slope_err) < 1e-9
    # it actually reaches 600 fpm and 0.1 g (trapezoid), and lands exactly on the command
    assert np.max(rate) == pytest.approx(vmax) and np.min(rate) == pytest.approx(-vmax)
    assert np.max(np.diff(rate)) / sim.DT == pytest.approx(a, rel=1e-6)
    t_move = 200.0 / vmax + vmax / a  # trapezoid duration: D/vmax + vmax/a = 23.108 s
    arrive_up = t[np.argmax(ref >= 4200.0 - 1e-9)]
    assert arrive_up == pytest.approx(5.0 + t_move, abs=sim.DT)
    assert sc.target(5.0 + t_move + 1e-6) == (4200.0, 5.0) and sc.target_rate(5.0 + t_move + 1e-6) == 0.0
    # never overshoots, starts at rest at h0 until the command, symmetric descent
    assert ref.max() == 4200.0 and ref.min() == 4000.0
    assert np.all(ref[t < 5.0] == 4000.0) and np.all(rate[t < 5.0] == 0.0)
    up = np.array([sc.target(5.0 + x)[0] - 4000.0 for x in np.linspace(0, t_move, 50)])
    down = np.array([4200.0 - sc.target(50.0 + x)[0] for x in np.linspace(0, t_move, 50)])
    assert np.max(np.abs(up - down)) < 1e-9
    # the ITAE clock starts at the command change, not at reference arrival; at rest ref == command
    assert set(tstep[(t >= 5) & (t < 50)]) == {5.0} and set(tstep[t >= 50]) == {50.0}
    rest = (rate == 0.0) & ~np.isin(t, [5.0, 50.0])  # at the command instant itself the reference hasn't moved yet
    assert np.all(ref[rest] == cmd[rest])
    assert sc.target(5.0) == (4000.0, 5.0) and sc.target_cmd(5.0) == (4200.0, 5.0)


def test_ramp_triangle_and_midmove_command_change():
    a, vmax = 0.1 * G, 10.0
    tri = sim.Scenario(seed=0, steps=[(0.0, 0.0), (1.0, 20.0)], ramp_fpm=600.0, ramp_accel_g=0.1)
    t, ref, rate, _, _ = sample(tri, 12.0)
    peak = math.sqrt(a * 20.0)  # short move: triangle, peak sqrt(a*D) < 600 fpm (sampled peak within a*DT)
    assert peak - a * sim.DT <= np.max(rate) <= peak < vmax
    assert ref[-1] == 20.0 and rate[-1] == 0.0
    # command reversed while climbing at full rate: brakes at 0.1 g (no rate jump), then comes back to 0
    mid = sim.Scenario(seed=0, steps=[(0.0, 0.0), (5.0, 200.0), (12.0, 0.0)], ramp_fpm=600.0, ramp_accel_g=0.1)
    t, ref, rate, tstep, _ = sample(mid, 60.0)
    assert np.max(np.abs(np.diff(rate))) <= a * sim.DT + 1e-9 and np.max(np.abs(rate)) <= vmax + 1e-12
    assert 0 < ref.max() < 200.0 and ref.min() >= -1e-9 and ref[-1] == 0.0 and rate[-1] == 0.0
    assert set(tstep[t >= 12.0]) == {12.0}
    # linear ramp (no accel limit): constant 600 fpm, sharp corners
    lin = sim.Scenario(seed=0, steps=[(0.0, 0.0), (5.0, 200.0)], ramp_fpm=600.0)
    assert lin.target(15.0)[0] == pytest.approx(100.0) and lin.target_rate(15.0) == 10.0 and lin.target(26.0)[0] == 200.0


def test_ramp_off_is_the_original_instant_step():
    sc = sim.make_scenarios(3, 1, sim.Profile())[1]
    assert sc.ramp_fpm is None and sc.target(5.0) == (4200.0, 5.0) and sc.target_rate(20.0) == 0.0
    with pytest.raises(ValueError):
        sim.Profile.from_dict({"ramp_accel_g": 0.1})


# ------------------------------------------------------------------ feed-forward + comfort wiring
def test_feedforward_wiring():
    # outer loop reduced to the D term (kp at its floor, ki off): FF on => commanded pitch follows the
    # reference climb rate (hdot ~ 10 ft/s mid-ramp); FF off => the D term damps hdot to ~0, the aircraft doesn't climb
    g = {"kp_alt": 0.002, "ki_alt": 0.0, "kd_alt": 0.938, "kp_pitch": 0.0235, "ki_pitch": 0.0005, "kd_pitch": 0.0135}
    hdot = {}
    for ff in (True, False):
        P = sim.Profile(ramp_fpm=600, ramp_accel_g=0.1, alt_ref_ff=ff, duration_s=30, steps_rel_ft=[(0, 0), (5, 200)])
        r = sim.simulate(g, sim.make_scenarios(1, 1, P)[0], P, record=True)
        assert r["status"] == "ok"
        tr = r["trajectory"]
        d = np.array(tr["data"])
        m = (d[:, 0] > 12) & (d[:, 0] < 24)
        hdot[ff] = d[m, tr["channels"].index("vz")].mean() / sim.FT
    assert 8.0 < hdot[True] < 11.0 and abs(hdot[False]) < 2.0
    # without a ramp h_ref_dot == 0, so the FF flag cannot change anything (bit-identical)
    r = [sim.simulate(g, sim.make_scenarios(2, 1, P)[1], P)["cost"]
         for P in (sim.Profile(duration_s=20, alt_ref_ff=True), sim.Profile(duration_s=20))]
    assert r[0] == r[1]


def test_comfort_term_and_roll_gains_wiring():
    g = genome.decode([0.6] * genome.N_GENES)
    base = dict(duration_s=30.0, ramp_fpm=600.0, ramp_accel_g=0.1, alt_ref_ff=True)
    off = sim.simulate(g, sim.make_scenarios(2, 1, sim.Profile(**base))[1], sim.Profile(**base))
    P = sim.Profile(w_comfort=0.05, **base)
    on = sim.simulate(g, sim.make_scenarios(2, 1, P)[1], P)
    assert "comfort" not in off and off["cost"] == off["track"] + 2.0 * off["effort"]
    assert on["track"] == off["track"] and on["effort"] == off["effort"]   # comfort only adds a term
    assert on["cost"] == pytest.approx(on["track"] + 2.0 * on["effort"] + 0.05 * on["comfort"], rel=1e-15)
    assert on["comfort"] == pytest.approx(sum(sim.COMFORT_WEIGHTS[k] * v for k, v in on["comfort_terms"].items()))
    # comfort_terms against a hand calculation (constant 1.1 g, 7 deg above trim, 4 deg/s)
    n = 100
    terms = sim.comfort_terms(np.full(n, 1.1), np.full(n, 9.0), np.full(n, -4.0), 2.0)
    assert terms["rms_dn"] == pytest.approx(1.0) and terms["max_dn"] == pytest.approx(1 / 3)
    assert terms["jerk"] == pytest.approx(0.0) and terms["pitch_excess"] == pytest.approx(2.0)
    assert terms["pitch_rate_excess"] == pytest.approx(1.0)
    # roll gains are live (turbulent scenario excites roll) and the defaults are the prototype's
    assert (sim.Profile().roll_kp, sim.Profile().roll_kd) == (0.05, 0.02)
    Pr = sim.Profile(roll_kp=0.5, roll_kd=0.2, **base)
    assert sim.simulate(g, sim.make_scenarios(2, 1, Pr)[1], Pr)["cost"] != off["cost"]


# ------------------------------------------------------------------ log0
def test_log0_decode_encode():
    gn = genome.Gene("ki", 1e-6, 0.05, kind="log0")
    assert gn.decode(0.0) == 0.0 and gn.decode(0.05) == 0.0 and gn.decode(-1) == 0.0
    assert gn.decode(0.0500001) == pytest.approx(1e-6, rel=1e-4)
    assert gn.decode(1.0) == pytest.approx(0.05) and gn.decode(2.0) == pytest.approx(0.05)
    assert gn.decode(0.525) == pytest.approx(math.sqrt(1e-6 * 0.05))   # middle of the log part = geometric mean
    for v in (0.06, 0.3, 0.77, 1.0):
        assert gn.encode(gn.decode(v)) == pytest.approx(v)
    assert gn.encode(0.0) == 0.0 and gn.encode(1e-9) == pytest.approx(0.05)  # below min clamps to the band edge
    # make_schema: kinds per gene, order and other genes unchanged, bad input rejected
    sch = genome.make_schema({"ki_alt": [1e-6, 0.05]}, {"ki_alt": "log0", "ki_pitch": "log0"})
    assert [x.name for x in sch] == genome.GENE_NAMES
    assert [x.kind for x in sch] == ["log", "log0", "log", "log", "log0", "log"]
    assert genome.decode([0.5, 0.01, 0.5, 0.5, 0.0, 0.5], sch)["ki_alt"] == 0.0
    assert genome.make_schema() == genome.SCHEMA
    with pytest.raises(ValueError):
        genome.make_schema(kinds={"ki_alt": "exp"})
    with pytest.raises(ValueError):
        genome.make_schema(kinds={"nope": "log0"})
    # zeroed integrator is reported as such, not as "min"
    assert batch._genes_at_bound(sch, [0.5, 0.01, 0.5, 0.99, 0.06, 0.5]) == {"ki_alt": "zero", "kp_pitch": "max", "ki_pitch": "min"}


def test_phase1_config_profiles():
    cfg = phase1_cfg()
    assert [a["name"] for a in cfg["aircraft"]] == ["c172x", "T38", "737", "f16"]
    assert cfg["scenario_seed"] == 1 and cfg["ga"]["pop_size"] == 32 and cfg["ga"]["generations"] == 20 and cfg["scenarios"] == 3
    for a in cfg["aircraft"]:
        p = a["resolved_profile"]
        assert (p["ramp_fpm"], p["ramp_accel_g"], p["alt_ref_ff"], p["w_comfort"], p["w_effort"]) == (600.0, 0.1, True, 0.05, 2.0)
        assert p["steps_rel_ft"] == [[0.0, 0.0], [5.0, 200.0], [50.0, 0.0]]          # 200 ft for EVERY aircraft
        assert (p["alt_err_scale_ft"], p["max_alt_err_ft"]) == (100.0, 1000.0)
        assert p["gene_kinds"] == {"ki_alt": "log0", "ki_pitch": "log0"} and p["gear_up"] and p["throttle_all_engines"]
        assert p["aircraft_root"] == FD_ROOT
    # the baseline profile and the existing configs are untouched
    assert sim.Profile.from_dict(cfg["profiles"]["baseline"]) == sim.Profile()


@pytest.mark.skipif(not os.path.exists(os.path.join(GENOME, "exports", "evolution_phase1_profiles.json")),
                    reason="genome/ export not present")
def test_phase1_config_matches_genome_export():
    with open(os.path.join(GENOME, "exports", "evolution_phase1_profiles.json")) as f:
        full = json.load(f)
    exp = full["profiles"]
    # Genome flags keys/genes that need new evolution code (heading hold, since 04:50 PT) under _needs_code and says
    # to drop them to fly the Phase-1 task; compare everything else
    needs = full.get("_needs_code", {})
    # heading hold is opt-in (configs/phase1_hdg.json, tests/test_heading.py); phase1.json is the flag-off task
    skip_keys = set(needs.get("profile_keys", [])) | {"heading_hold", "bank_limit_deg", "hdg_i_limit_deg", "w_heading", "hdg_rms_ref_deg"}
    skip_genes = set(needs.get("genes", [])) | {"kp_hdg", "ki_hdg"}
    cfg = phase1_cfg()
    for name, key in (("c172x", "phase1_c172x"), ("T38", "phase1_T38"), ("737", "phase1_737")):
        p = next(a for a in cfg["aircraft"] if a["name"] == name)["resolved_profile"]
        for k, v in exp[key].items():
            if k.startswith("_") or k in skip_keys:
                continue
            if isinstance(v, dict) and k in ("gain_bounds", "gene_kinds"):
                v = {g: x for g, x in v.items() if g not in skip_genes}
            if k == "aircraft_root":   # the export is repo-relative; resolve_config makes it absolute
                v = sim.abs_root(v)
            if k in ("thr_kp", "thr_ki"):   # export rounds to 6 digits; we use the full-precision adapter values
                assert p[k] == pytest.approx(v, rel=2e-5), (name, k)
            else:
                assert json.loads(json.dumps(p[k])) == v, (name, k)


# ------------------------------------------------------------------ Flight Dynamics jsbsim_root
@need_fd
def test_load_via_fd_jsbsim_root():
    def tree_sha(root):
        h = hashlib.sha256()
        for dp, dns, fs in os.walk(root):
            dns.sort()
            for f in sorted(fs):
                p = os.path.join(dp, f)
                h.update(p.encode() + open(p, "rb").read())
        return h.hexdigest()

    fd_before = tree_sha(os.path.join(FD_ROOT, "aircraft"))
    present = {m: os.path.exists(os.path.join(FD_ROOT, "aircraft", m, m + ".xml")) for m in ("c172x", "T38", "737", "f16")}
    assert all(present.values()), present  # f16 added by FD 2026-10-06 (INTERFACE.md s5)
    for name in ("c172x", "T38", "737"):
        P = phase1_profile(name)
        sc = sim.make_scenarios(1, 1, P)[0]
        pf = sim.preflight(P.to_dict(), sc.to_dict())
        assert pf["ok"], (name, pf)
        # the FD copies (zero-weight flex point masses) fly bit-identically to the stock model
        Pd = sim.Profile.from_dict({**P.to_dict(), "duration_s": 15.0})
        Ps = sim.Profile.from_dict({**Pd.to_dict(), "aircraft_root": None})
        g = genome.decode([0.55] * genome.N_GENES, genome.make_schema(P.gain_bounds, P.gene_kinds))
        assert sim.simulate(g, sc, Pd)["cost"] == sim.simulate(g, sc, Ps)["cost"]
        # cache key: different root path and content hash => different key
        assert sim.model_files_sha(name, FD_ROOT) != sim.model_files_sha(name)
    # whatever FD ships, the model we load never declares socket I/O: if FD's file has socket elements (their 737 did
    # until 04:48 PT) they are stripped in a /tmp copy, never in FD's folder; otherwise FD's folder is used directly
    import re
    for m in ("c172x", "T38", "737", "f16"):
        raw = re.sub(r"<!--.*?-->", "", open(os.path.join(FD_ROOT, "aircraft", m, m + ".xml")).read(), flags=re.S)
        has_sock = bool(re.search(r"<(input|output)\b[^>]*\bport\s*=", raw))
        r = sim._aircraft_root(m, FD_ROOT)
        if has_sock:
            assert r and r.startswith(sim._SAN_ROOT) and not r.startswith(FD_ROOT)
            loaded = open(os.path.join(r, m, m + ".xml")).read()
        else:
            assert r is None
            loaded = raw
        assert not re.search(r"<(input|output)\b[^>]*\bport\s*=", re.sub(r"<!--.*?-->", "", loaded, flags=re.S)), m
    assert tree_sha(os.path.join(FD_ROOT, "aircraft")) == fd_before


def test_model_missing_from_root_is_load_failed(tmp_path):
    """A model a root doesn't have is a clean load_failed, not a crash or a silent fallback to the stock package."""
    (tmp_path / "aircraft").mkdir()
    P = sim.Profile(aircraft="f16", aircraft_root=str(tmp_path), h0_ft=10000, speed_kts=350)
    r = sim.simulate(genome.decode([0.5] * 6), sim.make_scenarios(1, 1, P)[0], P)
    assert r["status"] == "load_failed" and "not found" in r["error"] and r["cost"] == 2 * P.fail_base


@need_fd
@pytest.mark.skipif(not os.path.exists(os.path.join(FD_ROOT, "aircraft", "f16", "f16.xml")), reason="FD f16 not present")
def test_fd_f16_loads_trims_like_fd_and_has_no_sockets():
    """FD INTERFACE.md s5: prepared f16, 350 KCAS / 10000 ft, gear up, mode 1 -> alpha 1.045, throttle 0.2836 (dry)."""
    P = phase1_profile("f16")
    assert P.aircraft_root == FD_ROOT and P.throttle_max == 0.5 and P.gear_up
    # no network sockets (only a commented-out CSV output): loaded straight from FD's folder, no stripped copy
    raw = open(os.path.join(FD_ROOT, "aircraft", "f16", "f16.xml")).read()
    assert "port=" not in raw and sim._aircraft_root("f16", FD_ROOT) is None
    sc = sim.make_scenarios(1, 1, P)[0]
    fdm, info = sim.trim(P, sc)
    assert info["alpha_deg"] == pytest.approx(1.045, abs=5e-4) and info["theta_trim_deg"] == pytest.approx(1.045, abs=5e-4)
    assert info["throttle_trim"] == pytest.approx(0.2836, abs=5e-5) and info["throttle_trim"] < P.throttle_max
    assert fdm["fcs/throttle-pos-norm"] == pytest.approx(0.567, abs=5e-4)          # 57 % of MIL, dry
    assert info["pitch_trim_cmd"] == pytest.approx(-0.0601, abs=5e-5) and info["elev_trim"] == 0.0
    assert fdm["fcs/elevator-pos-deg"] == pytest.approx(-1.111, abs=5e-4)
    assert info["n_engines"] == 1 and info["gear_cmd"] == 0.0 and fdm["gear/gear-pos-norm"] == 0.0
    # point masses: 0 = zero-weight placeholder, 1 = Pilot (230 lb); total weight unchanged at 20 630 lb.
    # (Nothing in evolution/ addresses point masses by index; this just pins FD's layout.)
    assert fdm["inertia/pointmass-weight-lbs[0]"] == 0.0 and fdm["inertia/pointmass-weight-lbs[1]"] == 230.0
    assert fdm["inertia/weight-lbs"] == pytest.approx(20630.0)
    # 20 s hands-off: within 0.03 deg pitch and 4 ft (FD: +0.028 deg, +3.9 ft)
    th0, h0 = fdm["attitude/theta-deg"], fdm["position/h-sl-ft"]
    for _ in range(20 * 120):
        fdm.run()
    assert abs(fdm["attitude/theta-deg"] - th0) < 0.03 and abs(fdm["position/h-sl-ft"] - h0) < 4.0
    # FD's copy flies bit-identically to the stock f16, repeat runs are bit-identical, throttle never above MIL
    g = genome.decode([0.55] * genome.N_GENES, genome.make_schema(P.gain_bounds, P.gene_kinds))
    Pd = sim.Profile.from_dict({**P.to_dict(), "duration_s": 20.0})
    Ps = sim.Profile.from_dict({**Pd.to_dict(), "aircraft_root": None})
    sc2 = sim.make_scenarios(2, 1, P)[1]
    a, b, c = sim.simulate(g, sc2, Pd, record=True), sim.simulate(g, sc2, Pd), sim.simulate(g, sc2, Ps)
    assert a["status"] == "ok" and a["cost"] == b["cost"] == c["cost"]
    thr = np.array(a["trajectory"]["data"])[:, a["trajectory"]["channels"].index("throttle")]
    assert thr.max() <= 0.5


@need_fd
def test_phase1_eval_key_covers_new_fields():
    P = phase1_profile("T38")
    sc = sim.make_scenarios(1, 1, P)[0].to_dict()
    gnm = np.full(6, 0.5)
    k = lambda pd, msha="x": cache.eval_key("T38", gnm, pd, sc, 1, "v", "c", msha)  # noqa: E731
    base = k(P.to_dict())
    for over in ({"w_comfort": 0.1}, {"roll_kp": 0.2}, {"alt_ref_ff": False}, {"ramp_accel_g": 0.2},
                 {"gene_kinds": {}}, {"aircraft_root": None}):
        assert k({**P.to_dict(), **over}) != base, over
    assert k(P.to_dict(), "y") != base


# ------------------------------------------------------------------ agreement with genome/'s simulator
GENOME_RUNS = {"c172x": "phase1_default_c172x_v4", "T38": "phase1_t38_v4", "737": "phase1_b737_v4"}


@need_fd
@pytest.mark.skipif(not all(os.path.exists(os.path.join(GENOME, "runs", r, "best_gains.json")) for r in GENOME_RUNS.values()),
                    reason="genome/ phase-1 runs not present")
@pytest.mark.parametrize("name", list(GENOME_RUNS))
def test_reproduces_genome_phase1_costs_bitwise(name):
    """genome/'s best Phase-1 genomes re-flown here give their per-scenario cost/track/effort/comfort exactly."""
    d = os.path.join(GENOME, "runs", GENOME_RUNS[name])
    best = json.load(open(os.path.join(d, "best_gains.json")))
    det = json.load(open(os.path.join(d, "fitness_detail.json")))
    P = phase1_profile(name)
    for sc, theirs, obj in zip(sim.make_scenarios(3, 1, P), best["per_scenario"], det["objectives_per_scenario"]):
        r = sim.simulate(best["gains"], sc, P)
        assert (r["status"], r["cost"], r["track"], r["effort"], r["comfort"]) == \
               ("ok", theirs["cost"], theirs["track"], theirs["effort"], obj["comfort"])


# ------------------------------------------------------------------ trajectory export with a ramped reference
@need_fd
def test_phase1_trajectory_validates_and_hold_window(tmp_path):
    P = phase1_profile("c172x")
    sc = sim.make_scenarios(1, 1, P)[0]
    g = json.load(open(os.path.join(GENOME, "runs", GENOME_RUNS["c172x"], "best_gains.json")))["gains"] \
        if os.path.exists(os.path.join(GENOME, "runs", GENOME_RUNS["c172x"], "best_gains.json")) else genome.decode([0.6] * 6)
    r = sim.simulate(g, sc, P, record=True)
    doc = trajectory.build_doc(run_id="t", aircraft="c172x", jsbsim_version="x", git_sha="x", seed=1, generation=0,
                               fitness=r["cost"], gains=g, scenario=sc.to_dict(), scenario_index=0, sim_result=r,
                               profile=P.to_dict())
    assert "0.1 g corners" in doc["target"]["reference"]
    assert validate_traj.validate_doc(doc) == []
    ch = doc["channels"]
    d = np.array(doc["data"])
    assert np.max(np.abs(d[:, ch.index("target_rate_mps")])) == pytest.approx(10 * sim.FT, rel=1e-3)
    m = trajectory.hold_metrics(r)
    # hold window = >= 20 s after the command AND reference at rest: 28.1..50 s and 73.1..90 s at 30 Hz
    assert abs(m["hold_samples"] - round(30 * ((50 - 28.108) + (90 - 73.108)))) <= 3
    # a corrupted reference is caught by the validator
    bad = json.loads(json.dumps(doc))
    i = bad["channels"].index("target_alt_m")
    for row in bad["data"][300:310]:
        row[i] += 5.0
    assert any("target_alt_m" in e or "reference" in e for e in validate_traj.validate_doc(bad))
