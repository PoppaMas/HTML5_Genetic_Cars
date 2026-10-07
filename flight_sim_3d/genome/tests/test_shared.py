"""Shared Phase-1 profile set, ramp-corner smoothing, alignment knobs, Evolution Runner cross-check."""
import dataclasses
import json
import os

import numpy as np
import pytest

import adapter
import export_shared_profiles as X
import profiles as P
import sim_ext
from flightsim_path import orig_sim

S = orig_sim()
EVO = os.environ.get("EVOLUTION_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evolution"))


def test_shared_c172x_equals_prototype_constants():
    sh = P.load_shared()
    t, a = sh["task"], sh["aircraft"]["c172x"]
    assert t["alt_err_scale_ft"] == S.ALT_ERR_SCALE_FT and t["max_alt_err_ft"] == S.MAX_ALT_ERR_FT
    assert t["itae_t0_s"] == S.ITAE_T0_S and t["itae_cap_s"] == S.ITAE_CAP_S
    assert a["min_kcas"] == S.MIN_KCAS and tuple(a["nz_limits"]) == tuple(S.NZ_LIMITS)
    assert tuple(a["pitch_cmd_limits_deg"]) == tuple(S.PITCH_CMD_LIMITS_DEG)
    sc = S.make_scenarios(1, 1)[0]
    assert [(ts, sc.h0_ft + dh) for ts, dh in t["steps_rel_ft"]] == [tuple(x) for x in sc.steps]


def test_profiles_take_trim_envelope_from_shared_set():
    sh = P.load_shared()["aircraft"]
    for name in ("c172x", "t38", "b737"):
        p = P.load_profile(name)
        assert p.shared_set == P.SHARED_FILE
        assert p.design_point == {"kcas": sh[name]["trim_kcas"], "alt_ft": sh[name]["trim_alt_ft"]}
        assert p.task_conditions()["min_kcas"] == sh[name]["min_kcas"]
    assert P.load_profile("t38").task_conditions()["min_kcas"] == 180.0
    assert P.load_profile("b737").task_conditions()["nz_limits"] == (-1.0, 2.5)


def test_conflicting_profile_field_rejected(tmp_path):
    d = json.load(open(os.path.join(P.PROFILE_DIR, "t38.json")))
    d["design_point"] = {"kcas": 250.0, "alt_ft": 10000.0}
    f = tmp_path / "t38.json"
    f.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="conflicts with"):
        P.load_profile(str(f))


def test_presets_use_shared_set_legacy_does_not():
    t = adapter.load_task("phase1_default")
    assert t.conditions["ramp_accel_g"] == 0.1 and t.conditions["alt_ref_ff"] is True
    assert t.conditions["steps_rel_ft"] == [(0.0, 0.0), (5.0, 200.0), (50.0, 0.0)]
    w_h = P.load_shared()["task"]["heading_hold"]["weight"]
    assert t.fitness.weights == {"track_alt": 1.0, "effort": 2.0, "comfort": 0.05, "track_heading_rms": w_h}
    assert adapter.load_task("phase1_no_comfort").fitness.weights == {"track_alt": 1.0, "effort": 2.0, "track_heading_rms": w_h}
    leg = adapter.load_task("altitude_hold_legacy")
    assert leg.conditions == {} and leg.spec.genes[0].max == 0.5  # legacy kp_alt range untouched by shared bounds
    jet = adapter.load_task("phase1_default", {"aircraft": "b737"})
    g = {x.name: x for x in jet.spec.genes}
    assert g["kp_pitch"].min == 0.002  # union with the legacy range: Evolution Runner's 737 optimum (0.021) is inside


def test_export_matches_evolution_profile_schema():
    d = X.evolution_profiles()
    keys = {"h0_ft", "speed_kts", "duration_s", "steps_rel_ft", "min_agl_ft", "min_kcas", "max_abs_theta_deg",
            "max_abs_phi_deg", "nz_limits", "max_alt_err_ft", "pitch_cmd_limits_deg", "alt_i_limit_deg", "pitch_i_limit",
            "thr_kp", "thr_ki", "throttle_max", "throttle_all_engines", "gear_up", "alt_err_scale_ft", "itae_t0_s",
            "itae_cap_s", "w_effort", "gain_bounds", "fail_base", "origin_lat_deg", "origin_lon_deg", "extra_props", "aircraft",
            "ramp_fpm", "ramp_accel_g", "alt_ref_ff", "w_comfort", "comfort_params", "comfort_weights", "gene_kinds",
            "roll_kp", "roll_kd", "aircraft_root"} | set(X.HEADING_KEYS)  # evolution/sim.py Profile (heading hold since 05:33)
    pending = set(d["_needs_code"]["profile_keys"])
    assert pending == set() and d["_needs_code"]["genes"] == []
    for name, prof in d["profiles"].items():
        assert set(k for k in prof if not k.startswith("_")) <= keys, name
        assert set(d["_needs_code"]["genes"]) <= set(prof["gain_bounds"])
        assert prof["gene_kinds"] == {"ki_alt": "log0", "ki_pitch": "log0", "ki_hdg": "log0"}
    c = d["profiles"]["phase1_c172x"]
    assert c["thr_kp"] == 0.05 and c["thr_ki"] == 0.01 and c["gain_bounds"]["kp_alt"] == [0.002, 2.0]
    assert c["roll_kp"] == 0.05 and c["roll_kd"] == 0.02 and c["heading_hold"] is True and c["bank_limit_deg"] == 16.0
    t = d["profiles"]["phase1_T38"]
    assert t["thr_kp"] == adapter.load_task("phase1_default", {"aircraft": "t38"}).sim_fixed["kp_spd"]  # full precision
    v5 = X.evolution_v5_profiles()
    assert set(v5["_needs_code"]["profile_keys"]) == set(X.V5_KEYS)
    for name, prof in v5["profiles"].items():  # v5 = v4 + exactly the pending keys (+ the v5-only ki_alt upper bound 0.5)
        base = json.loads(json.dumps(d["profiles"][name]))
        assert prof["gain_bounds"]["ki_alt"] == [base["gain_bounds"]["ki_alt"][0], 0.5] and base["gain_bounds"]["ki_alt"][1] == 0.05
        base["gain_bounds"]["ki_alt"] = prof["gain_bounds"]["ki_alt"]
        assert {k: v for k, v in prof.items() if k in base and not k.startswith("_")} == {k: v for k, v in base.items() if not k.startswith("_")}
        assert set(prof) - set(base) == set(X.V5_KEYS), name
        assert prof["disturbance_scenario"]["downdraft_fps"] > 0 and prof["w_hold"] > 0


def _ramp_table(sc):
    ts = np.arange(0, sc.duration_s, S.DT)
    return ts, np.array([sc.target(x)[0] for x in ts]), np.array([sc.target_rate(x) for x in ts])


def test_smoothed_ramp_is_accel_limited_trapezoid():
    sc = adapter.load_task("phase1_default").make_scenarios(1, 1)[0]
    ts, P_, V = _ramp_table(sc)
    A = np.diff(V) / S.DT
    assert np.abs(A).max() <= 0.1 * 32.174 * (1 + 1e-9)
    assert np.abs(V).max() == pytest.approx(10.0)  # 600 fpm cruise
    assert P_.max() == pytest.approx(4200.0, abs=1e-9) and P_[-1] == 4000.0  # no reference overshoot
    assert np.allclose(np.diff(P_) / S.DT, 0.5 * (V[1:] + V[:-1]), atol=0.01)  # rate = d(ref)/dt (exact within a phase; O(a*dt) at phase joins)
    # mid-ramp reversal and a short (triangular) move stay limited
    for steps in ([(0, 4000.0), (5, 4200.0), (12, 3990.0), (13, 4010.0), (40, 4000.0)], [(0, 4000.0), (5, 4020.0), (30, 4000.0)]):
        s2 = dataclasses.replace(sc, steps=steps)
        _, _, V2 = _ramp_table(s2)
        assert np.abs(np.diff(V2) / S.DT).max() <= 0.1 * 32.174 * (1 + 1e-9) and np.abs(V2).max() <= 10.0 + 1e-9


def test_sharp_ramp_unchanged_without_accel_limit():
    sc = adapter.load_task("phase1_default").make_scenarios(1, 1)[0]
    sharp = dataclasses.replace(sc, ramp_accel_g=None)
    assert sharp.target(15.0)[0] == pytest.approx(4000 + 10.0 * 10.0)
    assert sharp.target_rate(15.0) == 10.0


@pytest.mark.sim
def test_alt_err_scale_and_trim_throttle_check():
    t = adapter.load_task("phase1_default", {"aircraft": "t38"})
    g = {**t.sim_fixed, **t.spec.decode(t.spec.default_genome())}
    sc = t.make_scenarios(1, 1)[0]
    a = sim_ext.simulate(g, sc, record=False)
    b = sim_ext.simulate(g, dataclasses.replace(sc, alt_err_scale_ft=300.0), record=False)
    assert b["track"] == pytest.approx(a["track"] / 3.0, rel=1e-12)
    r = sim_ext.simulate(g, dataclasses.replace(sc, throttle_max=0.3), record=False)  # T38 trims at 0.354
    assert r["status"] == "trim_failed"


@pytest.mark.sim
def test_network_io_detection():
    import fd_bridge
    assert fd_bridge.network_io('<input port="5137" />') == ['<input port="5137" />']
    assert len(fd_bridge.network_io('<input port="5139" type="QTJSBSIM" rate="20"> x </input>')) == 1
    assert len(fd_bridge.network_io('<output name="localhost" type="SOCKET" protocol="UDP" port="5138" rate="10"/>')) == 1
    assert fd_bridge.network_io('<!--output name="localhost" type="SOCKET" port="5138"-->') == []  # commented out
    assert fd_bridge.network_io('<output name="run.csv" type="CSV" rate="10"/>') == []           # file output
    assert fd_bridge.network_io('<fcs_function><input>fcs/elevator-cmd-norm</input><output>fcs/x</output>') == []


def test_fd_root_has_no_network_io_and_stock_737_is_refused():
    """FD strips network I/O at the source (prepare_aircraft); we load its copies directly and refuse anything else."""
    import jsbsim
    import fd_bridge
    for m in ("c172x", "T38", "737", "f16"):
        if fd_bridge.root_ok(m):
            assert fd_bridge.check_no_network_io(fd_bridge.FD_ROOT, m) == fd_bridge.FD_ROOT
    with pytest.raises(fd_bridge.NetworkIOError, match="network I/O"):
        sim_ext._new_fdm("737", None)  # the stock 737 in the venv declares telnet 5137 + QTJSBSIM 5139 inputs
    fd_bridge.check_no_network_io(jsbsim.get_default_root_dir(), "c172x")  # the legacy stock c172x is clean


@pytest.mark.skipif(not os.path.isdir(os.path.join(EVO, "runs", "bench_jets-j1")), reason="evolution runs not present")
def test_our_sim_reproduces_evolution_runner_t38_cost():
    import crosscheck_evolution as C
    cfg = json.load(open(os.path.join(EVO, "runs", "bench_jets-j1", "config.json")))["resolved"]
    summ = json.load(open(os.path.join(EVO, "runs", "bench_jets-j1", "summary.json")))
    ac = next(a for a in summ["aircraft"] if a["aircraft"] == "T38")
    prof = dict(cfg["profiles"][ac["profile"]], _model="T38")
    scs = C.their_scenarios(prof, 3, 1)
    cost = np.mean([C.their_cost(sim_ext.simulate(ac["best_gains"], sc, record=True), sc, prof) for sc in scs])
    assert cost == pytest.approx(ac["best_fitness"], rel=1e-12)
