"""Altitude ramp, per-aircraft pitch clamp / envelope, and T38 / 737 support."""
import numpy as np
import pytest

import adapter
import fitness as F
import profiles as P
import scenarios as SC
import sim_ext
from flightsim_path import orig_sim


def test_ramp_reference_closed_form():
    e = sim_ext.ExtScenario(seed=1, ramp_fpm=600)  # legacy steps 4000 -> 4200 @5s -> 4000 @50s
    assert e.target(0.0) == (4000.0, 0.0)
    assert e.target(10.0) == (4050.0, 5.0) and e.target_rate(10.0) == 10.0
    assert e.target(25.0)[0] == 4200.0 and e.target_rate(30.0) == 0.0
    assert e.target(55.0) == (4150.0, 50.0) and e.target_rate(55.0) == -10.0
    assert e.target(80.0)[0] == 4000.0
    assert e.target_cmd(10.0) == 4200.0
    # command reversal in the middle of a ramp starts from where the reference is
    e2 = sim_ext.ExtScenario(seed=1, ramp_fpm=600, steps=[(0.0, 4000.0), (5.0, 4200.0), (15.0, 4000.0)])
    assert e2.target(15.0)[0] == 4100.0 and e2.target(20.0)[0] == 4050.0 and e2.target(30.0)[0] == 4000.0
    # no ramp -> identical to sim.Scenario.target
    S = orig_sim()
    for t in (0, 4.9, 5, 30, 50, 89):
        assert sim_ext.ExtScenario(seed=1).target(t) == S.Scenario(seed=1).target(t)


def test_legacy_preset_has_no_conditions():
    t = adapter.load_task("altitude_hold_legacy")
    assert t.conditions == {} and t.sim_fixed == {}
    S = orig_sim()
    assert all(type(s) is S.Scenario for s in t.make_scenarios(3, 1))


def test_phase1_conditions_come_from_profile():
    t = adapter.load_task("phase1_default")
    assert t.conditions["ramp_fpm"] == 600 and t.conditions["alt_ref_ff"] is True
    assert t.conditions["pitch_cmd_limits_deg"] == (-8.0, 12.0)  # c172x profile (= legacy 12 deg)
    for ac, model, h0, clamp in (("t38", "T38", 10000.0, (-10.0, 15.0)), ("b737", "737", 10000.0, (-5.0, 10.0))):
        ta = adapter.load_task("phase1_default", {"aircraft": ac})
        s = ta.make_scenarios(2, 1)[1]
        assert s.aircraft == model and s.h0_ft == h0 and s.pitch_cmd_limits_deg == clamp
        assert [h for _, h in s.steps] == [h0, h0 + 200.0, h0] and s.ramp_fpm == 600
        assert s.nz_limits == tuple(ta.profile.sim_envelope["nz_limits"])
        assert ta.sim_fixed["kp_roll"] != 0.05  # helper gains rescaled for the jet
    # preset controller override beats the profile
    to = adapter.load_task("phase1_default", {"controller": {"pitch_cmd_limits_deg": [-3, 6]}})
    assert to.make_scenarios(1, 1)[0].pitch_cmd_limits_deg == (-3, 6)


def test_same_disturbances_as_legacy_set():
    S = orig_sim()
    a = adapter.load_task("phase1_default").make_scenarios(3, 1)
    b = S.make_scenarios(3, 1)
    for x, y in zip(a, b):
        assert (x.seed, x.wind_north_fps, x.gust_sigma_fps, x.discrete_gust_fps) == (y.seed, y.wind_north_fps, y.gust_sigma_fps, y.discrete_gust_fps)


def test_t38_uses_per_norm_derivatives():
    p = P.load_profile("t38")
    v = p.design_point["kcas"] * P.KT_TO_FPS
    q = 0.5 * P.RHO0 * v * v
    assert np.isclose(p.control_power()["pitch"], q * 170.0 * 6.73 * 0.195 / 36075.0)
    assert not any("cm_de" in w for w in p.warnings)


def test_overshoot_metric():
    t = np.arange(0, 90, 0.5)
    h = np.where(t < 5, 4000.0, 4230.0)
    h = np.where(t >= 50, 3990.0, h)
    tel = {"t": t, "h": h}
    assert F.overshoot(tel, [(0, 4000.0), (5, 4200.0), (50, 4000.0)]) == 30.0


@pytest.mark.sim
def test_pitch_clamp_and_ramp_reach_the_sim():
    g = dict(kp_alt=0.5, ki_alt=0.0, kd_alt=0.1, kp_pitch=0.05, ki_pitch=0.01, kd_pitch=0.02)
    sc = sim_ext.ExtScenario(seed=1, duration_s=30.0, ramp_fpm=600, pitch_cmd_limits_deg=(-2.0, 3.0))
    r = sim_ext.simulate(g, sc)
    tel = r["telemetry"]
    assert r["status"] == "ok"
    assert np.max(tel["theta_cmd"] - r["theta_trim"]) <= 3.0 + 1e-9
    assert np.isclose(np.interp(10.0, tel["t"], tel["target"]), 4050.0, atol=0.2)
    assert np.all(tel["target_cmd"][tel["t"] >= 5.0] == 4200.0)
    # default clamp is sim.py's 12 deg
    r12 = sim_ext.simulate(g, sim_ext.ExtScenario(seed=1, duration_s=30.0))
    assert np.max(r12["telemetry"]["theta_cmd"] - r12["theta_trim"]) > 3.0


@pytest.mark.sim
@pytest.mark.parametrize("ac", ["t38", "b737"])
def test_jets_load_trim_and_fly_default_genome(ac):
    t = adapter.load_task("phase1_default", {"aircraft": ac})
    sc = t.make_scenarios(1, 1)  # calm scenario
    r = t.evaluate(t.spec.decode(t.spec.default_genome()), sc, record=True)
    d = r["diagnostics"][0]
    assert r["per_scenario"][0]["status"] == "ok", r["per_scenario"]
    assert d["max_ref_err_ft"] < 150.0 and d["min_nz"] > 0.5 and d["max_nz"] < 1.5
