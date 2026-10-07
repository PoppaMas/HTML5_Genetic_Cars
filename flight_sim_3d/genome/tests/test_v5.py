"""Phase-1 v5: hold-quality term, downdraft disturbance scenario, frozen v4 preset, legacy untouched."""
import json
import math
import os

import numpy as np
import pytest

import adapter
import fitness as F
import profiles as P
import sim_ext

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4_RUNS = {"c172x": "hdg_after_c172x_s1", "t38": "hdg_after_t38_s1", "b737": "hdg_after_b737_s1"}


def _tel(t, h, target, cmd, rate):
    return {"t": np.asarray(t, float), "h": np.asarray(h, float), "target": np.asarray(target, float),
            "target_cmd": np.asarray(cmd, float), "target_rate": np.asarray(rate, float)}


def test_hold_windows_exclude_ramp_and_settle():
    dt = 0.5
    t = np.arange(0, 40, dt)
    cmd = np.where(t < 10, 1000.0, 1100.0)
    target = np.where(t < 10, 1000.0, np.minimum(1000.0 + 10.0 * (t - 10), 1100.0))  # ramp 10..20 s
    rate = np.where((t >= 10) & (t < 20), 10.0, 0.0)
    tel = _tel(t, target, target, cmd, rate)
    m = F.hold_mask(tel, settle_s=5.0)
    assert not m[(t < 5)].any() and m[(t >= 5) & (t < 10)].all()      # first window after its 5 s settle
    assert not m[(t >= 10) & (t < 25)].any() and m[t >= 25].all()     # ramp + settle excluded
    assert len(F.hold_windows(tel, 5.0)) == 2


def test_hold_osc_penalizes_oscillation_not_offset():
    t = np.arange(0, 60, 0.1)
    cmd = np.full_like(t, 1000.0)
    z = np.zeros_like(t)
    p = {"hold_settle_s": 5.0, "hold_ref_ft": 5.0}
    offset = _tel(t, cmd - 3.0, cmd, cmd, z)
    assert F.obj_hold_osc(offset, {}, p) == pytest.approx(0.0, abs=1e-12)
    osc = _tel(t, cmd + 5.0 * np.sin(2 * np.pi * t / 10), cmd, cmd, z)
    assert F.obj_hold_osc(osc, {}, p) == pytest.approx(1 / math.sqrt(2), rel=2e-2)  # RMS of a 5 ft sine / 5 ft


def test_draft_series_onset_and_level():
    sc = sim_ext.ExtScenario(label="d", steps=[(0.0, 1000.0)], seed=1, draft_fps=4.0, draft_t_s=10.0, draft_ramp_s=4.0)
    n = int(30 / sim_ext.S.DT)
    w = sc.vertical_gust_series(n)
    t = np.arange(n) * sim_ext.S.DT
    assert np.all(w[t < 10] == 0.0)
    assert w[np.searchsorted(t, 12.0)] == pytest.approx(2.0, rel=1e-6)
    assert np.allclose(w[t >= 14], 4.0)


def test_flags_off_for_legacy_and_default_on_for_v5():
    assert adapter.load_task("altitude_hold_legacy").disturbance == {}
    d = adapter.load_task("phase1_default")
    assert "hold_osc" not in d.fitness.weights and d.disturbance == {}
    v5 = adapter.load_task("phase1_v5")
    assert v5.fitness.weights["hold_osc"] > 0 and v5.disturbance["downdraft_fps"] > 0
    with pytest.raises(ValueError, match="shared"):
        adapter.load_task("altitude_hold_legacy", {"hold_quality": True})
    with pytest.raises(ValueError, match="shared"):
        adapter.load_task("altitude_hold_legacy", {"disturbance_scenario": True})


def test_v4_preset_equals_phase1_default():
    a, b = adapter.load_task("phase1_v4"), adapter.load_task("phase1_default")
    assert a.fitness.weights == b.fitness.weights and a.fitness.params == b.fitness.params
    assert [vars(s) for s in a.make_scenarios(3, 1)] == [vars(s) for s in b.make_scenarios(3, 1)]
    assert a.sim_fixed == b.sim_fixed


def test_disturbance_scenario_appended_and_scaled_per_aircraft():
    sh = P.load_shared()
    g = math.radians(sh["task"]["v5"]["disturbance"]["gamma_equiv_deg"])
    for ac in ("c172x", "t38", "b737"):
        t = adapter.load_task("phase1_v5", {"aircraft": ac})
        scs = t.make_scenarios(3, 1)
        assert len(scs) == 4 and scs[-1].label == "downdraft"
        d = scs[-1]
        assert d.gust_sigma_fps == 0 and d.discrete_gust_fps == 0 and d.wind_north_fps == 0 and d.wind_east_fps == 0
        assert d.steps == [(0.0, scs[0].steps[0][1])]
        assert d.draft_fps == pytest.approx(t.profile.tas_design_fps() * math.tan(g), rel=2e-3)  # V_TAS tan(1.5 deg)


@pytest.mark.sim
@pytest.mark.parametrize("ac", sorted(V4_RUNS))
def test_v4_preset_reproduces_v4_runs_bit_for_bit(ac):
    f = os.path.join(HERE, "runs", V4_RUNS[ac], "best_gains.json")
    if not os.path.exists(f):
        pytest.skip("v4 run not present")
    g = json.load(open(f))
    t = adapter.load_task("phase1_v4", {"aircraft": ac})
    assert t.evaluate(g["gains"], t.make_scenarios(3, g["config"]["scenario_seed"]))["cost"] == g["best_cost"]


@pytest.mark.sim
def test_downdraft_needs_integrator_to_null_residual():
    t = adapter.load_task("phase1_v5")
    sc = t.make_scenarios(3, 1)[-1]
    base = {"kp_alt": 0.382, "kd_alt": 0.938, "kp_pitch": 0.0235, "ki_pitch": 5.1e-4, "kd_pitch": 0.0135}
    res = {}
    for ki in (0.0, 0.0141):
        r = sim_ext.simulate({**t.sim_fixed, **base, "ki_alt": ki}, sc, record=True)
        assert r["status"] == "ok"
        res[ki] = F.diagnostics(r)["draft_residual_ft"]
    assert res[0.0] > 1.0                 # P-D only: standing offset below the commanded altitude
    assert abs(res[0.0141]) < 0.5 * res[0.0]


# ----------------------------------------------------------------------------------------------- ki_alt (v5 only)
def test_ki_alt_upper_bound_raised_for_v5_only():
    for ac in ("c172x", "t38", "b737"):
        v4, v5 = adapter.load_task("phase1_v4", {"aircraft": ac}), adapter.load_task("phase1_v5", {"aircraft": ac})
        k4, k5 = (next(g for g in t.spec.genes if g.name == "ki_alt") for t in (v4, v5))
        assert k4.max == 0.05 and k5.max == 0.5 and k4.min == k5.min and k4.scale == k5.scale == "log0"
        assert [g for g in v4.spec.genes if g.name != "ki_alt"] == [g for g in v5.spec.genes if g.name != "ki_alt"]
    ex = json.load(open(os.path.join(HERE, "exports", "evolution_phase1_v5_profiles.json")))
    assert all(p["gain_bounds"]["ki_alt"][1] == 0.5 for p in ex["profiles"].values())
    ex4 = json.load(open(os.path.join(HERE, "exports", "evolution_phase1_profiles.json")))
    assert all(p["gain_bounds"]["ki_alt"][1] == 0.05 for p in ex4["profiles"].values())


@pytest.mark.sim
@pytest.mark.parametrize("ac,run", [("c172x", "v5_sweep_w01"), ("t38", "v5_t38_s1"), ("b737", "v5_b737_s1")])
def test_v5_costs_unchanged_by_wider_bound(ac, run):
    # the bound only widens the search space: the v5 bests (gains) still score exactly the same
    f = os.path.join(HERE, "runs", run, "best_gains.json")
    if not os.path.exists(f):
        pytest.skip("v5 run not present")
    g = json.load(open(f))
    t = adapter.load_task("phase1_v5", {"aircraft": ac})
    assert t.evaluate(g["gains"], t.make_scenarios(3, g["config"]["scenario_seed"]))["cost"] == g["best_cost"]
