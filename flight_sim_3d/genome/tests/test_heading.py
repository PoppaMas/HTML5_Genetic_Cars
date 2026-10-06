"""Phase-1 heading hold: outer loop heading -> bank cmd -> wing leveller, cost term, shared-set flag, legacy untouched."""
import dataclasses
import json
import math
import os

import numpy as np
import pytest

import adapter
import fitness as F
import genome_schema as GS
import profiles as P
import sim_ext

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4 = os.path.join(HERE, "runs", "phase1_default_c172x_v4", "best_gains.json")
PITCH = {"kp_alt": 0.382, "ki_alt": 0.0141, "kd_alt": 0.938, "kp_pitch": 0.0235, "ki_pitch": 5.1e-4, "kd_pitch": 0.0135}


def _calm(task):
    return task.make_scenarios(1, 1)[0]


def _fly(task, gains, sc):
    return sim_ext.simulate({**task.sim_fixed, **gains}, sc, record=True)


def test_wrap180():
    assert sim_ext._wrap180(0.0 - 359.0) == pytest.approx(1.0)
    assert sim_ext._wrap180(359.0 - 1.0) == pytest.approx(-2.0)
    assert np.allclose(F.wrap180([350.0, -350.0, 10.0, 180.0]), [-10.0, 10.0, 10.0, -180.0])


def test_heading_rms_objective_wraps():
    tel = {"psi": np.array([359.0, 1.0, 358.0, 2.0]), "psi_target": np.zeros(4)}
    v = F.OBJECTIVES["track_heading_rms"].fn(tel, {}, {"hdg_rms_ref_deg": 2.0})
    assert v == pytest.approx(math.sqrt((1 + 1 + 4 + 4) / 4) / 2.0)


def test_schema_subset_reuses_roll_heading_block():
    sp = GS.build_spec(["pitch_altitude"], gene_subsets={"roll_heading": list(GS.HEADING_HOLD_GENES)})
    assert sp.names[-2:] == ["kp_hdg", "ki_hdg"] and sp.n_genes == 8
    assert {g.block for g in sp.genes[-2:]} == {"roll_heading"}
    assert sp.fixed["kp_roll"] == 0.05 and sp.fixed["kd_roll"] == 0.02  # inner loop = sim.py's wing leveller
    assert sp.genes[-1] == GS.ALL_GENES["ki_hdg"]  # same GeneSpec, not a copy with other ranges
    with pytest.raises(ValueError, match="not genes of that block"):
        GS.build_spec(["pitch_altitude"], gene_subsets={"roll_heading": ["kp_alt"]})


def test_shared_flag_on_for_phase1_off_for_legacy():
    sh = P.load_shared()
    for ac in ("c172x", "t38", "b737"):
        assert sh["aircraft"][ac]["heading_hold"]["enabled"] is True
        t = adapter.load_task("phase1_default", {"aircraft": ac})
        assert t.spec.names[-2:] == ["kp_hdg", "ki_hdg"]
        assert t.fitness.weights["track_heading_rms"] == sh["task"]["heading_hold"]["weight"]
        assert t.conditions["bank_cmd_limit_deg"] == sh["aircraft"][ac]["heading_hold"]["bank_limit_deg"]
    leg = adapter.load_task("altitude_hold_legacy")
    assert leg.spec.n_genes == 6 and "track_heading_rms" not in leg.fitness.weights and leg.conditions == {}
    off = adapter.load_task("phase1_default", {"heading_hold": False})
    assert off.spec.n_genes == 6 and "bank_cmd_limit_deg" not in off.conditions
    with pytest.raises(ValueError, match="heading_hold needs"):
        adapter.load_task("altitude_hold_legacy", {"heading_hold": True})


def test_heading_gene_bounds_follow_the_range_rule():
    """Shared kp_hdg/ki_hdg bounds = reference (c172x) range x V/V_ref (TAS), to 3 significant figures."""
    sh = P.load_shared()["aircraft"]
    for ac in ("c172x", "t38", "b737"):
        f = P.range_factors(P.load_profile(ac))["outer_heading"]
        for g in GS.HEADING_HOLD_GENES:
            ref = GS.ALL_GENES[g]
            b = sh[ac]["heading_hold"]["gain_bounds"][g]
            assert b["min"] == pytest.approx(ref.min * f, rel=5e-3) and b["max"] == pytest.approx(ref.max * f, rel=5e-3)
            assert b["scale"] == ref.scale


def test_bank_limits_follow_standard_rate_rule():
    sh = P.load_shared()["aircraft"]
    for ac in ("c172x", "t38", "b737"):
        v = P.load_profile(ac).tas_design_fps()
        std = math.degrees(math.atan(v * math.radians(3.0) / 32.174))
        assert sh[ac]["heading_hold"]["bank_limit_deg"] == math.floor(min(std, 25.0))


@pytest.mark.sim
def test_flag_off_reproduces_v4_bit_for_bit():
    if not os.path.exists(V4):
        pytest.skip("v4 run not present")
    g = json.load(open(V4))
    t = adapter.load_task("phase1_default", {"heading_hold": False})
    assert t.evaluate(g["gains"], t.make_scenarios(3, 1))["cost"] == g["best_cost"]


@pytest.mark.sim
def test_c172x_drift_without_hold_and_stopped_with_it():
    t = adapter.load_task("phase1_default")
    sc = _calm(t)
    drift = _fly(adapter.load_task("phase1_default", {"heading_hold": False}), PITCH, sc)["telemetry"]["psi"]
    assert F.wrap180(drift[-1] - drift[0]) > 15.0  # ~25 deg: prop torque + P-only wing leveller
    tel = _fly(t, {**PITCH, "kp_hdg": 1.0, "ki_hdg": 0.03}, sc)["telemetry"]
    assert abs(F.wrap180(tel["psi"][-1] - tel["psi"][0])) < 1.0


@pytest.mark.sim
def test_heading_change_takes_short_way_and_respects_bank_limit():
    t = adapter.load_task("phase1_default")
    sc = dataclasses.replace(_calm(t), heading_steps=[(0.0, 0.0), (5.0, 330.0)])  # 30 deg LEFT, not 330 right
    tel = _fly(t, {**PITCH, "kp_hdg": 5.0, "ki_hdg": 0.0}, sc)["telemetry"]
    lim = t.conditions["bank_cmd_limit_deg"]
    assert np.nanmax(np.abs(tel["phi_cmd"])) == pytest.approx(lim)  # saturated at the per-aircraft clamp
    assert np.nanmin(tel["phi_cmd"]) == pytest.approx(-lim)          # banked left
    rel = F.wrap180(tel["psi"])                       # heading relative to north, (-180, 180]
    assert rel.min() > -40.0 and rel.max() < 2.0      # only ever turned left, overshoot bounded
    assert abs(F.wrap180(330.0 - tel["psi"][-1])) < 5.0


@pytest.mark.sim
def test_jets_hold_heading_with_default_gains():
    for ac in ("t38", "b737"):
        t = adapter.load_task("phase1_default", {"aircraft": ac})
        g = {x.name: x.default for x in t.spec.genes}
        g["ki_hdg"] = t.spec.genes[-1].max / 10
        r = _fly(t, g, _calm(t))
        assert r["status"] == "ok"
        psi = r["telemetry"]["psi"]
        assert np.max(np.abs(F.wrap180(psi - psi[0]))) < 1.0
