import numpy as np
import pytest

import adapter
import fitness as F
import scenarios as SC
import sim_ext


def test_inert_blocks_refused_unless_allowed():
    with pytest.raises(ValueError, match="not consumed"):
        adapter.build_task({"blocks": {"pitch_altitude": True, "yaw_damper": True}})
    with pytest.raises(ValueError, match="not consumed"):  # structure without flex mode: nothing consumes it
        adapter.build_task({"blocks": {"pitch_altitude": True, "structure": True}})
    t = adapter.build_task({"blocks": {"pitch_altitude": True, "structure": True}, "allow_inert_blocks": True})
    assert any(w.startswith("INERT") for w in t.warnings) and any("notional" in w for w in t.warnings)
    # flex mode on: FD's coupled_sim consumes the structure genes -> guard lifted
    t = adapter.build_task({"blocks": {"pitch_altitude": True, "structure": True}, "flex": {"enabled": True}})
    assert not any(w.startswith("INERT") for w in t.warnings) and t.conditions["flex_mode"] == "twoway"


def test_skip_flag_for_unavailable_objectives():
    t = adapter.build_task({"blocks": {"pitch_altitude": True},
                            "fitness": {"weights": {"track_alt": 1, "effort": 2, "sideslip": 1}}, "backend": "legacy_sim"})
    assert any(w.startswith("SKIP objective sideslip") for w in t.warnings)


def test_all_presets_load():
    import glob
    import os
    files = glob.glob(os.path.join(adapter.PRESET_DIR, "*.json")) + glob.glob(os.path.join(adapter.HERE, "experiments", "*.json"))
    assert len(files) >= 9
    import json
    import fd_bridge
    try:  # Phase 2 presets need FD's flex v2 (absent in the Phase 1 repo layout)
        fd_bridge.flexbody()
        have_v2 = True
    except Exception:
        have_v2 = False
    for f in files:
        if json.load(open(f)).get("genome") == "phase4_rings":
            continue  # standalone loader phase4_rings.load_preset (tests/test_phase4_rings.py), not adapter.load_task
        if not have_v2 and int(json.load(open(f)).get("flex", {}).get("version", 1)) == 2:
            continue
        t = adapter.load_task(f)
        assert t.spec.n_genes > 0
        name = os.path.basename(f)
        if name.startswith(("altitude_hold_legacy", "step_")):
            assert "ramp_fpm" not in t.conditions  # legacy / historical experiments keep the instant step
        else:
            assert t.conditions.get("ramp_fpm") == 600, name  # every new preset ramps at 600 fpm


def test_robust_scenarios_deterministic_and_perturbed():
    a = SC.make_scenarios(4, 7, {"set": "robust"})
    b = SC.make_scenarios(4, 7, {"set": "robust"})
    assert [vars(x) for x in a] == [vars(x) for x in b]
    assert a[0].payload_delta_lb == 0 and a[0].noise_alt_ft == 0  # nominal calm
    assert all(x.payload_delta_lb != 0 for x in a[1:])
    with pytest.raises(NotImplementedError):
        SC.make_scenarios(2, 1, {"set": "robust", "aircraft_variants": ["a320"]})


@pytest.mark.sim
def test_payload_perturbation_reaches_jsbsim():
    e = sim_ext.ExtScenario(seed=1, duration_s=2.0, payload_delta_lb=200.0)
    base = sim_ext.simulate(dict(kp_alt=0.05, ki_alt=0.001, kd_alt=0.3, kp_pitch=0.05, ki_pitch=0.01, kd_pitch=0.02),
                            sim_ext.ExtScenario(seed=1, duration_s=2.0), record=False)
    pert = sim_ext.simulate(dict(kp_alt=0.05, ki_alt=0.001, kd_alt=0.3, kp_pitch=0.05, ki_pitch=0.01, kd_pitch=0.02), e, record=False)
    assert np.isclose(pert["weight_lb"] - base["weight_lb"], 200.0)


@pytest.mark.sim
def test_heading_block_drives_real_heading_change():
    t = adapter.load_task("c172_3axis")
    sc = sim_ext.ExtScenario(seed=1, duration_s=40.0, steps=[(0.0, 4000.0)], heading_steps=[(0.0, 0.0), (5.0, 30.0)])
    gains = t.spec.decode(t.spec.default_genome())
    r = sim_ext.simulate(gains, sc, record=True)
    assert r["status"] == "ok"
    psi = r["telemetry"]["psi"]
    assert abs(((psi[-1] - 30.0 + 180) % 360) - 180) < 5.0  # turned onto the new heading
    res = t.evaluate(gains, [sc])
    assert "track_heading" in res["objectives"] and not res["skipped"]
