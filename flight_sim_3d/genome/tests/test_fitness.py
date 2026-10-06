import numpy as np
import pytest

import fitness as F


def fake_result(n=1200, dt=1 / 120, status="ok", pitch_amp=2.0, track=0.1, effort=0.05, cost=None):
    t = np.arange(n) * dt
    theta = 1.0 + pitch_amp * np.sin(2 * np.pi * t / 10)
    tel = {"t": t, "theta": theta, "q": np.gradient(theta, dt), "nz": 1 + 0.05 * np.sin(2 * np.pi * t / 3),
           "vc": 100 + np.sin(t), "v_target": np.full(n, 100.0), "throttle": 0.6 + 0.01 * np.sin(t)}
    return {"status": status, "track": track, "effort": effort, "cost": cost if cost is not None else track + 2 * effort,
            "telemetry": tel, "theta_trim": 1.0, "dt": dt, "n_steps": n, "k_end": n, "t_end": n * dt}


def test_legacy_weights_reproduce_legacy_cost_exactly():
    r = fake_result(track=0.123456789, effort=0.0456789)
    cfg = F.FitnessConfig()
    c, vals, skipped = F.score_scenario(r, cfg, cfg.needed())
    assert c == r["track"] + 2.0 * r["effort"] and not skipped
    assert not cfg.needs_telemetry()


def test_missing_telemetry_is_skipped_and_flagged():
    r = fake_result()
    cfg = F.FitnessConfig(weights={"track_alt": 1, "effort": 2, "track_heading": 1.0, "sideslip": 1.0})
    c, vals, skipped = F.score_scenario(r, cfg, cfg.needed())
    assert set(skipped) == {"track_heading", "sideslip"} and "missing telemetry" in skipped["sideslip"]
    assert c == r["track"] + 2 * r["effort"]  # skipped objectives contribute nothing (not faked)
    av = F.availability(cfg, ["theta", "nz"])
    assert "track_heading" in av and "track_alt" not in av


def test_comfort_penalizes_pitch_excess():
    cfg = F.FitnessConfig(weights={"comfort": 1.0})
    gentle = F.score_scenario(fake_result(pitch_amp=3.0), cfg, cfg.needed())[1]["comfort"]
    steep = F.score_scenario(fake_result(pitch_amp=14.0), cfg, cfg.needed())[1]["comfort"]
    terms_g = F.comfort_terms(fake_result(pitch_amp=3.0)["telemetry"], fake_result(), {})
    assert terms_g["pitch_excess"] == 0.0  # below the 5 deg free band
    assert steep > gentle


def test_structural_uses_flex_channel_when_available():
    r = fake_result()
    p = {"n_limits": (-1.52, 3.8)}
    rigid = F.obj_structural(r["telemetry"], r, p)
    assert np.isclose(rigid, 1.05 / 3.8 + np.sqrt(np.mean((r["telemetry"]["nz"] - 1) ** 2)) / 2.8, rtol=1e-3)
    n = r["n_steps"]
    r["telemetry"].update(wing_root_bending=np.full(n, 150.0), tip_deflection=np.full(n, 0.5), tip_twist=np.full(n, 1.0))
    # no flex_params -> still the rigid proxy (the run was not coupled)
    assert np.isclose(F.obj_structural(r["telemetry"], r, p), rigid)
    r["flex_params"] = {"wing_root_bending_1g": 100.0, "wing_root_bending_limit": 120.0, "tip_deflection_limit": 1.0,
                        "tip_twist_limit": 3.0, "wing_mass_frac_delta": 0.1}
    flex = F.obj_structural(r["telemetry"], r, p)
    # 0.25 * RMS(M - M1g)/M1g + 2 * (150/120 - 1)^2 + 0 + 0 + 0.3 * 0.1
    assert np.isclose(flex, 0.25 * 0.5 + 2 * 0.25 ** 2 + 0.03)
    r["flex_params"]["wing_mass_frac_delta"] = 0.3  # heavier (stiffer) wing costs more
    assert F.obj_structural(r["telemetry"], r, p) > flex


def test_failure_keeps_hard_penalty():
    r = fake_result(status="crash", cost=1500.0)
    cfg = F.FitnessConfig(weights={"track_alt": 1, "comfort": 1})
    c, vals, _ = F.score_scenario(r, cfg, cfg.needed())
    assert c == 1500.0 and np.isnan(vals["comfort"])


def test_aggregation_modes():
    v = [1.0, 2.0, 3.0, 10.0]
    assert F.aggregate(v, {"mode": "mean"}) == 4.0
    assert F.aggregate(v, {"mode": "worst"}) == 10.0
    assert F.aggregate(v, {"mode": "cvar", "alpha": 0.5}) == 6.5
    assert F.aggregate(v, {"mode": "mean_cvar", "alpha": 0.25, "lambda": 0.5}) == 7.0
    with pytest.raises(ValueError):
        F.aggregate(v, {"mode": "median"})


def test_unknown_objective_rejected():
    with pytest.raises(ValueError):
        F.FitnessConfig(weights={"nope": 1}).needed()
