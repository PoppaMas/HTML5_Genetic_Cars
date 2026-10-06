import math

import numpy as np
import pytest

import genome_schema as GS
import profiles as P


@pytest.mark.parametrize("name", P.list_profiles())
def test_profiles_load_and_pass_checks(name):
    p = P.load_profile(name)
    assert p.approximate is True and p.sources, "profiles must be labelled approximate and sourced"
    assert p.check() == []


def test_reference_factors_are_exactly_one():
    f = P.range_factors(P.load_profile("c172x"))
    assert all(v == 1.0 for v in f.values())
    a = GS.build_spec(["pitch_altitude"], f)
    b = GS.build_spec(["pitch_altitude"])
    assert [(g.min, g.max) for g in a.genes] == [(g.min, g.max) for g in b.genes]


def test_range_derivation_rule_a320():
    ref, ac = P.load_profile("c172x"), P.load_profile("a320")
    f = P.range_factors(ac, ref)
    # inner pitch: inverse control power ratio, computed by hand
    def cp(p, cm):
        v = p.design_point["kcas"] * P.KT_TO_FPS
        return 0.5 * P.RHO0 * v * v * p.wing_area_ft2 * p.chord_ft * cm * math.radians(p.control_limits_deg["elevator"]) / p.iyy
    assert np.isclose(f["inner_pitch"], cp(ref, 1.28) / cp(ac, 1.2))  # A320 uses the generic Cm_de (flagged)
    assert any("cm_de_per_rad" in w for w in ac.warnings)
    assert np.isclose(f["outer_vertical"], ref.tas_design_fps() / ac.tas_design_fps())
    assert np.isclose(f["outer_heading"], 1.0 / f["outer_vertical"])
    assert f["inner_pitch"] > 1  # less pitch accel per unit elevator -> larger gains
    assert f["outer_vertical"] < 1  # faster -> less pitch per ft/s
    spec = GS.build_spec(["pitch_altitude", "structure"], f)
    g = {x.name: x for x in spec.genes}
    base = {x.name: x for x in GS.build_spec(["pitch_altitude", "structure"]).genes}
    assert np.isclose(g["kp_pitch"].max, base["kp_pitch"].max * f["inner_pitch"])
    assert np.isclose(g["kd_alt"].min, base["kd_alt"].min * f["outer_vertical"])
    assert g["stiffness_scale"].min == base["stiffness_scale"].min  # 'none' scaling
    # normalized geometry preserved for log genes: same v -> value scales by the factor
    assert np.isclose(g["kp_pitch"].decode(0.37), base["kp_pitch"].decode(0.37) * f["inner_pitch"])


def test_profile_and_task_overrides_win(tmp_path):
    import adapter
    t = adapter.build_task({"aircraft": "f16", "blocks": {"pitch_altitude": True},
                            "gene_overrides": {"kp_pitch": {"min": 0.01, "max": 0.2}}})
    g = {x.name: x for x in t.spec.genes}
    assert (g["kp_pitch"].min, g["kp_pitch"].max) == (0.01, 0.2)


def test_check_flags_bad_design_point():
    p = P.load_profile("a320")
    p.design_point = {"kcas": 330.0, "alt_ft": 35000.0}
    assert any("MMO" in s for s in p.check())
    p.design_point = {"kcas": 100.0, "alt_ft": 35000.0}
    assert any("Vstall" in s for s in p.check())
