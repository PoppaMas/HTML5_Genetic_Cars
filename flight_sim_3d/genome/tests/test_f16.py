"""F-16 extra profile (not Phase 1): FD's prepared copy, FD's fixed flex values / point masses, provisional gains."""
import json
import re

import numpy as np
import pytest

import adapter
import fd_bridge
import profiles as P
import sim_ext

pytestmark = pytest.mark.skipif(not fd_bridge.root_ok("f16"), reason="FD prepared f16 not present")


def test_f16_profile_uses_fd_fixed_values():
    p = P.load_profile("f16")
    assert p.design_point == {"kcas": 350.0, "alt_ft": 10000.0}
    fc = p.flex_constants
    assert fd_bridge.fd_chord_defaults("f16") == {"elastic_axis_frac": fc["elastic_axis_frac"], "section_cg_frac": fc["section_cg_frac"]}
    assert (fc["elastic_axis_frac"], fc["section_cg_frac"], fc["tip_mass_frac"]) == (0.40, 0.43, 0.0)
    fd_bridge.check_chord_constants(fc["elastic_axis_frac"], fc["section_cg_frac"])
    assert not p.check()


def test_f16_point_mass_indices_match_fd_copy():
    xml = open(f"{fd_bridge.FD_ROOT}/aircraft/f16/f16.xml").read()
    names = re.findall(r'<pointmass\s+name="([^"]+)"', re.sub(r"<!--.*?-->", "", xml, flags=re.S))
    pm = P.load_profile("f16").point_masses
    assert [names[pm[k]] for k in ("payload_placeholder", "pilot", "flexwing_dm_R", "flexwing_dm_L")] == \
        ["payload_placeholder", "Pilot", "flexwing_dm_R", "flexwing_dm_L"]


def test_f16_flex_margins_match_fd_interface():
    def pre(s, r):
        return fd_bridge.precheck("f16", {"stiffness_scale": s, "torsion_bend_ratio": r, "struct_damping_ratio": 0.02,
                                          "nonstructural_mass_scale": 1.0}, {})
    d = pre(1.0, 1.0)
    assert d["margins"]["flutter_margin"] == pytest.approx(1.39, abs=0.005) and not d["fail"]
    assert d["margins"]["div_not_found_below_cap"]
    c = pre(0.6, 0.8)
    assert c["margins"]["flutter_margin"] == pytest.approx(0.963, abs=0.001) and c["fail"]


def test_f16_gain_bounds_marked_provisional():
    t = adapter.build_task({"aircraft": "f16", "blocks": {"pitch_altitude": True}})
    assert all(g.provisional for g in t.spec.genes)
    assert any(w.startswith("PROVISIONAL gain bounds for f16") for w in t.warnings)


@pytest.mark.sim
def test_f16_loads_and_trims_from_fd_copy():
    t = adapter.build_task({"aircraft": "f16", "blocks": {"pitch_altitude": True}, "scenarios": {"set": "legacy"}})
    sc = t.make_scenarios(1, 1)[0]
    sc.duration_s = 2.0
    g = {x.name: x.default for x in t.spec.genes}
    r = sim_ext.simulate({**t.sim_fixed, **g}, sc, record=True)
    assert r["status"] == "ok" and r["aircraft"] == "f16"
    assert abs(r["theta_trim"] - 1.045) < 0.05  # FD INTERFACE.md 5b: alpha = theta = 1.045 deg at 350 KCAS / 10 kft
    assert np.isfinite(r["telemetry"]["h"]).all()
