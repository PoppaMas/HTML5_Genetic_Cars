"""FD integration: flex genes, margin constraints, gear/root/throttle/min-speed fixes (non-legacy paths only)."""
import dataclasses

import numpy as np
import pytest

import adapter
import fd_bridge
import fitness as F
import genome_schema as GS
import profiles as P
import sim_ext
from flightsim_path import orig_sim

S = orig_sim()


def test_structure_genes_match_fd_interface():
    """Exactly FD's final Phase-1 block (INTERFACE.md 1a / flexwing.STRUCT_SCHEMA)."""
    fw = fd_bridge.flexwing()
    st = {g.name: g for g in GS.BLOCKS["structure"]}
    assert list(st) == list(fd_bridge.FLEX_GENES)
    want = {"stiffness_scale": (0.6, 2.0, "log"), "torsion_bend_ratio": (0.8, 1.15, "linear"),
            "struct_damping_ratio": (0.005, 0.05, "log"), "nonstructural_mass_scale": (0.8, 1.25, "log")}
    assert set(st) == set(want)
    for k, (lo, hi, sc) in want.items():
        assert (st[k].min, st[k].max, st[k].scale) == (lo, hi, sc), k
        name, flo, fhi = fw.GENOME_GENES[k]  # FD's own ranges
        assert (flo, fhi) == (lo, hi)
    fd_sched = {g.name: (g.min, g.max, g.kind) for g in fw.STRUCT_SCHEMA}
    assert fd_sched["stiffness_scale"] == (0.6, 2.0, "log") and fd_sched["torsion_bend_ratio"] == (0.8, 1.15, "linear")
    for gone in ("mass_centroid_shift", "wing_mass_scale", "elastic_axis_frac", "section_cg_frac", "tip_mass_frac",
                 "bend_stiffness_scale", "torsion_stiffness_scale"):
        assert gone not in GS.ALL_GENES
    # every decodable genome (corners + middle) is accepted by FD's strict parser and maps to tied_stiffness(s, r)
    spec = GS.build_spec(["structure"])
    for v in (0.0, 0.5, 1.0):
        d = spec.decode(np.full(spec.n_genes, v))
        ov = fd_bridge.flex_overrides(d)
        assert ov["ei_scale"] == pytest.approx(d["stiffness_scale"])
        assert ov["gj_scale"] == pytest.approx(d["stiffness_scale"] * d["torsion_bend_ratio"])


def test_only_flex_genes_are_sent_to_fd():
    g = {"stiffness_scale": 1.2, "torsion_bend_ratio": 1.0, "kp_alt": 0.1, "elastic_axis_frac": 0.5,
         "section_cg_frac": 0.3, "tip_mass_frac": 0.05, "bend_stiffness_scale": 1.0}
    ov = fd_bridge.flex_overrides(g)  # non-flex keys are never forwarded, so FD's raising guards are never tripped
    assert set(ov) <= {"ei_scale", "gj_scale", "zeta", "nonstruct_scale"}
    with pytest.raises(ValueError):  # and FD itself rejects what we must never send
        fd_bridge.flexwing().overrides_from_genome({"bend_stiffness_scale": 1.0}, strict=True)
    with pytest.raises(ValueError):
        fd_bridge.flexwing().overrides_from_genome({"stiffness_scale": 2.5}, strict=True)


def test_per_aircraft_constants_fixed_cg_aft_of_ea():
    for name, ea, cg in (("c172x", 0.38, 0.42), ("t38", 0.40, 0.42), ("b737", 0.36, 0.38)):
        t = adapter.load_task("phase1_flex", {"aircraft": name})
        assert t.spec.n_genes == 6 + 2 + 4  # pitch + heading hold (shared default) + FD structure
        assert fd_bridge.fd_chord_defaults(P.load_profile(name).jsbsim_model) == {"elastic_axis_frac": ea, "section_cg_frac": cg}
        assert t.flex_constants == {"elastic_axis_frac": ea, "section_cg_frac": cg, "tip_mass_frac": 0.0}
        assert not any(k in t.sim_fixed for k in ("elastic_axis_frac", "section_cg_frac", "tip_mass_frac"))
        assert not P.load_profile(name).check()


@pytest.mark.parametrize("ea,cg", [(0.42, 0.42), (0.45, 0.40), (0.35, 0.45)])
def test_chord_overrides_rejected(ea, cg):
    # Phase 1: no config route to the chordwise axes at all (FD fixed them); CG at/ahead of EA gets the explicit message
    with pytest.raises(ValueError, match="fixed per aircraft"):
        adapter.load_task("phase1_flex", {"flex": {"elastic_axis_frac": ea, "section_cg_frac": cg}})
    with pytest.raises(ValueError, match="overrides for unknown genes|not genes in Phase 1"):
        adapter.load_task("phase1_flex", {"gene_overrides": {"section_cg_frac": {"min": 0.3, "max": 0.5}}})
    if cg <= ea:
        with pytest.raises(ValueError, match="behind the elastic axis"):
            fd_bridge.check_chord_constants(ea, cg)


def test_profile_flex_constants_guard():
    import copy
    prof = copy.deepcopy(P.load_profile("c172x"))
    prof.flex_constants = {"elastic_axis_frac": 0.42, "section_cg_frac": 0.40}
    assert any("behind the elastic axis" in m for m in prof.check())


def test_flutter_summary_handles_inf_capped_and_fd_flags():
    cap = fd_bridge.PROVISIONAL_FLUTTER_CAP
    s = fd_bridge.flutter_summary({"flutter_margin": float("inf")})
    assert s["flutter_margin"] == cap and s["no_flutter_below_cap"] and np.isfinite(s["flutter_margin"])
    s = fd_bridge.flutter_summary({"flutter_margin": 1.3})
    assert s["flutter_margin"] == 1.3 and not s["no_flutter_below_cap"]
    # FD's future output: capped value + explicit flag + documented cap
    s = fd_bridge.flutter_summary({"flutter_margin": 2.5, "flutter_margin_cap": 2.5, "no_flutter_below_cap": True})
    assert s["flutter_margin"] == 2.5 and s["no_flutter_below_cap"] and s["flutter_cap_source"] == "fd"
    # FD's actual field names (flexwing.margins, 2026-10-06)
    s = fd_bridge.flutter_summary({"flutter_margin": 3.0, "margin_cap": 3.0, "flutter_not_found_below_cap": True})
    assert s["flutter_margin"] == 3.0 and s["no_flutter_below_cap"]
    s = fd_bridge.flutter_summary({"flutter_margin": 1.24, "margin_cap": 3.0, "flutter_not_found_below_cap": False})
    assert s["flutter_margin"] == 1.24 and not s["no_flutter_below_cap"]
    s = fd_bridge.flutter_summary({"flutter_margin": 1.1, "flutter_found": True})
    assert not s["no_flutter_below_cap"]


def test_profile_fd_decisions():
    assert P.load_profile("b737").task_conditions()["min_kcas"] == 195.0
    assert P.load_profile("t38").task_conditions()["throttle_max"] == 0.5
    assert P.load_profile("c172x").task_conditions()["throttle_max"] is None


def _gains(task):
    return {**task.sim_fixed, **task.spec.decode(task.spec.default_genome())}


@pytest.mark.sim
def test_margin_constraints():
    t = adapter.load_task("phase1_flex")
    sc = t.make_scenarios(1, 1)
    names = [x.name for x in t.spec.genes]
    st = {g.name: g for g in GS.BLOCKS["structure"]}
    g = t.spec.default_genome().copy()
    soft = g.copy()  # low-stiffness corner s 0.6, r 0.8: FD min margin ~0.84 -> intended hard fail, no flight
    soft[names.index("stiffness_scale")] = 0.0
    soft[names.index("torsion_bend_ratio")] = 0.0
    r = t.evaluate(t.spec.decode(soft), sc)
    assert r["aeroelastic"]["flutter_margin"] < 1.0 and r["cost"] == F.FAIL_COST and r["violation"] > 0
    assert r["per_scenario"][0]["status"].startswith("aeroelastic_")
    assert np.isfinite(r["aeroelastic"]["flutter_margin"]) and np.isfinite(r["aeroelastic"]["div_margin"])
    # 1.0 <= margin < 1.2 -> flies, pays the hinge penalty on top of the aggregate cost
    mid = g.copy()
    mid[names.index("stiffness_scale")] = st["stiffness_scale"].encode(0.9)
    pre = fd_bridge.precheck("c172x", t.spec.decode(mid))
    assert 1.0 <= pre["margins"]["flutter_margin"] < 1.2 and not pre["fail"]
    assert pre["terms"]["J_flutter_margin"] > 0
    # stiff + high ratio: finite, capped, flag consistent
    hi = fd_bridge.flutter_summary(fd_bridge.precheck("737", {"stiffness_scale": 2.0, "torsion_bend_ratio": 1.15})["margins"])
    assert hi["flutter_margin"] <= hi["flutter_margin_cap"] == 3.0


@pytest.mark.sim
def test_mass_term_makes_stiffness_cost():
    pre1 = fd_bridge.precheck("c172x", {"stiffness_scale": 1.0, "torsion_bend_ratio": 1.0})
    pre2 = fd_bridge.precheck("c172x", {"stiffness_scale": 2.0, "torsion_bend_ratio": 1.0})
    assert pre2["terms"]["delta_wing_mass_lb"] > 0 and pre2["terms"]["J_mass"] > pre1["terms"]["J_mass"]


@pytest.mark.sim
def test_fd_root_and_gear_do_not_change_c172x():
    t = adapter.load_task("phase1_default")
    g = _gains(t)
    sc = t.make_scenarios(2, 1)[1]
    a = sim_ext.simulate(g, dataclasses.replace(sc, use_fd_root=False), record=False)["cost"]
    b = sim_ext.simulate(g, sc, record=False)["cost"]
    assert a == b


@pytest.mark.sim
def test_jets_gear_up_throttle_clamp():
    t = adapter.load_task("phase1_default", {"aircraft": "t38"})
    sc = t.make_scenarios(1, 1)[0]
    r = sim_ext.simulate(_gains(t), sc)
    assert r["status"] == "ok" and r["telemetry"]["throttle"].max() <= 0.5 + 1e-12
    t = adapter.load_task("phase1_default", {"aircraft": "b737"})
    sc = t.make_scenarios(1, 1)[0]
    assert sc.min_kcas == 195.0
    assert sim_ext.simulate(_gains(t), sc, record=False)["status"] == "ok"


@pytest.mark.sim
def test_flex_sim_c172x_short():
    t = adapter.load_task("phase1_flex")
    sc = dataclasses.replace(t.make_scenarios(1, 1)[0], duration_s=10.0)
    r = sim_ext.simulate(_gains(t), sc)
    assert r["status"] == "ok" and r["flex_mode"] == "twoway"
    tel = r["telemetry"]
    assert len(tel["wing_root_bending"]) == r["k_end"]
    # 1-g root moment within 10 % of FD's estimate in level flight (start of run)
    assert abs(np.median(tel["wing_root_bending"][:60]) / r["flex_params"]["wing_root_bending_1g"] - 1) < 0.1
    assert np.isfinite(F.obj_structural(tel, r, {}))


def test_legacy_scenario_has_no_fd_path():
    sc = S.make_scenarios(1, 1)[0]
    assert not isinstance(sc, sim_ext.ExtScenario)
