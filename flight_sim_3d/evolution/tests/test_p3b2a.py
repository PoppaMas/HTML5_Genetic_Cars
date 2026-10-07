"""P3-B2a wiring: fidelity full_a1_b2a (FD flexeval_b2 / FlexBodyModelB2 / planform_b2, INTERFACE_v2 section 15),
genome kind phase3_b2a (29 genes, wing_tc_* locked 1.0) and the opt-in Evolution-side energy term
(J_energy + J_speed_guard, outside TERM_KEYS = 24).

- fidelity wiring: FIDELITIES / RANK / pins / resolve_config / 29-gene schema with tc locked;
- B2a at identity (every encoding of the defaults) == B1 r1 bit for bit; energy off == B1 bit for bit on a B1 fly shape;
- energy on: thick_both J_energy / cost == analysis/p3b2a_energy_sanity.json exactly (same inputs as that script);
  thin section (tc_lo) clamps J_energy to 0; speed guard rule + the throttle-limited probe;
- tc genes rejected without energy (decode and config), AR / area (B2b) rejected;
- cache key carries the energy flag; B1 trajectories (node fields / header / cost) unchanged by the B2 wiring."""
import importlib.util
import json
import math
import os

import numpy as np
import pytest

from evolution import batch, cache as cache_mod, eval as ev, fidelity as F, sim
from evolution.tests.test_eval import FD_DIR, PKG, SHORT, need_fd
from evolution.tests.test_p3b1 import G8, SHAPED, SOFT, _sc, profile_d

need_b2 = pytest.mark.skipif(not os.path.exists(os.path.join(FD_DIR, "flexeval_b2.py")), reason="FD P3-B2a not present")
SANITY = os.path.join(PKG, "analysis", "p3b2a_energy_sanity.json")
PINS_B2A = {"c172x": "full_a1_b2a:flexv2b2a:847bed9b", "T38": "full_a1_b2a:flexv2b2a:b635a51d",
            "737": "full_a1_b2a:flexv2b2a:5d5a8f17", "f16": "full_a1_b2a:flexv2b2a:07b08913"}
LOCKED = ("wing_tc_root_scale", "wing_tc_tip_ratio")
pytestmark = [need_fd, need_b2]


def smoke_cfg(**over):
    u = json.load(open(os.path.join(PKG, "configs", "phase3b2a_smoke.json")))
    u.update(over)
    return batch.resolve_config(u, "t")


# ----------------------------------------------------------------------------- fidelity wiring
def test_fidelity_wiring_pins_and_29_gene_schema():
    assert F.FIDELITIES[-1] == F.B2 == "full_a1_b2a" and F.RANK[F.B2] == 5 > F.RANK[F.B1]
    assert F.B2 in F.FULL_LIKE and F.B2 in F.A1_LIKE and F.B2 in F.SHAPED
    for ac, mv in PINS_B2A.items():
        assert F.model_version(profile_d("c172x", aircraft=ac), F.B2) == mv
    assert len(F.TERM_KEYS) == 24 and not set(F.ENERGY_TERM_KEYS) & set(F.TERM_KEYS)
    cfg = smoke_cfg()
    assert cfg["fidelity"] == F.B2 and cfg["genome_kind"] == "phase3_b2a" and cfg["energy_cost"] is True
    for a in cfg["aircraft"]:
        prof = sim.Profile.from_dict(a["resolved_profile"])
        sch, groups = batch.full_schema(prof, True, False, "phase3_b2a")
        names = [g.name for g in sch]
        assert len(sch) == 29 and groups.count("shape") == 9 and not set(LOCKED) & set(names)
        assert names[20:26] == [g.name for g in F.shape_schema()]
        assert names[26:] == ["wing_dihedral_delta_deg", "wing_camber_root_delta_pct", "wing_camber_tip_delta_pct"]
    assert batch.shape_locked_of(cfg) == list(LOCKED)


# ----------------------------------------------------------------------------- identity / energy off == B1
def test_identity_equals_b1_r1_bit_for_bit():
    pd = profile_d("T38", **SHORT)
    scs = _sc(pd, 2)
    b1 = F.evaluate_genome(pd, G8, SOFT, scs, F.B1, 1.0)
    full = F.shape_schema_b2("T38")
    u0 = [g.encode(g.default) for g in full]
    for shp in (None, {}, {g.name: g.default for g in full}, u0):
        for en in (False, True):
            b2 = F.evaluate_genome(pd, G8, SOFT, scs, F.B2, 1.0, shape=shp, energy_cost=en)
            assert b2["model_version"] == PINS_B2A["T38"] and b2["status"] == b1["status"] == "ok"
            assert b2["cost"] == b1["cost"] and b2["terms"] == b1["terms"] and len(b2["terms"]) == 24
            assert [p["cost"] for p in b2["per_scenario"]] == [p["cost"] for p in b1["per_scenario"]]
            assert b2["tip_bm"] == b1["tip_bm"] and b2["mass_lb"] == b1["mass_lb"]
            if en:
                assert b2["energy_terms"]["J_energy"] == 0.0 and b2["energy_terms"]["J_speed_guard"] == 0.0


def test_energy_off_equals_b1_on_fly_shape():
    pd = profile_d("c172x", **SHORT)
    scs = _sc(pd, 2)
    b1 = F.evaluate_genome(pd, G8, SOFT, scs, F.B1, 0.9, shape=SHAPED)
    b2 = F.evaluate_genome(pd, G8, SOFT, scs, F.B2, 0.9, shape=SHAPED)
    assert b2["cost"] == b1["cost"] and b2["terms"] == b1["terms"]
    assert [p["cost"] for p in b2["per_scenario"]] == [p["cost"] for p in b1["per_scenario"]]
    # energy off: the terms may be reported (informational) but are never added to the cost
    assert b2["cost"] == float(np.mean([p["cost"] for p in b1["per_scenario"]])) or b2["cost"] == b1["cost"]
    on = F.evaluate_genome(pd, G8, SOFT, scs, F.B2, 0.9, shape=SHAPED, energy_cost=True)
    assert on["energy_terms"]["cost_fd"] == b1["cost"]
    assert on["cost"] == b1["cost"] + on["energy_terms"]["J_energy"] + on["energy_terms"]["J_speed_guard"] or \
        math.isclose(on["cost"], b1["cost"] + on["energy_terms"]["J_energy"] + on["energy_terms"]["J_speed_guard"], abs_tol=1e-15)


# ----------------------------------------------------------------------------- energy term values
@pytest.fixture(scope="module")
def sanity_mod():
    fe = F.fd_modules()["fe"]
    saved = (fe.PHASE1_CONFIG, fe.default_profile.__defaults__)
    spec = importlib.util.spec_from_file_location("p3b2a_energy_sanity", os.path.join(PKG, "analysis", "p3b2a_energy_sanity.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    yield m
    fe.PHASE1_CONFIG, fe.default_profile.__defaults__ = saved


@pytest.mark.parametrize("ac", ["c172x", "T38", "737"])
def test_energy_thick_both_matches_sanity_json(sanity_mod, ac):
    ref = json.load(open(SANITY))["aircraft"][ac]["thick_both"]
    sh = dict(sanity_mod.cases(ac))["thick_both"]
    assert sh == ref["genes"]
    scs = [s.to_dict() for s in sanity_mod.fe.phase1_scenarios(ac, sim)]
    r = F.evaluate_genome(sanity_mod.profile_d(ac), sanity_mod.gains(ac), {}, scs, F.B2, None, shape=sh, energy_cost=True)
    et = r["energy_terms"]
    assert r["status"] == "ok" and r["model_version"] == ref["model_version"]
    assert et["cost_fd"] == ref["cost_fd_raw"] and r["cost"] == ref["cost_energy_on"]
    assert et["J_energy"] == ref["J_energy"] and et["J_speed_guard"] == ref["J_speed_guard"] == 0.0
    assert et["energy_drag_increment"] == ref["energy_drag_increment"] > 0
    assert math.isclose(r["cost"], et["cost_fd"] + et["J_energy"], rel_tol=0, abs_tol=1e-15)
    assert len(r["terms"]) == 24 and "J_energy" not in r["terms"]


def test_thin_section_clamps_energy_to_zero():
    pd = profile_d("c172x", **SHORT)
    lo = next(g for g in F.shape_schema_b2("c172x") if g.name == "wing_tc_root_scale").min
    r = F.evaluate_genome(pd, G8, SOFT, _sc(pd, 1), F.B2, 0.9, shape={"wing_tc_root_scale": lo}, energy_cost=True)
    et = r["energy_terms"]
    assert et["energy_drag_increment"] < 0 and et["J_energy"] == 0.0 and r["cost"] == et["cost_fd"] + et["J_speed_guard"]


def test_speed_guard_rule_and_probe_record():
    w = F.ENERGY_W["c172x"]
    assert F._energy_j({"energy_drag_increment": 0.0, "speed_deficit_kts_mean": 2.0, "v_target_kcas": 100.0}, w) == (0.0, 0.0)
    je, js = F._energy_j({"energy_drag_increment": -0.01, "speed_deficit_kts_mean": 5.0, "v_target_kcas": 100.0}, w)
    assert je == 0.0 and js == w * 3.0 * 3.0 / 100.0
    e = {"per_scenario": [{"status": "ok", "energy_drag_increment": 0.01, "speed_deficit_kts_mean": 4.0, "v_target_kcas": 100.0},
                          {"status": "trim_failed", "energy_drag_increment": 0.5, "speed_deficit_kts_mean": 50.0, "v_target_kcas": 100.0}]}
    t = F.energy_terms(e, "c172x")
    assert t["J_energy"] == w * 0.01 / 2 and t["J_speed_guard"] == w * 3.0 * 2.0 / 100.0 / 2      # failed scenario -> 0
    with pytest.raises(ValueError, match="no energy weight"):
        F.energy_terms(e, "a320")
    p = json.load(open(SANITY))["speed_guard_probe_c172x_throttle_max"]["0.85"]
    assert p["status"] == "ok" and p["speed_deficit_kts_mean"] > 2.0 and p["J_speed_guard"] > 0
    assert math.isclose(p["cost_on"], p["cost_fd"] + p["J_speed_guard"] + p["J_energy"], abs_tol=1e-15)


# ----------------------------------------------------------------------------- rejections
def test_tc_rejected_without_energy():
    with pytest.raises(ValueError, match="energy"):
        F.shape_from_b2({"wing_tc_root_scale": 1.2}, "c172x", energy_cost=False)
    assert F.shape_from_b2({"wing_tc_root_scale": 1.2}, "c172x", energy_cost=True)["wing_tc_root_scale"] == 1.2
    pd = profile_d("c172x", **SHORT)
    with pytest.raises(ValueError, match="energy"):
        F.evaluate_genome(pd, G8, None, _sc(pd, 1), F.B2, 0.9, shape={"wing_tc_tip_ratio": 1.1})
    with pytest.raises(ValueError, match="energy_cost"):
        smoke_cfg(energy_cost=False, shape_locked=[])
    with pytest.raises(ValueError, match="energy_cost is only defined"):
        F.evaluate_genome(pd, G8, None, _sc(pd, 1), F.B1, 0.9, energy_cost=True)
    assert len(batch.full_schema(sim.Profile.from_dict(smoke_cfg(shape_locked=[])["aircraft"][0]["resolved_profile"]),
                                 True, False, "phase3_b2a", [])[0]) == 31


@pytest.mark.parametrize("gene", ["wing_aspect_scale", "wing_area_scale"])
def test_aspect_and_area_rejected(gene):
    with pytest.raises(ValueError, match="B2b"):
        F.shape_from_b2({gene: 1.1}, "c172x", energy_cost=True)


# ----------------------------------------------------------------------------- cache key
def test_cache_key_includes_energy_flag():
    pd = profile_d()
    g = np.random.default_rng(5).random(29)
    unit = {"scenarios": _sc(pd, 3), "reduced_gate": None, "unit": "genome"}
    sk = F.shape_cache_key_b2(None, "c172x", None)
    kw = dict(fidelity=F.B2, model_version=PINS_B2A["c172x"], shape_key=sk)
    k_on = cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "m", energy_cost=True, **kw)
    k_off = cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "m", energy_cost=False, **kw)
    assert k_on != k_off and sk.startswith("b2a|")
    with pytest.raises(ValueError, match="energy_cost"):
        cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "m", **kw)
    with pytest.raises(ValueError, match="energy_cost"):
        cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "m", fidelity=F.B1,
                           model_version=F.model_version(pd, F.B1), shape_key=F.shape_cache_key(None), energy_cost=True)


# ----------------------------------------------------------------------------- B1 trajectories unchanged
def _first_state(pd, fid, shape, sc):
    st = []
    F.evaluate_genome(pd, G8, None, [sc], fid, 1.0, shape=shape, recorder=lambda t, fdm, fs: st.append(fs) if not st else None)
    return st[0]


def test_b1_trajectories_unchanged():
    pd = profile_d("T38", **SHORT)
    sc = _sc(pd)[0]
    fs1 = _first_state(pd, F.B1, SHAPED, sc)
    comps = {c["name"]: c for c in fs1.structure["components"]}
    assert fs1.structure["node_layout"] == "FD flexbody_b1.node_layout_b1 (P3-B1 r1)"
    for nm in ("wingR", "wingL"):
        assert not set(F.B2_NODE_KEYS) & set(comps[nm])
    # B2a with only the B1 genes set: same node geometry, B1 per-node fields, same flight
    fs2 = _first_state(pd, F.B2, SHAPED, sc)
    c2 = {c["name"]: c for c in fs2.structure["components"]}
    for nm in comps:
        for k, v in comps[nm].items():
            if k in ("axis_nodes_body_m", "le_nodes_body_m", "te_nodes_body_m", "chord_m", "geometric_twist_rad"):
                assert c2[nm][k] == v, (nm, k)
    r1 = ev.task(pd, G8, None, sc, F.B1, True, 30.0, None, "sb", shape=SHAPED)
    r2 = ev.task(pd, G8, None, sc, F.B2, True, 30.0, None, "sb", shape=SHAPED)
    assert r1["cost"] == r2["cost"] and r1["telemetry_check"]["sim_cost_bit_identical"] is True
    assert r1["planform_header"]["schema"] == "fd-planform/1" and r1["planform_header"]["source"] == "P3-B1"
    assert r1["model_version"] == F.model_version(pd, F.B1)


# ----------------------------------------------------------------------------- P3B2A_DETERMINISM regression
SMOKE_RUN = os.path.join(PKG, "runs", "phase3b2a-smoke-s1")


@pytest.mark.skipif(not os.path.exists(os.path.join(SMOKE_RUN, "genomes.jsonl")), reason="phase3b2a-smoke-s1 not present")
def test_energy_per_scenario_independent_of_genome_status():
    """c172x:g0:r9 is an overload genome (s1 overloads, s0 / s2 are ok). Each ok scenario must carry its own
    J_energy + J_speed_guard whether it is flown alone (Sim Bridge per-scenario replay) or with the overloading
    scenario (batch whole-genome eval): per-scenario cost of the whole-genome eval == single-scenario cost, bit for bit.
    Before the fix the genome-level status gate (fidelity.py energy block) dropped the energy terms in the whole-genome
    eval of a non-ok genome only (s0 rel diff 2.0e-3, s2 1.8e-2)."""
    rj = json.load(open(os.path.join(SMOKE_RUN, "run.json")))
    if F.model_version(profile_d("c172x"), F.B2) != rj["model_version"]["c172x"]:
        pytest.skip("FD tree is not the phase3b2a-smoke-s1 pin")
    row = next(json.loads(x) for x in open(os.path.join(SMOKE_RUN, "genomes.jsonl")) if '"c172x:g0:r9"' in x)
    whole = ev.evaluate(row["genome"], "c172x", None, rj)
    assert whole["status"] == "overload" and [p["status"] for p in whole["per_scenario"]] == ["ok", "overload", "ok"]
    et = whole["energy_terms"]
    assert et["in_cost"] and any(d["J_energy"] + d["J_speed_guard"] > 0 for d in et["per_scenario"])
    assert et["per_scenario"][1] == {"J_energy": 0.0, "J_speed_guard": 0.0}
    for k in range(3):
        single = ev.evaluate(row["genome"], "c172x", k, rj)
        assert single["cost"] == whole["per_scenario_cost"][k]
        assert single["per_scenario"][0]["cost"] == whole["per_scenario"][k]["cost"]
    assert whole["cost"] == float(np.mean(whole["per_scenario_cost"]))
    assert et["cost_fd"] == row["cost"]   # FD's own cost unchanged (the energy-free value logged in the smoke)
