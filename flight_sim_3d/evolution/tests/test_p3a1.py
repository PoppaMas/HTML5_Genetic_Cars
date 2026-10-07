"""P3-A1 wiring: FD's opt-in fidelity full_a1 (flexeval_a1 / FlexBodyModelA1, INTERFACE_v2 section 13).

- full_a1 resolves to FD's published post_p3a1 strings; rigid / reduced / full still resolve to post_p25.
- full and full_a1 cache keys are disjoint (fidelity + model_version in the key, prefix guard in cache.eval_key).
- full is unchanged: a P2.5 smoke genome re-flown at full equals its stored P2.5 cache entry bit for bit (found with the
  P2.5 run's own key), and Genome's tip-verify baseline still gives 0.24220696133846464.
- pin mismatches raise (check_pins, eval.evaluate(pin=), cache guard, batch.run before any evaluation).
- configs, ladders, FlexState / v2_map with 65-node A1 wings, ga-flightsim-traj/2 export validates.
"""
import json
import os
import sqlite3

import numpy as np
import pytest

from evolution import batch, cache as cache_mod, eval as ev, fidelity as F, sim, trajectory, validate_traj
from evolution.tests.test_eval import FD_DIR, PKG, SHORT, TEAM, cfg_file, need_fd

A1_PINS_FILE = os.path.join(FD_DIR, "v2_results", "model_versions_post_p3a1.json")
P25_PINS_FILE = os.path.join(FD_DIR, "v2_results", "model_versions_post_p25.json")
need_a1 = pytest.mark.skipif(not os.path.exists(os.path.join(FD_DIR, "flexeval_a1.py")), reason="FD P3-A1 not present")
P25_SMOKE = os.path.join(PKG, "runs", "phase2-smoke-p25-s1")
G8 = {"kp_alt": 0.24, "ki_alt": 0.01, "kd_alt": 0.4, "kp_pitch": 0.07, "ki_pitch": 2e-5, "kd_pitch": 0.015,
      "kp_hdg": 2.0, "ki_hdg": 4e-4}


def resolved(name, **over):
    u = cfg_file(name)
    u.update(over)
    return batch.resolve_config(u, "t")


def profile_d(name="c172x", **over):
    cfg = resolved("phase2_smoke_p25.json")
    d = dict(next(a for a in cfg["aircraft"] if a["name"] == name)["resolved_profile"])
    d.update(over)
    return d


# ----------------------------------------------------------------------------- versions
@need_fd
@need_a1
def test_full_a1_resolves_to_published_strings():
    a1, p25 = json.load(open(A1_PINS_FILE)), json.load(open(P25_PINS_FILE))
    # P3-B1 appended full_a1_b1 after full_a1 (tests/test_p3b1.py); the first four are unchanged
    assert F.FIDELITIES[:4] == ("rigid", "reduced", "full", "full_a1") and F.RANK["full_a1"] > F.RANK["full"]
    for ac in ("c172x", "T38", "737", "f16"):
        pd = profile_d("c172x", aircraft=ac)
        assert F.model_version(pd, "full_a1") == a1[ac]["full_a1"]
        for f in ("rigid", "reduced", "full"):            # unchanged by P3-A1
            assert F.model_version(pd, f, None) == p25[ac][f] == a1[ac][f], (ac, f)
    assert (a1["c172x"]["full_a1"], a1["T38"]["full_a1"], a1["737"]["full_a1"], a1["f16"]["full_a1"]) == (
        "full_a1:flexv2a1:36fb4f5a", "full_a1:flexv2a1:f248873e", "full_a1:flexv2a1:522189cb", "full_a1:flexv2a1:4a9e12bc")


# ----------------------------------------------------------------------------- cache
@need_fd
@need_a1
def test_cache_keys_full_and_full_a1_disjoint(tmp_path):
    pd = profile_d()
    g = np.random.default_rng(3).random(20)
    scs = [s.to_dict() for s in sim.make_scenarios(3, 1, sim.Profile.from_dict(pd))]
    unit = {"scenarios": scs, "reduced_gate": None, "unit": "genome"}
    mv = {f: F.model_version(pd, f) for f in ("full", "full_a1")}
    assert mv["full"] != mv["full_a1"]
    k = {f: cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "msha", fidelity=f, model_version=mv[f])
         for f in mv}
    assert k["full"] != k["full_a1"]
    # model_version alone (or fidelity alone) also separates them
    assert cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "msha", fidelity="full_a1",
                              model_version=mv["full_a1"]) != \
        cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "msha", fidelity="full", model_version=mv["full"])
    # a string of the other fidelity can never be filed under this one
    for f, other in (("full", "full_a1"), ("full_a1", "full")):
        with pytest.raises(ValueError, match="does not belong"):
            cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "msha", fidelity=f, model_version=mv[other])
    # store a full result, a full_a1 lookup misses it (and vice versa); pins per fidelity are enforced
    c = cache_mod.EvalCache(str(tmp_path / "c.sqlite"))
    c.pins = {"c172x": {"full": mv["full"], "full_a1": mv["full_a1"]}}
    c.put_many([(k["full"], "c172x", {"fidelity": "full", "model_version": mv["full"], "cost": 1.0})])
    assert c.get_many([k["full_a1"]]) == {} and c.get_many([k["full"]])[k["full"]]["cost"] == 1.0
    c.put_many([(k["full_a1"], "c172x", {"fidelity": "full_a1", "model_version": mv["full_a1"], "cost": 2.0})])
    got = c.get_many([k["full"], k["full_a1"]])
    assert got[k["full"]]["cost"] == 1.0 and got[k["full_a1"]]["cost"] == 2.0
    with pytest.raises(RuntimeError, match="refusing"):
        c.put_many([("kx", "c172x", {"fidelity": "full_a1", "model_version": mv["full"]})])
    assert c.count() == 2


def test_rigid_reduced_full_keys_unchanged_by_guard():
    """eval_key's payload is untouched: same key as before for rigid / reduced / full (and None fidelity)."""
    import hashlib
    g = np.arange(4, dtype=float) / 7
    p = sim.Profile().to_dict()
    sc = sim.make_scenarios(1, 1, sim.Profile())[0].to_dict()
    for fid, mv in ((None, None), ("rigid", "rigid:jsbsim1.3.1:e0a73fc9"), ("full", "full:flexv2:e11b8214"),
                    ("reduced", "reduced:flexv1:68dc59aa")):
        payload = {"fidelity": fid, "model_version": mv, "model_files": "m", "aircraft": "c172x",
                   "genome": np.ascontiguousarray(g, dtype=np.float64).tobytes().hex(), "profile": p, "scenario": sc,
                   "scenario_seed": 1, "jsbsim": "1.3.1", "code_sha": "c"}
        want = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert cache_mod.eval_key("c172x", g, p, sc, 1, "1.3.1", "c", "m", fidelity=fid, model_version=mv) == want


# ----------------------------------------------------------------------------- full unchanged
@need_fd
@pytest.mark.skipif(not os.path.isdir(P25_SMOKE), reason="P2.5 smoke run not present")
@pytest.mark.skipif(not os.path.exists(os.path.join(PKG, "cache", "evals.sqlite")),
                    reason="ER's sqlite result cache (evolution/cache/evals.sqlite) is not shipped in the repo")
def test_full_unchanged_vs_p25_cache_entry():
    """A P2.5 smoke genome (c172x g0 best) re-flown at full now == its P2.5 cache entry (looked up with the P2.5 run's
    own key: code_sha b7ec7996..., read-only) and == its genomes.jsonl row, bit for bit."""
    c = json.load(open(os.path.join(P25_SMOKE, "config.json")))
    cfg, prov = c["resolved"], c["provenance"]
    rows = [json.loads(x) for x in open(os.path.join(P25_SMOKE, "genomes.jsonl"))]
    r = next(x for x in rows if x["aircraft"] == "c172x" and x["generation"] == 0 and x["rank"] == 0)
    ac = next(a for a in cfg["aircraft"] if a["name"] == "c172x")
    prof = sim.Profile.from_dict(ac["resolved_profile"])
    scs = [s.to_dict() for s in sim.make_scenarios(cfg["scenarios"], cfg["scenario_seed"], prof)]
    unit = {"scenarios": scs, "reduced_gate": None, "unit": "genome"}
    k = cache_mod.eval_key("c172x", np.array(r["genome_norm"]), ac["resolved_profile"], unit, cfg["scenario_seed"],
                           prov["jsbsim_version"], prov["code_sha"], sim.model_files_sha("c172x", prof.aircraft_root),
                           fidelity="full", model_version=r["model_version"])
    db = os.path.join(PKG, "cache", "evals.sqlite")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    hit = con.execute("SELECT result FROM evals WHERE key = ?", (k,)).fetchone()
    con.close()
    assert hit is not None, "P2.5 cache entry no longer reachable with its own key"
    cached = json.loads(hit[0])
    gains = {kk: v for kk, v in r["genome"].items() if kk in r["gains"]}
    now = F.evaluate_genome(ac["resolved_profile"], gains, r["struct"], scs, "full", ac.get("reduced_gate", 0.9))
    assert now["model_version"] == cached["model_version"] == "full:flexv2:e11b8214"
    assert now["cost"] == cached["cost"] == r["cost"]
    assert [p["cost"] for p in now["per_scenario"]] == [p["cost"] for p in cached["per_scenario"]] == r["per_scenario_cost"]
    assert now["terms"] == cached["terms"]
    assert "structural_model" not in now and "tip_bm" not in now     # A1 extras never appear at full


@need_fd
def test_full_tip_verify_baseline_unchanged():
    """Genome's c172x tip-verify baseline at full (P2.5 numbers, evolution/analysis/p25_tip_verify_evolution.json)."""
    ref = json.load(open(os.path.join(TEAM, "genome", "runs", "p25_tip_verify_c172x.json")))["baseline"]
    pd = profile_d()
    scs = [s.to_dict() for s in sim.make_scenarios(3, 1, sim.Profile.from_dict(pd))]
    r = F.evaluate_genome(pd, ref["gains"], ref["struct"], scs, "full", 0.9)
    assert r["cost"] == 0.24220696133846464 == ref["cost"]
    assert [p["cost"] for p in r["per_scenario"]] == ref["per_scenario_cost"]
    assert r["terms"]["J_wing_tip_bm_limit"] == 0.0


# ----------------------------------------------------------------------------- pins
def test_check_pins_full_a1_and_mismatch_raises():
    cfg = {"pin_model_version": {"c172x": {"rigid": "rigid:jsbsim1.3.1:e0a73fc9", "full_a1": "full_a1:flexv2a1:36fb4f5a"}}}
    mv = {"rigid": "rigid:jsbsim1.3.1:e0a73fc9", "full_a1": "full_a1:flexv2a1:36fb4f5a"}
    assert batch.check_pins(cfg, "c172x", ["rigid", "full_a1"], mv) == mv
    with pytest.raises(SystemExit, match="pin mismatch for full_a1"):
        batch.check_pins(cfg, "c172x", ["rigid", "full_a1"], dict(mv, full_a1="full_a1:flexv2a1:00000000"))
    with pytest.raises(SystemExit, match="pin mismatch for full_a1"):     # a full string is not an A1 pin
        batch.check_pins({"pin_model_version": {"c172x": {"full_a1": "full:flexv2:e11b8214"}}}, "c172x", ["full_a1"],
                         {"full_a1": "full_a1:flexv2a1:36fb4f5a"})
    with pytest.raises(SystemExit, match="no entry"):
        batch.check_pins({"pin_model_version": {"c172x": {"full": "full:flexv2:e11b8214"}}}, "c172x", ["full_a1"],
                         {"full_a1": "full_a1:flexv2a1:36fb4f5a"})
    with pytest.raises(ValueError, match="pin_model_version"):
        resolved("phase3a1_smoke.json", pin_model_version={"c172x": {"full_a2": "x"}})


@need_fd
@need_a1
def test_run_refuses_mismatched_a1_pin_before_any_evaluation(tmp_path):
    u = cfg_file("phase3a1_smoke.json")
    u.update(run_id="mm", runs_dir=str(tmp_path / "runs"), cache={"enabled": True, "path": str(tmp_path / "c.sqlite")},
             scenarios=1, ga={"pop_size": 4, "generations": 1}, trajectories={"generations": [], "scenario": 0, "sample_hz": 30},
             aircraft=[{"name": "c172x", "profile": "phase2_c172x", "overrides": dict(SHORT)}],
             pin_model_version={"c172x": {"full_a1": "full_a1:flexv2a1:00000000"}})
    b = batch.Batch(batch.resolve_config(u, "t"), log=lambda *a: None)
    b.workers = 1
    with pytest.raises(SystemExit, match="pin mismatch"):
        b.run()
    p = tmp_path / "c.sqlite"
    assert not p.exists() or sqlite3.connect(str(p)).execute("SELECT COUNT(*) FROM evals").fetchone()[0] == 0
    assert not (tmp_path / "runs" / "mm" / "run.json").exists()


@need_fd
@need_a1
def test_eval_pin_raises_and_a1_result_shape():
    run = json.load(open(os.path.join(P25_SMOKE, "run.json"))) if os.path.isdir(P25_SMOKE) else None
    if run is None:
        pytest.skip("P2.5 smoke run not present")
    ac = "c172x"
    entry = dict(next(a for a in run["aircraft"] if a["name"] == ac))
    entry["resolved_profile"] = dict(entry["resolved_profile"], **SHORT)
    run = dict(run, aircraft=[entry], scenarios=[s for s in run["scenarios"] if s["aircraft"] == ac][:1])
    entry["scenario_ids"] = [run["scenarios"][0]["id"]]
    genome = dict(G8)
    for g in entry["genes"]:
        if g["group"] == "struct":
            genome[g["name"]] = 0.02 if g["name"] == "struct_damping_ratio" else 1.0
    pin = json.load(open(A1_PINS_FILE))[ac]["full_a1"]
    with pytest.raises(RuntimeError, match="pin mismatch"):
        ev.evaluate(genome, ac, 0, run, fidelity="full_a1", pin="full:flexv2:e11b8214")
    out = ev.evaluate(genome, ac, 0, run, fidelity="full_a1", pin=pin)
    assert out["model_version"] == pin and out["fidelity"] == "full_a1" and out["margins_fidelity"] == "full_a1"
    assert set(F.TERM_KEYS) == set(out["terms"]) and len(F.TERM_KEYS) == 24
    assert out["structural_model"]["wing"]["n_el"] == 64 and out["structural_model"]["modal_dof"] == 31
    assert out["tip_bm"]["method"] == "station_exact"
    assert out["terms"]["J_wing_tip_bm_limit"] == 0.0 and out["terms"]["J_mass"] == 0.0     # baseline genome


# ----------------------------------------------------------------------------- configs / ladders
@need_fd
@need_a1
def test_p3a1_configs_match_p25_and_pins():
    a1, p25 = json.load(open(A1_PINS_FILE)), json.load(open(P25_PINS_FILE))
    for new, old in (("phase3a1_smoke.json", "phase2_smoke_p25.json"), ("phase3a1_pilot.json", "phase2_pilot_p25.json")):
        n, o = resolved(new), resolved(old)
        assert n["fidelity"] == "full_a1" and o["fidelity"] == "full"
        strip = lambda c: {k: v for k, v in batch.identity(c).items() if k not in ("fidelity", "pin_model_version", "multi_fidelity")}
        assert strip(n) == strip(o), new
        mfn, mfo = dict(n["multi_fidelity"]), dict(o["multi_fidelity"])
        if mfo.get("enabled"):
            assert mfn.pop("ladder") == ["rigid", "full_a1"] and mfo.pop("ladder") == ["rigid", "full"]
        assert mfn == mfo
        for ac, pins in n["pin_model_version"].items():
            assert pins["full_a1"] == a1[ac]["full_a1"]
            if "rigid" in pins:
                assert pins["rigid"] == p25[ac]["rigid"]
        assert "/workspace" not in open(os.path.join(PKG, "configs", new)).read()
    s = resolved("phase3a1_smoke.json")
    assert (s["ga"]["pop_size"], s["ga"]["generations"], s["struct_asymmetric"], s["init"]["sigma"]) == (16, 5, False, 0.10)
    assert [a["name"] for a in s["aircraft"]] == ["c172x", "T38", "737"] and not s["multi_fidelity"]["enabled"]
    p = resolved("phase3a1_pilot.json")
    assert (p["ga"]["pop_size"], p["ga"]["generations"]) == (64, 60)
    assert {a["name"]: a["seed"] for a in p["aircraft"]} == {"c172x": 1, "T38": 2, "737": 3}
    for cfg in (s, p):
        for ac, d in batch.model_versions_report(cfg).items():
            assert all(x["match"] is True for f, x in d.items() if f != "error"), (ac, d)


def test_ladder_accepts_full_a1_top_rung_only_ascending():
    base = cfg_file("phase3a1_pilot.json")
    assert resolved("phase3a1_pilot.json")["multi_fidelity"]["ladder"] == ["rigid", "full_a1"]
    assert resolved("phase3a1_pilot.json", multi_fidelity=dict(base["multi_fidelity"], screen=["rigid", "reduced"])
                    )["multi_fidelity"]["ladder"] == ["rigid", "reduced", "full_a1"]
    with pytest.raises(ValueError, match="screen"):       # full_a1 cannot screen for full
        resolved("phase2_pilot_p25.json", multi_fidelity=dict(base["multi_fidelity"], screen="full_a1"))
    with pytest.raises(ValueError, match="screen"):
        resolved("phase3a1_pilot.json", multi_fidelity=dict(base["multi_fidelity"], screen="full_a1"))


# ----------------------------------------------------------------------------- telemetry / trajectory export
@need_fd
@need_a1
def test_a1_flex_state_65_node_wings_and_traj2_validates():
    pd = profile_d("T38", **SHORT)
    sc = sim.make_scenarios(1, 1, sim.Profile.from_dict(pd))[0].to_dict()
    states = []

    def rec(t, fdm, fs):
        if not states:
            states.append(fs)

    F.evaluate_genome(pd, G8, None, [sc], "full_a1", 1.0, recorder=rec)
    fs = states[0]
    assert fs.fidelity == "full_a1" and fs.schema == F.FLEX_STATE_SCHEMA and fs.v2_map_version == "2.0.0"
    assert type(fs.fd_model).__name__ == "FlexBodyModelA1" and fs.fd_model.wingR.sp.n_el == 64
    comps = {c["name"]: c for c in fs.structure["components"]}
    assert len(comps["wingR"]["axis_nodes_body_m"]) == 65 and len(comps["wingR_modal"]["axis_nodes_body_m"]) == F.N_NODES
    assert len(fs.nodes()["wingR"]["w_ft"]) == 65 and {c["name"] for c in fs.node_layout()} >= {"wingR", "fusV"}
    assert "64 strips" in comps["wingR_modal"]["node_values"]
    ch = fs.channels()
    assert max(int(k.split(".")[-1]) for k in ch if k.startswith("wingR.dz.")) == 64
    assert F.v2_map_mod().validate_structure(fs.structure, ch) == []
    # export path (batch _export: eval.task(..., viz=True, telemetry='sb')): SB flight bit-identical to FD's, traj/2 valid
    r = ev.task(pd, G8, None, sc, "full_a1", True, 30.0, None, "sb")
    assert r["telemetry_check"]["sim_cost_bit_identical"] is True and r["model_version"] == F.model_version(pd, "full_a1")
    doc = trajectory.build_doc(run_id="t", aircraft="T38", jsbsim_version="x", git_sha="x", seed=1, generation=0,
                               fitness=r["cost"], gains=G8, scenario=sc, scenario_index=0, sim_result=r, profile=pd,
                               extra={"fidelity": r["fidelity"], "model_version": r["model_version"]})
    assert doc["schema"] == "ga-flightsim-traj/2" and validate_traj.validate_doc(doc, "a1") == []
    assert sum(c.startswith("wingL.dz.") for c in doc["channels"]) == 65
    # full keeps its 33-node wings and its original structure text
    st2 = []
    F.evaluate_genome(pd, G8, None, [sc], "full", 1.0, recorder=lambda t, fdm, fs: st2.append(fs) if not st2 else None)
    c2 = {c["name"]: c for c in st2[0].structure["components"]}
    assert len(c2["wingR"]["axis_nodes_body_m"]) == 33
    assert c2["wingR_modal"]["node_values"] == ("strip values (32 strips) linearly interpolated, 0 at the clamped root, "
                                                 "FD tip value at xi = 1")
