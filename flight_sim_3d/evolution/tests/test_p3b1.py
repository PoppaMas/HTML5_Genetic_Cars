"""P3-B1 wiring: FD's opt-in fidelity full_a1_b1 (flexeval_b1 / FlexBodyModelB1 / planform_b1, INTERFACE_v2 section 14)
and genome kind phase3_b1 (controller 8 | structure 12 | FD's 6 shape genes).

- full_a1_b1 resolves to FD's post_p3b1r1 strings (B1 r1; r0 post_p3b1 superseded); rigid / reduced / full / full_a1
  strings unchanged.
- node layout (r1): full_a1_b1 uses flexbody_b1.node_layout_b1; baseline shape = flexbody.node_layout exactly; shaped
  EA moves with sweep and chord (not twist); per-node chord / twist / LE / TE exported in SI on the FE wings.
- shape schema = FD's locked list; storage = FD's [0,1] vector encoding (bit-identical decode).
- caches disjoint: full_a1 vs full_a1_b1, and two different shapes.
- baseline shape at full_a1_b1 == full_a1 exactly (cost, 24 terms, per scenario, margins, mass).
- geometry gate reject: status geometry_gate:<reason>, cost = fail_cost, model not built, no flight.
- pin mismatch raises; phase2_flex decode / gen-0 unchanged; Genome's operator spec (log-space chord mutation,
  0.25 x half-range sigma, whole-block crossover); planform trajectory header (fd-planform/1) validates.
"""
import json
import math
import os
import sqlite3

import numpy as np
import pytest

from evolution import batch, cache as cache_mod, eval as ev, fidelity as F, ga, sim, trajectory, validate_traj
from evolution.tests.test_eval import FD_DIR, PKG, SHORT, TEAM, cfg_file, need_fd

B1_PINS_FILE = os.path.join(FD_DIR, "v2_results", "model_versions_post_p3b1r1.json")       # B1 r1 (current)
B1_R0_PINS_FILE = os.path.join(FD_DIR, "v2_results", "model_versions_post_p3b1.json")     # r0, superseded
A1_PINS_FILE = os.path.join(FD_DIR, "v2_results", "model_versions_post_p3a1.json")
P25_PINS_FILE = os.path.join(FD_DIR, "v2_results", "model_versions_post_p25.json")
need_b1 = pytest.mark.skipif(not os.path.exists(os.path.join(FD_DIR, "planform_b1.py")), reason="FD P3-B1 not present")
G8 = {"kp_alt": 0.24, "ki_alt": 0.01, "kd_alt": 0.4, "kp_pitch": 0.07, "ki_pitch": 2e-5, "kd_pitch": 0.015,
      "kp_hdg": 2.0, "ki_hdg": 4e-4}
SOFT = {"wing_ei_root": 1.0, "wing_ei_taper_1": 1.0, "wing_ei_taper_2": 1.0, "wing_ei_taper_3": 1.0, "wing_ei_taper_4": 0.75,
        "wing_gj_ratio_root": 1.0, "wing_gj_ratio_tip": 1.0, "wing_nsm_root": 1.0, "wing_nsm_tip": 1.0,
        "tail_stiffness_scale": 1.0, "fuselage_stiffness_scale": 1.0, "struct_damping_ratio": 0.02}
SHAPED = {"wing_chord_taper_1": 0.95, "wing_chord_taper_2": 0.95, "wing_chord_taper_3": 0.85,
          "wing_twist_mid_deg": -0.5, "wing_twist_tip_deg": -3.0, "wing_sweep_qc_delta_deg": 5.0}
CHORD = ("wing_chord_taper_1", "wing_chord_taper_2", "wing_chord_taper_3")


def resolved(name, **over):
    u = cfg_file(name)
    u.update(over)
    return batch.resolve_config(u, "t")


def profile_d(name="c172x", **over):
    cfg = resolved("phase3a1_smoke.json")
    d = dict(next(a for a in cfg["aircraft"] if a["name"] == name)["resolved_profile"])
    d.update(over)
    return d


def _sc(pd, n=1):
    return [s.to_dict() for s in sim.make_scenarios(n, 1, sim.Profile.from_dict(pd))]


# ----------------------------------------------------------------------------- versions / schema
@need_fd
@need_b1
def test_full_a1_b1_resolves_to_published_strings():
    b1, a1, p25 = (json.load(open(f)) for f in (B1_PINS_FILE, A1_PINS_FILE, P25_PINS_FILE))
    assert F.FIDELITIES == ("rigid", "reduced", "full", "full_a1", "full_a1_b1") and F.B1 == "full_a1_b1"
    assert F.RANK["full_a1_b1"] > F.RANK["full_a1"] and "full_a1_b1" in F.FULL_LIKE
    for ac in ("c172x", "T38", "737", "f16"):
        pd = profile_d("c172x", aircraft=ac)
        assert F.model_version(pd, "full_a1_b1") == b1[ac]["full_a1_b1"]
        assert F.model_version(pd, "full_a1") == a1[ac]["full_a1"] == b1[ac]["full_a1"]
        for f in ("rigid", "reduced", "full"):
            assert F.model_version(pd, f, None) == p25[ac][f] == b1[ac][f], (ac, f)
    assert [b1[a]["full_a1_b1"] for a in ("c172x", "T38", "737", "f16")] == [
        "full_a1_b1:flexv2b1:56ee798e", "full_a1_b1:flexv2b1:7e871977", "full_a1_b1:flexv2b1:6523753c",
        "full_a1_b1:flexv2b1:617078a9"]
    r0 = json.load(open(B1_R0_PINS_FILE))      # r0 file untouched; every B1 string changed, nothing else did
    for ac in ("c172x", "T38", "737", "f16"):
        assert r0[ac]["full_a1_b1"] != b1[ac]["full_a1_b1"]
        assert {k: v for k, v in r0[ac].items() if k != "full_a1_b1"} == {k: v for k, v in b1[ac].items() if k != "full_a1_b1"}
    assert F.label("full_a1_b1").startswith("full_a1_b1(")


@need_fd
@need_b1
def test_shape_schema_is_fd_list_and_decode_bit_identical():
    pb1 = F.fd_b1_modules()["pb1"]
    sch = F.shape_schema()
    assert [g.name for g in sch] == list(pb1.SHAPE_NAMES) == [
        "wing_chord_taper_1", "wing_chord_taper_2", "wing_chord_taper_3", "wing_twist_mid_deg", "wing_twist_tip_deg",
        "wing_sweep_qc_delta_deg"]
    assert [(g.min, g.max, g.default) for g in sch] == [(0.85, 1.05, 1.0)] * 3 + [(-2.0, 1.0, 0.0), (-4.0, 1.0, 0.0),
                                                                                  (-5.0, 5.0, 0.0)]
    assert all(g.kind == "linear" for g in sch)
    U = np.random.default_rng(7).random((200, 6))
    U[0], U[1] = 0.0, 1.0
    for u in U:   # our Gene.decode == FD's [0,1]^6 vector decode, bit for bit
        assert {g.name: g.decode(v) for g, v in zip(sch, u)} == pb1.decode_shape_b1(list(u))
    u0 = [g.encode(g.default) for g in sch]     # identity planform decodes to FD's defaults EXACTLY (baseline short-cut)
    assert pb1.is_baseline_shape({g.name: g.decode(v) for g, v in zip(sch, u0)})
    with pytest.raises(ValueError):             # FD never clips
        F.shape_from({"wing_chord_taper_3": 0.80})
    with pytest.raises(ValueError):
        F.shape_from({"wing_dihedral_delta_deg": 2.0})


# ----------------------------------------------------------------------------- cache
@need_fd
@need_b1
def test_cache_keys_disjoint_a1_vs_b1_and_between_shapes(tmp_path):
    pd = profile_d()
    g = np.random.default_rng(3).random(26)
    unit = {"scenarios": _sc(pd, 3), "reduced_gate": None, "unit": "genome"}
    mv = {f: F.model_version(pd, f) for f in ("full_a1", "full_a1_b1")}
    base = dict(fidelity="full_a1_b1", model_version=mv["full_a1_b1"])
    k_a1 = cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "m", fidelity="full_a1", model_version=mv["full_a1"])
    k_b0 = cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "m", shape_key=F.shape_cache_key(None), **base)
    k_b1 = cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "m", shape_key=F.shape_cache_key(SHAPED), **base)
    k_b2 = cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "code", "m",
                              shape_key=F.shape_cache_key(dict(SHAPED, wing_sweep_qc_delta_deg=4.0)), **base)
    assert len({k_a1, k_b0, k_b1, k_b2}) == 4      # same genome bytes: fidelity / version / shape key separate them
    assert F.shape_cache_key(None) == F.shape_cache_key({}) == F.shape_cache_key(F.fd_b1_modules()["pb1"].shape_defaults())
    with pytest.raises(ValueError, match="does not belong"):     # an A1 string can never be filed under B1 (or back)
        cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "c", "m", fidelity="full_a1_b1", model_version=mv["full_a1"],
                           shape_key="b1|x")
    with pytest.raises(ValueError, match="does not belong"):
        cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "c", "m", fidelity="full_a1", model_version=mv["full_a1_b1"])
    with pytest.raises(ValueError, match="needs FD's shape_cache_key"):
        cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "c", "m", **base)
    with pytest.raises(ValueError, match="only part of full_a1_b1"):
        cache_mod.eval_key("c172x", g, pd, unit, 1, "1.3.1", "c", "m", fidelity="full_a1", model_version=mv["full_a1"],
                           shape_key="b1|x")
    c = cache_mod.EvalCache(str(tmp_path / "c.sqlite"))
    c.pins = {"c172x": {"full_a1": mv["full_a1"], "full_a1_b1": mv["full_a1_b1"]}}
    c.put_many([(k_a1, "c172x", {"fidelity": "full_a1", "model_version": mv["full_a1"], "cost": 1.0}),
                (k_b1, "c172x", {"fidelity": "full_a1_b1", "model_version": mv["full_a1_b1"], "cost": 2.0})])
    got = c.get_many([k_a1, k_b0, k_b1, k_b2])
    assert set(got) == {k_a1, k_b1} and got[k_a1]["cost"] == 1.0 and got[k_b1]["cost"] == 2.0
    with pytest.raises(RuntimeError, match="refusing"):
        c.put_many([("kx", "c172x", {"fidelity": "full_a1_b1", "model_version": mv["full_a1"]})])


# ----------------------------------------------------------------------------- baseline shape == A1
@need_fd
@need_b1
def test_baseline_shape_equals_full_a1_exactly():
    pd = profile_d("T38", **SHORT)
    scs = _sc(pd, 2)
    a1 = F.evaluate_genome(pd, G8, SOFT, scs, "full_a1", 1.0)
    pb1 = F.fd_b1_modules()["pb1"]
    u0 = [g.encode(g.default) for g in F.shape_schema()]
    for shp in (None, {}, pb1.shape_defaults(), u0):
        b1 = F.evaluate_genome(pd, G8, SOFT, scs, "full_a1_b1", 1.0, shape=shp)
        assert b1["model_version"] == F.model_version(pd, "full_a1_b1") and b1["fidelity"] == "full_a1_b1"
        assert b1["cost"] == a1["cost"] and b1["status"] == a1["status"] == "ok"
        assert len(b1["terms"]) == 24 and b1["terms"] == a1["terms"]
        assert [p["cost"] for p in b1["per_scenario"]] == [p["cost"] for p in a1["per_scenario"]]
        assert b1["margins"] == a1["margins"] and b1["mass_lb"] == a1["mass_lb"] and b1["tip_bm"] == a1["tip_bm"]
        assert b1["geometry_gate"]["ok"] and not b1["geometry_gate_reject"] and b1["planform"]["planform_baseline"]
    assert a1["terms"]["J_wing_tip_bm_limit"] > 0.0          # non-trivial genome (tip-soft)
    # a non-baseline shape is refused below full_a1_b1, ignored by rigid
    with pytest.raises(ValueError, match="only consumed"):
        F.evaluate_genome(pd, G8, SOFT, scs, "full_a1", 1.0, shape=SHAPED)
    r0 = F.evaluate_genome(pd, G8, None, scs[:1], "rigid")
    assert F.evaluate_genome(pd, G8, None, scs[:1], "rigid", shape=SHAPED)["cost"] == r0["cost"]


# ----------------------------------------------------------------------------- geometry gate
@need_fd
@need_b1
def test_geometry_gate_rejects_without_flying(monkeypatch):
    """FD's whole B1 box is gate-feasible on all 4 aircraft (FD test + our corner scan), so an illegal planform is
    injected with a stricter in-process taper limit (wrapping FD's own gate function, version strings unchanged)."""
    m = F.fd_b1_modules()
    pb1, fbb1, fe = m["pb1"], m["fbb1"], F.fd_modules()["fe"]
    orig = pb1.geometry_gate_strips

    def strict(pf):
        r = orig(pf)
        if r.ok and r.details["taper_tip_root"] < 0.6:
            return pb1.GeometryGateResult(False, "extreme_taper", r.details)
        return r
    monkeypatch.setattr(pb1, "geometry_gate_strips", strict)
    calls = {"sim": 0, "build": 0}

    def no_sim(*a, **k):
        calls["sim"] += 1
        raise AssertionError("flown")

    class NoBuild:
        def __init__(self, *a, **k):
            calls["build"] += 1
            raise AssertionError("built")
    monkeypatch.setattr(fe, "_simulate_with_hook", no_sim)
    monkeypatch.setattr(fbb1, "FlexBodyModelB1", NoBuild)
    pd = profile_d("c172x", **SHORT)
    scs = _sc(pd, 2)
    r = F.evaluate_genome(pd, G8, None, scs, "full_a1_b1", 0.9, shape=SHAPED)
    P = sim.Profile.from_dict(pd)
    assert r["status"] == "geometry_gate:extreme_taper" and r["geometry_gate_reject"] is True
    assert r["cost"] == 2 * P.fail_base == 2000.0 and r["feasible"] is False
    assert [p["cost"] for p in r["per_scenario"]] == [2000.0, 2000.0] and all(p["not_flown"] for p in r["per_scenario"])
    assert all(p["status"] == "geometry_gate:extreme_taper" for p in r["per_scenario"])
    assert r["geometry_gate"]["ok"] is False and r["geometry_gate"]["details"]["taper_tip_root"] < 0.6
    assert all(v == 0.0 for v in r["terms"].values())         # no credit of any kind
    assert calls == {"sim": 0, "build": 0}
    # gene box corners pass FD's real gate on every aircraft (documented: no in-range gate failure exists)
    monkeypatch.setattr(pb1, "geometry_gate_strips", orig)
    fb, fw = F.fd_modules()["fb"], F.fd_modules()["fw"]
    import itertools
    for ac in ("c172x", "T38", "737", "f16"):
        geom = fb.geometry_for(ac, F.roots(sim.Profile.from_dict(dict(pd, aircraft=ac)))[1])
        pw = fw.params_for(ac, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
        assert all(pb1.geometry_gate(pw, list(u)).ok for u in itertools.product([0.0, 1.0], repeat=6)), ac


# ----------------------------------------------------------------------------- pins
def test_check_pins_full_a1_b1_and_mismatch_raises():
    mv = {"rigid": "rigid:jsbsim1.3.1:e0a73fc9", "full_a1_b1": "full_a1_b1:flexv2b1:56ee798e"}
    cfg = {"pin_model_version": {"c172x": dict(mv)}}
    assert batch.check_pins(cfg, "c172x", ["rigid", "full_a1_b1"], mv) == mv
    with pytest.raises(SystemExit, match="pin mismatch for full_a1_b1"):
        batch.check_pins(cfg, "c172x", ["rigid", "full_a1_b1"], dict(mv, full_a1_b1="full_a1_b1:flexv2b1:00000000"))
    with pytest.raises(SystemExit, match="pin mismatch for full_a1_b1"):      # an A1 string is not a B1 pin
        batch.check_pins({"pin_model_version": {"c172x": {"full_a1_b1": "full_a1:flexv2a1:36fb4f5a"}}}, "c172x",
                         ["full_a1_b1"], {"full_a1_b1": mv["full_a1_b1"]})
    with pytest.raises(SystemExit, match="no entry"):
        batch.check_pins({"pin_model_version": {"c172x": {"full_a1": "full_a1:flexv2a1:36fb4f5a"}}}, "c172x",
                         ["full_a1_b1"], {"full_a1_b1": mv["full_a1_b1"]})


@need_fd
@need_b1
def test_run_refuses_mismatched_b1_pin_and_eval_pin(tmp_path):
    u = cfg_file("phase3b1_smoke.json")
    u.update(run_id="mm", runs_dir=str(tmp_path / "runs"), cache={"enabled": True, "path": str(tmp_path / "c.sqlite")},
             scenarios=1, ga={"pop_size": 4, "generations": 1}, trajectories={"generations": [], "scenario": 0, "sample_hz": 30},
             aircraft=[{"name": "c172x", "profile": "phase2_c172x", "overrides": dict(SHORT)}],
             pin_model_version={"c172x": {"full_a1_b1": "full_a1_b1:flexv2b1:00000000"}})
    b = batch.Batch(batch.resolve_config(u, "t"), log=lambda *a: None)
    b.workers = 1
    with pytest.raises(SystemExit, match="pin mismatch"):
        b.run()
    p = tmp_path / "c.sqlite"
    assert not p.exists() or sqlite3.connect(str(p)).execute("SELECT COUNT(*) FROM evals").fetchone()[0] == 0
    assert not (tmp_path / "runs" / "mm" / "run.json").exists()


# ----------------------------------------------------------------------------- genome kind / operators
@need_fd
@need_b1
def test_phase2_flex_decode_unchanged_and_phase3_b1_schema():
    s2, s3 = resolved("phase3a1_smoke.json"), resolved("phase3b1_smoke.json")
    for ac2, ac3 in zip(s2["aircraft"], s3["aircraft"]):
        prof = sim.Profile.from_dict(ac2["resolved_profile"])
        sch2, g2 = batch.full_schema(prof, True, False)
        sch2b, g2b = batch.full_schema(prof, True, False, None)
        sch3, g3 = batch.full_schema(prof, True, False, "phase3_b1")
        assert (sch2, g2) == (sch2b, g2b) and len(sch2) == 20 and "shape" not in g2
        assert sch3[:20] == sch2 and g3 == g2 + ["shape"] * 6 and [g.name for g in sch3[20:]] == list(
            F.fd_b1_modules()["pb1"].SHAPE_NAMES)
        u = np.random.default_rng(5).random(26)
        from evolution import genome as gm
        assert {k: v for k, v in gm.decode(u, sch3).items() if k not in F.fd_b1_modules()["pb1"].SHAPE_NAMES} == \
            gm.decode(u[:20], sch2)
        assert prof.gain_bounds["ki_alt"][1] == 0.5 and prof.heading_hold
    assert batch.identity(s2).get("genome_kind") is None and "shape_ops" not in batch.identity(s2)
    assert s3["shape_ops"] == dict(batch.SHAPE_OPS_DEFAULT, mutation_rate=s3["ga"]["mutation_rate"])
    with pytest.raises(ValueError, match="only 'rigid' screens"):
        resolved("phase3b1_pilot.json", multi_fidelity=dict(cfg_file("phase3b1_pilot.json")["multi_fidelity"],
                                                            screen=["rigid", "full_a1"]))
    with pytest.raises(ValueError, match="fidelity must be 'full_a1_b1'"):
        resolved("phase3b1_smoke.json", fidelity="full_a1", pin_model_version={})
    with pytest.raises(ValueError, match="struct_asymmetric false"):
        resolved("phase3b1_smoke.json", struct_asymmetric=True)
    with pytest.raises(ValueError, match="genome_kind"):
        resolved("phase3a1_smoke.json", genome_kind="phase3_b2")


def _spec():
    sch = F.shape_schema()
    return ga.ShapeSpec(idx=list(range(20, 26)), lo=[g.min for g in sch], hi=[g.max for g in sch],
                        log=[g.name in CHORD for g in sch], default=[g.default for g in sch], mutation_rate=1.0)


@need_fd
@need_b1
def test_shape_mutation_is_log_space_for_chord_genes_and_in_bounds():
    """Genome Architect's spec: chord tapers mutate ln(x) (clipped to [ln .85, ln 1.05]) with sigma = 0.25 x half-range
    of ln(x); twist / sweep linear with sigma = 0.25 x half-range. Log-space steps are scale-free: the spread of ln(x)
    is the same from x = 1.0 and from x = 0.88 (a linear Gaussian would give sigma_lin / x, 14 % larger at 0.88)."""
    sp = _spec()
    sig = sp.sigma(0.25)
    assert np.allclose(sig[:3], 0.25 * 0.5 * math.log(1.05 / 0.85)) and np.allclose(sig[3:], [0.375, 0.625, 1.25])
    cfg = ga.GAConfig(mutation_rate=0.0)        # controller / structure untouched here
    rng = np.random.default_rng(11)
    N = 20000
    for x0 in (1.0, 0.88):
        g0 = np.full(26, 0.5)
        g0[20:23] = (x0 - 0.85) / 0.2
        g0[23:] = sp.identity_u()[3:]
        M = np.array([ga.mutate_blocks(rng, g0, cfg, sp) for _ in range(N)])
        assert np.all(M >= 0.0) and np.all(M <= 1.0) and np.all(M[:, :20] == 0.5)
        x = 0.85 + M[:, 20:23] * 0.2
        assert np.all(x >= 0.85) and np.all(x <= 1.05 + 1e-12)
        lx = np.log(x)
        for j in range(3):
            d = lx[:, j] - math.log(x0)
            # Gaussian in LOG space: median step 0, 68.3 % within +-sigma_log, symmetric halves. A linear Gaussian with
            # the same nominal width would put 64.8 % (x0 = 0.88) / 70.9 % (x0 = 1.0) inside +-sigma_log.
            assert abs(np.median(d)) < 0.001, (x0, j, np.median(d))
            inside = np.abs(d) < sig[j]
            assert abs(inside.mean() - 0.6827) < 0.012, (x0, j, inside.mean())
            assert abs(np.mean((d > 0) & inside) - np.mean((d < 0) & inside)) < 0.012
        tw = -4.0 + M[:, 24] * 5.0                 # linear gene: +-1 sigma (0.625 deg) holds 68.3 %
        assert abs(np.mean(np.abs(tw) < 0.625) - 0.6827) < 0.02 and tw.min() >= -4.0 and tw.max() <= 1.0
    # gen-0: identity + clipped Gaussian; bounds respected; chord median at ln(1) = 0
    Z = ga.shape_generation_zero(np.random.default_rng(1), 5000, sp)
    xc = 0.85 + Z[:, :3] * 0.2
    assert np.all((Z >= 0) & (Z <= 1)) and abs(np.median(np.log(xc))) < 0.002
    assert abs(np.median(-5.0 + Z[:, 5] * 10.0)) < 0.05


def test_block_crossover_swaps_whole_blocks():
    rng = np.random.default_rng(2)
    a, b = np.zeros(26), np.ones(26)
    blocks = batch.gene_blocks(["gains"] * 8 + ["struct"] * 12 + ["shape"] * 6)
    assert blocks == [list(range(8)), list(range(8, 20)), list(range(20, 26))]
    seen = set()
    for _ in range(200):
        c = ga.crossover_blocks(rng, a, b, blocks)
        pat = tuple(int(c[blk].min()) for blk in blocks)
        assert all(c[blk].min() == c[blk].max() for blk in blocks)
        seen.add(pat)
    assert len(seen) == 8


@need_fd
@need_b1
def test_gen0_controller_and_structure_match_phase3a1_seeding():
    """phase3_b1 gen-0: the 20 controller + structure genes are drawn exactly as the A1 smoke (same seed), then the shape
    block around identity; first 20 columns of the B1 population == the A1 population."""
    s3 = resolved("phase3b1_smoke.json")
    ac = s3["aircraft"][0]
    sch, groups = batch.full_schema(sim.Profile.from_dict(ac["resolved_profile"]), True, False, "phase3_b1")
    rng = np.random.default_rng(ac["seed"])
    pa1 = batch.seed_generation_zero(ga.generation_zero(rng, 16, 20), rng, sch[:20], groups[:20], s3["init"])
    sp = batch.shape_spec(sch, groups, s3["shape_ops"])
    assert sp.idx == list(range(20, 26)) and sp.log == [True, True, True, False, False, False]
    rng2 = np.random.default_rng(ac["seed"])
    p = batch.seed_generation_zero(ga.generation_zero(rng2, 16, 20), rng2, sch[:20], groups[:20], s3["init"])
    p = np.hstack([p, ga.shape_generation_zero(rng2, 16, sp)])
    assert np.array_equal(p[:, :20], pa1) and p.shape == (16, 26)


# ----------------------------------------------------------------------------- telemetry / trajectory header
@need_fd
@need_b1
def test_planform_header_flex_state_and_traj2_validates():
    pd = profile_d("T38", **SHORT)
    sc = _sc(pd)[0]
    states = []
    F.evaluate_genome(pd, G8, None, [sc], "full_a1_b1", 1.0, shape=SHAPED,
                      recorder=lambda t, fdm, fs: states.append(fs) if not states else None)
    fs = states[0]
    obj = fs.fd_model
    assert type(obj).__name__ == "FlexBodyModelB1" and not obj.planform_baseline and fs.fidelity == "full_a1_b1"
    pf = fs.planform
    assert pf["schema"] == "fd-planform/1" and pf["source"] == "P3-B1" and pf["symmetric"] is True
    assert pf["genes"] == F.shape_from(SHAPED) and pf["n_strips"] == 64
    w = pf["wing"]
    assert {len(v) for v in w.values()} == {64}
    assert w["chord_m"] == [float(c * sim.FT) for c in obj.planform.c_ft]          # FD's values, units only
    assert w["le_x_m"] == [float(-x * sim.FT) for x in obj.planform.le_x_ft]
    assert w["twist_rad"] == [float(t) for t in obj.planform.twist_rad] and w["twist_rad"][-1] < 0   # washout, LE up +
    assert pf["sweep_qc_rad"] == math.radians(obj.planform.sweep_qc_deg) == math.radians(24.0 + 5.0)
    # r1: structure.axis_nodes_body_m (FE wings) = FD node_layout_b1 of the shaped model (+ per-node fields, SI)
    comps = {c["name"]: c for c in fs.structure["components"]}
    assert len(comps["wingR"]["axis_nodes_body_m"]) == 65 and fs.structure["node_layout"].startswith("FD flexbody_b1")
    lay = {c["name"]: c for c in F.fd_b1_modules()["fbb1"].node_layout_b1(obj, fs._rp_offset_ft)}
    assert fs.node_layout() == list(lay.values())
    for nm in ("wingR", "wingL"):
        c, l = comps[nm], lay[nm]
        assert c["axis_nodes_body_m"] == [[round(v * sim.FT, 6) for v in p] for p in l["axis_nodes_body_ft"]]
        assert c["chord_m"] == [round(v * sim.FT, 6) for v in l["chord_ft"]] and len(c["chord_m"]) == 65
        assert c["geometric_twist_rad"] == [round(math.radians(v), 9) for v in l["geometric_twist_deg"]]
        assert c["geometric_twist_rad"][-1] < 0 and c["geometric_twist_rad"][0] == 0.0     # washout, + = LE up
        assert len(c["le_nodes_body_m"]) == len(c["te_nodes_body_m"]) == 65
        assert all(le[0] > te[0] for le, te in zip(c["le_nodes_body_m"], c["te_nodes_body_m"]))   # LE forward (+x)
    assert F.v2_map_mod().validate_structure(fs.structure, fs.channels()) == []
    assert fs.as_dict()["planform"] is pf
    # export path: SB flight bit-identical to FD's, header 'planform' once per file, traj/2 valid
    r = ev.task(pd, G8, None, sc, "full_a1_b1", True, 30.0, None, "sb", shape=SHAPED)
    assert r["telemetry_check"]["sim_cost_bit_identical"] is True and r["planform_header"]["genes"] == pf["genes"]
    doc = trajectory.build_doc(run_id="t", aircraft="T38", jsbsim_version="x", git_sha="x", seed=1, generation=0,
                               fitness=r["cost"], gains=dict(G8, **SHAPED), scenario=sc, scenario_index=0, sim_result=r,
                               profile=pd, extra={"fidelity": r["fidelity"], "model_version": r["model_version"]})
    assert doc["schema"] == "ga-flightsim-traj/2" and doc["planform"]["schema"] == "fd-planform/1"
    assert validate_traj.validate_doc(doc, "b1") == []
    assert not any("planform" in c for c in doc["channels"])        # header only, not per frame
    bad = json.loads(json.dumps(doc))
    bad["planform"]["wing"]["chord_m"] = bad["planform"]["wing"]["chord_m"][:-1]
    assert any("length" in e for e in validate_traj.validate_doc(bad, "b1"))
    bad2 = json.loads(json.dumps(doc))
    bad2["planform"]["schema"] = "x"
    assert validate_traj.validate_doc(bad2, "b1")
    nopf = {k: v for k, v in doc.items() if k != "planform"}
    assert validate_traj.validate_doc(nopf, "b1") == []
    # other fidelities never carry the field
    r1 = ev.task(pd, G8, None, sc, "full_a1", True, 30.0, None, "sb")
    assert "planform_header" not in r1


@need_fd
@need_b1
def test_r1_node_layout_baseline_equals_flexbody_and_shaped_ea_moves_with_sweep_and_chord():
    fb, fbb1 = F.fd_modules()["fb"], F.fd_b1_modules()["fbb1"]
    rp = [1.5, 0.0, -0.4]
    for ac in ("c172x", "T38", "737"):
        pd = profile_d("c172x", aircraft=ac)
        base, a1 = F.make_fd_model(pd, None, "full_a1_b1"), F.make_fd_model(pd, None, "full_a1")
        assert base.planform_baseline
        lb = F.fd_node_layout(base, "full_a1_b1", rp)
        assert lb == fb.node_layout(base, rp) == fb.node_layout(a1, rp) == F.fd_node_layout(a1, "full_a1", rp), ac
        assert not any("chord_ft" in c for c in lb) and F.b1_node_fields(lb) == {}

        def wx(shape):
            m = F.make_fd_model(pd, None, "full_a1_b1", shape=shape)
            assert m.node_layout(rp) == F.fd_node_layout(m, "full_a1_b1", rp)
            return {c["name"]: c for c in F.fd_node_layout(m, "full_a1_b1", rp)}
        b = {c["name"]: c for c in lb}
        x0 = np.array(b["wingR"]["axis_nodes_body_ft"])
        sw = np.array(wx({"wing_sweep_qc_delta_deg": 5.0})["wingR"]["axis_nodes_body_ft"])
        ch = wx({"wing_chord_taper_1": 0.85, "wing_chord_taper_2": 0.85, "wing_chord_taper_3": 0.85})
        tw = wx({"wing_twist_mid_deg": -2.0, "wing_twist_tip_deg": -4.0})
        xc, xt = np.array(ch["wingR"]["axis_nodes_body_ft"]), np.array(tw["wingR"]["axis_nodes_body_ft"])
        assert np.array_equal(sw[:, 1:], x0[:, 1:]) and np.array_equal(xc[:, 1:], x0[:, 1:])   # y, z unchanged
        assert sw[-1, 0] < x0[-1, 0] - 0.1, ac            # more sweep -> tip EA node aft (FRD x fwd)
        assert np.max(np.abs(xc[:, 0] - x0[:, 0])) > 1e-3, ac          # EA follows the shaped chord
        bc = np.array(ch["wingR"]["chord_ft"])
        assert bc[-1] < bc[0] and np.all(np.isfinite(bc))
        # twist rotates sections about the EA: EA x / y identical to the chord-only effect of twist (none)
        assert np.allclose(xt[:, :2], x0[:, :2], atol=1e-6) and np.array(tw["wingR"]["geometric_twist_deg"])[-1] < -3.5
        tl = np.array(tw["wingR"]["le_nodes_body_ft"])
        assert tl[-1, 2] > xt[-1, 2]                   # washout: LE down (body z + down) at the tip
        f = F.b1_node_fields(list(ch.values()))
        assert set(f) == {"wingR", "wingL"} and len(f["wingR"]["chord_m"]) == 65


def test_cache_key_r0_vs_r1_model_version_disjoint():
    g = np.linspace(0, 1, 26)
    k = lambda mv, sk: cache_mod.eval_key("c172x", g, {"a": 1}, {"s": 1}, 1, "1.3.1", "code", "m",   # noqa: E731
                                          fidelity="full_a1_b1", model_version=mv, shape_key=sk)
    for sk in ("baseline", "shaped-abc"):
        assert k("full_a1_b1:flexv2b1:3e40908a", sk) != k("full_a1_b1:flexv2b1:56ee798e", sk)


# ----------------------------------------------------------------------------- configs / tiny end-to-end
@need_fd
@need_b1
def test_p3b1_configs_match_a1_and_pins():
    b1, p25 = json.load(open(B1_PINS_FILE)), json.load(open(P25_PINS_FILE))
    for new, old in (("phase3b1_smoke.json", "phase3a1_smoke.json"), ("phase3b1_pilot.json", "phase3a1_pilot.json")):
        n, o = resolved(new), resolved(old)
        assert n["fidelity"] == "full_a1_b1" and n["genome_kind"] == "phase3_b1" and o["fidelity"] == "full_a1"
        strip = lambda c: {k: v for k, v in batch.identity(c).items()   # noqa: E731
                           if k not in ("fidelity", "pin_model_version", "multi_fidelity", "genome_kind", "shape_ops")}
        assert strip(n) == strip(o), new
        mfn, mfo = dict(n["multi_fidelity"]), dict(o["multi_fidelity"])
        if mfo.get("enabled"):
            assert mfn.pop("ladder") == ["rigid", "full_a1_b1"] and mfo.pop("ladder") == ["rigid", "full_a1"]
        assert mfn == mfo
        for ac, pins in n["pin_model_version"].items():
            assert pins["full_a1_b1"] == b1[ac]["full_a1_b1"]
            if "rigid" in pins:
                assert pins["rigid"] == p25[ac]["rigid"]
        for ac, d in batch.model_versions_report(n).items():
            assert all(x["match"] is True for f, x in d.items() if f != "error"), (ac, d)
    r0 = json.load(open(B1_R0_PINS_FILE))
    for new in ("phase3b1_smoke", "phase3b1_pilot"):      # r0 copies kept: identical except the B1 pins
        a, b = cfg_file(new + ".json"), cfg_file(new + "_r0.json")
        assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"_comment", "_pin_source", "pin_model_version"}
        assert all(p["full_a1_b1"] == r0[ac]["full_a1_b1"] for ac, p in b["pin_model_version"].items())
        assert {ac: {f: v for f, v in p.items() if f != "full_a1_b1"} for ac, p in a["pin_model_version"].items()} == \
            {ac: {f: v for f, v in p.items() if f != "full_a1_b1"} for ac, p in b["pin_model_version"].items()}
    s = resolved("phase3b1_smoke.json")
    assert (s["ga"]["pop_size"], s["ga"]["generations"], s["init"]["sigma"]) == (16, 5, 0.10)
    assert {a["name"]: a["seed"] for a in resolved("phase3b1_pilot.json")["aircraft"]} == {"c172x": 1, "T38": 2, "737": 3}


@need_fd
@need_b1
def test_tiny_phase3_b1_batch_end_to_end(tmp_path):
    u = cfg_file("phase3b1_smoke.json")
    u.update(run_id="b1tiny", runs_dir=str(tmp_path / "runs"), cache={"enabled": True, "path": str(tmp_path / "c.sqlite")},
             scenarios=1, ga={"pop_size": 4, "generations": 2},
             trajectories={"generations": [0, 1], "scenario": 0, "sample_hz": 30},
             aircraft=[{"name": "c172x", "profile": "phase2_c172x", "overrides": dict(SHORT)}],
             pin_model_version={"c172x": {"full_a1_b1": json.load(open(B1_PINS_FILE))["c172x"]["full_a1_b1"]}})
    b = batch.Batch(batch.resolve_config(u, "t"), log=lambda *a: None)
    b.workers = 2
    summ = b.run()
    assert "error" not in summ["aircraft"][0], summ["aircraft"][0].get("traceback")
    rows = [json.loads(x) for x in open(os.path.join(b.run_dir, "genomes.jsonl"))]
    assert len(rows) == 8 and all(len(r["shape"]) == 6 and len(r["genome_norm"]) == 26 for r in rows)
    assert all(r["fidelity"] == "full_a1_b1" and r["model_version"].startswith("full_a1_b1:") for r in rows)
    run = json.load(open(os.path.join(b.run_dir, "run.json")))
    assert run["genome_kind"] == "phase3_b1" and sum(g["group"] == "shape" for g in run["aircraft"][0]["genes"]) == 6
    # re-fly a row through eval.evaluate (decoded genome incl. shape) == its stored cost
    r0 = next(r for r in rows if r["generation"] == 1 and r["rank"] == 0)
    out = ev.evaluate(r0["genome"], "c172x", None, run)
    assert out["cost"] == r0["cost"] and out["shape_genes"] == r0["shape"]
    tdir = os.path.join(b.run_dir, "trajectories")
    assert validate_traj.main([tdir]) == 0
    for fn in os.listdir(tdir):
        if fn.startswith("traj_"):
            d = json.load(open(os.path.join(tdir, fn)))
            assert d["planform"]["schema"] == "fd-planform/1"
    hist = [json.loads(x) for x in open(os.path.join(b.run_dir, "history.jsonl"))]
    assert all("geometry_gate_rejects" in h for h in hist)
