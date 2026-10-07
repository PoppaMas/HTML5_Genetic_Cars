"""Phase 2: phase2_flex preset on FD flex v2 (schema parity, errors, seeding, constraints, bit-identity of v1 tasks)."""
import json
import os

import numpy as np
import pytest

import adapter
import fd_bridge
import fitness as F
import genome_schema as GS
import init_pop
import flightsim_path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _have_fd_v2():
    try:
        fd_bridge.flexbody()
        fd_bridge.require_root_v2("c172x")
        return True
    except Exception:
        return False


# Phase 2 needs FD's flex v2 (flexbody.py + jsbsim_root_v2); skip cleanly where it isn't present (e.g. the Phase 1 repo)
pytestmark = pytest.mark.skipif(not _have_fd_v2(), reason="FD flex v2 (flexbody.py, jsbsim_root_v2) not present")
V2 = GS.STRUCTURE_V2_BLOCK
CTRL = ["kp_alt", "ki_alt", "kd_alt", "kp_pitch", "ki_pitch", "kd_pitch", "kp_hdg", "ki_hdg"]


def _v2(t):
    return [g for g in t.spec.genes if g.block == V2]


# ----------------------------------------------------------------------------------------------- schema parity
@pytest.mark.parametrize("asym", [False, True])
def test_v2_block_tracks_fd_gene_schema(asym):
    t = adapter.load_task("phase2_flex", {"flex": {"asymmetric": asym}})
    fd = fd_bridge.flexbody().gene_schema(asymmetric=asym)
    mine = _v2(t)
    assert t.spec.names[:8] == CTRL                       # phase1_v4 controller genes first, unchanged order
    assert [g.name for g in mine] == [g.name for g in fd]  # FD's order
    assert len(mine) == (14 if asym else 12) and t.spec.n_genes == 8 + len(mine)
    for g, f in zip(mine, fd):
        assert (g.min, g.max, g.scale, g.default) == (f.lo, f.hi, f.scale, f.default)
    rng = np.random.default_rng(3)
    for u in rng.random((20, len(mine))):
        for g, f, x in zip(mine, fd, u):
            assert g.decode(x) == pytest.approx(f.decode(x), rel=1e-12, abs=1e-15)
            assert g.encode(f.decode(x)) == pytest.approx(x, abs=1e-9)
    # every decoded genome is accepted by FD's own validator
    gains = t.spec.decode(rng.random(t.spec.n_genes))
    fd_bridge.struct_genes_v2(gains, asym)
    assert t.conditions["flex_mode"] == "v2" and t.conditions["flex_asymmetric"] is asym


def test_v2_defaults_and_controller_bounds_equal_phase1_v4():
    t, v4 = adapter.load_task("phase2_flex"), adapter.load_task("phase1_v4")
    for a in ("c172x", "t38", "b737"):
        t, v4 = adapter.load_task("phase2_flex", {"aircraft": a}), adapter.load_task("phase1_v4", {"aircraft": a})
        exp = [(g.name, g.min, 0.5 if g.name == "ki_alt" else g.max, g.scale) for g in v4.spec.genes]  # phase2-only ki_alt <= 0.5
        assert exp == [(g.name, g.min, g.max, g.scale) for g in t.spec.genes[:8]]
        assert {k: v for k, v in t.conditions.items() if not k.startswith("flex_")} == \
               {k: v for k, v in v4.conditions.items() if not k.startswith("flex_")}
    base = fd_bridge.flexbody().baseline_genes()
    assert {g.name: g.default for g in _v2(t)} == {k: v for k, v in base.items() if k in {g.name for g in _v2(t)}}


# ----------------------------------------------------------------------------------------------- config errors
@pytest.mark.parametrize("over,msg", [
    ({"blocks": {"pitch_altitude": True, "structure": True, "structure_v2": True}}, "cannot be evolved together"),
    ({"gene_overrides": {"wing_ei_root": {"min": 0.5, "max": 3.0}}}, "come from FD's gene_schema"),
    ({"flex": {"version": 1, "asymmetric": False}}, "needs flex v2"),
    ({"flex": {"version": 3}}, "flex.version"),
    ({"flex": {"x_ea": 0.4}}, "fixed per aircraft"),
    ({"flex": {"bogus": 1}}, "unknown flex v2 keys"),
    ({"flex": {"mode": "oneway"}}, "two-way"),
    ({"init": {"mode": "lhs"}}, "init.mode"),
    ({"init": {"sigma": 0.9}}, "init.sigma"),
    ({"init": {"blocks": ["structure"]}}, "not evolved"),
    ({"init": {"bogus": 1}}, "unknown init keys"),
])
def test_config_errors(over, msg):
    with pytest.raises(ValueError, match=msg):
        adapter.load_task("phase2_flex", over)


@pytest.mark.parametrize("genes,asym", [
    ({"wing_ei_root": 2.5}, False),            # out of range
    ({"wing_ei_root": float("nan")}, False),   # NaN
    ({"wing_flutter_magic": 1.0}, False),      # unknown v2-looking key
    ({"wing_asym_ei_delta": 0.05}, False),     # asymmetric gene without the flag
    ({"stiffness_scale": 1.0}, False),         # v1 gene in a v2 genome
])
def test_fd_decoder_rejects_bad_structure_genes(genes, asym):
    fb = fd_bridge.flexbody()
    with pytest.raises(ValueError):
        fb.decode_genome_v2(genes, asymmetric=asym)
    if set(genes) <= set(fd_bridge.gene_names_v2(asym)):   # the bridge passes the task's v2 genes to FD's decoder
        with pytest.raises(ValueError):
            fd_bridge.struct_genes_v2(genes, asym)


# ----------------------------------------------------------------------------------------------- seeding
def test_baseline_seeding_keeps_controller_draw_and_ranges():
    t = adapter.load_task("phase2_flex")
    n, pop = t.spec.n_genes, 400
    seeded = init_pop.generation_zero(np.random.default_rng(7), pop, n, t.spec, t.init)
    uni = np.random.default_rng(7).random((pop, n))                       # == ga.generation_zero
    idx, base = init_pop.baseline_u(t.spec, [V2])
    ctrl = [j for j in range(n) if j not in idx]
    assert np.array_equal(seeded[:, ctrl], uni[:, ctrl])                 # controller genes as the current presets
    assert seeded.min() >= 0 and seeded.max() <= 1
    d = seeded[:, idx] - base
    assert t.init["sigma"] == 0.10                                          # = ER phase2_pilot init.sigma
    # P2.5: nsm genes sit at their floor (baseline u=0); N(0,σ) clipped to [0,1] biases the mean upward (~0.04 at σ 0.10).
    nsm_i = [j for j, g in enumerate(_v2(t)) if "nsm" in g.name]
    other = [j for j in range(len(idx)) if j not in nsm_i]
    assert np.all(np.abs(np.mean(d[:, other], 0)) < 0.03)
    assert np.all(np.mean(d[:, nsm_i], 0) > 0) and np.all(np.mean(d[:, nsm_i], 0) < 0.08)
    assert np.all(np.abs(np.std(d[:, other], 0) - 0.10) < 0.02)
    assert np.all(np.std(d[:, nsm_i], 0) < 0.08)          # floor clipping truncates the lower tail
    assert np.array_equal(seeded, init_pop.generation_zero(np.random.default_rng(7), pop, n, t.spec, t.init))
    # ranges stay the full FD ranges (seeding is init only)
    assert [(g.min, g.max) for g in _v2(t)] == [(g.lo, g.hi) for g in fd_bridge.flexbody().gene_schema()]
    # sigma 0 -> exactly the baseline
    z = init_pop.generation_zero(np.random.default_rng(7), 5, n, t.spec, {**t.init, "sigma": 0.0})
    dec = t.spec.decode(z[0])
    assert all(dec[g.name] == pytest.approx(g.default, rel=1e-12) for g in _v2(t))


def test_uniform_init_and_tasks_without_init_are_untouched():
    ga = flightsim_path.orig_ga()
    t = adapter.load_task("phase2_flex", {"init": {"mode": "uniform"}})
    a = init_pop.generation_zero(np.random.default_rng(1), 24, t.spec.n_genes, t.spec, t.init)
    assert np.array_equal(a, ga.generation_zero(np.random.default_rng(1), 24, t.spec.n_genes))
    for name in ("altitude_hold_legacy", "phase1_v4", "phase1_flex", "phase1_v5"):
        assert adapter.load_task(name).init == {}                         # run_evolve leaves evolve.ga alone
    proxy = init_pop.seeded_ga(ga, t)
    for k, v in vars(ga).items():
        if not k.startswith("__") and k != "generation_zero":
            assert getattr(proxy, k) is v
    assert np.array_equal(proxy.generation_zero(np.random.default_rng(1), 24, t.spec.n_genes), a)


def test_seed_runs_put_best_genomes_first():
    run = os.path.join("runs", "hdg_after_c172x_s1")
    if not os.path.exists(os.path.join(HERE, run, "best_gains.json")):
        pytest.skip("v4 run not present")
    t = adapter.load_task("phase2_flex", {"init": {"seed_runs": {"c172x": [run]}}})
    assert t.init["seed_runs"] == [run]
    pop = init_pop.generation_zero(np.random.default_rng(1), 6, t.spec.n_genes, t.spec, t.init)
    g = t.spec.decode(pop[0])
    best = json.load(open(os.path.join(HERE, run, "best_gains.json")))["gains"]
    assert all(g[k] == pytest.approx(v, rel=1e-9, abs=1e-12) for k, v in best.items())
    assert all(g[x.name] == pytest.approx(x.default, rel=1e-12) for x in _v2(t))
    assert adapter.load_task("phase2_flex", {"aircraft": "t38", "init": {"seed_runs": {"c172x": [run]}}}).init["seed_runs"] == []


# ----------------------------------------------------------------------------------------------- constraints
def _blk(**kw):
    b = {"flutter_margin": 3.0, "flutter_margin_qs": 3.0, "coalescence_margin": 3.0, "div_margin": 3.0,
         "elevator_reversal_margin": 3.0, "margin_error": False}
    b.update(kw)
    return b


def test_conservative_min_over_bodies_and_methods():
    m = {"margin_cap": 3.0, "blocks": {"wingR": _blk(flutter_margin=1.3, coalescence_margin=1.25, flutter_margin_qs=1.3),
                                       "wingL": _blk(flutter_margin_qs=1.21, flutter_margin=1.21, div_margin=1.5),
                                       "empennage_pitch": _blk(elevator_reversal_margin=1.1),
                                       "empennage_yaw": _blk(div_margin_not_found_below_cap=True)}}
    c = fd_bridge.conservative_margins_v2(m)
    assert c["flutter_margin"] == 1.21 and c["flutter_binding"] == "wingL.flutter_margin"
    assert c["divergence_margin"] == 1.5 and c["reversal_margin"] == 1.1
    assert c["min_margin"] == 1.1 and c["binding_kind"] == "reversal"
    assert c["flags"]["empennage_yaw.div_margin_not_found_below_cap"] is True
    # not-found / huge / inf: never above the cap; inf or NaN or margin_error -> hard fail (0)
    assert fd_bridge.conservative_margins_v2({"margin_cap": 3.0, "blocks": {"w": _blk(div_margin=50.0)}})["divergence_margin"] == 3.0
    for bad in (float("inf"), float("nan")):
        c = fd_bridge.conservative_margins_v2({"margin_cap": 3.0, "blocks": {"w": _blk(div_margin=bad)}})
        assert c["margin_error"] and c["min_margin"] == 0.0
    c = fd_bridge.conservative_margins_v2({"margin_cap": 3.0, "blocks": {"w": _blk(margin_error=True)}})
    assert c["margin_error"] and c["min_margin"] == 0.0


@pytest.mark.parametrize("model,jf", [("c172x", 0.0), ("T38", 0.03608), ("737", 0.01317)])
def test_baseline_precheck_matches_fd_table(model, jf):
    p = fd_bridge.precheck_v2(model, {})
    assert p["fail"] is None and p["J_mass"] == 0.0
    assert p["terms"]["J_flutter_margin"] == pytest.approx(jf, abs=2e-5)
    fdm = p["fd_margins"]
    for kind, key in (("flutter", "flutter_margin"), ("divergence", "div_margin"), ("reversal", "reversal_margin")):
        assert p["margins"][f"{kind}_margin"] <= fdm[key] + 1e-12       # never less conservative than FD's overall
        assert p["margins"][f"{kind}_margin"] <= fdm["margin_cap"]


def test_gate_and_penalty_band():
    w = fd_bridge.struct_weights_v2(None)
    soft = fd_bridge.precheck_v2("c172x", {"wing_ei_root": 0.6})       # FD: min margin ~0.96 < 1.0
    assert soft["fail"] == "flutter" and soft["margins"]["min_margin"] < 1.0
    mid = fd_bridge.precheck_v2("c172x", {"wing_ei_taper_1": 0.75})    # ~1.09: flown, penalised
    m = mid["margins"]["flutter_margin"]
    assert mid["fail"] is None and 1.0 < m < 1.2
    assert mid["terms"]["J_flutter_margin"] == pytest.approx(w.w_flutter * ((w.margin_req - m) / (w.margin_req - 1)) ** 2)
    assert w.margin_req == 1.2


def test_hard_fail_is_not_flown_and_costs_fail_cost():
    t = adapter.load_task("phase2_flex")
    gains = dict(t.spec.decode(t.spec.default_genome()), wing_ei_root=0.6)
    r = t.evaluate(gains, t.make_scenarios(1, 1))
    assert r["cost"] == F.FAIL_COST and r["per_scenario"][0]["status"] == "aeroelastic_flutter"
    assert r["per_scenario"][0].get("t_end", 0.0) == 0.0


def test_structural_v2_terms_normalised_and_mass_counted():
    loads = {k: {"peak_abs": 10.0, "rms_dev_1g": 1.0} for ks in F.V2_LOAD_BODIES.values() for k in ks}
    loads["wingR_bm"] = {"peak_abs": 50.0, "rms_dev_1g": 5.0}            # worse side counts
    loads["wingL_bm"] = {"peak_abs": 40.0, "rms_dev_1g": 4.0}
    res = {"flex_v2": {"loads": loads, "allowables": {k: -100.0 for k in F.V2_ALLOW_KEY.values()}, "bm_allow": 100.0,
                       "terms": {k: 0.0 for k in F.V2_HINGE_TERMS}},
           "flex_v2_pre": {"J_mass": 0.03, "J_smooth": 0.0}}
    lr = F.load_ratios_v2(res["flex_v2"])
    assert lr["wing"] == pytest.approx({"peak": 0.5, "rms": 0.05}) and lr["vt"] == pytest.approx({"peak": 0.1, "rms": 0.01})
    terms = F.structural_v2_terms(res, dict(F.STRUCT_V2_DEFAULTS))
    assert terms["mass"] == 0.03                                         # stiffer (heavier) is not free
    assert terms["load_peak"] == pytest.approx(F.STRUCT_V2_DEFAULTS["struct_v2_w_peak"] * (0.5 + 4 * 0.1) / 5)
    assert terms["load_rms"] == pytest.approx(F.STRUCT_V2_DEFAULTS["struct_v2_w_rms"] * (0.05 + 4 * 0.01) / 5)
    res["flex_v2_pre"]["J_mass"] = -0.01                                 # FD's J_mass is signed (lighter -> credit)
    assert F.structural_v2_terms(res, {})["mass"] == -0.01


# ----------------------------------------------------------------------------------------------- flight (v2)
@pytest.mark.sim
def test_baseline_structure_flies_with_v4_gains():
    f = os.path.join(HERE, "runs", "hdg_after_c172x_s1", "best_gains.json")
    if not os.path.exists(f):
        pytest.skip("v4 run not present")
    t = adapter.load_task("phase2_flex", {"fitness": {"params": {"struct_v2_source": "genome"}}})
    gains = {**t.spec.decode(t.spec.default_genome()), **json.load(open(f))["gains"]}
    r = t.evaluate(gains, t.make_scenarios(1, 1))
    assert r["per_scenario"][0]["status"] == "ok"
    sv = r["structural_v2_terms"]
    assert sv["mass"] == 0.0 and 0.0 < sv["load_peak"] < 0.1
    assert all(0.0 < v["peak_max"] < 1.0 for v in r["load_ratios_v2"].values())
    assert r["aeroelastic"]["min_margin"] == pytest.approx(1.2279, abs=1e-3)


# ----------------------------------------------------------------------------------------------- v1 contracts frozen
@pytest.mark.sim
def test_phase1_flex_v1_reproduces_flex12_run_bit_for_bit():
    f = os.path.join(HERE, "runs", "flex12_c172x_s1", "best_gains.json")
    if not os.path.exists(f):
        pytest.skip("flex12 run not present")
    g = json.load(open(f))
    t = adapter.load_task("phase1_flex")
    assert t.conditions["flex_mode"] == "twoway" and t.init == {}
    assert t.evaluate(g["gains"], t.make_scenarios(3, g["config"]["scenario_seed"]))["cost"] == g["best_cost"]


def test_phase2_ki_alt_upper_0p5_override_only_in_phase2():
    for ac in ("c172x", "t38", "b737"):
        k4 = next(g for g in adapter.load_task("phase1_v4", {"aircraft": ac}).spec.genes if g.name == "ki_alt")
        k5 = next(g for g in adapter.load_task("phase1_v5", {"aircraft": ac}).spec.genes if g.name == "ki_alt")
        k2 = adapter.load_task("phase2_flex", {"aircraft": ac}).spec.genes[1]
        assert k4.max == 0.05 and k2.max == 0.5 == k5.max                 # shared v4 bound untouched
        assert (k2.name, k2.min, k2.scale) == (k4.name, k4.min, k4.scale) == (k5.name, k5.min, k5.scale)


# ----------------------------------------------------------------------------------------------- margins: no reward
def test_v2_margins_penalised_only_below_1p2_never_rewarded_above():
    wts = fd_bridge.struct_weights_v2(None)
    assert wts.margin_req == 1.2

    def run(m, **flags):
        blk = _blk(flutter_margin=m, flutter_margin_qs=m, coalescence_margin=m, div_margin=3.0, elevator_reversal_margin=3.0)
        blk.update(flags)
        return fd_bridge.margin_penalties_v2(fd_bridge.conservative_margins_v2({"margin_cap": 3.0, "blocks": {"w": blk}}), wts)

    for m in (0.5, 0.999):
        assert run(m)[1] == "flutter"
    pen = [run(m)[0]["J_flutter_margin"] for m in (1.0, 1.05, 1.1, 1.15, 1.199)]
    assert run(1.0)[1] is None and pen[0] == pytest.approx(1.0) and all(a > b > 0 for a, b in zip(pen, pen[1:]))
    for m in (1.2, 1.3, 2.0, 2.99, 3.0, 50.0):                         # 50 -> capped at 3.0
        terms, fail = run(m)
        assert fail is None and set(terms.values()) == {0.0}
    terms, fail = run(3.0, div_margin_not_found_below_cap=True)          # not-found = the cap, no bonus
    assert fail is None and set(terms.values()) == {0.0}
    # end to end: two flyable c172x structures with all margins >= 1.2 get exactly the same (zero) margin cost
    a, b = fd_bridge.precheck_v2("c172x", {}), fd_bridge.precheck_v2("c172x", {"wing_ei_root": 2.0})
    assert a["margins"]["min_margin"] >= 1.2 and b["margins"]["min_margin"] > a["margins"]["min_margin"] + 0.3
    assert sum(a["terms"].values()) == sum(b["terms"].values()) == 0.0


# ----------------------------------------------------------------------------------------------- mass clip (A/B only)
def test_mass_credit_clip_off_by_default_code_path_kept():
    t = adapter.load_task("phase2_flex")
    assert "struct_v2_mass_credit_clip" not in t.fitness.params
    assert not any("mass_credit_clip" in w or "INTERIM" in w for w in t.warnings)
    w = fd_bridge.struct_weights_v2(None).w_mass
    clip = ("ht", "vt", "fus")
    soft = fd_bridge.precheck_v2("c172x", {"fuselage_stiffness_scale": 0.6, "tail_stiffness_scale": 0.6}, params={"struct_v2_mass_credit_clip": clip})
    assert soft["J_mass_fd"] < 0 and soft["J_mass"] == 0.0                 # decrease earns nothing
    stiff = fd_bridge.precheck_v2("c172x", {"fuselage_stiffness_scale": 2.0}, params={"struct_v2_mass_credit_clip": clip})
    assert stiff["J_mass"] == pytest.approx(stiff["J_mass_fd"]) and stiff["J_mass"] > 0.0   # increase still costs
    wing = fd_bridge.precheck_v2("c172x", {"wing_ei_root": 0.7, "fuselage_stiffness_scale": 0.6}, params={"struct_v2_mass_credit_clip": clip})
    m = wing["mass"]
    assert wing["J_mass"] == pytest.approx(w * (m["wingR_lb"] + m["wingL_lb"]) / m["baseline_flexible_lb"])  # wing kept
    assert wing["J_mass"] < 0 and m["fus_lb"] < 0
    for g in ({}, {"fuselage_stiffness_scale": 0.6}, {"wing_ei_root": 1.7, "tail_stiffness_scale": 0.8}):
        p = fd_bridge.precheck_v2("c172x", g)                              # off: FD's own J_mass exactly
        assert p["J_mass"] == pytest.approx(p["J_mass_fd"], rel=1e-12, abs=1e-15) and p["mass_credit_clip"] == []
    t2 = adapter.load_task("phase2_flex", {"fitness": {"params": {"struct_v2_source": "genome", "struct_v2_mass_credit_clip": list(clip)}}})
    assert any("A/B only" in x for x in t2.warnings)
    with pytest.raises(ValueError, match="unsupported bodies"):
        adapter.load_task("phase2_flex", {"fitness": {"params": {"struct_v2_source": "genome", "struct_v2_mass_credit_clip": ["wing"]}}})
    with pytest.raises(ValueError, match="needs flex v2"):
        adapter.load_task("phase1_v4", {"fitness": {"params": {"struct_v2_mass_credit_clip": ["fus"]}}})


# ----------------------------------------------------------------------------------------------- FD §12 terms
def test_sizing_terms_match_fd_and_weights_are_fds():
    fb = fd_bridge.flexbody()
    assert "J_wing_tip_bm_limit" in fb.SIZING_TERMS
    assert tuple(fb.SIZING_TERMS) == F.V2_SIZING_TERMS == fd_bridge.SIZING_TERMS_V2 == fd_bridge.SIZING_TERMS_V2_PIN
    assert set(fd_bridge.FLOWN_FULL_ONLY_TERMS_V2) <= set(F.V2_HINGE_TERMS)
    assert fd_bridge.struct_weights_v2({**F.FLEX_DEFAULTS}) == fb.StructWeightsV2()   # no reweighting
    assert fb.StructWeightsV2().w_wing_tip_bm_limit == 1.0
    for model in ("c172x", "T38", "737"):
        p = fd_bridge.precheck_v2(model, {}, params={**F.FLEX_DEFAULTS})
        assert set(p["sizing_terms"].values()) == {0.0}                     # 0 at the baseline (incl. tip)
        assert p["sizing_terms"]["J_wing_tip_bm_limit"] == 0.0
        g = {"wing_ei_root": 0.6, "tail_stiffness_scale": 0.6, "fuselage_stiffness_scale": 0.6}
        p = fd_bridge.precheck_v2(model, g, params={**F.FLEX_DEFAULTS})
        sz = fb.sizing_v2(fd_bridge.model_v2(model, fd_bridge.struct_genes_v2(g)), fb.StructWeightsV2())["terms"]
        assert p["sizing_terms"] == pytest.approx(sz, rel=1e-12)
        # P2.5: tip BM term ~0.49–0.51 at s=0.6 adds to the pre-P2.5 ~1.47 total
        assert sum(sz.values()) == pytest.approx(1.96, abs=0.04) and -0.04 < p["J_mass"] < -0.025
        assert sz["J_wing_tip_bm_limit"] == pytest.approx(0.50, abs=0.03)
    for g in ({"tail_stiffness_scale": 1.5, "fuselage_stiffness_scale": 1.5},
              {"fuselage_stiffness_scale": 2.0, "wing_ei_root": 1.3},
              {"wing_ei_root": 1.5, "tail_stiffness_scale": 1.5, "fuselage_stiffness_scale": 1.5}):
        assert set(fd_bridge.precheck_v2("c172x", g)["sizing_terms"].values()) == {0.0}   # 0 when all genes >= 1
    # Cross-body coupling (not in FD's "0 for genes >= 1" wording): a stiffer (heavier) tail raises the fuselage demand,
    # so with the fuselage gene at 1.0 J_fus_bm_limit is small but nonzero (c172x tail 1.5 -> fusV ratio 1.018).
    p = fd_bridge.precheck_v2("c172x", {"tail_stiffness_scale": 1.5})
    assert 0 < p["sizing_terms"]["J_fus_bm_limit"] < 1e-3 and p["sizing_ratios"]["fusV"] > 1.0


def test_structural_v2_includes_sizing_and_missing_full_only_terms_are_zero():
    loads = {k: {"peak_abs": 10.0, "rms_dev_1g": 1.0} for ks in F.V2_LOAD_BODIES.values() for k in ks}
    base_terms = {k: 0.0 for k in ("J_bm_peak", "J_tip", "J_twist", "J_tail_bm_peak", "J_fus_bm_peak")}  # no torque/ip keys
    res = {"flex_v2": {"loads": loads, "allowables": {k: 100.0 for k in F.V2_ALLOW_KEY.values()}, "bm_allow": 100.0,
                       "terms": base_terms},
           "flex_v2_pre": {"J_mass": 0.0, "J_smooth": 0.0, "sizing": {"J_tail_bm_limit": 0.44, "J_fus_bm_limit": 0.1}}}
    t = F.structural_v2_terms(res, {})
    assert t["J_wing_torque_peak"] == t["J_wing_ip_peak"] == 0.0           # missing at reduced -> 0, no error
    assert t["J_tail_bm_limit"] == 0.44 and t["J_fus_bm_limit"] == 0.1 and t["J_wing_bm_limit"] == 0.0
    res["flex_v2"]["terms"] = dict(base_terms, J_wing_torque_peak=0.02, J_wing_ip_peak=0.01)
    t2 = F.structural_v2_terms(res, {})
    assert sum(t2.values()) == pytest.approx(sum(t.values()) + 0.03)
    res["flex_v2_pre"].pop("sizing")                                       # older pre dict: sizing absent -> 0
    assert F.structural_v2_terms(res, {})["J_tail_bm_limit"] == 0.0


# ----------------------------------------------------------------------------------------------- FD model_version
def test_recorded_fd_model_versions_match_current():
    # Fails on purpose when FD changes code / data / genes / weights: re-check phase2_flex and update the record.
    for model in ("c172x", "T38", "737", "f16"):
        for fid in ("reduced", "full"):
            r = fd_bridge.check_model_version(model, fid, "warn")
            assert r["match"], r["message"]
    assert fd_bridge.V2_MARGIN_GATES == fd_bridge._import("flexeval").MARGIN_GATE
    t = adapter.load_task("phase2_flex")
    assert t.flex_constants["fidelity"] == "full" and t.flex_constants["margin_gate"] == F.V2_GATE == 1.0
    assert t.flex_constants["fd_model_version"] == fd_bridge.FD_V2_MODEL_VERSIONS["c172x"]["full"]


def test_model_version_mismatch_warns_or_raises(monkeypatch):
    # Unknown version: warn by default, raise when asked
    monkeypatch.setattr(fd_bridge, "fd_model_version", lambda model, fidelity="full": "full:flexv2:deadbeef")
    t = adapter.load_task("phase2_flex")
    assert any(w.startswith("FD MODEL_VERSION") and "deadbeef" in w and "unknown" in w.lower() or "deadbeef" in w for w in t.warnings)
    with pytest.raises(RuntimeError, match="deadbeef"):
        adapter.load_task("phase2_flex", {"flex": {"model_version_check": "raise"}})
    assert not any("MODEL_VERSION" in w for w in adapter.load_task("phase2_flex", {"flex": {"model_version_check": "off"}}).warnings)
    # Previous post-mass: warn only, never raise (ER pilot transition)
    prev = fd_bridge.FD_V2_MODEL_VERSIONS_PREV_POST_MASS["c172x"]["full"]
    monkeypatch.setattr(fd_bridge, "fd_model_version", lambda model, fidelity="full": prev)
    t2 = adapter.load_task("phase2_flex")
    assert any("previous post-mass" in w for w in t2.warnings)
    t3 = adapter.load_task("phase2_flex", {"flex": {"model_version_check": "raise"}})  # must NOT raise
    assert any("previous post-mass" in w for w in t3.warnings)
    with pytest.raises(ValueError, match="model_version_check"):
        adapter.load_task("phase2_flex", {"flex": {"model_version_check": "maybe"}})


def test_asymmetric_genes_off_by_default():
    t = adapter.load_task("phase2_flex")
    assert t.raw["flex"]["asymmetric"] is False and t.spec.n_genes == 20 and t.conditions["flex_asymmetric"] is False
    ta = adapter.load_task("experiments/phase2_flex_asym.json")
    assert ta.spec.n_genes == 22 and any("lateral/roll scenarios" in w for w in ta.warnings)


# ----------------------------------------------------------------------------------------------- struct_v2_source
def test_struct_v2_source_fd_default_in_phase2_and_validated():
    t = adapter.load_task("phase2_flex")
    assert t.fitness.params["struct_v2_source"] == "fd" and "struct_v2_mass_credit_clip" not in t.fitness.params
    assert t.fitness.weights == {"track_alt": 1.0, "effort": 2.0, "comfort": 0.05, "track_heading_rms": 0.01, "structural_v2": 1.0}
    v4 = adapter.load_task("phase1_v4")
    assert {k: v for k, v in t.fitness.weights.items() if k != "structural_v2"} == v4.fitness.weights
    assert t.raw["scenarios"] == v4.raw["scenarios"] and t.raw["scenarios"]["ramp_fpm"] == 600.0
    assert t.raw["scenarios"]["ramp_accel_g"] == 0.1 and t.raw["controller"]["alt_ref_ff"] is True
    assert {k: v for k, v in t.conditions.items() if not k.startswith("flex_")} == v4.conditions
    with pytest.raises(ValueError, match="struct_v2_source"):
        adapter.load_task("phase2_flex", {"fitness": {"params": {"struct_v2_source": "both"}}})
    with pytest.raises(ValueError, match="needs flex v2"):
        adapter.load_task("phase1_v4", {"fitness": {"params": {"struct_v2_source": "fd"}}})
    with pytest.raises(ValueError, match="struct_v2_source 'genome'"):
        adapter.load_task("phase2_flex", {"fitness": {"params": {"struct_v2_mass_credit_clip": ["fus"]}}})
    t2 = adapter.load_task("phase2_flex", {"fitness": {"weights": {"structural_v2": 0.5}}})
    assert any("no longer flexeval's" in w for w in t2.warnings)
    tg = adapter.load_task("phase2_flex", {"fitness": {"params": {"struct_v2_source": "genome", "struct_v2_mass_credit_clip": ["fus"]}}})
    assert tg.fitness.params["struct_v2_source"] == "genome"


def test_fd_term_lists_equal_flexeval():
    fe = fd_bridge._import("flexeval")
    assert F.FD_PRE_TERMS_FULL == tuple(fe.PRE_TERMS["full"]) and F.FD_RESP_TERMS_FULL == tuple(fe.RESP_TERMS["full"])
    rigid = tuple(fe.RIGID_TERMS)
    assert rigid == ("track", "effort", "comfort", "heading", "hold")      # controller terms: in flexeval's sim cost
    assert set(F.FD_PRE_TERMS_FULL) | set(F.FD_RESP_TERMS_FULL) | set(rigid) == set(fe.TERM_KEYS) and len(fe.TERM_KEYS) == 24
    assert not (set(F.FD_PRE_TERMS_FULL) | set(F.FD_RESP_TERMS_FULL)) & set(rigid)


def test_fd_structural_v2_is_flexeval_composition():
    resp = {k: 0.0 for k in F.FD_RESP_TERMS_FULL} | {"J_bm_rms": 0.03, "J_tail_bm_peak": 0.002}
    res = {"flex_v2": {"terms": resp}, "flex_v2_pre": {"fd_pre_sum": 0.011}}
    assert F.obj_structural_v2({}, res, {"struct_v2_source": "fd"}) == (0.03 + 0.002) + 0.011
    with pytest.raises(ValueError, match="struct_v2_source"):
        F.obj_structural_v2({}, res, {"struct_v2_source": "x"})


@pytest.mark.sim
def test_fd_mode_reproduces_evolution_phase2_smoke_costs_bitwise():
    """Pre-P2.5 ER smoke genomes (nsm < 1.0) are rejected by FD's P2.5 decode; Phase 2 pilots stay as-is on the
    previous post-mass model_version. This test pins that rejection and that a P2.5-valid genome evaluates."""
    f = os.path.join(os.path.dirname(HERE), "evolution", "runs", "phase2-smoke-s1", "genomes.jsonl")
    if not os.path.exists(f):
        pytest.skip("evolution phase2-smoke-s1 not present")
    rows = [json.loads(l) for l in open(f)]
    best = min((r for r in rows if r["aircraft"] == "c172x" and r.get("feasible")), key=lambda r: r["cost"])
    assert best["struct"]["wing_nsm_root"] < 1.0                         # why it cannot re-score under P2.5
    t = adapter.load_task("phase2_flex")
    with pytest.raises(ValueError, match="wing_nsm_root"):
        t.evaluate({**best["gains"], **best["struct"]}, t.make_scenarios(1, 1))
    # P2.5-valid: same controller gains + FD baseline structure (nsm = 1.0) evaluates under fd mode
    base = {g.name: g.default for g in _v2(t)}
    r = t.evaluate({**best["gains"], **base}, t.make_scenarios(1, 1))
    assert r["struct_v2_source"] == "fd" and r["per_scenario"][0]["status"] == "ok"
    assert r["fd_struct_terms"]["J_wing_tip_bm_limit"] == 0.0            # baseline tip term
    assert r["objectives"]["aeroelastic_margin_penalty"] == 0.0

# ----------------------------------------------------------------------------------------------- P2.5 tip BM + NSM floor
def test_p25_soft_tip_taper_fires_tip_bm_term():
    """FD §12 P2.5: softening only wing_ei_taper_4 to 0.75 fires J_wing_tip_bm_limit while root stays ~0."""
    base = fd_bridge.precheck_v2("c172x", {})
    soft = fd_bridge.precheck_v2("c172x", {"wing_ei_taper_4": 0.75})
    assert base["sizing_terms"]["J_wing_tip_bm_limit"] == 0.0
    tip = soft["sizing_terms"]["J_wing_tip_bm_limit"]
    assert tip == pytest.approx(0.017509318343313852, rel=1e-9) and tip > 0.01
    assert soft["sizing_terms"]["J_wing_bm_limit"] < 1e-4          # root almost untouched
    # fd mode carries the tip term in the pre-flight sum
    t = adapter.load_task("phase2_flex")
    assert t.fitness.params["struct_v2_source"] == "fd"
    g = {**{x.name: x.default for x in t.spec.genes}, "wing_ei_taper_4": 0.75}
    # v4 best gains not required for precheck; structural pre only
    assert "J_wing_tip_bm_limit" in F.FD_PRE_TERMS_FULL
    assert tip == pytest.approx(soft["fd_terms"]["J_wing_tip_bm_limit"])


def test_p25_nsm_floors_from_fd_schema_and_decode_rejects_below_1():
    fb = fd_bridge.flexbody()
    sch = {g.name: g for g in fb.gene_schema()}
    assert sch["wing_nsm_root"].lo == sch["wing_nsm_tip"].lo == 1.0
    assert sch["wing_nsm_root"].hi == sch["wing_nsm_tip"].hi == 1.25
    t = adapter.load_task("phase2_flex")
    for name in ("wing_nsm_root", "wing_nsm_tip"):
        g = next(x for x in t.spec.genes if x.name == name)
        assert (g.min, g.max, g.scale) == (1.0, 1.25, "log")          # built from FD gene_schema(), not a copy
        assert g.decode(0.0) == pytest.approx(1.0)                    # u=0 -> floor 1.0
    with pytest.raises(ValueError, match=r"wing_nsm_root=0\.9 outside \[1\.0, 1\.25\]"):
        fd_bridge.struct_genes_v2({"wing_nsm_root": 0.9})
    with pytest.raises(ValueError, match=r"wing_nsm_tip=0\.99 outside"):
        fd_bridge.struct_genes_v2({"wing_nsm_tip": 0.99})
    assert fd_bridge.struct_genes_v2({"wing_nsm_root": 1.0, "wing_nsm_tip": 1.25})["wing_nsm_tip"] == 1.25
