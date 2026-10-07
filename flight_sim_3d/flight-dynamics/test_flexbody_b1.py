"""pytest -q test_flexbody_b1.py   (P3-B1: planform shape genes on the full_a1 host, fidelity 'full_a1_b1')"""
import json
import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flexwing as fw  # noqa: E402
import coupled_sim as cs  # noqa: E402
import flexbody as fb  # noqa: E402
import flexeval as fe  # noqa: E402
import flexbody_a1 as a1  # noqa: E402
import flexeval_a1 as fa  # noqa: E402
import planform_b1 as pb1  # noqa: E402
import flexbody_b1 as b1  # noqa: E402
import flexeval_b1 as fb1  # noqa: E402

MODELS = ("c172x", "T38", "737", "f16")
ROOT = cs.ROOT
RES = os.path.join(HERE, "v2_results")
SHAPED = {"wing_chord_taper_1": 0.95, "wing_chord_taper_3": 0.95, "wing_twist_mid_deg": -0.5, "wing_twist_tip_deg": -2.0,
          "wing_sweep_qc_delta_deg": 2.0}
STRUCT = {"wing_ei_root": 1.3, "wing_ei_taper_4": 0.8, "wing_nsm_tip": 1.1, "tail_stiffness_scale": 0.9,
          "fuselage_stiffness_scale": 1.2, "struct_damping_ratio": 0.015}
fb.blas_threads(1)


@pytest.fixture(scope="module", autouse=True)
def _roots():
    for m in MODELS:
        cs.ensure_root(m)
        fb.ensure_root_v2(m)


@pytest.fixture(scope="module")
def sim():
    return fe.load_sim()


def _gains(model):
    S = json.load(open(os.path.join(fe.TEAM, "evolution", "runs", "phase1-s1", "summary.json")))["aircraft"]
    return next(a["best_gains"] for a in S if a["aircraft"] == model)


def _short(sim, model, T=3.0, n=1):
    scs = fe.phase1_scenarios(model, sim)[:n]
    for s in scs:
        s.duration_s = T
    return scs


def _pw(model):
    g = fb.geometry_for(model)
    return fw.params_for(model, g.bw_ft, g.sw_ft2, g.empty_wt_lb)


# ------------------------------------------------------------------------------------------------- locked schema
def test_b1_locked_gene_list():
    assert pb1.SHAPE_NAMES == ("wing_chord_taper_1", "wing_chord_taper_2", "wing_chord_taper_3",
                               "wing_twist_mid_deg", "wing_twist_tip_deg", "wing_sweep_qc_delta_deg")
    rng = {g.name: (g.lo, g.hi, g.default, g.scale) for g in pb1.SHAPE_GENES_B1}
    assert rng == {"wing_chord_taper_1": (0.85, 1.05, 1.0, "linear"), "wing_chord_taper_2": (0.85, 1.05, 1.0, "linear"),
                   "wing_chord_taper_3": (0.85, 1.05, 1.0, "linear"), "wing_twist_mid_deg": (-2.0, 1.0, 0.0, "linear"),
                   "wing_twist_tip_deg": (-4.0, 1.0, 0.0, "linear"), "wing_sweep_qc_delta_deg": (-5.0, 5.0, 0.0, "linear")}
    assert pb1.N_SHAPE_GENES == 6 and fb1.TERM_KEYS == fe.TERM_KEYS and len(fb1.TERM_KEYS) == 24
    assert len(fb.gene_schema(False)) == 12                        # structure block untouched
    h = pb1.shape_params_for_hash()
    assert h["sweep_convention"] == "quarter_chord" and h["symmetry"] == "L_equals_R"
    assert {"wing_dihedral_delta_deg", "wing_thickness_scale", "wing_camber_scale"} <= set(pb1.DEFERRED_B1)


def test_b1_decode_vector_dict_roundtrip():
    assert pb1.decode_shape_b1(None) == pb1.shape_defaults() == pb1.decode_shape_b1({})
    rs = np.random.default_rng(3)
    for _ in range(50):
        u = rs.random(6)
        d = pb1.decode_shape_b1(u)
        assert np.allclose(pb1.encode_shape_b1(d), u, atol=1e-12)
        assert pb1.decode_shape_b1(d) == d
    assert pb1.is_baseline_shape(None) and pb1.is_baseline_shape({"wing_dihedral_delta_deg": 0.0})
    assert not pb1.is_baseline_shape({"wing_twist_tip_deg": -0.1})
    assert pb1.shape_cache_key({}) != pb1.shape_cache_key({"wing_twist_tip_deg": -1e-9})


@pytest.mark.parametrize("bad", [
    {"wing_chord_taper_1": 0.84}, {"wing_chord_taper_2": 1.06}, {"wing_twist_tip_deg": -4.01}, {"wing_twist_mid_deg": 1.5},
    {"wing_sweep_qc_delta_deg": 5.5}, {"wing_sweep_qc_delta_deg": float("nan")}, {"wing_twist_tip_deg": "x"},
    {"wing_ei_root": 1.0},                      # structure gene in the shape block
    {"wing_chord_root": 1.1},                   # deferred: wing size
    {"wing_dihedral_delta_deg": 2.0},           # deferred: dihedral
    {"wing_thickness_scale": 1.1},              # deferred: B2
    {"wing_twist_root_deg": 0.5},               # root twist fixed at 0
    {"chord": 1.0},
    np.full(5, 0.5), np.full(7, 0.5), np.r_[np.full(5, 0.5), 1.01], np.r_[np.full(5, 0.5), -0.01], np.r_[np.full(5, 0.5), np.nan],
])
def test_b1_decode_rejects_out_of_range_unknown_deferred(bad):
    with pytest.raises(ValueError):
        pb1.decode_shape_b1(bad)


def test_b1_planform_area_preserving_and_feasible_by_construction():
    rs = np.random.default_rng(7)
    for m in MODELS:
        pw = _pw(m)
        base = pb1.planform_from_wingparams(pw, None, 64)
        assert base.area_norm == 1.0 and np.array_equal(base.c_ft, base.c0_ft)
        for _ in range(40):
            pf = pb1.planform_from_wingparams(pw, rs.random(6), 64)
            assert np.sum(pf.c_ft) * pf.dy_ft == pytest.approx(np.sum(pf.c0_ft) * pf.dy_ft, rel=1e-12)
            assert pb1.geometry_gate_strips(pf).ok                  # the whole gene box passes the gate (all 4 aircraft)
        for corner in ({k: 0.85 for k in ("wing_chord_taper_1", "wing_chord_taper_2", "wing_chord_taper_3")},
                       {k: 1.05 for k in ("wing_chord_taper_1", "wing_chord_taper_2", "wing_chord_taper_3")}):
            assert pb1.geometry_gate(pw, dict(corner, wing_sweep_qc_delta_deg=5.0)).ok, (m, corner)


def test_b1_geometry_gate_rejects_illegal_shapes():
    pw = _pw("c172x")

    def inj(**kw):
        pf = pb1.planform_from_wingparams(pw, None, 64)
        for k, v in kw.items():
            setattr(pf, k, v)
        return pb1.geometry_gate_strips(pf)
    pf0 = pb1.planform_from_wingparams(pw, None, 64)
    c = pf0.c_ft.copy(); c[20] = -0.2
    assert inj(c_ft=c).reason == "negative_or_tiny_chord"
    assert inj(te_x_ft=pf0.le_x_ft - 0.1).reason == "self_intersect"
    c = pf0.c_ft.copy(); c[-1] = 0.05 * c[0]
    assert inj(c_ft=c, te_x_ft=pf0.le_x_ft + c).reason == "extreme_taper"
    assert inj(sweep_qc_deg=60.0).reason == "extreme_sweep"
    le = pf0.le_x_ft.copy(); le[32:] += 5.0
    assert inj(le_x_ft=le, te_x_ft=le + pf0.c_ft).reason == "extreme_le_kink"
    assert pb1.geometry_gate(pw, {"wing_chord_taper_1": 2.0}).reason.startswith("decode")


# ------------------------------------------------------------------------------------------------------- model
@pytest.mark.parametrize("model", MODELS)
def test_b1_baseline_shape_bit_identical_to_a1(model):
    for g in (None, STRUCT):
        A, B = a1.FlexBodyModelA1(model, g), b1.FlexBodyModelB1(model, g, shape_genes={})
        assert B.planform_baseline
        for k in ("K", "C", "M", "Qbasis", "RB", "RE", "RD", "RA", "R_de", "R_d", "R_beta", "A_beta", "A_C_nc", "TIP", "TIP_TEL"):
            assert np.array_equal(getattr(A, k), getattr(B, k)), (model, k)
        for gk in fb.GROUPS:
            assert np.array_equal(A.A_K[gk], B.A_K[gk]) and np.array_equal(A.A_C[gk], B.A_C[gk])
        assert A.mass_summary() == B.mass_summary()
        ra, rb = a1.margin_terms_a1(A), b1.margin_terms_b1(B)
        assert ra["terms"] == rb["terms"] and ra["margins"] == rb["margins"] and ra["fail"] == rb["fail"]


def test_b1_shaped_rules_area_ac_twist_mass():
    for m in ("c172x", "737"):
        A, S = a1.FlexBodyModelA1(m), b1.FlexBodyModelB1(m, shape_genes=SHAPED)
        bs = S.body_strips
        wr = np.r_[np.arange(S.ns)[bs["wingR"]], np.arange(S.ns)[bs["wingL"]]]
        lift = fb.CI["lift"]
        assert S.Lb[lift, wr] @ S.st["x"][wr] == pytest.approx(A.Lb[lift, wr] @ A.st["x"][wr], abs=1e-9)   # AC hold
        assert abs(np.sum(S.twist_L)) < 1e-9 * np.sum(np.abs(S.twist_L))                                    # zero-net
        assert S.mass_summary()["total_lb"] == pytest.approx(0.0, abs=1e-9)                                 # area fixed
        assert S.wingR.sp.sweep_deg == pytest.approx(A.pw.sweep_deg + 2.0)
        assert np.array_equal(S.htR.Phi, A.htR.Phi) and np.array_equal(S.fusV.Phi, A.fusV.Phi)             # tails untouched
        # geometry-derived stiffness: baseline root constants x (c/c_root0)^3
        assert np.allclose(S.wingR.EI, A.wingR.cal["EI_root0"] * (S.wingR.c / A.wingR.c_root) ** 3 * S.wingR.ei_mult)
        # aileron Cl_da calibration kept on the shaped chords
        fl = lambda M: float(np.sum((M.st["c"] * M.st["dy"] * M.st["cl_d"] * M.st["mask"] * M.st["y"])[M.body_strips["wingR"]]))  # noqa: E731
        assert fl(S) == pytest.approx(fl(A), rel=1e-9)


def test_b1_sizing_trades_and_no_twist_credit():
    w = fb.StructWeightsV2()
    sweep = b1.sizing_b1(b1.FlexBodyModelB1("c172x", shape_genes={"wing_sweep_qc_delta_deg": 5.0}), w)
    assert all(abs(sweep["ratios"][f"wingR_{k}"] - 1.0) < 1e-9 for k in ("bm", "torque", "ip", "tip_bm"))
    taper = b1.sizing_b1(b1.FlexBodyModelB1("c172x", shape_genes={f"wing_chord_taper_{k}": 0.85 for k in (1, 2, 3)}), w)
    assert taper["ratios"]["wingR_bm"] < 1.0 < taper["ratios"]["wingR_tip_bm"] and taper["terms"]["J_wing_tip_bm_limit"] > 0
    washin = b1.sizing_b1(b1.FlexBodyModelB1("c172x", shape_genes={"wing_twist_tip_deg": 1.0}), w)
    assert washin["terms"]["J_wing_tip_bm_limit"] > 0
    washout = b1.sizing_b1(b1.FlexBodyModelB1("c172x", shape_genes={"wing_twist_tip_deg": -4.0}), w)
    assert washout["ratios"]["wingR_bm"] >= 1.0 - 1e-12 and washout["ratios"]["wingR_tip_bm"] >= 1.0 - 1e-12   # no relief credit
    assert set(taper["terms"]) == set(fb.SIZING_TERMS)


def test_b1_shape_changes_margins_not_term_keys():
    r0 = b1.margin_terms_b1(b1.FlexBodyModelB1("737", shape_genes={}))
    r1 = b1.margin_terms_b1(b1.FlexBodyModelB1("737", shape_genes={"wing_sweep_qc_delta_deg": -5.0}))
    assert r1["margins"]["flutter_margin"] != r0["margins"]["flutter_margin"]
    assert set(r1["terms"]) == set(r0["terms"])
    tw = b1.margin_terms_b1(b1.FlexBodyModelB1("737", shape_genes={"wing_twist_tip_deg": -3.0}))
    assert tw["margins"]["flutter_margin"] == pytest.approx(r0["margins"]["flutter_margin"], rel=1e-12)  # rigid twist: linear margins unchanged


# ----------------------------------------------------------------------------------------------------- evaluate
def test_b1_fidelity_list_and_delegation(sim):
    assert fb1.FIDELITIES == ("rigid", "reduced", "full", "full_a1", "full_a1_b1")
    r = fb1.evaluate(_gains("c172x"), {}, _short(sim, "c172x", 2.0), "c172x", fidelity="full_a1", root=ROOT, sim=sim,
                     shape_genome=None)
    assert r["fidelity"] == "full_a1" and r["model_version"].startswith("full_a1:flexv2a1:")
    with pytest.raises(ValueError):
        fb1.evaluate(_gains("c172x"), {}, _short(sim, "c172x"), "c172x", fidelity="full", root=ROOT, sim=sim,
                     shape_genome={"wing_twist_tip_deg": -1.0})
    with pytest.raises(ValueError):
        fb1.evaluate(_gains("c172x"), {}, _short(sim, "c172x"), "c172x", fidelity="full_a1_b1", root=ROOT, sim=sim,
                     shape_genome={"wing_sweep_qc_delta_deg": 9.0})


def test_b1_geometry_gate_short_circuits_flight(sim, monkeypatch):
    monkeypatch.setattr(pb1, "GATE_MAX_ABS_SWEEP_DEG", 1.0)
    r = fb1.evaluate(_gains("c172x"), {}, _short(sim, "c172x"), "c172x", fidelity="full_a1_b1", root=ROOT, sim=sim,
                     shape_genome={"wing_sweep_qc_delta_deg": 2.0})
    assert r["status"] == "geometry_gate:extreme_sweep" and r["per_scenario"] == [] and r["margins"] is None
    assert r["geometry_gate"]["ok"] is False and r["cost"] > 100 and set(r["terms"]) == set(fe.TERM_KEYS)


@pytest.mark.parametrize("model", ("c172x", "737"))
def test_b1_baseline_evaluate_bit_identical_to_a1(sim, model):
    gains, scs = _gains(model), _short(sim, model, T=3.0, n=1)
    for g in ({}, STRUCT):
        ra = fa.evaluate(gains, g, scs, model, fidelity="full_a1", root=ROOT, sim=sim, record=True)
        rb = fb1.evaluate(gains, g, scs, model, fidelity="full_a1_b1", root=ROOT, sim=sim, record=True, shape_genome={})
        assert rb["status"] == ra["status"] and rb["cost"] == ra["cost"] and rb["terms"] == ra["terms"]
        assert rb["mass"] == ra["mass"] and rb["margins"] == ra["margins"]
        sa, sb = ra["telemetry"][0]["structure"], rb["telemetry"][0]["structure"]
        assert sa.keys() == sb.keys() and all(sa[k] == sb[k] for k in sa)
        assert rb["model_version"].startswith("full_a1_b1:flexv2b1:")


def test_b1_shaped_evaluate_flies_and_zero_feedback_at_t0(sim):
    m = "c172x"
    rb = fb1.evaluate(_gains(m), {}, _short(sim, m, T=3.0), m, fidelity="full_a1_b1", root=ROOT, sim=sim, record=True,
                      shape_genome=SHAPED)
    assert rb["status"] == "ok" and set(rb["terms"]) == set(fe.TERM_KEYS)
    st = rb["telemetry"][0]["structure"]
    assert abs(st["dPitch_lbft"][0]) < 1e-6 * max(1.0, max(abs(x) for x in st["dPitch_lbft"])) + 1e-6
    assert rb["planform"]["planform_baseline"] is False and rb["telemetry"][0]["planform"]["sweep_qc_deg"] == 2.0


R0_PINS = {"c172x": "full_a1_b1:flexv2b1:3e40908a", "T38": "full_a1_b1:flexv2b1:982bce54",
           "737": "full_a1_b1:flexv2b1:1bc748ac", "f16": "full_a1_b1:flexv2b1:bfb25718"}


def test_b1_model_versions_pinned_and_a1_untouched():
    pins = json.load(open(os.path.join(RES, "model_versions_post_p3b1r1.json")))
    p25 = json.load(open(os.path.join(RES, "model_versions_post_p25.json")))
    pa1 = json.load(open(os.path.join(RES, "model_versions_post_p3a1.json")))
    for m in MODELS:
        assert fa.model_version("full_a1", m, ROOT) == pa1[m]["full_a1"] == pins[m]["full_a1"]
        for f in ("rigid", "reduced", "full"):
            assert fb1.model_version(f, m, ROOT) == p25[m][f] == pins[m][f]
        v = fb1.model_version("full_a1_b1", m, ROOT)
        assert v == pins[m]["full_a1_b1"] and v.startswith("full_a1_b1:flexv2b1:")


def test_b1_model_version_tracks_shape_schema(monkeypatch):
    v0 = fb1.model_version("full_a1_b1", "c172x", ROOT)
    monkeypatch.setattr(pb1, "GATE_MIN_TAPER", 0.13)
    assert fb1.model_version("full_a1_b1", "c172x", ROOT) != v0
    monkeypatch.undo()
    assert fb1.model_version("full_a1_b1", "c172x", ROOT) == v0


# --------------------------------------------------------------------------------------------- r1 (P3-B1 follow-up)
def test_b1r1_old_pin_file_intact_and_superseded():
    r0 = json.load(open(os.path.join(RES, "model_versions_post_p3b1.json")))
    r1 = json.load(open(os.path.join(RES, "model_versions_post_p3b1r1.json")))
    assert pb1.B1_REV == 1 and r1["b1_schema"]["b1_rev"] == 1 and "b1_rev" not in r0["b1_schema"]
    for m in MODELS:
        assert r0[m]["full_a1_b1"] == R0_PINS[m]                       # r0 strings untouched
        assert r1[m]["full_a1_b1"] != R0_PINS[m] and r1[m]["full_a1_b1"].startswith("full_a1_b1:flexv2b1:")
        for f in ("rigid", "reduced", "full", "full_a1"):
            assert r1[m][f] == r0[m][f]


def test_b1_gene_encoding_is_linear_in_value():
    sch = pb1.shape_params_for_hash()
    assert sch["encoding"].startswith("linear_in_value") and all(g["scale"] == "linear" for g in sch["genes"])
    d = pb1.decode_shape_b1([0.5] * pb1.N_SHAPE_GENES)
    for g in pb1.SHAPE_GENES_B1:
        assert d[g.name] == g.lo + 0.5 * (g.hi - g.lo)                  # arithmetic, not geometric, midpoint
    assert d["wing_chord_taper_1"] == pytest.approx(0.95) and d["wing_chord_taper_1"] != pytest.approx((0.85 * 1.05) ** 0.5)
    np.testing.assert_allclose(pb1.encode_shape_b1(d), 0.5, rtol=0, atol=1e-15)


def test_b1r1_twist_pitch_not_fed_back(sim, monkeypatch):
    """Regression (T38 wash-in drift): the basic-twist rigid pitch moment is a Cm0 shift absorbed by trim; it must not
    enter the dynamics. Scaling twist_pitch by 1e3 must leave the flight bit-identical."""
    m = "c172x"
    shape = {"wing_twist_mid_deg": 1.0, "wing_twist_tip_deg": -2.0, "wing_sweep_qc_delta_deg": 2.0}
    run = lambda: fb1.evaluate(_gains(m), {}, _short(sim, m, T=4.0), m, fidelity="full_a1_b1", root=ROOT, sim=sim,  # noqa: E731
                               shape_genome=shape)
    ra = run()
    mdl = b1.FlexBodyModelB1(m, None, shape_genes=shape)
    assert abs(mdl.twist_pitch) > 0.1 and np.any(mdl.twist_Q != 0)
    orig = b1.FlexBodyModelB1._build_load_bases

    def scaled(self):
        orig(self)
        if getattr(self, "_pf_active", False):
            self.twist_pitch *= 1e3
            self._scaled = True
    monkeypatch.setattr(b1.FlexBodyModelB1, "_build_load_bases", scaled)
    rb = run()
    assert b1.FlexBodyModelB1(m, None, shape_genes=shape).twist_pitch == pytest.approx(1e3 * mdl.twist_pitch)
    assert ra["status"] == rb["status"] == "ok" and ra["cost"] == rb["cost"]
    assert [e["track"] for e in ra["per_scenario"]] == [e["track"] for e in rb["per_scenario"]]


T38_BEST_S1 = {   # evolution/runs/phase3b1-smoke-s1 T38:g4:r0 (twist_mid at the +1 ceiling) and its scenario T38:s1
    "gains": {"kp_alt": 0.034950915794722016, "ki_alt": 0.04466679397876709, "kd_alt": 0.3622517831200915,
              "kp_pitch": 0.026924731422016988, "ki_pitch": 0.001245546081493409, "kd_pitch": 0.00824213639495903,
              "kp_hdg": 0.164, "ki_hdg": 0.00010435501938010242},
    "struct": {"wing_ei_root": 1.0698833683711366, "wing_ei_taper_1": 1.040720742248903, "wing_ei_taper_2": 1.0320304765594823,
               "wing_ei_taper_3": 1.0023642361077911, "wing_ei_taper_4": 1.0227087599404778,
               "wing_gj_ratio_root": 1.025022739159911, "wing_gj_ratio_tip": 1.00826624987313,
               "wing_nsm_root": 1.012577491178753, "wing_nsm_tip": 1.0005106340857062,
               "tail_stiffness_scale": 0.972414633948742, "fuselage_stiffness_scale": 0.9763513966130488,
               "struct_damping_ratio": 0.02067525744440856},
    "shape": {"wing_chord_taper_1": 0.9837583686328314, "wing_chord_taper_2": 1.0200350891016143,
              "wing_chord_taper_3": 0.997698105376601, "wing_twist_mid_deg": 1.0, "wing_twist_tip_deg": 0.2912565945361667,
              "wing_sweep_qc_delta_deg": -0.42188558627786144},
    "scenario": {"seed": 1099128569, "duration_s": 90.0, "h0_ft": 10000.0, "speed_kts": 300.0,
                 "steps": [[0.0, 10000.0], [5.0, 10200.0], [50.0, 10000.0]], "wind_north_fps": 14.815420728420232,
                 "wind_east_fps": 18.893081885137402, "gust_sigma_fps": 3.8459483414117317, "gust_tau_s": 2.0,
                 "discrete_gust_fps": -10.96328514280803, "discrete_gust_t_s": 41.554051876408835,
                 "discrete_gust_len_s": 3.0, "ramp_fpm": 600.0, "ramp_accel_g": 0.1}}


def test_b1r1_t38_mid_washin_is_near_neutral(sim):
    """r0: T38 best, scenario s1, twist_mid 0 -> +1 lowered cost by 8.9e-4 (track -0.6%, J_bm_rms -1.6%), monotone past the
    ceiling. r1: < 1e-4 and the track change < 0.05%."""
    B = T38_BEST_S1
    out = {}
    for tm in (0.0, 1.0):
        r = fb1.evaluate(B["gains"], B["struct"], [dict(B["scenario"])], "T38", fidelity="full_a1_b1", root=ROOT, sim=sim,
                         shape_genome=dict(B["shape"], wing_twist_mid_deg=tm))
        assert r["status"] == "ok"
        out[tm] = r
    e0, e1 = out[0.0]["per_scenario"][0], out[1.0]["per_scenario"][0]
    assert abs(out[1.0]["cost"] - out[0.0]["cost"]) < 1e-4
    assert abs(e1["track"] - e0["track"]) < 5e-4 * e0["track"]
    assert abs(e1["struct"]["J_bm_rms"] - e0["struct"]["J_bm_rms"]) < 2e-3 * e0["struct"]["J_bm_rms"]
    assert e1["m_root_1g"] > e0["m_root_1g"] and e1["m_root_1g_ref"] == pytest.approx(e0["m_root_1g_ref"], rel=1e-9)


def test_b1r1_flown_bm_reference_not_inflated_by_shape():
    m = "T38"
    A = a1.FlexBodyModelA1(m, STRUCT)
    tw = b1.FlexBodyModelB1(m, STRUCT, shape_genes={"wing_twist_mid_deg": 1.0, "wing_twist_tip_deg": -1.0})
    assert tw.bm_ref_ratio == pytest.approx(1.0, abs=1e-12)          # twist only: same chord / Schrenk / masses
    qk, m1g = 250.0, 1.0e4
    t = 0.5 * (tw.twist_RB[0] + tw.twist_RB[1])
    assert abs(t) > 0 and b1.wing_bm_reference_b1(tw, m1g + qk * t, qk) == pytest.approx(m1g * tw.bm_ref_ratio, rel=1e-12)
    ch = b1.FlexBodyModelB1(m, STRUCT, shape_genes={"wing_chord_taper_1": 1.05, "wing_chord_taper_2": 1.05,
                                                    "wing_chord_taper_3": 1.05})
    assert ch.bm_ref_ratio < 1.0                                     # more outboard chord -> bigger own 1-g BM
    assert ch.bm_ref_ratio * ch._root_bm_1g_per_n() == pytest.approx(A._root_bm_1g_per_n() if hasattr(A, "_root_bm_1g_per_n")
                                                                       else b1.FlexBodyModelB1._root_bm_1g_per_n(A), rel=1e-12)
    base = b1.FlexBodyModelB1(m, STRUCT)
    assert base.bm_ref_ratio == 1.0 and base.planform_baseline


def _wing(comps, nm="wingR"):
    return next(c for c in comps if c["name"] == nm)


@pytest.mark.parametrize("model", ["c172x", "T38"])
def test_b1r1_node_layout_follows_shaped_wing(model):
    base = b1.FlexBodyModelB1(model, None)
    assert b1.node_layout_b1(base) == fb.node_layout(base)             # baseline: flexbody.node_layout itself
    rp = [0.3, 0.0, -0.2]
    assert b1.node_layout_b1(base, rp) == fb.node_layout(base, rp)
    mdl = b1.FlexBodyModelB1(model, None, shape_genes=SHAPED)
    comps, ref = b1.node_layout_b1(mdl), fb.node_layout(mdl)
    assert [c["name"] for c in comps] == [c["name"] for c in ref]
    for c, r in zip(comps, ref):
        if c["name"] not in ("wingR", "wingL"):
            assert c == r
    w, pw, sp = _wing(comps), mdl.pw, mdl.wingR.sp
    ax = np.array(w["axis_nodes_body_ft"])
    st = np.array(w["node_station_ft"])
    y = mdl.wingR.y0 + st
    xi = st / mdl.wingR.L
    ch = pb1.baseline_chord_ft(pw.span_ft, pw.area_ft2, pw.taper, y) * pb1.chord_multipliers_raw(SHAPED, xi) * mdl.planform.area_norm
    wy_mac = pw.span_ft / 2 / 3 * (1 + 2 * pw.taper) / (1 + pw.taper)
    aft = (y - wy_mac) * np.tan(np.radians(mdl.planform.sweep_qc_deg)) + mdl.ac_shift_ft + (sp.x_ea - 0.25) * ch
    np.testing.assert_allclose(ax[:, 0], -aft, atol=2e-6)
    np.testing.assert_allclose(w["chord_ft"], ch, atol=2e-6)
    # strip-centre chord = the PlanformStrips chord (same law), tip twist = gene, LE/TE one chord apart about the EA
    np.testing.assert_allclose(np.interp(mdl.planform.y_ft, y, ch), mdl.planform.c_ft, rtol=2e-3)
    assert w["geometric_twist_deg"][0] == 0.0 and w["geometric_twist_deg"][-1] == pytest.approx(SHAPED["wing_twist_tip_deg"])
    le, te = np.array(w["le_nodes_body_ft"]), np.array(w["te_nodes_body_ft"])
    np.testing.assert_allclose(np.linalg.norm(le - te, axis=1), ch, atol=5e-6)
    assert le[-1, 2] > ax[-1, 2] - 1e-9                               # washout: LE down (z down +) at the tip
    wl = _wing(comps, "wingL")
    np.testing.assert_allclose(np.array(wl["axis_nodes_body_ft"])[:, 1], -ax[:, 1])


def test_b1r1_node_ea_slope_vs_quarter_chord_sweep():
    """The exported axis is the ELASTIC axis (x_ea of the local chord), not the quarter chord: on a tapered wing it is
    less swept, and d(EA angle)/d(QC sweep) = sec^2(L_qc) / (1 + tan^2(L_ea)) > 1 (T38: +1.68 deg QC -> +1.81 deg EA)."""
    def ang(mdl):
        a = np.array(_wing(b1.node_layout_b1(mdl))["axis_nodes_body_ft"])
        return np.degrees(np.arctan2(-(a[-1, 0] - a[0, 0]), a[-1, 1] - a[0, 1]))
    b0 = b1.FlexBodyModelB1("T38", None)
    s1 = b1.FlexBodyModelB1("T38", None, shape_genes={"wing_sweep_qc_delta_deg": 1.68})
    a0, a1_ = ang(b0), ang(s1)
    assert a0 == pytest.approx(18.70, abs=0.01) and a1_ - a0 == pytest.approx(1.81, abs=0.01)
    k = np.tan(np.radians(24.0)) - np.tan(np.radians(a0))
    assert a1_ - a0 == pytest.approx(np.degrees(np.arctan(np.tan(np.radians(25.68)) - k)) - a0, abs=1e-6)
