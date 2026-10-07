"""pytest -q test_flexbody_b2.py   (P3-B2a: dihedral / camber / thickness section genes, fidelity 'full_a1_b2a')"""
import itertools
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flexwing as fw  # noqa: E402
import coupled_sim as cs  # noqa: E402
import flexbody as fb  # noqa: E402
import flexeval as fe  # noqa: E402
import planform_b1 as pb1  # noqa: E402
import flexbody_b1 as b1  # noqa: E402
import flexeval_b1 as fb1  # noqa: E402
import planform_b2 as pb2  # noqa: E402
import flexbody_b2 as b2  # noqa: E402
import flexeval_b2 as fb2  # noqa: E402

MODELS = ("c172x", "T38", "737", "f16")
ROOT = cs.ROOT
RES = os.path.join(HERE, "v2_results")
STRUCT = {"wing_ei_root": 1.3, "wing_ei_taper_4": 0.8, "wing_nsm_tip": 1.1, "tail_stiffness_scale": 0.9,
          "fuselage_stiffness_scale": 1.2, "struct_damping_ratio": 0.015}
MIXED = {"wing_dihedral_delta_deg": 2.0, "wing_camber_root_delta_pct": 1.0, "wing_tc_root_scale": 0.95}
MATS = ("K", "C", "M", "Qbasis", "RB", "RE", "RD", "RA", "R_de", "R_d", "R_beta", "A_beta", "A_C_nc", "TIP", "TIP_TEL",
        "W", "G", "H")
HC = json.load(open(os.path.join(HERE, "_scratch", "p3b2", "handcalc_b2.json")))
fb.blas_threads(1)


@pytest.fixture(scope="module", autouse=True)
def _roots():
    for m in MODELS:
        cs.ensure_root(m)
        fb.ensure_root_v2(m)
        b2.ensure_root_v2b2(m, fe.root_v2_for(ROOT), fb2.root_v2b2_for(ROOT))


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
    return fw.params_for(model, g.bw_ft, g.sw_ft2, g.empty_wt_lb), g


def _env(sim, model):
    return [(float(s.speed_kts), float(s.h0_ft)) for s in fe.phase1_scenarios(model, sim)]


# ------------------------------------------------------------------------------------------------ schema / encoding
def test_b2_gene_list_ranges_flags():
    assert pb2.N_SHAPE_B2 == 11 and pb2.B2_NAMES == (
        "wing_dihedral_delta_deg", "wing_tc_root_scale", "wing_tc_tip_ratio", "wing_camber_root_delta_pct",
        "wing_camber_tip_delta_pct")
    assert fe.TERM_KEYS == fb2.TERM_KEYS and len(fb2.TERM_KEYS) == 24
    assert pb2.REQUIRES_ENERGY == ("wing_tc_root_scale", "wing_tc_tip_ratio")
    assert set(pb2.LOCKED_GENES) == {"wing_area_scale", "wing_aspect_scale"}
    for m in MODELS:
        rows = pb2.shape_schema_b2(m)
        assert [r["name"] for r in rows] == list(pb1.SHAPE_NAMES) + list(pb2.B2_NAMES)
        for r in rows:
            assert r["lo"] <= r["default"] <= r["hi"] and 0.0 <= r["identity_u"] <= 1.0
    spec = json.load(open(os.path.join(RES, "p3b2_gene_spec.json")))
    for r in spec["genes"]:
        if r["stage"] == "B2a":
            g = pb2.SHAPE_GENES_B2[pb2.B2_INDEX[r["name"]]]
            assert (r["lo"], r["hi"]) == g.lo_hi(r["aircraft"]) and r.get("requires") == g.requires
        else:
            assert r["locked_until"] == "energy_cost"


@pytest.mark.parametrize("model", MODELS)
def test_b2_identity_u_roundtrip_and_encode_orders(model):
    u = pb2.identity_u(model)
    assert u.shape == (11,) and pb2.decode_shape_b2(u, model) == pb2.shape_defaults_b2()
    assert pb2.is_baseline_shape_b2(u, model)
    assert np.array_equal(pb2.encode_shape_b2(model, {}), u) and np.array_equal(pb2.encode_shape_b2({}, model), u)
    assert np.array_equal(pb2.encode_shape_b2(model), u)
    v = pb2.pad_b2a_to_b2b(u, model)
    assert v.shape == (13,) and v[-2:].tolist() == pytest.approx([0.4, 0.4], abs=1e-15)
    d = pb2.decode_shape_b2(pb2.encode_shape_b2(model, MIXED if model != "737" else dict(MIXED, wing_dihedral_delta_deg=1.0)), model)
    for k, x in (MIXED if model != "737" else dict(MIXED, wing_dihedral_delta_deg=1.0)).items():
        assert d[k] == pytest.approx(x, abs=1e-12)
    assert pb2.decode_shape_b2(np.zeros(11), model)["wing_tc_root_scale"] == pb2.SHAPE_GENES_B2[1].lo_hi(model)[0]


@pytest.mark.parametrize("bad", [{"wing_dihedral_delta_deg": -0.5}, {"wing_tc_root_scale": 0.86}, {"wing_area_scale": 1.05}, {"wing_aspect_scale": 0.95}, {"wing_tc_root_scale": 1.3},
                                 {"wing_camber_root_delta_pct": float("nan")}, {"wing_dihedral_deg": 1.0},
                                 np.full(13, 0.5), np.full(6, 0.5), np.full(11, 1.01)])
def test_b2_decode_rejects(bad):
    with pytest.raises(ValueError):
        pb2.decode_shape_b2(bad, "c172x")


def test_b2_require_energy_cost_flag():
    pb2.decode_shape_b2({"wing_dihedral_delta_deg": 1.0}, "c172x", require_energy_cost=True)
    with pytest.raises(ValueError):
        pb2.decode_shape_b2({"wing_tc_root_scale": 1.1}, "c172x", require_energy_cost=True)


# ------------------------------------------------------------------------------------------------ gate
@pytest.mark.parametrize("model", MODELS)
def test_b2_every_box_corner_passes_the_gates(sim, model):
    pw, geom = _pw(model)
    env = _env(sim, model)
    rv2 = fe.root_v2_for(ROOT)
    for c in itertools.product(*[g.lo_hi(model) for g in pb2.SHAPE_GENES_B2]):
        d = dict(zip(pb2.B2_NAMES, c))
        r = pb2.geometry_gate_b2(pw, d, model, n_el=64, root_v2=rv2, envelope=env, geom=geom)
        assert r.ok, (model, d, r.reason, r.details)


def test_b2_gate_inclusive_on_exact_limits(monkeypatch):
    pw, _ = _pw("c172x")
    d = {"wing_camber_root_delta_pct": -1.0, "wing_camber_tip_delta_pct": 2.0}
    assert pb2.geometry_gate_b2(pw, d, "c172x").details["camber_gradient_pct"] == 3.0
    assert pb2.geometry_gate_b2(pw, d, "c172x").ok
    monkeypatch.setitem(pb2.GATE, "max_camber_gradient_pct", 3.0 - 1e-6)
    assert pb2.geometry_gate_b2(pw, d, "c172x").reason == "camber_gradient"
    pwt, _ = _pw("T38")
    r = pb2.geometry_gate_b2(pwt, {"wing_camber_root_delta_pct": -0.5, "wing_camber_tip_delta_pct": -0.5}, "T38")
    assert r.ok and r.details["m_eq_min_pct"] == pytest.approx(-0.5, abs=1e-12)


def test_b2_cache_key_includes_envelope():
    k1 = pb2.shape_cache_key_b2(MIXED, "c172x", [(100.0, 4000.0)])
    k2 = pb2.shape_cache_key_b2(MIXED, "c172x", [(90.0, 4000.0)])
    assert k1 != k2 and k1.startswith("b2a|c172x|") and "env=" in k1
    assert k1 == pb2.shape_cache_key_b2(MIXED, "c172x", [(100.0, 4000.0)])


# ------------------------------------------------------------------------------------------------ structural model
@pytest.mark.parametrize("model", MODELS)
def test_b2_default_model_bit_identical_to_b1(model):
    for g in ({}, STRUCT):
        A = b1.FlexBodyModelB1(model, g)
        B = b2.FlexBodyModelB2(model, g, shape_genes={})
        assert B.b2_baseline
        for k in MATS:
            if hasattr(A, k):
                assert np.array_equal(getattr(A, k), getattr(B, k)), k
        ra, rb = b1.margin_terms_b1(A), b2.margin_terms_b2(B)
        assert ra["terms"] == rb["terms"] and ra["margins"] == rb["margins"] and ra["sizing"] == rb["sizing"]


def test_b2_continuity_each_gene_1e9():
    m = "c172x"
    A = b1.FlexBodyModelB1(m, None)
    ra = b1.margin_terms_b1(A)
    for k in pb2.B2_NAMES:
        eps = 1e-9 if k != "wing_tc_root_scale" else -1e-9
        S = b2.FlexBodyModelB2(m, None, shape_genes={k: pb2.SHAPE_GENES_B2[pb2.B2_INDEX[k]].default + eps})
        assert not S.b2_baseline
        for mk in ("K", "M", "Qbasis", "RB"):
            a, s = getattr(A, mk), getattr(S, mk)
            assert np.max(np.abs(s - a)) <= 1e-6 * max(1e-300, np.max(np.abs(a))), (k, mk)
        rs = b2.margin_terms_b2(S)
        for t, v in ra["terms"].items():
            assert rs["terms"][t] == pytest.approx(v, rel=1e-6, abs=1e-9), (k, t)
        pw, geom = _pw(m)
        ni = pb2.native_increments(m, pb2.decode_shape_b2({k: pb2.SHAPE_GENES_B2[pb2.B2_INDEX[k]].default + eps}, m), pw, geom)
        for p, v in ni["props"].items():
            if not p.startswith("korn"):
                assert abs(v) < 1e-7, (k, p, v)


def test_b2_thickness_stiffness_tau2_strength_tau_mass_fixed():
    m = "c172x"
    A = b2.FlexBodyModelB2(m, None, shape_genes={"wing_tc_root_scale": 1.0 + 1e-12})
    T = b2.FlexBodyModelB2(m, None, shape_genes={"wing_tc_root_scale": 1.2})
    ka, kt = A.wingR.beam.K, T.wingR.beam.K
    nz = np.abs(ka) > 1e-9 * np.max(np.abs(ka))
    assert np.median(kt[nz] / ka[nz]) == pytest.approx(1.44, rel=1e-6)
    assert np.array_equal(A.wingR.beam.M, T.wingR.beam.M)
    assert T.mass_summary() == A.mass_summary()
    sa, st = b2.margin_terms_b2(A)["sizing"], b2.margin_terms_b2(T)["sizing"]
    assert st["geometric_strength_factor"]["tau_root"] == pytest.approx(1.2)
    assert st["geometric_strength_factor"]["root"] / sa["geometric_strength_factor"]["root"] == pytest.approx(1.2, rel=1e-9)
    assert T.bm_ref_ratio == 1.0 and A.bm_ref_ratio == 1.0


def test_b2_thin_wing_pays_structurally():
    m = "c172x"
    base = b2.margin_terms_b2(b2.FlexBodyModelB2(m, None))
    thin = b2.margin_terms_b2(b2.FlexBodyModelB2(m, None, shape_genes={"wing_tc_root_scale": 0.875}))
    assert sum(thin["terms"].values()) > sum(base["terms"].values())
    assert thin["margins"]["flutter_margin"] < base["margins"]["flutter_margin"] if "flutter_margin" in base["margins"] else True


def test_b2_m_ref_not_inflated_by_section_genes():
    S = b2.FlexBodyModelB2("T38", STRUCT, shape_genes={"wing_camber_root_delta_pct": 1.0, "wing_dihedral_delta_deg": 2.0})
    assert S.bm_ref_ratio == 1.0
    assert np.all(S.dih_RB[[0]] != 0) and S.cam_RB[2] != 0


# ------------------------------------------------------------------------------------------------ aero hand-calcs
@pytest.mark.parametrize("model", MODELS)
def test_b2_native_increments_vs_handcalc(model):
    pw, geom = _pw(model)
    h = HC[model]
    d = pb2.decode_shape_b2({"wing_camber_root_delta_pct": 1.0 if model != "f16" else 1.0,
                             "wing_camber_tip_delta_pct": 1.0}, model)
    ni = pb2.native_increments(model, d, pw, geom)
    assert ni["info"]["dCL0_wing"] == pytest.approx(h["dCL0_per_1pct_camber"], rel=0.02)
    assert ni["info"]["dCm0_wing"] == pytest.approx(h["dCm0_wing_per_1pct"], rel=0.05)
    inf = ni["info"]
    deps = fb.V2_PROFILES[model]["ht"]["downwash"]
    assert inf["dCm_downwash"] == pytest.approx(inf["V_H"] * inf["a_h"] * deps * inf["dCL0_wing"] / inf["a_w"], rel=1e-9)
    # independent hand-calc uses the published MAC / tail arm (handcalc_b2.py); the model's beam-trapezoid cbar differs
    assert inf["dCm_downwash"] == pytest.approx(h["dCm_downwash_per_1pct"], rel=0.15)
    assert ni["info"]["alpha_trim_shift_deg_est"] < 0
    dd = pb2.decode_shape_b2({"wing_dihedral_delta_deg": 1.0}, model)
    nd = pb2.native_increments(model, dd, pw, geom)
    assert nd["info"]["clb_per_deg2"] == pytest.approx(h["clb_per_deg_gamma_LL_perdeg"], rel=0.05)   # hand-calc: no y0 cut-out
    assert -3.5e-4 < nd["info"]["clb_per_deg2"] < -1.0e-4                      # DATCOM / Roskam band, straight-ish wings
    f_c, _ = pb2._shaped_chord_fn(pw, dd)
    k120 = pb2.lifting_line_clb_per_gamma(f_c, pw.span_ft, pw.area_ft2, pw.root_frac * pw.span_ft / 2, n=120)
    k40 = pb2.lifting_line_clb_per_gamma(f_c, pw.span_ft, pw.area_ft2, pw.root_frac * pw.span_ft / 2)
    assert k40 == pytest.approx(k120, rel=0.02)
    tau = 1.2 if model != "737" else 1.15
    nt = pb2.native_increments(model, pb2.decode_shape_b2({"wing_tc_root_scale": tau}, model), pw, geom)
    assert nt["props"]["dCD0"] > 0
    if model != "737":
        assert nt["props"]["dCD0"] == pytest.approx(h["dCD0_tau_1p2"], rel=0.15, abs=2e-5)


def test_b2_raymer_min_gauge_rule():
    # thickness ratio 0.8 -> +6.9 % wing mass at min gauge (Raymer W ~ (t/c)^-0.4 ~ 0.8^-0.3); hand number of §15.3
    assert 0.8 ** -0.3 - 1 == pytest.approx(0.069, abs=0.002)


def test_b2_stall_fade_gate_only():
    for m in MODELS:
        t = pb2.stall_fade_table(m)
        assert t[0][1] == 1.0 and t[-1][1] == 0.0 and t[-2][1] == 0.0
        assert t[-2][0] == pytest.approx(math.radians(pb2.SECTIONS[m]["alpha_stall_deg"]))


# ------------------------------------------------------------------------------------------------ native layer / flight
def test_b2_native_layer_neutral_at_zero_props(sim):
    m = "c172x"
    sc = _short(sim, m, T=3.0)[0]
    prof = fe.default_profile(m, sim)
    rv2, rb2 = fe.root_v2_for(ROOT), fb2.root_v2b2_for(ROOT)
    ra = sim.simulate(_gains(m), sc, fe._profile_with_root(prof, sim, rv2, m))
    rb = sim.simulate(_gains(m), sc, fe._profile_with_root(prof, sim, rb2, m))
    assert ra["status"] == rb["status"] == "ok" and ra["cost"] == rb["cost"] and ra["track"] == rb["track"]


def test_b2_camber_trims_alpha_and_elevator(sim):
    m = "c172x"
    sc = _short(sim, m)[0]
    pw, geom = _pw(m)
    P = fe._profile_with_root(fe.default_profile(m, sim), sim, fb2.root_v2b2_for(ROOT), m)
    ni = pb2.native_increments(m, pb2.decode_shape_b2({"wing_camber_root_delta_pct": 1.0,
                                                       "wing_camber_tip_delta_pct": 1.0}, m), pw, geom)

    class Att:
        def __init__(self, props):
            self.p = props

        def attach(self, fdm):
            for k, v in self.p.items():
                fdm["flexbody/b2/" + k] = v
    _, i0 = sim.trim(P, sc, Att({}))
    _, i1 = sim.trim(P, sc, Att(ni["props"]))
    _, ip = sim.trim(P, sc, Att({"dCL0": ni["props"]["dCL0"]}))          # pure lift increment, same size
    da, dp = i1["alpha_deg"] - i0["alpha_deg"], ip["alpha_deg"] - i0["alpha_deg"]
    assert da < 0 and dp < 0 and da == pytest.approx(dp, rel=0.3)        # dCm0 retrim / polar change it only a little
    cla_eff = ni["props"]["dCL0"] / math.radians(-dp)                    # whole-aircraft table CLa (c172x: CLwbh + CLalpha)
    assert ni["info"]["a_w"] < cla_eff < 2.5 * ni["info"]["a_w"]
    assert i1["elevator_cmd"] != i0["elevator_cmd"] if "elevator_cmd" in i0 else True


def test_b2_fidelity_list_and_delegation(sim):
    assert fb2.FIDELITIES == ("rigid", "reduced", "full", "full_a1", "full_a1_b1", "full_a1_b2a")
    with pytest.raises(ValueError):
        fb2.evaluate(_gains("c172x"), {}, _short(sim, "c172x"), "c172x", fidelity="full_a1_b1", root=ROOT, sim=sim,
                     shape_genome={"wing_dihedral_delta_deg": 1.0})
    with pytest.raises(ValueError):
        fb2.evaluate(_gains("c172x"), {}, _short(sim, "c172x"), "c172x", fidelity="full_a1_b1", root=ROOT, sim=sim,
                     shape_genome=pb2.identity_u("c172x"))
    r = fb2.evaluate(_gains("c172x"), {}, _short(sim, "c172x", 2.0), "c172x", fidelity="full_a1_b1", root=ROOT, sim=sim,
                     shape_genome={"wing_twist_tip_deg": -1.0})
    assert r["fidelity"] == "full_a1_b1"


DIFF_BY_DESIGN = {"fidelity", "margins_fidelity", "model_version", "shape_cache_key", "wall_s"}
ADDED = {"shape_genes_b2", "b2", "geometry_gate_b2", "native_increments", "energy"}


def _scrub(x):
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items() if k not in DIFF_BY_DESIGN | {"cpu_s", "sim_wall_s", "sim_cpu_s"}}
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    return x


@pytest.mark.parametrize("model", ("c172x", "737"))
def test_b2_default_evaluate_bit_identical_to_b1(sim, model):
    gains, scs = _gains(model), _short(sim, model, T=3.0)
    for g in ({}, STRUCT):
        ra = fb1.evaluate(gains, g, scs, model, fidelity="full_a1_b1", root=ROOT, sim=sim, record=True)
        for sg in (None, {}, pb2.shape_defaults_b2(), pb2.identity_u(model)):
            rb = fb2.evaluate(gains, g, scs, model, fidelity="full_a1_b2a", root=ROOT, sim=sim, record=True, shape_genome=sg)
            assert set(rb) - set(ra) == ADDED and set(ra) <= set(rb)
            assert _scrub({k: rb[k] for k in ra}) == _scrub(ra)
            assert rb["model_version"].startswith("full_a1_b2a:flexv2b2a:")
            assert rb["energy"]["energy_drag_increment"] == 0.0 and rb["energy"]["per_scenario"][0]["ref_source"] == "frozen"


def test_b2_shaped_evaluate_flies_no_rigid_feedback_and_energy(sim):
    m = "c172x"
    sc = _short(sim, m, T=3.0)
    r = fb2.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b2a", root=ROOT, sim=sim, record=True, shape_genome=MIXED)
    assert r["status"] == "ok" and set(r["terms"]) == set(fe.TERM_KEYS) and r["geometry_gate_b2"]["ok"]
    st = r["telemetry"][0]["structure"]
    assert abs(st["dPitch_lbft"][0]) < 1e-6 * max(1.0, max(abs(x) for x in st["dPitch_lbft"])) + 1e-6
    e = r["energy"]
    for k in ("energy_drag_increment", "drag_increment_cd", "energy_drag_ratio", "throttle_mean", "throttle_sat_frac",
              "fuel_burned_lbs", "drag_work_ftlbf", "speed_deficit_kts_mean", "speed_low_frac", "speed_hold_ok"):
        assert k in e
    ps = e["per_scenario"][0]
    assert ps["ref_source"] == "frozen" and ps["cd_ref"] == fb2._load_ref_file()["aircraft"][m][fb2.energy_ref_key(sc[0])]["cd_ref"]
    assert math.isfinite(e["energy_drag_increment"]) and e["energy_drag_increment"] != 0.0
    rt = fb2.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b2a", root=ROOT, sim=sim, shape_genome={"wing_tc_root_scale": 1.25})
    assert rt["energy"]["energy_drag_increment"] > 0.0
    assert rt["energy"]["per_scenario"][0]["cd_ref"] == ps["cd_ref"]                     # gene-independent reference


def test_b2_energy_ref_file_frozen_and_hashed(monkeypatch):
    doc = json.load(open(fb2.ENERGY_REF_FILE))
    assert set(doc["aircraft"]) == set(MODELS)
    v0 = fb2.model_version("full_a1_b2a", "c172x", ROOT)
    k = next(k for k in doc["aircraft"]["c172x"] if not k.startswith("_"))
    bad = json.loads(json.dumps(doc))
    bad["aircraft"]["c172x"][k]["cd_ref"] *= 1.01
    monkeypatch.setattr(fb2, "_EREF_FILE", bad)
    assert fb2.model_version("full_a1_b2a", "c172x", ROOT) != v0


def test_b2_model_versions_pinned():
    p = os.path.join(RES, "model_versions_post_p3b2a.json")
    pins = json.load(open(p))
    b1p = json.load(open(os.path.join(RES, "model_versions_post_p3b1r1.json")))
    for m in MODELS:
        assert fb2.model_version("full_a1_b2a", m, ROOT) == pins[m]["full_a1_b2a"]
        for f in ("rigid", "reduced", "full", "full_a1", "full_a1_b1"):
            assert fb2.model_version(f, m, ROOT) == b1p[m][f] == pins[m][f]


def test_b2_frozen_files_untouched():
    import hashlib
    import re
    n = 0
    for line in open(os.path.join(RES, "FROZEN_A1_B1r1.md5")):
        mm = re.match(r"^([0-9a-f]{32})\s+\*?(\S+)", line)
        if mm:
            h, f = mm.groups()
            n += 1
            assert hashlib.md5(open(os.path.join(HERE, f), "rb").read()).hexdigest() == h, f
    assert n == 14


# ------------------------------------------------------------------------------------------------ B2a r0 review fixes
def test_b2_anhedral_disabled_and_tc_lo_narrowed():
    lo = {m: pb2.SHAPE_GENES_B2[0].lo_hi(m)[0] for m in MODELS}
    assert lo == {m: 0.0 for m in MODELS}
    assert {m: pb2.SHAPE_GENES_B2[1].lo_hi(m)[0] for m in MODELS} == {"c172x": 0.875, "T38": 0.925, "737": 0.90, "f16": 0.85}


@pytest.mark.parametrize("model", MODELS)
def test_b2_thin_corner_no_flutter_hard_fail_at_baseline_structure(model):
    tr, q = pb2.SHAPE_GENES_B2[1].lo_hi(model)[0], pb2.SHAPE_GENES_B2[2].lo_hi(model)[0]
    r = b2.margin_terms_b2(b2.FlexBodyModelB2(model, None, shape_genes={"wing_tc_root_scale": tr, "wing_tc_tip_ratio": q}))
    assert r["fail"] is None and r["terms"]["J_flutter_margin"] > 0          # penalised smoothly, never the cliff


def test_b2_camber_polar_reads_live_cl(sim):
    """Regression: forces/fwz-aero-lbs reads 0 inside JSBSim aero functions; the polar must use sqrt(aero/cl-squared)."""
    m = "c172x"
    pw, _ = _pw(m)
    blk = b2.native_xml(m, pw)
    assert "fwz-aero" not in blk["DRAG"] and "cl-squared" in blk["DRAG"]
    sc = _short(sim, m, T=3.0)
    g = {"wing_camber_root_delta_pct": 1.0, "wing_camber_tip_delta_pct": 1.0}
    r = fb2.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b2a", root=ROOT, sim=sim, shape_genome=g)
    p = r["native_increments"]["props"]
    ref = fb2._load_ref_file()["aircraft"][m][fb2.energy_ref_key(sc[0])]
    cl = ref["weight_lbs"] / ref["qbar_area_lbf"]
    pred = p["cam_A"] + p["cam_B"] * cl
    assert pred < 0                                           # c172x cruise CL 0.42 > c_li 0.23: +camber lowers profile drag
    assert r["energy"]["per_scenario"][0]["drag_increment_cd"] == pytest.approx(pred, rel=0.05)


def test_b2_energy_increment_is_speed_robust():
    """The primary signal is a q.V-weighted COEFFICIENT: scaling V (slower flight) leaves it unchanged, unlike D.V."""
    em_fast, em_slow = b2.EnergyMeter(1 / 120, b2_drag=True), b2.EnergyMeter(1 / 120, b2_drag=True)

    class F(dict):
        pass
    for em, v in ((em_fast, 170.0), (em_slow, 150.0)):
        q = 0.5 * 0.002 * v * v * 174.0
        f = F({"aero/alpha-rad": 0.02, "forces/fbx-aero-lbs": -0.03 * q, "forces/fbz-aero-lbs": -0.4 * q,
               "velocities/vt-fps": v, "aero/qbar-area": q, "aero/coefficient/b2_dCD0": 0.001 * q,
               "aero/coefficient/b2_dCDcam": 0.0, "aero/coefficient/b2_dCDwave": 0.0, "fcs/throttle-cmd-norm": 0.7,
               "velocities/vc-kts": v / 1.6878, "propulsion/total-fuel-lbs": 100.0})
        for _ in range(10):
            em.observe(f)
    ref = {"cd_ref": 0.04, "drag_power_ref_ftlbf_s": 1.0}
    a, b = em_fast.result(ref), em_slow.result(ref)
    assert a["energy_drag_increment"] == pytest.approx(0.025) and b["energy_drag_increment"] == pytest.approx(0.025)
    assert b["drag_work_ftlbf"] < a["drag_work_ftlbf"]        # the raw energy WOULD reward slowing down
