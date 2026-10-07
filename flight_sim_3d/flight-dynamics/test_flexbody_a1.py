"""pytest -q test_flexbody_a1.py   (P3-A1: opt-in denser full structural model, fidelity 'full_a1')"""
import json
import math
import os
import shutil
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

MODELS = ("c172x", "T38", "737", "f16")
ROOT = cs.ROOT
RES = os.path.join(HERE, "v2_results")
TIP_SOFT = {"wing_ei_taper_4": 0.75}
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


def _short(sim, model, T=4.0, n=1):
    scs = fe.phase1_scenarios(model, sim)[:n]
    for s in scs:
        s.duration_s = T
    return scs


# ------------------------------------------------------------------------------------------------------ structure
def test_a1_selectable_mesh_and_mode_set():
    assert "full_a1" in fa.FIDELITIES and fa.FIDELITIES[:3] == fe.FIDELITIES == ("rigid", "reduced", "full")
    assert 48 <= a1.WING_A1["n_el"] <= 64
    assert (a1.WING_A1["n_b"], a1.WING_A1["n_t"], a1.WING_A1["n_ip"]) == (4, 3, 2)
    for m in MODELS:
        A, F = a1.FlexBodyModelA1(m), fb.FlexBodyModel(m)
        assert A.wingR.sp.n_el == a1.WING_A1["n_el"] and A.wingR.beam.n_el == a1.WING_A1["n_el"]
        assert sorted(A.wingR.cls.tolist()) == ["b"] * 4 + ["t"] * 3 + ["v"] * 2
        assert A.N == 31 and F.N == 25
        # same tails and fuselage as full (identical FE, modes and mass)
        for b in ("htR", "htL", "vt", "fusV", "fusL"):
            assert np.array_equal(getattr(A, b).omega, getattr(F, b).omega), (m, b)
            assert np.array_equal(getattr(A, b).Phi, getattr(F, b).Phi), (m, b)
        assert A.ht_spec == F.ht_spec and A.vt_spec == F.vt_spec
        # calibration redone on the A1 mesh: first bending / torsion / in-plane at the profile targets
        f = A.frequencies_hz()["wingR"]
        assert abs(f[0] - A.pw.f_b1_hz) / A.pw.f_b1_hz < 0.02
        assert abs(min(x for x, c in zip(f, A.wingR.cls) if c == "v") - 2.5 * A.pw.f_b1_hz) / (2.5 * A.pw.f_b1_hz) < 0.02
        # aileron Cl_da calibration preserved on the denser strips
        fl = lambda M: float(np.sum((M.st["c"] * M.st["dy"] * M.st["cl_d"] * M.st["mask"] * M.st["y"])[M.body_strips["wingR"]]))  # noqa: E731
        assert fl(A) == pytest.approx(fl(F), rel=1e-9)
        assert A.mass_summary()["total_lb"] == 0.0


def test_a1_model_class_reproduces_full_at_full_mesh():
    """FlexBodyModelA1 with full's wing config (32 strips, 3b+2t+1ip) is bit-identical to flexbody.FlexBodyModel: the
    A1 constructor is flexbody's with only the wing mesh / modes swapped (guards drift of that copy)."""
    full_cfg = {k: fb.WING_V2[k] for k in ("n_el", "n_b", "n_t", "n_ip", "f_ip_ratio")}
    for m in MODELS:
        for g in (None, {"wing_ei_root": 1.4, "wing_ei_taper_2": 0.8, "wing_nsm_tip": 1.2, "tail_stiffness_scale": 0.7,
                         "fuselage_stiffness_scale": 1.3, "struct_damping_ratio": 0.01}):
            F, X = fb.FlexBodyModel(m, g), a1.FlexBodyModelA1(m, g, wing_mesh=full_cfg)
            for k in ("K", "C", "Qbasis", "RB", "RE", "RD", "RA", "R_de", "R_d", "R_beta", "A_beta", "TIP", "TIP_TEL", "A_C_nc"):
                assert np.array_equal(getattr(F, k), getattr(X, k)), (m, k)
            for gk in fb.GROUPS:
                assert np.array_equal(F.A_K[gk], X.A_K[gk]) and np.array_equal(F.A_C[gk], X.A_C[gk])
            assert F.mass_summary() == X.mass_summary()
    with pytest.raises(ValueError):
        a1.FlexBodyModelA1("c172x", wing_mesh={"n_strips": 64})


def test_a1_baseline_margins_pass_gate_and_close_to_full():
    for m in MODELS:
        ra, rf = fb.margins_v2(a1.FlexBodyModelA1(m)), fb.margins_v2(fb.FlexBodyModel(m))
        assert min(ra["flutter_margin"], ra["div_margin"], ra["reversal_margin"]) >= 1.0 and not ra["margin_error"], m
        assert abs(ra["flutter_margin"] - rf["flutter_margin"]) < 0.01, m
        assert abs(ra["reversal_margin"] - rf["reversal_margin"]) < 0.02, m


# --------------------------------------------------------------------------------------------------------- sizing
@pytest.mark.parametrize("model", MODELS)
def test_a1_baseline_sizing_terms_zero_and_tip_soft_fires(model):
    w = fb.StructWeightsV2()
    sz = a1.sizing_a1(a1.FlexBodyModelA1(model), w)
    assert set(sz["terms"]) == set(fb.SIZING_TERMS)
    assert all(v == 0.0 for v in sz["terms"].values()), sz["terms"]
    assert all(abs(r - 1.0) < 1e-12 for r in sz["ratios"].values())
    assert sz["tip_bm_method"] == "station_exact"
    soft = a1.sizing_a1(a1.FlexBodyModelA1(model, TIP_SOFT), w)
    assert soft["terms"]["J_wing_tip_bm_limit"] > 0.01 and soft["ratios"]["wingR_tip_bm"] > 1.0
    assert soft["terms"]["J_wing_bm_limit"] < soft["terms"]["J_wing_tip_bm_limit"]
    # every non-tip check is flexbody.sizing_v2's own, unchanged
    ref = fb.sizing_v2(a1.FlexBodyModelA1(model, TIP_SOFT), w)
    for k in fb.SIZING_TERMS:
        if k != "J_wing_tip_bm_limit":
            assert soft["terms"][k] == ref["terms"][k]
    # weight 1.0, station 0.875 (P2.5)
    assert w.w_wing_tip_bm_limit == 1.0 and w.tip_bm_eta == 0.875


def test_a1_tip_bm_station_exact_converges_with_strips():
    """Station-exact outboard check: mesh independent (< 1 % between 32 and 128 strips), and equal to the strip-discrete
    demand when eta 0.875 falls on a strip boundary (64 strips: 0.875 * 64 = 56)."""
    w = fb.StructWeightsV2()
    for m in ("c172x", "737"):
        J = [a1.sizing_a1(a1.FlexBodyModelA1(m, TIP_SOFT, wing_mesh={"n_el": n}), w)["terms"]["J_wing_tip_bm_limit"]
             for n in (32, 64, 128)]
        assert max(J) / min(J) - 1 < 0.01, (m, J)
        M = a1.FlexBodyModelA1(m, TIP_SOFT)
        assert a1.wing_tip_bm_station(M, w)["R"] == pytest.approx(fb.wing_design_loads(M, w)["R"]["tip_bm"], rel=1e-12)


# --------------------------------------------------------------------------------------------------------- versions
def test_post_p25_model_version_strings_still_resolve():
    """P3-A1 must not move any pinned string: rigid / reduced / full == model_versions_post_p25.json (flexeval and the
    flexeval_a1 wrapper)."""
    pin = json.load(open(os.path.join(RES, "model_versions_post_p25.json")))
    for m in MODELS:
        for fid in ("rigid", "reduced", "full"):
            assert fe.model_version(fid, m, ROOT) == pin[m][fid], (m, fid)
            assert fa.model_version(fid, m, ROOT) == pin[m][fid], (m, fid)


def test_a1_model_version_distinct_pinned_and_tracking(tmp_path, monkeypatch):
    pin = json.load(open(os.path.join(RES, "model_versions_post_p3a1.json")))
    p25 = json.load(open(os.path.join(RES, "model_versions_post_p25.json")))
    seen = set()
    for m in MODELS:
        v = fa.model_version("full_a1", m, ROOT)
        assert v.startswith("full_a1:flexv2a1:") and len(v.split(":")[-1]) == 8
        assert v == pin[m]["full_a1"], f"{m}: regenerate v2_results/model_versions_post_p3a1.json (python p3a1_study.py versions)"
        assert v not in p25[m].values()
        assert {k: pin[m][k] for k in ("rigid", "reduced", "full")} == p25[m]
        seen.add(v)
    assert len(seen) == 4
    v0 = fa.model_version("full_a1", "c172x", ROOT)
    # mesh / mode config is hashed
    monkeypatch.setattr(a1, "WING_A1", dict(a1.WING_A1, n_el=48))
    assert fa.model_version("full_a1", "c172x", ROOT) != v0
    monkeypatch.setattr(a1, "WING_A1", dict(a1.WING_A1, n_el=64, n_t=2))
    assert fa.model_version("full_a1", "c172x", ROOT) != v0
    monkeypatch.undo()
    assert fa.model_version("full_a1", "c172x", ROOT) == v0
    # A1 code is hashed (copy of flexbody_a1.py with one extra byte)
    src = os.path.join(HERE, "flexbody_a1.py")
    cp = tmp_path / "flexbody_a1.py"
    shutil.copyfile(src, cp)  # content only: the frozen source is read-only
    with open(cp, "a") as f:
        f.write("\n")
    monkeypatch.setattr(fa, "CODE_FILES_A1", tuple(cp if p == src else p for p in fa.CODE_FILES_A1))
    assert fa.model_version("full_a1", "c172x", ROOT) != v0
    monkeypatch.undo()
    # weights are hashed; rigid / reduced / full do not depend on the A1 files
    assert fa.model_version("full_a1", "c172x", ROOT, weights=fb.StructWeightsV2(w_wing_tip_bm_limit=2.0)) != v0
    assert all(os.path.basename(p) not in ("flexbody_a1.py", "flexeval_a1.py") for f in fe.CODE_FILES.values() for p in f)
    with pytest.raises(ValueError):
        fa.model_version("full_b2", "c172x", ROOT)


def test_term_keys_and_gene_schema_unchanged():
    assert fa.TERM_KEYS is fe.TERM_KEYS and len(fe.TERM_KEYS) == 24
    assert fa.PRE_TERMS["full_a1"] == fe.PRE_TERMS["full"] and fa.RESP_TERMS["full_a1"] == fe.RESP_TERMS["full"]
    assert fa.SUBSTEPS["full_a1"] == fe.SUBSTEPS["full"] and fa.MARGIN_GATE["full_a1"] == fe.MARGIN_GATE["full"] == 1.0
    assert len(fb.GENES_V2) == 12
    nsm = {g.name: (g.lo, g.hi, g.scale) for g in fb.GENES_V2 if g.name.startswith("wing_nsm")}
    assert nsm == {"wing_nsm_root": (1.0, 1.25, "log"), "wing_nsm_tip": (1.0, 1.25, "log")}


# --------------------------------------------------------------------------------------------------------- evaluate
def test_a1_evaluate_contract_deterministic(sim):
    u = {"wing_ei_root": 1.3, "wing_ei_taper_1": 0.9, "wing_ei_taper_2": 0.85, "wing_ei_taper_3": 0.95, "wing_ei_taper_4": 0.8,
         "wing_gj_ratio_root": 1.05, "wing_gj_ratio_tip": 1.1, "wing_nsm_root": 1.1, "wing_nsm_tip": 1.05,
         "tail_stiffness_scale": 1.2, "fuselage_stiffness_scale": 0.8, "struct_damping_ratio": 0.015}
    for m in ("c172x", "f16"):
        scs = _short(sim, m, T=4.0, n=2)
        r1 = fa.evaluate(_gains(m), u, scs, m, fidelity="full_a1", root=ROOT)
        r2 = fa.evaluate(_gains(m), u, scs, m, fidelity="full_a1", root=ROOT)
        assert r1["cost"] == r2["cost"] and r1["terms"] == r2["terms"] and r1["loads"] == r2["loads"]
        assert tuple(r1["terms"]) == fe.TERM_KEYS and all(math.isfinite(v) for v in r1["terms"].values())
        assert set(r1["terms_available"]) <= set(fe.TERM_KEYS)
        assert all(r1["terms"][k] == 0.0 for k in fe.TERM_KEYS if k not in r1["terms_available"])
        assert r1["fidelity"] == "full_a1" and r1["margins_fidelity"] == "full_a1" and r1["margin_gate"] == 1.0
        assert r1["model_version"] == fa.model_version("full_a1", m, ROOT) and r1["status"] == "ok"
        assert r1["structural_model"]["modal_dof"] == 31
        assert len(r1["margins"]["f_modes_hz"]["wingR"]) == 9
        assert r1["terms"]["J_wing_tip_bm_limit"] == r1["sizing"]["terms"]["J_wing_tip_bm_limit"]
        assert r1["sizing"]["tip_bm_method"] == "station_exact"
    # baseline genome: every sizing term 0 in evaluate as well
    rb = fa.evaluate(_gains("737"), None, _short(sim, "737", T=2.0), "737", fidelity="full_a1", root=ROOT)
    assert all(rb["terms"][k] == 0.0 for k in fb.SIZING_TERMS) and rb["status"] == "ok"


def test_wrapper_passes_existing_fidelities_through_unchanged(sim):
    """flexeval_a1.evaluate(fidelity in rigid/reduced/full) == flexeval.evaluate exactly (same code path)."""
    u = dict(TIP_SOFT, tail_stiffness_scale=0.9)
    scs = _short(sim, "c172x", T=3.0)
    for fid in fe.FIDELITIES:
        ra = fa.evaluate(_gains("c172x"), u, scs, "c172x", fidelity=fid, root=ROOT)
        rb = fe.evaluate(_gains("c172x"), u, scs, "c172x", fidelity=fid, root=ROOT)
        assert ra["cost"] == rb["cost"] and ra["terms"] == rb["terms"] and ra["model_version"] == rb["model_version"], fid
    with pytest.raises(ValueError):
        fa.evaluate(_gains("c172x"), None, scs, "c172x", fidelity="full_b2", root=ROOT)
    with pytest.raises(ValueError):
        fa.evaluate(_gains("c172x"), {"wing_nsm_tip": 0.9}, scs, "c172x", fidelity="full_a1", root=ROOT)


def test_a1_record_telemetry_nodes(sim):
    scs = _short(sim, "c172x", T=1.0)
    r = fa.evaluate(_gains("c172x"), None, scs, "c172x", fidelity="full_a1", root=ROOT, record=True)
    tel = r["telemetry"][0]
    nodes = tel["nodes"]
    comp = {c["name"]: c for c in nodes["components"]}
    assert len(comp["wingR"]["node_station_ft"]) == a1.WING_A1["n_el"] + 1 and len(comp["htR"]["node_station_ft"]) == 13
    w = np.asarray(nodes["values"]["wingR"]["w_ft"])
    assert np.allclose(w[:, -1], np.asarray(tel["structure"]["tip_w_ft_R"]), rtol=0, atol=1e-9)   # exports rounded to 1e-9
    assert np.all(w[:, 0] == 0.0)
    r0 = fa.evaluate(_gains("c172x"), None, scs, "c172x", fidelity="full_a1", root=ROOT)
    assert "telemetry" not in r0 and r0["cost"] == r["cost"]


# ---------------------------------------------------------------------------------------------------------- coupler
def _fly(model, root, coupler_mdl=None, zero=False, T=2.0):
    md = cs.MANEUVER_DEFAULTS[model]
    f = fw.new_fdm(model, root)
    c = None
    if coupler_mdl is not None:
        c = fb.FlexBodyCoupler(coupler_mdl, "twoway", fa.SUBSTEPS["full_a1"], zero_feedback=zero)
        if not zero:
            fb.apply_mass_v2(f, coupler_mdl)
    cs.trim(f, md["h_ft"], md["kcas"])
    e0 = f["fcs/elevator-cmd-norm"]
    px = fb.FlexBodyFDM(f, c, 1 / 120)
    out = []
    for k in range(int(T * 120)):
        t = k / 120
        f["fcs/elevator-cmd-norm"] = e0 + (md["d_elev"] if 0.3 <= t < 0.9 else 0.0)
        f["fcs/aileron-cmd-norm"] = md["d_ail"] if 0.9 <= t < 1.3 else 0.0
        f["fcs/rudder-cmd-norm"] = 0.3 if 1.3 <= t < 1.7 else 0.0
        px.run()
        out.append([f[x] for x in ("position/h-sl-ft", "attitude/theta-deg", "velocities/q-rad_sec", "velocities/p-rad_sec",
                                   "velocities/r-rad_sec", "aero/alpha-rad", "aero/beta-rad", "accelerations/Nz")])
    return np.array(out), px


@pytest.mark.parametrize("model", ("c172x", "737"))
def test_a1_zero_feedback_identical_to_stock_and_coupled_deterministic(model):
    stock = None if model in ("c172x", "f16") else ROOT
    a, _ = _fly(model, stock)
    b, _ = _fly(model, fb.ROOT_V2, a1.FlexBodyModelA1(model), zero=True)
    assert np.array_equal(a, b)
    c1, p1 = _fly(model, fb.ROOT_V2, a1.FlexBodyModelA1(model, TIP_SOFT))
    c2, p2 = _fly(model, fb.ROOT_V2, a1.FlexBodyModelA1(model, TIP_SOFT))
    assert np.array_equal(c1, c2) and not np.array_equal(c1, a)
    h1, h2 = p1.history(), p2.history()
    for k in h1:
        assert np.array_equal(h1[k], h2[k]) and np.all(np.isfinite(h1[k])), k


# ------------------------------------------------------------------------------------------------------- truncation
def test_a1_modal_truncation_within_2pct():
    """A1 (4b+3t+2ip) -> +1 per family and +1 / +2 next-lowest modes: wing flutter speed / margin, root and eta 0.875
    bending moments (static aeroelastic + 1-cos gust) and tip deflection change < 2 %, baseline and tip-soft genomes."""
    var = {k: v for k, v in a1.truncation_variants().items() if k in ("N", "b+1", "t+1", "ip+1", "next+2")}
    for m in MODELS:
        for g in (None, TIP_SOFT):
            r = a1.truncation_check_a1(m, g, variants=var)
            assert r["values"]["N"]["n_wing_modes"] == 9
            assert r["max_abs_pct"] < 2.0, (m, g, r["max_abs_pct_vs_N"])
