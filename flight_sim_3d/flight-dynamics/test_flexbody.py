"""pytest -q test_flexbody.py   (flex v2 + multi-fidelity hook; v1 tests stay in test_flexwing.py)"""
import json
import math
import os
import shutil
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True          # never write __pycache__ into evolution/ (read-only imports)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flexwing as fw  # noqa: E402
import coupled_sim as cs  # noqa: E402
import flexbody as fb  # noqa: E402
import flexeval as fe  # noqa: E402

MODELS = ("c172x", "T38", "737", "f16")
ROOT = cs.ROOT
fb.blas_threads(1)


@pytest.fixture(scope="module", autouse=True)
def _roots():
    for m in MODELS:
        cs.ensure_root(m)
        fb.ensure_root_v2(m)


def _rand_genomes(n, seed, asym=False):
    rng = np.random.default_rng(seed)
    return [rng.random(len(fb.gene_schema(asym))) for _ in range(n)]


# ---------------------------------------------------------------------------------------------------------------- v1
def test_v1_legacy_bit_identical():
    """v1 4-gene path (flexwing.py margins + coupled trajectories, all 4 aircraft) unchanged by v2, full float64."""
    import legacy_fingerprint as lf
    ref = json.load(open(os.path.join(HERE, "v1_legacy_fingerprint.json")))
    assert lf.compute() == ref


# ------------------------------------------------------------------------------------------------------- FE + genes
def test_element_matrices_vs_gauss_quadrature():
    h = 0.37
    xg, wg = np.polynomial.legendre.leggauss(6)
    x = 0.5 * h * (xg + 1); w = 0.5 * h * wg
    xi = x / h
    H = np.stack([1 - 3 * xi ** 2 + 2 * xi ** 3, h * (xi - 2 * xi ** 2 + xi ** 3), 3 * xi ** 2 - 2 * xi ** 3, h * (-xi ** 2 + xi ** 3)])
    N = np.stack([1 - xi, xi])
    assert np.allclose(fb._elem_coupling(h), (H * w) @ N.T, rtol=1e-12, atol=1e-15)
    k, m = fb._elem_bend(h)
    assert np.allclose(m, (H * w) @ H.T, rtol=1e-12, atol=1e-15)
    H2 = np.stack([(-6 + 12 * xi) / h ** 2, (-4 + 6 * xi) / h, (6 - 12 * xi) / h ** 2, (-2 + 6 * xi) / h])
    assert np.allclose(k, (H2 * w) @ H2.T, rtol=1e-12, atol=1e-12)


def test_hand_calcs_uniform_cantilever():
    """FE (default truncation 3b+2t) vs closed form: bending/torsion frequency, divergence q, aileron reversal q, tip
    deflection under uniform load."""
    r = fb.hand_calcs()
    tol = {"f_bend1_hz": 0.05, "f_tors1_hz": 0.1, "q_div_psf": 0.5, "q_reversal_psf": 0.5, "tip_defl_uniform_load_ft": 0.1}
    for k, t in tol.items():
        assert abs(r[k]["pct"]) < t, (k, r[k])


def test_gene_schema():
    assert len(fb.GENES_V2) == 12 and len(fb.gene_schema(True)) == 14
    assert len(set(fb.GENE_NAMES_V2)) == 12
    for asym in (False, True):
        for u in _rand_genomes(5, 1, asym):
            d = fb.decode_genome_v2(u, asym)
            assert np.allclose(fb.encode_genome_v2(d, asym), u, atol=1e-12)
    assert fb.decode_genome_v2(None) == fb.baseline_genes()


@pytest.mark.parametrize("bad", [
    {"wing_ei_rot": 1.0},                       # unknown
    {"stiffness_scale": 1.0},                   # v1 gene
    {"torsion_bend_ratio": 1.0},
    {"x_ea": 0.4}, {"elastic_axis_frac": 0.4}, {"section_cg_frac": 0.45}, {"tip_mass_frac": 0.1},   # fixed per aircraft
    {"wing_ei_root": 2.5}, {"wing_ei_taper_2": 0.7}, {"wing_ei_taper_3": 1.06}, {"struct_damping_ratio": 0.1},             # out of range
    {"wing_ei_root": float("nan")}, {"wing_nsm_tip": float("inf")}, {"wing_ei_root": "abc"},
    {"wing_asym_ei_delta": 0.05},               # asymmetric key without the flag
])
def test_bad_named_genes_raise(bad):
    with pytest.raises(ValueError):
        fb.decode_genome_v2(bad)


@pytest.mark.parametrize("vec", [np.full(11, 0.5), np.full(13, 0.5), np.r_[np.full(11, 0.5), 1.2], np.r_[np.full(11, 0.5), -0.1],
                                 np.r_[np.full(11, 0.5), np.nan]])
def test_bad_vector_genes_raise(vec):
    with pytest.raises(ValueError):
        fb.decode_genome_v2(vec)


def test_spanwise_distributions_smooth_monotone_feasible():
    """PCHIP in log space: positive, no overshoot between control points, C1 (no slope jumps), and the absolute EI(y),
    GJ(y) never grow outboard by more than the taper-ratio bound (1.05 per CP segment)."""
    xi = np.linspace(0, 1, 2001)
    for u in _rand_genomes(200, 2) + [np.zeros(12), np.ones(12)]:
        g = fb.decode_genome_v2(u)
        cp = fb.wing_cp_values(g)
        d = fb.wing_distributions(g, xi)
        for key in ("ei", "gj", "nsm"):
            v = d[key + "_R"]
            assert np.all(v > 0) and np.all(np.isfinite(v))
            for k in range(4):     # no overshoot inside each CP segment
                seg = v[(xi >= fb.CP_ETA[k]) & (xi <= fb.CP_ETA[k + 1])]
                lo, hi = sorted((cp[key][k], cp[key][k + 1]))
                assert seg.min() >= lo * (1 - 1e-9) and seg.max() <= hi * (1 + 1e-9), key
            dl = np.diff(np.log(v)) / np.diff(xi)
            assert np.max(np.abs(np.diff(dl))) < 0.05 * max(1.0, np.max(np.abs(dl))), key    # C1: slope continuous
        assert fb.smoothness_penalty(g) >= 0
    for m in MODELS:
        for u in _rand_genomes(10, 3):
            M = fb.FlexBodyModel(m, u)
            for arr in (M.wingR.EI, M.wingR.GJ):
                assert np.max(arr[1:] / arr[:-1]) <= 1.05 ** 0.25 + 1e-9


def test_mode_selection_and_counts():
    for m in MODELS:
        M = fb.FlexBodyModel(m)
        assert sorted(M.wingR.cls.tolist()) == ["b", "b", "b", "t", "t", "v"]
        assert sorted(M.htR.cls.tolist()) == ["b", "b", "t"] and sorted(M.vt.cls.tolist()) == ["b", "b", "t"]
        assert M.N == 25
        # calibration: uncoupled baseline first bending = profile f_b1 (the coupled FE mode is close)
        assert abs(M.wingR.omega[0] / (2 * math.pi) - M.pw.f_b1_hz) / M.pw.f_b1_hz < 0.02


def test_modal_truncation_converged_all_aircraft():
    """Wing root BM / tip deflection (static aeroelastic and 1-cos gust) and flutter speed change < 2 % from N=6 to N+1, N+2."""
    for m in MODELS:
        for genes in (None, {"wing_ei_root": 0.7, "wing_ei_taper_1": 0.75, "wing_ei_taper_4": 1.05, "wing_nsm_tip": 1.25}):
            r = fb.truncation_check(m, genes)
            for k in ("pct_N+1", "pct_N+2"):
                for q, v in r[k].items():
                    assert abs(v) < 2.0, f"{m} {genes} {k} {q}: {v:.3f}% exceeds the 2% truncation tolerance"


# ------------------------------------------------------------------------------------------------------------ margins
def test_margins_finite_capped_flagged_all_aircraft():
    for m in MODELS:
        for u in [None] + _rand_genomes(3, 4):
            r = fb.margins_v2(fb.FlexBodyModel(m, u))
            for k in ("flutter_margin", "div_margin", "reversal_margin"):
                assert 0.0 <= r[k] <= fb.MARGIN_CAP and math.isfinite(r[k])
            for b in r["blocks"].values():
                for k, v in b.items():
                    if isinstance(v, float):
                        assert math.isfinite(v), (m, k)
                for base in ("flutter", "div"):
                    if b[f"{base}_not_found_below_cap"]:
                        assert b[f"{base}_margin"] == fb.MARGIN_CAP
            assert r["flutter_margin"] == min(b["flutter_margin"] for b in r["blocks"].values())
            assert not r["margin_error"]
    for m in MODELS:     # baseline genome passes the hard gate everywhere
        r = fb.margins_v2(fb.FlexBodyModel(m))
        assert min(r["flutter_margin"], r["div_margin"], r["reversal_margin"]) >= 1.0, m


def test_asymmetric_option():
    g = dict(fb.baseline_genes(True))
    M0 = fb.FlexBodyModel("c172x", g, asymmetric=True)
    assert M0.wingL is M0.wingR                       # zero deltas -> identical to symmetric
    assert np.array_equal(M0.K, fb.FlexBodyModel("c172x").K)
    g["wing_asym_ei_delta"] = 0.08
    M1 = fb.FlexBodyModel("c172x", g, asymmetric=True)
    assert M1.wingL is not M1.wingR and M1.wingR.omega[0] > M1.wingL.omega[0]
    assert "wingL" in fb.margins_v2(M1)["blocks"]


def test_mass_deltas_exact_zero_at_baseline_and_fed_to_jsbsim():
    for m in MODELS:
        assert all(v == 0.0 for k, v in fb.FlexBodyModel(m).mass_summary().items() if k.endswith("_lb") and k != "baseline_flexible_lb")
    M = fb.FlexBodyModel("737", {"wing_ei_root": 1.6, "tail_stiffness_scale": 1.5, "fuselage_stiffness_scale": 1.4, "wing_nsm_root": 1.2})
    f0 = fw.new_fdm("737", fb.ROOT_V2)
    f1 = fw.new_fdm("737", fb.ROOT_V2)
    ms = fb.apply_mass_v2(f1, M)
    for f in (f0, f1):
        f["ic/h-sl-ft"] = 10000.0; f["ic/vc-kts"] = 250.0
        f.run_ic()
    assert abs((f1["inertia/weight-lbs"] - f0["inertia/weight-lbs"]) - ms["total_lb"]) < 1e-6 * f0["inertia/weight-lbs"]
    assert ms["total_lb"] > 0 and f1["inertia/cg-x-in"] > f0["inertia/cg-x-in"]          # tail/fuselage mass moves the CG aft
    assert f1["inertia/iyy-slugs_ft2"] > f0["inertia/iyy-slugs_ft2"]


# ------------------------------------------------------------------------------------------------------------ coupler
def _fly(model, root, coupler_mdl=None, zero=False, T=3.0):
    md = cs.MANEUVER_DEFAULTS[model]
    f = fw.new_fdm(model, root)
    c = None
    if coupler_mdl is not None:
        c = fb.FlexBodyCoupler(coupler_mdl, "twoway", 2, zero_feedback=zero)
        if not zero:
            fb.apply_mass_v2(f, coupler_mdl)
    cs.trim(f, md["h_ft"], md["kcas"])
    e0 = f["fcs/elevator-cmd-norm"]
    px = fb.FlexBodyFDM(f, c, 1 / 120)
    out = []
    for k in range(int(T * 120)):
        t = k / 120
        f["fcs/elevator-cmd-norm"] = e0 + (md["d_elev"] if 0.5 <= t < 1.5 else 0.0)
        f["fcs/aileron-cmd-norm"] = md["d_ail"] if 1.5 <= t < 2.0 else 0.0
        f["fcs/rudder-cmd-norm"] = 0.3 if 2.0 <= t < 2.5 else 0.0
        f["atmosphere/wind-down-fps"] = 0.5 * md["gust_fps"] * (1 - math.cos(2 * math.pi * (t - 2.5) / 0.5)) if t >= 2.5 else 0.0
        px.run()
        keys = ("position/h-sl-ft", "attitude/theta-deg", "velocities/q-rad_sec", "velocities/p-rad_sec", "velocities/r-rad_sec",
                "inertia/cg-x-in", "aero/alpha-rad", "aero/beta-rad", "accelerations/Nz")
        if coupler_mdl is not None and not zero:
            keys += ("forces/fbz-external-lbs", "moments/n-external-lbsft")
        out.append([f[x] for x in keys])
    return np.array(out), px


@pytest.mark.parametrize("model", MODELS)
def test_v2_zero_feedback_identical_to_stock(model):
    """v2 prepared copy + coupler running with exact-zero feedback == stock model (c172x, f16: installed package data;
    T38, 737: the v1 prepared copy, itself verified identical to the socket-stripped stock in test_flexwing.py)."""
    stock = None if model in ("c172x", "f16") else ROOT
    a, _ = _fly(model, stock)
    b, _ = _fly(model, fb.ROOT_V2, fb.FlexBodyModel(model), zero=True)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("model", MODELS)
def test_v2_coupled_deterministic_finite_nonzero(model):
    u = _rand_genomes(1, 5)[0]
    a, pa = _fly(model, fb.ROOT_V2, fb.FlexBodyModel(model, u))
    b, pb = _fly(model, fb.ROOT_V2, fb.FlexBodyModel(model, u))
    assert np.array_equal(a, b)
    ha, hb = pa.history(), pb.history()
    for k in ha:
        assert np.array_equal(ha[k], hb[k]) and np.all(np.isfinite(ha[k])), k
    assert np.max(np.abs(a[:, 9])) > 0 and np.max(np.abs(a[:, 10])) > 0      # force and yaw moment fed back
    assert ha["dL_lbf"][0] == 0.0 or abs(ha["dL_lbf"][0]) < 1e-6 * max(1.0, np.max(np.abs(ha["dL_lbf"])))


# ---------------------------------------------------------------------------------------------------- fidelity hook
@pytest.fixture(scope="module")
def sim():
    return fe.load_sim()


def _gains(model):
    S = json.load(open(os.path.join(fe.TEAM, "evolution", "runs", "phase1-s1", "summary.json")))["aircraft"]
    return next(a["best_gains"] for a in S if a["aircraft"] == model)


def _short(sim, model, T=8.0, n=2):
    scs = fe.phase1_scenarios(model, sim)[:n]
    for s in scs:
        s.duration_s = T
    return scs


def test_rigid_bit_identical_to_evolution_simulate(sim):
    """fidelity='rigid' == the Runner's own evolution.sim.simulate (package import), mean over scenarios as batch.py."""
    import importlib
    sys.path.insert(0, fe.TEAM)
    es = importlib.import_module("evolution.sim")
    for m in ("c172x", "737"):
        P = es.Profile.from_dict(fe.default_profile(m, sim).to_dict())
        scs = es.make_scenarios(3, 1, P)
        rs = [es.simulate(_gains(m), s, P) for s in scs]
        ref = float(np.mean([x["cost"] for x in rs]))
        r = fe.evaluate(_gains(m), None, scs, m, fidelity="rigid", root=ROOT, sim=es)
        assert r["cost"] == ref and r["terms"]["track"] == float(np.mean([x["track"] for x in rs]))
        assert r["terms_available"][:2] == ["track", "effort"] and r["status"] == "ok"
        r2 = fe.evaluate(_gains(m), None, [s.to_dict() for s in scs], m, fidelity="rigid", root=ROOT)   # private module instance
        assert r2["cost"] == ref


def test_fidelity_contract_keys_determinism_versions(sim):
    u = {"wing_ei_root": 1.3, "wing_ei_taper_1": 0.9, "wing_ei_taper_2": 0.85, "wing_ei_taper_3": 0.95, "wing_ei_taper_4": 1.0,
         "wing_gj_ratio_root": 1.05, "wing_gj_ratio_tip": 1.1, "wing_nsm_root": 1.1, "wing_nsm_tip": 1.05,
         "tail_stiffness_scale": 1.2, "fuselage_stiffness_scale": 0.8, "struct_damping_ratio": 0.015}
    for m in ("c172x", "f16"):
        scs = _short(sim, m)
        for fid in fe.FIDELITIES:
            r1 = fe.evaluate(_gains(m), u, scs, m, fidelity=fid, root=ROOT)
            r2 = fe.evaluate(_gains(m), u, scs, m, fidelity=fid, root=ROOT)
            assert r1["cost"] == r2["cost"] and r1["terms"] == r2["terms"], (m, fid)
            assert tuple(r1["terms"]) == fe.TERM_KEYS
            assert all(math.isfinite(v) for v in r1["terms"].values())
            assert set(r1["terms_available"]) <= set(fe.TERM_KEYS)
            assert all(r1["terms"][k] == 0.0 for k in fe.TERM_KEYS if k not in r1["terms_available"])
            assert r1["fidelity"] == fid and r1["model_version"].startswith({"rigid": "rigid:jsbsim", "reduced": "reduced:flexv1:",
                                                                             "full": "full:flexv2:"}[fid])
            assert r1["margins_fidelity"] == (None if fid == "rigid" else fid)
            if fid != "rigid":
                assert r1["margin_gate"] == {"reduced": 0.9, "full": 1.0}[fid]
            assert r1["status"] == "ok"


def test_model_version_tracks_code_and_data(tmp_path, monkeypatch):
    m = "c172x"
    v = {f: fe.model_version(f, m, ROOT) for f in fe.FIDELITIES}
    assert v == {f: fe.model_version(f, m, ROOT) for f in fe.FIDELITIES}          # stable
    assert len(set(v.values())) == 3
    # data: a one-byte change in a copied prepared aircraft changes every fidelity's version
    r1, r2 = tmp_path / "jsbsim_root", tmp_path / "jsbsim_root_v2"
    shutil.copytree(os.path.join(ROOT, "aircraft", m), r1 / "aircraft" / m)
    shutil.copytree(os.path.join(fb.ROOT_V2, "aircraft", m), r2 / "aircraft" / m)
    w = {f: fe.model_version(f, m, str(r1)) for f in fe.FIDELITIES}
    assert w == v                                                                  # identical bytes -> identical version
    for root in (r1, r2):
        x = root / "aircraft" / m / f"{m}.xml"
        x.write_bytes(x.read_bytes() + b" ")
    w2 = {f: fe.model_version(f, m, str(r1)) for f in fe.FIDELITIES}
    assert all(w2[f] != v[f] for f in fe.FIDELITIES)
    # code: a changed copy of the fidelity's code files changes reduced/full but not rigid
    for fid in ("reduced", "full"):
        files = []
        for fp in fe.CODE_FILES[fid]:
            dst = tmp_path / f"{fid}_{os.path.basename(fp)}"
            dst.write_bytes(open(fp, "rb").read() + b"\n# changed\n")
            files.append(str(dst))
        monkeypatch.setitem(fe.CODE_FILES, fid, tuple(files))
        assert fe.model_version(fid, m, ROOT) != v[fid]
    assert fe.model_version("rigid", m, ROOT) == v["rigid"]


def test_model_version_changes_with_v2_params(monkeypatch):
    v0 = {f: fe.model_version(f, "c172x", ROOT) for f in ("reduced", "full")}
    monkeypatch.setitem(fb.V2_PROFILES["c172x"]["ht"], "f_t1", 41.0)
    assert fe.model_version("full", "c172x", ROOT) != v0["full"]
    monkeypatch.undo()
    g = fb.GENES_V2[1]                    # a gene range change re-maps [0,1] vectors -> must change the version
    genes = list(fb.GENES_V2)
    genes[1] = fb.GeneV2(g.name, 0.8, g.hi, g.scale, g.default, g.doc)
    monkeypatch.setattr(fb, "GENES_V2", genes)
    assert all(fe.model_version(f, "c172x", ROOT) != v0[f] for f in v0)
    monkeypatch.undo()
    assert fe.model_version("full", "c172x", ROOT, weights=fb.StructWeightsV2(w_mass=0.5)) != v0["full"]


def test_reduced_substeps_in_version(monkeypatch):
    v0 = fe.model_version("reduced", "c172x", ROOT)
    monkeypatch.setitem(fe.SUBSTEPS, "reduced", 3)
    assert fe.model_version("reduced", "c172x", ROOT) != v0


def test_project_to_reduced():
    p = fe.project_to_reduced(None, "c172x")
    assert abs(p["stiffness_scale"] - 1) < 1e-12 and abs(p["torsion_bend_ratio"] - 1) < 1e-12
    assert abs(p["nonstructural_mass_scale"] - 1) < 1e-12 and p["struct_damping_ratio"] == 0.02
    g = {"wing_ei_root": 1.4, "wing_gj_ratio_root": 1.1, "wing_gj_ratio_tip": 1.1, "wing_nsm_root": 1.2, "wing_nsm_tip": 1.2,
         "struct_damping_ratio": 0.01}
    p = fe.project_to_reduced(g, "737")       # uniform multipliers project exactly
    assert abs(p["stiffness_scale"] - 1.4) < 1e-12 and abs(p["torsion_bend_ratio"] - 1.1) < 1e-12
    assert abs(p["nonstructural_mass_scale"] - 1.2) < 1e-12 and p["struct_damping_ratio"] == 0.01
    soft = {"wing_ei_root": 0.6, "wing_ei_taper_1": 0.75, "wing_ei_taper_2": 0.75}
    p = fe.project_to_reduced(soft, "c172x")
    assert "stiffness_scale" in p["_detail"]["outside_v1_gene_range"]
    assert fe.project_to_reduced(soft, "c172x") == p                                 # deterministic


def test_reduced_screen_gate_and_hard_fail(sim):
    soft = {"wing_ei_root": 0.6, "wing_ei_taper_1": 0.75, "wing_ei_taper_2": 0.75, "wing_ei_taper_3": 0.75, "wing_ei_taper_4": 0.75,
            "wing_gj_ratio_root": 0.8, "wing_gj_ratio_tip": 0.8}
    scs = _short(sim, "c172x", T=2.0, n=1)
    for fid in ("reduced", "full"):
        r = fe.evaluate(_gains("c172x"), soft, scs, "c172x", fidelity=fid, root=ROOT)
        assert r["status"] in ("flutter", "divergence", "reversal") and r["cost"] == fb.StructWeightsV2().fail_cost
        assert all(math.isfinite(v) for v in r["terms"].values())


def test_evaluate_rejects_bad_inputs(sim):
    scs = _short(sim, "c172x", T=1.0, n=1)
    with pytest.raises(ValueError):
        fe.evaluate(_gains("c172x"), None, scs, "c172x", fidelity="medium", root=ROOT)
    with pytest.raises(ValueError):
        fe.evaluate(_gains("c172x"), None, scs, "c172x", fidelity="full", root=ROOT, dt=1 / 60)
    with pytest.raises(ValueError):
        fe.evaluate(_gains("c172x"), {"stiffness_scale": 1.0}, scs, "c172x", fidelity="full", root=ROOT)
    with pytest.raises(ValueError):
        fe.evaluate(_gains("c172x"), None, scs, "c172x", fidelity="rigid", root=ROOT, profile=fe.default_profile("737", sim))


# ------------------------------------------------------------------------------------------- guarantees added at resume
@pytest.mark.parametrize("model", MODELS)
def test_flex_off_on_v2_prepared_copy_identical_to_stock(model):
    """Flex is opt-in: the v2 prepared copy flown with NO coupler (zero-weight point masses, external_reactions never
    written) == stock (c172x/f16: installed package data; T38/737: v1 prepared copy, see test_flexwing.py)."""
    stock = None if model in ("c172x", "f16") else ROOT
    a, _ = _fly(model, stock)
    b, _ = _fly(model, fb.ROOT_V2, None)
    assert np.array_equal(a, b)


def _runner_fidelity():
    sys.dont_write_bytecode = True
    if fe.TEAM not in sys.path:
        sys.path.insert(0, fe.TEAM)
    try:
        from evolution import fidelity as F, sim as S   # read-only import (no bytecode written)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"evolution.fidelity not importable: {e}")
    return F, S


def test_rigid_model_version_and_term_keys_match_runner():
    F, S = _runner_fidelity()
    for m in ("c172x", "737"):
        d = fe.default_profile(m, S).to_dict()
        d["aircraft_root"] = ROOT
        assert F.model_version(d, "rigid") == fe.model_version("rigid", m, ROOT)
    assert set(F.TERM_KEYS) <= set(fe.TERM_KEYS)


def test_hook_path_equals_patch_path(sim, monkeypatch):
    """simulate(flex=hook) (Runner protocol) and the _new_fdm-swap fallback give bit-identical results."""
    u = _rand_genomes(1, 11)[0]
    scs = _short(sim, "c172x", T=4.0, n=1)
    assert fe.sim_supports_flex_hook(sim)
    for fid in ("reduced", "full"):
        a = fe.evaluate(_gains("c172x"), u, scs, "c172x", fidelity=fid, root=ROOT)
        monkeypatch.setattr(fe, "sim_supports_flex_hook", lambda s: False)
        b = fe.evaluate(_gains("c172x"), u, scs, "c172x", fidelity=fid, root=ROOT)
        monkeypatch.undo()
        assert a["cost"] == b["cost"] and a["terms"] == b["terms"], fid


def test_record_telemetry_structure_channels(sim):
    scs = _short(sim, "c172x", T=2.0, n=1)
    r = fe.evaluate(_gains("c172x"), None, scs, "c172x", fidelity="full", root=ROOT, record=True)
    tr = r["telemetry"][0]["trajectory"]
    flex_ch = [c for c in tr["channels"] if c.startswith("flex.")]
    assert "flex.wingR_bm" in flex_ch and "flex.fusV_bm" in flex_ch and "flex.ht_incidence_deg" in flex_ch
    assert all(len(row) == len(tr["channels"]) for row in tr["data"])
    assert np.all(np.isfinite(np.asarray(tr["data"], float)))
    assert len(r["telemetry"][0]["structure"]["wingR_bm"]) == 240
    r0 = fe.evaluate(_gains("c172x"), None, scs, "c172x", fidelity="full", root=ROOT)
    assert r0["cost"] == r["cost"]                       # recording does not change the result


def test_rigid_validates_struct_genome(sim):
    scs = _short(sim, "c172x", T=1.0, n=1)
    with pytest.raises(ValueError):
        fe.evaluate(_gains("c172x"), {"wing_ei_root": 9.0}, scs, "c172x", fidelity="rigid", root=ROOT)


def test_reduced_first_mode_frequency_error_bounded():
    """Reduced (v1, projected genome) vs full (v2 FE) first bending / torsion frequency over random genomes. The
    projection is biased low (documented); this pins the size so a regression is visible."""
    for m in ("c172x", "737"):
        eb, et = [], []
        for u in _rand_genomes(16, 12):
            M = fb.FlexBodyModel(m, u)
            fr = fe.reduced_wing(u, m).margins()["f_modes_hz"]
            cls = list(M.wingR.cls)
            f = M.wingR.omega / (2 * math.pi)
            eb.append((fr[0] - f[cls.index("b")]) / f[cls.index("b")])
            et.append((fr[1] - f[cls.index("t")]) / f[cls.index("t")])
        assert abs(np.mean(eb)) < 0.08 and np.max(np.abs(eb)) < 0.13, (m, eb)
        assert abs(np.mean(et)) < 0.06 and np.max(np.abs(et)) < 0.10, (m, et)
    p0 = fe.project_to_reduced(None, "c172x")
    w0 = fe.reduced_wing(None, "c172x").margins()["f_modes_hz"]
    assert abs(w0[0] - fb.FlexBodyModel("c172x").wingR.omega[0] / (2 * math.pi)) < 0.01 and p0["stiffness_scale"] == pytest.approx(1.0)


def test_control_and_tail_effectiveness_outputs():
    """Elastic/rigid effectiveness at V_D for aileron, elevator, rudder and HT/fin lift: finite, in (0, 1] at the baseline,
    lower for a softer structure; all-moving tails: elevator effectiveness == HT incidence effectiveness."""
    for m in MODELS:
        b0 = fb.margins_v2(fb.FlexBodyModel(m))["blocks"]
        soft = fb.margins_v2(fb.FlexBodyModel(m, {"wing_ei_root": 0.8, "tail_stiffness_scale": 0.7,
                                                   "fuselage_stiffness_scale": 0.7}))["blocks"]
        for blk, key in (("wingR", "aileron"), ("empennage_pitch", "elevator"), ("empennage_pitch", "ht_alpha"),
                         ("empennage_yaw", "rudder"), ("empennage_yaw", "vt_alpha")):
            e0, e1 = b0[blk][f"{key}_effectiveness_at_VD"], soft[blk][f"{key}_effectiveness_at_VD"]
            assert 0.0 < e0 <= 1.0 and math.isfinite(e1) and e1 < e0, (m, key, e0, e1)
        if fb.V2_PROFILES[m]["ht"]["all_moving"]:
            assert b0["empennage_pitch"]["elevator_effectiveness_at_VD"] == pytest.approx(
                b0["empennage_pitch"]["ht_alpha_effectiveness_at_VD"], rel=1e-12)


# ------------------------------------------------------------------------------------- telemetry signs + new fields
def _sgn(L, keys, scale):
    return {k: (0 if abs(L[k]) <= 1e-6 * scale else (1 if L[k] > 0 else -1)) for k in keys}


@pytest.mark.parametrize("model", MODELS)
def test_sign_probe_vertical(model):
    """w + up (wing, HT both sides, aft fuselage), bending + = up-bending / tail-up, twist + = LE up (nose-up), torque +
    nose-up, ht_incidence + nose-up: 1-g inertia (no lift), pitch rate (tail upwash), roll rate (R up / L down)."""
    mdl = fb.FlexBodyModel(model, {})
    keys = ["tip_w_ft_R", "tip_w_ft_L", "wingR_bm", "ht_tip_w_ft", "htL_tip_w_ft", "htR_bm", "htL_bm", "fusV_tip_w_ft",
            "fusV_bm", "tip_twist_R_deg", "wingR_torque", "ht_tip_twist_deg", "htL_tip_twist_deg", "ht_incidence_deg"]
    L = fb.static_probe(mdl, {"accelerations/Nz": 1.0})          # inertia down; section CG aft of the EA -> nose-up
    s = _sgn(L, keys, 1e-3)
    assert s == {**{k: -1 for k in keys[:9]}, **{k: 1 for k in keys[9:]}}
    L = fb.static_probe(mdl, {"velocities/q-rad_sec": 0.1})      # tail up-load
    s = _sgn(L, ["ht_tip_w_ft", "htL_tip_w_ft", "htR_bm", "fusV_tip_w_ft", "fusV_bm", "ht_incidence_deg"], 1e-3)
    assert s == {"ht_tip_w_ft": 1, "htL_tip_w_ft": 1, "htR_bm": 1, "fusV_tip_w_ft": 1, "fusV_bm": 1, "ht_incidence_deg": -1}
    L = fb.static_probe(mdl, {"velocities/p-rad_sec": 0.1})      # roll right: right surfaces see upwash
    s = _sgn(L, ["tip_w_ft_R", "tip_w_ft_L", "ht_tip_w_ft", "htL_tip_w_ft", "ht_tip_twist_deg", "htL_tip_twist_deg"], 1e-3)
    assert s == {"tip_w_ft_R": 1, "tip_w_ft_L": -1, "ht_tip_w_ft": 1, "htL_tip_w_ft": -1, "ht_tip_twist_deg": 1,
                 "htL_tip_twist_deg": -1}


@pytest.mark.parametrize("model", MODELS)
def test_sign_probe_lateral(model):
    """vt / fusL w + = body +y, vt_bm / fusL_bm + = toward +y, vt twist + = LE toward +y, vt_sideslip = -w'_fusL,tip
    (+ = fin LE toward +y, i.e. the opposite sense of aircraft beta)."""
    mdl = fb.FlexBodyModel(model, {})
    keys = ["vt_tip_w_ft", "vt_bm", "fusL_tip_w_ft", "fusL_bm", "vt_tip_twist_deg", "vt_sideslip_deg"]
    L = fb.static_probe(mdl, {"velocities/r-rad_sec": 0.1})      # yaw rate nose right: fin side load toward +y
    assert _sgn(L, keys, 1e-3) == {"vt_tip_w_ft": 1, "vt_bm": 1, "fusL_tip_w_ft": 1, "fusL_bm": 1, "vt_tip_twist_deg": 1,
                                   "vt_sideslip_deg": -1}
    L = fb.static_probe(mdl, {"aero/beta-rad": 0.05})            # wind from the right: fin load toward -y
    assert _sgn(L, keys, 1e-3) == {"vt_tip_w_ft": -1, "vt_bm": -1, "fusL_tip_w_ft": -1, "fusL_bm": -1,
                                   "vt_tip_twist_deg": -1, "vt_sideslip_deg": 1}
    L = fb.static_probe(mdl, {"accelerations/rdot-rad_sec2": 1.0})   # tail inertia toward +y, fin CG aft of EA
    assert _sgn(L, ["vt_tip_w_ft", "fusL_tip_w_ft", "vt_tip_twist_deg"], 1e-3) == {"vt_tip_w_ft": 1, "fusL_tip_w_ft": 1,
                                                                                   "vt_tip_twist_deg": -1}


@pytest.mark.parametrize("model", MODELS)
def test_sign_probe_inplane(model):
    """Wing in-plane: drag (aero force aft) -> wing*_ip_bm > 0 and wing*_tip_ip_ft > 0 (+ = aft)."""
    mdl = fb.FlexBodyModel(model, {})
    L = fb.static_probe(mdl, {"forces/fbx-aero-lbs": -100.0})
    assert L["wingR_ip_bm"] > 0 and L["wingL_ip_bm"] > 0
    assert L["wingR_tip_ip_ft"] > 1e-6 and L["wingL_tip_ip_ft"] > 1e-6
    L = fb.static_probe(mdl, {"accelerations/Nz": 1.0})          # no in-plane load -> no in-plane output
    assert abs(L["wingR_tip_ip_ft"]) < 1e-9 and L["wingR_ip_bm"] == 0.0


@pytest.mark.parametrize("model", MODELS)
def test_sign_probe_feedback_axes(model):
    """dL + up (written as body x = dL sin a, z = -dL cos a; JSBSim body z is DOWN), dY + = body +y, dRoll/dPitch/dYaw
    = JSBSim body l/m/n (+ right wing down / nose up / nose right), moments about the AERORP (arms l_h, l_v, z_v)."""
    mdl = fb.FlexBodyModel(model, {})
    a = 0.1
    L, er = fb.mode_probe(mdl, "htR", "t", a)                    # HT LE up -> tail lift up -> nose down, roll left
    assert L["dL_lbf"] > 0 and L["dPitch_lbft"] < 0 and L["dRoll_lbft"] < 0
    assert er["z"] == -L["dL_lbf"] * math.cos(a) and er["x"] == L["dL_lbf"] * math.sin(a) and er["z"] < 0
    assert er["m"] == L["dPitch_lbft"] and er["l"] == L["dRoll_lbft"]
    xs = mdl.st["x"][mdl.body_strips["htR"]]
    assert xs.min() - 1e-9 <= -L["dPitch_lbft"] / L["dL_lbf"] <= xs.max() + 1e-9     # arm measured from the AERORP
    L, er = fb.mode_probe(mdl, "vt", "t", a)                     # fin LE toward +y -> side force +y, nose left
    assert L["dY_lbf"] > 0 and er["y"] == L["dY_lbf"] and L["dYaw_lbft"] < 0 and er["n"] == L["dYaw_lbft"]
    assert L["dRoll_lbft"] > 0                                   # fin AC above the AERORP
    L, _ = fb.mode_probe(mdl, "wingR", "t", a)                   # right wing LE up -> lift up, roll left
    assert L["dL_lbf"] > 0 and L["dRoll_lbft"] < 0


def test_external_reactions_location_is_aerorp():
    import re
    for m in MODELS:
        txt = open(os.path.join(fb.ROOT_V2, "aircraft", m, m + ".xml"), encoding="utf-8", errors="replace").read()
        num = r"\s*<x>\s*([-\d.]+)\s*</x>\s*<y>\s*([-\d.]+)\s*</y>\s*<z>\s*([-\d.]+)\s*</z>"
        rp = re.search(r'<location name="AERORP" unit="IN">' + num, txt)
        fl = re.search(r'name="flexwing_F" frame="BODY">\s*<location unit="IN">' + num, txt)
        assert rp and fl and [float(v) for v in rp.groups()] == [float(v) for v in fl.groups()]
        assert re.search(r'name="flexwing_M" frame="BODY"', txt)


def test_telemetry_fields_nodes_record_only(sim):
    """New scalars are in coupler.last every frame; their histories + FE node exports only with record=True; node tip
    values == the tip scalars; layout coordinates finite; recording changes neither cost nor any term / margin."""
    scs = _short(sim, "c172x", T=2.0, n=1)
    u = {"wing_asym_ei_delta": 0.05}
    r = fe.evaluate(_gains("c172x"), u, scs, "c172x", fidelity="full", root=ROOT, record=True)
    r0 = fe.evaluate(_gains("c172x"), u, scs, "c172x", fidelity="full", root=ROOT)
    assert r0["cost"] == r["cost"] and r0["terms"] == r["terms"] and r0["margins"] == r["margins"]
    assert "telemetry" not in r0
    tel = r["telemetry"][0]
    st, nd = tel["structure"], tel["nodes"]
    for k in fb.TELEMETRY_KEYS_V2:
        assert len(st[k]) == 240 and np.all(np.isfinite(st[k]))
        assert "flex." + k in tel["trajectory"]["channels"]
    V = {b: {f: np.asarray(a) for f, a in fv.items()} for b, fv in nd["values"].items()}
    assert set(V) == set(fb.NODE_BODIES)
    pairs = [("tip_w_ft_R", "wingR", "w_ft"), ("tip_w_ft_L", "wingL", "w_ft"), ("tip_twist_R_deg", "wingR", "theta_deg"),
             ("tip_twist_L_deg", "wingL", "theta_deg"), ("ht_tip_w_ft", "htR", "w_ft"), ("htL_tip_w_ft", "htL", "w_ft"),
             ("ht_tip_twist_deg", "htR", "theta_deg"), ("htL_tip_twist_deg", "htL", "theta_deg"), ("vt_tip_w_ft", "vt", "w_ft"),
             ("vt_tip_twist_deg", "vt", "theta_deg"), ("fusV_tip_w_ft", "fusV", "w_ft"), ("fusL_tip_w_ft", "fusL", "w_ft"),
             ("wingR_tip_ip_ft", "wingR", "v_ft"), ("wingL_tip_ip_ft", "wingL", "v_ft")]
    for k, b, f in pairs:
        assert V[b][f].shape == (240, len(nd["components"][fb.NODE_BODIES.index(b)]["axis_nodes_body_ft"]))
        assert np.allclose(V[b][f][:, -1], st[k], rtol=1e-7, atol=1e-9), k
        assert np.all(V[b][f][:, 0] == 0.0)                      # clamped root
    assert not np.allclose(st["htL_tip_w_ft"], st["ht_tip_w_ft"])  # separate left HT DOFs
    for c in nd["components"]:
        xyz = np.asarray(c["axis_nodes_body_ft"])
        assert np.all(np.isfinite(xyz)) and c["node_span_frac"][0] == 0.0 and c["node_span_frac"][-1] == pytest.approx(1.0)
    comp = {c["name"]: np.asarray(c["axis_nodes_body_ft"]) for c in nd["components"]}
    assert np.all(comp["wingR"][1:, 1] > 0) and np.all(comp["wingL"][1:, 1] < 0)   # body y right
    assert comp["vt"][-1, 2] < comp["vt"][0, 2]                                    # fin tip above root (z down)
    assert comp["fusV"][-1, 0] < comp["fusV"][0, 0]                                # tail aft (x fwd)


def test_telemetry_keys_not_in_hot_path_histories():
    mdl = fb.FlexBodyModel("c172x", {})
    px = fb.FlexBodyFDM(None, fb.FlexBodyCoupler(mdl), 1 / 120)
    assert list(px.hist) == fb.DIAG_KEYS_V2 and px.telemetry is False
    assert not set(fb.TELEMETRY_KEYS_V2) & set(fb.DIAG_KEYS_V2)


# ------------------------------------------------------------------- mass-exploit fix: min gauge, sizing, margins
def test_min_gauge_mass_floor():
    assert fb.gauge_mass_factor(1.0) == 1.0
    assert fb.gauge_mass_factor(0.6) == pytest.approx(0.8) and fb.gauge_mass_factor(2.0) == pytest.approx(1.5)
    m0 = fb.FlexBodyModel("c172x", {})
    assert all(v == 0.0 for k, v in m0.mass_summary().items() if k.endswith("_lb") and k != "baseline_flexible_lb")
    m = fb.FlexBodyModel("c172x", {"fuselage_stiffness_scale": 0.6, "tail_stiffness_scale": 0.6})
    f = fb.V2_PROFILES["c172x"]["fus"]
    assert m.fusV.dmass_lb == pytest.approx(f["aft_mass_lb"] * f["struct_frac"] * -0.2)   # was -0.4 without the floor
    sp = m.ht_spec
    assert m.htR.dmass_lb == pytest.approx(sp.mass_lb * sp.struct_frac * -0.2)
    assert m.fusV.m == pytest.approx(f["aft_mass_lb"] * fb.LB2SLUG / m.l_h * (1 - f["struct_frac"] * 0.2))
    assert fb.v2_params("c172x", m.geom)["min_gauge"] == fb.MIN_GAUGE


@pytest.mark.parametrize("model", MODELS)
def test_sizing_binding_allowables(model):
    """Baseline: every design ratio == 1 and every sizing term 0. Each stiffness gene at its 0.6 floor makes its body's
    sizing term outweigh that body's mass credit; raising a gene earns no sizing reward."""
    w = fb.StructWeightsV2()
    base = fb.FlexBodyModel(model, {})
    sz = fb.sizing_v2(base, w)
    assert all(abs(r - 1.0) < 1e-12 for r in sz["ratios"].values())
    assert all(v == 0.0 for v in sz["terms"].values())
    for gene, term, keys in (("fuselage_stiffness_scale", "J_fus_bm_limit", ("fusV", "fusL")),
                             ("tail_stiffness_scale", "J_tail_bm_limit", ("ht", "vt")),
                             ("wing_ei_root", "J_wing_bm_limit", ("wingR_bm", "wingL_bm"))):
        mdl = fb.FlexBodyModel(model, {gene: 0.6})
        s = fb.sizing_v2(mdl, w)
        dJm = w.w_mass * mdl.mass_summary()["total_frac"]           # negative: the mass credit
        assert s["terms"][term] > 0 and s["terms"][term] + dJm > 0, (gene, s["terms"][term], dJm)
        assert all(s["ratios"][k] > 1.5 for k in keys)
        hi = fb.sizing_v2(fb.FlexBodyModel(model, {gene: 1.8}), w)
        assert hi["terms"][term] == 0.0
    s = fb.sizing_v2(fb.FlexBodyModel(model, {"wing_ei_root": 0.6}), w)
    assert s["terms"]["J_wing_torque_limit"] > 0 and s["terms"]["J_wing_ip_limit"] > 0
    s = fb.sizing_v2(fb.FlexBodyModel(model, {"wing_gj_ratio_root": 0.8}), w)
    assert s["terms"]["J_wing_torque_limit"] > 0
    assert s["terms"]["J_wing_bm_limit"] < 1e-4      # only the (lighter-wing) inertia-relief loss raises bending demand
    d = fb.wing_design_loads(base, w)["R"]
    assert d["torque"] > 0 and d["ip"] >= w.ip_floor_frac * d["bm"] - 1e-9          # floors
    assert d["tip_bm"] > 0 and "tip_bm" in d
    soft = fb.sizing_v2(fb.FlexBodyModel(model, {"wing_ei_taper_4": 0.75}), w)
    assert soft["terms"]["J_wing_tip_bm_limit"] > 0
    assert soft["ratios"]["wingR_tip_bm"] > 1.0
    # root BM term stays ~0 (root EI unchanged); tip term is the one that bites
    assert soft["terms"]["J_wing_bm_limit"] < soft["terms"]["J_wing_tip_bm_limit"]


def test_nsm_floor_and_tip_bm_term():
    """P2.5: nsm genes cannot go below 1.0; soft tip fires J_wing_tip_bm_limit; baseline tip term is 0."""
    for name in ("wing_nsm_root", "wing_nsm_tip"):
        g = next(x for x in fb.GENES_V2 if x.name == name)
        assert g.lo == 1.0 and g.hi == 1.25 and g.scale == "log"
        try:
            fb.decode_genome_v2({name: 0.9})
            assert False, "expected ValueError"
        except ValueError:
            pass
    w = fb.StructWeightsV2()
    base = fb.sizing_v2(fb.FlexBodyModel("c172x", {}), w)
    assert base["terms"]["J_wing_tip_bm_limit"] == 0.0
    assert abs(base["ratios"]["wingR_tip_bm"] - 1.0) < 1e-12
    soft = fb.sizing_v2(fb.FlexBodyModel("c172x", {"wing_ei_taper_4": 0.75}), w)
    assert soft["terms"]["J_wing_tip_bm_limit"] > 0.01
    # heavier nsm (only direction still allowed) adds inertia relief -> demand drops, tip term stays 0
    heavy = fb.sizing_v2(fb.FlexBodyModel("c172x", {"wing_nsm_root": 1.25, "wing_nsm_tip": 1.25}), w)
    assert heavy["terms"]["J_wing_tip_bm_limit"] == 0.0 and heavy["ratios"]["wingR_tip_bm"] <= 1.0 + 1e-9
    # cross-body: stiffer tail can load fuselage even when fuselage gene is 1.0
    cross = fb.sizing_v2(fb.FlexBodyModel("c172x", {"tail_stiffness_scale": 1.5}), w)
    assert cross["terms"]["J_fus_bm_limit"] > 0


def test_margin_terms_shaping_no_reward_above_target(monkeypatch):
    """Margin terms: hinge^2 only below margin_req = 1.2, zero at and above it (so the 3.0 not-found cap is harmless),
    hard fail below the gate (1.0 full)."""
    mdl = fb.FlexBodyModel("c172x", {})
    real = fb.margins_v2(mdl)

    def fake(m_):
        def f(*a, **k):
            d = dict(real)
            d.update(flutter_margin=m_, div_margin=3.0, reversal_margin=3.0, margin_error=False)
            return d
        return f
    vals = {}
    for m_ in (3.0, 2.0, 1.2, 1.1, 1.0, 0.999):
        monkeypatch.setattr(fb, "margins_v2", fake(m_))
        r = fb.margin_terms_v2(mdl, gate=1.0)
        vals[m_] = (r["terms"]["J_flutter_margin"], r["fail"])
    assert vals[3.0] == (0.0, None) and vals[2.0] == (0.0, None) and vals[1.2] == (0.0, None)
    assert vals[1.1][0] == pytest.approx(0.25) and vals[1.1][1] is None
    assert vals[1.0][0] == pytest.approx(1.0) and vals[1.0][1] is None
    assert vals[0.999][1] == "flutter"
    assert real["margin_cap"] == 3.0


def test_sizing_terms_in_evaluate_both_fidelities(sim):
    scs = _short(sim, "c172x", T=2.0, n=1)
    u = {"fuselage_stiffness_scale": 0.6, "tail_stiffness_scale": 0.7}
    rf = fe.evaluate(_gains("c172x"), u, scs, "c172x", fidelity="full", root=ROOT)
    rr = fe.evaluate(_gains("c172x"), u, scs, "c172x", fidelity="reduced", root=ROOT)
    for k in fb.SIZING_TERMS:
        assert rf["terms"][k] == rr["terms"][k] and k in rf["terms_available"] and k in rr["terms_available"]
    assert rf["terms"]["J_fus_bm_limit"] > 0 and rf["terms"]["J_tail_bm_limit"] > 0
    assert rf["terms"]["J_fus_bm_limit"] + rf["terms"]["J_tail_bm_limit"] + rf["terms"]["J_mass"] > 0
    for k in ("J_wing_torque_peak", "J_wing_ip_peak"):
        assert k in rf["terms_available"] and math.isfinite(rf["terms"][k])
    e = rf["per_scenario"][0]
    assert 0 < e["torque_ratio"] < 1 and 0 < e["ip_ratio"] < 1
    assert set(fe.TERM_KEYS) >= set(fb.SIZING_TERMS) and len(fe.TERM_KEYS) == len(set(fe.TERM_KEYS)) == 24
