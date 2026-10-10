"""pytest -q test_ctrlsurf_p4.py  (Phase 4 control-surface fidelity 'full_a1_b2a_cs')"""
import json, math, os, subprocess, sys
import numpy as np
import pytest
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import coupled_sim as cs, flexbody as fb, flexeval as fe, flexeval_b2 as fb2, flexeval_p4 as fp4, ctrlsurf_p4 as cs4
import flexbody_b2 as b2
MODELS = ("c172x", "T38", "737", "f16")
ROOT = cs.ROOT
fb.blas_threads(1)


@pytest.fixture(scope="module")
def sim():
    for m in MODELS:
        cs.ensure_root(m); fb.ensure_root_v2(m); b2.ensure_root_v2b2(m, fe.root_v2_for(ROOT), fb2.root_v2b2_for(ROOT))
    return fe.load_sim()


def _gains(m):
    S = json.load(open(os.path.join(fe.TEAM, "evolution", "runs", "phase1-s1", "summary.json")))["aircraft"]
    return next(a["best_gains"] for a in S if a["aircraft"] == m)


def _short(sim, m, T=2.0):
    s = fe.phase1_scenarios(m, sim)[:1]
    for x in s: x.duration_s = T
    return s


DIFF = {"wall_s", "model_version", "fidelity", "cpu_s", "sim_wall_s", "sim_cpu_s"}


def _eq(x, y):
    if isinstance(x, dict):
        return all(k in y and _eq(v, y[k]) for k, v in x.items() if k not in DIFF)
    if isinstance(x, list):
        return len(x) == len(y) and all(_eq(u, v) for u, v in zip(x, y))
    if isinstance(x, float) and math.isnan(x):
        return isinstance(y, float) and math.isnan(y)
    return x == y


@pytest.mark.parametrize("model", MODELS)
def test_passthrough_bit_identical_to_b2a(sim, model):
    g, sc = _gains(model), _short(sim, model)
    a = fb2.evaluate(g, {}, sc, model, fidelity="full_a1_b2a", root=ROOT, sim=sim, record=True)
    p = fp4.evaluate(g, {}, sc, model, fidelity="full_a1_b2a_cs", root=ROOT, sim=sim, record=True, cs_mode="passthrough")
    assert _eq(a, p) and p["cost"] == a["cost"]
    assert p["model_version_b2a"] == a["model_version"] and p["model_version"].startswith("full_a1_b2a_cs:p4cs0:passthrough:")
    assert "ctrl_surfaces" in p["telemetry"][0]


def test_passthrough_shaped_bit_identical(sim):
    m, sg = "c172x", {"wing_dihedral_delta_deg": 2.0, "wing_camber_root_delta_pct": 1.0}
    g, sc = _gains(m), _short(sim, m)
    a = fb2.evaluate(g, {"wing_ei_root": 1.3}, sc, m, fidelity="full_a1_b2a", root=ROOT, sim=sim, shape_genome=sg)
    p = fp4.evaluate(g, {"wing_ei_root": 1.3}, sc, m, fidelity="full_a1_b2a_cs", root=ROOT, sim=sim, shape_genome=sg,
                     cs_mode="passthrough")
    assert _eq(a, p)


def test_delegation_other_fidelities(sim):
    g, sc = _gains("c172x"), _short(sim, "c172x", 1.0)
    a = fb2.evaluate(g, {}, sc, "c172x", fidelity="full_a1_b1", root=ROOT, sim=sim)
    b = fp4.evaluate(g, {}, sc, "c172x", fidelity="full_a1_b1", root=ROOT, sim=sim)
    assert _eq(a, b) and "ctrl_surfaces" not in b
    assert fb2.FlexHookB2.__name__ == "FlexHookB2"


@pytest.mark.parametrize("model", ("c172x", "f16"))
def test_active_deterministic_and_differs(sim, model):
    g, sc = _gains(model), _short(sim, model)
    r1 = fp4.evaluate(g, {}, sc, model, fidelity="full_a1_b2a_cs", root=ROOT, sim=sim, record=True)
    r2 = fp4.evaluate(g, {}, sc, model, fidelity="full_a1_b2a_cs", root=ROOT, sim=sim, record=True)
    assert _eq(r1, r2) and r1["status"] == "ok"
    a = fb2.evaluate(g, {}, sc, model, fidelity="full_a1_b2a", root=ROOT, sim=sim)
    assert r1["cost"] != a["cost"]
    t = r1["telemetry"][0]["ctrl_surfaces"]
    n = len(t["t_s"])
    for k in ("elev_cmd_norm", "elev_act_norm", "elev_deg", "elev_rate_dps", "elev_hinge_lbft", "elev_eta", "ail_deg",
              "rud_deg", "ail_pos_sat", "ail_rate_sat"):
        assert len(t[k]) == n
    assert set(r1["terms"]) == set(fe.TERM_KEYS)


def test_actuator_rate_limit():
    s = cs4.SURFACES["c172x"][0]
    a = cs4.Actuator(s, 1 / 120, 0.0)
    ys = [a.step(1.0) for _ in range(240)]
    d = np.diff([0.0] + ys) * s.deg_per_norm * 120
    assert d.max() <= s.rate_dps + 1e-9 and abs(ys[-1] - 1.0) < 1e-6 and a.pos_sat


def test_actuator_lag_small_step():
    s = cs4.SURFACES["737"][0]
    a = cs4.Actuator(s, 1 / 120, 0.0)
    u = 0.01                                     # small: rate limit inactive
    n = 10
    ys = [a.step(u) for _ in range(n)]
    assert not a.rate_sat
    assert abs(ys[-1] / u - (1 - math.exp(-n / 120 / s.tau_s))) < 1e-9


def test_hinge_blowdown():
    s = cs4.SURFACES["c172x"][0]
    a = cs4.Actuator(s, 1 / 120, 0.0)
    for _ in range(200): y = a.step(1.0, u_hinge_max=0.3)
    assert abs(y - 0.3) < 1e-12 and a.hinge_sat


@pytest.mark.parametrize("model", MODELS)
def test_eta_reversal_trend_handcalc(model):
    obj = b2.FlexBodyModelB2(model, {}, shape_genes=None, asymmetric=False, root_v2=fb.ROOT_V2)
    T = cs4.eta_tables(obj)["ail"]
    e, q = T["eta"], T["q_psf"]
    assert e[0] == 1.0 and np.all(np.diff(e) <= 1e-9)            # stiffness loss: monotone decreasing in q
    # soft torsion (GJ x 0.5): eta lower at every q (hand calc: e ~ (1 - q/qR)/(1 - q/qD), qR ~ GJ)
    soft = b2.FlexBodyModelB2(model, {"wing_gj_ratio_root": 0.8, "wing_gj_ratio_tip": 0.8}, shape_genes=None, asymmetric=False, root_v2=fb.ROOT_V2)
    es = cs4.eta_tables(soft)["ail"]["eta"]
    assert np.all(es[1:] <= e[1:] + 1e-9)
    # 2-D typical-section fit through the margin-screen q_R reproduces the tabulated curve within 0.1 below 0.5 q_D
    m = fb.margins_v2(obj, blocks=["wingR"])["blocks"]["wingR"]
    eVD = m["aileron_effectiveness_at_VD"]
    qd = fb._q_of_keas(obj.pw.v_dive_keas)
    assert abs(np.interp(qd, q, e) - eVD) < 0.02


def test_limits_file():
    d = json.load(open(os.path.join(HERE, "v2_results", "p4_aircraft_limits.json")))["aircraft"]
    for m in MODELS:
        a = d[m]
        assert 1 < a["n_sus_est"] <= a["n_inst"] <= a["n_struct_limit"] + 1e-9
        assert a["R_inst_ft"] <= a["R_sus_ft_est"] and a["roll_rate_max_dps"] > 10


def test_frozen_md5_unchanged():
    for f in ("FROZEN_A1_B1r1.md5", "FROZEN_B2a.md5"):
        r = subprocess.run(["md5sum", "-c", "--quiet", os.path.join("v2_results", f)], cwd=HERE, capture_output=True, text=True)
        assert "FAILED" not in r.stdout + r.stderr


@pytest.mark.parametrize("model", ("c172x", "737"))
def test_fly_course_contract_and_determinism(sim, model):
    P = fe.default_profile(model, sim)
    c = {"start": {"alt_ft": P.h0_ft, "kcas": P.speed_kts}, "duration_s": 4.0, "demo_heading_deg": 30}
    r1 = fp4.fly_course(None, {}, {}, None, None, c, model=model, guidance=fp4.demo_guidance, sim=sim)
    r2 = fp4.fly_course(None, {}, {}, None, None, c, model=model, guidance=fp4.demo_guidance, sim=sim)
    assert _eq(r1, r2) and r1["status"] == "ok" and r1["struct_failed"] is False
    for k in ("t", "pos", "att", "v_kcas", "nz", "alpha", "agl", "surfaces", "surface_limits", "terms", "energy", "limits"):
        assert k in r1
    assert set(r1["terms"]) == set(fe.TERM_KEYS) and len(r1["pos"][0]) == 3 and len(r1["t"]) == len(r1["surfaces"]["ail"])
    assert r1["att"][-1][2] > 0.0                                   # turned right toward +30 deg
    rp = fp4.fly_course(None, {}, {}, None, None, c, model=model, guidance=fp4.demo_guidance, sim=sim, cs_mode="passthrough")
    assert rp["model_version"] != r1["model_version"] and rp["status"] == "ok"
