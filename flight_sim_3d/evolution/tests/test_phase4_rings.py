import math

import numpy as np
import pytest

from evolution import phase4_eval as P
from evolution import rings as R

V = R.scales("c172x")["v_ref"]


def straight(y=0.0, z=1219.2, T=60.0, dt=0.1, v=V, **extra):
    t = np.arange(0, T + 1e-9, dt)
    pos = np.c_[v * t, np.full_like(t, y), np.full_like(t, z)]
    tr = {"t": t, "pos": pos, "att": np.zeros((len(t), 3)), "nz": np.ones_like(t), "v": np.full_like(t, v),
          "agl": np.full_like(t, 1000.0), "status": "ok"}
    tr.update(extra)
    return tr


def line_course(n=5, sp=500.0, r=20.0, z=1219.2, ys=None):
    ys = ys or [0.0] * n
    return [R.Ring((sp * (k + 1), ys[k], z), (1, 0, 0), r) for k in range(n)]


def test_exact_pass():
    out = P.score_course(straight(), line_course(), "c172x")
    assert out["n_pass"] == 5 and out["terms"]["J_ring_miss"] == 0.0 and out["terms"]["J_ring_acc"] == 0.0
    assert all(abs(x["t"] - 500.0 * (x["ring"] + 1) / V) < 1e-9 for x in out["crossings"])


def test_interpolated_crossing_between_samples():
    tr = straight(dt=1.0)
    x = R.ring_crossings(tr["t"], tr["pos"], [R.Ring((123.4, 0, 1219.2), (1, 0, 0), 5.0)])[0]
    assert x["passed"] and abs(x["t"] - 123.4 / V) < 1e-12


def test_miss_and_partial():
    out = P.score_course(straight(y=30.0), line_course(r=20.0), "c172x")
    assert out["n_pass"] == 0 and out["terms"]["J_ring_miss"] == 1.0
    assert out["terms"]["J_ring_acc"] == pytest.approx(0.5 + 0.25 * 0.5)
    part = P.score_course(straight(y=10.0), line_course(r=20.0), "c172x")
    assert part["n_pass"] == 5 and part["terms"]["J_ring_acc"] == pytest.approx(0.5 * 0.25)
    mixed = P.score_course(straight(), line_course(ys=[0, 0, 100, 0, 0]), "c172x")
    assert mixed["n_pass"] == 4 and mixed["terms"]["J_ring_miss"] == pytest.approx(0.2)


def test_order_rule_and_backwards():
    c = [R.Ring((1000, 0, 1219.2), (1, 0, 0), 20), R.Ring((500, 0, 1219.2), (1, 0, 0), 20)]
    xs = R.ring_crossings(*(lambda tr: (tr["t"], tr["pos"]))(straight()), c)
    assert xs[0]["passed"] and not xs[1]["passed"]          # ring 2 crossed before ring 1 -> does not count
    back = [R.Ring((500, 0, 1219.2), (-1, 0, 0), 20)]
    tr = straight()
    assert not R.ring_crossings(tr["t"], tr["pos"], back)[0]["passed"]


def lim():
    return {"elev": (-25.0, 20.0, 60.0)}


def test_chatter_and_saturation():
    t = np.arange(0, 60.0001, 0.1)
    smooth = P.surface_terms(t, {"elev": 5 * np.sin(2 * np.pi * 0.05 * t)}, lim())
    chat = P.surface_terms(t, {"elev": 5 * np.sign(np.sin(2 * np.pi * 2.0 * t))}, lim())
    assert smooth["J_chatter"] < 0.2 and chat["J_chatter"] == 1.0 and chat["J_rate_rms"] > smooth["J_rate_rms"]
    sat = P.surface_terms(t, {"elev": np.where(t < 30, 20.0, 0.0)}, lim())
    assert sat["J_sat"] == pytest.approx(0.5, abs=0.01)
    assert P.surface_terms(t, {"elev": np.zeros_like(t)}, lim())["J_sat"] == 0.0


def test_g_violation():
    tr = straight()
    tr["nz"] = np.where(tr["t"] < 30, 4.8, 1.0)       # 1 g over the 3.8 limit for half the run
    out = P.score_course(tr, line_course(), "c172x")
    assert out["terms"]["J_g"] == pytest.approx(0.5 * 1.0 / 3.8, rel=0.01)
    assert out["cost"] == pytest.approx(P.WEIGHTS["J_g"] * out["terms"]["J_g"] + P.WEIGHTS["J_time"] * out["terms"]["J_time"])


def test_overspeed_not_rewarded():
    slow, fast = straight(), straight(v=1.6 * V)
    a, b = P.score_course(slow, line_course(), "c172x"), P.score_course(fast, line_course(), "c172x")
    assert a["terms"]["J_time"] == 0.0 and b["terms"]["J_time"] == 0.0 and b["cost"] > a["cost"]


@pytest.mark.parametrize("kind", ["ground", "struct", "nan"])
def test_hard_fail(kind):
    tr = straight()
    if kind == "ground":
        tr["agl"] = np.where(tr["t"] > 20, -1.0, 100.0)
    elif kind == "struct":
        tr["struct_failed"] = True
    else:
        tr["pos"] = tr["pos"].copy(); tr["pos"][100, 0] = np.nan
    out = P.score_course(tr, line_course(), "c172x")
    assert out["hard_fail"] and out["cost"] >= P.HARD_FAIL_BASE


def test_curriculum_stage():
    assert R.curriculum_stage(0, 60) == "easy"
    assert R.curriculum_stage(20, 60) == "medium" and R.curriculum_stage(59, 60) == "hard"
    assert R.curriculum_stage(5, 60, [0.9] * 5) == "hard"   # promoted twice by pass rate
    assert R.curriculum_stage(5, 60, [0.9, 0.5, 0.9, 0.5, 0.9]) == "easy"


def test_multicourse_determinism():
    seeds = [R.course_seed(1, 3, k) for k in range(4)]
    assert seeds == [R.course_seed(1, 3, k) for k in range(4)] and len(set(seeds)) == 4
    assert R.course_seed(1, 3, 0, holdout=True) != seeds[0]
    cs = [R.make_course("T38", "medium", s) for s in seeds]
    assert cs[0] == R.make_course("T38", "medium", seeds[0])
    tr = straight()
    res = [P.score_course(tr, line_course(ys=[0, 0, 15 * k, 0, 0]), "c172x") for k in range(4)]
    a, b = P.aggregate_courses(res), P.aggregate_courses(res[::-1])
    assert a == b and a["cvar"] >= a["mean"] and a["cost"] == pytest.approx(0.7 * a["mean"] + 0.3 * a["cvar"])


def test_phase4_operator_trace_matches_genome():
    import importlib.util, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("p4x", os.path.join(here, "analysis", "p4_operator_crosscheck.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    if not os.path.exists(m.TRACE):
        pytest.skip("genome trace not present")
    r = m.check()
    assert r["all_match"] and r["final_pop_sha256"].startswith("7e5c38ba") and r["final_cost_sha256"].startswith("78c01f82")


def test_phase4_decode_identity():
    import json, os
    from evolution import phase4_ga as G
    tr = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "genome", "runs", "p4_operator_trace.json")
    if not os.path.exists(tr):
        pytest.skip("genome trace not present")
    d = G.decode(json.load(open(tr))["identity_u"])
    for name, _, lo, hi, sc, default in G.GENES:
        assert d[name] == pytest.approx(default, rel=1e-9, abs=1e-12), name
