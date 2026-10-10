"""phase4_rings chromosome: schema, counts, identity, ranges, operator determinism; existing goldens unchanged. No flights."""
import json
import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
import phase4_rings as R  # noqa: E402
import p4_operator_trace as T  # noqa: E402
from flightsim_path import orig_ga  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P4_FINAL = ("7e5c38baf2dac5f02200b2767e6294b07fc43fcab86011e887978e462bcb5974",
            "78c01f8246034e6a8b71e55af1a1f4a6a18c95ab3f1d425573a70822b26acf70")


@pytest.fixture(scope="module")
def P():
    return R.load_preset()


def test_schema_and_counts(P):
    assert len(P["genes"]) == 29
    assert {k: len(v) for k, v in P["blocks"].items()} == {"guidance": 12, "inner_loop": 11, "mixing": 6}
    names = [g.name for g in P["genes"]]
    assert len(set(names)) == 29
    assert P["elite"] == 2 and P["selection_p"] == 0.2 and P["mutation_rate"] == 0.15 and P["mutation_sigma"] == 0.08
    raw = P["raw"]
    assert raw["fitness"]["params"]["struct_v2_source"] == "fd" and raw["carry_over"] == "none"


def test_identity_decode_and_ranges(P):
    u = R.identity_u(P["genes"])
    assert ((u >= 0) & (u <= 1)).all()
    d = R.decode(u, P["genes"])
    for g in P["genes"]:
        assert d[g.name] == pytest.approx(g.default, rel=1e-12, abs=1e-12)
        assert g.min <= g.default <= g.max or (g.scale == "log0" and g.default == 0.0)
        assert g.decode(0.0) == pytest.approx(0.0 if g.scale == "log0" else g.min)
        assert g.decode(1.0) == pytest.approx(g.max)
    t = {g.name: g for g in P["genes"]}
    assert (t["tau_cmd_s"].min, t["tau_cmd_s"].max, t["tau_cmd_s"].scale, t["tau_cmd_s"].default) == (0.05, 0.5, "log", 0.1)
    assert not {"bank_max_deg", "nz_max_g", "nz_min_g"} & set(t)


def test_inner_loop_matches_existing_controller_genes(P):
    import genome_schema as GS
    src = GS.ALL_GENES
    for g in P["genes"]:
        if g.block == "inner_loop":
            s = src[g.name]
            assert (g.min, g.max, g.scale, g.default) == (s.min, s.max, s.scale, s.default)


def test_flat_rank_select_is_ga(P):
    G = orig_ga()
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    assert [R.flat_rank_select(a, 64, 0.2) for _ in range(200)] == [G.flat_rank_select(b, 64, 0.2) for _ in range(200)]


def test_crossover_whole_blocks(P):
    rng = np.random.default_rng(3)
    a, b = np.zeros(29), np.ones(29)
    for _ in range(50):
        rec = {}
        c = R.block_ops.block_crossover(rng, a, b, P["ops"], rec)
        for (blk, idx), pa in zip(P["blocks"].items(), rec["blk"]):
            assert (c[idx] == (0.0 if pa else 1.0)).all()


def test_operator_trace_deterministic_and_golden(P):
    _, o1 = T.run()
    _, o2 = T.run()
    assert o1 == o2
    last = o1["generations_detail"][-1]
    assert (last["pop_sha256"], last["cost_sha256"]) == P4_FINAL
    assert all(o1["elite_preserved"])
    saved = json.load(open(os.path.join(HERE, "runs", "p4_operator_trace.json")))
    assert saved["generations_detail"] == o1["generations_detail"]


def test_existing_presets_untouched():
    import adapter
    with pytest.raises(Exception):
        adapter.load_task("phase4_rings")   # not routed through adapter by design
    from test_phase3_b1_x import GOLDEN_FINAL, _trace
    for name, (ps, cs) in GOLDEN_FINAL.items():   # (gen0 unranked sha, final ranked pop sha), pre-phase3_b1_x golden
        t = adapter.load_task(name)
        gens = _trace(t, 2)[0]
        assert gens[0]["gen0_unranked_sha256"] == ps and gens[-1]["pop_sha256"] == cs


# ------------------------------------------------------------------ per-aircraft limits / guidance (no flight)
import p4_guidance as G  # noqa: E402

LIM = {"c172x": (60.0, 3.559838903903134, -1.0), "T38": (75.0, 4.197637884847775, -3.0),
       "737": (60.0, 2.5, -1.0), "f16": (80.0, 8.793068117992789, -3.0)}


@pytest.mark.parametrize("model", sorted(LIM))
def test_limits_are_fractions_of_fd_file(P, model):
    bank, ninst, nlo = LIM[model]
    a = G.aircraft_limits(model)
    assert (a["bank_course_deg"], a["n_inst"], a["n_profile"][0]) == (bank, ninst, nlo)
    for u in (np.zeros(29), np.ones(29), R.identity_u(P["genes"])):
        d = G.decode_physical(u, P["genes"], model)
        assert 0 < d["bank_max_deg"] <= bank and d["nz_max_g"] <= ninst and nlo <= d["nz_min_g"] <= 1.0
    d = G.decode_physical(np.ones(29), P["genes"], model)
    assert d["bank_max_deg"] == bank and d["nz_max_g"] == pytest.approx(ninst)
    assert model != "737" or d["nz_max_g"] <= 2.5


def test_gain_scales_fixed_constants():
    assert G.gain_scales("c172x") == {"roll": 1.0, "pitch": 1.0, "speed": 1.0, "n_inst": LIM["c172x"][1]}
    s = G.gain_scales("737")
    assert s["roll"] > 1 and s["pitch"] > 1 and s["speed"] < 1
    import inspect
    src = inspect.getsource(G.make_guidance)
    assert '["eta' not in src and "flex" not in src     # never divide by FD's flex eta


def _state(**k):
    s = {"t": 0.0, "pos_m": [0.0, 0.0, 1219.2], "phi": 0.0, "theta": 0.0, "psi": 0.0, "p": 0.0, "q": 0.0, "r": 0.0,
         "vc_kts": 100.0, "vt_fps": 179.0, "nz": 1.0, "alpha": 0.0, "beta": 0.0, "agl_m": 1219.2, "h_dot_fps": 0.0}
    s.update(k)
    return s


def test_guidance_signs_and_determinism(P):
    RC = G._sb()
    course = RC.make_course("c172x", "easy", 7)
    g = G.decode_physical(R.identity_u(P["genes"]), P["genes"], "c172x")
    c0 = RC.ring_at(course, 0)["centre_m"]
    # put the aircraft left of / below ring 0 -> right bank (+ail), nose up (-elev)
    st = _state(pos_m=[c0[0] - 500.0, c0[1] - 200.0, -c0[2] - 100.0])
    outs = []
    for _ in range(2):
        f = G.make_guidance(g, "c172x", course)
        o = [f(dict(st, t=k * G.DT)) for k in range(60)]
        outs.append(o)
    assert outs[0] == outs[1]
    last = outs[0][-1]
    assert last["aileron"] > 0 and last["elevator"] < 0 and 0 <= last["throttle"] <= 1
    assert set(last) == {"aileron", "elevator", "rudder", "throttle"}


def test_trace_covers_4_aircraft():
    d = json.load(open(os.path.join(HERE, "runs", "p4_operator_trace.json")))
    assert sorted(d["identity_physical"]) == ["737", "T38", "c172x", "f16"]
    assert d["identity_physical"]["f16"]["bank_max_deg"] == 60.0 and d["identity_physical"]["737"]["nz_max_g"] == 1.75


def test_cache_key_includes_stage_seeds_K(P):
    u = R.identity_u(P["genes"])
    k = G.cache_key(u, "c172x", "easy", [1, 2, 3, 4], 4, "mv")
    assert k != G.cache_key(u, "c172x", "medium", [1, 2, 3, 4], 4, "mv")
    assert k != G.cache_key(u, "c172x", "easy", [1, 2, 3, 5], 4, "mv")
    assert k != G.cache_key(u, "c172x", "easy", [1, 2, 3, 4], 3, "mv")
    with pytest.raises(ValueError):
        G.evaluate(u, P["genes"], "c172x", "easy", [1, 2, 3], K=4)


@pytest.mark.sim
def test_smoke_identity_c172x_one_course():
    out = G.smoke("c172x", "easy")
    assert out["status"] == "ok" and out["model_version"].startswith("full_a1_b2a_cs:p4cs0:active:")
    assert out["pass_rate"] == 1.0 and out["cost_ER_single_course"] < 0.05


# ---- f16 FBW demand mapping (FD Q-FD10) + SB v0.5 TAS geometry ----
def test_fbw_only_f16_and_constants_match_f16_xml():
    assert set(G.FBW) == {"f16"}
    import re
    xml = os.path.join(G.FD_PIN, "jsbsim_root_v2b2", "aircraft", "f16", "f16.xml")
    if not os.path.exists(xml):
        xml = os.path.join(G.FD_PIN, "..", "..", "..", "flight-dynamics", "jsbsim_root_v2b2", "aircraft", "f16", "f16.xml")
    if not os.path.exists(xml):
        pytest.skip("f16.xml not present")
    t = open(xml).read()
    kp = float(re.search(r'name="fcs/roll-rate-norm">.*?<gain>([\d.]+)</gain>', t, re.S).group(1))
    kq = float(re.search(r'name="fcs/pitch-rate-norm">.*?<gain>([\d.]+)</gain>', t, re.S).group(1))
    assert G.FBW["f16"]["p_dps_per_norm"] == pytest.approx(np.degrees(1 / kp))
    assert G.FBW["f16"]["q_dps_per_norm"] == pytest.approx(np.degrees(1 / kq))


def _one_step(P, model, **k):
    RC = G._sb()
    course = RC.make_course(model, "easy", 7)
    g = G.decode_physical(R.identity_u(P["genes"]), P["genes"], model)
    c0 = RC.ring_at(course, 0)["centre_m"]
    st = _state(pos_m=[c0[0] - 500.0, c0[1] - 200.0, -c0[2] - 100.0], **k)
    f = G.make_guidance(g, model, course)
    return [f(dict(st, t=i * G.DT)) for i in range(60)][-1], f


def test_fbw_f16_demand_scale_and_signs(P):
    o, f = _one_step(P, "f16")
    assert o["aileron"] > 0 and o["elevator"] < 0          # right-roll / nose-up demands, same signs as surfaces
    lim, ref = G.aircraft_limits("f16"), G.aircraft_limits("c172x")
    # the FBW scale is reference rate / FCS demand gain, not the surface-authority ratio
    assert ref["roll_rate_max_dps"] / G.FBW["f16"]["p_dps_per_norm"] != pytest.approx(G.gain_scales("f16")["roll"])
    assert f.state["n"] == 0


def test_leg_time_uses_tas_geometry_speed(P):
    RC = G._sb()
    c = RC.make_course("T38", "easy", 7)
    assert c["params"]["v_tas_ms"] == c["params"]["v_ref_mps"]   # SB v0.5 alias
    assert c["params"]["v_tas_ms"] > c["params"]["v_ref_kcas"] * G.KT   # TAS > CAS at 10 000 ft


# ---- leg-time timeout (SB ring_course/1.2) + phase4_rings_qs (opt-in) ----
def test_timeout_uses_rc_leg_time():
    RC = G._sb()
    if not hasattr(RC, "leg_time"):
        pytest.skip("ring_course < 1.2")
    course = RC.make_course("f16", "hard", 5)
    g = G.decode_physical(R.identity_u(R.load_preset()["genes"]), R.load_preset()["genes"], "f16")
    f = G.make_guidance(g, "f16", course)
    dl = 3.0 * RC.leg_time(course, 0)
    assert dl > 3.0 * course["params"]["spacing_m"] / course["params"]["v_tas_ms"]      # real 3-D leg is longer
    st = _state(pos_m=[-50.0, 0.0, 3048.0])
    f(dict(st, t=0.0)); f(dict(st, t=dl - 0.05)); assert f.state["n"] == 0
    f(dict(st, t=dl + 0.05)); assert f.state["n"] == 1 and f.state["log"][-1][2] == "timeout" and f.state["t_res"] == dl + 0.05


def test_qs_preset_same_layout_and_default_untouched(P):
    q = R.load_preset(os.path.join(HERE, "presets", "phase4_rings_qs.json"))
    assert [g.name for g in q["genes"]] == [g.name for g in P["genes"]] and len(q["genes"]) == 29
    assert q["raw"]["variant"] == "qs" and "variant" not in P["raw"]
    u = R.identity_u(P["genes"])
    for m in ("c172x", "T38", "737", "f16"):
        d, dq = G.decode_physical(u, P["genes"], m), G.decode_physical_qs(u, P["genes"], m)
        changed = {k for k in d if d[k] != dq[k]}
        assert changed <= set(G.QS_KEYS)
        if m == "c172x":
            assert not changed                              # reference aircraft keeps its decode
        else:
            assert dq["k_lat"] > d["k_lat"] and abs(dq["k_lat"] / d["k_lat"] - G.qs_scale(m)) < 1e-12


def test_qs_scale_monotone_in_speed():
    s = {m: G.qs_scale(m) for m in ("c172x", "737", "T38", "f16")}
    assert s["c172x"] == 1.0 and 1.0 < s["737"] < s["T38"] < s["f16"]
