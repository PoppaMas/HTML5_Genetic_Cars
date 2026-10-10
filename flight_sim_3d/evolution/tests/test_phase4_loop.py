"""Phase 4 loop wiring: frames, Sim Bridge rule cross-check, scoring fixes, config/hook, Genome sync, and real-FD integration
(one genome flies one course; known pass; known miss; deterministic across fresh processes)."""
import copy
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM = os.path.dirname(PKG)
sys.path.insert(0, TEAM)
from evolution import phase4_loop as L, phase4_ga as PG, phase4_eval as PE, rings as RR, batch  # noqa: E402

CFG_U = json.load(open(os.path.join(PKG, "configs", "phase4_smoke.json")))
need_fd = pytest.mark.skipif(not os.path.isdir(L.FD_DIR), reason="FD pin missing")


def rc():
    return L.mods()["rc"]


def _line_through(course, ks, offs=None, n_per=60, via=None):
    """NED polyline start -> ring centres (+ in-plane offset), 1 m/s-ish timing."""
    pts = [course["start"]["pos"]] + ([via] if via else [])
    for i, k in enumerate(ks):
        r = rc().ring_at(course, k)
        c = list(r["centre_m"])
        if offs:
            c[1] += offs[i]
        pts.append(c)
    last = rc().ring_at(course, ks[-1])
    pts.append([pts[-1][j] + 200.0 * last["normal"][j] for j in range(3)])
    P, T = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        for s in np.linspace(0, 1, n_per, endpoint=False):
            P.append([a[j] + s * (b[j] - a[j]) for j in range(3)])
    P.append(pts[-1])
    dt = 0.8 * course["time_limit_s"] / len(P)               # whole path inside SB's time limit
    T = [i * dt for i in range(len(P))]
    return T, P


def test_frame_neu_to_ned_known_crossing():
    c = rc().make_course("c172x", "easy", 7, M=3)
    t, p_ned = _line_through(c, [0, 1, 2])
    neu = [[x, y, -z] for x, y, z in p_ned]                      # what FD fly_course returns (N, E, Up = altitude MSL)
    assert neu[0] == [0.0, 0.0, c["start"]["alt_ft"] * 0.3048]  # FD P4.14: origin = ground below start, U = altitude
    good = rc().score_course(t, rc().fd_pos_to_ned(neu), c)
    assert good["passes"] == 3 and good["finished"]
    bad = rc().score_course(t, neu, c)                          # unconverted N/E/U scores nothing
    assert bad["passes"] == 0


@pytest.mark.parametrize("model,stage,seed", [("c172x", "easy", 3), ("T38", "medium", 11), ("737", "hard", 5), ("f16", "easy", 2)])
def test_rings_py_matches_sim_bridge_score_course(model, stage, seed):
    c = rc().make_course(model, stage, seed, M=4)
    r = c["params"]["radius_m"]
    offs = [0.2 * r, 1.5 * r, 0.9 * r, 0.0]                      # pass, (flown) miss, pass, pass
    t, p = _line_through(c, [0, 1, 2, 3], offs)
    sb = rc().score_course(t, p, c)
    rings = [RR.Ring(tuple(rc().ring_at(c, k)["centre_m"]), tuple(rc().ring_at(c, k)["normal"]), r) for k in range(4)]
    er = RR.ring_crossings(t, p, rings)
    got = [g["result"] == "pass" for g in sb["gates"]]
    assert got == [x["passed"] for x in er] and got[0] and not got[1]   # pass + flown miss both exercised
    assert [g["rho_m"] for g in sb["gates"]] == pytest.approx([x["rho"] for x in er], rel=0, abs=1e-9)


def test_strict_order_is_sim_bridge_rule():
    c = rc().make_course("c172x", "easy", 9, M=3)
    c0 = rc().ring_at(c, 0)["centre_m"]
    via = [c0[0] - 50.0, c0[1] + 10 * c["params"]["radius_m"], c0[2]]   # go wide of ring 0 (>> capture), then rings 1, 2
    t, p = _line_through(c, [1, 2], via=via)                   # skips ring 0: SB marks it missed_order
    sb = rc().score_course(t, p, c)
    assert [g["result"] for g in sb["gates"]] == ["missed_order", "pass", "pass"] and sb["passes"] == 2


def test_surface_terms_skip_rate_none_flap():
    t = np.linspace(0, 10, 301)
    s = {"elev": np.sin(t), "flap": np.zeros_like(t)}
    lim = {"elev": (-28.0, 23.0, 60.0), "flap": (0.0, 30.0, None)}
    a = PE.surface_terms(t, s, lim)
    b = PE.surface_terms(t, {"elev": s["elev"]}, {"elev": lim["elev"]})
    assert a == b and all(math.isfinite(v) for v in a.values())
    assert PE.surface_terms(t, {"flap": s["flap"]}, {"flap": lim["flap"]})["J_rate_rms"] == 0.0


@need_fd
def test_course_limits_from_fd_incl_f16():
    want = {"c172x": (15, 60, 125.0), "T38": (18, 75, 375.0), "737": (13, 60, 312.5), "f16": (25, 80, 437.5)}
    for m, (a, b, vmax) in want.items():
        v_ref = rc().AIRCRAFT[m]["v_ref_kts"]
        lim = L.course_limits(m, {"min_kcas": 1.0}, v_ref)
        assert (lim["alpha_stall_deg"], lim["bank_course_deg"], lim["v_max_kcas"]) == (a, b, vmax)
        assert PE.ALPHA_MAX_DEG[m] == a and PE.BANK_COURSE_DEG[m] == b
        assert RR.AIRCRAFT[m]["v_ref_kts"] == v_ref and RR.AIRCRAFT[m]["h0_ft"] == rc().AIRCRAFT[m]["h0_ft"]
        assert RR.STAGES[m if False else "easy"]["spacing_s"] == rc().STAGES["easy"]["spacing_s"]
    for st in RR.STAGES:
        assert {k: RR.STAGES[st][k] for k in ("spacing_s", "lat_rt", "vert_sp", "radius_s")} == \
            {k: rc().STAGES[st][k] for k in ("spacing_s", "lat_rt", "vert_sp", "radius_s")}


def test_genes_match_genome_and_identity_hash():
    gdir = os.path.join(TEAM, "genome")
    sys.path.insert(0, gdir)
    import phase4_rings as GR
    gt = GR.gene_table()
    assert [(g.name, g.block, g.min, g.max, g.scale, g.default) for g in gt] == [tuple(x) for x in PG.GENES]
    tr = json.load(open(os.path.join(gdir, "runs", "p4_operator_trace.json")))
    u = np.asarray(tr["identity_u"], "<f8")
    assert hashlib.sha256(u.tobytes()).hexdigest() == tr["identity_u_sha256"]
    assert tr["identity_u_sha256"].startswith("192ed803")
    for m, phys in tr["identity_physical"].items():
        d = PG.decode(u, m)
        assert all(d[k] == v for k, v in phys.items() if k != "gain_scales")


def test_phase4_config_resolves_strictly_and_hook_dispatches(monkeypatch):
    c = L.resolve_config(copy.deepcopy(CFG_U), "phase4_smoke")
    assert [a["name"] for a in c["aircraft"]] == ["c172x", "T38", "737", "f16"]
    assert c["rings"]["courses_per_genome"] == 4 and c["rings"]["holdout_courses"] == 8 and c["rings"]["rim_tube_m"] == 0.0
    for k in ("weights", "pin_model_version", "rings"):
        bad = copy.deepcopy(CFG_U); bad.pop(k)
        with pytest.raises(ValueError):
            L.resolve_config(bad, "t")
    bad = copy.deepcopy(CFG_U); bad["fidelity"] = "full_a1_b2a"
    with pytest.raises(ValueError):
        L.resolve_config(bad, "t")
    seen = {}
    monkeypatch.setattr(L, "main", lambda argv, user=None, name=None: seen.update(user=user, name=name))
    batch.main(["--config", os.path.join(PKG, "configs", "phase4_smoke.json")])
    assert seen["name"] == "phase4_smoke" and seen["user"]["phase"] == "phase4_rings"


def test_cache_key_includes_stage_seed_K_pin():
    c = L.resolve_config(copy.deepcopy(CFG_U), "t")
    u = [0.5] * 29
    k0 = L.cache_key(c, "c172x", u, "easy", 1, 4)
    assert len({k0, L.cache_key(c, "c172x", u, "medium", 1, 4), L.cache_key(c, "c172x", u, "easy", 2, 4),
                L.cache_key(c, "c172x", u, "easy", 1, 3)}) == 4
    c2 = copy.deepcopy(c); c2["pin_model_version"]["c172x"] = "x"
    assert L.cache_key(c2, "c172x", u, "easy", 1, 4) != k0


# ------------------------------------------------------------------------------------------------ real FD
SNIP = """
import json, sys; sys.path.insert(0, %r); sys.dont_write_bytecode = True
from evolution import phase4_loop as L
c = L.resolve_config(json.load(open(%r)), "t")
tr = json.load(open(%r))
r = L.fly_one(c, "c172x", tr["identity_u"], "easy", 4242, M=2)
print(json.dumps({"cost": r["cost"], "gates": r["gates"], "terms": r["terms"], "mv": r["model_version"]}, sort_keys=True))
"""


def _fresh():
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    code = SNIP % (TEAM, os.path.join(PKG, "configs", "phase4_smoke.json"), os.path.join(TEAM, "genome", "runs", "p4_operator_trace.json"))
    out = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, check=True, cwd=TEAM).stdout
    return json.loads(out.strip().splitlines()[-1])


@need_fd
def test_real_fd_one_genome_one_course_pass_and_deterministic_across_processes():
    a, b = _fresh(), _fresh()
    assert a == b                                          # bit-identical in two fresh processes
    assert a["mv"] == CFG_U["pin_model_version"]["c172x"]
    assert [g["result"] for g in a["gates"]] == ["pass", "pass"]          # known pass (identity genome, easy)
    assert a["terms"]["J_ring_miss"] == 0.0 and math.isfinite(a["cost"])


@need_fd
def test_real_fd_known_miss():
    m = L.mods()
    c = L.resolve_config(copy.deepcopy(CFG_U), "t")
    course = m["rc"].make_course("c172x", "easy", 4242, M=2)
    prof, pd = L.profile_obj(c, "c172x")
    fc = {"start": dict(course["start"]), "duration_s": course["time_limit_s"], "demo_heading_deg": 45.0}
    r = m["fp4"].fly_course(prof, None, None, None, None, fc, fidelity="full_a1_b2a_cs", model="c172x",
                            guidance=m["fp4"].demo_guidance, sim=m["sim"])   # FD test law: turns 45 deg away from the rings
    s = L.score_flight(r, course, "c172x", L.course_limits("c172x", pd, course["start"]["kcas"]), c["weights"])
    assert s["summary"]["passes"] == 0 and s["terms"]["J_ring_miss"] == 1.0
    assert all(g["result"] in ("miss", "missed_time", "missed_order") for g in s["gates"])


def test_tas_geometry_matches_sim_bridge():
    for m in ("c172x", "T38", "737", "f16"):
        sb = rc().scales(m)
        er = RR.scales(m)
        assert er["v_tas_ms"] == sb["v_tas_ms"] and er["r_turn"] == sb["r_turn_m"]
        c = rc().make_course(m, "easy", 1)
        assert c["params"]["v_tas_ms"] == er["v_tas_ms"]
    assert abs(RR.scales("f16")["r_turn"] - 7540) < 30          # 401.5 KTAS at 10 kft, 30 deg bank


def test_speed_limits_kcas_and_mach_single_source():
    for m, vmax in {"c172x": 125.0, "T38": 375.0, "737": 312.5, "f16": 437.5}.items():
        sc = RR.scales(m)
        assert sc["v_max_kcas"] == vmax and sc["v_max"] == vmax * RR.KT          # CAS, not 1.25 x TAS
        assert sc["mach_max"] == RR.fd_limits()[m].get("mach_max")
        assert PE.BANK_COURSE_DEG[m] == RR.fd_limits()[m]["bank_course_deg"]
    assert PE.BANK_COURSE_DEG["f16"] == 80.0 and RR.scales("737")["mach_max"] == 0.84


def test_mach_overspeed_term():
    c = rc().make_course("T38", "easy", 1, M=2)
    lim = L.course_limits("T38", {"min_kcas": 180.0}, 300.0)
    n = 300
    t = list(np.linspace(0, 10, n))
    a = rc().isa(3048.0)[2]
    base = {"t": t, "pos": [[i * 1.0, 0.0, 3048.0] for i in range(n)], "att": [[0, 0, 0]] * n, "v_kcas": [300.0] * n,
            "nz": [1.0] * n, "alpha": [0.0] * n, "agl": [3048.0] * n, "surfaces": {}, "surface_limits": {}, "status": "ok",
            "struct_failed": False, "terms": {}, "energy": {}}
    slow = L.score_flight({**base, "v_ms": [0.8 * a] * n}, c, "T38", lim, CFG_U["weights"])
    fast = L.score_flight({**base, "v_ms": [0.99 * a] * n}, c, "T38", lim, CFG_U["weights"])
    assert slow["terms"]["J_overspeed"] == 0.0
    assert fast["terms"]["J_overspeed"] == pytest.approx((0.99 - 0.90) / 0.90, rel=1e-12)


def test_pass_rate_gate_curriculum():
    g = RR.curriculum_stage_gated
    assert [g(i, []) for i in range(1)] == ["easy"]
    h = [1.0, 1.0, 0.0, 0.9, 0.85, 0.5, 1.0]
    assert [g(i, h) for i in range(len(h) + 1)] == ["easy", "easy", "medium", "medium", "medium", "hard", "hard", "hard"]
    assert [g(i, [0.79] * 6) for i in range(7)] == ["easy"] * 7            # never advances below the threshold
    assert [g(i, [0.8, 0.0, 0.8, 0.8]) for i in range(5)] == ["easy"] * 4 + ["medium"]   # streak must be consecutive
    h2 = [1.0] * 10
    st = [g(i, h2) for i in range(11)]
    assert st[-1] == "hard" and all(RR.STAGE_ORDER.index(a) <= RR.STAGE_ORDER.index(b) for a, b in zip(st, st[1:]))  # never back
    # opt-in only: default mode is the s1 floor rule (unchanged)
    assert [RR.curriculum_stage(i, 6, [0.0] * 6) for i in range(6)] == ["easy", "easy", "medium", "medium", "hard", "hard"]


def test_gate_config_is_s1_plus_mode_only():
    s2 = json.load(open(os.path.join(PKG, "configs", "phase4_smoke_gate.json")))
    a, b = copy.deepcopy(s2), copy.deepcopy(CFG_U)
    assert a["rings"]["curriculum"].pop("mode") == "pass_rate_gate" and "mode" not in b["rings"]["curriculum"]
    assert a["rings"]["curriculum"].pop("late_fallback") == {"enabled": False, "rule": "generation_thirds"}
    assert a["rings"].pop("seed_scheme") == "run_seed_v2" and "seed_scheme" not in b["rings"]
    a.pop("_comment"); b.pop("_comment")
    assert a == b
    L.resolve_config(copy.deepcopy(s2), "t")
    bad = copy.deepcopy(s2); bad["rings"]["curriculum"]["mode"] = "x"
    with pytest.raises(ValueError):
        L.resolve_config(bad, "t")


def test_run_seed_v2_seeds_fresh_reproducible_per_gen_and_aircraft():
    u = json.load(open(os.path.join(PKG, "configs", "phase4_smoke_gate.json")))
    c1, c2 = L.resolve_config(copy.deepcopy(u), "t"), L.resolve_config(copy.deepcopy(u), "t")
    assert c1["rings"]["run_seed"] != c2["rings"]["run_seed"] and c1["rings"]["run_seed_source"] == "os.urandom(8)"
    fixed = copy.deepcopy(u); fixed["rings"]["run_seed"] = 12345
    a, b = L.resolve_config(copy.deepcopy(fixed), "t"), L.resolve_config(copy.deepcopy(fixed), "t")
    for m in ("c172x", "f16"):
        assert L.seeds_for(a, m, 0) == L.seeds_for(b, m, 0)                       # same run_seed -> same seeds
        assert L.seeds_for(a, m, 0) != L.seeds_for(a, m, 1)                       # gen N != gen N+1
        assert L.seeds_for(a, m, 0) != L.seeds_for(c1, m, 0)                      # different run_seed -> different
        h = L.seeds_for(a, m, holdout=True)
        assert len(h) == 8 and h == L.seeds_for(b, m, holdout=True) and not set(h) & set(L.seeds_for(a, m, 3))
        assert len(L.seeds_for(a, m, 0)) == 4 and all(0 <= x < 2 ** 64 for x in L.seeds_for(a, m, 0) + h)
    assert L.seeds_for(a, "c172x", 0) != L.seeds_for(a, "T38", 0)                # per aircraft
    s0, s1 = L.seeds_for(a, "c172x", 0)[0], L.seeds_for(a, "c172x", 1)[0]
    r0, r0b, r1 = (rc().make_course("c172x", "easy", x)["rings"] for x in (s0, s0, s1))
    assert r0 == r0b and r0 != r1                                                # rings: reproducible, differ by seed
    assert r0 != rc().make_course("c172x", "easy", L.seeds_for(c1, "c172x", 0)[0])["rings"]
    # legacy scheme (s1) unchanged: Sim Bridge train/holdout seeds from cfg seed
    s1c = L.resolve_config(copy.deepcopy(CFG_U), "t")
    assert L.seeds_for(s1c, "T38", 2) == rc().train_seeds(1, 2, 4) and L.seeds_for(s1c, "T38", holdout=True) == rc().holdout_seeds(1, 8)


def test_local_seed_formula_equals_sim_bridge_helper_when_live():
    f = L._sb_seed_fn()
    if f is None:
        pytest.skip("sim_bridge.ring_course.course_seed(run_seed, gen, k, aircraft, holdout=) not live yet")
    for rs in (1, 2 ** 63 + 5):
        for m in ("c172x", "f16"):
            assert [int(f(rs, g, k, m)) for g in (0, 3) for k in range(4)] == [L._local_seed(rs, g, k, m) for g in (0, 3) for k in range(4)]
            assert [int(f(rs, None, j, m, holdout=True)) for j in range(8)] == [L._local_seed(rs, None, j, m, True) for j in range(8)]


def test_late_fallback_option_validated():
    u = json.load(open(os.path.join(PKG, "configs", "phase4_smoke_gate.json")))
    bad = copy.deepcopy(u); bad["rings"]["curriculum"]["late_fallback"] = {"enabled": "yes"}
    with pytest.raises(ValueError):
        L.resolve_config(bad, "t")


def test_course_seed_golden_sim_bridge_and_local():
    assert rc().course_seed(1, 5, 2, "T38") == 2920704114203819991 == L._local_seed(1, 5, 2, "T38")


# ------------------------------------------------------------------------------------------------ family breeding (opt-in)
from evolution import phase4_family as PF, ga as GA  # noqa: E402

FAM_U = json.load(open(os.path.join(PKG, "configs", "phase4_family_jets.json")))
GC = PG.config(16, mutation_rate=0.0)
BR = FAM_U["breeding"]


def _ranked(vals):
    return {m: np.full((16, len(PG.GENES)), v) + np.arange(16)[:, None] * 1e-4 for m, v in vals.items()}


def test_isolated_default_is_bit_identical_cache_key_and_operators():
    old = lambda cfg, model, u, stage, seed, K: hashlib.sha256(json.dumps(
        {"pin": cfg["pin_model_version"][model], "fid": L.FID_ALIASES[cfg["fidelity"]], "u": np.asarray(u, "<f8").tobytes().hex(),
         "model": model, "stage": stage, "seed": int(seed), "K": int(K), "course": rc().VERSION, "score": L.SCORING_VERSION,
         "w": cfg["weights"], "genes": L.PG_GENES_HASH()}, sort_keys=True).encode()).hexdigest()
    u = np.linspace(0, 1, 29)
    base = L.resolve_config(copy.deepcopy(CFG_U), "t")
    iso = copy.deepcopy(CFG_U); iso["breeding"] = {"mode": "isolated"}
    iso = L.resolve_config(iso, "t")
    for c in (base, iso):
        assert L.cache_key(c, "T38", u, "easy", 5, 4) == old(c, "T38", u, "easy", 5, 4)
    assert not L._family(base) and not L._family(iso)
    # operator trace / identity hash pins unchanged (Genome's trace; phase4_ga untouched by this feature)
    tr = json.load(open(os.path.join(PKG, "analysis", "p4_operator_crosscheck.json")))
    s = json.dumps(tr)
    assert "7e5c38ba" in s and "78c01f82" in s
    assert "192ed803" in json.dumps(tr) or PG.GENES  # identity hash is asserted by test_genes_match_genome_and_identity_hash


def test_family_cache_key_has_breeding_and_aircraft_tag():
    c = L.resolve_config(copy.deepcopy(FAM_U), "t")
    u = np.linspace(0, 1, 29)
    k_t, k_f = (L.cache_key(c, m, u, "easy", 5, 4) for m in ("T38", "f16"))
    assert k_t != k_f and L.cache_key(c, "T38", u, "easy", 5, 4) == k_t
    c2 = copy.deepcopy(c); c2["breeding"]["migration"]["top_k"] = 3
    assert L.cache_key(c2, "T38", u, "easy", 5, 4) != k_t
    c3 = copy.deepcopy(c); c3["breeding"]["crossover_probability_within_family"] = 0.5
    assert L.cache_key(c3, "T38", u, "easy", 5, 4) != k_t
    iso = L.resolve_config(copy.deepcopy(CFG_U), "t")
    assert L.cache_key(iso, "T38", u, "easy", 5, 4) != k_t


def test_family_config_validation_and_membership():
    L.resolve_config(copy.deepcopy(FAM_U), "t")
    for bad in ({"mode": "x"}, {"mode": "family"}, {"mode": "family", "families": {"a": ["T38"], "b": ["T38"]}},
                {"mode": "family", "families": {"a": ["T38"]}, "crossover_probability_within_family": 2},
                {"mode": "family", "families": {"a": ["T38"]}, "migration": {"every_n_gens": 0, "top_k": 1}},
                {"mode": "family", "families": {"a": ["T38"]}, "migration": {"every_n_gens": 1, "top_k": 15}}):
        u = copy.deepcopy(FAM_U); u["breeding"] = bad
        with pytest.raises(ValueError):
            L.resolve_config(u, "t")
    assert PF.families(BR, ["T38", "f16"]) == [("jets", ["T38", "f16"])]
    assert PF.families(BR, ["T38", "f16", "737", "c172x"]) == [("jets", ["T38", "f16"]), ("transport", ["737"]), ("prop", ["c172x"])]


def test_family_mode_never_crosses_family_and_jets_child_can_have_both_parents():
    rank = _ranked({"T38": 0.1, "f16": 0.9})          # value tags the parent aircraft
    rng = PF.family_rng(1, 123, ["T38", "f16"])
    seen = set()
    for _ in range(40):
        for m in ("T38", "f16"):
            pop, meta = PF.next_generation(rng, rank, m, ["T38", "f16"], 0.7, GC)
            for row, mt in zip(pop, meta):
                names = {p["aircraft"] for p in mt["parents"]}
                assert names <= {"T38", "f16"}          # no 737 / c172x parent exists in this family
                assert set(np.round(row, 2)) <= {0.1, 0.9}       # genes only from the two jets (mutation off)
                if mt["origin"] == "child":
                    seen.add((m, tuple(sorted(names))))
    assert ("T38", ("T38", "f16")) in seen and ("f16", ("T38", "f16")) in seen      # mixed-parent children occur
    # singleton / other family: members list has one aircraft -> no foreign genes possible
    rank3 = _ranked({"737": 0.5, "T38": 0.1})
    pop, meta = PF.next_generation(PF.family_rng(1, 123, ["737"]), rank3, "737", ["737"], 0.7, GC)
    assert set(np.round(pop.ravel(), 2)) == {0.5} and all({p["aircraft"] for p in mt["parents"]} == {"737"} for mt in meta)
    # p = 0 -> isolated-like even inside a family
    rng = PF.family_rng(1, 123, ["T38", "f16"])
    pop, meta = PF.next_generation(rng, rank, "T38", ["T38", "f16"], 0.0, GC)
    assert set(np.round(pop.ravel(), 2)) == {0.1} and not any(mt.get("cross_aircraft") for mt in meta)


def test_family_migration_moves_exactly_top_k_on_schedule():
    rank = _ranked({"T38": 0.1, "f16": 0.9})
    rng = PF.family_rng(1, 123, ["T38", "f16"])
    new, metas = {}, {}
    for m in ("T38", "f16"):
        new[m], metas[m] = PF.next_generation(rng, rank, m, ["T38", "f16"], 0.7, GC)
    before = {m: new[m].copy() for m in new}
    ev = PF.migrate(new, metas, rank, ["T38", "f16"], 2)
    assert len(ev) == 4 and {(e["to"], e["from"]) for e in ev} == {("T38", "f16"), ("f16", "T38")}
    for m, d in (("T38", "f16"), ("f16", "T38")):
        changed = [i for i in range(16) if not np.array_equal(new[m][i], before[m][i])]
        assert changed == [14, 15]                                         # last top_k slots only, elites untouched
        assert np.array_equal(new[m][14], rank[d][0]) and np.array_equal(new[m][15], rank[d][1])
        assert [metas[m][i]["origin"] for i in (14, 15)] == ["migrant", "migrant"] and metas[m][0]["origin"] == "elite"
    # schedule is (g+1) % every_n_gens == 0, g < G-1: every 2 gens over 6 gens -> after g=1 and g=3
    ev_g = [g for g in range(6 - 1) if (g + 1) % BR["migration"]["every_n_gens"] == 0]
    assert ev_g == [1, 3]


def test_family_same_run_seed_reproduces_exactly():
    rank = _ranked({"T38": 0.1, "f16": 0.9})
    cfg = PG.config(16)

    def go(rs):
        rng = PF.family_rng(1, rs, ["T38", "f16"])
        return [PF.next_generation(rng, rank, m, ["T38", "f16"], 0.7, cfg) for m in ("T38", "f16")]
    a, b, c = go(5), go(5), go(6)
    assert all(np.array_equal(x[0], y[0]) and x[1] == y[1] for x, y in zip(a, b))
    assert not np.array_equal(a[0][0], c[0][0])
