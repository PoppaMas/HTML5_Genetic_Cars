"""Phase 2 wiring: Genome phase2_flex genome (20 genes), baseline-seeded generation 0, tail/fuselage mass-credit clip,
per-aircraft ladders, FD model_version pins, configs, the check script. Mocked or tiny evaluations only."""
import copy
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from evolution import batch, cache as cache_mod, fidelity as F, ga, genome, runinfo, sim
from evolution.tests.test_eval import FD_DIR, PKG, SHORT, TEAM, cfg_file, need_fd

FD_PINS_FILE = os.path.join(FD_DIR, "v2_results", "model_versions_post_mass.json")
# FD's CURRENT full/reduced strings (P2.5 since 2026-10-06 ~14:00 PT; unchanged by P3-A1). The post-mass configs keep
# their historical post_mass pins (so check_pins now correctly refuses them); live-match checks use the P2.5 configs.
FD_PINS_P25_FILE = os.path.join(FD_DIR, "v2_results", "model_versions_post_p25.json")
GENOME_PRESET = os.path.join(TEAM, "genome", "presets", "phase2_flex.json")
STRUCT12 = ["wing_ei_root", "wing_ei_taper_1", "wing_ei_taper_2", "wing_ei_taper_3", "wing_ei_taper_4",
            "wing_gj_ratio_root", "wing_gj_ratio_tip", "wing_nsm_root", "wing_nsm_tip", "tail_stiffness_scale",
            "fuselage_stiffness_scale", "struct_damping_ratio"]


def resolved(name="phase2_pilot.json", **over):
    u = cfg_file(name)
    u.update(over)
    return batch.resolve_config(u, "t")


# ----------------------------------------------------------------------------- genome / bounds
@need_fd
def test_phase2_genome_is_20_genes_with_fd_bounds_and_v5_ki_alt():
    cfg = resolved()
    fb = F.fd_modules()["fb"]
    fd = {g.name: g for g in fb.gene_schema(False)}
    for ac in cfg["aircraft"]:
        P = sim.Profile.from_dict(ac["resolved_profile"])
        sch, groups = batch.full_schema(P, cfg["struct_genes"], cfg["struct_asymmetric"])
        assert len(sch) == 20 and groups.count("gains") == 8 and groups.count("struct") == 12
        assert [g.name for g in sch[8:]] == STRUCT12
        for g in sch[8:]:
            assert (g.min, g.max, g.kind, g.default) == (fd[g.name].lo, fd[g.name].hi, fd[g.name].scale, fd[g.name].default)
        assert P.gain_bounds["ki_alt"][1] == 0.5
        assert P.flex_mass_credit_clip is None   # Genome's phase2_flex dropped the interim clip at 08:15 PT (A/B only)
        assert ac["resolved_profile"]["aircraft_root"] == os.path.join(TEAM, "flight-dynamics", "jsbsim_root")
        # decode/encode round trip of the baseline
        u = F.baseline_u(sch[8:])
        vals = genome.decode(np.concatenate([np.full(8, 0.5), u]), sch)
        for g in sch[8:]:
            assert vals[g.name] == pytest.approx(fd[g.name].default, rel=1e-12)


@need_fd
def test_genome_preset_struct_block_matches():
    if not os.path.exists(GENOME_PRESET):
        pytest.skip("genome preset absent")
    pre = json.load(open(GENOME_PRESET))
    assert pre["flex"]["asymmetric"] is False and pre["blocks"]["structure_v2"]
    assert not pre["fitness"]["params"].get("struct_v2_mass_credit_clip")   # off since 08:15 PT; our configs follow
    assert pre["init"]["mode"] == "baseline"


@need_fd
def test_asymmetric_schema_is_14_and_off_by_default():
    assert len(F.struct_schema()) == 12 and len(F.struct_schema(True)) == 14
    assert batch.DEFAULTS["struct_asymmetric"] is False
    cfg = resolved(struct_asymmetric=True)
    P = sim.Profile.from_dict(cfg["aircraft"][0]["resolved_profile"])
    assert len(batch.full_schema(P, True, True)[0]) == 22
    with pytest.raises(ValueError):
        resolved(struct_genes=False, struct_asymmetric=True, init={"mode": "uniform", "sigma": 0.1, "blocks": ["struct"]})


# ----------------------------------------------------------------------------- generation 0
@need_fd
def test_baseline_seeding_matches_genome_algorithm_and_uniform_is_untouched():
    cfg = resolved()
    P = sim.Profile.from_dict(cfg["aircraft"][0]["resolved_profile"])
    sch, groups = batch.full_schema(P, True)
    n = 4000
    for sigma in (0.10, 0.15):
        rng = np.random.default_rng(7)
        pop = batch.seed_generation_zero(ga.generation_zero(rng, n, 20), rng, sch, groups,
                                         {"mode": "baseline", "sigma": sigma, "blocks": ["struct"]})
        # Genome init_pop.generation_zero: uniform draw first, then standard_normal noise on the struct columns
        rng2 = np.random.default_rng(7)
        ref = rng2.random((n, 20))
        ref[:, 8:] = np.clip(F.baseline_u(sch[8:]) + sigma * rng2.standard_normal((n, 12)), 0, 1)
        assert np.array_equal(pop, ref)
        u = F.baseline_u(sch[8:])
        inner = (u > 3 * sigma) & (u < 1 - 3 * sigma)
        assert np.allclose(pop[:, 8:][:, inner].mean(0), u[inner], atol=0.01)
        assert np.allclose(pop[:, 8:][:, inner].std(0), sigma, rtol=0.05)
        assert pop[:, 8:].min() >= 0 and pop[:, 8:].max() <= 1
    rng = np.random.default_rng(7)
    p0 = ga.generation_zero(rng, 8, 20)
    st = rng.bit_generator.state
    assert batch.seed_generation_zero(p0, rng, sch, groups, batch.DEFAULTS["init"]) is p0
    assert rng.bit_generator.state == st


def test_init_validation():
    with pytest.raises(ValueError, match="sigma"):
        resolved(init={"mode": "baseline", "sigma": 0.05, "blocks": ["struct"]})
    with pytest.raises(ValueError, match="sigma"):
        resolved(init={"mode": "baseline", "sigma": 0.2, "blocks": ["struct"]})
    with pytest.raises(ValueError, match="struct_genes"):
        resolved(struct_genes=False, multi_fidelity={"enabled": False})
    with pytest.raises(ValueError, match="init.mode"):
        resolved(init={"mode": "lhs", "sigma": 0.1, "blocks": ["struct"]})
    assert resolved(init={"mode": "baseline", "sigma": 0.15, "blocks": ["struct"]})["init"]["sigma"] == 0.15


def test_new_keys_keep_old_run_ids():
    u = cfg_file("phase1_hdg.json")
    cfg = batch.resolve_config(copy.deepcopy(u), "phase1hdg")
    ident = batch.identity(cfg)
    for k in batch.OPTIONAL_DEFAULTS:
        assert k not in ident
    old = {k: v for k, v in cfg.items() if k not in batch.OPTIONAL_DEFAULTS}
    assert batch.identity(old) == ident
    assert "flex_mass_credit_clip" not in cfg["aircraft"][0]["resolved_profile"]
    assert "multi_fidelity" not in cfg["aircraft"][0]


# ----------------------------------------------------------------------------- mass clip
def _genome_mass_term(mass, w, clip):   # verbatim formula of genome/fd_bridge.mass_term_v2
    bodies = {"wing": ("wingR_lb", "wingL_lb"), "ht": ("ht_lb",), "vt": ("vt_lb",), "fus": ("fus_lb",)}
    dm = {b: float(sum(mass[k] for k in ks)) for b, ks in bodies.items()}
    used = {b: (max(0.0, v) if b in clip else v) for b, v in dm.items()}
    return w * sum(used.values()) / float(mass["baseline_flexible_lb"])


def _mass(**kw):
    m = {"wingR_lb": -3.0, "wingL_lb": -3.0, "ht_lb": -1.5, "vt_lb": 0.7, "fus_lb": -4.0, "baseline_flexible_lb": 200.0}
    m.update(kw)
    m["total_lb"] = sum(m[k] for k in ("wingR_lb", "wingL_lb", "ht_lb", "vt_lb", "fus_lb"))
    m["total_frac"] = m["total_lb"] / m["baseline_flexible_lb"]
    return m


def _out(mass, w=0.3):
    jm = w * mass["total_frac"]
    per = [{"cost": 1.0 + jm}, {"cost": 2.0 + jm}, {"cost": 4.0 + jm}]
    return {"cost": float(np.mean([p["cost"] for p in per])), "per_scenario": per, "terms": {"J_mass": jm}}


def test_mass_clip_equals_genome_formula():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("gfd", os.path.join(TEAM, "genome", "fd_bridge.py"))
        gfd = None   # importing Genome's module pulls FD; the verbatim copy above is the reference
    except Exception:  # noqa: BLE001
        gfd = None
    for m in (_mass(), _mass(ht_lb=2.0, fus_lb=1.0), _mass(vt_lb=-0.1)):
        for clip in ((), ("ht", "vt", "fus"), ("fus",)):
            assert F.mass_term_clipped(m, 0.3, clip)["J_mass"] == pytest.approx(_genome_mass_term(m, 0.3, clip), abs=1e-15)
    m = _mass()
    o = F.apply_mass_credit_clip(_out(m), m, ("ht", "vt", "fus"), 0.3)
    want = _genome_mass_term(m, 0.3, ("ht", "vt", "fus"))
    assert o["terms"]["J_mass"] == pytest.approx(want)
    assert o["J_mass_fd"] == pytest.approx(0.3 * m["total_frac"])
    assert o["mass_credit_delta"] == pytest.approx(0.3 * 5.5 / 200)
    assert [p["cost"] for p in o["per_scenario"]] == pytest.approx([1 + want, 2 + want, 4 + want])
    assert o["cost"] == pytest.approx(7 / 3 + want)
    with pytest.raises(ValueError):
        F.mass_term_clipped(m, 0.3, ("wing",))


def test_mass_clip_bit_identical_when_nothing_clipped():
    m = _mass(ht_lb=0.0, vt_lb=0.5, fus_lb=2.0)
    o0 = _out(m)
    o = F.apply_mass_credit_clip(copy.deepcopy(o0), m, ("ht", "vt", "fus"), 0.3)
    assert o["cost"] == o0["cost"] and o["terms"] == o0["terms"]
    assert [p["cost"] for p in o["per_scenario"]] == [p["cost"] for p in o0["per_scenario"]]
    assert o["mass_credit_delta"] == 0.0


def test_profile_clip_validation_and_omission():
    d = sim.Profile().to_dict()
    assert "flex_mass_credit_clip" not in d
    assert sim.Profile.from_dict(d).to_dict() == d
    p = sim.Profile.from_dict({**d, "flex_mass_credit_clip": ["ht", "vt", "fus"]})
    assert p.flex_mass_credit_clip == ("ht", "vt", "fus") and p.to_dict()["flex_mass_credit_clip"] == ["ht", "vt", "fus"]
    for bad in (["wing"], ["ht", "ht"]):
        with pytest.raises(ValueError):
            sim.Profile.from_dict({**d, "flex_mass_credit_clip": bad})
    assert sim.Profile.from_dict({**d, "flex_mass_credit_clip": ["fus", "ht"]}).flex_mass_credit_clip == ("ht", "fus")


# ----------------------------------------------------------------------------- ladders, gates
def test_pilot_ladders_gates_and_budget():
    cfg = resolved()
    assert cfg["ga"]["pop_size"] == 64 and cfg["ga"]["generations"] == 60 and cfg["seed"] == 1 and cfg["viz"] == "off"
    assert cfg["fidelity"] == "full" and cfg["init"] == {"mode": "baseline", "sigma": 0.10, "blocks": ["struct"]}
    acs = {a["name"]: a for a in cfg["aircraft"]}
    assert set(acs) == {"c172x", "T38", "737"}
    # Pilot uses global rigid→full for all three (ladder-bench preferred; s1 c172x had reduced, s2+ do not).
    assert "multi_fidelity" not in acs["c172x"]
    mf = cfg["multi_fidelity"]
    assert mf["enabled"] and mf["ladder"] == ["rigid", "full"]
    k_full = max(mf["top_k"], int(np.ceil(mf["min_full_frac"] * 64)))
    assert k_full >= 16
    for n in ("c172x", "T38", "737"):
        assert acs[n].get("multi_fidelity") is None
        gate = F.per_aircraft(n, cfg["fidelity_per_aircraft"])["reduced_gate"]
        assert gate == (0.9 if n == "c172x" else 1.0)
    smoke = resolved("phase2_smoke.json")
    assert (smoke["ga"]["pop_size"], smoke["ga"]["generations"], smoke["fidelity"]) == (16, 5, "full")
    assert not smoke["multi_fidelity"]["enabled"] and not any("multi_fidelity" in a for a in smoke["aircraft"])
    rf, rrf = resolved("phase2_bench_rf.json"), resolved("phase2_bench_rrf.json")
    assert rf["multi_fidelity"]["ladder"] == ["rigid", "full"] and rrf["multi_fidelity"]["ladder"] == ["rigid", "reduced", "full"]
    strip = lambda c: {k: v for k, v in batch.identity(c).items() if k not in ("multi_fidelity", "pin_model_version")}
    assert strip(rf) == strip(rrf)


def test_per_aircraft_mf_validation():
    with pytest.raises(ValueError, match="not in the batch"):
        resolved(multi_fidelity_per_aircraft={"f16": {"screen": "rigid"}})
    with pytest.raises(ValueError, match="unknown keys"):
        resolved(multi_fidelity_per_aircraft={"c172x": {"ladder": ["rigid"]}})
    with pytest.raises(ValueError, match="screen"):
        resolved(multi_fidelity_per_aircraft={"c172x": {"screen": ["reduced", "rigid"]}})
    cfg = resolved(multi_fidelity_per_aircraft={"T38": {"enabled": False}})
    t = next(a for a in cfg["aircraft"] if a["name"] == "T38")
    assert t["multi_fidelity"]["enabled"] is False and "ladder" not in t["multi_fidelity"]


# ----------------------------------------------------------------------------- pins
@need_fd
def test_config_pins_equal_fd_published_versions():
    fd = json.load(open(FD_PINS_FILE))
    for fn in ("phase2_pilot.json", "phase2_smoke.json", "phase2_bench_rf.json", "phase2_bench_rrf.json"):
        cfg = resolved(fn)   # historical post-mass configs: pins = FD's post_mass strings (no longer FD's current model)
        for ac, pins in cfg["pin_model_version"].items():
            for f, v in pins.items():
                assert v == fd[ac][f], (fn, ac, f)
    fd25 = json.load(open(FD_PINS_P25_FILE))
    for fn in ("phase2_smoke_p25.json", "phase2_pilot_p25.json"):
        cfg = resolved(fn)
        for ac, pins in cfg["pin_model_version"].items():
            for f, v in pins.items():
                assert v == fd25[ac][f], (fn, ac, f)
        rep = batch.model_versions_report(cfg)
        for ac, d in rep.items():
            for f, x in d.items():
                if f != "rigid":
                    assert x["match"] is True, (fn, ac, f, x)


def test_check_pins_logic():
    cfg = {"pin_model_version": {"c172x": {"full": "full:flexv2:aaaa"}}}
    assert batch.check_pins(cfg, "T38", ["full"], {"full": "x"}) == {}
    assert batch.check_pins(cfg, "c172x", ["rigid", "full"], {"rigid": "r", "full": "full:flexv2:aaaa"}) == \
        {"full": "full:flexv2:aaaa"}
    with pytest.raises(SystemExit, match="pin mismatch"):
        batch.check_pins(cfg, "c172x", ["full"], {"full": "full:flexv2:bbbb"})
    with pytest.raises(SystemExit, match="no entry"):
        batch.check_pins(cfg, "c172x", ["reduced", "full"], {"reduced": "q", "full": "full:flexv2:aaaa"})
    with pytest.raises(SystemExit, match="placeholder"):
        batch.check_pins({"pin_model_version": {"c172x": {"full": batch.PIN_PLACEHOLDER}}}, "c172x", ["full"],
                         {"full": "full:flexv2:aaaa"})


def test_cache_guard_refuses_unpinned_results(tmp_path):
    c = cache_mod.EvalCache(str(tmp_path / "c.sqlite"))
    c.pins = {"c172x": {"full": "full:flexv2:aaaa"}}
    c.put_many([("k0", "c172x", {"fidelity": "rigid", "model_version": "rigid:x"})])
    c.put_many([("k1", "c172x", {"fidelity": "full", "model_version": "full:flexv2:aaaa"})])
    c.put_many([("k2", "T38", {"fidelity": "full", "model_version": "anything"})])
    with pytest.raises(RuntimeError, match="refusing"):
        c.put_many([("k3", "c172x", {"fidelity": "full", "model_version": "full:flexv2:bbbb"})])
    c.pins = {"c172x": {}}
    with pytest.raises(RuntimeError):
        c.put_many([("k4", "c172x", {"fidelity": "reduced", "model_version": "reduced:flexv1:aaaa"})])
    assert c.count() == 3


def _tiny_p2(tmp_path, run_id, pins, **kw):
    u = cfg_file("phase2_smoke.json")
    u.update(run_id=run_id, runs_dir=str(tmp_path / "runs"), cache={"enabled": True, "path": str(tmp_path / "c.sqlite")},
             scenarios=1, ga={"pop_size": 4, "generations": 1}, trajectories={"generations": [], "scenario": 0, "sample_hz": 30},
             aircraft=[{"name": "c172x", "profile": "phase2_c172x", "overrides": dict(SHORT)}],
             pin_model_version={"c172x": pins})
    u.update(kw)
    return batch.resolve_config(u, "t")


def _cache_rows(tmp_path):
    p = tmp_path / "c.sqlite"
    if not p.exists():
        return 0
    return sqlite3.connect(str(p)).execute("SELECT COUNT(*) FROM evals").fetchone()[0]


@need_fd
def test_run_refuses_placeholder_before_touching_anything(tmp_path):
    cfg = _tiny_p2(tmp_path, "ph", {"full": batch.PIN_PLACEHOLDER})
    with pytest.raises(SystemExit, match="PENDING"):
        batch.Batch(cfg, log=lambda *a: None).run()
    assert not os.path.exists(tmp_path / "runs" / "ph")
    assert _cache_rows(tmp_path) == 0


@need_fd
def test_run_refuses_mismatched_pin_with_no_evaluation_or_cache(tmp_path):
    cfg = _tiny_p2(tmp_path, "mm", {"full": "full:flexv2:00000000"})
    b = batch.Batch(cfg, log=lambda *a: None)
    b.workers = 1
    with pytest.raises(SystemExit, match="pin mismatch"):
        b.run()
    assert _cache_rows(tmp_path) == 0
    rd = tmp_path / "runs" / "mm"
    g = rd / "genomes.jsonl"
    assert not g.exists() or g.stat().st_size == 0
    assert not (rd / "run.json").exists() and not os.listdir(rd / "checkpoints")


# ----------------------------------------------------------------------------- paths
def test_abs_root_follows_evolution_fd_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("EVOLUTION_FD_DIR", raising=False)
    assert sim.abs_root("flight-dynamics/jsbsim_root") == os.path.join(sim.TEAM_ROOT, "flight-dynamics", "jsbsim_root")
    assert sim.fd_dir() == os.path.join(sim.TEAM_ROOT, "flight-dynamics")
    monkeypatch.setenv("EVOLUTION_FD_DIR", str(tmp_path / "fdx"))
    assert sim.abs_root("flight-dynamics/jsbsim_root") == str(tmp_path / "fdx" / "jsbsim_root")
    assert sim.abs_root("other/thing") == os.path.join(sim.TEAM_ROOT, "other", "thing")
    assert sim.abs_root("/abs/x") == "/abs/x"
    P = sim.Profile.from_dict({**sim.Profile().to_dict(), "aircraft_root": "flight-dynamics/jsbsim_root"})
    assert F.roots(P) == (str(tmp_path / "fdx" / "jsbsim_root"), str(tmp_path / "fdx" / "jsbsim_root_v2"))


def test_no_absolute_paths_in_phase2_configs():
    for fn in ("phase2_pilot.json", "phase2_smoke.json", "phase2_bench_rf.json", "phase2_bench_rrf.json"):
        txt = open(os.path.join(PKG, "configs", fn)).read()
        assert "/workspace" not in txt, fn


# ----------------------------------------------------------------------------- check script
def _write_run(d, rows, genes):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "genomes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"run_id": "x", "aircraft": [{"name": "c172x", "genes": genes}]}, open(os.path.join(d, "run.json"), "w"))


def test_check_script_flags(tmp_path):
    sys.path.insert(0, os.path.join(PKG, "analysis"))
    import phase2_check
    genes = [{"name": "kp_alt", "min": 0.1, "max": 1, "group": "gains"},
             {"name": "fuselage_stiffness_scale", "min": 0.6, "max": 2.0, "group": "struct"},
             {"name": "wing_nsm_tip", "min": 0.8, "max": 1.25, "group": "struct"}]
    rows = []
    for i in range(10):
        rows.append({"aircraft": "c172x", "generation": 1, "rank": i, "individual_id": f"c172x:g1:r{i}", "cost": 1.0 + i,
                     "status": "ok" if i < 9 else "crash", "genome_norm": [0.5, 0.01 if i < 4 else 0.5, 0.0],
                     "terms": {"track": 0.5, "J_mass": -0.03 if i == 0 else 0.01, "J_tail_bm_limit": 0.0},
                     "per_scenario_cost": [1.0 + i]})
    rows.append({**rows[0], "generation": 0, "status": "crash"})
    _write_run(str(tmp_path / "r"), rows, genes)
    res = phase2_check.check_run(str(tmp_path / "r"))["aircraft"]["c172x"]
    assert res["structural_sum"] == pytest.approx(-0.03) and res["negative_terms"] == {"J_mass": -0.03}
    assert res["floor_fraction"] == {"fuselage_stiffness_scale": 0.4, "wing_nsm_tip": 1.0}
    assert res["invalid_rate_last"] == pytest.approx(0.1) and res["invalid_rate_all"] == pytest.approx(2 / 11)
    fl = "\n".join(res["flags"])
    assert "negative structural cost" in fl and "fuselage_stiffness_scale" in fl and "wing_nsm_tip" not in fl
    assert phase2_check.main([str(tmp_path / "r")]) == 1


@need_fd
def test_tiny_pinned_full_run_end_to_end(tmp_path):
    """pop 4 x 1 generation, 1 scenario of 12 s, c172x at full fidelity with the real (current, P2.5) FD pin (tmp cache)."""
    fd = json.load(open(FD_PINS_P25_FILE))
    cfg = _tiny_p2(tmp_path, "ok", {"full": fd["c172x"]["full"]})
    b = batch.Batch(cfg, log=lambda *a: None)
    b.workers = 2
    b.run()
    rd = tmp_path / "runs" / "ok"
    rows = [json.loads(x) for x in open(rd / "genomes.jsonl")]
    rj = json.load(open(rd / "run.json"))
    assert len(rows) == 4 and len(rows[0]["genome"]) == 20 and len(rj["aircraft"][0]["genes"]) == 20
    assert rj["init"]["mode"] == "baseline" and rj["pin_model_version"]["c172x"]["full"] == fd["c172x"]["full"]
    for r in rows:
        assert r["fidelity"] == "full" and r["model_version"] == fd["c172x"]["full"]
        assert set(F.TERM_KEYS) <= set(r["terms"]) and "mass_total_frac" in r and "mass_lb" in r
    con = sqlite3.connect(str(tmp_path / "c.sqlite"))
    for (res,) in con.execute("SELECT result FROM evals"):
        assert json.loads(res)["model_version"] == fd["c172x"]["full"]
