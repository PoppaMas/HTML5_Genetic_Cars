"""Opt-in TWEAKED GA preset for phase3_b1 (Corleone 2026-10-06; Genome Architect's locked spec = Genome preset
phase3_b1_x; evolution/analysis/TWEAKED_PRESET_SPEC.md): ga.elite 4 + ga.shape_crossover 'uniform'.

- every existing config / operator path is bit-identical to the code before the change (golden captured pre-change:
  tests/data/tweaked_preset_golden.json, from analysis/tweaked_preset_golden.py): same draws, same draw order;
- options absent or at their defaults resolve exactly as before (no new key in the resolved config / run identity);
- elite 4 carries exactly 4 elites unchanged;
- uniform shape crossover: controller / structure whole blocks, shape genes per gene, one rng.random(8) per child that
  replaces the shape block's draw; equals an inline reference of the spec text;
- the cross-check fixture (analysis/tweaked_preset_crosscheck.json) reproduces;
- tweaked pilot configs differ from phase3b1_pilot.json only by the two GA options;
- a tiny real GA run with the tweaked options: monotone best, 4 carried elites, full per-generation persistence."""
import copy
import importlib.util
import json
import os

import numpy as np
import pytest

from evolution import batch, ga
from evolution.tests.test_eval import PKG, SHORT, cfg_file, need_fd
from evolution.tests.test_p3b1 import B1_PINS_FILE, need_b1

GOLDEN = os.path.join(PKG, "tests", "data", "tweaked_preset_golden.json")
FIXTURE = os.path.join(PKG, "analysis", "tweaked_preset_crosscheck.json")
GOLDEN_SOURCE_REPO = "/workspace/sandbox-run-20261006-023947/HTML5_Genetic_Cars"   # default source_repo when captured
IN_TEAM = batch.DEFAULT_SOURCE_REPO == GOLDEN_SOURCE_REPO   # team layout: compare the identity hashes too
SPEC = dict(idx=list(range(20, 26)), lo=[0.5, 0.5, 0.5, -3.0, -3.0, -5.0], hi=[1.5, 1.5, 1.5, 3.0, 3.0, 5.0],
            log=[True, True, True, False, False, False], default=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
BLOCKS = [list(range(0, 8)), list(range(8, 20)), list(range(20, 26))]
TWEAK = {"elite": 4, "shape_crossover": "uniform"}


def _load(name):
    sp = importlib.util.spec_from_file_location(f"_tw_{name}", os.path.join(PKG, "analysis", name + ".py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------- defaults bit-identical to before
@need_fd
@need_b1
def test_existing_configs_and_operators_bit_identical_to_pre_change(monkeypatch):
    """Recompute every golden case (operators, every config's gen-0 + 3 generations, smoke T38 g4) with the current
    code: populations AND final RNG state equal the pre-change capture bit for bit."""
    # packaging: the golden was captured in the team layout. The run identity hash / run_id cover machine paths (the
    # default source_repo and, in the repo copies of older configs, the repo-relative aircraft_root), so in the repo
    # those two fields are not compared; GA settings, every population and the final RNG state still are.
    monkeypatch.setitem(batch.DEFAULTS, "source_repo", GOLDEN_SOURCE_REPO)
    ref = json.load(open(GOLDEN))["cases"]
    got = _load("tweaked_preset_golden").compute()
    assert len(ref) == 205

    def strip(k, v):
        if k.startswith("config/") and k.count("/") >= 1 and isinstance(v, dict) and "identity" in v and not IN_TEAM:
            return {x: y for x, y in v.items() if x not in ("identity", "run_id")}
        return v
    bad = [k for k in ref if strip(k, got.get(k)) != strip(k, ref[k])]
    assert not bad, bad[:10]


def test_default_options_resolve_exactly_as_before():
    for fn in sorted(os.listdir(os.path.join(PKG, "configs"))):
        if not fn.endswith(".json") or "tweaked" in fn:
            continue
        u = cfg_file(fn)
        assert "shape_crossover" not in u.get("ga", {}) and "elite" not in u.get("ga", {}), fn
        c = batch.resolve_config(copy.deepcopy(u), "t")
        assert "shape_crossover" not in c["ga"] and c["ga"]["elite"] == 2, fn
        assert ga.GAConfig(**{k: v for k, v in c["ga"].items() if k != "generations"}).shape_crossover == "block"
        if u.get("genome_kind") == "phase3_b1":   # explicit defaults == absent (same identity / run id)
            u2 = copy.deepcopy(u)
            u2["ga"] = {**u2["ga"], "elite": 2, "shape_crossover": "block"}
            assert batch.identity(batch.resolve_config(u2, "t")) == batch.identity(c), fn
            assert batch.resolve_config(u2, "t")["run_id"] == c["run_id"]


def test_shape_crossover_validation():
    u = cfg_file("phase3b1_smoke.json")
    with pytest.raises(ValueError, match="shape_crossover must be one of"):
        batch.resolve_config({**u, "ga": {**u["ga"], "shape_crossover": "two_point"}}, "t")
    p2 = cfg_file("phase2_smoke_p25.json")
    with pytest.raises(ValueError, match="needs genome_kind 'phase3_b1'"):
        batch.resolve_config({**p2, "ga": {**p2["ga"], "shape_crossover": "uniform"}}, "t")
    with pytest.raises(ValueError, match="elite must be smaller"):
        batch.resolve_config({**u, "ga": {**u["ga"], "elite": 16}}, "t")
    c = batch.resolve_config({**u, "ga": {**u["ga"], **TWEAK}}, "t")
    assert c["ga"]["shape_crossover"] == "uniform" and c["ga"]["elite"] == 4
    assert batch.identity(c) != batch.identity(batch.resolve_config(u, "t"))     # tweaked = its own run identity
    with pytest.raises(ValueError, match="unknown shape_crossover"):
        ga.next_generation_blocks(np.random.default_rng(0), np.zeros((8, 26)), ga.GAConfig(pop_size=8, shape_crossover="x"),
                                  BLOCKS, ga.ShapeSpec(**SPEC))


def test_block_path_still_draws_random3():
    for seed in range(20):
        rng, ref = np.random.default_rng(seed), np.random.default_rng(seed)
        a, b = rng.random(26), rng.random(26)
        ref.random(52)
        c = ga.crossover_blocks(rng, a, b, BLOCKS)
        pa = ref.random(3) < 0.5
        exp = np.concatenate([(a if p else b)[blk] for blk, p in zip(BLOCKS, pa)])
        assert np.array_equal(c, exp) and rng.bit_generator.state == ref.bit_generator.state


# ----------------------------------------------------------------------------- elite 4
@pytest.mark.parametrize("sx", ["block", "uniform"])
def test_elite4_carries_exactly_4_unchanged(sx):
    spec = ga.ShapeSpec(**SPEC, mutation_rate=1.0)
    cfg = ga.GAConfig(pop_size=16, elite=4, mutation_rate=1.0, mutation_sigma=0.5, shape_crossover=sx)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        ranked = rng.random((16, 26))
        before = ranked.copy()
        new = ga.next_generation_blocks(rng, ranked, cfg, BLOCKS, spec)
        assert new.shape == (16, 26)
        assert np.array_equal(new[:4], before[:4]) and np.array_equal(ranked, before)
        # exactly 4: with mutation rate 1 no child (index >= 4) is a clone of any parent
        assert not any(np.array_equal(new[k], before[j]) for k in range(4, 16) for j in range(16))
        new[:4] += 1.0
        assert np.array_equal(ranked, before)                     # copies, not views


# ----------------------------------------------------------------------------- uniform shape crossover
def test_uniform_shape_crossover_per_gene_blocks_whole():
    a, b = np.zeros(26), np.ones(26)
    rng = np.random.default_rng(5)
    pats, ctrl, struct, shape = set(), [], [], []
    for _ in range(4000):
        ref = np.random.default_rng()
        ref.bit_generator.state = rng.bit_generator.state
        c = ga.crossover_blocks_uniform_shape(rng, a, b, BLOCKS, SPEC["idx"])
        d = ref.random(8)                                          # exactly 8 doubles consumed
        assert rng.bit_generator.state == ref.bit_generator.state
        assert c[0:8].min() == c[0:8].max() and c[8:20].min() == c[8:20].max()     # controller / structure whole
        assert np.array_equal(c[0:8] == 0, np.full(8, d[0] < 0.5)) and np.array_equal(c[8:20] == 0, np.full(12, d[1] < 0.5))
        assert np.array_equal(c[20:26] == 0, d[2:] < 0.5)          # draw < 0.5 -> parent a, per shape gene, gene order
        pats.add(tuple(c[20:26].astype(int)))
        ctrl.append(c[0])
        struct.append(c[8])
        shape.append(c[20:26])
    assert len(pats) == 64                                         # every per-gene mix occurs
    p = np.mean(shape, axis=0)
    assert np.all(np.abs(p - 0.5) < 0.04) and abs(np.mean(ctrl) - 0.5) < 0.04 and abs(np.mean(struct) - 0.5) < 0.04


def _ref_next_gen(seed, ranked, elite, pop, spec, cfg):
    """Inline reference of the locked spec text (only flat_rank_select / mutate_blocks reused)."""
    rng = np.random.default_rng(seed)
    out = [ranked[k].copy() for k in range(elite)]
    while len(out) < pop:
        i = ga.flat_rank_select(rng, len(ranked), 0.2)
        j = i
        while j == i:
            j = ga.flat_rank_select(rng, len(ranked), 0.2)
        d = rng.random(8)
        A, B = ranked[i], ranked[j]
        child = np.concatenate([A[:8] if d[0] < 0.5 else B[:8], A[8:20] if d[1] < 0.5 else B[8:20],
                                [A[20 + g] if d[2 + g] < 0.5 else B[20 + g] for g in range(6)]])
        out.append(ga.mutate_blocks(rng, child, cfg, spec))
    return np.array(out), rng


def test_uniform_next_generation_equals_inline_spec_reference():
    spec = ga.ShapeSpec(**SPEC)
    cfg = ga.GAConfig(pop_size=24, **TWEAK)
    for seed in range(10):
        ranked = np.random.default_rng(50 + seed).random((24, 26))
        rng = np.random.default_rng(seed)
        new = ga.next_generation_blocks(rng, ranked, cfg, BLOCKS, spec)
        exp, rref = _ref_next_gen(seed, ranked, 4, 24, spec, cfg)
        assert np.array_equal(new, exp) and rng.bit_generator.state == rref.bit_generator.state
        # and the default path differs (the option really changes the operator)
        assert not np.array_equal(new, ga.next_generation_blocks(np.random.default_rng(seed), ranked,
                                                                 ga.GAConfig(pop_size=24, elite=4), BLOCKS, spec))


@need_fd
@need_b1
def test_crosscheck_fixture_reproduces():
    fx = json.load(open(FIXTURE))
    new = _load("tweaked_preset_crosscheck").build()
    assert new["inputs"]["ranked_population_sha256"] == fx["inputs"]["ranked_population_sha256"]
    assert fx["variants"] == new["variants"]
    assert fx["variants"]["tweaked"]["elite"] == 4 and fx["variants"]["tweaked"]["shape_crossover"] == "uniform"
    for v in ("default", "tweaked"):
        for s, r in fx["results"][v].items():
            assert new["results"][v][s]["sha256"] == r["sha256"] and new["results"][v][s]["rng_after_sha256"] == r["rng_after_sha256"]
            assert new["results"][v][s]["next_generation"] == r["next_generation"]


# ----------------------------------------------------------------------------- A/B configs
def test_tweaked_pilot_configs_differ_only_by_ga_options():
    base, tw, t38 = (cfg_file(f) for f in ("phase3b1_pilot.json", "phase3b1_pilot_tweaked.json", "phase3b1_pilot_tweaked_T38.json"))
    assert {k for k in set(base) | set(tw) if base.get(k) != tw.get(k)} == {"_comment", "ga"}
    assert tw["ga"] == {**base["ga"], **TWEAK} and base["ga"] == {"pop_size": 64, "generations": 60}
    rb, rt = batch.resolve_config(copy.deepcopy(base), "x"), batch.resolve_config(copy.deepcopy(tw), "x")
    ib, it = batch.identity(rb), batch.identity(rt)
    assert {k for k in set(ib) | set(it) if ib.get(k) != it.get(k)} == {"ga"}
    assert {k: v for k, v in rt["ga"].items() if k not in TWEAK} == {k: v for k, v in rb["ga"].items() if k != "elite"}
    # T38-only file == what `--aircraft T38` makes of the 3-aircraft tweaked config (batch.main's filter)
    exp = copy.deepcopy(tw)
    exp["aircraft"] = [a for a in tw["aircraft"] if a["name"] == "T38"]
    exp["pin_model_version"] = {"T38": tw["pin_model_version"]["T38"]}
    assert {k: v for k, v in t38.items() if k != "_comment"} == {k: v for k, v in exp.items() if k != "_comment"}
    assert t38["aircraft"] == [{"name": "T38", "profile": "phase2_T38", "seed": 2}]
    r38 = batch.resolve_config(copy.deepcopy(t38), "x")
    assert r38["aircraft"] == [a for a in rt["aircraft"] if a["name"] == "T38"]   # same seed / resolved profile


@need_fd
@need_b1
def test_tweaked_pilot_model_versions_match():
    for fn in ("phase3b1_pilot_tweaked.json", "phase3b1_pilot_tweaked_T38.json"):
        for ac, d in batch.model_versions_report(batch.resolve_config(cfg_file(fn), "x")).items():
            assert all(x["match"] is True for f, x in d.items() if f != "error"), (fn, ac, d)


# ----------------------------------------------------------------------------- tiny real run with the tweaked options
@need_fd
@need_b1
def test_tiny_tweaked_phase3_b1_run(tmp_path):
    u = cfg_file("phase3b1_smoke.json")
    G, N, E = 4, 8, 4
    u.update(run_id="b1tweak", runs_dir=str(tmp_path / "runs"), cache={"enabled": True, "path": str(tmp_path / "c.sqlite")},
             scenarios=1, ga={"pop_size": N, "generations": G, **TWEAK},
             trajectories={"generations": [G - 1], "scenario": 0, "sample_hz": 30},
             aircraft=[{"name": "T38", "profile": "phase2_T38", "overrides": dict(SHORT), "seed": 2}],
             pin_model_version={"T38": {"full_a1_b1": json.load(open(B1_PINS_FILE))["T38"]["full_a1_b1"]}})
    b = batch.Batch(batch.resolve_config(u, "t"), log=lambda *a: None)
    b.workers = 2
    summ = b.run()
    assert "error" not in summ["aircraft"][0], summ["aircraft"][0].get("traceback")
    stored = json.load(open(os.path.join(b.run_dir, "config.json")))["resolved"]
    assert stored["ga"]["elite"] == E and stored["ga"]["shape_crossover"] == "uniform"
    rows = [json.loads(x) for x in open(os.path.join(b.run_dir, "genomes.jsonl"))]
    byg = {g: [r for r in rows if r["generation"] == g] for g in range(G)}
    assert len(rows) == G * N and all(sorted(r["rank"] for r in byg[g]) == list(range(N)) for g in range(G))
    assert all(len(r["genome_norm"]) == 26 and len(r["shape"]) == 6 for r in rows)
    ck = json.load(open(os.path.join(b.run_dir, "checkpoints", "T38.json")))
    assert ck["done"] and len(ck["best_per_gen"]) == len(ck["history"]) == G
    best = [min(byg[g], key=lambda r: r["rank"]) for g in range(G)]
    costs = [r["cost"] for r in best]
    assert all(y <= x for x, y in zip(costs, costs[1:])), costs                 # monotone best
    for g in range(G):
        assert ck["best_per_gen"][g]["genome"] == best[g]["genome_norm"] and ck["history"][g]["best"] == best[g]["cost"]
        assert sum(r["is_elite"] for r in byg[g]) == E
    for g in range(G - 1):                                                       # exactly 4 carried, bit for bit
        src = sorted((r for r in byg[g] if r["rank"] < E), key=lambda r: r["rank"])
        dst = sorted((r for r in byg[g + 1] if r["carried_elite"]), key=lambda r: r["index"])
        assert [r["index"] for r in dst] == list(range(E))
        for s, t in zip(src, dst):
            assert t["genome_norm"] == s["genome_norm"] and t["cost"] == s["cost"] and t["status"] == s["status"]
    # generation 1 == the tweaked operator applied to generation 0's ranking (gen 0 rebuilt from the seed exactly as
    # batch does, so the RNG state after gen 0 is known): the run really used elite 4 + uniform shape crossover
    gold = _load("tweaked_preset_golden")
    cfg = b.cfg
    pop0, rng, blocks, sspec = gold.gen0(cfg, cfg["aircraft"][0])
    assert [r["genome_norm"] for r in sorted(byg[0], key=lambda r: r["index"])] == pop0.tolist()
    ranked0 = np.array([r["genome_norm"] for r in sorted(byg[0], key=lambda r: r["rank"])])
    nxt = ga.next_generation_blocks(rng, ranked0, gold.gcfg_of(cfg), blocks, sspec)
    assert gold.gcfg_of(cfg).shape_crossover == "uniform" and gold.gcfg_of(cfg).elite == E
    assert [r["genome_norm"] for r in sorted(byg[1], key=lambda r: r["index"])] == nxt.tolist()
