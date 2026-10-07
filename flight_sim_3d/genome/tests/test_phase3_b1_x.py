"""phase3_b1_x (opt-in): ga.shape_crossover 'uniform' + ga.elite 4 on top of phase3_b1. phase3_b1 / phase2_flex must
stay bit for bit (operator traces vs the pre-change golden). No flights."""
import copy
import json
import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True

import adapter  # noqa: E402
import block_ops  # noqa: E402
import p3b1x_operator_trace as TR  # noqa: E402
import run_evolve  # noqa: E402
from flightsim_path import orig_ga  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Literal pre-change constants (also in runs/p3b1x_golden_pre_change.json): seed 1, pop 64, 5 generations, synthetic cost
GOLDEN_FINAL = {"phase3_b1": ("0390a1ce6f10ec824fbeeaf782b1bca474f68f63ce7a15e077716026da513e55",
                              "0b1194cdf07eab252d4a2a9dd2a602f4c70bf77e3ed7261c0b26ad769001d1a3"),
                "phase2_flex": ("728ca2f61434c5d0c0bd8291ed987baf7040419dc9212b7567a45900953e320d",
                                "a2ab24e7d6779b3e1827a9f14d79d3205804de75b80ac8fd2cfb2392b50a9e74")}


@pytest.fixture(scope="module")
def tx():
    return adapter.load_task("phase3_b1_x")


@pytest.fixture(scope="module")
def tb():
    return adapter.load_task("phase3_b1")


def _trace(task, elite):
    GA = orig_ga()
    cfg = GA.GAConfig(pop_size=TR.POP, elite=elite)
    if getattr(task, "operators", None):
        shim = block_ops.block_ga(GA, task)
        nxt = lambda rng, r, c, tr: block_ops.next_generation(GA, rng, r, c, task.operators, tr)
    else:
        import init_pop
        shim = init_pop.seeded_ga(GA, task)
        nxt = lambda rng, r, c, tr: shim.next_generation(rng, r, c)
    return TR.drive(shim.generation_zero, nxt, shim.rank_order, task.spec.n_genes, cfg, elite)


# ------------------------------------------------------------------------------------------------ preset
def test_preset_loads_and_differs_only_in_ga(tx, tb):
    assert tx.name == "phase3_b1_x" and tx.spec.n_genes == 26
    assert [(g.name, g.block, g.min, g.max, g.scale, g.default) for g in tx.spec.genes] == \
        [(g.name, g.block, g.min, g.max, g.scale, g.default) for g in tb.spec.genes]
    assert tx.init == tb.init and tx.fitness.weights == tb.fitness.weights
    assert tx.flex_constants == tb.flex_constants and tx.er_profile_d == tb.er_profile_d
    assert tx.ga_overrides == {"elite": 4} and tb.ga_overrides == {}
    assert tx.operators["shape_crossover"] == "uniform" and tb.operators["shape_crossover"] == "block"
    strip = lambda o: {k: v for k, v in o.items() if k not in ("shape_ops", "shape_crossover")}
    assert strip(tx.operators) == strip(tb.operators)
    rx = json.load(open(os.path.join(HERE, "presets", "phase3_b1_x.json")))
    rb = json.load(open(os.path.join(HERE, "presets", "phase3_b1.json")))
    assert rx["ga"] == {"elite": 4, "shape_crossover": "uniform"} and "ga" not in rb
    assert {k: v for k, v in rx.items() if k not in ("ga", "name", "_comment")} == \
        {k: v for k, v in rb.items() if k not in ("name", "_comment")}


def test_ga_block_guards():
    assert block_ops.resolve_ga(None) == {"elite": None, "shape_crossover": "block"}
    for bad in ({"elite": -1}, {"elite": 2.0}, {"elite": True}, {"shape_crossover": "blocks"}, {"selection_p": 0.3}):
        with pytest.raises(ValueError):
            block_ops.resolve_ga(bad)
    raw = json.load(open(os.path.join(HERE, "presets", "phase3_b1_x.json")))
    bad = copy.deepcopy(raw); bad["ga"]["shape_crossover"] = "two_point"
    with pytest.raises(ValueError, match="shape_crossover"):
        adapter.build_task(bad)


# ------------------------------------------------------------------------------------------------ crossover
def test_shape_uniform_mixes_shape_keeps_controller_structure_whole(tx):
    ops = tx.operators
    ctrl, struct, shape = (ops["blocks"][b] for b in ("controller", "structure", "shape"))
    a, b = np.zeros(26), np.ones(26)
    mixed = 0
    for seed in range(200):
        c = block_ops.block_crossover(np.random.default_rng(seed), a, b, ops)
        assert len(set(c[ctrl])) == 1 and len(set(c[struct])) == 1        # whole blocks
        mixed += 0 < c[shape].sum() < len(shape)
    assert mixed > 150                                                    # ~ 1 - 2/64 of children mix the shape block


def test_shape_uniform_draw_order_is_rng_random_8(tx):
    """Locked with ER: 8 doubles = rng.random(8) (controller, structure, 6 shape genes in order); < 0.5 -> parent A.
    Same as 8 scalar rng.random() calls; no other draw."""
    ops = tx.operators
    a, b = np.arange(26.0), -np.arange(26.0) - 1.0
    for seed in range(20):
        rec = {}
        r1 = np.random.default_rng(seed)
        c = block_ops.block_crossover(r1, a, b, ops, rec)
        d = np.random.default_rng(seed).random(8)
        r2 = np.random.default_rng(seed)
        assert np.array_equal(d, np.array([r2.random() for _ in range(8)]))
        assert r1.random() == r2.random()                                 # exactly 8 doubles consumed
        pa = d < 0.5
        want = np.where(np.r_[[pa[0]] * 8, [pa[1]] * 12, pa[2:]], a, b)
        assert np.array_equal(c, want)
        assert rec == {"ctrl_a": bool(pa[0]), "struct_a": bool(pa[1]), "shape_mask_a": [bool(x) for x in pa[2:]]}


def test_default_block_mode_draws_exactly_rng_random_3(tb):
    a, b = np.zeros(26), np.ones(26)
    for seed in range(20):
        r1 = np.random.default_rng(seed)
        c = block_ops.block_crossover(r1, a, b, tb.operators)
        r2 = np.random.default_rng(seed)
        pa = r2.random(3) < 0.5
        assert r1.random() == r2.random()
        for (blk, idx), p in zip(tb.operators["blocks"].items(), pa):
            assert np.all(c[idx] == (0.0 if p else 1.0))


# ------------------------------------------------------------------------------------------------ elites
def test_elite_4_preserved_unchanged(tx):
    GA = orig_ga()
    cfg = GA.GAConfig(pop_size=32, elite=4)
    for seed in range(5):
        ranked = np.random.default_rng(50 + seed).random((32, 26))
        new = block_ops.next_generation(GA, np.random.default_rng(seed), ranked, cfg, tx.operators)
        assert new.shape == (32, 26) and np.array_equal(new[:4], ranked[:4])
        assert not any(np.array_equal(new[4], ranked[k]) for k in range(4))   # child 5 is bred, not a 5th elite


def test_run_evolve_elite_override_only_for_x(tx, tb):
    base = {"elite": 2, "pop_size": 64}
    assert run_evolve.apply_ga_overrides(tx, dict(base), [])["elite"] == 4
    assert run_evolve.apply_ga_overrides(tx, dict(base, elite=3), ["--elite", "3"])["elite"] == 3    # CLI wins
    assert run_evolve.apply_ga_overrides(tx, dict(base, elite=3), ["--elite=3"])["elite"] == 3
    for t in (tb, adapter.load_task("phase2_flex"), adapter.load_task("phase1_default")):
        for cfg, rest in ((dict(base), []), (dict(base, elite=5), ["--elite", "5"])):
            assert run_evolve.apply_ga_overrides(t, dict(cfg), rest) == cfg                         # untouched
    with pytest.raises(SystemExit):
        run_evolve.apply_ga_overrides(tx, {"elite": 2, "pop_size": 4}, [])


def test_next_generation_tweaked_entry_point(tb, tx):
    GA = orig_ga()
    cfg = GA.GAConfig(pop_size=24, elite=4)
    ranked = np.random.default_rng(9).random((24, 26))
    a = block_ops.next_generation_tweaked(GA, np.random.default_rng(3), ranked, cfg, tb.operators, "uniform")
    b = block_ops.next_generation(GA, np.random.default_rng(3), ranked, cfg, tx.operators)
    assert np.array_equal(a, b)
    assert tb.operators["shape_crossover"] == "block"                     # caller's ops not mutated


# ------------------------------------------------------------------------------------------------ controls
@pytest.mark.parametrize("name", ["phase3_b1", "phase2_flex"])
def test_control_operator_trace_unchanged(name):
    """phase3_b1 / phase2_flex: same RNG stream, same populations as before phase3_b1_x (golden captured pre-change)."""
    golden = json.load(open(TR.GOLDEN_FILE))[name]
    gens, _, _, elites_ok = _trace(adapter.load_task(name), 2)
    assert gens[0]["gen0_unranked_sha256"] == golden["gen0_unranked_sha256"] == GOLDEN_FINAL[name][0]
    assert [g["pop_sha256"] for g in gens[1:]] == golden["pop_sha256"]
    assert [g["cost_sha256"] for g in gens[1:]] == golden["cost_sha256"]
    assert gens[-1]["pop_sha256"] == GOLDEN_FINAL[name][1] and all(elites_ok)


def test_x_generation_zero_equals_phase3_b1(tx, tb):
    """The tweak changes breeding only: generation 0 is phase3_b1's, bit for bit."""
    GA = orig_ga()
    for seed in (1, 2):
        assert np.array_equal(block_ops.block_ga(GA, tx).generation_zero(np.random.default_rng(seed), 64, 26),
                              block_ops.block_ga(GA, tb).generation_zero(np.random.default_rng(seed), 64, 26))


def test_x_trace_matches_recorded_run():
    rec = json.load(open(os.path.join(HERE, "runs", "p3b1x_operator_trace.json")))["presets"]["phase3_b1_x"]
    gens, _, traces, elites_ok = _trace(adapter.load_task("phase3_b1_x"), 4)
    assert [g["pop_sha256"] for g in gens[1:]] == [g["pop_sha256"] for g in rec["generations"][1:]]
    assert traces[0] == rec["child_draws"][0]["children"] and all(elites_ok)


def test_shape_uniform_bitwise_equal_to_er_ga(tx):
    """Genome block_ops (ga.shape_crossover 'uniform', elite 4) == ER evolution/ga.py next_generation_blocks with
    GAConfig(shape_crossover='uniform') (read-only import), same seeds -> same arrays and same RNG end state."""
    import dataclasses
    ega, batch, _ = TR.er_modules()
    if "shape_crossover" not in {f.name for f in dataclasses.fields(ega.GAConfig)}:
        pytest.skip("ER ga.py has no shape_crossover option yet")
    es = TR.er_shape_spec(ega, tx.operators["shape_ops"])
    blocks = list(tx.operators["blocks"].values())
    assert blocks == batch.gene_blocks(["gains"] * 8 + ["struct"] * 12 + ["shape"] * 6)
    GA = orig_ga()
    for seed in range(8):
        ranked = np.random.default_rng(200 + seed).random((24, 26))
        r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
        mine = block_ops.next_generation(GA, r1, ranked, GA.GAConfig(pop_size=24, elite=4), tx.operators)
        theirs = ega.next_generation_blocks(r2, ranked, ega.GAConfig(pop_size=24, elite=4, shape_crossover="uniform"),
                                            blocks, es)
        assert np.array_equal(mine, theirs) and r1.bit_generator.state == r2.bit_generator.state
        a, b = ranked[0], ranked[1]
        assert np.array_equal(block_ops.crossover_shape_uniform(np.random.default_rng(seed), a, b, tx.operators),
                              ega.crossover_blocks_uniform_shape(np.random.default_rng(seed), a, b, blocks, es.idx))


def test_er_tweaked_preset_fixture_bit_identical():
    """ER's evolution/analysis/tweaked_preset_crosscheck.json (T38 gen-4, seeds 0..4, default + tweaked) regenerated
    from genome's presets/block_ops: rows, SHA-256, RNG end state and per-child traces bit for bit."""
    import p3b1x_er_fixture_check as FX
    if not os.path.exists(FX.FIXTURE):
        pytest.skip("ER fixture not present")
    out = FX.check(json.load(open(FX.FIXTURE)))
    assert out["summary"] == {"default": "BIT-IDENTICAL", "tweaked": "BIT-IDENTICAL",
                              "ranked_population_sha256_matches": True}
