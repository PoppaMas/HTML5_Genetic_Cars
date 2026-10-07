"""Regression: elitism, best-so-far monotonicity and per-generation genome persistence (ELITISM_AUDIT.md).

* ga.next_generation / ga.next_generation_blocks (every genome kind) put ranked[:elite] first, bit-identical, as copies;
* flat_rank_select's rank distribution (selection pressure) is the geometric one documented in the audit;
* a tiny real GA run (conftest small_run: 2 aircraft, rigid, pop 12 x 6 gens): every generation's ranks < elite reappear
  at index 0..elite-1 of the next generation with bit-identical genes AND cost, the best cost never rises, and
  genomes.jsonl holds the full population for every generation (rank 0 == checkpoint best_per_gen == history best)."""
import json
import os
from collections import defaultdict

import numpy as np
import pytest

from evolution import eval as ev
from evolution import ga, genome


def _ranked(rng, n, m):
    return rng.random((n, m))


@pytest.mark.parametrize("mode", ["gauss", "reset"])
@pytest.mark.parametrize("xo", ["uniform", "blx"])
@pytest.mark.parametrize("elite", [1, 2, 4])
def test_next_generation_copies_elites_unchanged(mode, xo, elite):
    cfg = ga.GAConfig(pop_size=16, elite=elite, crossover=xo, mutation_mode=mode, mutation_rate=1.0, mutation_sigma=0.5)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        ranked = _ranked(rng, 16, 20)
        before = ranked.copy()
        new = ga.next_generation(rng, ranked, cfg)
        assert new.shape == (16, 20)
        assert np.array_equal(new[:elite], before[:elite])          # bit-identical, not re-mutated / re-crossed
        assert np.array_equal(ranked, before)                         # ranked population untouched
        new[:elite] += 1.0                                            # copies, not views
        assert np.array_equal(ranked, before)


@pytest.mark.parametrize("elite", [1, 2, 3])
def test_next_generation_blocks_copies_elites_unchanged(elite):
    """phase3_b1: controller (8) | structure (12) | shape (6), full-rate block mutation."""
    spec = ga.ShapeSpec(idx=list(range(20, 26)), lo=[0.5, 0.5, 0.5, -3.0, -3.0, -5.0], hi=[1.5, 1.5, 1.5, 3.0, 3.0, 5.0],
                        log=[True, True, True, False, False, False], default=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        mutation_rate=1.0)
    blocks = [list(range(0, 8)), list(range(8, 20)), list(range(20, 26))]
    cfg = ga.GAConfig(pop_size=16, elite=elite, mutation_rate=1.0, mutation_sigma=0.5)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        ranked = _ranked(rng, 16, 26)
        before = ranked.copy()
        new = ga.next_generation_blocks(rng, ranked, cfg, blocks, spec)
        assert np.array_equal(new[:elite], before[:elite])
        assert np.array_equal(ranked, before)
        # with mutation_rate 1 every non-elite child differs from both parents somewhere (elites are the only clones)
        assert not any(np.array_equal(new[k], before[j]) for k in range(elite, 16) for j in range(16))


def test_flat_rank_select_pressure():
    """P(rank k) = p (1-p)^k + (1-p)^n / n (uniform fallback): pop 16, p 0.2 -> rank 0 ~0.202, top 4 ~0.597."""
    n, p, draws = 16, 0.2, 200_000
    rng = np.random.default_rng(0)
    counts = np.bincount([ga.flat_rank_select(rng, n, p) for _ in range(draws)], minlength=n) / draws
    q = np.array([p * (1 - p) ** k for k in range(n)]) + (1 - p) ** n / n
    assert abs(q.sum() - 1.0) < 1e-12
    assert np.max(np.abs(counts - q)) < 0.005
    assert counts[0] > counts[n // 2] > counts[-1] > 0     # failed (bottom) ranks stay selectable, rarely


def _rows(run_dir):
    R = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(run_dir, "genomes.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            R[r["aircraft"]][r["generation"]].append(r)
    return R


def test_real_run_elites_monotone_and_rows_every_gen(small_run):
    cfg, run_dir = small_run["cfg"], small_run["run_dir"]
    G, N, E = cfg["ga"]["generations"], cfg["ga"]["pop_size"], cfg["ga"]["elite"]
    assert E >= 1
    R = _rows(run_dir)
    with open(os.path.join(run_dir, "run.json")) as f:
        rj = json.load(f)
    assert set(R) == {a["name"] for a in cfg["aircraft"]}
    for ac, byg in R.items():
        # per-generation genome records: full population, every generation
        assert sorted(byg) == list(range(G))
        assert all(len(byg[g]) == N and sorted(r["rank"] for r in byg[g]) == list(range(N)) for g in byg)
        with open(os.path.join(run_dir, "checkpoints", f"{ac}.json")) as f:
            ck = json.load(f)
        schema = ev.schema_for(next(a for a in rj["aircraft"] if a["name"] == ac))
        assert ck["done"] and ck["gen_next"] == G and len(ck["best_per_gen"]) == len(ck["history"]) == G
        best = [min(byg[g], key=lambda r: r["rank"]) for g in range(G)]
        for g, b in enumerate(best):
            assert b["is_best"] and b["cost"] == min(r["cost"] for r in byg[g])
            assert ck["best_per_gen"][g]["genome"] == b["genome_norm"] and ck["best_per_gen"][g]["fitness"] == b["cost"]
            assert ck["history"][g]["best"] == b["cost"]
            assert genome.decode(b["genome_norm"], schema) == b["genome"]     # reloadable: norm genes -> decoded genes
        # best-so-far never gets worse (deterministic sims + cache: zero re-evaluation noise)
        costs = [b["cost"] for b in best]
        assert all(y <= x for x, y in zip(costs, costs[1:])), costs
        # elites of gen g reappear unchanged (genes and cost, bit for bit) at index 0..E-1 of gen g+1
        for g in range(G - 1):
            src = sorted((r for r in byg[g] if r["rank"] < E), key=lambda r: r["rank"])
            dst = sorted((r for r in byg[g + 1] if r["carried_elite"]), key=lambda r: r["index"])
            assert [r["index"] for r in dst] == list(range(E))
            for s, t in zip(src, dst):
                assert t["genome_norm"] == s["genome_norm"] and t["genome"] == s["genome"]
                assert t["cost"] == s["cost"] and t["per_scenario_cost"] == s["per_scenario_cost"]
                assert t["status"] == s["status"]
