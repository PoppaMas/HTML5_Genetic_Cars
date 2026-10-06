"""Quick checks: python -m pytest flight_sim  (or: python flight_sim/test_flight_ga.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ga  # noqa: E402
import genome  # noqa: E402


def test_decode_ranges():
    lo = genome.decode([0.0] * genome.N_GENES)
    hi = genome.decode([1.0] * genome.N_GENES)
    mid = genome.decode([0.5] * genome.N_GENES)
    for g in genome.SCHEMA:
        assert np.isclose(lo[g.name], g.min) and np.isclose(hi[g.name], g.max)
        assert np.isclose(mid[g.name], np.sqrt(g.min * g.max))  # log scale midpoint = geometric mean
        assert np.isclose(g.encode(mid[g.name]), 0.5)


def test_rank_select_distribution():
    rng = np.random.default_rng(0)
    counts = np.bincount([ga.flat_rank_select(rng, 20, 0.2) for _ in range(50000)], minlength=20) / 50000
    assert abs(counts[0] - 0.2028) < 0.01  # 0.2 + 0.8**20 / 20
    assert np.all(np.diff(counts[:10]) < 0)  # better ranks picked more often


def test_next_generation_deterministic_and_elitist():
    cfg = ga.GAConfig(pop_size=10, elite=2)
    ranked = np.random.default_rng(1).random((10, genome.N_GENES))
    a = ga.next_generation(np.random.default_rng(7), ranked, cfg)
    b = ga.next_generation(np.random.default_rng(7), ranked, cfg)
    assert np.array_equal(a, b)
    assert np.array_equal(a[:2], ranked[:2])
    assert a.shape == (10, genome.N_GENES) and a.min() >= 0 and a.max() <= 1


def test_simulation_deterministic():
    import sim
    gains = dict(kp_alt=0.05, ki_alt=0.001, kd_alt=0.3, kp_pitch=0.05, ki_pitch=0.01, kd_pitch=0.02)
    sc = sim.make_scenarios(2, 3)
    r1, r2 = sim.evaluate(gains, sc), sim.evaluate(gains, sc)
    assert r1 == r2
    assert all(s["status"] == "ok" for s in r1["per_scenario"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
