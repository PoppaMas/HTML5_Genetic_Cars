"""The generalised sim / batch runner must reproduce the original prototype exactly for c172x."""
import os
import sys

import numpy as np
import pytest

from evolution import batch, sim

ORIG = os.environ.get("FLIGHT_SIM_DIR", os.path.join(batch.DEFAULT_SOURCE_REPO, "flight_sim"))  # repo-relative in the push layout
pytestmark = pytest.mark.skipif(not os.path.isdir(ORIG), reason="original clone not available")


def _orig_sim():
    import importlib.util
    spec = importlib.util.spec_from_file_location("orig_flight_sim_sim", os.path.join(ORIG, "sim.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_simulate_bit_identical_to_original():
    osim = _orig_sim()
    rng = np.random.default_rng(0)
    from evolution import genome
    o_sc = osim.make_scenarios(3, 5)
    n_sc = sim.make_scenarios(3, 5, sim.Profile())
    for _ in range(4):
        gains = genome.decode(rng.random(genome.N_GENES))
        for a, b in zip(o_sc, n_sc):
            ro, rn = osim.simulate(gains, a), sim.simulate(gains, b)
            for k in ("cost", "status", "t_end"):
                assert ro[k] == rn[k], (k, ro[k], rn[k])


def test_batch_reproduces_original_evolve(tmp_path):
    sys.path.insert(0, ORIG)
    try:
        import evolve as orig_evolve
    finally:
        sys.path.remove(ORIG)
    cfg = dict(orig_evolve.DEFAULTS, pop_size=10, generations=4, seed=11, scenario_seed=11, scenarios=2, workers=4,
               out=str(tmp_path / "orig"))
    orig = orig_evolve.run(cfg)
    from .conftest import run_batch
    from evolution import batch
    bcfg = batch.resolve_config({"run_id": "eq", "runs_dir": str(tmp_path / "runs"), "cache": {"enabled": False},
                                 "seed": 11, "scenarios": 2, "ga": {"pop_size": 10, "generations": 4},
                                 "aircraft": [{"name": "c172x", "profile": "baseline"}]})
    summary, _ = run_batch(bcfg)
    a = summary["aircraft"][0]
    assert a["best_fitness"] == orig["best_cost"]
    assert a["best_gains"] == orig["gains"]
    assert a["best_genome_norm"] == orig["genome"]
