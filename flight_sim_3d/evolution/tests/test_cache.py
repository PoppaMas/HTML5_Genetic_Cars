import numpy as np

from evolution import cache as cache_mod
from evolution import sim
from .conftest import load_run, make_cfg, run_batch


def test_rerun_hits_cache_and_is_identical(small_run):
    tmp = small_run["tmp"]
    first = small_run["summary"]
    assert first["this_session"]["sims_computed"] > 0
    cfg2 = make_cfg(tmp, "second")              # same config, new run id, same cache file
    summary2, run_dir2 = run_batch(cfg2)
    s = summary2["this_session"]
    assert s["sims_computed"] == 0
    assert s["cache_hit_rate"] == 1.0
    for a in summary2["aircraft"]:
        assert a["this_session"]["sims_computed"] == 0
        assert a["this_session"]["cache_hits_disk"] > 0
    assert load_run(small_run["run_dir"]) == load_run(run_dir2)


def test_cache_key_sensitivity():
    g = np.random.default_rng(0).random(6)
    p = sim.Profile().to_dict()
    sc = sim.make_scenarios(2, 1, sim.Profile())[1].to_dict()
    base = cache_mod.eval_key("c172x", g, p, sc, 1, "1.3.1", "abc")
    assert base == cache_mod.eval_key("c172x", g.copy(), dict(p), dict(sc), 1, "1.3.1", "abc")
    g2 = g.copy(); g2[0] = np.nextafter(g2[0], 1)            # one ulp
    variants = [
        cache_mod.eval_key("f16", g, p, sc, 1, "1.3.1", "abc"),
        cache_mod.eval_key("c172x", g2, p, sc, 1, "1.3.1", "abc"),
        cache_mod.eval_key("c172x", g, {**p, "speed_kts": 101.0}, sc, 1, "1.3.1", "abc"),
        cache_mod.eval_key("c172x", g, {**p, "gain_bounds": {"kp_alt": [0.001, 0.5]}}, sc, 1, "1.3.1", "abc"),
        cache_mod.eval_key("c172x", g, p, {**sc, "seed": sc["seed"] + 1}, 1, "1.3.1", "abc"),
        cache_mod.eval_key("c172x", g, p, sc, 2, "1.3.1", "abc"),
        cache_mod.eval_key("c172x", g, p, sc, 1, "1.3.2", "abc"),
        cache_mod.eval_key("c172x", g, p, sc, 1, "1.3.1", "abd"),
    ]
    assert base not in variants and len(set(variants)) == len(variants)


def test_cached_result_equals_fresh_simulation(small_run, tmp_path):
    """A cache hit must return exactly what a fresh simulation returns."""
    import json, os, sqlite3
    db = sqlite3.connect(str(small_run["tmp"] / "evals.sqlite"))
    rows = db.execute("SELECT result FROM evals WHERE aircraft='c172x' LIMIT 3").fetchall()
    assert rows
    # re-run the best genome of gen 0 directly and compare with its cached per-scenario costs
    ck = json.load(open(os.path.join(small_run["run_dir"], "checkpoints", "c172x.json")))
    b0 = ck["best_per_gen"][0]
    ac = small_run["cfg"]["aircraft"][0]
    prof = sim.Profile.from_dict(ac["resolved_profile"])
    scs = sim.make_scenarios(small_run["cfg"]["scenarios"], small_run["cfg"]["scenario_seed"], prof)
    from evolution import genome
    for s, sc in enumerate(scs):
        r = sim.simulate(genome.decode(b0["genome"], genome.make_schema(prof.gain_bounds, prof.gene_kinds)), sc, prof)
        assert r["cost"] == b0["per_scenario"][s]["cost"]
