from evolution import genome, sim
from .conftest import load_run, make_cfg, run_batch


def test_sim_repeatable():
    P = sim.Profile(aircraft="737", h0_ft=10000, speed_kts=250, min_kcas=150, duration_s=30)
    sc = sim.make_scenarios(2, 4, P)[1]
    g = genome.decode([0.5] * genome.N_GENES)
    r1, r2 = sim.simulate(g, sc, P), sim.simulate(g, sc, P)
    r1.pop("wall_s"); r2.pop("wall_s")
    assert r1 == r2
    # recording must not perturb the result
    assert sim.simulate(g, sc, P, record=True)["cost"] == r1["cost"]


def test_workers_and_schedule_do_not_change_results(small_run, tmp_path):
    cfg = make_cfg(tmp_path, "w1seq", cache=False)
    _, run_dir = run_batch(cfg, workers=1, schedule="sequential")
    assert load_run(run_dir) == load_run(small_run["run_dir"])
