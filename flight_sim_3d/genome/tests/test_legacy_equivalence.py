"""Legacy preset must reproduce the original code exactly (same seed -> same numbers)."""
import os
import subprocess
import sys

import numpy as np
import pytest

import adapter
import fitness as F
import sim_ext
from flightsim_path import FLIGHT_SIM_DIR, orig_genome, orig_sim

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def same(a, b):
    """Exact equality, treating NaN == NaN (failed runs report NaN track/effort)."""
    import json
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


HAND = dict(kp_alt=0.05, ki_alt=0.001, kd_alt=0.3, kp_pitch=0.05, ki_pitch=0.01, kd_pitch=0.02)


@pytest.mark.sim
def test_sim_ext_bit_identical_to_sim():
    S = orig_sim()
    rng = np.random.default_rng(4)
    og = orig_genome()
    gains_list = [HAND] + [og.decode(rng.random(6)) for _ in range(2)]
    for gains in gains_list:
        for sc in S.make_scenarios(2, 11):
            a, b = S.simulate(gains, sc), sim_ext.simulate(gains, sc, record=False)
            for k in ("cost", "status", "t_end", "track", "effort"):
                assert same(a[k], b[k]), (k, a[k], b[k])


@pytest.mark.sim
def test_legacy_preset_fitness_equals_sim_evaluate():
    S = orig_sim()
    task = adapter.load_task("altitude_hold_legacy")
    scs = task.make_scenarios(3, 1)
    assert [vars(s) for s in scs] == [vars(s) for s in S.make_scenarios(3, 1)]
    for gains in (HAND, orig_genome().decode([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])):  # second one leaves the envelope
        a, b = S.evaluate(gains, scs), task.evaluate(gains, scs)
        assert a["cost"] == b["cost"] and same(a["per_scenario"], b["per_scenario"])


@pytest.mark.sim
def test_roll_speed_defaults_reproduce_fixed_helpers():
    S = orig_sim()
    sc = S.make_scenarios(2, 5)[1]
    g = {**HAND, "kp_roll": 0.05, "ki_roll": 0.0, "kd_roll": 0.02, "kp_spd": 0.05, "ki_spd": 0.01, "kd_spd": 0.0}
    assert sim_ext.simulate(g, sc, record=False)["cost"] == S.simulate(HAND, sc)["cost"]


def test_shim_interface_matches_original_genome_module():
    task = adapter.load_task("altitude_hold_legacy")
    g = adapter.make_genome_module(task)
    og = orig_genome()
    assert g.N_GENES == og.N_GENES and g.GENE_NAMES == og.GENE_NAMES
    assert [x.units for x in g.SCHEMA] == [x.units for x in og.SCHEMA]
    v = np.random.default_rng(0).random(6)
    assert g.decode(v) == og.decode(v)


@pytest.mark.slow
def test_short_evolution_identical_through_evolve_py(tmp_path):
    """Run the original evolve.py and the adapter-wrapped evolve.py with the same seed; outputs must match."""
    py = sys.executable
    args = ["--pop-size", "8", "--generations", "3", "--seed", "5", "--scenarios", "2", "--workers", "4"]
    a, b = tmp_path / "orig", tmp_path / "adapter"
    subprocess.run([py, os.path.join(FLIGHT_SIM_DIR, "evolve.py"), *args, "--out", str(a)], check=True,
                   cwd=FLIGHT_SIM_DIR, capture_output=True)
    subprocess.run([py, os.path.join(HERE, "run_evolve.py"), "--task", "altitude_hold_legacy", "--", *args,
                    "--out", str(b)], check=True, cwd=HERE, capture_output=True)
    r = subprocess.run([py, os.path.join(HERE, "verify_legacy.py"), str(a), str(b)], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    assert (b / "at_bounds.json").exists() and (b / "fitness_detail.json").exists()
