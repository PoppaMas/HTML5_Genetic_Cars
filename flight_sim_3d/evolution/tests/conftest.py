import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # flight_sim_3d/ (repo) or /workspace/flight-sim-team (team)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evolution import batch  # noqa: E402

# Small but non-trivial: two aircraft (piston + jet with its own IC profile), 2 scenarios, short task.
SMALL = {
    "seed": 3,
    "scenarios": 2,
    "ga": {"pop_size": 12, "generations": 6},
    "profiles": {
        "short_c172x": {"duration_s": 40.0, "steps_rel_ft": [[0, 0], [5, 200]]},
        "short_f16": {"duration_s": 40.0, "steps_rel_ft": [[0, 0], [5, 200]], "h0_ft": 10000, "speed_kts": 350,
                      "min_kcas": 200},
    },
    "aircraft": [{"name": "c172x", "profile": "short_c172x"}, {"name": "f16", "profile": "short_f16"}],
}


def make_user(tmp_path, run_id, cache=True, cache_name="evals.sqlite", **over):
    user = copy.deepcopy(SMALL)
    user.update(over)
    user["run_id"] = run_id
    user["runs_dir"] = str(tmp_path / "runs")
    user["cache"] = {"enabled": cache, "path": str(tmp_path / cache_name)}
    return user


def make_cfg(tmp_path, run_id, **kw):
    return batch.resolve_config(make_user(tmp_path, run_id, **kw), "test")


def run_batch(cfg, workers=None, schedule=None):
    b = batch.Batch(cfg, log=lambda *a, **k: None)
    if workers:
        b.workers = workers
    if schedule:
        b.schedule = schedule
    return b.run(), b.run_dir


def load_run(run_dir):
    """Everything that defines the *result* of a run (no timing / cache counters)."""
    out = {}
    for f in sorted(os.listdir(os.path.join(run_dir, "checkpoints"))):
        with open(os.path.join(run_dir, "checkpoints", f)) as fh:
            ck = json.load(fh)
        out[f] = {
            "done": ck["done"], "pop": ck["pop"], "rng_state": ck["rng_state"], "best_per_gen": ck["best_per_gen"],
            "history": [{k: h[k] for k in ("generation", "best", "mean", "median", "std", "valid_rate", "crash_rate",
                                           "status_counts", "best_genome", "best_gains")} for h in ck["history"]],
        }
    tdir = os.path.join(run_dir, "trajectories")
    for f in sorted(os.listdir(tdir)):
        if f.startswith("traj_"):
            with open(os.path.join(tdir, f)) as fh:
                d = json.load(fh)
            out[f.replace(d["run_id"], "RUN")] = {k: d[k] for k in ("fitness", "genome", "data", "events", "generation")}
    return out


@pytest.fixture(scope="session")
def small_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("small")
    cfg = make_cfg(tmp, "first")
    summary, run_dir = run_batch(cfg)
    return {"tmp": tmp, "cfg": cfg, "summary": summary, "run_dir": run_dir}


# Default-layout path tests: they assert how relative "flight-dynamics/..." roots resolve in the TEAM layout (and the
# golden run identities that embed those resolved roots). With EVOLUTION_FD_DIR exported (e.g. =evolution/_fd_pin_p4cs for
# Phase 4 work) sim.abs_root deliberately follows the pin, so these 8 tests run with the variable removed (per test,
# monkeypatch; nothing else changes). Added 2026-10-07 for the Phase 4 suite-both-ways requirement.
_DEFAULT_LAYOUT_TESTS = {
    "test_phase1_hdg_config_equals_genome_export", "test_abs_root_relative_to_team_root",
    "test_relative_aircraft_root_resolves_to_the_same_profile", "test_team_rel_is_the_inverse_of_abs_root",
    "test_phase1_config_matches_genome_export", "test_phase2_genome_is_20_genes_with_fd_bounds_and_v5_ki_alt",
    "test_existing_configs_and_operators_bit_identical_to_pre_change", "test_v5_config_equals_genome_export_and_v4_plus_four_keys",
}


@pytest.fixture(autouse=True)
def _default_layout_env(request, monkeypatch):
    if request.node.originalname in _DEFAULT_LAYOUT_TESTS:
        monkeypatch.delenv("EVOLUTION_FD_DIR", raising=False)
    yield
