#!/usr/bin/env python3
"""Run the flight_sim GA unchanged and export the best individual of EVERY generation
as ga-flightsim-traj/1 trajectory files plus an index.json.

How it hooks in (no project files are modified):
  * evolve.run() is called as-is (same RNG stream, same Pool, same CSV/plots).
  * ga.next_generation is wrapped so that, before breeding, we copy the ranked
    population's best genome (pop[0]) for that generation; the last generation's
    best comes from evolve's summary.
  * Each captured genome is then re-flown through trajlog.fly() (which calls
    sim.simulate unchanged) on all GA scenarios; the recomputed fitness is checked
    against the GA's fitness_history.csv.

Output layout:
  <out-root>/<run_id>/ga_output/          evolve.py's own outputs (CSV, best_gains.json, plots)
  <out-root>/<run_id>/trajectories/       traj_<aircraft>_<run_id>_g<NNN>.json + index.json

Example (the project's committed example config: seed 1, pop 48, 40 gens, 3 scenarios):
  python export_run.py --config <flight_sim>/config.example.json --run-id seed1-pop48
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trajlog  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

_MODS = None
_JOB = None


def _init(flight_sim_dir, job):
    global _MODS, _JOB
    _MODS = trajlog.import_flight_sim(flight_sim_dir)
    _JOB = job


def _log_one(item):
    gen, vec = item
    mods, job = _MODS, _JOB
    sim = mods["sim"]
    scenarios = [sim.Scenario(**{**s, "steps": [tuple(x) for x in s["steps"]]}) for s in job["scenarios"]]
    gains = mods["genome"].decode(vec)
    fitness, results, rec = trajlog.fly(mods, gains, scenarios, job["record_scenario"], job["sample_hz"],
                                        job["aircraft"])
    traj = trajlog.build_trajectory(
        mods, rec, aircraft=job["aircraft"], run_id=job["run_id"], generation=gen, fitness=fitness,
        genome_vec=vec, scenario=scenarios[job["record_scenario"]], scenario_index=job["record_scenario"],
        scenario_result=results[job["record_scenario"]], per_scenario=results, seed=job["seed"],
        extra={"ga_fitness_reported": job["csv_best"].get(gen), "ga_config": job["config"]})
    fn = trajlog.traj_filename(job["aircraft"], job["run_id"], gen, job["gzip"])
    trajlog.write_json(traj, os.path.join(job["traj_dir"], fn), job["gzip"])
    return {"generation": gen, "fitness": fitness, "aircraft": job["aircraft"], "file": fn,
            "status": results[job["record_scenario"]]["status"], "samples": len(traj["data"]),
            "duration_s": traj["data"][-1][0], "genome": traj["genome"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flight-sim-dir", default=None, help=f"default {trajlog.DEFAULT_FLIGHT_SIM_DIR}")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-root", default=os.path.join(HERE, "data", "runs"))
    p.add_argument("--aircraft", default=None, help="JSBSim model (default: the project's sim.AIRCRAFT)")
    p.add_argument("--record-scenario", type=int, default=0, help="scenario to log (0 = calm air)")
    p.add_argument("--sample-hz", type=float, default=30.0)
    p.add_argument("--gzip", action="store_true", help="write .json.gz trajectories")
    p.add_argument("--log-workers", type=int, default=0, help="processes for re-flying/logging (0 = all cores)")
    p.add_argument("--relog", action="store_true",
                   help="don't re-run the GA: re-log from an existing export (genomes from its trajectory files, "
                        "scenarios/config from ga_output/best_gains.json); use after changing the logger/schema")
    a, ga_argv = p.parse_known_args(argv)  # everything else goes to evolve.py's own parser

    mods = trajlog.import_flight_sim(a.flight_sim_dir)
    evolve, ga, sim, genome = mods["evolve"], mods["ga"], mods["sim"], mods["genome"]
    flight_sim_dir = os.path.dirname(os.path.abspath(sim.__file__))
    run_dir = os.path.join(a.out_root, a.run_id)
    ga_out = os.path.join(run_dir, "ga_output")
    traj_dir = os.path.join(run_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    t0 = time.time()
    if a.relog:
        with open(os.path.join(ga_out, "best_gains.json")) as f:
            summary = json.load(f)
        cfg = summary["config"]
        aircraft = a.aircraft or summary.get("aircraft") or sim.AIRCRAFT
        with open(os.path.join(traj_dir, "index.json")) as f:
            old = json.load(f)
        old_entries = old.get("entries") or old.get("trajectories")
        captured = []
        for e in sorted(old_entries, key=lambda e: e["generation"]):
            with open(os.path.join(traj_dir, e["file"])) as f:
                captured.append(np.array(json.load(f)["genome_normalized"], dtype=float))
        for e in old_entries:
            os.remove(os.path.join(traj_dir, e["file"]))
        assert len(captured) == cfg["generations"]
    else:
        cfg = evolve.parse_args(ga_argv + ["--out", ga_out])
        if a.aircraft:
            sim.AIRCRAFT = a.aircraft  # set before evolve forks its worker pool
        aircraft = sim.AIRCRAFT

        # --- capture best-of-generation genomes without touching the GA's RNG stream ---
        captured = []
        orig_next = ga.next_generation

        def next_generation_spy(rng, ranked, gcfg):
            captured.append(np.array(ranked[0], copy=True))
            return orig_next(rng, ranked, gcfg)

        ga.next_generation = next_generation_spy
        try:
            summary = evolve.run(cfg)
        finally:
            ga.next_generation = orig_next
        captured.append(np.array(summary["genome"], dtype=float))
        assert len(captured) == cfg["generations"], (len(captured), cfg["generations"])
    t_ga = time.time() - t0

    csv_best = {}
    with open(os.path.join(ga_out, "fitness_history.csv")) as f:
        for row in csv.DictReader(f):
            csv_best[int(row["generation"])] = float(row["best_cost"])

    job = {"scenarios": summary["scenarios"], "record_scenario": a.record_scenario, "sample_hz": a.sample_hz,
           "aircraft": aircraft, "run_id": a.run_id, "seed": cfg["seed"], "csv_best": csv_best,
           "config": cfg, "traj_dir": traj_dir, "gzip": a.gzip}
    items = [(g, [float(x) for x in vec]) for g, vec in enumerate(captured)]
    with mp.Pool(a.log_workers or os.cpu_count() or 1, initializer=_init,
                 initargs=(flight_sim_dir, job)) as pool:
        entries = pool.map(_log_one, items)

    # --- verify recomputed fitness vs GA's CSV (printed with 6 decimals there) ---
    max_diff = 0.0
    for e in entries:
        ref = csv_best[e["generation"]]
        e["ga_fitness_reported"] = ref
        e["fitness_match"] = abs(e["fitness"] - ref) <= 5e-7 + 1e-9 * abs(ref)
        max_diff = max(max_diff, abs(e["fitness"] - ref))
    final_exact = abs(entries[-1]["fitness"] - summary["best_cost"])
    all_ok = all(e["fitness_match"] for e in entries) and final_exact == 0.0

    index = {
        "schema": trajlog.INDEX_SCHEMA,
        "traj_schema": trajlog.SCHEMA,
        "run_id": a.run_id,
        "aircraft": aircraft,
        "jsbsim_version": trajlog.jsbsim_version(),
        "git_sha": trajlog.git_sha(flight_sim_dir),
        "seed": cfg["seed"],
        "ga_config": cfg,
        "recorded_scenario": a.record_scenario,
        "fitness_kind": "cost (lower is better): mean over all GA scenarios",
        "fitness_verification": {"max_abs_diff_vs_csv": max_diff, "final_best_exact_diff": final_exact,
                                 "all_match": all_ok,
                                 "note": "CSV stores 6 decimals; the final best is compared to best_gains.json exactly"},
        "created_by": "sim-bridge/export_run.py",
        # entries carry exactly {generation, fitness, aircraft, file} (shared index schema);
        # per-generation extras live in `details`, keyed by generation.
        "entries": [{k: e[k] for k in ("generation", "fitness", "aircraft", "file")} for e in entries],
        "details": {str(e["generation"]): {k: e[k] for k in ("status", "samples", "duration_s",
                                                             "ga_fitness_reported", "fitness_match")}
                    for e in entries},
    }
    with open(os.path.join(traj_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)

    print(f"\nGA: {cfg['generations']} gens, pop {cfg['pop_size']}, seed {cfg['seed']}, "
          f"{cfg['scenarios']} scenarios in {t_ga:.1f}s; logged {len(entries)} trajectories in "
          f"{time.time() - t0 - t_ga:.1f}s")
    for e in entries:
        print(f"  g{e['generation']:03d}  fitness {e['fitness']:.6f}  (GA csv {e['ga_fitness_reported']:.6f})  "
              f"{'OK' if e['fitness_match'] else 'MISMATCH'}  {e['file']}")
    print(f"fitness check: max |diff| vs CSV = {max_diff:.2e}; final best exact diff = {final_exact:.2e}; "
          f"{'ALL MATCH' if all_ok else 'MISMATCH'}")
    print(f"index: {os.path.join(traj_dir, 'index.json')}")
    update_runs_list(a.out_root)
    return 0 if all_ok else 1


def update_runs_list(out_root: str) -> str:
    """(Re)write <out_root>/runs.json listing every <run_id>/trajectories/index.json (used by the viewer)."""
    runs = []
    for rid in sorted(os.listdir(out_root)):
        idx = os.path.join(out_root, rid, "trajectories", "index.json")
        if not os.path.isfile(idx):
            continue
        with open(idx) as f:
            meta = json.load(f)
        ents = meta.get("entries") or meta.get("trajectories") or []
        runs.append({"run_id": meta.get("run_id", rid), "aircraft": meta.get("aircraft"),
                     "generations": len(ents), "seed": meta.get("seed"),
                     "best_fitness": min((e["fitness"] for e in ents), default=None),
                     "index": f"{rid}/trajectories/index.json"})
    path = os.path.join(out_root, "runs.json")
    with open(path, "w") as f:
        json.dump({"runs": runs}, f, indent=1)
    return path


if __name__ == "__main__":
    sys.exit(main())
