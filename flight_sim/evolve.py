#!/usr/bin/env python3
"""Evolve altitude-hold autopilot gains for a JSBSim Cessna 172 with a GA.

Example:
    python evolve.py --pop-size 40 --generations 25 --seed 1 --out results/example

Settings come from defaults, then an optional JSON config file (--config),
then any CLI flags given explicitly. Runs are deterministic for a given seed
and config (the worker count does not affect results).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ga  # noqa: E402
import genome  # noqa: E402
import sim  # noqa: E402

DEFAULTS = {
    "pop_size": 40,
    "generations": 25,
    "elite": 2,
    "selection_p": 0.2,
    "crossover": "uniform",
    "blx_alpha": 0.3,
    "mutation_rate": 0.15,
    "mutation_sigma": 0.08,
    "mutation_mode": "gauss",
    "seed": 1,
    "scenarios": 3,
    "scenario_seed": None,  # defaults to seed
    "workers": 0,           # 0 = os.cpu_count()
    "out": "results/run",
}

_SCENARIOS: List[sim.Scenario] = []


def _init_worker(scenarios):
    global _SCENARIOS
    _SCENARIOS = scenarios


def _eval_genome(g):
    return sim.evaluate(genome.decode(g), _SCENARIOS)


def parse_args(argv=None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file with any of the settings below (CLI flags override it)")
    p.add_argument("--pop-size", type=int)
    p.add_argument("--generations", type=int)
    p.add_argument("--elite", type=int, help="individuals copied unchanged to the next generation")
    p.add_argument("--selection-p", type=float, help="rank-selection pressure (car GA uses 0.2)")
    p.add_argument("--crossover", choices=["uniform", "blx"])
    p.add_argument("--blx-alpha", type=float)
    p.add_argument("--mutation-rate", type=float, help="per-gene mutation probability")
    p.add_argument("--mutation-sigma", type=float, help="Gaussian step size in normalized [0,1] gene units")
    p.add_argument("--mutation-mode", choices=["gauss", "reset"], help="'reset' = car GA's full U[0,1] redraw")
    p.add_argument("--seed", type=int)
    p.add_argument("--scenarios", type=int, help="number of disturbance cases averaged per evaluation")
    p.add_argument("--scenario-seed", type=int)
    p.add_argument("--workers", type=int, help="parallel processes (0 = all cores)")
    p.add_argument("--out", help="output directory")
    a = p.parse_args(argv)

    cfg = dict(DEFAULTS)
    if a.config:
        with open(a.config) as f:
            file_cfg = json.load(f)
        unknown = set(file_cfg) - set(DEFAULTS)
        if unknown:
            p.error(f"unknown config keys: {sorted(unknown)}")
        cfg.update(file_cfg)
    for k in DEFAULTS:
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    if cfg["scenario_seed"] is None:
        cfg["scenario_seed"] = cfg["seed"]
    if cfg["elite"] >= cfg["pop_size"]:
        p.error("elite must be smaller than pop_size")
    return cfg


def run(cfg: Dict) -> Dict:
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    gcfg = ga.GAConfig(**{k: cfg[k] for k in asdict(ga.GAConfig()).keys()})
    scenarios = sim.make_scenarios(cfg["scenarios"], cfg["scenario_seed"])
    workers = cfg["workers"] or os.cpu_count() or 1

    pop = ga.generation_zero(rng, cfg["pop_size"], genome.N_GENES)
    cache: Dict[bytes, Dict] = {}  # sims are deterministic, so elites needn't be re-flown
    history = []
    gen0_best = None
    t_start = time.time()

    csv_path = os.path.join(out, "fitness_history.csv")
    with mp.Pool(workers, initializer=_init_worker, initargs=(scenarios,)) as pool, open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv, lineterminator="\n")
        w.writerow(["generation", "best_cost", "mean_cost", "median_cost", "n_failed", "evaluations", "elapsed_s"]
                   + [f"best_{n}" for n in genome.GENE_NAMES])
        for gen in range(cfg["generations"]):
            todo = [g for g in pop if g.tobytes() not in cache]
            # de-duplicate while preserving order
            uniq = list({g.tobytes(): g for g in todo}.values())
            for g, r in zip(uniq, pool.map(_eval_genome, uniq)):
                cache[g.tobytes()] = r
            results = [cache[g.tobytes()] for g in pop]
            costs = np.array([r["cost"] for r in results])
            order = ga.rank_order(costs)
            pop, costs, results = pop[order], costs[order], [results[i] for i in order]
            n_failed = sum(any(s["status"] != "ok" for s in r["per_scenario"]) for r in results)
            best_gains = genome.decode(pop[0])
            row = {
                "generation": gen, "best_cost": float(costs[0]), "mean_cost": float(costs.mean()),
                "median_cost": float(np.median(costs)), "n_failed": int(n_failed), "evaluations": len(uniq),
                "elapsed_s": round(time.time() - t_start, 1),
            }
            history.append(row)
            w.writerow([row["generation"], f"{row['best_cost']:.6f}", f"{row['mean_cost']:.6f}",
                        f"{row['median_cost']:.6f}", row["n_failed"], row["evaluations"], row["elapsed_s"]]
                       + [f"{best_gains[n]:.6g}" for n in genome.GENE_NAMES])
            fcsv.flush()
            print(f"gen {gen:3d}  best {costs[0]:10.4f}  mean {costs.mean():10.4f}  median {np.median(costs):10.4f}"
                  f"  failed {n_failed:3d}/{len(pop)}  ({row['elapsed_s']}s)", flush=True)
            if gen == 0:
                gen0_best = pop[0].copy()
            if gen < cfg["generations"] - 1:
                pop = ga.next_generation(rng, pop, gcfg)

    best = pop[0]
    best_res = cache[best.tobytes()]
    summary = {
        "aircraft": sim.AIRCRAFT,
        "best_cost": best_res["cost"],
        "initial_best_cost": history[0]["best_cost"],
        "gains": genome.decode(best),
        "gain_units": {g.name: g.units for g in genome.SCHEMA},
        "genome": [float(x) for x in best],
        "per_scenario": best_res["per_scenario"],
        "initial_best_gains": genome.decode(gen0_best),
        "scenarios": [asdict(s) for s in scenarios],
        "config": cfg,
    }
    with open(os.path.join(out, "best_gains.json"), "w") as f:
        json.dump(summary, f, indent=2)

    import plot_results  # imported late so evolution doesn't need matplotlib loaded per worker
    plot_results.plot_all(out, summary, history, scenarios)
    print(f"done in {time.time() - t_start:.1f}s; best cost {best_res['cost']:.4f}; results in {out}")
    return summary


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
