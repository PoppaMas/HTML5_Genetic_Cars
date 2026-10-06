#!/usr/bin/env python3
"""NSGA-II driver that reuses the unmodified ga.py operators and the task's fitness.

    python evolve_pareto.py --task altitude_hold_pareto --pop-size 24 --generations 6 --seed 1 --out runs/pareto

Loop: parents ordered by (constrained front, crowding) -> ga.next_generation
(rank selection, uniform crossover, Gaussian mutation, exactly as in the scalar
GA) makes N offspring -> evaluate -> (mu + lambda) NSGA-II environmental
selection back to N. Writes pareto_front.json/.csv and a history CSV.
evolve.py is untouched; this is a separate entry point.
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import adapter  # noqa: E402
import nsga2  # noqa: E402
from flightsim_path import orig_ga  # noqa: E402

_TASK = None
_SCEN = None


def _init(task, scen):
    global _TASK, _SCEN
    _TASK, _SCEN = task, scen


def _eval(g):
    return _TASK.evaluate(_TASK.spec.decode(g), _SCEN)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="altitude_hold_pareto")
    ap.add_argument("--pop-size", type=int, default=24)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scenarios", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="runs/pareto")
    a = ap.parse_args(argv)

    task = adapter.load_task(a.task, {"fitness": {"mode": "pareto"}})
    for w in task.warnings:
        print(f"[genome] WARNING: {w}")
    names = task.fitness.pareto_objectives
    ga = orig_ga()
    gcfg = ga.GAConfig(pop_size=a.pop_size)
    rng = np.random.default_rng(a.seed)
    scen = task.make_scenarios(a.scenarios, a.seed)
    os.makedirs(a.out, exist_ok=True)
    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)
    cache = {}
    t0 = time.time()

    def evaluate_all(pool, pop):
        todo = list({g.tobytes(): g for g in pop if g.tobytes() not in cache}.values())
        for g, r in zip(todo, pool.map(_eval, todo)):
            cache[g.tobytes()] = r
        res = [cache[g.tobytes()] for g in pop]
        F = np.array([r["pareto"] for r in res], float)
        V = np.array([r["violation"] for r in res], float)
        return res, F, V

    hist = []
    with mp.Pool(a.workers or os.cpu_count(), initializer=_init, initargs=(task, scen)) as pool:
        pop = ga.generation_zero(rng, a.pop_size, task.spec.n_genes)
        res, F, V = evaluate_all(pool, pop)
        for gen in range(a.generations):
            order = nsga2.rank_population(F, V)
            pop, F, V, res = pop[order], F[order], V[order], [res[i] for i in order]
            fronts = nsga2.non_dominated_sort(F, V)
            feas = V == 0
            row = {"generation": gen, "front0_size": len(fronts[0]), "n_infeasible": int((~feas).sum()),
                   "elapsed_s": round(time.time() - t0, 1)}
            for j, n in enumerate(names):
                row[f"min_{n}"] = float(np.min(F[feas, j])) if feas.any() else float("nan")
            hist.append(row)
            print(f"gen {gen:3d} front0 {row['front0_size']:3d} infeasible {row['n_infeasible']:3d} "
                  + " ".join(f"min_{n} {row[f'min_{n}']:.4g}" for n in names) + f" ({row['elapsed_s']}s)", flush=True)
            if gen == a.generations - 1:
                break
            kids = ga.next_generation(rng, pop, gcfg)[gcfg.elite:]  # elites already survive via (mu+lambda)
            kres, kF, kV = evaluate_all(pool, kids)
            allpop = np.vstack([pop, kids])
            allF, allV, allres = np.vstack([F, kF]), np.concatenate([V, kV]), res + kres
            _, uniq = np.unique(allpop, axis=0, return_index=True)  # drop duplicate genomes
            uniq = np.sort(uniq)
            allpop, allF, allV, allres = allpop[uniq], allF[uniq], allV[uniq], [allres[i] for i in uniq]
            keep = nsga2.select(allF, a.pop_size, allV)
            pop, F, V, res = allpop[keep], allF[keep], allV[keep], [allres[i] for i in keep]

    front = nsga2.non_dominated_sort(F, V)[0]
    pf = []
    for i in front:
        gains = task.spec.decode(pop[i])
        pf.append({"genome": [float(x) for x in pop[i]], "gains": gains,
                   "objectives": dict(zip(names, map(float, F[i]))), "scalar_cost": res[i]["cost"],
                   "feasible": bool(V[i] == 0)})
    pf.sort(key=lambda d: d["scalar_cost"])
    json.dump({"task": task.raw, "objectives": names, "front": pf,
               "scenarios": [asdict(s) for s in scen], "history": hist,
               "at_bounds_front": task.spec.at_bounds(pop[front])}, open(os.path.join(a.out, "pareto_front.json"), "w"), indent=2)
    with open(os.path.join(a.out, "pareto_front.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names + ["scalar_cost"] + task.spec.names)
        for d in pf:
            w.writerow([f"{d['objectives'][n]:.6g}" for n in names] + [f"{d['scalar_cost']:.6g}"]
                       + [f"{d['gains'][n]:.6g}" for n in task.spec.names])
    print(f"done: {len(pf)} non-dominated solutions -> {a.out}/pareto_front.json; "
          f"lowest scalar cost {pf[0]['scalar_cost']:.4f}")


if __name__ == "__main__":
    main()
