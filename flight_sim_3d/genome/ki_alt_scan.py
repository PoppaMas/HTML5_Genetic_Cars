"""ki_alt upper-bound study for the phase1_v5 preset (v5-only override; v4's bound stays 0.05).

Sweeps ki_alt on the v5 best genomes (other genes fixed) under the phase1_v5 task, on the run's own scenario seed and
on an unseen seed (+100). Writes runs/ki_alt_scan.json.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
RUNS = {"c172x": "runs/v5_sweep_w01", "t38": "runs/v5_t38_s1", "b737": "runs/v5_b737_s1"}
GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]


def _one(job):
    import adapter
    ac, ki, ss = job
    t = adapter.load_task("phase1_v5", {"aircraft": ac, "gene_overrides": {"ki_alt": {"max": 1.0}}})
    bg = json.load(open(os.path.join(RUNS[ac], "best_gains.json")))
    g = dict(bg["gains"], ki_alt=ki)
    r = t.evaluate(g, t.make_scenarios(3, ss))
    return {"aircraft": ac, "ki_alt": ki, "scenario_seed": ss, "cost": r["cost"],
            "track_alt": r["objectives"].get("track_alt"), "hold_osc": r["objectives"].get("hold_osc"),
            "status": [p["status"] for p in r["per_scenario"]]}


if __name__ == "__main__":
    jobs = []
    for ac, run in RUNS.items():
        ss = json.load(open(os.path.join(run, "best_gains.json")))["config"]["scenario_seed"]
        jobs += [(ac, k, s) for k in GRID for s in (ss, ss + 100)]
    with mp.Pool(int(os.environ.get("WORKERS", "3"))) as pool:
        res = pool.map(_one, jobs)
    json.dump(res, open("runs/ki_alt_scan.json", "w"), indent=1)
    for r in res:
        print(r["aircraft"], r["scenario_seed"], r["ki_alt"], round(r["cost"], 5), r["track_alt"] and round(r["track_alt"], 5),
              r["hold_osc"], set(r["status"]))
