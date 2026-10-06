#!/usr/bin/env python3
"""f16 gain-bounds probe (Phase-1 task, FD jsbsim_root): 1-D sweeps through each seed's best genome, extended
beyond the current bounds, plus a uniform random sample of the current box (invalid fraction per gene decile).
Writes analysis/f16_bounds_probe.json. Read-only w.r.t. runs/."""
import concurrent.futures as cf
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from evolution import genome, sim  # noqa: E402

RUNS = os.path.join(os.path.dirname(HERE), "runs")


def mean_eval(args):
    prof_d, gains, scs = args
    rs = [sim.eval_task(prof_d, gains, sc) for sc in scs]
    return {"cost": float(np.mean([r["cost"] for r in rs])), "status": [r["status"] for r in rs]}


def main():
    cfg = json.load(open(os.path.join(RUNS, "phase1-f16fd-s1", "config.json")))["resolved"]
    prof_d = cfg["aircraft"][0]["resolved_profile"]
    P = sim.Profile.from_dict(prof_d)
    schema = genome.make_schema(P.gain_bounds, P.gene_kinds)
    scs = [s.to_dict() for s in sim.make_scenarios(cfg["scenarios"], cfg["scenario_seed"], P)]
    bests = {s: json.load(open(os.path.join(RUNS, f"phase1-f16fd-s{s}", "summary.json")))["aircraft"][0]["best_gains"]
             for s in (1, 2, 3)}
    jobs, meta = [], []
    for s, g in bests.items():
        for gn in schema:
            lo, hi = gn.min, gn.max
            vals = list(np.geomspace(lo / 30, hi * 10, 29))
            if gn.kind == "log0":
                vals = [0.0] + vals
            for v in vals:
                jobs.append((prof_d, {**g, gn.name: float(v)}, scs))
                meta.append(("sweep", s, gn.name, float(v), bool(lo <= v <= hi or v == 0.0)))
    rng = np.random.default_rng(12345)
    U = rng.random((256, len(schema)))
    for u in U:
        jobs.append((prof_d, genome.decode(u, schema), scs))
        meta.append(("random", None, None, [float(x) for x in u], True))
    out = {"bounds": {gn.name: [gn.min, gn.max, gn.kind] for gn in schema}, "best_gains": bests,
           "best_cost_run": {s: json.load(open(os.path.join(RUNS, f"phase1-f16fd-s{s}", "summary.json")))["aircraft"][0]["best_fitness"]
                             for s in (1, 2, 3)},
           "sweeps": [], "random": []}
    with cf.ProcessPoolExecutor(8, mp_context=__import__("multiprocessing").get_context("forkserver"),
                                initializer=sim.worker_init) as ex:
        for m, r in zip(meta, ex.map(mean_eval, jobs, chunksize=4)):
            if m[0] == "sweep":
                out["sweeps"].append({"seed": m[1], "gene": m[2], "value": m[3], "in_bounds": m[4], **r})
            else:
                out["random"].append({"u": m[3], **r})
    json.dump(out, open(os.path.join(HERE, "f16_bounds_probe.json"), "w"), indent=0, default=lambda o: o.item())


if __name__ == "__main__":
    main()
