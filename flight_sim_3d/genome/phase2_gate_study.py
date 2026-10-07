"""Phase 2: fraction of generation-0 genomes that fail the flex-v2 margin gate, uniform vs baseline-seeded init.

Only FD's no-flight margin precheck runs (fd_bridge.precheck_v2: conservative min over bodies and methods, gate 1.0),
so this is cheap (~1 s per genome). Writes runs/phase2_gate_study.json.

  python phase2_gate_study.py [--n 64] [--seed 1] [--sigma 0.05] [--aircraft c172x t38 b737]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True


def _one(args):
    import fd_bridge
    model, genes, asym = args
    p = fd_bridge.precheck_v2(model, genes, asym)
    m = p["margins"]
    return {"fail": p["fail"], "min_margin": m["min_margin"], "binding": m["binding"],
            "penalty": sum(p["terms"].values()), "mass_frac": p["mass"]["total_frac"]}


def study(aircraft, n, seed, sigma, asym=False, workers=8):
    import adapter
    import init_pop
    t = adapter.load_task("phase2_flex", {"aircraft": aircraft, "flex": {"asymmetric": asym}})
    sidx = [j for j, g in enumerate(t.spec.genes) if g.block == "structure_v2"]
    out = {}
    for mode in ("uniform", "baseline"):
        init = {**t.init, "mode": mode, "sigma": sigma, "seed_runs": []}
        pop = init_pop.generation_zero(np.random.default_rng(seed), n, t.spec.n_genes, t.spec, init)
        jobs = []
        for row in pop:
            g = t.spec.decode(row)
            jobs.append((t.profile.jsbsim_model, {t.spec.genes[j].name: g[t.spec.genes[j].name] for j in sidx}, asym))
        t0 = time.time()
        with mp.Pool(workers) as pool:
            res = pool.map(_one, jobs)
        mm = np.array([r["min_margin"] for r in res])
        out[mode] = {"n": n, "n_fail": sum(r["fail"] is not None for r in res),
                     "fail_kinds": {k: sum(r["fail"] == k for r in res) for k in {r["fail"] for r in res} if k},
                     "n_penalized": int(sum((r["fail"] is None) and r["penalty"] > 0 for r in res)),
                     "min_margin_quartiles": np.percentile(mm, [0, 25, 50, 75, 100]).round(3).tolist(),
                     "mass_frac_mean": float(np.mean([r["mass_frac"] for r in res])),
                     "wall_s": round(time.time() - t0, 1)}
        print(aircraft, mode, out[mode], flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--aircraft", nargs="+", default=["c172x", "t38", "b737"])
    ap.add_argument("--asymmetric", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "phase2_gate_study.json"))
    a = ap.parse_args(argv)
    res = {ac: study(ac, a.n, a.seed, a.sigma, a.asymmetric) for ac in a.aircraft}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"args": vars(a), "results": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
