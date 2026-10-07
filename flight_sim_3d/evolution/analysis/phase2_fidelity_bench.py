"""CPU per scenario at rigid / reduced / full and Spearman rank correlations, re-scoring a sample of a finished run's
genomes (read-only: direct worker calls, nothing goes into the evaluation cache or the run dir).
  $PY evolution/analysis/phase2_fidelity_bench.py RUN_DIR [--per-aircraft 16] [--workers 8] [--json OUT]
Sample: unique genomes of the run spread over generations and ranks (deterministic). Full costs are re-computed too and
compared with the run's logged costs (bit-for-bit check). Reduced uses the run's per-aircraft reduced_gate."""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from evolution import batch, eval as ev, fidelity as F, sim  # noqa: E402

FIDS = ("rigid", "reduced", "full")


def sample(rows, n):
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: (r["generation"], r["rank"])):
        k = tuple(np.round(r["genome_norm"], 12))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    if len(uniq) <= n:
        return uniq
    idx = np.unique(np.linspace(0, len(uniq) - 1, n).round().astype(int))
    return [uniq[i] for i in idx]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--per-aircraft", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    run = ev.load_run_cfg(os.path.join(a.run_dir, "run.json"))
    rows = [json.loads(x) for x in open(os.path.join(a.run_dir, "genomes.jsonl")) if x.strip()]
    load0, t0 = os.getloadavg(), time.perf_counter()
    jobs = []
    for ent in run["aircraft"]:
        name = ent["name"]
        prof_d = ev._profile_d(ent)
        scs = ev.scenario_dicts(name, None, run)
        groups = ev.gene_groups(ent)
        gate = ent.get("reduced_gate", F.per_aircraft(name)["reduced_gate"])
        for r in sample([r for r in rows if r["aircraft"] == name], a.per_aircraft):
            gains, struct = ev.split_values(r["genome"], groups)
            for f in FIDS:
                jobs.append((name, r["individual_id"], r["cost"], len(scs), f, (prof_d, gains, struct, scs, f, gate)))
    ctx = mp.get_context("forkserver")
    res = {}
    with cf.ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=sim.worker_init) as pool:
        futs = {pool.submit(ev.task_genome, *j[5]): j for j in jobs}
        for fu in cf.as_completed(futs):
            j = futs[fu]
            r = fu.result()
            res[(j[0], j[1], j[4])] = {"cost": r["cost"], "status": r.get("status"), "feasible": r.get("feasible"),
                                       "cpu_s": r["task_cpu_s"], "n_scen": j[3], "logged_full": j[2],
                                       "model_version": r.get("model_version")}
    out = {"run_dir": a.run_dir, "wall_s": time.perf_counter() - t0, "load_start": load0, "load_end": os.getloadavg(),
           "workers": a.workers, "aircraft": {}}
    for ent in run["aircraft"]:
        name = ent["name"]
        ids = sorted({k[1] for k in res if k[0] == name})
        d = {"n": len(ids)}
        for f in FIDS:
            cpu = [res[(name, i, f)]["cpu_s"] / res[(name, i, f)]["n_scen"] for i in ids]
            d[f"cpu_per_scenario_{f}_s"] = {"mean": float(np.mean(cpu)), "median": float(np.median(cpu)),
                                            "min": float(np.min(cpu)), "max": float(np.max(cpu))}
            d[f"model_version_{f}"] = sorted({res[(name, i, f)]["model_version"] for i in ids})
        c = {f: [res[(name, i, f)]["cost"] for i in ids] for f in FIDS}
        ok = [k for k, i in enumerate(ids) if all(res[(name, i, f)]["status"] == "ok" for f in FIDS)]
        d["spearman_reduced_full_all"] = batch.spearman(c["reduced"], c["full"])
        d["spearman_rigid_full_all"] = batch.spearman(c["rigid"], c["full"])
        d["spearman_reduced_full_ok"] = batch.spearman([c["reduced"][k] for k in ok], [c["full"][k] for k in ok])
        d["spearman_rigid_full_ok"] = batch.spearman([c["rigid"][k] for k in ok], [c["full"][k] for k in ok])
        d["n_ok_all_fidelities"] = len(ok)
        d["full_matches_logged"] = sum(1 for i in ids if res[(name, i, "full")]["cost"] == res[(name, i, "full")]["logged_full"])
        d["invalid"] = {f: sum(1 for i in ids if res[(name, i, f)]["status"] != "ok") for f in FIDS}
        out["aircraft"][name] = d
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
