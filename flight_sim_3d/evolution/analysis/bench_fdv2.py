"""FD v2 fidelity-ladder benchmark (flexeval reduced/full): c172x + 737 (+T38 with --aircraft), pop 32, seed 1, cache off.

cases: full-only (viz on = the reference, and viz off), rigid->full, reduced->full, rigid->reduced->full.
usage: python evolution/analysis/bench_fdv2.py [--gens 4] [--only NAME,...] [--aircraft c172x,737]
Sequential batches, 8 workers, into evolution/bench_results/fdv2/runs; results.json records per case and aircraft:
gen wall, task CPU (sum of worker process CPU), child-process CPU (rusage), best full cost per gen, feasibility rate of
the full-scored individuals, Spearman per gen, the final elite set; and the box load (other agents share the box).
child_cpu_s (rusage of the batch subprocess) does NOT include the pool workers (not reaped into it); use task_cpu_s.
"""
import argparse, json, os, resource, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TEAM = os.path.dirname(PKG)
OUT = os.path.join(PKG, "bench_results", "fdv2")
PY = sys.executable
MF = lambda screen: {"enabled": True, "screen": screen, "top_k": 4}
CASES = [  # name, viz, mf
    ("full-vizon", "on", None),
    ("full-vizoff", "off", None),
    ("rigid-full", "off", MF("rigid")),
    ("reduced-full", "off", MF("reduced")),
    ("rigid-reduced-full", "off", MF(["rigid", "reduced"])),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--only")
    ap.add_argument("--aircraft", default="c172x,737")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    acs = a.aircraft.split(",")
    with open(os.path.join(PKG, "configs", "phase1_hdg.json")) as f:
        base = json.load(f)
    base["aircraft"] = [x for x in base["aircraft"] if x["name"] in acs]
    base["ga"] = dict(base["ga"], pop_size=32, generations=a.gens)
    base["seed"] = a.seed
    base["runs_dir"] = os.path.join(OUT, "runs")
    base["cache"] = {"enabled": False, "path": "cache/unused.sqlite"}
    base["trajectories"] = {"generations": [0, a.gens - 1], "scenario": 0, "sample_hz": 30}
    res_path = os.path.join(OUT, f"results{a.tag}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    only = set(a.only.split(",")) if a.only else None
    for name, viz, mf in CASES:
        if only and name not in only:
            continue
        cfg = dict(base, run_id=f"fdv2{a.tag}-{name}", fidelity="full", viz=viz, struct_genes=True)
        if mf:
            cfg["multi_fidelity"] = mf
        p = os.path.join(OUT, f"cfg{a.tag}-{name}.json")
        json.dump(cfg, open(p, "w"), indent=1)
        rd = os.path.join(base["runs_dir"], cfg["run_id"])
        if os.path.exists(rd):
            subprocess.run(["rm", "-rf", rd], check=True)
        ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        l0 = os.getloadavg(); t0 = time.perf_counter(); s0 = time.strftime("%H:%M:%S")
        r = subprocess.run([PY, "-m", "evolution.batch", "--config", p, "--workers", "8"], cwd=TEAM,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
        wall = time.perf_counter() - t0
        l1 = os.getloadavg(); s1 = time.strftime("%H:%M:%S")
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        open(os.path.join(OUT, f"log{a.tag}-{name}.txt"), "w").write(r.stdout + r.stderr)
        if r.returncode:
            print(name, "FAILED", r.stderr[-3000:], flush=True); results[name] = {"error": r.stderr[-3000:]}; continue
        hist = [json.loads(x) for x in open(os.path.join(rd, "history.jsonl"))]
        rows = [json.loads(x) for x in open(os.path.join(rd, "genomes.jsonl"))]
        sm = json.load(open(os.path.join(rd, "summary.json")))
        G = a.gens
        per = {}
        for ac in acs:
            h = [x for x in hist if x["aircraft"] == ac]
            rr = [x for x in rows if x["aircraft"] == ac]
            fullrows = [x for x in rr if x.get("feasibility_fidelity") == "full"]
            per[ac] = {"best_final": h[-1]["best"], "best_per_gen": [x["best"] for x in h],
                       "best_feasible_final": h[-1].get("best_feasible"),
                       "gen_wall_s": [x["eval_wall_s"] for x in h], "task_cpu_s": sum(x["eval_cpu_s"] for x in h),
                       "n_full_scored": len(fullrows), "n_full_ok": sum(1 for x in fullrows if x["status"] == "ok"),
                       "feasible_rate_full_scored": (sum(1 for x in fullrows if x["status"] == "ok") / len(fullrows)) if fullrows else None,
                       "status_full_scored": {s: sum(1 for x in fullrows if x["status"] == s) for s in {x["status"] for x in fullrows}},
                       "final_elites": [x["genome_norm"] for x in sorted([x for x in rr if x["generation"] == G - 1 and x["is_elite"]], key=lambda x: x["rank"])],
                       "final_elite_costs": [x["cost"] for x in sorted([x for x in rr if x["generation"] == G - 1 and x["is_elite"]], key=lambda x: x["rank"])],
                       "spearman": [x.get("spearman") for x in h], "spearman_both_ok": [x.get("spearman_both_ok") for x in h],
                       "n_rescored": [x.get("n_rescored") for x in h], "k_full": [x.get("k_full") for x in h],
                       "resim_mismatches": next((x.get("resim_mismatches") for x in sm["aircraft"] if x["aircraft"] == ac), None)}
            if mf:
                per[ac]["stage_cpu_s"] = {f: sum(x["stages"][f]["eval_cpu_s"] for x in h) for f in h[0]["ladder"]}
                per[ac]["stage_wall_s"] = {f: sum(x["stages"][f]["eval_wall_s"] for x in h) for f in h[0]["ladder"]}
                per[ac]["stage_n"] = {f: sum(x["stages"][f]["n"] for x in h) for f in h[0]["ladder"]}
        results[name] = {"viz": viz, "multi_fidelity": mf, "wall_s": wall, "batch_wall_s": sm["wall_s_this_session"],
                         "start": s0, "end": s1, "load_start": l0, "load_end": l1,
                         "child_cpu_s": (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime),
                         "per_aircraft": per, "run_dir": rd}
        json.dump(results, open(res_path, "w"), indent=1)
        print(f"{name:20s} {s0}-{s1} wall {wall:7.1f}s cpu {results[name]['child_cpu_s']:7.0f}s load {l0[0]:.1f}->{l1[0]:.1f} "
              + "  ".join(f"{ac} best {per[ac]['best_final']:.5f} feas {per[ac]['feasible_rate_full_scored']}" for ac in per), flush=True)


if __name__ == "__main__":
    main()
