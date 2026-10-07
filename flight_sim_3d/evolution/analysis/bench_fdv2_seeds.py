"""Across seeds 1-3 of the FD v2 ladder benchmark: results.json (seed 1), results-s2.json, results-s3.json and the
matching report*.json (counterfactual). Writes bench_results/fdv2/seeds.json + seeds.md."""
import json, os, statistics as st
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_results", "fdv2")
tags = {1: "", 2: "-s2", 3: "-s3"}
res = {sd: json.load(open(os.path.join(OUT, f"results{t}.json"))) for sd, t in tags.items() if os.path.exists(os.path.join(OUT, f"results{t}.json"))}
rep = {sd: json.load(open(os.path.join(OUT, f"report{t}.json"))) for sd, t in tags.items() if os.path.exists(os.path.join(OUT, f"report{t}.json"))}
cases = ["full-vizoff", "rigid-full", "reduced-full", "rigid-reduced-full"]
acs = ["c172x", "737"]
out = {"seeds": sorted(res), "cases": {}, "counterfactual": {}}
L = [f"Seeds {sorted(res)}; pop 32 x 4 gens; full-only = full-vizoff. Δ = ladder best full cost - full-only best (same seed).", "",
     "| case | aircraft | best full cost per seed | mean | Δ per seed | mean Δ | feasible rate (full-scored) per seed | batch wall s per seed (load start→end) | task CPU s per seed |",
     "|---|---|---|---|---|---|---|---|---|"]
for c in cases:
    for ac in acs:
        bests, deltas, feas, walls, cpus = [], [], [], [], []
        for sd in sorted(res):
            r = res[sd].get(c)
            if not r or "error" in r:
                continue
            p, b = r["per_aircraft"][ac], res[sd]["full-vizoff"]["per_aircraft"][ac]
            bests.append(p["best_final"]); deltas.append(p["best_final"] - b["best_final"]); feas.append(p["feasible_rate_full_scored"])
            walls.append(f"{r['batch_wall_s']:.0f} ({r['load_start'][0]:.1f}→{r['load_end'][0]:.1f})"); cpus.append(p["task_cpu_s"])
        out["cases"].setdefault(c, {})[ac] = {"best": bests, "delta": deltas, "feasible_rate": feas, "task_cpu_s": cpus, "wall": walls}
        L.append(f"| {c} | {ac} | {', '.join(f'{x:.5f}' for x in bests)} | {st.mean(bests):.5f} | {', '.join(f'{x:+.5f}' for x in deltas)} | "
                 f"{st.mean(deltas):+.5f} | {', '.join(f'{x:.2f}' for x in feas)} | {'; '.join(walls)} | {', '.join(f'{x:.0f}' for x in cpus)} |")
L += ["", "Counterfactual (identical full-only populations; all gens of all seeds):", "",
      "| aircraft | ladder | gens elite set differs | gens best differs | mean elite overlap |", "|---|---|---|---|---|"]
for ac in acs:
    for lad in ["rigid-full", "reduced-full", "rigid-reduced-full"]:
        d = b_ = n = 0; ov = []
        for sd, r in rep.items():
            for ge in r["per_aircraft"][ac]["gens"]:
                n += 1; d += 0 if ge[f"{lad}_elite_same"] else 1; b_ += 0 if ge[f"{lad}_best_same"] else 1
            ov.append(r["per_aircraft"][ac]["counterfactual"][lad]["mean_elite_overlap"])
        out["counterfactual"].setdefault(ac, {})[lad] = {"elite_differs": d, "best_differs": b_, "gens": n, "mean_overlap": st.mean(ov) if ov else None}
        L.append(f"| {ac} | {lad} | {d}/{n} | {b_}/{n} | {st.mean(ov):.2f} |")
L += ["", "| aircraft | Spearman over 32 genomes, all gens of all seeds: reduced vs full (both ok) | rigid vs full (both ok) | reduced pass & full fail (of n) |", "|---|---|---|---|"]
for ac in acs:
    g = [ge for r in rep.values() for ge in r["per_aircraft"][ac]["gens"]]
    rng = lambda k: f"{min(x[k] for x in g):+.2f}…{max(x[k] for x in g):+.2f} (median {st.median(x[k] for x in g):+.2f})"
    L.append(f"| {ac} | {rng('spearman_reduced_vs_full')} ({rng('spearman_reduced_vs_full_both_ok')}) | {rng('spearman_rigid_vs_full')} "
             f"({rng('spearman_rigid_vs_full_both_ok')}) | {sum(x['reduced_ok_full_fail'] for x in g)} of {sum(x['n'] for x in g)} |")
    out["counterfactual"][ac]["spearman_gens"] = [{k: x[k] for k in x if k.startswith("spearman") or k in ("generation", "reduced_ok_full_fail", "n")} for x in g]
json.dump(out, open(os.path.join(OUT, "seeds.json"), "w"), indent=1)
open(os.path.join(OUT, "seeds.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
