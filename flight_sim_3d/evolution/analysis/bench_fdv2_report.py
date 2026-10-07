"""Summarise bench_results/fdv2/results.json and run the counterfactual ladder analysis.

Counterfactual: every genome of the full-only (viz off) run (all 32 per generation, all at full) is also scored at rigid
and at reduced (same per-aircraft gate as the batch), through evolution.fidelity.evaluate_genome in a process pool.
Per generation this gives (a) Spearman rigid/reduced vs full over the whole population (comparable with FD's
32-genome table, INTERFACE_v2 §8), and (b) for each ladder, the elite set the ladder's selection rule would pick
(batch._ladder rule: stage 0 all, k_mid / k_full + carried elites, full-scored first by full cost) vs the true full-only
elite set, on identical populations.
usage: python evolution/analysis/bench_fdv2_report.py [--tag ""] [--workers 8]
writes bench_results/fdv2/report{tag}.json and report{tag}.md
"""
import argparse, concurrent.futures as cf, json, math, multiprocessing as mp, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TEAM = os.path.dirname(PKG)
sys.path.insert(0, TEAM)
OUT = os.path.join(PKG, "bench_results", "fdv2")


def score(args):
    prof_d, gains, struct, scs, fid, gate = args
    from evolution import fidelity as F
    r = F.evaluate_genome(prof_d, gains, struct, scs, fid, gate)
    return r["cost"], r["status"]


def ladder_pick(full, scr, carried, ladder, k_full, k_mid, elite):
    """indices of the elite set the ladder would select; scr[f][i] = (cost, status)."""
    n = len(full)
    idx = list(range(n))
    seen = {}
    for si, f in enumerate(ladder):
        seen[f] = set(idx)
        if si + 1 < len(ladder):
            keep = k_full if si + 1 == len(ladder) - 1 else k_mid
            cost = (lambda i: full[i][0]) if f == "full" else (lambda i, f=f: scr[f][i][0])
            ordr = sorted(idx, key=lambda i: (cost(i), i))
            idx = sorted(set(ordr[:keep]) | carried)
    fs = sorted(seen["full"], key=lambda i: (full[i][0], i))
    rest = []
    for f in reversed(ladder[:-1]):
        lv = sorted([i for i in seen[f] if i not in seen["full"] and i not in rest], key=lambda i: (scr[f][i][0], i))
        rest += lv
    order = fs + [i for i in rest]
    return order[:elite], len(seen["full"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from evolution import batch, fidelity as F, genome as GM, eval as EV, runinfo, sim
    from evolution.batch import spearman
    res = json.load(open(os.path.join(OUT, f"results{a.tag}.json")))
    base = res["full-vizoff"]
    ref = res.get("full-vizon", base)          # seeds 2/3 have no viz-on case: speedups there are vs full-only viz off
    rd = base["run_dir"]
    run_cfg = EV.load_run_cfg(rd)
    rows = runinfo.read_rows(os.path.join(rd, "genomes.jsonl"))
    cfg = json.load(open(os.path.join(rd, "config.json")))["resolved"]
    elite, pop = cfg["ga"]["elite"], cfg["ga"]["pop_size"]
    acs = [x["name"] for x in run_cfg["aircraft"]]
    # ---- counterfactual screen scores
    cf_path = os.path.join(OUT, f"counterfactual{a.tag}.json")
    cfx = json.load(open(cf_path)) if os.path.exists(cf_path) else {}
    jobs = []
    for ac in acs:
        ent = next(x for x in run_cfg["aircraft"] if x["name"] == ac)
        prof_d = ent["resolved_profile"]
        scs = [runinfo.scenario_fields(s) for s in EV.scenario_dicts(ac, None, run_cfg)]
        groups = EV.gene_groups(ent)
        for r in rows:
            if r["aircraft"] != ac:
                continue
            gains, struct = EV.split_values(r["genome"], groups)
            for fid in ("rigid", "reduced"):
                key = f"{ac}|{r['generation']}|{r['index']}|{fid}"
                if key not in cfx:
                    jobs.append((key, (prof_d, gains, struct, scs, fid, ent.get("reduced_gate"))))
    t0, l0 = time.perf_counter(), os.getloadavg()
    if jobs:
        ctx = mp.get_context("forkserver")
        from evolution import sim as _s
        with cf.ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_s.worker_init) as ex:
            futs = {ex.submit(score, j[1]): j[0] for j in jobs}
            for fu in cf.as_completed(futs):
                cfx[futs[fu]] = list(fu.result())
        json.dump(cfx, open(cf_path, "w"))
    cf_wall = time.perf_counter() - t0
    # ---- per aircraft analysis
    FD = {"c172x": {"reduced_vs_full": "0.942 / 0.940 (genomes / joint)", "both_ok": "0.909 (27) / 0.920 (29)",
                    "rigid_vs_full": "undefined / -0.047"},
          "737": {"reduced_vs_full": "0.843 / 0.870", "both_ok": "0.617 (23) / 0.585 (20)", "rigid_vs_full": "undefined / 0.158"}}
    rep = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "counterfactual_jobs": len(jobs), "counterfactual_wall_s": cf_wall,
           "counterfactual_load_start": l0, "counterfactual_load_end": os.getloadavg(), "cases": {}, "per_aircraft": {}}
    G = cfg["ga"]["generations"]
    ladders = {"rigid-full": ["rigid", "full"], "reduced-full": ["reduced", "full"],
               "rigid-reduced-full": ["rigid", "reduced", "full"]}
    for ac in acs:
        mff = F.per_aircraft(ac)["min_full_frac"]
        k_full = min(pop, max(4, math.ceil(mff * pop - 1e-9)))
        k_mid = min(pop, 2 * k_full)
        pa = {"k_full": k_full, "k_mid": k_mid, "reduced_gate": F.per_aircraft(ac)["reduced_gate"], "gens": []}
        diff = {L: 0 for L in ladders}
        overlap = {L: [] for L in ladders}
        best_hit = {L: 0 for L in ladders}
        sp_all = {"rigid": [], "reduced": []}
        for g in range(G):
            gr = sorted([r for r in rows if r["aircraft"] == ac and r["generation"] == g], key=lambda r: r["index"])
            full = [(r["cost"], r["status"]) for r in gr]
            scr = {f: [tuple(cfx[f"{ac}|{g}|{r['index']}|{f}"]) for r in gr] for f in ("rigid", "reduced")}
            carried = set(range(elite)) if g > 0 else set()
            true_el = sorted(range(len(gr)), key=lambda i: (full[i][0], i))[:elite]
            ge = {"generation": g, "n": len(gr), "n_full_ok": sum(1 for c in full if c[1] == "ok"),
                  "n_reduced_ok": sum(1 for c in scr["reduced"] if c[1] == "ok"),
                  "n_rigid_ok": sum(1 for c in scr["rigid"] if c[1] == "ok")}
            for f in ("rigid", "reduced"):
                s = spearman([c[0] for c in scr[f]], [c[0] for c in full])
                ok = [(x[0], y[0]) for x, y in zip(scr[f], full) if x[1] == "ok" and y[1] == "ok"]
                ge[f"spearman_{f}_vs_full"] = s
                ge[f"spearman_{f}_vs_full_both_ok"] = spearman([p[0] for p in ok], [p[1] for p in ok])
                ge[f"n_both_ok_{f}"] = len(ok)
                sp_all[f].append(s)
            # reduced passes but full fails (gate optimism) / reduced fails but full ok
            ge["reduced_ok_full_fail"] = sum(1 for x, y in zip(scr["reduced"], full) if x[1] == "ok" and y[1] != "ok")
            ge["reduced_fail_full_ok"] = sum(1 for x, y in zip(scr["reduced"], full) if x[1] != "ok" and y[1] == "ok")
            for L, lad in ladders.items():
                pick, nf = ladder_pick(full, scr, carried, lad, k_full, k_mid, elite)
                same = set(pick) == set(true_el)
                diff[L] += 0 if same else 1
                overlap[L].append(len(set(pick) & set(true_el)) / elite)
                best_hit[L] += 1 if pick[0] == true_el[0] else 0
                ge[f"{L}_elite_same"] = same
                ge[f"{L}_best_same"] = pick[0] == true_el[0]
                ge[f"{L}_elites_feasible"] = sum(1 for i in pick if full[i][1] == "ok")
            ge["true_elites_feasible"] = sum(1 for i in true_el if full[i][1] == "ok")
            pa["gens"].append(ge)
        pa["counterfactual"] = {L: {"gens_elite_set_differs": diff[L], "of_gens": G, "mean_elite_overlap": sum(overlap[L]) / G,
                                    "gens_best_same": best_hit[L]} for L in ladders}
        pa["fd_reference"] = FD.get(ac)
        rep["per_aircraft"][ac] = pa
    # ---- actual runs
    for name, c in res.items():
        if "error" in c:
            rep["cases"][name] = c
            continue
        tcpu = sum(p["task_cpu_s"] for p in c["per_aircraft"].values())
        ref_tcpu = sum(p["task_cpu_s"] for p in ref["per_aircraft"].values())
        e = {"wall_s": c["wall_s"], "batch_wall_s": c["batch_wall_s"], "task_cpu_s_total": tcpu,
             "start": c["start"], "end": c.get("end"), "load_start": c["load_start"][0], "load_end": c["load_end"][0],
             "speedup_vs_full_vizon_wall": ref["batch_wall_s"] / c["batch_wall_s"],
             "speedup_vs_full_vizon_cpu": ref_tcpu / tcpu if tcpu else None, "per_aircraft": {}}
        for ac, p in c["per_aircraft"].items():
            b = base["per_aircraft"][ac]
            ref_el = [tuple(x) for x in b["final_elites"]]
            el = [tuple(x) for x in p["final_elites"]]
            e["per_aircraft"][ac] = {
                "mean_gen_wall_s": sum(p["gen_wall_s"]) / len(p["gen_wall_s"]), "gen_wall_s": p["gen_wall_s"],
                "task_cpu_s": p["task_cpu_s"], "best_final_full": p["best_final"],
                "best_final_minus_full_only": p["best_final"] - b["best_final"], "best_per_gen": p["best_per_gen"],
                "best_feasible_final": p["best_feasible_final"],
                "feasible_rate_full_scored": p["feasible_rate_full_scored"], "n_full_scored": p["n_full_scored"],
                "final_elite_set_same_as_full_only": set(el) == set(ref_el),
                "final_elite_overlap_with_full_only": len(set(el) & set(ref_el)) / max(1, len(ref_el)),
                "spearman_per_gen": p["spearman"], "spearman_both_ok_per_gen": p["spearman_both_ok"],
                "n_rescored": p["n_rescored"], "stage_cpu_s": p.get("stage_cpu_s"), "stage_n": p.get("stage_n"),
                "resim_mismatches": p.get("resim_mismatches")}
        rep["cases"][name] = e
    json.dump(rep, open(os.path.join(OUT, f"report{a.tag}.json"), "w"), indent=1)
    # ---- markdown
    f3 = lambda v: "n/a" if v is None else f"{v:+.3f}"
    L = [f"FD v2 ladder benchmark ({', '.join(acs)}; pop {pop} x {G} gens, seed {cfg['seed']}, cache off, 8 workers). Times PT."
         + ("" if "full-vizon" in res else " Speedups are vs full-only viz OFF (no viz-on case for this seed)."), "",
         "| case | start-end PT | batch wall s | speedup vs full-only viz on (wall / CPU) | total task CPU s | load 1-min start→end | "
         + " | ".join(f"{ac} mean gen wall s | {ac} best full cost (Δ vs full-only) | {ac} feasible rate (full-scored) | {ac} final elites = full-only? (overlap)" for ac in acs) + " |",
         "|" + "---|" * (6 + 4 * len(acs))]
    for name, e in rep["cases"].items():
        if "error" in e:
            continue
        cells = [name, f"{e['start']}-{e['end']}", f"{e['batch_wall_s']:.1f}",
                 f"{e['speedup_vs_full_vizon_wall']:.2f}x / {e['speedup_vs_full_vizon_cpu']:.2f}x", f"{e['task_cpu_s_total']:.0f}",
                 f"{e['load_start']:.1f}→{e['load_end']:.1f}"]
        for ac in acs:
            p = e["per_aircraft"][ac]
            cells += [f"{p['mean_gen_wall_s']:.1f}", f"{p['best_final_full']:.5f} ({p['best_final_minus_full_only']:+.5f})",
                      f"{p['feasible_rate_full_scored']:.2f} (n={p['n_full_scored']})",
                      f"{'yes' if p['final_elite_set_same_as_full_only'] else 'no'} ({p['final_elite_overlap_with_full_only']:.2f})"]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", f"Counterfactual on the full-only populations ({rep['counterfactual_jobs']} extra genome evaluations, "
          f"{cf_wall:.0f} s, load {l0[0]:.1f}→{rep['counterfactual_load_end'][0]:.1f}):", "",
          "| aircraft | gen | n full ok / reduced ok / rigid ok | reduced pass & full fail | Spearman reduced vs full (both ok, n) | rigid vs full (both ok, n) | "
          "elite set same as full-only: rigid→full / reduced→full / rigid→reduced→full |", "|---|---|---|---|---|---|---|"]
    for ac in acs:
        for ge in rep["per_aircraft"][ac]["gens"]:
            L.append(f"| {ac} | {ge['generation']} | {ge['n_full_ok']} / {ge['n_reduced_ok']} / {ge['n_rigid_ok']} | {ge['reduced_ok_full_fail']} | "
                     f"{f3(ge['spearman_reduced_vs_full'])} ({f3(ge['spearman_reduced_vs_full_both_ok'])}, {ge['n_both_ok_reduced']}) | "
                     f"{f3(ge['spearman_rigid_vs_full'])} ({f3(ge['spearman_rigid_vs_full_both_ok'])}, {ge['n_both_ok_rigid']}) | "
                     f"{'/'.join('yes' if ge[f'{x}_elite_same'] else 'no' for x in ladders)} |")
    L += ["", "| aircraft | k_full (+ elites) | ladder | gens elite set differs (of G) | mean elite overlap | gens best same |", "|---|---|---|---|---|---|"]
    for ac in acs:
        pa = rep["per_aircraft"][ac]
        for Ln, v in pa["counterfactual"].items():
            L.append(f"| {ac} | {pa['k_full']} | {Ln} | {v['gens_elite_set_differs']}/{v['of_gens']} | {v['mean_elite_overlap']:.2f} | {v['gens_best_same']}/{v['of_gens']} |")
    L += ["", "Per-generation Spearman logged by the ladder runs (screen vs full on the re-scored set):", ""]
    for name, e in rep["cases"].items():
        if "error" in e or not e["per_aircraft"][acs[0]]["spearman_per_gen"][0]:
            continue
        for ac in acs:
            p = e["per_aircraft"][ac]
            L.append(f"- {name} {ac} (n re-scored {p['n_rescored']}): " + "; ".join(
                ", ".join(f"{k} {f3(v)}" for k, v in (s or {}).items()) for s in p["spearman_per_gen"]))
    open(os.path.join(OUT, f"report{a.tag}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
