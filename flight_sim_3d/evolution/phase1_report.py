#!/usr/bin/env python3
"""Phase-1 seed study report: reads runs/<prefix>-s{1,2,3}/summary.json (+ bench_jets-j1 for context) and prints
markdown tables. Also re-flies bench_jets-j1's best genomes under the Phase-1 task (the only like-for-like comparison).

    python -m evolution.phase1_report [--prefix phase1] [--seeds 1 2 3] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import genome, sim

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
ACS = ["c172x", "T38", "737", "f16"]


def load(run_id):
    p = os.path.join(R, run_id, "summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        s = json.load(f)
    s["by_ac"] = {a["aircraft"]: a for a in s["aircraft"]}
    return s


def hold_mean(a):
    return float(np.mean([m["hold_rms_err_ft"] for m in a["final_metrics"]]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="phase1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    runs = {s: load(f"{a.prefix}-s{s}") for s in a.seeds}
    missing = [s for s, r in runs.items() if r is None]
    if missing:
        raise SystemExit(f"missing runs for seeds {missing}")
    out = {"runs": {s: r["run_id"] for s, r in runs.items()}, "aircraft": {}}

    print("### Per aircraft, per seed (GA seed varies; scenario set fixed, scenario_seed 1)\n")
    print("| aircraft | seed | best cost | gen-0 best | track / effort / comfort (mean of 3 scen.) | hold RMS mean (calm) ft | overshoot max ft | invalid rate all gens / final | genes at bound | evolve wall s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for ac in ACS:
        rows = []
        for s in a.seeds:
            x = runs[s]["by_ac"][ac]
            if "best_fitness" not in x:
                print(f"| {ac} | {s} | SKIPPED: {x.get('skipped') or x.get('error')} | | | | | | | |")
                continue
            ps = x["final_per_scenario"]
            tec = [np.mean([p[k] for p in ps]) for k in ("track", "effort", "comfort")]
            hm = hold_mean(x)
            ov = max(m["overshoot_ft_max"] for m in x["final_metrics"])
            gb = ", ".join(f"{k}@{v}" for k, v in x["genes_at_bound"].items()) or "-"
            print(f"| {ac} | {s} | {x['best_fitness']:.4f} | {x['gen0_best_fitness']:.4f} | {tec[0]:.4f} / {tec[1]:.4f} / {tec[2]:.3f} | "
                  f"{hm:.2f} ({x['final_metrics'][0]['hold_rms_err_ft']:.2f}) | {ov:.1f} | "
                  f"{x['invalid_rate_all_gens']:.3f} / {x['invalid_rate_final']:.3f} | {gb} | {x['evolve_wall_s_this_session']:.1f} |")
            rows.append({"seed": s, "best": x["best_fitness"], "gen0": x["gen0_best_fitness"], "hold_mean": hm,
                         "hold_calm": x["final_metrics"][0]["hold_rms_err_ft"], "overshoot": ov,
                         "track": tec[0], "effort": tec[1], "comfort": tec[2],
                         "invalid_all": x["invalid_rate_all_gens"], "invalid_final": x["invalid_rate_final"],
                         "genes_at_bound": x["genes_at_bound"], "best_gains": x["best_gains"],
                         "evolve_wall_s": x["evolve_wall_s_this_session"], "status_counts": x["status_counts_all_gens"]})
        out["aircraft"][ac] = rows

    print("\n### Across seeds\n")
    print("| aircraft | best cost per seed | mean | std (ddof=1) | min-max | hold RMS mean across seeds ft | invalid rate (all gens, mean over seeds) | bench_jets-j1 best (different task) |")
    print("|---|---|---|---|---|---|---|---|")
    j1 = load("bench_jets-j1")
    for ac in ACS:
        rows = out["aircraft"][ac]
        if not rows:
            continue
        b = np.array([r["best"] for r in rows])
        j = j1["by_ac"][ac]["best_fitness"] if j1 and ac in j1["by_ac"] else float("nan")
        agg = {"mean": float(b.mean()), "std": float(b.std(ddof=1)) if len(b) > 1 else 0.0, "min": float(b.min()),
               "max": float(b.max()), "hold_mean": float(np.mean([r["hold_mean"] for r in rows])),
               "invalid_all_mean": float(np.mean([r["invalid_all"] for r in rows])), "j1_best": j}
        out.setdefault("agg", {})[ac] = agg
        print(f"| {ac} | {' / '.join(f'{v:.4f}' for v in b)} | {agg['mean']:.4f} | {agg['std']:.4f} | {agg['min']:.4f}-{agg['max']:.4f} | "
              f"{agg['hold_mean']:.2f} | {agg['invalid_all_mean']:.3f} | {j:.4f} |")

    print("\n### Wall time and box load\n")
    print("| run | seed | wall s | loadavg 1/5/15 start | end | sims computed | cache hits | sims/s |")
    print("|---|---|---|---|---|---|---|---|")
    tot = 0.0
    for s in a.seeds:
        r = runs[s]
        tot += r["wall_s_this_session"]
        ts = r["this_session"]
        print(f"| {r['run_id']} | {s} | {r['wall_s_this_session']:.1f} | {'/'.join(f'{v:.1f}' for v in r['loadavg_1_5_15_start'])} | "
              f"{'/'.join(f'{v:.1f}' for v in r['loadavg_1_5_15_end'])} | {ts['sims_computed']} | {ts['cache_hits']} | {ts['sims_per_s']:.1f} |")
    print(f"| total | | {tot:.1f} | | | | | |")
    out["wall"] = {s: runs[s]["wall_s_this_session"] for s in a.seeds}
    out["wall_total_s"] = tot
    out["load"] = {s: [runs[s]["loadavg_1_5_15_start"], runs[s]["loadavg_1_5_15_end"]] for s in a.seeds}

    if j1:
        print("\n### bench_jets-j1 best genomes re-flown under the Phase-1 task (3 scenarios, scenario_seed 1)\n")
        print("| aircraft | j1 best cost (its own task) | j1 genome under Phase-1 | Phase-1 GA best, seed 1 / mean of seeds |")
        print("|---|---|---|---|")
        cfg_path = os.path.join(R, runs[a.seeds[0]]["run_id"], "config.json")
        with open(cfg_path) as f:
            p1 = {x["name"]: x["resolved_profile"] for x in json.load(f)["resolved"]["aircraft"]}
        out["j1_refly"] = {}
        for ac in ACS:
            if ac not in j1["by_ac"] or "best_gains" not in j1["by_ac"][ac]:
                continue
            P = sim.Profile.from_dict(p1[ac])
            costs = [sim.simulate(j1["by_ac"][ac]["best_gains"], sc, P) for sc in sim.make_scenarios(3, 1, P)]
            c = float(np.mean([r["cost"] for r in costs]))
            st = ",".join(sorted({r["status"] for r in costs}))
            out["j1_refly"][ac] = {"cost": c, "status": st}
            b1 = out["aircraft"][ac][0]["best"] if out["aircraft"][ac] else float("nan")
            print(f"| {ac} | {j1['by_ac'][ac]['best_fitness']:.4f} | {c:.4f} ({st}) | {b1:.4f} / {out['agg'][ac]['mean']:.4f} |")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
