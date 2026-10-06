#!/usr/bin/env python3
"""Phase-1 v4 vs v5 comparison: flight metrics, gene spread across seeds, I-gains, plateau generation.

    $PY v5_report.py [--out runs/v5_report.json]

v4 = presets/phase1_v4.json (frozen; the runs are the heading-hold "after" runs), v5 = presets/phase1_v5.json.
Both best genomes of each seed are re-flown under BOTH tasks (v5 adds the downdraft scenario and hold term), so the
hold oscillation and the downdraft residual are measured the same way for v4 and v5 genomes. Costs are reported
under each genome's own task (not comparable across presets). evolve.py's scenario seed = GA seed for c172x.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapter  # noqa: E402

RUNS = {  # (preset, aircraft key, seed) -> run dir. v5 c172x s1 = the sweep run at the chosen weight 0.1 (identical task, verified)
    ("v4", "c172x", 1): "hdg_after_c172x_s1", ("v4", "c172x", 2): "hdg_after_c172x_s2", ("v4", "c172x", 3): "hdg_after_c172x_s3",
    ("v4", "t38", 1): "hdg_after_t38_s1", ("v4", "b737", 1): "hdg_after_b737_s1",
    ("v5", "c172x", 1): "v5_sweep_w01", ("v5", "c172x", 2): "v5_c172x_s2", ("v5", "c172x", 3): "v5_c172x_s3",
    ("v5", "t38", 1): "v5_t38_s1", ("v5", "b737", 1): "v5_b737_s1",
}
PRESET = {"v4": "phase1_v4", "v5": "phase1_v5"}
GENES = ["kp_alt", "ki_alt", "kd_alt", "kp_pitch", "ki_pitch", "kd_pitch", "kp_hdg", "ki_hdg"]


def plateau(run_dir, tol=0.01):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "fitness_history.csv"))))
    b = np.array([float(r["best_cost"]) for r in rows])
    final = b[-1]
    within = int(np.argmax(b <= final * (1 + tol)))
    imp = [g for g in range(1, len(b)) if b[g] < b[g - 1] * (1 - 1e-3)]
    return {"gen_within_1pct": within, "last_improvement_gen": imp[-1] if imp else 0, "n_gens": len(b),
            "gen0_best": float(b[0]), "final_best": float(final)}


def metrics(task, gains, ss):
    r = task.evaluate(gains, task.make_scenarios(3, ss), record=True)
    d = r["diagnostics"]
    base = d[:3]
    out = {"cost": r["cost"], "objectives": r["objectives"],
           "max_pitch_calm": base[0]["max_pitch_deg"], "max_pitch_worst": max(x["max_pitch_deg"] for x in base),
           "nz_calm": [base[0]["min_nz"], base[0]["max_nz"]],
           "nz_all": [min(x["min_nz"] for x in d), max(x["max_nz"] for x in d)],
           "hdg_final_max_abs": max(abs(x["hdg_final_deg"]) for x in d),
           "hold_pp_calm": base[0].get("hold_pp_ft"), "hold_pp_worst": max(x.get("hold_pp_ft", 0) for x in base),
           "statuses": [p["status"] for p in r["per_scenario"]]}
    if len(d) > 3:
        out["draft_residual_ft"] = d[3]["draft_residual_ft"]
        out["draft_max_err_ft"] = d[3]["draft_max_err_ft"]
    return out


def spread(gene_sets):
    rep = {}
    for g in GENES:
        v = np.array([gs[g] for gs in gene_sets if g in gs])
        nz = v[v > 0]
        rep[g] = {"values": v.tolist(), "n_zero": int((v == 0).sum()),
                  "max_over_min": float(nz.max() / nz.min()) if nz.size >= 2 else None,
                  "log10_std": float(np.std(np.log10(nz), ddof=1)) if nz.size >= 2 else None}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "v5_report.json"))
    a = ap.parse_args(argv)
    tasks = {(p, ac): adapter.load_task(PRESET[p], {"aircraft": ac}) for p in PRESET for ac in ("c172x", "t38", "b737")}
    rows = []
    for (p, ac, seed), run in RUNS.items():
        d = os.path.join(HERE, "runs", run)
        if not os.path.exists(os.path.join(d, "best_gains.json")):
            continue
        bg = json.load(open(os.path.join(d, "best_gains.json")))
        ss = bg["config"]["scenario_seed"]
        own = metrics(tasks[(p, ac)], bg["gains"], ss)
        assert own["cost"] == bg["best_cost"], (run, own["cost"], bg["best_cost"])
        v5m = own if p == "v5" else metrics(tasks[("v5", ac)], bg["gains"], ss)  # v4 genome measured on the v5 scenarios
        ab = json.load(open(os.path.join(d, "at_bounds.json")))["best"]
        rows.append({"preset": p, "aircraft": ac, "seed": seed, "run": run, "gains": bg["gains"], "own": own,
                     "hold_pp_calm": v5m["hold_pp_calm"], "hold_pp_worst": v5m["hold_pp_worst"],
                     "draft_residual_ft": v5m["draft_residual_ft"], "draft_max_err_ft": v5m["draft_max_err_ft"],
                     "at_bounds": [f"{e['gene']}@{e['bound']}({e['value']:.3g})" for e in ab], "plateau": plateau(d)})
    sp = {p: spread([r["gains"] for r in rows if r["preset"] == p and r["aircraft"] == "c172x"]) for p in PRESET}
    json.dump({"rows": rows, "c172x_gene_spread": sp}, open(a.out, "w"), indent=2, default=float)
    print("| preset | aircraft s | best cost (own task) | track_alt | max pitch calm/worst | nz calm / all | max abs hdg drift | hold p-p calm/worst ft | downdraft residual / max err ft | ki_alt | ki_pitch | at bounds | plateau gen (1 % / last impr.) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        o = r["own"]
        print(f"| {r['preset']} | {r['aircraft']} s{r['seed']} | {o['cost']:.4f} | {o['objectives']['track_alt']:.4f} | "
              f"{o['max_pitch_calm']:.1f} / {o['max_pitch_worst']:.1f} | {o['nz_calm'][0]:.2f}-{o['nz_calm'][1]:.2f} / {o['nz_all'][0]:.2f}-{o['nz_all'][1]:.2f} | "
              f"{o['hdg_final_max_abs']:.1f} | {r['hold_pp_calm']:.1f} / {r['hold_pp_worst']:.1f} | {r['draft_residual_ft']:+.2f} / {r['draft_max_err_ft']:.1f} | "
              f"{r['gains']['ki_alt']:.3g} | {r['gains']['ki_pitch']:.3g} | {', '.join(r['at_bounds']) or 'none'} | "
              f"{r['plateau']['gen_within_1pct']} / {r['plateau']['last_improvement_gen']} |")
    print("\nc172x gene spread across seeds (max/min of nonzero values; log10 std; zeros)")
    print("| gene | v4 max/min | v4 log10-std | v4 zeros | v5 max/min | v5 log10-std | v5 zeros |")
    print("|---|---|---|---|---|---|---|")
    f = lambda x, fmt: (fmt % x) if x is not None else "-"
    for g in GENES:
        a4, a5 = sp["v4"].get(g, {}), sp["v5"].get(g, {})
        print(f"| {g} | {f(a4.get('max_over_min'), '%.1f')} | {f(a4.get('log10_std'), '%.2f')} | {a4.get('n_zero', '-')} | "
              f"{f(a5.get('max_over_min'), '%.1f')} | {f(a5.get('log10_std'), '%.2f')} | {a5.get('n_zero', '-')} |")


if __name__ == "__main__":
    main()
