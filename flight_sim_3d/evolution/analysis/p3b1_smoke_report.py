"""P3-B1 smoke report (read-only): phase3b1-smoke-s1 (full_a1_b1, phase3_b1 genome) vs phase3a1-smoke-s1 (full_a1).
Reuses p3a1_smoke_report.report() for the shared fields (best cost, structural sum, J_mass, J_wing_tip_bm_limit,
mass delta %, flutter margin, stiffness floor fractions, wing_nsm pile, rejects by status, wall, CPU s/scenario
= eval_cpu_s / sims_computed) and adds, for the B1 run:
  * the 6 decoded shape genes at the best and their encoded u;
  * final-population movement from the baseline shape: mean / std of each decoded shape gene, mean |delta| from the
    baseline value, and fractions within 2 % (encoded) of the floor / ceiling;
  * bound-piling flags: a shape gene with >= 25 % of the final population within 2 % of a bound (possible exploit);
  * geometry_gate rejects (all gens, from rows and from history.jsonl).
Run from the team root:  $PY evolution/analysis/p3b1_smoke_report.py [B1_RUN] [A1_RUN] -> analysis/phase3b1_smoke_s1_check.json"""
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3a1_smoke_report  # noqa: E402

RUNS = os.path.join(os.path.dirname(HERE), "runs")
TOL, PILE = 0.02, 0.25


def shape_report(run_dir):
    rj = json.load(open(os.path.join(run_dir, "run.json")))
    rows = [json.loads(x) for x in open(os.path.join(run_dir, "genomes.jsonl")) if x.strip()]
    hist = [json.loads(x) for x in open(os.path.join(run_dir, "history.jsonl")) if x.strip()]
    out = {}
    for a in rj["aircraft"]:
        name = a["name"]
        idx = [(j, g) for j, g in enumerate(a["genes"]) if g.get("group") == "shape"]
        rs = [r for r in rows if r["aircraft"] == name]
        by = defaultdict(list)
        for r in rs:
            by[r["generation"]].append(r)
        last = max(by)
        final = by[last]
        best = min(final, key=lambda r: r["rank"])
        genes, flags = {}, []
        for j, g in idx:
            vals = [r["genome"][g["name"]] for r in final]
            us = [r["genome_norm"][j] for r in final]
            d0 = g.get("default", 1.0 if "chord" in g["name"] else 0.0)
            lo_f = sum(1 for u in us if u <= TOL) / len(us)
            hi_f = sum(1 for u in us if u >= 1 - TOL) / len(us)
            genes[g["name"]] = {"bounds": [g["min"], g["max"]], "baseline": d0,
                                "best": best["genome"][g["name"]], "best_u": best["genome_norm"][j],
                                "final_mean": statistics.fmean(vals), "final_std": statistics.pstdev(vals),
                                "final_mean_abs_delta_from_baseline": statistics.fmean(abs(v - d0) for v in vals),
                                "final_frac_at_floor": lo_f, "final_frac_at_ceiling": hi_f,
                                "final_frac_exactly_baseline": sum(1 for v in vals if abs(v - d0) < 1e-12) / len(vals),
                                "best_per_gen": [min(by[k], key=lambda r: r["rank"])["genome"][g["name"]] for k in sorted(by)]}
            if lo_f >= PILE or hi_f >= PILE:
                flags.append(f"shape gene piled at a bound (possible exploit): {g['name']} floor {lo_f:.0%} ceiling {hi_f:.0%}")
        hs = [h for h in hist if h["aircraft"] == name]
        out[name] = {"final_generation": last, "best_id": best["individual_id"], "best_shape": best.get("shape"),
                     "shape_genes": genes, "bound_pile_flags": flags,
                     "geometry_gate_rejects_rows": sum(1 for r in rs if str(r["status"]).startswith("geometry_gate")),
                     "geometry_gate_rejects_history": sum(h.get("geometry_gate_rejects", 0) for h in hs),
                     "geometry_gate_reasons_history": dict(sum((Counter(h.get("geometry_gate_reasons") or {}) for h in hs),
                                                               Counter())),
                     "best_model_version": best.get("model_version")}
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    b1 = argv[0] if argv else os.path.join(RUNS, "phase3b1-smoke-s1")
    a1 = argv[1] if len(argv) > 1 else os.path.join(RUNS, "phase3a1-smoke-s1")
    B, A = p3a1_smoke_report.report(b1), p3a1_smoke_report.report(a1)
    S = shape_report(b1)
    res = {"B1_full_a1_b1": B, "A1_full_a1": A, "B1_shape": S,
           "delta_B1_minus_A1": {n: {k: (B["aircraft"][n][k] - A["aircraft"][n][k])
                                     if None not in (B["aircraft"][n][k], A["aircraft"][n][k]) else None
                                     for k in ("best_cost", "structural_sum", "J_mass", "J_wing_tip_bm_limit",
                                               "mass_delta_pct", "flutter_margin", "cpu_s_per_scenario")}
                                 for n in B["aircraft"] if n in A["aircraft"]},
           "wall_ratio_B1_over_A1": B["wall_s"] / A["wall_s"],
           "note": "CPU/wall comparisons are confounded by box load: see loadavg_start of each run."}
    for n, x in B["aircraft"].items():
        y = A["aircraft"].get(n, {})
        print(f"{n:6s} B1 best {x['best_cost']:.6f} struct {x['structural_sum']:.6f} Jm {x['J_mass']:.3g} tip "
              f"{x['J_wing_tip_bm_limit']:.3g} mass {x['mass_delta_pct']:+.3f}% flut {x['flutter_margin']:.4f} rej "
              f"{x['rejected_rows_all_gens']}/{x['rows_all_gens']} {x['rejected_by_status']} cpu/sc "
              f"{x['cpu_s_per_scenario']:.2f} | A1 best {y.get('best_cost', float('nan')):.6f} struct "
              f"{y.get('structural_sum', float('nan')):.6f} flut {y.get('flutter_margin', float('nan')):.4f} cpu/sc "
              f"{y.get('cpu_s_per_scenario', float('nan')):.2f}")
        s = S[n]
        print("        stiff@floor", {k: v for k, v in x["stiffness_floor_fraction"].items() if v},
              "nsm@1.0", x["nsm_floor_fraction"], "gate rejects", s["geometry_gate_rejects_rows"],
              s["geometry_gate_rejects_history"])
        for g, d in s["shape_genes"].items():
            print(f"        {g:24s} best {d['best']:+.4f} mean {d['final_mean']:+.4f} sd {d['final_std']:.4f} "
                  f"|d| {d['final_mean_abs_delta_from_baseline']:.4f} lo {d['final_frac_at_floor']:.2f} "
                  f"hi {d['final_frac_at_ceiling']:.2f} =base {d['final_frac_exactly_baseline']:.2f}")
        for f in s["bound_pile_flags"]:
            print("        FLAG", f)
    with open(os.path.join(HERE, "phase3b1_smoke_s1_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wall B1", B["wall_s"], "A1", A["wall_s"], "load B1", B["loadavg_start"], "A1", A["loadavg_start"])


if __name__ == "__main__":
    main()
