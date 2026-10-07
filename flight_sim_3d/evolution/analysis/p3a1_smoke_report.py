"""P3-A1 smoke report (read-only): phase3a1-smoke-s1 (full_a1) vs phase2-smoke-p25-s1 (full), per aircraft.
Fields: best cost, total structural cost at the best (sum of J_* = phase2_check structural_sum), J_mass,
J_wing_tip_bm_limit, mass delta %, flutter margin, stiffness genes at the floor (final pop fraction within 2 %),
wing_nsm_* pile, rejected (status != ok) rows, wall time, measured CPU s per scenario (eval_cpu_s / sims_computed).
NOTE: J_wing_tip_bm_limit is station-exact in full_a1 and strip-discrete in full (not the same definition).
phase2_check exits 1 whenever any J_* < 0 (J_mass < 0 is legitimate); judge by the total structural cost.
Run from the team root:  $PY evolution/analysis/p3a1_smoke_report.py [A_RUN] [B_RUN]  -> analysis/phase3a1_smoke_s1_check.json"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase2_check  # noqa: E402

RUNS = os.path.join(os.path.dirname(HERE), "runs")
STIFF = phase2_check.STIFFNESS_GENES


def report(run_dir):
    chk = phase2_check.check_run(run_dir)
    rows, _ = phase2_check.load(run_dir)
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    hist = [json.loads(x) for x in open(os.path.join(run_dir, "history.jsonl")) if x.strip()]
    out = {"run_id": chk["run_id"], "wall_s": summ["wall_s_this_session"], "workers": summ["workers"],
           "loadavg_start": summ.get("loadavg_1_5_15_start"), "aircraft": {}}
    for name, x in chk["aircraft"].items():
        rs = [r for r in rows if r["aircraft"] == name]
        hs = [h for h in hist if h["aircraft"] == name]
        sa = next(a for a in summ["aircraft"] if a["aircraft"] == name)
        st = x["structural_terms"]
        cpu = sum(h["eval_cpu_s"] for h in hs)
        sims = sum(h["sims_computed"] for h in hs)
        out["aircraft"][name] = {
            "best_id": x["best_id"], "best_cost": x["best_cost"], "best_status": x["best_status"],
            "model_version": x["best_model_version"],
            "structural_sum": x["structural_sum"], "J_mass": st.get("J_mass"),
            "J_wing_tip_bm_limit": st.get("J_wing_tip_bm_limit"),
            "negative_terms": x["negative_terms"], "nonzero_struct_terms": {k: v for k, v in st.items() if v != 0.0},
            "mass_delta_pct": 100 * x["mass_total_frac"] if x["mass_total_frac"] is not None else None,
            "mass_lb": x["mass_lb"], "margins": x["margins"], "flutter_margin": (x["margins"] or {}).get("flutter_margin"),
            "stiffness_floor_fraction": {g: x["floor_fraction"].get(g) for g in STIFF},
            "stiffness_genes_piled_ge_25pct": [g for g in STIFF if (x["floor_fraction"].get(g) or 0) >= 0.25],
            "stiffness_any_in_final_pop_at_floor": {g: x["floor_fraction"][g] for g in STIFF if x["floor_fraction"].get(g)},
            "best_genes_at_bounds": x["best_genes_within_tol_of_bounds"],
            "nsm_floor_fraction": {g: x["floor_fraction"].get(g) for g in ("wing_nsm_root", "wing_nsm_tip")},
            "best_struct": x["best_struct"],
            "rejected_rows_all_gens": sum(1 for r in rs if r["status"] != "ok"), "rows_all_gens": len(rs),
            "rejected_by_status": dict(Counter(r["status"] for r in rs if r["status"] != "ok")),
            "rejected_last_gen": sum(1 for r in rs if r["status"] != "ok" and r["generation"] == x["generation"]),
            "best_per_gen": [h["best"] for h in hs],
            "evolve_wall_s": sa.get("evolve_wall_s_this_session"), "eval_wall_s": sum(h["eval_wall_s"] for h in hs),
            "eval_cpu_s": cpu, "sims_computed": sims, "cpu_s_per_scenario": cpu / sims if sims else None,
            "resim_mismatches": sa.get("resim_mismatches"), "trajectories": [t["file"] for t in sa.get("trajectories", [])],
            "flags_phase2_check": x["flags"]}
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    a = argv[0] if argv else os.path.join(RUNS, "phase3a1-smoke-s1")
    b = argv[1] if len(argv) > 1 else os.path.join(RUNS, "phase2-smoke-p25-s1")
    A, B = report(a), report(b)
    res = {"A_full_a1": A, "B_full_p25": B,
           "note": "J_wing_tip_bm_limit: A = station-exact at eta 0.875 (sizing_a1), B = strip-discrete (sizing_v2); "
                   "not the same definition. Judge structure by structural_sum (>= 0), not by phase2_check's J_mass<0 flag.",
           "delta_A_minus_B": {n: {k: A["aircraft"][n][k] - B["aircraft"][n][k]
                                   for k in ("best_cost", "structural_sum", "cpu_s_per_scenario")}
                               for n in A["aircraft"] if n in B["aircraft"]},
           "wall_ratio_A_over_B": A["wall_s"] / B["wall_s"]}
    for n, x in A["aircraft"].items():
        y = B["aircraft"].get(n, {})
        print(f"{n:6s} A1 best {x['best_cost']:.6f} struct {x['structural_sum']:.6f} Jm {x['J_mass']:.3g} tip "
              f"{x['J_wing_tip_bm_limit']:.3g} mass {x['mass_delta_pct']:+.3f}% flut {x['flutter_margin']:.4f} rej "
              f"{x['rejected_rows_all_gens']}/{x['rows_all_gens']} cpu/sc {x['cpu_s_per_scenario']:.2f} | P2.5 best "
              f"{y.get('best_cost', float('nan')):.6f} struct {y.get('structural_sum', float('nan')):.6f} cpu/sc "
              f"{y.get('cpu_s_per_scenario', float('nan')):.2f}")
    with open(os.path.join(HERE, "phase3a1_smoke_s1_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wall A", A["wall_s"], "B", B["wall_s"])


if __name__ == "__main__":
    main()
