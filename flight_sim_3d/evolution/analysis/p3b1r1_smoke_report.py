"""P3-B1 r1 smoke report (read-only): phase3b1r1-smoke-s1 (B1 r1) vs phase3b1-smoke-s1 (B1 r0, superseded) vs
phase3a1-smoke-s1 (A1), per aircraft. Same fields as p3b1_smoke_report.py (shared ones from p3a1_smoke_report.report;
shape-gene movement / bound piling / geometry_gate rejects from p3b1_smoke_report.shape_report), plus the box load.
Run from the team root:  $PY -B evolution/analysis/p3b1r1_smoke_report.py [R1_RUN] [R0_RUN] [A1_RUN]
-> analysis/phase3b1r1_smoke_s1_check.json"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3a1_smoke_report  # noqa: E402
from p3b1_smoke_report import shape_report  # noqa: E402

RUNS = os.path.join(os.path.dirname(HERE), "runs")
F = ("best_cost", "structural_sum", "J_mass", "J_wing_tip_bm_limit", "mass_delta_pct", "flutter_margin", "cpu_s_per_scenario")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    paths = [argv[i] if len(argv) > i else os.path.join(RUNS, d)
             for i, d in enumerate(("phase3b1r1-smoke-s1", "phase3b1-smoke-s1", "phase3a1-smoke-s1"))]
    R1, R0, A1 = (p3a1_smoke_report.report(p) for p in paths)
    S1, S0 = shape_report(paths[0]), shape_report(paths[1])
    loads = {}
    for nm, p in zip(("r1", "r0", "a1"), paths):
        s = json.load(open(os.path.join(p, "summary.json")))
        loads[nm] = {"loadavg_start": s.get("loadavg_1_5_15_start"), "loadavg_end": s.get("loadavg_1_5_15_end"),
                     "wall_s": s.get("wall_s_this_session"), "workers": s.get("workers")}
    res = {"r1": R1, "r0": R0, "a1": A1, "r1_shape": S1, "r0_shape": S0, "box": loads,
           "delta_r1_minus_r0": {n: {k: (R1["aircraft"][n][k] - R0["aircraft"][n][k])
                                     if None not in (R1["aircraft"][n][k], R0["aircraft"][n][k]) else None for k in F}
                                 for n in R1["aircraft"]},
           "delta_r1_minus_a1": {n: {k: (R1["aircraft"][n][k] - A1["aircraft"][n][k])
                                     if None not in (R1["aircraft"][n][k], A1["aircraft"][n][k]) else None for k in F}
                                 for n in R1["aircraft"]}}
    for n, x in R1["aircraft"].items():
        y, z = R0["aircraft"][n], A1["aircraft"][n]
        print(f"{n:6s} best r1 {x['best_cost']:.6f} r0 {y['best_cost']:.6f} a1 {z['best_cost']:.6f} | struct r1 "
              f"{x['structural_sum']:.6f} r0 {y['structural_sum']:.6f} a1 {z['structural_sum']:.6f} | Jm {x['J_mass']:.4g} "
              f"tip {x['J_wing_tip_bm_limit']:.3g} mass {x['mass_delta_pct']:+.3f}% flut {x['flutter_margin']:.4f} | rej "
              f"{x['rejected_by_status']} | cpu/sc r1 {x['cpu_s_per_scenario']:.2f} r0 {y['cpu_s_per_scenario']:.2f} a1 "
              f"{z['cpu_s_per_scenario']:.2f}")
        s = S1[n]
        print("        stiff@floor", {k: v for k, v in x["stiffness_floor_fraction"].items() if v}, "nsm@1.0",
              x["nsm_floor_fraction"], "gate rejects", s["geometry_gate_rejects_rows"], s["geometry_gate_rejects_history"],
              "mv", s["best_model_version"])
        for g, d in s["shape_genes"].items():
            print(f"        {g:24s} best {d['best']:+.4f} mean {d['final_mean']:+.4f} sd {d['final_std']:.4f} "
                  f"lo {d['final_frac_at_floor']:.2f} hi {d['final_frac_at_ceiling']:.2f} | r0 best "
                  f"{S0[n]['shape_genes'][g]['best']:+.4f} hi {S0[n]['shape_genes'][g]['final_frac_at_ceiling']:.2f}")
        for f in s["bound_pile_flags"]:
            print("        FLAG", f)
    with open(os.path.join(HERE, "phase3b1r1_smoke_s1_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("box", loads)


if __name__ == "__main__":
    main()
