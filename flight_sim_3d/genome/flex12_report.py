#!/usr/bin/env python3
"""phase1_flex (12 genes, heading hold) small reruns: per-gen failures, best structure, margins, heading drift,
and a constraint probe (margin < 1 fails, 1..1.2 penalised) around seed 1's best. Writes runs/flex12_report.json."""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import adapter

t = adapter.load_task("phase1_flex")
out = {"runs": [], "probe": []}
for s in (1, 2, 3):
    d = os.path.join(HERE, "runs", f"flex12_c172x_s{s}")
    if not os.path.exists(os.path.join(d, "best_gains.json")):
        continue
    bg = json.load(open(os.path.join(d, "best_gains.json")))
    hist = list(csv.DictReader(open(os.path.join(d, "fitness_history.csv"))))
    r = t.evaluate(bg["gains"], t.make_scenarios(3, bg["config"]["scenario_seed"]), record=True)
    assert r["cost"] == bg["best_cost"]
    ae = r.get("aeroelastic", {})
    row = {"seed": s, "best": r["cost"], "objectives": r["objectives"], "failed_per_gen": [h.get("n_failed") or h.get("failed") for h in hist],
           "structure": {k: bg["gains"][k] for k in ("stiffness_scale", "torsion_bend_ratio", "struct_damping_ratio", "nonstructural_mass_scale")},
           "flutter_margin": ae.get("flutter_margin"), "div_margin": ae.get("div_margin"), "margin_penalty": ae.get("margin_penalty"),
           "delta_wing_mass_lb": ae.get("delta_wing_mass_lb"),
           "hdg_final_deg": [x.get("hdg_final_deg") for x in r["diagnostics"]],
           "max_root_bending": [x.get("max_root_bending") for x in r["diagnostics"]],
           "max_tip_twist_deg": [x.get("max_tip_twist_deg") for x in r["diagnostics"]],
           "statuses": [p["status"] for p in r["per_scenario"]], "at_bounds": json.load(open(os.path.join(d, "at_bounds.json")))["best"]}
    out["runs"].append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "objectives"}, default=str))
    print("   objectives", {k: round(v, 4) for k, v in r["objectives"].items()})
b = out["runs"][0]
g0 = json.load(open(os.path.join(HERE, "runs", "flex12_c172x_s1", "best_gains.json")))["gains"]
for s_, r_ in ((0.6, 0.8), (0.7, 0.9), (0.8, 1.0), (b["structure"]["stiffness_scale"], b["structure"]["torsion_bend_ratio"]), (1.2, 1.1), (2.0, 1.15)):
    g = {**g0, "stiffness_scale": s_, "torsion_bend_ratio": r_}
    r = t.evaluate(g, t.make_scenarios(3, 1))
    ae = r.get("aeroelastic", {})
    p = {"s": s_, "r": r_, "cost": r["cost"], "flutter": ae.get("flutter_margin"), "div": ae.get("div_margin"),
         "penalty": ae.get("margin_penalty"), "fail": ae.get("fail"), "status": [x["status"] for x in r["per_scenario"]][0]}
    out["probe"].append(p); print("probe", p)
json.dump(out, open(os.path.join(HERE, "runs", "flex12_report.json"), "w"), indent=2, default=str)
