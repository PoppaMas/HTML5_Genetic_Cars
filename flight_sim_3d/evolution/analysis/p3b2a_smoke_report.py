"""phase3b2a-smoke-s1 report (read-only on the run dir): per aircraft rank-0 best of the last generation, energy share,
B2a gene use (dihedral / camber) and bound piling, rejects (status != ok, geometry gate), wall time.
  $PY -B evolution/analysis/p3b2a_smoke_report.py [RUN_DIR] -> analysis/phase3b2a_smoke_s1_check.json"""
import json, os, sys
from collections import Counter
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)
from evolution import fidelity as F  # noqa: E402
run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TEAM, "evolution/runs/phase3b2a-smoke-s1")
rows = [json.loads(x) for x in open(os.path.join(run, "genomes.jsonl")) if x.strip()]
summ = json.load(open(os.path.join(run, "summary.json")))
B2G = ("wing_dihedral_delta_deg", "wing_camber_root_delta_pct", "wing_camber_tip_delta_pct")
out = {"run_dir": os.path.relpath(run, TEAM), "aircraft": {}}
for k in ("wall_s_this_session",):
    if k in summ:
        out["wall_s"] = summ[k]
for ac in sorted({r["aircraft"] for r in rows}):
    R = [r for r in rows if r["aircraft"] == ac]
    last = max(r["generation"] for r in R)
    L = [r for r in R if r["generation"] == last]
    b = min(L, key=lambda r: r["rank"])
    sch = {g.name: g for g in F.shape_schema_b2(ac)}
    et = b.get("energy_terms") or {}
    gene = {}
    for nm in B2G:
        g = sch[nm]
        vals = [r["shape"][nm] for r in L if r.get("shape")]
        span = g.max - g.min
        gene[nm] = {"range": [g.min, g.max], "default": g.default, "best": b["shape"][nm],
                    "last_gen_mean": sum(vals) / len(vals), "last_gen_min": min(vals), "last_gen_max": max(vals),
                    "frac_moved_from_default": sum(abs(v - g.default) > 1e-9 for v in vals) / len(vals),
                    "frac_at_lo": sum(v - g.min < 0.02 * span for v in vals) / len(vals),
                    "frac_at_hi": sum(g.max - v < 0.02 * span for v in vals) / len(vals)}
    pile = {}
    for nm, g in sch.items():
        if nm in ("wing_tc_root_scale", "wing_tc_tip_ratio"):
            continue
        vals = [r["shape"][nm] for r in L if r.get("shape")]
        span = g.max - g.min
        lo = sum(v - g.min < 0.02 * span for v in vals) / len(vals); hi = sum(g.max - v < 0.02 * span for v in vals) / len(vals)
        if lo >= 0.5 or hi >= 0.5:
            pile[nm] = {"at_lo": lo, "at_hi": hi}
    ok = [r for r in R if r["status"] == "ok" and r.get("energy_terms")]
    share = [(r["energy_terms"]["J_energy"] + r["energy_terms"]["J_speed_guard"]) / r["cost"] for r in ok if r["cost"]]
    out["aircraft"][ac] = {
        "best": {"individual_id": b["individual_id"], "cost": b["cost"], "rank": b["rank"], "status": b["status"],
                 "model_version": b["model_version"], "cost_fd": et.get("cost_fd"), "J_energy": et.get("J_energy"),
                 "J_speed_guard": et.get("J_speed_guard"), "energy_share": (et.get("J_energy", 0) + et.get("J_speed_guard", 0)) / b["cost"],
                 "energy_drag_increment": et.get("energy_drag_increment"), "shape": b["shape"]},
        "best_per_gen": [min((r for r in R if r["generation"] == g), key=lambda r: r["rank"])["cost"] for g in range(last + 1)],
        "energy_share_all_ok_rows": {"n": len(share), "mean": sum(share) / len(share), "max": max(share)},
        "frac_rows_J_speed_guard_pos": sum(r["energy_terms"]["J_speed_guard"] > 0 for r in ok) / len(ok),
        "b2a_genes_last_gen": gene, "bound_piling_ge50pct_last_gen": pile,
        "status_counts": dict(Counter(r["status"] for r in R)),
        "geometry_gate_rejects": sum(str(r["status"]).startswith("geometry_gate") for r in R),
        "tc_locked_all_rows": all(r["shape"].get("wing_tc_root_scale", 1.0) == 1.0 and r["shape"].get("wing_tc_tip_ratio", 1.0) == 1.0
                                  for r in R if r.get("shape"))}
json.dump(out, open(os.path.join(HERE, "phase3b2a_smoke_s1_check.json"), "w"), indent=1)
print(json.dumps({a: {"best": v["best"]["cost"], "share": round(v["best"]["energy_share"], 4), "shareall": v["energy_share_all_ok_rows"],
                      "genes": {k: (round(g["best"], 3), round(g["frac_moved_from_default"], 2), g["frac_at_lo"], g["frac_at_hi"]) for k, g in v["b2a_genes_last_gen"].items()},
                      "pile": v["bound_piling_ge50pct_last_gen"], "status": v["status_counts"], "gate": v["geometry_gate_rejects"],
                      "tc": v["tc_locked_all_rows"], "bpg": v["best_per_gen"]} for a, v in out["aircraft"].items()}, indent=1))
print("wall", out.get("wall_s"), list(summ)[:20])
