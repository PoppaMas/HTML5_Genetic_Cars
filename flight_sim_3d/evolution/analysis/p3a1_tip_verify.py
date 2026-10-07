"""P3-A1 cross-check: re-fly Genome's c172x tip-verify pair (genome/runs/p25_tip_verify_c172x.json: baseline and
wing_ei_taper_4 = 0.75 'tip-soft', same gains) at full_a1 through evolution.fidelity, and re-confirm that full still
reproduces the P2.5 numbers bit for bit (evolution/analysis/p25_tip_verify_evolution.json).
Method = the P2.5 re-fly: phase2_smoke_p25.json c172x profile, 3 scenarios, scenario_seed 1, evolution.fidelity.evaluate_genome.
Run from the team root:  $PY evolution/analysis/p3a1_tip_verify.py   -> evolution/analysis/p3a1_tip_verify_evolution.json"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)

from evolution import batch, fidelity as F, sim  # noqa: E402

RIGID = ("track", "effort", "comfort", "heading", "hold")


def main():
    cfg = batch.resolve_config(json.load(open(os.path.join(TEAM, "evolution", "configs", "phase2_smoke_p25.json"))), "tv")
    pd = next(a for a in cfg["aircraft"] if a["name"] == "c172x")["resolved_profile"]
    prof = sim.Profile.from_dict(pd)
    scs = [s.to_dict() for s in sim.make_scenarios(cfg["scenarios"], cfg["scenario_seed"], prof)]
    gref = json.load(open(os.path.join(TEAM, "genome", "runs", "p25_tip_verify_c172x.json")))
    eref = json.load(open(os.path.join(HERE, "p25_tip_verify_evolution.json")))["cases"]
    pins = json.load(open(os.path.join(F.FD_DIR, "v2_results", "model_versions_post_p3a1.json")))["c172x"]
    out = {"method": "evolution.fidelity.evaluate_genome(phase2_smoke_p25 c172x profile, 3 scenarios, scenario_seed 1); "
                     "genomes from genome/runs/p25_tip_verify_c172x.json",
           "model_version_expected": {"full": pins["full"], "full_a1": pins["full_a1"]}, "cases": {}}
    for case, g in gref.items():
        res = {}
        for fid in ("full", "full_a1"):
            c0, w0 = time.process_time(), time.perf_counter()
            r = F.evaluate_genome(pd, g["gains"], g["struct"], scs, fid, 0.9)
            cpu, wall = time.process_time() - c0, time.perf_counter() - w0
            struct_sum = float(sum(v for k, v in r["terms"].items() if k not in RIGID))
            e = {"model_version": r["model_version"], "model_version_ok": r["model_version"] == pins[fid],
                 "status": r["status"], "cost": r["cost"], "per_scenario_cost": [p["cost"] for p in r["per_scenario"]],
                 "J_wing_tip_bm_limit": r["terms"]["J_wing_tip_bm_limit"], "J_mass": r["terms"]["J_mass"],
                 "J_smooth": r["terms"]["J_smooth"], "fd_struct_sum": struct_sum,
                 "margins": r["margins"], "mass_total_frac": r["mass_total_frac"],
                 "cpu_s": cpu, "wall_s": wall, "cpu_s_per_scenario": cpu / len(scs),
                 "terms": r["terms"]}
            if fid == "full":
                ref = eref[case]
                e.update(ref_total=ref["cost"], ref_per=ref["per_scenario_cost"], ref_tip=ref["J_wing_tip_bm_limit"],
                         match_total=e["cost"] == ref["cost"], match_per=e["per_scenario_cost"] == ref["per_scenario_cost"],
                         match_tip=e["J_wing_tip_bm_limit"] == ref["J_wing_tip_bm_limit"],
                         genome_total_match=e["cost"] == g["cost"])
                e["bit_identical"] = e["match_total"] and e["match_per"] and e["match_tip"]
            else:
                e["structural_model"] = r.get("structural_model")
                e["tip_bm"] = r.get("tip_bm")
            res[fid] = e
        res["delta_a1_minus_full"] = {"cost": res["full_a1"]["cost"] - res["full"]["cost"],
                                      "J_wing_tip_bm_limit": res["full_a1"]["J_wing_tip_bm_limit"] - res["full"]["J_wing_tip_bm_limit"],
                                      "fd_struct_sum": res["full_a1"]["fd_struct_sum"] - res["full"]["fd_struct_sum"]}
        out["cases"][case] = res
        print(case, {f: (res[f]["cost"], res[f]["J_wing_tip_bm_limit"], res[f]["status"]) for f in ("full", "full_a1")})
    out["full_all_bit_identical"] = all(c["full"]["bit_identical"] for c in out["cases"].values())
    out["full_a1_model_version_ok"] = all(c["full_a1"]["model_version_ok"] for c in out["cases"].values())
    with open(os.path.join(HERE, "p3a1_tip_verify_evolution.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("full bit-identical:", out["full_all_bit_identical"], " A1 mv ok:", out["full_a1_model_version_ok"])


if __name__ == "__main__":
    main()
