"""ER-side re-run of genome/p3b2a_verify.py V1 (genome/ read-only): full_a1_b2a through ER evaluate_genome vs
full_a1_b1 through ER, on Genome's setup (phase2_smoke_p25 c172x, 3 scen seed 1, p25_tip_verify baseline gains+struct).
Expect baseline 0.2419400885448951, soft tip 0.25879473336084763, fly shapes B2a == B1 bit-for-bit.
  EVOLUTION_FD_DIR=evolution/_fd_pin_p3b2a $PY -B evolution/analysis/p3b2a_er_verify.py -> p3b2a_er_verify.json"""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)
from evolution import fidelity as F, batch, sim  # noqa: E402
cfg = batch.resolve_config(json.load(open(os.path.join(TEAM, "evolution/configs/phase2_smoke_p25.json"))), "er_b2a_verify")
pd = next(a for a in cfg["aircraft"] if a["name"] == "c172x")["resolved_profile"]
scs = [s.to_dict() for s in sim.make_scenarios(3, 1, sim.Profile.from_dict(pd))]
g = json.load(open(os.path.join(TEAM, "genome/runs/p25_tip_verify_c172x.json")))["baseline"]
gains, struct = g["gains"], g["struct"]
EXP = {"baseline": 0.2419400885448951, "soft_tip_taper4_0.75": 0.25879473336084763}
FLY = {"washout_tip_m2": {"wing_twist_tip_deg": -2.0}, "chord_taper3_0.9": {"wing_chord_taper_3": 0.9},
       "sweep_p3": {"wing_sweep_qc_delta_deg": 3.0},
       "combo_fd_bench": {"wing_chord_taper_1": 0.95, "wing_chord_taper_2": 0.95, "wing_chord_taper_3": 0.95,
                          "wing_twist_mid_deg": -0.5, "wing_twist_tip_deg": -2.0, "wing_sweep_qc_delta_deg": 2.0},
       "twist_mid_m2": {"wing_twist_mid_deg": -2.0}, "twist_mid_p1": {"wing_twist_mid_deg": 1.0}}
cases = [("baseline", struct, {}), ("soft_tip_taper4_0.75", dict(struct, wing_ei_taper_4=0.75), {})] + \
        [(k, struct, v) for k, v in FLY.items()]
out = {"fd_dir": F.FD_DIR, "cases": {}}
ps = lambda r: [float(e["cost"]) for e in r["per_scenario"]]
for name, st, sh in cases:
    b1 = F.evaluate_genome(pd, gains, st, scs, F.B1, None, shape=sh)
    b2 = F.evaluate_genome(pd, gains, st, scs, F.B2, None, shape=sh)
    b2e = F.evaluate_genome(pd, gains, st, scs, F.B2, None, shape=sh, energy_cost=True)
    rec = {"b1": b1["cost"], "b2a": b2["cost"], "b2a_energy_on": b2e["cost"], "status": b2["status"],
           "bit_identical_b2a_vs_b1": b2["cost"] == b1["cost"] and ps(b2) == ps(b1) and dict(b2["terms"]) == dict(b1["terms"]),
           "energy_on_equal": b2e["cost"] == b2["cost"], "J_energy": b2e["energy_terms"]["J_energy"],
           "J_speed_guard": b2e["energy_terms"]["J_speed_guard"], "mv_b1": b1["model_version"], "mv_b2a": b2["model_version"]}
    if name in EXP:
        rec["expected"] = EXP[name]; rec["matches_expected"] = b2["cost"] == EXP[name]
    out["cases"][name] = rec; print(name, rec, flush=True)
out["all_ok"] = all(r["bit_identical_b2a_vs_b1"] and r.get("matches_expected", True) for r in out["cases"].values())
json.dump(out, open(os.path.join(HERE, "p3b2a_er_verify.json"), "w"), indent=1)
print("ALL_OK", out["all_ok"])
