"""Fresh-process replay of one phase3b2a-smoke-s1 trajectory through the run's own frozen FD tree:
  EVOLUTION_FD_DIR=evolution/_fd_pin_p3b2a $PY -B evolution/analysis/p3b2a_replay_check.py [TRAJ_FILE] [--out NAME]
fd_dir check (run.json fd_dir == this process's FD_DIR == evolution/_fd_pin_p3b2a, flexeval_b2 loaded from it), then
evolution.eval.evaluate(genome, aircraft, scenario, run.json) vs trajectory header / genomes.jsonl, bit for bit
(cost incl. energy terms, model_version). -> analysis/<out> (default phase3b2a_smoke_s1_replay_check.json)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)
from evolution import eval as ev, fidelity as F  # noqa: E402
a = sys.argv[1:]
out_name = "phase3b2a_smoke_s1_replay_check.json"
if "--out" in a:
    k = a.index("--out"); out_name = a[k + 1]; del a[k:k + 2]
RUN = os.path.join(TEAM, "evolution/runs/phase3b2a-smoke-s1")
tf = a[0] if a else os.path.join(RUN, "trajectories", "traj_c172x_phase3b2a-smoke-s1_g2.json")
rj = json.load(open(os.path.join(RUN, "run.json")))
doc = json.load(open(tf))
rows = [json.loads(x) for x in open(os.path.join(RUN, "genomes.jsonl")) if x.strip()]
row = min((r for r in rows if r["aircraft"] == doc["aircraft"] and r["generation"] == doc["generation"]), key=lambda r: r["rank"])
out = {"run_json_fd_dir": rj.get("fd_dir"), "this_process_fd_dir": os.path.relpath(F.FD_DIR, TEAM),
       "this_process_flexeval_b2": os.path.relpath(F.fd_b2_modules()["fb2"].__file__, TEAM), "energy_cost": rj.get("energy_cost")}
out["fd_dir_ok"] = (out["run_json_fd_dir"] == "evolution/_fd_pin_p3b2a" == out["this_process_fd_dir"]
                    and out["this_process_flexeval_b2"].startswith("evolution/_fd_pin_p3b2a/"))
sid = doc["scenario_id"]; k = row["scenario_ids"].index(sid)
r = ev.evaluate(row["genome"], doc["aircraft"], sid, rj)
out["replay"] = {"trajectory": os.path.relpath(tf, TEAM), "individual_id": row["individual_id"], "scenario_id": sid,
                 "dihedral_delta_deg": row["shape"]["wing_dihedral_delta_deg"],
                 "traj_genome_equals_row": doc["genome"] == row["genome"], "traj_scenario_cost": doc["scenario_cost"],
                 "row_per_scenario_cost": row["per_scenario_cost"][k], "replayed_cost": r["cost"],
                 "replayed_energy_terms": r.get("energy_terms"),
                 "bit_identical_vs_traj": r["cost"] == doc["scenario_cost"], "bit_identical_vs_row": r["cost"] == row["per_scenario_cost"][k],
                 "replayed_model_version": r["model_version"],
                 "model_version_match": r["model_version"] == doc["model_version"] == row["model_version"]}
p = out_name if os.path.dirname(out_name) else os.path.join(HERE, out_name)
json.dump(out, open(p, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out["replay"].items() if k != "replayed_energy_terms"}), "fd_dir_ok", out["fd_dir_ok"])
