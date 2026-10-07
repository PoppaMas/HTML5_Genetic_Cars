"""B1 r1 smoke checks (read-only on the run dir), in a FRESH process; run with the run's own FD tree:
  EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1 $PY -B evolution/analysis/p3b1r1_replay_check.py [RUN_DIR]
1. fd_dir: run.json "fd_dir" (and runs/<id>/fd_pin.json if present) must be evolution/_fd_pin_p3b1r1, and this process
   must have loaded FD from that tree (evolution.fidelity.FD_DIR, flexeval_b1.__file__).
2. Replay: for one exported trajectory (default T38 g4) re-fly its genome + scenario through evolution.eval.evaluate
   (run.json interface, fresh process) and compare the scenario cost / model_version bit for bit with the trajectory
   header and with genomes.jsonl.
3. Shape vs controller split (as FD did on r0): the T38 best of the last generation re-flown on all its scenarios with
   the shape genes reset to the baseline (FD defaults), vs the logged cost and a re-fly with its own shape.
Options: --out NAME (file name under analysis/, or a path; default phase3b1r1_replay_check.json), so a second run
(e.g. the tweaked A/B arm) does not overwrite the baseline's record:
  ... p3b1r1_replay_check.py runs/phase3b1r1-pilot-tweaked-s1 --out phase3b1r1_replay_check_tweaked.json
-> analysis/<out>"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)
from evolution import eval as ev, fidelity as F  # noqa: E402


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    out_name = "phase3b1r1_replay_check.json"
    if "--out" in argv:
        k = argv.index("--out")
        out_name = argv[k + 1]
        del argv[k:k + 2]
    out_path = out_name if os.path.dirname(out_name) else os.path.join(HERE, out_name)
    run_dir = argv[0] if argv else os.path.join(TEAM, "evolution", "runs", "phase3b1r1-smoke-s1")
    traj_ac = argv[1] if len(argv) > 1 else "T38"
    rj = json.load(open(os.path.join(run_dir, "run.json")))
    rows = [json.loads(x) for x in open(os.path.join(run_dir, "genomes.jsonl")) if x.strip()]
    pin_side = os.path.join(run_dir, "fd_pin.json")
    out = {"run_dir": os.path.relpath(run_dir, TEAM), "run_json_fd_dir": rj.get("fd_dir"),
           "fd_pin_sidecar": json.load(open(pin_side)) if os.path.exists(pin_side) else None,
           "this_process_fd_dir": os.path.relpath(F.FD_DIR, TEAM),
           "this_process_flexeval_b1": os.path.relpath(F.fd_b1_modules()["fb1"].__file__, TEAM)}
    out["fd_dir_ok"] = (out["run_json_fd_dir"] == "evolution/_fd_pin_p3b1r1" == out["this_process_fd_dir"]
                        and out["this_process_flexeval_b1"].startswith("evolution/_fd_pin_p3b1r1/"))
    # 2. replay one trajectory
    tf = sorted(glob.glob(os.path.join(run_dir, "trajectories", f"traj_{traj_ac}_*_g*.json")))[-1]
    doc = json.load(open(tf))
    gen, sid = doc["generation"], doc["scenario_id"]
    row = min((r for r in rows if r["aircraft"] == traj_ac and r["generation"] == gen), key=lambda r: r["rank"])
    r = ev.evaluate(row["genome"], traj_ac, sid, rj)
    k = row["scenario_ids"].index(sid)
    out["replay"] = {"trajectory": os.path.relpath(tf, TEAM), "individual_id": row["individual_id"], "scenario_id": sid,
                     "traj_genome_equals_row": doc["genome"] == row["genome"],
                     "traj_scenario_cost": doc["scenario_cost"], "row_per_scenario_cost": row["per_scenario_cost"][k],
                     "replayed_cost": r["cost"], "replayed_model_version": r["model_version"],
                     "traj_model_version": doc["model_version"],
                     "bit_identical_vs_traj": r["cost"] == doc["scenario_cost"],
                     "bit_identical_vs_row": r["cost"] == row["per_scenario_cost"][k],
                     "model_version_match": r["model_version"] == doc["model_version"] == row["model_version"]}
    print("replay", out["replay"])
    # 3. T38 best: own shape vs baseline shape (controller + structure unchanged)
    last = max(x["generation"] for x in rows if x["aircraft"] == "T38")
    best = min((x for x in rows if x["aircraft"] == "T38" and x["generation"] == last), key=lambda x: x["rank"])
    defaults = {g.name: g.default for g in F.shape_schema()}
    g_base = dict(best["genome"], **defaults)
    own = ev.evaluate(best["genome"], "T38", None, rj)
    bas = ev.evaluate(g_base, "T38", None, rj)
    out["t38_shape_split"] = {"best_id": best["individual_id"], "logged_cost": best["cost"], "shape": best.get("shape"),
                              "refly_own_shape_cost": own["cost"], "refly_own_bit_identical": own["cost"] == best["cost"],
                              "baseline_shape_cost": bas["cost"], "baseline_shape_status": bas["status"],
                              "shape_contribution": best["cost"] - bas["cost"],
                              "per_scenario_own": own["per_scenario_cost"], "per_scenario_baseline_shape": bas["per_scenario_cost"],
                              "terms_own": own["terms"], "terms_baseline_shape": bas["terms"],
                              "r0_reference": {"best": 0.1562, "baseline_shape": 0.1571, "source": "FD on phase3b1-smoke-s1"}}
    print("T38 split", {k: v for k, v in out["t38_shape_split"].items() if not k.startswith("terms")})
    out["out_file"] = os.path.relpath(out_path, TEAM)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", out_path)
    print("fd_dir_ok", out["fd_dir_ok"])


if __name__ == "__main__":
    main()
