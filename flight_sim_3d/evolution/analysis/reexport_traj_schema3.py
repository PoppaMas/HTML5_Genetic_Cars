"""Re-export gen 0 / mid / final best trajectories for an existing run through evaluate (telemetry=sb),
writing ga-flightsim-traj/2 + FlexState/3. Backs up old files to trajectories_traj1/. Costs must stay bit-identical.
  $PY evolution/analysis/reexport_traj_schema3.py RUN_DIR
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evolution import eval as ev, fidelity as F, genome, runinfo, sim, trajectory  # noqa: E402


def main(run_dir):
    run = ev.load_run_cfg(os.path.join(run_dir, "run.json"))
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "genomes.jsonl")) if l.strip()]
    G = max(r["generation"] for r in rows) + 1
    gens = [0, (G - 1) // 2, G - 1]
    traj_dir = os.path.join(run_dir, "trajectories")
    bak = os.path.join(run_dir, "trajectories_traj1")
    if os.path.isdir(traj_dir) and not os.path.isdir(bak):
        shutil.copytree(traj_dir, bak)
        print("backed up", traj_dir, "->", bak)
    elif os.path.isdir(traj_dir):
        print("backup already exists:", bak)
    entries, mismatches = [], []
    s_idx = 0
    for ent in run["aircraft"]:
        name = ent["name"]
        groups = ev.gene_groups(ent)
        sch = ev.schema_for(ent)
        prof = ev._profile_d(ent)
        scs = ev.scenario_dicts(name, None, run)
        gate = ent.get("reduced_gate")
        fid = run.get("fidelity", "full")
        for g in gens:
            best = min((r for r in rows if r["aircraft"] == name and r["generation"] == g), key=lambda r: r["rank"])
            gains, struct = ev.split_values(best["genome"], groups)
            # evaluate_genome with record + sb telemetry (FlexState /3)
            out = F.evaluate_genome(prof, gains, struct, [runinfo.scenario_fields(s) for s in scs], fid, gate,
                                    record=True, sample_hz=30.0, telemetry="sb")
            r = {**out["per_scenario"][s_idx], "trajectory": (out.get("trajectories") or [None])[s_idx],
                 "cost": out["per_scenario"][s_idx]["cost"], "status": out["per_scenario"][s_idx]["status"],
                 "t_end": out["per_scenario"][s_idx].get("t_end"), "fidelity": out["fidelity"],
                 "model_version": out["model_version"]}
            # attach structure from trajectory if present
            logged = best["per_scenario_cost"][s_idx]
            if r["cost"] != logged:
                mismatches.append({"aircraft": name, "generation": g, "logged": logged, "resim": r["cost"]})
            # build_doc expects sim_result with trajectory key from TrajRecorder shape
            tr = r.get("trajectory")
            if tr is None:
                raise RuntimeError(f"no trajectory for {name} g{g}")
            # TrajRecorder puts structure on the trajectory dict
            doc = trajectory.build_doc(run_id=run["run_id"], aircraft=name, jsbsim_version=run.get("jsbsim_version"),
                                       git_sha=(run.get("git") or {}).get("sha") or run.get("git_sha"),
                                       seed=ent["seed"], generation=g, fitness=best["cost"],
                                       gains=best["genome"], scenario=scs[s_idx], scenario_index=s_idx,
                                       sim_result={"cost": r["cost"], "status": r["status"], "t_end": r["t_end"],
                                                   "trajectory": tr},
                                       profile=prof,
                                       extra={"fidelity": r["fidelity"], "model_version": r["model_version"],
                                              "scenario_id": best["scenario_ids"][s_idx],
                                              "reexported": "schema3_v2_map", "logged_scenario_cost": logged})
            fn = trajectory.write_doc(doc, traj_dir)
            entries.append({"generation": g, "fitness": best["cost"], "aircraft": name, "file": fn})
            print(name, f"g{g}", doc["schema"], "structure", (doc.get("structure") or {}).get("schema"),
                  "cost", r["cost"], "match", r["cost"] == logged,
                  "comps", [c["name"] for c in (doc.get("structure") or {}).get("components", [])][:4])
    trajectory.write_index(traj_dir, run["run_id"], entries)
    print("mismatches", mismatches)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
