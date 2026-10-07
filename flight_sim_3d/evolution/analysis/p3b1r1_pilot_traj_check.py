"""P3-B1 r1 pilot trajectory checks (read-only on the run dir), FRESH process, pilot code snapshot + frozen FD tree:
  EVOLUTION_CODE_ROOT=/workspace/er_pilot_code_b1r1 EVOLUTION_TEAM_ROOT=/workspace/flight-sim-team \
  EVOLUTION_FD_DIR=/workspace/flight-sim-team/evolution/_fd_pin_p3b1r1 $PY -B evolution/analysis/p3b1r1_pilot_traj_check.py [RUN_DIR] [AIRCRAFT] [GEN]
1. every exported trajectory (+ index.json): validate_traj (in-process) and header checks: planform (fd-planform/1),
   structure.node_layout = FD flexbody_b1.node_layout_b1 (P3-B1 r1) with per-node chord / twist / LE / TE, and
   structure.modal_twist_sign_fixed true; fidelity / model_version = run pin; scenario cost = genomes.jsonl best row.
2. replay: one file's genome + scenario re-flown through evolution.eval.evaluate (run.json interface) in this fresh
   process; cost / model_version compared bit for bit with the trajectory header and genomes.jsonl.
-> analysis/phase3b1r1_pilot_s1_traj_check.json"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.environ.get("EVOLUTION_TEAM_ROOT") or os.path.dirname(os.path.dirname(HERE))
CODE = os.environ.get("EVOLUTION_CODE_ROOT") or TEAM
sys.path.insert(0, CODE)
from evolution import eval as ev, fidelity as F, validate_traj as V  # noqa: E402

NODE_LAYOUT = "FD flexbody_b1.node_layout_b1 (P3-B1 r1)"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    run = argv[0] if argv else os.path.join(TEAM, "evolution", "runs", "phase3b1r1-pilot-s1")
    ac = argv[1] if len(argv) > 1 else "T38"
    gen = int(argv[2]) if len(argv) > 2 else 59
    rj = json.load(open(os.path.join(run, "run.json")))
    rows = [json.loads(x) for x in open(os.path.join(run, "genomes.jsonl")) if x.strip()]
    tdir = os.path.join(run, "trajectories")
    files = sorted(glob.glob(os.path.join(tdir, "traj_*.json")))
    out = {"run_dir": os.path.relpath(run, TEAM), "code_root": CODE, "run_json_fd_dir": rj.get("fd_dir"),
           "this_process_fd_dir": os.path.relpath(F.FD_DIR, TEAM),
           "this_process_flexeval_b1": os.path.relpath(F.fd_b1_modules()["fb1"].__file__, TEAM),
           "validate_traj_exit": V.main([tdir, "--min-gens-per-aircraft", "3"]), "files": {}}
    out["fd_dir_ok"] = (out["run_json_fd_dir"] == "evolution/_fd_pin_p3b1r1" == out["this_process_fd_dir"]
                        and out["this_process_flexeval_b1"].startswith("evolution/_fd_pin_p3b1r1/"))
    idx = json.load(open(os.path.join(tdir, "index.json")))
    out["index_entries"] = len(idx.get("entries", idx) if isinstance(idx, dict) else idx)
    pins = {a: v for a, v in rj["model_version"].items()}
    for f in files:
        d = json.load(open(f))
        st = d.get("structure") or {}
        pl = d.get("planform") or {}
        row = min((r for r in rows if r["aircraft"] == d["aircraft"] and r["generation"] == d["generation"]),
                  key=lambda r: r["rank"])
        k = row["scenario_ids"].index(d["scenario_id"])
        comps = {c["name"]: c for c in st.get("components", [])}
        per_node = {w: (w in comps and all(len(comps[w].get(key) or []) == len(comps[w]["axis_nodes_body_m"])
                                           for key in ("chord_m", "geometric_twist_rad", "le_nodes_body_m", "te_nodes_body_m")))
                    for w in ("wingR", "wingL")}
        out["files"][os.path.basename(f)] = {
            "schema": d["schema"], "planform_schema": pl.get("schema"), "planform_source": pl.get("source"),
            "planform_genes_equal_row_shape": pl.get("genes") == row.get("shape"),
            "node_layout": st.get("node_layout"), "node_layout_r1": st.get("node_layout") == NODE_LAYOUT,
            "shaped_wings_with_per_node_fields": per_node, "modal_twist_sign_fixed": st.get("modal_twist_sign_fixed"),
            "fidelity": d.get("fidelity"), "model_version": d.get("model_version"),
            "model_version_is_pin": d.get("model_version") == pins.get(d["aircraft"]),
            "scenario_cost": d["scenario_cost"], "row_id": row["individual_id"], "row_scenario_cost": row["per_scenario_cost"][k],
            "cost_equals_row": d["scenario_cost"] == row["per_scenario_cost"][k], "genome_equals_row": d["genome"] == row["genome"]}
    out["all_headers_ok"] = all(v["planform_schema"] == "fd-planform/1" and v["node_layout_r1"] and v["modal_twist_sign_fixed"] is True
                                and v["model_version_is_pin"] and v["cost_equals_row"] and v["genome_equals_row"]
                                and v["shaped_wings_with_per_node_fields"] and all(v["shaped_wings_with_per_node_fields"].values())
                                for v in out["files"].values())
    tf = next(f for f in files if os.path.basename(f).startswith(f"traj_{ac}_") and f.endswith(f"_g{gen}.json"))
    d = json.load(open(tf))
    row = min((r for r in rows if r["aircraft"] == ac and r["generation"] == gen), key=lambda r: r["rank"])
    k = row["scenario_ids"].index(d["scenario_id"])
    r = ev.evaluate(row["genome"], ac, d["scenario_id"], rj)
    out["replay"] = {"trajectory": os.path.relpath(tf, TEAM), "individual_id": row["individual_id"], "scenario_id": d["scenario_id"],
                     "traj_scenario_cost": d["scenario_cost"], "row_per_scenario_cost": row["per_scenario_cost"][k],
                     "replayed_cost": r["cost"], "replayed_model_version": r["model_version"], "traj_model_version": d["model_version"],
                     "bit_identical_vs_traj": r["cost"] == d["scenario_cost"], "bit_identical_vs_row": r["cost"] == row["per_scenario_cost"][k],
                     "model_version_match": r["model_version"] == d["model_version"] == row["model_version"]}
    with open(os.environ.get("REPORT_OUT") or os.path.join(HERE, "phase3b1r1_pilot_s1_traj_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    for fn, v in out["files"].items():
        print(fn, {kk: v[kk] for kk in ("planform_schema", "node_layout_r1", "modal_twist_sign_fixed", "model_version_is_pin",
                                        "cost_equals_row", "planform_genes_equal_row_shape")}, v["shaped_wings_with_per_node_fields"])
    print("validate_traj exit", out["validate_traj_exit"], "index entries", out["index_entries"], "all_headers_ok", out["all_headers_ok"],
          "fd_dir_ok", out["fd_dir_ok"])
    print("replay", out["replay"])


if __name__ == "__main__":
    main()
