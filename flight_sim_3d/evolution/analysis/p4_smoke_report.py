"""phase4-smoke-s1 report + trajectory export + fresh-process replay (read-only on the run dir except trajectories/).
  cd /workspace/er_smoke_code_p4 && EVOLUTION_FD_DIR=/workspace/flight-sim-team/evolution/_fd_pin_p4cs OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -B evolution/analysis/p4_smoke_report.py export   # rank-0 trajectories
  ... p4_smoke_report.py replay    # FRESH process: re-fly one exported (genome, course), compare bit for bit
-> evolution/analysis/phase4_smoke_s1_{export,replay}.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); EVO = os.path.dirname(HERE); ROOT = os.path.dirname(EVO)
sys.path.insert(0, ROOT)
from evolution import phase4_loop as L, validate_traj as V  # noqa: E402
RUN = os.path.join(EVO, "runs", os.environ.get("P4_RUN", "phase4-smoke-s1"))
cfg = json.load(open(os.path.join(RUN, "config.json")))["resolved"]
summ = json.load(open(os.path.join(RUN, "summary.json")))
rows = [json.loads(x) for x in open(os.path.join(RUN, "genomes.jsonl"))]
TD = os.path.join(RUN, "trajectories")
mode = sys.argv[1] if len(sys.argv) > 1 else "export"
G = int(cfg["ga"]["generations"])
if mode == "export":
    rc = L.mods()["rc"]
    ents, idx = [], []
    for a in summ["aircraft"]:
        m = a["aircraft"]
        row = next(r for r in rows if r["aircraft"] == m and r["gen"] == G - 1 and r["rank"] == 0)
        jobs = [("train", row["stage"], row["seeds"][0], next(c["cost"] for c in row["courses"] if c["seed"] == row["seeds"][0])),
                ("holdout", a["holdout"]["stage"], a["holdout"]["seeds"][0], a["holdout"]["per_course"][0]["cost"])]
        for label, stage, seed, logged in jobs:
            fn, doc, res = L.export_traj(cfg, m, row["u"], stage, seed, summ["run_id"], G - 1, TD, label)
            errs = V.validate_doc(doc, fn)
            cb = doc["course"]   # SB spec 5: course block (canonical NED + centre_enu_m/normal_enu), gates, gate_summary, ctrl_surfaces
            if len(cb["rings"]) != cb["M"] or any(k not in rg for rg in cb["rings"] for k in ("centre_m", "normal", "radius_m", "centre_enu_m", "normal_enu", "id")):
                errs.append("course block rings incomplete")
            if [g["k"] for g in doc["gates"]] != sorted(g["k"] for g in doc["gates"]) or doc["gate_summary"]["passes"] != sum(g["result"] == "pass" for g in doc["gates"]):
                errs.append("gates inconsistent")
            if cb["version"] != rc.VERSION:
                errs.append("course version")
            ents.append({"aircraft": m, "file": fn, "label": label, "stage": stage, "seed": seed, "cost": res["cost"],
                         "logged_cost": logged, "bit_identical_vs_log": res["cost"] == logged, "validate_errors": errs,
                         "passes": res["summary"]["passes"], "rings_in_file": len(doc["course"]["rings"])})
            idx.append({"generation": G - 1, "fitness": res["cost"], "aircraft": m, "file": fn})
    from evolution import trajectory as TJ
    TJ.write_index(TD, summ["run_id"], [e for e in idx])
    json.dump({"entries": ents}, open(os.path.join(HERE, f"{os.path.basename(RUN).replace('-','_')}_export.json"), "w"), indent=1)
    print(json.dumps([(e["file"], e["bit_identical_vs_log"], len(e["validate_errors"])) for e in ents]))
else:
    ex = json.load(open(os.path.join(HERE, f"{os.path.basename(RUN).replace('-','_')}_export.json")))["entries"]
    e = next(x for x in ex if x["aircraft"] == "T38" and x["label"] == "holdout") if any(x["aircraft"] == "T38" for x in ex) else ex[0]
    doc = json.load(open(os.path.join(TD, e["file"])))
    row = next(r for r in rows if r["aircraft"] == e["aircraft"] and r["gen"] == G - 1 and r["rank"] == 0)
    r = L.fly_one(cfg, e["aircraft"], row["u"], e["stage"], e["seed"])
    out = {"trajectory": e["file"], "individual_id": row["id"], "course_seed": e["seed"], "stage": e["stage"],
           "logged_cost": e["logged_cost"], "file_cost": doc["fitness"], "replayed_cost": r["cost"],
           "bit_identical": r["cost"] == e["logged_cost"] == doc["fitness"], "gates_identical": r["gates"] == doc["gates"],
           "model_version": r["model_version"], "pin": cfg["pin_model_version"][e["aircraft"]], "fd_dir": L.FD_DIR,
           "check_value": repr(r["cost"])}
    json.dump(out, open(os.path.join(HERE, f"{os.path.basename(RUN).replace('-','_')}_replay.json"), "w"), indent=1)
    print(json.dumps(out))
