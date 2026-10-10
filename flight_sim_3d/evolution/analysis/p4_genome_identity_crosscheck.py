"""Cross-check vs Genome's identity K=4 evaluations (genome/runs/p4_eval_identity_<m>_K4.json), one aircraft per process:
  OMP_NUM_THREADS=1 ... $PY -B evolution/analysis/p4_genome_identity_crosscheck.py <model>  -> p4_genome_identity_crosscheck_<m>.json
(a) Genome's own evaluator (genome/p4_guidance.fly_one, legacy phase4_eval.score_course) re-run here: per-course cost vs file.
(b) ER loop (phase4_loop.fly_one, SB score_course + amended terms + carried): passes, terms, cost; and FD flight identical to (a)."""
import json, os, sys, platform
HERE = os.path.dirname(os.path.abspath(__file__)); TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)
from evolution import phase4_loop as L  # noqa: E402
m = sys.argv[1]
ref = json.load(open(os.path.join(TEAM, "genome", "runs", f"p4_eval_identity_{m}_K4.json")))
cfg = L.resolve_config(json.load(open(os.path.join(TEAM, "evolution", "configs", "phase4_smoke.json"))), "x")
L.mods(); gg = L.genome_guidance()
import phase4_rings as GR  # noqa: E402
P = GR.load_preset(); u = GR.identity_u(P["genes"])
out = {"model": m, "python": sys.executable, "py_version": platform.python_version(), "genome_file_cost": ref["cost"], "courses": []}
for seed, rc_ in zip(ref["course_seeds"], ref["per_course"]):
    sco, r, course = gg.fly_one(u, P["genes"], m, ref["stage"], seed)
    mine = L.fly_one(cfg, m, list(u), ref["stage"], seed, record=True)
    f = mine["_flight"]
    out["courses"].append({"seed": seed, "genome_file": rc_, "genome_rerun_cost": sco["cost"], "genome_rerun_n_pass": sco.get("n_pass"),
                           "genome_rerun_sb_passes": sco["sb_diag"]["passes"], "rerun_equals_file": sco["cost"] == rc_["cost"],
                           "fd_flight_identical": f["pos"] == r["pos"] and f["att"] == r["att"],
                           "er_cost": mine["cost"], "er_passes": mine["summary"]["passes"], "er_terms": mine["terms"],
                           "genome_terms": sco["terms"], "er_carried": mine["summary"].get("carried")})
json.dump(out, open(os.path.join(HERE, f"p4_genome_identity_crosscheck_{m}.json"), "w"), indent=1, default=str)
print(m, [(c["rerun_equals_file"], c["fd_flight_identical"], c["genome_rerun_n_pass"], c["er_passes"], round(c["er_cost"], 4)) for c in out["courses"]])
