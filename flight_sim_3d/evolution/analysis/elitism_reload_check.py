# usage: EVOLUTION_FD_DIR=<run fd pin> $PY -B evolution/analysis/elitism_reload_check.py <run_dir> <aircraft> <gen> <out.json>  (read-only)
"""Fresh-process reload: take a NON-trajectory generation's rank-0 row from genomes.jsonl, re-fly it via
evolution.eval.evaluate (run.json interface) on scenario s0, compare bit for bit. argv: run_dir aircraft gen out.json"""
import json, os, sys
TEAM = os.environ.get("EVOLUTION_TEAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # flight_sim_3d/
sys.path.insert(0, TEAM)
from evolution import eval as ev, fidelity as F, genome as gm
run_dir, ac, gen, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rj = json.load(open(os.path.join(run_dir, "run.json")))
row = None
for line in open(os.path.join(run_dir, "genomes.jsonl")):
    r = json.loads(line)
    if r["aircraft"] == ac and r["generation"] == gen and r["rank"] == 0:
        row = r; break
ck = json.load(open(os.path.join(run_dir, "checkpoints", f"{ac}.json")))
bpg = next(b for b in ck["best_per_gen"] if b["generation"] == gen)
out = {"run_dir": os.path.relpath(run_dir, TEAM), "aircraft": ac, "generation": gen,
       "individual_id": row["individual_id"], "fd_dir": os.path.relpath(F.FD_DIR, TEAM),
       "logged_cost_s0": row["per_scenario_cost"][0], "logged_model_version": row["model_version"]}
r1 = ev.evaluate(row["genome"], ac, 0, rj)                                     # decoded genes from genomes.jsonl
r2 = ev.evaluate(bpg["genome"], ac, 0, rj)                                     # normalized genes from checkpoint best_per_gen
out.update({"replay_from_genomes_jsonl_genome": r1["cost"], "replay_from_checkpoint_genome_norm": r2["cost"],
            "model_version": r1["model_version"],
            "bit_identical_genomes_jsonl": r1["cost"] == row["per_scenario_cost"][0],
            "bit_identical_checkpoint": r2["cost"] == row["per_scenario_cost"][0],
            "model_version_match": r1["model_version"] == row["model_version"]})
print(out)
json.dump(out, open(outp, "w"), indent=1)
