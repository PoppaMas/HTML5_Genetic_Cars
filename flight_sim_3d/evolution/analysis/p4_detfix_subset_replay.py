"""Bit-for-bit guard after the Phase 4 wiring: fresh-process replay of a fixed subset of phase3b2a-smoke-detfix-s1
genomes.jsonl rows (the 3 formerly-mismatching c172x rows + 27 rows at a fixed stride), every scenario of each row,
through evolution.eval.evaluate with the run's own FD pin:
  EVOLUTION_FD_DIR=evolution/_fd_pin_p3b2a $PY -B evolution/analysis/p4_detfix_subset_replay.py -> p4_detfix_subset_replay.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)
from evolution import eval as ev, fidelity as F  # noqa: E402
RUN = os.path.join(TEAM, "evolution/runs/phase3b2a-smoke-detfix-s1")
rj = json.load(open(os.path.join(RUN, "run.json")))
rows = [json.loads(x) for x in open(os.path.join(RUN, "genomes.jsonl")) if x.strip()]
must = {"c172x:g0:r9", "c172x:g0:r10", "c172x:g1:r15"}
pick = [r for r in rows if r["individual_id"] in must]
pick += [r for i, r in enumerate(rows) if i % 8 == 3 and r["individual_id"] not in must][:27]
res, bad = [], 0
for r in pick:
    for k, sid in enumerate(r["scenario_ids"]):
        c = ev.evaluate(r["genome"], r["aircraft"], sid, rj)["cost"]
        ok = c == r["per_scenario_cost"][k]
        bad += not ok
        res.append({"id": r["individual_id"], "scenario": sid, "logged": r["per_scenario_cost"][k], "replayed": c, "exact": ok})
out = {"fd_dir": os.path.relpath(F.FD_DIR, TEAM), "rows": len(pick), "replays": len(res), "mismatches": bad, "all_exact": bad == 0, "results": res}
json.dump(out, open(os.path.join(HERE, "p4_detfix_subset_replay.json"), "w"), indent=1)
print(out["rows"], out["replays"], out["mismatches"], out["fd_dir"])
