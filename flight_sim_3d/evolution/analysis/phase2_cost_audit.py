"""Audit the Phase 2 cost composition on real genomes (read-only, no cache):  per scenario
   cost = sim_cost + sum(flown struct terms) + sum(FD PRE_TERMS)        (FD flexeval, full)
   sim_cost = track + w_effort*effort + w_comfort*comfort + w_heading*heading_rms(/hdg_rms_ref) [+ w_hold*hold_osc]
and the genome cost = mean over scenarios. Usage: $PY evolution/analysis/phase2_cost_audit.py RUN_DIR"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evolution import eval as ev, fidelity as F, sim  # noqa: E402


def main(run_dir):
    run = ev.load_run_cfg(os.path.join(run_dir, "run.json"))
    rows = [json.loads(x) for x in open(os.path.join(run_dir, "genomes.jsonl"))]
    fe = F.fd_modules()["fe"]
    worst = 0.0
    for ent in run["aircraft"]:
        n = ent["name"]
        best = min((r for r in rows if r["aircraft"] == n), key=lambda r: (-r["generation"], r["rank"]))
        prof = ev._profile_d(ent)
        P = sim.Profile.from_dict(prof)
        gains, struct = ev.split_values(best["genome"], ev.gene_groups(ent))
        r = F.evaluate_genome(prof, gains, struct, [ev.runinfo.scenario_fields(s) for s in ev.scenario_dicts(n, None, run)],
                              "full", ent.get("reduced_gate"))
        pre = sum(r["terms"][k] for k in fe.PRE_TERMS["full"])
        print(f"{n} {best['individual_id']}: cost {r['cost']!r} (logged {best['cost']!r}); pre-flight sum {pre:.6g}")
        for i, e in enumerate(r["per_scenario"]):
            resp = sum(e.get("struct", {}).values())
            simc = e["track"] + P.w_effort * e["effort"] + P.w_comfort * e["comfort"] + P.w_heading * e["heading_rms"] \
                + (P.w_hold * e["hold_osc"] if P.w_hold > 0 else 0.0)
            d1 = e["cost"] - (e["sim_cost"] + resp + pre)
            d2 = e["sim_cost"] - simc
            worst = max(worst, abs(d1), abs(d2))
            print(f"  s{i}: cost {e['cost']:.9g} = sim {e['sim_cost']:.9g} [track {e['track']:.6g} + 2.0*effort "
                  f"{e['effort']:.6g} + 0.05*comfort {e['comfort']:.6g} + 0.01*hdg {e['heading_rms']:.6g}] + flown "
                  f"{resp:.6g} + pre {pre:.6g}   residuals {d1:.2e} {d2:.2e}")
        print(f"  mean check {r['cost'] - np.mean([e['cost'] for e in r['per_scenario']]):.2e}; "
              f"weights w_effort {P.w_effort} w_comfort {P.w_comfort} w_heading {P.w_heading} (hdg ref "
              f"{P.hdg_rms_ref_deg} deg) w_hold {P.w_hold} ramp {P.ramp_fpm} fpm / {P.ramp_accel_g} g ff {P.alt_ref_ff}")
    print("max |residual|", worst)


if __name__ == "__main__":
    main(sys.argv[1])
