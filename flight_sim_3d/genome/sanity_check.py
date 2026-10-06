#!/usr/bin/env python3
"""Quick closed-loop sanity check of a task on one or more aircraft (no evolution).

    python sanity_check.py --task phase1_default --aircraft c172x t38 b737 [--random 24]

Flies the profile-scaled DEFAULT genome on the task's scenarios and (optionally)
N random genomes from the derived ranges, and prints status / tracking / pitch / nz.
"""
import argparse
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapter  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="phase1_default")
    ap.add_argument("--aircraft", nargs="+", default=["c172x", "t38", "b737"])
    ap.add_argument("--scenarios", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    report = {}
    for ac in a.aircraft:
        t = adapter.load_task(a.task, {"aircraft": ac})
        scs = t.make_scenarios(a.scenarios, a.seed)
        g = t.spec.decode(t.spec.default_genome())
        r = t.evaluate(g, scs, record=True)
        rep = {"default_gains": g, "cost": r["cost"], "objectives": r["objectives"],
               "per_scenario": r["per_scenario"], "diagnostics": r["diagnostics"], "warnings": t.warnings}
        print(f"== {ac} ({t.profile.jsbsim_model}) default genome: cost {r['cost']:.4f}")
        for i, (ps, d) in enumerate(zip(r["per_scenario"], r["diagnostics"])):
            if ps["status"] != "ok" and "max_pitch_deg" not in d:
                print(f"   scen {i}: {ps['status']} at t={ps['t_end']:.1f}s")
                continue
            print(f"   scen {i}: {ps['status']:8s} t_end {ps['t_end']:5.1f}s  max ref err {d['max_ref_err_ft']:6.1f} ft  "
                  f"overshoot {d['max_overshoot_ft']:5.1f} ft  pitch {d['max_pitch_deg']:5.1f} deg (trim {d['theta_trim_deg']:.1f})  "
                  f"nz [{d['min_nz']:.2f}, {d['max_nz']:.2f}]  climb {d['max_climb_fpm']:.0f} fpm")
        if a.random:
            rng = np.random.default_rng(a.seed)
            G = rng.random((a.random, t.spec.n_genes))
            res = [t.evaluate(t.spec.decode(x), scs) for x in G]
            ok = [all(s["status"] == "ok" for s in x["per_scenario"]) for x in res]
            st = {}
            for x in res:
                for s in x["per_scenario"]:
                    st[s["status"]] = st.get(s["status"], 0) + 1
            best = min(x["cost"] for x in res)
            print(f"   random genomes: {sum(ok)}/{a.random} fly all scenarios; statuses {st}; best cost {best:.4f}")
            rep["random"] = {"n": a.random, "all_ok": int(sum(ok)), "statuses": st, "best_cost": best}
        report[ac] = rep
    if a.json:
        json.dump(report, open(a.json, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
