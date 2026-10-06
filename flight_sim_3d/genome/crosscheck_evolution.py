"""Cross-check against Evolution Runner's jet runs (read-only on evolution/).

1. Re-fly their best T38/737 genomes in OUR sim (sim_ext) under THEIR task (instant speed-scaled step, legacy
   pitch clamp, unscaled helper loops, their envelope) and recompute their cost from our telemetry with their
   alt_err_scale -> checks that the two simulators agree.
2. Score the same genomes under OUR Phase-1 fitness (shared profile set) -> how their controllers do on our task.

Usage: $PY crosscheck_evolution.py [--run ../evolution/runs/bench_jets-j1] [--out runs/crosscheck.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import adapter  # noqa: E402
import sim_ext  # noqa: E402
from flightsim_path import orig_sim  # noqa: E402

S = orig_sim()
EVO = os.environ.get("EVOLUTION_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evolution"))
OURS = {"T38": "t38", "737": "b737", "c172x": "c172x"}


def their_cost(r, sc, prof):
    """Evolution Runner's cost formula (evolution/sim.py) recomputed from our telemetry."""
    if r["status"] != "ok":
        return float("nan")
    tel = r["telemetry"]
    n = int(round(sc.duration_s / S.DT))
    t = tel["t"]
    tstep = np.array([sc.target(x)[1] for x in t])
    err = np.abs(tel["target"] - tel["h"])
    track = float(np.sum(err / prof["alt_err_scale_ft"] * np.minimum(t - tstep, prof.get("itae_cap_s", 30.0))
                         / prof.get("itae_t0_s", 10.0))) / n
    return track + prof.get("w_effort", 2.0) * r["effort"]


def their_scenarios(prof, n, seed):
    import dataclasses
    out = []
    for s in S.make_scenarios(n, seed):
        d = {f.name: getattr(s, f.name) for f in dataclasses.fields(s)}
        d.update(h0_ft=prof["h0_ft"], speed_kts=prof["speed_kts"],
                 steps=[(float(a), float(prof["h0_ft"] + b)) for a, b in prof["steps_rel_ft"]])
        out.append(sim_ext.ExtScenario(**d, aircraft=prof["_model"], pitch_cmd_limits_deg=tuple(prof.get("pitch_cmd_limits_deg", (-8, 12))),
                                       min_kcas=prof["min_kcas"], nz_limits=tuple(prof["nz_limits"]),
                                       throttle_max=prof.get("throttle_max")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "crosscheck_evolution.json"))
    a = ap.parse_args(argv)
    runs = a.run or [os.path.join(EVO, "runs", "bench_jets-j1"), os.path.join(EVO, "runs", "bench_jets_step200-j1")]
    report = {}
    for run in runs:
        cfg = json.load(open(os.path.join(run, "config.json")))
        cfg = cfg.get("resolved", cfg)
        summ = json.load(open(os.path.join(run, "summary.json")))
        n_sc, seed = cfg.get("scenarios", 3), cfg.get("scenario_seed") or cfg.get("seed", 1)
        for ac in summ["aircraft"]:
            model = ac["aircraft"]
            if model not in ("T38", "737"):
                continue
            prof = dict(cfg["profiles"][ac["profile"]], _model=model)
            g = ac["best_gains"]
            # (1) their task in our sim
            scs = their_scenarios(prof, n_sc, seed)
            res = [sim_ext.simulate(g, sc, record=True) for sc in scs]
            ours_their = float(np.mean([their_cost(r, sc, prof) for r, sc in zip(res, scs)]))
            # (2) our Phase-1 fitness
            task = adapter.load_task("phase1_default", {"aircraft": OURS[model]})
            ev = task.evaluate(g, task.make_scenarios(n_sc, seed), record=True)
            d0 = ev["diagnostics"]
            key = f"{os.path.basename(run)}/{model}"
            report[key] = {
                "their_best_cost": ac["best_fitness"],
                "their_task_cost_in_our_sim": ours_their,
                "rel_diff": abs(ours_their - ac["best_fitness"]) / ac["best_fitness"],
                "statuses_their_task_our_sim": [r["status"] for r in res],
                "our_phase1_cost": ev["cost"], "our_phase1_objectives": ev["objectives"],
                "our_phase1_statuses": [p["status"] for p in ev["per_scenario"]],
                "our_phase1_calm": {k: d0[0].get(k) for k in ("max_pitch_deg", "max_climb_fpm", "min_nz", "max_nz", "max_overshoot_ft")} if d0 else None,
                "gains": g,
                "in_our_ranges": {gn.name: (gn.min <= g[gn.name] <= gn.max) for gn in task.spec.genes},
            }
            r = report[key]
            print(f"{key}: their cost {r['their_best_cost']:.5f} | same task in our sim {ours_their:.5f} (rel diff {r['rel_diff']:.1e}) "
                  f"| our phase1 cost {ev['cost']:.4f} {r['our_phase1_statuses']} track {ev['objectives'].get('track_alt', float('nan')):.4f} "
                  f"| outside our ranges: {[k for k, v in r['in_our_ranges'].items() if not v] or 'none'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
