#!/usr/bin/env python3
"""Run the UNMODIFIED flight_sim/evolve.py on a genome/fitness task preset.

    python run_evolve.py --task altitude_hold_legacy -- --pop-size 24 --generations 5 --seed 1 --out /tmp/r
    python run_evolve.py --task presets/my_task.json -- --config $FLIGHT_SIM_DIR/config.example.json

Everything after ``--`` goes to evolve.py's own argument parser. After the run
this writes ``task.json``, ``fitness_detail.json`` (objective breakdown and
flight diagnostics of the best genome per scenario) and ``at_bounds.json``
into the output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import adapter  # noqa: E402
import genome_schema as GS  # noqa: E402
from flightsim_path import FLIGHT_SIM_DIR  # noqa: E402


def post_run(task, summary, out):
    import numpy as np
    spec = task.spec
    best = np.array(summary["genome"])
    scen = summary["scenarios"]
    scs = task.make_scenarios(summary["config"]["scenarios"], summary["config"]["scenario_seed"])
    detail = task.evaluate(spec.decode(best), scs, record=True)
    detail.pop("per_scenario_traces", None)
    # at-bounds: best genome, and the per-generation best genes over the last 10 generations (from the CSV)
    rows = list(csv.DictReader(open(os.path.join(out, "fitness_history.csv"))))[-10:]
    hist = np.array([[spec.genes[j].encode(float(r[f"best_{n}"])) for j, n in enumerate(spec.names)] for r in rows])
    rep = {"best": spec.at_bounds(best), "last10_generation_bests": spec.at_bounds(hist),
           "suggestions": {}}
    for e in rep["best"]:
        if e["bound"] in ("lower", "upper"):
            g = next(g for g in spec.genes if g.name == e["gene"])
            rep["suggestions"][g.name] = {"bound": e["bound"], "suggested_range": GS.suggest_widened_range(g, e["bound"])}
    json.dump(task.raw, open(os.path.join(out, "task.json"), "w"), indent=2)
    json.dump({k: v for k, v in detail.items()}, open(os.path.join(out, "fitness_detail.json"), "w"), indent=2, default=float)
    json.dump(rep, open(os.path.join(out, "at_bounds.json"), "w"), indent=2)
    pinned = [f"{e['gene']}@{e['bound']}({e['value']:.4g})" for e in rep["best"]]
    print(f"[genome] objectives (best): { {k: round(v, 4) for k, v in detail['objectives'].items()} }")
    for i, d in enumerate(detail.get("diagnostics", [])):
        print(f"[genome] scenario {i}: max pitch {d.get('max_pitch_deg', float('nan')):.1f} deg "
              f"({d.get('max_pitch_above_trim_deg', float('nan')):+.1f} vs trim), nz [{d.get('min_nz', 0):.2f}, {d.get('max_nz', 0):.2f}], "
              f"max climb {d.get('max_climb_fpm', 0):.0f} fpm, overshoot {d.get('max_overshoot_ft', float('nan')):.1f} ft, "
              f"max ref err {d.get('max_ref_err_ft', float('nan')):.1f} ft")
    if detail.get("aeroelastic"):
        print(f"[genome] aeroelastic: { {k: (round(v, 3) if isinstance(v, float) else v) for k, v in detail['aeroelastic'].items()} }")
    print(f"[genome] best genes at bounds: {pinned or 'none'}")
    if detail.get("skipped"):
        print(f"[genome] skipped objectives: {detail['skipped']}")
    return detail, rep


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="phase1_default", help="preset name in presets/ or path to a task JSON")
    ap.add_argument("--aircraft", help="override the task's aircraft profile")
    ap.add_argument("--no-post", action="store_true", help="skip the post-run detail/at-bounds report")
    a = ap.parse_args(argv)

    overrides = {"aircraft": a.aircraft} if a.aircraft else {}
    task = adapter.load_task(a.task, overrides)
    for w in task.warnings:
        print(f"[genome] WARNING: {w}")
    if task.fitness.mode == "pareto":
        print("[genome] note: pareto task under evolve.py ranks by the weighted scalar; use evolve_pareto.py for NSGA-II")
    print(f"[genome] task {task.name}: {task.spec.n_genes} genes {task.spec.names} on {task.profile.name}")

    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)  # workers inherit the shim modules
    adapter.install(task)
    sys.path.insert(0, FLIGHT_SIM_DIR)
    import evolve  # the original, unmodified file; binds to the shims
    assert evolve.genome is sys.modules["genome"] and evolve.sim is sys.modules["sim"]
    if getattr(task, "operators", None):  # P3-B1: per-block operators (block_ops.py) + seeded generation 0
        import block_ops
        evolve.ga = block_ops.block_ga(evolve.ga, task)
        print(f"[genome] generation 0: {task.init}; operators: "
              f"{ {k: v for k, v in task.operators.items() if k in ('crossover', 'mutation_sigma_u')} }")
    elif task.init:  # Phase 2: seeded generation 0 (init_pop.py); tasks without "init" use evolve.py's draw unchanged
        import init_pop
        evolve.ga = init_pop.seeded_ga(evolve.ga, task)
        print(f"[genome] generation 0: {task.init}")
    cfg = evolve.parse_args(rest)
    summary = evolve.run(cfg)
    if not a.no_post:
        post_run(task, summary, cfg["out"])
    return summary


if __name__ == "__main__":
    main()
