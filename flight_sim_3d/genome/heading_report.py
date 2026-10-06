#!/usr/bin/env python3
"""Before/after table for the Phase-1 heading hold (re-flies each run's best genome with record=True).

    $PY heading_report.py [--out runs/heading_report.json]

Each entry names the run directory and the task it was evolved on. "before" runs are phase1_default with the
heading hold off (identical to the v4 task; tests/test_heading.py checks bit-identity), "after" runs use the
shared default (heading hold on). Heading drift = psi(t) - psi(0), wrapped to +-180 deg; the c172x and jets start
at psi = 0 with heading target 0.

Seeds: evolve.py sets scenario_seed = GA seed unless --scenario-seed is given, so the c172x s2/s3 before/after pairs
share their (seed-specific) scenario sets. The T38 s2/s3 after-runs use --scenario-seed 1 to match Evolution Runner's
Phase-1 s2/s3 (0.0938 / 0.0928, same task bit-for-bit), which serve as their "before".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapter  # noqa: E402

RUNS = [  # (label, run dir, preset, overrides)
    ("c172x before s1", "phase1_default_c172x_v4", "phase1_default", {"heading_hold": False}),
    ("c172x before s2", "hdg_before_c172x_s2", "phase1_default", {"heading_hold": False}),
    ("c172x before s3", "hdg_before_c172x_s3", "phase1_default", {"heading_hold": False}),
    ("c172x after s1", "hdg_after_c172x_s1", "phase1_default", {}),
    ("c172x after s2", "hdg_after_c172x_s2", "phase1_default", {}),
    ("c172x after s3", "hdg_after_c172x_s3", "phase1_default", {}),
    ("c172x w=0 s1", "hdg_sweep_w0", "experiments/phase1_hdg_w0.json", {}),
    ("c172x w=0.003 s1", "hdg_sweep_w003", "experiments/phase1_hdg_w003.json", {}),
    ("c172x w=0.03 s1", "hdg_sweep_w03", "experiments/phase1_hdg_w03.json", {}),
    ("c172x w=0.1 s1", "hdg_sweep_w1", "experiments/phase1_hdg_w1.json", {}),
    ("T38 before s1", "phase1_t38_v4", "phase1_default", {"aircraft": "t38", "heading_hold": False}),
    ("T38 after s1", "hdg_after_t38_s1", "phase1_default", {"aircraft": "t38"}),
    ("T38 after s2 (scen. seed 1)", "hdg_after_t38_s2", "phase1_default", {"aircraft": "t38"}),
    ("T38 after s3 (scen. seed 1)", "hdg_after_t38_s3", "phase1_default", {"aircraft": "t38"}),
    ("737 before s1", "phase1_b737_v4", "phase1_default", {"aircraft": "b737", "heading_hold": False}),
    ("737 after s1", "hdg_after_b737_s1", "phase1_default", {"aircraft": "b737"}),
]


def row(label, run, preset, ov):
    d = os.path.join(HERE, "runs", run)
    if not os.path.exists(os.path.join(d, "best_gains.json")):
        return None
    bg = json.load(open(os.path.join(d, "best_gains.json")))
    task = adapter.load_task(preset, ov)
    cfg = bg["config"]
    r = task.evaluate(bg["gains"], task.make_scenarios(cfg["scenarios"], cfg["scenario_seed"]), record=True)
    dg = r["diagnostics"]
    ab = json.load(open(os.path.join(d, "at_bounds.json")))["best"] if os.path.exists(os.path.join(d, "at_bounds.json")) else []
    return {
        "label": label, "run": run, "best_cost": bg["best_cost"], "recomputed_cost": r["cost"],
        "cost_ex_heading": r["cost"] - task.fitness.weights.get("track_heading_rms", 0.0) * r["objectives"].get("track_heading_rms", 0.0),
        "objectives": r["objectives"], "gains": bg["gains"],
        "hdg_final_deg": [x["hdg_final_deg"] for x in dg], "hdg_max_abs_deg": [x["hdg_max_abs_deg"] for x in dg],
        "max_abs_phi_deg": [x["max_abs_phi_deg"] for x in dg],
        "max_pitch_deg": [x["max_pitch_deg"] for x in dg], "nz": [[x["min_nz"], x["max_nz"]] for x in dg],
        "overshoot_ft": [x["max_overshoot_ft"] for x in dg],
        "at_bounds": [f"{e['gene']}@{e['bound']}({e['value']:.3g})" for e in ab],
    }


def existing_jets():
    """v4 jet best genomes (no heading genes) re-flown with the flag on at the default heading gains (kp_hdg scaled default, ki_hdg 0)."""
    out = []
    for ac, run in (("t38", "phase1_t38_v4"), ("b737", "phase1_b737_v4")):
        bg = json.load(open(os.path.join(HERE, "runs", run, "best_gains.json")))
        task = adapter.load_task("phase1_default", {"aircraft": ac})
        g = {**{x.name: x.default for x in task.spec.genes}, **bg["gains"]}
        scs = task.make_scenarios(3, 1)
        r = task.evaluate(g, scs, record=True)
        w = task.fitness.weights["track_heading_rms"]
        out.append({"aircraft": ac, "cost_before": bg["best_cost"], "cost_flag_on": r["cost"],
                    "cost_flag_on_ex_heading": r["cost"] - w * r["objectives"]["track_heading_rms"],
                    "track_alt_before": None, "objectives": r["objectives"], "kp_hdg": g["kp_hdg"], "ki_hdg": g["ki_hdg"],
                    "hdg_max_abs_deg": max(d["hdg_max_abs_deg"] for d in r["diagnostics"]),
                    "hdg_final_deg": [d["hdg_final_deg"] for d in r["diagnostics"]],
                    "max_abs_phi_deg": max(d["max_abs_phi_deg"] for d in r["diagnostics"])})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "heading_report.json"))
    a = ap.parse_args(argv)
    rows = [x for x in (row(*e) for e in RUNS) if x]
    jets = existing_jets()
    json.dump({"runs": rows, "existing_jet_genomes_flag_on": jets}, open(a.out, "w"), indent=2, default=float)
    for j in jets:
        print(f"existing {j['aircraft']} v4 genome, flag on (kp_hdg {j['kp_hdg']:.3g}, ki_hdg 0): cost {j['cost_before']:.4f} -> "
              f"{j['cost_flag_on']:.4f} (w/o heading term {j['cost_flag_on_ex_heading']:.4f}), track_alt {j['objectives']['track_alt']:.4f}, "
              f"hdg final {['%+.2f' % v for v in j['hdg_final_deg']]}, max |dpsi| {j['hdg_max_abs_deg']:.2f}, max |phi| {j['max_abs_phi_deg']:.1f}")
    print("| run | best cost | cost w/o heading term | track_alt | hdg final deg (calm/s1/s2) | max abs hdg deg | max pitch deg (calm/worst) | nz calm / all | kp_hdg / ki_hdg | at bounds |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        nz = x["nz"]
        print(f"| {x['label']} | {x['best_cost']:.4f} | {x['cost_ex_heading']:.4f} | {x['objectives']['track_alt']:.4f} | "
              f"{' / '.join(f'{v:+.1f}' for v in x['hdg_final_deg'])} | {max(abs(v) for v in x['hdg_max_abs_deg']):.1f} | "
              f"{x['max_pitch_deg'][0]:.1f} / {max(x['max_pitch_deg']):.1f} | "
              f"{nz[0][0]:.2f}-{nz[0][1]:.2f} / {min(n[0] for n in nz):.2f}-{max(n[1] for n in nz):.2f} | "
              f"{x['gains'].get('kp_hdg', float('nan')):.3g} / {x['gains'].get('ki_hdg', float('nan')):.3g} | {', '.join(x['at_bounds']) or 'none'} |")
        if abs(x["recomputed_cost"] - x["best_cost"]) > 1e-12:
            print(f"   WARNING {x['label']}: recomputed {x['recomputed_cost']} != {x['best_cost']}")


if __name__ == "__main__":
    main()
