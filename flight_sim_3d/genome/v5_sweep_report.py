#!/usr/bin/env python3
"""hold_osc weight sweep (c172x seed 1, pop 48 x 40): re-fly each best genome under its own task, under phase1_v4
(the v4 yardstick, 3 scenarios) and report hold quality, downdraft residual and I-gains. Writes runs/v5_sweep.json."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import adapter

RUNS = [("0 (v4 run)", "hdg_after_c172x_s1", "phase1_v4"), ("0 (+downdraft only)", "v5_sweep_w0", "experiments/phase1_v5_hold_w0.json"),
        ("0.03", "v5_sweep_w003", "experiments/phase1_v5_hold_w003.json"), ("0.1", "v5_sweep_w01", "experiments/phase1_v5_hold_w01.json"),
        ("0.3", "v5_sweep_w03", "experiments/phase1_v5_hold_w03.json")]
v4 = adapter.load_task("phase1_v4"); probe = adapter.load_task("phase1_v5")
rows = []
print("| hold weight | run | own cost | v4-yardstick cost | v4 track_alt | hold_osc | hold p-p calm / worst ft | RMS h_dot hold fps | downdraft residual / max ft | ki_alt | ki_pitch | max pitch | nz range | at bounds |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for w, run, task in RUNS:
    f = os.path.join(HERE, "runs", run, "best_gains.json")
    if not os.path.exists(f):
        continue
    bg = json.load(open(f)); g = bg["gains"]
    own = adapter.load_task(task); assert own.evaluate(g, own.make_scenarios(3, 1))["cost"] == bg["best_cost"]
    r4 = v4.evaluate(g, v4.make_scenarios(3, 1))
    r5 = probe.evaluate(g, probe.make_scenarios(3, 1), record=True); d = r5["diagnostics"]
    ab = [e["gene"] + "@" + e["bound"] for e in json.load(open(os.path.join(HERE, "runs", run, "at_bounds.json")))["best"]]
    row = {"w": w, "run": run, "own": bg["best_cost"], "v4": r4["cost"], "v4_track": r4["objectives"]["track_alt"],
           "hold_osc": r5["objectives"]["hold_osc"], "pp_calm": d[0]["hold_pp_ft"], "pp_worst": max(x["hold_pp_ft"] for x in d[:3]),
           "hdot": float(np.mean([x["hold_rms_hdot_fps"] for x in d[:3]])), "resid": d[3]["draft_residual_ft"], "dmax": d[3]["draft_max_err_ft"],
           "ki_alt": g["ki_alt"], "ki_pitch": g["ki_pitch"], "pitch": max(x["max_pitch_deg"] for x in d),
           "nz": [min(x["min_nz"] for x in d), max(x["max_nz"] for x in d)], "at_bounds": ab}
    rows.append(row)
    print(f"| {w} | {run} | {row['own']:.4f} | {row['v4']:.4f} | {row['v4_track']:.4f} | {row['hold_osc']:.3f} | {row['pp_calm']:.1f} / {row['pp_worst']:.1f} | "
          f"{row['hdot']:.2f} | {row['resid']:+.2f} / {row['dmax']:.1f} | {g['ki_alt']:.3g} | {g['ki_pitch']:.3g} | {row['pitch']:.1f} | "
          f"{row['nz'][0]:.2f}-{row['nz'][1]:.2f} | {', '.join(ab) or 'none'} |")
json.dump(rows, open(os.path.join(HERE, "runs", "v5_sweep.json"), "w"), indent=2)
