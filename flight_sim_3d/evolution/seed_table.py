#!/usr/bin/env python3
"""Compare stage B (ic) vs stage C (adjusted) across seeds 1-3 for the non-c172x aircraft. Reads summary.json files."""
import glob
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def load(pattern):
    fs = sorted(glob.glob(os.path.join(R, pattern, "summary.json")))
    return {a["aircraft"]: a for a in json.load(open(fs[0]))["aircraft"]} if fs else None


runs = {("ic", 1): load("bench_ic-b1"), ("adjusted", 1): load("bench_adjusted-b1")}
for s in (2, 3):
    for st in ("ic", "adjusted"):
        runs[(st, s)] = load(f"{st}_s{s}-*")
print("| aircraft | seed | stage B (IC only) best fitness | stage C (adjusted) best fitness | change | B hold RMS calm/mean ft | C hold RMS calm/mean ft | B overshoot ft | C overshoot ft | B genes at bound | C genes at bound |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for ac in ("f16", "737", "t6texan2"):
    ch = []
    for s in (1, 2, 3):
        b, c = runs[("ic", s)], runs[("adjusted", s)]
        if not b or not c:
            print(f"| {ac} | {s} | missing | | | | | | | | |")
            continue
        b, c = b[ac], c[ac]
        def hm(a):
            m = a["final_metrics"]
            return f"{m[0]['hold_rms_err_ft']:.2f} / {np.mean([x['hold_rms_err_ft'] for x in m]):.2f}"
        def ov(a):
            return f"{max(x['overshoot_ft_max'] for x in a['final_metrics']):.1f}"
        d = (c["best_fitness"] - b["best_fitness"]) / b["best_fitness"] * 100
        ch.append(d)
        gb = lambda a: ", ".join(f"{k}@{v}" for k, v in a["genes_at_bound"].items()) or "-"
        print(f"| {ac} | {s} | {b['best_fitness']:.4f} | {c['best_fitness']:.4f} | {d:+.1f}% | {hm(b)} | {hm(c)} | {ov(b)} | {ov(c)} | {gb(b)} | {gb(c)} |")
    if ch:
        print(f"| {ac} | mean | | | {np.mean(ch):+.1f}% | | | | | | |")
