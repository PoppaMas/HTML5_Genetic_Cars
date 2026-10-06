"""Rigid vs flexible demo maneuvers (c172x, 737, T38) + composition with the repo's altitude-hold fitness.

Writes flex_demo_<model>.png, flex_demo_summary.json.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coupled_sim as cs  # noqa: E402
import flexwing as fw  # noqa: E402

CASES = [("rigid (one-way loads)", "oneway", {}), ("flex baseline (two-way)", "twoway", {}),
         ("flex soft EI,GJ x0.4 (two-way)", "twoway", dict(ei_scale=0.4, gj_scale=0.4))]


def demo(model):
    runs = []
    for label, mode, ov in CASES:
        t0 = time.perf_counter()
        r = cs.run_maneuver(model, mode, ov)
        r["wall_s"] = time.perf_counter() - t0
        runs.append((label, r))
    # determinism: repeat the baseline two-way run
    rep = cs.run_maneuver(model, "twoway", {})
    det = all(np.array_equal(rep[k], runs[1][1][k]) for k in ("alt_ft", "p_dps", "nz", "tip_w_ft_R", "tip_twist_deg_R", "root_bm_lbft_R"))
    fig, ax = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    panels = [("tip_w_ft_R", "right tip deflection [ft]"), ("tip_twist_deg_R", "right tip elastic twist [deg]"),
              ("root_bm_lbft_R", "right root bending moment [lbf ft]"), ("root_bm_lbft_L", "left root bending moment [lbf ft]"),
              ("nz", "Nz [g]"), ("p_dps", "roll rate p [deg/s]"), ("dL_lbf", "elastic lift increment fed back [lbf]"),
              ("dRoll_lbft", "elastic roll moment increment [lbf ft]")]
    for a, (k, title) in zip(ax.flat, panels):
        for label, r in runs:
            a.plot(r["t"], r[k], label=label, lw=1.2)
        a.set_title(title, fontsize=10); a.grid(alpha=0.3)
    for a in ax[-1]:
        a.set_xlabel("time [s]")
    ax[0, 0].legend(fontsize=8)
    md = cs.MANEUVER_DEFAULTS[model]
    fig.suptitle(f"{model}: elevator pull 1-2.5 s, aileron doublet 4-6 s, 1-cos gust {abs(md['gust_fps'])} ft/s at 8-9.5 s "
                 f"({md['kcas']:.0f} KCAS, {md['h_ft']:.0f} ft). 'rigid' feedback off; dL etc. shown = what would be fed back.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(HERE, f"flex_demo_{model}.png")
    fig.savefig(png, dpi=110); plt.close(fig)
    summ = {"png": png, "deterministic_repeat": bool(det), "cases": {}}
    for label, r in runs:
        tt = r["t"]
        summ["cases"][label] = {
            "wall_s": round(r["wall_s"], 3), "rtf": round(tt[-1] / r["wall_s"], 1),
            "m_root_1g_lbft": round(float(r["m_root_1g"]), 1),
            "root_bm_peak_lbft": round(float(np.max(np.abs(np.r_[r["root_bm_lbft_R"], r["root_bm_lbft_L"]]))), 1),
            "tip_w_max_ft": round(float(np.max(np.abs(r["tip_w_ft_R"]))), 3),
            "tip_twist_range_deg": [round(float(r["tip_twist_deg_R"].min()), 3), round(float(r["tip_twist_deg_R"].max()), 3)],
            "nz_max": round(float(r["nz"].max()), 3), "p_max_dps": round(float(r["p_dps"].max()), 2),
            "p_min_dps": round(float(r["p_dps"].min()), 2),
            "alt_end_ft": round(float(r["alt_ft"][-1]), 1),
            "margins": {k: (round(float(v), 3) if np.isscalar(v) else [round(x, 2) for x in v]) for k, v in r["margins"].items()},
        }
    return summ


def ga_composition():
    """Repo best gains on the repo's 3 scenarios: rigid (repo sim.simulate unpatched) vs flex genomes."""
    base = cs._import_base_sim()
    best = json.load(open(os.path.join(cs.REPO_FLIGHT_SIM, "results", "example", "best_gains.json")))
    gains = best["gains"]
    scen = base.make_scenarios(3, 1)
    out = {"gains_from": "flight_sim/results/example/best_gains.json"}
    t0 = time.perf_counter()
    rigid = base.evaluate(gains, scen)
    out["rigid_repo_cost"] = round(rigid["cost"], 5)
    out["rigid_wall_s"] = round(time.perf_counter() - t0, 2)
    enc = lambda d: [  # noqa: E731
        (np.log(d[g.name] / g.min) / np.log(g.max / g.min)) if g.kind == "log" else (d[g.name] - g.min) / (g.max - g.min)
        for g in fw.STRUCT_SCHEMA]
    genomes = {
        # x_ea / x_cg / tip mass fixed per aircraft (Phase 1); stiffness tied: EI x = s, GJ x = s * ratio
        "baseline": dict(stiffness_scale=1.0, torsion_bend_ratio=1.0, zeta=0.02, nonstruct_scale=1.0),
        "soft_light": dict(stiffness_scale=0.6, torsion_bend_ratio=1.0, zeta=0.02, nonstruct_scale=1.0),
        "stiff_heavy": dict(stiffness_scale=2.0, torsion_bend_ratio=1.0, zeta=0.02, nonstruct_scale=1.0),
        "soft_torsion_corner": dict(stiffness_scale=0.8, torsion_bend_ratio=0.8, zeta=0.005, nonstruct_scale=1.0),
    }
    for name, d in genomes.items():
        t0 = time.perf_counter()
        r = cs.evaluate_flex(gains, enc(d), scen)
        wall = time.perf_counter() - t0
        pre = r["pre"]
        out[name] = {"genes": d, "cost_total": round(r["cost"], 5), "status": r["status"], "wall_s": round(wall, 2),
                     "margins": {k: round(float(pre["margins"][k]), 3) for k in ("flutter_margin", "div_margin", "coalescence_margin",
                                                                                   "flutter_not_found_below_cap", "div_not_found_below_cap")},
                     "pre_terms": {k: round(v, 5) for k, v in pre["terms"].items()},
                     "per_scenario": [{k: (round(v, 5) if isinstance(v, float) else ({kk: round(vv, 5) for kk, vv in v.items()} if isinstance(v, dict) else v))
                                       for k, v in e.items()} for e in r["per_scenario"]]}
    return out


if __name__ == "__main__":
    summary = {m: demo(m) for m in ("c172x", "737", "T38")}
    summary["ga_composition_c172x"] = ga_composition()
    json.dump(summary, open(os.path.join(HERE, "flex_demo_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))
