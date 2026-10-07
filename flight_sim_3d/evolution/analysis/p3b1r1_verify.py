"""P3-B1 r1 verify (Evolution side), run with EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1 (frozen FD B1 r1).
(a) Genome's c172x tip-verify pair (genome/runs/p3a1_tip_verify_c172x.json) at full_a1_b1 with the baseline shape
    (None / identity [0,1] vector) must still equal A1 bit for bit (Genome/ER reference numbers + in-process full_a1 on
    all 24 terms, margins, mass); model_version must be the r1 pin.
(b) Re-fly on r1: our r0 shaped cases (analysis/p3b1_verify_evolution.json b_shaped) + Genome's verify shapes
    (genome/runs/p3b1_verify_c172x.json cases c_*; r0 numbers from Genome's r0 copy p3b1r0_verify_c172x.json, since
    Genome rewrote p3b1_verify_c172x.json for r1 at ~19:00 PT); report r1 numbers, r1 - r0 deltas (incl. per term) and
    whether our r1 equals Genome's r1 bit for bit.
(c) FD's geometry gate on r1 over the in-range box (64 corners + 2000 random points x 4 aircraft).
Method = the A1 tip-verify re-fly: phase2_smoke_p25 c172x profile, 3 scenarios, scenario_seed 1.
Run from the team root:
  EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1 $PY -B evolution/analysis/p3b1r1_verify.py -> analysis/p3b1r1_verify_evolution.json"""
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)
sys.path.insert(0, HERE)

from evolution import batch, fidelity as F, sim  # noqa: E402
from p3b1_verify import SHAPES, run  # noqa: E402

KEYS = ("status", "cost", "fd_struct_sum", "J_mass", "J_wing_tip_bm_limit", "mass_total_frac")


def slim(e):
    m = e.get("margins") or {}
    return {**{k: e.get(k) for k in KEYS}, "per_scenario_cost": e.get("per_scenario_cost"),
            "flutter_margin": m.get("flutter_margin"), "div_margin": m.get("div_margin"),
            "reversal_margin": m.get("reversal_margin"), "model_version": e.get("model_version"), "terms": e.get("terms")}


def delta(r1, r0):
    out = {}
    for k in ("cost", "fd_struct_sum", "J_wing_tip_bm_limit", "flutter_margin", "div_margin", "reversal_margin"):
        a, b = r1.get(k), r0.get(k)
        out[k] = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    return out


def main():
    cfg = batch.resolve_config(json.load(open(os.path.join(TEAM, "evolution", "configs", "phase2_smoke_p25.json"))), "tv")
    pd = next(a for a in cfg["aircraft"] if a["name"] == "c172x")["resolved_profile"]
    prof = sim.Profile.from_dict(pd)
    scs = [s.to_dict() for s in sim.make_scenarios(cfg["scenarios"], cfg["scenario_seed"], prof)]
    gref = json.load(open(os.path.join(TEAM, "genome", "runs", "p3a1_tip_verify_c172x.json")))
    gb1 = json.load(open(os.path.join(TEAM, "genome", "runs", "p3b1_verify_c172x.json")))        # Genome, now r1 (~19:00 PT)
    g0p = os.path.join(TEAM, "genome", "runs", "p3b1r0_verify_c172x.json")                       # Genome's r0 copy
    gb0 = json.load(open(g0p)) if os.path.exists(g0p) else {"cases": {}}
    er0 = json.load(open(os.path.join(HERE, "p3b1_verify_evolution.json")))
    pins = json.load(open(os.path.join(F.FD_DIR, "v2_results", "model_versions_post_p3b1r1.json")))["c172x"]
    u0 = [g.encode(g.default) for g in F.shape_schema()]
    out = {"fd_dir": os.path.relpath(F.FD_DIR, TEAM), "model_version_expected": pins["full_a1_b1"],
           "method": "evolution.fidelity.evaluate_genome(phase2_smoke_p25 c172x profile, 3 scenarios, scenario_seed 1)",
           "created": time.strftime("%Y-%m-%d %H:%M %Z"), "loadavg_start": os.getloadavg(),
           "a_reproduce_a1": {}, "b_shaped_r1_vs_r0": {}, "c_geometry_gate": {}}
    # (a)
    for case, g in gref["cases"].items():
        exp = gref["er_a1_expected"][case]
        a1 = run(pd, scs, g, "full_a1")
        res = {"expected": {"cost": exp["cost"], "J_wing_tip_bm_limit": exp["J_wing_tip_bm_limit"]},
               "full_a1_inprocess": {"cost": a1["cost"], "model_version": a1["model_version"]}}
        for enc, shp in (("None", None), ("identity_vector", u0)):
            b = run(pd, scs, g, "full_a1_b1", shp)
            m = {"model_version_r1_pin": b["model_version"] == pins["full_a1_b1"],
                 "cost": b["cost"] == exp["cost"], "per_scenario": b["per_scenario_cost"] == exp["per_scenario_cost"],
                 "tip": b["J_wing_tip_bm_limit"] == exp["J_wing_tip_bm_limit"],
                 "inprocess_a1_24_terms": b["terms"] == a1["terms"] and len(b["terms"]) == 24,
                 "inprocess_a1_margins_mass": b["margins"] == a1["margins"] and b["mass_lb"] == a1["mass_lb"]}
            res[enc] = {"cost": b["cost"], "J_wing_tip_bm_limit": b["J_wing_tip_bm_limit"], "model_version": b["model_version"],
                        "match": m, "bit_identical": all(m.values()), "cpu_s_per_scenario": b["cpu_s_per_scenario"]}
        out["a_reproduce_a1"][case] = res
        print(case, {k: (v["cost"], v["J_wing_tip_bm_limit"], v["bit_identical"]) for k, v in res.items() if "bit_identical" in v})
    out["a_all_bit_identical"] = all(v["bit_identical"] for c in out["a_reproduce_a1"].values() for v in c.values()
                                     if isinstance(v, dict) and "bit_identical" in v)
    # (b) ours (baseline genome of the tip pair)
    g = gref["cases"]["baseline"]
    cases = [("er:baseline_shape", g, None, er0["b_shaped"]["baseline_shape"])]
    cases += [(f"er:{nm}", g, shp, er0["b_shaped"][nm]) for nm, shp in SHAPES.items()]
    genome_r1 = {}
    for nm, c in gb1["cases"].items():
        if nm.startswith("c_"):
            genome_r1[f"genome:{nm}"] = c
            r0 = gb0["cases"].get(nm) or {}            # r0 numbers from Genome's r0 copy ({} for r1-only cases)
            cases.append((f"genome:{nm}", {"gains": c["gains"], "struct": c["struct"]}, c["shape_input"], r0))
    for nm, gg, shp, r0 in cases:
        e = slim(run(pd, scs, gg, "full_a1_b1", shp))
        r0s = {k: r0.get(k) for k in ("cost", "fd_struct_sum", "J_wing_tip_bm_limit", "status", "model_version", "terms")}
        for k in ("flutter_margin", "div_margin", "reversal_margin"):
            r0s[k] = r0.get(k) if k in r0 else (r0.get("margins") or {}).get(k)
        td = {t: e["terms"][t] - r0s["terms"][t] for t in e["terms"]
              if r0s.get("terms") and t in r0s["terms"] and e["terms"][t] != r0s["terms"][t]}
        gr1 = genome_r1.get(nm)
        out["b_shaped_r1_vs_r0"][nm] = {"shape": shp, "r1": e, "r0": r0s, "delta_r1_minus_r0": delta(e, r0s),
                                        "terms_changed_r1_minus_r0": td,
                                        **({"genome_r1_cost": gr1["cost"], "equals_genome_r1_bitwise": gr1["cost"] == e["cost"]}
                                           if gr1 else {}),
                                        "r0_equals_r1_bitwise": (e["cost"] == r0s["cost"]) if r0s["cost"] is not None else None}
        print(f"{nm:28s} r1 {e['cost']:.6f} r0 {r0s['cost'] if r0s['cost'] is None else round(r0s['cost'], 6)} struct {e['fd_struct_sum']:.5f} "
              f"tip {e['J_wing_tip_bm_limit']:.5f} flut {e['flutter_margin']}")
    # (c)
    fb, fw, pb1 = F.fd_modules()["fb"], F.fd_modules()["fw"], F.fd_b1_modules()["pb1"]
    pts = [list(u) for u in itertools.product([0.0, 1.0], repeat=6)] + [list(u) for u in np.random.default_rng(0).random((2000, 6))]
    for ac in ("c172x", "T38", "737", "f16"):
        geom = fb.geometry_for(ac, F.roots(sim.Profile.from_dict(dict(pd, aircraft=ac)))[1])
        pw = fw.params_for(ac, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
        rs = [pb1.geometry_gate(pw, u, n_el=64) for u in pts]
        tp = [r.details["taper_tip_root"] for r in rs]
        out["c_geometry_gate"][ac] = {"n_points": len(pts), "n_fail": sum(not r.ok for r in rs),
                                      "taper_tip_root_min_max": [min(tp), max(tp)]}
    out["loadavg_end"] = os.getloadavg()
    with open(os.path.join(HERE, "p3b1r1_verify_evolution.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("A1 bit for bit on r1:", out["a_all_bit_identical"], "gate fails:", {k: v["n_fail"] for k, v in out["c_geometry_gate"].items()})


if __name__ == "__main__":
    main()
