"""P3-B1 verify (Evolution side): FD's full_a1_b1 through evolution.fidelity.
(a) Genome's c172x tip-verify pair (genome/runs/p3a1_tip_verify_c172x.json; ours evolution/analysis/p3a1_tip_verify_evolution.json)
    at full_a1_b1 with the baseline shape (None / {} / FD defaults / identity [0,1] vector) must reproduce A1 bit for bit
    (and equal an in-process full_a1 re-fly on all 24 terms, margins, mass).
(b) The baseline genome with hand-picked non-baseline shapes (tip taper 0.85, tip twist -3 deg, sweep +5 deg, sweep -5 deg,
    FD's benchmark shape): cost / 24 terms / mass / margins / planform.
(c) Geometry gate: scan of the whole in-range gene box (64 corners + 2000 random points x 4 aircraft) with FD's own gate;
    if nothing fails in range, the reject path is demonstrated with an INJECTED stricter taper limit (labelled as such).
Method = the A1 tip-verify re-fly: phase2_smoke_p25 c172x profile, 3 scenarios, scenario_seed 1.
Run from the team root:  $PY evolution/analysis/p3b1_verify.py   -> evolution/analysis/p3b1_verify_evolution.json"""
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)

from evolution import batch, fidelity as F, sim  # noqa: E402

RIGID = ("track", "effort", "comfort", "heading", "hold")
SHAPES = {
    "tip_taper_0.85": {"wing_chord_taper_3": 0.85},
    "tip_twist_-3deg": {"wing_twist_tip_deg": -3.0},
    "sweep_+5deg": {"wing_sweep_qc_delta_deg": 5.0},
    "sweep_-5deg": {"wing_sweep_qc_delta_deg": -5.0},
    "fd_bench_shape": {"wing_chord_taper_1": 0.95, "wing_chord_taper_2": 0.95, "wing_chord_taper_3": 0.95,
                       "wing_twist_mid_deg": -0.5, "wing_twist_tip_deg": -2.0, "wing_sweep_qc_delta_deg": 2.0},
}


def run(pd, scs, g, fid, shape=None):
    c0, w0 = time.process_time(), time.perf_counter()
    r = F.evaluate_genome(pd, g["gains"], g["struct"], scs, fid, 0.9, shape=shape)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    e = {"model_version": r["model_version"], "status": r["status"], "cost": r["cost"],
         "per_scenario_cost": [p["cost"] for p in r["per_scenario"]], "terms": r["terms"],
         "fd_struct_sum": float(sum(v for k, v in r["terms"].items() if k not in RIGID)),
         "J_mass": r["terms"]["J_mass"], "J_wing_tip_bm_limit": r["terms"]["J_wing_tip_bm_limit"],
         "margins": r["margins"], "mass_total_frac": r["mass_total_frac"], "mass_lb": r["mass_lb"],
         "cpu_s": cpu, "wall_s": wall, "cpu_s_per_scenario": cpu / len(scs)}
    for k in ("shape_genes", "shape_cache_key", "geometry_gate", "planform", "geometry_gate_reject"):
        if k in r:
            e[k] = r[k]
    return e


def main():
    cfg = batch.resolve_config(json.load(open(os.path.join(TEAM, "evolution", "configs", "phase2_smoke_p25.json"))), "tv")
    pd = next(a for a in cfg["aircraft"] if a["name"] == "c172x")["resolved_profile"]
    prof = sim.Profile.from_dict(pd)
    scs = [s.to_dict() for s in sim.make_scenarios(cfg["scenarios"], cfg["scenario_seed"], prof)]
    gref = json.load(open(os.path.join(TEAM, "genome", "runs", "p3a1_tip_verify_c172x.json")))
    eref = json.load(open(os.path.join(HERE, "p3a1_tip_verify_evolution.json")))["cases"]
    pins = json.load(open(os.path.join(F.FD_DIR, "v2_results", "model_versions_post_p3b1.json")))["c172x"]
    pb1 = F.fd_b1_modules()["pb1"]
    u0 = [g.encode(g.default) for g in F.shape_schema()]
    out = {"method": "evolution.fidelity.evaluate_genome(phase2_smoke_p25 c172x profile, 3 scenarios, scenario_seed 1), "
                     "genomes from genome/runs/p3a1_tip_verify_c172x.json (cases baseline / soft_tip_taper4_0.75)",
           "model_version_expected": {"full_a1": pins["full_a1"], "full_a1_b1": pins["full_a1_b1"]},
           "created": time.strftime("%Y-%m-%d %H:%M %Z"), "loadavg_start": os.getloadavg(),
           "a_reproduce_a1": {}, "b_shaped": {}, "c_geometry_gate": {}}
    # (a)
    for case, g in gref["cases"].items():
        exp = gref["er_a1_expected"][case]
        a1 = run(pd, scs, g, "full_a1")
        res = {"full_a1_inprocess": {k: a1[k] for k in ("model_version", "cost", "per_scenario_cost", "J_wing_tip_bm_limit",
                                                        "cpu_s_per_scenario")}}
        for enc, shp in (("None", None), ("empty_dict", {}), ("fd_defaults", pb1.shape_defaults()), ("identity_vector", u0)):
            b = run(pd, scs, g, "full_a1_b1", shp)
            res[enc] = {"model_version": b["model_version"], "model_version_ok": b["model_version"] == pins["full_a1_b1"],
                        "cost": b["cost"], "per_scenario_cost": b["per_scenario_cost"],
                        "J_wing_tip_bm_limit": b["J_wing_tip_bm_limit"], "status": b["status"],
                        "match_genome_ref_cost": b["cost"] == exp["cost"],
                        "match_genome_ref_per_scenario": b["per_scenario_cost"] == exp["per_scenario_cost"],
                        "match_genome_ref_tip": b["J_wing_tip_bm_limit"] == exp["J_wing_tip_bm_limit"],
                        "match_er_a1_json_cost": b["cost"] == eref[case]["full_a1"]["cost"],
                        "match_inprocess_a1_all_24_terms": b["terms"] == a1["terms"] and len(b["terms"]) == 24,
                        "match_inprocess_a1_margins_mass": b["margins"] == a1["margins"] and b["mass_lb"] == a1["mass_lb"],
                        "cpu_s_per_scenario": b["cpu_s_per_scenario"]}
            res[enc]["bit_identical"] = all(v for k, v in res[enc].items() if k.startswith("match_"))
        res["expected"] = {"cost": exp["cost"], "J_wing_tip_bm_limit": exp["J_wing_tip_bm_limit"]}
        out["a_reproduce_a1"][case] = res
        print(case, exp["cost"], {k: (v["cost"], v["bit_identical"]) for k, v in res.items() if isinstance(v, dict) and "bit_identical" in v})
    out["a_all_bit_identical"] = all(v["bit_identical"] for c in out["a_reproduce_a1"].values() for v in c.values()
                                     if isinstance(v, dict) and "bit_identical" in v)
    # (b)
    g = gref["cases"]["baseline"]
    base = run(pd, scs, g, "full_a1_b1", None)
    out["b_shaped"]["baseline_shape"] = base
    for nm, shp in SHAPES.items():
        e = run(pd, scs, g, "full_a1_b1", shp)
        e["delta_vs_baseline_shape"] = {"cost": e["cost"] - base["cost"], "fd_struct_sum": e["fd_struct_sum"] - base["fd_struct_sum"],
                                        "flutter_margin": (e["margins"]["flutter_margin"] - base["margins"]["flutter_margin"])
                                        if e["margins"].get("flutter_margin") is not None else None}
        out["b_shaped"][nm] = e
        print(nm, e["status"], e["cost"], e["fd_struct_sum"], e["margins"], e["mass_total_frac"])
    # (c)
    fb, fw = F.fd_modules()["fb"], F.fd_modules()["fw"]
    scan = {}
    pts = [list(u) for u in itertools.product([0.0, 1.0], repeat=6)] + [list(u) for u in np.random.default_rng(0).random((2000, 6))]
    for ac in ("c172x", "T38", "737", "f16"):
        geom = fb.geometry_for(ac, F.roots(sim.Profile.from_dict(dict(pd, aircraft=ac)))[1])
        pw = fw.params_for(ac, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
        rs = [pb1.geometry_gate(pw, u, n_el=64) for u in pts]
        tp = [r.details["taper_tip_root"] for r in rs]
        scan[ac] = {"n_points": len(pts), "n_fail": sum(not r.ok for r in rs),
                    "taper_tip_root_min_max": [min(tp), max(tp)], "gate_taper_limits": [pb1.GATE_MIN_TAPER, pb1.GATE_MAX_TAPER],
                    "le_kink_max_deg": max(r.details["le_kink_max_deg"] for r in rs), "gate_le_kink_max_deg": pb1.GATE_MAX_LE_KINK_DEG,
                    "c_min_ft": min(r.details["c_min_ft"] for r in rs), "gate_min_chord_ft": pb1.GATE_MIN_CHORD_FT,
                    "abs_sweep_max_deg": max(abs(r.details["sweep_qc_deg"]) for r in rs), "gate_max_abs_sweep_deg": pb1.GATE_MAX_ABS_SWEEP_DEG}
    out["c_geometry_gate"]["in_range_scan"] = scan
    out["c_geometry_gate"]["in_range_failures_exist"] = any(v["n_fail"] for v in scan.values())
    # injected reject (stricter in-process taper limit wrapped around FD's own gate; FD files / version strings unchanged)
    orig = pb1.geometry_gate_strips
    flown = {"n": 0}
    fe = F.fd_modules()["fe"]
    orig_sim = fe._simulate_with_hook

    def strict(pf):
        r = orig(pf)
        if r.ok and r.details["taper_tip_root"] < 0.6:
            return pb1.GeometryGateResult(False, "extreme_taper", r.details)
        return r

    def counting(*a, **k):
        flown["n"] += 1
        return orig_sim(*a, **k)
    pb1.geometry_gate_strips, fe._simulate_with_hook = strict, counting
    try:
        shp = {"wing_chord_taper_1": 0.85, "wing_chord_taper_2": 0.85, "wing_chord_taper_3": 0.85}
        e = run(pd, scs, g, "full_a1_b1", shp)
    finally:
        pb1.geometry_gate_strips, fe._simulate_with_hook = orig, orig_sim
    out["c_geometry_gate"]["injected_reject_demo"] = {
        "note": "INJECTED: no in-range shape fails FD's gate, so the taper limit was raised in-process to 0.6 (c172x baseline "
                "0.75) to exercise the reject path; FD's real limit is [0.12, 1.25]",
        "shape": shp, "status": e["status"], "cost": e["cost"], "per_scenario_cost": e["per_scenario_cost"],
        "geometry_gate": e.get("geometry_gate"), "jsbsim_flights": flown["n"], "cpu_s": e["cpu_s"],
        "terms_all_zero": all(v == 0.0 for v in e["terms"].values())}
    try:
        F.shape_from({"wing_chord_taper_3": 0.80})
        out["c_geometry_gate"]["out_of_range_decode"] = "accepted (unexpected)"
    except ValueError as ex:
        out["c_geometry_gate"]["out_of_range_decode"] = f"ValueError (FD never clips): {ex}"
    out["loadavg_end"] = os.getloadavg()
    with open(os.path.join(HERE, "p3b1_verify_evolution.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("A1 reproduced bit for bit:", out["a_all_bit_identical"], " in-range gate failures:",
          out["c_geometry_gate"]["in_range_failures_exist"], " injected:", out["c_geometry_gate"]["injected_reject_demo"]["status"],
          out["c_geometry_gate"]["injected_reject_demo"]["cost"], "flights", flown["n"])


if __name__ == "__main__":
    main()
