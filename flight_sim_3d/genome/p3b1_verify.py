#!/usr/bin/env python3
"""P3-B1 verify set on the genome side (read-only on evolution/ and flight-dynamics/; bytecode writing off).

Same setup as the A1 tip-verify (p3a1_tip_verify.py): ER phase2_smoke_p25 c172x resolved profile, 3 scenarios,
scenario_seed 1, gains + structure from runs/p25_tip_verify_c172x.json. Fidelity full_a1_b1 through
shape_b1.evaluate_b1 (default: ER evolution.fidelity.evaluate_genome(..., 'full_a1_b1', shape=...); 'fd' = the direct
FD flexeval_b1.evaluate path, cross-checked).

  a  baseline shape + baseline structure         == A1 baseline bit for bit (ER numbers + same-process A1 run)
  a2 same genome through the phase3_b1 Task       (26-gene u-vector encode -> decode -> evaluate; FD GENE_ENCODING)
  b  baseline shape + soft tip (ei_taper_4 0.75)  == A1 soft
  c  non-baseline shapes that fly                 (costs / terms recorded; r0 -> r1 deltas vs runs/p3b1r0_verify_c172x.json;
                                                   twist_mid range ends -2 / +1 are r1-only cases)
  e  ER's own B1 verify genomes (p3b1_verify_evolution.json b_shaped) re-flown on both paths vs ER's numbers
  d  reject paths: out-of-range / deferred decode raise (FD policy); FD gate reject exercised with an in-memory
     tightened bound (FD: the whole B1 box passes the gate on all 4 aircraft, so no in-range genome rejects)

  f  ER's r1 verify (evolution/analysis/p3b1r1_verify_evolution.json, b_shaped_r1_vs_r0: ER's 6 + genome's 4 shapes)
     vs the r1 numbers above, bit for bit (no extra flights)
  enc FD GENE_ENCODING (r1: linear in value) vs genome GeneSpec decode/encode, bit for bit on random + corner vectors

B1 r1 (FD 2026-10-06 ~18:33 PT): pins v2_results/model_versions_post_p3b1r1.json (r0 superseded). The r0 output of this
script is kept as runs/p3b1r0_verify_c172x.json for the deltas.

Usage: $PY genome/p3b1_verify.py   -> genome/runs/p3b1_verify_c172x.json
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("FLIGHT_SIM_DIR", os.path.join(os.path.dirname(os.path.dirname(HERE)), "flight_sim"))  # repo ../../flight_sim

import shape_b1 as SB  # noqa: E402

RIGID = SB.RIGID_TERMS
ER_A1 = {   # evolution/analysis/p3a1_tip_verify_evolution.json (= genome runs/p3a1_tip_verify_c172x.json)
    "baseline": {"cost": 0.2419400885448951,
                 "per_scenario_cost": [0.12772764189700128, 0.39710530396908106, 0.2009873197686029],
                 "J_wing_tip_bm_limit": 0.0},
    "soft_tip_taper4_0.75": {"cost": 0.25879473336084763,
                             "per_scenario_cost": [0.14271980987754557, 0.4179016740566261, 0.21576271614837125],
                             "J_wing_tip_bm_limit": 0.011874078070487388},
}
SHAPES_FLY = {
    "washout_tip_m2": {"wing_twist_tip_deg": -2.0},
    "chord_taper3_0.9": {"wing_chord_taper_3": 0.9},
    "sweep_p3": {"wing_sweep_qc_delta_deg": 3.0},
    "combo_fd_bench": {"wing_chord_taper_1": 0.95, "wing_chord_taper_2": 0.95, "wing_chord_taper_3": 0.95,
                       "wing_twist_mid_deg": -0.5, "wing_twist_tip_deg": -2.0, "wing_sweep_qc_delta_deg": 2.0},
    "twist_mid_m2": {"wing_twist_mid_deg": -2.0},     # r1-only: twist_mid range ends (r1 fixed its one-sided loophole)
    "twist_mid_p1": {"wing_twist_mid_deg": 1.0},
}
R0_FILE = os.path.join(HERE, "runs", "p3b1r0_verify_c172x.json")


def r0_delta(cur, r0):
    """r0 -> r1 delta block for one case (None when the case did not exist in r0)."""
    if not r0:
        return None
    return {"r0_cost": r0["cost"], "r0_J_wing_tip_bm_limit": r0["J_wing_tip_bm_limit"],
            "r0_model_version": r0.get("model_version"),
            "delta_cost": cur["cost"] - r0["cost"], "delta_J_wing_tip_bm_limit": cur["J_wing_tip_bm_limit"] - r0["J_wing_tip_bm_limit"],
            "identical": (cur["cost"], cur["J_wing_tip_bm_limit"]) == (r0["cost"], r0["J_wing_tip_bm_limit"])}


def encoding_check(pb1, genes, n=20000, seed=0):
    """FD decode_shape_b1 / encode_shape_b1 (GENE_ENCODING) vs genome GeneSpec decode / encode, bit for bit."""
    import numpy as np
    rng = np.random.default_rng(seed)
    U = np.vstack([rng.random((n, 6)), np.zeros(6), np.ones(6), [[g.encode(g.default) for g in genes]]])
    dec_mis = enc_mis = 0
    for u in U:
        fd = pb1.decode_shape_b1(list(u))
        gd = {g.name: g.decode(x) for g, x in zip(genes, u)}
        dec_mis += any(fd[k] != gd[k] for k in fd)
        enc_mis += not np.array_equal(pb1.encode_shape_b1(fd), np.array([g.encode(gd[g.name]) for g in genes]))
    idu = [g.encode(g.default) for g in genes]
    return {"fd_gene_encoding": pb1.GENE_ENCODING, "genome_pin": SB.B1_GENE_ENCODING_PIN,
            "pin_matches": pb1.GENE_ENCODING == SB.B1_GENE_ENCODING_PIN, "fd_b1_rev": pb1.B1_REV,
            "fd_scales": [g.scale for g in pb1.shape_schema()], "genome_scales": [g.scale for g in genes],
            "n_vectors": len(U), "decode_mismatches": int(dec_mis), "encode_mismatches": int(enc_mis),
            "identity_u": idu, "identity_decodes_to_fd_defaults": pb1.is_baseline_shape(pb1.decode_shape_b1(idu)),
            "shape_from_values_is_passthrough": True,
            "exact": dec_mis == 0 and enc_mis == 0 and pb1.GENE_ENCODING == SB.B1_GENE_ENCODING_PIN}


def summarize(r, cpu_s, n_sc):
    t = r["terms"]
    fd = {k: float(v) for k, v in t.items() if k not in RIGID}
    return {"cost": r["cost"], "per_scenario_cost": [float(e["cost"]) for e in r["per_scenario"]],
            "status": r["status"], "per_scenario_status": [e.get("status") for e in r["per_scenario"]],
            "J_wing_tip_bm_limit": float(t.get("J_wing_tip_bm_limit", 0.0)), "fd_struct_terms": fd,
            "fd_struct_sum": float(sum(fd.values())), "controller_terms": {k: float(t[k]) for k in RIGID},
            "terms": {k: float(v) for k, v in t.items()}, "margins": r.get("margins"),
            "mass_total_frac": r.get("mass_total_frac"), "model_version": r["model_version"],
            "shape_genes": r.get("shape_genes"), "shape_cache_key": r.get("shape_cache_key"),
            "geometry_gate": r.get("geometry_gate"), "planform": r.get("planform"),
            "not_flown": any(e.get("not_flown") for e in r["per_scenario"]),
            "cpu_s": cpu_s, "cpu_s_per_scenario": cpu_s / n_sc}


def timed(fn, *a, **k):
    c0 = time.process_time()
    r = fn(*a, **k)
    return r, time.process_time() - c0


def match_a1(s, ref, a1_same=None):
    m = {"vs_er_a1": {"bit_identical_total": s["cost"] == ref["cost"],
                      "bit_identical_per_scenario": s["per_scenario_cost"] == ref["per_scenario_cost"],
                      "bit_identical_tip": s["J_wing_tip_bm_limit"] == ref["J_wing_tip_bm_limit"]}}
    m["vs_er_a1"]["bit_identical"] = all(m["vs_er_a1"].values())
    if a1_same is not None:
        diff = {k: s["terms"][k] - a1_same["terms"][k] for k in a1_same["terms"] if s["terms"][k] != a1_same["terms"][k]}
        m["vs_same_process_a1"] = {"bit_identical_total": s["cost"] == a1_same["cost"],
                                   "bit_identical_per_scenario": s["per_scenario_cost"] == a1_same["per_scenario_cost"],
                                   "all_24_terms_identical": not diff, "term_diffs": diff,
                                   "margins_identical": s["margins"] == a1_same["margins"]}
        m["vs_same_process_a1"]["bit_identical"] = (m["vs_same_process_a1"]["bit_identical_total"]
                                                    and m["vs_same_process_a1"]["bit_identical_per_scenario"]
                                                    and not diff)
    if not m["vs_er_a1"]["bit_identical_total"]:
        m["vs_er_a1"]["delta_total"] = s["cost"] - ref["cost"]
    return m


def _find_mv(obj):
    """First 'full_a1_b1:...' model_version string anywhere in ER's verify json."""
    if isinstance(obj, str):
        return obj if obj.startswith("full_a1_b1:") else None
    vals = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, list) else ()
    for v in vals:
        m = _find_mv(v)
        if m:
            return m
    return None


def main():
    F, esim, _, fb1 = SB.er_modules()
    pb1 = SB.planform()
    pd = SB.er_profile("evolution/configs/phase2_smoke_p25.json", "c172x")
    scs = SB.er_scenarios(pd, 3, 1)
    gref = json.load(open(os.path.join(HERE, "runs", "p25_tip_verify_c172x.json")))
    pins = json.load(open(os.path.join(F.FD_DIR, "v2_results", "model_versions_post_p3b1r1.json")))["c172x"]
    pins_r0 = json.load(open(os.path.join(F.FD_DIR, "v2_results", "model_versions_post_p3b1.json")))["c172x"]
    mv_b1, mv_a1 = pins["full_a1_b1"], pins["full_a1"]
    r0 = json.load(open(R0_FILE))["cases"] if os.path.exists(R0_FILE) else {}
    out = {"method": ("shape_b1.evaluate_b1 -> evolution.fidelity.evaluate_genome(..., 'full_a1_b1', shape=...) (= FD "
                      "flexeval_b1.evaluate; direct FD path cross-checked); ER phase2_smoke_p25 c172x profile, 3 scenarios, scenario_seed 1; "
                      "gains/struct from genome/runs/p25_tip_verify_c172x.json"),
           "b1_revision": "r1 (FD planform_b1.B1_REV = %d; pins model_versions_post_p3b1r1.json)" % pb1.B1_REV,
           "model_version_pins": {"full_a1_b1": mv_b1, "full_a1": mv_a1,
                                  "genome_recorded_full_a1_b1": SB.FD_B1_MODEL_VERSIONS["c172x"],
                                  "superseded_r0_full_a1_b1": pins_r0["full_a1_b1"],
                                  "genome_recorded_r0_full_a1_b1": SB.FD_B1_MODEL_VERSIONS_R0["c172x"]},
           "encoding_check": encoding_check(pb1, SB.shape_genes()),
           "fd_schema_live": [list(x) for x in ((g.name, g.lo, g.hi, g.scale, g.default) for g in pb1.shape_schema())],
           "cases": {}}
    cases = out["cases"]
    n = len(scs)
    gains = gref["baseline"]["gains"]

    # A1 same process (reference for full 24-term identity) + B1 baseline-shape cases
    for case in ("baseline", "soft_tip_taper4_0.75"):
        g = gref[case]
        a1, cpu_a1 = timed(F.evaluate_genome, pd, g["gains"], g["struct"], scs, "full_a1", 0.9)
        a1s = summarize(a1, cpu_a1, n)
        b1, cpu_b1 = timed(SB.evaluate_b1, pd, g["gains"], g["struct"], {}, scs, check_mv=mv_b1)
        b1s = summarize(b1, cpu_b1, n)
        b1fd = summarize(SB.evaluate_b1(pd, g["gains"], g["struct"], {}, scs, check_mv=mv_b1, via="fd"), 0.0, n)
        key = "a_baseline" if case == "baseline" else "b_soft_tip_taper4_0.75"
        cases[key] = {"gains": g["gains"], "struct": g["struct"], "shape": "baseline (FD defaults)",
                      "evaluator": b1.get("evaluator"), **b1s,
                      "match": match_a1(b1s, ER_A1[case], a1s), "r0": r0_delta(b1s, r0.get(key)),
                      "direct_fd_path_identical": (b1fd["cost"], b1fd["per_scenario_cost"], b1fd["terms"]) ==
                                                  (b1s["cost"], b1s["per_scenario_cost"], b1s["terms"]),
                      "a1_same_process": {k: a1s[k] for k in ("cost", "J_wing_tip_bm_limit", "model_version",
                                                               "cpu_s", "cpu_s_per_scenario")}}
        print(key, b1s["cost"], b1s["J_wing_tip_bm_limit"], b1s["status"], b1s["model_version"],
              cases[key]["match"]["vs_er_a1"]["bit_identical"], f"cpu/sc B1 {b1s['cpu_s_per_scenario']:.3f} "
              f"A1 {a1s['cpu_s_per_scenario']:.3f}")

    # a2: through the phase3_b1 Task. (i) physical values straight in (Task.split -> evaluate_b1); (ii) the 26-gene
    # u-vector round trip (encode -> decode), which perturbs log/linear genes at the ulp level (genome encoding, not B1)
    import adapter
    task = adapter.load_task("phase3_b1")
    vals = {**gains, **gref["baseline"]["struct"], **pb1.shape_defaults()}
    t_scs = task.make_scenarios(3, 1)
    tr, cpu_t = timed(task.evaluate, vals, t_scs)
    ts = summarize(tr, cpu_t, n)
    dec = task.spec.decode(task.spec.encode(vals))
    _, _, sh = task.split(dec)
    moved = {k: {"in": vals[k], "roundtrip": dec[k], "delta": dec[k] - vals[k]} for k in vals if dec[k] != vals[k]}
    tr2, cpu_t2 = timed(task.evaluate, dec, t_scs)
    ts2 = summarize(tr2, cpu_t2, n)
    cases["a2_baseline_via_task"] = {
        "n_genes": task.spec.n_genes, "scenarios_equal_er": [x.to_dict() for x in t_scs] == scs,
        "physical_values": {**{k: ts[k] for k in ("cost", "per_scenario_cost", "status", "J_wing_tip_bm_limit",
                                                    "model_version")},
                            "objectives": tr["objectives"], "match": match_a1(ts, ER_A1["baseline"])},
        "u_vector_roundtrip": {"genes_moved": moved, "shape_moved": {k: v for k, v in moved.items() if k in pb1.SHAPE_NAMES},
                               "shape_passed_to_fd": sh,
                               **{k: ts2[k] for k in ("cost", "per_scenario_cost", "status", "J_wing_tip_bm_limit")},
                               "delta_cost_vs_er_a1": ts2["cost"] - ER_A1["baseline"]["cost"],
                               "note": ("controller/structure genes move by ~1 ulp in log/linear encode->decode; that "
                                        "perturbs the cost at ~1e-16. The identity shape decodes to FD's defaults exactly (FD GENE_ENCODING).")}}
    print("a2_baseline_via_task phys", ts["cost"], cases["a2_baseline_via_task"]["physical_values"]["match"]["vs_er_a1"]["bit_identical"],
          "| roundtrip", ts2["cost"], "moved", sorted(moved))

    # c: non-baseline shapes that fly (baseline structure, same gains)
    for name, shp in SHAPES_FLY.items():
        r, cpu = timed(SB.evaluate_b1, pd, gains, gref["baseline"]["struct"], shp, scs, check_mv=mv_b1)
        s = summarize(r, cpu, n)
        s["delta_vs_a1_baseline"] = {"cost": s["cost"] - ER_A1["baseline"]["cost"],
                                     "J_wing_tip_bm_limit": s["J_wing_tip_bm_limit"]}
        cases[f"c_{name}"] = {"gains": gains, "struct": gref["baseline"]["struct"], "shape_input": shp, **s,
                              "r0": r0_delta(s, r0.get(f"c_{name}"))}
        dr = cases[f"c_{name}"]["r0"]
        print(f"c_{name}", s["cost"], s["J_wing_tip_bm_limit"], s["status"], f"cpu/sc {s['cpu_s_per_scenario']:.3f}",
              "r0->r1 dcost", None if dr is None else dr["delta_cost"])

    # d: reject paths
    d = {}
    for name, shp in {"out_of_range_chord_taper3_0.5": {"wing_chord_taper_3": 0.5},
                      "deferred_dihedral_2deg": {"wing_dihedral_delta_deg": 2.0},
                      "structure_gene_in_shape_block": {"wing_ei_root": 1.0}}.items():
        try:
            SB.evaluate_b1(pd, gains, gref["baseline"]["struct"], shp, scs)
            d[name] = {"raised": False}
        except ValueError as e:
            d[name] = {"raised": True, "error": f"ValueError: {e}", "flown": False}
    # in-range box: FD gate on the 2^6 corners (all must pass per FD)
    import itertools
    geom = F.fd_modules()["fb"].geometry_for("c172x", F.roots(esim.Profile.from_dict(pd))[1])
    fw = F.fd_modules()["fw"]
    pw = fw.params_for("c172x", geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
    corners = [dict(zip(pb1.SHAPE_NAMES, c)) for c in itertools.product(*[(g.lo, g.hi) for g in pb1.shape_schema()])]
    corner_fail = [c for c in corners if not pb1.geometry_gate(pw, c, n_el=64).ok]
    d["in_range_corners_gate"] = {"n": len(corners), "n_rejected": len(corner_fail), "rejected": corner_fail[:5]}
    # injected: tighten FD's max taper in memory so an in-range shape fails -> exercise evaluate's reject path
    old = pb1.GATE_MAX_TAPER
    try:
        pb1.GATE_MAX_TAPER = 0.5
        r, cpu = timed(SB.evaluate_b1, pd, gains, gref["baseline"]["struct"], {"wing_chord_taper_3": 0.9}, scs, via="fd")
    finally:
        pb1.GATE_MAX_TAPER = old
    s = summarize(r, cpu, n)
    d["injected_gate_reject"] = {"how": "planform_b1.GATE_MAX_TAPER = 0.5 in memory only (FD file untouched), restored",
                                 "shape_input": {"wing_chord_taper_3": 0.9},
                                 **{k: s[k] for k in ("cost", "per_scenario_cost", "status", "per_scenario_status",
                                                      "not_flown", "geometry_gate", "model_version", "cpu_s")},
                                 "fail_cost": float(2 * esim.Profile.from_dict(pd).fail_base),
                                 "is_geometry_reject": SB.geometry_rejected(r),
                                 "note": ("model_version differs here because FD hashes the gate constants; run on the direct FD path "
                                         "(ER's evaluate_genome would refuse the changed model_version, by design)")}
    print("d_injected_gate_reject", s["status"], s["cost"], s["not_flown"])
    assert pb1.GATE_MAX_TAPER == old
    cases["d_reject"] = d

    # e: Evolution Runner's own B1 verify genomes (evolution/analysis/p3b1_verify_evolution.json, b_shaped), re-flown
    er_path = os.path.join(SB.TEAM, "evolution", "analysis", "p3b1_verify_evolution.json")
    e = {}
    if os.path.exists(er_path):
        erv = json.load(open(er_path))
        er_mv = (erv.get("model_version_pins") or {}).get("full_a1_b1") or _find_mv(erv)
        for name, x in erv.get("b_shaped", {}).items():
            shp = x["shape_genes"]
            r_er = summarize(SB.evaluate_b1(pd, gains, gref["baseline"]["struct"], shp, scs, via="er"), 0.0, n)
            r_fd = summarize(SB.evaluate_b1(pd, gains, gref["baseline"]["struct"], shp, scs, via="fd"), 0.0, n)
            mv_g = x.get("model_version") or er_mv
            rec = {"cost": x["cost"], "per_scenario_cost": x["per_scenario_cost"], "J_wing_tip_bm_limit": x["J_wing_tip_bm_limit"]}
            e[name] = {"shape": shp, "er_recorded": rec, "er_recorded_model_version": mv_g,
                       "er_recorded_revision": {"match": "r1", "superseded_r0": "r0"}.get(
                           SB.classify_b1_model_version("c172x", mv_g), "unknown"),
                       "model_version": r_er["model_version"],
                       "r0": r0_delta(r_er, rec) if mv_g == pins_r0["full_a1_b1"] else None,
                       "genome_cost": r_er["cost"], "genome_J_wing_tip_bm_limit": r_er["J_wing_tip_bm_limit"],
                       "bit_identical_vs_er_recorded": (r_er["cost"], r_er["per_scenario_cost"], r_er["J_wing_tip_bm_limit"])
                                                       == (x["cost"], x["per_scenario_cost"], x["J_wing_tip_bm_limit"]),
                       "er_path_equals_fd_path": (r_er["cost"], r_er["per_scenario_cost"], r_er["terms"]) ==
                                                 (r_fd["cost"], r_fd["per_scenario_cost"], r_fd["terms"])}
            print(f"e_{name}", r_er["cost"], e[name]["bit_identical_vs_er_recorded"], e[name]["er_path_equals_fd_path"],
                  "r0->r1 dcost", (e[name]["r0"] or {}).get("delta_cost"))
    cases["e_er_b1_verify_genomes"] = e

    # f: ER's own r1 re-fly (if present) vs our r1 numbers: cost, per-scenario, tip, model_version
    er1_path = os.path.join(SB.TEAM, "evolution", "analysis", "p3b1r1_verify_evolution.json")
    f = {}
    if os.path.exists(er1_path):
        er1 = json.load(open(er1_path))
        for k, x in er1.get("b_shaped_r1_vs_r0", {}).items():
            src, name = k.split(":", 1)
            ours = (e.get(name) and {"cost": e[name]["genome_cost"], "J_wing_tip_bm_limit": e[name]["genome_J_wing_tip_bm_limit"],
                                     "model_version": e[name]["model_version"]}) if src == "er" else cases.get(name)
            if not ours:
                f[k] = {"compared": False}
                continue
            r1 = x["r1"]
            f[k] = {"er_r1_cost": r1["cost"], "genome_r1_cost": ours["cost"],
                    "er_r1_J_wing_tip_bm_limit": r1["J_wing_tip_bm_limit"], "genome_r1_J_wing_tip_bm_limit": ours["J_wing_tip_bm_limit"],
                    "er_model_version": r1.get("model_version"), "genome_model_version": ours["model_version"],
                    "bit_identical": (r1["cost"], r1["J_wing_tip_bm_limit"], r1.get("model_version")) ==
                                     (ours["cost"], ours["J_wing_tip_bm_limit"], ours["model_version"])}
            if src == "genome":
                f[k]["per_scenario_identical"] = r1.get("per_scenario_cost") == ours["per_scenario_cost"]
            print(f"f_{k}", r1["cost"], f[k]["bit_identical"])
        f["_er_a_all_bit_identical"] = er1.get("a_all_bit_identical")
        f["_er_model_version_expected"] = er1.get("model_version_expected")
    cases["f_er_r1_verify"] = f

    out["summary"] = {
        "a_bit_identical_vs_er_a1": cases["a_baseline"]["match"]["vs_er_a1"]["bit_identical"],
        "a_bit_identical_vs_same_process_a1_all_terms": cases["a_baseline"]["match"]["vs_same_process_a1"]["bit_identical"],
        "a2_task_path_physical_bit_identical_vs_er_a1":
            cases["a2_baseline_via_task"]["physical_values"]["match"]["vs_er_a1"]["bit_identical"],
        "a2_task_u_roundtrip_delta_cost": cases["a2_baseline_via_task"]["u_vector_roundtrip"]["delta_cost_vs_er_a1"],
        "b_bit_identical_vs_er_a1": cases["b_soft_tip_taper4_0.75"]["match"]["vs_er_a1"]["bit_identical"],
        "b_bit_identical_vs_same_process_a1_all_terms":
            cases["b_soft_tip_taper4_0.75"]["match"]["vs_same_process_a1"]["bit_identical"],
        "b1_model_version": cases["a_baseline"]["model_version"],
        "b1_model_version_matches_pin": cases["a_baseline"]["model_version"] == mv_b1,
        "c_all_ok": all(cases[f"c_{k}"]["status"] == "ok" for k in SHAPES_FLY),
        "c_r0_to_r1_delta_cost": {k: (cases[f"c_{k}"]["r0"] or {}).get("delta_cost") for k in SHAPES_FLY},
        "ab_unchanged_vs_r0": all((cases[k]["r0"] or {}).get("identical", False) for k in ("a_baseline", "b_soft_tip_taper4_0.75")),
        "encoding_exact": out["encoding_check"]["exact"],
        "evaluator": cases["a_baseline"]["evaluator"],
        "ab_direct_fd_path_identical": cases["a_baseline"]["direct_fd_path_identical"]
                                       and cases["b_soft_tip_taper4_0.75"]["direct_fd_path_identical"],
        "e_er_genomes_bit_identical": (all(v["bit_identical_vs_er_recorded"] for v in e.values()) if e else None),
        "e_er_recorded_revision": sorted({v["er_recorded_revision"] for v in e.values()}) if e else None,
        "e_r0_to_r1_delta_cost": {k: (v["r0"] or {}).get("delta_cost") for k, v in e.items()},
        "f_er_r1_bit_identical": (all(v["bit_identical"] for k, v in f.items() if not k.startswith("_") and v.get("compared", True))
                                  if f else None),
        "e_er_path_equals_fd_path": (all(v["er_path_equals_fd_path"] for v in e.values()) if e else None),
        "cpu_s_per_scenario": {"a1_baseline": cases["a_baseline"]["a1_same_process"]["cpu_s_per_scenario"],
                               "b1_baseline": cases["a_baseline"]["cpu_s_per_scenario"],
                               **{f"b1_{k}": cases[f"c_{k}"]["cpu_s_per_scenario"] for k in SHAPES_FLY}},
    }
    dest = os.path.join(HERE, "runs", "p3b1_verify_c172x.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, default=float)
        f.write("\n")
    print("wrote", dest)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
