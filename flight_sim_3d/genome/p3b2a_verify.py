#!/usr/bin/env python3
"""P3-B2a verify set on the genome side (read-only on evolution/ and flight-dynamics/; bytecode writing off).

Same setup as p3b1_verify.py: ER phase2_smoke_p25 c172x resolved profile, 3 scenarios, scenario_seed 1, gains +
structure from runs/p25_tip_verify_c172x.json. Fidelity full_a1_b2a through shape_b2.evaluate_b2 (FD path; ER has not
wired full_a1_b2a yet). Locked genes = wing_tc_* injected as FD defaults (1.0).

  enc  decode/encode/identity/gate vs FD live (no flight)
  V1   B2a at defaults ≡ B1 r1 bit-for-bit on the p3b1 verify cases (baseline, soft tip, fly shapes)
       encodings: empty dict, full defaults, encoded-default u, lock-injected active-only
  task phase3_b2a Task path on baseline

Usage: $PY genome/p3b2a_verify.py  -> genome/runs/p3b2a_verify_c172x.json
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

import numpy as np

import adapter  # noqa: E402
import shape_b1 as SB1  # noqa: E402
import shape_b2 as SB  # noqa: E402

RIGID = SB.RIGID_TERMS
ER_A1 = {
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
    "twist_mid_m2": {"wing_twist_mid_deg": -2.0},
    "twist_mid_p1": {"wing_twist_mid_deg": 1.0},
}
LOCKED = list(SB.DEFAULT_LOCKED_GENES)


def summarize(r, cpu_s, n_sc):
    t = r["terms"]
    fd = {k: float(v) for k, v in t.items() if k not in RIGID}
    return {"cost": r["cost"], "per_scenario_cost": [float(e["cost"]) for e in r["per_scenario"]],
            "status": r["status"], "per_scenario_status": [e.get("status") for e in r["per_scenario"]],
            "J_wing_tip_bm_limit": float(t.get("J_wing_tip_bm_limit", 0.0)), "fd_struct_terms": fd,
            "fd_struct_sum": float(sum(fd.values())), "controller_terms": {k: float(t[k]) for k in RIGID if k in t},
            "terms": {k: float(v) for k, v in t.items()}, "margins": r.get("margins"),
            "mass_total_frac": r.get("mass_total_frac"), "model_version": r["model_version"],
            "fidelity": r.get("fidelity"), "shape_genes": r.get("shape_genes"),
            "shape_genes_b2": r.get("shape_genes_b2"), "shape_cache_key": r.get("shape_cache_key"),
            "geometry_gate": r.get("geometry_gate"), "geometry_gate_b2": r.get("geometry_gate_b2"),
            "locked_genes": r.get("locked_genes"), "not_flown": any(e.get("not_flown") for e in r["per_scenario"]),
            "cpu_s": cpu_s, "cpu_s_per_scenario": cpu_s / n_sc, "evaluator": r.get("evaluator")}


def timed(fn, *a, **k):
    c0 = time.process_time()
    r = fn(*a, **k)
    return r, time.process_time() - c0


def match_b1(s, ref):
    m = {"bit_identical_total": s["cost"] == ref["cost"],
         "bit_identical_per_scenario": s["per_scenario_cost"] == ref["per_scenario_cost"],
         "bit_identical_tip": s["J_wing_tip_bm_limit"] == ref["J_wing_tip_bm_limit"]}
    if "terms" in ref:
        diff = {k: s["terms"][k] - ref["terms"][k] for k in ref["terms"] if s["terms"].get(k) != ref["terms"][k]}
        m["all_24_terms_identical"] = not diff
        m["term_diffs"] = diff
        m["bit_identical"] = m["bit_identical_total"] and m["bit_identical_per_scenario"] and not diff
    else:
        m["bit_identical"] = all(m[k] for k in ("bit_identical_total", "bit_identical_per_scenario", "bit_identical_tip"))
    if not m["bit_identical_total"]:
        m["delta_total"] = s["cost"] - ref["cost"]
    return m


def encoding_check(pb2, model="c172x", n=5000, seed=0):
    genes = SB.shape_genes(model, LOCKED)
    rng = np.random.default_rng(seed)
    n_s = len(genes)
    U = np.vstack([rng.random((n, n_s)), np.zeros(n_s), np.ones(n_s),
                   [SB.active_identity_u(model, LOCKED)]])
    dec_mis = enc_mis = 0
    for u in U:
        full_u = SB.pack_shape_u(u, LOCKED, model, genes)
        fd = pb2.decode_shape_b2(full_u, model)
        active = {g.name: g.decode(float(x)) for g, x in zip(genes, u)}
        packed = SB.pack_shape_dict(active, LOCKED, model)
        dec_mis += any(fd[k] != packed[k] for k in packed)
        enc_mis += not np.allclose(pb2.encode_shape_b2(model, packed), full_u)
    idu = SB.active_identity_u(model, LOCKED)
    full_id = SB.pack_shape_dict({g.name: g.decode(float(x)) for g, x in zip(genes, idu)}, LOCKED, model)
    gate = None
    try:
        F, esim, _, _ = SB.er_modules()
        P = esim.Profile.from_dict(SB.er_profile("evolution/configs/phase2_smoke_p25.json", model))
        root, rv2 = F.roots(P)
        m = F.fd_modules()
        geom = m["fb"].geometry_for(model, rv2)
        pw = m["fw"].params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
        g = pb2.geometry_gate_b2(pw, full_id, model)
        gate = {"ok": bool(g.ok), "reason": getattr(g, "reason", None)}
    except Exception as e:
        gate = {"error": f"{type(e).__name__}: {e}"}
    return {"fd_gene_encoding": pb2.GENE_ENCODING, "genome_pin": SB.B2_GENE_ENCODING_PIN,
            "pin_matches": pb2.GENE_ENCODING == SB.B2_GENE_ENCODING_PIN, "fd_b2_rev": pb2.B2_REV,
            "n_active": n_s, "locked_genes": list(LOCKED), "n_vectors": len(U),
            "decode_mismatches": int(dec_mis), "encode_mismatches": int(enc_mis),
            "identity_u_active": [float(x) for x in idu],
            "identity_full_is_baseline_b2": bool(pb2.is_baseline_b2(full_id, model)),
            "locked_injected_defaults": SB.locked_defaults(model, LOCKED),
            "geometry_gate_identity": gate,
            "exact": dec_mis == 0 and enc_mis == 0 and pb2.GENE_ENCODING == SB.B2_GENE_ENCODING_PIN}


def main():
    pb2 = SB.planform()
    SB.check_schema_pin("c172x")
    SB.check_encoding_pin()
    pd = SB.er_profile("evolution/configs/phase2_smoke_p25.json", "c172x")
    scs = SB.er_scenarios(pd, 3, 1)
    gref = json.load(open(os.path.join(HERE, "runs", "p25_tip_verify_c172x.json")))
    gains = gref["baseline"]["gains"]
    struct = gref["baseline"]["struct"]
    soft = dict(struct, wing_ei_taper_4=0.75)
    n_sc = len(scs)
    cases = {}
    summary = {"evaluator": "shape_b2 via=fd (ER missing full_a1_b2a)", "locked_genes": list(LOCKED),
               "n_genes_phase3_b2a": 29, "n_shape_active": 9}

    enc = encoding_check(pb2)
    cases["encoding"] = enc
    summary["encoding_exact"] = enc["exact"]
    summary["identity_gate_ok"] = (enc.get("geometry_gate_identity") or {}).get("ok")

    # ---- V1 encodings on baseline (empty / defaults / encoded-u / lock-injected active)
    encodings = {
        "empty_dict": {},
        "full_defaults": pb2.shape_defaults_b2(),
        "encoded_identity_u": pb2.decode_shape_b2(pb2.identity_u("c172x"), "c172x"),
        "lock_injected_active_defaults": SB.pack_shape_dict(
            {g.name: g.default for g in SB.shape_genes("c172x", LOCKED)}, LOCKED, "c172x"),
    }
    b1_base, t_b1 = timed(SB1.evaluate_b1, pd, gains, struct, {}, scs, via="fd")
    cases["b1_baseline_same_process"] = summarize(b1_base, t_b1, n_sc)
    enc_results = {}
    all_enc_ok = True
    for name, shape in encodings.items():
        r, dt = timed(SB.evaluate_b2, pd, gains, struct, shape, scs, locked=LOCKED, via="fd",
                      check_mv=SB.FD_B2_MODEL_VERSIONS["c172x"])
        s = summarize(r, dt, n_sc)
        s["match_b1"] = match_b1(s, {"cost": b1_base["cost"],
                                     "per_scenario_cost": [e["cost"] for e in b1_base["per_scenario"]],
                                     "J_wing_tip_bm_limit": b1_base["terms"]["J_wing_tip_bm_limit"],
                                     "terms": {k: float(v) for k, v in b1_base["terms"].items()}})
        s["match_er_a1"] = match_b1(s, ER_A1["baseline"])
        enc_results[name] = s
        all_enc_ok &= s["match_b1"]["bit_identical"]
    cases["v1_baseline_encodings"] = enc_results
    summary["v1_baseline_all_encodings_bit_identical_vs_b1"] = all_enc_ok
    summary["b2a_model_version"] = enc_results["empty_dict"]["model_version"]
    summary["b2a_model_version_matches_pin"] = (
        enc_results["empty_dict"]["model_version"] == SB.FD_B2_MODEL_VERSIONS["c172x"])

    # soft tip
    r_soft, t_soft = timed(SB.evaluate_b2, pd, gains, soft, {}, scs, locked=LOCKED, via="fd")
    s_soft = summarize(r_soft, t_soft, n_sc)
    b1_soft, _ = timed(SB1.evaluate_b1, pd, gains, soft, {}, scs, via="fd")
    s_soft["match_b1"] = match_b1(s_soft, {"cost": b1_soft["cost"],
                                           "per_scenario_cost": [e["cost"] for e in b1_soft["per_scenario"]],
                                           "J_wing_tip_bm_limit": b1_soft["terms"]["J_wing_tip_bm_limit"],
                                           "terms": {k: float(v) for k, v in b1_soft["terms"].items()}})
    s_soft["match_er_a1"] = match_b1(s_soft, ER_A1["soft_tip_taper4_0.75"])
    cases["v1_soft_tip_taper4_0.75"] = s_soft
    summary["v1_soft_tip_bit_identical_vs_b1"] = s_soft["match_b1"]["bit_identical"]

    # fly shapes (B1 shape deltas; B2 part at defaults via lock inject)
    fly = {}
    fly_ok = True
    for name, shape in SHAPES_FLY.items():
        r, dt = timed(SB.evaluate_b2, pd, gains, struct, shape, scs, locked=LOCKED, via="fd")
        s = summarize(r, dt, n_sc)
        r1, _ = timed(SB1.evaluate_b1, pd, gains, struct, shape, scs, via="fd")
        s["match_b1"] = match_b1(s, {"cost": r1["cost"],
                                     "per_scenario_cost": [e["cost"] for e in r1["per_scenario"]],
                                     "J_wing_tip_bm_limit": r1["terms"]["J_wing_tip_bm_limit"],
                                     "terms": {k: float(v) for k, v in r1["terms"].items()}})
        s["shape_input"] = shape
        fly[name] = s
        fly_ok &= s["match_b1"]["bit_identical"] and s["status"] == "ok"
    cases["v1_fly_shapes"] = fly
    summary["v1_fly_shapes_all_bit_identical_vs_b1"] = fly_ok

    # Task path
    task = adapter.load_task("phase3_b2a")
    vals = {**gains, **struct, **{g.name: g.default for g in task.shape_gene_specs}}
    rt, t_task = timed(task.evaluate, vals, task.make_scenarios(3, 1))
    cases["task_baseline"] = {"n_genes": task.spec.n_genes, "locked_genes": task.locked_genes,
                              "cost": rt["cost"], "status": rt["status"], "model_version": rt["model_version"],
                              "J_wing_tip_bm_limit": rt["J_wing_tip_bm_limit"],
                              "struct_v2_source": rt["struct_v2_source"], "cpu_s": t_task,
                              "bit_identical_vs_b1": rt["cost"] == b1_base["cost"]}
    summary["task_baseline_bit_identical_vs_b1"] = rt["cost"] == b1_base["cost"]
    summary["task_n_genes"] = task.spec.n_genes

    summary["v1_all_bit_identical"] = (summary["v1_baseline_all_encodings_bit_identical_vs_b1"]
                                       and summary["v1_soft_tip_bit_identical_vs_b1"]
                                       and summary["v1_fly_shapes_all_bit_identical_vs_b1"]
                                       and summary["task_baseline_bit_identical_vs_b1"])

    out = {"method": "genome p3b2a_verify.py", "b2_revision": pb2.B2_REV, "b2_stage": pb2.B2_STAGE,
           "model_version_pins": dict(SB.FD_B2_MODEL_VERSIONS),
           "fd_schema_live_c172x": [{"name": g["name"], "lo": g["lo"], "hi": g["hi"], "default": g["default"],
                                    "identity_u": g["identity_u"], "requires": g["requires"]}
                                   for g in pb2.shape_schema_b2("c172x")],
           "cases": cases, "summary": summary}
    path = os.path.join(HERE, "runs", "p3b2a_verify_c172x.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    print("wrote", path)
    return 0 if summary["v1_all_bit_identical"] and summary["encoding_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
