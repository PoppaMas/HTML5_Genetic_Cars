"""P3-B1 studies (checkpointed: finished items in v2_results/p3b1_*.partial.jsonl are skipped on re-run).

  python p3b1_study.py versions        -> v2_results/model_versions_post_p3b1.json
  python p3b1_study.py acceptance      -> v2_results/p3b1_acceptance.json   (baseline shape == full_a1)
  python p3b1_study.py bench [reps]    -> v2_results/p3b1_benchmark.json    (shape rebuild + one 90 s scenario)

Acceptance ("baseline shape + structure genes = full_a1 bit-intent"):
  A. exact: fidelity full_a1_b1 with shape_genome None / {} / all-defaults vs full_a1, 4 aircraft x 2 structure genomes
     (baseline, mixed STRUCT) x the 3 Phase-1 90 s scenarios: cost, all 24 terms, margins, mass, sizing, per-scenario
     entries and every coupler history channel compared with == (record=True). Required: max |delta| = 0.
  B. structural model: every FlexBodyModel matrix of FlexBodyModelB1(shape={}) == FlexBodyModelA1 (np.array_equal).
  C. continuity of the REBUILD path (not the identity short-circuit): shape genes 1e-9 off baseline (chord taper, twist,
     sweep) -> shaped code path; preflight margins / terms / matrices vs A1 within eps (recorded), and one 90 s scenario.
Bench: method of p3a1_study bench (Phase-1 best gains, BLAS 1 thread, process CPU, min over repeats): per aircraft
  (i) model build A1 vs B1-shaped, (ii) preflight (margins + sizing) A1 vs B1-shaped, (iii) full evaluate() of ONE 90 s
  scenario: full_a1 (baseline shape) vs full_a1_b1 baseline shape vs full_a1_b1 shaped, interleaved per repeat.
"""
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

OUT = os.path.join(HERE, "v2_results")
_REV = __import__("planform_b1").B1_REV   # r0 -> p3b1_*, rN -> p3b1rN_* (r0 outputs are never overwritten)
_PFX = "p3b1" if _REV == 0 else f"p3b1r{_REV}"
MODELS = ("c172x", "T38", "737", "f16")
STRUCT = {"wing_ei_root": 1.3, "wing_ei_taper_4": 0.8, "wing_nsm_tip": 1.1, "tail_stiffness_scale": 0.9,
          "fuselage_stiffness_scale": 1.2, "struct_damping_ratio": 0.015}
SHAPED = {"wing_chord_taper_1": 0.95, "wing_chord_taper_3": 0.95, "wing_twist_mid_deg": -0.5, "wing_twist_tip_deg": -2.0,
          "wing_sweep_qc_delta_deg": 2.0}
NEAR = {"wing_chord_taper_1": 1 - 1e-9, "wing_twist_tip_deg": -1e-9, "wing_sweep_qc_delta_deg": 1e-9}
MATS = ("K", "C", "M", "Qbasis", "RB", "RE", "RD", "RA", "R_de", "R_d", "R_beta", "A_beta", "A_C_nc", "TIP", "TIP_TEL", "W", "G", "H")


def _done(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                out[r["key"]] = r
            except Exception:  # noqa: BLE001
                pass
    return out


def _append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _gains(model):
    import flexeval as fe
    S = json.load(open(os.path.join(fe.TEAM, "evolution", "runs", "phase1-s1", "summary.json")))["aircraft"]
    return next(a["best_gains"] for a in S if a["aircraft"] == model)


def _maxdiff(a, b, path=""):
    """(max abs numeric difference, list of non-numeric mismatches) between two nested JSON-like objects."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return math.inf, [f"{path}: keys differ"]
        m, bad = 0.0, []
        for k in a:
            d, bb = _maxdiff(a[k], b[k], f"{path}.{k}")
            m, bad = max(m, d), bad + bb
        return m, bad
    if isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray)):
        if len(a) != len(b):
            return math.inf, [f"{path}: length differs"]
        m, bad = 0.0, []
        for i, (x, y) in enumerate(zip(a, b)):
            d, bb = _maxdiff(x, y, f"{path}[{i}]")
            m, bad = max(m, d), bad + bb
        return m, bad
    if isinstance(a, (int, float, np.floating)) and isinstance(b, (int, float, np.floating)) and not isinstance(a, bool):
        if a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)):
            return 0.0, []
        return (abs(float(a) - float(b)) if math.isfinite(a) and math.isfinite(b) else math.inf), []
    return (0.0, []) if a == b else (0.0, [f"{path}: {a!r} != {b!r}"])


def _strip(r):
    """Physics outputs only. Dropped by design: fidelity / margins_fidelity / model_version (B1 tag differs),
    wall_s / cpu_s (wall-clock of the run), and B1-only keys (shape_genes, planform, geometry_gate, ...)."""
    keep = ("cost", "terms", "status", "margins", "mass", "sizing", "terms_available", "margin_gate", "loads")
    out = {k: r[k] for k in keep if k in r}
    drop_ps = {"wall_s", "cpu_s", "sim_wall_s", "sim_cpu_s"}
    if "per_scenario" in r:
        out["per_scenario"] = [{k: v for k, v in e.items() if k not in drop_ps
                                and k not in ("trajectory",)} for e in r["per_scenario"]]
    if "telemetry" in r:
        out["structure"] = [t["structure"] for t in r["telemetry"]]
    return out


def versions():
    import coupled_sim as cs
    import flexbody as fb
    import flexeval_b1 as fb1
    import planform_b1 as pb1
    p25 = json.load(open(os.path.join(OUT, "model_versions_post_p25.json")))
    pa1 = json.load(open(os.path.join(OUT, "model_versions_post_p3a1.json")))
    res = {"doc": "P3-B1 model_version strings (flexeval_b1.model_version). 'full_a1_b1' is new (A1 64-strip host + B1 "
                  "planform shape block). rigid / reduced / full are byte-identical to model_versions_post_p25.json and "
                  "full_a1 to model_versions_post_p3a1.json (neither file modified; no hashed source edited).",
           "b1_schema": pb1.shape_params_for_hash(),
           "hash_covers": [os.path.basename(f) for f in fb1.CODE_FILES_B1] + [
               "b1_params (A1 params + shape schema/rules)", "terms", "weights", "struct gene schema", "shape gene schema",
               "substeps", "gate", "<root>_v2 aircraft files"]}
    for m in MODELS:
        cs.ensure_root(m)
        fb.ensure_root_v2(m)
        d = {f: fb1.model_version(f, m, cs.ROOT) for f in fb1.FIDELITIES}
        for f in ("rigid", "reduced", "full"):
            assert d[f] == p25[m][f], (m, f)
        assert d["full_a1"] == pa1[m]["full_a1"], m
        res[m] = d
    fn = os.path.join(OUT, f"model_versions_post_{_PFX}.json")
    assert not fn.endswith("model_versions_post_p3b1.json"), "r0 pin file is frozen; bump planform_b1.B1_REV"
    if _REV:
        res["supersedes"] = "model_versions_post_p3b1.json (r0 strings, left intact; valid only for the r0 B1 code)"
        res["doc"] += (f" r{_REV}: trim-consistent twist (no rigid basic-twist pitch feedback), baseline-anchored flown "
                       "wing-BM reference (J_bm_rms denominator / J_bm_peak allowable), shaped node_layout_b1, gene "
                       "encoding stated (linear in value).")
    json.dump(res, open(fn, "w"), indent=1)
    print(json.dumps({m: res[m]["full_a1_b1"] for m in MODELS}, indent=1))


def acceptance():
    import coupled_sim as cs
    import flexbody as fb
    import flexbody_a1 as a1
    import flexbody_b1 as b1
    import flexeval as fe
    import flexeval_a1 as fa
    import flexeval_b1 as fb1
    fb.blas_threads(1)
    sim = fe.load_sim()
    part = os.path.join(OUT, f"{_PFX}_acceptance.partial.jsonl")
    done = _done(part)
    for m in MODELS:
        cs.ensure_root(m)
        fb.ensure_root_v2(m)
        # ---- B: structural model matrices (cheap, always recomputed)
        key = f"B|{m}"
        if key not in done:
            rec = {"key": key, "model": m, "genomes": {}}
            for gn, g in (("baseline", None), ("STRUCT", STRUCT)):
                A, B = a1.FlexBodyModelA1(m, g), b1.FlexBodyModelB1(m, g, shape_genes={})
                eq = {k: bool(np.array_equal(getattr(A, k), getattr(B, k))) for k in MATS}
                eq.update({f"A_K.{gk}": bool(np.array_equal(A.A_K[gk], B.A_K[gk])) for gk in fb.GROUPS})
                eq.update({f"A_C.{gk}": bool(np.array_equal(A.A_C[gk], B.A_C[gk])) for gk in fb.GROUPS})
                eq["mass_summary"] = A.mass_summary() == B.mass_summary()
                ra, rb = a1.margin_terms_a1(A), b1.margin_terms_b1(B)
                eq["preflight_terms"] = ra["terms"] == rb["terms"]
                eq["margins"] = ra["margins"] == rb["margins"]
                rec["genomes"][gn] = {"all_equal": all(eq.values()), "checks": eq}
            _append(part, rec)
            done[key] = rec
            print(key, {g: v["all_equal"] for g, v in rec["genomes"].items()}, flush=True)
        # ---- A: whole evaluate, 3 x 90 s scenarios, record=True
        for gn, g in (("baseline", {}), ("STRUCT", STRUCT)):
            key = f"A|{m}|{gn}"
            if key in done:
                continue
            scs = fe.phase1_scenarios(m, sim)
            t0 = time.process_time()
            ra = fa.evaluate(_gains(m), g, scs, m, fidelity="full_a1", root=cs.ROOT, sim=sim, record=True)
            ta = time.process_time() - t0
            variants = {}
            for vn, sg in (("None", None), ("empty", {}), ("defaults", __import__("planform_b1").shape_defaults())):
                rb = fb1.evaluate(_gains(m), g, scs, m, fidelity="full_a1_b1", root=cs.ROOT, sim=sim, record=True,
                                  shape_genome=sg)
                d, bad = _maxdiff(_strip(ra), _strip(rb))
                variants[vn] = {"max_abs_diff": d, "non_numeric_mismatch": bad[:5], "cost": rb["cost"],
                                "status": rb["status"], "model_version": rb["model_version"]}
            rec = {"key": key, "model": m, "struct_genome": gn, "scenarios": [s.name if hasattr(s, "name") else str(i) for i, s in enumerate(scs)],
                   "duration_s": [float(s.duration_s) for s in scs], "a1_cost": ra["cost"], "a1_status": ra["status"],
                   "a1_terms": ra["terms"], "a1_model_version": ra["model_version"], "a1_cpu_s": ta, "b1": variants,
                   "n_history_channels": len(ra["telemetry"][0]["structure"]),
                   "n_frames": len(next(iter(ra["telemetry"][0]["structure"].values())))}
            _append(part, rec)
            done[key] = rec
            print(key, {v: (x["max_abs_diff"], len(x["non_numeric_mismatch"])) for v, x in variants.items()}, flush=True)
        # ---- C: continuity of the rebuild path
        key = f"C|{m}"
        if key not in done:
            A, S = a1.FlexBodyModelA1(m), b1.FlexBodyModelB1(m, shape_genes=NEAR)
            assert not S.planform_baseline
            ra, rs = a1.margin_terms_a1(A), b1.margin_terms_b1(S)
            mats = {k: float(np.max(np.abs(getattr(S, k) - getattr(A, k))) / max(1e-300, float(np.max(np.abs(getattr(A, k))))))
                    for k in ("K", "Qbasis", "RB", "RE", "R_de", "W", "G")}
            scs = fe.phase1_scenarios(m, sim)[:1]
            ea = fa.evaluate(_gains(m), {}, scs, m, fidelity="full_a1", root=cs.ROOT, sim=sim)
            es = fb1.evaluate(_gains(m), {}, scs, m, fidelity="full_a1_b1", root=cs.ROOT, sim=sim, shape_genome=NEAR)
            rec = {"key": key, "model": m, "shape_genome": NEAR,
                   "matrix_rel_maxdiff": mats,
                   "margins_abs_maxdiff": max(abs(rs["margins"][k] - ra["margins"][k]) for k in ("flutter_margin", "div_margin", "reversal_margin")),
                   "preflight_terms_abs_maxdiff": max(abs(rs["terms"][k] - ra["terms"][k]) for k in ra["terms"]),
                   "flight_90s_cost_abs_diff": abs(es["cost"] - ea["cost"]),
                   "flight_90s_terms_abs_maxdiff": max(abs(es["terms"][k] - ea["terms"][k]) for k in ea["terms"]),
                   "a1_cost": ea["cost"], "b1_near_cost": es["cost"], "status": [ea["status"], es["status"]]}
            _append(part, rec)
            done[key] = rec
            print(key, {k: rec[k] for k in ("margins_abs_maxdiff", "preflight_terms_abs_maxdiff", "flight_90s_cost_abs_diff")}, flush=True)
    A_ = [v for k, v in done.items() if k.startswith("A|")]
    B_ = [v for k, v in done.items() if k.startswith("B|")]
    C_ = [v for k, v in done.items() if k.startswith("C|")]
    worstA = max(x["max_abs_diff"] for r in A_ for x in r["b1"].values())
    mismA = sum(len(x["non_numeric_mismatch"]) for r in A_ for x in r["b1"].values())
    res = {
        "doc": "P3-B1 acceptance: baseline shape (None / {} / all defaults) + structure genes at fidelity full_a1_b1 vs "
               "full_a1 (p3b1_study.py acceptance). JSBSim 1.3.1, BLAS 1 thread, Phase-1 best gains, 3 Phase-1 90 s "
               "scenarios, record=True.",
        "criteria": {
            "exact_required": ["cost", "all 24 TERM_KEYS", "status", "margins (all blocks)", "mass summary",
                               "sizing (terms, ratios, allowables, demands)", "per-scenario physics entries", "loads",
                               "every coupler history channel, every frame", "all FlexBodyModel matrices",
                               "rigid / reduced / full / full_a1 model_version strings"],
            "exact_definition": "Python == on floats (max |delta| = 0.0), no tolerance",
            "different_by_design": ["model_version string (full_a1_b1:flexv2b1:<sha8> vs full_a1:flexv2a1:<sha8>)",
                                    "fidelity / margins_fidelity = 'full_a1_b1' (vs 'full_a1')",
                                    "per_scenario[*].wall_s (wall-clock of the run; not physics)",
                                    "extra output keys: shape_genes, shape_cache_key, geometry_gate, planform, telemetry[i].planform"],
            "continuity_eps (rebuild path at 1e-9 shape perturbation)": {"margins_abs": 1e-6, "preflight_terms_abs": 1e-6,
                                                                        "matrices_rel": 1e-6, "flight_90s_cost_abs": 1e-6},
            "fallback_if_platform_nondeterminism": "none needed (exact achieved); a consumer on another BLAS / thread count "
                                                   "should compare full_a1 vs full_a1_b1 in the SAME process, where it is exact"},
        "results": {
            "A_evaluate_exact": {"cases": len(A_) * 3, "max_abs_diff": worstA, "non_numeric_mismatches": mismA,
                                 "pass": worstA == 0.0 and mismA == 0},
            "B_matrices_exact": {"cases": sum(len(r["genomes"]) for r in B_),
                                 "pass": all(g["all_equal"] for r in B_ for g in r["genomes"].values())},
            "C_continuity": {"worst": {k: max(r[k] for r in C_) for k in ("margins_abs_maxdiff", "preflight_terms_abs_maxdiff",
                                                                          "flight_90s_cost_abs_diff", "flight_90s_terms_abs_maxdiff")},
                             "worst_matrix_rel": max(max(r["matrix_rel_maxdiff"].values()) for r in C_)},
        },
        "per_case": {"A": A_, "B": B_, "C": C_},
    }
    w = res["results"]["C_continuity"]
    w["pass"] = (w["worst"]["margins_abs_maxdiff"] <= 1e-6 and w["worst"]["preflight_terms_abs_maxdiff"] <= 1e-6
                 and w["worst"]["flight_90s_cost_abs_diff"] <= 1e-6 and w["worst_matrix_rel"] <= 1e-6)
    json.dump(res, open(os.path.join(OUT, f"{_PFX}_acceptance.json"), "w"), indent=1, default=float)
    print(json.dumps(res["results"], indent=1, default=float))


def bench(reps=3):
    import coupled_sim as cs
    import flexbody as fb
    import flexbody_a1 as a1
    import flexbody_b1 as b1
    import flexeval as fe
    import flexeval_a1 as fa
    import flexeval_b1 as fb1
    fb.blas_threads(1)
    sim = fe.load_sim()
    part = os.path.join(OUT, f"{_PFX}_bench.partial.jsonl")
    done = _done(part)
    for m in MODELS:
        cs.ensure_root(m)
        fb.ensure_root_v2(m)
        b1.FlexBodyModelB1(m, shape_genes=SHAPED)       # warm caches (calibration, baseline design loads)
        for rep in range(reps):
            key = f"{m}|{rep}"
            if key in done:
                continue
            rec = {"key": key, "model": m, "rep": rep, "load1": os.getloadavg()[0]}
            t = time.process_time(); A = a1.FlexBodyModelA1(m); rec["build_a1_s"] = time.process_time() - t
            t = time.process_time(); S = b1.FlexBodyModelB1(m, shape_genes=SHAPED); rec["build_b1_shaped_s"] = time.process_time() - t
            t = time.process_time(); a1.margin_terms_a1(A); rec["preflight_a1_s"] = time.process_time() - t
            t = time.process_time(); b1.margin_terms_b1(S); rec["preflight_b1_shaped_s"] = time.process_time() - t
            sc = fe.phase1_scenarios(m, sim)[:1]
            for lab, f in (("eval90_full_a1_s", lambda: fa.evaluate(_gains(m), {}, sc, m, fidelity="full_a1", root=cs.ROOT, sim=sim)),
                           ("eval90_b1_baseline_s", lambda: fb1.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b1", root=cs.ROOT, sim=sim, shape_genome={})),
                           ("eval90_b1_shaped_s", lambda: fb1.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b1", root=cs.ROOT, sim=sim, shape_genome=SHAPED))):
                t = time.process_time(); r = f(); rec[lab] = time.process_time() - t
                rec[lab.replace("_s", "_status")] = r["status"]
                rec[lab.replace("_s", "_cost")] = r["cost"]
            rec["scenario"] = getattr(sc[0], "name", "0")
            rec["duration_s"] = float(sc[0].duration_s)
            _append(part, rec)
            done[key] = rec
            print(key, {k: round(v, 3) for k, v in rec.items() if k.endswith("_s") and isinstance(v, float)}, flush=True)
    res = {"doc": "P3-B1 CPU (p3b1_study.py bench): process CPU seconds, min over repeats, BLAS 1 thread, Phase-1 best gains, "
                  "ONE Phase-1 90 s scenario per evaluate() (whole call: build + preflight + trim + flight + terms). "
                  "Shaped genome = " + json.dumps(SHAPED) + ". Warm process (calibration / baseline-design caches filled once).",
           "reps": reps, "aircraft": {}}
    for m in MODELS:
        rows = [v for k, v in done.items() if v["model"] == m]
        if not rows:
            continue
        mn = {k: min(r[k] for r in rows) for k in rows[0] if k.endswith("_s") and isinstance(rows[0][k], float)}
        mn["rebuild_overhead_s"] = mn["build_b1_shaped_s"] - mn["build_a1_s"]
        mn["preflight_overhead_s"] = mn["preflight_b1_shaped_s"] - mn["preflight_a1_s"]
        mn["eval90_ratio_b1_shaped_vs_a1"] = mn["eval90_b1_shaped_s"] / mn["eval90_full_a1_s"]
        mn["eval90_ratio_b1_baseline_vs_a1"] = mn["eval90_b1_baseline_s"] / mn["eval90_full_a1_s"]
        mn["status"] = {k: rows[0][k] for k in rows[0] if k.endswith("_status")}
        mn["load1_max"] = max(r["load1"] for r in rows)
        res["aircraft"][m] = mn
    json.dump(res, open(os.path.join(OUT, f"{_PFX}_benchmark.json"), "w"), indent=1, default=float)
    print(json.dumps(res["aircraft"], indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "versions"
    if cmd == "versions":
        versions()
    elif cmd == "acceptance":
        acceptance()
    elif cmd == "bench":
        bench(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        raise SystemExit(__doc__)
