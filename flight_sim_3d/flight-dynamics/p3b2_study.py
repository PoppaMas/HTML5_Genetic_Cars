"""P3-B2a studies (checkpointed: finished items in v2_results/p3b2a_*.partial.jsonl are skipped on re-run).

  python p3b2_study.py energyref         -> v2_results/p3b2a_energy_ref.json   (frozen baseline drag reference)
  python p3b2_study.py versions          -> v2_results/model_versions_post_p3b2a.json
  python p3b2_study.py acceptance [M..]  -> v2_results/p3b2a_acceptance.json
  python p3b2_study.py gatescan          -> v2_results/p3b2a_gatescan.json      (corners + 2000 random, 4 aircraft)
  python p3b2_study.py sweep M           -> _scratch/p3b2/sweep_M.jsonl         (gene-safety / energy calibration probe)
  python p3b2_study.py calib             -> v2_results/p3b2a_energy_calibration.json
  python p3b2_study.py bench [reps]      -> v2_results/p3b2a_benchmark.json

Acceptance:
  A. exact: full_a1_b2a with B2 at default, encodings None / {} / defaults dict / identity u-vector, vs full_a1_b1,
     4 aircraft x {baseline, STRUCT} x {B1 baseline shape, B1 bench SHAPED} x the 3 Phase-1 90 s scenarios, record=True:
     every pre-existing output key compared with == after removing the by-design keys (fidelity, margins_fidelity,
     model_version, shape_cache_key, wall/cpu times). Required: max |delta| = 0, no non-numeric mismatch.
  B. FlexBodyModelB2(B2 default) matrices == FlexBodyModelB1 (np.array_equal), {baseline, STRUCT}.
  C. continuity: each B2 gene 1e-9 off default (forced shaped path): matrices / preflight terms vs B1 (rel), and ONE
     Phase-1 90 s scenario cost vs B1.
  D. native layer neutral: rigid 90 s flight on <root>_v2b2 (all B2 properties 0) == <root>_v2 (cost, track, effort).
"""
import itertools
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
SCR = os.path.join(HERE, "_scratch", "p3b2")
MODELS = ("c172x", "T38", "737", "f16")
STRUCT = {"wing_ei_root": 1.3, "wing_ei_taper_4": 0.8, "wing_nsm_tip": 1.1, "tail_stiffness_scale": 0.9,
          "fuselage_stiffness_scale": 1.2, "struct_damping_ratio": 0.015}
SHAPED = {"wing_chord_taper_1": 0.95, "wing_chord_taper_3": 0.95, "wing_twist_mid_deg": -0.5, "wing_twist_tip_deg": -2.0,
          "wing_sweep_qc_delta_deg": 2.0}
MIXED = {"wing_dihedral_delta_deg": 1.0, "wing_camber_root_delta_pct": 0.5, "wing_camber_tip_delta_pct": 0.5,
         "wing_tc_root_scale": 1.1}
MATS = ("K", "C", "M", "Qbasis", "RB", "RE", "RD", "RA", "R_de", "R_d", "R_beta", "A_beta", "A_C_nc", "TIP", "TIP_TEL",
        "W", "G", "H")
DIFF_BY_DESIGN = {"fidelity", "margins_fidelity", "model_version", "shape_cache_key", "wall_s", "cpu_s", "sim_wall_s",
                  "sim_cpu_s"}
ADDED = {"shape_genes_b2", "b2", "geometry_gate_b2", "native_increments", "energy"}


def _done(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                out[r["key"] if "key" in r else r["label"]] = r
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


def _scrub(x):
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items() if k not in DIFF_BY_DESIGN}
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    return x


def _maxdiff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return math.inf, [f"{path}: keys differ {sorted(set(a) ^ set(b))[:4]}"]
        m, bad = 0.0, []
        for k in a:
            d, bb = _maxdiff(a[k], b[k], f"{path}.{k}")
            m, bad = max(m, d), bad + bb
        return m, bad
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return math.inf, [f"{path}: len {len(a)} != {len(b)}"]
        m, bad = 0.0, []
        for i, (x, y) in enumerate(zip(a, b)):
            d, bb = _maxdiff(x, y, f"{path}[{i}]")
            m, bad = max(m, d), bad + bb
        return m, bad
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return (0.0, []) if a == b else (0.0, [f"{path}: {a!r} != {b!r}"])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return 0.0, []
        return (abs(a - b) if a != b else 0.0), []
    return (0.0, []) if a == b else (0.0, [f"{path}: type/value mismatch"])


def energyref():
    import flexeval_b2 as fb2
    d = fb2.build_energy_ref_file()
    print(json.dumps({m: {k: v["cd_ref"] for k, v in x.items() if not k.startswith("_")} for m, x in d["aircraft"].items()}))


def versions():
    import coupled_sim as cs
    import flexbody as fb
    import flexeval_b2 as fb2
    import planform_b2 as pb2
    b1p = json.load(open(os.path.join(OUT, "model_versions_post_p3b1r1.json")))
    res = {"doc": "P3-B2a model_version strings (flexeval_b2.model_version). 'full_a1_b2a' is new (B1 r1 + section genes: "
                  "dihedral / camber / thickness, native JSBSim increments on <root>_v2b2, energy export). rigid / reduced / "
                  "full / full_a1 / full_a1_b1 are byte-identical to model_versions_post_p3b1r1.json (no hashed source "
                  "edited; that file is frozen).",
           "b2_schema": pb2.shape_params_for_hash(),
           "energy_ref_file": os.path.relpath(fb2.ENERGY_REF_FILE, HERE),
           "hash_covers": [os.path.basename(f) for f in fb2.CODE_FILES_B2] + [
               "b2_params", "terms", "weights", "struct gene schema", "B1+B2 shape schema / sections / gate", "native XML",
               "energy schema + frozen reference (p3b2a_energy_ref.json, this aircraft)", "<root>_v2 and <root>_v2b2 files"]}
    for m in MODELS:
        cs.ensure_root(m)
        fb.ensure_root_v2(m)
        d = {f: fb2.model_version(f, m, cs.ROOT) for f in fb2.FIDELITIES}
        for f in ("rigid", "reduced", "full", "full_a1", "full_a1_b1"):
            assert d[f] == b1p[m][f], (m, f)
        res[m] = d
    fn = os.path.join(OUT, "model_versions_post_p3b2a.json")
    json.dump(res, open(fn, "w"), indent=1, default=str)
    print({m: res[m]["full_a1_b2a"] for m in MODELS})


def acceptance(models=MODELS):
    import coupled_sim as cs
    import flexbody as fb
    import flexbody_b1 as b1
    import flexbody_b2 as b2
    import flexeval as fe
    import flexeval_b1 as fb1
    import flexeval_b2 as fb2
    import planform_b2 as pb2
    fb.blas_threads(1)
    sim = fe.load_sim()
    part = os.path.join(OUT, "p3b2a_acceptance.partial.jsonl")
    done = _done(part)
    for m in models:
        cs.ensure_root(m)
        fb.ensure_root_v2(m)
        rv2, rb2 = fe.root_v2_for(cs.ROOT), fb2.root_v2b2_for(cs.ROOT)
        b2.ensure_root_v2b2(m, rv2, rb2)
        # ---- B
        key = f"B|{m}"
        if key not in done:
            rec = {"key": key, "model": m, "genomes": {}}
            for gn, g in (("baseline", {}), ("STRUCT", STRUCT)):
                A, B = b1.FlexBodyModelB1(m, g), b2.FlexBodyModelB2(m, g, shape_genes={})
                eq = {k: bool(np.array_equal(getattr(A, k), getattr(B, k))) for k in MATS if hasattr(A, k)}
                ra, rb = b1.margin_terms_b1(A), b2.margin_terms_b2(B)
                eq.update(preflight_terms=ra["terms"] == rb["terms"], margins=ra["margins"] == rb["margins"],
                          sizing=ra["sizing"] == rb["sizing"], mass=A.mass_summary() == B.mass_summary())
                rec["genomes"][gn] = {"all_equal": all(eq.values()), "checks": eq}
            _append(part, rec)
            done[key] = rec
            print(key, {g: v["all_equal"] for g, v in rec["genomes"].items()}, flush=True)
        # ---- A
        scs = fe.phase1_scenarios(m, sim)
        for gn, g in (("baseline", {}), ("STRUCT", STRUCT)):
            for sn, s1 in (("b1_baseline", {}), ("b1_SHAPED", SHAPED)):
                key = f"A|{m}|{gn}|{sn}"
                if key in done:
                    continue
                t0 = time.process_time()
                ra = fb1.evaluate(_gains(m), g, scs, m, fidelity="full_a1_b1", root=cs.ROOT, sim=sim, record=True,
                                  shape_genome=s1)
                ta = time.process_time() - t0
                variants = {}
                encs = (("None", None if not s1 else dict(s1)), ("empty", dict(s1)),
                        ("defaults", dict(pb2.shape_defaults_b2(), **s1)),
                        ("u_vector", pb2.encode_shape_b2(m, dict(s1))))
                for vn, sg in encs:
                    t1 = time.process_time()
                    rb = fb2.evaluate(_gains(m), g, scs, m, fidelity="full_a1_b2a", root=cs.ROOT, sim=sim, record=True,
                                      shape_genome=sg)
                    tb = time.process_time() - t1
                    added = sorted(set(rb) - set(ra))
                    d, bad = _maxdiff(_scrub(ra), _scrub({k: rb[k] for k in ra if k in rb}))
                    variants[vn] = {"max_abs_diff": d, "non_numeric_mismatch": bad[:5], "n_non_numeric": len(bad),
                                    "added_keys": added, "missing_keys": sorted(set(ra) - set(rb)), "cost": rb["cost"],
                                    "status": rb["status"], "model_version": rb["model_version"], "cpu_s": tb,
                                    "energy_drag_increment": rb["energy"]["energy_drag_increment"],
                                    "energy_ref_source": rb["energy"]["per_scenario"][0]["ref_source"]}
                rec = {"key": key, "model": m, "struct_genome": gn, "b1_shape": sn, "b1_cost": ra["cost"],
                       "b1_status": ra["status"], "b1_model_version": ra["model_version"], "b1_cpu_s": ta,
                       "n_scenarios": len(scs), "duration_s": [float(s.duration_s) for s in scs], "b2": variants,
                       "n_history_channels": len(ra["telemetry"][0]["structure"])}
                _append(part, rec)
                done[key] = rec
                print(key, {v: (x["max_abs_diff"], x["n_non_numeric"]) for v, x in variants.items()}, flush=True)
        # ---- C
        key = f"C|{m}"
        if key not in done:
            A = b1.FlexBodyModelB1(m, None)
            ra = b1.margin_terms_b1(A)
            sc = scs[:1]
            r_b1 = fb1.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b1", root=cs.ROOT, sim=sim)
            genes = {}
            for gg in pb2.SHAPE_GENES_B2:
                lo, hi = gg.lo_hi(m)
                eps = 1e-9 if gg.default + 1e-9 <= hi else -1e-9
                sgd = {gg.name: gg.default + eps}
                S = b2.FlexBodyModelB2(m, None, shape_genes=sgd)
                rs = b2.margin_terms_b2(S)
                mats = {k: float(np.max(np.abs(getattr(S, k) - getattr(A, k))) / max(1e-300, float(np.max(np.abs(getattr(A, k))))))
                        for k in ("K", "M", "Qbasis", "RB")}
                terms = {k: abs(rs["terms"][k] - v) for k, v in ra["terms"].items()}
                r = fb2.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b2a", root=cs.ROOT, sim=sim, shape_genome=sgd)
                genes[gg.name] = {"eps": eps, "mat_rel": mats, "max_term_abs": max(terms.values()),
                                  "cost90": r["cost"], "cost90_b1": r_b1["cost"], "dcost90": r["cost"] - r_b1["cost"],
                                  "status": r["status"], "energy_drag_increment": r["energy"]["energy_drag_increment"]}
            rec = {"key": key, "model": m, "genes": genes}
            _append(part, rec)
            done[key] = rec
            print(key, {k: (v["max_term_abs"], v["dcost90"]) for k, v in genes.items()}, flush=True)
        # ---- D
        key = f"D|{m}"
        if key not in done:
            P0 = fe.default_profile(m, sim)
            sc = scs[0]
            ra = sim.simulate(_gains(m), sc, fe._profile_with_root(P0, sim, rv2, m))
            rb = sim.simulate(_gains(m), sc, fe._profile_with_root(P0, sim, rb2, m))
            rec = {"key": key, "model": m, "equal": all(ra[k] == rb[k] for k in ("cost", "track", "effort", "status")),
                   "cost_v2": ra["cost"], "cost_v2b2": rb["cost"]}
            _append(part, rec)
            done[key] = rec
            print(key, rec["equal"], flush=True)
    rows = list(done.values())
    A = [r for r in rows if r["key"].startswith("A|")]
    res = {"doc": __doc__, "summary": {
        "A_cases": len(A) * 4, "A_max_abs_diff": max([v["max_abs_diff"] for r in A for v in r["b2"].values()], default=None),
        "A_non_numeric": sum(v["n_non_numeric"] for r in A for v in r["b2"].values()),
        "B_all_equal": all(g["all_equal"] for r in rows if r["key"].startswith("B|") for g in r["genomes"].values()),
        "C_max_term_abs": max([g["max_term_abs"] for r in rows if r["key"].startswith("C|") for g in r["genes"].values()], default=None),
        "C_max_abs_dcost90": max([abs(g["dcost90"]) for r in rows if r["key"].startswith("C|") for g in r["genes"].values()], default=None),
        "D_all_equal": all(r["equal"] for r in rows if r["key"].startswith("D|"))}, "records": rows}
    json.dump(res, open(os.path.join(OUT, "p3b2a_acceptance.json"), "w"), indent=1, default=float)
    print(json.dumps(res["summary"], default=float))


def gatescan(n_random=2000, seed=7):
    import coupled_sim as cs
    import flexbody as fb
    import flexwing as fw
    import flexeval as fe
    import planform_b2 as pb2
    sim = fe.load_sim()
    rng = np.random.default_rng(seed)
    res = {"doc": "B2a gate scan: all 2^5 corners of the B2a box (B1 at default) + random points of the full 11-gene box "
                  "(B1 uniform too), Phase-1 envelope, tip clearance from <root>_v2 gear, trim authority (native dCm0).",
           "n_random": n_random, "seed": seed, "aircraft": {}}
    rv2 = fe.root_v2_for(cs.ROOT)
    for m in MODELS:
        fb.ensure_root_v2(m)
        g = fb.geometry_for(m, rv2)
        pw = fw.params_for(m, g.bw_ft, g.sw_ft2, g.empty_wt_lb)
        env = [(float(s.speed_kts), float(s.h0_ft)) for s in fe.phase1_scenarios(m, sim)]
        fails, worst = [], {}
        pts = [dict(zip(pb2.B2_NAMES, c)) for c in itertools.product(*[x.lo_hi(m) for x in pb2.SHAPE_GENES_B2])]
        pts += [pb2.decode_shape_b2(rng.uniform(0, 1, pb2.N_SHAPE_B2), m) for _ in range(n_random)]
        t0 = time.process_time()
        for d in pts:
            r = pb2.geometry_gate_b2(pw, d, m, n_el=64, root_v2=rv2, envelope=env, geom=g)
            if not r.ok:
                fails.append({"genes": d, "reason": r.reason})
            for k, v in r.details.items():
                if isinstance(v, (int, float)):
                    lo, hi = worst.get(k, (math.inf, -math.inf))
                    worst[k] = (min(lo, v), max(hi, v))
        sec = pb2.SECTIONS[m]
        res["aircraft"][m] = {"n": len(pts), "n_fail": len(fails), "fails": fails[:20], "cpu_s": time.process_time() - t0,
                              "detail_ranges": worst, "trim_auth_limit": pb2.K_TRIM_AUTH * sec["cm_de_full"],
                              "tip_clear_limit_ft": sec["tip_clear_ft"]}
        print(m, len(pts), "fails", len(fails), {k: worst[k] for k in ("clmax_est", "dCm_total", "tip_clearance_ft",
                                                                       "effective_twist_max_deg", "camber_gradient_pct") if k in worst}, flush=True)
    json.dump(res, open(os.path.join(OUT, "p3b2a_gatescan.json"), "w"), indent=1, default=float)


def bench(reps=3):
    import coupled_sim as cs
    import flexbody as fb
    import flexbody_b1 as b1
    import flexbody_b2 as b2
    import flexeval as fe
    import flexeval_b1 as fb1
    import flexeval_b2 as fb2
    fb.blas_threads(1)
    sim = fe.load_sim()
    part = os.path.join(OUT, "p3b2a_bench.partial.jsonl")
    done = _done(part)
    CD = {"wing_dihedral_delta_deg": 1.0, "wing_camber_root_delta_pct": 0.5, "wing_camber_tip_delta_pct": 0.5}
    for m in MODELS:
        cs.ensure_root(m)
        fb.ensure_root_v2(m)
        b2.FlexBodyModelB2(m, shape_genes=MIXED)
        for rep in range(reps):
            key = f"{m}|{rep}"
            if key in done:
                continue
            rec = {"key": key, "model": m, "rep": rep, "load1": os.getloadavg()[0]}
            t = time.process_time(); b1.FlexBodyModelB1(m, shape_genes=SHAPED); rec["build_b1_shaped_s"] = time.process_time() - t
            t = time.process_time(); S = b2.FlexBodyModelB2(m, shape_genes=MIXED); rec["build_b2_mixed_s"] = time.process_time() - t
            t = time.process_time(); b2.margin_terms_b2(S); rec["preflight_b2_mixed_s"] = time.process_time() - t
            sc = fe.phase1_scenarios(m, sim)[:1]
            for lab, f in (("eval90_b1_baseline_s", lambda: fb1.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b1", root=cs.ROOT, sim=sim)),
                           ("eval90_b1_shaped_s", lambda: fb1.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b1", root=cs.ROOT, sim=sim, shape_genome=SHAPED)),
                           ("eval90_b2_default_s", lambda: fb2.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b2a", root=cs.ROOT, sim=sim)),
                           ("eval90_b2_cam_dih_s", lambda: fb2.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b2a", root=cs.ROOT, sim=sim, shape_genome=CD)),
                           ("eval90_b2_mixed_s", lambda: fb2.evaluate(_gains(m), {}, sc, m, fidelity="full_a1_b2a", root=cs.ROOT, sim=sim, shape_genome=MIXED))):
                t = time.process_time(); r = f(); rec[lab] = time.process_time() - t
                rec[lab[:-2] + "_status"] = r["status"]
            _append(part, rec)
            done[key] = rec
            print(key, {k: round(v, 3) for k, v in rec.items() if k.endswith("_s") and isinstance(v, float)}, flush=True)
    res = {"doc": "P3-B2a CPU: process CPU s, min over repeats, BLAS 1 thread, Phase-1 best gains, ONE 90 s scenario per "
                  "evaluate(). MIXED = " + json.dumps(MIXED) + " (thickness -> modal rebuild); cam_dih = " + json.dumps(CD) +
                  ". Box shared with ER's 8-worker pilot (load ~10 on 8 cores): process CPU, not wall.",
           "reps": reps, "aircraft": {}}
    for m in MODELS:
        rows = [v for k, v in done.items() if v["model"] == m]
        if not rows:
            continue
        mn = {k: min(r[k] for r in rows) for k in rows[0] if k.endswith("_s") and isinstance(rows[0][k], float)}
        b = mn["eval90_b1_baseline_s"]
        for k in ("eval90_b1_shaped_s", "eval90_b2_default_s", "eval90_b2_cam_dih_s", "eval90_b2_mixed_s"):
            mn[k[:-2] + "_ratio_vs_b1_baseline"] = mn[k] / b
        mn["load1_max"] = max(r["load1"] for r in rows)
        res["aircraft"][m] = mn
    json.dump(res, open(os.path.join(OUT, "p3b2a_benchmark.json"), "w"), indent=1, default=float)
    print(json.dumps(res["aircraft"], indent=1, default=float))


def calib():
    """Energy-term calibration from the gene sweeps (_scratch/p3b2/sweep_<M>.jsonl, Phase-1 best gains, 3 x 90 s)."""
    spec = json.load(open(os.path.join(OUT, "p3b2_gene_spec.json")))
    W = spec["energy_cost_recommendation"]["weights"]
    res = {"doc": "J_energy = w_E * max(0, energy_drag_increment); J_speed = w_E * 3 * max(0, deficit - 2 kt) / V_target. "
                  "dcost_off = B2a cost - baseline cost (no energy term); dcost_on adds J_energy + J_speed (baseline: both 0).",
           "weights": W, "aircraft": {}}
    md = ["| aircraft | probe | Δcost (off) | energy_drag_increment | J_energy | J_speed | Δcost (on) | main structural Δ |",
          "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        fn = os.path.join(SCR, f"sweep_{m}.jsonl")
        if not os.path.exists(fn):
            continue
        R = [json.loads(l) for l in open(fn)]
        b = next(r for r in R if r["label"] == "baseline")
        w = W[m]["w_E"]
        rows = {}
        for r in R:
            e = r.get("energy") or {}
            x = e.get("energy_drag_increment")
            dv = e.get("speed_deficit_kts_mean")
            vt = (r.get("energy_ps") or [{}])[0].get("v_target_kcas") if r.get("energy_ps") else None
            je = w * max(0.0, x) if isinstance(x, (int, float)) and math.isfinite(x) else 0.0
            js = w * 3 * max(0.0, dv - 2.0) / vt if (isinstance(dv, (int, float)) and vt) else 0.0
            d_off = r["cost"] - b["cost"]
            st = {k: r["terms"][k] - b["terms"][k] for k in r["terms"] if k.startswith("J_") and abs(r["terms"][k] - b["terms"][k]) > 1e-4}
            rows[r["label"]] = {"genes": r["genes"], "status": r["status"], "dcost_off": d_off, "x": x, "J_energy": je,
                                "J_speed": js, "dcost_on": d_off + je + js, "struct_delta": st,
                                "throttle_mean": e.get("throttle_mean"), "throttle_sat_frac": e.get("throttle_sat_frac"),
                                "energy_drag_ratio": e.get("energy_drag_ratio"), "speed_deficit_kts_mean": dv,
                                "speed_hold_ok": e.get("speed_hold_ok")}
            if r["label"] in ("tc_hi", "thick_both", "tc_lo", "cam_lo", "cam_hi", "cam_+0.5", "cam_-0.25", "dih+3", "dih+1"):
                top = ", ".join(f"{k} {v:+.4f}" for k, v in sorted(st.items(), key=lambda kv: -abs(kv[1]))[:2]) or "–"
                xs = f"{x:+.4f}" if isinstance(x, (int, float)) else "–"
                md.append(f"| {m} | {r['label']} | {d_off:+.4f} | {xs} | {je:.4f} | {js:.4f} | {d_off + je + js:+.4f} | {top} |")
        res["aircraft"][m] = {"baseline_cost": b["cost"], "w_E": w, "probes": rows}
    res["markdown"] = "\n".join(md)
    json.dump(res, open(os.path.join(OUT, "p3b2a_energy_calibration.json"), "w"), indent=1, default=float)
    print(res["markdown"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "versions"
    if cmd == "energyref":
        energyref()
    elif cmd == "versions":
        versions()
    elif cmd == "acceptance":
        acceptance(tuple(sys.argv[2:]) or MODELS)
    elif cmd == "gatescan":
        gatescan()
    elif cmd == "calib":
        calib()
    elif cmd == "bench":
        bench(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        raise SystemExit(__doc__)
