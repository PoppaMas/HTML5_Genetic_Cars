"""P3-A1 studies (checkpointed; safe to re-run, finished items are skipped).

  python p3a1_study.py truncation   -> v2_results/p3a1_truncation.json   (modal truncation + strip convergence)
  python p3a1_study.py bench [reps] -> v2_results/p3a1_benchmark.json    (CPU per 90 s scenario, A1 vs full)
  python p3a1_study.py versions     -> v2_results/model_versions_post_p3a1.json

Truncation: baseline genome and tip-soft genome (wing_ei_taper_4 = 0.75, others baseline), all 4 aircraft. For the A1 model
(64 strips, 4b + 3t + 2ip) and each variant (+1 / +2 modes per family, +1 every family, +1 / +2 next-lowest of any type):
right-wing static aeroelastic root BM / eta 0.875 BM / tip deflection (0.9 V_D, 1 deg + 1 g), 1-cos gust peaks of the same,
in-plane 1 g fore-aft static tip deflection and 1-cos n_x pulse peak root in-plane moment / tip deflection, flutter speed (searched to 4 V_D) and the gate's wing-block flutter / divergence / aileron-reversal margins; % vs N.
Strip convergence: the same metrics with the A1 mode set at 32 / 48 / 64 / 96 / 128 strips (% vs 128).
Bench: exactly v2_compare.py bench's method (fe.evaluate with the Phase-1 best gains, baseline genome, the 3 Phase-1 90 s
scenarios, process CPU / 3, min over repeats, BLAS 1 thread), 'full' and 'full_a1' interleaved per repeat.
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
MODELS = ("c172x", "T38", "737", "f16")
GENOMES = {"baseline": {}, "tip_soft": {"wing_ei_taper_4": 0.75}}
STRIPS = (32, 48, 64, 96, 128)
THRESH_PCT = 2.0


def _done(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                out[r["key"]] = r
            except Exception:  # noqa: BLE001  (partial last line after a crash)
                pass
    return out


def _append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _jsonable(x):
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _sizing_metrics(mdl):
    """J_wing_tip_bm_limit / ratio both ways: flexbody.sizing_v2 (strip-discrete, what 'full' uses) and
    flexbody_a1.sizing_a1 (station-exact, what 'full_a1' uses); root J_wing_bm_limit."""
    import flexbody as fb
    import flexbody_a1 as a1
    sd, se = fb.sizing_v2(mdl), a1.sizing_a1(mdl)
    return {"J_wing_tip_bm_limit_strip_discrete": sd["terms"]["J_wing_tip_bm_limit"],
            "ratio_tip_bm_strip_discrete": sd["ratios"]["wingR_tip_bm"],
            "J_wing_tip_bm_limit": se["terms"]["J_wing_tip_bm_limit"], "ratio_tip_bm": se["ratios"]["wingR_tip_bm"],
            "J_wing_bm_limit": se["terms"]["J_wing_bm_limit"],
            "sizing_terms_sum": float(sum(se["terms"].values()))}


def truncation():
    import flexbody as fb
    import flexbody_a1 as a1
    fb.blas_threads(1)
    part = os.path.join(OUT, "p3a1_truncation.partial.jsonl")
    done = _done(part)
    for m in MODELS:
        fb.ensure_root_v2(m)
        for gname, g in GENOMES.items():
            key = f"trunc/{m}/{gname}"
            if key not in done:
                c0 = time.process_time()
                r = a1.truncation_check_a1(m, g)
                rec = {"key": key, "model": m, "genome": gname, "result": _jsonable(r), "cpu_s": time.process_time() - c0}
                _append(part, rec)
                done[key] = rec
                print(key, f"max |d| {r['max_abs_pct']:.3f} %", flush=True)
            for n in STRIPS:
                key = f"strips/{m}/{gname}/{n}"
                if key in done:
                    continue
                mdl = a1.FlexBodyModelA1(m, g, wing_mesh={"n_el": n})
                met = a1.wing_metrics_a1(mdl)
                met.update(_sizing_metrics(mdl))
                mg = fb.margins_v2(mdl)
                met.update(flutter_margin_all_blocks=mg["flutter_margin"], reversal_margin_all_blocks=mg["reversal_margin"])
                rec = {"key": key, "model": m, "genome": gname, "n_el": n, "metrics": _jsonable(met)}
                _append(part, rec)
                done[key] = rec
                print(key, flush=True)
            # reference: the current 'full' model (32 strips, 3b + 2t + 1ip) on the same metrics
            key = f"full_ref/{m}/{gname}"
            if key not in done:
                mdl = fb.FlexBodyModel(m, g)
                met = a1.wing_metrics_a1(mdl)
                met.update(_sizing_metrics(mdl))
                mg = fb.margins_v2(mdl)
                met.update(flutter_margin_all_blocks=mg["flutter_margin"], reversal_margin_all_blocks=mg["reversal_margin"])
                rec = {"key": key, "model": m, "genome": gname, "metrics": _jsonable(met)}
                _append(part, rec)
                done[key] = rec
    summarize_truncation(done)


def summarize_truncation(done=None):
    import flexbody_a1 as a1
    done = done or _done(os.path.join(OUT, "p3a1_truncation.partial.jsonl"))
    res = {"doc": __doc__.split("Bench:")[0].strip(), "threshold_pct": THRESH_PCT,
           "a1_wing": a1.WING_A1, "genomes": GENOMES, "models": {}}
    worst_all = {}
    for m in MODELS:
        res["models"][m] = {}
        for gname in GENOMES:
            t = done[f"trunc/{m}/{gname}"]["result"]
            pct = t["pct_vs_N"]
            fam = {}
            for fname, vs in (("bending", ("b+1", "b+2")), ("torsion", ("t+1", "t+2")), ("in_plane", ("ip+1", "ip+2")),
                              ("all_families_+1", ("all+1",)), ("total_next_lowest", ("next+1", "next+2"))):
                fam[fname] = {v: {"max_abs_pct": max(abs(float(x)) for x in pct[v].values()),
                                  "flutter_speed_pct": pct[v]["flutter_keas_min_4VD"],
                                  "wing_flutter_margin_pct": pct[v]["wing_flutter_margin"],
                                  "root_bm_static_pct": pct[v]["static_root_bm"], "root_bm_gust_pct": pct[v]["gust_peak_root_bm"],
                                  "tip_bm_eta0875_static_pct": pct[v]["static_tip_bm_eta"],
                                  "tip_bm_eta0875_gust_pct": pct[v]["gust_peak_tip_bm_eta"],
                                  "ip_root_bm_pulse_pct": pct[v]["ip_pulse_peak_root_bm"],
                                  "ip_tip_static_pct": pct[v]["ip_static_tip_v"],
                                  "n_wing_modes": t["values"][v]["n_wing_modes"], "n_sel": t["values"][v]["n_sel"]} for v in vs}
            worst = max(abs(float(x)) for d in pct.values() for x in d.values())
            worst_all[f"{m}/{gname}"] = worst
            strips = {}
            ref = done[f"strips/{m}/{gname}/{STRIPS[-1]}"]["metrics"]
            for n in STRIPS:
                met = done[f"strips/{m}/{gname}/{n}"]["metrics"]
                strips[str(n)] = {k: a1.pct_delta(float(ref[k]), float(met[k])) for k in a1.SCALAR_METRICS}
                strips[str(n)]["f_wing1_hz"] = met["f_wing_hz"][0]
                strips[str(n)]["J_wing_tip_bm_limit_station_exact"] = met["J_wing_tip_bm_limit"]
                strips[str(n)]["J_wing_tip_bm_limit_strip_discrete"] = met["J_wing_tip_bm_limit_strip_discrete"]
                strips[str(n)]["max_abs_pct_vs_128"] = max(abs(float(strips[str(n)][k])) for k in a1.SCALAR_METRICS)
            fr = done[f"full_ref/{m}/{gname}"]["metrics"]
            a1v = t["values"]["N"]
            res["models"][m][gname] = {
                "N_wing_modes": t["values"]["N"]["n_wing_modes"], "values_N": t["values"]["N"],
                "pct_vs_N_by_family": fam, "pct_vs_N_raw": pct, "max_abs_pct": worst, "passes_2pct": worst <= THRESH_PCT,
                "strip_convergence_pct_vs_128": strips,
                "full_v2_reference": {"values": fr, "a1_minus_full_pct": {k: a1.pct_delta(float(fr[k]), float(a1v[k]))
                                                                          for k in a1.SCALAR_METRICS}},
                "J_wing_tip_bm_limit": {
                    "full_a1 (64 strips, station-exact)": done[f"strips/{m}/{gname}/64"]["metrics"]["J_wing_tip_bm_limit"],
                    "full (32 strips, strip-discrete, as flown by 'full')": fr["J_wing_tip_bm_limit_strip_discrete"],
                    "full model, station-exact (diagnostic)": fr["J_wing_tip_bm_limit"],
                    "a1 model, strip-discrete (diagnostic)":
                        done[f"strips/{m}/{gname}/64"]["metrics"]["J_wing_tip_bm_limit_strip_discrete"]},
                "a1_sizing_terms_sum": done[f"strips/{m}/{gname}/64"]["metrics"]["sizing_terms_sum"]}
    res["max_abs_pct_by_case"] = worst_all
    res["all_pass_2pct"] = all(v <= THRESH_PCT for v in worst_all.values())
    with open(os.path.join(OUT, "p3a1_truncation.json"), "w") as f:
        json.dump(_jsonable(res), f, indent=1)
    print(json.dumps(worst_all, indent=1), "all pass:", res["all_pass_2pct"])


def bench(reps=3):
    import flexeval as fe
    import flexeval_a1 as fa
    import flexbody as fb
    import coupled_sim as cs
    import v2_compare as vc
    fb.blas_threads(1)
    sim = fe.load_sim()
    part = os.path.join(OUT, "p3a1_bench.partial.jsonl")
    done = _done(part)
    for m in MODELS:
        scs = fe.phase1_scenarios(m, sim)
        g = vc.best_gains(m)
        for rep in range(reps):
            for fid in ("full", "full_a1"):
                key = f"{m}/{fid}/{rep}"
                if key in done:
                    continue
                t0, c0 = time.perf_counter(), time.process_time()
                r = fa.evaluate(g, None, scs, m, fidelity=fid, root=cs.ROOT)
                wall, cpu = time.perf_counter() - t0, time.process_time() - c0
                rec = {"key": key, "model": m, "fidelity": fid, "rep": rep, "cpu_total_s": cpu, "wall_total_s": wall,
                       "cpu_per_scen_s": cpu / len(scs), "wall_per_scen_s": wall / len(scs), "n_scen": len(scs),
                       "scen_duration_s": [s.duration_s for s in scs], "cost": r["cost"], "status": r["status"],
                       "model_version": r["model_version"], "loadavg": os.getloadavg(), "t": time.strftime("%H:%M:%S")}
                _append(part, rec)
                done[key] = rec
                print(key, f"cpu {cpu / len(scs):.3f} s/scen wall {wall / len(scs):.3f}", r["cost"], r["status"], flush=True)
    out = {"units": "seconds per 90 s Phase-1 scenario (whole fe.evaluate incl. model build + margin screen, mean of the 3 "
                    "Phase-1 scenarios; min over repeats); cpu = process CPU time, wall = elapsed; ratio uses cpu",
           "method": "same as v2_compare.py bench (best Phase-1 gains, baseline struct genome, dt 1/120, BLAS 1 thread); "
                     "full and full_a1 interleaved per repeat in one process",
           "reps": reps, "models": {}}
    for m in MODELS:
        d = {}
        for fid in ("full", "full_a1"):
            rr = [done[f"{m}/{fid}/{k}"] for k in range(reps)]
            best = min(rr, key=lambda x: x["cpu_per_scen_s"])
            d[fid] = {"cpu_per_scen_s": best["cpu_per_scen_s"], "wall_per_scen_s": best["wall_per_scen_s"],
                      "all_cpu_per_scen_s": [x["cpu_per_scen_s"] for x in rr], "cost": best["cost"], "status": best["status"],
                      "model_version": best["model_version"], "loadavg_1min": [x["loadavg"][0] for x in rr]}
        d["ratio_a1_over_full_cpu"] = d["full_a1"]["cpu_per_scen_s"] / d["full"]["cpu_per_scen_s"]
        ref = json.load(open(os.path.join(HERE, "v2_benchmarks.json")))["models"][m]
        d["v2_benchmarks_json_reference"] = {k: ref[k]["cpu_per_scen_s"] for k in ("rigid", "reduced", "full", "v1_flex_nbend2")}
        out["models"][m] = d
    with open(os.path.join(OUT, "p3a1_benchmark.json"), "w") as f:
        json.dump(out, f, indent=1)
    for m, d in out["models"].items():
        print(m, f"full {d['full']['cpu_per_scen_s']:.3f}  A1 {d['full_a1']['cpu_per_scen_s']:.3f}  x{d['ratio_a1_over_full_cpu']:.3f}")


def bench_build(reps=5):
    """Pre-flight CPU per genome (model build + margin_terms incl. sizing; once per evaluate(), amortised over the
    scenarios): full vs A1 at 48 / 64 strips, min over reps. Merged into p3a1_benchmark.json['preflight_cpu_s']."""
    import flexbody as fb
    import flexbody_a1 as a1
    fb.blas_threads(1)
    res = {}
    for m in MODELS:
        d = {}
        for nm, mk, mt in (("full", lambda: fb.FlexBodyModel(m), fb.margin_terms_v2),
                           ("full_a1_48strips", lambda: a1.FlexBodyModelA1(m, wing_mesh={"n_el": 48}), a1.margin_terms_a1),
                           ("full_a1_64strips", lambda: a1.FlexBodyModelA1(m), a1.margin_terms_a1)):
            tb, tm = [], []
            for _ in range(reps):
                c0 = time.process_time()
                mdl = mk()
                c1 = time.process_time()
                mt(mdl)
                c2 = time.process_time()
                tb.append(c1 - c0)
                tm.append(c2 - c1)
            d[nm] = {"build_cpu_s": min(tb), "margins_sizing_cpu_s": min(tm), "total_cpu_s": min(tb) + min(tm)}
        res[m] = d
        print(m, {k: round(v["total_cpu_s"], 3) for k, v in d.items()}, flush=True)
    path = os.path.join(OUT, "p3a1_benchmark.json")
    out = json.load(open(path))
    out["preflight_cpu_s"] = {"doc": "per genome, once per evaluate() call (included in the per-scenario numbers above, "
                                     "divided by 3); min over %d reps" % reps, "models": res}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def versions():
    import flexeval_a1 as fa
    import coupled_sim as cs
    out = {"doc": "P3-A1 model_version strings (flexeval_a1.model_version). 'full_a1' is new; rigid / reduced / full are "
                  "listed for reference and are byte-identical to model_versions_post_p25.json (unchanged by P3-A1).",
           "a1_wing": fa.fba1.WING_A1}
    pin = json.load(open(os.path.join(OUT, "model_versions_post_p25.json")))
    for m in MODELS:
        d = {f: fa.model_version(f, m, cs.ROOT) for f in ("rigid", "reduced", "full", "full_a1")}
        assert {k: d[k] for k in ("rigid", "reduced", "full")} == pin[m], (m, d, pin[m])
        out[m] = d
    with open(os.path.join(OUT, "model_versions_post_p3a1.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "truncation"
    if cmd == "truncation":
        truncation()
    elif cmd == "summarize_truncation":
        summarize_truncation()
    elif cmd == "bench":
        bench(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == "bench_build":
        bench_build()
    elif cmd == "versions":
        versions()
    else:
        raise SystemExit(__doc__)
