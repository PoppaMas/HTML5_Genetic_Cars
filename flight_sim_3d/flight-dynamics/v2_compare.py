"""v2 benchmarks and fidelity comparison (checkpointed; safe to re-run, finished items are skipped).

  python v2_compare.py bench   [models...]   -> v2_results/bench_<model>.jsonl  (wall per 90 s Phase-1 scenario)
  python v2_compare.py rank    <model> <set> -> v2_results/rank_<model>_<set>.jsonl  (set: genomes | joint)
  python v2_compare.py summarize             -> v2_benchmarks.json, spearman.json

rank sets (32 individuals each, seed fixed):
  genomes  Phase-1 best gains (evolution/runs/phase1-s1) + 32 random v2 genomes (uniform in [0,1]^12)
           -> rigid cost is the same for every individual (structure genes do not enter the rigid model)
  joint    32 random (gains, v2 genome) pairs: each gain = best x exp(U(-ln 1.5, ln 1.5)), clipped to the profile's
           gain_bounds (0 stays 0)
All runs: the Runner's Phase-1 profile and its 3 Phase-1 scenarios (90 s, scenario_seed 1), dt = 1/120, BLAS 1 thread.
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
N_RANK = 32


def _setup():
    import flexeval as fe
    import flexbody as fb
    fb.blas_threads(1)
    return fe, fb


def best_gains(model):
    import flexeval as fe
    S = json.load(open(os.path.join(fe.TEAM, "evolution", "runs", "phase1-s1", "summary.json")))["aircraft"]
    return next(a["best_gains"] for a in S if a["aircraft"] == model)


def _done(path):
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path):
        try:
            out.add(json.loads(line)["key"])
        except Exception:  # noqa: BLE001  (partial last line after a crash)
            pass
    return out


def _append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _slim(r):
    m = r.get("margins") or {}
    mg = {k: m.get(k) for k in ("flutter_margin", "div_margin", "reversal_margin")} if m else None
    if m and "blocks" in m:
        mg["blocks"] = {b: {k: v for k, v in d.items() if k.endswith("_margin")} for b, d in m["blocks"].items()}
    return {"cost": r["cost"], "status": r["status"], "terms": r["terms"], "model_version": r["model_version"],
            "margins": mg, "per_scenario_status": [p["status"] for p in r["per_scenario"]],
            "per_scenario_wall_s": [p.get("wall_s") for p in r["per_scenario"]],
            "mass_total_lb": (r.get("mass") or {}).get("total_lb")}


def bench(model, reps=2):
    fe, fb = _setup()
    sim = fe.load_sim()
    path = os.path.join(OUT, f"bench_{model}.jsonl")
    done = _done(path)
    scs = fe.phase1_scenarios(model, sim)
    g = best_gains(model)
    import coupled_sim as cs
    for rep in range(reps):
        for fid in ("rigid", "reduced", "full", "v1_flex_nbend2"):
            key = f"{fid}/{rep}"
            if key in done:
                continue
            t0, c0 = time.perf_counter(), time.process_time()
            if fid == "v1_flex_nbend2":     # v1 wing with 2 bending + torsion (the Runner's current 'full(v1)' stand-in)
                geom = fb.geometry_for(model)
                wing = fe.fw.FlexWing(fe.fw.params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb, n_bend=2))
                P = fe._profile_with_root(fe.default_profile(model, sim), sim, cs.ROOT, model)
                rr = []
                for sc in scs:
                    hook = fe.FlexHookV2("reduced", model, wing, fb.StructWeightsV2(), sim.DT, cs.ROOT, fe.root_v2_for(cs.ROOT), "v1")
                    rr.append(fe._simulate_with_hook(sim, g, sc, P, hook, False)[0])
                cost, status = float(np.mean([x["cost"] for x in rr])), rr[0]["status"]
                walls = [x.get("wall_s") for x in rr]
            else:
                r = fe.evaluate(g, None, scs, model, fidelity=fid, root=cs.ROOT)
                cost, status = r["cost"], r["status"]
                walls = [p.get("wall_s") for p in r["per_scenario"]]
            wall, cpu = time.perf_counter() - t0, time.process_time() - c0
            _append(path, {"key": key, "model": model, "fidelity": fid, "rep": rep, "wall_total_s": wall,
                           "cpu_total_s": cpu, "cpu_per_scen_s": cpu / len(scs),
                           "n_scen": len(scs), "scen_duration_s": [s.duration_s for s in scs],
                           "wall_per_scen_s": wall / len(scs), "sim_wall_s": walls, "cost": cost, "status": status,
                           "loadavg": os.getloadavg(), "t": time.strftime("%H:%M:%S")})
            print(model, key, f"wall {wall / len(scs):.2f} s/scen cpu {cpu / len(scs):.2f}", cost, status, flush=True)


def rank_items(model, which):
    fe, fb = _setup()
    rng = np.random.default_rng({"genomes": 101, "joint": 202}[which] + MODELS.index(model))
    g0 = best_gains(model)
    P = fe.default_profile(model)
    bounds = P.to_dict()["gain_bounds"]
    items = []
    for i in range(N_RANK):
        u = rng.random(len(fb.GENES_V2))
        if which == "joint":
            g = {}
            for k, v in g0.items():
                f = math.exp(rng.uniform(-math.log(1.5), math.log(1.5)))
                lo, hi = bounds[k]
                g[k] = 0.0 if v == 0.0 else float(min(max(v * f, lo), hi))
        else:
            g = dict(g0)
        items.append((i, g, u.tolist()))
    return items


def rank(model, which, shard=0, nshard=1):
    fe, fb = _setup()
    import coupled_sim as cs
    sim = fe.load_sim()
    path = os.path.join(OUT, f"rank_{model}_{which}.s{shard}.jsonl")
    done = _done(path)
    scs = fe.phase1_scenarios(model, sim)
    for i, g, u in rank_items(model, which):
        if i % nshard != shard:
            continue
        for fid in ("rigid", "reduced", "full"):
            key = f"{i}/{fid}"
            if key in done:
                continue
            if fid == "rigid" and which == "genomes" and i > 0:
                continue                        # identical for every genome (same gains); evaluated once (i = 0)
            t0 = time.perf_counter()
            r = fe.evaluate(g, u, scs, model, fidelity=fid, root=cs.ROOT)
            rec = {"key": key, "i": i, "fidelity": fid, "gains": g, "genome_u": u, "wall_s": time.perf_counter() - t0,
                   **_slim(r)}
            if fid == "reduced":
                rec["projection"] = {k: v for k, v in r["projection"].items() if not k.startswith("_")}
                rec["f_modes_hz"] = r["margins"]["f_modes_hz"]
            if fid == "full":
                M = fb.FlexBodyModel(model, u)
                rec["f_wingR_hz"] = M.frequencies_hz()["wingR"]
                rec["wing_cls"] = M.wingR.cls.tolist()
            _append(path, rec)
            print(model, which, key, f"{rec['wall_s']:.1f}s", r["cost"], r["status"], flush=True)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)

    def rk(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        for v in np.unique(x):                  # average ranks for ties
            m = x == v
            r[m] = r[m].mean()
        return r
    ra, rb = rk(a), rk(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _load_rank(model, which):
    recs = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith(f"rank_{model}_{which}.") and fn.endswith(".jsonl"):
            for line in open(os.path.join(OUT, fn)):
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                recs[r["key"]] = r
    return recs


def summarize():
    import flexeval as fe
    bench_out = {"units": "seconds per 90 s Phase-1 scenario (mean of the 3 scenarios; min over repeats); cpu = process CPU "
                          "time (robust to the shared box's load), wall = elapsed; ratios use cpu",
                 "note": "box shared with other agents (load average recorded per run); BLAS 1 thread", "models": {}}
    for m in MODELS:
        p = os.path.join(OUT, f"bench_{m}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(x) for x in open(p)]
        d = {}
        for fid in ("rigid", "reduced", "full", "v1_flex_nbend2"):
            rr = [r for r in rows if r["fidelity"] == fid]
            if rr:
                d[fid] = {"wall_per_scen_s": min(r["wall_per_scen_s"] for r in rr),
                          "cpu_per_scen_s": min(r.get("cpu_per_scen_s", math.nan) for r in rr),
                          "all_reps": [r["wall_per_scen_s"] for r in rr], "status": rr[0]["status"], "cost": rr[0]["cost"],
                          "loadavg_1min": [r["loadavg"][0] for r in rr]}
        for a, b in (("full", "rigid"), ("full", "v1_flex_nbend2"), ("full", "reduced"), ("reduced", "rigid")):
            if a in d and b in d:
                d[f"ratio_{a}_over_{b}"] = d[a]["cpu_per_scen_s"] / d[b]["cpu_per_scen_s"]
        bench_out["models"][m] = d
    with open(os.path.join(HERE, "v2_benchmarks.json"), "w") as f:
        json.dump(bench_out, f, indent=1)
    sp = {"method": "Spearman rank correlation of the genome total cost (3 Phase-1 scenarios x 90 s, c.f. v2_compare.py "
                    "docstring); ties -> average ranks; fail_cost genomes included (ties) and, separately, excluded",
          "sets": {}}
    for m in MODELS:
        for which in ("genomes", "joint"):
            R = _load_rank(m, which)
            if not R:
                continue
            idx = sorted({r["i"] for r in R.values()})
            full = {i: R[f"{i}/full"] for i in idx if f"{i}/full" in R}
            red = {i: R[f"{i}/reduced"] for i in idx if f"{i}/reduced" in R}
            rig = {i: (R.get(f"{i}/rigid") or R.get("0/rigid")) for i in idx}
            both = [i for i in idx if i in full and i in red and rig.get(i)]
            fc = [full[i]["cost"] for i in both]
            res = {"n": len(both),
                   "spearman_reduced_vs_full": _spearman([red[i]["cost"] for i in both], fc),
                   "spearman_rigid_vs_full": _spearman([rig[i]["cost"] for i in both], fc)}
            ok = [i for i in both if full[i]["status"] == "ok" and red[i]["status"] == "ok"]
            res["n_both_ok"] = len(ok)
            res["spearman_reduced_vs_full_both_ok"] = _spearman([red[i]["cost"] for i in ok], [full[i]["cost"] for i in ok])
            res["spearman_rigid_vs_full_both_ok"] = _spearman([rig[i]["cost"] for i in ok], [full[i]["cost"] for i in ok])
            res["status_full"] = {s: sum(full[i]["status"] == s for i in both) for s in {full[i]["status"] for i in both}}
            res["status_reduced"] = {s: sum(red[i]["status"] == s for i in both) for s in {red[i]["status"] for i in both}}
            res["gate_agreement"] = float(np.mean([(full[i]["status"] == "ok") == (red[i]["status"] == "ok") for i in both])) if both else None
            # margins (reduced v1 vs full v2), first-mode frequencies
            fm = [(red[i]["margins"]["flutter_margin"], full[i]["margins"]["flutter_margin"]) for i in both if red[i]["margins"] and full[i]["margins"]]
            dm = [(red[i]["margins"]["div_margin"], full[i]["margins"]["div_margin"]) for i in both if red[i]["margins"] and full[i]["margins"]]
            if fm:
                a = np.array(fm); b = np.array(dm)
                wing_fl = np.array([full[i]["margins"]["blocks"]["wingR"]["flutter_margin"] for i in both])
                res["margins"] = {
                    "flutter_reduced_minus_full_overall": {"mean": float(np.mean(a[:, 0] - a[:, 1])), "max_abs": float(np.max(np.abs(a[:, 0] - a[:, 1])))},
                    "flutter_reduced_minus_full_wingR_block": {"mean": float(np.mean(a[:, 0] - wing_fl)), "max_abs": float(np.max(np.abs(a[:, 0] - wing_fl)))},
                    "spearman_flutter_reduced_vs_full_wingR": _spearman(a[:, 0], wing_fl),
                    "div_reduced_minus_full": {"mean": float(np.mean(b[:, 0] - b[:, 1])), "max_abs": float(np.max(np.abs(b[:, 0] - b[:, 1])))},
                    "sample": [{"i": i, "reduced": {k: red[i]["margins"][k] for k in ("flutter_margin", "div_margin")},
                                "full": {k: full[i]["margins"][k] for k in ("flutter_margin", "div_margin", "reversal_margin")},
                                "full_wingR_flutter": full[i]["margins"]["blocks"]["wingR"]["flutter_margin"]} for i in both[:8]]}
            fe1 = []
            for i in both:
                if "f_modes_hz" in red[i] and "f_wingR_hz" in full[i]:
                    cls = full[i]["wing_cls"]
                    fb1 = full[i]["f_wingR_hz"][cls.index("b")]
                    ft1 = full[i]["f_wingR_hz"][cls.index("t")]
                    fe1.append((100 * (red[i]["f_modes_hz"][0] - fb1) / fb1, 100 * (red[i]["f_modes_hz"][1] - ft1) / ft1))
            if fe1:
                e = np.array(fe1)
                res["first_mode_freq_error_reduced_vs_full_pct"] = {
                    "bending1": {"mean": float(e[:, 0].mean()), "max_abs": float(np.abs(e[:, 0]).max())},
                    "torsion1": {"mean": float(e[:, 1].mean()), "max_abs": float(np.abs(e[:, 1]).max())}}
            res["wall_s_mean"] = {fid: float(np.mean([R[k]["wall_s"] for k in R if k.endswith("/" + fid)])) for fid in ("rigid", "reduced", "full")}
            sp["sets"][f"{m}/{which}"] = res
    with open(os.path.join(HERE, "spearman.json"), "w") as f:
        json.dump(sp, f, indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "spearman_reduced_vs_full", "spearman_rigid_vs_full", "n_both_ok",
                                                "spearman_reduced_vs_full_both_ok", "gate_agreement")} for k, v in sp["sets"].items()}, indent=1))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "bench":
        for m in (sys.argv[2:] or MODELS):
            bench(m)
    elif cmd == "rank":
        rank(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0, int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    elif cmd == "summarize":
        summarize()
