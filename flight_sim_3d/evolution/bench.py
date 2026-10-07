#!/usr/bin/env python3
"""Benchmark driver: fresh run + cache rerun (+ optional sequential-schedule run) per config.

    cd flight_sim_3d              # (team layout: cd <team folder>)
    python -m evolution.bench --tag T evolution/configs/bench_baseline.json evolution/configs/bench_ic.json ...

For each config:
  1. fresh run   run_id <stem>-<T>        with a NEW cache file cache/bench-<T>-<stem>.sqlite
  2. rerun       run_id <stem>-<T>-rerun  same config + same cache  -> cache hit rate, must be identical
  3. --schedules run_id <stem>-<T>-seq    cache off, schedule=sequential (aircraft one after another,
                                          individuals in parallel) -> wall-time comparison
Writes bench_results/<T>/results.json and table.md. Every number comes from the runs' summary/history.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import batch

PKG = os.path.dirname(os.path.abspath(__file__))


def _run(user, stem, schedule=None, quiet=True):
    cfg = batch.resolve_config(user, stem)
    b = batch.Batch(cfg, log=(lambda *a, **k: None) if quiet else print)
    if schedule:
        b.schedule = schedule
    t0 = time.perf_counter()
    s = b.run()
    return s, b.run_dir, time.perf_counter() - t0


def _result_view(run_dir):
    out = {}
    d = os.path.join(run_dir, "checkpoints")
    for f in sorted(os.listdir(d)):
        ck = json.load(open(os.path.join(d, f)))
        out[f] = (ck["pop"], ck["best_per_gen"], [(h["best"], h["mean"], h["median"], h["std"]) for h in ck["history"]])
    return out


def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def row_for(a, rerun_a, seq_wall=None):
    if "skipped" in a:
        return {"aircraft": a["aircraft"], "profile": a["profile"], "skipped": a["skipped"]}
    ms = a["final_metrics"]
    tot = a["this_session"]
    hist_tot = a["totals_from_history"]
    r_tot = rerun_a.get("this_session", {}) if rerun_a else {}
    r_hits = r_tot.get("cache_hits_mem", 0) + r_tot.get("cache_hits_disk", 0)
    return {
        "aircraft": a["aircraft"], "profile": a["profile"],
        "trim": {k: a["preflight"].get(k) for k in ("vc_kts", "h_sl_ft", "alpha_deg", "throttle_trim", "pitch_trim_cmd")},
        "best_fitness": a["best_fitness"], "gen0_best_fitness": a["gen0_best_fitness"],
        "hold_rms_ft_calm": ms[0]["hold_rms_err_ft"], "hold_rms_ft_mean": float(np.mean([m["hold_rms_err_ft"] for m in ms])),
        "hold_max_abs_ft": max(m["hold_max_abs_err_ft"] for m in ms),
        "rms_ft_mean_incl_steps": float(np.mean([m["rms_err_ft"] for m in ms])),
        "overshoot_ft_max": max(m["overshoot_ft_max"] for m in ms),
        "settle_s_max": None if any(m["settle_s_max"] is None for m in ms) else max(m["settle_s_max"] for m in ms),
        "final_status": [m["status"] for m in ms],
        "invalid_rate_gen0": a["invalid_rate_gen0"], "invalid_rate_all_gens": a["invalid_rate_all_gens"],
        "crash_rate_all_gens": a["crash_rate_all_gens"], "status_counts_all_gens": a["status_counts_all_gens"],
        "sims_computed": tot["sims_computed"], "unique_sims": tot["unique_sims"],
        "cpu_s_per_sim": tot["eval_cpu_s"] / tot["sims_computed"] if tot["sims_computed"] else None,
        "aircraft_wall_s": a["evolve_wall_s_this_session"],
        "sims_per_s_aircraft": tot["sims_computed"] / a["evolve_wall_s_this_session"],
        "rerun_cache_hit_rate": (r_hits / r_tot["unique_sims"]) if r_tot.get("unique_sims") else None,
        "rerun_sims_computed": r_tot.get("sims_computed"),
        "genes_at_bound": a["genes_at_bound"], "best_gains": a["best_gains"], "gain_bounds": a["gain_bounds"],
        "trajectories": [e["file"] for e in a["trajectories"]],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--schedules", action="store_true", help="also time schedule=sequential (cache off)")
    a = ap.parse_args(argv)
    out_dir = os.path.join(PKG, "bench_results", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    results = {"tag": a.tag, "configs": []}
    for cpath in a.configs:
        stem = os.path.splitext(os.path.basename(cpath))[0]
        user = json.load(open(cpath))
        cache_path = os.path.join(PKG, "cache", f"bench-{a.tag}-{stem}.sqlite")
        if os.path.exists(cache_path):
            sys.exit(f"{cache_path} exists; use a new --tag for a clean benchmark")
        user["cache"] = {"enabled": True, "path": cache_path}
        print(f"== {stem}: fresh run", flush=True)
        s1, d1, w1 = _run(dict(user, run_id=f"{stem}-{a.tag}"), stem, quiet=False)
        print(f"== {stem}: cache rerun", flush=True)
        s2, d2, w2 = _run(dict(user, run_id=f"{stem}-{a.tag}-rerun"), stem)
        identical = _result_view(d1) == _result_view(d2)
        entry = {"config": cpath, "run_id": s1["run_id"], "run_dir": d1, "wall_s": s1["wall_s_this_session"],
                 "sims_computed": s1["this_session"]["sims_computed"], "sims_per_s": s1["this_session"]["sims_per_s"],
                 "workers": s1["workers"], "loadavg_start": s1.get("loadavg_1_5_15_start"),
                 "loadavg_end": s1.get("loadavg_1_5_15_end"), "rerun_run_id": s2["run_id"], "rerun_wall_s": s2["wall_s_this_session"],
                 "rerun_cache_hit_rate": s2["this_session"]["cache_hit_rate"],
                 "rerun_sims_computed": s2["this_session"]["sims_computed"], "rerun_identical": identical}
        seq = None
        if a.schedules:
            print(f"== {stem}: sequential schedule, cache off", flush=True)
            u3 = dict(user, run_id=f"{stem}-{a.tag}-seq", cache={"enabled": False, "path": cache_path})
            s3, d3, w3 = _run(u3, stem, schedule="sequential")
            entry.update(seq_run_id=s3["run_id"], seq_wall_s=s3["wall_s_this_session"],
                         seq_loadavg_start=s3.get("loadavg_1_5_15_start"), seq_loadavg_end=s3.get("loadavg_1_5_15_end"),
                         seq_sims_computed=s3["this_session"]["sims_computed"], seq_identical=_result_view(d1) == _result_view(d3))
        r2 = {x["aircraft"]: x for x in s2["aircraft"]}
        entry["aircraft"] = [row_for(x, r2.get(x["aircraft"])) for x in s1["aircraft"]]
        results["configs"].append(entry)
        json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
        print(f"   wall {entry['wall_s']:.1f}s, rerun {entry['rerun_wall_s']:.1f}s hit {entry['rerun_cache_hit_rate']:.3f} identical={identical}"
              + (f", sequential {entry['seq_wall_s']:.1f}s identical={entry['seq_identical']}" if a.schedules else ""), flush=True)
    write_table(results, os.path.join(out_dir, "table.md"))
    print(open(os.path.join(out_dir, "table.md")).read())


def write_table(results, path):
    L = [f"# Benchmark {results['tag']}", ""]
    for c in results["configs"]:
        L += [f"## {os.path.basename(c['config'])}  (run `{c['run_id']}`)", "",
              f"batch wall {c['wall_s']:.1f} s on {c['workers']} workers, {c['sims_computed']} sims computed "
              f"({c['sims_per_s']:.1f} sims/s); cache rerun: {c['rerun_wall_s']:.1f} s, hit rate {c['rerun_cache_hit_rate']:.3f}, "
              f"{c['rerun_sims_computed']} sims computed, identical results: {c['rerun_identical']}"
              + (f"; box load avg (1 min) at start/end {c['loadavg_start'][0]:.1f}/{c['loadavg_end'][0]:.1f}" if c.get("loadavg_start") else "")
              + (f"; sequential schedule (cache off): {c['seq_wall_s']:.1f} s, identical: {c['seq_identical']}" if "seq_wall_s" in c else ""), "",
              "| aircraft | profile | best fitness (gen0 -> final) | hold RMS ft (calm / mean) | hold max abs ft | overshoot ft | settle s (+-20 ft) | invalid gen0 / all gens | sims | cpu s/sim | aircraft wall s | sims/s | rerun hit rate | genes at bound |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in c["aircraft"]:
            if "skipped" in r:
                L.append(f"| {r['aircraft']} | {r['profile']} | **not run: {r['skipped']}** | | | | | | | | | | | |")
                continue
            L.append(f"| {r['aircraft']} | {r['profile']} | {r['gen0_best_fitness']:.4f} -> **{r['best_fitness']:.4f}** | "
                     f"{r['hold_rms_ft_calm']:.2f} / {r['hold_rms_ft_mean']:.2f} | {r['hold_max_abs_ft']:.2f} | {r['overshoot_ft_max']:.1f} | "
                     f"{_f(r['settle_s_max'], 1)} | {r['invalid_rate_gen0']:.3f} / {r['invalid_rate_all_gens']:.3f} | {r['sims_computed']} | "
                     f"{_f(r['cpu_s_per_sim'], 3)} | {r['aircraft_wall_s']:.1f} | {r['sims_per_s_aircraft']:.1f} | {_f(r['rerun_cache_hit_rate'], 3)} | "
                     f"{', '.join(f'{k}@{v}' for k, v in r['genes_at_bound'].items()) or '-'} |")
        L.append("")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
