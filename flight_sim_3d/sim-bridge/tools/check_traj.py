#!/usr/bin/env python3
"""Physics / plausibility sanity report for ga-flightsim-traj/1 runs (read-only; numpy only).

    python tools/check_traj.py ../evolution/runs/bench_jets-j1 [--json out.json]
    python tools/check_traj.py runs/phase1-s1 runs/phase1-s2 runs/phase1-s3 --baseline runs/bench_jets-j1

Several runs: per-run tables, then a cross-seed section (gen-19 fitness spread, gain spread, genes at
bounds, g9->g19 stall) and, with --baseline, a per-aircraft comparison against an older run.
Target channels are told apart by shape like the viewer does: a piecewise-constant one is the step
command, a continuous one (e.g. phase 1's 600 fpm ramp in target_alt_m) is the ramp reference.

Per trajectory: status, NaNs, trim point (target speed/alt vs t=0 state), altitude tracking
(max/RMS error, hold-phase error, overshoot, settle time in a 20 ft band and in a 5%-of-step band),
airspeed excursion vs trim, attitude / load-factor ranges, throttle/elevator saturation,
quaternion vs Euler agreement, position vs integrated velocity. Per aircraft: fitness ordering
over generations, duplicate trajectories. Prints a table and a list of flags.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

FT = 0.3048
M_EN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def quat_to_matrix(q):
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def euler_ned(phi, th, psi):
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
    return np.stack([
        np.stack([ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp], -1),
        np.stack([ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp], -1),
        np.stack([-st, sf * ct, cf * ct], -1)], -2)


def split_targets(doc, d, ch, t, steps):
    """-> (step_cmd_m, ramp_ref_m or None, ramp channel name)."""
    step, ramp, ramp_name = None, None, None
    for name in ("target_alt_m", "alt_target_m", "target_cmd_alt_m", "alt_cmd_m", "target_ref_alt_m"):
        if name not in ch:
            continue
        a = d[:, ch[name]]
        rng = float(np.nanmax(a) - np.nanmin(a))
        jump = float(np.nanmax(np.abs(np.diff(a)))) if len(a) > 1 else 0.0
        if rng > 1 and jump < max(5.0, 0.25 * rng):
            if ramp is None:
                ramp, ramp_name = a, name
        elif step is None:
            step = a
    if step is None and steps:
        step = np.array([[s["alt_m"] for s in steps if s["t"] <= tt][-1] for tt in t])
    return step, ramp, ramp_name


def analyse(doc, throttle_max=None):
    ch = {c: i for i, c in enumerate(doc["channels"])}
    d = np.array(doc["data"], dtype=float)
    c = lambda n: d[:, ch[n]] if n in ch else None  # noqa: E731
    t = c("t")
    r = {"aircraft": doc["aircraft"], "gen": doc["generation"], "fitness": doc["fitness"],
         "scenario_cost": doc.get("scenario_cost"), "status": doc.get("status"), "n": len(t), "t_end": float(t[-1]),
         "nan": int(np.sum(~np.isfinite(d)))}
    tgt = doc.get("target", {})
    steps = tgt.get("steps") or tgt.get("schedule") or []
    r["trim_kcas"] = tgt.get("speed_kcas", tgt.get("speed_kts"))
    r["trim_alt_ft"] = (steps[0]["alt_m"] if steps else tgt.get("alt_m", float("nan"))) / FT
    r["step_ft"] = (max(s["alt_m"] for s in steps) - min(s["alt_m"] for s in steps)) / FT if steps else 0.0
    start = next((e["detail"] for e in doc.get("events", []) if e["type"] == "start"), "")
    r["start_event"] = start
    alt = c("alt_msl_m") / FT
    ta, ramp, r["ramp_channel"] = split_targets(doc, d, ch, t, steps)
    ta = ta / FT
    if ramp is not None:
        er = alt - ramp / FT
        r["ramp_err_max_ft"] = float(np.max(np.abs(er)))
        r["ramp_err_rms_ft"] = float(np.sqrt(np.mean(er ** 2)))
        moving = np.abs(np.gradient(ramp, t)) > 0.05
        r["ramp_err_moving_max_ft"] = float(np.max(np.abs(er[moving]))) if moving.any() else None
        r["ramp_rate_fpm"] = float(np.max(np.abs(np.gradient(ramp, t))) / FT * 60)
    e = alt - ta
    r["alt0_err_ft"] = float(alt[0] - r["trim_alt_ft"])
    kcas = c("kcas")
    if kcas is not None:
        r["kcas0"] = float(kcas[0])
        r["kcas_min"], r["kcas_max"] = float(kcas.min()), float(kcas.max())
        r["kcas_dev_max"] = float(np.max(np.abs(kcas - r["trim_kcas"])))
    v = np.stack([c("vx"), c("vy"), c("vz")], 1)
    r["tas_kt_mean"] = float(np.mean(np.linalg.norm(v, axis=1)) / 0.514444)
    r["err_max_ft"] = float(np.max(np.abs(e)))
    r["err_rms_ft"] = float(np.sqrt(np.mean(e ** 2)))
    # segments between target changes
    chg = [0] + [i for i in range(1, len(ta)) if abs(ta[i] - ta[i - 1]) > 1e-6]
    seg_start = np.zeros_like(t)
    for a, b in zip(chg, chg[1:] + [len(t)]):
        seg_start[a:b] = t[a]
    hold = (t - seg_start) >= 20.0
    if ramp is not None:
        # with a ramped reference the step is only "due" once the ramp has arrived: hold = >= 10 s after arrival
        arrived = np.abs(ramp / FT - ta) < 1.0
        t_arr = np.full_like(t, np.inf)
        for a_, b_ in zip(chg, chg[1:] + [len(t)]):
            k = np.nonzero(arrived[a_:b_])[0]
            if len(k):
                t_arr[a_:b_] = t[a_ + k[0]]
        hold = (t - t_arr) >= 10.0
    r["hold_def"] = "10 s after ramp arrival" if ramp is not None else "20 s after step"
    r["hold_rms_ft"] = float(np.sqrt(np.mean(e[hold] ** 2))) if hold.any() else None
    r["hold_max_ft"] = float(np.max(np.abs(e[hold]))) if hold.any() else None
    overs, settle20, settle5 = [], [], []
    for a, b in zip(chg[1:], chg[2:] + [len(t)]):
        step = ta[a] - ta[a - 1]
        se, st = e[a:b], t[a:b]
        overs.append(float(max(0.0, np.max(np.sign(step) * se))))
        for band, out in ((20.0, settle20), (0.05 * abs(step), settle5)):
            o = np.nonzero(np.abs(se) > band)[0]
            out.append(0.0 if len(o) == 0 else (None if o[-1] == len(se) - 1 else float(st[o[-1] + 1] - st[0])))
    r["overshoot_ft"] = max(overs) if overs else None
    r["overshoot_pct"] = 100 * r["overshoot_ft"] / r["step_ft"] if r["step_ft"] else None
    r["settle20_s"] = settle20
    r["settle5pct_s"] = settle5
    r["theta_deg"] = [float(np.degrees(c("theta").min())), float(np.degrees(c("theta").max()))]
    r["phi_absmax_deg"] = float(np.degrees(np.abs(c("phi")).max()))
    psi = np.unwrap(c("psi"))
    r["heading_drift_deg"] = float(np.degrees(psi[-1] - psi[0]))
    if "nz" in ch:
        r["nz"] = [float(c("nz").min()), float(c("nz").max())]
        r["nz_dev_max"] = float(np.max(np.abs(c("nz") - 1.0)))
    vz = c("vz")
    r["climb_fpm"] = [float(vz.min() / FT * 60), float(vz.max() / FT * 60)]
    thr, el = c("throttle"), c("elevator")
    tmax = throttle_max if throttle_max is not None else 1.0
    r["throttle_max_limit"] = tmax
    r["throttle_sat_hi_pct"] = float(100 * np.mean(thr >= tmax - 1e-3))
    r["throttle_sat_lo_pct"] = float(100 * np.mean(thr <= 1e-3))
    r["thr_tv_per_s"] = float(np.sum(np.abs(np.diff(thr))) / max(t[-1], 1e-9))
    r["genome"] = doc.get("genome")
    r["throttle_range"] = [float(thr.min()), float(thr.max())]
    r["throttle_at_max_pct"] = float(100 * np.mean(thr >= thr.max() - 1e-6)) if thr.max() > 0 else 0.0
    r["throttle_at_min_pct"] = float(100 * np.mean(thr <= 1e-6))
    r["elev_range"] = [float(el.min()), float(el.max())]
    r["elev_sat_pct"] = float(100 * np.mean(np.abs(el) >= 0.999))
    r["elev_tv_per_s"] = float(np.sum(np.abs(np.diff(el))) / max(t[-1], 1e-9))
    q = np.stack([c("qw"), c("qx"), c("qy"), c("qz")], 1)
    qn = np.linalg.norm(q, axis=1)
    r["quat_norm_err"] = float(np.max(np.abs(qn - 1)))
    Ra, Rb = quat_to_matrix(q / qn[:, None]), M_EN @ euler_ned(c("phi"), c("theta"), c("psi"))
    f = np.sqrt(np.einsum("nij,nij->n", Ra - Rb, Ra - Rb))
    r["quat_euler_deg"] = float(np.degrees(np.max(2 * np.arcsin(np.clip(f / (2 * np.sqrt(2)), 0, 1)))))
    pos = np.stack([c("x"), c("y"), c("z")], 1)
    pred = pos[:-1] + 0.5 * (v[:-1] + v[1:]) * np.diff(t)[:, None]
    r["pos_vel_err_m"] = float(np.max(np.linalg.norm(pred - pos[1:], axis=1)))
    r["alt_vs_z_err_m"] = float(np.max(np.abs(c("alt_msl_m") - (c("z") + doc["frame"]["origin_alt_m"]))))
    r["track_len_km"] = float(np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)) / 1000)
    r["_hash"] = hash(d.tobytes())
    return r


# rough plausible cruise/approach KCAS ranges by type (for flagging only)
TYPE_KCAS = {"c172": (50, 165), "t38": (150, 500), "737": (130, 340), "f16": (130, 600)}


def flags_for(r):
    out = []
    if r["nan"]:
        out.append(f"{r['nan']} NaN values")
    if r["status"] != "ok" or r["t_end"] < 89.9:
        out.append(f"terminated: {r['status']} at {r['t_end']:.1f}s")
    if abs(r["alt0_err_ft"]) > 5:
        out.append(f"starts {r['alt0_err_ft']:+.1f} ft off the trim altitude")
    if r.get("kcas0") is not None and abs(r["kcas0"] - r["trim_kcas"]) > 2:
        out.append(f"t=0 KCAS {r['kcas0']:.1f} != trim {r['trim_kcas']}")
    if r.get("kcas_dev_max", 0) > 0.1 * r["trim_kcas"]:
        out.append(f"airspeed excursion up to {r['kcas_dev_max']:.0f} kt from trim "
                   f"({r['kcas_min']:.0f}-{r['kcas_max']:.0f} KCAS)")
    if r["hold_max_ft"] is not None and r["hold_max_ft"] > max(20.0, 0.05 * r["step_ft"]):
        out.append(f"poor hold: max |err| {r['hold_max_ft']:.0f} ft >= 20 s after a step")
    if any(s is None for s in r["settle5pct_s"]):
        out.append("never settles within 5% of the step in some segment")
    if r["overshoot_pct"] is not None and r["overshoot_pct"] > 10:
        out.append(f"overshoot {r['overshoot_pct']:.0f}% of step")
    if r["quat_euler_deg"] > 0.02:
        out.append(f"quaternion vs Euler disagree by {r['quat_euler_deg']:.3f} deg")
    if r["quat_norm_err"] > 1e-5:
        out.append(f"quaternion norm error {r['quat_norm_err']:.1e}")
    if r["pos_vel_err_m"] > 0.5:
        out.append(f"position inconsistent with velocity ({r['pos_vel_err_m']:.2f} m)")
    if r["elev_sat_pct"] > 1:
        out.append(f"elevator saturated {r['elev_sat_pct']:.1f}% of the time")
    if r["throttle_sat_hi_pct"] > 0 or r["throttle_sat_lo_pct"] > 0:
        out.append(f"throttle at its limits: {r['throttle_sat_hi_pct']:.1f}% at max ({r['throttle_max_limit']}), "
                   f"{r['throttle_sat_lo_pct']:.1f}% at idle")
    if r["throttle_at_max_pct"] > 20 and r["throttle_range"][1] > 0 and r["throttle_sat_hi_pct"] == 0:
        out.append(f"throttle pinned at its max ({r['throttle_range'][1]:.2f}) {r['throttle_at_max_pct']:.0f}% of the time")
    if r["throttle_at_min_pct"] > 20:
        out.append(f"throttle at idle {r['throttle_at_min_pct']:.0f}% of the time")
    if r["phi_absmax_deg"] > 10:
        out.append(f"bank up to {r['phi_absmax_deg']:.1f} deg (wing leveler not holding)")
    if abs(r["heading_drift_deg"]) > 20:
        out.append(f"heading drifts {r['heading_drift_deg']:+.0f} deg over the run")
    for k, (lo, hi) in TYPE_KCAS.items():
        if r["aircraft"].lower().startswith(k) and r.get("kcas_min") is not None and \
                (r["kcas_min"] < lo or r["kcas_max"] > hi):
            out.append(f"KCAS {r['kcas_min']:.0f}-{r['kcas_max']:.0f} outside plausible {lo}-{hi} for type")
    return out


def load_run(path):
    """-> (run_id, rows, meta) for a run dir / trajectories dir / index.json. Reads config/summary if present."""
    p = path
    if os.path.isdir(os.path.join(p, "trajectories")):
        p = os.path.join(p, "trajectories")
    if os.path.isdir(p):
        p = os.path.join(p, "index.json")
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(p)))
    idx = json.load(open(p))
    entries = idx["entries"] if isinstance(idx, dict) else idx
    base = os.path.dirname(p)
    meta = {"profiles": {}, "summary": {}}
    try:
        cfg = json.load(open(os.path.join(run_dir, "config.json")))["resolved"]
        meta["profiles"] = {a["name"]: a.get("resolved_profile", {}) for a in cfg.get("aircraft", [])}
    except (OSError, KeyError, ValueError):
        pass
    try:
        meta["summary"] = {a["aircraft"]: a for a in json.load(open(os.path.join(run_dir, "summary.json")))["aircraft"]}
    except (OSError, KeyError, ValueError):
        pass
    rows = []
    for e in entries:
        doc = json.load(open(os.path.join(base, e["file"])))
        prof = meta["profiles"].get(doc["aircraft"], {})
        r = analyse(doc, prof.get("throttle_max"))
        r["index_fitness_match"] = doc["fitness"] == e["fitness"]
        r["run"] = doc.get("run_id") or (idx.get("run_id") if isinstance(idx, dict) else None)
        r["flags"] = flags_for(r)
        rows.append(r)
    rows.sort(key=lambda r: (r["aircraft"], r["gen"]))
    return (idx.get("run_id") if isinstance(idx, dict) else None) or os.path.basename(run_dir), rows, meta


def report_run(run_id, rows):
    print(f"=== {run_id}")
    hdr = (f"{'aircraft':9s}{'gen':>4s}{'fitness':>9s}{'scen0':>8s}{'trim':>14s}{'step':>6s}{'errmax':>7s}{'holdmax':>8s}"
           f"{'ovs%':>6s}{'rampMax':>8s}{'rampRMS':>8s}{'climb fpm':>14s}{'KCAS range':>12s}{'theta':>13s}{'nz':>12s}"
           f"{'thr':>11s}{'q-eul°':>8s}")
    print(hdr)
    for r in rows:
        print(f"{r['aircraft']:9s}{r['gen']:4d}{r['fitness']:9.4f}{r['scenario_cost']:8.4f}"
              f"{r['trim_kcas']:6.0f}kt/{r['trim_alt_ft']:5.0f}{r['step_ft']:6.0f}{r['err_max_ft']:7.0f}"
              f"{r['hold_max_ft']:8.1f}{r['overshoot_pct']:6.1f}"
              f"{r.get('ramp_err_max_ft', float('nan')):8.1f}{r.get('ramp_err_rms_ft', float('nan')):8.2f}"
              f"{r['climb_fpm'][0]:7.0f}/{r['climb_fpm'][1]:<6.0f}"
              f"{r.get('kcas_min', float('nan')):6.0f}-{r.get('kcas_max', float('nan')):<5.0f}"
              f"{r['theta_deg'][0]:6.1f}..{r['theta_deg'][1]:<5.1f}{r.get('nz', [0, 0])[0]:5.2f}..{r.get('nz', [0, 0])[1]:<5.2f}"
              f"{r['throttle_range'][0]:5.2f}-{r['throttle_range'][1]:<5.2f}{r['quat_euler_deg']:8.4f}")
    print()
    by_ac = {}
    for r in rows:
        by_ac.setdefault(r["aircraft"], []).append(r)
    run_flags = []
    for ac, rs in by_ac.items():
        f = [r["fitness"] for r in rs]
        if any(b > a + 1e-12 for a, b in zip(f, f[1:])):
            run_flags.append(f"{ac}: best fitness gets WORSE over generations {f}")
        hs = [r["_hash"] for r in rs]
        for i in range(1, len(rs)):
            if hs[i] == hs[i - 1]:
                run_flags.append(f"{ac}: g{rs[i]['gen']} trajectory identical to g{rs[i - 1]['gen']}")
        g0, gl = rs[0], rs[-1]
        print(f"{ac:8s} g{g0['gen']}->g{gl['gen']}: fitness {g0['fitness']:.4f} -> {gl['fitness']:.4f} "
              f"({100 * (gl['fitness'] / g0['fitness'] - 1):+.1f}%), hold max {g0['hold_max_ft']:.1f} -> {gl['hold_max_ft']:.1f} ft, "
              f"overshoot {g0['overshoot_pct']:.1f}% -> {gl['overshoot_pct']:.1f}%, "
              f"elev TV {g0['elev_tv_per_s']:.4f} -> {gl['elev_tv_per_s']:.4f}/s, trim '{g0['start_event']}'")
    print("\nFLAGS")
    for r in rows:
        for fl in r["flags"]:
            print(f"  {r['aircraft']} g{r['gen']}: {fl}")
        if not r["index_fitness_match"]:
            print(f"  {r['aircraft']} g{r['gen']}: index fitness != file fitness")
    for fl in run_flags:
        print("  " + fl)
    print()
    return run_flags


def _last(rows, ac):
    rs = [r for r in rows if r["aircraft"] == ac]
    return max(rs, key=lambda r: r["gen"]) if rs else None


def _gen(rows, ac, g):
    return next((r for r in rows if r["aircraft"] == ac and r["gen"] == g), None)


def cross_report(runs, baseline=None):
    """runs: [(run_id, rows, meta)] (seeds of one config); baseline: (run_id, rows, meta) or None."""
    flags = []
    acs = sorted({r["aircraft"] for _, rows, _ in runs for r in rows})
    names = [rid for rid, _, _ in runs]
    print(f"=== cross-seed: {', '.join(names)}" + (f"  (baseline {baseline[0]})" if baseline else ""))
    print(f"{'aircraft':9s}{'g19 fitness per seed':>30s}{'mean':>8s}{'CV%':>6s}{'peak nz':>16s}{'|nz-1|max':>10s}"
          f"{'ramp max ft':>18s}{'ramp RMS':>17s}{'max climb fpm':>16s}{'thr sat%':>17s}{'hdg drift':>13s}")
    out = {}
    for ac in acs:
        L = [_last(rows, ac) for _, rows, _ in runs]
        L = [x for x in L if x]
        fit = np.array([x["fitness"] for x in L])
        nzs = [x.get("nz", [np.nan, np.nan]) for x in L]
        o = out[ac] = {
            "fitness": fit.tolist(), "fitness_mean": float(fit.mean()), "fitness_cv_pct": float(100 * fit.std() / fit.mean()),
            "nz_min": float(min(n[0] for n in nzs)), "nz_max": float(max(n[1] for n in nzs)),
            "nz_dev_max": float(max(x.get("nz_dev_max", np.nan) for x in L)),
            "ramp_max": [x.get("ramp_err_max_ft") for x in L], "ramp_rms": [x.get("ramp_err_rms_ft") for x in L],
            "climb_max": [max(abs(x["climb_fpm"][0]), abs(x["climb_fpm"][1])) for x in L],
            "thr_sat": [x["throttle_sat_hi_pct"] + x["throttle_sat_lo_pct"] for x in L],
            "hdg": [x["heading_drift_deg"] for x in L],
            "overshoot_pct": [x["overshoot_pct"] for x in L], "elev_tv": [x["elev_tv_per_s"] for x in L],
        }
        fmt = lambda v, f: "/".join(("—" if a is None else format(a, f)) for a in v)  # noqa: E731
        print(f"{ac:9s}{fmt(fit, '.4f'):>30s}{o['fitness_mean']:8.4f}{o['fitness_cv_pct']:6.1f}"
              f"{o['nz_min']:7.2f}..{o['nz_max']:<6.2f}{o['nz_dev_max']:10.2f}{fmt(o['ramp_max'], '.1f'):>18s}"
              f"{fmt(o['ramp_rms'], '.2f'):>17s}{fmt(o['climb_max'], '.0f'):>16s}{fmt(o['thr_sat'], '.1f'):>17s}{fmt(o['hdg'], '+.0f'):>13s}")
        if o["fitness_cv_pct"] > 10:
            flags.append(f"{ac}: gen-19 fitness varies {o['fitness_cv_pct']:.0f}% (CV) across seeds {fmt(fit, '.4f')}")
        # g9 -> g19 stall per seed
        for rid, rows, _ in runs:
            a9, a19 = _gen(rows, ac, 9), _gen(rows, ac, 19)
            if a9 and a19:
                d = 100 * (a19["fitness"] / a9["fitness"] - 1)
                o.setdefault("g9_g19_pct", []).append(d)
        if o.get("g9_g19_pct") and max(o["g9_g19_pct"]) > -1.0:
            flags.append(f"{ac}: g9->g19 improvement < 1% in some seed ({'/'.join(f'{v:+.1f}%' for v in o['g9_g19_pct'])}) - stalled")
    print("\ng9 -> g19 fitness change per seed: " + "; ".join(f"{ac} {'/'.join(f'{v:+.1f}%' for v in out[ac].get('g9_g19_pct', []))}" for ac in acs))
    # gains: per seed (best of last gen), spread as max/min ratio; bounds from summary
    print("\ngen-19 gains per seed (spread = max/min) and genes at bounds:")
    for ac in acs:
        gs = [(_last(rows, ac) or {}).get("genome") or {} for _, rows, _ in runs]
        keys = list(gs[0].keys()) if gs and gs[0] else []
        parts = []
        for k in keys:
            v = np.array([g.get(k, np.nan) for g in gs], dtype=float)
            pos = v[v > 0]
            spread = (pos.max() / pos.min()) if len(pos) == len(v) and len(pos) else float("inf")
            parts.append(f"{k} {'/'.join(f'{x:.3g}' for x in v)} (x{spread:.1f})" if np.isfinite(spread) else
                         f"{k} {'/'.join(f'{x:.3g}' for x in v)} (has 0)")
        bounds = [(rid, meta["summary"].get(ac, {}).get("genes_at_bound")) for rid, _, meta in runs]
        print(f"  {ac:7s} " + "; ".join(parts))
        print(f"  {'':7s} at bounds: " + "; ".join(f"{rid}: {b or '{}'}" for rid, b in bounds))
        out[ac]["gains"] = gs
        out[ac]["genes_at_bound"] = dict(bounds)
        for k in keys:
            v = np.array([g.get(k, np.nan) for g in gs], dtype=float)
            if np.all(v > 0) and v.max() / v.min() > 5:
                flags.append(f"{ac}: {k} differs x{v.max() / v.min():.0f} across seeds ({'/'.join(f'{x:.3g}' for x in v)}) - flat/ill-conditioned direction")
        nb = {}
        for _, b in bounds:
            for k, why in (b or {}).items():
                nb.setdefault(f"{k}={why}", 0)
                nb[f"{k}={why}"] += 1
        for k, n in nb.items():
            if n == len(runs):
                flags.append(f"{ac}: {k} at its bound in all {n} seeds")
    if baseline:
        bid, brows, bmeta = baseline
        print(f"\nvs baseline {bid} (gen 19; seeds shown as min..max):")
        print(f"{'aircraft':9s}{'step ft':>10s}{'max climb fpm':>22s}{'nz range':>26s}{'overshoot %':>20s}{'elev TV/s':>22s}"
              f"{'thr TV/s':>22s}{'thr sat%':>16s}")
        for ac in acs:
            b = _last(brows, ac)
            if not b:
                continue
            L = [_last(rows, ac) for _, rows, _ in runs]
            rng = lambda v, f: f"{min(v):{f}}..{max(v):{f}}"  # noqa: E731
            bc = max(abs(b["climb_fpm"][0]), abs(b["climb_fpm"][1]))
            cl = [max(abs(x["climb_fpm"][0]), abs(x["climb_fpm"][1])) for x in L]
            bnz = b.get("nz", [np.nan, np.nan])
            print(f"{ac:9s}{b['step_ft']:5.0f}->{L[0]['step_ft']:<4.0f}{bc:9.0f} -> {rng(cl, '.0f'):<11s}"
                  f"{bnz[0]:6.2f}..{bnz[1]:<5.2f}-> {min(x['nz'][0] for x in L):.2f}..{max(x['nz'][1] for x in L):<5.2f}"
                  f"{b['overshoot_pct']:7.1f} -> {rng([x['overshoot_pct'] for x in L], '.1f'):<10s}"
                  f"{b['elev_tv_per_s']:8.4f} -> {rng([x['elev_tv_per_s'] for x in L], '.4f'):<12s}"
                  f"{b['thr_tv_per_s']:8.4f} -> {rng([x['thr_tv_per_s'] for x in L], '.4f'):<12s}"
                  f"{b['throttle_sat_hi_pct'] + b['throttle_sat_lo_pct']:5.1f} -> {rng([x['throttle_sat_hi_pct'] + x['throttle_sat_lo_pct'] for x in L], '.1f')}")
            out[ac]["baseline"] = {"run": bid, "fitness": b["fitness"], "climb_max": bc, "nz": bnz, "overshoot_pct": b["overshoot_pct"],
                                   "elev_tv": b["elev_tv_per_s"], "thr_tv": b["thr_tv_per_s"], "step_ft": b["step_ft"],
                                   "thr_sat": b["throttle_sat_hi_pct"] + b["throttle_sat_lo_pct"]}
            if L[0]["step_ft"] != b["step_ft"]:
                flags.append(f"{ac}: step size changed {b['step_ft']:.0f} -> {L[0]['step_ft']:.0f} ft vs {bid}; "
                             f"climb/nz/overshoot comparisons are partly due to the smaller step")
        print("  note: fitness definitions differ between these runs (ramp reference, comfort term, alt_err_scale), "
              "so fitness values are not comparable across them.")
    print("\nCROSS FLAGS")
    for fl in flags:
        print("  " + fl)
    return out, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="+", help="run dir(s), trajectories dir or index.json")
    ap.add_argument("--baseline", help="older run to compare gen-19 behaviour against (cross-seed mode)")
    ap.add_argument("--json")
    a = ap.parse_args()
    runs = [load_run(p) for p in a.path]
    res = {"runs": {}}
    for rid, rows, _ in runs:
        res["runs"][rid] = {"rows": rows, "run_flags": report_run(rid, rows)}
    base = load_run(a.baseline) if a.baseline else None
    if len(runs) > 1 or base:
        res["cross"], res["cross_flags"] = cross_report(runs, base)
    if a.json:
        for v in res["runs"].values():
            for r in v["rows"]:
                r.pop("_hash", None)
        json.dump(res, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
