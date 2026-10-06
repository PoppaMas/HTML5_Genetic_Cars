"""Channel-by-channel diff of trajectory files (ga-flightsim-traj/1), matched by (aircraft, generation, scenario).

    python tools/compare_traj.py <replay trajectories dir> <reference trajectories dir> [--json out.json]
    (library: sim_bridge.trajdiff.compare_dirs(a, b) -> report dict)

Rows are aligned on t (tolerance 1e-6 s). Reports max |diff| per channel, rows compared, metadata differences.
Exit 1 if any shared channel exceeds its tolerance (default: half a unit in the last written decimal + 1e-9).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# written decimals (ER trajectory.py); tolerance = 0.5 * 10**-dec (one rounding step) + tiny
DEC = {"t": 4, "x": 3, "y": 3, "z": 3, "qw": 8, "qx": 8, "qy": 8, "qz": 8, "vx": 4, "vy": 4, "vz": 4,
       "alt_msl_m": 3, "phi": 7, "theta": 7, "psi": 7, "throttle": 6, "elevator": 6, "aileron": 6,
       "rudder": 6, "target_alt_m": 3, "kcas": 3, "nz": 4, "ub": 4, "vb": 4, "wb": 4, "lat_deg": 9, "lon_deg": 9,
       "target_cmd_alt_m": 3, "target_rate_mps": 5}


def load_dir(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "traj_*.json"))):
        doc = json.load(open(p))
        key = (doc["aircraft"], int(doc["generation"]), int(doc.get("scenario_index", 0) or 0), doc.get("individual_id", "r0").split(":")[-1])
        out[key] = (p, doc)
    return out


def diff(a, b, tol_scale=1.0):
    ca, cb = a["channels"], b["channels"]
    A, B = np.array(a["data"], dtype=float), np.array(b["data"], dtype=float)
    ta, tb = A[:, ca.index("t")], B[:, cb.index("t")]
    common_t = np.intersect1d(np.round(ta, 6), np.round(tb, 6))
    ia = np.searchsorted(np.round(ta, 6), common_t)
    ib = np.searchsorted(np.round(tb, 6), common_t)
    res = {"rows_a": len(ta), "rows_b": len(tb), "rows_compared": len(common_t), "channels": {}, "fail": []}
    qn = ["qw", "qx", "qy", "qz"]
    qflip = None
    if all(q in ca and q in cb for q in qn):  # q and -q are the same attitude: compare sign-invariantly
        qa = A[ia][:, [ca.index(q) for q in qn]]
        qb = B[ib][:, [cb.index(q) for q in qn]]
        qflip = np.sum(qa * qb, axis=1) < 0
        res["quat_sign_flipped_rows"] = int(qflip.sum())
    for c in ca:
        if c not in cb:
            continue
        vb = B[ib, cb.index(c)]
        if qflip is not None and c in qn:
            vb = np.where(qflip, -vb, vb)
        d = np.abs(A[ia, ca.index(c)] - vb)
        m = float(np.nanmax(d)) if len(d) else 0.0
        tol = tol_scale * (0.5 * 10 ** -DEC.get(c, 6) + 1e-9)
        res["channels"][c] = {"max_abs": m, "tol": tol, "n_exact": int(np.sum(d == 0)), "n": int(len(d))}
        if m > tol:
            res["fail"].append(c)
    res["only_in_a"] = [c for c in ca if c not in cb]
    res["only_in_b"] = [c for c in cb if c not in ca]
    meta = {}
    for k in ("fitness", "scenario_cost", "status", "genome", "frame", "target", "events", "sample_hz", "dt_s", "sim_dt_s"):
        if a.get(k) != b.get(k):
            meta[k] = {"a": a.get(k), "b": b.get(k)}
    res["meta_diff"] = meta
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json")
    ap.add_argument("--tol-scale", type=float, default=1.0)
    x = ap.parse_args()
    report = compare_dirs(x.a, x.b, x.tol_scale, verbose=True)
    bad = any(v["fail"] for v in report.values())
    if x.json:
        json.dump(report, open(x.json, "w"), indent=1)
    print("RESULT:", "FAIL" if bad else "PASS", f"({len(report)} files compared)")
    return 1 if bad else 0


def compare_dirs(a_dir, b_dir, tol_scale=1.0, verbose=False):
    A, B = load_dir(a_dir), load_dir(b_dir)
    report = {}
    for k, (pa, da) in A.items():
        kb = k if k in B else None
        if kb is None:
            continue
        r = diff(da, B[kb][1], tol_scale)
        report[f"{k[0]} g{k[1]} sc{k[2]}"] = {"a": os.path.basename(pa), "b": os.path.basename(B[kb][0]), **r}
        worst = max(r["channels"].items(), key=lambda kv: kv[1]["max_abs"] / kv[1]["tol"])
        exact = sum(1 for v in r["channels"].values() if v["max_abs"] == 0)
        if verbose:
            print(f"{k[0]:6s} g{k[1]:<3d} sc{k[2]}  rows {r['rows_compared']}/{r['rows_a']}/{r['rows_b']}  channels {len(r['channels'])} "
                  f"({exact} bit-identical)  worst {worst[0]} {worst[1]['max_abs']:.3g} (tol {worst[1]['tol']:.1g})"
                  f"{'  FAIL ' + ','.join(r['fail']) if r['fail'] else ''}  meta-diff: {sorted(r['meta_diff']) or '-'}")
    return report


if __name__ == "__main__":
    sys.exit(main())
