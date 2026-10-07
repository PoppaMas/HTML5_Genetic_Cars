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


def _fidelity_tag(doc):
    """'' for rigid; '<fidelity>:<model_version>' otherwise, so flex telemetry recorded with one FD model_version is
    never matched against telemetry of another (rigid trajectories match across model_versions: ER's rigid hash
    covers files that need not change the physics, and the channel check is what decides)."""
    rp = doc.get("replay") or {}
    fid = rp.get("fidelity") or doc.get("fidelity") or "rigid"
    if fid == "rigid":
        return ""
    return f"{fid}:{rp.get('model_version') or doc.get('model_version')}"


def doc_key(doc):
    """(aircraft, generation, scenario_id, 'best' | individual_id, fidelity tag). Ids are opaque (never parsed).
    scenario_id comes from the header (ER writes '<aircraft>:s<index>' since 2026-10-06); older files without it get
    the same string from scenario_index (= position in the aircraft's scenario_ids, ER's documented format). ER's
    own trajectory files carry no individual_id and are the best of their generation; a replay doc says is_best."""
    sid = doc.get("scenario_id")
    if sid in (None, ""):
        si = doc.get("scenario_index", 0)
        si = 0 if si in (None, "") else si
        sid = f"{doc['aircraft']}:s{si}"
    who = "best" if doc.get("is_best", True) else str(doc.get("individual_id"))
    return (str(doc["aircraft"]), int(doc["generation"]), str(sid), who, _fidelity_tag(doc))


def key_label(k):
    return f"{k[0]} g{k[1]} {k[2]}" + ("" if k[3] == "best" else f" {k[3]}") + (f" [{k[4]}]" if len(k) > 4 and k[4] else "")


def load_dir(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "traj_*.json"))):
        doc = json.load(open(p))
        key = doc_key(doc)
        if key in out:
            print(f"trajdiff: {os.path.basename(p)} duplicates {os.path.basename(out[key][0])} ({key_label(key)}); first kept",
                  file=sys.stderr)
            continue
        out[key] = (p, doc)
    return out


def _span_frac(comp):
    f = comp.get("node_span_frac")
    n = len(comp.get("axis_nodes_body_m") or [])
    if f and len(f) == n:
        return [float(x) for x in f]
    return [i / (n - 1) for i in range(n)] if n > 1 else [0.0] * n


def structure_remap(a, b):
    """Components present in both docs with a DIFFERENT node discretisation (e.g. the replay's FD nodal wings, 33 nodes,
    vs ER's modal 9-node wings): those channels share names but not span positions, so they must not be compared by
    name. Returns {component: {"a_nodes", "b_nodes", "pairs": [(i_a, i_b), ...] at coincident span fractions,
    "dofs": common dofs, "a_only_dofs", "b_only_dofs"}}."""
    sa = {c.get("name"): c for c in ((a.get("structure") or {}).get("components") or [])}
    sb = {c.get("name"): c for c in ((b.get("structure") or {}).get("components") or [])}
    out = {}
    for name in sa.keys() & sb.keys():
        fa, fb = _span_frac(sa[name]), _span_frac(sb[name])
        if len(fa) == len(fb) and all(abs(x - y) < 1e-9 for x, y in zip(fa, fb)):
            continue
        pairs = [(i, j) for j, y in enumerate(fb) for i, x in enumerate(fa) if abs(x - y) < 1e-9]
        da, db = list(sa[name].get("dof") or []), list(sb[name].get("dof") or [])
        out[name] = {"a_nodes": len(fa), "b_nodes": len(fb), "pairs": pairs, "dofs": [d for d in da if d in db],
                     "a_only_dofs": [d for d in da if d not in db], "b_only_dofs": [d for d in db if d not in da]}
    return out


# ER's fd_to_structure_channels once chose the +twist sign with `name == "wingR"`; after the FlexState /3 rename the
# modal right wing is called wingR_modal, so ER files written before the fix (`name.startswith("wingR")`) carry
# wingR_modal.twist with the WRONG (wingL) sign. Nothing in the header tells pre-fix from post-fix files (same
# traj/flex-state schema, v2_map version, git_sha and twist_doc; only phase2-pilot-s1 has `reexported`), so the
# convention is decided from the data: the modal tip twist must track the FE tip twist (same span fraction 1.0).
MODAL_SIGN_MIN_RAD = 1e-4   # need at least this much tip twist to decide
MODAL_SIGN_MIN_CORR = 0.9


def modal_twist_sign(channels, data, side="R"):
    """+1 if wing<side>_modal.twist agrees in sign with the FE wing<side>.twist at the tip (post-fix / correct),
    -1 if it is mirrored (pre-fix ER wingR_modal bug), None if undecidable (no modal or FE channels, too little
    twist, or neither correlated nor anti-correlated). `channels` = list of names, `data` = rows (or 2-D array)."""
    mod, fe = f"wing{side}_modal", f"wing{side}"
    nm = sum(1 for c in channels if c.startswith(mod + ".twist."))
    nf = sum(1 for c in channels if c.startswith(fe + ".twist."))
    if nm < 2 or nf < 2:
        return None
    try:
        ia, ib = channels.index(f"{mod}.twist.{nm - 1}"), channels.index(f"{fe}.twist.{nf - 1}")
    except ValueError:
        return None
    D = np.asarray(data, dtype=float)
    if D.ndim != 2 or not len(D):
        return None
    a, b = D[:, ia], D[:, ib]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if not len(a) or max(np.max(np.abs(a)), np.max(np.abs(b))) < MODAL_SIGN_MIN_RAD:
        return None
    r = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return 1 if r >= MODAL_SIGN_MIN_CORR else (-1 if r <= -MODAL_SIGN_MIN_CORR else None)


def declared_modal_sign(doc):
    """+1 when ER declares the post-fix sign (structure.modal_twist_sign_fixed: true, ER trajectory.py since the
    wingR_modal fix / B1 r1); None when not declared (older files: detect from the data)."""
    st = doc.get("structure") if isinstance(doc, dict) else None
    return 1 if isinstance(st, dict) and st.get("modal_twist_sign_fixed") is True else None


def modal_twist_convention(doc):
    """{'wingR_modal': +1/-1/None, 'wingL_modal': ..., 'pre_fix': bool|None, 'source': 'header'|'data'}.
    A declared `structure.modal_twist_sign_fixed: true` is trusted (no correlation detection)."""
    if declared_modal_sign(doc) == 1:
        return {"wingR_modal": 1, "wingL_modal": 1, "pre_fix": False, "source": "header"}
    r = modal_twist_sign(doc["channels"], doc["data"], "R")
    l = modal_twist_sign(doc["channels"], doc["data"], "L")
    return {"wingR_modal": r, "wingL_modal": l, "pre_fix": None if r is None else (r < 0), "source": "data"}


def diff(a, b, tol_scale=1.0):
    ca, cb = a["channels"], b["channels"]
    remap = structure_remap(a, b)
    rprefix = tuple(f"{n}." for n in remap)
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
    # pre-fix vs post-fix wingR_modal.twist (see modal_twist_sign): compare with the reference's sign undone
    sa_ = declared_modal_sign(a) or modal_twist_sign(ca, A, "R")   # header flag trusted, else detected
    sb_ = declared_modal_sign(b) or modal_twist_sign(cb, B, "R")
    mflip = sa_ is not None and sb_ is not None and sa_ != sb_
    res["wingR_modal_twist_sign"] = {"a": sa_, "b": sb_, "compared_sign_corrected": bool(mflip)}
    for c in ca:
        if c not in cb or (rprefix and c.startswith(rprefix)):
            continue
        vb = B[ib, cb.index(c)]
        if qflip is not None and c in qn:
            vb = np.where(qflip, -vb, vb)
        if mflip and c.startswith("wingR_modal.twist."):
            vb = -vb
        d = np.abs(A[ia, ca.index(c)] - vb)
        m = float(np.nanmax(d)) if len(d) else 0.0
        tol = tol_scale * (0.5 * 10 ** -DEC.get(c, 6) + 1e-9)
        res["channels"][c] = {"max_abs": m, "tol": tol, "n_exact": int(np.sum(d == 0)), "n": int(len(d))}
        if m > tol:
            res["fail"].append(c)
    # re-discretised components: compared at coincident span fractions, reported separately (not pass/fail: a
    # different discretisation of the same deflection, e.g. FD's nodal values vs ER's modal interpolation)
    res["remapped"] = {}
    for name, r in remap.items():
        per = {}
        for d in r["dofs"]:
            m = 0.0
            for i, j in r["pairs"]:
                xa, xb = f"{name}.{d}.{i}", f"{name}.{d}.{j}"
                if xa in ca and xb in cb:
                    m = max(m, float(np.nanmax(np.abs(A[ia, ca.index(xa)] - B[ib, cb.index(xb)]))) if len(ia) else 0.0)
            per[d] = m
        res["remapped"][name] = {**{k: v for k, v in r.items() if k != "pairs"}, "n_pairs": len(r["pairs"]),
                                 "max_abs_at_coincident_nodes": per}
    res["only_in_a"] = [c for c in ca if c not in cb and not (rprefix and c.startswith(rprefix))]
    res["only_in_b"] = [c for c in cb if c not in ca and not (rprefix and c.startswith(rprefix))]
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
        report[key_label(k)] = {"a": os.path.basename(pa), "b": os.path.basename(B[kb][0]), **r}
        worst = max(r["channels"].items(), key=lambda kv: kv[1]["max_abs"] / kv[1]["tol"])
        exact = sum(1 for v in r["channels"].values() if v["max_abs"] == 0)
        if verbose:
            print(f"{k[0]:6s} g{k[1]:<3d} sc{k[2]}  rows {r['rows_compared']}/{r['rows_a']}/{r['rows_b']}  channels {len(r['channels'])} "
                  f"({exact} bit-identical)  worst {worst[0]} {worst[1]['max_abs']:.3g} (tol {worst[1]['tol']:.1g})"
                  f"{'  FAIL ' + ','.join(r['fail']) if r['fail'] else ''}  meta-diff: {sorted(r['meta_diff']) or '-'}")
    return report


if __name__ == "__main__":
    sys.exit(main())
