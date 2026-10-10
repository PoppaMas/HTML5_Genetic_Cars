#!/usr/bin/env python3
"""SYNTHETIC Phase 4 ring-course demo (no JSBSim, no ER, no FD): kinematic paths through sim_bridge.rings courses.
    $PY tools/make_phase4_demo.py   -> data/phase4_demo/{index.json, traj_*.json}, data/phase4_demo_standalone.html,
                                       screenshots/phase4_*.png
c172x easy: one miss (ring 4); T38 medium: detour -> ring 6 missed_order; 737 hard at 0.55 v_ref: time-limit misses.
Gate results come from sim_bridge.ring_course.score_course on the synthetic CG path (not hand-written)."""
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sim_bridge import rings as R  # noqa: E402

OUT = os.path.join(HERE, "data", "phase4_demo")
PAGE = os.path.join(HERE, "data", "phase4_demo_standalone.html")
NOTE = "SYNTHETIC Phase 4 ring-course demo: kinematic paths, not a simulated flight. Rings from sim_bridge.rings (seeded)."
CASES = [  # aircraft, stage, seed, {ring: in-plane offset as a fraction of r}, speed factor, M
    ("c172x", "easy", 4101, {4: 2.2}, 1.0, 10),       # one plain miss (ring 4)
    ("T38", "medium", 4102, {6: 6.0}, 1.0, 10),       # detour around ring 6 (outside capture) -> ring 6 missed_order
    ("737", "hard", 4103, {}, 0.55, 10),              # too slow -> rings after the time limit are missed_time
    ("f16", "hard", 4104, {}, 1.0, 10),               # Phase 4 amendment: f16 clean run (all pass)
]
HZ = 20.0
ORIGIN_ALT = 0.0


def hermite(p0, p1, m0, m1, s):
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1)[:, None] * p0 + (s3 - 2 * s2 + s)[:, None] * m0 +
            (-2 * s3 + 3 * s2)[:, None] * p1 + (s3 - s2)[:, None] * m1)


def path(course, offsets, rng):
    """NED waypoints: start, then each ring centre (+ offset), Hermite with ring normals as tangents."""
    from sim_bridge import ring_course as C
    st = course["start"]
    h = math.radians(st["heading_deg"])
    pts = [np.array(st["pos"], float)]
    tans = [np.array([math.cos(h), math.sin(h), 0.0])]
    for k in range(course["M"] + 1):                 # +1: fly on through the first preview ring
        r = C.ring_at(course, k)
        n = np.array(r["normal"], float)
        right = np.cross([0, 0, 1.0], n)            # NED: down x fwd = right
        right /= np.linalg.norm(right)
        up = np.cross(right, n)
        off = offsets.get(k)
        if off is None:
            a = rng.uniform(0, 2 * math.pi)
            off_v = 0.3 * r["radius_m"] * (math.cos(a) * right + math.sin(a) * up)
        else:
            off_v = off * r["radius_m"] * right
        pts.append(np.array(r["centre_m"]) + off_v)
        tans.append(n)
    seg = []
    for i in range(len(pts) - 1):
        L = np.linalg.norm(pts[i + 1] - pts[i])
        s = np.linspace(0, 1, 400, endpoint=False)
        seg.append(hermite(pts[i], pts[i + 1], tans[i] * L, tans[i + 1] * L, s))
    return np.vstack(seg)


def quat_from_axes(f, r, d):
    m = np.column_stack([f, r, d])   # body FRD -> ENU
    tr = m.trace()
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        return [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    S = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k, j] - m[j, k]) / S
    q[1 + i] = 0.25 * S
    q[1 + j] = (m[j, i] + m[i, j]) / S
    q[1 + k] = (m[k, i] + m[i, k]) / S
    return q


def fly(course, offsets, seed, vfac):
    from sim_bridge import ring_course as C
    rng = np.random.default_rng(seed)
    Pn = path(course, offsets, rng)
    v = course["params"]["v_ref_mps"] * vfac
    ds = np.linalg.norm(np.diff(Pn, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(ds)])
    t = np.arange(0, s[-1] / v, 1 / HZ)
    XN = np.column_stack([np.interp(t * v, s, Pn[:, i]) for i in range(3)])
    res = C.score_course(t.tolist(), XN.tolist(), course)
    h0 = -course["start"]["pos"][2]
    X = np.array([C.ned_to_traj_enu(p, h0) for p in XN])          # trajectory ENU (origin at the start, alt h0 MSL)
    V = np.gradient(X, 1 / HZ, axis=0)
    psi = np.unwrap(np.arctan2(V[:, 0], V[:, 1]))
    psidot = np.gradient(psi, 1 / HZ)
    phi = np.clip(np.arctan(v * psidot / 9.80665), -1.2, 1.2)
    phi = np.convolve(phi, np.ones(9) / 9, mode="same")
    rows = []
    phidot = np.gradient(phi, 1 / HZ)
    gam = np.arctan2(V[:, 2], np.hypot(V[:, 0], V[:, 1]))
    gamdot = np.gradient(gam, 1 / HZ)
    for i in range(len(t)):
        f = V[i] / np.linalg.norm(V[i])
        r0 = np.cross(f, [0, 0, 1.0])
        r0 /= np.linalg.norm(r0)
        d0 = np.cross(f, r0)
        c, sn = math.cos(phi[i]), math.sin(phi[i])
        r = c * r0 + sn * d0
        d = -sn * r0 + c * d0
        q = R.quat_from_body_axes(f, r, d)
        rows.append([float(t[i]), *map(float, X[i]), *map(float, q), *map(float, V[i]), float(X[i, 2] + h0),
                     float(phi[i]), float(gam[i]), float(psi[i] % (2 * math.pi)), 0.7,   # angles in radians, like ER files
                     float(np.clip(-3.0 * gamdot[i], -1, 1)), float(np.clip(1.5 * phidot[i], -1, 1)), 0.0,
                     float(1.0 / math.cos(phi[i])),
                     float(np.clip(-3.0 * gamdot[i], -1, 1)) * 20.0, float(np.clip(1.5 * phidot[i], -1, 1)) * 15.0, 0.0])
    return t, rows, res, h0


def main():
    os.makedirs(OUT, exist_ok=True)
    chans = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m", "phi", "theta", "psi",
             "throttle", "elevator", "aileron", "rudder", "nz", "elev_deg", "ail_deg", "rud_deg"]
    entries, report, tmiss = [], {}, {}
    for ac, stage, seed, offs, vfac, M in CASES:
        course = R.make_course(ac, stage, seed, M=M)
        t, rows, res, h0 = fly(course, offs, seed, vfac)
        gates = res["gates"]
        ev = [{"t": 0.0, "type": "start", "detail": f"SYNTHETIC kinematic path at {vfac:g} x v_ref (no flight model)"}]
        ev += [{"t": g["t"], "type": "gate_pass" if g["result"] == "pass" else "gate_miss",
                "detail": f"ring {g['k']} {g['result']}" + (f" miss {g['miss_m']:.1f} m" if g.get("miss_m") else "")}
               for g in gates]
        blk = R.course_block(course, h0)
        blk["synthetic"] = True
        fn = f"traj_{ac}_phase4-demo_g0.json"
        summ = {k: v for k, v in res.items() if k != "gates"}
        doc = {"schema": "ga-flightsim-traj/2", "run_id": "phase4-demo-SYNTHETIC", "aircraft": ac, "generation": 0,
               "fitness": res["passes"], "fitness_sense": "max", "fitness_doc": "SYNTHETIC: rings passed (not an ER fitness)",
               "status": "synthetic", "frame": {"origin_lat_deg": 0.0, "origin_lon_deg": 0.0, "origin_alt_m": h0,
               "axes": "ENU metres, x=east y=north z=up", "attitude": "quat body->ENU [w,x,y,z]"},
               "dt_s": 1 / HZ, "sample_hz": HZ, "events": ev, "course": blk, "gates": gates, "gate_summary": summ,
               "synthetic": True, "channels": chans, "data": rows}
        assert all(len(r) == len(chans) for r in rows), "row/channel length mismatch"
        json.dump(doc, open(os.path.join(OUT, fn), "w"))
        entries.append({"generation": 0, "fitness": res["passes"], "aircraft": ac, "file": fn,
                        "label": f"SYNTHETIC {ac} {stage} seed {seed}"})
        report[ac] = {**{k: summ[k] for k in ("passes", "misses", "M", "J_ring_miss", "finished", "timed_out",
                                              "nominal_time_s", "time_limit_s")},
                      "results": [g["result"] for g in gates]}
        tmiss[ac] = next((g["t"] for g in gates if g["result"] != "pass"), t[-1])
    json.dump({"schema": "ga-flightsim-traj-index/1", "traj_schema": "ga-flightsim-traj/2", "run_id": "phase4-demo-SYNTHETIC",
               "fitness_sense": "max", "entries": entries}, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
    import colab_viewer
    html = colab_viewer.build_standalone_html(os.path.join(OUT, "index.json"), hz=10, params={
        "mode": "single", "gen": "c172x:0", "cam": "orbit", "layout": "true", "note": NOTE},
        title="SYNTHETIC: Phase 4 ring-course demo (sim_bridge.rings)")
    open(PAGE, "w").write(html)
    print(f"page {PAGE}: {len(html.encode()) / 1e6:.2f} MB")
    if "--no-shots" in sys.argv:
        return
    spec = [   # times follow the gate results (first non-pass per aircraft)
        ["c172x_course_chase", "?mode=single&gen=c172x:0&cam=chase", round(tmiss["c172x"] - 10, 2)],
        ["c172x_after_miss", "?mode=single&gen=c172x:0&cam=chase", round(tmiss["c172x"] + 2, 2)],
        ["f16_hard_chase", "?mode=single&gen=f16:0&cam=chase", round(tmiss["f16"] * 0.5, 2)],
        ["T38_missed_order", "?mode=single&gen=T38:0&cam=chase", round(tmiss["T38"] + 1, 2)],
        ["737_time_limit_missed", "?mode=single&gen=737:0&cam=chase&rings=all", round(tmiss["737"] + 0.5, 2)],
        ["c172x_top_all_rings", "?mode=single&gen=c172x:0&cam=top&rings=all", round(tmiss["c172x"] + 2, 2)],
    ]
    sp = "/tmp/phase4_shots.json"
    json.dump(spec, open(sp, "w"))
    py = os.path.join(HERE, ".venv-shots", "bin", "python")
    p = subprocess.run([py, os.path.join(HERE, "tools", "screenshots.py"), "--bench", PAGE, "--shot-spec", sp,
                        "--prefix", "phase4_", "--strict"], capture_output=True, text=True)
    open("/tmp/phase4_shots.log", "w").write(p.stdout + p.stderr)
    print(p.stdout[:3000] if p.returncode == 0 else (p.stdout + p.stderr)[-3000:])
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
