import hashlib
import math
import numpy as np
"""Phase 4 ring course: sim_bridge.ring_course (stdlib, ER API) + sim_bridge.rings + viewer/js/rings.js."""
import ast
import json
import math
import os
import shutil
import subprocess

import pytest

from sim_bridge import ring_course as C
from sim_bridge import rings as R

SB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACS = ["c172x", "T38", "737"]


def test_ring_course_is_stdlib_only():
    tree = ast.parse(open(os.path.join(SB, "sim_bridge", "ring_course.py")).read())
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
           {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= {"__future__", "hashlib", "math", "struct"}


def test_er_api_and_determinism():
    for ac in ACS:
        for st in C.STAGES:
            a, b = C.make_course(ac, st, 77), C.make_course(ac, st, 77)
            assert json.dumps(C.public(a), sort_keys=True) == json.dumps(C.public(b), sort_keys=True)
            assert len(a["rings"]) == 5 and a["M"] == 15 and a["window"] == 5
            assert set(a["start"]) == {"pos", "heading_deg", "alt_ft", "kcas"}
            r = a["rings"][0]
            assert {"centre_m", "normal", "radius_m", "id"} <= set(r) and abs(math.hypot(*r["normal"]) - 1) < 1e-8
            assert C.public(C.make_course(ac, st, 78))["rings"] != C.public(a)["rings"]
    # ring_at(k) is the same whether reached incrementally or directly (fresh course object), any k
    c1, c2 = C.make_course("T38", "hard", 5), C.make_course("T38", "hard", 5)
    assert C.ring_at(c1, 22) == [C.ring_at(c2, k) for k in range(23)][22]
    assert C.ring_at(c1, 14)["scored"] and not C.ring_at(c1, 15)["scored"]


@pytest.mark.parametrize("ac", ACS)
def test_stage_geometry_matches_er(ac):
    v = C.scales(ac)["v_tas_ms"]          # spec v0.5: TAS at h0
    rt = v * v / (C.G * math.tan(math.radians(30)))
    for st, p in C.STAGES.items():
        c = C.make_course(ac, st, 3)
        assert abs(c["params"]["spacing_m"] - p["spacing_s"] * v) < 1e-9
        assert abs(c["params"]["radius_m"] - p["radius_s"] * v) < 1e-3
        prev, head = c["start"]["pos"], [1.0, 0.0]
        for k in range(c["M"]):
            r = C.ring_at(c, k)
            d = [r["centre_m"][i] - prev[i] for i in range(3)]
            along = d[0] * head[0] + d[1] * head[1]
            lat = -d[0] * head[1] + d[1] * head[0]
            assert abs(along - p["spacing_s"] * v) < 1e-3                    # spacing along the current track
            assert abs(lat) <= p["lat_rt"] * rt + 1e-3                       # lateral offset x R_turn(30 deg)
            assert abs(d[2]) <= p["vert_sp"] * p["spacing_s"] * v + 1e-3     # vertical offset x spacing
            assert r["centre_m"][2] < -2 * C.AIRCRAFT[ac]["min_agl_ft"] * C.FT + 1e-6 or k == 0
            n = r["normal"]
            head = [n[0] / math.hypot(n[0], n[1]), n[1] / math.hypot(n[0], n[1])]
            prev = r["centre_m"]
    e = C.make_course(ac, "easy", 1)
    assert abs(e["time_limit_s"] - 1.5 * e["nominal_time_s"]) < 1e-9 and e["nominal_time_s"] == C.nominal_time(e)
    assert e["start"]["pos"] == [0.0, 0.0, -C.AIRCRAFT[ac]["h0_ft"] * C.FT]     # NED, ground origin below the start


def test_seed_policy():
    tr = {s for g in range(100) for s in C.train_seeds(7, g, K=4)}
    ho = set(C.holdout_seeds(7, 8))
    assert len(tr) == 400 and len(ho) == 8 and not (tr & ho)
    assert C.train_seeds(7, 3) == C.train_seeds(7, 3) and C.holdout_seeds(7) == C.holdout_seeds(7)
    # same formula as ER's evolution/rings.course_seed
    import hashlib
    tag = b"p4ring|T|7|3|2"
    assert C.legacy_course_seed_v11(7, 3, 2) == int.from_bytes(hashlib.sha256(tag).digest()[:4], "little")


def _ring(r=10.0, tube=0.0):
    return {"centre_m": [100.0, 0, -500], "normal": [1.0, 0, 0], "radius_m": r, "tube_m": tube}


def test_gate_check():
    assert C.gate_check([90, 0, -500], [110, 0, -500], _ring())["result"] == "pass"
    g = C.gate_check([99, 3, -496], [101, 3, -496], _ring())
    assert g["result"] == "pass" and abs(g["rho_m"] - 5) < 1e-9 and abs(g["frac"] - 0.5) < 1e-12
    m = C.gate_check([99, 12, -500], [101, 12, -500], _ring())
    assert m["result"] == "miss" and abs(m["miss_m"] - 2) < 1e-9
    assert C.gate_check([99, 10.2, -500], [101, 10.2, -500], _ring(tube=0.5))["result"] == "rim"
    assert C.gate_check([99, 10.2, -500], [101, 10.2, -500], _ring())["result"] == "miss"   # no rim by default
    assert C.gate_check([110, 0, -500], [90, 0, -500], _ring()) is None                     # backwards


def _fly(course, waypoints, v, hz=20.0):
    """Straight segments through waypoints at speed v -> (t, pos)."""
    t, P, tt = [0.0], [list(waypoints[0])], 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        L = math.dist(a, b)
        n = max(1, int(L / v * hz))
        for i in range(1, n + 1):
            P.append([a[j] + (b[j] - a[j]) * i / n for j in range(3)])
            t.append(tt + L / v * i / n)
        tt += L / v
    return t, P


def _centres(c, ks):
    return [C.ring_at(c, k)["centre_m"] for k in ks]


def test_course_finishes_after_ring_M():
    c = C.make_course("c172x", "medium", 11, M=6)
    wp = [c["start"]["pos"]] + _centres(c, range(8))           # fly on through previews 6, 7
    t, P = _fly(c, wp, c["params"]["v_ref_mps"])
    r = C.score_course(t, P, c)
    assert r["finished"] and r["passes"] == 6 and r["misses"] == 0 and r["J_ring_miss"] == 0
    assert [g["k"] for g in r["gates"]] == list(range(6))     # preview rings never scored
    assert not r["timed_out"] and r["t_last"] < c["time_limit_s"]


def test_time_limit_miss():
    c = C.make_course("737", "hard", 12, M=8)
    wp = [c["start"]["pos"]] + _centres(c, range(8))
    t, P = _fly(c, wp, 0.55 * c["params"]["v_ref_mps"])        # too slow
    r = C.score_course(t, P, c)
    res = [g["result"] for g in r["gates"]]
    assert r["timed_out"] and not r["finished"] and "missed_time" in res and len(res) == 8
    assert all(g["t"] <= c["time_limit_s"] + 1e-9 for g in r["gates"])
    assert r["J_ring_miss"] == res.count("missed_time") / 8 == r["misses"] / 8
    # never reaching anything: every ring missed_time
    r0 = C.score_course([0.0, c["time_limit_s"] + 1], [c["start"]["pos"], c["start"]["pos"]], c)
    assert [g["result"] for g in r0["gates"]] == ["missed_time"] * 8 and r0["J_ring_miss"] == 1.0


def test_out_of_order_miss():
    c = C.make_course("T38", "medium", 13, M=8)
    r3 = C.ring_at(c, 3)
    n = r3["normal"]
    side = [-n[1] / math.hypot(n[0], n[1]), n[0] / math.hypot(n[0], n[1]), 0.0]
    detour = [r3["centre_m"][i] + 6 * r3["radius_m"] * side[i] for i in range(3)]   # outside capture (4 r)
    wp = [c["start"]["pos"]] + _centres(c, range(3)) + [detour] + _centres(c, range(4, 10))
    t, P = _fly(c, wp, c["params"]["v_ref_mps"])
    r = C.score_course(t, P, c)
    res = [(g["k"], g["result"]) for g in r["gates"]]
    assert res == [(0, "pass"), (1, "pass"), (2, "pass"), (3, "missed_order")] + [(k, "pass") for k in range(4, 8)]
    assert r["finished"] and r["misses"] == 1 and r["J_ring_miss"] == 1 / 8
    g3 = r["gates"][3]
    assert g3["t"] == r["gates"][4]["t"]                         # resolved at the ring-4 crossing


def test_window_slides_by_one_up_to_M():
    c = C.make_course("c172x", "easy", 1, M=15)
    assert C.window(c, 0) == [0, 1, 2, 3, 4] and C.window(c, 1) == [1, 2, 3, 4, 5]
    assert C.window(c, 12) == [12, 13, 14] and C.window(c, 15) == []
    gates = [{"k": k, "t": float(k + 1), "result": "pass" if k != 2 else "miss"} for k in range(15)]
    for t in [0.0, 1.5, 2.5, 3.5, 10.5]:
        st = R.ring_state_at(gates, 15, t)
        act = sorted(k for k, s in st.items() if s in ("active", "target"))
        n = int(t)
        assert act == C.window(c, n) and st[n] == "target"
    # guidance always sees n and n+1 (ring n+1 >= M is an unscored preview)
    v = C.guidance_view(c["start"]["pos"], [1, 0, 0, 0], c, 14)
    assert [x["k"] for x in v] == [14, 15] and v[0]["scored"] and not v[1]["scored"]


def test_guidance_view_frames():
    c = C.make_course("c172x", "easy", 8)
    psi = math.radians(30.0)                                       # level, heading 30 deg, NED world
    f = [math.cos(psi), math.sin(psi), 0.0]
    r = [-math.sin(psi), math.cos(psi), 0.0]
    d = [0.0, 0.0, 1.0]
    q = C.quat_from_body_axes(f, r, d)
    assert all(abs(a - b) < 1e-9 for a, b in zip(C.quat_rotate(q, [1, 0, 0]), f))
    c0 = C.ring_at(c, 0)["centre_m"]
    pos = [c0[i] - 300 * f[i] + 40 * r[i] - 20 * d[i] for i in range(3)]   # 300 behind, 40 right, 20 above
    v = C.guidance_view(pos, q, c, 0)[0]
    b = [x * v["range_m"] for x in v["los_body_frd"]]
    assert all(abs(a - e) < 1e-6 for a, e in zip(b, [300, -40, 20]))       # ring ahead, left, below
    assert v["bearing_deg"] < 0 and v["elevation_deg"] < 0
    ln, le = v["los_ned"], v["los_enu"]
    assert all(abs(a - e) < 1e-12 for a, e in zip(le, [ln[1], ln[0], -ln[2]]))


def test_frame_conversions_and_course_block():
    p = [100.0, 20.0, -1219.2]
    assert C.ned_to_enu(p) == [20.0, 100.0, 1219.2] and C.enu_to_ned(C.ned_to_enu(p)) == p
    assert C.ned_to_traj_enu(p, 1219.2) == [20.0, 100.0, 0.0]                # start altitude = traj origin
    assert C.ned_to_three(p) == [20.0, 1219.2, -100.0]
    c = C.make_course("c172x", "easy", 2, M=7)
    blk = R.course_block(c, 1219.2)
    assert len(blk["rings"]) == 7 and "_centres" not in blk
    r = blk["rings"][3]
    assert r["centre_enu_m"] == C.ned_to_traj_enu(r["centre_m"], 1219.2)
    json.dumps(blk)


@pytest.mark.skipif(not shutil.which("node") or not os.path.isdir(os.path.join(SB, "tools", "build", "node_modules", "esbuild")),
                    reason="no node / esbuild")
def test_viewer_ring_state_matches_python(tmp_path):
    c = R.course_block(C.make_course("c172x", "easy", 4, M=8), 1219.2)
    gates = [{"k": 0, "t": 1.0, "result": "pass", "rho_norm": 0.2, "miss_m": 0.0},
             {"k": 1, "t": 2.0, "result": "miss", "miss_m": 7.5},
             {"k": 2, "t": 3.0, "result": "missed_order", "miss_m": None},
             {"k": 3, "t": 3.0, "result": "pass", "miss_m": 0.0}]
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"course": c, "gates": gates}))
    ts = [0.5, 1.5, 2.5, 3.5]
    out = subprocess.run(["node", os.path.join(SB, "tools", "build", "ringcheck.mjs"), str(f), ",".join(map(str, ts))],
                         capture_output=True, text=True, cwd=os.path.join(SB, "tools", "build"), check=True)
    js = json.loads(out.stdout)
    for t in ts:
        py = R.ring_state_at(gates, 8, t)
        assert js[str(t)]["st"] == [py[k] for k in range(8)]
    assert "pass 2 miss 2 rim 0" in js["3.5"]["hud"] and "M 8" in js["3.5"]["hud"]


def test_ned_vs_fd_neu_mapping():
    c = C.make_course("737", "medium", 9)
    r = C.ring_at(c, 2)
    n, e, d = r["centre_m"]
    assert r["centre_neu_m"] == [n, e, -d] and r["centre_neu_m"][2] > 0           # Up = -D, above ground
    assert r["normal_neu"] == [r["normal"][0], r["normal"][1], -r["normal"][2]]
    assert C.neu_to_ned(C.ned_to_neu(r["centre_m"])) == r["centre_m"]
    assert C.fd_pos_to_ned([r["centre_neu_m"]]) == [r["centre_m"]]
    # scoring an FD-style (N, E, Up) track after conversion equals scoring the NED track
    wp = [c["start"]["pos"]] + [C.ring_at(c, k)["centre_m"] for k in range(c["M"] + 1)]
    t, P = _fly(c, wp, c["params"]["v_ref_mps"])
    neu = [C.ned_to_neu(p) for p in P]
    assert C.score_course(t, C.fd_pos_to_ned(neu), c)["gates"] == C.score_course(t, P, c)["gates"]
    # feeding N/E/U unconverted is detected as nonsense (every ring missed)
    assert C.score_course(t, neu, c)["passes"] == 0


def test_ctrl_surface_channels_decimate_120_to_30hz():
    n = 120
    cs = {"t_s": [(i + 1) / 120 for i in range(n)], "elev_deg": [float(i) for i in range(n)],
          "ail_deg": [0.5] * n, "rud_deg": [0.0] * n}
    ch = R.ctrl_surface_channels(cs)
    assert len(ch["t"]) == 30 and abs(ch["t"][0] - 4 / 120) < 1e-12 and ch["elev_deg"][:2] == [3.0, 7.0]
    assert set(ch) == {"t", "elev_deg", "ail_deg", "rud_deg"}


def _st(**kw):
    s = {"t": 1.0, "phi": 0.0, "theta": 0.0, "psi": 0.0, "p": 0.0, "q": 0.0, "r": 0.0, "vc_kts": 100.0,
         "vt_fps": 180.0, "nz": 1.0, "alpha": 0.05, "beta": 0.0, "agl_m": 1219.2, "h_dot_fps": 0.0}
    s.update(kw)
    return s


def test_guidance_inputs_known_geometry():
    c = C.make_course("c172x", "easy", 3)
    r0 = C.ring_at(c, 0)
    N, E, D = r0["centre_m"]
    # aircraft 300 m south, 40 m west, 20 m above ring 0, level, heading north
    g = C.guidance_inputs(_st(pos_m=[N - 300, E - 40, -D + 20]), c, 0)
    a = g["rings"][0]
    assert abs(a["range_m"] - math.sqrt(300**2 + 40**2 + 20**2)) < 1e-9
    assert np.allclose(a["los_body_frd"], np.array([300, 40, 20]) / a["range_m"])
    assert a["bearing_deg"] > 0 and a["elevation_deg"] < 0          # right of nose, below
    assert a["normal_ned"] == r0["normal"] and [x["k"] for x in g["rings"]] == [0, 1]
    # yawed 90 deg right (east): the ring (north) is now on the left
    g2 = C.guidance_inputs(_st(pos_m=[N - 300, E, -D], psi=math.pi / 2), c, 0)
    assert np.allclose(g2["rings"][0]["los_body_frd"], [0, -1, 0], atol=1e-9)


def test_guidance_inputs_alt_gamma_signs():
    c = C.make_course("c172x", "easy", 3)
    g = C.guidance_inputs(_st(pos_m=[0, 0, 1219.2], h_dot_fps=18.0), c, 0)
    assert g["pos_ned"][2] == -1219.2 and g["alt_m"] == 1219.2          # Up = -D
    assert g["gamma_deg"] > 0 and abs(g["gamma_rad"] - math.asin(0.1)) < 1e-12
    assert C.guidance_inputs(_st(pos_m=[0, 0, 1219.2], h_dot_fps=-18.0), c, 0)["gamma_deg"] < 0
    with pytest.raises(ValueError):
        C.guidance_inputs(_st(pos_m=[0, 0, 1], pos_ned=[0, 0, -1]), c, 0)


def test_guidance_inputs_ned_equals_fd_neu():
    c = C.make_course("737", "hard", 11)
    for k, (n, e, d) in enumerate([(100.0, -50.0, -3000.0), (2500.0, 300.0, -3100.0), (-10.0, 5.0, -2900.0)]):
        att = dict(phi=0.3, theta=0.05, psi=0.4 * k, h_dot_fps=10.0 * k)
        a = C.guidance_inputs(_st(pos_m=[n, e, -d], **att), c, k)
        b = C.guidance_inputs(_st(pos_ned=[n, e, d], **att), c, k)
        assert a == b


def test_f16_course_deterministic_and_scaled():
    a, b = C.make_course("f16", "hard", 77), C.make_course("f16", "hard", 77)
    assert C.public(a) == C.public(b) and C.ring_at(a, 14) == C.ring_at(b, 14)
    assert C.public(C.make_course("f16", "hard", 78))["rings"] != a["rings"]
    sc = {m: C.scales(m) for m in ("c172x", "737", "T38", "f16")}
    v = sc["f16"]["v_tas_ms"]
    assert abs(sc["f16"]["r_turn_m"] - v * v / (C.G * math.tan(math.radians(30)))) < 1e-6
    for st in ("easy", "medium", "hard"):
        sp = {m: C.make_course(m, st, 1)["params"] for m in sc}
        assert sp["c172x"]["spacing_m"] < sp["737"]["spacing_m"] < sp["T38"]["spacing_m"] < sp["f16"]["spacing_m"]
        assert sp["T38"]["radius_m"] < sp["f16"]["radius_m"]
        r = sc["f16"]["v_tas_ms"] / sc["T38"]["v_tas_ms"]
        assert abs(sp["f16"]["spacing_m"] / sp["T38"]["spacing_m"] - r) < 1e-9
        assert abs(sp["f16"]["r_turn_m"] / sp["T38"]["r_turn_m"] - r * r) < 1e-9
    assert all(-r["centre_m"][2] >= 2 * a["params"]["min_agl_m"] - 1e-6 for r in (C.ring_at(a, k) for k in range(15)))


# hand calculations (ISA, compressible CAS->TAS; cross-checked with an E6B-style table) -> KTAS
TAS_HAND = {"c172x": (100, 4000, 106.1), "T38": (300, 10000, 345.4), "737": (250, 10000, 288.7), "f16": (350, 10000, 401.5)}


def test_isa_cas_to_tas():
    T, p, a = C.isa(0.0)
    assert abs(T - 288.15) < 1e-9 and abs(p - 101325) < 1e-6 and abs(a - 340.294) < 1e-3
    T, p, a = C.isa(3048.0)
    assert abs(T - 268.338) < 1e-3 and abs(p - 69681.7) < 2.0                 # ISA table, 10 000 ft
    assert abs(C.cas_to_tas(100 * C.KT, 0.0) - 100 * C.KT) < 1e-9             # sea level: TAS = CAS
    for m, (kcas, h, ktas) in TAS_HAND.items():
        assert C.AIRCRAFT[m]["v_ref_kts"] == kcas and C.AIRCRAFT[m]["h0_ft"] == h
        assert abs(C.scales(m)["v_tas_kts"] - ktas) < 0.3, m


def test_f16_matches_fd():
    sc = C.scales("f16")
    assert abs(sc["v_tas_kts"] - 401.5) < 0.5 and abs(sc["v_tas_ms"] - 206.5) < 0.3           # FD trim 401.5 KTAS
    assert abs(sc["v_tas_ms"] / C.FT - 677.7) < 1.0                                            # FD vt 677.7 ft/s
    assert abs(sc["r_turn_m"] - 7540) < 25                                                     # FD ~24 700 ft
    c = C.make_course("f16", "easy", 2)
    assert c["v_tas_ms"] == sc["v_tas_ms"] == c["params"]["v_tas_ms"] and C.aircraft_info("f16")["v_tas_ms"] == sc["v_tas_ms"]


def _circumradius(a, b, c):
    ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
    cr = [(b[i] - a[i]) for i in range(3)], [(c[i] - a[i]) for i in range(3)]
    x = [cr[0][1] * cr[1][2] - cr[0][2] * cr[1][1], cr[0][2] * cr[1][0] - cr[0][0] * cr[1][2],
         cr[0][0] * cr[1][1] - cr[0][1] * cr[1][0]]
    area2 = math.sqrt(sum(v * v for v in x))
    return math.inf if area2 == 0 else ab * bc * ca / (2 * area2)


@pytest.mark.parametrize("ac", ["c172x", "T38", "737", "f16"])
def test_fd_hints_and_min_agl_all_stages(ac):
    import json, os
    lim = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "flight-dynamics", "v2_results",
                                      "p4_aircraft_limits.json")))["aircraft"][ac]["ring_hint"]
    min_sp, min_r = lim["min_spacing_ft"] * C.FT, lim["min_path_radius_ft"] * C.FT
    for st in C.STAGES:
        for seed in range(6):
            c = C.make_course(ac, st, seed)
            pts = [c["start"]["pos"]] + [C.ring_at(c, k)["centre_m"] for k in range(c["M"] + 2)]
            assert min(math.dist(a, b) for a, b in zip(pts, pts[1:])) >= min_sp, (st, seed)
            assert min(_circumradius(*pts[i:i + 3]) for i in range(len(pts) - 2)) >= min_r, (st, seed)
            assert all(-p[2] >= 2 * c["params"]["min_agl_m"] - 1e-6 for p in pts[1:])


# ---- ring_course 1.2: seeds (spec v0.6 section 4) ----
def test_version_string():
    assert C.VERSION == "ring_course/1.2" and C.make_course("T38", "easy", 1)["version"] == "ring_course/1.2"


def test_course_seed_golden_and_encoding():
    assert C.course_seed_string(1, 5, 2, "T38") == "p4course-seed/1|run=1|gen=5|k=2|ac=T38"
    assert C.course_seed(1, 5, 2, "T38") == 2920704114203819991          # golden (also: printf | sha256sum)
    assert C.course_seed_string(1, 99, 3, "f16", holdout=True) == "p4course-seed/1|run=1|gen=holdout|k=3|ac=f16"
    assert C.course_seed(1, 99, 3, "f16", holdout=True) == 6831254776690595837
    s = "p4course-seed/1|run=1|gen=5|k=2|ac=T38"
    assert C.course_seed(1, 5, 2, "T38") == int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") < 2 ** 64
    with pytest.raises(KeyError):
        C.course_seed(1, 5, 2, "t38")


def _rings(seed, ac="T38", st="medium"):
    c = C.make_course(ac, st, seed)
    return [C.ring_at(c, k)["centre_m"] for k in range(c["M"])]


def test_course_seed_varies_by_run_gen_k_aircraft_not_stage():
    base = C.course_seed(1, 5, 2, "T38")
    others = [C.course_seed(2, 5, 2, "T38"), C.course_seed(1, 6, 2, "T38"), C.course_seed(1, 5, 3, "T38"),
              C.course_seed(1, 5, 2, "737")]
    assert len({base, *others}) == 5
    for o in others[:3]:
        assert _rings(o) != _rings(base)                               # run / gen / k -> different rings
    assert _rings(C.course_seed(1, 5, 2, "T38")) == _rings(base)      # same inputs -> identical rings
    tr = C.train_courses(1, 5, "T38", "hard", K=4)
    assert [c["seed"] for c in tr] == [C.course_seed(1, 5, k, "T38") for k in range(4)]
    assert tr[2]["provenance"] == {"course_seed": base, "run_seed": 1, "gen": 5, "k": 2, "holdout": False,
                                   "aircraft": "T38", "stage": "hard", "version": C.VERSION, "seed_scheme": C.SEED_SCHEME}


def test_holdout_fixed_within_run_changes_between_runs():
    a = [C.course_seed(1, g, 3, "f16", holdout=True) for g in (0, 5, 99, None)]
    assert len(set(a)) == 1
    assert C.course_seed(2, 0, 3, "f16", holdout=True) != a[0]
    assert a[0] not in {C.course_seed(1, g, k, "f16") for g in range(50) for k in range(8)}
    h = C.holdout_courses(1, "f16", "hard", n=8)
    assert [c["seed"] for c in h] == [C.course_seed(1, None, j, "f16", holdout=True) for j in range(8)]


def test_replay_regenerates_course_from_trajectory_fields():
    c = C.train_courses(7, 12, "c172x", "medium", K=4)[3]
    blk = R.course_block(c, traj_origin_alt_m=1219.2)
    blk = json.loads(json.dumps(blk))                                   # as stored in a trajectory file
    for key in ("course_seed", "run_seed", "gen", "k", "stage", "version"):
        assert key in blk
    assert (blk["run_seed"], blk["gen"], blk["k"]) == (7, 12, 3)
    c2 = C.regenerate(blk)
    assert C.check_rings(c2, blk["rings"]) < 1e-6
    assert C.public(c2)["rings"] == C.public(c)["rings"] and c2["time_limit_s"] == c["time_limit_s"]
    bad = json.loads(json.dumps(blk)); bad["rings"][4]["centre_m"][0] += 1.0
    with pytest.raises(ValueError):
        C.check_rings(c2, bad["rings"])
    old = dict(blk, version="ring_course/1.0", provenance=dict(blk["provenance"], version="ring_course/1.0"))
    with pytest.raises(ValueError):
        C.regenerate(old)


# ---- per-ring timeout (spec v0.6 section 3) ----
def test_leg_time_definition():
    c = C.make_course("737", "hard", 4)
    v = c["params"]["v_tas_ms"]
    assert abs(C.leg_time(c, 0) - math.dist(c["start"]["pos"], C.ring_at(c, 0)["centre_m"]) / v) < 1e-12
    assert abs(C.leg_time(c, 3) - math.dist(C.ring_at(c, 2)["centre_m"], C.ring_at(c, 3)["centre_m"]) / v) < 1e-12
    assert c["params"]["ring_timeout_factor"] == 3.0
    assert C.make_course("737", "hard", 4, ring_timeout_factor=2.0)["params"]["ring_timeout_factor"] == 2.0


def test_ring_timeout_fires_and_window_slides():
    c = C.make_course("c172x", "easy", 21)
    s0, r0, r1 = c["start"]["pos"], C.ring_at(c, 0)["centre_m"], C.ring_at(c, 1)["centre_m"]
    # loiter far off to the side (no crossings) for > 3 leg-0 times, then fly through ring 1
    v = c["params"]["v_tas_ms"]
    side = [s0[0], s0[1] + 1.6 * C.leg_time(c, 0) * v, s0[2]]        # out and back = 3.2 leg-0 times
    t, P = _fly(c, [s0, side, s0], v)
    assert t[-1] > 3 * C.leg_time(c, 0)
    t2, P2 = _fly(c, [P[-1], r0[:2] + [r0[2] - 500], r1], v)            # passes above ring 0, then through ring 1
    T, PP = t + [t[-1] + x for x in t2[1:]], P + P2[1:]
    sc = C.score_course(T, PP, c)
    g0 = sc["gates"][0]
    assert g0["k"] == 0 and g0["result"] == "timeout"
    assert abs(g0["deadline_t"] - 3 * C.leg_time(c, 0)) < 1e-9 and 0 <= g0["t"] - g0["deadline_t"] <= 1 / 20 + 1e-9
    assert sc["timeouts"] >= 1 and sc["misses"] >= 1
    assert any(g["k"] == 1 and g["result"] == "pass" for g in sc["gates"])
    # 1.1 scoring (no per-ring timeout) never emits 'timeout'
    assert all(g["result"] != "timeout" for g in C.score_course(T, PP, c, ring_timeout=False)["gates"])


def test_timeout_clock_restarts_at_previous_resolution():
    c = C.make_course("T38", "easy", 5, ring_timeout_factor=3.0)
    v = c["params"]["v_tas_ms"]
    wp = [c["start"]["pos"]] + [C.ring_at(c, k)["centre_m"] for k in range(c["M"] + 1)]
    t, P = _fly(c, wp, 0.40 * v)                     # 2.5 leg times per leg: never times out (clock restarts each ring)
    sc = C.score_course(t, P, c)
    assert sc["timeouts"] == 0
    assert sc["passes"] == sum(g["t"] <= c["time_limit_s"] for g in sc["gates"] if g["result"] == "pass")
    t, P = _fly(c, wp, 0.30 * v)                     # 3.33 leg times per leg -> every reached ring times out first
    sc = C.score_course(t, P, c)
    assert sc["timeouts"] >= 1 and sc["gates"][0]["result"] == "timeout"
