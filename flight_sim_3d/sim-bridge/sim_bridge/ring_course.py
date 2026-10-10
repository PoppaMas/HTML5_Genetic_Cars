"""Phase 4 ring course (Sim Bridge, canonical): stdlib only (like v2_map.py) so ER/FD/Genome can import it directly.

Spec: flight_sim_3d/PHASE4_RINGS_SPEC.md. Frame: **NED metres, origin on the ground directly below the start
position** (x north, y east, z down; the start is at [0, 0, -h0]). Heading 0 = north, clockwise positive.

    c = make_course("c172x", "easy", seed)             # dict: rings[0:5] + start + M, nominal_time_s, time_limit_s
    r = ring_at(c, k)                                  # ring k for any k (deterministic from seed + k; k >= M = preview)
    res = score_course(t, pos_ned, c)                  # sequential pass/miss with the window + order + time-limit rules

Geometry follows ER's evolution/rings.py STAGES (spacing s, lateral x R_turn(30 deg bank), vertical x spacing, radius s)
and its centre/normal/radius ring convention; the random stream is stdlib (sha256), so courses are NOT bit-identical
to ER's numpy reference generator (ER: use this module for the smoke).
"""
from __future__ import annotations

import hashlib
import math
import struct

SCHEMA = "ga-flightsim-course/1"
VERSION = "ring_course/1.2"   # 1.2 (spec v0.6): per-ring timeout in score_course + course_seed(run, gen, k, aircraft)
# 1.1 (spec v0.5): geometry = TAS at h0. 1.0 (spec <= v0.4.1): KCAS-as-m/s geometry (not reproducible here).
# Ring GEOMETRY is identical in 1.1 and 1.2 for the same (model, stage, seed); 1.2 changes scoring + seed derivation.
GEOMETRY_COMPATIBLE = ("ring_course/1.1", "ring_course/1.2")
RING_TIMEOUT_FACTOR = 3.0
G = 9.80665
KT = 0.514444
FT = 0.3048
WINDOW = 5
M_DEFAULT = 15
TIME_LIMIT_FACTOR = 1.5
CAPTURE = 4.0          # a plane crossing of ring n only counts if rho <= CAPTURE * r (ER's capture rule)
BANK_REF_DEG = 30.0

# ER evolution/rings.py AIRCRAFT (v_ref = KCAS from the phase2 profiles; start altitude h0)
AIRCRAFT = {
    "c172x": {"v_ref_kts": 100.0, "h0_ft": 4000.0, "min_agl_ft": 500.0},
    "T38": {"v_ref_kts": 300.0, "h0_ft": 10000.0, "min_agl_ft": 500.0},
    "737": {"v_ref_kts": 250.0, "h0_ft": 10000.0, "min_agl_ft": 500.0},
    # f16 (Phase 4 amendment 02:45 PT). Source: FD flight-dynamics/v2_results/p4_aircraft_limits.json f16.trim
    # (350 KCAS at 10 000 ft = 401.5 KTAS, 677.7 ft/s); FD pins active 430793a6 / pass-through 43245063
    # (v2_results/model_versions_post_p4cs.json, FROZEN_P4cs.md5). FD's limits file has no min AGL -> 500 ft like the
    # other jets (flagged Q-FD9). fd_* = FD's reference values (info only; geometry uses ER's KCAS-as-m/s convention,
    # so SB R_turn 5726 m vs FD's TAS-based 7540 m, see Q-ER5).
    "f16": {"v_ref_kts": 350.0, "h0_ft": 10000.0, "min_agl_ft": 500.0,
            "fd_ktas": 401.5, "fd_vt_fps": 677.7, "fd_r_turn30_ft": 24700.0, "fd_min_path_radius_ft": 4690.0,
            "fd_min_spacing_ft": 2401.0, "fd_r_inst_ft": 1634.0, "fd_n_inst": 8.8, "fd_roll_rate_dps": 110.0,
            "fd_bank_course_deg": 80.0, "fd_v_max_kcas": 565.0, "fd_mach_max": 0.90,
            "fd_pins": {"active": "430793a6", "passthrough": "43245063"}},
}
# ER evolution/rings.py STAGES: spacing [s at v_ref], lateral [x R_turn], vertical [x spacing], radius [s at v_ref]
STAGES = {
    "easy": {"spacing_s": 20.0, "lat_rt": 0.05, "vert_sp": 0.02, "radius_s": 0.60, "M": M_DEFAULT},
    "medium": {"spacing_s": 14.0, "lat_rt": 0.15, "vert_sp": 0.05, "radius_s": 0.40, "M": M_DEFAULT},
    "hard": {"spacing_s": 10.0, "lat_rt": 0.30, "vert_sp": 0.08, "radius_s": 0.25, "M": M_DEFAULT},
}


# ISA (troposphere, stdlib): CAS -> TAS, compressible (subsonic, isentropic)
ISA_T0, ISA_P0, ISA_L, ISA_R, GAMMA = 288.15, 101325.0, 0.0065, 287.05287, 1.4
ISA_A0 = math.sqrt(GAMMA * ISA_R * ISA_T0)        # 340.294 m/s


def isa(h_m):
    """(T K, p Pa, a m/s) of the ISA troposphere (h <= 11 km)."""
    if not 0.0 <= h_m <= 11000.0:
        raise ValueError("isa(): troposphere only (0..11000 m)")
    T = ISA_T0 - ISA_L * h_m
    p = ISA_P0 * (T / ISA_T0) ** (G / (ISA_R * ISA_L))
    return T, p, math.sqrt(GAMMA * ISA_R * T)


def cas_to_tas(v_cas_ms, h_m):
    """Compressible CAS -> TAS (m/s) at ISA altitude h_m: qc = p0[(1 + 0.2 (Vc/a0)^2)^3.5 - 1];
    M = sqrt(5 [(qc/p + 1)^(2/7) - 1]); TAS = M a(h)."""
    _, p, a = isa(h_m)
    qc = ISA_P0 * ((1.0 + 0.2 * (v_cas_ms / ISA_A0) ** 2) ** 3.5 - 1.0)
    mach = math.sqrt(5.0 * ((qc / p + 1.0) ** (2.0 / 7.0) - 1.0))
    if mach >= 1.0:
        raise ValueError("cas_to_tas(): supersonic not supported")
    return mach * a


def scales(model):
    """V_ref is KCAS (input); geometry uses TAS at the start altitude h0 (ER, Q-ER5 decided 2026-10-07 02:53 PT)."""
    a = AIRCRAFT[model]
    vcas = a["v_ref_kts"] * KT
    v = cas_to_tas(vcas, a["h0_ft"] * FT)
    return {"v_ref_kcas": a["v_ref_kts"], "v_cas_ms": vcas, "v_tas_ms": v, "v_tas_kts": v / KT,
            "v_ref_mps": v,   # back-compat alias: = v_tas_ms (geometry speed)
            "r_turn_m": v * v / (G * math.tan(math.radians(BANK_REF_DEG)))}


def aircraft_info(model):
    """AIRCRAFT entry + derived speeds/turn radius (v_tas_ms etc.)."""
    return {**AIRCRAFT[model], **scales(model)}


SEED_SCHEME = "p4course-seed/1"


def course_seed_string(run_seed, gen, k, aircraft, holdout=False):
    """Canonical UTF-8 string hashed by course_seed (spec v0.6 section 4):
    train:   "p4course-seed/1|run=<run_seed>|gen=<gen>|k=<k>|ac=<aircraft>"
    holdout: "p4course-seed/1|run=<run_seed>|gen=holdout|k=<j>|ac=<aircraft>"   (gen ignored)
    integers in base-10 ASCII without sign padding/leading zeros (Python int()), aircraft = the AIRCRAFT id verbatim."""
    if aircraft not in AIRCRAFT:
        raise KeyError(f"unknown aircraft {aircraft!r}; known {sorted(AIRCRAFT)}")
    g = "holdout" if holdout else str(int(gen))
    return f"{SEED_SCHEME}|run={int(run_seed)}|gen={g}|k={int(k)}|ac={aircraft}"


def course_seed(run_seed, gen, k, aircraft, holdout=False):
    """THE course-seed helper (ER contract, 05:01 PT): first 8 bytes of sha256(course_seed_string(...)), big-endian,
    unsigned 64-bit. Training courses change every run AND every generation (and per k, aircraft); hold-out courses
    (holdout=True, j in the k slot, gen ignored) are fixed within a run and change between runs. Stage is NOT an input."""
    h = hashlib.sha256(course_seed_string(run_seed, gen, k, aircraft, holdout).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def train_courses(run_seed, gen, aircraft, stage, K=4, **kw):
    """Thin wrapper: the K training courses for (run_seed, gen, aircraft) at `stage`, each tagged with its provenance."""
    return [with_provenance(make_course(aircraft, stage, course_seed(run_seed, gen, k, aircraft), **kw),
                            run_seed=run_seed, gen=gen, k=k) for k in range(K)]


def holdout_courses(run_seed, aircraft, stage, n=8, **kw):
    """Thin wrapper: the n fixed hold-out courses of a run (j = 0..n-1)."""
    return [with_provenance(make_course(aircraft, stage, course_seed(run_seed, None, j, aircraft, holdout=True), **kw),
                            run_seed=run_seed, gen=None, k=j, holdout=True) for j in range(n)]


def with_provenance(course, run_seed, gen, k, holdout=False):
    course["provenance"] = {"course_seed": course["seed"], "run_seed": int(run_seed), "gen": None if holdout else int(gen),
                            "k": int(k), "holdout": bool(holdout), "aircraft": course["model"], "stage": course["stage"],
                            "version": course["version"], "seed_scheme": SEED_SCHEME}
    return course


# ---- LEGACY (ring_course 1.1, ER phase4-smoke-s1): 32-bit seeds, no aircraft, not used for new runs ----
def legacy_course_seed_v11(run_seed, gen, k, holdout=False):
    tag = f"p4ring|{'H' if holdout else 'T'}|{run_seed}|{gen if not holdout else -1}|{k}".encode()
    return int.from_bytes(hashlib.sha256(tag).digest()[:4], "little")


def train_seeds(run_seed, gen, K=4):
    """LEGACY 1.1 (kept so ER's frozen phase4-smoke code replays). New code: course_seed(..., aircraft)."""
    return [legacy_course_seed_v11(run_seed, gen, k) for k in range(K)]


def holdout_seeds(run_seed, n=8):
    """LEGACY 1.1 (see train_seeds)."""
    return [legacy_course_seed_v11(run_seed, -1, m, holdout=True) for m in range(n)]


def _u(seed, k, j):
    """Uniform [-1, 1) from (seed, k, j); stdlib, platform independent."""
    h = hashlib.sha256(f"p4course|{int(seed)}|{int(k)}|{int(j)}".encode()).digest()
    return struct.unpack("<Q", h[:8])[0] / 2.0 ** 63 - 1.0


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else list(v)


def _walk(course, k_max):
    """Centres 0..k_max (memoised on the course dict). Ring k: spacing ahead along the current track (horizontal
    heading of the previous chord) + lateral u1 * lat_rt * R_turn (right of track) + vertical u2 * vert_sp * spacing
    (NED z), altitude kept >= 2 * min_agl by reflecting the vertical step."""
    cache = course.setdefault("_centres", [])
    if len(cache) > k_max:
        return cache
    p = course["params"]
    sp, lat, vz = p["spacing_m"], p["lat_rt"] * p["r_turn_m"], p["vert_sp"] * p["spacing_m"]
    zmax = -2.0 * p["min_agl_m"]
    if cache:
        pos, head = cache[-1]["c"], cache[-1]["h"]
    else:
        s = course["start"]
        pos = list(s["pos"])
        h = math.radians(s["heading_deg"])
        head = [math.cos(h), math.sin(h)]
    for k in range(len(cache), k_max + 1):
        right = [-head[1], head[0]]
        a, b = _u(course["seed"], k, 0), _u(course["seed"], k, 1)
        dz = b * vz
        if pos[2] + dz > zmax:
            dz = -abs(dz)
        nxt = [pos[0] + sp * head[0] + a * lat * right[0], pos[1] + sp * head[1] + a * lat * right[1], pos[2] + dz]
        nrm = _norm([nxt[i] - pos[i] for i in range(3)])
        hh = math.hypot(nrm[0], nrm[1])
        head = [nrm[0] / hh, nrm[1] / hh]
        cache.append({"c": nxt, "n": nrm, "h": head})
        pos = nxt
    return cache


def ring_at(course, k):
    """Ring k (any k >= 0). k < M: scored; k >= M: unscored preview (guidance look-ahead only, never displayed/scored)."""
    w = _walk(course, k)[k]
    c = [round(x, 4) for x in w["c"]]
    nn = [round(x, 9) for x in w["n"]]
    return {"centre_neu_m": ned_to_neu(c), "normal_neu": ned_to_neu(nn), "k": k, "id": f"{course['model']}:{course['stage']}:{course['seed']}:{k}",
            "centre_m": c, "normal": nn,   # canonical NED; *_neu = FD fly_course frame (N, E, Up = -D)
            "radius_m": course["params"]["radius_m"], "tube_m": course["params"]["tube_m"],
            "scored": k < course["M"]}


def window(course, n):
    """Active ring indices when ring n is the next unresolved one: [n, min(M, n+5)). Slides by one per resolution."""
    return list(range(n, min(course["M"], n + WINDOW)))


def nominal_time(course):
    """Polyline length start -> ring 0 -> ... -> ring M-1, divided by v_ref (s). ER's T_ref."""
    pts = [course["start"]["pos"]] + [ring_at(course, k)["centre_m"] for k in range(course["M"])]
    L = sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:]))
    return L / course["params"]["v_ref_mps"]


def make_course(model, stage, seed, n_rings=5, M=None, rim_tube_m=0.0, ring_timeout_factor=RING_TIMEOUT_FACTOR):
    """ER API. Returns a dict: rings (the first n_rings, normally the initial window), start {pos, heading_deg,
    alt_ft, kcas}, M, window, nominal_time_s, time_limit_s, frame and params. Use ring_at(course, k) for later rings."""
    if model not in AIRCRAFT:
        raise KeyError(f"unknown model {model!r}; known {sorted(AIRCRAFT)}")
    st = STAGES[stage]
    a = AIRCRAFT[model]
    sc = scales(model)
    M = int(M or st["M"])
    h0 = a["h0_ft"] * FT
    c = {"schema": SCHEMA, "generator": "sim_bridge.ring_course", "version": VERSION, "model": model,
         "stage": stage, "seed": int(seed), "M": M, "window": WINDOW,
         "frame": "NED m; origin on the ground directly below the start position; x north, y east, z down",
         "start": {"pos": [0.0, 0.0, -h0], "heading_deg": 0.0, "alt_ft": a["h0_ft"], "kcas": a["v_ref_kts"]},
         "v_tas_ms": sc["v_tas_ms"], "v_ref_kcas": a["v_ref_kts"],
         "params": {"v_ref_mps": sc["v_ref_mps"], "v_tas_ms": sc["v_tas_ms"], "v_ref_kcas": a["v_ref_kts"], "r_turn_m": sc["r_turn_m"], "spacing_m": st["spacing_s"] * sc["v_ref_mps"],
                    "radius_m": round(st["radius_s"] * sc["v_ref_mps"], 4), "tube_m": float(rim_tube_m),
                    "lat_rt": st["lat_rt"], "vert_sp": st["vert_sp"], "spacing_s": st["spacing_s"], "radius_s": st["radius_s"],
                    "min_agl_m": a["min_agl_ft"] * FT, "capture": CAPTURE, "ring_timeout_factor": float(ring_timeout_factor), "bank_ref_deg": BANK_REF_DEG}}
    c["rings"] = [ring_at(c, k) for k in range(min(n_rings, M))]
    c["nominal_time_s"] = nominal_time(c)
    c["time_limit_s"] = TIME_LIMIT_FACTOR * c["nominal_time_s"]
    return c


def leg_time(course, k):
    """Leg time of ring k (s) = |centre_k - centre_{k-1}| / v_tas_ms; leg 0 runs from start.pos. Per-ring timeout of ring
    k = ring_timeout_factor * leg_time(course, k), counted from the resolution time of ring k-1 (t0 for ring 0)."""
    a = course["start"]["pos"] if k == 0 else ring_at(course, k - 1)["centre_m"]
    return math.dist(a, ring_at(course, k)["centre_m"]) / course["params"]["v_tas_ms"]


def regenerate(prov_or_block):
    """Course from a trajectory's provenance (course_seed, stage, aircraft, version [, M, params...]). Raises if the
    version is not geometry-compatible with this module."""
    b = prov_or_block
    p = b.get("provenance", b)
    ver = p.get("version") or b.get("version")
    if ver not in GEOMETRY_COMPATIBLE:
        raise ValueError(f"course version {ver!r} not reproducible by {VERSION} (compatible: {GEOMETRY_COMPATIBLE})")
    par = b.get("params", {})
    c = make_course(p.get("aircraft") or b["model"], p.get("stage") or b["stage"], int(p.get("course_seed", b.get("seed"))),
                    M=b.get("M"), rim_tube_m=par.get("tube_m", 0.0),
                    ring_timeout_factor=par.get("ring_timeout_factor", RING_TIMEOUT_FACTOR))
    if "provenance" in b:
        c["provenance"] = dict(b["provenance"])
    return c


def check_rings(course, rings, tol_m=1e-3):
    """Max |difference| (m / unit) between regenerated rings and embedded ones (centre_m, normal, radius_m)."""
    worst = 0.0
    for r in rings:
        q = ring_at(course, int(r["k"]))
        for key in ("centre_m", "normal"):
            worst = max(worst, max(abs(x - y) for x, y in zip(q[key], r[key])))
        worst = max(worst, abs(q["radius_m"] - r["radius_m"]))
    if worst > tol_m:
        raise ValueError(f"embedded rings differ from regenerated course by {worst}")
    return worst


def public(course):
    """JSON-safe copy without the memo."""
    return {k: v for k, v in course.items() if not k.startswith("_")}


def _cross(p0, p1, ring):
    c, n = ring["centre_m"], ring["normal"]
    s0 = sum((p0[i] - c[i]) * n[i] for i in range(3))
    s1 = sum((p1[i] - c[i]) * n[i] for i in range(3))
    if not (s0 < 0.0 <= s1):
        return None
    f = s0 / (s0 - s1)
    q = [p0[i] + f * (p1[i] - p0[i]) for i in range(3)]
    v = [q[i] - c[i] for i in range(3)]
    vn = sum(v[i] * n[i] for i in range(3))
    rho = math.sqrt(max(0.0, sum((v[i] - vn * n[i]) ** 2 for i in range(3))))
    return f, q, rho


def gate_check(p0, p1, ring):
    """Pure test of one CG segment against one ring: None (no forward plane crossing) or
    {result: pass|rim|miss, frac, point_m, rho_m, rho_norm, miss_m}. rim only when tube_m > 0."""
    x = _cross(p0, p1, ring)
    if x is None:
        return None
    f, q, rho = x
    r, a = float(ring["radius_m"]), float(ring.get("tube_m", 0.0) or 0.0)
    if a > 0 and abs(rho - r) <= a:
        res = "rim"
    elif rho <= r - a:
        res = "pass"
    else:
        res = "miss"
    return {"result": res, "frac": f, "point_m": q, "rho_m": rho, "rho_norm": rho / r, "miss_m": max(0.0, rho - r)}


def score_course(t, pos, course, rim_crash=None, ring_timeout=True):
    """Sequential scoring (spec 3). t[i] s, pos[i] = [x,y,z] NED m (course frame).
    - ring n resolves on its first forward plane crossing with rho <= CAPTURE*r: pass if rho <= r (rim if tube>0
      and |rho-r| <= tube), else miss;
    - out of order: a forward crossing *inside* ring j (n < j < n+5) before ring n resolves marks n..j-1 'missed_order'
      and resolves j at that crossing; scanning continues at j+1;
    - per-ring timeout (1.2): if no crossing resolves ring n on segment (t[i-1], t[i]] and
      t[i] - t_res > ring_timeout_factor * leg_time(n) (t_res = resolution time of ring n-1, t[0] for ring 0), ring n is
      'timeout' at t[i] (= the first sample past the deadline) and t_res = t[i]; at most one timeout per sample;
    - time limit: at t > time_limit_s every unresolved ring < M is 'missed_time'; finished = all M resolved.
    ring_timeout=False reproduces 1.1 scoring (ER phase4-smoke-s1)."""
    M, lim = course["M"], course["time_limit_s"]
    fac = float(course["params"].get("ring_timeout_factor", RING_TIMEOUT_FACTOR))
    rim_crash = (course["params"]["tube_m"] > 0) if rim_crash is None else rim_crash
    gates, n, crash = [], 0, False
    t0 = t[0]
    t_res = t0
    for i in range(1, len(t)):
        if n >= M or crash:
            break
        if t[i] - t0 > lim:
            break
        hit = None
        for j in window(course, n):
            r = ring_at(course, j)
            g = gate_check(pos[i - 1], pos[i], r)
            if g is None:
                continue
            if g["rho_m"] > CAPTURE * r["radius_m"]:  # outside capture
                continue
            if j == n:
                hit = (j, g)
                break
            if j > n and g["result"] == "pass":  # out of order: later ring crossed *inside* its radius
                hit = (j, g)
                break
        if hit is None:
            if ring_timeout and t[i] - t_res > fac * leg_time(course, n):
                gates.append({"k": n, "id": ring_at(course, n)["id"], "result": "timeout", "t": t[i], "rho_m": None,
                              "miss_m": None, "deadline_t": t_res + fac * leg_time(course, n)})
                n += 1
                t_res = t[i]
            continue
        j, g = hit
        te = t[i - 1] + g["frac"] * (t[i] - t[i - 1])
        for kk in range(n, j):
            gates.append({"k": kk, "id": ring_at(course, kk)["id"], "result": "missed_order", "t": te, "rho_m": None,
                          "miss_m": None})
        gates.append({"k": j, "id": ring_at(course, j)["id"], "result": g["result"], "t": te, "rho_m": g["rho_m"],
                      "rho_norm": g["rho_norm"], "miss_m": g["miss_m"], "point_m": g["point_m"]})
        n = j + 1
        t_res = te
        if g["result"] == "rim" and rim_crash:
            crash = True
    t_end = min(t[-1] - t0, lim)
    finished = n >= M and not crash
    timed_out = (not finished) and (not crash) and (t[-1] - t0) > lim
    if timed_out:
        for kk in range(n, M):
            gates.append({"k": kk, "id": ring_at(course, kk)["id"], "result": "missed_time", "t": t0 + lim,
                          "rho_m": None, "miss_m": None})
    passes = sum(g["result"] == "pass" for g in gates)
    misses = sum(g["result"] in ("miss", "missed_order", "missed_time", "timeout") for g in gates)
    return {"gates": gates, "passes": passes, "misses": misses, "rim": sum(g["result"] == "rim" for g in gates),
            "timeouts": sum(g["result"] == "timeout" for g in gates),
            "M": M, "J_ring_miss": (M - passes) / M,
            "finished": finished, "crash": crash, "resolved": len(gates), "timed_out": timed_out,
            "t_last": max((g["t"] for g in gates), default=None), "t_end": t_end,
            "nominal_time_s": course["nominal_time_s"], "time_limit_s": lim, "version": VERSION}


# ---- frames ---------------------------------------------------------------------------------------------------
def quat_rotate(q, v):
    w, x, y, z = q
    return [(1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - w * z) * v[1] + 2 * (x * z + w * y) * v[2],
            2 * (x * y + w * z) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - w * x) * v[2],
            2 * (x * z - w * y) * v[0] + 2 * (y * z + w * x) * v[1] + (1 - 2 * (x * x + y * y)) * v[2]]


def ned_to_enu(v):
    """Vector NED -> ENU (also positions when both origins coincide)."""
    return [v[1], v[0], -v[2]]


def enu_to_ned(v):
    return [v[1], v[0], -v[2]]


def ned_to_neu(v):
    """NED -> FD fly_course pos frame (x north, y east, z UP): (N, E, -D). Same origin (ground below the start)."""
    return [v[0], v[1], -v[2]]


def neu_to_ned(v):
    """FD (N, E, Up) -> canonical NED (N, E, D = -Up)."""
    return [v[0], v[1], -v[2]]


def fd_pos_to_ned(pos_neu_seq):
    """FD fly_course pos[N,3] (N, E, Up m) -> list of NED points for score_course."""
    return [[p[0], p[1], -p[2]] for p in pos_neu_seq]


def ned_to_traj_enu(p, traj_origin_alt_m, ground_alt_msl_m=0.0):
    """Course NED position -> Sim Bridge trajectory ENU (origin = start lat/lon at traj frame.origin_alt_m MSL).
    Assumes the course ground point is at ground_alt_msl_m (JSBSim default terrain 0 m MSL)."""
    return [p[1], p[0], -p[2] + ground_alt_msl_m - traj_origin_alt_m]


def ned_to_three(p):
    """NED -> the viewer's three.js scene axes (x east, y up, z south) before layout offsets/exaggeration."""
    return [p[1], -p[2], -p[0]]


def guidance_view(pos_ned, q_body_to_ned, course, n, n_ahead=2):
    """Per-step guidance inputs for rings n .. n+n_ahead-1 (k >= M are unscored previews, so n+1 always exists).
    Canonical: NED. Derived: body FRD (from q body->NED [w,x,y,z]) and ENU."""
    qi = [q_body_to_ned[0], -q_body_to_ned[1], -q_body_to_ned[2], -q_body_to_ned[3]]
    out = []
    for k in range(n, n + n_ahead):
        r = ring_at(course, k)
        rel = [r["centre_m"][i] - pos_ned[i] for i in range(3)]
        rng = math.sqrt(sum(x * x for x in rel))
        b = quat_rotate(qi, rel)
        u = (lambda v: [x / rng for x in v]) if rng > 0 else (lambda v: [0.0, 0.0, 0.0])
        out.append({"k": k, "id": r["id"], "scored": r["scored"], "radius_m": r["radius_m"], "range_m": rng,
                    "los_ned": u(rel), "los_enu": u(ned_to_enu(rel)), "los_body_frd": u(b),
                    "bearing_deg": math.degrees(math.atan2(b[1], b[0])),
                    "elevation_deg": math.degrees(math.atan2(-b[2], math.hypot(b[0], b[1]))),
                    "normal_ned": r["normal"], "normal_body_frd": quat_rotate(qi, r["normal"]),
                    "centre_ned": r["centre_m"]})
    return out


def quat_from_body_axes(f, r, d):
    """[w,x,y,z] of body FRD -> world, from the body axes (columns f, r, d) expressed in the world frame."""
    m = [[f[0], r[0], d[0]], [f[1], r[1], d[1]], [f[2], r[2], d[2]]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        return [0.25 * S, (m[2][1] - m[1][2]) / S, (m[0][2] - m[2][0]) / S, (m[1][0] - m[0][1]) / S]
    i = max(range(3), key=lambda a: m[a][a])
    j, k = (i + 1) % 3, (i + 2) % 3
    S = math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k][j] - m[j][k]) / S
    q[1 + i] = 0.25 * S
    q[1 + j] = (m[j][i] + m[i][j]) / S
    q[1 + k] = (m[k][i] + m[i][k]) / S
    return q


# ---- bridge: FD fly_course guidance state (N/E/Up) -> canonical NED guidance inputs (Q-G4: convert ONCE, here) ----
FPS = 0.3048
FD_PASSTHROUGH = ("t", "phi", "theta", "psi", "p", "q", "r", "vc_kts", "vt_fps", "nz", "alpha", "beta", "agl_m", "h_dot_fps")


def quat_from_euler(phi, theta, psi):
    """[w,x,y,z] body FRD -> NED from aerospace ZYX Euler angles (rad)."""
    cr, sr, cp, sp, cy, sy = (math.cos(phi / 2), math.sin(phi / 2), math.cos(theta / 2), math.sin(theta / 2),
                              math.cos(psi / 2), math.sin(psi / 2))
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def guidance_inputs(fd_state, course, n, n_ahead=2):
    """Bridge FD's per-step guidance `state` -> canonical NED inputs for the genome.

    fd_state: FD fly_course `state` dict {t, pos_m [N, E, Up] m, phi, theta, psi (rad), p, q, r, vc_kts, vt_fps, nz,
    alpha, beta, agl_m, h_dot_fps}. A raw-NED caller may give `pos_ned` [N, E, D] instead of `pos_m` (exactly one).
    The only sign flip in the pipeline is here: D = -Up. Neither the genome nor FD flips signs.

    Returns {pos_ned, alt_m (= -D, height above the course origin's ground), gamma_rad, gamma_deg, q_body_to_ned,
    the FD_PASSTHROUGH fields as given, n, rings: [for k = n .. n+n_ahead-1: {k, id, scored, radius_m, range_m,
    los_body_frd, bearing_deg, elevation_deg, los_ned, centre_ned, normal_ned, normal_body_frd}]}.
    gamma = asin(h_dot / V_true) (climb > 0); bearing + = right, elevation + = above (body FRD)."""
    if ("pos_m" in fd_state) == ("pos_ned" in fd_state):
        raise ValueError("give exactly one of pos_m (FD N/E/Up) or pos_ned")
    pos = [float(x) for x in fd_state["pos_ned"]] if "pos_ned" in fd_state else neu_to_ned(
        [float(x) for x in fd_state["pos_m"]])
    vt, hd = float(fd_state["vt_fps"]), float(fd_state["h_dot_fps"])
    gamma = math.asin(max(-1.0, min(1.0, hd / vt))) if vt > 0 else 0.0
    q = quat_from_euler(float(fd_state["phi"]), float(fd_state["theta"]), float(fd_state["psi"]))
    out = {k: fd_state[k] for k in FD_PASSTHROUGH if k in fd_state}
    out.update({"pos_ned": pos, "alt_m": -pos[2], "gamma_rad": gamma, "gamma_deg": math.degrees(gamma),
                "q_body_to_ned": q, "n": n})
    keep = ("k", "id", "scored", "radius_m", "range_m", "los_body_frd", "bearing_deg", "elevation_deg", "los_ned",
            "centre_ned", "normal_ned", "normal_body_frd")
    out["rings"] = [{k: g[k] for k in keep} for g in guidance_view(pos, q, course, n, n_ahead)]
    return out
