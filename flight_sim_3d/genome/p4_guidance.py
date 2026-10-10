"""Phase 4 genome guidance law (callable for FD flexeval_p4.fly_course) + evaluator wiring (FD plant + SB course + ER cost).

Frames: canonical NED (SB ring_course; Q-G4 answered by Grok Bot: SB converts at the bridge). FD state pos_m is (N, E, Up) with
the same origin as SB's course (ground below the start), so p_ned = (x, y, -z). LOS errors use body FRD bearing/elevation
(SB spec v0.3 3a, sim_bridge.ring_course.guidance_inputs). Signs (FD limits file + c172x step check 02:45 PT): +aileron -> +p; +elevator -> nose down; +rudder -> nose LEFT.

Per-aircraft gain scaling: fixed constants from FD v2_results/p4_aircraft_limits.json, reference c172x (scale 1 there):
roll/yaw x p_max_ref/p_max, pitch x q_ref/q (pitch_rate_g_limited_dps), speed x vt_trim_ref/vt_trim, nz protection / n_inst.
Never the flex eta. Genome never computes a cost: score = ER evolution/phase4_eval.score_course + aggregate_courses.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
# Read-only copy of ER's FD pin evolution/_fd_pin_p4cs (== flight-dynamics flexeval_p4.py md5 8c1f9f89...), placed so that
# FD's TEAM-relative paths resolve to the real evolution/ (symlink). Nothing is written into evolution/ or flight-dynamics/.
FD_PIN = os.path.join(HERE, "_p4team", "flight-dynamics")
if not os.path.isdir(FD_PIN):   # repo layout: use the shipped frozen pin directly (the _p4team symlink copy is team-layout only)
    FD_PIN = os.path.join(TEAM, "evolution", "_fd_pin_p4cs")
LIMITS_FILE = os.path.join(FD_PIN, "v2_results", "p4_aircraft_limits.json")
FIDELITY = "full_a1_b2a_cs"
REF = "c172x"
DT = 1.0 / 120.0
G = 9.80665
KT = 0.514444
AGG = {"mode": "blend", "alpha": 0.25, "blend": 0.3}   # ER: 0.7 mean + 0.3 CVaR25
K_DEFAULT = 4
# FBW models (FD Q-FD10 / P4.8): fcs/aileron-cmd-norm and fcs/elevator-cmd-norm are *demands* into the JSBSim FCS, not
# surface positions. f16.xml: roll-rate-norm = 0.31821 * p  -> 1 norm = 1/0.31821 rad/s = 180.06 deg/s roll-rate demand;
# pitch-rate-norm = 6.2 * q (+ 0.02 * corrected nz) -> 1 norm ~ 1/6.2 rad/s = 9.24 deg/s pitch-rate demand (+ = nose down,
# FCS clips + at 0.44). The genome's inner-loop output u is in c172x-equivalent surface units; for FBW models it is sent as
# the rate demand that u produces on the reference: p_dem = u * roll_rate_max_ref, q_dem = u * pitch_rate_g_limited_ref.
FBW = {"f16": {"p_dps_per_norm": math.degrees(1.0 / 0.31821), "q_dps_per_norm": math.degrees(1.0 / 6.2)}}


def aircraft_limits(model: str) -> Dict:
    return json.load(open(LIMITS_FILE))["aircraft"][model]


def gain_scales(model: str) -> Dict[str, float]:
    a, r = aircraft_limits(model), aircraft_limits(REF)
    return {"roll": r["roll_rate_max_dps"] / a["roll_rate_max_dps"],
            "pitch": r["pitch_rate_g_limited_dps"] / a["pitch_rate_g_limited_dps"],
            "speed": r["trim"]["vt_fps"] / a["trim"]["vt_fps"], "n_inst": a["n_inst"]}


def decode_physical(u, genes, model: str) -> Dict[str, float]:
    """Gene decode + per-aircraft limits (fractions of FD's file, so never above the aircraft's limit)."""
    d = {g.name: g.decode(x) for g, x in zip(genes, u)}
    a = aircraft_limits(model)
    d["bank_max_deg"] = d["bank_max_frac"] * a["bank_course_deg"]
    d["nz_max_g"] = min(d["nz_max_frac"] * a["n_inst"], a["n_inst"])
    d["nz_min_g"] = max(1.0 - d["nz_min_frac"] * (1.0 - a["n_profile"][0]), a["n_profile"][0])
    return d


# --- phase4_rings_qs (opt-in): speed-scaled lateral outer loop. Same 29 genes / same u; only the decode of the two lateral
# guidance gains changes. Course offsets scale with R_turn ~ V^2 (SB geometry) while the default k_lat is a fixed deg-bank/deg,
# so jets at the same gene were ~3-4x too weak laterally (see PHASE4_RINGS_GENOME.md "jet scaling"). Exponent 1.5 picked from
# targeted flights (T38 medium k_lat 3.25 -> 31 %, 5.0 -> 96 %, 8.0 -> 73 %; f16 3.78 -> 91 %, 5.5/8 -> 100 %; 737 2.7 -> 93 %, 4 -> 100 %).
QS_EXPONENT = 1.5
QS_KEYS = ("k_lat", "k_lat_rate")


def qs_scale(model: str) -> float:
    RC = _sb()
    return (RC.scales(model)["v_tas_ms"] / RC.scales(REF)["v_tas_ms"]) ** QS_EXPONENT


def decode_physical_qs(u, genes, model, course=None):
    d = decode_physical(u, genes, model)
    f = qs_scale(model)
    for k in QS_KEYS:
        d[k] = d[k] * f
    return d


def qs_tune(model):
    return None


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _sb():
    p = os.path.join(TEAM, "sim-bridge")
    if p not in sys.path:
        sys.path.append(p)
    from sim_bridge import ring_course as RC
    return RC


def make_guidance(gains: Dict[str, float], model: str, sb_course: Dict, tune: Optional[Dict[str, float]] = None):
    """Stateful closure guidance(state, gains, course, model) -> command norms. One closure per flight."""
    RC = _sb()
    g, sc, lim = gains, gain_scales(model), aircraft_limits(model)
    fbw = FBW.get(model)
    if fbw is not None:                       # FBW: demand mapping replaces the surface-authority scale (see FBW above)
        ref = aircraft_limits(REF)
        sc = dict(sc, yaw=sc["roll"], roll=ref["roll_rate_max_dps"] / fbw["p_dps_per_norm"],
                  pitch=ref["pitch_rate_g_limited_dps"] / fbw["q_dps_per_norm"])
    if tune:                                  # opt-in (phase4_rings_qs / experiments): multipliers on the gain scales; None = default
        sc = dict(sc, yaw=sc.get("yaw", sc["roll"]) * tune.get("yaw", 1.0), roll=sc["roll"] * tune.get("roll", 1.0),
                  pitch=sc["pitch"] * tune.get("pitch", 1.0))
    v_ref_kts = float(sb_course["start"]["kcas"])
    thr_trim = float(lim["trim"]["throttle"])   # FD trim throttle at the course start condition (feed-forward)
    # Ring sequencing = SB ring_course/1.2 score_course (spec v0.6): ring n resolves on its first forward plane crossing with
    # rho <= CAPTURE*r (or a later ring j crossed inside its radius); the clock restarts at the INTERPOLATED crossing time
    # te = t_prev + frac*(t - t_prev); timeout when t - t_res > ring_timeout_factor * RC.leg_time(course, n) (true 3-D leg
    # length / v_tas_ms). Speed *command* stays KCAS (v_cmd vs vc_kts below).
    fac = float(sb_course["params"].get("ring_timeout_factor", getattr(RC, "RING_TIMEOUT_FACTOR", 3.0)))
    if hasattr(RC, "leg_time"):
        leg_t = lambda k: RC.leg_time(sb_course, k)   # noqa: E731
    else:                                              # ring_course <= 1.1 fallback (spacing / TAS)
        _lt = sb_course["params"]["spacing_m"] / sb_course["params"].get("v_tas_ms", sb_course["params"]["v_ref_mps"])
        leg_t = lambda k: _lt                          # noqa: E731
    S = {"n": 0, "p_prev": None, "t_prev": None, "t_res": None, "I_phi": 0.0, "I_th": 0.0, "I_v": 0.0, "r_lp": 0.0,
         "eb_prev": None, "ee_prev": None, "eb_d": 0.0, "ee_d": 0.0, "y": {"aileron": 0.0, "elevator": None, "rudder": 0.0},
         "log": []}

    def advance(p, t):
        if S["t_res"] is None:
            S["t_res"] = t                                   # t0 (SB: t_res = t[0] for ring 0)
        n, hit = S["n"], False
        if S["p_prev"] is not None:
            # past the last ring the target keeps sliding (as before this change): SB scoring is finished there, but FD
            # still flies to the time limit and a frozen target behind the aircraft would turn it round into a crash.
            for j in RC.window(sb_course, n) or [n]:
                r = RC.ring_at(sb_course, j)
                gc = RC.gate_check(S["p_prev"], p, r)
                if gc is None or gc["rho_m"] > RC.CAPTURE * r["radius_m"]:
                    continue
                if j == n or gc["result"] == "pass":
                    te = S["t_prev"] + gc["frac"] * (t - S["t_prev"])
                    S["n"], S["t_res"], hit = j + 1, te, True
                    S["log"].append((te, j, gc["result"]))
                    break
            if not hit and t - S["t_res"] > fac * (leg_t(n) if n < sb_course["M"] else leg_t(sb_course["M"] - 1)):   # SB per-ring timeout: ring n missed, window slides
                S["log"].append((t, n, "timeout")); S["n"] = n + 1; S["t_res"] = t
        S["p_prev"], S["t_prev"] = p, t

    def guidance(s, _gains=None, _course=None, _model=None):
        t = s["t"]
        gi0 = RC.guidance_inputs(s, sb_course, S["n"])          # SB bridge (canonical NED, body FRD LOS), read-only
        p = gi0["pos_ned"]
        advance(p, t)
        gi = gi0 if S["n"] == gi0["n"] else RC.guidance_inputs(s, sb_course, S["n"])
        r0, r1 = gi["rings"][0], gi["rings"][1]
        V = max(s["vt_fps"] * 0.3048, 1.0)
        w = g["w_next_ring"] * _clamp(1.0 - (r0["range_m"] / V) / g["t_preview_s"], 0.0, 1.0)   # preview blend n -> n+1
        if w == 0.0:
            brg, elev = r0["bearing_deg"], r0["elevation_deg"]
        else:
            c0, c1 = r0["centre_ned"], r1["centre_ned"]
            aim = [c0[i] + w * (c1[i] - c0[i]) - p[i] for i in range(3)]
            qb = gi["q_body_to_ned"]
            xb, yb, zb = RC.quat_rotate([qb[0], -qb[1], -qb[2], -qb[3]], aim)
            brg = math.degrees(math.atan2(yb, xb))
            elev = math.degrees(math.atan2(-zb, math.hypot(xb, yb)))
        a_f = DT / (0.1 + DT)                                                   # LOS-rate filter, 0.1 s
        if S["eb_prev"] is not None:
            S["eb_d"] += a_f * ((brg - S["eb_prev"]) / DT - S["eb_d"])
            S["ee_d"] += a_f * ((elev - S["ee_prev"]) / DT - S["ee_d"])
        S["eb_prev"], S["ee_prev"] = brg, elev
        # lateral: bank command
        phi_c = _clamp(g["k_lat"] * brg + g["k_lat_rate"] * S["eb_d"], -g["bank_max_deg"], g["bank_max_deg"])
        # vertical: pitch-error command from body elevation, gamma limit, nz protection (scaled by n_inst)
        e_th = g["k_vert"] * elev + g["k_vert_rate"] * S["ee_d"]
        gam = gi["gamma_deg"]
        if gam > g["gamma_max_deg"] and e_th > 0:
            e_th = min(e_th, g["gamma_max_deg"] - gam)
        if gam < -g["gamma_max_deg"] and e_th < 0:
            e_th = max(e_th, -g["gamma_max_deg"] - gam)
        k_nz = 10.0 / max(sc["n_inst"] - 1.0, 0.5)
        if s["nz"] > g["nz_max_g"]:
            e_th -= k_nz * (s["nz"] - g["nz_max_g"])
        elif s["nz"] < g["nz_min_g"]:
            e_th += k_nz * (g["nz_min_g"] - s["nz"])
        phi = math.degrees(s["phi"])
        # inner loops (scaled by fixed per-aircraft constants)
        e_phi = phi_c - phi
        S["I_phi"] = _clamp(S["I_phi"] + e_phi * DT, -200.0, 200.0)
        ail = sc["roll"] * (g["kp_roll"] * e_phi + g["ki_roll"] * S["I_phi"] - g["kd_roll"] * math.degrees(s["p"]))
        S["I_th"] = _clamp(S["I_th"] + e_th * DT, -400.0, 400.0)
        cphi = max(math.cos(s["phi"]), 0.2)
        elv = -sc["pitch"] * (g["kp_pitch"] * e_th + g["ki_pitch"] * S["I_th"] - g["kd_pitch"] * math.degrees(s["q"])) \
            - g["k_elev_bank"] * (1.0 / cphi - 1.0)
        S["r_lp"] += DT / g["tau_washout"] * (s["r"] - S["r_lp"])
        r_w = math.degrees(s["r"] - S["r_lp"])
        r_cmd = G * math.tan(_clamp(s["phi"], -1.4, 1.4)) / V                    # rad/s
        rud = sc.get("yaw", sc["roll"]) * (g["kr_yaw"] * r_w - g["kbeta_yaw"] * math.degrees(s["beta"])) \
            - g["k_turn_coord"] * r_cmd - g["k_ari"] * ail                       # +rudder = nose left
        ail = _clamp(ail, -g["ail_auth"], g["ail_auth"])
        rud = _clamp(rud, -g["rud_auth"], g["rud_auth"])
        elv = _clamp(elv, -1.0, 1.0)
        a = DT / (g["tau_cmd_s"] + DT)
        y = S["y"]
        if y["elevator"] is None:
            y["elevator"] = elv
        for k, v in (("aileron", ail), ("elevator", elv), ("rudder", rud)):
            y[k] += a * (v - y[k])
        # speed
        v_cmd = g["v_cmd_scale"] * v_ref_kts * (1.0 + g["v_turn_comp"] * (1.0 / cphi - 1.0))
        e_v = v_cmd - s["vc_kts"]
        S["I_v"] = _clamp(S["I_v"] + e_v * DT, -500.0, 500.0)
        thr = _clamp(thr_trim + sc["speed"] * (g["kp_spd"] * e_v + g["ki_spd"] * S["I_v"]), 0.0, 1.0)
        return {"aileron": y["aileron"], "elevator": y["elevator"], "rudder": y["rudder"], "throttle": thr}

    guidance.state = S
    return guidance


# ---------------------------------------------------------------------------------------------- evaluator
def _fd():
    sys.dont_write_bytecode = True
    if FD_PIN not in sys.path:
        sys.path.append(FD_PIN)
    import flexeval_p4 as fp4
    import flexeval as _fe
    if not os.path.exists(_fe.EVOLUTION_SIM):   # repo layout: the frozen pin sits inside evolution/ (same fix as evolution/phase4_loop.py)
        _fe.load_sim.__defaults__ = (os.path.join(TEAM, "evolution", "sim.py"),)
        _fe.PHASE1_CONFIG = os.path.join(TEAM, "evolution", "configs", "phase1.json")
        _fe.default_profile.__defaults__ = (None, _fe.PHASE1_CONFIG)
    return fp4


def _er():
    sys.dont_write_bytecode = True
    if TEAM not in sys.path:
        sys.path.append(TEAM)
    from evolution import phase4_eval as PE, rings as ER
    return PE, ER


def cache_key(u, model: str, stage: str, seeds: Sequence[int], K: int, mv: str, preset: str = "phase4_rings") -> str:
    blob = json.dumps({"preset": preset, "fidelity": FIDELITY, "cs_mode": "active", "model": model, "model_version": mv,
                       "stage": stage, "course_seeds": [int(x) for x in seeds], "K": int(K), "agg": AGG,
                       "u": np.ascontiguousarray(u, dtype="<f8").tobytes().hex()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def fly_one(u, genes, model: str, stage: str, seed: int, M: Optional[int] = None, variant: str = "default",
            tune: Optional[Dict[str, float]] = None, gain_hook=None):
    RC, fp4 = _sb(), _fd()
    PE, ER = _er()
    course = RC.make_course(model, stage, seed, M=M)
    gains = decode_physical(u, genes, model) if variant == "default" else decode_physical_qs(u, genes, model, course)
    if gain_hook:
        gains = gain_hook(dict(gains))
    guid = make_guidance(gains, model, course, tune if variant == "default" else qs_tune(model))
    fd_course = {"start": {"alt_ft": course["start"]["alt_ft"], "kcas": course["start"]["kcas"]},
                 "duration_s": course["time_limit_s"], "sb_course_id": f"{model}:{stage}:{seed}"}
    r = fp4.fly_course(None, gains, None, None, None, fd_course, FIDELITY, model=model, guidance=guid, cs_mode="active")
    rings = [ER.Ring(tuple(RC.ring_at(course, k)["centre_neu_m"]), tuple(RC.ring_at(course, k)["normal_neu"]),
                     float(course["params"]["radius_m"])) for k in range(course["M"])]
    traj = {"t": r.get("t", []), "pos": r.get("pos", []), "att": r.get("att", []), "nz": r.get("nz", []),
            "v": (np.asarray(r.get("v_kcas", []), float) * KT).tolist(),     # ER v_ref is KCAS-as-m/s
            "alpha": r.get("alpha", []), "agl": r.get("agl", []),
            # GAP FD<->ER: FD reports flap with rate_max None, ER's surface_terms divides by it. Genome never commands flap,
            # so only rate-limited surfaces (elev/ail/rud) are handed to ER until ER/FD agree.
            "surfaces": {k: v for k, v in r.get("surfaces", {}).items() if (r.get("surface_limits", {}).get(k) or (0, 0, None))[2]},
            "surface_limits": {k: v for k, v in r.get("surface_limits", {}).items() if v[2]}, "status": r["status"], "struct_failed": r.get("struct_failed"),
            "carried": {}}   # GAP: ER to name which FD terms/energy are carried 1:1 (not guessed here)
    if r["status"] in ("geometry_gate", "margin_fail", "trim_failed") or len(traj["t"]) < 2:
        return {"cost": None, "status": r["status"], "hard_fail": r["status"], "pass_rate": 0.0, "terms": {},
                "model_version": r.get("model_version")}, r, course
    sco = PE.score_course(traj, rings, model, start=[0.0, 0.0, course["start"]["alt_ft"] * 0.3048])
    sco["model_version"] = r["model_version"]
    sco["sb_diag"] = {k: v for k, v in RC.score_course(r["t"], RC.fd_pos_to_ned(r["pos"]), course).items() if k != "gates"}
    sco["guidance_log"] = guid.state["log"]
    return sco, r, course


def evaluate(u, genes, model: str, stage: str, seeds: Sequence[int], K: int = K_DEFAULT) -> Dict:
    if len(seeds) != K:
        raise ValueError(f"need K={K} shared course seeds, got {len(seeds)}")
    PE, _ = _er()
    res = [fly_one(u, genes, model, stage, s)[0] for s in seeds]
    if any(x["cost"] is None for x in res):
        return {"cost": None, "status": "pre_flight_fail", "per_course": res}   # FD reject path: ER owns the fail cost
    agg = PE.aggregate_courses(res, **AGG)
    mv = res[0]["model_version"]
    agg.update(cache_key=cache_key(u, model, stage, seeds, K, mv), stage=stage, course_seeds=list(seeds),
               per_course=[{k: x[k] for k in ("cost", "status", "pass_rate", "n_pass")} for x in res])
    return agg


def smoke(model: str = "c172x", stage: str = "easy", seed: Optional[int] = None):
    import phase4_rings as R
    P = R.load_preset()
    RC = _sb()
    seed = RC.train_seeds(1, 0, K=1)[0] if seed is None else seed
    sco, r, course = fly_one(R.identity_u(P["genes"]), P["genes"], model, stage, seed)
    return {"model": model, "stage": stage, "seed": seed, "M": course["M"], "status": r["status"], "t_end": r.get("t_end"),
            "model_version": r.get("model_version"), "cost_ER_single_course": sco["cost"], "pass_rate": sco["pass_rate"],
            "n_pass": sco.get("n_pass"), "terms": sco.get("terms"), "sb_diag": sco.get("sb_diag"),
            "guidance_log": sco.get("guidance_log")}


if __name__ == "__main__":
    out = smoke()
    json.dump(out, open(os.path.join(HERE, "runs", "p4_smoke_identity_c172x.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k not in ("terms", "guidance_log")}, default=str, indent=1))
