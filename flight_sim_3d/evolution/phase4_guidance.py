"""Phase 4 ring guidance law -- ER INTERIM implementation of Genome's 29-gene phase4_rings chromosome.

STATUS: INTERIM / ER-AUTHORED. Genome Architect owns the guidance law (FD P4.11); genome/phase4_rings.py ships only the gene
table + operators (its evaluate() is a stub), so this file implements the law literally from the gene descriptions in
genome/PHASE4_RINGS_GENOME.md so the loop can fly. Replace with Genome's callable when it lands (make_guidance() is the seam).

guidance(state, gains, course, model) is called by FD fly_course every 1/120 s. state: pos_m [x N, y E, z up(altitude)],
phi/theta/psi rad, p/q/r rad/s, vc_kts, vt_fps, nz, alpha, beta rad, agl_m, h_dot_fps.
Ring tracking: course is Sim Bridge's NED dict; the tracker resolves ring n with sim_bridge.ring_course.gate_check using the
same window/order/capture rule as ring_course.score_course (forward crossing of n with rho <= CAPTURE r, or a pass of a
later window ring), so guidance only ever sees the active window. Deterministic; one closure per flight (stateful).
Signs (FD P4.2): +aileron = right roll; +elevator = nose down; +rudder = nose LEFT.
"""
from __future__ import annotations

import math
from typing import Dict

D = math.degrees
R = math.radians
G = 9.80665


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def fd_to_ned(pos_up):
    """FD fly_course pos (x N, y E, z UP = altitude above the flat ground, m) -> Sim Bridge course NED (origin on the ground
    below the start, z down). Verified: FD pos[0] = [0, 0, h0_m] and SB start = [0, 0, -h0_m]."""
    return [pos_up[0], pos_up[1], -pos_up[2]]


def make_guidance(genes: Dict[str, float], model: str, limits: Dict, rc, course: Dict, dt: float = 1.0 / 120.0):
    g = genes
    bank_lim = min(g["bank_max_deg"], float(limits["bank_course_deg"]))
    nz_hi = min(g["nz_max_g"], float(limits.get("n_inst", g["nz_max_g"])), float(limits.get("n_struct_limit", 99.0)))
    nz_lo = g["nz_min_g"]
    v_ref_kcas = float(course["start"]["kcas"])
    v_ref_mps = float(course["params"]["v_ref_mps"])
    a_tc = math.exp(-dt / max(g["tau_cmd_s"], dt))
    a_wo = math.exp(-dt / g["tau_washout"])
    S = {"n": 0, "prev": None, "ei_r": 0.0, "ei_p": 0.0, "ei_v": 0.0, "elat0": None, "evert0": None, "rw": 0.0, "r0": 0.0,
         "f": None, "th0": None}

    def track(p):
        prev = S["prev"]
        S["prev"] = p
        if prev is None:
            return
        n = S["n"]
        for j in rc.window(course, n) or [n]:
            ring = rc.ring_at(course, j)
            x = rc.gate_check(prev, p, ring)
            if x is None:
                continue
            if (j == n and x["rho_m"] <= rc.CAPTURE * ring["radius_m"]) or (j > n and x["result"] == "pass"):
                S["n"] = j + 1
                return

    def guidance(s, gains=None, course_=None, model_=None):
        p = rc.neu_to_ned(s["pos_m"])
        track(p)
        n = S["n"]
        c0 = rc.ring_at(course, n)["centre_m"]
        c1 = rc.ring_at(course, n + 1)["centre_m"]
        rel0 = [c0[i] - p[i] for i in range(3)]
        rng0 = math.sqrt(sum(x * x for x in rel0))
        vt = max(s["vt_fps"] * 0.3048, 1.0)
        wb = g["w_next_ring"] * math.exp(-rng0 / (vt * g["t_preview_s"]))
        aim = [(1 - wb) * c0[i] + wb * c1[i] for i in range(3)]
        rel = [aim[i] - p[i] for i in range(3)]
        hz = math.hypot(rel[0], rel[1])
        brg = math.atan2(rel[1], rel[0])
        e_lat = D((brg - s["psi"] + math.pi) % (2 * math.pi) - math.pi)
        gam = math.asin(_clip(s["h_dot_fps"] * 0.3048 / vt, -1.0, 1.0))
        e_vert = D(math.atan2(-rel[2], max(hz, 1.0)) - gam)
        de_lat = 0.0 if S["elat0"] is None else (e_lat - S["elat0"]) / dt
        de_vert = 0.0 if S["evert0"] is None else (e_vert - S["evert0"]) / dt
        S["elat0"], S["evert0"] = e_lat, e_vert
        phi_c = _clip(g["k_lat"] * e_lat + g["k_lat_rate"] * de_lat, -bank_lim, bank_lim)
        gam_c = _clip(g["k_vert"] * e_vert + g["k_vert_rate"] * de_vert, -g["gamma_max_deg"], g["gamma_max_deg"])
        # roll
        e_r = phi_c - D(s["phi"])
        S["ei_r"] = _clip(S["ei_r"] + e_r * dt, -50.0, 50.0)
        ail = g["kp_roll"] * e_r + g["ki_roll"] * S["ei_r"] - g["kd_roll"] * D(s["p"])
        ail = _clip(ail, -g["ail_auth"], g["ail_auth"])
        # pitch: theta_cmd = gamma_cmd + alpha; u_p > 0 = nose up; elevator = -u_p (+elev = nose down)
        th_c = gam_c + D(s["alpha"])
        e_p = th_c - D(s["theta"])
        S["ei_p"] = _clip(S["ei_p"] + e_p * dt, -100.0, 100.0)
        u_p = g["kp_pitch"] * e_p + g["ki_pitch"] * S["ei_p"] - g["kd_pitch"] * D(s["q"])
        u_p += g["k_elev_bank"] * (1.0 / max(math.cos(s["phi"]), 0.2) - 1.0)
        if s["nz"] > nz_hi:
            u_p -= 0.5 * (s["nz"] - nz_hi)
        elif s["nz"] < nz_lo:
            u_p += 0.5 * (nz_lo - s["nz"])
        elev = _clip(-u_p, -1.0, 1.0)
        # yaw: washed-out r damper, sideslip, ARI, turn coordination (all toward nose-right = negative rudder)
        S["rw"] = a_wo * (S["rw"] + s["r"] - S["r0"])
        S["r0"] = s["r"]
        r_cmd = D(G * math.tan(s["phi"]) / vt)
        rud = g["kr_yaw"] * D(S["rw"]) - g["kbeta_yaw"] * D(s["beta"]) - g["k_ari"] * ail - g["k_turn_coord"] * r_cmd * 0.01
        rud = _clip(rud, -g["rud_auth"], g["rud_auth"])
        # speed
        v_c = g["v_cmd_scale"] * v_ref_kcas * (1.0 + g["v_turn_comp"] * max(0.0, s["nz"] - 1.0))
        e_v = v_c - s["vc_kts"]
        S["ei_v"] = _clip(S["ei_v"] + e_v * dt, -200.0, 200.0)
        thr = _clip(0.6 + g["kp_spd"] * e_v + g["ki_spd"] * S["ei_v"], 0.0, 1.0)
        # first-order command prefilter on the surfaces
        u = (ail, elev, rud)
        S["f"] = u if S["f"] is None else tuple(a_tc * f + (1 - a_tc) * x for f, x in zip(S["f"], u))
        return {"aileron": S["f"][0], "elevator": S["f"][1], "rudder": S["f"][2], "throttle": thr}

    guidance.state = S
    return guidance
