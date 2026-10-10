"""Phase 4 ring-course geometry (Evolution side, opt-in; imports nothing from sim/genome/fidelity/eval).

Ring = centre c (m, NED-agnostic world frame, z up), unit normal n (direction of travel), radius r (m).
Crossing detection is pure geometry: signed distance d_i = (p_i - c).n; a forward crossing is d_i < 0 <= d_{i+1};
the crossing point is linearly interpolated, f = d_i / (d_i - d_{i+1}), p* = p_i + f (p_{i+1} - p_i), t* likewise.
Order rule: ring k is only searched AFTER the crossing time of ring k-1 (or of the last scored ring), so out-of-order
or backwards crossings never count. A ring is passed if its first forward crossing after t_prev with radial offset
<= capture * r has radial offset <= r. If no crossing exists, the miss distance is the closest approach of the
trajectory (after t_prev) to the ring centre.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

G = 9.80665
KT = 0.514444
FT = 0.3048


@dataclass(frozen=True)
class Ring:
    centre: tuple
    normal: tuple
    radius: float

    def arrays(self):
        c = np.asarray(self.centre, float)
        n = np.asarray(self.normal, float)
        return c, n / np.linalg.norm(n), float(self.radius)


def _ring(obj) -> Ring:
    if isinstance(obj, Ring):
        return obj
    return Ring(tuple(obj["centre"]), tuple(obj["normal"]), float(obj["radius"]))


def ring_crossings(t, pos, rings: Sequence, capture: float = 4.0) -> List[Dict]:
    """Per ring: {passed, crossed, t, rho, rho_norm, idx}. rho = radial offset at the crossing (or closest approach)."""
    t = np.asarray(t, float)
    p = np.asarray(pos, float)
    out = []
    t_prev = -math.inf
    for k, rg in enumerate(rings):
        c, n, r = _ring(rg).arrays()
        d = (p - c) @ n
        hit = None
        for i in np.nonzero((d[:-1] < 0.0) & (d[1:] >= 0.0))[0]:
            f = d[i] / (d[i] - d[i + 1])
            tc = t[i] + f * (t[i + 1] - t[i])
            if tc <= t_prev:
                continue
            pc = p[i] + f * (p[i + 1] - p[i])
            v = pc - c
            rho = float(np.linalg.norm(v - (v @ n) * n))
            if rho <= capture * r:
                hit = (float(tc), rho, int(i))
                break
        if hit is not None:
            tc, rho, i = hit
            out.append({"ring": k, "crossed": True, "passed": rho <= r, "t": tc, "rho": rho, "rho_norm": rho / r, "idx": i})
            t_prev = tc
        else:
            m = t > t_prev
            dist = np.linalg.norm(p[m] - c, axis=1) if m.any() else np.array([math.inf])
            rho = float(dist.min())
            out.append({"ring": k, "crossed": False, "passed": False, "t": None, "rho": rho, "rho_norm": rho / r, "idx": None})
    return out


# ---- per-aircraft scaling (real numbers from configs/phase3b2a_smoke.json phase2_* profiles) ----
AIRCRAFT = {
    "c172x": {"v_ref_kts": 100.0, "v_min_kts": 55.0, "nz": (-1.0, 3.8), "h0_ft": 4000.0, "min_agl_ft": 500.0},
    "T38": {"v_ref_kts": 300.0, "v_min_kts": 180.0, "nz": (-3.0, 7.33), "h0_ft": 10000.0, "min_agl_ft": 500.0},
    "737": {"v_ref_kts": 250.0, "v_min_kts": 195.0, "nz": (-1.0, 2.5), "h0_ft": 10000.0, "min_agl_ft": 500.0},
    # f16 (Phase 4 amendment 02:45 PT) = sim_bridge.ring_course AIRCRAFT f16 (v_ref 350 KCAS, h0 10 000 ft, min AGL 500 ft);
    # v_min / nz from configs/phase1_hdg.json phase1_f16 + FD p4_aircraft_limits.json n_profile. STAGES are relative.
    "f16": {"v_ref_kts": 350.0, "v_min_kts": 200.0, "nz": (-3.0, 9.0), "h0_ft": 10000.0, "min_agl_ft": 500.0},
}
# bank / alpha / v_max come from fd_limits() (FD p4_aircraft_limits.json), not from this table
OVERSPEED_FACTOR = 1.25  # v_max = 1.25 v_ref until FD gives VNE/MMO per model (see proposal section 7)


# ISA troposphere CAS -> TAS (= sim_bridge.ring_course 1.1 cas_to_tas; test asserts equality). Geometry and T_ref use
# TAS at the start altitude h0 (decision 2026-10-07 02:53 PT); V_ref is still given as KCAS.
ISA_T0, ISA_P0, ISA_L, ISA_R, GAMMA = 288.15, 101325.0, 0.0065, 287.05287, 1.4
ISA_A0 = math.sqrt(GAMMA * ISA_R * ISA_T0)


def cas_to_tas(v_cas_ms: float, h_m: float) -> float:
    T = ISA_T0 - ISA_L * h_m
    p = ISA_P0 * (T / ISA_T0) ** (G / (ISA_R * ISA_L))
    a = math.sqrt(GAMMA * ISA_R * T)
    qc = ISA_P0 * ((1.0 + 0.2 * (v_cas_ms / ISA_A0) ** 2) ** 3.5 - 1.0)
    return math.sqrt(5.0 * ((qc / p + 1.0) ** (2.0 / 7.0) - 1.0)) * a


def scales(model: str) -> Dict[str, float]:
    a = AIRCRAFT[model]
    v = cas_to_tas(a["v_ref_kts"] * KT, a["h0_ft"] * FT)      # TAS at h0 (m/s)
    # sustained-turn radius at the 30 deg reference bank used for course geometry
    r_turn = v * v / (G * math.tan(math.radians(30.0)))
    L = fd_limits()[model]
    # speed LIMITS are calibrated (KCAS, as m/s CAS for phase4_eval's KCAS*KT input): overspeed = min(1.25 v_ref_kcas, FD
    # v_max_kcas), underspeed = v_min_kcas. Geometry (v_ref/v_tas_ms/r_turn) is TAS.
    return {"v_ref": v, "v_tas_ms": v, "r_turn": r_turn,
            "v_max": min(OVERSPEED_FACTOR * a["v_ref_kts"], float(L["v_max_kcas"])) * KT, "v_min": a["v_min_kts"] * KT,
            "v_max_kcas": min(OVERSPEED_FACTOR * a["v_ref_kts"], float(L["v_max_kcas"])), "mach_max": L.get("mach_max")}


_LIM = {}


def fd_limits() -> Dict:
    """Single source for per-aircraft limits: FD v2_results/p4_aircraft_limits.json from the frozen pin (EVOLUTION_FD_DIR)."""
    if not _LIM:
        import json, os
        fd = os.environ.get("EVOLUTION_FD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fd_pin_p4cs"))
        _LIM.update(json.load(open(os.path.join(fd, "v2_results", "p4_aircraft_limits.json")))["aircraft"])
    return _LIM


def bank_course_deg(model: str) -> float:
    return float(fd_limits()[model]["bank_course_deg"])


def alpha_stall_deg(model: str) -> float:
    return float(fd_limits()[model]["alpha_stall_deg"])


# ---- curriculum (geometry in aircraft-relative units) ----
STAGES = {
    #           spacing [s of flight at v_ref], lateral offset [x R_turn], vertical offset [x spacing], radius [s at v_ref]
    "easy":   {"spacing_s": 20.0, "lat_rt": 0.05, "vert_sp": 0.02, "radius_s": 0.60},
    "medium": {"spacing_s": 14.0, "lat_rt": 0.15, "vert_sp": 0.05, "radius_s": 0.40},
    "hard":   {"spacing_s": 10.0, "lat_rt": 0.30, "vert_sp": 0.08, "radius_s": 0.25},
}
STAGE_ORDER = ("easy", "medium", "hard")


def curriculum_stage(gen: int, generations: int, pass_history: Sequence[float] = (), threshold: float = 0.8,
                     streak: int = 2) -> str:
    """Stage = max(generation floor, pass-rate promotion). Floor: thirds of the run. Promotion: the gen-best mean pass
    rate >= threshold for `streak` consecutive gens at the current stage moves up one stage. Deterministic."""
    floor = min(2, (3 * gen) // max(generations, 1))
    st, run = 0, 0
    for g, pr in enumerate(pass_history[:gen]):
        st = max(st, min(2, (3 * g) // max(generations, 1)))
        run = run + 1 if pr >= threshold else 0
        if run >= streak and st < 2:
            st, run = st + 1, 0
    return STAGE_ORDER[max(st, floor)]


def curriculum_stage_gated(gen: int, pass_history: Sequence[float] = (), threshold: float = 0.8, streak: int = 2) -> str:
    """Pass-rate gate only (Corleone 2026-10-07 04:51 PT): start 'easy'; advance one stage after the gen-best pass rate is
    >= threshold for `streak` consecutive generations AT the current stage (the streak resets on promotion); no
    generation floor; never goes back. Stage of gen g depends on pass_history[:g] only. Deterministic."""
    st, run = 0, 0
    for pr in pass_history[:gen]:
        run = run + 1 if pr >= threshold else 0
        if run >= streak and st < len(STAGE_ORDER) - 1:
            st, run = st + 1, 0
    return STAGE_ORDER[st]


def course_seed(run_seed: int, gen: int, k: int, holdout: bool = False) -> int:
    tag = f"p4ring|{'H' if holdout else 'T'}|{run_seed}|{gen if not holdout else -1}|{k}".encode()
    return int.from_bytes(hashlib.sha256(tag).digest()[:4], "little")


def make_course(model: str, stage: str, seed: int, n_rings: int = 5, start=(0.0, 0.0, None)) -> List[Ring]:
    """Reference generator (Evolution draft; Sim Bridge's generator replaces it for the smoke). Starts straight ahead
    along +x at h0, then each ring is spacing ahead along the current track with a bounded lateral/vertical offset."""
    s, st, a = scales(model), STAGES[stage], AIRCRAFT[model]
    rng = np.random.default_rng(seed)
    sp = st["spacing_s"] * s["v_ref"]
    h0 = (start[2] if start[2] is not None else a["h0_ft"] * FT)
    pos = np.array([start[0], start[1], h0], float)
    head = np.array([1.0, 0.0, 0.0])
    rings = []
    for _ in range(n_rings):
        lat = np.array([-head[1], head[0], 0.0])
        nxt = pos + sp * head + rng.uniform(-1, 1) * st["lat_rt"] * s["r_turn"] * lat \
            + np.array([0, 0, rng.uniform(-1, 1) * st["vert_sp"] * sp])
        nrm = (nxt - pos) / np.linalg.norm(nxt - pos)
        rings.append(Ring(tuple(float(x) for x in nxt), tuple(float(x) for x in nrm), float(st["radius_s"] * s["v_ref"])))
        head = np.array([nrm[0], nrm[1], 0.0]) / max(np.hypot(nrm[0], nrm[1]), 1e-9)
        pos = nxt
    return rings


def course_length(rings: Sequence, start) -> float:
    pts = [np.asarray(start, float)] + [_ring(r).arrays()[0] for r in rings]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])))
