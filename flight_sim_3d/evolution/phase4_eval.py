"""Phase 4 ring-course scoring (opt-in; old presets never import this). See analysis/PHASE4_SCORING_PROPOSAL.md.

trajectory dict (SI, z up): t [s] (N,), pos [m] (N,3), att [rad] (N,3) phi/theta/psi, nz [g] (N,), v [m/s] (N,)
true/cal airspeed, alpha [rad] (N,) optional, agl [m] (N,) optional (else pos z - ground_z), surfaces {name: deg (N,)},
surface_limits {name: (min_deg, max_deg, rate_max_dps)}, status str ("ok" or FD/sim status), struct_failed bool,
carried {J_*: float} FD structural / flutter / energy terms already in cost units (added 1:1).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

from . import rings as R

P4_TERM_KEYS = ("J_ring_miss", "J_ring_acc", "J_time", "J_defl_rms", "J_rate_rms", "J_sat", "J_chatter",
                "J_g", "J_bank", "J_aoa", "J_overspeed", "J_underspeed", "J_carried")
WEIGHTS = {"J_ring_miss": 1.0, "J_ring_acc": 0.30, "J_time": 0.10, "J_defl_rms": 0.02, "J_rate_rms": 0.02,
           "J_sat": 0.05, "J_chatter": 0.03, "J_g": 0.50, "J_bank": 0.20, "J_aoa": 0.30, "J_overspeed": 0.30,
           "J_underspeed": 0.30, "J_carried": 1.0}
HARD_FAIL_BASE = 10.0
SAT_FRAC = 0.98
CHATTER_REF_HZ = 1.0
CHATTER_MIN_RATE_FRAC = 0.05   # rate reversals smaller than 5 % of rate_max are not chatter
# FD P4.10 (v2_results/p4_aircraft_limits.json bank_course_deg / alpha_stall_deg); the old 45 deg attitude fail does not apply
BANK_COURSE_DEG = {m: R.bank_course_deg(m) for m in ("c172x", "T38", "737", "f16")}   # single source: FD limits file
ALPHA_MAX_DEG = {m: R.alpha_stall_deg(m) for m in ("c172x", "T38", "737", "f16")}
BANK_SOFT_DEG = BANK_COURSE_DEG   # back-compat name (now per aircraft)


def _acc_q(rho_norm: float, passed: bool) -> float:
    """Soft pass bonus: 0.5 rho^2 inside (0 at centre, 0.5 at rim); outside 0.5 + 0.25 min(rho-1, 2) (continuous, 1 cap)."""
    if passed:
        return 0.5 * rho_norm * rho_norm
    return 0.5 + 0.25 * min(max(rho_norm - 1.0, 0.0), 2.0)


def surface_terms(t, surfaces: Dict, limits: Dict) -> Dict[str, float]:
    t = np.asarray(t, float)
    dt = np.diff(t)
    if not surfaces:
        return {"J_defl_rms": 0.0, "J_rate_rms": 0.0, "J_sat": 0.0, "J_chatter": 0.0}
    dr, rr, sat, ch = [], [], [], []
    T = float(t[-1] - t[0])
    for name in sorted(surfaces):
        d = np.asarray(surfaces[name], float)
        lo, hi, rmax = limits[name]
        if rmax is None:          # FD reports flaps with rate None: not a guidance surface, excluded from all surface terms
            continue
        half = max(abs(lo), abs(hi))
        dr.append(math.sqrt(float(np.mean((d / half) ** 2))))
        rate = np.diff(d) / dt
        rr.append(math.sqrt(float(np.mean((rate / rmax) ** 2))))
        at = ((d >= SAT_FRAC * hi) | (d <= SAT_FRAC * lo)).astype(float)
        sat.append(float(np.sum(0.5 * (at[:-1] + at[1:]) * dt) / T))
        big = np.abs(rate) > CHATTER_MIN_RATE_FRAC * rmax
        s = np.sign(rate[big])
        rev = int(np.sum(s[1:] != s[:-1])) if s.size > 1 else 0
        ch.append(min((rev / 2.0) / T / CHATTER_REF_HZ, 1.0))   # reversal cycles per s, normalised, capped
    if not dr:
        return {"J_defl_rms": 0.0, "J_rate_rms": 0.0, "J_sat": 0.0, "J_chatter": 0.0}
    return {"J_defl_rms": float(np.mean(dr)), "J_rate_rms": float(np.mean(rr)), "J_sat": float(np.mean(sat)),
            "J_chatter": float(np.mean(ch))}


def _tmean(t, x):
    t = np.asarray(t, float); x = np.asarray(x, float)
    return float(np.sum(0.5 * (x[:-1] + x[1:]) * np.diff(t)) / (t[-1] - t[0]))


def score_course(traj: Dict, course: Sequence, model: str, start=None, weights: Optional[Dict] = None) -> Dict:
    w = dict(WEIGHTS, **(weights or {}))
    t = np.asarray(traj["t"], float)
    pos = np.asarray(traj["pos"], float)
    N = len(course)
    sc = R.scales(model)
    a = R.AIRCRAFT[model]
    finite = all(np.all(np.isfinite(np.asarray(traj[k], float))) for k in ("t", "pos", "v", "nz"))
    agl = np.asarray(traj["agl"], float) if "agl" in traj else pos[:, 2] - float(traj.get("ground_z", 0.0))
    xs = R.ring_crossings(t, pos, course) if finite else [{"passed": False, "rho_norm": 3.0, "crossed": False, "t": None}] * N
    n_pass = sum(x["passed"] for x in xs)
    hard = None
    if not finite or traj.get("status") == "diverged":
        hard = "divergence"
    elif traj.get("struct_failed") or traj.get("status") == "structural_failure":
        hard = "structural_failure"
    elif float(np.min(agl)) <= 0.0 or traj.get("status") == "ground":
        hard = "ground_impact"
    if hard:
        cost = HARD_FAIL_BASE + (1.0 - n_pass / N)
        return {"cost": cost, "status": hard, "hard_fail": hard, "n_pass": n_pass, "n_rings": N, "pass_rate": n_pass / N,
                "terms": {k: 0.0 for k in P4_TERM_KEYS}, "crossings": xs}
    J = {}
    J["J_ring_miss"] = (N - n_pass) / N
    J["J_ring_acc"] = float(np.mean([_acc_q(x["rho_norm"], x["passed"]) for x in xs]))
    # agility: time to the last ring scored against the course length flown at v_ref; faster than v_ref earns nothing
    # (clamped at 0) so overspeed is never rewarded; missed final ring -> time to end of run.
    s0 = start if start is not None else pos[0]
    t_end = xs[-1]["t"] if xs[-1]["t"] is not None else float(t[-1])
    t_ref = R.course_length(course, s0) / sc["v_ref"]
    J["J_time"] = min(max(0.0, (t_end - t[0]) / t_ref - 1.0), 1.0)
    J.update(surface_terms(t, traj.get("surfaces") or {}, traj.get("surface_limits") or {}))
    nz = np.asarray(traj["nz"], float)
    lo, hi = a["nz"]
    J["J_g"] = _tmean(t, np.maximum(0, nz - hi) / hi + np.maximum(0, lo - nz) / abs(lo))
    phi = np.degrees(np.abs(np.asarray(traj["att"], float)[:, 0]))
    B = BANK_COURSE_DEG[model]
    J["J_bank"] = _tmean(t, np.maximum(0, phi - B) / B)
    if "alpha" in traj:
        al = np.degrees(np.asarray(traj["alpha"], float))
        J["J_aoa"] = _tmean(t, np.maximum(0, al - ALPHA_MAX_DEG[model]) / ALPHA_MAX_DEG[model])
    else:
        J["J_aoa"] = 0.0
    v = np.asarray(traj["v"], float)
    J["J_overspeed"] = _tmean(t, np.maximum(0, v - sc["v_max"]) / sc["v_max"])
    J["J_underspeed"] = _tmean(t, np.maximum(0, sc["v_min"] - v) / sc["v_min"])
    J["J_carried"] = float(sum((traj.get("carried") or {}).values()))
    cost = float(sum(w[k] * J[k] for k in P4_TERM_KEYS))
    return {"cost": cost, "status": "ok", "hard_fail": None, "n_pass": n_pass, "n_rings": N, "pass_rate": n_pass / N,
            "terms": J, "crossings": xs}


def aggregate_courses(results: Sequence[Dict], mode: str = "blend", alpha: float = 0.25, blend: float = 0.3) -> Dict:
    """Genome cost over K courses: mean, cvar (mean of worst ceil(alpha K)), or blend = (1-b) mean + b cvar.
    Order-independent (sorted), so the result is deterministic for any course evaluation order."""
    c = np.sort(np.array([r["cost"] for r in results], float))
    k = max(1, math.ceil(alpha * len(c)))
    mean = math.fsum(c) / len(c); cvar = math.fsum(c[-k:]) / k   # fsum: exact, order-free
    cost = {"mean": mean, "cvar": cvar, "blend": (1 - blend) * mean + blend * cvar}[mode]
    n = len(results)
    terms = {key: math.fsum(r["terms"][key] for r in results) / n for key in P4_TERM_KEYS}
    hf = sorted({r["hard_fail"] for r in results if r["hard_fail"]})
    return {"cost": float(cost), "mean": mean, "cvar": cvar, "mode": mode, "K": len(c),
            "pass_rate": math.fsum(r["pass_rate"] for r in results) / n, "terms": terms,
            "status": "ok" if not hf else hf[0], "hard_fails": hf}
