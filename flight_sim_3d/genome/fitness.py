"""Multi-objective fitness for the flight-controller GA.

Per scenario, a set of named objectives is computed from the sim result and
(optionally) its telemetry. Each objective declares the telemetry channels it
needs; if the backend does not provide them, the objective is SKIPPED and
reported in ``result["skipped"]`` -- never synthesized.

Two modes:
  scalar (default)  cost_s = sum_k w_k * obj_k  per scenario, then aggregated
                    over scenarios (mean | worst | cvar | mean_cvar). Drop-in for
                    evolve.py (``result["cost"]``).
  pareto            additionally returns an objective vector (``result["pareto"]``)
                    and a constraint violation for nsga2.py. ``cost`` is still
                    filled with the scalar so logs/plots keep working.

Envelope violations keep the legacy hard penalty: the scenario's cost is
sim.py's FAIL_BASE * (1 + fraction not flown) and all its objectives are NaN;
in Pareto mode the individual is infeasible (constraint-domination).

Legacy weights {"track_alt": 1.0, "effort": 2.0} with mean aggregation
reproduce sim.evaluate's cost bit-for-bit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

G_FPS2 = 32.174

# --------------------------------------------------------------------------- objectives


def wrap180(a):
    """Angle difference(s) in degrees wrapped to [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def hold_mask(tel: Dict[str, np.ndarray], settle_s: float = 5.0) -> np.ndarray:
    """Hold samples: the altitude reference has finished moving (reference == commanded altitude, reference rate 0)
    for at least ``settle_s`` seconds. Each contiguous ref-complete stretch is one hold window; the first settle_s
    seconds of every window are excluded (capture transient). Independent of the controller, so it can't be gamed
    by never 'capturing'."""
    t = tel["t"]
    done = (np.abs(tel["target"] - tel["target_cmd"]) < 1e-6) & (tel["target_rate"] == 0.0)
    out = np.zeros(t.size, bool)
    start = None
    for i, d in enumerate(done):
        if d and start is None:
            start = t[i]
        elif not d:
            start = None
        if d and t[i] >= start + settle_s - 1e-9:
            out[i] = True
    return out


def hold_windows(tel: Dict[str, np.ndarray], settle_s: float = 5.0) -> List[np.ndarray]:
    m = hold_mask(tel, settle_s)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    return np.split(idx, cuts)


def obj_hold_osc(tel, res, p):
    """RMS altitude error about its per-window mean over the hold windows / hold_ref_ft (oscillation, not offset)."""
    wins = hold_windows(tel, p.get("hold_settle_s", 5.0))
    if not wins:
        return 0.0
    e = tel["target_cmd"] - tel["h"]
    dev = np.concatenate([e[w] - np.mean(e[w]) for w in wins])
    return float(np.sqrt(np.mean(dev ** 2))) / p.get("hold_ref_ft", 5.0)


def _tv_rate(x: np.ndarray, duration: float) -> float:
    return float(np.sum(np.abs(np.diff(x))) / duration) if x.size > 1 else 0.0


def _lowpass(x: np.ndarray, dt: float, tau: float) -> np.ndarray:
    a = dt / (tau + dt)
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    return y


def comfort_terms(tel: Dict[str, np.ndarray], res: Dict, p: Dict) -> Dict[str, float]:
    """Passenger-comfort sub-metrics, each normalized by a reference so ~1 means 'noticeable'."""
    dt = res["dt"]
    dn = tel["nz"] - 1.0
    nz_f = _lowpass(tel["nz"], dt, p.get("jerk_filter_s", 0.2))
    jerk = np.diff(nz_f) / dt  # g/s
    dtheta = np.abs(tel["theta"] - res["theta_trim"])
    pitch_ex = np.maximum(0.0, dtheta - p.get("pitch_free_deg", 5.0))
    q_ex = np.maximum(0.0, np.abs(tel["q"]) - p.get("q_free_dps", 3.0))
    return {
        "rms_dn": float(np.sqrt(np.mean(dn ** 2))) / p.get("rms_dn_ref_g", 0.1),
        "max_dn": float(np.max(np.abs(dn))) / p.get("max_dn_ref_g", 0.3),
        "jerk": float(np.sqrt(np.mean(jerk ** 2))) / p.get("jerk_ref_gps", 0.2) if jerk.size else 0.0,
        "pitch_excess": float(np.sqrt(np.mean(pitch_ex ** 2))) / p.get("pitch_ref_deg", 1.0),
        "pitch_rate_excess": float(np.sqrt(np.mean(q_ex ** 2))) / p.get("q_ref_dps", 1.0),
    }


DEFAULT_COMFORT_WEIGHTS = {"rms_dn": 1.0, "max_dn": 0.5, "jerk": 0.25, "pitch_excess": 1.0, "pitch_rate_excess": 0.5}


def obj_comfort(tel, res, p):
    terms = comfort_terms(tel, res, p)
    w = p.get("comfort_weights", DEFAULT_COMFORT_WEIGHTS)
    return float(sum(w.get(k, 0.0) * v for k, v in terms.items()))


FAIL_COST = 2000.0  # = 2 * sim.FAIL_BASE: aeroelastic / structural hard fails rank with the worst envelope failures
FLEX_DEFAULTS = {"w_bm_rms": 0.25, "w_bm_peak": 2.0, "w_tip": 1.0, "w_twist": 1.0, "w_mass": 0.3,
                 "w_flutter": 1.0, "w_div": 1.0, "margin_req": 1.2}  # = flexwing.StructWeights defaults (FD)


def flex_structural_terms(tel, fp: Dict, p: Dict) -> Dict[str, float]:
    """FD-style structural terms from flex telemetry (flexwing.response_terms + a wing-mass term).

    The RMS term is relative to the 1-g root moment (independent of the stiffness genes), the peak term only
    bites above the limit load (n_limit * M_1g * EI scale), and the mass term charges the structural mass that
    the stiffness genes add -- so "stiffer" is not free.
    """
    w = {**FLEX_DEFAULTS, **{k: v for k, v in p.items() if k in FLEX_DEFAULTS}}
    M, m1g, lim = tel["wing_root_bending"], fp["wing_root_bending_1g"], fp["wing_root_bending_limit"]
    peak = float(np.max(np.abs(M)))
    return {
        "bm_rms": w["w_bm_rms"] * float(np.sqrt(np.mean((M - m1g) ** 2))) / abs(m1g),
        "bm_peak": w["w_bm_peak"] * max(0.0, peak / lim - 1.0) ** 2,
        "tip": w["w_tip"] * max(0.0, float(np.max(np.abs(tel["tip_deflection"]))) / fp["tip_deflection_limit"] - 1.0) ** 2,
        "twist": w["w_twist"] * max(0.0, float(np.max(np.abs(tel["tip_twist"]))) / fp["tip_twist_limit"] - 1.0) ** 2,
        "mass": w["w_mass"] * fp["wing_mass_frac_delta"],
    }


def obj_structural(tel, res, p):
    """Wing structure: FD flex-model terms if the run was coupled, else a rigid load-factor proxy (M_root ~ n * W * y_cp).

    Rigid proxy = peak fraction of limit load factor + fatigue-ish RMS term (no structure genes, no mass term).
    """
    fp = res.get("flex_params")
    if fp and "wing_root_bending" in tel:
        return float(sum(flex_structural_terms(tel, fp, p).values()))
    n_neg, n_pos = p.get("n_limits", (-1.0, 3.8))
    nz = tel["nz"]
    peak = max(float(np.max(nz)) / n_pos, float(np.min(nz)) / n_neg if n_neg < 0 else 0.0)
    rms = float(np.sqrt(np.mean((nz - 1.0) ** 2))) / (n_pos - 1.0)
    return peak + p.get("fatigue_weight", 1.0) * rms


@dataclass(frozen=True)
class Objective:
    name: str
    channels: Tuple[str, ...]          # telemetry needed ( () = legacy result fields only )
    fn: Callable
    axis: str
    doc: str


OBJECTIVES: Dict[str, Objective] = {o.name: o for o in [
    Objective("track_alt", (), lambda tel, r, p: r["track"], "pitch",
              "legacy ITAE-like altitude error (sim.py 'track')"),
    Objective("effort", (), lambda tel, r, p: r["effort"], "pitch",
              "legacy elevator total variation per second (sim.py 'effort')"),
    Objective("track_speed", ("vc", "v_target"),
              lambda tel, r, p: float(np.mean(np.abs(tel["vc"] - tel["v_target"]))) / p.get("speed_ref_kt", 5.0),
              "speed", "mean |airspeed error| / 5 kt"),
    Objective("effort_throttle", ("throttle",),
              lambda tel, r, p: _tv_rate(tel["throttle"], r["n_steps"] * r["dt"]), "speed", "throttle TV per second"),
    Objective("track_heading", ("psi", "psi_target"),
              lambda tel, r, p: float(np.mean(np.abs((tel["psi_target"] - tel["psi"] + 180.0) % 360.0 - 180.0))) / p.get("hdg_ref_deg", 5.0),
              "roll", "mean |heading error| / 5 deg"),
    Objective("track_heading_rms", ("psi", "psi_target"),
              lambda tel, r, p: float(np.sqrt(np.mean(wrap180(tel["psi_target"] - tel["psi"]) ** 2))) / p.get("hdg_rms_ref_deg", 5.0),
              "roll", "RMS heading error (wrapped to +-180) / hdg_rms_ref_deg (5 deg); Phase-1 heading-hold term"),
    Objective("hold_osc", ("t", "h", "target", "target_cmd", "target_rate"), obj_hold_osc, "pitch",
              "v5: RMS altitude error about its window mean in the hold windows (ref complete + settle_s) / hold_ref_ft"),
    Objective("track_bank", ("phi", "phi_cmd"),
              lambda tel, r, p: float(np.mean(np.abs(tel["phi"] - tel["phi_cmd"]))) / p.get("bank_ref_deg", 5.0),
              "roll", "mean |bank error| / 5 deg"),
    Objective("effort_aileron", ("aileron",),
              lambda tel, r, p: _tv_rate(tel["aileron"], r["n_steps"] * r["dt"]), "roll", "aileron TV per second"),
    Objective("sideslip", ("beta",),
              lambda tel, r, p: float(np.mean(np.abs(tel["beta"]))) / p.get("beta_ref_deg", 1.0), "yaw", "mean |beta| / 1 deg"),
    Objective("comfort", ("nz", "theta", "q"), obj_comfort, "comfort",
              "RMS/peak |n-1|, jerk, pitch-attitude and pitch-rate excess (weighted, normalized)"),
    Objective("structural", ("nz",), obj_structural, "structure",
              "wing-root bending: flex channel if available else rigid load-factor proxy"),
]}

# Pseudo-objectives over the scenario set (Pareto mode only).
ROBUST_OBJECTIVES = ("robust_cvar", "robust_worst")


# --------------------------------------------------------------------------- config


@dataclass
class FitnessConfig:
    mode: str = "scalar"                                   # scalar | pareto
    weights: Dict[str, float] = field(default_factory=lambda: {"track_alt": 1.0, "effort": 2.0})
    aggregate: Dict = field(default_factory=lambda: {"mode": "mean"})  # mean | worst | cvar | mean_cvar
    params: Dict = field(default_factory=dict)             # comfort refs, n_limits, ...
    pareto_objectives: List[str] = field(default_factory=lambda: ["track_alt", "effort", "comfort"])
    backend: str = "jsbsim_ext"                            # jsbsim_ext | legacy_sim

    def needed(self) -> List[str]:
        names = [k for k, w in self.weights.items() if w]
        if self.mode == "pareto":
            names += [o for o in self.pareto_objectives if o not in ROBUST_OBJECTIVES]
        out = []
        for n in names:
            if n not in OBJECTIVES:
                raise ValueError(f"unknown objective {n!r}; known: {sorted(OBJECTIVES)}")
            if n not in out:
                out.append(n)
        return out

    def needs_telemetry(self) -> bool:
        return any(OBJECTIVES[n].channels for n in self.needed())


def availability(cfg: FitnessConfig, channels: Sequence[str]) -> Dict[str, str]:
    """{objective: reason} for objectives that cannot be evaluated with these channels."""
    have = set(channels)
    return {n: f"missing telemetry {sorted(set(OBJECTIVES[n].channels) - have)}"
            for n in cfg.needed() if not set(OBJECTIVES[n].channels) <= have}


def aggregate(values: Sequence[float], agg: Dict) -> float:
    v = np.asarray(values, dtype=float)
    mode = agg.get("mode", "mean")
    if mode == "mean":
        return float(np.mean(v))
    if mode == "worst":
        return float(np.max(v))
    k = max(1, int(math.ceil(agg.get("alpha", 0.25) * v.size)))
    cvar = float(np.mean(np.sort(v)[-k:]))
    if mode == "cvar":
        return cvar
    if mode == "mean_cvar":
        lam = agg.get("lambda", 0.5)
        return float((1.0 - lam) * np.mean(v) + lam * cvar)
    raise ValueError(f"unknown aggregate mode {mode!r}")


# --------------------------------------------------------------------------- evaluation

LEGACY_KEYS = ("cost", "status", "t_end", "track", "effort")


def score_scenario(res: Dict, cfg: FitnessConfig, needed: Sequence[str]) -> Tuple[float, Dict[str, float], Dict[str, str]]:
    """(scalar cost, objective values, skipped) for one simulated scenario."""
    tel = res.get("telemetry", {})
    skipped = {}
    vals: Dict[str, float] = {}
    for n in needed:
        o = OBJECTIVES[n]
        miss = [c for c in o.channels if c not in tel or tel[c].size == 0 or np.all(np.isnan(tel[c]))]
        if miss:
            skipped[n] = f"missing telemetry {miss}"
            continue
        vals[n] = float(o.fn(tel, res, cfg.params)) if res["status"] == "ok" else float("nan")
    if res["status"] != "ok":
        return res["cost"], vals, skipped  # legacy hard penalty, objectives NaN
    total = 0.0
    for n, w in cfg.weights.items():
        if w and n in vals:
            total += w * vals[n]
    return float(total), vals, skipped


def evaluate(gains: Dict[str, float], scenarios: Sequence, cfg: FitnessConfig, record: Optional[bool] = None) -> Dict:
    """Same contract as sim.evaluate (``cost`` + legacy-shaped ``per_scenario``) plus extra keys."""
    needed = cfg.needed()
    if cfg.backend == "legacy_sim":
        from flightsim_path import orig_sim
        bad = [n for n in needed if OBJECTIVES[n].channels]
        r = orig_sim().evaluate(gains, list(scenarios))
        if bad:
            r["skipped"] = {n: "legacy_sim backend provides no telemetry" for n in bad}
        return r
    import sim_ext
    rec = cfg.needs_telemetry() if record is None else record
    # Flex: pre-simulation aeroelastic screening (FD margin_terms, ~6 ms). Margins are constraints:
    # < 1.0 -> fail without flying; 1.0..margin_req (1.2) -> hinge^2 penalty added once per genome.
    flex_sc = [sc for sc in scenarios if getattr(sc, "flex_mode", None)]
    pre, margin_pen = None, 0.0
    if flex_sc:
        import fd_bridge
        pre = fd_bridge.precheck(flex_sc[0].aircraft, gains, {**FLEX_DEFAULTS, **cfg.params})
        margin_pen = float(pre["terms"]["J_flutter_margin"] + pre["terms"]["J_div_margin"])
        for key in ("flutter_margin", "div_margin"):  # NaN would slip through FD's "< 1.0" test: treat as a fail
            if np.isnan(float(pre["margins"][key])):
                pre = {**pre, "fail": pre["fail"] or key.replace("_margin", "") + "_nan"}
        if pre["margins"].get("margin_error"):  # FD: numerical failure -> margin 0 (already a fail); make it explicit
            pre = {**pre, "fail": pre["fail"] or "margin_error"}
        if pre["fail"]:
            return _aeroelastic_fail(pre, scenarios, cfg)
    sims = [sim_ext.simulate(gains, sc, record=rec) for sc in scenarios]
    for r in sims:  # FD hard fail: root moment above ultimate (1.5 x limit)
        fp = r.get("flex_params")
        tel = r.get("telemetry", {})
        if r["status"] == "ok" and fp and "wing_root_bending" in tel:
            if float(np.max(np.abs(tel["wing_root_bending"]))) > 1.5 * fp["wing_root_bending_limit"]:
                r.update(status="structural_ultimate", cost=FAIL_COST, track=float("nan"), effort=float("nan"))
    costs, objs, skipped = [], [], {}
    for r in sims:
        c, v, sk = score_scenario(r, cfg, needed)
        costs.append(c)
        objs.append(v)
        skipped.update(sk)
    out = {
        "cost": aggregate(costs, cfg.aggregate) + margin_pen,
        "per_scenario": [{**{k: r[k] for k in LEGACY_KEYS}, "cost": float(c)} for r, c in zip(sims, costs)],
        "objectives_per_scenario": objs,
        "skipped": skipped,
    }
    feasible = all(r["status"] == "ok" for r in sims)
    out["violation"] = 0.0 if feasible else float(np.mean([r["cost"] for r in sims if r["status"] != "ok"]))
    agg_obj = {}
    for n in needed:
        col = [v[n] for v in objs if n in v]
        agg_obj[n] = aggregate(col, {"mode": "mean"}) if feasible and col else float("nan")
    out["objectives"] = agg_obj
    if pre is not None:
        out["aeroelastic"] = _pre_summary(pre, margin_pen)
        for v in objs:
            v.setdefault("aeroelastic_margin_penalty", margin_pen)
        out["objectives"]["aeroelastic_margin_penalty"] = margin_pen
    if cfg.mode == "pareto":
        vec = []
        for n in cfg.pareto_objectives:
            if n == "robust_cvar":
                vec.append(aggregate(costs, {"mode": "cvar", "alpha": cfg.aggregate.get("alpha", 0.25)}))
            elif n == "robust_worst":
                vec.append(float(np.max(costs)))
            else:
                v = agg_obj.get(n, float("nan"))
                if n == "structural":
                    v += margin_pen  # margin penalty rides on the structural objective in Pareto mode
                vec.append(v)
        out["pareto"] = vec
    if rec:
        out["diagnostics"] = [diagnostics(r) for r in sims]
    return out


def _pre_summary(pre: Dict, pen: float) -> Dict:
    import fd_bridge
    m = pre["margins"]
    div = float(m["div_margin"])
    cap = fd_bridge.flutter_summary(m)
    return {**cap, "div_margin": min(div, cap["flutter_margin_cap"]) if not np.isnan(div) else None,
            "div_not_found_below_cap": bool(m.get("div_not_found_below_cap", not np.isfinite(div))),
            "margin_penalty": pen, "delta_wing_mass_lb": float(pre["terms"]["delta_wing_mass_lb"]), "fail": pre["fail"]}


def _aeroelastic_fail(pre: Dict, scenarios, cfg: FitnessConfig) -> Dict:
    status = f"aeroelastic_{pre['fail']}"
    per = [{"cost": FAIL_COST, "status": status, "t_end": 0.0, "track": float("nan"), "effort": float("nan")} for _ in scenarios]
    out = {"cost": FAIL_COST, "per_scenario": per, "objectives_per_scenario": [{} for _ in scenarios], "skipped": {},
           "violation": FAIL_COST, "objectives": {n: float("nan") for n in cfg.needed()},
           "aeroelastic": _pre_summary(pre, float("nan"))}
    if cfg.mode == "pareto":
        out["pareto"] = [float("nan")] * len(cfg.pareto_objectives)
    return out


def diagnostics(res: Dict) -> Dict[str, float]:
    """Human-readable flight metrics (not used for selection)."""
    tel = res.get("telemetry")
    if not tel or tel["theta"].size == 0:
        return {"status": res["status"]}
    th = tel["theta"] - res["theta_trim"]
    return {
        "status": res["status"],
        "max_pitch_deg": float(np.max(tel["theta"])),
        "max_pitch_above_trim_deg": float(np.max(th)),
        "min_pitch_below_trim_deg": float(np.min(th)),
        "max_abs_q_dps": float(np.max(np.abs(tel["q"]))),
        "max_nz": float(np.max(tel["nz"])),
        "min_nz": float(np.min(tel["nz"])),
        "rms_dn_g": float(np.sqrt(np.mean((tel["nz"] - 1.0) ** 2))),
        "max_climb_fpm": float(np.max(tel["h_dot"]) * 60.0),
        "max_sink_fpm": float(-np.min(tel["h_dot"]) * 60.0),
        "theta_trim_deg": float(res["theta_trim"]),
        "max_overshoot_ft": overshoot(tel, res.get("steps")),
        "max_ref_err_ft": float(np.max(np.abs(tel["target"] - tel["h"]))),
        **_hold_diag(tel, res),
        **({"hdg_final_deg": float(wrap180(tel["psi"][-1] - tel["psi"][0])),
            "hdg_max_abs_deg": float(np.max(np.abs(wrap180(tel["psi"] - tel["psi"][0])))),
            "max_abs_phi_deg": float(np.max(np.abs(tel["phi"])))}
           if "psi" in tel and tel["psi"].size and np.all(np.isfinite(tel["psi"])) else {}),
        **({"max_root_bm_over_limit": float(np.max(np.abs(tel["wing_root_bending"]))) / res["flex_params"]["wing_root_bending_limit"],
            "max_tip_deflection_ft": float(np.max(np.abs(tel["tip_deflection"]))),
            "max_tip_twist_deg": float(np.max(np.abs(tel["tip_twist"]))),
            "wing_mass_delta_lb": res["flex_params"]["wing_mass_delta_lb"]}
           if res.get("flex_params") and "wing_root_bending" in tel else {}),
    }


def _hold_diag(tel: Dict[str, np.ndarray], res: Dict) -> Dict[str, float]:
    """Hold peak-to-peak (worst window, about the commanded altitude), hold RMS h_dot, and the downdraft residual."""
    out = {}
    if not all(c in tel for c in ("target_cmd", "target_rate")):
        return out
    wins = hold_windows(tel, 5.0)
    e = tel["target_cmd"] - tel["h"]
    if wins:
        out["hold_pp_ft"] = float(max(np.ptp(e[w]) for w in wins))
        out["hold_rms_hdot_fps"] = float(np.sqrt(np.mean(np.concatenate([tel["h_dot"][w] for w in wins]) ** 2)))
    d = res.get("draft")
    if d and tel["t"].size:
        t = tel["t"]
        late = t >= t[-1] - 20.0
        after = t >= d["t_s"]
        out["draft_residual_ft"] = float(np.mean(e[late]))       # + = below the commanded altitude
        # a run that ended (envelope violation) before the onset has no post-onset samples
        out["draft_max_err_ft"] = float(np.max(np.abs(e[after]))) if after.any() else float("nan")
    return out


def overshoot(tel: Dict[str, np.ndarray], steps) -> float:
    """Largest excursion past the *commanded* altitude after each command change (ft, >= 0)."""
    if not steps:
        return float("nan")
    t, h = tel["t"], tel["h"]
    worst = 0.0
    prev = steps[0][1]
    changes = [(ts, hs) for ts, hs in steps[1:]]
    for i, (ts, hs) in enumerate(changes):
        if hs == prev:
            continue
        t_end = changes[i + 1][0] if i + 1 < len(changes) else np.inf
        m = (t >= ts) & (t < t_end)
        if m.any():
            worst = max(worst, float(np.max(np.sign(hs - prev) * (h[m] - hs))))
        prev = hs
    return max(worst, 0.0)
