"""Scenario sets for robustness objectives.

  legacy  sim.make_scenarios(n, seed) unchanged: scenario 0 calm, the rest seeded
          wind + Gauss-Markov turbulence + one 1-cosine gust.
  robust  the legacy disturbance draws, plus per-scenario perturbations that the
          jsbsim_ext backend really simulates: payload mass delta and CG shift
          (JSBSim point mass), controller-side sensor noise, optional heading
          steps. Scenario 0 stays calm and nominal.

Task conditions (``apply_conditions``) are layered on top of either set for
the non-legacy presets: aircraft model + its benchmark altitude/speed (the
legacy 4000 -> 4200 -> 4000 ft profile is shifted to the aircraft's altitude),
altitude-reference ramp rate, reference-rate feed-forward, pitch-command clamp
and envelope limits. With no conditions the legacy set is returned untouched
(the exact sim.Scenario objects), which keeps the legacy preset bit-identical.

Aircraft *variants inside one scenario set* (mixing JSBSim models in one
fitness) are still NOT generated: requesting them raises NotImplementedError.
Run one task per aircraft instead (run_evolve.py --aircraft ...).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np

from flightsim_path import orig_sim

ROBUST_DEFAULTS = {
    "payload_delta_lb": [-150.0, 250.0],   # uniform range, added to point mass 0 (c172x: pilot seat)
    "payload_index": 0,
    "cg_shift_in": [-4.0, 6.0],            # uniform range, moves that point mass in body X
    "noise_alt_ft": [0.0, 3.0],            # 1-sigma ranges, drawn per scenario
    "noise_hdot_fps": [0.0, 0.5],
    "noise_theta_deg": [0.0, 0.2],
    "noise_q_dps": [0.0, 0.3],
    "heading_steps": None,                 # e.g. [[0, 0], [20, 30]] to add a heading change
    "aircraft_variants": [],               # reserved; not implemented
}


def make_scenarios(n: int, seed: int, cfg: Dict = None) -> List:
    cfg = dict(cfg or {"set": "legacy"})
    S = orig_sim()
    base = S.make_scenarios(n, seed)
    if cfg.get("set", "legacy") == "legacy":
        return base
    if cfg["set"] != "robust":
        raise ValueError(f"unknown scenario set {cfg['set']!r}")
    p = {**ROBUST_DEFAULTS, **{k: v for k, v in cfg.items() if k != "set"}}
    if p["aircraft_variants"]:
        raise NotImplementedError("aircraft variants need per-model trim/envelope from Flight Dynamics")
    from sim_ext import ExtScenario
    rng = np.random.default_rng([seed, 0xC0FFEE])
    out = []
    for i, b in enumerate(base):
        d = asdict(b)
        d["steps"] = [tuple(x) for x in d["steps"]]
        e = ExtScenario(**d, label="nominal calm" if i == 0 else f"perturbed {i}")
        if i > 0:
            u = lambda key: float(rng.uniform(*p[key]))
            e.payload_delta_lb = u("payload_delta_lb")
            e.payload_index = int(p["payload_index"])
            e.cg_shift_in = u("cg_shift_in")
            e.noise_alt_ft, e.noise_hdot_fps = u("noise_alt_ft"), u("noise_hdot_fps")
            e.noise_theta_deg, e.noise_q_dps = u("noise_theta_deg"), u("noise_q_dps")
            e.noise_seed = int(rng.integers(2**31))
        if p["heading_steps"]:
            e.heading_steps = [tuple(x) for x in p["heading_steps"]]
        out.append(e)
    return out


CONDITION_KEYS = ("aircraft", "h0_ft", "speed_kts", "ramp_fpm", "alt_ref_ff", "pitch_cmd_limits_deg", "min_kcas", "nz_limits",
                  "throttle_max", "flex_mode", "flex_substeps", "ramp_accel_g",
                  "steps_rel_ft", "duration_s", "alt_err_scale_ft", "max_alt_err_ft",
                  "bank_cmd_limit_deg", "hdg_i_limit_deg")


def apply_conditions(scens: List, cond: Dict) -> List:
    """Return ExtScenario copies with task conditions applied (no-op if ``cond`` is empty)."""
    cond = {k: v for k, v in (cond or {}).items() if v is not None}
    if not cond:
        return scens
    bad = set(cond) - set(CONDITION_KEYS)
    if bad:
        raise ValueError(f"unknown scenario conditions {sorted(bad)}")
    from sim_ext import ExtScenario
    out = []
    for sc in scens:
        d = asdict(sc)
        d["steps"] = [tuple(x) for x in d["steps"]]
        if "heading_steps" in d:
            d["heading_steps"] = [tuple(x) for x in d["heading_steps"]]
        if "h0_ft" in cond:
            dh = float(cond["h0_ft"]) - d["h0_ft"]
            d["steps"] = [(t, h + dh) for t, h in d["steps"]]
        if "steps_rel_ft" in cond:  # shared task: target profile relative to the start altitude
            h0 = float(cond.get("h0_ft", d["h0_ft"]))
            d["steps"] = [(float(t), h0 + float(dh)) for t, dh in cond["steps_rel_ft"]]
        for k, v in cond.items():
            if k == "steps_rel_ft":
                continue
            d[k] = tuple(v) if isinstance(v, list) else v
        out.append(ExtScenario(**d))
    return out
