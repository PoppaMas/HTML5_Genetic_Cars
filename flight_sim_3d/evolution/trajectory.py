"""Best-trajectory export (format ``ga-flightsim-traj/2``; /1 still accepted by validate_traj) and altitude-hold metrics."""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np

SCHEMA = "ga-flightsim-traj/2"  # /2: full FE wings via v2_map + wing*_modal; see FLEX_STATE_SCHEMA /3
INDEX_SCHEMA = "ga-flightsim-traj-index/1"
REQUIRED_CHANNELS = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m",
                     "phi", "theta", "psi", "throttle", "elevator", "aileron", "rudder"]
UNITS = {"phi": "rad", "theta": "rad", "psi": "rad", "controls": "norm -1..1, throttle 0..1"}
FT = 0.3048

# decimals per channel when writing (mm positions, ~1e-7 quaternion)
_ROUND = {"t": 4, "x": 3, "y": 3, "z": 3, "qw": 8, "qx": 8, "qy": 8, "qz": 8, "vx": 4, "vy": 4, "vz": 4,
          "alt_msl_m": 3, "phi": 7, "theta": 7, "psi": 7, "throttle": 6, "elevator": 6, "aileron": 6,
          "rudder": 6, "target_alt_m": 3, "kcas": 3, "nz": 4, "ub": 4, "vb": 4, "wb": 4, "lat_deg": 9, "lon_deg": 9,
          "target_cmd_alt_m": 3, "target_rate_mps": 5}


CONTROLS_TIMING = "pre_step"
CONTROLS_TIMING_DOC = ("throttle/elevator/aileron/rudder in the row at time t are the commands applied over the sim step that "
                       "STARTS at t (set by the controller from the state at t). A recorder called after the step ending "
                       "at t reads, in fcs/*-cmd-norm, the commands of that ended step (= the previous row's controls). "
                       "The final row repeats the last applied commands. State channels are the state at t.")
# every non-core channel (core = REQUIRED_CHANNELS, documented by frame/units)
CHANNEL_DOC = {
    "target_alt_m": "altitude REFERENCE the controller tracks at t, m MSL (sim.Scenario.target): the ramped reference "
                    "(Phase-1: 600 fpm with 0.1 g corners) moving toward target_cmd_alt_m; equals target_cmd_alt_m when "
                    "the scenario has no ramp. The tracking cost and hold metrics use this channel.",
    "target_cmd_alt_m": "COMMANDED altitude at t, m MSL (sim.Scenario.target_cmd): the instant step input (h0 + steps_rel_ft); "
                        "the reference target_alt_m ramps toward it.",
    "target_rate_mps": "reference climb rate d(target_alt_m)/dt, m/s (+ up); 0 while the reference is at rest or without a ramp.",
    "kcas": "calibrated airspeed, knots (velocities/vc-kts).",
    "nz": "normal load factor, g (accelerations/Nz; +1.0 in level flight, + = pull-up).",
    "ub": "body x velocity (forward), m/s, Earth-relative (velocities/u-fps), body FRD axes.",
    "vb": "body y velocity (right), m/s, Earth-relative (velocities/v-fps), body FRD axes.",
    "wb": "body z velocity (down), m/s, Earth-relative (velocities/w-fps), body FRD axes.",
    "lat_deg": "geodetic latitude, deg (position/lat-geod-deg).",
    "lon_deg": "longitude, deg (position/long-gc-deg).",
}
_STRUCT_DOF_DOC = {"dz": "m, body z (+ down), elastic deflection of the elastic axis vs the undeformed (jig) shape incl. the 1-g trim deflection",
                   "dy": "m, body y (+ right); modal wings = 0.0; htail/vtail/fuselage from v2_map",
                   "dx": "m, body x (+ forward); FE wing in-plane (full, v2_map): dx = -v*0.3048 (FD v + aft)",
                   "twist": "rad, right-hand rotation about the component axis node0->nodeN (wingR + = leading edge up, wingL + = leading edge down)"}


def channel_doc(channels: List[str]) -> Dict[str, str]:
    """{name: description} for every non-core channel in `channels`."""
    out = {}
    for c in channels:
        if c in REQUIRED_CHANNELS:
            continue
        if c in CHANNEL_DOC:
            out[c] = CHANNEL_DOC[c]
        elif c.count(".") == 2:
            comp, dof, i = c.split(".")
            out[c] = f"structure channel: component {comp}, node {i} (structure.components[].axis_nodes_body_m), {dof}: " \
                     + _STRUCT_DOF_DOC.get(dof, "see structure block")
        elif c.startswith("flex."):
            out[c] = _flex_doc(c[5:])
        elif c.startswith("struct."):
            out[c] = ("Sim Bridge scalar from sim-bridge v2_map (V2_MAP_VERSION, FD INTERFACE_v2 §11 signs); "
                      "see structure.v2_map.scalars / SIGN_TABLE")
        else:
            out[c] = "undocumented extra channel"
    return out


_FLEX_UNITS = (("_lbft", "lbf*ft"), ("_lbf", "lbf"), ("_deg", "deg"), ("_ft", "ft"), ("_bm", "lbf*ft (root bending moment)"),
               ("_torque", "lbf*ft (root torque)"), ("_shear", "lbf (root shear)"))


def _flex_doc(key: str) -> str:
    unit = next((u for suf, u in _FLEX_UNITS if suf + "_" in key + "_"), "see flight-dynamics/INTERFACE_v2.md")
    sign = ""
    if "twist" in key:
        sign = "; FD raw sign: + = leading edge up (nose-up) on BOTH wings (Sim Bridge wingL.twist = -this)"
    elif key.startswith("tip_w") or key.endswith("_w_ft"):
        sign = "; FD raw sign: + = up (Sim Bridge dz = -0.3048 * this)"
    return f"Flight Dynamics raw coupler diagnostic '{key}' (FD names and sign convention), {unit}{sign}"


def traj_filename(aircraft: str, run_id: str, gen: int) -> str:
    return f"traj_{aircraft}_{run_id}_g{gen}.json"


def build_doc(*, run_id: str, aircraft: str, jsbsim_version: str, git_sha: str, seed: int, generation: int,
              fitness: float, gains: Dict[str, float], scenario: Dict, scenario_index: int, sim_result: Dict,
              profile: Dict, extra: Optional[Dict] = None) -> Dict:
    tr = sim_result["trajectory"]
    chans = tr["channels"]
    rnd = [_ROUND.get(c, 6) for c in chans]
    data = [[round(float(v), d) for v, d in zip(row, rnd)] for row in tr["data"]]
    events: List[Dict] = [{"t": 0.0, "type": "start", "detail": f"trimmed level flight {scenario['h0_ft']:.0f} ft, "
                                                                f"{scenario['speed_kts']:.0f} KCAS, throttle {tr['trim']['throttle_trim']:.3f}"}]
    prev = scenario["steps"][0][1]
    for ts, hs in scenario["steps"]:
        if hs != prev:
            events.append({"t": float(ts), "type": "target_change", "detail": f"altitude target {prev:.0f} -> {hs:.0f} ft ({hs * FT:.1f} m)"})
            prev = hs
    if scenario.get("discrete_gust_fps"):
        events.append({"t": float(scenario["discrete_gust_t_s"]), "type": "gust",
                       "detail": f"1-cosine vertical gust {scenario['discrete_gust_fps']:+.1f} ft/s (down +) over {scenario['discrete_gust_len_s']:.1f} s"})
    t_end = float(sim_result["t_end"])
    if sim_result["status"] == "ok":
        events.append({"t": t_end, "type": "end", "detail": "completed"})
    else:
        events.append({"t": t_end, "type": "terminated", "detail": sim_result["status"]})
    events.sort(key=lambda e: e["t"])
    o = tr["origin"]
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "aircraft": aircraft,
        "jsbsim_version": jsbsim_version,
        "git_sha": git_sha,
        "seed": int(seed),
        "generation": int(generation),
        "fitness": float(fitness),
        "fitness_sense": "min",
        "fitness_doc": "GA cost, mean over the aircraft's scenarios; lower is better",
        "scenario_index": int(scenario_index),
        "scenario_id": f"{aircraft}:s{int(scenario_index)}",
        "scenario_cost": float(sim_result["cost"]),
        "status": sim_result["status"],
        "genome": {k: float(v) for k, v in gains.items()},
        "frame": {
            "origin_lat_deg": float(o["lat_deg"]), "origin_lon_deg": float(o["lon_deg"]), "origin_alt_m": float(o["alt_m"]),
            "axes": "ENU metres, x=east y=north z=up",
            "attitude": "quat body->ENU [w,x,y,z]",
            "body_axes": "JSBSim body FRD: x forward, y right wing, z down",
            "attitude_source_of_truth": "quaternion; phi/theta/psi are HUD-only (JSBSim Euler ZYX vs local NED)",
        },
        "units": dict(UNITS),
        "dt_s": float(tr["dt_s"]),
        "sim_dt_s": float(tr["sim_dt_s"]),
        "sample_hz": 30 if abs(tr["sample_hz"] - 30) < 1e-9 else tr["sample_hz"],
        "target": {
            "alt_m": float(scenario["steps"][0][1] * FT),
            "steps": [{"t": float(ts), "alt_m": float(hs * FT)} for ts, hs in scenario["steps"]],
            "speed_kcas": float(scenario["speed_kts"]),
            "reference": ("instant steps" if scenario.get("ramp_fpm") is None else
                          f"ramped at {scenario['ramp_fpm']:g} fpm" + (f" with {scenario['ramp_accel_g']:g} g corners"
                                                                       if scenario.get("ramp_accel_g") else "")
                          + " (channel target_alt_m = reference, target_cmd_alt_m = command)"),
        },
        "wind": {"north_mps": scenario["wind_north_fps"] * FT, "east_mps": scenario["wind_east_fps"] * FT,
                 "gust_sigma_mps": scenario["gust_sigma_fps"] * FT},
        "control_conventions": "JSBSim fcs/*-cmd-norm: elevator + = trailing-edge down (nose down); aileron + = right roll; rudder + = per model; throttle 0..1",
        "events": events,
        "controls_timing": tr.get("controls_timing", CONTROLS_TIMING),
        "controls_timing_doc": CONTROLS_TIMING_DOC,
        "channel_doc": channel_doc(chans),
        **({"structure": tr["structure"]} if tr.get("structure") is not None else {}),
        **(extra or {}),
        "channels": chans,
        "data": data,
    }


def write_doc(doc: Dict, traj_dir: str) -> str:
    os.makedirs(traj_dir, exist_ok=True)
    fn = traj_filename(doc["aircraft"], doc["run_id"], doc["generation"])
    tmp = os.path.join(traj_dir, fn + ".tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(traj_dir, fn))
    return fn


def write_index(traj_dir: str, run_id: str, entries: List[Dict]) -> str:
    entries = sorted(entries, key=lambda e: (e["aircraft"], e["generation"]))
    doc = {"schema": INDEX_SCHEMA, "traj_schema": SCHEMA, "run_id": run_id,
           "entries": [{"generation": int(e["generation"]), "fitness": float(e["fitness"]),
                        "aircraft": e["aircraft"], "file": e["file"]} for e in entries]}
    p = os.path.join(traj_dir, "index.json")
    with open(p + ".tmp", "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(p + ".tmp", p)
    return p


def hold_metrics(sim_result: Dict, band_ft: float = 20.0, hold_after_s: float = 20.0) -> Dict:
    """Altitude-hold error metrics from a recorded run (all in ft / s).

    Segments start at *command* changes (``target_cmd_alt_m``; older files without it: ``target_alt_m``).
    hold_*: samples >= hold_after_s after the last command change AND with the reference at rest
    (``target_rate_mps`` == 0), so with a ramped reference the hold window only starts once the
    reference has arrived; there reference == command. Without a ramp this is the original definition.
    settle / overshoot are measured against the command (settle time therefore includes the ramp).
    """
    tr = sim_result["trajectory"]
    ch = {c: i for i, c in enumerate(tr["channels"])}
    d = np.asarray(tr["data"], dtype=float)
    out = {"status": sim_result["status"], "t_end_s": float(sim_result["t_end"])}
    if d.size == 0:
        return out
    t = d[:, ch["t"]]
    h = d[:, ch["alt_msl_m"]] / FT
    ref = d[:, ch["target_alt_m"]] / FT
    cmd = d[:, ch["target_cmd_alt_m"]] / FT if "target_cmd_alt_m" in ch else ref
    rate = d[:, ch["target_rate_mps"]] if "target_rate_mps" in ch else np.zeros_like(t)
    e = h - ref
    e_cmd = h - cmd
    chg = [0] + [i for i in range(1, len(cmd)) if abs(cmd[i] - cmd[i - 1]) > 1e-6]
    t_step = np.zeros_like(t)
    for a, b in zip(chg, chg[1:] + [len(t)]):
        t_step[a:b] = t[a]
    out["rms_err_ft"] = float(np.sqrt(np.mean(e ** 2)))
    hold = ((t - t_step) >= hold_after_s) & (np.abs(rate) < 1e-9)
    out["hold_rms_err_ft"] = float(np.sqrt(np.mean(e[hold] ** 2))) if hold.any() else None
    out["hold_max_abs_err_ft"] = float(np.max(np.abs(e[hold]))) if hold.any() else None
    out["hold_samples"] = int(hold.sum())
    settles, overs = [], []
    for a, b in zip(chg[1:], chg[2:] + [len(t)]):
        step = cmd[a] - cmd[a - 1]
        seg_e, seg_t = e_cmd[a:b], t[a:b]
        outside = np.nonzero(np.abs(seg_e) > band_ft)[0]
        if len(outside) == 0:
            settles.append(0.0)
        elif outside[-1] == len(seg_e) - 1:
            settles.append(None)  # never settled in this segment
        else:
            settles.append(float(seg_t[outside[-1] + 1] - seg_t[0]))
        overs.append(float(max(0.0, np.max(np.sign(step) * seg_e))))
    out["settle_s_per_step"] = settles
    out["settle_s_max"] = None if any(s is None for s in settles) or not settles else max(settles)
    out["overshoot_ft_max"] = max(overs) if overs else None
    return out
