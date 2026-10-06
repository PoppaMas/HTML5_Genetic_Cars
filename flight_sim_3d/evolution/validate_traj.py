#!/usr/bin/env python3
"""Validate ga-flightsim-traj/1 trajectory files and their index.json.

usage:
    python -m evolution.validate_traj PATH [PATH ...] [--min-gens-per-aircraft 3]

PATH may be a trajectory file, an index.json, a trajectories/ dir or a run dir.
Exit code 0 = all valid. Only needs the standard library + numpy.

Checks: schema/version string, required fields and types, `units` block,
frame block, channel list (required names present, unique, row widths), finite
values, 30 Hz monotonic time base, control ranges (surfaces -1..1, throttle
0..1), Euler angles plausibly in radians, unit quaternions, quaternion vs
phi/theta/psi agreement (quat is the source of truth), and -- when the extra
body-velocity channels ub/vb/wb are present -- that rotating the body velocity
by the quaternion reproduces the ENU velocity (an independent check of the
quaternion convention), plus position vs integrated velocity. For index.json:
entry shape {generation, fitness, aircraft, file}, file exists, and the entry
matches the file it points to.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

SCHEMA = "ga-flightsim-traj/1"
REQUIRED_CHANNELS = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m",
                     "phi", "theta", "psi", "throttle", "elevator", "aileron", "rudder"]
UNITS = {"phi": "rad", "theta": "rad", "psi": "rad", "controls": "norm -1..1, throttle 0..1"}
TOP = {"schema": str, "run_id": str, "aircraft": str, "jsbsim_version": str, "git_sha": str, "seed": int,
       "generation": int, "fitness": (int, float), "genome": dict, "frame": dict, "units": dict, "dt_s": (int, float),
       "sample_hz": (int, float), "target": dict, "events": list, "channels": list, "data": list}
FRAME = {"origin_lat_deg": (int, float), "origin_lon_deg": (int, float), "origin_alt_m": (int, float),
         "axes": str, "attitude": str}
# NED -> ENU fixed rotation
M_EN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

TOL_QNORM = 1e-5
TOL_QUAT_EULER_RAD = 2e-4      # rotation angle between quat and quat(phi,theta,psi)
TOL_VEL_MPS = 0.05             # R(q) v_body vs v_enu (abs) ...
TOL_VEL_REL = 2e-3             # ... + relative to speed
TOL_POS_M = 0.5                # finite-difference position vs mean velocity, per sample


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """q: (N,4) [w,x,y,z] -> (N,3,3) rotation matrices (body -> world)."""
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def euler_to_matrix_ned(phi, theta, psi) -> np.ndarray:
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    return np.stack([
        np.stack([ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp], -1),
        np.stack([ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp], -1),
        np.stack([-st, sf * ct, cf * ct], -1),
    ], -2)


def rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Rotation angle between two rotation matrices; ||Ra-Rb||_F = 2*sqrt(2)*sin(angle/2) (accurate near 0)."""
    f = np.sqrt(np.einsum("nij,nij->n", Ra - Rb, Ra - Rb))
    return 2 * np.arcsin(np.clip(f / (2 * np.sqrt(2)), 0, 1))


def validate_doc(doc: Dict, name: str = "") -> List[str]:
    err: List[str] = []
    E = lambda m: err.append(f"{name}: {m}")  # noqa: E731
    for k, ty in TOP.items():
        if k not in doc:
            E(f"missing field {k!r}")
        elif not isinstance(doc[k], ty) or (ty is int and isinstance(doc[k], bool)):
            E(f"field {k!r} has type {type(doc[k]).__name__}")
    if err:
        return err
    if doc["schema"] != SCHEMA:
        E(f"schema {doc['schema']!r} != {SCHEMA!r}")
    if doc["units"] != UNITS:
        E(f"units {doc['units']!r} != {UNITS!r}")
    for k, ty in FRAME.items():
        if not isinstance(doc["frame"].get(k), ty):
            E(f"frame.{k} missing or wrong type")
    if doc["frame"].get("axes") != "ENU metres, x=east y=north z=up":
        E("frame.axes text mismatch")
    if doc["frame"].get("attitude") != "quat body->ENU [w,x,y,z]":
        E("frame.attitude text mismatch")
    if not isinstance(doc["target"].get("alt_m"), (int, float)):
        E("target.alt_m missing")
    if not all(isinstance(v, (int, float)) for v in doc["genome"].values()) or not doc["genome"]:
        E("genome must be a non-empty {name: number} map")
    for i, ev in enumerate(doc["events"]):
        if not (isinstance(ev, dict) and isinstance(ev.get("t"), (int, float)) and isinstance(ev.get("type"), str)
                and isinstance(ev.get("detail"), str)):
            E(f"events[{i}] must be {{t:number, type:str, detail:str}}")
    if doc["sample_hz"] != 30:
        E(f"sample_hz {doc['sample_hz']} != 30")
    if abs(doc["dt_s"] - 1.0 / doc["sample_hz"]) > 1e-9:
        E(f"dt_s {doc['dt_s']} != 1/sample_hz")
    ch = doc["channels"]
    if len(set(ch)) != len(ch):
        E("duplicate channel names")
    missing = [c for c in REQUIRED_CHANNELS if c not in ch]
    if missing:
        E(f"missing channels {missing}")
        return err
    rows = doc["data"]
    if len(rows) < 2:
        E("fewer than 2 samples")
        return err
    if any(not isinstance(r, list) or len(r) != len(ch) for r in rows):
        E("row width != len(channels)")
        return err
    try:
        d = np.asarray(rows, dtype=float)
    except (TypeError, ValueError):
        E("non-numeric data")
        return err
    if not np.all(np.isfinite(d)):
        E("non-finite values in data")
        return err
    c = {n: d[:, i] for i, n in enumerate(ch)}
    dt = np.diff(c["t"])
    if c["t"][0] != 0 or np.any(dt <= 0):
        E("t must start at 0 and increase")
    elif np.any(np.abs(dt[:-1] - doc["dt_s"]) > 1e-3) or dt[-1] > doc["dt_s"] + 1e-3:
        E("t is not a uniform dt_s grid (only the last interval may be shorter)")
    for k in ("elevator", "aileron", "rudder"):
        if np.any(np.abs(c[k]) > 1 + 1e-9):
            E(f"{k} outside -1..1")
    if np.any(c["throttle"] < -1e-9) or np.any(c["throttle"] > 1 + 1e-9):
        E("throttle outside 0..1")
    if np.any(np.abs(c["phi"]) > math.pi + 1e-6) or np.any(np.abs(c["theta"]) > math.pi / 2 + 1e-6) \
            or np.any(c["psi"] < -math.pi - 1e-6) or np.any(c["psi"] > 2 * math.pi + 1e-6):
        E("phi/theta/psi out of radian range (degrees exported?)")
    q = np.stack([c["qw"], c["qx"], c["qy"], c["qz"]], 1)
    qn = np.linalg.norm(q, axis=1)
    if np.max(np.abs(qn - 1)) > TOL_QNORM:
        E(f"quaternion not unit (max |1-|q|| = {np.max(np.abs(qn - 1)):.2e})")
    R_eb = quat_to_matrix(q / qn[:, None])
    R_eb_euler = M_EN @ euler_to_matrix_ned(c["phi"], c["theta"], c["psi"])
    ang = rot_angle(R_eb, R_eb_euler)
    if np.max(ang) > TOL_QUAT_EULER_RAD:
        E(f"quaternion disagrees with phi/theta/psi (max {np.max(ang):.2e} rad)")
    if all(k in c for k in ("ub", "vb", "wb")):
        v_body = np.stack([c["ub"], c["vb"], c["wb"]], 1)
        v_enu = np.stack([c["vx"], c["vy"], c["vz"]], 1)
        dv = np.linalg.norm(np.einsum("nij,nj->ni", R_eb, v_body) - v_enu, axis=1)
        lim = TOL_VEL_MPS + TOL_VEL_REL * np.linalg.norm(v_enu, axis=1)
        if np.any(dv > lim):
            E(f"R(q)*v_body != v_ENU (max {np.max(dv):.3f} m/s): quaternion convention wrong")
    pos = np.stack([c["x"], c["y"], c["z"]], 1)
    vel = np.stack([c["vx"], c["vy"], c["vz"]], 1)
    pred = pos[:-1] + 0.5 * (vel[:-1] + vel[1:]) * dt[:, None]
    if np.max(np.linalg.norm(pred - pos[1:], axis=1)) > TOL_POS_M:
        E("position inconsistent with velocity (wrong axes/units?)")
    if abs(c["z"][0]) > 1e-3 or abs(c["x"][0]) > 1e-3 or abs(c["y"][0]) > 1e-3:
        E("trajectory does not start at the frame origin")
    if np.max(np.abs(c["alt_msl_m"] - (c["z"] + doc["frame"]["origin_alt_m"]))) > 0.01:
        E("alt_msl_m != z + origin_alt_m")
    if all(k in c for k in ("target_alt_m", "target_cmd_alt_m", "target_rate_mps")):
        # ramped reference: its slope must match target_rate_mps, it must be at the command whenever at rest,
        # and it may never overshoot the command it is moving toward
        slope = np.diff(c["target_alt_m"]) / dt
        mid = 0.5 * (c["target_rate_mps"][:-1] + c["target_rate_mps"][1:])
        chg = np.abs(np.diff(c["target_cmd_alt_m"])) > 1e-6
        bad = (np.abs(slope - mid) > 0.02 + 0.02 * np.abs(mid)) & ~chg
        if np.any(bad):
            E(f"target_alt_m slope != target_rate_mps (max {np.max(np.abs(slope - mid)[~chg]):.3f} m/s)")
        r0 = np.abs(c["target_rate_mps"]) < 1e-9
        rest = np.append(r0[:-1] & r0[1:], r0[-1])  # at rest and not just starting to move (command-change sample)
        if np.any(np.abs(c["target_alt_m"][rest] - c["target_cmd_alt_m"][rest]) > 0.002):
            E("reference at rest away from the command")
    return err


def validate_index(path: str, min_gens: int = 0) -> List[str]:
    err: List[str] = []
    with open(path) as f:
        idx = json.load(f)
    entries = idx["entries"] if isinstance(idx, dict) else idx
    if not isinstance(entries, list) or not entries:
        return [f"{path}: index must be {{'entries': [...]}} or a non-empty list"]
    base = os.path.dirname(path)
    per_ac: Dict[str, set] = {}
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or set(e) != {"generation", "fitness", "aircraft", "file"}:
            err.append(f"{path}: entry {i} must have exactly generation, fitness, aircraft, file")
            continue
        if not isinstance(e["generation"], int) or not isinstance(e["fitness"], (int, float)) \
                or not isinstance(e["aircraft"], str) or not isinstance(e["file"], str):
            err.append(f"{path}: entry {i} has wrong types")
            continue
        fp = os.path.join(base, e["file"])
        if not os.path.exists(fp):
            err.append(f"{path}: entry {i} file {e['file']} missing")
            continue
        with open(fp) as f:
            doc = json.load(f)
        for k in ("generation", "fitness", "aircraft"):
            if doc.get(k) != e[k]:
                err.append(f"{path}: entry {i} {k}={e[k]!r} but file says {doc.get(k)!r}")
        per_ac.setdefault(e["aircraft"], set()).add(e["generation"])
    for ac, gens in per_ac.items():
        if len(gens) < min_gens:
            err.append(f"{path}: {ac} has {len(gens)} generations saved, need >= {min_gens}")
    return err


def collect(path: str):
    if os.path.isdir(os.path.join(path, "trajectories")):
        path = os.path.join(path, "trajectories")
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.startswith("traj_") and f.endswith(".json"))
        idx = os.path.join(path, "index.json")
        return files, ([idx] if os.path.exists(idx) else [])
    if os.path.basename(path) == "index.json":
        return [], [path]
    return [path], []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--min-gens-per-aircraft", type=int, default=0)
    a = ap.parse_args(argv)
    errors, nf, ni = [], 0, 0
    for p in a.paths:
        files, idxs = collect(p)
        if os.path.isdir(p) and not idxs:
            errors.append(f"{p}: no index.json")
        for f in files:
            nf += 1
            with open(f) as fh:
                errors += validate_doc(json.load(fh), os.path.basename(f))
        for i in idxs:
            ni += 1
            errors += validate_index(i, a.min_gens_per_aircraft)
    for e in errors:
        print("INVALID", e)
    print(f"checked {nf} trajectory file(s), {ni} index file(s): {'OK' if not errors else f'{len(errors)} error(s)'}")
    return 1 if errors or nf + ni == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
