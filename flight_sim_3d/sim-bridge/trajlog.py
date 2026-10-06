#!/usr/bin/env python3
"""Trajectory logger for JSBSim flights of the flight_sim GA (schema ga-flightsim-traj/1).

Non-invasive: the project's own ``sim.simulate()`` is called unchanged. We only
swap ``sim._new_fdm`` for a factory that wraps the real ``jsbsim.FGFDMExec`` in a
thin proxy. Every ``fdm.run()`` call made by sim.py's loop first samples the
current state (read-only property access, so the physics and the cost are
bit-identical to what the GA computes), downsampled to ``sample_hz``.

Only generic JSBSim property names are used (position/*, attitude/*,
velocities/*, fcs/*-cmd-norm, fcs/*-pos-norm), so any aircraft model works.
The aircraft name comes from the caller / CLI (default: whatever the project's
sim.py uses); it is never hardcoded here.

CLI example (log the project's committed best genome for the example run):
    python trajlog.py --best-gains <flight_sim>/results/example/best_gains.json --out /tmp/t.json
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import json
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

SCHEMA = "ga-flightsim-traj/1"
INDEX_SCHEMA = "ga-flightsim-traj-index/1"
DEFAULT_FLIGHT_SIM_DIR = os.environ.get(
    "FLIGHT_SIM_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flight_sim"))

FT = 0.3048
KT = 0.514444

REQUIRED_CHANNELS = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m",
                     "phi", "theta", "psi", "throttle", "elevator", "aileron", "rudder"]
# Channel order of the data rows (required first, then extras). Readers MUST look channels up by name.
CHANNELS = [
    "t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m",
    "phi", "theta", "psi", "throttle", "elevator", "aileron", "rudder",
    # extras (ignored by readers that don't know them)
    "target_alt_m", "kcas", "ktas", "nz", "ub", "vb", "wb", "lat_deg", "lon_deg",
    "p", "q", "r", "alt_agl_m", "alpha", "beta", "elevator_pos", "aileron_pos", "rudder_pos", "throttle_pos",
]

# JSBSim property behind each control channel (cmd = what the controller asked for).
CONTROL_PROPS = {
    "throttle": "fcs/throttle-cmd-norm",
    "elevator": "fcs/elevator-cmd-norm",
    "aileron": "fcs/aileron-cmd-norm",
    "rudder": "fcs/rudder-cmd-norm",
    # surface/actuator positions; aircraft without a given property log NaN->null
    "elevator_pos": "fcs/elevator-pos-norm",
    "aileron_pos": ["fcs/aileron-pos-norm", "fcs/right-aileron-pos-norm"],
    "rudder_pos": "fcs/rudder-pos-norm",
    "throttle_pos": ["fcs/throttle-pos-norm", "fcs/throttle-pos-norm[0]"],
}

# Decimals kept per channel when writing JSON (keeps files ~4x smaller).
PRECISION = {"t": 4, "x": 3, "y": 3, "z": 3, "qw": 8, "qx": 8, "qy": 8, "qz": 8, "vx": 4, "vy": 4, "vz": 4,
             "ub": 4, "vb": 4, "wb": 4, "alt_msl_m": 3, "alt_agl_m": 3, "target_alt_m": 3, "lat_deg": 9,
             "lon_deg": 9, "kcas": 3, "ktas": 3, "nz": 4, "phi": 7, "theta": 7, "psi": 7}
DEFAULT_PRECISION = 6

# `units` must be exactly this block (shared with Evolution Runner's validator).
UNITS = {"phi": "rad", "theta": "rad", "psi": "rad", "controls": "norm -1..1, throttle 0..1"}
# Per-channel units (informational extra field).
CHANNEL_UNITS = {
    "t": "s", "x": "m", "y": "m", "z": "m", "qw": "1", "qx": "1", "qy": "1", "qz": "1",
    "vx": "m/s", "vy": "m/s", "vz": "m/s", "alt_msl_m": "m", "phi": "rad", "theta": "rad", "psi": "rad",
    "throttle": "0..1", "elevator": "-1..1", "aileron": "-1..1", "rudder": "-1..1",
    "target_alt_m": "m", "kcas": "kt", "ktas": "kt", "nz": "g", "ub": "m/s", "vb": "m/s", "wb": "m/s",
    "lat_deg": "deg (geodetic)", "lon_deg": "deg", "p": "rad/s", "q": "rad/s", "r": "rad/s",
    "alt_agl_m": "m", "alpha": "rad", "beta": "rad", "elevator_pos": "-1..1", "aileron_pos": "-1..1",
    "rudder_pos": "-1..1", "throttle_pos": "0..1",
}


# ----------------------------------------------------------------------------
# project import
# ----------------------------------------------------------------------------
def import_flight_sim(flight_sim_dir: Optional[str] = None):
    """Import the project's sim/genome/ga/evolve modules from its directory (no copies)."""
    d = os.path.abspath(flight_sim_dir or DEFAULT_FLIGHT_SIM_DIR)
    if not os.path.isfile(os.path.join(d, "sim.py")):
        raise FileNotFoundError(f"sim.py not found in {d} (set --flight-sim-dir or FLIGHT_SIM_DIR)")
    if d not in sys.path:
        sys.path.insert(0, d)
    mods = {}
    for name in ("sim", "genome", "ga", "evolve"):
        mods[name] = importlib.import_module(name)
        f = os.path.abspath(getattr(mods[name], "__file__", ""))
        if os.path.dirname(f) != d:
            raise ImportError(f"module {name} was imported from {f}, expected {d}")
    return mods


def git_sha(path: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def jsbsim_version() -> Optional[str]:
    try:
        import jsbsim
        return getattr(jsbsim, "__version__", None)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h_m) * np.cos(lat) * np.cos(lon)
    y = (n + h_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - _WGS84_E2) + h_m) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def east_north(lat_deg, lon_deg, lat0, lon0, alt0_m=0.0):
    """East/north (m) from the origin: local equirectangular projection using the WGS84 meridian /
    prime-vertical radii of curvature at the origin, evaluated at the origin altitude (so x/y are
    consistent with integrating the NED velocity at flight altitude). Same convention as
    evolution/sim.py, so both producers give identical x/y for the same flight."""
    lat = math.radians(lat0)
    s2 = math.sin(lat) ** 2
    r_north = _WGS84_A * (1 - _WGS84_E2) / (1 - _WGS84_E2 * s2) ** 1.5 + alt0_m
    r_east = (_WGS84_A / math.sqrt(1 - _WGS84_E2 * s2) + alt0_m) * math.cos(lat)
    e = np.radians(np.asarray(lon_deg, float) - lon0) * r_east
    n = np.radians(np.asarray(lat_deg, float) - lat0) * r_north
    return e, n


def qmul(a, b):
    """Hamilton product of [..., 4] arrays in [w, x, y, z] order."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


_S = math.sqrt(0.5)
Q_ENU_FROM_NED = np.array([0.0, _S, _S, 0.0])  # 180 deg about (1,1,0)/sqrt2: N->y, E->x, D->-z


def euler_to_quat_body_to_enu(phi, theta, psi):
    """Aero ZYX Euler angles (w.r.t. local NED) -> quaternion rotating body-FRD vectors into ENU."""
    phi, theta, psi = (np.asarray(v, float) for v in (phi, theta, psi))
    z = np.zeros_like(phi)
    qz = np.stack([np.cos(psi / 2), z, z, np.sin(psi / 2)], -1)
    qy = np.stack([np.cos(theta / 2), z, np.sin(theta / 2), z], -1)
    qx = np.stack([np.cos(phi / 2), np.sin(phi / 2), z, z], -1)
    q_ned_body = qmul(qmul(qz, qy), qx)
    q = qmul(np.broadcast_to(Q_ENU_FROM_NED, q_ned_body.shape), q_ned_body)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def make_continuous(q):
    """Flip signs so consecutive quaternions stay in the same hemisphere (q and -q are the same rotation)."""
    q = np.array(q, float)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def qrotate(q, v):
    """Rotate vectors v [...,3] by unit quaternions q [...,4]."""
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    qc = q * np.array([1, -1, -1, -1])
    return qmul(qmul(q, qv), qc)[..., 1:]


# ----------------------------------------------------------------------------
# recording proxy
# ----------------------------------------------------------------------------
_RAW_PROPS = {
    "lat": "position/lat-geod-deg", "lon": "position/long-gc-deg", "h_sl_ft": "position/h-sl-ft",
    "h_agl_ft": "position/h-agl-ft", "phi": "attitude/phi-rad", "theta": "attitude/theta-rad",
    "psi": "attitude/psi-rad", "p": "velocities/p-rad_sec", "q": "velocities/q-rad_sec",
    "r": "velocities/r-rad_sec", "vn": "velocities/v-north-fps", "ve": "velocities/v-east-fps",
    "vd": "velocities/v-down-fps", "vc_kts": "velocities/vc-kts", "vtrue_kts": "velocities/vtrue-kts",
    "u": "velocities/u-fps", "v": "velocities/v-fps", "w": "velocities/w-fps",
    "alpha": "aero/alpha-rad", "beta": "aero/beta-rad", "nz": "accelerations/Nz",
}


class Recorder:
    def __init__(self, sample_hz: float = 30.0, target_fn=None):
        self.sample_hz = sample_hz
        self.target_fn = target_fn  # t -> target altitude (ft) or None
        self.rows: List[Dict[str, float]] = []
        self.ticks = 0
        self.dt = None
        self.every = 1
        self.fdm = None
        self._resolved = None

    def attach(self, fdm):
        self.fdm = fdm
        self.dt = fdm.get_delta_t()
        self.every = max(1, int(round(1.0 / (self.dt * self.sample_hz))))

    def _resolve(self, fdm):
        out = {}
        for ch, props in CONTROL_PROPS.items():
            props = [props] if isinstance(props, str) else props
            out[ch] = next((p for p in props if fdm.get_property_manager().hasNode(p)), None)
        self._resolved = out

    def sample(self, fdm, t: float):
        if self._resolved is None:
            self._resolve(fdm)
        row = {k: fdm[p] for k, p in _RAW_PROPS.items()}
        for ch, p in self._resolved.items():
            row[ch] = fdm[p] if p else float("nan")
        row["t"] = t
        row["target_ft"] = self.target_fn(t) if self.target_fn else float("nan")
        self.rows.append(row)

    def before_run(self, fdm):
        if self.ticks % self.every == 0:
            self.sample(fdm, self.ticks * self.dt)
        self.ticks += 1

    def finish(self):
        """Sample the final state (after the last step, or the state that ended the run)."""
        if self.fdm is not None and (not self.rows or self.rows[-1]["t"] < self.ticks * self.dt - 1e-9):
            self.sample(self.fdm, self.ticks * self.dt)


class RecordingFDM:
    """Transparent proxy around jsbsim.FGFDMExec that samples state before each run()."""

    def __init__(self, fdm, recorder: Recorder):
        object.__setattr__(self, "_fdm", fdm)
        object.__setattr__(self, "_rec", recorder)

    def __getitem__(self, k):
        return self._fdm[k]

    def __setitem__(self, k, v):
        self._fdm[k] = v

    def __getattr__(self, name):
        return getattr(self._fdm, name)

    def run(self):
        if self._rec.fdm is None:
            self._rec.attach(self._fdm)
        self._rec.before_run(self._fdm)
        return self._fdm.run()


class _patched_fdm_factory:
    """Context manager: make sim.py build RecordingFDMs (and optionally another aircraft)."""

    def __init__(self, sim, recorder: Optional[Recorder], aircraft: Optional[str]):
        self.sim, self.rec, self.aircraft = sim, recorder, aircraft

    def __enter__(self):
        self._orig_new, self._orig_ac = self.sim._new_fdm, self.sim.AIRCRAFT
        if self.aircraft:
            self.sim.AIRCRAFT = self.aircraft
        orig, rec = self._orig_new, self.rec

        def factory():
            fdm = orig()
            return RecordingFDM(fdm, rec) if rec is not None else fdm

        self.sim._new_fdm = factory
        return self

    def __exit__(self, *exc):
        self.sim._new_fdm, self.sim.AIRCRAFT = self._orig_new, self._orig_ac
        return False


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def fly(mods, gains: Dict[str, float], scenarios, record_index: int = 0, sample_hz: float = 30.0,
        aircraft: Optional[str] = None):
    """Fly all scenarios with sim.simulate (unchanged); record scenario ``record_index``.

    Returns (fitness, per_scenario results, recorder). ``fitness`` is the GA's
    fitness (mean cost over all scenarios, exactly as sim.evaluate computes it).
    """
    sim = mods["sim"]
    results, rec = [], None
    for i, sc in enumerate(scenarios):
        r_i = Recorder(sample_hz, target_fn=lambda t, sc=sc: sc.target(t)[0]) if i == record_index else None
        with _patched_fdm_factory(sim, r_i, aircraft):
            res = sim.simulate(gains, sc)
        if r_i is not None:
            r_i.finish()
            rec = r_i
        results.append(res)
    fitness = float(np.mean([r["cost"] for r in results]))
    return fitness, results, rec


def build_trajectory(mods, rec: Recorder, *, aircraft: str, run_id: str, generation: Optional[int],
                     fitness: float, genome_vec: Sequence[float], scenario, scenario_index: int,
                     scenario_result: Dict, per_scenario: List[Dict], seed=None, extra: Optional[Dict] = None) -> Dict:
    sim, genome = mods["sim"], mods["genome"]
    R = {k: np.array([row[k] for row in rec.rows], float) for k in rec.rows[0]}
    lat0, lon0, alt0 = float(R["lat"][0]), float(R["lon"][0]), float(R["h_sl_ft"][0] * FT)
    e, n = east_north(R["lat"], R["lon"], lat0, lon0, alt0)
    alt = R["h_sl_ft"] * FT
    q = make_continuous(euler_to_quat_body_to_enu(R["phi"], R["theta"], R["psi"]))
    cols = {
        "t": R["t"], "x": e, "y": n, "z": alt - alt0,
        "qw": q[:, 0], "qx": q[:, 1], "qy": q[:, 2], "qz": q[:, 3],
        "vx": R["ve"] * FT, "vy": R["vn"] * FT, "vz": -R["vd"] * FT, "alt_msl_m": alt,
        "phi": R["phi"], "theta": R["theta"], "psi": R["psi"],
        "throttle": R["throttle"], "elevator": R["elevator"], "aileron": R["aileron"], "rudder": R["rudder"],
        "target_alt_m": R["target_ft"] * FT, "kcas": R["vc_kts"], "ktas": R["vtrue_kts"], "nz": R["nz"],
        "ub": R["u"] * FT, "vb": R["v"] * FT, "wb": R["w"] * FT, "lat_deg": R["lat"], "lon_deg": R["lon"],
        "p": R["p"], "q": R["q"], "r": R["r"], "alt_agl_m": R["h_agl_ft"] * FT,
        "alpha": R["alpha"], "beta": R["beta"],
        "elevator_pos": R["elevator_pos"], "aileron_pos": R["aileron_pos"], "rudder_pos": R["rudder_pos"],
        "throttle_pos": R["throttle_pos"],
    }
    assert list(cols) == CHANNELS
    # Drop optional channels that are not fully finite (e.g. a model without fcs/rudder-pos-norm, or no
    # altitude target): files never contain null/NaN. The 19 required channels must be finite.
    channels = [c for c in CHANNELS if c in REQUIRED_CHANNELS or np.all(np.isfinite(cols[c]))]
    bad = [c for c in REQUIRED_CHANNELS if not np.all(np.isfinite(cols[c]))]
    if bad:
        raise ValueError(f"non-finite values in required channels {bad}")
    prec = [PRECISION.get(c, DEFAULT_PRECISION) for c in channels]
    mat = [cols[c] for c in channels]
    data = [[round(float(m[i]), d) for m, d in zip(mat, prec)] for i in range(len(R["t"]))]

    # target / events
    steps = [[float(ts), float(hs) * FT] for ts, hs in scenario.steps]
    events = [{"t": 0.0, "type": "start", "detail": f"trimmed level flight {scenario.h0_ft:.0f} ft, "
                                                     f"{scenario.speed_kts:.0f} KCAS"}]
    prev = None
    for ts, hs in scenario.steps:
        if prev is not None and hs != prev:
            events.append({"t": float(ts), "type": "target_change",
                           "detail": f"altitude target {prev:.0f} -> {hs:.0f} ft ({hs * FT:.1f} m)"})
        prev = hs
    if getattr(scenario, "discrete_gust_fps", 0.0):
        events.append({"t": float(scenario.discrete_gust_t_s), "type": "gust",
                       "detail": f"1-cosine vertical gust {scenario.discrete_gust_fps:+.1f} ft/s (+down) over "
                                 f"{scenario.discrete_gust_len_s:.1f} s"})
    st = scenario_result["status"]
    events.append({"t": float(scenario_result["t_end"]), "type": "end" if st == "ok" else "terminated",
                   "detail": "completed" if st == "ok" else st})
    events.sort(key=lambda ev: ev["t"])

    gains = genome.decode(genome_vec)
    hz = 1.0 / (rec.dt * rec.every)
    hz = int(round(hz)) if abs(hz - round(hz)) < 1e-9 else hz
    out = {
        "schema": SCHEMA,
        "run_id": run_id,
        "aircraft": aircraft,
        "jsbsim_version": jsbsim_version(),
        "git_sha": git_sha(os.path.dirname(os.path.abspath(sim.__file__))) or "unknown",
        "seed": int(seed if seed is not None else scenario.seed),
        "generation": int(generation if generation is not None else 0),
        "fitness": float(fitness),
        "fitness_sense": "minimize (GA cost, mean over scenarios)",
        "scenario_index": int(scenario_index),
        "scenario_cost": float(scenario_result["cost"]),
        "status": st,
        "genome": {k: float(v) for k, v in gains.items()},
        "genome_normalized": [float(v) for v in genome_vec],
        "gain_units": {g.name: g.units for g in genome.SCHEMA},
        "frame": {
            "origin_lat_deg": lat0, "origin_lon_deg": lon0, "origin_alt_m": alt0,
            "axes": "ENU metres, x=east y=north z=up",
            "attitude": "quat body->ENU [w,x,y,z]",
            "body_axes": "JSBSim body FRD: x forward, y right wing, z down",
            "attitude_source_of_truth": "quaternion; phi/theta/psi are HUD-only (JSBSim Euler ZYX vs local NED)",
            "projection": "x/y: local equirectangular, WGS84 radii of curvature at the origin evaluated at "
                          "origin altitude; z = alt_msl_m - origin_alt_m (no curvature drop)",
        },
        "units": dict(UNITS),
        "channel_units": {c: CHANNEL_UNITS.get(c, "") for c in channels},
        "dt_s": float(rec.dt * rec.every),
        "sim_dt_s": float(rec.dt),
        "sample_hz": hz,
        "target": {
            "alt_m": steps[0][1],
            "steps": [{"t": ts, "alt_m": hm} for ts, hm in steps],
            "speed_kcas": float(scenario.speed_kts),
        },
        "wind": {"north_mps": scenario.wind_north_fps * FT, "east_mps": scenario.wind_east_fps * FT,
                 "gust_sigma_mps": scenario.gust_sigma_fps * FT},
        "events": events,
        "scenario": {"index": scenario_index, **{k: v for k, v in scenario.__dict__.items()},
                     "cost": scenario_result["cost"], "status": st, "t_end": scenario_result["t_end"]},
        "per_scenario": [{k: r[k] for k in ("cost", "status", "t_end", "track", "effort")} for r in per_scenario],
        "control_props": {ch: rec._resolved.get(ch) for ch in CONTROL_PROPS},
        "control_conventions": "JSBSim fcs/*-cmd-norm: elevator + = trailing-edge down (nose down); "
                               "aileron + = right roll; rudder + = per model; throttle 0..1",
        "sign_conventions": "phi>0 right wing down, theta>0 nose up, psi true heading clockwise from north",
        "channels": channels,
        "data": data,
    }
    if extra:
        out.update(extra)
    return out


def write_json(obj, path: str, gz: bool = False) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    if gz:
        path = path if path.endswith(".gz") else path + ".gz"
        with gzip.GzipFile(path, "wb", mtime=0) as f:  # mtime=0 -> byte-reproducible
            f.write(text.encode())
    else:
        with open(path, "w") as f:
            f.write(text)
    return path


def traj_filename(aircraft: str, run_id: str, gen: int, gz: bool = False) -> str:
    return f"traj_{aircraft}_{run_id}_g{int(gen)}.json" + (".gz" if gz else "")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flight-sim-dir", default=None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--best-gains", help="best_gains.json written by evolve.py (uses its genome, scenarios, config)")
    g.add_argument("--genome", help="comma-separated normalized genome (6 floats in [0,1])")
    p.add_argument("--scenarios", type=int, default=3, help="(with --genome) number of GA scenarios")
    p.add_argument("--scenario-seed", type=int, default=1, help="(with --genome) scenario seed")
    p.add_argument("--record-scenario", type=int, default=0, help="which scenario to log (0 = calm air)")
    p.add_argument("--aircraft", default=None, help="JSBSim model name (default: sim.AIRCRAFT)")
    p.add_argument("--speed-kts", type=float, default=None,
                   help="override the scenarios' trim airspeed (other aircraft may not trim at the c172x's 100 KCAS)")
    p.add_argument("--run-id", default="adhoc")
    p.add_argument("--generation", type=int, default=None)
    p.add_argument("--sample-hz", type=float, default=30.0)
    p.add_argument("--gzip", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    mods = import_flight_sim(a.flight_sim_dir)
    sim = mods["sim"]
    seed = None
    if a.best_gains:
        bg = json.load(open(a.best_gains))
        vec = bg["genome"]
        scenarios = [sim.Scenario(**{**s, "steps": [tuple(x) for x in s["steps"]]}) for s in bg["scenarios"]]
        ga_cost = bg.get("best_cost")
        seed = bg.get("config", {}).get("seed")
    else:
        vec = [float(x) for x in a.genome.split(",")]
        scenarios = sim.make_scenarios(a.scenarios, a.scenario_seed)
        ga_cost = None
    if a.speed_kts is not None:
        for sc in scenarios:
            sc.speed_kts = a.speed_kts
        ga_cost = None  # different task -> GA cost no longer comparable
    aircraft = a.aircraft or sim.AIRCRAFT
    fitness, results, rec = fly(mods, mods["genome"].decode(vec), scenarios, a.record_scenario, a.sample_hz,
                                aircraft)
    traj = build_trajectory(mods, rec, aircraft=aircraft, run_id=a.run_id, generation=a.generation,
                            fitness=fitness, genome_vec=vec, scenario=scenarios[a.record_scenario],
                            scenario_index=a.record_scenario, scenario_result=results[a.record_scenario],
                            per_scenario=results, seed=seed)
    path = write_json(traj, a.out, a.gzip)
    msg = f"wrote {path}: {len(traj['data'])} samples @ {traj['sample_hz']:.1f} Hz, fitness {fitness:.6f}"
    if ga_cost is not None:
        msg += f" (GA reported {ga_cost:.6f}, diff {fitness - ga_cost:.3g})"
    print(msg)


if __name__ == "__main__":
    main()
