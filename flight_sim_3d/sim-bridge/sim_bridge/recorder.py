"""Read-only per-step recorder producing ga-flightsim-traj/1 rows from a JSBSim FDM.

Usable as the ``recorder`` callback of ``evolution.eval.evaluate`` (``recorder(t, fdm)`` or, with flex active,
``recorder(t, fdm, flex_state)``) and of sim_bridge.er_adapter.evaluate. It only *reads* properties.
Channel set, frame and formulas are identical to ER's evolution/sim.py ``_Recorder`` (so files compare 1:1);
structure channels ``<component>.<dof>.<node>`` are appended when a flex_state provides them.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

FT = 0.3048
_WGS_A = 6378137.0
_WGS_E2 = 6.69437999014e-3
_Q_ENU_NED = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)
BASE_CHANNELS = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m",
                 "phi", "theta", "psi", "throttle", "elevator", "aileron", "rudder",
                 "target_alt_m", "kcas", "nz", "ub", "vb", "wb", "lat_deg", "lon_deg",
                 "target_cmd_alt_m", "target_rate_mps"]
# same write precision as ER's trajectory.py (mm positions, ~1e-7 quaternion)
ROUND = {"t": 4, "x": 3, "y": 3, "z": 3, "qw": 8, "qx": 8, "qy": 8, "qz": 8, "vx": 4, "vy": 4, "vz": 4,
         "alt_msl_m": 3, "phi": 7, "theta": 7, "psi": 7, "throttle": 6, "elevator": 6, "aileron": 6,
         "rudder": 6, "target_alt_m": 3, "kcas": 3, "nz": 4, "ub": 4, "vb": 4, "wb": 4, "lat_deg": 9, "lon_deg": 9,
         "target_cmd_alt_m": 3, "target_rate_mps": 5}
STRUCT_DECIMALS = 6


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def quat_body_to_enu(phi, theta, psi):
    cr, sr = math.cos(phi / 2), math.sin(phi / 2)
    cp, sp = math.cos(theta / 2), math.sin(theta / 2)
    cy, sy = math.cos(psi / 2), math.sin(psi / 2)
    q_nb = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
    return _qmul(_Q_ENU_NED, q_nb)


def flex_channels(flex_state) -> Dict[str, float]:
    """Accepts a dict {'wingL.dz.0': v, ...}, or an object with .channels() / .as_channels() returning one."""
    if flex_state is None:
        return {}
    for m in ("channels", "as_channels", "to_channels"):
        f = getattr(flex_state, m, None)
        if callable(f):
            return dict(f())
    if isinstance(flex_state, dict):
        return {k: v for k, v in flex_state.items() if isinstance(k, str) and k.count(".") == 2}
    return {}


CTRL_IDX = (15, 16, 17, 18)  # throttle, elevator, aileron, rudder
CTRL_PROPS = ("fcs/throttle-cmd-norm", "fcs/elevator-cmd-norm", "fcs/aileron-cmd-norm", "fcs/rudder-cmd-norm")


class TrajRecorder:
    """recorder(t, fdm[, flex_state]); decimates the sim rate to ``sample_hz`` on the absolute step grid.

    timing="pre":  called before each fdm.run() with the commands for [t, t+dt) already set (sim_bridge adapter;
                   identical to ER's in-sim _Recorder), plus final(t_end, fdm).
    timing="post": called after each step (ER's agreed evolution.eval contract). The state read at t is exact; the
                   fcs commands read then are the ones flown over [t-dt, t), so a sampled row's controls are
                   filled from the *next* call (commands for [t, t+dt)), which reproduces ER's convention exactly.
                   A t=0 row exists only if evaluate also calls recorder(0.0, fdm) once before the first step.
                   finish() adds the final off-grid row (t_end) if needed.
    """

    def __init__(self, scenario, sample_hz: float = 30.0, sim_dt: float = 1.0 / 120.0, timing: str = "pre"):
        if timing not in ("pre", "post"):
            raise ValueError(timing)
        self.timing = timing
        self._pending = None  # (k, row) awaiting its commands (post timing)
        self._last_call = None  # (t, k, fdm, flex_state) for finish()
        self.sc = scenario  # ER Scenario (target / target_cmd / target_rate / ramp_fpm)
        self.sim_dt = sim_dt
        self.every = max(1, int(round(1.0 / (sim_dt * sample_hz))))
        self.sample_hz = sample_hz
        self.rows: List[List[float]] = []
        self.struct_names: List[str] = []
        self.structure: Optional[Dict] = None
        self.origin = None
        self.origin_source = None
        self.q_prev = None
        self.n_calls = 0
        self.t_last = None

    def _init_origin(self, fdm, t=0.0):
        if t > 1e-9:  # no t=0 call (post-step contract without the initial call): JSBSim's IC = trimmed start position
            lat0, lon0, alt0 = fdm["ic/lat-geod-deg"], fdm["ic/long-gc-deg"], fdm["ic/h-sl-ft"] * FT
            self.origin_source = "ic/* properties (no t=0 recorder call)"
        else:
            lat0, lon0, alt0 = fdm["position/lat-geod-deg"], fdm["position/long-gc-deg"], fdm["position/h-sl-ft"] * FT
            self.origin_source = "state at t=0"
        lat = math.radians(lat0)
        s2 = math.sin(lat) ** 2
        self.origin = {"lat_deg": lat0, "lon_deg": lon0, "alt_m": alt0,
                       "r_north": _WGS_A * (1 - _WGS_E2) / (1 - _WGS_E2 * s2) ** 1.5 + alt0,
                       "r_east": (_WGS_A / math.sqrt(1 - _WGS_E2 * s2) + alt0) * math.cos(lat)}

    def __call__(self, t, fdm, flex_state=None):
        self.n_calls += 1
        if self.origin is None:
            self._init_origin(fdm, t)
        k = int(round(t / self.sim_dt))
        if self.timing == "post":
            if self._pending is not None:
                self._fill_ctrls(self._pending, fdm)
                self._pending = None
            self._last_call = (t, k, fdm, flex_state)
        if k % self.every:
            return
        self._add(t, fdm, flex_state)
        if self.timing == "post":
            self._pending = self.rows[-1]

    def _fill_ctrls(self, row, fdm):
        for i, p in zip(CTRL_IDX, CTRL_PROPS):
            row[i] = fdm[p]

    def final(self, t, fdm, flex_state=None):
        if self.t_last is None or t > self.t_last + 1e-9:
            self._add(t, fdm, flex_state)

    def finish(self):
        """post timing: close the run (last row keeps the last commands, as ER's recorder does)."""
        self._pending = None
        if self.timing == "post" and self._last_call is not None:
            t, k, fdm, fx = self._last_call
            self.final(t, fdm, fx)
        self._last_call = None

    def _add(self, t, fdm, flex_state):
        if self.origin is None:
            self._init_origin(fdm)
        o = self.origin
        phi, theta, psi = fdm["attitude/phi-rad"], fdm["attitude/theta-rad"], fdm["attitude/psi-rad"]
        q = quat_body_to_enu(phi, theta, psi)
        if self.q_prev is not None and sum(a * b for a, b in zip(q, self.q_prev)) < 0:
            q = tuple(-c for c in q)
        self.q_prev = q
        lat, lon = fdm["position/lat-geod-deg"], fdm["position/long-gc-deg"]
        alt_m = fdm["position/h-sl-ft"] * FT
        sc = self.sc
        ramp = getattr(sc, "ramp_fpm", None) is not None
        row = [t, math.radians(lon - o["lon_deg"]) * o["r_east"], math.radians(lat - o["lat_deg"]) * o["r_north"], alt_m - o["alt_m"],
               *q,
               fdm["velocities/v-east-fps"] * FT, fdm["velocities/v-north-fps"] * FT, -fdm["velocities/v-down-fps"] * FT,
               alt_m, phi, theta, psi,
               fdm["fcs/throttle-cmd-norm"], fdm["fcs/elevator-cmd-norm"], fdm["fcs/aileron-cmd-norm"], fdm["fcs/rudder-cmd-norm"],
               sc.target(t)[0] * FT, fdm["velocities/vc-kts"], fdm["accelerations/Nz"],
               fdm["velocities/u-fps"] * FT, fdm["velocities/v-fps"] * FT, fdm["velocities/w-fps"] * FT, lat, lon,
               sc.target_cmd(t)[0] * FT, (sc.target_rate(t) if ramp else 0.0) * FT]
        fx = flex_channels(flex_state)
        if fx:
            if not self.struct_names:
                self.struct_names = sorted(fx)
                st = getattr(flex_state, "structure", None) or (flex_state.get("structure") if isinstance(flex_state, dict) else None)
                self.structure = st
            row += [float(fx.get(n, float("nan"))) for n in self.struct_names]
        elif self.struct_names:
            row += [float("nan")] * len(self.struct_names)
        self.rows.append(row)
        self.t_last = t

    @property
    def channels(self) -> List[str]:
        return BASE_CHANNELS + self.struct_names

    def rounded_rows(self) -> List[List[float]]:
        dec = [ROUND.get(c, STRUCT_DECIMALS) for c in self.channels]
        return [[round(float(v), d) if math.isfinite(v) else None for v, d in zip(r, dec)] for r in self.rows]
