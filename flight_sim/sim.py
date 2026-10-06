"""Headless JSBSim evaluation of an altitude-hold controller.

``evaluate(gains, scenario)`` flies one scenario and returns a cost (lower is
better). This plays the role of the car GA's ``carRun`` contract
(getInitialState / updateState / getStatus / calculateScore): each step updates
the state, checks termination ("health" in the car GA, here the flight
envelope), and a score is computed at the end.

Cost = tracking + W_EFFORT * effort, where tracking is a time-weighted mean
absolute altitude error (ITAE-like, see ITAE_* below) and effort is the
elevator's total variation per second. Envelope violations (crash, stall,
attitude, load factor, divergence) end the run with cost >= FAIL_BASE.

Controller (gains are evolved, everything else is fixed):

    theta_cmd = theta_trim + clip(kp_alt*e_h + ki_alt*int(e_h) - kd_alt*h_dot)
    elevator  = elev_trim  - clip(kp_pitch*e_theta + ki_pitch*int(e_theta) - kd_pitch*q)

JSBSim's elevator command is positive trailing-edge-down (nose down), hence the
minus sign. Throttle is a fixed-gain PI airspeed hold and the ailerons a fixed
wing leveler, so the GA only tunes the longitudinal altitude/pitch loops.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

AIRCRAFT = "c172x"
DT = 1.0 / 120.0

# Envelope limits; violating any ends the run with a large penalty.
MIN_AGL_FT = 500.0
MIN_KCAS = 55.0          # C172 clean stall is ~48 KCAS
MAX_ABS_THETA_DEG = 30.0
MAX_ABS_PHI_DEG = 45.0
NZ_LIMITS = (-1.0, 3.8)  # normal-category load factor limits
MAX_ALT_ERR_FT = 1000.0

# Controller saturation limits.
PITCH_CMD_LIMITS_DEG = (-8.0, 12.0)   # relative to trim pitch
ALT_I_LIMIT_DEG = 5.0                  # max |ki_alt * integral|
PITCH_I_LIMIT = 0.4                    # max |ki_pitch * integral| (normalized elevator)

# Cost weights. Tracking uses a capped ITAE-style weight: altitude error counts
# little right after a target step (when some error is unavoidable) and up to
# ITAE_CAP_S / ITAE_T0_S times more once the aircraft should have settled.
ALT_ERR_SCALE_FT = 100.0
ITAE_T0_S = 10.0
ITAE_CAP_S = 30.0
W_EFFORT = 2.0           # weight on elevator activity: total variation sum|d elev| per second of flight
FAIL_BASE = 1000.0       # cost of an envelope violation (plus up to FAIL_BASE more for failing early)


@dataclass
class Scenario:
    seed: int
    duration_s: float = 90.0
    h0_ft: float = 4000.0
    speed_kts: float = 100.0
    # (time_s, altitude_target_ft) breakpoints; the target is piecewise constant.
    steps: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 4000.0), (5.0, 4200.0), (50.0, 4000.0)])
    wind_north_fps: float = 0.0
    wind_east_fps: float = 0.0
    gust_sigma_fps: float = 0.0   # RMS of vertical gust (first-order Gauss-Markov)
    gust_tau_s: float = 2.0
    discrete_gust_fps: float = 0.0  # amplitude of one 1-cosine vertical gust
    discrete_gust_t_s: float = 30.0
    discrete_gust_len_s: float = 3.0

    def target(self, t: float) -> Tuple[float, float]:
        """(target altitude, time of the most recent target change) at time t."""
        h, t_step = self.steps[0][1], self.steps[0][0]
        for ts, hs in self.steps:
            if t >= ts:
                if hs != h:
                    t_step = ts
                h = hs
        return h, t_step

    def vertical_gust_series(self, n: int) -> np.ndarray:
        """Deterministic vertical wind (ft/s, positive down) for n steps."""
        rng = np.random.default_rng(self.seed)
        w = np.zeros(n)
        if self.gust_sigma_fps > 0:
            a = math.exp(-DT / self.gust_tau_s)
            b = self.gust_sigma_fps * math.sqrt(1 - a * a)
            noise = rng.standard_normal(n)
            for k in range(1, n):
                w[k] = a * w[k - 1] + b * noise[k]
        if self.discrete_gust_fps:
            t = np.arange(n) * DT
            m = (t >= self.discrete_gust_t_s) & (t < self.discrete_gust_t_s + self.discrete_gust_len_s)
            w[m] += 0.5 * self.discrete_gust_fps * (1 - np.cos(2 * np.pi * (t[m] - self.discrete_gust_t_s) / self.discrete_gust_len_s))
        return w


def make_scenarios(n: int, seed: int) -> List[Scenario]:
    """Scenario 0 is calm air; the rest get seeded random wind and gusts.

    This is the analog of the car GA's seeded track (``floorseed``), but with
    several cases averaged so the controller doesn't overfit one of them.
    """
    rng = np.random.default_rng(seed)
    out = [Scenario(seed=int(rng.integers(2**31)))]
    for _ in range(n - 1):
        s = Scenario(seed=int(rng.integers(2**31)))
        spd = rng.uniform(5, 25)
        hdg = rng.uniform(0, 2 * np.pi)
        s.wind_north_fps = float(spd * np.cos(hdg))
        s.wind_east_fps = float(spd * np.sin(hdg))
        s.gust_sigma_fps = float(rng.uniform(1.0, 4.0))
        s.discrete_gust_fps = float(rng.choice([-1, 1]) * rng.uniform(8, 15))
        s.discrete_gust_t_s = float(rng.uniform(25, 45))
        out.append(s)
    return out


_JSBSIM_OUT_DIR = os.path.join(tempfile.gettempdir(), "flight_sim_jsbsim")


def _new_fdm():
    os.makedirs(_JSBSIM_OUT_DIR, exist_ok=True)
    import jsbsim

    jsbsim.FGJSBBase().debug_lvl = 0  # silence the startup banner
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    # c172x declares a CSV <output>; keep its (header-only) file out of the cwd.
    fdm.set_output_path(_JSBSIM_OUT_DIR)
    fdm.load_model(AIRCRAFT)
    fdm.disable_output()
    fdm.set_dt(DT)
    return fdm


def simulate(gains: Dict[str, float], sc: Scenario, record: bool = False) -> Dict:
    """Fly one scenario. Returns cost, status and (optionally) time histories."""
    fdm = _new_fdm()
    fdm["ic/h-sl-ft"] = sc.h0_ft
    fdm["ic/vc-kts"] = sc.speed_kts
    fdm["ic/gamma-deg"] = 0.0
    fdm["ic/psi-true-deg"] = 0.0
    fdm.run_ic()
    fdm["propulsion/set-running"] = -1
    fdm["fcs/mixture-cmd-norm"] = 1.0
    fdm["simulation/do_simple_trim"] = 1  # longitudinal trim in calm air

    theta_trim = fdm["attitude/theta-deg"]
    elev_trim = fdm["fcs/elevator-cmd-norm"]
    thr_trim = fdm["fcs/throttle-cmd-norm"]
    v_target = fdm["velocities/vc-kts"]

    # Wind is switched on after trimming, so it acts as a disturbance.
    fdm["atmosphere/wind-north-fps"] = sc.wind_north_fps
    fdm["atmosphere/wind-east-fps"] = sc.wind_east_fps

    n = int(round(sc.duration_s / DT))
    w_down = sc.vertical_gust_series(n)

    kp_a, ki_a, kd_a = gains["kp_alt"], gains["ki_alt"], gains["kd_alt"]
    kp_p, ki_p, kd_p = gains["kp_pitch"], gains["ki_pitch"], gains["kd_pitch"]
    lo_cmd, hi_cmd = PITCH_CMD_LIMITS_DEG

    i_alt = 0.0
    i_pitch = 0.0
    i_spd = 0.0
    elev_prev = elev_trim
    sum_err = 0.0
    sum_tv = 0.0
    status = "ok"
    k_end = n
    rec = {k: [] for k in ("t", "h", "target", "theta", "theta_cmd", "elevator", "vc", "throttle", "w_down")} if record else None

    for k in range(n):
        t = k * DT
        fdm["atmosphere/wind-down-fps"] = float(w_down[k])

        h = fdm["position/h-sl-ft"]
        h_agl = fdm["position/h-agl-ft"]
        h_dot = fdm["velocities/h-dot-fps"]
        theta = fdm["attitude/theta-deg"]
        phi = fdm["attitude/phi-deg"]
        q = math.degrees(fdm["velocities/q-rad_sec"])
        p = math.degrees(fdm["velocities/p-rad_sec"])
        vc = fdm["velocities/vc-kts"]
        nz = fdm["accelerations/Nz"]
        target, t_step = sc.target(t)
        e_h = target - h

        # --- envelope / termination checks (the car GA's "health") ---
        if not all(map(math.isfinite, (h, theta, vc, nz))):
            status = "diverged"
        elif h_agl < MIN_AGL_FT:
            status = "crash"
        elif vc < MIN_KCAS:
            status = "stall"
        elif abs(theta) > MAX_ABS_THETA_DEG or abs(phi) > MAX_ABS_PHI_DEG:
            status = "attitude"
        elif not (NZ_LIMITS[0] <= nz <= NZ_LIMITS[1]):
            status = "overload"
        elif abs(e_h) > MAX_ALT_ERR_FT:
            status = "diverged"
        if status != "ok":
            k_end = k
            break

        # --- outer loop: altitude -> pitch command ---
        if ki_a > 0:
            lim = ALT_I_LIMIT_DEG / ki_a
            i_alt = min(max(i_alt + e_h * DT, -lim), lim)
        u_alt = kp_a * e_h + ki_a * i_alt - kd_a * h_dot
        theta_cmd = theta_trim + min(max(u_alt, lo_cmd), hi_cmd)

        # --- inner loop: pitch -> elevator ---
        e_th = theta_cmd - theta
        if ki_p > 0:
            lim = PITCH_I_LIMIT / ki_p
            i_pitch = min(max(i_pitch + e_th * DT, -lim), lim)
        u_p = kp_p * e_th + ki_p * i_pitch - kd_p * q
        elev = min(max(elev_trim - u_p, -1.0), 1.0)

        # --- fixed helpers: airspeed hold on throttle, wing leveler ---
        e_v = v_target - vc
        i_spd = min(max(i_spd + e_v * DT, -50.0), 50.0)
        thr = min(max(thr_trim + 0.05 * e_v + 0.01 * i_spd, 0.0), 1.0)
        ail = min(max(-0.05 * phi - 0.02 * p, -0.5), 0.5)

        fdm["fcs/elevator-cmd-norm"] = elev
        fdm["fcs/throttle-cmd-norm"] = thr
        fdm["fcs/aileron-cmd-norm"] = ail
        fdm["fcs/rudder-cmd-norm"] = 0.0

        sum_tv += abs(elev - elev_prev)
        elev_prev = elev
        sum_err += abs(e_h) / ALT_ERR_SCALE_FT * min(t - t_step, ITAE_CAP_S) / ITAE_T0_S

        if record:
            for key, val in (("t", t), ("h", h), ("target", target), ("theta", theta), ("theta_cmd", theta_cmd),
                             ("elevator", elev), ("vc", vc), ("throttle", thr), ("w_down", w_down[k])):
                rec[key].append(val)

        fdm.run()

    if status == "ok":
        track = sum_err / n
        effort = sum_tv / sc.duration_s
        cost = track + W_EFFORT * effort
    else:
        track = effort = float("nan")
        cost = FAIL_BASE + FAIL_BASE * (1.0 - k_end / n)  # failing earlier is worse

    out = {"cost": float(cost), "status": status, "t_end": k_end * DT, "track": float(track), "effort": float(effort)}
    if record:
        out["trace"] = {k: np.asarray(v) for k, v in rec.items()}
    return out


def evaluate(gains: Dict[str, float], scenarios: List[Scenario]) -> Dict:
    """Mean cost over scenarios (the GA's fitness; lower is better)."""
    results = [simulate(gains, sc) for sc in scenarios]
    return {
        "cost": float(np.mean([r["cost"] for r in results])),
        "per_scenario": [{k: r[k] for k in ("cost", "status", "t_end", "track", "effort")} for r in results],
    }
