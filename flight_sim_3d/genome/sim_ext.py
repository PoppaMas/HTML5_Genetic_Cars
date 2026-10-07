"""Telemetry-rich JSBSim backend: a faithful extension of flight_sim/sim.py.

``simulate`` reproduces ``sim.simulate`` operation-for-operation (same
controller, same envelope checks, same legacy cost terms -- verified
bit-identical in tests/test_legacy_equivalence.py) and additionally:

* records the channels the multi-objective fitness needs (nz, q, p, r, phi,
  psi, beta, throttle, aileron, vc, ...) as numpy arrays;
* consumes roll/heading and speed/throttle block genes *if present* in the
  gains dict (otherwise the original fixed wing leveler / speed hold run);
* applies ExtScenario perturbations: payload mass / CG shift via JSBSim point
  masses, controller-side sensor noise, heading-target steps;
* applies ExtScenario *task conditions*: a rate-limited (ramped) altitude
  reference, optional reference-rate feed-forward in the altitude D term,
  a per-aircraft pitch-command clamp, per-aircraft envelope limits, and other
  JSBSim models (multi-engine throttles, trim failure reported as a status).
  All default to sim.py's behaviour, so the legacy path is unchanged.
* for ExtScenario only (never the legacy sim.Scenario path): loads Flight
  Dynamics' patched aircraft copies (flight-dynamics/jsbsim_root), retracts the
  gear before run_ic (every stock model loads gear-down), clamps throttle per
  aircraft (T38: 0.5 = MIL), and optionally runs FD's two-way flex-wing
  coupling (coupled_sim.FlexFDM) with the structure genes.
  Trim is always do_simple_trim = 1 (full trim), as in sim.py; mode 0 is never used.

Not consumed (no dynamics here): yaw_damper and structure genes. The adapter
refuses to evolve inert blocks unless explicitly allowed. Everything here is
real JSBSim dynamics -- nothing is synthesized.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import fd_bridge
from flightsim_path import orig_sim

S = orig_sim()  # the unmodified flight_sim/sim.py

CAPABILITIES = {
    "name": "jsbsim_ext",
    "aircraft_verified": ["c172x", "T38", "737"],  # closed-loop sanity-checked (sanity_check.py, tests)
    "aircraft_supported": ["c172x", "T38", "737"],
    "consumes": ["kp_alt", "ki_alt", "kd_alt", "kp_pitch", "ki_pitch", "kd_pitch",
                 "kp_hdg", "ki_hdg", "kp_roll", "ki_roll", "kd_roll",
                 "kp_spd", "ki_spd", "kd_spd"],
    "channels": ["t", "h", "target", "target_cmd", "target_rate", "theta", "theta_cmd", "elevator", "vc", "v_target", "throttle", "aileron",
                 "rudder", "nz", "q", "p", "r", "phi", "phi_cmd", "psi", "psi_target", "beta", "h_dot", "w_down"],
    # Channels a flex model would add (wing_root_bending, tip_deflection, ...) are absent here.
}

BANK_CMD_LIMIT_DEG = 25.0   # default bank-command clamp; ExtScenario.bank_cmd_limit_deg overrides it per aircraft
HDG_I_LIMIT_DEG = 10.0      # max bank (deg) the heading integrator may contribute; ExtScenario.hdg_i_limit_deg overrides
ROLL_I_LIMIT = 0.2


@dataclass
class ExtScenario(S.Scenario):
    label: str = ""
    payload_delta_lb: float = 0.0          # added to point mass `payload_index`
    payload_index: int = 0
    cg_shift_in: float = 0.0               # moves that point mass in body X (inches, + aft in JSBSim structural frame)
    noise_alt_ft: float = 0.0              # controller-side white sensor noise (1 sigma, per sample)
    noise_hdot_fps: float = 0.0
    noise_theta_deg: float = 0.0
    noise_q_dps: float = 0.0
    noise_seed: int = 0
    heading_steps: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0)])
    # --- task conditions (None/False = sim.py behaviour) ---
    aircraft: str = "c172x"                                # JSBSim model name
    ramp_fpm: Optional[float] = None                       # altitude reference rate limit; None = instant step
    alt_ref_ff: bool = False                               # D term on (h_dot - h_ref_dot) instead of h_dot
    ramp_accel_g: Optional[float] = None                   # vertical-accel limit on the ramp reference (g); None = sharp corners
    alt_err_scale_ft: Optional[float] = None               # tracking normaliser; None = sim.ALT_ERR_SCALE_FT (100)
    max_alt_err_ft: Optional[float] = None                 # envelope; None = sim.MAX_ALT_ERR_FT (1000)
    pitch_cmd_limits_deg: Optional[Tuple[float, float]] = None  # relative to trim; None = sim.PITCH_CMD_LIMITS_DEG
    min_kcas: Optional[float] = None                       # envelope; None = sim.MIN_KCAS
    nz_limits: Optional[Tuple[float, float]] = None        # envelope; None = sim.NZ_LIMITS
    throttle_max: Optional[float] = None                   # throttle-cmd clamp; None = 1.0 (sim.py)
    use_fd_root: bool = True                               # FD patched aircraft copies + gear up
    flex_mode: Optional[str] = None                        # None (rigid) | "twoway" | "oneway" (FD coupled_sim, v1) | "v2" (FD flexbody)
    flex_substeps: int = 2
    flex_asymmetric: bool = False                          # v2 only: FD's optional left/right asymmetry genes
    bank_cmd_limit_deg: Optional[float] = None             # heading loop bank clamp (deg); None = BANK_CMD_LIMIT_DEG
    hdg_i_limit_deg: Optional[float] = None                # heading integrator authority (deg bank); None = HDG_I_LIMIT_DEG
    # sustained vertical wind (v5 disturbance scenario): + = downdraft (JSBSim wind-down), 1-cos onset
    draft_fps: float = 0.0
    draft_t_s: float = 10.0
    draft_ramp_s: float = 4.0

    def vertical_gust_series(self, n: int) -> np.ndarray:
        w = S.Scenario.vertical_gust_series(self, n)
        if self.draft_fps:
            t = np.arange(n) * S.DT
            x = np.clip((t - self.draft_t_s) / self.draft_ramp_s, 0.0, 1.0)
            w = w + self.draft_fps * 0.5 * (1.0 - np.cos(np.pi * x))
        return w

    def _ramp(self, t: float) -> Tuple[float, float, float, float]:
        """(reference, time of last command change, reference rate ft/s, commanded value)."""
        cmd, t_step = self.steps[0][1], self.steps[0][0]
        ref0, t0 = cmd, self.steps[0][0]
        rate = self.ramp_fpm / 60.0
        for ts, hs in self.steps:
            if t >= ts and hs != cmd:
                # reference position when the new command arrives
                d = cmd - ref0
                ref0 = ref0 + math.copysign(min(abs(d), rate * (ts - t0)), d) if d else ref0
                t0, cmd, t_step = ts, hs, ts
        d = cmd - ref0
        moved = rate * (t - t0)
        if abs(d) <= moved:
            return cmd, t_step, 0.0, cmd
        return ref0 + math.copysign(moved, d), t_step, math.copysign(rate, d), cmd

    def _smooth_plan(self):
        """Accel-limited reference (trapezoid in climb rate), as exact constant-acceleration phases.

        The reference (p, v) moves toward the commanded altitude with |v| <= ramp rate and |dv/dt| <= ramp_accel_g*g:
        accelerate, cruise at the ramp rate (if the move is long enough), decelerate to rest exactly on the target.
        A command change mid-move continues from the current (p, v). Returns [(t0, p0, v0, accel), ...].
        """
        plan = self.__dict__.get("_smooth_plan_cache")
        if plan is not None:
            return plan
        a, vmax = self.ramp_accel_g * 32.174, self.ramp_fpm / 60.0

        def moves(t0, p0, v0, c):
            out = []
            while True:
                d = c - p0
                if abs(d) < 1e-9 and abs(v0) < 1e-9:
                    out.append((t0, c, 0.0, 0.0))
                    return out
                sg = math.copysign(1.0, d) if abs(d) >= 1e-9 else -math.copysign(1.0, v0)
                u0, D = sg * v0, abs(d)
                if u0 < 0 or u0 * u0 / (2 * a) > D + 1e-9:  # moving away / can't stop in time: brake to rest first
                    tb = abs(v0) / a
                    out.append((t0, p0, v0, -math.copysign(a, v0)))
                    p0, v0, t0 = p0 + v0 * tb / 2, 0.0, t0 + tb
                    continue
                if (2 * vmax * vmax - u0 * u0) / (2 * a) <= D:  # trapezoid
                    up = vmax
                    t_acc = (vmax - u0) / a
                    t_cru = (D - (vmax * vmax - u0 * u0) / (2 * a) - vmax * vmax / (2 * a)) / vmax
                else:  # triangle
                    up = math.sqrt(a * D + u0 * u0 / 2)
                    t_acc, t_cru = (up - u0) / a, 0.0
                t_dec = up / a
                for dur, acc in ((t_acc, sg * a), (t_cru, 0.0), (t_dec, -sg * a)):
                    if dur > 0:
                        out.append((t0, p0, v0, acc))
                        p0, v0, t0 = p0 + v0 * dur + 0.5 * acc * dur * dur, v0 + acc * dur, t0 + dur
                out.append((t0, c, 0.0, 0.0))
                return out

        plan = [(self.steps[0][0], float(self.steps[0][1]), 0.0, 0.0)]
        cmd = self.steps[0][1]
        for ts, hs in self.steps[1:]:
            if hs == cmd:
                continue
            p0, v0 = self._eval_plan(plan, ts)
            plan = [ph for ph in plan if ph[0] < ts] + moves(ts, p0, v0, float(hs))
            cmd = hs
        self.__dict__["_smooth_plan_cache"] = plan
        return plan

    @staticmethod
    def _eval_plan(plan, t):
        ph = plan[0]
        for q in plan:
            if q[0] <= t:
                ph = q
            else:
                break
        t0, p0, v0, acc = ph
        dt = t - t0
        return p0 + v0 * dt + 0.5 * acc * dt * dt, v0 + acc * dt

    def _smooth(self, t: float) -> Tuple[float, float]:
        return self._eval_plan(self._smooth_plan(), t)

    def target(self, t: float) -> Tuple[float, float]:
        if self.ramp_fpm is None:
            return S.Scenario.target(self, t)
        ref, t_step, _, _ = self._ramp(t)
        if self.ramp_accel_g:
            ref = self._smooth(t)[0]
        return ref, t_step

    def target_rate(self, t: float) -> float:
        if self.ramp_fpm is None:
            return 0.0
        return self._smooth(t)[1] if self.ramp_accel_g else self._ramp(t)[2]

    def target_cmd(self, t: float) -> float:
        return S.Scenario.target(self, t)[0]

    def heading_target(self, t: float) -> float:
        psi = self.heading_steps[0][1]
        for ts, ps in self.heading_steps:
            if t >= ts:
                psi = ps
        return psi


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _new_fdm(model: str, root: Optional[str] = None):
    """sim._new_fdm for the stock c172x (identical call), generalized to other models / FD's jsbsim_root."""
    if model == S.AIRCRAFT and root is None:
        return S._new_fdm()  # the legacy path, unchanged (stock c172x has no network I/O; checked in tests)
    import os
    import jsbsim
    fd_bridge.check_no_network_io(root or jsbsim.get_default_root_dir(), model)  # never load a file with network I/O
    os.makedirs(S._JSBSIM_OUT_DIR, exist_ok=True)
    jsbsim.FGJSBBase().debug_lvl = 0
    fdm = jsbsim.FGFDMExec(root)
    fdm.set_debug_level(0)
    fdm.set_output_path(S._JSBSIM_OUT_DIR)
    if not fdm.load_model(model):
        raise RuntimeError(f"JSBSim could not load model {model!r}")
    fdm.disable_output()
    fdm.set_dt(S.DT)
    return fdm


def _fail(status: str, msg: str, record: bool) -> Dict:
    out = {"cost": float(2 * S.FAIL_BASE), "status": status, "t_end": 0.0, "track": float("nan"),
           "effort": float("nan"), "n_steps": 0, "k_end": 0, "theta_trim": float("nan"), "weight_lb": float("nan"),
           "dt": S.DT, "error": msg}
    if record:
        out["telemetry"] = out["trace"] = {c: np.zeros(0) for c in CAPABILITIES["channels"]}
    return out


def simulate(gains: Dict[str, float], sc, record: bool = True) -> Dict:
    """Fly one scenario (sim.Scenario or ExtScenario). Returns legacy fields + 'telemetry'."""
    ext = isinstance(sc, ExtScenario)
    model = sc.aircraft if ext else S.AIRCRAFT
    coupler = None
    v2 = ext and sc.flex_mode == "v2"
    mdl_v2 = None
    try:
        if v2:  # FD flex v2: own prepared root (jsbsim_root_v2), structural mass of all bodies applied before IC/trim
            root = fd_bridge.require_root_v2(model)
        else:
            root = fd_bridge.require_root(model) if ext and (sc.use_fd_root or sc.flex_mode) else None
        fdm = _new_fdm(model, root)
        if ext and root:
            fdm["gear/gear-cmd-norm"] = 0.0  # before run_ic (FD INTERFACE.md 4a); no effect on the fixed-gear c172x
        if v2:
            mdl_v2 = fd_bridge.model_v2(model, fd_bridge.struct_genes_v2(gains, sc.flex_asymmetric), sc.flex_asymmetric)
            v2_mass = fd_bridge.flexbody().apply_mass_v2(fdm, mdl_v2, root)
        elif ext and sc.flex_mode:
            # builds the wing from the structure genes and pushes its mass change into JSBSim (before trim)
            coupler = fd_bridge.coupled_sim().make_coupler(fdm, model, sc.flex_mode, fd_bridge.flex_overrides(gains, model),
                                                           sc.flex_substeps, root)
    except Exception as e:  # noqa: BLE001 - reported, not hidden
        return _fail("load_failed", f"{type(e).__name__}: {e}", record)
    if ext and (sc.payload_delta_lb or sc.cg_shift_in):
        i = sc.payload_index
        fdm[f"inertia/pointmass-weight-lbs[{i}]"] = fdm[f"inertia/pointmass-weight-lbs[{i}]"] + sc.payload_delta_lb
        fdm[f"inertia/pointmass-location-X-inches[{i}]"] = fdm[f"inertia/pointmass-location-X-inches[{i}]"] + sc.cg_shift_in
    fdm["ic/h-sl-ft"] = sc.h0_ft
    fdm["ic/vc-kts"] = sc.speed_kts
    fdm["ic/gamma-deg"] = 0.0
    fdm["ic/psi-true-deg"] = 0.0
    fdm.run_ic()
    fdm["propulsion/set-running"] = -1
    fdm["fcs/mixture-cmd-norm"] = 1.0
    if model == S.AIRCRAFT:
        fdm["simulation/do_simple_trim"] = 1  # exactly as sim.py
    else:
        try:
            fdm["simulation/do_simple_trim"] = 1
        except Exception as e:  # noqa: BLE001 - JSBSim raises TrimFailureError
            return _fail("trim_failed", f"{type(e).__name__}: {e}", record)
    n_eng = fdm.get_propulsion().get_num_engines()
    if ext:
        for i in range(1, n_eng):  # all engines (no effect on turbines/c172x; matches Evolution Runner)
            fdm[f"fcs/mixture-cmd-norm[{i}]"] = 1.0
        if sc.throttle_max is not None and fdm["fcs/throttle-cmd-norm"] > sc.throttle_max + 1e-9:
            return _fail("trim_failed", f"trim throttle {fdm['fcs/throttle-cmd-norm']:.3f} > throttle_max {sc.throttle_max}", record)
    if coupler is not None:
        fdm = fd_bridge.coupled_sim().FlexFDM(fdm, coupler, S.DT, record=True)  # structure step before each run()
    if mdl_v2 is not None:  # v2 coupler + proxy after trim (FD's FlexHookV2.wrap): run() = structure step, then JSBSim
        fb = fd_bridge.flexbody()
        fdm = fb.FlexBodyFDM(fdm, fb.FlexBodyCoupler(mdl_v2, mode="twoway", substeps=sc.flex_substeps), S.DT, record=True)

    theta_trim = fdm["attitude/theta-deg"]
    elev_trim = fdm["fcs/elevator-cmd-norm"]
    thr_trim = fdm["fcs/throttle-cmd-norm"]
    v_target = fdm["velocities/vc-kts"]
    weight_lb = fdm["inertia/weight-lbs"]

    fdm["atmosphere/wind-north-fps"] = sc.wind_north_fps
    fdm["atmosphere/wind-east-fps"] = sc.wind_east_fps

    n = int(round(sc.duration_s / S.DT))
    DT = S.DT
    w_down = sc.vertical_gust_series(n)

    kp_a, ki_a, kd_a = gains["kp_alt"], gains["ki_alt"], gains["kd_alt"]
    kp_p, ki_p, kd_p = gains["kp_pitch"], gains["ki_pitch"], gains["kd_pitch"]
    lo_cmd, hi_cmd = sc.pitch_cmd_limits_deg if ext and sc.pitch_cmd_limits_deg else S.PITCH_CMD_LIMITS_DEG
    min_kcas = sc.min_kcas if ext and sc.min_kcas is not None else S.MIN_KCAS
    nz_lo, nz_hi = sc.nz_limits if ext and sc.nz_limits else S.NZ_LIMITS
    ramp = ext and sc.ramp_fpm is not None
    ff = ext and sc.alt_ref_ff
    thr_max = sc.throttle_max if ext and sc.throttle_max is not None else None
    err_scale = sc.alt_err_scale_ft if ext and sc.alt_err_scale_ft is not None else S.ALT_ERR_SCALE_FT
    max_alt_err = sc.max_alt_err_ft if ext and sc.max_alt_err_ft is not None else S.MAX_ALT_ERR_FT

    bank_lim = sc.bank_cmd_limit_deg if ext and sc.bank_cmd_limit_deg is not None else BANK_CMD_LIMIT_DEG
    hdg_i_lim = sc.hdg_i_limit_deg if ext and sc.hdg_i_limit_deg is not None else HDG_I_LIMIT_DEG
    use_roll = "kp_roll" in gains
    use_hdg = "kp_hdg" in gains
    use_spd = "kp_spd" in gains
    if use_roll:
        kp_r, ki_r, kd_r = gains["kp_roll"], gains.get("ki_roll", 0.0), gains["kd_roll"]
    if use_hdg:
        kp_h, ki_h = gains["kp_hdg"], gains.get("ki_hdg", 0.0)
    if use_spd:
        kp_s, ki_s, kd_s = gains["kp_spd"], gains.get("ki_spd", 0.0), gains.get("kd_spd", 0.0)

    noisy = ext and any((sc.noise_alt_ft, sc.noise_hdot_fps, sc.noise_theta_deg, sc.noise_q_dps))
    if noisy:
        nrng = np.random.default_rng(sc.noise_seed)
        noise = nrng.standard_normal((n, 4)) * np.array([sc.noise_alt_ft, sc.noise_hdot_fps, sc.noise_theta_deg, sc.noise_q_dps])

    i_alt = 0.0
    i_pitch = 0.0
    i_spd = 0.0
    i_hdg = 0.0
    i_roll = 0.0
    vc_prev = None
    elev_prev = elev_trim
    sum_err = 0.0
    sum_tv = 0.0
    status = "ok"
    k_end = n
    chans = CAPABILITIES["channels"]
    rec = {c: np.full(n, np.nan) for c in chans} if record else None

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
        h_ref_dot = sc.target_rate(t) if ramp else 0.0

        if not all(map(math.isfinite, (h, theta, vc, nz))):
            status = "diverged"
        elif h_agl < S.MIN_AGL_FT:
            status = "crash"
        elif vc < min_kcas:
            status = "stall"
        elif abs(theta) > S.MAX_ABS_THETA_DEG or abs(phi) > S.MAX_ABS_PHI_DEG:
            status = "attitude"
        elif not (nz_lo <= nz <= nz_hi):
            status = "overload"
        elif abs(e_h) > max_alt_err:
            status = "diverged"
        if status != "ok":
            k_end = k
            break

        # Controller inputs (sensor noise only affects what the controller sees).
        if noisy:
            e_h_m = e_h - noise[k, 0]
            h_dot_m = h_dot + noise[k, 1]
            theta_m = theta + noise[k, 2]
            q_m = q + noise[k, 3]
        else:
            e_h_m, h_dot_m, theta_m, q_m = e_h, h_dot, theta, q

        # --- outer loop: altitude -> pitch command (identical to sim.py) ---
        if ki_a > 0:
            lim = S.ALT_I_LIMIT_DEG / ki_a
            i_alt = min(max(i_alt + e_h_m * DT, -lim), lim)
        if ff:
            u_alt = kp_a * e_h_m + ki_a * i_alt - kd_a * (h_dot_m - h_ref_dot)
        else:
            u_alt = kp_a * e_h_m + ki_a * i_alt - kd_a * h_dot_m
        theta_cmd = theta_trim + min(max(u_alt, lo_cmd), hi_cmd)

        # --- inner loop: pitch -> elevator ---
        e_th = theta_cmd - theta_m
        if ki_p > 0:
            lim = S.PITCH_I_LIMIT / ki_p
            i_pitch = min(max(i_pitch + e_th * DT, -lim), lim)
        u_p = kp_p * e_th + ki_p * i_pitch - kd_p * q_m
        elev = min(max(elev_trim - u_p, -1.0), 1.0)

        # --- speed / throttle ---
        e_v = v_target - vc
        i_spd = min(max(i_spd + e_v * DT, -50.0), 50.0)
        if use_spd:
            vdot = 0.0 if vc_prev is None else (vc - vc_prev) / DT
            thr = min(max(thr_trim + kp_s * e_v + ki_s * i_spd - kd_s * vdot, 0.0), 1.0)
        else:
            thr = min(max(thr_trim + 0.05 * e_v + 0.01 * i_spd, 0.0), 1.0)
        if thr_max is not None and thr > thr_max:
            thr = thr_max
        vc_prev = vc

        # --- lateral: heading -> bank cmd -> aileron (or the original wing leveler) ---
        phi_cmd = 0.0
        psi = psi_t = float("nan")
        if use_hdg or record:
            psi = fdm["attitude/psi-deg"]
            psi_t = sc.heading_target(t) if ext else 0.0
        if use_hdg:
            e_psi = _wrap180(psi_t - psi)  # shortest way round: (-180, 180]
            if ki_h > 0:
                lim = hdg_i_lim / ki_h
                i_hdg = min(max(i_hdg + e_psi * DT, -lim), lim)
            phi_cmd = min(max(kp_h * e_psi + ki_h * i_hdg, -bank_lim), bank_lim)
        if use_roll:
            e_phi = phi - phi_cmd
            if ki_r > 0:
                lim = ROLL_I_LIMIT / ki_r
                i_roll = min(max(i_roll + e_phi * DT, -lim), lim)
            ail = min(max(-kp_r * e_phi - ki_r * i_roll - kd_r * p, -0.5), 0.5)
        else:
            ail = min(max(-0.05 * (phi - phi_cmd) - 0.02 * p, -0.5), 0.5)  # phi_cmd = 0 unless heading hold is on

        fdm["fcs/elevator-cmd-norm"] = elev
        fdm["fcs/throttle-cmd-norm"] = thr
        for i_e in range(1, n_eng):  # multi-engine models (no-op for the c172x)
            fdm[f"fcs/throttle-cmd-norm[{i_e}]"] = thr
        fdm["fcs/aileron-cmd-norm"] = ail
        fdm["fcs/rudder-cmd-norm"] = 0.0

        sum_tv += abs(elev - elev_prev)
        elev_prev = elev
        sum_err += abs(e_h) / err_scale * min(t - t_step, S.ITAE_CAP_S) / S.ITAE_T0_S

        if record:
            for c, val in (("t", t), ("h", h), ("target", target),
                           ("target_cmd", sc.target_cmd(t) if ramp else target), ("target_rate", h_ref_dot), ("theta", theta), ("theta_cmd", theta_cmd),
                           ("elevator", elev), ("vc", vc), ("v_target", v_target), ("throttle", thr),
                           ("aileron", ail), ("rudder", 0.0), ("nz", nz), ("q", q), ("p", p),
                           ("r", math.degrees(fdm["velocities/r-rad_sec"])), ("phi", phi), ("phi_cmd", phi_cmd),
                           ("psi", psi), ("psi_target", psi_t), ("beta", fdm["aero/beta-deg"]),
                           ("h_dot", h_dot), ("w_down", w_down[k])):
                rec[c][k] = val

        fdm.run()

    if status == "ok":
        track = sum_err / n
        effort = sum_tv / sc.duration_s
        cost = track + S.W_EFFORT * effort
    else:
        track = effort = float("nan")
        cost = S.FAIL_BASE + S.FAIL_BASE * (1.0 - k_end / n)

    out = {"cost": float(cost), "status": status, "t_end": k_end * DT, "track": float(track), "effort": float(effort),
           "n_steps": n, "k_end": k_end, "theta_trim": theta_trim, "weight_lb": weight_lb, "dt": DT,
           "aircraft": model, "n_engines": n_eng, "steps": [tuple(x) for x in sc.steps]}
    if ext and sc.draft_fps:
        out["draft"] = {"fps": sc.draft_fps, "t_s": sc.draft_t_s, "ramp_s": sc.draft_ramp_s}
    if record:
        tel = {c: v[:k_end] for c, v in rec.items()}
        out["telemetry"] = tel
        out["trace"] = tel  # plot_results.py reads r["trace"]
    if mdl_v2 is not None:
        out["flex_mode"] = "v2"
        out["struct_mass_delta_lb"] = float(v2_mass["total_lb"])
        if status == "ok" and fdm.started:  # FD's post-flight v2 terms (FlexHookV2.finish): loads, hinge terms, ultimate
            r2 = fd_bridge.flexbody().response_terms_v2(fdm.history(), mdl_v2, fdm.out_1g, fd_bridge.struct_weights_v2())
            out["flex_v2"] = {"terms": r2["terms"], "fail": r2["fail"], "loads": r2["loads"], "allowables": r2["allowables"],
                              "bm_allow": r2["bm_allow"], "tip_max_ft": r2["tip_max_ft"], "twist_max_deg": r2["twist_max_deg"],
                              "tail_ratio": r2["tail_ratio"], "fus_ratio": r2["fus_ratio"], "m_root_1g": r2["m_root_1g"]}
    if coupler is not None:
        out["flex_mode"] = sc.flex_mode
        out["wing_mass_delta_lb"] = coupler.delta_mass_lb
        hist = fdm.history()
        if len(hist["root_bm_lbft_R"]):
            ch = fd_bridge.flexwing().telemetry_channels(hist, coupler.w, fdm.m_root_1g)
            out["flex_params"] = ch.pop("params")
            if record:
                n_h = len(hist["root_bm_lbft_R"])
                for c, v in ch.items():
                    out["telemetry"][c] = np.asarray(v)[:n_h]
    return out
