"""ctrlsurf_p4.py -- Phase 4 control-surface model (PHASE4_FD_CONTROL_SURFACES.md).

Surface tables (JSBSim-sourced / literature / notional), a deterministic first-order-lag + rate-limit + hinge blow-down
actuator in JSBSim command-norm space, Ch_alpha/Ch_delta hinge moments, flex control effectiveness eta(q) from the SAME
B2a model and solver as the margin screen (diagnostic only: the flown elastic effect is already in the B2a coupler,
which loads the structure from the JSBSim surface positions), and a per-step signal recorder.

Read-only with respect to every frozen A1 / B1 r1 / B2a module (imports only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import flexbody as fb
import flexwing as fw

CS_SCHEMA = "fd-ctrlsurf/1"
CS_TAG = "p4cs0"
D2R = math.pi / 180.0


@dataclass(frozen=True)
class PosProp:
    prop: str
    unit: str            # 'rad' | 'deg' | 'norm'
    deg_per_unit: float  # deg per unit of the property (rad: 57.2958, deg: 1, norm: notional scale)


@dataclass(frozen=True)
class SurfaceDef:
    name: str                       # elev | ail | rud | flap | sbrk
    cmd_prop: str
    pos: Tuple[PosProp, ...]        # one (symmetric) or two (L, R) position properties
    deg_neg: float                  # deflection (deg) at command -1 (0 for flap / sbrk)
    deg_pos: float                  # deflection (deg) at command +1
    rate_dps: Optional[float]       # FD rate limit (deg/s); None = pass-through (JSBSim FCS only)
    tau_s: Optional[float]          # FD first-order lag (s); None = pass-through
    src: str                        # J = JSBSim XML, L = literature, N = notional (per field, see doc)
    cmd_lo: float = -1.0
    cmd_hi: float = 1.0
    hinge: bool = True              # hinge moment modelled (flaps / speedbrake: no)
    all_moving: bool = False

    @property
    def deg_per_norm(self) -> float:
        return max(abs(self.deg_neg), abs(self.deg_pos), 1e-9)

    def cmd_to_deg(self, u: float) -> float:
        return u * (self.deg_pos if u >= 0 else -self.deg_neg)


R2D = 180.0 / math.pi
_rad = lambda p: PosProp(p, "rad", R2D)        # noqa: E731
_deg = lambda p: PosProp(p, "deg", 1.0)        # noqa: E731
_nrm = lambda p, k: PosProp(p, "norm", k)      # noqa: E731

# Sources: c172x/T38/737/f16 XML in jsbsim_root_v2 (JSBSim 1.3.1 shipped models + FD prep); F-16 rates/tau: Stevens &
# Lewis, Aircraft Control and Simulation (2003) F-16 model (elev 60 deg/s, ail 80, rud 120, tau = 1/20.2 s).
SURFACES: Dict[str, Tuple[SurfaceDef, ...]] = {
    "c172x": (
        SurfaceDef("elev", "fcs/elevator-cmd-norm", (_rad("fcs/elevator-pos-rad"),), -28.0, 23.0, 60.0, 0.05,
                   "range J (clip +-19.5 J, lag 60/s J); rate N; tau N"),
        SurfaceDef("ail", "fcs/aileron-cmd-norm", (_rad("fcs/left-aileron-pos-rad"), _rad("fcs/right-aileron-pos-rad")),
                   -20.0, 15.0, 90.0, 0.05, "range J; rate J (1.57 rad/s); tau N"),
        SurfaceDef("rud", "fcs/rudder-cmd-norm", (_rad("fcs/rudder-pos-rad"),), -16.0, 16.0, 60.0, 0.05,
                   "range J; rate N; tau N"),
        SurfaceDef("flap", "fcs/flap-cmd-norm", (_deg("fcs/flap-pos-deg"),), 0.0, 30.0, None, None,
                   "J kinematic 0-10 2 s, 10-20 1 s, 20-30 1 s", cmd_lo=0.0, hinge=False),
    ),
    "T38": (
        SurfaceDef("elev", "fcs/elevator-cmd-norm", (_nrm("fcs/elevator-pos-norm", 17.0),), -17.0, 9.9, 40.0, 0.05,
                   "norm range J (-1..0.583); 17 deg/norm N; rate N; tau N"),
        SurfaceDef("ail", "fcs/aileron-cmd-norm", (_nrm("fcs/left-aileron-pos-norm", 20.0),
                                                   _nrm("fcs/right-aileron-pos-norm", 20.0)),
                   -20.0, 15.0, 60.0, 0.05, "norm range J (L -1/0.75, R -1/0.65); 20 deg/norm = flexwing assumption N"),
        SurfaceDef("rud", "fcs/rudder-cmd-norm", (_nrm("fcs/rudder-pos-norm", 30.0),), -30.0, 30.0, 60.0, 0.05,
                   "norm range J; 30 deg/norm N; rate N; tau N"),
        SurfaceDef("flap", "fcs/flap-cmd-norm", (_deg("fcs/flap-pos-deg"),), 0.0, 40.0, None, None,
                   "J kinematic 24 deg in 5 s, 40 in 3 s more", cmd_lo=0.0, hinge=False),
        SurfaceDef("sbrk", "fcs/speedbrake-cmd-norm", (_nrm("fcs/speedbrake-pos-norm", 1.0),), 0.0, 1.0, None, None,
                   "J kinematic 0..1 in 1 s (norm, not deg)", cmd_lo=0.0, hinge=False),
    ),
    "737": (
        SurfaceDef("elev", "fcs/elevator-cmd-norm", (_rad("fcs/elevator-pos-rad"),), -17.19, 17.19, 40.0, 0.08,
                   "range J (+-0.3 rad); rate N; tau N"),
        SurfaceDef("ail", "fcs/aileron-cmd-norm", (_rad("fcs/left-aileron-pos-rad"), _rad("fcs/right-aileron-pos-rad")),
                   -20.05, 20.05, 45.0, 0.08, "range J (+-0.35 rad); rate N; tau N"),
        SurfaceDef("rud", "fcs/rudder-cmd-norm", (_rad("fcs/rudder-pos-rad"),), -20.05, 20.05, 40.0, 0.08,
                   "range J (+-0.35 rad); rate N; tau N"),
        SurfaceDef("flap", "fcs/flap-cmd-norm", (_nrm("fcs/flap-pos-norm", 1.0),), 0.0, 1.0, None, None,
                   "J kinematic (norm, 22 s full)", cmd_lo=0.0, hinge=False),
        SurfaceDef("sbrk", "fcs/speedbrake-cmd-norm", (_nrm("fcs/speedbrake-pos-norm", 1.0),), 0.0, 1.0, None, None,
                   "J kinematic 0..1 in 0.6 s (norm)", cmd_lo=0.0, hinge=False),
    ),
    "f16": (
        SurfaceDef("elev", "fcs/elevator-cmd-norm", (_rad("fcs/elevator-pos-rad"),), -25.0, 25.0, 60.0, 0.0495,
                   "range J (+-0.436 rad, cmd limiter -1/+0.44 J); rate L; tau L", all_moving=True),
        SurfaceDef("ail", "fcs/aileron-cmd-norm", (_rad("fcs/aileron-pos-rad"),), -21.5, 21.5, 80.0, 0.0495,
                   "range J (+-0.375 rad; aero uses fcs/aileron-pos-rad); rate L; tau L"),
        SurfaceDef("rud", "fcs/rudder-cmd-norm", (_rad("fcs/rudder-pos-rad"),), -30.0, 30.0, 120.0, 0.0495,
                   "range J (+-0.524 rad); rate L; tau L"),
        SurfaceDef("sbrk", "fcs/speedbrake-cmd-norm", (_deg("fcs/speedbrake-pos-deg"),), 0.0, 43.0, None, None,
                   "J (FCS-scheduled, 43 deg limit)", cmd_lo=0.0, hinge=False),
    ),
}
SURFACES["c172p"] = SURFACES["c172r"] = SURFACES["c172x"]

# Hinge-moment model (notional, labelled N): plain sealed surface with partial aero balance.
CH_ALPHA = -0.10     # /rad
CH_DELTA = -0.35     # /rad
CH_DELTA_ALL_MOVING = -0.05
SURF_GEOM = {"ail": (0.12, 0.25), "elev": (0.35, 0.35), "rud": (0.30, 0.30)}   # (area fraction, chord fraction)
H_MAX_FACTOR = 1.25
ETA_GRID_N = 48
ETA_Q_FRAC = 1.2
ETA_BLOCK = {"ail": ("wingR", "aileron"), "elev": ("empennage_pitch", "elevator"), "rud": ("empennage_yaw", "rudder")}


def surfaces_for(model: str) -> Tuple[SurfaceDef, ...]:
    return SURFACES[model]


def surface_table(model: str) -> List[Dict]:
    out = []
    for s in SURFACES[model]:
        d = asdict(s)
        d["pos"] = [asdict(p) for p in s.pos]
        d["deg_per_norm"] = s.deg_per_norm
        d["mode_default"] = "fd_actuator" if s.tau_s is not None else "passthrough (JSBSim FCS only)"
        out.append(d)
    return out


class Actuator:
    """First-order lag (exact ZOH) + rate limit + clip, in command-norm space. Deterministic."""

    def __init__(self, s: SurfaceDef, dt: float, y0: float):
        self.s, self.dt = s, dt
        self.y = float(y0)
        self.a = 1.0 - math.exp(-dt / s.tau_s) if s.tau_s else 1.0
        self.dmax = (s.rate_dps / s.deg_per_norm) * dt if s.rate_dps else math.inf
        self.rate_sat = self.pos_sat = self.hinge_sat = False

    def step(self, cmd: float, u_hinge_max: float = math.inf) -> float:
        s = self.s
        uc = min(max(cmd, s.cmd_lo), s.cmd_hi)
        self.pos_sat = (cmd <= s.cmd_lo) or (cmd >= s.cmd_hi)
        target = self.y + self.a * (uc - self.y)
        d = target - self.y
        self.rate_sat = abs(d) > self.dmax
        if self.rate_sat:
            d = math.copysign(self.dmax, d)
        y = self.y + d
        self.hinge_sat = abs(y) > u_hinge_max
        if self.hinge_sat:
            y = math.copysign(u_hinge_max, y)
        self.y = y
        return y


@dataclass
class HingeGeom:
    S_c: float          # ft^2 (per side for ailerons)
    c_c: float          # ft
    ch_alpha: float
    ch_delta: float
    h_max: float        # lbf ft
    eta_mid: float = 0.0


def hinge_geometry(model: str, obj) -> Dict[str, HingeGeom]:
    """Notional per-aircraft hinge geometry from the FIXED baseline geometry (not gene dependent)."""
    geo = obj.geom
    pw = obj.pw
    qd = fb._q_of_keas(pw.v_dive_keas)
    e0, e1 = getattr(pw, "ail_eta", (0.6, 0.95)) or (0.6, 0.95)
    cbar_w = geo.sw_ft2 / geo.bw_ft
    hs, vs = obj.ht_spec, obj.vt_spec
    out = {}
    sdefs = {s.name: s for s in SURFACES[model]}
    for nm, (fa, fc) in SURF_GEOM.items():
        s = sdefs[nm]
        if nm == "ail":
            S, c = fa * geo.sw_ft2 * (e1 - e0), fc * cbar_w
        elif nm == "elev":
            S, c = fa * hs.area_ft2, fc * hs.area_ft2 / hs.span_ft
        else:
            S, c = fa * vs.area_ft2 / 2.0, fc * vs.area_ft2 / vs.span_ft
        if s.all_moving:
            S, c = hs.area_ft2, hs.area_ft2 / hs.span_ft
        chd = CH_DELTA_ALL_MOVING if s.all_moving else CH_DELTA
        hmax = H_MAX_FACTOR * qd * S * c * abs(chd) * s.deg_per_norm * D2R
        out[nm] = HingeGeom(S, c, CH_ALPHA, chd, hmax, 0.5 * (e0 + e1))
    return out


def eta_tables(obj) -> Dict[str, Dict]:
    """Elastic / rigid control effectiveness eta(q) per control on a fixed q grid (0 .. 1.2 q_D), same solver as the
    margin screen. Plus the rigid control load per (q * rad) (aileron: roll moment lbf ft; elevator: HT lift lbf;
    rudder: fin side force lbf)."""
    qd = fb._q_of_keas(obj.pw.v_dive_keas)
    qs = np.concatenate([[0.0], np.geomspace(1e-3 * qd, ETA_Q_FRAC * qd, ETA_GRID_N - 1)])
    out = {}
    for nm, (bname, ctrl) in ETA_BLOCK.items():
        idx = fb._block(obj, fb.BLOCKS[bname])
        B = fb.BlockAero(obj, idx)
        try:
            b, r_rig, rG, gk, inc = fb.control_vectors(obj, ctrl, idx)
            e = [1.0] + [float(B.effectiveness(float(q), b, r_rig, rG, gk, inc)) for q in qs[1:]]
        except (np.linalg.LinAlgError, ValueError):
            r_rig, e = float("nan"), [float("nan")] * qs.size
        e = np.asarray(e)
        e = np.where(np.isfinite(e), np.clip(e, -5.0, 5.0), np.nan)
        out[nm] = {"q_psf": qs, "eta": e, "r_rigid": float(r_rig)}
    return out


def eta_at(tab: Dict, q: float) -> float:
    return float(np.interp(q, tab["q_psf"], tab["eta"]))


class CSRecorder:
    """Per-step arrays (120 Hz) + summary."""

    def __init__(self, sdefs, record: bool):
        self.sdefs = sdefs
        self.record = record
        self.keys = ["t_s"]
        for s in sdefs:
            n = s.name
            self.keys += [f"{n}_cmd_norm", f"{n}_act_norm", f"{n}_cmd_deg", f"{n}_deg", f"{n}_rate_dps",
                          f"{n}_pos_sat", f"{n}_rate_sat", f"{n}_hinge_sat"]
            if len(s.pos) == 2:
                self.keys += [f"{n}_L_deg", f"{n}_R_deg"]
            if s.hinge:
                self.keys += [f"{n}_hinge_lbft", f"{n}_eta", f"{n}_power_eff"]
                if len(s.pos) == 2:
                    self.keys += [f"{n}_L_hinge_lbft"]
        self.h = {k: [] for k in self.keys}

    def add(self, row: Dict[str, float]):
        for k in self.keys:
            self.h[k].append(row[k])

    def arrays(self):
        return {k: np.asarray(v, dtype=float) for k, v in self.h.items()}


def summarize(arr: Dict[str, np.ndarray], sdefs, hg: Dict[str, HingeGeom], dt: float) -> Dict:
    n = arr["t_s"].size
    T = n * dt
    out = {"n_steps": int(n), "duration_s": float(T)}
    if n == 0:
        return out
    for s in sdefs:
        k = s.name
        d = arr[f"{k}_deg"]
        rate = arr[f"{k}_rate_dps"]
        o = {"use_tv_deg_per_s": float(np.sum(np.abs(rate)) * dt / T),
             "use_rms_deg": float(np.sqrt(np.mean((d - d[0]) ** 2))),
             "use_rms_deg_abs": float(np.sqrt(np.mean(d ** 2))),
             "sat_frac_pos": float(np.mean(arr[f"{k}_pos_sat"])),
             "sat_frac_rate": float(np.mean(arr[f"{k}_rate_sat"])),
             "sat_frac_hinge": float(np.mean(arr[f"{k}_hinge_sat"])),
             "lag_rms_deg": float(np.sqrt(np.mean((arr[f"{k}_cmd_deg"] - d) ** 2))),
             "rate_peak_dps": float(np.max(np.abs(rate))), "defl_peak_deg": float(np.max(np.abs(d))),
             "norm_defl_max_deg": s.deg_per_norm, "norm_rate_max_dps": s.rate_dps}
        if s.hinge:
            H = np.abs(arr[f"{k}_hinge_lbft"])
            if f"{k}_L_hinge_lbft" in arr:
                H = np.maximum(H, np.abs(arr[f"{k}_L_hinge_lbft"]))
            eta = arr[f"{k}_eta"]
            moving = np.abs(rate) > 0.5
            o.update({"hinge_peak_lbft": float(H.max()), "hinge_peak_frac": float(H.max() / hg[k].h_max),
                      "eta_min": float(np.nanmin(eta)), "eta_mean": float(np.nanmean(eta)),
                      "reversal_flown": bool(np.any((eta <= 0) & moving))})
        out[k] = o
    return out


class CSProxy:
    """Wraps the B2a FlexBodyFDM proxy. Intercepts the controller's fcs/*-cmd-norm writes; on run(): FD actuator
    (active) -> write -> inner run (B2a coupler + JSBSim) -> read-only signal capture. passthrough: writes go through
    unchanged at once; capture is read-only (bit-identical flight)."""

    def __init__(self, inner, raw_fdm, model: str, obj, dt: float, mode: str, record: bool,
                 eta_tab: Dict, hg: Dict[str, HingeGeom], overrides: Optional[Dict] = None):
        self._inner, self._raw = inner, raw_fdm
        self.mode, self.dt = mode, dt
        sdefs = []
        for s in SURFACES[model]:
            ov = (overrides or {}).get(s.name)
            sdefs.append(SurfaceDef(**{**asdict(s), "pos": s.pos, **ov}) if ov else s)
        self.sdefs = tuple(sdefs)
        self.by_cmd = {s.cmd_prop: s for s in self.sdefs}
        self.cmd = {s.name: float(raw_fdm[s.cmd_prop]) for s in self.sdefs}
        self.act = {s.name: Actuator(s, dt, self.cmd[s.name]) for s in self.sdefs}
        self.eta_tab, self.hg = eta_tab, hg
        self.rec = CSRecorder(self.sdefs, record)
        self.prev_deg = {s.name: self._pos_deg(s)[0] for s in self.sdefs}
        self.k = 0
        self.rho_hint = None

    # proxy protocol ------------------------------------------------------------------------------------------------
    def __getitem__(self, k):
        return self._inner[k]

    def __setitem__(self, k, v):
        s = self.by_cmd.get(k)
        if s is not None:
            self.cmd[s.name] = float(v)
            if self.mode == "passthrough" or s.tau_s is None:
                self._inner[k] = v
            return
        self._inner[k] = v

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # ---------------------------------------------------------------------------------------------------------------
    def _pos_deg(self, s: SurfaceDef):
        vals = [self._raw[p.prop] * p.deg_per_unit for p in s.pos]
        if len(vals) == 2:
            return 0.5 * (vals[0] - vals[1]) if s.name == "ail" else 0.5 * (vals[0] + vals[1]), vals
        return vals[0], vals

    def _hinge_limit_norm(self, s: SurfaceDef, q: float, alpha_s_rad: float) -> float:
        if not s.hinge or q <= 0:
            return math.inf
        g = self.hg[s.name]
        k = q * g.S_c * g.c_c
        # |k (Cha a + Chd d)| <= Hmax  ->  |d| <= (Hmax/k + sign * Cha a)/|Chd| (worst side)
        dlim = (g.h_max / k - abs(g.ch_alpha * alpha_s_rad)) / abs(g.ch_delta)
        return max(dlim, 0.0) * R2D / s.deg_per_norm

    def _alpha_s(self, name: str, side: int = 0) -> float:
        f = self._raw
        last = getattr(self._inner.coupler, "last", None) or {}
        if name == "ail":
            tw = last.get("tip_twist_R_deg" if side == 0 else "tip_twist_L_deg", 0.0)
            return f["aero/alpha-rad"] + self.hg["ail"].eta_mid * tw * D2R
        if name == "elev":
            return f["aero/alpha-rad"] + last.get("ht_incidence_deg", 0.0) * D2R
        return f["aero/beta-rad"] + last.get("vt_sideslip_deg", 0.0) * D2R

    def run(self):
        f = self._raw
        q = f["aero/qbar-psf"]
        if self.mode == "active":
            for s in self.sdefs:
                if s.tau_s is None:
                    continue
                ulim = self._hinge_limit_norm(s, q, self._alpha_s(s.name)) if s.hinge else math.inf
                self._inner[s.cmd_prop] = self.act[s.name].step(self.cmd[s.name], ulim)
        r = self._inner.run()
        self._capture()
        return r

    def _capture(self):
        f = self._raw
        self.k += 1
        q = f["aero/qbar-psf"]
        row = {"t_s": self.k * self.dt}
        for s in self.sdefs:
            n = s.name
            a = self.act[n]
            d, vals = self._pos_deg(s)
            u = self.cmd[n]
            act = a.y if (self.mode == "active" and s.tau_s is not None) else min(max(u, s.cmd_lo), s.cmd_hi)
            row[f"{n}_cmd_norm"] = u
            row[f"{n}_act_norm"] = act
            row[f"{n}_cmd_deg"] = s.cmd_to_deg(min(max(u, s.cmd_lo), s.cmd_hi))
            row[f"{n}_deg"] = d
            row[f"{n}_rate_dps"] = (d - self.prev_deg[n]) / self.dt
            self.prev_deg[n] = d
            psat = (u <= s.cmd_lo) or (u >= s.cmd_hi)
            if self.mode == "active" and s.tau_s is not None:
                row[f"{n}_pos_sat"] = float(a.pos_sat or a.hinge_sat)
                row[f"{n}_rate_sat"] = float(a.rate_sat)
                row[f"{n}_hinge_sat"] = float(a.hinge_sat)
            else:
                row[f"{n}_pos_sat"] = float(psat)
                row[f"{n}_rate_sat"] = 0.0
                row[f"{n}_hinge_sat"] = 0.0
            if len(vals) == 2:
                row[f"{n}_L_deg"], row[f"{n}_R_deg"] = vals
            if s.hinge:
                g = self.hg[n]
                k = q * g.S_c * g.c_c
                if len(vals) == 2:
                    row[f"{n}_hinge_lbft"] = k * (g.ch_alpha * self._alpha_s(n, 0) + g.ch_delta * vals[1] * D2R)
                    row[f"{n}_L_hinge_lbft"] = k * (g.ch_alpha * self._alpha_s(n, 1) + g.ch_delta * vals[0] * D2R)
                else:
                    row[f"{n}_hinge_lbft"] = k * (g.ch_alpha * self._alpha_s(n) + g.ch_delta * d * D2R)
                tab = self.eta_tab[n]
                eta = eta_at(tab, q)
                row[f"{n}_eta"] = eta
                row[f"{n}_power_eff"] = eta * tab["r_rigid"] * q * D2R
        self.rec.add(row)
