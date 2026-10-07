"""Minimal aeroelastic (flexible-wing) model coupled to JSBSim.

Units: ft, slug, lbf, s, rad (US customary, like JSBSim internals).

Model (per semi-span wing, left and right wings integrated independently):
  * Cantilever beam from the fuselage side (y0) to the tip (s), strip-discretised.
  * Assumed modes (Rayleigh-Ritz): 1st and 2nd cantilever bending
    (uniform-beam eigenfunctions) + 1st torsion (sin(pi*xi/2)).
  * Generalised mass M (bending-torsion inertial coupling via section CG offset
    x_theta from the elastic axis, plus an optional tip mass-balance weight),
    stiffness K (EI, GJ calibrated so the uncoupled baseline frequencies match
    f_b1 / f_t1), modal damping C (zeta).
  * Strip-theory aero: incremental local angle of attack from elastic
    deformation  d_alpha = theta*cos(L) - w'*sin(L) - wdot/V + d34*thetadot/V
    (+ beta*w' elastic-dihedral term), lift per span = qbar*c*a*d_alpha, acting
    at the quarter chord (moment arm e*c about the elastic axis).
  * External ("rigid") loads from the JSBSim state each frame: total aero lift
    spread with Schrenk's approximation, antisymmetric roll-rate and aileron
    strip loads, inertia relief (-m*g*Nz, + m*pdot*y).
  * Integration: Newmark average-acceleration (implicit, unconditionally
    stable) with aero stiffness/damping inside the system matrices, sub-stepped
    within each JSBSim frame.
  * Feedback to the rigid-body FDM: elastic lift increments (relative to the
    1-g trim deformation, which the aircraft's aero tables are assumed to
    already contain) summed into a body-frame force at AERORP and roll / pitch
    moments, injected through JSBSim <external_reactions> (see
    prepare_aircraft()).
  * Stability margins (cheap, per genome): divergence q from det(K - q A_K)=0,
    quasi-steady modal flutter (eigenvalues of the aeroelastic state matrix
    swept over EAS), and steady-aero frequency coalescence.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

G0 = 32.174            # ft/s^2
RHO0 = 0.0023769       # slug/ft^3, ISA sea level
# Aeroelastic margin ceiling (x V_D). Margins are searched up to this value and reported as min(value, cap);
# "not found below the cap" is flagged instead of returning inf. Fitness penalties act below 1.2, so the cap
# never changes a penalty; it only keeps outputs finite and comparable.
MARGIN_CAP = 3.0
# Section CG must lie at least this far (chord fraction) aft of the elastic axis. CG ahead of the EA is a
# mass-balanced section; with notional chord fractions the GA exploited it (flutter "never" occurs).
# Enforced by raising ValueError (WingParams construction and FlexWing construction).
MIN_CG_AFT_OF_EA = 0.02
_SECTION_AXES_CHECK = [True]
KT2FPS = 1.6878099
LB2SLUG = 1.0 / G0

# Uniform cantilever bending eigenvalues and shape constants.
_BETA = (1.8751040687, 4.6940911330)
_SIGMA = (0.7340955137, 1.0184673165)


def _bend(n: int, xi: np.ndarray, deriv: int = 0) -> np.ndarray:
    """n-th uniform cantilever bending shape (normalised to 1 at the tip) and derivatives wrt xi."""
    b, s = _BETA[n], _SIGMA[n]
    tip = math.cosh(b) - math.cos(b) - s * (math.sinh(b) - math.sin(b))
    x = b * np.asarray(xi, float)
    if deriv == 0:
        f = np.cosh(x) - np.cos(x) - s * (np.sinh(x) - np.sin(x))
    elif deriv == 1:
        f = b * (np.sinh(x) + np.sin(x) - s * (np.cosh(x) - np.cos(x)))
    elif deriv == 2:
        f = b * b * (np.cosh(x) + np.cos(x) - s * (np.sinh(x) + np.sin(x)))
    else:
        raise ValueError(deriv)
    return f / tip


def _tors(xi: np.ndarray, deriv: int = 0) -> np.ndarray:
    x = np.asarray(xi, float)
    if deriv == 0:
        return np.sin(0.5 * math.pi * x)
    return 0.5 * math.pi * np.cos(0.5 * math.pi * x)


def datcom_cla(aspect: float, sweep_rad: float, eta: float = 0.95, mach: float = 0.0) -> float:
    """DATCOM/Helmbold lift-curve slope (per rad), with Prandtl-Glauert beta (Mach clamped to 0.9)."""
    b2 = 1.0 - min(max(mach, 0.0), 0.9) ** 2
    return 2 * math.pi * aspect / (2 + math.sqrt(4 + aspect ** 2 * b2 / eta ** 2 * (1 + math.tan(sweep_rad) ** 2 / b2)))


def speed_of_sound_fps(rho: float) -> float:
    """ISA troposphere speed of sound from density (rho/rho0 = (T/T0)^4.256)."""
    return 1116.45 * math.sqrt(min(1.0, rho / RHO0) ** (1 / 4.256))


def flap_coeffs(cf_c: float) -> Tuple[float, float]:
    """Thin-airfoil flap (aileron) dCl/d(delta) and dCm_ac/d(delta), per rad."""
    th = math.acos(2 * cf_c - 1)
    cl_d = 2 * (math.pi - th + math.sin(th))
    cm_d = -0.5 * math.sin(th) * (1 - math.cos(th))
    return cl_d, cm_d


@dataclass
class WingParams:
    # geometry (whole aircraft): normally filled from JSBSim metrics
    span_ft: float = 36.0
    area_ft2: float = 174.0
    taper: float = 0.75
    sweep_deg: float = 0.0            # sweep of the elastic axis / quarter chord
    root_frac: float = 0.10           # fuselage half-width as fraction of semi-span (beam root)
    n_strips: int = 16
    cla: Optional[float] = None       # strip lift slope per rad (None -> DATCOM from AR & sweep)
    # structure (baseline, before gene scaling)
    wing_mass_lb: float = 180.0       # both wings, total
    f_b1_hz: float = 7.0              # uncoupled 1st bending frequency at gene scale 1
    f_t1_hz: float = 22.0             # uncoupled 1st torsion frequency at gene scale 1
    n_bend: int = 2                   # 1 or 2 bending modes
    ei_taper_exp: float = 3.0         # EI, GJ ~ (c/c_root)^exp
    mass_taper_exp: float = 1.0       # m ~ (c/c_root)^exp
    r_gyr: float = 0.25               # section radius of gyration about its CG / chord
    # genes (defaults = baseline)
    ei_scale: float = 1.0
    gj_scale: float = 1.0
    x_ea: float = 0.38                # elastic axis, fraction of chord from LE
    x_cg: float = 0.42                # section CG, fraction of chord from LE
    zeta: float = 0.02                # structural damping ratio
    tip_mass_frac: float = 0.0        # tip mass-balance weight / semi-wing baseline mass
    x_tip_mass: float = 0.10          # chordwise position of tip mass (fraction of chord)
    cal_x_ea: Optional[float] = None  # baseline EA/CG used to calibrate GJ from f_t1 (None -> x_ea/x_cg);
    cal_x_cg: Optional[float] = None  # keeps GJ independent of the x_ea/x_cg genes
    # mass model for the stiffness-mass trade-off
    struct_frac: float = 0.55         # fraction of wing mass that is primary structure
    nonstruct_scale: float = 1.0      # multiplier on the non-structural (1 - struct_frac) mass share (fuel/systems/secondary)
    w_ei: float = 0.5                 # share of structure sized by bending (rest by torsion)
    # aileron (strip loads only)
    ail_eta: Tuple[float, float] = (0.60, 0.95)
    ail_cf: float = 0.25
    ail_cl_da_target: Optional[float] = None   # FDM rolling-moment derivative per rad (None -> 3D strip theory)
    ail_norm_deg: Optional[float] = None       # if set, read fcs/*-aileron-pos-norm x this (models that only write -norm, e.g. T38)
    ail_antisym_prop: Optional[str] = None     # if set, read this single antisymmetric property (rad, + = right roll) as
                                               # left = +x, right = -x (f16: fcs/aileron-pos-rad, the property its Clda uses)
    wing_lift_share: float = 1.0      # fraction of total aero lift carried by the wing
    # certification-style reference speeds/limits
    v_dive_keas: float = 180.0
    n_limit: float = 3.8
    tip_defl_limit_frac: float = 0.08  # of semi-span
    twist_limit_deg: float = 3.0
    flutter_criterion: str = "min"    # "min" (conservative: min of quasi-steady p-method and steady coalescence) | "qs" | "coalescence"

    def __post_init__(self):
        check_section_axes(self)


def check_section_axes(p: "WingParams") -> None:
    """Raise ValueError unless x_cg >= x_ea + MIN_CG_AFT_OF_EA (Phase-1 policy: raise, never clamp silently)."""
    if not _SECTION_AXES_CHECK[0]:
        return
    if not (p.x_cg >= p.x_ea + MIN_CG_AFT_OF_EA - 1e-9):
        raise ValueError(f"section CG x_cg={p.x_cg:.3f} must be >= elastic axis x_ea={p.x_ea:.3f} + {MIN_CG_AFT_OF_EA} chord "
                         "(CG at/ahead of the EA is a mass-balanced section; not allowed in Phase 1, see INTERFACE.md)")


class unchecked_section_axes:
    """Context manager that disables check_section_axes. For analytic validation cases and tests ONLY
    (e.g. uncoupled sections with x_cg == x_ea). Not reachable through params_for/genome overrides."""

    def __enter__(self):
        self._prev = _SECTION_AXES_CHECK[0]
        _SECTION_AXES_CHECK[0] = False
        return self

    def __exit__(self, *exc):
        _SECTION_AXES_CHECK[0] = self._prev
        return False


AIRCRAFT_PROFILES: Dict[str, Dict] = {
    # All structural numbers are notional (no public GVT data used); see AEROELASTIC_DESIGN.md.
    "c172x": dict(taper=0.75, sweep_deg=0.0, wing_mass_lb=180.0, f_b1_hz=7.0, f_t1_hz=22.0, x_ea=0.38, x_cg=0.42,
                  ail_cl_da_target=0.23,   # c172x.xml aero/coefficient/Clda
                  v_dive_keas=180.0, n_limit=3.8, tip_defl_limit_frac=0.08),
    "737": dict(taper=0.28, sweep_deg=25.0, wing_mass_lb=10500.0, f_b1_hz=2.7, f_t1_hz=11.0, x_ea=0.36, x_cg=0.38,
                v_dive_keas=400.0, n_limit=2.5, tip_defl_limit_frac=0.12, ail_eta=(0.70, 0.95),
                ail_cl_da_target=0.09),     # 737.xml Clda table (0.100 @ M0 -> 0.033 @ M2), ~M0.45
}
AIRCRAFT_PROFILES["T38"] = dict(
    taper=0.20, sweep_deg=24.0, wing_mass_lb=800.0, f_b1_hz=10.0, f_t1_hz=35.0, x_ea=0.40, x_cg=0.42,
    v_dive_keas=595.0,          # screening capped at M0.9 at sea level: strip theory + Prandtl-Glauert is not valid transonic
    n_limit=7.33, tip_defl_limit_frac=0.06, ail_eta=(0.45, 0.85),
    ail_norm_deg=20.0,          # T38.xml writes only fcs/*-aileron-pos-norm; 20 deg per unit norm assumed
    ail_cl_da_target=0.11)      # T38.xml CldaL/CldaR 0.0193 per unit norm per side -> 0.0386 / 0.349 rad
AIRCRAFT_PROFILES["f16"] = dict(
    # F-16A (JSBSim f16.xml: S 300 ft2, b 30 ft, AR 3.0). Planform: taper 0.23 (root 16.3 ft / tip 3.75 ft chord),
    # quarter-chord sweep ~32 deg (LE 40 deg). Beam root at the side of the blended body: 0.17 * 15 ft = 2.55 ft.
    taper=0.23, sweep_deg=32.0, root_frac=0.17,
    wing_mass_lb=1600.0,        # both wings, notional (~9 % of the 17 400 lb empty weight)
    f_b1_hz=8.5, f_t1_hz=28.0,  # notional clean-wing uncoupled frequencies (no tip launchers / stores modelled)
    x_ea=0.40, x_cg=0.43,       # fixed Phase-1 axes, see INTERFACE.md section 5 (multi-spar box; integral fuel centred in box)
    v_dive_keas=595.0,          # screening capped at M0.9 at sea level like T38 (real placard 800 KCAS is outside strip theory)
    n_limit=9.0, tip_defl_limit_frac=0.05,
    ail_eta=(0.20, 0.65),       # flaperons: inboard trailing edge, side of body to ~2/3 semi-span
    ail_cl_da_target=0.051,     # f16.xml Clda ~0.051 per rad of fcs/aileron-pos-rad at alpha 0 (lumps flaperon + diff. tail)
    # The FDM's roll moment uses fcs/aileron-pos-rad (FCS roll command x 0.375, no actuator lag, no Mach schedule); the
    # flaperon actuator outputs fcs/left|right-aileron-pos-rad are Mach-scaled (x(1-0.85M), 0.46 at M0.63) and unused
    # by the aero. Strip loads follow the FDM so the elastic/rigid roll ratio is consistent with JSBSim.
    ail_antisym_prop="fcs/aileron-pos-rad")
AIRCRAFT_PROFILES["c172p"] = AIRCRAFT_PROFILES["c172x"]
AIRCRAFT_PROFILES["c172r"] = AIRCRAFT_PROFILES["c172x"]


def params_for(model: str, span_ft: float, area_ft2: float, empty_wt_lb: Optional[float] = None, **overrides) -> WingParams:
    """Profile lookup; unknown aircraft get a generic scaling (f_b1 ~ 126/semispan Hz, f_t/f_b = 3)."""
    prof = dict(AIRCRAFT_PROFILES.get(model, {}))
    if not prof:
        s = span_ft / 2
        prof = dict(f_b1_hz=126.0 / s, f_t1_hz=3 * 126.0 / s,
                    wing_mass_lb=0.12 * (empty_wt_lb or 10 * area_ft2))
    prof.update(span_ft=span_ft, area_ft2=area_ft2)
    prof.setdefault("cal_x_ea", prof.get("x_ea", WingParams.x_ea))
    prof.setdefault("cal_x_cg", prof.get("x_cg", WingParams.x_cg))
    prof.update(overrides)
    return WingParams(**prof)


class FlexWing:
    """Structural + aero matrices of one semi-span (identical for left and right)."""

    def __init__(self, p: WingParams):
        check_section_axes(p)          # again here: catches params mutated after construction
        self.p = p
        s = p.span_ft / 2
        self.s = s
        self.y0 = p.root_frac * s
        self.L = s - self.y0
        self.lam = math.radians(p.sweep_deg)
        self.c_root = p.area_ft2 / (s * (1 + p.taper))          # centreline chord of the trapezoid
        n = p.n_strips
        self.dy = self.L / n
        self.y = self.y0 + (np.arange(n) + 0.5) * self.dy          # strip midpoints (outboard coordinate)
        self.xi = (self.y - self.y0) / self.L
        self.c = self.chord(self.y)
        aspect = p.span_ft ** 2 / p.area_ft2
        self.a = p.cla if p.cla is not None else datcom_cla(aspect, self.lam)
        self.n_modes = p.n_bend + 1
        nb = p.n_bend
        # mode shapes at strips: w = PhiW @ eta, theta = PsiT @ eta, w' = dPhiW @ eta
        self.PhiW = np.zeros((n, self.n_modes))
        self.dPhiW = np.zeros((n, self.n_modes))
        self.PsiT = np.zeros((n, self.n_modes))
        for k in range(nb):
            self.PhiW[:, k] = _bend(k, self.xi)
            self.dPhiW[:, k] = _bend(k, self.xi, 1) / self.L
        self.PsiT[:, nb] = _tors(self.xi)
        self.e_c = (p.x_ea - 0.25) * self.c          # AC ahead of EA (ft)
        self.d34 = (0.75 - p.x_ea) * self.c          # 3/4-chord point aft of EA (ft)
        self.x_theta = (p.x_cg - p.x_ea) * self.c    # section CG aft of EA (ft)
        self._build_structure()
        self._build_aero()
        self._build_loads()

    # ---------------- geometry / structure ----------------
    def chord(self, y):
        return self.c_root * (1 - (1 - self.p.taper) * np.asarray(y) / self.s)

    def mass_distribution(self, scaled: bool = True):
        """Mass per unit span (slug/ft) at strips, and semi-wing mass (lb)."""
        p = self.p
        w = (self.c / self.c_root) ** p.mass_taper_exp
        m0_semi_lb = p.wing_mass_lb / 2
        if scaled:
            f = (1 - p.struct_frac) * p.nonstruct_scale + p.struct_frac * (p.w_ei * p.ei_scale + (1 - p.w_ei) * p.gj_scale)
        else:
            f = 1.0
        m_semi_lb = m0_semi_lb * f
        m = m_semi_lb * LB2SLUG * w / (np.sum(w) * self.dy)
        return m, m_semi_lb

    def _build_structure(self):
        p, nb, nm = self.p, self.p.n_bend, self.n_modes
        dy, xi, L = self.dy, self.xi, self.L
        # fine quadrature for stiffness integrals (derivatives of shapes)
        nq = 400
        xq = (np.arange(nq) + 0.5) / nq
        cq = self.chord(self.y0 + xq * L)
        stiff_shape = (cq / self.c_root) ** p.ei_taper_exp
        # baseline (unscaled) mass for calibration
        m0, _ = self.mass_distribution(scaled=False)
        xth0 = ((p.cal_x_cg if p.cal_x_cg is not None else p.x_cg) - (p.cal_x_ea if p.cal_x_ea is not None else p.x_ea)) * self.c
        Ia0 = m0 * ((p.r_gyr * self.c) ** 2 + xth0 ** 2)
        kb1 = np.sum(stiff_shape * (_bend(0, xq, 2) / L ** 2) ** 2) * (L / nq)     # per unit EI_root
        mb1 = np.sum(m0 * self.PhiW[:, 0] ** 2) * dy
        kt1 = np.sum(stiff_shape * (_tors(xq, 1) / L) ** 2) * (L / nq)             # per unit GJ_root
        mt1 = np.sum(Ia0 * self.PsiT[:, nb] ** 2) * dy
        self.EI_root0 = (2 * math.pi * p.f_b1_hz) ** 2 * mb1 / kb1
        self.GJ_root0 = (2 * math.pi * p.f_t1_hz) ** 2 * mt1 / kt1
        self.EI_root = self.EI_root0 * p.ei_scale
        self.GJ_root = self.GJ_root0 * p.gj_scale
        self.EI_q = self.EI_root * stiff_shape
        self._xq, self._cq = xq, cq
        # actual (gene-scaled) mass
        m, m_semi_lb = self.mass_distribution(scaled=True)
        self.m = m
        self.Ia = m * ((p.r_gyr * self.c) ** 2 + self.x_theta ** 2)
        self.m_semi_lb = m_semi_lb
        mt = p.tip_mass_frac * p.wing_mass_lb / 2 * LB2SLUG
        self.m_tip_slug = mt
        self.m_semi_total_lb = m_semi_lb + mt * G0
        M = np.zeros((nm, nm))
        K = np.zeros((nm, nm))
        for i in range(nb):
            for j in range(nb):
                M[i, j] = np.sum(m * self.PhiW[:, i] * self.PhiW[:, j]) * dy
                K[i, j] = self.EI_root * np.sum(stiff_shape * _bend(i, xq, 2) * _bend(j, xq, 2)) * (L / nq) / L ** 4
            M[i, nb] = M[nb, i] = -np.sum(m * self.x_theta * self.PhiW[:, i] * self.PsiT[:, nb]) * dy
        M[nb, nb] = np.sum(self.Ia * self.PsiT[:, nb] ** 2) * dy
        K[nb, nb] = self.GJ_root * np.sum(stiff_shape * _tors(xq, 1) ** 2) * (L / nq) / L ** 2
        if mt > 0:  # tip mass at xi=1, chord position x_tip_mass
            c_tip = float(self.chord(self.s))
            d = (p.x_tip_mass - p.x_ea) * c_tip
            phi_t = np.array([float(_bend(k, 1.0)) for k in range(nb)] + [0.0])
            psi_t = np.zeros(nm); psi_t[nb] = 1.0
            M += mt * (np.outer(phi_t, phi_t) - d * (np.outer(phi_t, psi_t) + np.outer(psi_t, phi_t)) + d * d * np.outer(psi_t, psi_t))
        self.M, self.K = M, K
        w2 = np.diag(K) / np.diag(M)
        self.C = np.diag(2 * p.zeta * np.sqrt(w2) * np.diag(M))
        self.Minv = np.linalg.inv(M)

    def _build_aero(self):
        """A_K (per unit qbar) and A_C (per unit qbar/V) so that Q_aero = q*A_K@eta + (q/V)*A_C@etadot."""
        lam = self.lam
        G = math.cos(lam) * self.PsiT - math.sin(lam) * self.dPhiW            # d_alpha per eta
        H = -self.PhiW + self.d34[:, None] * self.PsiT                           # d_alpha per (etadot / V)
        self.G, self.H = G, H
        self.lift_w = self.c * self.dy * self.a                                  # L_i = q*lift_w*d_alpha
        self.W_lift = self.PhiW + self.e_c[:, None] * self.PsiT                  # gen. force per unit strip lift
        self.A_K = self.W_lift.T @ (self.lift_w[:, None] * G)
        self.A_C = self.W_lift.T @ (self.lift_w[:, None] * H)
        # Non-circulatory (apparent-mass) pitch-rate terms of Theodorsen's theory, per unit q/V:
        #   L_nc = (pi/2) c^2 thetadot ,  M_nc,EA = -(pi/2) c^2 d34 thetadot
        # (needed: the circulatory 3/4-chord term alone gives spurious negative torsional damping)
        Lnc = 0.5 * math.pi * self.c ** 2 * self.dy
        self.A_C_nc = self.PhiW.T @ (Lnc[:, None] * self.PsiT) - self.PsiT.T @ ((Lnc * self.d34)[:, None] * self.PsiT)
        self.A_C_circ = self.A_C
        self.A_C = self.A_C + self.A_C_nc
        self._aspect = self.p.span_ft ** 2 / self.p.area_ft2
        self._cla0 = datcom_cla(self._aspect, self.lam)

    def kappa(self, mach: float) -> float:
        """Compressibility factor on circulatory strip lift: CLa(M)/CLa(0) (DATCOM + Prandtl-Glauert, M<=0.9)."""
        return datcom_cla(self._aspect, self.lam, mach=mach) / self._cla0

    def _build_loads(self):
        p = self.p
        s, y = self.s, self.y
        # Schrenk load shape (per unit total semi-span lift), integrated over the whole semi-span
        yy = (np.arange(2000) + 0.5) / 2000 * s
        def shape(v):
            cv = self.chord(v)
            return 0.5 * cv / (p.area_ft2 / 2 / s) + 0.5 * 4 / math.pi * np.sqrt(np.clip(1 - (v / s) ** 2, 0, None))
        norm = np.sum(shape(yy)) * (s / 2000)
        self.schrenk = shape(y) / norm * self.dy                 # fraction of semi-span lift on each strip
        eta = y / s
        self.ail_mask = ((eta >= p.ail_eta[0]) & (eta <= p.ail_eta[1])).astype(float)
        cl_d, cm_d = flap_coeffs(p.ail_cf)
        # Calibrate the aileron strip loads to the FDM's rigid roll power (Cl_da per rad of
        # aileron), so elastic/rigid ratios are consistent with JSBSim; else 3D-correct thin-airfoil.
        strip_cl_da = 2 * np.sum(self.c * self.dy * cl_d * self.ail_mask * y) / (p.area_ft2 * p.span_ft)
        if p.ail_cl_da_target is not None and strip_cl_da > 0:
            k = p.ail_cl_da_target / strip_cl_da
        else:
            k = self.a / (2 * math.pi)
        self.ail_scale = k
        self.cl_d, self.cm_d = k * cl_d, k * cm_d
        self.arm_root = y - self.y0
        # chordwise position of each strip AC relative to the wing MAC quarter chord (aft +)
        y_mac = s / 3 * (1 + 2 * p.taper) / (1 + p.taper)
        self.dx_ac = (y - y_mac) * math.tan(self.lam)

    # ---------------- frequencies / margins ----------------
    def frequencies_hz(self, q: float = 0.0) -> np.ndarray:
        A = self.Minv @ (self.K - q * self.A_K)
        ev = np.linalg.eigvals(A)
        return np.sort(np.sqrt(np.abs(ev.real)) / (2 * math.pi))

    def divergence_q(self, rho: Optional[float] = None) -> float:
        """Smallest positive qbar (psf) with det(K - q*kappa*A_K) = 0 (inf if none).

        rho=None: incompressible. Else kappa(M) with M from the EAS->TAS at density rho (fixed point)."""
        mu = np.linalg.eigvals(np.linalg.solve(self.K, self.A_K))
        mu = mu[(np.abs(mu.imag) < 1e-9 * max(1.0, np.abs(mu).max())) & (mu.real > 0)].real
        if not mu.size:
            return math.inf
        q0 = float(1 / mu.max())
        if rho is None:
            return q0
        q = q0
        a = speed_of_sound_fps(rho)
        for _ in range(30):
            mach = math.sqrt(2 * q / rho) / a
            q = q0 / self.kappa(mach)
        return q

    def state_matrix(self, q: float, v_fps: float, aero_damping: bool = True, kap: float = 1.0) -> np.ndarray:
        n = self.n_modes
        Ke = self.K - q * kap * self.A_K
        Ce = self.C - (q / v_fps * (kap * self.A_C_circ + self.A_C_nc) if aero_damping and v_fps > 0 else 0)
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -self.Minv @ Ke
        A[n:, n:] = -self.Minv @ Ce
        return A

    def flutter(self, v_max_keas: float, rho: float = RHO0, n_grid: int = 120, aero_damping: bool = True,
                compressible: bool = True) -> Dict:
        """Sweep EAS; first speed where an oscillatory mode has Re(lambda) > 0.

        rho sets the TAS used in the aero-damping term (q/V); qbar is from EAS.
        Returns flutter and divergence speeds (KEAS, inf if none up to v_max) and frequency.
        """
        a_snd = speed_of_sound_fps(rho)

        def growth(v_keas):
            ve = v_keas * KT2FPS
            q = 0.5 * RHO0 * ve * ve
            vt = ve * math.sqrt(RHO0 / rho)
            kap = self.kappa(vt / a_snd) if compressible else 1.0
            lam = np.linalg.eigvals(self.state_matrix(q, vt, aero_damping, kap))
            osc = lam[np.abs(lam.imag) > 1e-6]
            real = lam[np.abs(lam.imag) <= 1e-6]
            g_osc = osc.real.max() if osc.size else -math.inf
            g_real = real.real.max() if real.size else -math.inf
            f = abs(osc[np.argmax(osc.real)].imag) / (2 * math.pi) if osc.size else 0.0
            return g_osc, g_real, f
        vs = np.linspace(1.0, v_max_keas, n_grid)
        out = {"v_flutter_keas": math.inf, "f_flutter_hz": float("nan"), "v_div_keas": math.inf}
        prev = None
        for v in vs:
            go, gr, f = growth(v)
            if out["v_flutter_keas"] == math.inf and go > 0:
                lo = prev if prev is not None else 0.0
                hi = v
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if growth(mid)[0] > 0: hi = mid
                    else: lo = mid
                out["v_flutter_keas"] = hi; out["f_flutter_hz"] = growth(hi)[2]
            if out["v_div_keas"] == math.inf and gr > 0:
                out["v_div_keas"] = v
            prev = v
            if out["v_flutter_keas"] < math.inf and out["v_div_keas"] < math.inf:
                break
        return out

    def coalescence_q(self, q_max: float, n_grid: int = 400, rho: Optional[float] = None) -> float:
        """Steady-aero frequency coalescence: first qbar where eig(M^-1 (K - q kappa A_K)) turn complex.
        With rho, kappa(M) is evaluated at the TAS for that qbar (EAS-based), so A becomes q-dependent."""
        if rho is None:
            return coalescence_q_matrices(self.M, self.K, self.A_K, q_max, n_grid)
        a = speed_of_sound_fps(rho)
        Minv = np.linalg.inv(self.M)
        def cplx(q):
            kap = self.kappa(math.sqrt(2 * q / rho) / a)
            ev = np.linalg.eigvals(Minv @ (self.K - q * kap * self.A_K))
            return bool(np.any(np.abs(ev.imag) > 1e-6 * np.abs(ev).max()))
        qs = np.linspace(0, q_max, n_grid)
        for i, q in enumerate(qs):
            if cplx(q):
                lo, hi = (qs[i - 1] if i else 0.0), q
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if cplx(mid): hi = mid
                    else: lo = mid
                return float(hi)
        return math.inf

    def margins(self, altitude_rho: float = RHO0, cap: float = MARGIN_CAP) -> Dict:
        """Divergence and flutter margins (x V_D), always finite.

        Search range: quasi-steady p-method EAS sweep 1 kt .. cap*V_D (endpoint included) and steady-aero
        coalescence q = 0 .. cap^2 * q_D; divergence from the exact eigenproblem. Anything not found below the
        cap (no instability at all, or only above the cap) is reported as the cap with a *_not_found_below_cap
        flag. A numerical failure (NaN / LinAlgError) is reported as margin 0 with margin_error=True
        (conservative -> hard fail in margin_terms)."""
        p = self.p
        vd = p.v_dive_keas
        qd = 0.5 * RHO0 * (vd * KT2FPS) ** 2
        v_cap = cap * vd
        err = False

        def fin(v_keas):
            """(capped margin, found_below_cap) for a speed in KEAS that may be inf/NaN."""
            nonlocal err
            if v_keas is None or (isinstance(v_keas, float) and math.isnan(v_keas)) or np.isnan(v_keas):
                err = True
                return 0.0, True
            if not math.isfinite(v_keas) or v_keas >= v_cap:
                return float(cap), False
            return float(v_keas / vd), True

        try:
            fl = self.flutter(v_cap, rho=altitude_rho)
        except (np.linalg.LinAlgError, ValueError):
            fl = {"v_flutter_keas": float("nan"), "f_flutter_hz": float("nan")}
        try:
            q_div = self.divergence_q(altitude_rho)
        except (np.linalg.LinAlgError, ValueError):
            q_div = float("nan")
        try:
            q_co = self.coalescence_q(cap ** 2 * qd, rho=altitude_rho)
        except (np.linalg.LinAlgError, ValueError):
            q_co = float("nan")
        to_v = lambda q: (math.sqrt(2 * q / RHO0) / KT2FPS if (math.isfinite(q) and q >= 0) else q)  # noqa: E731
        m_div, div_found = fin(to_v(q_div))
        m_qs, qs_found = fin(fl["v_flutter_keas"])
        m_co, co_found = fin(to_v(q_co))
        crit = p.flutter_criterion
        if crit == "min":
            m_fl, fl_found = min(m_qs, m_co), (qs_found or co_found)
        elif crit == "coalescence":
            m_fl, fl_found = m_co, co_found
        else:
            m_fl, fl_found = m_qs, qs_found
        f_fl = fl.get("f_flutter_hz", 0.0)
        f_fl = float(f_fl) if (qs_found and math.isfinite(f_fl)) else 0.0
        return {
            "margin_cap": float(cap),
            "q_dive_psf": qd,
            "q_div_psf": float(min(q_div, cap ** 2 * qd)) if math.isfinite(q_div) else float(cap ** 2 * qd),
            "v_div_keas": m_div * vd, "div_margin": m_div, "div_not_found_below_cap": not div_found,
            "v_flutter_keas": m_qs * vd, "f_flutter_hz": f_fl, "flutter_margin_qs": m_qs,
            "v_coalescence_keas": m_co * vd, "coalescence_margin": m_co,
            # screening value used for GA penalties: conservative min of the two estimates, capped
            "flutter_margin": float(m_fl), "flutter_not_found_below_cap": not fl_found,
            "margin_error": err,
            "f_modes_hz": self.frequencies_hz(0.0).tolist(),
        }

    # ---------------- statics / loads ----------------
    def static_tip_deflection_uniform(self, load_per_ft: float) -> float:
        """Tip bending deflection for a uniform spanwise load (per ft), no aero coupling."""
        Q = self.PhiW.T @ (np.full(self.y.size, load_per_ft) * self.dy)
        eta = np.linalg.solve(self.K, Q)
        return self.tip_values(eta)[0]

    def tip_values(self, eta: np.ndarray) -> Tuple[float, float]:
        nb = self.p.n_bend
        w = sum(float(_bend(k, 1.0)) * eta[k] for k in range(nb))
        return w, float(eta[nb])     # tip deflection ft (up +), tip twist rad (nose-up +)


@dataclass
class WingState:
    eta: np.ndarray
    etad: np.ndarray
    etadd: np.ndarray


class FlexCoupler:
    """Couples two FlexWings (left/right) to a running JSBSim FGFDMExec.

    mode: 'oneway' -> structure is driven by JSBSim loads, no feedback;
          'twoway' -> elastic aero increments are fed back via external_reactions.
    reference: 'trim' -> feedback relative to the 1-g trim deformation (default);
               'jig'  -> feedback relative to the undeformed (jig) shape.
    """

    FORCE = "flexwing_F"
    MOMENT = "flexwing_M"

    def __init__(self, wing: FlexWing, mode: str = "twoway", substeps: int = 4, reference: str = "trim"):
        assert mode in ("oneway", "twoway")
        self.w = wing
        self.mode = mode
        self.substeps = substeps
        self.reference = reference
        n = wing.n_modes
        z = np.zeros(n)
        self.st = [WingState(z.copy(), z.copy(), z.copy()) for _ in range(2)]   # 0 = right, 1 = left
        self.eta_ref = [z.copy(), z.copy()]
        self.last = {}

    # --- external loads from the JSBSim state ---
    def read_state(self, fdm) -> Dict[str, float]:
        g = lambda k: fdm[k]
        alpha = g("aero/alpha-rad")
        fbx, fbz = g("forces/fbx-aero-lbs"), g("forces/fbz-aero-lbs")
        d = dict(
            qbar=g("aero/qbar-psf"), vt=max(g("velocities/vt-fps"), 1.0), alpha=alpha, beta=g("aero/beta-rad"),
            kap=self.w.kappa(g("velocities/mach")),
            p=g("velocities/p-rad_sec"), pdot=g("accelerations/pdot-rad_sec2"), nz=g("accelerations/Nz"),
            lift=-fbz * math.cos(alpha) + fbx * math.sin(alpha),
            ail_r=g("fcs/right-aileron-pos-rad"), ail_l=g("fcs/left-aileron-pos-rad"),
        )
        ap = self.w.p.ail_antisym_prop
        nd = self.w.p.ail_norm_deg
        if ap is not None:
            a = g(ap)
            d["ail_r"], d["ail_l"] = -a, a
        elif nd is not None:
            k = math.radians(nd)
            d["ail_r"], d["ail_l"] = g("fcs/right-aileron-pos-norm") * k, g("fcs/left-aileron-pos-norm") * k
        return d

    def external_loads(self, s: Dict[str, float], side: int) -> Tuple[np.ndarray, np.ndarray]:
        """Strip lift (lbf) and pitching moment about EA (ft*lbf) from the rigid aircraft, for one wing."""
        w, p = self.w, self.w.p
        sign = 1.0 if side == 0 else -1.0            # signed lateral position = sign * y
        q, V = s["qbar"], s["vt"]
        Li = 0.5 * p.wing_lift_share * s["lift"] * w.schrenk
        Li = Li + q * s.get("kap", 1.0) * w.lift_w * (sign * s["p"] * w.y / V)              # roll-rate antisymmetric load
        dail = s["ail_r"] if side == 0 else s["ail_l"]
        Lail = q * w.c * w.dy * w.cl_d * dail * w.ail_mask
        Mail = q * w.c ** 2 * w.dy * w.cm_d * dail * w.ail_mask
        Li = Li + Lail
        # inertia + gravity (Nz) relief and roll acceleration, acting at the section CG
        Fin = w.m * w.dy * (-G0 * s["nz"] + sign * s["pdot"] * w.y)
        Lnet = Li + Fin
        Mea = Li * w.e_c + Mail - Fin * w.x_theta
        return Lnet, Mea

    def gen_force(self, Lnet, Mea, s, side):
        w = self.w
        Q = w.PhiW.T @ Lnet + w.PsiT.T @ Mea
        if w.m_tip_slug > 0:
            sign = 1.0 if side == 0 else -1.0
            nb = w.p.n_bend
            ftip = w.m_tip_slug * (-G0 * s["nz"] + sign * s["pdot"] * w.s)
            phi_t = np.array([float(_bend(k, 1.0)) for k in range(nb)] + [0.0])
            psi_t = np.zeros(w.n_modes); psi_t[nb] = 1.0
            d = (w.p.x_tip_mass - w.p.x_ea) * float(w.chord(w.s))
            Q = Q + ftip * phi_t - ftip * d * psi_t
        return Q

    def _elastic_dalpha(self, side, eta, etad, s):
        w = self.w
        sign = 1.0 if side == 0 else -1.0
        de = eta - self.eta_ref[side] if self.reference == "trim" else eta
        da = w.G @ de + (w.H @ etad) / s["vt"]
        da = da + sign * s["beta"] * (w.dPhiW @ de)       # elastic dihedral (right wing sees +beta*w')
        return da

    # --- precomputed linear operators (fast path used every frame) ---
    def _precompute(self):
        w, p = self.w, self.w.p
        nb, n = p.n_bend, w.n_modes
        lw, y = w.lift_w, w.y
        W_in = w.PhiW - w.x_theta[:, None] * w.PsiT               # gen. force per unit inertial force at the CG
        mdy = w.m * w.dy
        ail_L = w.c * w.dy * w.cl_d * w.ail_mask
        ail_M = w.c ** 2 * w.dy * w.cm_d * w.ail_mask
        self.Qs = w.W_lift.T @ (0.5 * p.wing_lift_share * w.schrenk)   # x total aero lift
        self.Qp = w.W_lift.T @ (lw * y)                                # x q*p/V*sign
        self.Qa = w.W_lift.T @ ail_L + w.PsiT.T @ ail_M                # x q*delta_ail
        self.Qn = -G0 * (W_in.T @ mdy)                                 # x Nz
        self.Qpd = W_in.T @ (mdy * y)                                  # x sign*pdot
        phi_t = np.array([float(_bend(k, 1.0)) for k in range(nb)] + [0.0])
        psi_t = np.zeros(n); psi_t[nb] = 1.0
        self.tipvec = phi_t
        if w.m_tip_slug > 0:
            d = (p.x_tip_mass - p.x_ea) * float(w.chord(w.s))
            wt = phi_t - d * psi_t
            self.Qn = self.Qn - G0 * w.m_tip_slug * wt
            self.Qpd = self.Qpd + w.m_tip_slug * w.s * wt
        self.A_beta = w.W_lift.T @ (lw[:, None] * w.dPhiW)
        # feedback rows
        self.sG, self.sH, self.sD = lw @ w.G, lw @ w.H, lw @ w.dPhiW
        ly = lw * y
        self.yG, self.yH, self.yD = ly @ w.G, ly @ w.H, ly @ w.dPhiW
        lx = lw * w.dx_ac
        self.xG, self.xH, self.xD = lx @ w.G, lx @ w.H, lx @ w.dPhiW
        # root bending moment rows (force summation)
        arm = w.arm_root
        self.bm_s = float(arm @ (0.5 * p.wing_lift_share * w.schrenk))
        self.bm_p = float(arm @ (lw * y))
        self.bm_a = float(arm @ ail_L)
        self.bm_n = float(-G0 * (arm @ mdy)) - (G0 * w.m_tip_slug * w.L if w.m_tip_slug > 0 else 0.0)
        self.bm_pd = float(arm @ (mdy * y))
        self.bm_G, self.bm_H = (arm * lw) @ w.G, (arm * lw) @ w.H
        self.bm_acc = (arm * mdy) @ W_in
        self.signs = np.array([1.0, -1.0])
        # stacked rows -> one matmul per state vector in _apply
        e_t = np.zeros(n); e_t[nb] = 1.0
        self.R_de = np.vstack([self.sG, self.sD, self.yG, self.yD, self.xG, self.xD, self.bm_G]).T      # (n, 7)
        self.R_d = np.vstack([self.sH, self.yH, self.xH, self.bm_H]).T                                  # (n, 4)
        self.R_eta = np.vstack([self.tipvec, e_t]).T                                                    # (n, 2)
        self.Qbasis = np.vstack([self.Qs, self.Qn, self.Qp, self.Qa, self.Qpd])                         # (5, n)

    def gen_forces_fast(self, s) -> np.ndarray:
        """(2, n) generalised external forces for right/left wings (same as gen_force(external_loads))."""
        q, V = s["qbar"], s["vt"]
        qp = q * s.get("kap", 1.0) * s["p"] / V
        coef = np.array([[s["lift"], s["nz"], qp, q * s["ail_r"], s["pdot"]],
                         [s["lift"], s["nz"], -qp, q * s["ail_l"], -s["pdot"]]])
        return coef @ self.Qbasis

    def initialize(self, fdm):
        """Static equilibrium at the current (trimmed) state; zero feedback at t0."""
        if not hasattr(self, "Qs"):
            self._precompute()
        s = self.read_state(fdm)
        w = self.w
        Q = self.gen_forces_fast(s)
        Ke = w.K - s["qbar"] * s["kap"] * w.A_K if self.reference == "jig" else w.K
        eta = np.linalg.solve(Ke, Q.T).T
        self.eta = eta.copy()
        self.etad = np.zeros_like(eta)
        self.etadd = np.zeros_like(eta)
        self.eta_ref_arr = eta.copy() if self.reference == "trim" else np.zeros_like(eta)
        self.eta_ref = [self.eta_ref_arr[0], self.eta_ref_arr[1]]
        self.Qref_per_q = self.eta_ref_arr @ w.A_K.T                  # (2, n), times qbar each frame
        self._sync_states()
        self._apply(fdm, s)

    def _sync_states(self):
        self.st = [WingState(self.eta[i], self.etad[i], self.etadd[i]) for i in range(2)]

    def step(self, fdm, dt: float):
        """Advance both wings over one JSBSim frame (Newmark, sub-stepped) and set the feedback."""
        s = self.read_state(fdm)
        w = self.w
        q, V = s["qbar"], s["vt"]
        h = dt / self.substeps
        c1, c2, c3 = 4 / h ** 2, 4 / h, 2 / h
        qk = q * s["kap"]                                             # compressibility on circulatory terms
        Ce = w.C - (q / V) * (s["kap"] * w.A_C_circ + w.A_C_nc)
        Keff_inv = _inv(w.K - qk * w.A_K + c3 * Ce + c1 * w.M)
        KMC = Keff_inv @ np.hstack([w.M, Ce])                         # (n, 2n)
        de = self.eta - self.eta_ref_arr
        Q = self.gen_forces_fast(s) - qk * self.Qref_per_q \
            + (qk * s["beta"]) * (self.signs[:, None] * (de @ self.A_beta.T))   # elastic dihedral (explicit)
        KQ = Keff_inv @ Q.T                                           # (n, 2)
        x, v, a = self.eta.T, self.etad.T, self.etadd.T               # (n, 2)
        for _ in range(self.substeps):
            x1 = KQ + KMC @ np.vstack([c1 * x + c2 * v + a, c3 * x + v])
            v1 = c3 * (x1 - x) - v
            a = c1 * (x1 - x) - c2 * v - a
            x, v = x1, v1
        self.eta, self.etad, self.etadd = x.T.copy(), v.T.copy(), a.T.copy()
        self._apply(fdm, s)

    def _apply(self, fdm, s):
        q, V, beta = s["qbar"] * s["kap"], s["vt"], s["beta"]       # q here = compressible circulatory q
        P = ((self.eta - self.eta_ref_arr) @ self.R_de).tolist()     # [sG, sD, yG, yD, xG, xD, bmG] per wing
        Pd = (self.etad @ self.R_d).tolist()                          # [sH, yH, xH, bmH]
        Pe = (self.eta @ self.R_eta).tolist()                         # [tip w, tip twist]
        Pa = (self.etadd @ self.bm_acc).tolist()
        Ltot = Mroll = Mpitch = 0.0
        bm = [0.0, 0.0]
        qpV = q * s["p"] / V
        ail = (s["ail_r"], s["ail_l"])
        for i, sg in ((0, 1.0), (1, -1.0)):
            g = P[i]; d = Pd[i]
            Ltot += q * (g[0] + d[0] / V + sg * beta * g[1])
            Mroll += -sg * q * (g[2] + d[1] / V + sg * beta * g[3])
            Mpitch += -q * (g[4] + d[2] / V + sg * beta * g[5])
            bm[i] = (s["lift"] * self.bm_s + sg * qpV * self.bm_p + s["qbar"] * ail[i] * self.bm_a + s["nz"] * self.bm_n
                     + sg * s["pdot"] * self.bm_pd + q * (g[6] + d[3] / V) - Pa[i])
        r2d = 180.0 / math.pi
        self.last = {"tip_w_ft_R": Pe[0][0], "tip_w_ft_L": Pe[1][0],
                     "tip_twist_deg_R": Pe[0][1] * r2d, "tip_twist_deg_L": Pe[1][1] * r2d,
                     "root_bm_lbft_R": bm[0], "root_bm_lbft_L": bm[1],
                     "dL_lbf": Ltot, "dRoll_lbft": Mroll, "dPitch_lbft": Mpitch}
        if self.mode == "twoway":
            a = s["alpha"]
            F, Mo = self.FORCE, self.MOMENT
            fdm[f"external_reactions/{F}/magnitude"] = 1.0
            fdm[f"external_reactions/{F}/x"] = Ltot * math.sin(a)
            fdm[f"external_reactions/{F}/z"] = -Ltot * math.cos(a)
            fdm[f"external_reactions/{Mo}/magnitude-lbsft"] = 1.0
            fdm[f"external_reactions/{Mo}/l"] = Mroll
            fdm[f"external_reactions/{Mo}/m"] = Mpitch


def _inv(A: np.ndarray) -> np.ndarray:
    """Inverse; closed-form for 3x3 (avoids LAPACK call overhead in the per-frame loop)."""
    if A.shape != (3, 3):
        return np.linalg.inv(A)
    (a, b, c), (d, e, f), (g, h, i) = A.tolist()
    A1, B1, C1 = e * i - f * h, -(d * i - f * g), d * h - e * g
    det = a * A1 + b * B1 + c * C1
    return np.array([[A1, -(b * i - c * h), b * f - c * e],
                     [B1, a * i - c * g, -(a * f - c * d)],
                     [C1, -(a * h - b * g), a * e - b * d]]) / det


# ---------------- generic helpers (also used by tests) ----------------
def coalescence_q_matrices(M: np.ndarray, K: np.ndarray, A: np.ndarray, q_max: float, n_grid: int = 400) -> float:
    """First q in [0, q_max] where eig(M^-1 (K - q A)) become complex (steady-aero flutter), inf if none."""
    Minv = np.linalg.inv(M)
    def cplx(q):
        ev = np.linalg.eigvals(Minv @ (K - q * A))
        return bool(np.any(np.abs(ev.imag) > 1e-6 * np.abs(ev).max()))
    qs = np.linspace(0, q_max, n_grid)
    for i, q in enumerate(qs):
        if cplx(q):
            lo, hi = (qs[i - 1] if i else 0.0), q
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if cplx(mid): hi = mid
                else: lo = mid
            return float(hi)
    return math.inf


def newmark(M, C, K, Q, dt, x0=None, v0=None):
    """Newmark average-acceleration integration of M x'' + C x' + K x = Q(t).

    Q: array (n_t, n) of generalised forces at each time sample. Returns x, v, a arrays (n_t, n).
    """
    Q = np.asarray(Q, float)
    nt, n = Q.shape
    x = np.zeros((nt, n)); v = np.zeros((nt, n)); a = np.zeros((nt, n))
    x[0] = 0 if x0 is None else x0
    v[0] = 0 if v0 is None else v0
    a[0] = np.linalg.solve(M, Q[0] - C @ v[0] - K @ x[0])
    Keff_inv = np.linalg.inv(K + (2 / dt) * C + (4 / dt ** 2) * M)
    for k in range(nt - 1):
        rhs = Q[k + 1] + M @ ((4 / dt ** 2) * x[k] + (4 / dt) * v[k] + a[k]) + C @ ((2 / dt) * x[k] + v[k])
        x[k + 1] = Keff_inv @ rhs
        v[k + 1] = (2 / dt) * (x[k + 1] - x[k]) - v[k]
        a[k + 1] = (4 / dt ** 2) * (x[k + 1] - x[k]) - (4 / dt) * v[k] - a[k]
    return x, v, a


def modal_response(wing: "FlexWing", lift_hist, dt: float, moment_hist=None, qbar: float = 0.0, v_fps: float = math.inf):
    """Modal response of one semi-span to a strip load history.

    lift_hist: (n_t, n_strips) strip lift (lbf, up +, acting at the quarter chord);
    moment_hist: optional (n_t, n_strips) extra pitching moment about the EA (ft*lbf).
    qbar / v_fps: flight condition for the aeroelastic stiffness/damping (0 -> in vacuo).
    Returns dict with eta, tip deflection (ft), tip twist (deg), root bending moment (lbf*ft, force summation).
    """
    L = np.atleast_2d(np.asarray(lift_hist, float))
    Mh = np.zeros_like(L) if moment_hist is None else np.atleast_2d(np.asarray(moment_hist, float))
    Q = L @ wing.W_lift + Mh @ wing.PsiT
    Ke = wing.K - qbar * wing.A_K
    Ce = wing.C - (qbar / v_fps if math.isfinite(v_fps) and v_fps > 0 else 0.0) * wing.A_C
    x, v, a = newmark(wing.M, Ce, Ke, Q, dt)
    nb = wing.p.n_bend
    tip_w = x[:, :nb] @ np.array([float(_bend(k, 1.0)) for k in range(nb)])
    acc = a @ wing.PhiW.T - (a @ wing.PsiT.T) * wing.x_theta
    aero_el = qbar * ((x @ wing.G.T) + (0 if not math.isfinite(v_fps) else (v @ wing.H.T) / v_fps)) * wing.lift_w
    f = L + aero_el - wing.m * wing.dy * acc
    return {"eta": x, "etad": v, "tip_w_ft": tip_w, "tip_twist_deg": np.degrees(x[:, nb]),
            "root_bm_lbft": f @ wing.arm_root}

# ---------------- JSBSim aircraft preparation ----------------
EXTERNAL_BLOCK = """
  <!-- flexwing.py: elastic aero increments (set every frame from Python) -->
  <force name="{F}" frame="BODY">
   <location unit="IN"> <x> {x} </x> <y> {y} </y> <z> {z} </z> </location>
   <direction> <x> 0 </x> <y> 0 </y> <z> 0 </z> </direction>
  </force>
  <moment name="{M}" frame="BODY">
   <direction> <x> 0 </x> <y> 0 </y> <z> 0 </z> </direction>
  </moment>
"""


PREPARE_FMT = 3   # 3: network <input>/<output> sockets stripped; f16 placeholder-first point masses

# Network I/O declared in stock aircraft files (737: telnet <input port="5137"/> and QTJSBSIM UDP <input port="5139">,
# both bound on 0.0.0.0 at run_ic and able to set fcs/* and simulation/terminate; c172x: commented SOCKET outputs).
# prepare_aircraft removes every such element (live or inside comments). File outputs (type="CSV") are kept; new_fdm
# calls disable_output() anyway.
_NET_TYPES = r"(?:SOCKET|TCP|UDP|QTJSBSIM|FLIGHTGEAR|FLIGHTGEAR\w*)"
_NET_IO_RES = (
    # commented-out form used by the stock 737: <!--output ... type="SOCKET" port=...> ... </output-->
    re.compile(r"<!--\s*(?:input|output)\b[^>]*\b(?:port\s*=|type\s*=\s*\"" + _NET_TYPES + r"\")[^>]*>.*?</(?:input|output)\s*-->",
               re.S | re.I),
    re.compile(r"<input\b[^>]*\bport\s*=[^>]*/>", re.S),
    re.compile(r"<input\b[^>]*\bport\s*=[^>]*>.*?</input\s*>", re.S),
    re.compile(r"<output\b[^>]*\b(?:port\s*=|type\s*=\s*\"" + _NET_TYPES + r"\")[^>]*/>", re.S | re.I),
    re.compile(r"<output\b[^>]*\b(?:port\s*=|type\s*=\s*\"" + _NET_TYPES + r"\")[^>]*>.*?</output\s*>", re.S | re.I),
)
NET_IO_GREP = r'port\s*=|type\s*=\s*"(SOCKET|TCP|UDP|QTJSBSIM|FLIGHTGEAR)|protocol\s*='   # grep -E pattern used in the check
# Models that get a zero-weight centreline payload placeholder as point mass index 0 (genome/sim_ext.py payload_index 0).
# For f16 this moves the stock "Pilot" point mass from index 0 to index 1.
PAYLOAD_PLACEHOLDER_MODELS = ("737", "T38", "f16")


def strip_network_io(txt: str) -> Tuple[str, int]:
    """Remove network <input>/<output> socket declarations. Returns (text, number removed)."""
    n = 0
    for rx in _NET_IO_RES:
        txt, k = rx.subn("", txt)
        n += k
    if n:   # orphaned labels of the removed blocks (737: "this is the telnet interface", "N properties input/output ...")
        txt = re.sub(r"<!--\s*(?:this is the telnet interface|\d+ properties (?:input|output)[^>]*?)\s*-->", "", txt, flags=re.I)
    return txt, n


def jsbsim_data_dir() -> str:
    import jsbsim
    return os.path.dirname(jsbsim.__file__)


def prepare_aircraft(model: str, root: str) -> str:
    """Copy aircraft/<model> into a local JSBSim root and add the flexwing external reactions.

    engine/ and systems/ are symlinked read-only to the installed package data; the
    installed data is never modified. Returns the root dir (pass to FGFDMExec(root)).
    """
    src_root = jsbsim_data_dir()
    os.makedirs(os.path.join(root, "aircraft"), exist_ok=True)
    for d in ("engine", "systems"):
        link = os.path.join(root, d)
        if not os.path.exists(link):
            os.symlink(os.path.join(src_root, d), link)
    dst = os.path.join(root, "aircraft", model)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(src_root, "aircraft", model), dst)
    xml = os.path.join(dst, model + ".xml")
    txt = open(xml, encoding="utf-8", errors="replace").read()
    if "flexwing_F" in txt:
        return root
    txt, n_net = strip_network_io(txt)
    for extra in os.listdir(dst):   # any other XML in the copied dir (init files, systems) gets the same treatment
        fp = os.path.join(dst, extra)
        if extra.endswith(".xml") and extra != model + ".xml" and os.path.isfile(fp):
            t2, k = strip_network_io(open(fp, encoding="utf-8", errors="replace").read())
            if k:
                open(fp, "w", encoding="utf-8").write(t2)
                n_net += k
    m = re.search(r'<location\s+name="AERORP"\s+unit="(\w+)"\s*>(.*?)</location>', txt, re.S)
    if not m:
        raise RuntimeError(f"{model}: AERORP not found in metrics")
    unit = m.group(1).upper()
    vals = [float(re.search(rf"<{a}>\s*([-+0-9.eE]+)\s*</{a}>", m.group(2)).group(1)) for a in "xyz"]
    scale = {"IN": 1.0, "FT": 12.0, "M": 39.3700787}[unit]
    x, y, z = (v * scale for v in vals)
    block = EXTERNAL_BLOCK.format(F=FlexCoupler.FORCE, M=FlexCoupler.MOMENT, x=x, y=y, z=z)
    # Two point masses (right/left wing) carrying the gene-driven wing-mass change; weight and Y are set at
    # runtime through inertia/pointmass-weight-lbs[i] / inertia/pointmass-location-Y-inches[i].
    meta = {"fmt": PREPARE_FMT, "aerorp_in": [x, y, z], "pm_index_R": None, "pm_index_L": None,
            "network_io_removed": n_net}
    mb = re.search(r"<mass_balance[^>]*>(.*?)</mass_balance>", txt, re.S)
    if mb:
        n_pm = len(re.findall(r"<pointmass\b", mb.group(1)))
        pm = ""
        if n_pm == 0 or model in PAYLOAD_PLACEHOLDER_MODELS:
            # 737/T38 (no point masses) and f16 (stock "Pilot" at index 0): add a zero-weight centreline placeholder at
            # the CG as index 0, so genome/sim_ext.py's payload point mass (payload_index 0) is on the centreline and
            # never lands on a wing point mass (or the pilot).
            cg = re.search(r"<location\s+name=\"CG\"\s+unit=\"(\w+)\"\s*>(.*?)</location>", mb.group(1), re.S)
            cxyz = [x, 0.0, z]
            if cg:
                cs_ = {"IN": 1.0, "FT": 12.0, "M": 39.3700787}[cg.group(1).upper()]
                cxyz = [float(re.search(rf"<{a}>\s*([-+0-9.eE]+)\s*</{a}>", cg.group(2)).group(1)) * cs_ for a in "xyz"]
            ph = f"""
        <pointmass name="payload_placeholder">
            <weight unit="LBS"> 0.0 </weight>
            <location unit="IN"> <x> {cxyz[0]} </x> <y> 0.0 </y> <z> {cxyz[2]} </z> </location>
        </pointmass>"""
            meta["payload_placeholder_index"] = 0
            if n_pm == 0:
                pm += ph
            else:   # insert before the first existing point mass so it becomes index 0
                first = mb.start(1) + re.search(r"<pointmass\b", mb.group(1)).start()
                txt = txt[:first] + ph.lstrip("\n").lstrip() + "\n  " + txt[first:]
                meta["stock_pointmass_index_shift"] = 1
                mb = re.search(r"<mass_balance[^>]*>(.*?)</mass_balance>", txt, re.S)
            n_pm += 1
        pm += "".join(f"""
        <pointmass name="flexwing_dm_{s}">
            <weight unit="LBS"> 0.0 </weight>
            <location unit="IN"> <x> {x} </x> <y> 0.0 </y> <z> {z} </z> </location>
        </pointmass>""" for s in "RL")
        txt = txt[:mb.end(1)] + pm + "\n    " + txt[mb.end(1):]
        meta.update(pm_index_R=n_pm, pm_index_L=n_pm + 1)
    import json as _json
    _json.dump(meta, open(os.path.join(dst, "flexwing_meta.json"), "w"))
    if re.search(r"<external_reactions\s*/>", txt):
        txt = re.sub(r"<external_reactions\s*/>", "<external_reactions>" + block + "</external_reactions>", txt, count=1)
    elif "</external_reactions>" in txt:
        txt = txt.replace("</external_reactions>", block + " </external_reactions>", 1)
    else:
        txt = txt.replace("</fdm_config>", " <external_reactions>" + block + " </external_reactions>\n</fdm_config>", 1)
    open(xml, "w", encoding="utf-8").write(txt)
    return root


def apply_wing_mass(fdm, model: str, root: str, wing: "FlexWing") -> float:
    """Push the wing-mass change (genes) into JSBSim weight & inertia via the two injected point masses.

    Call BEFORE run_ic/trim. Each point mass = semi-wing mass delta (lb, may be negative) at the semi-wing
    mass centroid (spanwise) and AERORP x/z. Returns the total delta (lb)."""
    import json as _json
    meta_p = os.path.join(root, "aircraft", model, "flexwing_meta.json")
    meta = _json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    dm = wing.m_semi_total_lb - wing.p.wing_mass_lb / 2
    if meta.get("pm_index_R") is None:
        return 0.0
    m_lb = wing.m * wing.dy * G0
    y_c = float(np.sum(m_lb * wing.y) + wing.m_tip_slug * G0 * wing.s) / float(np.sum(m_lb) + wing.m_tip_slug * G0)
    for idx, sgn in ((meta["pm_index_R"], 1.0), (meta["pm_index_L"], -1.0)):
        fdm[f"inertia/pointmass-weight-lbs[{idx}]"] = dm
        fdm[f"inertia/pointmass-location-Y-inches[{idx}]"] = sgn * y_c * 12.0
    return 2 * dm


def new_fdm(model: str, root: Optional[str], dt: float = 1 / 120):
    import jsbsim
    jsbsim.FGJSBBase().debug_lvl = 0
    fdm = jsbsim.FGFDMExec(root)
    fdm.set_debug_level(0)
    out = os.path.join(tempfile.gettempdir(), "flexwing_jsbsim")
    os.makedirs(out, exist_ok=True)
    fdm.set_output_path(out)
    fdm.load_model(model)
    fdm.disable_output()
    fdm.set_dt(dt)
    return fdm


def wing_from_fdm(fdm, model: str, **overrides) -> FlexWing:
    p = params_for(model, fdm["metrics/bw-ft"], fdm["metrics/Sw-sqft"], fdm["inertia/empty-weight-lbs"], **overrides)
    return FlexWing(p)


# ---------------- GA genes and structural fitness terms ----------------
@dataclass(frozen=True)
class StructGene:
    name: str
    min: float
    max: float
    kind: str
    units: str
    doc: str

    def decode(self, v: float) -> float:
        v = min(max(float(v), 0.0), 1.0)
        return self.min * (self.max / self.min) ** v if self.kind == "log" else self.min + v * (self.max - self.min)


STRUCT_SCHEMA: List[StructGene] = [
    # Phase-1 gene set (matches INTERFACE.md section 1a). Chordwise mass/axis placement (x_ea, x_cg, tip_mass_frac)
    # is FIXED per aircraft in Phase 1. Bending and torsion stiffness are TIED (stiffness_scale x torsion_bend_ratio)
    # to keep the GA off the swept-wing torsion/2nd-bending crossing ridge (INTERFACE.md section 3).
    StructGene("stiffness_scale", 0.6, 2.0, "log", "-", "wing stiffness multiplier: EI x = stiffness_scale (also scales structural mass and allowable root moment)"),
    StructGene("torsion_bend_ratio", 0.8, 1.15, "linear", "-", "GJ multiplier / EI multiplier: GJ x = stiffness_scale * torsion_bend_ratio"),
    StructGene("zeta", 0.005, 0.05, "log", "-", "structural modal damping ratio"),
    StructGene("nonstruct_scale", 0.8, 1.25, "log", "-", "non-structural wing mass multiplier (fuel/systems/secondary structure)"),
]


def tied_stiffness(stiffness_scale: float, torsion_bend_ratio: float) -> Dict[str, float]:
    """Tied genes -> internal multipliers: ei_scale = stiffness_scale, gj_scale = stiffness_scale * torsion_bend_ratio."""
    return {"ei_scale": float(stiffness_scale), "gj_scale": float(stiffness_scale) * float(torsion_bend_ratio)}


def genes_to_overrides(genes: Dict[str, float]) -> Dict[str, float]:
    """STRUCT_SCHEMA gene values (decode_struct output) -> WingParams overrides (internal ei/gj kept)."""
    out = {k: float(v) for k, v in genes.items() if k not in ("stiffness_scale", "torsion_bend_ratio")}
    out.update(tied_stiffness(genes.get("stiffness_scale", 1.0), genes.get("torsion_bend_ratio", 1.0)))
    return out


def decode_struct(genome) -> Dict[str, float]:
    return {g.name: g.decode(v) for g, v in zip(STRUCT_SCHEMA, genome)}


@dataclass
class StructWeights:
    w_bm_rms: float = 0.25       # RMS of root-bending-moment fluctuation / 1-g root moment
    w_bm_peak: float = 2.0       # hinge^2 on peak root moment above limit (n_limit * M_1g * ei_scale)
    w_tip: float = 1.0           # hinge^2 on tip deflection above limit
    w_twist: float = 1.0         # hinge^2 on elastic twist above limit
    w_flutter: float = 1.0       # hinge^2 on flutter margin below 1.2 V_D (hard fail below 1.0)
    w_div: float = 1.0           # same for divergence
    w_mass: float = 0.3          # relative wing-mass change
    fail_cost: float = 2000.0    # = sim.FAIL_BASE * 2 (worst possible altitude-hold cost)
    margin_req: float = 1.20     # CS-23/25.629 style: aeroelastically stable up to 1.2 V_D


def margin_terms(wing: FlexWing, wts: StructWeights = StructWeights(), rho: float = RHO0) -> Dict:
    """Pre-simulation terms (cheap): margins and mass. 'fail' -> skip the flight sim."""
    m = wing.margins(rho)
    terms = {}
    fail = None
    for key, wk in (("flutter_margin", wts.w_flutter), ("div_margin", wts.w_div)):
        mg = m[key]
        if mg < 1.0:
            fail = fail or key.replace("_margin", "")
        terms["J_" + key] = wk * max(0.0, (wts.margin_req - mg) / (wts.margin_req - 1.0)) ** 2
    p = wing.p
    m0 = p.wing_mass_lb / 2
    terms["J_mass"] = wts.w_mass * (wing.m_semi_total_lb - m0) / m0
    terms["delta_wing_mass_lb"] = 2 * (wing.m_semi_total_lb - m0)
    return {"margins": m, "terms": terms, "fail": fail}


def response_terms(hist: Dict[str, np.ndarray], wing: FlexWing, m_root_1g: float, wts: StructWeights = StructWeights()) -> Dict:
    """Post-simulation terms from time histories of root BM, tip deflection and twist (both wings)."""
    p = wing.p
    bm = np.concatenate([hist["root_bm_lbft_R"], hist["root_bm_lbft_L"]])
    bm_rms = float(np.sqrt(np.mean((bm - m_root_1g) ** 2)))
    bm_allow = p.n_limit * m_root_1g * p.ei_scale
    peak = float(np.max(np.abs(bm)))
    tip = float(np.max(np.abs(np.concatenate([hist["tip_w_ft_R"], hist["tip_w_ft_L"]]))))
    tw = float(np.max(np.abs(np.concatenate([hist["tip_twist_deg_R"], hist["tip_twist_deg_L"]]))))
    w_lim = p.tip_defl_limit_frac * wing.s
    terms = {
        "J_bm_rms": wts.w_bm_rms * bm_rms / m_root_1g,
        "J_bm_peak": wts.w_bm_peak * max(0.0, peak / bm_allow - 1) ** 2,
        "J_tip": wts.w_tip * max(0.0, tip / w_lim - 1) ** 2,
        "J_twist": wts.w_twist * max(0.0, tw / p.twist_limit_deg - 1) ** 2,
    }
    fail = "structural_ultimate" if peak > 1.5 * bm_allow else None
    return {"terms": terms, "fail": fail, "bm_rms": bm_rms, "bm_peak": peak, "bm_allow": bm_allow,
            "tip_max_ft": tip, "twist_max_deg": tw}


# ---------------- interface to genome/ (Genome Architect schema) ----------------
# genome/ `structure` block gene name -> WingParams field. See INTERFACE.md.
# Genome (genome/ schema) names -> (internal gene name, allowed range). Inputs outside the range raise.
GENOME_GENES = {
    "stiffness_scale": ("stiffness_scale", 0.6, 2.0),
    "torsion_bend_ratio": ("torsion_bend_ratio", 0.8, 1.15),
    "struct_damping_ratio": ("zeta", 0.005, 0.05),
    "nonstructural_mass_scale": ("nonstruct_scale", 0.8, 1.25),
    "wing_mass_scale": ("nonstruct_scale", 0.8, 1.25),       # provisional name, alias of nonstructural_mass_scale
}
GENOME_ALIASES = {k: v[0] for k, v in GENOME_GENES.items()}
NOT_MODELLED = ("mass_centroid_shift", "aspect_ratio_delta", "sweep_delta_deg")
# Fixed per aircraft in Phase 1; passing them from a genome always raises (exploit guard).
FIXED_PHASE1 = ("elastic_axis_frac", "section_cg_frac", "x_ea", "x_cg", "cal_x_ea", "cal_x_cg",
                "tip_mass_frac", "x_tip_mass", "mass_balance_frac")
# Independent stiffness keys are replaced by the tied pair; passing them from a genome always raises.
UNTIED_STIFFNESS = ("bend_stiffness_scale", "torsion_stiffness_scale", "ei_scale", "gj_scale")


def overrides_from_genome(decoded: Dict[str, float], strict: bool = False) -> Dict[str, float]:
    """Map a decoded genome/ structure block to WingParams overrides.

    Raises ValueError for: fixed Phase-1 genes, independent bend/torsion keys, values outside the gene ranges,
    duplicate aliases with different values, and raw WingParams field names (the genome path is not a config
    back door). NOT_MODELLED genes are ignored when 0 and raise when non-zero only if strict."""
    genes: Dict[str, float] = {}
    for k, v in decoded.items():
        if k in FIXED_PHASE1:
            raise ValueError(f"gene {k} is fixed per aircraft in Phase 1 (elastic axis / section CG / tip mass are not genes; see INTERFACE.md)")
        if k in UNTIED_STIFFNESS:
            raise ValueError(f"gene {k} is replaced by the tied pair stiffness_scale / torsion_bend_ratio (see INTERFACE.md)")
        if k in GENOME_GENES:
            name, lo, hi = GENOME_GENES[k]
            x = float(v)
            if not (math.isfinite(x) and lo - 1e-9 <= x <= hi + 1e-9):
                raise ValueError(f"gene {k}={v} outside its range [{lo}, {hi}]")
            if name in genes and abs(genes[name] - x) > 1e-12:
                raise ValueError(f"gene {k} conflicts with another alias of {name}")
            genes[name] = x
        elif k in NOT_MODELLED:
            if strict and abs(float(v)) > 1e-12:
                raise ValueError(f"gene {k} is not modelled by flexwing (see INTERFACE.md)")
        elif k in WingParams.__dataclass_fields__:
            raise ValueError(f"{k} is a WingParams field, not a genome gene (use params_for overrides in code)")
        elif strict:
            raise ValueError(f"unknown structure gene {k}")
    return genes_to_overrides(genes)


def telemetry_channels(hist: Dict[str, np.ndarray], wing: "FlexWing", m_root_1g: float) -> Dict:
    """Channels/params named as genome/fitness.py expects (obj_structural) plus extras.

    wing_root_bending [lbf*ft]: per sample, the root moment of the more-loaded wing (signed, up-bending +).
    wing_root_bending_limit [lbf*ft]: n_limit * M_root_1g(trim) * ei_scale (limit load; ultimate = 1.5x).
    """
    R, L = np.asarray(hist["root_bm_lbft_R"]), np.asarray(hist["root_bm_lbft_L"])
    wrb = np.where(np.abs(R) >= np.abs(L), R, L)
    tw = np.where(np.abs(hist["tip_twist_deg_R"]) >= np.abs(hist["tip_twist_deg_L"]), hist["tip_twist_deg_R"], hist["tip_twist_deg_L"])
    tip = np.where(np.abs(hist["tip_w_ft_R"]) >= np.abs(hist["tip_w_ft_L"]), hist["tip_w_ft_R"], hist["tip_w_ft_L"])
    return {
        "wing_root_bending": wrb, "wing_root_bending_R": R, "wing_root_bending_L": L,
        "tip_deflection": tip, "tip_twist": tw,
        "params": {"wing_root_bending_limit": wing.p.n_limit * m_root_1g * wing.p.ei_scale,
                   "wing_root_bending_1g": m_root_1g,
                   "tip_deflection_limit": wing.p.tip_defl_limit_frac * wing.s,
                   "tip_twist_limit": wing.p.twist_limit_deg,
                   "wing_mass_lb": 2 * wing.m_semi_total_lb,
                   "wing_mass_delta_lb": 2 * wing.m_semi_total_lb - wing.p.wing_mass_lb,
                   "wing_mass_frac_delta": (2 * wing.m_semi_total_lb - wing.p.wing_mass_lb) / wing.p.wing_mass_lb},
    }
