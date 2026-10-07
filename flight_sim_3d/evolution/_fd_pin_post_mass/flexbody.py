"""flexbody.py -- flex v2: distributed-stiffness beam FE wings + flexible empennage + fuselage bending.

Built ALONGSIDE flexwing.py (v1). Nothing here modifies v1: flexwing.py, coupled_sim.py, the v1 4-gene path and the
jsbsim_root prepared copies are untouched. v2 uses its own prepared root (jsbsim_root_v2 = v1 preparation + three
zero-weight point masses for tail / fin / aft-fuselage mass changes) and the same external_reactions force/moment
names (flexwing_F / flexwing_M), now writing all three force and all three moment components.

Model (per aircraft, all structural numbers notional -- no GVT data; see INTERFACE_v2.md / AEROELASTIC_DESIGN.md):
  bodies   wingR, wingL      Euler-Bernoulli bending (w), St Venant torsion (theta), in-plane bending (v); 32 elements
           htR, htL          bending + torsion, 12 elements (all-moving tail for T38/f16)
           vt                bending + torsion, 12 elements (lateral "lift" = side force)
           fusV, fusL        aft-fuselage vertical / lateral bending, 12 elements, clamped at the wing, tail mass at tip
  FE       Hermite cubic (w, v), linear (theta), consistent mass incl. the -m*x_theta w/theta coupling; clamped root
  modes    per body: lowest n_b bending / n_t torsion / n_ip in-plane modes (classified by strain-energy fraction);
           defaults wing 3b+2t+1ip, HT 2b+1t, VT 2b+1t, fuselage 2+2 -> 25 modal DOF
  aero     strip theory as v1 (DATCOM CLa, kappa(M) per surface, quasi-steady 3/4-chord + Theodorsen apparent mass);
           tail strips see fuselage slope/plunge (d_alpha = -w'_f,tip - w_f,tip_dot/V), fin likewise with fusL
  coupling Newmark (average acceleration) on the 25-DOF modal system every JSBSim frame, relative to the 1-g trim
           shape; elastic aero increments fed back as body-axis F x/y/z and M l/m/n at AERORP.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import flexwing as fw

MODEL_VERSION_TAG = "flexv2"
MARGIN_CAP = fw.MARGIN_CAP          # 3.0
G0, LB2SLUG, RHO0, KT2FPS = fw.G0, fw.LB2SLUG, fw.RHO0, fw.KT2FPS
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_V2 = os.path.join(HERE, "jsbsim_root_v2")
PREPARE_FMT_V2 = 1
# Minimum-gauge mass floor (2026-10-06, mass-exploit fix): the stiffness-sized (structural) mass share of every body
# scales as f_min + (1 - f_min) * s**k with its stiffness multiplier s, so lowering stiffness only removes the
# non-minimum-gauge part (skins/webs cannot go below manufacturing / handling / damage-tolerance gauges). s = 1 -> 1.
MIN_GAUGE = {"f_min": 0.5, "k": 1.0}


def gauge_mass_factor(s):
    """Structural-mass factor of a stiffness multiplier s (array or float): f_min + (1 - f_min) * s**k (= 1 at s = 1)."""
    return MIN_GAUGE["f_min"] + (1.0 - MIN_GAUGE["f_min"]) * np.asarray(s, float) ** MIN_GAUGE["k"]


# =====================================================================================================================
# 1. Beam finite elements
# =====================================================================================================================
def _elem_bend(h: float):
    """Hermite cubic beam element: stiffness per unit EI and consistent mass per unit m (dofs w1, w1', w2, w2')."""
    k = np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                  [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]]) / h ** 3
    m = np.array([[156, 22 * h, 54, -13 * h], [22 * h, 4 * h * h, 13 * h, -3 * h * h],
                  [54, 13 * h, 156, -22 * h], [-13 * h, -3 * h * h, -22 * h, 4 * h * h]]) * h / 420
    return k, m


def _elem_coupling(h: float):
    """C = int_0^h N_w^T N_theta dx (4x2) for Hermite w and linear theta (closed form; checked by Gauss quadrature in tests)."""
    return h * np.array([[7 / 20, 3 / 20], [h / 20, h / 30], [3 / 20, 7 / 20], [-h / 30, -h / 20]])


class Beam:
    """Clamped-free beam FE. Fields per node: w, w' (always), theta (if GJ given), v, v' (if EIv given).

    Section properties are per element (constant within an element). x_theta > 0: section CG aft of the EA.
    tip_mass_slug: lumped mass at the free end on w (and v)."""

    def __init__(self, L: float, EI, m, GJ=None, Ia=None, mxt=None, EIv=None, tip_mass_slug: float = 0.0):
        EI = np.asarray(EI, float)
        n = EI.size
        self.L, self.n_el, self.h = float(L), n, float(L) / n
        self.has_t, self.has_v = GJ is not None, EIv is not None
        f = ["w", "wp"] + (["t"] if self.has_t else []) + (["v", "vp"] if self.has_v else [])
        self.fields = f
        nd = len(f)
        self.nd = nd
        N = (n + 1) * nd
        K = np.zeros((N, N)); M = np.zeros((N, N))
        kb, mb = _elem_bend(self.h)
        Cwt = _elem_coupling(self.h)
        h = self.h
        g = lambda node, fld: node * nd + f.index(fld)  # noqa: E731
        m = np.broadcast_to(np.asarray(m, float), (n,))
        Kp = {"w": np.zeros((N, N)), "t": np.zeros((N, N)), "v": np.zeros((N, N))}
        for e in range(n):
            iw = [g(e, "w"), g(e, "wp"), g(e + 1, "w"), g(e + 1, "wp")]
            K[np.ix_(iw, iw)] += EI[e] * kb
            Kp["w"][np.ix_(iw, iw)] += EI[e] * kb
            M[np.ix_(iw, iw)] += m[e] * mb
            if self.has_t:
                it = [g(e, "t"), g(e + 1, "t")]
                kt = GJ[e] / h * np.array([[1, -1], [-1, 1]])
                K[np.ix_(it, it)] += kt
                Kp["t"][np.ix_(it, it)] += kt
                M[np.ix_(it, it)] += Ia[e] * h / 6 * np.array([[2, 1], [1, 2]])
                if mxt is not None and mxt[e] != 0.0:
                    c = -mxt[e] * Cwt
                    M[np.ix_(iw, it)] += c
                    M[np.ix_(it, iw)] += c.T
            if self.has_v:
                iv = [g(e, "v"), g(e, "vp"), g(e + 1, "v"), g(e + 1, "vp")]
                K[np.ix_(iv, iv)] += EIv[e] * kb
                Kp["v"][np.ix_(iv, iv)] += EIv[e] * kb
                M[np.ix_(iv, iv)] += m[e] * mb
        if tip_mass_slug:
            M[g(n, "w"), g(n, "w")] += tip_mass_slug
            if self.has_v:
                M[g(n, "v"), g(n, "v")] += tip_mass_slug
        free = np.arange(nd, N)      # node 0 clamped
        self.free = free
        self.K, self.M = K[np.ix_(free, free)], M[np.ix_(free, free)]
        self.Kparts = {k: v[np.ix_(free, free)] for k, v in Kp.items()}
        nf = free.size
        Nw = np.zeros((n, N)); dNw = np.zeros((n, N)); Nt = np.zeros((n, N)); Nv = np.zeros((n, N))
        for e in range(n):
            iw = [g(e, "w"), g(e, "wp"), g(e + 1, "w"), g(e + 1, "wp")]
            Nw[e, iw] = [0.5, h / 8, 0.5, -h / 8]
            dNw[e, iw] = [-1.5 / h, -0.25, 1.5 / h, -0.25]
            if self.has_t:
                Nt[e, [g(e, "t"), g(e + 1, "t")]] = 0.5
            if self.has_v:
                iv = [g(e, "v"), g(e, "vp"), g(e + 1, "v"), g(e + 1, "vp")]
                Nv[e, iv] = [0.5, h / 8, 0.5, -h / 8]
        self.Nw, self.dNw, self.Nt, self.Nv = Nw[:, free], dNw[:, free], Nt[:, free], Nv[:, free]
        tw = np.zeros(N); tw[g(n, "w")] = 1.0
        twp = np.zeros(N); twp[g(n, "wp")] = 1.0
        self.tip_w, self.tip_wp = tw[free], twp[free]
        self.tip_t = np.zeros(nf)
        if self.has_t:
            tt = np.zeros(N); tt[g(n, "t")] = 1.0
            self.tip_t = tt[free]
        self.tip_v = np.zeros(nf)
        if self.has_v:
            tv = np.zeros(N); tv[g(n, "v")] = 1.0
            self.tip_v = tv[free]
        self.x_mid = (np.arange(n) + 0.5) * h          # from the root
        self._g = g

    def rigid_field(self, kind: str) -> np.ndarray:
        """Nodal vector on free dofs: 'w1' -> w = 1 everywhere; 'wx' -> w = x, w' = 1."""
        N = (self.n_el + 1) * self.nd
        r = np.zeros(N)
        for node in range(self.n_el + 1):
            x = node * self.h
            if kind == "w1":
                r[self._g(node, "w")] = 1.0
            elif kind == "wx":
                r[self._g(node, "w")] = x
                r[self._g(node, "wp")] = 1.0
        return r[self.free]

    def modes(self):
        """All modes: (omega rad/s ascending, mass-normalised shapes (n_free, n_free), class array 'b'/'t'/'v').
        Class = field with the largest strain-energy share. Sign fixed deterministically (dominant tip dof positive)."""
        Lc = np.linalg.cholesky(self.M)
        Li = np.linalg.inv(Lc)
        A = Li @ self.K @ Li.T
        A = 0.5 * (A + A.T)
        w2, Y = np.linalg.eigh(A)
        Phi = Li.T @ Y
        cls = []
        tipvec = {"w": self.tip_w, "t": self.tip_t, "v": self.tip_v}
        for j in range(Phi.shape[1]):
            ph = Phi[:, j]
            e = {k: float(ph @ Kp @ ph) for k, Kp in self.Kparts.items()}
            c = max(e, key=e.get)
            cls.append({"w": "b", "t": "t", "v": "v"}[c])
            tv = float(tipvec[c] @ ph)
            if tv < 0 or (tv == 0 and ph[np.argmax(np.abs(ph))] < 0):
                Phi[:, j] = -ph
        return np.sqrt(np.clip(w2, 0.0, None)), Phi, np.array(cls)


def select_modes(omega, Phi, cls, n_b: int, n_t: int = 0, n_v: int = 0, extra: int = 0):
    """Indices of the lowest n_b bending, n_t torsion, n_v in-plane modes, plus `extra` next-lowest modes of any type."""
    idx = []
    for c, k in (("b", n_b), ("t", n_t), ("v", n_v)):
        idx += [i for i in range(len(cls)) if cls[i] == c][:k]
    if extra:
        rest = [i for i in range(len(cls)) if i not in idx]
        idx += rest[:extra]
    return sorted(idx)


# =====================================================================================================================
# 2. Distributed genes (v2 genome)
# =====================================================================================================================
@dataclass(frozen=True)
class GeneV2:
    name: str
    lo: float
    hi: float
    scale: str       # "log" | "linear"
    default: float
    doc: str

    def decode(self, u: float) -> float:
        return self.lo * (self.hi / self.lo) ** u if self.scale == "log" else self.lo + u * (self.hi - self.lo)

    def encode(self, x: float) -> float:
        return math.log(x / self.lo) / math.log(self.hi / self.lo) if self.scale == "log" else (x - self.lo) / (self.hi - self.lo)


N_CP = 5                       # wing control points at eta = 0, .25, .5, .75, 1 (beam root .. tip)
CP_ETA = np.linspace(0.0, 1.0, N_CP)
GENES_V2: List[GeneV2] = [
    GeneV2("wing_ei_root", 0.6, 2.0, "log", 1.0, "wing bending-stiffness multiplier at the beam root (CP0)"),
    GeneV2("wing_ei_taper_1", 0.75, 1.05, "linear", 1.0, "EI multiplier ratio CP1/CP0 (eta .25 / 0)"),
    GeneV2("wing_ei_taper_2", 0.75, 1.05, "linear", 1.0, "EI multiplier ratio CP2/CP1 (eta .5 / .25)"),
    GeneV2("wing_ei_taper_3", 0.75, 1.05, "linear", 1.0, "EI multiplier ratio CP3/CP2 (eta .75 / .5)"),
    GeneV2("wing_ei_taper_4", 0.75, 1.05, "linear", 1.0, "EI multiplier ratio CP4/CP3 (eta 1 / .75)"),
    GeneV2("wing_gj_ratio_root", 0.8, 1.15, "linear", 1.0, "GJ multiplier / EI multiplier at the root"),
    GeneV2("wing_gj_ratio_tip", 0.8, 1.15, "linear", 1.0, "GJ multiplier / EI multiplier at the tip (linear in eta between)"),
    GeneV2("wing_nsm_root", 0.8, 1.25, "log", 1.0, "non-structural mass multiplier at the root"),
    GeneV2("wing_nsm_tip", 0.8, 1.25, "log", 1.0, "non-structural mass multiplier at the tip (log-linear between)"),
    GeneV2("tail_stiffness_scale", 0.6, 2.0, "log", 1.0, "HT + VT EI and GJ multiplier (also scales their structural mass)"),
    GeneV2("fuselage_stiffness_scale", 0.6, 2.0, "log", 1.0, "aft-fuselage EI multiplier (also scales its structural mass)"),
    GeneV2("struct_damping_ratio", 0.005, 0.05, "log", 0.02, "modal structural damping ratio, all bodies"),
]
ASYM_GENES: List[GeneV2] = [
    GeneV2("wing_asym_ei_delta", -0.1, 0.1, "linear", 0.0, "left/right EI+GJ split: right x(1+d), left x(1-d)"),
    GeneV2("wing_asym_nsm_delta", -0.1, 0.1, "linear", 0.0, "left/right non-structural mass split: right x(1+d), left x(1-d)"),
]
GENE_NAMES_V2 = [g.name for g in GENES_V2]
_V1_KEYS = (set(fw.GENOME_GENES) | {g.name for g in fw.STRUCT_SCHEMA} | set(fw.UNTIED_STIFFNESS)) - set(GENE_NAMES_V2)
_FIXED_KEYS = set(fw.FIXED_PHASE1) | {"tip_mass", "tip_mass_lb", "elastic_axis", "section_cg", "ea", "cg"}


def gene_schema(asymmetric: bool = False) -> List[GeneV2]:
    return GENES_V2 + (ASYM_GENES if asymmetric else [])


def baseline_genes(asymmetric: bool = False) -> Dict[str, float]:
    return {g.name: g.default for g in gene_schema(asymmetric)}


def decode_genome_v2(genome, asymmetric: bool = False) -> Dict[str, float]:
    """Named dict (physical values) or a vector in [0,1]^n (n = 12, or 14 with asymmetric) -> validated dict.

    Raises ValueError for: unknown keys, v1 gene names, fixed per-aircraft keys (EA / CG / tip mass), NaN/inf,
    values outside the ranges, asymmetric keys without asymmetric=True, vector entries outside [0,1] or wrong length.
    Missing named keys take the baseline value (documented)."""
    sch = gene_schema(asymmetric)
    if genome is None:
        return baseline_genes(asymmetric)
    if isinstance(genome, dict):
        out = baseline_genes(asymmetric)
        names = {g.name: g for g in sch}
        for k, v in genome.items():
            if k in _FIXED_KEYS:
                raise ValueError(f"v2 gene {k!r}: elastic axis / section CG / tip mass are fixed per aircraft, not genes")
            if k in _V1_KEYS:
                raise ValueError(f"{k!r} is a v1 (flexwing) gene; the v2 genome uses {GENE_NAMES_V2} "
                                 "(use fidelity='reduced' + project_to_reduced for a v1 model)")
            if k not in names:
                if k in {g.name for g in ASYM_GENES}:
                    raise ValueError(f"{k!r} needs asymmetric=True")
                raise ValueError(f"unknown v2 structure gene {k!r}")
            try:
                x = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"v2 gene {k}={v!r} is not a number")
            g = names[k]
            if not math.isfinite(x) or not (g.lo - 1e-12 <= x <= g.hi + 1e-12):
                raise ValueError(f"v2 gene {k}={v!r} outside [{g.lo}, {g.hi}]")
            out[k] = x
        return out
    u = np.asarray(genome, float).ravel()
    if u.size != len(sch):
        raise ValueError(f"v2 genome vector must have {len(sch)} entries (asymmetric={asymmetric}), got {u.size}")
    if not np.all(np.isfinite(u)) or np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("v2 genome vector entries must be finite and in [0, 1] (no silent clipping)")
    return {g.name: g.decode(float(x)) for g, x in zip(sch, u)}


def encode_genome_v2(genes: Dict[str, float], asymmetric: bool = False) -> np.ndarray:
    d = decode_genome_v2(genes, asymmetric)
    return np.array([g.encode(d[g.name]) for g in gene_schema(asymmetric)])


def pchip(x, y, xq) -> np.ndarray:
    """Monotone piecewise-cubic Hermite interpolation (Fritsch-Carlson interior slopes, 3-point end slopes with the
    Fritsch-Butland limiter), no scipy. Monotone data -> monotone interpolant (no overshoot)."""
    x = np.asarray(x, float); y = np.asarray(y, float); xq = np.asarray(xq, float)
    h = np.diff(x); d = np.diff(y) / h
    n = x.size
    m = np.zeros(n)
    for k in range(1, n - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def end(h0, h1, d0, d1):
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if s * d0 <= 0:
            return 0.0
        if d0 * d1 <= 0 and abs(s) > abs(3 * d0):
            return 3 * d0
        return s
    m[0] = end(h[0], h[1], d[0], d[1]) if n > 2 else d[0]
    m[-1] = end(h[-1], h[-2], d[-1], d[-2]) if n > 2 else d[-1]
    k = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    t = (xq - x[k]) / h[k]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1]


def wing_cp_values(genes: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Control-point multipliers (5 each): EI = root x cumulative taper ratios, GJ = EI x ratio(eta), NSM log-linear."""
    ei = [genes["wing_ei_root"]]
    for k in range(1, N_CP):
        ei.append(ei[-1] * genes[f"wing_ei_taper_{k}"])
    ei = np.array(ei)
    r = genes["wing_gj_ratio_root"] + (genes["wing_gj_ratio_tip"] - genes["wing_gj_ratio_root"]) * CP_ETA
    nsm = np.exp(np.log(genes["wing_nsm_root"]) + (np.log(genes["wing_nsm_tip"]) - np.log(genes["wing_nsm_root"])) * CP_ETA)
    return {"ei": ei, "gj": ei * r, "nsm": nsm}


def wing_distributions(genes: Dict[str, float], xi) -> Dict[str, np.ndarray]:
    """Spanwise multipliers at beam stations xi (0 root .. 1 tip), right and left wing (PCHIP in log space)."""
    cp = wing_cp_values(genes)
    out = {k: np.exp(pchip(CP_ETA, np.log(cp[k]), xi)) for k in ("ei", "gj", "nsm")}
    dE, dN = genes.get("wing_asym_ei_delta", 0.0), genes.get("wing_asym_nsm_delta", 0.0)
    res = {}
    for side, s in (("R", 1.0), ("L", -1.0)):
        res["ei_" + side] = out["ei"] * (1 + s * dE)
        res["gj_" + side] = out["gj"] * (1 + s * dE)
        res["nsm_" + side] = out["nsm"] * (1 + s * dN)
    return res


def smoothness_penalty(genes: Dict[str, float], w: float = 0.05) -> float:
    """J_smooth = w * sum of squared second differences of log(EI multiplier) over the 5 control points (3 terms)."""
    lr = np.log([genes[f"wing_ei_taper_{k}"] for k in range(1, N_CP)])
    return float(w * np.sum(np.diff(lr) ** 2))


# =====================================================================================================================
# 3. Per-aircraft v2 profiles (empennage / fuselage numbers: all notional)
# =====================================================================================================================
@dataclass
class SurfaceSpec:
    name: str
    kind: str                       # 'wing' | 'ht' | 'vt'
    span_ft: float                  # tip-to-tip (VT: 2 x height)
    area_ft2: float                 # whole surface (VT: 2 x S_v)
    taper: float
    sweep_deg: float
    root_frac: float
    mass_lb: float                  # one side (VT: the fin)
    f_b1_hz: float
    f_t1_hz: float
    f_ip_hz: Optional[float]
    x_ea: float
    x_cg: float
    aspect_eff: float               # aspect ratio used for the DATCOM lift slope
    n_el: int = 12
    n_b: int = 2
    n_t: int = 1
    n_ip: int = 0
    r_gyr: float = 0.25
    ei_taper_exp: float = 3.0
    mass_taper_exp: float = 1.0
    struct_frac: float = 0.55
    w_ei: float = 0.5
    ctrl_eta: Optional[Tuple[float, float]] = None
    ctrl_cf: float = 0.25
    ctrl_scale: Optional[float] = None    # strip cl_d/cm_d multiplier (None -> a/2pi 3D correction)
    all_moving: bool = False


# Notional empennage torsion frequencies are chosen so that the BASELINE genome's empennage blocks meet the 1.2 V_D
# requirement (CS-23/25.629 style) like a certified aircraft: T38 ht f_t1 45 -> 50 Hz, vt f_t1 40 -> 48 Hz (resume
# 2026-10-06; before: fin coalescence margin 1.07). Wing f_b1/f_t1 are v1's (flexwing.params_for), unchanged.
V2_PROFILES: Dict[str, Dict] = {
    # ht: AR (geometric), taper, sweep, root_frac, mass_lb (both halves), f_b1/f_t1 (uncoupled, Hz), x_ea/x_cg, elevator cf/eta
    #     (None for all-moving), downwash gradient (rigid tail-load perturbation only), elevator property (+ deg per unit
    #     norm for models that only write -norm)
    # vt: AR (h^2/S_v), taper, sweep, mass_lb, f_b1/f_t1, rudder cf/eta, z_ft (fin AC above AERORP), arm_ft (None ->
    #     metrics/lv-ft), rudder property
    # fus: aft_mass_lb (aft of the wing box, excluding tail surfaces), f_v1/f_l1 (first vertical / lateral bending with the
    #     tail mass at the tip), struct_frac (share scaled by fuselage_stiffness_scale)
    "c172x": dict(
        ht=dict(AR=5.8, taper=0.70, sweep=0.0, root_frac=0.10, mass_lb=26.0, f_b1=14.0, f_t1=40.0, x_ea=0.30, x_cg=0.36,
                cf=0.40, eta=(0.10, 1.0), all_moving=False, downwash=0.45, elev_prop="fcs/elevator-pos-rad", norm_deg=None),
        vt=dict(AR=1.5, taper=0.50, sweep=35.0, mass_lb=16.0, f_b1=12.0, f_t1=35.0, x_ea=0.30, x_cg=0.36, cf=0.40,
                eta=(0.05, 1.0), z_ft=2.0, arm_ft=None, rud_prop="fcs/rudder-pos-rad", norm_deg=None),
        fus=dict(aft_mass_lb=120.0, f_v1=9.0, f_l1=10.0, struct_frac=0.6)),
    "T38": dict(
        ht=dict(AR=4.0, taper=0.33, sweep=25.0, root_frac=0.25, mass_lb=150.0, f_b1=20.0, f_t1=50.0, x_ea=0.30, x_cg=0.34,
                cf=0.0, eta=None, all_moving=True, downwash=0.35, elev_prop="fcs/elevator-pos-norm", norm_deg=15.0),
        vt=dict(AR=1.4, taper=0.30, sweep=35.0, mass_lb=120.0, f_b1=14.0, f_t1=48.0, x_ea=0.35, x_cg=0.40, cf=0.30,
                eta=(0.05, 0.60), z_ft=4.0, arm_ft=None, rud_prop="fcs/rudder-pos-norm", norm_deg=20.0),
        fus=dict(aft_mass_lb=1000.0, f_v1=12.0, f_l1=13.0, struct_frac=0.5)),
    "737": dict(
        ht=dict(AR=6.2, taper=0.20, sweep=30.0, root_frac=0.15, mass_lb=2400.0, f_b1=6.0, f_t1=18.0, x_ea=0.35, x_cg=0.40,
                cf=0.30, eta=(0.10, 0.95), all_moving=False, downwash=0.40, elev_prop="fcs/elevator-pos-rad", norm_deg=None),
        vt=dict(AR=1.8, taper=0.30, sweep=35.0, mass_lb=1800.0, f_b1=4.5, f_t1=14.0, x_ea=0.35, x_cg=0.40, cf=0.30,
                eta=(0.05, 0.90), z_ft=12.0, arm_ft=None, rud_prop="fcs/rudder-pos-rad", norm_deg=None),
        fus=dict(aft_mass_lb=12000.0, f_v1=3.0, f_l1=3.5, struct_frac=0.4)),
    "f16": dict(
        ht=dict(AR=2.5, taper=0.30, sweep=40.0, root_frac=0.20, mass_lb=400.0, f_b1=20.0, f_t1=50.0, x_ea=0.30, x_cg=0.34,
                cf=0.0, eta=None, all_moving=True, downwash=0.30, elev_prop="fcs/elevator-pos-rad", norm_deg=None),
        vt=dict(AR=1.3, taper=0.40, sweep=47.0, mass_lb=250.0, f_b1=15.0, f_t1=40.0, x_ea=0.35, x_cg=0.40, cf=0.30,
                eta=(0.05, 0.75), z_ft=4.5, arm_ft=15.0,    # f16.xml metrics vtailarm = 0 -> notional 15 ft
                rud_prop="fcs/rudder-pos-rad", norm_deg=None),
        fus=dict(aft_mass_lb=2500.0, f_v1=10.0, f_l1=11.0, struct_frac=0.5)),
}
WING_V2 = dict(n_el=32, n_b=3, n_t=2, n_ip=1, f_ip_ratio=2.5)
TAIL_N_EL, FUS_N_EL = 12, 12


@dataclass
class Geometry:
    """Metrics read once from the prepared model (stored in flexbody_meta.json)."""
    bw_ft: float
    sw_ft2: float
    sh_ft2: float
    lh_ft: float
    sv_ft2: float
    lv_ft: float
    empty_wt_lb: float


def v2_params(model: str, geom: Geometry) -> Dict:
    """Everything that defines the v2 model of an aircraft (used for building AND for model_version hashing)."""
    if model not in V2_PROFILES:
        raise ValueError(f"no v2 profile for {model!r} (have {sorted(V2_PROFILES)})")
    pw = fw.params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
    return {"wing": asdict(pw), "v2": V2_PROFILES[model], "wing_v2": WING_V2, "tail_n_el": TAIL_N_EL,
            "fus_n_el": FUS_N_EL, "geom": asdict(geom), "fmt": PREPARE_FMT_V2, "min_gauge": dict(MIN_GAUGE)}


# =====================================================================================================================
# 4. Surfaces (one semi-span / fin) and fuselage beams
# =====================================================================================================================
class Surface:
    """Beam FE + strip data of one lifting surface (semi-span) with spanwise multipliers on EI/GJ/NSM."""

    def __init__(self, sp: SurfaceSpec, ei_mult=None, gj_mult=None, nsm_mult=None, zeta: float = 0.02,
                 n_sel: Optional[Tuple[int, int, int]] = None, extra_modes: int = 0, cal: Optional[Dict] = None,
                 cla: Optional[float] = None):
        self.sp = sp
        n = sp.n_el
        s = sp.span_ft / 2
        self.s, self.y0 = s, sp.root_frac * s
        self.L = s - self.y0
        self.lam = math.radians(sp.sweep_deg)
        self.c_root = sp.area_ft2 / (s * (1 + sp.taper))
        self.dy = self.L / n
        self.y = self.y0 + (np.arange(n) + 0.5) * self.dy
        self.xi = (self.y - self.y0) / self.L
        self.c = self.c_root * (1 - (1 - sp.taper) * self.y / s)
        one = np.ones(n)
        self.ei_mult = one if ei_mult is None else np.asarray(ei_mult, float)
        self.gj_mult = one if gj_mult is None else np.asarray(gj_mult, float)
        self.nsm_mult = one if nsm_mult is None else np.asarray(nsm_mult, float)
        self.e_c = (sp.x_ea - 0.25) * self.c
        self.d34 = (0.75 - sp.x_ea) * self.c
        self.x_theta = (sp.x_cg - sp.x_ea) * self.c
        self.a = cla if cla is not None else fw.datcom_cla(sp.aspect_eff, self.lam)
        self._cla0 = self.a
        # baseline mass (slug/ft, v1 normalisation) and gene-scaled mass (v1 formula applied per element)
        wsh = (self.c / self.c_root) ** sp.mass_taper_exp
        self.m0 = sp.mass_lb * LB2SLUG * wsh / (np.sum(wsh) * self.dy)
        f = (1 - sp.struct_frac) * self.nsm_mult + sp.struct_frac * (sp.w_ei * gauge_mass_factor(self.ei_mult)
                                                                     + (1 - sp.w_ei) * gauge_mass_factor(self.gj_mult))
        self.m = self.m0 * f
        self.mass_lb = float(np.sum(self.m) * self.dy * G0)
        self.dmass_lb = float(np.sum(self.m0 * (f - 1.0)) * self.dy * G0)     # exactly 0.0 at the baseline genome
        self.Ia = self.m * ((sp.r_gyr * self.c) ** 2 + self.x_theta ** 2)
        shape = (self.c / self.c_root) ** sp.ei_taper_exp
        self.cal = cal if cal is not None else calibrate_surface(sp)
        self.EI = self.cal["EI_root0"] * shape * self.ei_mult
        self.GJ = self.cal["GJ_root0"] * shape * self.gj_mult
        self.EIv = self.cal["EIv_root0"] * shape * self.ei_mult if sp.f_ip_hz else None
        self.beam = Beam(self.L, self.EI, self.m, GJ=self.GJ, Ia=self.Ia, mxt=self.m * self.x_theta, EIv=self.EIv)
        om, Phi, cls = self.beam.modes()
        nb, nt, nv = n_sel or (sp.n_b, sp.n_t, sp.n_ip)
        idx = select_modes(om, Phi, cls, nb, nt, nv if sp.f_ip_hz else 0, extra=extra_modes)
        self.all_omega, self.all_cls, self.all_Phi = om, cls, Phi
        self.idx = idx
        self.omega, self.cls, self.Phi = om[idx], cls[idx], Phi[:, idx]
        self.zeta = zeta
        b = self.beam
        self.PhiW, self.dPhiW, self.PsiT, self.PhiV = b.Nw @ self.Phi, b.dNw @ self.Phi, b.Nt @ self.Phi, b.Nv @ self.Phi
        self.tipW, self.tipT = b.tip_w @ self.Phi, b.tip_t @ self.Phi
        self.tipV = b.tip_v @ self.Phi       # in-plane tip deflection (zero rows without in-plane DOFs); telemetry only
        eta = self.y / s
        self.ctrl_mask = np.zeros(n)
        self.cl_d = self.cm_d = 0.0
        if sp.ctrl_eta is not None and sp.ctrl_cf > 0:
            self.ctrl_mask = ((eta >= sp.ctrl_eta[0]) & (eta <= sp.ctrl_eta[1])).astype(float)
            cl_d, cm_d = fw.flap_coeffs(sp.ctrl_cf)
            k = sp.ctrl_scale if sp.ctrl_scale is not None else self.a / (2 * math.pi)
            self.cl_d, self.cm_d = k * cl_d, k * cm_d
        self.lift_w = self.c * self.dy * self.a

    def kappa(self, mach: float) -> float:
        return fw.datcom_cla(self.sp.aspect_eff, self.lam, mach=mach) / fw.datcom_cla(self.sp.aspect_eff, self.lam)


def _torsion_only_omega1(L, GJ, Ia) -> float:
    n = len(GJ); h = L / n
    K = np.zeros((n + 1, n + 1)); M = np.zeros((n + 1, n + 1))
    for e in range(n):
        K[e:e + 2, e:e + 2] += GJ[e] / h * np.array([[1, -1], [-1, 1]])
        M[e:e + 2, e:e + 2] += Ia[e] * h / 6 * np.array([[2, 1], [1, 2]])
    K, M = K[1:, 1:], M[1:, 1:]
    Li = np.linalg.inv(np.linalg.cholesky(M))
    A = Li @ K @ Li.T
    return float(np.sqrt(np.linalg.eigvalsh(0.5 * (A + A.T))[0]))


@lru_cache(maxsize=256)
def _calibrate_cached(key: str) -> Tuple[float, float, float]:
    d = json.loads(key)
    sp = SurfaceSpec(**d)
    n = sp.n_el
    s = sp.span_ft / 2
    y0 = sp.root_frac * s
    L = s - y0
    dy = L / n
    y = y0 + (np.arange(n) + 0.5) * dy
    c_root = sp.area_ft2 / (s * (1 + sp.taper))
    c = c_root * (1 - (1 - sp.taper) * y / s)
    wsh = (c / c_root) ** sp.mass_taper_exp
    m0 = sp.mass_lb * LB2SLUG * wsh / (np.sum(wsh) * dy)
    Ia0 = m0 * ((sp.r_gyr * c) ** 2 + ((sp.x_cg - sp.x_ea) * c) ** 2)
    shape = (c / c_root) ** sp.ei_taper_exp
    w1 = Beam(L, shape, m0).modes()[0][0]                 # uncoupled bending, unit root EI
    EI0 = (2 * math.pi * sp.f_b1_hz / w1) ** 2
    GJ0 = (2 * math.pi * sp.f_t1_hz / _torsion_only_omega1(L, shape, Ia0)) ** 2
    EIv0 = EI0 * (sp.f_ip_hz / sp.f_b1_hz) ** 2 if sp.f_ip_hz else 0.0   # same mass & shape: f ~ sqrt(EI)
    return EI0, GJ0, EIv0


def calibrate_surface(sp: SurfaceSpec) -> Dict[str, float]:
    """Root stiffnesses such that the UNCOUPLED baseline FE frequencies equal f_b1 / f_t1 / f_ip (same rule as v1)."""
    d = asdict(sp)
    d["ctrl_eta"] = list(d["ctrl_eta"]) if d["ctrl_eta"] is not None else None
    EI0, GJ0, EIv0 = _calibrate_cached(json.dumps(d, sort_keys=True))
    return {"EI_root0": EI0, "GJ_root0": GJ0, "EIv_root0": EIv0}


class Fuselage:
    """Aft fuselage: uniform clamped-free bending beam from the wing station to the tail, tail mass lumped at the tip."""

    def __init__(self, L: float, aft_mass_lb: float, tail_mass_lb_base: float, tail_mass_lb: float, f1_hz: float,
                 struct_frac: float, scale: float, n_el: int, n_modes: int = 2):
        self.L = L
        m0 = aft_mass_lb * LB2SLUG / L
        g = float(gauge_mass_factor(scale))                                     # minimum-gauge floor (MIN_GAUGE)
        self.m = m0 * ((1 - struct_frac) + struct_frac * g)
        self.mass_lb = float(self.m * L * G0)
        self.base_mass_lb = aft_mass_lb
        self.dmass_lb = float(aft_mass_lb * struct_frac * (g - 1.0))          # exactly 0.0 at scale 1
        b0 = Beam(L, np.ones(n_el), np.full(n_el, m0), tip_mass_slug=tail_mass_lb_base * LB2SLUG)
        self.EI0 = (2 * math.pi * f1_hz / b0.modes()[0][0]) ** 2
        self.EI = self.EI0 * scale
        self.tip_mass_slug = tail_mass_lb * LB2SLUG
        self.beam = b = Beam(L, np.full(n_el, self.EI), np.full(n_el, self.m), tip_mass_slug=self.tip_mass_slug)
        om, Phi, _ = b.modes()
        self.all_omega = om
        self.omega, self.Phi = om[:n_modes], Phi[:, :n_modes]
        self.tipW, self.tipWp = b.tip_w @ self.Phi, b.tip_wp @ self.Phi
        self.PhiW_mid = b.Nw @ self.Phi
        self.x_mid, self.dx = b.x_mid, b.h
        self.Q_w1 = self.Phi.T @ (b.M @ b.rigid_field("w1"))       # gen. force per unit uniform acceleration field
        self.Q_wx = self.Phi.T @ (b.M @ b.rigid_field("wx"))       # per unit acceleration field = x (rotation)


# =====================================================================================================================
# 5. The global v2 model
# =====================================================================================================================
BODY_NAMES = ("wingR", "wingL", "htR", "htL", "vt", "fusV", "fusL")
GROUPS = ("w", "h", "v")               # lift-slope / compressibility groups: wing, horizontal tail, vertical tail
OUT_NAMES = ("wingR_bm", "wingL_bm", "wingR_torque", "wingL_torque", "wingR_ip_bm", "wingL_ip_bm",
             "htR_bm", "htL_bm", "vt_bm", "fusV_bm", "fusL_bm")
COEF_NAMES = ("lift", "nz", "qkw_p_V", "q_dail_R", "q_dail_L", "pdot", "qdot", "qkh_alpha_ht", "q_delev",
              "qkv_alpha_vt", "q_drud", "ny", "rdot", "nx", "drag", "qkh_p_V")
CI = {nm: i for i, nm in enumerate(COEF_NAMES)}


def load_meta(model: str, root_v2: str = ROOT_V2) -> Dict:
    p = os.path.join(root_v2, "aircraft", model, "flexbody_meta.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: run flexbody.prepare_aircraft_v2({model!r}, root_v2)")
    with open(p) as f:
        return json.load(f)


def geometry_for(model: str, root_v2: str = ROOT_V2) -> Geometry:
    return Geometry(**load_meta(model, root_v2)["geom"])


class FlexBodyModel:
    """Global modal model (all bodies) of one aircraft for one v2 genome. Modal coordinates are mass-normalised per
    body (M = I); bodies couple through the aero (tail strips ride on the fuselage tip), not through inertia."""

    def __init__(self, model: str, genes=None, geom: Optional[Geometry] = None, asymmetric: bool = False,
                 wing_extra_modes: int = 0, root_v2: str = ROOT_V2, wing_n_sel: Optional[Tuple[int, int, int]] = None):
        self.model = model
        self.asymmetric = bool(asymmetric)
        self.genes = decode_genome_v2(genes, self.asymmetric)
        self.geom = geom or geometry_for(model, root_v2)
        self.params = v2_params(model, self.geom)
        g, P = self.genes, self.params
        pw = fw.WingParams(**P["wing"])
        self.pw = pw
        pr = P["v2"]
        zeta = g["struct_damping_ratio"]
        wsp = SurfaceSpec(name="wing", kind="wing", span_ft=pw.span_ft, area_ft2=pw.area_ft2, taper=pw.taper,
                          sweep_deg=pw.sweep_deg, root_frac=pw.root_frac, mass_lb=pw.wing_mass_lb / 2, f_b1_hz=pw.f_b1_hz,
                          f_t1_hz=pw.f_t1_hz, f_ip_hz=WING_V2["f_ip_ratio"] * pw.f_b1_hz, x_ea=pw.x_ea, x_cg=pw.x_cg,
                          aspect_eff=pw.span_ft ** 2 / pw.area_ft2, n_el=WING_V2["n_el"], n_b=WING_V2["n_b"],
                          n_t=WING_V2["n_t"], n_ip=WING_V2["n_ip"], r_gyr=pw.r_gyr, ei_taper_exp=pw.ei_taper_exp,
                          mass_taper_exp=pw.mass_taper_exp, struct_frac=pw.struct_frac, w_ei=pw.w_ei,
                          ctrl_eta=tuple(pw.ail_eta), ctrl_cf=pw.ail_cf)
        # aileron strip calibration to the FDM's Cl_da (same rule as v1, on the 32 v2 strips)
        n = wsp.n_el
        s_ = pw.span_ft / 2
        y0 = pw.root_frac * s_
        dy = (s_ - y0) / n
        yy = y0 + (np.arange(n) + 0.5) * dy
        cc = pw.area_ft2 / (s_ * (1 + pw.taper)) * (1 - (1 - pw.taper) * yy / s_)
        mask = ((yy / s_ >= pw.ail_eta[0]) & (yy / s_ <= pw.ail_eta[1])).astype(float)
        cl_d0, _ = fw.flap_coeffs(pw.ail_cf)
        strip = 2 * np.sum(cc * dy * cl_d0 * mask * yy) / (pw.area_ft2 * pw.span_ft)
        wsp.ctrl_scale = (pw.ail_cl_da_target / strip) if (pw.ail_cl_da_target is not None and strip > 0) else None
        self.wing_spec = wsp
        xi = (np.arange(n) + 0.5) / n
        self.dist = dist = wing_distributions(g, xi)
        self.wingR = Surface(wsp, dist["ei_R"], dist["gj_R"], dist["nsm_R"], zeta, n_sel=wing_n_sel, extra_modes=wing_extra_modes)
        if self.asymmetric and (g.get("wing_asym_ei_delta", 0.0) != 0.0 or g.get("wing_asym_nsm_delta", 0.0) != 0.0):
            self.wingL = Surface(wsp, dist["ei_L"], dist["gj_L"], dist["nsm_L"], zeta, n_sel=wing_n_sel, extra_modes=wing_extra_modes)
        else:
            self.wingL = self.wingR          # identical (deterministic) -> share the FE solution
        ts = g["tail_stiffness_scale"]
        geo = self.geom
        h, v, f = pr["ht"], pr["vt"], pr["fus"]
        hsp = SurfaceSpec(name="ht", kind="ht", span_ft=math.sqrt(h["AR"] * geo.sh_ft2), area_ft2=geo.sh_ft2, taper=h["taper"],
                          sweep_deg=h["sweep"], root_frac=h["root_frac"], mass_lb=h["mass_lb"] / 2, f_b1_hz=h["f_b1"],
                          f_t1_hz=h["f_t1"], f_ip_hz=None, x_ea=h["x_ea"], x_cg=h["x_cg"], aspect_eff=h["AR"], n_el=TAIL_N_EL,
                          n_b=2, n_t=1, ctrl_eta=tuple(h["eta"]) if h["eta"] else None, ctrl_cf=h["cf"], all_moving=h["all_moving"])
        hv = math.sqrt(v["AR"] * geo.sv_ft2)
        vsp = SurfaceSpec(name="vt", kind="vt", span_ft=2 * hv, area_ft2=2 * geo.sv_ft2, taper=v["taper"], sweep_deg=v["sweep"],
                          root_frac=0.0, mass_lb=v["mass_lb"], f_b1_hz=v["f_b1"], f_t1_hz=v["f_t1"], f_ip_hz=None,
                          x_ea=v["x_ea"], x_cg=v["x_cg"], aspect_eff=1.55 * v["AR"],   # end-plate effect of fuselage/HT
                          n_el=TAIL_N_EL, n_b=2, n_t=1, ctrl_eta=tuple(v["eta"]), ctrl_cf=v["cf"])
        self.ht_spec, self.vt_spec = hsp, vsp
        tm = np.full(TAIL_N_EL, ts)
        self.htR = Surface(hsp, tm, tm, None, zeta)
        self.htL = self.htR
        self.vt = Surface(vsp, tm, tm, None, zeta)
        self.l_h = geo.lh_ft
        self.l_v = v["arm_ft"] if v.get("arm_ft") else geo.lv_ft
        self.z_v = v["z_ft"]
        self.downwash = h["downwash"]
        fs = g["fuselage_stiffness_scale"]
        tail0 = h["mass_lb"] + v["mass_lb"]
        tail = 2 * self.htR.mass_lb + self.vt.mass_lb
        self.fusV = Fuselage(self.l_h, f["aft_mass_lb"], tail0, tail, f["f_v1"], f["struct_frac"], fs, FUS_N_EL)
        self.fusL = Fuselage(self.l_h, f["aft_mass_lb"], tail0, tail, f["f_l1"], f["struct_frac"], fs, FUS_N_EL)
        self._assemble()

    # ------------------------------------------------------------------------------------------------------------
    def _assemble(self):
        bodies = [self.wingR, self.wingL, self.htR, self.htL, self.vt, self.fusV, self.fusL]
        sizes = [b.Phi.shape[1] for b in bodies]
        offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        N = int(offs[-1])
        self.N = N
        self.slices = {nm: slice(int(offs[i]), int(offs[i + 1])) for i, nm in enumerate(BODY_NAMES)}
        om = np.concatenate([b.omega for b in bodies])
        self.omega = om
        self.M = np.eye(N)
        self.K = np.diag(om ** 2)
        self.C = np.diag(2 * self.genes["struct_damping_ratio"] * om)
        surf = [("wingR", self.wingR, 1.0, 0), ("wingL", self.wingL, -1.0, 0), ("htR", self.htR, 1.0, 1),
                ("htL", self.htL, -1.0, 1), ("vt", self.vt, 0.0, 2)]
        ns = sum(s.sp.n_el for _, s, _, _ in surf)
        self.ns = ns
        PhiW, dPhiW, PsiT, PhiV, W_in = (np.zeros((ns, N)) for _ in range(5))
        st = {k: np.zeros(ns) for k in ("lw", "y", "sign", "c", "dy", "ec", "d34", "xth", "mdy", "x", "z", "arm", "group",
                                         "cl_d", "cm_d", "mask", "lam")}
        body_strips = {}
        i0 = 0
        fV, fL = self.slices["fusV"], self.slices["fusL"]
        wy_mac = self.pw.span_ft / 2 / 3 * (1 + 2 * self.pw.taper) / (1 + self.pw.taper)
        for nm, s, sg, grp in surf:
            n = s.sp.n_el
            r = slice(i0, i0 + n)
            body_strips[nm] = r
            cs = self.slices[nm]
            PhiW[r, cs] = s.PhiW
            dPhiW[r, cs] = s.dPhiW
            PsiT[r, cs] = s.PsiT
            PhiV[r, cs] = s.PhiV
            W_in[r, cs] = s.PhiW - s.x_theta[:, None] * s.PsiT      # own body only (tail mass is in the fuselage tip)
            if grp == 1:      # tail strips ride on the fuselage tip: plunge w_f,tip and nose-down rotation -w'_f,tip
                PhiW[r, fV] += self.fusV.tipW[None, :]
                PsiT[r, fV] += -self.fusV.tipWp[None, :]
            elif grp == 2:
                PhiW[r, fL] += self.fusL.tipW[None, :]
                PsiT[r, fL] += -self.fusL.tipWp[None, :]
            st["lw"][r] = s.lift_w; st["y"][r] = s.y; st["sign"][r] = sg
            st["c"][r] = s.c; st["dy"][r] = s.dy; st["ec"][r] = s.e_c; st["d34"][r] = s.d34; st["xth"][r] = s.x_theta
            st["mdy"][r] = s.m * s.dy; st["arm"][r] = s.y - s.y0; st["group"][r] = grp; st["lam"][r] = s.lam
            st["cl_d"][r] = s.cl_d; st["cm_d"][r] = s.cm_d; st["mask"][r] = s.ctrl_mask
            if grp == 0:
                st["x"][r] = (s.y - wy_mac) * math.tan(s.lam)
            else:
                y_mac = s.s / 3 * (1 + 2 * s.sp.taper) / (1 + s.sp.taper)
                st["x"][r] = (self.l_h if grp == 1 else self.l_v) + (s.y - y_mac) * math.tan(s.lam)
                if grp == 2:
                    st["z"][r] = self.z_v + (s.y - y_mac)
            i0 += n
        self.st, self.body_strips = st, body_strips
        lam = st["lam"]
        G = np.cos(lam)[:, None] * PsiT - np.sin(lam)[:, None] * dPhiW
        H = -PhiW + st["d34"][:, None] * PsiT
        W = PhiW + st["ec"][:, None] * PsiT
        self.PhiW, self.dPhiW, self.PsiT, self.PhiV, self.W_in = PhiW, dPhiW, PsiT, PhiV, W_in
        self.G, self.H, self.W = G, H, W
        lw = st["lw"]
        self.A_K, self.A_C = {}, {}
        for gi, gname in enumerate(GROUPS):
            sel = (st["group"] == gi).astype(float)
            self.A_K[gname] = W.T @ ((lw * sel)[:, None] * G)
            self.A_C[gname] = W.T @ ((lw * sel)[:, None] * H)
        Lnc = 0.5 * math.pi * st["c"] ** 2 * st["dy"]
        self.A_C_nc = PhiW.T @ (Lnc[:, None] * PsiT) - PsiT.T @ ((Lnc * st["d34"])[:, None] * PsiT)
        self.kappa_fns = {"w": self.wingR.kappa, "h": self.htR.kappa, "v": self.vt.kappa}
        self._build_load_bases()

    # ------------------------------------------------------------------------------------------------------------
    def _build_load_bases(self):
        st, N, ns = self.st, self.N, self.ns
        k = len(COEF_NAMES)
        Lb, Mb, Fb, Fv = (np.zeros((k, ns)) for _ in range(4))
        bs = self.body_strips
        grp, sg, y, lw, mdy = st["group"], st["sign"], st["y"], st["lw"], st["mdy"]
        isw, ish, isv = grp == 0, grp == 1, grp == 2
        p = self.pw
        s = p.span_ft / 2
        c_root = p.area_ft2 / (s * (1 + p.taper))

        def shape(v):
            cv = c_root * (1 - (1 - p.taper) * v / s)
            return 0.5 * cv / (p.area_ft2 / 2 / s) + 0.5 * 4 / math.pi * np.sqrt(np.clip(1 - (v / s) ** 2, 0, None))
        yy = (np.arange(2000) + 0.5) / 2000 * s
        norm = np.sum(shape(yy)) * (s / 2000)
        for nm in ("wingR", "wingL"):           # Schrenk share of the total aero lift (as v1)
            r = bs[nm]
            Lb[CI["lift"], r] = 0.5 * p.wing_lift_share * shape(y[r]) / norm * st["dy"][r]
        Lb[CI["qkw_p_V"], isw] = lw[isw] * y[isw] * sg[isw]
        for nm, key in (("wingR", "q_dail_R"), ("wingL", "q_dail_L")):
            r = bs[nm]
            Lb[CI[key], r] = st["c"][r] * st["dy"][r] * st["cl_d"][r] * st["mask"][r]
            Mb[CI[key], r] = st["c"][r] ** 2 * st["dy"][r] * st["cm_d"][r] * st["mask"][r]
        lift_dir = isw | ish
        Fb[CI["nz"], lift_dir] = -G0 * mdy[lift_dir]
        Fb[CI["pdot"], lift_dir] = mdy[lift_dir] * y[lift_dir] * sg[lift_dir]
        Fb[CI["pdot"], isv] = -mdy[isv] * st["z"][isv]
        Fb[CI["qdot"], ish] = mdy[ish] * st["x"][ish]
        Fb[CI["ny"], isv] = -G0 * mdy[isv]
        Fb[CI["rdot"], isv] = mdy[isv] * st["x"][isv]
        Lb[CI["qkh_alpha_ht"], ish] = lw[ish]
        Lb[CI["qkh_p_V"], ish] = lw[ish] * y[ish] * sg[ish]
        if not self.ht_spec.all_moving:
            Lb[CI["q_delev"], ish] = st["c"][ish] * st["dy"][ish] * st["cl_d"][ish] * st["mask"][ish]
            Mb[CI["q_delev"], ish] = st["c"][ish] ** 2 * st["dy"][ish] * st["cm_d"][ish] * st["mask"][ish]
        Lb[CI["qkv_alpha_vt"], isv] = lw[isv]
        Lb[CI["q_drud"], isv] = st["c"][isv] * st["dy"][isv] * st["cl_d"][isv] * st["mask"][isv]
        Mb[CI["q_drud"], isv] = st["c"][isv] ** 2 * st["dy"][isv] * st["cm_d"][isv] * st["mask"][isv]
        area_w = float(np.sum((st["c"] * st["dy"])[isw]))      # both wings
        Fv[CI["nx"], isw] = G0 * mdy[isw]                       # in-plane, aft +: inertia of forward acceleration
        Fv[CI["drag"], isw] = 0.5 * (st["c"] * st["dy"])[isw] / area_w   # half the airframe drag carried by the wings
        self.Lb, self.Mb, self.Fb, self.Fv = Lb, Mb, Fb, Fv
        Qb = Lb @ self.W + Mb @ self.PsiT + Fb @ self.W_in + Fv @ self.PhiV
        fV, fL = self.slices["fusV"], self.slices["fusL"]
        Qb[CI["nz"], fV] += -G0 * self.fusV.Q_w1
        Qb[CI["qdot"], fV] += self.fusV.Q_wx
        Qb[CI["ny"], fL] += -G0 * self.fusL.Q_w1
        Qb[CI["rdot"], fL] += self.fusL.Q_wx
        self.Qbasis = Qb
        # ---- feedback rows (elastic aero increments): per group [force, roll, pitch|yaw]
        x, z = st["x"], st["z"]
        rows_G, rows_H = [], []
        for msk, kind in ((isw, "lift"), (ish, "lift"), (isv, "side")):
            l = lw * msk
            vecs = (l, -sg * y * l, -x * l) if kind == "lift" else (l, z * l, -x * l)
            for vv in vecs:
                rows_G.append(vv @ self.G)
                rows_H.append(vv @ self.H)
        self.R_de = np.vstack(rows_G).T          # (N, 9): [Lw, Rw, Pw, Lh, Rh, Ph, Yv, Rv, Nv]
        self.R_d = np.vstack(rows_H).T
        lW = lw * isw                            # elastic dihedral (wing): d_alpha = sign * beta * w'
        self.R_beta = np.vstack([(lW * sg) @ self.dPhiW, (-sg * y * lW * sg) @ self.dPhiW, (-x * lW * sg) @ self.dPhiW]).T
        self.A_beta = self.W.T @ ((lW * sg)[:, None] * self.dPhiW)
        # ---- root-load outputs (force summation)
        n_out = len(OUT_NAMES)
        RB = np.zeros((k, n_out)); RE = np.zeros((N, n_out)); RD = np.zeros((N, n_out)); RA = np.zeros((N, n_out))
        og = np.zeros(n_out, int)
        arm = st["arm"]

        def add_bm(o, r, armv, gi):
            RB[:, o] = (Lb[:, r] + Fb[:, r]) @ armv
            RE[:, o] = (armv * lw[r]) @ self.G[r]
            RD[:, o] = (armv * lw[r]) @ self.H[r]
            RA[:, o] = -(armv * mdy[r]) @ self.W_in[r]
            og[o] = gi
        add_bm(0, bs["wingR"], arm[bs["wingR"]], 0)
        add_bm(1, bs["wingL"], arm[bs["wingL"]], 0)
        for o, nm in ((2, "wingR"), (3, "wingL")):        # torque about the EA at the root (nose-up +)
            r = bs[nm]
            ec, xth = st["ec"][r], st["xth"][r]
            RB[:, o] = Lb[:, r] @ ec + Mb[:, r].sum(axis=1) - Fb[:, r] @ xth
            RE[:, o] = (ec * lw[r]) @ self.G[r]
            RD[:, o] = (ec * lw[r]) @ self.H[r]
            RA[:, o] = (xth * mdy[r]) @ self.W_in[r]
        for o, nm in ((4, "wingR"), (5, "wingL")):        # in-plane root bending (aft load +)
            r = bs[nm]
            RB[:, o] = Fv[:, r] @ arm[r]
            RA[:, o] = -(arm[r] * mdy[r]) @ self.PhiV[r]
        add_bm(6, bs["htR"], arm[bs["htR"]], 1)
        add_bm(7, bs["htL"], arm[bs["htL"]], 1)
        add_bm(8, bs["vt"], arm[bs["vt"]], 2)
        # fuselage bending at the wing station: tail aero + tail strip inertia at arm x, own distributed inertia
        for o, msk, fus, sl, gi, cN, cX in ((9, ish, self.fusV, self.slices["fusV"], 1, "nz", "qdot"),
                                             (10, isv, self.fusL, self.slices["fusL"], 2, "ny", "rdot")):
            xa = x[msk]
            RB[:, o] = (Lb[:, msk] + Fb[:, msk]) @ xa
            RE[:, o] = (xa * lw[msk]) @ self.G[msk]
            RD[:, o] = (xa * lw[msk]) @ self.H[msk]
            mx = fus.m * fus.dx
            RB[CI[cN], o] += -G0 * float(np.sum(mx * fus.x_mid))
            RB[CI[cX], o] += float(np.sum(mx * fus.x_mid ** 2))
            RA[sl, o] = -(mx * fus.x_mid) @ fus.PhiW_mid
            og[o] = gi
        self.RB, self.RE, self.RD, self.RA, self.out_group = RB, RE, RD, RA, og
        self.tip_rows = {
            "tip_w_ft_R": self._pad("wingR", self.wingR.tipW), "tip_w_ft_L": self._pad("wingL", self.wingL.tipW),
            "tip_twist_R": self._pad("wingR", self.wingR.tipT), "tip_twist_L": self._pad("wingL", self.wingL.tipT),
            "ht_tip_w_ft": self._pad("htR", self.htR.tipW), "vt_tip_w_ft": self._pad("vt", self.vt.tipW),
            "fusV_tip_w_ft": self._pad("fusV", self.fusV.tipW), "fusL_tip_w_ft": self._pad("fusL", self.fusL.tipW),
            "ht_incidence": self._pad("fusV", -self.fusV.tipWp), "vt_sideslip": self._pad("fusL", -self.fusL.tipWp)}
        self.TIP = np.vstack(list(self.tip_rows.values())).T     # (N, 10)
        # telemetry-only tip outputs, kept in a SEPARATE matrix so TIP and every cost / margin input stay bit-identical
        self.tip_rows_tel = {
            "htL_tip_w_ft": self._pad("htL", self.htL.tipW),
            "ht_tip_twist": self._pad("htR", self.htR.tipT), "htL_tip_twist": self._pad("htL", self.htL.tipT),
            "vt_tip_twist": self._pad("vt", self.vt.tipT),
            "wingR_tip_ip_ft": self._pad("wingR", self.wingR.tipV), "wingL_tip_ip_ft": self._pad("wingL", self.wingL.tipV)}
        self.TIP_TEL = np.vstack(list(self.tip_rows_tel.values())).T     # (N, 6)

    def _pad(self, body, vec):
        out = np.zeros(self.N)
        out[self.slices[body]] = vec
        return out

    def kappas(self, mach: float) -> Dict[str, float]:
        return {g: f(mach) for g, f in self.kappa_fns.items()}

    def mass_summary(self) -> Dict[str, float]:
        w0, h0 = self.pw.wing_mass_lb / 2, self.ht_spec.mass_lb
        d = {"wingR_lb": self.wingR.dmass_lb, "wingL_lb": self.wingL.dmass_lb,
             "ht_lb": self.htR.dmass_lb + self.htL.dmass_lb, "vt_lb": self.vt.dmass_lb, "fus_lb": self.fusV.dmass_lb}
        d["total_lb"] = float(sum(d.values()))
        d["baseline_flexible_lb"] = 2 * w0 + 2 * h0 + self.vt_spec.mass_lb + self.fusV.base_mass_lb
        d["total_frac"] = d["total_lb"] / d["baseline_flexible_lb"]
        return d

    def wing_mass_centroid_y(self, side: str) -> float:
        s = self.wingR if side == "R" else self.wingL
        mm = s.m * s.dy
        return float(np.sum(mm * s.y) / np.sum(mm))

    def frequencies_hz(self) -> Dict[str, List[float]]:
        return {nm: (self.omega[self.slices[nm]] / (2 * math.pi)).tolist() for nm in BODY_NAMES}

    def mode_classes(self) -> Dict[str, List[str]]:
        out = {}
        for nm, b in (("wingR", self.wingR), ("wingL", self.wingL), ("htR", self.htR), ("htL", self.htL), ("vt", self.vt)):
            out[nm] = b.cls.tolist()
        out["fusV"] = out["fusL"] = ["b", "b"]
        return out


# =====================================================================================================================
# 6. Margins (flutter, divergence, control reversal) per body block
# =====================================================================================================================
def _block(mdl: FlexBodyModel, bodies: Sequence[str]) -> np.ndarray:
    return np.concatenate([np.arange(mdl.N)[mdl.slices[b]] for b in bodies])


def _q_of_keas(v): return 0.5 * RHO0 * (v * KT2FPS) ** 2           # noqa: E704
def _keas_of_q(q): return math.sqrt(2 * q / RHO0) / KT2FPS        # noqa: E704


class BlockAero:
    """Restriction of the global model to a set of DOF (one body block). M = I (mass-normalised modes)."""

    def __init__(self, mdl: FlexBodyModel, idx: np.ndarray, rho: float = RHO0, compressible: bool = True):
        self.mdl, self.idx, self.rho, self.compressible = mdl, idx, rho, compressible
        ix = np.ix_(idx, idx)
        self.K, self.C = mdl.K[ix], mdl.C[ix]
        self.AK = {g: mdl.A_K[g][ix] for g in GROUPS}
        self.AC = {g: mdl.A_C[g][ix] for g in GROUPS}
        self.Anc = mdl.A_C_nc[ix]
        self.a_snd = fw.speed_of_sound_fps(rho)
        self.n = idx.size

    def kap_q(self, q_eas: float) -> Dict[str, float]:
        if not self.compressible:
            return {g: 1.0 for g in GROUPS}
        return self.mdl.kappas(math.sqrt(2 * q_eas / self.rho) / self.a_snd)

    def AKq(self, q):
        k = self.kap_q(q)
        return k["w"] * self.AK["w"] + k["h"] * self.AK["h"] + k["v"] * self.AK["v"]

    def growth(self, v_keas: float):
        ve = v_keas * KT2FPS
        q = 0.5 * RHO0 * ve * ve
        vt = ve * math.sqrt(RHO0 / self.rho)
        k = self.kap_q(q)
        n = self.n
        Ke = self.K - q * (k["w"] * self.AK["w"] + k["h"] * self.AK["h"] + k["v"] * self.AK["v"])
        Ce = self.C - (q / vt) * (k["w"] * self.AC["w"] + k["h"] * self.AC["h"] + k["v"] * self.AC["v"] + self.Anc)
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Ke
        A[n:, n:] = -Ce
        lam = np.linalg.eigvals(A)
        osc = lam[np.abs(lam.imag) > 1e-6]
        g_osc = osc.real.max() if osc.size else -math.inf
        f = abs(osc[np.argmax(osc.real)].imag) / (2 * math.pi) if osc.size else 0.0
        return g_osc, f

    def flutter_keas(self, v_max: float, n_grid: int = 120) -> Tuple[float, float]:
        """Quasi-steady p-method EAS sweep (as v1): first speed with an oscillatory root Re > 0; 40-step bisection."""
        prev = 0.0
        for v in np.linspace(1.0, v_max, n_grid):
            if self.growth(v)[0] > 0:
                lo, hi = prev, v
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if self.growth(mid)[0] > 0: hi = mid
                    else: lo = mid
                return hi, self.growth(hi)[1]
            prev = v
        return math.inf, 0.0

    def coalescence_q(self, q_max: float, n_grid: int = 400, zeta: Optional[float] = None) -> float:
        """Steady-aero frequency coalescence: first q where eig(K - q A(q)) turn complex (as v1). With zeta, a coalesced
        pair only counts once its steady-aero growth rate |Im s| exceeds the structural damping zeta*|Re s|
        (s = sqrt(eig)); this removes the spurious 'instant' coalescence of near-degenerate, weakly coupled modes."""
        def cplx(q):
            ev = np.linalg.eigvals(self.K - q * self.AKq(q))
            c = np.abs(ev.imag) > 1e-6 * np.abs(ev).max()
            if zeta is None or not np.any(c):
                return bool(np.any(c))
            s = np.sqrt(ev[c].astype(complex))
            return bool(np.any(np.abs(s.imag) > zeta * np.abs(s.real)))
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

    def divergence_q(self) -> float:
        """Smallest q with det(K - q A(q)) = 0 (kappa fixed point, as v1)."""
        def qd(A):
            mu = np.linalg.eigvals(np.linalg.solve(self.K, A))
            mu = mu[(np.abs(mu.imag) < 1e-9 * max(1.0, np.abs(mu).max())) & (mu.real > 0)].real
            return float(1 / mu.max()) if mu.size else math.inf
        q = qd(self.AK["w"] + self.AK["h"] + self.AK["v"])
        if not self.compressible or not math.isfinite(q):
            return q
        for _ in range(30):
            q = qd(self.AKq(q))
            if not math.isfinite(q):
                break
        return q

    def effectiveness(self, q: float, b: np.ndarray, r_rigid: float, rG: np.ndarray, gk: str, incidence: bool = False) -> float:
        """e(q) = (rigid + elastic) / rigid = 1 + kappa rG eta / r_rigid with (K - q A(q)) eta = q b (x kappa when the control
        acts as a rigid incidence change, i.e. all-moving tail: then b and r_rigid are per q*kappa*delta)."""
        k = self.kap_q(q)
        lk = k[gk] if incidence else 1.0
        eta = np.linalg.solve(self.K - q * (k["w"] * self.AK["w"] + k["h"] * self.AK["h"] + k["v"] * self.AK["v"]), q * lk * b)
        return 1.0 + k[gk] * float(rG @ eta) / (lk * r_rigid)

    def reversal_q(self, b, r_rigid, rG, gk, q_max, n_grid: int = 240, incidence: bool = False) -> float:
        """First q where e(q) crosses zero before divergence (a sign change across the divergence pole is not reversal)."""
        qs = np.concatenate([[0.0], np.geomspace(q_max * 1e-3, q_max, n_grid)])
        det_sign = lambda q: np.linalg.slogdet(self.K - q * self.AKq(q))[0]  # noqa: E731
        e_prev, s_prev = 1.0, det_sign(0.0)
        for i in range(1, qs.size):
            q = qs[i]
            s = det_sign(q)
            if s != s_prev:
                return math.inf
            e = self.effectiveness(q, b, r_rigid, rG, gk, incidence)
            if e <= 0.0 < e_prev:
                lo, hi = qs[i - 1], q
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if self.effectiveness(mid, b, r_rigid, rG, gk, incidence) > 0: lo = mid
                    else: hi = mid
                return float(hi)
            e_prev, s_prev = e, s
        return math.inf


def control_vectors(mdl: FlexBodyModel, which: str, idx: np.ndarray):
    """(b, r_rigid, rG, group) for 'aileron' (right-wing rolling moment), 'elevator' (HT lift), 'rudder' (fin side force).
    For all-moving tails the elevator is a rigid incidence change of the whole tail (b, r per unit q*kappa_h*delta)."""
    st = mdl.st
    lw = st["lw"]
    if which == "aileron":
        r = mdl.body_strips["wingR"]
        wt = np.zeros(mdl.ns); wt[r] = st["y"][r]
        key, gk = "q_dail_R", "w"
    elif which == "elevator":
        wt = (st["group"] == 1).astype(float)
        key, gk = ("qkh_alpha_ht" if mdl.ht_spec.all_moving else "q_delev"), "h"
    elif which == "rudder":
        wt = (st["group"] == 2).astype(float)
        key, gk = "q_drud", "v"
    elif which == "ht_alpha":            # tail lift-curve effectiveness: rigid incidence change of the whole HT
        wt = (st["group"] == 1).astype(float)
        key, gk = "qkh_alpha_ht", "h"
    elif which == "vt_alpha":            # fin side-force effectiveness: rigid sideslip at the fin
        wt = (st["group"] == 2).astype(float)
        key, gk = "qkv_alpha_vt", "v"
    else:
        raise ValueError(which)
    b = mdl.Qbasis[CI[key], idx]
    r_rig = float(wt @ mdl.Lb[CI[key]])
    rG = ((wt * lw) @ mdl.G)[idx]
    return b, r_rig, rG, gk, key in ("qkh_alpha_ht", "qkv_alpha_vt")


BLOCKS = {"wingR": ("wingR",), "wingL": ("wingL",), "empennage_pitch": ("htR", "htL", "fusV"), "empennage_yaw": ("vt", "fusL")}
BLOCK_CONTROL = {"wingR": "aileron", "wingL": None, "empennage_pitch": "elevator", "empennage_yaw": "rudder"}
BLOCK_TAIL_EFF = {"empennage_pitch": "ht_alpha", "empennage_yaw": "vt_alpha"}


def margins_v2(mdl: FlexBodyModel, rho: float = RHO0, cap: float = MARGIN_CAP, blocks: Optional[Sequence[str]] = None) -> Dict:
    """Flutter / divergence / control-reversal margins (x V_D, EAS) per body block, capped at `cap` with flags.
    Overall values = conservative min over blocks. NaN / LinAlgError -> margin 0 + margin_error (hard fail)."""
    vd = mdl.pw.v_dive_keas
    qd = _q_of_keas(vd)
    v_cap = cap * vd
    if blocks is None:
        blocks = ["wingR", "empennage_pitch", "empennage_yaw"] + (["wingL"] if mdl.wingL is not mdl.wingR else [])
    out = {"margin_cap": float(cap), "v_dive_keas": float(vd), "q_dive_psf": qd, "blocks": {}}
    err_any = False
    to_v = lambda q: _keas_of_q(q) if (math.isfinite(q) and q >= 0) else q  # noqa: E731
    for bname in blocks:
        idx = _block(mdl, BLOCKS[bname])
        B = BlockAero(mdl, idx, rho)
        err = [False]

        def fin(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                err[0] = True
                return 0.0, True
            if not math.isfinite(v) or v >= v_cap:
                return float(cap), False
            return float(v / vd), True
        try:
            v_fl, f_fl = B.flutter_keas(v_cap)
        except (np.linalg.LinAlgError, ValueError):
            v_fl, f_fl = float("nan"), 0.0
        try:
            q_co = B.coalescence_q(cap ** 2 * qd, zeta=mdl.genes["struct_damping_ratio"])
            q_co0 = B.coalescence_q(cap ** 2 * qd)             # undamped (v1 criterion), diagnostic only
        except (np.linalg.LinAlgError, ValueError):
            q_co = q_co0 = float("nan")
        try:
            q_dv = B.divergence_q()
        except (np.linalg.LinAlgError, ValueError):
            q_dv = float("nan")
        m_qs, f_qs = fin(v_fl)
        m_co, f_co = fin(to_v(q_co))
        m_dv, f_dv = fin(to_v(q_dv))
        m_co0, _ = fin(to_v(q_co0))
        res = {"flutter_margin_qs": m_qs, "coalescence_margin": m_co, "coalescence_margin_undamped": m_co0,
               "flutter_margin": min(m_qs, m_co),
               "flutter_not_found_below_cap": not (f_qs or f_co), "f_flutter_hz": float(f_fl) if f_qs else 0.0,
               "div_margin": m_dv, "div_not_found_below_cap": not f_dv}
        ctrl = BLOCK_CONTROL.get(bname)
        if ctrl is not None:
            try:
                b, r_rig, rG, gk, inc = control_vectors(mdl, ctrl, idx)
                if abs(r_rig) < 1e-12:
                    raise ValueError("no control load")
                q_r = B.reversal_q(b, r_rig, rG, gk, cap ** 2 * qd, incidence=inc)
                e_vd = B.effectiveness(qd, b, r_rig, rG, gk, inc)
            except (np.linalg.LinAlgError, ValueError):
                q_r, e_vd = float("nan"), float("nan")
            m_r, f_r = fin(to_v(q_r))
            res.update({f"{ctrl}_reversal_margin": m_r, f"{ctrl}_reversal_not_found_below_cap": not f_r,
                        f"{ctrl}_effectiveness_at_VD": float(e_vd) if math.isfinite(e_vd) else 0.0})
            if not math.isfinite(e_vd):
                err[0] = True
        tail = BLOCK_TAIL_EFF.get(bname)
        if tail is not None:          # tail / fin lift effectiveness at V_D (elastic / rigid, incl. fuselage bending)
            try:
                b, r_rig, rG, gk, inc = control_vectors(mdl, tail, idx)
                e_t = B.effectiveness(qd, b, r_rig, rG, gk, inc)
            except (np.linalg.LinAlgError, ValueError):
                e_t = float("nan")
            res[f"{tail}_effectiveness_at_VD"] = float(e_t) if math.isfinite(e_t) else 0.0
            if not math.isfinite(e_t):
                err[0] = True
        res["margin_error"] = err[0]
        err_any = err_any or err[0]
        out["blocks"][bname] = res
    bl = list(out["blocks"].values())
    out["flutter_margin"] = float(min(b["flutter_margin"] for b in bl))
    out["div_margin"] = float(min(b["div_margin"] for b in bl))
    rev = [v for b in bl for k, v in b.items() if k.endswith("_reversal_margin")]
    out["reversal_margin"] = float(min(rev)) if rev else float(cap)
    out["margin_error"] = bool(err_any)
    out["f_modes_hz"] = mdl.frequencies_hz()
    return out


# =====================================================================================================================
# 7. Coupler (JSBSim)
# =====================================================================================================================
class FlexBodyCoupler:
    """25-DOF modal structure driven by the JSBSim state every frame; elastic aero increments (relative to the 1-g trim
    shape) fed back through external_reactions flexwing_F (BODY, at AERORP: x, y, z) and flexwing_M (l, m, n)."""
    FORCE = fw.FlexCoupler.FORCE
    MOMENT = fw.FlexCoupler.MOMENT

    def __init__(self, mdl: FlexBodyModel, mode: str = "twoway", substeps: int = 2, zero_feedback: bool = False):
        assert mode in ("oneway", "twoway")
        self.mdl, self.mode, self.substeps = mdl, mode, int(substeps)
        self.zero_feedback = zero_feedback      # tests: structure integrated, feedback written as exact zeros
        pw = mdl.pw
        self.ail_antisym, self.ail_norm = pw.ail_antisym_prop, pw.ail_norm_deg
        h, v = mdl.params["v2"]["ht"], mdl.params["v2"]["vt"]
        self.elev_prop, self.elev_k = h["elev_prop"], (math.radians(h["norm_deg"]) if h["norm_deg"] else 1.0)
        self.rud_prop, self.rud_k = v["rud_prop"], (math.radians(v["norm_deg"]) if v["norm_deg"] else 1.0)
        self.last: Dict[str, float] = {}
        self.ref: Dict[str, float] = {}

    def read_state(self, fdm) -> Dict[str, float]:
        g = fdm.__getitem__
        alpha = g("aero/alpha-rad")
        fbx, fbz = g("forces/fbx-aero-lbs"), g("forces/fbz-aero-lbs")
        d = dict(qbar=g("aero/qbar-psf"), vt=max(g("velocities/vt-fps"), 1.0), alpha=alpha, beta=g("aero/beta-rad"),
                 mach=g("velocities/mach"), p=g("velocities/p-rad_sec"), q=g("velocities/q-rad_sec"), r=g("velocities/r-rad_sec"),
                 pdot=g("accelerations/pdot-rad_sec2"), qdot=g("accelerations/qdot-rad_sec2"), rdot=g("accelerations/rdot-rad_sec2"),
                 nz=g("accelerations/Nz"), ny=g("accelerations/Ny"), nx=g("accelerations/Nx"),
                 lift=-fbz * math.cos(alpha) + fbx * math.sin(alpha), drag=-fbx * math.cos(alpha) - fbz * math.sin(alpha),
                 elev=g(self.elev_prop) * self.elev_k, rud=g(self.rud_prop) * self.rud_k)
        if self.ail_antisym is not None:
            a = g(self.ail_antisym)
            d["ail_r"], d["ail_l"] = -a, a
        elif self.ail_norm is not None:
            k = math.radians(self.ail_norm)
            d["ail_r"], d["ail_l"] = g("fcs/right-aileron-pos-norm") * k, g("fcs/left-aileron-pos-norm") * k
        else:
            d["ail_r"], d["ail_l"] = g("fcs/right-aileron-pos-rad"), g("fcs/left-aileron-pos-rad")
        return d

    def coefs(self, s, kap) -> np.ndarray:
        """Load-basis coefficients (COEF_NAMES order). Tail/fin rigid loads are perturbations from trim (the trim tail
        load itself is not known to the coupler and is excluded)."""
        m, ref = self.mdl, self.ref
        q, V = s["qbar"], s["vt"]
        a_ht = (s["alpha"] - ref["alpha"]) * (1 - m.downwash) + s["q"] * m.l_h / V
        de = s["elev"] - ref["elev"]
        if m.ht_spec.all_moving:
            a_ht += de
        a_vt = -(s["beta"] - ref["beta"]) + s["r"] * m.l_v / V - s["p"] * m.z_v / V
        return np.array([s["lift"], s["nz"], q * kap["w"] * s["p"] / V, q * s["ail_r"], q * s["ail_l"], s["pdot"], s["qdot"],
                         q * kap["h"] * a_ht, q * de, q * kap["v"] * a_vt, q * (s["rud"] - ref["rud"]), s["ny"], s["rdot"],
                         s["nx"], s["drag"], q * kap["h"] * s["p"] / V])

    def initialize(self, fdm):
        """Static equilibrium at the trimmed state (K eta = Q): zero elastic feedback at t0."""
        m = self.mdl
        s = self.read_state(fdm)
        self.ref = {"alpha": s["alpha"], "elev": s["elev"], "rud": s["rud"], "beta": s["beta"]}
        kap = m.kappas(s["mach"])
        Q = self.coefs(s, kap) @ m.Qbasis
        eta = np.linalg.solve(m.K, Q)
        self.eta, self.etad, self.etadd = eta.copy(), np.zeros(m.N), np.zeros(m.N)
        self.eta_ref = eta.copy()
        self.Qref_per_q = {g: m.A_K[g] @ eta for g in GROUPS}
        self._apply(fdm, s, kap)
        self.out_1g = self.out.copy()

    def step(self, fdm, dt: float):
        m = self.mdl
        s = self.read_state(fdm)
        kap = m.kappas(s["mach"])
        q, V = s["qbar"], s["vt"]
        hh = dt / self.substeps
        c1, c2, c3 = 4 / hh ** 2, 4 / hh, 2 / hh
        qw, qh, qv = q * kap["w"], q * kap["h"], q * kap["v"]
        AK = qw * m.A_K["w"] + qh * m.A_K["h"] + qv * m.A_K["v"]
        Ce = m.C - (q / V) * (kap["w"] * m.A_C["w"] + kap["h"] * m.A_C["h"] + kap["v"] * m.A_C["v"] + m.A_C_nc)
        Keff_inv = np.linalg.inv(m.K - AK + c3 * Ce + c1 * m.M)
        de = self.eta - self.eta_ref
        Q = (self.coefs(s, kap) @ m.Qbasis - (qw * self.Qref_per_q["w"] + qh * self.Qref_per_q["h"] + qv * self.Qref_per_q["v"])
             + (qw * s["beta"]) * (m.A_beta @ de))
        KQ, KM, KC = Keff_inv @ Q, Keff_inv @ m.M, Keff_inv @ Ce
        x, v, a = self.eta, self.etad, self.etadd
        for _ in range(self.substeps):
            x1 = KQ + KM @ (c1 * x + c2 * v + a) + KC @ (c3 * x + v)
            v1 = c3 * (x1 - x) - v
            a = c1 * (x1 - x) - c2 * v - a
            x, v = x1, v1
        self.eta, self.etad, self.etadd = x, v, a
        self._apply(fdm, s, kap)

    def _apply(self, fdm, s, kap):
        m = self.mdl
        q, V, beta = s["qbar"], s["vt"], s["beta"]
        de = self.eta - self.eta_ref
        e = de @ m.R_de + (self.etad @ m.R_d) / V
        Pb = de @ m.R_beta
        kw, kh, kv = q * kap["w"], q * kap["h"], q * kap["v"]
        Lw, Rw, Pw = kw * (e[0] + beta * Pb[0]), kw * (e[1] + beta * Pb[1]), kw * (e[2] + beta * Pb[2])
        dL = Lw + kh * e[3]
        dY = kv * e[6]
        dl = Rw + kh * e[4] + kv * e[7]
        dm = Pw + kh * e[5]
        dn = kv * e[8]
        cf = self.coefs(s, kap)
        qg = np.array([kw, kh, kv])[m.out_group]
        out = cf @ m.RB + qg * (de @ m.RE) + qg * (self.etad @ m.RD) / V + self.etadd @ m.RA
        self.out = out
        tips = self.eta @ m.TIP
        r2d = 180.0 / math.pi
        last = {nm: float(out[i]) for i, nm in enumerate(OUT_NAMES)}
        for i, k in enumerate(m.tip_rows):
            last[k] = float(tips[i])
        for k in ("tip_twist_R", "tip_twist_L", "ht_incidence", "vt_sideslip"):
            last[k + "_deg"] = last.pop(k) * r2d
        last.update(dL_lbf=dL, dY_lbf=dY, dRoll_lbft=dl, dPitch_lbft=dm, dYaw_lbft=dn)
        tt = self.eta @ m.TIP_TEL            # telemetry-only outputs (never read by the cost, margins or feedback)
        for i, k in enumerate(m.tip_rows_tel):
            if k.endswith("_twist"):
                last[k + "_deg"] = float(tt[i]) * r2d
            else:
                last[k] = float(tt[i])
        self.last = last
        if self.mode == "twoway":
            a = s["alpha"]
            F, Mo = self.FORCE, self.MOMENT
            if self.zero_feedback:
                dL = dY = dl = dm = dn = 0.0
            fdm[f"external_reactions/{F}/magnitude"] = 1.0
            fdm[f"external_reactions/{F}/x"] = dL * math.sin(a)
            fdm[f"external_reactions/{F}/y"] = dY
            fdm[f"external_reactions/{F}/z"] = -dL * math.cos(a)
            fdm[f"external_reactions/{Mo}/magnitude-lbsft"] = 1.0
            fdm[f"external_reactions/{Mo}/l"] = dl
            fdm[f"external_reactions/{Mo}/m"] = dm
            fdm[f"external_reactions/{Mo}/n"] = dn


DIAG_KEYS_V2 = list(OUT_NAMES) + ["tip_w_ft_R", "tip_w_ft_L", "tip_twist_R_deg", "tip_twist_L_deg", "ht_tip_w_ft", "vt_tip_w_ft",
                                  "fusV_tip_w_ft", "fusL_tip_w_ft", "ht_incidence_deg", "vt_sideslip_deg",
                                  "dL_lbf", "dY_lbf", "dRoll_lbft", "dPitch_lbft", "dYaw_lbft"]
# telemetry-only scalars (in coupler.last every frame; recorded into FlexBodyFDM histories only with telemetry=True).
# Signs: see INTERFACE_v2.md 'Telemetry sign conventions' (probed by test_flexbody.py::test_sign_probe_*).
TELEMETRY_KEYS_V2 = ["htL_tip_w_ft", "ht_tip_twist_deg", "htL_tip_twist_deg", "vt_tip_twist_deg", "wingR_tip_ip_ft",
                     "wingL_tip_ip_ft"]


class FlexBodyFDM:
    """Proxy around FGFDMExec (like coupled_sim.FlexFDM): run() = structure step + feedback, then the JSBSim step."""

    def __init__(self, fdm, coupler: Optional[FlexBodyCoupler], dt: float, record: bool = True, telemetry: bool = False):
        self._fdm, self.coupler, self.dt, self.record = fdm, coupler, dt, record
        self.telemetry = bool(telemetry)     # + TELEMETRY_KEYS_V2 histories and the modal state per frame (node exports)
        self.started = False
        self._keys = DIAG_KEYS_V2 + (TELEMETRY_KEYS_V2 if self.telemetry else [])
        self.hist = {k: [] for k in self._keys}
        self.eta_hist: List[np.ndarray] = []
        self.out_1g = None

    def __getitem__(self, k):
        return self._fdm[k]

    def __setitem__(self, k, v):
        self._fdm[k] = v

    def __getattr__(self, name):
        return getattr(self._fdm, name)

    def run(self):
        c = self.coupler
        if c is not None:
            if not self.started:
                c.initialize(self._fdm)
                self.out_1g = c.out_1g.copy()
                self.started = True
            c.step(self._fdm, self.dt)
            if self.record:
                L = c.last
                for k in self._keys:
                    self.hist[k].append(L[k])
            if self.telemetry:
                self.eta_hist.append(np.array(c.eta, copy=True))
        return self._fdm.run()

    def history(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.hist.items()}


class _DictFDM(dict):
    """Minimal property store standing in for FGFDMExec in the sign probes (reads default to 0.0)."""

    def __getitem__(self, k):
        return dict.get(self, k, 0.0)


_PROBE_STATE = {"aero/qbar-psf": 50.0, "velocities/vt-fps": 150.0, "velocities/mach": 0.13}


def static_probe(mdl: FlexBodyModel, props: Dict[str, float]) -> Dict[str, float]:
    """Deterministic sign probe: static structural response K eta = Q to ONE physical JSBSim input (everything else 0,
    trim reference alpha = beta = elevator = rudder = 0, qbar 50 psf, V 150 ft/s), elastic feedback of that shape
    (eta_ref = 0), -> coupler.last (all diagnostics + telemetry scalars). Used by the tests / INTERFACE_v2 sign table."""
    c = FlexBodyCoupler(mdl, mode="oneway")
    fdm = _DictFDM(_PROBE_STATE)
    fdm.update(props)
    st = c.read_state(fdm)
    c.ref = {"alpha": 0.0, "elev": 0.0, "rud": 0.0, "beta": 0.0}
    kap = mdl.kappas(st["mach"])
    c.eta = np.linalg.solve(mdl.K, c.coefs(st, kap) @ mdl.Qbasis)
    c.eta_ref, c.etad, c.etadd = np.zeros(mdl.N), np.zeros(mdl.N), np.zeros(mdl.N)
    c._apply(fdm, st, kap)
    return dict(c.last)


def mode_probe(mdl: FlexBodyModel, body: str, cls: str, alpha_rad: float = 0.1) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Unit elastic deflection of the first `cls` ('b' bending / 't' torsion / 'v' in-plane) mode of `body`, sign fixed
    so the tip value of that field is POSITIVE (FD's mode-sign rule), fed through a TWO-WAY coupler at alpha_rad
    -> (coupler.last, the external_reactions values written to the FDM)."""
    c = FlexBodyCoupler(mdl, mode="twoway")
    fdm = _DictFDM(_PROBE_STATE)
    fdm["aero/alpha-rad"] = alpha_rad
    st = c.read_state(fdm)
    c.ref = {"alpha": alpha_rad, "elev": 0.0, "rud": 0.0, "beta": 0.0}
    b = getattr(mdl, body)
    j = list(b.cls).index(cls) if hasattr(b, "cls") else 0
    eta = np.zeros(mdl.N)
    eta[np.arange(mdl.N)[mdl.slices[body]][j]] = 1e-3
    c.eta, c.eta_ref, c.etad, c.etadd = eta, np.zeros(mdl.N), np.zeros(mdl.N), np.zeros(mdl.N)
    c._apply(fdm, st, mdl.kappas(st["mach"]))
    F, Mo = c.FORCE, c.MOMENT
    er = {k: float(fdm[f"external_reactions/{F}/{k}"]) for k in ("x", "y", "z")}
    er.update({k: float(fdm[f"external_reactions/{Mo}/{k}"]) for k in ("l", "m", "n")})
    return dict(c.last), er


NODE_BODIES = BODY_NAMES          # wingR, wingL, htR, htL, vt, fusV, fusL
NODE_FIELDS = {"w": "w_ft", "t": "theta_deg", "v": "v_ft"}


def _node_matrices(mdl: FlexBodyModel) -> Dict[str, Dict[str, np.ndarray]]:
    """Per body and field: (n_el + 1, n_modes_body) matrix eta_body -> FE nodal value (node 0 = clamped root = 0).
    Built lazily (telemetry only) and cached on the model."""
    cache = getattr(mdl, "_node_mats", None)
    if cache is not None:
        return cache
    out = {}
    for nm in NODE_BODIES:
        body = getattr(mdl, nm)
        b = body.beam
        nn = b.n_el + 1
        N = nn * b.nd
        mats = {}
        for fld in b.fields:
            if fld not in NODE_FIELDS or (fld == "t" and not b.has_t) or (fld == "v" and not b.has_v):
                continue
            S = np.zeros((nn, N))
            for i in range(nn):
                S[i, b._g(i, fld)] = 1.0
            mats[NODE_FIELDS[fld]] = S[:, b.free] @ body.Phi * (180.0 / math.pi if fld == "t" else 1.0)
        out[nm] = mats
    mdl._node_mats = out
    return out


def node_values(mdl: FlexBodyModel, eta) -> Dict[str, Dict[str, np.ndarray]]:
    """FE nodal values of every body from the global modal state eta ((N,) or (T, N)) -> {body: {field: (.., n_nodes)}}.
    Fields (FD sign conventions, own elastic deflection relative to the body's clamped root, incl. the 1-g trim shape):
      w_ft      wings / HT / fusV: + up (body -z); vt / fusL: + toward body +y (right)
      theta_deg wings / HT: + leading edge up (nose-up, increases local alpha, both sides); vt: + leading edge toward +y
      v_ft      wings only (in-plane): + aft (body -x)
    HT strips additionally ride on the fusV tip (plunge fusV.w[-1], incidence ht_incidence = -w'_fusV,tip) and VT strips
    on the fusL tip (lateral fusL.w[-1], incidence vt_sideslip = -w'_fusL,tip); those are NOT added here."""
    e = np.asarray(eta, float)
    mats = _node_matrices(mdl)
    out = {}
    for nm, fm in mats.items():
        eb = e[..., mdl.slices[nm]]
        out[nm] = {f: eb @ M.T for f, M in fm.items()}
    return out


def node_layout(mdl: FlexBodyModel, rp_offset_body_ft: Optional[Sequence[float]] = None) -> List[Dict]:
    """Undeformed elastic-axis FE node coordinates per body, body FRD (x fwd, y right, z down) in ft. Origin = AERORP
    (where flexwing_F acts) unless rp_offset_body_ft = AERORP position in body FRD relative to the CG is given (then
    origin = CG, the trajectory frame). Same geometry the strip arms use: quarter chord swept through the MAC quarter
    chord (wing: at the AERORP; HT at l_h, VT at l_v aft), EA at x_ea of the local chord; wing / HT z = AERORP z
    (dihedral and HT height not modelled); VT height above the AERORP = z_v + (y - y_mac); aft fuselage from the wing
    station to l_h aft. Approximate, for display: the physics only uses the arms."""
    off = np.zeros(3) if rp_offset_body_ft is None else np.asarray(rp_offset_body_ft, float)
    comps = []
    wy_mac = mdl.pw.span_ft / 2 / 3 * (1 + 2 * mdl.pw.taper) / (1 + mdl.pw.taper)
    for nm in NODE_BODIES:
        body = getattr(mdl, nm)
        b = body.beam
        nn = b.n_el + 1
        st = np.arange(nn) * b.h                    # beam coordinate from the clamped root
        if nm in ("fusV", "fusL"):
            xyz = np.stack([-st, np.zeros(nn), np.zeros(nn)], axis=1)
            fields = {"w_ft": "+ up (body -z)" if nm == "fusV" else "+ toward body +y (right)"}
            root = "wing station (AERORP x), clamped; tail at l_h aft"
        else:
            sp = body.sp
            y = body.y0 + st
            c = body.c_root * (1 - (1 - sp.taper) * y / body.s)
            if nm in ("wingR", "wingL"):
                aft = (y - wy_mac) * math.tan(body.lam)
            else:
                y_mac = body.s / 3 * (1 + 2 * sp.taper) / (1 + sp.taper)
                aft = (mdl.l_h if nm in ("htR", "htL") else mdl.l_v) + (y - y_mac) * math.tan(body.lam)
            aft = aft + (sp.x_ea - 0.25) * c
            if nm == "vt":
                y_mac = body.s / 3 * (1 + 2 * sp.taper) / (1 + sp.taper)
                xyz = np.stack([-aft, np.zeros(nn), -(mdl.z_v + (y - y_mac))], axis=1)
                fields = {"w_ft": "+ toward body +y (right)", "theta_deg": "+ leading edge toward +y"}
            else:
                sg = -1.0 if nm.endswith("L") else 1.0
                xyz = np.stack([-aft, sg * y, np.zeros(nn)], axis=1)
                fields = {"w_ft": "+ up (body -z)", "theta_deg": "+ leading edge up (nose-up), both sides"}
                if b.has_v:
                    fields["v_ft"] = "+ aft (body -x)"
            root = "clamped at the beam root (station 0)" + ("; rides on the fusV tip" if nm in ("htR", "htL") else
                                                             "; rides on the fusL tip" if nm == "vt" else "")
        xyz = xyz + off[None, :]
        comps.append({"name": nm, "axis_nodes_body_ft": [[round(float(v), 6) for v in p] for p in xyz],
                      "node_station_ft": [round(float(v), 6) for v in st],
                      "node_span_frac": [round(float(v), 6) for v in st / b.L], "fields": fields, "root": root})
    return comps


# =====================================================================================================================
# 8. Prepared aircraft (jsbsim_root_v2) and mass feedback
# =====================================================================================================================
V2_POINTMASSES = ("flexbody_ht", "flexbody_vt", "flexbody_fus")


def prepare_aircraft_v2(model: str, root_v2: str = ROOT_V2) -> str:
    """v1 preparation into root_v2 (sockets stripped, flexwing_F/M, wing point masses) + 3 zero-weight point masses for the
    tail / fin / aft-fuselage mass changes at fixed notional locations. Writes flexbody_meta.json (incl. metrics).
    Never touches jsbsim_root (v1) or the installed JSBSim data."""
    import jsbsim
    fw.prepare_aircraft(model, root_v2)
    dst = os.path.join(root_v2, "aircraft", model)
    xml = os.path.join(dst, model + ".xml")
    with open(xml, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    with open(os.path.join(dst, "flexwing_meta.json")) as f:
        meta1 = json.load(f)
    fdm = fw.new_fdm(model, root_v2)
    geom = Geometry(bw_ft=fdm["metrics/bw-ft"], sw_ft2=fdm["metrics/Sw-sqft"], sh_ft2=fdm["metrics/Sh-sqft"],
                    lh_ft=fdm["metrics/lh-ft"], sv_ft2=fdm["metrics/Sv-sqft"], lv_ft=fdm["metrics/lv-ft"],
                    empty_wt_lb=fdm["inertia/empty-weight-lbs"])
    del fdm
    pr = V2_PROFILES[model]
    lv = pr["vt"]["arm_ft"] or geom.lv_ft
    x, _, z = meta1["aerorp_in"]
    locs = {"flexbody_ht": (x + 12 * geom.lh_ft, z), "flexbody_vt": (x + 12 * lv, z + 12 * pr["vt"]["z_ft"]),
            "flexbody_fus": (x + 6 * geom.lh_ft, z)}
    mb = re.search(r"<mass_balance[^>]*>(.*?)</mass_balance>", txt, re.S)
    n_pm = len(re.findall(r"<pointmass\b", mb.group(1)))
    pm = "".join(f"""
        <pointmass name="{nm}">
            <weight unit="LBS"> 0.0 </weight>
            <location unit="IN"> <x> {locs[nm][0]:.4f} </x> <y> 0.0 </y> <z> {locs[nm][1]:.4f} </z> </location>
        </pointmass>""" for nm in V2_POINTMASSES)
    txt = txt[:mb.end(1)] + pm + "\n    " + txt[mb.end(1):]
    with open(xml, "w", encoding="utf-8") as f:
        f.write(txt)
    meta = {"fmt": PREPARE_FMT_V2, "v1_meta": meta1, "geom": asdict(geom),
            "pm_index": {nm: n_pm + i for i, nm in enumerate(V2_POINTMASSES)}, "pm_loc_in": locs,
            "jsbsim_version": jsbsim.__version__}
    with open(os.path.join(dst, "flexbody_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return root_v2


def ensure_root_v2(model: str, root_v2: str = ROOT_V2) -> str:
    p = os.path.join(root_v2, "aircraft", model, "flexbody_meta.json")
    ok = False
    if os.path.exists(p):
        with open(p) as f:
            m = json.load(f)
        ok = m.get("fmt", 0) >= PREPARE_FMT_V2 and m["v1_meta"].get("fmt", 0) >= fw.PREPARE_FMT
    if not ok:
        prepare_aircraft_v2(model, root_v2)
    return root_v2


def apply_mass_v2(fdm, mdl: FlexBodyModel, root_v2: str = ROOT_V2) -> Dict[str, float]:
    """Push all structural mass changes into JSBSim (weight, CG, inertia) via the point masses. Call BEFORE run_ic."""
    meta = load_meta(mdl.model, root_v2)
    m1 = meta["v1_meta"]
    ms = mdl.mass_summary()
    for idx, side, sg in ((m1["pm_index_R"], "R", 1.0), (m1["pm_index_L"], "L", -1.0)):
        fdm[f"inertia/pointmass-weight-lbs[{idx}]"] = ms["wing" + side + "_lb"]
        fdm[f"inertia/pointmass-location-Y-inches[{idx}]"] = sg * mdl.wing_mass_centroid_y(side) * 12.0
    for nm, key in (("flexbody_ht", "ht_lb"), ("flexbody_vt", "vt_lb"), ("flexbody_fus", "fus_lb")):
        fdm[f"inertia/pointmass-weight-lbs[{meta['pm_index'][nm]}]"] = ms[key]
    return ms


# =====================================================================================================================
# 9. Cost terms (v2)
# =====================================================================================================================
@dataclass
class StructWeightsV2(fw.StructWeights):
    w_reversal: float = 1.0
    w_tail_bm_peak: float = 1.0
    w_fus_bm_peak: float = 1.0
    w_smooth: float = 0.05
    tail_design_alpha_deg: float = 5.0    # notional design incidence for the tail/fin allowable root moments
    # limit-load sizing checks (pre-flight, design demand vs allowable ~ stiffness gene; 2026-10-06 mass-exploit fix)
    w_wing_bm_limit: float = 1.0
    w_tail_bm_limit: float = 1.0
    w_fus_bm_limit: float = 1.0
    w_wing_torque_limit: float = 0.1
    w_wing_ip_limit: float = 0.1
    # flown peak over the wing torque / in-plane allowables (post-flight)
    w_wing_torque_peak: float = 0.1
    w_wing_ip_peak: float = 0.1
    design_margin_of_safety: float = 0.0  # allowable = (1 + MS) x baseline design load x stiffness gene
    design_ctrl_deg: float = 5.0          # aileron deflection at q_D for the hinge-moment part of the design torque
    torque_floor_chord_frac: float = 0.05  # design torque >= this x MAC x semi-span design lift (never ~0)
    ip_design_cd: float = 0.03            # in-plane design drag coefficient at q_D (on S_w)
    ip_design_nx: float = 0.5             # fore-aft design inertia load factor (gust / roll-yaw coupling, notional)
    ip_floor_frac: float = 0.1            # in-plane design moment >= this x wing root bending design moment


def allowables_v2(mdl: FlexBodyModel, wts: StructWeightsV2 = StructWeightsV2()) -> Dict[str, float]:
    """Notional limit root moments: tail/fin lift at q_D with a 5 deg incidence (+ n_limit inertia for the vertical
    fuselage), scaled by the stiffness gene (strength ~ stiffness, as v1 does with ei_scale)."""
    qd = _q_of_keas(mdl.pw.v_dive_keas)
    a = math.radians(wts.tail_design_alpha_deg)
    ts, fs = mdl.genes["tail_stiffness_scale"], mdl.genes["fuselage_stiffness_scale"]
    st, bs = mdl.st, mdl.body_strips
    ht = float(np.sum(qd * st["lw"][bs["htR"]] * a * st["arm"][bs["htR"]]))
    vt = float(np.sum(qd * st["lw"][bs["vt"]] * a * st["arm"][bs["vt"]]))
    Lht = 2 * float(np.sum(qd * st["lw"][bs["htR"]] * a))
    fus_m = mdl.fusV.base_mass_lb * LB2SLUG / mdl.l_h
    tail0 = (2 * mdl.ht_spec.mass_lb + mdl.vt_spec.mass_lb) * LB2SLUG
    fusV = mdl.l_h * Lht + mdl.pw.n_limit * G0 * (fus_m * mdl.l_h ** 2 / 2 + tail0 * mdl.l_h)
    fusL = mdl.l_v * float(np.sum(qd * st["lw"][bs["vt"]] * a))
    return {"ht_bm_allow": ht * ts, "vt_bm_allow": vt * ts, "fusV_bm_allow": fusV * fs, "fusL_bm_allow": fusL * fs}


SIZING_TERMS = ("J_wing_bm_limit", "J_wing_torque_limit", "J_wing_ip_limit", "J_tail_bm_limit", "J_fus_bm_limit")


def wing_design_loads(mdl: FlexBodyModel, wts: StructWeightsV2 = StructWeightsV2(), baseline_mass: bool = False) -> Dict:
    """Wing root design limit loads per side (lbf*ft, magnitudes), rigid load bases at n_limit (no elastic relief):
      bending = n W_d (Schrenk lift share x arm) - n g (m dy x arm)                    (W_d = empty weight, notional)
      torque  = |n W_d (lift x (x_ea - 0.25) c) + n g (m dy x x_theta)| + |q_D delta_d (aileron lift x e + cm_d c^2)|,
                floored at torque_floor_chord_frac x MAC x semi-span design lift
      in-plane = q_D S_w C_D,d (drag share x arm) + n_x,d g (m dy x arm), floored at ip_floor_frac x bending
    baseline_mass=True uses the baseline (genome-independent) section mass m0 -> the allowable reference."""
    pw, st, bs = mdl.pw, mdl.st, mdl.body_strips
    n, W = pw.n_limit, mdl.geom.empty_wt_lb
    qd = _q_of_keas(pw.v_dive_keas)
    lam = pw.taper
    c_mac = 2.0 / 3.0 * mdl.wingR.c_root * (1 + lam + lam * lam) / (1 + lam)
    out = {}
    for side, surf, ob, ot, oi, ck in (("R", mdl.wingR, 0, 2, 4, "q_dail_R"), ("L", mdl.wingL, 1, 3, 5, "q_dail_L")):
        r = bs["wing" + side]
        arm, xth = st["arm"][r], st["xth"][r]
        mdy = (surf.m0 if baseline_mass else surf.m) * surf.dy
        lift = mdl.Lb[CI["lift"], r]
        bend = n * W * float(lift @ arm) - n * G0 * float(mdy @ arm)
        t_lift = n * W * float(lift @ st["ec"][r]) + n * G0 * float(mdy @ xth)
        t_hinge = qd * math.radians(wts.design_ctrl_deg) * float(mdl.RB[CI[ck], ot])
        t_floor = wts.torque_floor_chord_frac * c_mac * n * W * float(np.sum(lift))
        torque = max(abs(t_lift) + abs(t_hinge), t_floor)
        ip = qd * pw.area_ft2 * wts.ip_design_cd * float(mdl.Fv[CI["drag"], r] @ arm) + wts.ip_design_nx * G0 * float(mdy @ arm)
        ip = max(ip, wts.ip_floor_frac * abs(bend))
        out[side] = {"bm": abs(bend), "torque": torque, "ip": ip}
    return out


def sizing_v2(mdl: FlexBodyModel, wts: StructWeightsV2 = StructWeightsV2()) -> Dict:
    """Pre-flight limit-load sizing checks. For every body whose mass can drop: ratio = design demand (current masses)
    / allowable, allowable = (1 + MS) x design load of the BASELINE structure x that body's stiffness gene (strength
    ~ stiffness, as the v1 wing allowable n_limit M_1g ei_scale). ratio ~ 1/s, so lowering a stiffness gene below 1
    costs J = w (ratio - 1)^2 while the mass credit is only linear and floored by MIN_GAUGE.
      wing bending: root, wing_ei_root (x (1 +- asym delta)); torque: wing_ei_root x wing_gj_ratio_root; in-plane:
      wing_ei_root (in-plane EI follows the EI multiplier); HT / VT root: tail_stiffness_scale (allowables_v2);
      aft fuselage vertical / lateral at the wing station (the cantilever maximum; the tail-station moment is the HT/VT
      root check): fuselage_stiffness_scale, vertical demand with the current fuselage + tail masses."""
    g = mdl.genes
    ms = 1.0 + wts.design_margin_of_safety
    cur, base = wing_design_loads(mdl, wts), wing_design_loads(mdl, wts, baseline_mass=True)
    d = g.get("wing_asym_ei_delta", 0.0)
    ratios, allow, demand = {}, {}, {}
    for side, sg in (("R", 1.0), ("L", -1.0)):
        s_ei = g["wing_ei_root"] * (1 + sg * d)
        s_gj = s_ei * g["wing_gj_ratio_root"]
        for k, sc in (("bm", s_ei), ("torque", s_gj), ("ip", s_ei)):
            nm = f"wing{side}_{k}"
            allow[nm] = ms * base[side][k] * sc
            demand[nm] = cur[side][k]
            ratios[nm] = demand[nm] / allow[nm]
    al = allowables_v2(mdl, wts)
    ts, fs = g["tail_stiffness_scale"], g["fuselage_stiffness_scale"]
    qd = _q_of_keas(mdl.pw.v_dive_keas)
    a = math.radians(wts.tail_design_alpha_deg)
    Lht = 2 * float(np.sum(qd * mdl.st["lw"][mdl.body_strips["htR"]] * a))
    tail_cur = (2 * mdl.htR.mass_lb + mdl.vt.mass_lb) * LB2SLUG
    fusV_cur = mdl.l_h * Lht + mdl.pw.n_limit * G0 * (mdl.fusV.m * mdl.l_h ** 2 / 2 + tail_cur * mdl.l_h)
    for nm, dem, alw in (("ht", al["ht_bm_allow"] / ts, al["ht_bm_allow"]), ("vt", al["vt_bm_allow"] / ts, al["vt_bm_allow"]),
                         ("fusV", fusV_cur, al["fusV_bm_allow"]), ("fusL", al["fusL_bm_allow"] / fs, al["fusL_bm_allow"])):
        allow[nm] = ms * alw
        demand[nm] = dem
        ratios[nm] = dem / allow[nm]
    mx = lambda *k: max(ratios[x] for x in k)  # noqa: E731
    hinge = lambda r: max(0.0, r - 1.0) ** 2   # noqa: E731
    terms = {"J_wing_bm_limit": wts.w_wing_bm_limit * hinge(mx("wingR_bm", "wingL_bm")),
             "J_wing_torque_limit": wts.w_wing_torque_limit * hinge(mx("wingR_torque", "wingL_torque")),
             "J_wing_ip_limit": wts.w_wing_ip_limit * hinge(mx("wingR_ip", "wingL_ip")),
             "J_tail_bm_limit": wts.w_tail_bm_limit * hinge(mx("ht", "vt")),
             "J_fus_bm_limit": wts.w_fus_bm_limit * hinge(mx("fusV", "fusL"))}
    return {"terms": terms, "ratios": ratios, "allowables_lbft": allow, "demand_lbft": demand}


def margin_terms_v2(mdl: FlexBodyModel, wts: StructWeightsV2 = StructWeightsV2(), gate: float = 1.0) -> Dict:
    """Pre-flight terms: margin hinge^2 below margin_req (1.2) only -- zero at or above 1.2, so the 3.0 not-found cap
    neither rewards nor penalises -- hard fail below the gate; J_mass (credit for weight savings, MIN_GAUGE-floored);
    J_smooth; limit-load sizing terms (sizing_v2)."""
    m = margins_v2(mdl)
    terms, fail = {}, None
    for key, wk, nm in (("flutter_margin", wts.w_flutter, "flutter"), ("div_margin", wts.w_div, "divergence"),
                        ("reversal_margin", wts.w_reversal, "reversal")):
        mg = m[key]
        if mg < gate:
            fail = fail or nm
        terms["J_" + key] = wk * max(0.0, (wts.margin_req - mg) / (wts.margin_req - 1.0)) ** 2
    if m["margin_error"]:
        fail = fail or "margin_error"
    ms = mdl.mass_summary()
    terms["J_mass"] = wts.w_mass * ms["total_frac"]
    terms["J_smooth"] = smoothness_penalty(mdl.genes, wts.w_smooth)
    sz = sizing_v2(mdl, wts)
    terms.update(sz["terms"])
    return {"margins": m, "terms": terms, "fail": fail, "mass": ms, "sizing": sz}


def response_terms_v2(hist: Dict[str, np.ndarray], mdl: FlexBodyModel, out_1g: np.ndarray,
                      wts: StructWeightsV2 = StructWeightsV2()) -> Dict:
    """Post-flight terms: wing terms as v1 (allowable = n_limit x M_1g x wing_ei_root) + empennage/fuselage peak terms."""
    pw = mdl.pw
    o1 = {nm: float(out_1g[i]) for i, nm in enumerate(OUT_NAMES)}
    m1g = 0.5 * (o1["wingR_bm"] + o1["wingL_bm"])
    bm = np.concatenate([hist["wingR_bm"], hist["wingL_bm"]])
    bm_rms = float(np.sqrt(np.mean((bm - m1g) ** 2)))
    allow = pw.n_limit * m1g * mdl.genes["wing_ei_root"]
    peak = float(np.max(np.abs(bm)))
    tip = float(np.max(np.abs(np.concatenate([hist["tip_w_ft_R"], hist["tip_w_ft_L"]]))))
    tw = float(np.max(np.abs(np.concatenate([hist["tip_twist_R_deg"], hist["tip_twist_L_deg"]]))))
    w_lim = pw.tip_defl_limit_frac * pw.span_ft / 2
    al = allowables_v2(mdl, wts)
    r_tail = max(float(np.max(np.abs(np.concatenate([hist["htR_bm"], hist["htL_bm"]])))) / al["ht_bm_allow"],
                 float(np.max(np.abs(hist["vt_bm"]))) / al["vt_bm_allow"])
    r_fus = max(float(np.max(np.abs(hist["fusV_bm"]))) / al["fusV_bm_allow"],
                float(np.max(np.abs(hist["fusL_bm"]))) / al["fusL_bm_allow"])
    sz = sizing_v2(mdl, wts)["allowables_lbft"]
    al["wing_torque_allow"] = min(sz["wingR_torque"], sz["wingL_torque"])
    al["wing_ip_allow"] = min(sz["wingR_ip"], sz["wingL_ip"])
    r_tq = max(float(np.max(np.abs(hist[k]))) / sz[k] for k in ("wingR_torque", "wingL_torque"))
    r_ip = max(float(np.max(np.abs(hist[k + "_ip_bm"]))) / sz[k + "_ip"] for k in ("wingR", "wingL"))
    terms = {"J_bm_rms": wts.w_bm_rms * bm_rms / m1g, "J_bm_peak": wts.w_bm_peak * max(0.0, peak / allow - 1) ** 2,
             "J_tip": wts.w_tip * max(0.0, tip / w_lim - 1) ** 2,
             "J_twist": wts.w_twist * max(0.0, tw / pw.twist_limit_deg - 1) ** 2,
             "J_tail_bm_peak": wts.w_tail_bm_peak * max(0.0, r_tail - 1) ** 2,
             "J_fus_bm_peak": wts.w_fus_bm_peak * max(0.0, r_fus - 1) ** 2,
             "J_wing_torque_peak": wts.w_wing_torque_peak * max(0.0, r_tq - 1) ** 2,
             "J_wing_ip_peak": wts.w_wing_ip_peak * max(0.0, r_ip - 1) ** 2}
    fail = None
    if peak > 1.5 * allow:
        fail = "structural_ultimate"
    elif r_tail > 1.5 or r_fus > 1.5:
        fail = "structural_ultimate_empennage"
    loads = {nm: {"peak_abs": float(np.max(np.abs(hist[nm]))),
                  "rms_dev_1g": float(np.sqrt(np.mean((np.asarray(hist[nm]) - o1[nm]) ** 2))), "trim_1g": o1[nm]}
             for nm in OUT_NAMES}
    return {"terms": terms, "fail": fail, "loads": loads, "bm_allow": allow, "tip_max_ft": tip, "twist_max_deg": tw,
            "tail_ratio": r_tail, "fus_ratio": r_fus, "torque_ratio": r_tq, "ip_ratio": r_ip, "allowables": al,
            "m_root_1g": m1g}


# =====================================================================================================================
# 10. Validation helpers (hand calcs) and modal-truncation check
# =====================================================================================================================
def uniform_test_surface(L=10.0, c=2.0, EI=2.0e6, GJ=1.5e6, m=0.5, r_gyr=0.25, x_ea=0.40, x_cg=0.40, a=2 * math.pi,
                         cf=0.25, n_el=32, n_b=3, n_t=2) -> Surface:
    """Uniform rectangular unswept cantilever (strip theory, full-span flap, incompressible) for hand-calc validation.
    Stiffnesses are set directly (calibration bypassed)."""
    sp = SurfaceSpec(name="test", kind="wing", span_ft=2 * L, area_ft2=2 * L * c, taper=1.0, sweep_deg=0.0, root_frac=0.0,
                     mass_lb=m * L * G0, f_b1_hz=1.0, f_t1_hz=1.0, f_ip_hz=None, x_ea=x_ea, x_cg=x_cg, aspect_eff=10.0,
                     n_el=n_el, n_b=n_b, n_t=n_t, r_gyr=r_gyr, ei_taper_exp=0.0, mass_taper_exp=0.0, ctrl_eta=(0.0, 1.0),
                     ctrl_cf=cf, ctrl_scale=1.0)
    return Surface(sp, cal={"EI_root0": EI, "GJ_root0": GJ, "EIv_root0": 0.0}, cla=a)


def surface_static_matrices(s: Surface):
    """Single surface, incompressible: modal K, A_K, control load vector b (per q*delta), rigid root rolling moment r,
    elastic rolling-moment row rG (per q per eta)."""
    G = math.cos(s.lam) * s.PsiT - math.sin(s.lam) * s.dPhiW
    W = s.PhiW + s.e_c[:, None] * s.PsiT
    K = np.diag(s.omega ** 2)
    A = W.T @ (s.lift_w[:, None] * G)
    Lc = s.c * s.dy * s.cl_d * s.ctrl_mask
    Mc = s.c ** 2 * s.dy * s.cm_d * s.ctrl_mask
    b = W.T @ Lc + s.PsiT.T @ Mc
    arm = s.y - s.y0
    return K, A, b, float(arm @ Lc), (arm * s.lift_w) @ G


def hand_calcs(L=10.0, c=2.0, EI=2.0e6, GJ=1.5e6, m=0.5, r_gyr=0.25, x_ea=0.40, a=2 * math.pi, cf=0.25,
               n_el=32, n_b=3, n_t=2) -> Dict[str, Dict[str, float]]:
    """FE (with the default modal truncation) vs closed form, uniform cantilever. Returns {quantity: fe, exact, pct}."""
    s = uniform_test_surface(L, c, EI, GJ, m, r_gyr, x_ea, x_ea, a, cf, n_el, n_b, n_t)
    Ia = m * (r_gyr * c) ** 2
    res = {}
    fb = s.omega[list(s.cls).index("b")] / (2 * math.pi)
    ft = s.omega[list(s.cls).index("t")] / (2 * math.pi)
    res["f_bend1_hz"] = (fb, 1.875104 ** 2 / (2 * math.pi) * math.sqrt(EI / (m * L ** 4)))
    res["f_tors1_hz"] = (ft, 1 / (4 * L) * math.sqrt(GJ / Ia))
    K, A, b, r_rig, rG = surface_static_matrices(s)
    mu = np.linalg.eigvals(np.linalg.solve(K, A))
    mu = mu[(abs(mu.imag) < 1e-9) & (mu.real > 0)].real
    qD = float(1 / mu.max())
    e = (x_ea - 0.25) * c
    qD_x = (math.pi / (2 * L)) ** 2 * GJ / (c * e * a)
    res["q_div_psf"] = (qD, qD_x)
    cl_d, cm_d = s.cl_d, s.cm_d

    def eff(q):
        return 1 + float(rG @ np.linalg.solve(K - q * A, q * b)) / r_rig

    def eff_exact(q):
        lam = math.sqrt(q * c * e * a / GJ)
        thp = -(e * cl_d + c * cm_d) / (e * a)
        return 1 + a * thp * (L ** 2 / 2 - (1 - math.cos(lam * L)) / (lam ** 2 * math.cos(lam * L))) / (cl_d * L ** 2 / 2)

    def first_zero(f, hi):
        qs = np.linspace(hi * 1e-4, hi, 4000)
        prev = f(qs[0])
        for q0, q1 in zip(qs[:-1], qs[1:]):
            v = f(q1)
            if v <= 0 < prev:
                lo_, hi_ = q0, q1
                for _ in range(60):
                    mm = 0.5 * (lo_ + hi_)
                    if f(mm) > 0: lo_ = mm
                    else: hi_ = mm
                return hi_
            prev = v
        return math.inf
    res["q_reversal_psf"] = (first_zero(eff, 0.999 * qD), first_zero(eff_exact, 0.999 * qD_x))
    p = 10.0
    eta = np.linalg.solve(K, s.PhiW.T @ (np.full(s.y.size, p) * s.dy))
    res["tip_defl_uniform_load_ft"] = (float(s.tipW @ eta), p * L ** 4 / (8 * EI))
    return {k: {"fe": float(v[0]), "exact": float(v[1]), "pct": float(100 * (v[0] - v[1]) / v[1])} for k, v in res.items()}


def _wing_metrics(mdl: FlexBodyModel) -> Dict[str, float]:
    """Right-wing block responses used for the truncation check."""
    s = mdl.wingR
    idx = _block(mdl, ("wingR",))
    B = BlockAero(mdl, idx)
    vd = mdl.pw.v_dive_keas
    q = _q_of_keas(0.9 * vd)
    kap = B.kap_q(q)
    A = B.AKq(q)
    st, r = mdl.st, mdl.body_strips["wingR"]
    lw, arm, mdy = st["lw"][r], st["arm"][r], st["mdy"][r]
    Wr, Gr, Hr, Wi = mdl.W[r][:, idx], mdl.G[r][:, idx], mdl.H[r][:, idx], mdl.W_in[r][:, idx]
    # static aeroelastic: 1 deg rigid incidence + 1 g
    L0 = q * kap["w"] * lw * math.radians(1.0)
    F0 = -G0 * mdy
    eta = np.linalg.solve(B.K - q * A, Wr.T @ L0 + Wi.T @ F0)
    bm = float(arm @ (L0 + q * kap["w"] * lw * (Gr @ eta) + F0))
    tip = float(s.tipW @ eta)
    # dynamic: 1-cos vertical gust, 0.5 s long, 20 ft/s peak, Newmark dt = 1/960 s over 2 s
    vt = math.sqrt(2 * q / RHO0)
    dt = 1 / 960
    t = np.arange(int(2.0 / dt)) * dt
    wg = np.where(t < 0.5, 10.0 * (1 - np.cos(2 * math.pi * t / 0.5)), 0.0)
    Lg = np.outer(q * kap["w"] * wg / vt, lw)
    Ce = B.C - (q / vt) * (kap["w"] * B.AC["w"] + kap["h"] * B.AC["h"] + kap["v"] * B.AC["v"] + B.Anc)
    x, v, a = fw.newmark(np.eye(idx.size), Ce, B.K - q * A, Lg @ Wr, dt)
    f = Lg + q * kap["w"] * ((x @ Gr.T) + (v @ Hr.T) / vt) * lw - (a @ Wi.T) * mdy
    bm_t, tip_t = f @ arm, x @ s.tipW
    v_fl, _ = B.flutter_keas(4 * vd, n_grid=160)
    q_co = B.coalescence_q(16 * _q_of_keas(vd), n_grid=600)
    v_co = _keas_of_q(q_co) if math.isfinite(q_co) else math.inf
    return {"static_root_bm": bm, "static_tip_w": tip, "gust_peak_root_bm": float(np.max(np.abs(bm_t))),
            "gust_peak_tip_w": float(np.max(np.abs(tip_t))), "flutter_keas_min_4VD": float(min(v_fl, v_co))}


def truncation_check(model: str, genes=None, extra=(1, 2), root_v2: str = ROOT_V2) -> Dict:
    """Wing modal-truncation convergence: static aeroelastic root BM / tip deflection (q at 0.9 V_D, 1 deg + 1 g), 1-cos gust
    peak root BM / tip deflection, wing flutter speed (searched to 4 V_D, inf if none) for the default N modes and N + extra
    next-lowest FE modes. Returns % change vs N (0 when both are inf)."""
    vals = {0: _wing_metrics(FlexBodyModel(model, genes, root_v2=root_v2))}
    for ex in extra:
        vals[ex] = _wing_metrics(FlexBodyModel(model, genes, wing_extra_modes=ex, root_v2=root_v2))
    out = {"N": int(FlexBodyModel(model, genes, root_v2=root_v2).wingR.Phi.shape[1]), "values": {}}
    for ex, v in vals.items():
        out["values"]["N" if ex == 0 else f"N+{ex}"] = v
    for ex in extra:
        d = {}
        for k, v0 in vals[0].items():
            v1 = vals[ex][k]
            if math.isinf(v0) and math.isinf(v1):
                d[k] = 0.0
            elif math.isinf(v0) or math.isinf(v1):
                d[k] = math.inf
            else:
                d[k] = float(100 * (v1 - v0) / v0)
        out[f"pct_N+{ex}"] = d
    return out


# =====================================================================================================================
# 11. BLAS threading (numpy's bundled OpenBLAS): small dense ops are ~40x slower when OpenBLAS threads fight a loaded box
# =====================================================================================================================
def blas_threads(n: Optional[int] = None) -> Optional[int]:
    """Return numpy's OpenBLAS thread count; if n is given set it first and return the previous count. None if the
    bundled scipy-openblas library is not found (then nothing is changed)."""
    import ctypes
    import glob
    libs = glob.glob(os.path.join(os.path.dirname(np.__file__), os.pardir, "numpy.libs", "libscipy_openblas*.so*"))
    if not libs:
        return None
    try:
        lib = ctypes.CDLL(libs[0])
        get, set_ = lib.scipy_openblas_get_num_threads64_, lib.scipy_openblas_set_num_threads64_
    except (OSError, AttributeError):
        return None
    get.restype = ctypes.c_int
    prev = int(get())
    if n is not None:
        set_(ctypes.c_int(int(n)))
    return prev
