"""flexbody_b1.py -- P3-B1: planform-shaped A1 structural model (opt-in fidelity 'full_a1_b1').

Additive on flexbody_a1 / flexbody / flexwing (hashed into the pinned A1 / full / reduced strings; NOT edited).

Baseline shape (all 6 shape genes at default): FlexBodyModelB1 IS the FlexBodyModelA1 object state (the constructor
returns before touching anything), margin_terms_b1 = flexbody_a1.margin_terms_a1, response terms = flexbody's, coupler =
FlexBodyCoupler arithmetic -> bit-identical to full_a1 (tested; v2_results/p3b1_acceptance.json).

Non-baseline shape (planform_b1 for the gene definitions) -- rules, all strip-theory, no CFD:
 1. Planform strips: shaped chord c(y) on the same 64 A1 strips, semi-wing AREA and SPAN fixed (= JSBSim S_ref, b), so
    the FDM's rigid tables stay valid for the whole aircraft; quarter-chord sweep = baseline + delta.
 2. AC hold: the wing is re-positioned along x so the Schrenk-weighted quarter-chord x of the shaped wing equals the
    baseline one (static margin / Cm tables unchanged; a designer re-places the wing). Strip x arms otherwise follow sweep.
 3. Aero strips: lift slope DATCOM(AR fixed, new sweep), strip lift weight c dy a, e_c / d34 / x_theta from the local
    chord, G = cos(sweep) theta - sin(sweep) w', Schrenk share from the shaped chord, Theodorsen apparent mass on c^2,
    aileron Cl_da strip calibration redone on the shaped chords (same FDM target).
 4. Geometry-derived structural baseline: EI, GJ, EIv = the BASELINE root constants (A1 calibration) x
    (c / c_root_baseline)^ei_taper_exp, mass ~ (c / c_root)^mass_taper_exp with the wing total fixed, Ia from local c.
    So a planform with less tip chord IS softer at the tip (frequencies move); structure genes then multiply as usual.
 5. Geometric twist (root = 0): the BASIC (zero-net) strip load q kappa_w lw (twist - Schrenk-share mean) -- a uniform
    incidence part would be absorbed by trim. It loads the structure (static trim shape, root/outboard moments, torque)
    and feeds back only its rigid pitch moment relative to trim, (q kappa_w - trim) x sum(-x L_basic) (zero at t0).
 6. Sizing (pre-flight, same 6 SIZING_TERMS / weights): wing allowables are anchored to the BASELINE planform design
    loads (structure designed for the baseline wing) x structure gene x geometric strength factor (c/c0)^3 at the check
    station (root: norm^3; eta 0.875: local); demand = current planform design loads (Schrenk share + basic twist load
    at q_D, counted only where it raises the demand: no sizing credit for twist relief) with current masses; outboard
    check station-exact (sizing_a1 method). Tail / fuselage checks unchanged.
 7. Flown terms: flexbody.response_terms_v2 formulas; wing torque / in-plane peak allowables taken from sizing_b1.
Not modelled (deferred): geometric dihedral, thickness / camber (B2), wing size (chord_root / area / span: needs rescaled
tables), wing-mass CG / Ixx shift from sweep or chord redistribution (the FDM point masses carry only the gene Δmass),
(r1: node_layout_b1 / FlexBodyModelB1.node_layout follow the shaped wing; flexbody.node_layout itself is the baseline layout).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np

import flexwing as fw
import flexbody as fb
import flexbody_a1 as fba1
import planform_b1 as pb1

B1_FIDELITY = pb1.B1_FIDELITY
B1_TAG = pb1.B1_TAG
B1_FMT = pb1.B1_FMT


def b1_params(model: str, geom: fb.Geometry, wing: Optional[Dict] = None) -> Dict:
    """A1 parameters + the frozen B1 shape schema / rules (hashed into model_version). Shape VALUES are per-genome inputs."""
    p = fba1.a1_params(model, geom, wing)
    p["variant"] = "a1_b1"
    p["b1_fmt"] = B1_FMT
    p["shape_schema"] = pb1.shape_params_for_hash()
    return p


def planform_wing_surface(sp: fb.SurfaceSpec, cal: Dict[str, float], pf: pb1.PlanformStrips, ei_mult, gj_mult, nsm_mult,
                          zeta: float, n_sel: Optional[Tuple[int, int, int]] = None, extra_modes: int = 0) -> fb.Surface:
    """flexbody.Surface with every chord-dependent array from the shaped planform (rules 3-4). `sp` is the shaped spec
    (sweep, ctrl_scale already replaced); `cal` = the BASELINE wing calibration (root EI/GJ/EIv constants)."""
    n = sp.n_el
    s = sp.span_ft / 2.0
    S = object.__new__(fb.Surface)
    S.sp = sp
    S.s, S.y0 = s, sp.root_frac * s
    S.L = s - S.y0
    S.lam = math.radians(sp.sweep_deg)
    S.c_root = sp.area_ft2 / (s * (1 + sp.taper))           # baseline centreline chord (reference of the c^k laws)
    S.dy = float(pf.dy_ft)
    S.y = np.array(pf.y_ft, float)
    S.xi = np.array(pf.xi, float)
    S.c = np.array(pf.c_ft, float)
    S.ei_mult, S.gj_mult, S.nsm_mult = (np.ones(n) if v is None else np.asarray(v, float) for v in (ei_mult, gj_mult, nsm_mult))
    S.e_c = (sp.x_ea - 0.25) * S.c
    S.d34 = (0.75 - sp.x_ea) * S.c
    S.x_theta = (sp.x_cg - sp.x_ea) * S.c
    S.a = fw.datcom_cla(sp.aspect_eff, S.lam)
    S._cla0 = S.a
    wsh = (S.c / S.c_root) ** sp.mass_taper_exp
    S.m0 = sp.mass_lb * fb.LB2SLUG * wsh / (np.sum(wsh) * S.dy)          # wing total fixed (area fixed)
    f = (1 - sp.struct_frac) * S.nsm_mult + sp.struct_frac * (sp.w_ei * fb.gauge_mass_factor(S.ei_mult)
                                                             + (1 - sp.w_ei) * fb.gauge_mass_factor(S.gj_mult))
    S.m = S.m0 * f
    S.mass_lb = float(np.sum(S.m) * S.dy * fb.G0)
    S.dmass_lb = float(np.sum(S.m0 * (f - 1.0)) * S.dy * fb.G0)
    S.Ia = S.m * ((sp.r_gyr * S.c) ** 2 + S.x_theta ** 2)
    shape = (S.c / S.c_root) ** sp.ei_taper_exp
    S.cal = dict(cal)
    S.EI = cal["EI_root0"] * shape * S.ei_mult
    S.GJ = cal["GJ_root0"] * shape * S.gj_mult
    S.EIv = cal["EIv_root0"] * shape * S.ei_mult if sp.f_ip_hz else None
    S.beam = fb.Beam(S.L, S.EI, S.m, GJ=S.GJ, Ia=S.Ia, mxt=S.m * S.x_theta, EIv=S.EIv)
    om, Phi, cls = S.beam.modes()
    nb, nt, nv = n_sel or (sp.n_b, sp.n_t, sp.n_ip)
    idx = fb.select_modes(om, Phi, cls, nb, nt, nv if sp.f_ip_hz else 0, extra=extra_modes)
    S.all_omega, S.all_cls, S.all_Phi = om, cls, Phi
    S.idx = idx
    S.omega, S.cls, S.Phi = om[idx], cls[idx], Phi[:, idx]
    S.zeta = zeta
    b = S.beam
    S.PhiW, S.dPhiW, S.PsiT, S.PhiV = b.Nw @ S.Phi, b.dNw @ S.Phi, b.Nt @ S.Phi, b.Nv @ S.Phi
    S.tipW, S.tipT = b.tip_w @ S.Phi, b.tip_t @ S.Phi
    S.tipV = b.tip_v @ S.Phi
    eta = S.y / s
    S.ctrl_mask = np.zeros(n)
    S.cl_d = S.cm_d = 0.0
    if sp.ctrl_eta is not None and sp.ctrl_cf > 0:
        S.ctrl_mask = ((eta >= sp.ctrl_eta[0]) & (eta <= sp.ctrl_eta[1])).astype(float)
        cl_d, cm_d = fw.flap_coeffs(sp.ctrl_cf)
        k = sp.ctrl_scale if sp.ctrl_scale is not None else S.a / (2 * math.pi)
        S.cl_d, S.cm_d = k * cl_d, k * cm_d
    S.lift_w = S.c * S.dy * S.a
    return S


def _schrenk_share(pw, pf: pb1.PlanformStrips, shape_genes, y: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Per-strip share of the TOTAL aero lift for one semi-wing: flexbody's Schrenk rule
    0.5 wing_lift_share x [0.5 c/c_mean + 0.5 elliptic] / norm x dy with the shaped chord c(v) = c0(v) x M(xi(v)) x norm
    (M = log-PCHIP chord multiplier, clamped to the root value inboard of the beam root, i.e. over the carry-through)."""
    s = pw.span_ft / 2
    y0 = pw.root_frac * s
    c_mean = pw.area_ft2 / 2 / s

    def shape(v):
        xi = np.clip((v - y0) / (s - y0), 0.0, 1.0)
        cv = pb1.baseline_chord_ft(pw.span_ft, pw.area_ft2, pw.taper, v) * pb1.chord_multipliers_raw(shape_genes, xi) * pf.area_norm
        return 0.5 * cv / c_mean + 0.5 * 4 / math.pi * np.sqrt(np.clip(1 - (v / s) ** 2, 0, None))
    yy = (np.arange(2000) + 0.5) / 2000 * s
    norm = np.sum(shape(yy)) * (s / 2000)
    return 0.5 * pw.wing_lift_share * shape(y) / norm * dy


class FlexBodyModelB1(fba1.FlexBodyModelA1):
    """FlexBodyModelA1 + B1 planform. shape_genes None / {} / defaults -> the A1 model, untouched (bit-identical)."""

    def __init__(self, model: str, genes=None, shape_genes=None, geom: Optional[fb.Geometry] = None,
                 asymmetric: bool = False, wing_extra_modes: int = 0, root_v2: str = fb.ROOT_V2,
                 wing_n_sel: Optional[Tuple[int, int, int]] = None, wing_mesh: Optional[Dict] = None):
        self.shape_genes = pb1.decode_shape_b1(shape_genes)          # raises before any build
        self.planform_baseline = pb1.is_baseline_shape(self.shape_genes)
        self._pf_active = False
        self.root_v2 = root_v2
        super().__init__(model, genes, geom=geom, asymmetric=asymmetric, wing_extra_modes=wing_extra_modes,
                         root_v2=root_v2, wing_n_sel=wing_n_sel, wing_mesh=wing_mesh)
        self.planform = pb1.planform_from_wingparams(self.pw, self.shape_genes, int(self.wing_a1["n_el"]))
        ns = self.ns
        self.twist_L = np.zeros(ns)            # basic twist strip load per unit q*kappa_w (lbf / psf)
        self.twist_Q = np.zeros(self.N)
        self.twist_RB = np.zeros(len(fb.OUT_NAMES))
        self.twist_pitch = 0.0
        self.ac_shift_ft = 0.0
        self.bm_ref_ratio = 1.0                # r1: baseline-planform / shaped 1-g root BM (no twist), flown-term reference
        if self.planform_baseline:
            return
        bm1_a1 = self._root_bm_1g_per_n()       # baseline planform, this genome's masses (before the shaped rebuild)
        # ---- baseline wing quantities needed for the AC hold, then the shaped rebuild
        bs = self.body_strips
        wr = np.r_[np.arange(ns)[bs["wingR"]], np.arange(ns)[bs["wingL"]]]
        self._ac_base = float(self.Lb[fb.CI["lift"], wr] @ self.st["x"][wr])
        cal = dict(self.wingR.cal)
        pf, pw = self.planform, self.pw
        s_ = pw.span_ft / 2
        mask = ((pf.y_ft / s_ >= pw.ail_eta[0]) & (pf.y_ft / s_ <= pw.ail_eta[1])).astype(float)
        cl_d0, _ = fw.flap_coeffs(pw.ail_cf)
        strip = 2 * np.sum(pf.c_ft * pf.dy_ft * cl_d0 * mask * pf.y_ft) / (pw.area_ft2 * pw.span_ft)
        ctrl_scale = (pw.ail_cl_da_target / strip) if (pw.ail_cl_da_target is not None and strip > 0) else None
        self.wing_spec = wsp = replace(self.wing_spec, sweep_deg=pf.sweep_qc_deg, ctrl_scale=ctrl_scale)
        g, zeta, dist = self.genes, self.genes["struct_damping_ratio"], self.dist
        self.wingR = planform_wing_surface(wsp, cal, pf, dist["ei_R"], dist["gj_R"], dist["nsm_R"], zeta,
                                           n_sel=wing_n_sel, extra_modes=wing_extra_modes)
        if self.asymmetric and (g.get("wing_asym_ei_delta", 0.0) != 0.0 or g.get("wing_asym_nsm_delta", 0.0) != 0.0):
            self.wingL = planform_wing_surface(wsp, cal, pf, dist["ei_L"], dist["gj_L"], dist["nsm_L"], zeta,
                                               n_sel=wing_n_sel, extra_modes=wing_extra_modes)
        else:
            self.wingL = self.wingR
        self._pf_active = True
        self._assemble()                        # -> _build_load_bases (overridden below: Schrenk, AC hold, twist)
        self.bm_ref_ratio = float(bm1_a1 / self._root_bm_1g_per_n())

    def _root_bm_1g_per_n(self) -> float:
        """Mean (R, L) rigid 1-g wing root bending per unit load factor: W_d (Schrenk lift share x arm) - g (m dy x arm),
        the flexbody.wing_design_loads bending formula at n = 1 (no twist load)."""
        st, bs, W = self.st, self.body_strips, self.geom.empty_wt_lb
        v = []
        for nm, surf in (("wingR", self.wingR), ("wingL", self.wingL)):
            r = bs[nm]
            v.append(W * float(self.Lb[fb.CI["lift"], r] @ st["arm"][r]) - fb.G0 * float((surf.m * surf.dy) @ st["arm"][r]))
        return 0.5 * (v[0] + v[1])

    # ------------------------------------------------------------------------------------------------------------
    def _build_load_bases(self):
        if not getattr(self, "_pf_active", False):
            return super()._build_load_bases()
        st, bs, pw, pf = self.st, self.body_strips, self.pw, self.planform
        ns = self.ns
        wr = np.r_[np.arange(ns)[bs["wingR"]], np.arange(ns)[bs["wingL"]]]
        share = np.zeros(ns)
        for nm in ("wingR", "wingL"):
            r = bs[nm]
            share[r] = _schrenk_share(pw, pf, self.shape_genes, st["y"][r], st["dy"][r])
        # rule 2: AC hold (constant x shift of the wing strips)
        self.ac_shift_ft = (self._ac_base - float(share[wr] @ st["x"][wr])) / float(np.sum(share[wr]))
        st["x"][wr] = st["x"][wr] + self.ac_shift_ft
        super()._build_load_bases()             # everything from st (x shifted), baseline Schrenk lift row
        dLb = np.zeros(ns)
        dLb[wr] = share[wr] - self.Lb[fb.CI["lift"], wr]
        self.Lb[fb.CI["lift"]] += dLb           # shaped Schrenk share; propagate to the only consumers of that row:
        self.Qbasis[fb.CI["lift"]] += dLb @ self.W
        for o, nm in ((0, "wingR"), (1, "wingL")):
            r = bs[nm]
            self.RB[fb.CI["lift"], o] += dLb[r] @ st["arm"][r]
        for o, nm in ((2, "wingR"), (3, "wingL")):
            r = bs[nm]
            self.RB[fb.CI["lift"], o] += dLb[r] @ st["ec"][r]
        # rule 5: basic twist load per unit q*kappa_w (zero net lift; symmetric -> zero roll)
        tw = np.zeros(ns)
        for nm in ("wingR", "wingL"):
            tw[bs[nm]] = pf.twist_rad
        Ltw = st["lw"] * tw
        Ltw[~np.isin(np.arange(ns), wr)] = 0.0
        tot = float(np.sum(Ltw[wr]))
        L_basic = Ltw.copy()
        L_basic[wr] -= tot * share[wr] / float(np.sum(share[wr]))
        self.twist_L = L_basic
        self.twist_Q = self.W.T @ L_basic
        self.twist_RB = np.zeros(len(fb.OUT_NAMES))
        for o, nm in ((0, "wingR"), (1, "wingL")):
            r = bs[nm]
            self.twist_RB[o] = float(L_basic[r] @ st["arm"][r])
        for o, nm in ((2, "wingR"), (3, "wingL")):
            r = bs[nm]
            self.twist_RB[o] = float(L_basic[r] @ st["ec"][r])
        self.twist_pitch = float(np.sum(-st["x"] * L_basic))

    def planform_summary(self) -> Dict:
        d = self.planform.summary()
        S_, lam_ = self.pw.area_ft2, self.pw.taper
        cbar = 2.0 / 3.0 * S_ / (self.pw.span_ft / 2 * (1 + lam_)) * (1 + lam_ + lam_ * lam_) / (1 + lam_)
        d.update(planform_baseline=bool(self.planform_baseline), ac_shift_ft=float(self.ac_shift_ft),
                 twist_pitch_lbft_per_qkw=float(self.twist_pitch), twist_dCm0_equiv=float(self.twist_pitch / (S_ * cbar)),
                 bm_ref_ratio=float(self.bm_ref_ratio),
                 shape_genes=dict(self.shape_genes), shape_cache_key=pb1.shape_cache_key(self.shape_genes),
                 f_wing_hz=[float(w / (2 * math.pi)) for w in self.wingR.omega])
        return d

    def node_layout(self, rp_offset_body_ft=None):
        """= node_layout_b1(self, rp_offset_body_ft)."""
        return node_layout_b1(self, rp_offset_body_ft)


def node_layout_b1(mdl: FlexBodyModelB1, rp_offset_body_ft=None):
    """flexbody.node_layout for the B1 model. Baseline shape: flexbody.node_layout itself (identical lists).
    Shaped planform (r1): the wing elastic-axis nodes follow the SHAPED wing -- quarter-chord line at the shaped sweep
    through the baseline MAC quarter-chord station, plus the AC-hold x shift, EA at x_ea of the LOCAL SHAPED chord
    (same geometry as the strip arms / e_c). Twist does not move the EA (sections rotate about it); it is exported per
    node with the shaped chord and the twisted LE / TE points:
      chord_ft, geometric_twist_deg (+ LE up), le_nodes_body_ft / te_nodes_body_ft (body FRD, same origin as the axis).
    Other bodies unchanged. Display geometry only; the physics uses the strip arrays."""
    comps = fb.node_layout(mdl, rp_offset_body_ft)
    if getattr(mdl, "planform_baseline", True):
        return comps
    off = np.zeros(3) if rp_offset_body_ft is None else np.asarray(rp_offset_body_ft, float)
    pw = mdl.pw
    wy_mac = pw.span_ft / 2 / 3 * (1 + 2 * pw.taper) / (1 + pw.taper)
    for c in comps:
        nm = c["name"]
        if nm not in ("wingR", "wingL"):
            continue
        body = getattr(mdl, nm)
        b, sp = body.beam, body.sp
        st = np.arange(b.n_el + 1) * b.h
        y = body.y0 + st
        xi = np.clip(st / body.L, 0.0, 1.0)
        ch = pb1.baseline_chord_ft(pw.span_ft, pw.area_ft2, pw.taper, y) * pb1.chord_multipliers_raw(mdl.shape_genes, xi) \
            * float(mdl.planform.area_norm)
        th = pb1.twist_rad(mdl.shape_genes, xi)
        aft = (y - wy_mac) * math.tan(body.lam) + float(mdl.ac_shift_ft) + (sp.x_ea - 0.25) * ch
        sg = -1.0 if nm.endswith("L") else 1.0
        ax = np.stack([-aft, sg * y, np.zeros_like(y)], axis=1) + off[None, :]
        le = ax + np.stack([sp.x_ea * ch * np.cos(th), np.zeros_like(y), -sp.x_ea * ch * np.sin(th)], axis=1)
        te = ax + np.stack([-(1 - sp.x_ea) * ch * np.cos(th), np.zeros_like(y), (1 - sp.x_ea) * ch * np.sin(th)], axis=1)
        r6 = lambda a: [[round(float(v), 6) for v in p] for p in a]   # noqa: E731
        c["axis_nodes_body_ft"] = r6(ax)
        c["chord_ft"] = [round(float(v), 6) for v in ch]
        c["geometric_twist_deg"] = [round(math.degrees(float(v)), 6) for v in th]
        c["le_nodes_body_ft"] = r6(le)
        c["te_nodes_body_ft"] = r6(te)
        c["planform"] = "P3-B1 r1 shaped: EA at x_ea of the shaped local chord, shaped sweep, AC-hold shift; twist about EA"
    return comps


# =====================================================================================================================
# Coupler: FlexBodyCoupler + basic twist load (rule 5). Baseline shape -> twist arrays are zero and skipped entirely.
# =====================================================================================================================
class FlexBodyCouplerB1(fb.FlexBodyCoupler):
    def __init__(self, mdl: FlexBodyModelB1, mode: str = "twoway", substeps: int = 2, zero_feedback: bool = False):
        super().__init__(mdl, mode=mode, substeps=substeps, zero_feedback=zero_feedback)
        self._tw = bool(np.any(mdl.twist_L != 0.0))
        self._qk_ref = 0.0

    def initialize(self, fdm):
        if not self._tw:
            return super().initialize(fdm)
        m = self.mdl
        s = self.read_state(fdm)
        self.ref = {"alpha": s["alpha"], "elev": s["elev"], "rud": s["rud"], "beta": s["beta"]}
        kap = m.kappas(s["mach"])
        self._qk_ref = s["qbar"] * kap["w"]
        Q = self.coefs(s, kap) @ m.Qbasis + self._qk_ref * m.twist_Q
        eta = np.linalg.solve(m.K, Q)
        self.eta, self.etad, self.etadd = eta.copy(), np.zeros(m.N), np.zeros(m.N)
        self.eta_ref = eta.copy()
        self.Qref_per_q = {g: m.A_K[g] @ eta for g in fb.GROUPS}
        self._apply(fdm, s, kap)
        self.out_1g = self.out.copy()

    def step(self, fdm, dt: float):
        if not self._tw:
            return super().step(fdm, dt)
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
             + (qw * s["beta"]) * (m.A_beta @ de) + qw * m.twist_Q)
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
        super()._apply(fdm, s, kap)
        if not self._tw:
            return
        m = self.mdl
        qk = s["qbar"] * kap["w"]
        self.out = self.out + qk * m.twist_RB                     # rigid basic-twist root loads (absolute, physical)
        for i, nm in enumerate(fb.OUT_NAMES):
            self.last[nm] = float(self.out[i])
        # r1: NO rigid pitch feedback from the basic twist load. Its moment q kappa_w twist_pitch is a constant Cm0
        # shift (both scale with q); a trimmed aircraft absorbs it with the elevator, whose moment also scales with q,
        # so dM/dq at fixed alpha / elevator stays zero. r0 fed back (q kappa_w - trim) x twist_pitch, an unphysical
        # q-proportional pitch moment (pseudo speed-stability term) that the T38 GA exploited (wash-in drift).
        # The ELASTIC response to the twist load (twist_Q -> eta -> dPitch via A_K, relative to eta_ref) is kept.


# =====================================================================================================================
# Sizing / margins / flown terms on the shaped model (rules 6-7)
# =====================================================================================================================
_BASE_DESIGN: Dict = {}


def _baseline_wing_design(model: str, root_v2: str, wts: fb.StructWeightsV2) -> Dict:
    """Design loads of the BASELINE planform (A1 model, baseline genes, baseline masses): the allowable reference.
    Cached per (model, root_v2, weights) -- one A1 build per process."""
    key = (model, root_v2, repr(wts))
    if key not in _BASE_DESIGN:
        A = fba1.FlexBodyModelA1(model, None, root_v2=root_v2)
        _BASE_DESIGN[key] = {"root": fb.wing_design_loads(A, wts, baseline_mass=True),
                             "tip": fba1.wing_tip_bm_station(A, wts, baseline_mass=True)}
    return _BASE_DESIGN[key]


def wing_design_loads_b1(mdl: FlexBodyModelB1, wts: fb.StructWeightsV2 = fb.StructWeightsV2(), twist_scale: float = 1.0) -> Dict:
    """flexbody.wing_design_loads (current masses, shaped Schrenk share) + twist_scale x the basic twist load at q_D
    (kappa_w at V_D), and the station-exact outboard moment (flexbody_a1 method) with the same twist load."""
    pw, st, bs = mdl.pw, mdl.st, mdl.body_strips
    n, W = pw.n_limit, mdl.geom.empty_wt_lb
    qd = fb._q_of_keas(pw.v_dive_keas)
    qk = qd * mdl.kappas(math.sqrt(2 * qd / fb.RHO0) / fw.speed_of_sound_fps(fb.RHO0))["w"]
    lam = pw.taper
    c_mac = 2.0 / 3.0 * mdl.wingR.c_root * (1 + lam + lam * lam) / (1 + lam)
    out = {}
    for side, surf, ot, ck in (("R", mdl.wingR, 2, "q_dail_R"), ("L", mdl.wingL, 3, "q_dail_L")):
        r = bs["wing" + side]
        arm, xth, ec = st["arm"][r], st["xth"][r], st["ec"][r]
        mdy = surf.m * surf.dy
        lift = mdl.Lb[fb.CI["lift"], r]
        Lt = twist_scale * qk * mdl.twist_L[r]
        bend = n * W * float(lift @ arm) - n * fb.G0 * float(mdy @ arm) + float(Lt @ arm)
        t_lift = n * W * float(lift @ ec) + n * fb.G0 * float(mdy @ xth) + float(Lt @ ec)
        t_hinge = qd * math.radians(wts.design_ctrl_deg) * float(mdl.RB[fb.CI[ck], ot])
        t_floor = wts.torque_floor_chord_frac * c_mac * n * W * float(np.sum(lift))
        torque = max(abs(t_lift) + abs(t_hinge), t_floor)
        ip = qd * pw.area_ft2 * wts.ip_design_cd * float(mdl.Fv[fb.CI["drag"], r] @ arm) + wts.ip_design_nx * fb.G0 * float(mdy @ arm)
        ip = max(ip, wts.ip_floor_frac * abs(bend))
        y_c = float(wts.tip_bm_eta) * surf.L
        lo = np.maximum(arm - 0.5 * surf.dy, y_c)
        frac = np.clip((arm + 0.5 * surf.dy - lo) / surf.dy, 0.0, 1.0)
        lev = np.where(frac > 0.0, 0.5 * (lo + arm + 0.5 * surf.dy) - y_c, 0.0)
        tip = abs(n * W * float(lift @ (frac * lev)) - n * fb.G0 * float(mdy @ (frac * lev)) + float(Lt @ (frac * lev)))
        out[side] = {"bm": abs(bend), "torque": torque, "ip": ip, "tip_bm": tip}
    return out


def sizing_b1(mdl: FlexBodyModelB1, wts: fb.StructWeightsV2 = fb.StructWeightsV2()) -> Dict:
    if getattr(mdl, "planform_baseline", True):
        return fba1.sizing_a1(mdl, wts)
    sz = fb.sizing_v2(mdl, wts)                         # tail / fuselage entries (unchanged rules)
    g = mdl.genes
    ms = 1.0 + wts.design_margin_of_safety
    base = _baseline_wing_design(mdl.model, mdl.root_v2, wts)
    # twist enters the design demand only where it RAISES it (no sizing credit for twist load relief: B1 has no drag /
    # off-design cost of washout); flown terms see the full twist load either way
    c_tw, c_no = wing_design_loads_b1(mdl, wts, 1.0), wing_design_loads_b1(mdl, wts, 0.0)
    cur = {sd: {k: max(c_tw[sd][k], c_no[sd][k]) for k in c_tw[sd]} for sd in c_tw}
    pf, ex = mdl.planform, mdl.pw.ei_taper_exp
    g_root = float(pf.area_norm) ** ex                  # (c/c0)^k at beam root (raw multiplier = 1 there)
    g_tip = float(pb1.chord_multipliers_raw(mdl.shape_genes, [wts.tip_bm_eta])[0] * pf.area_norm) ** ex
    d = g.get("wing_asym_ei_delta", 0.0)
    dist = fb.wing_distributions(g, np.array([float(wts.tip_bm_eta)]))
    for side, sg in (("R", 1.0), ("L", -1.0)):
        s_ei = g["wing_ei_root"] * (1 + sg * d)
        s_gj = s_ei * g["wing_gj_ratio_root"]
        s_tip = float(dist["ei_" + side][0])
        for k, sc, gs, b in (("bm", s_ei, g_root, base["root"][side]["bm"]), ("torque", s_gj, g_root, base["root"][side]["torque"]),
                             ("ip", s_ei, g_root, base["root"][side]["ip"]), ("tip_bm", s_tip, g_tip, base["tip"][side])):
            nm = f"wing{side}_{k}"
            sz["allowables_lbft"][nm] = ms * b * sc * gs
            sz["demand_lbft"][nm] = cur[side][k]
            sz["ratios"][nm] = cur[side][k] / sz["allowables_lbft"][nm]
    r = sz["ratios"]
    hinge = lambda x: max(0.0, x - 1.0) ** 2   # noqa: E731
    sz["terms"]["J_wing_bm_limit"] = wts.w_wing_bm_limit * hinge(max(r["wingR_bm"], r["wingL_bm"]))
    sz["terms"]["J_wing_torque_limit"] = wts.w_wing_torque_limit * hinge(max(r["wingR_torque"], r["wingL_torque"]))
    sz["terms"]["J_wing_ip_limit"] = wts.w_wing_ip_limit * hinge(max(r["wingR_ip"], r["wingL_ip"]))
    sz["terms"]["J_wing_tip_bm_limit"] = wts.w_wing_tip_bm_limit * hinge(max(r["wingR_tip_bm"], r["wingL_tip_bm"]))
    sz["tip_bm_method"] = "station_exact"
    sz["wing_allowable_reference"] = "baseline_planform"
    sz["geometric_strength_factor"] = {"root": g_root, "tip_bm_eta": g_tip}
    return sz


def margin_terms_b1(mdl: FlexBodyModelB1, wts: fb.StructWeightsV2 = fb.StructWeightsV2(), gate: float = 1.0) -> Dict:
    if getattr(mdl, "planform_baseline", True):
        return fba1.margin_terms_a1(mdl, wts, gate=gate)
    r = fb.margin_terms_v2(mdl, wts, gate=gate)
    sz = sizing_b1(mdl, wts)
    r["terms"].update(sz["terms"])
    r["sizing"] = sz
    return r


def response_terms_b1(hist, mdl: FlexBodyModelB1, out_1g, wts: fb.StructWeightsV2 = fb.StructWeightsV2(),
                      qk_ref: float = 0.0) -> Dict:
    """Flown terms. Baseline shape: flexbody.response_terms_v2 itself. Shaped planform: the same formulas line by line
    (kept in step with flexbody.response_terms_v2; tested equal at a baseline shape), with the wing torque / in-plane
    peak allowables from sizing_b1 (baseline-planform anchored) instead of sizing_v2.
    r1: the J_bm_rms denominator and the J_bm_peak allowable use m_ref = (flown 1-g root BM - trim twist basic-load BM)
    x bm_ref_ratio (baseline-planform 1-g BM / shaped 1-g BM) x [allowable only: (c/c0)^3 root strength factor], so a
    shape cannot buy credit by inflating its own 1-g reference (r0: twist wash-in raised m_1g -> lower J_bm_rms).
    qk_ref = q kappa_w at trim (FlexBodyCouplerB1._qk_ref). The rms deviation itself is about the flown 1-g value."""
    if getattr(mdl, "planform_baseline", True):
        return fb.response_terms_v2(hist, mdl, out_1g, wts)
    return _response_terms_with_sizing(hist, mdl, out_1g, wts, sizing_b1(mdl, wts), qk_ref=qk_ref)


def wing_bm_reference_b1(mdl: FlexBodyModelB1, m1g: float, qk_ref: float) -> float:
    """r1 flown wing-root BM reference (see response_terms_b1)."""
    tw = 0.5 * (float(mdl.twist_RB[0]) + float(mdl.twist_RB[1]))
    return (m1g - qk_ref * tw) * float(mdl.bm_ref_ratio)


def _response_terms_with_sizing(hist, mdl, out_1g, wts, sizing: Dict, qk_ref: float = 0.0) -> Dict:
    pw = mdl.pw
    o1 = {nm: float(out_1g[i]) for i, nm in enumerate(fb.OUT_NAMES)}
    m1g = 0.5 * (o1["wingR_bm"] + o1["wingL_bm"])
    m_ref = wing_bm_reference_b1(mdl, m1g, qk_ref)
    bm = np.concatenate([hist["wingR_bm"], hist["wingL_bm"]])
    bm_rms = float(np.sqrt(np.mean((bm - m1g) ** 2)))
    allow = pw.n_limit * m_ref * mdl.genes["wing_ei_root"] * float(sizing["geometric_strength_factor"]["root"])
    peak = float(np.max(np.abs(bm)))
    tip = float(np.max(np.abs(np.concatenate([hist["tip_w_ft_R"], hist["tip_w_ft_L"]]))))
    tw = float(np.max(np.abs(np.concatenate([hist["tip_twist_R_deg"], hist["tip_twist_L_deg"]]))))
    w_lim = pw.tip_defl_limit_frac * pw.span_ft / 2
    al = fb.allowables_v2(mdl, wts)
    r_tail = max(float(np.max(np.abs(np.concatenate([hist["htR_bm"], hist["htL_bm"]])))) / al["ht_bm_allow"],
                 float(np.max(np.abs(hist["vt_bm"]))) / al["vt_bm_allow"])
    r_fus = max(float(np.max(np.abs(hist["fusV_bm"]))) / al["fusV_bm_allow"],
                float(np.max(np.abs(hist["fusL_bm"]))) / al["fusL_bm_allow"])
    sz = sizing["allowables_lbft"]
    al["wing_torque_allow"] = min(sz["wingR_torque"], sz["wingL_torque"])
    al["wing_ip_allow"] = min(sz["wingR_ip"], sz["wingL_ip"])
    r_tq = max(float(np.max(np.abs(hist[k]))) / sz[k] for k in ("wingR_torque", "wingL_torque"))
    r_ip = max(float(np.max(np.abs(hist[k + "_ip_bm"]))) / sz[k + "_ip"] for k in ("wingR", "wingL"))
    terms = {"J_bm_rms": wts.w_bm_rms * bm_rms / m_ref, "J_bm_peak": wts.w_bm_peak * max(0.0, peak / allow - 1) ** 2,
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
             for nm in fb.OUT_NAMES}
    return {"terms": terms, "fail": fail, "loads": loads, "bm_allow": allow, "tip_max_ft": tip, "twist_max_deg": tw,
            "tail_ratio": r_tail, "fus_ratio": r_fus, "torque_ratio": r_tq, "ip_ratio": r_ip, "allowables": al,
            "m_root_1g": m1g, "m_root_1g_ref": m_ref}
