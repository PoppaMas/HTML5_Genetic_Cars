"""flexbody_b2.py -- P3-B2a: section-shaped B1 model (opt-in fidelity 'full_a1_b2a'). INTERFACE_v2.md section 15.

Additive on flexbody_b1 / flexbody_a1 / flexbody / flexwing (all frozen, imported, never edited).

B2 genes at default: FlexBodyModelB2 IS the FlexBodyModelB1 object (constructor delegates before anything else),
FlexBodyCouplerB2 delegates to FlexBodyCouplerB1 (it only READS energy signals from the FDM), margin / sizing / response
terms are B1's -> bit-identical to full_a1_b1 (tested).

Non-default B2 genes (planform_b2 for the gene definitions); structure sees LOAD VECTORS only, rigid aero is native
(JSBSim functions in <root>_v2b2, properties set before IC/trim by the hook):
 dihedral  antisymmetric strip load dih_L = lw sign dGamma per unit q kappa_w beta (delta only) -> dih_Q, dih_RB.
 camber    aero twist 2 dm(eta) added to the B1 geometric twist in the B1 basic (zero-net) load builder (uniform part is a
           trimmed lift change), strip couple q kappa_w c^2 dy dcm (dcm = -pi dm) -> cam_Q (torsion), cam_RB (root torque).
           No rigid pitch / roll feedback from either (the native increments own the rigid moments).
 thickness EI, GJ x tau(eta)^2 (fixed-gauge box), EIv x 1, mass unchanged; sizing strength x tau (bending, tip station,
           torque), in-plane x 1; modal rebuild.
Flown wing-BM reference: r1 rule (trim basic load incl. camber aero twist removed, x bm_ref_ratio to the A1 planform).
Design torque: camber couple counted only where it RAISES the demand (no reflex credit).
"""
from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np

import flexwing as fw
import flexbody as fb
import flexbody_a1 as fba1
import flexbody_b1 as fbb1
import planform_b1 as pb1
import planform_b2 as pb2

B2_FIDELITY = pb2.B2_FIDELITY
B2_TAG = pb2.B2_TAG
HERE = os.path.dirname(os.path.abspath(__file__))


def b2_params(model: str, geom: fb.Geometry, wing: Optional[Dict] = None) -> Dict:
    p = fbb1.b1_params(model, geom, wing)
    p["variant"] = "a1_b2a"
    p["b2_fmt"] = pb2.B2_FMT
    p["shape_schema_b2"] = pb2.shape_params_for_hash()
    return p


def planform_wing_surface_b2(sp, cal, pf, ei_mult, gj_mult, nsm_mult, zeta, tau, n_sel=None, extra_modes=0) -> fb.Surface:
    """flexbody_b1.planform_wing_surface with EI, GJ x tau^2 (thickness, fixed gauge). EIv, mass unchanged."""
    n = sp.n_el
    s = sp.span_ft / 2.0
    S = object.__new__(fb.Surface)
    S.sp = sp
    S.s, S.y0 = s, sp.root_frac * s
    S.L = s - S.y0
    S.lam = math.radians(sp.sweep_deg)
    S.c_root = sp.area_ft2 / (s * (1 + sp.taper))
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
    S.m0 = sp.mass_lb * fb.LB2SLUG * wsh / (np.sum(wsh) * S.dy)
    f = (1 - sp.struct_frac) * S.nsm_mult + sp.struct_frac * (sp.w_ei * fb.gauge_mass_factor(S.ei_mult)
                                                             + (1 - sp.w_ei) * fb.gauge_mass_factor(S.gj_mult))
    S.m = S.m0 * f
    S.mass_lb = float(np.sum(S.m) * S.dy * fb.G0)
    S.dmass_lb = float(np.sum(S.m0 * (f - 1.0)) * S.dy * fb.G0)
    S.Ia = S.m * ((sp.r_gyr * S.c) ** 2 + S.x_theta ** 2)
    shape = (S.c / S.c_root) ** sp.ei_taper_exp
    t2 = np.asarray(tau, float) ** 2
    S.cal = dict(cal)
    S.EI = cal["EI_root0"] * shape * S.ei_mult * t2
    S.GJ = cal["GJ_root0"] * shape * S.gj_mult * t2
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


class FlexBodyModelB2(fbb1.FlexBodyModelB1):
    """FlexBodyModelB1 + B2a section genes. B2 genes at default -> the B1 model, untouched (bit-identical)."""

    def __init__(self, model: str, genes=None, shape_genes=None, geom: Optional[fb.Geometry] = None,
                 asymmetric: bool = False, wing_extra_modes: int = 0, root_v2: str = fb.ROOT_V2,
                 wing_n_sel: Optional[Tuple[int, int, int]] = None, wing_mesh: Optional[Dict] = None):
        d = pb2.decode_shape_b2(shape_genes, model)                   # raises before any build
        self.shape_b2_all = d
        self.b2_genes = pb2.b2_part(d)
        self.b2_baseline = pb2.is_baseline_b2(d, model)
        kw = dict(geom=geom, asymmetric=asymmetric, wing_extra_modes=wing_extra_modes, root_v2=root_v2,
                  wing_n_sel=wing_n_sel, wing_mesh=wing_mesh)
        if self.b2_baseline:
            super().__init__(model, genes, shape_genes=pb2.b1_part(d), **kw)
            self._b2_zero()
            return
        # ---- forced shaped path (B1 rules 1-7 + B2). B1's constructor would short-circuit on a baseline B1 shape.
        self.shape_genes = pb1.decode_shape_b1(pb2.b1_part(d))
        self.planform_baseline = False
        self._pf_active = False
        self.root_v2 = root_v2
        fba1.FlexBodyModelA1.__init__(self, model, genes, **kw)
        n_el = int(self.wing_a1["n_el"])
        pf_geo = pb1.planform_from_wingparams(self.pw, self.shape_genes, n_el)
        self.planform_geo = pf_geo
        xi = pf_geo.xi
        F = pb2.FEATURES
        self.tau = pb2.tau_dist(d, xi) if F["thickness"] else np.ones_like(xi)
        self.dm = pb2.dm_dist(d, xi) if F["camber"] else np.zeros_like(xi)
        self.dgamma = math.radians(d["wing_dihedral_delta_deg"]) if F["dihedral"] else 0.0
        self.planform = replace(pf_geo, twist_rad=pf_geo.twist_rad + 2.0 * self.dm)   # aero twist from camber
        ns = self.ns
        self.twist_L = np.zeros(ns)
        self.twist_Q = np.zeros(self.N)
        self.twist_RB = np.zeros(len(fb.OUT_NAMES))
        self.twist_pitch = 0.0
        self.ac_shift_ft = 0.0
        self.bm_ref_ratio = 1.0
        self._b2_zero()
        bm1_a1 = self._root_bm_1g_per_n()
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
        self.wingR = planform_wing_surface_b2(wsp, cal, pf, dist["ei_R"], dist["gj_R"], dist["nsm_R"], zeta, self.tau,
                                              n_sel=wing_n_sel, extra_modes=wing_extra_modes)
        if self.asymmetric and (g.get("wing_asym_ei_delta", 0.0) != 0.0 or g.get("wing_asym_nsm_delta", 0.0) != 0.0):
            self.wingL = planform_wing_surface_b2(wsp, cal, pf, dist["ei_L"], dist["gj_L"], dist["nsm_L"], zeta, self.tau,
                                                  n_sel=wing_n_sel, extra_modes=wing_extra_modes)
        else:
            self.wingL = self.wingR
        self._pf_active = True
        self._assemble()                       # -> _build_load_bases (B1 Schrenk / AC hold / twist incl. camber, + B2)
        self.bm_ref_ratio = float(bm1_a1 / self._root_bm_1g_per_n())

    def _b2_zero(self):
        N, n_out = self.N, len(fb.OUT_NAMES)
        self.cam_M = np.zeros(self.ns)
        self.cam_Q, self.cam_RB = np.zeros(N), np.zeros(n_out)
        self.dih_L = np.zeros(self.ns)
        self.dih_Q, self.dih_RB = np.zeros(N), np.zeros(n_out)
        if not hasattr(self, "tau"):
            self.tau = None
            self.dm = None
            self.dgamma = 0.0

    def _build_load_bases(self):
        super()._build_load_bases()
        if not getattr(self, "_pf_active", False) or getattr(self, "b2_baseline", True):
            return
        st, bs = self.st, self.body_strips
        ns = self.ns
        self._b2_zero()
        # camber couple per unit q kappa_w (about the AC = pure couple, torque about the EA), wing strips only
        if np.any(self.dm != 0.0):
            cm = np.zeros(ns)
            for nm in ("wingR", "wingL"):
                r = bs[nm]
                cm[r] = st["c"][r] ** 2 * st["dy"][r] * (-math.pi * self.dm)
            self.cam_M = cm
            self.cam_Q = self.PsiT.T @ cm
            for o, nm in ((2, "wingR"), (3, "wingL")):
                self.cam_RB[o] = float(np.sum(cm[bs[nm]]))
        # dihedral: antisymmetric incidence sign * beta * dGamma per unit q kappa_w beta
        if self.dgamma != 0.0:
            L = np.zeros(ns)
            for nm in ("wingR", "wingL"):
                r = bs[nm]
                L[r] = st["lw"][r] * st["sign"][r] * self.dgamma
            self.dih_L = L
            self.dih_Q = self.W.T @ L
            for o, nm in ((0, "wingR"), (1, "wingL")):
                r = bs[nm]
                self.dih_RB[o] = float(L[r] @ st["arm"][r])
            for o, nm in ((2, "wingR"), (3, "wingL")):
                r = bs[nm]
                self.dih_RB[o] = float(L[r] @ st["ec"][r])

    def planform_summary(self) -> Dict:
        if self.b2_baseline:
            return super().planform_summary()
        pfa = self.planform
        self.planform = self.planform_geo                # geometric twist in the B1 summary keys
        try:
            d = super().planform_summary()
        finally:
            self.planform = pfa
        d["shape_cache_key"] = pb1.shape_cache_key(self.shape_genes)
        return d

    def b2_summary(self, model: Optional[str] = None) -> Dict:
        m = model or self.model
        xi = np.linspace(0.0, 1.0, 65)
        out = {"b2_baseline": bool(self.b2_baseline), "b2_genes": dict(self.b2_genes),
               "features": dict(pb2.FEATURES), **pb2.section_summary(self.shape_b2_all, m, xi)}
        if not self.b2_baseline:
            out.update(tau_root=float(self.tau[0]), tau_tip=float(self.tau[-1]),
                       cam_root_torque_per_qk=float(self.cam_RB[2]), dih_root_bm_per_qkb=float(self.dih_RB[0]),
                       f_wing_hz=[float(w / (2 * math.pi)) for w in self.wingR.omega])
        return out

    def node_layout(self, rp_offset_body_ft=None):
        return node_layout_b2(self, rp_offset_body_ft)


def node_layout_b2(mdl, rp_offset_body_ft=None):
    """node_layout_b1 + B2 (non-default B2 only): wing EA / LE / TE nodes lifted by z = -(y - y0) tan dGamma (delta only,
    so z -> 0 continuously at baseline), per-node absolute t/c and camber, dihedral metadata. Display geometry."""
    comps = fbb1.node_layout_b1(mdl, rp_offset_body_ft)
    if getattr(mdl, "b2_baseline", True):
        return comps
    d = mdl.shape_b2_all
    tg = math.tan(mdl.dgamma)
    for c in comps:
        if c["name"] not in ("wingR", "wingL"):
            continue
        body = getattr(mdl, c["name"])
        b = body.beam
        st = np.arange(b.n_el + 1) * b.h
        xi = np.clip(st / body.L, 0.0, 1.0)
        dz = -st * tg
        for key in ("axis_nodes_body_ft", "le_nodes_body_ft", "te_nodes_body_ft"):
            if key in c:
                c[key] = [[p[0], p[1], round(p[2] + float(z), 6)] for p, z in zip(c[key], dz)]
        c["tc_local"] = [round(float(v), 6) for v in pb2.tc_base(mdl.model, xi) * pb2.tau_dist(d, xi)]
        c["camber_meq_pct_local"] = [round(float(v), 6) for v in 100 * (pb2.m_base(mdl.model, xi) + pb2.dm_dist(d, xi))]
        c["dihedral_delta_deg"] = float(d["wing_dihedral_delta_deg"])
        c["dihedral_baseline_deg"] = float(pb2.SECTIONS[mdl.model]["dihedral0_deg"])
        c["section_baseline"] = pb2.SECTIONS[mdl.model]["section"]
        c["planform"] = c.get("planform", "") + " | P3-B2a: z = -(y - y0) tan(dGamma) (delta only), tc_local / camber_meq_pct_local"
    return comps


# =====================================================================================================================
# Energy meter (read-only FDM observations; never written back)
# =====================================================================================================================
class EnergyMeter:
    """Read-only drag / energy meter, one sample per coupler step (= per JSBSim frame, before the step, after the
    controller wrote its commands). INTERFACE_v2.md §15.10.

    PRIMARY (speed-robust, exactly 0 at B2 default):
      drag_increment_cd       = sum(dD_B2 V dt) / sum(qS V dt)   q.V-weighted mean B2 drag-coefficient increment, where
                                dD_B2 = aero/coefficient/b2_dCD0 + b2_dCDcam + b2_dCDwave (lbf, the native B2 drag functions)
      energy_drag_increment   = drag_increment_cd / CD_ref         (CD_ref frozen per aircraft / scenario IC)
    INFORMATIONAL: drag_work_ftlbf = sum(D V dt) (total aero drag), energy_drag_ratio = work / (P_ref t_flown),
      throttle_mean / _max_cmd / _sat_frac, fuel_burned_lbs, alpha_max_deg.
    SPEED-HOLD CHECK: speed_deficit_kts_mean = mean(max(0, V_target - V_cas)), speed_low_frac = fraction of frames with
      V_cas < V_target - 5 kt, speed_hold_ok = deficit_mean <= 2 kt (V_target = IC KCAS, as sim.py)."""
    B2_DRAG = ("aero/coefficient/b2_dCD0", "aero/coefficient/b2_dCDcam", "aero/coefficient/b2_dCDwave")
    DEFICIT_OK_KTS = 2.0
    SAT_OK_FRAC = None          # informational only: Phase-1 baselines saturate 15-25 % of frames on climb steps
    LOW_KTS = 5.0

    def __init__(self, dt: float, throttle_max: float = 1.0, v_target_kts: Optional[float] = None,
                 b2_drag: bool = False):
        self.dt = float(dt)
        self.thr_max = float(throttle_max)
        self.v_target = v_target_kts
        self.b2_drag = bool(b2_drag)
        self.n_sat = 0
        self.thr_peak = -math.inf
        self.alpha_peak = -math.inf
        self.n = 0
        self.work = 0.0
        self.wd = 0.0
        self.wq = 0.0
        self.deficit = 0.0
        self.n_low = 0
        self.thr = 0.0
        self.fuel0 = None
        self.fuel = None

    def observe(self, fdm):
        g = fdm.__getitem__
        a = g("aero/alpha-rad")
        V = g("velocities/vt-fps")
        drag = -g("forces/fbx-aero-lbs") * math.cos(a) - g("forces/fbz-aero-lbs") * math.sin(a)
        self.work += drag * V * self.dt
        self.wq += g("aero/qbar-area") * V * self.dt
        if self.b2_drag:
            self.wd += (g(self.B2_DRAG[0]) + g(self.B2_DRAG[1]) + g(self.B2_DRAG[2])) * V * self.dt
        tc = g("fcs/throttle-cmd-norm")
        self.thr += tc
        self.thr_peak = max(self.thr_peak, tc)
        self.n_sat += tc >= self.thr_max - 1e-6
        self.alpha_peak = max(self.alpha_peak, a)
        vc = g("velocities/vc-kts")
        if self.v_target is None:
            self.v_target = vc
        self.deficit += max(0.0, self.v_target - vc)
        self.n_low += vc < self.v_target - self.LOW_KTS
        f = g("propulsion/total-fuel-lbs")
        if self.fuel0 is None:
            self.fuel0 = f
        self.fuel = f
        self.n += 1

    def result(self, ref: Optional[Dict]) -> Dict:
        nan = float("nan")
        t = self.n * self.dt
        P0 = (ref or {}).get("drag_power_ref_ftlbf_s")
        CD0 = (ref or {}).get("cd_ref")
        dcd = (self.wd / self.wq if self.wq > 0 else nan) if self.b2_drag else 0.0
        n = max(self.n, 1)
        r = {"drag_increment_cd": float(dcd),
             "energy_drag_increment": (float(dcd / CD0) if CD0 else nan) if self.b2_drag else 0.0,
             "drag_work_ftlbf": float(self.work), "t_flown_s": float(t),
             "energy_drag_ratio": float(self.work / (P0 * t)) if (P0 and t > 0) else nan,
             "throttle_mean": float(self.thr / n) if self.n else nan,
             "throttle_max_cmd": float(self.thr_peak) if self.n else nan,
             "throttle_sat_frac": float(self.n_sat / n) if self.n else nan,
             "fuel_burned_lbs": float(self.fuel0 - self.fuel) if self.fuel0 is not None else nan,
             "alpha_max_deg": float(math.degrees(self.alpha_peak)) if self.n else nan,
             "v_target_kcas": float(self.v_target) if self.v_target is not None else nan,
             "speed_deficit_kts_mean": float(self.deficit / n) if self.n else nan,
             "speed_low_frac": float(self.n_low / n) if self.n else nan,
             "drag_power_ref_ftlbf_s": float(P0) if P0 else nan, "cd_ref": float(CD0) if CD0 else nan,
             "ref_source": (ref or {}).get("source", "none")}
        r["speed_hold_ok"] = bool(self.n > 0 and r["speed_deficit_kts_mean"] <= self.DEFICIT_OK_KTS)
        return r


# =====================================================================================================================
# Coupler
# =====================================================================================================================
class FlexBodyCouplerB2(fbb1.FlexBodyCouplerB1):
    """FlexBodyCouplerB1 + camber couple and dihedral beta loads (load vectors only, no rigid feedback) + optional
    read-only EnergyMeter. No B2 load vectors -> FlexBodyCouplerB1 arithmetic exactly."""

    def __init__(self, mdl, mode: str = "twoway", substeps: int = 2, zero_feedback: bool = False,
                 energy: Optional[EnergyMeter] = None):
        super().__init__(mdl, mode=mode, substeps=substeps, zero_feedback=zero_feedback)
        cq = getattr(mdl, "cam_Q", None)
        dq = getattr(mdl, "dih_Q", None)
        self._x = bool((cq is not None and np.any(cq != 0.0)) or (dq is not None and np.any(dq != 0.0)))
        self.energy = energy

    def _extra_Q(self, s, qk):
        m = self.mdl
        return qk * (m.twist_Q + m.cam_Q) + (qk * s["beta"]) * m.dih_Q

    def initialize(self, fdm):
        if not self._x:
            return super().initialize(fdm)
        m = self.mdl
        s = self.read_state(fdm)
        self.ref = {"alpha": s["alpha"], "elev": s["elev"], "rud": s["rud"], "beta": s["beta"]}
        kap = m.kappas(s["mach"])
        self._qk_ref = s["qbar"] * kap["w"]
        Q = self.coefs(s, kap) @ m.Qbasis + self._extra_Q(s, self._qk_ref)
        eta = np.linalg.solve(m.K, Q)
        self.eta, self.etad, self.etadd = eta.copy(), np.zeros(m.N), np.zeros(m.N)
        self.eta_ref = eta.copy()
        self.Qref_per_q = {g: m.A_K[g] @ eta for g in fb.GROUPS}
        self._apply(fdm, s, kap)
        self.out_1g = self.out.copy()

    def step(self, fdm, dt: float):
        if self.energy is not None:
            self.energy.observe(fdm)
        if not self._x:
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
             + (qw * s["beta"]) * (m.A_beta @ de) + self._extra_Q(s, qw))
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
        super()._apply(fdm, s, kap)           # B1: + q kappa_w twist_RB when twist loads exist; no rigid feedback
        if not self._x:
            return
        m = self.mdl
        qk = s["qbar"] * kap["w"]
        if not self._tw:                      # B1 adds twist_RB only when its own flag is set
            self.out = self.out + qk * m.twist_RB
        self.out = self.out + qk * m.cam_RB + (qk * s["beta"]) * m.dih_RB
        for i, nm in enumerate(fb.OUT_NAMES):
            self.last[nm] = float(self.out[i])


# =====================================================================================================================
# Sizing / margins / flown terms
# =====================================================================================================================
def wing_design_loads_b2(mdl, wts: fb.StructWeightsV2 = fb.StructWeightsV2(), load_scale: float = 1.0) -> Dict:
    """flexbody_b1.wing_design_loads_b1 + the camber couple at q_D in the torque (load_scale scales twist + camber)."""
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
        Lt = load_scale * qk * mdl.twist_L[r]
        bend = n * W * float(lift @ arm) - n * fb.G0 * float(mdy @ arm) + float(Lt @ arm)
        t_lift = (n * W * float(lift @ ec) + n * fb.G0 * float(mdy @ xth) + float(Lt @ ec)
                  + load_scale * qk * float(np.sum(mdl.cam_M[r])))
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


def sizing_b2(mdl, wts: fb.StructWeightsV2 = fb.StructWeightsV2()) -> Dict:
    if getattr(mdl, "b2_baseline", True):
        return fbb1.sizing_b1(mdl, wts)
    sz = fb.sizing_v2(mdl, wts)
    g = mdl.genes
    ms = 1.0 + wts.design_margin_of_safety
    base = fbb1._baseline_wing_design(mdl.model, mdl.root_v2, wts)
    c_on, c_off = wing_design_loads_b2(mdl, wts, 1.0), wing_design_loads_b2(mdl, wts, 0.0)
    cur = {sd: {k: max(c_on[sd][k], c_off[sd][k]) for k in c_on[sd]} for sd in c_on}   # one-sided (no relief credit)
    pf, ex = mdl.planform, mdl.pw.ei_taper_exp
    eta_t = float(wts.tip_bm_eta)
    tau_r = float(pb2.tau_dist(mdl.shape_b2_all, [0.0])[0]) if pb2.FEATURES["thickness"] else 1.0
    tau_t = float(pb2.tau_dist(mdl.shape_b2_all, [eta_t])[0]) if pb2.FEATURES["thickness"] else 1.0
    c_root = float(pf.area_norm) ** ex
    c_tip = float(pb1.chord_multipliers_raw(mdl.shape_genes, [eta_t])[0] * pf.area_norm) ** ex
    g_root, g_tip = c_root * tau_r, c_tip * tau_t
    d = g.get("wing_asym_ei_delta", 0.0)
    dist = fb.wing_distributions(g, np.array([eta_t]))
    for side, sg in (("R", 1.0), ("L", -1.0)):
        s_ei = g["wing_ei_root"] * (1 + sg * d)
        s_gj = s_ei * g["wing_gj_ratio_root"]
        s_tip = float(dist["ei_" + side][0])
        for k, sc, gs, b in (("bm", s_ei, g_root, base["root"][side]["bm"]), ("torque", s_gj, g_root, base["root"][side]["torque"]),
                             ("ip", s_ei, c_root, base["root"][side]["ip"]), ("tip_bm", s_tip, g_tip, base["tip"][side])):
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
    sz["geometric_strength_factor"] = {"root": g_root, "tip_bm_eta": g_tip, "ip_root": c_root, "tau_root": tau_r,
                                       "tau_tip_bm_eta": tau_t}
    return sz


def margin_terms_b2(mdl, wts: fb.StructWeightsV2 = fb.StructWeightsV2(), gate: float = 1.0) -> Dict:
    if getattr(mdl, "b2_baseline", True):
        return fbb1.margin_terms_b1(mdl, wts, gate=gate)
    r = fb.margin_terms_v2(mdl, wts, gate=gate)
    sz = sizing_b2(mdl, wts)
    r["terms"].update(sz["terms"])
    r["sizing"] = sz
    return r


def response_terms_b2(hist, mdl, out_1g, wts: fb.StructWeightsV2 = fb.StructWeightsV2(), qk_ref: float = 0.0) -> Dict:
    if getattr(mdl, "b2_baseline", True):
        return fbb1.response_terms_b1(hist, mdl, out_1g, wts, qk_ref=qk_ref)
    return fbb1._response_terms_with_sizing(hist, mdl, out_1g, wts, sizing_b2(mdl, wts), qk_ref=qk_ref)


# =====================================================================================================================
# Prepared root <root>_v2b2: <root>_v2 + native B2 increment functions (property-driven, 0 by default)
# =====================================================================================================================
PREPARE_FMT_B2 = 2   # 2: camber polar CL from sqrt(aero/cl-squared)


def _tab(var: str, rows) -> str:
    body = "\n".join(f"          {x:.6f} {y:.9f}" for x, y in rows)
    return (f"<table><independentVar lookup=\"row\">{var}</independentVar><tableData>\n{body}\n"
            f"        </tableData></table>")


def native_xml(model: str, pw) -> Dict[str, str]:
    """Per-axis function blocks + the interface property declarations (hashed into model_version)."""
    P = "flexbody/b2/"
    mt, pg, sf = pb2.mach_table(model, pw), pb2.pg_table(), pb2.stall_fade_table(model)
    decl = "".join(f'\n    <property value="0">{P}{k}</property>' for k in pb2.NATIVE_PROPS) + "\n"

    def fn(name, desc, inner):
        return (f"\n      <function name=\"aero/coefficient/b2_{name}\">\n        <description>{desc}</description>\n"
                f"        {inner}\n      </function>")
    cl = fn("dCL0", "P3-B2a camber dCL0 + downwash (native, trimmed)",
            f"<product><property>aero/qbar-area</property><property>{P}dCL0</property>"
            f"{_tab('velocities/mach', mt)}{_tab('aero/alpha-rad', sf)}</product>")
    cd0 = fn("dCD0", "P3-B2a thickness form-factor dCD0", f"<product><property>aero/qbar-area</property><property>{P}dCD0</property></product>")
    # CL source: sqrt(aero/cl-squared) (JSBSim's own CDi input; forces/fwz-aero-lbs reads 0 INSIDE aero functions)
    cdc = fn("dCDcam", "P3-B2a camber profile polar Kp[(CL-cli')^2-(CL-cli0)^2] = cam_A + cam_B |CL|",
             f"<product><property>aero/qbar-area</property><sum><property>{P}cam_A</property>"
             f"<product><property>{P}cam_B</property><pow><property>aero/cl-squared</property><value>0.5</value></pow>"
             f"</product></sum></product>")

    def mcr(which):
        return (f"<difference><property>velocities/mach</property><difference><property>{P}korn_c_{which}</property>"
                f"<product><property>{P}korn_kcl</property><pow><property>aero/cl-squared</property><value>0.5</value></pow>"
                f"</product></difference></difference>")
    cdw = fn("dCDwave", "P3-B2a Korn/Lock wave-drag difference 20[(M-Mcr')^4-(M-Mcr0)^4]+",
             f"<product><property>aero/qbar-area</property><value>20.0</value><difference>"
             f"<pow><max><value>0.0</value>{mcr('new')}</max><value>4.0</value></pow>"
             f"<pow><max><value>0.0</value>{mcr('base')}</max><value>4.0</value></pow></difference></product>")
    clb = fn("dClb", "P3-B2a dihedral dClb (lifting line, native)",
             f"<product><property>aero/qbar-area</property><property>metrics/bw-ft</property>"
             f"<property>aero/beta-rad</property><property>{P}dClb</property>{_tab('velocities/mach', mt)}</product>")
    cm = fn("dCm0", "P3-B2a camber dCm0 wing (DATCOM) + downwash at tail (native, trimmed)",
            f"<product><property>aero/qbar-area</property><property>metrics/cbarw-ft</property>"
            f"<property>{P}dCm0</property>{_tab('velocities/mach', pg)}</product>")
    return {"decl": decl, "LIFT": cl, "DRAG": cd0 + cdc + cdw, "ROLL": clb, "PITCH": cm}


def root_v2b2_for(root: str) -> str:
    return os.path.normpath(root) + "_v2b2"


def prepare_aircraft_v2b2(model: str, root_v2: str, root_v2b2: str) -> str:
    """Copy <root_v2>/{engine,systems,aircraft/<model>} -> root_v2b2 and append the native B2 functions. Never touches
    root_v2 or the installed JSBSim data."""
    fb.ensure_root_v2(model, root_v2)
    os.makedirs(os.path.join(root_v2b2, "aircraft"), exist_ok=True)
    for sub in ("engine", "systems"):
        src = os.path.join(root_v2, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(root_v2b2, sub), dirs_exist_ok=True)
    final = os.path.join(root_v2b2, "aircraft", model)
    dst = os.path.join(root_v2b2, "aircraft", f".{model}.tmp{os.getpid()}")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(root_v2, "aircraft", model), dst)
    xml = os.path.join(dst, model + ".xml")
    with open(xml, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    geom = fb.geometry_for(model, root_v2)
    pw = fw.params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
    blk = native_xml(model, pw)
    m = re.search(r"<aerodynamics[^>]*>", txt)
    txt = txt[:m.end()] + blk["decl"] + txt[m.end():]
    for ax in ("LIFT", "DRAG", "ROLL", "PITCH"):
        m = re.search(rf'<axis\s+name="{ax}"[^>]*>(.*?)</axis>', txt, re.S)
        if m is None:
            raise RuntimeError(f"{model}: no <axis name=\"{ax}\"> in {xml}")
        txt = txt[:m.end(1)] + blk[ax] + "\n    " + txt[m.end(1):]
    with open(xml, "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(dst, "flexbody_b2_meta.json"), "w") as f:
        import json
        json.dump({"fmt": PREPARE_FMT_B2, "props": list(pb2.NATIVE_PROPS), "source_root_v2": os.path.basename(root_v2),
                   "xml_sha": _native_sha(model, pw)}, f)
    old = None
    if os.path.isdir(final):                       # swap in (no half-written root visible to a concurrent loader)
        old = os.path.join(root_v2b2, "aircraft", f".{model}.old{os.getpid()}")
        os.rename(final, old)
    os.rename(dst, final)
    if old:
        shutil.rmtree(old, ignore_errors=True)
    return root_v2b2


def _native_sha(model: str, pw) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(native_xml(model, pw), sort_keys=True).encode()).hexdigest()[:16]


def ensure_root_v2b2(model: str, root_v2: str, root_v2b2: str) -> str:
    import json
    p = os.path.join(root_v2b2, "aircraft", model, "flexbody_b2_meta.json")
    ok = False
    if os.path.exists(p):
        with open(p) as f:
            meta = json.load(f)
        geom = fb.geometry_for(model, root_v2)
        pw = fw.params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
        ok = meta.get("fmt") == PREPARE_FMT_B2 and meta.get("xml_sha") == _native_sha(model, pw)
    if not ok:
        prepare_aircraft_v2b2(model, root_v2, root_v2b2)
    return root_v2b2
