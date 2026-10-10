"""flexbody_a1.py -- P3-A1: denser 'full' structural model (opt-in fidelity 'full_a1'), built ON TOP of flexbody.py.

Nothing in flexbody.py / flexwing.py / flexeval.py / coupled_sim.py is modified: those files are hashed into the pinned
reduced / full model_version strings (v2_results/model_versions_post_p25.json), so A1 lives in its own modules
(flexbody_a1.py + flexeval_a1.py) and the existing 'full' fidelity resolves to byte-identical code and outputs.

A1 versus 'full' (flexv2):
  wing (each side)   WING_A1["n_el"] Hermite bending + linear torsion + in-plane elements = aero strips (full: 32)
                     modes 4 bending + 3 torsion + 2 in-plane = 9 per semi-wing          (full: 3 + 2 + 1 = 6)
  HT / VT / fuselage identical to full (12 elements; HT 2b+1t, VT 2b+1t, fuselage 2+2)
  -> 31 modal DOF per aircraft (full: 25)
  genes, P2.5 ranges, weights, margins, sizing, coupler, mass feedback and telemetry are flexbody's own functions applied
  to the denser model (FlexBodyModelA1 is a FlexBodyModel), with ONE numerical difference: J_wing_tip_bm_limit (eta 0.875,
  w 1.0) is evaluated station-exactly (sizing_a1: partial-strip load integration outboard of eta 0.875 and the local EI
  multiplier interpolated AT 0.875). flexbody.sizing_v2 takes whole strips with centre >= 0.875 and the EI multiplier of
  the strip centre nearest 0.875 (ties -> inboard), which makes the term jump non-monotonically with the strip count
  (c172x tip-soft genome: 0.0175 / 0.0155 / 0.0096 / 0.0136 / 0.0107 at 32 / 48 / 64 / 96 / 128 strips); the station-exact
  value converges (0.01183 -> 0.01189). Baseline value is still exactly 0. The strip-discrete value is kept as a
  diagnostic (sizing['tip_bm_strip_discrete']).
The wing stiffness calibration (uncoupled baseline f_b1 / f_t1 / f_ip) and the aileron Cl_da strip calibration are redone
on the A1 mesh with flexbody's own rules, so the baseline first frequencies match full's targets.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

import flexwing as fw
import flexbody as fb

A1_FIDELITY = "full_a1"
A1_TAG = "flexv2a1"
A1_FMT = 1
# 64 strips per semi-wing (strip-count study, v2_results/p3a1_truncation.json 'strip_convergence_pct_vs_128', A1 modes,
# baseline + tip-soft, 4 aircraft): every metric (flutter speed / margin, divergence, root and eta 0.875 bending moments
# static + gust, tip deflections, in-plane responses) is within 0.08 % of 128 strips at 64 (48: 0.15 %, 32: 0.37 %, worst =
# T38 eta 0.875 moment, O(h)), except the aileron-reversal margin (<= 0.51 % at 48-96, 1.05 % at 32; non-monotone:
# aileron-edge strip quantisation). Strips only enter the one-off per-genome model build + margin screen (the coupler
# integrates 31 modal DOF whatever the strip count): 64 vs 48 strips costs +0.03-0.07 s CPU per genome (~1 % of a
# 3-scenario evaluate, p3a1_benchmark.json 'preflight_cpu_s') and halves the outboard-moment discretisation error.
WING_A1 = dict(n_el=64, n_b=4, n_t=3, n_ip=2, f_ip_ratio=fb.WING_V2["f_ip_ratio"])


def a1_wing_config(overrides: Optional[Dict] = None) -> Dict:
    w = dict(WING_A1)
    if overrides:
        bad = set(overrides) - set(WING_A1)
        if bad:
            raise ValueError(f"unknown A1 wing mesh keys {sorted(bad)} (have {sorted(WING_A1)})")
        w.update(overrides)
    if int(w["n_el"]) < 8 or min(int(w["n_b"]), int(w["n_t"]), int(w["n_ip"])) < 1:
        raise ValueError(f"invalid A1 wing config {w}")
    return w


def a1_params(model: str, geom: fb.Geometry, wing: Optional[Dict] = None) -> Dict:
    """Everything that defines the A1 model of an aircraft (building AND model_version hashing): flexbody.v2_params with
    the wing mesh / mode set replaced by the A1 one, plus the A1 marker and format."""
    p = fb.v2_params(model, geom)
    p["wing_v2"] = dict(wing or WING_A1)
    p["variant"] = "a1"
    p["a1_fmt"] = A1_FMT
    return p


class FlexBodyModelA1(fb.FlexBodyModel):
    """FlexBodyModel with the A1 wing (denser strips, 4b + 3t + 2ip). Same constructor as FlexBodyModel plus
    `wing_mesh` (dict overriding WING_A1 keys; used by the strip-convergence / truncation studies only -- the 'full_a1'
    fidelity always uses WING_A1). wing_n_sel / wing_extra_modes as in FlexBodyModel (truncation studies).

    The body is flexbody.FlexBodyModel.__init__ with the wing mesh / mode numbers taken from the A1 config instead of
    flexbody.WING_V2 (flexbody.py itself cannot be edited without changing the pinned full / reduced model_version)."""

    def __init__(self, model: str, genes=None, geom: Optional[fb.Geometry] = None, asymmetric: bool = False,
                 wing_extra_modes: int = 0, root_v2: str = fb.ROOT_V2, wing_n_sel: Optional[Tuple[int, int, int]] = None,
                 wing_mesh: Optional[Dict] = None):
        W = a1_wing_config(wing_mesh)
        self.wing_a1 = W
        self.model = model
        self.asymmetric = bool(asymmetric)
        self.genes = fb.decode_genome_v2(genes, self.asymmetric)
        self.geom = geom or fb.geometry_for(model, root_v2)
        self.params = a1_params(model, self.geom, W)
        g, P = self.genes, self.params
        pw = fw.WingParams(**P["wing"])
        self.pw = pw
        pr = P["v2"]
        zeta = g["struct_damping_ratio"]
        wsp = fb.SurfaceSpec(name="wing", kind="wing", span_ft=pw.span_ft, area_ft2=pw.area_ft2, taper=pw.taper,
                             sweep_deg=pw.sweep_deg, root_frac=pw.root_frac, mass_lb=pw.wing_mass_lb / 2, f_b1_hz=pw.f_b1_hz,
                             f_t1_hz=pw.f_t1_hz, f_ip_hz=W["f_ip_ratio"] * pw.f_b1_hz, x_ea=pw.x_ea, x_cg=pw.x_cg,
                             aspect_eff=pw.span_ft ** 2 / pw.area_ft2, n_el=int(W["n_el"]), n_b=int(W["n_b"]),
                             n_t=int(W["n_t"]), n_ip=int(W["n_ip"]), r_gyr=pw.r_gyr, ei_taper_exp=pw.ei_taper_exp,
                             mass_taper_exp=pw.mass_taper_exp, struct_frac=pw.struct_frac, w_ei=pw.w_ei,
                             ctrl_eta=tuple(pw.ail_eta), ctrl_cf=pw.ail_cf)
        # aileron strip calibration to the FDM's Cl_da (flexbody / v1 rule, on the A1 strips)
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
        self.dist = dist = fb.wing_distributions(g, xi)
        self.wingR = fb.Surface(wsp, dist["ei_R"], dist["gj_R"], dist["nsm_R"], zeta, n_sel=wing_n_sel,
                                extra_modes=wing_extra_modes)
        if self.asymmetric and (g.get("wing_asym_ei_delta", 0.0) != 0.0 or g.get("wing_asym_nsm_delta", 0.0) != 0.0):
            self.wingL = fb.Surface(wsp, dist["ei_L"], dist["gj_L"], dist["nsm_L"], zeta, n_sel=wing_n_sel,
                                    extra_modes=wing_extra_modes)
        else:
            self.wingL = self.wingR
        # ---- empennage + fuselage: exactly flexbody.FlexBodyModel's (same specs, meshes and mode sets as 'full')
        ts = g["tail_stiffness_scale"]
        geo = self.geom
        h, v, f = pr["ht"], pr["vt"], pr["fus"]
        hsp = fb.SurfaceSpec(name="ht", kind="ht", span_ft=math.sqrt(h["AR"] * geo.sh_ft2), area_ft2=geo.sh_ft2, taper=h["taper"],
                             sweep_deg=h["sweep"], root_frac=h["root_frac"], mass_lb=h["mass_lb"] / 2, f_b1_hz=h["f_b1"],
                             f_t1_hz=h["f_t1"], f_ip_hz=None, x_ea=h["x_ea"], x_cg=h["x_cg"], aspect_eff=h["AR"],
                             n_el=fb.TAIL_N_EL, n_b=2, n_t=1, ctrl_eta=tuple(h["eta"]) if h["eta"] else None, ctrl_cf=h["cf"],
                             all_moving=h["all_moving"])
        hv = math.sqrt(v["AR"] * geo.sv_ft2)
        vsp = fb.SurfaceSpec(name="vt", kind="vt", span_ft=2 * hv, area_ft2=2 * geo.sv_ft2, taper=v["taper"], sweep_deg=v["sweep"],
                             root_frac=0.0, mass_lb=v["mass_lb"], f_b1_hz=v["f_b1"], f_t1_hz=v["f_t1"], f_ip_hz=None,
                             x_ea=v["x_ea"], x_cg=v["x_cg"], aspect_eff=1.55 * v["AR"],
                             n_el=fb.TAIL_N_EL, n_b=2, n_t=1, ctrl_eta=tuple(v["eta"]), ctrl_cf=v["cf"])
        self.ht_spec, self.vt_spec = hsp, vsp
        tm = np.full(fb.TAIL_N_EL, ts)
        self.htR = fb.Surface(hsp, tm, tm, None, zeta)
        self.htL = self.htR
        self.vt = fb.Surface(vsp, tm, tm, None, zeta)
        self.l_h = geo.lh_ft
        self.l_v = v["arm_ft"] if v.get("arm_ft") else geo.lv_ft
        self.z_v = v["z_ft"]
        self.downwash = h["downwash"]
        fs = g["fuselage_stiffness_scale"]
        tail0 = h["mass_lb"] + v["mass_lb"]
        tail = 2 * self.htR.mass_lb + self.vt.mass_lb
        self.fusV = fb.Fuselage(self.l_h, f["aft_mass_lb"], tail0, tail, f["f_v1"], f["struct_frac"], fs, fb.FUS_N_EL)
        self.fusL = fb.Fuselage(self.l_h, f["aft_mass_lb"], tail0, tail, f["f_l1"], f["struct_frac"], fs, fb.FUS_N_EL)
        self._assemble()


# =====================================================================================================================
# Sizing on the A1 model: flexbody.sizing_v2 with a station-exact outboard (eta = tip_bm_eta) bending check
# =====================================================================================================================
def wing_tip_bm_station(mdl: fb.FlexBodyModel, wts: fb.StructWeightsV2 = fb.StructWeightsV2(),
                        baseline_mass: bool = False) -> Dict[str, float]:
    """Design bending moment (lbf*ft, magnitude) about the beam station y_c = tip_bm_eta * L, per side: the same formula as
    flexbody.wing_design_loads 'tip_bm' (n W_d Schrenk lift - n g m dy, rigid bases at n_limit) but integrating each
    strip's load only over the part of the strip outboard of y_c (uniform within the strip, lever = centroid of that part).
    Equals the strip-discrete value when y_c falls on a strip boundary; converges O(h^2) with the strip count."""
    pw, st, bs = mdl.pw, mdl.st, mdl.body_strips
    n, W = pw.n_limit, mdl.geom.empty_wt_lb
    out = {}
    for side, surf in (("R", mdl.wingR), ("L", mdl.wingL)):
        r = bs["wing" + side]
        arm = st["arm"][r]
        mdy = (surf.m0 if baseline_mass else surf.m) * surf.dy
        lift = mdl.Lb[fb.CI["lift"], r]
        y_c = float(wts.tip_bm_eta) * surf.L
        lo = np.maximum(arm - 0.5 * surf.dy, y_c)
        hi = arm + 0.5 * surf.dy
        frac = np.clip((hi - lo) / surf.dy, 0.0, 1.0)
        lev = np.where(frac > 0.0, 0.5 * (lo + hi) - y_c, 0.0)
        out[side] = abs(n * W * float(lift @ (frac * lev)) - n * fb.G0 * float(mdy @ (frac * lev)))
    return out


def sizing_a1(mdl: fb.FlexBodyModel, wts: fb.StructWeightsV2 = fb.StructWeightsV2()) -> Dict:
    """flexbody.sizing_v2 (same checks, allowables, weights) with the outboard bending check evaluated AT tip_bm_eta:
    demand = wing_tip_bm_station(current masses), allowable = (1 + MS) x wing_tip_bm_station(baseline masses) x local EI
    multiplier interpolated at tip_bm_eta (flexbody.wing_distributions, PCHIP of the taper chain, asym delta included).
    All other entries are flexbody.sizing_v2's unchanged. The strip-discrete tip values are kept under
    'tip_bm_strip_discrete' (diagnostic)."""
    sz = fb.sizing_v2(mdl, wts)
    disc = {"term": sz["terms"]["J_wing_tip_bm_limit"],
            "ratios": {k: sz["ratios"][k] for k in ("wingR_tip_bm", "wingL_tip_bm")}}
    ms = 1.0 + wts.design_margin_of_safety
    cur, base = wing_tip_bm_station(mdl, wts), wing_tip_bm_station(mdl, wts, baseline_mass=True)
    dist = fb.wing_distributions(mdl.genes, np.array([float(wts.tip_bm_eta)]))
    for side in ("R", "L"):
        nm = f"wing{side}_tip_bm"
        sz["allowables_lbft"][nm] = ms * base[side] * float(dist["ei_" + side][0])
        sz["demand_lbft"][nm] = cur[side]
        sz["ratios"][nm] = sz["demand_lbft"][nm] / sz["allowables_lbft"][nm]
    r = max(sz["ratios"]["wingR_tip_bm"], sz["ratios"]["wingL_tip_bm"])
    sz["terms"]["J_wing_tip_bm_limit"] = wts.w_wing_tip_bm_limit * max(0.0, r - 1.0) ** 2
    sz["tip_bm_method"] = "station_exact"
    sz["tip_bm_strip_discrete"] = disc
    return sz


def margin_terms_a1(mdl: fb.FlexBodyModel, wts: fb.StructWeightsV2 = fb.StructWeightsV2(), gate: float = 1.0) -> Dict:
    """flexbody.margin_terms_v2 (margins, gate, J_mass, J_smooth, sizing) with sizing_a1 in place of sizing_v2."""
    r = fb.margin_terms_v2(mdl, wts, gate=gate)
    sz = sizing_a1(mdl, wts)
    r["terms"].update(sz["terms"])
    r["sizing"] = sz
    return r


# =====================================================================================================================
# Modal truncation / strip convergence (P3-A1 deliverable 2)
# =====================================================================================================================
def _station_bm(arm: np.ndarray, f: np.ndarray, y_c: float) -> np.ndarray:
    """Bending moment at beam station y_c by force summation of the strip loads outboard of it (f: (..., n_strips))."""
    ob = arm >= y_c
    return f[..., ob] @ (arm[ob] - y_c)


def wing_metrics_a1(mdl: fb.FlexBodyModel, tip_eta: float = 0.875, with_margins: bool = True) -> Dict[str, float]:
    """Right-wing block responses (works for any FlexBodyModel): flexbody._wing_metrics' quantities (static aeroelastic at
    0.9 V_D with 1 deg + 1 g, 1-cos gust, flutter speed to 4 V_D) PLUS the elastic bending moment at the sizing station
    eta = tip_eta (force summation outboard; static and gust peak) and the gate's wing-block flutter / divergence /
    aileron-reversal margins (flexbody.margins_v2, wingR block), and in-plane responses (1 g fore-aft static tip
    deflection, 1-cos n_x pulse peak root in-plane moment / tip deflection)."""
    s = mdl.wingR
    idx = fb._block(mdl, ("wingR",))
    B = fb.BlockAero(mdl, idx)
    vd = mdl.pw.v_dive_keas
    q = fb._q_of_keas(0.9 * vd)
    kap = B.kap_q(q)
    A = B.AKq(q)
    st, r = mdl.st, mdl.body_strips["wingR"]
    lw, arm, mdy = st["lw"][r], st["arm"][r], st["mdy"][r]
    Wr, Gr, Hr, Wi = mdl.W[r][:, idx], mdl.G[r][:, idx], mdl.H[r][:, idx], mdl.W_in[r][:, idx]
    y_c = tip_eta * s.L
    L0 = q * kap["w"] * lw * math.radians(1.0)
    F0 = -fb.G0 * mdy
    eta = np.linalg.solve(B.K - q * A, Wr.T @ L0 + Wi.T @ F0)
    f_st = L0 + q * kap["w"] * lw * (Gr @ eta) + F0
    out = {"static_root_bm": float(arm @ f_st), "static_tip_bm_eta": float(_station_bm(arm, f_st, y_c)),
           "static_tip_w": float(s.tipW @ eta)}
    vt = math.sqrt(2 * q / fb.RHO0)
    dt = 1 / 960
    t = np.arange(int(2.0 / dt)) * dt
    wg = np.where(t < 0.5, 10.0 * (1 - np.cos(2 * math.pi * t / 0.5)), 0.0)
    Lg = np.outer(q * kap["w"] * wg / vt, lw)
    Ce = B.C - (q / vt) * (kap["w"] * B.AC["w"] + kap["h"] * B.AC["h"] + kap["v"] * B.AC["v"] + B.Anc)
    x, v, a = fw.newmark(np.eye(idx.size), Ce, B.K - q * A, Lg @ Wr, dt)
    f = Lg + q * kap["w"] * ((x @ Gr.T) + (v @ Hr.T) / vt) * lw - (a @ Wi.T) * mdy
    out.update(gust_peak_root_bm=float(np.max(np.abs(f @ arm))),
               gust_peak_tip_bm_eta=float(np.max(np.abs(_station_bm(arm, f, y_c)))),
               gust_peak_tip_w=float(np.max(np.abs(x @ s.tipW))))
    # in-plane (chordwise): 1 g fore-aft inertia load -> static tip in-plane deflection; 1-cos n_x pulse (0.1 s, peak 1 g)
    # -> peak root in-plane moment incl. the modal inertia term (the static root moment is pure force summation and
    # therefore independent of the mode set; the out-of-plane metrics above do not see the in-plane modes at all)
    Qnx = mdl.Qbasis[fb.CI["nx"], idx]
    eta_v = np.linalg.solve(B.K - q * A, Qnx)
    o_ip = fb.OUT_NAMES.index("wingR_ip_bm")
    nx = np.where(t < 0.1, 0.5 * (1 - np.cos(2 * math.pi * t / 0.1)), 0.0)
    xv, vv, av = fw.newmark(np.eye(idx.size), Ce, B.K - q * A, np.outer(nx, Qnx), dt)
    m_ip = nx * float(mdl.RB[fb.CI["nx"], o_ip]) + av @ mdl.RA[idx, o_ip]
    out.update(ip_static_tip_v=float(s.tipV @ eta_v), ip_pulse_peak_root_bm=float(np.max(np.abs(m_ip))),
               ip_pulse_peak_tip_v=float(np.max(np.abs(xv @ s.tipV))))
    v_fl, _ = B.flutter_keas(4 * vd, n_grid=160)
    q_co = B.coalescence_q(16 * fb._q_of_keas(vd), n_grid=600)
    v_co = fb._keas_of_q(q_co) if math.isfinite(q_co) else math.inf
    out["flutter_keas_min_4VD"] = float(min(v_fl, v_co))
    if with_margins:
        m = fb.margins_v2(mdl, blocks=["wingR"])["blocks"]["wingR"]
        out.update(wing_flutter_margin=m["flutter_margin"], wing_div_margin=m["div_margin"],
                   wing_aileron_reversal_margin=m["aileron_reversal_margin"])
    out["f_wing_hz"] = [float(w / (2 * math.pi)) for w in s.omega]
    out["wing_mode_classes"] = s.cls.tolist()
    return out


def pct_delta(v0: float, v1: float) -> float:
    if math.isinf(v0) and math.isinf(v1):
        return 0.0
    if math.isinf(v0) or math.isinf(v1):
        return math.inf
    if v0 == 0.0:
        return 0.0 if v1 == 0.0 else math.inf
    return float(100 * (v1 - v0) / v0)


SCALAR_METRICS = ("static_root_bm", "static_tip_bm_eta", "static_tip_w", "gust_peak_root_bm", "gust_peak_tip_bm_eta",
                  "gust_peak_tip_w", "ip_static_tip_v", "ip_pulse_peak_root_bm", "ip_pulse_peak_tip_v",
                  "flutter_keas_min_4VD", "wing_flutter_margin", "wing_div_margin", "wing_aileron_reversal_margin")


def truncation_variants(base: Tuple[int, int, int] = None) -> Dict[str, Dict]:
    """N = the A1 set; per family +1 / +2 (b, t, ip), all families +1, and +1 / +2 next-lowest modes of any type."""
    b, t, v = base or (WING_A1["n_b"], WING_A1["n_t"], WING_A1["n_ip"])
    return {"N": dict(n_sel=(b, t, v)),
            "b+1": dict(n_sel=(b + 1, t, v)), "b+2": dict(n_sel=(b + 2, t, v)),
            "t+1": dict(n_sel=(b, t + 1, v)), "t+2": dict(n_sel=(b, t + 2, v)),
            "ip+1": dict(n_sel=(b, t, v + 1)), "ip+2": dict(n_sel=(b, t, v + 2)),
            "all+1": dict(n_sel=(b + 1, t + 1, v + 1)),
            "next+1": dict(n_sel=(b, t, v), extra=1), "next+2": dict(n_sel=(b, t, v), extra=2)}


def truncation_check_a1(model: str, genes=None, root_v2: str = fb.ROOT_V2, wing_mesh: Optional[Dict] = None,
                        variants: Optional[Dict[str, Dict]] = None) -> Dict:
    """Wing modal-truncation convergence of the A1 model: metrics for N (4b+3t+2ip) and each variant, % change vs N."""
    variants = variants or truncation_variants()
    vals = {}
    for nm, cfg in variants.items():
        mdl = FlexBodyModelA1(model, genes, root_v2=root_v2, wing_n_sel=cfg["n_sel"], wing_extra_modes=cfg.get("extra", 0),
                              wing_mesh=wing_mesh)
        vals[nm] = wing_metrics_a1(mdl)
        vals[nm]["n_wing_modes"] = int(mdl.wingR.Phi.shape[1])
        vals[nm]["n_sel"] = list(cfg["n_sel"]) + ([f"+{cfg['extra']} any"] if cfg.get("extra") else [])
    pct = {nm: {k: pct_delta(vals["N"][k], vals[nm][k]) for k in SCALAR_METRICS} for nm in variants if nm != "N"}
    worst = {nm: max(abs(x) for x in d.values()) for nm, d in pct.items()}
    return {"values": vals, "pct_vs_N": pct, "max_abs_pct_vs_N": worst, "max_abs_pct": max(worst.values()) if worst else 0.0}
