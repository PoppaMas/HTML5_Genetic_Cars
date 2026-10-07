"""planform_b2.py -- P3-B2a section shape genes: schema, decode, section model, geometry gate, native increments.

Owner: Flight Dynamics (INTERFACE_v2.md section 15). Additive on planform_b1 (frozen, imported). Pure numpy + flexwing /
flexbody helpers. Genome / Evolution mirror THIS module.

B2a = 5 genes appended after the 6 B1 r1 genes (vector length 11), PER-AIRCRAFT ranges (decode needs the model):
  wing_dihedral_delta_deg      uniform geometric dihedral delta (deg, tip up +)
  wing_tc_root_scale           t/c multiplier at the beam root (baseline section of that aircraft)
  wing_tc_tip_ratio            tip multiplier / root multiplier; tau(eta) = tau_r (1 + (ratio - 1) eta)
  wing_camber_root_delta_pct   additive delta of the thin-airfoil-equivalent max camber m_eq (% chord) at the root
  wing_camber_tip_delta_pct    same at the tip; delta_m(eta) linear in eta
B2b (deferred, decode accepts only the default): wing_area_scale, wing_aspect_scale.

Rigid aero of every B2 gene is a NATIVE JSBSim increment (property-driven functions appended to the <root>_v2b2 aero
axes, properties written by the hook BEFORE IC/trim) -> trimmed, live q / alpha / beta / Mach, no (q - q_trim) term, no
coupler rigid feedback (r1 lesson). Structure sees load vectors only (flexbody_b2).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import flexwing as fw
import flexbody as fb
import planform_b1 as pb1

B2_TAG = "flexv2b2a"
B2_FMT = 1
B2_REV = 0
B2_FIDELITY = "full_a1_b2a"
B2_STAGE = "B2a"
AIRCRAFT = ("c172x", "T38", "737", "f16")

# Feature flags (hashed). Staged landing order: dihedral -> camber -> thickness. A gene whose flag is off is rejected by
# decode at non-default values (it would otherwise be a silent no-op).
FEATURES = {"dihedral": True, "camber": True, "thickness": True}


@dataclass(frozen=True)
class ShapeGeneB2:
    name: str
    feature: str
    ranges: Tuple[Tuple[str, float, float], ...]     # (aircraft, lo, hi)
    default: float
    unit: str
    meaning: str
    requires: Optional[str] = None                   # ER may evolve the gene only once this cost is wired (§15.10)

    def lo_hi(self, model: str) -> Tuple[float, float]:
        for a, lo, hi in self.ranges:
            if a == model:
                return lo, hi
        raise ValueError(f"no B2 range for aircraft {model!r} (have {AIRCRAFT})")


LOCK_ENERGY = "energy_cost"   # thickness: requires ER's energy term; size (B2b): locked until it (INTERFACE_v2 §15.10)
GATE_TOL = 1e-9               # every gate comparison is inclusive with this tolerance (box corners sit ON limits)


def _r(c, t, b, f):
    return (("c172x",) + c, ("T38",) + t, ("737",) + b, ("f16",) + f)


SHAPE_GENES_B2: Tuple[ShapeGeneB2, ...] = (
    # dihedral lo = 0 (B2a r0 review): anhedral earned a one-sided comfort credit (f16 monotone, c172x noisy) with no
    # lateral-stability cost in the scenarios -> disabled until a spiral / Dutch-roll gate exists (INTERFACE_v2 §15.10)
    ShapeGeneB2("wing_dihedral_delta_deg", "dihedral", _r((0.0, 3.0), (0.0, 3.0), (0.0, 2.0), (0.0, 3.0)), 0.0, "deg",
                "uniform geometric dihedral delta (deg, tip up +) on top of the JSBSim baseline"),
    # tc_root lo narrowed so that baseline structure never hits the flutter HARD fail at any tip ratio (scan §15.10)
    ShapeGeneB2("wing_tc_root_scale", "thickness", _r((0.875, 1.25), (0.925, 1.25), (0.90, 1.15), (0.85, 1.25)), 1.0, "-",
                "multiplier on the baseline t/c at the beam root", LOCK_ENERGY),
    ShapeGeneB2("wing_tc_tip_ratio", "thickness", _r((0.85, 1.15), (0.85, 1.15), (0.85, 1.15), (0.85, 1.15)), 1.0, "-",
                "tip t/c multiplier / root multiplier (linear in eta between)", LOCK_ENERGY),
    ShapeGeneB2("wing_camber_root_delta_pct", "camber", _r((-1.0, 2.0), (-0.5, 1.5), (-1.0, 1.0), (-0.5, 1.0)), 0.0, "%c",
                "additive delta on the thin-airfoil-equivalent max camber m_eq at the root (% chord)"),
    ShapeGeneB2("wing_camber_tip_delta_pct", "camber", _r((-1.0, 2.0), (-0.5, 1.5), (-1.0, 1.0), (-0.5, 1.0)), 0.0, "%c",
                "additive delta on m_eq at the tip (% chord); linear in eta"),
)
B2_NAMES = tuple(g.name for g in SHAPE_GENES_B2)
B2_INDEX = {g.name: i for i, g in enumerate(SHAPE_GENES_B2)}
N_B2 = len(SHAPE_GENES_B2)                                   # 5
N_SHAPE_B2 = pb1.N_SHAPE_GENES + N_B2                        # 11
DEFERRED_B2A = {"wing_area_scale": 1.0, "wing_aspect_scale": 1.0}
# B2b (NOT implemented): published ranges / locks so Genome can pad a B2a vector to the 13-gene B2b layout.
B2B_PUBLISHED = {"wing_area_scale": {"lo": 0.90, "hi": 1.15, "default": 1.0, "locked_until": LOCK_ENERGY},
                 "wing_aspect_scale": {"lo": 0.90, "hi": 1.15, "default": 1.0, "locked_until": LOCK_ENERGY}}
REQUIRES_ENERGY = tuple(g.name for g in SHAPE_GENES_B2 if g.requires == LOCK_ENERGY)   # thickness (B2a)
LOCKED_GENES = tuple(B2B_PUBLISHED)                                                       # size (B2b, not implemented)
# B1-deferred names that B2a does NOT implement (still only their B1 default)
_B1_DEFERRED_PASS = {k: v for k, v in pb1.DEFERRED_B1.items() if k not in B2_NAMES and k not in DEFERRED_B2A}
GENE_ENCODING = pb1.GENE_ENCODING + "; B2 ranges per aircraft (decode needs the model)"

# ---------------------------------------------------------------------------------------------------------------------
# Baseline sections + handbook constants (notional, hashed). m_eq = parabolic camber with the same thin-airfoil a_L0.
# ---------------------------------------------------------------------------------------------------------------------
SECTIONS = {
    "c172x": dict(section="NACA 2412", tc_root=0.12, tc_tip=0.12, m_root=0.018, m_tip=0.018, x_t=0.30, korn=None,
                  cf=0.0040, kp=0.010, m_ref=0.15, dihedral0_deg=1.73, alpha_stall_deg=15.0, clmax0=1.5,
                  w_gate_lb=2400.0, cm_de_full=0.56, high_wing=True, tip_bank_deg=None, tip_clear_ft=None),
    "T38": dict(section="NACA 65A004.8", tc_root=0.048, tc_tip=0.048, m_root=0.0, m_tip=0.0, x_t=0.40, korn=0.87,
                cf=0.0030, kp=0.008, m_ref=0.6, dihedral0_deg=0.0, alpha_stall_deg=18.0, clmax0=0.95,
                w_gate_lb=12000.0, cm_de_full=0.195, high_wing=False, tip_bank_deg=8.0, tip_clear_ft=1.0),
    "737": dict(section="BAC 449/450/451 (supercritical-ish, notional equivalents)", tc_root=0.15, tc_tip=0.105,
                m_root=0.025, m_tip=0.020, x_t=0.37, korn=0.95, cf=0.0024, kp=0.010, m_ref=0.6, dihedral0_deg=6.0,
                alpha_stall_deg=13.0, clmax0=1.2, w_gate_lb=115000.0, cm_de_full=0.42, high_wing=False,
                tip_bank_deg=6.0, tip_clear_ft=1.5),
    "f16": dict(section="NACA 64A204", tc_root=0.040, tc_tip=0.040, m_root=0.016, m_tip=0.016, x_t=0.40, korn=0.87,
                cf=0.0028, kp=0.008, m_ref=0.6, dihedral0_deg=0.0, alpha_stall_deg=30.0, clmax0=1.5,
                w_gate_lb=26000.0, cm_de_full=0.26, high_wing=False, tip_bank_deg=8.0, tip_clear_ft=1.0),
}
SWET_EXP_FACTOR = 2.0                      # S_wet = 2 (1 + 0.2 t/c) S_exposed
K_TRIM_AUTH = 0.30                         # |dCm_total| <= 0.30 x Cm per full elevator travel
GATE = {"tc": {"c172x": (0.08, 0.18), "T38": (0.033, 0.070), "737": (0.07, 0.18), "f16": (0.028, 0.060)},
        "m_eq_pct": {"c172x": (-0.5, 4.5), "T38": (-0.5, 2.0), "737": (0.5, 4.0), "f16": (0.0, 3.0)},
        "max_camber_gradient_pct": 3.0, "max_abs_effective_twist_deg": 8.0, "stall_factor": 1.2}


def schema(model: str) -> List[Dict]:
    out = []
    for g in SHAPE_GENES_B2:
        lo, hi = g.lo_hi(model)
        out.append({"name": g.name, "lo": lo, "hi": hi, "default": g.default, "scale": "linear", "unit": g.unit,
                    "feature": g.feature, "enabled": bool(FEATURES[g.feature]), "meaning": g.meaning,
                    "requires": g.requires, "locked_until": None, "identity_u": (g.default - lo) / (hi - lo)})
    return out


def shape_schema_b2(model: str) -> List[Dict]:
    """B1 rows (aircraft-independent) + B2a rows for `model`, in vector order."""
    b1 = [{"name": g.name, "lo": g.lo, "hi": g.hi, "default": g.default, "scale": g.scale, "unit": "", "feature": "b1",
           "enabled": True, "meaning": g.meaning, "requires": None, "locked_until": None,
           "identity_u": float(pb1.encode_shape_b1({g.name: g.default})[pb1.SHAPE_NAMES.index(g.name)])}
          for g in pb1.SHAPE_GENES_B1]
    return b1 + schema(model)


def shape_defaults_b2() -> Dict[str, float]:
    d = pb1.shape_defaults()
    d.update({g.name: g.default for g in SHAPE_GENES_B2})
    return d


def _check_model(model: str):
    if model not in SECTIONS:
        raise ValueError(f"B2 has no section data for aircraft {model!r} (have {AIRCRAFT})")


def decode_shape_b2(genes: Union[None, Dict[str, float], Sequence[float], np.ndarray], model: str,
                    require_energy_cost: bool = False) -> Dict[str, float]:
    """Merged B1 + B2a physical-value dict (missing -> default) or a [0,1]^11 vector (B1 order then B2 order, linear).
    Raises ValueError (never clips): unknown keys, out of range, NaN/inf, deferred / feature-off genes at non-default;
    with require_energy_cost=True (a caller WITHOUT the energy term) also any `requires='energy_cost'` gene
    (thickness) at a non-default value."""
    _check_model(model)
    if require_energy_cost:
        d = decode_shape_b2(genes, model)
        bad = [k for k in REQUIRES_ENERGY if d[k] != SHAPE_GENES_B2[B2_INDEX[k]].default]
        if bad:
            raise ValueError(f"genes {bad} require the Evolution energy cost term (requires={LOCK_ENERGY!r})")
        return d
    if genes is None:
        return shape_defaults_b2()
    if isinstance(genes, dict):
        b1, b2 = {}, {}
        for k, v in genes.items():
            try:
                x = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"shape gene {k}={v!r} is not a number") from None
            if not math.isfinite(x):
                raise ValueError(f"shape gene {k}={v} is not finite")
            if k in B2_INDEX:
                g = SHAPE_GENES_B2[B2_INDEX[k]]
                lo, hi = g.lo_hi(model)
                if x < lo or x > hi:
                    raise ValueError(f"shape gene {k}={v} outside its {model} range [{lo}, {hi}]")
                if not FEATURES[g.feature] and x != g.default:
                    raise ValueError(f"shape gene {k!r}: feature {g.feature!r} is not enabled in this B2 build")
                b2[k] = x
            elif k in DEFERRED_B2A:
                if x != DEFERRED_B2A[k]:
                    raise ValueError(f"shape gene {k!r} is deferred to B2b (INTERFACE_v2 §15); only {DEFERRED_B2A[k]} allowed")
            else:
                b1[k] = x                          # B1 decode validates (incl. its own deferred keys / unknown keys)
        out = pb1.decode_shape_b1(b1)
        out.update({g.name: g.default for g in SHAPE_GENES_B2})
        out.update(b2)
        return out
    vec = np.asarray(genes, dtype=float).ravel()
    if vec.size != N_SHAPE_B2:
        raise ValueError(f"B2a shape vector has length {vec.size}, expected {N_SHAPE_B2} (6 B1 + 5 B2a)")
    if not np.all(np.isfinite(vec)) or np.any(vec < 0.0) or np.any(vec > 1.0):
        raise ValueError("B2a shape vector entries must be finite and in [0, 1]")
    out = pb1.decode_shape_b1(vec[:pb1.N_SHAPE_GENES])
    for i, g in enumerate(SHAPE_GENES_B2):
        lo, hi = g.lo_hi(model)
        x = lo + float(vec[pb1.N_SHAPE_GENES + i]) * (hi - lo)
        if not FEATURES[g.feature] and x != g.default:
            raise ValueError(f"shape gene {g.name!r}: feature {g.feature!r} is not enabled in this B2 build")
        out[g.name] = x
    return out


def encode_shape_b2(model, genes=None) -> np.ndarray:
    """[0,1]^11 vector of a (partial) physical-value dict. Canonical call encode_shape_b2(model, genes); the legacy
    order encode_shape_b2(genes, model) is accepted. encode_shape_b2(model) = identity_u(model)."""
    if not isinstance(model, str):
        model, genes = genes, model
    d = decode_shape_b2(genes, model)
    v = list(pb1.encode_shape_b1({k: d[k] for k in pb1.SHAPE_NAMES}))
    for g in SHAPE_GENES_B2:
        lo, hi = g.lo_hi(model)
        v.append((d[g.name] - lo) / (hi - lo))
    return np.array(v)


def identity_u(model: str, stage: str = "B2a") -> np.ndarray:
    """Encoded identity (all genes at default): length 11 (B2a) or 13 (B2b layout: + area, aspect at u = 0.4)."""
    v = encode_shape_b2(model, None)
    if stage == "B2a":
        return v
    if stage == "B2b":
        return np.concatenate([v, [(r["default"] - r["lo"]) / (r["hi"] - r["lo"]) for r in B2B_PUBLISHED.values()]])
    raise ValueError("stage must be 'B2a' or 'B2b'")


def pad_b2a_to_b2b(u, model: str) -> np.ndarray:
    """Append the B2b identity (area, aspect) to an 11-vector -> 13-vector (B2b itself is not implemented)."""
    u = np.asarray(u, dtype=float).ravel()
    if u.size != N_SHAPE_B2:
        raise ValueError(f"expected an {N_SHAPE_B2}-vector")
    return np.concatenate([u, identity_u(model, "B2b")[N_SHAPE_B2:]])


def envelope_hash(envelope: Optional[Sequence[Tuple[float, float]]]) -> str:
    """8-hex hash of the gate envelope [(speed_kts, h_ft)] (sorted, repr floats). None -> 'none'."""
    if not envelope:
        return "none"
    import hashlib
    t = "|".join(f"{float(v)!r},{float(h)!r}" for v, h in sorted(envelope))
    return hashlib.sha256(t.encode()).hexdigest()[:8]


def b1_part(d: Dict[str, float]) -> Dict[str, float]:
    return {k: d[k] for k in pb1.SHAPE_NAMES}


def b2_part(d: Dict[str, float]) -> Dict[str, float]:
    return {k: d[k] for k in B2_NAMES}


def is_baseline_b2(genes, model: str) -> bool:
    """True if every B2 gene equals its default exactly (the B1 part may be shaped)."""
    d = decode_shape_b2(genes, model)
    return all(d[g.name] == g.default for g in SHAPE_GENES_B2)


def is_baseline_shape_b2(genes, model: str) -> bool:
    d = decode_shape_b2(genes, model)
    return is_baseline_b2(d, model) and pb1.is_baseline_shape(b1_part(d))


def shape_cache_key_b2(genes, model: str, envelope: Optional[Sequence[Tuple[float, float]]] = None) -> str:
    """Includes the gate envelope hash (the stall gate depends on the scenario speeds)."""
    d = decode_shape_b2(genes, model)
    return (f"b2a|{model}|" + pb1.shape_cache_key(b1_part(d)) + "|" + "|".join(f"{k}={d[k]!r}" for k in B2_NAMES)
            + f"|env={envelope_hash(envelope)}")


def modal_cache_key_b2(genes, model: str) -> str:
    """Modal basis depends on B1 + thickness only (camber / dihedral are loads)."""
    d = decode_shape_b2(genes, model)
    return f"b2a-modal|{model}|" + pb1.shape_cache_key(b1_part(d)) + \
        f"|tc_r={d['wing_tc_root_scale']!r}|tc_q={d['wing_tc_tip_ratio']!r}"


# ---------------------------------------------------------------------------------------------------------------------
# Section distributions on the strips
# ---------------------------------------------------------------------------------------------------------------------
def tau_dist(d: Dict[str, float], xi) -> np.ndarray:
    xi = np.asarray(xi, float)
    return d["wing_tc_root_scale"] * (1.0 + (d["wing_tc_tip_ratio"] - 1.0) * xi)


def dm_dist(d: Dict[str, float], xi) -> np.ndarray:
    """Camber delta (fraction of chord) at beam stations xi."""
    xi = np.asarray(xi, float)
    r, t = d["wing_camber_root_delta_pct"] / 100.0, d["wing_camber_tip_delta_pct"] / 100.0
    return r + (t - r) * xi


def tc_base(model: str, xi) -> np.ndarray:
    s = SECTIONS[model]
    return s["tc_root"] + (s["tc_tip"] - s["tc_root"]) * np.asarray(xi, float)


def m_base(model: str, xi) -> np.ndarray:
    s = SECTIONS[model]
    return s["m_root"] + (s["m_tip"] - s["m_root"]) * np.asarray(xi, float)


def section_summary(d: Dict[str, float], model: str, xi) -> Dict:
    t = tc_base(model, xi) * tau_dist(d, xi)
    m = m_base(model, xi) + dm_dist(d, xi)
    return {"section_baseline": SECTIONS[model]["section"], "tc_root": float(t[0]), "tc_tip": float(t[-1]),
            "m_eq_root_pct": float(100 * m[0]), "m_eq_tip_pct": float(100 * m[-1]),
            "dihedral_delta_deg": float(d["wing_dihedral_delta_deg"]),
            "dihedral_baseline_deg": float(SECTIONS[model]["dihedral0_deg"])}


# ---------------------------------------------------------------------------------------------------------------------
# Lifting line (Multhopp collocation, sine series) for the antisymmetric dihedral incidence
# ---------------------------------------------------------------------------------------------------------------------
def lifting_line_clb_per_gamma(chord_fn, span_ft: float, area_ft2: float, y0_ft: float, a2d: float = 2 * math.pi,
                               n: int = 40) -> float:
    """Rolling-moment coefficient per unit (beta * Gamma) [rad x rad] for incidence +1 on the right wing panel (|y| > y0),
    -1 on the left, 0 over the carry-through. Prandtl lifting line on chord_fn(|y|) (ft). Negative = stable."""
    s = span_ft / 2
    th = np.arange(1, n + 1) * math.pi / (n + 1)
    y = s * np.cos(th)
    c = chord_fn(np.abs(y))
    rhs = np.where(np.abs(y) > y0_ft, np.sign(y), 0.0)
    k = np.arange(1, n + 1)
    A = np.sin(np.outer(th, k)) * (4 * span_ft / (a2d * c))[:, None] + np.sin(np.outer(th, k)) * (k[None, :] / np.sin(th)[:, None])
    an = np.linalg.solve(A, rhs)
    AR = span_ft ** 2 / area_ft2
    return float(-math.pi * AR / 4 * an[1])


def _shaped_chord_fn(pw, d):
    s = pw.span_ft / 2
    y0 = pw.root_frac * s
    pf = pb1.planform_from_wingparams(pw, b1_part(d), 64)

    def f(v):
        xi = np.clip((np.asarray(v, float) - y0) / (s - y0), 0.0, 1.0)
        return pb1.baseline_chord_ft(pw.span_ft, pw.area_ft2, pw.taper, v) * pb1.chord_multipliers_raw(b1_part(d), xi) * pf.area_norm
    return f, pf


# ---------------------------------------------------------------------------------------------------------------------
# Native increments (values of the flexbody/b2/* properties). All exactly 0 (or base == new) at the B2 defaults.
# ---------------------------------------------------------------------------------------------------------------------
NATIVE_PROPS = ("dCL0", "dCm0", "dCD0", "cam_A", "cam_B", "korn_c_new", "korn_c_base", "korn_kcl", "dClb")


def _schrenk_weights(pw, pf, d, y):
    s = pw.span_ft / 2
    c_mean = pw.area_ft2 / 2 / s
    f, _ = _shaped_chord_fn(pw, d)
    return 0.5 * f(y) / c_mean + 0.5 * 4 / math.pi * np.sqrt(np.clip(1 - (y / s) ** 2, 0, None))


def native_increments(model: str, d: Dict[str, float], pw, geom: fb.Geometry) -> Dict:
    """Rigid aero increments (coefficients on S0 / b0 / cbar0) for the decoded B2 genes + bookkeeping. See §15.2."""
    sec = SECTIONS[model]
    s = pw.span_ft / 2
    y0 = pw.root_frac * s
    pr = fb.V2_PROFILES[model]
    f_c, pf = _shaped_chord_fn(pw, d)
    lam = math.radians(pf.sweep_qc_deg)
    AR = pw.span_ft ** 2 / pw.area_ft2
    a_w = fw.datcom_cla(AR, lam)
    a_h = fw.datcom_cla(pr["ht"]["AR"], math.radians(pr["ht"]["sweep"]))
    deps = pr["ht"]["downwash"]
    lam_b = pw.taper
    cbar = 2.0 / 3.0 * pw.area_ft2 / (s * (1 + lam_b)) * (1 + lam_b + lam_b * lam_b) / (1 + lam_b)
    V_H = geom.sh_ft2 * geom.lh_ft / (pw.area_ft2 * cbar)
    # spanwise quadrature over the semi-span (carry-through: root section values)
    yy = (np.arange(2000) + 0.5) / 2000 * s
    xi = np.clip((yy - y0) / (s - y0), 0.0, 1.0)
    dy = s / 2000
    w = _schrenk_weights(pw, pf, d, yy)
    w = w / np.sum(w)
    cc = f_c(yy)
    expo = yy > y0
    out = {k: 0.0 for k in NATIVE_PROPS}
    info = {"a_w": a_w, "a_h": a_h, "V_H": V_H, "cbar_ft": cbar, "AR": AR, "sweep_qc_deg": pf.sweep_qc_deg}
    # --- camber
    dm = dm_dist(d, xi)
    if FEATURES["camber"] and np.any(dm != 0.0):
        dCL0_w = a_w * 2.0 * float(np.sum(w * dm))
        dcm = -math.pi * dm
        fac = AR * math.cos(lam) ** 2 / (AR + 2 * math.cos(lam))
        dCm0_w = fac * 2.0 * float(np.sum(cc ** 2 * dcm) * dy) / (pw.area_ft2 * cbar)
        deps0 = deps * dCL0_w / a_w
        dCm_dw = V_H * a_h * deps0
        dCL_dw = -(geom.sh_ft2 / pw.area_ft2) * a_h * deps0
        out["dCL0"] = dCL0_w + dCL_dw
        out["dCm0"] = dCm0_w + dCm_dw
        m_abs0, m_abs = m_base(model, xi), m_base(model, xi) + dm
        cli0 = 4 * math.pi * float(np.sum(w * m_abs0))
        cli1 = 4 * math.pi * float(np.sum(w * m_abs))
        kp = sec["kp"]
        out["cam_A"] = kp * (cli1 ** 2 - cli0 ** 2)          # x qbar-area
        out["cam_B"] = -2.0 * kp * (cli1 - cli0)              # x |CL| = sqrt(aero/cl-squared) (current frame)
        info.update(dCL0_wing=dCL0_w, dCL_downwash=dCL_dw, dCm0_wing=dCm0_w, dCm_downwash=dCm_dw, cli_base=cli0,
                    cli_new=cli1, alpha_trim_shift_deg_est=-math.degrees(out["dCL0"] / a_w))
    # --- thickness
    tau = tau_dist(d, xi)
    if FEATURES["thickness"] and np.any(tau != 1.0):
        t0 = tc_base(model, xi)
        t1 = t0 * tau
        aw_ = (cc * expo)
        tm0 = float(np.sum(aw_ * t0) / np.sum(aw_))
        tm1 = float(np.sum(aw_ * t1) / np.sum(aw_))
        S_exp = 2 * float(np.sum(aw_) * dy)

        def ff(t):
            return (1 + 0.6 / sec["x_t"] * t + 100 * t ** 4) * 1.34 * sec["m_ref"] ** 0.18 * math.cos(lam) ** 0.28
        swet0 = SWET_EXP_FACTOR * (1 + 0.2 * tm0) * S_exp / pw.area_ft2
        swet1 = SWET_EXP_FACTOR * (1 + 0.2 * tm1) * S_exp / pw.area_ft2
        out["dCD0"] = sec["cf"] * (swet1 * ff(tm1) - swet0 * ff(tm0))
        info.update(tc_mean_base=tm0, tc_mean_new=tm1)
        if sec["korn"] is not None:
            cl_ = math.cos(lam)
            out["korn_c_base"] = sec["korn"] / cl_ - tm0 / cl_ ** 2 - 0.108
            out["korn_c_new"] = sec["korn"] / cl_ - tm1 / cl_ ** 2 - 0.108
            out["korn_kcl"] = 1.0 / (10 * cl_ ** 3)
    if out["korn_c_new"] == 0.0 and out["korn_c_base"] == 0.0:
        out["korn_c_new"] = out["korn_c_base"] = 9.0           # never critical (exact zero difference)
    # --- dihedral (lifting line, normal-section incidence ~ cos(sweep))
    dG = math.radians(d["wing_dihedral_delta_deg"])
    if FEATURES["dihedral"] and dG != 0.0:
        k_ll = lifting_line_clb_per_gamma(f_c, pw.span_ft, pw.area_ft2, y0)
        out["dClb"] = k_ll * math.cos(lam) * dG
        info.update(clb_ll_per_rad2=k_ll * math.cos(lam),
                    clb_per_deg2=k_ll * math.cos(lam) * math.radians(1) ** 2)
    return {"props": out, "info": info}


def mach_table(model: str, pw) -> List[Tuple[float, float]]:
    """a_w(M) / a_w(0) at the BASELINE planform (fixed per aircraft; hashed)."""
    AR = pw.span_ft ** 2 / pw.area_ft2
    lam = math.radians(pw.sweep_deg)
    a0 = fw.datcom_cla(AR, lam)
    return [(m, fw.datcom_cla(AR, lam, mach=m) / a0) for m in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0)]


def pg_table() -> List[Tuple[float, float]]:
    return [(m, 1.0 / math.sqrt(1 - min(m, 0.9) ** 2)) for m in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0)]


def stall_fade_table(model: str) -> List[Tuple[float, float]]:
    a = SECTIONS[model]["alpha_stall_deg"]
    return [(-1.6, 1.0), (math.radians(a - 4.0), 1.0), (math.radians(a), 0.0), (1.6, 0.0)]   # no flown CLmax change


# ---------------------------------------------------------------------------------------------------------------------
# Geometry gate (B1 gate + B2 checks). Cheap, before build; reject = hard fail.
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class GateB2:
    ok: bool
    reason: Optional[str]
    details: Dict[str, float]

    def as_dict(self):
        return {"ok": bool(self.ok), "reason": self.reason, "details": dict(self.details)}


@lru_cache(maxsize=64)
def _gear_contacts(model: str, root_v2: str):
    import os
    import re
    xml = os.path.join(root_v2, "aircraft", model, model + ".xml")
    t = open(xml, encoding="utf-8", errors="replace").read()
    cs = []
    for m in re.finditer(r'<contact\b([^>]*)>(.*?)</contact>', t, re.S):
        if "BOGEY" not in m.group(1):
            continue
        loc = re.search(r'<location[^>]*>\s*<x>\s*([-\d.eE+]+)\s*</x>\s*<y>\s*([-\d.eE+]+)\s*</y>\s*<z>\s*([-\d.eE+]+)', m.group(2))
        if loc:
            cs.append(tuple(float(v) for v in loc.groups()))
    a = re.search(r'<location name="AERORP" unit="IN">\s*<x>\s*([-\d.eE+]+)\s*</x>\s*<y>\s*([-\d.eE+]+)\s*</y>\s*<z>\s*([-\d.eE+]+)', t)
    return cs, (tuple(float(v) for v in a.groups()) if a else (0.0, 0.0, 0.0))


def tip_clearance_ft(model: str, pw, d, root_v2: str) -> Optional[float]:
    """Lower-tip height above ground at the static attitude banked by tip_bank_deg about the main-gear contact
    (wing plane z = AERORP z; ft). None for high-wing aircraft."""
    sec = SECTIONS[model]
    if sec["high_wing"] or sec["tip_bank_deg"] is None:
        return None
    cs, (_, _, z_rp) = _gear_contacts(model, root_v2)
    mains = [c for c in cs if abs(c[1]) > 1.0] or cs
    z_g = min(c[2] for c in mains) / 12.0
    y_m = max(abs(c[1]) for c in mains) / 12.0
    s = pw.span_ft / 2
    y0 = pw.root_frac * s
    G = math.radians(sec["dihedral0_deg"] + d["wing_dihedral_delta_deg"])
    dz = z_rp / 12.0 - z_g + (s - y0) * math.tan(G)
    ph = math.radians(sec["tip_bank_deg"])
    return dz * math.cos(ph) - (s - y_m) * math.sin(ph)


def stall_ok(model: str, d, pw, envelope: Optional[Sequence[Tuple[float, float]]]) -> Tuple[bool, float]:
    """1.2 V_s(W_gate, CLmax') <= every envelope speed. envelope = [(speed_kts (KCAS), h_ft)]; None -> skipped."""
    sec = SECTIONS[model]
    xi = np.linspace(0, 1, 41)
    t = float(np.mean(tc_base(model, xi) * tau_dist(d, xi)))
    t0 = float(np.mean(tc_base(model, xi)))
    m1 = float(np.mean(dm_dist(d, xi)))

    def dcl2(tc):
        return 4.0 * (min(tc, 0.12) - 0.12) - 2.0 * max(tc - 0.15, 0.0)
    lam = math.radians(pw.sweep_deg)
    dclmax = 0.9 * math.cos(lam) * ((dcl2(t) - dcl2(t0)) + 5.0 * m1)
    clmax = sec["clmax0"] + dclmax
    if not envelope:
        return True, clmax
    worst = math.inf
    for v_kts, _h in envelope:
        q = 0.5 * fb.RHO0 * (v_kts * fb.KT2FPS) ** 2           # KCAS ~ EAS at these speeds
        cl_req = sec["w_gate_lb"] / (q * pw.area_ft2)
        worst = min(worst, clmax / (GATE["stall_factor"] ** 2 * cl_req))
    return worst >= 1.0 - GATE_TOL, clmax


def geometry_gate_b2(pw, genes, model: str, n_el: int = 64, root_v2: Optional[str] = None,
                     envelope: Optional[Sequence[Tuple[float, float]]] = None, geom: Optional[fb.Geometry] = None) -> GateB2:
    try:
        d = decode_shape_b2(genes, model)
    except ValueError as e:
        return GateB2(False, f"decode: {e}", {})
    g1 = pb1.geometry_gate_strips(pb1.planform_from_wingparams(pw, b1_part(d), n_el))
    det = dict(g1.details)
    if not g1.ok:
        return GateB2(False, g1.reason, det)
    xi = np.linspace(0.0, 1.0, 65)
    tc = tc_base(model, xi) * tau_dist(d, xi)
    m = 100 * (m_base(model, xi) + dm_dist(d, xi))
    dm = dm_dist(d, xi)
    th = np.degrees(pb1.twist_rad(b1_part(d), xi) + 2.0 * (dm - dm[0]))
    det.update(tc_min=float(tc.min()), tc_max=float(tc.max()), m_eq_min_pct=float(m.min()), m_eq_max_pct=float(m.max()),
               camber_gradient_pct=float(abs(d["wing_camber_tip_delta_pct"] - d["wing_camber_root_delta_pct"])),
               effective_twist_max_deg=float(np.max(np.abs(th))))
    lo, hi = GATE["tc"][model]
    if tc.min() < lo - GATE_TOL or tc.max() > hi + GATE_TOL:
        return GateB2(False, "tc_out_of_band", det)
    lo, hi = GATE["m_eq_pct"][model]
    if m.min() < lo - GATE_TOL or m.max() > hi + GATE_TOL:
        return GateB2(False, "camber_out_of_band", det)
    if det["camber_gradient_pct"] > GATE["max_camber_gradient_pct"] + GATE_TOL:
        return GateB2(False, "camber_gradient", det)
    if det["effective_twist_max_deg"] > GATE["max_abs_effective_twist_deg"] + GATE_TOL:
        return GateB2(False, "effective_twist", det)
    if root_v2 is not None:
        h = tip_clearance_ft(model, pw, d, root_v2)
        if h is not None:
            det["tip_clearance_ft"] = float(h)
            if h < SECTIONS[model]["tip_clear_ft"] - GATE_TOL:
                return GateB2(False, "tip_clearance", det)
    ok, clmax = stall_ok(model, d, pw, envelope)
    det["clmax_est"] = float(clmax)
    if not ok:
        return GateB2(False, "stall_margin", det)
    if geom is not None:
        ni = native_increments(model, d, pw, geom)["props"]
        det["dCm_total"] = float(ni["dCm0"])
        if abs(ni["dCm0"]) > K_TRIM_AUTH * SECTIONS[model]["cm_de_full"] + GATE_TOL:
            return GateB2(False, "trim_authority", det)
    return GateB2(True, None, det)


def shape_params_for_hash() -> Dict:
    return {"b2_fmt": B2_FMT, "b2_rev": B2_REV, "tag": B2_TAG, "stage": B2_STAGE, "features": dict(FEATURES),
            "genes": [asdict(g) for g in SHAPE_GENES_B2], "encoding": GENE_ENCODING, "deferred": dict(DEFERRED_B2A),
            "sections": SECTIONS, "gate": GATE, "gate_tol": GATE_TOL, "b2b_published": B2B_PUBLISHED, "k_trim_auth": K_TRIM_AUTH, "swet_exp_factor": SWET_EXP_FACTOR,
            "native_props": list(NATIVE_PROPS), "b1": pb1.shape_params_for_hash(),
            "rules": {"rigid_aero": "native JSBSim increments, properties set before IC/trim (no coupler rigid feedback)",
                      "camber": "thin airfoil (a_L0 = -2 m, cm = -pi m); dCL0 Schrenk mean x stall fade (1 -> 0 over a_s-4..a_s: CLmax change gate-only); DATCOM Cm0 factor; "
                                "downwash at tail; profile polar Kp[(CL-cli')^2-(CL-cli0)^2] with |CL| = sqrt(aero/cl-squared); structure: aero twist 2 dm "
                                "in the B1 basic-load builder + strip cm couple (torsion), torque sizing one-sided",
                      "thickness": "EI,GJ x tau^2; EIv x 1; strength x tau (bend, tip, torque); mass unchanged at fixed gauge; "
                                   "dCD0 Raymer form factor; Korn/Lock wave-drag difference; no CLa change; CLmax gate only",
                      "dihedral": "lifting-line dClb x cos(sweep) x Mach table; structure: antisymmetric strip load "
                                  "lw sign dGamma per q kappa beta; delta only (no Gamma0 beta loads); node z = -(y-y0) tan dGamma"}}
