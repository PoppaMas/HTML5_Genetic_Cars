"""planform_b1.py -- P3-B1 LOCKED wing-shape gene block: schema, decode, planform strips, geometry gate.

Owner: Flight Dynamics. Genome / Evolution mirror THIS module (INTERFACE_v2.md §14). Pure numpy + flexbody.pchip.

Shape is a SEPARATE gene block from the Phase-2 / A1 structure block (12 genes + optional 2 asymmetry, P2.5 ranges).
Decode order: shape -> planform strips + geometry gate -> wing rebuild (aero strips, geometry-derived EI/GJ/mass baseline)
-> structure genes applied on that baseline -> margins + sizing -> flight. Host fidelity: 'full_a1' (64 strips).

B1 locked list (6 genes, symmetric L = R):
  wing_chord_taper_1..3   chord-multiplier ratios between 4 control stations at beam eta 0, 1/3, 2/3, 1 (root + 3
                          segment ratios, cumulative product, monotone log-PCHIP onto the 64 strips). The distribution
                          is then rescaled so the semi-wing PLANFORM AREA equals the baseline (S = JSBSim S_ref, span
                          fixed): chord genes redistribute area spanwise (taper / planform shape), they never resize the
                          wing. (Resizing would need rescaled JSBSim tables -> later gate.)
  wing_twist_mid_deg      geometric twist at beam eta 0.5, relative to the root (deg, nose-up +)
  wing_twist_tip_deg      geometric twist at the tip, relative to the root (deg; negative = washout)
                          Root twist is fixed at 0: a uniform incidence change is absorbed by the trim (fixed tables).
                          Twist(eta) = PCHIP through (0, 0), (0.5, mid), (1, tip) (no overshoot).
  wing_sweep_qc_delta_deg additive delta on the baseline QUARTER-CHORD sweep (not leading edge), whole semi-span.

Deferred (decode raises on non-baseline values): wing_dihedral_delta_deg (geometric dihedral not represented by the strip
model / node layout / external_reactions path -- only elastic beta*w' exists), wing_thickness_scale and
wing_camber_scale (section shape -> B2 / CST), wing_chord_root / span / area scale (wing size -> needs table rescale).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import flexbody as fb

B1_TAG = "flexv2b1"
B1_FMT = 1
B1_REV = 1        # r1 (P3-B1 follow-up): trim-consistent twist pitch, baseline-anchored flown wing-BM reference,
                  # shaped node layout. r0 strings (model_versions_post_p3b1.json) stay valid for r0 code only.
B1_FIDELITY = "full_a1_b1"

SHAPE_CP_ETA = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])     # chord control stations (beam eta)
TWIST_CP_ETA = np.array([0.0, 0.5, 1.0])                      # twist control stations (root fixed 0)
N_SHAPE_CP = SHAPE_CP_ETA.size


@dataclass(frozen=True)
class ShapeGeneB1:
    name: str
    lo: float
    hi: float
    scale: str
    default: float
    meaning: str


SHAPE_GENES_B1: Tuple[ShapeGeneB1, ...] = (
    ShapeGeneB1("wing_chord_taper_1", 0.85, 1.05, "linear", 1.0,
                "chord-multiplier ratio CP1/CP0 (eta 1/3 vs root); area-renormalised"),
    ShapeGeneB1("wing_chord_taper_2", 0.85, 1.05, "linear", 1.0,
                "chord-multiplier ratio CP2/CP1 (eta 2/3 vs 1/3); area-renormalised"),
    ShapeGeneB1("wing_chord_taper_3", 0.85, 1.05, "linear", 1.0,
                "chord-multiplier ratio CP3/CP2 (tip vs eta 2/3); area-renormalised"),
    ShapeGeneB1("wing_twist_mid_deg", -2.0, 1.0, "linear", 0.0,
                "geometric twist at beam eta 0.5 relative to root (deg, nose-up +)"),
    ShapeGeneB1("wing_twist_tip_deg", -4.0, 1.0, "linear", 0.0,
                "geometric twist at tip relative to root (deg, nose-up +; negative = washout)"),
    ShapeGeneB1("wing_sweep_qc_delta_deg", -5.0, 5.0, "linear", 0.0,
                "additive delta on baseline quarter-chord sweep (deg)"),
)
SHAPE_NAMES = tuple(g.name for g in SHAPE_GENES_B1)
# How a normalised gene u in [0, 1] maps to the physical value: LINEAR IN VALUE for all 6 genes (scale="linear"),
# value = lo + u (hi - lo), and encode is its exact inverse. There is no log mapping in FD's decode. (The chord-taper
# genes are RATIOS whose spanwise interpolation is log-PCHIP -- that is the planform rule, not the gene encoding.)
# Callers that mutate in log space internally must hand FD physical values ({name: value}) or a linear-normalised vector.
GENE_ENCODING = "linear_in_value: value = lo + u*(hi-lo), u in [0,1]; encode = (value-lo)/(hi-lo); no log scale"
SHAPE_INDEX = {g.name: i for i, g in enumerate(SHAPE_GENES_B1)}
N_SHAPE_GENES = len(SHAPE_GENES_B1)       # 6

# Deferred keys and the only value decode tolerates for each (anything else raises)
DEFERRED_B1 = {"wing_dihedral_delta_deg": 0.0, "wing_thickness_scale": 1.0, "wing_camber_scale": 1.0,
               "wing_chord_root": 1.0, "wing_span_scale": 1.0, "wing_area_scale": 1.0,
               "wing_twist_root_deg": 0.0}

# Geometry gate (reject before flight -> hard fail, never a fitness credit)
GATE_MIN_CHORD_FT = 0.05
GATE_MIN_TAPER = 0.12           # tip chord / root chord of the shaped planform
GATE_MAX_TAPER = 1.25
GATE_MAX_ABS_SWEEP_DEG = 45.0   # |baseline + delta| (strip theory / DATCOM sweep validity, f16 32 + 5 = 37)
GATE_MAX_LE_KINK_DEG = 25.0     # max change of local LE sweep between adjacent strips


def shape_schema() -> List[ShapeGeneB1]:
    return list(SHAPE_GENES_B1)


def shape_defaults() -> Dict[str, float]:
    return {g.name: g.default for g in SHAPE_GENES_B1}


def decode_shape_b1(genes: Union[None, Dict[str, float], Sequence[float], np.ndarray] = None) -> Dict[str, float]:
    """Named dict of physical values (missing keys -> default) or a vector in [0,1]^6 (table order, linear decode
    lo + u (hi - lo)). Raises ValueError (never clips): unknown keys, structure-gene names, deferred keys at non-baseline
    values, NaN/inf, values outside the range, wrong vector length / entries outside [0,1]."""
    out = shape_defaults()
    if genes is None:
        return out
    if isinstance(genes, dict):
        for k, v in genes.items():
            try:
                x = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"shape gene {k}={v!r} is not a number") from None
            if not math.isfinite(x):
                raise ValueError(f"shape gene {k}={v} is not finite")
            if k in DEFERRED_B1:
                if x != DEFERRED_B1[k]:
                    raise ValueError(f"shape gene {k!r} is deferred past B1 (INTERFACE_v2 §14); only {DEFERRED_B1[k]} allowed")
                continue
            if k not in SHAPE_INDEX:
                raise ValueError(f"unknown B1 shape gene {k!r} (have {list(SHAPE_NAMES)})")
            g = SHAPE_GENES_B1[SHAPE_INDEX[k]]
            if x < g.lo or x > g.hi:
                raise ValueError(f"shape gene {k}={v} outside its range [{g.lo}, {g.hi}]")
            out[k] = x
        return out
    vec = np.asarray(genes, dtype=float).ravel()
    if vec.size != N_SHAPE_GENES:
        raise ValueError(f"B1 shape vector has length {vec.size}, expected {N_SHAPE_GENES}")
    if not np.all(np.isfinite(vec)) or np.any(vec < 0.0) or np.any(vec > 1.0):
        raise ValueError("B1 shape vector entries must be finite and in [0, 1]")
    return {g.name: g.lo + float(vec[i]) * (g.hi - g.lo) for i, g in enumerate(SHAPE_GENES_B1)}


def encode_shape_b1(genes) -> np.ndarray:
    d = decode_shape_b1(genes)
    return np.array([(d[g.name] - g.lo) / (g.hi - g.lo) for g in SHAPE_GENES_B1])


def is_baseline_shape(genes) -> bool:
    """True if every decoded shape gene equals its default exactly (identity planform)."""
    d = decode_shape_b1(genes)
    return all(d[g.name] == g.default for g in SHAPE_GENES_B1)


def shape_cache_key(genes) -> str:
    """Deterministic key on the decoded shape (repr round-trips floats) -- cache the rebuilt model / margins on it."""
    d = decode_shape_b1(genes)
    return "b1|" + "|".join(f"{k}={d[k]!r}" for k in SHAPE_NAMES)


def chord_cp_multipliers(genes) -> np.ndarray:
    d = decode_shape_b1(genes)
    m = [1.0]
    for k in range(1, N_SHAPE_CP):
        m.append(m[-1] * d[f"wing_chord_taper_{k}"])
    return np.array(m)


def chord_multipliers_raw(genes, xi) -> np.ndarray:
    """Un-normalised chord multipliers at beam stations xi (log-PCHIP of the CP chain; positive, no overshoot)."""
    return np.exp(fb.pchip(SHAPE_CP_ETA, np.log(chord_cp_multipliers(genes)), np.asarray(xi, float)))


def twist_rad(genes, xi) -> np.ndarray:
    d = decode_shape_b1(genes)
    cp = np.array([0.0, d["wing_twist_mid_deg"], d["wing_twist_tip_deg"]])
    return np.radians(fb.pchip(TWIST_CP_ETA, cp, np.asarray(xi, float)))


def baseline_chord_ft(span_ft: float, area_ft2: float, taper: float, y) -> np.ndarray:
    """Baseline trapezoid chord (same formula as flexbody.Surface / flexwing.FlexWing)."""
    s = span_ft / 2.0
    c_root = area_ft2 / (s * (1.0 + taper))
    return c_root * (1.0 - (1.0 - taper) * np.asarray(y, float) / s)


@dataclass
class PlanformStrips:
    """Shaped semi-wing on the beam strips (right = left)."""
    xi: np.ndarray
    y_ft: np.ndarray
    dy_ft: float
    c0_ft: np.ndarray            # baseline chord
    c_ft: np.ndarray             # shaped chord (area-preserving)
    c_mult: np.ndarray           # c / c0 (after renormalisation)
    area_norm: float             # renormalisation factor applied to the raw multipliers
    twist_rad: np.ndarray
    sweep_qc_deg: float
    sweep_qc_deg_baseline: float
    le_x_ft: np.ndarray          # LE / TE x (aft +) relative to the quarter-chord line through the beam root
    te_x_ft: np.ndarray

    def summary(self) -> Dict[str, float]:
        return {"c_root_ft": float(self.c_ft[0]), "c_tip_ft": float(self.c_ft[-1]),
                "taper_tip_root": float(self.c_ft[-1] / self.c_ft[0]),
                "c_root_baseline_ft": float(self.c0_ft[0]), "c_tip_baseline_ft": float(self.c0_ft[-1]),
                "area_norm": float(self.area_norm), "sweep_qc_deg": float(self.sweep_qc_deg),
                "twist_mid_deg": float(np.degrees(np.interp(0.5, self.xi, self.twist_rad))),
                "twist_tip_deg": float(np.degrees(self.twist_rad[-1]))}


def build_planform_strips(span_ft: float, area_ft2: float, taper: float, sweep_deg: float, root_frac: float,
                          shape_genes, n_el: int) -> PlanformStrips:
    d = decode_shape_b1(shape_genes)
    s = span_ft / 2.0
    y0 = root_frac * s
    L = s - y0
    dy = L / int(n_el)
    y = y0 + (np.arange(int(n_el)) + 0.5) * dy
    xi = (y - y0) / L
    c0 = baseline_chord_ft(span_ft, area_ft2, taper, y)
    raw = chord_multipliers_raw(d, xi)
    norm = float(np.sum(c0) / np.sum(c0 * raw))     # exactly 1.0 for identity multipliers
    mult = raw * norm
    c = c0 * mult
    sw = float(sweep_deg + d["wing_sweep_qc_delta_deg"])
    xq = (y - y0) * math.tan(math.radians(sw))
    return PlanformStrips(xi=xi, y_ft=y, dy_ft=dy, c0_ft=c0, c_ft=c, c_mult=mult, area_norm=norm,
                          twist_rad=twist_rad(d, xi), sweep_qc_deg=sw, sweep_qc_deg_baseline=float(sweep_deg),
                          le_x_ft=xq - 0.25 * c, te_x_ft=xq + 0.75 * c)


def planform_from_wingparams(pw, shape_genes, n_el: int) -> PlanformStrips:
    return build_planform_strips(pw.span_ft, pw.area_ft2, pw.taper, pw.sweep_deg, pw.root_frac, shape_genes, n_el)


@dataclass
class GeometryGateResult:
    ok: bool
    reason: Optional[str]
    details: Dict[str, float]

    def as_dict(self) -> Dict:
        return {"ok": bool(self.ok), "reason": self.reason, "details": dict(self.details)}


def geometry_gate_strips(pf: PlanformStrips) -> GeometryGateResult:
    """Checks on built strips (also used directly by tests to inject illegal shapes)."""
    det = {"c_min_ft": float(np.min(pf.c_ft)), "taper_tip_root": float(pf.c_ft[-1] / pf.c_ft[0]) if pf.c_ft[0] > 0 else 0.0,
           "sweep_qc_deg": float(pf.sweep_qc_deg)}
    le_sw = np.degrees(np.arctan2(np.diff(pf.le_x_ft), np.diff(pf.y_ft)))
    det["le_kink_max_deg"] = float(np.max(np.abs(np.diff(le_sw)))) if le_sw.size > 1 else 0.0
    if not np.all(np.isfinite(pf.c_ft)) or det["c_min_ft"] < GATE_MIN_CHORD_FT:
        return GeometryGateResult(False, "negative_or_tiny_chord", det)
    if np.any(pf.te_x_ft <= pf.le_x_ft):
        return GeometryGateResult(False, "self_intersect", det)
    if not (GATE_MIN_TAPER <= det["taper_tip_root"] <= GATE_MAX_TAPER):
        return GeometryGateResult(False, "extreme_taper", det)
    if abs(det["sweep_qc_deg"]) > GATE_MAX_ABS_SWEEP_DEG:
        return GeometryGateResult(False, "extreme_sweep", det)
    if det["le_kink_max_deg"] > GATE_MAX_LE_KINK_DEG:
        return GeometryGateResult(False, "extreme_le_kink", det)
    return GeometryGateResult(True, None, det)


def geometry_gate(pw, shape_genes, n_el: int = 64) -> GeometryGateResult:
    """Cheap pre-flight gate (~0.1 ms). ok=False -> the individual hard-fails (status 'geometry_gate:<reason>',
    cost = fail_cost, not flown). Decode errors are returned as reason 'decode: ...' (evaluate() raises them instead)."""
    try:
        d = decode_shape_b1(shape_genes)
    except ValueError as e:
        return GeometryGateResult(False, f"decode: {e}", {})
    return geometry_gate_strips(planform_from_wingparams(pw, d, n_el))


def shape_params_for_hash() -> Dict:
    """Frozen B1 schema (hashed into the full_a1_b1 model_version)."""
    return {"b1_fmt": B1_FMT, "b1_rev": B1_REV, "tag": B1_TAG, "genes": [asdict(g) for g in SHAPE_GENES_B1],
            "encoding": GENE_ENCODING,
            "chord_cp_eta": SHAPE_CP_ETA.tolist(), "twist_cp_eta": TWIST_CP_ETA.tolist(),
            "chord_interp": "log-pchip, area-renormalised (S fixed)", "twist_interp": "pchip, root = 0",
            "sweep_convention": "quarter_chord", "symmetry": "L_equals_R", "deferred": dict(DEFERRED_B1),
            "gate": {"min_chord_ft": GATE_MIN_CHORD_FT, "min_taper": GATE_MIN_TAPER, "max_taper": GATE_MAX_TAPER,
                     "max_abs_sweep_deg": GATE_MAX_ABS_SWEEP_DEG, "max_le_kink_deg": GATE_MAX_LE_KINK_DEG},
            "rules": {"ac_hold": "wing re-positioned so the Schrenk-weighted quarter-chord x equals baseline",
                      "stiffness": "EI, GJ, EIv = baseline root constants x (c / c_root_baseline)^ei_taper_exp x structure multipliers",
                      "mass": "wing mass total fixed (area fixed), distribution ~ c^mass_taper_exp",
                      "twist": "basic (zero-net) strip load q kappa lw (twist - mean): loads the structure; its rigid pitch "
                               "moment is a Cm0 shift absorbed by the trim elevator -> no pitch feedback (r1)",
                      "flown_wing_bm_reference": "J_bm_rms denominator and J_bm_peak allowable use the 1-g root BM without "
                                                 "the twist basic load, scaled to the baseline planform (r1)",
                      "sizing": "wing allowables = baseline-planform design loads x structure gene x (c/c0)^3 at the station"}}
