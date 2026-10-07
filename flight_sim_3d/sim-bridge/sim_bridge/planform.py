"""Optional P3-B1 planform header block (FD INTERFACE_v2 section 14) on ga-flightsim-traj/2 trajectories.

Layout agreed with ER (no traj-schema bump; the field is optional and ignored when absent):

    "planform": {
      "schema": "fd-planform/1", "source": "P3-B1",
      "genes": {"wing_chord_taper_1": 0.95, ...},            # FD B1 shape genes, physical values
      "symmetric": true, "wing": {...}                       # ER's form (one semi-wing = right, left = mirror y)
      # or "wingR": {...}, "wingL": {...} (also accepted: symmetric + wingR only)
      #   side = {"span_frac": [...], "y_m": [...], "chord_m": [...], "le_x_m": [...], "twist_rad": [...]}
      #   (ER also writes chord_baseline_m / te_x_m; ignored here)
      "sweep_qc_rad": 0.47,
      "synthetic": true                                      # test fixtures only
    }

SI units, body FRD (x forward, y right, z down), twist + = leading edge up. One entry per strip (FD: 64 strips per
semi-wing, 65 FE nodes), span_frac = FD's beam eta (0 at the wing root station, 1 at the tip). `le_x_m` as ER writes
it (evolution/fidelity.planform_header, phase3b1-smoke-s1, 2026-10-06 18:20 PT) is relative to the shaped wing's
quarter-chord point at the beam root (+ forward; FD exposes no absolute LE position); the viewer only uses LE
offsets between strips, so any constant reference works. The field name `planform_b1` is accepted as well (also
inside `structure`). Stdlib only.
"""
import math

PLANFORM_SCHEMA = "fd-planform/1"
FIELD_NAMES = ("planform", "planform_b1")
SIDE_KEYS = ("span_frac", "y_m", "chord_m", "le_x_m", "twist_rad")
FT = 0.3048


def find_planform(doc):
    """(field_name, block) of the first planform block found in a trajectory doc, else (None, None)."""
    if not isinstance(doc, dict):
        return None, None
    st = doc.get("structure") if isinstance(doc.get("structure"), dict) else {}
    for where, o in (("", doc), ("structure.", st)):
        for k in FIELD_NAMES:
            if isinstance(o.get(k), dict):
                return where + k, o[k]
    return None, None


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_side(w, name="wingR"):
    errs = []
    if not isinstance(w, dict):
        return [f"{name}: not an object"]
    c = w.get("chord_m")
    if not (isinstance(c, list) and len(c) >= 2 and all(_finite(x) and x > 0 for x in c)):
        return [f"{name}.chord_m: need >= 2 positive finite values"]
    n = len(c)
    for k in SIDE_KEYS:
        v = w.get(k)
        if v is None:
            continue
        if not (isinstance(v, list) and len(v) == n and all(_finite(x) for x in v)):
            errs.append(f"{name}.{k}: need {n} finite values")
    for k in ("span_frac", "y_m"):
        v = w.get(k)
        if isinstance(v, list) and len(v) == n and all(_finite(x) for x in v):
            a = [abs(x) for x in v]
            if any(b <= a_ for a_, b in zip(a, a[1:])):
                errs.append(f"{name}.{k}: must increase outboard")
    return errs


def validate_planform(block):
    """List of problems (empty = usable by the viewer)."""
    if not isinstance(block, dict):
        return ["planform: not an object"]
    errs = []
    if block.get("schema") not in (None, PLANFORM_SCHEMA):
        errs.append(f"schema {block.get('schema')!r} != {PLANFORM_SCHEMA!r} (viewer still tries)")
    sides = [k for k in ("wingR", "wingL", "wing") if k in block]
    if not sides:
        errs.append("no wingR / wingL / wing")
    for k in sides:
        errs += validate_side(block[k], k)
    if len(sides) == 1 and not block.get("symmetric"):
        errs.append("one side only but symmetric is not true (viewer mirrors it anyway)")
    sw = block.get("sweep_qc_rad")
    if sw is not None and not _finite(sw):
        errs.append("sweep_qc_rad: not finite")
    return errs


def summary(block):
    """{sweep_qc_deg, taper (tip/root chord), twist_tip_deg, n_strips} of the right (or only) wing."""
    w = block.get("wingR") or block.get("wing") or block.get("wingL") or {}
    c = w.get("chord_m") or []
    tw = w.get("twist_rad") or []
    sw = block.get("sweep_qc_rad")
    return {"sweep_qc_deg": math.degrees(sw) if _finite(sw) else None,
            "taper": (c[-1] / c[0]) if len(c) >= 2 and c[0] > 0 else None,
            "twist_tip_deg": math.degrees(tw[-1]) if tw else None, "n_strips": len(c)}


def from_fd_strips(xi, y_ft, c_ft, le_x_ft_aft, twist_rad, sweep_qc_deg, x_qc_root_body_m=0.0, genes=None,
                   source="P3-B1", synthetic=False, extra=None, side_key="wingR"):
    """FD planform_b1.PlanformStrips arrays (feet; LE x measured AFT from the quarter-chord line through the beam
    root) -> the header block (SI, body FRD: x forward). `x_qc_root_body_m` = body x of that quarter-chord root point
    (0 = ER's convention: relative). Symmetric (FD B1 runs with L/R symmetry on); `side_key` "wing" = ER's form."""
    w = {"span_frac": [float(v) for v in xi], "y_m": [float(v) * FT for v in y_ft],
         "chord_m": [float(v) * FT for v in c_ft],
         "le_x_m": [float(x_qc_root_body_m) - float(v) * FT for v in le_x_ft_aft],
         "twist_rad": [float(v) for v in twist_rad]}
    out = {"schema": PLANFORM_SCHEMA, "source": source, "genes": dict(genes or {}), "symmetric": True,
           side_key: w, "sweep_qc_rad": math.radians(float(sweep_qc_deg))}
    if synthetic:
        out["synthetic"] = True
    if extra:
        out.update(extra)
    return out
