"""Read-only bridge to Flight Dynamics' flex-wing model (flexwing.py, coupled_sim.py) and patched JSBSim copies.

Location: $FLIGHT_DYNAMICS_DIR (default ../flight-dynamics, i.e. flight_sim_3d/flight-dynamics). Nothing there is written:
FD modules are imported with bytecode writing disabled, and the prepared aircraft in jsbsim_root/ are used
as-is (missing or stale copies raise instead of being regenerated, because regenerating them is FD's job).
"""
from __future__ import annotations

import importlib
import json
import math
import os
import re
import sys
from typing import Dict, Optional

FD_DIR = os.path.abspath(os.environ.get("FLIGHT_DYNAMICS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flight-dynamics")))
FD_ROOT = os.path.join(FD_DIR, "jsbsim_root")

# evolved structure genes (Phase 1) -> consumed by flexwing.overrides_from_genome
FLEX_GENES = ("stiffness_scale", "torsion_bend_ratio", "struct_damping_ratio", "nonstructural_mass_scale")  # = FD STRUCT_SCHEMA
# Per-aircraft constants, not genes in Phase 1 (FD decision 2026-10-06; FD's overrides_from_genome raises on them).
# They are NEVER passed to FD: FD's flexwing.AIRCRAFT_PROFILES values are used. Our profiles carry copies (checked
# equal in tests) only for reporting and for the CG-aft-of-EA guard.
CHORD_CONSTANTS = ("elastic_axis_frac", "section_cg_frac")
CHORD_TO_WINGPARAM = {"elastic_axis_frac": "x_ea", "section_cg_frac": "x_cg"}
FLEX_INPUTS = FLEX_GENES + CHORD_CONSTANTS

# Flutter-margin ceiling used when FD's output does not state one (FD will publish theirs in INTERFACE.md).
# Margins are reported as min(margin, cap) plus a no-flutter flag; nothing relies on an infinite margin.
PROVISIONAL_FLUTTER_CAP = 3.0


def min_cg_aft() -> float:
    """FD's minimum CG-aft-of-EA gap (flexwing.MIN_CG_AFT_OF_EA, chord fraction); 0 if FD doesn't define one."""
    try:
        return float(getattr(flexwing(), "MIN_CG_AFT_OF_EA", 0.0))
    except Exception:  # noqa: BLE001 - FD folder missing: the strict "behind" rule still applies
        return 0.0


def fd_chord_defaults(model: str) -> Dict[str, float]:
    p = flexwing().AIRCRAFT_PROFILES.get(model, {})
    return {k: float(p[w]) for k, w in CHORD_TO_WINGPARAM.items() if w in p}


def check_chord_constants(ea: float, cg: float, where: str = "") -> None:
    """Phase-1 guard: the section CG must lie behind (aft of) the elastic axis, by FD's minimum gap if defined."""
    if not (0.0 < ea < 1.0 and 0.0 < cg < 1.0):
        raise ValueError(f"{where}elastic_axis_frac={ea} / section_cg_frac={cg} must be chord fractions in (0, 1)")
    gap = min_cg_aft()
    if not (cg > ea and cg >= ea + gap - 1e-9):
        raise ValueError(
            f"{where}section_cg_frac={cg} is at or ahead of elastic_axis_frac={ea} (or less than FD's minimum gap {gap} "
            "chord behind it). Phase 1 requires the section CG behind "
            "the elastic axis (Flight Dynamics decision 2026-10-06): with CG ahead of the EA the flutter model reports no "
            "flutter at all, which the GA exploits. Fix the config (profile flex_constants or task flex.*).")

_mods: Dict[str, object] = {}
_metrics: Dict[str, tuple] = {}


def _import(name: str):
    if name not in _mods:
        old = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            if FD_DIR not in sys.path:
                sys.path.append(FD_DIR)  # appended: never shadows genome/ or flight_sim modules
            _mods[name] = importlib.import_module(name)
        finally:
            sys.dont_write_bytecode = old
    return _mods[name]


def flexwing():
    return _import("flexwing")


def coupled_sim():
    return _import("coupled_sim")


# JSBSim network I/O: <input port=...> (telnet / QTJSBSIM), <output type="SOCKET|FLIGHTGEAR|..." port=...>.
# FCS <input>/<output> property elements carry no port/protocol attribute; file outputs (CSV, TABULAR) use name= only.
_NET_IO_RE = re.compile(r"<(input|output)\b[^>]*\b(port|protocol)\s*=[^>]*>", re.S | re.I)
_NET_TYPE_RE = re.compile(r"<output\b[^>]*\btype\s*=\s*[\"'](socket|flightgear|qtjsbsim)[\"'][^>]*>", re.S | re.I)
_net_checked: Dict[str, str] = {}


class NetworkIOError(RuntimeError):
    """An aircraft file declares JSBSim network input/output (would bind ports / accept remote control writes)."""


def network_io(xml_text: str) -> list:
    """Network <input>/<output> declarations in a JSBSim XML text (comments ignored)."""
    txt = re.sub(r"<!--.*?-->", "", xml_text, flags=re.S)
    hits = [m.group(0)[:80] for m in _NET_IO_RE.finditer(txt)] + [m.group(0)[:80] for m in _NET_TYPE_RE.finditer(txt)]
    return sorted(set(hits))


def check_no_network_io(root: str, model: str) -> str:
    """Raise NetworkIOError if any XML file of aircraft/<model>/ under ``root`` declares network I/O.

    FD's prepare_aircraft strips network I/O at the source (and tests it); the stock JSBSim 737 in the venv still
    declares a telnet input (5137) and a QTJSBSIM UDP input (5139). This guard makes loading such a file -- e.g. the
    stock 737 by mistake -- a clear error instead of a silent port bind. Cheap: files are scanned once per process
    (keyed by path + size + mtime). Returns ``root``.
    """
    adir = os.path.join(root, "aircraft", model)
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(adir) for f in fs if f.lower().endswith(".xml"))
    key = "|".join(f"{f}:{os.path.getsize(f)}:{os.path.getmtime(f)}" for f in files)
    if _net_checked.get(adir) == key:
        return root
    bad = {}
    for f in files:
        hits = network_io(open(f, "rb").read().decode("utf-8", errors="replace"))
        if hits:
            bad[os.path.relpath(f, root)] = hits
    if bad:
        raise NetworkIOError(f"aircraft {model!r} under {root} declares JSBSim network I/O {bad}; refusing to load it. "
                             "Use Flight Dynamics' prepared copy (flight-dynamics/jsbsim_root, prepare_aircraft strips it).")
    _net_checked[adir] = key
    return root




def root_ok(model: str) -> bool:
    d = os.path.join(FD_ROOT, "aircraft", model)
    return os.path.exists(os.path.join(d, model + ".xml")) and os.path.exists(os.path.join(d, "flexwing_meta.json"))


def require_root(model: str) -> str:
    if not root_ok(model):
        raise FileNotFoundError(f"FD patched aircraft {model!r} not found in {FD_ROOT} (ask Flight Dynamics to run prepare_aircraft)")
    return FD_ROOT


def meta(model: str) -> Dict:
    return json.load(open(os.path.join(FD_ROOT, "aircraft", model, "flexwing_meta.json")))


def flex_overrides(gains: Dict[str, float], model: Optional[str] = None) -> Dict[str, float]:
    """WingParams overrides for FD: only the four tied-form genes, through FD's own (range-checking, raising)
    overrides_from_genome -> tied_stiffness(s, r). Nothing else from the gains dict is ever sent to FD."""
    return flexwing().overrides_from_genome({k: gains[k] for k in FLEX_GENES if k in gains}, strict=True)


def flutter_summary(margins: Dict) -> Dict:
    """Normalise FD's flutter output: capped margin + explicit no-flutter flag, whether FD returns inf or a capped value.

    Accepts FD's future fields if present (flutter_margin_cap / margin_cap, no_flutter_below_cap / no_flutter /
    flutter_found); otherwise uses PROVISIONAL_FLUTTER_CAP and derives the flag from the raw value.
    """
    fw = flexwing()
    cap = margins.get("flutter_margin_cap", margins.get("margin_cap", getattr(fw, "MARGIN_CAP", None)))
    cap_src = "fd" if cap is not None else "provisional"
    cap = float(cap) if cap is not None else PROVISIONAL_FLUTTER_CAP
    raw = float(margins.get("flutter_margin", math.inf))
    flag = None
    for k in ("flutter_not_found_below_cap", "no_flutter_below_cap", "no_flutter"):
        if k in margins:
            flag = bool(margins[k])
    if flag is None and "flutter_found" in margins:
        flag = not bool(margins["flutter_found"])
    if flag is None:
        flag = (not math.isfinite(raw)) or raw >= cap
    return {"flutter_margin": min(raw, cap) if math.isfinite(raw) else cap, "no_flutter_below_cap": flag,
            "flutter_margin_cap": cap, "flutter_cap_source": cap_src, "margin_error": bool(margins.get("margin_error", False))}


def wing_for(model: str, gains: Dict[str, float]):
    """FlexWing for this genome without flying (metrics read once per model from the FD copy)."""
    fw = flexwing()
    if model not in _metrics:
        fdm = fw.new_fdm(model, require_root(model), 1 / 120)
        _metrics[model] = (fdm["metrics/bw-ft"], fdm["metrics/Sw-sqft"], fdm["inertia/empty-weight-lbs"])
    span, area, empty = _metrics[model]
    return fw.FlexWing(fw.params_for(model, span, area, empty, **flex_overrides(gains, model)))


def struct_weights(params: Optional[Dict] = None):
    fw = flexwing()
    p = params or {}
    kw = {k: p[k] for k in fw.StructWeights.__dataclass_fields__ if k in p}
    return fw.StructWeights(**kw)


def precheck(model: str, gains: Dict[str, float], params: Optional[Dict] = None) -> Dict:
    """flexwing.margin_terms: flutter/divergence margins (V/V_D), hinge penalties below margin_req, mass term, fail flag."""
    return flexwing().margin_terms(wing_for(model, gains), struct_weights(params))


# =====================================================================================================================
# Flex v2 (FD flexbody.py / INTERFACE_v2.md): distributed wing structure + flexible empennage and fuselage.
# Read-only like v1: FD's prepared copies in jsbsim_root_v2/ are used as they are (never regenerated here), FD's
# gene_schema() is the source of truth for the structure genes, and FD's decode_genome_v2 validates (raises) every
# genome before anything is built or flown.
# =====================================================================================================================
FD_ROOT_V2 = os.path.join(FD_DIR, "jsbsim_root_v2")
V2_MARGIN_KINDS = ("flutter", "divergence", "reversal")


def flexbody():
    return _import("flexbody")


def gene_schema_v2(asymmetric: bool = False):
    """FD's v2 structure gene list (flexbody.gene_schema): name, lo, hi, scale, default, doc."""
    return flexbody().gene_schema(bool(asymmetric))


def gene_names_v2(asymmetric: bool = False):
    return tuple(g.name for g in gene_schema_v2(asymmetric))


def require_root_v2(model: str) -> str:
    """FD's v2 prepared root for this model, checked read-only (FD's flexbody.ensure_root_v2 would regenerate a stale
    copy, i.e. write into flight-dynamics/; we raise instead). Also refuses any JSBSim network I/O in the copy."""
    fb, fw_ = flexbody(), flexwing()
    p = os.path.join(FD_ROOT_V2, "aircraft", model, "flexbody_meta.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: Flight Dynamics must prepare {model} (flexbody.prepare_aircraft_v2)")
    m = json.load(open(p))
    if m.get("fmt", 0) < fb.PREPARE_FMT_V2 or m.get("v1_meta", {}).get("fmt", 0) < fw_.PREPARE_FMT:
        raise RuntimeError(f"{p} is stale (fmt {m.get('fmt')}/{m.get('v1_meta', {}).get('fmt')}, need "
                           f"{fb.PREPARE_FMT_V2}/{fw_.PREPARE_FMT}): Flight Dynamics must re-prepare {model}")
    check_no_network_io(FD_ROOT_V2, model)
    return FD_ROOT_V2


def struct_genes_v2(gains: Dict[str, float], asymmetric: bool = False) -> Dict[str, float]:
    """The v2 structure genes of a gains dict, validated by FD (unknown / v1 / fixed keys, NaN and out-of-range raise).
    Only the FD schema's names are ever sent to FD."""
    names = gene_names_v2(asymmetric)
    return flexbody().decode_genome_v2({k: gains[k] for k in names if k in gains}, bool(asymmetric))


_v2_models: "Dict[tuple, object]" = {}
_V2_CACHE_MAX = 8


def model_v2(model: str, genes: Dict[str, float], asymmetric: bool = False):
    """flexbody.FlexBodyModel for one genome (FE + modes, ~0.1-0.3 s), memoised per process for the margin pre-check
    and the scenarios of the same genome. The model is read-only once built."""
    key = (model, bool(asymmetric), tuple(sorted(genes.items())))
    m = _v2_models.get(key)
    if m is None:
        if len(_v2_models) >= _V2_CACHE_MAX:
            _v2_models.pop(next(iter(_v2_models)))
        m = flexbody().FlexBodyModel(model, genes, asymmetric=bool(asymmetric), root_v2=require_root_v2(model))
        _v2_models[key] = m
    return m


def struct_weights_v2(params: Optional[Dict] = None):
    """flexbody.StructWeightsV2 (defaults = FD's; any of its fields can be overridden from the task params)."""
    fb = flexbody()
    p = params or {}
    kw = {k: p[k] for k in fb.StructWeightsV2.__dataclass_fields__ if k in p}
    return fb.StructWeightsV2(**kw)


def conservative_margins_v2(m: Dict) -> Dict:
    """Conservative minimum over bodies (blocks) AND methods, recomputed from FD's per-block output:
    flutter = min over blocks of min(flutter_margin_qs, coalescence_margin, flutter_margin) (FD's flutter_margin is
    already min(quasi-steady, coalescence); taking all three guards against either method being dropped); divergence = min of div_margin; reversal = min of every
    <ctrl>_reversal_margin. Every value is FD's capped one (<= cap); a not-found flag means "no instability below the
    cap", i.e. the value IS the cap -- nothing here ever uses an uncapped or infinite margin. Non-finite values or
    margin_error -> margin 0 (hard fail)."""
    cap = float(m.get("margin_cap", flexbody().MARGIN_CAP))
    blocks = m.get("blocks") or {}
    vals = {k: [] for k in V2_MARGIN_KINDS}
    flags = {}
    err = bool(m.get("margin_error", False))

    def fin(x):
        nonlocal err
        try:
            x = float(x)
        except (TypeError, ValueError):
            err = True
            return 0.0
        if not math.isfinite(x):
            err = True
            return 0.0
        return min(x, cap)

    for bn, b in blocks.items():
        err = err or bool(b.get("margin_error", False))
        for k in ("flutter_margin", "flutter_margin_qs", "coalescence_margin"):
            if k in b:
                vals["flutter"].append((fin(b[k]), bn, k))
        if "div_margin" in b:
            vals["divergence"].append((fin(b["div_margin"]), bn, "div_margin"))
        for k, v in b.items():
            if k.endswith("_reversal_margin"):
                vals["reversal"].append((fin(v), bn, k))
        for k, v in b.items():
            if k.endswith("not_found_below_cap"):
                flags[f"{bn}.{k}"] = bool(v)
    out = {"margin_cap": cap, "flags": flags}
    binding = None
    for kind in V2_MARGIN_KINDS:
        lst = vals[kind]
        # no block reports this kind (e.g. a model without control surfaces) -> FD's overall value (capped)
        v, bn, k = min(lst) if lst else (fin(m.get({"flutter": "flutter_margin", "divergence": "div_margin",
                                                     "reversal": "reversal_margin"}[kind], cap)), "overall", kind)
        out[f"{kind}_margin"] = v
        out[f"{kind}_binding"] = f"{bn}.{k}"
        if binding is None or v < binding[0]:
            binding = (v, kind, f"{bn}.{k}")
    out["min_margin"], out["binding_kind"], out["binding"] = binding
    out["margin_error"] = err
    if err:
        out["min_margin"] = 0.0
    return out


def margin_penalties_v2(cons: Dict, wts, gate: float = 1.0):
    """Margins are constraints only (FD, 2026-10-06): m < gate (1.0) -> hard fail; gate <= m < margin_req (1.2) ->
    w * ((margin_req - m) / (margin_req - 1))^2; m >= margin_req -> exactly 0 (no reward for more margin). Inputs are
    the capped conservative margins (not-found = cap 3.0), so nothing above the cap can matter either."""
    terms, fail = {}, None
    for kind, wk in (("flutter", wts.w_flutter), ("divergence", wts.w_div), ("reversal", wts.w_reversal)):
        mg = cons[f"{kind}_margin"]
        if mg < gate and fail is None:
            fail = kind
        terms[{"flutter": "J_flutter_margin", "divergence": "J_div_margin", "reversal": "J_reversal_margin"}[kind]] = \
            wk * max(0.0, (wts.margin_req - mg) / (wts.margin_req - 1.0)) ** 2
    if cons["margin_error"]:
        fail = fail or "margin_error"
    return terms, fail


def precheck_v2(model: str, gains: Dict[str, float], asymmetric: bool = False, params: Optional[Dict] = None,
                gate: float = 1.0) -> Dict:
    """Pre-flight v2 screening (once per genome): FD margins_v2 + mass, our conservative minimum, the hard gate and the
    hinge^2 margin penalties  J_x = w_x * max(0, (margin_req - m_x) / (margin_req - 1))^2  (x = flutter, divergence,
    reversal; same rule and weights as FD's margin_terms_v2 and as v1). fail = first of flutter / divergence /
    reversal below the gate, or margin_error."""
    fb = flexbody()
    genes = struct_genes_v2(gains, asymmetric)
    mdl = model_v2(model, genes, asymmetric)
    wts = struct_weights_v2(params)
    fd = fb.margin_terms_v2(mdl, wts, gate=gate)
    cons = conservative_margins_v2(fd["margins"])
    terms, fail = margin_penalties_v2(cons, wts, gate)
    clip = tuple((params or {}).get("struct_v2_mass_credit_clip") or ())
    jm = mass_term_v2(fd["mass"], wts.w_mass, clip)
    sizing = {k: float(fd["terms"].get(k, 0.0)) for k in SIZING_TERMS_V2}  # FD §12 limit-load sizing (pre-flight)
    return {"genes": genes, "model": mdl, "weights": wts, "margins": cons, "fd_margins": fd["margins"],
            "fd_terms": fd["terms"], "fd_fail": fd["fail"], "terms": terms, "fail": fail, "mass": fd["mass"],
            "J_mass": jm["J_mass"], "J_mass_fd": float(fd["terms"]["J_mass"]), "mass_credit_clip": list(clip),
            "J_smooth": float(fd["terms"]["J_smooth"]), "sizing_terms": sizing,
            "sizing_ratios": dict(fd.get("sizing", {}).get("ratios", {})), "gate": gate}


# FD §12 sizing terms (pre-flight, reduced and full) + full-only flown torque/ip peaks. Source of truth = FD's
# flexbody.SIZING_TERMS / StructWeightsV2 (P2.5 added J_wing_tip_bm_limit). Prefer reading FD live so new keys land
# without a closed list; the literal below is the P2.5 pin used when FD is not importable (tests assert equality).
SIZING_TERMS_V2_PIN = ("J_wing_bm_limit", "J_wing_torque_limit", "J_wing_ip_limit", "J_wing_tip_bm_limit",
                       "J_tail_bm_limit", "J_fus_bm_limit")
FLOWN_FULL_ONLY_TERMS_V2 = ("J_wing_torque_peak", "J_wing_ip_peak")


def sizing_terms_v2():
    """FD's current SIZING_TERMS (tuple). Falls back to the P2.5 pin if flexbody is not importable."""
    try:
        return tuple(flexbody().SIZING_TERMS)
    except Exception:
        return SIZING_TERMS_V2_PIN


SIZING_TERMS_V2 = sizing_terms_v2()  # resolved at import; tests re-check against FD live

# Expected = P2.5 (flight-dynamics/v2_results/model_versions_post_p25.json, 2026-10-06 ~13:50 MST). phase2_flex flies
# FULL fidelity. Previous post-mass strings are accepted with a warning during the ER pilot transition (seed-3 pins
# still use them); anything else is unknown.
FD_V2_MODEL_VERSIONS = {
    "c172x": {"reduced": "reduced:flexv1:68dc59aa", "full": "full:flexv2:e11b8214"},
    "T38": {"reduced": "reduced:flexv1:ce107fcf", "full": "full:flexv2:8bf7a250"},
    "737": {"reduced": "reduced:flexv1:d3198780", "full": "full:flexv2:eeb82fb9"},
    "f16": {"reduced": "reduced:flexv1:66a7d816", "full": "full:flexv2:8baca00c"},
}
FD_V2_MODEL_VERSIONS_PREV_POST_MASS = {
    "c172x": {"reduced": "reduced:flexv1:983378a2", "full": "full:flexv2:11df8fe4"},
    "T38": {"reduced": "reduced:flexv1:52be19ae", "full": "full:flexv2:e9535bc7"},
    "737": {"reduced": "reduced:flexv1:870ff0b7", "full": "full:flexv2:2c73464b"},
    "f16": {"reduced": "reduced:flexv1:7ac4c5d4", "full": "full:flexv2:adfcae2f"},
}
V2_MARGIN_GATES = {"reduced": 0.9, "full": 1.0}  # = flexeval.MARGIN_GATE (FD §5/§7/§12)


def fd_model_version(model: str, fidelity: str = "full") -> str:
    """FD's own model_version string (flexeval.model_version; read-only: hashes FD's code/parameter/aircraft files)."""
    fe = _import("flexeval")
    return fe.model_version(fidelity, model, FD_ROOT)


def check_model_version(model: str, fidelity: str = "full", mode: str = "warn") -> Dict:
    """Compare FD's current model_version with the recorded P2.5 one.
    mode: warn (default) | raise | off.
    - match expected P2.5 -> ok
    - match previous post-mass (ER pilots still pinning those) -> warn only, never raise (transition window)
    - anything else (unknown) -> warn, or raise when mode='raise'."""
    if mode not in ("warn", "raise", "off"):
        raise ValueError(f"flex.model_version_check must be warn | raise | off, got {mode!r}")
    rec = FD_V2_MODEL_VERSIONS.get(model, {}).get(fidelity)
    prev = FD_V2_MODEL_VERSIONS_PREV_POST_MASS.get(model, {}).get(fidelity)
    out = {"model": model, "fidelity": fidelity, "recorded": rec, "previous_post_mass": prev,
           "current": None, "match": None, "status": None, "message": None}
    if mode == "off":
        return out
    try:
        cur = fd_model_version(model, fidelity)
    except Exception as e:
        out["message"] = f"could not compute FD model_version for {model} {fidelity} ({type(e).__name__}: {e})"
        out["status"] = "unverified"
        if mode == "raise":
            raise RuntimeError(out["message"]) from e
        return out
    out["current"] = cur
    if cur == rec:
        out.update(match=True, status="match")
        return out
    if cur == prev:
        out.update(match=False, status="previous_post_mass",
                   message=(f"FD model_version for {model} {fidelity} is the previous post-mass string {cur} "
                            f"(genome expects P2.5 {rec}): accepted during the ER pilot transition; re-pin when ER cuts "
                            "new runs against model_versions_post_p25.json"))
        return out  # never raise on the known previous set
    out.update(match=False, status="unknown",
               message=(f"FD model_version for {model} {fidelity} is {cur}, genome recorded P2.5 {rec} "
                        f"(previous post-mass was {prev}): unknown -- re-check phase2_flex terms and update "
                        "fd_bridge.FD_V2_MODEL_VERSIONS"))
    if mode == "raise":
        raise RuntimeError(out["message"])
    return out


# Per-body structural mass changes from FD's FlexBodyModel.mass_summary() (exact, per body; FD's J_mass is their total)
MASS_BODIES = {"wing": ("wingR_lb", "wingL_lb"), "ht": ("ht_lb",), "vt": ("vt_lb",), "fus": ("fus_lb",)}


def mass_term_v2(mass: Dict, w_mass: float, clip=()) -> Dict:
    """J_mass = w_mass * sum_b dm_b / m_flexible_baseline (= FD's 0.3 * total_frac when nothing is clipped).
    clip: bodies (ht, vt, fus) whose mass *decrease* earns nothing (dm_b -> max(0, dm_b)); an increase still costs.
    Default OFF. It was the interim fix for the tail/fuselage mass-credit exploit (2026-10-06 ~08:00); FD's §12 fix
    (minimum-gauge floor + pre-flight limit-load sizing terms, model_version full:flexv2:11df8fe4 for c172x) replaced
    it, so phase2_flex no longer sets it. Kept only for A/B comparisons."""
    bad = set(clip) - (set(MASS_BODIES) - {"wing"})
    if bad:
        raise ValueError(f"struct_v2_mass_credit_clip: unknown or unsupported bodies {sorted(bad)} (allowed: ht, vt, fus)")
    dm = {b: float(sum(mass[k] for k in ks)) for b, ks in MASS_BODIES.items()}
    used = {b: (max(0.0, v) if b in clip else v) for b, v in dm.items()}
    base = float(mass["baseline_flexible_lb"])
    return {"J_mass": w_mass * sum(used.values()) / base, "dm_lb": dm, "dm_used_lb": used}
