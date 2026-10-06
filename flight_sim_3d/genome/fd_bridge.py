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
