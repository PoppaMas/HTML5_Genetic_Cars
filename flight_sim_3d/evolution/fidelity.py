"""Structural-fidelity adapter: the ONLY place evolution/ touches Flight Dynamics' code.

    rigid    JSBSim rigid body, our sim.simulate unchanged (bit-identical to the Phase-1 runs); per-scenario
    reduced  FD flexeval.evaluate(fidelity='reduced'): v1 FlexWing n_bend=1 + torsion on the v2 genome projected by FD's
             project_to_reduced (wing only); per genome
    full     FD flexeval.evaluate(fidelity='full'): flex v2 (flexbody.py: 25-DOF wings + empennage + fuselage); per genome
    full_a1  FD flexeval_a1.evaluate(fidelity='full_a1'): P3-A1 denser full model (flexbody_a1.FlexBodyModelA1: 64 strips
             and 4b+3t+2ip modes per semi-wing, tails / fuselage as full, 31 DOF; J_wing_tip_bm_limit station-exact at
             eta 0.875); per genome. Opt-in (INTERFACE_v2 section 13). rigid / reduced / full never touch flexeval_a1.

FD's modules (flight-dynamics/flexeval.py, flexbody.py, flexwing.py, coupled_sim.py; flexeval_a1.py / flexbody_a1.py
only for full_a1, imported lazily) are IMPORTED, never copied, with
bytecode writing disabled so nothing lands in their folder. flexeval.evaluate calls FD's ensure_root/ensure_root_v2,
which write only when a prepared root is stale: we check freshness read-only first (same tests) and refuse if stale, and
in our processes FD's prepare_aircraft / prepare_aircraft_v2 are replaced by functions that raise, so FD's folder can
never be written from here. FD spec: flight-dynamics/INTERFACE_v2.md (section 7 = the agreed fidelity contract).

* model_version comes from FD's result: rigid:jsbsim1.3.1:<sha8> (byte-identical to ours), reduced:flexv1:<sha8>,
  full:flexv2:<sha8>, full_a1:flexv2a1:<sha8> (flexeval_a1.model_version). full and full_a1 never share a cache entry:
  the cache key carries fidelity + model_version and cache.eval_key refuses a model_version of another fidelity. The reduced gate is per aircraft (`reduced_gate`, see batch config `fidelity_per_aircraft`): FD's
  MARGIN_GATE['reduced'] is set to it for the duration of each call, so FD's own model_version hashes the gate too.
* Terms: FD's 24 TERM_KEYS at every fidelity (inapplicable = 0.0, not in terms_available, never NaN).
* Cost = float(np.mean(per-scenario costs)) at every fidelity; a margin-gate fail is fail_cost (2*fail_base), not flown;
  we then list every scenario with cost = fail_cost so per-scenario rows stay aligned (mean unchanged).
* feasible = (status == 'ok') at the fidelity it was scored at (feasibility_fidelity). In multi-fidelity runs the batch
  trusts feasibility only from the authoritative fidelity: screen-only rows get feasible = None.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import genome as genome_mod
from . import sim

FD_DIR = sim.fd_dir()   # $EVOLUTION_FD_DIR or <team root>/flight-dynamics; no absolute default
FIDELITIES = ("rigid", "reduced", "full", "full_a1")
RANK = {"rigid": 0, "reduced": 1, "full": 2, "full_a1": 3}
LABELS = {"rigid": "rigid", "reduced": "reduced(flexv1 on projected v2 genome)", "full": "full(flexv2)",
          "full_a1": "full_a1(flexv2a1: P3-A1, 64-strip 4b+3t+2ip wings)"}
A1 = "full_a1"
# FE-model fidelities (FD FlexBodyModel family: FE nodal wings + empennage + fuselage, prepared root <root>_v2, v2_map)
FULL_LIKE = ("full", "full_a1")
# reduced margin gate (FD default 0.9) and minimum fraction of the population re-scored at full in multi-fidelity mode.
# Swept wings: FD found reduced margins optimistic (737: reduced passed all 32, full failed 9-12) -> gate 1.0, >= 25 %.
DEFAULT_PER_AIRCRAFT = {"c172x": {"reduced_gate": 0.9, "min_full_frac": 0.0},
                        "T38": {"reduced_gate": 1.0, "min_full_frac": 0.25},
                        "737": {"reduced_gate": 1.0, "min_full_frac": 0.25},
                        "f16": {"reduced_gate": 1.0, "min_full_frac": 0.25}}
FALLBACK_PER_AIRCRAFT = {"reduced_gate": 1.0, "min_full_frac": 0.25}   # unknown aircraft: conservative


class FidelityUnavailable(Exception):
    pass


_FD: Dict[str, object] = {}


def _refuse_prepare(*a, **k):
    raise FidelityUnavailable("evolution never prepares FD roots (FD's folder is read-only from here); "
                              "ask Flight Dynamics to re-prepare jsbsim_root / jsbsim_root_v2")


def fd_modules():
    """dict(fe=flexeval, fb=flexbody, fw=flexwing, cs=coupled_sim), imported from FD's folder without bytecode."""
    if not _FD:
        if not os.path.isdir(FD_DIR):
            raise FidelityUnavailable(f"Flight Dynamics folder not found: {FD_DIR}")
        prev = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            if FD_DIR not in sys.path:
                sys.path.insert(0, FD_DIR)
            import flexwing, coupled_sim, flexbody, flexeval   # noqa: E401,E402
        finally:
            sys.dont_write_bytecode = prev
        flexwing.prepare_aircraft = _refuse_prepare           # in-process guard (FD's files are untouched)
        flexbody.prepare_aircraft_v2 = _refuse_prepare
        _FD.update(fe=flexeval, fb=flexbody, fw=flexwing, cs=coupled_sim)
    return _FD


_FDA1: Dict[str, object] = {}


def fd_a1_modules():
    """dict(fa=flexeval_a1, fba1=flexbody_a1), imported from FD's folder without bytecode, only when full_a1 is used
    (an FD tree without the P3-A1 files still serves rigid / reduced / full)."""
    if not _FDA1:
        fd_modules()
        prev = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            import flexbody_a1, flexeval_a1   # noqa: E401,E402
        except ImportError as e:
            raise FidelityUnavailable(f"full_a1 needs FD's flexeval_a1.py / flexbody_a1.py in {FD_DIR}: {e}") from e
        finally:
            sys.dont_write_bytecode = prev
        _FDA1.update(fa=flexeval_a1, fba1=flexbody_a1)
    return _FDA1


def label(fidelity: str) -> str:
    return LABELS[fidelity]


def per_aircraft(name: str, overrides: Optional[Dict] = None) -> Dict:
    d = dict(DEFAULT_PER_AIRCRAFT.get(name, FALLBACK_PER_AIRCRAFT))
    d.update((overrides or {}).get(name, {}))
    return d


# ----------------------------------------------------------------------------- struct genes (FD v2, 12 genes)
def struct_schema(asymmetric: bool = False) -> List[genome_mod.Gene]:
    """FD's v2 structure genes (flexbody.gene_schema): 12, or 14 with FD's 2 optional asymmetry genes (off by default;
    FD detects them from the keys). Gene.default = FD's baseline (1.0 multipliers, 0.02 damping)."""
    fb = fd_modules()["fb"]
    return [genome_mod.Gene(g.name, g.lo, g.hi, kind=g.scale, doc=g.doc, default=g.default) for g in fb.gene_schema(asymmetric)]


def baseline_u(schema: Sequence[genome_mod.Gene]) -> np.ndarray:
    """Normalized position of FD's baseline value of each gene (= Genome's init_pop.baseline_u)."""
    return np.array([g.encode(g.default) for g in schema], float)


def struct_from(struct_genome) -> Optional[Dict[str, float]]:
    """None | {gene: value} | normalized vector (FD order) -> {gene: value} clamped to FD's ranges (a log decode at
    u = 1 can exceed hi by 1 ulp, which FD rejects)."""
    if struct_genome is None:
        return None
    sch = {g.name: g for g in struct_schema(True)}
    if not isinstance(struct_genome, dict):
        struct_genome = genome_mod.decode(list(struct_genome), struct_schema(len(struct_genome) > 12))
    return {k: min(max(float(v), sch[k].min), sch[k].max) if k in sch else float(v) for k, v in struct_genome.items()}


# ----------------------------------------------------------------------------- prepared roots (read-only checks)
def roots(P: sim.Profile):
    if P.aircraft_root is None:
        raise FidelityUnavailable(f"{P.aircraft}: flex fidelities need FD's prepared root (aircraft_root = FD jsbsim_root)")
    return P.aircraft_root, os.path.normpath(P.aircraft_root) + "_v2"


def check_prepared(P: sim.Profile, fidelity: str):
    """Same staleness tests as FD's ensure_root / ensure_root_v2, read-only. Raises FidelityUnavailable if stale."""
    m = fd_modules()
    fw, fb = m["fw"], m["fb"]
    root, rv2 = roots(P)
    model = P.aircraft
    xml = os.path.join(root, "aircraft", model, model + ".xml")
    meta = os.path.join(root, "aircraft", model, "flexwing_meta.json")
    if (not os.path.exists(xml) or not os.path.exists(meta) or "flexwing_F" not in open(xml, encoding="utf-8", errors="replace").read()
            or json.load(open(meta)).get("fmt", 1) < fw.PREPARE_FMT):
        raise FidelityUnavailable(f"{model}: {root} is not FD-prepared (flexwing) or stale")
    p2 = os.path.join(rv2, "aircraft", model, "flexbody_meta.json")
    ok = False
    if os.path.exists(p2):
        mm = json.load(open(p2))
        ok = mm.get("fmt", 0) >= fb.PREPARE_FMT_V2 and mm["v1_meta"].get("fmt", 0) >= fw.PREPARE_FMT
    if not ok:
        raise FidelityUnavailable(f"{model}: {rv2} is not FD-prepared (flexbody v2) or stale")


@contextlib.contextmanager
def _gate(reduced_gate: Optional[float]):
    fe = fd_modules()["fe"]
    old = dict(fe.MARGIN_GATE)
    if reduced_gate is not None:
        fe.MARGIN_GATE["reduced"] = float(reduced_gate)
    try:
        yield fe
    finally:
        fe.MARGIN_GATE.clear()
        fe.MARGIN_GATE.update(old)


_MV: Dict[tuple, str] = {}


def _sha8(payload) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]


def model_version(profile_d: Dict, fidelity: str, reduced_gate: Optional[float] = None) -> str:
    key = (json.dumps(profile_d, sort_keys=True), fidelity, reduced_gate if fidelity == "reduced" else None)
    if key in _MV:
        return _MV[key]
    P = sim.Profile.from_dict(profile_d)
    if fidelity == "rigid":
        import jsbsim
        mv = f"rigid:jsbsim{jsbsim.__version__}:" + _sha8({"model_files": sim.model_files_sha(P.aircraft, P.aircraft_root)})
    else:
        check_prepared(P, fidelity)
        root, rv2 = roots(P)
        if fidelity == A1:
            mv = fd_a1_modules()["fa"].model_version(A1, P.aircraft, root, rv2)
        else:
            with _gate(reduced_gate) as fe:
                mv = fe.model_version(fidelity, P.aircraft, root, rv2)
    _MV[key] = mv
    return mv


# ----------------------------------------------------------------------------- flex_state for Sim Bridge (v2 wings + v2_map)
N_NODES = 9          # modal display nodes per semi-wing, xi = 0 (beam root) .. 1 (tip), evenly spaced
FLEX_STATE_SCHEMA = "evolution-flex-state/3"   # /3: full = FD FE wings (v2_map) + wing*_modal; reduced = modal wingR/L
V2_MAP_COMPONENTS = ("wingR", "wingL", "htail", "vtail", "fuselage")
TWIST_DOC = ("twist [rad]: right-hand rotation about the component axis node0 -> nodeN (root -> tip). wingR: + = leading "
             "edge up; wingL: + = leading edge DOWN. FD raw twist (flexwing v1 PsiT@eta and flexbody v2 Surface.PsiT@eta, "
             "tip_twist_*_deg) is + = leading edge up / nose-up (increases local alpha: G = cos(sweep)*PsiT - "
             "sin(sweep)*dPhiW) on BOTH semi-wings, so wingR.twist = +theta_R, wingL.twist = -theta_L. Same signs in "
             "sim-bridge v2_map (V2_MAP_VERSION) for FE nodal wings.")
_V2_MAP = {}


def v2_map_mod():
    """sim-bridge/sim_bridge/v2_map.py loaded read-only (stdlib only; no evolution / sim_bridge package import)."""
    if not _V2_MAP:
        path = os.path.join(sim.TEAM_ROOT, "sim-bridge", "sim_bridge", "v2_map.py")
        if not os.path.isfile(path):
            raise FidelityUnavailable(f"sim-bridge v2_map not found: {path}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("evolution._v2_map_ro", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        prev = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = prev
        if getattr(mod, "V2_MAP_VERSION", None) != "2.0.0":
            raise RuntimeError(f"v2_map V2_MAP_VERSION {getattr(mod, 'V2_MAP_VERSION', None)!r} != '2.0.0'")
        _V2_MAP["m"] = mod
    return _V2_MAP["m"]


def rp_offset_body_ft(fdm) -> list:
    """AERORP relative to the CG in body FRD ft (read-only property reads); FD node_layout origin = CG when given this."""
    return [-(fdm["metrics/aero-rp-x-in"] - fdm["inertia/cg-x-in"]) / 12.0,
            (fdm["metrics/aero-rp-y-in"] - fdm["inertia/cg-y-in"]) / 12.0,
            -(fdm["metrics/aero-rp-z-in"] - fdm["inertia/cg-z-in"]) / 12.0]


def make_fd_model(profile_d: Dict, struct, fidelity: str):
    """Public: FD FlexBodyModel (full), FlexBodyModelA1 (full_a1) or projected FlexWing (reduced) for (profile, struct
    genome). Sim Bridge can hold this and call FlexState.nodes / node_layout without using private `_struct_obj` /
    `_aircraft_entry`."""
    if fidelity not in ("reduced",) + FULL_LIKE:
        raise ValueError(f"make_fd_model: fidelity must be reduced|full|full_a1, got {fidelity!r}")
    P = sim.Profile.from_dict(profile_d)
    check_prepared(P, fidelity)
    _root, rv2 = roots(P)
    m = fd_modules()
    return _struct_obj(m["fe"], m["fb"], struct_from(struct), P.aircraft, fidelity, rv2)


def _wing_surfaces(obj, fidelity):
    """[(name, geometry dict, function eta_global -> (w_nodes ft +up, theta_nodes rad +nose-up))] for wingR / wingL."""
    xi_n = np.linspace(0.0, 1.0, N_NODES)
    out = []
    if fidelity in FULL_LIKE:
        mdl = obj
        for name, surf in (("wingR", mdl.wingR), ("wingL", mdl.wingL)):
            sl = mdl.slices[name]

            def fn(eta, surf=surf, sl=sl):
                e = np.asarray(eta, float)[sl]
                w, th = surf.PhiW @ e, surf.PsiT @ e
                xs = np.concatenate([[0.0], surf.xi, [1.0]])
                return (np.interp(xi_n, xs, np.concatenate([[0.0], w, [float(surf.tipW @ e)]])),
                        np.interp(xi_n, xs, np.concatenate([[0.0], th, [float(surf.tipT @ e)]])))
            geo = {"y0": surf.y0, "L": surf.L, "s": surf.s, "lam": surf.lam, "x_ea": surf.sp.x_ea, "taper": surf.sp.taper,
                   "c_root": surf.c_root, "interp": f"strip values ({int(surf.sp.n_el)} strips) linearly interpolated, 0 "
                                                    "at the clamped root, FD tip value at xi = 1"}
            out.append((name, geo, fn))
    else:
        fw = fd_modules()["fw"]
        w = obj
        nb = w.p.n_bend
        B = np.array([[float(fw._bend(k, x)) for k in range(nb)] for x in xi_n])
        T = np.array([float(fw._tors(x)) for x in xi_n])
        for side, name in ((0, "wingR"), (1, "wingL")):
            def fn(eta, side=side):
                e = np.asarray(eta, float)[side]
                return B @ e[:nb], T * e[nb]
            geo = {"y0": w.y0, "L": w.L, "s": w.s, "lam": w.lam, "x_ea": w.p.x_ea, "taper": w.p.taper,
                   "c_root": None, "chord": w.chord, "interp": "v1 mode shapes evaluated at the nodes"}
            out.append((name, geo, fn))
    return out


_SOURCE_NAME = {"full": "flexbody v2", "reduced": "flexwing v1 (reduced)", "full_a1": "flexbody_a1 (P3-A1, 64-strip wings)"}


def structure_geometry(obj, fidelity, fdm) -> Dict:
    """Undeformed elastic axes (one beam per semi-wing) in body FRD metres, origin at the CG (trajectory frame).
    x: quarter chord swept through the MAC quarter chord (taken as JSBSim's AERORP), EA at x_ea of the local chord;
    z = AERORP z (no dihedral). Approximate geometry for display; the physics does not use it."""
    xi = np.linspace(0.0, 1.0, N_NODES)
    x_rp = -(fdm["metrics/aero-rp-x-in"] - fdm["inertia/cg-x-in"]) / 12.0
    z_rp = -(fdm["metrics/aero-rp-z-in"] - fdm["inertia/cg-z-in"]) / 12.0
    ft = sim.FT
    comps, fns = [], []
    for name, g, fn in _wing_surfaces(obj, fidelity):
        y = g["y0"] + xi * g["L"]
        c = np.asarray(g["chord"](y), float) if g.get("chord") else g["c_root"] * (1 - (1 - g["taper"]) * y / g["s"])
        y_mac = g["s"] / 3 * (1 + 2 * g["taper"]) / (1 + g["taper"])
        aft = (y - y_mac) * math.tan(g["lam"]) + (g["x_ea"] - 0.25) * c
        sgn = 1.0 if name == "wingR" else -1.0
        nodes = [[round(float((x_rp - a) * ft), 6), round(float(sgn * yy * ft), 6), round(float(z_rp * ft), 6)]
                 for a, yy in zip(aft, y)]
        comps.append({"name": name, "axis_nodes_body_m": nodes, "dof": ["dz", "dy", "twist"],
                      "node_span_frac": [float(v) for v in xi], "node_values": g["interp"]})
        fns.append((name, fn))
    structure = {"schema": FLEX_STATE_SCHEMA, "synthetic": False, "fidelity": fidelity,
                 "source": f"FD {_SOURCE_NAME.get(fidelity, fidelity)} 9-node modal wings "
                           "(adapter: evolution/fidelity.py fd_to_structure_channels); at full renamed to wing*_modal "
                           "and FE nodal wings / htail / vtail / fuselage come from sim-bridge v2_map",
                 "components": comps,
                 "units": {"axis_nodes_body_m": "m, body FRD (x fwd, y right, z down), origin CG", "dz": "m (body z, + down)",
                           "dy": "m (body y, + right)", "dx": "m (body x, + forward)", "twist": "rad"},
                 "dz_doc": "elastic deflection of the elastic axis relative to the undeformed (jig) shape, including the "
                           "1-g trim deflection; = -w (FD w is + up)",
                 "dy_doc": "modal wings: 0.0 (in-plane not in the 9-node modal map). FE wings (full, v2_map): use dx.",
                 "dx_doc": "FE wing in-plane (full only, v2_map): dx = -v * 0.3048 (FD v + aft -> body + forward)",
                 "twist_doc": TWIST_DOC,
                 "raw_channels_doc": "flex.<key> = FD coupler diagnostics as FD names them (root loads lbf*ft incl. tail "
                                     "and fuselage, tip deflections ft, twists/incidences deg); see INTERFACE_v2.md"}
    return {"structure": structure, "fns": fns}


def fd_to_structure_channels(eta, geom) -> Dict[str, float]:
    """THE FD -> Sim Bridge mapping (wings): dz = -w * 0.3048, dy = 0, wingR[.modal].twist = +theta_R, wingL[.modal].twist = -theta_L."""
    out: Dict[str, float] = {}
    ft = sim.FT
    for name, fn in geom["fns"]:
        w, th = fn(eta)
        # startswith: after schema /3 full renames modal wings to wingR_modal / wingL_modal
        tsign = 1.0 if name.startswith("wingR") else -1.0
        for i, v in enumerate(w):
            out[f"{name}.dz.{i}"] = float(-v * ft)
        for i in range(len(w)):
            out[f"{name}.dy.{i}"] = 0.0
        for i, v in enumerate(th):
            out[f"{name}.twist.{i}"] = float(tsign * v)
    return out


class FlexState:
    """flex_state handed to recorders (schema evolution-flex-state/3), valid during the recorder call.

    Public surface (Sim Bridge):
      .schema / .structure / .fidelity / .model_version / .eta / .raw
      .fd_model          FD FlexBodyModel (full), FlexBodyModelA1 (full_a1) or FlexWing (reduced); same object the flight uses
      .nodes()           current-frame FE nodal values {body: {field: [float]}}, or None at reduced
      .node_layout()     FD node_layout list (body FRD ft, origin CG), or None at reduced
      .v2_geometry       sim-bridge v2_map geometry dict (full / full_a1), or None
      .v2_map_version    "2.0.0" when v2_map is wired (full / full_a1), else None
      .channels()        SB channels + flex.* raw (see below)
      .as_dict()

    Channels at full and full_a1 (A1: 65 FE wing nodes per side instead of 33): FE nodal wingR/wingL + htail/vtail/fuselage + struct.* (v2_map, FD §11 signs) and the 9-node
    modal wings under wingR_modal.* / wingL_modal.* (node_span_frac in the structure header). At reduced: modal
    wingR/wingL only (unchanged names) + flex.*.
    """
    __slots__ = ("structure", "eta", "raw", "_geom", "fidelity", "model_version",
                 "_fd_model", "_v2_geo", "_rp_offset_ft")
    schema = FLEX_STATE_SCHEMA

    def __init__(self, geom, eta, raw, fidelity, model_version, *, fd_model=None, v2_geo=None, rp_offset_ft=None):
        self._geom, self.structure = geom, geom["structure"]
        self.eta, self.raw, self.fidelity, self.model_version = eta, raw, fidelity, model_version
        self._fd_model, self._v2_geo, self._rp_offset_ft = fd_model, v2_geo, rp_offset_ft

    @property
    def fd_model(self):
        """FD FlexBodyModel (full), FlexBodyModelA1 (full_a1) or FlexWing (reduced). Read-only: do not mutate."""
        return self._fd_model

    @property
    def v2_geometry(self):
        return self._v2_geo

    @property
    def v2_map_version(self):
        return None if self._v2_geo is None else v2_map_mod().V2_MAP_VERSION

    def nodes(self):
        """Current-frame FD FE nodal values in the form v2_map.nodes_frame expects, or None at reduced / without a model."""
        if self.fidelity not in FULL_LIKE or self._fd_model is None:
            return None
        fb = fd_modules()["fb"]
        return v2_map_mod().nodes_frame(fb.node_values(self._fd_model, self.eta))

    def node_layout(self):
        """FD flexbody.node_layout (body FRD ft, origin CG when rp offset was supplied), or None at reduced."""
        if self.fidelity not in FULL_LIKE or self._fd_model is None:
            return None
        return fd_modules()["fb"].node_layout(self._fd_model, self._rp_offset_ft)

    def channels(self) -> Dict[str, float]:
        ch = fd_to_structure_channels(self.eta, self._geom)   # modal (wing*_modal at full; wingR/L at reduced)
        if self._v2_geo is not None:
            vm = v2_map_mod()
            ch.update(vm.map_v2_record(self.raw, self._v2_geo, self.nodes(), components=V2_MAP_COMPONENTS, scalars=True))
        ch.update({"flex." + k: float(v) for k, v in self.raw.items()})
        return ch

    def as_dict(self) -> Dict:
        return {"schema": self.schema, "fidelity": self.fidelity, "model_version": self.model_version,
                "v2_map_version": self.v2_map_version, "structure": self.structure, "channels": self.channels(),
                "modal_eta": np.asarray(self.eta, float).tolist(),
                **({"nodes": self.nodes()} if self.fidelity in FULL_LIKE else {})}


class SBHook:
    """Wraps FD's FlexHookV2 (attach / wrap / finish unchanged) and serves Sim Bridge flex_state. Used only for the
    telemetry flight (recorder / trajectory export); the cost always comes from flexeval.evaluate."""

    def __init__(self, fd_hook, obj, fidelity, model_version):
        self.h, self.obj, self.fidelity, self.model_version = fd_hook, obj, fidelity, model_version
        self.geom = None
        self._v2_geo = None
        self._rp_offset_ft = None
        self._probe = None
        self._fdm = None

    def attach(self, fdm):
        self.h.attach(fdm)

    def wrap(self, fdm):
        self._fdm = fdm
        return self.h.wrap(fdm)

    def finish(self, status):
        return self.h.finish(status)

    def _ensure_geom(self, eta):
        if self.geom is not None:
            return
        modal = structure_geometry(self.obj, self.fidelity, self._fdm)
        if self.fidelity not in FULL_LIKE:
            self.geom = modal
            return
        # full / full_a1: rename 9-node modal wings; FE nodal wings + empennage + fuselage from v2_map
        for c in modal["structure"]["components"]:
            c["name"] = c["name"] + "_modal"
            c["role"] = "modal_9node"
        modal["fns"] = [(nm + "_modal", fn) for nm, fn in modal["fns"]]
        vm, fb = v2_map_mod(), fd_modules()["fb"]
        self._rp_offset_ft = rp_offset_body_ft(self._fdm)
        layout = fb.node_layout(self.obj, self._rp_offset_ft)
        self._v2_geo = vm.geometry_from_layout(layout, getattr(self.obj, "model", None))
        nodes = vm.nodes_frame(fb.node_values(self.obj, eta))
        status = vm.component_status(self._v2_geo, nodes)
        modal["structure"] = vm.structure_block(self._v2_geo, components=V2_MAP_COMPONENTS, estimated=status,
                                                base=modal["structure"],
                                                extra={"evolution_flex_state": FLEX_STATE_SCHEMA,
                                                       "modal_wings": ["wingR_modal", "wingL_modal"]})
        self.geom = modal

    def state(self) -> FlexState:
        px = self.h.px
        if px is not None and px.started:
            eta, raw = self.h.coupler.eta, self.h.coupler.last
        else:
            if self._probe is None:   # t = 0: trim equilibrium on a throw-away ONE-WAY coupler through a read-only FDM
                probe = self.h._new_coupler("oneway")
                probe.initialize(sim.ReadOnlyFDM(self._fdm))
                self._probe = (np.array(probe.eta, copy=True), dict(probe.last))
            eta, raw = self._probe
        self._ensure_geom(eta)
        return FlexState(self.geom, eta, raw, self.fidelity, self.model_version,
                         fd_model=self.obj, v2_geo=self._v2_geo, rp_offset_ft=self._rp_offset_ft)


# ----------------------------------------------------------------------------- evaluation
PER_KEYS = ("cost", "sim_cost", "status", "t_end", "track", "effort", "comfort", "heading_rms", "hdg_drift_deg",
            "hdg_max_abs_err_deg", "hold_osc", "hold_pp_ft", "draft_residual_ft", "draft_max_err_ft", "struct",
            "tip_max_ft", "twist_max_deg", "delta_mass_lb_applied")


def _flex_profile(P: sim.Profile, fidelity: str) -> sim.Profile:
    root, rv2 = roots(P)
    return sim.Profile.from_dict(dict(P.to_dict(), aircraft_root=root if fidelity == "reduced" else rv2))


def _struct_obj(fe, fb, struct, model, fidelity, rv2):
    if fidelity == "reduced":
        return fe.reduced_wing(struct, model, rv2)
    if fidelity == A1:
        return fd_a1_modules()["fba1"].FlexBodyModelA1(model, struct, asymmetric=fe._is_asym(struct), root_v2=rv2)
    return fb.FlexBodyModel(model, struct, asymmetric=fe._is_asym(struct), root_v2=rv2)


def _fd_evaluate(fidelity: str):
    """FD's evaluate for a flex fidelity: flexeval.evaluate for reduced / full (exactly as before P3-A1), flexeval_a1.evaluate
    for full_a1."""
    return fd_a1_modules()["fa"].evaluate if fidelity == A1 else fd_modules()["fe"].evaluate


def _fd_hook(fe, fidelity: str):
    """FD's hook class: flexeval.FlexHookV2 (reduced / full) or flexeval_a1.FlexHookA1 (full_a1, = FlexHookV2 with the A1
    model; only its node-telemetry gate differs)."""
    return fd_a1_modules()["fa"].FlexHookA1 if fidelity == A1 else fe.FlexHookV2


# ----------------------------------------------------------------------------- Phase 2 mass-credit clip (Genome)
MASS_BODIES = {"wing": ("wingR_lb", "wingL_lb"), "ht": ("ht_lb",), "vt": ("vt_lb",), "fus": ("fus_lb",)}


def mass_term_clipped(mass: Dict, w_mass: float, clip=()) -> Dict:
    """= Genome fd_bridge.mass_term_v2: J_mass = w_mass * sum_b dm_b / baseline_flexible_lb, with dm_b -> max(0, dm_b)
    for the clipped bodies (ht, vt, fus): a decrease earns nothing, an increase still costs. mass = FD's
    FlexBodyModel.mass_summary() (flexeval result 'mass'; MIN_GAUGE-floored since FD section 12)."""
    bad = set(clip) - {"ht", "vt", "fus"}
    if bad:
        raise ValueError(f"mass credit clip: unsupported bodies {sorted(bad)} (allowed: ht, vt, fus)")
    dm = {b: float(sum(mass[k] for k in ks)) for b, ks in MASS_BODIES.items()}
    used = {b: (max(0.0, v) if b in clip else v) for b, v in dm.items()}
    return {"J_mass": w_mass * sum(used.values()) / float(mass["baseline_flexible_lb"]), "dm_lb": dm, "dm_used_lb": used}


def apply_mass_credit_clip(out: Dict, mass: Dict, clip, w_mass: float) -> Dict:
    """Replace FD's J_mass by the clipped one in a flown reduced/full result: every per-scenario cost carries FD's
    pre-flight sum (J_mass included), so each gets delta = J_clipped - J_fd and cost = mean again. Untouched (bit for
    bit) when no clipped body lost mass. Records J_mass_fd, mass_credit_clip, mass_credit_delta."""
    jm_fd = float(out["terms"]["J_mass"])
    tm = mass_term_clipped(mass, w_mass, clip)
    clipped = any(tm["dm_lb"][b] < 0.0 for b in clip)
    out["J_mass_fd"], out["mass_credit_clip"] = jm_fd, list(clip)
    out["mass_credit_delta"] = 0.0
    if clipped:
        d = tm["J_mass"] - jm_fd
        for e in out["per_scenario"]:
            e["cost"] = float(e["cost"]) + d
        out["cost"] = float(np.mean([e["cost"] for e in out["per_scenario"]]))
        out["terms"]["J_mass"] = float(tm["J_mass"])
        out["mass_credit_delta"] = float(d)
    return out


def evaluate_genome(profile_d: Dict, gains: Dict[str, float], struct: Optional[Dict[str, float]], scs_d: Sequence[Dict],
                    fidelity: str, reduced_gate: Optional[float] = None, recorder=None, record: bool = False,
                    sample_hz: float = 30.0, telemetry: str = "sb") -> Dict:
    """One genome over its scenarios at one fidelity -> {cost, status, per_scenario[...], terms, terms_available,
    feasible, feasibility_fidelity, fidelity, model_version, margins, margins_fidelity, margin_gate, reduced_gate,
    mass_total_frac, projection?, telemetry_check?, trajectories? (record)}. JSON-able (what the batch caches).
    record + telemetry='fd' (viz): trajectories from FD's own record=True flight (raw flex.* channels only, no extra
    flight); telemetry='sb' (export / recorder): a second flight with FD's FlexHookV2 wrapped by SBHook (Sim Bridge
    wingR/wingL channels + raw flex.*), whose cost is checked bit for bit against FD's per-scenario sim_cost."""
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}")
    P = sim.Profile.from_dict(profile_d)
    scs = [sim.Scenario.from_dict(s) for s in scs_d]
    mv = model_version(profile_d, fidelity, reduced_gate)
    struct = struct_from(struct)
    if fidelity == "rigid":
        per = [sim.simulate(gains, sc, P, record=record, sample_hz=sample_hz, recorder=recorder) for sc in scs]
        for r in per:
            r["fidelity"], r["model_version"] = "rigid", mv
        out = aggregate(per)
        out["per_scenario"] = per
        return out
    m = fd_modules()
    fe, fb = m["fe"], m["fb"]
    check_prepared(P, fidelity)
    root, rv2 = roots(P)
    with _gate(reduced_gate):
        fd_rec = bool(record and telemetry == "fd" and recorder is None)
        r = _fd_evaluate(fidelity)(gains, struct, scs, P.aircraft, fidelity=fidelity, root=root, root_v2=rv2, profile=P,
                                   sim=sim, record=fd_rec, blas_threads=1)
    if r["model_version"] != mv:
        raise RuntimeError(f"FD model_version {r['model_version']} != {mv}")
    fail_cost = float(2 * P.fail_base)
    if r["per_scenario"]:
        per = [{k: e[k] for k in PER_KEYS if k in e} for e in r["per_scenario"]]
    else:   # margin-gate fail: not flown; one aligned entry per scenario (mean = FD's cost)
        per = [{"cost": float(r["cost"]), "status": r["status"], "t_end": 0.0, "not_flown": True} for _ in scs]
    for e in per:
        e["fidelity"], e["model_version"] = fidelity, mv
    mg = r.get("margins") or {}
    out = {"cost": float(r["cost"]), "status": r["status"], "terms": dict(r["terms"]),
           "terms_available": list(r["terms_available"]), "fidelity": fidelity, "model_version": mv,
           "feasible": r["status"] == "ok", "feasibility_fidelity": fidelity,
           "margins_fidelity": r.get("margins_fidelity"), "margin_gate": r.get("margin_gate"), "reduced_gate": reduced_gate,
           "margins": {k: mg.get(k) for k in ("flutter_margin", "div_margin", "reversal_margin")},
           "mass_total_frac": (r.get("mass") or {}).get("total_frac"), "mass_lb": r.get("mass"), "per_scenario": per,
           "wall_s": float(sum(e.get("wall_s", 0.0) for e in r["per_scenario"]))}
    if P.flex_mass_credit_clip and r["per_scenario"] and r.get("mass"):
        apply_mass_credit_clip(out, r["mass"], P.flex_mass_credit_clip, fb.StructWeightsV2().w_mass)
    if r.get("projection"):
        out["projection"] = {k: v for k, v in r["projection"].items() if not k.startswith("_")}
    if fidelity == A1:   # FD's A1 extras (absent at reduced / full, whose results are unchanged)
        out["structural_model"] = r.get("structural_model")
        sz = r.get("sizing") or {}
        out["tip_bm"] = {"method": sz.get("tip_bm_method"),
                         "strip_discrete_term": (sz.get("tip_bm_strip_discrete") or {}).get("term")}
    if fd_rec and r["per_scenario"]:
        out["trajectories"] = [t.get("trajectory") for t in r.get("telemetry") or []]
    elif (recorder is not None or record) and r["per_scenario"]:
        # telemetry flight with FD's own hook (bit-identical physics) wrapped for Sim Bridge flex_state; checked against
        # FD's per-scenario sim_cost
        wts = dataclasses.replace(fb.StructWeightsV2(), fail_cost=fail_cost)
        obj = _struct_obj(fe, fb, struct, P.aircraft, fidelity, rv2)
        Pf = _flex_profile(P, fidelity)
        trajs, match = [], []
        for sc, e in zip(scs, r["per_scenario"]):
            hk = SBHook(_fd_hook(fe, fidelity)(fidelity, P.aircraft, obj, wts, sim.DT, root, rv2, mv), obj, fidelity, mv)
            t = sim.simulate(gains, sc, Pf, record=record, sample_hz=sample_hz, recorder=recorder, flex=hk)
            t.pop("_flex", None)
            match.append(t["cost"] == e["sim_cost"])
            if record:
                trajs.append(t.get("trajectory"))
        out["telemetry_check"] = {"sim_cost_bit_identical": all(match), "per_scenario": match}
        if record:
            out["trajectories"] = trajs
    return out


def aggregate(per: Sequence[Dict]) -> Dict:
    """Rigid per-scenario results -> {cost, status, feasible, feasibility_fidelity, terms (FD's 16 keys), terms_available,
    fidelity, model_version}: exactly the keys/values of the Phase-1 rows (rigid rows unchanged)."""
    costs = [p["cost"] for p in per]
    fids = {p.get("fidelity", "rigid") for p in per}
    mvs = {p.get("model_version") for p in per}
    if len(fids) != 1 or len(mvs) != 1:
        raise ValueError(f"aggregate needs one fidelity/model_version, got {fids} / {mvs}")
    fid = fids.pop()
    status = next((p["status"] for p in per if p["status"] != "ok"), "ok")
    ok = [p for p in per if p["status"] == "ok"]

    def mean_of(key):
        vals = [p.get(key) for p in ok]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return float(np.mean(vals)) if vals else 0.0
    terms = {k: 0.0 for k in TERM_KEYS}
    avail = ["track", "effort"]
    terms["track"], terms["effort"] = mean_of("track"), mean_of("effort")
    for key, src in (("comfort", "comfort"), ("heading", "heading_rms"), ("hold", "hold_osc")):
        if any(src in p for p in per):
            terms[key] = mean_of(src)
            avail.append(key)
    return {"cost": float(np.mean(costs)), "status": status, "feasible": status == "ok", "feasibility_fidelity": fid,
            "terms": terms, "terms_available": avail, "fidelity": fid, "model_version": per[0].get("model_version")}


# FD's 24 keys (flexeval.TERM_KEYS, INTERFACE_v2 section 12 + P2.5 J_wing_tip_bm_limit);
# kept literal so rigid-only use never imports FD (test_model_version_formats_and_uniform_terms pins equality)
TERM_KEYS = ("track", "effort", "comfort", "heading", "hold", "J_flutter_margin", "J_div_margin", "J_mass", "J_bm_rms",
             "J_bm_peak", "J_tip", "J_twist", "J_reversal_margin", "J_tail_bm_peak", "J_fus_bm_peak", "J_smooth",
             "J_wing_bm_limit", "J_wing_torque_limit", "J_wing_ip_limit", "J_wing_tip_bm_limit", "J_tail_bm_limit",
             "J_fus_bm_limit", "J_wing_torque_peak", "J_wing_ip_peak")


def evaluate_scenario(profile_d: Dict, gains: Dict[str, float], struct, sc_d: Dict, fidelity: str = "rigid",
                      recorder=None, sample_hz: float = 30.0, record: bool = False,
                      reduced_gate: Optional[float] = None, telemetry: str = "sb") -> Dict:
    """One genome x one scenario. rigid: exactly sim.simulate's result (+ fidelity / model_version), bit-identical to
    Phase 1. flex: FD's per-scenario entry (scenarios are independent in flexeval, so this equals that scenario's entry
    of the all-scenario call bit for bit) + the genome-level fields; 'trajectory' when record=True."""
    if fidelity == "rigid":
        P = sim.Profile.from_dict(profile_d)
        r = sim.simulate(gains, sim.Scenario.from_dict(sc_d), P, record=record, sample_hz=sample_hz, recorder=recorder)
        r["fidelity"], r["model_version"] = fidelity, model_version(profile_d, fidelity)
        return r
    g = evaluate_genome(profile_d, gains, struct, [sc_d], fidelity, reduced_gate, recorder=recorder, record=record,
                        sample_hz=sample_hz, telemetry=telemetry)
    out = dict(g["per_scenario"][0])
    for k in ("feasible", "feasibility_fidelity", "margins", "margins_fidelity", "margin_gate", "reduced_gate",
              "terms", "terms_available", "mass_total_frac", "telemetry_check"):
        if k in g:
            out["genome_" + k if k in ("terms", "terms_available") else k] = g[k]
    out["genome_status"] = g["status"]
    if g.get("trajectories"):
        out["trajectory"] = g["trajectories"][0]
    return out


# ----------------------------------------------------------------------------- the agreed hook (pass-through)
def evaluate(gains: Dict[str, float], struct_genome, scenarios, model, *, fidelity: str, root: Optional[str] = None,
             dt: Optional[float] = None, record: bool = False, profile_d: Optional[Dict] = None,
             reduced_gate: Optional[float] = None) -> Dict:
    """evaluate(gains, struct_genome, scenarios, model, *, fidelity, root, dt, record) as agreed with FD (INTERFACE_v2 §7);
    `model` = aircraft name (profile from configs/phase1_hdg.json unless profile_d is given) or a resolved profile dict."""
    if dt is not None and dt != sim.DT:
        raise ValueError(f"dt must be sim.DT = {sim.DT}")
    if profile_d is None:
        if isinstance(model, dict):
            profile_d = model
        else:
            from . import batch
            cfg = batch.resolve_config(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    "configs", "phase1_hdg.json"))), "hook")
            profile_d = next(a for a in cfg["aircraft"] if a["name"] == model)["resolved_profile"]
    if root is not None:
        profile_d = dict(profile_d, aircraft_root=root)
    name = profile_d.get("aircraft", model)
    if reduced_gate is None:
        reduced_gate = per_aircraft(name)["reduced_gate"]
    scs_d = [s.to_dict() if hasattr(s, "to_dict") else s for s in scenarios]
    return evaluate_genome(profile_d, gains, struct_from(struct_genome), scs_d, fidelity, reduced_gate, record=record)
