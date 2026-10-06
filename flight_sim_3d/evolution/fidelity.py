"""Structural-fidelity adapter: the ONLY place evolution/ touches Flight Dynamics' flex code.

    rigid    JSBSim rigid body, our sim.simulate unchanged (bit-identical to the Phase-1 runs)
    reduced  FD flexwing v1, n_bend = 1 (+ 1st torsion), two-way coupling, 2 Newmark substeps
    full     FD's v2 when it lands; until then FD flexwing v1, n_bend = 2 (+ torsion), two-way, 2 substeps,
             labelled "full(v1)" and versioned "full:flexv1:<sha8>"

FD's modules (flight-dynamics/flexwing.py, coupled_sim.py) are IMPORTED, never copied, with bytecode writing
disabled so nothing lands in their folder; ensure_root / prepare_aircraft (which write into jsbsim_root) are never
called. Models must already be FD-prepared (external_reactions flexwing_F + flexwing_meta.json).

Agreed hook (FD <-> evolution, 2026-10-06):

    evaluate(gains, struct_genome, scenarios, model, *, fidelity, root=None, dt=None, record=False)
      -> {cost, terms, terms_available, status, telemetry?, model_version, fidelity, margins_fidelity, ...}

* model_version: "rigid:jsbsim<ver>:<sha8>" | "reduced:flexv1:<sha8>" | "full:flexv1:<sha8>" (v2: "full:flexv2:...").
  sha8 = sha256 over the structural code (FD flexwing.py + coupled_sim.py; nothing for rigid), the resolved
  baseline parameters (WingParams for this aircraft, coupler mode / n_bend / substeps, StructWeights, the margin
  gate) and every file of the prepared aircraft directory that is loaded. Per-genome struct genes are inputs, not
  part of the version. The cache key is built from the *returned* fidelity + model_version.
* Every fidelity returns the same term keys (TERM_KEYS). Inapplicable terms are 0.0 and are left out of
  terms_available; never NaN. Ranking uses the total cost only.
* Feasibility gate: full fails a genome with flutter/divergence margin < 1.0 (FD's rule, not flown, cost =
  fail_cost); reduced screens with a tolerance, failing only below 0.9. Every verdict carries margins_fidelity.
* Per-scenario cost = rigid altitude-hold cost + response terms J_bm_rms/J_bm_peak/J_tip/J_twist (fail_cost on
  structural_ultimate) + pre terms J_flutter_margin + J_div_margin + J_mass; fail_cost (not flown) if the margin
  gate fails. Genome cost = float(np.mean(per-scenario costs)) at every fidelity. Mathematically FD's
  coupled_sim.evaluate_flex composition (which adds the pre terms once to the mean); last bits may differ.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import genome as genome_mod
from . import sim

FD_DIR = os.environ.get("EVOLUTION_FD_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flight-dynamics"))
FIDELITIES = ("rigid", "reduced", "full")
PRE_TERMS = ("J_flutter_margin", "J_div_margin", "J_mass")
RESP_TERMS = ("J_bm_rms", "J_bm_peak", "J_tip", "J_twist")
TERM_KEYS = ("track", "effort", "comfort", "heading") + PRE_TERMS + RESP_TERMS   # heading = RMS(e_psi)/hdg_rms_ref_deg

# FD v1 stand-ins. "full" switches to FD's v2 when it exists (see _full_impl()).
SPECS: Dict[str, Optional[Dict]] = {
    "rigid": None,
    "reduced": {"impl": "flexv1", "label": "reduced", "mode": "twoway", "n_bend": 1, "substeps": 2, "reference": "trim",
                "margin_fail_below": 0.9},
    "full": {"impl": "flexv1", "label": "full(v1)", "mode": "twoway", "n_bend": 2, "substeps": 2, "reference": "trim",
             "margin_fail_below": 1.0},
}

_FD: Dict[str, object] = {}


class FidelityUnavailable(Exception):
    pass


@contextlib.contextmanager
def _no_bytecode():
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = prev


def fd_modules():
    """(flexwing, coupled_sim) imported from FD's folder (read-only; no __pycache__ written there)."""
    if not _FD:
        if not os.path.isdir(FD_DIR):
            raise FidelityUnavailable(f"Flight Dynamics folder not found: {FD_DIR}")
        with _no_bytecode():
            if FD_DIR not in sys.path:
                sys.path.insert(0, FD_DIR)
            import flexwing  # noqa: E402
            import coupled_sim  # noqa: E402
        _FD.update(fw=flexwing, cs=coupled_sim)
    return _FD["fw"], _FD["cs"]


def fd_code_sha() -> str:
    h = hashlib.sha256()
    for f in ("flexwing.py", "coupled_sim.py"):
        with open(os.path.join(FD_DIR, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read() + b"\0")
    return h.hexdigest()


def label(fidelity: str) -> str:
    spec = SPECS[fidelity]
    return "rigid" if spec is None else spec["label"]


# ----------------------------------------------------------------------------- struct genes
def struct_schema() -> List[genome_mod.Gene]:
    """FD's Phase-1 STRUCT_SCHEMA as evolution genes (same decode formula: log / linear on [0, 1])."""
    fw, _ = fd_modules()
    return [genome_mod.Gene(g.name, g.min, g.max, kind=g.kind, units=g.units, doc=g.doc) for g in fw.STRUCT_SCHEMA]


def struct_from(struct_genome) -> Optional[Dict[str, float]]:
    """None | {gene: value} (decoded) | sequence of normalized values in STRUCT_SCHEMA order -> {gene: value}."""
    if struct_genome is None:
        return None
    if isinstance(struct_genome, dict):
        return {k: float(v) for k, v in struct_genome.items()}
    fw, _ = fd_modules()
    return fw.decode_struct(struct_genome)


def project(struct: Optional[Dict[str, float]], fidelity: str, model: str):
    """reduced: v2 struct genomes go through FD's project_to_reduced(struct_genome, model) when FD provides it.
    The v1 genes (STRUCT_SCHEMA) are already valid for the 1-bending-mode wing, so without it this is identity."""
    if struct is None or fidelity != "reduced":
        return struct, None
    fw, _ = fd_modules()
    fn = getattr(fw, "project_to_reduced", None)
    if fn is None:
        return struct, "identity (v1 genes)"
    return {k: float(v) for k, v in fn(struct, model).items()}, "fd.project_to_reduced"


# ----------------------------------------------------------------------------- wing / model version
_METRICS: Dict[tuple, Dict] = {}


def _metrics(profile: sim.Profile) -> Dict:
    key = (profile.aircraft, profile.aircraft_root)
    if key not in _METRICS:
        fdm = sim._new_fdm(profile)
        _METRICS[key] = {"span_ft": fdm["metrics/bw-ft"], "area_ft2": fdm["metrics/Sw-sqft"],
                         "empty_wt_lb": fdm["inertia/empty-weight-lbs"]}
        del fdm
    return _METRICS[key]


def _check_prepared(profile: sim.Profile):
    root = profile.aircraft_root
    if root is None:
        raise FidelityUnavailable(f"{profile.aircraft}: flex fidelities need an FD-prepared model (aircraft_root = FD jsbsim_root)")
    xml = os.path.join(sim.model_dir(profile.aircraft, root), profile.aircraft + ".xml")
    if not os.path.exists(xml) or "flexwing_F" not in open(xml, encoding="utf-8", errors="replace").read():
        raise FidelityUnavailable(f"{profile.aircraft}: {xml} has no flexwing external_reactions (not FD-prepared)")


def build_wing(profile: sim.Profile, fidelity: str, struct: Optional[Dict[str, float]] = None):
    fw, _ = fd_modules()
    spec = SPECS[fidelity]
    m = _metrics(profile)
    ov = {"n_bend": spec["n_bend"]}
    if struct:
        ov.update(fw.genes_to_overrides(struct))
    return fw.FlexWing(fw.params_for(profile.aircraft, m["span_ft"], m["area_ft2"], m["empty_wt_lb"], **ov))


def _weights(profile: sim.Profile):
    fw, _ = fd_modules()
    return fw.StructWeights(fail_cost=float(2 * profile.fail_base))


_MV: Dict[tuple, str] = {}


def model_version(profile_d: Dict, fidelity: str) -> str:
    key = (json.dumps(profile_d, sort_keys=True), fidelity)
    if key in _MV:
        return _MV[key]
    P = sim.Profile.from_dict(profile_d)
    files = sim.model_files_sha(P.aircraft, P.aircraft_root)
    spec = SPECS[fidelity]
    if spec is None:
        payload = {"model_files": files}
        mv = f"rigid:jsbsim{_jsbsim_version()}:" + _sha8(payload)
    else:
        fw, _ = fd_modules()
        _check_prepared(P)
        wing = build_wing(P, fidelity)
        payload = {"fd_code": fd_code_sha(), "wing_params": _jsonable(dataclasses.asdict(wing.p)),
                   "coupler": {k: spec[k] for k in ("mode", "n_bend", "substeps", "reference")},
                   "margin_fail_below": spec["margin_fail_below"], "weights": dataclasses.asdict(_weights(P)),
                   "model_files": files}
        mv = f"{fidelity}:{spec['impl']}:" + _sha8(payload)
    _MV[key] = mv
    return mv


def _jsonable(d):
    return json.loads(json.dumps(d, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))


def _sha8(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]


def _jsbsim_version() -> str:
    import jsbsim
    return jsbsim.__version__


# ----------------------------------------------------------------------------- flex_state (recorder payload)
N_NODES = 9          # structure nodes per semi-wing, xi = 0 (beam root) .. 1 (tip), evenly spaced
FLEX_STATE_SCHEMA = "evolution-flex-state/1"
TWIST_DOC = ("twist [rad]: right-hand rotation about the component axis node0 -> nodeN (root -> tip). wingR: + = leading "
             "edge up; wingL: + = leading edge DOWN. FD flexwing v1 raw twist (PsiT @ eta, tip_twist_deg_R/L) is + = "
             "leading edge up / nose-up (increases local alpha) on BOTH semi-wings, so wingR.twist = +theta_R, "
             "wingL.twist = -theta_L.")


def structure_geometry(wing, fdm, n_nodes: int = N_NODES) -> Dict:
    """Undeformed structure axes (FD v1 has one beam per semi-wing along its elastic axis) in body FRD metres,
    origin at the CG (as in the trajectory frame), plus the mode-shape matrices at the nodes.

    Node i at xi_i = i/(n-1): outboard y = y0 + xi*L (FD's beam from the side of the fuselage, y0 = root_frac*s, to
    the tip). x: the elastic axis is placed with FD's planform model (quarter chord swept by sweep_deg through the
    MAC quarter chord, which is taken to be JSBSim's AERORP; EA at x_ea of the local chord), so
    x_aft_of_AERORP = (y - y_mac)*tan(sweep) + (x_ea - 0.25)*c(y). z = AERORP z (no dihedral in v1).
    Approximate geometry for display; the physics does not use it."""
    fw, _ = fd_modules()
    p = wing.p
    nb = p.n_bend
    xi = np.linspace(0.0, 1.0, n_nodes)
    y = wing.y0 + xi * wing.L
    c = np.asarray(wing.chord(y), float)
    y_mac = wing.s / 3 * (1 + 2 * p.taper) / (1 + p.taper)
    aft = (y - y_mac) * math.tan(wing.lam) + (p.x_ea - 0.25) * c
    # structural frame (in, x aft, z up) -> body FRD (ft, x fwd, z down), relative to the CG
    x_rp = -(fdm["metrics/aero-rp-x-in"] - fdm["inertia/cg-x-in"]) / 12.0
    z_rp = -(fdm["metrics/aero-rp-z-in"] - fdm["inertia/cg-z-in"]) / 12.0
    ft = sim.FT
    comps = []
    for name, sgn in (("wingR", 1.0), ("wingL", -1.0)):
        nodes = [[round(float((x_rp - a) * ft), 6), round(float(sgn * yy * ft), 6), round(float(z_rp * ft), 6)]
                 for a, yy in zip(aft, y)]
        comps.append({"name": name, "axis_nodes_body_m": nodes, "dof": ["dz", "dy", "twist"],
                      "node_span_frac": [float(v) for v in xi]})
    B = np.array([[float(fw._bend(k, x)) for k in range(nb)] for x in xi])      # w(xi) = B @ eta[:nb]   (ft, + up)
    T = np.array([float(fw._tors(x)) for x in xi])                              # theta(xi) = T * eta[nb] (rad, + LE up)
    structure = {"schema": FLEX_STATE_SCHEMA, "synthetic": False, "source": "FD flexwing v1 modal state (adapter: "
                 "evolution/fidelity.py)", "components": comps,
                 "units": {"axis_nodes_body_m": "m, body FRD (x fwd, y right, z down), origin CG", "dz": "m (body z, + down)",
                           "dy": "m (body y, + right)", "twist": "rad"},
                 "dz_doc": "elastic deflection of the elastic axis relative to the undeformed (jig) shape, including the "
                           "1-g trim deflection; = -w (FD w is + up)",
                 "dy_doc": "0.0 in FD v1 (no in-plane / fore-aft modes)", "twist_doc": TWIST_DOC}
    return {"structure": structure, "B": B, "T": T, "nb": nb}


class FlexState:
    """flex_state handed to recorders (schema evolution-flex-state/1). Valid during the recorder call.

      .structure    static dict: {"components": [{"name": "wingR"|"wingL", "axis_nodes_body_m": [[x,y,z] x 9],
                    "dof": ["dz","dy","twist"], "node_span_frac": [...]}], units, sign docs}
      .channels()   {"wingR.dz.0": m, ..., "wingR.dy.i": m, "wingR.twist.i": rad, "wingL...."} (Sim Bridge
                    <component>.<dof>.<node_idx>, Sim Bridge sign convention; see TWIST_DOC)
      .eta          FD modal coordinates, array (2, n_modes): row 0 right wing, row 1 left wing
                    [bending_1 (ft at tip), (bending_2), torsion (rad at tip)]
      .raw          FD coupler.last: tip_w_ft_R/L, tip_twist_deg_R/L (FD convention), root_bm_lbft_R/L, dL_lbf, ...
      .as_dict()    all of the above as plain JSON-able data
    The mapping FD -> Sim Bridge is fd_to_structure_channels() (one function)."""
    __slots__ = ("structure", "eta", "raw", "_geom", "fidelity", "model_version")
    schema = FLEX_STATE_SCHEMA

    def __init__(self, geom, eta, raw, fidelity, model_version):
        self._geom, self.structure = geom, geom["structure"]
        self.eta, self.raw = eta, raw
        self.fidelity, self.model_version = fidelity, model_version

    def channels(self) -> Dict[str, float]:
        return fd_to_structure_channels(self.eta, self._geom)

    def as_dict(self) -> Dict:
        return {"schema": self.schema, "fidelity": self.fidelity, "model_version": self.model_version,
                "structure": self.structure, "channels": self.channels(),
                "modal_eta": {"wingR": [float(v) for v in self.eta[0]], "wingL": [float(v) for v in self.eta[1]]},
                "fd_raw": {k: float(v) for k, v in self.raw.items()}}


def fd_to_structure_channels(eta, geom) -> Dict[str, float]:
    """THE FD v1 -> Sim Bridge mapping. eta: (2, n_modes) FD modal coordinates (row 0 right, row 1 left wing).
    dz = -w (FD w + up -> body z + down), dy = 0, twist: wingR = +theta_R, wingL = -theta_L (see TWIST_DOC)."""
    B, T, nb = geom["B"], geom["T"], geom["nb"]
    ft = sim.FT
    out: Dict[str, float] = {}
    for side, name, tsign in ((0, "wingR", 1.0), (1, "wingL", -1.0)):
        e = np.asarray(eta[side], float)
        w = B @ e[:nb]
        th = T * e[nb]
        for i, v in enumerate(w):
            out[f"{name}.dz.{i}"] = float(-v * ft)
        for i in range(len(w)):
            out[f"{name}.dy.{i}"] = 0.0
        for i, v in enumerate(th):
            out[f"{name}.twist.{i}"] = float(tsign * v)
    return out


# ----------------------------------------------------------------------------- hook for sim.simulate
class FlexHook:
    """sim.simulate's `flex` object: attach() before trim, wrap() after trim, state() per recorder call, finish().
    Uses FD's FlexCoupler / FlexFDM / apply_wing_mass / response_terms as they are."""

    def __init__(self, profile: sim.Profile, wing, spec: Dict, weights, fidelity: str = "", model_version: str = ""):
        self.fw, self.cs = fd_modules()
        self.P, self.wing, self.spec, self.weights = profile, wing, spec, weights
        self.fidelity, self.model_version = fidelity, model_version
        self.coupler = None
        self.px = None
        self.geom = None
        self._eta0 = None
        self.delta_mass_lb = 0.0

    def attach(self, fdm):
        # reads flexwing_meta.json from FD's root (read-only) and sets the two wing point masses
        self.delta_mass_lb = self.fw.apply_wing_mass(fdm, self.P.aircraft, sim.abs_root(self.P.aircraft_root), self.wing)

    def wrap(self, fdm):
        self.coupler = self.fw.FlexCoupler(self.wing, mode=self.spec["mode"], substeps=self.spec["substeps"],
                                           reference=self.spec["reference"])
        self.px = self.cs.FlexFDM(fdm, self.coupler, sim.DT, record=True)
        self._fdm = fdm
        return self.px

    def state(self) -> FlexState:
        """Only called when a recorder is attached. Before the first step (t = 0) the coupler is not initialised yet;
        the trim equilibrium is then computed on a throw-away ONE-WAY coupler from read-only property reads (same
        FD initialize(), no writes; the two-way coupler computes the same state at its first step)."""
        if self.geom is None:
            self.geom = structure_geometry(self.wing, self._fdm)
        if self.px.started:
            return FlexState(self.geom, self.coupler.eta, self.coupler.last, self.fidelity, self.model_version)
        if self._eta0 is None:
            probe = self.fw.FlexCoupler(self.wing, mode="oneway", substeps=self.spec["substeps"],
                                        reference=self.spec["reference"])
            probe.initialize(sim.ReadOnlyFDM(self._fdm))
            self._eta0 = (probe.eta.copy(), dict(probe.last))
        return FlexState(self.geom, self._eta0[0], self._eta0[1], self.fidelity, self.model_version)

    def finish(self, status: str) -> Dict:
        if status != "ok" or self.px is None:
            return {"terms": None, "fail": None}
        post = self.fw.response_terms(self.px.history(), self.wing, self.px.m_root_1g, self.weights)
        return {"terms": post["terms"], "fail": post["fail"], "bm_rms": post["bm_rms"], "bm_peak": post["bm_peak"],
                "bm_allow": post["bm_allow"], "tip_max_ft": post["tip_max_ft"], "twist_max_deg": post["twist_max_deg"],
                "m_root_1g": self.px.m_root_1g}


# ----------------------------------------------------------------------------- per-scenario evaluation
def evaluate_scenario(profile_d: Dict, gains: Dict[str, float], struct: Optional[Dict[str, float]], sc_d: Dict,
                      fidelity: str = "rigid", recorder=None, sample_hz: float = 30.0, record: bool = False) -> Dict:
    """One genome x one scenario at one fidelity. JSON-able result (what the batch caches per scenario).

    rigid: exactly sim.simulate's result (+ fidelity / model_version). flex: 'cost' = rigid cost + response
    terms (fail_cost on structural_ultimate), 'sim_cost' = the altitude-hold part, 'pre' = margin gate + terms,
    'struct' = response terms. A genome failing the margin gate is not flown (cost = fail_cost)."""
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}")
    P = sim.Profile.from_dict(profile_d)
    sc = sim.Scenario.from_dict(sc_d)
    mv = model_version(profile_d, fidelity)
    spec = SPECS[fidelity]
    if spec is None:
        r = sim.simulate(gains, sc, P, record=record, sample_hz=sample_hz, recorder=recorder)
        r["fidelity"], r["model_version"] = fidelity, mv
        return r
    fw, _ = fd_modules()
    _check_prepared(P)
    struct_p, projection = project(struct, fidelity, P.aircraft)
    wing = build_wing(P, fidelity, struct_p)
    wts = _weights(P)
    pre = fw.margin_terms(wing, wts)
    gate = spec["margin_fail_below"]
    mg = pre["margins"]
    fail = None
    for key in ("flutter_margin", "div_margin"):
        if mg[key] < gate:
            fail = fail or key.replace("_margin", "")
    pre_out = {"terms": {k: float(pre["terms"][k]) for k in PRE_TERMS}, "fail": fail, "margins_fidelity": fidelity,
               "margin_fail_below": gate, "delta_wing_mass_lb": float(pre["terms"]["delta_wing_mass_lb"]),
               "margins": {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                           for k, v in mg.items() if k != "f_modes_hz"},
               "f_modes_hz": [float(x) for x in mg.get("f_modes_hz", [])]}
    if projection:
        pre_out["projection"] = projection
    if fail:
        return {"cost": float(wts.fail_cost), "sim_cost": None, "status": fail, "t_end": 0.0,
                "track": None, "effort": None, "wall_s": 0.0, "pre": pre_out, "struct": None,
                "fidelity": fidelity, "model_version": mv}
    hook = FlexHook(P, wing, spec, wts, fidelity, mv)
    r = sim.simulate(gains, sc, P, record=record, sample_hz=sample_hz, recorder=recorder, flex=hook)
    post = r.pop("_flex", None) or {"terms": None, "fail": None}
    r["sim_cost"] = r["cost"]
    r["pre"] = pre_out
    r["delta_wing_mass_lb_applied"] = float(hook.delta_mass_lb)
    if r["status"] == "ok":
        r["struct"] = {k: float(v) for k, v in post["terms"].items()}
        r.update({k: float(post[k]) for k in ("bm_rms", "bm_peak", "bm_allow", "tip_max_ft", "twist_max_deg", "m_root_1g")})
        if post["fail"]:
            r["status"] = post["fail"]
            r["cost"] = float(wts.fail_cost)
        else:
            r["cost"] = r["cost"] + sum(post["terms"].values())   # FD: r["cost"] + sum(post["terms"].values())
    else:
        r["struct"] = None
    # pre terms folded into every scenario's cost, so cost = mean(per_scenario_cost) at every fidelity (FD adds the
    # same sum once to the mean: mathematically identical, may differ in the last bit)
    r["cost"] = r["cost"] + sum(pre_out["terms"].values())
    r["fidelity"], r["model_version"] = fidelity, mv
    return r


# ----------------------------------------------------------------------------- aggregation (single definition)
def aggregate(per: Sequence[Dict]) -> Dict:
    """Per-scenario results (same genome, same fidelity) -> {cost, status, terms, terms_available, feasible, ...}.

    cost = float(np.mean(per-scenario costs)) at every fidelity (rigid: exactly what the Phase-1 batch computed; flex:
    per-scenario costs already contain the pre terms, a margin-gate fail is fail_cost in every scenario)."""
    costs = [p["cost"] for p in per]
    fids = {p.get("fidelity", "rigid") for p in per}
    mvs = {p.get("model_version") for p in per}
    if len(fids) != 1 or len(mvs) != 1:
        raise ValueError(f"aggregate needs one fidelity/model_version, got {fids} / {mvs}")
    fid = fids.pop()
    pre = per[0].get("pre")
    cost = float(np.mean(costs))
    status = next((p["status"] for p in per if p["status"] != "ok"), "ok")
    ok = [p for p in per if p["status"] == "ok"]

    def mean_of(get):
        vals = [get(p) for p in ok]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return float(np.mean(vals)) if vals else 0.0

    terms = {k: 0.0 for k in TERM_KEYS}
    avail = ["track", "effort"]
    terms["track"] = mean_of(lambda p: p.get("track"))
    terms["effort"] = mean_of(lambda p: p.get("effort"))
    if any("comfort" in p for p in per):
        terms["comfort"] = mean_of(lambda p: p.get("comfort"))
        avail.append("comfort")
    if any("heading_rms" in p for p in per):
        terms["heading"] = mean_of(lambda p: p.get("heading_rms"))
        avail.append("heading")
    if pre is not None:
        for k in PRE_TERMS:
            terms[k] = float(pre["terms"][k])
        for k in RESP_TERMS:
            terms[k] = mean_of(lambda p, k=k: (p.get("struct") or {}).get(k))
        avail += list(PRE_TERMS) + list(RESP_TERMS)
    out = {"cost": cost, "status": status, "feasible": status == "ok", "feasibility_fidelity": fid,
           "terms": terms, "terms_available": avail, "fidelity": fid, "model_version": per[0].get("model_version")}
    if pre is not None:
        out["margins_fidelity"] = pre["margins_fidelity"]
        out["margins"] = {k: pre["margins"][k] for k in ("flutter_margin", "div_margin") if k in pre["margins"]}
    return out


# ----------------------------------------------------------------------------- the agreed FD hook
def evaluate(gains: Dict[str, float], struct_genome, scenarios, model, *, fidelity: str, root: Optional[str] = None,
             dt: Optional[float] = None, record=False) -> Dict:
    """Agreed hook. model: aircraft name (Profile defaults for it) or a resolved profile dict / sim.Profile;
    root overrides the profile's aircraft_root; dt must be sim.DT (1/120 s); scenarios: sim.Scenario objects or
    dicts; record: False | True (telemetry = per-scenario trajectory incl. flex channels) | a recorder callable."""
    if dt is not None and abs(dt - sim.DT) > 1e-15:
        raise ValueError(f"dt must be {sim.DT} (sim.DT); got {dt}")
    if isinstance(model, str):
        prof_d = sim.Profile(aircraft=model).to_dict()
    elif isinstance(model, sim.Profile):
        prof_d = model.to_dict()
    else:
        prof_d = dict(model)
    if root is not None:
        prof_d["aircraft_root"] = root
    struct = struct_from(struct_genome)
    per, tele = [], []
    for sc in scenarios:
        sc_d = sc.to_dict() if isinstance(sc, sim.Scenario) else dict(sc)
        rec = record if callable(record) else None
        r = evaluate_scenario(prof_d, gains, struct, sc_d, fidelity, recorder=rec, record=record is True)
        if "trajectory" in r:
            tele.append(r.pop("trajectory"))
        per.append(r)
    out = aggregate(per)
    out["per_scenario"] = per
    if record is True:
        out["telemetry"] = tele
    return out
