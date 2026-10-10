"""flexeval.py -- multi-fidelity evaluation hook for the Evolution Runner (flex side + reference wrapper).

    evaluate(gains, struct_genome, scenarios, model, *, fidelity, root, dt=1/120, record=False)
        -> {cost, terms, terms_available, status, model_version, fidelity, margins, margins_fidelity, per_scenario,
            telemetry (record=True only)}

fidelity
  'rigid'   no coupler: evolution/sim.py simulate() exactly as the Runner calls it (bit-identical; cost = np.mean of the
            per-scenario costs, as evolution/batch.py)
  'reduced' v1 FlexWing with n_bend=1 + torsion (2 modes per semi-wing), v1 coupler, substeps 2 (the v1 GA path). A v2
            genome is mapped by project_to_reduced() (strain-energy / kinetic-energy weighted spanwise averages).
            Margins are screening values: hard fail only below 0.9 (margins_fidelity='reduced').
  'full'    flex v2 (flexbody.py): 25-DOF wings + empennage + fuselage, prepared root <root>_v2, gate 1.0.

Cost terms: TERM_KEYS (same keys at every fidelity, aligned with evolution/fidelity.py: track/effort/comfort/heading/hold +
structural J_*); inapplicable terms are 0.0 and not in terms_available. cost = mean of per-scenario costs.

Nothing in evolution/ or genome/ is modified: evolution/sim.py is loaded as a PRIVATE module instance (importlib, not
the 'evolution' package). Flex fidelities use the Runner's own structural-hook protocol (simulate(flex=FlexHookV2):
attach before IC/trim, wrap after trim, finish after the flight); with an older sim.py without that parameter the
private instance's _new_fdm is swapped for the duration of the call instead (same physics: trim never steps the proxy).
See INTERFACE_v2.md, section "Fidelity contract".
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

import flexwing as fw
import flexbody as fb
import coupled_sim as cs

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
EVOLUTION_SIM = os.path.join(TEAM, "evolution", "sim.py")
PHASE1_CONFIG = os.path.join(TEAM, "evolution", "configs", "phase1.json")
FIDELITIES = ("rigid", "reduced", "full")
# Same keys at every fidelity, aligned with evolution/fidelity.py TERM_KEYS (+ the v2-only structural terms).
# Inapplicable terms are 0.0 and left out of terms_available. Ranking uses the total cost only.
RIGID_TERMS = ("track", "effort", "comfort", "heading", "hold")   # hold = Runner v5 hold_osc (0.0 unless w_hold > 0)
# limit-load sizing terms (flexbody.sizing_v2, pre-flight, from the v2 model) at BOTH flex fidelities: the reduced
# fidelity also credits v2 mass (J_mass), so it must carry the same binding allowables
PRE_TERMS = {"reduced": ("J_flutter_margin", "J_div_margin", "J_mass", "J_smooth") + fb.SIZING_TERMS,
             "full": ("J_flutter_margin", "J_div_margin", "J_reversal_margin", "J_mass", "J_smooth") + fb.SIZING_TERMS}
RESP_TERMS = {"reduced": ("J_bm_rms", "J_bm_peak", "J_tip", "J_twist"),
              "full": ("J_bm_rms", "J_bm_peak", "J_tip", "J_twist", "J_tail_bm_peak", "J_fus_bm_peak",
                       "J_wing_torque_peak", "J_wing_ip_peak")}
TERM_KEYS = RIGID_TERMS + ("J_flutter_margin", "J_div_margin", "J_mass", "J_bm_rms", "J_bm_peak", "J_tip", "J_twist",
                           "J_reversal_margin", "J_tail_bm_peak", "J_fus_bm_peak", "J_smooth") + fb.SIZING_TERMS + \
    ("J_wing_torque_peak", "J_wing_ip_peak")
SUBSTEPS = {"reduced": 2, "full": 2}
MARGIN_GATE = {"reduced": 0.9, "full": 1.0}
REDUCED_N_BEND = 1
# code that defines each fidelity's numbers (hashed into model_version); tests monkeypatch this
CODE_FILES = {
    "rigid": (),
    "reduced": (os.path.join(HERE, "flexwing.py"), os.path.join(HERE, "coupled_sim.py"), os.path.join(HERE, "flexbody.py"),
                os.path.join(HERE, "flexeval.py")),
    "full": (os.path.join(HERE, "flexbody.py"), os.path.join(HERE, "flexwing.py"), os.path.join(HERE, "flexeval.py")),
}

_SIM_CACHE: Dict[str, object] = {}


def load_sim(path: str = EVOLUTION_SIM):
    """Private module instance of evolution/sim.py (read-only use; not registered as 'evolution.sim')."""
    if path not in _SIM_CACHE:
        spec = importlib.util.spec_from_file_location("_flexeval_evolution_sim", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod          # dataclasses need the module registered (private name only)
        spec.loader.exec_module(mod)
        _SIM_CACHE[path] = mod
    return _SIM_CACHE[path]


def root_v2_for(root: str) -> str:
    return os.path.normpath(root) + "_v2"


def default_profile(model: str, sim=None, config: str = PHASE1_CONFIG):
    """The Runner's resolved Phase-1 profile for `model`, built like evolution/batch.py resolve_config:
    {**profiles[<aircraft entry>.profile], **<aircraft entry>.overrides, "aircraft": model}."""
    sim = sim or load_sim()
    with open(config) as f:
        cfg = json.load(f)
    ent = next((a if isinstance(a, dict) else {"name": a} for a in cfg["aircraft"]
                if (a.get("name") if isinstance(a, dict) else a) == model), None)
    if ent is None:
        raise ValueError(f"{model!r} not in {config} aircraft list")
    prof = {**cfg["profiles"][ent.get("profile", f"phase1_{model}")], **ent.get("overrides", {}), "aircraft": model}
    return sim.Profile.from_dict(prof)


def _profile_with_root(profile, sim, root: str, model: str):
    d = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)
    d = copy.deepcopy(d)
    if d.get("aircraft", model) != model:
        raise ValueError(f"profile is for aircraft {d.get('aircraft')!r}, evaluate() called with model {model!r}")
    d["aircraft"] = model
    d["aircraft_root"] = root
    return sim.Profile.from_dict(d)


def _as_scenarios(scenarios, sim):
    return [s if hasattr(s, "target") else sim.Scenario.from_dict(s) for s in scenarios]


# ------------------------------------------------------------------------------------------------------------------
# model_version
# ------------------------------------------------------------------------------------------------------------------
def aircraft_files_sha(model: str, root: str) -> str:
    """sha256 over every file in <root>/aircraft/<model> (relative names + bytes), same walk as evolution model_files_sha."""
    src = os.path.join(root, "aircraft", model)
    h = hashlib.sha256()
    if not os.path.isdir(src):
        return "missing"
    for dp, dns, fs in os.walk(src):
        dns.sort()
        for f in sorted(fs):
            fp = os.path.join(dp, f)
            h.update(os.path.relpath(fp, src).encode() + b"\0")
            with open(fp, "rb") as fh:
                h.update(fh.read() + b"\0")
    return h.hexdigest()


def _sha8(parts: Sequence[bytes]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(hashlib.sha256(p).digest())
    return h.hexdigest()[:8]


def _reduced_params(model: str, geom: fb.Geometry) -> Dict:
    p = fw.params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb, n_bend=REDUCED_N_BEND)
    return asdict(p)


def rigid_model_version(model: str, root: str) -> str:
    """Exactly evolution/fidelity.py model_version(profile, 'rigid'): sha8 of {"model_files": sim.model_files_sha}
    (= sha256 over every file of <root>/aircraft/<model>, first 16 hex), so both sides key their caches the same."""
    import jsbsim
    files = aircraft_files_sha(model, root)
    files = files[:16] if files != "missing" else files
    payload = json.dumps({"model_files": files}, sort_keys=True, separators=(",", ":")).encode()
    return f"rigid:jsbsim{jsbsim.__version__}:{hashlib.sha256(payload).hexdigest()[:8]}"


def model_version(fidelity: str, model: str, root: str, root_v2: Optional[str] = None,
                  weights: Optional[fb.StructWeightsV2] = None) -> str:
    """Stable per-fidelity version string; changes whenever the prepared aircraft files, the fidelity's code files,
    its parameters or the structural cost weights change (per-genome genes are inputs, not part of the version):  rigid:jsbsim<ver>:<sha8>, reduced:flexv1:<sha8>, full:flexv2:<sha8>."""
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}")
    if fidelity == "rigid":
        return rigid_model_version(model, root)
    rv2 = root_v2 or root_v2_for(root)
    code = []
    for fp in CODE_FILES[fidelity]:
        with open(fp, "rb") as f:
            code.append(f.read())
    if fidelity == "reduced":
        geom = fb.geometry_for(model, rv2) if os.path.exists(os.path.join(rv2, "aircraft", model, "flexbody_meta.json")) else None
        params = _reduced_params(model, geom) if geom else {"model": model}
        extra = {"substeps": SUBSTEPS["reduced"], "n_bend": REDUCED_N_BEND, "gate": MARGIN_GATE["reduced"], "params": params,
                 "v2_params": fb.v2_params(model, geom) if geom else None, "terms": PRE_TERMS["reduced"] + RESP_TERMS["reduced"],
                 "weights": asdict(weights or fb.StructWeightsV2()),
                 "genes": [asdict(g) for g in fb.gene_schema(True)]}
        return f"reduced:flexv1:{_sha8(code + [json.dumps(extra, sort_keys=True, default=str).encode(), aircraft_files_sha(model, root).encode()])}"
    geom = fb.geometry_for(model, rv2)
    extra = {"substeps": SUBSTEPS["full"], "gate": MARGIN_GATE["full"], "params": fb.v2_params(model, geom),
             "terms": PRE_TERMS["full"] + RESP_TERMS["full"], "weights": asdict(weights or fb.StructWeightsV2()),
             "genes": [asdict(g) for g in fb.gene_schema(True)]}
    return f"full:{fb.MODEL_VERSION_TAG}:{_sha8(code + [json.dumps(extra, sort_keys=True, default=str).encode(), aircraft_files_sha(model, rv2).encode()])}"


# ------------------------------------------------------------------------------------------------------------------
# v2 genome -> v1 (reduced) parameters
# ------------------------------------------------------------------------------------------------------------------
_PROJ_CACHE: Dict = {}


def _projection_weights(model: str, root_v2: str):
    key = (model, root_v2)
    if key not in _PROJ_CACHE:
        base = fb.FlexBodyModel(model, None, root_v2=root_v2)
        s = base.wingR
        b = s.beam
        cls = list(s.all_cls)
        ph_b = s.all_Phi[:, cls.index("b")]
        ph_t = s.all_Phi[:, cls.index("t")]
        n = b.n_el
        g = b._g
        free_pos = {int(d): i for i, d in enumerate(b.free)}

        def nodal(ph, fld):
            v = np.zeros(n + 1)
            for node in range(1, n + 1):
                v[node] = ph[free_pos[g(node, fld)]]
            return v
        wp = nodal(ph_b, "wp")
        th = nodal(ph_t, "t")
        curv = np.diff(wp) / b.h
        twist_rate = np.diff(th) / b.h
        w_mid = b.Nw @ ph_b
        sp = s.sp
        _PROJ_CACHE[key] = {
            "bend": s.EI * curv ** 2,                                  # bending strain energy density, 1st bending mode
            "tors": s.GJ * twist_rate ** 2,                            # torsion strain energy density, 1st torsion mode
            "nsm": (1 - sp.struct_frac) * s.m0 * w_mid ** 2,          # kinetic energy of the non-structural mass, 1st bending
            "n_el": n,
        }
    return _PROJ_CACHE[key]


def _is_asym(struct_genome) -> bool:
    if isinstance(struct_genome, dict):
        return any(k in struct_genome for k in ("wing_asym_ei_delta", "wing_asym_nsm_delta"))
    if struct_genome is None:
        return False
    return np.asarray(struct_genome).size == len(fb.gene_schema(True))


def project_to_reduced(struct_genome, model: str, root_v2: Optional[str] = None) -> Dict:
    """Deterministic v2 genome -> v1 parameters for the reduced fidelity.

    s   = sum(EI_base phi''^2 * ei_mult) / sum(EI_base phi''^2)         (1st bending mode strain energy weights)
    s*r = sum(GJ_base psi'^2 * gj_mult)  / sum(GJ_base psi'^2)          (1st torsion mode strain energy weights)
    nsm = sum(m_ns,base phi^2 * nsm_mult) / sum(m_ns,base phi^2)        (1st bending mode kinetic energy weights)
    zeta passes through. Weights come from the BASELINE v2 wing (genome independent). Asymmetric genomes: mean of L/R.
    Tail / fuselage genes have no v1 counterpart (listed in 'dropped'). No clipping: values outside the v1 gene ranges are
    used as-is and flagged (outside_v1_gene_range).

    Known bias (v2_validation.json 'reduced_projection'): the v1 wing ties its structural mass to the uniform s, while v2
    structural mass follows the (outboard-decreasing) local EI, so the reduced first bending frequency is ~5 % low on
    average (max ~10 %), first torsion ~2-4 % low (max ~8 %), over random v2 genomes. A kinetic-energy mass projection fixes bending but then needs
    non-physical nsm (< 0.2) and moves the error to torsion (tried, rejected; see INTERFACE_v2.md)."""
    rv2 = root_v2 or fb.ROOT_V2
    asym = _is_asym(struct_genome)
    genes = fb.decode_genome_v2(struct_genome, asym)
    W = _projection_weights(model, rv2)
    xi = (np.arange(W["n_el"]) + 0.5) / W["n_el"]
    d = fb.wing_distributions(genes, xi)
    vals = []
    for side in ("R", "L"):
        s = float(W["bend"] @ d["ei_" + side] / W["bend"].sum())
        gj = float(W["tors"] @ d["gj_" + side] / W["tors"].sum())
        nsm = float(W["nsm"] @ d["nsm_" + side] / W["nsm"].sum())
        vals.append((s, gj, nsm))
    s, gj, nsm = (0.5 * (a + b) for a, b in zip(*vals))
    r = gj / s
    out = {"stiffness_scale": s, "torsion_bend_ratio": r, "struct_damping_ratio": genes["struct_damping_ratio"],
           "nonstructural_mass_scale": nsm}
    rng = {k: v[1:] for k, v in fw.GENOME_GENES.items()}
    outside = [k for k, v in out.items() if not (rng[k][0] - 1e-12 <= v <= rng[k][1] + 1e-12)]
    out["_detail"] = {"outside_v1_gene_range": outside, "asymmetry_averaged": asym,
                      "dropped": ["tail_stiffness_scale", "fuselage_stiffness_scale"], "gj_eff": gj}
    return out


def reduced_overrides(proj: Dict) -> Dict:
    ov = fw.tied_stiffness(proj["stiffness_scale"], proj["torsion_bend_ratio"])
    ov.update(zeta=proj["struct_damping_ratio"], nonstruct_scale=proj["nonstructural_mass_scale"], n_bend=REDUCED_N_BEND)
    return ov


def reduced_wing(struct_genome, model: str, root_v2: Optional[str] = None) -> fw.FlexWing:
    rv2 = root_v2 or fb.ROOT_V2
    geom = fb.geometry_for(model, rv2)
    proj = project_to_reduced(struct_genome, model, rv2)
    return fw.FlexWing(fw.params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb, **reduced_overrides(proj)))


# ------------------------------------------------------------------------------------------------------------------
# evaluate
# ------------------------------------------------------------------------------------------------------------------
@contextmanager
def _patched_new_fdm(sim, factory):
    orig = sim._new_fdm
    sim._new_fdm = factory(orig)
    try:
        yield
    finally:
        sim._new_fdm = orig


def _zero_terms() -> Dict[str, float]:
    return {k: 0.0 for k in TERM_KEYS}


def _finite(d: Dict[str, float]) -> Dict[str, float]:
    for k, v in d.items():
        if not math.isfinite(v):
            raise FloatingPointError(f"non-finite cost term {k}={v}")
    return d


def sim_supports_flex_hook(sim) -> bool:
    """True if evolution/sim.py simulate() takes the structural hook (flex=..., attach before trim, wrap after)."""
    try:
        return "flex" in inspect.signature(sim.simulate).parameters
    except (TypeError, ValueError):
        return False


class _FlexStateV2:
    """flex_state for the Runner's recorders: .structure (static description) and .channels() (ordered, every call the
    same keys). Channels are the coupler's diagnostic outputs ('flex.<key>'; units in the key / INTERFACE_v2.md)."""
    __slots__ = ("structure", "_last")
    schema = "fd-flexeval-state/1"

    def __init__(self, structure, last):
        self.structure, self._last = structure, last

    def channels(self) -> Dict[str, float]:
        return {"flex." + k: float(v) for k, v in self._last.items()}


class FlexHookV2:
    """Structural hook in the Runner's protocol (evolution/sim.py simulate(flex=...)):
    attach(fdm) before IC/trim (structural mass changes -> point masses), wrap(fdm) after trim (coupler + proxy whose
    run() = structure step + external_reactions feedback, then the JSBSim step), state() for recorders, finish(status)
    -> post-flight structural terms. fidelity 'reduced' = v1 FlexWing (projected genome), 'full' = flexbody v2."""

    def __init__(self, fidelity: str, model: str, obj, wts, dt: float, root: str, root_v2: str, model_version: str,
                 telemetry: bool = False):
        self.fidelity, self.model, self.obj, self.wts, self.dt = fidelity, model, obj, wts, dt
        self.root, self.root_v2, self.model_version = root, root_v2, model_version
        self.telemetry = bool(telemetry)    # full only: telemetry scalars + FE node histories (record=True path only)
        self._rp_offset_ft = None
        self.px = None
        self.coupler = None
        self._fdm = None
        self._probe_last = None
        self.delta_mass_lb = 0.0

    def attach(self, fdm):
        if self.fidelity == "reduced":
            self.delta_mass_lb = float(fw.apply_wing_mass(fdm, self.model, self.root, self.obj))
        else:
            self.delta_mass_lb = float(fb.apply_mass_v2(fdm, self.obj, self.root_v2)["total_lb"])

    def _new_coupler(self, mode="twoway"):
        if self.fidelity == "reduced":
            return fw.FlexCoupler(self.obj, mode=mode, substeps=SUBSTEPS["reduced"])
        return fb.FlexBodyCoupler(self.obj, mode=mode, substeps=SUBSTEPS["full"])

    def wrap(self, fdm):
        self._fdm = fdm
        self.coupler = self._new_coupler()
        if self.fidelity == "reduced":
            self.px = cs.FlexFDM(fdm, self.coupler, self.dt, record=True)
        else:
            self.px = fb.FlexBodyFDM(fdm, self.coupler, self.dt, record=True, telemetry=self.telemetry)
            if self.telemetry:      # AERORP relative to the CG in body FRD ft (read-only property reads)
                self._rp_offset_ft = [-(fdm["metrics/aero-rp-x-in"] - fdm["inertia/cg-x-in"]) / 12.0,
                                      (fdm["metrics/aero-rp-y-in"] - fdm["inertia/cg-y-in"]) / 12.0,
                                      -(fdm["metrics/aero-rp-z-in"] - fdm["inertia/cg-z-in"]) / 12.0]
        return self.px

    def node_telemetry(self) -> Optional[Dict]:
        """Full fidelity with telemetry=True: FE node layout (body FRD ft, origin CG at trim) + per-frame nodal values
        of every body (one row per coupler step, aligned with the 'structure' histories). None otherwise."""
        if self.fidelity != "full" or not self.telemetry or self.px is None or not self.px.eta_hist:
            return None
        E = np.vstack(self.px.eta_hist)
        vals = fb.node_values(self.obj, E)
        return {"schema": "fd-flexbody-nodes/1", "frame": "body FRD (x fwd, y right, z down), ft, origin = CG at trim",
                "rp_offset_body_ft": self._rp_offset_ft,
                "components": fb.node_layout(self.obj, self._rp_offset_ft),
                "values": {b: {f: np.round(a, 9).tolist() for f, a in fv.items()} for b, fv in vals.items()},
                "doc": "values = own elastic deflection relative to the clamped body root (incl. the 1-g trim shape); "
                       "signs: INTERFACE_v2.md 'Telemetry sign conventions'"}

    def state(self) -> _FlexStateV2:
        st = {"schema": _FlexStateV2.schema, "fidelity": self.fidelity, "model_version": self.model_version,
              "doc": "channels = FD coupler diagnostics (root loads lbf*ft, tip deflections ft, twists/incidences deg, "
                     "elastic feedback lbf / lbf*ft); see flight-dynamics/INTERFACE_v2.md"}
        if self.px is not None and self.px.started:
            return _FlexStateV2(st, self.coupler.last)
        if self._probe_last is None:       # t = 0: trim equilibrium on a throw-away ONE-WAY coupler (no writes)
            probe = self._new_coupler("oneway")
            probe.initialize(self._fdm)
            self._probe_last = dict(probe.last)
        return _FlexStateV2(st, self._probe_last)

    def finish(self, status: str) -> Dict:
        px = self.px
        if status != "ok" or px is None or not px.started:
            return {"terms": None, "fail": None}
        if self.fidelity == "reduced":
            r = fw.response_terms(px.history(), self.obj, px.m_root_1g, self.wts)
            return {"terms": r["terms"], "fail": r["fail"],
                    "info": {"bm_peak": r["bm_peak"], "bm_allow": r["bm_allow"], "tip_max_ft": r["tip_max_ft"],
                             "twist_max_deg": r["twist_max_deg"], "bm_rms": r["bm_rms"], "m_root_1g": px.m_root_1g}}
        r = fb.response_terms_v2(px.history(), self.obj, px.out_1g, self.wts)
        return {"terms": r["terms"], "fail": r["fail"], "loads": r["loads"],
                "info": {"tip_max_ft": r["tip_max_ft"], "twist_max_deg": r["twist_max_deg"], "tail_ratio": r["tail_ratio"],
                         "fus_ratio": r["fus_ratio"], "torque_ratio": r["torque_ratio"], "ip_ratio": r["ip_ratio"],
                         "bm_allow": r["bm_allow"], "m_root_1g": r["m_root_1g"]}}


def _simulate_with_hook(sim, gains, sc, P, hook: FlexHookV2, record: bool) -> Dict:
    if sim_supports_flex_hook(sim):
        r = sim.simulate(gains, sc, P, record=record, flex=hook)
        post = r.pop("_flex", None) or {"terms": None, "fail": None}
        return r, post

    def factory(orig):                      # older sim.py: wrap at load time (trim never calls run())
        def _new(Pp):
            fdm = orig(Pp)
            hook.attach(fdm)
            return hook.wrap(fdm)
        return _new
    with _patched_new_fdm(sim, factory):
        r = sim.simulate(gains, sc, P, record=record)
    return r, hook.finish(r["status"])


def _rigid_terms(per: Sequence[Dict], terms: Dict[str, float]) -> List[str]:
    """track / effort (+ comfort, heading, hold when the profile produces them): mean over scenarios flown to the end with
    finite values, exactly evolution/fidelity.py aggregate()."""
    ok = [p for p in per if p.get("status") == "ok"]

    def mean_of(key):
        vals = [p.get(key) for p in ok]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return float(np.mean(vals)) if vals else 0.0
    avail = ["track", "effort"]
    terms["track"], terms["effort"] = mean_of("track"), mean_of("effort")
    if any("comfort" in p for p in per):
        terms["comfort"] = mean_of("comfort")
        avail.append("comfort")
    if any("heading_rms" in p for p in per):
        terms["heading"] = mean_of("heading_rms")
        avail.append("heading")
    if any("hold_osc" in p for p in per):
        terms["hold"] = mean_of("hold_osc")
        avail.append("hold")
    return avail


_PER_KEYS = ("cost", "status", "t_end", "track", "effort", "comfort", "heading_rms", "hold_osc", "wall_s")


def evaluate(gains: Dict[str, float], struct_genome, scenarios, model: str, *, fidelity: str, root: str,
             dt: float = 1 / 120, record: bool = False, profile=None, weights: Optional[fb.StructWeightsV2] = None,
             root_v2: Optional[str] = None, sim=None, blas_threads: Optional[int] = 1) -> Dict:
    """Multi-fidelity cost of one individual (gains + v2 structure genome) over the given scenarios. See module doc.

    cost = float(np.mean(per-scenario costs)) at every fidelity. Per-scenario cost: rigid = evolution simulate() cost;
    flex = simulate() cost (+ response terms, or fail_cost on a structural-ultimate fail) + pre-flight terms (margins,
    mass, smoothness); a genome failing the margin gate is not flown (cost = fail_cost = 2 * profile.fail_base)."""
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}, got {fidelity!r}")
    sim = sim or load_sim()
    if dt != sim.DT:
        raise ValueError(f"dt={dt} must equal evolution sim.DT={sim.DT} (the Runner's frame time)")
    if blas_threads is not None:
        fb.blas_threads(blas_threads)
    prof0 = profile if profile is not None else default_profile(model, sim)
    scs = _as_scenarios(scenarios, sim)
    rv2 = root_v2 or root_v2_for(root)
    terms = _zero_terms()
    out = {"fidelity": fidelity, "margins": None, "margins_fidelity": None, "per_scenario": []}
    if fidelity == "rigid":
        if struct_genome is not None:
            fb.decode_genome_v2(struct_genome, _is_asym(struct_genome))      # validate (raises) even though unused
        P = _profile_with_root(prof0, sim, root, model)
        out["model_version"] = model_version("rigid", model, root)
        rs = [sim.simulate(gains, sc, P, record=record) for sc in scs]
        cost = float(np.mean([r["cost"] for r in rs]))
        per = [{k: r[k] for k in _PER_KEYS if k in r} for r in rs]
        out["terms_available"] = _rigid_terms(per, terms)
        out.update(cost=cost, terms=_finite(terms), status=next((r["status"] for r in rs if r["status"] != "ok"), "ok"),
                   per_scenario=per)
        if record:
            out["telemetry"] = [{"trajectory": r.get("trajectory")} for r in rs]
        return out

    gate = MARGIN_GATE[fidelity]
    asym = _is_asym(struct_genome)
    if fidelity == "reduced":
        cs.ensure_root(model, root)
        fb.ensure_root_v2(model, rv2)
        P = _profile_with_root(prof0, sim, root, model)
        out["model_version"] = model_version("reduced", model, root, rv2, weights)
        proj = project_to_reduced(struct_genome, model, rv2)
        obj = reduced_wing(struct_genome, model, rv2)
        m = obj.margins()
        mdl_mass = fb.FlexBodyModel(model, struct_genome, asymmetric=asym, root_v2=rv2)
        pre_terms, fail = {}, None
        wts0 = weights or fb.StructWeightsV2()
        for key, wk, nm in (("flutter_margin", wts0.w_flutter, "flutter"), ("div_margin", wts0.w_div, "divergence")):
            if m[key] < gate:
                fail = fail or nm
            pre_terms["J_" + key] = wk * max(0.0, (wts0.margin_req - m[key]) / (wts0.margin_req - 1.0)) ** 2
        if m["margin_error"]:
            fail = fail or "margin_error"
        ms = mdl_mass.mass_summary()
        pre_terms["J_mass"] = wts0.w_mass * ms["total_frac"]                  # same definition as full (v2 mass model)
        pre_terms["J_smooth"] = fb.smoothness_penalty(mdl_mass.genes, wts0.w_smooth)
        sz = fb.sizing_v2(mdl_mass, wts0)
        pre_terms.update(sz["terms"])
        out["sizing"] = sz
        out["margins"] = {"flutter_margin": m["flutter_margin"], "div_margin": m["div_margin"], "reversal_margin": None,
                          "margin_error": m["margin_error"], "f_modes_hz": m["f_modes_hz"], "detail": m}
        out["projection"] = proj
        out["mass"] = ms
        out["mass_applied_to_fdm"] = "wing point masses of the projected v1 wing only"
    else:
        fb.ensure_root_v2(model, rv2)
        P = _profile_with_root(prof0, sim, rv2, model)
        out["model_version"] = model_version("full", model, root, rv2, weights)
        obj = fb.FlexBodyModel(model, struct_genome, asymmetric=asym, root_v2=rv2)
        wts0 = weights or fb.StructWeightsV2()
        pre = fb.margin_terms_v2(obj, wts0, gate=gate)
        pre_terms, fail = dict(pre["terms"]), pre["fail"]
        out["margins"] = pre["margins"]
        out["mass"] = pre["mass"]
        out["sizing"] = pre["sizing"]
        out["mass_applied_to_fdm"] = "all bodies (wing L/R, HT, VT, aft fuselage point masses)"
    wts = weights or dataclasses.replace(wts0, fail_cost=float(2 * P.fail_base))
    out["terms_available"] = ["track", "effort"] + list(PRE_TERMS[fidelity]) + list(RESP_TERMS[fidelity])
    out["margins_fidelity"] = fidelity
    out["margin_gate"] = gate
    for k, v in pre_terms.items():
        terms[k] = float(v)
    pre_sum = float(sum(pre_terms[k] for k in PRE_TERMS[fidelity]))
    if fail:
        out.update(cost=float(wts.fail_cost), terms=_finite(terms), status=fail)
        return out
    per, telemetry, loads_acc = [], [], []
    for sc in scs:
        hook = FlexHookV2(fidelity, model, obj, wts, dt, root, rv2, out["model_version"], telemetry=record)
        r, post = _simulate_with_hook(sim, gains, sc, P, hook, record)
        entry = {k: r[k] for k in _PER_KEYS if k in r}
        entry["sim_cost"] = r["cost"]
        entry["delta_mass_lb_applied"] = hook.delta_mass_lb
        cost_s = r["cost"]
        if r["status"] == "ok" and post.get("terms") is not None:
            entry["struct"] = {k: float(v) for k, v in post["terms"].items()}
            entry.update(post["info"])
            if "loads" in post:
                entry["loads"] = post["loads"]
                loads_acc.append(post["loads"])
            if post["fail"]:
                entry["status"] = post["fail"]
                cost_s = float(wts.fail_cost)
            else:
                cost_s = cost_s + sum(post["terms"].values())
        entry["cost"] = cost_s + pre_sum
        per.append(entry)
        if record:
            telemetry.append({"trajectory": r.get("trajectory"),
                              "structure": {k: v.tolist() for k, v in hook.px.history().items()} if hook.px is not None else None,
                              "nodes": hook.node_telemetry()})
    avail_rigid = _rigid_terms(per, terms)
    out["terms_available"] = avail_rigid + list(PRE_TERMS[fidelity]) + list(RESP_TERMS[fidelity])
    ok = [e for e in per if e["status"] == "ok" and "struct" in e]
    for k in RESP_TERMS[fidelity]:
        vals = [e["struct"][k] for e in ok if math.isfinite(e["struct"].get(k, math.nan))]
        terms[k] = float(np.mean(vals)) if vals else 0.0
    if loads_acc:
        out["loads"] = {nm: {"peak_abs_lbft": float(max(L[nm]["peak_abs"] for L in loads_acc)),
                             "rms_dev_1g_lbft": float(np.mean([L[nm]["rms_dev_1g"] for L in loads_acc])),
                             "trim_1g_lbft": float(loads_acc[0][nm]["trim_1g"])} for nm in fb.OUT_NAMES}
    cost = float(np.mean([e["cost"] for e in per]))
    out.update(cost=cost, terms=_finite(terms), per_scenario=per,
               status=next((e["status"] for e in per if e["status"] != "ok"), "ok"))
    if record:
        out["telemetry"] = telemetry
    return out


def phase1_scenarios(model: str, sim=None, n: Optional[int] = None, seed: Optional[int] = None):
    """The Runner's Phase-1 scenarios for a model (configs/phase1.json: 3 scenarios, scenario_seed 1)."""
    sim = sim or load_sim()
    with open(PHASE1_CONFIG) as f:
        cfg = json.load(f)
    P = default_profile(model, sim)
    return sim.make_scenarios(n or cfg["scenarios"], cfg["scenario_seed"] if seed is None else seed, P)
