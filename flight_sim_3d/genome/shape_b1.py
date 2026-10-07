"""P3-B1 shape block, genome side (opt-in preset ``phase3_b1``; nothing here is used by any other preset).

Source of truth: Flight Dynamics ``planform_b1.py`` (INTERFACE_v2 §14). The 6 shape genes are built at load time from
``planform_b1.shape_schema()`` (names, ranges, scale, default). ``B1_SCHEMA_PIN`` is a literal copy, and a test asserts
it equals FD's live module, so drift fails loudly.

Decode / evaluation order (one genome, fidelity ``full_a1_b1``):
    26 genes -> {controller 8, structure 12, shape 6}
    shape   -> FD decode_shape_b1 (raises, never clips) -> FD geometry_gate
               gate fail -> FD status 'geometry_gate:<reason>', cost = FD fail_cost, NOT flown (hard reject, no credit)
    -> FD wing rebuild on the shaped planform -> structure genes on that baseline -> margins + sizing -> flight
All of that is ``flexeval_b1.evaluate``: genome adds no cost term. Default evaluator: Evolution Runner's
``evolution.fidelity.evaluate_genome(..., 'full_a1_b1', shape=...)`` (ER wired it ~18:07 PT). Fallback (``via='fd'``):
the same flex path re-assembled here (ER profile / scenarios, ``struct_from`` clamp, ``sim=evolution.sim``,
``blas_threads=1``, mass-credit clip hook) calling FD's ``flexeval_b1.evaluate(shape_genome=)`` directly. Both agree bit
for bit (p3b1_verify.py).

Read-only on evolution/ and flight-dynamics/: imports run with bytecode writing off.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)

SHAPE_B1_BLOCK = "shape_b1"
B1_FIDELITY = "full_a1_b1"
HOST_FIDELITY = "full_a1"            # B1 rides on the A1 64-strip host; same prepared roots

# Literal mirror of FD planform_b1.SHAPE_GENES_B1 (name, lo, hi, scale, default). Test: == FD live module.
B1_SCHEMA_PIN = (
    ("wing_chord_taper_1", 0.85, 1.05, "linear", 1.0),
    ("wing_chord_taper_2", 0.85, 1.05, "linear", 1.0),
    ("wing_chord_taper_3", 0.85, 1.05, "linear", 1.0),
    ("wing_twist_mid_deg", -2.0, 1.0, "linear", 0.0),
    ("wing_twist_tip_deg", -4.0, 1.0, "linear", 0.0),
    ("wing_sweep_qc_delta_deg", -5.0, 5.0, "linear", 0.0),
)

# FD's gene encoding (planform_b1.GENE_ENCODING, stated in r1): all 6 genes LINEAR IN VALUE, value = lo + u (hi - lo).
# ("log" in FD's docs is the spanwise chord interpolation, log-PCHIP, not the gene encoding.) Test + load: == FD live.
B1_GENE_ENCODING_PIN = "linear_in_value: value = lo + u*(hi-lo), u in [0,1]; encode = (value-lo)/(hi-lo); no log scale"
B1_REV_PIN = 1                                            # planform_b1.B1_REV the pins below belong to (r1)

# flight-dynamics/v2_results/model_versions_post_p3b1r1.json (FD B1 r1, 2026-10-06 ~18:33 PT). Current.
FD_B1_MODEL_VERSIONS = {
    "c172x": "full_a1_b1:flexv2b1:56ee798e",
    "T38": "full_a1_b1:flexv2b1:7e871977",
    "737": "full_a1_b1:flexv2b1:6523753c",
    "f16": "full_a1_b1:flexv2b1:617078a9",
}
# flight-dynamics/v2_results/model_versions_post_p3b1.json (B1 r0, ~17:50 PT). SUPERSEDED by r1 (FD: valid for r0 code
# only). Transition policy = P2.5 post-mass: a live or recorded r0 string is warn-only, never raises (even in 'raise').
FD_B1_MODEL_VERSIONS_R0 = {
    "c172x": "full_a1_b1:flexv2b1:3e40908a",
    "T38": "full_a1_b1:flexv2b1:982bce54",
    "737": "full_a1_b1:flexv2b1:1bc748ac",
    "f16": "full_a1_b1:flexv2b1:bfb25718",
}
# A1 host pins (model_versions_post_p3a1.json), for the bit-identity verify only
FD_A1_MODEL_VERSIONS = {
    "c172x": "full_a1:flexv2a1:36fb4f5a",
    "T38": "full_a1:flexv2a1:f248873e",
    "737": "full_a1:flexv2a1:522189cb",
    "f16": "full_a1:flexv2a1:4a9e12bc",
}

# Operators: block_ops.py (= ER evolution/ga.py P3-B1; sigma 0.25 x half-range in encoded space, ln x for chord tapers).


# ----------------------------------------------------------------------------------------------------- FD imports
@contextlib.contextmanager
def _no_bytecode():
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = old


def planform():
    """FD planform_b1 (read-only import through fd_bridge: FD dir appended to sys.path, no bytecode)."""
    import fd_bridge
    return fd_bridge._import("planform_b1")


def er_modules():
    """(evolution.fidelity, evolution.sim, evolution.batch, flexeval_b1), imported read-only, no bytecode."""
    with _no_bytecode():
        if TEAM not in sys.path:
            sys.path.append(TEAM)
        from evolution import batch, fidelity, sim  # noqa: E402
        fidelity.fd_a1_modules()                 # FD dir on sys.path + ER's in-process prepare guard
        fb1 = importlib.import_module("flexeval_b1")
    return fidelity, sim, batch, fb1


# ----------------------------------------------------------------------------------------------------- schema
def fd_shape_schema():
    return list(planform().shape_schema())


def check_schema_pin(fd_schema=None) -> None:
    """Raise if FD's live shape schema differs from B1_SCHEMA_PIN (drift must fail loudly)."""
    live = tuple((g.name, float(g.lo), float(g.hi), g.scale, float(g.default)) for g in (fd_schema or fd_shape_schema()))
    if live != B1_SCHEMA_PIN:
        raise ValueError(f"FD planform_b1 shape schema drifted from genome's pin:\n live {live}\n pin  {B1_SCHEMA_PIN}")


def check_encoding_pin(live: Optional[str] = None) -> None:
    """Raise if FD's stated gene encoding (planform_b1.GENE_ENCODING) differs from B1_GENE_ENCODING_PIN."""
    live = planform().GENE_ENCODING if live is None else live
    if live != B1_GENE_ENCODING_PIN:
        raise ValueError(f"FD planform_b1.GENE_ENCODING drifted from genome's pin:\n live {live!r}\n pin  {B1_GENE_ENCODING_PIN!r}")


def shape_genes():
    """GeneSpecs for the B1 shape block, built from FD's schema at load time (like genes_from_fd_schema for structure).
    Stored in FD's own encoding (all linear in FD's lock): a genome's shape slice in u is a valid FD [0,1]^6 vector.
    The log treatment of the chord tapers lives in the operators (block_ops.py, = ER's ga.py), not in the storage."""
    import genome_schema as GS
    sch = fd_shape_schema()
    check_schema_pin(sch)
    check_encoding_pin()
    out = []
    for g in sch:
        if g.scale != "linear":
            raise ValueError(f"FD shape gene {g.name}: scale {g.scale!r} (genome stores FD's vector encoding, linear)")
        units = "deg" if g.name.endswith("_deg") else "x"
        out.append(GS.GeneSpec(g.name, SHAPE_B1_BLOCK, float(g.lo), float(g.hi), g.scale, float(g.default), units,
                               g.meaning, "none"))
    return tuple(out)


def shape_from_values(values: Dict[str, float], genes) -> Dict[str, float]:
    """Physical shape dict for FD: exactly the genome decode (GeneSpec.decode, linear: min + u (max - min) = FD
    GENE_ENCODING lo + u (hi - lo), bit for bit), no post-processing. Same as ER (eval.shape_values -> fidelity.shape_from).
    The identity u decodes to FD's defaults exactly (A1 short-circuit); anything out of range raises in FD's decode.
    (r0 had a clamp + 1e-12 snap-to-default here; both were never triggered -- 0 of 20k random u plus corners -- and are
    removed so the path is FD's encoding with nothing in between.)"""
    return {g.name: float(values[g.name]) for g in genes}


# ----------------------------------------------------------------------------------------------------- model_version
def fd_b1_model_version(profile_d: Dict) -> str:
    F, esim, _, fb1 = er_modules()
    P = esim.Profile.from_dict(profile_d)
    root, rv2 = F.roots(P)
    return fb1.model_version(B1_FIDELITY, P.aircraft, root, rv2)


def classify_b1_model_version(model: str, mv: Optional[str]) -> str:
    """'match' (post_p3b1r1) | 'superseded_r0' (post_p3b1, warn-only) | 'unknown'. For live or recorded strings
    (e.g. an ER run / cache entry carrying an r0 model_version)."""
    if mv is not None and mv == FD_B1_MODEL_VERSIONS.get(model):
        return "match"
    if mv is not None and mv == FD_B1_MODEL_VERSIONS_R0.get(model):
        return "superseded_r0"
    return "unknown"


def check_b1_model_version(model: str, profile_d: Optional[Dict], mode: str = "warn") -> Dict:
    """Same policy as fd_bridge.check_model_version (P2.5 post-mass transition), against post_p3b1r1 pins.
    mode: warn (default) | raise | off.
    - match post_p3b1r1 -> ok
    - r0 string (post_p3b1, superseded) -> warn only, never raise (transition window)
    - anything else (unknown) -> warn, or raise when mode='raise'."""
    if mode not in ("warn", "raise", "off"):
        raise ValueError(f"flex.model_version_check must be warn | raise | off, got {mode!r}")
    rec = FD_B1_MODEL_VERSIONS.get(model)
    r0 = FD_B1_MODEL_VERSIONS_R0.get(model)
    out = {"model": model, "fidelity": B1_FIDELITY, "recorded": rec, "superseded_r0": r0, "current": None,
           "match": None, "status": None, "message": None}
    if mode == "off":
        return out
    try:
        cur = fd_b1_model_version(profile_d)
    except Exception as e:
        out.update(status="unverified",
                   message=f"could not compute FD model_version for {model} {B1_FIDELITY} ({type(e).__name__}: {e})")
        if mode == "raise":
            raise RuntimeError(out["message"]) from e
        return out
    out["current"] = cur
    status = classify_b1_model_version(model, cur)
    if status == "match":
        out.update(match=True, status="match")
        return out
    if status == "superseded_r0":
        out.update(match=False, status="superseded_r0",
                   message=(f"FD model_version for {model} {B1_FIDELITY} is the superseded B1 r0 string {cur} (genome "
                            f"expects post_p3b1r1 {rec}): accepted during the r0 -> r1 transition, warn only"))
        return out  # never raise on the known r0 set
    out.update(match=False, status="unknown",
               message=(f"FD model_version for {model} {B1_FIDELITY} is {cur}, genome recorded post_p3b1r1 {rec} "
                        f"(r0 was {r0}): unknown -- re-check phase3_b1 and update shape_b1.FD_B1_MODEL_VERSIONS"))
    if mode == "raise":
        raise RuntimeError(out["message"])
    return out


# ----------------------------------------------------------------------------------------------------- ER profile
def er_profile(config: str, aircraft: str) -> Dict:
    """ER's resolved profile for `aircraft` from an ER batch config (read-only; same call as ER's tip-verify)."""
    _, _, batch, _ = er_modules()
    path = config if os.path.isabs(config) else os.path.join(TEAM, config)
    cfg = batch.resolve_config(json.load(open(path)), "genome_b1")
    ent = next((a for a in cfg["aircraft"] if a["name"] == aircraft), None)
    if ent is None:
        raise ValueError(f"aircraft {aircraft!r} not in ER config {path}")
    return ent["resolved_profile"]


def er_scenarios(profile_d: Dict, n: int, seed: int) -> List[Dict]:
    _, esim, _, _ = er_modules()
    P = esim.Profile.from_dict(profile_d)
    return [s.to_dict() for s in esim.make_scenarios(n, seed, P)]


# ----------------------------------------------------------------------------------------------------- evaluation
CONTROLLER_OBJ = {"track": "track_alt", "effort": "effort", "comfort": "comfort", "heading": "track_heading_rms"}
RIGID_TERMS = ("track", "effort", "comfort", "heading", "hold")


def er_has_b1() -> bool:
    F = er_modules()[0]
    return B1_FIDELITY in getattr(F, "FIDELITIES", ())


def evaluate_b1(profile_d: Dict, gains: Dict[str, float], struct: Dict[str, float], shape: Optional[Dict[str, float]],
                scs_d: Sequence[Dict], *, check_mv: Optional[str] = None, via: str = "auto") -> Dict:
    """One genome at full_a1_b1. via 'auto' (default): ER's evolution.fidelity.evaluate_genome(..., 'full_a1_b1',
    shape=...) when ER has wired it (it has since 2026-10-06 ~18:07 PT), else the direct path. via 'er' / 'fd' force one.
    Both are FD flexeval_b1.evaluate with ER's inputs; p3b1_verify.py checks they agree bit for bit.
    Gate reject: status 'geometry_gate:<reason>', cost fail_cost, one aligned not-flown entry per scenario.
    Decode errors raise (FD policy)."""
    if via not in ("auto", "er", "fd"):
        raise ValueError(f"via must be auto | er | fd, got {via!r}")
    if via == "er" or (via == "auto" and er_has_b1()):
        return _evaluate_b1_er(profile_d, gains, struct, shape, scs_d, check_mv)
    return _evaluate_b1_fd(profile_d, gains, struct, shape, scs_d, check_mv)


_OUT_KEYS = ("cost", "status", "terms", "terms_available", "fidelity", "model_version", "feasible", "margins",
             "mass_total_frac", "per_scenario", "shape_genes", "shape_cache_key", "geometry_gate", "planform",
             "structural_model", "tip_bm")


def _evaluate_b1_er(profile_d, gains, struct, shape, scs_d, check_mv):
    F = er_modules()[0]
    if shape is not None:   # FD decode policy (raise, never clip) before ER's clamp-free shape_from
        planform().decode_shape_b1(shape)
    with _no_bytecode():
        r = F.evaluate_genome(profile_d, gains, struct, list(scs_d), B1_FIDELITY, None, shape=shape)
    if check_mv is not None and r["model_version"] != check_mv:
        raise RuntimeError(f"FD model_version {r['model_version']} != expected {check_mv}")
    out = {k: r.get(k) for k in _OUT_KEYS}
    out["planform"] = r.get("planform")
    out["evaluator"] = "evolution.fidelity.evaluate_genome(full_a1_b1)"
    for k in ("J_mass_fd", "mass_credit_clip", "mass_credit_delta"):
        if k in r:
            out[k] = r[k]
    return out


def _evaluate_b1_fd(profile_d, gains, struct, shape, scs_d, check_mv):
    F, esim, _, fb1 = er_modules()
    P = esim.Profile.from_dict(profile_d)
    scs = [esim.Scenario.from_dict(s) for s in scs_d]
    struct = F.struct_from(struct)
    F.check_prepared(P, HOST_FIDELITY)
    root, rv2 = F.roots(P)
    with F._gate(None):
        r = fb1.evaluate(gains, struct, scs, P.aircraft, fidelity=B1_FIDELITY, root=root, root_v2=rv2, profile=P,
                         sim=esim, record=False, blas_threads=1, shape_genome=shape)
    mv = r["model_version"]
    if check_mv is not None and mv != check_mv:
        raise RuntimeError(f"FD model_version {mv} != expected {check_mv}")
    if r["per_scenario"]:
        per = [{k: e[k] for k in F.PER_KEYS if k in e} for e in r["per_scenario"]]
    else:   # geometry-gate or margin-gate fail: not flown
        per = [{"cost": float(r["cost"]), "status": r["status"], "t_end": 0.0, "not_flown": True} for _ in scs]
    for e in per:
        e["fidelity"], e["model_version"] = B1_FIDELITY, mv
    mg = r.get("margins") or {}
    out = {"cost": float(r["cost"]), "status": r["status"], "terms": dict(r["terms"]),
           "terms_available": list(r.get("terms_available") or []), "fidelity": B1_FIDELITY, "model_version": mv,
           "feasible": r["status"] == "ok",
           "margins": {k: mg.get(k) for k in ("flutter_margin", "div_margin", "reversal_margin")},
           "mass_total_frac": (r.get("mass") or {}).get("total_frac"), "per_scenario": per,
           "shape_genes": r.get("shape_genes"), "shape_cache_key": r.get("shape_cache_key"),
           "geometry_gate": r.get("geometry_gate"), "planform": r.get("planform"),
           "structural_model": r.get("structural_model")}
    sz = r.get("sizing") or {}
    out["tip_bm"] = {"method": sz.get("tip_bm_method"),
                     "strip_discrete_term": (sz.get("tip_bm_strip_discrete") or {}).get("term")}
    if P.flex_mass_credit_clip and r["per_scenario"] and r.get("mass"):
        fb = F.fd_modules()["fb"]
        F.apply_mass_credit_clip(out, r["mass"], P.flex_mass_credit_clip, fb.StructWeightsV2().w_mass)
    out["evaluator"] = "shape_b1 direct: FD flexeval_b1.evaluate with ER inputs"
    return out


def geometry_rejected(res: Dict) -> bool:
    return str(res.get("status", "")).startswith("geometry_gate")


# ----------------------------------------------------------------------------------------------------- task
def _check_weights(weights: Dict[str, float], profile_d: Dict) -> None:
    """The cost is FD+ER's (ER sim cost with the profile's controller weights + FD structural J_*); genome adds nothing.
    So the preset's named weights must equal the ER profile's, or the reported weights would lie."""
    want = {"track_alt": 1.0, "effort": float(profile_d.get("w_effort", 2.0)),
            "comfort": float(profile_d.get("w_comfort", 0.0)), "track_heading_rms": float(profile_d.get("w_heading", 0.0)),
            "structural_v2": 1.0}
    got = {k: float(weights.get(k, 0.0)) for k in want}
    if got != want:
        raise ValueError(f"phase3_b1 fitness.weights {got} must equal the ER profile's cost weights {want} "
                         "(full_a1_b1 cost is FD flexeval_b1 + ER sim cost; no parallel formula)")


def build_task_b1(raw: Dict, build_task):
    """Build the phase3_b1 task: the phase2_flex controller + structure_v2 spec via the normal build_task (shape, init,
    operators and the B1 flex keys stripped), then append FD's 6 shape genes, resolve init over all 26 genes, attach
    operators and the full_a1_b1 evaluator."""
    import adapter
    import block_ops
    orig = copy.deepcopy(raw)
    raw = copy.deepcopy(raw)
    blocks = dict(raw.get("blocks", {}))
    if blocks.pop(SHAPE_B1_BLOCK) is not True:
        raise ValueError("blocks.shape_b1 must be true (FD's 6 genes are evolved as a whole block)")
    if not blocks.get("structure_v2"):
        raise ValueError("phase3_b1 needs the structure_v2 block (shape only changes the baseline the 12 genes scale)")
    flex = dict(raw.get("flex", {}))
    fid = flex.pop("fidelity", None)
    if fid != B1_FIDELITY:
        raise ValueError(f"blocks.shape_b1 needs flex.fidelity = {B1_FIDELITY!r}, got {fid!r}")
    if flex.get("asymmetric"):
        raise ValueError("phase3_b1: flex.asymmetric is not supported (B1 is L = R)")
    mv_mode = flex.get("model_version_check", "warn")
    flex["model_version_check"] = "off"      # the base full-fidelity check does not apply; B1 is checked below
    b1cfg = {k: v for k, v in dict(raw.pop("shape_b1", {})).items() if not k.startswith("_")}
    bad = set(b1cfg) - {"er_profile_config"}
    if bad:
        raise ValueError(f"unknown shape_b1 keys {sorted(bad)}")
    init_cfg = raw.pop("init", None)
    ops_cfg = raw.pop("operators", None)
    base = build_task(dict(raw, blocks=blocks, flex=flex))
    genes = shape_genes()
    spec = base.spec
    spec.genes = list(spec.genes) + list(genes)
    spec.enabled_blocks = tuple(spec.enabled_blocks) + (SHAPE_B1_BLOCK,)
    init = adapter.resolve_init(init_cfg, spec, base.profile)
    ops = block_ops.resolve_operators(ops_cfg, spec)
    model = base.profile.jsbsim_model
    prof_d = er_profile(b1cfg.get("er_profile_config", "evolution/configs/phase2_smoke_p25.json"), model)
    _check_weights(base.fitness.weights, prof_d)
    warnings = list(base.warnings)
    gb = prof_d.get("gain_bounds") or {}
    for g in spec.genes:
        if g.name in gb and (g.min, g.max) != tuple(float(x) for x in gb[g.name]):
            warnings.append(f"controller gene {g.name} range [{g.min}, {g.max}] differs from ER profile {gb[g.name]}")
    mv = check_b1_model_version(model, prof_d, mv_mode)
    if mv["message"]:
        warnings.append("FD MODEL_VERSION: " + mv["message"])
    flex_consts = dict(base.flex_constants, fidelity=B1_FIDELITY, margin_gate=1.0, fd_model_version=mv["current"],
                       fd_model_version_recorded=mv["recorded"])
    return B1Task(base.name, spec, base.fitness, base.scenarios, base.profile, orig, warnings, base.conditions, {},
                  flex_consts, {}, init, er_profile_d=prof_d, operators=ops, shape_gene_specs=tuple(genes),
                  b1_model_version=mv)


import adapter as _adapter  # noqa: E402  (adapter imports this module lazily, so no cycle at import time)


@dataclasses.dataclass
class B1Task(_adapter.Task):
    """phase3_b1 task: ER scenarios (ER sim.make_scenarios on ER's resolved profile: the inputs FD's evaluate takes) and
    the full_a1_b1 evaluator. sim_fixed is empty: like ER, only the 8 controller genes go to the flight."""
    er_profile_d: Dict = dataclasses.field(default_factory=dict)
    operators: Dict = dataclasses.field(default_factory=dict)
    shape_gene_specs: tuple = ()
    b1_model_version: Dict = dataclasses.field(default_factory=dict)

    def make_scenarios(self, n: int, seed: int):
        """ER sim.Scenario objects (dataclasses: evolve.py asdict()s them) for ER's resolved profile."""
        _, esim, _, _ = er_modules()
        return list(esim.make_scenarios(n, seed, esim.Profile.from_dict(self.er_profile_d)))

    def split(self, values: Dict[str, float]):
        blk = {g.name: g.block for g in self.spec.genes}
        gains = {k: float(v) for k, v in values.items() if k in blk and blk[k] not in ("structure_v2", SHAPE_B1_BLOCK)}
        struct = {k: float(v) for k, v in values.items() if blk.get(k) == "structure_v2"}
        shape = shape_from_values({k: v for k, v in values.items() if blk.get(k) == SHAPE_B1_BLOCK}, self.shape_gene_specs)
        return gains, struct, shape

    def evaluate(self, gains, scenarios, record=None):
        g, s, sh = self.split(gains)
        scs = [sc if isinstance(sc, dict) else sc.to_dict() for sc in scenarios]
        r = evaluate_b1(self.er_profile_d, g, s, sh, scs)
        t = r["terms"]
        r["objectives"] = {CONTROLLER_OBJ[k]: float(t[k]) for k in CONTROLLER_OBJ}
        r["objectives"]["structural_v2"] = float(sum(v for k, v in t.items() if k not in RIGID_TERMS))
        r["fd_struct_terms"] = {k: float(v) for k, v in t.items() if k not in RIGID_TERMS}
        r["J_wing_tip_bm_limit"] = float(t.get("J_wing_tip_bm_limit", 0.0))
        r["struct_v2_source"] = "fd"
        if geometry_rejected(r):
            r["violation"] = r["cost"]
        return r
