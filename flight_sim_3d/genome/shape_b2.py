"""P3-B2a shape block, genome side (opt-in preset ``phase3_b2a``; nothing here is used by any other preset).

Source of truth: Flight Dynamics ``planform_b2.py`` (INTERFACE_v2 §15.10–15.11; B2a SIGNED OFF 2026-10-06).
The shape genes are built at load time from ``planform_b2.shape_schema_b2(model)`` (per-aircraft ranges), minus
``locked_genes`` (excluded from the encoded GA vector). ``B2_SCHEMA_PIN`` / ``B2_B1_SCHEMA_PIN`` are literal copies;
tests assert equality with FD's live module.

Decode / evaluation order (one genome, fidelity ``full_a1_b2a``):
    29 genes -> {controller 8, structure 12, shape 9 active}   (tc locked)
    active shape + locked genes injected as FD schema defaults (1.0)
               -> FD decode_shape_b2 (raises, never clips) -> FD geometry_gate_b2
               gate fail -> FD status 'geometry_gate:<reason>', cost = FD fail_cost, NOT flown (hard reject, no credit)
    -> FD wing rebuild -> structure genes -> margins + sizing -> flight
All of that is ``flexeval_b2.evaluate``: genome adds no cost term. Default evaluator: Evolution Runner's
``evolution.fidelity.evaluate_genome(..., 'full_a1_b2a', shape=...)`` when ER has wired it; else ``via='fd'``
calling FD's ``flexeval_b2.evaluate(shape_genome=)`` with ER profile / scenarios / sim / blas_threads=1.
``struct_v2_source`` stays ``fd``.

Read-only on evolution/ and flight-dynamics/: imports run with bytecode writing off.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)

SHAPE_B2_BLOCK = "shape_b2"
B2_FIDELITY = "full_a1_b2a"
HOST_FIDELITY = "full_a1"            # B2a at B2-defaults flies on the A1 64-strip host (<root>_v2)
B2_REV_PIN = 0                       # planform_b2.B2_REV

# B1 part of the B2a vector (aircraft-independent) — must equal shape_b1.B1_SCHEMA_PIN / planform_b1.
B2_B1_SCHEMA_PIN = (
    ("wing_chord_taper_1", 0.85, 1.05, "linear", 1.0),
    ("wing_chord_taper_2", 0.85, 1.05, "linear", 1.0),
    ("wing_chord_taper_3", 0.85, 1.05, "linear", 1.0),
    ("wing_twist_mid_deg", -2.0, 1.0, "linear", 0.0),
    ("wing_twist_tip_deg", -4.0, 1.0, "linear", 0.0),
    ("wing_sweep_qc_delta_deg", -5.0, 5.0, "linear", 0.0),
)

# Literal mirror of FD planform_b2.schema(model) for all 4 aircraft (name, aircraft, lo, hi, default, feature, requires).
B2_SCHEMA_PIN = (
    ("wing_dihedral_delta_deg", "c172x", 0.0, 3.0, 0.0, "dihedral", None),
    ("wing_tc_root_scale", "c172x", 0.875, 1.25, 1.0, "thickness", "energy_cost"),
    ("wing_tc_tip_ratio", "c172x", 0.85, 1.15, 1.0, "thickness", "energy_cost"),
    ("wing_camber_root_delta_pct", "c172x", -1.0, 2.0, 0.0, "camber", None),
    ("wing_camber_tip_delta_pct", "c172x", -1.0, 2.0, 0.0, "camber", None),
    ("wing_dihedral_delta_deg", "T38", 0.0, 3.0, 0.0, "dihedral", None),
    ("wing_tc_root_scale", "T38", 0.925, 1.25, 1.0, "thickness", "energy_cost"),
    ("wing_tc_tip_ratio", "T38", 0.85, 1.15, 1.0, "thickness", "energy_cost"),
    ("wing_camber_root_delta_pct", "T38", -0.5, 1.5, 0.0, "camber", None),
    ("wing_camber_tip_delta_pct", "T38", -0.5, 1.5, 0.0, "camber", None),
    ("wing_dihedral_delta_deg", "737", 0.0, 2.0, 0.0, "dihedral", None),
    ("wing_tc_root_scale", "737", 0.9, 1.15, 1.0, "thickness", "energy_cost"),
    ("wing_tc_tip_ratio", "737", 0.85, 1.15, 1.0, "thickness", "energy_cost"),
    ("wing_camber_root_delta_pct", "737", -1.0, 1.0, 0.0, "camber", None),
    ("wing_camber_tip_delta_pct", "737", -1.0, 1.0, 0.0, "camber", None),
    ("wing_dihedral_delta_deg", "f16", 0.0, 3.0, 0.0, "dihedral", None),
    ("wing_tc_root_scale", "f16", 0.85, 1.25, 1.0, "thickness", "energy_cost"),
    ("wing_tc_tip_ratio", "f16", 0.85, 1.15, 1.0, "thickness", "energy_cost"),
    ("wing_camber_root_delta_pct", "f16", -0.5, 1.0, 0.0, "camber", None),
    ("wing_camber_tip_delta_pct", "f16", -0.5, 1.0, 0.0, "camber", None),
)

B2_GENE_ENCODING_PIN = (
    "linear_in_value: value = lo + u*(hi-lo), u in [0,1]; encode = (value-lo)/(hi-lo); no log scale; "
    "B2 ranges per aircraft (decode needs the model)"
)

# flight-dynamics/v2_results/model_versions_post_p3b2a.json (FD B2a signed off, 2026-10-06 ~21:31 PT).
FD_B2_MODEL_VERSIONS = {
    "c172x": "full_a1_b2a:flexv2b2a:847bed9b",
    "T38": "full_a1_b2a:flexv2b2a:b635a51d",
    "737": "full_a1_b2a:flexv2b2a:5d5a8f17",
    "f16": "full_a1_b2a:flexv2b2a:07b08913",
}
# B1 r1 host pins (unchanged under B2a).
FD_B1_MODEL_VERSIONS = {
    "c172x": "full_a1_b1:flexv2b1:56ee798e",
    "T38": "full_a1_b1:flexv2b1:7e871977",
    "737": "full_a1_b1:flexv2b1:6523753c",
    "f16": "full_a1_b1:flexv2b1:617078a9",
}

DEFAULT_LOCKED_GENES = ("wing_tc_root_scale", "wing_tc_tip_ratio")  # REQUIRES_ENERGY until ER wires J_energy


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
    """FD planform_b2 (read-only import through fd_bridge: FD dir appended to sys.path, no bytecode)."""
    import fd_bridge
    return fd_bridge._import("planform_b2")


def er_modules():
    """(evolution.fidelity, evolution.sim, evolution.batch, flexeval_b2), imported read-only, no bytecode."""
    with _no_bytecode():
        if TEAM not in sys.path:
            sys.path.append(TEAM)
        from evolution import batch, fidelity, sim  # noqa: E402
        fidelity.fd_a1_modules()
        fb2 = importlib.import_module("flexeval_b2")
    return fidelity, sim, batch, fb2


# ----------------------------------------------------------------------------------------------------- schema
def fd_shape_schema(model: str):
    return list(planform().shape_schema_b2(model))


def check_schema_pin(model: Optional[str] = None) -> None:
    """Raise if FD's live B1+B2a schema differs from the genome pins."""
    pb2 = planform()
    live_b1 = tuple((g.name, float(g.lo), float(g.hi), g.scale, float(g.default)) for g in pb2.pb1.SHAPE_GENES_B1)
    if live_b1 != B2_B1_SCHEMA_PIN:
        raise ValueError(f"FD planform_b1 shape schema drifted from genome's B2_B1 pin:\n live {live_b1}\n pin  {B2_B1_SCHEMA_PIN}")
    models = (model,) if model else pb2.AIRCRAFT
    live = []
    for m in models:
        for g in pb2.schema(m):
            live.append((g["name"], m, float(g["lo"]), float(g["hi"]), float(g["default"]), g["feature"], g["requires"]))
    want = [row for row in B2_SCHEMA_PIN if row[1] in models]
    if tuple(live) != tuple(want):
        raise ValueError(f"FD planform_b2 schema drifted from genome's pin:\n live {tuple(live)}\n pin  {tuple(want)}")


def check_encoding_pin(live: Optional[str] = None) -> None:
    live = planform().GENE_ENCODING if live is None else live
    if live != B2_GENE_ENCODING_PIN:
        raise ValueError(f"FD planform_b2.GENE_ENCODING drifted from genome's pin:\n live {live!r}\n pin  {B2_GENE_ENCODING_PIN!r}")


def check_locked_genes(locked: Sequence[str], model: str) -> None:
    """Validate locked_genes: known FD shape names; every REQUIRES_ENERGY gene must be locked until energy cost exists."""
    pb2 = planform()
    names = {g["name"] for g in fd_shape_schema(model)}
    locked = list(locked)
    bad = [n for n in locked if n not in names]
    if bad:
        raise ValueError(f"locked_genes {bad} are not FD shape genes for {model} (have {sorted(names)})")
    if len(set(locked)) != len(locked):
        raise ValueError(f"locked_genes has duplicates: {locked}")
    missing = [n for n in pb2.REQUIRES_ENERGY if n not in locked]
    if missing:
        raise ValueError(
            f"FD-requires-energy genes {missing} must be in locked_genes until ER's energy cost (J_energy) is wired "
            f"(planform_b2.REQUIRES_ENERGY={list(pb2.REQUIRES_ENERGY)})")


def active_schema(model: str, locked: Sequence[str]):
    """FD shape_schema_b2 rows with locked genes stripped (FD vector order preserved)."""
    lock = set(locked)
    return [g for g in fd_shape_schema(model) if g["name"] not in lock]


def shape_genes(model: str, locked: Sequence[str]):
    """GeneSpecs for the active B2a shape block (locked genes excluded from the encoded vector)."""
    import genome_schema as GS
    check_schema_pin(model)
    check_encoding_pin()
    check_locked_genes(locked, model)
    out = []
    for g in active_schema(model, locked):
        if g["scale"] != "linear":
            raise ValueError(f"FD shape gene {g['name']}: scale {g['scale']!r} (genome stores FD's vector encoding, linear)")
        units = g.get("unit") or ("deg" if g["name"].endswith("_deg") else "x")
        if units in ("-", ""):
            units = "x"
        out.append(GS.GeneSpec(g["name"], SHAPE_B2_BLOCK, float(g["lo"]), float(g["hi"]), "linear",
                               float(g["default"]), units, g.get("meaning") or "", "none"))
    return tuple(out)


def shape_from_values(values: Dict[str, float], genes) -> Dict[str, float]:
    """Physical active-shape dict (GeneSpec.decode = FD GENE_ENCODING), no post-processing."""
    return {g.name: float(values[g.name]) for g in genes}


def locked_defaults(model: str, locked: Sequence[str]) -> Dict[str, float]:
    """FD schema default literals for locked genes (tc = 1.0)."""
    by_name = {g["name"]: float(g["default"]) for g in fd_shape_schema(model)}
    return {n: by_name[n] for n in locked}


def pack_shape_dict(active: Dict[str, float], locked: Sequence[str], model: str) -> Dict[str, float]:
    """Full 11-key physical dict for FD: active genes + locked genes forced to FD defaults."""
    pb2 = planform()
    full = pb2.shape_defaults_b2()
    full.update({k: float(v) for k, v in active.items()})
    full.update(locked_defaults(model, locked))
    return full


def pack_shape_u(active_u: Sequence[float], locked: Sequence[str], model: str, genes=None) -> np.ndarray:
    """Active u-vector (n_s) -> full FD [0,1]^11 with locked slots at identity_u."""
    pb2 = planform()
    genes = genes or shape_genes(model, locked)
    if len(active_u) != len(genes):
        raise ValueError(f"active u length {len(active_u)} != n_active {len(genes)}")
    active_phys = {g.name: g.decode(float(u)) for g, u in zip(genes, active_u)}
    return pb2.encode_shape_b2(model, pack_shape_dict(active_phys, locked, model))


def active_identity_u(model: str, locked: Sequence[str]) -> np.ndarray:
    """FD identity_u(model) with locked genes stripped (use this, not zeros)."""
    pb2 = planform()
    lock = set(locked)
    names = [g["name"] for g in fd_shape_schema(model)]
    idu = np.asarray(pb2.identity_u(model), float)
    return np.array([idu[i] for i, n in enumerate(names) if n not in lock], float)


# ----------------------------------------------------------------------------------------------------- model_version
def fd_b2_model_version(profile_d: Dict) -> str:
    F, esim, _, fb2 = er_modules()
    P = esim.Profile.from_dict(profile_d)
    root, rv2 = F.roots(P)
    return fb2.model_version(B2_FIDELITY, P.aircraft, root, rv2)


def classify_b2_model_version(model: str, mv: Optional[str]) -> str:
    if mv is not None and mv == FD_B2_MODEL_VERSIONS.get(model):
        return "match"
    return "unknown"


def check_b2_model_version(model: str, profile_d: Optional[Dict], mode: str = "warn") -> Dict:
    if mode not in ("warn", "raise", "off"):
        raise ValueError(f"flex.model_version_check must be warn | raise | off, got {mode!r}")
    rec = FD_B2_MODEL_VERSIONS.get(model)
    out = {"model": model, "fidelity": B2_FIDELITY, "recorded": rec, "current": None,
           "match": None, "status": None, "message": None}
    if mode == "off":
        return out
    try:
        cur = fd_b2_model_version(profile_d)
    except Exception as e:
        out.update(status="unverified",
                   message=f"could not compute FD model_version for {model} {B2_FIDELITY} ({type(e).__name__}: {e})")
        if mode == "raise":
            raise RuntimeError(out["message"]) from e
        return out
    out["current"] = cur
    status = classify_b2_model_version(model, cur)
    if status == "match":
        out.update(match=True, status="match")
        return out
    out.update(match=False, status="unknown",
               message=(f"FD model_version for {model} {B2_FIDELITY} is {cur}, genome recorded post_p3b2a {rec}: "
                        "unknown -- re-check phase3_b2a and update shape_b2.FD_B2_MODEL_VERSIONS"))
    if mode == "raise":
        raise RuntimeError(out["message"])
    return out


# ----------------------------------------------------------------------------------------------------- ER profile
def er_profile(config: str, aircraft: str) -> Dict:
    _, _, batch, _ = er_modules()
    path = config if os.path.isabs(config) else os.path.join(TEAM, config)
    cfg = batch.resolve_config(json.load(open(path)), "genome_b2a")
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


def er_has_b2a() -> bool:
    F = er_modules()[0]
    return B2_FIDELITY in getattr(F, "FIDELITIES", ())


def evaluate_b2(profile_d: Dict, gains: Dict[str, float], struct: Dict[str, float], shape: Optional[Dict[str, float]],
                scs_d: Sequence[Dict], *, locked: Sequence[str] = DEFAULT_LOCKED_GENES,
                check_mv: Optional[str] = None, via: str = "auto") -> Dict:
    """One genome at full_a1_b2a. ``shape`` may be active-only or a full dict; locked genes are injected as FD defaults.
    via 'auto': ER evaluate_genome when wired, else FD flexeval_b2. Gate reject = hard fail_cost, not flown."""
    if via not in ("auto", "er", "fd"):
        raise ValueError(f"via must be auto | er | fd, got {via!r}")
    _, esim, _, _ = er_modules()
    model = esim.Profile.from_dict(profile_d).aircraft
    check_locked_genes(locked, model)
    full_shape = None if shape is None else pack_shape_dict(shape, locked, model)
    if via == "er" or (via == "auto" and er_has_b2a()):
        return _evaluate_b2_er(profile_d, gains, struct, full_shape, scs_d, locked, check_mv, model)
    return _evaluate_b2_fd(profile_d, gains, struct, full_shape, scs_d, locked, check_mv)


_OUT_KEYS = ("cost", "status", "terms", "terms_available", "fidelity", "model_version", "feasible", "margins",
             "mass_total_frac", "per_scenario", "shape_genes", "shape_cache_key", "geometry_gate", "planform",
             "structural_model", "tip_bm", "shape_genes_b2", "geometry_gate_b2", "energy", "locked_genes")


def _evaluate_b2_er(profile_d, gains, struct, shape, scs_d, locked, check_mv, model):
    F = er_modules()[0]
    if shape is not None:
        planform().decode_shape_b2(shape, model)
    with _no_bytecode():
        r = F.evaluate_genome(profile_d, gains, struct, list(scs_d), B2_FIDELITY, None, shape=shape)
    if check_mv is not None and r["model_version"] != check_mv:
        raise RuntimeError(f"FD model_version {r['model_version']} != expected {check_mv}")
    out = {k: r.get(k) for k in _OUT_KEYS}
    out["locked_genes"] = list(locked)
    out["evaluator"] = "evolution.fidelity.evaluate_genome(full_a1_b2a)"
    for k in ("J_mass_fd", "mass_credit_clip", "mass_credit_delta"):
        if k in r:
            out[k] = r[k]
    return out


def _evaluate_b2_fd(profile_d, gains, struct, shape, scs_d, locked, check_mv):
    F, esim, _, fb2 = er_modules()
    P = esim.Profile.from_dict(profile_d)
    scs = [esim.Scenario.from_dict(s) for s in scs_d]
    struct = F.struct_from(struct)
    F.check_prepared(P, HOST_FIDELITY)
    root, rv2 = F.roots(P)
    with F._gate(None):
        r = fb2.evaluate(gains, struct, scs, P.aircraft, fidelity=B2_FIDELITY, root=root, root_v2=rv2, profile=P,
                         sim=esim, record=False, blas_threads=1, shape_genome=shape)
    mv = r["model_version"]
    if check_mv is not None and mv != check_mv:
        raise RuntimeError(f"FD model_version {mv} != expected {check_mv}")
    if r["per_scenario"]:
        per = [{k: e[k] for k in F.PER_KEYS if k in e} for e in r["per_scenario"]]
    else:
        per = [{"cost": float(r["cost"]), "status": r["status"], "t_end": 0.0, "not_flown": True} for _ in scs]
    for e in per:
        e["fidelity"], e["model_version"] = B2_FIDELITY, mv
    mg = r.get("margins") or {}
    out = {"cost": float(r["cost"]), "status": r["status"], "terms": dict(r["terms"]),
           "terms_available": list(r.get("terms_available") or []), "fidelity": B2_FIDELITY, "model_version": mv,
           "feasible": r["status"] == "ok",
           "margins": {k: mg.get(k) for k in ("flutter_margin", "div_margin", "reversal_margin")},
           "mass_total_frac": (r.get("mass") or {}).get("total_frac"), "per_scenario": per,
           "shape_genes": r.get("shape_genes"), "shape_cache_key": r.get("shape_cache_key"),
           "geometry_gate": r.get("geometry_gate"), "planform": r.get("planform"),
           "structural_model": r.get("structural_model"),
           "shape_genes_b2": r.get("shape_genes_b2"), "geometry_gate_b2": r.get("geometry_gate_b2"),
           "energy": r.get("energy"), "locked_genes": list(locked)}
    sz = r.get("sizing") or {}
    out["tip_bm"] = {"method": sz.get("tip_bm_method"),
                     "strip_discrete_term": (sz.get("tip_bm_strip_discrete") or {}).get("term")}
    if P.flex_mass_credit_clip and r["per_scenario"] and r.get("mass"):
        fb = F.fd_modules()["fb"]
        F.apply_mass_credit_clip(out, r["mass"], P.flex_mass_credit_clip, fb.StructWeightsV2().w_mass)
    out["evaluator"] = "shape_b2 direct: FD flexeval_b2.evaluate with ER inputs"
    return out


def geometry_rejected(res: Dict) -> bool:
    st = str(res.get("status", ""))
    return st.startswith("geometry_gate")


# ----------------------------------------------------------------------------------------------------- task
def _check_weights(weights: Dict[str, float], profile_d: Dict) -> None:
    want = {"track_alt": 1.0, "effort": float(profile_d.get("w_effort", 2.0)),
            "comfort": float(profile_d.get("w_comfort", 0.0)), "track_heading_rms": float(profile_d.get("w_heading", 0.0)),
            "structural_v2": 1.0}
    got = {k: float(weights.get(k, 0.0)) for k in want}
    if got != want:
        raise ValueError(f"phase3_b2a fitness.weights {got} must equal the ER profile's cost weights {want} "
                         "(full_a1_b2a cost is FD flexeval_b2 + ER sim cost; no parallel formula)")


def build_task_b2(raw: Dict, build_task):
    """Build the phase3_b2a task: phase2_flex controller + structure_v2 via build_task, then append active B2a shape
    genes (locked_genes excluded), resolve init/operators, attach full_a1_b2a evaluator."""
    import adapter
    import block_ops
    orig = copy.deepcopy(raw)
    raw = copy.deepcopy(raw)
    blocks = dict(raw.get("blocks", {}))
    if blocks.pop(SHAPE_B2_BLOCK) is not True:
        raise ValueError("blocks.shape_b2 must be true (FD's B2a genes are evolved as one shape block)")
    if blocks.get("shape_b1"):
        raise ValueError("shape_b1 and shape_b2 cannot both be enabled (one shape block; use shape_b2 for B2a)")
    if not blocks.get("structure_v2"):
        raise ValueError("phase3_b2a needs the structure_v2 block (shape only changes the baseline the 12 genes scale)")
    flex = dict(raw.get("flex", {}))
    fid = flex.pop("fidelity", None)
    if fid != B2_FIDELITY:
        raise ValueError(f"blocks.shape_b2 needs flex.fidelity = {B2_FIDELITY!r}, got {fid!r}")
    if flex.get("asymmetric"):
        raise ValueError("phase3_b2a: flex.asymmetric is not supported (B2a is L = R)")
    mv_mode = flex.get("model_version_check", "warn")
    flex["model_version_check"] = "off"
    b2cfg = {k: v for k, v in dict(raw.pop("shape_b2", {})).items() if not k.startswith("_")}
    bad = set(b2cfg) - {"er_profile_config"}
    if bad:
        raise ValueError(f"unknown shape_b2 keys {sorted(bad)}")
    locked = list(raw.pop("locked_genes", list(DEFAULT_LOCKED_GENES)))
    init_cfg = raw.pop("init", None)
    ops_cfg = raw.pop("operators", None)
    ga_cfg = block_ops.resolve_ga(raw.pop("ga", None))
    base = build_task(dict(raw, blocks=blocks, flex=flex))
    model = base.profile.jsbsim_model
    genes = shape_genes(model, locked)
    spec = base.spec
    spec.genes = list(spec.genes) + list(genes)
    spec.enabled_blocks = tuple(spec.enabled_blocks) + (SHAPE_B2_BLOCK,)
    init = adapter.resolve_init(init_cfg, spec, base.profile)
    ops = block_ops.resolve_operators(ops_cfg, spec, shape_crossover=ga_cfg["shape_crossover"])
    # Seed shape init centre on FD identity_u (locked stripped), not a hard-coded 0.5 / zero vector.
    if ops:
        sp = ops["shape_ops"]
        idu = active_identity_u(model, locked)
        if len(idu) != len(sp.idx):
            raise ValueError(f"active identity_u length {len(idu)} != shape block {len(sp.idx)}")
        # Replace ShapeOps.default so identity_u() == FD identity (asymmetric ranges).
        phys = [g.decode(float(u)) for g, u in zip(genes, idu)]
        ops["shape_ops"] = block_ops.ShapeOps(
            idx=sp.idx, lo=sp.lo, hi=sp.hi, log=sp.log, default=phys,
            init_sigma_frac=sp.init_sigma_frac, mut_sigma_frac=sp.mut_sigma_frac, mutation_rate=sp.mutation_rate)
    prof_d = er_profile(b2cfg.get("er_profile_config", "evolution/configs/phase2_smoke_p25.json"), model)
    _check_weights(base.fitness.weights, prof_d)
    warnings = list(base.warnings)
    gb = prof_d.get("gain_bounds") or {}
    for g in spec.genes:
        if g.name in gb and (g.min, g.max) != tuple(float(x) for x in gb[g.name]):
            warnings.append(f"controller gene {g.name} range [{g.min}, {g.max}] differs from ER profile {gb[g.name]}")
    mv = check_b2_model_version(model, prof_d, mv_mode)
    if mv["message"]:
        warnings.append("FD MODEL_VERSION: " + mv["message"])
    if not er_has_b2a():
        warnings.append("ER fidelity.FIDELITIES has no full_a1_b2a yet: phase3_b2a evaluator uses via='fd' "
                        "(flexeval_b2.evaluate with ER inputs)")
    flex_consts = dict(base.flex_constants, fidelity=B2_FIDELITY, margin_gate=1.0, fd_model_version=mv["current"],
                       fd_model_version_recorded=mv["recorded"], locked_genes=list(locked))
    return TaskB2(base.name, spec, base.fitness, base.scenarios, base.profile, orig, warnings, base.conditions, {},
                  flex_consts, {}, init, er_profile_d=prof_d, operators=ops, shape_gene_specs=tuple(genes),
                  b2_model_version=mv, locked_genes=list(locked),
                  ga_overrides={k: v for k, v in ga_cfg.items() if k == "elite" and v is not None})


import adapter as _adapter  # noqa: E402


@dataclasses.dataclass
class TaskB2(_adapter.Task):
    """phase3_b2a task: ER scenarios + full_a1_b2a evaluator. Locked genes excluded from the vector; injected at eval."""
    er_profile_d: Dict = dataclasses.field(default_factory=dict)
    operators: Dict = dataclasses.field(default_factory=dict)
    shape_gene_specs: tuple = ()
    b2_model_version: Dict = dataclasses.field(default_factory=dict)
    locked_genes: List[str] = dataclasses.field(default_factory=list)
    ga_overrides: Dict = dataclasses.field(default_factory=dict)

    def make_scenarios(self, n: int, seed: int):
        _, esim, _, _ = er_modules()
        return list(esim.make_scenarios(n, seed, esim.Profile.from_dict(self.er_profile_d)))

    def split(self, values: Dict[str, float]):
        blk = {g.name: g.block for g in self.spec.genes}
        gains = {k: float(v) for k, v in values.items() if k in blk and blk[k] not in ("structure_v2", SHAPE_B2_BLOCK)}
        struct = {k: float(v) for k, v in values.items() if blk.get(k) == "structure_v2"}
        active = shape_from_values({k: v for k, v in values.items() if blk.get(k) == SHAPE_B2_BLOCK},
                                   self.shape_gene_specs)
        shape = pack_shape_dict(active, self.locked_genes, self.profile.jsbsim_model)
        return gains, struct, shape

    def evaluate(self, gains, scenarios, record=None):
        g, s, sh = self.split(gains)
        scs = [sc if isinstance(sc, dict) else sc.to_dict() for sc in scenarios]
        # shape already packed (full dict); pass locked=[] to avoid double-inject, but still record locked list
        r = evaluate_b2(self.er_profile_d, g, s, sh, scs, locked=self.locked_genes)
        # evaluate_b2 re-packs; locked already at defaults in sh — pack_shape_dict is idempotent for full dicts
        t = r["terms"]
        r["objectives"] = {CONTROLLER_OBJ[k]: float(t[k]) for k in CONTROLLER_OBJ}
        r["objectives"]["structural_v2"] = float(sum(v for k, v in t.items() if k not in RIGID_TERMS))
        r["fd_struct_terms"] = {k: float(v) for k, v in t.items() if k not in RIGID_TERMS}
        r["J_wing_tip_bm_limit"] = float(t.get("J_wing_tip_bm_limit", 0.0))
        r["struct_v2_source"] = "fd"
        r["locked_genes"] = list(self.locked_genes)
        r["shape_genes_full"] = sh
        if geometry_rejected(r):
            r["violation"] = r["cost"]
        return r
