#!/usr/bin/env python3
"""Evolve altitude-hold controllers for several JSBSim aircraft in one batch.

    cd flight_sim_3d              # (team layout: cd <team folder>)
    python -m evolution.batch --config evolution/configs/bench_baseline.json
    python -m evolution.batch --resume evolution/runs/<run_id>      # continue a killed run

Parallelism: ONE process pool of `workers` (default = usable cores) is shared by
all aircraft. Each aircraft's GA runs in a lightweight driver thread in the
parent and submits fine-grained tasks (one individual x one scenario) to that
pool, so there are never more than `workers` JSBSim processes, and a
generation barrier for one aircraft (stragglers, a slow jet) is filled with
other aircraft's work. See README.md "Parallelism model".

Determinism: every aircraft has its own numpy Generator; results are collected
by key, not by completion order; the cache only stores deterministic results.
So worker count, schedule, cache state and kill/resume do not change results.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import datetime as _dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import cache as cache_mod
from . import eval as eval_mod
from . import fidelity as fid_mod
from . import ga, genome, runinfo, sim, trajectory

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_LEGACY_SOURCE_REPO = os.path.dirname(os.path.dirname(PKG_DIR))   # outside the push layout: set $EVOLUTION_SOURCE_REPO


def _default_source_repo() -> str:
    """Repo whose git sha is recorded. 1) $EVOLUTION_SOURCE_REPO; 2) repo-relative (push layout
    <repo>/flight_sim_3d/evolution -> <repo>, recognised by its flight_sim/ prototype dir); 3) the legacy sandbox clone
    (team layout <team>/evolution, whose grandparent is not the repo: falls back to that grandparent; set
    $EVOLUTION_SOURCE_REPO to the git clone there)."""
    env = os.environ.get("EVOLUTION_SOURCE_REPO")
    if env:
        return env
    repo = os.path.dirname(os.path.dirname(PKG_DIR))
    if os.path.isdir(os.path.join(repo, "flight_sim")):
        return repo
    return _LEGACY_SOURCE_REPO


DEFAULT_SOURCE_REPO = _default_source_repo()

DEFAULTS: Dict = {
    "run_id": None,                 # default: <config-stem>-<hash of resolved config>  (same command => resume)
    "runs_dir": "runs",             # relative paths are relative to the evolution/ package dir
    "cache": {"enabled": True, "path": "cache/evals.sqlite"},
    "workers": 0,                   # 0 = all usable cores
    "schedule": "concurrent",       # "concurrent" (all aircraft share the pool) | "sequential" (one aircraft at a time)
    "seed": 1,                      # GA seed (each aircraft gets this unless it sets its own "seed")
    "scenario_seed": None,          # None = seed (as in the original evolve.py)
    "scenarios": 3,
    # ga.shape_crossover (genome_kind phase3_b1 only): "block" (default: whole-block crossover, Genome's B1 spec) |
    # "uniform" (opt-in tweaked preset = Genome's phase3_b1_x: per-gene uniform crossover inside the shape block, see
    # ga.crossover_blocks_uniform_shape). At "block" it is dropped from the resolved config (old run ids / resume unchanged).
    "ga": {"pop_size": 24, "generations": 15, "elite": 2, "selection_p": 0.2, "crossover": "uniform",
           "blx_alpha": 0.3, "mutation_rate": 0.15, "mutation_sigma": 0.08, "mutation_mode": "gauss",
           "shape_crossover": "block"},
    "trajectories": {"generations": "auto", "scenario": 0, "sample_hz": 30},   # auto = [0, (G-1)//2, G-1]
    "metrics": {"band_ft": 20.0, "hold_after_s": 20.0},
    "source_repo": DEFAULT_SOURCE_REPO,
    "profiles": {"baseline": {}},   # name -> sim.Profile overrides; {} = original c172x constants
    "aircraft": [{"name": "c172x", "profile": "baseline"}],
    # structural fidelity (evolution/fidelity.py): rigid | reduced (FD flex v1 on the projected v2 genome) | full (FD flex
    # v2, flexbody) | full_a1 (FD P3-A1, flexbody_a1: 64-strip wings, 4b+3t+2ip; opt-in, separate cache/pins) |
    # full_a1_b1 (FD P3-B1: A1 host + 6 planform shape genes, flexeval_b1; opt-in, separate cache/pins)
    "fidelity": "rigid",
    # screen everyone at `screen`, re-score the top_k by screen cost + all elites at `fidelity` (see README)
    "multi_fidelity": {"enabled": False, "screen": "reduced", "top_k": 4, "min_full_frac": None, "mid_k": None},
    # per-aircraft overrides of evolution.fidelity.DEFAULT_PER_AIRCRAFT: {name: {reduced_gate, min_full_frac}}
    "fidelity_per_aircraft": {},
    "struct_genes": False,          # append FD's STRUCT_SCHEMA genes (stiffness_scale, torsion_bend_ratio, zeta, nonstruct_scale)
    "viz": "off",                   # on = record a trajectory on EVERY evaluation + live best-of-generation file (slow path)
    # ---- Phase 2 (all default-off; omitted from the run identity while at their defaults, so old run ids are unchanged)
    "struct_asymmetric": False,     # with struct_genes: + FD's 2 optional asymmetry genes (14 struct genes)
    # generation 0: "uniform" (= ga.generation_zero) | "baseline": struct genes = encode(FD baseline) + N(0, sigma),
    # clipped to [0, 1], after the identical uniform draw (= Genome init_pop.generation_zero); sigma in [0.10, 0.15]
    "init": {"mode": "uniform", "sigma": 0.10, "blocks": ["struct"]},
    # per-aircraft multi-fidelity overrides {name: {enabled, screen, top_k, min_full_frac, mid_k}} on top of multi_fidelity
    "multi_fidelity_per_aircraft": {},
    # {aircraft: {fidelity: model_version}}: refuse to evaluate / cache / resume unless FD's model_version matches;
    # every reduced/full fidelity an aircraft uses must be pinned once the aircraft appears here
    "pin_model_version": {},
    # ---- Phase 3 B1 (default-off; omitted from the run identity while None, so old run ids are unchanged)
    # genome_kind 'phase3_b1': controller (8, v4) | FD's 12 structure genes | FD's 6 planform shape genes (group 'shape');
    # needs fidelity full_a1_b1 (rigid screens only), struct_genes, init.mode 'baseline'. shape_ops: operator settings
    # (None -> SHAPE_OPS_DEFAULT, filled in by resolve_config so config.json records them).
    "genome_kind": None,
    "shape_ops": None,
    # ---- Phase 3 B2a (default-off; omitted from the run identity while None)
    # genome_kind 'phase3_b2a': controller (8) | 12 structure genes | FD's B2a shape block (6 B1 + 5 B2a, per-aircraft
    # ranges) minus shape_locked (None -> wing_tc_root_scale / wing_tc_tip_ratio = Genome's phase3_b2a: 29 genes; locked
    # genes are not in the GA vector, FD decode fills their default 1.0). Needs fidelity full_a1_b2a.
    # energy_cost (full_a1_b2a only): True adds the Evolution-side J_energy + J_speed_guard (INTERFACE_v2 15.10.3,
    # outside TERM_KEYS) to the cost; required before a thickness gene may be unlocked.
    "shape_locked": None,
    "energy_cost": None,
}
OPTIONAL_DEFAULTS = {"struct_asymmetric": False, "init": {"mode": "uniform", "sigma": 0.10, "blocks": ["struct"]},
                     "multi_fidelity_per_aircraft": {}, "pin_model_version": {}, "genome_kind": None, "shape_ops": None,
                     "shape_locked": None, "energy_cost": None}
GENOME_KINDS = ("phase3_b1", "phase3_b2a")
SHAPED_KINDS = {"phase3_b1": "full_a1_b1", "phase3_b2a": "full_a1_b2a"}
B2A_DEFAULT_LOCKED = ["wing_tc_root_scale", "wing_tc_tip_ratio"]   # planform_b2.REQUIRES_ENERGY (Genome phase3_b2a)
B2A_THICKNESS = ("wing_tc_root_scale", "wing_tc_tip_ratio")
B2A_NAMES = ("wing_dihedral_delta_deg", "wing_tc_root_scale", "wing_tc_tip_ratio", "wing_camber_root_delta_pct",
             "wing_camber_tip_delta_pct")
# Genome Architect's B1 operator spec (2026-10-06): sigma = 0.25 x half-range in FD's encoded space (ln x for the chord
# tapers), gen-0 around the identity planform, whole-block crossover controller | structure | shape
SHAPE_OPS_DEFAULT = {"init": "identity", "init_sigma_half_range": 0.25, "mutation_sigma_half_range": 0.25,
                     "mutation_rate": None, "log_genes": ["wing_chord_taper_1", "wing_chord_taper_2", "wing_chord_taper_3"],
                     "crossover": "blocks"}
PIN_PLACEHOLDER = "PENDING-FD-NEW-MODEL-VERSION"
MF_KEYS = {"enabled", "screen", "top_k", "min_full_frac", "mid_k"}
AIRCRAFT_KEYS = {"name", "profile", "overrides", "seed"}
# execution-only settings: cannot change results, so they are excluded from the run-id hash and the resume check
EXEC_KEYS = ("workers", "schedule", "cache", "viz")


def identity(cfg: Dict) -> Dict:
    return {k: v for k, v in cfg.items() if k not in EXEC_KEYS and k != "run_id"
            and not (k in OPTIONAL_DEFAULTS and v == OPTIONAL_DEFAULTS[k])}


# --------------------------------------------------------------------------- config
def _merge(base: Dict, over: Dict, path: str = "") -> Dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):
            continue  # "_comment" etc.
        if k not in base and path not in ("profiles",):
            raise ValueError(f"unknown config key {path + k!r}")
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k not in ("profiles", "fidelity_per_aircraft",
                                                                                 "multi_fidelity_per_aircraft", "pin_model_version",
                                                                                 "shape_ops"):
            out[k] = _merge(base[k], v, path + k + ".")
        else:
            out[k] = copy.deepcopy(v)
    return out


def _abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(PKG_DIR, p))


def resolve_config(user: Dict, stem: str = "batch") -> Dict:
    cfg = _merge(DEFAULTS, user)
    cfg["profiles"] = {"baseline": {}, **cfg["profiles"]}
    if cfg["scenario_seed"] is None:
        cfg["scenario_seed"] = cfg["seed"]
    if cfg["ga"]["elite"] >= cfg["ga"]["pop_size"]:
        raise ValueError("ga.elite must be smaller than ga.pop_size")
    _resolve_shape_crossover(cfg)
    if cfg["schedule"] not in ("concurrent", "sequential"):
        raise ValueError("schedule must be 'concurrent' or 'sequential'")
    if cfg["fidelity"] not in fid_mod.FIDELITIES:
        raise ValueError(f"fidelity must be one of {fid_mod.FIDELITIES}")
    if cfg["viz"] not in ("on", "off"):
        raise ValueError("viz must be 'on' or 'off'")
    mf = _validate_mf(cfg["multi_fidelity"], cfg)
    _validate_phase2(cfg)
    for nm, d in (cfg.get("fidelity_per_aircraft") or {}).items():
        if set(d) - {"reduced_gate", "min_full_frac"}:
            raise ValueError(f"fidelity_per_aircraft.{nm}: unknown keys {sorted(set(d) - {'reduced_gate', 'min_full_frac'})}")
    if cfg["struct_genes"] and cfg["fidelity"] == "rigid" and not mf["enabled"]:
        raise ValueError("struct_genes only matter at reduced/full fidelity")
    mfa = cfg["multi_fidelity_per_aircraft"] or {}
    unknown_ac = set(mfa) - {(a if isinstance(a, str) else a["name"]) for a in cfg["aircraft"]}
    if unknown_ac:
        raise ValueError(f"multi_fidelity_per_aircraft: aircraft not in the batch: {sorted(unknown_ac)}")
    seen = set()
    acs = []
    for a in cfg["aircraft"]:
        a = {"name": a} if isinstance(a, str) else dict(a)
        if set(a) - AIRCRAFT_KEYS:
            raise ValueError(f"unknown aircraft keys {sorted(set(a) - AIRCRAFT_KEYS)}")
        if a["name"] in seen:
            raise ValueError(f"aircraft {a['name']} listed twice (use separate batch configs to compare profiles)")
        seen.add(a["name"])
        pname = a.get("profile", "baseline")
        if pname not in cfg["profiles"]:
            raise ValueError(f"aircraft {a['name']}: unknown profile {pname!r}")
        prof = sim.Profile.from_dict({**cfg["profiles"][pname], **a.get("overrides", {}), "aircraft": a["name"]})
        genome.make_schema(prof.gain_bounds, prof.gene_kinds, prof.heading_hold)  # validate
        if cfg.get("genome_kind") in SHAPED_KINDS and not prof.heading_hold:
            raise ValueError(f"aircraft {a['name']}: genome_kind {cfg['genome_kind']} = phase2_flex controller (8 genes): needs heading_hold")
        acs.append({"name": a["name"], "profile": pname, "overrides": a.get("overrides", {}),
                    "seed": int(a.get("seed", cfg["seed"])), "resolved_profile": prof.to_dict()})
        if a["name"] in mfa:   # only then (old configs keep their aircraft entries / run ids)
            if set(mfa[a["name"]]) - MF_KEYS:
                raise ValueError(f"multi_fidelity_per_aircraft.{a['name']}: unknown keys {sorted(set(mfa[a['name']]) - MF_KEYS)}")
            base = {k: v for k, v in cfg["multi_fidelity"].items() if k != "ladder"}
            acs[-1]["multi_fidelity"] = _validate_mf({**base, **copy.deepcopy(mfa[a["name"]])}, cfg,
                                                     f"multi_fidelity_per_aircraft.{a['name']}")
    cfg["aircraft"] = acs
    G = cfg["ga"]["generations"]
    tg = cfg["trajectories"]["generations"]
    gens = sorted({0, (G - 1) // 2, G - 1}) if tg == "auto" else sorted({int(g) for g in tg if 0 <= int(g) < G} | {G - 1})
    cfg["trajectories"]["generations_resolved"] = gens
    if cfg["run_id"] is None:
        h = hashlib.sha256(json.dumps(identity(cfg), sort_keys=True).encode()).hexdigest()[:8]
        cfg["run_id"] = f"{stem}-{h}"
    return cfg


def _resolve_shape_crossover(cfg: Dict) -> None:
    """ga.shape_crossover: 'block' (default) is removed from the resolved config, so every config without the option (or
    with it at the default) resolves exactly as before (same identity / run id / resume check, GAConfig default 'block').
    'uniform' (tweaked preset) needs genome_kind phase3_b1 and stays in the resolved config (part of the run identity)."""
    sx = cfg["ga"].get("shape_crossover", "block")
    if sx not in ga.SHAPE_CROSSOVERS:
        raise ValueError(f"ga.shape_crossover must be one of {ga.SHAPE_CROSSOVERS}, got {sx!r}")
    if sx == "block":
        cfg["ga"].pop("shape_crossover", None)
    elif cfg.get("genome_kind") not in SHAPED_KINDS:
        raise ValueError("ga.shape_crossover 'uniform' needs genome_kind 'phase3_b1' / 'phase3_b2a' (it acts on the shape block)")


def _validate_mf(mf: Dict, cfg: Dict, where: str = "multi_fidelity") -> Dict:
    if mf["enabled"]:
        scr = [mf["screen"]] if isinstance(mf["screen"], str) else list(mf["screen"])
        rk = fid_mod.RANK
        if (not scr or any(f not in fid_mod.FIDELITIES for f in scr)
                or any(rk[a] >= rk[b] for a, b in zip(scr + [cfg["fidelity"]], scr[1:] + [cfg["fidelity"]]))):
            raise ValueError(f"{where}.screen: a fidelity or an ascending list of fidelities, all below `fidelity` "
                             "(the authoritative one), e.g. 'rigid', 'reduced' or ['rigid', 'reduced'] for fidelity 'full'; "
                             "'rigid' for fidelity 'full_a1'")
        mf["ladder"] = scr + [cfg["fidelity"]]
        if mf.get("min_full_frac") is not None and not 0.0 <= float(mf["min_full_frac"]) <= 1.0:
            raise ValueError(f"{where}.min_full_frac must be in [0, 1] (or null = per-aircraft default)")
        if not (isinstance(mf["top_k"], int) and 1 <= mf["top_k"] <= cfg["ga"]["pop_size"]):
            raise ValueError(f"{where}.top_k must be an int in [1, pop_size]")
    else:
        mf.pop("ladder", None)
    return mf


def _validate_phase2(cfg: Dict) -> None:
    ini = cfg["init"]
    if set(ini) - {"mode", "sigma", "blocks"}:
        raise ValueError(f"init: unknown keys {sorted(set(ini) - {'mode', 'sigma', 'blocks'})}")
    if ini["mode"] not in ("uniform", "baseline"):
        raise ValueError("init.mode must be 'uniform' or 'baseline'")
    if ini["mode"] == "baseline":
        if not cfg["struct_genes"]:
            raise ValueError("init.mode 'baseline' seeds the struct genes: needs struct_genes")
        if list(ini["blocks"]) != ["struct"]:
            raise ValueError("init.blocks: only ['struct'] (FD's structure genes) can be seeded")
        if not 0.10 <= float(ini["sigma"]) <= 0.15:
            raise ValueError("init.sigma must be in [0.10, 0.15] (normalized units)")
    if cfg["struct_asymmetric"] and not cfg["struct_genes"]:
        raise ValueError("struct_asymmetric needs struct_genes")
    _validate_p3b1(cfg)
    for ac, pins in (cfg["pin_model_version"] or {}).items():
        if not isinstance(pins, dict) or set(pins) - set(fid_mod.FIDELITIES) or not all(isinstance(v, str) for v in pins.values()):
            raise ValueError(f"pin_model_version.{ac}: {{fidelity: model_version string}}")


def _validate_p3b1(cfg: Dict) -> None:
    """genome_kind 'phase3_b1' (P3-B1): controller | 12 structure genes (P2.5, symmetric) | FD's 6 shape genes."""
    kind = cfg.get("genome_kind")
    if cfg.get("energy_cost") is not None:
        if not isinstance(cfg["energy_cost"], bool):
            raise ValueError("energy_cost must be true / false / null")
        if cfg["fidelity"] != fid_mod.B2:
            raise ValueError(f"energy_cost is only defined for fidelity {fid_mod.B2!r}")
        if cfg["energy_cost"] is False:
            cfg["energy_cost"] = None          # false == default (identity / run id unchanged)
    if cfg.get("shape_locked") is not None and kind != "phase3_b2a":
        raise ValueError("shape_locked needs genome_kind 'phase3_b2a'")
    if kind is None:
        if cfg.get("shape_ops") is not None:
            raise ValueError("shape_ops needs genome_kind 'phase3_b1'")
        return
    if kind not in GENOME_KINDS:
        raise ValueError(f"genome_kind must be null or one of {GENOME_KINDS}, got {kind!r}")
    if not cfg["struct_genes"] or cfg["struct_asymmetric"]:
        raise ValueError("genome_kind phase3_b1: struct_genes true and struct_asymmetric false (12 P2.5 structure genes)")
    if kind in SHAPED_KINDS and cfg["fidelity"] != SHAPED_KINDS[kind]:
        raise ValueError(f"genome_kind {kind}: fidelity must be {SHAPED_KINDS[kind]!r} (the fidelity that consumes its shape genes)")
    if kind == "phase3_b2a":
        lk = B2A_DEFAULT_LOCKED if cfg.get("shape_locked") is None else list(cfg["shape_locked"])
        if set(lk) - set(B2A_NAMES):
            raise ValueError(f"shape_locked: only B2a genes can be locked, got {sorted(set(lk) - set(B2A_NAMES))}")
        if set(B2A_THICKNESS) - set(lk) and not cfg.get("energy_cost"):
            raise ValueError("shape_locked: thickness genes (wing_tc_*) require energy_cost true (FD requires='energy_cost')")
        if cfg.get("shape_locked") is not None and sorted(lk) == sorted(B2A_DEFAULT_LOCKED):
            cfg["shape_locked"] = None        # default spelled out == default (identity unchanged)
    if cfg["init"]["mode"] != "baseline":
        raise ValueError("genome_kind phase3_b1: init.mode 'baseline' (structure block seeded at FD's baseline)")
    for where, mf in [("multi_fidelity", cfg["multi_fidelity"])] + [
            (f"multi_fidelity_per_aircraft.{n}", m) for n, m in (cfg.get("multi_fidelity_per_aircraft") or {}).items()]:
        if mf.get("enabled", cfg["multi_fidelity"]["enabled"]):
            scr = mf.get("screen", cfg["multi_fidelity"]["screen"])
            scr = [scr] if isinstance(scr, str) else list(scr)
            if any(f != "rigid" for f in scr):
                raise ValueError(f"{where}.screen: genome_kind {kind} allows only 'rigid' screens below {cfg['fidelity']} "
                                 "(rigid ignores the shape; reduced / full / full_a1 cannot fly a shaped planform)")
    ops = dict(SHAPE_OPS_DEFAULT, **(cfg.get("shape_ops") or {}))
    bad = set(ops) - set(SHAPE_OPS_DEFAULT)
    if bad:
        raise ValueError(f"shape_ops: unknown keys {sorted(bad)}")
    if ops["init"] != "identity" or ops["crossover"] != "blocks":
        raise ValueError("shape_ops: init 'identity' and crossover 'blocks' (Genome's B1 spec) are the only options")
    for k in ("init_sigma_half_range", "mutation_sigma_half_range"):
        if not 0.0 < float(ops[k]) <= 1.0:
            raise ValueError(f"shape_ops.{k} must be in (0, 1] (fraction of the half-range in encoded space)")
    if ops["mutation_rate"] is None:
        ops["mutation_rate"] = cfg["ga"]["mutation_rate"]
    if not 0.0 <= float(ops["mutation_rate"]) <= 1.0:
        raise ValueError("shape_ops.mutation_rate must be in [0, 1]")
    names = set(SHAPE_OPS_DEFAULT["log_genes"]) | {"wing_twist_mid_deg", "wing_twist_tip_deg", "wing_sweep_qc_delta_deg"}
    if kind == "phase3_b2a":
        names |= set(B2A_NAMES)
    if set(ops["log_genes"]) - names:
        raise ValueError(f"shape_ops.log_genes: unknown shape genes {sorted(set(ops['log_genes']) - names)}")
    if cfg["ga"]["mutation_mode"] != "gauss":
        raise ValueError("genome_kind phase3_b1: ga.mutation_mode 'gauss'")
    cfg["shape_ops"] = ops


def shape_spec(schema, groups, shape_ops: Dict) -> "ga.ShapeSpec":
    """ga.ShapeSpec of the shape block (contiguous tail, FD order) from the run schema and the resolved shape_ops."""
    idx = [j for j, g in enumerate(groups) if g == "shape"]
    if idx != list(range(len(groups) - len(idx), len(groups))):
        raise ValueError("shape genes must be the contiguous tail block of the genome")
    sch = [schema[j] for j in idx]
    return ga.ShapeSpec(idx=idx, lo=[g.min for g in sch], hi=[g.max for g in sch],
                        log=[g.name in shape_ops["log_genes"] for g in sch], default=[g.default for g in sch],
                        init_sigma_frac=float(shape_ops["init_sigma_half_range"]),
                        mut_sigma_frac=float(shape_ops["mutation_sigma_half_range"]),
                        mutation_rate=float(shape_ops["mutation_rate"]))


def gene_blocks(groups) -> List[List[int]]:
    """[controller idx, structure idx, shape idx] (non-empty blocks only, genome order)."""
    out = []
    for grp in ("gains", "struct", "shape"):
        idx = [j for j, g in enumerate(groups) if g == grp]
        if idx:
            out.append(idx)
    return out


def check_pins(cfg: Dict, name: str, fids, mv: Dict, profile_d: Optional[Dict] = None) -> Dict[str, str]:
    """Refuse (SystemExit, before any evaluation, cache write or checkpoint) unless every reduced/full fidelity this
    aircraft uses is pinned and equals FD's model_version. Pins are FD's own strings (v2_results/model_versions_*.json),
    i.e. at FD's default reduced gate 0.9; evolution's reduced string also hashes the per-aircraft reduced_gate (1.0 on
    swept wings), so a reduced pin is checked against FD's default-gate string of the same model and the run's own
    gated string is what the cache guard then allows. Returns {fidelity: allowed run model_version}. Aircraft absent
    from pin_model_version: no check, {}."""
    pins = (cfg.get("pin_model_version") or {}).get(name)
    if pins is None:
        return {}
    for f in fids:
        if f != "rigid" and f not in pins:
            raise SystemExit(f"[{name}] pin_model_version has no entry for fidelity {f!r}; refusing to evaluate or cache")
    allowed = {}
    for f, want in pins.items():
        got = mv.get(f)
        if got is None:
            raise SystemExit(f"[{name}] pinned fidelity {f!r} is not used by this run (ladder {list(fids)})")
        fd_got = got
        if f == "reduced" and profile_d is not None:
            fd_got = fid_mod.model_version(profile_d, "reduced", None)   # FD's default gate (what FD publishes)
        if want == PIN_PLACEHOLDER or want not in (fd_got, got):
            why = ("placeholder: fill in Flight Dynamics' new model_version" if want == PIN_PLACEHOLDER
                   else "Flight Dynamics' model changed")
            raise SystemExit(f"[{name}] model_version pin mismatch for {f}: pinned {want!r}, FD reports {fd_got!r}"
                             + (f" (run string {got!r})" if got != fd_got else "") + f" ({why}); "
                             "refusing to evaluate, write cache entries or resume")
        allowed[f] = got
    return allowed


def model_versions_report(cfg: Dict) -> Dict:
    """{aircraft: {fidelity: {current, pinned, match}}} for every fidelity of each aircraft's ladder (in-process, read-only;
    the same eval.describe the batch runs first). Use it to fill pin_model_version once FD sends its new strings: the
    reduced string includes the per-aircraft reduced_gate (FD's default 0.9; 1.0 on swept wings in evolution)."""
    rep = {}
    for ac in cfg["aircraft"]:
        mf = ac.get("multi_fidelity") or cfg.get("multi_fidelity") or {}
        fids = list(mf["ladder"]) if mf.get("enabled") else [cfg["fidelity"]]
        gate = fid_mod.per_aircraft(ac["name"], cfg.get("fidelity_per_aircraft"))["reduced_gate"]
        d = eval_mod.describe(ac["resolved_profile"], fids, gate)
        pins = (cfg.get("pin_model_version") or {}).get(ac["name"]) or {}
        rep[ac["name"]] = {}
        for f in fids:
            cur = d["model_version"].get(f)
            fd = fid_mod.model_version(ac["resolved_profile"], "reduced", None) if f == "reduced" and cur else cur
            rep[ac["name"]][f] = {"current": cur, **({"fd_default_gate": fd} if fd != cur else {}), "pinned": pins.get(f),
                                  "match": (pins.get(f) in (cur, fd)) if f in pins else None}
        if d.get("error"):
            rep[ac["name"]]["error"] = d["error"]
    return rep


def git_sha(repo: str) -> Dict:
    try:
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
        branch = subprocess.run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
        return {"sha": sha, "dirty": dirty, "branch": branch, "repo": repo}
    except Exception as e:  # noqa: BLE001
        return {"sha": "unknown", "dirty": None, "branch": None, "repo": repo, "error": str(e)}


def usable_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# --------------------------------------------------------------------------- io helpers
def _atomic_json(path: str, obj, indent=None):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- runner
def shape_locked_of(cfg: Dict) -> List[str]:
    """phase3_b2a: genes excluded from the GA vector (FD default injected by decode)."""
    return list(B2A_DEFAULT_LOCKED if cfg.get("shape_locked") is None else cfg["shape_locked"])


def full_schema(prof: "sim.Profile", struct_genes: bool, asymmetric: bool = False, genome_kind: Optional[str] = None,
                shape_locked: Optional[Sequence[str]] = None):
    """(schema, groups): controller genes (6, +2 heading) then, with struct_genes, FD's v2 struct genes (12, 14 asym);
    genome_kind 'phase3_b1' appends FD's 6 P3-B1 shape genes (group 'shape', FD order, FD [0,1] linear storage)."""
    sch = genome.make_schema(prof.gain_bounds, prof.gene_kinds, prof.heading_hold)
    groups = ["gains"] * len(sch)
    if struct_genes:
        st = fid_mod.struct_schema(asymmetric)
        sch, groups = sch + st, groups + ["struct"] * len(st)
    if genome_kind == "phase3_b1":
        sh = fid_mod.shape_schema()
        sch, groups = sch + sh, groups + ["shape"] * len(sh)
    elif genome_kind == "phase3_b2a":   # per-aircraft B2 ranges; locked genes excluded
        sh = fid_mod.shape_schema_b2(prof.aircraft, B2A_DEFAULT_LOCKED if shape_locked is None else shape_locked)
        sch, groups = sch + sh, groups + ["shape"] * len(sh)
    return sch, groups


def seed_generation_zero(pop: np.ndarray, rng: np.random.Generator, schema, groups, init: Dict) -> np.ndarray:
    """init.mode 'baseline' (Phase 2): after the unchanged uniform draw, the struct genes become encode(FD baseline) +
    sigma * N(0, 1), clipped to [0, 1] (Genome init_pop.generation_zero; mutation/crossover still reach the full
    ranges). 'uniform' (default): pop returned untouched, no RNG draws (bit-identical to earlier runs)."""
    if (init or {}).get("mode", "uniform") != "baseline":
        return pop
    idx = [j for j, g in enumerate(groups) if g == "struct"]
    if not idx:
        return pop
    base = fid_mod.baseline_u([schema[j] for j in idx])
    noise = rng.standard_normal((pop.shape[0], len(idx)))
    pop = pop.copy()
    pop[:, idx] = np.clip(base + float(init["sigma"]) * noise, 0.0, 1.0)
    return pop


def spearman(a, b) -> Optional[float]:
    """Spearman rank correlation with average ranks for ties (scipy.stats.spearmanr equivalent); None if n < 3."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None

    def rk(x):
        o = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[o] = np.arange(len(x), dtype=float)
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    ra, rb = rk(a), rk(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _genes_at_bound(schema, norm) -> Dict[str, str]:
    """min/max edge hits; a log0 gene inside its zero band is reported as 'zero' (integrator off), not 'min'."""
    out = {}
    for gn, v in zip(schema, norm):
        if gn.kind == "log0" and v <= gn.zero_band:
            out[gn.name] = "zero"
        elif gn.kind == "log0" and v < gn.zero_band + 0.02:
            out[gn.name] = "min"
        elif gn.kind != "log0" and v < 0.02:
            out[gn.name] = "min"
        elif v > 0.98:
            out[gn.name] = "max"
    return out


class Batch:
    def __init__(self, cfg: Dict, log=print):
        self.cfg = cfg
        self.log = log
        self.run_dir = os.path.join(_abspath(cfg["runs_dir"]), cfg["run_id"])
        self.ck_dir = os.path.join(self.run_dir, "checkpoints")
        self.traj_dir = os.path.join(self.run_dir, "trajectories")
        self.jsbsim_version = cache_mod.jsbsim_version()
        self.code_sha = cache_mod.code_sha()
        self.git = git_sha(cfg["source_repo"])
        self.workers = cfg["workers"] or usable_cores()
        self.schedule = cfg["schedule"]
        self.cache = cache_mod.EvalCache(_abspath(cfg["cache"]["path"]) if cfg["cache"]["enabled"] else None)
        self.hist_lock = threading.Lock()
        self.rows_lock = threading.Lock()
        self.session = 0
        self.viz = cfg.get("viz", "off")
        self.fidelity = cfg.get("fidelity", "rigid")
        self.mf = cfg.get("multi_fidelity") or {"enabled": False}
        self.info: Dict[str, Dict] = {}

    # ---- run directory / provenance
    def _prepare_run_dir(self):
        os.makedirs(self.ck_dir, exist_ok=True)
        cpath = os.path.join(self.run_dir, "config.json")
        prov = {"seed": self.cfg["seed"], "scenario_seed": self.cfg["scenario_seed"], "git": self.git,
                "git_sha": self.git["sha"], "jsbsim_version": self.jsbsim_version, "code_sha": self.code_sha,
                "host": socket.gethostname(), "cpu_count": os.cpu_count(), "usable_cores": usable_cores(),
                "workers": self.workers, "python": sys.version.split()[0], "numpy": np.__version__,
                "platform": platform.platform()}
        if os.path.exists(cpath):
            with open(cpath) as f:
                old = json.load(f)
            if identity(old["resolved"]) != identity(self.cfg):
                raise SystemExit(f"run dir {self.run_dir} exists with a different resolved config; "
                                 "pass a new --run-id or use --resume with the stored config")
            for k in ("jsbsim_version", "code_sha"):
                if old["provenance"][k] != prov[k]:
                    raise SystemExit(f"cannot resume: {k} changed ({old['provenance'][k]} -> {prov[k]}); results would not be identical")
        else:
            _atomic_json(cpath, {"resolved": self.cfg, "provenance": prov, "created": _now()}, indent=2)
        self.prov = prov
        spath = os.path.join(self.run_dir, "sessions.jsonl")
        if os.path.exists(spath):
            with open(spath) as f:
                self.session = sum(1 for _ in f)
        resumed = {}
        for ac in self.cfg["aircraft"]:
            ck = self._load_ck(ac["name"])
            if ck:
                resumed[ac["name"]] = ck["gen_next"]
        with open(spath, "a") as f:
            f.write(json.dumps({"session": self.session, "started": _now(), "host": prov["host"], "workers": self.workers,
                                "schedule": self.schedule, "resumed_at_generation": resumed,
                                "argv": sys.argv}) + "\n")
        self._rebuild_history()
        # genomes.jsonl = exactly the checkpointed generations (rows are written before each checkpoint)
        runinfo.truncate_rows(os.path.join(self.run_dir, "genomes.jsonl"),
                              {ac["name"]: resumed.get(ac["name"], 0) for ac in self.cfg["aircraft"]})
        return resumed

    def _ck_path(self, name):
        return os.path.join(self.ck_dir, f"{name}.json")

    def _load_ck(self, name) -> Optional[Dict]:
        p = self._ck_path(name)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    def _rebuild_history(self):
        """history.jsonl = exactly the generations covered by checkpoints (drops anything past a kill)."""
        recs = []
        for i, ac in enumerate(self.cfg["aircraft"]):
            ck = self._load_ck(ac["name"])
            if ck:
                recs += [(r["generation"], i, r) for r in ck["history"]]
        recs.sort(key=lambda x: (x[0], x[1]))
        tmp = os.path.join(self.run_dir, "history.jsonl.tmp")
        with open(tmp, "w") as f:
            for _, _, r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, os.path.join(self.run_dir, "history.jsonl"))

    def _append_history(self, rec):
        with self.hist_lock, open(os.path.join(self.run_dir, "history.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _append_rows(self, rows):
        with self.rows_lock, open(os.path.join(self.run_dir, "genomes.jsonl"), "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
            f.flush()
            os.fsync(f.fileno())

    def _mf(self, ac: Dict) -> Dict:
        """This aircraft's multi-fidelity settings (multi_fidelity_per_aircraft override, else the global block)."""
        return ac.get("multi_fidelity") or self.mf

    def _viz_on(self) -> bool:
        return self.viz in (True, "on")

    def _describe_all(self):
        """model_version per aircraft and fidelity (computed in a worker), then run.json."""
        fids_of = {ac["name"]: (list(self._mf(ac)["ladder"]) if self._mf(ac).get("enabled") else [self.fidelity])
                   for ac in self.cfg["aircraft"]}
        pa = {ac["name"]: fid_mod.per_aircraft(ac["name"], self.cfg.get("fidelity_per_aircraft")) for ac in self.cfg["aircraft"]}
        futs = {ac["name"]: self.pool.submit(eval_mod.describe, ac["resolved_profile"], fids_of[ac["name"]],
                                             pa[ac["name"]]["reduced_gate"])
                for ac in self.cfg["aircraft"]}
        for ac in self.cfg["aircraft"]:
            d = futs[ac["name"]].result()
            fids, mfa = fids_of[ac["name"]], self._mf(ac)
            if not d.get("error") and ac["name"] in (self.cfg.get("pin_model_version") or {}):
                self.cache.pins[ac["name"]] = check_pins(self.cfg, ac["name"], fids, d["model_version"], ac["resolved_profile"])
            prof = sim.Profile.from_dict(ac["resolved_profile"])
            sch, groups = full_schema(prof, self.cfg.get("struct_genes", False), self.cfg.get("struct_asymmetric", False),
                                      self.cfg.get("genome_kind"),
                                      shape_locked_of(self.cfg) if self.cfg.get("genome_kind") == "phase3_b2a" else None)
            self.info[ac["name"]] = {"schema": sch, "groups": groups, "model_files_sha": d["model_files_sha"],
                                     "model_version": d["model_version"].get(self.fidelity),
                                     "mv": d["model_version"], "error": d.get("error")}
            if any(f in fids for f in ("reduced", "full", "full_a1", "full_a1_b1", "full_a1_b2a")):
                self.info[ac["name"]]["reduced_gate"] = pa[ac["name"]]["reduced_gate"]
            if mfa.get("enabled"):
                self.info[ac["name"]]["screen_model_version"] = d["model_version"].get(mfa["ladder"][0])
                self.info[ac["name"]]["ladder_model_version"] = {f: d["model_version"].get(f) for f in fids}
                mff = mfa.get("min_full_frac")
                self.info[ac["name"]]["min_full_frac"] = float(pa[ac["name"]]["min_full_frac"] if mff is None else mff)
        rj = os.path.join(self.run_dir, "run.json")
        doc = runinfo.build_run_json(self.cfg, self.prov, self.info, _now())
        if os.path.exists(rj):
            with open(rj) as f:
                old = json.load(f)
            if old.get("model_version") != doc["model_version"] and not old.get("backfill"):
                raise SystemExit(f"cannot resume: model_version changed {old.get('model_version')} -> {doc['model_version']}")
        else:
            runinfo.write_json_atomic(rj, doc)

    # ---- evaluation
    KEEP = ("cost", "status", "t_end", "track", "effort", "comfort", "heading_rms", "hdg_drift_deg", "hdg_max_abs_err_deg",
            "hold_osc", "hold_pp_ft", "draft_residual_ft", "draft_max_err_ft")

    def _trim_result(self, r: Dict) -> Dict:
        out = {k: r[k] for k in self.KEEP if k in r}
        if r.get("fidelity", "rigid") != "rigid":   # flex (FD flexeval per-scenario entry); rigid rows stay exactly as before
            for k in ("sim_cost", "struct", "not_flown", "tip_max_ft", "twist_max_deg", "fidelity", "model_version"):
                if k in r:
                    out[k] = r[k]
        return out

    def _evaluate(self, st: Dict, pop: np.ndarray, fid: str):
        if fid != "rigid":
            return self._evaluate_flex(st, pop, fid)
        ac, prof_d, scs = st["ac"], st["profile_d"], st["scenarios_d"]
        schema, groups = st["schema"], st["groups"]
        mv = st["mv"][fid]
        keys = [[cache_mod.eval_key(ac["name"], g, prof_d, sc, self.cfg["scenario_seed"], self.jsbsim_version,
                                     self.code_sha, st["model_sha"], fidelity=fid, model_version=mv)
                 for sc in scs] for g in pop]
        need: Dict[str, tuple] = {}
        for i, row in enumerate(keys):
            for s, k in enumerate(row):
                if k not in need:
                    need[k] = (i, s)
        memo = st["memo"]
        mem_hits = sum(1 for k in need if k in memo)
        todo = [k for k in need if k not in memo]
        disk = self.cache.get_many(todo)
        memo.update(disk)
        miss = [k for k in todo if k not in disk]
        viz = self._viz_on()
        futs = {}
        for k in miss:
            i, s = need[k]
            gains, struct = eval_mod.split_values(genome.decode(pop[i], schema), dict(zip([g.name for g in schema], groups)))
            futs[self.pool.submit(eval_mod.task, prof_d, gains, struct, scs[s], fid, viz,
                                  self.cfg["trajectories"]["sample_hz"])] = k
        new = []
        cpu = 0.0
        for fu in cf.as_completed(futs):
            r = fu.result()
            k = futs[fu]
            if r.get("fidelity") != fid or r.get("model_version") != mv:
                raise RuntimeError(f"{ac['name']}: result fidelity/model_version {r.get('fidelity')}/{r.get('model_version')} "
                                   f"!= expected {fid}/{mv}")
            tr = r.pop("trajectory", None)
            if tr is not None and need[k][1] == self.cfg["trajectories"]["scenario"]:
                st["live_traj"][k] = tr
            memo[k] = r
            new.append((k, ac["name"], r))
            cpu += r.get("task_cpu_s", r.get("wall_s", 0.0))
        self.cache.put_many(new)
        results = []
        for row in keys:
            per = [memo[k] for k in row]
            agg = eval_mod.aggregate(per)
            results.append({"cost": agg["cost"], "per_scenario": [self._trim_result(r) for r in per], "agg": agg,
                            "keys": row})
        return results, {"unique_sims": len(need), "sims_computed": len(miss), "cache_hits_mem": mem_hits,
                         "cache_hits_disk": len(disk), "eval_cpu_s": cpu}

    def _ladder(self, st, pop, gen, mf):
        """Multi-fidelity generation. ladder = screens (ascending) + authoritative fidelity. Stage 0 scores everyone;
        each later stage scores the best of the previous stage (by that stage's cost) plus every carried elite:
        the authoritative stage gets k_full = max(top_k, ceil(min_full_frac * pop)) (min_full_frac per aircraft: 0.25 on
        swept wings), a middle stage mid_k (default 2 * k_full). Ranking: authoritative-scored first (by that cost), then
        by the highest fidelity reached and its cost. Feasibility is trusted only from the authoritative stage."""
        ladder = mf["ladder"]
        n = len(pop)
        elite = self.cfg["ga"]["elite"]
        elites = set(range(min(elite, n))) if gen > 0 else set()
        k_full = min(n, max(int(mf["top_k"]), math.ceil(st["min_full_frac"] * n - 1e-9)))
        k_mid = min(n, int(mf.get("mid_k") or 2 * k_full))
        idx = list(range(n))
        scored: Dict[str, Dict[int, Dict]] = {}
        stages = {}
        for si, f in enumerate(ladder):
            ts = time.perf_counter()
            res, es_x = self._evaluate(st, pop[idx], f)
            scored[f] = dict(zip(idx, res))
            stages[f] = {**es_x, "n": len(idx), "eval_wall_s": time.perf_counter() - ts}
            if si + 1 < len(ladder):
                keep = k_full if si + 1 == len(ladder) - 1 else k_mid
                ordr = sorted(idx, key=lambda i: (scored[f][i]["cost"], i))
                idx = sorted(set(ordr[:keep]) | elites)
        top = ladder[-1]
        full_idx = set(scored[top])

        def level(i):
            return max(si for si, f in enumerate(ladder) if i in scored[f])
        sel = [scored[ladder[level(i)]][i] for i in range(n)]
        order = np.array(sorted(range(n), key=lambda i: (len(ladder) - 1 - level(i), sel[i]["cost"], i)), dtype=int)
        costs = np.array([r["cost"] for r in sel])
        es = {k: sum(stages[f][k] for f in ladder) for k in ("unique_sims", "sims_computed", "cache_hits_mem",
                                                             "cache_hits_disk", "eval_cpu_s")}
        fl = sorted(full_idx)
        sp, sp_ok, pairs = {}, {}, {}
        for f in ladder[:-1]:
            pr = [[scored[f][i]["cost"], scored[top][i]["cost"]] for i in fl]
            okp = [p for p, i in zip(pr, fl) if scored[f][i]["agg"]["status"] == "ok" and scored[top][i]["agg"]["status"] == "ok"]
            sp[f"{f}_vs_{top}"] = spearman([p[0] for p in pr], [p[1] for p in pr])
            sp_ok[f"{f}_vs_{top}"] = spearman([p[0] for p in okp], [p[1] for p in okp])
            pairs[f] = pr
        if len(ladder) == 3:
            mid = sorted(scored[ladder[1]])
            pr = [[scored[ladder[0]][i]["cost"], scored[ladder[1]][i]["cost"]] for i in mid]
            sp[f"{ladder[0]}_vs_{ladder[1]}"] = spearman([p[0] for p in pr], [p[1] for p in pr])
        extra = {"ladder": ladder, "screen_fidelity": ladder[0], "top_k": mf["top_k"], "k_full": k_full,
                 "k_mid": k_mid if len(ladder) == 3 else None, "min_full_frac": st["min_full_frac"],
                 "n_rescored": len(fl), "rescored_idx": fl, "spearman": sp, "spearman_both_ok": sp_ok,
                 "spearman_screen_vs_full": sp[f"{ladder[0]}_vs_{top}"], "rescored_pairs": pairs[ladder[0]],
                 "ladder_pairs": pairs, "stages": stages,
                 "screen": stages[ladder[0]], "full": stages[top],
                 "eval_wall_s_screen": sum(stages[f]["eval_wall_s"] for f in ladder[:-1]),
                 "eval_wall_s_full": stages[top]["eval_wall_s"],
                 "best_screen_cost": float(min(r["cost"] for r in scored[ladder[0]].values())),
                 "n_ok_authoritative": sum(1 for i in fl if scored[top][i]["agg"]["status"] == "ok")}
        return sel, order, costs, es, scored, full_idx, extra

    def _evaluate_flex(self, st: Dict, pop: np.ndarray, fid: str):
        """reduced/full: one FD flexeval.evaluate per genome over all scenarios (margins + model build once per genome);
        cache key = genome x all scenarios x reduced gate x model_version."""
        ac, prof_d, scs = st["ac"], st["profile_d"], st["scenarios_d"]
        schema, groups = st["schema"], st["groups"]
        mv, gate = st["mv"][fid], st["reduced_gate"]
        unit = {"scenarios": scs, "reduced_gate": gate if fid == "reduced" else None, "unit": "genome"}
        gmap = dict(zip([g.name for g in schema], groups))
        # full_a1_b1: the decoded shape genes (None = baseline planform) and FD's shape_cache_key go into the key
        shapes = [eval_mod.shape_values(genome.decode(g, schema), gmap) if fid in fid_mod.SHAPED else None for g in pop]
        b2 = fid == fid_mod.B2
        ecost = bool(self.cfg.get("energy_cost")) if b2 else None
        env = [(float(s["speed_kts"]), float(s["h0_ft"])) for s in scs] if b2 else None

        def skey(sh):
            if b2:
                return fid_mod.shape_cache_key_b2(sh, ac["name"], env)
            return fid_mod.shape_cache_key(sh) if fid == fid_mod.B1 else None
        keys = [cache_mod.eval_key(ac["name"], g, prof_d, unit, self.cfg["scenario_seed"], self.jsbsim_version,
                                   self.code_sha, st["model_sha"], fidelity=fid, model_version=mv,
                                   shape_key=skey(sh), **({"energy_cost": ecost} if b2 else {}))
                for g, sh in zip(pop, shapes)]
        need: Dict[str, int] = {}
        for i, k in enumerate(keys):
            need.setdefault(k, i)
        memo = st["memo"]
        mem_hits = sum(1 for k in need if k in memo)
        todo = [k for k in need if k not in memo]
        disk = self.cache.get_many(todo)
        memo.update(disk)
        miss = [k for k in todo if k not in disk]
        viz = self._viz_on()
        s_idx = self.cfg["trajectories"]["scenario"]
        futs = {}
        for k in miss:
            gains, struct = eval_mod.split_values(genome.decode(pop[need[k]], schema), gmap)
            kw = {"shape": shapes[need[k]]} if fid in fid_mod.SHAPED else {}
            if b2 and ecost:
                kw["energy_cost"] = True
            futs[self.pool.submit(eval_mod.task_genome, prof_d, gains, struct, scs, fid, gate, viz,
                                  self.cfg["trajectories"]["sample_hz"], **kw)] = k
        new, cpu = [], 0.0
        for fu in cf.as_completed(futs):
            r = fu.result()
            k = futs[fu]
            if r.get("fidelity") != fid or r.get("model_version") != mv:
                raise RuntimeError(f"{ac['name']}: result fidelity/model_version {r.get('fidelity')}/{r.get('model_version')} "
                                   f"!= expected {fid}/{mv}")
            trs = r.pop("trajectories", None)
            if trs and len(trs) > s_idx and trs[s_idx] is not None:
                st["live_traj"][k] = trs[s_idx]
            r["per_scenario"] = [self._trim_result(p) for p in r["per_scenario"]]
            memo[k] = r
            new.append((k, ac["name"], r))
            cpu += r.get("task_cpu_s", r.get("wall_s", 0.0))
        self.cache.put_many(new)
        results = []
        for k in keys:
            r = memo[k]
            agg = {kk: v for kk, v in r.items() if kk != "per_scenario"}
            results.append({"cost": r["cost"], "per_scenario": r["per_scenario"], "agg": agg, "keys": [k] * len(scs)})
        n_s = len(scs)
        return results, {"unique_sims": len(need) * n_s, "sims_computed": len(miss) * n_s, "cache_hits_mem": mem_hits * n_s,
                         "cache_hits_disk": len(disk) * n_s, "eval_cpu_s": cpu}

    # ---- one aircraft
    def _rows(self, st, gen, pop_orig, order, sel, scored, full_idx, mf):
        """genomes.jsonl rows of one generation: individual_id <ac>:g<gen>:r<rank> (r0 = best, as Sim Bridge's adapter),
        index = position in the evaluated population (elites carried from the previous generation are 0..elite-1)."""
        name, schema, groups = st["ac"]["name"], st["schema"], st["groups"]
        elite = self.cfg["ga"]["elite"]
        rank_of = {int(i): r for r, i in enumerate(order)}
        rows = []
        for i in range(len(pop_orig)):
            res = sel[i]
            agg = res["agg"]
            vals = genome.decode(pop_orig[i], schema)
            gains, struct = eval_mod.split_values(vals, dict(zip([g.name for g in schema], groups)))
            row = {"schema": runinfo.GENOMES_SCHEMA, "individual_id": f"{name}:g{gen}:r{rank_of[i]}", "run_id": self.cfg["run_id"],
                   "aircraft": name, "generation": gen, "eval_seed": self.cfg["scenario_seed"], "index": i, "rank": rank_of[i], "is_best": rank_of[i] == 0,
                   "is_elite": rank_of[i] < elite, "carried_elite": gen > 0 and i < elite,
                   "genome": vals, "genome_norm": [float(x) for x in pop_orig[i]],
                   "gains": gains, "struct": struct, "cost": res["cost"], "fitness": res["cost"],
                   "per_scenario_cost": [p["cost"] for p in res["per_scenario"]],
                   "scenario_ids": st["scenario_ids"], "status": agg["status"], "feasible": agg["feasible"],
                   "feasibility_fidelity": agg["feasibility_fidelity"], "terms": agg["terms"],
                   "terms_available": agg["terms_available"], "fidelity": agg["fidelity"],
                   "model_version": agg["model_version"], "session": self.session}
            shape = eval_mod.shape_values(vals, dict(zip([g.name for g in schema], groups)))
            if shape is not None:      # phase3_b1 rows only (earlier kinds unchanged)
                row["shape"] = shape
            if agg.get("geometry_gate_reject"):
                row["geometry_gate"] = agg.get("geometry_gate")
                if agg.get("geometry_gate_b2") is not None:
                    row["geometry_gate_b2"] = agg.get("geometry_gate_b2")
            if agg.get("energy_terms") is not None:     # full_a1_b2a rows only
                et = agg["energy_terms"]
                row["energy_terms"] = {k: et.get(k) for k in ("J_energy", "J_speed_guard", "in_cost", "cost_fd",
                                                              "energy_drag_increment", "speed_deficit_kts_mean")}
            if "margins" in agg:
                row["margins"], row["margins_fidelity"] = agg["margins"], agg["margins_fidelity"]
            for k in ("mass_total_frac", "mass_lb", "J_mass_fd", "mass_credit_clip", "mass_credit_delta", "task_cpu_s"):
                if k in agg and agg.get("fidelity", "rigid") != "rigid":   # flex rows only (rigid rows unchanged)
                    row[k] = agg[k]
            if mf:
                scr = scored[mf["ladder"][0]]
                row["rescored_at_full"] = i in full_idx
                row["screen_cost"] = scr[i]["cost"]
                row["screen_fidelity"] = scr[i]["agg"]["fidelity"]
                row["screen_model_version"] = scr[i]["agg"]["model_version"]
                row["screen_per_scenario_cost"] = [p["cost"] for p in scr[i]["per_scenario"]]
                row["ladder_cost"] = {f: scored[f][i]["cost"] for f in mf["ladder"] if i in scored[f]}
                row["ladder_status"] = {f: scored[f][i]["agg"]["status"] for f in mf["ladder"] if i in scored[f]}
                row["ladder_model_version"] = {f: scored[f][i]["agg"]["model_version"] for f in mf["ladder"] if i in scored[f]}
                if i not in full_idx:      # feasibility is trusted only from the authoritative fidelity
                    row["feasible"], row["feasibility_fidelity"] = None, None
            rows.append(row)
        return rows

    def _live(self, st, gen, best_norm, best_key, fitness):
        """viz on: runs/<id>/live/<aircraft>.json = trajectory of the current best (scenario trajectories.scenario)."""
        s_idx = self.cfg["trajectories"]["scenario"]
        tr = st["live_traj"].get(best_key)
        r = None
        vals = genome.decode(best_norm, st["schema"])
        gains, struct = eval_mod.split_values(vals, dict(zip([g.name for g in st["schema"]], st["groups"])))
        if tr is None:   # cache hit: re-fly once with the recorder
            kw = {"shape": eval_mod.shape_values(vals, dict(zip([g.name for g in st["schema"]], st["groups"])))} \
                if self.fidelity in fid_mod.SHAPED else {}
            if self.fidelity == fid_mod.B2 and self.cfg.get("energy_cost"):
                kw["energy_cost"] = True
            r = self.pool.submit(eval_mod.task, st["profile_d"], gains, struct, st["scenarios_d"][s_idx], self.fidelity,
                                 True, self.cfg["trajectories"]["sample_hz"], st["reduced_gate"], "fd", **kw).result()
        elif self.fidelity == "rigid":
            r = dict(st["memo"][best_key])
            r["trajectory"] = tr
        else:
            g = st["memo"][best_key]
            r = dict(g["per_scenario"][s_idx], fidelity=g["fidelity"], model_version=g["model_version"])
            r["trajectory"] = tr
        st["live_traj"].clear()
        doc = trajectory.build_doc(run_id=self.cfg["run_id"], aircraft=st["ac"]["name"], jsbsim_version=self.jsbsim_version,
                                   git_sha=self.git["sha"], seed=st["ac"]["seed"], generation=gen, fitness=fitness,
                                   gains=vals, scenario=st["scenarios_d"][s_idx], scenario_index=s_idx, sim_result=r,
                                   profile=st["profile_d"],
                                   extra={"live": True, "fidelity": r.get("fidelity"), "model_version": r.get("model_version"),
                                          "scenario_id": st["scenario_ids"][s_idx]})
        d = os.path.join(self.run_dir, "live")
        os.makedirs(d, exist_ok=True)
        _atomic_json(os.path.join(d, f"{st['ac']['name']}.json"), doc)

    def _evolve_aircraft(self, ac: Dict) -> Dict:
        name = ac["name"]
        G = self.cfg["ga"]["generations"]
        prof = sim.Profile.from_dict(ac["resolved_profile"])
        scs = sim.make_scenarios(self.cfg["scenarios"], self.cfg["scenario_seed"], prof)
        info = self.info[name]
        mf = self._mf(ac) if self._mf(ac).get("enabled") else None
        fid = self.fidelity
        st = {"ac": ac, "profile_d": ac["resolved_profile"], "scenarios_d": [s.to_dict() for s in scs],
              "scenario_ids": [runinfo.scenario_id(name, i) for i in range(len(scs))],
              "schema": info["schema"], "groups": info["groups"], "memo": {}, "live_traj": {},
              "mv": info["mv"], "model_sha": sim.model_files_sha(name, prof.aircraft_root),
              "reduced_gate": info.get("reduced_gate"), "min_full_frac": info.get("min_full_frac", 0.0)}
        gcfg = ga.GAConfig(**{k: v for k, v in self.cfg["ga"].items() if k != "generations"})
        out = {"aircraft": name, "profile": ac["profile"], "aircraft_root": prof.aircraft_root,
               "model_files_sha": st["model_sha"], "fidelity": fid, "fidelity_label": fid_mod.label(fid),
               "model_version": info["model_version"]}
        if mf:
            out["multi_fidelity"] = {**mf, "screen_model_version": info.get("screen_model_version"),
                                     "ladder_model_version": info.get("ladder_model_version"),
                                     "min_full_frac": info.get("min_full_frac")}
        if info.get("reduced_gate") is not None:
            out["reduced_gate"] = info["reduced_gate"]
        pre = self.pool.submit(sim.preflight, st["profile_d"], st["scenarios_d"][0]).result()
        out["preflight"] = pre
        if pre.get("socket_io_elements"):
            self.log(f"[{name}] WARNING: model declares {pre['socket_io_elements']} socket I/O element(s); policy "
                     f"'{pre['socket_policy']}' ({'stripped copy in /tmp' if pre['socket_policy'] == 'strip' else 'refused'})")
        if not pre["ok"] or info.get("error"):
            why = f"{pre['status']}: {pre.get('error')}" if not pre["ok"] else f"fidelity unavailable: {info['error']}"
            self.log(f"[{name}] SKIPPED: {why} at {prof.h0_ft:.0f} ft / {prof.speed_kts:.0f} KCAS")
            out["skipped"] = why
            return out

        ck = self._load_ck(name)
        kind = self.cfg.get("genome_kind")
        sspec = shape_spec(st["schema"], st["groups"], self.cfg["shape_ops"]) if kind in SHAPED_KINDS else None
        blocks = gene_blocks(st["groups"]) if kind in SHAPED_KINDS else None
        if ck is None:
            rng = np.random.default_rng(ac["seed"])
            if sspec is None:
                pop = ga.generation_zero(rng, self.cfg["ga"]["pop_size"], len(st["schema"]))  # 8 with heading hold, +12/14 struct
                pop = seed_generation_zero(pop, rng, st["schema"], st["groups"], self.cfg.get("init") or {})
            else:   # phase3_b1: controller + structure drawn exactly as phase2 / phase3a1 (same seed -> same 20 genes),
                n_cs = sspec.idx[0]   # then the shape block around the identity planform (Genome B1 spec)
                pop = ga.generation_zero(rng, self.cfg["ga"]["pop_size"], n_cs)
                pop = seed_generation_zero(pop, rng, st["schema"][:n_cs], st["groups"][:n_cs], self.cfg.get("init") or {})
                pop = np.hstack([pop, ga.shape_generation_zero(rng, self.cfg["ga"]["pop_size"], sspec)])
            ck = {"aircraft": name, "gen_next": 0, "done": False, "pop": pop.tolist(),
                  "rng_state": rng.bit_generator.state, "history": [], "best_per_gen": []}
        else:
            self.log(f"[{name}] resuming at generation {ck['gen_next']}/{G}")
        rng = np.random.default_rng()
        rng.bit_generator.state = ck["rng_state"]
        pop = np.array(ck["pop"], dtype=np.float64)
        elite = self.cfg["ga"]["elite"]
        kill_after = os.environ.get("EVOLUTION_TEST_KILL_AFTER_ROWS")
        t_ac = time.perf_counter()
        tot = {"sims_computed": 0, "unique_sims": 0, "cache_hits_mem": 0, "cache_hits_disk": 0, "eval_cpu_s": 0.0}
        while ck["gen_next"] < G:
            gen = ck["gen_next"]
            t0 = time.perf_counter()
            extra = {}
            scored, full_idx = None, set()
            if not mf:
                sel, es = self._evaluate(st, pop, fid)
                costs = np.array([r["cost"] for r in sel])
                order = ga.rank_order(costs)
            else:
                sel, order, costs, es, scored, full_idx, extra = self._ladder(st, pop, gen, mf)
            wall = time.perf_counter() - t0
            for k in tot:
                tot[k] += es[k]
            rows = self._rows(st, gen, pop, order, sel, scored, full_idx, mf)
            pop, costs, sel = pop[order], costs[order], [sel[i] for i in order]
            invalid = [any(s["status"] != "ok" for s in r["per_scenario"]) for r in sel]
            crash = [any(s["status"] == "crash" for s in r["per_scenario"]) for r in sel]
            status_counts: Dict[str, int] = {}
            for r in sel:
                for s in r["per_scenario"]:
                    status_counts[s["status"]] = status_counts.get(s["status"], 0) + 1
            hits = es["cache_hits_mem"] + es["cache_hits_disk"]
            rec = {
                "aircraft": name, "generation": gen, "best": float(costs[0]), "mean": float(costs.mean()),
                "median": float(np.median(costs)), "std": float(costs.std()),
                "valid_rate": 1.0 - float(np.mean(invalid)), "invalid_rate": float(np.mean(invalid)),
                "crash_rate": float(np.mean(crash)), "status_counts": status_counts,
                "n_individuals": len(pop), **es, "cache_hit_rate": hits / es["unique_sims"] if es["unique_sims"] else None,
                "eval_wall_s": wall, "best_genome": [float(x) for x in pop[0]],
                "best_gains": genome.decode(pop[0], st["schema"]), "session": self.session,
            }
            if fid == fid_mod.B1:
                rec["geometry_gate_rejects"] = sum(1 for r in sel if r["agg"].get("geometry_gate_reject"))
                gr = [r["agg"]["status"] for r in sel if r["agg"].get("geometry_gate_reject")]
                rec["geometry_gate_reasons"] = {x: gr.count(x) for x in sorted(set(gr))}
            if fid != "rigid" or mf:
                fs = [r for r in sel if r["agg"]["fidelity"] == fid]
                rec["n_scored_authoritative"] = len(fs)
                rec["feasible_rate_authoritative"] = (sum(1 for r in fs if r["agg"]["status"] == "ok") / len(fs)) if fs else None
                rec.update({"fidelity": fid, "model_version": info["model_version"],
                            "best_fidelity": sel[0]["agg"]["fidelity"], "best_feasible": sel[0]["agg"]["feasible"],
                            "best_terms": sel[0]["agg"]["terms"], **extra})
            best = {"generation": gen, "genome": [float(x) for x in pop[0]], "fitness": float(costs[0]),
                    "per_scenario": sel[0]["per_scenario"]}
            if fid != "rigid" or mf:
                best["fidelity"] = sel[0]["agg"]["fidelity"]
            if gen < G - 1:
                nxt = ga.next_generation(rng, pop, gcfg) if sspec is None else \
                    ga.next_generation_blocks(rng, pop, gcfg, blocks, sspec)
            else:
                nxt = pop  # final ranked population
            self._append_rows(rows)                      # rows first (truncated back to the checkpoint on resume) ...
            if kill_after is not None and gen >= int(kill_after):
                os._exit(17)                             # test hook: die between rows and checkpoint
            ck = {"aircraft": name, "gen_next": gen + 1, "done": gen + 1 >= G, "pop": nxt.tolist(),
                  "rng_state": rng.bit_generator.state, "history": ck["history"] + [rec],
                  "best_per_gen": ck["best_per_gen"] + [best]}
            _atomic_json(self._ck_path(name), ck)       # ... then checkpoint ...
            self._append_history(rec)                    # ... then log (history is rebuilt from checkpoints on resume)
            if self._viz_on():
                self._live(st, gen, pop[0], sel[0]["keys"][self.cfg["trajectories"]["scenario"]], float(costs[0]))
            pop = nxt
            msg = (f"[{name:>9}] gen {gen:3d}  best {costs[0]:10.4f}  median {np.median(costs):10.4f}  "
                   f"invalid {sum(invalid):3d}/{len(pop)}  sims {es['sims_computed']:3d} hits {hits:3d}  {wall:5.2f}s")
            if rec.get("geometry_gate_rejects"):
                msg += f"  geometry_gate rejects {rec['geometry_gate_rejects']} {rec['geometry_gate_reasons']}"
            if mf:
                msg += f"  rescored {extra['n_rescored']} rho " + " ".join(
                    f"{k.split('_vs_')[0]}:{'n/a' if v is None else f'{v:+.3f}'}" for k, v in extra["spearman"].items())
            self.log(msg)
        out["evolve_wall_s_this_session"] = time.perf_counter() - t_ac
        out["this_session"] = tot
        out.update(self._export(st, ck))
        return out

    # ---- trajectories + metrics (re-simulation of saved elites through eval.task; never in the hot path)
    def _export(self, st: Dict, ck: Dict) -> Dict:
        name = st["ac"]["name"]
        tcfg = self.cfg["trajectories"]
        s_idx = tcfg["scenario"]
        bests = {b["generation"]: b for b in ck["best_per_gen"]}
        G = self.cfg["ga"]["generations"]
        final = bests[G - 1]
        gmap = dict(zip([g.name for g in st["schema"]], st["groups"]))

        def submit(norm, s, fidl):
            vals = genome.decode(norm, st["schema"])
            gains, struct = eval_mod.split_values(vals, gmap)
            kw = {"shape": eval_mod.shape_values(vals, gmap)} if fidl in fid_mod.SHAPED else {}
            if fidl == fid_mod.B2 and self.cfg.get("energy_cost"):
                kw["energy_cost"] = True
            return self.pool.submit(eval_mod.task, st["profile_d"], gains, struct, st["scenarios_d"][s], fidl, True,
                                    tcfg["sample_hz"], st["reduced_gate"], "sb", **kw)
        jobs = {}
        for g in tcfg["generations_resolved"]:
            jobs[("traj", g)] = submit(bests[g]["genome"], s_idx, bests[g].get("fidelity", self.fidelity))
        for s in range(len(st["scenarios_d"])):
            if not (s == s_idx and ("traj", G - 1) in jobs):
                jobs[("met", s)] = submit(final["genome"], s, final.get("fidelity", self.fidelity))
        res = {k: f.result() for k, f in jobs.items()}
        entries, mismatches = [], []
        for g in tcfg["generations_resolved"]:
            r = res[("traj", g)]
            if r["cost"] != bests[g]["per_scenario"][s_idx]["cost"]:
                mismatches.append({"generation": g, "eval": bests[g]["per_scenario"][s_idx]["cost"], "resim": r["cost"]})
            doc = trajectory.build_doc(run_id=self.cfg["run_id"], aircraft=name, jsbsim_version=self.jsbsim_version,
                                       git_sha=self.git["sha"], seed=st["ac"]["seed"], generation=g,
                                       fitness=bests[g]["fitness"], gains=genome.decode(bests[g]["genome"], st["schema"]),
                                       scenario=st["scenarios_d"][s_idx], scenario_index=s_idx, sim_result=r,
                                       profile=st["profile_d"],
                                       extra=({"fidelity": r.get("fidelity"), "model_version": r.get("model_version"),
                                               "scenario_id": st["scenario_ids"][s_idx]}
                                              if r.get("fidelity", "rigid") != "rigid" else None))
            fn = trajectory.write_doc(doc, self.traj_dir)
            entries.append({"generation": g, "fitness": bests[g]["fitness"], "aircraft": name, "file": fn})
        mcfg = self.cfg["metrics"]
        per_s = []
        for s in range(len(st["scenarios_d"])):
            r = res[("traj", G - 1)] if (s == s_idx and ("traj", G - 1) in res) else res[("met", s)]
            if r["cost"] != final["per_scenario"][s]["cost"]:
                mismatches.append({"generation": G - 1, "scenario": s, "eval": final["per_scenario"][s]["cost"], "resim": r["cost"]})
            if r.get("trajectory"):
                m = trajectory.hold_metrics(r, mcfg["band_ft"], mcfg["hold_after_s"])
            else:
                m = {}
            m["cost"] = r["cost"]
            for k in ("hdg_drift_deg", "hdg_max_abs_err_deg", "hold_osc", "hold_pp_ft", "draft_residual_ft", "draft_max_err_ft"):
                if k in r:
                    m[k] = r[k]
            per_s.append(m)
        hist = ck["history"]
        schema = st["schema"]
        norm = final["genome"]
        return {
            "best_fitness": final["fitness"], "gen0_best_fitness": bests[0]["fitness"],
            "best_gains": genome.decode(norm, schema), "best_genome_norm": norm,
            "genes_at_bound": _genes_at_bound(schema, norm),
            "gain_bounds": {gn.name: [gn.min, gn.max] for gn in schema},
            "gene_kinds": {gn.name: gn.kind for gn in schema},
            "final_per_scenario": final["per_scenario"], "final_metrics": per_s,
            "invalid_rate_all_gens": float(np.mean([h["invalid_rate"] for h in hist])),
            "invalid_rate_gen0": hist[0]["invalid_rate"], "invalid_rate_final": hist[-1]["invalid_rate"],
            "crash_rate_all_gens": float(np.mean([h["crash_rate"] for h in hist])),
            "status_counts_all_gens": _sum_counts([h["status_counts"] for h in hist]),
            "totals_from_history": {k: sum(h[k] for h in hist) for k in
                                    ("unique_sims", "sims_computed", "cache_hits_mem", "cache_hits_disk", "eval_cpu_s", "eval_wall_s")},
            "trajectories": entries, "resim_mismatches": mismatches,
        }

    # ---- top level
    def run(self) -> Dict:
        for name, pins in (self.cfg.get("pin_model_version") or {}).items():   # before anything touches the run dir
            bad = sorted(f for f, v in pins.items() if v == PIN_PLACEHOLDER)
            if bad:
                raise SystemExit(f"[{name}] pin_model_version {bad} still {PIN_PLACEHOLDER!r}: fill in Flight Dynamics' "
                                 "new model_version strings first; refusing to evaluate, write cache entries or resume")
        # cache guard: pinned aircraft store nothing until check_pins (in _describe_all) allows their run strings
        self.cache.pins = {n: {} for n in (self.cfg.get("pin_model_version") or {})}
        resumed = self._prepare_run_dir()
        self.log(f"run {self.cfg['run_id']}  dir {self.run_dir}\n  {len(self.cfg['aircraft'])} aircraft, "
                 f"{self.workers} workers, schedule={self.schedule}, cache={'on' if self.cache.enabled else 'off'}, "
                 f"jsbsim {self.jsbsim_version}, code {self.code_sha}, src {self.git['sha'][:10]}\n  "
                 f"fidelity {fid_mod.label(self.fidelity)}, viz {'on' if self._viz_on() else 'off'}"
                 + (f", multi-fidelity ladder={'->'.join(self.mf['ladder'])} top_k={self.mf['top_k']}" if self.mf.get('enabled') else "")
                 + "".join(f"\n  {ac['name']}: ladder={'->'.join(ac['multi_fidelity']['ladder'])}"
                           f" min_full_frac={ac['multi_fidelity'].get('min_full_frac')}"
                           for ac in self.cfg["aircraft"] if (ac.get("multi_fidelity") or {}).get("enabled"))
                 + (f"\n  resuming: {resumed}" if resumed else ""))
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload(["evolution._fs_guard", "evolution.sim", "evolution.fidelity", "evolution.eval",
                                    "jsbsim", "numpy"])
        t0 = time.perf_counter()
        load_start = os.getloadavg()
        n_threads = len(self.cfg["aircraft"]) if self.schedule == "concurrent" else 1
        per_ac: Dict[str, Dict] = {}
        with cf.ProcessPoolExecutor(self.workers, mp_context=ctx, initializer=sim.worker_init) as pool, cf.ThreadPoolExecutor(n_threads) as tp:
            self.pool = pool
            self._describe_all()
            futs = {tp.submit(self._evolve_aircraft, ac): ac["name"] for ac in self.cfg["aircraft"]}
            for fu in cf.as_completed(futs):
                try:
                    per_ac[futs[fu]] = fu.result()
                except Exception as e:  # noqa: BLE001
                    per_ac[futs[fu]] = {"aircraft": futs[fu], "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
                    self.log(f"[{futs[fu]}] ERROR {e}")
        wall = time.perf_counter() - t0
        entries = [e for r in per_ac.values() for e in r.get("trajectories", [])]
        if entries:
            trajectory.write_index(self.traj_dir, self.cfg["run_id"], entries)
        ordered = [per_ac[a["name"]] for a in self.cfg["aircraft"]]
        sims_c = sum(r.get("this_session", {}).get("sims_computed", 0) for r in ordered)
        uniq = sum(r.get("this_session", {}).get("unique_sims", 0) for r in ordered)
        hits = sum(r.get("this_session", {}).get("cache_hits_mem", 0) + r.get("this_session", {}).get("cache_hits_disk", 0) for r in ordered)
        summary = {"run_id": self.cfg["run_id"], "session": self.session, "finished": _now(), "wall_s_this_session": wall,
                   "workers": self.workers, "schedule": self.schedule, "viz": "on" if self._viz_on() else "off",
                   "fidelity": self.fidelity, "multi_fidelity": self.mf if self.mf.get("enabled") else None,
                   "loadavg_1_5_15_start": load_start, "loadavg_1_5_15_end": os.getloadavg(),
                   "this_session": {"sims_computed": sims_c, "unique_sims": uniq, "cache_hits": hits,
                                    "cache_hit_rate": hits / uniq if uniq else None,
                                    "sims_per_s": sims_c / wall if wall else None},
                   "aircraft": ordered}
        _atomic_json(os.path.join(self.run_dir, f"summary.json"), summary, indent=2)
        self.cache.close()
        self.log(f"done in {wall:.1f}s  ({sims_c} sims computed, {hits}/{uniq} cache hits)  -> {self.run_dir}")
        return summary


def _sum_counts(ds):
    out: Dict[str, int] = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="batch JSON config (see README)")
    g.add_argument("--resume", metavar="RUN_DIR", help="resume a run from its stored config.json")
    ap.add_argument("--run-id", help="override run id (a new id = a fresh run that can still hit the cache)")
    ap.add_argument("--aircraft", help="comma-separated subset of the config's aircraft to run (with --config; "
                                       "part of the run identity)")
    ap.add_argument("--profile-for", action="append", default=[], metavar="AIRCRAFT=PROFILE",
                    help="use another profile from the config's 'profiles' for one aircraft (with --config; part of the "
                         "run identity), e.g. --profile-for f16=phase1_f16_kialt03")
    ap.add_argument("--seed", type=int, help="override the GA seed (with --config; part of the run identity). "
                                             "The scenario set follows only if the config has no explicit scenario_seed")
    ap.add_argument("--workers", type=int, help="override workers (does not change results)")
    ap.add_argument("--schedule", choices=["concurrent", "sequential"], help="override schedule (does not change results)")
    ap.add_argument("--no-cache", action="store_true", help="disable the on-disk cache for this session")
    ap.add_argument("--viz", choices=["on", "off"], help="on = record a trajectory on every evaluation + live best file "
                                                        "(slow); off (default) = no logging in the hot path, trajectories "
                                                        "re-flown afterwards. Does not change results")
    ap.add_argument("--fidelity", choices=list(fid_mod.FIDELITIES), help="rigid (default) | reduced (FD flexeval: v1 wing "
                                                                        "on the projected v2 genome) | full (FD flexeval: "
                                                                        "flex v2) | full_a1 (FD flexeval_a1: P3-A1 64-strip "
                                                                        "model) | full_a1_b1 (FD flexeval_b1: P3-B1 "
                                                                        "planform on A1). Part of the run identity")
    ap.add_argument("--multi-fidelity", action="store_true", help="screen everyone at --screen, re-score top-k + elites "
                                                                 "at --fidelity")
    ap.add_argument("--screen", help="screen fidelity for --multi-fidelity (default reduced); comma list for a ladder, "
                                     "e.g. rigid,reduced (= rigid -> reduced -> full)")
    ap.add_argument("--top-k", type=int, help="individuals re-scored at --fidelity per generation (default 4)")
    ap.add_argument("--struct-genes", action="store_true", help="add FD's structural genes to the genome")
    ap.add_argument("--model-versions", action="store_true", help="print each aircraft's model_version per fidelity of its "
                    "ladder next to the config's pin_model_version, then exit (read-only: no run dir, no cache, no flight)")
    a = ap.parse_args(argv)
    ident = a.seed is not None or a.aircraft or a.profile_for or a.fidelity or a.multi_fidelity or a.screen or \
        a.top_k is not None or a.struct_genes
    if a.resume and ident:
        ap.error("--seed / --aircraft / --profile-for / --fidelity / --multi-fidelity / --screen / --top-k / "
                 "--struct-genes only apply to --config (a resumed run keeps its stored config)")
    if a.resume:
        with open(os.path.join(a.resume, "config.json")) as f:
            cfg = json.load(f)["resolved"]
    else:
        with open(a.config) as f:
            user = json.load(f)
        if a.run_id:
            user["run_id"] = a.run_id
        if a.seed is not None:
            user["seed"] = a.seed
        if a.fidelity:
            user["fidelity"] = a.fidelity
        if a.multi_fidelity or a.screen or a.top_k is not None:
            mfc = dict(user.get("multi_fidelity") or {})
            if a.multi_fidelity:
                mfc["enabled"] = True
            if a.screen:
                mfc["screen"] = a.screen.split(",") if "," in a.screen else a.screen
            if a.top_k is not None:
                mfc["top_k"] = a.top_k
            user["multi_fidelity"] = mfc
        if a.struct_genes:
            user["struct_genes"] = True
        for spec in a.profile_for:
            ac_name, _, prof = spec.partition("=")
            if prof not in user.get("profiles", {}):
                ap.error(f"--profile-for {spec}: no profile {prof!r} in the config")
            acs = [({"name": x} if isinstance(x, str) else dict(x)) for x in user.get("aircraft", DEFAULTS["aircraft"])]
            if ac_name not in [x["name"] for x in acs]:
                ap.error(f"--profile-for {spec}: aircraft {ac_name!r} not in the config")
            user["aircraft"] = [{**x, "profile": prof} if x["name"] == ac_name else x for x in acs]
        if a.aircraft:
            want = [x.strip() for x in a.aircraft.split(",") if x.strip()]
            names = [(x if isinstance(x, str) else x["name"]) for x in user.get("aircraft", DEFAULTS["aircraft"])]
            bad = sorted(set(want) - set(names))
            if bad:
                ap.error(f"--aircraft {bad} not in the config (has {names})")
            user["aircraft"] = [x for x, n in zip(user.get("aircraft", DEFAULTS["aircraft"]), names) if n in want]
            for k in ("multi_fidelity_per_aircraft", "pin_model_version"):
                if user.get(k):
                    user[k] = {n: v for n, v in user[k].items() if n in want}
        cfg = resolve_config(user, os.path.splitext(os.path.basename(a.config))[0])
    if a.model_versions:
        print(json.dumps(model_versions_report(cfg), indent=1))
        return
    b = Batch(cfg)
    # execution-only overrides: not part of the resolved config (they cannot change results)
    if a.workers:
        b.workers = a.workers
    if a.schedule:
        b.schedule = a.schedule
    if a.viz:
        b.viz = a.viz
    if a.no_cache:
        b.cache.close()
        b.cache = cache_mod.EvalCache(None)
    summary = b.run()
    if any("error" in r for r in summary["aircraft"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
