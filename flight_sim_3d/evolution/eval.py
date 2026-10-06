"""evolution.eval: the single source of truth for evaluating one genome.

The GA hot path (batch workers, via task()) and every re-flight (trajectory export, Sim Bridge replay, via
evaluate()) go through evaluate_one() -> evolution.fidelity.evaluate_scenario() -> evolution.sim.simulate().

    evaluate(genome, aircraft, scenario, run_cfg, recorder=None, *, fidelity=None)
        -> {cost, per_scenario_cost, terms, terms_available, status, feasible, feasibility_fidelity,
            fidelity, model_version, scenario_ids, scenarios, per_scenario, genome, ...}

* genome:   decoded {gene: value} (genomes.jsonl "genome"), or a normalized list in run.json aircraft[].genes order
* aircraft: name in run.json aircraft[]
* scenario: a run.json scenarios[] entry (dict: everything incl. steps / ramp parameters / ramp_plan; it is used as
            given, nothing is rebuilt), its id ("c172x:s1") or index, a list of these, or None = all of the
            aircraft's scenario_ids. A plain sim.Scenario dict (to_dict()) works too.
* run_cfg:  run.json (dict, or a path to the run dir / run.json). run_cfg["fidelity"] picks the fidelity unless the
            fidelity keyword is given.
* recorder: optional recorder(t, fdm[, flex_state]); called once at t=0 after IC + trim (before the first step),
            then after every 1/120 s step with t = (k+1)*DT; recorder.final(t_end, fdm[, flex_state]) at the end if
            defined. Read-only (writes raise). flex_state (reduced/full only) is evolution.fidelity.FlexState.
            recorder=None is the fast path; a recorder never changes the result (tested bit for bit).
* cost = float(numpy.mean(per_scenario_cost)) at every fidelity; for one scenario it is that scenario's cost.

model_version is returned with every result; if it differs from the one recorded in run.json for the aircraft (e.g.
FD changed jsbsim_root or full moved to v2) the result says so ("model_version_logged", "model_version_match").
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from . import fidelity as fid_mod
from . import genome as genome_mod
from . import runinfo
from . import sim


# ----------------------------------------------------------------------------- run config helpers
def load_run_cfg(run) -> Dict:
    if isinstance(run, dict):
        return run
    p = run
    if os.path.isdir(p):
        p = os.path.join(p, "run.json")
    with open(p) as f:
        return json.load(f)


def _aircraft_entry(run_cfg: Dict, aircraft: str) -> Dict:
    for a in run_cfg["aircraft"]:
        if a["name"] == aircraft:
            return a
    raise KeyError(f"aircraft {aircraft!r} not in run {run_cfg.get('run_id')}")


def _profile_d(entry: Dict) -> Dict:
    return entry.get("resolved_profile") or entry["profile"]


def gene_groups(entry: Dict) -> Dict[str, str]:
    """gene name -> 'gains' | 'struct' (from run.json; default: pitch/altitude/heading genes are gains)."""
    if entry.get("genes"):
        return {g["name"]: g.get("group", "gains") for g in entry["genes"]}
    P = sim.Profile.from_dict(_profile_d(entry))
    return {g.name: "gains" for g in genome_mod.make_schema(P.gain_bounds, P.gene_kinds, P.heading_hold)}


def schema_for(entry: Dict) -> List[genome_mod.Gene]:
    if entry.get("genes"):
        return [genome_mod.Gene(g["name"], g["min"], g["max"], kind=g["kind"],
                                zero_band=g["zero_band"] if g.get("zero_band") is not None else 0.05) for g in entry["genes"]]
    P = sim.Profile.from_dict(_profile_d(entry))
    return genome_mod.make_schema(P.gain_bounds, P.gene_kinds, P.heading_hold)


def scenario_dicts(aircraft: str, scenario, run_cfg: Dict) -> List[Dict]:
    """Resolve the `scenario` argument to run.json scenario entries (dicts, used as given)."""
    entry = _aircraft_entry(run_cfg, aircraft)
    allsc = {s["id"]: s for s in run_cfg.get("scenarios", []) if s.get("aircraft", aircraft) == aircraft}
    ids = entry.get("scenario_ids") or list(allsc)
    if scenario is None:
        return [allsc[i] for i in ids]
    items = scenario if isinstance(scenario, (list, tuple)) else [scenario]
    out = []
    for s in items:
        if isinstance(s, dict):
            out.append(s)
        elif isinstance(s, int) and not isinstance(s, bool):
            out.append(allsc[ids[s]])
        else:
            key = str(s)
            if key not in allsc:      # tolerate bare index strings ("1")
                key = ids[int(key)]
            out.append(allsc[key])
    return out


def scenario_object(aircraft: str, scenario, run_cfg: Dict) -> sim.Scenario:
    """sim.Scenario (target(t), target_cmd(t), target_rate(t)) for one scenario entry or id; nothing is re-drawn."""
    run_cfg = load_run_cfg(run_cfg)
    return sim.Scenario.from_dict(runinfo.scenario_fields(scenario_dicts(aircraft, scenario, run_cfg)[0]))


# ----------------------------------------------------------------------------- the one evaluation path
def split_values(values: Dict[str, float], groups: Dict[str, str]):
    gains = {k: float(v) for k, v in values.items() if groups.get(k, "gains") == "gains"}
    struct = {k: float(v) for k, v in values.items() if groups.get(k) == "struct"} or None
    return gains, struct


def evaluate_one(profile_d: Dict, gains: Dict[str, float], struct: Optional[Dict[str, float]], sc_d: Dict,
                 fidelity: str = "rigid", recorder=None, record: bool = False, sample_hz: float = 30.0) -> Dict:
    """One genome x one scenario at one fidelity (JSON-able; what the batch caches)."""
    return fid_mod.evaluate_scenario(profile_d, gains, struct, runinfo.scenario_fields(sc_d), fidelity,
                                     recorder=recorder, record=record, sample_hz=sample_hz)


def task(profile_d: Dict, gains: Dict[str, float], struct: Optional[Dict[str, float]], sc_d: Dict, fidelity: str,
         viz: bool = False, sample_hz: float = 30.0) -> Dict:
    """Batch worker entry (picklable). viz=True records the ga-flightsim-traj/1 trajectory on every evaluation."""
    return evaluate_one(profile_d, gains, struct, sc_d, fidelity, record=viz, sample_hz=sample_hz)


def aggregate(per: Sequence[Dict]) -> Dict:
    return fid_mod.aggregate(per)


def describe(profile_d: Dict, fidelities: Sequence[str]) -> Dict:
    """Worker task run once per aircraft at batch start: model_files_sha + model_version per fidelity (for run.json,
    the cache key and the per-result check). A flex fidelity that cannot be built is reported, not raised."""
    P = sim.Profile.from_dict(profile_d)
    out = {"model_files_sha": sim.model_files_sha(P.aircraft, P.aircraft_root), "model_version": {}}
    for f in fidelities:
        try:
            out["model_version"][f] = fid_mod.model_version(profile_d, f)
        except Exception as e:  # noqa: BLE001  (FidelityUnavailable, FD import errors)
            out["error"] = f"{f}: {type(e).__name__}: {e}"
    return out


def evaluate(genome, aircraft: str, scenario, run_cfg, recorder=None, *, fidelity: Optional[str] = None) -> Dict:
    run_cfg = load_run_cfg(run_cfg)
    entry = _aircraft_entry(run_cfg, aircraft)
    fid = fidelity or run_cfg.get("fidelity") or "rigid"
    if not isinstance(genome, dict):
        genome = genome_mod.decode(list(genome), schema_for(entry))
    gains, struct = split_values(genome, gene_groups(entry))
    prof_d = _profile_d(entry)
    scs = scenario_dicts(aircraft, scenario, run_cfg)
    per = [evaluate_one(prof_d, gains, struct, s, fid, recorder=recorder) for s in scs]
    out = aggregate(per)
    out["per_scenario_cost"] = [p["cost"] for p in per]
    out["scenario_ids"] = [s.get("id") for s in scs]
    out["scenarios"] = scs
    out["per_scenario"] = per
    out["genome"] = dict(genome)
    out["aircraft"] = aircraft
    if len(per) == 1:
        for k in ("t_end", "track", "effort", "comfort", "heading_rms", "hdg_drift_deg"):
            if k in per[0]:
                out[k] = per[0][k]
    logged = (run_cfg.get("model_version") or {}).get(aircraft) if fid == run_cfg.get("fidelity", "rigid") else \
        entry.get("screen_model_version") if (run_cfg.get("multi_fidelity") or {}).get("screen") == fid else None
    if logged:
        out["model_version_logged"] = logged
        out["model_version_match"] = logged == out["model_version"]
    return out
