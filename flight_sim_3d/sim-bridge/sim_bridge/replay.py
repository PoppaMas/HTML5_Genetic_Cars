"""Re-fly selected genomes of an Evolution Runner run with trajectory logging (ER's fast mode logs nothing).

    python replay.py --run phase1-s1 --gens 0,9,19 [--aircraft c172x,T38] [--scenario all|<scenario id>|<position>]
                     [--fidelity rigid|reduced|full] [--hz 30] [--html] [--out DIR]
    python replay.py --run phase1v5-s1 --best-per-gen | --gens 19 --elites | --ids 'T38:g19:r0,c172x:g9:best'

Interface (``--interface auto``):
  * ``er`` (primary): <run>/run.json + <run>/genomes.jsonl (ER's real files, schema ga-flightsim-run/1 /
    ga-flightsim-genomes/1) + ``evolution.eval.evaluate``. Auto-selected whenever run.json exists.
    ``--eval-module`` overrides the module (dotted name or .py path; the fixture tests use it).
  * ``adapter`` (fallback, only for runs WITHOUT run.json, e.g. bench_jets-j1): config.json + checkpoints/ via
    sim_bridge.er_adapter on top of ER's evolution.sim.

Ids are opaque strings everywhere: individual_id ("c172x:g19:best", "T38:g19:r0") and scenario ids ("c172x:s0") are
matched as given and never parsed. The scenarios of a row are its own ``scenario_ids`` (else its aircraft's); a row's
``per_scenario_cost`` is aligned with that list. ``scenario_index`` in our outputs is the position in that list
(that is what ER writes in its trajectory files); ``scenario_id`` is the string.

Output (default <SIMBRIDGE_DATA_DIR>/replays/<run_id>/<replay_id>/; ER's tree stays read-only, ``--out`` overrides):
  trajectories/traj_*.json (ga-flightsim-traj/2; /1 still accepted), trajectories/index.json, replay_manifest.json, viewer.html (--html)

Cost is "lower is better" (run.json fitness_sense is exactly "min"; anything starting with "max" would flip it).
A replay at the row's own fidelity with the same model_version must reproduce the logged cost exactly (relative error
<= ``--tol``, default 1e-6); otherwise the replay is flagged and the exit code is 2. If the model_version differs a
mismatch is "expected" (both versions recorded); an exact cost despite a different model_version is still "match" and
the manifest says ``model_version_match: false`` (legitimate: ER's rigid model_version hashes the whole
aircraft/<model>/ folder, so an edit that does not change the flown physics changes it). At another fidelity than the
row's the logged screen_cost (if screen_fidelity equals it) or ladder_cost[fid] is used, else "not comparable".

model_version per fidelity (logged_model_version): row ladder_model_version[fid] > the row's own model_version (rows
not rescored at full keep the cost / fidelity / model_version of the highest stage they reached) > screen_model_version
> aircraft ladder_model_version[fid] > run.json model_version. Every replayed version is also checked against FD's
published current strings (SIMBRIDGE_FD_MODEL_VERSIONS, default flight-dynamics/v2_results/
model_versions_post_mass.json): ``model_version_current``; cached FD models are keyed on that version.

Full fidelity: FD's exact FE nodal values (fd-flexbody-nodes/1) are computed from the modal state ER hands the
recorder (sim_bridge.fd_nodes) and mapped by sim_bridge.v2_map; ``--no-nodes`` forces the tip-only estimate.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import datetime as _dt
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from sim_bridge import paths  # noqa: E402

paths.ensure_import_paths()
SIM_BRIDGE = paths.SIM_BRIDGE
TEAM_ROOT = paths.TEAM_ROOT
DEFAULT_RUNS_ROOT = paths.RUNS_ROOT

from sim_bridge import er_adapter  # noqa: E402
from sim_bridge.recorder import TrajRecorder  # noqa: E402

TRAJ_SCHEMA = "ga-flightsim-traj/2"
INDEX_SCHEMA = "ga-flightsim-traj-index/1"
MANIFEST_SCHEMA = "sim-bridge-replay-manifest/2"
FIDELITIES = ("rigid", "reduced", "full")
FT = 0.3048


# ------------------------------------------------------------------ row fields (as ER writes them)
def sense_of(cfg_or_value) -> str:
    v = cfg_or_value.get("fitness_sense", "min") if isinstance(cfg_or_value, dict) else cfg_or_value
    return "max" if str(v or "min").strip().lower().startswith("max") else "min"


def _f(v) -> Optional[float]:
    return None if v is None else float(v)


def row_cost(row: Dict) -> Optional[float]:
    return _f(row.get("cost", row.get("fitness")))


def row_per_scenario(row: Dict) -> Optional[List[float]]:
    v = row.get("per_scenario_cost", row.get("per_scenario_fitness"))
    return None if v is None else [_f(x) for x in v]


def aircraft_entry(run_cfg: Dict, aircraft: str) -> Dict:
    return next((a for a in run_cfg.get("aircraft", []) if a.get("name") == aircraft), {})


def row_scenario_ids(row: Dict, run_cfg: Dict) -> List[str]:
    ids = row.get("scenario_ids")
    if ids is None:
        ids = aircraft_entry(run_cfg, row["aircraft"]).get("scenario_ids")
    if ids is None:
        ids = [s["id"] for s in run_cfg.get("scenarios", []) if s.get("aircraft", row["aircraft"]) == row["aircraft"]]
    return [str(s) for s in ids]


def row_fidelity(row: Dict, run_cfg: Dict) -> str:
    return str(row.get("fidelity") or run_cfg.get("fidelity") or "rigid")


def row_model_version(row: Dict, run_cfg: Dict) -> Optional[str]:
    mv = row.get("model_version")
    if mv is None:
        m = run_cfg.get("model_version")
        mv = m.get(row["aircraft"]) if isinstance(m, dict) else m
    if mv is None:
        mv = aircraft_entry(run_cfg, row["aircraft"]).get("model_version")
    return mv


def _per_fid(v, fid):
    """ladder_* field -> value at fid (dict {fid: v}, or a list of [fid, v] / {fidelity, ...} entries)."""
    if isinstance(v, dict):
        return v.get(fid)
    if isinstance(v, list):
        for e in v:
            if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0] == fid:
                return e[1]
            if isinstance(e, dict) and e.get("fidelity") == fid:
                return e.get("model_version", e.get("cost"))
    return None


def logged_model_version(row: Dict, run_cfg: Dict, fid: str) -> Tuple[Optional[str], Optional[str]]:
    """(model_version the run logged for this row at fidelity fid, where it came from)."""
    mv = _per_fid(row.get("ladder_model_version"), fid)
    if mv:
        return mv, f"ladder_model_version.{fid}"
    if fid == row_fidelity(row, run_cfg) and row.get("model_version"):
        return row["model_version"], "model_version"
    if row.get("screen_fidelity") == fid and row.get("screen_model_version"):
        return row["screen_model_version"], "screen_model_version"
    ac = aircraft_entry(run_cfg, row.get("aircraft"))
    mv = _per_fid(ac.get("ladder_model_version"), fid)
    if mv:
        return mv, f"aircraft.ladder_model_version.{fid}"
    if fid == row_fidelity(row, run_cfg):
        mv = row_model_version(row, run_cfg)
        if mv:
            return mv, "run.json model_version"
    return None, None


def fd_current_model_version(aircraft: str, fid: str) -> Optional[str]:
    return (paths.fd_model_versions().get(aircraft) or {}).get(fid)


def row_eval_seed(row: Dict, run_cfg: Dict):
    return row.get("eval_seed", run_cfg.get("eval_seed", run_cfg.get("scenario_seed")))


def logged_at(row: Dict, run_cfg: Dict, fid: str) -> Dict:
    """What the run logged for this row at fidelity ``fid``: {cost, per_scenario, model_version, mv_source, source}."""
    mv, mv_src = logged_model_version(row, run_cfg, fid)
    base = {"model_version": mv, "mv_source": mv_src}
    if fid == row_fidelity(row, run_cfg):
        return dict(base, cost=row_cost(row), per_scenario=row_per_scenario(row), source="cost")
    if row.get("screen_fidelity") == fid and row.get("screen_cost") is not None:
        sp = row.get("screen_per_scenario_cost")
        return dict(base, cost=_f(row["screen_cost"]), per_scenario=None if sp is None else [_f(x) for x in sp],
                    source="screen_cost")
    lc = _per_fid(row.get("ladder_cost"), fid)
    if lc is not None:
        return dict(base, cost=_f(lc), per_scenario=None, source=f"ladder_cost.{fid}")
    return dict(base, cost=None, per_scenario=None, source=None)


def safe_name(s) -> str:
    """Opaque id -> file-name-safe token (the raw id stays in the doc, index and manifest)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-") or "id"


# ------------------------------------------------------------------ interface loading
def _load_module(spec: str):
    if spec.endswith(".py") or os.path.sep in spec:
        name = "replay_eval_" + os.path.splitext(os.path.basename(spec))[0]
        sp = importlib.util.spec_from_file_location(name, spec)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        return mod
    return importlib.import_module(spec)


def _read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def eval_run_cfg(run_cfg: Dict) -> Tuple[Dict, List[str]]:
    """Copy of run.json for evaluate(): aircraft_root resolved here. Relative roots (ER's paths_relative_to = team
    root) go through evolution.sim.abs_root when it is importable and agrees with an existing directory, else
    paths.resolve_model_root; foreign absolute roots are remapped onto FLIGHT_DYNAMICS_DIR. In memory only."""
    cfg = copy.deepcopy(run_cfg)
    notes = []
    try:
        from evolution import sim as _er_sim
        abs_root = getattr(_er_sim, "abs_root", None)
    except Exception:  # noqa: BLE001 - fixture / adapter runs without ER
        abs_root = None
    for a in cfg.get("aircraft", []):
        for holder in (a, a.get("resolved_profile") or {}, a.get("profile") if isinstance(a.get("profile"), dict) else {}):
            if holder.get("aircraft_root"):
                src = holder["aircraft_root"]
                if abs_root is not None and not os.path.isabs(src):
                    er_p = abs_root(src)
                    if er_p and os.path.isdir(er_p):
                        notes.append(f"{a.get('name')}: relative {src!r} -> evolution.sim.abs_root")
                        holder["aircraft_root"] = er_p
                        continue
                new, note = paths.resolve_model_root(src)
                if note:
                    notes.append(f"{a.get('name')}: {note}")
                holder["aircraft_root"] = new
    return cfg, sorted(set(notes))


def resolve_interface(run_dir: str, mode: str = "auto", eval_module: str = "evolution.eval") -> Dict:
    """-> dict(kind, run_cfg, rows, evaluate, module, module_spec, fallback_reason)."""
    run_json = os.path.join(run_dir, "run.json")
    if mode in ("auto", "er") and os.path.exists(run_json):
        with open(run_json) as f:
            run_cfg = json.load(f)
        gpath = os.path.join(run_dir, run_cfg.get("genomes_file") or "genomes.jsonl")
        if not os.path.exists(gpath):
            raise SystemExit(f"{run_dir}: run.json present but {os.path.basename(gpath)} missing")
        try:
            mod = _load_module(eval_module)
            evaluate = mod.evaluate
        except (ImportError, AttributeError, FileNotFoundError) as e:
            raise SystemExit(f"{run_dir} has run.json (ER format) but {eval_module} cannot be loaded: "
                             f"{type(e).__name__}: {e}")
        cfg_eval, notes = eval_run_cfg(run_cfg)
        return {"kind": "er", "run_cfg": run_cfg, "run_cfg_eval": cfg_eval, "path_notes": notes,
                "rows": _read_jsonl(gpath), "evaluate": evaluate, "module": mod, "module_spec": eval_module,
                "module_file": getattr(mod, "__file__", None), "fallback_reason": None}
    if mode == "er":
        raise SystemExit(f"--interface er: {run_dir} has no run.json")
    if not os.path.exists(os.path.join(run_dir, "config.json")):
        raise SystemExit(f"{run_dir}: neither run.json (ER format) nor a legacy config.json")
    run_cfg = er_adapter.load_run_cfg(run_dir)
    return {"kind": "adapter", "run_cfg": run_cfg, "run_cfg_eval": run_cfg, "path_notes": [],
            "rows": er_adapter.genome_rows(run_dir, run_cfg), "evaluate": er_adapter.evaluate, "module": er_adapter,
            "module_spec": "sim_bridge.er_adapter", "module_file": er_adapter.__file__,
            "fallback_reason": "no run.json (legacy run)" if mode == "auto" else "--interface adapter"}


_IFACE_CACHE: Dict[tuple, Dict] = {}


def _iface_cached(run_dir: str, kind: str, eval_module: str) -> Dict:
    key = (os.path.abspath(run_dir), kind, eval_module)
    if key not in _IFACE_CACHE:
        _IFACE_CACHE[key] = resolve_interface(run_dir, kind, eval_module)
    return _IFACE_CACHE[key]


# ------------------------------------------------------------------ selection (ids opaque)
def best_flags(rows: Sequence[Dict], sense: str = "min") -> List[bool]:
    """is_best per row: ER's flag where the rows carry it, else the best cost of each (aircraft, generation)."""
    groups: Dict[tuple, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["aircraft"], r["generation"]), []).append(i)
    out = [False] * len(rows)
    for idx in groups.values():
        if any("is_best" in rows[i] for i in idx):
            for i in idx:
                out[i] = bool(rows[i].get("is_best"))
        else:
            scored = [i for i in idx if row_cost(rows[i]) is not None]
            if scored:
                pick = (max if sense == "max" else min)(scored, key=lambda i: row_cost(rows[i]))
                out[pick] = True
    return out


def select_rows(rows: List[Dict], gens=None, best_per_gen=False, elites=False, ids=None, aircraft=None,
                sense: str = "min") -> List[Dict]:
    flags = best_flags(rows, sense)
    pairs = [(r, b) for r, b in zip(rows, flags) if not aircraft or r["aircraft"] in aircraft]
    if ids:
        want = [str(i) for i in ids]
        got = {str(r.get("individual_id")) for r, _ in pairs}
        missing = [i for i in want if i not in got]
        if missing:
            raise SystemExit(f"unknown individual ids: {missing}")
        out = [r for i in dict.fromkeys(want) for r, _ in pairs if str(r.get("individual_id")) == i]
    else:
        if gens is not None:
            pairs = [(r, b) for r, b in pairs if int(r["generation"]) in gens]
        if elites:
            out = [r for r, b in pairs if b or r.get("is_elite")]
        else:  # --gens or --best-per-gen: the best of each selected generation
            out = [r for r, b in pairs if b]
    best_ids = {id(r) for r, b in zip(rows, flags) if b}
    for r in out:
        r["_is_best"] = id(r) in best_ids
    return sorted(out, key=lambda r: (r["aircraft"], int(r["generation"]), r.get("rank") if r.get("rank") is not None else 0,
                                      str(r.get("individual_id"))))


def resolve_scenarios(row: Dict, run_cfg: Dict, token: Optional[str]) -> List[Tuple[int, str]]:
    """[(position in the row's scenario_ids, scenario id)] for --scenario. Tokens: 'all', an exact scenario id, or a
    position (digits). A token that is another aircraft's scenario id is skipped for this row."""
    sids = row_scenario_ids(row, run_cfg)
    if token in (None, "", "all"):
        return list(enumerate(sids))
    known = {str(s.get("id")) for s in run_cfg.get("scenarios", [])}
    out = []
    for t in [x.strip() for x in str(token).split(",") if x.strip()]:
        if t in sids:
            out.append((sids.index(t), t))
        elif t.isdigit() and t not in known:
            k = int(t)
            if k >= len(sids):
                raise SystemExit(f"--scenario {t}: {row['aircraft']} has {len(sids)} scenarios {sids}")
            out.append((k, sids[k]))
        elif t not in known:
            raise SystemExit(f"--scenario {t!r}: not a scenario id of this run ({sorted(known)}) nor a position")
    return list(dict.fromkeys(out))


def scenario_entry(run_cfg: Dict, sid: str):
    for s in run_cfg.get("scenarios", []):
        if str(s.get("id")) == sid:
            return s
    raise SystemExit(f"scenario {sid!r} not in run.json scenarios[]")


# ------------------------------------------------------------------ evaluation (worker)
def _scenario_obj(mod, aircraft: str, scenario: Dict, run_cfg: Dict):
    """Scenario object the recorder queries for target / target_cmd / target_rate (ER's own, never re-drawn)."""
    for name in ("scenario_object", "make_scenario"):
        f = getattr(mod, name, None) if mod is not None else None
        if callable(f):
            o = f(aircraft, scenario, run_cfg)
            return o[0] if isinstance(o, tuple) else o
    return er_adapter.scenario_object(aircraft, scenario, run_cfg)[0]


def run_job(job: Dict) -> Dict:
    """One (genome, scenario) re-flight with logging. Runs in a worker process."""
    iface = _iface_cached(job["run_dir"], job["interface"], job["eval_module"])
    run_cfg = dict(iface["run_cfg_eval"])
    run_cfg["fidelity"] = job["fidelity"]
    scen = scenario_entry(run_cfg, job["scenario_id"])
    sc_obj = _scenario_obj(iface.get("module"), job["aircraft"], scen, run_cfg)
    # Prefer FlexState /3 public API (.nodes / .v2_geometry); NodeSource is only a fallback for older ER.
    node_source, node_note = None, None
    prefer_flexstate = job["fidelity"] == "full" and job.get("nodes", True) and iface["kind"] == "er"
    if prefer_flexstate:
        node_note = ("FD FE nodal values via FlexState.nodes()/v2_geometry (evolution-flex-state/3); "
                     "NodeSource fallback only if FlexState has no public API")
        # Build NodeSource as fallback for older FlexState without .nodes(); recorder prefers FlexState when present
        try:
            from sim_bridge.fd_nodes import NodeSource
            node_source = NodeSource.for_genome(job["genome"], job["aircraft"], run_cfg,
                                                model_version=fd_current_model_version(job["aircraft"], "full"))
        except Exception as e:  # noqa: BLE001 - FlexState /3 still works without this
            node_note += f"; NodeSource unavailable ({type(e).__name__}: {e})"
    elif job["fidelity"] == "full":
        node_note = "nodal data disabled (--no-nodes)" if not job.get("nodes", True) else "adapter: no FD nodal data"
    elif job["fidelity"] == "reduced":
        node_note = ("reduced = FD flexwing v1: wings only, no node export; ER's modal wings kept, no tail / fin / "
                     "fuselage components (FD does not model them at reduced)")
    rec = TrajRecorder(sc_obj, job["hz"], float(run_cfg.get("sim_dt_s", 1 / 120)), timing=job["recorder_timing_mode"],
                       aircraft=job["aircraft"], node_source=node_source,
                       use_nodes=bool(job.get("nodes", True)))
    # pin=: ER raises RuntimeError if the evaluated model_version differs from the run's pin
    pin = _per_fid((run_cfg.get("pin_model_version") or {}).get(job["aircraft"]), job["fidelity"])
    t0 = time.perf_counter()
    try:
        import inspect
        ev = iface["evaluate"]
        kw = {}
        if pin is not None and "pin" in getattr(inspect.signature(ev), "parameters", {}):
            kw["pin"] = pin
        res = ev(job["genome"], job["aircraft"], scen, run_cfg, recorder=rec, **kw)
    except Exception as e:  # FidelityUnavailable, NotImplementedError, pin mismatch, ... -> reported, exit 3
        return {"job": job, "error": f"{type(e).__name__}: {e}"}
    rec.finish()
    wall = time.perf_counter() - t0
    api = (rec.v2_stats or {}).get("flex_api")
    if api == "flexstate3":
        node_note = "FD FE nodal values via FlexState.nodes()/v2_geometry (evolution-flex-state/3)"
    elif api == "nodesource" and prefer_flexstate:
        node_note = "FD FE nodal values via NodeSource fallback (older FlexState without .nodes())"
    elif api == "estimate" and prefer_flexstate:
        node_note = "no FD nodal data available; tail / fin / fuselage ESTIMATED from tip scalars"
    keep = {k: v for k, v in res.items() if isinstance(v, (int, float, str, bool, type(None)))}
    for k in ("telemetry_check", "per_scenario_cost", "scenario_ids", "terms"):
        if k in res:
            keep[k] = res[k]
    st = rec.structure
    return {"job": job, "result": keep, "channels": rec.channels, "rows": rec.rounded_rows(), "origin": rec.origin,
            "origin_source": rec.origin_source, "structure": st, "recorder_calls": rec.n_calls, "wall_s": wall,
            "node_status": rec.node_status() if st else None, "node_note": node_note,
            "v2_stats": rec.v2_stats if st else None,
            "scenario": sc_obj.to_dict() if hasattr(sc_obj, "to_dict") else dict(scen)}


# ------------------------------------------------------------------ output docs
def build_traj_doc(out: Dict, run_cfg: Dict, replay_id: str, sense: str, fitness_val: float) -> Dict:
    job, res, sc = out["job"], out["result"], out["scenario"]
    o = out["origin"]
    events = [{"t": 0.0, "type": "start", "detail": f"trimmed level flight {sc.get('h0_ft', 0):.0f} ft, {sc.get('speed_kts', 0):.0f} KCAS"
               + (f", throttle {out['rows'][0][15]:.3f}" if out["rows"] and out["rows"][0][15] is not None else "")}]
    steps = sc.get("steps") or []
    if steps:
        prev = steps[0][1]
        for ts, hs in steps:
            if hs != prev:
                events.append({"t": float(ts), "type": "target_change", "detail": f"altitude target {prev:.0f} -> {hs:.0f} ft ({hs * FT:.1f} m)"})
                prev = hs
    if sc.get("discrete_gust_fps"):
        events.append({"t": float(sc["discrete_gust_t_s"]), "type": "gust",
                       "detail": f"1-cosine vertical gust {sc['discrete_gust_fps']:+.1f} ft/s (down +) over {sc.get('discrete_gust_len_s', 3.0):.1f} s"})
    t_end = float(res.get("t_end", out["rows"][-1][0] if out["rows"] else 0.0))
    events.append({"t": t_end, "type": "end" if res.get("status") == "ok" else "terminated",
                   "detail": "completed" if res.get("status") == "ok" else str(res.get("status"))})
    events.sort(key=lambda e: e["t"])
    ac = aircraft_entry(run_cfg, job["aircraft"])
    doc = {
        "schema": TRAJ_SCHEMA, "run_id": run_cfg["run_id"], "aircraft": job["aircraft"],
        "jsbsim_version": run_cfg.get("jsbsim_version"), "git_sha": run_cfg.get("git_sha"), "seed": ac.get("seed", run_cfg.get("seed")),
        "generation": int(job["generation"]), "individual_id": job["individual_id"], "rank": job.get("rank"),
        "is_best": job.get("is_best"), "is_elite": job.get("is_elite"),
        "fitness": float(fitness_val), "cost": float(fitness_val), "fitness_sense": sense,
        "scenario_index": job["scenario_index"], "scenario_id": job["scenario_id"],
        "scenario_cost": float(res["cost"]), "status": res.get("status"),
        "genome": {k: float(v) for k, v in job["genome"].items()},
        "frame": {"origin_lat_deg": o["lat_deg"], "origin_lon_deg": o["lon_deg"], "origin_alt_m": o["alt_m"],
                  "axes": "ENU metres, x=east y=north z=up", "attitude": "quat body->ENU [w,x,y,z]",
                  "body_axes": "JSBSim body FRD: x forward, y right wing, z down",
                  "attitude_source_of_truth": "quaternion; phi/theta/psi are HUD-only (JSBSim Euler ZYX vs local NED)"} if o else {},
        "units": {"phi": "rad", "theta": "rad", "psi": "rad", "controls": "norm -1..1, throttle 0..1"},
        "dt_s": 1.0 / job["hz"], "sim_dt_s": float(run_cfg.get("sim_dt_s", 1 / 120)), "sample_hz": job["hz"],
        "target": {"alt_m": steps[0][1] * FT if steps else None,
                   "steps": [{"t": float(ts), "alt_m": float(hs * FT)} for ts, hs in steps],
                   "speed_kcas": sc.get("speed_kts"),
                   "reference": ("instant steps" if sc.get("ramp_fpm") is None else
                                 f"ramped at {sc['ramp_fpm']:g} fpm" + (f" with {sc['ramp_accel_g']:g} g corners" if sc.get("ramp_accel_g") else "")
                                 + " (channel target_alt_m = reference, target_cmd_alt_m = command)")},
        "wind": {"north_mps": (sc.get("wind_north_fps") or 0) * FT, "east_mps": (sc.get("wind_east_fps") or 0) * FT,
                 "gust_sigma_mps": (sc.get("gust_sigma_fps") or 0) * FT},
        "events": events,
        "replay": {"of_run": run_cfg["run_id"], "replay_id": replay_id, "fidelity": job["fidelity"],
                   "model_version": res.get("model_version"), "logged_model_version": job.get("logged_model_version"),
                   "logged_fidelity": job.get("row_fidelity"), "logged_source": job.get("logged_source"),
                   "logged_scenario_cost": job.get("logged_scenario_cost"),
                   "replayed_scenario_cost": float(res["cost"]), "eval_seed": job.get("eval_seed"),
                   "recorder_timing": job.get("recorder_timing"), "origin_source": out.get("origin_source"),
                   "fd_current_model_version": fd_current_model_version(job["aircraft"], job["fidelity"]),
                   "model_version_current": (None if fd_current_model_version(job["aircraft"], job["fidelity"]) is None
                                             else res.get("model_version") == fd_current_model_version(job["aircraft"], job["fidelity"])),
                   **({"node_status": out.get("node_status"), "node_note": out.get("node_note"),
                       "v2_map_version": _v2_version()} if out.get("structure") else {})},
        "channels": out["channels"], "data": out["rows"],
    }
    if out.get("structure"):
        doc["structure"] = out["structure"]
    return doc


def _v2_version():
    from sim_bridge import v2_map
    return v2_map.V2_MAP_VERSION


def rel_err(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(b), 1e-300)


def compare(logged: Dict, replayed_mean: Optional[float], replayed_per: Dict[str, Tuple[int, float]], fid: str,
            row_fid: str, mv_replay, tol: float) -> Dict:
    """logged = logged_at(...); replayed_per = {scenario_id: (position, cost)} (ids opaque, positions index logged)."""
    lp_all = logged["per_scenario"]
    per = {}
    for sid, (k, c) in replayed_per.items():
        lp = lp_all[k] if lp_all is not None and k < len(lp_all) else None
        per[sid] = {"scenario_index": k, "logged": lp, "replayed": c, "rel_err": rel_err(c, lp)}
    errs = [v["rel_err"] for v in per.values() if v["rel_err"] is not None]
    re_mean = rel_err(replayed_mean, logged["cost"])
    if re_mean is not None:
        errs.append(re_mean)
    worst = max(errs) if errs else None
    mv_logged = logged["model_version"]
    mv_match = None if mv_logged is None or mv_replay is None else mv_replay == mv_logged
    if worst is None:
        verdict = "no logged cost" if fid == row_fid else "not comparable (fidelity differs, nothing logged at it)"
    elif worst <= tol:
        verdict = "match"
    elif mv_match is False:
        verdict = "mismatch expected (model_version differs)"
    else:
        verdict = "MISMATCH"
    return {"logged_cost": logged["cost"], "logged_source": logged["source"], "replayed_cost": replayed_mean,
            "rel_err": re_mean, "per_scenario": per, "max_rel_err": worst, "logged_fidelity": row_fid,
            "replay_fidelity": fid, "logged_model_version": mv_logged, "logged_model_version_source": logged.get("mv_source"),
            "replay_model_version": mv_replay, "model_version_match": mv_match, "verdict": verdict}


def traj_file_name(row: Dict, run_id: str, scenario_id: str, *, tag_individual: bool, tag_scenario: bool) -> str:
    fn = f"traj_{safe_name(row['aircraft'])}_{safe_name(run_id)}_g{int(row['generation'])}"
    if tag_individual:
        fn += f"__{safe_name(row.get('individual_id'))}"
    if tag_scenario:
        fn += f"__{safe_name(scenario_id)}"
    return fn + ".json"


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run id (directory under --runs-root) or a run directory path")
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT, help="default $SIMBRIDGE_RUNS_ROOT or <team>/evolution/runs")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--gens", help="comma list of generations (best of each; with --elites all elites)")
    sel.add_argument("--best-per-gen", action="store_true", help="the is_best row of every generation")
    sel.add_argument("--ids", help="comma list of individual_id (opaque strings, exact match)")
    ap.add_argument("--elites", action="store_true", help="elite rows (is_elite) as well as the best; combine with --gens/--best-per-gen")
    ap.add_argument("--aircraft", help="comma list (default: all)")
    ap.add_argument("--scenario", default="all", help="'all' (default, full cost check), scenario id(s) or position(s)")
    ap.add_argument("--fidelity", choices=FIDELITIES, help="default: each row's own fidelity")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--html", action="store_true", help="also write a standalone viewer.html (colab_viewer)")
    ap.add_argument("--out", help="output dir (default $SIMBRIDGE_DATA_DIR/replays/<run_id>/<replay_id>)")
    ap.add_argument("--replay-id")
    ap.add_argument("--interface", choices=("auto", "er", "adapter"), default="auto")
    ap.add_argument("--eval-module", default="evolution.eval")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--tol", type=float, default=1e-6, help="relative cost tolerance for an exact replay")
    ap.add_argument("--no-nodes", action="store_true", help="full fidelity: skip FD nodal data (tip-only estimate)")
    ap.add_argument("--compare-traj", help="reference trajectories dir for the channel check (default <run>/trajectories; 'none' to skip)")
    ap.add_argument("--recorder-timing", choices=("auto", "pre", "post"), default="auto",
                    help="adapter=pre-step; evolution.eval=post-step + t=0 call (ER contract) unless the module sets RECORDER_TIMING")
    return ap.parse_args(argv)


TIMING_DOC = {"pre": "pre-step (state at t, commands for [t,t+dt)) + final state",
              "post": "post-step + t=0 call after trim (evolution.eval contract); a row's controls come from the next "
                      "call = commands for [t,t+dt); recorder.final(t_end) closes the run"}


def main(argv=None) -> int:
    a = parse_args(argv)
    run_dir = a.run if os.path.isdir(a.run) and (os.path.sep in a.run or not os.path.isdir(os.path.join(a.runs_root, a.run))) \
        else os.path.join(a.runs_root, a.run)
    iface = resolve_interface(run_dir, a.interface, a.eval_module)
    _IFACE_CACHE[(os.path.abspath(run_dir), iface["kind"], a.eval_module)] = iface
    run_cfg, sense = iface["run_cfg"], sense_of(iface["run_cfg"])
    run_id = str(run_cfg["run_id"])
    gens = {int(g) for g in a.gens.split(",")} if a.gens else None
    rows = select_rows(iface["rows"], gens=gens, best_per_gen=a.best_per_gen, elites=a.elites,
                       ids=a.ids.split(",") if a.ids else None, aircraft=a.aircraft.split(",") if a.aircraft else None,
                       sense=sense)
    if not rows:
        raise SystemExit("no genomes selected")
    started = _dt.datetime.now().astimezone()
    sel_tag = (f"g{a.gens.replace(',', '.')}" if a.gens else "best" if a.best_per_gen else "ids") + ("-elites" if a.elites else "")
    replay_id = a.replay_id or f"{started:%Y%m%dT%H%M%S}-{a.fidelity or 'own'}-{sel_tag}"
    out_dir = a.out or os.path.join(paths.DATA_DIR, "replays", safe_name(run_id), safe_name(replay_id))
    if not a.out and os.path.realpath(out_dir).startswith(os.path.realpath(paths.EVOLUTION_ROOT) + os.sep):
        raise SystemExit("refusing to write under ER's tree without an explicit --out")
    traj_dir = os.path.join(out_dir, "trajectories")
    timing = a.recorder_timing
    if timing == "auto":
        timing = getattr(iface.get("module"), "RECORDER_TIMING", None) or ("pre" if iface["kind"] == "adapter" else "post")
        if iface["kind"] == "adapter" and hasattr(er_adapter, "recorder_timing"):
            timing = er_adapter.recorder_timing()
    jobs, row_scen = [], []
    for ri, r in enumerate(rows):
        fid = a.fidelity or row_fidelity(r, run_cfg)
        lg = logged_at(r, run_cfg, fid)
        scs = resolve_scenarios(r, run_cfg, a.scenario)
        if not scs:
            raise SystemExit(f"--scenario {a.scenario}: nothing for {r.get('individual_id')}")
        row_scen.append(scs)
        for k, sid in scs:
            lp = lg["per_scenario"]
            jobs.append({"run_dir": run_dir, "interface": iface["kind"], "eval_module": a.eval_module, "hz": a.hz,
                         "row": ri, "fidelity": fid, "row_fidelity": row_fidelity(r, run_cfg),
                         "logged_model_version": lg["model_version"], "logged_source": lg["source"],
                         "aircraft": r["aircraft"], "generation": int(r["generation"]),
                         "individual_id": None if r.get("individual_id") is None else str(r["individual_id"]),
                         "rank": r.get("rank"), "is_best": bool(r.get("_is_best")), "is_elite": bool(r.get("is_elite") or r.get("_is_best")),
                         "eval_seed": row_eval_seed(r, run_cfg),
                         "genome": r["genome"], "scenario_id": sid, "scenario_index": k,
                         "logged_scenario_cost": lp[k] if lp is not None and k < len(lp) else None,
                         "recorder_timing_mode": timing, "recorder_timing": TIMING_DOC[timing], "nodes": not a.no_nodes})
    n_sc = sorted({len(s) for s in row_scen})
    print(f"replay {run_id} via {iface['kind']} ({iface['module_spec']}): {len(rows)} genomes, {'/'.join(map(str, n_sc))} "
          f"scenarios each = {len(jobs)} flights -> {out_dir}", flush=True)
    for n in iface.get("path_notes") or []:
        print(f"  path: {n}")
    t0 = time.perf_counter()
    if a.workers > 1 and len(jobs) > 1:
        with cf.ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=mp.get_context("fork")) as ex:
            outs = list(ex.map(run_job, jobs))
    else:
        outs = [run_job(j) for j in jobs]
    wall = time.perf_counter() - t0
    errors = [o for o in outs if "error" in o]
    if errors:
        for o in errors:
            print(f"ERROR {o['job']['individual_id']} {o['job']['scenario_id']}: {o['error']}", file=sys.stderr)
        return 3
    os.makedirs(traj_dir, exist_ok=True)
    by_row: Dict[int, List[Dict]] = {}
    for o in outs:
        by_row.setdefault(o["job"]["row"], []).append(o)
    n_per_group: Dict[tuple, int] = {}
    for r in rows:
        n_per_group[(r["aircraft"], int(r["generation"]))] = n_per_group.get((r["aircraft"], int(r["generation"])), 0) + 1
    import numpy as np  # ER aggregates with np.mean in scenario_ids order
    entries, comparisons, flagged, used_names = [], [], False, set()
    for ri, r in enumerate(rows):
        os_ = sorted(by_row[ri], key=lambda o: o["job"]["scenario_index"])
        per = {o["job"]["scenario_id"]: (o["job"]["scenario_index"], float(o["result"]["cost"])) for o in os_}
        full = [k for k, _ in row_scen[ri]] == list(range(len(row_scenario_ids(r, run_cfg))))
        mean = float(np.mean([c for _, c in per.values()])) if full else None
        mv = os_[0]["result"].get("model_version")
        fid = os_[0]["job"]["fidelity"]
        cmpd = compare(logged_at(r, run_cfg, fid), mean, per, fid, row_fidelity(r, run_cfg), mv, a.tol)
        cmpd.update({"individual_id": os_[0]["job"]["individual_id"], "aircraft": r["aircraft"], "generation": int(r["generation"]),
                     "rank": r.get("rank"), "is_best": bool(r.get("_is_best")), "is_elite": bool(r.get("is_elite")),
                     "eval_seed": row_eval_seed(r, run_cfg), "scenario_ids": [sid for _, sid in row_scen[ri]],
                     "status": [o["result"].get("status") for o in os_],
                     "evaluate_model_version_match": os_[0]["result"].get("model_version_match"),
                     "fd_current_model_version": fd_current_model_version(r["aircraft"], fid),
                     "model_version_current": (None if fd_current_model_version(r["aircraft"], fid) is None or mv is None
                                               else mv == fd_current_model_version(r["aircraft"], fid))})
        pin = _per_fid(((run_cfg.get("pin_model_version") or {}).get(r["aircraft"])), fid)
        if pin:                                       # run.json pin_model_version {ac: {fid: mv}} (ER, phase2 runs)
            cmpd.update({"pinned_model_version": pin, "pinned_model_version_match": mv == pin if mv else None})
        if any(o.get("structure") for o in os_):
            cmpd["structure"] = {"node_status": {o["job"]["scenario_id"]: o.get("node_status") for o in os_},
                                 "node_note": os_[0].get("node_note"),
                                 "tip_check_max_abs_fd_units": max((o["v2_stats"] or {}).get("tip_check_max_abs", 0.0) for o in os_),
                                 "wing_modal_vs_nodal": {o["job"]["scenario_id"]: (o["v2_stats"] or {}).get("wing_modal_vs_nodal")
                                                         for o in os_}}
            if "telemetry_check" in os_[0]["result"]:
                cmpd["structure"]["telemetry_check"] = [o["result"].get("telemetry_check") for o in os_]
        if r.get("screen_cost") is not None:
            cmpd["screen"] = {"screen_cost": r.get("screen_cost"), "screen_fidelity": r.get("screen_fidelity"),
                              "rescored_at_full": r.get("rescored_at_full")}
        comparisons.append(cmpd)
        flagged |= cmpd["verdict"] == "MISMATCH"
        sole_best = bool(r.get("_is_best")) and n_per_group[(r["aircraft"], int(r["generation"]))] == 1
        for o in os_:
            k, sid = o["job"]["scenario_index"], o["job"]["scenario_id"]
            fit = mean if mean is not None else float(o["result"]["cost"])
            doc = build_traj_doc(o, run_cfg, replay_id, sense, fit)
            fn = traj_file_name(r, run_id, sid, tag_individual=not sole_best, tag_scenario=k != 0)
            base, n = fn, 1
            while fn in used_names:  # two ids mapping to the same safe name
                n += 1
                fn = base[:-5] + f"__{n}.json"
            used_names.add(fn)
            with open(os.path.join(traj_dir, fn), "w") as f:
                json.dump(doc, f, separators=(",", ":"))
            entries.append({"generation": int(r["generation"]), "fitness": fit, "cost": fit, "aircraft": r["aircraft"], "file": fn,
                            "run": f"{run_id}-replay", "scenario": sid, "scenario_id": sid, "scenario_index": k,
                            "individual_id": o["job"]["individual_id"], "rank": r.get("rank"),
                            "is_best": bool(r.get("_is_best")), "is_elite": bool(r.get("is_elite")),
                            "logged_cost": cmpd["logged_cost"], "verdict": cmpd["verdict"], "fidelity": fid,
                            "model_version": mv, **({"node_status": o.get("node_status")} if o.get("structure") else {})})
    entries.sort(key=lambda e: (e["aircraft"], e["generation"], not e["is_best"], e["rank"] if e["rank"] is not None else 0,
                                str(e["individual_id"]), e["scenario_index"]))
    index = {"schema": INDEX_SCHEMA, "traj_schema": TRAJ_SCHEMA, "run_id": f"{run_id}-replay", "fitness_sense": sense,
             "replay_of": run_id, "replay_id": replay_id, "id_scheme": "opaque strings (individual_id, scenario_id)",
             "entries": entries}
    with open(os.path.join(traj_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    # channel-by-channel check against the run's own recorded trajectories (ER writes best-of-gen, scenario position 0)
    traj_check = None
    ref_dir = a.compare_traj or os.path.join(run_dir, "trajectories")
    if a.compare_traj != "none" and os.path.isdir(ref_dir):
        from sim_bridge.trajdiff import compare_dirs
        rep = compare_dirs(traj_dir, ref_dir)
        by_file = {e["file"]: e for e in entries}
        cmp_by_id = {(c["aircraft"], c["generation"], c["individual_id"]): c for c in comparisons}
        files = {}
        for key, v in rep.items():
            e = by_file.get(v["a"], {})
            c = cmp_by_id.get((e.get("aircraft"), e.get("generation"), e.get("individual_id")), {})
            same_mv = c.get("model_version_match") in (None, True)
            verdict = "match" if not v["fail"] else ("MISMATCH" if same_mv else "mismatch expected (model_version differs)")
            flagged |= verdict == "MISMATCH"
            files[key] = {
                "replay_file": v["a"], "reference_file": v["b"], "verdict": verdict, "individual_id": e.get("individual_id"),
                "rows_compared": v["rows_compared"], "rows_replay": v["rows_a"], "rows_reference": v["rows_b"],
                "bit_identical_channels": sum(1 for ch in v["channels"].values() if ch["max_abs"] == 0),
                "n_channels": len(v["channels"]), "failed_channels": v["fail"],
                "max_abs": {ch: x["max_abs"] for ch, x in v["channels"].items()}, "meta_diff_keys": sorted(v["meta_diff"]),
                **({"rediscretised_components": v["remapped"],
                    "rediscretised_note": "components whose node layout differs from the reference (e.g. FD nodal wings "
                                          "vs ER's 9-node modal wings) are not compared by channel name; values at "
                                          "coincident span fractions are reported here (informational, not pass/fail)"}
                   if v.get("remapped") else {})}
        traj_check = {"reference_dir": os.path.relpath(os.path.abspath(ref_dir), TEAM_ROOT), "n_files": len(files),
                      "tolerance": "half a unit of each channel's written decimal (ER trajectory.py rounding)",
                      "summary": {vv: sum(1 for f in files.values() if f["verdict"] == vv) for vv in sorted({f["verdict"] for f in files.values()})},
                      "files": files}
    finished = _dt.datetime.now().astimezone()
    try:
        import jsbsim
        jv = jsbsim.__version__
    except ImportError:
        jv = None

    def rel(p):
        return None if p is None else os.path.relpath(os.path.abspath(p), TEAM_ROOT)

    manifest = {
        "schema": MANIFEST_SCHEMA, "replay_id": replay_id, "run_id": run_id, "run_dir": rel(run_dir),
        "paths_relative_to": "FLIGHT_SIM_TEAM_ROOT",
        "interface": iface["kind"], "evaluate": iface["module_spec"], "evaluate_file": rel(iface.get("module_file")),
        "fallback_reason": iface.get("fallback_reason"), "path_notes": iface.get("path_notes"),
        "fitness_sense": sense, "fitness_sense_raw": run_cfg.get("fitness_sense"), "metric": "cost" if sense == "min" else "fitness",
        "fidelity_requested": a.fidelity, "scenario_selection": a.scenario, "sample_hz": a.hz, "tolerance_rel": a.tol,
        "selection": {"gens": sorted(gens) if gens else None, "best_per_gen": a.best_per_gen, "elites": a.elites,
                      "ids": a.ids.split(",") if a.ids else None, "aircraft": a.aircraft},
        "provenance": {"run_git_sha": run_cfg.get("git_sha"), "run_jsbsim_version": run_cfg.get("jsbsim_version"),
                       "run_code_sha": run_cfg.get("code_sha"), "run_model_version": run_cfg.get("model_version"),
                       "replay_jsbsim_version": jv,
                       "replay_model_versions": sorted({str(c["replay_model_version"]) for c in comparisons if c["replay_model_version"]}),
                       "python": platform.python_version()},
        "started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"), "wall_s": wall,
        "flagged": flagged, "n_genomes": len(rows), "n_flights": len(jobs),
        "summary": {v: sum(1 for c in comparisons if c["verdict"] == v) for v in sorted({c["verdict"] for c in comparisons})},
        "model_version_mismatches": sum(1 for c in comparisons if c["model_version_match"] is False),
        "model_versions": {"fd_published_file": rel(paths.FD_MODEL_VERSIONS),
                           "not_current": sorted({f"{c['aircraft']}:{c['replay_model_version']}" for c in comparisons
                                                  if c.get("model_version_current") is False}),
                           "pinned_mismatch": sorted({f"{c['aircraft']}:{c['replay_model_version']}!={c['pinned_model_version']}"
                                                      for c in comparisons if c.get("pinned_model_version_match") is False}),
                           "rule": "logged: row ladder_model_version[fid] > row model_version > screen_model_version > "
                                   "aircraft ladder_model_version[fid] > run.json model_version; current: FD's published "
                                   "string for (aircraft, fidelity); pinned: run.json pin_model_version[ac][fid] (reported, "
                                   "not flagged: a replay after an FD change legitimately differs)"},
        "recorder_timing": timing,
        "genomes": comparisons,
        "trajectory_check": traj_check,
    }
    with open(os.path.join(out_dir, "replay_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if a.html:
        import colab_viewer
        acs = sorted({r["aircraft"] for r in rows})
        prm = {"mode": "compare", "cam": "chase"}
        if len(acs) > 1:
            prm.update({"preset": "last", "layout": "formation", "vref": "norm", "spacing": "40"})
        html = colab_viewer.build_standalone_html(os.path.join(traj_dir, "index.json"), gens="all", hz=min(a.hz, 10.0),
                                                  params=prm, slim=True, title=f"replay {run_id} / {replay_id}")
        with open(os.path.join(out_dir, "viewer.html"), "w") as f:
            f.write(html)
    for c in comparisons:
        rp = c["replayed_cost"] if c["replayed_cost"] is not None else {k: v["replayed"] for k, v in c["per_scenario"].items()}
        lg = c["logged_cost"] if c["replayed_cost"] is not None else {k: v["logged"] for k, v in c["per_scenario"].items()}
        mvn = "" if c["model_version_match"] in (None, True) else "  (model_version differs)"
        print(f"  {c['aircraft']:6s} g{c['generation']:<3d} {str(c['individual_id']):16s} logged {lg} replayed {rp} "
              f"rel {c['max_rel_err']}  {c['verdict']}{mvn}")
    if traj_check:
        print(f"  trajectory check vs {traj_check['reference_dir']}: {traj_check['n_files']} files {traj_check['summary']}")
        for k, f in traj_check["files"].items():
            print(f"    {k:18s} rows {f['rows_compared']}/{f['rows_replay']}/{f['rows_reference']}  {f['bit_identical_channels']}/{f['n_channels']} channels "
                  f"bit-identical  worst {max(f['max_abs'].items(), key=lambda kv: kv[1])}  {f['verdict']}")
            for comp, r in (f.get("rediscretised_components") or {}).items():
                print(f"      {comp}: {r['a_nodes']} nodes vs reference {r['b_nodes']} (not compared by name); at "
                      f"{r['n_pairs']} coincident nodes max |diff| {r['max_abs_at_coincident_nodes']}")
    print(f"{manifest['summary']}  wall {wall:.1f}s  -> {out_dir}" + ("  FLAGGED" if flagged else ""))
    return 2 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
