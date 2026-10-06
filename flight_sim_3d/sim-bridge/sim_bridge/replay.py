"""Re-fly selected genomes of an Evolution Runner run with trajectory logging (ER's fast mode logs nothing).

    python replay.py --run phase1-s1 --gens 0,9,19 [--aircraft c172x,T38] [--scenario all|<id>]
                     [--fidelity rigid|reduced|full] [--hz 30] [--html] [--out DIR]
    python replay.py --run phase1-s1 --best-per-gen | --elites | --ids 'f16:g19:r0,737:g9:r0'

Interface resolution (``--interface auto``):
  * ``er``: <run>/run.json + <run>/genomes.jsonl + ``evolution.eval.evaluate`` (the agreed interface; ER ships it
    with fast mode). ``--eval-module`` overrides the module (dotted name or .py path; used by the fixture test).
  * ``adapter``: legacy runs (config.json + checkpoints/), via sim_bridge.er_adapter on top of ER's evolution.sim.
Output: sim-bridge/data/replays/<run_id>/<replay_id>/ by default (ER's tree stays read-only; ``--out`` overrides,
e.g. <runs-root>/<run_id>/replays/<replay_id>/ once ER wants it there):
  trajectories/traj_*.json (ga-flightsim-traj/1), trajectories/index.json, replay_manifest.json, viewer.html (--html)

Cost is "lower is better" unless run.json says ``fitness_sense: "max"``. A replay at the row's own fidelity with the
same model_version must reproduce the logged cost exactly (relative error <= 1e-6, ``--tol``); otherwise the replay
is flagged and the exit code is 2. A different model_version (e.g. ``full`` moving from v1 flex to FD's v2) makes a
mismatch "expected" (both versions recorded). A different fidelity than the row's is reported as "not comparable".
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as _dt
import importlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from typing import Dict, List, Optional

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SIM_BRIDGE = os.path.dirname(HERE)
TEAM_ROOT = os.environ.get("FLIGHT_SIM_TEAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_RUNS_ROOT = os.path.join(TEAM_ROOT, "evolution", "runs")
for p in (TEAM_ROOT, SIM_BRIDGE):
    if p not in sys.path:
        sys.path.insert(0, p)

from sim_bridge import er_adapter  # noqa: E402
from sim_bridge.recorder import TrajRecorder  # noqa: E402

TRAJ_SCHEMA = "ga-flightsim-traj/1"
INDEX_SCHEMA = "ga-flightsim-traj-index/1"
MANIFEST_SCHEMA = "sim-bridge-replay-manifest/1"
FIDELITIES = ("rigid", "reduced", "full")
FT = 0.3048


# ------------------------------------------------------------------ interface loading
def sense_of(run_cfg: Dict) -> str:
    s = str(run_cfg.get("fitness_sense", "min")).lower()
    return "max" if s.startswith("max") else "min"


def row_cost(row: Dict) -> Optional[float]:
    v = row.get("cost", row.get("fitness"))
    return None if v is None else float(v)


def row_per_scenario(row: Dict) -> Optional[List[float]]:
    v = row.get("per_scenario_cost", row.get("per_scenario_fitness"))
    return None if v is None else [float(x) for x in v]


def _load_module(spec: str):
    if spec.endswith(".py") or os.path.sep in spec:
        name = "replay_eval_" + os.path.splitext(os.path.basename(spec))[0]
        sp = importlib.util.spec_from_file_location(name, spec)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        return mod
    return importlib.import_module(spec)


def resolve_interface(run_dir: str, mode: str = "auto", eval_module: str = "evolution.eval"):
    """-> dict(kind, run_cfg, rows, evaluate, module_spec)."""
    has_files = os.path.exists(os.path.join(run_dir, "run.json")) and os.path.exists(os.path.join(run_dir, "genomes.jsonl"))
    err = None
    if mode in ("auto", "er") and has_files:
        try:
            mod = _load_module(eval_module)
            with open(os.path.join(run_dir, "run.json")) as f:
                run_cfg = json.load(f)
            with open(os.path.join(run_dir, "genomes.jsonl")) as f:
                rows = [json.loads(l) for l in f if l.strip()]
            return {"kind": "er", "run_cfg": run_cfg, "rows": rows, "evaluate": mod.evaluate, "module_spec": eval_module,
                    "module": mod}
        except (ImportError, AttributeError, FileNotFoundError) as e:
            err = f"{type(e).__name__}: {e}"
            if mode == "er":
                raise SystemExit(f"--interface er: cannot load {eval_module}: {err}")
    elif mode == "er":
        raise SystemExit(f"--interface er: {run_dir} has no run.json + genomes.jsonl")
    if not os.path.exists(os.path.join(run_dir, "config.json")):
        raise SystemExit(f"{run_dir}: neither run.json+genomes.jsonl (+{eval_module}) nor a legacy config.json")
    run_cfg = er_adapter.load_run_cfg(run_dir)
    return {"kind": "adapter", "run_cfg": run_cfg, "rows": er_adapter.genome_rows(run_dir, run_cfg),
            "evaluate": er_adapter.evaluate, "module_spec": "sim_bridge.er_adapter", "fallback_reason": err}


def select_rows(rows: List[Dict], gens=None, best_per_gen=False, elites=False, ids=None, aircraft=None) -> List[Dict]:
    out = rows
    if aircraft:
        out = [r for r in out if r["aircraft"] in aircraft]
    if ids:
        want = set(ids)
        out = [r for r in out if str(r.get("individual_id")) in want]
        missing = want - {str(r.get("individual_id")) for r in out}
        if missing:
            raise SystemExit(f"unknown individual ids: {sorted(missing)}")
        return out
    if gens is not None:
        out = [r for r in out if int(r["generation"]) in gens]
    if elites:
        out = [r for r in out if r.get("is_elite") or r.get("is_best")]
    else:  # --gens or --best-per-gen: the best of each selected generation
        out = [r for r in out if r.get("is_best")]
    return out


# ------------------------------------------------------------------ evaluation (worker)
def _scenario_obj(kind: str, mod, aircraft: str, scenario: Dict, run_cfg: Dict):
    """Scenario object the recorder can query for target / target_cmd / target_rate."""
    for name in ("scenario_object", "make_scenario"):
        f = getattr(mod, name, None) if mod is not None else None
        if callable(f):
            return f(aircraft, scenario, run_cfg)
    return er_adapter.scenario_object(aircraft, scenario, run_cfg)[0]


def _call_eval(evaluate, genome, aircraft, scenario, run_cfg, recorder):
    return evaluate(genome, aircraft, scenario, run_cfg, recorder=recorder)


def run_job(job: Dict) -> Dict:
    """One (genome, scenario) re-flight with logging. Runs in a worker process."""
    iface = resolve_interface(job["run_dir"], job["interface"], job["eval_module"])
    run_cfg = dict(iface["run_cfg"])
    run_cfg["fidelity"] = job["fidelity"]
    scen = next(s for s in run_cfg["scenarios"] if str(s["id"]) == str(job["scenario_id"]))
    sc_obj = _scenario_obj(iface["kind"], iface.get("module"), job["aircraft"], scen, run_cfg)
    rec = TrajRecorder(sc_obj, job["hz"], float(run_cfg.get("sim_dt_s", 1 / 120)), timing=job["recorder_timing_mode"])
    t0 = time.perf_counter()
    err = None
    try:
        res = _call_eval(iface["evaluate"], job["genome"], job["aircraft"], scen, run_cfg, rec)
    except NotImplementedError as e:
        res, err = None, str(e)
    rec.finish()
    wall = time.perf_counter() - t0
    if res is None:
        return {"job": job, "error": err}
    return {"job": job, "result": {k: v for k, v in res.items() if isinstance(v, (int, float, str, type(None), dict, list))},
            "channels": rec.channels, "rows": rec.rounded_rows(), "origin": rec.origin, "structure": rec.structure,
            "recorder_calls": rec.n_calls, "wall_s": wall,
            "scenario": sc_obj.to_dict() if hasattr(sc_obj, "to_dict") else dict(scen)}


# ------------------------------------------------------------------ output docs
def build_traj_doc(out: Dict, run_cfg: Dict, replay_id: str, sense: str, fitness_val: float) -> Dict:
    job, res, sc = out["job"], out["result"], out["scenario"]
    o = out["origin"]
    events = [{"t": 0.0, "type": "start", "detail": f"trimmed level flight {sc.get('h0_ft', 0):.0f} ft, {sc.get('speed_kts', 0):.0f} KCAS"
               + (f", throttle {out['rows'][0][15]:.3f}" if out["rows"] else "")}]
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
    ac = next((a for a in run_cfg["aircraft"] if a["name"] == job["aircraft"]), {})
    doc = {
        "schema": TRAJ_SCHEMA, "run_id": run_cfg["run_id"], "aircraft": job["aircraft"],
        "jsbsim_version": run_cfg.get("jsbsim_version"), "git_sha": run_cfg.get("git_sha"), "seed": ac.get("seed", run_cfg.get("seed")),
        "generation": int(job["generation"]), "individual_id": job["individual_id"],
        "fitness": float(fitness_val), "cost": float(fitness_val), "fitness_sense": sense,
        "scenario_index": job["scenario_id"], "scenario_cost": float(res["cost"]), "status": res.get("status"),
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
        "wind": {"north_mps": sc.get("wind_north_fps", 0) * FT, "east_mps": sc.get("wind_east_fps", 0) * FT,
                 "gust_sigma_mps": sc.get("gust_sigma_fps", 0) * FT},
        "events": events,
        "replay": {"of_run": run_cfg["run_id"], "replay_id": replay_id, "fidelity": job["fidelity"],
                   "model_version": res.get("model_version"), "logged_model_version": job.get("model_version"),
                   "logged_fidelity": job.get("row_fidelity"), "logged_scenario_cost": job.get("logged_scenario_cost"),
                   "replayed_scenario_cost": float(res["cost"]),
                   "recorder_timing": job.get("recorder_timing")},
        "channels": out["channels"], "data": out["rows"],
    }
    if out.get("structure"):
        doc["structure"] = out["structure"]
    return doc


def rel_err(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(b), 1e-300)


def compare(row: Dict, replayed_mean: Optional[float], replayed_per: Dict[int, float], fid: str, mv_replay: str, tol: float) -> Dict:
    logged = row_cost(row)
    logged_per = row_per_scenario(row)
    same_fid = fid == row.get("fidelity", "rigid")
    same_mv = mv_replay == row.get("model_version")
    per = {}
    for sid, c in replayed_per.items():
        lp = logged_per[int(sid)] if logged_per is not None and int(sid) < len(logged_per) else None
        per[str(sid)] = {"logged": lp, "replayed": c, "rel_err": rel_err(c, lp)}
    errs = [v["rel_err"] for v in per.values() if v["rel_err"] is not None]
    re_mean = rel_err(replayed_mean, logged)
    if re_mean is not None:
        errs.append(re_mean)
    worst = max(errs) if errs else None
    if worst is None:
        verdict = "no logged cost"
    elif not same_fid:
        verdict = "not comparable (fidelity differs)"
    elif worst <= tol:
        verdict = "match"
    elif same_mv:
        verdict = "MISMATCH"
    else:
        verdict = "mismatch expected (model_version differs)"
    return {"logged_cost": logged, "replayed_cost": replayed_mean, "rel_err": re_mean, "per_scenario": per,
            "max_rel_err": worst, "logged_fidelity": row.get("fidelity", "rigid"), "replay_fidelity": fid,
            "logged_model_version": row.get("model_version"), "replay_model_version": mv_replay, "verdict": verdict}


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run id (directory under --runs-root) or a run directory path")
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--gens", help="comma list of generations (best of each; with --elites all elites)")
    sel.add_argument("--best-per-gen", action="store_true", help="the is_best row of every generation")
    sel.add_argument("--ids", help="comma list of individual_id")
    ap.add_argument("--elites", action="store_true", help="elite rows (is_elite) instead of only the best; combine with --gens")
    ap.add_argument("--aircraft", help="comma list (default: all)")
    ap.add_argument("--scenario", default="all", help="'all' or a scenario id (default all = full cost check)")
    ap.add_argument("--fidelity", choices=FIDELITIES, help="default: each row's own fidelity")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--html", action="store_true", help="also write a standalone viewer.html (colab_viewer)")
    ap.add_argument("--out", help="output dir (default sim-bridge/data/replays/<run_id>/<replay_id>)")
    ap.add_argument("--replay-id")
    ap.add_argument("--interface", choices=("auto", "er", "adapter"), default="auto")
    ap.add_argument("--eval-module", default="evolution.eval")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--tol", type=float, default=1e-6, help="relative cost tolerance for an exact replay")
    ap.add_argument("--compare-traj", help="reference trajectories dir for the channel check (default <run>/trajectories; 'none' to skip)")
    ap.add_argument("--recorder-timing", choices=("auto", "pre", "post"), default="auto",
                    help="when evaluate calls the recorder: adapter=pre-step; evolution.eval=post-step (agreed) unless "
                         "the module sets RECORDER_TIMING")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    run_dir = a.run if os.path.isdir(a.run) and os.path.sep in a.run else os.path.join(a.runs_root, a.run)
    iface = resolve_interface(run_dir, a.interface, a.eval_module)
    run_cfg, sense = iface["run_cfg"], sense_of(iface["run_cfg"])
    run_id = run_cfg["run_id"]
    gens = {int(g) for g in a.gens.split(",")} if a.gens else None
    rows = select_rows(iface["rows"], gens=gens, best_per_gen=a.best_per_gen, elites=a.elites,
                       ids=a.ids.split(",") if a.ids else None, aircraft=a.aircraft.split(",") if a.aircraft else None)
    if not rows:
        raise SystemExit("no genomes selected")
    started = _dt.datetime.now().astimezone()
    sel_tag = (f"g{a.gens.replace(',', '.')}" if a.gens else "best" if a.best_per_gen else "ids") + ("-elites" if a.elites else "")
    replay_id = a.replay_id or f"{started:%Y%m%dT%H%M%S}-{a.fidelity or 'own'}-{sel_tag}"
    out_dir = a.out or os.path.join(SIM_BRIDGE, "data", "replays", run_id, replay_id)
    if os.path.realpath(out_dir).startswith(os.path.realpath(os.path.join(TEAM_ROOT, "evolution")) + os.sep) and not a.out:
        raise SystemExit("refusing to write under ER's tree without an explicit --out")
    traj_dir = os.path.join(out_dir, "trajectories")
    scen_ids = [s["id"] for s in run_cfg["scenarios"]] if a.scenario == "all" else [a.scenario]
    timing = a.recorder_timing
    if timing == "auto":
        timing = "pre" if iface["kind"] == "adapter" else getattr(iface.get("module"), "RECORDER_TIMING", "post")
    jobs = []
    for r in rows:
        fid = a.fidelity or r.get("fidelity", "rigid")
        lp = row_per_scenario(r)
        for sid in scen_ids:
            jobs.append({"run_dir": run_dir, "interface": iface["kind"], "eval_module": a.eval_module, "hz": a.hz,
                         "fidelity": fid, "row_fidelity": r.get("fidelity", "rigid"), "model_version": r.get("model_version"),
                         "aircraft": r["aircraft"], "generation": r["generation"], "individual_id": r.get("individual_id"),
                         "genome": r["genome"], "scenario_id": sid,
                         "logged_scenario_cost": lp[int(sid)] if lp is not None and int(sid) < len(lp) else None,
                         "recorder_timing_mode": timing,
                         "recorder_timing": {"pre": "pre-step (state at t, commands for [t,t+dt)) + final state",
                                             "post": "post-step (agreed evolution.eval contract); row controls filled "
                                                     "from the next call = commands for [t,t+dt)"}[timing]})
    print(f"replay {run_id} via {iface['kind']} ({iface['module_spec']}): {len(rows)} genomes x {len(scen_ids)} scenarios "
          f"= {len(jobs)} flights -> {out_dir}", flush=True)
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
            print(f"ERROR {o['job']['individual_id']} sc{o['job']['scenario_id']}: {o['error']}", file=sys.stderr)
        return 3
    os.makedirs(traj_dir, exist_ok=True)
    # group per genome
    by_row: Dict[str, List[Dict]] = {}
    for o in outs:
        by_row.setdefault(o["job"]["individual_id"], []).append(o)
    entries, comparisons, flagged = [], [], False
    for r in rows:
        os_ = sorted(by_row[r.get("individual_id")], key=lambda o: str(o["job"]["scenario_id"]))
        per = {o["job"]["scenario_id"]: float(o["result"]["cost"]) for o in os_}
        import numpy as np  # ER aggregates with np.mean in scenario order
        mean = float(np.mean([per[s] for s in scen_ids])) if a.scenario == "all" else None
        mv = os_[0]["result"].get("model_version")
        cmpd = compare(r, mean, per, os_[0]["job"]["fidelity"], mv, a.tol)
        cmpd.update({"individual_id": r.get("individual_id"), "aircraft": r["aircraft"], "generation": r["generation"],
                     "status": [o["result"].get("status") for o in os_]})
        comparisons.append(cmpd)
        flagged |= cmpd["verdict"] == "MISMATCH"
        for o in os_:
            sid = o["job"]["scenario_id"]
            single = len(scen_ids) == 1 or str(sid) == str(scen_ids[0])
            tag = "" if (r.get("is_best") and single and len([x for x in rows if x["aircraft"] == r["aircraft"] and x["generation"] == r["generation"]]) == 1) \
                else f"_{str(r.get('individual_id')).split(':')[-1]}"
            if not single:
                tag += f"_sc{sid}"
            fit = mean if mean is not None else float(o["result"]["cost"])
            doc = build_traj_doc(o, run_cfg, replay_id, sense, fit)
            fn = f"traj_{r['aircraft']}_{run_id}_g{int(r['generation'])}{tag}.json"
            with open(os.path.join(traj_dir, fn), "w") as f:
                json.dump(doc, f, separators=(",", ":"))
            entries.append({"generation": int(r["generation"]), "fitness": fit, "cost": fit, "aircraft": r["aircraft"], "file": fn,
                            "run": f"{run_id}-replay", "scenario": sid, "individual_id": r.get("individual_id"),
                            "logged_cost": row_cost(r), "verdict": cmpd["verdict"]})
    entries.sort(key=lambda e: (e["aircraft"], e["generation"], str(e["scenario"]), str(e["individual_id"])))
    index = {"schema": INDEX_SCHEMA, "traj_schema": TRAJ_SCHEMA, "run_id": f"{run_id}-replay", "fitness_sense": sense,
             "replay_of": run_id, "replay_id": replay_id, "entries": entries}
    with open(os.path.join(traj_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    # channel-by-channel check against the run's own recorded trajectories (ER writes best-of-gen, scenario 0)
    traj_check = None
    ref_dir = a.compare_traj or os.path.join(run_dir, "trajectories")
    if a.compare_traj != "none" and os.path.isdir(ref_dir):
        from sim_bridge.trajdiff import compare_dirs
        rep = compare_dirs(traj_dir, ref_dir)
        mv_logged = {c["individual_id"]: c.get("logged_model_version") for c in comparisons}
        mv_rep = {c["individual_id"]: c.get("replay_model_version") for c in comparisons}
        files = {}
        for key, v in rep.items():
            d = json.load(open(os.path.join(traj_dir, v["a"])))
            iid = d.get("individual_id")
            same_mv = mv_logged.get(iid) in (None, mv_rep.get(iid))
            verdict = "match" if not v["fail"] else ("MISMATCH" if same_mv else "mismatch expected (model_version differs)")
            flagged |= verdict == "MISMATCH"
            files[key] = {"replay_file": v["a"], "reference_file": v["b"], "verdict": verdict,
                          "rows_compared": v["rows_compared"], "rows_replay": v["rows_a"], "rows_reference": v["rows_b"],
                          "bit_identical_channels": sum(1 for c in v["channels"].values() if c["max_abs"] == 0),
                          "n_channels": len(v["channels"]), "failed_channels": v["fail"],
                          "max_abs": {c: x["max_abs"] for c, x in v["channels"].items()},
                          "meta_diff_keys": sorted(v["meta_diff"])}
        traj_check = {"reference_dir": os.path.abspath(ref_dir), "n_files": len(files),
                      "tolerance": "half a unit of each channel's written decimal (ER trajectory.py rounding)",
                      "summary": {vv: sum(1 for f in files.values() if f["verdict"] == vv) for vv in sorted({f["verdict"] for f in files.values()})},
                      "files": files}
    finished = _dt.datetime.now().astimezone()
    try:
        import jsbsim
        jv = jsbsim.__version__
    except ImportError:
        jv = None
    manifest = {
        "schema": MANIFEST_SCHEMA, "replay_id": replay_id, "run_id": run_id, "run_dir": os.path.abspath(run_dir),
        "interface": iface["kind"], "evaluate": iface["module_spec"], "fallback_reason": iface.get("fallback_reason"),
        "fitness_sense": sense, "metric": "cost" if sense == "min" else "fitness",
        "fidelity_requested": a.fidelity, "scenarios": scen_ids, "sample_hz": a.hz, "tolerance_rel": a.tol,
        "selection": {"gens": sorted(gens) if gens else None, "best_per_gen": a.best_per_gen, "elites": a.elites,
                      "ids": a.ids, "aircraft": a.aircraft},
        "provenance": {"run_git_sha": run_cfg.get("git_sha"), "run_jsbsim_version": run_cfg.get("jsbsim_version"),
                       "run_code_sha": run_cfg.get("code_sha"), "replay_jsbsim_version": jv,
                       "replay_model_versions": sorted({c["replay_model_version"] for c in comparisons if c["replay_model_version"]}),
                       "python": platform.python_version(), "host": platform.node(), "argv": sys.argv},
        "started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"), "wall_s": wall,
        "flagged": flagged, "n_genomes": len(rows), "n_flights": len(jobs),
        "summary": {v: sum(1 for c in comparisons if c["verdict"] == v) for v in sorted({c["verdict"] for c in comparisons})},
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
        print(f"  {c['aircraft']:6s} g{c['generation']:<3d} {str(c['individual_id']):18s} logged {lg} replayed {rp} "
              f"rel {c['max_rel_err']}  {c['verdict']}")
    if traj_check:
        print(f"  trajectory check vs {traj_check['reference_dir']}: {traj_check['n_files']} files {traj_check['summary']}")
        for k, f in traj_check["files"].items():
            print(f"    {k:14s} rows {f['rows_compared']}/{f['rows_replay']}/{f['rows_reference']}  {f['bit_identical_channels']}/{f['n_channels']} channels "
                  f"bit-identical  worst {max(f['max_abs'].items(), key=lambda kv: kv[1])}  {f['verdict']}")
    print(f"{manifest['summary']}  wall {wall:.1f}s  -> {out_dir}" + ("  FLAGGED" if flagged else ""))
    return 2 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
