#!/usr/bin/env python3
"""Evolve altitude-hold controllers for several JSBSim aircraft in one batch.

    cd flight_sim_3d
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
import multiprocessing as mp
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional

import numpy as np

from . import cache as cache_mod
from . import eval as eval_mod
from . import fidelity as fid_mod
from . import ga, genome, runinfo, sim, trajectory

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE_REPO = os.path.dirname(os.path.dirname(PKG_DIR))  # repo root (flight_sim_3d/..), recorded as git sha

DEFAULTS: Dict = {
    "run_id": None,                 # default: <config-stem>-<hash of resolved config>  (same command => resume)
    "runs_dir": "runs",             # relative paths are relative to the evolution/ package dir
    "cache": {"enabled": True, "path": "cache/evals.sqlite"},
    "workers": 0,                   # 0 = all usable cores
    "schedule": "concurrent",       # "concurrent" (all aircraft share the pool) | "sequential" (one aircraft at a time)
    "seed": 1,                      # GA seed (each aircraft gets this unless it sets its own "seed")
    "scenario_seed": None,          # None = seed (as in the original evolve.py)
    "scenarios": 3,
    "ga": {"pop_size": 24, "generations": 15, "elite": 2, "selection_p": 0.2, "crossover": "uniform",
           "blx_alpha": 0.3, "mutation_rate": 0.15, "mutation_sigma": 0.08, "mutation_mode": "gauss"},
    "trajectories": {"generations": "auto", "scenario": 0, "sample_hz": 30},   # auto = [0, (G-1)//2, G-1]
    "metrics": {"band_ft": 20.0, "hold_after_s": 20.0},
    "source_repo": DEFAULT_SOURCE_REPO,
    "profiles": {"baseline": {}},   # name -> sim.Profile overrides; {} = original c172x constants
    "aircraft": [{"name": "c172x", "profile": "baseline"}],
    # structural fidelity (evolution/fidelity.py): rigid | reduced (FD flex v1, 1 bending mode) | full (FD v1, 2 modes)
    "fidelity": "rigid",
    # screen everyone at `screen`, re-score the top_k by screen cost + all elites at `fidelity` (see README)
    "multi_fidelity": {"enabled": False, "screen": "reduced", "top_k": 4},
    "struct_genes": False,          # append FD's STRUCT_SCHEMA genes (stiffness_scale, torsion_bend_ratio, zeta, nonstruct_scale)
    "viz": "off",                   # on = record a trajectory on EVERY evaluation + live best-of-generation file (slow path)
}
AIRCRAFT_KEYS = {"name", "profile", "overrides", "seed"}
# execution-only settings: cannot change results, so they are excluded from the run-id hash and the resume check
EXEC_KEYS = ("workers", "schedule", "cache", "viz")


def identity(cfg: Dict) -> Dict:
    return {k: v for k, v in cfg.items() if k not in EXEC_KEYS and k != "run_id"}


# --------------------------------------------------------------------------- config
def _merge(base: Dict, over: Dict, path: str = "") -> Dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):
            continue  # "_comment" etc.
        if k not in base and path not in ("profiles",):
            raise ValueError(f"unknown config key {path + k!r}")
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k not in ("profiles",):
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
    if cfg["schedule"] not in ("concurrent", "sequential"):
        raise ValueError("schedule must be 'concurrent' or 'sequential'")
    if cfg["fidelity"] not in fid_mod.FIDELITIES:
        raise ValueError(f"fidelity must be one of {fid_mod.FIDELITIES}")
    if cfg["viz"] not in ("on", "off"):
        raise ValueError("viz must be 'on' or 'off'")
    mf = cfg["multi_fidelity"]
    if mf["enabled"]:
        if mf["screen"] not in fid_mod.FIDELITIES or mf["screen"] == cfg["fidelity"]:
            raise ValueError("multi_fidelity.screen must be a fidelity different from `fidelity` (the authoritative one)")
        if not (isinstance(mf["top_k"], int) and 1 <= mf["top_k"] <= cfg["ga"]["pop_size"]):
            raise ValueError("multi_fidelity.top_k must be an int in [1, pop_size]")
    if cfg["struct_genes"] and cfg["fidelity"] == "rigid" and not mf["enabled"]:
        raise ValueError("struct_genes only matter at reduced/full fidelity")
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
        acs.append({"name": a["name"], "profile": pname, "overrides": a.get("overrides", {}),
                    "seed": int(a.get("seed", cfg["seed"])), "resolved_profile": prof.to_dict()})
    cfg["aircraft"] = acs
    G = cfg["ga"]["generations"]
    tg = cfg["trajectories"]["generations"]
    gens = sorted({0, (G - 1) // 2, G - 1}) if tg == "auto" else sorted({int(g) for g in tg if 0 <= int(g) < G} | {G - 1})
    cfg["trajectories"]["generations_resolved"] = gens
    if cfg["run_id"] is None:
        h = hashlib.sha256(json.dumps(identity(cfg), sort_keys=True).encode()).hexdigest()[:8]
        cfg["run_id"] = f"{stem}-{h}"
    return cfg


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
def full_schema(prof: "sim.Profile", struct_genes: bool):
    """(schema, groups): controller genes (6, +2 heading) then, with struct_genes, FD's STRUCT_SCHEMA genes."""
    sch = genome.make_schema(prof.gain_bounds, prof.gene_kinds, prof.heading_hold)
    groups = ["gains"] * len(sch)
    if struct_genes:
        st = fid_mod.struct_schema()
        sch, groups = sch + st, groups + ["struct"] * len(st)
    return sch, groups


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

    def _viz_on(self) -> bool:
        return self.viz in (True, "on")

    def _describe_all(self):
        """model_version per aircraft and fidelity (computed in a worker), then run.json."""
        fids = [self.fidelity] + ([self.mf["screen"]] if self.mf.get("enabled") else [])
        futs = {ac["name"]: self.pool.submit(eval_mod.describe, ac["resolved_profile"], fids) for ac in self.cfg["aircraft"]}
        for ac in self.cfg["aircraft"]:
            d = futs[ac["name"]].result()
            prof = sim.Profile.from_dict(ac["resolved_profile"])
            sch, groups = full_schema(prof, self.cfg.get("struct_genes", False))
            self.info[ac["name"]] = {"schema": sch, "groups": groups, "model_files_sha": d["model_files_sha"],
                                     "model_version": d["model_version"].get(self.fidelity),
                                     "mv": d["model_version"], "error": d.get("error")}
            if self.mf.get("enabled"):
                self.info[ac["name"]]["screen_model_version"] = d["model_version"].get(self.mf["screen"])
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
    KEEP = ("cost", "status", "t_end", "track", "effort", "comfort", "heading_rms", "hdg_drift_deg", "hdg_max_abs_err_deg")

    def _trim_result(self, r: Dict) -> Dict:
        out = {k: r[k] for k in self.KEEP if k in r}
        if r.get("pre") is not None:   # flex: margins + structural terms (rigid rows stay exactly as before)
            pre = r["pre"]
            out["sim_cost"] = r.get("sim_cost")
            out["pre"] = {"fail": pre["fail"], "margins_fidelity": pre["margins_fidelity"],
                          "flutter_margin": pre["margins"].get("flutter_margin"), "div_margin": pre["margins"].get("div_margin"),
                          "terms": pre["terms"]}
            out["struct"] = r.get("struct")
            out["fidelity"], out["model_version"] = r.get("fidelity"), r.get("model_version")
        return out

    def _evaluate(self, st: Dict, pop: np.ndarray, fid: str):
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
            cpu += r.get("wall_s", 0.0)
        self.cache.put_many(new)
        results = []
        for row in keys:
            per = [memo[k] for k in row]
            agg = eval_mod.aggregate(per)
            results.append({"cost": agg["cost"], "per_scenario": [self._trim_result(r) for r in per], "agg": agg,
                            "keys": row})
        return results, {"unique_sims": len(need), "sims_computed": len(miss), "cache_hits_mem": mem_hits,
                         "cache_hits_disk": len(disk), "eval_cpu_s": cpu}

    # ---- one aircraft
    def _rows(self, st, gen, pop_orig, order, sel, scr, full_idx, mf):
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
                   "aircraft": name, "generation": gen, "index": i, "rank": rank_of[i], "is_best": rank_of[i] == 0,
                   "is_elite": rank_of[i] < elite, "carried_elite": gen > 0 and i < elite,
                   "genome": vals, "genome_norm": [float(x) for x in pop_orig[i]],
                   "gains": gains, "struct": struct, "cost": res["cost"], "fitness": res["cost"],
                   "per_scenario_cost": [p["cost"] for p in res["per_scenario"]],
                   "scenario_ids": st["scenario_ids"], "status": agg["status"], "feasible": agg["feasible"],
                   "feasibility_fidelity": agg["feasibility_fidelity"], "terms": agg["terms"],
                   "terms_available": agg["terms_available"], "fidelity": agg["fidelity"],
                   "model_version": agg["model_version"], "session": self.session}
            if "margins" in agg:
                row["margins"], row["margins_fidelity"] = agg["margins"], agg["margins_fidelity"]
            if mf:
                row["rescored_at_full"] = i in full_idx
                row["screen_cost"] = scr[i]["cost"]
                row["screen_fidelity"] = scr[i]["agg"]["fidelity"]
                row["screen_model_version"] = scr[i]["agg"]["model_version"]
                row["screen_per_scenario_cost"] = [p["cost"] for p in scr[i]["per_scenario"]]
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
            r = self.pool.submit(eval_mod.task, st["profile_d"], gains, struct, st["scenarios_d"][s_idx], self.fidelity,
                                 True, self.cfg["trajectories"]["sample_hz"]).result()
        else:
            r = dict(st["memo"][best_key])
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
        mf = self.mf if self.mf.get("enabled") else None
        fid = self.fidelity
        st = {"ac": ac, "profile_d": ac["resolved_profile"], "scenarios_d": [s.to_dict() for s in scs],
              "scenario_ids": [runinfo.scenario_id(name, i) for i in range(len(scs))],
              "schema": info["schema"], "groups": info["groups"], "memo": {}, "live_traj": {},
              "mv": info["mv"], "model_sha": sim.model_files_sha(name, prof.aircraft_root)}
        gcfg = ga.GAConfig(**{k: v for k, v in self.cfg["ga"].items() if k != "generations"})
        out = {"aircraft": name, "profile": ac["profile"], "aircraft_root": prof.aircraft_root,
               "model_files_sha": st["model_sha"], "fidelity": fid, "fidelity_label": fid_mod.label(fid),
               "model_version": info["model_version"]}
        if mf:
            out["multi_fidelity"] = {**mf, "screen_model_version": info.get("screen_model_version")}
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
        if ck is None:
            rng = np.random.default_rng(ac["seed"])
            pop = ga.generation_zero(rng, self.cfg["ga"]["pop_size"], len(st["schema"]))  # 6, 8 with heading hold, +4 struct
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
            scr, full_idx = None, set()
            if not mf:
                sel, es = self._evaluate(st, pop, fid)
                costs = np.array([r["cost"] for r in sel])
                order = ga.rank_order(costs)
            else:
                ts = time.perf_counter()
                scr, es_s = self._evaluate(st, pop, mf["screen"])
                t_scr = time.perf_counter() - ts
                order_s = ga.rank_order(np.array([r["cost"] for r in scr]))
                full_idx = {int(i) for i in order_s[:mf["top_k"]]} | (set(range(min(elite, len(pop)))) if gen > 0 else set())
                fl = sorted(full_idx)
                tf = time.perf_counter()
                fres, es_f = self._evaluate(st, pop[fl], fid)
                t_full = time.perf_counter() - tf
                fmap = dict(zip(fl, fres))
                sel = [fmap.get(i, scr[i]) for i in range(len(pop))]
                order = np.array(sorted(range(len(pop)), key=lambda i: (0, fmap[i]["cost"], i) if i in fmap
                                        else (1, scr[i]["cost"], i)), dtype=int)
                costs = np.array([r["cost"] for r in sel])
                es = {k: es_s[k] + es_f[k] for k in es_s}
                pairs = [[scr[i]["cost"], fmap[i]["cost"]] for i in fl]
                extra = {"screen_fidelity": mf["screen"], "top_k": mf["top_k"], "n_rescored": len(fl), "rescored_idx": fl,
                         "spearman_screen_vs_full": spearman([p[0] for p in pairs], [p[1] for p in pairs]),
                         "rescored_pairs": pairs, "screen": es_s, "full": es_f,
                         "eval_wall_s_screen": t_scr, "eval_wall_s_full": t_full,
                         "best_screen_cost": float(min(r["cost"] for r in scr))}
            wall = time.perf_counter() - t0
            for k in tot:
                tot[k] += es[k]
            rows = self._rows(st, gen, pop, order, sel, scr, full_idx, mf)
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
            if fid != "rigid" or mf:
                rec.update({"fidelity": fid, "model_version": info["model_version"],
                            "best_fidelity": sel[0]["agg"]["fidelity"], "best_feasible": sel[0]["agg"]["feasible"],
                            "best_terms": sel[0]["agg"]["terms"], **extra})
            best = {"generation": gen, "genome": [float(x) for x in pop[0]], "fitness": float(costs[0]),
                    "per_scenario": sel[0]["per_scenario"]}
            if fid != "rigid" or mf:
                best["fidelity"] = sel[0]["agg"]["fidelity"]
            if gen < G - 1:
                nxt = ga.next_generation(rng, pop, gcfg)
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
            if mf:
                sp = extra["spearman_screen_vs_full"]
                msg += f"  rescored {extra['n_rescored']} rho {'n/a' if sp is None else f'{sp:+.3f}'}"
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
            gains, struct = eval_mod.split_values(genome.decode(norm, st["schema"]), gmap)
            return self.pool.submit(eval_mod.task, st["profile_d"], gains, struct, st["scenarios_d"][s], fidl, True,
                                    tcfg["sample_hz"])
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
            for k in ("hdg_drift_deg", "hdg_max_abs_err_deg"):
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
        resumed = self._prepare_run_dir()
        self.log(f"run {self.cfg['run_id']}  dir {self.run_dir}\n  {len(self.cfg['aircraft'])} aircraft, "
                 f"{self.workers} workers, schedule={self.schedule}, cache={'on' if self.cache.enabled else 'off'}, "
                 f"jsbsim {self.jsbsim_version}, code {self.code_sha}, src {self.git['sha'][:10]}\n  "
                 f"fidelity {fid_mod.label(self.fidelity)}, viz {'on' if self._viz_on() else 'off'}"
                 + (f", multi-fidelity screen={self.mf['screen']} top_k={self.mf['top_k']}" if self.mf.get('enabled') else "")
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
    ap.add_argument("--fidelity", choices=list(fid_mod.FIDELITIES), help="rigid (default) | reduced | full (= full(v1) "
                                                                        "until FD v2). Part of the run identity")
    ap.add_argument("--multi-fidelity", action="store_true", help="screen everyone at --screen, re-score top-k + elites "
                                                                 "at --fidelity")
    ap.add_argument("--screen", choices=list(fid_mod.FIDELITIES), help="screen fidelity for --multi-fidelity (default reduced)")
    ap.add_argument("--top-k", type=int, help="individuals re-scored at --fidelity per generation (default 4)")
    ap.add_argument("--struct-genes", action="store_true", help="add FD's structural genes to the genome")
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
                mfc["screen"] = a.screen
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
        cfg = resolve_config(user, os.path.splitext(os.path.basename(a.config))[0])
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
