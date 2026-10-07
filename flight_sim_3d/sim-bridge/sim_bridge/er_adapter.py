"""Fallback only: expose Evolution Runner's *legacy* runs (no run.json) through the replay interface.

Runs that have ER's real run.json / genomes.jsonl are replayed through evolution.eval (sim_bridge.replay
--interface er); this module is used only for older runs such as bench_jets-j1 (config.json + checkpoints/).
It mirrors ER's real format: per-aircraft string scenario ids "<aircraft>:s<k>" and opaque individual ids.

Interface (ER, 2026-10-06):
  <run>/run.json        {run_id, schema, git_sha, jsbsim_version, seed, sim_dt_s, fitness_sense: "min",
                         aircraft[], scenarios[{id, ...}], fitness_cfg}
  <run>/genomes.jsonl   {generation, individual_id, aircraft, scenario_ids, cost, per_scenario_cost?, genome{name:value},
                         eval_seed, fidelity, model_version, is_best, is_elite, screen_cost?, screen_fidelity?}
  evolution.eval.evaluate(genome, aircraft, scenario, run_cfg, recorder=None)
      -> calls recorder(t, fdm) after each step (recorder(t, fdm, flex_state) when flex is active), read-only.

Existing runs (phase1-s1..s3, bench_jets-j1, ...) only have config.json (resolved profiles, scenario seed/count,
provenance code_sha), checkpoints/<ac>.json (best_per_gen: normalised genome + mean cost + per-scenario costs, and
the final ranked population) and trajectories/. This module rebuilds run.json / genomes.jsonl equivalents from those
and an ``evaluate`` with the same signature on top of ER's own ``evolution.sim.simulate`` (imported read-only, never
edited). The recorder hook is provided by wrapping the JSBSim FDM in a read-only proxy (``sim._new_fdm`` is swapped in
this process only, like sim-bridge/trajlog.py does for the prototype).

Recorder timing: if ER's evolution.sim.simulate accepts ``recorder=`` (current ER) it is passed straight through
(post-step + t=0 call, ER's own contract; ``recorder_timing() == "post"``). Older ER code: ``recorder(t, fdm)`` is
called once per 120 Hz step *after the controller has set the fcs/*-cmd-norm commands and before fdm.run()* through a
read-only FDM proxy ("pre"), plus once at the end (``recorder.final``) for the last state.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, List, Optional

sys.dont_write_bytecode = True  # never drop __pycache__ into ER's tree
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_bridge import paths  # noqa: E402

paths.ensure_import_paths()
TEAM_ROOT = paths.TEAM_ROOT

INTERFACE_SCHEMA = "ga-flightsim-run/1"
ADAPTER_FIDELITIES = ("rigid",)


def _er():
    from evolution import cache, genome, sim  # noqa: WPS433  (read-only import)
    return sim, genome, cache


def _native_recorder() -> bool:
    import inspect
    sim, _, _ = _er()
    return "recorder" in inspect.signature(sim.simulate).parameters


def recorder_timing() -> str:
    return "post" if _native_recorder() else "pre"


def model_version_current(fidelity: str = "rigid") -> str:
    _, _, cache = _er()
    return f"evolution.sim@{cache.code_sha()}"


def load_run_cfg(run_dir: str) -> Dict:
    """run.json equivalent for a legacy run directory."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    res, prov = cfg["resolved"], cfg.get("provenance", {})
    sim, _, _ = _er()
    n_sc = int(res["scenarios"])
    scen, ac_sids = [], {}
    for a in res["aircraft"]:  # per aircraft, as ER's run.json: id "<aircraft>:s<k>" + the GA's own draws
        prof = sim.Profile.from_dict(a["resolved_profile"])
        ac_sids[a["name"]] = []
        for i, s in enumerate(sim.make_scenarios(n_sc, int(res["scenario_seed"]), prof)):
            sid = f"{a['name']}:s{i}"
            ac_sids[a["name"]].append(sid)
            scen.append({"id": sid, "aircraft": a["name"], "index": i, **s.to_dict()})
    return {
        "run_id": res["run_id"], "schema": INTERFACE_SCHEMA + "+legacy-adapter",
        "git_sha": prov.get("git_sha"), "jsbsim_version": prov.get("jsbsim_version"), "code_sha": prov.get("code_sha"),
        "seed": res.get("seed"), "eval_seed": int(res["scenario_seed"]), "sim_dt_s": sim.DT,
        "fitness_sense": "min",
        "aircraft": [{"name": a["name"], "profile": a["profile"], "seed": a.get("seed"),
                      "resolved_profile": a["resolved_profile"], "scenario_ids": ac_sids[a["name"]]}
                     for a in res["aircraft"]],
        "scenarios": scen,
        "fitness_cfg": {"aggregate": "mean over scenarios", "per_profile": "see aircraft[].resolved_profile (alt_err_scale_ft, "
                        "itae_*, w_effort, w_comfort, fail_base)"},
        "ga": res.get("ga"), "trajectories": res.get("trajectories"),
        "_source": {"run_dir": os.path.basename(os.path.abspath(run_dir)), "from": ["config.json", "checkpoints/*.json"]},
    }


def genome_rows(run_dir: str, run_cfg: Optional[Dict] = None) -> List[Dict]:
    """genomes.jsonl equivalent: best of every generation (cost known) + the final ranked population of the last
    generation (genomes known, costs only for the best; elites = the first ``ga.elite``)."""
    run_cfg = run_cfg or load_run_cfg(run_dir)
    sim, genome, _ = _er()
    mv = f"evolution.sim@{run_cfg.get('code_sha')}"
    elite_n = int((run_cfg.get("ga") or {}).get("elite", 1))
    rows = []
    for ac in run_cfg["aircraft"]:
        p = os.path.join(run_dir, "checkpoints", f"{ac['name']}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            ck = json.load(f)
        prof = sim.Profile.from_dict(ac["resolved_profile"])
        schema = genome.make_schema(prof.gain_bounds, prof.gene_kinds)
        last_gen = max(b["generation"] for b in ck["best_per_gen"]) if ck["best_per_gen"] else -1
        for b in ck["best_per_gen"]:
            rows.append({
                "generation": b["generation"], "individual_id": f"{ac['name']}:g{b['generation']}:r0",
                "aircraft": ac["name"], "scenario_ids": list(ac["scenario_ids"]),
                "cost": b["fitness"], "per_scenario_cost": [s["cost"] for s in b["per_scenario"]],
                "per_scenario_status": [s["status"] for s in b["per_scenario"]],
                "genome": genome.decode(b["genome"], schema), "genome_norm": b["genome"],
                "eval_seed": run_cfg["eval_seed"], "fidelity": "rigid", "model_version": mv,
                "is_best": True, "is_elite": True, "rank": 0,
            })
        if ck.get("done") and ck.get("pop"):
            for r, g in enumerate(ck["pop"][1:], start=1):  # rank 0 == best_per_gen[last]
                rows.append({
                    "generation": last_gen, "individual_id": f"{ac['name']}:g{last_gen}:r{r}", "aircraft": ac["name"],
                    "scenario_ids": list(ac["scenario_ids"]), "cost": None, "per_scenario_cost": None,
                    "genome": genome.decode(g, schema), "genome_norm": g, "eval_seed": run_cfg["eval_seed"],
                    "fidelity": "rigid", "model_version": mv, "is_best": False, "is_elite": r < elite_n, "rank": r,
                })
    rows.sort(key=lambda r: (r["aircraft"], r["generation"], r["rank"]))
    return rows


def _scenario_entry(aircraft: str, scenario, run_cfg: Dict) -> Dict:
    if isinstance(scenario, dict):
        return scenario
    for s in run_cfg["scenarios"]:
        if str(s["id"]) == str(scenario) and s.get("aircraft", aircraft) == aircraft:
            return s
    raise KeyError(f"scenario {scenario!r} not in run {run_cfg.get('run_id')}")


def scenario_object(aircraft: str, scenario, run_cfg: Dict):
    """ER Scenario for (aircraft, scenario entry or id): the GA's own draws (make_scenarios(n, eval_seed, profile))
    at the entry's position ``index`` (the id itself is opaque)."""
    sim, _, _ = _er()
    ac = next(a for a in run_cfg["aircraft"] if a["name"] == aircraft)
    prof = sim.Profile.from_dict(ac["resolved_profile"])
    ent = _scenario_entry(aircraft, scenario, run_cfg)
    n = len(ac.get("scenario_ids") or [s for s in run_cfg["scenarios"] if s.get("aircraft", aircraft) == aircraft])
    scs = sim.make_scenarios(n, int(run_cfg["eval_seed"]), prof)
    return scs[ent["index"]], prof


class _RecordingFDM:
    """Read-only pass-through around jsbsim.FGFDMExec that calls the recorder before every run()."""

    def __init__(self, fdm, recorder, dt):
        object.__setattr__(self, "_fdm", fdm)
        object.__setattr__(self, "_rec", recorder)
        object.__setattr__(self, "_k", 0)
        object.__setattr__(self, "_dt", dt)

    def run(self):
        if self._rec is not None:
            self._rec(self._k * self._dt, self._fdm)
        object.__setattr__(self, "_k", self._k + 1)
        return self._fdm.run()

    def __getitem__(self, k):
        return self._fdm[k]

    def __setitem__(self, k, v):  # the *controller* (ER's simulate) sets commands; recorders never do
        self._fdm[k] = v

    def __getattr__(self, name):
        return getattr(self._fdm, name)


def evaluate(genome: Dict[str, float], aircraft: str, scenario: Dict, run_cfg: Dict,
             recorder: Optional[Callable] = None) -> Dict:
    """Same signature as the agreed evolution.eval.evaluate. Rigid fidelity only (ER's current sim)."""
    fid = run_cfg.get("fidelity", "rigid")
    if fid not in ADAPTER_FIDELITIES:
        raise NotImplementedError(f"fidelity '{fid}' is not available through the legacy adapter (ER's current "
                                  f"evolution.sim is rigid only); it needs ER's evolution.eval")
    sim, _, _ = _er()
    sc, prof = scenario_object(aircraft, scenario, run_cfg)
    if _native_recorder():
        r = sim.simulate(dict(genome), sc, prof, record=False, recorder=recorder)
        r = {k: v for k, v in r.items() if k != "trajectory"}
        r["model_version"] = model_version_current(fid)
        r["fidelity"] = fid
        return r
    holder = {}
    orig = sim._new_fdm

    def patched(profile):
        fdm = orig(profile)
        prox = _RecordingFDM(fdm, recorder, sim.DT)
        holder["fdm"] = prox
        return prox

    sim._new_fdm = patched
    try:
        r = sim.simulate(dict(genome), sc, prof, record=False)
    finally:
        sim._new_fdm = orig
    if recorder is not None and "fdm" in holder and hasattr(recorder, "final"):
        p = holder["fdm"]
        recorder.final(p._k * sim.DT, p._fdm)
    r = {k: v for k, v in r.items() if k != "trajectory"}
    r["model_version"] = model_version_current(fid)
    r["fidelity"] = fid
    return r


def export_interface(run_dir: str, out_dir: str) -> Dict[str, str]:
    """Write run.json + genomes.jsonl in the agreed format (fixtures / what ER's fast mode should produce)."""
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_run_cfg(run_dir)
    rows = genome_rows(run_dir, cfg)
    pr = os.path.join(out_dir, "run.json")
    with open(pr, "w") as f:
        json.dump({k: v for k, v in cfg.items() if not k.startswith("_")} | {"schema": INTERFACE_SCHEMA}, f, indent=1)
    pg = os.path.join(out_dir, "genomes.jsonl")
    with open(pg, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("genome_norm", "rank", "per_scenario_status")}) + "\n")
    return {"run.json": pr, "genomes.jsonl": pg}
