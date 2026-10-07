"""run.json + genomes.jsonl: the replay interface (Sim Bridge) for a run directory.

    runs/<run_id>/run.json        everything needed to rebuild an evaluation exactly (schema ga-flightsim-run/1)
    runs/<run_id>/genomes.jsonl   one row per individual per generation (schema ga-flightsim-genomes/1)

The batch writes both for new runs. Completed older runs can be backfilled (best-of-generation rows only, marked
"backfill": true, because older checkpoints keep only the best genome of each generation and the final population):

    python -m evolution.runinfo --backfill evolution/runs/phase1-s1 [more run dirs ...]
    python -m evolution.runinfo --backfill-all          # every runs/* with a config.json and no run.json

A replay of a row is evolution.eval.evaluate(row["genome"], row["aircraft"], <scenario>, run.json,
fidelity=row["fidelity"]) and must give row["per_scenario_cost"] (and row["cost"] = float(np.mean(...)) of them)
exactly while model_version is unchanged.

``fd_dir`` records the resolved Flight Dynamics tree used for the run (``$EVOLUTION_FD_DIR`` or the default
``flight-dynamics/``, team-relative). Older runs may instead carry a sidecar ``fd_pin.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

from . import genome as genome_mod
from . import sim

RUN_SCHEMA = "ga-flightsim-run/1"
# Ids are opaque, stable strings (never renumbered across resumes); consumers must match them exactly, not parse them.
INDIVIDUAL_ID_FORMAT = ("'<aircraft>:g<generation>:r<rank>' (string; rank 0 = best of that generation by the generation's "
                        "ranking; unique per (aircraft, generation)); backfilled best-only rows use '<aircraft>:g<generation>:best'")
SCENARIO_ID_FORMAT = "'<aircraft>:s<index>' (string; index = position in the aircraft's scenario_ids)"
GENOMES_SCHEMA = "ga-flightsim-genomes/1"
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_SEMANTICS = {
    "steps": "absolute altitude commands [[t_s, alt_ft], ...] (instant steps) = target_cmd (channel target_cmd_alt_m)",
    "ramp_fpm / ramp_accel_g": "if ramp_fpm is set, the tracked/scored reference (channel target_alt_m) moves toward the "
                               "command at <= ramp_fpm with <= ramp_accel_g corners; null = the reference is the step",
    "ramp_plan": "the resolved accel-limited reference as constant-acceleration phases [[t0_s, p0_ft, v0_fps, a_fps2], ...]; "
                 "reference(t) = p0 + v0*(t-t0) + a*(t-t0)^2/2 in the last phase with t0 <= t (null without ramp corners)",
    "how to evaluate": "evolution.eval.scenario_object(aircraft, scenario, run_cfg) -> sim.Scenario with target(t), "
                       "target_cmd(t), target_rate(t); evolution.eval.evaluate accepts these dicts directly",
}


def scenario_id(aircraft: str, index: int) -> str:
    return f"{aircraft}:s{index}"


def scenario_entries(aircraft: str, profile_d: Dict, n: int, scenario_seed: int) -> List[Dict]:
    P = sim.Profile.from_dict(profile_d)
    out = []
    for i, sc in enumerate(sim.make_scenarios(n, scenario_seed, P)):
        d = {"id": scenario_id(aircraft, i), "aircraft": aircraft, "index": i}
        d.update(sc.to_dict())
        d["ramp_plan"] = [list(ph) for ph in sc.ramp_plan()] if (sc.ramp_fpm is not None and sc.ramp_accel_g) else None
        out.append(d)
    return out


def scenario_fields(d: Dict) -> Dict:
    """run.json scenario entry -> the sim.Scenario constructor fields (extra keys dropped)."""
    names = set(sim.Scenario.__dataclass_fields__)
    return {k: v for k, v in d.items() if k in names}


def gene_entries(schema, groups: Optional[List[str]] = None) -> List[Dict]:
    groups = groups or ["gains"] * len(schema)
    return [{"name": g.name, "min": g.min, "max": g.max, "kind": g.kind, "zero_band": g.zero_band if g.kind == "log0" else None,
             "units": g.units, "group": grp} for g, grp in zip(schema, groups)]


def fitness_cfg(profile_d: Dict) -> Dict:
    P = sim.Profile.from_dict(profile_d)
    f = {"per_scenario": "track + w_effort*effort (+ w_comfort*comfort) (+ w_heading*RMS(e_psi)/hdg_rms_ref_deg); "
                         "failed flight: fail_base*(2 - t_end/duration); load/trim failure: 2*fail_base",
         "track": "mean over steps of |e_h|/alt_err_scale_ft * min(t - t_cmd, itae_cap_s)/itae_t0_s",
         "effort": "sum |delta elevator cmd| / duration_s",
         "alt_err_scale_ft": P.alt_err_scale_ft, "itae_t0_s": P.itae_t0_s, "itae_cap_s": P.itae_cap_s,
         "w_effort": P.w_effort, "w_comfort": P.w_comfort, "comfort_params": P.comfort_params,
         "comfort_weights": P.comfort_weights, "fail_base": P.fail_base,
         "heading_hold": P.heading_hold, "w_heading": P.w_heading, "hdg_rms_ref_deg": P.hdg_rms_ref_deg}
    if P.w_hold > 0 or P.disturbance_scenario:
        f["per_scenario"] += " (+ w_hold*hold_osc, v5)"
        f.update({"w_hold": P.w_hold, "hold_ref_ft": P.hold_ref_ft, "hold_settle_s": P.hold_settle_s,
                  "hold_osc": "RMS over hold samples of (e - mean of e in its window), e = h_cmd - h, / hold_ref_ft; hold "
                              "= reference == command (|ref-cmd| < 1e-6 ft) and reference rate 0, minus the first "
                              "hold_settle_s of each stretch",
                  "disturbance_scenario": P.disturbance_scenario,
                  "disturbance_doc": "one extra calm scenario appended last (index = scenarios): holds h0 in a sustained "
                                     "downdraft w = draft_fps*0.5*(1-cos(pi*clip((t-draft_t_s)/draft_ramp_s,0,1))) added "
                                     "to wind-down-fps (+ = down); cost = mean over all scenarios"})
    return f


def team_rel(path: Optional[str]) -> Optional[str]:
    """Absolute path -> relative to the team root (sim.TEAM_ROOT, the dir holding evolution/ and flight-dynamics/);
    relative paths and None pass through. sim.abs_root() is the inverse (Sim Bridge resolves the same way)."""
    if not path or not os.path.isabs(path):
        return path
    return os.path.relpath(path, sim.TEAM_ROOT)


def build_run_json(cfg: Dict, prov: Dict, per_ac: Dict[str, Dict], created: str) -> Dict:
    """cfg: resolved batch config; prov: provenance; per_ac[name]: {schema, groups, model_files_sha, model_version,
    screen_model_version?}."""
    fid = cfg.get("fidelity", "rigid")
    mf = cfg.get("multi_fidelity") or {"enabled": False}
    acs, scen = [], []
    for a in cfg["aircraft"]:
        info = per_ac.get(a["name"], {})
        ents = scenario_entries(a["name"], a["resolved_profile"], cfg["scenarios"], cfg["scenario_seed"])
        scen += ents
        rp = dict(a["resolved_profile"])
        if rp.get("aircraft_root"):
            rp["aircraft_root"] = team_rel(rp["aircraft_root"])
        e = {"name": a["name"], "profile": a["profile"], "seed": a["seed"], "resolved_profile": rp,
             "aircraft_root": rp.get("aircraft_root"), "scenario_ids": [s["id"] for s in ents],
             "genes": gene_entries(info["schema"], info.get("groups")) if "schema" in info else None,
             "model_files_sha": info.get("model_files_sha"), "model_version": info.get("model_version"),
             "fitness_cfg": fitness_cfg(a["resolved_profile"])}
        if info.get("screen_model_version"):
            e["screen_model_version"] = info["screen_model_version"]
        if a.get("multi_fidelity"):   # multi_fidelity_per_aircraft override (else the run-level block applies)
            e["multi_fidelity"] = a["multi_fidelity"] if a["multi_fidelity"].get("enabled") else None
        for k in ("reduced_gate", "ladder_model_version", "min_full_frac", "model_files_sha_source",
                  "model_files_current_differs"):
            if k in info:
                e[k] = info[k]
        acs.append(e)
    labels = {"rigid": "rigid", "reduced": "reduced(flexv1 on projected v2 genome)", "full": "full(flexv2)",
              "full_a1": "full_a1(flexv2a1: P3-A1, 64-strip 4b+3t+2ip wings)",
              "full_a1_b1": "full_a1_b1(flexv2b1: P3-B1, A1 host + 6 planform shape genes)",
              "full_a1_b2a": "full_a1_b2a(flexv2b2a: P3-B2a, B1 r1 + dihedral / thickness / camber)"}
    ladders = [list(m.get("ladder") or []) for m in [mf] + [a.get("multi_fidelity") or {} for a in cfg["aircraft"]]
               if m.get("enabled")]
    uses_a1 = fid == "full_a1" or any("full_a1" in ld for ld in ladders)
    uses_b1 = fid == "full_a1_b1" or any("full_a1_b1" in ld for ld in ladders)
    uses_b2 = fid == "full_a1_b2a" or any("full_a1_b2a" in ld for ld in ladders)
    return {
        "schema": RUN_SCHEMA, "run_id": cfg["run_id"], "created": created,
        "git_sha": prov.get("git_sha"),
        "git": dict(prov["git"], repo=team_rel(prov["git"].get("repo"))) if isinstance(prov.get("git"), dict) else prov.get("git"),
        "paths_relative_to": "team root: the directory holding evolution/ and flight-dynamics/ (flight_sim_3d/ in the repo); "
                             "resolve with evolution.sim.abs_root",
        "fd_dir": team_rel(sim.fd_dir()),  # resolved FD tree actually used ($EVOLUTION_FD_DIR or flight-dynamics/)
        "jsbsim_version": prov.get("jsbsim_version"),
        "code_sha": prov.get("code_sha"), "seed": cfg["seed"], "eval_seed": cfg["scenario_seed"],
        "scenario_seed": cfg["scenario_seed"], "fitness_sense": "min", "sim_dt_s": sim.DT,
        "fidelity": fid, "fidelity_label": labels.get(fid, fid),
        "multi_fidelity": mf if mf.get("enabled") else None,
        "struct_genes": bool(cfg.get("struct_genes", False)),
        **({"struct_asymmetric": True} if cfg.get("struct_asymmetric") else {}),
        **({"init": cfg["init"]} if (cfg.get("init") or {}).get("mode", "uniform") != "uniform" else {}),
        **({"pin_model_version": cfg["pin_model_version"]} if cfg.get("pin_model_version") else {}),
        **({"genome_kind": cfg["genome_kind"], "shape_ops": cfg.get("shape_ops")} if cfg.get("genome_kind") else {}),
        **({"shape_locked": (cfg.get("shape_locked") if cfg.get("shape_locked") is not None
                             else ["wing_tc_root_scale", "wing_tc_tip_ratio"])} if cfg.get("genome_kind") == "phase3_b2a" else {}),
        **({"energy_cost": bool(cfg.get("energy_cost"))} if uses_b2 else {}),
        "model_version": {a["name"]: a["model_version"] for a in acs},
        "aircraft": acs, "scenarios": scen, "target_semantics": TARGET_SEMANTICS,
        "fitness_cfg": {"sense": "min", "aggregate": "cost = float(numpy.mean(per_scenario_cost)) over the aircraft's "
                        "scenario_ids, at every fidelity",
                        "flex": "reduced/full = Flight Dynamics flexeval.evaluate (flight-dynamics/INTERFACE_v2.md): "
                                "per-scenario cost = rigid sim cost + response terms (fail_cost on a structural-ultimate "
                                "fail) + pre-flight terms (margins, mass, smoothness); a genome with a margin below the "
                                "gate is not flown (cost = fail_cost = 2 * fail_base). Gates: full 1.0; reduced = "
                                "aircraft[].reduced_gate (c172x 0.9, swept wings 1.0)",
                        **({"flex_a1": "full_a1 = Flight Dynamics flexeval_a1.evaluate (INTERFACE_v2.md section 13): the "
                                       "full contract (gate 1.0, same 24 terms, <root>_v2, 2 substeps) on the P3-A1 model "
                                       "flexbody_a1.FlexBodyModelA1 (64 strips, 4b+3t+2ip per semi-wing); "
                                       "J_wing_tip_bm_limit is station-exact at eta 0.875, so full_a1 costs are not "
                                       "comparable with full costs and never share a cache entry with them"}
                           if uses_a1 else {}),
                        **({"flex_b1": "full_a1_b1 = Flight Dynamics flexeval_b1.evaluate (INTERFACE_v2.md section 14): "
                                       "the full_a1 contract on flexbody_b1.FlexBodyModelB1 with FD's 6 planform shape "
                                       "genes (aircraft[].genes group 'shape'; chord tapers 1..3, twist mid / tip, "
                                       "quarter-chord sweep delta; L = R). Decode order shape -> geometry gate -> "
                                       "rebuild -> structure genes -> margins + flight. A geometry-gate reject is NOT "
                                       "flown: status 'geometry_gate:<reason>', cost = fail_cost (2 * fail_base), never a "
                                       "credit. Baseline shape = full_a1 bit for bit; costs share no cache entry with "
                                       "full_a1 (key carries fidelity, model_version and FD's shape_cache_key). Rigid "
                                       "screens ignore the shape genes"}
                           if uses_b1 else {}),
                        **({"flex_b2a": "full_a1_b2a = Flight Dynamics flexeval_b2.evaluate (INTERFACE_v2.md section 15): "
                                        "the full_a1_b1 contract plus FD's 5 B2a section genes (dihedral, t/c root / tip "
                                        "ratio, camber root / tip; per-aircraft ranges; locked genes = FD default). "
                                        "Geometry gates B1 + B2 (reject = fail_cost, not flown). energy_cost true: per "
                                        "scenario cost += J_energy_s + J_speed_guard_s (w_E max(0, energy_drag_increment); "
                                        "w_E 3 max(0, speed_deficit_kts_mean - 2) / v_target_kcas; Evolution-side, outside "
                                        "TERM_KEYS, INTERFACE_v2 15.10.3) for status-ok genomes; row energy_terms. Baseline "
                                        "B2 with energy off = full_a1_b1 cost bit for bit"}
                           if uses_b2 else {}),
                        "per_aircraft": "aircraft[].fitness_cfg"},
        "ga": cfg.get("ga"), "generations": cfg["ga"]["generations"], "pop_size": cfg["ga"]["pop_size"],
        "trajectories": cfg.get("trajectories"),
        "eval": {"entry": "evolution.eval:evaluate", "signature": "evaluate(genome, aircraft, scenario, run_cfg, recorder=None, *, fidelity=None)",
                 "genome_format": "decoded {gene name: value} (genomes.jsonl 'genome'); a normalized list in aircraft[].genes order also works",
                 "scenario_format": "a run.json scenarios[] entry, its id, a list of either, or None (= all of the aircraft's)",
                 "recorder": "recorder(t, fdm[, flex_state]) at t=0 after IC+trim, then after every step; optional "
                             "recorder.final(t_end, fdm[, flex_state]); read-only",
                 "controls_timing": "pre_step (see trajectory files' controls_timing_doc)"},
        "genomes_file": "genomes.jsonl", "genomes_schema": GENOMES_SCHEMA,
        "individual_id_format": INDIVIDUAL_ID_FORMAT,
        "scenario_id_format": SCENARIO_ID_FORMAT,
    }


def write_json_atomic(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- genomes.jsonl
def read_rows(path: str) -> List[Dict]:
    """Rows of a genomes.jsonl; a torn last line (SIGKILL mid-write) is skipped."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def truncate_rows(path: str, gen_next: Dict[str, int]) -> int:
    """Keep rows with generation < gen_next[aircraft] (the checkpointed generations), drop duplicates; returns kept."""
    rows = read_rows(path)
    seen, keep = set(), []
    for r in rows:
        k = (r["aircraft"], r["individual_id"])
        if r["generation"] < gen_next.get(r["aircraft"], 0) and k not in seen:
            seen.add(k)
            keep.append(r)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    return len(keep)


# ----------------------------------------------------------------------------- backfill
def _rigid_mv(jsbsim_version: str, model_files_sha: str) -> str:
    from .fidelity import _sha8
    return f"rigid:jsbsim{jsbsim_version}:" + _sha8({"model_files": model_files_sha})


def backfill(run_dir: str, force: bool = False, log=print) -> Dict:
    """run.json (+ best-of-generation genomes.jsonl) for a completed run written before the interface existed."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cj = json.load(f)
    cfg, prov = cj["resolved"], cj.get("provenance", {})
    rj, gj = os.path.join(run_dir, "run.json"), os.path.join(run_dir, "genomes.jsonl")
    if os.path.exists(rj) and not force:
        return {"run_dir": run_dir, "skipped": "run.json exists"}
    summ = {}
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            summ = {a["aircraft"]: a for a in json.load(f).get("aircraft", []) if "aircraft" in a}
    per_ac = {}
    for a in cfg["aircraft"]:
        P = sim.Profile.from_dict(a["resolved_profile"])
        schema = genome_mod.make_schema(P.gain_bounds, P.gene_kinds, P.heading_hold)
        cur = sim.model_files_sha(a["name"], P.aircraft_root)
        stored = summ.get(a["name"], {}).get("model_files_sha")
        mfs = stored or cur
        per_ac[a["name"]] = {"schema": schema, "model_files_sha": mfs,
                             "model_files_sha_source": "summary.json (as flown)" if stored else "current files at backfill time",
                             "model_files_current_differs": bool(stored and stored != cur),
                             "model_version": _rigid_mv(prov.get("jsbsim_version", "?"), mfs)}
    doc = build_run_json(dict(cfg, fidelity="rigid"), prov, per_ac, cj.get("created", "?"))
    doc["backfill"] = {"from": ["config.json", "summary.json", "checkpoints/*.json"],
                       "note": "pre-interface run: genomes.jsonl holds best-of-generation rows only (backfill: true); "
                               "code_sha is the code that produced the run"}
    write_json_atomic(rj, doc)
    n = 0
    tmp = gj + ".tmp"
    with open(tmp, "w") as f:
        for a in cfg["aircraft"]:
            ckp = os.path.join(run_dir, "checkpoints", f"{a['name']}.json")
            if not os.path.exists(ckp):
                continue
            with open(ckp) as fh:
                ck = json.load(fh)
            schema = per_ac[a["name"]]["schema"]
            sids = [scenario_id(a["name"], i) for i in range(cfg["scenarios"])]
            for b in ck["best_per_gen"]:
                row = {"generation": b["generation"], "individual_id": f"{a['name']}:g{b['generation']}:best",
                       "aircraft": a["name"], "eval_seed": cfg["scenario_seed"], "scenario_ids": sids, "cost": b["fitness"],
                       "per_scenario_cost": [s["cost"] for s in b["per_scenario"]],
                       "per_scenario_status": [s["status"] for s in b["per_scenario"]],
                       "genome": genome_mod.decode(b["genome"], schema), "genome_norm": b["genome"],
                       "eval_seed": cfg["scenario_seed"], "fidelity": "rigid",
                       "model_version": per_ac[a["name"]]["model_version"], "rank": 0, "is_best": True, "is_elite": True,
                       "status": next((s["status"] for s in b["per_scenario"] if s["status"] != "ok"), "ok"),
                       "backfill": True}
                f.write(json.dumps(row) + "\n")
                n += 1
    os.replace(tmp, gj)
    log(f"backfilled {run_dir}: run.json + {n} best-of-generation rows"
        + "".join(f"  [{k}: model files changed since the run]" for k, v in per_ac.items() if v["model_files_current_differs"]))
    return {"run_dir": run_dir, "rows": n, "model_files_current_differs": {k: v["model_files_current_differs"] for k, v in per_ac.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill", nargs="*", default=[], metavar="RUN_DIR")
    ap.add_argument("--backfill-all", action="store_true", help="every runs/* with config.json + summary.json and no run.json")
    ap.add_argument("--force", action="store_true", help="rewrite an existing run.json/genomes.jsonl (backfilled runs only)")
    a = ap.parse_args(argv)
    dirs = list(a.backfill)
    if a.backfill_all:
        rd = os.path.join(PKG_DIR, "runs")
        dirs += [os.path.join(rd, d) for d in sorted(os.listdir(rd))
                 if os.path.exists(os.path.join(rd, d, "config.json")) and os.path.exists(os.path.join(rd, d, "summary.json"))]
    for d in dirs:
        try:
            if a.force and os.path.exists(os.path.join(d, "run.json")):
                with open(os.path.join(d, "run.json")) as f:
                    if "backfill" not in json.load(f):
                        print(f"{d}: native run.json, not overwritten")
                        continue
            r = backfill(d, force=a.force)
            if "skipped" in r:
                print(f"{d}: {r['skipped']}")
        except Exception as e:  # noqa: BLE001
            print(f"{d}: FAILED {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
