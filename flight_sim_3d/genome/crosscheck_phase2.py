#!/usr/bin/env python3
"""Bit-for-bit cross-check of genome's phase2_flex against Evolution Runner's Phase 2 evaluator (read-only on evolution/).

ER's code is imported from a temporary COPY of evolution/*.py (package 'evolution' under a /tmp dir), in its own
subprocess, with bytecode writing off; FD is found through EVOLUTION_FD_DIR (ER's own switch). Nothing is written into
evolution/ or flight-dynamics/. Each side runs in a separate process (both import FD's flexbody; ER patches FD's prepare
functions in-process). Genomes: ER's phase2-smoke-s1 best per aircraft (or --aircraft) and the same gains with FD's
baseline structure. Full fidelity, ER's 3 scenarios at scenario_seed 1.

Usage: $PY crosscheck_phase2.py [--aircraft c172x|T38|737] [--out runs/crosscheck_phase2_<aircraft>.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
EVO = os.environ.get("EVOLUTION_DIR", os.path.join(TEAM, "evolution"))
FD = os.environ.get("FLIGHT_DYNAMICS_DIR", os.path.join(TEAM, "flight-dynamics"))
SMOKE = os.path.join(EVO, "runs", "phase2-smoke-s1")
PILOT = os.path.join(EVO, "configs", "phase2_pilot.json")
OURS = {"c172x": "c172x", "T38": "t38", "737": "b737"}


def pick_genomes(ac: str):
    rows = [json.loads(l) for l in open(os.path.join(SMOKE, "genomes.jsonl"))]
    best = min((r for r in rows if r["aircraft"] == ac and r.get("feasible")), key=lambda r: r["cost"])
    return best


def er_side(args):
    """Runs in its own process: ER's evaluate_genome from a /tmp copy of evolution/."""
    tmp = tempfile.mkdtemp(prefix="er_copy_")
    pkg = os.path.join(tmp, "evolution")
    os.makedirs(pkg)
    for f in glob.glob(os.path.join(EVO, "*.py")):
        shutil.copy2(f, pkg)
    os.environ["EVOLUTION_FD_DIR"] = FD
    sys.dont_write_bytecode = True
    sys.path.insert(0, tmp)
    from evolution import fidelity, sim, genome as egenome  # noqa: E402
    cfg = json.load(open(os.path.join(SMOKE, "config.json")))["resolved"]
    prof_d = next(a for a in cfg["aircraft"] if a["name"] == args.aircraft)["resolved_profile"]
    pilot = json.load(open(PILOT))
    pp = pilot["profiles"][f"phase2_{args.aircraft}"]
    prof_diff = {k: (v, prof_d.get(k)) for k, v in pp.items() if not k.startswith("_") and k != "aircraft_root"
                 and json.dumps(v) != json.dumps(prof_d.get(k))}
    P = sim.Profile.from_dict(prof_d)
    scs = [s.to_dict() for s in sim.make_scenarios(cfg["scenarios"], cfg["scenario_seed"], P)]
    sch = egenome.make_schema(prof_d.get("gain_bounds"), prof_d.get("gene_kinds"), heading_hold=prof_d.get("heading_hold", False))
    ssch = fidelity.struct_schema(False)
    out = {"profile_vs_pilot_diff": prof_diff, "scenarios": scs, "pilot_init": pilot["init"],
           "schema": [[g.name, g.min, g.max, g.kind] for g in sch + ssch], "evals": {}}
    for name, (gains, struct) in json.load(open(args.genomes)).items():
        r = fidelity.evaluate_genome(prof_d, gains, struct, scs, "full", None)
        out["evals"][name] = {"cost": r["cost"], "status": r["status"], "terms": r["terms"],
                              "per_scenario": [{k: e.get(k) for k in ("cost", "sim_cost", "status", "track", "effort",
                                                                       "comfort", "heading_rms")} for e in r["per_scenario"]],
                              "per_scenario_struct": [e.get("struct") for e in r["per_scenario"]],
                              "margins": r["margins"], "mass_total_frac": r["mass_total_frac"], "model_version": r["model_version"]}
    shutil.rmtree(tmp)
    json.dump(out, open(args.result, "w"), indent=1)


def genome_side(args):
    sys.path.insert(0, HERE)
    import adapter  # noqa: E402
    t = adapter.load_task("phase2_flex", {"aircraft": OURS[args.aircraft]})
    scs = t.make_scenarios(3, 1)
    out = {"schema": [[g.name, g.min, g.max, g.scale] for g in t.spec.genes], "init": t.init,
           "scenarios": [{k: getattr(s, k) for k in ("seed", "wind_north_fps", "wind_east_fps", "gust_sigma_fps",
                                                     "discrete_gust_fps", "discrete_gust_t_s", "duration_s", "h0_ft",
                                                     "speed_kts", "ramp_fpm", "ramp_accel_g")} | {"steps": [list(x) for x in s.steps]}
                         for s in scs],
           "weights": t.fitness.weights, "params": {k: v for k, v in t.fitness.params.items() if k.startswith("struct_v2")},
           "warnings": t.warnings, "evals": {}}
    for name, (gains, struct) in json.load(open(args.genomes)).items():
        r = t.evaluate({**gains, **struct}, scs)
        ps = r["per_scenario"]
        objs = r["objectives_per_scenario"]
        out["evals"][name] = {"cost": r["cost"], "per_scenario": [{"cost": p["cost"], "status": p["status"],
                                                                   "track": p["track"], "effort": p["effort"],
                                                                   "comfort": o.get("comfort"), "heading_rms": o.get("track_heading_rms"),
                                                                   "structural_v2": o.get("structural_v2")} for p, o in zip(ps, objs)],
                              "fd_struct_terms": r.get("fd_struct_terms"), "fd_pre_sum": r.get("fd_pre_sum"),
                              "objectives": r["objectives"], "aeroelastic": {k: r["aeroelastic"][k] for k in
                                                                             ("flutter_margin", "div_margin", "reversal_margin", "min_margin", "binding", "fail")},
                              "mass_frac": r["aeroelastic"]["mass_lb"]["total_frac"],
                              "encode": t.spec.encode({**gains, **struct})}
    json.dump(out, open(args.result, "w"), indent=1)


def compare(er, ge, best):
    rep = {"schema_equal": [[a[0], a[1], a[2], a[3]] for a in er["schema"]] == [[b[0], b[1], b[2], b[3]] for b in ge["schema"]],
           "schema_diff": [(a, b) for a, b in zip(er["schema"], ge["schema"]) if a != b],
           "profile_vs_pilot_diff": er["profile_vs_pilot_diff"], "init": {"er_pilot": er["pilot_init"], "genome": ge["init"]}}
    sc_diff = []
    for a, b in zip(er["scenarios"], ge["scenarios"]):
        for k, v in b.items():
            av = a.get(k)
            if json.dumps(av) != json.dumps(v):
                sc_diff.append((k, av, v))
    rep["scenario_diff"] = sc_diff
    rep["encode_max_abs_diff_vs_er_genome_norm"] = max(abs(x - y) for x, y in zip(ge["evals"]["er_best"]["encode"], best["genome_norm"]))
    rep["evals"] = {}
    for name in er["evals"]:
        e, g = er["evals"][name], ge["evals"][name]
        row = {"er_cost": e["cost"], "genome_cost": g["cost"], "cost_bitwise": e["cost"] == g["cost"],
               "cost_abs_diff": abs(e["cost"] - g["cost"]), "status": (e["status"], [p["status"] for p in g["per_scenario"]])}
        row["per_scenario"] = [{"er": a["cost"], "genome": b["cost"], "bitwise": a["cost"] == b["cost"],
                                "sim_cost_er": a["sim_cost"], "status": (a["status"], b["status"])}
                               for a, b in zip(e["per_scenario"], g["per_scenario"])]
        ft = g["fd_struct_terms"] or {}
        row["terms"] = {k: {"er": e["terms"][k], "genome": ft.get(k), "bitwise": e["terms"][k] == ft.get(k)}
                        for k in e["terms"] if k.startswith("J_")}
        gc = {"track": "track_alt", "effort": "effort", "comfort": "comfort", "heading": "track_heading_rms"}
        row["controller_terms"] = {k: {"er": e["terms"][k], "genome": g["objectives"].get(v), "bitwise": e["terms"][k] == g["objectives"].get(v)}
                                   for k, v in gc.items()}
        row["margins"] = {"er": e["margins"], "genome": g["aeroelastic"]}
        row["mass_frac"] = {"er": e["mass_total_frac"], "genome": g["mass_frac"], "bitwise": e["mass_total_frac"] == g["mass_frac"]}
        rep["evals"][name] = row
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aircraft", default="c172x", choices=sorted(OURS))
    ap.add_argument("--out", default=None)  # default runs/crosscheck_phase2_<aircraft>.json
    ap.add_argument("--side", choices=("er", "genome"))
    ap.add_argument("--genomes")
    ap.add_argument("--result")
    a = ap.parse_args(argv)
    a.out = a.out or os.path.join(HERE, "runs", f"crosscheck_phase2_{a.aircraft}.json")
    if a.side == "er":
        return er_side(a)
    if a.side == "genome":
        return genome_side(a)
    best = pick_genomes(a.aircraft)
    sys.path.insert(0, HERE)
    base_struct = {k: (0.02 if k == "struct_damping_ratio" else 1.0) for k in best["struct"]}
    genomes = {"er_best": [best["gains"], best["struct"]], "er_best_gains_fd_baseline_struct": [best["gains"], base_struct]}
    rows = {r["individual_id"]: r for r in map(json.loads, open(os.path.join(SMOKE, "genomes.jsonl")))}
    fails = [r for r in rows.values() if r["aircraft"] == a.aircraft and r["status"] != "ok"]
    picked = {}
    for r in sorted(fails, key=lambda r: r["individual_id"]):  # one infeasible genome per failure kind (gate, envelope, ultimate)
        picked.setdefault(r["status"], r)
    for st, r in picked.items():
        genomes[f"er_{st}:{r['individual_id']}"] = [r["gains"], r["struct"]]
    recorded = {"er_best": best["cost"], **{f"er_{st}:{r['individual_id']}": r["cost"] for st, r in picked.items()}}
    tmp = tempfile.mkdtemp(prefix="xc_p2_")
    gpath = os.path.join(tmp, "genomes.json")
    json.dump(genomes, open(gpath, "w"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    res = {}
    for side in ("er", "genome"):
        rp = os.path.join(tmp, f"{side}.json")
        subprocess.run([sys.executable, os.path.abspath(__file__), "--side", side, "--aircraft", a.aircraft,
                        "--genomes", gpath, "--result", rp], check=True, env=env)
        res[side] = json.load(open(rp))
    rep = {"aircraft": a.aircraft, "er_run": "phase2-smoke-s1", "er_best_id": best["individual_id"],
           "er_recorded_cost": best["cost"], **compare(res["er"], res["genome"], best), "raw": res}
    rep["er_recompute_equals_recorded"] = {k: res["er"]["evals"][k]["cost"] == v for k, v in recorded.items()}
    shutil.rmtree(tmp)
    json.dump(rep, open(a.out, "w"), indent=1)
    s = {k: v for k, v in rep.items() if k != "raw"}
    print(json.dumps({k: s[k] for k in ("schema_equal", "schema_diff", "scenario_diff", "profile_vs_pilot_diff",
                                        "encode_max_abs_diff_vs_er_genome_norm", "er_recompute_equals_recorded")}, indent=1))
    for n, r in s["evals"].items():
        print(n, "ER", r["er_cost"], "genome", r["genome_cost"], "bitwise", r["cost_bitwise"], "diff", r["cost_abs_diff"])
        print("  per-scenario bitwise", [p["bitwise"] for p in r["per_scenario"]],
              "controller", {k: v["bitwise"] for k, v in r["controller_terms"].items()}, "status", r["status"])
        print("  J terms not bitwise", {k: (v["er"], v["genome"]) for k, v in r["terms"].items() if not v["bitwise"]})


if __name__ == "__main__":
    main()
