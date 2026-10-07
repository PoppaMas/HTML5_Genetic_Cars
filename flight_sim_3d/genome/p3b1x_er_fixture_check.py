#!/usr/bin/env python3
"""Genome-side diff against Evolution Runner's tweaked-preset fixture (read-only):
evolution/analysis/tweaked_preset_crosscheck.json (T38 gen-4 population of phase3b1r1-smoke-s1, ranked; seeds 0..4;
variants default = phase3_b1 operators (elite 2, shape_crossover 'block') and tweaked = phase3_b1_x (elite 4, 'uniform')).

For each (variant, seed): rng = default_rng(seed); block_ops.next_generation(ga, rng, ranked, GAConfig(variant), ops)
with ops from genome's own presets (phase3_b1 / phase3_b1_x), trace hook on. Diffed bit for bit against ER's
next-generation rows, their SHA-256, the RNG end state, and the per-child trace (parents i, j; crossover draws -> from_a).
The fixture's inputs (blocks, shape spec) are also checked against genome's preset operators.

Usage: $PY genome/p3b1x_er_fixture_check.py   -> genome/runs/p3b1x_er_fixture_check.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("FLIGHT_SIM_DIR", os.path.join(os.path.dirname(os.path.dirname(HERE)), "flight_sim"))  # repo ../../flight_sim

import numpy as np  # noqa: E402

import adapter  # noqa: E402
import block_ops  # noqa: E402
from flightsim_path import orig_ga  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(HERE), "evolution", "analysis", "tweaked_preset_crosscheck.json")
PRESET = {"default": "phase3_b1", "tweaked": "phase3_b1_x"}


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype="<f8")).tobytes()).hexdigest()


def rng_sha(rng) -> str:
    return hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()


def from_a(rec) -> list:
    return rec["blk"] if "blk" in rec else [rec["ctrl_a"], rec["struct_a"], *rec["shape_mask_a"]]


def check(fx: dict) -> dict:
    GA = orig_ga()
    inp = fx["inputs"]
    ranked = np.array(inp["ranked_population"], dtype=np.float64)
    out = {"fixture": os.path.relpath(FIXTURE, os.path.dirname(HERE)), "fixture_created": fx.get("created"),
           "fixture_schema": fx.get("schema"), "fixture_generator": fx.get("generator"),
           "block_ops_sha256": hashlib.sha256(open(os.path.join(HERE, "block_ops.py"), "rb").read()).hexdigest(),
           "inputs": {"source": inp["source"], "aircraft": inp["aircraft"], "generation": inp["generation"],
                      "seeds": inp["seeds"], "ranked_population_sha256_matches": sha(ranked) == inp["ranked_population_sha256"]},
           "variants": {}}
    for var, preset in PRESET.items():
        task = adapter.load_task(preset)
        ops, sp = task.operators, task.operators["shape_ops"]
        fs = inp["shape_spec"]
        cfgv = fx["variants"][var]
        inputs_match = {
            "blocks": [list(v) for v in ops["blocks"].values()] == [inp["blocks"][b] for b in ("controller", "structure", "shape")],
            "shape_spec": (list(sp.idx), list(sp.lo), list(sp.hi), list(sp.log), list(sp.default), sp.mut_sigma_frac)
                          == (fs["idx"], fs["lo"], fs["hi"], fs["log"], fs["default"], fs["mut_sigma_frac"]),
            "shape_mutation_rate": (sp.mutation_rate if sp.mutation_rate is not None else cfgv["mutation_rate"]) == fs["mutation_rate"],
            "gene_names": task.spec.names == inp["gene_names"],
            "elite": int(task.ga_overrides.get("elite", 2)) == cfgv["elite"],
            "shape_crossover": ops["shape_crossover"] == cfgv["shape_crossover"],
        }
        cfg = GA.GAConfig(pop_size=cfgv["pop_size"], elite=cfgv["elite"], selection_p=cfgv["selection_p"],
                          mutation_rate=cfgv["mutation_rate"], mutation_sigma=cfgv["mutation_sigma"],
                          mutation_mode=cfgv["mutation_mode"])
        per = {}
        for seed in inp["seeds"]:
            er = fx["results"][var][f"seed{seed}"]
            rng = np.random.default_rng(seed)
            tr: list = []
            mine = block_ops.next_generation(GA, rng, ranked, cfg, ops, tr) if var == "default" else \
                block_ops.next_generation_tweaked(GA, rng, ranked, cfg, ops, "uniform", tr)
            theirs = np.array(er["next_generation"], dtype=np.float64)
            my_tr = [{"child": cfg.elite + k, "i": r["i"], "j": r["j"], "from_a": from_a(r)} for k, r in enumerate(tr)]
            er_tr = [{"child": t["child"], "i": t["i"], "j": t["j"], "from_a": list(t["from_a"])} for t in er["trace"]]
            per[f"seed{seed}"] = {
                "genome_sha256": sha(mine), "er_sha256": er["sha256"],
                "bit_identical_rows": bool(np.array_equal(mine, theirs)) and mine.tobytes() == theirs.tobytes(),
                "sha256_identical": sha(mine) == er["sha256"],
                "rng_end_identical": rng_sha(rng) == er["rng_after_sha256"],
                "trace_identical": my_tr == er_tr,
                "elites_unchanged": bool(np.array_equal(mine[:cfg.elite], ranked[:cfg.elite])),
                "n_diff_cells": int((mine != theirs).sum()),
            }
        ok = all(all(v for k, v in p.items() if isinstance(v, bool)) for p in per.values()) and all(inputs_match.values())
        out["variants"][var] = {"preset": preset, "entry": "block_ops.next_generation" if var == "default"
                                else "block_ops.next_generation_tweaked", "ga": cfgv, "inputs_match": inputs_match,
                                "status": "BIT-IDENTICAL" if ok else "MISMATCH", "per_seed": per}
    out["summary"] = {v: r["status"] for v, r in out["variants"].items()}
    out["summary"]["ranked_population_sha256_matches"] = out["inputs"]["ranked_population_sha256_matches"]
    if "genome_crosscheck" in fx:
        out["er_recorded_genome_crosscheck"] = {
            "note": "ER ran genome block_ops read-only from their side too (fixture's genome_crosscheck)",
            "default": fx["genome_crosscheck"]["default"]["status"], "tweaked": fx["genome_crosscheck"]["tweaked"]["status"],
            "block_ops_sha256_then": fx["genome_crosscheck"].get("block_ops_sha256"),
            "same_block_ops_as_now": fx["genome_crosscheck"].get("block_ops_sha256") == out["block_ops_sha256"]}
    return out


def main():
    if not os.path.exists(FIXTURE):
        print("ER fixture not present:", FIXTURE)
        sys.exit(2)
    out = check(json.load(open(FIXTURE)))
    dest = os.path.join(HERE, "runs", "p3b1x_er_fixture_check.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", dest)
    print(json.dumps({"summary": out["summary"], "er_recorded": out.get("er_recorded_genome_crosscheck"),
                      "inputs_match": {v: r["inputs_match"] for v, r in out["variants"].items()}}, indent=1))


if __name__ == "__main__":
    main()
