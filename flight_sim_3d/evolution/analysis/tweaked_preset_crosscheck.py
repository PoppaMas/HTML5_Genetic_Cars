"""Deterministic cross-check fixture for the opt-in TWEAKED GA preset (ga.elite 4 + ga.shape_crossover 'uniform';
Genome Architect's name: phase3_b1_x) vs the default phase3_b1 operators, for Genome's block_ops to diff against.

Run from the team root:
  $PY evolution/analysis/tweaked_preset_crosscheck.py            # (re)write evolution/analysis/tweaked_preset_crosscheck.json
  $PY evolution/analysis/tweaked_preset_crosscheck.py --genome   # + run genome/block_ops.py READ-ONLY on the same inputs
                                                                 #   and record the bit-for-bit result in the fixture

Inputs: the T38 generation-4 population of runs/phase3b1r1-smoke-s1 (genomes.jsonl, 16 rows sorted by rank = the cost
ranking the GA loop passes to next_generation_blocks), GA settings of configs/phase3b1_pilot.json (selection_p 0.2,
mutation_rate 0.15, mutation_sigma 0.08, shape_ops from the resolved config), pop_size 16 (the smoke's), RNG
numpy.random.default_rng(seed) (PCG64) for seeds 0..4, one next_generation_blocks call per (variant, seed).
Variants: default (elite 2, shape_crossover 'block') and tweaked (elite 4, shape_crossover 'uniform').
Outputs per (variant, seed): next-generation genome_norm rows (JSON floats round-trip exactly), sha256 of the float64
little-endian bytes (row-major), sha256 of the final RNG state, and a per-child trace (parents i, j = ranks; crossover
draws; True = from parent a = ranked[i]) replayed from ga.py's own building blocks and asserted equal to the real call.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True       # never write __pycache__ into genome/ (read-only)
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TEAM = os.path.dirname(PKG)
if TEAM not in sys.path:
    sys.path.insert(0, TEAM)

from evolution import batch, ga, sim  # noqa: E402

OUT = os.path.join(HERE, "tweaked_preset_crosscheck.json")
SMOKE = os.path.join(PKG, "runs", "phase3b1r1-smoke-s1", "genomes.jsonl")
GENOME_DIR = os.path.join(TEAM, "genome")
SEEDS = list(range(5))
POP = 16
VARIANTS = {"default": {"elite": 2, "shape_crossover": "block"},
            "tweaked": {"elite": 4, "shape_crossover": "uniform"}}


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype="<f8")).tobytes()).hexdigest()


def rng_sha(rng) -> str:
    return hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()


def inputs():
    rows = [json.loads(x) for x in open(SMOKE)]
    rows = sorted((r for r in rows if r["aircraft"] == "T38" and r["generation"] == 4), key=lambda r: r["rank"])
    assert [r["rank"] for r in rows] == list(range(POP))
    ranked = np.array([r["genome_norm"] for r in rows], dtype=np.float64)
    with open(os.path.join(PKG, "configs", "phase3b1_pilot.json")) as f:
        cfg = batch.resolve_config(json.load(f), "phase3b1_pilot")
    ac = next(a for a in cfg["aircraft"] if a["name"] == "T38")
    sch, groups = batch.full_schema(sim.Profile.from_dict(ac["resolved_profile"]), cfg["struct_genes"],
                                    cfg["struct_asymmetric"], cfg["genome_kind"])
    spec = batch.shape_spec(sch, groups, cfg["shape_ops"])
    blocks = batch.gene_blocks(groups)
    names = [g.name for g in sch]
    assert names == list(rows[0]["genome"]), "schema order != genomes.jsonl gene order"
    return rows, ranked, cfg, spec, blocks, names


def gacfg(cfg, variant):
    d = {k: v for k, v in cfg["ga"].items() if k != "generations"}
    d.update(pop_size=POP, **VARIANTS[variant])
    return ga.GAConfig(**d)


def traced(rng, ranked, gcfg, blocks, spec):
    """Replay of ga.next_generation_blocks from its building blocks, recording parents and crossover draws."""
    n = len(ranked)
    new, trace = [ranked[k].copy() for k in range(min(gcfg.elite, n))], []
    while len(new) < gcfg.pop_size:
        i = ga.flat_rank_select(rng, n, gcfg.selection_p)
        j = i
        while j == i:
            j = ga.flat_rank_select(rng, n, gcfg.selection_p)
        k = len(blocks) - 1 + len(spec.idx) if gcfg.shape_crossover == "uniform" else len(blocks)
        draws = rng.bit_generator.state
        peek = np.random.default_rng()
        peek.bit_generator.state = draws
        u = peek.random(k)
        child = (ga.crossover_blocks_uniform_shape(rng, ranked[i], ranked[j], blocks, spec.idx)
                 if gcfg.shape_crossover == "uniform" else ga.crossover_blocks(rng, ranked[i], ranked[j], blocks))
        assert rng.bit_generator.state == peek.bit_generator.state     # crossover consumed exactly k doubles
        trace.append({"child": len(new), "i": int(i), "j": int(j), "xo_draws": [float(x) for x in u],
                      "from_a": [bool(x < 0.5) for x in u]})
        new.append(ga.mutate_blocks(rng, child, gcfg, spec))
    return np.array(new), trace


def build():
    rows, ranked, cfg, spec, blocks, names = inputs()
    out = {"schema": "tweaked-preset-crosscheck/1",
           "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "generator": "evolution/analysis/tweaked_preset_crosscheck.py", "numpy": np.__version__,
           "preset_names": {"evolution": "ga.elite 4 + ga.shape_crossover 'uniform' (configs/phase3b1_pilot_tweaked.json)",
                            "genome": "phase3_b1_x (genome/presets/phase3_b1_x.json)"},
           "spec": "evolution/analysis/TWEAKED_PRESET_SPEC.md",
           "inputs": {"source": "evolution/runs/phase3b1r1-smoke-s1/genomes.jsonl", "aircraft": "T38", "generation": 4,
                      "order": "rank ascending (= stable argsort of cost, the ranking the loop passes to next_generation)",
                      "individual_ids": [r["individual_id"] for r in rows], "costs": [r["cost"] for r in rows],
                      "gene_names": names,
                      "blocks": dict(zip(("controller", "structure", "shape"), blocks)),
                      "shape_spec": {"idx": spec.idx, "lo": spec.lo, "hi": spec.hi, "log": spec.log, "default": spec.default,
                                     "mut_sigma_frac": spec.mut_sigma_frac, "mutation_rate": spec.mutation_rate},
                      "ranked_population": ranked.tolist(), "ranked_population_sha256": sha(ranked),
                      "rng": "numpy.random.default_rng(seed) (PCG64), one next_generation_blocks call per (variant, seed)",
                      "seeds": SEEDS},
           "variants": {}, "results": {}}
    for v in VARIANTS:
        g = gacfg(cfg, v)
        out["variants"][v] = {k: getattr(g, k) for k in ("pop_size", "elite", "selection_p", "mutation_rate",
                                                          "mutation_sigma", "mutation_mode", "shape_crossover")}
        out["results"][v] = {}
        for s in SEEDS:
            rng = np.random.default_rng(s)
            new = ga.next_generation_blocks(rng, ranked, g, blocks, spec)
            rng2 = np.random.default_rng(s)
            new2, trace = traced(rng2, ranked, g, blocks, spec)
            assert np.array_equal(new, new2) and rng.bit_generator.state == rng2.bit_generator.state
            out["results"][v][f"seed{s}"] = {"next_generation": new.tolist(), "sha256": sha(new), "rng_after_sha256": rng_sha(rng),
                                             "trace": trace}
    return out


def _genome_block_ops():
    if GENOME_DIR not in sys.path:
        sys.path.insert(0, GENOME_DIR)
    os.environ.setdefault("FLIGHT_SIM_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "flight_sim"))  # repo flight_sim/
    bo = importlib.import_module("block_ops")
    from flightsim_path import orig_ga   # genome's own loader of the unmodified flight_sim ga.py
    return bo, orig_ga()


def genome_check(fx):
    """Run genome/block_ops.py (read-only) on the fixture inputs; compare bit for bit. Returns the record."""
    rec = {"checked": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    try:
        bo, oga = _genome_block_ops()
    except Exception as e:     # noqa: BLE001
        return {**rec, "status": f"genome block_ops not importable: {e!r}"}
    src = open(bo.__file__).read()
    rec["block_ops_sha256"] = hashlib.sha256(src.encode()).hexdigest()
    I = fx["inputs"]
    ranked = np.array(I["ranked_population"], dtype=np.float64)
    sp = I["shape_spec"]
    sops = bo.ShapeOps(idx=sp["idx"], lo=sp["lo"], hi=sp["hi"], log=sp["log"], default=sp["default"],
                       mut_sigma_frac=sp["mut_sigma_frac"], mutation_rate=sp["mutation_rate"])
    ops = {"crossover": "blocks", "shape_ops": sops, "blocks": dict(I["blocks"])}
    for v, gv in fx["variants"].items():
        cfg = ga.GAConfig(**gv)      # Genome's next_generation reads cfg.elite / pop_size / selection_p / mutation_*
        if gv["shape_crossover"] == "block":
            call, entry = (lambda rng, tr: bo.next_generation(oga, rng, ranked, cfg, ops, trace=tr)), "block_ops.next_generation"
        elif hasattr(bo, "next_generation_tweaked"):
            call = lambda rng, tr, sx=gv["shape_crossover"]: bo.next_generation_tweaked(oga, rng, ranked, cfg, ops,   # noqa: E731
                                                                                       shape_crossover=sx, trace=tr)
            entry = "block_ops.next_generation_tweaked"
        else:
            rec[v] = {"status": "not implemented in genome/block_ops.py yet (no next_generation_tweaked)"}
            continue
        res = {}
        for s in I["seeds"]:
            rng, tr = np.random.default_rng(s), []
            try:
                mine = call(rng, tr)
            except Exception as e:   # noqa: BLE001
                res[f"seed{s}"] = f"error: {e!r}"
                continue
            ref = fx["results"][v][f"seed{s}"]
            # Genome trace -> (i, j, from_a): 'blk' (3 block draws) or ctrl_a, struct_a, shape_mask_a (8 draws)
            gtr = [(t["i"], t["j"], t["blk"] if "blk" in t else [t["ctrl_a"], t["struct_a"], *t["shape_mask_a"]]) for t in tr]
            etr = [(t["i"], t["j"], t["from_a"]) for t in ref["trace"]]
            res[f"seed{s}"] = {"bit_identical": sha(mine) == ref["sha256"] and np.array_equal(mine, np.array(ref["next_generation"])),
                               "rng_end_identical": rng_sha(rng) == ref["rng_after_sha256"],
                               "trace_identical": gtr == etr, "genome_sha256": sha(mine)}
        ok = all(isinstance(x, dict) and x["bit_identical"] and x["rng_end_identical"] and x["trace_identical"]
                 for x in res.values())
        rec[v] = {"status": "BIT-IDENTICAL" if ok else "MISMATCH", "entry": entry, "per_seed": res}
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    fx = build()
    if a.genome:
        fx["genome_crosscheck"] = genome_check(fx)
    else:
        fx["genome_crosscheck"] = {"status": "not run (use --genome)"}
    with open(a.out, "w") as f:
        json.dump(fx, f, indent=1)
        f.write("\n")
    print(f"wrote {a.out}")
    for v in fx["variants"]:
        print(v, {s: r["sha256"][:12] for s, r in fx["results"][v].items()})
    print("genome:", json.dumps({k: (v.get("status") if isinstance(v, dict) else v)
                                 for k, v in fx["genome_crosscheck"].items()}))


if __name__ == "__main__":
    main()
