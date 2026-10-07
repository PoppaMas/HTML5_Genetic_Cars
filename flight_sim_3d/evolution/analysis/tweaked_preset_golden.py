"""Golden next-generation outputs of the GA operators for every existing config (pre-tweak reference).

Run from the team root:  $PY evolution/analysis/tweaked_preset_golden.py [--out PATH] [--check PATH]

Captured with the code BEFORE the opt-in tweaked preset (ga.elite / ga.shape_crossover) was added
(2026-10-06 ~20:10 PT) -> evolution/tests/data/tweaked_preset_golden.json. tests/test_tweaked_preset.py recomputes every
case with the current code and requires bit-identical results (sha256 of the float64 bytes of each population, plus
the final RNG state), i.e. same draws AND same draw order.

Cases (no flight, operators only):
  * ops/next_generation:        ga.next_generation, crossover uniform|blx x mutation gauss|reset x elite 1|2|4, 5 seeds
  * ops/next_generation_blocks: ga.next_generation_blocks (8 | 12 | 6 layout), elite 1..4, 5 seeds
  * config/<file>/<aircraft>:   every configs/*.json and configs/seeds/*.json: resolve_config -> run identity hash +
                                run_id, then batch's gen-0 path for the aircraft (same seed) and 3 generations of
                                ga.next_generation(_blocks) with batch's GAConfig, ranked by a fixed permutation
  * smoke_t38_g4/seed<k>:       runs/phase3b1r1-smoke-s1 T38 gen-4 population (rank order) -> next_generation_blocks
                                with the phase3b1_pilot GA settings, seeds 0..9
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TEAM = os.path.dirname(PKG)
if TEAM not in sys.path:
    sys.path.insert(0, TEAM)

from evolution import batch, ga, sim  # noqa: E402

SMOKE_ROWS = os.path.join(PKG, "runs", "phase3b1r1-smoke-s1", "genomes.jsonl")
TEST_SPEC = dict(idx=list(range(20, 26)), lo=[0.5, 0.5, 0.5, -3.0, -3.0, -5.0], hi=[1.5, 1.5, 1.5, 3.0, 3.0, 5.0],
                 log=[True, True, True, False, False, False], default=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
TEST_BLOCKS = [list(range(0, 8)), list(range(8, 20)), list(range(20, 26))]


def h(a) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def rng_h(rng) -> str:
    return hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()


def config_files():
    return sorted(glob.glob(os.path.join(PKG, "configs", "*.json"))) + sorted(glob.glob(os.path.join(PKG, "configs", "seeds", "*.json")))


def gcfg_of(cfg) -> "ga.GAConfig":
    return ga.GAConfig(**{k: v for k, v in cfg["ga"].items() if k != "generations"})   # = batch.Batch._evolve


def gen0(cfg, ac):
    """batch.Batch._evolve's generation-0 path (fresh run) for one aircraft -> (pop, rng, blocks, sspec)."""
    prof = sim.Profile.from_dict(ac["resolved_profile"])
    kind = cfg.get("genome_kind")
    sch, groups = batch.full_schema(prof, cfg["struct_genes"], cfg["struct_asymmetric"], kind)
    sspec = batch.shape_spec(sch, groups, cfg["shape_ops"]) if kind == "phase3_b1" else None
    blocks = batch.gene_blocks(groups) if kind == "phase3_b1" else None
    rng = np.random.default_rng(ac["seed"])
    P = cfg["ga"]["pop_size"]
    if sspec is None:
        pop = ga.generation_zero(rng, P, len(sch))
        pop = batch.seed_generation_zero(pop, rng, sch, groups, cfg.get("init") or {})
    else:
        n_cs = sspec.idx[0]
        pop = ga.generation_zero(rng, P, n_cs)
        pop = batch.seed_generation_zero(pop, rng, sch[:n_cs], groups[:n_cs], cfg.get("init") or {})
        pop = np.hstack([pop, ga.shape_generation_zero(rng, P, sspec)])
    return pop, rng, blocks, sspec


def step(rng, pop, gcfg, blocks, sspec):
    return ga.next_generation(rng, pop, gcfg) if sspec is None else ga.next_generation_blocks(rng, pop, gcfg, blocks, sspec)


def smoke_t38_g4():
    rows = [json.loads(x) for x in open(SMOKE_ROWS)]
    rows = sorted((r for r in rows if r["aircraft"] == "T38" and r["generation"] == 4), key=lambda r: r["rank"])
    return np.array([r["genome_norm"] for r in rows], dtype=np.float64), rows


def pilot_ops(cfg_name="phase3b1_pilot.json", user_over=None):
    with open(os.path.join(PKG, "configs", cfg_name)) as f:
        u = json.load(f)
    u.update(user_over or {})
    cfg = batch.resolve_config(u, os.path.splitext(cfg_name)[0])
    ac = next(a for a in cfg["aircraft"] if a["name"] == "T38")
    prof = sim.Profile.from_dict(ac["resolved_profile"])
    sch, groups = batch.full_schema(prof, cfg["struct_genes"], cfg["struct_asymmetric"], cfg["genome_kind"])
    return cfg, batch.shape_spec(sch, groups, cfg["shape_ops"]), batch.gene_blocks(groups)


def compute() -> dict:
    out = {}
    for xo in ("uniform", "blx"):
        for mode in ("gauss", "reset"):
            for elite in (1, 2, 4):
                for seed in range(5):
                    cfg = ga.GAConfig(pop_size=16, elite=elite, crossover=xo, mutation_mode=mode)
                    rng = np.random.default_rng(seed)
                    ranked = rng.random((16, 8))
                    new = ga.next_generation(rng, ranked, cfg)
                    out[f"ops/next_generation/{xo}/{mode}/e{elite}/s{seed}"] = {"pop": h(new), "rng": rng_h(rng)}
    spec = ga.ShapeSpec(**TEST_SPEC)
    for elite in (1, 2, 3, 4):
        for seed in range(5):
            cfg = ga.GAConfig(pop_size=16, elite=elite)
            rng = np.random.default_rng(seed)
            ranked = rng.random((16, 26))
            new = ga.next_generation_blocks(rng, ranked, cfg, TEST_BLOCKS, spec)
            out[f"ops/next_generation_blocks/e{elite}/s{seed}"] = {"pop": h(new), "rng": rng_h(rng)}
    for path in config_files():
        rel = os.path.relpath(path, os.path.join(PKG, "configs"))
        with open(path) as f:
            user = json.load(f)
        cfg = batch.resolve_config(user, os.path.splitext(os.path.basename(path))[0])
        ident = hashlib.sha256(json.dumps(batch.identity(cfg), sort_keys=True).encode()).hexdigest()
        out[f"config/{rel}"] = {"identity": ident, "run_id": cfg["run_id"], "ga": cfg["ga"]}
        gcfg = gcfg_of(cfg)
        for ac in cfg["aircraft"]:
            pop, rng, blocks, sspec = gen0(cfg, ac)
            rec = {"gen0": h(pop)}
            perm_rng = np.random.default_rng(1000 + ac["seed"])
            for g in range(1, 4):
                pop = pop[perm_rng.permutation(len(pop))]          # a fixed "ranking"
                pop = step(rng, pop, gcfg, blocks, sspec)
                rec[f"gen{g}"] = h(pop)
            rec["rng"] = rng_h(rng)
            out[f"config/{rel}/{ac['name']}"] = rec
    pop4, _ = smoke_t38_g4()
    cfg, sspec, blocks = pilot_ops()
    gcfg = gcfg_of(cfg)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        new = ga.next_generation_blocks(rng, pop4, gcfg, blocks, sspec)
        out[f"smoke_t38_g4/seed{seed}"] = {"pop": h(new), "rng": rng_h(rng)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--check")
    a = ap.parse_args(argv)
    got = compute()
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"_doc": __doc__.strip().splitlines()[0], "numpy": np.__version__, "cases": got}, f, indent=1, sort_keys=True)
        print(f"wrote {len(got)} cases -> {a.out}")
    if a.check:
        ref = json.load(open(a.check))["cases"]
        bad = [k for k in sorted(set(ref) | set(got)) if ref.get(k) != got.get(k)]
        print(f"{len(ref)} reference cases, {len(got)} computed, {len(bad)} differ")
        for k in bad[:20]:
            print("  DIFF", k)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
