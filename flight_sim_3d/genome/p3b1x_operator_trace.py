#!/usr/bin/env python3
"""Fixed-seed GA operator trace for phase3_b1_x, with phase3_b1 and phase2_flex as controls. Nothing flies.

Driver (= evolve.run's loop without the sims): rng = default_rng(seed); pop = ga.generation_zero(rng, pop, n) through the
same ga stand-in run_evolve.py installs (block_ops.block_ga for phase3_b1 / _x, init_pop.seeded_ga for phase2_flex);
then per generation: synthetic cost -> ga.rank_order (stable argsort) -> next_generation(rng, ranked, GAConfig).
Synthetic cost (deterministic, no RNG): cost(g) = sum_k (1 + k/n) (g_k - c_k)^2, c_k = (k * 0.6180339887498949) mod 1.

Recorded: per generation SHA-256 of the ranked population (float64 little-endian, C order) and of the costs, best cost;
the full ranked generation 0 and generation 1; parent picks (i, j) and crossover draws per child (block_ops trace hook,
which never draws from the RNG); elite preservation checks. Controls: genome's phase3_b1 / phase2_flex hashes are compared
with GOLDEN (captured from the code BEFORE the phase3_b1_x change) and re-generated directly with Evolution Runner's
functions (evolution/ga.py + batch.py, read-only). phase3_b1_x vs ER: compared only if ER has a shape_crossover option.

Usage: $PY genome/p3b1x_operator_trace.py   -> genome/runs/p3b1x_operator_trace.json
"""
from __future__ import annotations

import hashlib
import inspect
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
import init_pop  # noqa: E402
from flightsim_path import orig_ga  # noqa: E402

SEED, POP, GENS = 1, 64, 5          # generation 0 + 4 bred generations
PHI = 0.6180339887498949
GOLDEN_FILE = os.path.join(HERE, "runs", "p3b1x_golden_pre_change.json")


def synthetic_cost(pop: np.ndarray) -> np.ndarray:
    n = pop.shape[1]
    k = np.arange(n)
    return (((pop - (k * PHI) % 1.0) ** 2) * (1.0 + k / n)).sum(axis=1)


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def drive(gen0, nextgen, rank_order, n_genes, cfg, elite):
    """Returns per-generation records, ranked gen 0 / gen 1 arrays, and per-generation child traces."""
    rng = np.random.default_rng(SEED)
    pop = gen0(rng, POP, n_genes)
    gens, arrays, traces, elites_ok = [{"gen0_unranked_sha256": sha(pop)}], {}, [], []
    for g in range(GENS):
        cs = synthetic_cost(pop)
        order = rank_order(cs)
        pop, cs = pop[order], cs[order]
        gens.append({"gen": g, "pop_sha256": sha(pop), "cost_sha256": sha(cs), "best_cost": float(cs[0]),
                     "mean_cost": float(cs.mean())})
        if g < 2:
            arrays[f"gen{g}_ranked"] = pop.tolist()
        if g < GENS - 1:
            tr: list = []
            new = nextgen(rng, pop, cfg, tr)
            elites_ok.append(bool(np.array_equal(new[:elite], pop[:elite])))
            traces.append(tr)
            pop = new
    return gens, arrays, traces, elites_ok


def er_modules():
    sys.path.append(os.path.dirname(HERE))
    import shape_b1 as SB
    F = SB.er_modules()[0]
    from evolution import batch, ga as ega
    return ega, batch, F


def er_shape_spec(ega, sp):
    return ega.ShapeSpec(idx=list(sp.idx), lo=list(sp.lo), hi=list(sp.hi), log=list(sp.log), default=list(sp.default),
                         init_sigma_frac=sp.init_sigma_frac, mut_sigma_frac=sp.mut_sigma_frac, mutation_rate=0.15)


def main():
    GA = orig_ga()
    golden = json.load(open(GOLDEN_FILE))
    out = {"method": __doc__.split("\n\nUsage")[0], "seed": SEED, "pop_size": POP, "generations": GENS,
           "synthetic_cost": "sum_k (1 + k/n) (g_k - c_k)^2, c_k = (k * 0.6180339887498949) mod 1",
           "ga_config_common": {"selection_p": 0.2, "mutation_rate": 0.15, "mutation_sigma": 0.08, "mutation_mode": "gauss"},
           "golden_pre_change": golden, "presets": {}}

    # genome: phase3_b1_x (elite 4, shape uniform), phase3_b1 (control, elite 2 = evolve.py default), phase2_flex (control)
    runs = {}
    for name in ("phase3_b1_x", "phase3_b1"):
        t = adapter.load_task(name)
        elite = int(t.ga_overrides.get("elite", 2))
        cfg = GA.GAConfig(pop_size=POP, elite=elite)
        shim = block_ops.block_ga(GA, t)
        nextgen = lambda rng, ranked, c, tr, t=t: block_ops.next_generation(GA, rng, ranked, c, t.operators, tr)
        runs[name] = (t, cfg, drive(shim.generation_zero, nextgen, shim.rank_order, t.spec.n_genes, cfg, elite))
    t2 = adapter.load_task("phase2_flex")
    cfg2 = GA.GAConfig(pop_size=POP)
    shim2 = init_pop.seeded_ga(GA, t2)
    runs["phase2_flex"] = (t2, cfg2, drive(shim2.generation_zero, lambda rng, r, c, tr: shim2.next_generation(rng, r, c),
                                           shim2.rank_order, t2.spec.n_genes, cfg2, cfg2.elite))

    # ER direct re-generation (read-only import)
    ega, batch, F = er_modules()
    er_has_b1 = hasattr(ega, "next_generation_blocks")
    er_src = inspect.getsource(ega) + inspect.getsource(batch)
    er_has_uniform = "shape_crossover" in {f.name for f in __import__("dataclasses").fields(ega.GAConfig)}
    er = {}
    if er_has_b1:
        t = runs["phase3_b1"][0]
        sp, es = t.operators["shape_ops"], er_shape_spec(ega, t.operators["shape_ops"])
        blocks = batch.gene_blocks(["gains"] * 8 + ["struct"] * 12 + ["shape"] * 6)

        def er_gen0_b1(rng, n, _):
            pop = ega.generation_zero(rng, n, 20)
            pop = batch.seed_generation_zero(pop, rng, [None] * 8 + F.struct_schema(), ["gains"] * 8 + ["struct"] * 12,
                                             {"mode": "baseline", "sigma": 0.1})
            return np.hstack([pop, ega.shape_generation_zero(rng, n, es)])
        ecfg = ega.GAConfig(pop_size=POP, elite=2)
        er["phase3_b1"] = drive(er_gen0_b1, lambda rng, r, c, tr: ega.next_generation_blocks(rng, r, c, blocks, es),
                                ega.rank_order, 26, ecfg, 2)

    def er_gen0_p2(rng, n, _):
        pop = ega.generation_zero(rng, n, 20)
        return batch.seed_generation_zero(pop, rng, [None] * 8 + F.struct_schema(), ["gains"] * 8 + ["struct"] * 12,
                                          {"mode": "baseline", "sigma": 0.1})
    er["phase2_flex"] = drive(er_gen0_p2, lambda rng, r, c, tr: ega.next_generation(rng, r, c), ega.rank_order, 20,
                              ega.GAConfig(pop_size=POP), 2)

    if er_has_b1 and er_has_uniform:   # ER's own tweaked operators: GAConfig(shape_crossover='uniform', elite=4)
        ex_ = er_shape_spec(ega, runs["phase3_b1_x"][0].operators["shape_ops"])
        er["phase3_b1_x"] = drive(er_gen0_b1, lambda rng, r, c, tr: ega.next_generation_blocks(rng, r, c, blocks, ex_),
                                  ega.rank_order, 26, ega.GAConfig(pop_size=POP, elite=4, shape_crossover="uniform"), 4)

    # independent reference for phase3_b1_x: ER primitives (flat_rank_select, mutate_blocks) + the locked crossover
    # written out inline from the spec text (not block_ops), to check block_ops against the spec independently
    tx = runs["phase3_b1_x"][0]
    ex = er_shape_spec(ega, tx.operators["shape_ops"])
    bl = tx.operators["blocks"]

    def ref_next_x(rng, ranked, c, tr):
        n = ranked.shape[0]
        new = [ranked[i].copy() for i in range(min(c.elite, n))]
        while len(new) < c.pop_size:
            i = ega.flat_rank_select(rng, n, c.selection_p)
            j = i
            while j == i:
                j = ega.flat_rank_select(rng, n, c.selection_p)
            a, b = ranked[i], ranked[j]
            child = a.copy()
            if not rng.random() < 0.5:          # controller
                child[bl["controller"]] = b[bl["controller"]]
            if not rng.random() < 0.5:          # structure
                child[bl["structure"]] = b[bl["structure"]]
            for k in bl["shape"]:               # 6 shape genes in gene order
                if not rng.random() < 0.5:
                    child[k] = b[k]
            new.append(ega.mutate_blocks(rng, child, c, ex))
        return np.array(new)
    shimx = block_ops.block_ga(GA, tx)
    ref_x = drive(shimx.generation_zero, ref_next_x, ega.rank_order, 26, ega.GAConfig(pop_size=POP, elite=4), 4)

    def hashes(d):
        return [g["pop_sha256"] for g in d[0][1:]]

    for name, (t, cfg, d) in runs.items():
        gens, arrays, traces, elites_ok = d
        rec = {"n_genes": t.spec.n_genes, "elite": cfg.elite,
               "shape_crossover": (t.operators or {}).get("shape_crossover") if getattr(t, "operators", None) else None,
               "generations": gens, "elites_preserved_each_generation": elites_ok, **arrays}
        if traces and traces[0]:
            rec["child_draws"] = [{"gen_from": g, "children": tr} for g, tr in enumerate(traces)]
        if name in golden:
            rec["control_unchanged_vs_pre_change"] = hashes(d) == golden[name]["pop_sha256"] and \
                gens[0]["gen0_unranked_sha256"] == golden[name]["gen0_unranked_sha256"]
        if name in er:
            rec["er_direct"] = {"function": ("evolution.ga.next_generation_blocks + batch.seed_generation_zero + "
                                             "ga.shape_generation_zero" if name == "phase3_b1" else
                                             "evolution.ga.next_generation + batch.seed_generation_zero"),
                                "er_pop_sha256": hashes(er[name]), "bit_identical": hashes(d) == hashes(er[name])}
        if name == "phase3_b1_x":
            sh = t.operators["blocks"]["shape"]
            mixed = sum(1 for tr in traces for c in tr if 0 < sum(c["shape_mask_a"]) < len(sh))
            total = sum(len(tr) for tr in traces)
            rec["shape_mixed_children"] = {"mixed": mixed, "total": total}
            if name in er:
                rec["er_direct"] = {"er_wired": True,
                                    "function": ("evolution.ga.next_generation_blocks with GAConfig(shape_crossover="
                                                 "'uniform', elite=4) (-> crossover_blocks_uniform_shape) + ER gen-0"),
                                    "er_pop_sha256": hashes(er[name]), "bit_identical": hashes(d) == hashes(er[name])}
            else:
                rec["er_direct"] = {"er_wired": False, "bit_identical": None,
                                    "note": "ER evolution/ga.py GAConfig has no shape_crossover option (read-only check)"}
            rec["independent_reference"] = {"what": "ER flat_rank_select + ER mutate_blocks + spec crossover written inline",
                                            "ref_pop_sha256": hashes(ref_x), "bit_identical": hashes(d) == hashes(ref_x)}
        out["presets"][name] = rec

    s = {n: {"control_unchanged_vs_pre_change": r.get("control_unchanged_vs_pre_change"),
             "er_direct_bit_identical": (r.get("er_direct") or {}).get("bit_identical"),
             "elites_preserved": all(r["elites_preserved_each_generation"]),
             "final_pop_sha256": r["generations"][-1]["pop_sha256"]} for n, r in out["presets"].items()}
    s["phase3_b1_x"]["independent_reference_bit_identical"] = out["presets"]["phase3_b1_x"]["independent_reference"]["bit_identical"]
    s["phase3_b1_x"]["er_wired"] = er_has_uniform
    s["phase3_b1_x"]["differs_from_phase3_b1"] = (out["presets"]["phase3_b1_x"]["generations"][2]["pop_sha256"]
                                                  != out["presets"]["phase3_b1"]["generations"][2]["pop_sha256"])
    out["summary"] = s
    dest = os.path.join(HERE, "runs", "p3b1x_operator_trace.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", dest)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
