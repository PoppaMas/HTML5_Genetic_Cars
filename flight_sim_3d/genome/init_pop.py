"""Generation-0 seeding for tasks with an ``init`` config (Phase 2: phase2_flex).

FD reports that about a third of uniformly random flex-v2 structure genomes fail the flutter gate (INTERFACE_v2.md
§10.2), by design: the baseline sits only 1.16-1.31 V_D from flutter. Seeding generation 0 near the baseline keeps
those evaluations useful without narrowing the search space (mutation and crossover still reach the full ranges).

  * The first draw is exactly ``ga.generation_zero``'s (``rng.random((pop, n))``), so every gene not in a seeded block
    (the controller genes) is initialised exactly as the current presets do.
  * mode "baseline": genes of ``init.blocks`` (default structure_v2) are replaced by encode(baseline) + N(0, sigma)
    (normalized units, clipped to [0, 1]); FD's schema baseline is 1.0 for every multiplier and 0.02 for damping.
  * seed_runs: best genomes of earlier runs (e.g. the v4 bests) are copied into the first individuals for the genes
    they carry; their seeded-block genes sit exactly at the baseline.

Without an ``init`` config nothing is patched and evolve.py's own uniform draw is used (bit-identical runs).
"""
from __future__ import annotations

import json
import os
import types
from typing import Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def baseline_u(spec, blocks):
    idx = [j for j, g in enumerate(spec.genes) if g.block in blocks]
    return idx, np.array([spec.genes[j].encode(spec.genes[j].default) for j in idx], float)


def generation_zero(rng: np.random.Generator, pop_size: int, n_genes: int, spec, init: Dict) -> np.ndarray:
    if n_genes != spec.n_genes:
        raise ValueError(f"generation_zero: {n_genes} genes requested, task spec has {spec.n_genes}")
    pop = rng.random((pop_size, n_genes))                      # == ga.generation_zero
    idx, base = baseline_u(spec, init.get("blocks", []))
    if init.get("mode") == "baseline" and idx:
        noise = rng.standard_normal((pop_size, len(idx)))
        pop[:, idx] = np.clip(base + float(init.get("sigma", 0.05)) * noise, 0.0, 1.0)
    for i, run in enumerate(init.get("seed_runs", [])[:pop_size]):
        path = run if os.path.isabs(run) else os.path.join(HERE, run)
        gains = json.load(open(os.path.join(path, "best_gains.json")))["gains"]
        for j, g in enumerate(spec.genes):
            if j in idx:
                pop[i, j] = base[idx.index(j)]
            elif g.name in gains:
                pop[i, j] = g.encode(float(gains[g.name]))
    return pop


def seeded_ga(ga_module, task) -> types.ModuleType:
    """A stand-in for evolve.py's ``ga`` module: everything delegates to the original, except generation_zero."""
    m = types.ModuleType("ga")
    m.__dict__.update({k: v for k, v in vars(ga_module).items() if not k.startswith("__")})
    m.__doc__ = f"ga with seeded generation 0 for task {task.name!r}"
    m.generation_zero = lambda rng, pop_size, n_genes: generation_zero(rng, pop_size, n_genes, task.spec, task.init)
    return m
