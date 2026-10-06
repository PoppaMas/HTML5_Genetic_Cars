"""Generational GA operators, ported from the car GA in src/app.js.

Vendored unchanged from PoppaMas/HTML5_Genetic_Cars@flight-sim-prototype flight_sim/ga.py.

Correspondence with the original:

* ``generation_zero``      <- manageRound.generationZero / createInstance.createGenerationZero
                              (every gene drawn uniformly from [0, 1])
* ``flat_rank_select``     <- flatRankSelect (geometric rank selection, p = 0.2)
* ``crossover``            <- createInstance.createCrossBreed + pickParent, but
                              uniform or BLX-alpha instead of the two-point
                              chooser, and the gene count comes from the schema
* ``mutate``               <- random.mutateReplace, but with small Gaussian
                              steps by default (``mode="reset"`` reproduces the
                              original full U[0,1] redraw)
* ``next_generation``      <- manageRound.nextGeneration (elites copied
                              unchanged, parent 2 redrawn until != parent 1)

All randomness comes from one ``numpy.random.Generator`` so a run is
reproducible for a given seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class GAConfig:
    pop_size: int = 40
    elite: int = 2
    selection_p: float = 0.2      # flatRankSelect's per-rank pick probability
    crossover: str = "uniform"    # "uniform" or "blx"
    blx_alpha: float = 0.3
    mutation_rate: float = 0.15   # per-gene probability
    mutation_sigma: float = 0.08  # Gaussian step in normalized gene units
    mutation_mode: str = "gauss"  # "gauss" or "reset" (original behaviour)


def generation_zero(rng: np.random.Generator, pop_size: int, n_genes: int) -> np.ndarray:
    return rng.random((pop_size, n_genes))


def flat_rank_select(rng: np.random.Generator, n: int, p: float = 0.2) -> int:
    """Walk ranks from the best; pick rank k with probability p at each step.

    Falls back to a uniform pick if no rank was chosen (same as src/app.js).
    Works on ranks only, so the scale of the fitness values does not matter.
    """
    for k in range(n):
        if rng.random() <= p:
            return k
    return int(rng.integers(n))


def crossover(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, cfg: GAConfig) -> np.ndarray:
    if cfg.crossover == "uniform":
        mask = rng.random(a.shape[0]) < 0.5
        return np.where(mask, a, b)
    if cfg.crossover == "blx":
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        span = hi - lo
        child = rng.uniform(lo - cfg.blx_alpha * span, hi + cfg.blx_alpha * span)
        return np.clip(child, 0.0, 1.0)
    raise ValueError(f"unknown crossover {cfg.crossover!r}")


def mutate(rng: np.random.Generator, g: np.ndarray, cfg: GAConfig) -> np.ndarray:
    g = g.copy()
    hit = rng.random(g.shape[0]) < cfg.mutation_rate
    if cfg.mutation_mode == "gauss":
        g[hit] += rng.normal(0.0, cfg.mutation_sigma, int(hit.sum()))
    elif cfg.mutation_mode == "reset":
        g[hit] = rng.random(int(hit.sum()))
    else:
        raise ValueError(f"unknown mutation mode {cfg.mutation_mode!r}")
    return np.clip(g, 0.0, 1.0)


def next_generation(rng: np.random.Generator, ranked: np.ndarray, cfg: GAConfig) -> np.ndarray:
    """``ranked`` is the current population sorted best-first (lower cost first)."""
    n = ranked.shape[0]
    new: List[np.ndarray] = [ranked[i].copy() for i in range(min(cfg.elite, n))]
    while len(new) < cfg.pop_size:
        i = flat_rank_select(rng, n, cfg.selection_p)
        j = i
        while j == i:
            j = flat_rank_select(rng, n, cfg.selection_p)
        child = crossover(rng, ranked[i], ranked[j], cfg)
        new.append(mutate(rng, child, cfg))
    return np.array(new)


def rank_order(costs: Sequence[float]) -> np.ndarray:
    """Indices sorting costs ascending; stable so ties keep population order."""
    return np.argsort(np.asarray(costs, dtype=float), kind="stable")
