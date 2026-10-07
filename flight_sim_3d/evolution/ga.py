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


# ---------------------------------------------------------------------------------------------------------------------
# P3-B1 block operators (genome kind 'phase3_b1': controller | structure | shape). Added for Phase 3; the functions above
# are untouched, so every earlier genome kind / run is bit-identical. Spec: Genome Architect (2026-10-06 steering):
#   * crossover swaps WHOLE blocks (controller | structure | shape), each block from parent a or b with p = 0.5;
#   * controller / structure mutation = the existing clipped Gaussian above (mutation_rate, mutation_sigma, normalized);
#   * shape mutation = clipped Gaussian in FD's ENCODED space: the three wing_chord_taper_* genes are log-scale (mutate
#     ln(x), clip to [ln lo, ln hi]); twist / sweep linear (clip to [lo, hi]); sigma = 0.25 x half-range in that space;
#   * generation 0 for the shape block: identity (FD baseline shape) + the same clipped Gaussian (sigma 0.25 x half-range).
# Storage stays FD's linear [0,1] vector (planform_b1.decode_shape_b1: x = lo + u (hi - lo)); the log transform is only
# applied inside the operator, so a genome_norm slice is always a valid FD shape vector.
@dataclass
class ShapeSpec:
    idx: List[int]                 # positions of the shape genes in the genome (contiguous tail block)
    lo: List[float]
    hi: List[float]
    log: List[bool]                # True: mutate ln(x) (FD chord tapers), False: linear
    default: List[float]           # FD defaults (identity planform)
    init_sigma_frac: float = 0.25  # x half-range in encoded space
    mut_sigma_frac: float = 0.25
    mutation_rate: float = 0.15    # per shape gene

    def sigma(self, frac: float) -> np.ndarray:
        return np.array([frac * (0.5 * (np.log(h) - np.log(l)) if lg else 0.5 * (h - l))
                         for l, h, lg in zip(self.lo, self.hi, self.log)])

    def identity_u(self) -> np.ndarray:
        return np.array([(d - l) / (h - l) for d, l, h in zip(self.default, self.lo, self.hi)])


def shape_perturb(u: np.ndarray, z: np.ndarray, spec: ShapeSpec, sigma: np.ndarray) -> np.ndarray:
    """u (..., n_shape) normalized FD values, z standard normals of the same shape -> clipped Gaussian step in encoded
    space (ln x for log genes, x for linear), returned as normalized u in [0, 1]."""
    lo, hi = np.asarray(spec.lo, float), np.asarray(spec.hi, float)
    lg = np.asarray(spec.log, bool)
    x = lo + np.asarray(u, float) * (hi - lo)
    one = np.ones_like(lo)
    ln = np.clip(np.log(np.where(lg, x, 1.0)) + sigma * z, np.log(np.where(lg, lo, one)), np.log(np.where(lg, hi, one)))
    out = np.where(lg, np.exp(ln), np.clip(x + sigma * z, lo, hi))
    return np.clip((out - lo) / (hi - lo), 0.0, 1.0)


def shape_generation_zero(rng: np.random.Generator, pop_size: int, spec: ShapeSpec) -> np.ndarray:
    """(pop, n_shape): identity planform + clipped Gaussian (sigma = init_sigma_frac x half-range, encoded space)."""
    z = rng.standard_normal((pop_size, len(spec.idx)))
    u0 = np.broadcast_to(spec.identity_u(), z.shape)
    return shape_perturb(u0, z, spec, spec.sigma(spec.init_sigma_frac))


def crossover_blocks(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, blocks: Sequence[Sequence[int]]) -> np.ndarray:
    """Whole-block swap: each block (index list) comes from a or b with p = 0.5; no cut inside a block."""
    pick_a = rng.random(len(blocks)) < 0.5
    child = a.copy()
    for blk, pa in zip(blocks, pick_a):
        if not pa:
            child[list(blk)] = b[list(blk)]
    return child


def mutate_blocks(rng: np.random.Generator, g: np.ndarray, cfg: GAConfig, spec: ShapeSpec) -> np.ndarray:
    """Controller / structure genes: the existing clipped Gaussian (cfg). Shape genes: shape_perturb at
    spec.mutation_rate with sigma = mut_sigma_frac x half-range in encoded space (log for the chord tapers)."""
    if cfg.mutation_mode != "gauss":
        raise ValueError("phase3_b1 block mutation supports mutation_mode 'gauss' only")
    g = g.copy()
    n = g.shape[0]
    sh = np.zeros(n, bool)
    sh[spec.idx] = True
    hit = rng.random(n) < np.where(sh, spec.mutation_rate, cfg.mutation_rate)
    other = hit & ~sh
    g[other] += rng.normal(0.0, cfg.mutation_sigma, int(other.sum()))
    g = np.clip(g, 0.0, 1.0)
    sh_hit = hit[spec.idx]
    if sh_hit.any():
        z = np.where(sh_hit, rng.standard_normal(len(spec.idx)), 0.0)
        u_new = shape_perturb(g[spec.idx], z, spec, spec.sigma(spec.mut_sigma_frac))
        g[spec.idx] = np.where(sh_hit, u_new, g[spec.idx])     # genes not hit keep their exact bits
    return g


def next_generation_blocks(rng: np.random.Generator, ranked: np.ndarray, cfg: GAConfig,
                           blocks: Sequence[Sequence[int]], spec: ShapeSpec) -> np.ndarray:
    """next_generation with per-block crossover and block mutation (elites copied unchanged, parent 2 != parent 1)."""
    n = ranked.shape[0]
    new: List[np.ndarray] = [ranked[i].copy() for i in range(min(cfg.elite, n))]
    while len(new) < cfg.pop_size:
        i = flat_rank_select(rng, n, cfg.selection_p)
        j = i
        while j == i:
            j = flat_rank_select(rng, n, cfg.selection_p)
        child = crossover_blocks(rng, ranked[i], ranked[j], blocks)
        new.append(mutate_blocks(rng, child, cfg, spec))
    return np.array(new)
