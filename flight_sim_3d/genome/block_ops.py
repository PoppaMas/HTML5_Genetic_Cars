"""Per-block GA operators for phase3_b1 / phase3_b2a (opt-in via a task's ``operators`` config). Other presets never
reach this module: run_evolve.py swaps it in only when ``task.operators`` is set, so evolve.py / ga.py are unchanged for them.

Mirrors Evolution Runner's evolution/ga.py P3-B1 operators exactly (same formulas, same RNG draw order), so a genome
run and an ER run with the same seed draw the same stream (tests compare against ER's functions read-only):

Blocks: controller (every gene not structure_v2 / shape_b1 / shape_b2) | structure (structure_v2) | shape
(shape_b1 or shape_b2, the contiguous tail, FD order). Storage of every gene is normalized u in [0, 1]; shape genes use FD's own linear vector
encoding (x = lo + u (hi - lo)), so the shape slice is always a valid FD [0,1]^6 vector.

* Generation 0: ``rng.random((pop, n_cs))`` over controller + structure (= ga.generation_zero), structure seeded at
  FD's baseline + N(0, init.sigma) clipped (= init_pop / ER seed_generation_zero), THEN the shape block: identity planform
  + clipped Gaussian, sigma = init_sigma_half_range x half-range in the operator space (``standard_normal((pop, 6))``).
* Operator space for the shape init / mutation (internal only): ln(x) for ``log_genes`` (default: the 3
  wing_chord_taper_* genes), x for the rest (twist, sweep); clip to [ln lo, ln hi] / [lo, hi]. sigma = 0.25 x
  half-range there (twist/sweep: 0.125 in u). The genes themselves are LINEAR in value for all 6 (FD r1
  ``planform_b1.GENE_ENCODING``: x = lo + u (hi - lo)); FD's "log" is the spanwise chord interpolation (log-PCHIP), not
  the gene encoding. Results go back to FD as linear u.
* Crossover ``blocks`` (``ga.shape_crossover`` 'block', the default): each WHOLE block from parent a or b,
  ``rng.random(3) < 0.5`` (True = a), no intra-block cut.
* ``ga.shape_crossover`` 'uniform' (phase3_b1_x; spec locked with ER 2026-10-06 ~20:07 PT): per child, after parent
  selection, ``d = rng.random(8)`` (= 8 scalar rng.random() calls: controller, structure, then the 6 shape genes in gene
  order); ``d < 0.5`` -> parent A (ranked[i]). Controller and structure stay whole blocks; each shape gene is drawn on its
  own. This REPLACES the shape block's whole-block draw (no unused draw). Mutation unchanged, after crossover.
* Mutation: ``hit = rng.random(n) < rate`` (shape genes: shape mutation_rate, default = evolve.py's --mutation-rate;
  others: --mutation-rate); controller/structure hits += N(0, --mutation-sigma) (one rng.normal over their hits) and
  clip; if any shape gene was hit, draw ``standard_normal(6)`` and apply the shape perturbation to the hit genes only
  (non-hit shape genes keep their exact bits).
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

BLOCK_ORDER = ("controller", "structure", "shape")
_BLOCK_OF = {"structure_v2": "structure", "shape_b1": "shape", "shape_b2": "shape"}
CHORD_LOG_GENES = ("wing_chord_taper_1", "wing_chord_taper_2", "wing_chord_taper_3")
SHAPE_OPS_DEFAULT = {"init": "identity", "crossover": "blocks", "log_genes": list(CHORD_LOG_GENES),
                     "init_sigma_half_range": 0.25, "mutation_sigma_half_range": 0.25, "mutation_rate": None}
# Preset 'ga' block (mirrors ER's batch-config GA options ga.elite / ga.shape_crossover). Absent = today's behaviour.
SHAPE_CROSSOVER_MODES = ("block", "uniform")
GA_OVERRIDE_KEYS = ("elite", "shape_crossover")


def resolve_ga(cfg: Optional[Dict]) -> Dict:
    """Preset ``ga`` block -> {'elite': int | None, 'shape_crossover': 'block' | 'uniform'}. None / {} -> no elite override
    (evolve.py CLI / default) and shape_crossover 'block' (phase3_b1 exactly)."""
    cfg = {k: v for k, v in dict(cfg or {}).items() if not k.startswith("_")}
    bad = set(cfg) - set(GA_OVERRIDE_KEYS)
    if bad:
        raise ValueError(f"unknown ga keys {sorted(bad)} (allowed: {list(GA_OVERRIDE_KEYS)})")
    elite = cfg.get("elite")
    if elite is not None and (isinstance(elite, bool) or not isinstance(elite, int) or elite < 0):
        raise ValueError(f"ga.elite must be a non-negative int, got {elite!r}")
    mode = cfg.get("shape_crossover", "block")
    if mode not in SHAPE_CROSSOVER_MODES:
        raise ValueError(f"ga.shape_crossover must be one of {SHAPE_CROSSOVER_MODES}, got {mode!r}")
    return {"elite": elite, "shape_crossover": mode}


def gene_block(g) -> str:
    return _BLOCK_OF.get(g.block, "controller")


@dataclass
class ShapeOps:
    idx: List[int]
    lo: List[float]
    hi: List[float]
    log: List[bool]
    default: List[float]
    init_sigma_frac: float = 0.25
    mut_sigma_frac: float = 0.25
    mutation_rate: Optional[float] = None    # None = the GA's mutation_rate

    def sigma(self, frac: float) -> np.ndarray:
        return np.array([frac * (0.5 * (np.log(h) - np.log(l)) if lg else 0.5 * (h - l))
                         for l, h, lg in zip(self.lo, self.hi, self.log)])

    def identity_u(self) -> np.ndarray:
        return np.array([(d - l) / (h - l) for d, l, h in zip(self.default, self.lo, self.hi)])


def resolve_operators(cfg: Optional[Dict], spec, shape_crossover: str = "block") -> Dict:
    """{} when the task has no operators config. Keys mirror ER's batch config: crossover 'blocks'; shape = ER shape_ops
    (init 'identity', log_genes, init_sigma_half_range, mutation_sigma_half_range, mutation_rate). ``shape_crossover``
    comes from the preset's ``ga`` block (resolve_ga): 'block' (default, phase3_b1) or 'uniform' (phase3_b1_x)."""
    if shape_crossover not in SHAPE_CROSSOVER_MODES:
        raise ValueError(f"shape_crossover must be one of {SHAPE_CROSSOVER_MODES}, got {shape_crossover!r}")
    if not cfg:
        return {}
    cfg = {k: v for k, v in dict(cfg).items() if not k.startswith("_")}
    bad = set(cfg) - {"crossover", "shape"}
    if bad:
        raise ValueError(f"unknown operators keys {sorted(bad)}")
    if cfg.get("crossover", "blocks") != "blocks":
        raise ValueError(f"operators.crossover must be 'blocks' (whole-block swap), got {cfg.get('crossover')!r}")
    ops = dict(SHAPE_OPS_DEFAULT, **{k: v for k, v in dict(cfg.get("shape") or {}).items() if not k.startswith("_")})
    bad = set(ops) - set(SHAPE_OPS_DEFAULT)
    if bad:
        raise ValueError(f"operators.shape: unknown keys {sorted(bad)}")
    if ops["init"] != "identity" or ops["crossover"] != "blocks":
        raise ValueError("operators.shape: init 'identity' and crossover 'blocks' are the only options")
    for k in ("init_sigma_half_range", "mutation_sigma_half_range"):
        if not 0.0 < float(ops[k]) <= 1.0:
            raise ValueError(f"operators.shape.{k} must be in (0, 1] (fraction of the half-range in the operator space)")
    if ops["mutation_rate"] is not None and not 0.0 <= float(ops["mutation_rate"]) <= 1.0:
        raise ValueError("operators.shape.mutation_rate must be in [0, 1] or null")
    groups = [gene_block(g) for g in spec.genes]
    idx = [j for j, b in enumerate(groups) if b == "shape"]
    if not idx or idx != list(range(len(groups) - len(idx), len(groups))):
        raise ValueError("shape genes must be the contiguous tail block of the genome")
    sch = [spec.genes[j] for j in idx]
    names = {g.name for g in sch}
    if set(ops["log_genes"]) - names:
        raise ValueError(f"operators.shape.log_genes: unknown shape genes {sorted(set(ops['log_genes']) - names)}")
    if any(g.scale != "linear" for g in sch):
        raise ValueError("shape genes must be stored in FD's linear encoding (log lives in the operator)")
    sops = ShapeOps(idx=idx, lo=[g.min for g in sch], hi=[g.max for g in sch], log=[g.name in ops["log_genes"] for g in sch],
                    default=[g.default for g in sch], init_sigma_frac=float(ops["init_sigma_half_range"]),
                    mut_sigma_frac=float(ops["mutation_sigma_half_range"]),
                    mutation_rate=None if ops["mutation_rate"] is None else float(ops["mutation_rate"]))
    blocks = {b: [j for j, x in enumerate(groups) if x == b] for b in BLOCK_ORDER}
    blocks = {b: v for b, v in blocks.items() if v}
    if shape_crossover == "uniform" and list(blocks) != list(BLOCK_ORDER):
        raise ValueError("ga.shape_crossover 'uniform' needs all three blocks controller | structure | shape")
    return {"crossover": "blocks", "shape": ops, "shape_ops": sops, "blocks": blocks, "shape_crossover": shape_crossover}


# -------------------------------------------------------------------------------------------------- operators
def shape_perturb(u: np.ndarray, z: np.ndarray, sp: ShapeOps, sigma: np.ndarray) -> np.ndarray:
    """u (..., 6) normalized FD values, z standard normals -> clipped Gaussian step in the operator space (ln x for log_genes), back to linear u."""
    lo, hi = np.asarray(sp.lo, float), np.asarray(sp.hi, float)
    lg = np.asarray(sp.log, bool)
    x = lo + np.asarray(u, float) * (hi - lo)
    one = np.ones_like(lo)
    ln = np.clip(np.log(np.where(lg, x, 1.0)) + sigma * z, np.log(np.where(lg, lo, one)), np.log(np.where(lg, hi, one)))
    out = np.where(lg, np.exp(ln), np.clip(x + sigma * z, lo, hi))
    return np.clip((out - lo) / (hi - lo), 0.0, 1.0)


def shape_generation_zero(rng: np.random.Generator, pop_size: int, sp: ShapeOps) -> np.ndarray:
    z = rng.standard_normal((pop_size, len(sp.idx)))
    return shape_perturb(np.broadcast_to(sp.identity_u(), z.shape), z, sp, sp.sigma(sp.init_sigma_frac))


def generation_zero(rng: np.random.Generator, pop_size: int, n_genes: int, spec, init: Dict, ops: Dict) -> np.ndarray:
    """Controller + structure exactly as init_pop.generation_zero over the first n_cs genes, then the shape block."""
    sp = ops["shape_ops"]
    if n_genes != spec.n_genes:
        raise ValueError(f"generation_zero: {n_genes} genes requested, task spec has {spec.n_genes}")
    n_cs = sp.idx[0]
    pop = rng.random((pop_size, n_cs))
    if (init or {}).get("mode") == "baseline":
        idx = [j for j in range(n_cs) if spec.genes[j].block in init.get("blocks", [])]
        if idx:
            base = np.array([spec.genes[j].encode(spec.genes[j].default) for j in idx], float)
            noise = rng.standard_normal((pop_size, len(idx)))
            pop[:, idx] = np.clip(base + float(init["sigma"]) * noise, 0.0, 1.0)
    return np.hstack([pop, shape_generation_zero(rng, pop_size, sp)])


def block_crossover(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, ops: Dict, rec: Optional[Dict] = None) -> np.ndarray:
    """Dispatch on ops['shape_crossover']: 'block' (default; whole blocks, rng.random(3)) or 'uniform'
    (crossover_shape_uniform). ``rec`` (optional dict) receives the draws; it never touches the RNG."""
    if ops.get("shape_crossover", "block") == "uniform":
        return crossover_shape_uniform(rng, a, b, ops, rec)
    pick_a = rng.random(len(ops["blocks"])) < 0.5
    if rec is not None:
        rec["blk"] = [bool(x) for x in pick_a]
    child = a.copy()
    for idx, pa in zip(ops["blocks"].values(), pick_a):
        if not pa:
            child[idx] = b[idx]
    return child


def crossover_shape_uniform(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, ops: Dict,
                            rec: Optional[Dict] = None) -> np.ndarray:
    """ga.shape_crossover 'uniform' (locked with ER): d = rng.random(2 + n_shape) = rng.random(8) for B1 (controller,
    structure, then the shape genes in gene order). d < 0.5 -> parent a. Controller / structure whole; shape per gene.
    Replaces the shape block's whole-block draw (no unused draw)."""
    blk = ops["blocks"]
    ctrl, struct, shape = blk["controller"], blk["structure"], blk["shape"]
    d = rng.random(2 + len(shape))
    pick_a = d < 0.5
    if rec is not None:
        rec["ctrl_a"], rec["struct_a"], rec["shape_mask_a"] = bool(pick_a[0]), bool(pick_a[1]), [bool(x) for x in pick_a[2:]]
    child = a.copy()
    if not pick_a[0]:
        child[ctrl] = b[ctrl]
    if not pick_a[1]:
        child[struct] = b[struct]
    child[shape] = np.where(pick_a[2:], a[shape], b[shape])
    return child


def block_mutate(rng: np.random.Generator, g: np.ndarray, cfg, ops: Dict) -> np.ndarray:
    if cfg.mutation_mode != "gauss":
        raise ValueError("phase3_b1 block mutation supports mutation_mode 'gauss' only")
    sp = ops["shape_ops"]
    rate_sh = cfg.mutation_rate if sp.mutation_rate is None else sp.mutation_rate
    g = g.copy()
    n = g.shape[0]
    sh = np.zeros(n, bool)
    sh[sp.idx] = True
    hit = rng.random(n) < np.where(sh, rate_sh, cfg.mutation_rate)
    other = hit & ~sh
    g[other] += rng.normal(0.0, cfg.mutation_sigma, int(other.sum()))
    g = np.clip(g, 0.0, 1.0)
    sh_hit = hit[sp.idx]
    if sh_hit.any():
        z = np.where(sh_hit, rng.standard_normal(len(sp.idx)), 0.0)
        u_new = shape_perturb(g[sp.idx], z, sp, sp.sigma(sp.mut_sigma_frac))
        g[sp.idx] = np.where(sh_hit, u_new, g[sp.idx])
    return g


def next_generation(ga_module, rng: np.random.Generator, ranked: np.ndarray, cfg, ops: Dict,
                    trace: Optional[List] = None) -> np.ndarray:
    """ga.next_generation (elites = ranked[:cfg.elite] copied unchanged, flat-rank selection of two distinct parents)
    with block crossover (ops['shape_crossover']) / mutation. ``trace`` (optional list) gets one record per child
    (parents + crossover draws); recording never draws from the RNG."""
    n = ranked.shape[0]
    new: List[np.ndarray] = [ranked[i].copy() for i in range(min(cfg.elite, n))]
    while len(new) < cfg.pop_size:
        i = ga_module.flat_rank_select(rng, n, cfg.selection_p)
        j = i
        while j == i:
            j = ga_module.flat_rank_select(rng, n, cfg.selection_p)
        rec = {"i": int(i), "j": int(j)} if trace is not None else None
        new.append(block_mutate(rng, block_crossover(rng, ranked[i], ranked[j], ops, rec), cfg, ops))
        if trace is not None:
            trace.append(rec)
    return np.array(new)


def next_generation_tweaked(ga_module, rng: np.random.Generator, ranked: np.ndarray, cfg, ops: Dict,
                            shape_crossover: str = "uniform", trace: Optional[List] = None) -> np.ndarray:
    """Public entry for the phase3_b1_x operators (ER option names: ga.elite via cfg.elite, ga.shape_crossover):
    next_generation with ops['shape_crossover'] set to ``shape_crossover``."""
    return next_generation(ga_module, rng, ranked, cfg, dict(ops, shape_crossover=shape_crossover), trace)


def block_ga(ga_module, task) -> types.ModuleType:
    """Stand-in for evolve.py's ``ga``: B1 generation 0 + block crossover / mutation; everything else delegates."""
    m = types.ModuleType("ga")
    m.__dict__.update({k: v for k, v in vars(ga_module).items() if not k.startswith("__")})
    m.__doc__ = f"ga with per-block operators for task {task.name!r}"
    ops = task.operators
    m.generation_zero = lambda rng, pop_size, n_genes: generation_zero(rng, pop_size, n_genes, task.spec, task.init, ops)
    m.next_generation = lambda rng, ranked, cfg: next_generation(ga_module, rng, ranked, cfg, ops)
    return m
