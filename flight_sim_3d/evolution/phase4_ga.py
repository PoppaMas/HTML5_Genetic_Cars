"""Phase 4 genome kind 'phase4_rings' (Evolution side; opt-in, old presets never import this).
Mirrors genome/phase4_rings.py (Genome Architect): 29 genes in whole blocks guidance 12 | inner_loop 11 | mixing 6.
Operators are ga.py's own building blocks, unchanged: elite 2, ga.flat_rank_select p 0.2 (i, then j != i),
ga.crossover_blocks (one rng.random(3), < 0.5 -> parent a = ranked[i]), ga.mutate gauss (rng.random(29) < 0.15,
N(0, 0.08) on hits, clip [0,1]); gen-0 = ga.generation_zero = rng.random((pop, 29)); rank = stable argsort."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from . import ga

# (name, block, lo, hi, scale, default) = genome/phase4_rings.py gene_table() re-synced 2026-10-07 02:47 PT (Genome per FD P4.8:
# bank/nz as fractions of the per-aircraft FD limits, tau_cmd_s 0.05-0.5 default 0.1). Test asserts equality with Genome.
GENES = [
    ("t_preview_s", "guidance", 1.0, 8.0, "log", 3.0), ("w_next_ring", "guidance", 0.0, 0.6, "linear", 0.25),
    ("k_lat", "guidance", 0.1, 5.0, "log", 1.0), ("k_lat_rate", "guidance", 1e-3, 2.0, "log0", 0.0),
    ("k_vert", "guidance", 0.05, 3.0, "log", 0.5), ("k_vert_rate", "guidance", 1e-3, 1.0, "log0", 0.0),
    ("bank_max_frac", "guidance", 0.25, 1.0, "linear", 0.75), ("nz_max_frac", "guidance", 0.4, 1.0, "linear", 0.7),
    ("nz_min_frac", "guidance", 0.0, 0.75, "linear", 0.5), ("gamma_max_deg", "guidance", 5.0, 25.0, "linear", 15.0),
    ("v_cmd_scale", "guidance", 0.85, 1.25, "linear", 1.0), ("v_turn_comp", "guidance", 0.0, 0.15, "linear", 0.0),
    ("kp_roll", "inner_loop", 0.002, 0.5, "log", 0.05), ("ki_roll", "inner_loop", 1e-5, 0.2, "log0", 0.0),
    ("kd_roll", "inner_loop", 1e-4, 0.5, "log", 0.02), ("kp_pitch", "inner_loop", 0.002, 0.5, "log", 0.05),
    ("ki_pitch", "inner_loop", 1e-5, 0.2, "log0", 0.01), ("kd_pitch", "inner_loop", 1e-4, 0.5, "log", 0.02),
    ("kr_yaw", "inner_loop", 1e-4, 0.5, "log0", 0.0), ("kbeta_yaw", "inner_loop", 1e-4, 0.5, "log0", 0.0),
    ("tau_washout", "inner_loop", 0.3, 10.0, "log", 2.0), ("kp_spd", "inner_loop", 0.002, 0.5, "log", 0.05),
    ("ki_spd", "inner_loop", 1e-4, 0.1, "log0", 0.01),
    ("k_ari", "mixing", 0.0, 0.6, "linear", 0.0), ("k_turn_coord", "mixing", 0.0, 1.5, "linear", 0.0),
    ("k_elev_bank", "mixing", 0.0, 0.5, "linear", 0.0), ("ail_auth", "mixing", 0.4, 1.0, "linear", 1.0),
    ("rud_auth", "mixing", 0.2, 1.0, "linear", 1.0), ("tau_cmd_s", "mixing", 0.05, 0.5, "log", 0.1),
]
NAMES = [g[0] for g in GENES]
BLOCK_ORDER = ("guidance", "inner_loop", "mixing")
BLOCKS = [[j for j, g in enumerate(GENES) if g[1] == b] for b in BLOCK_ORDER]
CFG = ga.GAConfig(pop_size=64, elite=2, selection_p=0.2, mutation_rate=0.15, mutation_sigma=0.08, mutation_mode="gauss")


def config(pop_size: int, **over) -> ga.GAConfig:
    d = dict(CFG.__dict__, pop_size=pop_size)
    d.update(over)
    return ga.GAConfig(**d)


def generation_zero(rng, pop_size: int) -> np.ndarray:
    return ga.generation_zero(rng, pop_size, len(GENES))


def next_generation(rng, ranked: np.ndarray, cfg: ga.GAConfig) -> np.ndarray:
    n = ranked.shape[0]
    new: List[np.ndarray] = [ranked[i].copy() for i in range(min(cfg.elite, n))]
    while len(new) < cfg.pop_size:
        i = ga.flat_rank_select(rng, n, cfg.selection_p)
        j = i
        while j == i:
            j = ga.flat_rank_select(rng, n, cfg.selection_p)
        new.append(ga.mutate(rng, ga.crossover_blocks(rng, ranked[i], ranked[j], BLOCKS), cfg))
    return np.array(new)


def rank_order(costs) -> np.ndarray:
    return np.argsort(np.asarray(costs), kind="stable")


# decode: log lo*(hi/lo)^u; linear lo+u(hi-lo); log0 = 0 for u <= 0.05 else log over (0.05,1] (genome_schema.GeneSpec)
LOG0_CUT = 0.05


def decode(u, model: str = None) -> Dict[str, float]:
    """Raw gene decode; with model, also Genome's per-aircraft physical limits (= genome p4_guidance.decode_physical)."""
    out = {}
    for (name, _, lo, hi, sc, _d), x in zip(GENES, u):
        x = float(x)
        if sc == "linear":
            out[name] = lo + x * (hi - lo)
        elif sc == "log":
            out[name] = lo * (hi / lo) ** x
        else:
            out[name] = 0.0 if x <= LOG0_CUT else lo * (hi / lo) ** ((x - LOG0_CUT) / (1 - LOG0_CUT))
    if model is not None:
        a = _limits()[model]
        out["bank_max_deg"] = out["bank_max_frac"] * a["bank_course_deg"]
        out["nz_max_g"] = min(out["nz_max_frac"] * a["n_inst"], a["n_inst"])
        out["nz_min_g"] = max(1.0 - out["nz_min_frac"] * (1.0 - a["n_profile"][0]), a["n_profile"][0])
    return out


def _limits():
    import json, os
    fd = os.environ.get("EVOLUTION_FD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fd_pin_p4cs"))
    return json.load(open(os.path.join(fd, "v2_results", "p4_aircraft_limits.json")))["aircraft"]
