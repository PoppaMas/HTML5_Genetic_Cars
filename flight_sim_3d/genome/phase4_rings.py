"""Phase 4 ring-course chromosome (Genome Architect). Standalone: does NOT go through adapter.load_task, so no existing
preset (legacy .. phase3_b2a_x) can change. Reuses block_ops.block_crossover (whole-block path, rng.random(n_blocks)).

Blocks (canonical order): guidance | inner_loop | mixing [| structure if carry_over == 'structure'].
Gene storage: u in [0,1]; decode by genome_schema.GeneSpec (log / linear / log0). Ranges are c172x reference values;
per-aircraft scaling tags follow profiles.py conventions (open question: FD/ER to confirm per-aircraft rescale).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

import block_ops
import genome_schema as GS

HERE = os.path.dirname(os.path.abspath(__file__))
PRESET = os.path.join(HERE, "presets", "phase4_rings.json")
_g = GS.GeneSpec

GUIDANCE = [
    _g("t_preview_s", "guidance", 1.0, 8.0, "log", 3.0, "s", "aim-point lookahead time along the LOS to the next ring", "none"),
    _g("w_next_ring", "guidance", 0.0, 0.6, "linear", 0.25, "-", "aim-point blend toward ring n+1 (0 = next ring only)", "none"),
    _g("k_lat", "guidance", 0.1, 5.0, "log", 1.0, "deg bank/deg", "bank cmd per deg lateral LOS bearing error", "outer_heading"),
    _g("k_lat_rate", "guidance", 1e-3, 2.0, "log0", 0.0, "deg bank/(deg/s)", "lead on lateral LOS rate (0 = off)", "outer_heading"),
    _g("k_vert", "guidance", 0.05, 3.0, "log", 0.5, "deg gamma/deg", "flight-path cmd per deg vertical LOS error", "outer_vertical"),
    _g("k_vert_rate", "guidance", 1e-3, 1.0, "log0", 0.0, "deg gamma/(deg/s)", "lead on vertical LOS rate (0 = off)", "outer_vertical"),
    _g("bank_max_frac", "guidance", 0.25, 1.0, "linear", 0.75, "x bank_course_deg", "bank limit = frac x FD bank_course_deg (60/75/60/80)", "none"),
    _g("nz_max_frac", "guidance", 0.4, 1.0, "linear", 0.7, "x n_inst", "nz protection ceiling = frac x FD n_inst", "none"),
    _g("nz_min_frac", "guidance", 0.0, 0.75, "linear", 0.5, "-", "nz floor = 1 - frac x (1 - FD n_profile[0])", "none"),
    _g("gamma_max_deg", "guidance", 5.0, 25.0, "linear", 15.0, "deg", "flight-path-angle command limit (climb and dive)", "none"),
    _g("v_cmd_scale", "guidance", 0.85, 1.25, "linear", 1.0, "x V_ref", "speed command as multiple of course reference speed", "none"),
    _g("v_turn_comp", "guidance", 0.0, 0.15, "linear", 0.0, "x V_cmd per g", "speed cmd increase per (nz_cmd - 1) g in turns", "none"),
]
INNER = [
    _g("kp_roll", "inner_loop", 0.002, 0.5, "log", 0.05, "ail/deg", "aileron per deg bank error", "inner_roll"),
    _g("ki_roll", "inner_loop", 1e-5, 0.2, "log0", 0.0, "ail/(deg*s)", "roll I", "inner_roll"),
    _g("kd_roll", "inner_loop", 1e-4, 0.5, "log", 0.02, "ail/(deg/s)", "roll-rate damping", "inner_roll"),
    _g("kp_pitch", "inner_loop", 0.002, 0.5, "log", 0.05, "elev/deg", "elevator per deg pitch/gamma error", "inner_pitch"),
    _g("ki_pitch", "inner_loop", 1e-5, 0.2, "log0", 0.01, "elev/(deg*s)", "pitch I", "inner_pitch"),
    _g("kd_pitch", "inner_loop", 1e-4, 0.5, "log", 0.02, "elev/(deg/s)", "pitch-rate damping", "inner_pitch"),
    _g("kr_yaw", "inner_loop", 1e-4, 0.5, "log0", 0.0, "rud/(deg/s)", "yaw damper on washed-out r", "inner_yaw"),
    _g("kbeta_yaw", "inner_loop", 1e-4, 0.5, "log0", 0.0, "rud/deg", "rudder per deg sideslip", "inner_yaw"),
    _g("tau_washout", "inner_loop", 0.3, 10.0, "log", 2.0, "s", "yaw-rate washout time constant", "none"),
    _g("kp_spd", "inner_loop", 0.002, 0.5, "log", 0.05, "thr/kt", "throttle per kt speed error", "speed"),
    _g("ki_spd", "inner_loop", 1e-4, 0.1, "log0", 0.01, "thr/(kt*s)", "speed I", "speed"),
]
MIXING = [
    _g("k_ari", "mixing", 0.0, 0.6, "linear", 0.0, "rud/ail", "aileron-rudder interconnect", "none"),
    _g("k_turn_coord", "mixing", 0.0, 1.5, "linear", 0.0, "rud/(rad/s)", "rudder feed-forward on r_cmd = g tan(phi)/V", "none"),
    _g("k_elev_bank", "mixing", 0.0, 0.5, "linear", 0.0, "elev", "elevator feed-forward x (1/cos(phi) - 1)", "none"),
    _g("ail_auth", "mixing", 0.4, 1.0, "linear", 1.0, "x travel", "aileron command authority limit", "none"),
    _g("rud_auth", "mixing", 0.2, 1.0, "linear", 1.0, "x travel", "rudder command authority limit", "none"),
    _g("tau_cmd_s", "mixing", 0.05, 0.5, "log", 0.1, "s", "first-order guidance/pilot command prefilter (FD actuator lag is separate)", "none"),
]
CARRY_OVER = ("none", "structure")


def gene_table(carry_over: str = "none") -> List[GS.GeneSpec]:
    if carry_over not in CARRY_OVER:
        raise ValueError(f"carry_over must be one of {CARRY_OVER} (shape carry-over needs block_ops shape ops + FD fidelity; not wired)")
    genes = GUIDANCE + INNER + MIXING
    if carry_over == "structure":
        import adapter
        genes = genes + [g for g in adapter.load_task("phase3_b1").spec.genes if g.block == "structure_v2"]
    return genes


def blocks_of(genes) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for j, g in enumerate(genes):
        out.setdefault(g.block, []).append(j)
    return out


def load_preset(path: str = PRESET) -> Dict:
    raw = json.load(open(path))
    if raw.get("genome") != "phase4_rings":
        raise ValueError("not a phase4_rings preset")
    genes = gene_table(raw.get("carry_over", "none"))
    blk = blocks_of(genes)
    want = list(raw["blocks"]) + (["structure_v2"] if raw.get("carry_over") == "structure" else [])
    if list(blk) != want:
        raise ValueError(f"block order {list(blk)} != preset {want}")
    if raw["operators"].get("crossover") != "blocks" or raw["operators"].get("mutation_mode") != "gauss":
        raise ValueError("phase4_rings operators: crossover 'blocks' + mutation 'gauss' only")
    if raw.get("fitness", {}).get("params", {}).get("struct_v2_source", "fd") != "fd":
        raise ValueError("struct_v2_source must stay 'fd'")
    return {"raw": raw, "genes": genes, "blocks": blk, "elite": int(raw["ga"]["elite"]),
            "selection_p": float(raw["ga"]["selection_p"]), "mutation_rate": float(raw["operators"]["mutation_rate"]),
            "mutation_sigma": float(raw["operators"]["mutation_sigma"]),
            "ops": {"blocks": blk, "shape_crossover": "block"}}


def identity_u(genes) -> np.ndarray:
    return np.array([g.encode(g.default) for g in genes], float)


def decode(u, genes) -> Dict[str, float]:
    return {g.name: g.decode(x) for g, x in zip(genes, u)}


def generation_zero(rng, pop_size: int, P: Dict) -> np.ndarray:
    return rng.random((pop_size, len(P["genes"])))


def mutate(rng, g: np.ndarray, P: Dict) -> np.ndarray:
    """Same form as block_ops.block_mutate's controller/structure path (no shape block)."""
    g = g.copy()
    hit = rng.random(g.shape[0]) < P["mutation_rate"]
    g[hit] += rng.normal(0.0, P["mutation_sigma"], int(hit.sum()))
    return np.clip(g, 0.0, 1.0)


def flat_rank_select(rng, n: int, p: float) -> int:
    """= ga.flat_rank_select (flight_sim/ga.py); duplicated so this module needs no FLIGHT_SIM_DIR. Test checks equality."""
    from flightsim_path import orig_ga
    return orig_ga().flat_rank_select(rng, n, p)


def next_generation(rng, ranked: np.ndarray, P: Dict, trace: Optional[List] = None) -> np.ndarray:
    n = ranked.shape[0]
    new = [ranked[i].copy() for i in range(min(P["elite"], n))]
    while len(new) < n:
        i = flat_rank_select(rng, n, P["selection_p"])
        j = i
        while j == i:
            j = flat_rank_select(rng, n, P["selection_p"])
        rec = {"i": int(i), "j": int(j)} if trace is not None else None
        new.append(mutate(rng, block_ops.block_crossover(rng, ranked[i], ranked[j], P["ops"], rec), P))
        if trace is not None:
            trace.append(rec)
    return np.array(new)


def rank_order(costs) -> np.ndarray:
    return np.argsort(np.asarray(costs), kind="stable")


from p4_guidance import (aircraft_limits, decode_physical, gain_scales, make_guidance, evaluate, cache_key,  # noqa: E402,F401
                         smoke)
