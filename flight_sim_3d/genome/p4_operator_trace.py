#!/usr/bin/env python3
"""Fixed-seed operator trace for phase4_rings (mirrors p3b1x_operator_trace). Nothing flies.
rng = default_rng(1); pop = rng.random((64, 29)); per gen: synthetic cost (same formula as p3b1x trace) -> stable argsort
-> next_generation (elite 2, flat rank p 0.2, whole-block crossover rng.random(3) < 0.5 -> A, gauss 0.15 / 0.08).
Usage: FLIGHT_SIM_DIR=... $PY genome/p4_operator_trace.py -> genome/runs/p4_operator_trace.json"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("FLIGHT_SIM_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flight_sim"))

import numpy as np  # noqa: E402

import phase4_rings as R  # noqa: E402
from p3b1x_operator_trace import synthetic_cost, sha, SEED, POP, GENS  # noqa: E402


def run():
    P = R.load_preset()
    rng = np.random.default_rng(SEED)
    pop = R.generation_zero(rng, POP, P)
    gens, arrays, traces, elites_ok = [{"gen0_unranked_sha256": sha(pop)}], {}, [], []
    for g in range(GENS):
        cs = synthetic_cost(pop)
        o = R.rank_order(cs)
        pop, cs = pop[o], cs[o]
        gens.append({"gen": g, "pop_sha256": sha(pop), "cost_sha256": sha(cs), "best_cost": float(cs[0]), "mean_cost": float(cs.mean())})
        if g < 2:
            arrays[f"gen{g}_ranked"] = pop.tolist()
        if g < GENS - 1:
            tr = []
            new = R.next_generation(rng, pop, P, tr)
            elites_ok.append(bool(np.array_equal(new[:P["elite"]], pop[:P["elite"]])))
            traces.append(tr)
            pop = new
    return P, {"preset": "phase4_rings", "seed": SEED, "pop": POP, "generations": GENS,
               "n_genes": len(P["genes"]), "blocks": {k: [P["genes"][j].name for j in v] for k, v in P["blocks"].items()},
               "operators": {"elite": P["elite"], "selection": "flat_rank", "selection_p": P["selection_p"],
                             "crossover": "whole-block, d = rng.random(3) in block order guidance|inner_loop|mixing, d < 0.5 -> parent A (ranked[i])",
                             "mutation": "hit = rng.random(29) < 0.15; g[hit] += rng.normal(0, 0.08, hit.sum()); clip [0,1]",
                             "draw_order_per_child": "select i, select j (!= i), crossover rng.random(3), mutation rng.random(29), rng.normal(n_hit)",
                             "rank": "np.argsort(cost, kind='stable')"},
               "synthetic_cost": "sum_k (1 + k/n)(g_k - (k*0.6180339887498949 mod 1))^2",
               "identity_u": R.identity_u(P["genes"]).tolist(), "identity_u_sha256": sha(R.identity_u(P["genes"])),
               "identity_physical": {m: {k: v for k, v in R.decode_physical(R.identity_u(P["genes"]), P["genes"], m).items()
                                         if k in ("bank_max_deg", "nz_max_g", "nz_min_g")} | {"gain_scales": R.gain_scales(m)}
                                     for m in ("c172x", "T38", "737", "f16")},
               "generations_detail": gens, "elite_preserved": elites_ok, "child_traces": traces, **arrays}


if __name__ == "__main__":
    _, out = run()
    path = os.path.join(HERE, "runs", "p4_operator_trace.json")
    json.dump(out, open(path, "w"), indent=1)
    print(path, out["generations_detail"][-1])
