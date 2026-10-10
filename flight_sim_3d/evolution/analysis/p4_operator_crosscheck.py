"""Phase 4 operator bit cross-check: Evolution's phase4_ga vs genome/runs/p4_operator_trace.json (read-only).
Same protocol as the B1_x check: default_rng(1); pop = rng.random((64, 29)); 5 generations; synthetic cost
sum_k (1 + k/n)(g_k - (k*phi mod 1))^2; stable argsort; next_generation between gens. Compares the per-gen ranked
pop / cost sha256 (float64 LE) and the gen0/gen1 arrays. Run: $PY evolution/analysis/p4_operator_crosscheck.py"""
import hashlib, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)
import numpy as np  # noqa: E402
from evolution import phase4_ga as P4  # noqa: E402

TRACE = os.path.join(TEAM, "genome", "runs", "p4_operator_trace.json")
OUT = os.path.join(HERE, "p4_operator_crosscheck.json")
PHI = 0.6180339887498949


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def cost(pop):
    n = pop.shape[1]; k = np.arange(n)
    return (((pop - (k * PHI) % 1.0) ** 2) * (1.0 + k / n)).sum(axis=1)


def run(seed=1, pop_n=64, gens=5):
    rng = np.random.default_rng(seed)
    cfg = P4.config(pop_n)
    pop = P4.generation_zero(rng, pop_n)
    det = [{"gen0_unranked_sha256": sha(pop)}]; arrays = {}
    for g in range(gens):
        cs = cost(pop); o = P4.rank_order(cs); pop, cs = pop[o], cs[o]
        det.append({"gen": g, "pop_sha256": sha(pop), "cost_sha256": sha(cs), "best_cost": float(cs[0]), "mean_cost": float(cs.mean())})
        if g < 2:
            arrays[f"gen{g}_ranked"] = pop.tolist()
        if g < gens - 1:
            pop = P4.next_generation(rng, pop, cfg)
    return det, arrays


def check():
    det, arrays = run()
    ref = json.load(open(TRACE))
    gen_match = [a == b for a, b in zip(det, ref["generations_detail"])]
    arr_match = {k: arrays[k] == ref[k] for k in arrays}
    names_match = [n for b in ref["blocks"].values() for n in b] == P4.NAMES
    return {"trace": os.path.relpath(TRACE, TEAM), "generations_detail": det, "per_gen_match": gen_match,
            "arrays_match": arr_match, "gene_names_match": names_match,
            "all_match": all(gen_match) and len(gen_match) == len(ref["generations_detail"]) and all(arr_match.values()) and names_match,
            "final_pop_sha256": det[-1]["pop_sha256"], "final_cost_sha256": det[-1]["cost_sha256"]}


if __name__ == "__main__":
    r = check()
    json.dump(r, open(OUT, "w"), indent=1)
    print(OUT, r["all_match"], r["final_pop_sha256"][:8], r["final_cost_sha256"][:8])
