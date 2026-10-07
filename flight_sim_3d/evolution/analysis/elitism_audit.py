# usage: cd flight_sim_3d && $PY -B evolution/analysis/elitism_audit.py /tmp/out.json  (read-only on runs/)
"""Read-only elitism / selection / persistence audit over evolution/runs/<id>. Writes JSON to argv[1]."""
import json, os, sys, hashlib, math
from collections import defaultdict
import numpy as np
TEAM = os.environ.get("EVOLUTION_TEAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # flight_sim_3d/
sys.path.insert(0, TEAM)
from evolution import ga, genome as genome_mod, eval as eval_mod

RUNS = ["phase2-pilot-s1", "phase2-pilot-s2", "phase2-pilot-s3", "phase3a1-smoke-s1", "phase3b1-smoke-s1",
        "phase3b1r1-smoke-s1"]
RD = os.path.join(TEAM, "evolution", "runs")


def sel_probs(n, p):
    """exact P(rank k) of flat_rank_select, plus exact pair marginals with the j != i redraw."""
    q = np.array([p * (1 - p) ** k for k in range(n)]) + (1 - p) ** n / n
    # parent 2: drawn from q conditioned on != i
    p2 = np.zeros(n)
    for i in range(n):
        c = q.copy(); c[i] = 0; c /= c.sum(); p2 += q[i] * c
    return q, p2


def audit_run(rid):
    d = os.path.join(RD, rid)
    cfg = json.load(open(os.path.join(d, "config.json")))["resolved"]
    G, N, E, P = cfg["ga"]["generations"], cfg["ga"]["pop_size"], cfg["ga"]["elite"], cfg["ga"]["selection_p"]
    run_json = json.load(open(os.path.join(d, "run.json")))
    rows = defaultdict(lambda: defaultdict(list))
    nlines = 0
    with open(os.path.join(d, "genomes.jsonl")) as f:
        for line in f:
            r = json.loads(line); nlines += 1
            rows[r["aircraft"]][r["generation"]].append(r)
    q, p2 = sel_probs(N, P)
    out = {"run_id": rid, "genome_kind": cfg.get("genome_kind") or ("phase2_flex(struct_genes)" if cfg.get("struct_genes") else "controller"),
           "fidelity": cfg["fidelity"], "multi_fidelity_ladder": cfg["multi_fidelity"].get("ladder") if cfg["multi_fidelity"]["enabled"] else None,
           "ga": cfg["ga"], "genomes_jsonl_rows": nlines, "genomes_jsonl_bytes": os.path.getsize(os.path.join(d, "genomes.jsonl")),
           "aircraft": {}}
    for ac in cfg["aircraft"]:
        name = ac["name"]
        ent = next(a for a in run_json["aircraft"] if a["name"] == name)
        schema = eval_mod.schema_for(ent)
        ck = json.load(open(os.path.join(d, "checkpoints", f"{name}.json")))
        R = rows[name]
        gens = sorted(R)
        a = {"checkpoint": {"gen_next": ck["gen_next"], "done": ck["done"], "n_history": len(ck["history"]),
                            "n_best_per_gen": len(ck["best_per_gen"]), "next_pop_rows": len(ck["pop"])},
             "gens_with_rows": gens, "rows_per_gen": {g: len(R[g]) for g in gens},
             "all_gens_have_full_population": gens == list(range(G)) and all(len(R[g]) == N for g in gens)}
        best = [min(R[g], key=lambda r: r["rank"]) for g in gens]
        series = [b["cost"] for b in best]
        a["best_cost_per_gen"] = series
        a["best_cost_monotone_nonincreasing"] = all(y <= x for x, y in zip(series, series[1:]))
        a["n_strict_improvements"] = sum(1 for x, y in zip(series, series[1:]) if y < x)
        a["max_regression"] = max([y - x for x, y in zip(series, series[1:])] + [0.0])
        # consistency with checkpoint history / best_per_gen
        hist = {h["generation"]: h for h in ck["history"]}
        bpg = {b["generation"]: b for b in ck["best_per_gen"]}
        a["history_best_equals_rank0_cost"] = all(hist[g]["best"] == b["cost"] for g, b in zip(gens, best))
        a["checkpoint_best_per_gen_equals_rank0"] = all(
            bpg[g]["fitness"] == b["cost"] and bpg[g]["genome"] == b["genome_norm"]
            and [p["cost"] for p in bpg[g]["per_scenario"]] == b["per_scenario_cost"] for g, b in zip(gens, best))
        # rank-0 rows reloadable: decode(genome_norm) == stored decoded genome (bit-identical)
        a["rank0_genome_norm_decodes_to_genome"] = all(genome_mod.decode(b["genome_norm"], schema) == b["genome"] for b in best)
        # rank-0 is always the min of authoritative-fidelity costs
        a["rank0_is_min_authoritative_cost"] = all(
            b["cost"] == min(r["cost"] for r in R[g] if r["fidelity"] == cfg["fidelity"]) and b["fidelity"] == cfg["fidelity"]
            for g, b in zip(gens, best))
        # elites: ranks 0..E-1 of gen g appear at index 0..E-1 of gen g+1, carried_elite, identical genes + cost
        el = []
        for g in gens[:-1]:
            src = sorted([r for r in R[g] if r["rank"] < E], key=lambda r: r["rank"])
            dst = sorted([r for r in R[g + 1] if r["carried_elite"]], key=lambda r: r["index"])
            per = []
            for k, s in enumerate(src):
                t = dst[k] if k < len(dst) else None
                per.append({"rank": s["rank"], "dest_index": t["index"] if t else None,
                            "genes_bit_identical": bool(t) and t["genome_norm"] == s["genome_norm"] and t["genome"] == s["genome"],
                            "cost_bit_identical": bool(t) and t["cost"] == s["cost"] and t["per_scenario_cost"] == s["per_scenario_cost"],
                            "status_same": bool(t) and t["status"] == s["status"], "cost": s["cost"],
                            "cost_next": t["cost"] if t else None, "fidelity": s["fidelity"], "fidelity_next": t["fidelity"] if t else None})
            el.append({"from_gen": g, "n_carried": len(dst), "elites": per})
        a["elite_carry"] = el
        a["elites_all_bit_identical"] = all(e["n_carried"] == E and all(x["genes_bit_identical"] and x["cost_bit_identical"]
                                                                         for x in e["elites"]) for e in el)
        # carried elite also in checkpoint pop -> next gen: genomes.jsonl index i == checkpoint row? (final gen: pop == ranked)
        a["final_checkpoint_pop_equals_final_ranked_rows"] = [r["genome_norm"] for r in sorted(R[gens[-1]], key=lambda r: r["rank"])] == ck["pop"]
        # selection: failed genomes and their rank-probability mass
        sp = []
        for g in gens:
            fail_ranks = sorted(r["rank"] for r in R[g] if r["status"] != "ok")
            nf = len(fail_ranks)
            sp.append({"gen": g, "n_failed": nf, "best_failed_rank": fail_ranks[0] if fail_ranks else None,
                       "p_parent1_failed": float(q[fail_ranks].sum()) if nf else 0.0,
                       "p_parent2_failed": float(p2[fail_ranks].sum()) if nf else 0.0,
                       "fail_statuses": sorted({r["status"] for r in R[g] if r["status"] != "ok"}),
                       "fail_cost_min": min((r["cost"] for r in R[g] if r["status"] != "ok"), default=None),
                       "ok_cost_max": max((r["cost"] for r in R[g] if r["status"] == "ok"), default=None),
                       "all_failed_ranked_below_all_ok": (not nf) or nf == len(R[g]) or
                           min(fail_ranks) > max(r["rank"] for r in R[g] if r["status"] == "ok")})
        a["selection_failed_mass"] = sp
        a["failed_always_ranked_below_ok"] = all(s["all_failed_ranked_below_all_ok"] for s in sp)
        out["aircraft"][name] = a
    return out


def main():
    res = {"selection_model": {}}
    for n in (12, 16, 24, 32, 64):
        q, p2 = sel_probs(n, 0.2)
        res["selection_model"][f"pop{n}"] = {"p_rank0": q[0], "p_top4": float(q[:4].sum()), "p_top10pct": float(q[:max(1, n // 10)].sum()),
                                             "p_top25pct": float(q[:n // 4].sum()), "p_top50pct": float(q[:n // 2].sum()),
                                             "p_bottom50pct": float(q[n // 2:].sum()), "p_worst": float(q[-1]),
                                             "ratio_best_to_median": float(q[0] / q[n // 2]),
                                             "expected_rank_parent1": float((np.arange(n) * q).sum()),
                                             "expected_rank_parent2": float((np.arange(n) * p2).sum()),
                                             "uniform_fallback_prob": float(0.8 ** n)}
    res["runs"] = {r: audit_run(r) for r in RUNS if os.path.isdir(os.path.join(RD, r))}
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
    for rid, r in res["runs"].items():
        for ac, a in r["aircraft"].items():
            print(f"{rid:22s} {ac:6s} gens={len(a['gens_with_rows'])} fullpop={a['all_gens_have_full_population']} mono={a['best_cost_monotone_nonincreasing']} "
                  f"impr={a['n_strict_improvements']} elites_bit={a['elites_all_bit_identical']} hist={a['history_best_equals_rank0_cost']} "
                  f"ck={a['checkpoint_best_per_gen_equals_rank0']} dec={a['rank0_genome_norm_decodes_to_genome']} auth={a['rank0_is_min_authoritative_cost']} "
                  f"finalpop={a['final_checkpoint_pop_equals_final_ranked_rows']} failbelow={a['failed_always_ranked_below_ok']} "
                  f"best {a['best_cost_per_gen'][0]:.5f}->{a['best_cost_per_gen'][-1]:.5f} maxPfail1={max(s['p_parent1_failed'] for s in a['selection_failed_mass']):.3f}")
main()
