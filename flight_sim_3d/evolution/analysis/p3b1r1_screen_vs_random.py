"""P3-B1 r1 pilot ladder probe (evidence only; read-only on the run dir; no cache writes): for one generation per aircraft,
re-score the genomes the rigid screen did NOT promote at full_a1_b1 (fresh process, frozen FD tree, run.json interface)
so the whole population has an authoritative cost, then compare the rigid promotion (top k_full by rigid + elites) with
random selection of the same size.
  EVOLUTION_CODE_ROOT=/workspace/er_pilot_code_b1r1 EVOLUTION_TEAM_ROOT=/workspace/flight-sim-team \
  EVOLUTION_FD_DIR=/workspace/flight-sim-team/evolution/_fd_pin_p3b1r1 nice -n 19 $PY -B p3b1r1_screen_vs_random.py GEN [AIRCRAFT,...]
-> analysis/phase3b1r1_screen_vs_random_g<GEN>.json"""
import json
import os
import random
import statistics as stt
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.environ.get("EVOLUTION_TEAM_ROOT") or os.path.dirname(os.path.dirname(HERE))
CODE = os.environ.get("EVOLUTION_CODE_ROOT") or TEAM
sys.path.insert(0, CODE)
from evolution import eval as ev, fidelity as F, cache as C  # noqa: E402


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = stt.fmean(ra), stt.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    gen = int(argv[0])
    run = os.path.join(TEAM, "evolution", "runs", "phase3b1r1-pilot-s1")
    rj = json.load(open(os.path.join(run, "run.json")))
    acs = argv[1].split(",") if len(argv) > 1 else [a["name"] for a in rj["aircraft"]]
    rows = [json.loads(x) for x in open(os.path.join(run, "genomes.jsonl")) if x.strip()]
    out = {"run": "phase3b1r1-pilot-s1", "generation": gen, "code_sha": C.code_sha(), "fd_dir": os.path.relpath(F.FD_DIR, TEAM),
           "aircraft": {}}
    for ac in acs:
        t0 = time.time()
        rs = sorted((r for r in rows if r["aircraft"] == ac and int(r["generation"]) == gen), key=lambda r: int(r["index"]))
        rig, full, prom = [], [], []
        for r in rs:
            rig.append(float(r["screen_cost"]))
            if r.get("rescored_at_full"):
                full.append(float(r["cost"])); prom.append(True)
            else:
                e = ev.evaluate(r["genome"], ac, None, rj)
                assert e["model_version"] == rj["model_version"][ac]
                full.append(float(e["cost"])); prom.append(False)
        n, k = len(rs), sum(prom)
        P = [i for i in range(n) if prom[i]]
        N = [i for i in range(n) if not prom[i]]
        okf = [i for i in range(n) if full[i] < 1000 and rig[i] < 1000]
        true_top = set(sorted(range(n), key=lambda i: full[i])[:k])
        rng = random.Random(12345)
        rnd_best, rnd_mean, rnd_prec = [], [], []
        for _ in range(20000):
            S = rng.sample(range(n), k)
            rnd_best.append(min(full[i] for i in S)); rnd_mean.append(stt.median([full[i] for i in S]))
            rnd_prec.append(len(true_top & set(S)) / k)
        best_p = min(full[i] for i in P)
        x = {"n": n, "n_promoted": k, "eval_wall_s": time.time() - t0,
             "spearman_rigid_vs_full_all_pop": spearman([rig[i] for i in okf], [full[i] for i in okf]), "n_ok_both": len(okf),
             "spearman_rigid_vs_full_promoted_only": spearman([rig[i] for i in P], [full[i] for i in P]),
             "full_median_promoted": stt.median([full[i] for i in P]), "full_median_not_promoted": stt.median([full[i] for i in N]),
             "full_infeasible_promoted": sum(1 for i in P if full[i] >= 1000), "full_infeasible_not_promoted": sum(1 for i in N if full[i] >= 1000),
             "pop_full_best": min(full), "pop_full_best_promoted": prom[min(range(n), key=lambda i: full[i])],
             "promoted_full_best": best_p,
             "random_k_full_best_median": stt.median(rnd_best), "random_k_frac_best_le_promoted_best": sum(1 for v in rnd_best if v <= best_p) / len(rnd_best),
             "random_k_full_median_median": stt.median(rnd_mean),
             "precision_at_k_promoted": len(true_top & set(P)) / k, "precision_at_k_random_mean": stt.fmean(rnd_prec),
             "rigid": rig, "full": full, "promoted": prom}
        out["aircraft"][ac] = x
        print(f"{ac} g{gen}: rho all-pop {x['spearman_rigid_vs_full_all_pop']:+.3f} (n_ok {len(okf)}) promoted-only "
              f"{x['spearman_rigid_vs_full_promoted_only']:+.3f} | median full prom {x['full_median_promoted']:.4f} not {x['full_median_not_promoted']:.4f} "
              f"| infeas prom {x['full_infeasible_promoted']} not {x['full_infeasible_not_promoted']} | pop best {min(full):.5f} promoted? "
              f"{x['pop_full_best_promoted']} | prec@{k} {x['precision_at_k_promoted']:.2f} vs random {x['precision_at_k_random_mean']:.2f} "
              f"| P(random best<=prom best) {x['random_k_frac_best_le_promoted_best']:.3f} | {x['eval_wall_s']:.0f}s", flush=True)
        with open(os.path.join(HERE, f"phase3b1r1_screen_vs_random_g{gen}.json"), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
