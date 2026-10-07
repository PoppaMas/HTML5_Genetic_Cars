"""P3-B1 r1 pilot LADDER CONCERN (read-only on run dirs; evidence only, no fixes).
Per run / aircraft, from history.jsonl (spearman_screen_vs_full = rank corr. of rigid screen vs authoritative cost on the
genomes re-scored that generation: the rigid top k_full + carried elites) and genomes.jsonl rows (screen_cost + cost):
  * rho over all gens and gens >= 1 (gen 0 contains infeasible random genomes): mean, median, fraction < 0, < -0.3;
  * rho_both_ok (both rungs status ok);
  * range restriction: within the re-scored set, spread (IQR, sd) of the rigid cost vs spread of (auth - rigid);
  * where the generation's auth-best sat in the rigid order of the re-scored set (0 = rigid best), and how often it was in
    the rigid top_k (4);
  * gap = auth - rigid at the re-scored genomes (mean).
Usage: $PY evolution/analysis/p3b1r1_ladder_rho.py [RUN ...]  -> analysis/phase3b1r1_ladder_rho.json"""
import json
import os
import statistics as stt
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")
DEFAULT = ["phase3b1r1-pilot-s1", "phase2-pilot-s1", "phase2-pilot-s2", "phase2-pilot-s3"]


def q(v, p):
    v = sorted(v)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summ(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean": stt.fmean(v), "median": stt.median(v), "min": min(v), "max": max(v),
            "frac_lt_0": sum(1 for x in v if x < 0) / len(v), "frac_lt_m0.3": sum(1 for x in v if x < -0.3) / len(v),
            "frac_gt_0.5": sum(1 for x in v if x > 0.5) / len(v)}


def run_report(run):
    d = run if os.path.isabs(run) else os.path.join(RUNS, run)
    rj = json.load(open(os.path.join(d, "run.json")))
    hist = [json.loads(x) for x in open(os.path.join(d, "history.jsonl")) if x.strip()]
    rows = [json.loads(x) for x in open(os.path.join(d, "genomes.jsonl")) if x.strip()]
    top = rj["fidelity"]
    by = defaultdict(list)
    for r in rows:
        if r.get("rescored_at_full") and r.get("fidelity") == top and r.get("screen_cost") is not None:
            by[(r["aircraft"], int(r["generation"]))].append(r)
    out = {"run_id": rj["run_id"], "fidelity": top, "ladder": rj.get("multi_fidelity", {}).get("ladder"),
           "top_k": rj.get("multi_fidelity", {}).get("top_k"), "aircraft": {}}
    for ac in [a["name"] for a in rj["aircraft"]]:
        hs = sorted((h for h in hist if h["aircraft"] == ac), key=lambda h: h["generation"])
        # last session wins on duplicates (resumes)
        hd = {}
        for h in hs:
            hd[h["generation"]] = h
        hs = [hd[g] for g in sorted(hd)]
        rho = [h.get("spearman_screen_vs_full") for h in hs]
        rho_ok = [list((h.get("spearman_both_ok") or {}).values())[0] if h.get("spearman_both_ok") else None for h in hs]
        sd_r, sd_gap, iqr_r, iqr_gap, pos_best, in_topk, gap, nres = [], [], [], [], [], [], [], []
        for h in hs:
            g = h["generation"]
            rs = [r for r in by.get((ac, g), []) if r["status"] == "ok" and r.get("ladder_status", {}).get(rj.get("multi_fidelity", {}).get("screen", "rigid"), "ok") == "ok"]
            if len(rs) < 4:
                continue
            nres.append(len(rs))
            sc = [float(r["screen_cost"]) for r in rs]
            au = [float(r["cost"]) for r in rs]
            dg = [a - s for a, s in zip(au, sc)]
            sd_r.append(stt.pstdev(sc)); sd_gap.append(stt.pstdev(dg))
            iqr_r.append(q(sc, .75) - q(sc, .25)); iqr_gap.append(q(dg, .75) - q(dg, .25))
            gap.append(stt.fmean(dg))
            ordr = sorted(range(len(rs)), key=lambda i: sc[i])
            ib = min(range(len(rs)), key=lambda i: au[i])
            pos_best.append(ordr.index(ib))
            in_topk.append(ordr.index(ib) < (out["top_k"] or 4))
        out["aircraft"][ac] = {
            "rho_all_gens": summ(rho), "rho_gens_ge1": summ(rho[1:]), "rho_gens_ge30": summ(rho[30:]),
            "rho_both_ok_gens_ge1": summ(rho_ok[1:]), "rho_per_gen": rho,
            "n_rescored_ok_mean": stt.fmean(nres) if nres else None,
            "rigid_cost_sd_in_rescored_median": stt.median(sd_r) if sd_r else None,
            "gap_sd_in_rescored_median": stt.median(sd_gap) if sd_gap else None,
            "rigid_cost_iqr_in_rescored_median": stt.median(iqr_r) if iqr_r else None,
            "gap_iqr_in_rescored_median": stt.median(iqr_gap) if iqr_gap else None,
            "gap_sd_over_rigid_sd_median": stt.median([a / b for a, b in zip(sd_gap, sd_r) if b > 0]) if sd_r else None,
            "gap_auth_minus_rigid_mean": stt.fmean(gap) if gap else None,
            "auth_best_rigid_position_mean": stt.fmean(pos_best) if pos_best else None,
            "auth_best_rigid_position_expected_if_random": (stt.fmean(nres) - 1) / 2 if nres else None,
            "auth_best_in_rigid_topk_frac": sum(in_topk) / len(in_topk) if in_topk else None,
            "auth_best_in_rigid_topk_expected_if_random": (out["top_k"] or 4) / stt.fmean(nres) if nres else None}
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    res = {r: run_report(r) for r in (argv or DEFAULT)}
    with open(os.environ.get("REPORT_OUT") or os.path.join(HERE, "phase3b1r1_ladder_rho.json"), "w") as f:
        json.dump(res, f, indent=1)
    for r, x in res.items():
        print("==", r, x["ladder"], "top_k", x["top_k"])
        for ac, a in x["aircraft"].items():
            s, s1, s30 = a["rho_all_gens"], a["rho_gens_ge1"], a["rho_gens_ge30"]
            if not s:
                print(f"  {ac:6s} no rho"); continue
            print(f"  {ac:6s} rho all: mean {s['mean']:+.3f} med {s['median']:+.3f} <0 {s['frac_lt_0']:.2f} <-.3 "
                  f"{s['frac_lt_m0.3']:.2f} [{s['min']:+.2f},{s['max']:+.2f}] n{s['n']} | g>=1 mean {s1['mean']:+.3f} med "
                  f"{s1['median']:+.3f} <0 {s1['frac_lt_0']:.2f} | g>=30 mean {s30['mean']:+.3f} <0 {s30['frac_lt_0']:.2f}")
            print(f"         sd rigid {a['rigid_cost_sd_in_rescored_median']:.4g} sd gap {a['gap_sd_in_rescored_median']:.4g} "
                  f"ratio {a['gap_sd_over_rigid_sd_median']:.2f} | gap mean {a['gap_auth_minus_rigid_mean']:+.4f} | "
                  f"auth-best rigid pos {a['auth_best_rigid_position_mean']:.2f} (random {a['auth_best_rigid_position_expected_if_random']:.2f}) "
                  f"in top{x['top_k']} {a['auth_best_in_rigid_topk_frac']:.2f} (random {a['auth_best_in_rigid_topk_expected_if_random']:.2f})")


if __name__ == "__main__":
    main()
