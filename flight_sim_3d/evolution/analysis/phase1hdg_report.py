"""Heading-hold Phase-1 runs (phase1hdg-s1..3) vs Phase-1 (phase1-s*) and Genome's handoff numbers."""
import json, os, statistics as stt, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
ACS = ["c172x", "T38", "737", "f16"]
GENOME = {"c172x": "0.1935 mean (drift ~25 deg -> <1 deg)", "737": "0.1068", "T38": "0.0966 / 0.0930 / 0.0925", "f16": "(off)"}


def load(run):
    p = os.path.join(R, run, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main(prefix="phase1hdg-s", base="phase1-s", base_f16="phase1-f16fd-s"):
    out = {}
    for ac in ACS:
        rows = []
        for s in (1, 2, 3):
            sm = load(f"{prefix}{s}")
            if not sm:
                continue
            a = next(x for x in sm["aircraft"] if x["aircraft"] == ac)
            fps = a["final_per_scenario"]
            drift = max((abs(p.get("hdg_drift_deg", 0.0)) for p in fps), default=None) if "hdg_drift_deg" in fps[0] else None
            mxe = max((p.get("hdg_max_abs_err_deg", 0.0) for p in fps), default=None) if "hdg_max_abs_err_deg" in fps[0] else None
            b = load(f"{(base_f16 if ac == 'f16' else base)}{s}")
            bb = next(x for x in b["aircraft"] if x["aircraft"] == ac)["best_fitness"] if b else None
            rows.append({"seed": s, "best": a["best_fitness"], "gen0": a["gen0_best_fitness"], "drift_deg": drift,
                         "max_abs_err_deg": mxe, "phase1_best": bb, "wall_s": a.get("evolve_wall_s_this_session"),
                         "run_wall_s": sm["wall_s_this_session"], "load_start": sm["loadavg_1_5_15_start"][0],
                         "load_end": sm["loadavg_1_5_15_end"][0], "resim_mismatches": len(a["resim_mismatches"]),
                         "genes_at_bound": a["genes_at_bound"], "best_gains": a["best_gains"],
                         "invalid_rate_final": a["invalid_rate_final"]})
        bests = [r["best"] for r in rows]
        out[ac] = {"rows": rows, "mean": stt.mean(bests) if bests else None,
                   "std": stt.pstdev(bests) if len(bests) > 1 else None, "genome": GENOME[ac]}
    return out


if __name__ == "__main__":
    o = main(*sys.argv[1:])
    for ac, d in o.items():
        print(f"\n{ac}: mean {d['mean']:.5f} std {d['std'] if d['std'] is None else round(d['std'], 5)}  genome: {d['genome']}")
        for r in d["rows"]:
            print(f"  s{r['seed']} best {r['best']:.6f} (phase1 {r['phase1_best']:.6f})  gen0 {r['gen0']:.4f}  drift "
                  f"{r['drift_deg']}  max|e| {r['max_abs_err_deg']}  wall {r['wall_s']:.1f}s  mism {r['resim_mismatches']}  "
                  f"bound {r['genes_at_bound']}")
    json.dump(o, open(os.path.join(R, "..", "logs", "phase1hdg_report.json"), "w"), indent=1)
