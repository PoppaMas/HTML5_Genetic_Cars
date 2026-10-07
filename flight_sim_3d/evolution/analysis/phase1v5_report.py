"""Phase-1 v5 candidate runs (phase1v5-s1..3) vs Genome's reported v5 results; cross-evaluation v4 <-> v5 genomes."""
import json, os, statistics as stt, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PKG))
import numpy as np                                 # noqa: E402
from evolution import batch, sim                   # noqa: E402
R = os.path.join(PKG, "runs")


def profiles(cfg_name):
    cfg = batch.resolve_config(json.load(open(os.path.join(PKG, "configs", cfg_name))), "x")
    return {a["name"]: a["resolved_profile"] for a in cfg["aircraft"]}, cfg


def task_cost(pd, cfg, gains):
    P = sim.Profile.from_dict(pd)
    rs = [sim.simulate(gains, s, P) for s in sim.make_scenarios(cfg["scenarios"], cfg["scenario_seed"], P)]
    return float(np.mean([r["cost"] for r in rs])), rs


def main():
    p5, c5 = profiles("phase1_v5.json")
    p4, c4 = profiles("phase1_hdg.json")
    out = {}
    for ac in ("c172x", "T38", "737"):
        rows = []
        for s in (1, 2, 3):
            f = os.path.join(R, f"phase1v5-s{s}", "summary.json")
            if not os.path.exists(f):
                continue
            sm = json.load(open(f))
            a = next(x for x in sm["aircraft"] if x["aircraft"] == ac)
            fps = a["final_per_scenario"]
            g5 = a["best_gains"]
            v4sm = json.load(open(os.path.join(R, f"phase1hdg-s{s}", "summary.json")))
            g4 = next(x for x in v4sm["aircraft"] if x["aircraft"] == ac)["best_gains"]
            v5_on_v4, _ = task_cost(p4[ac], c4, g5)
            v4_on_v5, rs45 = task_cost(p5[ac], c5, g4)
            rows.append({"seed": s, "best": a["best_fitness"], "gen0": a["gen0_best_fitness"],
                         "hold_osc": [p.get("hold_osc") for p in fps],
                         "hold_pp_calm_ft": fps[0].get("hold_pp_ft"), "hold_pp_worst_ft": max(p.get("hold_pp_ft", 0) for p in fps),
                         "draft_residual_ft": fps[3].get("draft_residual_ft"), "draft_max_err_ft": fps[3].get("draft_max_err_ft"),
                         "ki_alt": g5["ki_alt"], "ki_pitch": g5["ki_pitch"], "genes_at_bound": a["genes_at_bound"],
                         "hdg_drift_max": max(abs(p.get("hdg_drift_deg", 0)) for p in fps),
                         "v4_genome_on_v5": v4_on_v5, "v5_genome_on_v4": v5_on_v4,
                         "v4_best_on_v4": next(x for x in v4sm["aircraft"] if x["aircraft"] == ac)["best_fitness"],
                         "v4_genome_hold_pp_calm_on_v5": rs45[0].get("hold_pp_ft"),
                         "v4_genome_draft_residual_ft": rs45[3].get("draft_residual_ft"),
                         "wall_s": a.get("evolve_wall_s_this_session"), "run_wall_s": sm["wall_s_this_session"],
                         "load": [sm["loadavg_1_5_15_start"][0], sm["loadavg_1_5_15_end"][0]],
                         "resim_mismatches": len(a["resim_mismatches"])})
        b = [r["best"] for r in rows]
        out[ac] = {"rows": rows, "mean": stt.mean(b) if b else None, "std_pop": stt.pstdev(b) if len(b) > 1 else None}
    json.dump(out, open(os.path.join(PKG, "logs", "phase1v5_report.json"), "w"), indent=1)
    for ac, d in out.items():
        print(f"\n{ac}: mean {d['mean']}, std {d['std_pop']}")
        for r in d["rows"]:
            print(f"  s{r['seed']} best {r['best']:.6f} gen0 {r['gen0']:.4f} pp calm/worst {r['hold_pp_calm_ft']:.2f}/{r['hold_pp_worst_ft']:.2f} "
                  f"draft res {r['draft_residual_ft']:+.2f} max {r['draft_max_err_ft']:.2f} ki_alt {r['ki_alt']:.3g} ki_pitch {r['ki_pitch']:.3g} "
                  f"bound {r['genes_at_bound']} | v4 genome on v5 {r['v4_genome_on_v5']:.4f} (pp {r['v4_genome_hold_pp_calm_on_v5']:.2f}, res "
                  f"{r['v4_genome_draft_residual_ft']:+.2f}) v5 genome on v4 {r['v5_genome_on_v4']:.4f} vs v4 best {r['v4_best_on_v4']:.4f} "
                  f"wall {r['run_wall_s']:.0f}s load {r['load'][0]:.1f}->{r['load'][1]:.1f} mism {r['resim_mismatches']}")


if __name__ == "__main__":
    main()
