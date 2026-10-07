"""P3-B1 r1 64x60 PILOT report (read-only on the run dir): phase3b1r1-pilot-s1.
Run with the pilot's code snapshot and the frozen FD tree (fresh process):
  EVOLUTION_CODE_ROOT=/workspace/er_pilot_code_b1r1 EVOLUTION_TEAM_ROOT=/workspace/flight-sim-team \
  EVOLUTION_FD_DIR=/workspace/flight-sim-team/evolution/_fd_pin_p3b1r1 $PY -B evolution/analysis/p3b1r1_pilot_report.py [RUN_DIR]
1. per aircraft (p3a1_smoke_report.report + p3b1_smoke_report.shape_report): best = rank 0 of g59 at full_a1_b1, Σ J_*,
   J_mass, tip term, mass Δ %, flutter margin, stiffness floor fractions, nsm pile, shape genes at the best + final-pop
   bound fractions, rejects by status (incl. geometry_gate), best-so-far monotone, wall, CPU s/scenario, box load;
   vs Phase 2 seeds 1..3 and the r1 16x5 smoke.
2. shape split: each aircraft's best re-flown (all scenarios, full_a1_b1) with its own shape and with the shape reset
   to FD's baseline defaults (controller + structure unchanged).
-> analysis/phase3b1r1_pilot_s1_check.json"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.environ.get("EVOLUTION_TEAM_ROOT") or os.path.dirname(os.path.dirname(HERE))
CODE = os.environ.get("EVOLUTION_CODE_ROOT") or TEAM
sys.path.insert(0, HERE)
sys.path.insert(0, CODE)
import p3a1_smoke_report  # noqa: E402
from p3b1_smoke_report import shape_report  # noqa: E402
from evolution import eval as ev, fidelity as F, cache as C  # noqa: E402

P2 = {"c172x": (0.21539, 0.01337), "T38": (0.09580, 0.00321), "737": (0.12348, 0.00284)}   # Phase 2 seeds 1..3 mean ± sd
SMOKE_R1 = {"c172x": 0.383003, "T38": 0.157143, "737": 0.272112}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    run = argv[0] if argv else os.path.join(TEAM, "evolution", "runs", "phase3b1r1-pilot-s1")
    rj = json.load(open(os.path.join(run, "run.json")))
    summ = json.load(open(os.path.join(run, "summary.json")))
    hist = [json.loads(x) for x in open(os.path.join(run, "history.jsonl")) if x.strip()]
    rows = [json.loads(x) for x in open(os.path.join(run, "genomes.jsonl")) if x.strip()]
    R, S = p3a1_smoke_report.report(run), shape_report(run)
    out = {"run_id": rj.get("run_id"), "code_root": CODE, "code_sha_this_process": C.code_sha(),
           "code_sha_run": rj.get("code_sha"), "fd_dir_run": rj.get("fd_dir"),
           "fd_dir_this_process": os.path.relpath(F.FD_DIR, TEAM),
           "wall_s": summ["wall_s_this_session"], "workers": summ["workers"],
           "loadavg_start": summ.get("loadavg_1_5_15_start"), "loadavg_end": summ.get("loadavg_1_5_15_end"),
           "sessions": [json.loads(x) for x in open(os.path.join(run, "sessions.jsonl")) if x.strip()],
           "aircraft": {}}
    for n, x in R["aircraft"].items():
        hs = sorted((h for h in hist if h["aircraft"] == n), key=lambda h: h["generation"])
        bests = [h["best"] for h in hs]
        mono = all(b2 <= b1 for b1, b2 in zip(bests, bests[1:]))
        sa = next(a for a in summ["aircraft"] if a["aircraft"] == n)
        rs = [r for r in rows if r["aircraft"] == n]
        auth_rows = [r for r in rs if r.get("fidelity") == "full_a1_b1"]
        x2 = dict(x)
        x2.update({"best_so_far_monotone": mono, "best_fidelity_all_gens": sorted({h.get("best_fidelity") for h in hs}),
                   "gen_first_reached_final_best": next(i for i, b in enumerate(bests) if b == bests[-1]),
                   "rejected_by_status_authoritative": {}, "resim_mismatches": sa.get("resim_mismatches"),
                   "p2_seed_mean_sd": P2[n], "vs_p2_mean": x["best_cost"] - P2[n][0],
                   "vs_p2_in_sd": (x["best_cost"] - P2[n][0]) / P2[n][1], "smoke_r1_best": SMOKE_R1[n],
                   "vs_smoke_r1": x["best_cost"] - SMOKE_R1[n], "n_rows": len(rs), "n_rows_full_a1_b1": len(auth_rows),
                   "spearman_rigid_vs_full_mean": (lambda v: sum(v) / len(v) if v else None)(
                       [h["spearman_screen_vs_full"] for h in hs if h.get("spearman_screen_vs_full") is not None])})
        for r in auth_rows:
            if r["status"] != "ok":
                d = x2["rejected_by_status_authoritative"]
                d[r["status"]] = d.get(r["status"], 0) + 1
        x2["shape"] = S[n]
        out["aircraft"][n] = x2
    # 2. shape split (fresh evaluations through the frozen FD tree)
    defaults = {g.name: g.default for g in F.shape_schema()}
    out["shape_split"] = {}
    for n, x in (out["aircraft"].items() if not os.environ.get("SKIP_SPLIT") else []):
        best = next(r for r in rows if r["individual_id"] == x["best_id"])
        own = ev.evaluate(best["genome"], n, None, rj)
        bas = ev.evaluate(dict(best["genome"], **defaults), n, None, rj)
        out["shape_split"][n] = {"best_id": best["individual_id"], "logged_cost": best["cost"], "shape": best.get("shape"),
                                 "refly_own_shape_cost": own["cost"], "refly_own_bit_identical": own["cost"] == best["cost"],
                                 "baseline_shape_cost": bas["cost"], "baseline_shape_status": bas["status"],
                                 "shape_contribution": best["cost"] - bas["cost"],
                                 "per_scenario_own": own["per_scenario_cost"], "per_scenario_baseline_shape": bas["per_scenario_cost"],
                                 "terms_own": own["terms"], "terms_baseline_shape": bas["terms"],
                                 "margins_own": own.get("margins"), "margins_baseline_shape": bas.get("margins")}
        print(n, "split own", own["cost"], "base-shape", bas["cost"], "shape contrib", best["cost"] - bas["cost"],
              "bit-identical own", own["cost"] == best["cost"])
    with open(os.environ.get("REPORT_OUT") or os.path.join(HERE, "phase3b1r1_pilot_s1_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    for n, x in out["aircraft"].items():
        s = x["shape"]
        print(f"{n:6s} best {x['best_cost']:.6f} ({x['best_id']}, {x['model_version']}) P2 {P2[n][0]:.5f}±{P2[n][1]:.5f} "
              f"({x['vs_p2_in_sd']:+.1f} sd) smoke {SMOKE_R1[n]:.4f} | ΣJ {x['structural_sum']:.6f} Jm {x['J_mass']:.4g} "
              f"tip {x['J_wing_tip_bm_limit']:.3g} mass {x['mass_delta_pct']:+.3f}% flut {x['flutter_margin']:.4f} | "
              f"mono {x['best_so_far_monotone']} | rej {x['rejected_by_status']} auth {x['rejected_by_status_authoritative']} "
              f"gate {s['geometry_gate_rejects_rows']}/{s['geometry_gate_rejects_history']} | cpu/sc {x['cpu_s_per_scenario']:.3f}")
        print("        stiff@floor", {k: v for k, v in x["stiffness_floor_fraction"].items() if v}, "nsm@1.0",
              x["nsm_floor_fraction"], "best@bounds", list(x["best_genes_at_bounds"]))
        for g, d in s["shape_genes"].items():
            print(f"        {g:24s} best {d['best']:+.4f} mean {d['final_mean']:+.4f} sd {d['final_std']:.4f} "
                  f"lo {d['final_frac_at_floor']:.2f} hi {d['final_frac_at_ceiling']:.2f}")
        for fl in s["bound_pile_flags"]:
            print("        FLAG", fl)
    print("wall", out["wall_s"], "load", out["loadavg_start"], out["loadavg_end"], "code_sha", out["code_sha_this_process"],
          out["code_sha_run"], "fd_dir", out["fd_dir_run"], out["fd_dir_this_process"])


if __name__ == "__main__":
    main()
