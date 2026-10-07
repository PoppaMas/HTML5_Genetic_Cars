"""Phase 2 run checks (read-only).  $PY evolution/analysis/phase2_check.py RUN_DIR [--json OUT] [--floor-tol 0.02]
                                                                           [--pile-frac 0.25] [--compare RUN_DIR_B]
Per aircraft, on the last generation in genomes.jsonl:
  * the optimum (rank 0): cost, every structural term (non-rigid TERM_KEYS) with the negative ones flagged, the
    structural sum (flagged if < 0), margins, mass change (FD total_frac, per-body lb, J_mass FD vs clipped);
  * for every struct gene: the fraction of the final population within --floor-tol (normalized, = 2 % of the gene's
    encoded range) of its floor; a STIFFNESS gene with fraction >= --pile-frac is flagged as piled at the floor;
  * invalid rate (status != ok, all generations and the last one).
--compare B: the same summary for a second run (e.g. the ladder benchmark) plus best-cost / wall / evaluation deltas.
Exit code 1 when any flag is raised (so it can gate a pilot launch)."""
import argparse
import json
import os
import sys
from collections import defaultdict

STIFFNESS_GENES = ("wing_ei_root", "wing_ei_taper_1", "wing_ei_taper_2", "wing_ei_taper_3", "wing_ei_taper_4",
                   "wing_gj_ratio_root", "wing_gj_ratio_tip", "tail_stiffness_scale", "fuselage_stiffness_scale")
RIGID_TERMS = ("track", "effort", "comfort", "heading", "hold")   # FD flexeval.RIGID_TERMS (sim cost); all J_* are structural


def load(run_dir):
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "genomes.jsonl")) if line.strip()]
    rj = os.path.join(run_dir, "run.json")
    return rows, (json.load(open(rj)) if os.path.exists(rj) else {})


def struct_terms(terms):
    return {k: v for k, v in (terms or {}).items() if k.startswith("J_") and v is not None}


def check_aircraft(name, rows, genes, floor_tol=0.02, pile_frac=0.25):
    by_gen = defaultdict(list)
    for r in rows:
        by_gen[r["generation"]].append(r)
    last = max(by_gen)
    final = by_gen[last]
    best = min(final, key=lambda r: r["rank"])
    flags = []
    st = struct_terms(best.get("terms"))
    neg = {k: v for k, v in st.items() if v < 0}
    s_sum = sum(st.values())
    if s_sum < 0:
        flags.append(f"negative structural cost at the optimum: {s_sum:.6g}")
    for k, v in neg.items():
        flags.append(f"negative structural term at the optimum: {k} = {v:.6g}")
    floors, gain_floors, ceilings, best_at_bounds = {}, {}, {}, {}
    for j, g in enumerate(genes):
        u = best["genome_norm"][j]
        if u <= floor_tol or u >= 1 - floor_tol:
            best_at_bounds[g["name"]] = {"side": "floor" if u <= floor_tol else "ceiling", "u": u,
                                         "value": (best.get("genome") or {}).get(g["name"]), "bounds": [g["min"], g["max"]]}
        ceilings[g["name"]] = sum(1 for r in final if r["genome_norm"][j] >= 1 - floor_tol) / len(final)
        if g.get("group") != "struct":
            gain_floors[g["name"]] = sum(1 for r in final if r["genome_norm"][j] <= floor_tol) / len(final)
            continue
        frac = sum(1 for r in final if r["genome_norm"][j] <= floor_tol) / len(final)
        floors[g["name"]] = frac
        if g["name"] in STIFFNESS_GENES and frac >= pile_frac:
            flags.append(f"stiffness gene piled at its floor: {g['name']} ({frac:.0%} of the final population within "
                         f"{floor_tol:.0%} of {g['min']})")
    n_bad_all = sum(1 for r in rows if r["status"] != "ok")
    n_bad_last = sum(1 for r in final if r["status"] != "ok")
    return {"aircraft": name, "generation": last, "pop": len(final), "best_id": best["individual_id"],
            "best_cost": best["cost"], "best_status": best["status"], "best_feasible": best.get("feasible"),
            "best_fidelity": best.get("fidelity"), "best_model_version": best.get("model_version"),
            "structural_terms": st, "structural_sum": s_sum, "negative_terms": neg,
            "sim_cost_mean": (sum(best["per_scenario_cost"]) / len(best["per_scenario_cost"]) - s_sum
                              if best.get("per_scenario_cost") else None),
            "margins": best.get("margins"), "mass_total_frac": best.get("mass_total_frac"), "mass_lb": best.get("mass_lb"),
            "J_mass_fd": best.get("J_mass_fd"), "mass_credit_delta": best.get("mass_credit_delta"),
            "best_struct": best.get("struct"), "best_gains": best.get("gains"), "floor_fraction": floors,
            "floor_fraction_gains": gain_floors, "ceiling_fraction": ceilings, "best_genes_within_tol_of_bounds": best_at_bounds,
            "invalid_rate_all": n_bad_all / len(rows), "invalid_rate_last": n_bad_last / len(final),
            "n_rows": len(rows), "flags": flags}


def check_run(run_dir, floor_tol=0.02, pile_frac=0.25):
    rows, rj = load(run_dir)
    genes_of = {a["name"]: a.get("genes") or [] for a in rj.get("aircraft", [])}
    per = defaultdict(list)
    for r in rows:
        per[r["aircraft"]].append(r)
    out = {"run_dir": run_dir, "run_id": rj.get("run_id"), "aircraft": {}}
    for name, rs in per.items():
        out["aircraft"][name] = check_aircraft(name, rs, genes_of.get(name, []), floor_tol, pile_frac)
    hp = os.path.join(run_dir, "history.jsonl")
    if os.path.exists(hp):
        hist = [json.loads(x) for x in open(hp) if x.strip()]
        for name, x in out["aircraft"].items():
            hs = [h for h in hist if h.get("aircraft") == name]
            sp = [h["spearman_screen_vs_full"] for h in hs if h.get("spearman_screen_vs_full") is not None]
            stages = defaultdict(list)
            for h in hs:
                for k, v in (h.get("spearman_both_ok") or {}).items():
                    if v is not None:
                        stages[k].append(v)
            x["ladder"] = {"spearman_screen_vs_full_mean": (sum(sp) / len(sp)) if sp else None,
                           "spearman_screen_vs_full_last": sp[-1] if sp else None, "n_gens": len(sp),
                           "spearman_both_ok_mean": {k: sum(v) / len(v) for k, v in stages.items()},
                           "n_rescored_mean": (sum(h.get("n_rescored", 0) for h in hs) / len(hs)) if hs else None,
                           "eval_cpu_s": sum(h.get("eval_cpu_s", 0.0) for h in hs),
                           "eval_wall_s": sum(h.get("eval_wall_s", 0.0) for h in hs),
                           "sims_computed": sum(h.get("sims_computed", 0) for h in hs)}
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        out["wall_s"] = s.get("wall_s")
        out["summary_aircraft"] = {a.get("aircraft"): {k: a.get(k) for k in ("wall_s", "evals", "sims_computed",
                                                                              "eval_cpu_s") if k in a}
                                   for a in s.get("aircraft", [])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--compare")
    ap.add_argument("--json")
    ap.add_argument("--floor-tol", type=float, default=0.02)
    ap.add_argument("--pile-frac", type=float, default=0.25)
    a = ap.parse_args(argv)
    res = {"A": check_run(a.run_dir, a.floor_tol, a.pile_frac)}
    if a.compare:
        res["B"] = check_run(a.compare, a.floor_tol, a.pile_frac)
        res["delta_best_cost_B_minus_A"] = {n: res["B"]["aircraft"][n]["best_cost"] - x["best_cost"]
                                            for n, x in res["A"]["aircraft"].items() if n in res["B"]["aircraft"]}
        if res["A"].get("wall_s") and res["B"].get("wall_s"):
            res["wall_ratio_B_over_A"] = res["B"]["wall_s"] / res["A"]["wall_s"]
    flags = [f"[{k}:{n}] {f}" for k in ("A", "B") if k in res for n, x in res[k]["aircraft"].items() for f in x["flags"]]
    res["flags"] = flags
    for k in ("A", "B"):
        if k not in res:
            continue
        for n, x in res[k]["aircraft"].items():
            print(f"{k} {n}: g{x['generation']} best {x['best_cost']:.6g} ({x['best_status']}), structural sum "
                  f"{x['structural_sum']:.6g}, mass_frac {x['mass_total_frac']}, invalid {x['invalid_rate_all']:.1%}")
            print("   floor fractions: " + ", ".join(f"{g}={v:.0%}" for g, v in x["floor_fraction"].items()))
    print("FLAGS:" if flags else "no flags", *flags, sep="\n  ")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
