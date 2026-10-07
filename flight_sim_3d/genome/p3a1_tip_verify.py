#!/usr/bin/env python3
"""P3-A1 tip-verify on the genome side (read-only on evolution/ and flight-dynamics/).

Re-flies genome/runs/p25_tip_verify_c172x.json (baseline + soft_tip_taper4_0.75) through
evolution.fidelity.evaluate_genome exactly as evolution/analysis/p3a1_tip_verify.py does:
phase2_smoke_p25 c172x profile, 3 scenarios, scenario_seed 1, fidelities full + full_a1.

Writes genome/runs/p3a1_tip_verify_c172x.json with gains/struct/cost/per_scenario/tip/
fd_struct terms/model_version and a match block vs ER's A1 expected numbers.

Usage (from team root or genome/):
  $PY genome/p3a1_tip_verify.py
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
sys.path.insert(0, TEAM)

from evolution import batch, fidelity as F, sim  # noqa: E402

RIGID = ("track", "effort", "comfort", "heading", "hold")

# Pre-flight sizing / mass / smooth terms (FD PRE_TERMS["full"] family) — matches genome p25 fd_pre_sum.
PRE_KEYS = (
    "J_mass", "J_smooth",
    "J_wing_bm_limit", "J_wing_torque_limit", "J_wing_ip_limit",
    "J_wing_tip_bm_limit", "J_tail_bm_limit", "J_fus_bm_limit",
)

# ER A1 numbers to match bit-for-bit (evolution/analysis/p3a1_tip_verify_evolution.json)
ER_A1_EXPECTED = {
    "baseline": {
        "cost": 0.2419400885448951,
        "per_scenario_cost": [
            0.12772764189700128,
            0.39710530396908106,
            0.2009873197686029,
        ],
        "J_wing_tip_bm_limit": 0.0,
        "model_version": "full_a1:flexv2a1:36fb4f5a",
    },
    "soft_tip_taper4_0.75": {
        "cost": 0.25879473336084763,
        "per_scenario_cost": [
            0.14271980987754557,
            0.4179016740566261,
            0.21576271614837125,
        ],
        "J_wing_tip_bm_limit": 0.011874078070487388,
        "model_version": "full_a1:flexv2a1:36fb4f5a",
    },
}


def _fd_struct_terms(terms: dict) -> dict:
    return {k: float(v) for k, v in terms.items() if k not in RIGID}


def _match_a1(cost, per, tip, model_version, er_exp):
    m = {
        "er_expected_cost": er_exp["cost"],
        "er_expected_per_scenario_cost": er_exp["per_scenario_cost"],
        "er_expected_J_wing_tip_bm_limit": er_exp["J_wing_tip_bm_limit"],
        "er_expected_model_version": er_exp["model_version"],
        "bit_identical_total": cost == er_exp["cost"],
        "bit_identical_per_scenario": per == er_exp["per_scenario_cost"],
        "bit_identical_tip": tip == er_exp["J_wing_tip_bm_limit"],
        "model_version_ok": model_version == er_exp["model_version"],
    }
    m["bit_identical"] = (
        m["bit_identical_total"]
        and m["bit_identical_per_scenario"]
        and m["bit_identical_tip"]
        and m["model_version_ok"]
    )
    if not m["bit_identical_total"]:
        m["delta_total"] = cost - er_exp["cost"]
    if not m["bit_identical_tip"]:
        m["delta_tip"] = tip - er_exp["J_wing_tip_bm_limit"]
    return m


def _match_p25(cost, per, tip, gref_case):
    m = {
        "genome_p25_cost": gref_case["cost"],
        "genome_p25_per_scenario_cost": gref_case["per_scenario_cost"],
        "genome_p25_tip": gref_case["J_wing_tip_bm_limit"],
        "bit_identical_total": cost == gref_case["cost"],
        "bit_identical_per_scenario": per == gref_case["per_scenario_cost"],
        "bit_identical_tip": tip == gref_case["J_wing_tip_bm_limit"],
    }
    m["bit_identical"] = (
        m["bit_identical_total"]
        and m["bit_identical_per_scenario"]
        and m["bit_identical_tip"]
    )
    return m


def _eval_block(r, wall_s, *, er_exp=None, gref_case=None):
    terms = r["terms"]
    fd_terms = _fd_struct_terms(terms)
    tip = float(terms.get("J_wing_tip_bm_limit", 0.0))
    per = [float(p["cost"]) for p in r["per_scenario"]]
    cost = float(r["cost"])
    if isinstance(r["status"], str):
        status = [p.get("status", r["status"]) for p in r["per_scenario"]]
    else:
        status = list(r["status"])
    block = {
        "cost": cost,
        "per_scenario_cost": per,
        "status": status,
        "J_wing_tip_bm_limit": tip,
        "fd_struct_terms": fd_terms,
        "fd_pre_sum": float(sum(fd_terms.get(k, 0.0) for k in PRE_KEYS)),
        "margins": r.get("margins"),
        "mass_total_frac": r.get("mass_total_frac"),
        "model_version": r["model_version"],
        "struct_v2_source": "fd",
        "wall_s": wall_s,
    }
    if er_exp is not None:
        block["match_vs_er_a1"] = _match_a1(cost, per, tip, r["model_version"], er_exp)
        if r.get("structural_model") is not None:
            block["structural_model"] = r["structural_model"]
        if r.get("tip_bm") is not None:
            block["tip_bm"] = r["tip_bm"]
    if gref_case is not None:
        block["match_vs_genome_p25"] = _match_p25(cost, per, tip, gref_case)
    return block


def main():
    cfg = batch.resolve_config(
        json.load(open(os.path.join(TEAM, "evolution", "configs", "phase2_smoke_p25.json"))),
        "tv",
    )
    pd = next(a for a in cfg["aircraft"] if a["name"] == "c172x")["resolved_profile"]
    prof = sim.Profile.from_dict(pd)
    scs = [s.to_dict() for s in sim.make_scenarios(cfg["scenarios"], cfg["scenario_seed"], prof)]
    gref = json.load(open(os.path.join(HERE, "runs", "p25_tip_verify_c172x.json")))
    pins = json.load(
        open(os.path.join(F.FD_DIR, "v2_results", "model_versions_post_p3a1.json"))
    )["c172x"]

    out = {
        "method": (
            "evolution.fidelity.evaluate_genome(phase2_smoke_p25 c172x profile, "
            "3 scenarios, scenario_seed 1); genomes from genome/runs/p25_tip_verify_c172x.json; "
            "same setup as evolution/analysis/p3a1_tip_verify.py"
        ),
        "model_version_expected": {"full": pins["full"], "full_a1": pins["full_a1"]},
        "er_a1_expected": ER_A1_EXPECTED,
        "cases": {},
    }

    for case, g in gref.items():
        case_out = {"gains": g["gains"], "struct": g["struct"]}
        for fid in ("full", "full_a1"):
            t0 = time.perf_counter()
            r = F.evaluate_genome(pd, g["gains"], g["struct"], scs, fid, 0.9)
            wall = time.perf_counter() - t0
            if fid == "full":
                case_out["full"] = _eval_block(r, wall, gref_case=g)
            else:
                case_out["full_a1"] = _eval_block(r, wall, er_exp=ER_A1_EXPECTED[case])
            print(
                case, fid, r["cost"], r["terms"]["J_wing_tip_bm_limit"],
                r["model_version"], f"{wall:.2f}s",
            )

        case_out["delta_a1_minus_full"] = {
            "cost": case_out["full_a1"]["cost"] - case_out["full"]["cost"],
            "J_wing_tip_bm_limit": (
                case_out["full_a1"]["J_wing_tip_bm_limit"]
                - case_out["full"]["J_wing_tip_bm_limit"]
            ),
        }
        # Top-level mirrors of A1 (primary deliverable) for quick read / same shape as p25 file
        a1 = case_out["full_a1"]
        case_out["cost"] = a1["cost"]
        case_out["per_scenario_cost"] = a1["per_scenario_cost"]
        case_out["status"] = a1["status"]
        case_out["J_wing_tip_bm_limit"] = a1["J_wing_tip_bm_limit"]
        case_out["fd_struct_terms"] = a1["fd_struct_terms"]
        case_out["fd_pre_sum"] = a1["fd_pre_sum"]
        case_out["model_version"] = a1["model_version"]
        case_out["struct_v2_source"] = "fd"
        case_out["match_vs_er_a1"] = a1["match_vs_er_a1"]
        out["cases"][case] = case_out

    out["full_all_bit_identical_vs_p25"] = all(
        c["full"]["match_vs_genome_p25"]["bit_identical"] for c in out["cases"].values()
    )
    out["full_a1_all_bit_identical_vs_er"] = all(
        c["full_a1"]["match_vs_er_a1"]["bit_identical"] for c in out["cases"].values()
    )

    dest = os.path.join(HERE, "runs", "p3a1_tip_verify_c172x.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("wrote", dest)
    print("full vs p25 bit-identical:", out["full_all_bit_identical_vs_p25"])
    print("A1 vs ER bit-identical:", out["full_a1_all_bit_identical_vs_er"])


if __name__ == "__main__":
    main()
