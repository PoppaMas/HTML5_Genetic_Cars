"""P3-B2a energy-term sanity through ER's evaluate_genome (fidelity full_a1_b2a, energy_cost on/off).

Same inputs as FD's _scratch/p3b2/sweep.py (Phase-1 best gains, FD default profile, fe.phase1_scenarios, baseline
structure {}), flown on the frozen pin. Compares FD's raw cost / energy with the FD sweep rows and checks
J_energy / J_speed_guard against FD's calibration (INTERFACE_v2 15.10.3). Also a slowed-genome speed-guard probe
(profile throttle_max lowered so the speed hold saturates) and the B2b AR / area lock.

  EVOLUTION_FD_DIR=/workspace/flight-sim-team/evolution/_fd_pin_p3b2a $PY -B evolution/analysis/p3b2a_energy_sanity.py
  -> evolution/analysis/p3b2a_energy_sanity.json
"""
import json, math, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)
from evolution import fidelity as F, sim  # noqa: E402

fe = F.fd_modules()["fe"]
pb2 = F.fd_b2_modules()["pb2"]
fe.PHASE1_CONFIG = os.path.join(TEAM, "evolution", "configs", "phase1.json")  # pin path resolves relative to its own parent
fe.default_profile.__defaults__ = (None, fe.PHASE1_CONFIG)
SW = os.path.join(TEAM, "flight-dynamics", "_scratch", "p3b2")
CAL = json.load(open(os.path.join(TEAM, "flight-dynamics", "v2_results", "p3b2a_energy_calibration.json")))


def gains(m):
    S = json.load(open(os.path.join(TEAM, "evolution", "runs", "phase1-s1", "summary.json")))["aircraft"]
    return next(a["best_gains"] for a in S if a["aircraft"] == m)


def profile_d(m, **over):
    P = fe.default_profile(m, sim)
    d = dict(P.to_dict(), aircraft_root=os.path.join(F.FD_DIR, "jsbsim_root"))
    d.update(over)
    return d


def cases(m):
    G = {g.name: g.lo_hi(m) for g in pb2.SHAPE_GENES_B2}
    tlo, thi = G["wing_tc_root_scale"]; qlo, qhi = G["wing_tc_tip_ratio"]
    clo, chi = G["wing_camber_root_delta_pct"]; dlo, dhi = G["wing_dihedral_delta_deg"]
    return [("baseline", {}), ("thick_both", {"wing_tc_root_scale": thi, "wing_tc_tip_ratio": qhi}),
            ("tc_hi", {"wing_tc_root_scale": thi}), ("tc_lo", {"wing_tc_root_scale": tlo}),
            ("cam_lo", {"wing_camber_root_delta_pct": clo, "wing_camber_tip_delta_pct": clo}),
            (f"dih{dhi:+g}", {"wing_dihedral_delta_deg": float(dhi)})]


def fd_row(m, lab):
    fn = os.path.join(SW, f"sweep_{m}.jsonl")
    if not os.path.exists(fn):
        return None
    return next((json.loads(l) for l in open(fn) if json.loads(l)["label"] == lab), None)


def terms_share(r):
    t = dict(r["terms"]); et = r.get("energy_terms") or {}
    tot = float(r["cost"])
    sh = {k: v / tot for k, v in t.items() if v and tot}
    for k in F.ENERGY_TERM_KEYS:
        if et.get(k):
            sh[k] = et[k] / tot
    return dict(sorted(sh.items(), key=lambda kv: -abs(kv[1])))


def main():
    out = {"fd_dir": F.FD_DIR, "w_E": F.ENERGY_W, "aircraft": {}}
    for m in ("c172x", "T38", "737"):
        pd = profile_d(m)
        scs = [s.to_dict() for s in fe.phase1_scenarios(m, sim)]
        g = gains(m)
        res = {}
        for lab, sh in cases(m):
            t0 = time.process_time()
            off = None
            if not any(k.startswith("wing_tc") for k in sh):
                off = F.evaluate_genome(pd, g, {}, scs, F.B2, None, shape=sh)
            on = F.evaluate_genome(pd, g, {}, scs, F.B2, None, shape=sh, energy_cost=True)
            fd = fd_row(m, lab)
            cal = (CAL["aircraft"].get(m) or {}).get("probes", {}).get(lab)
            et = on["energy_terms"]
            rec = {"genes": sh, "status": on["status"], "model_version": on["model_version"],
                   "cost_fd_raw": et["cost_fd"], "cost_energy_on": on["cost"],
                   "cost_energy_off": None if off is None else off["cost"],
                   "off_equals_fd_raw": None if off is None else off["cost"] == et["cost_fd"],
                   "J_energy": et["J_energy"], "J_speed_guard": et["J_speed_guard"],
                   "energy_drag_increment": et["energy_drag_increment"],
                   "speed_deficit_kts_mean": et["speed_deficit_kts_mean"],
                   "per_scenario_J": et["per_scenario"],
                   "term_share_of_total": terms_share(on), "cpu_s": time.process_time() - t0}
            if fd is not None:
                rec["fd_sweep_cost"] = fd["cost"]
                rec["fd_raw_cost_bit_identical_vs_fd_sweep"] = fd["cost"] == et["cost_fd"]
                rec["fd_sweep_energy_drag_increment"] = fd["energy"].get("energy_drag_increment")
            if cal is not None:
                rec["fd_cal_J_energy"] = cal["J_energy"]; rec["fd_cal_J_speed"] = cal["J_speed"]
                rec["J_energy_minus_fd_cal"] = et["J_energy"] - cal["J_energy"]
            res[lab] = rec
            print(m, lab, on["status"], round(et["cost_fd"], 6), round(on["cost"], 6), "J_E", round(et["J_energy"], 5),
                  "J_S", round(et["J_speed_guard"], 5), flush=True)
        out["aircraft"][m] = res
    # slowed-genome speed guard: c172x baseline with throttle_max lowered -> speed hold saturates
    m = "c172x"; scs = [s.to_dict() for s in fe.phase1_scenarios(m, sim)]
    slow = {}
    for tm in (0.85, 0.75, 0.65):
        try:
            r = F.evaluate_genome(profile_d(m, throttle_max=tm), gains(m), {}, scs, F.B2, None, shape={}, energy_cost=True)
            et = r["energy_terms"]
            slow[str(tm)] = {"status": r["status"], "cost_fd": et["cost_fd"], "cost_on": r["cost"],
                             "speed_deficit_kts_mean": et["speed_deficit_kts_mean"], "J_speed_guard": et["J_speed_guard"],
                             "J_energy": et["J_energy"], "per_scenario_J": et["per_scenario"]}
        except Exception as e:  # noqa: BLE001
            slow[str(tm)] = {"error": f"{type(e).__name__}: {e}"}
        print("slow", tm, slow[str(tm)], flush=True)
    out["speed_guard_probe_c172x_throttle_max"] = slow
    # B2b AR / area lock + thickness lock without energy
    locks = {}
    for k, v in (("wing_aspect_scale", 1.1), ("wing_area_scale", 1.1)):
        try:
            F.shape_from_b2({k: v}, "c172x", energy_cost=True); locks[k] = "ACCEPTED (unexpected)"
        except ValueError as e:
            locks[k] = f"rejected: {e}"
    try:
        F.shape_from_b2({"wing_tc_root_scale": 1.2}, "c172x", energy_cost=False); locks["tc_without_energy"] = "ACCEPTED (unexpected)"
    except ValueError as e:
        locks["tc_without_energy"] = f"rejected: {e}"
    out["locks"] = locks
    json.dump(out, open(os.path.join(HERE, "p3b2a_energy_sanity.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
