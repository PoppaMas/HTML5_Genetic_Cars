"""Run fmq_worker.py over bundled JSBSim aircraft in parallel subprocesses; write fm_quality.csv/.jsonl.

Re-uses $FLIGHT_SIM_PLAN_DIR (the external flight-sim-plan/ folder, not in this repo; read-only): cases.txt for the nominal trim points,
inventory_raw.json for engine/thrust data and flytest_results.jsonl for the 10 s hold check.
"""
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.environ.get("FLIGHT_SIM_PLAN_DIR", "flight-sim-plan")  # external planning folder (cases.txt), not in this repo
PY = sys.executable
WORKERS = int(os.environ.get("FMQ_WORKERS", "6"))


def cases():
    out = []
    for line in open(os.path.join(PLAN, "cases.txt")):
        f = line.split()
        if len(f) >= 3 and "glide" not in f[3:]:
            out.append((f[0], f[1], float(f[2])))
    return out


def run_one(c):
    m, k, a = c
    t0 = time.time()
    try:
        p = subprocess.run([PY, os.path.join(HERE, "fmq_worker.py"), m, k, str(a)], capture_output=True, text=True,
                           timeout=400, cwd="/tmp")
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if lines:
            r = json.loads(lines[-1])
        else:
            r = {"model": m, "load": "crashed", "load_err": f"exit {p.returncode}: {p.stderr.strip()[-200:]}"}
    except subprocess.TimeoutExpired:
        r = {"model": m, "load": "timeout", "load_err": "worker exceeded 400 s"}
    r["worker_wall_s"] = round(time.time() - t0, 1)
    print(f"{m:16s} {r.get('trim_method', r.get('load'))} {r.get('doublet_status', '')} {r['worker_wall_s']}s", flush=True)
    return r


def rate(r, inv):
    """good-to-evolve / usable-with-caveats / avoid, with reasons."""
    why, cav = [], []
    if r.get("load") != "ok":
        return "avoid", [f"does not load: {r.get('load_err', r.get('load'))}"]
    tm = r.get("trim_method", "failed")
    if tm == "failed":
        return "avoid", ["no trim found (built-in ladder and engine-off Newton glide trim failed)"]
    if r.get("doublet_status") != "ok":
        why.append(f"elevator doublet (A=0.1, wing leveler on): {r.get('doublet_status')}"
                   + (f", q oscillation grows x{r['q_decay_ratio']}" if r.get("q_decay_ratio", 0) > 1 else ""))
    if not r.get("deterministic", False):
        why.append("non-deterministic")
    if why:
        return "avoid", why
    if tm.startswith("newton-glide"):
        cav.append("built-in powered trim fails (engine/propeller model gives no usable thrust response); only engine-off glide trim")
    elif r.get("trim_alt_ft") != r.get("alt_nominal_ft") or r.get("trim_kcas") not in r.get("speeds_tried", [r.get("kcas_nominal")]):
        cav.append(f"trims at {r['trim_kcas']:.0f} KCAS/{r['trim_alt_ft']:.0f} ft, not the nominal point")
    if r.get("created_props"):
        cav.append("needs FlightGear-only properties created: " + ", ".join(r["created_props"]))
    a = r.get("alpha_deg", 0.0)
    if not (-3.0 <= a <= 10.0):
        cav.append(f"odd trim alpha {a} deg")
    thr = r.get("throttle", 0.5)
    if tm.startswith("builtin") and thr > 0.95:
        cav.append(f"trim throttle {thr} (little thrust margin)")
    if not r.get("pstep_pass", False):
        cav.append(f"generic auto-scaled pitch-hold step did not pass ({r.get('pstep_status')}, final err {r.get('pstep_final_err_deg')})")
    if r.get("q_decay_ratio", 0) > 0.8:
        cav.append(f"lightly damped pitch response (q decay ratio {r['q_decay_ratio']})")
    if abs(r.get("hold10_phi_deg", 0)) > 10:
        cav.append(f"rolls off {r['hold10_phi_deg']} deg in 10 s with fixed controls (needs a wing leveler)")
    if r.get("rtf_loop", 1e9) < 50:
        cav.append(f"slow: RTF {r.get('rtf_loop')}")
    rel = (inv.get("release") or "").upper()
    if rel == "ALPHA":
        cav.append("model marked ALPHA release by its author")
    if not inv.get("aero_complete_6axis", True):
        cav.append("incomplete 6-axis aero")
    return ("good-to-evolve" if not cav else "usable-with-caveats"), cav


def main():
    inv = {d["model"]: d for d in json.load(open(os.path.join(PLAN, "inventory_raw.json")))}
    cat = {}
    if os.path.exists(os.path.join(PLAN, "aircraft_inventory.csv")):
        for row in csv.DictReader(open(os.path.join(PLAN, "aircraft_inventory.csv"))):
            cat[row["model"]] = row
    fly = {}
    if os.path.exists(os.path.join(PLAN, "flytest_results.jsonl")):
        for l in open(os.path.join(PLAN, "flytest_results.jsonl")):
            d = json.loads(l); fly[d["model"]] = d
    cs = cases()
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(run_one, cs))
    rows = []
    for r in results:
        i = inv.get(r["model"], {})
        rating, reasons = rate(r, i)
        r["rating"] = rating
        r["reasons"] = reasons
        ft = fly.get(r["model"], {})
        r["flytest_trim"] = ft.get("trim")
        r["flytest_fly10s_pass"] = ft.get("fly10s_pass")
        r["category"] = cat.get(r["model"], {}).get("category")
        r["real_type"] = cat.get(r["model"], {}).get("real_type")
        r["engine_kinds"] = i.get("engine_kinds")
        r["total_static_thrust_lbf"] = i.get("total_static_thrust_lbf")
        r["total_power_hp"] = i.get("total_power_hp")
        r["afterburner"] = i.get("afterburner")
        r["release"] = i.get("release")
        rows.append(r)
    with open(os.path.join(HERE, "fm_quality.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cols = ["model", "rating", "category", "real_type", "engine_kinds", "total_static_thrust_lbf", "total_power_hp", "afterburner", "release",
            "load", "n_engines", "trim_method", "trim_kcas", "trim_alt_ft", "mach", "weight_lb", "alpha_deg", "theta_deg", "gamma_deg", "throttle", "throttle_pos", "gear_cmd", "gear_pos",
            "pitch_trim", "resid_udot", "resid_wdot", "resid_qdot_dps2", "hold10_dtheta_deg", "hold10_dkcas", "hold10_phi_deg",
            "doublet_status", "doublet_peak_q_dps", "doublet_peak_nz", "doublet_min_nz", "doublet_max_dtheta", "doublet_max_phi",
            "doublet_dh_ft", "q_decay_ratio", "pitch_rate_gain_dps_per_unit", "pstep_status", "pstep_pass", "pstep_overshoot_deg",
            "pstep_t90_s", "pstep_final_err_deg", "pstep_peak_nz", "deterministic", "rtf_pure", "rtf_loop",
            "flytest_trim", "flytest_fly10s_pass", "created_props", "trim_errors", "load_err", "reasons"]
    order = {"good-to-evolve": 0, "usable-with-caveats": 1, "avoid": 2}
    rows.sort(key=lambda r: (order[r["rating"]], r["model"].lower()))
    with open(os.path.join(HERE, "fm_quality.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([("; ".join(map(str, r.get(c))) if isinstance(r.get(c), list) else r.get(c, "")) for c in cols])
    for r in rows:
        print(f"{r['model']:16s} {r['rating']:20s} {' | '.join(r['reasons'])[:150]}")


if __name__ == "__main__":
    main()
