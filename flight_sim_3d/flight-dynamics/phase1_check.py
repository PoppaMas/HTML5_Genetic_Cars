"""Verify trim settings, trimmable envelope, model CLmax/stall, control conventions for Phase-1 aircraft.

Writes phase1_check.json. Everything here is measured from the JSBSim 1.3.1 models (no guessing).
"""
import json
import math
import os
import tempfile

import numpy as np
import jsbsim

jsbsim.FGJSBBase().debug_lvl = 0
HERE = os.path.dirname(os.path.abspath(__file__))
DT = 1 / 120
RHO0 = 0.0023769

AC = {
    "c172x": dict(design=(100.0, 4000.0), speeds=range(40, 161, 5), alts=range(0, 20001, 1000), clmax_kcas=100.0),
    "T38": dict(design=(300.0, 10000.0), speeds=range(120, 701, 10), alts=range(0, 60001, 2500), clmax_kcas=300.0),
    "737": dict(design=(250.0, 10000.0), speeds=range(120, 421, 10), alts=range(0, 45001, 2500), clmax_kcas=250.0),
}


def new(model):
    f = jsbsim.FGFDMExec(None); f.set_debug_level(0); f.set_output_path(tempfile.gettempdir())
    f.load_model(model); f.disable_output(); f.set_dt(DT)
    return f


def trim(model, kcas, alt, mode=1, gear_up=True):
    f = new(model)
    if gear_up:
        f["gear/gear-cmd-norm"] = 0.0
    f["ic/h-sl-ft"] = alt; f["ic/vc-kts"] = kcas; f["ic/gamma-deg"] = 0.0; f["ic/psi-true-deg"] = 0.0
    f.run_ic()
    f["propulsion/set-running"] = -1
    n = f.get_propulsion().get_num_engines()
    for i in range(n):
        f[f"fcs/mixture-cmd-norm[{i}]"] = 1.0
    try:
        f["simulation/do_simple_trim"] = mode
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}"
    return f, None


def snap(f):
    keys = {"alpha_deg": "aero/alpha-deg", "theta_deg": "attitude/theta-deg", "throttle_cmd": "fcs/throttle-cmd-norm",
            "throttle_pos": "fcs/throttle-pos-norm", "elevator_cmd": "fcs/elevator-cmd-norm", "pitch_trim_cmd": "fcs/pitch-trim-cmd-norm",
            "elevator_pos_deg": "fcs/elevator-pos-deg", "gear_pos": "gear/gear-pos-norm", "flap_pos_deg": "fcs/flap-pos-deg",
            "mach": "velocities/mach", "qbar_psf": "aero/qbar-psf", "weight_lb": "inertia/weight-lbs",
            "thrust_lbf_eng0": "propulsion/engine/thrust-lbs", "udot": "accelerations/udot-ft_sec2",
            "wdot": "accelerations/wdot-ft_sec2", "qdot": "accelerations/qdot-rad_sec2"}
    out = {k: round(float(f[p]), 4) for k, p in keys.items()}
    for extra in ("propulsion/engine/n2", "propulsion/engine/engine-rpm"):
        try:
            out[extra.split("/")[-1]] = round(float(f[extra]), 2)
        except Exception:  # noqa: BLE001
            pass
    return out


def held(f, seconds=10.0):
    th0, v0, h0 = f["attitude/theta-deg"], f["velocities/vc-kts"], f["position/h-sl-ft"]
    for _ in range(int(seconds / DT)):
        f.run()
    return {"dtheta_deg": round(f["attitude/theta-deg"] - th0, 3), "dkcas": round(f["velocities/vc-kts"] - v0, 3),
            "dh_ft": round(f["position/h-sl-ft"] - h0, 2)}


def clmax(model, kcas, alt):
    rows = []
    for a in np.arange(-4, 40.1, 1.0):
        f = new(model)
        f["gear/gear-cmd-norm"] = 0.0
        f["ic/h-sl-ft"] = alt; f["ic/vc-kts"] = kcas; f["ic/alpha-deg"] = float(a); f["ic/gamma-deg"] = 0.0
        f.run_ic()
        al = f["aero/alpha-rad"]
        lift = -f["forces/fbz-aero-lbs"] * math.cos(al) + f["forces/fbx-aero-lbs"] * math.sin(al)
        rows.append((float(a), lift / (f["aero/qbar-psf"] * f["metrics/Sw-sqft"]), f["inertia/weight-lbs"], f["metrics/Sw-sqft"]))
    i = int(np.argmax([r[1] for r in rows]))
    a, cl, w, s = rows[i]
    vs = math.sqrt(2 * w / (RHO0 * s * cl)) / 1.6878
    return {"alpha_clmax_deg": a, "clmax_untrimmed": round(cl, 3), "vs1g_keas_from_clmax": round(vs, 1),
            "note": "CL from JSBSim aero at IC (elevator 0, power on, gear up), alpha swept -4..40 deg; peak may be the table edge"}


def response_signs(model, kcas, alt):
    out = {}
    for name, props, extra in (("elevator", ["fcs/elevator-cmd-norm"], None), ("aileron", ["fcs/aileron-cmd-norm"], None),
                               ("rudder", ["fcs/rudder-cmd-norm"], None), ("throttle", None, None)):
        f, err = trim(model, kcas, alt)
        n = f.get_propulsion().get_num_engines()
        ps = props or [f"fcs/throttle-cmd-norm[{i}]" for i in range(n)]
        base = {p: f[p] for p in ps}
        q0, p0, r0, t0 = f["velocities/q-rad_sec"], f["velocities/p-rad_sec"], f["velocities/r-rad_sec"], f["propulsion/engine/thrust-lbs"]
        for _ in range(int(0.5 / DT)):
            for p in ps:
                f[p] = base[p] + 0.1
            f.run()
        out[name] = {"cmd_step": "+0.1", "dq_dps": round(math.degrees(f["velocities/q-rad_sec"] - q0), 3),
                     "dp_dps": round(math.degrees(f["velocities/p-rad_sec"] - p0), 3),
                     "dr_dps": round(math.degrees(f["velocities/r-rad_sec"] - r0), 3),
                     "dthrust_lbf": round(f["propulsion/engine/thrust-lbs"] - t0, 1)}
    # surface travel for cmd = +/-1 (positions after 1 s)
    travel = {}
    for cmd in (-1.0, 1.0):
        f, _ = trim(model, kcas, alt)
        for _ in range(int(1.0 / DT)):
            f["fcs/elevator-cmd-norm"] = cmd; f["fcs/aileron-cmd-norm"] = cmd; f["fcs/rudder-cmd-norm"] = cmd
            f.run()
        travel[f"cmd{cmd:+.0f}"] = {k: round(float(f[k]), 3) for k in (
            "fcs/elevator-pos-deg", "fcs/left-aileron-pos-deg", "fcs/right-aileron-pos-deg", "fcs/left-aileron-pos-norm",
            "fcs/right-aileron-pos-norm", "fcs/rudder-pos-deg")}
    out["surface_travel"] = travel
    return out


def main():
    res = {}
    for model, cfg in AC.items():
        kc, alt = cfg["design"]
        r = {"design_point": {"kcas": kc, "alt_ft": alt}}
        f0 = new(model)
        r["defaults_after_load"] = {"gear_cmd": f0["gear/gear-cmd-norm"], "gear_pos": f0["gear/gear-pos-norm"],
                                    "flap_cmd": f0["fcs/flap-cmd-norm"], "n_engines": f0.get_propulsion().get_num_engines(),
                                    "Sw_ft2": f0["metrics/Sw-sqft"], "b_ft": f0["metrics/bw-ft"], "cbar_ft": f0["metrics/cbarw-ft"],
                                    "empty_wt_lb": f0["inertia/empty-weight-lbs"]}
        for gear_up in (False, True):
            for mode in (1, 0):
                f, err = trim(model, kc, alt, mode, gear_up)
                key = f"trim_mode{mode}_gear_{'up' if gear_up else 'default(down)'}"
                r[key] = {"error": err} if f is None else dict(snap(f), hold10s=held(f))
        # trimmable speed range at design altitude (gear up, mode 1)
        ok = []
        for v in cfg["speeds"]:
            f, err = trim(model, float(v), alt)
            if f is not None and f["fcs/throttle-cmd-norm"] <= 1.0:
                ok.append((v, round(f["aero/alpha-deg"], 2), round(f["fcs/throttle-cmd-norm"], 3)))
        r["trimmable_kcas_at_design_alt"] = {"min": ok[0] if ok else None, "max": ok[-1] if ok else None,
                                             "n_ok": len(ok), "tried": [min(cfg["speeds"]), max(cfg["speeds"])]}
        # level-flight ceiling in the model (any speed in the list trims)
        ceil = None
        for h in cfg["alts"]:
            hit = None
            for v in list(cfg["speeds"])[::2]:
                f, err = trim(model, float(v), float(h))
                if f is not None:
                    hit = v; break
            if hit is None and h > alt:
                break
            if hit is not None:
                ceil = (h, hit)
        r["level_trim_ceiling_ft"] = ceil
        r["clmax"] = clmax(model, cfg["clmax_kcas"], alt)
        r["control_response"] = response_signs(model, kc, alt)
        res[model] = r
        print(model, json.dumps({k: r[k] for k in ("trimmable_kcas_at_design_alt", "level_trim_ceiling_ft", "clmax")}))
    json.dump(res, open(os.path.join(HERE, "phase1_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
