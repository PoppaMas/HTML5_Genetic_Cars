"""F-16 (JSBSim f16.xml, prepared copy in ./jsbsim_root) verification for Evolution Runner's trim point 350 KCAS / 10 000 ft.

Writes f16_check.json. Everything is measured from JSBSim 1.3.1 (no guessing):
  * prepared-copy checks: no network sockets, placeholder point mass index 0, wing point masses, external reactions,
    zero-force patched copy bit-identical to the stock model;
  * trims (fgtrim mode 1 = full, mode 0 = longitudinal; gear up and gear default) + 20 s hands-off hold;
  * throttle mapping (fcs/throttle-pos-norm = 2 x cmd; MIL vs afterburner);
  * trimmable speed range at 10 000 ft and level-trim ceiling (mode 1, gear up);
  * control response signs at the design point;
  * pitch step (rigid and two-way flex), each run twice -> bit-identical check;
  * flex margins at default genes and at the tied-grid corners.
"""
import json, math, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flexwing as fw  # noqa: E402
import coupled_sim as cs  # noqa: E402

ROOT = os.path.join(HERE, "jsbsim_root")
M = "f16"
KCAS, ALT = 350.0, 10000.0
DT = 1 / 120
SNAP = {"alpha_deg": "aero/alpha-deg", "theta_deg": "attitude/theta-deg", "throttle_cmd": "fcs/throttle-cmd-norm",
        "throttle_pos": "fcs/throttle-pos-norm", "thrust_lbf": "propulsion/engine/thrust-lbs", "fuel_flow_pps": "propulsion/engine/fuel-flow-rate-pps",
        "n2": "propulsion/engine/n2", "elevator_cmd": "fcs/elevator-cmd-norm", "pitch_trim_cmd": "fcs/pitch-trim-cmd-norm",
        "elevator_pos_norm": "fcs/elevator-pos-norm", "elevator_pos_deg": "fcs/elevator-pos-deg", "g_load_pid": "fcs/g-load-pid",
        "pitch_trim_error": "fcs/pitch-trim-error", "aileron_pos_rad": "fcs/aileron-pos-rad", "rudder_pos_norm": "fcs/rudder-pos-norm",
        "gear_pos": "gear/gear-pos-norm", "lef_pos_deg": "fcs/lef-pos-deg", "tef_pos_rad": "fcs/tef-pos-rad",
        "speedbrake_pos_norm": "fcs/speedbrake-pos-norm", "mach": "velocities/mach", "kcas": "velocities/vc-kts", "qbar_psf": "aero/qbar-psf",
        "weight_lb": "inertia/weight-lbs", "udot": "accelerations/udot-ft_sec2", "wdot": "accelerations/wdot-ft_sec2",
        "qdot": "accelerations/qdot-rad_sec2", "nz": "accelerations/Nz"}


def snap(f):
    return {k: round(float(f[p]), 5) for k, p in SNAP.items()}


def trim(kcas=KCAS, alt=ALT, mode=1, gear_up=True, root=ROOT):
    f = fw.new_fdm(M, root, DT)
    if gear_up:
        f["gear/gear-cmd-norm"] = 0.0
    f["ic/h-sl-ft"] = alt; f["ic/vc-kts"] = kcas; f["ic/gamma-deg"] = 0.0; f["ic/psi-true-deg"] = 0.0
    f.run_ic()
    f["propulsion/set-running"] = -1
    try:
        f["simulation/do_simple_trim"] = mode
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    return f, None


def hold(f, seconds=20.0):
    th0, v0, h0, a0 = f["attitude/theta-deg"], f["velocities/vc-kts"], f["position/h-sl-ft"], f["aero/alpha-deg"]
    for _ in range(int(round(seconds / DT))):
        f.run()
    return {"seconds": seconds, "dtheta_deg": round(f["attitude/theta-deg"] - th0, 4), "dalpha_deg": round(f["aero/alpha-deg"] - a0, 4),
            "dkcas": round(f["velocities/vc-kts"] - v0, 4), "dh_ft": round(f["position/h-sl-ft"] - h0, 3),
            "phi_deg": round(f["attitude/phi-deg"], 4), "gear_pos": f["gear/gear-pos-norm"]}


def prepared_copy_checks():
    xml = open(os.path.join(ROOT, "aircraft", M, M + ".xml")).read()
    meta = json.load(open(os.path.join(ROOT, "aircraft", M, "flexwing_meta.json")))
    mb = re.search(r"<mass_balance[^>]*>(.*?)</mass_balance>", xml, re.S).group(1)
    names = re.findall(r'<pointmass\s+name="([^"]+)"', mb)
    f = fw.new_fdm(M, ROOT)
    net = subprocess.run(["grep", "-rniE", fw.NET_IO_GREP, os.path.join(ROOT, "aircraft", M)], capture_output=True, text=True).stdout
    return {"meta": meta, "pointmass_order": names, "network_io_grep": net.strip() or "none",
            "external_reactions": [n for n in re.findall(r'<(?:force|moment)\s+name="([^"]+)"', xml)],
            "pointmass_weights_lb": [f[f"inertia/pointmass-weight-lbs[{i}]"] for i in range(len(names))],
            "defaults_after_load": {"gear_cmd": f["gear/gear-cmd-norm"], "gear_pos": f["gear/gear-pos-norm"],
                                    "Sw_ft2": f["metrics/Sw-sqft"], "b_ft": f["metrics/bw-ft"], "cbar_ft": f["metrics/cbarw-ft"],
                                    "empty_wt_lb": f["inertia/empty-weight-lbs"], "n_engines": f.get_propulsion().get_num_engines()}}


def step_history(root, flex=False, d_elev=-0.10, t_on=1.0, t_off=2.0, duration=8.0):
    """Trim (mode 1, gear up) then elevator-cmd step d_elev for t_on..t_off s, hands off otherwise."""
    f = fw.new_fdm(M, root, DT)
    coupler = cs.make_coupler(f, M, "twoway", None, 4, root) if flex else None
    cs.trim(f, ALT, KCAS)
    px = cs.FlexFDM(f, coupler, DT, record=False)
    e0 = f["fcs/elevator-cmd-norm"]
    keys = ("velocities/q-rad_sec", "accelerations/Nz", "aero/alpha-deg", "attitude/theta-deg", "fcs/elevator-pos-deg",
            "velocities/vc-kts", "position/h-sl-ft", "forces/fbz-external-lbs", "moments/m-external-lbsft")
    out = []
    for k in range(int(round(duration / DT))):
        t = k * DT
        f["fcs/elevator-cmd-norm"] = e0 + (d_elev if t_on <= t < t_off else 0.0)
        px.run()
        out.append([f[x] for x in keys])
    return np.array(out)


def step_summary(h):
    q = np.degrees(h[:, 0]); nz = h[:, 1]
    t = (np.arange(len(h)) + 1) * DT
    return {"q_peak_dps": round(float(q.max()), 3), "t_q_peak_s": round(float(t[q.argmax()]), 3),
            "nz_peak": round(float(nz.max()), 4), "nz_min": round(float(nz.min()), 4),
            "alpha_peak_deg": round(float(h[:, 2].max()), 3), "elev_pos_range_deg": [round(float(h[:, 4].min()), 3), round(float(h[:, 4].max()), 3)],
            "q_at_end_dps": round(float(q[-1]), 4), "nz_at_end": round(float(nz[-1]), 4),
            "dtheta_end_deg": round(float(h[-1, 3] - h[0, 3]), 3), "dkcas_end": round(float(h[-1, 5] - h[0, 5]), 3),
            "dh_end_ft": round(float(h[-1, 6] - h[0, 6]), 2),
            "max_abs_ext_fz_lbf": round(float(np.abs(h[:, 7]).max()), 2), "max_abs_ext_m_lbft": round(float(np.abs(h[:, 8]).max()), 2)}


def zero_force_vs_stock():
    def hist(root):
        f, _ = trim(root=root)
        e0 = f["fcs/elevator-cmd-norm"]; out = []
        for k in range(720):
            f["fcs/elevator-cmd-norm"] = e0 + (-0.1 if 120 <= k < 240 else 0.0)
            f["fcs/aileron-cmd-norm"] = 0.3 if 360 <= k < 480 else 0.0
            f.run()
            out.append([f[x] for x in ("position/h-sl-ft", "attitude/theta-deg", "attitude/phi-deg", "velocities/vc-kts",
                                       "velocities/p-rad_sec", "velocities/q-rad_sec", "inertia/weight-lbs", "inertia/cg-x-in")])
        return np.array(out)
    return bool(np.array_equal(hist(None), hist(ROOT)))


def throttle_map():
    rows = []
    for cmd in (0.25, 0.45, 0.50, 0.51, 0.60, 0.75, 1.00):
        f, _ = trim()
        f["fcs/throttle-cmd-norm"] = cmd
        for _ in range(int(8 / DT)):
            f.run()
        thr = f["propulsion/engine/thrust-lbs"]
        rows.append({"throttle_cmd": cmd, "throttle_pos": round(f["fcs/throttle-pos-norm"], 4), "thrust_lbf_after_8s": round(thr, 1),
                     "n2": round(f["propulsion/engine/n2"], 2), "fuel_flow_pps": round(f["propulsion/engine/fuel-flow-rate-pps"], 3),
                     # FGTurbine augmethod 2: throttle-pos > 1 -> augmentation on, AB level = pos - 1 (see fuel-flow jump)
                     "afterburner": bool(f["fcs/throttle-pos-norm"] > 1.0 + 1e-9)})
    return rows


def envelope():
    ok = []
    for v in range(100, 801, 10):
        f, err = trim(float(v))
        if f is not None and f["fcs/throttle-cmd-norm"] <= 1.0:
            ok.append((v, round(f["aero/alpha-deg"], 2), round(f["fcs/throttle-cmd-norm"], 3), round(f["velocities/mach"], 3)))
    mil = [r for r in ok if r[2] <= 0.5]
    ceil = None
    for h in range(0, 60001, 2500):
        hit = None
        for v in range(100, 801, 20):
            f, err = trim(float(v), float(h))
            if f is not None and f["fcs/throttle-cmd-norm"] <= 1.0:
                hit = (v, round(f["fcs/throttle-cmd-norm"], 3), round(f["velocities/mach"], 3)); break
        if hit is None and h > ALT:
            break
        if hit is not None:
            ceil = (h, *hit)
    return {"trimmable_kcas_at_10kft": {"min": ok[0] if ok else None, "max": ok[-1] if ok else None, "n_ok": len(ok),
                                        "tried": [100, 800], "row": "(kcas, alpha_deg, throttle_cmd, mach)"},
            "mil_power_kcas_range_at_10kft": [mil[0][0], mil[-1][0]] if mil else None,
            "level_trim_ceiling": {"alt_ft": ceil[0], "first_kcas": ceil[1], "throttle_cmd": ceil[2], "mach": ceil[3]} if ceil else None,
            "table": ok}


def response_signs():
    out = {}
    for name, prop in (("elevator", "fcs/elevator-cmd-norm"), ("aileron", "fcs/aileron-cmd-norm"), ("rudder", "fcs/rudder-cmd-norm"),
                       ("throttle", "fcs/throttle-cmd-norm")):
        f, _ = trim(); b = f[prop]
        q0, p0, r0, t0 = (f["velocities/q-rad_sec"], f["velocities/p-rad_sec"], f["velocities/r-rad_sec"], f["propulsion/engine/thrust-lbs"])
        surf0 = {k: f[k] for k in ("fcs/elevator-pos-deg", "fcs/aileron-pos-rad", "fcs/left-aileron-pos-rad", "fcs/right-aileron-pos-rad",
                                   "fcs/rudder-pos-rad")}
        for _ in range(int(0.5 / DT)):
            f[prop] = b + 0.1; f.run()
        out[name] = {"cmd_step": "+0.1 for 0.5 s", "dq_dps": round(math.degrees(f["velocities/q-rad_sec"] - q0), 3),
                     "dp_dps": round(math.degrees(f["velocities/p-rad_sec"] - p0), 3), "dr_dps": round(math.degrees(f["velocities/r-rad_sec"] - r0), 3),
                     "dthrust_lbf": round(f["propulsion/engine/thrust-lbs"] - t0, 1),
                     "dsurf": {k.split("/")[-1]: round(f[k] - v, 5) for k, v in surf0.items()}}
    return out


def coupled_maneuver():
    """coupled_sim.run_maneuver (elevator pull 1-2.5 s, aileron doublet 4-6 s, 1-cos gust 8-9.5 s) rigid vs flex."""
    out = {}
    for name, use, ov in (("rigid", False, None), ("flex_baseline", True, None), ("flex_s0.6_r1.0", True, fw.tied_stiffness(0.6, 1.0))):
        r = cs.run_maneuver(M, mode="twoway", overrides=ov, use_coupler=use, root=ROOT)
        d = {"p_max_dps": round(float(np.max(np.abs(r["p_dps"]))), 3), "nz_max": round(float(np.max(r["nz"])), 4),
             "nz_min": round(float(np.min(r["nz"])), 4), "q_max_dps": round(float(np.max(r["q_dps"])), 3)}
        if use:
            bm = np.maximum(np.abs(r["root_bm_lbft_R"]), np.abs(r["root_bm_lbft_L"]))
            d.update(root_bm_peak_lbft=round(float(bm.max()), 1), m_root_1g_lbft=round(float(r["m_root_1g"]), 1),
                     tip_w_max_ft=round(float(np.max(np.abs(np.r_[r["tip_w_ft_R"], r["tip_w_ft_L"]]))), 4),
                     tip_twist_max_deg=round(float(np.max(np.abs(np.r_[r["tip_twist_deg_R"], r["tip_twist_deg_L"]]))), 4))
        out[name] = d
    return out


def margins():
    f = fw.new_fdm(M, ROOT)
    S, B, W = f["metrics/Sw-sqft"], f["metrics/bw-ft"], f["inertia/empty-weight-lbs"]
    keys = ("flutter_margin", "flutter_margin_qs", "coalescence_margin", "div_margin", "flutter_not_found_below_cap",
            "div_not_found_below_cap", "margin_error", "f_flutter_hz", "f_modes_hz")
    def m(**ov):
        r = fw.FlexWing(fw.params_for(M, B, S, W, **ov)).margins()
        return {k: (round(r[k], 4) if isinstance(r[k], float) else ([round(x, 3) for x in r[k]] if isinstance(r[k], list) else r[k])) for k in keys}
    out = {"default": m(), "corners": {}}
    for s_ in (0.6, 0.8, 1.0, 2.0):
        for r_ in (0.8, 1.0, 1.15):
            out["corners"][f"s{s_}_r{r_}"] = m(**fw.tied_stiffness(s_, r_))
    alt = {}
    for ea, cg in ((0.35, 0.38), (0.38, 0.42), (0.40, 0.42), (0.40, 0.43), (0.40, 0.45), (0.42, 0.45)):
        ov = dict(x_ea=ea, x_cg=cg, cal_x_ea=ea, cal_x_cg=cg)
        alt[f"{ea:.2f}/{cg:.2f}"] = {"default": m(**ov)["flutter_margin"], "s0.6_r0.8": m(**ov, **fw.tied_stiffness(0.6, 0.8))["flutter_margin"],
                                     "s0.8_r0.8": m(**ov, **fw.tied_stiffness(0.8, 0.8))["flutter_margin"]}
    out["ea_cg_sensitivity_flutter_margin"] = alt
    return out


def main():
    cs.ensure_root(M, ROOT)
    res = {"design_point": {"kcas": KCAS, "alt_ft": ALT}, "prepared_copy": prepared_copy_checks()}
    res["zero_force_patched_identical_to_stock"] = zero_force_vs_stock()
    trims = {}
    for gear_up in (True, False):
        for mode in (1, 0):
            f, err = trim(mode=mode, gear_up=gear_up)
            key = f"mode{mode}_gear_{'up' if gear_up else 'default(down)'}"
            trims[key] = {"error": err} if f is None else dict(snap(f), hold20s=hold(f))
    f, err = trim(mode=2)
    trims["mode2_ground"] = {"error": err} if f is None else {"note": "unexpected success"}
    res["trim_350kcas_10kft"] = trims
    res["throttle_map"] = throttle_map()
    res["control_response"] = response_signs()
    h1, h2 = step_history(ROOT), step_history(ROOT)
    g1, g2 = step_history(ROOT, flex=True), step_history(ROOT, flex=True)
    res["pitch_step"] = {"input": "fcs/elevator-cmd-norm trim-0.10 for t=1..2 s, 8 s total, dt 1/120",
                         "rigid": step_summary(h1), "rigid_repeat_bit_identical": bool(np.array_equal(h1, h2)),
                         "flex_twoway": step_summary(g1), "flex_repeat_bit_identical": bool(np.array_equal(g1, g2))}
    res["coupled_maneuver"] = coupled_maneuver()
    res["envelope"] = envelope()
    res["margins"] = margins()
    json.dump(res, open(os.path.join(HERE, "f16_check.json"), "w"), indent=1)
    t = trims["mode1_gear_up"]
    print(json.dumps({"trim_mode1": {k: t.get(k) for k in ("alpha_deg", "throttle_cmd", "throttle_pos", "elevator_pos_deg", "pitch_trim_cmd", "hold20s")},
                      "trim_mode0_alpha": trims["mode0_gear_up"].get("alpha_deg"), "zero_force": res["zero_force_patched_identical_to_stock"],
                      "pitch_step": res["pitch_step"], "coupled_maneuver": res["coupled_maneuver"], "envelope": {k: v for k, v in res["envelope"].items() if k != "table"},
                      "margins_default": res["margins"]["default"]}, indent=1))


if __name__ == "__main__":
    main()
