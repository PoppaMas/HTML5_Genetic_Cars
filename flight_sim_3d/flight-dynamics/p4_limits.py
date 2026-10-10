"""p4_limits.py -- writes v2_results/p4_aircraft_limits.json (PHASE4_FD_CONTROL_SURFACES.md P4.7). Rigid, baseline
aircraft (<root>_v2, B2 defaults) at the Phase-1 trim point. Estimates are labelled."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import flexeval as fe, flexbody as fb, flexwing as fw, coupled_sim as cs, planform_b2 as pb2, ctrlsurf_p4 as cs4

G = 32.174
E_OSW = 0.8


def drag(f):
    a = f["aero/alpha-rad"]
    return -f["forces/fbx-aero-lbs"] * math.cos(a) - f["forces/fbz-aero-lbs"] * math.sin(a)


def limits(m, sim):
    rv2 = fe.root_v2_for(cs.ROOT); fb.ensure_root_v2(m, rv2)
    P = fe._profile_with_root(fe.default_profile(m, sim), sim, rv2, m)
    sc = fe.phase1_scenarios(m, sim)[0]
    geo = fb.geometry_for(m, rv2); pw = fw.params_for(m, geo.bw_ft, geo.sw_ft2, geo.empty_wt_lb)
    fdm, info = sim.trim(P, sc)
    V, q, W, S, b = fdm["velocities/vt-fps"], fdm["aero/qbar-psf"], fdm["inertia/weight-lbs"], geo.sw_ft2, geo.bw_ft
    D0 = drag(fdm); CL = W / (q * S); AR = b * b / S
    Di = q * S * CL ** 2 / (math.pi * E_OSW * AR)
    ne = fdm.get_propulsion().get_num_engines()
    th0 = fdm["fcs/throttle-cmd-norm"]
    def thrust():
        return sum(fdm[f"propulsion/engine[{i}]/thrust-lbs"] for i in range(ne))
    T_trim = thrust()
    for i in range(ne): fdm[f"fcs/throttle-cmd-norm[{i}]"] = P.throttle_max
    for _ in range(int(6 / sim.DT)):   # spool-up, speed drifts a little (estimate)
        fdm["fcs/elevator-cmd-norm"] = info["elev_trim"]; fdm.run()
    T_max = thrust()
    clmax = pb2.SECTIONS[m]["clmax0"]
    n_aero = q * S * clmax / W
    n_inst = min(pw.n_limit, n_aero, P.nz_limits[1])
    D_par = max(D0 - Di, 0.0)
    n_sus = math.sqrt(max((T_max - D_par) / Di, 1.0))
    n_sus = min(n_sus, n_inst)
    R = lambda n: V * V / (G * math.sqrt(n * n - 1)) if n > 1.0001 else math.inf  # noqa: E731
    # roll / pitch step (rigid, FD actuator active) from a fresh trim
    sd = {s.name: s for s in cs4.SURFACES[m]}
    out_rate = {}
    for nm, prop, key in (("ail", "fcs/aileron-cmd-norm", "velocities/p-rad_sec"), ("elev", "fcs/elevator-cmd-norm", "velocities/q-rad_sec")):
        f2, inf2 = sim.trim(P, sc)
        act = cs4.Actuator(sd[nm], sim.DT, f2[prop]); pk = 0.0; nzpk = 1.0
        tgt = 1.0 if nm == "ail" else -1.0
        for k in range(int((3.0 if nm == "ail" else 1.0) / sim.DT)):
            f2[prop] = act.step(tgt if nm == "ail" else inf2["elev_trim"] - 0.5)
            f2.run(); pk = max(pk, abs(f2[key])); nzpk = max(nzpk, f2["accelerations/Nz"])
            if nm == "ail" and abs(f2["attitude/phi-deg"]) > 90: break
        out_rate[nm] = (math.degrees(pk), nzpk, math.copysign(1, f2[key]) if nm == "ail" else None)
    ROC = (T_max - D0) * V / W * 60
    sb = next((s for s in cs4.SURFACES[m] if s.name == "sbrk"), None)
    return {
        "trim": {"h_ft": sc.h0_ft, "kcas": sc.speed_kts, "vt_fps": V, "ktas": V / 1.68781, "alpha_deg": fdm["aero/alpha-deg"] if False else info["alpha_deg"],
                 "throttle": th0, "weight_lb": W, "qbar_psf": q, "CL": CL},
        "n_struct_limit": pw.n_limit, "n_profile": list(P.nz_limits), "n_aero_max_trim_speed": n_aero, "CLmax0": clmax,
        "n_inst": n_inst, "R_inst_ft": R(n_inst), "n_sus_est": n_sus, "R_sus_ft_est": R(n_sus),
        "bank_sus_deg_est": math.degrees(math.acos(1 / n_sus)), "bank_inst_deg": math.degrees(math.acos(1 / n_inst)),
        "bank_max_deg": min(P.max_abs_phi_deg, math.degrees(math.acos(1 / n_inst))), "bank_fail_limit_deg": P.max_abs_phi_deg,
        "roll_rate_max_dps": out_rate["ail"][0], "roll_sign_p_per_plus_ail": out_rate["ail"][2],
        "pitch_rate_g_limited_dps": math.degrees(G * (n_inst - 1) / V), "pitch_rate_step_1s_dps": out_rate["elev"][0],
        "thrust_max_lbf_est": T_max, "thrust_trim_lbf": T_trim, "drag_trim_lbf": D0, "induced_drag_est_lbf": Di,
        "climb_fpm_max_est": ROC, "descent_fpm_idle_est": D0 * V / W * 60 - T_trim * 0 ,
        "speedbrake": sb is not None, "min_kcas_fail": P.min_kcas, "max_abs_theta_deg": P.max_abs_theta_deg,
        "ring_hint": {"min_spacing_ft": V * (60.0 / max(out_rate["ail"][0], 1) + 3.0), "min_path_radius_ft": 1.5 * R(n_sus)},
    }


if __name__ == "__main__":
    sim = fe.load_sim()
    doc = {"schema": "fd-p4-aircraft-limits/1", "doc": "PHASE4_FD_CONTROL_SURFACES.md P4.7; *_est = estimate "
           "(induced drag e=0.8, thrust after 6 s at throttle_max); rigid baseline aircraft at Phase-1 trim",
           "units": "ft, ft/s, deg, deg/s, lbf, fpm, g", "aircraft": {}}
    for m in ("c172x", "T38", "737", "f16"):
        doc["aircraft"][m] = limits(m, sim); print(m, json.dumps({k: doc["aircraft"][m][k] for k in ("n_inst", "R_inst_ft", "n_sus_est", "R_sus_ft_est", "roll_rate_max_dps", "climb_fpm_max_est")}))
    fp = os.path.join(HERE, "v2_results", "p4_aircraft_limits.json")
    if os.path.exists(fp):          # keep the hand-sourced ER block (alpha_stall, v_max, mach_max, bank_course)
        old = json.load(open(fp))
        for m in doc["aircraft"]:
            for k in ("alpha_stall_deg", "vne_published_kias", "mach_max", "v_dive_keas", "bank_course_deg", "src_v", "v_max_kcas"):
                if k in old["aircraft"].get(m, {}):
                    doc["aircraft"][m][k] = old["aircraft"][m][k]
        doc["doc"] = old["doc"]
    json.dump(doc, open(fp, "w"), indent=1, sort_keys=True)
