"""Before/after regression for the jsbsim_root socket clean-up (writes _scratch/socket_regression_<tag>.npz/.json).

For c172x / T38 / 737 (patched root, zero external force):
  * full mode-1 trim snapshot at the design point (gear up, coupled_sim.trim)
  * a rigid 12 s run_maneuver trajectory (elevator pull, aileron doublet, gust), full float64
  * a two-way flex run_maneuver trajectory (baseline genes), full float64
  * zero-force patched-vs-stock trajectory equality
Run with tag 'before' (old root) and 'after' (cleaned root); compare with --compare.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flexwing as fw  # noqa: E402
import coupled_sim as cs  # noqa: E402

ROOT = os.path.join(HERE, "jsbsim_root")
KEYS = ("aero/alpha-deg", "attitude/theta-deg", "fcs/throttle-cmd-norm", "fcs/throttle-pos-norm", "fcs/elevator-cmd-norm",
        "fcs/pitch-trim-cmd-norm", "fcs/elevator-pos-deg", "gear/gear-pos-norm", "velocities/mach", "aero/qbar-psf",
        "inertia/weight-lbs", "accelerations/udot-ft_sec2", "accelerations/wdot-ft_sec2", "accelerations/qdot-rad_sec2")
TRAJ = ("alt_ft", "nz", "p_dps", "q_dps", "phi_deg", "theta_deg", "alpha_deg", "kcas", "qbar", "fbz_ext", "l_ext", "m_ext")


def trim_snapshot(model, root):
    md = cs.MANEUVER_DEFAULTS[model]
    f = fw.new_fdm(model, root)
    cs.trim(f, md["h_ft"], md["kcas"])
    return {k: float(f[k]) for k in KEYS}


def stock_vs_root(model):
    def hist(root):
        md = cs.MANEUVER_DEFAULTS[model]
        f = fw.new_fdm(model, root); cs.trim(f, md["h_ft"], md["kcas"])
        e0 = f["fcs/elevator-cmd-norm"]; out = []
        for k in range(600):
            f["fcs/elevator-cmd-norm"] = e0 + (md["d_elev"] if 120 <= k < 240 else 0.0)
            f["fcs/aileron-cmd-norm"] = md["d_ail"] if 300 <= k < 420 else 0.0
            f.run(); out.append([f[x] for x in ("position/h-sl-ft", "attitude/theta-deg", "attitude/phi-deg",
                                               "velocities/vc-kts", "velocities/p-rad_sec", "velocities/q-rad_sec")])
        return np.array(out)
    a, b = hist(None), hist(ROOT)
    return bool(np.array_equal(a, b)), b


def main(tag):
    res, arrs = {}, {}
    for m in ("c172x", "T38", "737"):
        res[m] = {"trim": trim_snapshot(m, ROOT)}
        same, h = stock_vs_root(m); res[m]["zero_force_identical_to_stock"] = same; arrs[f"{m}_stock_cmp"] = h
        for mode, use in (("rigid", False), ("twoway", True)):
            r = cs.run_maneuver(m, mode="twoway", use_coupler=use, root=ROOT)
            arrs[f"{m}_{mode}"] = np.stack([r[k] for k in TRAJ])
        r2 = cs.run_maneuver(m, mode="twoway", use_coupler=True, root=ROOT)
        res[m]["flex_repeat_identical"] = bool(np.array_equal(np.stack([r2[k] for k in TRAJ]), arrs[f"{m}_twoway"]))
        print(m, json.dumps(res[m]), flush=True)
    np.savez(os.path.join(HERE, "_scratch", f"socket_regression_{tag}.npz"), **arrs)
    json.dump(res, open(os.path.join(HERE, "_scratch", f"socket_regression_{tag}.json"), "w"), indent=1)


def compare():
    a = np.load(os.path.join(HERE, "_scratch", "socket_regression_before.npz"))
    b = np.load(os.path.join(HERE, "_scratch", "socket_regression_after.npz"))
    ja = json.load(open(os.path.join(HERE, "_scratch", "socket_regression_before.json")))
    jb = json.load(open(os.path.join(HERE, "_scratch", "socket_regression_after.json")))
    out = {"arrays_bit_identical": {k: bool(np.array_equal(a[k], b[k])) for k in a.files},
           "trim_identical": {m: ja[m]["trim"] == jb[m]["trim"] for m in ja},
           "after_checks": {m: {k: v for k, v in jb[m].items() if k != "trim"} for m in jb},
           "trim_737_after": {k: round(v, 4) for k, v in jb["737"]["trim"].items()}}
    out["all_identical"] = all(out["arrays_bit_identical"].values()) and all(out["trim_identical"].values())
    json.dump(out, open(os.path.join(HERE, "socket_cleanup_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    compare() if sys.argv[1] == "--compare" else main(sys.argv[1])
