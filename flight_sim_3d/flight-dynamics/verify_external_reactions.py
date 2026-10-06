"""Proof that the <external_reactions> coupling works in JSBSim 1.3.1 (writes external_reactions_proof.json).

1. prepare_aircraft() copies aircraft/c172x into ./jsbsim_root (engine/, systems/ symlinked
   read-only) and appends a body-frame <force> at AERORP and a body-frame <moment>.
2. With zero commanded force/moment the copy flies bit-identically to the stock model.
3. Commanded components show up exactly in forces/fb{x,y,z}-external-lbs and
   moments/{l,m,n}-external-lbsft (force = magnitude*(x,y,z), NOT re-normalised when the
   direction is set through properties; moment = magnitude*(l,m,n) + r_(CG->AERORP) x F).
4. A commanded roll moment produces the expected initial roll acceleration pdot ~= L/Ixx.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flexwing as fw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "jsbsim_root")
MODEL = "c172x"


def trimmed(root):
    fdm = fw.new_fdm(MODEL, root)
    fdm["ic/h-sl-ft"] = 4000; fdm["ic/vc-kts"] = 100; fdm["ic/gamma-deg"] = 0
    fdm.run_ic()
    fdm["propulsion/set-running"] = -1; fdm["fcs/mixture-cmd-norm"] = 1.0
    fdm["simulation/do_simple_trim"] = 1
    return fdm


def history(fdm, n=600):
    keys = ("position/h-sl-ft", "attitude/theta-deg", "attitude/phi-deg", "velocities/vc-kts", "velocities/p-rad_sec")
    out = []
    for k in range(n):
        fdm["fcs/aileron-cmd-norm"] = 0.3 if 120 <= k < 240 else 0.0
        fdm.run()
        out.append([fdm[x] for x in keys])
    return np.array(out)


def main():
    fw.prepare_aircraft(MODEL, ROOT)
    proof = {"jsbsim_root": ROOT}
    # 2. zero force -> identical to stock
    a = history(trimmed(None)); b = history(trimmed(ROOT))
    proof["zero_force_identical_to_stock"] = bool(np.array_equal(a, b))
    # 3. property -> force/moment bookkeeping
    fdm = trimmed(ROOT)
    F = np.array([10.0, 20.0, -100.0]); Mc = np.array([50.0, 25.0, -5.0])
    fdm["external_reactions/flexwing_F/magnitude"] = 1.0
    for ax, v in zip("xyz", F): fdm[f"external_reactions/flexwing_F/{ax}"] = v
    fdm["external_reactions/flexwing_M/magnitude-lbsft"] = 1.0
    for ax, v in zip("lmn", Mc): fdm[f"external_reactions/flexwing_M/{ax}"] = v
    fdm.run()
    Fj = np.array([fdm[f"forces/fb{a}-external-lbs"] for a in "xyz"])
    Mj = np.array([fdm[f"moments/{a}-external-lbsft"] for a in "lmn"])
    loc = np.array([fdm[f"external_reactions/flexwing_F/location-{a}-in"] for a in "xyz"])
    cg = np.array([fdm["inertia/cg-x-in"], fdm["inertia/cg-y-in"], fdm["inertia/cg-z-in"]])
    r_body = np.array([-(loc[0] - cg[0]), loc[1] - cg[1], -(loc[2] - cg[2])]) / 12.0   # structural -> body, ft
    M_expected = Mc + np.cross(r_body, F)
    proof["force_cmd"] = F.tolist(); proof["force_jsbsim"] = Fj.round(6).tolist()
    proof["moment_cmd"] = Mc.tolist(); proof["moment_jsbsim"] = Mj.round(4).tolist()
    proof["moment_expected_incl_arm"] = M_expected.round(4).tolist()
    proof["force_ok"] = bool(np.allclose(Fj, F, atol=1e-9))
    proof["moment_ok"] = bool(np.allclose(Mj, M_expected, atol=1e-6))
    # 4. dynamic effect: 2000 lbf*ft roll moment from trim, compare pdot with L/Ixx (Ixz = 0 for c172x)
    f1 = trimmed(ROOT); f0 = trimmed(ROOT)
    f1["external_reactions/flexwing_M/magnitude-lbsft"] = 1.0
    f1["external_reactions/flexwing_M/l"] = 2000.0
    f1.run(); f0.run()
    ixx = f1["inertia/ixx-slugs_ft2"]
    dp = f1["accelerations/pdot-rad_sec2"] - f0["accelerations/pdot-rad_sec2"]
    proof["roll_moment_cmd_lbft"] = 2000.0
    proof["ixx_slugft2"] = ixx
    proof["ixz_slugft2"] = f1["inertia/ixz-slugs_ft2"]
    proof["delta_pdot_measured"] = dp
    proof["delta_pdot_expected_L_over_Ixx"] = 2000.0 / ixx
    proof["roll_dynamics_ok"] = bool(abs(dp - 2000.0 / ixx) < 0.02 * 2000.0 / ixx)
    proof["all_ok"] = all(proof[k] for k in ("zero_force_identical_to_stock", "force_ok", "moment_ok", "roll_dynamics_ok"))
    json.dump(proof, open(os.path.join(HERE, "external_reactions_proof.json"), "w"), indent=1)
    print(json.dumps(proof, indent=1))


if __name__ == "__main__":
    main()
