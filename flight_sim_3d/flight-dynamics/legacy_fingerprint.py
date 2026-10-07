"""Fingerprint of the v1 (Phase-1, 4-gene) flex path: sha256 of margins and coupled trajectories, full float64.

python legacy_fingerprint.py --write   -> v1_legacy_fingerprint.json (taken 2026-10-06 05:2x PT, before flex v2)
test_flexbody.py::test_v1_legacy_bit_identical recomputes and compares (v2 must not change v1 results).
"""
import hashlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flexwing as fw  # noqa: E402
import coupled_sim as cs  # noqa: E402

GEO = {"c172x": (36.0, 174.0, 1454.0), "737": (94.7, 1171.0, 83000.0), "T38": (25.25, 170.0, 7574.0), "f16": (30.0, 300.0, 17400.0)}
KEYS = ("alt_ft", "nz", "p_dps", "q_dps", "alpha_deg", "fbz_ext", "l_ext", "m_ext", "tip_w_ft_R", "root_bm_lbft_R", "tip_twist_deg_L")


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:16]


def compute(maneuver_s: float = 3.0):
    out = {}
    for m, (b, S, ew) in GEO.items():
        for tag, ov in (("base", {}), ("s0.8r1.1", fw.genes_to_overrides({"stiffness_scale": 0.8, "torsion_bend_ratio": 1.1,
                                                                             "zeta": 0.01, "nonstruct_scale": 1.2}))):
            r = fw.FlexWing(fw.params_for(m, b, S, ew, **ov)).margins()
            out[f"{m}/{tag}/margins"] = _h([v for k, v in sorted(r.items()) if isinstance(v, float)] + r["f_modes_hz"])
        rr = cs.run_maneuver(m, "twoway", duration_s=maneuver_s)
        out[f"{m}/maneuver_twoway"] = _h(np.stack([rr[k] for k in KEYS]))
        rg = cs.run_maneuver(m, "twoway", duration_s=maneuver_s, use_coupler=False)
        out[f"{m}/maneuver_rigid"] = _h(np.stack([rg[k] for k in KEYS[:8]]))
    return out


if __name__ == "__main__":
    fp = compute()
    if "--write" in sys.argv:
        json.dump(fp, open(os.path.join(HERE, "v1_legacy_fingerprint.json"), "w"), indent=1)
    print(json.dumps(fp, indent=1))
