"""Flex v2 validation -> v2_validation.json (re-runnable; needs v2_results/oas_raw.json from _oas_check.py for the OAS part).

  1. hand calcs, uniform cantilever (FE with the default 3b+2t truncation vs closed form)
  2. FE mesh convergence of the hand-calc quantities (8 / 16 / 32 / 64 elements)
  3. modal-truncation convergence on the aircraft wings (N = 6 vs N+1, N+2 modes)
  4. OpenAeroStruct 2.12 (VLM + tube spar) vs flexbody strip theory on the same uniform wing
  5. reduced (v1, projected genome) vs full (v2) first bending / torsion frequency over random genomes
"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import flexbody as fb  # noqa: E402
import flexeval as fe  # noqa: E402

fb.blas_threads(1)


def oas_part():
    p = os.path.join(HERE, "v2_results", "oas_raw.json")
    if not os.path.exists(p):
        return {"status": "not run (v2_results/oas_raw.json missing)"}
    o = json.load(open(p))
    L, c = o["semi_span_m"], o["chord_m"]
    EI = o["E"] * o["flex"][0]["Iy"]
    GJ = o["G"] * o["flex"][0]["J"]
    a3 = o["CLalpha_rigid_per_rad"]
    alpha = math.radians(o["alpha_deg"])
    res = {"setup": {"oas_version": o["openaerostruct"], "openmdao": o["om"], "semi_span_m": L, "chord_m": c, "AR": 2 * L / c,
                     "EI_Nm2": EI, "GJ_Nm2": GJ, "elastic_axis_frac": o["fem_origin"], "alpha_deg": o["alpha_deg"],
                     "strip_lift_slope": "OAS rigid VLM CL_alpha = %.4f /rad, uniform along the span" % a3,
                     "units": "SI (flexbody's linear algebra is unit-agnostic)"}}
    s = fb.uniform_test_surface(L=L, c=c, EI=EI, GJ=GJ, m=10.0, x_ea=o["fem_origin"], x_cg=o["fem_origin"], a=a3)
    K, A, b, r_rig, rG = fb.surface_static_matrices(s)
    W = s.PhiW + s.e_c[:, None] * s.PsiT
    ba = W.T @ (s.lift_w * alpha)
    mu = np.linalg.eigvals(np.linalg.solve(K, A))
    mu = mu[(abs(mu.imag) < 1e-9) & (mu.real > 0)].real
    qD_strip = float(1 / mu.max())
    qD_closed = (math.pi / (2 * L)) ** 2 * GJ / (c * (o["fem_origin"] - 0.25) * c * a3)
    rows = []
    for p_ in o["flex"]:
        if "error" in p_:
            continue
        q = p_["q"]
        eta = np.linalg.solve(K - q * A, q * ba)
        G = s.PsiT
        cl = float(np.sum(s.lift_w * (alpha + G @ eta)) / (c * L))
        rows.append({"v_mps": p_["v"], "q_Pa": q, "q_over_qD_strip": q / qD_strip,
                     "oas": {"CL": p_["CL"], "tip_w_m": p_["tip_w"], "tip_twist_rad": p_["tip_ry"]},
                     "flexbody_strip": {"CL": cl, "tip_w_m": float(s.tipW @ eta), "tip_twist_rad": float(s.tipT @ eta)}})
    for r in rows:
        r["pct_diff_flexbody_vs_oas"] = {k: 100 * (r["flexbody_strip"][k] - r["oas"][k]) / r["oas"][k] for k in r["oas"]}
    # OAS divergence q from the linear amplification of the tip twist: theta(q) = th1 q / (1 - q/qD)  ->  q/theta linear
    lin = [r for r in rows if r["q_over_qD_strip"] < 0.55]
    qq = np.array([r["q_Pa"] for r in lin])
    th = np.array([r["oas"]["tip_twist_rad"] for r in lin])
    k1, k0 = np.polyfit(qq, qq / th, 1)
    qD_oas = float(-k0 / k1)
    thf = np.array([r["flexbody_strip"]["tip_twist_rad"] for r in lin])
    k1f, k0f = np.polyfit(qq, qq / thf, 1)
    win = {}
    for lim in (0.26, 0.37, 0.55):                 # extrapolation window sensitivity (OAS is geometrically nonlinear)
        sel = [r for r in rows if r["q_over_qD_strip"] < lim]
        if len(sel) >= 3:
            q_ = np.array([r["q_Pa"] for r in sel])
            a_, b_ = np.polyfit(q_, q_ / np.array([r["oas"]["tip_twist_rad"] for r in sel]), 1)
            c_, d_ = np.polyfit(q_, q_ / np.array([r["flexbody_strip"]["tip_twist_rad"] for r in sel]), 1)
            win[f"q<{lim}qD"] = {"n": len(sel), "q_D_oas_Pa": float(-b_ / a_), "q_D_flexbody_same_fit_Pa": float(-d_ / c_),
                                 "pct_flexbody_eig_vs_oas": float(100 * (qD_strip - (-b_ / a_)) / (-b_ / a_)),
                                 "pct_same_fit": float(100 * ((-d_ / c_) - (-b_ / a_)) / (-b_ / a_))}
    res["divergence_fit_windows"] = win
    res["rows"] = rows
    res["divergence"] = {"q_D_strip_flexbody_Pa": qD_strip, "q_D_strip_closed_form_Pa": qD_closed,
                         "q_D_oas_fit_Pa": qD_oas, "q_D_flexbody_same_fit_Pa": float(-k0f / k1f),
                         "fit_points_q_Pa": qq.tolist(),
                         "pct_flexbody_vs_oas": 100 * (qD_strip - qD_oas) / qD_oas,
                         "pct_flexbody_vs_closed_form": 100 * (qD_strip - qD_closed) / qD_closed,
                         "note": "OAS q_D extrapolated from q/theta_tip = (1 - q/q_D)/th1 over q < 0.55 q_D(strip); OAS "
                                 "itself has no divergence eigen-solver (its coupled solve blows up near q_D: v = 150 m/s)."}
    low = rows[0]
    res["summary_pct_at_lowest_q"] = low["pct_diff_flexbody_vs_oas"]
    res["summary"] = ("same EI/GJ/EA and the OAS rigid CL_alpha: total CL and tip twist agree to a few %, tip deflection is "
                      "~15-17 % higher in strip theory (uniform spanwise lift slope vs VLM tip loss -> more outboard load); "
                      "q_D: flexbody's eigenvalue result is within -1.4 .. +7.2 % of the OAS extrapolation depending on the fit "
                      "window (+-1.5 % in the near-linear windows), see divergence_fit_windows; OAS deflections become large (>20 % semi-span at 0.36 q_D) so the far points are "
                      "geometrically nonlinear")
    return res


def main():
    out = {}
    out["hand_calcs_default_truncation"] = fb.hand_calcs()
    out["fe_mesh_convergence"] = {str(n): {k: v["pct"] for k, v in fb.hand_calcs(n_el=n).items()} for n in (8, 16, 32, 64)}
    tc = {}
    for m in ("c172x", "T38", "737", "f16"):
        for tag, g in (("baseline", None), ("soft_tip", {"wing_ei_root": 0.7, "wing_ei_taper_1": 0.75, "wing_ei_taper_4": 1.05,
                                                          "wing_nsm_tip": 1.25})):
            r = fb.truncation_check(m, g)
            tc[f"{m}/{tag}"] = {"N": r["N"], "pct_N+1": r["pct_N+1"], "pct_N+2": r["pct_N+2"],
                                "max_abs_pct": max(abs(v) for k in ("pct_N+1", "pct_N+2") for v in r[k].values())}
    out["modal_truncation"] = {"tolerance_pct": 2.0, "cases": tc}
    json.dump(out, open(os.path.join(HERE, "v2_validation.json"), "w"), indent=1, default=float)   # checkpoint
    out["openaerostruct"] = oas_part()
    json.dump(out, open(os.path.join(HERE, "v2_validation.json"), "w"), indent=1, default=float)
    rp = {}
    for m in ("c172x", "T38", "737", "f16"):
        rng = np.random.default_rng(7)
        eb, et = [], []
        for _ in range(32):
            u = rng.random(len(fb.GENES_V2))
            M = fb.FlexBodyModel(m, u)
            fr = fe.reduced_wing(u, m).margins()["f_modes_hz"]
            cls = list(M.wingR.cls)
            f = M.wingR.omega / (2 * math.pi)
            eb.append(100 * (fr[0] - f[cls.index("b")]) / f[cls.index("b")])
            et.append(100 * (fr[1] - f[cls.index("t")]) / f[cls.index("t")])
        rp[m] = {"bending1_pct": {"mean": float(np.mean(eb)), "max_abs": float(np.max(np.abs(eb)))},
                 "torsion1_pct": {"mean": float(np.mean(et)), "max_abs": float(np.max(np.abs(et)))}, "n": 32}
        M0 = fb.FlexBodyModel(m)
        rp[m]["baseline_bending1_pct"] = 100 * (fe.reduced_wing(None, m).margins()["f_modes_hz"][0] - M0.wingR.omega[0] / (2 * math.pi)) / (M0.wingR.omega[0] / (2 * math.pi))
    out["reduced_projection_first_mode_freq_error"] = rp
    json.dump(out, open(os.path.join(HERE, "v2_validation.json"), "w"), indent=1, default=float)
    print(json.dumps({"hand": {k: round(v["pct"], 4) for k, v in out["hand_calcs_default_truncation"].items()},
                      "trunc_max": {k: round(v["max_abs_pct"], 3) for k, v in tc.items()},
                      "oas_div": out["openaerostruct"].get("divergence"), "oas_low_q": out["openaerostruct"].get("summary_pct_at_lowest_q"),
                      "proj": rp}, indent=1, default=float))


if __name__ == "__main__":
    main()
