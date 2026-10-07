"""OpenAeroStruct cross-check (run with _oas_venv/bin/python). Uniform rectangular unswept wing, tube spar, VLM.
Writes v2_results/oas_raw.json: rigid CL_alpha, flexible tip deflection / twist / CL at several dynamic pressures,
for the same EI/GJ/elastic-axis used by flexbody's strip-theory beam in v2_validate.py."""
import json
import math
import os
import sys
import time

import numpy as np
import openmdao.api as om
import openaerostruct
import openmdao
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

HERE = os.path.dirname(os.path.abspath(__file__))
SEMI, CHORD = 10.0, 2.0              # m: semi-span, chord (AR 10)
R, T = 0.10, 0.01                    # m: tube radius, wall thickness
E, G = 70.0e9, 30.0e9
FEM_ORIGIN = 0.35                    # elastic axis / chord
RHO = 1.225
ALPHA = 2.0                          # deg


def build(E_, G_, num_y=41, num_x=3):
    mesh = generate_mesh({"num_y": num_y, "num_x": num_x, "wing_type": "rect", "symmetry": True,
                          "span": 2 * SEMI, "root_chord": CHORD, "span_cos_spacing": 0.0, "chord_cos_spacing": 0.0})
    surface = {"name": "wing", "symmetry": True, "S_ref_type": "projected", "fem_model_type": "tube", "mesh": mesh,
               "twist_cp": np.zeros(1), "thickness_cp": np.array([T]), "radius_cp": np.array([R]),
               "CL0": 0.0, "CD0": 0.0, "k_lam": 0.05, "t_over_c_cp": np.array([0.12]), "c_max_t": 0.3,
               "with_viscous": False, "with_wave": False, "E": E_, "G": G_, "yield": 1e12, "mrho": 1.0,
               "fem_origin": FEM_ORIGIN, "wing_weight_ratio": 1.0, "struct_weight_relief": False,
               "distributed_fuel_weight": False, "exact_failure_constraint": False}
    prob = om.Problem(reports=False)
    iv = om.IndepVarComp()
    iv.add_output("v", val=50.0, units="m/s")
    iv.add_output("alpha", val=ALPHA, units="deg")
    iv.add_output("Mach_number", val=0.0)
    iv.add_output("re", val=1.0e6, units="1/m")
    iv.add_output("rho", val=RHO, units="kg/m**3")
    iv.add_output("CT", val=1e-5, units="1/s")
    iv.add_output("R", val=1e6, units="m")
    iv.add_output("W0", val=1000.0, units="kg")
    iv.add_output("speed_of_sound", val=340.0, units="m/s")
    iv.add_output("load_factor", val=1.0)
    iv.add_output("empty_cg", val=np.zeros(3), units="m")
    prob.model.add_subsystem("prob_vars", iv, promotes=["*"])
    prob.model.add_subsystem("wing", AerostructGeometry(surface=surface))
    pt = "AS_point_0"
    prob.model.add_subsystem(pt, AerostructPoint(surfaces=[surface]),
                             promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "CT", "R", "W0", "speed_of_sound",
                                              "empty_cg", "load_factor"])
    com = pt + ".wing_perf"
    c = prob.model.connect
    c("wing.local_stiff_transformed", pt + ".coupled.wing.local_stiff_transformed")
    c("wing.nodes", pt + ".coupled.wing.nodes")
    c("wing.mesh", pt + ".coupled.wing.mesh")
    c("wing.radius", com + ".radius")
    c("wing.thickness", com + ".thickness")
    c("wing.nodes", com + ".nodes")
    c("wing.cg_location", pt + ".total_perf.wing_cg_location")
    c("wing.structural_mass", pt + ".total_perf.wing_structural_mass")
    c("wing.t_over_c", com + ".t_over_c")
    prob.setup()
    cpl = prob.model.AS_point_0.coupled
    cpl.nonlinear_solver = om.NonlinearBlockGS(maxiter=500, atol=1e-12, rtol=1e-12, use_aitken=True, iprint=-1)
    cpl.linear_solver = om.DirectSolver()
    return prob


def solve(prob, v):
    prob.set_val("v", v, units="m/s")
    prob.run_model()
    disp = np.asarray(prob.get_val("AS_point_0.coupled.wing.disp"))     # (ny, 6): x, y, z, rx, ry, rz at nodes
    CL = float(prob.get_val("AS_point_0.wing_perf.CL")[0])
    iy = int(np.argmax(np.abs(np.asarray(prob.get_val("wing.mesh"))[0, :, 1])))  # tip node index (|y| max)
    return {"v": v, "q": 0.5 * RHO * v * v, "CL": CL, "tip_w": float(disp[iy, 2]), "tip_ry": float(disp[iy, 4]),
            "disp_z": disp[:, 2].tolist(), "disp_ry": disp[:, 4].tolist(),
            "y": np.asarray(prob.get_val("wing.mesh"))[0, :, 1].tolist(),
            "Iy": float(np.asarray(prob.get_val("wing.Iy"))[0]), "J": float(np.asarray(prob.get_val("wing.J"))[0])}


def main():
    t0 = time.time()
    out = {"openaerostruct": openaerostruct.__version__, "om": openmdao.__version__, "semi_span_m": SEMI, "chord_m": CHORD,
           "tube_r_m": R, "tube_t_m": T, "E": E, "G": G, "fem_origin": FEM_ORIGIN, "rho": RHO, "alpha_deg": ALPHA}
    rigid = build(E * 1e6, G * 1e6)
    rr = [solve(rigid, v) for v in (30.0, 60.0)]
    out["rigid"] = rr
    out["CLalpha_rigid_per_rad"] = rr[0]["CL"] / math.radians(ALPHA)
    flex = build(E, G)
    pts = []
    for v in (30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0):
        try:
            pts.append(solve(flex, v))
        except Exception as e:  # noqa: BLE001
            pts.append({"v": v, "error": repr(e)[:300]})
    out["flex"] = pts
    out["wall_s"] = time.time() - t0
    json.dump(out, open(os.path.join(HERE, "v2_results", "oas_raw.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("flex", "rigid")}, indent=1))
    for p in pts:
        print({k: p.get(k) for k in ("v", "q", "CL", "tip_w", "tip_ry", "error")})


if __name__ == "__main__":
    main()
