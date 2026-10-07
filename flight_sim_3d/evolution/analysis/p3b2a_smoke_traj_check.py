"""phase3b2a-smoke-s1 trajectory check vs sim-bridge/notes/b2_fd_spec.md (read-only on the run dir):
validate_traj on every file; B2 node fields (tc_local, camber_meq_pct_local, dihedral_delta_deg, dihedral_baseline_deg,
section_baseline) on wingR / wingL with per-node lengths = axis_nodes_body_m; dihedral z in axis / le / te nodes:
for each file, rebuild FD's node layout for the SAME genome at dihedral 0 (make_fd_model + fd_node_layout, B2) and at
its own dihedral, both without rp offset, and check z_own - z_dih0 == -(|y| - y0) tan(dGamma) (ft->m) and equals the
trajectory's z - (z of the dihedral-0 layout + the file's constant rp offset); x / y unchanged.
  EVOLUTION_FD_DIR=evolution/_fd_pin_p3b2a $PY -B evolution/analysis/p3b2a_smoke_traj_check.py -> phase3b2a_smoke_s1_traj_check.json"""
import glob, json, math, os, sys
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TEAM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TEAM)
from evolution import eval as ev, fidelity as F, sim, validate_traj  # noqa: E402
RUN = os.path.join(TEAM, "evolution/runs/phase3b2a-smoke-s1")
rj = ev.load_run_cfg(os.path.join(RUN, "run.json"))
KEYS = F.B2_NODE_KEYS
out = {"run_dir": os.path.relpath(RUN, TEAM), "spec": "sim-bridge/notes/b2_fd_spec.md", "files": {}}
for f in sorted(glob.glob(os.path.join(RUN, "trajectories", "traj_*.json"))):
    d = json.load(open(f)); ac = d["aircraft"]
    ent = ev._aircraft_entry(rj, ac); groups = ev.gene_groups(ent)
    gains, struct = ev.split_values(d["genome"], groups); shape = ev.shape_values(d["genome"], groups)
    comps = {c["name"]: c for c in d["structure"]["components"]}
    rec = {"validate_errors": validate_traj.validate_doc(d, "b1"), "dihedral_delta_deg_gene": shape["wing_dihedral_delta_deg"]}
    fields = {}
    for nm in ("wingR", "wingL"):
        c = comps[nm]; n = len(c["axis_nodes_body_m"])
        fields[nm] = {k: k in c for k in KEYS}
        fields[nm]["per_node_len_ok"] = all(len(c[k]) == n for k in ("tc_local", "camber_meq_pct_local") if k in c)
        fields[nm]["dihedral_delta_matches_gene"] = c.get("dihedral_delta_deg") == shape["wing_dihedral_delta_deg"]
    rec["fields"] = fields
    pd = ev._profile_d(ent)
    own = {c["name"]: c for c in F.fd_node_layout(F.make_fd_model(pd, struct, F.B2, shape), F.B2, None)}
    s0 = dict(shape, wing_dihedral_delta_deg=0.0)
    zero = {c["name"]: c for c in F.fd_node_layout(F.make_fd_model(pd, struct, F.B2, s0), F.B2, None)}
    tg = math.tan(math.radians(shape["wing_dihedral_delta_deg"]))
    dz = {}
    for nm in ("wingR", "wingL"):
        r = {}
        for key in ("axis_nodes_body", "le_nodes_body", "te_nodes_body"):
            A = np.array(own[nm][key + "_ft"]); Z = np.array(zero[nm][key + "_ft"]); T = np.array(comps[nm][key + "_m"])
            ay = np.abs(A[:, 1]); y0 = ay.min()
            expect = -(ay - y0) * tg          # spec: z += -s tan(dGamma); s = FE beam station (straight EA span)
            off = T[0] - np.round(A[0] * sim.FT, 6)     # constant body offset (rp) of the file, if any
            r[key] = {"xy_unchanged_vs_dih0": bool(np.array_equal(A[:, :2], Z[:, :2])),
                      "z_delta_max_abs_err_ft": float(np.max(np.abs((A[:, 2] - Z[:, 2]) - expect))),
                      "z_tip_delta_m": float((A[-1, 2] - Z[-1, 2]) * sim.FT),
                      "traj_equals_fd_layout_m": bool(np.max(np.abs(T - (np.round(A * sim.FT, 6) + off))) < 2e-6),
                      "traj_rp_offset_m": off.tolist()}
        dz[nm] = r
    rec["dihedral_z"] = dz
    rec["ok"] = (not rec["validate_errors"] and all(all(v.values()) for v in fields.values())
                 and all(x["xy_unchanged_vs_dih0"] and x["z_delta_max_abs_err_ft"] <= 1e-6   # FD rounds node coords to 1e-6 ft and x["traj_equals_fd_layout_m"]
                         for w in dz.values() for x in w.values()))
    out["files"][os.path.basename(f)] = rec
    print(os.path.basename(f), rec["ok"], round(rec["dihedral_delta_deg_gene"], 4),
          {nm: round(dz[nm]["axis_nodes_body"]["z_tip_delta_m"], 5) for nm in dz}, rec["validate_errors"][:2], flush=True)
out["all_ok"] = all(r["ok"] for r in out["files"].values())
json.dump(out, open(os.path.join(HERE, "phase3b2a_smoke_s1_traj_check.json"), "w"), indent=1)
print("ALL_OK", out["all_ok"])
