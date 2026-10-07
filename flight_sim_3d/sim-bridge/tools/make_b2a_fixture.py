#!/usr/bin/env python3
"""P3-B2a viewer fixture from FD's FROZEN B2a files (read-only; run with PYTHONDONTWRITEBYTECODE=1).

    $PY tools/make_b2a_fixture.py [--fd-dir ../flight-dynamics] [--seconds 30]

1. md5sum -c v2_results/FROZEN_B2a.md5 in the FD dir (any mismatch -> exit 2).
2. c172x, structure + B1 shape genes of ER's pilot best (phase3b1r1-pilot-s1 g59), B2a genes: dihedral +3 deg,
   tc_root 1.2, camber_root +1 %c (others default). FD flexbody_b2.node_layout_b2 (B2) and node_layout_b1 (same
   genes, B2 default); checks node_layout_b2(B2 default) == node_layout_b1 exactly.
3. tests/fixtures/b2a_c172x_node_layout.json: wings of both layouts in FD units (ft / deg) + genes + md5 + checks.
4. data/b2a_fixture/: index.json + two trajectory files built from the pilot c172x g59 flight (first --seconds s):
   g0 = B1 r1 geometry (as ER wrote it), g1 = the same flight with FD's B2a wing nodes (ft -> m, + ER's CG offset fitted
   from the B1 nodes) and the B2 keys (tc_local, camber_meq_pct_local, dihedral_delta_deg, dihedral_baseline_deg,
   section_baseline) passed through unchanged - the way ER's b1_node_fields converts B1 r1 keys. Display fixture only:
   the flight data are the B1 flight's, not a B2 simulation (flagged `fixture` in the files).
"""
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM = os.path.dirname(HERE)
FT = 0.3048
B2_KEYS = ("tc_local", "camber_meq_pct_local", "dihedral_delta_deg", "dihedral_baseline_deg", "section_baseline",
           "area_scale", "aspect_scale")
GENES_B2 = {"wing_dihedral_delta_deg": 3.0, "wing_tc_root_scale": 1.2, "wing_camber_root_delta_pct": 1.0}
TEMPLATE = os.path.join(TEAM, "evolution", "runs", "phase3b1r1-pilot-s1", "trajectories",
                        "traj_c172x_phase3b1r1-pilot-s1_g59.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fd-dir", default=os.path.join(TEAM, "flight-dynamics"))
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fixture", default=os.path.join(HERE, "tests", "fixtures", "b2a_c172x_node_layout.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "b2a_fixture"))
    a = ap.parse_args()
    md5f = os.path.join(a.fd_dir, "v2_results", "FROZEN_B2a.md5")
    p = subprocess.run(["md5sum", "-c", md5f], cwd=a.fd_dir, capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    if p.returncode != 0 or not lines or any(not l.endswith(": OK") for l in lines):
        print(p.stdout + p.stderr)
        sys.exit(2)
    print(f"md5 {md5f}: {len(lines)}/{len(lines)} OK")
    md5 = {l.split()[1]: l.split()[0] for l in open(md5f) if len(l.split()) == 2 and len(l.split()[0]) == 32}

    sys.path.insert(0, a.fd_dir)
    import numpy as np
    import flexbody_b1 as b1
    import flexbody_b2 as b2
    for mod in (b1, b2):   # imported from the FD dir checked above
        assert os.path.dirname(os.path.abspath(mod.__file__)) == os.path.abspath(a.fd_dir), mod.__file__
    assert hashlib.md5(open(b2.__file__, "rb").read()).hexdigest() == md5["flexbody_b2.py"]

    doc = json.load(open(a.template))
    g = doc["genome"]
    shape = {k: v for k, v in g.items() if k.startswith(("wing_chord_taper", "wing_twist", "wing_sweep"))}
    struct = {k: v for k, v in g.items() if k not in shape and not k.startswith(("kp_", "ki_", "kd_"))}
    m1 = b1.FlexBodyModelB1("c172x", struct, shape_genes=shape)
    L1 = b1.node_layout_b1(m1, None)
    m0 = b2.FlexBodyModelB2("c172x", struct, shape_genes=dict(shape))              # B2 at default
    L0 = b2.node_layout_b2(m0, None)
    m2 = b2.FlexBodyModelB2("c172x", struct, shape_genes={**shape, **GENES_B2})
    L2 = b2.node_layout_b2(m2, None)
    wings = lambda L: {c["name"]: c for c in L if c["name"] in ("wingR", "wingL")}
    w1, w0, w2 = wings(L1), wings(L0), wings(L2)
    default_equal = json.dumps(L0, sort_keys=True, default=float) == json.dumps(L1, sort_keys=True, default=float)
    print("node_layout_b2(B2 default) == node_layout_b1:", default_equal)

    # ER's CG offset (traj nodes = FD nodes * FT + const), fitted per wing from the B1 nodes of the template
    tcomp = {c["name"]: c for c in doc["structure"]["components"]}
    off, fit = {}, {}
    for nm in ("wingR", "wingL"):
        d = np.array(tcomp[nm]["axis_nodes_body_m"]) - np.array(w1[nm]["axis_nodes_body_ft"]) * FT
        off[nm] = d.mean(axis=0)
        fit[nm] = float(np.abs(d - off[nm]).max())
        dl = np.array(tcomp[nm]["le_nodes_body_m"]) - np.array(w1[nm]["le_nodes_body_ft"]) * FT - off[nm]
        fit[nm] = max(fit[nm], float(np.abs(dl).max()))
    print("CG offset (m)", {k: [round(float(x), 6) for x in v] for k, v in off.items()}, "fit residual", fit)

    checks = {}
    for nm in ("wingR", "wingL"):
        s = np.arange(len(w2[nm]["axis_nodes_body_ft"])) * m2.__getattribute__(nm).beam.h
        dz = np.array(w2[nm]["axis_nodes_body_ft"])[:, 2] - np.array(w1[nm]["axis_nodes_body_ft"])[:, 2]
        y = np.array(w1[nm]["axis_nodes_body_ft"])[:, 1]
        checks[nm] = {"dz_vs_station_max_err_ft": float(np.abs(dz + s * math.tan(math.radians(3.0))).max()),
                      "dz_vs_abs_y_minus_y0_max_err_ft": float(np.abs(dz + (np.abs(y) - abs(y[0])) * math.tan(math.radians(3.0))).max()),
                      "tip_dz_ft": float(dz[-1]), "xy_unchanged": bool(np.array_equal(np.array(w2[nm]["axis_nodes_body_ft"])[:, :2],
                                                                                   np.array(w1[nm]["axis_nodes_body_ft"])[:, :2]))}
    print("dihedral checks", json.dumps(checks))

    keep = lambda w: {k: w[k] for k in ("axis_nodes_body_ft", "le_nodes_body_ft", "te_nodes_body_ft", "chord_ft",
                                        "geometric_twist_deg", "node_span_frac", *B2_KEYS) if k in w}
    fx = {"schema": "sim-bridge-b2a-fixture/1", "aircraft": "c172x", "fd_dir": os.path.relpath(a.fd_dir, TEAM),
          "frozen_md5": md5, "template": os.path.relpath(a.template, TEAM), "genes_b1_shape": shape, "genes_b2": GENES_B2,
          "b2_default_equals_b1": default_equal, "dihedral_checks": checks, "cg_offset_m": {k: v.tolist() for k, v in off.items()},
          "b1": {k: keep(v) for k, v in w1.items()}, "b2": {k: keep(v) for k, v in w2.items()}}
    os.makedirs(os.path.dirname(a.fixture), exist_ok=True)
    json.dump(fx, open(a.fixture, "w"), indent=1)
    print("fixture", a.fixture)

    # trajectory pair for the page
    os.makedirs(a.out, exist_ok=True)
    ti = doc["channels"].index("t")
    rows = [r for r in doc["data"] if r[ti] <= a.seconds + 1e-9]
    base = copy.deepcopy({k: v for k, v in doc.items() if k != "data"})
    base["data"] = rows
    base["run_id"] = "b2a-fixture"
    base["fixture"] = (f"P3-B2a display fixture: flight = ER {doc['run_id']} c172x g{doc['generation']} (first "
                       f"{a.seconds:g} s); wing nodes from FD's frozen B2a node_layout_b2 / node_layout_b1")
    entries = []
    for gen, L, label in ((0, None, "B1 r1 (B2 default)"), (1, w2, "B2a: dihedral +3, tc_root 1.2, camber_root +1")):
        d = copy.deepcopy(base)
        d["generation"] = gen
        d["fixture_label"] = label
        if L is not None:
            d["genome"] = {**d["genome"], **GENES_B2}
            d["structure"]["node_layout"] = "FD flexbody_b2.node_layout_b2 (P3-B2a)"
            for c in d["structure"]["components"]:
                if c["name"] not in L:
                    continue
                w = L[c["name"]]
                o = off[c["name"]]
                for k in ("axis", "le", "te"):
                    c[f"{k}_nodes_body_m"] = [[round(float(v) * FT + float(oo), 6) for v, oo in zip(pt, o)]
                                              for pt in w[f"{k}_nodes_body_ft"]]
                c["chord_m"] = [round(float(v) * FT, 6) for v in w["chord_ft"]]
                c["geometric_twist_rad"] = [round(math.radians(float(v)), 9) for v in w["geometric_twist_deg"]]
                for k in B2_KEYS:
                    if k in w:
                        c[k] = w[k]
        fn = f"traj_c172x_b2a-fixture_g{gen}.json"
        json.dump(d, open(os.path.join(a.out, fn), "w"))
        entries.append({"generation": gen, "fitness": d["fitness"], "aircraft": "c172x", "file": fn, "label": label})
    json.dump({"schema": "ga-flightsim-traj-index/1", "traj_schema": doc["schema"], "run_id": "b2a-fixture",
               "entries": entries}, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print("trajectories", a.out, [e["file"] for e in entries])


if __name__ == "__main__":
    main()
