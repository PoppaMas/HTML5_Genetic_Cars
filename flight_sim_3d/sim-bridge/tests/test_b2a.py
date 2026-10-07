"""P3-B2a viewer support (FD INTERFACE_v2 section 15.6): B2 keys optional; dihedral from FD's node z; section metadata.
The fixture tests/fixtures/b2a_c172x_node_layout.json comes from FD's FROZEN B2a files (tools/make_b2a_fixture.py)."""
import glob
import json
import math
import os
import shutil
import subprocess

import pytest

SB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(SB, "tests", "fixtures", "b2a_c172x_node_layout.json")
NODE_CHECK = os.path.join(SB, "tools", "build", "jscheck.mjs")
FT = 0.3048
B2_KEYS = ("tc_local", "camber_meq_pct_local", "dihedral_delta_deg", "dihedral_baseline_deg", "section_baseline",
           "area_scale", "aspect_scale")
need_node = pytest.mark.skipif(not shutil.which("node") or not os.path.isdir(os.path.join(SB, "tools", "build", "node_modules", "esbuild")),
                               reason="no node / esbuild")
FX = json.load(open(FIX))


def _doc(layout, genome=None, extra=None, tag="FD flexbody_b2.node_layout_b2 (P3-B2a)"):
    """Minimal trajectory doc (structure only) from fixture wings (FD ft / deg -> m / rad, like ER's b1_node_fields)."""
    comps = []
    for nm in ("wingR", "wingL"):
        w = layout[nm]
        c = {"name": nm, "dof": ["dz", "dx", "twist"], "node_span_frac": w["node_span_frac"],
             "chord_m": [v * FT for v in w["chord_ft"]],
             "geometric_twist_rad": [math.radians(v) for v in w["geometric_twist_deg"]]}
        for k in ("axis", "le", "te"):
            c[f"{k}_nodes_body_m"] = [[v * FT for v in p] for p in w[f"{k}_nodes_body_ft"]]
        for k in B2_KEYS:
            if k in w:
                c[k] = w[k]
        c.update(extra or {})
        comps.append(c)
    return {"schema": "ga-flightsim-traj/2", "aircraft": "c172x", "genome": genome or {},
            "structure": {"node_layout": tag, "modal_twist_sign_fixed": True, "components": comps},
            "channels": ["t"], "data": [[0.0]]}


def _js(tmp_path, *docs):
    fs = []
    for i, d in enumerate(docs):
        f = tmp_path / f"traj_c172x_b2t_g{i}.json"
        f.write_text(json.dumps(d))
        fs.append(str(f))
    out = subprocess.run(["node", NODE_CHECK, *fs], capture_output=True, text=True, check=True, cwd=SB).stdout
    return [json.loads(l) for l in out.strip().splitlines()]


def test_fixture_from_frozen_fd():
    assert FX["b2_default_equals_b1"] is True
    assert set(FX["frozen_md5"]) >= {"planform_b2.py", "flexbody_b2.py", "flexeval_b2.py"}
    md5 = os.path.join(SB, "..", "flight-dynamics", "v2_results", "FROZEN_B2a.md5")
    if os.path.exists(md5):     # fixture made from the files FD froze (same checksums)
        frozen = {l.split()[1]: l.split()[0] for l in open(md5) if len(l.split()) == 2 and len(l.split()[0]) == 32}
        assert FX["frozen_md5"] == frozen
    for nm in ("wingR", "wingL"):
        ck = FX["dihedral_checks"][nm]
        assert ck["xy_unchanged"] and ck["dz_vs_abs_y_minus_y0_max_err_ft"] < 1e-5
        b1 = FX["b1"][nm]["axis_nodes_body_ft"]
        b2 = FX["b2"][nm]["axis_nodes_body_ft"]
        t = math.tan(math.radians(FX["genes_b2"]["wing_dihedral_delta_deg"]))
        for p1, p2 in zip(b1, b2):        # z += -(|y| - |y0|) tan dGamma: tip up (more negative z) on BOTH sides
            assert p2[2] - p1[2] == pytest.approx(-(abs(p1[1]) - abs(b1[0][1])) * t, abs=1e-5)
        assert not any(k in FX["b1"][nm] for k in B2_KEYS)        # B2 default -> no new keys
        assert FX["b2"][nm]["section_baseline"].startswith("NACA")


@need_node
def test_b2_present_vs_absent_and_dihedral_z(tmp_path):
    g = {**FX["genes_b1_shape"], **FX["genes_b2"]}
    a, b = _js(tmp_path, _doc(FX["b1"], FX["genes_b1_shape"], tag="FD flexbody_b1.node_layout_b1 (P3-B1 r1)"), _doc(FX["b2"], g))
    assert a["b2"] is None and a["b2raw"] is None and a["stations"]["hasZ"] is False and a["planform"]["geom"] == "nodes"
    assert b["b2"]["stage"] == "B2a" and b["b2"]["isDefault"] is False and b["stations"]["hasZ"] is True
    # z offsets in the viewer = FD's node z (B2 - B1, ft -> m), not recomputed from dGamma
    want = [(p2[2] - p1[2]) * FT for p1, p2 in zip(FX["b1"]["wingR"]["axis_nodes_body_ft"], FX["b2"]["wingR"]["axis_nodes_body_ft"])]
    want = [w - want[0] for w in want]
    assert b["stations"]["zRelNodes"] == pytest.approx(want, abs=1e-9)
    assert b["stations"]["zAtNodes"] == pytest.approx(want, abs=1e-6)
    assert b["stations"]["zRelNodes"][-1] < -0.2                       # tip up (FRD -z)
    # the planform (x / chord / twist) path is the B1 r1 one in both cases
    for k in ("leAtFirst", "leAtMid", "pivotAtMid", "twistAtMid", "chordRoot", "chordTip"):
        assert a["stations"][k] == pytest.approx(b["stations"][k], abs=1e-9)


@need_node
def test_section_and_hud_fields(tmp_path):
    (b,) = _js(tmp_path, _doc(FX["b2"], {**FX["genes_b1_shape"], **FX["genes_b2"]}))
    w = FX["b2"]["wingR"]
    B = b["b2"]
    assert B["section_baseline"] == w["section_baseline"] and B["camberPos"] == pytest.approx(0.4)
    assert B["dihedral_delta_deg"] == 3.0 and B["dihedral_baseline_deg"] == pytest.approx(w["dihedral_baseline_deg"])
    for end, i in (("root", 0), ("tip", -1)):
        assert B[end]["tc"] == pytest.approx(w["tc_local"][i]) and B[end]["camber"] == pytest.approx(w["camber_meq_pct_local"][i])
        assert B[end]["tmax"] == pytest.approx(w["tc_local"][i], rel=0.01)          # drawn thickness = t/c
        assert B[end]["cmax"] == pytest.approx(w["camber_meq_pct_local"][i] / 100, rel=0.03)   # drawn camber
    assert w["tc_local"][0] == pytest.approx(0.12 * 1.2, rel=1e-6)                 # tc_root 1.2 on a 12 % section
    h = B["hud"]
    assert "B2a dihedral Δ +3.0°" in h and f"baseline {w['dihedral_baseline_deg']:.1f}°" in h and w["section_baseline"] in h
    assert "t/c 14.4→14.4%" in h and "camber 2.8→1.8%c" in h and B["svg"] > 200
    assert B["note"].startswith("B2a: dihedral baked into FD node layout")


@need_node
def test_b2_defaults_behave_like_b1_and_b2b_keys(tmp_path):
    defaults = {"wing_dihedral_delta_deg": 0.0, "wing_tc_root_scale": 1.0, "wing_tc_tip_ratio": 1.0,
                "wing_camber_root_delta_pct": 0.0, "wing_camber_tip_delta_pct": 0.0}
    r1 = _doc(FX["b1"], FX["genes_b1_shape"], tag="FD flexbody_b1.node_layout_b1 (P3-B1 r1)")
    # an exporter writing the B2 keys at default values: treated as absent
    dflt = _doc(FX["b1"], {**FX["genes_b1_shape"], **defaults},
                extra={"dihedral_delta_deg": 0.0, "dihedral_baseline_deg": 1.73, "section_baseline": "NACA 2412",
                       "tc_local": [0.12] * 65, "camber_meq_pct_local": [1.8] * 65})
    b2b = _doc(FX["b2"], {**FX["genes_b1_shape"], **FX["genes_b2"]}, extra={"area_scale": 1.05, "aspect_scale": 0.95})
    a, d, c = _js(tmp_path, r1, dflt, b2b)
    assert d["b2"] is None and d["b2raw"]["isDefault"] is True and d["stations"]["hasZ"] is False
    assert {k: v for k, v in d["stations"].items()} == {k: v for k, v in a["stations"].items()}
    assert c["b2"]["stage"] == "B2b" and c["b2"]["area_scale"] == 1.05 and "area ×1.050 aspect ×0.950" in c["b2"]["hud"]


R1 = sorted(glob.glob(os.path.join(SB, "..", "evolution", "runs", "phase3b1r1-pilot-s1", "trajectories", "traj_*_g59.json")))


@need_node
@pytest.mark.skipif(not R1, reason="no ER B1 r1 pilot trajectories")
def test_no_regression_on_real_b1_r1():
    out = subprocess.run(["node", NODE_CHECK, *R1], capture_output=True, text=True, check=True, cwd=SB).stdout
    for r in map(json.loads, out.strip().splitlines()):
        assert r["b2"] is None and r["b2raw"] is None and r["stations"]["hasZ"] is False
        assert r["planform"]["geom"] == "nodes" and "node_layout_b1" in r["planform"]["node_layout"]
        st = r["stations"]
        assert st["leAtMid"] == pytest.approx(st["midLe"]) and st["pivotAtMid"] == pytest.approx(st["midEa"])
