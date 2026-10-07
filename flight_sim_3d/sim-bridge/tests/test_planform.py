"""P3-B1 planform header loader (sim_bridge.planform + viewer traj.parsePlanform / aircraft.planformStations),
the display-only structure slim used by the multi-seed page, and the data-driven wingR_modal twist-sign handling
(pre-fix vs post-fix ER files)."""
import glob
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

SB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SB)
import colab_viewer as cv  # noqa: E402
from sim_bridge import planform as P, trajdiff as T  # noqa: E402
from sim_bridge.recorder import TrajRecorder  # noqa: E402

FIX = os.path.join(SB, "tests", "fixtures", "planform_synthetic")
NODE_CHECK = os.path.join(SB, "tools", "build", "jscheck.mjs")


def _load(name):
    return json.load(open(os.path.join(FIX, name)))


# ---------------------------------------------------------------- python side of the header block
def test_fixture_blocks_valid_and_synthetic():
    shaped = _load("traj_737_planform-synthetic_shaped.json")
    f, blk = P.find_planform(shaped)
    assert f == "planform" and blk["schema"] == P.PLANFORM_SCHEMA and blk["synthetic"] is True
    assert P.validate_planform(blk) == []
    s = P.summary(blk)
    assert s["n_strips"] == 64 and abs(s["sweep_qc_deg"] - 30.0) < 1e-9 and 0.1 < s["taper"] < 0.3
    assert s["twist_tip_deg"] < -3.5          # washout (genes: tip -4 deg; strip centre slightly inboard)
    c = _load("traj_c172x_planform-synthetic_shaped.json")
    f, blk = P.find_planform(c)
    assert f == "planform_b1" and blk["symmetric"] is True and "wing" in blk and P.validate_planform(blk) == []
    assert P.find_planform(_load("traj_737_planform-synthetic_baseline.json")) == (None, None)


def test_from_fd_strips_units_and_frame():
    blk = P.from_fd_strips([0.0, 1.0], [2.0, 10.0], [5.0, 2.0], [-1.25, 3.0], [0.0, -0.05], 20.0, 1.5,
                           genes={"wing_sweep_qc_delta_deg": 2.0}, synthetic=True)
    w = blk["wingR"]
    assert "wing" in P.from_fd_strips([0.0, 1.0], [2.0, 10.0], [5.0, 2.0], [0.0, 3.0], [0.0, 0.0], 0.0, side_key="wing")
    assert w["y_m"] == pytest.approx([0.6096, 3.048]) and w["chord_m"] == pytest.approx([1.524, 0.6096])
    # FD LE x is measured aft; body FRD x is forward: LE forward of the root quarter-chord point -> larger x
    assert w["le_x_m"] == pytest.approx([1.5 + 1.25 * 0.3048, 1.5 - 3.0 * 0.3048])
    assert blk["sweep_qc_rad"] == pytest.approx(math.radians(20.0)) and blk["symmetric"] and blk["synthetic"]
    assert P.validate_planform(blk) == []


@pytest.mark.parametrize("mut,frag", [
    (lambda b: b["wingR"].__setitem__("chord_m", [1.0, -0.1]), "chord_m"),
    (lambda b: b["wingR"].__setitem__("y_m", [3.0, 1.0]), "increase outboard"),
    (lambda b: b["wingR"].__setitem__("twist_rad", [0.0]), "twist_rad"),
    (lambda b: b.pop("wingR"), "no wingR"),
    (lambda b: b.__setitem__("sweep_qc_rad", float("nan")), "sweep_qc_rad"),
])
def test_validate_rejects(mut, frag):
    blk = P.from_fd_strips([0.0, 1.0], [2.0, 10.0], [5.0, 2.0], [0.0, 3.0], [0.0, 0.0], 0.0, 0.0)
    mut(blk)
    assert any(frag in e for e in P.validate_planform(blk))


def test_standalone_build_keeps_planform():
    html = cv.build_standalone_html(os.path.join(FIX, "index.json"), gens="all", hz=None, slim=True)
    assert html.count('"schema":"fd-planform/1"') == 2 and '"planform_b1":{' in html


# ---------------------------------------------------------------- viewer (node) side
@pytest.mark.skipif(not shutil.which("node") or not os.path.isdir(os.path.join(SB, "tools", "build", "node_modules", "esbuild")),
                    reason="node / esbuild (tools/build) not available")
def test_viewer_parse_and_stations():
    files = sorted(glob.glob(os.path.join(FIX, "traj_*.json")))
    out = subprocess.run(["node", NODE_CHECK, *files], capture_output=True, text=True, check=True, cwd=SB).stdout
    res = {r["file"]: r for r in map(json.loads, out.strip().splitlines())}
    assert res["traj_737_planform-synthetic_baseline.json"]["planform"] is None      # absent -> ignored
    a = res["traj_737_planform-synthetic_shaped.json"]
    b = res["traj_c172x_planform-synthetic_shaped.json"]
    assert a["planform"]["field"] == "planform" and b["planform"]["field"] == "planform_b1"
    assert a["planform"]["symmetric"] and not a["planform"]["sameLR"]                  # explicit equal L / R
    assert b["planform"]["symmetric"] and b["planform"]["sameLR"]                      # mirrored from wingR
    for r in (a, b):
        st = r["stations"]
        assert st["f0"] == 0 and st["fEnd"] == 1 and st["nStations"] >= 64
        assert st["chordAtFirstStrip"] == pytest.approx(st["firstStripChord"])
        assert st["chordAtLastStrip"] == pytest.approx(st["lastStripChord"])
        assert st["dleTip"] < 0 and st["twistTip"] < 0      # swept back (LE moves aft = -x), washout
        assert r["planform"]["synthetic"] is True


from sim_bridge import paths as _paths  # noqa: E402
REAL = sorted(glob.glob(os.path.join(_paths.RUNS_ROOT, "phase3b1*", "trajectories", "traj_*.json")))


@pytest.mark.skipif(not REAL, reason="no ER phase3b1* trajectories with real planform data here")
def test_real_er_planform_headers():
    """ER's real P3-B1 exports (read-only): every file carries a valid fd-planform/1 block in ER's form."""
    for f in REAL:
        doc = json.load(open(f))
        field, blk = P.find_planform(doc)
        assert field == "planform" and blk["symmetric"] is True and "wing" in blk, f
        assert P.validate_planform(blk) == [], (f, P.validate_planform(blk))
        assert len(blk["wing"]["chord_m"]) == 64 and not blk.get("synthetic")
        wings = {c["name"]: len(c["axis_nodes_body_m"]) for c in doc["structure"]["components"]}
        assert wings.get("wingR") == 65 and wings.get("wingL") == 65


@pytest.mark.skipif(not REAL or not shutil.which("node") or
                    not os.path.isdir(os.path.join(SB, "tools", "build", "node_modules", "esbuild")),
                    reason="no real planform files or no node / esbuild")
def test_viewer_parses_real_er_planform():
    out = subprocess.run(["node", NODE_CHECK, *REAL[:3]], capture_output=True, text=True, check=True, cwd=SB).stdout
    for r in map(json.loads, out.strip().splitlines()):
        pf, st = r["planform"], r["stations"]
        assert pf["field"] == "planform" and pf["symmetric"] and pf["sameLR"] and pf["n"] == 64 and not pf["synthetic"]
        assert st["chordAtFirstStrip"] == pytest.approx(st["firstStripChord"])
        assert st["chordAtLastStrip"] == pytest.approx(st["lastStripChord"])


# ---------------------------------------------------------------- display-only structure slim
def _doc_with_wings(n=33, frames=3):
    comps = [{"name": "wingR", "axis_nodes_body_m": [[0, i, 0] for i in range(n)], "dof": ["dz", "dy", "twist"],
              "node_span_frac": [i / (n - 1) for i in range(n)]},
             {"name": "wingR_modal", "axis_nodes_body_m": [[0, i, 0] for i in range(9)], "dof": ["dz"]}]
    ch = ["t"] + [f"wingR.{d}.{i}" for d in ("dz", "dy", "twist") for i in range(n)] + [f"wingR_modal.dz.{i}" for i in range(9)]
    data = []
    for k in range(frames):
        row = [0.1 * k]
        for d in ("dz", "dy", "twist"):
            row += [(0.0 if d == "dy" else 0.001 * i * (k + 1) + 1.23456789e-5) for i in range(n)]
        row += [0.5] * 9
        data.append(row)
    return {"channels": ch, "data": data, "structure": {"components": comps}}


def test_slim_structure_stride_modal_zero_decimals():
    d = cv._slim_structure(_doc_with_wings(), {"node_stride": 2, "drop_modal": True, "drop_zero": True, "decimals": 4})
    comps = {c["name"]: c for c in d["structure"]["components"]}
    assert set(comps) == {"wingR"} and len(comps["wingR"]["axis_nodes_body_m"]) == 17
    assert comps["wingR"]["node_span_frac"][-1] == 1.0 and comps["wingR"]["display_nodes_from"] == 33
    assert not any(c.startswith(("wingR_modal.", "wingR.dy.")) for c in d["channels"])
    j = d["channels"].index("wingR.dz.16")              # node 32 (tip) renumbered to 16
    assert d["data"][2][j] == round(0.001 * 32 * 3 + 1.23456789e-5, 4)
    assert "wingR.dz.17" not in d["channels"]
    assert cv._slim_structure(_doc_with_wings(), None)["channels"] == _doc_with_wings()["channels"]


# ---------------------------------------------------------------- wingR_modal twist sign (pre-fix / post-fix ER)
def _modal_doc(sign, n=200):
    t = np.linspace(0, 10, n)
    fe = 0.01 + 0.005 * np.sin(t)
    ch = ["t", "wingR.twist.0", "wingR.twist.1", "wingR_modal.twist.0", "wingR_modal.twist.1",
          "wingL.twist.0", "wingL.twist.1", "wingL_modal.twist.0", "wingL_modal.twist.1"]
    data = np.c_[t, 0 * t, fe, 0 * t, sign * fe, 0 * t, -fe, 0 * t, -fe].tolist()
    return {"channels": ch, "data": data}


def test_modal_twist_sign_detection():
    assert T.modal_twist_sign(_modal_doc(+1)["channels"], _modal_doc(+1)["data"]) == 1
    assert T.modal_twist_sign(_modal_doc(-1)["channels"], _modal_doc(-1)["data"]) == -1
    assert T.modal_twist_convention(_modal_doc(-1)) == {"wingR_modal": -1, "wingL_modal": 1, "pre_fix": True, "source": "data"}
    z = _modal_doc(+1)
    z["data"] = [[r[0]] + [0.0] * 8 for r in z["data"]]
    assert T.modal_twist_sign(z["channels"], z["data"]) is None              # no twist -> undecidable
    assert T.modal_twist_sign(["t", "wingR.twist.0"], [[0, 0]]) is None      # no modal channels


def test_diff_undoes_pre_fix_reference_sign_only():
    post, pre = _modal_doc(+1), _modal_doc(-1)
    r = T.diff(post, pre)
    assert r["wingR_modal_twist_sign"] == {"a": 1, "b": -1, "compared_sign_corrected": True}
    assert r["fail"] == [] and r["channels"]["wingR_modal.twist.1"]["max_abs"] == 0.0
    r = T.diff(post, post)
    assert not r["wingR_modal_twist_sign"]["compared_sign_corrected"] and r["fail"] == []


@pytest.mark.parametrize("sign", [1, -1])
def test_recorder_modal_metric_follows_detected_sign(sign):
    rec = TrajRecorder.__new__(TrajRecorder)
    rec._geo = {"components": {"wingR": {"node_span_frac": [0.0, 0.5, 1.0]}, "wingL": {"node_span_frac": [0.0, 0.5, 1.0]}}}
    rec.v2_stats = {"wing_modal_vs_nodal": {}}
    rec._modal_acc = {}
    for k in range(20):
        fe = 0.01 + 0.001 * k
        ch = {}
        for side, s in (("R", sign), ("L", 1)):
            fsgn = 1 if side == "R" else -1
            for i, v in enumerate((0.0, fe / 2, fe)):
                ch[f"wing{side}.dz.{i}"] = -v
                ch[f"wing{side}.twist.{i}"] = fsgn * v
                ch[f"wing{side}_modal.dz.{i}"] = -v
                ch[f"wing{side}_modal.twist.{i}"] = s * fsgn * v
        rec._compare_wings_modal(ch)
    st = rec.v2_stats["wing_modal_vs_nodal"]
    assert st["wingR"]["modal_twist_sign"] == sign and st["wingR"]["modal_twist_sign_corrected"] == (sign == -1)
    assert st["wingR"]["twist_max_abs_diff_rad"] == 0.0 and st["wingL"]["modal_twist_sign"] == 1


# ---------------------------------------------------------------- B1 r1 per-node wing geometry (node_layout_b1)
def _r1_node_doc():
    """Tiny doc with FD r1-style per-node wing geometry (absolute body FRD) and a strip planform that differs."""
    n = 5
    ys = [1.0 + 2.0 * i for i in range(n)]
    chord = [3.0 - 0.5 * i for i in range(n)]
    le = [[2.0 - 0.6 * i, y, -1.5] for i, y in enumerate(ys)]
    te = [[le[i][0] - chord[i], y, -1.5] for i, y in enumerate(ys)]
    ax = [[le[i][0] - 0.4 * chord[i], y, -1.5] for i, y in enumerate(ys)]
    tw = [0.0, 0.004, 0.008, 0.006, 0.003]
    comp = lambda nm, s: {"name": nm, "axis_nodes_body_m": [[p[0], s * p[1], p[2]] for p in ax],
                          "le_nodes_body_m": [[p[0], s * p[1], p[2]] for p in le],
                          "te_nodes_body_m": [[p[0], s * p[1], p[2]] for p in te],
                          "chord_m": chord, "geometric_twist_rad": tw, "dof": ["dz", "dx", "twist"],
                          "node_span_frac": [i / (n - 1) for i in range(n)]}
    strips = {"span_frac": [0.1, 0.9], "y_m": [1.0, 9.0], "chord_m": [2.0, 1.0], "le_x_m": [0.0, -1.0],
              "twist_rad": [0.0, 0.0]}
    return {"aircraft": "737", "structure": {"node_layout": "FD flexbody_b1.node_layout_b1 (P3-B1 r1)",
                                              "modal_twist_sign_fixed": True,
                                              "components": [comp("wingR", 1), comp("wingL", -1)]},
            "planform": {"schema": P.PLANFORM_SCHEMA, "source": "P3-B1", "genes": {}, "symmetric": True,
                         "wing": strips, "sweep_qc_rad": 0.4}, "channels": ["t"], "data": [[0.0]]}


@pytest.mark.skipif(not shutil.which("node") or
                    not os.path.isdir(os.path.join(SB, "tools", "build", "node_modules", "esbuild")),
                    reason="no node / esbuild")
def test_viewer_uses_r1_per_node_le_te(tmp_path):
    """Per-node LE/TE present -> wing drawn from them (absolute body x, twist pivot = elastic axis); without them ->
    strips (relative LE)."""
    d = _r1_node_doc()
    f1 = tmp_path / "traj_737_r1nodes_g0.json"
    f1.write_text(json.dumps(d))
    d2 = json.loads(json.dumps(d))
    for c in d2["structure"]["components"]:
        for k in ("le_nodes_body_m", "te_nodes_body_m", "chord_m", "geometric_twist_rad"):
            c.pop(k)
    d2["structure"].pop("node_layout")
    f2 = tmp_path / "traj_737_r0strips_g0.json"
    f2.write_text(json.dumps(d2))
    out = subprocess.run(["node", NODE_CHECK, str(f1), str(f2)], capture_output=True, text=True, check=True, cwd=SB).stdout
    a, b = map(json.loads, out.strip().splitlines())
    pa, sa = a["planform"], a["stations"]
    assert pa["geom"] == "nodes" and pa["n"] == 5 and sa["absolute"] is True
    assert sa["leAtFirst"] == pytest.approx(2.0) and sa["leAtMid"] == pytest.approx(2.0 - 0.6 * 2)
    assert sa["pivotAtMid"] == pytest.approx((2.0 - 1.2) - 0.4 * 2.0)          # elastic axis, not the quarter chord
    assert sa["twistAtMid"] == pytest.approx(0.008) and sa["chordAtFirstStrip"] == pytest.approx(3.0)
    assert pa["taper"] == pytest.approx(0.5)            # HUD taper still from the planform strips
    pb, sb = b["planform"], b["stations"]
    assert pb["geom"] == "strips" and pb["n"] == 2 and sb["absolute"] is False and sb["chordAtFirstStrip"] == pytest.approx(2.0)


R1 = sorted(glob.glob(os.path.join(_paths.RUNS_ROOT, "phase3b1r1-*", "trajectories", "traj_*.json")))


@pytest.mark.skipif(not R1 or not shutil.which("node") or
                    not os.path.isdir(os.path.join(SB, "tools", "build", "node_modules", "esbuild")),
                    reason="no ER phase3b1r1-* trajectories or no node / esbuild")
def test_viewer_reads_real_r1_node_geometry():
    out = subprocess.run(["node", NODE_CHECK, *R1[:3]], capture_output=True, text=True, check=True, cwd=SB).stdout
    for r in map(json.loads, out.strip().splitlines()):
        pf, st = r["planform"], r["stations"]
        assert pf["geom"] == "nodes" and "node_layout_b1" in pf["node_layout"] and pf["n"] == 65
        assert st["leAtFirst"] == pytest.approx(st["firstLe"]) and st["leAtMid"] == pytest.approx(st["midLe"])
        assert st["pivotAtMid"] == pytest.approx(st["midEa"]) and st["twistAtMid"] == pytest.approx(st["midTwist"])


def test_slim_structure_subsamples_per_node_arrays():
    d = _r1_node_doc()
    n = 21
    c = d["structure"]["components"][0]
    for k in ("axis_nodes_body_m", "le_nodes_body_m", "te_nodes_body_m", "chord_m", "geometric_twist_rad", "node_span_frac"):
        c[k] = [c[k][0]] * n
    d["structure"]["components"] = [c]
    d["channels"] = ["t"] + [f"wingR.dz.{i}" for i in range(n)]
    d["data"] = [[0.0] + [0.1 * i for i in range(n)]]
    out = cv._slim_structure(d, {"node_stride": 4})
    oc = out["structure"]["components"][0]
    lens = {k: len(oc[k]) for k in ("axis_nodes_body_m", "le_nodes_body_m", "te_nodes_body_m", "chord_m",
                                    "geometric_twist_rad", "node_span_frac")}
    assert set(lens.values()) == {6}, lens          # 0,4,8,12,16,20
    assert oc["dof"] == ["dz", "dx", "twist"]


def test_declared_modal_sign_trusted():
    from sim_bridge import trajdiff as TD
    d = _r1_node_doc()
    assert TD.declared_modal_sign(d) == 1
    assert TD.modal_twist_convention(d)["source"] == "header"
    d["structure"]["modal_twist_sign_fixed"] = False
    assert TD.declared_modal_sign(d) is None and TD.modal_twist_convention(d)["source"] == "data"
