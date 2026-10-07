"""Display-only CG centring (viewer traj.cgDisplayOffset / centreOnFuselage): c172x nodes are measured from an
off-centre CG (asymmetric pointmasses). The viewer shifts node y for display only; the data are not touched."""
import copy
import json
import os
import shutil
import subprocess

import pytest

SB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_CHECK = os.path.join(SB, "tools", "build", "jscheck.mjs")
pytestmark = pytest.mark.skipif(not shutil.which("node") or not os.path.isdir(os.path.join(SB, "tools", "build", "node_modules", "esbuild")),
                                reason="no node / esbuild")


def _doc(dy, fus_bent=False):
    ys = [0.5 + i for i in range(5)]
    wing = lambda s: [[0.0, s * y + dy, -0.6] for y in ys]
    fus = [[2.0 - i, dy + (0.01 * i if fus_bent else 0.0), 0.0] for i in range(5)]
    comps = [{"name": "wingR", "axis_nodes_body_m": wing(1), "le_nodes_body_m": [[0.7, p[1], p[2]] for p in wing(1)],
              "te_nodes_body_m": [[-0.8, p[1], p[2]] for p in wing(1)], "dof": ["dz"]},
             {"name": "wingL", "axis_nodes_body_m": wing(-1), "le_nodes_body_m": [[0.7, p[1], p[2]] for p in wing(-1)],
              "te_nodes_body_m": [[-0.8, p[1], p[2]] for p in wing(-1)], "dof": ["dz"]},
             {"name": "fuselage", "axis_nodes_body_m": fus, "dof": ["dz"]}]
    return {"aircraft": "c172x", "structure": {"components": comps}, "channels": ["t", "wingR.dz.0"], "data": [[0.0, 0.01]]}


def _run(tmp_path, docs):
    fs = []
    for i, d in enumerate(docs):
        f = tmp_path / f"traj_c172x_cg_g{i}.json"
        f.write_text(json.dumps(d))
        fs.append(str(f))
    out = subprocess.run(["node", NODE_CHECK, *fs], capture_output=True, text=True, check=True, cwd=SB).stdout
    return [json.loads(l)["cg"] for l in out.strip().splitlines()]


def test_cg_centre_display_only(tmp_path):
    off, centred, bent, big = _doc(-0.107272), _doc(0.0), _doc(-0.107272, fus_bent=True), _doc(-0.8)
    before = copy.deepcopy(off)
    a, b, c, d = _run(tmp_path, [off, centred, bent, big])
    assert a["dy"] == pytest.approx(0.107272) and a["rootR"] == pytest.approx(0.5) and a["rootL"] == pytest.approx(-0.5)
    assert a["rawRootR"] == pytest.approx(0.5 - 0.107272) and a["sameData"] and a["rawUnchanged"]
    assert b["dy"] == 0 and c["dy"] == 0 and d["dy"] == 0      # centred / not a straight axis / implausibly large: no shift
    assert off == before


R = os.path.join(SB, "..", "evolution", "runs", "phase3b1r1-pilot-s1", "trajectories")


@pytest.mark.skipif(not os.path.isdir(R), reason="no ER pilot trajectories")
def test_real_c172x_symmetric_after_centring():
    fs = [os.path.join(R, f"traj_{ac}_phase3b1r1-pilot-s1_g59.json") for ac in ("c172x", "T38", "737")]
    out = subprocess.run(["node", NODE_CHECK, *fs], capture_output=True, text=True, check=True, cwd=SB).stdout
    c172x, t38, b737 = [json.loads(l)["cg"] for l in out.strip().splitlines()]
    assert c172x["dy"] == pytest.approx(0.107272, abs=1e-6)
    assert c172x["rootR"] == pytest.approx(-c172x["rootL"], abs=1e-6)          # symmetric wings after the shift
    assert t38["dy"] == 0 and b737["dy"] == 0                                   # already centred: untouched
