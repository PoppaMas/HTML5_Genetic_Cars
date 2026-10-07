#!/usr/bin/env python3
"""B2a fixture page + screenshots (after tools/make_b2a_fixture.py):
    $PY tools/build_b2a_page.py   -> data/b2a_fixture_standalone.html, screenshots/b2a_*.png (screenshots.py --strict)
g0 = B1 r1 geometry, g1 = FD B2a geometry (dihedral +3, tc_root 1.2, camber_root +1) on the same c172x flight."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import colab_viewer  # noqa: E402

idx = os.path.join(HERE, "data", "b2a_fixture", "index.json")
page = os.path.join(HERE, "data", "b2a_fixture_standalone.html")
html = colab_viewer.build_standalone_html(idx, gens="0,1", hz=10, params={
    "mode": "compare", "preset": "pair:c172x", "cam": "chase", "layout": "formation", "defl": "1", "spacing": "15"},
    title="P3-B2a viewer fixture: c172x B1 r1 (g0) vs B2a (g1)")
open(page, "w").write(html)
print(f"page {page}: {len(html.encode()) / 1e6:.2f} MB")
spec = [
    ["compare_chase_g0_r1_vs_g1_b2a", "", 8],
    ["compare_top_g0_r1_vs_g1_b2a", "?preset=plan:c172x&cam=top&defl=1&spacing=15", 8],
    ["single_g1_b2a_rear", "?mode=single&gen=c172x:1&cam=orbit&defl=1", 8, "rear+fit"],
    ["single_g0_r1_rear", "?mode=single&gen=c172x:0&cam=orbit&defl=1", 8, "rear+fit"],
    ["single_g1_b2a_front_defl8", "?mode=single&gen=c172x:1&cam=orbit&defl=8", 8, "front+fit"],
]
sp = "/tmp/b2a_shots.json"
json.dump(spec, open(sp, "w"))
py = os.path.join(HERE, ".venv-shots", "bin", "python")
p = subprocess.run([py, os.path.join(HERE, "tools", "screenshots.py"), "--bench", page, "--shot-spec", sp, "--prefix", "b2a_",
                    "--strict"], capture_output=True, text=True)
open("/tmp/b2a_shots.log", "w").write(p.stdout + p.stderr)
print("\n".join(l for l in p.stdout.splitlines() if l.startswith(("RESULT", "BLANK", "AIRCRAFT"))) or p.stderr[-2000:])
sys.exit(p.returncode)
