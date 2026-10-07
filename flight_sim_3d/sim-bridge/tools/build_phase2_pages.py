"""Build the Phase 2 pilot standalone pages from ER's trajectory exports (read-only):

    python tools/build_phase2_pages.py [--runs-root <evolution/runs>] [--out data/]

  data/phase2_pilot_standalone.html     s1 + s2 + s3, gens 0/29/59, scenario 0 (the files ER exports), 10 Hz slim,
                                        display-only structure slim (see colab_viewer._slim_structure) to stay < 40 MB
  data/phase2_pilot_s1_standalone.html  s1 only, gens 0/29/59, 10 Hz slim, full node resolution
  data/phase3b1_smoke_planform_standalone.html  ER's phase3b1-smoke-s1 (REAL P3-B1 planform headers), g0/2/4

Default view: gen 59 of all three aircraft from s1, formation layout, flex x8, chart = ramp error.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import colab_viewer  # noqa: E402
from sim_bridge import paths  # noqa: E402

PARAMS = {"mode": "compare", "preset": "last@phase2-pilot-s1", "cam": "chase", "layout": "formation", "vref": "norm",
          "cy": "rerr", "defl": "8", "spacing": "40"}
STRUCT_SLIM = {"node_stride": 2, "drop_modal": True, "drop_zero": True, "decimals": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=paths.RUNS_ROOT)
    ap.add_argument("--out", default=paths.DATA_DIR)
    ap.add_argument("--only", choices=["seeds", "s1", "p3b1"], default=None)
    a = ap.parse_args()
    root = a.runs_root
    idx = [os.path.join(root, f"phase2-pilot-s{s}", "trajectories", "index.json") for s in (1, 2, 3)]
    if a.only in (None, "seeds"):
        html = colab_viewer.build_standalone_html(idx, gens="all", hz=10.0, params=PARAMS, slim=True,
                                                  struct_slim=STRUCT_SLIM,
                                                  title="Phase 2 pilot: seeds s1/s2/s3, gens 0/29/59")
        p = os.path.join(a.out, "phase2_pilot_standalone.html")
        open(p, "w").write(html)
        print(f"wrote {p} {os.path.getsize(p) / 1e6:.1f} MB")
    if a.only in (None, "s1"):
        prm = {**PARAMS, "preset": "last"}
        html = colab_viewer.build_standalone_html(idx[0], gens="all", hz=10.0, params=prm, slim=True,
                                                  title="Phase 2 pilot s1: gens 0/29/59")
        p = os.path.join(a.out, "phase2_pilot_s1_standalone.html")
        open(p, "w").write(html)
        print(f"wrote {p} {os.path.getsize(p) / 1e6:.1f} MB")

    if a.only in (None, "p3b1"):   # ER's first REAL planform data (P3-B1 smoke, full_a1_b1, 65-node wings)
        p3 = os.path.join(root, "phase3b1-smoke-s1", "trajectories", "index.json")
        if os.path.exists(p3):
            prm = {"mode": "compare", "preset": "last", "cam": "chase", "layout": "formation", "vref": "norm",
                   "cy": "rerr", "defl": "8", "spacing": "40"}
            html = colab_viewer.build_standalone_html(p3, gens="all", hz=10.0, params=prm, slim=True,
                                                      struct_slim={**STRUCT_SLIM, "node_stride": 4},
                                                      title="phase3b1-smoke-s1: P3-B1 planform (real ER data)")
            p = os.path.join(a.out, "phase3b1_smoke_planform_standalone.html")
            open(p, "w").write(html)
            print(f"wrote {p} {os.path.getsize(p) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
