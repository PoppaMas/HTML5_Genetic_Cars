"""Make the SYNTHETIC planform test fixture (tests/fixtures/planform_synthetic/).

    PYTHONDONTWRITEBYTECODE=1 python tools/make_planform_fixture.py

ER has not exported real planform data yet (P3-B wiring waits on the B1 gene list / shape API), so this builds the
optional `planform` header the agreed way from FD's own B1 decode (flight-dynamics/planform_b1.py +
flexbody_b1.FlexBodyModelB1, imported read-only, no bytecode) on top of short, decimated copies of real
phase2-pilot-s1 g59 flights. The flight data is real; the planform is NOT what those aircraft flew (real planform data: ER's phase3b1-smoke-s1) -> every block is
marked `synthetic: true`. Three entries:
  737  shaped  (header field `planform`, wingR + wingL)            tapers 0.85, twist mid -1 / tip -4 deg, sweep +5 deg
  737  baseline (no planform field: the viewer must ignore it)
  c172x shaped (header field `planform_b1`, symmetric, `wing` = ER's form) tapers 0.9/0.9/0.85, twist tip -3, sweep +3
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.dirname(HERE)
sys.path.insert(0, SB)
sys.dont_write_bytecode = True
import colab_viewer as cv  # noqa: E402
from sim_bridge import paths, planform as P  # noqa: E402

OUT = os.path.join(SB, "tests", "fixtures", "planform_synthetic")
RUN = "phase2-pilot-s1"
SHAPES = {
    "737": {"wing_chord_taper_1": 0.85, "wing_chord_taper_2": 0.85, "wing_chord_taper_3": 0.85,
            "wing_twist_mid_deg": -1.0, "wing_twist_tip_deg": -4.0, "wing_sweep_qc_delta_deg": 5.0},
    "c172x": {"wing_chord_taper_1": 0.9, "wing_chord_taper_2": 0.9, "wing_chord_taper_3": 0.85,
              "wing_twist_mid_deg": -0.5, "wing_twist_tip_deg": -3.0, "wing_sweep_qc_delta_deg": 3.0},
}


def fd_planform(ac, genes, side_key="wingR"):
    sys.path.insert(0, paths.FLIGHT_DYNAMICS_DIR)
    import flexbody_b1  # noqa: E402  (FD, read-only)
    m = flexbody_b1.FlexBodyModelB1(ac, None, shape_genes=genes)
    pf = m.planform
    blk = P.from_fd_strips(pf.xi, pf.y_ft, pf.c_ft, pf.le_x_ft, pf.twist_rad, pf.sweep_qc_deg, 0.0,
                           genes=m.shape_genes, synthetic=True, side_key=side_key,
                           extra={"fd_summary": m.planform_summary(),
                                  "note": "SYNTHETIC test fixture: FD planform_b1 decode of these genes laid over a "
                                          "real phase2-pilot-s1 flight that did NOT fly this planform; le_x_m "
                                          "relative to the root quarter-chord point (+fwd), as ER writes it"})
    return blk


def short(doc, t_end=40.0, hz=5.0):
    """First `t_end` s at `hz`, slim channels, wings only (every 4th FE node + tip, 4 decimals)."""
    d = cv._decimate(doc, hz)
    ti = d["channels"].index("t")
    d = {**d, "data": [r for r in d["data"] if r[ti] <= t_end]}
    d = cv._slim(cv._slim_structure(d, {"node_stride": 4, "drop_modal": True, "drop_zero": True, "decimals": 4}),
                 cv.SLIM_EXTRAS)
    wings = ("wingR", "wingL")
    idx = [i for i, c in enumerate(d["channels"]) if c.count(".") != 2 or c.split(".")[0] in wings]
    st = d["structure"]
    return {**d, "structure": {**st, "components": [c for c in st["components"] if c["name"] in wings]},
            "channels": [d["channels"][i] for i in idx], "data": [[r[i] for i in idx] for r in d["data"]]}


def main():
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(paths.RUNS_ROOT, RUN, "trajectories")
    entries = []
    for ac, tag, field in (("737", "shaped", "planform"), ("737", "baseline", None), ("c172x", "shaped", "planform_b1")):
        doc = json.load(open(os.path.join(src, f"traj_{ac}_{RUN}_g59.json")))
        d = short(doc)
        d = {**d, "run_id": "planform-synthetic", "individual_id": tag,
             "fixture_note": f"SYNTHETIC planform fixture ({tag}); flight = {RUN} g59 scenario 0, first 40 s, 5 Hz"}
        if field:
            blk = fd_planform(ac, SHAPES[ac], "wingR" if field == "planform" else "wing")
            if field == "planform":  # explicit both sides (mirror y)
                blk["wingL"] = {**blk["wingR"], "y_m": [-v for v in blk["wingR"]["y_m"]]}
                blk["symmetric"] = False
            d[field] = blk
            assert not P.validate_planform(blk), P.validate_planform(blk)
        fn = f"traj_{ac}_planform-synthetic_{tag}.json"
        json.dump(d, open(os.path.join(OUT, fn), "w"), separators=(",", ":"))
        entries.append({"file": fn, "aircraft": ac, "generation": 59, "fitness": d.get("fitness"),
                        "individual_id": tag, "is_best": False, "scenario": 0})
        print(fn, os.path.getsize(os.path.join(OUT, fn)) // 1024, "KiB")
    json.dump({"schema": "ga-flightsim-traj-index/1", "traj_schema": "ga-flightsim-traj/2", "run_id": "planform-synthetic",
               "fitness_sense": "min", "synthetic": True, "trajectories": entries},
              open(os.path.join(OUT, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
