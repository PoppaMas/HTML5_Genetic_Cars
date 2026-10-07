"""Cross-check a full-fidelity replay trajectory against FD's OWN node export.

    python tools/verify_fd_nodes.py <replay trajectory .json> [--run-dir <ER run dir>] [--json out.json]

Re-flies the same genome / scenario with Flight Dynamics' flexeval.evaluate(record=True) (FD's exporter, the
'nodes' block fd-flexbody-nodes/1 and the full-rate 'structure' histories; arguments prepared exactly as
evolution.fidelity.evaluate_genome prepares them), maps every FD frame with sim_bridge.v2_map and compares it with
the replay's recorded channels (frame k = the coupler step ending at t = (k+1) * sim_dt). Also compares the node
layout and FD's per-scenario sim_cost with the replayed cost. FD / ER are imported read-only (no bytecode).
Exit 1 if any channel differs by more than the written rounding (5e-7) + 1e-8 (FD writes nodes to 9 decimals).
"""
import argparse
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sim_bridge import paths  # noqa: E402

paths.ensure_import_paths()
from sim_bridge import v2_map as vm  # noqa: E402
from sim_bridge.replay import eval_run_cfg  # noqa: E402


def fd_record(doc, run_dir):
    from evolution import eval as E, fidelity as F, runinfo, sim
    with open(os.path.join(run_dir, "run.json")) as f:
        rc, _ = eval_run_cfg(json.load(f))
    ac = doc["aircraft"]
    entry = E._aircraft_entry(rc, ac)
    gains, struct = E.split_values(doc["genome"], E.gene_groups(entry))
    P = sim.Profile.from_dict(E._profile_d(entry))
    sc = next(s for s in rc["scenarios"] if s["id"] == doc["scenario_id"])
    scs = [sim.Scenario.from_dict(runinfo.scenario_fields(sc))]
    m = F.fd_modules()
    root, rv2 = F.roots(P)
    with F._gate(None):
        r = m["fe"].evaluate(gains, F.struct_from(struct), scs, ac, fidelity="full", root=root, root_v2=rv2, profile=P,
                             sim=sim, record=True, blas_threads=1)
    return r


def compare(doc, r):
    tel = r["telemetry"][0]
    nodes, hist = tel["nodes"], tel["structure"]
    geo = vm.geometry_from_layout(nodes, doc["aircraft"])
    ch = doc["channels"]
    ti = ch.index("t")
    ours = {c["name"]: c for c in doc["structure"]["components"]}
    lay = {nm: max(math.dist(p, q) for p, q in zip(ours[nm]["axis_nodes_body_m"], geo["components"][nm]["axis_nodes_body_m"]))
           for nm in geo["components"] if nm in ours}
    sim_dt = float(doc["sim_dt_s"])
    n_frames = len(next(iter(next(iter(nodes["values"].values())).values())))
    worst, worst_ch, rows = 0.0, None, 0
    mapped_names = [c for c in ch if c.split(".")[0] in vm.COMPONENTS or c.startswith("struct.")]
    for row in doc["data"]:
        k = int(round(row[ti] / sim_dt)) - 1
        if k < 0 or k >= n_frames:
            continue
        raw = {key: hist[key][k] for key in hist}
        m = vm.map_v2_record(raw, geo, vm.nodes_frame(nodes["values"], k))
        rows += 1
        for c in mapped_names:
            j = ch.index(c)
            if row[j] is None or c not in m:
                continue
            d = abs(row[j] - m[c])
            if d > worst:
                worst, worst_ch = d, (c, row[ti])
    per = r["per_scenario"][0]
    return {"aircraft": doc["aircraft"], "individual_id": doc.get("individual_id"), "scenario_id": doc.get("scenario_id"),
            "fd_model_version": r["model_version"], "replay_model_version": (doc.get("replay") or {}).get("model_version"),
            "fd_cost": per["cost"], "fd_sim_cost": per["sim_cost"], "replay_cost": doc.get("scenario_cost"),
            "cost_bit_identical": per["cost"] == doc.get("scenario_cost"),
            "fd_node_frames": n_frames, "rows_compared": rows, "channels_compared": len(mapped_names),
            "max_abs_diff": worst, "worst": worst_ch, "layout_max_dist_m": max(lay.values()) if lay else None,
            "ok": worst <= 5e-7 + 1e-8 and (max(lay.values()) if lay else 0.0) <= 5e-7 + 1e-8}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj")
    ap.add_argument("--run-dir")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    with open(a.traj) as f:
        doc = json.load(f)
    run_dir = a.run_dir or os.path.join(paths.RUNS_ROOT, doc["run_id"])
    rep = compare(doc, fd_record(doc, run_dir))
    print(json.dumps(rep, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
