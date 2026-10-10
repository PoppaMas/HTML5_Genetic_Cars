"""Phase 4 smoke replay pages (ER phase4-smoke-s1 / -s2).
    PYTHONDONTWRITEBYTECODE=1 $PY tools/make_phase4_smoke_page.py [s1|s2] [--no-shots]
Reads evolution/runs/<run>/trajectories (read-only) and writes DISPLAY copies to data/<run>/ with:
  * the course REGENERATED from (course_seed, stage, aircraft, version) and checked against the embedded rings (<= 1 mm);
  * FD ctrl_surfaces (30 Hz) merged as channels elev_deg/ail_deg/rud_deg[/ail_L_deg/ail_R_deg] (SURF HUD);
  * ER's generation kept (5); each flight labelled 'train · <stage> · <passes>/15' or 'hold-out · <stage> · <passes>/15'
    (index entry `label`, shown in the run list and HUD; URL gen=T38:5@train / gen=T38:5@hold-out).
Gates/costs are ER's logged values. Replay proof (ER's frozen run code, fresh process): /workspace/b1work/<proofdir>/proof.json."""
import bisect, json, os, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sim_bridge import ring_course as C
RUNS = {
    "s1": {"run": "phase4-smoke-s1", "proof": "/workspace/b1work/p4smoke/proof.json", "code": "/workspace/er_smoke_code_p4",
           "prefix": "phase4_smoke_", "scoring": "ring_course 1.1 (pre per-ring timeout / legacy 32-bit seeds)"},
    "s2": {"run": "phase4-smoke-s2", "proof": "/workspace/b1work/p4smoke_s2/proof.json", "code": "/workspace/er_smoke_code_p4_s2",
           "prefix": "phase4_smoke_s2_", "scoring": "ring_course 1.2 (per-ring timeout; seed_scheme run_seed_v2)"},
}
SURF = ("elev_deg", "ail_deg", "rud_deg", "ail_L_deg", "ail_R_deg")
LAB = {"train": "train", "holdout": "hold-out"}


def main():
    key = next((a for a in sys.argv[1:] if a in RUNS), "s1")
    R = RUNS[key]
    run_dir = os.path.join(HERE, "..", "evolution", "runs", R["run"])
    out = os.path.join(HERE, "data", R["run"])
    page = os.path.join(HERE, "data", f"{R['run']}_standalone.html")
    os.makedirs(out, exist_ok=True)
    proof = json.load(open(R["proof"]))
    runj = json.load(open(os.path.join(run_dir, "run.json"))) if os.path.exists(os.path.join(run_dir, "run.json")) else {}
    idx = json.load(open(os.path.join(run_dir, "trajectories", "index.json")))
    entries, info = [], {}
    for e in idx["entries"]:
        f = e["file"]
        d = json.load(open(os.path.join(run_dir, "trajectories", f)))
        pr = proof[f]
        assert pr["exact"] and pr["file_identical"], f
        cb = d["course"]
        c = C.regenerate(cb)
        worst = C.check_rings(c, cb["rings"])
        assert len(cb["rings"]) == cb["M"]
        cs, ct = d["ctrl_surfaces"], d["ctrl_surfaces"]["t"]
        surf = [k for k in SURF if k in cs]          # f16: single fcs/aileron-pos (no per-side channels)
        d["channels"] = list(d["channels"]) + surf
        for row in d["data"]:
            j = max(0, bisect.bisect_right(ct, row[0] + 1e-9) - 1)   # last ctrl sample at or before t (sample-and-hold)
            row.extend(float(cs[k][j]) for k in surf)
        p4, gs = d["p4"], d["gate_summary"]
        lab = LAB[p4["label"]]
        label = f"{lab} · {p4['stage']} · {gs['passes']}/{gs['M']}"
        d["course"]["course_seed"] = cb.get("course_seed", p4["course_seed"])
        d["course"]["sb_check"] = {"regenerated_by": C.VERSION, "max_ring_diff_m": worst, "display_label": label}
        d["events"] = list(d["events"]) + [{"t": g["t"], "type": "gate_pass" if g["result"] == "pass" else "gate_miss",
                                             "detail": f"ring {g['k']} {g['result']}" + (f" miss {g['miss_m']:.1f} m" if g.get("miss_m") else "")}
                                            for g in d["gates"]]
        d["events"].sort(key=lambda x: x["t"])
        fn = f"{d['aircraft']}_{p4['label']}.json"
        json.dump(d, open(os.path.join(out, fn), "w"), separators=(",", ":"))
        entries.append({"generation": d["generation"], "fitness": d["fitness"], "aircraft": d["aircraft"], "file": fn,
                        "label": label})
        info[fn] = {"label": label, "seed": p4["course_seed"], "stage": p4["stage"], "cost": d["scenario_cost"],
                    "passes": gs["passes"], "misses": gs["misses"], "M": gs["M"], "ring_diff_m": worst,
                    "version": cb["version"], "course_seed_in_block": cb.get("course_seed"),
                    "provenance": {k: cb.get(k) for k in ("run_seed", "gen", "k")},
                    "first_pass_t": next((g["t"] for g in d["gates"] if g["result"] == "pass"), None),
                    "t_end": d["data"][-1][0], "gen": d["generation"]}
    json.dump({"schema": "ga-flightsim-traj-index/1", "traj_schema": "ga-flightsim-traj/2", "run_id": R["run"],
               "fitness_sense": "min", "entries": entries}, open(os.path.join(out, "index.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
    G = entries[0]["generation"]
    n_ok = sum(v["exact"] and v["file_identical"] for v in proof.values())
    seeds = (f"seeds {runj.get('seed_scheme')} (run_seed {runj.get('run_seed')}), " if runj else "legacy 32-bit seeds, ")
    note = (f"ER {R['run']}: gen {G} best per aircraft on its train course (k=0) and hold-out course (j=0); "
            f"label = stage · passes/15. Replay proof: {n_ok}/{len(proof)} costs exact and files byte-identical under "
            f"ER's run code ({R['code']}). {seeds}gates = {R['scoring']}.")
    import colab_viewer
    html = colab_viewer.build_standalone_html(os.path.join(out, "index.json"), gens="all", hz=10, params={
        "mode": "single", "gen": f"c172x:{G}@train", "cam": "chase", "layout": "true", "note": note},
        title=f"Phase 4 smoke replay: ER {R['run']} (train + hold-out, gen {G})")
    open(page, "w").write(html)
    print(f"page {page}: {len(html.encode()) / 1e6:.2f} MB ({len(html.encode())} bytes)")
    assert len(html.encode()) < 25e6
    if "--no-shots" in sys.argv:
        return
    i = {k.split(".")[0]: v for k, v in info.items()}
    def mid(k):   # a moment with rings in view: halfway to the first pass, else 30 s
        return round(i[k]["first_pass_t"] + 4.0, 2) if i[k]["first_pass_t"] else 30.0
    spec = []
    for ac in ("c172x", "T38", "737", "f16"):
        spec.append([f"{ac}_train_chase", f"?mode=single&gen={ac}:{G}@train&cam=chase", mid(f"{ac}_train")])
        spec.append([f"{ac}_holdout_chase", f"?mode=single&gen={ac}:{G}@hold-out&cam=chase", mid(f"{ac}_holdout")])
    spec.append(["c172x_train_overview_all_rings", f"?mode=single&gen=c172x:{G}@train&cam=free&rings=all",
                 round(i["c172x_train"]["t_end"] - 1, 2)])
    spec.append(["737_holdout_overview_all_rings", f"?mode=single&gen=737:{G}@hold-out&cam=free&rings=all",
                 round(i["737_holdout"]["t_end"] - 1, 2)])
    sp = f"/tmp/{R['prefix']}shots.json"
    json.dump(spec, open(sp, "w"))
    py = os.path.join(HERE, ".venv-shots", "bin", "python")
    p = subprocess.run([py, os.path.join(HERE, "tools", "screenshots.py"), "--bench", page, "--shot-spec", sp,
                        "--prefix", R["prefix"], "--strict"], capture_output=True, text=True)
    open(f"/tmp/{R['prefix']}shots.log", "w").write(p.stdout + p.stderr)
    print(p.stdout[-2500:] if p.returncode == 0 else (p.stdout + p.stderr)[-3000:])
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
