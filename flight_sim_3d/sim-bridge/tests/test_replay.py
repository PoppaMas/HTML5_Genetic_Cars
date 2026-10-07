"""Replay integration tests (fly JSBSim). Run:
    PYTHONDONTWRITEBYTECODE=1 <venv>/bin/python -m pytest -q tests/        (or: <venv>/bin/python tests/test_replay.py)
Needs a Python with jsbsim + numpy (ER's venv). Paths come from sim_bridge.paths (FLIGHT_SIM_TEAM_ROOT,
SIMBRIDGE_RUNS_ROOT, ...). Never writes into ER's tree (outputs go to a temp dir).

1. REAL interface (primary): ER's run.json / genomes.jsonl + ER's evolution.eval, auto-selected. phase1-s1 gens 0/9/19
   and best-per-gen, a new all-individuals run (phase1v5-s1) with elites / mixed aircraft / string ids, ER's newest
   format (team-relative roots, header scenario_id), and real full-fidelity (FD v2) flights with FD's nodal data
   mapped by sim_bridge.v2_map (cross-checked against FD's own record=True node export; --no-nodes = estimate).
   Costs exact, channels bit-identical to ER's trajectories.
2. Adapter fallback (runs without run.json, e.g. bench_jets-j1) and forced --interface adapter.
3. Protocol edge cases through tests/fixtures/fake_evolution_eval.py on a copy of ER's real phase1-s1 files:
   post-step recorder without the t=0 call, synthetic flex_state plumbing.
Skips (not fails) when a run / ER's evolution.eval / its trajectories are not present.
"""
import glob
import json
import re
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.dirname(HERE)
sys.path.insert(0, SB)
sys.dont_write_bytecode = True
from sim_bridge import paths  # noqa: E402

RUNS = paths.RUNS_ROOT
PY = sys.executable


def _skip(msg):
    try:
        import pytest
    except ImportError:
        raise RuntimeError("SKIP " + msg)
    pytest.skip(msg)


def _need_traj(run, traj=True, run_json=False):
    """Skip unless the run (and its reference trajectories / ER's run.json) is present. The repo ships run records
    (config.json, summary.json, checkpoints/) but not ER's bulky trajectory JSON dumps (~650 KB each); copy the
    original evolution/runs/<run>/trajectories/ into place to run the channel checks."""
    d = os.path.join(RUNS, run)
    if not os.path.isdir(d):
        _skip(f"run {run} not present under {RUNS}")
    if traj and not glob.glob(os.path.join(d, "trajectories", "traj_*.json")):
        _skip(f"reference trajectories {d}/trajectories not present (ER's trajectory dumps are not shipped in the repo)")
    if run_json and not os.path.exists(os.path.join(d, "run.json")):
        _skip(f"{run} has no run.json (ER's real format) here")
    return d


def _need_real_eval():
    if not os.path.exists(os.path.join(paths.EVOLUTION_ROOT, "eval.py")):
        _skip("ER's evolution/eval.py not present")


def _replay(args, env_extra=None):
    out = tempfile.mkdtemp(prefix="replay_test_")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env_extra or {}))
    p = subprocess.run([PY, os.path.join(SB, "replay.py"), *args, "--out", out], cwd=SB, env=env,
                       capture_output=True, text=True, timeout=1800)
    man_p = os.path.join(out, "replay_manifest.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else None
    if man is not None:
        man["_out"] = out
    return p, man


def _index(man):
    return json.load(open(os.path.join(man["_out"], "trajectories", "index.json")))


def _assert_exact(p, man, n_rows=2701, min_files=1):
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    assert man and not man["flagged"]
    assert all(g["verdict"] == "match" and g["max_rel_err"] == 0.0 for g in man["genomes"]), man["summary"]
    tc = man["trajectory_check"]
    assert tc and tc["n_files"] >= min_files and set(tc["summary"]) == {"match"}, tc and tc["summary"]
    for f in tc["files"].values():
        assert f["bit_identical_channels"] == f["n_channels"], f
        assert f["rows_compared"] == n_rows == f["rows_reference"], f


# ------------------------------------------------------------------ 1. real interface (ER's evolution.eval)
def test_real_phase1_s1_gens_0_9_19():
    _need_real_eval()
    _need_traj("phase1-s1", run_json=True)
    p, man = _replay(["--run", "phase1-s1", "--gens", "0,9,19"])          # no --interface: auto must pick er
    _assert_exact(p, man, min_files=12)
    assert man["interface"] == "er" and man["evaluate"] == "evolution.eval" and man["recorder_timing"] == "post"
    assert man["n_genomes"] == 12 and man["n_flights"] == 36
    ids = {g["individual_id"] for g in man["genomes"]}
    assert "c172x:g19:best" in ids and all(isinstance(i, str) for i in ids)
    for g in man["genomes"]:
        assert all(isinstance(k, str) and k.startswith(g["aircraft"] + ":") for k in g["per_scenario"]), g["per_scenario"]
        assert [v["scenario_index"] for v in g["per_scenario"].values()] == [0, 1, 2]


def test_real_phase1_s1_best_per_gen():
    _need_real_eval()
    _need_traj("phase1-s1", run_json=True)
    p, man = _replay(["--run", "phase1-s1", "--best-per-gen"])
    _assert_exact(p, man, min_files=12)
    assert man["n_genomes"] == 80 and man["summary"] == {"match": 80}


def _new_run():
    """Newest run whose genomes.jsonl has ER's per-individual rows (schema ga-flightsim-genomes/1)."""
    for run in ("phase1v5-s1", "phase1v5-s2", "phase1v5-s3"):
        g = os.path.join(RUNS, run, "genomes.jsonl")
        if os.path.exists(g) and os.path.exists(os.path.join(RUNS, run, "run.json")):
            with open(g) as f:
                if json.loads(f.readline()).get("schema", "").startswith("ga-flightsim-genomes/"):
                    return run
    _skip("no run with per-individual genomes.jsonl rows")


def test_real_new_run_elites_exact():
    _need_real_eval()
    run = _new_run()
    _need_traj(run)
    p, man = _replay(["--run", run, "--gens", "0,19", "--elites"])
    _assert_exact(p, man, min_files=1)
    acs = {g["aircraft"] for g in man["genomes"]}
    assert len(acs) >= 2, acs                                             # mixed aircraft in one replay
    for ac in acs:
        for gen in (0, 19):
            rows = [g for g in man["genomes"] if g["aircraft"] == ac and g["generation"] == gen]
            assert sum(g["is_best"] for g in rows) == 1 and all(g["is_elite"] for g in rows) and len(rows) >= 2, rows
    idx = _index(man)
    best = [e for e in idx["entries"] if e["is_best"]]
    elite = [e for e in idx["entries"] if not e["is_best"]]
    assert best and elite and all(isinstance(e["scenario"], str) and isinstance(e["scenario_index"], int) for e in idx["entries"])
    # only best-of-generation files are compared with ER's trajectories (ER writes those)
    assert all(f["individual_id"] in {e["individual_id"] for e in best} for f in man["trajectory_check"]["files"].values())


def test_real_mixed_aircraft_string_ids_and_scenario_position():
    _need_real_eval()
    run = _new_run()
    rows = [json.loads(l) for l in open(os.path.join(RUNS, run, "genomes.jsonl"))]
    acs = sorted({r["aircraft"] for r in rows})
    pick = [next(r for r in rows if r["aircraft"] == acs[0] and r["generation"] == 19 and r["rank"] == 1)["individual_id"],
            next(r for r in rows if r["aircraft"] == acs[1] and r["generation"] == 3 and r["is_best"])["individual_id"]]
    p, man = _replay(["--run", run, "--ids", ",".join(pick), "--scenario", "2", "--compare-traj", "none"])
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert {g["individual_id"] for g in man["genomes"]} == set(pick)
    for g in man["genomes"]:
        assert g["verdict"] == "match" and list(g["per_scenario"]) == [f"{g['aircraft']}:s2"], g
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(man["_out"], "trajectories", "traj_*.json")))
    assert len(files) == 2 and all(":" not in f for f in files) and all("__" in f for f in files), files


def _full_replay(extra=()):
    _need_real_eval()
    run = _new_run()
    p, man = _replay(["--run", run, "--gens", "19", "--aircraft", "T38", "--scenario", "0", "--fidelity", "full",
                      "--compare-traj", "none", *extra])
    if p.returncode == 3 and "FidelityUnavailable" in p.stderr:
        _skip("FD's v2 model root is not prepared here: " + p.stderr.strip()[-300:])
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    doc = json.load(open(glob.glob(os.path.join(man["_out"], "trajectories", "traj_T38_*.json"))[0]))
    return man, doc


def test_real_full_fidelity_v2_nodes():
    """Real FD v2 flight through ER's evaluate with FD's exact nodal values: all five components nodal (no estimate),
    last node == FD tip scalar on every frame, wing in-plane dx present, model_version = FD's published post-mass-fix
    string, recorder flight bit-identical to FD's cost flight, ER's modal wings agree with the nodal wings."""
    man, doc = _full_replay()
    from sim_bridge import v2_map
    g = man["genomes"][0]
    assert g["replay_model_version"].startswith("full:flexv2:")
    if g["fd_current_model_version"] is not None:
        assert g["model_version_current"] is True, (g["replay_model_version"], g["fd_current_model_version"])
    st = doc["structure"]
    names = [c["name"] for c in st["components"]]
    assert set(v2_map.COMPONENTS) <= set(names)  # FE components present (ER /3 also keeps wing*_modal)
    fe = [c for c in st["components"] if c["name"] in v2_map.COMPONENTS]
    assert not any(c.get("estimated") for c in fe) and st["v2_map"].get("estimated_components", []) == []
    assert doc["replay"]["node_status"] == {c: "fd_nodes" for c in v2_map.COMPONENTS}
    assert st["v2_map"]["tip_check_max_abs"] < 1e-9 and st["v2_map"].get("estimated_frames", 0) == 0
    assert st["v2_map"].get("flex_api") == "flexstate3" or "FlexState" in (doc["replay"].get("node_note") or "")
    assert v2_map.validate_structure(st, doc["channels"]) == []
    comps = {c["name"]: c for c in st["components"]}
    assert len(comps["wingR"]["axis_nodes_body_m"]) == 33 and comps["wingR"]["dof"] == ["dz", "dx", "twist"]
    assert len(comps["htail"]["axis_nodes_body_m"]) == 26 and len(comps["vtail"]["axis_nodes_body_m"]) == 13
    if "wingR_modal" in comps:
        assert len(comps["wingR_modal"]["axis_nodes_body_m"]) == 9 and "wingR_modal.dz.8" in doc["channels"]
    assert g["structure"]["telemetry_check"][0]["sim_cost_bit_identical"] is True
    wm = g["structure"]["wing_modal_vs_nodal"]["T38:s0"]["wingR"]
    assert wm["dz_max_abs_diff_m"] <= 0.01 * wm["dz_max_abs_m"] and wm["twist_max_abs_diff_rad"] <= 0.01 * wm["twist_max_abs_rad"]
    ch = doc["channels"]
    nR = len(comps["wingR"]["axis_nodes_body_m"]) - 1
    for row in doc["data"][::300]:
        r = dict(zip(ch, row))
        for side in ("R", "L"):
            assert abs(r[f"struct.wing{side}_tip_dz"] - r[f"wing{side}.dz.{nR}"]) <= 2e-6
            assert abs(r[f"struct.wing{side}_tip_dx"] - r[f"wing{side}.dx.{nR}"]) <= 2e-6
            assert abs(r[f"struct.wing{side}_tip_twist"] - r[f"wing{side}.twist.{nR}"]) <= 2e-6
        assert abs(r["struct.htR_tip_twist"] - r["htail.twist.25"]) <= 2e-6 and abs(r["struct.htL_tip_twist"] - r["htail.twist.0"]) <= 2e-6
        assert abs(r["struct.vt_tip_twist"] - r["vtail.twist.12"]) <= 2e-6
        assert abs(r["struct.wingR_root_bm"] - r["flex.wingR_bm"] * v2_map.LBFFT) <= 1e-3


def test_real_full_fidelity_matches_fd_own_node_export():
    """The replay's mapped channels equal FD's OWN record=True node export (flexeval.evaluate(record=True),
    fd-flexbody-nodes/1) mapped by v2_map, frame by frame, within the written rounding; same layout, same cost."""
    man, _doc = _full_replay()
    f = glob.glob(os.path.join(man["_out"], "trajectories", "traj_T38_*.json"))[0]
    p = subprocess.run([PY, os.path.join(SB, "tools", "verify_fd_nodes.py"), f], cwd=SB, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    rep = json.loads(p.stdout[p.stdout.index("{"):])
    assert rep["ok"] and rep["cost_bit_identical"] and rep["layout_max_dist_m"] == 0.0 and rep["rows_compared"] >= 2700
    assert rep["fd_model_version"] == rep["replay_model_version"]


def test_real_full_fidelity_no_nodes_is_flagged_estimated():
    man, doc = _full_replay(["--no-nodes"])
    st = doc["structure"]
    est = {c["name"]: bool(c.get("estimated")) for c in st["components"] if c["name"] in ("wingR", "wingL", "htail", "vtail", "fuselage")}
    assert est.get("wingR") is False and est.get("wingL") is False            # modal wings (real)
    assert est["htail"] and est["vtail"] and est["fuselage"]                  # tip-only estimate, labelled
    assert doc["replay"]["node_status"]["htail"] == "estimated"
    assert "wingR.dz.8" in doc["channels"] and "wingR.dz.32" not in doc["channels"]


def test_real_newest_run_relative_roots_and_header_scenario_id():
    """ER's newest FINISHED run (phase1v5-ki05-s1 rigid, phase2-smoke-s1 full-fidelity, ...): team-relative
    aircraft_root / git.repo resolved, scenario_id read from the trajectory headers, 'min', costs exact at the run's
    fidelity, channels bit-identical. For a full-fidelity run the wings are FD nodal (33) while ER wrote its 9-node
    modal wings: those are compared at coincident span fractions (rediscretised_components), not by name."""
    _need_real_eval()
    cands = []
    for rj in glob.glob(os.path.join(RUNS, "*", "run.json")):
        d = json.load(open(rj))
        roots = [a.get("aircraft_root") for a in d.get("aircraft", []) if a.get("aircraft_root")]
        rd = os.path.dirname(rj)
        finished = os.path.exists(os.path.join(rd, "summary.json")) and glob.glob(os.path.join(rd, "trajectories", "traj_*.json"))
        if finished and d.get("paths_relative_to") and roots and not any(os.path.isabs(r) for r in roots):
            cands.append((os.path.getmtime(rj), os.path.basename(rd), d))
    if not cands:
        _skip("no finished run (summary.json + trajectories) with team-relative aircraft_root yet")
    _, run, cfg = max(cands, key=lambda c: c[0])
    d = _need_traj(run)
    gens = sorted({int(re.search(r"_g(\d+)\.json$", f).group(1)) for f in glob.glob(os.path.join(d, "trajectories", "traj_*_g*.json"))})
    g = gens[-1]
    hdr = json.load(open(sorted(glob.glob(os.path.join(d, "trajectories", f"traj_*_g{g}.json")))[0]))
    assert hdr["fitness_sense"] == "min" and hdr["scenario_id"] == f"{hdr['aircraft']}:s{hdr['scenario_index']}"
    p, man = _replay(["--run", run, "--gens", str(g)])
    if p.returncode == 3 and "pin mismatch" in (p.stdout + p.stderr):
        # FD moved on since the run (e.g. P3-B1 r0 -> r1): replay against a frozen FD copy matching the pins
        # (tools/build_b1_page.py discovery: run.json key, evolution/_fd_pin_*, $SIMBRIDGE_FD_DIRS), else skip
        sys.path.insert(0, os.path.join(SB, "tools"))
        import build_b1_page as B
        R = {"id": run, "dir": d, "run": cfg}
        fd, src, tried = B.pick_fd_dir(R, None, False)
        if not fd:
            _skip(f"{run}: live FD no longer matches the pins and no frozen FD copy does ({[t['dir'] for t in tried]})")
        p, man = _replay(["--run", run, "--gens", str(g)], {"EVOLUTION_FD_DIR": fd, "FLIGHT_DYNAMICS_DIR": fd})
    if p.returncode == 3 and "FidelityUnavailable" in p.stderr:
        _skip("FD's v2 model root is not prepared here")
    _assert_exact(p, man, min_files=1)
    assert any("abs_root" in n for n in man.get("path_notes", [])) or "abs_root" in p.stdout, man.get("path_notes")
    assert all(k.split(" ")[2].count(":s") == 1 for k in man["trajectory_check"]["files"])   # keyed on scenario_id
    if cfg.get("fidelity") == "full":
        for gm in man["genomes"]:
            assert gm["replay_model_version"] == gm["logged_model_version"]
            if gm.get("pinned_model_version"):
                assert gm["pinned_model_version_match"] is True, gm
        for f in man["trajectory_check"]["files"].values():
            rd = f["rediscretised_components"]
            assert set(rd) == {"wingR", "wingL"}, rd
            for w in rd.values():
                assert (w["a_nodes"], w["b_nodes"], w["n_pairs"]) == (33, 9, 9) and w["a_only_dofs"] == ["dx"]
                assert w["max_abs_at_coincident_nodes"]["dz"] < 2e-3 and w["max_abs_at_coincident_nodes"]["twist"] < 1e-4, w


def test_real_multi_fidelity_ladder_model_version():
    """ER multi-fidelity rows (screen rigid -> full, phase2-pilot-*): replay the same individual at rigid and at full;
    each is compared with ladder_cost[fid] and its logged model_version comes from ladder_model_version[fid]."""
    _need_real_eval()
    from sim_bridge import paths
    cands = []
    for gj in sorted(glob.glob(os.path.join(RUNS, "*", "genomes.jsonl")), key=os.path.getmtime, reverse=True):
        if not os.path.exists(os.path.join(os.path.dirname(gj), "run.json")):
            continue
        with open(gj) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:          # a run still being written
                    continue
                lmv = r.get("ladder_model_version") or {}
                if isinstance(lmv, dict) and {"rigid", "full"} <= set(lmv) and (r.get("ladder_status") or {}).get("full") == "ok":
                    cands.append((os.path.basename(os.path.dirname(gj)), r))
                    break
    if not cands:
        _skip("no run with ladder_model_version {rigid, full} yet")
    # prefer a run flown with FD's live full model; else (FD moved on, e.g. P2.5 after the phase-2 pilot) fly the
    # pinned run against ER's frozen copy of the FD sources it was pinned to (evolution/_fd_pin_post_mass), read-only
    cur = paths.fd_model_versions()
    live = [c for c in cands if (cur.get(c[1]["aircraft"]) or {}).get("full") == c[1]["ladder_model_version"]["full"]]
    env = None
    if live:
        run, row = live[0]
    else:
        run, row = cands[0]
        frozen = os.path.join(paths.EVOLUTION_ROOT, "_fd_pin_post_mass")
        fz = os.path.join(frozen, "v2_results", "model_versions_post_mass.json")
        if not (os.path.exists(fz) and (json.load(open(fz)).get(row["aircraft"]) or {}).get("full")
                == row["ladder_model_version"]["full"]):
            _skip("no ladder run matches FD's live full model_version and no matching frozen FD copy")
        env = {"EVOLUTION_FD_DIR": frozen, "FLIGHT_DYNAMICS_DIR": frozen}
    for fid in ("rigid", "full"):
        p, man = _replay(["--run", run, "--ids", row["individual_id"], "--scenario", "0", "--fidelity", fid,
                          "--compare-traj", "none"], env_extra=env)
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        g = man["genomes"][0]
        assert g["verdict"] == "match", g
        assert g["logged_model_version_source"] == f"ladder_model_version.{fid}", g["logged_model_version_source"]
        assert g["logged_model_version"] == row["ladder_model_version"][fid] == g["replay_model_version"]
        if fid == "full":
            assert set(g["structure"]["node_status"][f"{row['aircraft']}:s0"].values()) == {"fd_nodes"}


# ------------------------------------------------------------------ 2. adapter fallback
def test_adapter_fallback_bench_no_run_json():
    d = _need_traj("bench_jets-j1")
    assert not os.path.exists(os.path.join(d, "run.json"))
    p, man = _replay(["--run", "bench_jets-j1", "--gens", "19", "--aircraft", "f16,T38"])
    _assert_exact(p, man, min_files=2)
    assert man["interface"] == "adapter" and "no run.json" in man["fallback_reason"]
    for g in man["genomes"]:  # the adapter mirrors ER's per-aircraft string scenario ids
        assert list(g["per_scenario"]) == [f"{g['aircraft']}:s{i}" for i in range(len(g["per_scenario"]))]


def test_adapter_forced_on_run_with_run_json():
    _need_traj("phase1-s1")
    p, man = _replay(["--run", "phase1-s1", "--interface", "adapter", "--gens", "0,19", "--aircraft", "T38,c172x"])
    _assert_exact(p, man, min_files=4)
    assert man["interface"] == "adapter"


def test_adapter_rejects_flex_fidelity():
    _need_traj("bench_jets-j1", traj=False)
    p, man = _replay(["--run", "bench_jets-j1", "--gens", "19", "--aircraft", "f16", "--scenario", "0", "--fidelity", "full"])
    assert p.returncode == 3 and "fidelity 'full'" in p.stdout + p.stderr


# ------------------------------------------------------------------ 3. protocol edge cases (fixture eval)
FIX_ROOT = os.path.join(HERE, "fixtures", "er_interface")
FAKE = os.path.join(HERE, "fixtures", "fake_evolution_eval.py")


def _ref():
    return os.path.join(_need_traj("phase1-s1"), "trajectories")


def test_fixture_post_step_without_t0_call():
    p, man = _replay(["--run", "phase1-s1", "--runs-root", FIX_ROOT, "--eval-module", FAKE,
                      "--gens", "9,19", "--aircraft", "737,f16", "--compare-traj", _ref()])
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert man["interface"] == "er" and man["recorder_timing"] == "post"
    assert any("relative 'flight-dynamics/jsbsim_root'" in n for n in man["path_notes"]), man["path_notes"]
    assert all(g["verdict"] == "match" for g in man["genomes"])
    for f in man["trajectory_check"]["files"].values():  # no t=0 call -> t=0 row absent, everything else identical
        assert f["verdict"] == "match" and f["rows_replay"] == 2700 and f["rows_compared"] == 2700
        assert f["bit_identical_channels"] == f["n_channels"]


def test_fixture_with_t0_call():
    p, man = _replay(["--run", "phase1-s1", "--runs-root", FIX_ROOT, "--eval-module", FAKE,
                      "--gens", "19", "--aircraft", "c172x", "--compare-traj", _ref()], {"FAKE_EVAL_T0": "1"})
    _assert_exact(p, man, n_rows=2701)


def test_fixture_synthetic_flex_plumbing():
    p, man = _replay(["--run", "phase1-s1", "--runs-root", FIX_ROOT, "--eval-module", FAKE,
                      "--ids", "T38:g19:best", "--scenario", "T38:s0", "--compare-traj", "none"],
                     {"FAKE_EVAL_T0": "1", "FAKE_EVAL_SYNTH_FLEX": "1"})
    assert p.returncode == 0, p.stdout + p.stderr
    docs = [json.load(open(f)) for f in glob.glob(os.path.join(man["_out"], "trajectories", "traj_T38_*.json"))]
    assert len(docs) == 1 and docs[0].get("structure"), "no trajectory with a structure block"
    d = docs[0]
    assert d["structure"]["synthetic"] is True and "wingL.dz.2" in d["channels"] and len(d["channels"]) == 29 + 12
    assert d["individual_id"] == "T38:g19:best" and d["scenario_id"] == "T38:s0" and d["scenario_index"] == 0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            print("PASS", name)
        except BaseException as e:  # noqa: BLE001
            if type(e).__name__ == "Skipped" or str(e).startswith("SKIP"):
                print("SKIP", name, e)
                continue
            fails += 1
            print("FAIL", name, type(e).__name__, str(e)[:2000])
    sys.exit(1 if fails else 0)
