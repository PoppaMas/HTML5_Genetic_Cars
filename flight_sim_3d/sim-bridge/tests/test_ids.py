"""Fast unit tests (no JSBSim): opaque string ids, per-row scenario lists, elite/best selection, logged-cost lookup,
model_version per fidelity (ladder_model_version, FD's published strings), file names, trajdiff keys,
fitness_sense normalisation and path env overrides.
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_ids.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.dirname(HERE)
sys.path.insert(0, SB)
sys.dont_write_bytecode = True

from sim_bridge import replay as R  # noqa: E402
from sim_bridge import trajdiff  # noqa: E402

RUN_CFG = {  # shaped like ER's real run.json (two aircraft with different scenario counts)
    "run_id": "mix-s1", "fitness_sense": "min", "fidelity": "rigid", "eval_seed": 7,
    "model_version": {"c172x": "rigid:jsbsim1.3.1:aaaa", "T38": "rigid:jsbsim1.3.1:bbbb"},
    "aircraft": [{"name": "c172x", "scenario_ids": ["c172x:s0", "c172x:s1", "c172x:s2"]},
                 {"name": "T38", "scenario_ids": ["T38:s0", "T38:s1", "T38:s2", "T38:s3"]}],
    "scenarios": [{"id": f"c172x:s{i}", "aircraft": "c172x", "index": i} for i in range(3)]
               + [{"id": f"T38:s{i}", "aircraft": "T38", "index": i} for i in range(4)],
}


def _row(ac, gen, rank, cost, elite=2, **kw):
    n = 3 if ac == "c172x" else 4
    r = {"individual_id": f"{ac}:g{gen}:r{rank}", "aircraft": ac, "generation": gen, "rank": rank, "is_best": rank == 0,
         "is_elite": rank < elite, "cost": cost, "per_scenario_cost": [cost + 0.01 * k for k in range(n)],
         "scenario_ids": [f"{ac}:s{k}" for k in range(n)], "genome": {"kp": 1.0}, "fidelity": "rigid"}
    r.update(kw)
    return r


ROWS = [_row(ac, g, k, 1.0 / (g + 1) + 0.1 * k) for ac in ("c172x", "T38") for g in (0, 1, 2) for k in range(4)]


def test_select_best_elites_ids_mixed_aircraft():
    best = R.select_rows([dict(r) for r in ROWS], gens={0, 2})
    assert [(r["aircraft"], r["generation"], r["rank"]) for r in best] == [("T38", 0, 0), ("T38", 2, 0), ("c172x", 0, 0), ("c172x", 2, 0)]
    el = R.select_rows([dict(r) for r in ROWS], gens={1}, elites=True)
    assert sorted(r["individual_id"] for r in el) == ["T38:g1:r0", "T38:g1:r1", "c172x:g1:r0", "c172x:g1:r1"]
    assert sum(r["_is_best"] for r in el) == 2
    bpg = R.select_rows([dict(r) for r in ROWS], best_per_gen=True, aircraft=["T38"])
    assert [r["generation"] for r in bpg] == [0, 1, 2] and all(r["rank"] == 0 for r in bpg)
    ids = R.select_rows([dict(r) for r in ROWS], ids=["T38:g2:r3", "c172x:g0:r1"])
    assert [r["individual_id"] for r in ids] == ["T38:g2:r3", "c172x:g0:r1"]
    try:
        R.select_rows(ROWS, ids=["T38:g9:r0"])
        raise AssertionError("unknown id accepted")
    except SystemExit as e:
        assert "T38:g9:r0" in str(e)


def test_ids_are_opaque_never_parsed():
    weird = [_row("T38", 4, 0, 0.5, individual_id="007"), _row("T38", 4, 1, 0.6, individual_id="T38/g4 best?"),
             _row("T38", 4, 2, 0.7, individual_id="x:y:z:r9")]
    sel = R.select_rows(weird, ids=["007", "x:y:z:r9"])
    assert [r["individual_id"] for r in sel] == ["007", "x:y:z:r9"]       # strings stay strings (no int("007"))
    names = {R.traj_file_name(r, "run:1/a", "T38:s1", tag_individual=True, tag_scenario=True) for r in weird}
    assert len(names) == 3 and all("/" not in n and ":" not in n and " " not in n for n in names), names
    assert R.traj_file_name(weird[0], "r", "T38:s0", tag_individual=False, tag_scenario=False) == "traj_T38_r_g4.json"


def test_best_derived_when_flags_absent_respects_sense():
    rows = [{k: v for k, v in r.items() if k not in ("is_best", "is_elite")} for r in ROWS if r["generation"] == 0]
    assert {r["rank"] for r in R.select_rows([dict(r) for r in rows], gens={0}, sense="min")} == {0}
    assert {r["rank"] for r in R.select_rows([dict(r) for r in rows], gens={0}, sense="max")} == {3}


def test_scenarios_per_row_by_id_or_position():
    c, t = ROWS[0], ROWS[-1]
    assert R.resolve_scenarios(c, RUN_CFG, "all") == [(0, "c172x:s0"), (1, "c172x:s1"), (2, "c172x:s2")]
    assert len(R.resolve_scenarios(t, RUN_CFG, "all")) == 4                  # mixed aircraft: own list per row
    assert R.resolve_scenarios(t, RUN_CFG, "T38:s3") == [(3, "T38:s3")]
    assert R.resolve_scenarios(t, RUN_CFG, "2") == [(2, "T38:s2")]           # digits = position
    assert R.resolve_scenarios(c, RUN_CFG, "T38:s3") == []                   # another aircraft's id: skipped
    for bad in ("c172x:s9", "7"):
        try:
            R.resolve_scenarios(c, RUN_CFG, bad)
            raise AssertionError(bad)
        except SystemExit:
            pass
    row = dict(c)
    del row["scenario_ids"]                                                  # falls back to the aircraft's list
    assert R.row_scenario_ids(row, RUN_CFG) == ["c172x:s0", "c172x:s1", "c172x:s2"]


def test_logged_cost_fields_as_er_writes_them():
    r = _row("T38", 5, 0, 0.2, screen_cost=0.25, screen_fidelity="reduced", screen_model_version="reduced:x",
             screen_per_scenario_cost=[0.2, 0.3, 0.2, 0.3], ladder_cost={"rigid": 0.21, "reduced": 0.25, "full": 0.2},
             fidelity="full", model_version="full:flexv2:cc")
    own = R.logged_at(r, RUN_CFG, "full")
    assert own == {"cost": 0.2, "per_scenario": r["per_scenario_cost"], "model_version": "full:flexv2:cc",
                   "mv_source": "model_version", "source": "cost"}
    scr = R.logged_at(r, RUN_CFG, "reduced")
    assert scr["source"] == "screen_cost" and scr["cost"] == 0.25 and scr["model_version"] == "reduced:x"
    assert R.logged_at(r, RUN_CFG, "rigid")["source"] == "ladder_cost.rigid"
    plain = _row("c172x", 1, 0, 0.3)
    del plain["fidelity"]
    assert R.row_fidelity(plain, RUN_CFG) == "rigid" and R.row_model_version(plain, RUN_CFG) == "rigid:jsbsim1.3.1:aaaa"
    assert R.row_eval_seed(plain, RUN_CFG) == 7 and R.row_eval_seed(dict(plain, eval_seed=3), RUN_CFG) == 3


def test_model_version_per_fidelity_ladder_first():
    """ER: rows not rescored at full keep cost / fidelity / model_version of the highest stage reached; per-stage
    versions are in ladder_model_version (row first, then the aircraft entry)."""
    r = _row("T38", 5, 0, 0.25, fidelity="reduced", model_version="reduced:flexv1:r1", rescored_at_full=False,
             ladder_cost={"rigid": 0.21, "reduced": 0.25}, ladder_model_version={"rigid": "rigid:jsbsim1.3.1:bbbb",
                                                                                "reduced": "reduced:flexv1:r1"})
    assert R.logged_model_version(r, RUN_CFG, "reduced") == ("reduced:flexv1:r1", "ladder_model_version.reduced")
    lg = R.logged_at(r, RUN_CFG, "rigid")
    assert lg["cost"] == 0.21 and lg["model_version"] == "rigid:jsbsim1.3.1:bbbb" and lg["mv_source"] == "ladder_model_version.rigid"
    assert R.logged_at(r, RUN_CFG, "full")["model_version"] is None                    # never reached full
    cfg = dict(RUN_CFG, aircraft=[dict(RUN_CFG["aircraft"][1], ladder_model_version={"full": "full:flexv2:e9535bc7"})])
    assert R.logged_model_version(r, cfg, "full") == ("full:flexv2:e9535bc7", "aircraft.ladder_model_version.full")
    lst = dict(r, ladder_model_version=[["rigid", "rigid:a"], ["reduced", "reduced:b"]])    # list form tolerated
    assert R.logged_model_version(lst, RUN_CFG, "rigid")[0] == "rigid:a"
    plain = _row("c172x", 1, 0, 0.3)
    del plain["fidelity"]
    plain.pop("model_version", None)
    assert R.logged_model_version(plain, RUN_CFG, "rigid") == ("rigid:jsbsim1.3.1:aaaa", "run.json model_version")


def test_fd_published_model_versions(tmp_path=None):
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mv.json")
    json.dump({"T38": {"full": "full:flexv2:e9535bc7"}}, open(p, "w"))
    code = ("import json,sys; sys.path.insert(0, %r); from sim_bridge import paths, replay as R; "
            "print(json.dumps([paths.FD_MODEL_VERSIONS, R.fd_current_model_version('T38', 'full'), "
            "R.fd_current_model_version('T38', 'reduced')]))" % SB)
    out = subprocess.run([sys.executable, "-B", "-c", code], env=dict(os.environ, SIMBRIDGE_FD_MODEL_VERSIONS=p),
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out.strip().splitlines()[-1]) == [p, "full:flexv2:e9535bc7", None]
    from sim_bridge import paths
    if os.path.exists(paths.FD_MODEL_VERSIONS):            # FD's real file: post-mass-fix strings for all four
        mv = paths.fd_model_versions()
        assert set(mv) >= {"c172x", "T38", "737"} and all(mv[a]["full"].startswith("full:flexv2:") for a in mv)


def test_compare_by_position_with_string_ids():
    r = _row("T38", 1, 0, 0.5)
    lg = R.logged_at(r, RUN_CFG, "rigid")
    per = {"T38:s1": (1, 0.51), "T38:s3": (3, 0.53)}
    c = R.compare(lg, None, per, "rigid", "rigid", "rigid:jsbsim1.3.1:bbbb", 1e-6)
    assert c["verdict"] == "match" and c["per_scenario"]["T38:s3"]["logged"] == 0.53 and c["model_version_match"] is True
    c = R.compare(dict(lg, model_version="rigid:jsbsim1.3.1:old"), 0.5 + 0.015, per, "rigid", "rigid", "new", 1e-6)
    assert c["verdict"] == "mismatch expected (model_version differs)" and c["model_version_match"] is False
    c = R.compare(dict(lg, model_version="same"), 0.6, per, "rigid", "rigid", "same", 1e-6)
    assert c["verdict"] == "MISMATCH"
    c = R.compare(R.logged_at(r, RUN_CFG, "full"), None, per, "full", "rigid", "full:x", 1e-6)
    assert c["verdict"].startswith("not comparable")


def test_fitness_sense_normalised():
    assert R.sense_of({"fitness_sense": "min"}) == "min"
    assert R.sense_of({"fitness_sense": "minimize (GA cost, mean over scenarios)"}) == "min"
    assert R.sense_of({"fitness_sense": " Maximize fitness"}) == "max"
    assert R.sense_of({}) == "min" and R.sense_of("max") == "max"


def test_trajdiff_keys_opaque():
    er_old = {"aircraft": "T38", "generation": 19, "scenario_index": 0}                 # ER before headers: no id
    er_new = {"aircraft": "T38", "generation": 19, "scenario_index": 0, "scenario_id": "T38:s0"}   # ER header now
    best = {"aircraft": "T38", "generation": 19, "scenario_index": 0, "scenario_id": "T38:s0", "individual_id": "T38:g19:r0",
            "is_best": True, "replay": {"fidelity": "rigid", "model_version": "rigid:x"}}
    elite = {"aircraft": "T38", "generation": 19, "scenario_index": 0, "individual_id": "T38:g19:r1", "is_best": False}
    assert trajdiff.doc_key(er_old) == trajdiff.doc_key(er_new) == trajdiff.doc_key(best) == ("T38", 19, "T38:s0", "best", "")
    assert trajdiff.doc_key(elite) == ("T38", 19, "T38:s0", "T38:g19:r1", "")
    full = dict(best, replay={"fidelity": "full", "model_version": "full:flexv2:e9535bc7"})
    stale = dict(best, replay={"fidelity": "full", "model_version": "full:flexv2:0c55bea0"})
    assert trajdiff.doc_key(full) != trajdiff.doc_key(stale) != trajdiff.doc_key(er_new)   # flex keyed on model_version
    assert trajdiff.key_label(trajdiff.doc_key(full)) == "T38 g19 T38:s0 [full:full:flexv2:e9535bc7]"


def test_paths_env_overrides(tmp_path=None):
    code = "import json,sys; sys.path.insert(0, %r); from sim_bridge import paths; print(json.dumps(paths.describe()))" % SB
    base = {k: v for k, v in os.environ.items() if not k.startswith(("SIMBRIDGE_", "FLIGHT_SIM", "EVOLUTION_DIR", "FLIGHT_DYNAMICS"))}
    d = json.loads(subprocess.run([sys.executable, "-B", "-c", code], env=base, capture_output=True, text=True, check=True).stdout)
    assert d["team_root"] == os.path.dirname(SB) and d["runs_root"] == os.path.join(os.path.dirname(SB), "evolution", "runs")
    assert d["sandbox_dir"] == os.path.join(os.path.dirname(os.path.dirname(SB)), "flight_sim")
    env = dict(base, FLIGHT_SIM_TEAM_ROOT="/opt/team", SIMBRIDGE_RUNS_ROOT="/data/runs", FLIGHT_SIM_DIR="/opt/fs")
    d = json.loads(subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert d == dict(d, team_root="/opt/team", evolution_root="/opt/team/evolution", runs_root="/data/runs",
                     sandbox_dir="/opt/fs", flight_dynamics_dir="/opt/team/flight-dynamics")
    env = dict(base, EVOLUTION_DIR="/x/evo", SIMBRIDGE_SANDBOX="/y/fs")
    d = json.loads(subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert d["evolution_root"] == "/x/evo" and d["runs_root"] == "/x/evo/runs" and d["sandbox_dir"] == "/y/fs"


def test_model_root_resolution():
    from sim_bridge import paths
    p, note = paths.resolve_model_root("flight-dynamics/jsbsim_root")          # ER writes team-relative roots now
    assert p == os.path.join(paths.FLIGHT_DYNAMICS_DIR, "jsbsim_root") and "relative" in note
    p, note = paths.resolve_model_root("other/root")
    assert p == os.path.join(paths.TEAM_ROOT, "other", "root")
    p, note = paths.resolve_model_root("/elsewhere/team/flight-dynamics/jsbsim_root")
    if os.path.isdir(os.path.join(paths.FLIGHT_DYNAMICS_DIR, "jsbsim_root")):
        assert p == os.path.join(paths.FLIGHT_DYNAMICS_DIR, "jsbsim_root") and "remapped" in note
    assert paths.resolve_model_root(None) == (None, None)


if __name__ == "__main__":
    bad = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            try:
                f()
                print("PASS", k)
            except BaseException as e:  # noqa: BLE001
                bad += 1
                print("FAIL", k, type(e).__name__, e)
    sys.exit(1 if bad else 0)
