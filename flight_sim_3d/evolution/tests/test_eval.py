"""Replay interface (evolution.eval, run.json, genomes.jsonl), recorder protocol, fidelity adapter, viz flag,
multi-fidelity. Flex cases use shortened scenarios to stay fast."""
import copy
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from evolution import batch, eval as ev, fidelity as F, genome, runinfo, sim, trajectory, validate_traj

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TEAM = os.path.dirname(PKG)
FD_ROOT = os.path.join(TEAM, "flight-dynamics", "jsbsim_root")
FD_DIR = os.path.join(TEAM, "flight-dynamics")
need_fd = pytest.mark.skipif(not os.path.isdir(FD_ROOT), reason="flight-dynamics not present")
PY = sys.executable
SHORT = {"duration_s": 12.0, "steps_rel_ft": [[0.0, 0.0], [2.0, 100.0]]}
MV_RE = {"rigid": r"^rigid:jsbsim\d+\.\d+\.\d+:[0-9a-f]{8}$", "reduced": r"^reduced:flexv1:[0-9a-f]{8}$",
         "full": r"^full:flexv1:[0-9a-f]{8}$"}


def cfg_file(name):
    with open(os.path.join(PKG, "configs", name)) as f:
        return json.load(f)


def tiny_user(tmp_path, run_id, aircraft=("c172x",), short=True, gens=2, pop=4, scen=2, **kw):
    user = cfg_file("phase1_hdg.json")
    prof = {"c172x": "phase1_c172x", "T38": "phase1_T38", "737": "phase1_737", "f16": "phase1_f16"}
    user.update(run_id=run_id, runs_dir=str(tmp_path), cache={"enabled": False, "path": "x"}, scenarios=scen,
                ga={"pop_size": pop, "generations": gens}, trajectories={"generations": [0, gens - 1], "scenario": 0, "sample_hz": 30},
                aircraft=[{"name": a, "profile": prof[a], "overrides": dict(SHORT) if short else {}} for a in aircraft])
    user.update(kw)
    return user


def run_batch(user, workers=2, viz=None):
    cfg = batch.resolve_config(user, "t")
    b = batch.Batch(cfg, log=lambda *a: None)
    b.workers = workers
    if viz is not None:
        b.viz = viz
    b.run()
    return os.path.join(user["runs_dir"], user["run_id"])


def phase1_profile_d(name="c172x", **over):
    cfg = batch.resolve_config(cfg_file("phase1_hdg.json"), "p")
    d = dict(next(a for a in cfg["aircraft"] if a["name"] == name)["resolved_profile"])
    d.update(over)
    return d


G8 = {"kp_alt": 0.24, "ki_alt": 0.01, "kd_alt": 0.4, "kp_pitch": 0.07, "ki_pitch": 2e-5, "kd_pitch": 0.015,
      "kp_hdg": 2.0, "ki_hdg": 4e-4}


class Spy:
    """Post-step recorder: t, a few state props and the fcs commands visible at each call."""

    def __init__(self):
        self.calls = []

    def __call__(self, t, fdm, flex_state=None):
        self.calls.append((t, fdm["position/h-sl-ft"], fdm["attitude/theta-rad"],
                           [fdm[p] for p in sim.TrajRecorder.CTRL_PROPS], flex_state is not None))

    def final(self, t, fdm, flex_state=None):
        self.final_t = t


# ------------------------------------------------------------------ recorder protocol
@need_fd
def test_recorder_bit_identical_t0_call_and_pre_step_controls():
    P = sim.Profile.from_dict(phase1_profile_d())
    sc = sim.make_scenarios(3, 1, P)[1]
    base = sim.simulate(G8, sc, P)
    spy = Spy()
    with_spy = sim.simulate(G8, sc, P, recorder=spy)
    rec = sim.TrajRecorder(sc)
    with_rec = sim.simulate(G8, sc, P, recorder=rec)
    assert base["cost"] == with_spy["cost"] == with_rec["cost"]
    n = int(round(sc.duration_s / sim.DT))
    assert len(spy.calls) == n + 1 and spy.calls[0][0] == 0.0 and spy.final_t == n * sim.DT
    assert all(abs(c[0] - k * sim.DT) < 1e-12 for k, c in enumerate(spy.calls))
    rows = rec.rows
    # row 0 = exactly what the t=0 call sees (state)
    assert rows[0][0] == 0.0 and rows[0][11] == spy.calls[0][1] * sim.FT and rows[0][13] == spy.calls[0][2]
    # controls_timing "pre_step": row at t_k carries the commands a post-step recorder sees at t_{k+1}
    by_t = {round(c[0] / sim.DT): c for c in spy.calls}
    for r in rows[:-1]:
        k = round(r[0] / sim.DT)
        assert [r[i] for i in sim.TrajRecorder.CTRL_IDX] == by_t[k + 1][3]
    assert [rows[-1][i] for i in sim.TrajRecorder.CTRL_IDX] == spy.calls[-1][3]   # final row: hold-last
    tr = rec.trajectory()
    assert tr["controls_timing"] == "pre_step"


def test_recorder_is_read_only():
    P = sim.Profile()
    sc = sim.make_scenarios(1, 1, P)[0]

    def bad(t, fdm, fs=None):
        fdm["fcs/elevator-cmd-norm"] = 0.3

    with pytest.raises(TypeError):
        sim.simulate(genome.decode([0.5] * 6), sc, P, recorder=bad)


@need_fd
def test_trajectory_doc_channel_doc_and_controls_timing(tmp_path):
    P = sim.Profile.from_dict(phase1_profile_d())
    sc = sim.make_scenarios(1, 1, P)[0]
    r = sim.simulate(G8, sc, P, record=True)
    doc = trajectory.build_doc(run_id="t", aircraft="c172x", jsbsim_version="1.3.1", git_sha="x", seed=1, generation=0,
                               fitness=r["cost"], gains=G8, scenario=sc.to_dict(), scenario_index=0, sim_result=r,
                               profile=P.to_dict())
    assert doc["controls_timing"] == "pre_step" and "STARTS at t" in doc["controls_timing_doc"]
    extra = [c for c in doc["channels"] if c not in trajectory.REQUIRED_CHANNELS]
    assert set(doc["channel_doc"]) == set(extra)
    assert "REFERENCE" in doc["channel_doc"]["target_alt_m"] and "ramp" in doc["channel_doc"]["target_alt_m"]
    assert "COMMANDED" in doc["channel_doc"]["target_cmd_alt_m"] and "step" in doc["channel_doc"]["target_cmd_alt_m"]
    # the doc is right: target_alt_m ramps (rate-limited) while target_cmd_alt_m steps
    ch = {c: i for i, c in enumerate(doc["channels"])}
    d = np.array(doc["data"])
    ref, cmd = d[:, ch["target_alt_m"]], d[:, ch["target_cmd_alt_m"]]
    assert np.max(np.abs(np.diff(cmd))) > 50 and np.max(np.abs(np.diff(ref))) < 1.0
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    assert validate_traj.validate_file(str(p))["ok"] if hasattr(validate_traj, "validate_file") else True


# ------------------------------------------------------------------ fidelity adapter
@need_fd
def test_model_version_formats_and_uniform_terms():
    pd = phase1_profile_d(**SHORT)
    sc = sim.make_scenarios(1, 1, sim.Profile.from_dict(pd))[0].to_dict()
    out = {}
    for fid in F.FIDELITIES:
        a = F.aggregate([F.evaluate_scenario(pd, G8, None, sc, fid)])
        assert re.match(MV_RE[fid], a["model_version"]), a["model_version"]
        assert set(a["terms"]) == set(F.TERM_KEYS) and all(math.isfinite(v) for v in a["terms"].values())
        assert a["fidelity"] == fid
        out[fid] = a
    assert all(out["rigid"]["terms"][k] == 0.0 for k in F.PRE_TERMS + F.RESP_TERMS)
    assert not set(F.PRE_TERMS + F.RESP_TERMS) & set(out["rigid"]["terms_available"])
    assert set(F.PRE_TERMS + F.RESP_TERMS) <= set(out["full"]["terms_available"])
    assert "heading" in out["rigid"]["terms_available"]
    assert out["full"]["margins_fidelity"] == "full" and out["reduced"]["margins_fidelity"] == "reduced"


@need_fd
def test_oneway_coupling_reproduces_rigid_and_recorder_inert_with_flex(monkeypatch):
    pd = phase1_profile_d(**SHORT)
    sc = sim.make_scenarios(1, 1, sim.Profile.from_dict(pd))[0].to_dict()
    rigid = F.evaluate_scenario(pd, G8, None, sc, "rigid")
    spec = dict(F.SPECS["full"], mode="oneway")
    monkeypatch.setitem(F.SPECS, "full", spec)
    F._MV.clear()
    one = F.evaluate_scenario(pd, G8, None, sc, "full")
    assert one["sim_cost"] == rigid["cost"]        # structure computed, no feedback, baseline mass -> rigid exactly
    monkeypatch.undo()
    F._MV.clear()
    a = F.evaluate_scenario(pd, G8, None, sc, "reduced")
    spy = Spy()
    b = F.evaluate_scenario(pd, G8, None, sc, "reduced", recorder=spy)
    assert a["cost"] == b["cost"] and all(c[4] for c in spy.calls)


@need_fd
def test_flex_state_shape_t0_probe_and_twist_sign():
    fw, cs = F.fd_modules()
    pd = phase1_profile_d(**SHORT)
    P = sim.Profile.from_dict(pd)
    sc = sim.make_scenarios(1, 1, P)[0]
    states = []
    F.evaluate_scenario(pd, G8, None, sc.to_dict(), "full",
                        recorder=lambda t, fdm, fs: states.append((t, fs.channels(), fs.eta.copy())) if len(states) < 2 else None)
    t0, ch0, eta0 = states[0]
    assert t0 == 0.0 and len(ch0) == 2 * 3 * F.N_NODES
    assert set(k.split(".")[0] for k in ch0) == {"wingR", "wingL"} and all(re.match(r"^wing[RL]\.(dz|dy|twist)\.\d+$", k) for k in ch0)
    # t=0 probe = FD's initialize() on the same trimmed state (1-g equilibrium), and close to the first step
    assert np.allclose(eta0, states[1][2], rtol=1e-6, atol=1e-9)
    # 1 g: wings bend UP (dz < 0 in body z-down) and twist nose-up (lift ahead of the EA)
    assert ch0[f"wingR.dz.{F.N_NODES - 1}"] < 0 and ch0[f"wingL.dz.{F.N_NODES - 1}"] < 0
    assert ch0["wingR.dz.0"] == 0.0 and ch0["wingR.twist.0"] == 0.0 and ch0["wingR.dy.3"] == 0.0
    assert ch0[f"wingR.twist.{F.N_NODES - 1}"] > 0 and ch0[f"wingL.twist.{F.N_NODES - 1}"] < 0


def test_twist_convention_known_nose_up_case():
    """Pure nose-up torsion: FD raw twist > 0 on BOTH wings (it raises local alpha); Sim Bridge convention
    (right-hand about root->tip) => wingR > 0 (LE up), wingL < 0."""
    if not os.path.isdir(FD_DIR):
        pytest.skip("FD not present")
    fw, _ = F.fd_modules()
    w = fw.FlexWing(fw.params_for("c172x", 35.8, 174.0, 1500.0))
    nb = w.p.n_bend
    Q = w.PsiT.T @ np.full(w.p.n_strips, 10.0)          # +10 ft*lbf/ft nose-up torque about the elastic axis
    eta = np.linalg.solve(w.K, Q)
    assert eta[nb] > 0 and (w.G @ eta)[-1] > 0           # FD: + twist = nose-up = + local alpha
    c = fw.FlexCoupler(w, mode="oneway")                 # same through FD's external loads: lift ahead of the EA
    s = dict(qbar=50.0, vt=170.0, alpha=0.05, beta=0.0, kap=1.0, p=0.0, pdot=0.0, nz=0.0, lift=2000.0, ail_r=0.0, ail_l=0.0)
    etas = []
    for side in (0, 1):
        L, M = c.external_loads(s, side)
        etas.append(np.linalg.solve(w.K, c.gen_force(L, M, s, side)))
    assert etas[0][nb] > 0 and etas[1][nb] > 0
    geom = {"B": np.array([[float(fw._bend(k, x)) for k in range(nb)] for x in np.linspace(0, 1, 5)]),
            "T": np.array([float(fw._tors(x)) for x in np.linspace(0, 1, 5)]), "nb": nb}
    ch = F.fd_to_structure_channels(np.array(etas), geom)
    assert ch["wingR.twist.4"] == pytest.approx(etas[0][nb]) and ch["wingL.twist.4"] == pytest.approx(-etas[1][nb])
    assert ch["wingR.dz.4"] < 0 and ch["wingL.dz.4"] < 0       # lift bends the wings up = -z body


@need_fd
def test_margin_gates_reduced_09_full_10(monkeypatch):
    pd = phase1_profile_d(**SHORT)
    sc = sim.make_scenarios(1, 1, sim.Profile.from_dict(pd))[0].to_dict()
    fw, _ = F.fd_modules()
    real = fw.margin_terms

    def fake(wing, wts, rho=fw.RHO0):
        m = real(wing, wts, rho)
        m["margins"] = dict(m["margins"], flutter_margin=0.95)
        return m

    monkeypatch.setattr(fw, "margin_terms", fake)
    red = F.evaluate_scenario(pd, G8, None, sc, "reduced")
    full = F.evaluate_scenario(pd, G8, None, sc, "full")
    assert red["status"] == "ok" and red["pre"]["fail"] is None and red["pre"]["margins_fidelity"] == "reduced"
    assert full["status"] == "flutter" and full["cost"] == 2000.0 and full["pre"]["margins_fidelity"] == "full"


# ------------------------------------------------------------------ run.json / genomes.jsonl / evaluate()
@need_fd
def test_evaluate_reproduces_genomes_jsonl_exactly(tmp_path):
    rd = run_batch(tiny_user(tmp_path, "evrep", aircraft=("c172x", "f16")))
    run_cfg = ev.load_run_cfg(rd)
    rows = runinfo.read_rows(os.path.join(rd, "genomes.jsonl"))
    assert run_cfg["schema"] == "ga-flightsim-run/1" and run_cfg["fitness_sense"] == "min"
    assert len(rows) == 2 * 4 * 2 and len({(r["aircraft"], r["individual_id"]) for r in rows}) == len(rows)
    for r in rows:
        assert re.match(MV_RE["rigid"], r["model_version"]) and r["model_version"] == run_cfg["model_version"][r["aircraft"]]
    sel = [r for r in rows if r["is_best"]] + [r for r in rows if not r["is_best"]][:3]
    for r in sel:
        out = ev.evaluate(r["genome"], r["aircraft"], r["scenario_ids"], run_cfg, fidelity=r["fidelity"])
        assert out["cost"] == r["cost"] and out["per_scenario_cost"] == r["per_scenario_cost"]
        assert out["model_version"] == r["model_version"] and out["model_version_match"]
        one = ev.evaluate(r["genome"], r["aircraft"], run_cfg["scenarios"][[s["id"] for s in run_cfg["scenarios"]].index(r["scenario_ids"][1])], run_cfg)
        assert one["cost"] == r["per_scenario_cost"][1]
    # scenario definitions are used as given (not rebuilt): editing the entry changes the result
    r = sel[0]
    sc = copy.deepcopy(next(s for s in run_cfg["scenarios"] if s["id"] == r["scenario_ids"][0]))
    # (a larger step would not do: with the 600 fpm / 0.1 g reference the 12 s test scenario only reaches +84 ft)
    moved = dict(sc, steps=[[0.0, sc["steps"][0][1]], [5.0, sc["steps"][0][1] + 100.0]])
    windy = dict(sc, wind_north_fps=15.0)
    assert ev.evaluate(r["genome"], r["aircraft"], moved, run_cfg)["cost"] != r["per_scenario_cost"][0]
    assert ev.evaluate(r["genome"], r["aircraft"], windy, run_cfg)["cost"] != r["per_scenario_cost"][0]
    assert isinstance(ev.scenario_object(r["aircraft"], r["scenario_ids"][0], run_cfg), sim.Scenario)


@need_fd
def test_viz_on_off_bit_identical(tmp_path):
    a = run_batch(tiny_user(tmp_path, "viz-off", viz="off"))
    b = run_batch(tiny_user(tmp_path, "viz-on", viz="on"))
    ra, rb = (runinfo.read_rows(os.path.join(d, "genomes.jsonl")) for d in (a, b))
    strip = lambda rows: [(r["individual_id"], r["cost"], r["per_scenario_cost"], r["genome"]) for r in rows]
    assert strip(ra) == strip(rb)
    for d in (a, b):
        with open(os.path.join(d, "summary.json")) as f:
            assert f.read().count('"resim_mismatches": []') == 1
    assert os.path.exists(os.path.join(b, "live", "c172x.json")) and not os.path.exists(os.path.join(a, "live"))


@need_fd
def test_reduced_fidelity_run_reproduces_and_logs(tmp_path):
    rd = run_batch(tiny_user(tmp_path, "red", gens=2, pop=3, scen=1, fidelity="reduced"))
    run_cfg = ev.load_run_cfg(rd)
    rows = runinfo.read_rows(os.path.join(rd, "genomes.jsonl"))
    assert run_cfg["fidelity"] == "reduced" and re.match(MV_RE["reduced"], rows[0]["model_version"])
    for r in rows[:2]:
        out = ev.evaluate(r["genome"], r["aircraft"], r["scenario_ids"], run_cfg)
        assert out["cost"] == r["cost"] and r["feasibility_fidelity"] == "reduced"


def _mf_user(tmp_path, run_id):
    return tiny_user(tmp_path, run_id, gens=3, pop=5, scen=1, fidelity="reduced", struct_genes=True,
                     multi_fidelity={"enabled": True, "screen": "rigid", "top_k": 2})


@need_fd
def test_multi_fidelity_rescoring_ranking_and_resume(tmp_path):
    ref = run_batch(_mf_user(tmp_path, "mf-ref"))
    rows = runinfo.read_rows(os.path.join(ref, "genomes.jsonl"))
    elite = 2
    for g in range(3):
        gr = sorted([r for r in rows if r["generation"] == g], key=lambda r: r["rank"])
        full = [r for r in gr if r["rescored_at_full"]]
        assert gr[0]["is_best"] and gr[0]["rescored_at_full"] and gr[0]["fidelity"] == "reduced"
        assert all(r["rescored_at_full"] for r in gr if r["is_elite"] or r["is_best"])
        assert all(r["screen_fidelity"] == "rigid" and "screen_cost" in r for r in gr)
        # ranking: full-scored first by full cost, then the rest by screen cost
        assert gr[:len(full)] == sorted(full, key=lambda r: r["cost"])
        rest = gr[len(full):]
        assert rest == sorted(rest, key=lambda r: r["screen_cost"])
        assert all(r["feasibility_fidelity"] == ("reduced" if r["rescored_at_full"] else "rigid") for r in gr)
        if g > 0:   # carried-in elites are re-scored
            assert all(r["rescored_at_full"] for r in gr if r["carried_elite"])
    with open(os.path.join(ref, "history.jsonl")) as f:
        hist = [json.loads(l) for l in f]
    assert all("spearman_screen_vs_full" in h and h["n_rescored"] >= 2 for h in hist)
    # resume after a SIGKILL-like exit between writing generation 1's rows and its checkpoint: identical rows
    user = _mf_user(tmp_path, "mf-kill")
    cfgp = tmp_path / "mf.json"
    cfgp.write_text(json.dumps(user))
    env = dict(os.environ, EVOLUTION_TEST_KILL_AFTER_ROWS="1", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([PY, "-m", "evolution.batch", "--config", str(cfgp), "--workers", "2"], cwd=TEAM, env=env,
                       capture_output=True, text=True)
    assert p.returncode != 0
    env.pop("EVOLUTION_TEST_KILL_AFTER_ROWS")
    p = subprocess.run([PY, "-m", "evolution.batch", "--resume", str(tmp_path / "mf-kill"), "--workers", "2"], cwd=TEAM,
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    rk = runinfo.read_rows(os.path.join(tmp_path, "mf-kill", "genomes.jsonl"))
    key = lambda r: (r["individual_id"], r["cost"], r["screen_cost"], r["rank"], r["rescored_at_full"], r["genome"])
    assert [key(r) for r in rk] == [key(r) for r in rows]


@need_fd
def test_backfill_phase1_rows_replay_exactly(tmp_path):
    src = os.path.join(PKG, "runs", "phase1-s1")
    if not os.path.exists(os.path.join(src, "summary.json")):
        pytest.skip("phase1-s1 not present")
    import shutil
    dst = tmp_path / "phase1-s1"
    os.makedirs(dst / "checkpoints")
    for f in ("config.json", "summary.json"):
        shutil.copy(os.path.join(src, f), dst / f)
    for f in os.listdir(os.path.join(src, "checkpoints")):
        shutil.copy(os.path.join(src, "checkpoints", f), dst / "checkpoints" / f)
    runinfo.backfill(str(dst), log=lambda *a: None)
    run_cfg = ev.load_run_cfg(str(dst))
    rows = runinfo.read_rows(str(dst / "genomes.jsonl"))
    assert len(rows) == 4 * 20 and all(r["backfill"] for r in rows)
    for r in [r for r in rows if r["generation"] == 19 and r["aircraft"] in ("c172x", "f16")]:
        out = ev.evaluate(r["genome"], r["aircraft"], None, run_cfg)
        assert out["per_scenario_cost"] == r["per_scenario_cost"] and out["cost"] == r["cost"]
