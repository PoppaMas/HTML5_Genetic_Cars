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
         "full": r"^full:flexv2:[0-9a-f]{8}$", "full_a1": r"^full_a1:flexv2a1:[0-9a-f]{8}$"}


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


# ------------------------------------------------------------------ fidelity adapter (FD flexeval v2)
def _sc(pd, n=1):
    return [s.to_dict() for s in sim.make_scenarios(n, 1, sim.Profile.from_dict(pd))]


@need_fd
def test_model_version_formats_and_uniform_terms():
    pd = phase1_profile_d(**SHORT)
    sc = _sc(pd)
    fe = F.fd_modules()["fe"]
    assert tuple(fe.TERM_KEYS) == F.TERM_KEYS
    out = {}
    for fid in F.FIDELITIES:
        if fid == "full_a1" and not os.path.exists(os.path.join(F.FD_DIR, "flexeval_a1.py")):
            continue   # P3-A1 is opt-in: an FD tree without flexeval_a1/flexbody_a1 serves rigid / reduced / full
        a = F.evaluate_genome(pd, G8, None, sc, fid, 0.9)
        assert re.match(MV_RE[fid], a["model_version"]), a["model_version"]
        assert set(a["terms"]) == set(F.TERM_KEYS) and all(math.isfinite(v) for v in a["terms"].values())
        assert a["fidelity"] == fid and a["feasibility_fidelity"] == fid
        out[fid] = a
    J = [k for k in F.TERM_KEYS if k.startswith("J_")]
    assert all(out["rigid"]["terms"][k] == 0.0 for k in J) and not set(J) & set(out["rigid"]["terms_available"])
    assert {"J_flutter_margin", "J_tail_bm_peak", "J_fus_bm_peak", "J_reversal_margin"} <= set(out["full"]["terms_available"])
    assert "heading" in out["rigid"]["terms_available"]
    assert out["full"]["margins_fidelity"] == "full" and out["reduced"]["margins_fidelity"] == "reduced"
    assert out["full"]["margin_gate"] == 1.0 and out["reduced"]["margin_gate"] == 0.9
    # reduced model_version hashes the gate (per-aircraft config); full does not; FD's global is restored
    assert F.model_version(pd, "reduced", 0.9) != F.model_version(pd, "reduced", 1.0)
    assert F.model_version(pd, "full", 0.9) == F.model_version(pd, "full", 1.0)
    assert fe.MARGIN_GATE == {"reduced": 0.9, "full": 1.0}
    assert F.per_aircraft("c172x")["reduced_gate"] == 0.9
    assert all(F.per_aircraft(a)["reduced_gate"] == 1.0 and F.per_aircraft(a)["min_full_frac"] == 0.25 for a in ("737", "T38", "f16"))


@need_fd
def test_adapter_is_fd_flexeval_bit_for_bit():
    """reduced/full = FD's flexeval.evaluate itself (imported, not forked); one scenario = that scenario's entry."""
    pd = phase1_profile_d(**SHORT)
    scs = _sc(pd, 2)
    m = F.fd_modules()
    fe, P = m["fe"], sim.Profile.from_dict(pd)
    for fid, gate in (("full", 1.0), ("reduced", 0.9)):
        ours = F.evaluate_genome(pd, G8, None, scs, fid, gate)
        root, rv2 = F.roots(P)
        old = dict(fe.MARGIN_GATE)
        fe.MARGIN_GATE["reduced"] = gate
        try:
            theirs = fe.evaluate(G8, None, [sim.Scenario.from_dict(s) for s in scs], "c172x", fidelity=fid, root=root,
                                 root_v2=rv2, profile=P, sim=sim)
        finally:
            fe.MARGIN_GATE.update(old)
        assert ours["cost"] == theirs["cost"] and ours["model_version"] == theirs["model_version"]
        assert [e["cost"] for e in ours["per_scenario"]] == [e["cost"] for e in theirs["per_scenario"]]
        one = F.evaluate_scenario(pd, G8, None, scs[1], fid, reduced_gate=gate)
        assert one["cost"] == ours["per_scenario"][1]["cost"]
    # FD's prepare functions are disabled in our processes (FD's folder is read-only from here)
    with pytest.raises(F.FidelityUnavailable):
        m["fw"].prepare_aircraft("c172x", "/nonexistent")


@need_fd
def test_recorder_inert_t0_flex_state_and_twist_sign():
    pd = phase1_profile_d(**SHORT)
    sc = _sc(pd)
    a = F.evaluate_genome(pd, G8, None, sc, "full", 1.0)
    spy, states = Spy(), []

    def rec(t, fdm, fs):
        spy(t, fdm, fs)
        if len(states) < 2:
            states.append((t, fs.channels(), np.array(fs.eta, copy=True), fs))
    b = F.evaluate_genome(pd, G8, None, sc, "full", 1.0, recorder=rec)
    assert a["cost"] == b["cost"] and b["telemetry_check"]["sim_cost_bit_identical"]
    t0, ch0, eta0, fs0 = states[0]
    assert t0 == 0.0 and spy.calls[0][0] == 0.0 and all(c[4] for c in spy.calls)
    modal = [k for k in ch0 if re.match(r"^wing[RL]_modal\.(dz|dy|twist)\.\d+$", k)]
    assert len(modal) == 2 * 3 * F.N_NODES
    assert any(k.startswith("wingR.dz.") for k in ch0) and any(k.startswith("wingR.dx.") for k in ch0)
    assert any(k.startswith("htail.") for k in ch0) and any(k.startswith("struct.") for k in ch0)
    assert any(k.startswith("flex.") for k in ch0)
    assert fs0.structure["schema"] == "evolution-flex-state/3"
    names = [c["name"] for c in fs0.structure["components"]]
    assert names[:2] == ["wingR_modal", "wingL_modal"]
    assert set(names) >= {"wingR_modal", "wingL_modal", "wingR", "wingL", "htail", "vtail", "fuselage"}
    assert np.allclose(eta0, states[1][2], rtol=1e-3, atol=1e-6)     # t=0 probe = trim equilibrium ~ first step
    nR = max(int(k.split(".")[-1]) for k in ch0 if k.startswith("wingR.dz."))
    nL = max(int(k.split(".")[-1]) for k in ch0 if k.startswith("wingL.dz."))
    assert ch0[f"wingR.dz.{nR}"] < 0 and ch0[f"wingL.dz.{nL}"] < 0      # 1 g: wings bend up = -z body
    assert ch0["wingR.dz.0"] == 0.0 and ch0["wingR.twist.0"] == 0.0 and ch0["wingR.dx.0"] == 0.0
    assert ch0[f"wingR.twist.{nR}"] > 0 > ch0[f"wingL.twist.{nL}"]      # 1 g, c172x: FD raw nose-up on both wings
    nm = F.N_NODES - 1
    assert ch0[f"wingR_modal.dz.{nm}"] < 0 and ch0[f"wingL_modal.dz.{nm}"] < 0
    assert ch0[f"wingR_modal.dy.{nm}"] == 0.0
    # mapping vs FD's raw channels (FD: + = nose-up on both wings): SB wingR = +raw_R, wingL = -raw_L
    # FE tip node (last FE node, not the 9-node modal index) matches FD tip scalars via v2_map
    nR = max(int(k.split(".")[-1]) for k in ch0 if k.startswith("wingR.twist."))
    nL = max(int(k.split(".")[-1]) for k in ch0 if k.startswith("wingL.twist."))
    assert ch0[f"wingR.twist.{nR}"] == pytest.approx(math.radians(ch0["flex.tip_twist_R_deg"]), rel=1e-9, abs=1e-12)
    assert ch0[f"wingL.twist.{nL}"] == pytest.approx(-math.radians(ch0["flex.tip_twist_L_deg"]), rel=1e-9, abs=1e-12)
    assert fs0.fd_model is not None and fs0.nodes() is not None and fs0.v2_map_version == "2.0.0"
    assert "wingR" in fs0.nodes() and fs0.node_layout() is not None


def test_twist_convention_known_nose_up_case():
    """Pure nose-up torsion: FD raw twist > 0 on BOTH wings (it raises local alpha); Sim Bridge convention
    (right-hand about root->tip) => wingR > 0 (LE up), wingL < 0. v1 wing (reduced) through FD's own loads, and the
    v2 Surface (full) through its PsiT shape."""
    if not os.path.isdir(FD_DIR):
        pytest.skip("FD not present")
    m = F.fd_modules()
    fw, fb = m["fw"], m["fb"]
    w = fw.FlexWing(fw.params_for("c172x", 35.8, 174.0, 1500.0))
    nb = w.p.n_bend
    Q = w.PsiT.T @ np.full(w.p.n_strips, 10.0)          # +10 ft*lbf/ft nose-up torque about the elastic axis
    eta = np.linalg.solve(w.K, Q)
    assert eta[nb] > 0 and (w.G @ eta)[-1] > 0           # FD: + twist = nose-up = + local alpha
    c = fw.FlexCoupler(w, mode="oneway")
    s = dict(qbar=50.0, vt=170.0, alpha=0.05, beta=0.0, kap=1.0, p=0.0, pdot=0.0, nz=0.0, lift=2000.0, ail_r=0.0, ail_l=0.0)
    etas = []
    for side in (0, 1):
        L, M = c.external_loads(s, side)
        etas.append(np.linalg.solve(w.K, c.gen_force(L, M, s, side)))
    assert etas[0][nb] > 0 and etas[1][nb] > 0
    geom = {"fns": [(nm, fn) for nm, _, fn in F._wing_surfaces(w, "reduced")]}
    ch = F.fd_to_structure_channels(np.array(etas), geom)
    n = F.N_NODES - 1
    assert ch[f"wingR.twist.{n}"] == pytest.approx(etas[0][nb] * float(fw._tors(1.0)))
    assert ch[f"wingL.twist.{n}"] == pytest.approx(-etas[1][nb] * float(fw._tors(1.0)))
    assert ch[f"wingR.dz.{n}"] < 0 and ch[f"wingL.dz.{n}"] < 0
    rv2 = os.path.join(FD_DIR, "jsbsim_root_v2")
    mdl = fb.FlexBodyModel("c172x", None, asymmetric=False, root_v2=rv2)
    e = np.zeros(max(sl.stop for sl in mdl.slices.values()))
    for nm in ("wingR", "wingL"):
        surf, sl = getattr(mdl, nm), mdl.slices[nm]
        e[sl] = np.linalg.lstsq(surf.PsiT, np.full(surf.PsiT.shape[0], 0.01), rcond=None)[0]   # raw +0.01 rad nose-up
    ch2 = F.fd_to_structure_channels(e, {"fns": [(nm, fn) for nm, _, fn in F._wing_surfaces(mdl, "full")]})
    assert ch2["wingR.twist.4"] == pytest.approx(0.01, rel=0.05) and ch2["wingL.twist.4"] == pytest.approx(-0.01, rel=0.05)


def test_channel_doc_flex_raw_and_sb_channels():
    d = trajectory.channel_doc(["wingR.twist.8", "wingL.dz.0", "flex.tip_twist_R_deg", "flex.tip_w_ft_L", "flex.wingR_bm"])
    assert "right-hand rotation" in d["wingR.twist.8"] and "+ down" in d["wingL.dz.0"]
    assert "BOTH wings" in d["flex.tip_twist_R_deg"] and d["flex.tip_twist_R_deg"].count("deg")
    assert "dz = -0.3048" in d["flex.tip_w_ft_L"] and ", ft;" in d["flex.tip_w_ft_L"]
    assert "lbf*ft" in d["flex.wingR_bm"] and not any(v == "undocumented extra channel" for v in d.values())


@need_fd
def test_margin_gates_per_aircraft(monkeypatch):
    """reduced hard-fails below the aircraft's gate (c172x 0.9; 737/T38/f16 1.0); a gate fail is not flown,
    cost = fail_cost = 2 * fail_base, one aligned not_flown entry per scenario."""
    pd = phase1_profile_d(**SHORT)
    scs = _sc(pd, 2)
    fw = F.fd_modules()["fw"]
    real = fw.FlexWing.margins

    def fake(self, *a, **k):
        m = real(self, *a, **k)
        return dict(m, flutter_margin=0.95)
    monkeypatch.setattr(fw.FlexWing, "margins", fake)
    ok = F.evaluate_genome(pd, G8, None, scs, "reduced", 0.9)
    bad = F.evaluate_genome(pd, G8, None, scs, "reduced", 1.0)
    assert ok["status"] == "ok" and ok["margins"]["flutter_margin"] == 0.95 and ok["margin_gate"] == 0.9
    fail_cost = 2 * sim.Profile.from_dict(pd).fail_base
    assert bad["status"] == "flutter" and bad["cost"] == fail_cost and bad["margin_gate"] == 1.0
    assert [e["cost"] for e in bad["per_scenario"]] == [fail_cost] * 2 and all(e["not_flown"] for e in bad["per_scenario"])


# ------------------------------------------------------------------ run.json / genomes.jsonl / evaluate()
@need_fd
def test_evaluate_reproduces_genomes_jsonl_exactly(tmp_path):
    rd = run_batch(tiny_user(tmp_path, "evrep", aircraft=("c172x", "f16")))
    run_cfg = ev.load_run_cfg(rd)
    rows = runinfo.read_rows(os.path.join(rd, "genomes.jsonl"))
    assert run_cfg["schema"] == "ga-flightsim-run/1" and run_cfg["fitness_sense"] == "min"
    assert len(rows) == 2 * 4 * 2 and len({(r["aircraft"], r["individual_id"]) for r in rows}) == len(rows)
    # ids: stable opaque strings '<aircraft>:g<generation>:r<rank>' (documented in run.json + README; Sim Bridge matches
    # them exactly), and scenario ids '<aircraft>:s<index>'
    assert run_cfg["individual_id_format"].startswith("'<aircraft>:g<generation>:r<rank>'")
    assert run_cfg["scenario_id_format"].startswith("'<aircraft>:s<index>'")
    for r in rows:
        assert isinstance(r["individual_id"], str) and r["individual_id"] == f"{r['aircraft']}:g{r['generation']}:r{r['rank']}"
        assert (r["rank"] == 0) == r["is_best"]
        assert r["eval_seed"] == run_cfg["eval_seed"]
    # paths in run.json are team-relative (sim.abs_root resolves them); git.repo too
    assert not os.path.isabs(run_cfg["git"]["repo"])
    for a in run_cfg["aircraft"]:
        assert a["aircraft_root"] is None or not os.path.isabs(a["aircraft_root"])
        assert a["resolved_profile"].get("aircraft_root") == a["aircraft_root"]
        if a["aircraft_root"]:
            assert os.path.isdir(sim.abs_root(a["aircraft_root"]))
    # trajectory headers: fitness_sense exactly "min", scenario_id = <aircraft>:s<scenario_index>
    import glob
    for fp in glob.glob(os.path.join(rd, "trajectories", "traj_*.json")):
        d = json.load(open(fp))
        assert d["fitness_sense"] == "min" and d["scenario_id"] == f"{d['aircraft']}:s{d['scenario_index']}"
        assert r["scenario_ids"] == [f"{r['aircraft']}:s{i}" for i in range(len(r["scenario_ids"]))]
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
    assert run_cfg["aircraft"][0]["reduced_gate"] == 0.9 and len(run_cfg["aircraft"][0]["genes"]) == 8


def _mf_user(tmp_path, run_id, screen="rigid", **kw):
    return tiny_user(tmp_path, run_id, gens=3, pop=5, scen=1, fidelity="full", struct_genes=True,
                     multi_fidelity={"enabled": True, "screen": screen, "top_k": 2}, **kw)


@need_fd
def test_multi_fidelity_rescoring_ranking_and_resume(tmp_path):
    ref = run_batch(_mf_user(tmp_path, "mf-ref"))
    rows = runinfo.read_rows(os.path.join(ref, "genomes.jsonl"))
    run_cfg = ev.load_run_cfg(ref)
    assert len(run_cfg["aircraft"][0]["genes"]) == 8 + 12          # gains + FD's 12 v2 struct genes
    for g in range(3):
        gr = sorted([r for r in rows if r["generation"] == g], key=lambda r: r["rank"])
        full = [r for r in gr if r["rescored_at_full"]]
        assert gr[0]["is_best"] and gr[0]["rescored_at_full"] and gr[0]["fidelity"] == "full"
        assert all(r["rescored_at_full"] for r in gr if r["is_elite"] or r["is_best"])
        assert all(r["screen_fidelity"] == "rigid" and "screen_cost" in r for r in gr)
        # ranking: full-scored first by full cost, then the rest by screen cost
        assert gr[:len(full)] == sorted(full, key=lambda r: r["cost"])
        rest = gr[len(full):]
        assert rest == sorted(rest, key=lambda r: r["screen_cost"])
        # feasibility is trusted only from full
        assert all(r["feasibility_fidelity"] == "full" and r["feasible"] == (r["status"] == "ok") for r in full)
        assert all(r["feasible"] is None and r["feasibility_fidelity"] is None for r in rest)
        if g > 0:   # carried-in elites are re-scored
            assert all(r["rescored_at_full"] for r in gr if r["carried_elite"])
    for r in [r for r in rows if r["rescored_at_full"]][:2]:       # full rows replay exactly through evaluate()
        assert ev.evaluate(r["genome"], r["aircraft"], r["scenario_ids"], run_cfg, fidelity="full")["cost"] == r["cost"]
    with open(os.path.join(ref, "history.jsonl")) as f:
        hist = [json.loads(l) for l in f]
    assert all("rigid_vs_full" in h["spearman"] and h["n_rescored"] >= 2 for h in hist)
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
def test_three_level_ladder_and_min_full_fraction(tmp_path):
    """rigid -> reduced -> full; min_full_frac (per aircraft, 0.25 on swept wings) forces >= ceil(frac * pop) at full."""
    user = _mf_user(tmp_path, "mf3", screen=["rigid", "reduced"], fidelity_per_aircraft={"c172x": {"min_full_frac": 0.6}})
    user["ga"]["generations"] = 2
    user["trajectories"]["generations"] = [0, 1]
    rd = run_batch(user)
    rows = runinfo.read_rows(os.path.join(rd, "genomes.jsonl"))
    with open(os.path.join(rd, "history.jsonl")) as f:
        hist = [json.loads(l) for l in f]
    assert all(h["ladder"] == ["rigid", "reduced", "full"] and h["k_full"] == 3 and h["n_rescored"] >= 3 for h in hist)
    assert all(set(h["spearman"]) == {"rigid_vs_full", "reduced_vs_full", "rigid_vs_reduced"} for h in hist)
    for r in rows:
        assert "rigid" in r["ladder_cost"]
        assert ("full" in r["ladder_cost"]) == r["rescored_at_full"]
        if r["rescored_at_full"]:
            assert "reduced" in r["ladder_cost"] and r["cost"] == r["ladder_cost"]["full"]
    run_cfg = ev.load_run_cfg(rd)
    a = run_cfg["aircraft"][0]
    assert a["min_full_frac"] == 0.6 and a["reduced_gate"] == 0.9 and set(a["ladder_model_version"]) == {"rigid", "reduced", "full"}
    with pytest.raises(ValueError):
        batch.resolve_config(_mf_user(tmp_path, "bad", screen=["reduced", "rigid"]), "t")


@need_fd
def test_flex_viz_on_off_bit_identical(tmp_path):
    kw = dict(gens=2, pop=3, scen=1, fidelity="full", struct_genes=True)
    a = run_batch(tiny_user(tmp_path, "fviz-off", viz="off", **kw))
    b = run_batch(tiny_user(tmp_path, "fviz-on", viz="on", **kw))
    ra, rb = (runinfo.read_rows(os.path.join(d, "genomes.jsonl")) for d in (a, b))
    strip = lambda rows: [(r["individual_id"], r["cost"], r["per_scenario_cost"], r["genome"]) for r in rows]
    assert strip(ra) == strip(rb)
    for d in (a, b):
        with open(os.path.join(d, "summary.json")) as f:
            assert f.read().count('"resim_mismatches": []') == 1
    with open(os.path.join(b, "live", "c172x.json")) as f:
        live = json.load(f)
    assert live["fidelity"] == "full" if "fidelity" in live else True


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
