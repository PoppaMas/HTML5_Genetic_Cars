"""Heading hold (genome/HANDOFF_heading_hold.md): bit-for-bit agreement with genome/'s best genomes, flag-off path
unchanged, gene layout, config == genome export, cache key, gen-0 sizing, and the socket guard policy."""
import json
import os
import shutil

import numpy as np
import pytest

from evolution import batch, cache, genome, sim

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TEAM = os.path.dirname(PKG)
FD_ROOT = os.path.join(TEAM, "flight-dynamics", "jsbsim_root")
GENOME = os.path.join(TEAM, "genome")
EXPORT = os.path.join(GENOME, "exports", "evolution_phase1_profiles.json")
need_fd = pytest.mark.skipif(not os.path.isdir(FD_ROOT), reason="flight-dynamics/jsbsim_root not present")
HDG_KEYS = ("heading_hold", "bank_limit_deg", "hdg_i_limit_deg", "w_heading", "hdg_rms_ref_deg")

# genome/HANDOFF_heading_hold.md section 3 (genes in genome order, cost = mean of 3, per-scenario costs)
HANDOFF = {
    "c172x": ([0.2397326688652956, 0.05, 0.38523085001510937, 0.07049452717617341, 1.9653025820764143e-05,
               0.014608444547277827, 2.242870209249964, 0.0003706638173012445], 0.19600587132367517,
              [0.10529427560176219, 0.31566235928965425, 0.167060979079609]),
    "T38": ([0.22365803353253752, 4.1598113596235896e-07, 0.0941094102393032, 0.06574330468426912, 0.0,
             0.0024262902418641465, 1.945038464385377, 0.00475971406828075], 0.0966245736792507,
            [0.07072963520626355, 0.12908011182321508, 0.0900639740082735]),
    "737": ([0.19092554817716106, 0.030997595118024638, 0.34324408429885966, 0.05970803545301804, 0.005608195701037316,
             0.036460842911213105, 0.23541201575286405, 0.008622211397029298], 0.10678211633774594,
            [0.07842957400564782, 0.1370541422493403, 0.10486263275824974]),
}
HANDOFF_RUNS = {"c172x": "hdg_after_c172x_s1", "T38": "hdg_after_t38_s1", "737": "hdg_after_b737_s1"}


def hdg_cfg():
    with open(os.path.join(PKG, "configs", "phase1_hdg.json")) as f:
        return batch.resolve_config(json.load(f), "phase1_hdg")


def prof(cfg, name):
    return next(a for a in cfg["aircraft"] if a["name"] == name)["resolved_profile"]


@need_fd
@pytest.mark.parametrize("name", list(HANDOFF))
def test_reproduces_genome_heading_genomes_bitwise(name):
    P = sim.Profile.from_dict(prof(hdg_cfg(), name))
    schema = genome.make_schema(P.gain_bounds, P.gene_kinds, P.heading_hold)
    genes, cost, per = HANDOFF[name]
    gains = dict(zip([g.name for g in schema], genes))
    bg = os.path.join(GENOME, "runs", HANDOFF_RUNS[name], "best_gains.json")
    if os.path.exists(bg):  # genome's normalized genome decodes to exactly these gains with our schema
        with open(bg) as f:
            assert genome.decode(json.load(f)["genome"], schema) == gains
    rs = [sim.simulate(gains, sc, P) for sc in sim.make_scenarios(3, 1, P)]
    assert [r["cost"] for r in rs] == per
    assert float(np.mean([r["cost"] for r in rs])) == cost
    assert all(r["status"] == "ok" and abs(r["hdg_drift_deg"]) < 1.0 for r in rs)


@need_fd
def test_flag_off_is_bit_identical_and_heading_settings_inert():
    """heading_hold false: the heading keys (any values) change nothing; equals the phase1 (no-key) profile."""
    with open(os.path.join(PKG, "configs", "phase1.json")) as f:
        base = batch.resolve_config(json.load(f), "phase1")
    P0 = sim.Profile.from_dict(prof(base, "c172x"))
    d = dict(prof(hdg_cfg(), "c172x"), heading_hold=False, bank_limit_deg=10.0, w_heading=0.5)
    d["gain_bounds"] = {k: v for k, v in d["gain_bounds"].items() if k not in ("kp_hdg", "ki_hdg")}
    d["gene_kinds"] = {k: v for k, v in d["gene_kinds"].items() if k not in ("kp_hdg", "ki_hdg")}
    P1 = sim.Profile.from_dict(d)
    g = genome.decode([0.6, 0.3, 0.5, 0.5, 0.4, 0.5], genome.make_schema(P0.gain_bounds, P0.gene_kinds))
    sc = sim.make_scenarios(3, 1, P0)[1]
    a, b = sim.simulate(g, sc, P0), sim.simulate(dict(g, kp_hdg=3.0, ki_hdg=0.05), sc, P1)
    assert a["cost"] == b["cost"] and "heading_rms" not in a and "heading_rms" not in b
    assert P0.heading_hold is False and sim.Profile().heading_hold is False


def test_schema_layout_and_validation():
    s = genome.make_schema(heading_hold=True)
    assert [g.name for g in s] == genome.GENE_NAMES + ["kp_hdg", "ki_hdg"]
    assert (s[6].kind, s[7].kind) == ("log", "log0") and genome.N_GENES == 6 and len(genome.make_schema()) == 6
    with pytest.raises(ValueError, match="heading_hold"):
        genome.make_schema({"kp_hdg": [0.1, 1.0]})
    with pytest.raises(ValueError):
        sim.Profile.from_dict({"bank_limit_deg": 50.0})   # >= max_abs_phi_deg 45
    with pytest.raises(ValueError):
        sim.Profile.from_dict({"w_heading": -0.1})


@pytest.mark.skipif(not os.path.exists(EXPORT), reason="genome export not present")
def test_phase1_hdg_config_equals_genome_export():
    with open(EXPORT) as f:
        exp = json.load(f)["profiles"]
    cfg = hdg_cfg()
    for name, key in (("c172x", "phase1_c172x"), ("T38", "phase1_T38"), ("737", "phase1_737")):
        p = prof(cfg, name)
        for k, v in exp[key].items():
            if not k.startswith("_"):
                if k == "aircraft_root":   # the export is repo-relative; resolve_config makes it absolute
                    v = sim.abs_root(v)
                assert json.loads(json.dumps(p[k])) == v, (name, k)
        assert p["heading_hold"] is True and p["w_heading"] == 0.01
    f16 = prof(cfg, "f16")
    assert f16["heading_hold"] is False and f16["aircraft_root"] == FD_ROOT and f16["throttle_max"] == 0.5


def test_cache_key_changes_with_heading_settings():
    p = prof(hdg_cfg(), "c172x")
    sc = sim.make_scenarios(1, 1, sim.Profile.from_dict(p))[0].to_dict()
    g = np.full(8, 0.5)
    k = lambda pd: cache.eval_key("c172x", g, pd, sc, 1, "1.3.1", "x", "m")
    keys = {k(p), k(dict(p, heading_hold=False)), k(dict(p, bank_limit_deg=20.0)), k(dict(p, w_heading=0.02)),
            k(dict(p, hdg_i_limit_deg=5.0)), k(dict(p, hdg_rms_ref_deg=4.0))}
    assert len(keys) == 6


@need_fd
def test_batch_sizes_generation_zero_from_schema(tmp_path):
    with open(os.path.join(PKG, "configs", "phase1_hdg.json")) as f:
        user = json.load(f)
    user.update(run_id="hdg-gen0", runs_dir=str(tmp_path), cache={"enabled": False, "path": "x"}, scenarios=1,
                ga={"pop_size": 4, "generations": 1}, trajectories={"generations": [0], "scenario": 0, "sample_hz": 30},
                aircraft=[{"name": "c172x", "profile": "phase1_c172x", "overrides": {"duration_s": 8.0,
                                                                                   "steps_rel_ft": [[0.0, 0.0], [2.0, 50.0]]}}])
    cfg = batch.resolve_config(user, "t")
    b = batch.Batch(cfg, log=lambda *a: None)
    b.workers = 2
    b.run()
    with open(os.path.join(tmp_path, "hdg-gen0", "checkpoints", "c172x.json")) as f:
        ck = json.load(f)
    assert all(len(g) == 8 for g in ck["pop"])
    assert set(ck["history"][0]["best_gains"]) == set(genome.GENE_NAMES) | {"kp_hdg", "ki_hdg"}


def test_socket_guard_refuses_explicit_root_and_strips_package_loudly(tmp_path, capfd, monkeypatch):
    monkeypatch.delenv("EVOLUTION_SOCKET_POLICY", raising=False)
    src = sim.model_dir("c172x")
    root = tmp_path / "root"
    shutil.copytree(src, root / "aircraft" / "c172x")
    xml = root / "aircraft" / "c172x" / "c172x.xml"
    xml.write_text(xml.read_text().replace("</fdm_config>", '<input port="5137"/>\n</fdm_config>'))
    assert sim.socket_io_elements("c172x", str(root)) == 1 and sim.socket_policy(str(root)) == "refuse"
    with pytest.raises(sim.SimSetupError, match="refusing"):
        sim._aircraft_root("c172x", str(root))
    monkeypatch.setenv("EVOLUTION_SOCKET_POLICY", "strip")
    sim._ROOT_MEMO.pop((str(root), "c172x"), None)
    r = sim._aircraft_root("c172x", str(root))
    assert r and r.startswith(sim._SAN_ROOT) and "WARNING" in capfd.readouterr().err
    if os.path.isdir(FD_ROOT):   # FD's prepared models carry no socket I/O any more: the strip is a no-op there
        for m in ("c172x", "T38", "737", "f16"):
            if os.path.exists(os.path.join(FD_ROOT, "aircraft", m, m + ".xml")):
                assert sim.socket_io_elements(m, FD_ROOT) == 0
