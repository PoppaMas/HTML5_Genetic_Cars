"""Phase-1 v5 candidate (genome/HANDOFF_phase1_v5.md): hold-oscillation term + sustained-downdraft 4th scenario."""
import json
import os

import numpy as np
import pytest

from evolution import batch, cache, runinfo, sim

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM = os.path.dirname(PKG)
EXPORT5 = os.path.join(TEAM, "genome", "exports", "evolution_phase1_v5_profiles.json")
GENES = ["kp_alt", "ki_alt", "kd_alt", "kp_pitch", "ki_pitch", "kd_pitch", "kp_hdg", "ki_hdg"]

# genome/HANDOFF_phase1_v5.md section 3 (genes, cost = mean of 4, per-scenario costs, hold_osc per scenario)
REF = {
    "c172x_v5_s1": ("c172x", [0.31437738799504616, 0.05, 0.3366549881674388, 0.08049753722927444, 1.190156607327809e-05,
                              0.013658908124502132, 2.369220826682193, 0.00013558574893233208],
                    0.20625113824760744, [0.12785226753903256, 0.37197659717077186, 0.1995611326112344, 0.1256145556693909],
                    [0.256607, 0.519764, 0.366012, 0.349807]),
    "T38_v5_s1": ("T38", [0.22365803353253752, 0.05, 0.22164873364794022, 0.11246859204665012, 0.005717237540206061,
                          0.03970195171260285, 1.691686672038538, 0.001567717437456872],
                  0.08778028702197657, [0.05596533384050053, 0.15315702451795365, 0.10218697003758949, 0.03981181969186257],
                  [0.017774, 0.268917, 0.217608, 0.219374]),
    "737_v5_s1": ("737", [0.22211834556429566, 0.027826279920683275, 0.39030385285880465, 0.04034489490119028,
                          0.0019985506113642682, 0.019528305027763263, 0.29667099721658546, 0.006343030445800094],
                  0.11126020207078736, [0.08574017385551826, 0.17287390261551305, 0.1343919545022519, 0.0520347773098663],
                  [0.031471, 0.303025, 0.248167, 0.295660]),
    "c172x_v4_on_v5": ("c172x", [0.2397326688652956, 0.05, 0.38523085001510937, 0.07049452717617341, 1.9653025820764143e-05,
                                 0.014608444547277827, 2.242870209249964, 0.0003706638173012445],
                       0.22748161325788208, [0.1474148332864362, 0.377508656233405, 0.2141770857427959, 0.17082587776889113],
                       [0.421206, 0.618463, 0.471161, 0.593962]),
}


def v5_profiles():
    cfg = batch.resolve_config(json.load(open(os.path.join(PKG, "configs", "phase1_v5.json"))), "v5")
    return {a["name"]: a["resolved_profile"] for a in cfg["aircraft"]}, cfg


@pytest.mark.parametrize("key", list(REF))
def test_reproduces_genome_v5_reference_genomes_bitwise(key):
    ac, genes, cost, per, hold = REF[key]
    profs, cfg = v5_profiles()
    P = sim.Profile.from_dict(profs[ac])
    scs = sim.make_scenarios(cfg["scenarios"], cfg["scenario_seed"], P)
    assert len(scs) == 4 and scs[3].draft_fps == P.disturbance_scenario["downdraft_fps"]
    rs = [sim.simulate(dict(zip(GENES, genes)), s, P) for s in scs]
    assert [r["cost"] for r in rs] == per                         # bit for bit, every scenario
    assert float(np.mean([r["cost"] for r in rs])) == cost
    assert [round(r["hold_osc"], 6) for r in rs] == hold


def test_v5_config_equals_genome_export_and_v4_plus_four_keys():
    ex = json.load(open(EXPORT5))
    v5 = json.load(open(os.path.join(PKG, "configs", "phase1_v5.json")))
    v4 = json.load(open(os.path.join(PKG, "configs", "phase1_hdg.json")))
    new = set(ex["_needs_code"]["profile_keys"])
    assert new == {"w_hold", "hold_ref_ft", "hold_settle_s", "disturbance_scenario"}
    for name, p in ex["profiles"].items():
        assert v5["profiles"][name] == p
        # v5 = v4 + 4 keys, except handoff section 6 (07:30 MST): v5-only ki_alt upper bound 0.5 (v4 keeps 0.05).
        # aircraft_root may be spelled repo-relative (sim.abs_root resolves it); compare resolved.
        norm = lambda d: {k: (sim.abs_root(v) if k == "aircraft_root" else v) for k, v in d.items()
                          if k not in new and not k.startswith("_")}
        a, b = norm(p), norm(v4["profiles"][name])
        assert a["gain_bounds"]["ki_alt"] == [b["gain_bounds"]["ki_alt"][0], 0.5] and b["gain_bounds"]["ki_alt"][1] == 0.05
        a["gain_bounds"] = {g: x for g, x in a["gain_bounds"].items() if g != "ki_alt"}
        b["gain_bounds"] = {g: x for g, x in b["gain_bounds"].items() if g != "ki_alt"}
        assert a == b
        P = sim.Profile.from_dict(dict(p, aircraft=name.split("_")[1]))
        assert P.aircraft_root == sim.abs_root("flight-dynamics/jsbsim_root") and os.path.isabs(P.aircraft_root)
    assert [a["name"] for a in v5["aircraft"]] == ["c172x", "T38", "737"]
    assert "w_hold" not in json.dumps(v4["profiles"])             # v4 (default) untouched


def test_flags_off_are_inert_and_v4_scenarios_unchanged():
    cfg = batch.resolve_config(json.load(open(os.path.join(PKG, "configs", "phase1_hdg.json"))), "v4")
    pd = next(a for a in cfg["aircraft"] if a["name"] == "c172x")["resolved_profile"]
    P = sim.Profile.from_dict(pd)
    assert P.w_hold == 0.0 and P.disturbance_scenario is None
    scs = sim.make_scenarios(3, 1, P)
    assert len(scs) == 3 and all("draft_fps" not in s.to_dict() for s in scs)   # v4 scenario dicts / cache keys as before
    g = dict(zip(GENES, REF["c172x_v4_on_v5"][1]))
    r = sim.simulate(g, scs[0], P)
    assert "hold_osc" not in r
    # hold term alone / downdraft alone bisect the v5 composition exactly (handoff section 3)
    p5 = sim.Profile.from_dict(dict(pd, w_hold=0.1))
    r5 = sim.simulate(g, scs[0], p5)
    assert r5["cost"] == r["cost"] + 0.1 * r5["hold_osc"]
    d = sim.Profile.from_dict(dict(pd, disturbance_scenario={"downdraft_fps": 4.69, "onset_t_s": 10.0, "onset_ramp_s": 4.0,
                                                               "steps_rel_ft": [[0.0, 0.0]]}))
    sd = sim.make_scenarios(3, 1, d)
    assert [x.to_dict() for x in sd[:3]] == [x.to_dict() for x in scs]          # no RNG draws consumed
    w = sd[3].vertical_gust_series(int(round(sd[3].duration_s / sim.DT)))
    assert w[int(10.0 / sim.DT)] == 0.0 and w[-1] == pytest.approx(4.69) and w[int(12.0 / sim.DT)] == pytest.approx(4.69 / 2)
    assert sd[3].to_dict()["draft_fps"] == 4.69 and sim.Scenario.from_dict(sd[3].to_dict()).to_dict() == sd[3].to_dict()


def test_v5_validation_and_cache_key_changes():
    profs, cfg = v5_profiles()
    pd = profs["c172x"]
    with pytest.raises(ValueError):
        sim.Profile.from_dict(dict(pd, w_hold=-1))
    with pytest.raises(ValueError):
        sim.Profile.from_dict(dict(pd, disturbance_scenario={"downdraft_fps": 4.0, "onset_ramp_s": 0.0}))
    g = np.full(8, 0.5)
    base = {k: v for k, v in pd.items() if k not in ("w_hold", "hold_ref_ft", "hold_settle_s", "disturbance_scenario")}
    base = sim.Profile.from_dict(base).to_dict()
    sc = sim.make_scenarios(1, 1, sim.Profile.from_dict(base))[0].to_dict()
    k0 = cache.eval_key("c172x", g, base, sc, 1, "1.3.1", "x")
    for ch in ({"w_hold": 0.1}, {"hold_ref_ft": 4.0}, {"hold_settle_s": 3.0},
               {"disturbance_scenario": pd["disturbance_scenario"]}):
        p2 = sim.Profile.from_dict(dict(base, **ch)).to_dict()
        assert cache.eval_key("c172x", g, p2, sc, 1, "1.3.1", "x") != k0
    d = dict(sc, draft_fps=4.69, draft_t_s=10.0, draft_ramp_s=4.0)
    assert cache.eval_key("c172x", g, base, d, 1, "1.3.1", "x") != k0


def test_run_json_carries_the_downdraft_scenario():
    profs, cfg = v5_profiles()
    ents = runinfo.scenario_entries("T38", profs["T38"], cfg["scenarios"], cfg["scenario_seed"])
    assert [e["id"] for e in ents] == ["T38:s0", "T38:s1", "T38:s2", "T38:s3"]
    assert ents[3]["draft_fps"] == 15.43 and ents[3]["draft_t_s"] == 10.0 and ents[3]["draft_ramp_s"] == 4.0
    assert ents[3]["wind_north_fps"] == 0.0 and ents[3]["gust_sigma_fps"] == 0.0
    fc = runinfo.fitness_cfg(profs["T38"])
    assert fc["w_hold"] == 0.1 and "hold_osc" in fc and "disturbance_doc" in fc
