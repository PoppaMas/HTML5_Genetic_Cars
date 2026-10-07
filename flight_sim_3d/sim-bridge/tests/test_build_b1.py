"""tools/build_b1_page.py helpers (no ER run needed): frozen-FD discovery from run.json, proof verdict rules."""
import json
import os
import sys

import pytest

SB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SB, "tools"))
sys.path.insert(0, SB)
import build_b1_page as B  # noqa: E402


def test_fd_dir_from_run_json(tmp_path):
    fd = tmp_path / "frozen_fd"
    fd.mkdir()
    R = {"dir": str(tmp_path), "run": {"pin_model_version": {"737": {"full_a1_b1": "x"}},
                                         "fd": {"frozen_dir": str(fd)}, "paths_relative_to": "team root"}}
    c = B.fd_dir_candidates(R, None)
    assert c and c[0] == ("run.json.fd.frozen_dir", str(fd))
    assert B.fd_dir_candidates(R, str(tmp_path))[0][0] == "--fd-dir"


def _manifest(tmp_path, rel=0.0, pinned=True, failed=()):
    d = tmp_path / "proof"
    d.mkdir()
    m = {"wall_s": 1.0, "genomes": [{"aircraft": "737", "generation": 4, "logged_cost": 0.27, "replayed_cost": 0.27,
                                     "rel_err": rel, "per_scenario": {"737:s0": {"rel_err": rel}},
                                     "pinned_model_version_match": pinned, "replay_model_version": "v"}],
         "trajectory_check": {"files": {"k": {"reference_file": "traj_737_x_g4.json", "bit_identical_channels": 10,
                                              "n_channels": 10 + len(failed), "failed_channels": list(failed),
                                              "verdict": "match", "rows_compared": 5, "rows_reference": 5}}}}
    (d / "replay_manifest.json").write_text(json.dumps(m))
    return str(d)


def test_proof_verdict_rules(tmp_path):
    assert B.proof_verdict(_manifest(tmp_path), 0)["ok"]
    for kw in ({"rel": 1e-9}, {"pinned": False}, {"failed": ["theta"]}):
        sub = tmp_path / str(len(os.listdir(tmp_path)))
        sub.mkdir()
        assert not B.proof_verdict(_manifest(sub, **kw), 0)["ok"], kw
    sub = tmp_path / "wingonly"
    sub.mkdir()
    v = B.proof_verdict(_manifest(sub, failed=["wingR.dz.3"]), 0)
    assert v["ok"] and v["files"][0]["failed_wing"] == 1   # wing channels reported, not fatal
    assert not B.proof_verdict(str(tmp_path / "missing"), 3)["ok"]


# ---------------------------------------------------------------- A/B (baseline vs tweaked) and pilot gen numbers
def test_ab_names_and_families():
    a, b = "phase3b1r1-pilot-s1", "phase3b1r1-pilot-tweaked-s1"
    assert B.run_family(a) == "phase3b1r1-pilot" and B.run_family(b) == "phase3b1r1-pilot-tweaked"
    assert B.ab_tags([a, b]) == ["base", "tweaked"]
    assert B.ab_tags(["x-s1", "x-s2"]) == ["s1", "s2"]
    assert B.ab_tags(["same", "same"]) == ["same", "same"]          # not distinct -> full ids


def _R(rid, gens, acs=("737", "T38", "c172x")):
    return {"id": rid, "entries": [{"aircraft": ac, "generation": g, "fitness": 1.0 / (g + 1), "file": f"traj_{ac}_{rid}_g{g}.json"}
                                   for ac in acs for g in gens]}


def test_common_gens_matching_only():
    base, tw = _R("p-s1", [0, 29, 59]), _R("p-tweaked-s1", [0, 30, 59])
    assert B.common_gens([base, tw]) == [0, 59]
    assert B.common_gens([base, tw], "T38") == [0, 59]
    assert B.common_gens([base, _R("q-s1", [5])]) == []
    assert B.gens_of(base, "737") == [0, 29, 59]                   # pilot numbering (not 0/2/4)


def test_parse_shot_report_ignores_trailing_warnings():
    out = 'noise\n{"a": {"png": "/x/p_a.png", "render": {"covered": [["tipR"]]}}, "console": []}\n' \
          'AIRCRAFT UNDER AN OVERLAY (warning): {"a": [["tipR"]]}\nRESULT: PASS\n'
    rep = B.parse_shot_report(out)
    assert rep["a"]["png"] == "/x/p_a.png" and rep["console"] == []
    assert B.parse_shot_report("no json") == {}
