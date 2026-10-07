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
