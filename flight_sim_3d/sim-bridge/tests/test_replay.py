"""Replay tests. Run:  PYTHONDONTWRITEBYTECODE=1 <venv>/bin/python -m pytest -q tests/test_replay.py
(or plain:            PYTHONDONTWRITEBYTECODE=1 <venv>/bin/python tests/test_replay.py)
Needs ER's sandbox venv (jsbsim + numpy). Never writes into ER's tree (outputs go to a temp dir).

1. adapter path on ER's legacy artifacts (config.json + checkpoints + evolution.sim): costs exact, trajectories
   bit-identical to ER's own traj files.
2. agreed real-interface path (run.json + genomes.jsonl + evaluate with post-step recorder) using the fixture
   tests/fixtures/er_interface/phase1-s1 + tests/fixtures/fake_evolution_eval.py, with and without the t=0 call.
3. ER's REAL evolution.eval + runs/<run>/run.json: runs automatically once ER ships them, skipped until then.
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.dirname(HERE)
TEAM = os.path.dirname(SB)
RUNS = os.path.join(TEAM, "evolution", "runs")
PY = sys.executable


def _replay(args, env_extra=None):
    out = tempfile.mkdtemp(prefix="replay_test_")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env_extra or {}))
    p = subprocess.run([PY, os.path.join(SB, "replay.py"), *args, "--out", out], cwd=SB, env=env,
                       capture_output=True, text=True, timeout=900)
    man_p = os.path.join(out, "replay_manifest.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else None
    if man is not None:
        man["_out"] = out
    return p, man


def _assert_exact(p, man, n_rows=2701, min_files=1):
    assert p.returncode == 0, p.stdout + p.stderr
    assert man and not man["flagged"]
    assert all(g["verdict"] == "match" and g["max_rel_err"] == 0.0 for g in man["genomes"]), man["summary"]
    tc = man["trajectory_check"]
    assert tc and tc["n_files"] >= min_files and set(tc["summary"]) == {"match"}, tc and tc["summary"]
    for f in tc["files"].values():
        assert f["bit_identical_channels"] == f["n_channels"], f
        assert f["rows_compared"] == n_rows, f


def test_adapter_phase1_s1_exact():
    _need_traj("phase1-s1")
    p, man = _replay(["--run", "phase1-s1", "--gens", "0,19", "--aircraft", "T38,c172x"])
    _assert_exact(p, man, min_files=4)
    assert man["interface"] == "adapter" and man["recorder_timing"] == "pre"


def test_adapter_legacy_bench_exact():
    _need_traj("bench_jets-j1")
    p, man = _replay(["--run", "bench_jets-j1", "--gens", "19", "--aircraft", "f16"])
    _assert_exact(p, man)


def _need_traj(run):
    """The repo ships run records (config.json, summary.json, checkpoints/) but not ER's bulky trajectory JSON dumps."""
    d = os.path.join(RUNS, run, "trajectories")
    if not os.path.isdir(d):
        msg = (f"reference trajectories {d} not present: ER's recorded trajectory JSON dumps (~650 KB each) are not "
               f"shipped in the repo; copy the original evolution/runs/{run}/trajectories/ into place to run this check")
        try:
            import pytest
        except ImportError:
            raise RuntimeError("SKIP " + msg)
        pytest.skip(msg)


FIX_ROOT = os.path.join(HERE, "fixtures", "er_interface")
FAKE = os.path.join(HERE, "fixtures", "fake_evolution_eval.py")
REF = os.path.join(RUNS, "phase1-s1", "trajectories")


def test_er_interface_fixture_post_step():
    _need_traj("phase1-s1")
    p, man = _replay(["--run", "phase1-s1", "--runs-root", FIX_ROOT, "--interface", "er", "--eval-module", FAKE,
                      "--gens", "9,19", "--aircraft", "737,f16", "--compare-traj", REF])
    _assert_exact(p, man, n_rows=2700, min_files=4)  # no t=0 call -> t=0 row absent, everything else identical
    assert man["interface"] == "er" and man["recorder_timing"] == "post"


def test_er_interface_fixture_with_t0_call():
    _need_traj("phase1-s1")
    p, man = _replay(["--run", "phase1-s1", "--runs-root", FIX_ROOT, "--interface", "er", "--eval-module", FAKE,
                      "--gens", "19", "--aircraft", "c172x", "--compare-traj", REF], {"FAKE_EVAL_T0": "1"})
    _assert_exact(p, man, n_rows=2701)


def test_er_interface_flex_channels_plumbing():
    p, man = _replay(["--run", "phase1-s1", "--runs-root", FIX_ROOT, "--interface", "er", "--eval-module", FAKE,
                      "--ids", "T38:g19:r0", "--scenario", "0", "--compare-traj", "none"],
                     {"FAKE_EVAL_T0": "1", "FAKE_EVAL_SYNTH_FLEX": "1"})
    assert p.returncode == 0, p.stdout + p.stderr
    docs = [json.load(open(f)) for f in glob.glob(os.path.join(man["_out"], "trajectories", "traj_T38_*.json"))]
    assert docs and docs[0].get("structure"), "no trajectory with a structure block"
    d = docs[0]
    assert d["structure"]["synthetic"] is True and "wingL.dz.2" in d["channels"] and len(d["channels"]) == 29 + 12


def test_fidelity_full_needs_real_eval():
    p, man = _replay(["--run", "phase1-s1", "--ids", "f16:g19:r0", "--scenario", "0", "--fidelity", "full"])
    assert p.returncode == 3 and "fidelity 'full'" in p.stdout + p.stderr


def _real_er():
    has_eval = os.path.exists(os.path.join(TEAM, "evolution", "eval.py")) or os.path.isdir(os.path.join(TEAM, "evolution", "eval"))
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(RUNS, "*", "run.json"))
                  if os.path.exists(os.path.join(os.path.dirname(p), "genomes.jsonl")))
    return has_eval, runs


def test_real_er_interface_when_available():
    has_eval, runs = _real_er()
    if not (has_eval and runs):
        msg = f"ER real interface not shipped yet (evolution/eval.py: {has_eval}, runs with run.json+genomes.jsonl: {runs})"
        try:
            import pytest
            pytest.skip(msg)
        except ImportError:
            print("SKIP", msg)
            return
    for run in runs:
        p, man = _replay(["--run", run, "--interface", "er", "--best-per-gen"])
        assert p.returncode == 0, f"{run}: " + p.stdout[-3000:] + p.stderr[-3000:]
        assert man["interface"] == "er" and not man["flagged"], man["summary"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            print("PASS", name)
        except BaseException as e:  # noqa: BLE001
            if type(e).__name__ == "Skipped":
                print("SKIP", name, e)
                continue
            fails += 1
            print("FAIL", name, type(e).__name__, str(e)[:2000])
    sys.exit(1 if fails else 0)
