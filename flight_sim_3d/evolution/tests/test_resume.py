"""Kill a batch with SIGKILL mid-run, resume it, and compare with an uninterrupted run."""
import json
import os
import signal
import subprocess
import sys
import time

from .conftest import ROOT, load_run, make_cfg, make_user, run_batch


def _gen_next(run_dir):
    out = {}
    d = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(d, f)) as fh:
                        out[f] = json.load(fh)["gen_next"]
                except (json.JSONDecodeError, OSError):
                    pass
    return out


def test_kill_and_resume_matches_uninterrupted(tmp_path):
    # reference: uninterrupted, no cache
    ref_cfg = make_cfg(tmp_path, "ref", cache=False, ga={"pop_size": 16, "generations": 8})
    _, ref_dir = run_batch(ref_cfg)

    # interrupted: separate cache so nothing is shared with the reference
    user = make_user(tmp_path, "killed", cache=True, cache_name="killed.sqlite", ga={"pop_size": 16, "generations": 8})
    cfg = make_cfg(tmp_path, "killed", cache=True, cache_name="killed.sqlite", ga={"pop_size": 16, "generations": 8})
    cfg_path = tmp_path / "killed.json"
    cfg_path.write_text(json.dumps(user))
    run_dir = os.path.join(cfg["runs_dir"], "killed")
    cmd = [sys.executable, "-m", "evolution.batch", "--config", str(cfg_path)]
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 120:
        g = _gen_next(run_dir)
        if g and max(g.values()) >= 3:
            break
        time.sleep(0.02)
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    killed_at = _gen_next(run_dir)
    assert killed_at and min(killed_at.values()) < 8, f"run finished before it could be killed: {killed_at}"
    assert not os.path.exists(os.path.join(run_dir, "summary.json"))

    # resume (same command would also do it; use --resume to exercise that path)
    r = subprocess.run([sys.executable, "-m", "evolution.batch", "--resume", run_dir], cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "resuming at generation" in r.stdout
    assert load_run(run_dir) == load_run(ref_dir)
    # history.jsonl: exactly one line per (aircraft, generation), sessions recorded
    lines = [json.loads(l) for l in open(os.path.join(run_dir, "history.jsonl"))]
    keys = [(h["aircraft"], h["generation"]) for h in lines]
    assert len(keys) == len(set(keys)) == 2 * 8
    assert {h["session"] for h in lines} == {0, 1}
    print("killed at", killed_at)
