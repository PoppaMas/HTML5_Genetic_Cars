"""On-disk evaluation cache (sqlite).

One row per *single-scenario simulation*. The key is a sha256 over a canonical
JSON of everything that can change that simulation's result:

    aircraft, genome bytes (float64, hex), resolved profile (IC, envelope, cost
    weights incl. comfort, ramp / feed-forward / roll gains, gain bounds and
    gene kinds, aircraft_root), the scenario dict (incl. its turbulence seed
    and ramp), the scenario-set seed, the JSBSim version, the evaluation code
    sha (sha256 of sim.py + genome.py + fidelity.py + eval.py), the fidelity and its
    model_version (FD code + wing params + coupler + prepared XML for flex) and a sha256 of every file in the
    aircraft/<model>/ directory that is actually loaded (so an in-place edit
    of a model, e.g. Flight Dynamics updating jsbsim_root, is a cache miss).

The GA seed is deliberately NOT in the key: it only decides *which* genomes get
evaluated, not what a given genome scores, so different GA seeds can share
results. Anything that does change a result (scenario seed, profile, code,
JSBSim) changes the key. Remaining blind spot: in-place edits to shared engine/ or
systems/ files outside the aircraft directory.

Only the main process touches the database (workers return results), so one
connection guarded by a lock is enough. WAL mode + per-generation commits make
it safe against SIGKILL (a killed run loses at most the uncommitted batch).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from typing import Dict, Iterable, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_CODE_FILES = ("sim.py", "genome.py", "fidelity.py", "eval.py")


def code_sha() -> str:
    h = hashlib.sha256()
    for f in EVAL_CODE_FILES:
        with open(os.path.join(_HERE, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read() + b"\0")
    return h.hexdigest()[:16]


def jsbsim_version() -> str:
    import jsbsim
    return jsbsim.__version__


def eval_key(aircraft: str, genome: np.ndarray, profile_d: Dict, scenario_d: Dict, scenario_seed: int,
             jsbsim_ver: str, code: str, model_sha: str = "", fidelity: Optional[str] = None,
             model_version: Optional[str] = None) -> str:
    payload = {
        "fidelity": fidelity,            # the fidelity + model_version the worker returned (checked by the batch)
        "model_version": model_version,
        "model_files": model_sha,  # sim.model_files_sha: content of aircraft/<model>/ in the root actually loaded
        "aircraft": aircraft,
        "genome": np.ascontiguousarray(genome, dtype=np.float64).tobytes().hex(),
        "profile": profile_d,
        "scenario": scenario_d,
        "scenario_seed": int(scenario_seed),
        "jsbsim": jsbsim_ver,
        "code_sha": code,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


class EvalCache:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._lock = threading.Lock()
        self._db = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False, timeout=60)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS evals (key TEXT PRIMARY KEY, aircraft TEXT, result TEXT, "
                             "created REAL DEFAULT (julianday('now')))")
            self._db.commit()

    @property
    def enabled(self) -> bool:
        return self._db is not None

    def get_many(self, keys: Iterable[str]) -> Dict[str, Dict]:
        keys = list(keys)
        if not self._db or not keys:
            return {}
        out = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = "SELECT key, result FROM evals WHERE key IN (%s)" % ",".join("?" * len(chunk))
                for k, r in self._db.execute(q, chunk):
                    out[k] = json.loads(r)
        return out

    def put_many(self, items: List[tuple]) -> None:
        """items: [(key, aircraft, result_dict)]"""
        if not self._db or not items:
            return
        with self._lock:
            self._db.executemany("INSERT OR IGNORE INTO evals (key, aircraft, result) VALUES (?, ?, ?)",
                                 [(k, a, json.dumps(r)) for k, a, r in items])
            self._db.commit()

    def count(self) -> int:
        if not self._db:
            return 0
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM evals").fetchone()[0]

    def close(self):
        if self._db:
            with self._lock:
                self._db.close()
                self._db = None
