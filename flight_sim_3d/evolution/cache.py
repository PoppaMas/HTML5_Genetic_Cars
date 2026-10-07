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

full vs full_a1 (P3-A1): both fly the same prepared <root>_v2 files, so model_files alone would not separate them; the
key carries "fidelity" ('full' / 'full_a1') AND FD's model_version ('full:flexv2:..' / 'full_a1:flexv2a1:..'), so the two
key spaces are disjoint, and eval_key refuses a model_version whose prefix is not "<fidelity>:" (a full string can never
be filed under full_a1 or vice versa). The payload itself is unchanged, so rigid / reduced / full keys are as before.

full_a1 vs full_a1_b1 (P3-B1): same mechanism ('full_a1_b1:' prefix, its own model_version), and a full_a1_b1 key also
carries FD's planform_b1.shape_cache_key of the decoded shape ("shape_key"), so two shapes never share an entry even if
the rest of the genome bytes were equal. The "shape_key" field is added ONLY when given, so rigid / reduced / full /
full_a1 payloads (and keys) are byte-identical to before.

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
             model_version: Optional[str] = None, shape_key: Optional[str] = None) -> str:
    if fidelity is not None and model_version is not None and not str(model_version).startswith(f"{fidelity}:"):
        raise ValueError(f"cache key: model_version {model_version!r} does not belong to fidelity {fidelity!r}")
    if shape_key is not None and fidelity != "full_a1_b1":
        raise ValueError(f"cache key: shape_key is only part of full_a1_b1 keys (fidelity {fidelity!r})")
    if fidelity == "full_a1_b1" and not shape_key:
        raise ValueError("cache key: a full_a1_b1 key needs FD's shape_cache_key (planform_b1.shape_cache_key)")
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
    if shape_key is not None:        # full_a1_b1 only (keeps every other payload byte-identical)
        payload["shape_key"] = str(shape_key)
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


class EvalCache:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._lock = threading.Lock()
        self._db = None
        self.pins: Dict[str, Dict[str, str]] = {}   # {aircraft: {fidelity: model_version}} (config pin_model_version)
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
        for k, a, r in items:   # second line of defence behind batch.check_pins: never store an unpinned model's result
            pin = self.pins.get(a)
            fid = r.get("fidelity", "rigid")
            if pin is not None and fid != "rigid" and pin.get(fid) != r.get("model_version"):
                raise RuntimeError(f"cache: refusing to store {a} {fid} result with model_version "
                                   f"{r.get('model_version')!r} (pinned {pin.get(fid)!r})")
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
