"""Path defaults for sim-bridge: repo-relative, every one overridable from the environment.

Layout assumed (both the local team tree and the phase-1 repo use it)::

    <team root>/                 FLIGHT_SIM_TEAM_ROOT   default: the parent of sim-bridge/
        sim-bridge/              (this package's parent)
        evolution/               SIMBRIDGE_EVOLUTION_ROOT (alias EVOLUTION_DIR)   default <team root>/evolution
            runs/                SIMBRIDGE_RUNS_ROOT                              default <evolution root>/runs
        flight-dynamics/         FLIGHT_DYNAMICS_DIR                              default <team root>/flight-dynamics
            v2_results/model_versions_post_mass.json   SIMBRIDGE_FD_MODEL_VERSIONS (FD's current model_version strings)
    <team root>/../flight_sim/   SIMBRIDGE_SANDBOX (alias FLIGHT_SIM_DIR)         default <team root>/../flight_sim

``evolution`` is imported as a package, so the directory that contains it (its parent) goes on sys.path; with a
non-default SIMBRIDGE_EVOLUTION_ROOT that parent is used instead of the team root.
Nothing here is absolute in the source; values are resolved once at import time.
"""
from __future__ import annotations

import os
import sys

SIM_BRIDGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(*names, default=None):
    for n in names:
        v = os.environ.get(n)
        if v:
            return os.path.abspath(os.path.expanduser(v))
    return default


TEAM_ROOT = _env("FLIGHT_SIM_TEAM_ROOT", default=os.path.dirname(SIM_BRIDGE))
EVOLUTION_ROOT = _env("SIMBRIDGE_EVOLUTION_ROOT", "EVOLUTION_DIR", default=os.path.join(TEAM_ROOT, "evolution"))
RUNS_ROOT = _env("SIMBRIDGE_RUNS_ROOT", default=os.path.join(EVOLUTION_ROOT, "runs"))
FLIGHT_DYNAMICS_DIR = _env("FLIGHT_DYNAMICS_DIR", default=os.path.join(TEAM_ROOT, "flight-dynamics"))
SANDBOX_DIR = _env("SIMBRIDGE_SANDBOX", "FLIGHT_SIM_DIR", default=os.path.join(os.path.dirname(TEAM_ROOT), "flight_sim"))
DATA_DIR = _env("SIMBRIDGE_DATA_DIR", default=os.path.join(SIM_BRIDGE, "data"))
FD_MODEL_VERSIONS = _env("SIMBRIDGE_FD_MODEL_VERSIONS",
                         default=os.path.join(FLIGHT_DYNAMICS_DIR, "v2_results", "model_versions_post_mass.json"))

ENV_VARS = {
    "FLIGHT_SIM_TEAM_ROOT": ("team root (contains evolution/, flight-dynamics/, sim-bridge/)", "parent of sim-bridge/"),
    "SIMBRIDGE_EVOLUTION_ROOT": ("Evolution Runner package dir (alias EVOLUTION_DIR)", "<team root>/evolution"),
    "SIMBRIDGE_RUNS_ROOT": ("ER runs dir (replay --run <id> looks here)", "<evolution root>/runs"),
    "FLIGHT_DYNAMICS_DIR": ("Flight Dynamics dir (relative aircraft_root in run.json resolves against the team root)",
                            "<team root>/flight-dynamics"),
    "SIMBRIDGE_SANDBOX": ("prototype flight_sim checkout used by trajlog.py (alias FLIGHT_SIM_DIR)", "<team root>/../flight_sim"),
    "SIMBRIDGE_DATA_DIR": ("sim-bridge outputs (replays, demo data)", "sim-bridge/data"),
    "SIMBRIDGE_FD_MODEL_VERSIONS": ("FD's published current model_version per aircraft and fidelity (replay checks "
                                    "replayed versions against it)", "<flight-dynamics>/v2_results/model_versions_post_mass.json"),
}


def ensure_import_paths() -> None:
    """Make ``import evolution`` and ``import sim_bridge`` work (evolution is imported read-only, no bytecode)."""
    sys.dont_write_bytecode = True
    for p in (os.path.dirname(EVOLUTION_ROOT), SIM_BRIDGE):
        if p not in sys.path:
            sys.path.insert(0, p)


def fd_model_versions() -> dict:
    """FD's published current model_version strings {aircraft: {rigid|reduced|full: str}} ({} if absent)."""
    import json
    try:
        with open(FD_MODEL_VERSIONS) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def resolve_model_root(path):
    """run.json aircraft_root -> a directory that exists here, plus a note when it had to be remapped.

    Relative paths (ER writes them relative to the team root since run.json gained paths_relative_to, e.g.
    "flight-dynamics/jsbsim_root") resolve against the team root; a "flight-dynamics/..." path follows
    FLIGHT_DYNAMICS_DIR (as evolution.sim.abs_root follows EVOLUTION_FD_DIR). Absolute paths written on another
    machine are remapped by their flight-dynamics/... tail onto FLIGHT_DYNAMICS_DIR. Returns (path, note|None)."""
    if not path:
        return path, None
    if not os.path.isabs(path):
        parts = os.path.normpath(path).split(os.sep)
        if parts and parts[0] == "flight-dynamics":
            return os.path.join(FLIGHT_DYNAMICS_DIR, *parts[1:]), f"relative {path!r} -> FLIGHT_DYNAMICS_DIR"
        p = os.path.normpath(os.path.join(TEAM_ROOT, path))
        return p, f"relative {path!r} -> team root"
    if os.path.exists(path):
        return path, None
    parts = os.path.normpath(path).split(os.sep)
    if "flight-dynamics" in parts:
        tail = parts[len(parts) - 1 - parts[::-1].index("flight-dynamics") + 1:]
        p = os.path.join(FLIGHT_DYNAMICS_DIR, *tail)
        if os.path.exists(p):
            return p, f"remapped {path!r} -> FLIGHT_DYNAMICS_DIR"
    return path, f"{path!r} does not exist here"


def describe() -> dict:
    return {"team_root": TEAM_ROOT, "evolution_root": EVOLUTION_ROOT, "runs_root": RUNS_ROOT,
            "flight_dynamics_dir": FLIGHT_DYNAMICS_DIR, "sandbox_dir": SANDBOX_DIR, "data_dir": DATA_DIR,
            "fd_model_versions": FD_MODEL_VERSIONS}
