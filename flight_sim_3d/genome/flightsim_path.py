"""Locate and load the original flight_sim modules without modifying them.

The clone is found via $FLIGHT_SIM_DIR, else the repo's own flight_sim/ folder (../../flight_sim). If neither exists,
importing this module raises a FlightSimNotFound error telling you to set FLIGHT_SIM_DIR (the team's working layout
has no ../../flight_sim; there the test suite sets it in conftest.py, and run_evolve.py needs it exported). The
original ``sim`` and ``genome`` are loaded under private aliases
(``_orig_sim``, ``_orig_genome``) so that adapter shims can take the public
names ``sim`` / ``genome`` that evolve.py imports.
"""
from __future__ import annotations

import importlib.util
import os
import sys

# default: the repo's own flight_sim/ (this file is flight_sim_3d/genome/flightsim_path.py)
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flight_sim")


class FlightSimNotFound(ImportError):
    pass


def resolve_flight_sim_dir() -> str:
    env = os.environ.get("FLIGHT_SIM_DIR")
    if env:
        d = os.path.abspath(env)
        if not os.path.isfile(os.path.join(d, "sim.py")):
            raise FlightSimNotFound(f"FLIGHT_SIM_DIR={env!r} has no sim.py; set FLIGHT_SIM_DIR to the flight_sim folder "
                                    "(the one with sim.py, genome.py, ga.py, evolve.py)")
        return d
    if os.path.isfile(os.path.join(DEFAULT_DIR, "sim.py")):
        return os.path.abspath(DEFAULT_DIR)
    raise FlightSimNotFound(f"flight_sim not found: $FLIGHT_SIM_DIR is unset and the repo default {DEFAULT_DIR} does not "
                            "exist. Set FLIGHT_SIM_DIR to the flight_sim folder (the one with sim.py, genome.py, ga.py, "
                            "evolve.py), e.g. export FLIGHT_SIM_DIR=/path/to/HTML5_Genetic_Cars/flight_sim")


FLIGHT_SIM_DIR = resolve_flight_sim_dir()


def _load(alias: str, filename: str):
    if alias in sys.modules:
        return sys.modules[alias]
    path = os.path.join(FLIGHT_SIM_DIR, filename)
    if not os.path.exists(path):
        raise ImportError(f"{path} not found; set FLIGHT_SIM_DIR to the flight_sim folder")
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def orig_sim():
    return _load("_orig_sim", "sim.py")


def orig_genome():
    return _load("_orig_genome", "genome.py")


def orig_ga():
    return _load("_orig_ga", "ga.py")
