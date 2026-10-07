"""Gene schema and decoding for the flight-controller GA.

Vendored from PoppaMas/HTML5_Genetic_Cars@flight-sim-prototype flight_sim/genome.py.
Changes: ``make_schema`` / the ``schema`` argument, so a per-aircraft profile
can override gain bounds and gene kinds; and the ``log0`` kind (Phase-1 shared
set, same definition as genome/genome_schema.py). With no overrides the default
SCHEMA is used unchanged.

Mirrors the car GA's design (src/app.js, ``createInstance.applyTypes``): every
gene is stored as a normalized float in [0, 1] and only mapped to a physical
value when an individual is evaluated. This keeps the GA operators independent
of units. Gains span orders of magnitude, so they use a ``log`` type:

    value = min * (max / min) ** v

``log0`` (integral gains) adds an exact zero: v <= zero_band (0.05) decodes to 0,
the rest of [0, 1] is the log range: min * (max/min) ** ((v - zb) / (1 - zb)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Gene:
    name: str
    min: float
    max: float
    kind: str = "log"  # "log", "linear" or "log0"
    units: str = ""
    doc: str = ""
    zero_band: float = 0.05  # log0 only
    default: Optional[float] = None  # baseline value (FD struct genes); used by seeded generation 0

    def decode(self, v: float) -> float:
        v = min(max(float(v), 0.0), 1.0)
        if self.kind == "log":
            return self.min * (self.max / self.min) ** v
        if self.kind == "linear":
            return self.min + v * (self.max - self.min)
        if self.kind == "log0":
            zb = self.zero_band
            if v <= zb:
                return 0.0
            return self.min * (self.max / self.min) ** ((v - zb) / (1.0 - zb))
        raise ValueError(f"unknown gene kind {self.kind!r}")

    def encode(self, value: float) -> float:
        if self.kind == "log":
            v = math.log(value / self.min) / math.log(self.max / self.min)
        elif self.kind == "log0":
            if value <= 0.0:
                return 0.0
            value = min(max(value, self.min), self.max)
            v = self.zero_band + (1.0 - self.zero_band) * math.log(value / self.min) / math.log(self.max / self.min)
        else:
            v = (value - self.min) / (self.max - self.min)
        return min(max(v, 0.0), 1.0)


# Altitude-hold autopilot: altitude error -> pitch command (outer PID, with
# vertical-speed damping as the D term) -> elevator (inner PID on pitch, with
# pitch-rate damping as the D term).
SCHEMA: List[Gene] = [
    Gene("kp_alt", 0.002, 0.5, units="deg/ft", doc="outer P: pitch cmd per ft of altitude error"),
    Gene("ki_alt", 1e-5, 0.05, units="deg/(ft*s)", doc="outer I: pitch cmd per ft*s of integrated altitude error"),
    Gene("kd_alt", 0.01, 3.0, units="deg/(ft/s)", doc="outer D: pitch cmd per ft/s of vertical speed"),
    Gene("kp_pitch", 0.002, 0.5, units="elev/deg", doc="inner P: normalized elevator per deg of pitch error"),
    Gene("ki_pitch", 1e-5, 0.2, units="elev/(deg*s)", doc="inner I: normalized elevator per deg*s of pitch error"),
    Gene("kd_pitch", 1e-4, 0.5, units="elev/(deg/s)", doc="inner D: normalized elevator per deg/s of pitch rate"),
]

GENE_NAMES = [g.name for g in SCHEMA]   # legacy default (the 6 pitch/altitude genes)
N_GENES = len(SCHEMA)

# optional heading-hold outer loop (genome/HANDOFF_heading_hold.md); appended after the 6 genes, in this order
HEADING_GENES: List[Gene] = [
    Gene("kp_hdg", 0.05, 5.0, units="deg bank/deg hdg", doc="outer heading P: bank cmd per deg heading error"),
    Gene("ki_hdg", 1e-5, 0.1, kind="log0", units="deg bank/(deg*s)", doc="outer heading I"),
]


KINDS = ("log", "linear", "log0")


def make_schema(bounds: Optional[Dict[str, Sequence[float]]] = None,
                kinds: Optional[Dict[str, str]] = None, heading_hold: bool = False) -> List[Gene]:
    """Default SCHEMA (+ HEADING_GENES when heading_hold) with optional per-gene [min, max] and kind overrides.
    Order: kp_alt, ki_alt, kd_alt, kp_pitch, ki_pitch, kd_pitch[, kp_hdg, ki_hdg] (genome/'s canonical layout)."""
    bounds = bounds or {}
    kinds = kinds or {}
    base = list(SCHEMA) + (list(HEADING_GENES) if heading_hold else [])
    names = [g.name for g in base]
    for what, d in (("gain_bounds", bounds), ("gene_kinds", kinds)):
        unknown = set(d) - set(names)
        if unknown:
            raise ValueError(f"unknown genes in {what}: {sorted(unknown)}" + ("" if heading_hold else
                             " (kp_hdg/ki_hdg need heading_hold)" if unknown & {"kp_hdg", "ki_hdg"} else ""))
    out = []
    for g in base:
        if g.name in bounds:
            lo, hi = (float(x) for x in bounds[g.name])
            if not (0 < lo < hi):
                raise ValueError(f"bad bounds for {g.name}: {bounds[g.name]}")
            g = replace(g, min=lo, max=hi)
        if g.name in kinds:
            if kinds[g.name] not in KINDS:
                raise ValueError(f"bad kind for {g.name}: {kinds[g.name]!r} (one of {KINDS})")
            g = replace(g, kind=kinds[g.name])
        out.append(g)
    return out


def decode(genome: Sequence[float], schema: Optional[List[Gene]] = None) -> Dict[str, float]:
    schema = SCHEMA if schema is None else schema
    if len(genome) != len(schema):
        raise ValueError(f"expected {len(schema)} genes, got {len(genome)}")
    return {g.name: g.decode(v) for g, v in zip(schema, genome)}


def encode(gains: Dict[str, float], schema: Optional[List[Gene]] = None) -> List[float]:
    schema = SCHEMA if schema is None else schema
    return [g.encode(gains[g.name]) for g in schema]
