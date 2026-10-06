"""Gene schema and decoding for the flight-controller GA.

Mirrors the car GA's design (src/app.js, ``createInstance.applyTypes``): every
gene is stored as a normalized float in [0, 1] and only mapped to a physical
value when an individual is evaluated. This keeps the GA operators independent
of units. Gains span orders of magnitude, so they use a ``log`` type:

    value = min * (max / min) ** v
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Gene:
    name: str
    min: float
    max: float
    kind: str = "log"  # "log" or "linear"
    units: str = ""
    doc: str = ""

    def decode(self, v: float) -> float:
        v = min(max(float(v), 0.0), 1.0)
        if self.kind == "log":
            return self.min * (self.max / self.min) ** v
        if self.kind == "linear":
            return self.min + v * (self.max - self.min)
        raise ValueError(f"unknown gene kind {self.kind!r}")

    def encode(self, value: float) -> float:
        if self.kind == "log":
            v = math.log(value / self.min) / math.log(self.max / self.min)
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

GENE_NAMES = [g.name for g in SCHEMA]
N_GENES = len(SCHEMA)


def decode(genome: Sequence[float]) -> Dict[str, float]:
    if len(genome) != N_GENES:
        raise ValueError(f"expected {N_GENES} genes, got {len(genome)}")
    return {g.name: g.decode(v) for g, v in zip(SCHEMA, genome)}


def encode(gains: Dict[str, float]) -> List[float]:
    return [g.encode(gains[g.name]) for g in SCHEMA]
