"""Block-structured, task-dependent genome schema for the flight-controller GA.

Every gene is still stored as a normalized float in [0, 1] (the convention in
``flight_sim/genome.py`` and the car GA), so ``ga.py`` stays unit-agnostic.
What changes is *which* genes exist: genes live in named blocks, a task enables
a subset of blocks, and the genome is the concatenation of the enabled blocks'
genes in canonical order. Disabled blocks are held at their defaults.

Scales:
    log     value = min * (max/min) ** v                      (original behaviour)
    linear  value = min + v * (max - min)
    log0    v <= zero_band -> exactly 0; otherwise log over the remaining
            (v - zb)/(1 - zb). Lets the GA switch a term *off* (I or D terms)
            instead of pinning at a tiny log lower bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCALES = ("log", "linear", "log0")


@dataclass(frozen=True)
class GeneSpec:
    name: str
    block: str
    min: float
    max: float
    scale: str = "log"
    default: float = 0.0
    units: str = ""
    doc: str = ""
    # How the reference range is transformed for another aircraft (see profiles.py):
    # inner_pitch | inner_roll | inner_yaw | outer_vertical | outer_heading | speed | none
    scaling: str = "none"
    zero_band: float = 0.05
    provisional: bool = False

    def __post_init__(self):
        if self.scale not in SCALES:
            raise ValueError(f"{self.name}: unknown scale {self.scale!r}")
        if not self.max > self.min:
            raise ValueError(f"{self.name}: max must exceed min ({self.min}, {self.max})")
        if self.scale in ("log", "log0") and self.min <= 0:
            raise ValueError(f"{self.name}: log scales need min > 0")

    # Compatibility with flight_sim.genome.Gene (evolve.py reads .name/.units).
    @property
    def kind(self) -> str:
        return self.scale

    def decode(self, v: float) -> float:
        v = min(max(float(v), 0.0), 1.0)
        if self.scale == "log":
            return self.min * (self.max / self.min) ** v
        if self.scale == "linear":
            return self.min + v * (self.max - self.min)
        zb = self.zero_band
        if v <= zb:
            return 0.0
        return self.min * (self.max / self.min) ** ((v - zb) / (1.0 - zb))

    def encode(self, value: float) -> float:
        value = float(value)
        if self.scale == "log":
            v = math.log(value / self.min) / math.log(self.max / self.min)
        elif self.scale == "linear":
            v = (value - self.min) / (self.max - self.min)
        else:
            if value <= 0.0:
                return 0.0
            value = min(max(value, self.min), self.max)
            v = self.zero_band + (1.0 - self.zero_band) * math.log(value / self.min) / math.log(self.max / self.min)
        return min(max(v, 0.0), 1.0)

    def with_range(self, lo: float, hi: float, scale: Optional[str] = None) -> "GeneSpec":
        return replace(self, min=float(lo), max=float(hi), scale=scale or self.scale)

    def scaled(self, factor: float) -> "GeneSpec":
        """Multiply the physical range (and default) by ``factor``; normalized geometry is unchanged for log scales."""
        if factor == 1.0 or self.scaling == "none":
            return self
        return replace(self, min=self.min * factor, max=self.max * factor, default=self.default * factor)


def _g(name, block, lo, hi, scale="log", default=0.0, units="", doc="", scaling="none", provisional=False):
    return GeneSpec(name, block, lo, hi, scale, default, units, doc, scaling, provisional=provisional)


# Reference ranges are for the reference aircraft/condition (JSBSim c172x at
# 100 KCAS, 4000 ft -- the legacy task). profiles.py rescales them per aircraft.
# Defaults are the legacy hand-picked gains / the fixed helper gains in sim.py.
BLOCK_ORDER = ("pitch_altitude", "roll_heading", "yaw_damper", "speed_throttle", "structure")

BLOCKS: Dict[str, Tuple[GeneSpec, ...]] = {
    # Outer: altitude -> pitch cmd (PID, D on vertical speed). Inner: pitch -> elevator (PID, D on q).
    # Ranges re-centred vs legacy: kp_alt upper bound x4 (legacy pinned at 0.5),
    # ki_alt log0 so the integrator can be switched off (legacy pinned at 1e-5).
    "pitch_altitude": (
        _g("kp_alt", "pitch_altitude", 0.002, 2.0, "log", 0.05, "deg/ft", "outer P: pitch cmd per ft altitude error", "outer_vertical"),
        _g("ki_alt", "pitch_altitude", 1e-6, 0.05, "log0", 0.001, "deg/(ft*s)", "outer I", "outer_vertical"),
        _g("kd_alt", "pitch_altitude", 0.01, 3.0, "log", 0.3, "deg/(ft/s)", "outer D on vertical speed", "outer_vertical"),
        _g("kp_pitch", "pitch_altitude", 0.002, 0.5, "log", 0.05, "elev/deg", "inner P", "inner_pitch"),
        _g("ki_pitch", "pitch_altitude", 1e-5, 0.2, "log0", 0.01, "elev/(deg*s)", "inner I", "inner_pitch"),
        _g("kd_pitch", "pitch_altitude", 1e-4, 0.5, "log", 0.02, "elev/(deg/s)", "inner D on pitch rate", "inner_pitch"),
    ),
    # Outer: heading -> bank cmd (PI). Inner: bank -> aileron (PID, D on roll rate p).
    # Defaults kp_roll/kd_roll = the fixed wing leveler in sim.py (-0.05*phi - 0.02*p).
    "roll_heading": (
        _g("kp_hdg", "roll_heading", 0.05, 5.0, "log", 1.0, "deg bank/deg hdg", "outer P: bank cmd per deg heading error", "outer_heading"),
        _g("ki_hdg", "roll_heading", 1e-5, 0.1, "log0", 0.0, "deg bank/(deg*s)", "outer I", "outer_heading"),
        _g("kp_roll", "roll_heading", 0.002, 0.5, "log", 0.05, "ail/deg", "inner P: aileron per deg bank error", "inner_roll"),
        _g("ki_roll", "roll_heading", 1e-5, 0.2, "log0", 0.0, "ail/(deg*s)", "inner I", "inner_roll"),
        _g("kd_roll", "roll_heading", 1e-4, 0.5, "log", 0.02, "ail/(deg/s)", "inner D on roll rate", "inner_roll"),
    ),
    # Yaw damper on washed-out yaw rate + sideslip feedback. Defaults 0 = sim.py's rudder held at 0.
    "yaw_damper": (
        _g("kr_yaw", "yaw_damper", 1e-4, 0.5, "log0", 0.0, "rud/(deg/s)", "rudder per deg/s washed-out yaw rate", "inner_yaw"),
        _g("kbeta_yaw", "yaw_damper", 1e-4, 0.5, "log0", 0.0, "rud/deg", "rudder per deg sideslip", "inner_yaw"),
        _g("tau_washout", "yaw_damper", 0.3, 10.0, "log", 2.0, "s", "yaw-rate washout time constant", "none"),
    ),
    # Airspeed -> throttle (PI + acceleration damper). Defaults = fixed helper in sim.py (0.05, 0.01, 0).
    "speed_throttle": (
        _g("kp_spd", "speed_throttle", 0.002, 0.5, "log", 0.05, "thr/kt", "P: throttle per kt airspeed error", "speed"),
        _g("ki_spd", "speed_throttle", 1e-4, 0.1, "log0", 0.01, "thr/(kt*s)", "I", "speed"),
        _g("kd_spd", "speed_throttle", 1e-4, 0.5, "log0", 0.0, "thr/(kt/s)", "D on airspeed rate", "speed"),
    ),
    # Flex-wing parameters: FD's final Phase-1 interface (flight-dynamics/INTERFACE.md section 0/1a, 2026-10-06 04:05 PT;
    # = flexwing.STRUCT_SCHEMA). Consumed by FD's flexwing/coupled_sim only when the task enables flex mode.
    # Bending and torsion stiffness are TIED: EI x = stiffness_scale, GJ x = stiffness_scale * torsion_bend_ratio
    # (flexwing.tied_stiffness), keeping the GA off the swept-wing torsion/2nd-bending ridge.
    # NOT genes in Phase 1 (FD's model raises if they are passed from a genome): elastic_axis_frac / section_cg_frac
    # (fixed per aircraft, CG aft of EA: c172x 0.38/0.42, T38 0.40/0.42, 737 0.36/0.38), tip_mass_frac (fixed 0),
    # bend_/torsion_stiffness_scale (replaced by the tied pair), mass_centroid_shift (dropped),
    # aspect_ratio_delta / sweep_delta_deg (not modelled, fixed 0 -- see STRUCTURE_FIXED_ZERO).
    "structure": (
        _g("stiffness_scale", "structure", 0.6, 2.0, "log", 1.0, "x EI", "wing stiffness multiplier: EI x s (also scales structural mass and the allowable root moment)"),
        _g("torsion_bend_ratio", "structure", 0.8, 1.15, "linear", 1.0, "GJx/EIx", "torsion-to-bending stiffness ratio: GJ x = s * r"),
        _g("struct_damping_ratio", "structure", 0.005, 0.05, "log", 0.02, "-", "modal structural damping ratio (all modes)"),
        _g("nonstructural_mass_scale", "structure", 0.8, 1.25, "log", 1.0, "x m_nonstruct", "non-structural share of wing mass (fuel/systems)"),
    ),
}

# Exact copy of flight_sim/genome.py SCHEMA ranges (verified against the original in tests).
LEGACY_PITCH_RANGES = {
    "kp_alt": (0.002, 0.5, "log"),
    "ki_alt": (1e-5, 0.05, "log"),
    "kd_alt": (0.01, 3.0, "log"),
    "kp_pitch": (0.002, 0.5, "log"),
    "ki_pitch": (1e-5, 0.2, "log"),
    "kd_pitch": (1e-4, 0.5, "log"),
}

# Heading hold (Phase 1, shared-set flag "heading_hold"): only the OUTER heading loop of the roll_heading block is
# evolved -- heading error -> bank command (clamped to a per-aircraft bank limit) -> the existing wing leveller, whose
# kp_roll/kd_roll stay at their profile-scaled defaults (= sim.py's -0.05*phi - 0.02*p on the c172x). Same GeneSpecs as
# the full block (no duplicate genes); a task enables the subset with "blocks": {"roll_heading": ["kp_hdg", "ki_hdg"]}.
HEADING_HOLD_GENES = ("kp_hdg", "ki_hdg")

# Phase-1 fixed (non-gene) structure inputs, kept explicit so a future task can't silently assume otherwise.
STRUCTURE_FIXED_ZERO = {"aspect_ratio_delta": 0.0, "sweep_delta_deg": 0.0}

ALL_GENES: Dict[str, GeneSpec] = {g.name: g for b in BLOCK_ORDER for g in BLOCKS[b]}


@dataclass
class GenomeSpec:
    """Concrete, task-specific genome: enabled genes (in order) + fixed values for the rest."""
    genes: List[GeneSpec]
    fixed: Dict[str, float] = field(default_factory=dict)  # disabled-block genes at their defaults
    enabled_blocks: Tuple[str, ...] = ()
    notes: List[str] = field(default_factory=list)

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def names(self) -> List[str]:
        return [g.name for g in self.genes]

    def layout(self) -> Dict[str, slice]:
        out, i = {}, 0
        for b in self.enabled_blocks:
            n = sum(1 for g in self.genes if g.block == b)
            out[b] = slice(i, i + n)
            i += n
        return out

    def decode(self, genome: Sequence[float], include_fixed: bool = False) -> Dict[str, float]:
        if len(genome) != self.n_genes:
            raise ValueError(f"expected {self.n_genes} genes, got {len(genome)}")
        d = {g.name: g.decode(v) for g, v in zip(self.genes, genome)}
        if include_fixed:
            return {**self.fixed, **d}
        return d

    def encode(self, values: Mapping[str, float]) -> List[float]:
        return [g.encode(values[g.name] if g.name in values else g.default) for g in self.genes]

    def default_genome(self) -> np.ndarray:
        return np.array([g.encode(g.default) for g in self.genes])

    def transfer(self, genome: Sequence[float], from_spec: "GenomeSpec") -> np.ndarray:
        """Re-encode a genome from another spec (e.g. seed a 3-block task from a 1-block result)."""
        phys = from_spec.decode(genome, include_fixed=True)
        return np.array(self.encode(phys))

    def at_bounds(self, genomes, eps: float = 0.01, near: float = 0.1) -> List[Dict]:
        """Report genes sitting at (or within eps of) a normalized bound.

        ``genomes`` may be one genome or a 2-D array (e.g. the final population
        or its elite). For log0 genes the zero band is reported as 'zeroed'
        (the term is switched off), not as pinned. Genes within ``near`` (but
        not ``eps``) of a bound are reported as near_lower / near_upper.
        """
        G = np.atleast_2d(np.asarray(genomes, dtype=float))
        rep = []
        for j, g in enumerate(self.genes):
            col = G[:, j]
            lo_hit = col <= eps if g.scale != "log0" else np.zeros_like(col, bool)
            zeroed = col <= g.zero_band if g.scale == "log0" else np.zeros_like(col, bool)
            hi_hit = col >= 1.0 - eps
            lo_edge = g.zero_band if g.scale == "log0" else 0.0
            near_lo = (col > lo_edge + eps) & (col <= lo_edge + near) if g.scale != "log0" else (col > lo_edge) & (col <= lo_edge + near)
            near_hi = (col < 1.0 - eps) & (col >= 1.0 - near)
            for kind, mask in (("lower", lo_hit), ("upper", hi_hit), ("zeroed", zeroed),
                               ("near_lower", near_lo), ("near_upper", near_hi)):
                frac = float(mask.mean())
                if frac > 0:
                    rep.append({"gene": g.name, "block": g.block, "bound": kind, "fraction": frac,
                                "value": g.decode(float(np.median(col[mask]))),
                                "range": [g.min, g.max], "scale": g.scale})
        return rep


def suggest_widened_range(g: GeneSpec, bound: str, factor: float = 4.0) -> Tuple[float, float]:
    """Suggested new physical range for a gene pinned at ``bound`` (not applied automatically)."""
    if g.scale == "linear":
        span = g.max - g.min
        return (g.min - span * (factor - 1) / 2, g.max) if bound == "lower" else (g.min, g.max + span * (factor - 1) / 2)
    return (g.min / factor, g.max) if bound == "lower" else (g.min, g.max * factor)


def build_spec(enabled: Iterable[str],
               range_factors: Optional[Mapping[str, float]] = None,
               overrides: Optional[Mapping[str, Mapping]] = None,
               legacy_pitch_ranges: bool = False,
               gene_subsets: Optional[Mapping[str, Sequence[str]]] = None) -> GenomeSpec:
    """Assemble a GenomeSpec.

    enabled             blocks to evolve (canonical order is enforced, so the layout is stable)
    range_factors       {scaling_tag: factor} from an AircraftProfile (profiles.range_factors)
    overrides           {gene: {"min":..,"max":..,"scale":..,"default":..}} applied last, verbatim
    legacy_pitch_ranges use flight_sim/genome.py's original ranges for the pitch block (legacy preset)
    gene_subsets        {block: [gene, ...]}: evolve only these genes of an enabled block; its other genes are held
                        at their (scaled) defaults like a disabled block (e.g. heading hold = outer loop only)
    """
    enabled = list(enabled)
    unknown = set(enabled) - set(BLOCKS)
    if unknown:
        raise ValueError(f"unknown blocks {sorted(unknown)}")
    enabled = [b for b in BLOCK_ORDER if b in set(enabled)]
    range_factors = dict(range_factors or {})
    overrides = dict(overrides or {})
    bad = set(overrides) - set(ALL_GENES)
    if bad:
        raise ValueError(f"overrides for unknown genes {sorted(bad)}")
    subsets = {b: tuple(v) for b, v in (gene_subsets or {}).items()}
    for b, names in subsets.items():
        if b not in BLOCKS:
            raise ValueError(f"gene subset for unknown block {b!r}")
        wrong = set(names) - {g.name for g in BLOCKS[b]}
        if wrong or not names:
            raise ValueError(f"gene subset for block {b!r}: {sorted(wrong) or 'empty'} not genes of that block")
        if b not in enabled:
            enabled.append(b)
    enabled = [b for b in BLOCK_ORDER if b in set(enabled)]
    genes, fixed, notes = [], {}, []
    for b in BLOCK_ORDER:
        for g in BLOCKS[b]:
            if legacy_pitch_ranges and g.name in LEGACY_PITCH_RANGES:
                lo, hi, sc = LEGACY_PITCH_RANGES[g.name]
                g = g.with_range(lo, hi, sc)
            else:
                g = g.scaled(range_factors.get(g.scaling, 1.0))
            if g.name in overrides:
                o = overrides[g.name]
                g = replace(g, min=float(o.get("min", g.min)), max=float(o.get("max", g.max)),
                            scale=o.get("scale", g.scale), default=float(o.get("default", g.default)))
                notes.append(f"override {g.name}: [{g.min:g}, {g.max:g}] {g.scale}")
            if b in enabled and (b not in subsets or g.name in subsets[b]):
                genes.append(g)
            else:
                fixed[g.name] = g.default
    if any(g.block == "structure" for g in genes):
        notes.append("structure block: FD flex-wing parameters (notional structural data, see FD INTERFACE.md)")
    return GenomeSpec(genes, fixed, tuple(enabled), notes)
