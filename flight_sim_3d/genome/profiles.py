"""AircraftProfile: per-aircraft data and the rule that derives gene ranges from it.

Range-derivation rule (first-order, documented in DESIGN.md section 3):
reference ranges in genome_schema.BLOCKS are for the reference profile
(c172x at its design point, i.e. the legacy task condition). For another
aircraft each gene's range is multiplied by a factor chosen by the gene's
``scaling`` tag, so that the *closed-loop effect* of a normalized gene value
is roughly the same across aircraft:

  inner_pitch     CP_ref / CP, CP = qbar*S*cbar*|Cm_de|*de_max / Iyy   (pitch accel per unit elevator cmd;
                  if the profile gives cm_de_per_norm -- JSBSim models written per normalized
                  deflection, e.g. T38 -- then |Cm_de|*de_max is replaced by |cm_de_per_norm|)
  inner_roll      CP_ref / CP, CP = qbar*S*b*|Cl_da|*da_max / Ixx
  inner_yaw       CP_ref / CP, CP = qbar*S*b*|Cn_dr|*dr_max / Izz
  outer_vertical  V_ref / V    (TAS; gamma = hdot / V, so deg pitch per ft/s scales with 1/V)
  outer_heading   V / V_ref    (turn rate = g tan(phi) / V, so bank per deg heading error scales with V)
  speed           (m/T)/(m_ref/T_ref)  (dV/dt per unit throttle = T_max / m)
  none            1

qbar is taken at the profile's design point (KCAS ~ EAS, so qbar = 0.5*rho0*Vc^2,
compressibility ignored). This is a scheduling-free, single-point estimate:
good enough to place a search range, not a gain schedule. Explicit
``gain_overrides`` in the profile (or task) always win.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(HERE, "aircraft_profiles")
REFERENCE_PROFILE = "c172x"

KT_TO_FPS = 1.6878098571
RHO0 = 0.0023769  # slug/ft^3, ISA sea level

# Generic control-derivative magnitudes used only when a profile omits them (flagged in .warnings).
GENERIC_DERIVATIVES = {"cm_de_per_rad": 1.2, "cl_da_per_rad": 0.15, "cn_dr_per_rad": 0.08}


def isa_density(alt_ft: float) -> float:
    """ISA density (slug/ft^3), troposphere + lower stratosphere (to ~65 kft)."""
    h_m = alt_ft * 0.3048
    if h_m <= 11000.0:
        T = 288.15 - 0.0065 * h_m
        p = 101325.0 * (T / 288.15) ** 5.25588
    else:
        T = 216.65
        p = 22632.06 * math.exp(-9.80665 * (h_m - 11000.0) / (287.053 * T))
    rho_si = p / (287.053 * T)
    return rho_si / 515.379


@dataclass
class AircraftProfile:
    name: str
    description: str
    jsbsim_model: Optional[str]
    mass_lb: float                     # representative flying weight at design point
    ixx: float                         # slug*ft^2
    iyy: float
    izz: float
    wing_area_ft2: float
    span_ft: float
    chord_ft: float                    # mean aerodynamic chord
    vstall_kcas: float                 # clean, at mass_lb
    vcruise_ktas: float
    vne_kcas: float
    n_limits: Tuple[float, float]      # (neg, pos) limit load factors
    design_point: Dict[str, float]     # {"kcas":..., "alt_ft":...} where ranges are derived
    control_limits_deg: Dict[str, float]  # {"elevator":..,"aileron":..,"rudder":..} (max |deflection|)
    max_thrust_lbf: float
    control_derivatives: Dict[str, float] = field(default_factory=dict)
    mmo: Optional[float] = None
    gain_overrides: Dict[str, Dict] = field(default_factory=dict)
    # Controller/sim task conditions used by the non-legacy presets:
    pitch_cmd_limits_deg: Tuple[float, float] = (-8.0, 12.0)   # pitch-command clamp relative to trim (design choice)
    sim_envelope: Dict = field(default_factory=dict)            # {"min_kcas":..., "nz_limits":[lo, hi]} run-ending limits
    throttle_max: Optional[float] = None                        # throttle-cmd clamp (T38: 0.5 = military power, no AB)
    gene_defaults: Dict[str, float] = field(default_factory=dict)  # per-aircraft gene defaults (optional)
    flex_constants: Dict = field(default_factory=dict)            # flex-wing constants: elastic_axis_frac, section_cg_frac (CG aft of EA)
    shared_gain_bounds: Dict[str, Dict] = field(default_factory=dict)  # from phase1_shared.json (non-legacy tasks only)
    shared_set: Optional[str] = None                                # name of the shared set the profile was overlaid from
    heading_hold: Dict = field(default_factory=dict)                # from phase1_shared.json: enabled, bank_limit_deg, gain_bounds
    disturbance: Dict = field(default_factory=dict)                 # from phase1_shared.json: downdraft_fps (v5 scenario)
    point_masses: Dict[str, int] = field(default_factory=dict)      # JSBSim point-mass indices by role (FD prepared copies)
    aircraft_root: Optional[str] = None                             # informational: where the flown model comes from
    gain_bounds_provisional: Optional[str] = None                   # reason the derived controller bounds are placeholders
    sources: Dict[str, str] = field(default_factory=dict)
    approximate: bool = True
    warnings: List[str] = field(default_factory=list)

    # ---- derived quantities -------------------------------------------------
    def qbar_design(self) -> float:
        v = self.design_point["kcas"] * KT_TO_FPS
        return 0.5 * RHO0 * v * v  # lbf/ft^2 (EAS ~ CAS)

    def tas_design_fps(self) -> float:
        rho = isa_density(self.design_point["alt_ft"])
        return self.design_point["kcas"] * KT_TO_FPS * math.sqrt(RHO0 / rho)

    def _deriv(self, key: str) -> float:
        if key in self.control_derivatives:
            return abs(self.control_derivatives[key])
        msg = f"{self.name}: {key} not given, using generic {GENERIC_DERIVATIVES[key]}"
        if msg not in self.warnings:
            self.warnings.append(msg)
        return GENERIC_DERIVATIVES[key]

    def _per_norm(self, key: str, surface: str) -> float:
        """|moment coefficient| per unit normalized surface command."""
        norm_key = key.replace("_per_rad", "_per_norm")
        if norm_key in self.control_derivatives:
            return abs(self.control_derivatives[norm_key])
        return self._deriv(key) * math.radians(self.control_limits_deg[surface])

    def control_power(self) -> Dict[str, float]:
        """Angular acceleration (rad/s^2) per unit normalized surface command at the design point."""
        q, S = self.qbar_design(), self.wing_area_ft2
        return {
            "pitch": q * S * self.chord_ft * self._per_norm("cm_de_per_rad", "elevator") / self.iyy,
            "roll": q * S * self.span_ft * self._per_norm("cl_da_per_rad", "aileron") / self.ixx,
            "yaw": q * S * self.span_ft * self._per_norm("cn_dr_per_rad", "rudder") / self.izz,
        }

    def design_mach(self) -> float:
        h_m = self.design_point["alt_ft"] * 0.3048
        T = max(288.15 - 0.0065 * h_m, 216.65)
        return self.tas_design_fps() / (math.sqrt(1.4 * 287.053 * T) / 0.3048)

    def task_conditions(self) -> Dict:
        """Scenario/controller conditions for this aircraft (consumed by scenarios.apply_conditions)."""
        env = self.sim_envelope or {}
        return {"aircraft": self.jsbsim_model, "h0_ft": self.design_point["alt_ft"],
                "speed_kts": self.design_point["kcas"],
                "pitch_cmd_limits_deg": tuple(self.pitch_cmd_limits_deg),
                "min_kcas": env.get("min_kcas", round(1.15 * self.vstall_kcas, 1)),
                "nz_limits": tuple(env.get("nz_limits", self.n_limits)),
                "throttle_max": self.throttle_max}

    def accel_per_throttle(self) -> float:
        return self.max_thrust_lbf / self.mass_lb  # g per unit throttle (static, no lapse)

    def check(self) -> List[str]:
        """Sanity checks; returns a list of problems (empty if fine)."""
        p = []
        fc = self.flex_constants
        if fc and not fc.get("section_cg_frac", 1.0) >= fc.get("elastic_axis_frac", 0.0) + 0.02 - 1e-9:
            p.append(f"flex_constants: section CG {fc.get('section_cg_frac')} must be behind the elastic axis {fc.get('elastic_axis_frac')}")
        if not self.vstall_kcas < self.design_point["kcas"] < self.vne_kcas:
            p.append(f"design KCAS {self.design_point['kcas']} not between Vstall {self.vstall_kcas} and Vne {self.vne_kcas}")
        if not (self.n_limits[0] < 0 < 1 < self.n_limits[1]):
            p.append(f"odd n_limits {self.n_limits}")
        for k in ("elevator", "aileron", "rudder"):
            if k not in self.control_limits_deg:
                p.append(f"missing control limit {k}")
        lo, hi = self.pitch_cmd_limits_deg
        if not lo < 0 < hi:
            p.append(f"pitch_cmd_limits_deg {self.pitch_cmd_limits_deg} must straddle 0")
        hh = self.heading_hold or {}
        if hh.get("bank_limit_deg") is not None and not 0 < hh["bank_limit_deg"] < 45.0:
            p.append(f"heading_hold.bank_limit_deg {hh['bank_limit_deg']} must be in (0, 45) (envelope |phi| < 45 deg)")
        env = self.sim_envelope or {}
        if env.get("min_kcas") and not env["min_kcas"] < self.design_point["kcas"]:
            p.append("sim_envelope.min_kcas must be below the design KCAS")
        if self.mmo:
            mach = self.design_mach()
            if mach > self.mmo:
                p.append(f"design point Mach ~{mach:.2f} exceeds MMO {self.mmo}")
        return p


SHARED_FILE = "phase1_shared.json"


def load_shared(path: Optional[str] = None) -> Dict:
    """The Phase-1 shared aircraft/task set (genome/ + evolution/), see ALIGNMENT.md."""
    with open(path or os.path.join(PROFILE_DIR, SHARED_FILE)) as f:
        return json.load(f)


def _overlay_shared(d: Dict, name: str) -> Dict:
    """Shared set is authoritative for trim point, envelope, clamp, throttle and gain bounds of its aircraft.
    A profile JSON that also defines one of these fields must agree (else ValueError) -- no silent duplicates."""
    sh = load_shared()["aircraft"].get(name)
    if sh is None:
        return d
    want = {"jsbsim_model": sh["jsbsim_model"],
            "design_point": {"kcas": sh["trim_kcas"], "alt_ft": sh["trim_alt_ft"]},
            "sim_envelope": {"min_kcas": sh["min_kcas"], "nz_limits": list(sh["nz_limits"])},
            "pitch_cmd_limits_deg": list(sh["pitch_cmd_limits_deg"]),
            "throttle_max": sh.get("throttle_max")}
    for k, v in want.items():
        if k in d and d[k] is not None and (list(d[k]) if isinstance(d[k], tuple) else d[k]) != v:
            raise ValueError(f"profile {name}: {k}={d[k]} conflicts with {SHARED_FILE} ({v}); edit the shared set instead")
        d[k] = v
    hh = {k: v for k, v in sh.get("heading_hold", {}).items() if not k.startswith("_")}
    if "heading_hold" in d and d["heading_hold"]:
        raise ValueError(f"profile {name}: heading_hold is defined by {SHARED_FILE}; edit the shared set instead")
    d["heading_hold"] = hh
    d["disturbance"] = {k: v for k, v in sh.get("disturbance", {}).items() if not k.startswith("_")}
    d["shared_gain_bounds"] = {**sh.get("gain_bounds", {}), **hh.get("gain_bounds", {})}
    d["shared_set"] = SHARED_FILE
    return d


def load_profile(name_or_path: str) -> AircraftProfile:
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(PROFILE_DIR, f"{name_or_path}.json")
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml  # optional dependency
            d = yaml.safe_load(f)
        else:
            d = json.load(f)
    d = {k: v for k, v in d.items() if not k.startswith("_")}  # "_comment" keys
    d = _overlay_shared(d, d.get("name", os.path.splitext(os.path.basename(path))[0]))
    d["n_limits"] = tuple(d["n_limits"])
    if "pitch_cmd_limits_deg" in d:
        d["pitch_cmd_limits_deg"] = tuple(d["pitch_cmd_limits_deg"])
    return AircraftProfile(**d)


def list_profiles() -> List[str]:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(PROFILE_DIR)
                  if f.endswith((".json", ".yaml", ".yml")) and f != SHARED_FILE)


def range_factors(profile: AircraftProfile, reference: Optional[AircraftProfile] = None) -> Dict[str, float]:
    """{scaling_tag: factor} for genome_schema.build_spec. Exactly 1.0 for the reference profile."""
    ref = reference or load_profile(REFERENCE_PROFILE)
    if ref.name == profile.name:
        return {k: 1.0 for k in ("inner_pitch", "inner_roll", "inner_yaw", "outer_vertical", "outer_heading", "speed", "none")}
    cp, cpr = profile.control_power(), ref.control_power()
    v, vr = profile.tas_design_fps(), ref.tas_design_fps()
    return {
        "inner_pitch": cpr["pitch"] / cp["pitch"],
        "inner_roll": cpr["roll"] / cp["roll"],
        "inner_yaw": cpr["yaw"] / cp["yaw"],
        "outer_vertical": vr / v,
        "outer_heading": v / vr,
        "speed": ref.accel_per_throttle() / profile.accel_per_throttle(),
        "none": 1.0,
    }
