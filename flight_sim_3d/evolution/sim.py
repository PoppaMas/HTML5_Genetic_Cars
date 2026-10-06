"""Headless JSBSim evaluation of an altitude-hold controller, for any aircraft.

Generalised from PoppaMas/HTML5_Genetic_Cars@flight-sim-prototype flight_sim/sim.py.
The control law, cost function and envelope checks are the same code; every
constant that was a module global there is now a field of ``Profile``, whose
defaults are exactly the original c172x values. With ``Profile()`` (aircraft
c172x) ``simulate`` reproduces the original bit-for-bit (see
tests/test_equivalence.py).

Additions:
* trim / model-load failures are caught and reported as status ``trim_failed`` /
  ``load_failed`` (cost = 2 * fail_base) instead of crashing the worker;
* network I/O is stripped from the models: some stock aircraft declare JSBSim
  socket inputs/outputs (of the benchmark set only the 737: a telnet input on 5137 and a QTJSBSIM
  UDP input on 5139 that can *set control properties*; c172x's is commented out).
  Every FDM load would try to bind them, and a bound socket lets anything on
  the host poke the simulation. ``_aircraft_root`` serves a cached copy of the
  aircraft directory with only those top-level socket elements removed
  (physics files untouched; bit-identical results are tested);
* ``record=True`` returns a 30 Hz full-state trajectory (position, attitude
  quaternion, velocities, controls) for the 3D viewer. The GA hot path never
  records; the elite is re-simulated with record=True afterwards.

Phase-1 shared-profile options (genome/ALIGNMENT.md); all default OFF so the
prototype path is untouched:
* ``ramp_fpm`` / ``ramp_accel_g``: the altitude *reference* moves toward the
  commanded step at <= ramp_fpm with <= ramp_accel_g corners (exact
  constant-acceleration phases, same algorithm as genome/sim_ext.py
  ``_smooth_plan``); tracking error is scored against that reference and the
  ITAE clock still starts at the command change;
* ``alt_ref_ff``: outer D term on (h_dot - h_ref_dot) instead of h_dot, i.e.
  climb-rate feed-forward of the reference;
* ``w_comfort``: cost += w_comfort * comfort, comfort = genome/fitness.py
  ``comfort_terms`` (rms / max dn, nz jerk, pitch and pitch-rate excess);
* ``roll_kp`` / ``roll_kd``: wing-leveler gains (prototype 0.05 / 0.02);
* ``aircraft_root``: load models from another JSBSim root (Flight Dynamics'
  ``jsbsim_root``); socket stripping applies to those copies too.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Tuple

import numpy as np

DT = 1.0 / 120.0
FT = 0.3048
KT = 0.514444


@dataclass
class Profile:
    """Everything (besides gains and the disturbance draw) that defines one evaluation."""
    aircraft: str = "c172x"
    # initial condition / task (target steps are relative to h0_ft)
    h0_ft: float = 4000.0
    speed_kts: float = 100.0
    duration_s: float = 90.0
    steps_rel_ft: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (5.0, 200.0), (50.0, 0.0)])
    # envelope limits
    min_agl_ft: float = 500.0
    min_kcas: float = 55.0
    max_abs_theta_deg: float = 30.0
    max_abs_phi_deg: float = 45.0
    nz_limits: Tuple[float, float] = (-1.0, 3.8)
    max_alt_err_ft: float = 1000.0
    # controller saturation
    pitch_cmd_limits_deg: Tuple[float, float] = (-8.0, 12.0)
    alt_i_limit_deg: float = 5.0
    pitch_i_limit: float = 0.4
    # fixed helper loops
    thr_kp: float = 0.05
    thr_ki: float = 0.01
    throttle_max: float = 1.0          # e.g. 0.5 = T38 military power (cmd 0.5..1 is afterburner)
    throttle_all_engines: bool = False  # False = only fcs/throttle-cmd-norm[0], as the prototype did
    gear_up: bool = False               # gear/gear-cmd-norm = 0 before run_ic (stock models load gear-down)
    roll_kp: float = 0.05               # wing leveler: ail = clip(-roll_kp*phi - roll_kd*p, +-0.5), deg / deg/s
    roll_kd: float = 0.02
    # heading hold (Phase-1, genome/HANDOFF_heading_hold.md, 2026-10-06): heading error -> bank cmd -> wing leveller
    heading_hold: bool = False
    bank_limit_deg: float = 25.0
    hdg_i_limit_deg: float = 10.0
    w_heading: float = 0.0              # cost += w_heading * RMS(e_psi) / hdg_rms_ref_deg
    hdg_rms_ref_deg: float = 5.0
    # reference shaping (Phase-1); None/False = instant steps, prototype outer loop
    ramp_fpm: Optional[float] = None
    ramp_accel_g: Optional[float] = None   # needs ramp_fpm; None = linear ramp with sharp corners
    alt_ref_ff: bool = False
    # cost
    alt_err_scale_ft: float = 100.0
    itae_t0_s: float = 10.0
    itae_cap_s: float = 30.0
    w_effort: float = 2.0
    fail_base: float = 1000.0
    w_comfort: float = 0.0                 # 0 = comfort not computed (prototype cost)
    comfort_params: Dict[str, float] = field(default_factory=dict)   # overrides COMFORT_PARAMS
    comfort_weights: Dict[str, float] = field(default_factory=dict)  # overrides COMFORT_WEIGHTS
    # GA gene bounds overrides {gene: [min, max]} and kinds {gene: "log"|"linear"|"log0"} (empty = original)
    gain_bounds: Dict[str, List[float]] = field(default_factory=dict)
    gene_kinds: Dict[str, str] = field(default_factory=dict)
    # JSBSim root to load the model from (needs aircraft/<model>/); None = the jsbsim package data
    aircraft_root: Optional[str] = None
    # optional extras (None/{} = untouched JSBSim defaults)
    origin_lat_deg: Optional[float] = None
    origin_lon_deg: Optional[float] = None
    extra_props: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict) -> "Profile":
        names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if not k.startswith("_")}  # "_why" / "_comment" notes
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown profile keys: {sorted(unknown)}")
        d = dict(d)
        for k in ("nz_limits", "pitch_cmd_limits_deg"):
            if k in d:
                d[k] = tuple(float(x) for x in d[k])
        if "steps_rel_ft" in d:
            d["steps_rel_ft"] = [tuple(float(x) for x in s) for s in d["steps_rel_ft"]]
        if d.get("aircraft_root"):
            d["aircraft_root"] = abs_root(d["aircraft_root"])  # configs use "flight-dynamics/jsbsim_root" (relative to flight_sim_3d/)
        p = cls(**d)
        if p.ramp_accel_g is not None and p.ramp_fpm is None:
            raise ValueError("ramp_accel_g needs ramp_fpm")
        if p.ramp_fpm is not None and not p.ramp_fpm > 0 or p.ramp_accel_g is not None and not p.ramp_accel_g > 0:
            raise ValueError("ramp_fpm / ramp_accel_g must be > 0")
        if not (0 < p.bank_limit_deg < p.max_abs_phi_deg):
            raise ValueError(f"bank_limit_deg must be in (0, max_abs_phi_deg={p.max_abs_phi_deg})")
        if not (p.hdg_i_limit_deg > 0 and p.hdg_rms_ref_deg > 0 and p.w_heading >= 0):
            raise ValueError("need hdg_i_limit_deg > 0, hdg_rms_ref_deg > 0, w_heading >= 0")
        bad = set(p.comfort_params) - set(COMFORT_PARAMS) | set(p.comfort_weights) - set(COMFORT_WEIGHTS)
        if bad:
            raise ValueError(f"unknown comfort keys {sorted(bad)}")
        return p

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["nz_limits"] = list(self.nz_limits)
        d["pitch_cmd_limits_deg"] = list(self.pitch_cmd_limits_deg)
        d["steps_rel_ft"] = [list(s) for s in self.steps_rel_ft]
        return d


@dataclass
class Scenario:
    seed: int
    duration_s: float = 90.0
    h0_ft: float = 4000.0
    speed_kts: float = 100.0
    steps: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 4000.0), (5.0, 4200.0), (50.0, 4000.0)])
    wind_north_fps: float = 0.0
    wind_east_fps: float = 0.0
    gust_sigma_fps: float = 0.0
    gust_tau_s: float = 2.0
    discrete_gust_fps: float = 0.0
    discrete_gust_t_s: float = 30.0
    discrete_gust_len_s: float = 3.0
    ramp_fpm: Optional[float] = None
    ramp_accel_g: Optional[float] = None

    def target_cmd(self, t: float) -> Tuple[float, float]:
        """Commanded altitude (instant steps) and time of the last command change."""
        h, t_step = self.steps[0][1], self.steps[0][0]
        for ts, hs in self.steps:
            if t >= ts:
                if hs != h:
                    t_step = ts
                h = hs
        return h, t_step

    def target(self, t: float) -> Tuple[float, float]:
        """Altitude reference that is scored and tracked, and the ITAE start (last command change)."""
        if self.ramp_fpm is None:
            return self.target_cmd(t)
        if self.ramp_accel_g:
            return self._smooth(t)[0], self.target_cmd(t)[1]
        ref, t_step, _ = self._ramp(t)
        return ref, t_step

    def target_rate(self, t: float) -> float:
        """d(reference)/dt in ft/s (0 for instant steps)."""
        if self.ramp_fpm is None:
            return 0.0
        return self._smooth(t)[1] if self.ramp_accel_g else self._ramp(t)[2]

    # Reference shaping: same algorithms as genome/sim_ext.py (ExtScenario._ramp / _smooth_plan / _eval_plan),
    # re-implemented here so the runner does not import another team's module.
    def _ramp(self, t: float) -> Tuple[float, float, float]:
        """Linear ramp at ramp_fpm (sharp corners): (reference, last command change, rate ft/s)."""
        cmd, t_step = self.steps[0][1], self.steps[0][0]
        ref0, t0 = cmd, self.steps[0][0]
        rate = self.ramp_fpm / 60.0
        for ts, hs in self.steps:
            if t >= ts and hs != cmd:
                d = cmd - ref0
                ref0 = ref0 + math.copysign(min(abs(d), rate * (ts - t0)), d) if d else ref0
                t0, cmd, t_step = ts, hs, ts
        d = cmd - ref0
        moved = rate * (t - t0)
        if abs(d) <= moved:
            return cmd, t_step, 0.0
        return ref0 + math.copysign(moved, d), t_step, math.copysign(rate, d)

    def ramp_plan(self) -> List[Tuple[float, float, float, float]]:
        """Accel-limited reference as constant-acceleration phases [(t0, p0, v0, accel), ...].

        |v| <= ramp_fpm/60, |dv/dt| <= ramp_accel_g * g: accelerate, cruise (if the move is long enough),
        decelerate to rest exactly on the command. A command change mid-move continues from the current (p, v);
        if moving away or unable to stop in time it brakes to rest first.
        """
        plan = self.__dict__.get("_plan_cache")
        if plan is not None:
            return plan
        a, vmax = self.ramp_accel_g * 32.174, self.ramp_fpm / 60.0

        def moves(t0, p0, v0, c):
            out = []
            while True:
                d = c - p0
                if abs(d) < 1e-9 and abs(v0) < 1e-9:
                    out.append((t0, c, 0.0, 0.0))
                    return out
                sg = math.copysign(1.0, d) if abs(d) >= 1e-9 else -math.copysign(1.0, v0)
                u0, D = sg * v0, abs(d)
                if u0 < 0 or u0 * u0 / (2 * a) > D + 1e-9:
                    tb = abs(v0) / a
                    out.append((t0, p0, v0, -math.copysign(a, v0)))
                    p0, v0, t0 = p0 + v0 * tb / 2, 0.0, t0 + tb
                    continue
                if (2 * vmax * vmax - u0 * u0) / (2 * a) <= D:  # trapezoid
                    up = vmax
                    t_acc = (vmax - u0) / a
                    t_cru = (D - (vmax * vmax - u0 * u0) / (2 * a) - vmax * vmax / (2 * a)) / vmax
                else:  # triangle
                    up = math.sqrt(a * D + u0 * u0 / 2)
                    t_acc, t_cru = (up - u0) / a, 0.0
                t_dec = up / a
                for dur, acc in ((t_acc, sg * a), (t_cru, 0.0), (t_dec, -sg * a)):
                    if dur > 0:
                        out.append((t0, p0, v0, acc))
                        p0, v0, t0 = p0 + v0 * dur + 0.5 * acc * dur * dur, v0 + acc * dur, t0 + dur
                out.append((t0, c, 0.0, 0.0))
                return out

        plan = [(self.steps[0][0], float(self.steps[0][1]), 0.0, 0.0)]
        cmd = self.steps[0][1]
        for ts, hs in self.steps[1:]:
            if hs == cmd:
                continue
            p0, v0 = self._eval_plan(plan, ts)
            plan = [ph for ph in plan if ph[0] < ts] + moves(ts, p0, v0, float(hs))
            cmd = hs
        self.__dict__["_plan_cache"] = plan
        return plan

    @staticmethod
    def _eval_plan(plan, t):
        ph = plan[0]
        for q in plan:
            if q[0] <= t:
                ph = q
            else:
                break
        t0, p0, v0, acc = ph
        dt = t - t0
        return p0 + v0 * dt + 0.5 * acc * dt * dt, v0 + acc * dt

    def _smooth(self, t: float) -> Tuple[float, float]:
        return self._eval_plan(self.ramp_plan(), t)

    def vertical_gust_series(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        w = np.zeros(n)
        if self.gust_sigma_fps > 0:
            a = math.exp(-DT / self.gust_tau_s)
            b = self.gust_sigma_fps * math.sqrt(1 - a * a)
            noise = rng.standard_normal(n)
            for k in range(1, n):
                w[k] = a * w[k - 1] + b * noise[k]
        if self.discrete_gust_fps:
            t = np.arange(n) * DT
            m = (t >= self.discrete_gust_t_s) & (t < self.discrete_gust_t_s + self.discrete_gust_len_s)
            w[m] += 0.5 * self.discrete_gust_fps * (1 - np.cos(2 * np.pi * (t[m] - self.discrete_gust_t_s) / self.discrete_gust_len_s))
        return w

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["steps"] = [list(s) for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Scenario":
        d = dict(d)
        d["steps"] = [tuple(s) for s in d["steps"]]
        return cls(**d)


def make_scenarios(n: int, seed: int, profile: Optional[Profile] = None) -> List[Scenario]:
    """Same RNG draws as the original; the profile only sets IC/task fields."""
    rng = np.random.default_rng(seed)
    out = [Scenario(seed=int(rng.integers(2**31)))]
    for _ in range(n - 1):
        s = Scenario(seed=int(rng.integers(2**31)))
        spd = rng.uniform(5, 25)
        hdg = rng.uniform(0, 2 * np.pi)
        s.wind_north_fps = float(spd * np.cos(hdg))
        s.wind_east_fps = float(spd * np.sin(hdg))
        s.gust_sigma_fps = float(rng.uniform(1.0, 4.0))
        s.discrete_gust_fps = float(rng.choice([-1, 1]) * rng.uniform(8, 15))
        s.discrete_gust_t_s = float(rng.uniform(25, 45))
        out.append(s)
    if profile is not None:
        for s in out:
            s.duration_s = float(profile.duration_s)
            s.h0_ft = float(profile.h0_ft)
            s.speed_kts = float(profile.speed_kts)
            s.steps = [(float(t), float(profile.h0_ft + dh)) for t, dh in profile.steps_rel_ft]
            s.ramp_fpm = None if profile.ramp_fpm is None else float(profile.ramp_fpm)
            s.ramp_accel_g = None if profile.ramp_accel_g is None else float(profile.ramp_accel_g)
    return out


# comfort (genome/fitness.py comfort_terms / DEFAULT_COMFORT_WEIGHTS; refs make ~1 = "noticeable")
COMFORT_PARAMS = {"rms_dn_ref_g": 0.1, "max_dn_ref_g": 0.3, "jerk_filter_s": 0.2, "jerk_ref_gps": 0.2,
                  "pitch_free_deg": 5.0, "pitch_ref_deg": 1.0, "q_free_dps": 3.0, "q_ref_dps": 1.0}
COMFORT_WEIGHTS = {"rms_dn": 1.0, "max_dn": 0.5, "jerk": 0.25, "pitch_excess": 1.0, "pitch_rate_excess": 0.5}


def _lowpass(x: np.ndarray, dt: float, tau: float) -> np.ndarray:
    a = dt / (tau + dt)
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    return y


def comfort_terms(nz: np.ndarray, theta_deg: np.ndarray, q_dps: np.ndarray, theta_trim_deg: float,
                  dt: float = DT, params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Per-step samples in, normalised sub-metrics out (same formulas as genome/fitness.py)."""
    p = {**COMFORT_PARAMS, **(params or {})}
    dn = nz - 1.0
    jerk = np.diff(_lowpass(nz, dt, p["jerk_filter_s"])) / dt
    pitch_ex = np.maximum(0.0, np.abs(theta_deg - theta_trim_deg) - p["pitch_free_deg"])
    q_ex = np.maximum(0.0, np.abs(q_dps) - p["q_free_dps"])
    return {
        "rms_dn": float(np.sqrt(np.mean(dn ** 2))) / p["rms_dn_ref_g"],
        "max_dn": float(np.max(np.abs(dn))) / p["max_dn_ref_g"],
        "jerk": float(np.sqrt(np.mean(jerk ** 2))) / p["jerk_ref_gps"] if jerk.size else 0.0,
        "pitch_excess": float(np.sqrt(np.mean(pitch_ex ** 2))) / p["pitch_ref_deg"],
        "pitch_rate_excess": float(np.sqrt(np.mean(q_ex ** 2))) / p["q_ref_dps"],
    }


def comfort_score(terms: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    w = {**COMFORT_WEIGHTS, **(weights or {})}
    return float(sum(w.get(k, 0.0) * v for k, v in terms.items()))


_OUT_DIR = os.path.join(tempfile.gettempdir(), "evolution_jsbsim_out")
TEAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # flight_sim_3d/


def abs_root(src_root: Optional[str]) -> Optional[str]:
    """A relative aircraft_root (e.g. "flight-dynamics/jsbsim_root" in the configs) is relative to flight_sim_3d/."""
    if src_root is None or os.path.isabs(src_root):
        return src_root
    return os.path.normpath(os.path.join(TEAM_ROOT, src_root))



class SimSetupError(Exception):
    def __init__(self, status: str, msg: str):
        super().__init__(msg)
        self.status = status


_SOCKET_RE = [re.compile(r"<(input|output)\b[^>]*\bport\s*=[^>]*/>", re.S),
              re.compile(r"<(input|output)\b[^>]*\bport\s*=[^>]*>.*?</\1\s*>", re.S)]
_SAN_ROOT = os.path.join(tempfile.gettempdir(), "evolution_aircraft")
_ROOT_MEMO: Dict[Tuple[Optional[str], str], Optional[str]] = {}


def strip_sockets(xml_text: str) -> str:
    """Remove comments, then every <input|output ... port=...> element (socket I/O). FCS <input>s have no port attribute."""
    txt = re.sub(r"<!--.*?-->", "", xml_text, flags=re.S)
    for rx in _SOCKET_RE:
        txt = rx.sub("", txt)
    return txt


def model_dir(model: str, src_root: Optional[str] = None) -> str:
    """aircraft/<model> directory inside ``src_root`` (None = the jsbsim package data)."""
    if src_root is None:
        import jsbsim
        src_root = os.path.dirname(jsbsim.__file__)
    src_root = abs_root(src_root)
    return os.path.join(src_root, "aircraft", model)


def model_files_sha(model: str, src_root: Optional[str] = None) -> str:
    """sha256 over every file in aircraft/<model> (names + bytes): goes into the cache key, so editing a
    model in place (e.g. Flight Dynamics updating jsbsim_root) can never return stale cached costs."""
    src = model_dir(model, src_root)
    h = hashlib.sha256()
    if not os.path.isdir(src):
        return "missing"
    for dp, dns, fs in os.walk(src):
        dns.sort()
        for f in sorted(fs):
            fp = os.path.join(dp, f)
            h.update(os.path.relpath(fp, src).encode() + b"\0")
            with open(fp, "rb") as fh:
                h.update(fh.read() + b"\0")
    return h.hexdigest()[:16]


_SOCKET_WARNED: set = set()


def socket_io_elements(model: str, src_root: Optional[str] = None) -> int:
    """Number of top-level socket <input|output ... port=...> elements (comments ignored) in the model's main XML."""
    main = os.path.join(model_dir(model, src_root), model + ".xml")
    if not os.path.exists(main):
        return 0
    with open(main, "rb") as f:
        txt = re.sub(r"<!--.*?-->", "", f.read().decode("utf-8", errors="surrogateescape"), flags=re.S)
    return len(re.findall(r"<(?:input|output)\b[^>]*\bport\s*=", txt))


def socket_policy(src_root: Optional[str]) -> str:
    """'strip' for the jsbsim package data (stock 737 declares telnet/QTJSBSIM sockets), 'refuse' for an explicit
    root (FD's jsbsim_root has had its network I/O removed at source since fmt 3, 04:48 PT; a socket appearing there
    is a regression we want to see, not silently patch). EVOLUTION_SOCKET_POLICY=strip|refuse overrides."""
    pol = os.environ.get("EVOLUTION_SOCKET_POLICY")
    if pol in ("strip", "refuse"):
        return pol
    return "strip" if src_root is None else "refuse"


def _aircraft_root(model: str, src_root: Optional[str] = None) -> Optional[str]:
    """Aircraft path to use for ``model`` from ``src_root``: None = the root's own aircraft/ dir (no socket I/O),
    else a sanitized copy. Never writes into ``src_root``. A model with socket elements is stripped LOUDLY (stderr
    warning once per process) under policy 'strip', and refused (SimSetupError load_failed) under 'refuse'."""
    key = (src_root, model)
    if key in _ROOT_MEMO:
        return _ROOT_MEMO[key]
    import jsbsim
    src = model_dir(model, src_root)
    main = os.path.join(src, model + ".xml")
    root = None
    if os.path.exists(main):
        with open(main, "rb") as f:
            raw = f.read()
        txt = raw.decode("utf-8", errors="surrogateescape")
        if re.search(r"<(input|output)\b[^>]*\bport\s*=", re.sub(r"<!--.*?-->", "", txt, flags=re.S)):
            n_el = socket_io_elements(model, src_root)
            if socket_policy(src_root) == "refuse":
                raise SimSetupError("load_failed", f"{main} declares {n_el} socket I/O element(s); refusing to load from "
                                    f"an explicit root (set EVOLUTION_SOCKET_POLICY=strip to strip into a /tmp copy)")
            if (src_root, model) not in _SOCKET_WARNED:
                _SOCKET_WARNED.add((src_root, model))
                print(f"WARNING [evolution.sim] {main} declares {n_el} socket I/O element(s): loading a stripped copy "
                      f"from {_SAN_ROOT} (source untouched)", file=sys.stderr, flush=True)
            new = strip_sockets(txt)
            # sanity: still valid XML, no socket element left, and nothing but socket elements removed
            def tags(t):
                r = ET.fromstring(t[t.find("<", t.find("?>") + 2 if "?>" in t else 0):].encode("utf-8", "surrogateescape"))
                return r, sum(1 for _ in r.iter())
            r_old, n_old = tags(re.sub(r"<!--.*?-->", "", txt, flags=re.S))
            r_new, n_new = tags(new)
            n_sock = sum(1 for e in r_old.iter() if e.tag in ("input", "output") and "port" in e.attrib)
            n_sock_sub = sum(sum(1 for _ in e.iter()) for e in r_old.iter() if e.tag in ("input", "output") and "port" in e.attrib)
            assert n_sock > 0 and n_new == n_old - n_sock_sub, (model, n_old, n_new, n_sock_sub)
            assert not any(e.tag in ("input", "output") and "port" in e.attrib for e in r_new.iter())
            tag = (f"{jsbsim.__version__}-{hashlib.sha256(raw).hexdigest()[:10]}" if src_root is None
                   else f"{jsbsim.__version__}-ext-{model_files_sha(model, src_root)}")
            root = os.path.join(_SAN_ROOT, tag)
            dst = os.path.join(root, model)
            if not os.path.exists(os.path.join(dst, model + ".xml")):
                os.makedirs(root, exist_ok=True)
                tmp = tempfile.mkdtemp(prefix=f".{model}-", dir=root)
                shutil.copytree(src, os.path.join(tmp, model), symlinks=True)
                with open(os.path.join(tmp, model, model + ".xml"), "wb") as f:
                    f.write(new.encode("utf-8", errors="surrogateescape"))
                try:
                    os.rename(os.path.join(tmp, model), dst)  # atomic; loses the race harmlessly
                except OSError:
                    pass
                shutil.rmtree(tmp, ignore_errors=True)
    _ROOT_MEMO[key] = root
    return root


def worker_init():
    """Pool initializer: die with the parent (Linux), so a killed batch leaves no orphan sims."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass


def _new_fdm(profile: Profile):
    os.makedirs(_OUT_DIR, exist_ok=True)
    import jsbsim

    jsbsim.FGJSBBase().debug_lvl = 0
    src_root = profile.aircraft_root
    if src_root is not None and not os.path.exists(os.path.join(model_dir(profile.aircraft, src_root), profile.aircraft + ".xml")):
        raise SimSetupError("load_failed", f"{profile.aircraft} not found in {src_root}/aircraft")
    fdm = jsbsim.FGFDMExec(abs_root(src_root))  # None = package data; else engine/ and systems/ come from src_root
    fdm.set_debug_level(0)
    fdm.set_output_path(_OUT_DIR)
    root = _aircraft_root(profile.aircraft, src_root)
    if root:
        fdm.set_aircraft_path(root)
    try:
        ok = fdm.load_model(profile.aircraft)
    except Exception as e:  # noqa: BLE001
        raise SimSetupError("load_failed", f"{type(e).__name__}: {e}") from e
    if ok is False:
        raise SimSetupError("load_failed", "load_model returned False")
    fdm.disable_output()
    fdm.set_dt(DT)
    for p, v in profile.extra_props.items():
        fdm[p] = v
    return fdm


def trim(profile: Profile, sc: Scenario, flex=None):
    """Load + IC + JSBSim longitudinal trim. Returns (fdm, trim-dict) or raises SimSetupError.

    flex: optional structural hook (evolution.fidelity); flex.attach(fdm) runs BEFORE the IC/trim so the
    gene-driven wing-mass change (FD's injected point masses) is part of the trimmed state."""
    fdm = _new_fdm(profile)
    if flex is not None:
        flex.attach(fdm)
    if profile.origin_lat_deg is not None:
        fdm["ic/lat-geod-deg"] = profile.origin_lat_deg
    if profile.origin_lon_deg is not None:
        fdm["ic/long-gc-deg"] = profile.origin_lon_deg
    if profile.gear_up:
        fdm["gear/gear-cmd-norm"] = 0.0
    fdm["ic/h-sl-ft"] = sc.h0_ft
    fdm["ic/vc-kts"] = sc.speed_kts
    fdm["ic/gamma-deg"] = 0.0
    fdm["ic/psi-true-deg"] = 0.0
    try:
        fdm.run_ic()
        fdm["propulsion/set-running"] = -1
        fdm["fcs/mixture-cmd-norm"] = 1.0
        if profile.throttle_all_engines:
            for i in range(1, fdm.get_propulsion().get_num_engines()):
                fdm[f"fcs/mixture-cmd-norm[{i}]"] = 1.0
        fdm["simulation/do_simple_trim"] = 1  # 1 = full trim (JSBSim tFull)
    except Exception as e:  # noqa: BLE001  (jsbsim.TrimFailureError and friends)
        raise SimSetupError("trim_failed", f"{type(e).__name__}: {str(e)[:200]}") from e
    info = {
        "theta_trim_deg": fdm["attitude/theta-deg"], "elev_trim": fdm["fcs/elevator-cmd-norm"],
        "throttle_trim": fdm["fcs/throttle-cmd-norm"], "vc_kts": fdm["velocities/vc-kts"],
        "alpha_deg": fdm["aero/alpha-deg"], "pitch_trim_cmd": fdm["fcs/pitch-trim-cmd-norm"],
        "h_sl_ft": fdm["position/h-sl-ft"],
        "n_engines": fdm.get_propulsion().get_num_engines(),
        "gear_cmd": fdm["gear/gear-cmd-norm"] if profile.gear_up else -1.0,
    }
    if info["throttle_trim"] > profile.throttle_max + 1e-9:
        raise SimSetupError("trim_failed", f"trim throttle {info['throttle_trim']:.3f} > throttle_max {profile.throttle_max}")
    return fdm, info


def preflight(profile_d: Dict, sc_d: Dict) -> Dict:
    """Trim check used by the batch runner before evolving an aircraft."""
    t0 = time.perf_counter()
    P = Profile.from_dict(profile_d)
    sock = {"socket_io_elements": socket_io_elements(P.aircraft, P.aircraft_root), "socket_policy": socket_policy(P.aircraft_root)}
    try:
        _, info = trim(P, Scenario.from_dict(sc_d))
        return {"ok": True, "status": "ok", **{k: float(v) for k, v in info.items()}, **sock, "wall_s": time.perf_counter() - t0}
    except SimSetupError as e:
        return {"ok": False, "status": e.status, "error": str(e), **sock, "wall_s": time.perf_counter() - t0}


# --- trajectory helpers (record mode only) ---------------------------------
_WGS_A = 6378137.0
_WGS_E2 = 6.69437999014e-3
_Q_ENU_NED = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)  # 180 deg about (1,1,0)/sqrt2: NED -> ENU


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_body_to_enu(phi: float, theta: float, psi: float):
    """JSBSim Euler (rad, ZYX, body FRD relative to local NED) -> quaternion body->ENU [w,x,y,z]."""
    cr, sr = math.cos(phi / 2), math.sin(phi / 2)
    cp, sp = math.cos(theta / 2), math.sin(theta / 2)
    cy, sy = math.cos(psi / 2), math.sin(psi / 2)
    q_nb = (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)
    return _qmul(_Q_ENU_NED, q_nb)


class ReadOnlyFDM:
    """What a recorder sees: property reads only (writes raise), so a recorder cannot perturb the flight."""
    __slots__ = ("_f",)

    def __init__(self, fdm):
        object.__setattr__(self, "_f", fdm)

    def __getitem__(self, k):
        return self._f[k]

    def __setitem__(self, k, v):
        raise TypeError("recorders are read-only: cannot set " + str(k))

    def __setattr__(self, k, v):
        raise TypeError("recorders are read-only")


class TrajRecorder:
    """Recorder producing the ga-flightsim-traj/1 rows (simulate(record=True) and the trajectory export use it).

    Recorder protocol (evolution.eval.evaluate / sim.simulate):
      * recorder(t, fdm[, flex_state]) once at t = 0 after the IC + trim (before the first step), then after every
        fdm.run() with t = (k + 1) * DT. fdm is read-only. flex_state is passed only when flex is active.
      * recorder.final(t_end, fdm[, flex_state]) once when the flight ends (completed or terminated), if defined.
    Rows are sampled every round(1/(DT*hz)) steps on the absolute step grid (t = 0, 4*DT, ... at 30 Hz) plus the
    final state at t_end.

    controls_timing = "pre_step" (unchanged since the first export): the throttle/elevator/aileron/rudder of the
    row at time t are the commands applied over the step that STARTS at t, i.e. what the controller set from the
    state at t. A post-step recorder sees, in fcs/*-cmd-norm at time t, the commands of the step that ENDED at t, so
    this recorder fills each sampled row's controls at the next call (one step later). The final row keeps the
    last applied commands (hold-last). State channels of row 0 are exactly what the t = 0 call sees.
    """
    CHANNELS = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m",
                "phi", "theta", "psi", "throttle", "elevator", "aileron", "rudder",
                # extras (viewer ignores unknown channels)
                "target_alt_m", "kcas", "nz", "ub", "vb", "wb", "lat_deg", "lon_deg",
                "target_cmd_alt_m", "target_rate_mps"]
    CTRL_IDX = (15, 16, 17, 18)
    CTRL_PROPS = ("fcs/throttle-cmd-norm", "fcs/elevator-cmd-norm", "fcs/aileron-cmd-norm", "fcs/rudder-cmd-norm")
    CONTROLS_TIMING = "pre_step"

    def __init__(self, sc: "Scenario", sample_hz: float = 30.0, flex_channels: bool = False):
        self.sc = sc
        self.sample_hz = sample_hz
        self.every = max(1, int(round(1.0 / (DT * sample_hz))))
        self.flex_channels = flex_channels
        self.struct_names: List[str] = []
        self.structure = None
        self.rows: List[List[float]] = []
        self.q_prev = None
        self.trim = None
        self._pending = None
        self._last_t = None

    def _origin(self, fdm):
        self.lat0 = fdm["position/lat-geod-deg"]
        self.lon0 = fdm["position/long-gc-deg"]
        self.alt0_m = fdm["position/h-sl-ft"] * FT
        lat = math.radians(self.lat0)
        s2 = math.sin(lat) ** 2
        self.r_north = _WGS_A * (1 - _WGS_E2) / (1 - _WGS_E2 * s2) ** 1.5 + self.alt0_m
        self.r_east = (_WGS_A / math.sqrt(1 - _WGS_E2 * s2) + self.alt0_m) * math.cos(lat)
        self.trim = {"theta_trim_deg": fdm["attitude/theta-deg"], "elev_trim": fdm["fcs/elevator-cmd-norm"],
                     "throttle_trim": fdm["fcs/throttle-cmd-norm"], "vc_target_kts": fdm["velocities/vc-kts"]}

    def _fill_controls(self, fdm):
        if self._pending is not None:
            for i, p in zip(self.CTRL_IDX, self.CTRL_PROPS):
                self._pending[i] = fdm[p]
            self._pending = None

    def __call__(self, t, fdm, flex_state=None):
        if self.trim is None:
            self._origin(fdm)
        self._fill_controls(fdm)          # commands of the step that just ended = the pending row's "pre_step" controls
        if int(round(t / DT)) % self.every == 0:
            self._add(t, fdm, flex_state)
            self._pending = self.rows[-1]

    def final(self, t, fdm, flex_state=None):
        self._fill_controls(fdm)          # last row sampled exactly at t_end: hold-last commands
        if self._last_t is not None and abs(self._last_t - t) < 0.5 * DT:
            return
        if all(map(math.isfinite, (fdm["position/h-sl-ft"], fdm["attitude/theta-deg"], fdm["velocities/vc-kts"],
                                   fdm["accelerations/Nz"]))):
            self._add(t, fdm, flex_state)  # controls read now = last applied commands (hold-last)
            self._pending = None

    def _add(self, t, fdm, flex_state):
        phi, theta, psi = fdm["attitude/phi-rad"], fdm["attitude/theta-rad"], fdm["attitude/psi-rad"]
        q = quat_body_to_enu(phi, theta, psi)
        if self.q_prev is not None and sum(a * b for a, b in zip(q, self.q_prev)) < 0:
            q = tuple(-c for c in q)  # keep hemisphere continuous for interpolation
        self.q_prev = q
        lat, lon = fdm["position/lat-geod-deg"], fdm["position/long-gc-deg"]
        alt_m = fdm["position/h-sl-ft"] * FT
        sc = self.sc
        target_ft = sc.target(t)[0]
        rate_fps = sc.target_rate(t) if sc.ramp_fpm is not None else 0.0
        row = [
            t, math.radians(lon - self.lon0) * self.r_east, math.radians(lat - self.lat0) * self.r_north, alt_m - self.alt0_m,
            *q,
            fdm["velocities/v-east-fps"] * FT, fdm["velocities/v-north-fps"] * FT, -fdm["velocities/v-down-fps"] * FT,
            alt_m, phi, theta, psi,
            fdm["fcs/throttle-cmd-norm"], fdm["fcs/elevator-cmd-norm"], fdm["fcs/aileron-cmd-norm"], fdm["fcs/rudder-cmd-norm"],
            target_ft * FT, fdm["velocities/vc-kts"], fdm["accelerations/Nz"],
            fdm["velocities/u-fps"] * FT, fdm["velocities/v-fps"] * FT, fdm["velocities/w-fps"] * FT, lat, lon,
            sc.target_cmd(t)[0] * FT, rate_fps * FT,
        ]
        if self.flex_channels and flex_state is not None:
            ch = flex_state.channels()
            if not self.struct_names:
                self.struct_names = list(ch)
                self.structure = flex_state.structure
            row += [float(ch[k]) for k in self.struct_names]
        self.rows.append(row)
        self._last_t = t

    def trajectory(self) -> Dict:
        out = {"channels": list(self.CHANNELS) + list(self.struct_names), "data": self.rows, "sample_hz": self.sample_hz,
               "dt_s": self.every * DT, "sim_dt_s": DT, "controls_timing": self.CONTROLS_TIMING,
               "origin": {"lat_deg": self.lat0, "lon_deg": self.lon0, "alt_m": self.alt0_m}, "trim": self.trim}
        if self.structure is not None:
            out["structure"] = self.structure
        return out


_Recorder = TrajRecorder  # backward-compatible name


def simulate(gains: Dict[str, float], sc: Scenario, profile: Optional[Profile] = None,
             record: bool = False, sample_hz: float = 30.0, recorder=None, flex=None) -> Dict:
    """Fly one scenario. Returns cost and status (and a trajectory when record=True).

    recorder: optional read-only callback recorder(t, fdm[, flex_state]) (see TrajRecorder); record=True is
    shorthand for a TrajRecorder whose trajectory is returned. flex: optional structural hook
    (evolution.fidelity.FlexHook): attach() before trim, wrap(fdm) after trim (FD's FlexFDM proxy: structure step
    + external_reactions feedback before every JSBSim step). With recorder=None and flex=None this is the rigid
    fast path; neither option changes what the controller sees, so costs are bit-identical with or without a
    recorder."""
    P = profile or Profile()
    t_wall = time.perf_counter()
    if record and recorder is None:
        recorder = TrajRecorder(sc, sample_hz, flex_channels=flex is not None)
    try:
        fdm, _ = trim(P, sc, flex)
    except SimSetupError as e:
        return {"cost": float(2 * P.fail_base), "status": e.status, "t_end": 0.0, "track": float("nan"),
                "effort": float("nan"), "error": str(e), "wall_s": time.perf_counter() - t_wall}

    theta_trim = fdm["attitude/theta-deg"]
    elev_trim = fdm["fcs/elevator-cmd-norm"]
    thr_trim = fdm["fcs/throttle-cmd-norm"]
    v_target = fdm["velocities/vc-kts"]

    fdm["atmosphere/wind-north-fps"] = sc.wind_north_fps
    fdm["atmosphere/wind-east-fps"] = sc.wind_east_fps
    if flex is not None:
        fdm = flex.wrap(fdm)
    ro = ReadOnlyFDM(fdm) if recorder is not None else None
    flex_state = flex.state if flex is not None else None

    n = int(round(sc.duration_s / DT))
    w_down = sc.vertical_gust_series(n)

    thr_max = P.throttle_max
    extra_eng = range(1, fdm.get_propulsion().get_num_engines()) if P.throttle_all_engines else range(0)
    kp_a, ki_a, kd_a = gains["kp_alt"], gains["ki_alt"], gains["kd_alt"]
    kp_p, ki_p, kd_p = gains["kp_pitch"], gains["ki_pitch"], gains["kd_pitch"]
    lo_cmd, hi_cmd = P.pitch_cmd_limits_deg
    nz_lo, nz_hi = P.nz_limits
    roll_kp, roll_kd = P.roll_kp, P.roll_kd
    hh = P.heading_hold
    if hh:
        kp_h, ki_h = gains["kp_hdg"], gains["ki_hdg"]
        bank_lim = P.bank_limit_deg
    i_hdg = 0.0
    c_psi: List[float] = []             # psi per step flown (deg), for the heading cost term
    ramp = sc.ramp_fpm is not None
    ff = P.alt_ref_ff
    comfort_on = P.w_comfort > 0
    c_nz: List[float] = []
    c_th: List[float] = []
    c_q: List[float] = []

    i_alt = 0.0
    i_pitch = 0.0
    i_spd = 0.0
    elev_prev = elev_trim
    sum_err = 0.0
    sum_tv = 0.0
    status = "ok"
    k_end = n
    if recorder is not None:
        recorder(0.0, ro) if flex is None else recorder(0.0, ro, flex_state())

    for k in range(n):
        t = k * DT
        fdm["atmosphere/wind-down-fps"] = float(w_down[k])

        h = fdm["position/h-sl-ft"]
        h_agl = fdm["position/h-agl-ft"]
        h_dot = fdm["velocities/h-dot-fps"]
        theta = fdm["attitude/theta-deg"]
        phi = fdm["attitude/phi-deg"]
        q = math.degrees(fdm["velocities/q-rad_sec"])
        p = math.degrees(fdm["velocities/p-rad_sec"])
        vc = fdm["velocities/vc-kts"]
        nz = fdm["accelerations/Nz"]
        target, t_step = sc.target(t)
        e_h = target - h
        h_ref_dot = sc.target_rate(t) if ramp else 0.0

        if not all(map(math.isfinite, (h, theta, vc, nz))):
            status = "diverged"
        elif h_agl < P.min_agl_ft:
            status = "crash"
        elif vc < P.min_kcas:
            status = "stall"
        elif abs(theta) > P.max_abs_theta_deg or abs(phi) > P.max_abs_phi_deg:
            status = "attitude"
        elif not (nz_lo <= nz <= nz_hi):
            status = "overload"
        elif abs(e_h) > P.max_alt_err_ft:
            status = "diverged"
        if status != "ok":
            k_end = k
            break
        if comfort_on:
            c_nz.append(nz)
            c_th.append(theta)
            c_q.append(q)

        if ki_a > 0:
            lim = P.alt_i_limit_deg / ki_a
            i_alt = min(max(i_alt + e_h * DT, -lim), lim)
        if ff:
            u_alt = kp_a * e_h + ki_a * i_alt - kd_a * (h_dot - h_ref_dot)
        else:
            u_alt = kp_a * e_h + ki_a * i_alt - kd_a * h_dot
        theta_cmd = theta_trim + min(max(u_alt, lo_cmd), hi_cmd)

        e_th = theta_cmd - theta
        if ki_p > 0:
            lim = P.pitch_i_limit / ki_p
            i_pitch = min(max(i_pitch + e_th * DT, -lim), lim)
        u_p = kp_p * e_th + ki_p * i_pitch - kd_p * q
        elev = min(max(elev_trim - u_p, -1.0), 1.0)

        e_v = v_target - vc
        i_spd = min(max(i_spd + e_v * DT, -50.0), 50.0)
        thr = min(max(thr_trim + P.thr_kp * e_v + P.thr_ki * i_spd, 0.0), thr_max)
        if hh:
            psi = fdm["attitude/psi-deg"]                 # 0..360
            e_psi = (0.0 - psi + 180.0) % 360.0 - 180.0   # wrap180(psi_target - psi), psi_target = 0 (= IC heading)
            if ki_h > 0:
                lim = P.hdg_i_limit_deg / ki_h
                i_hdg = min(max(i_hdg + e_psi * DT, -lim), lim)
            phi_cmd = min(max(kp_h * e_psi + ki_h * i_hdg, -bank_lim), bank_lim)
            c_psi.append(psi)
            ail = min(max(-roll_kp * (phi - phi_cmd) - roll_kd * p, -0.5), 0.5)
        else:
            ail = min(max(-roll_kp * phi - roll_kd * p, -0.5), 0.5)

        fdm["fcs/elevator-cmd-norm"] = elev
        fdm["fcs/throttle-cmd-norm"] = thr
        for i_e in extra_eng:
            fdm[f"fcs/throttle-cmd-norm[{i_e}]"] = thr
        fdm["fcs/aileron-cmd-norm"] = ail
        fdm["fcs/rudder-cmd-norm"] = 0.0

        sum_tv += abs(elev - elev_prev)
        elev_prev = elev
        sum_err += abs(e_h) / P.alt_err_scale_ft * min(t - t_step, P.itae_cap_s) / P.itae_t0_s

        fdm.run()
        if recorder is not None:
            recorder((k + 1) * DT, ro) if flex is None else recorder((k + 1) * DT, ro, flex_state())

    if recorder is not None and hasattr(recorder, "final"):
        recorder.final(k_end * DT, ro) if flex is None else recorder.final(k_end * DT, ro, flex_state())

    comfort = float("nan")
    terms = None
    if status == "ok":
        track = sum_err / n
        effort = sum_tv / sc.duration_s
        cost = track + P.w_effort * effort
        if comfort_on:
            terms = comfort_terms(np.asarray(c_nz), np.asarray(c_th), np.asarray(c_q), theta_trim, DT, P.comfort_params)
            comfort = comfort_score(terms, P.comfort_weights)
            cost = cost + P.w_comfort * comfort
        if hh and P.w_heading > 0:
            e = (0.0 - np.asarray(c_psi) + 180.0) % 360.0 - 180.0   # exactly genome's numpy form (pairwise np.mean)
            heading_rms = float(np.sqrt(np.mean(e ** 2))) / P.hdg_rms_ref_deg
            cost = cost + P.w_heading * heading_rms
    else:
        track = effort = float("nan")
        cost = P.fail_base + P.fail_base * (1.0 - k_end / n)

    out = {"cost": float(cost), "status": status, "t_end": k_end * DT, "track": float(track),
           "effort": float(effort), "wall_s": time.perf_counter() - t_wall}
    if comfort_on:
        out["comfort"] = comfort
        out["comfort_terms"] = terms
    if hh:
        out["heading_rms"] = heading_rms if (status == "ok" and P.w_heading > 0) else float("nan")
        if c_psi:
            e_end = (c_psi[-1] - c_psi[0] + 180.0) % 360.0 - 180.0
            out["hdg_drift_deg"] = float(e_end)   # wrap180(psi_last_step - psi_0) over the steps flown
            out["hdg_max_abs_err_deg"] = float(np.max(np.abs((0.0 - np.asarray(c_psi) + 180.0) % 360.0 - 180.0)))
    if flex is not None:
        out["_flex"] = flex.finish(status)   # structural response terms (stripped by evolution.fidelity)
    if record:
        out["trajectory"] = recorder.trajectory()
    return out


# --- worker entry points (picklable, dict in / dict out) --------------------
def eval_task(profile_d: Dict, gains: Dict[str, float], sc_d: Dict) -> Dict:
    return simulate(gains, Scenario.from_dict(sc_d), Profile.from_dict(profile_d))


def record_task(profile_d: Dict, gains: Dict[str, float], sc_d: Dict, sample_hz: float = 30.0) -> Dict:
    return simulate(gains, Scenario.from_dict(sc_d), Profile.from_dict(profile_d), record=True, sample_hz=sample_hz)
