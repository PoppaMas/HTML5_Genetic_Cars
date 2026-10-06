"""JSBSim + flexwing coupling harness.

* run_maneuver(): open-loop demo maneuver (elevator pull, aileron doublet, 1-cos
  vertical gust) for rigid ('oneway': structure computed, no feedback) vs flexible
  ('twoway') flight.
* evaluate_flex(): composes with the existing GA fitness by re-using the repo's
  sim.simulate() *unchanged*: sim._new_fdm is temporarily replaced by a factory
  returning FlexFDM, a thin proxy whose run() advances the wing structure and
  sets the external_reactions feedback before each JSBSim step.
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO_FLIGHT_SIM = os.environ.get("FLIGHT_SIM_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flight_sim"))
ROOT = os.path.join(HERE, "jsbsim_root")

import flexwing as fw  # noqa: E402

DIAG_KEYS = ("tip_w_ft_R", "tip_w_ft_L", "tip_twist_deg_R", "tip_twist_deg_L", "root_bm_lbft_R", "root_bm_lbft_L",
             "dL_lbf", "dRoll_lbft", "dPitch_lbft")


def ensure_root(model: str, root: str = ROOT) -> str:
    xml = os.path.join(root, "aircraft", model, model + ".xml")
    meta = os.path.join(root, "aircraft", model, "flexwing_meta.json")
    import json as _json
    stale = (not os.path.exists(xml) or not os.path.exists(meta)
             or "flexwing_F" not in open(xml, encoding="utf-8", errors="replace").read()
             or _json.load(open(meta)).get("fmt", 1) < fw.PREPARE_FMT)
    if stale:
        fw.prepare_aircraft(model, root)
    return root


class FlexFDM:
    """Proxy around FGFDMExec: run() = structure step + feedback, then the real JSBSim step."""

    def __init__(self, fdm, coupler: Optional[fw.FlexCoupler], dt: float, record: bool = True):
        self._fdm = fdm
        self.coupler = coupler
        self.dt = dt
        self.started = False
        self.record = record
        self.hist = {k: [] for k in DIAG_KEYS}
        self.m_root_1g = float("nan")

    def __getitem__(self, k):
        return self._fdm[k]

    def __setitem__(self, k, v):
        self._fdm[k] = v

    def __getattr__(self, name):
        return getattr(self._fdm, name)

    def run(self):
        c = self.coupler
        if c is not None:
            if not self.started:
                c.initialize(self._fdm)
                self.m_root_1g = 0.5 * (c.last["root_bm_lbft_R"] + c.last["root_bm_lbft_L"])
                self.started = True
            c.step(self._fdm, self.dt)
            if self.record:
                for k in DIAG_KEYS:
                    self.hist[k].append(c.last[k])
        return self._fdm.run()

    def history(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.hist.items()}


def make_coupler(fdm, model: str, mode: str, overrides: Optional[Dict] = None, substeps: int = 4,
                 root: str = ROOT, apply_mass: bool = True) -> fw.FlexCoupler:
    """Build the coupler; with apply_mass the gene-driven wing-mass change is pushed into JSBSim
    (point masses) -- call before run_ic/trim."""
    wing = fw.wing_from_fdm(fdm, model, **(overrides or {}))
    c = fw.FlexCoupler(wing, mode=mode, substeps=substeps)
    c.delta_mass_lb = fw.apply_wing_mass(fdm, model, root, wing) if apply_mass else 0.0
    return c


def trim(fdm, h_ft: float, kcas: float, gear_up: bool = True):
    """Full (mode 1) trim. gear/gear-cmd-norm loads as 1 (DOWN) in every stock model; retract for cruise.
    The trimmer moves the gear actuator instantly, so gear-pos-norm is 0 right after trim (verified in phase1_check)."""
    if gear_up:
        fdm["gear/gear-cmd-norm"] = 0.0
    fdm["ic/h-sl-ft"] = h_ft
    fdm["ic/vc-kts"] = kcas
    fdm["ic/gamma-deg"] = 0.0
    fdm["ic/psi-true-deg"] = 0.0
    fdm.run_ic()
    fdm["propulsion/set-running"] = -1
    for i in range(max(1, fdm.get_propulsion().get_num_engines())):
        fdm[f"fcs/mixture-cmd-norm[{i}]"] = 1.0
    fdm["simulation/do_simple_trim"] = 1


MANEUVER_DEFAULTS = {
    "c172x": dict(h_ft=4000.0, kcas=100.0, d_elev=-0.30, d_ail=0.40, gust_fps=-20.0),
    "737": dict(h_ft=10000.0, kcas=250.0, d_elev=-0.12, d_ail=0.40, gust_fps=-30.0),
    "T38": dict(h_ft=10000.0, kcas=300.0, d_elev=-0.10, d_ail=0.30, gust_fps=-30.0),
    # f16: elevator/aileron cmds are FBW pitch-rate/g and roll-rate demands (FCS in f16.xml), not surface positions
    "f16": dict(h_ft=10000.0, kcas=350.0, d_elev=-0.10, d_ail=0.30, gust_fps=-30.0),
}


def run_maneuver(model: str = "c172x", mode: str = "twoway", overrides: Optional[Dict] = None, duration_s: float = 12.0,
                 dt: float = 1 / 120, substeps: int = 4, root: str = ROOT, use_coupler: bool = True) -> Dict:
    """Trim, then: elevator pull 1.0-2.5 s, aileron doublet 4-5 / 5-6 s, 1-cos vertical gust 8-9.5 s."""
    ensure_root(model, root)
    md = MANEUVER_DEFAULTS.get(model, MANEUVER_DEFAULTS["737"])
    fdm = fw.new_fdm(model, root, dt)
    coupler = make_coupler(fdm, model, mode, overrides, substeps, root) if use_coupler else None
    trim(fdm, md["h_ft"], md["kcas"])
    elev0 = fdm["fcs/elevator-cmd-norm"]
    px = FlexFDM(fdm, coupler, dt)
    n = int(round(duration_s / dt))
    keys = ("t", "alt_ft", "nz", "p_dps", "q_dps", "phi_deg", "theta_deg", "alpha_deg", "kcas", "qbar",
            "fbz_ext", "l_ext", "m_ext")
    rec = {k: np.zeros(n) for k in keys}
    for k in range(n):
        t = k * dt
        elev = elev0 + (md["d_elev"] if 1.0 <= t < 2.5 else 0.0)
        ail = md["d_ail"] if 4.0 <= t < 5.0 else (-md["d_ail"] if 5.0 <= t < 6.0 else 0.0)
        wd = 0.5 * md["gust_fps"] * (1 - math.cos(2 * math.pi * (t - 8.0) / 1.5)) if 8.0 <= t < 9.5 else 0.0
        fdm["fcs/elevator-cmd-norm"] = elev
        fdm["fcs/aileron-cmd-norm"] = ail
        fdm["atmosphere/wind-down-fps"] = wd
        px.run()
        for key, prop in (("alt_ft", "position/h-sl-ft"), ("nz", "accelerations/Nz"), ("phi_deg", "attitude/phi-deg"),
                          ("theta_deg", "attitude/theta-deg"), ("alpha_deg", "aero/alpha-deg"), ("kcas", "velocities/vc-kts"),
                          ("qbar", "aero/qbar-psf"), ("fbz_ext", "forces/fbz-external-lbs"),
                          ("l_ext", "moments/l-external-lbsft"), ("m_ext", "moments/m-external-lbsft")):
            rec[key][k] = fdm[prop]
        rec["t"][k] = t + dt
        rec["p_dps"][k] = math.degrees(fdm["velocities/p-rad_sec"])
        rec["q_dps"][k] = math.degrees(fdm["velocities/q-rad_sec"])
    out = dict(rec)
    if coupler is not None:
        out.update(px.history())
        out["m_root_1g"] = px.m_root_1g
        out["margins"] = coupler.w.margins()
    return out


# ---------------- composition with the existing GA fitness ----------------
def _import_base_sim():
    if REPO_FLIGHT_SIM not in sys.path:
        sys.path.insert(0, REPO_FLIGHT_SIM)
    import sim  # the repo's flight_sim/sim.py (read-only use)
    return sim


@contextlib.contextmanager
def patched_fdm_factory(base, model: str, mode: str, overrides: Dict, substeps: int, sink: list, root: str = ROOT):
    orig_new, orig_ac = base._new_fdm, base.AIRCRAFT

    def factory():
        fdm = fw.new_fdm(model, root, base.DT)
        coupler = make_coupler(fdm, model, mode, overrides, substeps, root)
        px = FlexFDM(fdm, coupler, base.DT)
        sink.append(px)
        return px

    base._new_fdm, base.AIRCRAFT = factory, model
    try:
        yield
    finally:
        base._new_fdm, base.AIRCRAFT = orig_new, orig_ac


def evaluate_flex(gains: Dict[str, float], struct_genome: Optional[Sequence[float]], scenarios, model: str = "c172x",
                  mode: str = "twoway", weights: fw.StructWeights = fw.StructWeights(), substeps: int = 2,
                  overrides: Optional[Dict] = None) -> Dict:
    """Altitude-hold cost (repo sim.simulate, unchanged) + structural terms. Lower is better."""
    base = _import_base_sim()
    ensure_root(model)
    ov = dict(overrides or {})
    if struct_genome is not None:
        ov.update(fw.genes_to_overrides(fw.decode_struct(struct_genome)))
    # pre-sim margins: a genome that flutters/diverges below V_D is rejected without flying
    probe = fw.new_fdm(model, ROOT, base.DT)
    wing = fw.wing_from_fdm(probe, model, **ov)
    pre = fw.margin_terms(wing, weights)
    if pre["fail"]:
        return {"cost": weights.fail_cost, "status": pre["fail"], "pre": pre, "per_scenario": []}
    per = []
    for sc in scenarios:
        sink: list = []
        with patched_fdm_factory(base, model, mode, ov, substeps, sink):
            r = base.simulate(gains, sc)
        px = sink[0]
        entry = {k: r[k] for k in ("cost", "status", "t_end", "track", "effort")}
        if r["status"] == "ok":
            post = fw.response_terms(px.history(), px.coupler.w, px.m_root_1g, weights)
            entry.update(struct=post["terms"], bm_peak=post["bm_peak"], bm_allow=post["bm_allow"], tip_max_ft=post["tip_max_ft"],
                         twist_max_deg=post["twist_max_deg"], bm_rms=post["bm_rms"], m_root_1g=px.m_root_1g)
            if post["fail"]:
                entry["status"] = post["fail"]
                entry["cost_total"] = weights.fail_cost
            else:
                entry["cost_total"] = r["cost"] + sum(post["terms"].values())
        else:
            entry["cost_total"] = r["cost"]
        per.append(entry)
    pre_sum = sum(v for k, v in pre["terms"].items() if k.startswith("J_"))
    cost = float(np.mean([e["cost_total"] for e in per])) + pre_sum
    return {"cost": cost, "status": "ok" if all(e["status"] == "ok" for e in per) else "fail", "pre": pre, "per_scenario": per}
