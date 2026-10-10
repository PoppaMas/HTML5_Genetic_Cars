"""flexeval_p4.py -- Phase 4 opt-in fidelity 'full_a1_b2a_cs' = full_a1_b2a + FD control-surface model (ctrlsurf_p4).

    import flexeval_p4 as fp4
    fp4.evaluate(gains, struct_genome, scenarios, model, fidelity='full_a1_b2a_cs', root=..., shape_genome=...,
                 cs_mode='active'|'passthrough', cs_overrides=None, record=False)
    any other fidelity -> flexeval_b2.evaluate (pure delegation, unchanged)

Implementation: flexeval_b2.evaluate is run unchanged with its hook class swapped (for the duration of the call only)
for FlexHookP4 = FlexHookB2 + a CSProxy around the B2a proxy. passthrough: no FD writes -> every B2a output field
bit-identical (tested). Not thread-safe across concurrent evaluate() calls in ONE process (the Runner uses processes).
"""
from __future__ import annotations

import contextlib
import json
import os
from typing import Dict, Optional

import numpy as np

import flexeval as fe
import flexeval_b2 as fb2
import ctrlsurf_p4 as cs4

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = "full_a1_b2a_cs"
B2 = fb2.B2
FIDELITIES = fb2.FIDELITIES + (P4,)
CS_MODES = ("active", "passthrough")
CODE_FILES_P4 = (os.path.join(HERE, "ctrlsurf_p4.py"), os.path.join(HERE, "flexeval_p4.py"))
surface_table = cs4.surface_table


def model_version(fidelity: str, model: str, root: str, cs_mode: str = "active", **k) -> str:
    if fidelity != P4:
        return fb2.model_version(fidelity, model, root, **k)
    base = fb2.model_version(B2, model, root, **k)
    code = [open(fp, "rb").read() for fp in CODE_FILES_P4]
    extra = json.dumps({"surfaces": cs4.surface_table(model), "ch": [cs4.CH_ALPHA, cs4.CH_DELTA, cs4.CH_DELTA_ALL_MOVING],
                        "geom": cs4.SURF_GEOM, "hmax": cs4.H_MAX_FACTOR, "eta": [cs4.ETA_GRID_N, cs4.ETA_Q_FRAC],
                        "mode": cs_mode, "base": base}, sort_keys=True, default=str).encode()
    return f"{P4}:{cs4.CS_TAG}:{cs_mode}:{fe._sha8(code + [extra])}"


class _Ctx:
    def __init__(self, mode, record, overrides, mv):
        self.mode, self.record, self.overrides, self.mv = mode, record, overrides, mv
        self.hooks = []
        self.eta = None
        self.hg = None


def _hook_class(ctx: _Ctx):
    class FlexHookP4(fb2.FlexHookB2):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.model_version = ctx.mv
            self.cs = None
            ctx.hooks.append(self)

        def wrap(self, fdm):
            px = super().wrap(fdm)
            if ctx.eta is None:
                ctx.eta = cs4.eta_tables(self.obj)
                ctx.hg = cs4.hinge_geometry(self.model, self.obj)
            self.cs = cs4.CSProxy(px, fdm, self.model, self.obj, self.dt, ctx.mode, ctx.record, ctx.eta, ctx.hg,
                                  ctx.overrides)
            return self.cs
    return FlexHookP4


@contextlib.contextmanager
def _swap_hook(cls):
    orig = fb2.FlexHookB2
    fb2.FlexHookB2 = cls
    try:
        yield
    finally:
        fb2.FlexHookB2 = orig


def evaluate(gains: Dict[str, float], struct_genome, scenarios, model: str, *, fidelity: str, root: str,
             cs_mode: str = "active", cs_overrides: Optional[Dict] = None, record: bool = False, **kw) -> Dict:
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}, got {fidelity!r}")
    if fidelity != P4:
        return fb2.evaluate(gains, struct_genome, scenarios, model, fidelity=fidelity, root=root, record=record, **kw)
    if cs_mode not in CS_MODES:
        raise ValueError(f"cs_mode must be one of {CS_MODES}")
    mv = model_version(P4, model, root, cs_mode=cs_mode)
    ctx = _Ctx(cs_mode, record, cs_overrides, mv)
    with _swap_hook(_hook_class(ctx)):
        out = fb2.evaluate(gains, struct_genome, scenarios, model, fidelity=B2, root=root, record=record, **kw)
    out["fidelity"] = P4
    out["model_version_b2a"] = out.get("model_version")
    out["model_version"] = mv
    per = []
    for i, h in enumerate(ctx.hooks):
        if h.cs is None:
            per.append(None)
            continue
        arr = h.cs.rec.arrays()
        sm = cs4.summarize(arr, h.cs.sdefs, ctx.hg, h.dt)
        per.append(sm)
        if record and "telemetry" in out and i < len(out["telemetry"]):
            out["telemetry"][i]["ctrl_surfaces"] = {k: np.round(v, 9).tolist() for k, v in arr.items()}
    agg = {}
    oks = [p for p in per if p]
    for s in cs4.SURFACES[model]:
        vals = [p[s.name] for p in oks if s.name in p]
        if vals:
            agg[s.name] = {k: (float(np.mean([v[k] for v in vals])) if not isinstance(vals[0][k], bool)
                               else bool(any(v[k] for v in vals))) for k in vals[0] if vals[0][k] is not None}
    out["ctrl_surfaces"] = {
        "schema": cs4.CS_SCHEMA, "mode": cs_mode, "sample_hz": round(1.0 / ctx.hooks[0].dt) if ctx.hooks else None,
        "surfaces": cs4.surface_table(model),
        "hinge": ({k: dict(vars(v)) for k, v in ctx.hg.items()} if ctx.hg else None),
        "eta_table": ({k: {"q_psf": v["q_psf"].tolist(), "eta": v["eta"].tolist(), "r_rigid": v["r_rigid"]}
                       for k, v in ctx.eta.items()} if ctx.eta else None),
        "per_scenario": per, "summary": agg, "in_cost": False}
    return out


# ======================================================================================================================
# fly_course: ring-course flight with an EXTERNAL guidance law (Genome owns it; FD = plant + actuators + limits)
# ======================================================================================================================
import math as _m
import dataclasses as _dc
import flexbody as _fb
import flexwing as _fw
import flexbody_b2 as _fbb2
import planform_b1 as _pb1
import planform_b2 as _pb2
import flexbody_a1 as _fba1

FT2M = 0.3048
KT2MS = 0.514444
COURSE_SCHEMA = "fd-p4-course/1"


def course_limits(model: str) -> Dict:
    d = json.load(open(os.path.join(HERE, "v2_results", "p4_aircraft_limits.json")))["aircraft"][model]
    return d


def _state(f, t, x0):
    return {"t": t, "pos_m": [f["position/distance-from-start-lat-mt"] * (1 if f["position/lat-geod-deg"] >= x0[0] else -1),
                               f["position/distance-from-start-lon-mt"] * (1 if f["position/long-gc-deg"] >= x0[1] else -1),
                               f["position/h-sl-ft"] * FT2M],
            "phi": _m.radians(f["attitude/phi-deg"]), "theta": _m.radians(f["attitude/theta-deg"]),
            "psi": _m.radians(f["attitude/psi-deg"]), "p": f["velocities/p-rad_sec"], "q": f["velocities/q-rad_sec"],
            "r": f["velocities/r-rad_sec"], "vc_kts": f["velocities/vc-kts"], "vt_fps": f["velocities/vt-fps"],
            "nz": f["accelerations/Nz"], "alpha": f["aero/alpha-rad"], "beta": f["aero/beta-rad"],
            "agl_m": f["position/h-agl-ft"] * FT2M, "h_dot_fps": f["velocities/h-dot-fps"]}


def fly_course(profile, gains, struct, shape, surfaces_genome, course, fidelity: str = P4, *, model: str,
               guidance, root: Optional[str] = None, sim=None, cs_mode: str = "active", record_hz: float = 30.0,
               energy: bool = True) -> Dict:
    """Fly one ring course. guidance(state: dict, gains, course, model) -> {'aileron','elevator','rudder','throttle'}
    command norms (optionally 'flap', 'speedbrake'); called every 1/120 s step BEFORE the FD actuator. profile = Profile or
    dict (None -> Phase-1 default for the model). course = {'start': {'alt_ft','kcas','heading_deg'}, 'duration_s', ...}
    (rings are only passed through to guidance; scoring of rings is Evolution's). surfaces_genome = cs_overrides (dict
    per surface, e.g. {'ail': {'rate_dps': 60}}) or None. Positions: local NED-aligned world, x north, y east, z UP (m),
    origin = start point. Returns arrays at record_hz + status / struct_failed / 24 terms / energy."""
    if fidelity == "full_a1_b2a_p4":        # ER's placeholder name (alias)
        fidelity = P4
    if fidelity not in (P4, B2):
        raise ValueError("fly_course supports full_a1_b2a_cs (and full_a1_b2a = surfaces pass-through)")
    import coupled_sim as _cs
    root = root or _cs.ROOT
    sim = sim or fe.load_sim()
    mode = cs_mode if fidelity == P4 else "passthrough"
    prof0 = profile if profile is not None and not isinstance(profile, dict) else (
        sim.Profile.from_dict(profile) if isinstance(profile, dict) else fe.default_profile(model, sim))
    rv2 = fe.root_v2_for(root)
    _fb.ensure_root_v2(model, rv2)
    shp = _pb2.decode_shape_b2(shape, model)
    base_b2 = _pb2.is_baseline_b2(shp, model)
    rfly = rv2 if base_b2 else _fbb2.ensure_root_v2b2(model, rv2, fb2.root_v2b2_for(root))
    P = fe._profile_with_root(prof0, sim, rfly, model)
    st0 = course.get("start", {})
    sc = _dc.replace(fe.phase1_scenarios(model, sim)[0], duration_s=float(course.get("duration_s", 90.0)),
                     h0_ft=float(st0.get("alt_ft", prof0.h0_ft if hasattr(prof0, "h0_ft") else 5000.0)),
                     speed_kts=float(st0.get("kcas", getattr(prof0, "speed_kts", 200.0))),
                     wind_north_fps=0.0, wind_east_fps=0.0, gust_sigma_fps=0.0, discrete_gust_fps=0.0, draft_fps=0.0)
    mv = model_version(P4, model, root, cs_mode=mode)
    wts0 = _fb.StructWeightsV2()
    wts = _dc.replace(wts0, fail_cost=float(2 * P.fail_base))
    asym = fe._is_asym(struct)
    terms = fe._zero_terms()
    out = {"schema": COURSE_SCHEMA, "fidelity": P4 if fidelity == P4 else B2, "cs_mode": mode, "model_version": mv,
           "model": model, "frame": "world x north, y east, z up (m), origin = start; att = (phi, theta, psi) rad"}
    geom = _fb.geometry_for(model, rv2)
    pw = _fw.params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
    env = [(float(sc.speed_kts), float(sc.h0_ft))]
    gr = _pb1.geometry_gate(pw, _pb2.b1_part(shp), n_el=int(_fba1.WING_A1["n_el"]))
    gr2 = (_pb2.geometry_gate_b2(pw, shp, model, n_el=int(_fba1.WING_A1["n_el"]), root_v2=rv2, envelope=env, geom=geom)
           if not base_b2 else _pb2.GateB2(True, None, {}))
    obj = _fbb2.FlexBodyModelB2(model, struct, shape_genes=shp, asymmetric=asym, root_v2=rv2)
    pre = _fbb2.margin_terms_b2(obj, wts0, gate=fb2.MARGIN_GATE[B2])
    for k, v in pre["terms"].items():
        terms[k] = float(v)
    lim = course_limits(model)
    out["limits"] = {k: lim[k] for k in ("alpha_stall_deg", "v_max_kcas", "mach_max", "bank_course_deg", "n_inst",
                                         "n_struct_limit") if k in lim}
    out["surface_limits"] = {s.name: (min(s.cmd_to_deg(s.cmd_lo), s.cmd_to_deg(s.cmd_hi)),
                                      max(s.cmd_to_deg(s.cmd_lo), s.cmd_to_deg(s.cmd_hi)), s.rate_dps)
                             for s in cs4.SURFACES[model]}
    if not gr.ok or not gr2.ok or pre["fail"]:
        out.update(status="geometry_gate" if (not gr.ok or not gr2.ok) else "margin_fail", struct_failed=bool(pre["fail"]),
                   terms=fe._finite(terms), energy=None, t=[], cost_fail=float(wts.fail_cost))
        return out
    ni = _pb2.native_increments(model, shp, pw, geom) if not base_b2 else {"props": {}}
    ctx = _Ctx(mode, True, surfaces_genome, mv)
    hook = _hook_class(ctx)(B2, model, obj, wts, sim.DT, root, rv2, mv, telemetry=False, native_props=ni["props"])
    hook.energy = _fbb2.EnergyMeter(sim.DT, throttle_max=float(P.throttle_max), v_target_kts=None, b2_drag=not base_b2)
    try:
        fdm, info = sim.trim(P, sc, hook)
    except Exception as e:  # noqa: BLE001
        out.update(status="trim_failed", struct_failed=False, terms=fe._finite(terms), energy=None, t=[], error=str(e)[:200])
        return out
    x0 = (fdm["position/lat-geod-deg"], fdm["position/long-gc-deg"])
    px = hook.wrap(fdm)
    n = int(round(sc.duration_s / sim.DT))
    dec = max(1, int(round(1.0 / (sim.DT * record_hz))))
    rec = {k: [] for k in ("t", "pos", "att", "v_kcas", "v_ms", "nz", "alpha", "agl")}
    surf = {s.name: [] for s in cs4.SURFACES[model]}
    names = {"aileron": "fcs/aileron-cmd-norm", "elevator": "fcs/elevator-cmd-norm", "rudder": "fcs/rudder-cmd-norm",
             "flap": "fcs/flap-cmd-norm", "speedbrake": "fcs/speedbrake-cmd-norm"}
    ne = fdm.get_propulsion().get_num_engines()
    n_ult = 1.5 * pw.n_limit
    status = "ok"

    def log(f, t):
        s = _state(f, t, x0)
        rec["t"].append(t); rec["pos"].append(s["pos_m"]); rec["att"].append([s["phi"], s["theta"], s["psi"]])
        rec["v_kcas"].append(s["vc_kts"]); rec["v_ms"].append(s["vt_fps"] * FT2M); rec["nz"].append(s["nz"])
        rec["alpha"].append(s["alpha"]); rec["agl"].append(s["agl_m"])
        for sd in cs4.SURFACES[model]:
            surf[sd.name].append(px._pos_deg(sd)[0])
        return s
    s = log(fdm, 0.0)
    k_end = n
    for k in range(n):
        t = k * sim.DT
        if k:
            s = _state(fdm, t, x0)
        if not all(map(_m.isfinite, (s["pos_m"][2], s["vc_kts"], s["nz"]))):
            status = "diverged"
        elif s["agl_m"] <= 0.0:
            status = "ground"
        elif abs(s["nz"]) > n_ult or (s["nz"] < -0.4 * n_ult):
            status = "structural_failure"
        if status != "ok":
            k_end = k
            break
        u = guidance(s, gains, course, model)
        for kk, prop in names.items():
            if kk in u:
                px[prop] = float(u[kk])
        if "throttle" in u:
            th = min(max(float(u["throttle"]), 0.0), float(P.throttle_max))
            px["fcs/throttle-cmd-norm"] = th
            for i in range(1, ne):
                px[f"fcs/throttle-cmd-norm[{i}]"] = th
        px.run()
        if (k + 1) % dec == 0:
            log(fdm, (k + 1) * sim.DT)
    post = hook.finish("ok" if status in ("ok",) else status)
    if post.get("fail"):
        status = "structural_failure"
    if post.get("terms"):
        for kk, v in post["terms"].items():
            terms[kk] = float(v)
    arr = hook.cs.rec.arrays()
    out.update({kk: np.asarray(v).tolist() for kk, v in rec.items()})
    out["surfaces"] = {kk: v for kk, v in surf.items()}
    out["surfaces_120hz"] = {kk: np.round(v, 9).tolist() for kk, v in arr.items()}
    out["ctrl_summary"] = cs4.summarize(arr, hook.cs.sdefs, ctx.hg, sim.DT)
    out.update(status=status, struct_failed=(status == "structural_failure"), t_end=k_end * sim.DT,
               terms=fe._finite(terms), margins=pre["margins"], sample_hz=record_hz)
    if energy:
        Pref = fe._profile_with_root(prof0, sim, rv2, model)
        e = hook.energy.result(fb2.energy_reference(sim, Pref, sc, model))
        out["energy"] = dict(e, schema=fb2.ENERGY_SCHEMA, in_cost=False)
    return out


def demo_guidance(state, gains, course, model):
    """Reference wings-level / bank-to-heading law used by FD tests only. Genome owns the real law."""
    g = gains or {}
    psi_t = _m.radians(course.get("demo_heading_deg", 0.0))
    e = (psi_t - state["psi"] + _m.pi) % (2 * _m.pi) - _m.pi
    phi_c = max(min(g.get("k_hdg", 1.5) * e, _m.radians(30)), -_m.radians(30))
    ail = max(min(g.get("k_phi", 0.8) * (phi_c - state["phi"]) - 0.15 * state["p"], 1), -1)
    th_c = _m.radians(2.0)
    elev = max(min(-(g.get("k_th", 1.5) * (th_c - state["theta"]) - 0.5 * state["q"]) + g.get("elev0", 0.0), 1), -1)
    return {"aileron": ail, "elevator": elev, "rudder": 0.0, "throttle": g.get("thr", 0.8)}
