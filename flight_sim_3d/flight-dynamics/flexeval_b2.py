"""flexeval_b2.py -- P3-B2a opt-in fidelity 'full_a1_b2a' (B1 r1 + section genes). INTERFACE_v2.md section 15.

    import flexeval_b2 as fb2
    fb2.evaluate(gains, struct_genome, scenarios, model, fidelity='full_a1_b2a', root=..., shape_genome={B1 + B2a keys})
    fb2.evaluate(..., fidelity='rigid'|'reduced'|'full'|'full_a1'|'full_a1_b1')   # delegates to flexeval_b1

B2 genes at default: B1 model / B1 coupler arithmetic / B1 terms, flown on <root>_v2 (no native functions) -> every
pre-existing output field equals full_a1_b1 exactly (tested). Different by design: fidelity / model_version /
margins_fidelity strings, shape_cache_key (b2a| key), wall_s, and the ADDED keys shape_genes_b2, b2, geometry_gate_b2,
native_increments, energy. Non-default B2: flown on <root>_v2b2 with the native increment properties set before trim.

energy (outside TERM_KEYS, never in cost): per scenario drag_work_ftlbf = sum(D V dt) (aero drag incl. native B2
increments), drag_power_ref_ftlbf_s = D V of the BASELINE aircraft (<root>_v2, A1 masses, no B2) trimmed at the
scenario IC (gene- and gain-independent, cached), energy_drag_ratio = work / (ref x t_flown), throttle_mean,
fuel_burned_lbs.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np

import flexwing as fw
import flexbody as fb
import flexeval as fe
import flexeval_a1 as fa
import flexeval_b1 as fb1
import flexbody_a1 as fba1
import flexbody_b2 as fbb2
import planform_b1 as pb1
import planform_b2 as pb2

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = pb2.B2_FIDELITY  # 'full_a1_b2a'
B1 = fb1.B1
FIDELITIES = fb1.FIDELITIES + (B2,)
TERM_KEYS = fe.TERM_KEYS
PRE_TERMS = dict(fb1.PRE_TERMS, **{B2: fb1.PRE_TERMS[B1]})
RESP_TERMS = dict(fb1.RESP_TERMS, **{B2: fb1.RESP_TERMS[B1]})
SUBSTEPS = dict(fb1.SUBSTEPS, **{B2: fb1.SUBSTEPS[B1]})
MARGIN_GATE = dict(fb1.MARGIN_GATE, **{B2: fb1.MARGIN_GATE[B1]})
CODE_FILES_B2 = tuple(fb1.CODE_FILES_B1) + (
    os.path.join(HERE, "planform_b2.py"),
    os.path.join(HERE, "flexbody_b2.py"),
    os.path.join(HERE, "flexeval_b2.py"),
)
ENERGY_SCHEMA = "fd-energy/2"
ENERGY_REF_RULE = "baseline aircraft (<root>_v2, A1 masses, no B2 increments) trimmed D x V_true at the scenario IC"


def root_v2b2_for(root: str) -> str:
    return fbb2.root_v2b2_for(root)


def model_version(fidelity: str, model: str, root: str, root_v2: Optional[str] = None,
                  weights: Optional[fb.StructWeightsV2] = None) -> str:
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}")
    if fidelity != B2:
        return fb1.model_version(fidelity, model, root, root_v2, weights)
    rv2 = root_v2 or fe.root_v2_for(root)
    rb2 = fbb2.ensure_root_v2b2(model, rv2, root_v2b2_for(root))
    code = []
    for fp in CODE_FILES_B2:
        with open(fp, "rb") as f:
            code.append(f.read())
    geom = fb.geometry_for(model, rv2)
    pw = fw.params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
    extra = {"fidelity": B2, "substeps": SUBSTEPS[B2], "gate": MARGIN_GATE[B2], "params": fbb2.b2_params(model, geom),
             "terms": PRE_TERMS[B2] + RESP_TERMS[B2], "weights": asdict(weights or fb.StructWeightsV2()),
             "genes": [asdict(g) for g in fb.gene_schema(True)], "shape_genes": pb2.shape_params_for_hash(),
             "native_xml": fbb2.native_xml(model, pw), "energy": {"schema": ENERGY_SCHEMA, "ref": ENERGY_REF_RULE,
             "ref_file": _load_ref_file()["aircraft"].get(model), "meter": [fbb2.EnergyMeter.B2_DRAG,
             fbb2.EnergyMeter.DEFICIT_OK_KTS, fbb2.EnergyMeter.SAT_OK_FRAC, fbb2.EnergyMeter.LOW_KTS]}}
    return (f"{B2}:{pb2.B2_TAG}:"
            f"{fe._sha8(code + [json.dumps(extra, sort_keys=True, default=str).encode(), fe.aircraft_files_sha(model, rv2).encode(), fe.aircraft_files_sha(model, rb2).encode()])}")


class FlexHookB2(fb1.FlexHookB1):
    """FlexHookB1 + native B2 properties written in attach() (before IC/trim), the B2 coupler (+ read-only energy
    meter), B2 flown terms / node layout. B2 at default: no properties written, B1 arithmetic."""

    def __init__(self, *a, native_props: Optional[Dict[str, float]] = None, **k):
        super().__init__(*a, **k)
        self.native_props = dict(native_props or {})
        self.energy = fbb2.EnergyMeter(self.dt)

    def attach(self, fdm):
        super().attach(fdm)
        for kk, v in self.native_props.items():
            fdm["flexbody/b2/" + kk] = float(v)

    def _new_coupler(self, mode="twoway"):
        return fbb2.FlexBodyCouplerB2(self.obj, mode=mode, substeps=SUBSTEPS[B2],
                                      energy=self.energy if mode == "twoway" else None)

    def finish(self, status: str) -> Dict:
        if getattr(self.obj, "b2_baseline", True):
            return super().finish(status)
        px = self.px
        if status != "ok" or px is None or not px.started:
            return {"terms": None, "fail": None}
        r = fbb2.response_terms_b2(px.history(), self.obj, px.out_1g, self.wts,
                                   qk_ref=float(getattr(self.coupler, "_qk_ref", 0.0)))
        info = {"tip_max_ft": r["tip_max_ft"], "twist_max_deg": r["twist_max_deg"], "tail_ratio": r["tail_ratio"],
                "fus_ratio": r["fus_ratio"], "torque_ratio": r["torque_ratio"], "ip_ratio": r["ip_ratio"],
                "bm_allow": r["bm_allow"], "m_root_1g": r["m_root_1g"]}
        if "m_root_1g_ref" in r:
            info["m_root_1g_ref"] = r["m_root_1g_ref"]
        return {"terms": r["terms"], "fail": r["fail"], "loads": r["loads"], "info": info}

    def node_telemetry(self):
        if not self.telemetry or self.px is None or not self.px.eta_hist:
            return None
        E = np.vstack(self.px.eta_hist)
        vals = fb.node_values(self.obj, E)
        return {"schema": "fd-flexbody-nodes/1", "frame": "body FRD (x fwd, y right, z down), ft, origin = CG at trim",
                "rp_offset_body_ft": self._rp_offset_ft,
                "components": fbb2.node_layout_b2(self.obj, self._rp_offset_ft),
                "values": {b: {f: np.round(a, 9).tolist() for f, a in fv.items()} for b, fv in vals.items()},
                "doc": "values = own elastic deflection relative to the clamped body root (incl. the 1-g trim shape); "
                       "signs: INTERFACE_v2.md 'Telemetry sign conventions'"}


node_layout = fbb2.node_layout_b2

ENERGY_REF_FILE = os.path.join(HERE, "v2_results", "p3b2a_energy_ref.json")
_EREF: Dict = {}
_EREF_FILE: Optional[Dict] = None


def energy_ref_key(sc) -> str:
    """Trim-relevant IC of a scenario (sim.trim uses h0_ft, speed_kts + per-aircraft profile constants)."""
    return f"h0={float(sc.h0_ft)!r}|kcas={float(sc.speed_kts)!r}"


def _load_ref_file() -> Dict:
    global _EREF_FILE
    if _EREF_FILE is None:
        _EREF_FILE = json.load(open(ENERGY_REF_FILE)) if os.path.exists(ENERGY_REF_FILE) else {"aircraft": {}}
    return _EREF_FILE


def compute_energy_reference(sim, profile_v2, sc, model: str) -> Optional[Dict]:
    """Baseline aircraft (<root>_v2, no flex, no B2) trimmed at the scenario IC: D0, V0, qS0, CD_ref = D0 / qS0,
    P_ref = D0 V0. Gene- and gain-independent. None if the baseline trim fails."""
    try:
        fdm, _ = sim.trim(profile_v2, sc)
    except Exception:      # noqa: BLE001 (SimSetupError)
        return None
    a = fdm["aero/alpha-rad"]
    D = -fdm["forces/fbx-aero-lbs"] * math.cos(a) - fdm["forces/fbz-aero-lbs"] * math.sin(a)
    V, qS = fdm["velocities/vt-fps"], fdm["aero/qbar-area"]
    out = {"h0_ft": float(sc.h0_ft), "speed_kcas": float(sc.speed_kts), "drag_lbf": float(D), "vt_fps": float(V),
           "qbar_area_lbf": float(qS), "cd_ref": float(D / qS), "drag_power_ref_ftlbf_s": float(D * V),
           "alpha_deg": float(math.degrees(a)), "weight_lbs": float(fdm["inertia/weight-lbs"])}
    del fdm
    return out


def energy_reference(sim, profile_v2, sc, model: str) -> Optional[Dict]:
    """Frozen reference from v2_results/p3b2a_energy_ref.json (source='frozen'); a scenario IC not in the file is
    computed with the same rule (source='computed', still gene / gain independent) and cached in-process."""
    key = energy_ref_key(sc)
    fr = _load_ref_file()["aircraft"].get(model, {}).get(key)
    if fr is not None:
        return dict(fr, source="frozen")
    ck = (model, key)
    if ck not in _EREF:
        r = compute_energy_reference(sim, profile_v2, sc, model)
        _EREF[ck] = dict(r, source="computed") if r else None
    return _EREF[ck]


def build_energy_ref_file(models=("c172x", "T38", "737", "f16"), sim=None, root=None) -> Dict:
    """Writes v2_results/p3b2a_energy_ref.json for the Phase-1 scenario ICs of each aircraft (default profile)."""
    import coupled_sim as cs
    sim = sim or fe.load_sim()
    root = root or cs.ROOT
    rv2 = fe.root_v2_for(root)
    doc = {"schema": "fd-p3b2a-energy-ref/1", "rule": ENERGY_REF_RULE, "key": "h0=<h0_ft>|kcas=<speed_kts>",
           "units": {"drag_lbf": "lbf", "vt_fps": "ft/s", "qbar_area_lbf": "lbf", "cd_ref": "-",
                     "drag_power_ref_ftlbf_s": "ft lbf / s", "weight_lbs": "lbf"}, "aircraft": {}}
    for m in models:
        fb.ensure_root_v2(m, rv2)
        P = fe._profile_with_root(fe.default_profile(m, sim), sim, rv2, m)
        doc["aircraft"][m] = {}
        for sc in fe.phase1_scenarios(m, sim):
            r = compute_energy_reference(sim, P, sc, m)
            if r is None:
                raise RuntimeError(f"{m}: baseline trim failed for {energy_ref_key(sc)}")
            doc["aircraft"][m][energy_ref_key(sc)] = r
        doc["aircraft"][m]["_aircraft_files_sha"] = fe.aircraft_files_sha(m, rv2)
    with open(ENERGY_REF_FILE, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    global _EREF_FILE
    _EREF_FILE = None
    return doc


def build_model(struct_genome, model: str, shape_genome=None, root_v2: Optional[str] = None) -> fbb2.FlexBodyModelB2:
    return fbb2.FlexBodyModelB2(model, struct_genome, shape_genes=shape_genome,
                                asymmetric=fe._is_asym(struct_genome), root_v2=root_v2 or fb.ROOT_V2)


def evaluate(gains: Dict[str, float], struct_genome, scenarios, model: str, *, fidelity: str, root: str,
             dt: float = 1 / 120, record: bool = False, profile=None, weights: Optional[fb.StructWeightsV2] = None,
             root_v2: Optional[str] = None, sim=None, blas_threads: Optional[int] = 1,
             shape_genome=None, energy: bool = True) -> Dict:
    """flexeval_b1.evaluate plus fidelity='full_a1_b2a' (shape_genome = merged B1 + B2a dict or [0,1]^11)."""
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}, got {fidelity!r}")
    if fidelity != B2:
        if shape_genome is not None and not isinstance(shape_genome, dict) and len(np.ravel(shape_genome)) == pb2.N_SHAPE_B2:
            raise ValueError(f"an 11-gene B2 vector is only consumed at fidelity={B2!r}")
        if isinstance(shape_genome, dict) and any(k in pb2.B2_INDEX and v != pb2.SHAPE_GENES_B2[pb2.B2_INDEX[k]].default
                                                  for k, v in shape_genome.items()):
            raise ValueError(f"non-default B2 genes are only consumed at fidelity={B2!r} (got {fidelity!r})")
        sg = shape_genome
        if isinstance(sg, dict):
            sg = {k: v for k, v in sg.items() if k not in pb2.B2_INDEX}
        return fb1.evaluate(gains, struct_genome, scenarios, model, fidelity=fidelity, root=root, dt=dt, record=record,
                            profile=profile, weights=weights, root_v2=root_v2, sim=sim, blas_threads=blas_threads,
                            shape_genome=sg)
    sim = sim or fe.load_sim()
    if dt != sim.DT:
        raise ValueError(f"dt={dt} must equal evolution sim.DT={sim.DT}")
    if blas_threads is not None:
        fb.blas_threads(blas_threads)
    prof0 = profile if profile is not None else fe.default_profile(model, sim)
    scs = fe._as_scenarios(scenarios, sim)
    rv2 = root_v2 or fe.root_v2_for(root)
    terms = fe._zero_terms()
    shape = pb2.decode_shape_b2(shape_genome, model)      # raises ValueError (never clips)
    b1s = pb2.b1_part(shape)
    base_b2 = pb2.is_baseline_b2(shape, model)
    out = {"fidelity": fidelity, "margins": None, "margins_fidelity": None, "per_scenario": [],
           "shape_genes": b1s, "shape_cache_key": None, "shape_genes_b2": pb2.b2_part(shape)}
    gate = MARGIN_GATE[fidelity]
    asym = fe._is_asym(struct_genome)
    fb.ensure_root_v2(model, rv2)
    rfly = rv2 if base_b2 else fbb2.ensure_root_v2b2(model, rv2, root_v2b2_for(root))
    P = fe._profile_with_root(prof0, sim, rfly, model)
    out["model_version"] = model_version(fidelity, model, root, rv2, weights)
    wts0 = weights or fb.StructWeightsV2()
    wts = weights or dataclasses.replace(wts0, fail_cost=float(2 * P.fail_base))
    fb.decode_genome_v2(struct_genome, asym)
    geom = fb.geometry_for(model, rv2)
    pw = fw.params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
    gr = pb1.geometry_gate(pw, b1s, n_el=int(fba1.WING_A1["n_el"]))
    out["geometry_gate"] = gr.as_dict()
    env = [(float(sc.speed_kts), float(sc.h0_ft)) for sc in scs]
    out["shape_cache_key"] = pb2.shape_cache_key_b2(shape, model, env)
    gr2 = (pb2.geometry_gate_b2(pw, shape, model, n_el=int(fba1.WING_A1["n_el"]), root_v2=rv2, envelope=env, geom=geom)
           if not base_b2 else pb2.GateB2(True, None, {"b2_baseline": 1.0}))
    out["geometry_gate_b2"] = gr2.as_dict()
    if not gr.ok or not gr2.ok:
        reason = gr.reason if not gr.ok else gr2.reason
        out.update(cost=float(wts.fail_cost), terms=fe._finite(terms), status=f"geometry_gate:{reason}",
                   terms_available=[], margin_gate=gate)
        return out
    ni = pb2.native_increments(model, shape, pw, geom) if not base_b2 else {"props": {}, "info": {}}
    out["native_increments"] = {"props": dict(ni["props"]), "info": dict(ni["info"])}
    obj = fbb2.FlexBodyModelB2(model, struct_genome, shape_genes=shape, asymmetric=asym, root_v2=rv2)
    pre = fbb2.margin_terms_b2(obj, wts0, gate=gate)
    pre_terms, fail = dict(pre["terms"]), pre["fail"]
    out["margins"] = pre["margins"]
    out["mass"] = pre["mass"]
    out["sizing"] = pre["sizing"]
    out["mass_applied_to_fdm"] = "all bodies (wing L/R, HT, VT, aft fuselage point masses)"
    out["structural_model"] = {"variant": "a1_b1", "wing": dict(obj.wing_a1), "modal_dof": int(obj.N)}
    out["planform"] = obj.planform_summary()
    out["b2"] = obj.b2_summary(model)
    out["terms_available"] = ["track", "effort"] + list(PRE_TERMS[fidelity]) + list(RESP_TERMS[fidelity])
    out["margins_fidelity"] = fidelity
    out["margin_gate"] = gate
    for k, v in pre_terms.items():
        terms[k] = float(v)
    pre_sum = float(sum(pre_terms[k] for k in PRE_TERMS[fidelity]))
    if fail:
        out.update(cost=float(wts.fail_cost), terms=fe._finite(terms), status=fail)
        return out
    Pref = fe._profile_with_root(prof0, sim, rv2, model)
    per, telemetry, loads_acc, en = [], [], [], []
    for sc in scs:
        hook = FlexHookB2(fidelity, model, obj, wts, dt, root, rv2, out["model_version"], telemetry=record,
                          native_props=ni["props"])
        hook.energy = fbb2.EnergyMeter(dt, throttle_max=float(P.throttle_max), v_target_kts=None,
                                       b2_drag=not base_b2)
        r, post = fe._simulate_with_hook(sim, gains, sc, P, hook, record)
        entry = {k: r[k] for k in fe._PER_KEYS if k in r}
        entry["sim_cost"] = r["cost"]
        entry["delta_mass_lb_applied"] = hook.delta_mass_lb
        cost_s = r["cost"]
        if r["status"] == "ok" and post.get("terms") is not None:
            entry["struct"] = {k: float(v) for k, v in post["terms"].items()}
            entry.update(post["info"])
            if "loads" in post:
                entry["loads"] = post["loads"]
                loads_acc.append(post["loads"])
            if post["fail"]:
                entry["status"] = post["fail"]
                cost_s = float(wts.fail_cost)
            else:
                cost_s = cost_s + sum(post["terms"].values())
        entry["cost"] = cost_s + pre_sum
        per.append(entry)
        if energy:
            ref = energy_reference(sim, Pref, sc, model)
            e = hook.energy.result(ref)
            e["status"] = entry["status"]
            en.append(e)
        if record:
            telemetry.append({"trajectory": r.get("trajectory"),
                              "structure": {k: v.tolist() for k, v in hook.px.history().items()} if hook.px is not None else None,
                              "nodes": hook.node_telemetry(), "planform": out["planform"]})
    avail_rigid = fe._rigid_terms(per, terms)
    out["terms_available"] = avail_rigid + list(PRE_TERMS[fidelity]) + list(RESP_TERMS[fidelity])
    ok = [e for e in per if e["status"] == "ok" and "struct" in e]
    for k in RESP_TERMS[fidelity]:
        vals = [e["struct"][k] for e in ok if math.isfinite(e["struct"].get(k, math.nan))]
        terms[k] = float(np.mean(vals)) if vals else 0.0
    if loads_acc:
        out["loads"] = {nm: {"peak_abs_lbft": float(max(L[nm]["peak_abs"] for L in loads_acc)),
                             "rms_dev_1g_lbft": float(np.mean([L[nm]["rms_dev_1g"] for L in loads_acc])),
                             "trim_1g_lbft": float(loads_acc[0][nm]["trim_1g"])} for nm in fb.OUT_NAMES}
    cost = float(np.mean([e["cost"] for e in per]))
    out.update(cost=cost, terms=fe._finite(terms), per_scenario=per,
               status=next((e["status"] for e in per if e["status"] != "ok"), "ok"))
    if energy:
        okE = [e for e in en if e["status"] == "ok"]

        def mean(k):
            v = [float(e[k]) for e in okE if math.isfinite(float(e[k]))]
            return float(np.mean(v)) if v else float("nan")
        out["energy"] = {"schema": ENERGY_SCHEMA, "reference": ENERGY_REF_RULE, "per_scenario": en, "in_cost": False,
                         "primary": "energy_drag_increment",
                         **{k: mean(k) for k in ("energy_drag_increment", "drag_increment_cd", "energy_drag_ratio",
                                                 "throttle_mean", "throttle_sat_frac", "fuel_burned_lbs",
                                                 "drag_work_ftlbf", "speed_deficit_kts_mean", "speed_low_frac")},
                         "speed_hold_ok": bool(okE) and all(e["speed_hold_ok"] for e in okE),
                         "n_ok": len(okE), "n_scenarios": len(en)}
    if record:
        out["telemetry"] = telemetry
    return out


phase1_scenarios = fe.phase1_scenarios
