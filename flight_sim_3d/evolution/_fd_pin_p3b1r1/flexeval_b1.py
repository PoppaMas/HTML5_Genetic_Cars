"""flexeval_b1.py -- P3-B1 opt-in fidelity 'full_a1_b1' (A1 mesh + planform shape genes).

    import flexeval_b1 as fb1
    fb1.evaluate(gains, struct_genome, scenarios, model, fidelity='full_a1_b1', root=...,
                 shape_genome={...} | None)
    fb1.evaluate(..., fidelity='rigid'|'reduced'|'full'|'full_a1')  # delegates to flexeval_a1 / flexeval

Baseline shape (None / {} / all defaults) + structure genes -> same preflight/flight as full_a1 (bit-intent; see
v2_results/p3b1_acceptance.json). Geometry gate reject -> status 'geometry_gate', cost = fail_cost, no flight.

model_version('full_a1_b1', ...) -> 'full_a1_b1:flexv2b1:<sha8>' over A1 code files + planform_b1.py + flexbody_b1.py +
flexeval_b1.py + shape schema + A1/B1 params. Per-genome shape values are inputs, not part of the version.
rigid / reduced / full / full_a1 strings are unchanged (those modules are not edited).
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
import flexbody_a1 as fba1
import flexbody_b1 as fbb1
import planform_b1 as pb1

HERE = os.path.dirname(os.path.abspath(__file__))
B1 = pb1.B1_FIDELITY  # 'full_a1_b1'
FIDELITIES = fa.FIDELITIES + (B1,)
TERM_KEYS = fe.TERM_KEYS
PRE_TERMS = dict(fa.PRE_TERMS, **{B1: fa.PRE_TERMS[fa.A1]})
RESP_TERMS = dict(fa.RESP_TERMS, **{B1: fa.RESP_TERMS[fa.A1]})
SUBSTEPS = dict(fa.SUBSTEPS, **{B1: fa.SUBSTEPS[fa.A1]})
MARGIN_GATE = dict(fa.MARGIN_GATE, **{B1: fa.MARGIN_GATE[fa.A1]})
CODE_FILES_B1 = tuple(fa.CODE_FILES_A1) + (
    os.path.join(HERE, "planform_b1.py"),
    os.path.join(HERE, "flexbody_b1.py"),
    os.path.join(HERE, "flexeval_b1.py"),
)


def model_version(fidelity: str, model: str, root: str, root_v2: Optional[str] = None,
                  weights: Optional[fb.StructWeightsV2] = None) -> str:
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}")
    if fidelity != B1:
        return fa.model_version(fidelity, model, root, root_v2, weights)
    rv2 = root_v2 or fe.root_v2_for(root)
    code = []
    for fp in CODE_FILES_B1:
        with open(fp, "rb") as f:
            code.append(f.read())
    geom = fb.geometry_for(model, rv2)
    extra = {"fidelity": B1, "substeps": SUBSTEPS[B1], "gate": MARGIN_GATE[B1],
             "params": fbb1.b1_params(model, geom),
             "terms": PRE_TERMS[B1] + RESP_TERMS[B1],
             "weights": asdict(weights or fb.StructWeightsV2()),
             "genes": [asdict(g) for g in fb.gene_schema(True)],
             "shape_genes": pb1.shape_params_for_hash()}
    return (f"{B1}:{pb1.B1_TAG}:"
            f"{fe._sha8(code + [json.dumps(extra, sort_keys=True, default=str).encode(), fe.aircraft_files_sha(model, rv2).encode()])}")


class FlexHookB1(fa.FlexHookA1):
    """flexeval_a1.FlexHookA1 with the B1 coupler (basic twist load) and B1 flown terms. For a baseline shape both reduce
    to the A1 arithmetic exactly (FlexBodyCouplerB1 delegates; response_terms_b1 = flexbody.response_terms_v2)."""

    def _new_coupler(self, mode="twoway"):
        return fbb1.FlexBodyCouplerB1(self.obj, mode=mode, substeps=SUBSTEPS[B1])

    def finish(self, status: str) -> Dict:
        px = self.px
        if status != "ok" or px is None or not px.started:
            return {"terms": None, "fail": None}
        r = fbb1.response_terms_b1(px.history(), self.obj, px.out_1g, self.wts,
                                   qk_ref=float(getattr(self.coupler, "_qk_ref", 0.0)))
        info = {"tip_max_ft": r["tip_max_ft"], "twist_max_deg": r["twist_max_deg"], "tail_ratio": r["tail_ratio"],
                "fus_ratio": r["fus_ratio"], "torque_ratio": r["torque_ratio"], "ip_ratio": r["ip_ratio"],
                "bm_allow": r["bm_allow"], "m_root_1g": r["m_root_1g"]}
        if "m_root_1g_ref" in r:                                  # shaped planform only (baseline = A1 keys exactly)
            info["m_root_1g_ref"] = r["m_root_1g_ref"]
        return {"terms": r["terms"], "fail": r["fail"], "loads": r["loads"], "info": info}

    def node_telemetry(self):
        """FlexHookA1.node_telemetry with the B1 node layout (shaped wing nodes; baseline shape -> identical)."""
        if not self.telemetry or self.px is None or not self.px.eta_hist:
            return None
        E = np.vstack(self.px.eta_hist)
        vals = fb.node_values(self.obj, E)
        return {"schema": "fd-flexbody-nodes/1", "frame": "body FRD (x fwd, y right, z down), ft, origin = CG at trim",
                "rp_offset_body_ft": self._rp_offset_ft,
                "components": fbb1.node_layout_b1(self.obj, self._rp_offset_ft),
                "values": {b: {f: np.round(a, 9).tolist() for f, a in fv.items()} for b, fv in vals.items()},
                "doc": "values = own elastic deflection relative to the clamped body root (incl. the 1-g trim shape); "
                       "signs: INTERFACE_v2.md 'Telemetry sign conventions'"}


node_layout = fbb1.node_layout_b1


def build_model(struct_genome, model: str, shape_genome=None, root_v2: Optional[str] = None) -> fbb1.FlexBodyModelB1:
    return fbb1.FlexBodyModelB1(model, struct_genome, shape_genes=shape_genome,
                                asymmetric=fe._is_asym(struct_genome), root_v2=root_v2 or fb.ROOT_V2)


def evaluate(gains: Dict[str, float], struct_genome, scenarios, model: str, *, fidelity: str, root: str,
             dt: float = 1 / 120, record: bool = False, profile=None, weights: Optional[fb.StructWeightsV2] = None,
             root_v2: Optional[str] = None, sim=None, blas_threads: Optional[int] = 1,
             shape_genome=None) -> Dict:
    """flexeval_a1.evaluate plus fidelity='full_a1_b1' and shape_genome=.

    shape_genome: None / {} / defaults = baseline planform (A1 bit-intent). Dict or [0,1]^6 as planform_b1.decode_shape_b1.
    Geometry gate failure returns status 'geometry_gate:<reason>', cost=fail_cost, no flight. TERM_KEYS stay 24.
    """
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}, got {fidelity!r}")
    if fidelity != B1:
        if shape_genome is not None and not pb1.is_baseline_shape(shape_genome):
            raise ValueError(f"shape_genome is only consumed at fidelity={B1!r} (got {fidelity!r})")
        return fa.evaluate(gains, struct_genome, scenarios, model, fidelity=fidelity, root=root, dt=dt, record=record,
                           profile=profile, weights=weights, root_v2=root_v2, sim=sim, blas_threads=blas_threads)
    sim = sim or fe.load_sim()
    if dt != sim.DT:
        raise ValueError(f"dt={dt} must equal evolution sim.DT={sim.DT}")
    if blas_threads is not None:
        fb.blas_threads(blas_threads)
    prof0 = profile if profile is not None else fe.default_profile(model, sim)
    scs = fe._as_scenarios(scenarios, sim)
    rv2 = root_v2 or fe.root_v2_for(root)
    terms = fe._zero_terms()
    shape = pb1.decode_shape_b1(shape_genome)             # raises ValueError (never clips), like the struct genome
    out = {"fidelity": fidelity, "margins": None, "margins_fidelity": None, "per_scenario": [],
           "shape_genes": shape, "shape_cache_key": pb1.shape_cache_key(shape)}
    gate = MARGIN_GATE[fidelity]
    asym = fe._is_asym(struct_genome)
    fb.ensure_root_v2(model, rv2)
    P = fe._profile_with_root(prof0, sim, rv2, model)
    out["model_version"] = model_version(fidelity, model, root, rv2, weights)
    wts0 = weights or fb.StructWeightsV2()
    wts = weights or dataclasses.replace(wts0, fail_cost=float(2 * P.fail_base))
    fb.decode_genome_v2(struct_genome, asym)              # validate the structure block before the gate, as full_a1
    geom = fb.geometry_for(model, rv2)
    pw = fw.params_for(model, geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
    gr = pb1.geometry_gate(pw, shape, n_el=int(fba1.WING_A1["n_el"]))
    out["geometry_gate"] = gr.as_dict()
    if not gr.ok:                                          # cheap reject: not built, not flown, hard cost
        out.update(cost=float(wts.fail_cost), terms=fe._finite(terms), status=f"geometry_gate:{gr.reason}",
                   terms_available=[], margin_gate=gate)
        return out
    obj = fbb1.FlexBodyModelB1(model, struct_genome, shape_genes=shape, asymmetric=asym, root_v2=rv2)
    pre = fbb1.margin_terms_b1(obj, wts0, gate=gate)       # = margin_terms_a1 for a baseline shape
    pre_terms, fail = dict(pre["terms"]), pre["fail"]
    out["margins"] = pre["margins"]
    out["mass"] = pre["mass"]
    out["sizing"] = pre["sizing"]
    out["mass_applied_to_fdm"] = "all bodies (wing L/R, HT, VT, aft fuselage point masses)"
    out["structural_model"] = {"variant": "a1_b1", "wing": dict(obj.wing_a1), "modal_dof": int(obj.N)}
    out["planform"] = obj.planform_summary()
    out["terms_available"] = ["track", "effort"] + list(PRE_TERMS[fidelity]) + list(RESP_TERMS[fidelity])
    out["margins_fidelity"] = fidelity
    out["margin_gate"] = gate
    for k, v in pre_terms.items():
        terms[k] = float(v)
    pre_sum = float(sum(pre_terms[k] for k in PRE_TERMS[fidelity]))
    if fail:
        out.update(cost=float(wts.fail_cost), terms=fe._finite(terms), status=fail)
        return out
    per, telemetry, loads_acc = [], [], []
    for sc in scs:
        hook = FlexHookB1(fidelity, model, obj, wts, dt, root, rv2, out["model_version"], telemetry=record)
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
        if record:
            telemetry.append({"trajectory": r.get("trajectory"),
                              "structure": {k: v.tolist() for k, v in hook.px.history().items()} if hook.px is not None else None,
                              "nodes": hook.node_telemetry(),
                              "planform": out["planform"]})
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
    if record:
        out["telemetry"] = telemetry
    return out


phase1_scenarios = fe.phase1_scenarios
