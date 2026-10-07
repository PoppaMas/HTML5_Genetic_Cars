"""flexeval_a1.py -- P3-A1 opt-in fidelity 'full_a1' (denser full structural model, flexbody_a1.py) for evaluate().

    import flexeval_a1 as fa
    fa.evaluate(gains, struct_genome, scenarios, model, fidelity='full_a1', root=...)   # A1
    fa.evaluate(..., fidelity='rigid' | 'reduced' | 'full')   # delegates UNCHANGED to flexeval.evaluate
    fa.model_version('full_a1', model, root)  -> 'full_a1:flexv2a1:<sha8>'
    fa.model_version('rigid' | 'reduced' | 'full', ...) -> flexeval.model_version (byte-identical strings)

Why a separate module: flexeval.py / flexbody.py / flexwing.py (/ coupled_sim.py) are hashed into the pinned reduced and
full model_version strings (v2_results/model_versions_post_p25.json); editing them would change those strings. A1 is
therefore added in new files only and the existing fidelities run exactly the code they ran before.

'full_a1' contract = 'full' contract (INTERFACE_v2.md §5, §7, §12) with the A1 model: same 24 TERM_KEYS, same pre-flight
(margins, J_mass, J_smooth, the 6 sizing terms) and post-flight terms, same gate 1.0, same prepared root <root>_v2, same 2
Newmark substeps, same hook protocol and telemetry (nodes: 65 per wing instead of 33). Only difference in the terms:
J_wing_tip_bm_limit is evaluated station-exactly at eta 0.875 (flexbody_a1.sizing_a1; see flexbody_a1 module doc).
Extra output key: 'structural_model' {variant, wing mesh/modes, modal_dof}.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np

import flexbody as fb
import flexeval as fe
import flexbody_a1 as fba1

HERE = os.path.dirname(os.path.abspath(__file__))
A1 = fba1.A1_FIDELITY                                    # 'full_a1'
FIDELITIES = fe.FIDELITIES + (A1,)
TERM_KEYS = fe.TERM_KEYS                                 # unchanged (24 keys)
PRE_TERMS = dict(fe.PRE_TERMS, **{A1: fe.PRE_TERMS["full"]})
RESP_TERMS = dict(fe.RESP_TERMS, **{A1: fe.RESP_TERMS["full"]})
SUBSTEPS = dict(fe.SUBSTEPS, **{A1: fe.SUBSTEPS["full"]})
MARGIN_GATE = dict(fe.MARGIN_GATE, **{A1: fe.MARGIN_GATE["full"]})
# code that defines the A1 numbers (hashed into its model_version); tests monkeypatch this
CODE_FILES_A1 = tuple(fe.CODE_FILES["full"]) + (os.path.join(HERE, "flexbody_a1.py"), os.path.join(HERE, "flexeval_a1.py"))


def model_version(fidelity: str, model: str, root: str, root_v2: Optional[str] = None,
                  weights: Optional[fb.StructWeightsV2] = None) -> str:
    """'full_a1' -> full_a1:flexv2a1:<sha8> over flexbody.py, flexwing.py, flexeval.py, flexbody_a1.py, flexeval_a1.py,
    the A1 parameters (v2 parameters with the A1 wing mesh / mode set), terms, weights, gene schema, substeps, gate and the
    prepared aircraft files. Other fidelities: flexeval.model_version unchanged."""
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}")
    if fidelity != A1:
        return fe.model_version(fidelity, model, root, root_v2, weights)
    rv2 = root_v2 or fe.root_v2_for(root)
    code = []
    for fp in CODE_FILES_A1:
        with open(fp, "rb") as f:
            code.append(f.read())
    geom = fb.geometry_for(model, rv2)
    extra = {"fidelity": A1, "substeps": SUBSTEPS[A1], "gate": MARGIN_GATE[A1], "params": fba1.a1_params(model, geom),
             "terms": PRE_TERMS[A1] + RESP_TERMS[A1], "weights": asdict(weights or fb.StructWeightsV2()),
             "genes": [asdict(g) for g in fb.gene_schema(True)]}
    return f"{A1}:{fba1.A1_TAG}:{fe._sha8(code + [json.dumps(extra, sort_keys=True, default=str).encode(), fe.aircraft_files_sha(model, rv2).encode()])}"


class FlexHookA1(fe.FlexHookV2):
    """flexeval.FlexHookV2 for the A1 model: identical 'full' behaviour (v2 mass feedback, FlexBodyCoupler with the full
    substeps, FlexBodyFDM, response_terms_v2); only the node-telemetry gate also accepts fidelity 'full_a1'."""

    def node_telemetry(self):
        if not self.telemetry or self.px is None or not self.px.eta_hist:
            return None
        E = np.vstack(self.px.eta_hist)
        vals = fb.node_values(self.obj, E)
        return {"schema": "fd-flexbody-nodes/1", "frame": "body FRD (x fwd, y right, z down), ft, origin = CG at trim",
                "rp_offset_body_ft": self._rp_offset_ft,
                "components": fb.node_layout(self.obj, self._rp_offset_ft),
                "values": {b: {f: np.round(a, 9).tolist() for f, a in fv.items()} for b, fv in vals.items()},
                "doc": "values = own elastic deflection relative to the clamped body root (incl. the 1-g trim shape); "
                       "signs: INTERFACE_v2.md 'Telemetry sign conventions'"}


def build_model(struct_genome, model: str, root_v2: Optional[str] = None) -> fba1.FlexBodyModelA1:
    """The A1 structural model of one genome (as evaluate(fidelity='full_a1') builds it)."""
    return fba1.FlexBodyModelA1(model, struct_genome, asymmetric=fe._is_asym(struct_genome), root_v2=root_v2 or fb.ROOT_V2)


def evaluate(gains: Dict[str, float], struct_genome, scenarios, model: str, *, fidelity: str, root: str,
             dt: float = 1 / 120, record: bool = False, profile=None, weights: Optional[fb.StructWeightsV2] = None,
             root_v2: Optional[str] = None, sim=None, blas_threads: Optional[int] = 1) -> Dict:
    """flexeval.evaluate plus fidelity='full_a1'. Rigid / reduced / full are passed straight to flexeval.evaluate (same
    code, same outputs). 'full_a1' follows flexeval.evaluate's 'full' branch line by line with the A1 model."""
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}, got {fidelity!r}")
    if fidelity != A1:
        return fe.evaluate(gains, struct_genome, scenarios, model, fidelity=fidelity, root=root, dt=dt, record=record,
                           profile=profile, weights=weights, root_v2=root_v2, sim=sim, blas_threads=blas_threads)
    sim = sim or fe.load_sim()
    if dt != sim.DT:
        raise ValueError(f"dt={dt} must equal evolution sim.DT={sim.DT} (the Runner's frame time)")
    if blas_threads is not None:
        fb.blas_threads(blas_threads)
    prof0 = profile if profile is not None else fe.default_profile(model, sim)
    scs = fe._as_scenarios(scenarios, sim)
    rv2 = root_v2 or fe.root_v2_for(root)
    terms = fe._zero_terms()
    out = {"fidelity": fidelity, "margins": None, "margins_fidelity": None, "per_scenario": []}
    gate = MARGIN_GATE[fidelity]
    asym = fe._is_asym(struct_genome)
    fb.ensure_root_v2(model, rv2)
    P = fe._profile_with_root(prof0, sim, rv2, model)
    out["model_version"] = model_version(fidelity, model, root, rv2, weights)
    obj = fba1.FlexBodyModelA1(model, struct_genome, asymmetric=asym, root_v2=rv2)
    wts0 = weights or fb.StructWeightsV2()
    pre = fba1.margin_terms_a1(obj, wts0, gate=gate)      # = margin_terms_v2 with the station-exact tip check
    pre_terms, fail = dict(pre["terms"]), pre["fail"]
    out["margins"] = pre["margins"]
    out["mass"] = pre["mass"]
    out["sizing"] = pre["sizing"]
    out["mass_applied_to_fdm"] = "all bodies (wing L/R, HT, VT, aft fuselage point masses)"
    out["structural_model"] = {"variant": "a1", "wing": dict(obj.wing_a1), "modal_dof": int(obj.N)}
    wts = weights or dataclasses.replace(wts0, fail_cost=float(2 * P.fail_base))
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
        hook = FlexHookA1(fidelity, model, obj, wts, dt, root, rv2, out["model_version"], telemetry=record)
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
                              "nodes": hook.node_telemetry()})
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
