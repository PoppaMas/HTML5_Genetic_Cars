"""FD nodal values for a replay flight: the same FE node layout / values FD exports with ``record=True``
(``telemetry[i]['nodes']``, schema fd-flexbody-nodes/1), computed from the modal state ``eta`` that ER's FlexState
hands to the recorder at every call. Sim Bridge side only (v2_map stays standalone).

FD and ER are imported read-only (no bytecode). The structural model is built exactly as ER builds the one that
flies (evolution.fidelity: struct_from -> roots -> _struct_obj), so eta and the mode shapes belong together; the
replay checks that the last node equals FD's tip scalar on every recorded frame (``tip_check``).

    src = NodeSource.for_genome(genome, aircraft, run_cfg)      # full fidelity only
    layout = src.layout(rp_offset_body_ft)                        # == flexbody.node_layout(mdl, rp_offset)
    frame = src.frame(eta)                                        # {body: {w_ft, theta_deg, v_ft: [..]}}
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional

NODES_SCHEMA = "fd-flexbody-nodes/1"
_CACHE: Dict[tuple, "NodeSource"] = {}

# (body, field) -> FD tip scalar that equals the last node (FD: "the last node equals the tip scalar (tested)")
TIP_EQUALS = {("wingR", "w_ft"): "tip_w_ft_R", ("wingL", "w_ft"): "tip_w_ft_L", ("wingR", "theta_deg"): "tip_twist_R_deg",
              ("wingL", "theta_deg"): "tip_twist_L_deg", ("wingR", "v_ft"): "wingR_tip_ip_ft", ("wingL", "v_ft"): "wingL_tip_ip_ft",
              ("htR", "w_ft"): "ht_tip_w_ft", ("htL", "w_ft"): "htL_tip_w_ft", ("htR", "theta_deg"): "ht_tip_twist_deg",
              ("htL", "theta_deg"): "htL_tip_twist_deg", ("vt", "w_ft"): "vt_tip_w_ft", ("vt", "theta_deg"): "vt_tip_twist_deg",
              ("fusV", "w_ft"): "fusV_tip_w_ft", ("fusL", "w_ft"): "fusL_tip_w_ft"}


def rp_offset_ft(fdm) -> List[float]:
    """AERORP relative to the CG, body FRD ft (what FD's FlexHookV2 reads after trim)."""
    return [-(fdm["metrics/aero-rp-x-in"] - fdm["inertia/cg-x-in"]) / 12.0,
            (fdm["metrics/aero-rp-y-in"] - fdm["inertia/cg-y-in"]) / 12.0,
            -(fdm["metrics/aero-rp-z-in"] - fdm["inertia/cg-z-in"]) / 12.0]


class NodeSource:
    def __init__(self, mdl, fb, key=None, model_version: Optional[str] = None):
        self.mdl, self.fb, self.key, self.model_version = mdl, fb, key, model_version

    @classmethod
    def for_genome(cls, genome: Dict, aircraft: str, run_cfg: Dict, model_version: Optional[str] = None) -> "NodeSource":
        """FD FlexBodyModel for this genome, built like ER's evaluate builds it. Cached per (aircraft, struct genes,
        aircraft_root, model_version): a new FD model_version never reuses a cached model."""
        sys.dont_write_bytecode = True
        from evolution import eval as E, fidelity as F, sim
        entry = E._aircraft_entry(run_cfg, aircraft)
        prof = dict(E._profile_d(entry))
        _gains, struct = E.split_values(genome, E.gene_groups(entry))
        P = sim.Profile.from_dict(prof)
        _root, rv2 = F.roots(P)
        key = (aircraft, tuple(sorted((struct or {}).items())), rv2, model_version)
        if key not in _CACHE:
            m = F.fd_modules()
            # Prefer public make_fd_model (ER 2026-10-06); fall back to private _struct_obj for older ER
            if hasattr(F, "make_fd_model"):
                obj = F.make_fd_model(prof, struct, "full")
            else:
                obj = F._struct_obj(m["fe"], m["fb"], F.struct_from(struct), aircraft, "full", rv2)
            _CACHE[key] = cls(obj, m["fb"], key, model_version)
        return _CACHE[key]

    def layout(self, rp_offset_body_ft=None) -> List[Dict]:
        return self.fb.node_layout(self.mdl, rp_offset_body_ft)

    def frame(self, eta) -> Dict[str, Dict[str, List[float]]]:
        vals = self.fb.node_values(self.mdl, eta)
        return {b: {f: [float(v) for v in a] for f, a in fv.items()} for b, fv in vals.items()}


def tip_mismatch(frame: Dict, raw: Dict) -> float:
    """max |last node - FD tip scalar| over the (body, field) pairs present in both (FD units)."""
    worst = 0.0
    for (b, f), k in TIP_EQUALS.items():
        if b in frame and f in frame[b] and k in raw:
            worst = max(worst, abs(frame[b][f][-1] - float(raw[k])))
    return worst
