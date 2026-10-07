"""FD v2 (flexbody) telemetry -> Sim Bridge trajectory channels. Standalone: stdlib only; imports nothing from
evolution/ and, unless asked for estimated geometry, nothing from flight-dynamics/. ER imports this module
read-only (tail, fin, fuselage and the struct.* scalars; ER keeps its own modal wing mapping).

PUBLIC API (stable; V2_MAP_VERSION is bumped on any change to names, signs or units)
-------------------------------------------------------------------------------------
    V2_MAP_VERSION                      "2.0.0"
    geometry_from_layout(layout)        FD node layout -> geometry (exact node positions, body FRD m, origin CG).
                                        layout = flexbody.node_layout(mdl, rp_offset_body_ft) (list), or an
                                        fd-flexbody-nodes/1 doc (telemetry[i]['nodes']); ft -> m.
    geometry_estimated(aircraft, ...)   display geometry when no FD layout exists (FD meta / profiles / generic).
    map_v2_record(telemetry_row, geometry, nodes=None, *, components=None) -> {channel: float}
                                        ONE frame. telemetry_row = FD scalars (coupler.last, a structure history
                                        row, or flex.<key> channels). nodes = that frame's FD nodal values
                                        {body: {"w_ft"|"theta_deg"|"v_ft": [per node]}} (nodes_frame() extracts
                                        frame k from an fd-flexbody-nodes/1 doc or from flexbody.node_values).
                                        Without nodes, node values are ESTIMATED from the tip scalars (assumed
                                        shapes below). Output: "struct.<name>" SI scalars + "<component>.<dof>.<i>".
    structure_block(geometry, *, components=None, estimated=..., base=None, synthetic=False) -> dict
                                        the trajectory's "structure" block (axis_nodes_body_m, dof, per-component
                                        "estimated" flag and source text, sign table).
    nodes_frame(values, k=None)         FD values (per-frame lists or one frame of arrays) -> one frame of lists.
    component_status(geometry, nodes, components=None) -> {component: "fd_nodes" | "estimated"}
    SIGN_TABLE, SCALARS                 the mapping as data (README / tests / ER docs).
    validate_structure(structure, channel_names) -> [problems]   convert_traj(doc, nodes_doc=None) -> doc

Viewer convention (body FRD: x fwd, y right, z down; metres, radians)
  dz + down, dy + right, dx + forward; twist = right-hand rotation about the component polyline direction
  node i -> i+1 (wings root -> tip; htail left tip -> right tip; vtail root -> tip = upward; fuselage wing station
  -> tail). FD sign conventions: flight-dynamics/INTERFACE_v2.md section 11 (probed: v2_results/sign_probe.json).

Components (FD bodies -> ours)
  wingR / wingL  <- wingR / wingL      dz = -w, dx = -v (in-plane, FD + aft), twist = +theta_R / -theta_L
  htail          <- htL (reversed) + htR, one polyline left tip -> right tip. dz = -(w_own + w_fusV,tip)
                    - ht_incidence * (x - x_tail); dy = w_fusL,tip + vt_sideslip * (x - x_tail) (the tail rides on
                    the aft-fuselage tip: plunge + slope rotation about the fuselage tip x_tail); twist = +theta
                    (ELASTIC only; ht_incidence stays in struct.ht_incidence)
  vtail          <- vt, root -> tip.  dy = w_own + w_fusL,tip + vt_sideslip * (x - x_tail);
                    dz = -w_fusV,tip - ht_incidence * (x - x_tail); twist = -theta (FD + = LE toward +y)
  fuselage       <- fusV (dz = -w) + fusL (dy = +w), wing station -> tail (forward fuselage rigid)
  (all lengths ft -> m: x 0.3048; angles deg -> rad)

Estimated node values (no nodal data): value(xi) = tip value * shape(xi), xi = node span fraction from the clamped
root; bending / in-plane phi(xi) = xi^2 (3 - xi) / 2 (cantilever tip load), torsion psi(xi) = xi (2 - xi)
(uniform torque). Flagged per component ("estimated": true) in the structure block and in the HUD.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence

V2_MAP_VERSION = "2.0.0"
SCHEMA = "sim-bridge-v2-map/2"
STRUCTURE_SCHEMA = "sim-bridge-structure/2"
GEOMETRY_SCHEMA = "sim-bridge-v2-geometry/2"
NODES_SCHEMA = "fd-flexbody-nodes/1"
FT = 0.3048
LBF = 4.4482216152605
LBFFT = LBF * FT          # 1.3558179483314004 N*m per lbf*ft
DEG = math.pi / 180.0
SYNTHETIC_DOC = "SYNTHETIC: built from the documented FD v2 format, not from an FD run"
COMPONENTS = ("wingR", "wingL", "htail", "vtail", "fuselage")
FD_BODIES = ("wingR", "wingL", "htR", "htL", "vt", "fusV", "fusL")
DOFS = ("dz", "dy", "dx", "twist")

# FD scalar -> (Sim Bridge scalar, factor, unit, meaning of + in Sim Bridge terms). Factor sign = the convention change.
SCALARS: Dict[str, tuple] = {
    # root loads (FD sign kept: + bends the tip toward the body's + direction; torque + nose-up)
    "wingR_bm": ("struct.wingR_root_bm", LBFFT, "N*m", "root bending, + = up-bending (tip toward body -z)"),
    "wingL_bm": ("struct.wingL_root_bm", LBFFT, "N*m", "root bending, + = up-bending (tip toward body -z)"),
    "wingR_torque": ("struct.wingR_root_torque", LBFFT, "N*m", "root torque about the EA, + = nose-up (LE up)"),
    "wingL_torque": ("struct.wingL_root_torque", LBFFT, "N*m", "root torque about the EA, + = nose-up (LE up)"),
    "wingR_ip_bm": ("struct.wingR_root_ip_bm", LBFFT, "N*m", "in-plane root bending, + = aft load (tip bends aft)"),
    "wingL_ip_bm": ("struct.wingL_root_ip_bm", LBFFT, "N*m", "in-plane root bending, + = aft load (tip bends aft)"),
    "htR_bm": ("struct.htR_root_bm", LBFFT, "N*m", "right HT root bending, + = up-bending (perturbation from trim)"),
    "htL_bm": ("struct.htL_root_bm", LBFFT, "N*m", "left HT root bending, + = up-bending (perturbation from trim)"),
    "vt_bm": ("struct.vt_root_bm", LBFFT, "N*m", "fin root bending, + = load / tip toward body +y"),
    "fusV_bm": ("struct.fusV_root_bm", LBFFT, "N*m", "aft-fuselage vertical bending at the wing station, + = tail up"),
    "fusL_bm": ("struct.fusL_root_bm", LBFFT, "N*m", "aft-fuselage lateral bending at the wing station, + = tail +y"),
    # tip deflections / twists in the viewer convention (relative to the body's clamped root)
    "tip_w_ft_R": ("struct.wingR_tip_dz", -FT, "m", "right wing tip, body z + down (FD w + up)"),
    "tip_w_ft_L": ("struct.wingL_tip_dz", -FT, "m", "left wing tip, body z + down (FD w + up)"),
    "tip_twist_R_deg": ("struct.wingR_tip_twist", DEG, "rad", "right-hand root->tip (+y): + = LE up"),
    "tip_twist_L_deg": ("struct.wingL_tip_twist", -DEG, "rad", "right-hand root->tip (-y): + = LE DOWN (FD + = LE up)"),
    "wingR_tip_ip_ft": ("struct.wingR_tip_dx", -FT, "m", "right wing tip in-plane, body x + forward (FD + aft)"),
    "wingL_tip_ip_ft": ("struct.wingL_tip_dx", -FT, "m", "left wing tip in-plane, body x + forward (FD + aft)"),
    "ht_tip_w_ft": ("struct.htR_tip_dz", -FT, "m", "right HT tip rel. its root (excl. fuselage), body z + down"),
    "htL_tip_w_ft": ("struct.htL_tip_dz", -FT, "m", "left HT tip rel. its root (excl. fuselage), body z + down"),
    "ht_tip_twist_deg": ("struct.htR_tip_twist", DEG, "rad", "right HT elastic twist, + = LE up (htail polyline +y)"),
    "htL_tip_twist_deg": ("struct.htL_tip_twist", DEG, "rad", "left HT elastic twist, + = LE up (htail polyline +y)"),
    "vt_tip_w_ft": ("struct.vt_tip_dy", FT, "m", "fin tip rel. its root (excl. fuselage), body y + right"),
    "vt_tip_twist_deg": ("struct.vt_tip_twist", -DEG, "rad", "fin elastic twist, right-hand about root->tip (up): "
                                                             "+ = LE toward -y (FD + = LE toward +y)"),
    "fusV_tip_w_ft": ("struct.fusV_tip_dz", -FT, "m", "aft-fuselage tip (tail) vertical, body z + down"),
    "fusL_tip_w_ft": ("struct.fusL_tip_dy", FT, "m", "aft-fuselage tip (tail) lateral, body y + right"),
    "ht_incidence_deg": ("struct.ht_incidence", DEG, "rad", "tail incidence from the fuselage slope, + = nose-up "
                                                            "(raises tail alpha; = -w'_fusV,tip)"),
    "vt_sideslip_deg": ("struct.vt_incidence", DEG, "rad", "fin incidence from the fuselage slope, + = fin LE toward "
                                                           "+y (= -w'_fusL,tip; OPPOSITE sense to aircraft beta)"),
    # elastic feedback to JSBSim (forces at the AERORP, moments about the AERORP; JSBSim transfers to the CG)
    "dL_lbf": ("struct.elastic_dlift", LBF, "N", "elastic lift increment, + = up, normal to V in the body x-z plane"),
    "dY_lbf": ("struct.elastic_dside", LBF, "N", "elastic side force, + = body +y"),
    "dRoll_lbft": ("struct.elastic_droll", LBFFT, "N*m", "elastic rolling moment (JSBSim l), + = right wing down"),
    "dPitch_lbft": ("struct.elastic_dpitch", LBFFT, "N*m", "elastic pitching moment (JSBSim m), + = nose up"),
    "dYaw_lbft": ("struct.elastic_dyaw", LBFFT, "N*m", "elastic yawing moment (JSBSim n), + = nose right"),
}
# derived scalar: the fin incidence expressed as an equivalent local sideslip (aircraft beta sense)
DERIVED = {"struct.vt_sideslip_equiv": ("vt_sideslip_deg", -DEG, "rad",
                                        "equivalent local sideslip increment at the fin, aircraft-beta sense (= -vt_sideslip)")}
RAW_KEYS = tuple(SCALARS)
DIAG_KEYS_V2_COUNT, TELEMETRY_KEYS_V2_COUNT = 26, 6

# nodal fields: FD body -> {field: (our component, our dof, factor)}
NODE_FIELDS = {
    "wingR": {"w_ft": ("wingR", "dz", -FT), "v_ft": ("wingR", "dx", -FT), "theta_deg": ("wingR", "twist", DEG)},
    "wingL": {"w_ft": ("wingL", "dz", -FT), "v_ft": ("wingL", "dx", -FT), "theta_deg": ("wingL", "twist", -DEG)},
    "htR": {"w_ft": ("htail", "dz", -FT), "theta_deg": ("htail", "twist", DEG)},
    "htL": {"w_ft": ("htail", "dz", -FT), "theta_deg": ("htail", "twist", DEG)},
    "vt": {"w_ft": ("vtail", "dy", FT), "theta_deg": ("vtail", "twist", -DEG)},
    "fusV": {"w_ft": ("fuselage", "dz", -FT)},
    "fusL": {"w_ft": ("fuselage", "dy", FT)},
}
# tip scalar per (body, field) for the estimate
TIP_KEYS = {("wingR", "w_ft"): "tip_w_ft_R", ("wingR", "theta_deg"): "tip_twist_R_deg", ("wingR", "v_ft"): "wingR_tip_ip_ft",
            ("wingL", "w_ft"): "tip_w_ft_L", ("wingL", "theta_deg"): "tip_twist_L_deg", ("wingL", "v_ft"): "wingL_tip_ip_ft",
            ("htR", "w_ft"): "ht_tip_w_ft", ("htR", "theta_deg"): "ht_tip_twist_deg",
            ("htL", "w_ft"): "htL_tip_w_ft", ("htL", "theta_deg"): "htL_tip_twist_deg",
            ("vt", "w_ft"): "vt_tip_w_ft", ("vt", "theta_deg"): "vt_tip_twist_deg",
            ("fusV", "w_ft"): "fusV_tip_w_ft", ("fusL", "w_ft"): "fusL_tip_w_ft"}
COMPONENT_DOFS = {"wingR": ["dz", "dx", "twist"], "wingL": ["dz", "dx", "twist"], "htail": ["dz", "dy", "twist"],
                  "vtail": ["dy", "dz", "twist"], "fuselage": ["dz", "dy"]}
COMPONENT_BODIES = {"wingR": ["wingR"], "wingL": ["wingL"], "htail": ["htL", "htR"], "vtail": ["vt"],
                    "fuselage": ["fusV", "fusL"]}

SIGN_TABLE: List[Dict[str, str]] = [
    {"fd": "wing / HT / fusV w_ft (+ up)", "sb": "dz = -w * 0.3048", "sb_plus": "down"},
    {"fd": "VT / fusL w_ft (+ toward body +y)", "sb": "dy = +w * 0.3048", "sb_plus": "right"},
    {"fd": "wing v_ft (+ aft, body -x)", "sb": "wing dx = -v * 0.3048", "sb_plus": "forward"},
    {"fd": "wingR theta_deg (+ LE up)", "sb": "wingR.twist = +theta", "sb_plus": "LE up (right-hand about +y)"},
    {"fd": "wingL theta_deg (+ LE up)", "sb": "wingL.twist = -theta", "sb_plus": "LE down (right-hand about -y)"},
    {"fd": "htR / htL theta_deg (+ LE up, both sides)", "sb": "htail.twist = +theta (elastic only)",
     "sb_plus": "LE up (polyline left -> right = +y)"},
    {"fd": "vt theta_deg (+ LE toward +y)", "sb": "vtail.twist = -theta", "sb_plus": "LE toward -y (right-hand about up)"},
    {"fd": "ht_incidence_deg (+ nose-up)", "sb": "struct.ht_incidence = +rad; carried into htail/vtail dz as "
                                                 "-inc * (x - x_tail)", "sb_plus": "nose-up"},
    {"fd": "vt_sideslip_deg (+ fin LE toward +y)", "sb": "struct.vt_incidence = +rad, struct.vt_sideslip_equiv = -rad; "
                                                         "carried into htail/vtail dy as +inc * (x - x_tail)",
     "sb_plus": "fin LE toward +y / aircraft-beta sense"},
    {"fd": "root bending / torque / in-plane (lbf*ft)", "sb": "struct.*_root_* N*m, FD sign kept", "sb_plus": "as FD"},
    {"fd": "dL / dY (lbf), dRoll / dPitch / dYaw (lbf*ft) at / about the AERORP", "sb": "struct.elastic_* N / N*m, "
                                                                                       "FD (JSBSim l, m, n) sign kept",
     "sb_plus": "up / +y / right wing down / nose up / nose right"},
]


# ------------------------------------------------------------------ small helpers
def _raw_get(raw: Dict, key: str) -> Optional[float]:
    v = raw.get(key)
    if v is None:
        v = raw.get("flex." + key)
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def has_raw(ch: Dict) -> bool:
    """True if the dict carries FD v2 scalars (with or without the flex. prefix)."""
    return any(_raw_get(ch, k) is not None for k in ("tip_w_ft_R", "wingR_bm", "fusV_tip_w_ft", "ht_tip_w_ft"))


def phi(xi: float) -> float:
    """Assumed bending shape (cantilever, tip load): 0 and slope 0 at the clamp, 1 at the tip."""
    xi = min(max(float(xi), 0.0), 1.0)
    return xi * xi * (3.0 - xi) / 2.0


def psi(xi: float) -> float:
    """Assumed torsion shape (uniform distributed torque): 0 at the clamp, 1 at the tip."""
    xi = min(max(float(xi), 0.0), 1.0)
    return xi * (2.0 - xi)


def scalars_from_raw(raw: Dict) -> Dict[str, float]:
    """FD scalars -> struct.* SI scalars (only the keys present)."""
    out = {}
    for k, (name, fac, _u, _d) in SCALARS.items():
        v = _raw_get(raw, k)
        if v is not None:
            out[name] = v * fac
    for name, (k, fac, _u, _d) in DERIVED.items():
        v = _raw_get(raw, k)
        if v is not None:
            out[name] = v * fac
    return out


def scalar_doc() -> Dict[str, Dict]:
    d = {name: {"unit": u, "plus": m, "from": f"flex.{k}", "factor": fac} for k, (name, fac, u, m) in SCALARS.items()}
    d.update({name: {"unit": u, "plus": m, "from": f"flex.{k}", "factor": fac} for name, (k, fac, u, m) in DERIVED.items()})
    return d


def _r6(v):
    return round(float(v), 6)


# ------------------------------------------------------------------ geometry
def _component_geometry(bodies: Dict[str, Dict]) -> Dict[str, Dict]:
    """Our components from FD body node lists {body: {nodes_m, span_frac}}."""
    comps = {}
    for nm in ("wingR", "wingL"):
        if nm in bodies:
            b = bodies[nm]
            comps[nm] = {"axis_nodes_body_m": b["nodes_m"], "node_span_frac": b["span_frac"],
                         "order": [(nm, i) for i in range(len(b["nodes_m"]))]}
    if "htR" in bodies and "htL" in bodies:
        L, R = bodies["htL"], bodies["htR"]
        nl = len(L["nodes_m"])
        order = [("htL", i) for i in reversed(range(nl))] + [("htR", i) for i in range(len(R["nodes_m"]))]
        nodes = [L["nodes_m"][i] for i in reversed(range(nl))] + list(R["nodes_m"])
        frac = [-L["span_frac"][i] for i in reversed(range(nl))] + list(R["span_frac"])
        if math.dist(L["nodes_m"][0], R["nodes_m"][0]) < 1e-6:   # roots coincide: one shared clamped node
            order.pop(nl)
            nodes.pop(nl)
            frac.pop(nl)
        comps["htail"] = {"axis_nodes_body_m": nodes, "node_span_frac": frac, "order": order}
    if "vt" in bodies:
        b = bodies["vt"]
        comps["vtail"] = {"axis_nodes_body_m": b["nodes_m"], "node_span_frac": b["span_frac"],
                          "order": [("vt", i) for i in range(len(b["nodes_m"]))]}
    if "fusV" in bodies:
        b = bodies["fusV"]
        comps["fuselage"] = {"axis_nodes_body_m": b["nodes_m"], "node_span_frac": b["span_frac"],
                             "order": [("fusV", i) for i in range(len(b["nodes_m"]))]}
    for nm, c in comps.items():
        c["dof"] = list(COMPONENT_DOFS[nm])
        c["fd_bodies"] = list(COMPONENT_BODIES[nm])
    return comps


def _finish_geometry(bodies, source, estimated, aircraft=None, extra=None) -> Dict:
    fus = bodies.get("fusV") or bodies.get("fusL")
    x_tail = fus["nodes_m"][-1][0] if fus else 0.0
    geo = {"schema": GEOMETRY_SCHEMA, "v2_map_version": V2_MAP_VERSION, "aircraft": aircraft, "source": source,
           "layout_estimated": bool(estimated), "frame": "body FRD (x fwd, y right, z down), m, origin = CG at trim",
           "x_tail_m": x_tail, "bodies": bodies, "components": _component_geometry(bodies)}
    if extra:
        geo.update(extra)
    return geo


def geometry_from_layout(layout, aircraft: Optional[str] = None) -> Dict:
    """FD's node layout (flexbody.node_layout(mdl, rp_offset_body_ft), or an fd-flexbody-nodes/1 doc) -> geometry.
    Node coordinates ft -> m (origin = CG at trim when FD was given the AERORP offset)."""
    comps = layout.get("components") if isinstance(layout, dict) else layout
    bodies = {}
    for c in comps or []:
        nm = c.get("name")
        if nm not in FD_BODIES:
            continue
        nodes = [[_r6(v * FT) for v in p] for p in c["axis_nodes_body_ft"]]
        frac = c.get("node_span_frac") or [i / (len(nodes) - 1) for i in range(len(nodes))]
        bodies[nm] = {"nodes_m": nodes, "span_frac": [float(v) for v in frac]}
    extra = {"rp_offset_body_m": [_r6(v * FT) for v in layout["rp_offset_body_ft"]]} \
        if isinstance(layout, dict) and layout.get("rp_offset_body_ft") else None
    return _finish_geometry(bodies, "FD node layout (fd-flexbody-nodes/1, flexbody.node_layout)", False, aircraft, extra)


GENERIC = {"ht": {"AR": 4.5, "sweep": 25.0, "root_frac": 0.15}, "vt": {"AR": 1.5, "sweep": 35.0, "z_ft": None, "arm_ft": None}}
N_EST = {"wing": 9, "ht": 5, "vt": 5, "fus": 7}


def _default_fd_dir():
    env = os.environ.get("FLIGHT_DYNAMICS_DIR")
    if env:
        return env
    team = os.environ.get("FLIGHT_SIM_TEAM_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(team, "flight-dynamics")


def _fd_profiles(fd_dir):
    """FD's V2_PROFILES (flexbody.py, imported read-only without bytecode) or None."""
    import sys
    try:
        if fd_dir not in sys.path:
            sys.path.insert(0, fd_dir)
        old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            import flexbody  # noqa: WPS433
        finally:
            sys.dont_write_bytecode = old
        return flexbody.V2_PROFILES
    except Exception:
        return None


def geometry_estimated(aircraft: Optional[str] = None, *, fdm=None, cg_in=None, fd_dir=None, wing_structure=None,
                       use_fd_profiles: bool = True) -> Dict:
    """Approximate display geometry (body FRD m, origin CG) for files without an FD node layout. Sources, best first:
    FD's jsbsim_root_v2/aircraft/<ac>/flexbody_meta.json, V2_PROFILES (read-only import, optional), the CG / AERORP /
    metrics of a live FDM, ER's wing structure; generic values otherwise (``source`` says which)."""
    fd_dir = fd_dir or _default_fd_dir()
    meta = {}
    if aircraft:
        p = os.path.join(fd_dir, "jsbsim_root_v2", "aircraft", aircraft, "flexbody_meta.json")
        if os.path.exists(p):
            with open(p) as f:
                meta = json.load(f)
    prof = (_fd_profiles(fd_dir) or {}).get(aircraft or "") if use_fd_profiles else None
    src = []
    g = dict(meta.get("geom") or {})
    aerorp = (meta.get("v1_meta") or {}).get("aerorp_in")
    if fdm is not None:
        try:
            aerorp = aerorp or [fdm["metrics/aero-rp-x-in"], 0.0, fdm["metrics/aero-rp-z-in"]]
            cg_in = cg_in or (fdm["inertia/cg-x-in"], fdm["inertia/cg-z-in"])
            for k, prop in (("lh_ft", "metrics/lh-ft"), ("lv_ft", "metrics/lv-ft"), ("sh_ft2", "metrics/Sh-sqft"),
                            ("sv_ft2", "metrics/Sv-sqft"), ("bw_ft", "metrics/bw-ft")):
                g.setdefault(k, fdm[prop])
            src.append("JSBSim metrics / CG from the live FDM")
        except Exception:
            pass
    if meta:
        src.append("FD flexbody_meta.json")
    x_rp = z_rp = None
    if aerorp and cg_in:
        x_rp = -(aerorp[0] - cg_in[0]) / 12.0 * FT
        z_rp = -(aerorp[2] - cg_in[1]) / 12.0 * FT
    elif wing_structure:
        wr = next((c for c in wing_structure.get("components", []) if c.get("name") == "wingR"), None)
        if wr:
            nodes = wr["axis_nodes_body_m"]
            z_rp, x_rp = nodes[0][2], nodes[len(nodes) // 3][0]
            src.append("AERORP approximated from ER's wing structure")
    if x_rp is None:
        x_rp, z_rp = 0.0, 0.0
        src.append("AERORP assumed at the CG")
    ht = dict(GENERIC["ht"], **({k: prof["ht"][k] for k in ("AR", "sweep", "root_frac")} if prof else {}))
    vt = dict(GENERIC["vt"], **({k: prof["vt"][k] for k in ("AR", "sweep", "z_ft", "arm_ft")} if prof else {}))
    if prof:
        src.append("FD V2_PROFILES")
    lh = float(g.get("lh_ft") or 0.0) or 4.0 * math.sqrt(float(g.get("sh_ft2") or 20.0))
    lv = float(vt.get("arm_ft") or g.get("lv_ft") or 0.0) or lh
    sh, sv = float(g.get("sh_ft2") or 20.0), float(g.get("sv_ft2") or 12.0)
    if not g:
        src.append("generic tail sizes")
    s_h, h_v = math.sqrt(ht["AR"] * sh) / 2.0, math.sqrt(vt["AR"] * sv)
    z_v = vt["z_ft"] if vt.get("z_ft") is not None else 0.45 * h_v
    lh_m, lv_m, s_h_m, h_v_m, z_v_m = lh * FT, lv * FT, s_h * FT, h_v * FT, z_v * FT
    ws = float(g.get("bw_ft") or 30.0) / 2 * FT
    bodies = {}
    n = N_EST["wing"]
    for nm, sg in (("wingR", 1.0), ("wingL", -1.0)):
        y0 = 0.1 * ws
        bodies[nm] = {"nodes_m": [[_r6(x_rp), _r6(sg * (y0 + (ws - y0) * i / (n - 1))), _r6(z_rp)] for i in range(n)],
                      "span_frac": [i / (n - 1) for i in range(n)]}
    n = N_EST["ht"]
    y0, tan_h = ht["root_frac"] * s_h_m, math.tan(math.radians(ht["sweep"]))
    for nm, sg in (("htR", 1.0), ("htL", -1.0)):
        ys = [y0 + (s_h_m - y0) * i / (n - 1) for i in range(n)]
        bodies[nm] = {"nodes_m": [[_r6(x_rp - lh_m - (y - y0) * tan_h), _r6(sg * y), _r6(z_rp)] for y in ys],
                      "span_frac": [i / (n - 1) for i in range(n)]}
    n, tan_v = N_EST["vt"], math.tan(math.radians(vt["sweep"]))
    bodies["vt"] = {"nodes_m": [[_r6(x_rp - lv_m - (h_v_m * i / (n - 1) - z_v_m) * tan_v), 0.0, _r6(z_rp - h_v_m * i / (n - 1))]
                                for i in range(n)], "span_frac": [i / (n - 1) for i in range(n)]}
    n = N_EST["fus"]
    fus = [[_r6(x_rp - lh_m * i / (n - 1)), 0.0, _r6(z_rp)] for i in range(n)]
    bodies["fusV"] = {"nodes_m": fus, "span_frac": [i / (n - 1) for i in range(n)]}
    bodies["fusL"] = {"nodes_m": [list(p) for p in fus], "span_frac": [i / (n - 1) for i in range(n)]}
    return _finish_geometry(bodies, "ESTIMATED layout: " + ("; ".join(src) or "generic"), True, aircraft)


# backwards-compatible alias (pre-2.0 name)
geometry = geometry_estimated


# ------------------------------------------------------------------ nodal values
def nodes_frame(values: Dict, k: Optional[int] = None) -> Dict[str, Dict[str, List[float]]]:
    """FD nodal values -> one frame {body: {field: [floats per node]}}. ``values`` is the 'values' dict of an
    fd-flexbody-nodes/1 doc (per-frame rows: pass k) or flexbody.node_values(mdl, eta) for one eta (1-D arrays)."""
    out = {}
    for b, fv in (values or {}).items():
        if b not in NODE_FIELDS:
            continue
        o = {}
        for f, arr in fv.items():
            a = arr[k] if k is not None else arr
            if a is not None and hasattr(a, "__len__") and len(a) and hasattr(a[0], "__len__"):
                raise ValueError(f"nodes_frame: {b}.{f} has several frames; pass k")
            o[f] = [float(v) for v in a]
        out[b] = o
    return out


def _body_values(raw: Dict, geo: Dict, nodes: Optional[Dict], body: str, field: str):
    """(values in FD units per node of `body`, estimated?) or (None, None) if neither nodes nor a tip scalar exist."""
    if nodes and body in nodes and field in nodes[body]:
        return nodes[body][field], False
    key = TIP_KEYS.get((body, field))
    tip = _raw_get(raw, key) if key else None
    if tip is None and body == "htL":       # pre-telemetry files: left HT tip not exported -> mirror the right one
        tip = _raw_get(raw, TIP_KEYS[("htR", field)])
    if tip is None:
        return None, None
    shape = psi if field == "theta_deg" else phi
    return [tip * shape(xi) for xi in geo["bodies"][body]["span_frac"]], True


def available_components(raw: Optional[Dict], nodes: Optional[Dict] = None,
                         components: Optional[Sequence[str]] = None) -> List[str]:
    """Components FD actually models in this record: every FD body has nodal values, or FD exported at least one
    of its tip keys. A fidelity without that body (reduced = flexwing v1: wings only) gets no component at all,
    rather than an all-zero 'estimate'."""
    out = []
    for nm in (COMPONENTS if components is None else components):
        bodies = COMPONENT_BODIES[nm]
        if nodes and all(b in nodes for b in bodies):
            out.append(nm)
        elif raw and any(_raw_get(raw, k) is not None for (b, _f), k in TIP_KEYS.items() if b in bodies):
            out.append(nm)
    return out


def component_status(geo: Dict, nodes: Optional[Dict], components: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """{component: 'fd_nodes' | 'estimated'}: fd_nodes only if every FD body of the component has nodal values."""
    out = {}
    for nm in (COMPONENTS if components is None else components):
        if nm not in geo["components"]:
            continue
        ok = bool(nodes) and all(b in nodes and nodes[b].get("w_ft") is not None for b in COMPONENT_BODIES[nm])
        out[nm] = "fd_nodes" if ok else "estimated"
    return out


def map_v2_record(telemetry_row: Dict, geometry: Optional[Dict] = None, nodes: Optional[Dict] = None, *,
                  components: Optional[Sequence[str]] = None, scalars: bool = True) -> Dict[str, float]:
    """ONE frame of FD v2 telemetry -> Sim Bridge channels (see module doc for every sign).

    telemetry_row  FD scalars: coupler.last / a 'structure' history row / flex.<key> channels (prefix optional)
    geometry       geometry_from_layout(...) (exact node positions) or geometry_estimated(...); default estimated
    nodes          this frame's FD nodal values {body: {w_ft, theta_deg, v_ft: [per node]}} (nodes_frame()); bodies
                   without nodal values are estimated from the tip scalars
    components     subset of COMPONENTS to emit (ER: ("htail", "vtail", "fuselage")); default all
    scalars        also emit the struct.* scalars
    """
    geo = geometry or geometry_estimated()
    raw = telemetry_row
    out: Dict[str, float] = scalars_from_raw(raw) if scalars else {}
    want = [c for c in (COMPONENTS if components is None else components) if c in geo["components"]]
    vals: Dict[tuple, Optional[List[float]]] = {}

    def bv(body, field):
        if (body, field) not in vals:
            vals[(body, field)] = _body_values(raw, geo, nodes, body, field)[0]
        return vals[(body, field)]

    fv, fl = bv("fusV", "w_ft"), bv("fusL", "w_ft")
    wv = fv[-1] if fv else 0.0              # tail plunge (ft, + up)
    wl = fl[-1] if fl else 0.0              # tail lateral (ft, + right)
    inc = (_raw_get(raw, "ht_incidence_deg") or 0.0) * DEG
    vsl = (_raw_get(raw, "vt_sideslip_deg") or 0.0) * DEG
    x_tail = geo["x_tail_m"]
    for nm in want:
        c = geo["components"][nm]
        nodes_m = c["axis_nodes_body_m"]
        if nm in ("wingR", "wingL"):
            for field, (_c, dof, fac) in NODE_FIELDS[nm].items():
                v = bv(nm, field)
                if v is None:       # field not exported (older files): 0, the component is flagged estimated
                    v = [0.0] * len(nodes_m)
                for i, x in enumerate(v):
                    out[f"{nm}.{dof}.{i}"] = x * fac
        elif nm == "fuselage":
            for i in range(len(nodes_m)):
                out[f"fuselage.dz.{i}"] = -FT * fv[i] if fv else 0.0
                out[f"fuselage.dy.{i}"] = FT * fl[i] if fl else 0.0
        else:
            for i, (body, j) in enumerate(c["order"]):
                dxr = nodes_m[i][0] - x_tail
                w = bv(body, "w_ft")
                th = bv(body, "theta_deg")
                w_own = w[j] if w else 0.0
                th_own = th[j] if th else 0.0
                if nm == "htail":
                    out[f"htail.dz.{i}"] = -FT * (w_own + wv) - inc * dxr
                    out[f"htail.dy.{i}"] = FT * wl + vsl * dxr
                    out[f"htail.twist.{i}"] = DEG * th_own
                else:
                    out[f"vtail.dy.{i}"] = FT * (w_own + wl) + vsl * dxr
                    out[f"vtail.dz.{i}"] = -FT * wv - inc * dxr
                    out[f"vtail.twist.{i}"] = -DEG * th_own
    return out


# ------------------------------------------------------------------ structure block
SOURCE_TEXT = {"fd_nodes": "FD FE nodal values (fd-flexbody-nodes/1), mapped by sim_bridge.v2_map",
               "estimated": "ESTIMATED from FD tip scalars x assumed shape (bending phi = xi^2(3-xi)/2, torsion "
                            "psi = xi(2-xi)); no FD nodal data"}


def structure_block(geometry: Dict, *, components: Optional[Sequence[str]] = None, estimated=None,
                    base: Optional[Dict] = None, synthetic: bool = False, extra: Optional[Dict] = None) -> Dict:
    """Trajectory 'structure' block. ``estimated``: bool for all, or {component: bool | 'fd_nodes' | 'estimated'}
    (component_status output); default = geometry['layout_estimated']. ``base`` (e.g. ER's FlexState.structure with
    the wings) is kept as is; our components are added for names it does not have."""
    st = json.loads(json.dumps(base)) if base else {
        "schema": STRUCTURE_SCHEMA, "synthetic": bool(synthetic), "components": [],
        "units": {"axis_nodes_body_m": "m, body FRD (x fwd, y right, z down), origin CG at trim",
                  "dz": "m (body z, + down)", "dy": "m (body y, + right)", "dx": "m (body x, + forward)",
                  "twist": "rad, right-hand about node i -> i+1"}}
    st.setdefault("units", {}).setdefault("dx", "m (body x, + forward)")
    have = {c["name"] for c in st.get("components", [])}
    added, est_list = [], []
    for nm in (COMPONENTS if components is None else components):
        c = geometry["components"].get(nm)
        if c is None or nm in have:
            continue
        e = estimated.get(nm, geometry["layout_estimated"]) if isinstance(estimated, dict) else estimated
        e = geometry["layout_estimated"] if e is None else (e == "estimated" if isinstance(e, str) else bool(e))
        st["components"].append({"name": nm, "axis_nodes_body_m": c["axis_nodes_body_m"], "dof": list(c["dof"]),
                                 "node_span_frac": c["node_span_frac"], "fd_bodies": c["fd_bodies"],
                                 "estimated": e, "node_values": SOURCE_TEXT["estimated" if e else "fd_nodes"]})
        added.append(nm)
        if e:
            est_list.append(nm)
    st["v2_map"] = {"schema": SCHEMA, "version": V2_MAP_VERSION, "geometry_source": geometry.get("source"),
                    "layout_estimated": geometry.get("layout_estimated"), "added_components": added,
                    "estimated_components": est_list, "x_tail_m": geometry.get("x_tail_m"),
                    "sign_table": SIGN_TABLE, "scalars": scalar_doc(),
                    "fd_conventions": "flight-dynamics/INTERFACE_v2.md section 11 (v2_results/sign_probe.json)"}
    if est_list:
        st["v2_map"]["estimate_shapes"] = "bending / in-plane phi(xi) = xi^2 (3 - xi) / 2; torsion psi(xi) = xi (2 - xi)"
    if extra:
        st["v2_map"].update(extra)
    if synthetic:
        st["synthetic"] = True
        st["synthetic_doc"] = SYNTHETIC_DOC
    return st


def validate_structure(structure: Dict, channels: Iterable[str]) -> List[str]:
    """Problems with a structure block vs. the channel list (empty list = OK)."""
    probs, names = [], set(channels)
    comps = structure.get("components") if isinstance(structure, dict) else None
    if not isinstance(comps, list) or not comps:
        return ["structure.components missing or empty"]
    for c in comps:
        nm, nodes = c.get("name"), c.get("axis_nodes_body_m")
        if not isinstance(nm, str) or not isinstance(nodes, list) or len(nodes) < 2:
            probs.append(f"component {nm!r}: needs a name and >= 2 axis nodes")
            continue
        if any(len(p) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) for p in nodes):
            probs.append(f"{nm}: axis nodes must be finite [x, y, z]")
        for d in c.get("dof") or []:
            if d not in DOFS:
                probs.append(f"{nm}: unknown dof {d!r}")
            missing = [i for i in range(len(nodes)) if f"{nm}.{d}.{i}" not in names]
            if missing:
                probs.append(f"{nm}.{d}: no channel for nodes {missing}")
        extra = [n for n in names if n.startswith(nm + ".") and n.count(".") == 2 and n.split(".")[2].isdigit()
                 and int(n.split(".")[2]) >= len(nodes)]
        if extra:
            probs.append(f"{nm}: channels beyond the last node {sorted(extra)[:3]}")
    return probs


# ------------------------------------------------------------------ whole trajectory docs
def convert_traj(doc: Dict, nodes_doc: Optional[Dict] = None, geometry: Optional[Dict] = None, decimals: int = 6,
                 components: Optional[Sequence[str]] = None) -> Dict:
    """ga-flightsim-traj/1 doc with flex.* channels -> doc + struct.* + component node channels + structure block.
    ``nodes_doc`` = FD's telemetry[i]['nodes'] (fd-flexbody-nodes/1; frame k = the coupler step ending at
    t = (k+1) * sim_dt) gives exact node values; rows without a frame (t = 0) and files without nodes are estimated.
    Components already present in the doc (e.g. ER's wings) are kept, ours are added for the others."""
    ch = list(doc["channels"])
    if not any(c.startswith("flex.") for c in ch):
        return doc
    present = {c.split(".")[0] for c in ch if c.count(".") == 2 and c.split(".")[0] in COMPONENTS}
    first = next((dict(zip(ch, r)) for r in doc.get("data") or []), {})
    comps = [c for c in (COMPONENTS if components is None else components) if c not in present]
    if not nodes_doc:
        comps = available_components(first, None, comps)
    geo = geometry or (geometry_from_layout(nodes_doc, doc.get("aircraft")) if nodes_doc
                       else geometry_estimated(doc.get("aircraft"), wing_structure=doc.get("structure")))
    sim_dt = float(doc.get("sim_dt_s") or 1 / 120)
    values = (nodes_doc or {}).get("values") or {}
    n_frames = min((len(a) for fv in values.values() for a in fv.values()), default=0)
    ti = ch.index("t")
    new_names: List[str] = []
    rows_out, statuses = [], {}
    mapped_rows = []
    for row in doc["data"]:
        d = {c: v for c, v in zip(ch, row) if v is not None}
        k = int(round(float(row[ti]) / sim_dt)) - 1
        nf = nodes_frame(values, k) if values and 0 <= k < n_frames else None
        m = map_v2_record(d, geo, nf, components=comps) if has_raw(d) else {}
        if k >= 0 or not values:    # FD exports no frame for t = 0 (trim); that row alone is estimated
            for nm, s in component_status(geo, nf, comps).items():
                statuses.setdefault(nm, set()).add(s)
        mapped_rows.append(m)
        for n in m:
            if n not in ch and n not in new_names:
                new_names.append(n)
    new_names.sort()
    for row, m in zip(doc["data"], mapped_rows):
        rows_out.append(list(row) + [round(m[n], decimals) if n in m and math.isfinite(m[n]) else None for n in new_names])
    out = dict(doc)
    out["channels"] = ch + new_names
    out["data"] = rows_out
    est = {nm: ("fd_nodes" if s == {"fd_nodes"} else "estimated") for nm, s in statuses.items()}
    out["structure"] = structure_block(geo, components=comps, estimated=est, base=doc.get("structure"),
                                       synthetic=bool((doc.get("structure") or {}).get("synthetic") or doc.get("synthetic")),
                                       extra={"nodes_frames": n_frames, "node_status": est})
    return out
