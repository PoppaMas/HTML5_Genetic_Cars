"""Sim Bridge v2_map wiring: FlexState schema /3, FE nodal wings + modal rename, public fd_model/nodes, evaluate(pin=)."""
import json
import math
import os
import re

import numpy as np
import pytest

from evolution import eval as ev, fidelity as F, trajectory, validate_traj
from evolution.tests.test_eval import FD_DIR, G8, SHORT, TEAM, need_fd, phase1_profile_d, _sc


@need_fd
def test_v2_map_loads_readonly_and_version():
    m = F.v2_map_mod()
    assert m.V2_MAP_VERSION == "2.0.0" and m.SCHEMA == "sim-bridge-v2-map/2"
    # loaded from the team tree, not as an installed package path under site-packages
    assert os.path.samefile(m.__file__, os.path.join(TEAM, "sim-bridge", "sim_bridge", "v2_map.py"))


@need_fd
def test_full_flex_state_uses_v2_map_and_modal_rename():
    pd = phase1_profile_d(**SHORT)
    states = []

    def rec(t, fdm, fs):
        if len(states) < 1:
            states.append(fs)

    F.evaluate_genome(pd, G8, None, _sc(pd), "full", 1.0, recorder=rec)
    fs = states[0]
    assert fs.schema == "evolution-flex-state/3" == F.FLEX_STATE_SCHEMA
    names = [c["name"] for c in fs.structure["components"]]
    assert names[0:2] == ["wingR_modal", "wingL_modal"]
    for req in ("wingR", "wingL", "htail", "vtail", "fuselage"):
        assert req in names
    for c in fs.structure["components"]:
        if c["name"] in ("wingR_modal", "wingL_modal"):
            assert c["dof"] == ["dz", "dy", "twist"] and len(c["node_span_frac"]) == F.N_NODES
        if c["name"] == "wingR":
            assert "dx" in c["dof"] and c.get("estimated") is False
    assert fs.structure.get("v2_map", {}).get("version") == "2.0.0"
    ch = fs.channels()
    assert any(k.startswith("wingR_modal.dz.") for k in ch)
    assert any(k.startswith("wingR.dx.") for k in ch) and any(k.startswith("htail.dz.") for k in ch)
    assert any(k.startswith("struct.") for k in ch) and any(k.startswith("flex.") for k in ch)
    # public accessors
    assert fs.fd_model is not None and fs.fd_model.model == "c172x"
    nodes = fs.nodes()
    assert nodes is not None and set(nodes) >= {"wingR", "wingL", "htR", "htL", "vt", "fusV", "fusL"}
    assert all(isinstance(nodes["wingR"][f][0], float) for f in ("w_ft", "theta_deg", "v_ft"))
    layout = fs.node_layout()
    assert isinstance(layout, list) and {c["name"] for c in layout} >= {"wingR", "fusV"}
    assert fs.v2_map_version == "2.0.0" and fs.v2_geometry is not None
    # FE tip matches FD tip scalars (v2_map)
    nR = max(int(k.split(".")[-1]) for k in ch if k.startswith("wingR.twist."))
    assert ch[f"wingR.twist.{nR}"] == pytest.approx(math.radians(ch["flex.tip_twist_R_deg"]), abs=1e-12)


@need_fd
def test_reduced_keeps_modal_wing_names_without_v2_map_components():
    pd = phase1_profile_d(**SHORT)
    states = []

    def rec(t, fdm, fs):
        if not states:
            states.append(fs)

    F.evaluate_genome(pd, G8, None, _sc(pd), "reduced", 0.9, recorder=rec)
    fs = states[0]
    names = [c["name"] for c in fs.structure["components"]]
    assert names == ["wingR", "wingL"]
    assert fs.nodes() is None and fs.v2_map_version is None and fs.fd_model is not None
    ch = fs.channels()
    assert any(re.match(r"^wingR\.dz\.\d+$", k) for k in ch)
    assert not any(k.startswith("wingR_modal.") for k in ch)
    assert not any(k.startswith("htail.") for k in ch)


@need_fd
def test_make_fd_model_public_matches_private_helper():
    pd = phase1_profile_d(**SHORT)
    m = F.fd_modules()
    fe, fb = m["fe"], m["fb"]
    from evolution import sim
    P = sim.Profile.from_dict(pd)
    _root, rv2 = F.roots(P)
    priv = F._struct_obj(fe, fb, F.struct_from(None), P.aircraft, "full", rv2)
    pub = F.make_fd_model(pd, None, "full")
    assert type(pub) is type(priv) and pub.model == priv.model == "c172x"


@need_fd
def test_evaluate_pin_raises_on_mismatch(tmp_path):
    # tiny run.json-like cfg via a real short batch would be heavy; call evaluate with a fake run cfg from smoke
    # P2.5 smoke run.json: its full model_version is FD's current one (phase2-smoke-s1 carries the older post-mass pin)
    rd = os.path.join(TEAM, "evolution", "runs", "phase2-smoke-p25-s1")
    if not os.path.isdir(rd):
        pytest.skip("phase2-smoke-p25-s1 not present")
    run = json.load(open(os.path.join(rd, "run.json")))
    # use only scenario 0 and a short override via profile - still full flight; keep pin check cheap by mismatching first
    ac = "c172x"
    entry = next(a for a in run["aircraft"] if a["name"] == ac)
    # shorten profile for speed
    entry = dict(entry)
    entry["resolved_profile"] = dict(entry["resolved_profile"], duration_s=8.0,
                                     steps_rel_ft=[[0.0, 0.0], [2.0, 50.0]])
    run = dict(run, aircraft=[entry], scenarios=[s for s in run["scenarios"] if s["aircraft"] == ac][:1])
    entry["scenario_ids"] = [run["scenarios"][0]["id"]]
    genome = {"kp_alt": 0.24, "ki_alt": 0.01, "kd_alt": 0.4, "kp_pitch": 0.07, "ki_pitch": 2e-5, "kd_pitch": 0.015,
              "kp_hdg": 1.0, "ki_hdg": 0.01}
    for g in entry["genes"]:
        if g["group"] == "struct":
            genome[g["name"]] = 0.02 if g["name"] == "struct_damping_ratio" else 1.0
    with pytest.raises(RuntimeError, match="pin mismatch"):
        ev.evaluate(genome, ac, 0, run, fidelity="full", pin="full:flexv2:00000000")
    pin_ok = (entry.get("ladder_model_version") or {}).get("full") or entry["model_version"]
    if isinstance(pin_ok, dict):
        pin_ok = pin_ok.get("full") or next(iter(pin_ok.values()))
    out = ev.evaluate(genome, ac, 0, run, fidelity="full", pin=pin_ok)
    assert out["model_version"] == pin_ok


def test_traj_schema_bump_and_validate_accepts_old():
    assert trajectory.SCHEMA == "ga-flightsim-traj/2"
    assert F.FLEX_STATE_SCHEMA == "evolution-flex-state/3"
    # old Phase-2 pilot traj still validates
    old = os.path.join(TEAM, "evolution", "runs", "phase2-pilot-s1", "trajectories")
    if os.path.isdir(old):
        assert validate_traj.main([old]) == 0


def test_channel_doc_covers_dx_modal_and_struct():
    d = trajectory.channel_doc(["wingR.dx.3", "wingR_modal.dz.0", "struct.wingR_tip_dz", "htail.twist.1", "flex.tip_w_ft_R"])
    assert "forward" in d["wingR.dx.3"] and "wingR_modal" in d["wingR_modal.dz.0"]
    assert "v2_map" in d["struct.wingR_tip_dz"] and "htail" in d["htail.twist.1"]
    assert "dz = -0.3048" in d["flex.tip_w_ft_R"]


@need_fd
def test_modal_twist_sign_after_rename():
    """wingR_modal must keep +twist for nose-up raw (startswith wingR); FE wingR also +. Regression for schema /3 rename."""
    pd = phase1_profile_d(**SHORT)
    states = []

    def rec(t, fdm, fs):
        if not states:
            states.append(fs)

    F.evaluate_genome(pd, G8, None, _sc(pd), "full", 1.0, recorder=rec)
    ch = states[0].channels()
    n = F.N_NODES - 1
    # 1 g trim: FD tip twist > 0 (nose-up) on both wings; modal and FE must agree in SB sign
    assert ch["flex.tip_twist_R_deg"] > 0 and ch["flex.tip_twist_L_deg"] > 0
    assert ch[f"wingR_modal.twist.{n}"] > 0 > ch[f"wingL_modal.twist.{n}"]
    assert ch[f"wingR_modal.twist.{n}"] == pytest.approx(math.radians(ch["flex.tip_twist_R_deg"]), rel=0.05, abs=1e-4)
    assert ch[f"wingL_modal.twist.{n}"] == pytest.approx(-math.radians(ch["flex.tip_twist_L_deg"]), rel=0.05, abs=1e-4)
    # pure synthetic geom rename: fd_to_structure_channels must treat wingR_modal like wingR
    eta = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.01]])  # reduced-shaped unused; build via _wing_surfaces full path
    from evolution import fidelity as F2
    rv2 = os.path.join(FD_DIR, "jsbsim_root_v2")
    mdl = F.fd_modules()["fb"].FlexBodyModel("c172x", None, asymmetric=False, root_v2=rv2)
    e = np.zeros(max(sl.stop for sl in mdl.slices.values()))
    for nm in ("wingR", "wingL"):
        surf, sl = getattr(mdl, nm), mdl.slices[nm]
        e[sl] = np.linalg.lstsq(surf.PsiT, np.full(surf.PsiT.shape[0], 0.02), rcond=None)[0]
    fns = [(nm + "_modal", fn) for nm, _, fn in F._wing_surfaces(mdl, "full")]
    ch2 = F.fd_to_structure_channels(e, {"fns": fns})
    assert ch2["wingR_modal.twist.4"] == pytest.approx(0.02, rel=0.05)
    assert ch2["wingL_modal.twist.4"] == pytest.approx(-0.02, rel=0.05)
