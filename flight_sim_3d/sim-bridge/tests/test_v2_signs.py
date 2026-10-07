"""Every v2 sign convention, cross-checked against Flight Dynamics' own probes (read-only):

* flight-dynamics/v2_results/sign_probe.json (FD's deterministic static / mode probes, all four aircraft): each probe's
  FD telemetry is mapped with sim_bridge.v2_map and the RESULT is checked against the physics of the probe in our
  viewer convention (dz + down, dy + right, dx + forward, twist right-hand about node i -> i+1).
* live FD model (flexbody.FlexBodyModel baseline, flexbody.mode_probe / node_values / node_layout): a unit elastic mode
  per body (FD's mode-sign rule: tip value positive) mapped from NODAL values: the last node equals the mapped tip
  scalar and points the documented way.
Skips (prints) when FD's files are not present. FD is imported without writing bytecode.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.dirname(HERE)
sys.path.insert(0, SB)
sys.dont_write_bytecode = True

from sim_bridge import paths, v2_map as V  # noqa: E402

PROBE = os.path.join(paths.FLIGHT_DYNAMICS_DIR, "v2_results", "sign_probe.json")
FT, DEG = 0.3048, math.pi / 180


def _probe():
    if not os.path.exists(PROBE):
        print("skip (no FD sign_probe.json):", PROBE)
        return None
    with open(PROBE) as f:
        return json.load(f)


def _case(d, prefix):
    k = next(k for k in d if k.startswith(prefix))
    return d[k]


def _map(raw, ac):
    g = V.geometry_estimated(ac, use_fd_profiles=False)          # tip-scalar mapping (the probe has no nodes)
    ch = V.map_v2_record(raw, g)
    last = {nm: len(g["components"][nm]["axis_nodes_body_m"]) - 1 for nm in g["components"]}
    return ch, last


def test_probe_file_covers_all_aircraft_and_new_keys():
    P = _probe()
    if P is None:
        return
    assert set(P) == {"c172x", "T38", "737", "f16"}
    for ac, d in P.items():
        nz = _case(d, "nz=+1")
        for k in ("htL_tip_w_ft", "ht_tip_twist_deg", "htL_tip_twist_deg", "vt_tip_twist_deg", "wingR_tip_ip_ft"):
            assert k in nz, (ac, k)


def test_vertical_1g_inertia_everything_sags():
    """nz = +1 (1 g down, no lift): wing tips, HT tips and the tail go DOWN -> dz > 0; tail bends tail-down -> fusV_bm < 0."""
    P = _probe()
    if P is None:
        return
    for ac, d in P.items():
        raw = _case(d, "nz=+1")
        ch, last = _map(raw, ac)
        assert ch["struct.wingR_tip_dz"] > 0 and ch["struct.wingL_tip_dz"] > 0, ac
        assert ch[f"wingR.dz.{last['wingR']}"] > 0 and ch[f"wingL.dz.{last['wingL']}"] > 0, ac
        assert ch["htail.dz.0"] > 0 and ch[f"htail.dz.{last['htail']}"] > 0, ac
        assert ch[f"fuselage.dz.{last['fuselage']}"] > 0 and ch["struct.fusV_tip_dz"] > 0, ac
        assert ch["struct.fusV_root_bm"] < 0 and ch["struct.wingR_root_bm"] < 0, ac          # + = up / tail-up
        assert math.isclose(ch["struct.fusV_tip_dz"], -FT * raw["fusV_tip_w_ft"])


def test_pitch_rate_tail_upload():
    """q = +0.1: tail up-load -> HT tips and tail go UP (dz < 0); HT elastic twist LE up -> htail.twist > 0 on both
    halves (polyline left -> right = +y); the fuselage slope gives a NOSE-DOWN tail incidence (ht_incidence < 0)."""
    P = _probe()
    if P is None:
        return
    for ac, d in P.items():
        raw = _case(d, "q=+0.1")
        ch, last = _map(raw, ac)
        n = last["htail"]
        assert ch["htail.dz.0"] < 0 and ch[f"htail.dz.{n}"] < 0, ac
        assert ch["htail.twist.0"] > 0 and ch[f"htail.twist.{n}"] > 0, ac
        assert math.isclose(ch[f"htail.twist.{n}"], DEG * raw["ht_tip_twist_deg"])            # elastic only
        assert ch["struct.ht_incidence"] < 0 and ch["struct.htR_root_bm"] > 0 and ch["struct.fusV_root_bm"] > 0, ac
        assert ch[f"fuselage.dz.{last['fuselage']}"] < 0, ac


def test_roll_rate_left_right_differ():
    """p = +0.1 (roll right): right wing / right HT see upwash -> UP (dz < 0), left ones DOWN; right tips twist LE up
    (wingR.twist > 0, htail right tip > 0), the left HT twists LE down (htail left tip < 0)."""
    P = _probe()
    if P is None:
        return
    for ac, d in P.items():
        raw = _case(d, "p=+0.1")
        ch, last = _map(raw, ac)
        n = last["htail"]
        assert ch[f"wingR.dz.{last['wingR']}"] < 0 < ch[f"wingL.dz.{last['wingL']}"], ac
        assert ch[f"htail.dz.{n}"] < 0 < ch["htail.dz.0"], ac                                  # R up, L down
        assert ch[f"htail.twist.{n}"] > 0 > ch["htail.twist.0"], ac
        assert ch["struct.wingR_tip_twist"] > 0 and ch[f"wingR.twist.{last['wingR']}"] > 0, ac
        assert ch["struct.htR_root_bm"] > 0 > ch["struct.htL_root_bm"], ac


def test_yaw_rate_fin_and_tail_lateral():
    """r = +0.1 (nose right): the fin swings left through the air -> side load toward +y: fin tip and tail go RIGHT
    (dy > 0); FD fin twist + (LE toward +y) -> our vtail.twist < 0; fin incidence vt_sideslip < 0 (= LE toward -y),
    i.e. an equivalent local sideslip > 0 (aircraft-beta sense)."""
    P = _probe()
    if P is None:
        return
    for ac, d in P.items():
        raw = _case(d, "r=+0.1")
        ch, last = _map(raw, ac)
        n = last["vtail"]
        assert raw["vt_tip_w_ft"] > 0 and ch[f"vtail.dy.{n}"] > 0 and ch["struct.vt_tip_dy"] > 0, ac
        assert raw["vt_tip_twist_deg"] > 0 and ch[f"vtail.twist.{n}"] < 0 and ch["struct.vt_tip_twist"] < 0, ac
        assert ch[f"fuselage.dy.{last['fuselage']}"] > 0 and ch["struct.vt_root_bm"] > 0 and ch["struct.fusL_root_bm"] > 0, ac
        assert ch["struct.vt_incidence"] < 0 < ch["struct.vt_sideslip_equiv"], ac


def test_sideslip_wind_from_right():
    """beta = +0.05 (wind from the right): the fin is pushed LEFT (dy < 0, root bending < 0); the fuselage slope turns
    the fin LE toward +y (vt_incidence > 0), relieving the fin (equivalent local sideslip < 0)."""
    P = _probe()
    if P is None:
        return
    for ac, d in P.items():
        raw = _case(d, "beta=+0.05")
        ch, last = _map(raw, ac)
        assert ch[f"vtail.dy.{last['vtail']}"] < 0 and ch["struct.vt_root_bm"] < 0, ac
        assert ch[f"fuselage.dy.{last['fuselage']}"] < 0 and ch["struct.fusL_tip_dy"] < 0, ac
        assert ch["struct.vt_incidence"] > 0 > ch["struct.vt_sideslip_equiv"], ac


def test_drag_inplane_bends_tip_aft():
    """aero force aft (drag): wing tip moves AFT -> our dx < 0 (body x forward); in-plane root moment + (aft load)."""
    P = _probe()
    if P is None:
        return
    for ac, d in P.items():
        raw = _case(d, "drag")
        ch, last = _map(raw, ac)
        assert raw["wingR_tip_ip_ft"] > 0 and ch["struct.wingR_tip_dx"] < 0 and ch[f"wingR.dx.{last['wingR']}"] < 0, ac
        assert ch["struct.wingR_root_ip_bm"] > 0, ac
        assert math.isclose(ch["struct.wingR_tip_dx"], -FT * raw["wingR_tip_ip_ft"])


def test_feedback_axes_from_mode_probes():
    """Elastic feedback (forces at / moments about the AERORP, JSBSim l/m/n signs kept):
    right-wing LE-up torsion -> more lift (dL > 0) and roll LEFT (l < 0: + = right wing down);
    right-HT LE-up torsion -> tail lift up -> nose DOWN (m < 0); fin LE toward +y -> side force +y, nose LEFT (n < 0)."""
    P = _probe()
    if P is None:
        return
    for ac, d in P.items():
        s = V.scalars_from_raw(_case(d, "mode wingR/t"))
        assert s["struct.elastic_dlift"] > 0 > s["struct.elastic_droll"], ac
        s = V.scalars_from_raw(_case(d, "mode htR/t"))
        assert s["struct.elastic_dpitch"] < 0 and s["struct.elastic_dlift"] > 0, ac
        s = V.scalars_from_raw(_case(d, "mode vt/t"))
        assert s["struct.elastic_dside"] > 0 > s["struct.elastic_dyaw"], ac
        raw = _case(d, "mode wingR/t")
        assert math.isclose(V.scalars_from_raw(raw)["struct.elastic_droll"], raw["dRoll_lbft"] * V.LBFFT)


def _fd():
    try:
        sys.path.insert(0, paths.FLIGHT_DYNAMICS_DIR)
        import flexbody as fb  # read-only, no bytecode
        return fb
    except Exception as e:  # noqa: BLE001
        print("skip (FD not importable):", e)
        return None


def test_live_fd_nodal_mode_signs_and_tip_equality():
    """Unit elastic mode per body on FD's baseline c172x and T38 models, mapped from NODAL values with FD's layout:
    the mapped last node equals the mapped tip scalar, and each points the documented way."""
    fb = _fd()
    if fb is None:
        return
    import numpy as np
    for ac in ("c172x", "T38"):
        try:
            mdl = fb.FlexBodyModel(ac, {})
        except Exception as e:  # noqa: BLE001
            print("skip", ac, e)
            continue
        geo = V.geometry_from_layout(fb.node_layout(mdl, [0.0, 0.0, 0.0]), ac)
        n = {nm: len(c["axis_nodes_body_m"]) - 1 for nm, c in geo["components"].items()}
        cases = [("wingR", "b", f"wingR.dz.{n['wingR']}", -1, "struct.wingR_tip_dz"),
                 ("wingR", "t", f"wingR.twist.{n['wingR']}", +1, "struct.wingR_tip_twist"),
                 ("wingL", "t", f"wingL.twist.{n['wingL']}", -1, "struct.wingL_tip_twist"),
                 ("htR", "b", f"htail.dz.{n['htail']}", -1, "struct.htR_tip_dz"),
                 ("htL", "b", "htail.dz.0", -1, "struct.htL_tip_dz"),
                 ("htR", "t", f"htail.twist.{n['htail']}", +1, "struct.htR_tip_twist"),
                 ("htL", "t", "htail.twist.0", +1, "struct.htL_tip_twist"),
                 ("vt", "b", f"vtail.dy.{n['vtail']}", +1, "struct.vt_tip_dy"),
                 ("vt", "t", f"vtail.twist.{n['vtail']}", -1, "struct.vt_tip_twist")]
        if getattr(mdl.wingR, "beam", None) is not None and mdl.wingR.beam.has_v and "v" in list(mdl.wingR.cls):
            cases.append(("wingR", "v", f"wingR.dx.{n['wingR']}", -1, "struct.wingR_tip_dx"))
        for body, cls, chan, sign, scal in cases:
            b = getattr(mdl, body)
            if cls not in list(getattr(b, "cls", "b")):
                continue
            raw, _er = fb.mode_probe(mdl, body, cls)
            j = list(b.cls).index(cls)
            eta = np.zeros(mdl.N)
            eta[np.arange(mdl.N)[mdl.slices[body]][j]] = 1e-3
            frame = V.nodes_frame(fb.node_values(mdl, eta))
            ch = V.map_v2_record(raw, geo, frame)
            assert ch[chan] * sign > 0, (ac, body, cls, chan, ch[chan])
            if body in ("htR", "htL", "vt"):       # tails ride on the fuselage: compare the own deflection
                own = ch[chan] - V.map_v2_record(dict(raw, ht_tip_w_ft=0, htL_tip_w_ft=0, vt_tip_w_ft=0, ht_tip_twist_deg=0,
                                                       htL_tip_twist_deg=0, vt_tip_twist_deg=0), geo,
                                                  {k: ({f: [0.0] * len(v) for f, v in fv.items()} if k == body else fv)
                                                   for k, fv in frame.items()})[chan]
            else:
                own = ch[chan]
            assert math.isclose(own, ch[scal], rel_tol=1e-6, abs_tol=1e-12), (ac, body, cls, own, ch[scal])


if __name__ == "__main__":
    bad = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            try:
                f()
                print("PASS", k)
            except BaseException as e:  # noqa: BLE001
                bad += 1
                import traceback
                traceback.print_exc()
                print("FAIL", k, type(e).__name__, e)
    sys.exit(1 if bad else 0)
