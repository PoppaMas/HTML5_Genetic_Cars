"""sim_bridge.v2_map tests (no JSBSim, no FD import). The fixture tests/fixtures/v2_synthetic_raw.json is SYNTHETIC
(documented FD v2 format, made-up values). Sign conventions are cross-checked against FD's own probes in
tests/test_v2_signs.py; real flights in tests/test_replay.py::test_real_full_fidelity_v2_nodes.
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.dirname(HERE)
sys.path.insert(0, SB)
sys.dont_write_bytecode = True

from sim_bridge import v2_map as V  # noqa: E402
from sim_bridge.recorder import TrajRecorder  # noqa: E402

FX = json.load(open(os.path.join(HERE, "fixtures", "v2_synthetic_raw.json")))
GEO = V.geometry_estimated("T38", use_fd_profiles=False)   # display-only geometry, no FD import
FT, DEG = 0.3048, math.pi / 180


def _layout():
    """A small FD-style node layout (fd-flexbody-nodes/1 'components', ft, origin CG): 5 nodes per body."""
    def body(name, pts):
        return {"name": name, "axis_nodes_body_ft": pts, "node_span_frac": [i / (len(pts) - 1) for i in range(len(pts))]}
    w = [[0.0, 2 + 2 * i, 0.0] for i in range(5)]
    ht = [[-20.0 - 0.5 * i, 1 + i, 0.0] for i in range(5)]
    return {"schema": "fd-flexbody-nodes/1", "rp_offset_body_ft": [0.5, 0.0, -0.2], "components": [
        body("wingR", w), body("wingL", [[x, -y, z] for x, y, z in w]),
        body("htR", ht), body("htL", [[x, -y, z] for x, y, z in ht]),
        body("vt", [[-18.0 - 0.8 * i, 0.0, -1.0 - 1.5 * i] for i in range(5)]),
        body("fusV", [[-5.0 * i, 0.0, 0.0] for i in range(5)]), body("fusL", [[-5.0 * i, 0.0, 0.0] for i in range(5)])]}


def test_public_api_and_version():
    assert V.V2_MAP_VERSION == "2.0.0" and V.SCHEMA.endswith("/2")
    for f in ("geometry_from_layout", "geometry_estimated", "map_v2_record", "structure_block", "nodes_frame",
              "component_status", "validate_structure", "convert_traj", "scalars_from_raw"):
        assert callable(getattr(V, f)), f
    assert V.COMPONENTS == ("wingR", "wingL", "htail", "vtail", "fuselage")


def test_standalone_imports_nothing_from_evolution_or_sim_bridge():
    """ER imports v2_map read-only: loading it must not pull evolution, sim_bridge.paths, FD or numpy."""
    code = ("import importlib.util, sys; sys.dont_write_bytecode = True\n"
            f"sp = importlib.util.spec_from_file_location('v2m', {os.path.join(SB, 'sim_bridge', 'v2_map.py')!r})\n"
            "m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)\n"
            "g = m.geometry_from_layout(" + json.dumps(_layout()) + ")\n"
            "ch = m.map_v2_record({'fusV_tip_w_ft': 0.1}, g, components=('htail', 'vtail', 'fuselage'))\n"
            "bad = [k for k in sys.modules if k.split('.')[0] in ('evolution', 'sim_bridge', 'flexbody', 'numpy')]\n"
            "print(len(ch), bad)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert out.returncode == 0, out.stderr
    n, bad = out.stdout.strip().split(" ", 1)
    assert int(n) > 0 and bad == "[]", out.stdout
    src = open(os.path.join(SB, "sim_bridge", "v2_map.py")).read()
    assert "from evolution" not in src and "import evolution" not in src and "from sim_bridge" not in src


def test_scalars_units_and_signs():
    raw = {k: 1.0 for k in V.RAW_KEYS}
    s = V.scalars_from_raw(raw)
    assert len(V.SCALARS) == V.DIAG_KEYS_V2_COUNT + V.TELEMETRY_KEYS_V2_COUNT == 32
    expect = {"struct.wingR_tip_dz": -FT, "struct.wingL_tip_dz": -FT, "struct.wingR_tip_twist": DEG,
              "struct.wingL_tip_twist": -DEG, "struct.wingR_tip_dx": -FT, "struct.wingL_tip_dx": -FT,
              "struct.htR_tip_dz": -FT, "struct.htL_tip_dz": -FT, "struct.htR_tip_twist": DEG, "struct.htL_tip_twist": DEG,
              "struct.vt_tip_dy": FT, "struct.vt_tip_twist": -DEG, "struct.fusV_tip_dz": -FT, "struct.fusL_tip_dy": FT,
              "struct.ht_incidence": DEG, "struct.vt_incidence": DEG, "struct.vt_sideslip_equiv": -DEG,
              "struct.wingR_root_bm": 1.3558179483314004, "struct.wingR_root_ip_bm": 1.3558179483314004,
              "struct.fusV_root_bm": 1.3558179483314004, "struct.elastic_dlift": 4.4482216152605,
              "struct.elastic_dside": 4.4482216152605, "struct.elastic_droll": 1.3558179483314004,
              "struct.elastic_dpitch": 1.3558179483314004, "struct.elastic_dyaw": 1.3558179483314004}
    for k, v in expect.items():
        assert math.isclose(s[k], v), k
    assert V.scalars_from_raw({"flex.tip_w_ft_R": 1.0}) == {"struct.wingR_tip_dz": -FT}     # ER's flex.<key> names too
    assert {r["sb"].split(" ")[0] for r in V.SIGN_TABLE} >= {"dz", "dy", "wing", "wingR.twist", "wingL.twist", "htail.twist",
                                                            "vtail.twist"}


def test_raw_keys_match_fd_keys_when_available():
    try:
        sys.path.insert(0, os.environ.get("FLIGHT_DYNAMICS_DIR") or os.path.join(os.path.dirname(SB), "flight-dynamics"))
        import flexbody  # read-only import (no bytecode)
    except Exception as e:  # noqa: BLE001
        print("skip (FD not importable):", e)
        return
    assert set(V.RAW_KEYS) == set(flexbody.DIAG_KEYS_V2) | set(flexbody.TELEMETRY_KEYS_V2)
    assert set(V.NODE_FIELDS) == set(flexbody.NODE_BODIES)


def test_geometry_from_layout_ft_to_m_and_htail_polyline():
    g = V.geometry_from_layout(_layout(), "X")
    assert not g["layout_estimated"] and g["rp_offset_body_m"] == [round(0.5 * FT, 6), 0.0, round(-0.2 * FT, 6)]
    ht = g["components"]["htail"]
    assert len(ht["axis_nodes_body_m"]) == 10                                         # 5 + 5 (roots at y = +-1 ft)
    assert ht["order"][0] == ("htL", 4) and ht["order"][4] == ("htL", 0) and ht["order"][5] == ("htR", 0)
    ys = [p[1] for p in ht["axis_nodes_body_m"]]
    assert ys == sorted(ys) and ys[0] < 0 < ys[-1]                                      # left tip -> right tip
    assert math.isclose(g["x_tail_m"], -20.0 * FT)                                      # fusV tip
    assert g["components"]["wingR"]["dof"] == ["dz", "dx", "twist"]
    assert g["components"]["vtail"]["axis_nodes_body_m"][-1][2] < 0                     # fin up = -z
    lay = _layout()
    for c in lay["components"]:
        if c["name"] in ("htR", "htL"):
            c["axis_nodes_body_ft"][0] = [-20.0, 0.0, 0.0]
    assert len(V.geometry_from_layout(lay)["components"]["htail"]["axis_nodes_body_m"]) == 9   # shared root node once


def test_nodal_mapping_every_sign():
    g = V.geometry_from_layout(_layout())
    n5 = lambda a: [a * i / 4 for i in range(5)]  # noqa: E731
    nodes = {"wingR": {"w_ft": n5(1.0), "theta_deg": n5(2.0), "v_ft": n5(0.1)},
             "wingL": {"w_ft": n5(0.5), "theta_deg": n5(1.0), "v_ft": n5(0.2)},
             "htR": {"w_ft": n5(0.3), "theta_deg": n5(0.4)}, "htL": {"w_ft": n5(-0.1), "theta_deg": n5(-0.2)},
             "vt": {"w_ft": n5(0.25), "theta_deg": n5(0.6)}, "fusV": {"w_ft": n5(0.05)}, "fusL": {"w_ft": n5(0.02)}}
    raw = {"ht_incidence_deg": 0.3, "vt_sideslip_deg": -0.1, "tip_w_ft_R": 1.0}
    ch = V.map_v2_record(raw, g, nodes)
    assert math.isclose(ch["wingR.dz.4"], -FT) and math.isclose(ch["wingL.dz.4"], -0.5 * FT)       # w + up -> dz < 0
    assert math.isclose(ch["wingR.twist.4"], 2 * DEG) and math.isclose(ch["wingL.twist.4"], -1 * DEG)  # L flips
    assert math.isclose(ch["wingR.dx.4"], -0.1 * FT) and math.isclose(ch["wingL.dx.4"], -0.2 * FT)    # + aft -> dx < 0
    assert math.isclose(ch["fuselage.dz.4"], -0.05 * FT) and math.isclose(ch["fuselage.dy.4"], 0.02 * FT)
    x_tail, inc, vs = g["x_tail_m"], 0.3 * DEG, -0.1 * DEG
    ht = g["components"]["htail"]["axis_nodes_body_m"]
    for i, (body, j) in enumerate(g["components"]["htail"]["order"]):
        w = nodes[body]["w_ft"][j]
        assert math.isclose(ch[f"htail.dz.{i}"], -FT * (w + 0.05) - inc * (ht[i][0] - x_tail), abs_tol=1e-12)
        assert math.isclose(ch[f"htail.dy.{i}"], FT * 0.02 + vs * (ht[i][0] - x_tail), abs_tol=1e-12)
        assert math.isclose(ch[f"htail.twist.{i}"], DEG * nodes[body]["theta_deg"][j], abs_tol=1e-15)  # elastic only
    assert ch["htail.twist.0"] < 0 < ch["htail.twist.9"]                                 # L / R independent
    vtn = g["components"]["vtail"]["axis_nodes_body_m"]
    assert math.isclose(ch["vtail.dy.4"], FT * (0.25 + 0.02) + vs * (vtn[4][0] - x_tail))
    assert math.isclose(ch["vtail.dz.4"], -FT * 0.05 - inc * (vtn[4][0] - x_tail))
    assert math.isclose(ch["vtail.twist.4"], -0.6 * DEG)                                 # FD + = LE toward +y -> ours -
    assert V.component_status(g, nodes) == {c: "fd_nodes" for c in V.COMPONENTS}
    er = V.map_v2_record(raw, g, nodes, components=("htail", "vtail", "fuselage"), scalars=False)
    assert not any(k.startswith(("wing", "struct.")) for k in er)


def test_incidence_never_enters_htail_twist():
    g = V.geometry_from_layout(_layout())
    ch = V.map_v2_record({"ht_incidence_deg": 5.0, "fusV_tip_w_ft": 0.0, "ht_tip_w_ft": 0.0}, g)
    assert all(ch[f"htail.twist.{i}"] == 0.0 for i in range(10))
    assert ch["struct.ht_incidence"] == 5 * DEG
    # nose-up incidence: the tail LE (forward of the fuselage tip) goes up (dz < 0), the TE region down
    ht = g["components"]["htail"]["axis_nodes_body_m"]
    for i, p in enumerate(ht):
        dxr = p[0] - g["x_tail_m"]
        assert (ch[f"htail.dz.{i}"] < 0) == (dxr > 0) or abs(dxr) < 1e-12


def test_estimate_without_nodes_flagged_and_shapes():
    g = V.geometry_from_layout(_layout())
    raw = {"tip_w_ft_R": 1.0, "tip_twist_R_deg": 2.0, "wingR_tip_ip_ft": 0.1, "ht_tip_w_ft": 0.2, "htL_tip_w_ft": -0.2,
           "ht_tip_twist_deg": 0.5, "htL_tip_twist_deg": -0.5, "vt_tip_w_ft": 0.3, "vt_tip_twist_deg": 1.0,
           "fusV_tip_w_ft": 0.0, "fusL_tip_w_ft": 0.0}
    ch = V.map_v2_record(raw, g)
    assert math.isclose(ch["wingR.dz.4"], -FT) and math.isclose(ch["wingR.dz.2"], -FT * V.phi(0.5))
    assert math.isclose(ch["wingR.twist.2"], 2 * DEG * V.psi(0.5)) and math.isclose(ch["wingR.dx.4"], -0.1 * FT)
    assert ch["htail.dz.0"] > 0 > ch["htail.dz.9"] and ch["htail.twist.0"] < 0 < ch["htail.twist.9"]
    assert math.isclose(ch["vtail.twist.4"], -DEG)
    assert V.component_status(g, None) == {c: "estimated" for c in V.COMPONENTS}
    st = V.structure_block(g, estimated=V.component_status(g, None))
    assert all(c["estimated"] for c in st["components"]) and st["v2_map"]["estimated_components"] == list(V.COMPONENTS)
    assert "ESTIMATED" in st["components"][0]["node_values"] and "estimate_shapes" in st["v2_map"]
    assert V.validate_structure(st, ch) == []
    old = V.map_v2_record({"ht_tip_w_ft": 0.2}, g)                                       # pre-telemetry file: htL mirrored
    assert math.isclose(old["htail.dz.0"], old["htail.dz.9"])
    assert V.phi(0) == 0 and V.phi(1) == 1 and V.psi(0) == 0 and V.psi(1) == 1


def test_structure_block_mixed_and_base():
    g = V.geometry_from_layout(_layout())
    base = {"schema": "evolution-flex-state/2", "components": [
        {"name": "wingR", "axis_nodes_body_m": [[0, i, 0] for i in range(9)], "dof": ["dz", "dy", "twist"]}]}
    st = V.structure_block(g, components=("wingR", "htail"), estimated={"htail": "fd_nodes"}, base=base)
    assert [c["name"] for c in st["components"]] == ["wingR", "htail"] and st["components"][0] is not base["components"][0]
    assert st["components"][1]["estimated"] is False and st["v2_map"]["added_components"] == ["htail"]
    assert st["units"]["dx"].startswith("m (body x")


def test_nodes_frame():
    vals = {"htR": {"w_ft": [[0.0, 0.1], [0.0, 0.2]]}, "bogus": {"w_ft": [[1]]}}
    assert V.nodes_frame(vals, 1) == {"htR": {"w_ft": [0.0, 0.2]}}
    assert V.nodes_frame({"vt": {"w_ft": (0.0, 0.5)}}) == {"vt": {"w_ft": [0.0, 0.5]}}
    try:
        V.nodes_frame(vals)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_validate_structure_catches_problems():
    assert V.validate_structure({}, []) == ["structure.components missing or empty"]
    st = {"components": [{"name": "htail", "axis_nodes_body_m": [[0, -1, 0], [0, 1, 0]], "dof": ["dz", "bend"]}]}
    p = V.validate_structure(st, ["htail.dz.0", "htail.dz.1", "htail.dz.5"])
    assert any("unknown dof 'bend'" in x for x in p) and any("beyond the last node" in x for x in p)
    p = V.validate_structure({"components": [{"name": "vtail", "axis_nodes_body_m": [[0, 0, 0], [0, 0, float("nan")]],
                                              "dof": ["dy", "dx"]}]}, ["vtail.dy.0", "vtail.dx.0", "vtail.dx.1"])
    assert any("finite" in x for x in p) and any("no channel for nodes [1]" in x for x in p)


def test_convert_traj_with_and_without_nodes():
    lay = _layout()
    raw_names = ["flex." + k for k in sorted(FX["frames"][0]["raw"])]
    doc = {"schema": "ga-flightsim-traj/1", "aircraft": "T38", "sim_dt_s": 1 / 120, "channels": ["t", "x"] + raw_names,
           "data": [[k * 4 / 120, 0.0] + [f["raw"][n[5:]] for n in raw_names] for k, f in enumerate(FX["frames"])],
           "structure": None, "synthetic": True}
    out = V.convert_traj(doc, geometry=GEO)
    assert out["structure"]["synthetic"] is True and out["structure"]["synthetic_doc"] == V.SYNTHETIC_DOC
    assert V.validate_structure(out["structure"], out["channels"]) == []
    assert set(out["structure"]["v2_map"]["estimated_components"]) == set(V.COMPONENTS)
    i = out["channels"].index("struct.wingR_tip_dz")
    assert all(math.isclose(r[i], round(-f["raw"]["tip_w_ft_R"] * FT, 6)) for r, f in zip(out["data"], FX["frames"]))
    nf = 4 * len(FX["frames"])
    lay["values"] = {c["name"]: {"w_ft": [[0.01 * k * i for i in range(5)] for k in range(nf)]} for c in lay["components"]}
    out2 = V.convert_traj(doc, nodes_doc=lay)
    assert out2["structure"]["v2_map"]["node_status"] == {c: "fd_nodes" for c in V.COMPONENTS}
    j = out2["channels"].index("fuselage.dz.4")
    assert math.isclose(out2["data"][1][j], round(-FT * 0.01 * 3 * 4, 6))                  # row t = 4 dt -> frame 3
    assert V.convert_traj({"channels": ["t"], "data": [[0.0]]}) == {"channels": ["t"], "data": [[0.0]]}


class _FakeFDM(dict):
    def __missing__(self, k):
        return 0.0


class _Sc:
    ramp_fpm = None

    def target(self, t):
        return (1000.0,)

    def target_cmd(self, t):
        return (1000.0,)


def test_recorder_maps_a_raw_v2_dict_as_estimated():
    fdm = _FakeFDM({"position/lat-geod-deg": 37.0, "position/long-gc-deg": -122.0, "position/h-sl-ft": 1000.0})
    rec = TrajRecorder(_Sc(), 30, 1 / 120, timing="post", aircraft="T38")
    for k, f in enumerate(FX["frames"]):
        rec(k * 4 / 120, fdm, dict(f["raw"]))
    rec.finish()
    st = rec.structure
    assert st and V.validate_structure(st, rec.channels) == []
    assert {c["name"] for c in st["components"]} == set(V.COMPONENTS) and all(c["estimated"] for c in st["components"])
    assert rec.node_status() == {c: "estimated" for c in V.COMPONENTS}
    assert "struct.wingR_root_bm" in rec.channels and "flex.wingR_bm" in rec.channels
    i = rec.channels.index("struct.wingL_tip_twist")
    assert math.isclose(rec.rows[0][i], -math.radians(FX["frames"][0]["raw"]["tip_twist_L_deg"]))


class _FakeFlexState:
    """ER FlexState look-alike: modal wings (9 nodes) + raw + eta."""
    def __init__(self, raw):
        self.raw, self.eta = raw, [0.0]
        self.structure = {"schema": "evolution-flex-state/2", "components": [
            {"name": n, "axis_nodes_body_m": [[0, s * (1 + i), 0] for i in range(9)], "dof": ["dz", "dy", "twist"]}
            for n, s in (("wingR", 1), ("wingL", -1))]}

    def channels(self):
        ch = {f"{n}.{d}.{i}": (-0.01 * i if d == "dz" else 0.0) for n in ("wingR", "wingL") for d in ("dz", "dy", "twist") for i in range(9)}
        ch.update({"flex." + k: v for k, v in self.raw.items()})
        return ch


class _FakeNodes:
    """NodeSource look-alike on _layout(): 33-node wings would be FD's; here 5 nodes per body."""
    def layout(self, rp):
        return _layout()["components"]

    def frame(self, eta):
        return {b: {"w_ft": [0.1 * i for i in range(5)], **({"theta_deg": [0.0] * 5} if b not in ("fusV", "fusL") else {}),
                    **({"v_ft": [0.01 * i for i in range(5)]} if b.startswith("wing") else {})} for b in V.FD_BODIES}


def test_recorder_with_nodes_replaces_er_wings_without_nodes_keeps_them():
    fdm = _FakeFDM({"position/lat-geod-deg": 37.0, "position/long-gc-deg": -122.0, "position/h-sl-ft": 1000.0})
    raw = dict(FX["frames"][0]["raw"])
    rec = TrajRecorder(_Sc(), 30, 1 / 120, timing="post", aircraft="T38", node_source=_FakeNodes())
    rec(0.0, fdm, _FakeFlexState(raw))
    rec.finish()
    st = rec.structure
    assert rec.node_status() == {c: "fd_nodes" for c in V.COMPONENTS} and V.validate_structure(st, rec.channels) == []
    assert "wingR.dx.4" in rec.channels and "wingR.dy.0" not in rec.channels and "wingR.dz.8" not in rec.channels
    assert not any(c["estimated"] for c in st["components"])
    rec2 = TrajRecorder(_Sc(), 30, 1 / 120, timing="post", aircraft="T38")
    rec2(0.0, fdm, _FakeFlexState(raw))
    rec2.finish()
    st2 = rec2.structure
    names = {c["name"]: c for c in st2["components"]}
    assert names["wingR"]["estimated"] is False and "ER modal" in names["wingR"]["node_values_source"]
    assert names["htail"]["estimated"] is True and "wingR.dz.8" in rec2.channels and "wingR.dx.0" not in rec2.channels


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


def test_no_component_for_bodies_fd_does_not_model():
    """Reduced fidelity (flexwing v1: wings only, v1 key names) must not get all-zero 'estimated' tail / fin /
    fuselage components: only components with FD nodes or at least one FD tip key are added."""
    v1 = {"tip_w_ft_R": 0.2, "tip_w_ft_L": 0.19, "tip_twist_deg_R": 0.3, "root_bm_lbft_R": 1e4, "dL_lbf": 5.0}
    assert V.available_components(v1) == ["wingR", "wingL"]
    assert V.available_components(v1, components=("htail", "vtail", "fuselage")) == []
    assert V.available_components({"fusV_tip_w_ft": 0.0}) == ["fuselage"]          # exported zero still counts
    full = {b: {"w_ft": [0.0]} for b in V.FD_BODIES}
    assert V.available_components({}, full) == list(V.COMPONENTS)
    fdm = _FakeFDM({"position/lat-geod-deg": 37.0, "position/long-gc-deg": -122.0, "position/h-sl-ft": 1000.0})
    rec = TrajRecorder(_Sc(), 30, 1 / 120, timing="post", aircraft="c172x")
    st = _FakeFlexState(v1)
    for k in range(3):
        rec(k * 4 / 120, fdm, st)
    rec.finish()
    names = [c["name"] for c in rec.structure["components"]]
    assert names == ["wingR", "wingL"] and not any(c.startswith(("htail.", "vtail.", "fuselage.")) for c in rec.channels)
    assert "wingR.dz.8" in rec.channels and "struct.wingR_tip_dz" in rec.channels
