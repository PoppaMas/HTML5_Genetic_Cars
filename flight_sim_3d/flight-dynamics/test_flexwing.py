"""pytest -q test_flexwing.py"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flexwing as fw  # noqa: E402


def uniform_wing(**kw):
    """Uniform, unswept, untapered cantilever (closed-form beam/torsion results apply)."""
    base = dict(span_ft=40.0, area_ft2=200.0, taper=1.0, sweep_deg=0.0, root_frac=0.0, n_strips=200,
                ei_taper_exp=0.0, mass_taper_exp=0.0, x_ea=0.40, x_cg=0.40, cla=2 * math.pi,
                wing_mass_lb=300.0, f_b1_hz=6.0, f_t1_hz=20.0, zeta=0.0)
    base.update(kw)
    with fw.unchecked_section_axes():      # analytic validation case: uncoupled section (x_cg == x_ea)
        return fw.FlexWing(fw.WingParams(**base))


def test_mode_shapes_cantilever():
    xi = np.array([0.0, 1.0])
    for n in range(2):
        assert abs(fw._bend(n, xi)[0]) < 1e-12 and abs(fw._bend(n, xi)[1] - 1) < 1e-12
        assert abs(fw._bend(n, xi, 1)[0]) < 1e-9                    # clamped slope
        assert abs(fw._bend(n, xi, 2)[1]) < 1e-5                    # free end: zero moment
    assert abs(fw._tors(1.0, 1)) < 1e-12                            # free end: zero torque


def test_frequency_calibration_and_uniform_ratio():
    w = uniform_wing()
    f = w.frequencies_hz()
    assert abs(f[0] - 6.0) < 0.01 and abs(f[1] - 20.0) < 0.05 or abs(f[2] - 20.0) < 0.05
    # 2nd bending / 1st bending of a uniform cantilever = (4.694/1.875)^2 = 6.267
    fb = sorted([f for f in f if abs(f - 20.0) > 0.05])
    assert abs(fb[1] / fb[0] - 6.267) < 0.03


def test_static_tip_deflection_vs_beam_theory():
    w = uniform_wing()
    q = 50.0  # lbf/ft
    exact = q * w.L ** 4 / (8 * w.EI_root)
    assert abs(w.static_tip_deflection_uniform(q) / exact - 1) < 0.02


def test_torsional_divergence_exact():
    w = uniform_wing()
    e = (w.p.x_ea - 0.25) * w.c[0]
    exact = (math.pi / 2) ** 2 * w.GJ_root / (e * w.c[0] * w.a * w.L ** 2)
    assert abs(w.divergence_q() / exact - 1) < 0.01
    # textbook quasi-steady estimate from the brief: q_div = K_theta/(e c S Cla) with K_theta = GJ*(pi/2)^2/L
    assert w.divergence_q() > 0


def test_coalescence_matches_semi_analytic_typical_section():
    # 2-DOF typical section (h up +, theta nose-up +), steady aero, AC ahead of EA by e*c
    m, S, Ia, kh, kt = 2.0, 0.3, 0.5, 400.0, 3000.0
    a, c, e = 2 * math.pi, 1.0, 0.15
    M = np.array([[m, -S], [-S, Ia]])
    K = np.diag([kh, kt])
    A = np.array([[0.0, c * a], [0.0, e * c * c * a]])          # Q = q*A@[h, theta]
    q_num = fw.coalescence_q_matrices(M, K, A, 2000.0, 2000)
    # closed form: det(B - w2 M) = 0, B = K - qA; discriminant(q) = 0
    detM = np.linalg.det(M)
    def disc_coeffs():
        # b(q) = -(M11 B22 + M22 B11 - M12 B21 - M21 B12), c(q) = det B; both polynomial in q
        qs = np.array([0.0, 1.0, 2.0])
        vals = []
        for q in qs:
            B = K - q * A
            b = -(M[0, 0] * B[1, 1] + M[1, 1] * B[0, 0] - M[0, 1] * B[1, 0] - M[1, 0] * B[0, 1])
            cc = np.linalg.det(B)
            vals.append(b * b - 4 * detM * cc)
        return np.polyfit(qs, vals, 2)
    roots = np.roots(disc_coeffs())
    roots = sorted(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0)
    assert abs(q_num / roots[0] - 1) < 2e-3


def test_newmark_free_vibration_energy_and_period():
    w = uniform_wing()
    M, K = w.M, w.K
    dt = 1e-4
    n = int(0.5 / dt)
    x0 = np.zeros(w.n_modes); x0[0] = 0.1
    x, v, a = fw.newmark(M, np.zeros_like(M), K, np.zeros((n, w.n_modes)), dt, x0=x0)
    E = 0.5 * np.einsum("ij,jk,ik->i", v, M, v) + 0.5 * np.einsum("ij,jk,ik->i", x, K, x)
    assert np.max(np.abs(E / E[0] - 1)) < 1e-9          # average-acceleration Newmark conserves energy
    zc = np.where(np.diff(np.sign(x[:, 0])) > 0)[0]
    period = np.mean(np.diff(zc)) * dt
    fb1 = 1.0 / period
    assert abs(fb1 / w.frequencies_hz()[0] - 1) < 0.01


def test_modal_response_to_load_history():
    w = uniform_wing(zeta=0.05)
    dt, n = 1 / 480, 960 * 4
    # slow ramp to a uniform 40 lbf/ft load -> quasi-static solution
    ramp = np.clip(np.arange(n) * dt / 4.0, 0, 1)
    lift = ramp[:, None] * 40.0 * w.dy * np.ones((1, w.y.size))
    r = fw.modal_response(w, lift, dt)
    static = 40.0 * w.L ** 4 / (8 * w.EI_root)
    assert abs(r["tip_w_ft"][-1] / static - 1) < 0.03
    # root bending moment (force summation) -> q L^2 / 2 for a uniform load
    assert abs(r["root_bm_lbft"][-1] / (40.0 * w.L ** 2 / 2) - 1) < 0.01
    # lift at the quarter chord ahead of the EA twists the wing nose-up
    assert r["tip_twist_deg"][-1] > 0


def test_margin_trends():
    p = dict(span_ft=36.0, area_ft2=174.0)
    base = fw.FlexWing(fw.params_for("c172x", **p))
    soft = fw.FlexWing(fw.params_for("c172x", gj_scale=0.5, **p))
    assert soft.divergence_q() < base.divergence_q()
    aft = fw.FlexWing(fw.params_for("c172x", x_cg=0.50, **p)).margins()
    bal = fw.FlexWing(fw.params_for("c172x", x_cg=0.50, tip_mass_frac=0.08, **p)).margins()
    assert math.isfinite(aft["v_flutter_keas"])
    assert bal["flutter_margin"] > aft["flutter_margin"]          # LE mass balance raises flutter speed
    swept = fw.FlexWing(fw.params_for("c172x", sweep_deg=25.0, **p))
    assert swept.divergence_q() > base.divergence_q()             # sweep-back bending/twist coupling (wash-out)
    # GJ calibration is independent of the x_ea / x_cg genes
    assert fw.FlexWing(fw.params_for("c172x", x_cg=0.5, **p)).GJ_root == pytest.approx(base.GJ_root)


def test_struct_genes_decode():
    lo = fw.decode_struct([0] * len(fw.STRUCT_SCHEMA)); hi = fw.decode_struct([1] * len(fw.STRUCT_SCHEMA))
    for g in fw.STRUCT_SCHEMA:
        assert lo[g.name] == pytest.approx(g.min) and hi[g.name] == pytest.approx(g.max)
    terms = fw.margin_terms(fw.FlexWing(fw.params_for("c172x", 36.0, 174.0)))
    assert terms["fail"] is None and terms["terms"]["J_mass"] == pytest.approx(0.0)


GEO = {"c172x": (36.0, 174.0, 1454.0), "737": (94.7, 1171.0, 83000.0), "T38": (25.25, 170.0, 7574.0),
       "f16": (30.0, 300.0, 17400.0)}


def _all_finite(m):
    for k, v in m.items():
        vals = v if isinstance(v, list) else [v]
        for x in vals:
            assert not isinstance(x, float) or math.isfinite(x), (k, x)


def test_cg_ahead_of_ea_gives_finite_capped_margin():
    # the GA exploit: section CG (0.31c) ahead of the elastic axis (0.48c). Previously flutter_margin = inf on c172x.
    for model, (b, S, ew) in GEO.items():
        with fw.unchecked_section_axes():
            w = fw.FlexWing(fw.params_for(model, b, S, ew, x_ea=0.48, x_cg=0.31))
        m = w.margins()
        _all_finite(m)
        assert 0.0 < m["flutter_margin"] <= fw.MARGIN_CAP and 0.0 < m["div_margin"] <= fw.MARGIN_CAP
        assert not m["margin_error"]
    with fw.unchecked_section_axes():
        w = fw.FlexWing(fw.params_for("c172x", 36.0, 174.0, 1454.0, x_ea=0.48, x_cg=0.31))
    m = w.margins()
    assert m["flutter_margin"] == fw.MARGIN_CAP and m["flutter_not_found_below_cap"] is True
    assert m["v_flutter_keas"] == pytest.approx(fw.MARGIN_CAP * w.p.v_dive_keas) and m["f_flutter_hz"] == 0.0
    # no aerodynamic coupling at all -> nothing anywhere -> both margins capped and flagged, still finite
    w0 = fw.FlexWing(fw.params_for("c172x", 36.0, 174.0, 1454.0))
    for k in ("A_K", "A_C", "A_C_circ", "A_C_nc"):
        setattr(w0, k, np.zeros_like(getattr(w0, k)))
    m0 = w0.margins()
    _all_finite(m0)
    assert m0["flutter_margin"] == m0["div_margin"] == fw.MARGIN_CAP
    assert m0["flutter_not_found_below_cap"] and m0["div_not_found_below_cap"]
    # cap is honoured in the fitness: no hard fail, zero penalty, bounded
    t = fw.margin_terms(w0)
    assert t["fail"] is None and t["terms"]["J_flutter_margin"] == 0.0


def test_section_axes_guard_raises():
    with pytest.raises(ValueError):
        fw.WingParams(x_ea=0.48, x_cg=0.31)
    with pytest.raises(ValueError):
        fw.params_for("c172x", 36.0, 174.0, x_ea=0.40, x_cg=0.41)          # gap 0.01 < 0.02
    p = fw.params_for("c172x", 36.0, 174.0)
    p.x_cg = 0.30                                                         # mutation after construction
    with pytest.raises(ValueError):
        fw.FlexWing(p)
    for g in ("section_cg_frac", "elastic_axis_frac", "x_cg", "x_ea", "tip_mass_frac"):
        with pytest.raises(ValueError):
            fw.overrides_from_genome({g: 0.4})
    assert [g.name for g in fw.STRUCT_SCHEMA] == ["stiffness_scale", "torsion_bend_ratio", "zeta", "nonstruct_scale"]
    for model, (b, S, ew) in GEO.items():                                # fixed defaults satisfy the rule
        q = fw.params_for(model, b, S, ew)
        assert q.x_cg >= q.x_ea + fw.MIN_CG_AFT_OF_EA - 1e-9


def test_default_margins_unchanged_per_aircraft():
    # values before the cap change (flexwing as of 2026-10-06 03:30): flutter identical; divergence above the
    # cap (swept 737/T38: 23.5 / 25.7 V_D) is now reported as 3.0 with the not-found-below-cap flag
    ref = {"c172x": (1.2386, 1.673), "737": (1.2308, 23.51), "T38": (1.2574, 25.69),
           "f16": (1.3895, 40.37)}   # f16 added 2026-10-06 04:44 PT (x_ea 0.40 / x_cg 0.43); raw divergence 40 V_D -> capped
    for model, (b, S, ew) in GEO.items():
        m = fw.FlexWing(fw.params_for(model, b, S, ew)).margins()
        fl, dv = ref[model]
        assert m["flutter_margin"] == pytest.approx(fl, abs=1e-3) and not m["flutter_not_found_below_cap"]
        if dv < fw.MARGIN_CAP:
            assert m["div_margin"] == pytest.approx(dv, abs=1e-3) and not m["div_not_found_below_cap"]
        else:
            assert m["div_margin"] == fw.MARGIN_CAP and m["div_not_found_below_cap"]


TIED_S = sorted(set(np.round(np.exp(np.linspace(np.log(0.6), np.log(2.0), 11)), 6)) | {0.8})
TIED_R = list(np.linspace(0.8, 1.15, 8))


def test_tied_stiffness_grid_finite_monotone_no_ridge():
    """Sample the tied space on c172x/T38/737/f16: margins finite, smooth/monotone (no torsion-2nd-bending ridge),
    margin >= 1.0 for stiffness_scale >= 0.8 (except the documented soft-torsion corner, see INTERFACE.md)."""
    for model, (b, S, ew) in GEO.items():
        G = np.zeros((len(TIED_R), len(TIED_S)))
        for i, r in enumerate(TIED_R):
            for j, sc in enumerate(TIED_S):
                ov = fw.overrides_from_genome({"stiffness_scale": sc, "torsion_bend_ratio": r})
                assert ov["ei_scale"] == pytest.approx(sc) and ov["gj_scale"] == pytest.approx(sc * r)
                m = fw.FlexWing(fw.params_for(model, b, S, ew, **ov)).margins()
                _all_finite(m)
                assert not m["margin_error"]
                G[i, j] = min(m["flutter_margin"], m["div_margin"])
        # no ridge: the margin never drops when either stiffness or the torsion/bending ratio increases
        assert np.all(np.diff(G, axis=0) >= -1e-6), model
        assert np.all(np.diff(G, axis=1) >= -1e-6), model
        s_arr = np.array(TIED_S)
        r_arr = np.array(TIED_R)
        # >= 1.0 for s >= 0.8 everywhere except the soft-torsion corner (s < 0.86 with r < 0.85:
        # GJ x = 0.64..0.68, c172x 0.967 / T38 0.999 at s = r = 0.8) -- smooth, not a ridge
        ok = (s_arr[None, :] >= 0.8 - 1e-9) & ~((s_arr[None, :] < 0.86) & (r_arr[:, None] < 0.85))
        assert G[ok].min() >= 1.0, (model, G[ok].min())
        assert G[:, s_arr >= 0.8 - 1e-9].min() >= 0.96, model


def test_genome_tied_ranges_and_untied_keys_raise():
    for bad in ({"stiffness_scale": 0.59}, {"stiffness_scale": 2.01}, {"torsion_bend_ratio": 0.79},
                {"torsion_bend_ratio": 1.16}, {"struct_damping_ratio": 0.08}, {"nonstructural_mass_scale": 1.3},
                {"stiffness_scale": float("nan")}):
        with pytest.raises(ValueError):
            fw.overrides_from_genome(bad)
    for k in ("bend_stiffness_scale", "torsion_stiffness_scale", "ei_scale", "gj_scale"):
        with pytest.raises(ValueError):
            fw.overrides_from_genome({k: 1.0})
    with pytest.raises(ValueError):                                       # raw WingParams field via genome
        fw.overrides_from_genome({"zeta": 0.02})
    assert fw.overrides_from_genome({}) == {"ei_scale": 1.0, "gj_scale": 1.0}
    assert fw.genes_to_overrides(fw.decode_struct([0, 1, 0.5, 0.5]))["gj_scale"] == pytest.approx(0.6 * 1.15)


# ---------------- JSBSim-coupled tests ----------------
jsbsim = pytest.importorskip("jsbsim")
import coupled_sim as cs  # noqa: E402


def test_fast_generalised_forces_match_reference():
    cs.ensure_root("c172x")
    fdm = fw.new_fdm("c172x", cs.ROOT)
    cs.trim(fdm, 4000, 100)
    fdm["fcs/aileron-cmd-norm"] = 0.4
    for _ in range(30):
        fdm.run()
    c = cs.make_coupler(fdm, "c172x", "oneway")
    c.initialize(fdm)
    s = c.read_state(fdm)
    fast = c.gen_forces_fast(s)
    for side in (0, 1):
        L, Mea = c.external_loads(s, side)
        assert np.allclose(fast[side], c.gen_force(L, Mea, s, side), rtol=1e-10, atol=1e-8)


def test_coupled_c172x_deterministic_and_differs_from_rigid():
    kw = dict(duration_s=7.0)
    r1 = cs.run_maneuver("c172x", "twoway", **kw)
    r2 = cs.run_maneuver("c172x", "twoway", **kw)
    rig = cs.run_maneuver("c172x", "oneway", **kw)
    for k in ("alt_ft", "p_dps", "nz", "tip_w_ft_R", "root_bm_lbft_L"):
        assert np.array_equal(r1[k], r2[k]), k                     # bit-identical repeat
    assert all(np.all(np.isfinite(r1[k])) for k in ("alt_ft", "tip_w_ft_R", "tip_twist_deg_R"))
    # zero feedback at the trim point (reference = 1-g trim shape)
    assert abs(r1["dL_lbf"][0]) < 1e-6 * 2480 and abs(r1["fbz_ext"][0]) < 1e-3
    assert np.all(rig["fbz_ext"] == 0.0)
    # flexible differs, but sensibly: aileron effectiveness reduced by elastic twist (straight wing),
    # roll rate within 5-30 % of rigid, Nz within 0.2 g
    ratio = r1["p_dps"].max() / rig["p_dps"].max()
    assert 0.70 < ratio < 0.97
    assert np.max(np.abs(r1["nz"] - rig["nz"])) < 0.2
    assert np.max(np.abs(r1["nz"] - rig["nz"])) > 1e-3
    # 1-g root bending moment is in the right ballpark: half weight x ~0.3-0.5 semispan arm
    assert 0.25 * 1240 * 18 < r1["m_root_1g"] < 0.55 * 1240 * 18


def test_softer_wing_bends_more():
    a = cs.run_maneuver("c172x", "twoway", duration_s=3.0)
    b = cs.run_maneuver("c172x", "twoway", dict(ei_scale=0.5, gj_scale=0.5), duration_s=3.0)
    assert np.abs(b["tip_w_ft_R"]).max() > 1.5 * np.abs(a["tip_w_ft_R"]).max()


def test_wing_mass_feeds_jsbsim_weight_and_inertia():
    cs.ensure_root("c172x")
    out = {}
    for name, ov in (("base", {}), ("heavy", dict(ei_scale=2.0, gj_scale=2.0)), ("light", dict(ei_scale=0.5, gj_scale=0.5))):
        fdm = fw.new_fdm("c172x", cs.ROOT)
        c = cs.make_coupler(fdm, "c172x", "twoway", ov)
        fdm["ic/h-sl-ft"] = 4000; fdm["ic/vc-kts"] = 100; fdm.run_ic()
        out[name] = (c.delta_mass_lb, fdm["inertia/weight-lbs"], fdm["inertia/ixx-slugs_ft2"])
    assert out["base"][0] == 0.0
    for k in ("heavy", "light"):
        assert out[k][1] - out["base"][1] == pytest.approx(out[k][0], abs=1e-6)     # weight follows the mass genes
    assert out["heavy"][2] > out["base"][2] > out["light"][2]                         # and so does Ixx


def test_genome_alias_mapping_and_channels():
    ov = fw.overrides_from_genome({"stiffness_scale": 1.5, "torsion_bend_ratio": 0.9, "struct_damping_ratio": 0.03,
                                   "wing_mass_scale": 1.1, "aspect_ratio_delta": 0.0})
    assert ov == pytest.approx({"ei_scale": 1.5, "gj_scale": 1.35, "zeta": 0.03, "nonstruct_scale": 1.1})
    with pytest.raises(ValueError):
        fw.overrides_from_genome({"sweep_delta_deg": 2.0}, strict=True)
    r = cs.run_maneuver("c172x", "twoway", duration_s=2.0)
    hist = {k: r[k] for k in ("root_bm_lbft_R", "root_bm_lbft_L", "tip_twist_deg_R", "tip_twist_deg_L", "tip_w_ft_R", "tip_w_ft_L")}
    w = fw.FlexWing(fw.params_for("c172x", 36.0, 174.0))
    ch = fw.telemetry_channels(hist, w, r["m_root_1g"])
    assert ch["wing_root_bending"].shape == r["t"].shape
    assert ch["params"]["wing_root_bending_limit"] == pytest.approx(3.8 * r["m_root_1g"])


# ---------------- prepared JSBSim copies (jsbsim_root) ----------------
def test_prepared_copies_have_no_network_sockets(tmp_path):
    """prepare_aircraft strips every network <input>/<output> (737 telnet 5137 / QTJSBSIM UDP 5139, commented UDP
    5138 output; c172x commented SOCKET outputs). Checked on a fresh prepare and on the shared ./jsbsim_root."""
    import re as _re
    rx = _re.compile(fw.NET_IO_GREP, _re.I)
    roots = [str(tmp_path)]
    for m in ("c172x", "T38", "737", "f16"):
        fw.prepare_aircraft(m, str(tmp_path))
    if os.path.isdir(cs.ROOT):
        roots.append(cs.ROOT)
    for root in roots:
        for m in ("c172x", "T38", "737", "f16"):
            d = os.path.join(root, "aircraft", m)
            if not os.path.isdir(d):
                continue
            for dirpath, _, files in os.walk(d):
                for fn in files:
                    if fn.endswith(".xml"):
                        txt = open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace").read()
                        assert not rx.search(txt), (root, m, fn, rx.search(txt).group(0))
    meta = __import__("json").load(open(os.path.join(str(tmp_path), "aircraft", "737", "flexwing_meta.json")))
    assert meta["network_io_removed"] == 3 and meta["fmt"] == fw.PREPARE_FMT


def test_f16_prepared_copy_layout(tmp_path):
    import json as _json
    import re as _re
    fw.prepare_aircraft("f16", str(tmp_path))
    d = os.path.join(str(tmp_path), "aircraft", "f16")
    meta = _json.load(open(os.path.join(d, "flexwing_meta.json")))
    assert meta["payload_placeholder_index"] == 0 and (meta["pm_index_R"], meta["pm_index_L"]) == (2, 3)
    xml = open(os.path.join(d, "f16.xml")).read()
    mb = _re.search(r"<mass_balance[^>]*>(.*?)</mass_balance>", xml, _re.S).group(1)
    assert _re.findall(r'<pointmass\s+name="([^"]+)"', mb) == ["payload_placeholder", "Pilot", "flexwing_dm_R", "flexwing_dm_L"]
    assert "flexwing_F" in xml and "flexwing_M" in xml
    fdm = fw.new_fdm("f16", str(tmp_path))
    assert fdm["inertia/pointmass-weight-lbs[0]"] == 0.0 and fdm["inertia/pointmass-weight-lbs[1]"] == 230.0
    assert fdm["inertia/pointmass-location-Y-inches[0]"] == 0.0


def test_f16_trim_350kcas_10kft_gear_up():
    """Evolution Runner trim point; numbers measured 2026-10-06 (f16_check.json)."""
    cs.ensure_root("f16")
    for mode in (1, 0):
        f = fw.new_fdm("f16", cs.ROOT)
        f["gear/gear-cmd-norm"] = 0.0
        f["ic/h-sl-ft"] = 10000.0; f["ic/vc-kts"] = 350.0; f["ic/gamma-deg"] = 0.0; f["ic/psi-true-deg"] = 0.0
        f.run_ic()
        f["propulsion/set-running"] = -1
        f["simulation/do_simple_trim"] = mode
        assert f["aero/alpha-deg"] == pytest.approx(1.045, abs=0.02)
        assert f["fcs/throttle-cmd-norm"] == pytest.approx(0.2836, abs=0.003)
        assert f["fcs/throttle-pos-norm"] == pytest.approx(2 * f["fcs/throttle-cmd-norm"])   # pos = 2 x cmd; MIL = cmd 0.5
        assert f["fcs/throttle-cmd-norm"] < 0.5                                               # dry thrust, no afterburner
        assert f["fcs/elevator-pos-deg"] == pytest.approx(-1.111, abs=0.02)
        assert f["gear/gear-pos-norm"] == 0.0
        th0 = f["attitude/theta-deg"]
        for _ in range(600):
            f.run()
        assert abs(f["attitude/theta-deg"] - th0) < 0.1 and abs(f["velocities/vc-kts"] - 350.0) < 0.5


def test_f16_zero_force_identical_to_stock_and_coupled_deterministic():
    cs.ensure_root("f16")

    def hist(root):
        f = fw.new_fdm("f16", root)
        cs.trim(f, 10000.0, 350.0)
        e0, out = f["fcs/elevator-cmd-norm"], []
        for k in range(360):
            f["fcs/elevator-cmd-norm"] = e0 + (-0.1 if 120 <= k < 240 else 0.0)
            f.run()
            out.append([f[x] for x in ("position/h-sl-ft", "attitude/theta-deg", "velocities/q-rad_sec", "inertia/cg-x-in")])
        return np.array(out)
    assert np.array_equal(hist(None), hist(cs.ROOT))
    kw = dict(duration_s=3.0)
    r1 = cs.run_maneuver("f16", "twoway", **kw)
    r2 = cs.run_maneuver("f16", "twoway", **kw)
    for k in ("alt_ft", "q_dps", "nz", "tip_w_ft_R", "root_bm_lbft_R"):
        assert np.array_equal(r1[k], r2[k]), k
    assert np.max(np.abs(r1["fbz_ext"])) > 0.0
