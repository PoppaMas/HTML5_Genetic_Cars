"""pytest -q test_ctrlsurf_p4r1.py  (Phase 4 r1 control-surface fidelity 'full_a1_b2a_cs1'; jets tracking fix)"""
import json, math, os, subprocess, sys
import numpy as np
import pytest
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import coupled_sim as cs, flexbody as fb, flexeval as fe, flexeval_b2 as fb2, flexeval_p4 as fp4, ctrlsurf_p4 as cs4
import flexeval_p4r1 as fp1, ctrlsurf_p4r1 as cs1, p4r1_scaling as sc1
import flexbody_b2 as b2
MODELS = ("c172x", "T38", "737", "f16")
ROOT = cs.ROOT
fb.blas_threads(1)


@pytest.fixture(scope="module")
def sim():
    for m in MODELS:
        cs.ensure_root(m); fb.ensure_root_v2(m); b2.ensure_root_v2b2(m, fe.root_v2_for(ROOT), fb2.root_v2b2_for(ROOT))
    return fe.load_sim()


def test_old_pins_resolve_bit_for_bit():
    pins = json.load(open(os.path.join(HERE, "v2_results", "model_versions_post_p4cs.json")))["full_a1_b2a_cs"]
    for m in MODELS:
        for mode in ("active", "passthrough"):
            assert fp4.model_version("full_a1_b2a_cs", m, ROOT, cs_mode=mode) == pins[m][mode]
        assert pins[m]["b2a"] == fb2.model_version("full_a1_b2a", m, ROOT)


def test_frozen_md5_unchanged():
    for f in ("FROZEN_A1_B1r1.md5", "FROZEN_B2a.md5", "FROZEN_P4cs.md5"):
        r = subprocess.run(["md5sum", "-c", "--quiet", os.path.join("v2_results", f)], cwd=HERE, capture_output=True, text=True)
        assert "FAILED" not in r.stdout + r.stderr and r.returncode == 0, f


def test_new_fidelity_pin_format_and_distinct():
    new = {}
    for m in MODELS:
        v = fp1.model_version("full_a1_b2a_cs1", m, ROOT)
        assert v.startswith("full_a1_b2a_cs1:p4cs1:active:") and len(v.split(":")[-1]) == 8
        assert v == fp1.model_version("full_a1_b2a_cs1", m, ROOT)
        assert v != fp4.model_version("full_a1_b2a_cs", m, ROOT)
        new[m] = v
    assert len(set(new.values())) == 4
    assert fp1.model_version("full_a1_b2a_cs1", "c172x", ROOT, cs_mode="passthrough") != new["c172x"]
    with pytest.raises(ValueError):
        fp1.evaluate({}, {}, [], "c172x", fidelity="nonsense", root=ROOT)


def test_pins_file_matches_code():
    p = json.load(open(os.path.join(HERE, "v2_results", "model_versions_post_p4r1.json")))["full_a1_b2a_cs1"]
    for m in MODELS:
        for mode in ("active", "passthrough"):
            assert p[m][mode] == fp1.model_version("full_a1_b2a_cs1", m, ROOT, cs_mode=mode)


def test_surface_table_r1():
    for m in MODELS:
        for s in cs1.SURFACES[m]:
            assert s.rate_dps is not None and s.rate_dps > 0, (m, s.name)          # Genome ask 1: no rate-None surface
            assert s.scored == (s.name in ("elev", "ail", "rud"))
            assert s.fbw == (m == "f16" and s.name in ("elev", "ail", "rud"))
            if s.fbw:
                assert s.tau_s is None                                             # no FD lag stacked ahead of the FBW loop
            elif s.scored:
                assert s.tau_s is not None
    # r0 table untouched (frozen module)
    assert any(s.rate_dps is None for s in cs4.SURFACES["T38"])
    assert all(s.tau_s is not None for s in cs4.SURFACES["f16"][:3])


def test_old_fidelity_name_rejected_by_r1():
    with pytest.raises(ValueError):
        fp1.fly_course(None, {}, {}, None, None, {"start": {}}, "full_a1_b2a_cs", model="c172x", guidance=fp1.demo_guidance)


@pytest.mark.parametrize("model", MODELS)
def test_fly_course_r1_contract(sim, model):
    P = fe.default_profile(model, sim)
    c = {"start": {"alt_ft": P.h0_ft, "kcas": P.speed_kts}, "duration_s": 3.0, "demo_heading_deg": 20}
    seen = {}

    def g(s, gains, course, mdl):
        seen.update(trim=s["trim"], q=s["qbar_psf"], M=s["mach"])
        return fp1.demo_guidance(s, gains, course, mdl)
    r = fp1.fly_course(None, {}, {}, None, None, c, model=model, guidance=g, sim=sim)
    assert r["status"] == "ok" and r["fidelity"] == "full_a1_b2a_cs1" and r["model_version"].startswith("full_a1_b2a_cs1:p4cs1:")
    for k in ("aileron", "elevator", "rudder", "throttle", "flap", "speedbrake", "pitch_trim", "roll_trim", "yaw_trim"):
        assert k in seen["trim"] and math.isfinite(seen["trim"][k])
    assert seen["trim"] == r["trim_cmd"] and seen["q"] > 0 and 0 < seen["M"] < 1
    # Genome ask 1: every limit carries a rate; scored flag separates flap / speedbrake
    assert set(r["surface_limits"]) == {"elev", "ail", "rud"}
    for nm, lim in r["surface_limits"].items():
        assert lim[2] is not None and lim[0] < 0 < lim[1]
    for nm, lim in r["surface_limits_all"].items():
        assert lim[2] is not None and lim[2] > 0, nm
    assert r["surface_scored"]["ail"] and not r["surface_scored"].get("flap", False)
    assert set(r["terms"]) == set(fe.TERM_KEYS) and len(fe.TERM_KEYS) == 24


@pytest.mark.parametrize("model", ("c172x", "T38"))
def test_nonfbw_plant_equals_r0_when_untouched(sim, model):
    """Direct-surface aircraft: r1 flies the same plant as r0 (the r1 changes are f16 FBW placement + exposed state)."""
    P = fe.default_profile(model, sim)
    c = {"start": {"alt_ft": P.h0_ft, "kcas": P.speed_kts}, "duration_s": 3.0, "demo_heading_deg": 20}
    a = fp4.fly_course(None, {}, {}, None, None, c, model=model, guidance=fp4.demo_guidance, sim=sim)
    b = fp1.fly_course(None, {}, {}, None, None, c, model=model, guidance=fp1.demo_guidance, sim=sim)
    for k in ("t", "pos", "att", "nz", "alpha", "surfaces", "v_kcas"):
        assert a[k] == b[k], k
    assert {k: v for k, v in b["surface_limits"].items()} == {k: v for k, v in a["surface_limits"].items() if v[2] is not None}


def test_f16_fbw_passthrough_vs_r0(sim):
    """f16: r0 stacks lag+rate in front of the FBW demand (commanded != applied), r1 applies the demand unchanged."""
    P = fe.default_profile("f16", sim)
    c = {"start": {"alt_ft": P.h0_ft, "kcas": P.speed_kts}, "duration_s": 2.0}

    def step(s, *a):
        return {"aileron": 0.5 if s["t"] > 0.5 else 0.0, "elevator": 0.0, "rudder": 0.0, "throttle": 0.3}
    r0 = fp4.fly_course(None, {}, {}, None, None, c, model="f16", guidance=step, sim=sim)
    r1 = fp1.fly_course(None, {}, {}, None, None, c, model="f16", guidance=step, sim=sim)
    a0, a1 = r0["surfaces_120hz"], r1["surfaces_120hz"]
    i = int(0.51 * 120) + 3
    assert abs(a1["ail_act_norm"][i] - a1["ail_cmd_norm"][i]) < 1e-12              # r1: no FD actuator state
    assert abs(a0["ail_act_norm"][i] - a0["ail_cmd_norm"][i]) > 1e-3               # r0: lagged demand
    assert r1["surface_fbw"]["ail"] and not r0.get("surface_fbw")


@pytest.mark.parametrize("model", MODELS)
def test_rudder_sign_plus_is_nose_left(sim, model):
    """Genome ask 3: +rudder -> negative body yaw rate r (nose left), all four aircraft."""
    for amp in (0.1, -0.1):
        a, r = sc1._pulse(model, "rudder", amp, sim, dur=2.2)
        t, y = a[:, 0], a[:, 3]
        i0, i1 = np.searchsorted(t, 1.0), np.searchsorted(t, 1.2)
        assert (y[i1] - y[i0]) * amp < 0, (model, amp)


@pytest.mark.parametrize("model", MODELS)
def test_ail_elev_signs(sim, model):
    a, _ = sc1._pulse(model, "aileron", 0.1, sim, dur=2.2)
    t = a[:, 0]; i0, i1 = np.searchsorted(t, 1.0), np.searchsorted(t, 1.2)
    assert a[i1, 1] - a[i0, 1] > 0                                                  # +aileron -> +p
    b, _ = sc1._pulse(model, "elevator", 0.1, sim, dur=2.2)
    assert b[i1, 2] - b[i0, 2] < 0                                                  # +elevator -> nose down


def test_scaling_file():
    d = json.load(open(os.path.join(HERE, "v2_results", "p4r1_gain_scaling.json")))
    assert d["schema"] == "fd-p4r1-scaling/1" and d["ref"] == "c172x"
    ref = d["aircraft"]["c172x"]
    assert ref["gain_scale"]["pitch"] == 1.0 and ref["gain_scale"]["roll"] == 1.0 and ref["outer_gain_scale"] == 1.0
    for m in MODELS:
        a = d["aircraft"][m]
        for ax in ("pitch", "roll", "yaw"):
            assert a[f"auth_{ax}_dps2_per_norm"] > 0 and a[f"sign_{ax}"] in (1.0, -1.0)
        assert 0.3 <= a["gain_scale"]["yaw"] <= 1.0
        assert abs(a["outer_gain_scale"] - (a["v_tas_ms"] / ref["v_tas_ms"]) ** 0.5) < 1e-9
    assert "fbw_rate_demand" in d["aircraft"]["f16"] and "fbw_rate_demand" not in d["aircraft"]["T38"]
    # gain_scale is the authority ratio (not the g-limited kinematic rate Genome used before)
    t = d["aircraft"]["T38"]
    assert abs(t["gain_scale"]["pitch"] - ref["auth_pitch_dps2_per_norm"] / t["auth_pitch_dps2_per_norm"]) < 1e-9


def test_f16_post_fcs_patch_is_off_and_well_formed(tmp_path):
    import xml.dom.minidom as md
    assert fp1.F16_POST_FCS is False
    assert fp1.ensure_root_p4("f16", ROOT) == ROOT and fp1.ensure_root_p4("T38", ROOT) == ROOT
    src = open(os.path.join(fe.root_v2_for(ROOT), "aircraft", "f16", "f16.xml")).read()
    new = fp1.patch_f16_xml(src)
    md.parseString(new)
    assert new.count("lag_filter name") == src.count("lag_filter name") + 3 and "fcs/aileron-act-norm" in new
    assert "<time>0.8333</time>" in new and "<time>0.5000</time>" in new and "<time>0.5375</time>" in new
