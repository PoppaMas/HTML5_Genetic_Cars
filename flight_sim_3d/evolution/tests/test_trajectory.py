import copy
import json
import math
import os

import numpy as np
import pytest

from evolution import genome, sim, trajectory, validate_traj


def _docs(run_dir):
    tdir = os.path.join(run_dir, "trajectories")
    return {f: json.load(open(os.path.join(tdir, f))) for f in sorted(os.listdir(tdir)) if f.startswith("traj_")}


def test_validator_accepts_run_and_index(small_run):
    assert validate_traj.main([small_run["run_dir"], "--min-gens-per-aircraft", "3"]) == 0


def test_index_shape_and_generations(small_run):
    idx = json.load(open(os.path.join(small_run["run_dir"], "trajectories", "index.json")))
    assert isinstance(idx, dict) and isinstance(idx["entries"], list)
    G = small_run["cfg"]["ga"]["generations"]
    for ac in ("c172x", "f16"):
        gens = sorted(e["generation"] for e in idx["entries"] if e["aircraft"] == ac)
        assert gens == [0, (G - 1) // 2, G - 1]
    for e in idx["entries"]:
        assert set(e) == {"generation", "fitness", "aircraft", "file"}
        assert e["file"] == f"traj_{e['aircraft']}_{small_run['cfg']['run_id']}_g{e['generation']}.json"


def test_doc_fields_units_and_frame(small_run):
    for name, d in _docs(small_run["run_dir"]).items():
        assert d["schema"] == "ga-flightsim-traj/2"
        assert d["units"] == {"phi": "rad", "theta": "rad", "psi": "rad", "controls": "norm -1..1, throttle 0..1"}
        assert d["sample_hz"] == 30 and abs(d["dt_s"] - 1 / 30) < 1e-12
        assert d["channels"][:19] == trajectory.REQUIRED_CHANNELS
        assert set(d["genome"]) == set(genome.GENE_NAMES)
        c = {n: np.array([r[i] for r in d["data"]]) for i, n in enumerate(d["channels"])}
        # radians, not degrees; heading north
        assert np.max(np.abs(c["theta"])) < math.radians(31) and np.max(np.abs(c["phi"])) < math.radians(46)
        # JSBSim psi is in [0, 2pi); no heading hold, so allow some drift from north
        assert np.max(np.abs(np.angle(np.exp(1j * c["psi"])))) < 0.5
        # flying (roughly) north: y grows much more than x
        assert c["y"][-1] > 1000 and abs(c["x"][-1]) < c["y"][-1]
        q0 = [c[k][0] for k in ("qw", "qx", "qy", "qz")]
        R = validate_traj.quat_to_matrix(np.array([q0]))[0]
        fwd = R @ np.array([1.0, 0, 0])   # body x (nose) in ENU
        assert fwd[1] > 0.98                 # nose points north
        down = R @ np.array([0, 0, 1.0])   # body z (down) in ENU
        assert down[2] < -0.98
        assert d["target"]["alt_m"] == pytest.approx(d["frame"]["origin_alt_m"], abs=0.01)
        assert any(e["type"] == "target_change" for e in d["events"])


def test_quaternion_matches_euler_and_body_velocity(small_run):
    """Quat is the source of truth: check it against phi/theta/psi AND against JSBSim's body velocity."""
    for name, d in _docs(small_run["run_dir"]).items():
        c = {n: np.array([r[i] for r in d["data"]]) for i, n in enumerate(d["channels"])}
        q = np.stack([c["qw"], c["qx"], c["qy"], c["qz"]], 1)
        R = validate_traj.quat_to_matrix(q)
        R2 = validate_traj.M_EN @ validate_traj.euler_to_matrix_ned(c["phi"], c["theta"], c["psi"])
        assert np.max(validate_traj.rot_angle(R, R2)) < 1e-4
        v = np.einsum("nij,nj->ni", R, np.stack([c["ub"], c["vb"], c["wb"]], 1))
        assert np.max(np.abs(v - np.stack([c["vx"], c["vy"], c["vz"]], 1))) < 0.05


def test_quat_function_against_euler_sweep():
    rng = np.random.default_rng(1)
    for _ in range(200):
        phi, th, psi = rng.uniform(-math.pi, math.pi), rng.uniform(-1.5, 1.5), rng.uniform(0, 2 * math.pi)
        q = np.array([sim.quat_body_to_enu(phi, th, psi)])
        R = validate_traj.quat_to_matrix(q)
        R2 = validate_traj.M_EN @ validate_traj.euler_to_matrix_ned(np.array([phi]), np.array([th]), np.array([psi]))
        assert np.max(validate_traj.rot_angle(R, R2)) < 1e-9


@pytest.mark.parametrize("mutation", ["degrees", "quat", "units", "throttle", "schema", "channel", "swap_xy"])
def test_validator_rejects_broken_docs(small_run, mutation):
    name, d = next(iter(_docs(small_run["run_dir"]).items()))
    d = copy.deepcopy(d)
    ch = {n: i for i, n in enumerate(d["channels"])}
    if mutation == "degrees":
        for r in d["data"]:
            r[ch["theta"]] = math.degrees(r[ch["theta"]]) + 10
    elif mutation == "quat":
        for r in d["data"]:  # swap x/y components: still unit, wrong attitude
            r[ch["qx"]], r[ch["qy"]] = r[ch["qy"]], r[ch["qx"]]
    elif mutation == "units":
        d["units"]["phi"] = "deg"
    elif mutation == "throttle":
        d["data"][5][ch["throttle"]] = 1.5
    elif mutation == "schema":
        d["schema"] = "ga-flightsim-traj/0"
    elif mutation == "channel":
        i = ch["qw"]
        d["channels"].pop(i)
        for r in d["data"]:
            r.pop(i)
    elif mutation == "swap_xy":
        for r in d["data"]:
            r[ch["x"]], r[ch["y"]] = r[ch["y"]], r[ch["x"]]
    assert validate_traj.validate_doc(d, name), mutation


def test_validator_rejects_bad_index(small_run, tmp_path):
    import shutil
    src = os.path.join(small_run["run_dir"], "trajectories")
    dst = tmp_path / "t"
    shutil.copytree(src, dst)
    idx = json.load(open(dst / "index.json"))
    idx["entries"][0]["extra"] = 1
    idx["entries"][1]["fitness"] += 1.0
    json.dump(idx, open(dst / "index.json", "w"))
    errs = validate_traj.validate_index(str(dst / "index.json"))
    assert any("exactly" in e for e in errs) and any("fitness" in e for e in errs)
    # top-level list form is also accepted
    json.dump(json.load(open(os.path.join(src, "index.json")))["entries"], open(dst / "index.json", "w"))
    assert validate_traj.validate_index(str(dst / "index.json"), min_gens=3) == []


def test_resim_matches_evaluation(small_run):
    for a in small_run["summary"]["aircraft"]:
        assert a["resim_mismatches"] == []


def test_modal_twist_sign_fixed_export_and_validator_optional(small_run):
    """New exports set structure.modal_twist_sign_fixed; validate_traj accepts with or without it."""
    name, d = next(iter(_docs(small_run["run_dir"]).items()))
    # build_doc path: structure is copied and flagged; source left untouched
    st_src = {"schema": "evolution-flex-state/3", "fidelity": "full", "components": []}
    tr = {
        "channels": d["channels"], "data": d["data"], "dt_s": d["dt_s"], "sim_dt_s": d.get("sim_dt_s", 1 / 120),
        "sample_hz": d["sample_hz"],
        "origin": {"lat_deg": d["frame"]["origin_lat_deg"], "lon_deg": d["frame"]["origin_lon_deg"],
                   "alt_m": d["frame"]["origin_alt_m"]},
        "trim": {"throttle_trim": 0.5}, "structure": st_src,
        "controls_timing": d.get("controls_timing", "pre_step"),
    }
    scenario = {"h0_ft": d["target"]["alt_m"] / trajectory.FT, "speed_kts": d["target"]["speed_kcas"],
                "steps": [(s["t"], s["alt_m"] / trajectory.FT) for s in d["target"]["steps"]],
                "wind_north_fps": 0.0, "wind_east_fps": 0.0, "gust_sigma_fps": 0.0, "ramp_fpm": None}
    doc = trajectory.build_doc(
        run_id=d["run_id"], aircraft=d["aircraft"], jsbsim_version=d["jsbsim_version"], git_sha=d["git_sha"],
        seed=d["seed"], generation=d["generation"], fitness=d["fitness"], gains=d["genome"],
        scenario=scenario, scenario_index=0,
        sim_result={"cost": d.get("scenario_cost", d["fitness"]), "status": d["status"],
                    "t_end": d["data"][-1][0], "trajectory": tr},
        profile={},
    )
    assert doc["structure"]["modal_twist_sign_fixed"] is True
    assert "modal_twist_sign_fixed" not in st_src
    assert validate_traj.validate_doc(doc, "with_flag") == []
    # without the flag (older exports): still valid
    doc2 = copy.deepcopy(doc)
    del doc2["structure"]["modal_twist_sign_fixed"]
    assert validate_traj.validate_doc(doc2, "no_flag") == []
    # present but not true: rejected
    doc3 = copy.deepcopy(doc)
    doc3["structure"]["modal_twist_sign_fixed"] = False
    assert any("modal_twist_sign_fixed" in e for e in validate_traj.validate_doc(doc3, "bad_flag"))
