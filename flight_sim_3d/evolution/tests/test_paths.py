"""Repo-relative path defaults (ported from the push snapshot): they must work in the team layout
(/workspace/flight-sim-team/evolution) and in the repo layout (<repo>/flight_sim_3d/evolution)."""
import os

from evolution import batch, fidelity, sim

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM = os.path.dirname(PKG)


def test_abs_root_relative_to_team_root():
    assert sim.TEAM_ROOT == TEAM
    assert sim.abs_root(None) is None
    assert sim.abs_root("/x/y") == "/x/y"
    assert sim.abs_root("flight-dynamics/jsbsim_root") == os.path.join(TEAM, "flight-dynamics", "jsbsim_root")


def test_relative_aircraft_root_resolves_to_the_same_profile():
    rel = sim.Profile.from_dict({"aircraft": "c172x", "aircraft_root": "flight-dynamics/jsbsim_root"})
    ab = sim.Profile.from_dict({"aircraft": "c172x", "aircraft_root": os.path.join(TEAM, "flight-dynamics", "jsbsim_root")})
    assert rel.to_dict() == ab.to_dict() and os.path.isabs(rel.aircraft_root)
    assert sim.model_dir("c172x", "flight-dynamics/jsbsim_root") == sim.model_dir("c172x", ab.aircraft_root)


def test_fd_dir_default_is_the_sibling_folder():
    if not os.environ.get("EVOLUTION_FD_DIR"):
        assert fidelity.FD_DIR == os.path.join(TEAM, "flight-dynamics")


def test_default_source_repo_both_layouts(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLUTION_SOURCE_REPO", raising=False)
    repo = tmp_path / "repo"
    (repo / "flight_sim_3d" / "evolution").mkdir(parents=True)
    monkeypatch.setattr(batch, "PKG_DIR", str(repo / "flight_sim_3d" / "evolution"))
    assert batch._default_source_repo() == batch._LEGACY_SOURCE_REPO      # grandparent is not the repo
    (repo / "flight_sim").mkdir()
    assert batch._default_source_repo() == str(repo)                       # push layout: repo-relative
    monkeypatch.setenv("EVOLUTION_SOURCE_REPO", "/some/clone")
    assert batch._default_source_repo() == "/some/clone"                   # env wins


def test_team_rel_is_the_inverse_of_abs_root():
    from evolution import runinfo
    ab = os.path.join(TEAM, "flight-dynamics", "jsbsim_root")
    assert runinfo.team_rel(ab) == os.path.join("flight-dynamics", "jsbsim_root")
    assert sim.abs_root(runinfo.team_rel(ab)) == ab
    assert runinfo.team_rel(None) is None and runinfo.team_rel("x/y") == "x/y"
