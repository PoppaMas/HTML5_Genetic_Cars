import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test-only local config for the team working layout (<team>/genome), which has no repo-relative
# ../../flight_sim: point FLIGHT_SIM_DIR at the sandbox clone, but only if the variable is unset, the repo default is
# absent and the clone exists. Library code (flightsim_path.py) has no absolute paths; in the repo layout this is a no-op.
_REPO_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flight_sim")
_TEAM_CLONE = os.environ.get("GENOME_TEAM_FLIGHT_SIM_DIR", "")  # team layout only; unset in the repo
if ("FLIGHT_SIM_DIR" not in os.environ and not os.path.isfile(os.path.join(_REPO_DEFAULT, "sim.py"))
        and os.path.isfile(os.path.join(_TEAM_CLONE, "sim.py"))):
    os.environ["FLIGHT_SIM_DIR"] = _TEAM_CLONE
