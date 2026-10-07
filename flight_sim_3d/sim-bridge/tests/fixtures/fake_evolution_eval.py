"""TEST FIXTURE: a minimal stand-in for ER's ``evolution.eval`` (not ER code). ER ships the real one now; the real
module is what the real-interface tests use. This fixture stays for protocol edge cases the real module does not
exercise: a post-step recorder WITHOUT the t=0 call, and a synthetic flex_state.

    evaluate(genome, aircraft, scenario, run_cfg, recorder=None) -> {"cost", "status", "model_version", "fidelity", ...}
      * genome: physical gains dict, aircraft: name, scenario: one entry of run_cfg["scenarios"] (ER's real format:
        id "c172x:s0", steps / ramp fields; used as given, like ER), run_cfg: run.json
      * calls recorder(t, fdm) AFTER each fdm step (t = time after the step), read-only;
        with flex active recorder(t, fdm, flex_state)
      * never sets properties on behalf of the recorder

Under the hood it flies ER's current evolution.sim (rigid) through a pass-through FDM proxy, i.e. the same physics
the legacy adapter uses, but with the agreed *post-step* callback timing. Env knobs for tests:
  FAKE_EVAL_T0=1         also call recorder(0.0, fdm) once before the first step (what we ask ER to add)
  FAKE_EVAL_SYNTH_FLEX=1 pass a SYNTHETIC flex_state (sinusoids, not physics) to exercise structure-channel plumbing
"""
import math
import os

from sim_bridge import er_adapter

RECORDER_TIMING = "post"
SYNTHETIC_STRUCTURE = {"synthetic": True, "components": [
    {"name": "wingL", "axis_nodes_body_m": [[0, 0, 0], [0, -2, 0], [0, -4, 0]], "dof": ["dz", "twist"]},
    {"name": "wingR", "axis_nodes_body_m": [[0, 0, 0], [0, 2, 0], [0, 4, 0]], "dof": ["dz", "twist"]}]}


class _SynthFlex(dict):
    structure = SYNTHETIC_STRUCTURE


def _synth_flex(t):
    fs = _SynthFlex()
    for comp in ("wingL", "wingR"):
        for i in range(3):
            fs[f"{comp}.dz.{i}"] = -0.05 * (i / 2) ** 2 * math.sin(2 * math.pi * 1.5 * t)
            fs[f"{comp}.twist.{i}"] = 0.01 * (i / 2) * math.sin(2 * math.pi * 1.5 * t + 0.5)
    return fs


class _PostStepFDM:
    def __init__(self, fdm, recorder, dt, flex):
        self._fdm, self._rec, self._k, self._dt, self._flex = fdm, recorder, 0, dt, flex
        self._t0 = recorder is not None and os.environ.get("FAKE_EVAL_T0") == "1"

    def _call(self, t):
        if self._flex:
            self._rec(t, self._fdm, _synth_flex(t))
        else:
            self._rec(t, self._fdm)

    def run(self):
        if self._t0:  # once, after trim + first commands, before the first step
            self._t0 = False
            self._call(0.0)
        r = self._fdm.run()
        self._k += 1
        if self._rec is not None:
            self._call(self._k * self._dt)
        return r

    def __getitem__(self, k):
        return self._fdm[k]

    def __setitem__(self, k, v):
        self._fdm[k] = v

    def __getattr__(self, name):
        return getattr(self._fdm, name)


def evaluate(genome, aircraft, scenario, run_cfg, recorder=None):
    fid = run_cfg.get("fidelity", "rigid")
    if fid != "rigid":
        raise NotImplementedError(f"fixture supports rigid only (got {fid})")
    sim, _, _ = er_adapter._er()
    sc, prof = scenario_object(aircraft, scenario, run_cfg), _profile(aircraft, run_cfg)
    orig = sim._new_fdm
    flex = os.environ.get("FAKE_EVAL_SYNTH_FLEX") == "1"
    first = {}

    def patched(profile):
        fdm = orig(profile)
        if "x" in first:  # only the flight itself (simulate creates one FDM)
            return fdm
        first["x"] = 1
        return _PostStepFDM(fdm, recorder, sim.DT, flex)

    sim._new_fdm = patched
    try:
        r = sim.simulate(dict(genome), sc, prof, record=False)
    finally:
        sim._new_fdm = orig
    r = {k: v for k, v in r.items() if k != "trajectory"}
    r.update({"model_version": er_adapter.model_version_current(fid), "fidelity": fid})
    return r


def _profile(aircraft, run_cfg):
    sim, _, _ = er_adapter._er()
    a = next(x for x in run_cfg["aircraft"] if x["name"] == aircraft)
    return sim.Profile.from_dict(a.get("resolved_profile") or a["profile"])


def scenario_object(aircraft, scenario, run_cfg):
    """sim.Scenario from the run.json entry, used as given (ER's real evolution.eval.scenario_object does the same)."""
    sim, _, _ = er_adapter._er()
    if not isinstance(scenario, dict):
        scenario = next(s for s in run_cfg["scenarios"] if str(s["id"]) == str(scenario))
    names = set(sim.Scenario.__dataclass_fields__)
    return sim.Scenario.from_dict({k: v for k, v in scenario.items() if k in names})
