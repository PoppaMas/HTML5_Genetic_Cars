"""Jet profile options (gear up, all-engine throttle, MIL clamp) against Flight Dynamics' verified trim points."""
import numpy as np

from evolution import genome, sim


def _p(**kw):
    return sim.Profile(gear_up=True, throttle_all_engines=True, **kw)


def test_verified_trim_points():
    t38 = _p(aircraft="T38", h0_ft=10000, speed_kts=300, throttle_max=0.5)
    b737 = _p(aircraft="737", h0_ft=10000, speed_kts=250)
    for P, thr, alpha in ((t38, 0.354, 4.61), (b737, 0.586, 3.28)):
        pf = sim.preflight(P.to_dict(), sim.make_scenarios(1, 1, P)[0].to_dict())
        assert pf["ok"] and abs(pf["throttle_trim"] - thr) < 1e-3 and abs(pf["alpha_deg"] - alpha) < 0.01
        assert pf["gear_cmd"] == 0.0 and pf["n_engines"] == 2
    low = _p(aircraft="737", h0_ft=10000, speed_kts=185)
    assert not sim.preflight(low.to_dict(), sim.make_scenarios(1, 1, low)[0].to_dict())["ok"]


def test_t38_throttle_clamped_to_mil_and_both_engines_commanded():
    P = _p(aircraft="T38", h0_ft=10000, speed_kts=300, throttle_max=0.5, duration_s=20,
           steps_rel_ft=[(0, 0), (2, 600)])
    r = sim.simulate(genome.decode([0.7] * genome.N_GENES), sim.make_scenarios(1, 1, P)[0], P, record=True)
    tr = r["trajectory"]
    thr = np.array([row[tr["channels"].index("throttle")] for row in tr["data"]])
    assert thr.max() <= 0.5 + 1e-12 and thr.max() > 0.354  # climbing needs more than trim; never afterburner
