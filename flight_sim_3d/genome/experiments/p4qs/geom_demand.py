"""Course demand per aircraft/stage: heading change between legs and the bank / nz needed to fly it (read-only on SB)."""
import math, sys, os, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "sim-bridge"))
from sim_bridge import ring_course as RC
G = 9.80665
out = {}
for ac in ("c172x", "T38", "737", "f16"):
    for st in ("easy", "medium", "hard"):
        V = RC.scales(ac)["v_tas_ms"]
        dpsi, dgam, leg = [], [], []
        for s in range(20):
            c = RC.make_course(ac, st, RC.course_seed(7, 0, s, ac))
            P = [c["start"]["pos"]] + [RC.ring_at(c, k)["centre_m"] for k in range(c["M"])]
            P = np.array(P)
            d = np.diff(P, axis=0)
            psi = np.arctan2(d[:, 1], d[:, 0]); gam = np.arctan2(-d[:, 2], np.hypot(d[:, 0], d[:, 1]))
            L = np.linalg.norm(d, axis=1)
            dpsi += list(np.abs(np.angle(np.exp(1j * np.diff(psi))))); dgam += list(np.abs(np.diff(gam))); leg += list(L[1:])
        dpsi, dgam, leg = map(np.array, (dpsi, dgam, leg))
        # turn spread over one leg (lower bound): a_lat = V * dpsi / (L/V)
        a = V * V * dpsi / leg
        bank = np.degrees(np.arctan(a / G))
        dn = V * V * dgam / leg / G
        out[f"{ac}:{st}"] = dict(V=round(V, 1), spacing=round(float(np.median(leg)), 0), radius=c["params"]["radius_m"],
                                 dpsi_med_deg=round(float(np.degrees(np.median(dpsi))), 1),
                                 bank_med=round(float(np.median(bank)), 1), bank_p90=round(float(np.percentile(bank, 90)), 1),
                                 dnz_p90=round(float(np.percentile(dn, 90)), 2))
for k, v in out.items():
    print(k, v)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "geom_demand.json"), "w"), indent=1)
