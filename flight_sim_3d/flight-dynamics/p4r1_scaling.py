"""p4r1_scaling.py -- measures the per-aircraft control power at the Phase 4 course start trim and writes
v2_results/p4r1_gain_scaling.json (schema fd-p4r1-scaling/1). Plant = full_a1_b2a_cs1 / active, baseline aircraft.

Why: Genome scales inner-loop gains by roll_rate_max / pitch_rate_g_limited (kinematic, not authority). The right normaliser is
the control power: initial angular acceleration per command norm (deg/s^2 per norm, mean of +-0.1 norm pulses, 0.15 s after
onset, FD actuator in the loop). Fixed per aircraft (baseline geometry, trim point) -> never gene dependent, never flex eta.

    python p4r1_scaling.py [model ...]        # default: all four, writes the json (merges)
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flexeval as fe                 # noqa: E402
import flexeval_p4r1 as fp            # noqa: E402

REF = "c172x"
AMP = 0.1
OUT = os.path.join(HERE, "v2_results", "p4r1_gain_scaling.json")
# course start (Sim Bridge ring_course.make_course start): alt ft, KCAS
START = {"c172x": (4000.0, 100.0), "T38": (10000.0, 300.0), "737": (10000.0, 250.0), "f16": (10000.0, 350.0)}
FBW = {"f16": {"roll_dps_per_norm": math.degrees(1.0 / 0.31821), "pitch_dps_per_norm": math.degrees(1.0 / 6.2)}}
YAW_SCALE_CAP = (0.3, 1.0)
V_EXP = 0.5


def _pulse(model, axis, amp, sim, dur=3.0, t0=1.0, tw=0.4):
    log = []

    def g(s, *a):
        t, tr = s["t"], s["trim"]
        u = {k: tr[k] for k in ("aileron", "elevator", "rudder", "throttle")}
        if t0 <= t < t0 + tw:
            u[axis] += amp
        log.append((t, s["p"], s["q"], s["r"], s["qbar_psf"], s["mach"], s["vt_fps"]))
        return u
    h, k = START[model]
    r = fp.fly_course(None, None, None, None, None, {"start": {"alt_ft": h, "kcas": k}, "duration_s": dur}, fp.P4,
                      model=model, guidance=g, sim=sim, cs_mode="active", energy=False)
    return np.array(log), r


def measure(model, sim):
    out = {"start": {"alt_ft": START[model][0], "kcas": START[model][1]}}
    for axis, idx, nm in (("elevator", 2, "pitch"), ("aileron", 1, "roll"), ("rudder", 3, "yaw")):
        vals = []
        for amp in (AMP, -AMP):
            a, r = _pulse(model, axis, amp, sim)
            t, y = a[:, 0], a[:, idx]
            i0, i1 = np.searchsorted(t, 1.0), np.searchsorted(t, 1.15)
            vals.append(abs(math.degrees((y[i1] - y[i0]) / (t[i1] - t[i0])) / AMP))
            sign = math.copysign(1.0, (y[i1] - y[i0]) * amp)
        out[f"auth_{nm}_dps2_per_norm"] = float(np.mean(vals))
        out[f"sign_{nm}"] = sign      # +norm -> + body rate? (q: + nose up; p: + right wing down; r: + nose right)
    out["qbar_trim_psf"] = float(a[0, 4]); out["mach_trim"] = float(a[0, 5]); out["vt_trim_fps"] = float(a[0, 6])
    out["v_tas_ms"] = out["vt_trim_fps"] * 0.3048
    out["trim_cmd"] = {k: float(v) for k, v in r["trim_cmd"].items()}
    return out


def build(models):
    sim = fe.load_sim()
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"schema": "fd-p4r1-scaling/1", "aircraft": {}}
    for m in models:
        doc["aircraft"][m] = measure(m, sim)
        print(m, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in doc["aircraft"][m].items() if k != "trim_cmd"}, flush=True)
    return doc


def finalize(doc):
    ref = doc["aircraft"][REF]
    for m, a in doc["aircraft"].items():
        raw = {ax: ref[f"auth_{ax}_dps2_per_norm"] / a[f"auth_{ax}_dps2_per_norm"] for ax in ("pitch", "roll", "yaw")}
        a["gain_scale"] = {"pitch": raw["pitch"], "roll": raw["roll"],
                           "yaw": min(max(raw["yaw"], YAW_SCALE_CAP[0]), YAW_SCALE_CAP[1]), "yaw_uncapped": raw["yaw"]}
        a["outer_gain_scale"] = (a["v_tas_ms"] / ref["v_tas_ms"]) ** V_EXP     # multiply k_lat, k_lat_rate, k_vert, k_vert_rate
        a["gene_caps"] = ({"kd_pitch_hi": 0.02, "kd_roll_hi": 0.05, "note": "f16 FBW: rate-demand gain 26*kd_pitch <= ~0.5 (untested cap)"}
                          if m in FBW else {"kd_pitch_hi": 0.05, "kd_roll_hi": 0.05,
                                            "note": "loop gain K = kd*gain_scale*authority = kd*authority_ref <= ~2 (pitch) / 8 (roll) 1/s; "
                                                    "evolved T38/f16 kd gave K = 73/69 1/s (limit cycle). Tested on T38 (chatter 0.34 -> 0.05)"}) 
        if m in FBW:
            a["fbw_rate_demand"] = FBW[m]
            a["gain_scale_note"] = ("FBW rate-command: keep the rate-demand mapping (roll = ref_roll_rate_max / roll_dps_per_norm, "
                                    "pitch = ref_pitch_rate / pitch_dps_per_norm); the authority numbers above are the closed-loop "
                                    "FBW response, informational only for f16")
    doc["doc"] = ("Control power at the course start trim, plant full_a1_b2a_cs1 active; gain_scale = ref(c172x)/aircraft per axis "
                  "(yaw capped %s); outer_gain_scale = (V_tas/V_tas_ref)^%.2f for the 4 guidance gains" % (YAW_SCALE_CAP, V_EXP))
    doc["ref"] = REF
    return doc


if __name__ == "__main__":
    ms = sys.argv[1:] or ["c172x", "T38", "737", "f16"]
    d = build(ms)
    if all(m in d["aircraft"] for m in ("c172x", "T38", "737", "f16")):
        d = finalize(d)
    json.dump(d, open(OUT, "w"), indent=1, sort_keys=True)
