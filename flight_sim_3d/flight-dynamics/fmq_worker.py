"""Flight-model quality check for one bundled JSBSim aircraft. Prints one JSON line.

usage: python fmq_worker.py MODEL KCAS[,KCAS...] ALT_FT

Steps: (a) load (auto-creating FlightGear-only properties a model references, if any),
(b) trim: JSBSim built-in trim (simulation/do_simple_trim=1 = tFull; 0 = tLongitudinal tried as
    fallback) at the cases.txt speeds, then a small ladder of alternative speeds/altitudes; if all fail, the engine-off Newton
    glide trim from flight-sim-plan/flytest.py (external, not in this repo) (re-implemented here),
(c1) open-loop elevator doublet (+A 1-2 s, -A 2-3 s, hold to 25 s),
(c2) closed-loop 3 deg pitch-attitude step with a PID auto-scaled by the doublet's
     measured pitch-rate gain (no per-aircraft tuning),
(d) determinism: doublet repeated in a fresh FDM; histories compared bit-for-bit,
plus real-time factor (RTF) measurements.
"""
import json
import math
import os
import re
import sys
import tempfile
import time

import numpy as np
import jsbsim

DT = 1 / 120
jsbsim.FGJSBBase().debug_lvl = 0
OUT = os.path.join(tempfile.gettempdir(), "fmq_jsbsim")
os.makedirs(OUT, exist_ok=True)

model, ALT = sys.argv[1], float(sys.argv[3])
SPEEDS = [float(v) for v in sys.argv[2].split(",")]
KCAS = SPEEDS[0]
res = {"model": model, "alt_nominal_ft": ALT, "kcas_nominal": KCAS, "speeds_tried": SPEEDS}
EXTRA = {}           # properties we had to create (FlightGear-only inputs)


GEAR_UP = os.environ.get("FMQ_GEAR_UP", "1") == "1"


def new_fdm():
    t0 = time.perf_counter()
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.set_output_path(OUT)
    fdm.load_model(model)
    fdm.disable_output()
    fdm.set_dt(DT)
    for p, v in EXTRA.items():
        fdm[p] = v
    if GEAR_UP:  # cruise configuration; gear/gear-cmd-norm defaults to 1 (down) in every JSBSim model
        try:
            fdm["gear/gear-cmd-norm"] = 0.0
        except Exception:  # noqa: BLE001
            pass
    res.setdefault("load_s", round(time.perf_counter() - t0, 3))
    return fdm


def with_missing_props(fn, *a):
    for _ in range(12):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            m = re.search(r"property (\S+) does not exist", str(e))
            if m and m.group(1) not in EXTRA:
                EXTRA[m.group(1)] = 0.0
                continue
            raise
    raise RuntimeError("too many missing properties")


def engines(fdm):
    return fdm.get_propulsion().get_num_engines()


def builtin_trim(kcas, alt, mode=1):
    fdm = new_fdm()
    fdm["ic/h-sl-ft"] = alt; fdm["ic/vc-kts"] = kcas; fdm["ic/gamma-deg"] = 0; fdm["ic/psi-true-deg"] = 0
    fdm.run_ic()
    fdm["propulsion/set-running"] = -1
    for i in range(max(1, engines(fdm))):
        fdm[f"fcs/mixture-cmd-norm[{i}]"] = 1.0
    fdm["simulation/do_simple_trim"] = mode
    return fdm


def retract_gear(fdm, alt, kcas, max_s=60.0):
    """run_ic advances FCS actuators by one dt per call (gear transit is not instantaneous outside the
    built-in trimmer), so retract the gear by stepping before a Newton trim built on repeated run_ic."""
    fdm["ic/h-sl-ft"] = alt; fdm["ic/vc-kts"] = kcas; fdm["ic/gamma-deg"] = 0
    fdm.run_ic()
    for _ in range(int(max_s / DT)):
        if fdm["gear/gear-pos-norm"] <= 0.0:
            break
        fdm.run()


def glide_trim(kcas, alt):
    fdm = new_fdm()
    if GEAR_UP:
        retract_gear(fdm, alt, kcas)
    def r(x):
        a, g, e = x
        fdm["ic/h-sl-ft"] = alt; fdm["ic/vc-kts"] = kcas; fdm["ic/alpha-deg"] = a; fdm["ic/gamma-deg"] = g
        fdm["ic/beta-deg"] = 0; fdm["ic/phi-deg"] = 0; fdm["ic/psi-true-deg"] = 0
        fdm["fcs/elevator-cmd-norm"] = e
        fdm.run_ic()
        return np.array([fdm["accelerations/udot-ft_sec2"], fdm["accelerations/wdot-ft_sec2"],
                         math.degrees(fdm["accelerations/qdot-rad_sec2"])])
    x = np.array([3.0, -3.0, 0.0])
    for _ in range(40):
        f0 = r(x)
        if np.linalg.norm(f0) < 1e-3:
            break
        J = np.zeros((3, 3)); h = [1e-3, 1e-3, 1e-4]
        for j in range(3):
            dx = x.copy(); dx[j] += h[j]; J[:, j] = (r(dx) - f0) / h[j]
        x = x - np.linalg.lstsq(J, f0, rcond=None)[0]
        x[2] = np.clip(x[2], -1, 1)
    f0 = r(x)
    if not (np.all(np.isfinite(f0)) and np.linalg.norm(f0) < 1e-2):
        raise RuntimeError("glide trim did not converge")
    return fdm


def trim_point():
    """Returns (factory, kcas, alt, method) for a trim that works, trying a small ladder."""
    ladder = [(k, ALT, 1) for k in SPEEDS] + [(KCAS, ALT, 0)] + \
             [(KCAS * 0.85, ALT, 1), (KCAS * 1.2, ALT, 1), (KCAS, max(ALT * 0.5, 2000.0), 1), (KCAS * 0.7, ALT, 1)]
    errs = []
    for kcas, alt, mode in ladder:
        try:
            with_missing_props(builtin_trim, kcas, alt, mode)
            name = "builtin-full" if mode == 1 else "builtin-longitudinal"
            return (lambda k=kcas, a=alt, m=mode: with_missing_props(builtin_trim, k, a, m)), kcas, alt, name, errs
        except Exception as e:  # noqa: BLE001
            errs.append(f"{kcas:.0f}kt/{alt:.0f}ft/mode{mode}:{type(e).__name__}:{str(e)[:50]}")
    try:
        with_missing_props(glide_trim, KCAS, ALT)
        return (lambda: with_missing_props(glide_trim, KCAS, ALT)), KCAS, ALT, "newton-glide(engines off)", errs
    except Exception as e:  # noqa: BLE001
        errs.insert(0, f"glide:{type(e).__name__}:{str(e)[:60]}")
    return None, None, None, "failed", errs


STATE = ("position/h-sl-ft", "position/h-agl-ft", "attitude/theta-deg", "attitude/phi-deg", "velocities/vc-kts",
         "aero/alpha-deg", "velocities/q-rad_sec", "velocities/p-rad_sec", "accelerations/Nz", "velocities/r-rad_sec")


def fly(fdm, n, elev_fn, record=True, leveler=True):
    """Fly n steps; elevator from elev_fn. Optional fixed-gain wing leveler (same gains as the repo's
    sim.py: ail = -0.05*phi - 0.02*p, deg) so the longitudinal tests are not polluted by roll-off."""
    elev0 = fdm["fcs/elevator-cmd-norm"]
    ail0 = fdm["fcs/aileron-cmd-norm"]
    hist = np.zeros((n, len(STATE)))
    t0 = time.perf_counter()
    k_end = n
    for k in range(n):
        fdm["fcs/elevator-cmd-norm"] = elev_fn(k * DT, elev0, fdm)
        if leveler:
            fdm["fcs/aileron-cmd-norm"] = min(max(ail0 - 0.05 * fdm["attitude/phi-deg"] - 0.02 * math.degrees(fdm["velocities/p-rad_sec"]), -0.5), 0.5)
        fdm.run()
        row = [fdm[p] for p in STATE]
        hist[k] = row
        if not all(map(math.isfinite, row)):
            k_end = k + 1
            break
    wall = time.perf_counter() - t0
    return hist[:k_end], wall


def envelope(h, h0_agl):
    """'ok' or the envelope criterion violated FIRST (with its time)."""
    if not np.all(np.isfinite(h)):
        return "nan"
    agl = h[:, 1]; th = h[:, 2]; ph = h[:, 3]; al = h[:, 5]; nz = h[:, 8]
    checks = {
        "ground": agl < min(200.0, 0.3 * h0_agl),
        "roll-departure": np.abs(ph) > 45,
        "alpha-excursion": (al < -10) | (al > 25),
        "nz-excursion": (nz < -2) | (nz > 5),
        "pitch-excursion": np.abs(th - th[0]) > 30,
    }
    first = {k: int(np.argmax(v)) for k, v in checks.items() if v.any()}
    if not first:
        return "ok"
    k = min(first, key=first.get)
    return f"{k}@{(first[k] + 1) * DT:.1f}s"


try:
    t0 = time.perf_counter()
    fdm = with_missing_props(new_fdm)
    res["load"] = "ok"
    res["n_engines"] = engines(fdm)
    del fdm
except Exception as e:  # noqa: BLE001
    res.update(load="failed", load_err=f"{type(e).__name__}: {str(e)[:120]}")
    print(json.dumps(res)); sys.exit()

factory, kcas, alt, method, errs = trim_point()
res["created_props"] = sorted(EXTRA)
res["gear_cmd"] = 0.0 if GEAR_UP else 1.0
res["trim_method"] = method
res["trim_errors"] = errs[:6]
if factory is None:
    print(json.dumps(res)); sys.exit()

fdm = factory()
res.update(trim_kcas=round(kcas, 1), trim_alt_ft=alt,
           alpha_deg=round(fdm["aero/alpha-deg"], 2), theta_deg=round(fdm["attitude/theta-deg"], 2),
           gamma_deg=round(fdm["flight-path/gamma-deg"], 2),
           throttle=round(fdm["fcs/throttle-cmd-norm"], 3), elevator_cmd=round(fdm["fcs/elevator-cmd-norm"], 3),
           pitch_trim=round(fdm["fcs/pitch-trim-cmd-norm"], 3), mach=round(fdm["velocities/mach"], 3),
           weight_lb=round(fdm["inertia/weight-lbs"], 0),
           throttle_pos=round(fdm["fcs/throttle-pos-norm"], 3), gear_pos=round(fdm["gear/gear-pos-norm"], 2),
           resid_udot=round(fdm["accelerations/udot-ft_sec2"], 4), resid_wdot=round(fdm["accelerations/wdot-ft_sec2"], 4),
           resid_qdot_dps2=round(math.degrees(fdm["accelerations/qdot-rad_sec2"]), 4))
h0_agl = fdm["position/h-agl-ft"]

# pure stepping speed (no property traffic), 10 s
t1 = time.perf_counter()
for _ in range(1200):
    fdm.run()
res["rtf_pure"] = round(10.0 / (time.perf_counter() - t1), 1)
# trimmed-hold drift over those 10 s
res["hold10_dtheta_deg"] = round(abs(fdm["attitude/theta-deg"] - res["theta_deg"]), 2)
res["hold10_dkcas"] = round(fdm["velocities/vc-kts"] - kcas, 2)
res["hold10_phi_deg"] = round(fdm["attitude/phi-deg"], 2)      # roll-off with all controls fixed
del fdm

# (c1) open-loop elevator doublet
A = 0.1
def doublet(t, e0, f):
    return e0 + (A if 1.0 <= t < 2.0 else (-A if 2.0 <= t < 3.0 else 0.0))
N = int(25 / DT)
runs = []
for rep in range(2):
    fdm = factory()
    h, wall = fly(fdm, N, doublet)
    runs.append(h)
    if rep == 0:
        res["rtf_loop"] = round(len(h) * DT / wall, 1)
    del fdm
h = runs[0]
res["doublet_status"] = envelope(h, h0_agl) if len(h) == N else "nan"
if np.all(np.isfinite(h)):
    q = np.degrees(h[:, 6]); t = (np.arange(len(h)) + 1) * DT
    res["doublet_peak_q_dps"] = round(float(np.abs(q[t < 3.5]).max()), 2)
    res["doublet_peak_nz"] = round(float(h[:, 8].max()), 2)
    res["doublet_min_nz"] = round(float(h[:, 8].min()), 2)
    res["doublet_max_dtheta"] = round(float(np.abs(h[:, 2] - h[0, 2]).max()), 2)
    res["doublet_max_phi"] = round(float(np.abs(h[:, 3]).max()), 2)
    res["doublet_dh_ft"] = round(float(h[-1, 0] - h[0, 0]), 1)
    early = np.abs(q[(t >= 3) & (t < 5)]).max()
    late = np.abs(q[(t >= 8) & (t < 12)]).max()
    res["q_decay_ratio"] = round(float(late / max(early, 1e-9)), 3)
    # pitch-rate gain from the first second of the doublet (deg/s per unit elevator), signed
    k1 = (t >= 1.0) & (t < 2.0)
    i = np.argmax(np.abs(q[k1] - q[0]))
    gq = float((q[k1][i] - q[0]) / A)
    res["pitch_rate_gain_dps_per_unit"] = round(gq, 2)
else:
    gq = 0.0
# (d) determinism
same = len(runs[0]) == len(runs[1]) and np.array_equal(runs[0], runs[1], equal_nan=True)
res["deterministic"] = bool(same)
if not same and len(runs[0]) == len(runs[1]):
    res["determinism_maxdiff"] = float(np.nanmax(np.abs(runs[0] - runs[1])))

# (c2) closed-loop pitch-attitude step, PID auto-scaled by gq
if abs(gq) > 0.05:
    fdm = factory()
    th0 = fdm["attitude/theta-deg"]
    st = {"i": 0.0}
    def pid(t, e0, f, st=st):
        cmd = th0 + (3.0 if t >= 2.0 else 0.0)
        e = cmd - f["attitude/theta-deg"]
        st["i"] = min(max(st["i"] + e * DT, -100.0), 100.0)
        u = 1.5 * e + 0.3 * st["i"] - 0.8 * math.degrees(f["velocities/q-rad_sec"])    # desired pitch-rate-ish (deg/s)
        return min(max(e0 + u / gq, -1.0), 1.0)
    h, _ = fly(fdm, int(22 / DT), pid)
    del fdm
    res["pstep_status"] = envelope(h, h0_agl) if len(h) == int(22 / DT) else "nan"
    if np.all(np.isfinite(h)) and len(h) == int(22 / DT):
        t = (np.arange(len(h)) + 1) * DT
        dth = h[:, 2] - th0
        post = t >= 2.0
        res["pstep_overshoot_deg"] = round(float(dth[post].max() - 3.0), 2)
        reach = np.where(post & (dth >= 2.7))[0]
        res["pstep_t90_s"] = round(float(t[reach[0]] - 2.0), 2) if reach.size else None
        res["pstep_final_err_deg"] = round(float(np.mean(3.0 - dth[t >= 19.0])), 3)
        res["pstep_peak_nz"] = round(float(h[:, 8].max()), 2)
        ok = res["pstep_status"] == "ok" and abs(res["pstep_final_err_deg"]) < 1.0 and res["pstep_overshoot_deg"] < 3.0
        res["pstep_pass"] = bool(ok)
    else:
        res["pstep_pass"] = False
else:
    res["pstep_status"] = "skipped (no measurable pitch-rate gain)"
    res["pstep_pass"] = False

print(json.dumps(res))
