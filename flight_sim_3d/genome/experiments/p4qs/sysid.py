"""Small-signal step ID per aircraft on the FD plant (full_a1_b2a_cs active): identity guidance for 15 s (settle), then
commands frozen at their 13-15 s mean; elevator step -de for 1.0 s at t=16; aileron step +da for 1.5 s at t=25.
Reports p/da (deg/s per norm), roll time constant (63 %), q/de, dnz/de, and pitch rise time. Read-only on FD/SB."""
import json, math, os, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import p4_guidance as PG, phase4_rings as R


def run(model, da=0.05, de=0.05):
    RC, fp4 = PG._sb(), PG._fd()
    P = R.load_preset()
    course = RC.make_course(model, "easy", RC.course_seed(1, 0, 0, model))
    gains = PG.decode_physical(R.identity_u(P["genes"]), P["genes"], model)
    gid = PG.make_guidance(gains, model, course)
    hist, frz = [], {}
    def guid(s, *a):
        t = s["t"]
        if t < 15.0:
            o = gid(s)
            if t >= 13.0:
                hist.append(o)
            return o
        if not frz:
            frz.update({k: float(np.mean([h[k] for h in hist])) for k in hist[0]})
        o = dict(frz)
        if 25.0 <= t < 26.5:
            o["aileron"] += da
        if 16.0 <= t < 17.0:
            o["elevator"] -= de
        return o
    fc = {"start": {"alt_ft": course["start"]["alt_ft"], "kcas": course["start"]["kcas"]}, "duration_s": 28.0}
    r = fp4.fly_course(None, gains, None, None, None, fc, PG.FIDELITY, model=model, guidance=guid, cs_mode="active",
                       record_hz=120.0)
    t = np.asarray(r["t"]); att = np.asarray(r["att"]); nz = np.asarray(r["nz"])
    phi = np.degrees(att[:, 0]); th = np.degrees(att[:, 1])
    p = np.gradient(phi, t); q = np.gradient(th, t)
    m = (t >= 25.0) & (t < 26.5)
    p0 = np.mean(p[(t > 24.0) & (t < 25.0)])
    dp = p[m] - p0
    pk = dp[np.argmax(np.abs(dp))]
    i63 = np.argmax(np.abs(dp) >= 0.63 * abs(pk))
    m2 = (t >= 16.0) & (t < 17.0)
    q0 = np.mean(q[(t > 15.2) & (t < 16.0)]); n0 = np.mean(nz[(t > 15.2) & (t < 16.0)])
    dq = q[m2] - q0; dn = nz[m2] - n0
    qk = dq[np.argmax(np.abs(dq))]; nk = dn[np.argmax(np.abs(dn))]
    j63 = np.argmax(np.abs(dq) >= 0.63 * abs(qk))
    # oscillation check on q after the elevator step (zero crossings of dq - mean)
    return {"model": model, "status": r["status"], "vt_ms": float(np.mean(np.asarray(r["v_ms"])[m])),
            "trim_cmd": frz, "p_per_ail_dps": float(pk / da), "tau_roll_s": float(t[m][i63] - 25.0),
            "q_per_elev_dps": float(qk / de), "dnz_per_elev_g": float(nk / de), "t63_pitch_s": float(t[m2][j63] - 16.0)}


if __name__ == "__main__":
    out = run(sys.argv[1])
    print(json.dumps(out))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), f"sysid_{sys.argv[1]}.json"), "w"), indent=1)
