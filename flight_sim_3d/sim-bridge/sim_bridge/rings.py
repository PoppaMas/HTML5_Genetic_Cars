"""Phase 4 rings: viewer/trajectory helpers on top of the canonical stdlib course module `sim_bridge.ring_course`
(generator, ring_at, score_course, frames). Kept separate so ER/FD/Genome import only ring_course."""
from __future__ import annotations

from .ring_course import *  # noqa: F401,F403
from .ring_course import WINDOW, ned_to_traj_enu, public, ring_at


def ring_state_at(gates, M, t, window=WINDOW):
    """{k: state} at time t for k < M; state in passed/missed/rim/target/active/hidden. Mirrors viewer/js/rings.js."""
    done = [g for g in gates if g["t"] <= t]
    st = {}
    for g in done:
        st[g["k"]] = {"pass": "passed", "rim": "rim"}.get(g["result"], "missed")
    nxt = len(done)
    if not any(g["result"] == "rim" for g in done):
        for k in range(nxt, min(M, nxt + window)):
            st[k] = "target" if k == nxt else "active"
    for k in range(M):
        st.setdefault(k, "hidden")
    return st


def course_block(course, traj_origin_alt_m, ground_alt_msl_m=0.0):
    """Trajectory `course` block: the canonical NED course (all M rings) + an ENU copy of each ring for the viewer."""
    c = public(course)
    rings = [ring_at(course, k) for k in range(course["M"])]
    for r in rings:
        r["centre_enu_m"] = ned_to_traj_enu(r["centre_m"], traj_origin_alt_m, ground_alt_msl_m)
        r["normal_enu"] = [r["normal"][1], r["normal"][0], -r["normal"][2]]
    c["rings"] = rings
    pv = c.get("provenance") or {}
    for key in ("course_seed", "run_seed", "gen", "k"):   # spec v0.6 section 5: every trajectory carries these
        c[key] = pv.get(key, c["seed"] if key == "course_seed" else None)
    c["enu"] = {"traj_origin_alt_m": traj_origin_alt_m, "ground_alt_msl_m": ground_alt_msl_m,
                "doc": "centre_enu_m / normal_enu = trajectory frame (ENU, origin at the start, z rel. frame.origin_alt_m)"}
    return c


CTRL_SURF_CHANNELS = ("elev_deg", "ail_deg", "rud_deg", "ail_L_deg", "ail_R_deg", "flap_deg", "sbrk_deg",
                      "elev_cmd_deg", "ail_cmd_deg", "rud_cmd_deg")


def ctrl_surface_channels(ctrl_series, step=4):
    """FD `ctrl_surfaces` per-step arrays (120 Hz, fd-ctrlsurf/1) -> trajectory channels decimated by `step`
    (4 -> 30 Hz: every 4th sample, as FD suggests). Returns {"t": [...], "<surface>_deg": [...], ...} for the keys
    present; the viewer shows them in the HUD (SURF line)."""
    t = list(ctrl_series["t_s"])[step - 1::step]
    out = {"t": t}
    for k in CTRL_SURF_CHANNELS:
        if k in ctrl_series:
            out[k] = list(ctrl_series[k])[step - 1::step]
    return out
