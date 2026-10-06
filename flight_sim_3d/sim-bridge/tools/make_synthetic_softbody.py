#!/usr/bin/env python3
"""Write data/examples/synthetic_softbody_test.json: a SYNTHETIC soft-body test pattern (not simulation output).

Flight: straight, wings-level 737-class cruise (made up: 250 KCAS-ish, gentle 0.1 Hz altitude wobble), 30 s at 30 Hz.
Structure: sinusoidal bending / twist written in the agreed v2 schema so the viewer's deformation path can be
exercised before Flight Dynamics' modal output exists. Node axes follow the viewer's procedural 737 geometry.
Twist sign: right-hand rotation about the node0->nodeN direction (wingR + = leading edge up; wingL + = LE down).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_bridge.recorder import quat_body_to_enu  # noqa: E402

FT = 0.3048
HZ, T_END = 30.0, 30.0
V = 128.6  # m/s
H0 = 3048.0

# procedural 737 preset geometry (viewer/js/aircraft.js)
L, H, W = 33.4, 4.0, 3.76
R = max(W, H) / 2
xNose = 0.45 * L
xWingLE, croot, ctip, semi, sweep, dih, zWing = xNose - L * 0.40, 7.3, 1.6, 28.9 / 2, 28.0, 6.0, H * 0.38
xTail = xNose - L + 3.6 * 1.05


def wing_nodes(side, n=6):
    out = []
    for i in range(n):
        y = semi * i / (n - 1)
        c = croot + (ctip - croot) * y / semi
        x = xWingLE - math.tan(math.radians(sweep)) * y - 0.25 * c
        out.append([round(x, 3), round(side * y, 3), round(zWing - y * math.tan(math.radians(dih)), 3)])
    return out


def ht_nodes(n=7):
    b = 12.7 / 2
    out = []
    for i in range(n):
        y = -b + 2 * b * i / (n - 1)
        x = xTail - math.tan(math.radians(32)) * abs(y) - 0.25 * 3.6 * (1 - 0.3 * abs(y) / b)
        out.append([round(x, 3), round(y, 3), round(-R * 0.3 - abs(y) * math.tan(math.radians(7)), 3)])
    return out


def vt_nodes(n=4):
    xle, h = xTail + 5.6 * 0.25, 6.0
    return [[round(xle - math.tan(math.radians(38)) * h * i / (n - 1) - 0.25 * 5.6 * (1 - 0.45 * i / (n - 1)), 3), 0.0,
             round(-R * 0.45 - h * i / (n - 1), 3)] for i in range(n)]


def fus_nodes(n=6):
    return [[round(xNose - L * i / (n - 1), 3), 0.0, 0.0] for i in range(n)]


comps = [
    {"name": "wingL", "axis_nodes_body_m": wing_nodes(-1), "dof": ["dz", "dy", "twist"]},
    {"name": "wingR", "axis_nodes_body_m": wing_nodes(1), "dof": ["dz", "dy", "twist"]},
    {"name": "htail", "axis_nodes_body_m": ht_nodes(), "dof": ["dz", "twist"]},
    {"name": "vtail", "axis_nodes_body_m": vt_nodes(), "dof": ["dy", "twist"]},
    {"name": "fuselage", "axis_nodes_body_m": fus_nodes(), "dof": ["dz", "dy"]},
]


def struct_values(t):
    v = {}
    tw = 2 * math.pi
    for c in comps:
        n = len(c["axis_nodes_body_m"])
        for i in range(n):
            s = i / (n - 1)
            if c["name"] in ("wingL", "wingR"):
                side = -1 if c["name"] == "wingL" else 1
                sym = -0.45 - 0.35 * math.sin(tw * 1.2 * t)            # 1g up-bend + first symmetric mode (m at tip)
                anti = 0.12 * math.sin(tw * 0.7 * t) * side            # small antisymmetric component
                v[f"{c['name']}.dz.{i}"] = (sym + anti) * s * s
                v[f"{c['name']}.dy.{i}"] = 0.04 * math.sin(tw * 1.9 * t) * s * s
                v[f"{c['name']}.twist.{i}"] = side * math.radians(1.5) * math.sin(tw * 2.6 * t + 0.6) * s  # LE up both sides
            elif c["name"] == "htail":
                sy = abs(2 * s - 1)
                v[f"htail.dz.{i}"] = -0.10 * math.sin(tw * 3.0 * t) * sy * sy
                v[f"htail.twist.{i}"] = math.radians(0.8) * math.sin(tw * 3.0 * t + 1.0) * (2 * s - 1)
            elif c["name"] == "vtail":
                v[f"vtail.dy.{i}"] = 0.15 * math.sin(tw * 2.0 * t) * s * s
                v[f"vtail.twist.{i}"] = math.radians(1.0) * math.sin(tw * 2.0 * t + 0.4) * s
            else:
                m = 1 - 4 * (s - 0.5) ** 2  # 1 in the middle, 0 at nose/tail -> ends move relative to the middle
                v[f"fuselage.dz.{i}"] = -0.06 * math.sin(tw * 2.2 * t) * (1 - m)
                v[f"fuselage.dy.{i}"] = 0.03 * math.sin(tw * 1.4 * t) * (1 - m) * (1 if s > 0.5 else -1)
    return v


names = sorted(struct_values(0.0))
base = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m", "phi", "theta", "psi",
        "throttle", "elevator", "aileron", "rudder", "kcas", "nz"]
rows = []
qprev = None
n = int(T_END * HZ) + 1
for k in range(n):
    t = k / HZ
    h = 5.0 * math.sin(2 * math.pi * 0.1 * t)
    vz = 5.0 * 2 * math.pi * 0.1 * math.cos(2 * math.pi * 0.1 * t)
    theta = math.radians(2.0) + math.atan2(vz, V)
    q = quat_body_to_enu(0.0, theta, 0.0)
    if qprev and sum(a * b for a, b in zip(q, qprev)) < 0:
        q = tuple(-c for c in q)
    qprev = q
    nz = 1 - 5.0 * (2 * math.pi * 0.1) ** 2 * math.sin(2 * math.pi * 0.1 * t) / 9.80665
    row = [t, 0.0, V * t, h, *q, 0.0, V, vz, H0 + h, 0.0, theta, 0.0, 0.62, -0.05, 0.0, 0.0, 250.0, nz]
    sv = struct_values(t)
    rows.append([round(x, 6) for x in row] + [round(sv[c], 6) for c in names])

doc = {
    "schema": "ga-flightsim-traj/1", "synthetic": True,
    "note": "SYNTHETIC TEST PATTERN - not simulation output. Made-up straight cruise + sinusoidal bending/twist to "
            "exercise the viewer's soft-body (structure block) path. Do not quote any number from this file.",
    "run_id": "synthetic-softbody-test", "aircraft": "737", "generation": None, "fitness": None,
    "frame": {"origin_lat_deg": 0.0, "origin_lon_deg": 0.0, "origin_alt_m": H0, "axes": "ENU metres, x=east y=north z=up",
              "attitude": "quat body->ENU [w,x,y,z]", "body_axes": "JSBSim body FRD: x forward, y right wing, z down"},
    "units": {"phi": "rad", "theta": "rad", "psi": "rad", "structure": "dz/dy metres (body FRD), twist rad"},
    "target": {"speed_kcas": 250},
    "events": [{"t": 0.0, "type": "start", "detail": "SYNTHETIC soft-body test pattern (not a simulation)"}],
    "structure": {"synthetic": True, "schema": "sim-bridge-structure/0 (proposed v2)",
                  "source": "tools/make_synthetic_softbody.py (sinusoids)",
                  "twist_sign": "right-hand about node0->nodeN direction",
                  "components": comps},
    "channels": base + names, "data": rows,
}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "examples", "synthetic_softbody_test.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(doc, f, separators=(",", ":"))
print(out, os.path.getsize(out), "bytes", len(names), "structure channels")
