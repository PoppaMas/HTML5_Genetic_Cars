"""Read-only per-step recorder producing ga-flightsim-traj/1|/2 rows from a JSBSim FDM.

Usable as the ``recorder`` callback of ``evolution.eval.evaluate`` (``recorder(t, fdm)`` or, with flex active,
``recorder(t, fdm, flex_state)``) and of sim_bridge.er_adapter.evaluate. It only *reads* properties.
Channel set, frame and formulas are identical to ER's evolution/sim.py ``_Recorder`` (so files compare 1:1);
structure channels ``<component>.<dof>.<node>`` are appended when a flex_state provides them.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

FT = 0.3048
_WGS_A = 6378137.0
_WGS_E2 = 6.69437999014e-3
_Q_ENU_NED = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)
BASE_CHANNELS = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m",
                 "phi", "theta", "psi", "throttle", "elevator", "aileron", "rudder",
                 "target_alt_m", "kcas", "nz", "ub", "vb", "wb", "lat_deg", "lon_deg",
                 "target_cmd_alt_m", "target_rate_mps"]
# same write precision as ER's trajectory.py (mm positions, ~1e-7 quaternion)
ROUND = {"t": 4, "x": 3, "y": 3, "z": 3, "qw": 8, "qx": 8, "qy": 8, "qz": 8, "vx": 4, "vy": 4, "vz": 4,
         "alt_msl_m": 3, "phi": 7, "theta": 7, "psi": 7, "throttle": 6, "elevator": 6, "aileron": 6,
         "rudder": 6, "target_alt_m": 3, "kcas": 3, "nz": 4, "ub": 4, "vb": 4, "wb": 4, "lat_deg": 9, "lon_deg": 9,
         "target_cmd_alt_m": 3, "target_rate_mps": 5}
STRUCT_DECIMALS = 6


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def quat_body_to_enu(phi, theta, psi):
    cr, sr = math.cos(phi / 2), math.sin(phi / 2)
    cp, sp = math.cos(theta / 2), math.sin(theta / 2)
    cy, sy = math.cos(psi / 2), math.sin(psi / 2)
    q_nb = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
    return _qmul(_Q_ENU_NED, q_nb)


def flex_channels(flex_state) -> Dict[str, float]:
    """Accepts a dict {'wingL.dz.0': v, ...}, or an object with .channels() / .as_channels() returning one."""
    if flex_state is None:
        return {}
    for m in ("channels", "as_channels", "to_channels"):
        f = getattr(flex_state, m, None)
        if callable(f):
            return dict(f())
    if isinstance(flex_state, dict):
        return {k: v for k, v in flex_state.items() if isinstance(k, str) and k.count(".") == 2}
    return {}


def _is_raw_v2(flex_state) -> bool:
    """A plain dict of FD v2 diagnostics (flexbody DIAG_KEYS_V2, e.g. {'tip_w_ft_R': ..}) rather than mapped channels."""
    return isinstance(flex_state, dict) and any(k in flex_state for k in ("tip_w_ft_R", "wingR_bm", "tip_twist_R_deg"))


CTRL_IDX = (15, 16, 17, 18)  # throttle, elevator, aileron, rudder
CTRL_PROPS = ("fcs/throttle-cmd-norm", "fcs/elevator-cmd-norm", "fcs/aileron-cmd-norm", "fcs/rudder-cmd-norm")


class TrajRecorder:
    """recorder(t, fdm[, flex_state]); decimates the sim rate to ``sample_hz`` on the absolute step grid.

    timing="pre":  called before each fdm.run() with the commands for [t, t+dt) already set (sim_bridge adapter;
                   identical to ER's in-sim _Recorder), plus final(t_end, fdm).
    timing="post": called after each step (ER's agreed evolution.eval contract). The state read at t is exact; the
                   fcs commands read then are the ones flown over [t-dt, t), so a sampled row's controls are
                   filled from the *next* call (commands for [t, t+dt)), which reproduces ER's convention exactly.
                   A t=0 row exists only if evaluate also calls recorder(0.0, fdm) once before the first step.
                   finish() adds the final off-grid row (t_end) if needed.
    """

    def __init__(self, scenario, sample_hz: float = 30.0, sim_dt: float = 1.0 / 120.0, timing: str = "pre",
                 aircraft: Optional[str] = None, v2_map: bool = True, node_source=None, wings: str = "auto",
                 use_nodes: bool = True):
        if timing not in ("pre", "post"):
            raise ValueError(timing)
        if wings not in ("auto", "fd_nodes", "er"):
            raise ValueError(wings)
        self.aircraft = aircraft
        self.v2_map = v2_map      # map FD v2 telemetry (flex.*) with sim_bridge.v2_map
        self.node_source = node_source  # fallback NodeSource for older FlexState without .nodes(); prefer FlexState API
        self.use_nodes = use_nodes  # False (--no-nodes): ignore FlexState.nodes / NodeSource; tip-estimate only
        self.wings = wings        # auto: FD nodal wings when nodal data exists (they carry dx), else ER's modal wings
        self._geo = None          # v2_map geometry once known
        self._comps = None        # components we emit (None = trust ER's already-mapped channels)
        self._base_structure = None
        self._er_mapped = False   # FlexState /3: ER already ran v2_map (wingR FE + wing*_modal + empennage)
        self._flex_api = None     # "flexstate3" | "nodesource" | "estimate" | None
        self._status_seen: Dict[str, set] = {}
        self.v2_stats = {"node_frames": 0, "estimated_frames": 0, "node_errors": [], "tip_check_max_abs": 0.0,
                         "wing_modal_vs_nodal": {}, "flex_api": None}
        self.timing = timing
        self._pending = None  # (k, row) awaiting its commands (post timing)
        self._last_call = None  # (t, k, fdm, flex_state) for finish()
        self.sc = scenario  # ER Scenario (target / target_cmd / target_rate / ramp_fpm)
        self.sim_dt = sim_dt
        self.every = max(1, int(round(1.0 / (sim_dt * sample_hz))))
        self.sample_hz = sample_hz
        self.rows: List[List[float]] = []
        self.struct_names: List[str] = []
        self.origin = None
        self.origin_source = None
        self.q_prev = None
        self.n_calls = 0
        self.t_last = None

    def _init_origin(self, fdm, t=0.0):
        if t > 1e-9:  # no t=0 call (post-step contract without the initial call): JSBSim's IC = trimmed start position
            lat0, lon0, alt0 = fdm["ic/lat-geod-deg"], fdm["ic/long-gc-deg"], fdm["ic/h-sl-ft"] * FT
            self.origin_source = "ic/* properties (no t=0 recorder call)"
        else:
            lat0, lon0, alt0 = fdm["position/lat-geod-deg"], fdm["position/long-gc-deg"], fdm["position/h-sl-ft"] * FT
            self.origin_source = "state at t=0"
        lat = math.radians(lat0)
        s2 = math.sin(lat) ** 2
        self.origin = {"lat_deg": lat0, "lon_deg": lon0, "alt_m": alt0,
                       "r_north": _WGS_A * (1 - _WGS_E2) / (1 - _WGS_E2 * s2) ** 1.5 + alt0,
                       "r_east": (_WGS_A / math.sqrt(1 - _WGS_E2 * s2) + alt0) * math.cos(lat)}

    def __call__(self, t, fdm, flex_state=None):
        self.n_calls += 1
        if self.origin is None:
            self._init_origin(fdm, t)
        k = int(round(t / self.sim_dt))
        if self.timing == "post":
            if self._pending is not None:
                self._fill_ctrls(self._pending, fdm)
                self._pending = None
            self._last_call = (t, k, fdm, flex_state)
        if k % self.every:
            return
        self._add(t, fdm, flex_state)
        if self.timing == "post":
            self._pending = self.rows[-1]

    def _fill_ctrls(self, row, fdm):
        for i, p in zip(CTRL_IDX, CTRL_PROPS):
            row[i] = fdm[p]

    def final(self, t, fdm, flex_state=None):
        """Last state (evaluate calls recorder.final(t_end, fdm[, flex_state])). A sampled row still waiting for its
        commands (post timing) takes the last commands flown, as ER's recorder does for its final row."""
        if self._pending is not None:
            self._fill_ctrls(self._pending, fdm)
            self._pending = None
        if self.t_last is None or t > self.t_last + 1e-9:
            self._add(t, fdm, flex_state)

    def finish(self):
        """post timing: close the run (last row keeps the last commands, as ER's recorder does)."""
        self._pending = None
        if self.timing == "post" and self._last_call is not None:
            t, k, fdm, fx = self._last_call
            self.final(t, fdm, fx)
        self._last_call = None

    def _add(self, t, fdm, flex_state):
        if self.origin is None:
            self._init_origin(fdm)
        o = self.origin
        phi, theta, psi = fdm["attitude/phi-rad"], fdm["attitude/theta-rad"], fdm["attitude/psi-rad"]
        q = quat_body_to_enu(phi, theta, psi)
        if self.q_prev is not None and sum(a * b for a, b in zip(q, self.q_prev)) < 0:
            q = tuple(-c for c in q)
        self.q_prev = q
        lat, lon = fdm["position/lat-geod-deg"], fdm["position/long-gc-deg"]
        alt_m = fdm["position/h-sl-ft"] * FT
        sc = self.sc
        ramp = getattr(sc, "ramp_fpm", None) is not None
        row = [t, math.radians(lon - o["lon_deg"]) * o["r_east"], math.radians(lat - o["lat_deg"]) * o["r_north"], alt_m - o["alt_m"],
               *q,
               fdm["velocities/v-east-fps"] * FT, fdm["velocities/v-north-fps"] * FT, -fdm["velocities/v-down-fps"] * FT,
               alt_m, phi, theta, psi,
               fdm["fcs/throttle-cmd-norm"], fdm["fcs/elevator-cmd-norm"], fdm["fcs/aileron-cmd-norm"], fdm["fcs/rudder-cmd-norm"],
               sc.target(t)[0] * FT, fdm["velocities/vc-kts"], fdm["accelerations/Nz"],
               fdm["velocities/u-fps"] * FT, fdm["velocities/v-fps"] * FT, fdm["velocities/w-fps"] * FT, lat, lon,
               sc.target_cmd(t)[0] * FT, (sc.target_rate(t) if ramp else 0.0) * FT]
        fx = self._flex(flex_state, fdm)
        if fx:
            if not self.struct_names:
                self.struct_names = sorted(fx)
                base = getattr(flex_state, "structure", None) or (
                    flex_state.get("structure") if isinstance(flex_state, dict) else None)
                if base and not self.use_nodes:
                    # keep only modal wings (renamed to wingR/L); drop FE comps for the estimate path
                    comps = []
                    for c in base.get("components") or []:
                        nm = c.get("name") or ""
                        if nm.endswith("_modal"):
                            comps.append(dict(c, name=nm.replace("_modal", ""), role=None, estimated=False))
                        elif nm in ("wingR", "wingL") and not any(x.get("name") == nm + "_modal" for x in base["components"]):
                            comps.append(dict(c, estimated=False))
                    base = dict(base, components=comps) if comps else None
                self._base_structure = base
            row += [float(fx.get(n, float("nan"))) for n in self.struct_names]
        elif self.struct_names:
            row += [float("nan")] * len(self.struct_names)
        self.rows.append(row)
        self.t_last = t

    def _flex(self, flex_state, fdm) -> Dict[str, float]:
        """flex_state -> channel dict.

        Prefer ER FlexState /3 public API (``.nodes()``, ``.v2_geometry``, ``.v2_map_version``): ER already maps FE
        nodal wings + empennage + ``struct.*`` via our v2_map and keeps the 9-node modal wings as ``wing*_modal``.
        Fall back to ``node_source`` (older FlexState /2) or tip-only estimate. A plain FD raw dict is mapped
        completely (estimated)."""
        if flex_state is None:
            return {}
        from sim_bridge import v2_map as vm
        raw_dict = _is_raw_v2(flex_state)
        if raw_dict:
            raw = {k: float(v) for k, v in flex_state.items() if isinstance(v, (int, float))}
            er_ch: Dict[str, float] = {}
        else:
            er_ch = flex_channels(flex_state)
            raw = getattr(flex_state, "raw", None)
            raw = dict(raw) if isinstance(raw, dict) else {k[5:]: v for k, v in er_ch.items() if k.startswith("flex.")}
        if not (self.v2_map or raw_dict) or not vm.has_raw(raw):
            return er_ch

        # --- resolve nodal frame + geometry (FlexState /3 first, then NodeSource fallback) ---
        frame = None
        nodes_fn = getattr(flex_state, "nodes", None) if (not raw_dict and self.use_nodes) else None
        v2_geo = getattr(flex_state, "v2_geometry", None) if (not raw_dict and self.use_nodes) else None
        er_v2 = getattr(flex_state, "v2_map_version", None) if (not raw_dict and self.use_nodes) else None
        if callable(nodes_fn) and er_v2:
            try:
                frame = nodes_fn()
                self._flex_api = self._flex_api or "flexstate3"
            except Exception as e:  # noqa: BLE001
                if len(self.v2_stats["node_errors"]) < 5:
                    self.v2_stats["node_errors"].append(f"FlexState.nodes {type(e).__name__}: {e}")
        eta = getattr(flex_state, "eta", None)
        if frame is None and self.use_nodes and self.node_source is not None and eta is not None:
            try:
                frame = self.node_source.frame(eta)
                self._flex_api = self._flex_api or "nodesource"
            except Exception as e:  # noqa: BLE001
                if len(self.v2_stats["node_errors"]) < 5:
                    self.v2_stats["node_errors"].append(f"{type(e).__name__}: {e}")

        if not self.use_nodes and not raw_dict:
            # --no-nodes: drop FE-mapped component channels ER may have already added; keep modal wings + flex.*
            keep = {}
            for k, v in er_ch.items():
                root = k.split(".")[0]
                if k.startswith("flex.") or k.startswith("struct.") or root.endswith("_modal"):
                    keep[k] = v
                elif root in ("wingR", "wingL") and f"{root}_modal.dz.0" not in er_ch:
                    keep[k] = v  # reduced / older: modal wings still named wingR/L
            # rename wing*_modal -> wingR/L for the viewer when we stripped FE wings
            if any(k.startswith("wingR_modal.") for k in er_ch):
                for k, v in list(er_ch.items()):
                    if k.startswith("wingR_modal.") or k.startswith("wingL_modal."):
                        keep[k.replace("_modal", "", 1)] = v
            er_ch = keep
            er_v2 = None  # force our estimate path for empennage

        if self._geo is None:
            geo = v2_geo if self.use_nodes else None
            if geo is None and self.use_nodes and self.node_source is not None:
                try:
                    from sim_bridge.fd_nodes import rp_offset_ft
                    geo = vm.geometry_from_layout(self.node_source.layout(rp_offset_ft(fdm)), self.aircraft)
                except Exception as e:  # noqa: BLE001
                    self.v2_stats["node_errors"].append(f"layout {type(e).__name__}: {e}")
            if geo is None and self.use_nodes and callable(getattr(flex_state, "node_layout", None)):
                try:
                    lay = flex_state.node_layout()
                    if lay is not None:
                        geo = vm.geometry_from_layout(lay, self.aircraft)
                        self._flex_api = self._flex_api or "flexstate3"
                except Exception as e:  # noqa: BLE001
                    self.v2_stats["node_errors"].append(f"node_layout {type(e).__name__}: {e}")
            if geo is None:
                geo = vm.geometry_estimated(self.aircraft, fdm=fdm, wing_structure=getattr(flex_state, "structure", None))
                self._flex_api = self._flex_api or "estimate"
            self._geo = geo
            self.v2_stats["flex_api"] = self._flex_api
            # ER /3 already mapped FE wings + empennage into channels / structure: trust them (no re-emit)
            if er_v2 and any(k.startswith("wingR_modal.") or k.startswith("wingR.dz.32") or k.startswith("htail.")
                             for k in er_ch):
                self._er_mapped = True
                self._comps = []
            else:
                er_present = {c.split(".")[0] for c in er_ch if c.count(".") == 2 and c.split(".")[0] in vm.COMPONENTS}
                use_nodal_wings = self.wings == "fd_nodes" or (self.wings == "auto" and frame is not None)
                self._comps = vm.available_components(
                    raw, frame, [c for c in vm.COMPONENTS if c not in er_present or
                                 (use_nodal_wings and c in ("wingR", "wingL"))])

        comps = self._comps
        if frame is not None:
            self.v2_stats["node_frames"] += 1
            from sim_bridge.fd_nodes import tip_mismatch
            self.v2_stats["tip_check_max_abs"] = max(self.v2_stats["tip_check_max_abs"], tip_mismatch(frame, raw))
        else:
            self.v2_stats["estimated_frames"] += 1

        if self._er_mapped:
            # ER's channels() already has FE wingR/L + htail/vtail/fuselage + struct.* + wing*_modal + flex.*
            for nm in vm.COMPONENTS:
                if f"{nm}.dz.0" in er_ch or f"{nm}.dy.0" in er_ch or f"{nm}.dx.0" in er_ch:
                    self._status_seen.setdefault(nm, set()).add("fd_nodes" if frame is not None else "estimated")
            self._compare_wings_modal(er_ch)
            return er_ch

        ch = vm.map_v2_record(raw, self._geo, frame, components=comps)
        for nm, st in vm.component_status(self._geo, frame, comps).items():
            self._status_seen.setdefault(nm, set()).add(st)
        self._compare_wings(er_ch, ch, comps)
        out = {k: v for k, v in er_ch.items() if k.split(".")[0] not in comps or k.count(".") != 2}
        out.update({"flex." + k: float(v) for k, v in raw.items() if isinstance(v, (int, float))})
        out.update(ch)
        return out

    def _compare_wings_modal(self, er_ch):
        """ER /3: FE wingR/wingL vs wingR_modal/wingL_modal at coincident span fractions."""
        geo = self._geo
        for side in ("R", "L"):
            fe, mod = f"wing{side}", f"wing{side}_modal"
            if f"{mod}.dz.0" not in er_ch or f"{fe}.dz.0" not in er_ch or fe not in (geo or {}).get("components", {}):
                continue
            fr = geo["components"][fe]["node_span_frac"]
            n_mod = sum(1 for k in er_ch if k.startswith(mod + ".dz."))
            st = self.v2_stats["wing_modal_vs_nodal"].setdefault(fe, {
                "n_er_nodes": n_mod, "n_fd_nodes": len(fr), "dz_max_abs_diff_m": 0.0, "dz_max_abs_m": 0.0,
                "twist_max_abs_diff_rad": 0.0, "twist_max_abs_rad": 0.0, "dx_max_abs_m": 0.0, "frames": 0,
                "modal_name": mod})
            st["frames"] += 1
            for j in range(n_mod):
                xi = j / (n_mod - 1) if n_mod > 1 else 0.0
                i = min(range(len(fr)), key=lambda q: abs(fr[q] - xi))
                if abs(fr[i] - xi) > 1e-9:
                    continue
                for dof, kd, km in (("dz", "dz_max_abs_diff_m", "dz_max_abs_m"),
                                    ("twist", "twist_max_abs_diff_rad", "twist_max_abs_rad")):
                    a, b = er_ch.get(f"{mod}.{dof}.{j}"), er_ch.get(f"{fe}.{dof}.{i}")
                    if a is None or b is None:
                        continue
                    # ER fd_to_structure_channels uses `name == "wingR"` for the +twist sign; after the /3 rename
                    # to wingR_modal that check fails and wingR_modal.twist is written with the wingL sign. Undo
                    # that for the comparison metric (channels themselves are passed through unchanged).
                    if dof == "twist" and side == "R" and mod.endswith("_modal"):
                        a = -a
                        st["wingR_modal_twist_sign_flipped"] = True
                    st[kd] = max(st[kd], abs(a - b))
                    st[km] = max(st[km], abs(b))
            st["dx_max_abs_m"] = max(st["dx_max_abs_m"],
                                     max((abs(v) for k, v in er_ch.items() if k.startswith(fe + ".dx.")), default=0.0))

    def _compare_wings(self, er_ch, ch, comps):
        """ER's modal wing nodes (9, xi = j/8) vs FD's FE nodal wing values at the same span fractions."""
        geo = self._geo
        for nm in ("wingR", "wingL"):
            if nm not in comps or f"{nm}.dz.0" not in er_ch or nm not in geo["components"]:
                continue
            fr = geo["components"][nm]["node_span_frac"]
            n_er = sum(1 for k in er_ch if k.startswith(nm + ".dz."))
            st = self.v2_stats["wing_modal_vs_nodal"].setdefault(nm, {"n_er_nodes": n_er, "n_fd_nodes": len(fr),
                                                                      "dz_max_abs_diff_m": 0.0, "dz_max_abs_m": 0.0,
                                                                      "twist_max_abs_diff_rad": 0.0, "twist_max_abs_rad": 0.0,
                                                                      "dx_max_abs_m": 0.0, "frames": 0})
            st["frames"] += 1
            for j in range(n_er):
                xi = j / (n_er - 1)
                i = min(range(len(fr)), key=lambda q: abs(fr[q] - xi))
                if abs(fr[i] - xi) > 1e-9:
                    continue
                for dof, kd, km in (("dz", "dz_max_abs_diff_m", "dz_max_abs_m"), ("twist", "twist_max_abs_diff_rad", "twist_max_abs_rad")):
                    a, b = er_ch.get(f"{nm}.{dof}.{j}"), ch.get(f"{nm}.{dof}.{i}")
                    if a is None or b is None:
                        continue
                    st[kd] = max(st[kd], abs(a - b))
                    st[km] = max(st[km], abs(b))
            st["dx_max_abs_m"] = max(st["dx_max_abs_m"], max((abs(v) for k, v in ch.items() if k.startswith(nm + ".dx.")), default=0.0))

    def node_status(self) -> Dict[str, str]:
        return {nm: ("fd_nodes" if s == {"fd_nodes"} else "estimated") for nm, s in self._status_seen.items()}

    @property
    def structure(self) -> Optional[Dict]:
        if not self.struct_names:
            return None
        if self._geo is None:
            return self._base_structure
        from sim_bridge import v2_map as vm
        stats = dict(self.v2_stats, node_status=self.node_status(),
                     fd_nodes_schema=("fd-flexbody-nodes/1 via FlexState.nodes()" if self._flex_api == "flexstate3"
                                      else ("fd-flexbody-nodes/1 (NodeSource / flexbody.node_values)"
                                            if self.node_source is not None else None)))
        # FlexState /3: ER already built the structure block (FE comps + wing*_modal); attach our stats
        if self._er_mapped and self._base_structure:
            out = dict(self._base_structure)
            vm_blk = dict(out.get("v2_map") or {})
            vm_blk.update({k: stats[k] for k in ("node_frames", "estimated_frames", "node_errors",
                                                 "tip_check_max_abs", "wing_modal_vs_nodal", "node_status",
                                                 "fd_nodes_schema", "flex_api") if k in stats})
            out["v2_map"] = vm_blk
            return out
        base = None
        comps = self._comps or []
        if self._base_structure:
            kept = [dict(c) for c in self._base_structure.get("components", []) if c.get("name") not in comps]
            for c in kept:
                c.setdefault("estimated", False)
                c.setdefault("node_values_source", "ER modal state (evolution/fidelity.py fd_to_structure_channels)")
            if kept:
                base = dict(self._base_structure, components=kept)
        return vm.structure_block(self._geo, components=comps, estimated=self.node_status(), base=base, extra=stats)

    @property
    def channels(self) -> List[str]:
        return BASE_CHANNELS + self.struct_names

    def rounded_rows(self) -> List[List[float]]:
        dec = [ROUND.get(c, STRUCT_DECIMALS) for c in self.channels]
        return [[round(float(v), d) if math.isfinite(v) else None for v, d in zip(r, dec)] for r in self.rows]
