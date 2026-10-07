"""Show the trajectory viewer from Python: Colab / Jupyter / plain scripts.

Two ways:

1. Serve the folder and embed it (full viewer, all generations, lazy loading):
       import colab_viewer
       colab_viewer.show_served("data/runs/seed1-pop48/trajectories/index.json")
   In Colab this uses google.colab.output.serve_kernel_port_as_iframe; elsewhere it
   starts a local http.server and returns/prints the URL (and embeds an IFrame in Jupyter).

2. Inline everything into one self-contained HTML (works offline, file://, nbviewer, email):
       colab_viewer.show_inline("data/runs/seed1-pop48/trajectories/index.json", gens="improvements", hz=10)
       colab_viewer.build_standalone(index, "c172x_viewer.html")   # just write the file
   Uses the prebuilt bundle viewer/dist/fv.bundle.js (three.js + viewer, no network needed).
   Trajectories are decimated (default 10 Hz) and a subset of generations is embedded to keep
   the output small (a full 30 Hz file is ~0.75 MB per generation).
"""
from __future__ import annotations

import functools
import html
import http.server
import json
import os
import threading
from typing import Iterable, Optional, Union

HERE = os.path.dirname(os.path.abspath(__file__))
_SERVERS = {}


# ------------------------------------------------------------------ served mode
def serve(port: int = 8000, root: str = HERE) -> str:
    """Start (once) a background static file server on 0.0.0.0:port serving `root`."""
    if port in _SERVERS:
        return f"http://localhost:{port}/"

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), functools.partial(Quiet, directory=root))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _SERVERS[port] = httpd
    return f"http://localhost:{port}/"


def viewer_path(index_rel: Optional[str] = None, **params) -> str:
    """Viewer URL path. index_rel is relative to the sim-bridge root (e.g. data/runs/<id>/trajectories/index.json)."""
    q = {}
    if index_rel:
        q["index"] = "../" + index_rel.lstrip("/")
    q.update({k: v for k, v in params.items() if v is not None})
    qs = "&".join(f"{k}={v}" for k, v in q.items())
    return "/viewer/" + (f"?{qs}" if qs else "")


def show_served(index_rel: Optional[str] = None, port: int = 8000, height: int = 820, **params):
    """Embed the served viewer. Extra params become URL params (mode='compare', gens='0,5,39', cam='orbit', ...)."""
    serve(port)
    path = viewer_path(index_rel, **params)
    try:
        from google.colab import output  # type: ignore
        output.serve_kernel_port_as_iframe(port, path=path, height=height)
        return None
    except ImportError:
        url = f"http://localhost:{port}{path}"
        print("viewer:", url)
        try:
            from IPython.display import IFrame, display
            display(IFrame(url, width="100%", height=height))
        except ImportError:
            pass
        return url


# ------------------------------------------------------------------ inline / standalone mode
def _load(path: str):
    import gzip
    with open(path, "rb") as f:
        b = f.read()
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    return json.loads(b)


def _decimate(traj: dict, hz: Optional[float]) -> dict:
    if not hz or not traj.get("sample_hz") or hz >= traj["sample_hz"]:
        return traj
    step = max(1, int(round(traj["sample_hz"] / hz)))
    data = traj["data"][::step]
    if traj["data"] and data[-1] is not traj["data"][-1]:
        data = data + [traj["data"][-1]]  # keep the final state (end / envelope violation)
    return {**traj, "data": data, "sample_hz": traj["sample_hz"] / step}


REQUIRED = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m",
            "phi", "theta", "psi", "throttle", "elevator", "aileron", "rudder"]
SLIM_EXTRAS = ["target_alt_m", "alt_target_m", "target_cmd_alt_m", "kcas", "vc_kts", "ktas", "vtrue_kts", "nz", "alt_agl_m"]


SLIM_DECIMALS = {"t": 3, "x": 2, "y": 2, "z": 2, "alt_msl_m": 2, "alt_agl_m": 1, "vx": 2, "vy": 2, "vz": 2,
                 "qw": 5, "qx": 5, "qy": 5, "qz": 5, "phi": 5, "theta": 5, "psi": 5,
                 "target_alt_m": 2, "alt_target_m": 2, "target_cmd_alt_m": 2, "kcas": 2, "vc_kts": 2,
                 "ktas": 2, "vtrue_kts": 2, "nz": 3}


def _slim(traj: dict, keep: Optional[Iterable[str]]) -> dict:
    """Keep only the required channels + `keep` extras (drops lat/lon, body velocities, rates...)."""
    if keep is None:
        return traj
    keep = set(REQUIRED) | set(keep)
    if traj.get("structure"):  # soft-body channels <component>.<dof>.<node> stay (5 decimals: 10 um / 1e-5 rad)
        keep |= {c for c in traj["channels"] if c.count(".") == 2}
    idx = [i for i, c in enumerate(traj["channels"]) if c in keep]
    chans = [traj["channels"][i] for i in idx]
    # display precision: cm for positions/altitudes, 1e-5 for quaternions/angles (~0.0006 deg), 1e-4 controls
    dec = [SLIM_DECIMALS.get(c, 5 if c.count(".") == 2 else 4) for c in chans]

    def r(v, d):
        return round(v, d) if isinstance(v, float) else v
    return {**traj, "channels": chans,
            "channel_units": {k: v for k, v in traj.get("channel_units", {}).items() if k in keep},
            "data": [[r(row[i], d) for i, d in zip(idx, dec)] for row in traj["data"]]}


def _slim_structure(traj: dict, opts: Optional[dict]) -> dict:
    """Display-only reduction of the soft-body block (page size): `node_stride` k keeps every k-th node (+ the last)
    of components with more than `min_nodes` (default 17) nodes, renumbering their channels; `drop_modal` drops the
    `*_modal` comparison components when FE wings exist (the viewer never renders them); `drop_zero` drops structure
    channels that are exactly 0 for the whole flight (e.g. dy of the wings; a missing channel displays as 0).
    The deformation interpolates linearly between the kept nodes. Never used for comparisons / replay proofs."""
    st = traj.get("structure")
    if not opts or not st or not isinstance(st.get("components"), list):
        return traj
    k = int(opts.get("node_stride") or 1)
    min_nodes = int(opts.get("min_nodes") or 17)
    ch = traj["channels"]
    names = {c.get("name") for c in st["components"]}
    drop_modal = bool(opts.get("drop_modal")) and any(n in names for n in ("wingR", "wingL"))
    comps, rename, dropped = [], {}, set()
    for c in st["components"]:
        nm = c.get("name")
        if drop_modal and str(nm).endswith("_modal"):
            dropped.add(nm)
            continue
        nodes = c.get("axis_nodes_body_m") or []
        n = len(nodes)
        if k > 1 and n > min_nodes:
            keep = list(range(0, n, k))
            if keep[-1] != n - 1:
                keep.append(n - 1)
            # every per-node list (axis_nodes_body_m, node_span_frac, B1 r1 chord_m / geometric_twist_rad /
            # le_nodes_body_m / te_nodes_body_m, ...) is subsampled with the same nodes
            per_node = {key: [v[i] for i in keep] for key, v in c.items()
                        if key != "dof" and isinstance(v, list) and len(v) == n}
            c = {**c, **per_node, "display_node_stride": k, "display_nodes_from": n}
            for j, i in enumerate(keep):
                for d in (c.get("dof") or ["dz", "dy", "dx", "twist"]):
                    rename[f"{nm}.{d}.{i}"] = f"{nm}.{d}.{j}"
            for i in range(n):
                if i not in keep:
                    for d in (c.get("dof") or ["dz", "dy", "dx", "twist"]):
                        rename.setdefault(f"{nm}.{d}.{i}", None)
        comps.append(c)
    idx, out_ch = [], []
    zero = bool(opts.get("drop_zero"))
    data = traj["data"]
    for i, c in enumerate(ch):
        comp = c.split(".")[0] if c.count(".") == 2 else None
        if comp in dropped:
            continue
        new = rename.get(c, c)
        if new is None:
            continue
        if zero and comp is not None and all((row[i] == 0 or row[i] is None) for row in data):
            continue
        idx.append(i)
        out_ch.append(new)
    st2 = {**st, "components": comps,
           "display_slim": {"node_stride": k, "min_nodes": min_nodes, "dropped_components": sorted(dropped),
                            "drop_zero_channels": zero,
                            "note": "display-only reduction (standalone page size); not the recorded data"}}
    dec = opts.get("decimals")
    if dec is not None:  # structure channels only (display precision, e.g. 4 = 0.1 mm / 1e-4 rad)
        st2["display_slim"]["decimals"] = int(dec)
        isd = [c.count(".") == 2 for c in out_ch]
        rows = [[(round(row[i], int(dec)) if (s_ and isinstance(row[i], float)) else row[i]) for i, s_ in zip(idx, isd)]
                for row in data]
    else:
        rows = [[row[i] for i in idx] for row in data]
    return {**traj, "structure": st2, "channels": out_ch, "data": rows}


def _pick(entries, gens, sense: str = "min") -> list:
    if gens is None or gens == "improvements":  # per aircraft, at most 8 each (respects fitness_sense)
        res = []
        sgn = -1.0 if sense == "max" else 1.0
        for ac in dict.fromkeys(e.get("aircraft") for e in entries):
            out, best = [], float("inf")
            for e in sorted((e for e in entries if e.get("aircraft") == ac and e.get("is_best") is not False),
                            key=lambda e: e.get("generation") or 0):
                f = e.get("cost", e.get("fitness"))
                f = None if f is None else sgn * f
                if f is not None and f < best - 1e-12:
                    best = f
                    out.append(e)
            if len(out) > 8:
                idx = sorted({round(k * (len(out) - 1) / 7) for k in range(8)})
                out = [out[i] for i in idx]
            res += out
        return res or entries[:1]
    if gens == "all":
        return entries
    want = {int(g) for g in (gens.split(",") if isinstance(gens, str) else gens)}
    return [e for e in entries if e.get("generation") in want]


def _split_spec(spec, default_gens):
    """'path/index.json' or 'path/index.json@0,19' -> (path, gens)."""
    if isinstance(spec, (tuple, list)):
        return spec[0], (spec[1] if len(spec) > 1 else default_gens)
    if "@" in spec and not os.path.exists(spec):
        p, g = spec.rsplit("@", 1)
        return p, g
    return spec, default_gens


def build_standalone_html(index_path, gens: Union[str, Iterable[int], None] = "improvements",
                          hz: Optional[float] = 10.0, params: Optional[dict] = None,
                          slim: bool = False, title: Optional[str] = None,
                          struct_slim: Optional[dict] = None) -> str:
    """Return a single self-contained HTML string with viewer + three.js + selected trajectories.

    `index_path` is one index.json, or a list of them (several runs, e.g. 3 seeds) -> one combined index whose
    entries carry `run`. Each item may be 'path@gens' (or a (path, gens) tuple) to override `gens` for that run.
    `struct_slim` (optional, display only): {"node_stride": 2, "drop_modal": True, "drop_zero": True} - see
    _slim_structure; for multi-seed pages that would otherwise be too large.
    """
    specs = [index_path] if isinstance(index_path, (str, os.PathLike)) else list(index_path)
    files, all_entries, runs, first_index, list_key = {}, [], [], None, None
    senses = set()
    for spec in specs:
        path, g = _split_spec(spec, gens)
        index = _load(path)
        base = os.path.dirname(os.path.abspath(path))
        lk = next((k for k in ("trajectories", "files", "entries", "generations", "items")
                   if isinstance(index, dict) and k in index), None)
        entries = index if isinstance(index, list) else index[lk]
        run_id = index.get("run_id") if isinstance(index, dict) else None
        if first_index is None:
            first_index, list_key = index, lk
        raw_sense = (index.get("fitness_sense") if isinstance(index, dict) else None) or "min"
        sense = "max" if str(raw_sense).strip().lower().startswith("max") else "min"  # ER also writes free text
        senses.add(sense)
        chosen = _pick(entries, g, sense)
        for e in chosen:
            fn = e.get("file") or e.get("path")
            key = os.path.basename(fn)
            if key in files:
                raise ValueError(f"duplicate trajectory file name {key} across runs")
            files[key] = _slim(_slim_structure(_decimate(_load(os.path.join(base, fn)), hz), struct_slim),
                               SLIM_EXTRAS if slim else None)
            all_entries.append({**e, "run": e.get("run") or run_id} if len(specs) > 1 else e)
        runs.append({"run_id": run_id, "index": os.path.relpath(path), "gens": g if isinstance(g, str) else list(g or []),
                     "n": len(chosen)})
    if len(specs) == 1:
        idx = dict(first_index) if isinstance(first_index, dict) else {}
        idx[list_key or "trajectories"] = all_entries
    else:
        idx = {"schema": "ga-flightsim-traj-index/1", "traj_schema": "ga-flightsim-traj/1",
               "run_id": " + ".join(r["run_id"] or "?" for r in runs), "runs": runs, "entries": all_entries}
        if len(senses) == 1:
            idx["fitness_sense"] = senses.pop()
    inline = {"index": idx, "files": files}

    page = open(os.path.join(HERE, "viewer", "index.html")).read()
    css = open(os.path.join(HERE, "viewer", "style.css")).read()
    bundle = open(os.path.join(HERE, "viewer", "dist", "fv.bundle.js")).read().replace("</script", "<\\/script")
    data_js = json.dumps(inline, separators=(",", ":")).replace("</", "<\\/")
    # strip importmap + external module/css; inline instead
    import re
    page = re.sub(r'<script type="importmap">.*?</script>', "", page, flags=re.S)
    page = page.replace('<link rel="stylesheet" href="./style.css">', f"<style>{css}</style>")
    if title:
        page = re.sub(r"<title>.*?</title>", f"<title>{html.escape(title)}</title>", page)
    page = page.replace('<script type="module" src="./js/main.js"></script>',
                        f"<script>window.FV_INLINE={data_js};</script>\n<script type=\"module\">{bundle}</script>")
    if params:  # e.g. {"mode": "compare", "cam": "orbit"}; same keys as the viewer's URL params
        page = page.replace("<script>window.FV_INLINE=",
                            f"<script>window.FV_PARAMS={json.dumps(params)};</script><script>window.FV_INLINE=", 1)
    return page


def build_standalone(index_path: str, out_html: str, **kw) -> str:
    page = build_standalone_html(index_path, **kw)
    with open(out_html, "w") as f:
        f.write(page)
    return out_html


def show_inline(index_path: str, height: int = 820, **kw):
    """Display the self-contained viewer in a notebook output cell (iframe srcdoc; no server needed)."""
    from IPython.display import HTML, display
    page = build_standalone_html(index_path, **kw)
    display(HTML(f'<iframe srcdoc="{html.escape(page, quote=True)}" style="width:100%;height:{height}px;border:0" '
                 f'allow="fullscreen"></iframe>'))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build a single-file HTML viewer with embedded trajectories.")
    ap.add_argument("index", nargs="+", help="trajectories/index.json of one or more runs; 'path@0,19' sets gens per run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gens", default="improvements", help="'improvements' (default), 'all', or e.g. 0,5,39")
    ap.add_argument("--hz", type=float, default=10.0, help="decimate to this rate (0 = keep)")
    ap.add_argument("--param", action="append", default=[], help="viewer URL param preset, e.g. --param mode=compare")
    ap.add_argument("--slim", action="store_true", help="keep only required channels + target/airspeed/nz/AGL")
    ap.add_argument("--title")
    a = ap.parse_args()
    prm = dict(x.split("=", 1) for x in a.param) or None
    p = build_standalone(a.index if len(a.index) > 1 else a.index[0], a.out, gens=a.gens, hz=a.hz or None, params=prm, slim=a.slim, title=a.title)
    print(f"wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
