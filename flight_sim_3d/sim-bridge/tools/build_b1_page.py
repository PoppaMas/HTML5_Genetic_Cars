#!/usr/bin/env python3
"""One-shot Phase 3-B1 replay page: validate -> build (auto-fit < 24 MB) -> screenshots -> [replay proof] -> summary.

    tools/build_b1_page.sh <run_id>[,<run_id2>,...] [--replay-proof] [--fd-dir DIR] [--proof-gens traj|ends|0,4]
                           [--max-mb 24] [--out data] [--no-shots] [--allow-live-fd]

Reads ER's run read-only (evolution/runs/<run_id>/: run.json, trajectories/). Several comma-separated run ids (seeds)
make one page with seed-overlay presets; the replay proof then covers every run.

A/B (baseline vs tweaked):  tools/build_b1_page.sh <baseline>,<tweaked> --ab [--replay-proof]
   (automatic for exactly two runs of different families, e.g. phase3b1r1-pilot-s1 vs phase3b1r1-pilot-tweaked-s1).
   Page data/<baseline>_vs_<tag>_ab_standalone.html with only the gens both runs logged (matching gens), default view
   = every aircraft from both runs at the final common gen; presets "A/B <a> vs <b> g<N>" per common gen and
   "A/B planform top"; screenshots ab_*; summary adds an A/B cost table per aircraft and gen.

1. Validation: ER's `evolution.validate_traj` (when importable), our tools/check_traj.py (hard checks: status ok, no
   NaN, quaternion norm / quat-vs-Euler, position vs velocity, index fitness), and per file: `planform` header
   fd-planform/1 valid (sim_bridge.planform), not synthetic, `fidelity` == full_a1_b1 (--fidelity), model_version ==
   run.json pin.  Any failure stops the build (exit 2).
2. Page data/<run_id>_standalone.html (several runs: data/<run1>+<n>_standalone.html): compare mode, final gen of all
   aircraft in formation, chase cam, flex x8, planform HUD line; presets gen 0 vs last, planform top view gen 0 vs last
   (plan:<ac>, camera 'top'), gen 0/mid/last, seeds when several runs; on-page caveat banner (NOTE).
3. Auto-fit: (10 Hz, wing node stride 2) -> (10 Hz, stride 4) -> (5 Hz, stride 4) -> drop middle gens (first and final
   always kept) until raw <= --max-mb. Display-only slimming (`*_modal` and all-zero structure channels dropped,
   structure values rounded to 1e-4); flight data otherwise as ER wrote it (decimated).
4. Screenshots (prefix <run_id>_; tools/screenshots.py --strict): default formation, top-view planform gen 0 vs final
   per aircraft, one flex close-up. FAIL on console / page errors, blank render or an aircraft out of view.
5. --replay-proof: replay.py at the rows' own fidelity with pin= (ER raises on a model_version mismatch) for the
   trajectory gens (default; 'ends' = gen 0 + final) of every aircraft, against ER's FROZEN FD copy: --fd-dir, else a
   path in run.json (any fd/pin dir key), else the newest evolution/_fd_pin_* whose model_versions (ER's own
   eval.describe, probed in a subprocess) equal run.json's pins. No match -> exit 3 (live FD only with --allow-live-fd).
   Requires: cost rel_err == 0 for every row/scenario, pinned_model_version_match, and every NON-WING channel
   bit-identical to ER's scenario-0 file (wing channels reported).  Runs in parallel with steps 2-4.
6. Summary (stdout + data/<page>_build_summary.json): path, raw/gzip size, settings, gens, cost table gen 0 vs final,
   planform final vs baseline (sweep, taper, tip twist, shape genes), screenshot paths, timings.
"""
import argparse
import glob
import gzip
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import colab_viewer  # noqa: E402
from sim_bridge import paths, planform as P  # noqa: E402

NOTE = "FD structure nodes follow evolved sweep only; chord/twist from planform strips"     # B1 r0 node_layout
NOTE_R1 = ("r1: FE nodes follow FD's shaped layout (chord, sweep, AC shift; twist about the elastic axis). "
           "The 9-node wing modal display axes still use approximate geometry.")
NOTE_B2A = "B2a: dihedral baked into FD node layout; t/c and camber shown as section metadata."   # == traj.js B2A_NOTE
B2_DEFAULTS = {"wing_dihedral_delta_deg": 0.0, "wing_tc_root_scale": 1.0, "wing_tc_tip_ratio": 1.0,
               "wing_camber_root_delta_pct": 0.0, "wing_camber_tip_delta_pct": 0.0}


def pick_note(runs, override):
    """The flex/planform caveat must describe the data: files whose structure carries the `node_layout` tag naming
    node_layout_b1 (B1 r1) -> NOTE_R1; files without the tag (r0) -> NOTE (sweep-only). --note overrides."""
    if override:
        return override, "--note"
    R = runs[0]
    e = R["entries"][-1]
    with open(os.path.join(R["dir"], "trajectories", e["file"])) as f:
        doc = json.load(f)
    nl = str((doc.get("structure") or {}).get("node_layout") or "")
    if "node_layout_b2" in nl:
        return NOTE_B2A, f"B2a data (structure.node_layout = {nl!r})"
    return (NOTE_R1, f"r1 data (structure.node_layout = {nl!r})") if "node_layout_b1" in nl else (NOTE, "r0 data")
WING_PREFIXES = ("wingR.", "wingL.", "wingR_modal.", "wingL_modal.")
LADDER = [(10.0, 2), (10.0, 4), (5.0, 4)]
SHAPE_DEFAULTS = {"wing_chord_taper_1": 1.0, "wing_chord_taper_2": 1.0, "wing_chord_taper_3": 1.0,
                  "wing_twist_mid_deg": 0.0, "wing_twist_tip_deg": 0.0, "wing_sweep_qc_delta_deg": 0.0}
PY = os.environ.get("PY") or sys.executable
T0 = time.perf_counter()


def log(*a):
    print(f"[{time.perf_counter() - T0:6.1f}s]", *a, flush=True)


def die(code, msg):
    print(f"\nBUILD FAILED ({code}): {msg}", flush=True)
    sys.exit(code)


def er_root():
    return os.path.dirname(os.path.abspath(paths.RUNS_ROOT))


def run_dir_of(rid):
    d = rid if os.path.isdir(rid) else os.path.join(paths.RUNS_ROOT, rid)
    if not os.path.isfile(os.path.join(d, "trajectories", "index.json")):
        die(2, f"{rid}: no trajectories/index.json under {d} (has ER handed the run over?)")
    return os.path.abspath(d)


def load_run(rid):
    d = run_dir_of(rid)
    run = json.load(open(os.path.join(d, "run.json")))
    idx = json.load(open(os.path.join(d, "trajectories", "index.json")))
    return {"id": run.get("run_id") or os.path.basename(d), "dir": d, "run": run, "index": idx,
            "entries": idx.get("entries") or idx.get("trajectories") or []}


# ------------------------------------------------------------------ 1. validation
def validate(R, fidelity, need_aircraft):
    errs, notes = [], []
    tdir = os.path.join(R["dir"], "trajectories")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.path.dirname(er_root())}
    p = subprocess.run([PY, "-m", "evolution.validate_traj", tdir], capture_output=True, text=True, env=env,
                       cwd=os.path.dirname(er_root()))
    if p.returncode != 0 and "No module named" in (p.stderr or ""):
        notes.append("ER validator not importable; skipped")
    elif p.returncode != 0:
        errs.append("ER validate_traj: " + (p.stdout + p.stderr).strip()[-1500:])
    else:
        notes.append("ER validate_traj: " + p.stdout.strip().splitlines()[-1])
    ctj = os.path.join("/tmp", f"b1_check_{R['id']}.json")
    p = subprocess.run([PY, os.path.join(HERE, "tools", "check_traj.py"), R["dir"], "--json", ctj],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        errs.append("check_traj exit %d: %s" % (p.returncode, (p.stdout + p.stderr)[-800:]))
    else:
        rows = json.load(open(ctj))["runs"].get(R["id"], {}).get("rows", [])
        soft = sum(len(r.get("flags") or []) for r in rows)
        for r in rows:
            tag = f"{r['aircraft']} g{r['gen']}"
            if r.get("status") != "ok" or r.get("nan"):
                errs.append(f"check_traj {tag}: status {r.get('status')} nan {r.get('nan')}")
            if (r.get("quat_norm_err") or 0) > 1e-4 or (r.get("quat_euler_deg") or 0) > 0.1:
                errs.append(f"check_traj {tag}: quaternion check {r.get('quat_norm_err')} / {r.get('quat_euler_deg')} deg")
            if (r.get("pos_vel_err_m") or 0) > 1.0:
                errs.append(f"check_traj {tag}: position vs velocity {r.get('pos_vel_err_m')} m")
            if r.get("index_fitness_match") is False:
                errs.append(f"check_traj {tag}: index fitness != file fitness")
        notes.append(f"check_traj: {len(rows)} files, hard checks ok={not errs}, {soft} behavioural flags (info)")
    pins = R["run"].get("pin_model_version") or {}
    acs = set()
    for e in R["entries"]:
        f = os.path.join(tdir, e["file"])
        with open(f) as fh:
            doc = json.load(fh)
        tag = f"{e['file']}"
        acs.add(doc.get("aircraft"))
        field, blk = P.find_planform(doc)
        if blk is None:
            errs.append(f"{tag}: no planform header")
        else:
            if blk.get("schema") != P.PLANFORM_SCHEMA:
                errs.append(f"{tag}: planform schema {blk.get('schema')!r}")
            bad = P.validate_planform(blk)
            if bad:
                errs.append(f"{tag}: planform invalid: {bad[:3]}")
            if blk.get("synthetic"):
                errs.append(f"{tag}: planform marked synthetic")
        if doc.get("fidelity") != fidelity:
            errs.append(f"{tag}: fidelity {doc.get('fidelity')!r} != {fidelity}")
        pin = (pins.get(doc.get("aircraft")) or {}).get(fidelity)
        if pin and doc.get("model_version") != pin:
            errs.append(f"{tag}: model_version {doc.get('model_version')} != run pin {pin}")
        comps = {c["name"]: len(c.get("axis_nodes_body_m") or []) for c in (doc.get("structure") or {}).get("components", [])}
        if not comps.get("wingR"):
            errs.append(f"{tag}: no FD nodal wing structure")
        del doc
    missing = [a for a in need_aircraft if a not in acs]
    if missing:
        errs.append(f"aircraft missing: {missing} (have {sorted(acs)})")
    notes.append(f"{len(R['entries'])} files: planform fd-planform/1, fidelity {fidelity}, model_version == pin checked")
    return errs, notes


# ------------------------------------------------------------------ 2./3. page with auto-fit
def gens_of(R, ac):
    return sorted({e["generation"] for e in R["entries"] if e["aircraft"] == ac})


def run_family(rid):
    """'phase3b1r1-pilot-s1' -> 'phase3b1r1-pilot' (same rule as the viewer's runFamily: drop the last '-' token)."""
    k = rid.rfind("-")
    return rid[:k] if k > 0 else rid


def ab_tags(ids):
    """Short A/B names: drop the '-' tokens all runs share at both ends (empty -> 'base'), like the viewer's abShort."""
    tok = [i.split("-") for i in ids]
    a = b = 0
    n = min(len(t) for t in tok)
    while a < n and all(t[a] == tok[0][a] for t in tok):
        a += 1
    while b < n - a and all(t[len(t) - 1 - b] == tok[0][len(tok[0]) - 1 - b] for t in tok):
        b += 1
    names = ["-".join(t[a:len(t) - b]) or "base" for t in tok]
    return names if len(set(names)) == len(names) else list(ids)


def common_gens(runs, ac=None):
    """Generations every run logged (for aircraft `ac`, or for every aircraft)."""
    acs = [ac] if ac else sorted({e["aircraft"] for R in runs for e in R["entries"]})
    sets = [set(g for a in acs for g in gens_of(R, a)) if not ac else set(gens_of(R, ac)) for R in runs]
    return sorted(set.intersection(*sets)) if sets else []


def build(runs, out_html, max_mb, params, title, only_gens=None):
    all_gens = list(only_gens) if only_gens else sorted({e["generation"] for R in runs for e in R["entries"]})
    first, final = all_gens[0], all_gens[-1]
    middles = [g for g in all_gens if g not in (first, final)]
    plans = [(hz, st, all_gens) for hz, st in LADDER]
    keep = list(middles)
    while keep:   # drop middle gens (farthest from the centre last), at 5 Hz stride 4
        mid = (first + final) / 2
        keep.remove(max(keep, key=lambda g: abs(g - mid)) if len(keep) > 1 else keep[0])
        plans.append((5.0, 4, sorted([first, *keep, final])))
    tried = []
    for hz, stride, gens in plans:
        t = time.perf_counter()
        specs = [(os.path.join(R["dir"], "trajectories", "index.json"), ",".join(map(str, gens))) for R in runs]
        html = colab_viewer.build_standalone_html(specs if len(specs) > 1 else specs[0][0], gens=specs[0][1], hz=hz,
                                                  params=params, slim=True, title=title,
                                                  struct_slim={"node_stride": stride, "drop_modal": True,
                                                               "drop_zero": True, "decimals": 4})
        raw = len(html.encode())
        tried.append({"hz": hz, "node_stride": stride, "gens": gens, "raw_mb": round(raw / 1e6, 2),
                      "build_s": round(time.perf_counter() - t, 1)})
        log(f"  try hz={hz:g} stride={stride} gens={gens}: {raw / 1e6:.1f} MB ({tried[-1]['build_s']} s)")
        if raw <= max_mb * 1e6:
            with open(out_html, "w") as f:
                f.write(html)
            gz = len(gzip.compress(html.encode(), 6))
            return {"path": out_html, "raw_bytes": raw, "gzip_bytes": gz, "hz": hz, "node_stride": stride,
                    "gens": gens, "dropped_gens": [g for g in all_gens if g not in gens], "tried": tried}
    die(4, f"page does not fit {max_mb} MB even at the last step: {tried[-1]}")


# ------------------------------------------------------------------ 4. screenshots
def shots(runs, page, prefix, final_by_ac, b2=False):
    R = runs[0]
    spec = [["default_formation_final_defl8", "", 12]]
    for ac in sorted(final_by_ac):
        fin = final_by_ac[ac]
        doc_pf = planform_of(R, ac, fin)
        span = 2 * abs(doc_pf["wing"]["y_m"][-1]) if doc_pf else 20
        sp = int(math.ceil(span * 1.3))
        g0 = gens_of(R, ac)[0]
        spec.append([f"planform_top_{ac}_g{g0}_vs_g{fin}", f"?preset=plan:{ac}&cam=top&defl=1&spacing={sp}", 8])
    if "T38" in final_by_ac:   # T38 wing at mid-flight (B1 r0 had a T38 mid-twist artifact): rear + top close-ups
        for view in ("rear", "top"):
            spec.append([f"T38_wing_closeup_g{final_by_ac['T38']}_t45_{view}_defl8",
                         f"?mode=single&gen=T38:{final_by_ac['T38']}&cam=orbit&defl=8", 45, view + "+fit"])
    if b2:   # B2a: dihedral from front/rear at defl 1 (geometry, not flex), plus HUD + section sketch
        for ac in ("c172x", "737"):
            if ac in final_by_ac:
                for view in ("front", "rear"):
                    spec.append([f"b2_dihedral_{ac}_g{final_by_ac[ac]}_{view}", f"?mode=single&gen={ac}:{final_by_ac[ac]}"
                                 "&cam=orbit&defl=1", 8, view + "+fit"])
        for ac in ("c172x", "737"):
            if ac in final_by_ac:
                spec.append([f"b2_hud_section_{ac}_g{final_by_ac[ac]}", f"?mode=single&gen={ac}:{final_by_ac[ac]}"
                             "&cam=chase&defl=1", 8])
    big = "737" if "737" in final_by_ac else sorted(final_by_ac)[0]
    spec.append([f"flex_closeup_{big}_g{final_by_ac[big]}_rear_defl8",
                 f"?mode=single&gen={big}:{final_by_ac[big]}&cam=orbit&defl=8", 12, "rear+fit"])
    return run_shots(spec, page, prefix)


def ab_shots(runs, page, prefix, fin):
    """A/B screenshots at the final common gen: all aircraft from both runs, planform top per aircraft, T38 pair."""
    spec = [[f"ab_formation_g{fin}_defl8", "", 12]]
    acs = sorted({e["aircraft"] for e in runs[0]["entries"]})
    for ac in acs:
        pf = planform_of(runs[0], ac, fin)
        sp = int(math.ceil((2 * abs(pf["wing"]["y_m"][-1]) if pf else 20) * 1.3))
        spec.append([f"ab_planform_top_{ac}_g{fin}", f"?preset=abplan:{ac}:{fin}&cam=top&defl=1&spacing={sp}", 8])
    if "T38" in acs:
        spec.append([f"ab_T38_g{fin}_t45_chase_defl8", f"?preset=ab:T38:{fin}&cam=chase&defl=8&spacing=12&hud=0", 45])
    return run_shots(spec, page, prefix)


def parse_shot_report(out):
    """screenshots.py prints its JSON report first; warning lines after it may contain braces too."""
    try:
        return json.JSONDecoder().raw_decode(out[out.index("{"):])[0] if "{" in out else {}
    except ValueError:
        return {}


def run_shots(spec, page, prefix):
    sp_path = f"/tmp/b1_shots_{prefix}.json"
    json.dump(spec, open(sp_path, "w"))
    py = os.path.join(HERE, ".venv-shots", "bin", "python")
    cmd = [py if os.path.exists(py) else PY, os.path.join(HERE, "tools", "screenshots.py"), "--bench", page,
           "--shot-spec", sp_path, "--prefix", prefix, "--strict"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout
    rep = parse_shot_report(out)
    res = {"ok": p.returncode == 0, "pngs": [v["png"] for v in rep.values() if isinstance(v, dict) and "png" in v],
           "console_errors": [l for l in rep.get("console", []) if l.startswith("[error]")] + rep.get("page_errors", []),
           "blank": [k for k, v in rep.items() if isinstance(v, dict) and v.get("render", {}).get("blank")],
           "out_of_view": [k for k, v in rep.items() if isinstance(v, dict) and "render" in v
                           and not all(v["render"].get("inView", [True]))],
           "covered": {k: v["render"]["covered"] for k, v in rep.items() if isinstance(v, dict) and "render" in v
                       and any(v["render"].get("covered") or [])},
           "hud_planform": sorted({l for v in rep.values() if isinstance(v, dict) for l in v.get("hud", []) if l.startswith("PLANFORM")}),
           "log": f"/tmp/b1_shots_{prefix}.log"}
    open(res["log"], "w").write(out + p.stderr)
    missing = [sp[0] for sp in spec if not any(os.path.basename(x) == f"{prefix}{sp[0]}.png" for x in res["pngs"])]
    if missing:
        res["ok"] = False
        res["missing"] = missing
        log(f"screenshots: missing / unreported {missing}")
    return res


# ------------------------------------------------------------------ 5. replay proof
def probe_mv(fd_dir, run_json_path):
    code = ("import json,sys\nfrom evolution import eval as E, fidelity as F\nrun=json.load(open(sys.argv[1]))\n"
            "fid=run['fidelity']\nout={}\nfor ac in run['aircraft']:\n"
            "    g=F.per_aircraft(ac['name'], run.get('fidelity_per_aircraft'))['reduced_gate'] if hasattr(F,'per_aircraft') else None\n"
            "    d=E.describe(ac['resolved_profile'],[fid],g)\n    out[ac['name']]=d['model_version'].get(fid) or d.get('error')\n"
            "print(json.dumps(out))\n")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "EVOLUTION_FD_DIR": fd_dir, "FLIGHT_DYNAMICS_DIR": fd_dir,
           "PYTHONPATH": os.path.dirname(er_root())}
    p = subprocess.run([PY, "-c", code, run_json_path], capture_output=True, text=True, env=env, cwd=os.path.dirname(er_root()))
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": (p.stderr or p.stdout)[-400:]}


def fd_dir_candidates(R, explicit):
    c = []
    if explicit:
        c.append(("--fd-dir", os.path.abspath(explicit)))

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, str) and re.search(r"(fd|flight.?dynamics|pin)[^.]*(dir|path|copy|frozen|pin)|frozen", path, re.I):
            yield_.append((("config.json" + path[7:]) if path.startswith(".config") else f"run.json{path}", o))
    yield_ = []
    walk(R["run"])
    cfgp = os.path.join(R["dir"], "config.json")
    if os.path.exists(cfgp):
        walk(json.load(open(cfgp)), ".config")
    side = os.path.join(R["dir"], "fd_pin.json")      # older ER runs: sidecar instead of run.json fd_dir
    if os.path.exists(side):
        for k, v in json.load(open(side)).items():
            if isinstance(v, str) and re.search(r"dir|path|fd", k, re.I):
                yield_.append((f"fd_pin.json.{k}", v))
    for src, v in yield_:
        for base in (os.path.dirname(er_root()), er_root(), R["dir"]):
            p = v if os.path.isabs(v) else os.path.join(base, v)
            if os.path.isdir(p):
                c.append((src, os.path.abspath(p)))
                break
    for d in sorted(glob.glob(os.path.join(er_root(), "_fd_pin_*")), key=os.path.getmtime, reverse=True):
        c.append(("evolution/_fd_pin_*", d))
    for d in filter(None, os.environ.get("SIMBRIDGE_FD_DIRS", "").split(os.pathsep)):   # extra frozen copies (ours)
        c.append(("$SIMBRIDGE_FD_DIRS", os.path.abspath(d)))
    return c


def pick_fd_dir(R, explicit, allow_live):
    pins = R["run"].get("pin_model_version") or {}
    fid = R["run"].get("fidelity")
    want = {ac: (v or {}).get(fid) for ac, v in pins.items()}
    tried = []
    cands = fd_dir_candidates(R, explicit)
    if allow_live:
        cands.append(("live FD", os.path.join(os.path.dirname(er_root()), "flight-dynamics")))
    for src, d in cands:
        got = probe_mv(d, os.path.join(R["dir"], "run.json"))
        ok = all(got.get(ac) == mv for ac, mv in want.items() if mv)
        tried.append({"source": src, "dir": d, "model_versions": got, "match": ok})
        if ok:
            return d, src, tried
    return None, None, tried


def replay_proof(R, gens, fd_dir, out_dir):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "EVOLUTION_FD_DIR": fd_dir, "FLIGHT_DYNAMICS_DIR": fd_dir}
    cmd = [PY, os.path.join(HERE, "replay.py"), "--run", R["dir"], "--gens", ",".join(map(str, gens)), "--out", out_dir]
    return subprocess.Popen(cmd, stdout=open(out_dir + ".log", "w"), stderr=subprocess.STDOUT, env=env, cwd=HERE)


def proof_verdict(out_dir, rc):
    mpath = os.path.join(out_dir, "replay_manifest.json")
    if not os.path.exists(mpath):
        return {"ok": False, "error": f"no manifest (exit {rc}); see {out_dir}.log"}
    m = json.load(open(mpath))
    rows, problems = [], []
    for g in m.get("genomes", []):
        per = g.get("per_scenario") or {}
        mx = max([abs(v.get("rel_err") or 0) for v in per.values()] + [abs(g.get("rel_err") or 0)])
        rows.append({"aircraft": g["aircraft"], "gen": g["generation"], "logged": g.get("logged_cost"),
                     "replayed": g.get("replayed_cost"), "max_rel_err": mx, "pinned_match": g.get("pinned_model_version_match"),
                     "model_version": g.get("replay_model_version")})
        if g.get("replayed_cost") is None or mx != 0.0:
            problems.append(f"{g['aircraft']} g{g['generation']}: cost rel_err {mx}")
        if g.get("pinned_model_version_match") is not True:
            problems.append(f"{g['aircraft']} g{g['generation']}: pinned model_version match {g.get('pinned_model_version_match')}")
    files = []
    for k, f in ((m.get("trajectory_check") or {}).get("files") or {}).items():
        failed = f.get("failed_channels") or []
        nonwing = [c for c in failed if not str(c).startswith(WING_PREFIXES)]
        mx = f.get("max_abs") or {}
        is_struct = lambda c: str(c).count(".") == 2 and not str(c).startswith("flex.")   # <component>.<dof>.<node>
        vals = lambda pred: [v for c, v in mx.items() if pred(c) and v is not None and v == v]
        files.append({"max_abs_nonwing": max(vals(lambda c: not str(c).startswith(WING_PREFIXES)) or [0.0]),
                      "max_abs_structure": max(vals(is_struct) or [0.0]),
                      "max_abs_wing": max(vals(lambda c: str(c).startswith(WING_PREFIXES)) or [0.0]),
                      "n_structure_channels": len(vals(is_struct)),
                      "file": f.get("reference_file"), "bit_identical": f.get("bit_identical_channels"),
                      "n_channels": f.get("n_channels"), "failed_nonwing": nonwing,
                      "failed_wing": len(failed) - len(nonwing), "verdict": f.get("verdict"),
                      "rows": f"{f.get('rows_compared')}/{f.get('rows_reference')}"})
        if nonwing or f.get("rows_compared") != f.get("rows_reference"):
            problems.append(f"{f.get('reference_file')}: non-wing channels differ {nonwing[:5]} rows {f.get('rows_compared')}/{f.get('rows_reference')}")
    want_files = {(r["aircraft"], r["gen"]) for r in rows}
    if len(files) < len(want_files):
        problems.append(f"channel check covered {len(files)} files for {len(want_files)} replayed rows")
    if rc != 0:
        problems.append(f"replay.py exit {rc}")
    return {"ok": not problems and bool(rows), "problems": problems, "rows": rows, "files": files,
            "wall_s": m.get("wall_s"), "manifest": mpath}


# ------------------------------------------------------------------ 6. summary helpers
def genes_of(R, ac, gen):
    pf = planform_of(R, ac, gen)
    return (pf or {}).get("_genome") or {}


def planform_of(R, ac, gen):
    e = next((e for e in R["entries"] if e["aircraft"] == ac and e["generation"] == gen), None)
    if not e:
        return None
    with open(os.path.join(R["dir"], "trajectories", e["file"])) as f:
        doc = json.load(f)
    _, blk = P.find_planform(doc)
    if blk is not None:
        blk = dict(blk)
        blk["_genome"] = doc.get("genome") or {}
        blk["_fitness"] = doc.get("fitness")
    return blk


def pf_stats(blk):
    w = blk.get("wing") or blk.get("wingR")
    c, cb, tw = w["chord_m"], w.get("chord_baseline_m"), w["twist_rad"]
    gene_names = blk["genes"] if isinstance(blk["genes"], list) else list(blk["genes"])
    genes = {k: blk["_genome"].get(k, (blk["genes"].get(k) if isinstance(blk["genes"], dict) else None)) for k in gene_names}
    return {"sweep_deg": round(math.degrees(blk["sweep_qc_rad"]), 2),
            "sweep_base_deg": None if blk.get("sweep_qc_baseline_rad") is None else round(math.degrees(blk["sweep_qc_baseline_rad"]), 2),
            "taper": round(c[-1] / c[0], 3), "taper_base": None if not cb else round(cb[-1] / cb[0], 3),
            "twist_tip_deg": round(math.degrees(tw[-1]), 2), "twist_mid_deg": round(math.degrees(tw[len(tw) // 2]), 2),
            "genes": {k: (None if v is None else round(float(v), 4)) for k, v in genes.items()}}


def gene_bounds(fd_dirs):
    """{gene: (lo, hi)} from FD's newest model_versions_post_p3b1*.json b1_schema (frozen copy first, then live FD),
    plus per-aircraft B2 ranges {(gene, ac): (lo, hi)} from model_versions_post_p3b2*.json b2_schema when present."""
    out = {}
    for d in fd_dirs:
        fs = sorted(glob.glob(os.path.join(d, "v2_results", "model_versions_post_p3b2*.json")), key=os.path.getmtime)
        for f in reversed(fs):
            try:
                for g in json.load(open(f))["b2_schema"]["genes"]:
                    for ac, lo, hi in g.get("ranges", []):
                        out[(g["name"], ac)] = (lo, hi)
                break
            except Exception:  # noqa: BLE001
                continue
        if out:
            break
    return {**_b1_bounds(fd_dirs), **out}


def _b1_bounds(fd_dirs):
    for d in fd_dirs:
        fs = sorted(glob.glob(os.path.join(d, "v2_results", "model_versions_post_p3b1*.json")), key=os.path.getmtime)
        for f in reversed(fs):
            try:
                return {g["name"]: (g["lo"], g["hi"]) for g in json.load(open(f))["b1_schema"]["genes"]}
            except Exception:  # noqa: BLE001
                continue
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", help="ER run id (or comma list of seeds / run dirs)")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--max-mb", type=float, default=24.0)
    ap.add_argument("--fidelity", help="expected fidelity (default: run.json fidelity, else full_a1_b1)")
    ap.add_argument("--aircraft", default="c172x,T38,737", help="required aircraft")
    ap.add_argument("--no-shots", action="store_true")
    ap.add_argument("--replay-proof", action="store_true")
    ap.add_argument("--proof-gens", default="traj", help="'traj' (every gen with an ER trajectory, default), 'ends' "
                    "(gen 0 + final) or a comma list")
    ap.add_argument("--fd-dir", help="frozen FD copy for the proof (default: run.json / evolution/_fd_pin_* by pins)")
    ap.add_argument("--allow-live-fd", action="store_true", help="accept the live flight-dynamics/ if it matches the pins")
    ap.add_argument("--shot-prefix", help="screenshot file prefix (default '<run_id>_')")
    ap.add_argument("--note", help="on-page caveat banner text (default: chosen from the data's node layout, r0 / r1)")
    ap.add_argument("--ab", action="store_true", help="A/B page for two runs <baseline>,<tweaked> at matching gens "
                    "(automatic for two runs of different families)")
    ap.add_argument("--reuse-proof", action="store_true", help="reuse an existing EXACT proof (same run, gens, frozen "
                    "FD) under <out>/replays/<run>/ instead of replaying again (e.g. the baseline in an A/B build)")
    ap.add_argument("--name", help="page/summary base name (default from the run ids)")
    ap.add_argument("--gens", help="comma list: only these generations on the page (e.g. 0,59)")
    ap.add_argument("--no-ab", action="store_true", help="two runs of different families -> plain multi-run page")
    a = ap.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    runs = [load_run(r) for r in a.run_id.split(",") if r]
    ab = len(runs) == 2 and not a.no_ab and (a.ab or run_family(runs[0]["id"]) != run_family(runs[1]["id"]))
    if a.ab and len(runs) != 2:
        die(2, f"--ab needs exactly two runs <baseline>,<tweaked> (got {len(runs)})")
    fams = list(dict.fromkeys(run_family(R["id"]) for R in runs))
    # multi-seed A/B: >2 runs from exactly two families (<base-s1>,<tweaked-s1>,<base-s2>,<tweaked-s2>); fams[0] = base
    seeds_ab = len(runs) > 2 and len(fams) == 2 and not a.no_ab
    labels = ({R["id"]: ("base" if run_family(R["id"]) == fams[0] else "tweaked") + "-" + R["id"].rsplit("-", 1)[-1]
               for R in runs} if seeds_ab else None)
    tags = ab_tags([R["id"] for R in runs]) if ab else None
    name = (f"{runs[0]['id']}_vs_{tags[1]}_ab" if ab else
            f"{fams[0]}_ab_{len(runs) // 2}seed" if seeds_ab else
            runs[0]["id"] + (f"+{len(runs) - 1}" if len(runs) > 1 else ""))
    name = a.name or name
    ab_gens = common_gens(runs) if (ab or seeds_ab) else None
    if a.gens:
        want = [int(x) for x in a.gens.split(",") if x]
        ab_gens = [g for g in (ab_gens or want) if g in want]
    if ab and not ab_gens:
        die(2, f"A/B: no generation logged by both runs ({[sorted({e['generation'] for e in R['entries']}) for R in runs]})")
    page = os.path.join(a.out, f"{name}_standalone.html")
    timings, summary = {}, {"runs": [R["id"] for R in runs], "page": page}
    if ab:
        summary["ab"] = {"baseline": runs[0]["id"], "tweaked": runs[1]["id"], "tags": tags, "matching_gens": ab_gens}
        log(f"A/B: {runs[0]['id']} ({tags[0]}) vs {runs[1]['id']} ({tags[1]}), matching gens {ab_gens}")
    need = [x for x in a.aircraft.split(",") if x]
    if not a.fidelity:
        fids = {R["run"].get("fidelity") or "full_a1_b1" for R in runs}
        if len(fids) != 1:
            die(2, f"runs have different fidelities {sorted(fids)}; pass --fidelity")
        a.fidelity = fids.pop()
        log(f"fidelity {a.fidelity} (run.json)")

    # 1. validation
    t = time.perf_counter()
    for R in runs:
        errs, notes = validate(R, a.fidelity, need)
        for n in notes:
            log(f"validate {R['id']}: {n}")
        if errs:
            for e in errs[:30]:
                print("  ERROR", e)
            die(2, f"{R['id']}: {len(errs)} validation error(s)")
    timings["validate_s"] = round(time.perf_counter() - t, 1)

    # 5. replay proof: start first (long pole), in parallel with page + screenshots
    procs = []
    if a.replay_proof:
        t = time.perf_counter()
        for R in runs:
            d, src, tried = pick_fd_dir(R, a.fd_dir, a.allow_live_fd)
            for x in tried:
                log(f"proof {R['id']}: FD candidate {x['source']} {x['dir']}: {x['model_versions']} match={x['match']}")
            if not d:
                die(3, f"{R['id']}: no frozen FD copy matches the run's pins {R['run'].get('pin_model_version')} "
                       f"(tried {[x['dir'] for x in tried]}); pass --fd-dir <ER's frozen copy>")
            gl = sorted({e['generation'] for e in R['entries']})
            gens = gl if a.proof_gens == "traj" else [gl[0], gl[-1]] if a.proof_gens == "ends" else \
                [int(x) for x in a.proof_gens.split(",")]
            od = os.path.join(a.out, "replays", R["id"], f"b1proof-full-g{'.'.join(map(str, gens))}")
            os.makedirs(os.path.dirname(od), exist_ok=True)
            if a.reuse_proof and os.path.exists(os.path.join(od, "replay_manifest.json")):
                old = proof_verdict(od, 0)
                mpath = os.path.join(od, "replay_manifest.json")
                rmv = set((json.load(open(mpath)).get("provenance") or {}).get("replay_model_versions") or [])
                pins = {v.get(a.fidelity) for v in (R["run"].get("pin_model_version") or {}).values()} - {None}
                newer = os.path.getmtime(mpath) > max(os.path.getmtime(os.path.join(R["dir"], "trajectories", e["file"]))
                                                      for e in R["entries"])
                if old["ok"] and pins and pins <= rmv and newer and sorted({r["gen"] for r in old["rows"]}) == sorted(gens):
                    log(f"proof {R['id']}: reusing EXACT proof {od} (--reuse-proof)")
                    procs.append((R, gens, d, src + " (reused)", od, None))
                    continue
            log(f"proof {R['id']}: replay gens {gens} with FD {d} ({src}) -> {od}")
            procs.append((R, gens, d, src, od, replay_proof(R, gens, d, od)))
        timings["proof_setup_s"] = round(time.perf_counter() - t, 1)

    # 2./3. page
    t = time.perf_counter()
    final_by_ac = {ac: gens_of(runs[0], ac)[-1] for ac in sorted({e["aircraft"] for e in runs[0]["entries"]})}
    note, why = pick_note(runs, a.note)
    log(f"on-page note ({why}): {note}")
    summary["note"] = note
    params = {"mode": "compare", "preset": "last", "cam": "chase", "layout": "formation", "vref": "norm", "cy": "rerr",
              "defl": "8", "spacing": "40", "note": note, "planpresets": "1"}
    fid = a.fidelity
    title = f"{name}: Phase 3-{'B2a' if 'b2a' in fid else 'B1'} ({fid}) shape genes + FD nodal flex"
    if ab:
        params.update({"preset": f"ab:*:{ab_gens[-1]}", "spacing": "30"})
        title = f"A/B {runs[0]['id']} vs {runs[1]['id']}: Phase 3-B1 ({fid}), matching gens {ab_gens}"
    if seeds_ab:
        params.update({"preset": f"ab:*:{ab_gens[-1]}", "spacing": "30", "abseeds": "1",
                       "labels": ",".join(f"{k}={v}" for k, v in labels.items())})
        title = f"{len(runs) // 2}-seed A/B {' / '.join(labels.values())}: Phase 3-B1 ({fid}), gens {ab_gens}"
        summary["seeds_ab"] = {"labels": labels, "gens": ab_gens}
        log(f"multi-seed A/B: {labels}, gens {ab_gens}")
    pg = build(runs, page, a.max_mb, params, title, only_gens=ab_gens)
    timings["page_s"] = round(time.perf_counter() - t, 1)
    log(f"page {page}: {pg['raw_bytes'] / 1e6:.2f} MB raw, {pg['gzip_bytes'] / 1e6:.2f} MB gzip; hz {pg['hz']:g}, "
        f"wing node stride {pg['node_stride']}, gens {pg['gens']}" + (f", dropped {pg['dropped_gens']}" if pg['dropped_gens'] else ""))
    summary["page_info"] = pg
    final_by_ac = {ac: max(g for g in pg["gens"] if g in gens_of(runs[0], ac)) for ac in final_by_ac}

    # 4. screenshots
    shot_res = None
    if not a.no_shots:
        t = time.perf_counter()
        shot_res = (ab_shots(runs, page, a.shot_prefix or f"{name}_", max(pg["gens"])) if (ab or seeds_ab) else
                    shots(runs, page, a.shot_prefix or f"{name}_", final_by_ac, b2="b2" in a.fidelity))
        timings["shots_s"] = round(time.perf_counter() - t, 1)
        log(f"screenshots: ok={shot_res['ok']} {len(shot_res['pngs'])} png, console errors {len(shot_res['console_errors'])}, "
            f"blank {shot_res['blank']}, out of view {shot_res['out_of_view']}, under an overlay (warning) {shot_res['covered']}")
        summary["screenshots"] = shot_res

    # proof results
    proof_ok = True
    if procs:
        t = time.perf_counter()
        summary["replay_proof"] = []
        for R, gens, d, src, od, pr in procs:
            rc = pr.wait() if pr is not None else 0
            v = proof_verdict(od, rc)
            v.update({"run": R["id"], "gens": gens, "fd_dir": d, "fd_source": src, "out": od})
            summary["replay_proof"].append(v)
            proof_ok &= v["ok"]
            log(f"proof {R['id']}: {'EXACT' if v['ok'] else 'FAILED'}; {len(v.get('rows', []))} rows, "
                f"{len(v.get('files', []))} channel-checked files; problems {v.get('problems')}")
        timings["proof_wait_s"] = round(time.perf_counter() - t, 1)

    # 6. summary
    timings["total_s"] = round(time.perf_counter() - T0, 1)
    summary["timings"] = timings
    print("\n================ Phase 3-" + ("B2a" if "b2a" in a.fidelity else "B1") + " page summary ================")
    print(f"page   {page}")
    print(f"size   {pg['raw_bytes'] / 1e6:.2f} MB raw / {pg['gzip_bytes'] / 1e6:.2f} MB gzip (limit {a.max_mb} MB raw); "
          f"settings: {pg['hz']:g} Hz, wing node stride {pg['node_stride']}, modal + all-zero structure channels dropped, "
          f"structure 1e-4; gens {pg['gens']}" + (f" (dropped {pg['dropped_gens']})" if pg['dropped_gens'] else ""))
    print(f"note   on page: \"{note}\"")
    tab = []
    bounds = gene_bounds([v["fd_dir"] for v in summary.get("replay_proof", [])] +
                         [os.path.join(os.path.dirname(er_root()), "flight-dynamics")])
    for R in runs:
        print(f"\ncost ({R['id']}; lower = better)      gen0        final    change")
        for ac in sorted({e["aircraft"] for e in R["entries"]}):
            gl = gens_of(R, ac)
            f0 = next(e["fitness"] for e in R["entries"] if e["aircraft"] == ac and e["generation"] == gl[0])
            f1 = next(e["fitness"] for e in R["entries"] if e["aircraft"] == ac and e["generation"] == gl[-1])
            tab.append({"run": R["id"], "aircraft": ac, "g0": gl[0], "final": gl[-1], "cost_g0": f0, "cost_final": f1})
            print(f"  {ac:6s} g{gl[0]} -> g{gl[-1]:<3d}            {f0:.6f}  {f1:.6f}  {100 * (f1 - f0) / f0:+.1f}%")
        print(f"\nplanform final vs baseline ({R['id']})")
        for ac in sorted({e["aircraft"] for e in R["entries"]}):
            gl = gens_of(R, ac)
            s1 = pf_stats(planform_of(R, ac, gl[-1]))
            s0 = pf_stats(planform_of(R, ac, gl[0]))
            print(f"  {ac:6s} sweep_qc {s1['sweep_deg']:+.2f} deg (baseline {s1['sweep_base_deg']:+.2f}, g{gl[0]} {s0['sweep_deg']:+.2f})"
                  f"  taper {s1['taper']:.3f} (baseline {s1['taper_base']:.3f}, g{gl[0]} {s0['taper']:.3f})"
                  f"  twist tip {s1['twist_tip_deg']:+.2f} deg mid {s1['twist_mid_deg']:+.2f} (baseline 0)")
            print("         genes final " + " ".join(f"{k.replace('wing_', '')}={v}" for k, v in s1["genes"].items())
                  + "  (defaults: tapers 1, twists 0, sweep delta 0)")
            at = [k for k, v in s1["genes"].items() if k in bounds and v is not None
                  and min(abs(v - bounds[k][0]), abs(v - bounds[k][1])) <= 1e-6 * max(1.0, abs(bounds[k][1] - bounds[k][0]))]
            s1["genes_at_bounds"] = at
            if at:
                print("         AT BOUND: " + ", ".join(f"{k}={s1['genes'][k]} [{bounds[k][0]}, {bounds[k][1]}]" for k in at))
            b2g = {k: genes_of(R, ac, gl[-1]).get(k) for k in B2_DEFAULTS}
            b2g0 = {k: genes_of(R, ac, gl[0]).get(k) for k in B2_DEFAULTS}
            if any(v is not None for v in b2g.values()):
                print("         B2 genes final " + " ".join(f"{k.replace('wing_', '')}={v if v is None else round(v, 4)}"
                      f" (g{gl[0]} {b2g0[k] if b2g0[k] is None else round(b2g0[k], 4)})" for k, v in b2g.items())
                      + "  (defaults: dihedral 0, tc 1, camber 0)")
                at2 = [k for k, v in b2g.items() if v is not None and (k, ac) in bounds
                       and min(abs(v - bounds[(k, ac)][0]), abs(v - bounds[(k, ac)][1])) <= 1e-6]
                if at2:
                    print("         B2 AT BOUND: " + ", ".join(f"{k}={b2g[k]} {list(bounds[(k, ac)])}" for k in at2))
                s1["b2_genes"], s1["b2_genes_gen0"], s1["b2_genes_at_bounds"] = b2g, b2g0, at2
            summary.setdefault("planform", {})[f"{R['id']}:{ac}"] = {"final": s1, "gen0": s0}
    summary["cost_table"] = tab
    if ab:
        print(f"\nA/B cost ({tags[0]} vs {tags[1]}; lower = better)")
        abt = []
        for ac in sorted({e["aircraft"] for e in runs[0]["entries"]}):
            for g in common_gens(runs, ac):
                c = [next(e["fitness"] for e in R["entries"] if e["aircraft"] == ac and e["generation"] == g) for R in runs]
                abt.append({"aircraft": ac, "gen": g, tags[0]: c[0], tags[1]: c[1], "delta_pct": 100 * (c[1] - c[0]) / c[0]})
                print(f"  {ac:6s} g{g:<3d} {c[0]:.6f}  {c[1]:.6f}  {100 * (c[1] - c[0]) / c[0]:+.1f}%")
        summary["ab"]["cost_table"] = abt
    if seeds_ab:
        g = max(pg["gens"])
        print(f"\nmulti-seed A/B cost at g{g} (base vs tweaked per seed; lower = better)")
        st = []
        for ac in sorted({e["aircraft"] for e in runs[0]["entries"]}):
            ds = []
            for sd in sorted({v.rsplit("-", 1)[-1] for v in labels.values()}):
                pair = [R for R in runs if labels[R["id"]].endswith("-" + sd)]
                c = [next(e["fitness"] for e in R["entries"] if e["aircraft"] == ac and e["generation"] == g) for R in
                     sorted(pair, key=lambda R: labels[R["id"]] != "base-" + sd)]
                d = 100 * (c[1] - c[0]) / c[0]
                ds.append((c[0], c[1], d))
                st.append({"aircraft": ac, "seed": sd, "gen": g, "base": c[0], "tweaked": c[1], "delta_pct": d})
                print(f"  {ac:6s} {sd:3s} {c[0]:.6f}  {c[1]:.6f}  {d:+.1f}%")
            mb, mt = sum(x[0] for x in ds) / len(ds), sum(x[1] for x in ds) / len(ds)
            print(f"  {ac:6s} mean {mb:.6f}  {mt:.6f}  {100 * (mt - mb) / mb:+.1f}% (mean of per-seed deltas "
                  f"{sum(x[2] for x in ds) / len(ds):+.1f}%)")
            st.append({"aircraft": ac, "seed": "mean", "gen": g, "base": mb, "tweaked": mt, "delta_pct": 100 * (mt - mb) / mb})
        summary["seeds_ab"]["cost_table"] = st
    if shot_res:
        print("\nscreenshots" + ("" if shot_res["ok"] else "  ** FAILED **"))
        for p_ in shot_res["pngs"]:
            print("  " + p_)
        print(f"  console errors {len(shot_res['console_errors'])}, blank {shot_res['blank']}, out of view {shot_res['out_of_view']}")
    if procs:
        for v in summary["replay_proof"]:
            print(f"\nreplay proof {v['run']} (gens {v['gens']}, FD {v['fd_dir']} via {v['fd_source']}): "
                  + ("EXACT" if v["ok"] else "FAILED " + "; ".join(v.get("problems", [])[:6])))
            for r in v.get("rows", []):
                print(f"  {r['aircraft']:6s} g{r['gen']:<3d} logged {r['logged']!r:22} replayed {r['replayed']!r:22} "
                      f"max rel {r['max_rel_err']:.1e} pinned {r['pinned_match']}")
            for f in v.get("files", []):
                print(f"  {f['file']}: {f['bit_identical']}/{f['n_channels']} channels bit-identical, rows {f['rows']}, "
                      f"max|d| non-wing {f['max_abs_nonwing']:.3g}, structure ({f['n_structure_channels']} ch) "
                      f"{f['max_abs_structure']:.3g}, wing {f['max_abs_wing']:.3g}; failed non-wing {len(f['failed_nonwing'])}, "
                      f"wing {f['failed_wing']}")
    print("\ntimings " + json.dumps(timings))
    sp = os.path.join(a.out, f"{name}_build_summary.json")
    json.dump(summary, open(sp, "w"), indent=1, default=str)
    print(f"summary {sp}")
    if shot_res and not shot_res["ok"]:
        die(5, "screenshots failed (console errors / blank render / aircraft out of view); see " + shot_res["log"])
    if not proof_ok:
        die(6, "replay proof not exact")
    print("BUILD OK")


if __name__ == "__main__":
    main()
