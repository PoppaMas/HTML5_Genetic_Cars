#!/usr/bin/env python3
"""Headless screenshots + physics-mapping checks of the viewer.

    .venv-shots/bin/python tools/screenshots.py            # uses system google-chrome
    .venv-shots/bin/python tools/screenshots.py --chromium # uses playwright's bundled chromium

Serves the sim-bridge folder on a local port, opens the viewer, checks the console for
errors, verifies altitude / nose-vs-track / roll-sign mapping, and writes PNGs to screenshots/.
"""
import argparse
import functools
import http.server
import json
import math
import os
import sys
import threading

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "screenshots")


def serve(port, root=ROOT):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    handler = functools.partial(Quiet, directory=root)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--chromium", action="store_true")
    ap.add_argument("--run", default="seed1-pop48")
    ap.add_argument("--bench", metavar="HTML", help="only screenshot a multi-aircraft standalone file "
                    "(e.g. data/bench_jets-j1_standalone.html) and write screenshots/<prefix>*.png")
    ap.add_argument("--prefix", default="bench_jets_")
    ap.add_argument("--shots", choices=["bench", "phase1", "replay", "softbody"], default="bench",
                    help="shot list for --bench: 'bench' (bench_jets-j1) or 'phase1' (3 seeds + bench g19)")
    a = ap.parse_args()
    if a.bench:
        return bench(a)
    os.makedirs(OUT, exist_ok=True)
    httpd = serve(a.port)
    base = f"http://127.0.0.1:{a.port}/viewer/"
    logs, errors = [], []
    report = {}
    with sync_playwright() as p:
        kw = dict(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        if not a.chromium:
            kw["channel"] = "chrome"
        browser = p.chromium.launch(**kw)
        page = browser.new_page(viewport={"width": 1400, "height": 860})
        page.set_default_timeout(180000)  # software WebGL can be slow on a busy box
        page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: errors.append(str(e)))

        def wait_ready():
            page.wait_for_function("window.fv && window.fv.ready === true", timeout=180000)
            page.wait_for_timeout(600)

        # 1) single generation, chase cam, mid-flight (t = 20 s: climbing toward 4200 ft target)
        page.goto(base + "?gen=39&cam=chase&t=20")
        wait_ready()
        page.evaluate("fv.setTime(20)")
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(OUT, "single_chase_g039.png"))

        # physics checks on the loaded single trajectory
        checks = page.evaluate("""() => {
          const out = {altErrMax: 0, noseVelMaxDeg: 0, samples: 0};
          const d = fv.S.shown[0], tr = d.tr;
          for (let k = 0; k < tr.n; k += 15) {
            fv.setTime(tr.t[k]);
            const v = fv.vectors(0);
            const altScene = v.modelY / fv.S.exag + v.originAlt;
            out.altErrMax = Math.max(out.altErrMax, Math.abs(altScene - v.altLogged));
            const dot = v.nose[0]*v.vel[0] + v.nose[1]*v.vel[1] + v.nose[2]*v.vel[2];
            out.noseVelMaxDeg = Math.max(out.noseVelMaxDeg, Math.acos(Math.min(1, dot)) * 180 / Math.PI);
            out.samples++;
          }
          // roll sign: find the sample with the largest |phi| and check the right wing's vertical component
          let kmax = 0; for (let k = 0; k < tr.n; k++) if (Math.abs(tr.hud.phi[k]) > Math.abs(tr.hud.phi[kmax])) kmax = k;
          fv.setTime(tr.t[kmax]); const v = fv.vectors(0);
          out.maxPhiDeg = v.phiDeg; out.rightWingUp = v.right[1];
          out.rollSignOK = Math.sign(v.phiDeg) === -Math.sign(v.right[1]);
          // synthetic check through the same code path as file parsing (Euler fallback): phi=+30deg -> right wing down
          return out;
        }""")
        synth = page.evaluate("""async () => {
          const m = await import('fv/traj.js');
          const q = m.eulerToQuatEnu(30*Math.PI/180, 0, 0);
          const qt = m.Q_THREE_FROM_ENU.clone().multiply(q);
          const right = new fv.THREE.Vector3(0,1,0).applyQuaternion(qt);
          const q2 = m.Q_THREE_FROM_ENU.clone().multiply(m.eulerToQuatEnu(0, 10*Math.PI/180, Math.PI/2));
          const nose = new fv.THREE.Vector3(1,0,0).applyQuaternion(q2);
          return {phi30_right_wing_scene: right.toArray(), theta10_psi90_nose_scene: nose.toArray()};
        }""")
        report["single"] = checks
        report["synthetic"] = synth

        # extra single views: orbit + overview with model scale for context
        page.evaluate("fv.setTime(55)")
        page.evaluate("fv.setCam('orbit')")
        page.wait_for_timeout(300)
        page.screenshot(path=os.path.join(OUT, "single_orbit_g039.png"))

        # 2) compare mode: generations where the best fitness improved, side-by-side, chase on g000
        page.goto(base + "?mode=compare&gens=0,1,2,4,31,39&cam=chase&t=20&spacing=25")
        wait_ready()
        page.evaluate("fv.setTime(20)")
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(OUT, "compare_chase.png"))
        page.goto(base + "?mode=compare&gens=0,1,2,4,31,39&cam=free&t=25&spacing=25&exag=5&scale=20")
        wait_ready()
        page.evaluate("fv.setTime(25)")
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(OUT, "compare_overview_exag5.png"))
        report["compare_shown"] = page.evaluate("fv.S.shown.map(d => [d.tr.generation, d.color, d.tr.fitness])")

        # 3) gzip loading via DecompressionStream: temp .json.gz copy of the last generation
        import gzip, shutil, tempfile
        src_dir = os.path.join(ROOT, "data", "runs", a.run, "trajectories")
        idx = json.load(open(os.path.join(src_dir, "index.json")))
        last = idx["entries"][-1]
        tmp = tempfile.mkdtemp(prefix=".gztest-", dir=ROOT)
        try:
            with open(os.path.join(src_dir, last["file"]), "rb") as fi, gzip.open(os.path.join(tmp, last["file"] + ".gz"), "wb") as fo:
                fo.write(fi.read())
            json.dump({**idx, "entries": [{**last, "file": last["file"] + ".gz"}]}, open(os.path.join(tmp, "index.json"), "w"))
            page.goto(base + f"?index=../{os.path.basename(tmp)}/index.json&t=10")
            wait_ready()
            report["gzip_load"] = page.evaluate("[fv.S.shown[0].tr.n, fv.S.shown[0].entry.file, fv.S.shown[0].tr.fitness]")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        # 3b) aircraft swap: f16 trajectory (same gains, 350 KCAS) loaded directly via ?traj=
        f16 = os.path.join(ROOT, "data", "examples", "traj_f16_swaptest_g39.json")
        if os.path.isfile(f16):
            page.goto(base + "?traj=../data/examples/traj_f16_swaptest_g39.json&cam=orbit&t=52")
            wait_ready()
            page.evaluate("fv.setTime(52)")
            page.wait_for_timeout(300)
            page.screenshot(path=os.path.join(OUT, "aircraft_swap_f16_orbit.png"))
            report["f16"] = page.evaluate("[fv.S.shown[0].preset.key, fv.S.shown[0].tr.aircraft]")

        # 3c) Evolution Runner output (served from the team root): newest run whose index files all exist
        team = os.path.dirname(ROOT)
        runs_dir = os.path.join(team, "evolution", "runs")
        cand = []
        if os.path.isdir(runs_dir):
            for rid in os.listdir(runs_dir):
                ip = os.path.join(runs_dir, rid, "trajectories", "index.json")
                try:
                    ix = json.load(open(ip))
                    ents = ix["entries"] if isinstance(ix, dict) else ix
                    if ents and all(os.path.isfile(os.path.join(os.path.dirname(ip), e["file"])) for e in ents):
                        cand.append((os.path.getmtime(ip), rid, ents))
                except Exception:
                    pass
        if cand:
            _, rid, ents = max(cand)
            acs = sorted({e["aircraft"] for e in ents})
            pick = next((x for x in ("t6texan2", "f16") if x in acs), acs[0])
            ents = [e for e in ents if e["aircraft"] == pick]  # one aircraft: same scenario -> meaningful overlay
            team_httpd = serve(a.port + 1, team)
            gens = ",".join(f"{e['aircraft']}:{e['generation']}" for e in ents)
            page.goto(f"http://127.0.0.1:{a.port + 1}/sim-bridge/viewer/?index=/evolution/runs/{rid}/trajectories/index.json"
                      f"&mode=compare&gens={gens}&cam=chase&t=20&spacing=30")
            wait_ready()
            page.evaluate("fv.setTime(20)")
            page.wait_for_timeout(300)
            page.screenshot(path=os.path.join(OUT, "evolution_run_compare.png"))
            report["evolution_run"] = {"run_id": rid, "shown": page.evaluate(
                "fv.S.shown.map(d => [d.tr.aircraft, d.tr.generation, d.preset.key, +d.tr.fitness.toFixed(6)])")}
            team_httpd.shutdown()

        # 4) single-file standalone build (inline data, file:// URL) if present
        sa = os.path.join(ROOT, "data", f"c172x_{a.run}_standalone.html")
        if os.path.isfile(sa):
            page.goto("file://" + sa + "?mode=compare&cam=chase&t=30")
            wait_ready()
            page.evaluate("fv.setTime(30)")
            page.wait_for_timeout(300)
            page.screenshot(path=os.path.join(OUT, "standalone_inline_compare.png"))
            report["standalone"] = page.evaluate("fv.S.shown.map(d => [d.tr.generation, d.tr.n])")
        browser.close()
    httpd.shutdown()
    report["console"] = logs
    report["page_errors"] = errors
    print(json.dumps(report, indent=1))
    bad = [l for l in logs if l.startswith("[error]")] + errors
    ok = (not bad and report["single"]["altErrMax"] < 0.01 and report["single"]["noseVelMaxDeg"] < 10
          and report["single"]["rollSignOK"])
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(a):
    """Screenshots + checks for a multi-aircraft standalone build (file:// URL, inline data)."""
    os.makedirs(OUT, exist_ok=True)
    url = "file://" + os.path.abspath(a.bench)
    logs, errors, report = [], [], {}
    if a.shots == "phase1":
        shots = [
            ("g19_s1_formation_chase", "", 20),                                              # file defaults (cy=rerr)
            ("ramp_c172x_g0_vs_g19_alt", "?preset=pair:c172x&cy=alt&cam=orbit&spacing=25", 15),
            ("g19_s1_rerr_orbit", "?cam=orbit&cy=rerr", 9),
            ("g19_s1_nz", "?cy=nz&cam=chase", 8),
            ("g19_s1_kcas", "?cy=dkcas&cam=free&layout=true&vref=rel&scale=25&spacing=150&notes=1", 30),
            ("f16_seeds_s1_s2_s3_g19", "?preset=seeds:f16&cam=chase&cy=rerr&spacing=25", 12),
            ("before_after_737_j1_vs_s1_nz", "?preset=cmp:737&cam=chase&cy=nz&spacing=45", 8),
            ("before_after_all_j1_vs_s1", "?preset=cmp:*&cam=chase&cy=pct&spacing=40", 20),
        ]
    elif a.shots == "replay":
        shots = [
            ("proof_formation_g19", "", 20),                                                # file defaults (latest gen, sc0)
            ("proof_c172x_g0_g9_g19_rerr", "?gens=c172x:0,c172x:9,c172x:19&cy=rerr&cam=orbit&spacing=25", 15),
            ("proof_T38_g19_scenarios", "?gens=T38:19%23sc0,T38:19%23sc1,T38:19%23sc2&cy=rerr&cam=chase&spacing=30", 25),
            ("proof_single_f16_g19_hud", "?mode=single&gen=f16:19&cam=chase", 12),
        ]
    elif a.shots == "softbody":
        shots = [
            ("chase_defl1", "?mode=single&cam=chase&defl=1", 0.2),
            ("chase_defl5", "?mode=single&cam=chase&defl=5", 0.2),
            ("orbit_defl10_up", "?mode=single&cam=orbit&defl=10", 0.2),
            ("orbit_defl10_down", "?mode=single&cam=orbit&defl=10", 0.62),
        ]
    else:
        shots = [  # (name, query, time)
            ("g19_formation_chase", "", 20),                                                   # file defaults
            ("g19_overview_truepos", "?layout=true&cam=free&vref=rel&scale=25&spacing=150&notes=1", 30),
            ("g19_formation_orbit", "?cam=orbit&t=12", 12),
            ("f16_g0_vs_g19", "?preset=pair:f16&cam=chase&cy=err&spacing=25", 12),
            ("T38_g0_vs_g19", "?preset=pair:T38&cam=chase&cy=pct&spacing=25", 12),
        ]
    with sync_playwright() as p:
        kw = dict(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        if not a.chromium:
            kw["channel"] = "chrome"
        browser = p.chromium.launch(**kw)
        page = browser.new_page(viewport={"width": 1400, "height": 860})
        page.set_default_timeout(180000)
        page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: errors.append(str(e)))
        for name, q, t in shots:
            page.goto(url + q)
            page.wait_for_function("window.fv && window.fv.ready === true")
            page.wait_for_timeout(500)
            page.evaluate(f"fv.setTime({t})")
            page.wait_for_timeout(400)
            path = os.path.join(OUT, f"{a.prefix}{name}.png")
            page.screenshot(path=path)
            info = page.evaluate("""() => ({
              shown: fv.S.shown.map(d => [d.tr.aircraft, d.entry.run, d.tr.generation, d.preset.key, +(+d.tr.fitness).toFixed(4), +d.alongScale.toFixed(3), d.tr.hasRamp]),
              hud: document.getElementById('hud').innerText.split(String.fromCharCode(10)).filter(l => /^(cost|fitness|replay|CMD|REF|TGT|KCAS|IAS|NZ|TIP|SYNTH|flex)/.test(l)),
              entries: fv.S.shown.map(d => [d.entry.individual, d.entry.scenario, d.entry.verdict]),
              deflRowHidden: document.getElementById('defl-row').hidden, defl: fv.S.defl,
              flex: fv.S.shown.map(d => d.deformer ? d.deformer.comps.map(c => {
                let mx = 0; for (const mm of c.meshes) { const a = mm.mesh.geometry.attributes.position.array; for (let k = 0; k < a.length; k++) mx = Math.max(mx, Math.abs(a[k] - mm.rest[k])); }
                return [c.name, c.meshes.length, +mx.toFixed(3)]; }) : null),
              layout: fv.S.layout, vref: fv.S.vref, cy: fv.S.cy, cam: fv.S.cam,
              altErrMax: Math.max(...fv.S.shown.map((d, k) => { const v = fv.vectors(k); return Math.abs(v.modelY / fv.S.exag + v.originAlt - v.altLogged); })),
              rollSignOK: fv.S.shown.every((d, k) => { const v = fv.vectors(k); return Math.abs(v.phiDeg) < 0.2 || Math.sign(v.phiDeg) === -Math.sign(v.right[1]); }),
              noseVelMaxDeg: fv.S.layout === 'formation' ? null : Math.max(...fv.S.shown.map((d, k) => { const v = fv.vectors(k); return Math.acos(Math.min(1, v.nose[0]*v.vel[0]+v.nose[1]*v.vel[1]+v.nose[2]*v.vel[2])) * 180 / Math.PI; })),
            })""")
            report[name] = {"png": path, **info}
        browser.close()
    report["console"] = logs
    report["page_errors"] = errors
    print(json.dumps(report, indent=1))
    bad = [l for l in logs if l.startswith("[error]")] + errors
    ok = not bad and all(r["altErrMax"] < 0.05 and r["rollSignOK"] for k, r in report.items() if isinstance(r, dict) and "png" in r)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
