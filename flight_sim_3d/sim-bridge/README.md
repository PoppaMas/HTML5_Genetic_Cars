# sim-bridge: trajectory logging + 3D viewer for the flight-sim GA

This folder turns GA results from `flight_sim/` (branch `flight-sim-prototype` of
PoppaMas/HTML5_Genetic_Cars: JSBSim 1.3.1, a 6-gain altitude-hold PID evolved on `c172x`)
into **per-generation 6-DOF trajectory files** and plays them back in a **static three.js
viewer**. Nothing here is specific to the c172x. The aircraft name comes from the data,
JSBSim properties are generic, and the 3D model comes from a small config.

* `trajlog.py` flies one genome with the project's own `sim.simulate()`, called unchanged, and
  records the state at 30 Hz. It is non-invasive. It swaps `sim._new_fdm` for a
  read-only recording proxy around `jsbsim.FGFDMExec`, so the logged cost is bit-identical to the
  GA's cost.
* `export_run.py` runs the GA with the project's own `evolve.run()`, unchanged. It wraps
  `ga.next_generation` to copy each generation's best genome without touching the RNG stream.
  It then re-flies and logs the best individual of every generation, checks the fitness against
  the GA's `fitness_history.csv`, and writes `traj_*.json` files plus `index.json`.
* `viewer/` is a static page: vendored three.js r169, no build step, works offline.
* `colab_viewer.py` lets you use the viewer from Colab or Jupyter, either served or as one
  self-contained HTML file.
* `tools/screenshots.py` takes headless Chrome screenshots, checks for console errors, and
  verifies the physics mapping.

The sandbox clone is only imported. It is never modified. Nothing is pushed anywhere.

## Layout

```
sim-bridge/
  trajlog.py            logger (library + CLI)
  export_run.py         GA run -> best-of-each-generation trajectories + index.json
  colab_viewer.py       Colab/Jupyter helpers + single-file HTML builder (CLI too)
  viewer/
    index.html style.css
    js/main.js          scene, playback, camera, HUD, chart, compare mode
    js/traj.js          loading (json / gzip), schema parsing, interpolation, frame conversion
    js/aircraft.js      procedural low-poly aircraft presets (+ optional glTF)
    vendor/             three.module.min.js r169 + addons (OrbitControls, Line2, GLTFLoader)
    dist/fv.bundle.js   prebuilt single-file bundle (only for the inline/standalone build)
  data/
    runs/runs.json                          list of runs (viewer's "Run" dropdown)
    runs/seed1-pop48/ga_output/             evolve.py's own outputs (CSV, best_gains.json, plots)
    runs/seed1-pop48/trajectories/          traj_c172x_seed1-pop48_g{0..39}.json + index.json
    examples/traj_f16_swaptest_g39.json     same gains flown on JSBSim f16 at 350 KCAS (aircraft-swap demo)
    c172x_seed1-pop48_standalone.html       self-contained viewer + 8 generations @10 Hz (3 MB)
    bench_jets-j1_standalone.html           self-contained 4-aircraft bench (c172x/T38/737/f16, g0/g9/g19 @10 Hz, 2.5 MB)
    phase1_standalone.html                  phase-1 seeds s1/s2/s3 (36 files) + bench_jets-j1 g19 (4 files) @10 Hz, 6.6 MB
    phase2_pilot_standalone.html            phase2-pilot s1/s2/s3 full nodal soft-body, gen 0/29/59 × {c172x,T38,737}
                                            (27 files) @10 Hz, display structure slim, 37 MB
    phase2_pilot_s1_standalone.html         phase2-pilot-s1 only (ER's /2 re-export), gen 0/29/59, full node resolution
    phase3b1_smoke_planform_standalone.html REAL P3-B1 planform data: ER's phase3b1-smoke-s1, g0/2/4 x 3 aircraft
    planform_synthetic_standalone.html      SYNTHETIC P3-B1 planform loader test page (tests/fixtures/planform_synthetic)
    replays/<run_id>/<replay_id>/           replay.py output (trajectories/, index.json, replay_manifest.json, viewer.html);
                                            v2nodes-full-g19[-sc1] = real FD v2 full fidelity with nodal data
    examples/synthetic_softbody_test.json   SYNTHETIC soft-body test pattern (sinusoids, not a simulation) + _standalone.html
  replay.py             CLI wrapper -> sim_bridge.replay
  sim_bridge/           importable package: replay.py (tool; ER's real run.json/genomes.jsonl + evolution.eval),
                        er_adapter.py (fallback for legacy runs without run.json), recorder.py (read-only per-step
                        recorder), trajdiff.py (channel-by-channel trajectory diff), paths.py (repo-relative defaults
                        + env overrides), v2_map.py (FD v2 telemetry + FD nodes -> struct.* scalars + wing/tail/fin/
                        fuselage node channels; stdlib only, ER imports it read-only), fd_nodes.py (FD's exact nodal
                        values per frame from flex_state.eta, same FlexBodyModel ER evaluates)
  tests/test_replay.py  replay integration tests (real evolution.eval, adapter fallback, protocol fixture)
  tests/test_ids.py     unit tests: opaque string ids, mixed aircraft, elite/best selection, paths
  tests/test_v2_map.py  unit tests for v2_map (synthetic fixture in the documented FD v2 format)
  tests/test_v2_signs.py every sign convention vs flight-dynamics/v2_results/sign_probe.json + live FD mode probes
  tests/test_planform.py P3-B1 planform header (python + node check of the viewer loader), display structure slim,
                        wingR_modal twist-sign detection (pre-fix / post-fix ER files)
  tests/fixtures/       er_interface/phase1-s1/{run.json,genomes.jsonl} (copy of ER's real files, aircraft_root
                        made relative) + fake_evolution_eval.py (protocol edge cases) + v2_synthetic_raw.json (SYNTHETIC)
                        + planform_synthetic/ (SYNTHETIC planform headers on short real phase2-pilot-s1 g59 flights)
  patches/              replay-er-format.patch (all of this work relative to the phase-1 staging snapshot);
                        v2-nodes.patch (only the FD/ER-answers update, on top of the previous replay-er-format.patch)
  screenshots/          PNGs from tools/screenshots.py (bench_jets_*.png = jet bench, phase1_*.png = phase 1,
                        replay_*.png = replay proof viewer, softbody_synthetic_*.png = synthetic soft-body test,
                        v2_*.png = real FD v2 full-fidelity replay, v2nodes_*.png = FD nodal data: wing in-plane,
                        tail L/R, fin bending, phase2_pilot_*.png = phase2-pilot-s1 full soft-body,
                        phase2_pilot_seeds_*.png = s1/s2/s3 page, planform_phase3b1_smoke_*.png = real planform,
                        planform_synthetic_*.png = synthetic planform fixture)
  tools/build_phase2_pages.py  rebuilds data/phase2_pilot_standalone.html + phase2_pilot_s1_standalone.html
  tools/make_planform_fixture.py  writes tests/fixtures/planform_synthetic/ (FD planform_b1 decode, read-only import)
  tools/build_b1_page.sh / .py  one-command Phase 3-B1 page: validate, auto-fit build, screenshots, --replay-proof
  tools/build/jscheck.mjs  node check of the viewer's planform loader (used by tests/test_planform.py)
  tools/screenshots.py  headless checks + screenshots (--bench for a standalone multi-aircraft file)
  tools/check_traj.py   read-only data sanity report for a run (fitness ordering, tracking, trim, quat/Euler, NaNs)
  tools/compare_traj.py channel-by-channel diff of two trajectory directories
  tools/verify_fd_nodes.py  re-fly with FD's evaluate(record=True); compare its node export with a replay file
  tools/make_synthetic_softbody.py  writes data/examples/synthetic_softbody_test.json
  tools/build/          esbuild script for viewer/dist (only needed if you change viewer JS)
```

## Regenerate the data

The sandbox venv already has jsbsim 1.3.1, numpy, and matplotlib:

```bash
cd sim-bridge                      # from the team root (flight_sim_3d/ in the repo)
PY=python   # any Python with flight_sim_3d/requirements.txt installed

# The project's committed example config: seed 1, pop 48, 40 generations, 3 scenarios, uniform crossover.
# This takes about 100 s on 8 idle cores and reproduces flight_sim/results/example/fitness_history.csv exactly.
$PY export_run.py --config ${SIMBRIDGE_SANDBOX:-../../flight_sim}/config.example.json \
    --run-id seed1-pop48
```

* Any `evolve.py` flag can be passed through, for example `--pop-size 24 --generations 15 --scenarios 2 --seed 7`.
* `export_run.py` options:
  * `--run-id` (required)
  * `--out-root` (default `data/runs`)
  * `--aircraft` (default: the project's `sim.AIRCRAFT`)
  * `--record-scenario N`: which GA scenario to log. The default 0 is calm air. The fitness is
    always the GA's mean over all scenarios.
  * `--sample-hz` (default 30)
  * `--gzip`: writes `.json.gz`
  * `--relog`: re-logs an existing export without re-running the GA. Use this after changing the
    logger or schema.
* Exit code 0 means every generation's re-flown fitness matched the GA. The CSV holds 6 decimals,
  so differences up to 5e-7 are allowed. The final best must match `best_gains.json` exactly.
* `data/runs/runs.json` is rewritten automatically.

Log a single genome:

```bash
$PY trajlog.py --best-gains data/runs/seed1-pop48/ga_output/best_gains.json --out /tmp/best.json
#   -> "fitness 0.185520 (GA reported 0.185520, diff 0)"
$PY trajlog.py --genome 0.88,0.34,0.36,0.55,0.13,0.44 --scenarios 3 --scenario-seed 1 --out /tmp/g.json
$PY trajlog.py --best-gains ... --aircraft f16 --speed-kts 350 --out /tmp/f16.json   # another JSBSim model
```

To point at another clone of `flight_sim/`, use `--flight-sim-dir DIR` or set `SIMBRIDGE_SANDBOX` (alias `FLIGHT_SIM_DIR`); the default is `<team root>/../flight_sim`.

## View locally

```bash
cd sim-bridge
python3 -m http.server 8000
# open http://localhost:8000/viewer/
```

The page opens the first run in `data/runs/runs.json` and shows the last generation in chase view.

To view Evolution Runner's output as well, serve the team root:

```bash
cd .. && python3 -m http.server 8000      # from sim-bridge/: the team root (flight_sim_3d/ in the repo)
# http://localhost:8000/sim-bridge/viewer/?index=/evolution/runs/<run_id>/trajectories/index.json
```

### URL parameters

All parameters are optional.

| param | meaning |
|---|---|
| `index=URL` | a run's `index.json`. Relative URLs resolve against the page. |
| `traj=URL[,URL]` | load trajectory file(s) directly, with no index. `.json` or `.json.gz` |
| `runs=URL` | runs list for the dropdown (default `../data/runs/runs.json`) |
| `mode=single\|compare` | |
| `gen=N` / `gens=0,4,39` | selected generation(s). Use `aircraft:gen` (e.g. `t6texan2:9`) for multi-aircraft indexes |
| `t=SEC`, `play=1`, `speed=0.25..8` | |
| `cam=chase\|orbit\|free` | |
| `exag=1\|2\|5\|10` | vertical exaggeration (display only; HUD and chart stay true) |
| `scale=1..30` | model scale (for overview shots) |
| `spacing=M` | side-by-side spacing in compare mode (0 = true overlay) |
| `focus=K` | compare-mode camera/HUD focus (-1 = formation) |
| `preset=last\|first\|pair:<ac>\|seeds:<ac>\|cmp:<ac>\|cmp:*` | compare selection: latest gen of every aircraft, gen 0 of every aircraft, first vs latest of one aircraft; multi-run pages also have `seeds:<ac>` (latest gen of <ac> from every run of the same family, e.g. phase1-s1/s2/s3) and `cmp:<ac>` (latest gen of <ac> from one run per family, e.g. bench_jets-j1 vs phase1-s1; `*` = all aircraft). Append `@run` to pick the run, e.g. `pair:f16@s2` |
| `run=<run id or short>` | multi-run pages: run used by the last/first/pair presets (short = suffix after the last `-`, e.g. `s2`, `j1`) |
| `gens=ac:gen@run,...` | explicit selection on multi-run pages, e.g. `gens=f16:19@s1,f16:19@j1` |
| `gens=ac:gen#sc1` / `#r2` / `#r2.sc1` | replay indexes (several scenarios / individuals per gen): pick scenario `sc<k>` and/or rank `r<k>`; an explicit `gens=` wins over a page-default `preset` |
| `defl=0..50` | soft-body files only: structural deflection exaggeration (display only; HUD tip values stay true scale) |
| `layout=true\|formation` | compare layout. `formation` = "lock to own track": each aircraft on a straight lane (spacing apart), distance flown scaled by mean speed / own speed so different trim speeds keep station, own heading drift removed |
| `vref=auto\|abs\|rel\|norm` | 3D vertical reference: MSL, relative to each aircraft's own trim altitude, or relative and scaled so every aircraft's altitude step looks the same height. `auto` = `rel` when the shown start altitudes differ by more than 30 m |
| `cy=auto\|alt\|rel\|err\|rerr\|pct\|nz\|kcas\|dkcas` | chart: altitude MSL, altitude vs own trim, error vs own step target (ft), error vs own ramp reference (ft), error as % of own step, load factor nz, KCAS, KCAS minus trim. `auto` = `rerr` (or `err` without ramp data) for mixed aircraft, else `alt` |
| `panel=0`, `chart=0`, `refgrid=1`, `notes=1` | UI toggles (`notes=1` opens the legend's layout notes) |

### Controls

* Use the panel to pick a run, enter an index or trajectory URL, or open files with the file
  picker. You can multi-select `index.json` together with its `traj_*.json(.gz)` files, or
  select loose trajectory files.
* **Single** mode shows one generation (◀/▶ step through them). **Compare** mode shows a
  checklist with colour swatches. The "improvements" button selects generations where the best
  fitness dropped; there are also "spread 5" and "none". All aircraft are time-synced. They fly
  side by side (spacing slider) or overlaid (spacing 0). The legend shows live altitude; click
  an entry to focus the camera and HUD on it, or pick "formation". Preset buttons: "latest gen,
  all aircraft", "gen 0 / first, all", and "<aircraft>: first vs latest" for each aircraft.
* Optional channels (by name; files without them work as before):
  * `nz` and `kcas`: the HUD shows `KCAS` and `NZ`, the legend shows nz, and there are chart modes `nz` / `kcas` / `dkcas`.
  * A ramped altitude reference: the viewer tells the target channels apart by shape, not name. A continuous one
    is the ramp reference and a piecewise-constant one is the step command. Evolution Runner's phase 1 logs
    `target_alt_m` = 600 fpm ramp reference and `target_cmd_alt_m` = step command (see the file's
    `target.reference`). When a ramp is present, the 3D dashed line and setpoint plane follow the ramp. The HUD
    shows `CMD` (step, with error) and `REF` (ramp, with error). In the `alt`/`rel` chart the ramp is dashed and the
    step dotted. `rerr` is the tracking error vs the ramp, and `err`/`pct` stay vs the step.
* Mixed-aircraft compare: **Layout** (true positions / formation locked to own track), **3D
  altitude** (absolute / relative to own trim / relative with steps normalised) and **Chart**
  (altitude / relative / error vs own target / error % of step). The legend shows each
  aircraft's trim point (KCAS/ft) plus live altitude, error and IAS; "ⓘ layout notes" lists the
  per-aircraft distance and height scale factors. The HUD always shows true values.
* Playback: play/pause (Space), timeline scrub, ←/→ to step ±1 s, speed 0.25x–8x. Click or drag
  on the chart to seek.
* Cameras: chase (behind the aircraft, heading-stabilised), orbit (OrbitControls that follow the
  aircraft), and free/overview. Press `c` to cycle.
* HUD shows:
  * time, altitude in ft and m MSL, target and error, AGL
  * IAS/TAS, vertical speed, ground speed
  * φ/θ/ψ
  * elevator/aileron/rudder/throttle bars
  * fitness
  * the latest event, and the reason if the run was terminated
* Scene:
  * full ghost path plus a bright flown-so-far trail
  * dashed target-altitude line along the path
  * translucent setpoint plane at the current target altitude
  * ground grid (200 m cells) at the start's ground elevation, with a drop line from the aircraft
  * optional faint start-altitude grid
* The altitude-vs-time chart covers all shown generations, with the target dashed and a cursor.

## Colab / Jupyter

Copy or upload this `sim-bridge/` folder into the runtime, e.g. unzip it to `/content/sim-bridge`.

**Served (full viewer, lazy-loads all generations):**

```python
%cd /content/sim-bridge
import colab_viewer
colab_viewer.show_served("data/runs/seed1-pop48/trajectories/index.json", port=8000, height=820)
# extra URL params pass through, e.g. mode="compare", gens="0,4,39", cam="orbit"
```

In Colab this calls `google.colab.output.serve_kernel_port_as_iframe(port, path="/viewer/?index=...")`.
Outside Colab it starts the same local server and embeds an `IFrame`.

**Inline (no server, single self-contained HTML in the output cell):**

```python
colab_viewer.show_inline("data/runs/seed1-pop48/trajectories/index.json",
                         gens="improvements", hz=10, params={"mode": "compare"})
# or write a file you can download or e-mail:
colab_viewer.build_standalone("data/runs/seed1-pop48/trajectories/index.json", "viewer.html", gens="0,4,39")
```

From the shell: `python3 colab_viewer.py <index.json> [<index2.json[@gens]> ...] --out viewer.html [--gens all|improvements|0,19] [--hz 10] [--slim] [--title T] [--param k=v]...`.
Several indexes are merged into one page. Each entry gets a `run` tag, and `path@0,19` overrides `--gens` for that run.
`--gens improvements` picks the improving generations per aircraft. `--slim` keeps only the
channels the viewer draws (the 19 required ones plus target/command, KCAS/KTAS, nz, AGL). It also rounds
values to display precision: cm for positions, 1e-5 for quaternions and angles. That is about 40% of the original size. `--param` values become the file's default URL parameters.
Inline mode embeds `viewer/dist/fv.bundle.js` (three.js + viewer, about 0.8 MB) and the selected
trajectories, decimated. A full 30 Hz file is about 0.8 MB per generation.

**Generating data in Colab:** clone the branch (as `flight_sim/playground.ipynb` does) and
`pip install jsbsim==1.3.1 numpy matplotlib`. Then run
`python export_run.py --flight-sim-dir /content/HTML5_Genetic_Cars/flight_sim --run-id myrun --pop-size 24 --generations 12 --scenarios 2`
and show it with `show_served("data/runs/myrun/trajectories/index.json")`.

## Jet bench: `bench_jets-j1` (c172x / T38 / 737 / f16)

`data/bench_jets-j1_standalone.html` is one self-contained file (2.8 MB; open it directly,
no server). It holds Evolution Runner's `evolution/runs/bench_jets-j1` best-of-generation
trajectories for gens 0, 9 and 19 of all four aircraft (12 files, 10 Hz, slim channels). It opens on
the **gen-19 best of all four in formation**: chase camera on the whole formation, legend with
trim points, HUD, and an error-vs-own-target chart. Each aircraft flies its own trim point,
so the defaults are set up for that:

* `layout=formation`: straight lanes, 40 m apart. Distance flown is scaled so the 100 kt
  Cessna and the 350 kt F-16 stay abreast (×2.72 … ×0.71, listed in the layout notes).
* `vref=norm`: the 3D altitude is relative to each aircraft's own trim altitude, scaled so the
  200/500/600/700 ft steps look the same height. The HUD and chart stay true.
* The chart is in `err` mode (ft vs own target). Switch to `pct` (% of own step) for a scale-free
  comparison, or to `alt` / `rel`.
* Use the preset buttons (or `?preset=pair:f16`) to compare **gen 0 vs gen 19 for one aircraft**.
  "gen 0 / first, all" shows the four gen-0 bests. You can also tick any combination in the list.
* The models are procedural, with no external assets. c172x is high wing with a propeller. T38 is
  a small low swept wing with a bubble canopy and two nozzles. f16 is a mid 40° swept wing with
  one nozzle. 737 is low wing with dihedral, a conventional tail, and two underwing engines on pylons.

Rebuild it (read-only on Evolution Runner's files):

```bash
cd sim-bridge
# only if viewer/js changed: (cd tools/build && node bundle.mjs)
python3 colab_viewer.py ../evolution/runs/bench_jets-j1/trajectories/index.json \
  --out data/bench_jets-j1_standalone.html --gens all --hz 10 --slim \
  --title "bench_jets-j1: c172x / T38 / 737 / f16" \
  --param mode=compare --param preset=last --param layout=formation --param vref=norm \
  --param spacing=40 --param cam=chase --param t=20
# screenshots + console/physics checks -> screenshots/bench_jets_*.png
.venv-shots/bin/python tools/screenshots.py --bench data/bench_jets-j1_standalone.html --prefix bench_jets_
# data sanity report (prints a table + flags; --json for machine-readable)
python3 tools/check_traj.py ../evolution/runs/bench_jets-j1
```

Screenshots:
* `bench_jets_g19_formation_chase.png`: the default view
* `bench_jets_g19_overview_truepos.png`: true positions, overview, 25× models
* `bench_jets_g19_formation_orbit.png`
* `bench_jets_f16_g0_vs_g19.png`
* `bench_jets_T38_g0_vs_g19.png`

| aircraft | trim (KCAS / ft) | step | fitness g0 → g9 → g19 | hold-phase max err g0 → g19 |
|---|---|---|---|---|
| c172x | 100 / 4000 | 200 ft | 0.3192 → 0.1970 → 0.1923 (−40%) | 21.1 → 5.2 ft |
| T38 | 300 / 10000 | 600 ft | 0.2036 → 0.1328 → 0.1315 (−35%) | 40.1 → 6.9 ft |
| 737 | 250 / 10000 | 500 ft | 0.3374 → 0.1925 → 0.1920 (−43%) | 20.8 → 2.9 ft |
| f16 | 350 / 10000 | 700 ft | 0.1944 → 0.1224 → 0.1217 (−37%) | 20.9 → 6.3 ft |

Fitness is a per-profile cost (each profile has its own `alt_err_scale` and step), so compare
generations within one aircraft, not across aircraft. Notes from `check_traj.py`: there are no NaNs or
terminations, quaternion and Euler agree (0.000°), and t=0 matches trim. g9 ≈ g19 (it converged early). The
737 g19 rides 0.20–1.63 g with up to about 4,900 fpm and 14° pitch: legal, but aggressive for a transport.
The c172x hits full throttle about 21% of the time, loses 13 kt in the zoom climb, and drifts +27° in heading (no heading
hold). The T38 sits at its 0.5 (MIL) throttle clamp about 14% of the time. Elevator activity *increased* g0→g19 for
the 737 and c172x: they trade effort for tracking.

## Phase 1: `phase1-s1/s2/s3` (+ bench_jets-j1 before/after)

`data/phase1_standalone.html` (6.6 MB, one file, open directly) contains Evolution Runner's three phase-1 seeds
(c172x / T38 / 737 / f16 at gens 0, 9, 19 = 36 files). It also has the gen-19 best of `bench_jets-j1` (4 files) for
before/after. Data is 10 Hz, slim channels. It opens on the **seed-1 gen-19 best of all four in formation** (same
layout as the jet bench) with the chart in `rerr` mode (tracking error vs the 600 fpm ramp reference). Presets
panel:

* **run** selector (`s1 = phase1-s1`, ... `j1 = bench_jets-j1`): drives the rows below.
* **all**: latest gen / gen 0 of every aircraft.
* **gen 0 vs last**: `pair:<ac>` for each aircraft.
* **seeds s1/s2/s3**: gen-19 best of one aircraft from all three seeds (seed robustness).
* **phase1 vs bench_jets**: gen-19 best of one aircraft from bench_jets-j1 and phase1-s1, side by side
  (before/after the fitness change), or `all` for all eight.

Remember that the phase-1 step is 200 ft for every aircraft (the bench used 200/600/500/700 ft). The fitness
definitions also differ (ramp reference, comfort term, alt_err_scale 100 ft everywhere), so fitness numbers are not
comparable between the two runs.

Rebuild (read-only on Evolution Runner's files):

```bash
cd sim-bridge
E=${SIMBRIDGE_RUNS_ROOT:-../evolution/runs}
# only if viewer/js changed: (cd tools/build && node bundle.mjs)
python3 colab_viewer.py $E/phase1-s1/trajectories/index.json $E/phase1-s2/trajectories/index.json \
  $E/phase1-s3/trajectories/index.json "$E/bench_jets-j1/trajectories/index.json@19" \
  --out data/phase1_standalone.html --gens all --hz 10 --slim --title "phase1 (seeds s1/s2/s3) + bench_jets-j1 g19" \
  --param mode=compare --param preset=last --param run=s1 --param layout=formation --param vref=norm \
  --param spacing=40 --param cam=chase --param t=20 --param cy=rerr
.venv-shots/bin/python tools/screenshots.py --bench data/phase1_standalone.html --prefix phase1_ --shots phase1
python3 tools/check_traj.py $E/phase1-s1 $E/phase1-s2 $E/phase1-s3 --baseline $E/bench_jets-j1 [--json out.json]
```

Screenshots:
* `phase1_g19_s1_formation_chase.png`: the default view
* `phase1_ramp_c172x_g0_vs_g19_alt.png`: altitude chart with the ramp dashed and the step dotted
* `phase1_g19_s1_rerr_orbit.png`
* `phase1_g19_s1_nz.png`
* `phase1_g19_s1_kcas.png`: true positions, KCAS vs trim
* `phase1_f16_seeds_s1_s2_s3_g19.png`
* `phase1_before_after_737_j1_vs_s1_nz.png`
* `phase1_before_after_all_j1_vs_s1.png`

| aircraft | g19 fitness s1/s2/s3 | peak nz (all seeds) | max ramp err ft s1/s2/s3 | max climb fpm s1/s2/s3 | bench_jets-j1 g19: climb fpm / nz |
|---|---|---|---|---|---|
| c172x | 0.2245 / 0.2083 / 0.2075 | 0.86–1.13 | 16.0 / 11.1 / 9.8 | 916 / 797 / 643 | 2343 / 0.45–1.80 |
| T38 | 0.0918 / 0.0938 / 0.0928 | 0.85–1.13 | 5.9 / 5.3 / 5.2 | 703 / 678 / 694 | 6929 / −0.15–2.47 |
| 737 | 0.1102 / 0.1124 / 0.1099 | 0.85–1.14 | 8.8 / 10.0 / 7.4 | 744 / 774 / 715 | 4890 / 0.20–1.63 |
| f16 | 0.0901 / 0.0888 / 0.0894 | 0.86–1.13 | 5.0 / 3.1 / 3.0 | 700 / 638 / 638 | 8446 / −0.48–2.44 |

`check_traj.py` findings:
* Evolution Runner's claims hold. Peak nz is 0.85–1.14 at gen 19. Jet throttle never reaches idle or its limit
  (0.0%). The c172x drifts +25° in heading in every seed, while the jets drift 0°.
* The nz band is mostly set by the 0.1 g ramp corners: even gen 0 sits at 0.83–1.16.
* Gen-19 fitness is consistent across seeds: CV 0.6–1.0% for the jets, 3.7% for the c172x. The gains are not: some
  pitch gains differ 20–60× between seeds, so the cost surface is flat in those directions. `ki_alt` evolves to 0 in
  most jet seeds (the T38 in all three).
* Gen 9 → 19 brings little: f16 −0.1 to −0.2%, and some T38/737 seeds below 1%.
* The c172x is the weak spot:
  * throttle at full 9–18% of the time;
  * climbs up to 916 fpm against the 600 fpm ramp;
  * seed 1 (the only seed with a real `ki_alt`, 0.026) has a lightly damped ±8 ft oscillation in the holds;
  * seed 1 doesn't settle within 5% of the step in segment 1.

## Paths and environment variables

No absolute paths are baked in. Every default is relative to the checkout, and each one can be overridden
(`sim_bridge/paths.py`). The layout is the same locally (`flight-sim-team/`) and in the repo (`flight_sim_3d/`):

| variable | meaning | default |
|---|---|---|
| `FLIGHT_SIM_TEAM_ROOT` | team / repo root (contains `evolution/`, `flight-dynamics/`, `sim-bridge/`) | the parent of `sim-bridge/` |
| `SIMBRIDGE_EVOLUTION_ROOT` (alias `EVOLUTION_DIR`) | Evolution Runner package dir (imported read-only) | `<team root>/evolution` |
| `SIMBRIDGE_RUNS_ROOT` | ER runs (`replay.py --run <id>` looks here) | `<evolution root>/runs` |
| `FLIGHT_DYNAMICS_DIR` | Flight Dynamics dir; a relative `flight-dynamics/...` root in run.json resolves here (other relative paths against the team root), and a foreign absolute `.../flight-dynamics/...` path is remapped here | `<team root>/flight-dynamics` |
| `SIMBRIDGE_SANDBOX` (alias `FLIGHT_SIM_DIR`) | prototype `flight_sim/` checkout used by `trajlog.py` / `export_run.py` | `<team root>/../flight_sim` |
| `SIMBRIDGE_DATA_DIR` | sim-bridge outputs (replays) | `sim-bridge/data` |
| `SIMBRIDGE_FD_MODEL_VERSIONS` | FD's published current model_versions (replay's `model_version_current`) | `<FD>/v2_results/model_versions_post_mass.json` |

The commands below assume `cd sim-bridge` from the team root (`flight_sim_3d/sim-bridge` in the repo) and
`PY=python`, meaning any Python that has jsbsim + numpy installed (ER's venv or `flight_sim_3d/requirements.txt`).

## Replay: `replay.py` / `sim_bridge.replay`

Replay re-flies logged genomes of an Evolution Runner (ER) run with a per-step, read-only recorder and writes
viewer trajectories. It checks that each replay reproduces the logged cost, and it compares the result channel by
channel against any trajectories ER already wrote.

Nothing is written into ER's tree. The default output is `$SIMBRIDGE_DATA_DIR/replays/<run_id>/<replay_id>/`, and
the tool refuses an ER path unless you pass `--out`. Run it with `PYTHONDONTWRITEBYTECODE=1` so that importing ER's
and FD's modules leaves no `__pycache__` in their trees.

```bash
cd sim-bridge                      # from the team root (flight_sim_3d/ in the repo)
PY=python                          # any Python with jsbsim + numpy (ER's venv / flight_sim_3d/requirements.txt)
export PYTHONDONTWRITEBYTECODE=1
$PY replay.py --run phase1-s1 --gens 0,9,19 --html             # best of g0/g9/g19, all scenarios, ER's evolution.eval
$PY replay.py --run phase1-s1 --best-per-gen                   # 80 genomes x 3 scenarios
$PY replay.py --run phase1v5-s1 --gens 0,19 --elites           # best + elites (ER logs every individual)
$PY replay.py --run phase1v5-s1 --ids T38:g19:r1,c172x:g3:r0 --scenario 2   # opaque ids, scenario position 2
$PY replay.py --run phase1v5-s1 --gens 19 --scenario 0 --fidelity full --html   # real FD v2 flight, FD nodal data
$PY replay.py --run phase2-smoke-s1 --gens 4                  # ER's full-fidelity run: costs exact at full
$PY replay.py --run bench_jets-j1 --gens 0,19                  # legacy run without run.json -> adapter fallback
# full syntax
$PY replay.py --run <run_id|dir> [--runs-root DIR] (--gens 0,9,19 | --best-per-gen | --ids a,b) [--elites]
              [--aircraft a,b] [--scenario all|<scenario id>|<position>] [--fidelity rigid|reduced|full] [--hz 30]
              [--html] [--out DIR] [--replay-id ID] [--interface auto|er|adapter] [--eval-module evolution.eval|file.py]
              [--recorder-timing auto|pre|post] [--compare-traj DIR|none] [--workers N] [--tol 1e-6] [--no-nodes]
```

### ER's real format (primary path, `--interface er`, auto-selected when `run.json` exists)
* `run.json` (`ga-flightsim-run/1`) provides `fitness_sense` (`"min"`), `fidelity`, `eval_seed`, and
  `model_version` per aircraft. `aircraft[]` carries `scenario_ids` (`"c172x:s0"`, ...), the resolved profile
  (with `aircraft_root`) and the genes. `scenarios[]` holds full entries (id, steps, ramp, wind, ...).
* `genomes.jsonl` is one row per individual:
  - identity: `individual_id`, `generation`, `aircraft`, `rank`, `is_best`, `is_elite`;
  - costs: `cost` (= `fitness`) and `per_scenario_cost`, aligned with the row's `scenario_ids`;
  - provenance: `fidelity`, `model_version`, `eval_seed` (backfilled runs), or `screen_cost` /
    `screen_fidelity` / `ladder_cost` (multi-fidelity).

  The backfilled phase1-s1..s3 rows are best-only, with ids like `c172x:g19:best`. Newer runs log every individual
  (`T38:g19:r0` … `r29`).
* `evolution.eval.evaluate(genome, aircraft, scenario_entry, run_cfg, recorder=rec)` flies it.
  `evolution.eval.scenario_object` supplies the targets for the `target_*` channels. ER calls the recorder once at
  t=0 after trim, then after every step, and finally `recorder.final(t_end, fdm)`. With flex active it passes
  `flex_state` too.
* **Ids are opaque strings everywhere.** Selection is exact string match, and a row's scenarios are its own
  `scenario_ids` (else its aircraft's). Logged per-scenario costs are looked up by **position** in that list.
  `scenario_index` (our field) is that position, which is also what ER writes in its trajectory files;
  `scenario_id` is the string. File names use a sanitised copy of the ids; the index and manifest keep the raw
  strings.
* The cost compared depends on the fidelity:
  - at the row's `fidelity`: `cost` / `per_scenario_cost` / `model_version`;
  - at `screen_fidelity`: `screen_cost` / `screen_per_scenario_cost` / `screen_model_version`;
  - else `ladder_cost[fid]`;
  - else "not comparable".
* A relative `aircraft_root` (ER's newer runs) resolves with `evolution.sim.abs_root` (else
  `paths.resolve_model_root`), and a foreign absolute path is remapped to `FLIGHT_DYNAMICS_DIR`. This happens in
  memory only, and the manifest records it in `path_notes`.
* `model_version`: see "model_version handling" below (logged / current / pinned).
* Full fidelity: the recorder adds FD's nodal structure (v2 section); `replay.node_status` per component is
  `fd_nodes` or `estimated`. Versus ER's own full-fidelity files, components with a different node layout (FD
  33-node wings vs ER's 9-node modal wings) are compared at coincident span fractions and reported under
  `trajectory_check.files[*].rediscretised_components` (informational); every other channel must be bit-identical.
* `fitness_sense` is normalised: anything starting with `max` is maximise, everything else is minimise (ER's
  trajectory files write `"minimize (GA cost, mean over scenarios)"`).
* **Adapter fallback** (`sim_bridge.er_adapter`) is used only for runs without `run.json`, such as bench_jets-j1
  (`config.json` + `checkpoints/`). It mirrors the real format: per-aircraft scenario ids `"<ac>:s<k>"`, best rows
  `<ac>:g<N>:r0`, and the final population `r1..`. It passes the recorder straight to ER's `sim.simulate` (post-step
  + t=0), or uses a pre-step proxy on older ER code. It supports rigid only; any other fidelity exits 3.

Output (`<out>/`):
* `trajectories/traj_<ac>_<run>_g<gen>[__<individual id>][__<scenario id>].json` plus `index.json`.
  - The individual id is added unless the file is the only (best) row of its generation. The scenario id is added
    unless the scenario is at position 0.
  - Index entries carry `cost`, `fitness`, `scenario` (= `scenario_id`), `scenario_index`, `individual_id`,
    `rank`, `is_best`, `is_elite`, `logged_cost` and `verdict`.
* `replay_manifest.json` (schema `sim-bridge-replay-manifest/2`; paths relative to `FLIGHT_SIM_TEAM_ROOT`) holds:
  - the interface and evaluate module, the fallback reason, and `path_notes`;
  - per genome: logged vs replayed cost (mean and per scenario id, with position), relative error, fidelity,
    logged vs replay `model_version`, and `model_version_match`;
  - `trajectory_check` (per file: rows compared, bit-identical channels, max |diff| per channel, verdict);
  - provenance and recorder timing.
* `viewer.html` (with `--html`). Labels show `r<rank>` for non-best rows and `sc<k>` for scenario positions.
  Address entries with `gens=T38:19#sc1`, `#r2`, `#best` or `#<full individual id>`.

| verdict | meaning |
|---|---|
| `match` | replayed cost within `--tol` (relative) of the logged cost at that fidelity. A differing `model_version` with an exact cost is still `match`; the manifest says `model_version_match: false` |
| `MISMATCH` | outside tolerance with the same `model_version`; **exit 2** (also for a channel mismatch vs ER's trajectories) |
| `mismatch expected (model_version differs)` | outside tolerance, model_version differs (both recorded); exit 0 |
| `not comparable (fidelity differs, nothing logged at it)` | `--fidelity` with no logged cost at that fidelity; exit 0 |
| `no logged cost` | e.g. non-best members of a legacy population |

Evaluate errors (e.g. `--fidelity full` through the adapter, FD root missing) exit with code 3.

### Proof (ER's real `evolution.eval`, 2026-10-06)
| run / selection | genomes x scenarios | costs | trajectories vs ER's files |
|---|---|---|---|
| phase1-s1 `--gens 0,9,19` | 12 x 3 | 12/12 exact (rel. err 0.0) | 12/12 files, 2701/2701 rows, 29/29 channels bit-identical |
| phase1-s1 `--best-per-gen` | 80 x 3 | 80/80 exact | 12/12 bit-identical |
| phase1v5-s1 `--best-per-gen` | 60 x 4 | 60/60 exact | 9/9 bit-identical |
| phase1v5-s1 `--gens 0,19 --elites` | 12 x 4 (best + r1) | 12/12 exact | 6/6 (best files) bit-identical |
| phase1-f16fd-s1 / phase1hdg-s1 `--best-per-gen` | 20 x 3 / 80 x 3 | all exact | 3/3 and 12/12 bit-identical |
| bench_jets-j1 (adapter, no run.json) `--gens 0,19` | 8 x 3 | 8/8 exact | 8/8, 27/27 channels bit-identical |
| phase2-smoke-s1 (ER full fidelity, FD v2 post-mass) `--gens 4` | 3 x 3 | 3/3 exact at full | 3/3 files, 61/61 channels bit-identical (+ wings compared at 9 coincident nodes) |
| phase1v5-ki05-s1 (team-relative roots) `--gens 19` | 3 x 4 | exact | bit-identical |
| phase1v5-s1 / phase1v5-ki05-s1 `--gens 19 --fidelity full --scenario 0` (and `--scenario 1`) | 3 x 1 | not logged at full; mv = FD current (`e9535bc7` / `2c73464b` / `11df8fe4`) | all five components `fd_nodes`; FD export cross-check max 5.0e-7 |

In phase1-s1 the logged `model_version` of c172x / T38 / 737 (e.g. `rigid:jsbsim1.3.1:01333e0c`) differs from the
current one (`…:e0a73fc9`), because FD's jsbsim_root changed after that run. The costs are still bit-exact. f16 and
the newer runs match.

### Answered by ER / FD (2026-10-06, implemented)
* ER: `scenario_index` = position in the aircraft's `scenario_ids`; trajectory headers carry `scenario_id`
  (`<ac>:s<i>`), which `trajdiff` now keys on. Individual ids are `<ac>:g<gen>:r<rank>` (`r0` = best; `:best` only
  in backfilled runs); ids stay opaque strings here. `fitness_sense` is exactly `"min"` (we still normalise by
  prefix for older files).
* ER: `aircraft_root` / `git.repo` are team-relative; replay resolves them with `evolution.sim.abs_root` (else
  `paths.resolve_model_root`), recorded in `path_notes`.
* ER: a stale rigid `model_version` with exact costs is legitimate (the hash covers the whole `aircraft/<model>/`
  folder): verdict `match`, `model_version_match: false`, as before.
* ER: rows not rescored at full keep the highest stage's cost / fidelity / mv; per-stage versions are in
  `ladder_model_version`; a missing `eval_seed` means run.json's. See the model_version rule below.
* FD: all sign conventions (table in the v2 section), independent htL, elastic tail twists, in-plane tip
  deflection, feedback axes, and node exports. INTERFACE_v2 §11 now documents the node mapping.

### model_version handling
* **Logged** (what the row was scored with), first hit wins: row `ladder_model_version[fid]` > row
  `model_version` (when the row's `fidelity` is the replay fidelity) > `screen_model_version` (screen fidelity) >
  aircraft `ladder_model_version[fid]` > run.json `model_version[ac]`. Recorded as `logged_model_version` +
  `logged_model_version_source`.
* **Current**: FD's published post-mass-fix strings, `flight-dynamics/v2_results/model_versions_post_mass.json`
  (override: `SIMBRIDGE_FD_MODEL_VERSIONS`). Each genome / trajectory records `fd_current_model_version` and
  `model_version_current` (replay mv == FD's published one); the manifest lists `model_versions.not_current`.
* **Pinned**: run.json `pin_model_version[ac][fid]` (phase2 runs) is reported as `pinned_model_version` /
  `pinned_model_version_match` (not flagged: a replay after an FD change legitimately differs).
* Trajectory comparison keys include the fidelity and mv (`trajdiff.doc_key`: aircraft, gen, scenario_id,
  individual, `fid:mv`; rigid = untagged), so a full-fidelity replay is never matched to a rigid file.

### Open questions for ER and FD (remaining, 2026-10-06)

ER (answered by NOTE_v2_map / STATUS_v2_map, 09:44 PT — implemented on our side):
- `ladder_model_version` stays `{fidelity: mv}` (confirmed).
- FlexState `/3` public API (`.fd_model`, `.nodes()`, `.node_layout()`, `.v2_geometry`) — recorder uses these.
- `evaluate(..., pin=)` raises on mismatch — replay passes run.json `pin_model_version`.
- Traj schema `/2` at full with FE `wingR`/`wingL` + `wing*_modal`; reduced keeps modal names.

Answered since (ER STATUS_P2_pilot, 2026-10-06 afternoon):
- phase2-pilot-s1 re-exported as `/2` + FlexState /3 (`reexported: "schema3_v2_map"`; old `/1` copies in
  `trajectories_traj1/`).
- `fd_to_structure_channels` now uses `name.startswith("wingR")` (wingR_modal twist sign fixed).

Still open:
1. **No header marker for the wingR_modal twist fix.** Pre-fix and post-fix files have the same traj / flex-state
   schema, v2_map version, `git_sha` and `twist_doc`; only s1 has `reexported`. phase2-pilot-s2 (written 13:14 PT)
   is pre-fix, s1 (re-export) and s3 (14:12 PT) are post-fix. We detect it from the data (see "wingR_modal twist
   sign" below). Could ER add e.g. `structure.v2_map.modal_twist_sign_fixed: true` (or bump a doc version) to new
   exports?
2. Planform (P3-B1): ANSWERED by ER's phase3b1-smoke-s1 export (18:20 PT): top-level `planform`, `symmetric: true` +
   one `wing` block, `le_x_m` relative to the root quarter-chord point (+ forward), not absolute body x. The viewer only
   uses LE offsets between strips, so this works as is. Still open: ER's `structure.axis_nodes_body_m` follow the
   shaped sweep only (not the chord / AC shift), so FE nodes can sit slightly off the drawn chord-wise position.

FD:
1. Node coordinates have no dihedral and no HT height (documented in §11). Is a 3-D layout planned?
2. HT nodes are relative to the fusV tip (vertical). We also carry fusL + `vt_sideslip` into `htail.dy`: confirm.
3. Node frames start after the first coupler step (no t=0). Recorder fills t=0 from trim eta.
4. Reduced fidelity still uses v1 key names — only a few `struct.*` scalars map there.
4. Reduced fidelity (flexwing v1) has no node export and no tail / fuselage bodies: those files keep ER's modal
   wings only (no estimated tail components; `v2_map.available_components`). Its v1 key names
   (`root_bm_lbft_R`, `tip_twist_deg_R`, ...) differ from v2's, so only `struct.wing*_tip_dz` and
   `struct.elastic_dlift/dpitch/droll` are mapped there. Will reduced move to v2 key names?

## Phase 2 pilot: `phase2-pilot-s1/s2/s3` (full soft-body, FD nodal)

ER's phase-2 pilot (60 gens, rigid → full ladder, pinned post-mass model_versions; s1 c172x used rigid → reduced →
full). ER's trajectory exports (scenario 0, best of g0 / g29 / g59) are all `ga-flightsim-traj/2` +
`evolution-flex-state/3`: FE nodal wings (33 nodes, dz/dx/twist), `wing*_modal` (9 nodes), htail / vtail / fuselage,
`struct.*`. s1 was re-exported by ER (old `/1` in `trajectories_traj1/`).

```bash
# pages (read ER's files only): data/phase2_pilot_standalone.html (s1/s2/s3) + data/phase2_pilot_s1_standalone.html
PYTHONDONTWRITEBYTECODE=1 $PY tools/build_phase2_pages.py
# replay proof at the PINNED post-mass FD (FD's live tree moved to P2.5 at ~13:55 PT -> full hashes changed, pin=
# refuses); ER keeps the frozen post-mass FD sources in evolution/_fd_pin_post_mass (read-only use):
FDP=$(cd ../evolution/_fd_pin_post_mass && pwd)
for s in 2 3; do EVOLUTION_FD_DIR=$FDP FLIGHT_DYNAMICS_DIR=$FDP PYTHONDONTWRITEBYTECODE=1 $PY replay.py \
  --run phase2-pilot-s$s --gens 0,59 --fidelity full --replay-id phase2-pilot-s$s-full-g0.59 \
  --compare-traj ../evolution/runs/phase2-pilot-s$s/trajectories; done
.venv-shots/bin/python tools/screenshots.py --bench data/phase2_pilot_standalone.html --shots phase2_seeds \
  --prefix phase2_pilot_seeds_
```

Best cost per logged generation (= ER's files and genomes.jsonl `is_best`; s2 matches ER's reported 0.2155 /
0.0977 / 0.1241):

| aircraft | seed | g0 | g29 | g59 | g29→59 |
|---|---|---:|---:|---:|---:|
| c172x | s1 | 0.383367 | 0.236099 | 0.228694 | −3.1 % |
| c172x | s2 | 0.340998 | 0.218916 | 0.215512 | −1.6 % |
| c172x | s3 | 0.382846 | 0.210513 | 0.201961 | −4.1 % |
| T38 | s1 | 0.155968 | 0.099236 | 0.097582 | −1.7 % |
| T38 | s2 | 0.196627 | 0.100785 | 0.097710 | −3.1 % |
| T38 | s3 | 0.142558 | 0.097465 | 0.092095 | −5.5 % |
| 737 | s1 | 0.264272 | 0.128061 | 0.125939 | −1.7 % |
| 737 | s2 | 0.215333 | 0.129068 | 0.124128 | −3.8 % |
| 737 | s3 | 0.310691 | 0.124095 | 0.120378 | −3.0 % |

Page: 27 entries (3 seeds × 3 aircraft × g0/29/59), 37.2 MB. To stay < 40 MB it uses 10 Hz slim channels plus a
**display-only** structure slim (`colab_viewer._slim_structure`, `struct_slim={"node_stride": 2, "drop_modal": true,
"drop_zero": true, "decimals": 4}`): wings 33 → 17 nodes, htail 26 → 14, `wing*_modal` and all-zero channels
(wing dy) dropped, structure values at 0.1 mm / 1e-4 rad. No generation was dropped. Default: gen 59 of all three
aircraft from s1 (`preset=last@phase2-pilot-s1`), formation, flex ×8, chart = ramp error. Presets: run selector,
latest / mid / gen 0, gen 0 vs last per aircraft, gen 0/mid/last per aircraft (`evo:<ac>`), seeds s1/s2/s3 per
aircraft (`seeds:<ac>`). The s1-only page was rebuilt from ER's `/2` files (full node resolution, 27.3 MB).

Replay proof (2026-10-06 18:10 PT, `pin=` passed, pinned_mismatch []): s1 g0/29/59 9/9, s2 g0/59 6/6, s3 g0/59
6/6 costs exact (rel 0.0); 21/21 files 489/489 channels bit-identical to ER's `/2` files (FE wings included, no
rediscretisation now). s2's `wingR_modal.twist` is compared sign-corrected (pre-fix file, detected from the data).

Seed spread (g59 cost, range / mean): c172x 0.2020–0.2287 (12 %), T38 0.0921–0.0977 (6 %), 737 0.1204–0.1259
(4.5 %). Every best was still improving after g29 (last improvement g56–g59) but slowly (1.6–5.5 % from g29 to g59).
Genes at bounds (normalised < 0.01 or > 0.99): `wing_nsm_root/tip` at the 0.8 floor on 7 of 9 bests (all but s1/s2
c172x), `ki_alt` = 0 on T38 (3/3) and 737 (s1, s2), `ki_pitch` = 0 on T38 s3, `struct_damping_ratio` at the 0.05 ceiling on 737 s2/s3 and
T38 s1, `wing_gj_ratio_root` ceiling on c172x s2 and T38 s3, `wing_ei_taper_4` ceiling c172x s1. Scenario 0 at g59:
max |ramp error| 1.7–3.9 m, nz 0.85–1.13 g, wing tip dz −0.53…−0.41 m (737), −0.064…−0.043 m (T38), −0.104…−0.074 m
(c172x), tip |dx| ≤ 5.7 mm, tip twist +0.3…+1.0°.
Against phase 1 (phase1v5-s1..s3, rigid, 20 gens; different fitness setup, so costs are only roughly comparable):
the g-final ranges are similar (phase1v5 c172x 0.206–0.232, T38 0.088–0.111, 737 0.109–0.114); phase 2's
seed spread is smaller for T38/737; phase 1's bests piled on `ki_alt` = 0.05 ceiling, phase 2's on `ki_alt` = 0.

### FlexState /3 migration (recorder)
- Prefer `flex_state.nodes()`, `.v2_geometry`, `.node_layout()`, `.v2_map_version` (no private helpers on the hot path).
- `NodeSource.for_genome` kept as fallback for older FlexState; builds via public `fidelity.make_fd_model` when present.
- `evaluate(..., pin=)` passed from run.json `pin_model_version[ac][fid]` (raises on mismatch).
- Trajectory schema written: `ga-flightsim-traj/2` (viewer accepts `/1` and `/2`).

### wingR_modal twist sign (pre-fix / post-fix ER files)
ER's `fd_to_structure_channels` used `name == "wingR"` for the +twist sign; after the /3 rename to `wingR_modal`
that wrote the modal right-wing twist with the left-wing sign. ER fixed it (`startswith`). There is **no header
marker** (same schemas, v2_map version, git_sha, twist_doc), so the convention is decided **from the data**:
`trajdiff.modal_twist_sign` correlates the modal tip twist with the FE tip twist over the flight (|twist| ≥ 1e-4 rad,
|r| ≥ 0.9): +1 = post-fix, −1 = pre-fix, None = undecidable (no modal/FE channels or no twist).
- Recorder metric (`wing_modal_vs_nodal`): reports `modal_twist_sign`, `modal_twist_sign_corrected`, and undoes the
  sign only when −1 is detected (current ER: +1, no correction).
- trajdiff / replay `--compare-traj`: when replay and reference disagree, the reference's `wingR_modal.twist.*` is
  compared negated and the file reports `wingR_modal_twist_sign {a, b, compared_sign_corrected}`.
- Channels are always passed through unchanged; the viewer never renders `*_modal` (HUD shows FE tips).
- On disk: phase2-pilot-s2 = pre-fix (−1); s1 (re-export) and s3 = post-fix (+1).

### Planform loader (P3-B1, groundwork)
Optional header field (no schema bump; ignored when absent; `planform_b1` and placement inside `structure` accepted):
`planform: {schema: "fd-planform/1", source: "P3-B1", genes: {name: value}, symmetric: true + wing (ER's form) or
wingR / wingL: {span_frac[], y_m[], chord_m[], le_x_m[], twist_rad[]}, sweep_qc_rad}` — SI, body FRD, twist + = LE up,
one entry per FD strip (64 per semi-wing); `le_x_m` relative to the root quarter-chord point, + forward (ER also writes
chord_baseline_m, te_x_m, sweep_qc_baseline_rad, planform_baseline, area_norm, ac_shift_x_m; shown/ignored). Viewer: `traj.parsePlanform` → `aircraft.planformStations` reshapes the procedural wing
(chord, LE x offsets relative to the first strip anchored at the procedural root LE, built-in twist about the local
quarter chord; inboard of the first strip the first strip is held; extrapolated to the tip) and the HUD shows
`PLANFORM P3-B1 Λqc … taper … twist tip …` (+ `SYNTHETIC planform` warning), the compare legend a `Λ… λ…` tag.
`sim_bridge/planform.py`: `find_planform`, `validate_planform`, `summary`, `from_fd_strips` (FD feet / LE aft →
SI / FRD forward; `side_key="wing"` = ER's form, LE reference default 0 = relative like ER).
**Status: REAL data** — ER's `runs/phase3b1-smoke-s1` (fidelity `full_a1_b1`, written 18:20 PT, g0/2/4 × 737/T38/c172x,
65-node wings) carries the header; `data/phase3b1_smoke_planform_standalone.html` (12.6 MB; 10 Hz, display slim with
wing node stride 4, default g4 formation) shows it: g4 737 Λqc 22.7° (baseline 25°) taper 0.30 twist tip +0.2°,
T38 Λqc 23.6° (24°) taper 0.23, c172x Λqc −1.4° (0°) taper 0.77. `tests/test_planform.py` validates every
`phase3b1-*` file (python + node viewer check) when present and skips otherwise. The synthetic fixture stays for the
always-run tests. Fixture
`tests/fixtures/planform_synthetic/` (`tools/make_planform_fixture.py`): FD's own B1 decode for 737 (tapers 0.85,
twist −1/−4°, sweep +5° → Λqc 30°, taper 0.19) and c172x (`planform_b1`, symmetric), laid over short real
phase2-pilot-s1 g59 flights, all marked `synthetic: true`, plus a 737 baseline without the field.

### Phase 3-B1 page in one command (`tools/build_b1_page.sh`)
```bash
tools/build_b1_page.sh <run_id>                                   # validate + page + screenshots (~1.5 min on a 16x5 smoke)
tools/build_b1_page.sh <run_id> --replay-proof                    # + full replay proof against ER's frozen FD copy
tools/build_b1_page.sh <run_id> --replay-proof --fd-dir <frozen FD>   # explicit frozen copy
tools/build_b1_page.sh <run1>,<run2>,<run3>                       # several seeds -> one page with seed presets
```
1. Validation (exit 2 on any error): ER's `evolution.validate_traj`, `tools/check_traj.py` hard checks (status, NaN,
   quaternion, position vs velocity, index fitness; behavioural flags are info), and per file: `planform` fd-planform/1
   valid and not synthetic, `fidelity == full_a1_b1`, `model_version` == run.json pin, FD nodal wing structure present.
2. `data/<run_id>_standalone.html`: compare, final gen of every aircraft in formation, chase, flex x8, planform HUD line;
   presets gen 0 vs last, **planform top: gen 0 vs last** (`plan:<ac>`, new camera `top` = plan view fitted between the
   side panel and the HUD, spacing 1.3 x span), gen 0/mid/last, seeds (several runs). On-page banner (`note=` param,
   bottom centre): r1 data (`structure.node_layout` contains `node_layout_b1`) -> "r1: FE nodes follow FD's shaped layout
   (chord, sweep, AC shift; twist about the elastic axis). The 9-node wing modal display axes still use approximate
   geometry."; files without the tag -> "FD structure nodes follow evolved sweep only; chord/twist from planform strips".
3. Auto-fit to `--max-mb` (24): 10 Hz stride 2 -> 10 Hz stride 4 -> 5 Hz stride 4 -> drop middle gens (first and final
   kept); display-only slim (modal + all-zero structure channels dropped, structure 1e-4). Prints raw and gzip size.
4. Screenshots `screenshots/<run_id>_*.png` (`screenshots.py --shot-spec ... --strict`): default formation, planform top
   gen 0 vs final per aircraft, 737 flex close-up (rear, x8). Fails (exit 5) on console/page errors, a blank render
   (canvas luminance s.d. < 2 or < 6 colours) or a shown aircraft outside the view.
5. `--replay-proof` (started first, runs in parallel): `replay.py` at the rows' own fidelity with pin= (ER raises on a
   model_version mismatch) for every gen with an ER trajectory (`--proof-gens ends` = 0 + final). FD copy: `--fd-dir`,
   else an fd/pin dir key in run.json / config.json, else the newest `evolution/_fd_pin_*` whose model_versions (ER's
   `eval.describe`, probed in a subprocess) equal the pins; none -> exit 3 (live FD only with `--allow-live-fd`). Pass =
   cost rel_err 0.0 everywhere, pinned match, every non-wing channel bit-identical to ER's scenario-0 file (exit 6 otherwise).
6. Summary (stdout + `data/<run_id>_build_summary.json`): sizes and settings, cost gen 0 vs final, planform final vs
   baseline (sweep, taper, tip/mid twist, shape genes, genes at FD's b1_schema bounds), screenshots, proof, timings.

Dry run on `phase3b1-smoke-s1` (2026-10-06 ~18:55 MST): 89 s total with the proof (validate 9 s, page 5 s, screenshots
46 s, proof 28 s in parallel); page 19.3 MB raw / 1.6 MB gzip at 10 Hz stride 2, all gens 0/2/4; proof EXACT 9/9 rows
(rel 0.0, pinned), 9/9 files 681/681 channels bit-identical. FD replaced the r0 B1 code at 18:32 (r1 hashes); the smoke's
r0 pins have no ER frozen copy, so the dry run used a private reconstruction `/workspace/b1work/fd_r0` (FD tree + FD's own
`_scratch/p3b1/*.r0.py`), whose model_versions equal the smoke pins exactly. Without `--fd-dir` the proof fails loudly (exit 3).
Since ER switched to `flexbody_b1.node_layout_b1` (r1, ~18:49) the r0 smoke is no longer exactly replayable with current ER
code; that is expected.

**Real build, `phase3b1r1-smoke-s1`** (r1, 2026-10-06 ~19:40 MST): `tools/build_b1_page.sh phase3b1r1-smoke-s1
--replay-proof --shot-prefix phase3b1r1_` -> 82 s total (validate 8.6, page 4.8, screenshots 44.5, proof wait 23.9);
19.39 MB raw / 1.61 MB gzip, 10 Hz stride 2, gens 0/2/4. FD copy `evolution/_fd_pin_p3b1r1` via run.json `fd_dir`
(`md5sum -c v2_results/FROZEN_A1_B1r1.md5`: 13/13 OK). Proof EXACT: 9 rows x 3 scenarios rel 0.0, pinned; 9/9 files
681/681 channels bit-identical (non-wing, structure 587 ch, wing max|d| 0). T38 g4 T38:s0 = 0.08165462998244637 in a
fresh process.

**r1 wing geometry in the viewer.** When wingR/wingL carry `le_nodes_body_m` / `te_nodes_body_m` / `chord_m` /
`geometric_twist_rad` (FD r1), `parsePlanform` builds the wing from them (`geom: 'nodes'`): LE at absolute body x,
chord and built-in twist per node, twist pivot = the elastic axis (`axis_nodes_body_m`), not the quarter chord; elastic
twist rotates sections about the EA in the wing plane. Otherwise the planform strips are used as before. HUD taper/
sweep still come from the strips; the HUD line adds "FD r1 node geometry". Fuselage/engines stay procedural, so on the
737 the FD wing root LE (2.87 m) sits aft of the procedural one (1.67 m). `trajdiff` trusts
`structure.modal_twist_sign_fixed: true` (source `header`) instead of detecting the modal twist sign from data.

## Soft-body v2: FD v2 node mapping (`sim_bridge/v2_map.py`, version 2.0.0)

A trajectory may carry an optional `structure` block plus channels named `<component>.<dof>.<node_idx>`:

```json
"structure": {"schema": "sim-bridge-structure/2", "components": [
  {"name": "wingR", "axis_nodes_body_m": [[x, y, z], ...], "dof": ["dz", "dx", "twist"],
   "node_span_frac": [...], "estimated": false, "node_values": "FD nodal values (fd-flexbody-nodes/1) ..."},
  ...], "v2_map": {"version": "2.0.0", "sign_table": [...], "scalars": {...}, "estimated_components": []}},
"channels": [..., "wingR.dz.0", ..., "wingR.dx.32", ..., "htail.twist.25", ..., "struct.wingR_root_bm", ...]
```

Viewer conventions: body FRD metres, origin CG; `dz` + down, `dy` + right, `dx` + forward (new: wing in-plane);
`twist` rad, right-hand about node i → i+1 (wings root → tip, `htail` left tip → right tip, `vtail` root → tip).

**Nodal data replaces the assumed shapes.** At full fidelity the recorder gets FD's exact nodal values every frame
(`sim_bridge/fd_nodes.py`: the same `FlexBodyModel` ER evaluates, `flexbody.node_values(mdl, eta)` /
`node_layout`; identical to FD's `telemetry[i]['nodes']`), maps node coordinates into `axis_nodes_body_m`
(FRD m) and values into `<component>.<dof>.<i>`:

| component | nodes | DOFs | built from (FD bodies) |
|---|---|---|---|
| `wingR`, `wingL` | 33 | dz, dx, twist | wingR / wingL `w`, `v`, `theta` (replaces ER's 9-node modal wings) |
| `htail` | 26 (htL reversed + htR) | dz, dy, twist | htR / htL own `w`, `theta` (independent L/R) + fusV tip `w` + ht_incidence carry; fusL tip `w` + vt_sideslip carry (dy) |
| `vtail` | 13 | dy, dz, twist | vt own `w`, `theta` + fusL tip `w` + vt_sideslip carry; fusV tip `w` + ht_incidence carry (dz) |
| `fuselage` | 13 | dz, dy | fusV `w` (dz), fusL `w` (dy) |

Carry about `x_tail` (fusV tip x): `htail dz = −FT(w_own + w_fusV,tip) − inc·(x − x_tail)`,
`dy = FT·w_fusL,tip + vs·(x − x_tail)`; `vtail dy = FT(w_own + w_fusL,tip) + vs·(x − x_tail)`,
`dz = −FT·w_fusV,tip − inc·(x − x_tail)` (FT = 0.3048, inc = ht_incidence rad, vs = vt_sideslip rad).
The tail twist is elastic only (`ht_incidence` never enters `htail.twist`).

**Fallback (estimate)**: full-fidelity files without nodes (pre-node telemetry, `replay --no-nodes`) map tip
scalars onto assumed shapes φ = ξ²(3−ξ)/2 (bending, in-plane), ψ = ξ(2−ξ) (torsion); htL mirrors htR when
`htL_tip_w_ft` is absent. Such components carry `"estimated": true`, are listed in
`structure.v2_map.estimated_components`, `replay.node_status` = `estimated`, and the HUD says
`ESTIMATED (tip-only, no FD nodes): …` with an `est.` tag per tip line.

**Sign mapping (as implemented, `v2_map.SIGN_TABLE`; each row tested against `flight-dynamics/v2_results/sign_probe.json`
and live FD mode probes in `tests/test_v2_signs.py`)**

| FD quantity | Sim Bridge | + means |
|---|---|---|
| wing / HT / fusV `w_ft` (+ up) | `dz = −w·0.3048` | down |
| VT / fusL `w_ft` (+ toward body +y) | `dy = +w·0.3048` | right |
| wing `v_ft`, `wing*_tip_ip_ft` (+ aft) | `dx = −v·0.3048` | forward |
| wingR `theta` (+ LE up) | `wingR.twist = +θ` | LE up |
| wingL `theta` (+ LE up) | `wingL.twist = −θ` | LE down (right-hand about −y) |
| htR / htL `theta`, `ht_tip_twist_deg`, `htL_tip_twist_deg` (+ LE up) | `htail.twist = +θ` (elastic only) | LE up |
| vt `theta`, `vt_tip_twist_deg` (+ LE toward +y) | `vtail.twist = −θ` | LE toward −y |
| `ht_incidence_deg` (+ nose-up) | `struct.ht_incidence` (rad); carry `−inc·(x − x_tail)` in dz | nose-up |
| `vt_sideslip_deg` (= −w′, + fin LE toward +y) | `struct.vt_incidence = +rad`, `struct.vt_sideslip_equiv = −rad` (β sense); carry `+vs·(x − x_tail)` in dy | fin LE toward +y |
| root `*_bm` / `*_torque` / `*_ip_bm` (lbf·ft) | `struct.*_root_*` N·m, FD sign | + toward the body's + direction / nose-up about the EA / aft load |
| `dL`, `dY` (lbf), `dRoll`, `dPitch`, `dYaw` (lbf·ft) | `struct.elastic_dlift/dside` N, `struct.elastic_droll/dpitch/dyaw` N·m | dL up ⟂ V, dY body +y; l/m/n as JSBSim; at / about the AERORP |

All 32 scalar channels (26 diagnostics + 6 telemetry, plus `struct.vt_sideslip_equiv`) are documented with unit, +
direction, source key and factor in `v2_map.scalar_doc()` / `structure.v2_map.scalars`.

**Wing: FD nodal vs ER modal** (same flight, ER's 9 modal nodes vs FD's nodes at the same span fractions; 2701 frames,
phase1v5-s1 g19 s0; both wings ≈ equal):

| aircraft | max abs dz diff | max abs dz | max abs twist diff | max abs twist | in-plane dx (ER: none, dy = 0) |
|---|---|---|---|---|---|
| T38 | 1.71e-5 m | 0.0611 m (0.03 %) | 5.4e-6 rad | 7.07e-3 rad (0.08 %) | 7.7e-4 m |
| 737 | 1.60e-4 m | 0.536 m (0.03 %) | 1.37e-5 rad | 1.90e-2 rad (0.07 %) | 5.9e-3 m |
| c172x | 3.5e-5 m | 0.0958 m (0.04 %) | 7.9e-6 rad | 1.31e-2 rad (0.06 %) | 1.1e-3 m |

Against ER's own written phase2-smoke-s1 g4 trajectories (rounded to 1e-6): dz 1.5e-4 / 1.7e-5 / 3.6e-5 m, twist
1.3e-5 / 6e-6 / 8e-6 rad (737 / T38 / c172x). ER's linear strip interpolation is the source; the nodal values are
FD's exact FE nodes.

**Checks**: the last node equals the FD tip scalar every frame (max 1.6e-9 ft); the recorder's channels equal FD's
own `evaluate(record=True)` node export mapped by `v2_map` (`tools/verify_fd_nodes.py`, max 5.0e-7 = FD's 9-decimal
rounding, layout distance 0, cost bit-identical) for T38 / 737 / c172x on phase1v5-s1 and phase1v5-ki05-s1.

**Public API for ER** (`sim_bridge/v2_map.py`: stdlib only, imports nothing from `evolution` or `sim_bridge`; safe to
import read-only; `V2_MAP_VERSION = "2.0.0"`, bumped on any output change):
* `map_v2_record(telemetry_row, geometry=None, nodes=None, *, components=None, scalars=True) -> dict` — one frame:
  `telemetry_row` = FD raw keys (coupler `last`: DIAG_KEYS_V2 + TELEMETRY_KEYS_V2), `nodes` = `{body: {w_ft,
  theta_deg, v_ft}}` for that frame (`nodes_frame(values, k)` slices FD's export); returns `{channel: value}`
  (`struct.*` + `<component>.<dof>.<i>` for `components`, default all five; ER would pass
  `components=("htail", "vtail", "fuselage")`).
* `structure_block(geometry, *, components=None, estimated=None, base=None, synthetic=False, extra=None) -> dict` —
  the `structure` header (`base` = ER's wing components to keep).
* `geometry_from_layout(nodes_doc_or_layout, aircraft=None)` (FD node layout → geometry) and
  `geometry_estimated(aircraft, *, fdm=None, ...)` (fallback); `component_status(geo, nodes)`;
  `scalars_from_raw(raw)`, `scalar_doc()`, `validate_structure(structure, channels)`,
  `convert_traj(doc, nodes_doc=None, ...)`.
* Constants: `V2_MAP_VERSION`, `SCHEMA`, `STRUCTURE_SCHEMA`, `NODES_SCHEMA`, `COMPONENTS`, `FD_BODIES`, `DOFS`,
  `COMPONENT_DOFS`, `SCALARS`, `SIGN_TABLE`.
* `tests/test_v2_map.py::test_standalone_imports_nothing_from_evolution_or_sim_bridge` imports it in a clean subprocess.

Other entry points: `TrajRecorder(..., node_source=fd_nodes.NodeSource...)` (replay builds it at full fidelity),
`replay --no-nodes` (estimate, for comparison), `tools/verify_fd_nodes.py <traj.json>` (FD export cross-check).
Viewer: `defl=` exaggeration (URL accepts > 50), `dofs=` display filter (e.g. `dofs=dx`, `dofs=htail,vtail` or
`dofs=htail.dz`; HUD warns "display shows only …"). Screenshots: `screenshots/v2nodes_*.png`.
`tests/fixtures/v2_synthetic_raw.json` and `data/examples/synthetic_softbody_test.json` are SYNTHETIC.

## Trajectory format v1: `ga-flightsim-traj/1`

This is the format shared with Evolution Runner (`evolution/trajectory.py`; `flight_sim_3d/evolution/trajectory.py` in the repo).
All files here pass their `evolution/validate_traj.py`, including its independent quaternion
check, `R(q)·v_body == v_ENU`. One file holds one saved best individual:

* file name: `traj_<aircraft>_<run_id>_g<gen>.json` (or `.json.gz`)
* location: `<runs>/<run_id>/trajectories/`, next to an `index.json`

```jsonc
{
  "schema": "ga-flightsim-traj/1",
  "run_id": "seed1-pop48", "aircraft": "c172x", "jsbsim_version": "1.3.1",
  "git_sha": "5850721f...", "seed": 1, "generation": 39,
  "fitness": 0.18552,                      // GA fitness = mean cost over ALL GA scenarios (lower is better)
  "fitness_sense": "minimize (GA cost, mean over scenarios)",
  "scenario_index": 0, "scenario_cost": 0.17, "status": "ok",   // the scenario actually logged
  "genome": {"kp_alt": 0.26, ...},         // decoded gains {name: value}
  "frame": {
    "origin_lat_deg": 0.0, "origin_lon_deg": 0.0, "origin_alt_m": 1219.2,
    "axes": "ENU metres, x=east y=north z=up",
    "attitude": "quat body->ENU [w,x,y,z]",
    "body_axes": "JSBSim body FRD: x forward, y right wing, z down"
  },
  "units": {"phi": "rad", "theta": "rad", "psi": "rad", "controls": "norm -1..1, throttle 0..1"},
  "dt_s": 0.0333, "sim_dt_s": 0.00833, "sample_hz": 30,
  "target": {"alt_m": 1219.2, "steps": [{"t": 0, "alt_m": 1219.2}, {"t": 5, "alt_m": 1280.16}, {"t": 50, "alt_m": 1219.2}],
             "speed_kcas": 100.0},
  "events": [{"t": 0, "type": "start", "detail": "..."}, {"t": 5, "type": "target_change", "detail": "..."},
             {"t": 90, "type": "end", "detail": "completed"}],          // or type "terminated", detail = crash/stall/...
  "channels": ["t","x","y","z","qw","qx","qy","qz","vx","vy","vz","alt_msl_m",
               "phi","theta","psi","throttle","elevator","aileron","rudder", ...extras],
  "data": [[...one row per sample, same order as channels...], ...]
}
```

* **Required channels** (19): `t` [s]; `x,y,z` ENU [m] from the origin; `qw,qx,qy,qz`, the
  body(FRD)→ENU unit quaternion; `vx,vy,vz` ENU velocity [m/s]; `alt_msl_m`; `phi,theta,psi`
  [rad] (JSBSim Euler ZYX relative to local NED, for the HUD only); `throttle` 0..1;
  `elevator`, `aileron`, `rudder` (normalised -1..1 JSBSim `fcs/*-cmd-norm`).
* **Extras this exporter writes**:
  * `target_alt_m`, `kcas`, `ktas`, `nz`
  * `ub,vb,wb` (body velocity, m/s), `lat_deg,lon_deg`
  * `p,q,r` [rad/s], `alt_agl_m`, `alpha,beta` [rad]
  * `elevator_pos, aileron_pos, rudder_pos, throttle_pos` (surface/actuator `fcs/*-pos-norm`)

  Extras an aircraft doesn't provide are omitted, never null. Readers look channels up **by
  name** and ignore unknown channels and fields.
* **Other fields** this exporter adds (ignored by readers that don't know them): `channel_units`,
  `genome_normalized` (the raw [0,1] genes), `gain_units`, `scenario`, `per_scenario`,
  `control_props` (which JSBSim property fed each control channel), `wind`, `ga_config`,
  `ga_fitness_reported`.
* **Signs**:
  * φ>0 means the right wing is down, θ>0 means nose up, ψ is the true heading clockwise from north
  * elevator>0 = trailing edge down (nose down); aileron>0 = roll right; rudder sign is per model
* **Projection**: x/y use a local equirectangular projection, with WGS84 radii at the origin
  evaluated at the origin altitude (same as Evolution Runner). `z = alt_msl_m - origin_alt_m`.
* **Quaternions** are kept hemisphere-continuous (no sign flips between samples).

`index.json` (`ga-flightsim-traj-index/1`):

```jsonc
{"schema": "ga-flightsim-traj-index/1", "traj_schema": "ga-flightsim-traj/1", "run_id": "seed1-pop48",
 "entries": [{"generation": 0, "fitness": 0.3192, "aircraft": "c172x", "file": "traj_c172x_seed1-pop48_g0.json"}, ...],
 // extras written by export_run.py:
 "aircraft": "c172x", "seed": 1, "ga_config": {...}, "fitness_verification": {...}, "details": {"0": {...}}}
```

The viewer also accepts a bare list, or a list under `trajectories`, `files`, `items` or
`generations`. File paths resolve relative to the index. An index may mix aircraft. Evolution
Runner's runs do, and the viewer labels them `aircraft gN`.

### How the viewer maps the data

* Position: three.js uses `(x, y, z)_three = (east, up·exag, −north)`.
* Attitude comes from the quaternion. φ/θ/ψ are only used for the HUD, or as a fallback when a
  file has no quaternion. The model is built in body FRD axes, and
  `object.quaternion = q(ENU→three) ⊗ q(body→ENU)`, where `q(ENU→three)` is −90° about x.
  If `frame.body_axes` says FLU, a 180° x-flip is applied.
* Samples are interpolated linearly, with slerp for attitude. Times past a trajectory's end
  (e.g. a crash) hold the last state and are marked "ENDED".

### Cost vs fitness, replay and structure fields
* Index entries and files may carry `cost` (preferred) or `fitness`. The viewer reads `cost ?? fitness`. Ranking,
  improvement picks and HUD labels follow `fitness_sense` (index or file, `"min"` = lower is better, the
  default; `"max"` flips it). Every ER file so far is a cost.
* Replay files add `individual_id`, `scenario_index`, `scenario_cost` and a `replay` block (`replay_id`,
  `fidelity`, `model_version`, logged vs replayed scenario cost, recorder timing).
* Optional `structure` block plus `<component>.<dof>.<node>` channels: see "Soft-body v2" above.

## Adding another aircraft (e.g. a jet)

1. **Logging/export**: pass `--aircraft <jsbsim_model>` to `export_run.py` or `trajlog.py`. The
   logger only uses generic properties. However, the project's `sim.py` task trims at 100 KCAS
   and 4000 ft, which many models can't do: `f16`, `737` and `c310` fail to trim there.
   `trajlog.py --speed-kts` overrides the trim speed for one-off logs.
   `data/examples/traj_f16_swaptest_g39.json` is the c172x's best gains flown on the f16 at
   350 KCAS. A real jet GA needs its own scenario and gain ranges in `flight_sim/`, or
   Evolution Runner's profiles.
2. **3D model**: add a preset to `PRESETS` in `viewer/js/aircraft.js` and a regex to `ALIASES`.
   The fields are:
   * `span_m`, `length_m`, `fuselage_width_m`, `fuselage_height_m`
   * `wing_root_chord_m`, `wing_tip_chord_m`, `wing_pos` (high/mid/low), `wing_x_frac`
   * `sweep_deg`, `dihedral_deg`
   * `htail_*`, `vtail_*`
   * `propeller`, `nozzle`, `engines_underwing`

   Other fields: `nose_frac`, `cabin_frac`, `canopy` (`bubble`/`airliner`/cabin),
   `nozzles` (1/2), `engine_span_frac`, `engine_len_m`, `engine_dia_m`, `htail_dihedral_deg`.
   Presets exist for `c172x`, `t6texan2`, `t38`, `f16`, `737`, `generic-prop`, `generic-jet` and
   `generic-airliner`. Matching is case-insensitive (`T-38`, `F16` and `b737` all work). Unknown names fall back to `generic-prop`. A trajectory or
   index can also carry an `aircraft_model: {...}` object with the same keys, which overrides
   the preset. No code change is needed.
3. **glTF**: set `gltf: "models/x.glb"`, `gltf_rotation_deg: [x,y,z]` (maps the model's axes onto
   body FRD: +x nose, +y right wing, +z down) and `gltf_scale` in the preset or in
   `aircraft_model`. Control-surface animation only works on procedural models.

## Verification done

These checks were run with `tools/screenshots.py`. To run them yourself:

```bash
python3 -m venv .venv-shots && .venv-shots/bin/pip install playwright==1.48.0
.venv-shots/bin/python tools/screenshots.py              # system google-chrome
# or: .venv-shots/bin/playwright install chromium && .venv-shots/bin/python tools/screenshots.py --chromium
```

* **Fitness**: all 40 re-flown best-of-generation fitnesses match the GA's CSV. The max
  difference is 5e-7, which is the CSV rounding. The final best matches `best_gains.json`
  exactly (diff 0). The GA run itself reproduces the committed
  `flight_sim/results/example/fitness_history.csv` exactly: every column except wall time.
* **Cross-producer**: for gen 0 of the same seed and config, every shared channel is
  bit-identical to Evolution Runner's `traj_c172x_bench_baseline-20261006_g0.json`.
  `evolution/validate_traj.py` passes on all 41 files and the index.
* **Physics mapping in the browser**:
  * Model altitude matches logged `alt_msl_m` to within 2e-6 m.
  * Nose vs velocity track stays within 3.0° (that's angle of attack plus sideslip).
  * The roll sign is correct: at the max-|φ| sample, φ=+2.6° and the right wing points down.
  * Synthetic check: φ=+30° puts the right wing at (0.87, −0.5, 0) in scene coordinates.
    θ=10°, ψ=90° points the nose east and 10° up.
* **Loading paths**: gzip loading via DecompressionStream, the single-file build over file://,
  and Evolution Runner runs all work. There are no console errors.
* **Replay** (`--bench data/replays/phase1-s1/proof-g0.9.19/viewer.html --shots replay --prefix replay_`): 4 shots,
  no console/page errors; HUD shows `cost … (lower=better; scenario k: …)` and the replay/fidelity/model_version line.
* **Tests** (`PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -q -p no:cacheprovider tests/`): 73 passed, 0 skipped
  (test_ids 12, test_v2_map 15, test_v2_signs 9, test_replay 15, test_planform 16, test_build_b1 2; FlexState /3 + pin= path exercised;
  the replay ladder uses ER's frozen `_fd_pin_post_mass` FD copy when FD's live hashes have moved on).
* **Phase 2 seeds** (`--bench data/phase2_pilot_standalone.html --shots phase2_seeds --prefix phase2_pilot_seeds_`):
  8 shots, PASS, no console/page errors: default s1 g59 formation (flex ×8), c172x and 737 seed overlays at g59
  (rerr / nz), g0 vs g59 for c172x/T38/737 s1 and T38 s2, mid gen (g29) s3 formation.
* **Planform (real)** (`--bench data/phase3b1_smoke_planform_standalone.html --shots planform_real
  --prefix planform_phase3b1_smoke_`): 4 shots (737/T38/c172x g4 top, g4 formation), PASS, no console/page errors;
  HUD `PLANFORM P3-B1 Λqc 22.7° taper 0.30 twist tip 0.2°` (no SYNTHETIC warning); 737 wing box x −6.24…1.67 m.
* **Planform (synthetic)** (`--bench data/planform_synthetic_standalone.html --shots planform --prefix planform_synthetic_`):
  4 shots, PASS, no console/page errors; shaped 737 wing box x −8.32…1.67 m vs baseline −7.60…1.67 m (sweep 30°),
  HUD `PLANFORM P3-B1 Λqc 30.0° taper 0.19 twist tip −3.9°` + SYNTHETIC warning; baseline entry shows no line.
* **Phase 2 pilot** (`--bench data/phase2_pilot_s1_standalone.html --shots phase2 --prefix phase2_pilot_`): 8 shots,
  no console/page errors; gen 59 formation with FD nodal flex; g0 vs g59 per aircraft; mid+final c172x.
* **FD v2 nodal** (`--bench data/replays/phase1v5-s1/v2nodes-full-g19/viewer.html --shots v2nodes --prefix v2nodes_`
  and `--bench data/replays/phase1v5-s1/v2nodes-full-g19-sc1/viewer.html --shots v2nodes_gust --prefix v2nodes_`):
  10 shots, no console/page errors: wing in-plane bending (top view, `dofs=dx`), HT left/right difference in the
  gust scenario (`dofs=htail.dz,htail.twist`; c172x L 6.1 mm vs R 4.7 mm at t = 0.2 s), fin bending both ways
  (737 vt tip dy +0.115 m at t = 0.1 s, −0.028 m at t = 2.03 s), true-scale reference shots.
* **FD v2** (`--bench data/replays/phase1v5-s1/v2nodes-full-g19/viewer.html --shots v2 --prefix v2_`): 5 shots,
  PASS, now on the nodal data. All other suites (c172x live viewer, bench, phase1, replay, replay_v5, softbody)
  re-run with the new bundle swapped into their standalone pages: PASS.
* **Soft-body** (`--bench data/examples/synthetic_softbody_standalone.html --shots softbody --prefix softbody_synthetic_`):
  4 shots, no console/page errors; max vertex displacement scales with `defl` (wingL 0.91 / 4.55 / 9.10 m at ×1/×5/×10
  for a 0.89 m tip value). The bench, phase1 and c172x suites were re-run on the same bundle: PASS, slider hidden.
* **Phase 1** (`--bench ... --shots phase1`): 8 shots with no console or page errors. The altitude and roll
  checks pass. The HUD shows CMD/REF/KCAS/NZ for phase-1 files. Files without the new channels (the bench and the
  c172x run) render as before.
* **Jet bench** (`--bench`): 5 shots, with no console or page errors. Model altitude matches the logged
  altitude to within 2e-6 m with the normalised/relative layouts undone. The roll sign is correct. Evolution
  Runner's `validate_traj.py` passes on all 12 files and the index.

## Known limitations

* The trajectory is **one scenario**: calm air by default. `fitness` is the GA's mean over all
  scenarios. Use `--record-scenario` to log a gusty case. Each file carries the logged
  scenario's own cost too.
* Gens without an improvement (elitism) produce identical trajectories. They are still written,
  one per generation, as requested.
* The trajectories are about 0.8 MB per generation at 30 Hz, so the 40-gen demo is 34 MB.
  `--gzip` cuts that about 4x, and the viewer reads `.gz` natively. Inline mode decimates.
* Vertical exaggeration scales positions only. Drawn attitude stays true, so the nose won't
  follow an exaggerated climb.
* The ground plane is flat at the start's ground elevation (from `alt_agl_m`, else sea level).
  There is no terrain or curvature. Lat/lon origins that differ between files are aligned with
  a flat-earth offset, which is fine for nearby origins.
* The procedural models are schematic, and the rudder deflection sign is approximate
  (JSBSim's convention is per model).
* Headless screenshots use SwiftShader (software WebGL), which is slow to render but faithful.
* Mixing aircraft in one compare view works, but they fly different scenarios (different
  altitudes and speeds). Use `layout=formation`, `vref=rel|norm` and `cy=err|pct` for that (see the
  jet bench). The formation layout distorts distance (and so the climb angles), and `norm` distorts height. Both
  are display only.

## Later option: FlightGear live link (not built)

JSBSim can stream its state to FlightGear over its native-FDM UDP socket. Add an `<output
type="FLIGHTGEAR" port="5500" protocol="UDP" rate="30"/>` directive, or use
`fdm.set_output_directive(...)`, and run `fgfs --fdm=null --native-fdm=socket,in,30,,5500,udp`.
Replaying a logged genome through `trajlog.py` with that output enabled would show the same
flight in FlightGear's scenery. This was not implemented.
