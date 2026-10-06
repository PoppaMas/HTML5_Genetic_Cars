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
    replays/<run_id>/<replay_id>/           replay.py output (trajectories/, index.json, replay_manifest.json, viewer.html)
    examples/synthetic_softbody_test.json   SYNTHETIC soft-body test pattern (sinusoids, not a simulation) + _standalone.html
  replay.py             CLI wrapper -> sim_bridge.replay
  sim_bridge/           importable package: replay.py (tool), er_adapter.py (shim for ER's current artifacts),
                        recorder.py (read-only per-step recorder), trajdiff.py (channel-by-channel trajectory diff)
  tests/test_replay.py  replay tests (adapter exact, agreed-interface fixture, real ER interface once it ships)
  tests/fixtures/       er_interface/phase1-s1/{run.json,genomes.jsonl} + fake_evolution_eval.py (agreed contract)
  screenshots/          PNGs from tools/screenshots.py (bench_jets_*.png = jet bench, phase1_*.png = phase 1,
                        replay_*.png = replay proof viewer, softbody_synthetic_*.png = synthetic soft-body test)
  tools/screenshots.py  headless checks + screenshots (--bench for a standalone multi-aircraft file)
  tools/check_traj.py   read-only data sanity report for a run (fitness ordering, tracking, trim, quat/Euler, NaNs)
  tools/compare_traj.py channel-by-channel diff of two trajectory directories
  tools/make_synthetic_softbody.py  writes data/examples/synthetic_softbody_test.json
  tools/build/          esbuild script for viewer/dist (only needed if you change viewer JS)
```

## Regenerate the data

The sandbox venv already has jsbsim 1.3.1, numpy, and matplotlib:

```bash
cd flight_sim_3d/sim-bridge
PY=python   # any Python with flight_sim_3d/requirements.txt installed

# The project's committed example config: seed 1, pop 48, 40 generations, 3 scenarios, uniform crossover.
# This takes about 100 s on 8 idle cores and reproduces flight_sim/results/example/fitness_history.csv exactly.
$PY export_run.py --config ../../flight_sim/config.example.json \
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

To point at another clone of `flight_sim/`, use `--flight-sim-dir DIR` or set `FLIGHT_SIM_DIR`.

## View locally

```bash
cd flight_sim_3d/sim-bridge
python3 -m http.server 8000
# open http://localhost:8000/viewer/
```

The page opens the first run in `data/runs/runs.json` and shows the last generation in chase view.

To view Evolution Runner's output as well, serve the team root:

```bash
cd flight_sim_3d && python3 -m http.server 8000
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
cd flight_sim_3d/sim-bridge
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
cd flight_sim_3d/sim-bridge
E=../evolution/runs
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

## Replay: `replay.py` / `sim_bridge.replay`

Re-flies logged genomes of an Evolution Runner (ER) run with a per-step read-only recorder and writes viewer
trajectories. It also checks that the replay reproduces the logged cost, and it compares channel by channel
against any trajectories ER already wrote. Nothing is written into ER's tree: the default output is
`data/replays/<run_id>/<replay_id>/`, and the tool refuses an ER path unless you pass `--out`. Run it with the
sandbox venv (jsbsim + numpy) and `PYTHONDONTWRITEBYTECODE=1` so that importing ER's modules leaves no `__pycache__` in ER's tree.

```bash
cd flight_sim_3d/sim-bridge
PY=python   # any Python with flight_sim_3d/requirements.txt installed
export PYTHONDONTWRITEBYTECODE=1
$PY replay.py --run phase1-s1 --gens 0,9,19 --html --replay-id proof-g0.9.19        # the proof below
$PY replay.py --run phase1-s3 --gens 19 --elites                                     # best + elites of g19
$PY replay.py --run phase1-s2 --best-per-gen --aircraft f16 --scenario 0             # one scenario, every gen
$PY replay.py --run phase1-s1 --ids T38:g19:r0,c172x:g19:r1 --hz 60
# full syntax
$PY replay.py --run <run_id|dir> [--runs-root DIR] (--gens 0,9,19 | --best-per-gen | --ids a,b) [--elites]
              [--aircraft a,b] [--scenario all|<id>] [--fidelity rigid|reduced|full] [--hz 30] [--html]
              [--out DIR] [--replay-id ID] [--interface auto|er|adapter] [--eval-module evolution.eval|file.py]
              [--recorder-timing auto|pre|post] [--compare-traj DIR|none] [--workers N] [--tol 1e-6]
```

Output (`<out>/`):
* `trajectories/traj_<ac>_<run>_g<gen>[_r<rank>][_sc<k>].json` and `index.json`. Index entries carry `cost`,
  `fitness` (equal to cost), `scenario`, `individual_id`, `logged_cost` and `verdict`; the index carries
  `fitness_sense`.
* `replay_manifest.json` holds:
  - interface used (`er`/`adapter`) and the evaluate module;
  - `fitness_sense`;
  - per genome: logged vs replayed cost (mean and per scenario), relative error, fidelity, and logged vs replay
    `model_version`;
  - `trajectory_check` (per file: rows compared, bit-identical channels, max |diff| per channel, verdict);
  - provenance: the run's git sha, code_sha and jsbsim version, the replay's jsbsim and Python versions, argv;
  - start/finish timestamps and recorder timing.
* `viewer.html` (with `--html`): a standalone viewer built by `colab_viewer`. Labels show `r<rank>` and `sc<k>`,
  and the HUD shows the replay id, fidelity and model_version. Address a scenario with `gens=T38:19#sc1`.

Verdicts and exit codes:

| verdict | meaning |
|---|---|
| `match` | replayed cost is within `--tol` (relative) of the logged cost, at the row's fidelity |
| `MISMATCH` | outside tolerance with the same `model_version`; **exit 2** |
| `mismatch expected (model_version differs)` | outside tolerance, logged and replay model_version differ (both recorded); exit 0 |
| `not comparable (fidelity differs)` | `--fidelity` differs from the row's fidelity; exit 0 |
| `no logged cost` | e.g. non-best population members in ER's legacy checkpoints |

Evaluate errors (e.g. `--fidelity full` through the adapter) exit with code 3. Cost is the metric:
`fitness_sense` comes from run.json (`"min"`), and older files fall back to `fitness`/min.

### Interface: agreed with ER vs what is implemented now

| | agreed (ER will ship) | implemented / used today |
|---|---|---|
| run config | `runs/<run>/run.json` with `fitness_sense:"min"`, GA `seed`, `eval_seed` (scenario seed), aircraft, scenarios | **adapter** builds the same dict from `config.json` + `summary.json` (`eval_seed = scenario_seed = 1`) |
| genomes | `genomes.jsonl`, ALL individuals, `cost`, `per_scenario_cost`, `fidelity`, `model_version`, `is_best`, `is_elite`, optional `screen_cost`/`screen_fidelity` | adapter: best-per-gen rows from `checkpoints/<ac>.json` (`<ac>:g<N>:r0`, costs logged) + the final population of the last gen (`r1..`, no logged cost, `is_elite` = rank < elite) |
| evaluate | `evolution.eval.evaluate(genome, aircraft, scenario, run_cfg, recorder=None)` | `sim_bridge.er_adapter.evaluate`, same signature, wraps `evolution.sim.simulate` (rigid only) |
| recorder | `recorder(t, fdm)` after each step, `recorder(t, fdm, flex_state)` with flex; read-only | `TrajRecorder` handles both: `timing="post"` (agreed) fills each sampled row's controls from the next call, which reproduces ER's convention exactly. The adapter uses `timing="pre"` (proxy FDM calls before `run()`) |
| fidelity | `rigid|reduced|full`; `full` = v1 flex for now (`full(v1)`) | adapter: `rigid` only (others exit 3 with a message) |
| model_version | per row; exact match expected at any fidelity when equal | adapter: `evolution.sim@<config.provenance.code_sha>` (phase1 = `b46f27784c2131b4` = current code; bench_jets-j1 = `40bd11dda99e1fdf`, which still replays bit-exactly) |

`--interface auto` picks `er` when `run.json` + `genomes.jsonl` exist and `evolution.eval` imports; otherwise it
uses the adapter. `--interface er --eval-module path/to/file.py` targets any module with that contract.
`tests/fixtures/fake_evolution_eval.py` is such a module (ER's current sim with the agreed post-step callback), and
`tests/fixtures/er_interface/phase1-s1/` holds run.json + genomes.jsonl written by `er_adapter.export_interface`.

### What ER still needs to ship (missing today)
1. `run.json` and `genomes.jsonl` (all individuals with `cost`, `per_scenario_cost`, `fidelity`, `model_version`,
   `is_best`, `is_elite`). Today only best-of-gen costs are logged; non-best rows cannot be checked.
2. `evolution/eval.py` with `evaluate(...)` as agreed. It does not exist yet, so the real-interface path is
   **untested on ER's code**. `tests/test_replay.py::test_real_er_interface_when_available` runs automatically
   once the module and files exist.
3. A scenario helper (`evolution.eval.scenario_object(aircraft, scenario, run_cfg)` or `make_scenario`). The
   recorder needs the reference/command/rate at t for `target_*` channels. Today we rebuild ER's `Scenario`
   through `evolution.sim.make_scenarios`.
4. One `recorder(0.0, fdm)` call after trim, before the first step. Without it the t=0 row is missing and the
   origin falls back to JSBSim's `ic/*` properties. That fallback is bit-identical today, but it is an assumption.
5. A `flex_state` format: a dict `{'<component>.<dof>.<node>': value}` (or an object with `.channels()`), plus a
   `structure` attribute/key with the component axes (see the soft-body section below).
6. An explicit `model_version` string for `full(v1)`/v2. The adapter's `evolution.sim@code_sha` is a stand-in.

### Proof (adapter path, phase1-s1 g0/g9/g19 x 4 aircraft x 3 scenarios = 36 flights)
* Costs: 12/12 `match`. Logged and replayed costs are identical (relative error 0.0), e.g. f16 g19
  0.09009480210402053, c172x g0 0.33164794391949093.
* Trajectories vs ER's `runs/phase1-s1/trajectories/*.json` (best of gen, scenario 0): 12/12 files,
  2701/2701 rows, **29/29 channels bit-identical** (max |diff| 0 for every channel). Metadata (`fitness`,
  `scenario_cost`, `genome`, `frame`, `target`, `events`) is identical too.
* bench_jets-j1 g0/g19 (older code_sha): 8/8 costs exact; 8/8 files, 27/27 shared channels bit-identical.
* Agreed-interface fixture (post-step recorder, no t=0 call): 12/12 costs exact; 29/29 channels bit-identical on
  the 2700 shared rows (only the t=0 row is absent). With the t=0 call: 2701/2701 rows bit-identical.
* `tests/test_replay.py`: 6 passed, 1 skipped (the real ER interface, not shipped yet).

## Soft-body v2 (structure block): viewer groundwork

A trajectory may carry an optional `structure` block plus channels named `<component>.<dof>.<node_idx>`:

```json
"structure": {"components": [
  {"name": "wingR", "axis_nodes_body_m": [[x, y, z], ...], "dof": ["dz", "dy", "twist"]},
  ...]},
"channels": [..., "wingR.dz.0", ..., "wingR.twist.5", ...]
```

* Components: `wingL`, `wingR`, `htail`, `vtail`, `fuselage`. Nodes are given in body FRD metres, with the origin at
  the trajectory's reference point (CG).
* `dz` and `dy` are translations along body z (down) and y (right), in metres.
* `twist` is in radians: a right-hand rotation about the node0→nodeN axis direction. So `wingR` + means leading edge
  up, and `wingL` + means leading edge down. ER/FD should confirm this sign convention.
* The viewer projects every vertex of the component's procedural meshes (engines, pylons and struts go with their
  wing) onto the node polyline at rest. The displacement is interpolated linearly along it.
* The **Flex deflection** slider (URL `defl=`, 0–50×) exaggerates the display; it is shown only when a structure
  is loaded.
* The HUD shows true-scale tip values per component (`TIP wingR dz … dy … tw …°`). When `structure.synthetic`
  or the file's `synthetic` is set it adds a red "SYNTHETIC" line.
* Files without the block render exactly as before: no deformer, slider hidden. All bench/phase1/replay
  screenshot runs pass.
* `colab_viewer --slim` keeps structure channels, at 5 decimals.

**Plug-in point.**
1. v2 arrives as a new `model_version` of fidelity `full` through ER's `evolution.eval.evaluate`.
2. With flex active, evaluate calls `recorder(t, fdm, flex_state)`.
3. `sim_bridge.recorder.TrajRecorder` appends `flex_state`'s `<component>.<dof>.<node>` values as channels and
   copies `flex_state.structure` into the file's `structure` block.
4. ER maps FD's modal output onto those names.

The plumbing is tested end to end with the fixture: `FAKE_EVAL_SYNTH_FLEX=1` gives synthetic channels in the
replayed file, which the viewer then picks up. `--fidelity full` is rejected (exit 3) until ER's eval exists.

**Synthetic test file.** `data/examples/synthetic_softbody_test.json` carries `synthetic: true` everywhere. It is a
made-up straight 737-class cruise with sinusoidal bending and twist, written by
`tools/make_synthetic_softbody.py`. **It is not simulation output; do not quote numbers from it.**
`data/examples/synthetic_softbody_standalone.html` is the viewer for it. Screenshots: `screenshots/softbody_synthetic_*.png`.

## Trajectory format v1: `ga-flightsim-traj/1`

This is the format shared with Evolution Runner (`flight_sim_3d/evolution/trajectory.py`).
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
  `tests/test_replay.py`: 6 passed, 1 skipped (ER's real `evolution.eval` not shipped).
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
