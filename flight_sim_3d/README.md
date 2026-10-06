# flight_sim_3d: multi-aircraft 3D flight-sim GA, Phase 1

This folder takes the single-aircraft JSBSim altitude-hold prototype in `../flight_sim/` and scales it to
several aircraft. Four teams built it, and each folder has its own README and design notes:

| folder | owner | what it contains |
|---|---|---|
| `genome/` | Genome Architect | Genome schema, task presets (`phase1_v4` = the default, `phase1_v5` = opt-in), per-aircraft profiles (`aircraft_profiles/phase1_shared.json`), `sim_ext.py` (extended sim and heading hold), fitness, NSGA-II, an adapter that runs the original `evolve.py` |
| `evolution/` | Evolution Runner | Multi-aircraft batch GA runner (`python -m evolution.batch`): process pool, sqlite result cache, checkpoints with exact resume, run logs, trajectory export. Configs include `phase1.json` and `phase1_hdg.json` |
| `flight-dynamics/` | Flight Dynamics | Modal flex-wing soft-body model (`flexwing.py`) and its two-way coupling to JSBSim (`coupled_sim.py`). Also `jsbsim_root/` (patched copies of the c172x, T38, 737 and f16 aircraft XML) and FM-quality checks |
| `sim-bridge/` | Sim Bridge | three.js 3D replay viewer (`viewer/`, three.js r169 vendored), `replay.py` (re-flies genomes with trajectory logging), `colab_viewer.py` |

## What Phase 1 is

* **JSBSim 1.3.1 rigid-body simulation of 4 aircraft:** c172x, T38, 737 and f16. Each has its own trim point:
  c172x 100 kt / 4000 ft, T38 300 kt / 10000 ft, 737 250 kt / 10000 ft, f16 350 kt / 10000 ft. All fly gear up,
  with all engines driven and a per-aircraft throttle clamp (T38 and f16 at 0.5). The models load from
  `flight-dynamics/jsbsim_root`, where socket I/O is stripped.
* **GA-tuned autopilot:** altitude hold with a 200 ft step and a 600 fpm / 0.1 g ramped reference, climb-rate
  feed-forward and a comfort term. There are 6 PID genes (altitude and pitch loops). An opt-in heading hold adds
  `kp_hdg` and `ki_hdg` (8 genes); it is on for c172x/T38/737 and off for the f16. The GA runs pop 32 × 20
  generations over 3 scenarios with turbulence/gust/sensor noise, on GA seeds 1, 2 and 3.
  Results: `evolution/runs/phase1_report.md` (mean best cost: c172x 0.213, T38 0.093, 737 0.111, f16 0.089) and
  the heading-hold table in `evolution/README.md`.
* **Optional modal flex-wing soft-body model** (`flight-dynamics/`, INTERFACE v1): bending and torsion modes with
  two-way coupling to JSBSim through `external_reactions`. There are 4 evolvable structure genes, and flutter and
  divergence margins act as constraints (margin < 1.0 fails). Use it through `genome` (`--task phase1_flex`) or
  `evolution` (`--fidelity reduced|full --struct-genes`, where `full` = FD flex v1 with 2 bending modes).
* **three.js 3D replay viewer** (`sim-bridge/viewer/`): single, compare and formation views, chase and orbit
  cameras, HUD and charts, and control-surface and wing-deflection animation, all driven by the shared
  `ga-flightsim-traj/1` trajectory files.

## Install

```bash
cd flight_sim_3d
python -m venv .venv && . .venv/bin/activate      # Python 3.11+ (tested with 3.13.5)
pip install -r requirements.txt                   # jsbsim==1.3.1, numpy, matplotlib, pytest
python flight-dynamics/link_jsbsim_data.py       # links jsbsim_root/engine + systems to the installed jsbsim data
```

Run the last step once per clone/venv. `flight-dynamics/jsbsim_root/` ships only the patched `aircraft/`
copies. Its `engine/` and `systems/` are links to the jsbsim package data, so they depend on the machine and
are not committed. Without them, the Phase 1 configs (which load every aircraft from `jsbsim_root`) fail to
load the engines.

The code finds the sibling folders through paths relative to the files themselves: `../flight_sim/`,
`flight-dynamics/` and `evolution/`. Environment variables (`FLIGHT_SIM_DIR`, `FLIGHT_DYNAMICS_DIR`,
`EVOLUTION_FD_DIR`, `EVOLUTION_DIR`, `FLIGHT_SIM_TEAM_ROOT`) override these if you move a folder. Configs give
`aircraft_root` as `flight-dynamics/jsbsim_root`, relative to `flight_sim_3d/`.

## Run the tests

```bash
cd flight_sim_3d
python -m pytest -q evolution/tests                                    # 64 tests, ~2 min
(cd genome && python -m pytest -q)                                     # 102 tests, ~30 s
(cd flight-dynamics && python -m pytest -q test_flexwing.py)           # 23 tests, ~10 s
(cd sim-bridge && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_replay.py)   # 2 pass, 5 skip (see caveats)
```

## Reproduce the Phase 1 runs (3 seeds)

Run from `flight_sim_3d/` so that `evolution` is importable. Each batch flies all 4 aircraft and takes about
2.5 min on 8 cores.

```bash
cd flight_sim_3d
python -m evolution.batch --config evolution/configs/phase1_smoke.json            # ~15 s smoke test (pop 8 x 3 gens, 4 aircraft)
for s in 1 2 3; do
  python -m evolution.batch --config evolution/configs/phase1.json     --seed $s --run-id phase1-repro-s$s
  python -m evolution.batch --config evolution/configs/phase1_hdg.json --seed $s --run-id phase1hdg-repro-s$s  # heading hold
done
python -m evolution.phase1_report --prefix phase1-repro          # seed-study tables, like evolution/runs/phase1_report.md
python -m evolution.validate_traj evolution/runs/phase1-repro-s1 # checks the exported trajectories
```

Always give a new `--run-id`. The shipped records `evolution/runs/phase1-s{1,2,3}` (and `phase1hdg-s*`) were
made with an earlier `code_sha`, so `--resume` on them refuses to run. The results still match bit for bit. A fresh seed-1 run from this layout
(`--run-id phase1-repro-s1`) reproduced `phase1-s1` exactly: c172x 0.22449138144713754, T38 0.0917745811719954,
737 0.11024576983937538, f16 0.09009480210402053, with the same gains and the same 5742 sims / 903 cache hits.
The genome-side equivalent through the original `evolve.py`:
`cd genome && python run_evolve.py --aircraft t38 -- --pop-size 32 --generations 20 --seed 1 --out runs/t38_s1`.

## Open the 3D viewer

New runs export best-of-generation trajectories (gens 0, middle and last) to `evolution/runs/<run>/trajectories/`.
Serve `flight_sim_3d/` and open the viewer on a run's index:

```bash
cd flight_sim_3d && python -m http.server 8000
# http://localhost:8000/sim-bridge/viewer/?index=/evolution/runs/phase1-repro-s1/trajectories/index.json
# compare/formation:  ...&mode=compare      single file: ?traj=<url to traj_*.json>
```

* `sim-bridge/replay.py` re-flies logged genomes with full logging. For now it only works on runs **without**
  `run.json`/`genomes.jsonl` (the legacy adapter path; see caveats). New runs already contain the exported
  best-of-generation trajectories, so the viewer does not need it.
* Colab or a single self-contained HTML file: `sim-bridge/colab_viewer.py`, which uses the prebuilt bundle
  `viewer/dist/fv.bundle.js`. Rebuild it with `cd sim-bridge/tools/build && npm ci && node bundle.mjs`
  (esbuild 0.24.0).
* The vendored libraries in `sim-bridge/viewer/vendor/` are three.js r169 (`three.module.min.js`) plus the addons
  OrbitControls, GLTFLoader, BufferGeometryUtils and the Line2 family, under the MIT license (`THREE_LICENSE`).
  If they are missing, get them from https://unpkg.com/three@0.169.0/build/three.module.min.js and
  https://unpkg.com/three@0.169.0/examples/jsm/... (same relative paths under `vendor/addons/`).

## Known caveats

* **Structural data are notional.** Stiffness, frequencies, mass and the elastic-axis/CG positions in
  `flexwing.py` are plausible values chosen per aircraft, not GVT or manufacturer data. Treat flutter and
  divergence margins as relative, for ranking genomes, not as certification numbers. The flex wing is opt-in;
  every Phase 1 result above is rigid-body.
* **The `phase1_v5` preset is opt-in.** It adds a hold-quality term and a downdraft scenario. `phase1_v4`
  (= `phase1_default`) is the frozen Phase 1 task, and v5 costs can't be compared with v4
  (`genome/HANDOFF_phase1_v5.md`).
* **Jets need their own trim points.** The prototype's c172x IC (100 kt / 4000 ft) does not trim the T38, 737
  or f16. Use the per-aircraft profiles (`phase1_shared.json` / `configs/phase1.json`) and give any new aircraft
  its own `h0_ft`, `speed_kts`, throttle clamp and gain bounds.
* **The f16 integral gain sits at its bound.** It often saturates `ki_alt` at zero. The opt-in profile
  `phase1_f16_kialt03` exists for that case (see `evolution/README.md`).
* **Not included in this snapshot:**
  * Flight Dynamics' in-progress "INTERFACE v2" expanded soft-body model (`flexbody.py`, `flexeval.py`,
    `jsbsim_root_v2/`, v2 tests and results). Nothing in Phase 1 imports it.
  * Evolution Runner's in-progress fast-mode benchmark outputs. `evolution/fidelity.py` and the
    `--viz`/`--fidelity`/`--multi-fidelity` flags are included and tested.
  * The result caches, standalone viewer HTML pages, screenshots and trajectory JSON dumps, because of their size.
  * The backfilled `run.json`/`genomes.jsonl` of the shipped runs. Regenerate them with
    `python -m evolution.runinfo --backfill evolution/runs/<run>`.
* **Open integration issue between Sim Bridge and Evolution Runner.** Current `evolution` writes `run.json` and
  `genomes.jsonl` for every run, with string scenario ids (`"c172x:s0"`). `sim-bridge/replay.py` still expects
  integer ids, so on such runs it stops with `ValueError: invalid literal for int()` (`sim_bridge/replay.py`, the
  `logged_scenario_cost` line). This affects every new run and the backfilled ones. The viewer itself is not
  affected.
* **5 Sim Bridge tests skip without the data they compare against.** 4 of them compare against ER's recorded
  trajectories, which are not shipped (~650 KB each). Copy the original `trajectories/` folders into
  `evolution/runs/phase1-s1/` and `evolution/runs/bench_jets-j1/` and those 4 pass. The fifth is the "real ER
  interface" test, which skips while no run has `run.json` + `genomes.jsonl`.
* `flight-dynamics/fm_quality.py` regenerates `FM_QUALITY*.md` and needs the external planning folder
  `$FLIGHT_SIM_PLAN_DIR`, which is not in this repo. The tables it produced are included.
