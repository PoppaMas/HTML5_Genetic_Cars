# flight_sim_3d: multi-aircraft 3D flight-sim GA, Phases 1 and 2

This folder takes the single-aircraft JSBSim altitude-hold prototype in `../flight_sim/` and scales it to
several aircraft. Four teams built it, and each folder has its own README and design notes:

| folder | owner | what it contains |
|---|---|---|
| `genome/` | Genome Architect | Genome schema, task presets (`phase1_v4` = the default, `phase1_v5` = opt-in), per-aircraft profiles (`aircraft_profiles/phase1_shared.json`), `sim_ext.py` (extended sim and heading hold), fitness, NSGA-II, an adapter that runs the original `evolve.py` |
| `evolution/` | Evolution Runner | Multi-aircraft batch GA runner (`python -m evolution.batch`): process pool, sqlite result cache, checkpoints with exact resume, run logs, trajectory export. Configs include `phase1.json` and `phase1_hdg.json` |
| `flight-dynamics/` | Flight Dynamics | Modal flex-wing soft-body model v1 (`flexwing.py`, `coupled_sim.py`) and the Phase 2 flex-body model v2 (`flexbody.py`, `flexeval.py`, `INTERFACE_v2.md`). Also `jsbsim_root/` and `jsbsim_root_v2/` (patched copies of the c172x, T38, 737 and f16 aircraft XML) and FM-quality checks |
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
python flight-dynamics/link_jsbsim_data.py       # links jsbsim_root{,_v2}/engine + systems to the installed jsbsim data
```

Run the last step once per clone/venv. `flight-dynamics/jsbsim_root/` and `jsbsim_root_v2/` ship only the patched
`aircraft/` copies. Its `engine/` and `systems/` are links to the jsbsim package data, so they depend on the machine and
are not committed. Without them, the Phase 1 configs (which load every aircraft from `jsbsim_root`) fail to
load the engines (and `full` fidelity, which flies `jsbsim_root_v2`, fails the same way).

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
  * (Phase 1 only: FD's "INTERFACE v2" model was left out. Phase 2 ships it; see below.)
  * Evolution Runner's in-progress fast-mode benchmark outputs. `evolution/fidelity.py` and the
    `--viz`/`--fidelity`/`--multi-fidelity` flags are included and tested.
  * The result caches, standalone viewer HTML pages, screenshots and trajectory JSON dumps, because of their size.
  * The backfilled `run.json`/`genomes.jsonl` of the shipped runs. Regenerate them with
    `python -m evolution.runinfo --backfill evolution/runs/<run>`.
* **(Fixed in Phase 2.)** Phase 1's `sim-bridge/replay.py` stopped on Evolution's string scenario ids
  (`"c172x:s0"`). Phase 2's replay reads ER's `run.json` / `genomes.jsonl` format directly.
* **5 Sim Bridge tests skip without the data they compare against.** 4 of them compare against ER's recorded
  trajectories, which are not shipped (~650 KB each). Copy the original `trajectories/` folders into
  `evolution/runs/phase1-s1/` and `evolution/runs/bench_jets-j1/` and those 4 pass. The fifth is the "real ER
  interface" test, which skips while no run has `run.json` + `genomes.jsonl`.
* `flight-dynamics/fm_quality.py` regenerates `FM_QUALITY*.md` and needs the external planning folder
  `$FLIGHT_SIM_PLAN_DIR`, which is not in this repo. The tables it produced are included.

---

# Phase 2: flex-body v2, `phase2_flex` and rigid→full fidelity ladders

## What's new

* **Flight Dynamics: flex v2** (`flight-dynamics/flexbody.py`, `flexeval.py`, contract in `INTERFACE_v2.md`).
  * Spanwise wing EI/GJ/NSM genes (12 struct genes), flexible horizontal/vertical tails and fuselage.
  * Control-effectiveness, divergence, reversal and flutter margins, limit-load sizing with a minimum gauge
    (§12, the mass-exploit fix).
  * A `rigid` / `reduced` / `full` fidelity contract, where `full` flies `jsbsim_root_v2/`.
  * The telemetry **sign table** is in `INTERFACE_v2.md` §11; FD's static/mode probes are in `v2_results/sign_probe.json`.
  * **P2.5:** the `wing_nsm_root` / `wing_nsm_tip` floor is now 1.0 (was 0.8). A new outboard sizing term
    `J_wing_tip_bm_limit` (the tip BM allowable at η 0.875) was added. The model versions are in
    `v2_results/model_versions_post_p25.json` (regression: `v2_results/p25_regression.json`).
  * Validation and benchmarks: `v2_validation.json`, `v2_benchmarks.json`, `spearman.json` and `v2_results/`.
    FD's push list is `PUSH_MANIFEST_v2.md`.
* **Genome: the `phase2_flex` preset** (`genome/presets/phase2_flex.json`).
  * 8 controller genes (v4 with heading hold), plus FD's 12 v2 struct genes.
  * The v5 `ki_alt` upper bound of 0.5.
  * Struct genes are seeded around the FD baseline.
  * The cross-check against Evolution (bit-identical costs) is in `genome/CROSSCHECK_phase2.md`.
* **Evolution: Phase 2 wiring** (`evolution/README.md`, "Phase 2 wiring").
  * Fast mode (`evolution/README.md`, "Fast mode"): `--viz off` (no per-evaluation trajectory recording) and
    multi-fidelity ladders, `rigid→full` or `rigid→reduced→full`, with `min_full_frac` and per-aircraft overrides.
  * FD `model_version` pins: the run refuses to start or resume on a mismatch.
  * The phase2 configs.
  * The `v2_map` channel mapping for Sim Bridge (trajectory schema `ga-flightsim-traj/2`, `evolution-flex-state/3`).
  * Status notes: `evolution/analysis/STATUS_*.md`. The 3-seed table is `analysis/phase2_pilot_seeds.json`.
* **Sim Bridge.**
  * `replay.py` reads ER's `run.json` / `genomes.jsonl` format (string scenario ids).
  * The `v2_map` channels.
  * Nodal flex rendering of the FE wing/tail/fuselage nodes in the viewer (`viewer/js/aircraft.js`, rebuilt
    `viewer/dist/fv.bundle.js`).

## Run `phase2_flex` (rigid→full)

Run from `flight_sim_3d/`, after the install steps above, including `link_jsbsim_data.py`:

```bash
cd flight_sim_3d
# tiny smoke: pop 8 x 2 generations, rigid screen -> full re-score, P2.5 pins (~1 min for c172x on 8 cores)
python -m evolution.batch --config evolution/configs/phase2_flex_smoke.json --aircraft c172x --run-id p2-smoke-c172x
python -m evolution.batch --config evolution/configs/phase2_pilot_p25.json --model-versions   # read-only pin check
# the pilot: 64 x 60 on c172x/T38/737, rigid->full, >= 25 % + elites re-scored at full (~50 min on 8 cores)
python -m evolution.batch --config evolution/configs/phase2_pilot_p25.json --run-id p2-pilot-s1
python evolution/analysis/phase2_check.py evolution/runs/p2-pilot-s1          # structural / floor-pile checks
# the same task through the genome side (original evolve.py loop + FD flexeval, full fidelity only):
cd genome && python run_evolve.py --task phase2_flex -- --pop-size 16 --generations 3 --seed 1 --out runs/p2_c172x
```

The ladder is set in the config's `multi_fidelity` (`"screen": "rigid"`). From the CLI it is
`--fidelity full --multi-fidelity --screen rigid --struct-genes`. For `rigid→reduced→full`, use `"screen": ["rigid", "reduced"]`
(CLI `--screen rigid,reduced`).
The phase2 trajectories open in the 3D viewer like the Phase 1 ones (`?index=/evolution/runs/<run>/trajectories/index.json`).

## Results: Phase 2 pilot, 3 seeds

Each seed was 64 × 60 on c172x / T38 / 737, with rigid→full ladders (seed 1 used rigid→reduced→full for c172x).
The values are mean ± std (ddof 1) of the best cost at generation 59, from `evolution/analysis/phase2_pilot_seeds.json`.
Per-seed checks are in `analysis/phase2_pilot_s{1,2,3}_check.json`. The run records (config, run.json, summary,
sessions; no genomes/trajectories) are in `evolution/runs/phase2-pilot-s{1,2,3}/`.

| | c172x | T38 | 737 |
|---|---|---|---|
| best cost | **0.21539 ± 0.01337** | **0.09580 ± 0.00321** | **0.12348 ± 0.00284** |
| flutter margin (≥ 1.0 required) | ~1.22 | ~1.20 | ~1.20 |
| wing+structure mass Δ vs FD baseline | −5.5 % | −3.5 % | −3.5 % |

In all three runs the structural cost at the optimum is positive, and no stiffness gene piles at its floor (the §12
fix holds). `wing_nsm_*` piled at the old 0.8 floor on T38/737, which is what P2.5 addresses.

**These seeds ran on the pre-P2.5 pins.** P2.5 changed the `reduced`/`full` model versions
(`model_versions_post_mass.json` → `model_versions_post_p25.json`). Against the current `flight-dynamics/`,
`phase2_pilot.json` / `_s2` / `_s3` / `phase2_smoke.json` (post-mass pins) are refused by the pin check, and their
genomes (nsm < 1.0) no longer decode. New runs use the `*_p25.json` configs. The P2.5 smoke reference is
`evolution/runs/phase2-smoke-p25-s1` (`analysis/phase2_smoke_p25_s1_check.json`).

To re-fly the pinned seed runs exactly, point Evolution at the frozen pre-P2.5 FD copy shipped in
`evolution/_fd_pin_post_mass/` (see its README). Nothing changes when `EVOLUTION_FD_DIR` is unset.

```bash
cd flight_sim_3d
EVOLUTION_FD_DIR=evolution/_fd_pin_post_mass python -m evolution.batch --config evolution/configs/phase2_pilot_s2.json --model-versions
EVOLUTION_FD_DIR=evolution/_fd_pin_post_mass python -m evolution.batch --config evolution/configs/phase2_pilot_s2.json --run-id p2-pilot-repro-s2
```

## Phase 2 caveats

* **The structural data are notional.** Stiffness, mass, frequencies and allowables are plausible per-aircraft values,
  not GVT or manufacturer data. Margins and mass changes are for ranking genomes, not for certification.
* **The tails and fuselage are placeholders.** The HT, VT and fuselage beams are simple, so their sizing and
  stiffness terms are coarse.
* **There is no unsteady or transonic aero.** The aero is quasi-steady strip theory and the flutter margin is a
  quasi-steady estimate. The T38 and 737 are not corrected for compressibility.
* **`reduced`** is the v1 wing on the projected v2 genome. Its ranking agreement with `full` is only middling on
  the jets (Spearman 0.36 / 0.42 in the ladder bench), so prefer rigid→full.
* **Phase 3 is not included.** That covers P3-A1 (denser 64-strip FE mesh, `full_a1` model version, FD's
  `flexbody_a1.py` / `flexeval_a1.py`) and P3-B (wing-shape/planform genes, `planform_b1.py`, `flexbody_b1.py`,
  `flexeval_b1.py`). Their studies, tests, configs, notes and results are left out.
  * Evolution's fidelity code already knows `full_a1`, which is entangled with the Phase 2 code paths. That
    wiring is **strictly opt-in and inert here**. `fidelity.fd_a1_modules()` imports FD's A1 modules lazily
    and raises `FidelityUnavailable` because they are not in this tree.
  * `rigid` / `reduced` / `full` are unchanged. A re-run of `phase2_smoke_p25.json` reproduces the pre-P3
    stored `phase2-smoke-p25-s1` result.
  * `evolution/README.md` still documents `full_a1` and `make_phase2_configs.py --p3a1`. Those need the P3-A1
    FD files.
  * Sim Bridge is the team's 18:00 MST state. Its later edits (the P3-B planform viewer `sim_bridge/planform.py`,
    `test_planform.py`, planform fixtures and the viewer/bundle changes made alongside them) are not included.
* **`flight-dynamics/coupled_sim.py` keeps one absolute default.** It is shipped byte-identical to FD's file, because
  its bytes are part of FD's `reduced` model_version hash. With the Phase 1 path fix, every `reduced` version would
  differ from FD's published `model_versions_post_p25.json`. Its `REPO_FLIGHT_SIM` default
  (`$FLIGHT_SIM_DIR`, else a team-machine path) is used only by FD's v1 `evaluate_flex()` / `flex_demo.py`. Nothing
  in genome/evolution/sim-bridge calls it. Before running those two, set `FLIGHT_SIM_DIR=<repo>/flight_sim`. The
  frozen copy in `evolution/_fd_pin_post_mass/` is byte-identical for the same reason.
* **Run records keep their provenance.** `aircraft_root` in the shipped configs and run records is
  `flight-dynamics/jsbsim_root`, relative to `flight_sim_3d/`. The `repo` / `source_repo` / `argv` / `run_dir`
  provenance fields still show the team machine's paths; they are not used to load anything.
* **Not shipped:**
  * ER's sqlite cache, genomes.jsonl of the pilot runs (~41 MB each), trajectory dumps and the fast-mode / fdv2
    bench run folders (the bench reports are in `evolution/bench_results/fdv2/`).
  * FD's `_scratch/`, OpenAeroStruct venv, logs and PNGs.
  * Sim Bridge's generated `data/`, `screenshots/` and the already-applied `patches/`.
* **Sim Bridge tests that need ER's trajectory dumps still skip** (as in Phase 1); copy a run's `trajectories/`
  folder into place to run them.

## Phase 2 tests

```bash
cd flight_sim_3d
python flight-dynamics/link_jsbsim_data.py
(cd flight-dynamics && python -m pytest -q test_flexwing.py test_flexbody.py)       # 106 passed, ~25 s
(cd genome && python -m pytest -q)                                                   # 152 passed, ~40 s
(cd sim-bridge && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests)               # 38 passed, 13 skipped (need ER trajectory dumps)
python -m pytest -q evolution/tests                                                  # 108 passed, ~1.5-2.5 min
```

These counts come from a clean `git archive` export of this branch, run with Python 3.13.5, jsbsim 1.3.1 and
numpy 2.5.3 on 2026-10-06 (MST). The same export also passed these checks:

* `phase2_flex_smoke.json --aircraft c172x` (8 × 2, rigid→full) ran in 42 s. The best was 0.53594, at pin
  `full:flexv2:e11b8214`.
* A re-run of `phase2_smoke_p25.json` matched the stored `phase2-smoke-p25-s1` exactly: the best cost, gains and
  genome agree bit for bit on all three aircraft (c172x 0.3300231645456844, T38 0.22570865752040178,
  737 0.25647872558108936), with the same 648 sims and 72 cache hits. So the P3-A1 wiring leaves the default
  `full` unchanged.
* A Phase 1 seed-1 re-run (`phase1.json --seed 1`) reproduced `phase1-s1` exactly, as in Phase 1.
* With `EVOLUTION_FD_DIR=evolution/_fd_pin_post_mass`, every pin in `phase2_pilot{,_s2,_s3}.json` and
  `phase2_smoke.json` matches. Under that copy, a re-run of `phase2_smoke.json` matched the stored post-mass
  `phase2-smoke-s1` exactly: c172x 0.39455837284417933, T38 0.22428507351607493, 737 0.27524389750858025.
