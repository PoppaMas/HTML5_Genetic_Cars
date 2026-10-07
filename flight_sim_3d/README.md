# flight_sim_3d: multi-aircraft 3D flight-sim GA, Phases 1, 2 and 3

This folder takes the single-aircraft JSBSim altitude-hold prototype in `../flight_sim/` and scales it to
several aircraft. Four teams built it, and each folder has its own README and design notes:

| folder | owner | what it contains |
|---|---|---|
| `genome/` | Genome Architect | Genome schema, task presets (`phase1_v4` = the default, `phase1_v5` = opt-in), per-aircraft profiles (`aircraft_profiles/phase1_shared.json`), `sim_ext.py` (extended sim and heading hold), fitness, NSGA-II, an adapter that runs the original `evolve.py` |
| `evolution/` | Evolution Runner | Multi-aircraft batch GA runner (`python -m evolution.batch`): process pool, sqlite result cache, checkpoints with exact resume, run logs, trajectory export. Configs include `phase1.json` and `phase1_hdg.json` |
| `flight-dynamics/` | Flight Dynamics | Modal flex-wing soft-body model v1 (`flexwing.py`, `coupled_sim.py`) the Phase 2 flex-body model v2 (`flexbody.py`, `flexeval.py`, `INTERFACE_v2.md`) and the Phase 3 opt-in models A1 (`flexbody_a1.py`, `flexeval_a1.py`) and B1 (`planform_b1.py`, `flexbody_b1.py`, `flexeval_b1.py`). Also `jsbsim_root/` and `jsbsim_root_v2/` (patched copies of the c172x, T38, 737 and f16 aircraft XML) and FM-quality checks |
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
* **(Phase 2 snapshot only.)** The Phase 2 commit left out P3-A1 and P3-B1, and its `full_a1` wiring was inert.
  Phase 3 now ships them; see the Phase 3 section below.
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

---

# Phase 3: denser wing mesh (P3-A1) and wing-shape genes (P3-B1)

## What's new

Both Phase 3 models are **opt-in** fidelities. `rigid`, `reduced` and `full`, the Phase 2 pins and every Phase 1/2
result above are unchanged (see the reproduction checks under "Phase 3 tests").

* **P3-A1, fidelity `full_a1`: a denser structural mesh.** FD's files are `flight-dynamics/flexbody_a1.py`,
  `flexeval_a1.py` and `INTERFACE_v2.md` §13.
  * Each semi-wing has 64 strips (was 32) and keeps 4 bending + 3 torsion + 2 in-plane modes per side (was 3 + 2 + 1).
    The HT, VT and fuselage are unchanged.
  * The outboard tip-bending-moment sizing check is now station-exact (`sizing_a1`), so it no longer jumps with the
    strip count.
  * Genes, ranges, weights and the 24 `TERM_KEYS` are unchanged. It costs about 1.07–1.09× the CPU of `full`.
  * Studies: `p3a1_study.py`, `v2_results/p3a1_truncation.json` and `p3a1_benchmark.json`. Pins are in
    `v2_results/model_versions_post_p3a1.json`. Tests: `test_flexbody_a1.py`.
* **P3-B1, fidelity `full_a1_b1`: 6 wing-shape genes on the A1 host.** FD's files are `planform_b1.py`,
  `flexbody_b1.py`, `flexeval_b1.py` and `INTERFACE_v2.md` §14. Left and right wings stay symmetric.

  | gene | range | meaning |
  |---|---|---|
  | `wing_chord_taper_1` / `_2` / `_3` | 0.85 – 1.05 | chord ratios between the control points at η 0, 1/3, 2/3 and 1 |
  | `wing_twist_mid_deg` | −2 – +1 | geometric twist at η 0.5, relative to the root (nose-up +) |
  | `wing_twist_tip_deg` | −4 – +1 | geometric twist at the tip, relative to the root (negative = washout) |
  | `wing_sweep_qc_delta_deg` | −5 – +5 | change to the baseline quarter-chord sweep |

  * Span and planform area stay at the JSBSim values, so the chord genes only move area along the span. The wing is
    re-placed so the aerodynamic centre does not move.
  * The structural baseline follows the shaped chord (EI/GJ ∝ c³, mass ∝ c). The 12 structure genes then apply as before.
  * A geometry gate rejects impossible shapes (status `geometry_gate:<reason>`, cost 2000). Every in-range shape
    passes it, so the gate is only a safety net.
  * The baseline shape is bit-identical to `full_a1` (`v2_results/p3b1_acceptance.json`, `p3b1r1_acceptance.json`).
* **The B1 r1 fix.** In the first B1 version (r0), the T38's `wing_twist_mid_deg` piled at its +1° ceiling.
  * FD traced this to a small model loophole (`v2_results/p3b1r1_twistmid_study.json`, §14 "r1 addendum"):
    * The rigid pitch moment of the basic twist load was fed back as a moment proportional to Δq.
    * The flown wing-bending reference included the shape's own twist load.
  * r1 changes two things:
    * The twist moment is now a Cm0 shift that the trim elevator absorbs.
    * The wing-bending reference is anchored to the baseline planform.
  * The twist effect fell from −6e-4 to −1.4e-5 per degree, which is effectively neutral.
  * r1 also lays out the FE nodes on the shaped wing and states the linear gene encoding.
  * The r1 pins are in `v2_results/model_versions_post_p3b1r1.json` (c172x `full_a1_b1:flexv2b1:56ee798e`, T38
    `7e871977`, 737 `6523753c`, f16 `617078a9`). The r0 pins (`model_versions_post_p3b1.json`) are superseded and kept
    only as a record.
* **Genome.**
  * Preset `presets/phase3_b1.json`: 26 genes = 8 controller (`phase2_flex`) + 12 structure (P2.5) + 6 shape. The
    code is `shape_b1.py` and `block_ops.py`.
  * Spec: `PHASE3_B1_SPEC.md`. Cross-checks: `CROSSCHECK_p3a1.md`, `CROSSCHECK_p3b1.md`, and the P3-A1 section in
    `CROSSCHECK_phase2.md`.
  * Verify sets: `runs/p3b1_verify_c172x.json` (r1) and `runs/p3b1r0_verify_c172x.json` (r0), plus
    `runs/p3a1_tip_verify_c172x.json`.
  * The sketch for later phases is `PHASE3_CHROMOSOME_SKETCH.md`. Tests: `tests/test_phase3_b1.py`.
* **Evolution.**
  * New fidelities `full_a1` and `full_a1_b1`, with ladders `rigid→full_a1` and `rigid→full_a1_b1`.
  * Genome kind `phase3_b1`:
    * Shape mutation is a clipped Gaussian with σ = 0.25 × half-range. The 3 chord tapers mutate in log space.
    * Crossover swaps whole blocks (controller | structure | shape).
  * `model_version` pins for both fidelities. The cache key gets a shape key only for `full_a1_b1`.
  * Configs `configs/phase3a1_{smoke,pilot}.json` and `phase3b1_{smoke,pilot}.json` (r1 pins). The `phase3b1_*_r0.json`
    files are kept only as a record.
  * The frozen FD copy `evolution/_fd_pin_p3b1r1/` (see its README).
  * Notes: `analysis/STATUS_P3.md` and `analysis/PHASE3_PLAN.md`. The elitism audit is `analysis/ELITISM_AUDIT.md`
    (below). Tests: `tests/test_p3a1.py`, `test_p3b1.py` and `test_elitism.py`.
* **Sim Bridge.**
  * Planform support: `sim_bridge/planform.py` and the viewer's planform reshaping. The viewer draws the r1 shaped-wing
    node geometry, and the HUD has a planform line and a top-view camera.
  * `tests/test_planform.py` with synthetic fixtures in `tests/fixtures/planform_synthetic/`.
  * The one-command replay page builder `tools/build_b1_page.sh` / `.py` and a rebuilt `viewer/dist/fv.bundle.js`.

## Run `full_a1_b1`

Run these from `flight_sim_3d/`, after the install steps above. `link_jsbsim_data.py` now also links the engines into
both frozen FD copies.

```bash
cd flight_sim_3d
python flight-dynamics/link_jsbsim_data.py
python -m evolution.batch --config evolution/configs/phase3b1_smoke.json --model-versions      # read-only pin check
# smoke: 16 x 5 on c172x/T38/737, single fidelity full_a1_b1, genome kind phase3_b1 (~5 min on 8 cores)
python -m evolution.batch --config evolution/configs/phase3b1_smoke.json --run-id p3b1-smoke-s1
# exact replay of the shipped r1 smoke through the frozen FD copy (also writes its trajectories/)
EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1 python -m evolution.batch --config evolution/configs/phase3b1_smoke.json --run-id p3b1r1-smoke-repro-s1
# A1 only (no shape genes)
python -m evolution.batch --config evolution/configs/phase3a1_smoke.json --run-id p3a1-smoke-s1
# the 64 x 60 B1 pilot (rigid -> full_a1_b1, r1 pins); configured but NOT run yet (deferred, see below)
python -m evolution.batch --config evolution/configs/phase3b1_pilot.json --run-id p3b1-pilot-s1
# genome side: the P3-B1 verify set (full_a1_b1 vs A1, c172x)
(cd genome && python p3b1_verify.py)
```

The live `flight-dynamics/` is byte-identical to the frozen r1 copy, so the shipped configs pass the pin check with or
without `EVOLUTION_FD_DIR`. Use the frozen copy for exact replays once FD's live files move on.

## Results: B1 r1 smoke (`evolution/runs/phase3b1r1-smoke-s1`)

The smoke ran 16 × 5 per aircraft, seed 1, with the same scenarios as the A1 smoke. The values are the best of
generation 4, from `analysis/phase3b1r1_smoke_s1_check.json`.

| | c172x | T38 | 737 |
|---|---|---|---|
| best cost, B1 r1 | **0.38300** (0.38300301361568806) | **0.15714** (0.15714319108695818) | **0.27211** (0.27211181306747817) |
| best cost, A1 smoke (`phase3a1-smoke-s1`) | 0.32340 | 0.22623 | 0.25648 |
| flutter margin (≥ 1.0 required) | **1.392** | **1.234** | **1.184** |
| wing+structure mass Δ vs FD baseline | +4.12 % | +1.19 % | +0.99 % |
| stiffness genes at their floor | none | none | none |
| geometry-gate rejects | 0 | 0 | 0 |

* **16 × 5 is only a smoke test.** Every aircraft starts from the same generation 0 as the A1 smoke. B1 is worse than A1
  on the c172x and 737 at this budget, and that is not evidence either way. The 64 × 60 pilot is deferred.
* **The T38 twist finding: the shape is neutral, and the gain comes from the controller and structure.** The T38 B1
  best (0.1571 vs A1's 0.2262) still has `wing_twist_mid_deg` at the +1° ceiling (69 % of the final population).
  * With its shape reset to the baseline, the same genome scores 0.1571354 vs 0.1571432 with its own shape, so the
    shape is worth +7.8e-6 (slightly worse).
  * FD's study traces the drop to controller gains (kd_pitch, kd_alt and ki_hdg) and structure, not to the wing shape.
  * Since r1, mid-span wash-in in [0, +1] is close to cost-neutral and can drift freely. Treat its value as
    uninformative until B2.
* **Exact replay.** The T38 generation-4 best, re-flown in a fresh process through `evolution/_fd_pin_p3b1r1` on
  scenario T38:s0, costs 0.08165462998244637, bit for bit (`analysis/phase3b1r1_replay_check.json`).

## Elitism, selection and persistence audit (`evolution/analysis/ELITISM_AUDIT.md`)

ER audited the three Phase 2 pilots and the three Phase 3 smokes (18 run/aircraft series). It found nothing missing,
so no GA code or config changed. `tests/test_elitism.py` (17 tests) locks the behaviour in.

* **Elitism:** every config resolves to **2 elites**, copied unchanged into the next generation. They are not crossed
  over or mutated again, and their costs come back bit-identical, so re-evaluation noise is 0.
* **Selection:** geometric **rank selection with p = 0.2** (`flat_rank_select`), so it does not depend on the scale of
  the cost. Failed genomes are not removed; they sort to the bottom by their 333–2000 cost.
* **Persistence:** **every genome of every generation is saved** in `runs/<id>/genomes.jsonl` (one row per individual).
  Any generation can be re-flown bit for bit from its row.
* **Best-so-far is monotone** in all 18 series, with a maximum regression of 0.0.
* `analysis/elitism_audit.py` / `elitism_reload_check.py` reproduce the audit. The pilot series need the pilots'
  `genomes.jsonl` (~42 MB each), which is not shipped. The shipped smoke records have theirs.

## Build the B1 replay page

Re-run the smoke first, because the shipped run records don't include trajectories:

```bash
cd flight_sim_3d
EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1 python -m evolution.batch --config evolution/configs/phase3b1_smoke.json --run-id p3b1r1-smoke-repro-s1
cd sim-bridge
tools/build_b1_page.sh p3b1r1-smoke-repro-s1 --replay-proof --no-shots   # -> data/p3b1r1-smoke-repro-s1_standalone.html (~19 MB)
```

* `--replay-proof` re-flies the exported generations through the frozen FD copy and requires bit-identical costs.
* Screenshots (drop `--no-shots`) need Playwright/Chromium, which is not in `requirements.txt`.
* Several seeds can go on one page: `tools/build_b1_page.sh <run1>,<run2>,<run3>`.
* See `sim-bridge/README.md` ("Phase 3-B1 page in one command").

## Phase 3 caveats

* **The 737 wing drawn aft of the fuselage is only cosmetic.** The viewer draws FD's r1 wing geometry on a procedural
  fuselage, so the 737's FD wing root LE (2.87 m) sits aft of the procedural one (1.67 m). The physics does not use the
  procedural fuselage.
* **Resume works only from the latest checkpoint.** The checkpoint is overwritten every generation and keeps only the
  latest RNG state. Older generations are in `genomes.jsonl`, but a run cannot restart bit-exactly from them.
* **The structural data are still notional.**
  * The aero is quasi-steady strip theory on fixed JSBSim tables.
  * B1 has no induced-drag or stall-margin cost for twist.
  * Wing size, dihedral, thickness and camber are not genes (see deferred items).
* **Superseded r0 runs.** `phase3b1-smoke-s1` ran on the superseded r0 pins and is kept as a record. Current code
  cannot replay it exactly, and the repo has no r0 FD copy.
* **The frozen copies are byte-identical to the team's files.** `evolution/_fd_pin_p3b1r1/` ships only what Evolution
  needs, byte-identical to the team's frozen copy, because `model_version` hashes those bytes. Its `coupled_sim.py`
  keeps the same team-machine default as `flight-dynamics/coupled_sim.py` (see the Phase 2 caveats).
* **Not shipped:**
  * the Phase 3 smoke trajectories (~142 MB per run)
  * ER's sqlite cache and logs
  * FD's `*.partial.jsonl` study checkpoints and logs
  * Sim Bridge's generated pages and screenshots

  `evolution/tests/test_p3a1.py::test_full_unchanged_vs_p25_cache_entry` reads ER's cache and skips when
  `evolution/cache/evals.sqlite` is missing. Run the tests before your first batch run: a fresh local cache exists but
  lacks that P2.5 entry, so the test then fails. The Phase 2 smoke re-run below covers the same check.

## Phase 3 tests

```bash
cd flight_sim_3d
python flight-dynamics/link_jsbsim_data.py
(cd flight-dynamics && python -m pytest -q test_flexwing.py test_flexbody.py test_flexbody_a1.py test_flexbody_b1.py)   # 167 passed, ~50 s
(cd genome && python -m pytest -q)                                                   # 179 passed, ~45 s
(cd sim-bridge && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests)               # 55 passed, 18 skipped (ER trajectory dumps / node+esbuild not present)
python -m pytest -q evolution/tests                                                  # 152 passed, 1 skipped (ER's sqlite cache), ~2 min
```

These counts come from a clean `git archive` export of this branch, run with Python 3.13.5, jsbsim 1.3.1 and
numpy 2.5.3 on 2026-10-06 (MST). Sim Bridge's skips need data the repo doesn't ship. With the team's trajectory
dumps, the phase1v5 `genomes.jsonl` and `tools/build/node_modules` (esbuild) copied in, all 73 Sim Bridge tests pass.
The same export also passed these checks:

* **B1 r1 smoke.** `EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1` with `phase3b1_smoke.json` (run id
  `p3b1r1-smoke-repro-s1`) matched the stored `phase3b1r1-smoke-s1` exactly:
  * The best cost, gains and genome agree bit for bit on all three aircraft: c172x 0.38300301361568806,
    T38 0.15714319108695818, 737 0.27211181306747817.
  * All 240 `genomes.jsonl` rows are identical (cost, per-scenario cost, genome, status, terms, model_version).
  * It used the same 648 sims and 72 cache hits. The run took 208 s on 8 cores.
* **Fresh-process re-flight.** `evolution/analysis/elitism_reload_check.py` re-flew `T38:g4:r0` on scenario T38:s0
  through the frozen copy: 0.08165462998244637, bit-identical to `genomes.jsonl` and to the checkpoint genome.
* **Replay page.** `tools/build_b1_page.sh p3b1r1-smoke-repro-s1 --replay-proof --no-shots` built a 19.4 MB page
  (1.6 MB gzip). The proof was EXACT: 9 generations re-flown with relative error 0.0, and 681/681 channels
  bit-identical in all 9 files.
* **Phase 2 default smoke.** A re-run of `phase2_smoke_p25.json` (run id `phase2-smoke-p25-repro-s1`) still matched
  the stored `phase2-smoke-p25-s1` exactly: c172x 0.3300231645456844, T38 0.22570865752040178, 737 0.25647872558108936.
  All 240 rows are identical, with the same 648 sims and 72 cache hits. So the Phase 3 code leaves the default `full`
  unchanged.
* **Pin check.** `--model-versions` matches for `phase3b1_smoke` / `phase3b1_pilot` (live FD and the frozen r1 copy),
  `phase3a1_smoke` and `phase2_smoke_p25`.

## Deferred

* **The 64 × 60 B1 pilot** (`configs/phase3b1_pilot.json`, r1 pins, rigid→full_a1_b1) is configured but has not been
  run. The 64 × 60 A1 pilot (`phase3a1_pilot.json`) is also on hold.
* **B2 genes:** dihedral, thickness, camber and wing size (chord root / span / area). They need a geometric-dihedral
  path and rescaled JSBSim tables, and FD's decode rejects them for now.
* **Uniform crossover inside the shape block.** Crossover currently swaps whole blocks.
* **Raising the elites from 2 to 3–4.**

# Phase 3b: overnight 2026-10-06/07 (B1 pilot, breeding A/B, P3-B2a section genes + energy cost)

Built on `flight-sim-3d-phase3`. Everything here is opt-in: v4 / v5 / phase2_flex / phase3_b1 runs, the A1 and B1 r1 pins
and every earlier run record reproduce unchanged (see Tests below).

## What's new

* **flight-dynamics:** P3-B2a. `planform_b2.py`, `flexbody_b2.py`, `flexeval_b2.py` (fidelity `full_a1_b2a`: dihedral,
  thickness and camber as native JSBSim increments on `jsbsim_root_v2b2/`, thickness→EI and mass coupling), the
  per-scenario drag-energy export (`v2_results/p3b2a_energy_ref.json`, `p3b2a_energy_calibration.json`),
  `test_flexbody_b2.py`, INTERFACE_v2 §15, `v2_results/p3b2_gene_spec.json`, `model_versions_post_p3b2a.json`,
  `FROZEN_B2a.md5`. B2a with every B2 gene at its default is bit-identical to B1 r1.
* **genome:** presets `phase3_b1_x` (elite 4 + uniform shape crossover) and `phase3_b2a` / `phase3_b2a_x`,
  `PHASE3_B2_SKETCH.md`, `tests/test_phase3_b1_x.py`, `tests/test_phase3_b2a.py`.
* **evolution:** `ga.elite` and `ga.shape_crossover` options (defaults 2 / `blocks` = old behaviour), fidelity
  `full_a1_b2a`, genome kind `phase3_b2a`, the opt-in `energy_cost` term (J_energy + J_speed_guard, outside TERM_KEYS,
  which stays 24; B2a configs only), the per-scenario determinism fix in `fidelity.py`, `tests/test_p3b2a.py`,
  `tests/test_tweaked_preset.py`, `tests/test_elitism.py`, frozen FD copy `_fd_pin_p3b2a/` (minimal, byte-identical),
  pilot / A/B configs and run records, `analysis/STATUS_P3B1_pilot.md`, `STATUS_P3B1_ab.md`, `STATUS_P3B2a.md`,
  `P3B2A_DETERMINISM.md`.
* **sim-bridge:** B2a viewer support (dihedral, t/c, camber from the per-node layout), c172x CG centring
  (display only; `?cgcentre=0` turns it off), A/B and multi-run page building in `tools/build_b1_page.{sh,py}`.

Run records shipped (config / summary / run / sessions / checkpoints): `phase3b1r1-pilot-s1`, `-s2`,
`phase3b1r1-pilot-tweaked-s1`, `-s2`, plus `phase3b2a-smoke-s1` and `phase3b2a-smoke-detfix-s1` with `genomes.jsonl`.
The four 64 × 60 pilots' `genomes.jsonl` (about 50 MB each) and all trajectories are **not** in the repo.

## Results: 64 × 60 B1 r1 pilot (`phase3b1r1-pilot-s1`, rigid → full_a1_b1, r1 pins)

| aircraft | gen-0 best | g59 best | change |
|---|---|---|---|
| c172x | 0.37991 | 0.24806 | −35 % |
| T38 | 0.15360 | 0.10928 | −29 % |
| 737 | 0.30962 | 0.13517 | −56 % |

All bests stay clear of flutter. The 737 and T38 cut sweep (737 to 20°, at the −5° bound) and twist the tips down
3–3.5°; the c172x uses mid-span twist (−2°, at the bound). Seed 2 (`-s2`): c172x 0.23654, T38 0.10599, 737 0.13079.
The Sim Bridge replay of gens 0 / 29 / 59 re-flies every cost and channel bit for bit.

## Breeding A/B: elite 4 + uniform shape crossover (`phase3_b1_x`) vs baseline, 2 seeds

The tweaked preset won 1 of 6 seed × aircraft pairs. Mean paired Δ (tweaked − baseline): c172x −0.85 %, T38 +0.22 %,
737 +2.46 %. Every Δ is within about one Phase 2 seed sd, so there is no evidence the tweak helps. **Recommendation:
keep the baseline operators (elite 2, whole-block crossover)**, which is what B2a uses. The two arms evolve different
wings: baseline lowers sweep, tweaked raises it (737 ≈ 27°). Write-up: `evolution/analysis/STATUS_P3B1_ab.md`.

## P3-B2a section genes with the energy cost

* Genes: `wing_dihedral_delta_deg`, two camber genes and the two thickness genes `wing_tc_root_scale` /
  `wing_tc_tip_ratio`. Fidelity `full_a1_b2a`, genome kind / preset `phase3_b2a` (29 encoded genes by default).
* Drag never reached the score before (sim.py holds speed with throttle). The approved `energy_cost: true` adds
  J_energy (FD's drag-energy signal against a frozen per-aircraft baseline-shape reference) plus J_speed_guard (so
  slowing down cannot game it). Weights w_E: c172x 2.175, T38 1.663, 737 1.202, f16 1.303. It applies to B2a configs
  only; at baseline shape it adds 0.
* **Thickness is unlocked by the energy cost:** thickness genes are accepted only with `energy_cost: true` and are
  freed with `"shape_locked": []`. The shipped default (`shape_locked: null`) and the shipped smoke still hold them at
  1.0, as in Genome's `phase3_b2a` preset.
* Smoke `phase3b2a-smoke-detfix-s1` (16 × 5): c172x 0.38783, T38 0.15516, 737 0.22352.

```bash
cd flight_sim_3d
EVOLUTION_FD_DIR=evolution/_fd_pin_p3b2a python -m evolution.batch --config evolution/configs/phase3b2a_smoke.json --run-id p3b2a-smoke-repro-s1
```

## Determinism fix (`evolution/analysis/P3B2A_DETERMINISM.md`)

Sim Bridge's full replay of the first B2a smoke found 3 weak c172x genomes that did not replay exactly. Cause
(fidelity.py): energy terms were added only when the whole genome's status was ok, so a genome that overloaded in one
scenario lost energy from its good scenarios in batch mode but kept it when flown one scenario at a time. The check is
now per scenario, with a regression test. `phase3b2a-smoke-detfix-s1` has the same genomes, ranks and bests as
`phase3b2a-smoke-s1` (only those 3 rows' energy terms changed), and all 720 scenario replays in fresh processes match.
B1, A1 and Phase 2 have no energy term and cannot hit this.

## Genes piling at their bounds

* dihedral delta at 0 (the lower bound) for every aircraft by gen 4 of the B2a smoke
* sweep at ±5° (737 −5° in the baseline pilot; T38 / c172x +5° in seed 2)
* `chord_taper_3` at 1.05
* c172x `twist_mid` at −2°

FD may widen some ranges; not done here.

## Phase 3b tests

On a clean export of this branch: FD 219 passed; genome 207 passed; sim-bridge 59 passed / 24 skipped (unshipped data);
evolution 178 passed / 1 skipped (unshipped sqlite cache). B1 r1 smoke (`phase3b1r1-smoke-s1`,
`_fd_pin_p3b1r1`) and `phase2_smoke_p25` re-run bit for bit; the B2a detfix smoke re-runs bit for bit through
`_fd_pin_p3b2a`; `--model-versions` matches for the pilot and smoke configs.

## Deferred (Phase 3b)

* **B2b wing size** (area and aspect ratio): needs JSBSim reference / coefficient rescaling, AR effects and wing-mass
  scaling; FD expects it on 2026-10-07. AR / area ≠ 1.0 is rejected for now.
* **The 64 × 60 B2a pilot** (baseline operators): waiting for approval.
* **A/B seed 3** (`phase3b1_pilot_tweaked_s3.json` / `phase3b1_pilot_s3.json` configured, not run).
* **Range widening** for the genes at their bounds.
