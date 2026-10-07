# genome/: scalable genome schema + multi-objective fitness for flight_sim

This folder adds block-structured genomes, per-aircraft gain ranges, multi-objective fitness, robustness scenarios and an
optional NSGA-II mode to the flight-controller GA in `HTML5_Genetic_Cars/flight_sim`. Nothing in the clone is modified:
the original `evolve.py` and `ga.py` run unchanged through an adapter. See `DESIGN.md` for the reasoning.

## Requirements
The flight_sim venv (jsbsim, numpy, matplotlib, pytest). The clone location comes from `$FLIGHT_SIM_DIR`; the default is
`../../flight_sim` (the repo's own `flight_sim/`); if neither exists, importing `flightsim_path` raises an error telling
you to set FLIGHT_SIM_DIR (no absolute paths in library code). Flight Dynamics' flex-wing model and patched
aircraft come from `$FLIGHT_DYNAMICS_DIR` (default `flight_sim_3d/flight-dynamics`, used read-only). The
legacy preset doesn't need it.

```bash
PY=python   # any Python with flight_sim_3d/requirements.txt installed
cd flight_sim_3d/genome
# Outside the repo layout (e.g. the team working copy <team>/genome) there is no ../../flight_sim:
# export FLIGHT_SIM_DIR=<path to flight_sim> for run_evolve.py & co (importing flightsim_path raises a clear error
# otherwise). The test suite sets it itself (conftest.py, test-only: the local sandbox clone, only if the variable is
# unset, the repo default is absent and the clone exists). flight-dynamics/ and evolution/ resolve as siblings.
$PY -m pytest -q                      # 152 tests, ~30-60 s (incl. the slow legacy-evolve check); -m "not sim and not slow" for the fast subset
```

## Run
```bash
# team default (Phase 1): 600 fpm altitude ramp, profile pitch clamp, comfort 0.05 -- through the ORIGINAL evolve.py
$PY run_evolve.py -- --config $FLIGHT_SIM_DIR/config.example.json --out runs/default_c172x      # --task phase1_default is the default
$PY run_evolve.py --aircraft t38  -- --pop-size 24 --generations 10 --out runs/default_t38      # Phase-1 jets
$PY run_evolve.py --aircraft b737 -- --pop-size 24 --generations 10 --out runs/default_b737
# legacy 6-gain task (instant step, original ranges and weights): reproduces results/example exactly
$PY run_evolve.py --task altitude_hold_legacy -- --config $FLIGHT_SIM_DIR/config.example.json --out runs/legacy_example
# closed-loop sanity check without evolving (default genome + N random genomes per aircraft)
$PY sanity_check.py --aircraft c172x t38 b737 --random 16
# opt-in flex wing (FD two-way coupling, structure genes evolved, flutter/divergence margins as constraints; ~5x slower)
$PY run_evolve.py --task phase1_flex -- --pop-size 16 --generations 5 --out runs/flex_c172x
# Evolution Runner alignment: shared profile export + cross-check of their jet genomes (read-only on evolution/)
$PY export_shared_profiles.py         # -> exports/evolution_phase1_profiles.json (v4) + exports/evolution_phase1_v5_profiles.json (v5 keys, HANDOFF_phase1_v5.md)
$PY heading_report.py                 # heading-hold before/after table -> runs/heading_report.json
# Phase-1 v5 candidate (hold-quality term + downdraft scenario), opt-in preset; v4 is frozen as presets/phase1_v4.json
$PY run_evolve.py --task phase1_v5 [--aircraft t38|b737] -- --config $FLIGHT_SIM_DIR/config.example.json --seed 1 --out runs/v5_c172x_s1
$PY v5_sweep_report.py                # hold-weight sweep table -> runs/v5_sweep.json
$PY v5_report.py                      # v4 vs v5: metrics, gene spread, I-gains, plateau -> runs/v5_report.json
$PY flex12_report.py                  # phase1_flex (12 genes) small reruns + margin-constraint probe -> runs/flex12_report.json
$PY crosscheck_evolution.py           # -> runs/crosscheck_evolution.json
$PY crosscheck_phase2.py --aircraft c172x   # Phase 2 vs ER's evaluator (tmp copy), bit for bit -> runs/crosscheck_phase2_<ac>.json
# Phase 2: phase1_v4 controller + FD flex v2 structure (12 genes from flexbody.gene_schema(); 20 total), seeded gen 0
$PY run_evolve.py --task phase2_flex [--aircraft t38|b737] -- --pop-size 64 --generations 60 --seed 1 --out runs/p2_c172x_s1
$PY run_evolve.py --task experiments/phase2_flex_asym.json -- ...      # + FD's 2 asymmetric genes (22)
$PY p3b1_verify.py                                              # P3-B1 verify set (full_a1_b1 vs A1)
$PY p3b1x_operator_trace.py                                      # phase3_b1_x operator trace (+ phase3_b1 / phase2_flex controls; no flights)
$PY p3b1x_er_fixture_check.py                                    # phase3_b1 / _x next generation vs ER's tweaked_preset_crosscheck.json
$PY evolve_pareto.py --task phase2_flex --pop-size 64 --generations 60 --out runs/p2_pareto_c172x   # track/effort/structural_v2
$PY phase2_gate_study.py              # gen-0 flutter-gate failures, uniform vs seeded -> runs/phase2_gate_study.json
$PY ki_alt_scan.py                    # v5 ki_alt bound study -> runs/ki_alt_scan.json
# NSGA-II (Pareto) mode
$PY evolve_pareto.py --task altitude_hold_pareto --pop-size 24 --generations 8 --out runs/pareto
# check a run against the original code (same seed): prints IDENTICAL / DIFFERENT
$PY verify_legacy.py $FLIGHT_SIM_DIR/results/example runs/legacy_example
```
`run_evolve.py` writes evolve.py's normal outputs (`fitness_history.csv`, `best_gains.json`, plots) and also:
`task.json`, `fitness_detail.json` (per-objective values, skipped objectives, and flight diagnostics such as max pitch,
nz range and climb rate for each scenario) and `at_bounds.json` (genes at or near a bound, with suggested widened ranges).

## Presets (`presets/*.json`)
Every non-legacy preset sets `"shared": "phase1"`. Task, trim point, envelope, clamp, throttle cap, pitch/altitude gain
bounds and default fitness weights then come from `aircraft_profiles/phase1_shared.json`, the set shared with
Evolution Runner (ALIGNMENT.md); keys in the preset override it.

| preset | genes | what it is |
|---|---|---|
| **`phase1_default`** | 8 | team default (signed off): 200 ft step, reference ramped at 600 fpm with **0.1 g corners** (scored against that reference), climb-rate feed-forward, **heading hold** (genes kp_hdg/ki_hdg, bank clamp 16° c172x / 25° jets), `track + 2*effort + 0.05*comfort + 0.01*heading_rms`, same disturbance draws as legacy. Use with `--aircraft c172x/t38/b737`; `"heading_hold": false` gives the 6-gene pre-heading task bit-for-bit. Identical to `phase1_v4` |
| `phase1_v4` | 8 | **frozen copy of today's `phase1_default`** (ramp with 0.1 g corners, feed-forward, heading hold; v5 flags explicitly off). Reproduces the `hdg_after_*` runs bit for bit (tested); use it when you need v4 even if the default moves |
| `phase1_v5` | 8 | **candidate, not the default**: v4 + hold-quality term (`hold_osc`, weight 0.1) + one extra calm scenario with a sustained downdraft (V_TAS·tan 1.5°: c172x 4.69, T38 15.43, 737 12.86 ft/s), mean of 4 scenarios; ki_alt upper bound 0.5 (v5 only, v4 keeps 0.05). Costs are not comparable with v4. See "Phase-1 v5" below and HANDOFF_phase1_v5.md |
| `phase1_no_comfort` | 8 | the same without the comfort term |
| `phase1_flex` | 12 | opt-in: `phase1_default` + FD's 4 structure genes (stiffness_scale, torsion_bend_ratio, damping, non-structural mass) on FD's two-way flex wing. Chord axes and tip mass are fixed by FD. Margins < 1.0 fail, < 1.2 penalised; structural objective (weight 1) includes a wing-mass term |
| `phase2_flex` | 20 | **Phase 2** (aligned with ER's `phase2_pilot`, cross-checked bit for bit: CROSSCHECK_phase2.md): the phase1_v4 task and controller genes (same scenarios and weights; ki_alt ≤ 0.5 phase2-only, v4 keeps 0.05) + FD flex v2 block `structure_v2` (12 genes built from FD's `flexbody.gene_schema()`; `flex.asymmetric: true` → 14, see `experiments/phase2_flex_asym.json`). Objective adds `structural_v2` = FD flexeval's structural cost (`struct_v2_source: "fd"`: FD's J_* terms at FD's weights incl. P2.5 `J_wing_tip_bm_limit`, 24 TERM_KEYS; our peak/RMS formula is the `"genome"` A/B option). NSM floors 1.0–1.25 from FD `gene_schema()`. Full fidelity; model_version = post_p25 (warns on previous post-mass pins, raises only on unknown). Gen 0 seeded at the structural baseline (σ 0.10). The interim mass-credit clip is off (superseded by FD §12; flag kept for A/B). Asymmetric genes off until lateral/roll scenarios exist. DESIGN.md §3c |
| `phase3_b1` | 26 | **P3-B1, opt-in** (2026-10-06): `phase2_flex` controller (8) + FD `structure_v2` (12, P2.5) + FD B1 shape block `shape_b1` (6 genes built from FD `planform_b1.shape_schema()`, L = R; literal pin + drift test). Fidelity `full_a1_b1` = FD `flexeval_b1.evaluate` (ER resolved profile + scenarios from `evolution/configs/phase2_smoke_p25.json`, read-only): shape → FD decode + geometry gate (reject = `geometry_gate:<reason>`, cost = fail_cost, not flown) → structure on the shaped baseline → margins + flight. Baseline shape ≡ `full_a1` bit for bit. Gen 0: structure baseline σ 0.10, shape = identity + clipped Gaussian σ 0.25 × half-range in the operator space (ln x for the 3 chord tapers, x otherwise; storage stays FD's linear u); whole-block crossover (controller \| structure \| shape) + same shape mutation (`block_ops.py`, bit-identical to ER `evolution/ga.py`). Evaluator = ER `evaluate_genome(full_a1_b1)` (direct FD path as cross-checked fallback). **FD B1 r1**: shape genes stored exactly as FD `GENE_ENCODING` (linear in value, all 6; encoding pin + drift test), model_version vs `model_versions_post_p3b1r1.json` (warn; raise opt-in; superseded r0 strings warn-only, never raise). Verify: `p3b1_verify.py` → `runs/p3b1_verify_c172x.json` (r0: `runs/p3b1r0_verify_c172x.json`). PHASE3_B1_SPEC.md, CROSSCHECK_p3b1.md |
| `phase3_b1_x` | 26 | **P3-B1 tweaked GA, opt-in** (Corleone 2026-10-06): `phase3_b1` unchanged (genes, decode, gate, evaluator, gen 0, mutation, flat-rank selection p 0.2) + a `ga` block with ER's option names `{"elite": 4, "shape_crossover": "uniform"}`. Shape crossover per child after parent selection: 8 draws = `rng.random(8)` (controller, structure, 6 shape genes in gene order), draw < 0.5 → parent A; controller and structure stay whole, each shape gene inherited on its own; replaces the shape block draw. Default `block` = `rng.random(3)`, unchanged. Elites: top 4 of the stable-argsort ranking copied unchanged (`run_evolve.py` sets evolve.py's `elite` from the preset unless `--elite` is on the CLI). Bit-identical to ER `ga.next_generation_blocks(shape_crossover='uniform')`. Trace: `p3b1x_operator_trace.py` → `runs/p3b1x_operator_trace.json`. PHASE3_B1_SPEC.md §9 |
| `altitude_hold_legacy` | 6 | original task: instant step, original ranges, `track + 2*effort`, mean over `sim.make_scenarios`, clamp 12°. Bit-identical to the original code; ignores the shared set |
| `altitude_hold_v2` | 6 | widened/log0 ranges + comfort + structural proxy, robust scenarios, mean/CVaR |
| `altitude_hold_pareto` | 6 | Pareto objectives track_alt, effort, comfort (use `evolve_pareto.py`) |
| `c172_3axis` | 14 | pitch + roll/heading + speed blocks, heading steps, robust scenarios |
| `a320_full_schema` | 21 | for inspecting ranges only: all blocks, A320-derived ranges. Do not evolve with it (inert blocks; no A320 in the backend) |

`experiments/` keeps the A/B tasks exactly as they were run: `phase1_default_ki_wide`, `phase1_default_ki_wide2`,
`phase1_ff_off`, `phase1_smooth_ramp`, `phase1_smooth_ff_off`, the heading-weight sweep `phase1_hdg_w0|w003|w03|w1`,
the v5 hold-weight sweep `phase1_v5_hold_w0|w003|w01|w03`, the v5 ki_alt reruns `phase1_v5_kialt02|05`, Phase 2
`phase2_flex_uniform_init` (init comparison, pre-fix), `phase2_flex_uniform_init_clip` (interim clip), `phase2_flex_uniform_init_massfix` (after FD §12) and `phase2_flex_asym`, plus the two historical step-comfort tasks. Heading hold is a shared-set flag (`phase1_shared.json` → `heading_hold`),
on for c172x/T38/737 and never read by the legacy preset.

## Results so far (2026-10-06, c172x, pop 48 × 40 gen, seed 1, 3 scenarios, `config.example.json`)
Pitch, climb, nz and overshoot are given as calm scenario / worst of the 3 scenarios. Overshoot is measured against
the commanded altitude.

| run | best cost | track_alt | max pitch ° | max climb fpm | nz | overshoot ft | genes at/near bounds |
|---|---|---|---|---|---|---|---|
| legacy preset (`runs/legacy_example_v5`, re-verified IDENTICAL to results/example after heading hold + network guard) | 0.1855 | 0.096 | 15.1 / 15.4 | 2313 / 2377 | 0.35–1.92 / 0.18–1.99 | 4.7 / 7.5 | kp_alt upper, ki_alt near lower |
| old default: 600 fpm sharp ramp, FF on (`runs/phase1_default_c172x_v3`) | 0.2512 | 0.076 | 6.2 / 6.7 | 952 / 988 | 0.58–1.24 / 0.46–1.50 | 14.0 / 17.6 | ki_alt upper (0.05), ki_pitch near lower |
| widened ki_alt to 0.5 (`runs/ab_ki_wide`) | 0.2656 | 0.092 | 5.0 / 6.9 | 743 / 1010 | 0.76–1.14 / 0.48–1.53 | 4.4 / 8.0 | ki_pitch near lower (ki_alt → 5e-4) |
| + ki_pitch floor 1e-5 → 1e-7 (`runs/ab_ki_wide2`) | 0.2685 | 0.087 | 5.0 / 6.3 | 742 / 1019 | 0.81–1.15 / 0.46–1.56 | 4.8 / 8.8 | none |
| **new default: ramp corners 0.1 g** (`runs/phase1_default_c172x_v4`) | **0.2063** | **0.061** | 5.1 / 6.1 | 765 / 872 | **0.89–1.11** / 0.52–1.47 | 6.8 / 6.8 | none |
| FF off, sharp ramp (`runs/ab_ff_off`) | 0.2489 | 0.096 | 4.5 / 6.2 | 649 / 930 | 0.80–1.18 / 0.50–1.55 | 2.7 / 6.6 | ki_pitch zeroed |
| FF off, smoothed ramp (`runs/ab_smooth_ff_off`) | 0.2704 | 0.098 | 5.4 / 6.3 | 801 / 858 | 0.86–1.13 / 0.48–1.52 | 4.8 / 7.6 | ki_pitch near lower |

- Widening the ki ranges doesn't help in this seed, so those ranges stay as experiments.
- Ramp-corner smoothing helps on every metric and is the default.
- Feed-forward stays on: it is neutral with sharp corners and clearly better with smoothed ones.
- Each row is one GA seed.
- The smoothed reference arrives about 2.9 s later than the sharp ramp, and tracking is scored against the reference
  each run actually used.

Heading hold before/after (`heading_report.py` → `runs/heading_report.json`). c172x: pop 48 × 40, 3 seeds; evolve.py sets
scenario seed = GA seed, so each before/after pair shares its scenario set. Jets: pop 32 × 20. "Before" = phase1_default
with `heading_hold: false` (= v4 task, bit-identical). Drift = ψ(90 s) − ψ(0) in calm / wind 1 / wind 2. Max |Δψ| is
dominated by the weathercock transient when the steady wind starts at t = 0 in the windy scenarios.

| run | best cost | cost w/o heading term | track_alt | final drift ° | max abs Δψ ° | max pitch ° calm / worst | nz calm / all | kp_hdg / ki_hdg | genes at/near bounds |
|---|---|---|---|---|---|---|---|---|---|
| c172x before s1 (v4) | 0.2063 | 0.2063 | 0.0610 | +25.4 / +19.4 / +27.1 | 27.1 | 5.1 / 6.1 | 0.89–1.11 / 0.52–1.47 | – | none |
| c172x before s2 | 0.2357 | 0.2357 | 0.0575 | +25.6 / +29.2 / +22.9 | 29.2 | 5.5 / 6.9 | 0.86–1.17 / 0.44–1.52 | – | kd_alt near upper |
| c172x before s3 | 0.1957 | 0.1957 | 0.0763 | +25.5 / +28.1 / +28.3 | 28.3 | 4.9 / 5.2 | 0.90–1.10 / 0.60–1.39 | – | ki_pitch zeroed |
| **c172x after s1** | 0.1960 | 0.1941 | 0.0586 | +0.7 / +0.8 / +0.7 | 9.3 | 4.7 / 5.9 | 0.89–1.10 / 0.44–1.55 | 2.24 / 3.7e-4 | ki_alt upper (0.05), ki_pitch near lower |
| **c172x after s2** | 0.1869 | 0.1856 | 0.0547 | −0.1 / +0.1 / +0.1 | 5.1 | 4.7 / 5.9 | 0.89–1.10 / 0.44–1.44 | 1.30 / 0.10 | ki_alt upper, ki_hdg upper (0.1) |
| **c172x after s3** | 0.1977 | 0.1964 | 0.0709 | −0.1 / −0.0 / +0.1 | 4.6 | 4.9 / 5.1 | 0.89–1.11 / 0.55–1.39 | 1.60 / 0.047 | ki_hdg near upper |
| T38 before s1 (v4) | 0.0918 | 0.0918 | 0.0282 | +0.0 / −1.9 / +0.4 | 2.9 | 6.1 / 6.4 | 0.85–1.13 / 0.68–1.22 | – | ki_alt zeroed |
| T38 after s1 / s2 / s3 | 0.0966 / 0.0930 / 0.0925 | 0.0963 / 0.0926 / 0.0921 | 0.033 / 0.031 / 0.028 | 0.0 | 2.7–2.8 | 6.1 / 6.3–6.4 | 0.85–1.14 / 0.68–1.22 | 1.3–1.9 | s3: ki_hdg near lower |
| 737 before s1 (v4) | 0.1102 | 0.1102 | 0.0391 | +0.0 / −2.2 / +0.4 | 2.8 | 5.2 / 5.2 | 0.85–1.14 / 0.72–1.20 | – | ki_alt zeroed |
| 737 after s1 | 0.1068 | 0.1059 | 0.0322 | +0.0 / +0.8 / −0.2 | 2.8 | 5.3 / 5.4 | 0.84–1.15 / 0.71–1.21 | 0.235 / 0.0086 | ki_alt near upper |

**c172x means over 3 seeds:**

| | cost | track_alt | final drift |
|---|---|---|---|
| before | 0.2126 | 0.0649 | 19–29° |
| after | 0.1935 (0.1920 without the heading term) | 0.0614 | ≤ 0.8° |

Holding the heading also helps altitude tracking: no standing bank, so no lift loss.

**Jets:**
- **T38:** Evolution Runner's flag-off s2/s3 on the same task gave 0.0938 / 0.0928 (scenario seed 1). Our flag-on
  s2/s3 used the same scenario seed and gave 0.0930 / 0.0925. s1 is 5 % worse (0.0966), which looks like GA search
  noise with 2 extra genes at a small budget.
- **Existing v4 genomes, flag on** (default heading gains, ki 0): T38 0.0918 → 0.0922 and 737 0.1102 → 0.1103
  (without the heading term), track_alt unchanged.

**Weight sweep** (c172x seed 1):

| weight | final drift | cost w/o heading term | track_alt |
|---|---|---|---|
| 0 | 15° (kp_hdg goes to its lower bound) | 0.1943 | 0.0632 |
| 0.003 | 5.4° | 0.1954 | 0.0657 |
| **0.01 (default)** | 0.7° | 0.1941 | 0.0586 |
| 0.03 | 0.8° | 0.1923 | 0.0542 |
| 0.1 | 0.6° | 0.1919 | 0.0599 |

The ki_hdg pin at 0.1 (s2) sits on a flat optimum: re-flying that genome with ki_hdg 0.03 / 0.1 / 0.2 / 0.4 gives
0.1875 / 0.1869 / 0.1870 / 0.1892, so the bound was not widened.

**Flex on vs off (small: pop 16 × 5 gen, seed 1, c172x, shared default task before heading hold; `runs/flex4_on|off`):**
- **flex-off:** best 0.2979 (track 0.155).
- **flex-on, 4 genes:** best 0.4466 = track 0.069 + 2·effort 0.090 + 0.05·comfort 2.27 + structural 0.040 +
  margin penalty 0.044.
- **Best flex-on structure:** s 0.94, r 0.94, ζ 0.038, non-structural mass 1.14. Flutter margin 1.16 (inside the
  penalty band, flutter found), divergence 1.58, wing +2.5 lb.
- **Same controller, coupled vs rigid:** the flex-off controller flown coupled (default structure) gives track 0.150 vs
  0.155 and comfort 2.01 vs 1.96. The coupling barely moves the rigid-body response on the c172x.
- The low-stiffness corner (s ≈ 0.6, r ≈ 0.8) fails on purpose.
- Too small to compare GAs. Earlier flex comparisons are under `runs/*_SUPERSEDED*`.

**phase1_flex with heading hold (12 genes; pop 16 × 5 gen, c172x, seeds 1–3; `runs/flex12_c172x_s*`, `flex12_report.py`):**
runs end to end. Best 0.4193 / 0.4055 / 0.3755; failures per generation (of 16) 6-2-2-3-1 / 3-1-1-2-3 / 7-4-2-1-0.
Flutter margin 1.207 / 1.166 / 1.221 (s2 sits in the penalty band, penalty 0.028), divergence 1.61 / 1.56 / 1.64,
wing mass −8.6 / −21.4 / −16.3 lb, final heading drift ≤ 0.5° on s1/s2 and 4.7° on s3 (ki_hdg at its lower bound after
5 generations). Constraint probe around s1's best (stiffness s, torsion ratio r): (0.6, 0.8) flutter 0.84 → fails
(cost 2000), (0.7, 0.9) 0.98 → fails, (0.8, 1.0) 1.11 → flies with penalty 0.19, s1 best 1.21 → no penalty,
(1.2, 1.1) 1.43 and (2.0, 1.15) 1.85 → no penalty. Too small a budget to compare with the rigid task.

**Phase-1 v5 candidate (`phase1_v5`; full tables in HANDOFF_phase1_v5.md §4, `v5_report.py`, `v5_cross_eval.py`):**
- Hold-weight sweep (c172x s1): 0.1 chosen, the largest weight with no loss on the v4 task (DESIGN.md §3a).
- c172x calm hold p-p 6.4 / 5.3 / 11.4 → 3.9 / 3.7 / 3.2 ft (worst scenario 12.1 / 15.7 / 11.6 → 9.7 / 7.6 / 8.5).
- T38: ki_alt 4e-7 → 0.05 (upper bound), downdraft residual 6.85 → 0.00 ft, and the v5 genome also scores better on
  the v4 task (0.0870 vs 0.0966). 737 already had ki_alt ≈ 0.03. c172x ki_alt 0.05 / 0.0094 / 1.6e-5.
- c172x gene spread narrows somewhat (ki_alt log10-std 2.30 → 1.84, ki_pitch 1.15 → 0.81, kp_pitch 0.25 → 0.06) but
  the plateau is not fixed (1 %-of-final at gen 22 / 11 / 8 vs 30 / 14 / 18), and on c172x s2 and the 737 the v5 GA
  doesn't beat the v4 genome on the v5 task. v4 stays the default.

**Jets (shared set, pop 32 × 20, seed 1):**
- **T38:** 0.0918 (`runs/phase1_t38_v4`).
- **737:** 0.1102 (`runs/phase1_b737_v4`).
- Evolution Runner's best genomes score 0.12–0.16 under the same fitness (ALIGNMENT.md).

## Phase 2 smoke test (2026-10-06, `phase2_flex`, seed 1, 3 scenarios; box shared with another 8-worker job)

Gen-0 flutter-gate failures (`phase2_gate_study.py`, no flight, 64 genomes; FD measured 20/24/26 of 64 uniform):

| aircraft | uniform init | seeded σ 0.05 | σ 0.10 | σ 0.15 | σ 0.25 |
|---|---|---|---|---|---|
| c172x | 15/64 (min margin median 1.17) | 0/64 (1.24) | 0/64 | 0/64 | 5/64 |
| T38 | 24/64 (1.06) | 0/64 (1.17) | 0/64 | 2/64 | 11/64 |
| 737 | 20/64 (1.06) | 0/64 (1.19) | – | – | – |

GA smoke runs (evolve.py, pop 24 × 8 gen c172x, 16 × 6 T38). "failed" = any scenario not ok (aeroelastic gate or envelope):

| run | failed per gen | best cost | track / effort / comfort | structural_v2 | min margin (binding) | Δ struct. mass |
|---|---|---|---|---|---|---|
| c172x seeded σ 0.05 | 8 (0 aero), 3, 0, 0, 0, 0, 0, 0 | 0.2477 | 0.069 / 0.031 / 1.81 | +0.0096 | 1.226 (wingR coalescence) | −6.3 % (fuselage 0.72) |
| c172x uniform | 14 (5 aero), 4, 0, 0, 0, 0, 0, 0 | 0.2219 | 0.082 / 0.026 / 1.71 | −0.0094 | 1.225 (wingR coalescence) | −13.3 % (fuselage 0.60, tapered wing) |
| c172x uniform, **clip on** (08:05 MST, FD min-gauge floor in) | 13, 2, 0, 0, 0, 0, 1, 0 | 0.2646 | 0.069 / 0.027 / 1.78 | +0.0369 (mass +0.0107) | 1.227 (wingR coalescence) | +2.8 % (fuselage 1.11, tail 0.76) |
| c172x uniform, **after FD §12** (08:16 MST, no clip, sizing terms in) | 13, 3, 2, 0, 0, 0, 0, 0 | 0.2923 | 0.105 / 0.028 / 1.79 | +0.0411 (mass +0.0165, sizing 0) | 1.203 (wingR coalescence) | +5.5 % (fuselage 1.19, tail 1.92, wing root 1.14 / tip taper 0.75) |
| T38 seeded σ 0.05, **after FD §12** (08:16 MST) | 0, 1, 0, 2, 1, 1 | 0.1431 | 0.029 / 0.026 / 0.95 | +0.0142 (mass +0.0045, sizing 3·10⁻⁶) | 1.200 (wingR flutter) | +1.5 % (fuselage 1.03) |
| T38 seeded σ 0.05 | 0, 1, 0, 2, 1, 0 | 0.1260 | 0.028 / 0.024 / 0.96 | −0.0005 (+0.0021 margin penalty) | 1.191 (wingR flutter) | −3.7 % |

After FD §12 (min-gauge floor + sizing terms, no clip) neither run softens the fuselage any more (c172x 1.19, T38
1.03 vs 0.60 / 0.79 before); at both bests every sizing ratio is < 1 (c172x max 0.88, T38 max 1.002 → 3·10⁻⁶), so
the terms act as a barrier and cost nothing at the optimum. The c172x tail went to 1.92 (uniform init, mass +0.009,
unpenalised drift in 8 gens); the wing tip taper sits at its 0.75 bound (root-only sizing does not see outboard
EI; not checked further). Controller-only cost: c172x 0.251 vs 0.228 (clip run), T38 0.129 vs 0.124; one seed
each, so these differences are GA noise as far as we can tell.
All rows above used `struct_v2_source "genome"` (our peak/RMS formula) and σ 0.05 / ki_alt ≤ 0.05; since 08:35
phase2_flex uses FD's flexeval structural cost, σ 0.10 and ki_alt ≤ 0.5 (ER's pilot), so costs are not comparable.
The first three rows ran before FD's minimum-gauge floor (07:53 MST) and before the interim tail/fuselage mass-credit
clip, so the clip rerun changes both. With the clip the fuselage no longer runs to the 0.6 floor (1.11; its peak
fusL/limit drops from 0.65 to 0.37); the mass term goes from −0.040 (credit) to +0.011, and the structure total from
−0.009 to +0.037. The pre-clip uniform best re-scored under today's FD: J_mass −0.021 unclipped, −0.006 clipped (wing
only). Controller-only cost is about the same (0.228 vs 0.231).
Best-design peak root load / limit: c172x seeded wing 0.44, HT 0.06, VT 0.44, fuselage V 0.13 / L 0.53 (RMS 0.01–0.03);
c172x uniform 0.34 / 0.08 / 0.54 / 0.14 / 0.65; T38 0.15 / 0.06 / 0.12 / 0.08 / 0.13. No FD hinge or ultimate term
active. Cost per genome: precheck 0.2 s + ~1.9 s CPU per 90 s scenario (v4 rigid 0.4 s), i.e. ~6 s CPU per 3-scenario
genome; wall ~4.8 s per scenario on today's loaded box. One seed each: the uniform run ending lower is not significant
and is mostly the mass credit; seeding removes the early gate failures but σ 0.05 explores the structure slowly.
Recommendation for the real runs: σ 0.10–0.15 (still ≤ 3 % gate failures), pop 64 × 60 gen (~3 800 genomes,
~6 CPU-h, ~50 min per aircraft and seed on 8 free cores), 2 seeds × 3 aircraft first; NSGA-II
(`evolve_pareto.py`, same budget) if the controller–structure trade-off front is wanted.

## Enabling blocks
In a task JSON:
```json
"blocks": {"pitch_altitude": true, "roll_heading": true, "speed_throttle": true, "yaw_damper": false, "structure": false}
```
Disabled blocks are fixed at their defaults, which equal sim.py's fixed helpers (wing leveler 0.05/0.02, speed PI 0.05/0.01,
rudder 0). The genome is the enabled blocks concatenated in canonical order. `spec.layout()` gives each block's slice, and
`spec.transfer(g, other_spec)` re-encodes a genome between tasks. If the backend has no dynamics for an enabled block
(today that is `yaw_damper`, and `structure` unless `"flex": {"enabled": true}`), loading the task fails unless you set
`"allow_inert_blocks": true`. Flex keys: `"flex": {"enabled": true, "mode": "twoway"|"oneway", "substeps": 2}`.
Task conditions: `"scenarios": {"set": "legacy", "ramp_fpm": 600, "ramp_accel_g": 0.1}` (omit `ramp_fpm` for an instant step,
set `ramp_accel_g` to null for sharp ramp corners) and
`"controller": {"alt_ref_ff": true, "pitch_cmd_limits_deg": [-5, 10]}` (the clamp defaults to the profile's).
`experiments/` holds the two historical instant-step comfort tasks from run 1; they are kept only for reproducibility.
Other task keys: `aircraft`, `legacy_pitch_ranges`, `gene_overrides`
(`{"kp_alt": {"min": 0.01, "max": 1.0, "scale": "log"}}`), `fitness`
(`mode`, `weights`, `aggregate`, `params`, `pareto_objectives`), `scenarios` (`{"set": "legacy"}` or
`{"set": "robust", ...ROBUST_DEFAULTS overrides}`), `controller.throttle_max`, and `backend` (`jsbsim_ext`, or `legacy_sim` to call the original sim.py).

## Adding an aircraft profile
1. Copy `aircraft_profiles/c172x.json` to `aircraft_profiles/<name>.json`.
2. Fill in mass, inertias (slug·ft²), S/b/c̄ (ft), Vstall/Vne (KCAS), Vcruise (KTAS), MMO if any, n-limits, max
   control deflections (deg), max thrust (lbf), a `design_point` (`kcas`, `alt_ft`; it is also the benchmark start condition),
   `pitch_cmd_limits_deg`, `sim_envelope` (`min_kcas`, `nz_limits`), and `control_derivatives` if you know them:
   `cm_de_per_rad` etc., or `cm_de_per_norm` etc. if the JSBSim model is written per normalized deflection. Otherwise
   generic values are used and a warning is printed.
3. **Fill in `sources` for every field group and keep `"approximate": true`** unless the data is authoritative.
   JSBSim XMLs (`<jsbsim>/aircraft/<model>/<model>.xml`, `<metrics>` and `<mass_balance>`) are a convenient source for
   geometry and inertia.
4. Run `$PY sanity_check.py --aircraft <name>` to check that the JSBSim model loads, trims and flies the default genome.
5. `$PY -m pytest tests/test_profiles.py` runs `AircraftProfile.check()` (design point within Vstall..Vne and below MMO,
   and so on).
6. To pin specific ranges, add `gain_overrides`. They beat the derived ranges; task `gene_overrides` beat both.

Python: `profiles.range_factors(load_profile("a320"))` → factors per scaling tag;
`genome_schema.build_spec(blocks, factors, overrides)` → `GenomeSpec`.

## Files
| file | role |
|---|---|
| `genome_schema.py` | GeneSpec (log/linear/log0), blocks, GenomeSpec (encode/decode/layout/transfer/at_bounds) |
| `profiles.py`, `aircraft_profiles/` | AircraftProfile + range-derivation rule; Phase 1: c172x, t38, b737; extras: global5000, a320, f16 (all approximate; f16 flies FD's prepared copy with FD's fixed flex values and point-mass indices, controller bounds PROVISIONAL because it is FBW) |
| `fitness.py` | objectives, telemetry-gated skipping, scalar & Pareto evaluation, scenario aggregation |
| `scenarios.py` | legacy and robust scenario sets |
| `sim_ext.py` | telemetry-recording, bit-identical extension of `sim.simulate` (+ roll/heading/speed genes, mass/CG, sensor noise; non-legacy: FD aircraft copies, gear up, throttle clamp, flex coupling) |
| `fd_bridge.py` | read-only access to FD's flexwing/coupled_sim/flexbody (v1 and v2) and jsbsim_root(_v2) (loaded directly; any aircraft file with JSBSim network I/O is refused); margin pre-check; flutter summary (capped + not-found flag) |
| `init_pop.py` | Phase 2 generation-0 seeding (`init` task key): baseline ± σ for structure genes, optional best-genome seeds; patched into evolve.py only for tasks with `init` |
| `phase2_gate_study.py`, `ki_alt_scan.py` | gen-0 flutter-gate failure study (uniform vs seeded); v5 ki_alt bound study |
| `heading_report.py`, `HANDOFF_heading_hold.md` | heading-hold before/after table; self-contained change list for Evolution Runner |
| `v5_sweep_report.py`, `v5_report.py`, `v5_cross_eval.py`, `flex12_report.py`, `HANDOFF_phase1_v5.md` | Phase-1 v5 candidate: hold-weight sweep, v4 vs v5 comparison (spread, I-gains, hold, downdraft, plateau), cross-evaluation, 12-gene flex rerun; self-contained change list for Evolution Runner |
| `aircraft_profiles/phase1_shared.json`, `ALIGNMENT.md` | shared Phase-1 set with Evolution Runner, and the per-difference decisions |
| `export_shared_profiles.py`, `crosscheck_evolution.py`, `crosscheck_phase2.py` | export the shared set as evolution/ profile blocks (v4 and v5); re-fly their genomes in our sim (read-only) |
| `adapter.py` | task loading; `genome`/`sim` shim modules for the unmodified evolve.py |
| `run_evolve.py` | entry point that runs the original evolve.py with a task |
| `nsga2.py`, `evolve_pareto.py` | NSGA-II sort/crowding, and a Pareto driver that reuses ga.py operators |
| `verify_legacy.py` | diff two run directories (original vs adapter) |
| `sanity_check.py` | closed-loop check of a task on several aircraft without evolving |
| `flightsim_path.py` | loads the original modules under private aliases |

Limitations: c172x, T38 and 737 are flown (the others are ranges only); yaw damper is schema only; structure has
dynamics only in flex mode, on FD's notional structural data; shims rely on the
`fork` start method (Linux). See DESIGN.md §6.
