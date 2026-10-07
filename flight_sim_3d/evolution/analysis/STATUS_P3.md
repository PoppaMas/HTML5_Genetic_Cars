# STATUS P3 — B1 r1 PINNED + SMOKED; r0 superseded; 64x60 B1 pilot (r1 pins) HELD for tomorrow pending Corleone (2026-10-06 ~19:01 PT)

**State:** FD shipped B1 r1 (signed off, frozen 18:47 PT). Evolution re-pinned to `model_versions_post_p3b1r1.json`,
froze an FD copy for replay (`evolution/_fd_pin_p3b1r1`), switched the full_a1_b1 node layout to FD's `node_layout_b1`,
re-verified, and ran the 16x5 smoke `phase3b1r1-smoke-s1` (exit 0, 297.1 s). **B1 r0 (`post_p3b1`, `phase3b1-smoke-s1`)
is superseded.** **64x60 B1 pilot `configs/phase3b1_pilot.json` (r1 pins, rigid→full_a1_b1) HELD for tomorrow, pending
Corleone — not launched.** A1 64x60 pilot still Held. Next gate after the pilot decision: P3-C.

## P3-B1 r1 (Evolution side) — 2026-10-06 ~19:01 PT
- **Frozen FD:** `evolution/_fd_pin_p3b1r1` = full copy of `flight-dynamics/` as for `_fd_pin_post_mass` (excluding
  `__pycache__`, `.pytest_cache`, `_scratch`); all 13 files in `FROZEN_A1_B1r1.md5` OK; `diff -r` vs live: none. README
  there (required for exact replay of B1 r1 runs; do not delete).
- **Pins:** full_a1_b1 c172x `flexv2b1:56ee798e`, T38 `7e871977`, 737 `6523753c`, f16 `617078a9`; rigid / reduced /
  full / full_a1 unchanged. `make_phase2_configs.py --p3b1` now writes r1 (`phase3b1_smoke.json`, `phase3b1_pilot.json`);
  `--p3b1 --r0` writes the superseded r0 pins as `phase3b1_smoke_r0.json` / `phase3b1_pilot_r0.json` (record only).
  `--model-versions`: all match for phase3b1_smoke / pilot (frozen and live FD), phase3a1_*, phase2_smoke_p25.
- **Cache:** the key holds model_version, so r1 never reads r0 rows (shaped or baseline); `code_sha` also changed.
- **Node layout:** full_a1_b1 uses `flexbody_b1.node_layout_b1` (FlexState.node_layout, SBHook → v2_map →
  `structure.axis_nodes_body_m`): EA follows shaped chord, sweep and AC shift; twist does not move the EA. Shaped FE
  wings also carry per-node `chord_m`, `geometric_twist_rad` (+ LE up), `le_nodes_body_m` / `te_nodes_body_m` (SI, body
  FRD, origin CG) and the structure block says `node_layout: "FD flexbody_b1.node_layout_b1 (P3-B1 r1)"`. Baseline shape
  = flexbody.node_layout exactly. Still `ga-flightsim-traj/2` with the `planform` header.
- **Tests:** 16 in `test_p3b1.py` (r1 strings; r0 file kept and differs only in the B1 strings; node_layout_b1 baseline
  = flexbody; shaped EA moves with sweep and chord, not twist; r0/r1 cache keys disjoint; r0 config copies). Whole
  suite **136 passed**.

### Verify on r1 (`analysis/p3b1r1_verify_evolution.json`, EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1)
- Baseline shape = A1 bit for bit: 0.2419400885448951 tip 0; tip-soft 0.25879473336084763 tip 0.011874078070487388
  (24 terms, margins, mass; model_version = r1 pin).
- r0 → r1 (c172x, baseline genome): baseline and sweep-only shapes bit-identical; chord / twist shapes lower through
  `J_bm_rms` only (new baseline-anchored BM reference): tip taper 0.85 0.256470 → 0.256213; tip twist −3° 0.247140 →
  0.243637; Genome washout −2° 0.245934 → 0.243685; Genome chord_taper3 0.9 0.247506 → 0.247337. FD bench / Genome combo
  (twist_mid −0.5) 0.264688 → 0.263477: J_bm_rms −0.00326 but flight terms up (track +0.00093, comfort +0.0115 raw) —
  r1's trim-consistent twist. Margins and tip term unchanged. Genome's r1 numbers (incl. r1-only twist_mid −2 → 0.345485
  with tip term 0.1016, twist_mid +1 → 0.241273) equal ours bit for bit. No in-range geometry-gate fails on r1.

### B1 r1 smoke `phase3b1r1-smoke-s1` vs r0 `phase3b1-smoke-s1` vs A1 `phase3a1-smoke-s1` (`analysis/phase3b1r1_smoke_s1_check.json`)
| | c172x r1 / r0 / A1 | T38 r1 / r0 / A1 | 737 r1 / r0 / A1 |
|---|---|---|---|
| best cost | 0.383003 / 0.386075 / 0.323399 | 0.157143 / 0.156200 / 0.226229 | 0.272112 / 0.270588 / 0.256482 |
| Σ J_* (≥ 0) | 0.044395 / 0.044285 / 0.036695 | 0.017878 / 0.017534 / 0.026206 | 0.023676 / 0.023594 / 0.090730 |
| J_mass | 0.01236 / 0.01236 / 0.00108 | 0.00357 / 0.00357 / 0.00491 | 0.00296 / 0.00296 / 0.00023 |
| J_wing_tip_bm_limit | 0 / 0 / 5.4e-4 | 0 / 0 / 0 | 0 / 0 / 5.4e-6 |
| mass Δ | +4.121 % (r0 same) / +0.359 % | +1.190 % (same) / +1.636 % | +0.986 % (same) / +0.077 % |
| flutter margin | 1.3924 / 1.3881 / 1.2350 | 1.2342 / 1.2342 / 1.1891 | 1.1837 / 1.1837 / 1.1678 |
| stiffness at floor | none | none | none |
| nsm root / tip at 1.0 | 94 % / 94 % | 31 % / 94 % | 75 % / 38 % |
| rejects (of 80) | 7 overload (g0) | 1 flutter | 1 overload |
| geometry_gate rejects | 0 | 0 | 0 |
| CPU s/scenario | 1.90 / 1.94 / 1.62 | 1.91 / 1.93 / 1.84 | 1.85 / 1.91 / 1.81 |
- Box load (1-min) start → end: r1 5.9 → 17.3, r0 8.7 → 21.1, A1 0.9 → 6.6 — CPU / wall confounded by other agents.
  Wall 297.1 s (r1) / 356.6 s (r0) / 168.3 s (A1).
- **T38 twist_mid:** still 69 % of the final population at the +1° ceiling (best = +1.0), same as r0. Not an exploit:
  the r1 T38 best re-flown with its shape reset to baseline scores 0.1571354 vs 0.1571432 with its own shape (shape
  contribution +7.8e-6, i.e. slightly worse) — the T38 gain over A1 is controller/structure, and twist_mid is a neutral
  hitch-hiker (FD: −1.4e-5 per degree). r0 reference (FD): 0.1571 vs 0.1562.
- Trajectories g0/g2/g4 × 3 + index valid; all carry `planform`, the r1 node layout (per-node chord / twist / LE / TE)
  and `structure.modal_twist_sign_fixed: true`. run.json `fd_dir` = `evolution/_fd_pin_p3b1r1`. Fresh replay of
  `traj_T38_phase3b1r1-smoke-s1_g4.json` (T38:s0) through the frozen dir: 0.08165462998244637 = trajectory = genomes.jsonl,
  bit for bit (`analysis/phase3b1r1_replay_check.json`).

## Gates
| Gate | Status |
|---|---|
| P3-A1 64x60 pilot (`configs/phase3a1_pilot.json`) | **Held** (Corleone @ ~17:40 PT) — not launched. |
| P3-B1 64x60 pilot (`configs/phase3b1_pilot.json`, **r1 pins**, rigid→full_a1_b1) | **Held for tomorrow, pending Corleone** — not launched. Replay needs `EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1`. |
| B1 r0 (`post_p3b1`, `phase3b1-smoke-s1`, `*_r0.json`) | **Superseded** by r1 — do not pilot on it. |
| P3-C | Next gate. |

---
## Earlier (B1 r0, superseded)
# (was) — P3-B1 WIRED + smoke done; B1 64x60 pilot READY, NOT launched (needs Corleone); next gate P3-C (2026-10-06 ~18:22 PT)

**State:** P3-B1 (FD `full_a1_b1`, `flexv2b1`) wired into Evolution; verify + 16x5 smoke `phase3b1-smoke-s1` done (exit 0,
356.6 s, box under heavy load from other agents). **B1 64x60 pilot `configs/phase3b1_pilot.json` (rigid→full_a1_b1) is
ready but NOT launched — needs Corleone.** A1 64x60 pilot still Held. **Next gate: P3-C.**

## P3-B1 (Evolution side) — WIRED 2026-10-06 ~18:22 PT
- **Genome kind `phase3_b1`** (config `"genome_kind": "phase3_b1"`): phase2_flex controller (8 genes, v4, ki_alt ≤ 0.5)
  + 12 P2.5 structure genes (nsm floor 1.0) + FD's 6 shape genes (`SHAPE_GENES_B1` order/bounds; storage = FD's
  linear [0,1] encoding, so `decode_shape_b1` is used unchanged). phase2_flex and older kinds unchanged.
- **Shape operators (Genome Architect spec, as implemented):** clipped Gaussian in FD's encoded space; the 3
  `wing_chord_taper_*` genes mutate in log(x) clipped to [log 0.85, log 1.05]; twist / sweep linear. Init + mutation σ =
  0.25 × half-range (encoded / log space), per-gene rate = GA mutation_rate (0.15). Gen-0 seeded around identity
  (baseline shape); ctrl + struct gen-0 identical to the A1 seeding (Phase 2 seeded σ 0.10). Whole-block crossover
  (controller | structure | shape). NOTE: FD's schema labels the chord genes `linear`; log is applied only inside the
  operator.
- **Fidelity `full_a1_b1`** (rank 4) in fidelity / eval / sim hooks / cache / runinfo / batch; ladder `rigid→full_a1_b1`
  allowed (rigid ignores shape); non-baseline shape at reduced/full/full_a1 raises. Cache key adds `shape_cache_key`
  only for full_a1_b1 (other keys byte-identical). Pins: post_p3b1 (`full_a1_b1:flexv2b1:3e40908a` c172x / `982bce54`
  T38 / `1bc748ac` 737 / `bfb25718` f16); rigid/reduced/full = post_p25, full_a1 = post_p3a1 (all `--model-versions` match).
- **Geometry gate:** FD's gate fails return `geometry_gate:<reason>`, cost 2000, not flown; rows carry `geometry_gate`,
  history `geometry_gate_rejects` / `_reasons`. The whole B1 gene box passes the gate on all 4 aircraft (64 corners +
  2000 random each, closest T38 taper_min 0.139 vs 0.12) → no in-range gate fails; the reject path is tested with an
  injected stricter limit.
- **Trajectories:** stay `ga-flightsim-traj/2`; optional header `planform` (`fd-planform/1`, source P3-B1, SI, body FRD,
  decoded genes, `symmetric: true` + `wing{span_frac, y_m, chord_m, chord_baseline_m, le_x_m, te_x_m, twist_rad}`,
  `sweep_qc_rad`), all taken from FD's `PlanformStrips`; `le_x_m` is relative to the root quarter-chord point (+fwd).
  `validate_traj` accepts files with or without it (equal-length check). `structure.axis_nodes_body_m` (65 nodes) moves
  with **sweep only** (FD node_layout uses shaped `lam`; chord / twist / AC shift not reflected — FD caveat).
- **Tests:** `tests/test_p3b1.py` (14). Whole suite **133 passed** (119 + 14). Two existing tests touched for the new
  fidelity tuple / MV regex only.
- **Configs:** `make_phase2_configs.py --p3b1` → `configs/phase3b1_smoke.json` (16x5) and `configs/phase3b1_pilot.json`
  (64x60 rigid→full_a1_b1); identical to the phase3a1 configs except fidelity / genome_kind / pins.
- `code_sha` changed (as with every code change): old cache rows miss for new runs, old runs can't be resumed; results
  for rigid/reduced/full/full_a1 unchanged (20 existing configs resolve to the same run ids).

### Verify (`analysis/p3b1_verify_evolution.json`, c172x, 3 scenarios, seed 1)
- Baseline shape at full_a1_b1 = A1 **bit for bit** (None / {} / defaults / identity vector): baseline
  0.2419400885448951 tip 0; tip-soft 0.25879473336084763 tip 0.011874078070487388.
- Shaped (baseline genome): tip_taper 0.85 → 0.25647 (tip_bm 0.01075, flutter 1.2804); tip twist −3° → 0.24714; sweep
  +5° → 0.25051 (flutter 1.1800); sweep −5° → 0.25861 (flutter 1.3279); FD bench shape → 0.26469.

### B1 smoke `phase3b1-smoke-s1` vs A1 smoke `phase3a1-smoke-s1` (best of g4; `analysis/phase3b1_smoke_s1_check.json`)
| | c172x B1 / A1 | T38 B1 / A1 | 737 B1 / A1 |
|---|---|---|---|
| best cost | 0.386075 / 0.323399 | **0.156200** / 0.226229 | 0.270588 / 0.256482 |
| total structural cost (Σ J_*) | 0.044285 / 0.036695 | 0.017534 / 0.026206 | 0.023594 / 0.090730 |
| J_mass | 0.01236 / 0.00108 | 0.00357 / 0.00491 | 0.00296 / 0.00023 |
| J_wing_tip_bm_limit | 0 / 0.000538 | 0 / 0 | 0 / 5.4e-6 |
| mass delta | +4.121 % / +0.359 % | +1.190 % / +1.636 % | +0.986 % / +0.077 % |
| flutter margin | 1.3881 / 1.2350 | 1.2342 / 1.1891 | 1.1837 / 1.1678 |
| stiffness genes at floor | none / none | none / none | none / none |
| wing_nsm root / tip at 1.0 | 94 % / 94 % | 31 % / 94 % | 75 % / 56 % |
| rejects (of 80) | 7 overload (all g0) / 17 | 1 flutter / 3 | 1 overload / 2 |
| geometry_gate rejects | 0 | 0 | 0 |
| CPU s/scenario | 1.94 / 1.62 | 1.93 / 1.84 | 1.91 / 1.81 |
- Wall 356.6 s vs 168.3 s; 648 sims each; load at start 8.7 (rose to ~22, other agents' batch/replay) vs 0.9 → CPU and
  wall are contention-confounded; FD-standalone shaped vs baseline was ~equal (verify: 1.84–2.12 vs 1.99 s).
- Structural sum ≥ 0 at every best. Traj g0/g2/g4: 9 files + index valid, `planform` present in all.
- **Shape movement (final pop):** chord tapers stay near 1.0 (|Δ| 0.006–0.064; 737 taper_1 mean 0.936); twist / sweep
  moved (|Δ| 0.4–1.6°). **Flag: T38 `wing_twist_mid_deg` piled at its +1.0° ceiling (69 % of final pop, best = 1.0)** —
  possible exploit; T38's big gain is mostly scenario 1 track (0.205 → 0.082). No other shape gene ≥ 25 % at a bound.
- 16x5 with identical gen-0 per aircraft (same seed, as in A1) — c172x / 737 bests are worse than A1 at this budget;
  not evidence either way.

## Gates
| Gate | Status |
|---|---|
| P3-A1 64x60 pilot (`configs/phase3a1_pilot.json`) | **Held** (Corleone @ ~17:40 PT) — not launched. |
| P3-B1 64x60 pilot (`configs/phase3b1_pilot.json`, rigid→full_a1_b1) | **Ready, NOT launched — needs Corleone.** Review T38 twist_mid ceiling pile first. |
| P3-C | **Next gate.** |

---
## Earlier (A1 / P3-B wait)
# (was) — A1 smoke done; A1 pilot HELD; P3-B IN PROGRESS (2026-10-06 ~17:40 PT)

**State:** A1 smoke done (`phase3a1-smoke-s1`, exit 0, 168.3 s). **64×60 A1 pilot (`configs/phase3a1_pilot.json`) HELD by Corleone (~17:40 PT) — NOT launched.** **P3-B IN PROGRESS** / waiting on FD + Genome B1 planform gene list (and shape→mesh API on `full_a1`). Evolution Runner messaged FD (priority) and Genome Architect (priority) to lock that list; no Evolution wiring until it lands.

## P3-A1 (Evolution side) — DONE 2026-10-06 ~15:00 PT
- **Wiring** (`fidelity.py`, `cache.py`, `eval.py`, `runinfo.py`, `batch.py`; `sim.py` / `trajectory.py` / `validate_traj.py`
  needed no change): `FIDELITIES = rigid|reduced|full|full_a1`, `RANK full_a1 = 3` (ladder top rung `rigid->full_a1`
  allowed; full_a1 cannot screen for full). `full_a1` → FD `flexeval_a1.evaluate` / `FlexHookA1` / `FlexBodyModelA1`
  (imported lazily, no bytecode in FD's folder); rigid/reduced/full still call `flexeval.evaluate` exactly as before.
  FlexState / node_layout / v2_geometry / `make_fd_model` / SBHook treat full_a1 like full (65 FE nodes per wing; modal
  9-node wings unchanged). A1 results also carry `structural_model` and `tip_bm` {method, strip_discrete_term}.
- **Cache:** key already holds fidelity + model_version → full / full_a1 disjoint; `eval_key` now also refuses a
  model_version whose prefix ≠ `<fidelity>:` (payload unchanged, so rigid/reduced/full keys are unchanged).
  Note: `code_sha` (sim+genome+fidelity+eval) moved `b7ec79967423ec7f → 7e0a797cdd1c905e` as with every earlier code
  change: new runs do not hit the old rows, the old rows stay on disk untouched and are still reachable with their own
  run's key (tested: P2.5 smoke c172x g0 best re-flown at full = its stored P2.5 cache entry, bit for bit).
- **Pins:** `pin_model_version` accepts `full_a1`; `--model-versions` reports it (smoke + pilot configs: all match
  FD's `model_versions_post_p3a1.json`; rigid pins = post_p25). Mismatch still raises (check_pins / batch.run before any
  eval or cache write / `eval.evaluate(pin=)` / cache guard) — tested.
- **Tests:** `tests/test_p3a1.py` (11 tests). Whole suite **119 passed** (108 + 11). Three existing tests were stale
  since FD's P2.5 (they compared FD's current full strings with the post_mass pins and failed on the pre-change code
  too): `test_phase2::test_config_pins_equal_fd_published_versions`, `test_phase2::test_tiny_pinned_full_run_end_to_end`,
  `test_v2_map::test_evaluate_pin_raises_on_mismatch` → now check live matches against post_p25 (post-mass configs still
  asserted equal to post_mass). `test_eval.MV_RE` gained the `full_a1` pattern.
- **Configs:** `make_phase2_configs.py --p3a1` → `configs/phase3a1_smoke.json` (= `phase2_smoke_p25.json` with fidelity
  full_a1; pins full_a1 only, no rigid rung) and `configs/phase3a1_pilot.json` (64x60 `rigid->full_a1`, seeds as P2.5 pilot
  c172x=1/T38=2/737=3, scenario_seed 1; pins rigid post_p25 + full_a1 post_p3a1). **Pilot HELD by Corleone (~17:40 PT) — not launched.**

### Tip-verify cross-check (`analysis/p3a1_tip_verify_evolution.json`, c172x, Genome's pair, 3 scenarios, seed 1)
| case | full total | full J_tip_bm | full_a1 total | full_a1 J_tip_bm |
|---|---|---|---|---|
| baseline | 0.24220696133846464 | 0.0 | 0.2419400885448951 | 0.0 |
| tip-soft (taper4 = 0.75) | 0.2657981420373094 | 0.017509318343313852 | 0.25879473336084763 | 0.011874078070487388 |
- full reproduces the P2.5 numbers **bit for bit** (total, per-scenario, tip). A1 tip-soft 0.01187 (FD: ~0.0119);
  A1 strip-discrete diagnostic at 64 strips 0.00961 (FD table: 0.0096). A1 model_version = pin.

### A1 smoke `phase3a1-smoke-s1` vs P2.5 smoke `phase2-smoke-p25-s1` (best of g4; `analysis/phase3a1_smoke_s1_check.json`)
J_wing_tip_bm_limit: A1 = station-exact, P2.5 = strip-discrete (different definitions).

| | c172x A1 / P2.5 | T38 A1 / P2.5 | 737 A1 / P2.5 |
|---|---|---|---|
| best cost | 0.323399 / 0.330023 | 0.226229 / 0.225709 | 0.256482 / 0.256479 |
| total structural cost (Σ J_*) | 0.036695 / 0.035038 | 0.026206 / 0.025715 | 0.090730 / 0.090722 |
| J_mass | 0.001077 / 0.003289 | 0.004908 / 0.004908 | 0.000232 / 0.000232 |
| J_wing_tip_bm_limit | 0.000538 / 0 | 0 / 0 | 5.4e-6 / 7.9e-8 |
| mass delta | +0.359 % / +1.096 % | +1.636 % / +1.636 % | +0.077 % / +0.077 % |
| flutter margin | 1.2350 / 1.2516 | 1.1891 / 1.1900 | 1.1678 / 1.1678 |
| stiffness genes at floor (final pop) | none / none | none / none | none / none |
| wing_nsm root / tip at floor 1.0 (final pop) | 81 % / 50 % (same in P2.5) | 81 % / 62 % (same) | 56 % / 56 % (same) |
| rejected rows (of 80) | 17 overload / 17 overload | 3 (2 diverged, 1 struct_ult_empennage) / same | 2 (1 overload, 1 flutter) / same |
| CPU s / scenario (measured) | 1.62 / 1.51 | 1.84 / 1.68 | 1.81 / 1.65 |
- Wall 168.3 s (A1) vs 153.6 s (P2.5); 8 workers, 648 sims computed each. Structural sum ≥ 0 on all bests; no J_* < 0 at
  any A1 best (phase2_check exit 0, no flags). T38 / 737 A1 bests are the same genomes as P2.5 (costs differ by the A1
  model only); c172x A1 found a different, lower-cost best. Trajectories g0/g2/g4 exported, 9 files + index valid
  (`ga-flightsim-traj/2`, wings 65 FE nodes + 9-node `wing*_modal`); resim mismatches: none.
- nsm still piles at the 1.0 floor exactly as in P2.5 (A1 does not change that; it is FD's P2.5 floor).

## Gates
| Gate | Status |
|---|---|
| P3-A1 64x60 pilot (`configs/phase3a1_pilot.json`) | **Held** (Corleone @ ~17:40 PT) — not launched. |
| P3-B | **Active** — waiting on FD exact B1 gene list + Genome aligned preset (shape→mesh API on `full_a1`). No Evolution wiring until that lands. |

---
## Earlier (P2.5 phase)
# (was) STATUS P3 — P2.5 APPROVED / in progress (2026-10-06 ~13:54 PT)

**State:** Planning only for Phase 3 gates. **No Phase 3 code.** Phase 2 runs not touched (`phase2-pilot-s3` and siblings left alone).

**P2.5:** **APPROVED** (Corleone choice A @ 13:53 PT) — **in progress with FD.** FD patching NSM floor at 1.0 + tip allowable + §12 wording; will publish new `model_versions`. Genome + Evolution re-pin after FD lands.

## Done
- Wrote `evolution/analysis/PHASE3_PLAN.md`.
- Merged FD sketch (A1/A2 mesh, planform-as-increments, multi-config list, D0 then D1 continuous) and Genome `PHASE3_CHROMOSOME_SKETCH.md` (B1 CPs, C scenario tags + mean/fail-on-gate, open-loop morph waypoints).
- Gate order locked: **P2.5 → P3-A1 → P3-B → P3-C → P3-D0**.
- Corleone approved P2.5 choice A; FD instructed to patch (no longer waiting).

## In progress
| Who | What |
|---|---|
| **FD** | **P2.5 patch:** NSM floor at 1.0; tip allowable; §12 wording; publish new `model_versions`. |

## Waiting on
| Who | What |
|---|---|
| **Genome + Evolution** | Re-pin / Genome preset after FD publishes P2.5 `model_versions` (optional `phase2_flex` follow-on smoke). |
| **FD** | A1 prototype + measured CPU; exact B gene list vs Genome 12–18; C first-class knobs; D0 DOFs / stiffness-follows-morph. |
| **Genome** | Align preset to final FD gene list after lock; morph_schedule block sketch when D0 starts. |
| **Sim Bridge** | Traj extensions for denser nodes / config_id / morph (when A1+ lands). |

## Non-goals (still)
No GitHub push; no Phase 2 pin/cache breakage; no Phase 3 implementation; no parallel structural cost formula.
