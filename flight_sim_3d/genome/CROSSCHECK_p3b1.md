# Cross-check: P3-B1 verify set (genome `phase3_b1` / `full_a1_b1` vs A1), **FD B1 r1** (2026-10-06 ~19:00 PT)

**Result (r1):**
- Baseline-shape B1 still equals A1 bit for bit, against both the ER A1 numbers and a same-process `full_a1` run (all
  24 TERM_KEYS, margins, per-scenario costs). Baseline and soft tip are **unchanged from r0**.
- The B1 `model_version` is FD's **post_p3b1r1** pin (c172x `full_a1_b1:flexv2b1:56ee798e`).
- Genome storage equals FD `GENE_ENCODING` bit for bit.
- All 10 shaped genomes (ER's 6 + genome's 4) match **ER's own r1 re-fly** bit for bit.
- Shaped costs moved r0 → r1, mostly through `J_bm_rms` (the new BM reference). Flight terms moved only where
  twist_mid ≠ 0. Pure sweep is unchanged.
- Gate rejects are still hard (cost 2000, not flown, status passthrough).

r0 version of this file's numbers: `runs/p3b1r0_verify_c172x.json` (kept for reference; r0 pins superseded).

## Method
- **Evaluator.** `shape_b1.evaluate_b1` → ER `evolution.fidelity.evaluate_genome(..., "full_a1_b1", shape=...)`
  (read-only, bytecode off), cross-checked against the direct FD path (`via="fd"`). The two are bit-identical on every
  case. There is no genome cost term: cost = FD structural + ER sim cost.
- **Setup:** ER `phase2_smoke_p25` c172x resolved profile, 3 scenarios, `scenario_seed` 1. Gains and structure come
  from `runs/p25_tip_verify_c172x.json`.
- **Script / output:** `p3b1_verify.py` → `runs/p3b1_verify_c172x.json`. Run twice; identical numbers. Nothing was
  written to evolution/ or flight-dynamics/.
- **Pins:** `flight-dynamics/v2_results/model_versions_post_p3b1r1.json` (`b1_rev` 1).
  - full_a1_b1: c172x `56ee798e`, T38 `7e871977`, 737 `6523753c`, f16 `617078a9`.
  - A1 host `full_a1:flexv2a1:36fb4f5a` is unchanged, as are rigid / reduced / full.
  - r0 (`model_versions_post_p3b1.json`, c172x `3e40908a`) is superseded and treated as warn-only.

## Encoding (FD r1 `planform_b1.GENE_ENCODING`)
FD: *"linear_in_value: value = lo + u\*(hi-lo), u in [0,1]; encode = (value-lo)/(hi-lo); no log scale"* for all 6
genes. ("log" refers to the spanwise chord interpolation, log-PCHIP, not to the gene encoding.)

Genome `GeneSpec.decode` / `encode` against FD `decode_shape_b1` / `encode_shape_b1` on 20 003 vectors (random,
all-0, all-1, identity): **0 decode mismatches, 0 encode mismatches**. The identity u
`[0.75⁻, 0.75⁻, 0.75⁻, 0.667, 0.8, 0.5]` decodes to FD's defaults exactly.

**No mismatch found.** One tidy-up: r0 genome had a clamp + 1e-12 snap-to-default between decode and FD. It never fired
(0 / 20 003) and is now removed, so FD receives the decoded physical values unchanged, as ER does. Log space is used
only inside the shape operators, which return linear u.

## Verify table (c172x, r1; baseline structure unless noted)

| case | shape | r1 cost | r1 J_wing_tip_bm_limit | status | model_version | vs A1 | r0 cost | Δcost r1−r0 | Δtip |
|---|---|---|---|---|---|---|---|---|---|
| a baseline | identity | 0.2419400885448951 | 0.0 | ok | full_a1_b1:flexv2b1:56ee798e | **bit-identical** (ER A1 + same-process, 24/24 terms) | 0.2419400885448951 | 0 | 0 |
| a2 via `phase3_b1` Task | identity (physical values) | 0.2419400885448951 | 0.0 | ok | same | **bit-identical** | same | 0 | 0 |
| b soft tip (`wing_ei_taper_4` 0.75) | identity | 0.25879473336084763 | 0.011874078070487388 | ok | same | **bit-identical** (ER A1 + same-process) | 0.25879473336084763 | 0 | 0 |
| c washout | twist_tip −2 | 0.24368540375238332 | 0.0 | ok | same | Δ +1.75e-3 | 0.24593427925148523 | **−2.249e-3** | 0 |
| c tip chord | chord_taper_3 0.9 | 0.24733718272264985 | 0.004255877892055006 | ok | same | Δ +5.40e-3 | 0.24750550578643518 | −1.683e-4 | 0 |
| c sweep | sweep +3 | 0.24342259340914543 | 0.0 | ok | same | Δ +1.48e-3 | 0.24342259340914543 | 0 (identical) | 0 |
| c combo (FD bench; **twist_mid −0.5**) | tapers 0.95 ×3, twist −0.5/−2, sweep +2 | 0.26347717007287635 | 0.019933515272561483 | ok | same | Δ +2.15e-2 | 0.26468755677972555 | **−1.210e-3** | 0 |
| c **twist_mid −2** (r1-only) | twist_mid −2 | 0.34548545795950564 | 0.10157397223827903 | ok | same | Δ +1.04e-1 | – | – | – |
| c **twist_mid +1** (r1-only) | twist_mid +1 | 0.2412728501294835 | 0.0 | ok | same | Δ −6.67e-4 | – | – | – |
| d out-of-range | chord_taper_3 0.5 | – | – | `ValueError` (FD decode) | – | not flown | | | |
| d deferred | dihedral 2 | – | – | `ValueError` (deferred) | – | not flown | | | |
| d gate reject (injected) | chord_taper_3 0.9, `GATE_MAX_TAPER` 0.5 in memory | 2000.0 | 0 | `geometry_gate:extreme_taper` ×3 | (gate hashed → other string) | not flown, all terms 0 | | | |

**Reading the r0 → r1 deltas** (term by term, `p3b1r0_verify_c172x.json` vs r1):
- **`J_bm_rms` (baseline-anchored flown wing-BM reference) explains most of it.**
  - washout −2: 0.03316 → 0.03091. This is the only term that changed; the flight terms are bit-identical.
  - chord_taper_3 0.9: 0.03082 → 0.03065. Again the only term that changed.
  - combo: 0.03351 → 0.03026.
- **Flight terms changed only where `wing_twist_mid_deg` ≠ 0** (the combo, twist_mid −0.5). This is r1's
  trim-consistent twist closing the twist_mid loophole: the basic-twist rigid pitch moment is now absorbed by trim.
  - track 0.061330 → 0.062262, effort 0.030106 → 0.030375, comfort 1.75523 → 1.76671, heading ~ same.
  - So r1 flies that genome slightly *worse*, which offsets part of the `J_bm_rms` drop.
  - Net combo: −1.21e-3.
- **Unchanged:** baseline, soft tip and sweep-only shapes are bit-identical r0 = r1. Margins and `J_wing_tip_bm_limit`
  are unchanged in every case.

**twist_mid range ends (r1):**
- −2° costs +0.104, almost all of it `J_wing_tip_bm_limit` 0.1016 (fd_struct 0.132 vs 0.031 at baseline). The
  controller terms are about baseline.
- +1° is slightly better than baseline (−6.7e-4).
- So the range is not free on either side. It is cheap toward +1 and expensive toward −2.
- Kept at FD's [−2, +1] (parity with ER). Narrowing is listed as an option only (SPEC §2).

- **No natural gate rejects.** All 64 corners of the in-range box pass FD's gate on c172x.
- **Task u round trip.** It moves `ki_alt`, `kd_pitch`, `kp_hdg` and `struct_damping_ratio` by about 1 ulp, giving cost
  Δ −6.1e-15. That comes from the genome controller/structure encoding, not B1, and is unchanged from r0.

## ER's B1 verify genomes, re-flown on the genome side (r1)

| ER case | r1 cost | r1 tip | ER r1 re-fly (`p3b1r1_verify_evolution.json`) | ER r0 record | Δcost r1−r0 |
|---|---|---|---|---|---|
| baseline_shape | 0.2419400885448951 | 0.0 | **bit-identical** | 0.2419400885448951 | 0 |
| tip_taper_0.85 | 0.25621291487481784 | 0.010754603840887837 | **bit-identical** | 0.25646956880787164 | −2.567e-4 |
| tip_twist_−3deg | 0.2436369103515783 | 0.0 | **bit-identical** | 0.24714010277442136 | **−3.503e-3** |
| sweep_+5deg | 0.25050749400799605 | 0.0 | **bit-identical** | 0.25050749400799605 | 0 |
| sweep_−5deg | 0.25861054939260436 | 0.0 | **bit-identical** | 0.25861054939260436 | 0 |
| fd_bench_shape (twist_mid −0.5) | 0.26347717007287635 | 0.019933515272561483 | **bit-identical** | 0.26468755677972555 | −1.210e-3 |

- ER's r1 file also re-flew genome's 4 c shapes. All are bit-identical to ours (cost, per-scenario, tip,
  model_version).
- ER path = direct FD path on all 6.
- `e_er_genomes_bit_identical: false` in the JSON summary is expected. That key compares with ER's **r0** record
  (`p3b1_verify_evolution.json`, model_version `3e40908a`). The r1 comparison is `f_er_r1_bit_identical: true`.

## Operators vs ER
Unchanged in r1. Genome `block_ops` produces bit-identical arrays to ER `evolution/ga.py` and to `batch.py`'s phase3_b1
gen 0 (`test_operators_bitwise_equal_to_er_ga`, `test_generation_zero_bitwise_equal_to_er_batch`).

## CPU per scenario (process CPU, 1 BLAS thread, shared box at load ~12 on 8 cores)

| | r1 run 2 | r0 |
|---|---|---|
| A1 baseline | 1.89 s | 1.83 s |
| B1 baseline | 1.90 s | 1.85 s |
| B1 shaped | 1.86–2.04 s | 1.79–1.91 s |

- Shaped/A1 is about 0.99–1.08×. The r1 run-1 outlier of 2.79 s for twist_mid −2 was load: it measured 2.04 s on the
  re-run.
- Treat these numbers as indicative. r1 has no measurable cost over r0.

## Tweaked GA `phase3_b1_x` (ga.elite 4 + ga.shape_crossover 'uniform'), 2026-10-06 ~20:20 PT

Spec: PHASE3_B1_SPEC.md §9. The RNG order was locked with ER. ER's `crossover_blocks_uniform_shape` (ga.py, 20:11 PT)
uses the **same order** as genome's `crossover_shape_uniform`: one `rng.random(8)` call (controller, structure, 6 shape
genes), draw < 0.5 means parent A, and it replaces the shape block draw. **No RNG-order difference.**

### Synthetic operator trace (`p3b1x_operator_trace.py` → `runs/p3b1x_operator_trace.json`)
- **Setup:** seed 1, pop 64, gen 0 + 4 bred generations, run through the same `ga` stand-ins `run_evolve.py` installs.
  Cost is deterministic and synthetic: Σ(1 + k/n)(g_k − c_k)², c_k = (k·φ⁻¹) mod 1. Nothing flies.
- **Recorded:** per-generation SHA-256 of the ranked population and the costs, the full ranked gen 0 and gen 1, and
  parent picks and crossover draws per child.
- **Golden:** captured from the **unmodified** code before the change (`runs/p3b1x_golden_pre_change.json`; block_ops
  md5 e73d629c…).

| preset | elite | shape crossover | gen 0 (ranked) | gen 4 | best synthetic cost g4 | unchanged vs pre-change golden | vs ER's own functions |
|---|---|---|---|---|---|---|---|
| phase3_b1_x | 4 | uniform | `7423a37e7f6f2234…` | `f5d1210525a38564…` | 2.600298 | n/a (new) | **bit-identical** |
| phase3_b1 | 2 | block | `7423a37e7f6f2234…` | `0b1194cdf07eab25…` | 3.155162 | **yes** | **bit-identical** |
| phase2_flex | 2 | — (ga.py uniform) | `ffbc2d2624f67346…` | `a2ab24e7d6779b3e…` | 2.121756 | **yes** | **bit-identical** |

- **ER functions used:**
  - phase3_b1_x: `ga.next_generation_blocks(GAConfig(shape_crossover="uniform", elite=4))`.
  - phase3_b1: the same function at its default.
  - phase2_flex: `ga.next_generation` + `batch.seed_generation_zero`.
- **ER's default path is unchanged by ER's edit**, because ER's code reproduces the pre-change phase3_b1 golden.
- **Shape uniform mixes genes:** 232 / 240 children have a shape block mixed from both parents (expected 1 − 2/64 ≈ 0.97).
- **Elites:** the top `elite` rows are copied unchanged in every generation for all three presets.
- **Independent reference:** ER's `flat_rank_select` and `mutate_blocks` plus the crossover written inline from the
  spec text, with 8 scalar `rng.random()` calls. It is bit-identical to block_ops' single `rng.random(8)` call.

### ER fixture (`evolution/analysis/tweaked_preset_crosscheck.json`, created 20:14 PT) → `runs/p3b1x_er_fixture_check.json`
- **Inputs:** T38 gen-4 population of `phase3b1r1-smoke-s1` (16 rows, rank order; input SHA-256 matches) with the
  phase3b1_pilot GA settings (selection_p 0.2, mutation 0.15 / 0.08 gauss), seeds 0–4.
- **Variants:** default = phase3_b1 (elite 2, `block`); tweaked = phase3_b1_x (elite 4, `uniform`).
- **Regenerated with genome's own presets:** `block_ops.next_generation` / `next_generation_tweaked`, trace hook on.
- **Inputs match:** the fixture's blocks, shape spec, shape mutation rate, gene names, elite and shape_crossover all
  equal genome's preset operators.

| seed | default (genome SHA-256) | rows + SHA + RNG end + trace identical | tweaked (genome SHA-256) | rows + SHA + RNG end + trace identical |
|---|---|---|---|---|
| seed0 | `3598a6456facd778…` | True | `7157c82b68f96d57…` | True |
| seed1 | `8f18a1eafb001130…` | True | `1d635e5470697a77…` | True |
| seed2 | `bdea92f69592d2f8…` | True | `57cf2e5b301fdcaf…` | True |
| seed3 | `e7e5a847813484c5…` | True | `925c215d4bd16f6e…` | True |
| seed4 | `c9beae8fbba839a5…` | True | `0ec9110ebe452d9f…` | True |

**Result: default BIT-IDENTICAL, tweaked BIT-IDENTICAL**, with 0 differing cells. ER also ran genome's block_ops
read-only from its side (the fixture's `genome_crosscheck`: both BIT-IDENTICAL), on the same block_ops file as now
(SHA-256 `c1dc377ddb0f3ea5…`).

