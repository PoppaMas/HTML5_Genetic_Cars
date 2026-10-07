# P3-B1 genome-side spec: planform shape genes. **LOCKED to FD's 6 genes; implemented (opt-in)**

**Status:** LOCKED / implemented, 2026-10-06 ~18:40 PT; **updated to FD B1 r1** ~18:55 PT. FD locked B1 in
`flight-dynamics/planform_b1.py` (INTERFACE_v2 §14). Current pins: `v2_results/model_versions_post_p3b1r1.json`
(`B1_REV = 1`); r0 pins (`model_versions_post_p3b1.json`) are superseded (warn-only, §5). Gene names, ranges and
defaults are unchanged in r1. Genome implements B1 as the opt-in preset `presets/phase3_b1.json` using `shape_b1.py` and
`block_ops.py`. No other preset changed behaviour.

**r1 in one line (FD):** trim-consistent twist (the basic-twist rigid pitch moment is a Cm0 shift absorbed by the trim
elevator, so no pitch feedback; this closes r0's one-sided loophole in `wing_twist_mid_deg`), a baseline-anchored flown
wing-BM reference (`J_bm_rms` denominator / `J_bm_peak` allowable), the shaped node layout, and the gene encoding stated
explicitly (`planform_b1.GENE_ENCODING`).
**Context:** Corleone held the A1 64×60 pilot; P3-B (B1) is next. ER is wiring a matching `phase3_b1` kind.
**Related:** `CROSSCHECK_p3b1.md` / `runs/p3b1_verify_c172x.json` (verify set), `CROSSCHECK_p3a1.md` (A1 reference),
`PHASE3_CHROMOSOME_SKETCH.md`.

---

## 1. Chromosome (26 genes) and decode order

| block | genes | source |
|---|---|---|
| controller | 8: kp/ki/kd_alt, kp/ki/kd_pitch, kp/ki_hdg | `phase2_flex` (= ER profile `gain_bounds`, checked at load) |
| structure | 12: P2.5 `structure_v2` | FD `flexbody.gene_schema()` at load time (unchanged) |
| shape | 6: `shape_b1` | FD `planform_b1.shape_schema()` at load time |

```
shape genes -> FD decode_shape_b1 (raises, never clips) -> FD geometry_gate
            -> FD wing rebuild on the shaped planform (aero strips + geometry-derived EI/GJ/mass baseline)
            -> structure genes applied on that new baseline -> margins + sizing -> flight (fidelity full_a1_b1)
```
All of this is `flexeval_b1.evaluate`. Genome adds **no** cost term. Cost = FD structural `J_*` (24 TERM_KEYS, FD
weights) + ER sim cost (controller weights from the ER profile; the preset's weights must equal them, or load fails).
`struct_v2_source: fd`.

**Evaluator.** `shape_b1.evaluate_b1` defaults to ER's `evolution.fidelity.evaluate_genome(profile, gains, struct,
scenarios, "full_a1_b1", shape=...)`. ER wired it ~18:07 PT; it is a read-only import with bytecode off. Fallback
(`via="fd"`) is the same flex path rebuilt in genome (`struct_from` clamp, `check_prepared`, `roots`, `_gate`,
`sim = evolution.sim`, `blas_threads = 1`, mass-clip hook), calling FD `flexeval_b1.evaluate(shape_genome=)` directly.
The two paths agree bit for bit. Profile and scenarios are ER's resolved profile from `shape_b1.er_profile_config`
(default `evolution/configs/phase2_smoke_p25.json`) and ER `sim.make_scenarios`. Only the 8 controller genes go to the
flight, as ER does.

## 2. Locked B1 shape genes (FD's code wins)

| # | name | range | FD scale | default | genome encoding |
|---|---|---|---|---|---|
| 0 | `wing_chord_taper_1` | 0.85–1.05 | **linear** | 1.0 | FD's (linear) |
| 1 | `wing_chord_taper_2` | 0.85–1.05 | **linear** | 1.0 | FD's (linear) |
| 2 | `wing_chord_taper_3` | 0.85–1.05 | **linear** | 1.0 | FD's (linear) |
| 3 | `wing_twist_mid_deg` | −2 – +1 | linear | 0.0 | linear |
| 4 | `wing_twist_tip_deg` | −4 – +1 | linear | 0.0 | linear |
| 5 | `wing_sweep_qc_delta_deg` | −5 – +5 | linear | 0.0 | linear |

- **Encoding = FD `GENE_ENCODING`, exactly (confirmed for r1).** FD r1 states it: *"linear_in_value: value = lo +
  u\*(hi-lo), u in [0,1]; encode = (value-lo)/(hi-lo); no log scale"*. The chord-taper "log" in FD's docs is the
  **spanwise chord interpolation** (log-PCHIP, area-renormalised), not the gene encoding. Genome stores every shape gene
  as that linear u: `GeneSpec.decode` = `min + u (max − min)`, `encode` = `(x − min)/(max − min)`, the same arithmetic
  in the same order as `decode_shape_b1` / `encode_shape_b1`. Checked bit for bit on 20 003 vectors (random, corners,
  identity) in `p3b1_verify.py` (`encoding_check`) and in `test_genome_decode_encode_equals_fd_gene_encoding_bitwise`:
  0 decode and 0 encode mismatches. The identity u decodes to FD's defaults exactly (A1 short-circuit).
- **What FD receives.** Physical values `{name: value}` from the genome decode, with **nothing in between**
  (`shape_from_values` is a passthrough, = ER's `eval.shape_values` → `fidelity.shape_from`). r0's clamp to [lo, hi]
  and 1e-12 snap-to-default were removed in r1: they never fired (0 of 20 003 vectors) and made the path differ from
  `GENE_ENCODING` in principle. Anything out of range raises in FD's decode.
- **Log lives only in the operators** (unchanged; FD: "we may still mutate in log space internally"). Init and
  mutation perturb ln(x) for the 3 chord tapers and hand back linear u. ER's `evolution/ga.py` does exactly this; both
  sides are identical bit for bit (§4).
- **Drift guard.** Literal pins `shape_b1.B1_SCHEMA_PIN` (names, ranges, scale, default) and
  `shape_b1.B1_GENE_ENCODING_PIN` (FD's `GENE_ENCODING` string). Load fails on drift of either; tests compare them with
  FD's live module and the r1 pin JSON (`b1_schema.genes`, `b1_schema.encoding`, `b1_rev`).
- **`wing_twist_mid_deg` stays [−2, +1]** (FD's range; parity with ER). Not narrowed. *Option, not adopted:* if r1 runs
  show twist_mid piling up at an end, a genome-side narrower range (e.g. [−2, 0]) is possible via `gene_overrides`, but
  it would break parity with ER and FD's schema pin and would need FD + ER agreement first.
- **Deferred** (FD decode raises at non-baseline values): dihedral, thickness, camber, chord_root/span/area scale, root
  twist → B2+.

## 3. Geometry gate = hard reject

- FD `planform_b1.geometry_gate` runs inside `flexeval_b1.evaluate` before the model is built. It checks for chord
  < 0.05 ft, LE/TE crossover, tip/root taper outside [0.12, 1.25], |sweep| > 45° and LE kink > 25°.
- **On fail:** status `geometry_gate:<reason>` is passed through, cost = FD fail_cost (2 × fail_base = 2000 on c172x),
  and the genome is **not flown**. One aligned `not_flown` entry is written per scenario, and all terms are 0.
  `shape_b1.geometry_rejected(res)` identifies it, and the Task sets `violation` = cost. There is never a fitness credit.
- **Decode errors** (out of range, deferred key, structure gene in the shape block, NaN) raise `ValueError`, which is
  FD's policy and the same as the structure genome. The GA cannot produce them because u is clipped to [0, 1].
- **No natural rejects so far.** FD states the whole in-range box passes the gate on all 4 aircraft; genome checked the
  64 corners on c172x. The reject path is tested with an in-memory tightened bound.

## 4. Operators. **Identical to ER's `evolution/ga.py` P3-B1 operators (bit for bit, tested)**

Agreed via the parent agent (2026-10-06 ~18:04 PT; wording corrected for FD r1):
- **Genes are linear in value.** All 6 shape genes, including the 3 chord tapers, decode as lo + u·(hi − lo) (FD r1
  `GENE_ENCODING`). FD's "log" describes only how chord is interpolated along the span (log-PCHIP). It is **not** a
  log-scale gene.
- **Init and mutation work in log space internally.** They use a clipped Gaussian in an internal operator space: ln x
  for the 3 `wing_chord_taper_*` genes, x for twist and sweep. σ = 0.25 × half-range in that space, and gen 0 sits at
  the identity. Results go back as FD-linear u, which FD explicitly allows: "we may still mutate in log space
  internally".
- **Crossover** swaps whole blocks controller | structure | shape (the default; see §9 for the opt-in per-gene shape
  variant).
- **Code and config:** `block_ops.py`; `operators` in `presets/phase3_b1.json`, using the same keys as ER's
  `shape_ops`.

| | controller (8) | structure (12) | shape (6) |
|---|---|---|---|
| Gen 0 | uniform U[0,1] (as `phase2_flex`) | FD baseline + N(0, **0.10**), clip [0,1] | **identity planform** + clipped Gaussian, σ = 0.25 × half-range (encoded) |
| Crossover (`blocks`) | whole block from parent a or b, p 0.5 | whole block, p 0.5 | whole block, p 0.5 |
| Mutation | per gene with prob `--mutation-rate`, + N(0, `--mutation-sigma`), clip (current behaviour) | same | per gene with prob = shape `mutation_rate` (default `--mutation-rate`), clipped Gaussian σ = 0.25 × half-range (encoded) |

- **σ values.** Chord tapers: 0.25 × ½ (ln 1.05 − ln 0.85) = 0.0264 in ln x, clipped to [ln 0.85, ln 1.05]. Twist mid
  0.375°, twist tip 0.625°, sweep 1.25° (= 0.125 in u), clipped to [lo, hi].
- **RNG order (same as ER).**
  - Gen 0: `rng.random((pop, 20))`, then `standard_normal((pop, 12))` for the structure seed, then
    `standard_normal((pop, 6))` for the shape block.
  - Child: `rng.random(3) < 0.5` for crossover (True → parent a; blocks in order controller, structure, shape).
  - Mutation: `hit = rng.random(26) < rate`. Then one `rng.normal(0, σ, k)` over the controller/structure hits, and a
    clip. Then, only if any shape gene was hit, `standard_normal(6)`, applied to the hit shape genes only.
  - Selection and elites are `ga.next_generation`'s.
- **Gen 0 is centred on the identity, not fixed at it.** ER does the same: identity + clipped Gaussian.
- **Tests.**
  - `test_operators_bitwise_equal_to_er_ga`: `next_generation` and `shape_generation_zero` against ER's functions.
  - `test_generation_zero_bitwise_equal_to_er_batch`: the full 26-gene gen 0 against ER `batch.py`'s phase3_b1
    sequence.
  - The first 20 gen-0 genes equal `phase2_flex`'s seeded gen 0 for the same seed.
- `run_evolve.py` swaps in `block_ops.block_ga` only when `task.operators` is set. Every other preset keeps `ga.py` /
  `init_pop` exactly as before.

## 5. model_version

`phase3_b1` checks FD's live `flexeval_b1.model_version("full_a1_b1", ...)` against `shape_b1.FD_B1_MODEL_VERSIONS`
= **post_p3b1r1**: c172x `56ee798e`, T38 `7e871977`, 737 `6523753c`, f16 `617078a9`. `flex.model_version_check` is
`warn` (default), `raise` or `off`. The base full-fidelity check is off for this preset because it does not apply.

| live (or recorded) string | status | warn | raise mode |
|---|---|---|---|
| post_p3b1r1 | `match` | — | ok |
| post_p3b1 (r0: c172x `3e40908a`, T38 `982bce54`, 737 `1bc748ac`, f16 `bfb25718`; `FD_B1_MODEL_VERSIONS_R0`) | `superseded_r0` | yes | **never raises** (transition window) |
| anything else | `unknown` | yes | raises |

Same policy as the P2.5 post-mass transition (`fd_bridge.check_model_version`). `shape_b1.classify_b1_model_version`
applies it to recorded strings too (e.g. ER runs / caches flown at r0, such as `phase3b1-smoke-s1`). Live FD now
computes only r1 strings, so `superseded_r0` shows up only for records. r0 results are **not** comparable with r1 for
shaped genomes (§6); baseline-shape results are.

## 6. Acceptance (r1: `runs/p3b1_verify_c172x.json`, `CROSSCHECK_p3b1.md`; r0 kept as `runs/p3b1r0_verify_c172x.json`)

| check | r1 result |
|---|---|
| baseline shape + structure 1.0 = A1 tip-verify baseline | **bit-identical** (0.2419400885448951, tip 0; 24/24 terms vs same-process A1). Unchanged from r0 |
| baseline shape + soft tip = A1 soft | **bit-identical** (0.25879473336084763, tip 0.011874078070487388). Unchanged from r0 |
| model_version | `full_a1_b1:flexv2b1:56ee798e` = post_p3b1r1 pin |
| genome storage vs FD `GENE_ENCODING` | exact: 0 / 20 003 decode and encode mismatches; identity → FD defaults |
| same genome through the 26-gene Task | bit-identical with physical values; u round trip → Δ −6.1e-15 (genome controller encoding) |
| ER's 6 B1 verify genomes + genome's 4 shapes vs ER's **r1** re-fly (`p3b1r1_verify_evolution.json`) | **bit-identical** (cost, tip, model_version; per-scenario for genome's 4); ER path = direct FD path |
| shaped r0 → r1 | washout −2.25e-3, tip chord 0.9 −1.68e-4, sweep +3 0, combo (twist_mid −0.5) −1.21e-3; ER: tip_taper −2.57e-4, tip_twist −3 −3.50e-3, sweep ±5 0, fd_bench −1.21e-3. Mostly `J_bm_rms`; flight terms move only where twist_mid ≠ 0 |
| twist_mid range ends (r1-only) | −2: 0.34549 (tip-BM 0.1016); +1: 0.24127 (−6.7e-4 vs baseline) |
| gate reject | `geometry_gate:extreme_taper`, cost 2000, not flown (injected bound) |
| Phase 2 / legacy | full genome suite green (**179** at r1; **193** with phase3_b1_x); `phase2_flex` and other presets unchanged |

## 7. Non-negotiables (unchanged)

- Local only; no commit or push without Corleone. Genome writes only under `genome/`; evolution/ and flight-dynamics/
  are read-only (bytecode off).
- Phase 2 pilots frozen; `struct_v2_source: fd` stays the default. Legacy / v4 / v5 / phase1_flex bit-identity is
  untouched.
- No parallel structural cost.

## 8. Open / follow-ups

1. **`run_evolve.py` end-of-run plots.** `plot_results` replays with genome's rigid `sim_ext.simulate`, which does not
   take ER scenarios. A genome-side `phase3_b1` evolve run needs that path adapted, or GA runs left to ER. Not exercised
   here; no genome GA smoke run.
2. T38 / 737 B1 verify not run on the genome side (c172x only); r1 pins recorded for all 4.
3. ER `evaluate_genome` refuses a model_version that differs from its cached pin (by design). Injected-gate
   experiments therefore use `via="fd"`.
4. **r0 → r1 transition.** r0 records (e.g. ER `phase3b1-smoke-s1`, ER `p3b1_verify_evolution.json`, genome
   `runs/p3b1r0_verify_c172x.json`) carry `3e40908a` and classify as `superseded_r0`. They are comparable to r1 only at
   the baseline shape. Drop the r0 table once ER has re-cut its runs at r1.
5. **twist_mid range option** (not adopted; FD range kept for parity): see §2. At r1, −2° costs +0.104 (tip-BM
   limit) on c172x and +1° −6.7e-4, so the GA should not pile up at −2; watch the r1 smoke runs for +1 pile-up.

## 9. `phase3_b1_x`: tweaked GA (opt-in; approved by Corleone 2026-10-06, RNG order LOCKED with Evolution Runner ~20:07 PT)

Preset `presets/phase3_b1_x.json` = `phase3_b1` + a `ga` block with ER's option names: `{"elite": 4, "shape_crossover": "uniform"}`.
Genes, decode, gate, evaluator, gen 0, mutation and selection (flat rank, p = 0.2) are unchanged.

**Tweaked shape crossover, per child, after parent selection:**
- Draws: `rng.random()` for controller, `rng.random()` for structure, then 6 `rng.random()` for the shape genes in gene
  order. That's 8 doubles total, the same as `rng.random(8)`.
- Uniform REPLACES the shape block's whole-block draw. There is no unused `blk[2]`.
- Draw < 0.5 means parent A (`ranked[i]`). Swap probability is 0.5.
- Default `block` mode stays exactly `rng.random(3) < 0.5`, unchanged.

**Mutation:** unchanged, drawn after crossover.

**Elites = 4:** the top 4 rows of the stable-argsort cost ranking, copied unchanged before reproduction. No re-rank, no
mutation.

**ER's side** uses GA config options, not a new genome kind: `ga.elite: 4` and `ga.shape_crossover: 'uniform'`
(default `'block'`). Genome mirrors those names in the preset's `ga` block.

Implementation (genome):
- `block_ops.resolve_ga` validates the `ga` block (keys `elite`, `shape_crossover` only).
- `block_ops.crossover_shape_uniform` implements the crossover with one `rng.random(8)` call. `block_ops.block_crossover`
  dispatches on `ops["shape_crossover"]`; the `block` branch is the original code.
- `block_ops.next_generation_tweaked` is the public entry point. `block_ops.next_generation` takes an optional `trace`
  hook that records parents and draws without touching the RNG.
- `run_evolve.apply_ga_overrides` sets evolve.py's `elite` from the preset. An explicit `--elite` on the CLI wins (with a
  warning). The preset beats `--config` files and evolve.py's default of 2. Presets without a `ga` block are untouched.
- `phase3_b1` resolves to `shape_crossover: "block"` and has no elite override. Its RNG stream and outputs are unchanged.

Verification (`runs/p3b1x_operator_trace.json`, `CROSSCHECK_p3b1.md` §tweaked):
- Fixed-seed operator trace: seed 1, pop 64, gen 0 + 4 bred generations, synthetic deterministic cost, nothing flown.
- The phase3_b1 and phase2_flex control traces equal the golden captured **before** the change
  (`runs/p3b1x_golden_pre_change.json`) and ER's own functions, bit for bit.
- phase3_b1_x equals ER's `next_generation_blocks(GAConfig(shape_crossover="uniform", elite=4))` bit for bit, and an
  independent inline reference of the spec text.
- ER fixture `evolution/analysis/tweaked_preset_crosscheck.json` (T38 gen-4, seeds 0–4): default and tweaked both
  **BIT-IDENTICAL** (rows, SHA-256, RNG end state, per-child traces) → `runs/p3b1x_er_fixture_check.json`.
- Tests: `tests/test_phase3_b1_x.py`.

