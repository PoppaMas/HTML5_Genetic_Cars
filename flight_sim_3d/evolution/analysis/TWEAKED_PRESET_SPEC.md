# Tweaked GA preset for P3-B1 (opt-in): `ga.elite: 4` + `ga.shape_crossover: "uniform"`

Requested by Corleone (2026-10-06). Operator details are owned by Genome Architect and were LOCKED with Evolution Runner
at ~20:07 PT. Genome's preset name: **`phase3_b1_x`** (`genome/presets/phase3_b1_x.json`, section 9 of
`genome/PHASE3_B1_SPEC.md`). Evolution's version is two GA config options on genome_kind `phase3_b1`. There is no new genome
kind, and `genome.py`, `sim.py`, `fidelity.py` and `eval.py` are unchanged, so the cache keys are unchanged.

| Evolution (batch config) | Genome (preset `ga` block) |
|---|---|
| `"genome_kind": "phase3_b1"`, `"ga": {..., "elite": 4, "shape_crossover": "uniform"}` | preset `phase3_b1_x` = `phase3_b1` + `"ga": {"elite": 4, "shape_crossover": "uniform"}` |
| `ga.next_generation_blocks(rng, ranked, GAConfig(elite=4, shape_crossover="uniform"), blocks, spec)` | `block_ops.next_generation_tweaked(ga, rng, ranked, cfg, ops)` |
| `ga.crossover_blocks_uniform_shape` | `block_ops.crossover_shape_uniform` |

## Operator (per child)
1. Elites: `ranked[:elite]` are copied unchanged. `ranked` is the population sorted by the stable ascending argsort of cost
   (ties keep population order), which is the ranking the loop already passes in. On a fidelity ladder this is the ladder's
   ranking (unchanged). There is no re-rank and no mutation of elites. Rejected individuals (cost 2000) sort last as
   usual. With elite 4, the ladder re-scores the 4 carried elites at the authoritative fidelity (the `_ladder` rule is
   unchanged: carried elites are always re-scored).
2. Parent selection is unchanged: `i = flat_rank_select(rng, n, 0.2)`, and `j` is redrawn until `j != i`.
   Parent A = `ranked[i]`, parent B = `ranked[j]`.
3. Crossover (`shape_crossover "uniform"`) uses ONE `rng.random(8)` call (= 8 scalar `rng.random()` calls; for numpy
   `Generator`/PCG64 both give the same values and the same end state, which was checked):
   - `d[0]` controller block (8 genes, whole), `d[1]` structure block (12 genes, whole),
   - `d[2..7]` the 6 shape genes in FD order: `wing_chord_taper_1`, `wing_chord_taper_2`, `wing_chord_taper_3`,
     `wing_twist_mid_deg`, `wing_twist_tip_deg`, `wing_sweep_qc_delta_deg`;
   - `d[k] < 0.5` means the block or gene comes from parent A, otherwise from parent B. The swap probability is 0.5 per shape gene.
   - The per-gene draws REPLACE the shape block's whole-block draw. There is no extra or unused draw.
4. Mutation is unchanged (`mutate_blocks`, drawn after crossover, same as today).

The default `shape_crossover "block"` path is still `crossover_blocks`: `rng.random(3) < 0.5`, bit for bit. With the options
absent (or explicitly `elite 2` / `"block"`) every config resolves to the same resolved config, run identity and run id as
before. `"block"` is dropped from the resolved config, and the populations and RNG streams are bit-identical (see Verification).

Validation: `shape_crossover` must be `"block"` or `"uniform"`. `"uniform"` requires genome_kind `phase3_b1`, and
`elite < pop_size` is required (existing check).

## Verification (2026-10-06, ~20:10–20:20 PT)
- `tests/data/tweaked_preset_golden.json` holds 205 cases captured with the code BEFORE the change
  (`analysis/tweaked_preset_golden.py`): ga operators for every crossover/mutation/elite mode, every `configs/*.json` and
  `configs/seeds/*.json` (gen 0 + 3 generations + final RNG state + run identity), and the smoke T38 g4 population.
  All 205 are bit-identical after the change (`tests/test_tweaked_preset.py`).
- A real tiny phase3_b1 run (T38, 6 x 3, 1 scenario) gives identical checkpoint population, RNG state and genome rows
  (genes + costs) before and after the change, both with the options absent and with explicit defaults.
- The running baseline pilot `phase3b1r1-pilot-s1` (launched from the code snapshot): its stored config has the same
  identity as `configs/phase3b1_pilot.json` re-resolved with the new code. Its gen-0 and gen-1 populations
  (c172x / T38 / 737) are reproduced bit for bit by the new code's default path, so a resume is not affected.
- Cross-check with Genome: `analysis/tweaked_preset_crosscheck.json` (`analysis/tweaked_preset_crosscheck.py --genome`)
  has inputs = runs/phase3b1r1-smoke-s1 T38 g4 (16, rank order) and seeds 0..4, variants default (elite 2, block) and tweaked
  (elite 4, uniform). `genome/block_ops.py` was run read-only on the same inputs (`next_generation` /
  `next_generation_tweaked`): next-gen genomes, final RNG state and per-child parents + crossover draws are
  **bit-identical for all 10 cases**.
