# Elitism / selection / per-generation persistence audit (2026-10-06 ~19:05 PT)

Evidence: `analysis/elitism_audit.json` (from `analysis/elitism_audit.py`, which only reads `runs/`) and fresh-process
replays (`analysis/elitism_reload_check.py`). Runs: phase2-pilot-s1/s2/s3 (64x60, rigid->full), phase3a1-smoke-s1,
phase3b1-smoke-s1 and phase3b1r1-smoke-s1 (16x5), with 3 aircraft each (c172x, T38, 737). **Result: nothing is
missing, so no GA code, config or existing test was changed.** One regression test file was added.

## (a) Elitism: on everywhere, 2 elites per generation, copied unchanged
- `ga.next_generation` (controller / phase2_flex / phase3a1 genomes) and `ga.next_generation_blocks` (phase3_b1) both
  start the new population with `[ranked[i].copy() for i in range(elite)]`. Only the children that follow go through
  selection, crossover and mutation, so elites are never re-crossed or re-mutated.
- None of the 28 configs (`configs/*.json`, `configs/seeds/*.json`) sets `ga.elite`. All of them resolve to the
  `DEFAULTS` value, **elite = 2**, with selection_p 0.2. `resolve_config` only rejects elite >= pop_size, so `elite: 0`
  would be accepted, but no config uses it.
- Multi-fidelity (`_ladder`): the carried elites (index 0..elite-1, gen > 0) are always re-scored at the
  authoritative fidelity.
- Real data: in all 18 run/aircraft series, ranks 0-1 of every generation reappear at index 0-1 of the next generation
  (`carried_elite: true`) with bit-identical `genome_norm`, `genome`, `cost`, `per_scenario_cost` and `status`. That
  is 59x2 carries per pilot aircraft and 4x2 per smoke aircraft. It also holds across the phase2-pilot-s3 resume
  boundary (session 1 resumed at c172x 41, T38 40, 737 40).

## (b) Selection: geometric rank selection (`flat_rank_select`, p = 0.2)
- The selector walks the ranks from the best and takes rank k with probability 0.2 at each step. If no rank is
  taken, it falls back to a uniform pick. So P(k) = 0.2*0.8^k + 0.8^n/n. Parent 2 is redrawn until it differs from
  parent 1. Selection uses ranks only, so the scale of the cost does not matter. There is no tournament and no
  truncation.
- Selection pressure:

  | pop | P(best) | P(top 4) | P(top 25%) | P(bottom 50%) | P(worst) | E[rank], parent 1 / parent 2 |
  |---|---|---|---|---|---|---|
  | 16 | 0.202 | 0.597 | 0.597 | 0.154 | 0.0088 | 3.65 / 3.91 |
  | 64 | 0.200 | 0.590 | 0.972 | 0.0008 | ~6e-7 | 4.00 / 4.31 |

- Failed genomes are **not excluded**. They sort to the bottom by cost:
  - envelope fail (overload, attitude, diverged, crash): 1000-2000 per scenario, averaged over the scenarios, so one
    failed scenario out of 3 gives a cost of at least ~333;
  - margin, geometry-gate or load fail: 2000.

  They can still be picked with the small rank probability. Largest per-generation chance that parent 1 is a failed
  genome: phase3a1/b1/b1r1 smoke c172x g0, 0.118 (7 overloads in 16); phase2-pilot-s1 T38 g0, 0.027. In the pilots it
  is otherwise ~0. There were no geometry-gate rejects in any B1 smoke.
- Ladder caveat: individuals scored at the authoritative fidelity rank above screen-only ones, so a full-scored
  *failed* genome can outrank screen-only *ok* genomes. This happened only at g0 of phase2-pilot-s1 T38 (ranks 13-15:
  structural_ultimate_empennage 718.0 / 733.6, flutter 2000.0) and phase2-pilot-s2 T38 (rank 15: diverged 361.8).

## (c) Per-generation persistence
- `runs/<id>/genomes.jsonl` (schema `ga-flightsim-genomes/1`) gets **one row per individual for every generation**.
  It is written by `_append_rows` before each checkpoint, and no option or config skips it. Fields:

  individual_id `<ac>:g<gen>:r<rank>`, run_id, aircraft, generation, index, rank, is_best, is_elite, carried_elite,
  genome (decoded), genome_norm (encoded [0,1]), gains, struct, shape (B1), cost, fitness, per_scenario_cost,
  scenario_ids, eval_seed, status, feasible, feasibility_fidelity, terms, fidelity, model_version, session, margins,
  mass_*; plus the screen_* and ladder_* fields in ladder runs.

  All 18 series have every generation with the full population: 64 rows x 60 gens (11520 rows, 42 MB, for
  phase2-pilot-s1) and 16 x 5 for the smokes.
- `checkpoints/<ac>.json` holds gen_next, done, pop (the next, unevaluated population; at the end, the final ranked
  population), rng_state, history[] (per gen: best, mean, median, std, rates, best_genome, best_gains, ...) and
  best_per_gen[] (generation, genome (norm), fitness, per_scenario, fidelity). `history.jsonl` mirrors the history.
  For every generation, rank 0 in genomes.jsonl matches checkpoint best_per_gen and history best bit for bit, and
  decode(genome_norm) == genome.
- **Reload works for any generation.** Fresh-process re-flights through `eval.evaluate` of generations that have no
  exported trajectory:
  - phase3b1r1-smoke-s1 c172x:g3:r0, s0 = 0.3749146296343817 (FD dir `_fd_pin_p3b1r1`);
  - phase2-pilot-s1 T38:g30:r0, s0 = 0.06540879802372773 (FD dir `_fd_pin_post_mass`).

  Both are bit-identical to genomes.jsonl, from both the decoded `genome` and the checkpoint `genome_norm`, and
  model_version matches.
- **Resume works only from the latest checkpoint** (`--resume runs/<id>`, or the same `--config`). The checkpoint is
  overwritten each generation and keeps only the latest rng_state, so restarting bit-exactly from an *older*
  generation is not possible. Its population is in genomes.jsonl, but its RNG state is not stored. Resume is tested
  by test_resume.py (SIGKILL), and phase2-pilot-s3 is a real resumed run.

## (d) Best-so-far is monotone in all 18 series (max regression 0.0)
| run | c172x g0 -> final | T38 | 737 |
|---|---|---|---|
| phase2-pilot-s1 | 0.38337 -> 0.22869 | 0.15597 -> 0.09758 | 0.26427 -> 0.12594 |
| phase2-pilot-s2 | 0.34100 -> 0.21551 | 0.19663 -> 0.09771 | 0.21533 -> 0.12413 |
| phase2-pilot-s3 | 0.38285 -> 0.20196 | 0.14256 -> 0.09210 | 0.31069 -> 0.12038 |
| phase3a1-smoke-s1 | 0.51341 -> 0.32340 | 0.23460 -> 0.22623 | 0.38149 -> 0.25648 |
| phase3b1-smoke-s1 | 0.52655 -> 0.38607 | 0.24114 -> 0.15620 | 0.38296 -> 0.27059 |
| phase3b1r1-smoke-s1 | 0.51978 -> 0.38300 | 0.24162 -> 0.15714 | 0.38251 -> 0.27211 |

Re-evaluation noise is exactly 0: every carried elite's cost is bit-identical.

**Pitfall:** in ladder runs, `cost` on a non-rescored row is the rigid screen cost. min(cost) over genomes.jsonl is
below the true best in 57-59 of 60 generations of every pilot series. Take the best from rank 0, or from rows with
fidelity == the run's fidelity.

## Tests added
`tests/test_elitism.py` (17 tests): elites copied bit-identically and as copies, for every crossover/mutation mode
and for the B1 block operators; the rank-selection distribution; and a tiny real 2-aircraft GA run asserting that
elites carry over unchanged, the best is monotone, and genome rows plus checkpoint best rows exist for every
generation.
