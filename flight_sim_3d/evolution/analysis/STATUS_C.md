# STATUS C: fast mode + fidelity (FD flexeval v2), verified/finished 2026-10-06 07:50 PT

Implementation verified (README "Fast mode"): --viz on|off (default off, execution-only, gen 0/mid/final best re-flown
with recorder, resim check), --fidelity rigid|reduced|full via FD's flight-dynamics/flexeval.py (imported, not forked;
model_version from FD's result; cache key has fidelity + model_version), reduced gate c172x 0.9 / 737,T38,f16 1.0,
multi-fidelity screen -> top-k + elites at full, >= 25 % on swept wings (k_full 8 of 32 + elites on 737), feasibility
only from full, per-gen Spearman in history.jsonl. Tests: test_eval.py (viz on/off bit-identical rigid + full, flexeval
bit-identity, gates, ladders, resume).

Benchmark: analysis/bench_fdv2.py, pop 32 x 4 gens, c172x + 737, cache off, 8 workers, struct genes, seeds 1-3
(seed 1 06:59-07:15 PT; seeds 2-3 07:18-07:40 PT, finished this session). Files: bench_results/fdv2/results{,-s2,-s3}.json,
report{,-s2}.md/json (counterfactual), seeds.md/json (analysis/bench_fdv2_seeds.py). Box load 1-min 6.5-23 throughout.
Counterfactual for seed 3 NOT done: Flight Dynamics edited flexeval.py/flexbody.py at 07:45:51/07:46:21 PT mid-run;
our guard raised "FD model_version reduced:flexv1:6fe189b7 != reduced:flexv1:25843500" (report-s3.log). Mixing FD versions
would be invalid, so the counterfactual covers seeds 1-2 (8 generations).

## viz on vs off (seed 1, full-only)
| | batch wall s | mean gen wall s c172x / 737 | task CPU s c172x / 737 / total | load |
|---|---|---|---|---|
| viz on | 300.1 | 61.8 / 60.2 | 632 / 562 / 1193 | 11.8->15.2 |
| viz off | 323.8 | 79.4 / 67.0 | 577 / 535 / 1113 | 15.2->18.7 |
CPU -6.7 % with viz off (wall is load-dominated: load rose 15->19). Identical best costs/elites. Historical rigid
(v1 bench, load ~8): wall 20.1 -> 17.6 s, CPU/sim 0.275 -> 0.245 s.

## CPU per 90 s scenario (task CPU / genomes / 3 scenarios, all ladder runs, 3 seeds)
rigid 0.169-0.234 s (FD 0.2); reduced 1.04-1.18 s (FD 1.2); full 0.89-1.48 s in ladders, 1.30-1.50 s full-only viz off,
1.46-1.65 s viz on (FD 1.7). Margin-gate fails are not flown, so averages include cheap genomes. reduced ~= full in cost.

## Ladders (best full-scored cost per seed s1/s2/s3; mean; delta vs full-only same seed; feasible rate of full-scored; task CPU; CPU speedup vs full-only)
| case | c172x best | mean (Δ) | feas | CPU s | CPU x | 737 best | mean (Δ) | feas | CPU s | CPU x |
|---|---|---|---|---|---|---|---|---|---|---|
| full-only | .25680/.28892/.39256 | .31276 | .78/.73/.70 | 577/545/556 | 1 | .20478/.16208/.18919 | .18535 | .77/.80/.74 | 535/557/501 | 1 |
| rigid->full | .28169/.35286/.38664 | .34040 (+.0276) | .90/.78/.95 | 145/133/155 | 3.6-4.1 | .19142/.19902/.19671 | .19572 (+.0104) | .89/.97/.92 | 209/213/220 | 2.3-2.6 |
| reduced->full | .30551/.29944/.34160 | .31552 (+.0028) | 1/1/1 | 525/522/513 | 1.04-1.10 | .22157/.18034/.19775 | .19989 (+.0145) | .94/.97/.97 | 556/561/576 | 0.87-0.99 |
| rigid->reduced->full | .28263/.27926/.34018 | .30069 (-.0121) | 1/1/1 | 245/275/276 | 1.98-2.36 | .26762/.20012/.22306 | .23027 (+.0449) | .94/1/.91 | 427/436/425 | 1.18-1.28 |
Batch-wall speedup vs full-only viz on (seed 1, both aircraft): full viz off 0.93x, rigid->full 4.74x, reduced->full 1.99x,
rigid->reduced->full 2.84x (CPU 1.07 / 3.37 / 1.10 / 1.78x). Seeds 2/3 vs full-only viz off: CPU 3.18/2.82x, 1.02/0.97x, 1.55/1.51x.
Elite set vs full-only: in the actual runs final elites differ in every ladder/seed (overlap 0; GA trajectories diverge).
Counterfactual on identical full-only populations (8 gens): elite set differs c172x rigid->full 1/8, reduced->full 2/8,
rigid->reduced->full 1/8; 737 1/8, 0/8, 0/8 (mean overlap 0.88-1.00).

## Spearman (ours vs FD's)
Whole population (32, counterfactual, 8 gens): c172x reduced vs full median +0.94 (range +0.67..+0.99), both-ok +0.96
(FD 0.94; 0.91-0.92 excl. hard fails); 737 median +0.69 (+0.41..+0.92), both-ok +0.87 (FD 0.84-0.87; 0.59-0.62 excl.).
Rigid vs full over the population: c172x +0.85, 737 +0.62 median (FD ~0): ours vary gains too, which rigid sees.
Logged per-gen on the re-scored top set (history.jsonl, 3 seeds x 4 gens): rigid vs full c172x med +0.50 (-0.60..+1.00),
737 +0.36 (-0.33..+0.86), 737 in 3-level -0.04: ~0 / unreliable, as FD found. reduced vs full c172x +0.85 / +0.90, 737 +0.24 / +0.18.
Reduced passes but full fails: c172x 21 of 256, 737 30 of 256.

## Recommendation (small budget, 3 seeds; deltas are within seed noise)
* c172x: rigid->reduced->full (mean best -0.012 vs full-only, 100 % feasible full-scored, ~2.0-2.4x less CPU);
  rigid->full for quick exploration (3.6-4.1x CPU, +0.028, feasibility 0.78-0.95).
* 737 (and by extension T38/f16, not benchmarked): full-only for authoritative runs; rigid->full (>= 25 % re-score)
  as the fast mode (2.3-2.6x CPU, +0.010). Don't use reduced in swept-wing ladders: no CPU saving and weak ranking (737 re-scored-set median +0.18..+0.24).

## Caveat added 08:03 PT
FD changed flex v2 after the benchmark (INTERFACE_v2 section 12: limit-load sizing terms, minimum gauge, margin shaping;
flexeval/flexbody edited 07:45-07:56 PT, TERM_KEYS 16 -> 23). All numbers above are pre-section-12. Our adapter's
TERM_KEYS were updated to the 23 keys (fidelity.py); 80 tests pass against FD's current code. The ladder benchmark
was not rerun at the new FD version.
