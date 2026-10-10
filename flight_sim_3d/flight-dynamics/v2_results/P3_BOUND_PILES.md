# P3 bound piles: FD verdicts (2026-10-07, read-only re-flies)
Method: _scratch/p3_bounds/bound_study.py re-flies the best genome at FULL fidelity (rank-0 full_a1_b1 / full_a1_b2a row of the last gen)
through evolution.eval.evaluate on the run's own frozen pin (EVOLUTION_FD_DIR=_fd_pin_p3b1r1 / _fd_pin_p3b2a), with the gene's
decode range widened in memory only (no file edits). The unmodified points reproduce the run rows bit for bit. Cost = track + 2 effort
+ 0.05 comfort + 0.01 heading + sum J_* + J_energy (residual <= 1e-16). Raw data: v2_results/p3_bound_piles_summary.json, _scratch/p3_bounds/*.jsonl.

Note: the lowest-cost rows in the pilot genomes.jsonl are RIGID-screen rows (shape-blind, no struct terms). Use rank/fidelity, not min cost.

## Common mechanism (Evolution side)
ga.shape_perturb CLIPS a Gaussian step with sigma = 0.25 x half-range. On a gene whose full-fidelity slope is ~1e-5 per unit,
the clipped tail makes the bound absorbing, so the population piles there by drift. About 72% of each generation is scored only by the
shape-blind rigid screen, which weakens selection further. Proposed ER fix: reflect instead of clip in shape_perturb (and BLX crossover).

| gene / case | d cost inside -> beyond bound (mean of 3 scen.) | driver | verdict | recommendation |
|---|---|---|---|---|
| wing_dihedral_delta_deg -> 0 (B2a smoke c172x/T38/737) | T38, 737: <= 1e-5 over [-1, 2] (flat). c172x: +0.00013 per deg; -2 deg gives -0.00048 (track) | No lateral or sideslip scenario, so dihedral has no stability benefit. c172x pays a small track/heading-coupling penalty that is monotone in dihedral (the same one-sided credit that disabled anhedral). Identity init sits ON the lower bound (default 0 = lo) and clipped steps return to 0. | model artifact (missing physics) + clip at an identity-on-bound gene | KEEP [0, +3] (737 +2); do NOT widen to anhedral. Lock dihedral at 0 until a lateral scenario (Phase 4 rings) plus a Cl_beta/spiral/Dutch-roll guard exist, then reopen symmetric +-3 with the identity mid-range. |
| 737 sweep_qc -5 (s1) | -5 -> -8: -0.00027 (0.2%); -5 -> -3: +0.00016 | effort falls as sweep goes forward (AC shifts forward, lower static margin, less elevator work); comfort and bm_rms push back weakly | loophole (unpenalised stability-margin cut via the AC-shift caveat), tiny | KEEP +-5, do not widen. Future FD rev: a static-margin/handling floor (gate or hinge on SM below baseline - x%). Seed 2 best is -4.28, so the effect is weak. |
| T38 sweep_qc +5 (s2) | optimum ~4.5 (interior); 5 -> 7: +0.00008; 5 -> 2: +0.0029 (J_flutter_margin) | flutter margin below ~4 deg; flat above | physical, flat-topped; pile = drift + clip | KEEP +-5 |
| c172x sweep_qc +5 (s2) | sharp dip only at 4.99-5.01: scenario c172x:s2 steps 0.3436 -> 0.3337 -> 0.3388 within 0.1 deg (discontinuous ~0.01); every other point 2-8 deg is +0.004 to +0.012 | scenario-2 roughness (discontinuous ~0.01 step, see below) | artifact (roughness); +5 deg aft sweep on a C172 is not a real optimum | KEEP +-5 for now (narrowing c172x to +-3 is OK); fix the s2 discontinuity first (FD to bisect which event/limiter flips in c172x:s2, likely a saturation/limit switch) |
| wing_chord_taper_3 1.05 (s2 T38, 737) | 1.05 -> 1.00: +0.0010 / +0.0016 (J_wing_tip_bm_limit); 1.05 -> 1.10: +0.0016 / +0.0027 (J_flutter_margin) | two hinges meet at about 1.05-1.06: the tip-BM allowable (local EI ~ chord) wants more tip chord; flutter margin wants less | physical constrained optimum, interior (T38 1.07 is only +0.00008) | KEEP 0.85-1.05 (optionally widen hi to 1.10 so the optimum is not on the edge; gains <= 1e-4) |
| c172x twist_mid -2 (s1) | flat: +-3e-5 over [-3, -1]; min near -1.9 | none (sub-noise) | neutral; pile = drift + clip | KEEP [-2, 1]; no widening is justified |

## c172x scenario-2 roughness
Yes, it explains the c172x sweep pile. Cost of c172x:s2 is piecewise with ~0.005-0.01 jumps on a 0.05 deg sweep scale, while s0 and s1
are smooth. It does not explain the dihedral pile (smooth monotone, s1 track) or the twist_mid pile (flat).

## Proposed changes (none applied; frozen A1/B1/B2a untouched)
1. ER: reflecting bounds in shape_perturb/BLX. Report a "fraction at bound" diagnostic only for genes with |d cost| > noise.
2. FD next rev (B2b/r2): dihedral locked until lateral scenarios exist; static-margin guard for sweep/AC shift; bisect the
   c172x:s2 discontinuity; optional chord_taper_3 hi 1.10.
