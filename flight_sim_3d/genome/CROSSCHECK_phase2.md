# Cross-check: genome `phase2_flex` vs Evolution Runner's Phase 2 evaluator (2026-10-06, 08:40–08:50 MST)

**Result: bit for bit.** For all 3 aircraft, the total cost, every per-scenario cost, all 18 FD structural terms, the
controller terms and the mass fraction are bit-identical. This covers 12 genomes, including envelope, ultimate-load
and margin-gate failures.

## What was compared
- **ER side.** `evolution/fidelity.evaluate_genome(profile, gains, struct, scenarios, "full")`. It is imported from a
  temporary copy of `evolution/*.py` in its own process, with bytecode writing off. `EVOLUTION_FD_DIR` points at
  flight-dynamics/. ER's evaluator is FD's `flexeval.evaluate(fidelity="full")` with ER's `sim.py`.
- **ER inputs.** Profile = the resolved profile of `runs/phase2-smoke-s1`. It is identical to
  `configs/phase2_pilot.json` (field-by-field diff: empty).
- **Scenarios.** ER's `sim.make_scenarios(3, scenario_seed=1, P)`.
- **genome side.** `adapter.load_task("phase2_flex", {"aircraft": ...}).evaluate(gains ∪ struct, make_scenarios(3, 1))`.
  It uses `struct_v2_source "fd"`.
- **Genomes per aircraft:**
  - ER's smoke best.
  - The same gains with FD's baseline structure.
  - One ER-infeasible genome per failure kind in the smoke run.
- Nothing was written into evolution/ or flight-dynamics/. Script: `crosscheck_phase2.py`. Output:
  `runs/crosscheck_phase2_{c172x,T38,737}.json`.

| aircraft | genome | ER cost | genome cost | per-scenario | 18 J_* terms | ER recompute = recorded |
|---|---|---|---|---|---|---|
| c172x | best c172x:g4:r0 | 0.39455837284417933 | same | 3/3 bitwise | 18/18 | yes |
| c172x | best gains + baseline struct | 0.4211725381804851 | same | 3/3 | 18/18 | – |
| c172x | overload c172x:g0:r10 | 1004.4630661469763 | same | 3/3 | 18/18 | yes |
| T38 | best T38:g3:r0 | 0.22428507351607493 | same | 3/3 | 18/18 | yes |
| T38 | best gains + baseline struct | 0.25206494284334585 | same | 3/3 | 18/18 | – |
| T38 | diverged T38:g1:r14 | 1211.1037992413178 | same | 3/3 | 18/18 | yes |
| T38 | structural_ultimate_empennage T38:g1:r15 | 2000.0783755157688 | same | 3/3 | 18/18 | yes |
| 737 | best 737:g4:r0 | 0.27524389750858025 | same | 3/3 | 18/18 | yes |
| 737 | best gains + baseline struct | 0.2237963059850432 | same | 3/3 | 18/18 | – |
| 737 | overload 737:g0:r15 | 1887.93356868726 | same | 3/3 | 18/18 | yes |
| 737 | flutter (gate) 737:g1:r15 | 2000.0 | same | 3/3 | 18/18 | yes |

## Also checked
- **Gene order, bounds and encoding.** The 20 genes (8 controller, then FD's 12 in `gene_schema` order) have
  identical names, min, max and kind. This includes ki_alt [1e-6, 0.5] log0 on c172x and [3.04e-7, 0.5] on T38 (737
  scaled likewise).
  - Encoding the physical genome reproduces ER's `genome_norm` to 3.3e-16, which is log round-off only.
  - Both sides use the same decode formula.
  - ER clamps decoded struct values to FD's range. At u = 0 and u = 1 our decode is already inside the range, so the
    clamp changes nothing.
- **Scenarios.** Seeds, winds, gusts, steps, ramp 600 fpm and 0.1 g are identical (diff empty).
- **Weights and aggregation.**
  - Controller weights: track 1, effort 2, comfort 0.05, heading 0.01.
  - Structural: FD's StructWeightsV2 defaults.
  - Per-scenario cost = sim cost + Σ response terms + pre-flight sum, in flexeval's order.
  - Total = mean over scenarios. On a gate fail the genome is not flown and the cost is 2000.
- **Init.** Both use σ 0.10, baseline mode, structure block only, and the same algorithm (ER ported
  `init_pop.generation_zero`).
- **Regression test.** `tests/test_phase2_flex.py::test_fd_mode_reproduces_evolution_phase2_smoke_costs_bitwise`
  re-scores ER's c172x best from their genomes.jsonl (read-only) and pins the cost, per-scenario costs, J_* terms and
  controller terms.

## Differences that do not affect the cost
- **Reporting on infeasible genomes.**
  - Controller terms: ER reports the mean over the scenarios that finished ok. genome reports NaN once any scenario
    failed (our convention). The costs are identical.
  - Gate-fail status labels: ER says `flutter`, genome says `aeroelastic_flutter`.
- **Search procedure (not the evaluator).** ER's pilot screens with multi-fidelity ladders: c172x rigid → reduced →
  full, T38/737 rigid → full with ≥ 25 % re-scored at full. genome's evolve runs full fidelity only. Full-fidelity
  costs are comparable; rigid or reduced screening costs are not.
- **GA stream.** I did not check whether ER's GA and genome's evolve.py draw the same generation 0 or offspring for a
  given seed. Only the evaluator was compared.

## P2.5 note (2026-10-06 ~14:00 MST)
FD's P2.5 raised NSM floors to 1.0–1.25 and added `J_wing_tip_bm_limit`. Pre-P2.5 ER smoke / pilot genomes with
`wing_nsm_* < 1.0` no longer decode under current FD (ValueError). Phase 2 pilots stay as-is on the previous post-mass
`model_version` pins; genome records post_p25 and warns (does not raise) if FD still reports the previous set.
Re-score for a joint smoke needs P2.5-valid genomes (nsm ≥ 1.0). Reference: `runs/p25_tip_verify_c172x.json`
(baseline tip=0, soft taper4=0.75 tip≈0.017509318343313852; `struct_v2_source=fd`).

## P3-A1 tip-verify (2026-10-06 ~17:45 PT)
FD published `full_a1` (`flexeval_a1` / FlexBodyModelA1; c172x pin `full_a1:flexv2a1:36fb4f5a`). Genome re-flew the
same tip-verify pair via `evolution.fidelity.evaluate_genome` — A1 totals/tip match ER bit-for-bit; full still matches
this file's P2.5 reference. Details: `CROSSCHECK_p3a1.md`, `runs/p3a1_tip_verify_c172x.json`, script
`p3a1_tip_verify.py`.
