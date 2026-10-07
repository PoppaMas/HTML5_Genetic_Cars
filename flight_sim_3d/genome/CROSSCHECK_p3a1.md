# Cross-check: P3-A1 tip-verify (genome vs Evolution Runner, 2026-10-06 ~17:45 PT)

**Result: bit for bit.** Genome re-flew the P2.5 tip-verify pair on fidelity `full_a1` through
`evolution.fidelity.evaluate_genome` (same setup as `evolution/analysis/p3a1_tip_verify.py`). Totals,
per-scenario costs, and `J_wing_tip_bm_limit` match ER's `p3a1_tip_verify_evolution.json`. Full fidelity
still matches `genome/runs/p25_tip_verify_c172x.json` bit-for-bit.

## Method
- **Genomes:** `genome/runs/p25_tip_verify_c172x.json` — baseline (all struct = 1.0 / damping 0.02) and
  `soft_tip_taper4_0.75` (`wing_ei_taper_4 = 0.75`), same controller gains.
- **Profile / scenarios:** `evolution/configs/phase2_smoke_p25.json` c172x resolved profile; 3 scenarios;
  `scenario_seed = 1`.
- **Evaluator:** `evolution.fidelity.evaluate_genome(..., fidelity, reduced_gate=0.9)` — read-only import;
  nothing written into evolution/ or flight-dynamics/.
- **Script / output:** `genome/p3a1_tip_verify.py` → `genome/runs/p3a1_tip_verify_c172x.json`.
- **Pins:** `flight-dynamics/v2_results/model_versions_post_p3a1.json` c172x
  `full_a1:flexv2a1:36fb4f5a` (full unchanged `full:flexv2:e11b8214`).

## A1 tip-verify table

| case | fidelity | cost | J_wing_tip_bm_limit | model_version | vs ER / p25 |
|---|---|---|---|---|---|
| baseline | full_a1 | 0.2419400885448951 | 0.0 | full_a1:flexv2a1:36fb4f5a | bit_identical vs ER |
| soft_tip_taper4_0.75 | full_a1 | 0.25879473336084763 | 0.011874078070487388 | full_a1:flexv2a1:36fb4f5a | bit_identical vs ER |
| baseline | full | 0.24220696133846464 | 0.0 | full:flexv2:e11b8214 | bit_identical vs genome p25 |
| soft_tip_taper4_0.75 | full | 0.2657981420373094 | 0.017509318343313852 | full:flexv2:e11b8214 | bit_identical vs genome p25 |

Δ(A1 − full): baseline cost ≈ −2.67e-4 (tip 0); soft tip cost ≈ −7.00e-3, tip ≈ −5.64e-3 (station-exact tip BM).

## Notes
- Same 12 structure genes on A1 as P2.5. Genome `phase2_flex` / `fd_bridge` defaults unchanged (still post_p25
  full pins); this verify uses ER's fidelity path opt-in, not the default adapter.
- Phase 2 pilots frozen. No commit/push.
