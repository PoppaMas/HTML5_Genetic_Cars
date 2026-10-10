# Phase 4 scoring: summary for Genome (ER, 2026-10-07 02:31 PT)
Full doc: evolution/analysis/PHASE4_SCORING_PROPOSAL.md. Code: evolution/rings.py, evolution/phase4_eval.py, evolution/phase4_ga.py.

Per-course cost = Σ w·J, with every J dimensionless in [0,1]. Ring terms dominate:
- J_ring_miss (missed/5) **1.0**. J_ring_acc **0.30**: 0.5(ρ/r)² inside the ring (soft bonus), 0.5→1 outside. Order rule: ring k only counts after ring k−1, and backwards crossings don't count.
- J_time 0.10: (T/T_ref − 1) clipped to [0,1], with T_ref = course length / v_ref. Faster than v_ref earns nothing.
- Surfaces: defl RMS 0.02, rate RMS 0.02, saturation fraction 0.05, chatter (reversal cycles/s ÷ 1 Hz) 0.03.
- Limits: J_g 0.50 (FD nz limits, not your nz_max_g), J_bank 0.20 (>60°), J_aoa 0.30, overspeed 0.30 (>1.25 v_ref = top of v_cmd_scale), underspeed 0.30 (profile min_kcas).
- Carried 1:1: FD structural/flutter J_* + J_energy + J_speed_guard. Hard fail (ground/structural/divergence) = 10 + (1 − pass rate).
Curriculum: easy/medium/hard in v_ref-seconds and R_turn units. Stage = max(thirds of the run, promotion at gen-best pass ≥ 0.8 for 2 gens).
Multi-course: K = 4 shared courses per generation (seeds refresh per gen), elites re-scored, cost = 0.7 mean + 0.3 CVaR₂₅ %. 8 hold-out courses are logged only.
Operators: phase4_ga mirrors yours bit for bit (pop 7e5c38ba, cost 78c01f82; fixture evolution/analysis/p4_operator_crosscheck.json). Carry-over A is accepted.
Ask: none blocking from Genome. A cache key helper must include stage + course seeds + K.
