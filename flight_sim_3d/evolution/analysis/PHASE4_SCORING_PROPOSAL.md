# Phase 4 ring course: Evolution rules and scoring (proposal, 2026-10-07 ~02:30 PT)

Status: DRAFT. Evaluator code and tests exist (`evolution/rings.py`, `evolution/phase4_eval.py`,
`tests/test_phase4_rings.py`, `configs/phase4_smoke.json`). No loop hook yet; no smoke run.
**`PHASE4_RINGS_SPEC.md` did not exist anywhere on the box at writing time** (searched `/`), and there are no
FD or Genome Phase 4 notes either. Everything below aligns to what Sim Bridge/FD/Genome must provide (section 8).

## 1. Normalisation (matches existing costs)
The existing costs are a sum of dimensionless, O(0.01–1) terms. Baseline B1/B2 genome costs are 0.11–0.25 (T38 seed-1 baseline 0.0778).
FD's 24 `TERM_KEYS` stay untouched. Phase 4 terms live in their own `P4_TERM_KEYS`, as the energy terms do.
Every term is dimensionless in [0, 1] per course (except hard fail), and cost = Σ w_k J_k. Ring terms can reach
1.3, which is 5–10× everything else combined at nominal, so they dominate.

## 2. Term table (per course)
| term | definition | w | rationale |
|---|---|---|---|
| J_ring_miss | missed rings / N | **1.00** | primary objective. One miss (0.2) outweighs a whole B2 baseline cost |
| J_ring_acc | mean q_i. Pass: 0.5(ρ/r)²; miss: 0.5+0.25·min(ρ/r−1, 2) | **0.30** | soft pass bonus: continuous at the rim, so it gives a gradient towards centre and towards a near miss |
| order rule | ring k is only searched after ring k−1's crossing time; backwards crossings don't count | (in miss) | no "fly through the course backwards" exploits |
| J_time | clip(T_last/T_ref − 1, 0, 1), with T_ref = polyline length / v_ref | 0.10 | agility. Clamped at 0, so going faster than v_ref earns nothing, which means no overspeed reward |
| J_defl_rms | mean over surfaces of RMS(δ / max\|limit\|) | 0.02 | small effort penalty, ~the old `effort` scale |
| J_rate_rms | mean RMS(δ̇ / rate_max) | 0.02 | actuator wear |
| J_sat | time fraction with δ ≥ 98 % of a limit | 0.05 | saturation means no control margin |
| J_chatter | rate-sign reversals (\|δ̇\| > 5 % rate_max) in cycles/s ÷ 1 Hz, capped at 1 | 0.03 | kills bang-bang/limit-cycle controllers |
| J_g | time-mean of overshoot past nz limits, normalised by the limit (c172x −1/3.8, T38 −3/7.33, 737 −1/2.5) | 0.50 | strong but soft. FD's structural terms already price loads |
| J_bank | time-mean of max(0, \|φ\|−60°)/60° | 0.20 | course bank limit (FD to confirm per model) |
| J_aoa | time-mean of max(0, α−α_max)/α_max | 0.30 | stall guard. α_max is a placeholder (15/18/12°) until FD exports it |
| J_overspeed | time-mean of max(0, V−1.25 v_ref)/v_max | 0.30 | stops dive-for-speed. v_max = 1.25 v_ref until FD gives VNE/MMO |
| J_underspeed | time-mean of max(0, v_min−V)/v_min (55/180/195 kt) | 0.30 | mirrors the existing min_kcas |
| J_carried | Σ of FD's structural/flutter J_* + J_energy + J_speed_guard, unchanged | 1.00 | carried over 1:1 with the weights from B2a |
| hard fail | ground impact (AGL ≤ 0), structural failure, divergence (non-finite) → cost = 10 + (1 − pass_rate) | — | dominates everything, with a gradient by progress |

Nominal good genome: ~0.02–0.1 non-ring + 0–0.15 accuracy. One missed ring adds ≥0.2 + 0.15.

## 3. Curriculum (aircraft-relative geometry, `rings.STAGES`)
| stage | spacing | lateral offset | vertical offset | ring radius |
|---|---|---|---|---|
| easy | 20 s at v_ref | ±0.05 R_turn | ±2 % spacing | 0.60 s · v_ref |
| medium | 14 s | ±0.15 R_turn | ±5 % | 0.40 s |
| hard | 10 s | ±0.30 R_turn | ±8 % | 0.25 s |

R_turn = v_ref²/(g tan 30°). Stage = max(generation floor in thirds of the run, pass-rate promotion). Promotion
happens when the gen-best mean pass rate is ≥ 0.8 for 2 consecutive gens. It never demotes and is deterministic
from history (`curriculum_stage`). Costs change scale when the stage changes, so elites are re-scored on the new
stage, and the cache key must include the stage and course seeds.

## 4. Multi-course scoring
- K = 4 courses per genome per generation. Every genome in a generation flies the same K courses (fair ranking),
  with seeds `sha256("p4ring|T|run_seed|gen|k")`. The courses refresh every generation, which prevents memorising one course.
- Elites are re-scored on the new courses each generation (no stale elite costs, which fixes the "identical cost at g29/g59" issue for P4).
- Aggregate = 0.7·mean + 0.3·CVaR₀.₂₅ (the worst ⌈K/4⌉ course), so good-on-average but crashes-sometimes genomes lose.
  It uses an order-free fsum, so it is deterministic under any evaluation order (tested).
- Hold-out: 8 fixed courses (`holdout=True` seed space, gen-independent) score only the gen best and final
  elites. They are logged but never used for selection, so they are the overfitting check.

## 5. Per-aircraft scaling (real profile numbers, configs/phase3b2a_smoke.json)
| | v_ref | v_min | nz | R_turn @30° | ring radius easy/hard | spacing easy/hard |
|---|---|---|---|---|---|---|
| c172x | 100 kt (51.4 m/s) | 55 kt | −1/3.8 | 467 m | 31/13 m | 1.03/0.51 km |
| T38 | 300 kt (154 m/s) | 180 kt | −3/7.33 | 4.20 km | 93/39 m | 3.09/1.54 km |
| 737 | 250 kt (129 m/s) | 195 kt | −1/2.5 | 2.92 km | 77/32 m | 2.57/1.29 km |

All the geometry is in time and turn-radius units, so the stages are equally hard across aircraft and the costs compare.

## 6. Files
`evolution/rings.py` (crossing geometry, scaling, curriculum, seeds, reference course generator for tests only),
`evolution/phase4_eval.py` (`score_course`, `surface_terms`, `aggregate_courses`, `P4_TERM_KEYS`, `WEIGHTS`),
`configs/phase4_smoke.json.draft` (draft; a `.json` name would trip test_tweaked_preset, which runs batch.resolve_config on every configs/*.json and rejects the new `phase` key. It is renamed to .json when the hook lands), `tests/test_phase4_rings.py`.
Loop hook (step 4): planned as a new fidelity `full_a1_b2a_p4`, dispatched from a new module
`evolution/phase4_loop.py`, plus a ≤5-line opt-in branch in batch/ga that is reached only when `cfg["phase"] == "phase4_rings"`.
Old presets never import phase4_*. It is deferred until the FD trajectory interface below exists, because wiring
against a guessed signature would just have to be redone.

## 7. Open items
v_max (VNE/MMO), α_max per model and the course bank limit are placeholders. FD should supply real values.

## 8. Interfaces needed
**Sim Bridge (course):**
- `make_course(model: str, stage: str, seed: int, n_rings: int = 5) -> list[dict]`, where each ring is
  `{"centre": [x,y,z] m (world, z up), "normal": unit [x,y,z] (direction of travel), "radius": m}` plus the start state
  `{"pos": [x,y,z] m, "heading_deg", "alt_ft", "kcas"}`. It must be deterministic in seed, and stages are named easy/medium/hard (or give a mapping).
- "Rolling sets of 5": confirm whether the next set spawns on pass/timeout and give `next_set(prev_rings, seed, idx)`.
  Otherwise we score fixed 5-ring courses.
- The frame convention (NED vs ENU, origin) and how ring data is written to trajectories (`rings` block) so replays match.

**FD (surfaces + sim):**
- `fly_course(profile_d, gains, struct, shape, surfaces_genome, course, fidelity="full_a1_b2a_p4") -> dict` with arrays
  `t, pos[N,3] m, att[N,3] rad, v m/s (KCAS), nz g, alpha rad, agl m`, `surfaces: {name: deg[N]}`,
  `surface_limits: {name: (min_deg, max_deg, rate_max_dps)}`, `status` ∈ {ok, ground, structural_failure, diverged, …},
  `struct_failed: bool`, `terms` (the 24 TERM_KEYS) and `energy` (fd-energy/2).
- A guidance law, or confirmation that the guidance law is Genome's (ring → attitude commands).
- Per model: α_stall, VNE/MMO, course bank limit, and a pin file + checksums as for B2a.

**Genome (chromosome): LANDED 02:23 PT** (genome/phase4_rings.py, PHASE4_RINGS_GENOME.md). Evolution mirrors it in
`evolution/phase4_ga.py`: the 29 gene names, ranges and scales (guidance 12 | inner_loop 11 | mixing 6), `decode(u)`, and operators
built from ga.py's own flat_rank_select / crossover_blocks (rng.random(3)) / mutate gauss 0.15/0.08, with elite 2.
The fixed-seed trace matches bit for bit: pop 7e5c38ba…, cost 78c01f82…, all 5 gens plus the gen0/gen1 arrays
(`analysis/p4_operator_crosscheck.json`, test `test_phase4_operator_trace_matches_genome`). identity_u decodes to every default.
Scoring couplings with the genome: J_g / J_bank use FD's aircraft limits, not the genome's nz_max_g / bank_max_deg
(those are command clamps, so a genome can't relax its own penalty). v_cmd_scale up to 1.25 × v_ref equals the overspeed
threshold exactly, so the max gene sits on the edge with no reward from J_time. Carry-over A ("none") means J_carried
is FD's baseline-structure terms only. Still needed:
- A `phase4` preset with the gene list and groups (guidance gains, surface/actuator gains), plus `decode_phase4(genes, model) -> dict`
  for the `fly_course` kwargs, `identity_u(model)`, and the cache key inputs. Note that the cache key must include
  stage, course seeds and K.

## 9. Smoke plan / ETA
After all three land: ~30–45 min to wire `phase4_loop.py`, the hook and the replay/golden checks, then a 16×6 smoke
on 3 aircraft at K = 4 (the cost is ~K× a B2a eval per genome, so expect roughly 4× the 4-min B2a smoke, ≈15–20 min).

## 10. As implemented after the Phase 4 amendment (approved by Corleone 2026-10-07 02:45 PT; the amendment wins)
Spec: `flight_sim_3d/PHASE4_RINGS_SPEC.md` §0 C1–C6 (no Architecture v1 document exists on the box; v1 = the team's lane-lock
records, per SB §0). Code: `evolution/phase4_loop.py` (+ `phase4_eval.py`, `phase4_ga.py`, batch opt-in hook).
- **C1 cost form:** weighted terms (§2) + hard fail (10 + J_ring_miss), not v1's lexicographic gatesHit ≫ finish ≫ time ≫ crash 0.25.
- **C2 envelope:** per-aircraft V_ref (SB `AIRCRAFT`: 100/300/250/350 KCAS) and FD `p4_aircraft_limits.json`:
  α_stall 15/18/13/25°, bank (course) 60/75/60/80°, v_max = min(1.25·V_ref, VNE/VMO) = 125/375/312.5/437.5 KCAS
  (Mach cap 0.90/0.84 not scored yet: fly_course exports no Mach), nz = FD n_profile, v_min = profile min_kcas.
  The old 45° attitude fail does not exist in fly_course; bank is scored against the course bank only.
- **C3:** K = 4 train courses per generation shared by the whole population (`ring_course.train_seeds`) + 8 fixed hold-out
  (`holdout_seeds`), reported for the final best only, never used for selection.
- **C4:** closed-loop guidance = Genome's `genome/p4_guidance.make_guidance` on Genome's 29-gene chromosome.
- **C5:** rim crash OFF (`rim_tube_m = 0`), so rim crash is never a hard fail in the smoke.
- **C6:** no reject/resample; course is SB's bounded generator.
Changes vs §2–§4 of this proposal:
- Ring rules (window of 5, strict order incl. `missed_order`, time limit `t_limit = 1.5 × nominal_time`, capture 4 r) are
  **Sim Bridge's `ring_course.score_course`** (called, not duplicated). M = 15. J_ring_miss = (M − passes)/M.
- J_ring_acc over all M rings: flown crossings use ρ/r; `missed_order` / `missed_time` / unreached rings count at the cap (ρ/r = 3).
- J_time: T_ref = `course.nominal_time_s`; finished → clip(t_last/T_ref − 1, 0, 1); not finished → 1.
- Surface terms use the **actual** surface positions from FD `ctrl_surfaces` (`<surf>_deg`, 120 Hz) and their rates vs
  `surface_limits` (never commands; f16 FBW commands are rate/g demands). Surfaces whose rate limit is `None` (flaps) are
  excluded from all surface terms (fixes the divide-by-None).
- Flight duration = `time_limit_s` (same as Genome's evaluator).
- Single-threaded BLAS is required (FD flex plant results change in the ~10th digit with thread count).
- f16 added (4 aircraft).
**J_carried (exact list, each 1:1):** FD's structural/aeroelastic subset of the 24 TERM_KEYS (all `J_*` keys; the rigid
track/effort/comfort/heading/hold are excluded):
`J_flutter_margin, J_div_margin, J_mass, J_bm_rms, J_bm_peak, J_tip, J_twist, J_reversal_margin, J_tail_bm_peak,
J_fus_bm_peak, J_smooth, J_wing_bm_limit, J_wing_torque_limit, J_wing_ip_limit, J_wing_tip_bm_limit, J_tail_bm_limit,
J_fus_bm_limit, J_wing_torque_peak, J_wing_ip_peak`
+ Evolution's `J_energy = w_E·max(0, energy_drag_increment)` and `J_speed_guard = w_E·3·max(0, speed_deficit_kts_mean − 2)/v_target_kcas`
(w_E = fidelity.ENERGY_W; J_speed_guard = 0 when FD's energy block has no v_target, which is the case in fly_course).
