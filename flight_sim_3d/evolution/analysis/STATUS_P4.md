# Phase 4 ring course: Evolution status (2026-10-07 ~04:35 PT). Local only, nothing pushed.

## Wired
- `evolution/phase4_loop.py` (new): config resolver (strict, `"phase": "phase4_rings"` only), GA loop (Genome operators via
  `phase4_ga`), process pool (FD is not thread-safe), on-disk cache keyed by pin + fidelity + genes + model + stage + course
  seed + K + `ring_course.VERSION` + scoring version + weights, curriculum, hold-out, trajectory export.
- Opt-in hook in `batch.main`: only `--config` files with `"phase": "phase4_rings"` go to `phase4_loop.main`. Nothing else changes for older configs.
- `configs/phase4_smoke.json` (was `.json.draft`): c172x / T38 / 737 / f16 (f16 profile = phase1_hdg `phase1_f16`), pins = FD active.
  `analysis/tweaked_preset_golden.py` and `test_default_options_resolve_exactly_as_before` route Phase 4 configs to
  `phase4_loop.resolve_config`. The 205 golden cases and the strict checks for every other config are unchanged.
- FD pin: `evolution/_fd_pin_p4cs`, a read-only copy of flight-dynamics/ (no `__pycache__` or `_scratch`). FROZEN_P4cs.md5 7/7 OK, FROZEN_B2a.md5 OK,
  FROZEN_A1_B1r1.md5 OK. Live `model_version` equals the pins for all 4 aircraft (the run checks this).
- Course and rules: Sim Bridge `sim_bridge.ring_course` 1.1 (`make_course`, `ring_at`, `score_course`, `train_seeds`/`holdout_seeds`,
  `fd_pos_to_ned`, `isa`). These are called directly, not reimplemented. TAS geometry; T_ref = `course.nominal_time_s`.
- Guidance: Genome's `genome/p4_guidance.make_guidance` + `phase4_rings` genes. ER's `phase4_guidance.py` is an interim law
  used only when `"guidance": "er_interim"`.
- Scoring (`phase4_loop.score_flight`, scoring version er-p4-score/2):
  - Amendment C1–C6 as approved. The changes vs the proposal are in PHASE4_SCORING_PROPOSAL.md §10.
  - KCAS limits: overspeed vs min(1.25·v_ref_kcas, v_max_kcas), checked on FD `v_kcas`; underspeed vs profile min_kcas.
  - Mach vs FD mach_max, using TAS / ISA a(altitude).
  - Bank vs FD course bank 60/75/60/80°; α vs 15/18/13/25°. There is no 45° fail.
  - Surface terms use ACTUAL positions (`ctrl_surfaces <s>_deg` from JSBSim `fcs/*-pos-*`, 120 Hz), never commands. For the f16 these are
    real deflections (`fcs/*-pos-rad`), so no f16 surface weight was zeroed. Surfaces with rate `None` (flap, f16 speedbrake) are skipped.
  - Rim crash off.
- Single source for limits: `rings.fd_limits()` (FD p4_aircraft_limits.json) feeds `rings.scales`, `phase4_eval.BANK_COURSE_DEG`
  and `ALPHA_MAX_DEG`. `rings.scales` v_max is now CAS: min(1.25·v_ref_kcas, v_max_kcas).
- **J_carried keys (1:1):** J_flutter_margin, J_div_margin, J_mass, J_bm_rms, J_bm_peak, J_tip, J_twist, J_reversal_margin,
  J_tail_bm_peak, J_fus_bm_peak, J_smooth, J_wing_bm_limit, J_wing_torque_limit, J_wing_ip_limit, J_wing_tip_bm_limit,
  J_tail_bm_limit, J_fus_bm_limit, J_wing_torque_peak, J_wing_ip_peak, plus J_energy and J_speed_guard.
  J_speed_guard is 0 in fly_course because there is no v_target.

## Tests and bit-for-bit evidence
- Full suite: **209 passed** (04:45 PT; run with EVOLUTION_FD_DIR unset, because test_paths/phase1/golden assume the default FD dir. With EVOLUTION_FD_DIR=_fd_pin_p4cs exported, 8 path tests fail; that is environment-only). Before the Phase 4 tests were added it was 193 passed, and 207 passed at 03:08.
  `tests/test_phase4_loop.py` adds 17 tests:
  - frame: a known NEU crossing passes; unconverted input scores 0;
  - rings.py vs SB score_course on 4 synthetic courses; strict order (`missed_order`);
  - flap rate None; FD limits incl. f16; KCAS/Mach;
  - GENES == Genome; identity hash 192ed803; config/hook; cache key;
  - real FD: the identity genome flies a c172x course with a known pass, bit-identical in 2 fresh processes; known miss with FD demo law
    (0/2 passes); TAS = SB.
- B1 r1 T38 s1 baseline replay: 0.07783981426439955 (exact vs traj and row). Tweaked: 0.07474663523979805 (exact). Both use `_fd_pin_p3b1r1`.
- Detfix B2a smoke subset: 30 rows × 3 scenarios = 90 replays, 0 mismatches (`p4_detfix_subset_replay.json`), incl. c172x g0:r9, g0:r10, g1:r15.
- Operator cross-check vs genome/runs/p4_operator_trace.json (02:46 version): all match, pop 7e5c38ba, cost 78c01f82.
  Identity physical decode is equal on 4 aircraft.

## Cross-check vs Genome's identity K=4 (`p4_genome_identity_crosscheck_<m>.json`, sandbox venv, jsbsim 1.3.1, numpy 2.5.3)
- FD flights from ER's loop and Genome's `fly_one` are **bit-identical** (pos and att) on all 16 courses.
- Passes per course are identical: c172x 15/15/15/15, T38 10/12/12/13, 737 15/14/15/15, f16 10/9/12/11.
- Costs differ by design:
  - ER cost = Genome cost + J_carried (Genome carries {}).
  - SB rules vs legacy ring_crossings.
  - Surface terms are 120 Hz actual positions vs 30 Hz.
- Genome's own evaluator re-run here does not exactly reproduce its files.
  - T38, 737 and f16 differ by ≈1e-10 relative. The likely cause is multi-threaded BLAS; the FD plant is not thread-count invariant (below).
  - c172x differs by 2e-3, because that file dates from 02:44 and predates p4_guidance 03:01 and ring_course 1.1.

## Smoke phase4-smoke-s1
- 16 pop × 6 gens; c172x, T38, 737, f16; K=4 train + 8 hold-out; curriculum by generation thirds, so stage = floor.
- Code snapshot /workspace/er_smoke_code_p4 (SNAPSHOT_SHA256.txt). FD pin _fd_pin_p4cs. BLAS 1 thread.
- First launch 03:09: a worker was OOM-killed at 03:23 (7 workers + 4 cross-check processes + other agents; 15 GB box).
  It was relaunched at 03:29 with 5 workers, reusing the 78 cached course results (deterministic), with genomes.jsonl restarted.
  An earlier launch at 02:56 was stopped by ER for the Mach and limits fix and moved out to /workspace/p4probe.
- Relaunch: wall 3270 s, worker CPU 15 806 s.

| aircraft | stages g0..g5 | train cost (rank 0) | train pass | hold-out cost | hold-out pass | gap (H − T) | mean ρ hold (m) | mean miss (m) | t_last / T_ref (hold) | finished hold | hard-fail courses (all gens) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c172x | E E M M H H | 0.1058 | 1.000 | 0.1072 | 1.000 | +0.0015 | 3.94 | – | 130.0 / ~152 s | 8/8 | 61 structural |
| T38 | E E M M H H | 1.4660 | 0.000 | 1.4647 | 0.000 | −0.0013 | 105.1 | 60.7 | – | 0/8 | 121 structural, 32 ground |
| 737 | E E M M H H | 0.8463 | 0.633 | 0.9538 | 0.600 | +0.1075 | 35.1 | 23.3 | 195.1 / ~164 s | 7/8 | 55 ground, 10 structural |
| f16 | E E M M H H | 1.5091 | 0.000 | 1.5130 | 0.008 | +0.0039 | 75.6 | 44.1 | – | 0/8 | 32 structural, 1 ground |

Gen-best pass rate by gen:
- c172x: 1,1,1,1,1,1
- T38: 1,1,.017,0,0,0
- 737: 1,1,1,1,.633,.633
- f16: .983,.883,.033,.017,0,0

**T38 and f16 fail the medium and hard stages.** The forced stage floor promotes them anyway.

Main train terms of the rank-0 genome:
- c172x: acc .058, defl .115, rate .015, chatter .017, overspeed .001, carried .082.
- 737: miss .367, acc .369, time .071, defl .376, rate .166, sat .108, chatter .142, bank .011, carried .307.

## Trajectories
`evolution/runs/phase4-smoke-s1/trajectories/` holds 8 files (rank 0 of g5, train course 339585700 and hold-out course 1985065222, per aircraft) plus index.json.
- Format: ga-flightsim-traj/2 plus the SB §5 `course` block (NED + centre_enu_m/normal_enu), `gates`, `gate_summary`, `ctrl_surfaces` (30 Hz decimation) and `p4`.
- `validate_traj` and the ring-block checks report 0 errors. The re-flown cost equals the logged cost for all 8.
- Fresh-process replay: T38_phase4-smoke-s1_g5_holdout_1985065222.json, check value **1.4662091848164185**. It is bit-identical vs genomes.jsonl
  and the file, gates are identical, and model_version e7da0a5a matches the pin.

## Spec mismatches / notes for FD, SB and Genome
- Frame: FD `pos` = [N, E, U = altitude MSL] (P4.14). The fly_course docstring says "origin = start". SB NED = (N, E, −U). Converted via `fd_pos_to_ned`, with a test.
- SB spec §3a describes a per-ring timeout (3 × leg / V_ref). `score_course` implements only the course time limit. Genome's guidance does
  apply the per-ring timeout, so guidance and scoring can disagree on which ring is "next" after a timeout.
- FD P4 results depend on the BLAS thread count at about 1e-10 (measured on T38: 0.23497544924795072 vs 0.23497544943415924).
  ER enforces 1 thread, and Genome's cross-check files look multi-threaded.
- FD loads `sim.py` and `configs/phase1.json` relative to its own folder. The pin inside evolution/ needs `fe.PHASE1_CONFIG` and `load_sim` defaults patched in-process (done in `phase4_loop.mods`).
- fly_course exports no Mach, so ER computes it from TAS and ISA.
- c172x Genome K4 file predates ring_course 1.1.

## Blockers / next
- T38 and f16 score 0 passes from the medium stage on. Curriculum promotion by generation floor alone outpaces them. Whether to gate on pass rate only is Corleone's call; the design is unchanged here.
- Architecture v1: SB §0 says there is no Architecture v1 document on the box; v1 comes from lane-lock records. C1–C6 are approved as amended.
