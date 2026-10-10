# STATUS_P4_s2: phase4-smoke-s2 (COMPLETE)

## Summary
- s2 = s1 settings + pass-rate gate (>=80% for 2 gens, late fallback OFF) + run_seed_v2 + ring_course/1.2 (per-ring timeout). 16 pop x 6 gens, 4 aircraft, K=4, 8 hold-out courses, 5 workers.
- **run_seed 1333430692384693154**. Wall 20 129 s (T38 alone 16 245 s because the box was at load ~55 on 8 cores), child CPU 45 821 s.
- The gate held T38 and f16 on medium for g2-g5 (s1's floor pushed them to hard at g4, pass 0.000). c172x reached hard (hold-out pass 1.000), 737 reached hard (hold-out pass 0.742). T38 and f16 never reached the 80% gate (best 0.033 and 0.017).
- Replay check (fresh process, T38 g5 rank 0, hold-out seed 12160044683718057565): logged = file = replayed cost **1.457678083082341**, bit-identical, gates identical. All 8 exported trajectories validate (0 errors) and match the log.
- Suite: **215 passed with EVOLUTION_FD_DIR set** (255 s) and **215 passed with it unset** (224 s).

## Setup
Run `runs/phase4-smoke-s2`, launched 05:08:24 PT from snapshot `/workspace/er_smoke_code_p4_s2` (old 04:55 snapshot kept as `er_smoke_code_p4_s2_stale0455`). OMP/OPENBLAS/MKL_NUM_THREADS=1. Differences from s1:
- `rings.curriculum.mode = pass_rate_gate`: advance one stage only after the gen-best passes >= 80% of rings for 2 gens in a row. `late_fallback` (by-generation floor, rule generation_thirds) exists but is `{"enabled": false}`.
- `rings.seed_scheme = run_seed_v2`; run_seed from os.urandom(8), recorded in run.json and config.json (override with `--run-seed`). Course seeds from Sim Bridge `ring_course.course_seed` (ring_course/1.2). All members, elites included, are re-flown on K=4 fresh courses each gen; hold-out seeds fixed per run; all seeds in `seeds.jsonl`.
- Scoring with ring_course/1.2 per-ring timeout (3x leg time, counts as a miss); cache key has course=ring_course/1.2.

## Stage and best train pass per generation (g0..g5; E=easy M=medium H=hard)
| ac | s1 | s2 |
|---|---|---|
| c172x | E1.00 E1.00 M1.00 M1.00 H1.00 H1.00 | E1.00 E1.00 M1.00 M1.00 H1.00 H1.00 |
| T38 | E1.00 E1.00 M.017 M.000 H.000 H.000 | E1.00 E1.00 M.017 M.000 M.033 M.000 |
| 737 | E1.00 E1.00 M1.00 M1.00 H.633 H.633 | E1.00 E1.00 M.983 M1.00 H.783 H.833 |
| f16 | E.983 E.883 M.033 M.017 H.000 H.000 | E.983 E.950 M.000 M.017 M.000 M.000 |

Best cost g5 (s2): c172x 0.1061, T38 1.5459, 737 0.5946, f16 1.6034. 737 promoted to hard at g4 after g2 .983 and g3 1.000.

## Hold-out (8 fixed courses, g5 best) vs train
| ac | train pass | hold-out pass | train cost | hold-out cost | hold-out minus train |
|---|---|---|---|---|---|
| c172x | 1.000 | 1.000 | 0.1061 | 0.1088 | +0.0027 |
| T38 | 0.000 | 0.033 | 1.5459 | 1.5120 | -0.0339 |
| 737 | 0.833 | 0.742 | 0.5946 | 0.8904 | +0.2958 |
| f16 | 0.000 | 0.000 | 1.6034 | 1.5980 | -0.0055 |

Files: `runs/phase4-smoke-s2/{run.json,seeds.jsonl,summary.json,trajectories/}`, `analysis/phase4_smoke_s2_export.json`, `analysis/phase4_smoke_s2_replay.json`, `/workspace/p4_s2_finish.log`.
