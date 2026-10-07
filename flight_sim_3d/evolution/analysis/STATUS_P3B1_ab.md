# STATUS P3-B1 A/B: baseline vs tweaked GA preset (Evolution side), updated 2026-10-07 ~01:00 PT

Every run is 64×60, uses FD B1 r1 (pin `evolution/_fd_pin_p3b1r1`) and 3 scenarios, with the same eval code_sha `6e8239d0e3c8634f`. Baseline = elite 2, whole-block crossover (snapshot `/workspace/er_pilot_code_b1r1`). Tweaked = elite 4, uniform shape crossover, Genome `phase3_b1_x` (snapshot `/workspace/er_pilot_code_b1r1_tweaked`).
- Seed 1: `phase3b1r1-pilot-s1` (3312 s) vs `phase3b1r1-pilot-tweaked-s1` (3098 s).
- Seed 2: `phase3b1r1-pilot-s2` (3841 s) vs `phase3b1r1-pilot-tweaked-s2` (3451 s, exited normally ~00:54 PT, 1 session, no resume).
- Gen 0 is identical between the arms on both seeds. Seed 2: all 192/192 gen-0 rows of genomes.jsonl have the same genome and cost.
- **Seed 3 skipped (decision):** two more runs at ~55–64 min each would have ended around 02:50 PT, past the 02:15 PT target.

## Per-seed bests (rank 0, gen 59, full_a1_b1, from checkpoint best_per_gen); Δ = tweaked − baseline
| seed | aircraft | baseline | tweaked | Δ | Δ % |
|---|---|---|---|---|---|
| 1 | c172x | 0.248063 | 0.240135 | −0.007927 | −3.20 % |
| 1 | T38 | 0.109280 | 0.109360 | +0.000080 | +0.07 % |
| 1 | 737 | 0.135173 | 0.140783 | +0.005610 | +4.15 % |
| 2 | c172x | 0.236544 | 0.240366 | +0.003822 | +1.62 % |
| 2 | T38 | 0.105985 | 0.106374 | +0.000388 | +0.37 % |
| 2 | 737 | 0.130790 | 0.131736 | +0.000946 | +0.72 % |

## Means over 2 seeds
| aircraft | baseline mean | tweaked mean | mean paired Δ | Δ % | seeds where tweaked is better | Phase 2 seed sd |
|---|---|---|---|---|---|---|
| c172x | 0.242303 | 0.240251 | −0.002053 | −0.85 % | 1/2 | 0.01337 |
| T38 | 0.107633 | 0.107867 | +0.000234 | +0.22 % | 0/2 | 0.00321 |
| 737 | 0.132982 | 0.136260 | +0.003278 | +2.46 % | 0/2 | 0.00284 |

Overall sign count: tweaked is better in 1 of 6 (seed, aircraft) pairs and worse in 5.

## Shape contribution at the best
This is the best's cost minus the cost of the same genome with the shape reset to FD's baseline defaults. Own-shape re-flies are bit-identical in every run.
| aircraft | baseline s1 | baseline s2 | baseline mean | tweaked s1 | tweaked s2 | tweaked mean |
|---|---|---|---|---|---|---|
| c172x | −0.007464 | −0.013560 | −0.010512 | −0.006335 | −0.011378 | −0.008857 |
| T38 | −0.007866 | −0.046160 | −0.027013 | −0.019638 | −0.031048 | −0.025343 |
| 737 | −0.000710 | −0.005418 | −0.003064 | −0.017429 | −0.000889 | −0.009159 |

The seed-1 finding that uniform crossover makes the shape block do more of the work does not hold up on seed 2. There, the baseline gets more from shape on all three aircraft.

## Seed 2 details (`phase3b1r1_pilot_tweaked_s2_check.json` vs `phase3b1r1_pilot_s2_check.json`)
- Best-so-far is monotone on all aircraft in both arms.
- **Flutter margin (tweaked / baseline):** c172x 1.2311 / 1.1998, T38 1.1998 / 1.2018, 737 1.2136 / 1.1981.
- **Rejected rows over all generations (tweaked / baseline):**
  - c172x: 19 / 21 (overload 15 / 17, diverged 3 / 3, attitude 1 / 1).
  - T38: 7 / 7 (attitude 2, diverged 5; 1 authoritative diverged in each arm).
  - 737: 18 / 12 (overload).
  - geometry_gate: 0 in both arms.
- **Shape genes piled at a bound (final population, ≥ 50 %):**
  - Tweaked: T38 sweep at the ceiling, 97 %.
  - Baseline: c172x sweep ceiling 95 %, T38 sweep ceiling 92 %, T38 chord_taper_3 ceiling 50 %, 737 chord_taper_3 ceiling 86 %, 737 sweep floor 50 %.
- **Own-shape vs baseline-shape refly (tweaked):** c172x 0.240366 vs 0.251744, T38 0.106374 vs 0.137422, 737 0.131736 vs 0.132625.
- **Trajectories:** `validate_traj` exit 0 on 9 files, all headers OK, fd_dir OK (`phase3b1r1_pilot_tweaked_s2_traj_check.json`). Baseline: `phase3b1r1_pilot_s2_traj_check.json`.
- **Replays through `_fd_pin_p3b1r1` (fresh process), T38:g59:r0 scenario T38:s0, both bit-identical vs trajectory and row:**
  - Tweaked: 0.07535579427530573 → `phase3b1r1_replay_check_tweaked_s2.json`.
  - Baseline: 0.07335263462466757 → `phase3b1r1_replay_check_s2.json`.

## Seed 1 details (`phase3b1r1_pilot_tweaked_s1_check.json` vs `phase3b1r1_pilot_s1_check.json`)
- **Shape contribution at the best:** this is the best re-flown with its own shape, minus the same genome with the shape reset to baseline. Own-shape re-flies are bit-identical in both arms.
  | aircraft | tweaked | baseline |
  |---|---|---|
  | c172x | −0.0063 | −0.0075 |
  | T38 | −0.0196 | −0.0079 |
  | 737 | −0.0174 | −0.0007 |
  - Under uniform crossover the shape block does more of the work, especially on T38 and 737.
- **Tweaked best shapes:**
  - c172x: sweep +4.92°, twist_tip −2.51°.
  - T38: twist_mid +1.0 and chord_taper_3 1.05, both at the ceiling. Sweep +3.27°.
  - 737: chord_taper_1 0.945, sweep +1.76°.
- **Rejected rows over all generations (tweaked / baseline):** c172x 23 / 18, T38 12 / 12, 737 33 / 45. There were no gate rejects.
- **Replays through `_fd_pin_p3b1r1` (fresh process), both bit-identical:**
  - Tweaked T38 g59 s0: 0.07474663523979805 → `phase3b1r1_replay_check_tweaked.json`.
  - Baseline T38 g59 s0: 0.07783981426439955 → `phase3b1r1_replay_check.json`, restored by re-running `p3b1r1_replay_check.py runs/phase3b1r1-pilot-s1`. The script now takes `--out`.

## Recommendation for B2
**Use the baseline operators (elite 2, whole-block crossover), not `_x`.**
- Over 2 seeds the tweaked preset is worse on mean for T38 (+0.22 %) and 737 (+2.46 %), worse on both seeds for those two, and only slightly better on c172x (−0.85 %, split 1/2).
- It also gives no consistent gain in shape contribution.

Caveats:
- Only 2 seeds.
- Every mean paired Δ is within one Phase 2 seed sd: c172x 0.002 vs sd 0.01337, T38 0.0002 vs 0.00321, 737 0.0033 vs 0.00284 (≈1.15 sd, the only one near 1 sd).
- So this is "no evidence the tweak helps" rather than "the tweak is worse".
- Seed 3 was skipped for time.
