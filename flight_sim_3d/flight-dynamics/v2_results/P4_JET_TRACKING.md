# P4 jet tracking: why T38 / f16 could not fly medium rings, and the r1 fix (FD, 2026-10-10)

Evidence: phase4-smoke-s2 hold-out (T38 medium 3 %, f16 medium 0 %, J_chatter 0.34, J_rate_rms 0.28-0.42) re-flown through
`fly_course` with the s2 guidance snapshot (`the s2 smoke code snapshot`, Genome `p4_guidance`) and the s2 best genomes.
Scratch scripts: `_scratch/p4/jets/` (exp.py, e*.out, F1*.out). Medium courses, Sim Bridge `make_course`, hold-out seeds.

## 1. Verdict in one table

Same genome (the c172x best, which passes `hard` 100 %), medium rings, 6 hold-out seeds, current plant + current Genome scaling vs the
r1 package (FD authority scaling + V^0.5 outer-gain scaling; f16 keeps Genome's FBW rate-demand mapping):

| aircraft | current pass | r1 pass | note |
|---|---|---|---|
| c172x | 1.00 | 1.00 | unchanged (scale = 1) |
| 737   | 0.56 | 1.00 | |
| T38   | 0.31 | 0.84 | 6/6 courses finish |
| f16   | 0.24 | 0.88 | 6/6 courses finish, J_chatter 0.11 |

So the jets are not plant-limited at medium: the plant flies them; the guidance is mis-scaled for them and the GA found a limit-cycle
corner. No genome was re-evolved for this (smoke genomes stay as they were); the table is a transplant demonstration.

## 2. Root causes (numbers)

1. **Outer-loop gains ignore speed (hyp. d / a).** `k_lat` is deg bank per deg bearing; the bearing closes at g*k/V, so the lateral
   time constant is V/(g k). c172x best (V 55 m/s, k 2.6): 2.1 s. T38 (178 m/s, k 4.15): 4.4 s. 737 (149, 4.15): 3.6 s. **f16 best
   (207 m/s, k 0.53): 40 s** for 14 s ring spacing. It flew straight past ring 0 (266 m lateral offset at 2.9 km), bank pinned at 20 deg,
   phugoid +-25 deg. f16 transplant check: c172x genome with k_lat/k_vert x (V/V_c172x)^a: pass 0.24 (a=0) -> 0.90 (0.5) -> 0.97 (1.0).
   a = 1 overshoots on T38 (structural failures, bank +-60 deg), a = 0.5 is the robust choice (T38 0.75-0.84, f16 0.89-0.90, 737 1.0).
2. **Inner-loop scaling uses a kinematic, not authority.** Genome scales pitch by `pitch_rate_g_limited` and roll by `roll_rate_max`.
   Measured control power at the course trim (deg/s^2 per command norm, `p4r1_gain_scaling.json`):

   | | pitch | roll | yaw | Genome scale p / r | effective authority vs c172x (p / r) |
   |---|---|---|---|---|---|
   | c172x | 42.2 | 163 | 22.8 | 1 / 1 | 1 / 1 |
   | 737 | 9.9 | 22.9 | 18.4 | 4.64 / 2.30 | 1.09 / **0.32** (roll 3x under-powered) |
   | T38 | 56.1 | 167 | 3.3 | 2.61 / 0.55 | **3.47** / 0.56 |
   | f16 (FBW closed loop) | 184 | 791 | 15.3 | 2.85 / 0.45 | **12.4** / 2.2 |

   737 with FD scaling: 0.56 -> 1.00. T38 / f16: see 3.
3. **The GA parks the jets at a limit cycle (kd genes at the bound).** s2 best genomes: T38 `kd_pitch` 0.499 (= gene hi), `kd_roll`
   0.122; f16 `kd_pitch` 0.132. c172x / 737 best have kd ~ 0.001-0.005. Rate-damper loop gain K = kd x scale x authority:
   **T38 73 1/s pitch, 11 roll; f16 69 1/s** (c172x 0.0, 737 0.0), against actuator poles at 12-20 rad/s: unstable, saturating the
   actuator. Signature: T38 elevator rate-saturated 93 % at 0.92 Hz; f16 elevator 91 % at 2.4 Hz (seeds 2, 3; seeds 0, 1 do not limit-cycle
   but never track, cause 1). Not the actuator: with rate 200 deg/s, tau 0.02 the T38 still oscillates at 2.5 Hz (chatter 0.34).
   With kd capped at 0.05 (authority-scaled) the same T38 genome: J_chatter 0.34 -> 0.05, pass 0.03 -> 0.15-0.27 (the other genes were
   co-adapted to the limit cycle, hence a re-evolution is needed). The chatter weight (0.03) is too small to stop the GA using this corner.
4. **f16 FD actuator stacked in front of the FBW loop (hyp. b): confirmed, but secondary.** `fcs/aileron|elevator|rudder-cmd-norm` on the
   f16 are rate / g demands closed by JSBSim's own FBW PIDs; FD's lag (0.0495 s) + rate limit (60/80/120 dps scaled by deg-per-norm
   25/21.5/30) acted on the demand, not on a surface. Same genome, v = 1: FD lag in front J_chatter **0.37**, no lag **0.04-0.16**;
   with tau >= 0.05 the oscillation is deterministic (J_chatter 0.35-0.38 on all four seeds, rate 60-250 dps alike); tau 0.02 gave 0.27 on the one seed that survived (3 of 4 hit the tail limit), so the lag is the driver but not cleanly isolated. Pass rate barely changes (0.97 vs 0.93).
   A post-FCS actuator (patched `f16.xml` copy: kinematic 60/80/120 dps + 20.2 rad/s lag) was built and tested
   (`flexeval_p4r1.F16_POST_FCS`, **OFF**): the shipped JSBSim FBW gains (g PID kp 0.3 / ki 0.025, q gain 6.2, roll PID kp 3) are tuned
   for no actuator lag; with it the elevator limit-cycles at every rate / lag tried (rate-sat 50-85 %), so it needs an FBW re-tune
   (out of scope). Decision: r1 passes the f16 demands through unchanged, reports JSBSim's measured surface rate and the S&L 60/80/120
   limits as reporting limits. Side effect: without FD's lag in front, aggressive guidance (V^1.0) overloads the f16 tail
   (`J_tail_bm_peak` 1.8-2.1 -> structural_failure on 3 of 4 seeds); V^0.5 has none.
5. **T38 deg-per-norm and rates (hyp. c): not binding.** With sane gains (c172x genome) T38 elevator / aileron rate-sat = 0 %, rudder 15 %
   (rudder power is 7x weaker than c172x: 3.3 vs 22.8 deg/s^2 per norm, so yaw gain is clamped to scale 1.0; a rate-sat rudder is the T38's
   remaining chatter, J_chatter 0.31). The 93 % elevator saturation of the smoke genome is cause 3.
6. **Geometry (hyp. d): flyable.** Medium ring spacing / turn radius at `bank_course`: c172x 4.4, 737 1.6, T38 **2.9**, f16 **3.8**;
   hard: 3.1 / 1.1 / 2.1 / 2.7 (sustained-turn radius R_sus: T38 1388 m, f16 953 m vs spacing 2487 / 2892 m). The jets have more room
   per ring than the 737, which passes. Course speed is not the problem.
7. **Trim not exposed (hyp. e):** the elevator-cmd trim is 0.0 on all four; the pitch trim sits in `fcs/pitch-trim-cmd-norm` (c172x +0.219,
   T38 -0.115, 737 -0.233, f16 -0.060, summed inside the FCS), so a zero-start pitch integrator is NOT a jet problem. Only c172x has a
   non-zero aileron trim command (-0.075) and rudder (-0.004). Now exposed (section 4).

## 3. What changed (new files only; r0 frozen and bit for bit)

| file | role |
|---|---|
| `ctrlsurf_p4r1.py` | surface model r1 (tag `p4cs1`): f16 elev/ail/rud = FBW demand channels (no FD lag/rate/hinge in front; measured sat flags), flap/sbrk rates published, `scored` flag |
| `flexeval_p4r1.py` | fidelity **`full_a1_b2a_cs1`** (alias `full_a1_b2a_p4r1`), same `evaluate` / `fly_course` API as r0; guidance state gains `trim`, `qbar_psf`, `mach`; output gains `trim_cmd`, `surface_limits_all`, `surface_scored`, `surface_fbw`; optional f16 post-FCS root (OFF) |
| `p4r1_scaling.py` -> `v2_results/p4r1_gain_scaling.json` | measured control power, per-axis `gain_scale`, `outer_gain_scale`, `gene_caps`, FBW mapping |
| `v2_results/model_versions_post_p4r1.json` | pins, e.g. T38 `full_a1_b2a_cs1:p4cs1:active:c6c58185`, f16 `...:33efc4e9`, 737 `...:07531f5f`, c172x `...:a1229008` (+ passthrough, + b2a) |
| `test_ctrlsurf_p4r1.py` | 23 tests; `v2_results/FROZEN_P4r1.md5` |

r1 vs r0 plant: c172x / T38 / 737 flights are bit-identical (tested; only the pin string differs); f16 differs (FBW passthrough).
`surface_limits` now lists only the scored surfaces (elev, ail, rud), so ER's surface scoring neither crashes on a None rate nor dilutes
its mean with never-moving flap / speedbrake. `p4_aircraft_limits.json` is unchanged.

## 4. Genome's four asks

1. **Flap rate:** published (`surface_limits_all`, `surface_table`): c172x flap 10 deg/s, T38 5.4, 737 0.045 norm/s, speedbrakes T38 1.0,
   737 1.67 norm/s, f16 20 deg/s (N). They are marked `scored: False` and are not in `surface_limits`, which holds elev/ail/rud only.
2. **Trim commands:** `state["trim"] = {aileron, elevator, rudder, throttle, flap, speedbrake, pitch_trim, roll_trim, yaw_trim}` (also
   `result["trim_cmd"]`); plus `state["qbar_psf"]`, `state["mach"]`. Elevator-cmd trim is 0.0 on all four (pitch trim in `pitch_trim`: c172x +0.219, T38 -0.115, 737 -0.233, f16 -0.060); c172x aileron -0.075. Recommended use: initialise the pitch integrator / feed-forward from trim, and
   schedule gains with qbar_trim / qbar.
3. **Rudder sign (tested, all four, +-0.1 norm pulses at the course trim):** +rudder -> negative body yaw rate r = **nose LEFT** on c172x,
   T38, 737 and f16. Also +aileron -> +p, +elevator -> nose down (q < 0), all four (`test_rudder_sign_plus_is_nose_left`, `test_ail_elev_signs`).
4. **f16 FBW placement:** confirmed (cause 4). r1 applies no FD lag/rate in front of the f16 demands; rate-sat flags are measured from
   the JSBSim surface positions; the FD 60/80/120 dps remain the reporting limits. Post-FCS limiting needs an FBW re-tune (flag OFF).

`tau_cmd_s`: not redundant with the FD actuator (tau 0.05 s c172x / T38, 0.08 s 737; f16 none now). It is a command prefilter inside the whole
inner loop (it also filters the damper term). Keep 0.05-0.5; no evidence that 0.2 is better or worse (T38 own genome with 0.05: no gain).

## 5. Genome-side change needed (Genome owns the law; FD cannot edit it)

1. `gain_scales`: take pitch / roll / yaw from `p4r1_gain_scaling.json` `aircraft[m].gain_scale` (c172x = 1). **f16: keep the FBW
   rate-demand mapping** (the closed-loop authority number is informational; auth-scaling the f16 failed in tests).
2. Multiply `k_lat, k_lat_rate, k_vert, k_vert_rate` by `outer_gain_scale` = (V_tas / V_tas_c172x)^0.5: T38 1.80, 737 1.65, f16 1.95.
3. Cap the damper genes: `kd_pitch`, `kd_roll` hi 0.5 -> **0.05** (f16 `kd_pitch` 0.02), per `gene_caps` in the json.
4. Optionally qbar-schedule the inner loops (x qbar_trim/qbar; v_cmd_scale up to 1.25 means q up to 1.56x).
5. Optionally seed gen 0 with the c172x best u (transplant demo: T38 0.84, f16 0.88, 737 1.0 on medium without any re-evolution).
No `tau_cmd_s` range change needed.

## 6. Re-run plan for ER (pin `full_a1_b2a_cs1`, jets only)

* c172x and 737 plants are bit-identical to r0, so their s2 results stand (737 gains from the scaling, so re-run it only if the Genome
  patch is adopted for all aircraft).
* T38 and f16 only, new run id, e.g. `phase4-jets-r1`: Genome patch in (section 5), FD pin copy = r1 files + `p4r1_gain_scaling.json`
  (`ctrlsurf_p4r1.py`, `flexeval_p4r1.py`, `v2_results/p4r1_gain_scaling.json`, `model_versions_post_p4r1.json`; no new JSBSim root while
  `F16_POST_FCS` is off), same scoring, `pin_model_version` = the r1 pins.
* Stages: start at `medium` (easy proven redundant; the gate still applies 0.8 x 2 gens, promote to `hard`), seed gen 0 with the c172x
  best `u` (+ mutated copies), pop 16, 8 generations (s2 had 6), K = 4 courses per genome, 8 hold-out courses.
* Budget: s2 = 20 128 s wall on 5 workers for four aircraft, about 25-35 s per 320 s course. Jets only with 8 generations ~ 2-2.5 h on 5 workers.
* Acceptance: hold-out pass >= 0.8 on medium for both, J_chatter < 0.15, no kd gene at its (new) bound, then `hard`.
* First check (5 min, no GA): fly the c172x best `u` on T38 / f16 / 737 / c172x through the patched guidance (this document, section 1).

## 7. Honest limits

* Transplant numbers use the c172x genome, monkeypatched scaling in the s2 snapshot (no Genome file edited); 6 hold-out seeds, medium only.
* T38 own-genome with caps only reaches 0.15-0.27: those genes co-adapted to the limit cycle; needs the re-evolution above.
* f16 `kd_pitch` 0.02 cap and f16 hard stage are untested; f16 V^1.0 overloads the tail without FD lag in front (use 0.5).
* Surface rates / lags / deflection scales for T38, 737 (and f16 rudder) remain notional (N), as in the r0 spec; not binding for the result above.
* The ER pin copy must be refreshed by ER (FD did not touch `evolution/`).
