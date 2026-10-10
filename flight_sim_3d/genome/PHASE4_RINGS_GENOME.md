# Phase 4 ring course: Genome section (Genome Architect, 2026-10-07 ~02:40 PT)

Meant for the Genome section of Sim Bridge's `flight_sim_3d/PHASE4_RINGS_SPEC.md`. That file and `flight_sim_3d/`
did not exist anywhere under /workspace/flight-sim-team at 02:20 PT, so this section lives here and gets pasted in once SB publishes the skeleton.

## 1. Chromosome: 29 genes = guidance 12 | inner_loop 11 | mixing 6
All genes are stored as u in [0,1] and decoded by `genome_schema.GeneSpec` (log: min·(max/min)^u, linear, log0 = exact 0 for u ≤ 0.05).
Identity = encode(default). Ranges are c172x reference values. Surfaces are JSBSim `fcs/{aileron,elevator,rudder,throttle}-cmd-norm`.

| block | gene | range | default | scale | units |
|---|---|---|---|---|---|
| guidance | t_preview_s | 1–8 | 3 | log | s (aim point along the line of sight (LOS) to the next ring) |
| guidance | w_next_ring | 0–0.6 | 0.25 | linear | blend toward ring n+1 |
| guidance | k_lat | 0.1–5 | 1.0 | log | deg bank / deg LOS bearing error |
| guidance | k_lat_rate | 1e-3–2 | 0 | log0 | deg bank/(deg/s) |
| guidance | k_vert | 0.05–3 | 0.5 | log | deg γ / deg vertical LOS error |
| guidance | k_vert_rate | 1e-3–1 | 0 | log0 | deg γ/(deg/s) |
| guidance | bank_max_frac | 0.25–1 | 0.75 | linear | × FD bank_course_deg (c172x 60, T38 75, 737 60, f16 80) |
| guidance | nz_max_frac | 0.4–1 | 0.7 | linear | × FD n_inst (3.56 / 4.20 / 2.5 / 8.79) |
| guidance | nz_min_frac | 0–0.75 | 0.5 | linear | nz_min = 1 − frac·(1 − n_profile[0]) (−1/−3/−1/−3) |
| guidance | gamma_max_deg | 5–25 | 15 | linear | deg |
| guidance | v_cmd_scale | 0.85–1.25 | 1.0 | linear | × course V_ref |
| guidance | v_turn_comp | 0–0.15 | 0 | linear | × V_cmd per (nz−1) g |
| inner_loop | kp/ki/kd_roll, kp/ki/kd_pitch, kr_yaw, kbeta_yaw, tau_washout, kp_spd, ki_spd | identical to genome_schema.ALL_GENES (test-enforced) | | | |
| mixing | k_ari | 0–0.6 | 0 | linear | rudder per aileron |
| mixing | k_turn_coord | 0–1.5 | 0 | linear | rudder per rad/s r_cmd = g·tanφ/V |
| mixing | k_elev_bank | 0–0.5 | 0 | linear | elevator × (1/cosφ − 1) |
| mixing | ail_auth | 0.4–1 | 1.0 | linear | × travel |
| mixing | rud_auth | 0.2–1 | 1.0 | linear | × travel |
| mixing | tau_cmd_s | 0.05–0.5 | 0.1 | log | s, guidance command prefilter (FD actuator lag is separate) |

With every mixing gene at identity, the airframe sees the plain inner-loop commands: no interconnect, no feed-forward, full authority.

## 2. Carry-over (recommendation: A)
- **A (recommended, wired):** `carry_over: "none"`. Structure is held at FD baseline and the planform at B1 r1 identity. Neither is in the vector.
  Reasons: there's no ring-course structural or energy cost yet (ER owns scoring), the moving-surface fidelity has no pins,
  and B1/B2a gave small shape gains (≤ 0.03 cost) that would compete with 29 new controller genes for search budget.
- **B (wired, opt-in):** `carry_over: "structure"` adds phase3_b1's 12 structure_v2 genes as a 4th whole block (41 genes).
  Needs FD's flexeval as the structural cost (struct_v2_source fd) on the ring envelope.
- **C (not wired):** structure plus B2a shape (9 genes, tc locked) means 50 genes. It needs block_ops shape ops plus FD's ring fidelity and pins, and the geometry gate stays a hard reject.

## 3. Operators (baseline per ER's B1 A/B)
Elite 2, flat rank with p 0.2, whole-block crossover (`d = rng.random(3)`, guidance|inner_loop|mixing, d < 0.5 → parent A), then
gauss mutation (`rng.random(29) < 0.15`, N(0, 0.08) on hits, clip). Gen-0 is `rng.random((pop, 29))`, the same as phase3_b1's controller block.
Evidence (evolution/analysis/STATUS_P3B1_ab.md): tweaked (elite 4 + uniform) beat baseline in only 1 of 6 (seed, aircraft) pairs.
The mean Δ was c172x −0.85 %, T38 +0.22 %, 737 +2.46 %, and there was no consistent shape-contribution gain. ER recommends baseline. Caveat: 2 seeds, and the effects are inside 1 seed sd.

## 4. Files
`phase4_rings.py` (standalone loader / encode / decode / operators / evaluator stub), `presets/phase4_rings.json`,
`p4_operator_trace.py` → `runs/p4_operator_trace.json`, `tests/test_phase4_rings.py`, and `tests/test_adapter.py` (skips the
phase4 preset in test_all_presets_load). adapter.py, block_ops.py and every existing preset are unchanged.

## 5. Open questions
- SB: the course spec (ring frames, radius, spacing, how many rings ahead are visible, V_ref per aircraft, pass/miss geometry, what counts as "rolling sets"); the guidance-state API (LOS to rings n and n+1).
- FD: are rudder and differential surfaces live on every aircraft in the moving-surface model? Per-aircraft nz/bank limits for clamping; ring fidelity name and pins; surface rate limits (tau_cmd_s may be redundant); per-aircraft gain rescale (profiles.py tags).
- ER: ring scoring terms (miss distance, time, effort, structural via FD); the A/B rerun on rings; mirror the phase4 operators and compare against runs/p4_operator_trace.json hashes.

## 6. Update 2026-10-07 ~02:50 PT: per-aircraft limits, guidance law, evaluator (FD P4.8–P4.12, SB spec v0.3, ER scoring)
- **Aircraft:** c172x, T38, 737, f16 (amendment C1–C6). Limits are read from FD `v2_results/p4_aircraft_limits.json`.
  The decoded bank, nz_max and nz_min never exceed the file values. Identity values: c172x 45° / 2.49 g / 0 g,
  T38 56° / 2.94 / −1, 737 45° / 1.75 / 0, f16 60° / 6.16 / −1. The f16 nz ceiling uses FD's n_inst (8.79), not n_struct (9).
  The f16 TEF/LEF are flight-computer scheduled and never commanded. Speedbrake (T38, 737, f16) is an FD pass-through channel that is not evolved.
  Adding it would be a 30th gene and needs ER to mirror the change.
- **Gain scaling:** fixed per-aircraft constants with reference c172x = 1.
  Roll and yaw gains × p_max,c172x / p_max, pitch × q_ref / pitch_rate_g_limited_dps, speed × vt_trim,ref / vt_trim. nz protection is 10°/(n_inst − 1) per g of overshoot.
  The flex η is never used.
- **Guidance inputs (Q-G4):** canonical NED. SB converts at the bridge, and the line of sight stays in body FRD. Each step the law
  calls `sim_bridge.ring_course.guidance_inputs(fd_state, course, n, n_ahead=2)` read-only and uses
  `pos_ned`, `gamma_deg`, `q_body_to_ned`, `n` and `rings[0..1].{range_m, bearing_deg (+ right), elevation_deg (+ above), centre_ned}`.
  It also passes through the FD state fields `phi, p, q, r, beta, nz, vc_kts, vt_fps`.
- **Law** (`p4_guidance.make_guidance`, a stateful closure per flight, called at 120 Hz by FD `fly_course`):
  1. Ring index: SB `gate_check` on ring n with the capture rule, an out-of-order pass inside the window, or a timeout of 3 × leg time. The window slides by one.
  2. Aim point = c_n + w·(c_{n+1} − c_n), with w = w_next_ring·clamp(1 − t_go/t_preview_s, 0, 1). Its body-FRD bearing and elevation are computed. When w = 0, SB's ring-n values are used directly.
  3. φ_c = clamp(k_lat·brg + k_lat_rate·d(brg)/dt, ±bank_max).
  4. Pitch error = k_vert·elev + k_vert_rate·d(elev)/dt, limited by gamma_max_deg and nz protection.
  5. PID roll and pitch loops. Washed-out yaw damper, β feedback, turn-coordination feed-forward and the aileron-rudder interconnect.
  6. Authority clamps, then the τ prefilter.
  7. Throttle = FD trim throttle + the PI speed loop on v_cmd_scale·V_ref·(1 + v_turn_comp(1/cosφ − 1)).
  - Signs: +ail → +p, +elev → nose down, +rud → nose left (checked on c172x).
- **Evaluator** (`p4_guidance.evaluate`): for each of the K=4 shared seeds:
  1. SB `make_course`.
  2. FD `fly_course` (full_a1_b2a_cs, active), run from a read-only copy of ER's pin `evolution/_fd_pin_p4cs` at `genome/_p4team/flight-dynamics`. The copy has the same md5; its `_p4team/evolution` is a symlink so FD's paths resolve.
  3. ER `score_course` per course, then ER `aggregate_courses` (blend 0.3, α 0.25).
  Genome adds no cost terms. The cache key covers the preset, fidelity, cs_mode, model, model_version, stage, course seeds, K, aggregation and u.
- **Smoke** (identity, c172x, easy, seed 3089335368, M=15): status ok, 15/15 passes, ER single-course cost 0.02840 (`runs/p4_smoke_identity_c172x.json`).
  The K=4 aggregate is 0.03004 (mean 0.02926, CVaR 0.03186), with 1.0 pass rate on all 4 courses (`runs/p4_eval_identity_c172x_K4.json`). Each course takes about 15 s CPU.
- **Gaps:**
  - SB `ring_course.AIRCRAFT` and ER `rings.AIRCRAFT` have no f16, so f16 courses and scoring can't run yet.
  - FD reports the flap with rate_max None, and ER `surface_terms` divides by it. The genome passes only rate-limited surfaces to ER for now.
  - ER `carried` (FD structural, energy and speed-guard terms) is not mapped; it is passed empty until ER names the keys.
  - The trim elevator is not in the FD state, so the pitch integrator starts at 0.
  - Course K: the spec defaults to 3 and ER uses 4. The genome uses 4 (ER).

## f16 FBW mapping + SB v0.5 TAS geometry (03:10 PT Oct 7)
* f16 `fcs/aileron-cmd-norm` / `fcs/elevator-cmd-norm` are FCS *demands* (f16.xml): 1 norm = 1/0.31821 rad/s = 180.06 deg/s
  roll rate; 1 norm ~ 1/6.2 rad/s = 9.24 deg/s pitch rate (+ 0.02 x corrected nz; + = nose down, FCS clips at +0.44).
  `p4_guidance.FBW["f16"]`: inner-loop output u (c172x-equivalent units) is sent as the demand it makes on the reference
  (roll scale = 81.73/180.06, pitch scale = 26.36/9.24); rudder keeps the surface scale. Signs unchanged. 29 genes unchanged;
  c172x/T38/737 bit-identical (T38 K4 re-run byte-identical). Identity effect on the f16: cost 0.4306 -> 0.4315 (no change in passes).
* SB v0.5: geometry = TAS at h0. `leg_t` (timeout) uses `params.v_tas_ms` (same value as SB's `v_ref_mps` alias). The speed
  command stays `v_cmd_scale x V_ref(KCAS)` vs `vc_kts`. Preview uses range / actual TAS (vt_fps). ER traj `v` is still KCAS.
