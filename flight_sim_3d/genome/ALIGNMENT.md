# Phase-1 profile alignment: genome/ vs evolution/ (2026-10-06)

**Shared set:** `genome/aircraft_profiles/phase1_shared.json`. It is the single source of truth for c172x, T38 and 737:
trim point, envelope, controller limits, task definition, fitness weights and pitch/altitude gain bounds.
- genome/ reads it through `profiles.load_profile`. Trim point, envelope, clamp, throttle cap and gain bounds are
  overlaid, and a profile JSON that disagrees raises an error. Presets opt in with `"shared": "phase1"`; that covers
  every non-legacy preset. `altitude_hold_legacy` doesn't use it and still reproduces results/example bit for bit.
- `export_shared_profiles.py` writes the same set as an Evolution Runner profile block, using only keys that
  `evolution/sim.Profile` accepts today: `exports/evolution_phase1_profiles.json`. I validated it read-only with their
  `Profile.from_dict` and `make_schema`. I wrote nothing in evolution/ or flight-dynamics/.

What I read in evolution/: `configs/bench_jets.json`, `configs/bench_jets_step200.json`, `runs/bench_jets-j1` and
`runs/bench_jets_step200-j1` (config.json, summary.json), `sim.py`, `genome.py` and README.md.

## Simulator agreement (cross-check)

`crosscheck_evolution.py` re-flies their best T38/737 genomes in our `sim_ext` under **their** task and recomputes
their cost from our telemetry. Their task here means an instant speed-scaled step, a [-8, 12]° clamp, unscaled helper
loops and their envelope.

| run / aircraft | their best cost | same task in our sim | rel. diff |
|---|---|---|---|
| bench_jets-j1 T38 | 0.13147 | 0.13147 | 2e-16 |
| bench_jets-j1 737 | 0.19201 | 0.19201 | 2e-15 |
| bench_jets_step200-j1 T38 | 0.08328 | 0.08328 | 3e-15 |
| bench_jets_step200-j1 737 | 0.12880 | 0.12880 | 9e-16 |

The two simulators agree to round-off. That covers trim, gear, throttle and the controller, and it also shows that
FD's patched aircraft copies fly identically to their sanitized stock copies (FD now strips network I/O at the source;
genome/ loads the copies directly and refuses any aircraft file that declares network I/O). The test is
`tests/test_shared.py::test_our_sim_reproduces_evolution_runner_t38_cost`.

**Their costs are not comparable to ours as numbers.** Their task uses a 600/500 ft step scored with
alt_err_scale = step/2, an instant step, no comfort term and legacy ranges. Ours uses a 200 ft step at 600 fpm with
0.1 g corners, scale 100 and comfort 0.05. What *is* comparable is their genomes flown under our Phase-1 fitness;
see the table at the end.

## Differences and decisions

| # | item | evolution/ (bench_jets) | genome/ (before) | **shared pick** | why |
|---|---|---|---|---|---|
| 1 | trim points | T38 300 KCAS/10 kft, 737 250/10 kft, c172x 100/4 kft | same | **same** | FD-verified trim points; both teams already agreed |
| 2 | altitude step | 200 ft × KCAS/100 (T38 600, 737 500), instant; error scale = step/2; max err = 5 × step | 200 ft for all, ramped at 600 fpm | **200 ft for all, ramped at 600 fpm with 0.1 g corners**; scale 100 ft; max err 1000 ft | The ramp is a team decision for every non-legacy task, and 600 fpm is a passenger-comfort rate that doesn't scale with speed. A 600 ft step at 600 fpm needs 60 s, longer than the 45 s segment (step at 5 s, return at 50 s). Their own step200 variant gave cleaner holds (per-scenario hold RMS 0.3–2.9 ft vs 1.2–5.1 ft with the scaled step). |
| 3 | 737 g limit | [-1, 2.5] | [-1, 2.5] | **[-1, 2.5]** | FAR-25 transport limit; no difference |
| 4 | T38 g limit | [-3, 7.33] | same | **same** | — |
| 5 | T38 min speed | 180 KCAS | 165 KCAS | **180 KCAS** | Theirs is documented as ~1.2 × Vs1g (146 KEAS) at the benchmark; ours was a generic 1.15 × Vstall. Only the envelope-failure threshold moves; no T38 run came near it |
| 6 | 737 min speed | 195 | 195 | **195** | JSBSim 737 trim fails below ~190 KCAS (FD) |
| 7 | T38 throttle | cmd ≤ 0.5 (MIL); trim above that = `trim_failed` | cmd ≤ 0.5 | **cmd ≤ 0.5 plus their trim check** | Adopted their trim check in `sim_ext` (non-legacy only) |
| 8 | engines / gear / trim mode | all engines' throttle and mixture, gear up before run_ic, `do_simple_trim=1` | same, except mixture on engine 0 only | **all engines, gear up, full trim** | Adopted mixture on all engines (no effect on turbines or the c172x) |
| 9 | aircraft files | sanitized stock copies (socket I/O stripped) | FD `jsbsim_root` copies as-is | **FD `jsbsim_root`, loaded directly** (update 2026-10-06 ~05:00 PT: FD strips network I/O at source; genome/ refuses any file declaring it, `fd_bridge.check_no_network_io`) | FD's copies carry the flex point masses (zero weight, bit-identical flight). Our earlier /tmp stripping copy is removed; c172x/T38/737 costs are bit-identical without it. |
| 10 | pitch-command clamp | [-8, 12]° for every aircraft | per aircraft: c172x [-8, 12], T38 [-10, 15], 737 [-5, 10] | **per aircraft** | Team decision (per-aircraft clamp); a 12° nose-up command is not a passenger-jet autopilot value |
| 11 | gain bounds | prototype ranges for every aircraft (log) | v2 ranges × control-power scaling, log0 on ki | **c172x: v2 ranges. Jets: union of the derived and the prototype ranges, written out explicitly per aircraft** (log, log0 on ki) | Evidence: their 737 optimum kp_pitch 0.021 (step200) lies **below** our derived 737 minimum (0.030), so the control-power scaling overshoots on the 737 inner loop. Their T38 ki_alt pinned at the prototype minimum 1e-5. The union keeps every known optimum inside. Explicit numbers mean both teams run identical ranges |
| 12 | helper loops (speed PI, wing leveler) | C172 constants (0.05/0.01, −0.05/−0.02) | profile-scaled (T38 0.0367/0.0073, 737 0.0351/0.0070; roll ×2.4 / ×12) | **profile-scaled** | C172 gains on jets are arbitrary. Evolution can take the throttle gains via `thr_kp`/`thr_ki` today; the roll gains need a code change (#C below). Effect on this longitudinal task is small: phi stays ~0 |
| 13 | climb-rate feed-forward | none | D term on (ḣ − ḣ_ref) | **on** | A/B on c172x with the smoothed ramp: FF off costs 0.2704 vs 0.2063 with FF, and track_alt goes 0.098 vs 0.061 (README) |
| 14 | fitness | track + 2·effort | track + 2·effort + 0.05·comfort | **track + 2·effort + 0.05·comfort** | Team default; comfort needs nz/θ/q each step |
| 15 | scenarios | `make_scenarios` draws, n 3, seed 1, calm scenario 0 | same | **same** | Already identical, including the RNG |
| 16 | ITAE (t0 10 s, cap 30 s), integrator limits (5°, 0.4), duration 90 s, min AGL 500 ft, θ 30°, φ 45° | same | same | **same** | — |
| 17 | GA budget | pop 32 × 20 gens (jets) | pop 48 × 40 (c172x), 24 × 10 (jets) | not part of the profile; **suggest 32 × 20 for jets** | Used for our jet runs below so the GA budget matches theirs |
| 18 | heading hold (added 2026-10-06) | none: wing leveller only, c172x drifts ~25° | none | **outer heading loop on for c172x/T38/737**: kp_hdg/ki_hdg genes, bank clamp 16/25/25°, cost + 0.01·RMS(e_ψ)/5° | Stops the c172x prop-torque drift; jets unchanged (HANDOFF_heading_hold.md) |
| 19 | v5 candidate (added 2026-10-06, opt-in) | none | none | **`phase1_v5` only**: cost + 0.1·hold_osc (RMS of the hold-window error about its window mean / 5 ft) and a 4th scenario with a sustained downdraft V_TAS·tan 1.5° | Hold oscillation and an altitude integrator were unrewarded (Sim Bridge replay); v4 stays the default (HANDOFF_phase1_v5.md) |

## What Evolution Runner would change to point at the shared set

A. **Profiles (config only, no code):** paste `profiles` and `aircraft` from
   `genome/exports/evolution_phase1_profiles.json` into a batch config, regenerated by
   `$PY genome/export_shared_profiles.py`. This sets the trim points, 200 ft steps, scale 100 / max error 1000,
   T38 min 180 KCAS, per-aircraft clamps, `thr_kp`/`thr_ki`, `throttle_max`, gear up, all engines and the explicit
   `gain_bounds`.

B. **Altitude ramp, 600 fpm with 0.1 g corners (code, `evolution/sim.py`):**
   - `Scenario.target(t)` returns the accel-limited reference instead of the instant step. The reference algorithm is
     the closed-form constant-acceleration phases in `genome/sim_ext.py::ExtScenario._smooth_plan`.
   - Score `e_h` against that reference. The ITAE clock still starts at the command change.

C. **Climb-rate feed-forward:** in the outer loop use `- kd_a * (h_dot - h_ref_dot)`, where `h_ref_dot` is the
   reference rate.

D. **Comfort term (weight 0.05):** compute nz, θ and q each step and add `comfort_terms` (copy or import
   `genome/fitness.py::comfort_terms`), or call genome's `fitness.evaluate` on recorded telemetry.

E. **Gene kinds:** add a `log0` kind for ki_alt/ki_pitch (v ≤ 0.05 → 0, log over the rest), or accept plain log with
   the shared minimum as the floor (small mismatch).

F. **Wing-leveler gains:** make `-0.05*phi - 0.02*p` profile fields (`roll_kp`, `roll_kd`). The values are in the
   genome adapter's `sim_fixed` for each aircraft.

G. **Aircraft path:** load from `flight-dynamics/jsbsim_root` (FD now strips network I/O at the source; done in their
   Phase-1 rerun). Physics is identical, as the cross-check above shows; this matters only for flex runs and the payload
   placeholder index.

With A alone their runs and ours share trim, envelope, clamp, step size, error scale and gain ranges. With A–D the
costs become directly comparable; A–G gives bit-level agreement, which our cross-check suggests is reachable.

**Status 2026-10-06 ~05:00 PT:** Evolution Runner implemented A–G; their 3-seed Phase-1 rerun matches our v4 runs
bit for bit (evolution/runs/phase1_report.md). Next change: the heading-hold outer loop and its cost term,
HANDOFF_heading_hold.md (profile block regenerated in exports/evolution_phase1_profiles.json).

**Status 2026-10-06 ~06:00 PT:** Evolution Runner implemented heading hold (configs/phase1_hdg.json, equal to
exports/evolution_phase1_profiles.json); the handoff's three genomes re-fly bit for bit there. Candidate next change
(not adopted): v5, HANDOFF_phase1_v5.md, profile block in exports/evolution_phase1_v5_profiles.json.

## Their best genomes under our Phase-1 fitness (`runs/crosscheck_evolution.json`)

Our numbers come from GA runs on the shared set: pop 32 × 20 gens, seed 1, 3 scenarios (`runs/phase1_{t38,b737}_v4`).

| genome | T38: our Phase-1 cost | T38 track_alt | 737: our Phase-1 cost | 737 track_alt |
|---|---|---|---|---|
| Evolution bench_jets-j1 best (evolved for the 600/500 ft instant step) | 0.1636 | 0.117 | 0.1237 | 0.061 |
| Evolution bench_jets_step200-j1 best (evolved for the 200 ft instant step) | 0.1205 | 0.067 | 0.1231 | 0.050 |
| genome/ GA on the shared set (pop 32 × 20, seed 1) | **0.0918** | 0.028 | **0.1102** | 0.039 |

All of their genomes fly every scenario under our task and lie inside the shared ranges. Before the union, their
step200 737 kp_pitch 0.021 was outside our old derived range. Our GA does better on our task, which is expected:
their genomes were tuned for a different reference and a different score. This shows consistency, not that one GA is
better.

Our jet runs:
- **T38:** calm max pitch 6.1° (+1.5° vs trim), climb ≤ 770 fpm, nz 0.68–1.22, overshoot ≤ 4.7 ft.
- **737:** pitch 5.2° (+1.9°), climb ≤ 810 fpm, nz 0.72–1.20, overshoot ≤ 4.8 ft.
- On both, ki_alt is "zeroed": the GA switched the outer integrator off via log0, which matches their T38 ki_alt
  pinned at the 1e-5 floor.
