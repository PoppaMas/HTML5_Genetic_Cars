# Genome & fitness design: flight-controller GA

Status: v0.2 (Genome Architect), 2026-10-06. v0.2 adds the team decisions: altitude ramp, per-aircraft
pitch clamp, default fitness with a modest comfort weight, and T-38 / 737 Phase-1 profiles (§3a). Plugs into `flight_sim/` on branch
`flight-sim-prototype` without changing `ga.py`, `evolve.py`, `sim.py` or `genome.py`.

## 1. Constraints from the existing code

- `ga.py` only works with a float array in [0,1]^n. It uses rank-based selection
  (p = 0.2), elites, uniform/BLX crossover and Gaussian mutation, and it never
  sees units. **Keep the normalized encoding** and let only the decoder know physics.
- `evolve.py` needs from `genome`: `N_GENES`, `GENE_NAMES`, `SCHEMA[i].name/.units`, `decode(g) -> dict`.
  From `sim`: `make_scenarios(n, seed)`, `evaluate(gains, scenarios) -> {"cost", "per_scenario":[{status,...}]}`, `AIRCRAFT`.
  `plot_results.py` also needs `sim.simulate(gains, sc, record=True)["trace"]`.
- Selection uses ranks only, so the absolute scale of the cost doesn't matter, but the weights *between* terms do.

## 2. Genome schema

A gene is `GeneSpec(name, block, min, max, scale, default, units, doc, scaling, provisional)`.

| scale | decode v ∈ [0,1] | use |
|---|---|---|
| `log` | `min·(max/min)^v` | gains spanning decades (the original behaviour) |
| `linear` | `min + v·(max−min)` | geometric deltas, signed quantities |
| `log0` | `0` if v ≤ 0.05, else log over the rest | I/D terms the GA should be able to switch **off** |

Blocks are listed in canonical order. The genome is the concatenation of the
**enabled** blocks, so its length depends on the task (6, 14, or 24 genes in the presets).
Disabled blocks are held at their `default`. Those defaults equal the fixed helper
gains in `sim.py`, so a disabled block reproduces legacy behaviour.
`GenomeSpec.transfer()` re-encodes a genome between tasks, e.g. to seed a 3-axis run from a 1-axis result.

| block | genes (reference range, C172 @ 100 KCAS) | law |
|---|---|---|
| `pitch_altitude` | kp_alt [0.002, 2.0] log · ki_alt [1e-6, 0.05] log0 · kd_alt [0.01, 3] · kp_pitch [0.002, 0.5] · ki_pitch [1e-5, 0.2] log0 · kd_pitch [1e-4, 0.5] | alt→θcmd PID (D on ḣ), θ→elevator PID (D on q). Same law as sim.py |
| `roll_heading` | kp_hdg [0.05, 5] · ki_hdg log0 · kp_roll [0.002, 0.5] · ki_roll log0 · kd_roll [1e-4, 0.5] | ψ→φcmd PI (±25°), φ→aileron PID (D on p) |
| `yaw_damper` | kr_yaw log0 · kbeta_yaw log0 · tau_washout [0.3, 10] s | rudder = −(kr·washout(r) + kβ·β) |
| `speed_throttle` | kp_spd [0.002, 0.5] · ki_spd log0 · kd_spd log0 | V→throttle PI + V̇ damper |
| `structure` | stiffness_scale [0.6, 2] log · torsion_bend_ratio [0.8, 1.15] linear · struct_damping_ratio [0.005, 0.05] log · nonstructural_mass_scale [0.8, 1.25] log | FD's final Phase-1 block (INTERFACE.md §0/§1a, = `flexwing.STRUCT_SCHEMA`). Tied stiffness: EI × s, GJ × s·r (`flexwing.tied_stiffness`). Elastic axis / section CG (c172x 0.38/0.42, T38 0.40/0.42, 737 0.36/0.38; CG aft of EA), tip mass (0), AR and sweep (0) are fixed and never sent from the genome. Wing mass follows from the stiffness genes (FD mass link). Structural data is notional |

### Pinned-gene fix (legacy: kp_alt at its 0.5 upper bound, ki_alt at its 1e-5 lower bound)
- kp_alt: upper bound ×4 (2.0). ki_alt and ki_pitch: `log0`, so "no integrator" is a reachable
  value (0) and doesn't show up as a pinned bound.
- The README says kp_alt = 0.5 already saturates the pitch command at about 25 ft of error. So the
  pin is really the **fitness** rewarding bang-bang climbs. Widening the range alone won't fix it;
  the comfort objective (§4) is the other half of the fix.
- `GenomeSpec.at_bounds(genomes)` reports, per gene, the fraction of genomes at the lower or upper
  bound (log0 genes are reported as `zeroed` instead). `suggest_widened_range()` proposes a ×4 widening.
  These suggestions are not applied automatically. `run_evolve.py` writes `at_bounds.json` for the best
  genome and for the last 10 generation-bests.
- The legacy preset keeps the original ranges exactly (`legacy_pitch_ranges: true`).

## 3. Aircraft profiles and range derivation

`aircraft_profiles/<name>.json` holds `AircraftProfile` data: mass, Ixx/Iyy/Izz, S, b, c̄, Vstall/Vcruise/Vne
(MMO optional), n-limits, max control deflections, max thrust, optional control derivatives
(Cm_δe, Cl_δa, Cn_δr per rad, or `*_per_norm` when the JSBSim model is written per normalized deflection, as the T38 is),
the design point (KCAS, altitude), `pitch_cmd_limits_deg` (pitch-command clamp relative to trim), `sim_envelope`
(`min_kcas`, `nz_limits` that end a run), optional `gain_overrides`, and `sources`.
Every profile has `approximate: true` and gives a source per field group. Geometry and inertia come from the
stock JSBSim model XMLs, which are themselves approximate. Speeds and limits are rounded public
figures. Fields with no good source are marked "rough" or use a generic value, and the loader emits a warning.

| profile | role | design point | notes |
|---|---|---|---|
| `c172x` | reference (legacy task), **Phase 1** | 100 KCAS, 4000 ft | derivatives and surface limits read from c172x.xml; clamp −8/+12° and envelope = sim.py constants |
| `t38` | jet trainer, **Phase 1** | 300 KCAS, 10 000 ft (≈M0.55) | JSBSim T38 (Aeromatic, BETA): geometry, inertia, per-norm Cm_δe/Cl_δa/Cn_δr, J85 mil thrust ×2; speeds and g limits rounded public figures; Vstall rough; clamp −10/+15° |
| `b737` | airliner, **Phase 1** | 250 KCAS, 10 000 ft (≈M0.46) | JSBSim 737 (generic, BETA): geometry, inertia, Cm_δe/Cl_δa from the Mach tables at ≈M0.45, surface limits from its FCS, CFM56 mil thrust ×2; clamp −5/+10° |
| `global5000` | business jet. JSBSim has no light jet, so this fills that slot | 260 KCAS, FL350 | generic derivatives |
| `a320` | airliner | 255 KCAS, FL350 (≈M0.80) | generic derivatives |
| `f16` | fighter | 350 KCAS, 10 000 ft (FD/ER trim point) | FD's prepared copy; flex EA/CG 0.40/0.43, point masses placeholder 0 / Pilot 1 / wings 2–3 (FD INTERFACE.md §5). FBW: stick = pitch-rate/g and roll-rate demands, so controller bounds are **provisional, untuned** (genes flagged, warning on load). Load/trim smoke-tested only |

global5000, a320 and f16 are extras: their ranges are derived, but they haven't been flown in closed loop.
Pitch clamps for every aircraft except c172x are design choices, not aircraft data, and are labelled as such.
The Phase-1 design points are benchmark conditions chosen for the altitude-hold task, not published cruise points.

**Rule.** The reference ranges are tuned for `c172x`. Each gene's range is multiplied by a factor chosen by
its `scaling` tag, so that a given normalized value has about the same closed-loop effect on every aircraft:

| tag | factor | reasoning |
|---|---|---|
| inner_pitch | CP_ref/CP, CP = q̄·S·c̄·\|Cm_δe\|·δe_max/Iyy | rad/s² of pitch per unit elevator command |
| inner_roll | same with b, Cl_δa, δa_max, Ixx | |
| inner_yaw | same with b, Cn_δr, δr_max, Izz | |
| outer_vertical | V_ref/V (TAS) | γ = ḣ/V, so the pitch needed per ft/s falls with speed |
| outer_heading | V/V_ref | turn rate = g·tanφ/V, so the bank needed for a given heading rate grows with V |
| speed | (m/T)/(m_ref/T_ref) | V̇ per unit throttle = T/m (static thrust, no lapse) |
| none | 1 | dimensionless structure deltas, time constants |

q̄ is taken at the design point (CAS≈EAS, no compressibility). This is a **single-point** estimate for
placing a search range, not a gain schedule. The factor is exactly 1.0 for the reference. Precedence:
derived < profile `gain_overrides` < task `gene_overrides`. `AircraftProfile.check()` flags inconsistent
data, for example a design point above MMO or outside Vstall..Vne. It caught an A320 design point at M0.84 > MMO 0.82
while I was writing the profiles. Example factors for a320: inner_pitch ≈ 10.9, outer_vertical ≈ 0.22.

## 3a. Task conditions (team decisions, 2026-10-06)

These apply to every non-legacy preset. `altitude_hold_legacy` sets none of them and gets the exact `sim.Scenario`
objects and sim.py constants.

- **Shared set.** Since 2026-10-06 the Phase-1 task, trim points, envelope, clamps, throttle cap and pitch/altitude
  gain bounds come from `aircraft_profiles/phase1_shared.json`, shared with Evolution Runner (presets with
  `"shared": "phase1"`; see ALIGNMENT.md). The bullets below describe what that set specifies.
- **Ramp corners** (`scenarios.ramp_accel_g`, 0.1 g in the shared set): the reference is a trapezoid in climb rate,
  with |ḧ_ref| ≤ 0.1 g at the start and end of each ramp. It is built from exact constant-acceleration phases
  (`ExtScenario._smooth_plan`); a command change mid-move continues from the current height and rate. Tracking is
  scored against this smoothed reference, which arrives about 2.9 s later than the sharp ramp on a 200 ft step.
  `ramp_accel_g: null` gives the sharp-cornered ramp.
- **Altitude ramp.** The reference moves toward each new commanded altitude at `scenarios.ramp_fpm` (600 fpm by
  default). A command change in the middle of a ramp starts from wherever the reference is. Tracking (`track_alt`,
  same ITAE-like formula) is scored against the **ramped** reference; the ITAE clock starts when the command changes.
  The overshoot diagnostic is measured against the *commanded* altitude.
- **Reference-rate feed-forward** (`controller.alt_ref_ff`, on in the new presets): the outer-loop D term acts on
  (ḣ − ḣ_ref) instead of ḣ. Without it, the D term fights a steady ramp, and the GA would need high kp_alt to hold
  the ramp, which brings back the pinning. This is my addition, not a team decision. It's one flag if you want it off.
- **Pitch-command clamp** comes from the profile's `pitch_cmd_limits_deg`, or from the preset's
  `controller.pitch_cmd_limits_deg` if set. sim.py's `PITCH_CMD_LIMITS_DEG` (−8/+12°) is only the fallback, and it is
  overridden in `sim_ext`, never edited. The c172x profile keeps −8/+12°.
- **Aircraft conditions**: JSBSim model, start altitude and speed (the design point; the 4000→4200→4000 ft profile is
  shifted to that altitude) and envelope limits all come from the profile. Disturbance draws stay those of `sim.make_scenarios`.
- **Helper loops on other aircraft.** When the roll and speed blocks are disabled, the backend gets their
  profile-scaled defaults, so jets don't fly with C172-tuned wing-leveler and throttle gains. For c172x these values
  equal sim.py's constants (tested bit-identical).
- **Heading hold** (shared-set flag `heading_hold`, on for c172x/T38/737; the legacy task never reads the shared set).
  Without it the c172x drifts ~25° in heading in 90 s on every seed: the propeller's torque reaction (235 lb·ft roll,
  99 lb·ft yaw at trim) is trimmed by -0.075 aileron, but the P-only wing leveller replaces the trim aileron with
  `-0.05·φ`, so it settles at φ ≈ +1.5° (with β ≈ 0) and turns at g·tanφ/V ≈ 0.28°/s. The jets have no prop torque and
  drift 0°. The fix reuses the `roll_heading` block's outer loop only (genes `kp_hdg`, `ki_hdg`; the block's inner
  `kp_roll`/`kd_roll` stay at the profile-scaled wing-leveller defaults): `e_ψ = wrap180(ψ_target − ψ)`,
  `φ_cmd = clip(kp_hdg·e_ψ + ki_hdg·∫e_ψ, ±bank_limit)` (integrator only if ki_hdg > 0, authority ±10° bank), aileron =
  `clip(−kp_roll·(φ − φ_cmd) − kd_roll·p, ±0.5)`. Bank limit = min(standard-rate bank at trim TAS, 25°), rounded down:
  c172x 16°, T38 25°, 737 25°. Gene ranges = reference (c172x) range × V/V_ref (TAS), like the other outer loops:
  kp_hdg 0.05–5 log, ki_hdg 1e-5–0.1 log0 on the c172x. Cost term `track_heading_rms` = RMS(e_ψ)/5° with weight 0.01:
  the smallest swept weight that holds the c172x within 1° (0 → 15° drift, 0.003 → 5°, 0.01 → 0.7°, 0.03 → 0.7°)
  without hurting track_alt. Once the heading is held the term is ~1 % of the cost, so it doesn't pull the GA toward
  aggressive banking in gusts (which costs altitude), but an unheld 25° drift costs ~15 % of the total. A preset can
  set `"heading_hold": false` (bit-identical to the v4 task, tested) or its own `track_heading_rms` weight. Hand-off
  for Evolution Runner: HANDOFF_heading_hold.md.
- **Presets v4 / v5** (2026-10-06). `phase1_v4` freezes the current default (everything above, heading hold on) so it
  stays reproducible bit for bit (tested against `hdg_after_c172x_s1`, `hdg_after_t38_s1`, `hdg_after_b737_s1`).
  `phase1_v5` is an opt-in candidate that adds two things, each a shared-set flag (`task.v5` in phase1_shared.json,
  `enabled_by_default: false`; preset keys `hold_quality` / `disturbance_scenario`; both need `"shared"`):
  - **Hold quality** (`hold_osc`). *Capture* is defined by the reference, not the aircraft, so it can't be gamed: a
    hold window is a stretch where the altitude reference equals the commanded altitude with zero reference rate,
    minus its first `hold_settle_s` = 5 s (the ramp's capture transient). `hold_osc` = RMS over all hold samples of
    (e − mean of e in that window), e = h_cmd − h, divided by `hold_ref_ft` = 5 ft. It penalises oscillation, not a
    standing offset (that is track_alt's and the integrator's job). Weight 0.1 from a sweep on c172x seed 1
    (`runs/v5_sweep_w*`): hold_osc 0.526 (v4 genome) → 0.443 / 0.373 / 0.254 at 0.03 / 0.1 / 0.3, calm hold p-p
    6.4 → 5.6 / 3.9 / 3.1 ft, v4-task cost 0.1960 → 0.1971 / 0.1951 / 0.2228. 0.1 is the largest weight with no loss on
    the v4 task; 0.3 buys its hold quality with more elevator activity (+14 % v4 cost).
  - **Downdraft scenario.** One extra scenario appended to the 3 standard ones (same mean aggregation): the calm
    scenario holding its start altitude (no step, no wind, no gusts) with a sustained downdraft from t = 10 s,
    1-cos onset over 4 s, added to `atmosphere/wind-down-fps`. Size = V_TAS·tan(1.5°), i.e. the same 1.5° air-relative
    flight-path offset on every aircraft, which a P-D altitude loop can only cancel with a standing altitude error
    (the ITAE weighting makes that late error expensive) and an integrator cancels exactly: c172x 4.69, T38 15.43,
    737 12.86 ft/s. Diagnostics: `draft_residual_ft` (mean e over the last 20 s, + = below) and `draft_max_err_ft`.
- `sim_ext` sets every engine's throttle (T38 and 737 have two engines). For non-c172x models it reports
  `load_failed`/`trim_failed` as a status instead of crashing.

## 3b. Flight Dynamics integration (2026-10-06, non-legacy presets only)

Model: FD's `flexwing.py` + `coupled_sim.py` in `$FLIGHT_DYNAMICS_DIR` (default `flight_sim_3d/flight-dynamics`),
imported read-only through `fd_bridge.py` (no bytecode written there; `coupled_sim.ensure_root` is never called, a
missing aircraft copy raises instead).

Applied to every `ExtScenario` run, i.e. all presets except `altitude_hold_legacy`, which still gets plain `sim.Scenario`
objects and sim.py's stock c172x path (re-verified IDENTICAL against results/example after these changes):
- aircraft loaded directly from FD's prepared copies `flight-dynamics/jsbsim_root/`. Since 2026-10-06 FD's
  `prepare_aircraft` strips all network I/O at the source (enforced by FD's tests), so our /tmp socket-stripping copy is
  gone. Instead `fd_bridge.check_no_network_io` scans the model's XML files once per process before every non-legacy
  load and raises `NetworkIOError` if any `<input|output>` declares a port/protocol or a SOCKET/FLIGHTGEAR/QTJSBSIM
  output. That makes loading e.g. the venv's stock 737 (telnet 5137 + QTJSBSIM 5139 inputs) a clear error. c172x/T38/737
  best-genome costs are bit-identical before and after the switch. Payload point-mass index 0 on T38/737/f16 is FD's
  placeholder, which the robust payload perturbation already uses;
- a trim throttle above `throttle_max` → `trim_failed` (as in evolution/); mixture is set on every engine;
- `gear/gear-cmd-norm = 0` before `run_ic` (every stock model loads gear-down). For the fixed-gear c172x this changes
  nothing: costs are bit-identical with and without FD root + gear-up (tested);
- trim stays `do_simple_trim = 1` (full trim) exactly as sim.py; trim mode 0 is not used anywhere;
- 737 min speed 195 KCAS (built-in trim fails below ~190 KCAS at 10 kft); T38 min speed 180 KCAS (shared set);
- T38 `throttle_max` = 0.5 (throttle pos = 2×cmd, so 0.5 = MIL; Phase 1 is MIL only). Generic `controller.throttle_max`
  override exists;
- Phase-1 trim points = profile design points: c172x 100 KCAS/4,000 ft, T38 300 KCAS/10,000 ft, 737 250 KCAS/10,000 ft.

**Flex mode** (opt-in; `phase1_flex` preset or `"flex": {"enabled": true, "mode": "twoway", "substeps": 2}`; rigid is the default):
0. *Genes → FD*: only the four structure genes go through FD's strict, range-checking `overrides_from_genome`, which
   maps (s, r) via `tied_stiffness`. Nothing else in the gains dict is ever forwarded. Chord axes and tip mass are FD's
   fixed per-aircraft values. A task config that tries to set them is rejected, with a "CG must be behind the elastic
   axis" message when it would put the CG at or ahead of the EA. Our profiles keep copies of FD's values, checked
   equal at load and in tests.
1. *Pre-screen* (`flexwing.margin_terms`, ~6–20 ms, before any flight). Flutter (the conservative minimum of FD's
   QS p-method and steady coalescence) and divergence margins (× V_D) are constraints. They are always finite,
   capped at 3.0, with `*_not_found_below_cap` flags; reports carry the capped margin plus the flag, and nothing
   depends on an infinite margin. `margin_error` or NaN counts as a fail.
   - Any margin < 1.0 → the genome fails without flying (cost 2000 = 2·FAIL_BASE, status `aeroelastic_<kind>`,
     violation > 0 for NSGA-II). This is intended at the soft corner s ≈ 0.6–0.86 with r ≈ 0.8 (FD min 0.84–0.90).
   - 1.0 ≤ margin < 1.2 → FD's hinge penalty `w·((1.2−m)/0.2)²` is added once to the aggregated cost (and to the
     `structural` entry in Pareto mode).
2. *Fly*: `coupled_sim.make_coupler` builds the wing from the genes and applies the wing-mass change to JSBSim before
   trim; after trim the FDM is wrapped in `coupled_sim.FlexFDM`, which steps the structure on every `run()`. The
   controller loop is unchanged. `flexwing.telemetry_channels` adds wing_root_bending(_L/_R), tip_deflection and
   tip_twist to telemetry and `flex_params` to the result.
3. *Score*: structural objective in the table in §4; root moment > 1.5× limit (ultimate) → scenario fails with cost 2000.
Cost: ~1.2 s per 90 s scenario coupled vs ~0.25 s rigid on this box today (FD measured 3.5 s vs 0.6 s under load).

## 3c. Phase 2: `phase2_flex` on FD flex v2 (2026-10-06)

**Source of truth.** The structure genes are built at load time from FD's `flexbody.gene_schema(asymmetric)`
(`genome_schema.genes_from_fd_schema`; block `structure_v2`), appended after the phase1_v4 controller genes, so names,
order, ranges, log/linear scale and baselines track FD's spec (INTERFACE_v2.md). 12 genes, or 14 with
`flex.asymmetric: true` (FD's wing_asym_ei_delta / wing_asym_nsm_delta). **Off by default and not for Phase 2 runs:**
they only matter once lateral or roll scenarios exist (FD), and the current scenario sets have none; the adapter
warns when they are switched on (`experiments/phase2_flex_asym.json`). Every decoded structure genome goes through FD's `decode_genome_v2`, which raises on unknown, v1, fixed, NaN,
out-of-range or unflagged asymmetric keys; a task cannot override v2 ranges (`gene_overrides` on a v2 name raises), and
`structure` (v1) + `structure_v2` together raise. The v1 path (`phase1_flex`) is untouched.

**Config.** `"blocks": {"pitch_altitude": true, "structure_v2": true}`,
`"flex": {"enabled": true, "version": 2, "asymmetric": false, "substeps": 2}` (two-way only). Chord axes and tip mass
stay FD's fixed per-aircraft values (task keys for them raise). The adapter checks FD's prepared v2 root read-only
(`fd_bridge.require_root_v2`; we never call FD's `ensure_root_v2`, which writes).

**Flight.** `sim_ext` mirrors FD's `FlexHookV2`: `apply_mass_v2` before the IC (structural mass change in JSBSim),
`FlexBodyFDM(FlexBodyCoupler(model, "twoway", substeps))` after trim, `response_terms_v2` after the flight. We use our own
`sim_ext` (bit-identical to the Phase 1 sim) instead of `flexeval.evaluate`, which imports evolution/sim.py.

**Constraints (once per genome, before flying).** `fd_bridge.precheck_v2` takes FD's per-block `margins_v2` (wingR, wingL,
empennage_pitch, empennage_yaw) and recomputes the conservative minimum over bodies and methods:
flutter = min over blocks of min(flutter_margin_qs, coalescence_margin, flutter_margin); divergence = min div_margin;
reversal = min of every `<ctrl>_reversal_margin`. Every value is FD's capped one (cap 3.0); a not-found flag means "no
instability below the cap", so the value used is the cap and the flag is reported (`aeroelastic.not_found_below_cap`);
nothing uses an uncapped margin. Non-finite values or `margin_error` → margin 0.
- min < 1.0 (FD's full-fidelity gate) → not flown, every scenario `aeroelastic_<kind>`, cost 2000 (FAIL_COST).
- 1.0 ≤ m < 1.2 → `w·((1.2−m)/0.2)²` per kind (FD's weights: flutter 1, divergence 1, reversal 1), added once.
- m ≥ 1.2 → exactly 0: no reward for more margin (FD, 2026-10-06). Margins enter the cost only through this hinge and
  the gate (`fd_bridge.margin_penalties_v2`); not-found = the 3.0 cap = 0. phase1_flex (v1) uses FD's
  `flexwing.margin_terms`, which has the same shape (tests: `test_v2_margins_penalised_only_below_1p2_never_rewarded_above`,
  `test_v1_margins_penalised_only_below_1p2_never_rewarded_above`).
- The conservative values are never higher than FD's overall ones (tested); at the baseline they equal FD's table
  (c172x 1.228/1.669/1.473; T38 J_flutter 0.0361; 737 0.0132).

**P2.5 (2026-10-06 ~13:50 MST, Corleone approved before P3-A).** FD added `J_wing_tip_bm_limit` (weight 1.0) to
`SIZING_TERMS` / `TERM_KEYS` (now 24 keys) and raised `wing_nsm_root` / `wing_nsm_tip` floors to **1.0–1.25** (decode
rejects < 1.0). Genome picks sizing keys from FD live (`fd_bridge.sizing_terms_v2()` / `flexbody.SIZING_TERMS`); NSM
ranges come from `gene_schema()` (no copied JSON). Recorded model_versions = `model_versions_post_p25.json`; the
previous post-mass set is accepted with a warning only (ER pilots still pin those; raise only on unknown). Phase 2
pilots stay as-is for Corleone's push plan; Phase 3 sketch unchanged. Soft `wing_ei_taper_4=0.75` → tip term ≈ 0.0175
on c172x (baseline tip = 0). `struct_v2_source` still defaults to `"fd"`.

**ER alignment (2026-10-06 08:35, decisions sent to Evolution Runner for the Phase 2 pilot).** phase2_flex =
ER `configs/phase2_pilot.json`: `fitness.params.struct_v2_source: "fd"`, init σ 0.10, ki_alt ≤ 0.5 (phase2-only
`gene_overrides`; the shared v4 bound stays 0.05), mass-credit clip off. Controller terms exactly v4 (track 1, effort 2,
comfort 0.05, heading 0.01; 600 fpm ramp, 0.1 g corners, feed-forward). Cross-checked bit for bit against ER's evaluator
(CROSSCHECK_phase2.md, `crosscheck_phase2.py`, `test_fd_mode_reproduces_evolution_phase2_smoke_costs_bitwise`).

**`struct_v2_source: "fd"` (phase2_flex default).** structural_v2 = FD flexeval's structural cost, nothing of ours:
per scenario `Σ response_terms_v2 (8: J_bm_rms, J_bm_peak, J_tip, J_twist, J_tail_bm_peak, J_fus_bm_peak,
J_wing_torque_peak, J_wing_ip_peak)` + the pre-flight sum over `PRE_TERMS["full"]` (J_flutter/div/reversal_margin,
J_mass, J_smooth, 5 sizing terms; once per genome, carried by every scenario). FD weights (StructWeightsV2 defaults).
- Gate and margin terms are FD's (`margin_terms_v2(..., gate=1.0)`: FD's margins, FD's hinge²); our separate
  aeroelastic margin penalty is 0 in this mode (it would count FD's J_*_margin twice). Our conservative minimum is
  still reported; on 288 sampled genomes (3 aircraft × uniform / seeded σ 0.10) it gave the same gate decision and
  the same margin penalty as FD's in every case.
- Cost composition is flexeval's, in flexeval's order: ok scenario `sim_cost + Σresp + pre_sum`; envelope failure
  `sim fail cost + pre_sum`; FD ultimate fail `2000 + pre_sum`; gate fail: not flown, 2000. Mean over scenarios.
  With a structural_v2 weight ≠ 1 the adapter warns (no longer flexeval's cost).
- **flexeval contents check.** flexeval's TERM_KEYS has 23 keys, 5 of them controller terms (track, effort,
  comfort, heading, hold) and flexeval's `cost` includes the Runner's whole sim cost (the controller cost). So
  flexeval's total cost must NOT be used as structural_v2 (it would count the controller twice); we take only the
  J_* add-on (18 terms). hold = 0 in v4 (w_hold 0). FD's J_bm_rms (wing RMS / 1-g moment) depends on the controller's
  load alleviation, but it is a structural term and our own RMS/peak terms are off in this mode, so nothing is
  counted twice.
- Reported: `fd_struct_terms` (FD's structural J_* as flexeval reports them, 19 after P2.5 tip term: pre-flight once, flown = mean over ok flights),
  `fd_pre_sum`, and our formula's terms as `genome_structural_v2_terms` (diagnostic only).
- The clip cannot be combined with "fd" (load error). `"genome"` = the formula below, kept for A/B.

**Structural objective `structural_v2`, `struct_v2_source: "genome"`** (A/B only), per scenario:
`0.1·mean_b(peak_b/limit_b) + 0.5·mean_b(RMS_b(M−M₁g)/limit_b) + J_mass + J_smooth + FD sizing terms + FD flown hinge
terms`, over the five root moments b ∈ {wing (worse side), HT (worse side), VT, aft fuselage vertical, aft fuselage
lateral}, limits from FD (`bm_allow` for the wing, `allowables_v2` for tail/fuselage; strength scales with stiffness,
as in FD).
- `J_mass = 0.3·Σ_b Δm_b/m_flexible` over b ∈ {wing, HT, VT, fuselage}, from FD's per-body `mass_summary()`; equals
  FD's J_mass exactly (tested). Since FD §12 (08:02 MST) the structural mass is `0.5 + 0.5·s` of baseline
  (minimum-gauge floor), so a fully softened body saves at most half its mass.
- **FD §12 sizing terms** (pre-flight, once per genome, injected into every scenario): `J_wing_bm_limit`,
  `J_tail_bm_limit`, `J_fus_bm_limit` (w 1.0), `J_wing_torque_limit`, `J_wing_ip_limit` (w 0.1), each
  `w·max(0, design demand / (baseline design load × s) − 1)²`, from `margin_terms_v2(...)["terms"]`, FD weights and
  values unchanged (`fd_bridge.SIZING_TERMS_V2`, names tested against `flexbody.SIZING_TERMS`). 0 at the baseline;
  all stiffness genes 0.6 → ≈1.47 against a mass credit of ≈−0.03 (c172x/T38/737, tested). Note: they are 0 only
  when every body's gene is ≥ 1; a stiffer (heavier) tail raises the fuselage demand (c172x tail 1.5, fuselage 1.0 →
  J_fus_bm_limit 3·10⁻⁴), which FD's "0 for genes ≥ 1" wording does not mention.
- **FD flown hinge terms** (`V2_HINGE_TERMS`): J_bm_peak, J_tip, J_twist, J_tail_bm_peak, J_fus_bm_peak and the §12
  full-fidelity-only J_wing_torque_peak, J_wing_ip_peak (w 0.1), as FD computes them. A term missing from FD's
  result (e.g. at reduced fidelity) counts 0 (`.get(k, 0)`, tested). FD's J_bm_rms is left out (our RMS ratio covers it).
- `J_smooth` = FD's 0.05 EI-taper smoothness term.
- **Mass-credit clip (off; A/B only).** `fitness.params.struct_v2_mass_credit_clip: ["ht","vt","fus"]` (Δm_b →
  max(0, Δm_b) for tails/fuselage) was the interim fix for the tail/fuselage mass exploit, 08:00–08:15 MST. FD §12
  (minimum gauge + sizing terms) supersedes it, so phase2_flex no longer sets it; the code path stays for A/B runs
  and prints a warning when set. As-run record: `experiments/phase2_flex_uniform_init_clip.json`.
- **Double counting (proposal, weights not changed).** Our `0.1·peak/limit` and FD's `J_*_bm_limit` both penalise
  soft structure, but not identically: FD's is a pre-flight hinge on the design load (0 for s ≥ 1, steep below);
  ours is linear in the flown peak over the stiffness-scaled limit, so it keeps rewarding stiffness above s = 1 and
  also carries the controller's load alleviation. They overlap for s < 1. The RMS term (fatigue / load alleviation)
  has no FD counterpart. Proposal: keep `load_rms`; change `load_peak` to use the baseline (unscaled) limit so it
  measures only load alleviation by the controller, or drop it (w_peak 0) and rely on FD's sizing + J_bm_peak.
  Not applied: needs the team's agreement.
- **FD model_version.** `fd_bridge.FD_V2_MODEL_VERSIONS` records FD's P2.5 versions (v2_results/
  model_versions_post_p25.json: full c172x e11b8214, T38 8bf7a250, 737 eeb82fb9, f16 8baca00c; reduced 68dc59aa /
  ce107fcf / d3198780 / 66a7d816). Previous post-mass strings are kept in `FD_V2_MODEL_VERSIONS_PREV_POST_MASS` and
  only warn (never raise) during the ER pilot transition. Completely unknown versions warn by default and raise when
  `flex.model_version_check: "raise"`. Stored in `flex_constants_used` with fidelity and gate.
- **Fidelity.** phase2_flex flies FD's **full** fidelity (flexbody v2 two-way) with hard gate 1.0 = FD's
  `MARGIN_GATE["full"]`. FD's reduced gate 0.9 belongs to the Runner's reduced path (v1 FlexWing on the projected
  genome), which genome does not use; `fd_bridge.V2_MARGIN_GATES` mirrors FD's table for a future reduced mode.
  phase1_flex stays the v1 contract (v1 genes, flexwing gate 1.0, bit-identical). At reduced fidelity FD applies
  only the projected v1 wing point masses to JSBSim (no floor, no tail/fuselage), while J_mass and the sizing
  terms come from the v2 floored model, so FDM mass and J_mass disagree there; FD documents this (§7: keeps v1
  bit-identical). It does not affect phase2_flex (full applies the floored v2 masses).
- FD's ultimate check (`structural_ultimate*`) fails the scenario (cost 2000).
- Weights. At the baseline structure with the v4 c172x best gains the load ratios are peak 0.07–0.45, RMS 0.01–0.03;
  the terms are 0.018 (peak) + 0.009 (RMS) ≈ 11 % of the controller cost (0.21). If every body reached its limit the
  peak term would be 0.1, half the controller cost; ±10 % structural mass is ±0.03 (FD's w_mass kept). So the
  controller still dominates while load and mass trade-offs are visible. In the first c172x smoke runs (before
  FD's floor and sizing) the GA used the mass credit (fuselage stiffness toward its 0.6 floor, Δm −6 to −13 %)
  while peaks stayed ≤ 0.65 of limit; see README "Phase 2 smoke test" for the clip and post-§12 reruns.

**Generation 0 (`init`).** `init_pop.py`. evolve.py's first draw `rng.random((pop, n))` is kept, so the controller genes
start exactly as in the current presets; with `mode: "baseline"` the `structure_v2` genes are replaced by
encode(FD baseline) + N(0, σ) (normalized units, clipped; σ 0.10 in phase2_flex since 08:35 = ER's pilot, was 0.05). Ranges stay FD's full ranges.
`seed_runs: {aircraft: [run dirs]}` copies earlier best genomes (e.g. v4) into the first individuals (structure at
the exact baseline). Without `init`, run_evolve does not touch `evolve.ga` (legacy, v4, v5 and phase1_flex unchanged).
`evolve_pareto.py` honours the same config.

**ki_alt (v5 and phase2_flex).** `presets/phase1_v5.json` and `presets/phase2_flex.json` raise the ki_alt upper
bound to 0.5 (gene_overrides, task-local); phase1_v4 and the shared set keep 0.05 (test pins v4 bit for bit). See
HANDOFF_phase1_v5.md.

## 4. Multi-objective fitness

Each scenario is flown once. Every objective declares the telemetry channels it needs.
If a channel is missing, the objective is **skipped and reported** (`result["skipped"]`, plus a load-time
`SKIP objective …` warning). Missing objectives are never approximated.

| objective | definition (per scenario) | needs |
|---|---|---|
| track_alt | legacy ITAE-like altitude error (`sim.py` track) | — |
| effort | legacy elevator total variation per second | — |
| track_speed / effort_throttle | mean \|V−V_tgt\|/5 kt · throttle TV/s | vc, v_target, throttle |
| track_heading / track_bank / effort_aileron | mean \|ψ err\|/5° · mean \|φ−φcmd\|/5° · aileron TV/s | psi, psi_target, phi, phi_cmd, aileron |
| sideslip | mean \|β\|/1° | beta |
| **comfort** | weighted sum of normalized terms: RMS\|n−1\|/0.1 g, max\|n−1\|/0.3 g, RMS jerk (nz low-passed at 0.2 s)/0.2 g/s, RMS(max(0, \|θ−θtrim\|−5°))/1°, RMS(max(0, \|q\|−3°/s))/1°/s | nz, theta, q |
| **structural** | flex run (see §3b): 0.25·RMS(M−M₁g)/M₁g + 2·max(0, peak/limit−1)² + tip-deflection and tip-twist hinge² + 0.3·Δm_wing/m_wing (FD's `response_terms` plus a wing-mass term, so stiffer is not free); rigid run: proxy M_root ∝ n·W: peak n/n_limit + RMS(n−1)/(n_lim−1) | nz (or the flex channels) |

The pitch-attitude and pitch-rate excess terms target the ~15° climb directly: the legacy best genome
scores pitch_excess ≈ 2.0 and reaches 14–15° pitch (about 14° above the 0.8° trim).

**Robustness across scenarios.** The per-scenario scalar costs are aggregated with `mean`
(legacy), `worst`, `cvar` (mean of the worst ⌈α·N⌉) or `mean_cvar` = (1−λ)·mean + λ·CVaR. Scenario sets:
- `legacy`: `sim.make_scenarios` unchanged.
- `robust`: the same disturbance draws (wind, Gauss–Markov turbulence, 1-cos gust) plus payload ±mass and CG
  shift through real JSBSim point masses, and controller-side sensor noise on h, ḣ, θ, q. Heading steps are optional.
  Scenario 0 stays calm and nominal. Mixing several aircraft in one fitness is still `NotImplementedError`;
  instead, run one task per aircraft (`--aircraft c172x|t38|b737`).

**Modes.**
- *Scalar (default)*: cost = Σ w_k·obj_k per scenario, then aggregated. This is a drop-in for evolve.py.
  The default preset `phase1_default` uses `{"track_alt": 1, "effort": 2, "comfort": 0.05}` (tuning in §6).
  `{"track_alt": 1, "effort": 2}` + mean is **bit-identical** to `sim.evaluate`.
- *Pareto*: the result also carries `pareto` (a vector over `pareto_objectives`, which can include the
  pseudo-objectives `robust_cvar` and `robust_worst`) and a `violation`. `nsga2.py` implements a constrained
  non-dominated sort (feasible beats infeasible; between infeasible solutions, smaller violation wins) plus
  crowding distance. `evolve_pareto.py` keeps ga.py's variation operators, orders parents by (front, −crowding)
  for its rank selection, and does (μ+λ) environmental selection.
- **Envelope violations** keep the legacy hard penalty, FAIL_BASE·(1 + fraction not flown). That scenario's
  objectives become NaN, and in Pareto mode the individual is infeasible.

## 5. Integration

```
run_evolve.py --task <preset> -- <evolve.py args>
  adapter.load_task  -> GenomeSpec (profile-derived ranges) + FitnessConfig + scenario set
  adapter.install    -> sys.modules["genome"], sys.modules["sim"] = shims
  import evolve      -> the ORIGINAL flight_sim/evolve.py, now bound to the shims (fork start method)
```
`ga.py` and `evolve.py` have **zero changes** and need no patch. `sim_ext.py` is a line-for-line extension of
`sim.simulate`. It reuses sim.py's constants, `Scenario` and `_new_fdm`, records telemetry, and also uses
roll/heading and speed genes when they are present. Its results are bit-identical to `sim.simulate` (tested).
Backends declare which genes they `consume`. Enabling a block whose genes have no dynamics
(`yaw_damper` today; `structure` unless `flex.enabled`) raises an error unless `allow_inert_blocks: true`, so the GA
never evolves genes that do nothing without someone noticing. With `flex.enabled` the structure genes are consumed by
FD's coupled_sim and the guard is lifted for them.

## 6. Limits / not done
- c172x, T38 and 737 are flown (T38/737 only sanity-checked plus short runs; their derived ranges are not tuned). The
  extra profiles (global5000, a320, f16) give derived ranges only.
- Yaw damper has no dynamics yet. Structure has dynamics only in flex mode, and FD's structural data is notional
  (no GVT/flutter test data): margins and loads rank designs, they don't certify anything.
- Range derivation is single-point and uses generic control derivatives where the profile has none.
- Comfort references (0.1 g, 0.2 g/s, 5°, 3°/s) and the preset weights are first guesses. Measured: comfort
  w=0.02 left the climb almost unchanged (15.1° → 14.7°). w=0.25 brought it to 6.2° and ~950 fpm, but tracking got
  5.7× worse and the capture overshoots by ~45 ft. The step target with ITAE-style tracking basically requires an
  aggressive climb. A rate-limited altitude reference (e.g. 500–700 fpm) would probably fix this better than a heavier
  penalty, but that changes the task definition and is Corleone's call.
- The v2 preset with sensor noise moved the optimum to low kp_alt and high ki_alt, and ki_alt now sits at its 0.05
  upper bound. Its integral contribution is still capped at 5° by sim.py's ALT_I_LIMIT_DEG.
- Phase-1 c172x A/B (pop 48 × 40, seed 1; table in README):
  - With the sharp ramp, ki_alt sat at its 0.05 bound. Widening it to 0.5 (and, separately, also lowering ki_pitch's
    floor 100× to 1e-7) did *not* help: 0.2656 / 0.2685 vs 0.2512, and the GA settled at ki_alt ≈ 5e-4, so the pin
    was a search artefact rather than a real demand. The widened ranges stay as experiments.
  - 0.1 g ramp corners help clearly (0.2063, calm nz 0.89–1.11, no genes at bounds) and are now the default.
  - Feed-forward off: about neutral with the sharp ramp (0.2489 vs 0.2512, but track 0.096 vs 0.076) and clearly
    worse with smoothed corners (0.2704 vs 0.2063). It stays on.
  - Each A/B is one seed; GA noise between seeds hasn't been measured.
- Flex: the earlier exploit (CG ahead of EA → no flutter) is closed by FD's Phase-1 decision; see §3b. Superseded
  runs are kept under `runs/*_SUPERSEDED*` with a note.

