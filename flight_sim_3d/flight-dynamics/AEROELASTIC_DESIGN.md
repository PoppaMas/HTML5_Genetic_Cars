# Minimal aeroelastic flex-wing model for the GA (no CFD)

Files:
- `flexwing.py`: model, margins, genes and fitness terms, aircraft preparation.
- `coupled_sim.py`: JSBSim coupling, maneuvers, composition with the repo's `sim.simulate`.
- `test_flexwing.py`: 23 tests.
- `verify_external_reactions.py` → `external_reactions_proof.json`.
- `flex_demo.py` → `flex_demo_{c172x,737,T38}.png` and `flex_demo_summary.json`.
- `INTERFACE.md`: genome field names, ranges and Phase-1 aircraft facts.

Environment: JSBSim 1.3.1, dt = 1/120 s. The aircraft are copied to `jsbsim_root/`; the venv data is never modified.

## 1. Structural model (per semi-wing, both wings integrated independently)

The semi-wing is a straight beam along the elastic axis, from the fuselage side (y0 = 0.10·s) to the tip. Each semi-wing has
3 assumed modes, η = [b1, b2, t1]:
- two uniform-cantilever bending shapes φ_k(ξ) (β = 1.8751, 4.6941);
- one torsion shape ψ(ξ) = sin(πξ/2).

There are 16 strips. Chord is trapezoidal from JSBSim span and area plus the profile taper.
- Mass per span m(y) ∝ c(y). The total comes from the gene mass model (INTERFACE §1b).
- Section pitch inertia I_α = m((r_g c)² + x_θ²), with r_g = 0.25 and x_θ = (x_cg − x_ea)c.
- EI(y), GJ(y) ∝ (c/c_root)³. Their levels are calibrated so the baseline uncoupled frequencies equal `f_b1_hz`/`f_t1_hz`. Calibration uses the baseline x_ea/x_cg (fixed in Phase 1, so overrides of them do not change GJ). They are then scaled by the internal `ei_scale`/`gj_scale`, which the GA sets only through the tied genes (§5).
- Optional tip mass `tip_mass_frac` sits at 10 % chord (fixed at 0 in Phase 1; not a gene).

Rayleigh–Ritz matrices:

```
M = ∫ [m φφᵀ, −m x_θ φψᵀ ; ·, I_α ψψᵀ] dy (+ tip mass)       K = ∫ [EI φ''φ''ᵀ, 0 ; 0, GJ ψ'ψ'ᵀ] dy
C = diag(2 ζ ω_i M_ii), ω_i² = K_ii/M_ii (diagonal, from the uncoupled assumed-mode terms)
```

## 2. Aerodynamics (strip theory, quasi-steady + apparent mass)

Elastic angle-of-attack increment at strip i (streamwise, sweep Λ of the EA):

```
Δα_i = cosΛ·θ_i − sinΛ·w'_i − ẇ_i/V + d34_i·θ̇_i/V   (+ β·w'_i·(±1) elastic-dihedral sideslip term)
ΔL_i = q·κ(M)·a·c_i·Δy·Δα_i     acting at c/4, i.e. arm e_c = (x_ea − 0.25)c ahead of the EA
```

- a = DATCOM/Helmbold CLα(AR, Λ). κ(M) = CLα(M)/CLα(0) with the Prandtl–Glauert β inside DATCOM, and Mach clamped at 0.9.
- The Theodorsen non-circulatory pitch terms are included: L_nc = (π/2)c²θ̇ and M_nc = −(π/2)c²·d34·θ̇.
  - Without them, the circulatory 3/4-chord term gives spurious negative torsional damping: QS "flutter" appeared at 25 kt.
- Generalised aero: Q_aero = q·κ·A_K·η + (q/V)(κ·A_C,circ + A_C,nc)·η̇. These go into the implicit system matrices (aero stiffness and damping), not into the explicit right-hand side.

External (rigid) loads per frame, all read from JSBSim state:
- Wing lift: L = −F_z,aero·cosα + F_x,aero·sinα, distributed with Schrenk's approximation (mean of elliptic and planform).
- Roll-rate damping load: q·κ·a·c·(p·y/V).
- Aileron strip loads use the FDM's own surface positions. The strip Cl_δa is calibrated to the FDM's Clda (c172x 0.23, 737 0.09, T38 0.11 per rad), because uncalibrated 2-D strips over-predicted roll and reversal.
- Inertia relief at the section CG: −m·g·Nz + m·ṗ·y.

## 3. Integration and coupling (two-way)

Each JSBSim frame (`FlexFDM.run()`):
1. `read_state(fdm)`: q̄, V_t, α, β, M, p, ṗ, Nz, lift, aileron positions.
2. Newmark average acceleration (implicit, unconditionally stable), with 4 substeps per frame in demos and 2 in the GA. Each wing uses a 3×3 closed-form inverse, and the external loads go through precomputed bases. The fast path matches the reference implementation to 1e-12.
3. Feedback relative to the **1-g trim shape** (`reference='trim'`). `initialize()` solves the static equilibrium at the trimmed state, so the feedback at t = 0 is exactly zero and JSBSim's own (rigid) tables keep owning the trimmed aerodynamics. Only elastic *increments* are fed back:
   - ΔL (BODY x = ΔL·sinα, z = −ΔL·cosα) as an `<external_reactions>` force at AERORP;
   - ΔL_roll and ΔM_pitch as an `<external_reactions>` moment.
4. Then the real `fdm.run()`.

**Why external_reactions** (vs editing aero tables or using `aero/function` increments):
- aircraft-agnostic: one XML insertion, merged into an existing `<external_reactions>` if present;
- physical units, and no per-aircraft aero-table edits or axis/sign conventions;
- exact bookkeeping;
- leaves the stock model bit-identical when the feedback is zero.

Proof (`external_reactions_proof.json`, `all_ok: true`):
- With zero commanded force, the copy is bit-identical to stock over the run.
- A commanded force appears exactly in `forces/fb{x,y,z}-external-lbs`.
  - Gotcha: force = magnitude·(x, y, z), and the direction is **not** re-normalised when set via properties.
- Moment readback equals mag·(l, m, n) + r(CG→AERORP)×F: (125.18, 24.10, 2.34) lbf·ft.
- A 2000 lbf·ft roll moment changes ṗ by 0.95438 rad/s² measured vs 0.95432 expected from L/Ixx.

Mass feedback: two injected point masses carry the gene-driven wing-mass delta, set before trim (INTERFACE §1c, tested).

## 4. Margins (cheap per-genome screening, about 6 ms)

- **Divergence:** smallest positive q with det(K − q·κ(M)·A_K) = 0. Mach comes from a fixed-point EAS→TAS iteration at the given density.
- **Flutter (quasi-steady p-method):** sweep EAS 0…3·V_D (120 points). Eigenvalues of the 6×6 state matrix [0 I; −M⁻¹(K − qκA_K), −M⁻¹(C − (q/V)(κA_C,circ + A_C,nc))]. The first oscillatory root with Re > 0 gives V_F.
- **Coalescence:** first q at which eig(M⁻¹(K − qκA_K)) turn complex (steady aero, conservative).
- `flutter_margin = min(V_F,QS, V_coal)/V_D` (criterion "min"). QS alone misses some soft-wing coalescence cases that coalescence catches. `div_margin = V_div/V_D`.
- **Always finite (since 2026-10-06).** Each margin is reported as min(value, `MARGIN_CAP` = 3.0). Searches cover the whole range up to the cap: QS over EAS 1 kt…3·V_D with the endpoint included, coalescence over q 0…9·q_D.
  - "Nothing found below the cap" sets `flutter_not_found_below_cap` / `div_not_found_below_cap` = true, and the margin is exactly 3.0. `f_flutter_hz` is 0.0 instead of NaN.
  - A numerical failure gives margin 0.0 with `margin_error` = true (conservative hard fail).
  - Previously "not found" returned `math.inf`. It happened when the section CG was ahead of the elastic axis (the inertial coupling changes sign, so the straight wing never coalesces and only diverges), or with no aero coupling at all.
  - Swept 737/T38 divergence (raw 23.5 / 25.7 V_D) now reads 3.0 with the flag.
- **Section-axis guard:** `x_cg ≥ x_ea + MIN_CG_AFT_OF_EA` (0.02 chord) is enforced by raising `ValueError` in `WingParams` and `FlexWing`. The only bypass is the test/validation context manager `unchecked_section_axes()`.
- Required margin is 1.2 (CS/FAR-25.629 style). The GA penalty is w·max(0, (1.2 − m)/0.2)², with a hard fail (cost 2000, sim skipped) below 1.0.

Validation tests:
- exact torsional divergence of a uniform wing;
- coalescence vs a semi-analytic typical section;
- tip deflection vs qL⁴/8EI and root moment qL²/2;
- 2nd/1st bending ratio 6.267;
- Newmark energy conservation and period;
- margin trends (softer → lower margins);
- CG ahead of EA → finite capped margin with flag; zero-aero wing → both margins capped; guard raises; per-aircraft default margins unchanged; tied-stiffness grid finite, monotone and ridge-free on c172x/T38/737; untied or out-of-range genome keys raise; prepared copies carry no network sockets; f16 copy layout, trim, zero-force identity and coupled determinism (23 tests in total);
- coupled run determinism; rigid vs flex differ in the right direction.

## 5. GA genes (flexwing.STRUCT_SCHEMA; genome names in INTERFACE.md)

Phase-1 decision (2026-10-06): chordwise placement is **fixed per aircraft**, not evolved. Elastic axis and section CG are
c172x 0.38/0.42, T38 0.40/0.42, 737 0.36/0.38, f16 0.40/0.43 (added 04:44 PT task; F-16 details and margins in INTERFACE.md §5); tip mass is fixed at 0 (confirmed 04:02 PT). Reason: with notional chord fractions the GA put the CG ahead of
the EA and obtained "infinite" flutter margins. Details, rationale and the Phase-2 option (constraint plus balance-mass cost)
are in INTERFACE.md §0.

| gene | unit | range | scale | effect |
|---|---|---|---|---|
| stiffness_scale (s) | × EI | 0.6–2.0 | log | overall stiffness; also strength and structural mass |
| torsion_bend_ratio (r) | × GJ / s | 0.8–1.15 | linear | torsion relative to bending (the flutter driver) |
| zeta | – | 0.005–0.05 | log | structural damping |
| nonstruct_scale | × non-structural mass | 0.8–1.25 | log | fuel/systems share of wing mass |

Stiffness genes are **tied** (decision 04:02 PT). `flexwing.tied_stiffness(s, r)` maps them to the internal parameters:

```
ei_scale = stiffness_scale
gj_scale = stiffness_scale * torsion_bend_ratio
```

`genes_to_overrides` / `overrides_from_genome` apply this mapping. They raise ValueError on any genome value outside the ranges
above (or NaN), and on independent `bend_stiffness_scale` / `torsion_stiffness_scale` / `ei_scale` / `gj_scale` keys.

Ridge rationale: with independent EI/GJ (0.6–2.0 each) the swept 737 and T38 show a narrow infeasible ridge where torsion
crosses 2nd bending (737 baseline t1 10.95 Hz vs b2 11.9 Hz). Examples: 737 EI 1 / GJ 1.25 → margin 0.52; T38 EI 0.6 / GJ 1 → 0.65.
A smooth GA landscape is broken by that ridge. Over r ∈ [0.8, 1.15] the margin is monotone in s and r on all three aircraft. The
737 cliff starts between r 1.15 and 1.20, so 1.15 is the upper bound.

Tied grid (12 s × 8 r, gene_range_margins.json), min flutter/div margin:

| aircraft | min over grid | min, s ≥ 0.8 | min, s ≥ 0.8 excl. corner (s < 0.86, r < 0.85) |
|---|---|---|---|
| c172x | 0.841 @ (0.6, 0.8) | 0.967 @ (0.8, 0.8) | 1.003 |
| 737 | 0.900 @ (0.6, 0.8) | 1.020 @ (0.8, 0.8) | 1.049 |
| T38 | 0.895 @ (0.6, 0.8) | 0.999 @ (0.8, 0.8) | 1.031 |

The corner s 0.8 / r 0.8 (GJ ×0.64) is below 1.0 on c172x and T38. The pytest allows it explicitly (≥ 0.96). Raising the r
lower bound to 0.85 would give ≥ 1.0 everywhere for s ≥ 0.8 (1.006 / 1.031 / 1.049); ranges were left as specified.

## 6. Fitness terms (compose additively with the repo's altitude-hold cost)

`evaluate_flex(gains, struct_genome, scenarios)` runs in two stages:
1. Margin screen. If it hard-fails, return 2000 and skip the simulation.
2. Run the repo's unchanged `sim.simulate` per scenario through a patched FDM factory (the FlexFDM proxy), plus the structural response terms.

```
J = J_repo(altitude hold, unchanged)
  + J_flutter_margin + J_div_margin                   w=1 each, hinge² below 1.2, fail < 1.0
  + J_mass   = 0.3 · Δm_wing / m_wing0                (negative for lighter wings)
  + J_bm_rms = 0.25 · RMS(M_root − M_1g) / M_1g
  + J_bm_peak= 2 · max(0, peak/limit − 1)²,  limit = n_limit · M_1g · stiffness_scale ; fail if peak > 1.5·limit
  + J_tip, J_twist = hinge² vs tip_defl_limit_frac·s, 3°
```

Measured composition (repo best gains, `make_scenarios(3, 1)`, c172x):

| genome | total cost | notes |
|---|---|---|
| rigid repo | 0.18552 | |
| flex baseline | 0.2448 | J_bm_rms ≈ 0.044–0.063 per scenario |
| stiff_heavy (s 2, r 1) | 0.40444 | +99 lb wing, J_mass 0.165; margins 1.712 / 2.265 |
| soft_light (s 0.6, r 1) | 2000 | flutter screen fail, margin 0.968 (−39.6 lb) |
| soft_torsion_corner (s 0.8, r 0.8, ζ 0.005) | 2000 | flutter screen fail, margin 0.967 (−27.7 lb) |

Cost per scenario is about 3.5 s coupled vs 0.6 s rigid (90 s scenario, about 25× real time).

## 7. Demo results (flex_demo_summary.json; maneuver = elevator pull 1–2.5 s, aileron doublet 4–6 s, 1-cos gust 8–9.5 s)

| aircraft | p_max rigid → flex → soft ×0.4 (°/s) | root BM peak (lbf·ft), rigid / flex | M_1g (lbf·ft) | tip w max flex / soft (ft) | margins flutter / div (baseline → soft) | deterministic |
|---|---|---|---|---|---|---|
| c172x 100 KCAS 4 kft | 30.8 → 27.3 → 22.7 | 19 300 / 18 810 | 7 169 | 0.70 / 1.98 | 1.24/1.67 → 0.79/1.09 | yes |
| 737 250 KCAS 10 kft | 12.9 → 9.8 → 5.5 | 1.007e6 / 1.016e6 | 720 209 | 2.45 / 6.0 | 1.23/3.0* → 0.83/3.0* | yes |
| T38 300 KCAS 10 kft | 44.0 → 40.6 → 36.2 | 34 448 / 35 053 | 20 730 | 0.35 / 0.86 | 1.26/3.0* → 0.84/3.0* | yes |

3.0* = capped, not found below 3·V_D (raw divergence 23.5 / 14.9 V_D on the 737, 25.7 / 16.3 V_D on the T38).

Rigid vs flex behaves as expected:
- Aileron effectiveness drops with flexibility, strongly on the swept 737 because of bending–torsion wash-out.
- Nz in the pull-up rises slightly on the straight c172x wing (wash-in) and falls on swept wings.
- Tip deflection scales about 1/EI.

Coupled RTF is about 20 in the demo (4 substeps, plot-history recording). Rigid JSBSim runs at about 500.

## 8. Limitations (honest list)

- **Structural data is notional.** Frequencies, masses, EA and CG are plausible guesses, not GVT or manufacturer data. Absolute margins are only meaningful relative to each other.
- The c172 is strut-braced but is modelled as a cantilever. Its real bending stiffness and root moment distribution differ.
- Strip theory, quasi-steady, Prandtl–Glauert:
  - no tip loss beyond Schrenk;
  - no unsteady wake (Theodorsen C(k) = 1);
  - no transonic effects. T38 V_D screening is capped at M0.9.
- Elastic increments use DATCOM CLα, while the 1-g load uses the FDM's lift. The c172x FDM's own CLα is about 2× DATCOM (INTERFACE gotcha 7).
- Feedback covers lift, roll and pitch only: no elastic drag or yaw, and no control-surface hinge or servo flexibility.
- Total aero lift is assigned to the wing (`wing_lift_share` 1). Tail load is ignored.
- Aileron steps are discontinuous in the FDM, so the root moment shows a short high-frequency transient at each step (visible in the PNGs). It is physical for an instantaneous step, but it inflates peak BM. Consider rate-limiting the input or low-pass filtering the peak metric.
- AR and sweep genes are not modelled. The geometry rebuild exists, but calibration holds frequency instead of EI (INTERFACE §3).
- Speed: about 25× real time coupled in the GA path, about 6× slower than rigid.

## 9. Next steps

1. Decide the final gene set and ranges with the Genome Architect (INTERFACE §1a/§3), and add a mass term to their structural objective. EI/GJ tie decided (stiffness_scale × torsion_bend_ratio); open: whether to raise the r lower bound to 0.85.
2. Phase 2 only: re-enable elastic axis, section CG and tip mass as genes, with the hard constraint CG ≥ EA + gap and an explicit balance-mass cost added to the wing mass.
3. Add C(k)/Wagner lag states (2 per wing) if gust-load accuracy matters. Calibrate against a published typical-section flutter case.
4. Make the structural sim optionally run every 2nd frame (with substeps) for about 2× speed. Precompute per-genome matrices once per individual (already done for margins).
5. Real data: replace notional frequencies with public GVT values where available (e.g. NASA/FAA reports for light aircraft).
6. If AR/sweep genes are wanted: hold EI/GJ fixed under geometry change, and add the CLα/CDi/cosΛ increments through the existing external_reactions channel.
