# Flight Dynamics ↔ Genome interface (flex wing + Phase-1 aircraft)

Owner: Flight Dynamics. Consumers: Genome Architect (`flight_sim_3d/genome/`, which I did **not** edit).
Code: `flexwing.py` (model), `coupled_sim.py` (JSBSim coupling), `phase1_check.py` (aircraft facts, results in `phase1_check.json`),
`f16_check.py` (F-16, results in `f16_check.json`, §5), `socket_regression.py` (socket clean-up regression, `socket_cleanup_check.json`),
`gene_range_margins.json` (range review numbers). Every number below was measured with JSBSim 1.3.1 on this box unless it is
marked **public/approx**.

---------------------------------------------------------------------------------------------------------------------
## 0. Phase-1 decision (Flight Dynamics, 2026-10-06): chordwise axes are fixed, not genes

**What happened.** The Genome Architect's flex GA evolved `section_cg_frac = 0.31` with `elastic_axis_frac = 0.48`, i.e.
the section CG 0.17 chord **ahead** of the elastic axis. `flexwing.margins()` then returned `flutter_margin = inf`.

**Why it returned inf** (investigated; no typical-section closed-form formula is involved, both estimates are eigenvalue-based):
- With the CG ahead of the EA the inertial bending–torsion coupling term (−m·x_θ in M) changes sign. On the straight c172x wing the bending and torsion branches then never merge: eig(M⁻¹(K − qA)) stay real up to divergence. So the steady-coalescence search finds nothing.
- The quasi-steady p-method finds no oscillatory root with positive real part up to 10 V_D either. The torsion branch just goes statically unstable at divergence (1.28 V_D for c172x at 0.48/0.31), which is a real root, not flutter.
- Both searches returned the `math.inf` "not found" sentinel, and `min(inf, inf) = inf`.
- The swept 737/T38 stayed finite (1.36 / 1.46): sweep supplies the coupling.

Physically this is a mass-balanced section, and that does suppress flutter. With **notional** chord fractions, though, it is the GA exploiting the model.

**Decision.**
1. `x_ea` and `x_cg` are **fixed per aircraft** in Phase 1 and are no longer genes. The tip mass-balance weight (`tip_mass_frac`) is also fixed at 0 in Phase 1: it is the same chordwise-mass lever. Measured: c172x at EI = GJ ×0.6 with an 8 % tip mass at 10 % chord gives "no flutter below the cap" while the wing is 25 lb **lighter** than baseline. **Confirmed by Flight Dynamics (04:02 PT): tip mass stays fixed at 0 for Phase 1.**
2. Enforcement in `flexwing.py`: **raise, never clamp silently.**
   - `WingParams.__post_init__` and `FlexWing.__init__` (which catches params mutated after construction) raise `ValueError` unless `x_cg ≥ x_ea + MIN_CG_AFT_OF_EA` (0.02 chord).
   - `overrides_from_genome()` raises `ValueError` for any of `elastic_axis_frac`, `section_cg_frac`, `x_ea`, `x_cg`, `cal_x_*`, `tip_mass_frac`, `x_tip_mass`, whatever their value.
   - The only bypass is the Python context manager `flexwing.unchecked_section_axes()`. It is meant for analytic validation and tests and cannot be reached through params/genome dicts.
3. Margins are **always finite** (§2): capped at `MARGIN_CAP = 3.0 × V_D`, with `*_not_found_below_cap` flags.
4. **Bending and torsion stiffness are tied** (decision 04:02 PT): `stiffness_scale` × `torsion_bend_ratio` replaces the independent scales (§1a, §3).

**Fixed per-aircraft values** (`flexwing.AIRCRAFT_PROFILES`, chord fractions from the LE):

| aircraft | x_ea (elastic axis) | x_cg (section CG) | CG aft of EA | rationale |
|---|---|---|---|---|
| c172x | 0.38 | 0.42 | 0.04 c | Unswept two-spar light-alloy box (front spar ~0.2c, rear spar ~0.6c), so EA ~0.35–0.40c. Fuel in the wing box, plus the aileron and flap at the TE, put the section CG ~0.40–0.45c. |
| T38 | 0.40 | 0.42 | 0.02 c | Thin (~5 %) multi-spar wing with a box spanning most of the chord, so EA ~0.40c. Little fuel in the wing; TE surfaces put the CG slightly aft of the EA. |
| f16 | 0.40 | 0.43 | 0.03 c | Thin multi-spar wing (many closely spaced spars, box over most of the chord), so EA ~0.40c. Integral fuel sits in the box (CG ~ box centre); LE flaps forward and flaperons aft roughly balance, net CG slightly aft of the EA. See §5e for the sensitivity. |
| 737 | 0.36 | 0.38 | 0.02 c | Box between the front spar (~0.15–0.20c) and rear spar (~0.60–0.65c), so EA ~0.35–0.40c. Fuel in the box gives CG ~0.38–0.42c. The underwing engines, ahead of the EA and strongly stabilising in reality, are **not** modelled, so this is conservative. |

All four keep CG aft of EA, which is the usual (flutter-relevant) case for wings without mass balance. All values are
**notional**: typical ranges, not manufacturer data.

**Possible Phase-2 option.** Re-enable `elastic_axis_frac`/`section_cg_frac` (and the tip mass) as genes **only** with both of:
- a hard constraint `x_cg ≥ x_ea + gap` (gap ≥ 0.02c, already enforced in code);
- an explicit balance-mass cost: the mass needed to move the section CG forward from its default, Δm_bal ≈ m_section · (x_cg,default − x_cg) / (x_cg,default − x_bal), with balance mass at x_bal ≈ 0.05–0.10c. It is added to the wing mass, so it flows into the JSBSim weight and the mass penalty.

---------------------------------------------------------------------------------------------------------------------
## 1. Flex-model inputs

### 1a. Final Phase-1 gene list for the `structure` block

**Mapping** (`flexwing.tied_stiffness` / `genes_to_overrides`; the internal `ei_scale`/`gj_scale` are kept):

```
ei_scale = stiffness_scale                       # EI(y) multiplier, also the root-bending limit multiplier
gj_scale = stiffness_scale * torsion_bend_ratio  # GJ(y) multiplier, range 0.48 .. 2.30
```

`flexwing.overrides_from_genome(decoded)` maps genome names to model fields. It **raises `ValueError`** for:
- any value outside the gene ranges below, or NaN;
- the independent stiffness keys;
- fixed Phase-1 genes;
- raw `WingParams` field names;
- two aliases with conflicting values.

Missing tied genes default to 1.0. Not-modelled genes are ignored when 0 and raise when non-zero only with `strict=True`.
`flexwing.STRUCT_SCHEMA` has exactly these four genes: `stiffness_scale`, `torsion_bend_ratio`, `zeta`, `nonstruct_scale`.

| genome name (final) | flexwing field | unit | default | range | scale | status |
|---|---|---|---|---|---|---|
| `stiffness_scale` | → `ei_scale` = s | × baseline EI(y) (and strength, structural mass) | 1.0 | **0.6 – 2.0** | log | gene (replaces `bend_stiffness_scale`) |
| `torsion_bend_ratio` | → `gj_scale` = s × r | GJ multiplier ÷ EI multiplier | 1.0 | **0.8 – 1.15** | linear | gene (replaces `torsion_stiffness_scale`) |
| `bend_stiffness_scale`, `torsion_stiffness_scale` (also raw `ei_scale`, `gj_scale`) | – | – | – | – | – | **removed: passing them raises** |
| `struct_damping_ratio` | `zeta` | – (modal ζ, all modes) | 0.02 | **0.005 – 0.05** | log | gene (lower the upper bound from 0.08) |
| `nonstructural_mass_scale` (rename of `wing_mass_scale`) | `nonstruct_scale` | × non-structural share of wing mass | 1.0 | 0.8 – 1.25 | log | gene (`wing_mass_scale` still accepted as an alias) |
| `elastic_axis_frac` | `x_ea` | chord fraction from LE | c172x 0.38, T38 0.40, 737 0.36 | – | – | **fixed (Phase 1)**; passing it raises |
| `section_cg_frac` | `x_cg` | chord fraction from LE | c172x 0.42, T38 0.42, 737 0.38 | – | – | **fixed (Phase 1)**; passing it raises |
| `tip_mass_frac` | `tip_mass_frac` | tip mass ÷ semi-wing mass | 0 | – | – | **fixed at 0 (Phase 1)**; passing it raises |
| `mass_centroid_shift` | – | – | – | – | – | **drop** |
| `aspect_ratio_delta` | – | – | – | – | – | **not modelled** (fixed 0; see §3) |
| `sweep_delta_deg` | – | – | – | – | – | **not modelled** (fixed 0; see §3) |

Notes:
- The fixed, non-gene inputs come from `flexwing.AIRCRAFT_PROFILES` and the JSBSim metrics. They are listed in `WingParams`:
  - span, area, taper, sweep;
  - `wing_mass_lb` (both wings): c172x 180, T38 800, 737 10 500 lb;
  - baseline uncoupled 1st bending and torsion frequencies `f_b1_hz`/`f_t1_hz`: c172x 7/22, T38 10/35, 737 2.7/11 Hz;
  - `struct_frac` 0.55 and `w_ei` 0.5;
  - `v_dive_keas`: c172x 180, T38 595 (capped at M0.9 SL), 737 400;
  - `n_limit`: c172x 3.8, T38 7.33, 737 2.5.
- **All structural numbers are notional.** No GVT or manufacturer data was used. They are calibrated only to plausible frequencies and wing masses.

### 1b. Stiffness ↔ structural mass link (exact formula, `FlexWing.mass_distribution`)

```
m_semi = (wing_mass_lb / 2) * [ (1 - struct_frac) * nonstruct_scale
                               + struct_frac * ( w_ei * ei_scale + (1 - w_ei) * gj_scale ) ]
         + tip_mass_frac * (wing_mass_lb / 2)            # tip_mass_frac fixed at 0 in Phase 1
struct_frac = 0.55, w_ei = 0.5
```

So structure mass scales linearly with the stiffness multipliers. This is a skin/box thickness model: EI ∝ t, GJ ∝ t, and
mass ∝ t. The spanwise mass distribution follows chord (m ∝ c). EI and GJ follow (c/c_root)³, and their absolute level is
calibrated so that the baseline uncoupled frequencies equal `f_b1_hz`/`f_t1_hz`.

Example: stiffness_scale 2, ratio 1 (ei = gj = 2) → m_semi ×1.55, i.e. c172x +99 lb for the pair of wings. Stiffness 0.6, ratio 1 → ×0.78 (−40 lb). In tied form the structural part is struct_frac · s · (w_ei + (1 − w_ei)·r).

### 1c. How mass is fed back to JSBSim

`flexwing.prepare_aircraft()` copies the aircraft into `jsbsim_root/` (the venv data is untouched). It injects two point
masses, `flexwing_dm_R` and `flexwing_dm_L`, into `<mass_balance>`; their indices are in `flexwing_meta.json`.

`apply_wing_mass(fdm, ...)` must be called **before `run_ic`/trim**. It sets:
- `inertia/pointmass-weight-lbs[i]` = m_semi − m_semi,baseline (this can be negative);
- `inertia/pointmass-location-Y-inches[i]` = ±(spanwise mass centroid) × 12. X and Z stay at AERORP.

JSBSim then updates the total weight, the CG and Ixx/Izz itself. The test `test_mass_feedback` checks this: weight changes
by exactly 2Δm, and Ixx increases.

Genome side:
- My point masses are appended after the stock ones: c172x indices 6/7.
- The 737 and T38 have **no** stock point masses. For them I first insert a zero-weight `payload_placeholder` at the CG as index 0, and the wing masses go to 1/2. Your `sim_ext` `payload_index = 0` therefore lands on the centreline on all three aircraft.
- Against the stock 737/T38 files, index 0 does not exist at all, so payload scenarios on those aircraft need the prepared copies (or their own placeholder).
- **f16** has one stock point mass (`Pilot`, 230 lb). The placeholder is inserted **before** it, so on the prepared f16: index 0 = `payload_placeholder` (0 lb, at the CG, centreline), 1 = `Pilot`, 2/3 = wing masses. Anything that addressed the pilot as index 0 on the stock f16 must use index 1 on the prepared copy (`flexwing_meta.json` has `"stock_pointmass_index_shift": 1`).
- A zero-mass prepared copy trims and flies bit-identically to stock (checked on the 737: identical α, throttle, altitude and CG after 10 s; f16: bit-identical 6 s trajectory incl. CG, `f16_check.json`).
- Trim and the GA fitness therefore see the heavier or lighter aircraft automatically (more throttle, higher α).
- `coupled_sim.make_coupler(...)` sets `coupler.delta_mass_lb`.

---------------------------------------------------------------------------------------------------------------------
## 2. Outputs (`flexwing.telemetry_channels(hist, wing, m_root_1g)`)

Telemetry arrays, one sample per JSBSim frame (dt 1/120 s). Names match `genome/fitness.py::obj_structural`:

| channel | unit | definition |
|---|---|---|
| `wing_root_bending` | lbf·ft | Root bending moment of the **more-loaded** semi-wing at each sample, signed (+ = tip-up bending). Includes the 1-g load. Computed from modal coordinates (`bm_*` rows: aero strip loads + inertia relief at the beam root, y = root_frac·semispan). |
| `wing_root_bending_R`, `wing_root_bending_L` | lbf·ft | Per side |
| `tip_deflection` | ft | Elastic tip deflection (+ up) of the side with the larger magnitude, relative to the jig shape. The 1-g trim shape is included. |
| `tip_twist` | deg | Elastic tip twist (+ nose-up, streamwise) of the side with the larger magnitude |

Per-run scalars (`["params"]`) — put them into `p` for `obj_structural`:

| param | unit | formula |
|---|---|---|
| `wing_root_bending_limit` | lbf·ft | `n_limit * M_root_1g * stiffness_scale` (= ei_scale). M_root_1g is the trim root moment of this genome. Limit (not ultimate) load; ultimate = 1.5 × limit. Strength scales with the same skin-thickness argument as EI. |
| `wing_root_bending_1g` | lbf·ft | M_root_1g (trim). Baseline values: c172x 7 169, T38 20 730, 737 720 209 |
| `tip_deflection_limit` | ft | `tip_defl_limit_frac * semispan`: c172x 8 %, T38 6 %, 737 12 % |
| `tip_twist_limit` | deg | 3.0 |
| `wing_mass_lb`, `wing_mass_delta_lb`, `wing_mass_frac_delta` | lb, lb, – | Total wing mass and its change vs baseline. **You need a mass term**: otherwise "stiffer = stronger" wins for free, because the limit scales with stiffness. |

Peak and RMS: `obj_structural` already computes `max(|M|/limit) + fatigue_weight * std(|M|/limit)`. My own GA terms
(`flexwing.response_terms`) are:
- J_bm_rms = 0.25 · RMS(M − M1g)/M1g;
- J_bm_peak = 2 · max(0, peak/limit − 1)²;
- hinge² terms on tip deflection and twist against the limits above;
- hard fail if peak > 1.5 · limit.

Pre-simulation screening (`FlexWing.margins()` / `margin_terms`, ~6–20 ms per genome). **Every margin is finite**, in
[0, `MARGIN_CAP` = 3.0], in units of V_D:

| field | type | meaning |
|---|---|---|
| `flutter_margin` | float | min(QS p-method, steady coalescence) / V_D, capped at 3.0. **Use this as the constraint.** |
| `flutter_not_found_below_cap` | bool | true if neither method finds flutter up to 3.0 V_D; `flutter_margin` is then exactly 3.0 |
| `flutter_margin_qs`, `coalescence_margin` | float | the two estimates, each capped at 3.0 |
| `v_flutter_keas`, `v_coalescence_keas` | float | capped speeds (= margin × V_D) |
| `f_flutter_hz` | float | QS flutter frequency, or 0.0 if none below the cap (was NaN) |
| `div_margin` | float | V_div / V_D, capped at 3.0. The exact eigenproblem is used, so there is no search range |
| `div_not_found_below_cap` | bool | true if divergence is absent or above 3.0 V_D. Always true on the swept 737/T38 (raw 23.5 / 25.7, previously reported as such) |
| `margin_error` | bool | numerical failure (NaN / LinAlgError). The affected margin is reported as **0.0**, a conservative hard fail. Not seen in any run |
| `margin_cap` | float | 3.0 |

- **Search coverage:** the QS sweep runs over EAS 1 kt … 3.0·V_D (120 points, endpoint included, bisection refinement). The coalescence sweep runs over q 0 … 9·q_D, i.e. (3.0·V_D)² (400 points, plus bisection).
- **Penalty:** w·max(0, (1.2 − m)/0.2)², with a **hard fail if either margin is below 1.0** (skip the sim, cost = fail_cost).
  - The cap does not change any penalty, because penalties act only below 1.2. It just keeps outputs bounded and comparable (no inf in a Pareto front).
- Expose `flutter_margin` and `div_margin` as constraints. This mirrors the CS/FAR 25.629 style rule: 1.2 · V_D required.

---------------------------------------------------------------------------------------------------------------------
## 3. Review of your provisional ranges

Margins come from `FlexWing.margins()` with the **fixed Phase-1 axes** (§0), zeta 0.02, no tip mass. Full grid:
`gene_range_margins.json` (regenerated 2026-10-06 ~04:10 PT; 12 stiffness × 8 ratio points per aircraft, value =
min(flutter, divergence)).

**Tied space** (`ei = s`, `gj = s·r`), min(flutter, divergence) margin:

| aircraft | min over whole grid (s 0.6–2.0, r 0.8–1.15) | min for s ≥ 0.8 | min for s ≥ 0.8 except the corner s < 0.86, r < 0.85 | baseline s = r = 1 | s = 2, r = 1.15 |
|---|---|---|---|---|---|
| c172x | 0.841 (s 0.6, r 0.8) | **0.967** (s 0.8, r 0.8) | 1.003 (s 0.86, r 0.8) | 1.239 | 1.85 |
| T38 | 0.895 (0.6, 0.8) | **0.999** (0.8, 0.8) | 1.031 (0.8, 0.85) | 1.257 | 1.91 |
| 737 | 0.900 (0.6, 0.8) | 1.020 (0.8, 0.8) | 1.049 (0.8, 0.85) | 1.231 | 1.71 |

What this shows:
- **No ridge.** On every aircraft the margin increases monotonically with both s and r over the whole tied grid (asserted in `test_tied_stiffness_grid_finite_monotone_no_ridge`). All values are finite. Swept-wing divergence is always 3.0* (not found below the cap).
- **Soft-torsion corner.** The requested "margin ≥ 1.0 for stiffness_scale ≥ 0.8" holds everywhere **except** at s ≈ 0.80–0.85 with r = 0.8: c172x 0.967, T38 0.999.
  - This is the smooth low-torsion corner (GJ × = 0.64), not a ridge. Those genomes hard-fail on the flutter screen, which is the intended behaviour at the range boundary.
  - The test asserts ≥ 1.0 outside that corner and ≥ 0.96 inside it.
  - If you want ≥ 1.0 for all s ≥ 0.8, raise the ratio's lower bound to **0.85**: the minimum at s = 0.8 is then 1.006 / 1.031 / 1.049.
- **Lower bound s = 0.6:** the hard-fail line (margin 1.0) lies inside the range, as intended. Feasible from s ≈ 0.6 at r 1.05–1.15 up to s ≈ 0.76 (737), 0.80 (T38) and 0.86 (c172x) at r = 0.8. Margin 1.2 is reached at about s 0.95 for r = 1.
- **Why tie, and why ratio ≤ 1.15** (ridge rationale): with independent EI/GJ genes the swept 737/T38 show a narrow infeasible ridge where the 1st torsion frequency crosses the 2nd bending frequency.
  - The 737 baseline already has t1 = 10.95 Hz next to b2 = 11.9 Hz. With GJ/EI ≈ 1.2–1.6 the margin drops off a cliff: 737 EI 1 / GJ 1.25 → 0.52, T38 EI 0.6 / GJ 1 → 0.65 (independent-gene grid, previous revision).
  - Classical flutter does have such a minimum, but here it depends entirely on the notional frequency placement. Keeping GJ/EI in 0.8–1.15 keeps the GA on the smooth side for all three aircraft. The 737 cliff starts between r = 1.15 and 1.20.
- Divergence never binds on the swept T38/737: aft sweep gives bending–torsion wash-out, so V_div > 3 V_D. It does bind on the straight-wing c172x (1.36 at s = 0.8, r = 0.8; 1.32 at s = 0.6, r = 1).
- **Damping 0.005–0.08 → 0.005–0.05.**
  - ζ > 5 % is unrealistic for metal or composite wing structure (typically 1–3 %).
  - The conservative flutter margin (coalescence) does not depend on ζ by construction. Only the QS margin and the gust/aileron response do.
  - A high ζ would mainly let the GA "buy" smooth loads for free.
- **wing_mass_scale 0.8–1.25 is OK as a range**, but rename it `nonstructural_mass_scale`. Structural mass already follows the stiffness genes (§1b), so a separate structural-mass gene would double-count.
- **mass_centroid_shift**: drop. Chordwise mass placement is what moves flutter, and it is fixed in Phase 1 (§0). A pure spanwise shift is under-determined.
- **AR ±15 % and sweep ±5°: cannot be applied to the fixed JSBSim aero tables.**
  - The tables are functions of α, Mach and control positions for a fixed geometry. Changing `metrics/bw-ft` would only rescale the moment reference.
  - If you want them anyway, apply them as increments through the same external_reactions channel the flex model already uses. Hold **wing area constant** (span ∝ √AR). For each frame:
    - lift-curve slope ratio from DATCOM/Helmbold, CLα = 2πA / (2 + √(A²β²/η² · (1 + tan²Λ/β²) + 4)) (`flexwing.datcom_cla`). ΔL = (r − 1) · CL_wing(α) · q · S, where CL_wing comes from JSBSim lift. Measured ratios r:
      - c172x: AR ±15 % → 0.957 / 1.033;
      - T38: 0.922 / 1.064;
      - 737: 0.959 / 1.032;
      - sweep ±5° → T38 1.024 / 0.970, 737 1.034 / 0.959;
      - sweep on c172x: ±0.3 % in the sweep direction only. Negative sweep makes it **forward-swept**: divergence margin drops 1.67 → 1.51.
    - induced drag: ΔCDi = CL²/(π e) · (1/A − 1/A0) with e ≈ 0.8. At fixed CL this is +17.6 % for −15 % AR and −13 % for +15 %;
    - sweep: Mach-normal increments ∝ cosΛ (drag-divergence Mach shifts by ≈ cosΛ0/cosΛ), CLα as above, and the aerodynamic-centre shift Δx_ac ≈ (span/4)·ΔtanΛ as a pitching moment;
    - roll damping Clp ∝ CLα·AR-ish → scale it by r · (A/A0).
  - On the structure side, `WingParams.span_ft`/`sweep_deg` already rebuild the beam, but my calibration re-targets frequency. For AR genes it must hold EI fixed instead: a longer span lowers frequencies by about (L0/L)², and adds mass and root moment. This has not been done, so the AR margins I computed (+15 % AR: flutter 1.24 → 1.18 on c172x, 1.26 → 1.19 on T38) are **optimistic**.
  - **Recommendation:** leave `aspect_ratio_delta` and `sweep_delta_deg` disabled (fixed at 0) in Phase 2. If enabled later, use AR ±10 % and sweep ±3° at constant area. Beyond that the increment approach is not credible on fixed tables.

---------------------------------------------------------------------------------------------------------------------
## 4. Phase-1 aircraft: verified trim recipes, limits, controls, gotchas

### 4a. Trim recipe that works (all three aircraft, JSBSim 1.3.1, Python API)

```python
fdm = jsbsim.FGFDMExec(root); fdm.load_model(model); fdm.set_dt(1/120)
fdm["gear/gear-cmd-norm"] = 0.0          # BEFORE run_ic. Every stock model loads with gear DOWN (cmd = pos = 1)
fdm["ic/h-sl-ft"] = h; fdm["ic/vc-kts"] = kcas; fdm["ic/gamma-deg"] = 0; fdm["ic/psi-true-deg"] = 0
fdm.run_ic()
fdm["propulsion/set-running"] = -1       # all engines running (no starter / magneto sequence needed)
for i in range(n_engines): fdm[f"fcs/mixture-cmd-norm[{i}]"] = 1.0
fdm["simulation/do_simple_trim"] = 1     # 1 = FULL trim (tFull). 0 = longitudinal only. sim.py's comment is wrong
```

Verified results (gear up; holding the trim open-loop for 10 s drifts less than 0.03°, 0.04 kt and 1 ft):

| | c172x | T38 | 737 |
|---|---|---|---|
| design point | 100 KCAS / 4 000 ft (M0.163) | 300 KCAS / 10 000 ft (M0.541) | 250 KCAS / 10 000 ft (M0.452) |
| weight as loaded | 2 480 lb | 11 474 lb | 107 000 lb |
| α = θ (deg) | 0.79 | 4.61 | 3.28 |
| throttle-cmd-norm | 0.780 (2 394 rpm) | 0.354 (**pos 0.708**, N2 86.3 %) | 0.586 (N2 83.4 %) |
| pitch-trim-cmd-norm (elevator-cmd stays 0) | +0.219 | −0.115 | −0.233 |
| gear down instead of up | no change (fixed gear) | thr 0.435, N2 93.9 % | thr 0.690, N2 87.6 % |
| mode 0 instead of 1 | **trim "succeeds" but the aircraft departs: θ −11°, +8.6 kt, −106 ft in 10 s** (lateral/torque not trimmed) | identical | identical |
| flaps | 0 (flap-cmd-norm 0 at load) | 0 | 0 |
| level-trimmable KCAS at the design altitude (gear up, throttle ≤ 1) | 45 – 115 | 150 – 540 (MIL-only, cmd ≤ 0.5: 150 – 490, M0.87) | 190 – 420 (M0.75) |
| level trim found at altitude (model) | up to ~25 000 ft (55 KCAS); fails at 30 000 | MIL: up to ~40 000 ft; with AB ≥ 50 000 ft | up to 45 000 ft (200 KCAS, thr 0.93); fails at 50 000 |
| model CL_max (IC α sweep, gear up) | the table never stalls in a static IC sweep (CL 2.7 at 18°, still rising; see gotchas) | 0.94 at α 19° → Vs1g ≈ 146 KEAS at 11 474 lb | 1.14 at α 13–14° → Vs1g ≈ 151 KEAS at 107 000 lb |
| determinism / RTF (6 concurrent workers) | yes / 510 pure, 184 with control loop | yes / 1 200 / 771 | yes / 571 / 324 |

### 4b. Operating limits (public/approx, consistent with `genome/aircraft_profiles`) vs what the model does

| | c172x | T38 | 737 |
|---|---|---|---|
| Vs (clean) | ~48 KCAS (public) — model trims down to 45 KCAS | ~140 KCAS (public, rough) — model Vs1g ≈ 146 KEAS, trims ≥ 150 KCAS at 10 kft | ~140 KCAS (public, rough) — **model built-in trim fails below 190 KCAS at 10 kft** (see gotchas). Physical CLmax gives ≈ 151 KEAS |
| Vne / Vmo / Mmo | Vne 163 KIAS, Vno 129 | ~710 KIAS / M1.3 (public) — model level max ~540 KCAS (M0.96) with AB at 10 kft | Vmo ~340 KIAS / Mmo 0.82 |
| n limits | +3.8 / −1.52 | +7.33 / −3.0 | +2.5 / −1.0 |
| max α (model) | alphalimits 16° (0.28 rad); stall hysteresis 5.2°/20.6° (0.09/0.36 rad) | CL peaks at 19–20°, then drops linearly | CL peaks at 14°, drops to 0.2 by 26° |
| service ceiling | ~13 000–14 000 ft (public) — model trims at 25 000 ft (engine has no realistic altitude power loss) | ~50 000 ft (public) — model MIL ~40 000 ft | ~37 000–41 000 ft (public) — model trims at 45 000 ft |
| flex V_D used for margins | 180 KEAS | 595 KEAS (M0.9 cap) | 400 KEAS |

### 4c. Control properties, travel and signs (measured)

| | c172x | T38 | 737 |
|---|---|---|---|
| pitch | `fcs/elevator-cmd-norm` + `fcs/pitch-trim-cmd-norm` → `fcs/elevator-pos-deg` ±19.5° (cmd + trim, clipped) | → `fcs/elevator-pos-norm` only (range **−1 … +0.583**), no deg/rad output | → `fcs/elevator-pos-deg`, sum clipped to ±17.2° (−17.2/+13.2° at cmd ±1 with trim −0.233) |
| roll | `fcs/aileron-cmd-norm` → `fcs/left/right-aileron-pos-deg`, differential −19.9/+14.9° | → `fcs/left/right-aileron-pos-norm` only, differential (+cmd: L +0.75, R −1.0) | → `left/right-aileron-pos-deg` ±20.1° (also has spoiler properties) |
| yaw | `fcs/rudder-cmd-norm` → `fcs/rudder-pos-deg` ±16° | → `fcs/rudder-pos-norm` ±1 (has a yaw SAS in the FCS) | → `fcs/rudder-pos-deg`, ~±14.8° reached after 1 s (has a yaw damper) |
| power | `fcs/throttle-cmd-norm`, `fcs/mixture-cmd-norm`, `fcs/advance-cmd-norm`, `propulsion/magneto_cmd`, `propulsion/starter_cmd` | `fcs/throttle-cmd-norm[0,1]` with **pos = 2 × cmd: cmd 0.5 = MIL, 0.5–1.0 = afterburner** (J85 `augmethod 2`); `propulsion/cutoff_cmd`, `propulsion/starter_cmd`, `propulsion/engine[i]/n2` | `fcs/throttle-cmd-norm[0,1]` (0–1 = idle–max), `cutoff_cmd`, `starter_cmd`, `engine[i]/n2` |
| gear / flaps | fixed gear; `fcs/flap-cmd-norm` → `fcs/flap-pos-deg` | `gear/gear-cmd-norm`, `fcs/flap-cmd-norm`, `fcs/speedbrake-*` | `gear/gear-cmd-norm`, `fcs/flap-cmd-norm` |

Signs (+0.1 step from trim, response after 0.5 s; identical on all three):
- **+elevator-cmd = nose DOWN.** q: c172x −2.5, T38 −3.3, 737 −1.2 °/s.
- **+aileron-cmd = roll RIGHT.** p: +6.9 / +10.8 / +2.3 °/s. Left aileron goes TE-down.
- **+rudder-cmd = nose LEFT.** r: −1.6 / −0.6 / −1.5 °/s.
- +throttle = more thrust.

Open-loop pitch-rate gain from the FM check: −26.3 / −39.7 / −15.8 °/s per unit elevator.

### 4d. Gotchas

1. **Gear defaults to DOWN** (`gear/gear-cmd-norm` = `gear-pos-norm` = 1) in every model, and `genome/sim_ext.py` does not retract it.
   - The T38 and 737 have been flown gear-down so far: about 23 % more trim throttle on the T38 and 18 % on the 737.
   - Fix: set `gear/gear-cmd-norm = 0` before `run_ic`. The built-in trimmer moves the gear actuator instantly.
   - **Outside the trimmer, `run_ic()` advances FCS actuators by one dt per call** (the gear creeps 0.0014 per call). A Newton trim built on repeated run_ic must first retract the gear by stepping; `fmq_worker.retract_gear` does this.
   - With gear up, F80C, OV10, T37 and t6texan2 move from caveat to good-to-evolve.
2. **Trim mode:** use `do_simple_trim = 1`. Mode 0 "succeeds" on the c172x but leaves the propeller-torque roll untrimmed.
3. **The trimmer carries pitch in `fcs/pitch-trim-cmd-norm`**; `elevator-cmd-norm` is 0 after trim. Controllers must add to elevator-cmd and leave pitch trim alone. sim_ext already does this.
4. **737 low-speed trim fails** below ~190 KCAS at 10 000 ft ("wdot doesn't appear to be trimmable"), although lift and moment balance exist below it (checked at 175 KCAS: wdot ≈ −0.4 ft/s² at α 9° and Cm = 0 near elevator −0.6; CLmax 1.14 implies about 151 KEAS).
   - Cause: the trimmer brackets α on its default range, and the post-stall CL drop makes wdot the same sign at both ends. At 175 KCAS, wdot is +38.7 at −5° and +4.7 at 20°, with a root near 9°.
   - Your profile's `sim_envelope.min_kcas` = 165 for the 737 is **below what the built-in trim can reach**. Use ≥ 195 KCAS at 10 kft, or write a powered Newton trim. `fmq_worker.glide_trim` is an engine-off Newton variant that could be extended.
5. **T38 throttle**: `throttle-pos-norm = 2 × throttle-cmd-norm`. sim_ext clamps cmd to [0, 1], so the GA can use the full afterburner. Clamp it to 0.5 if Phase 1 should be MIL-only. Trim at the design point uses cmd 0.354 (71 % of MIL).
6. **T38 writes only `-pos-norm`** for elevator, aileron and rudder; deg properties read 0. The flex model reads `*-aileron-pos-norm` × 20°, an assumed scaling.
7. **c172x lift is doubled.** `aero/coefficient/CLwbh` (table) **plus** a separate `CLalpha` term make total CLα ≈ 9.8/rad (a real 172 is about 4.6). The extra term is linear and never stalls, so the IC α sweep never shows a CL max.
   - Effects: trim α is tiny (0.8° at 100 KCAS), and stall behaviour depends on the in-flight hysteresis flag `aero/stall-hyst-norm`.
   - The flex model's elastic increments use DATCOM CLα (4.7/rad), not the FDM's; it uses the FDM only for the 1-g lift distribution.
8. **High-altitude power.** The c172x trims at 25 000 ft and the 737 at 45 000 ft. Both exceed the real ceilings, so do not use altitude scenarios above ~12 000 ft (c172x) or ~41 000 ft (737) as "realistic".
9. **Engine start.** `propulsion/set-running = -1` after `run_ic` gives running engines immediately. Turbine N2 then settles over about 1 s (737 thrust 13.3k → 10.9k lbf at cmd 0.9). Trim **after** set-running, as above.
10. Mixture is per engine: index all of `fcs/mixture-cmd-norm[i]`. It is irrelevant for turbines but harmless.
11. With wing mass feedback, call `apply_wing_mass` **before** trim, or the trim is for the wrong weight.
12. **Network sockets (removed from the prepared copies, 2026-10-06).** The stock 737 declares a telnet `<input port="5137"/>`
    and a QTJSBSIM UDP `<input port="5139">` that can write `fcs/aileron|elevator|rudder-cmd-norm`, `fcs/throttle-cmd-norm[0,1]`
    and `simulation/terminate`. Measured: they bind **0.0.0.0:5137/tcp and 0.0.0.0:5139/udp at `run_ic()`** (not at load).
    `prepare_aircraft()` (format 3) now strips every network `<input>`/`<output>` (live or commented: 737 also had a commented
    UDP output on 5138, c172x commented SOCKET outputs 1138/1140 and input 1137) from all XML in the copied aircraft dir; `grep`
    of `jsbsim_root/aircraft` for `port=`, `type="SOCKET|TCP|UDP|QTJSBSIM|FLIGHTGEAR"` and `protocol=` finds nothing, and no
    listener appears after `run_ic`. File outputs (`type="CSV"`) are kept; `new_fdm` calls `disable_output()`.
    The **stock** models in the venv are unchanged (by rule), so `FGFDMExec(None)` + `load_model("737")` (e.g. `phase1_check.py`,
    `genome/sim_ext.py` if it uses the stock root) still opens both ports. Use `jsbsim_root` for anything long-running.
    Regression (`socket_cleanup_check.json`): 737 trim at 250 KCAS / 10 000 ft gear up is bit-identical (α 3.2776°, throttle
    0.5857, pitch trim −0.2334); rigid and two-way flex 12 s maneuver trajectories, zero-force-vs-stock comparisons and
    repeat-determinism are bit-identical before/after on c172x, T38 and 737.

---------------------------------------------------------------------------------------------------------------------
## 5. f16 (General Dynamics F-16A, JSBSim `f16.xml`) — Evolution Runner trim point 350 KCAS / 10 000 ft

All numbers measured by `f16_check.py` → `f16_check.json` on the prepared copy `jsbsim_root/aircraft/f16`.

### 5a. Prepared copy
- Built by `prepare_aircraft("f16", root)` like the T38/737: no network sockets (the stock file has none; only a commented CSV
  output), `<external_reactions>` force `flexwing_F` (BODY, at AERORP x −189.5 in, z 3.9 in) and moment `flexwing_M` appended
  after the stock `pushback`/`hook` forces, point masses `payload_placeholder` (index 0, 0 lb, at the CG x −193 in, z −5.1 in),
  `Pilot` (1), `flexwing_dm_R/L` (2/3).
- Zero external force/mass: patched copy flies **bit-identically** to the stock f16 (6 s, elevator and aileron steps).
- Gear is retractable and works like the T38/737: loads DOWN (cmd = pos = 1); set `gear/gear-cmd-norm = 0` before `run_ic`;
  after the trim `gear-pos-norm = 0`. (After `run_ic` alone it is 0.9967 — the one-dt actuator creep, gotcha 1.)

### 5b. Trim at 350 KCAS / 10 000 ft (weight as loaded 20 630 lb: 17 400 empty + 230 pilot + 3 000 internal fuel)

Recipe = §4a (`propulsion/set-running = -1`, no mixture needed). M 0.629, q̄ 403 psf.

| | mode 1 (full), gear up | mode 0 (longitudinal), gear up | mode 1, gear down |
|---|---|---|---|
| trims? | **yes** | yes (identical to 5 digits) | yes |
| α = θ | **1.045°** | 1.045° | 1.042° |
| throttle-cmd-norm | **0.2836** → throttle-pos-norm 0.567 (**dry, 57 % of MIL**) | 0.2836 | 0.3726 (pos 0.745) |
| thrust / N2 | 3 231 lbf / 79.7 % | same | 6 479 lbf / 88.0 % |
| pitch-trim-cmd-norm (elevator-cmd stays 0) | **−0.0601** | −0.0601 | −0.0574 |
| stabilator `fcs/elevator-pos-deg` | **−1.111°** (norm −0.0445) | −1.111° | −1.028° |
| LEF / TEF / speedbrake | 0 / 0 / 0 | | |
| 20 s hands-off hold | Δθ +0.028°, Δα −0.002°, −0.10 kt, +3.9 ft | same | Δθ +0.024°, +3.8 ft |

- Mode 2 (ground) fails in the air as expected. Mode 3 (pull-up) also converges but is not needed.
- **Throttle mapping** (`fcs/throttle-pos-norm = 2 × throttle-cmd-norm`, F100-PW-229 `augmethod 2`, same as T38):
  cmd 0.5 = MIL (pos 1.0, 13 054 lbf at this point); **cmd > 0.5 = afterburner** (AB level = pos − 1; fuel flow jumps from
  2.5 to 7.6 pps at cmd 0.51); cmd 1.0 = max AB (24 048 lbf). The design-point trim (0.284) is dry, well below MIL.

### 5c. Envelope (model, mode 1, gear up, throttle-cmd ≤ 1)
- Level-trimmable at 10 000 ft: **130 – 800 KCAS** (α 16.4° at 130; fails at 120). MIL only (cmd ≤ 0.5): 130 – 610 KCAS.
  800 KCAS at 10 kft is M1.41 with AB — the model has supersonic tables but **the flex model is not valid above M0.9**.
- Below 250 KCAS the FCS drops the TEF (flaperons) to 20° automatically (`fcs/tef-pos-rad` switch), and the LEF schedules
  with α (15° above α 5°, 25° above α 15° with gear up). Low-speed trims include those surfaces.
- Level trim found up to **55 000 ft** (160 KCAS, M0.755, AB cmd 0.90); fails at 57 500 ft.
- Public/approx limits: +9 / −3 g (FCS clips stick to +1/−0.44 for that reason), VD ~800 KCAS / M2.05 (not usable here),
  ceiling ~50 000 ft.

### 5d. Controls, signs and FBW behaviour (measured, +0.1 step from trim, 0.5 s)

| channel | property → what it really is | sign / response |
|---|---|---|
| pitch | `fcs/elevator-cmd-norm` (+ `fcs/pitch-trim-cmd-norm`) → **pitch-rate/g demand** of the FCS, clipped to [−1, +0.44] → `fcs/elevator-pos-rad`/`-deg` (stabilator, ±25°, 0.3 s full-travel actuator). The aero tables use `fcs/elevator-pos-rad`. | **+cmd = nose DOWN**: q −3.0 °/s, stab +0.20° |
| roll | `fcs/aileron-cmd-norm` → **roll-rate demand** → `fcs/aileron-pos-rad` (= command × 0.375 rad, **no actuator lag**; this is what Clda/Cnda use) and the flaperon actuators `fcs/left|right-aileron-pos-rad` (Mach-scaled ×(1 − 0.85 M), unused by the aero) | **+cmd = roll RIGHT**: p +15.8 °/s |
| yaw | `fcs/rudder-cmd-norm` → yaw-rate/lateral-g loop → `fcs/rudder-pos-rad` ±30° | **+cmd = nose LEFT**: r −0.20 °/s |
| power | `fcs/throttle-cmd-norm` (single engine, no index needed) → pos = 2 × cmd | +0.1 → +3 705 lbf |
| other | `gear/gear-cmd-norm`, `fcs/speedbrake-cmd-norm`, `fcs/fbw-override` (1 = direct stick to surfaces), `fcs/hook-engage` | |

**FBW findings** (they matter for trim and for the GA controller):
1. **The pitch and roll PID integrators never integrate.** `fcs/g-load-pid` equals exactly kp × error (−0.3 × 0.01821 =
   −0.00546) at trim and is still −0.0055 after 20 s; an active ki = 0.025 would have added ~−0.009. JSBSim's PID `<trigger>`
   *holds* the integrator while the trigger property is non-zero, and the model's trigger is 1 above 5 KCAS (roll: 20 KCAS).
   So the FCS is a **proportional** pitch-rate + Nz + α feedback and roll-rate feedback system, not an integrating g-command
   system. Consequence: trim is well defined (the trimmer solves for `pitch-trim-cmd-norm`), and after a stick input the
   aircraft does **not** return to the old attitude: the pitch step below leaves +2° θ and a gentle climb.
2. **Gravity correction has the wrong sign.** `fcs/g-load-corrected` = n-pilot-z-norm − cosθ·cosφ = **−2.0** in level flight
   (n-pilot-z-norm is −1 at 1 g). With gain 0.02 it is a constant −0.04 bias, absorbed by the trim (part of the −0.06 pitch
   trim). Harmless for trim, but `elevator-cmd-norm` is a *biased* demand, not a pure Δg command.
3. **Controller gains do not transfer** from c172x/T38/737. On the f16 `elevator-cmd-norm` sets a pitch-rate/g demand with a
   static gain and the FCS already damps q; the repo's altitude-hold PID (sim.py) will need its own gains for f16. The
   +1/−0.44 stick clip and the α-limiter (`fcs/alpha-limiter-norm` = 1.047 × α, always active) also act on any GA command.
4. **Flex coupling and the FCS.** The FCS feeds back `accelerations/n-pilot-z-norm` and body rates, which include the
   `<external_reactions>` flex increments, so the FCS partly closes the loop around the elastic loads (as a real FBW does with
   its sensors). Flex-vs-rigid differences are therefore smaller in pitch than on the open-loop aircraft, and the roll-rate
   loop compensates aileron effectiveness loss with deflection: in a 0.3 roll demand the flex aircraft reaches p 48.4 °/s with
   4 % less `aileron-pos-rad` than rigid (47.4 °/s) — the elastic roll-damping reduction is larger than the aileron loss.
   Nothing unstable or odd numerically; the coupled runs are bit-for-bit repeatable.
5. Flex aileron strip loads read `fcs/aileron-pos-rad` (new `WingParams.ail_antisym_prop`, left = +x, right = −x), i.e. the
   property the FDM's roll moment uses, calibrated to Clda 0.051 /rad (α 0, table value; lumps flaperon + differential tail).
   The Mach-scaled flaperon actuator outputs are deliberately not used, since the FDM's roll power ignores them too.

**Pitch step** (trim, then `elevator-cmd-norm` −0.10 for t = 1–2 s, hands off, 8 s, dt 1/120):

| | q peak | t(q peak) | Nz peak | α peak | stab range | end (t = 8 s) | repeat run |
|---|---|---|---|---|---|---|---|
| rigid | 3.13 °/s | 1.27 s | 1.62 g | 2.58° | −4.11 … +1.53° | q −0.03 °/s, Nz 0.98, Δθ +2.0°, Δh +143 ft, −4.9 kt | **bit-identical** |
| two-way flex (baseline genes) | 3.18 °/s | 1.28 s | 1.62 g | 2.69° | −4.11 … +1.68° | q −0.03 °/s, Nz 0.98, Δθ +2.1°, Δh +147 ft | **bit-identical** |

Clean and well damped: q 2.86 °/s at t 1.5 s, one small undershoot after release (−0.77 °/s at 2.3 s), within ±0.12 °/s by
3.5 s and Nz back to 1.0 g by ~4 s; no oscillation. Peak external flex force 898 lbf, moment
2 180 lbf·ft. Demo maneuver (`run_maneuver("f16")`: pull, aileron doublet ±0.3, −30 ft/s 1-cos gust): rigid p_max 48.7 °/s,
Nz 3.45; flex baseline p_max 50.2 °/s, Nz 3.28 (gust relief), root BM peak 126 568 lbf·ft vs M_1g 36 144, tip deflection
0.72 ft, twist 0.91°.

### 5e. Flex wing: fixed axes, structure, margins

Profile (`flexwing.AIRCRAFT_PROFILES["f16"]`, all structural numbers **notional**): taper 0.23, quarter-chord sweep 32°, beam
root at 0.17 semi-span (side of body), wing mass 1 600 lb (both wings), f_b1 8.5 Hz / f_t1 28 Hz uncoupled (clean wing, no
tip launchers or stores), x_ea 0.40 / x_cg 0.43, V_D for screening 595 KEAS (M0.9 cap as on the T38; the real 800 KCAS placard
is transonic/supersonic and outside strip theory), n_limit 9, tip deflection limit 5 % semi-span, flaperons η 0.20–0.65.

Margins (×V_D, cap 3.0; coupled modes 7.9 / 28.0 / 36.6 Hz):

| stiffness_scale \ torsion_bend_ratio | 0.8 | 1.0 | 1.15 |
|---|---|---|---|
| 0.6 | **0.963** | 1.076 | 1.158 |
| 0.8 | 1.102 | 1.243 | 1.337 |
| 1.0 (default r 1.0) | 1.232 | **1.390** | 1.495 |
| 2.0 | 1.742 | 1.965 | 2.115 |

- Value shown = flutter margin (min of QS p-method and coalescence; coalescence is the lower one everywhere, flutter
  frequency 17.7–25.3 Hz, bending-torsion). **Divergence is never found below the cap** (raw 40 V_D at default: swept-back
  wash-out) → div_margin 3.0 with `div_not_found_below_cap`.
- Full tied grid (12 s × 8 r, `gene_range_margins.json`): min 0.963 at (0.6, 0.8); min for s ≥ 0.8 **1.102** (the f16 meets
  ≥ 1.0 for s ≥ 0.8 including the soft-torsion corner); monotone in both genes, no ridge (pytest).
- EA/CG sensitivity of the default flutter margin: 0.35/0.38 → 1.53, 0.38/0.42 → 1.43, 0.40/0.42 → 1.40, **0.40/0.43 → 1.39**,
  0.40/0.45 → 1.38, 0.42/0.45 → 1.35. The choice moves the margin by about ±0.1, i.e. it is not a knob that changes
  feasibility at the default; the s 0.6 / r 0.8 corner stays between 0.94 and 1.05 for all of them.
- Not modelled: AIM-9 tip launchers/missiles and underwing stores, which dominate real F-16 flutter/LCO behaviour (they would
  enter as a fixed tip mass, which Phase 1 keeps at 0).
