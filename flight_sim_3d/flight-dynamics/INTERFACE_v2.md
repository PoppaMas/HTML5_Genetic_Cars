# Flight Dynamics flex v2 interface: distributed wing structure, flexible empennage and fuselage, multi-fidelity hook

Owner: Flight Dynamics. Consumers: Genome Architect (`genome/`), Evolution Runner (`evolution/fidelity.py`). Neither folder
was edited. The v1 contract (`INTERFACE.md`, 4 genes, `flexwing.py`) is unchanged and still valid. This document only adds v2.

Files: `flexbody.py` (v2 model, coupler, margins, validation helpers), `flexeval.py` (`evaluate()` reference wrapper,
`project_to_reduced()`, `model_version()`), `jsbsim_root_v2/` (prepared copies), `test_flexbody.py` (v2 + fidelity tests),
`v2_compare.py` (benchmarks and rank correlation), `v2_validate.py` + `_oas_check.py` (validation), results in
`v2_benchmarks.json`, `spearman.json`, `v2_validation.json` and raw checkpoints in `v2_results/`.
All numbers were measured with JSBSim 1.3.1 on this box (2026-10-06). Structural data are **notional**, not GVT data.

---------------------------------------------------------------------------------------------------------------------
## 1. Gene list (v2 struct genome): 12 genes, plus 2 optional asymmetry genes

| # | name | range | scale | default | meaning |
|---|---|---|---|---|---|
| 0 | `wing_ei_root` | 0.6 – 2.0 | log | 1.0 | wing bending-stiffness multiplier at the beam root (control point CP0) |
| 1 | `wing_ei_taper_1` | 0.75 – 1.05 | linear | 1.0 | EI multiplier ratio CP1/CP0 (η 0.25 / 0) |
| 2 | `wing_ei_taper_2` | 0.75 – 1.05 | linear | 1.0 | ratio CP2/CP1 (η 0.5 / 0.25) |
| 3 | `wing_ei_taper_3` | 0.75 – 1.05 | linear | 1.0 | ratio CP3/CP2 (η 0.75 / 0.5) |
| 4 | `wing_ei_taper_4` | 0.75 – 1.05 | linear | 1.0 | ratio CP4/CP3 (η 1 / 0.75) |
| 5 | `wing_gj_ratio_root` | 0.8 – 1.15 | linear | 1.0 | GJ multiplier / EI multiplier at the root |
| 6 | `wing_gj_ratio_tip` | 0.8 – 1.15 | linear | 1.0 | GJ multiplier / EI multiplier at the tip (linear in η between) |
| 7 | `wing_nsm_root` | **1.0 – 1.25** | log | 1.0 | non-structural mass multiplier at the root (P2.5: floor raised from 0.8; cannot lighten below baseline) |
| 8 | `wing_nsm_tip` | **1.0 – 1.25** | log | 1.0 | non-structural mass multiplier at the tip (log-linear between; same floor) |
| 9 | `tail_stiffness_scale` | 0.6 – 2.0 | log | 1.0 | HT + VT EI and GJ multiplier (structural mass follows, min-gauge floored; root strength ∝ this gene, §12) |
| 10 | `fuselage_stiffness_scale` | 0.6 – 2.0 | log | 1.0 | aft-fuselage EI multiplier, vertical and lateral (structural mass follows, min-gauge floored; strength ∝ this gene, §12) |
| 11 | `struct_damping_ratio` | 0.005 – 0.05 | log | 0.02 | modal structural damping ratio, all bodies |
| 12* | `wing_asym_ei_delta` | −0.1 – 0.1 | linear | 0 | only with `asymmetric=True`: right EI,GJ ×(1+d), left ×(1−d) |
| 13* | `wing_asym_nsm_delta` | −0.1 – 0.1 | linear | 0 | only with `asymmetric=True`: right NSM ×(1+d), left ×(1−d) |

- **Encodings.** Either a named dict of physical values (missing keys take the default) or a vector in [0,1]^12
  (or [0,1]^14 with `asymmetric=True`) in the table order; decode: log `lo·(hi/lo)^u`, linear `lo + u·(hi−lo)`.
  `flexbody.decode_genome_v2` / `encode_genome_v2`; schema `flexbody.gene_schema(asymmetric)`.
- **Raises `ValueError`** (never clips): unknown keys; v1 gene names (`stiffness_scale`, `torsion_bend_ratio`, …: use
  `fidelity='reduced'`); fixed per-aircraft keys (elastic axis, section CG, tip mass: `x_ea`, `elastic_axis_frac`,
  `section_cg_frac`, `tip_mass_frac`, …); NaN/inf/non-numbers; values outside the range; asymmetric keys without the flag;
  vectors of the wrong length or with entries outside [0,1]. `evaluate()` validates the genome at every fidelity, rigid
  included.
- Why 12: 5 EI numbers are the minimum for the spec's ≥ 4 control points with bounded tapers; GJ and NSM vary
  linearly / log-linearly (2 numbers each). Tail and fuselage get one stiffness gene each (their spanwise shape stays the
  baseline one). Damping is shared. The reduced fidelity cannot see the tail/fuselage genes or the v2-only terms
  (empennage/fuselage loads, reversal). Together with its wing flutter bias (§8), this is why reduced and full rank the
  737 less alike than the c172x.

## 2. Parameterisation and smoothness rules

- Control points at η = 0, 0.25, 0.5, 0.75, 1 of the wing beam (from the fuselage side y0 = 0.10·s to the tip).
- EI multiplier at CP k = `wing_ei_root` · Π_{j≤k} `wing_ei_taper_j`. **Feasible by construction:** each quarter span can
  grow by at most 5 % or drop by at most 25 %, so there is no sawtooth and no outboard-heavy stiffness. Over the whole span
  the tip/root multiplier lies in [0.32, 1.22].
- GJ multiplier = EI multiplier × (`wing_gj_ratio_root` + (`wing_gj_ratio_tip` − `wing_gj_ratio_root`)·η).
  NSM multiplier = exp(linear in η between ln(root) and ln(tip)).
- Spanwise interpolation to the 32 strips: **monotone PCHIP in log space** (Fritsch–Carlson, no scipy). The result is
  positive, never overshoots between two control points, and is C¹. Tested on 200 random genomes: the strip-to-strip EI and
  GJ ratio never exceeds 1.05^(1/4) outboard.
- The multipliers act on the baseline distributions: EI, GJ ∝ (c/c_root)³ and mass ∝ c, calibrated so the baseline uncoupled
  first bending/torsion frequencies equal v1's `f_b1_hz` / `f_t1_hz`.
- Soft curvature term: `J_smooth = 0.05·Σ_k (ln taper_{k+1} − ln taper_k)²` (second difference of ln EI over the control
  points, 3 terms; ≤ 0.004 at the range corners).
- Structural mass follows stiffness per strip, through the **minimum-gauge factor** `g(s) = f_min + (1 − f_min)·s^k`
  (`flexbody.MIN_GAUGE`, f_min = 0.5, k = 1; changed with the §12 mass-exploit fix, the v1 rule was g(s) = s):
  `m = m0·[(1−sf)·nsm + sf·(w_ei·g(ei) + (1−w_ei)·g(gj))]` (sf = 0.55, w_ei = 0.5). HT/VT: struct_frac 0.55 with
  g(`tail_stiffness_scale`). Fuselage: `aft_mass·[(1−sf_f) + sf_f·g(fuselage_stiffness_scale)]`. The same floored mass
  goes into the FE mass matrix (frequencies, margins), `mass` / `J_mass` and the JSBSim point masses. g(1) = 1, so the
  baseline is unchanged; halving a stiffness gene saves only 25 % of that body's structural mass, not 50 %.
- Symmetric by default (left wing = right wing, sharing one FE solution). `asymmetric=True` adds genes 12–13. A
  separate wingL margin block is computed only when the deltas are non-zero.

## 3. Structural model, modes, truncation

| body | FE | modes kept | first modes at the baseline (Hz), c172x / T38 / 737 / f16 |
|---|---|---|---|
| wing (each side) | 32 Hermite bending + linear torsion + in-plane elements, CG-EA inertia coupling | 3 bending, 2 torsion, 1 in-plane | b1 6.99 / 10.0 / 2.70 / 8.50; ip 17.5 / 25.0 / 6.75 / 21.3; t1 22.3 / 34.9 / 10.8 / 28.0 |
| HT (each side) | 12 elements | 2 bending, 1 torsion | 14.0 / 20.0 / 6.0 / 20.0 |
| VT | 12 elements | 2 bending, 1 torsion | 12.0 / 14.0 / 4.5 / 15.0 |
| aft fuselage, vertical / lateral | 12 elements, tail mass at the tip | 2 + 2 | 9 / 12 / 3 / 10 and 10 / 13 / 3.5 / 11 |

That is 25 modal DOF per aircraft. Modes are classified by their strain-energy share (bending, torsion or in-plane), and
their signs are fixed deterministically. **Modal-truncation check** (`flexbody.truncation_check`, test tolerance 2 %): wing
root bending moment and tip deflection (static aeroelastic at 0.9 V_D with 1° + 1 g; 1-cos gust) and the wing flutter speed
change by at most 0.017 % (c172x), 0.57 / 0.75 % (T38 baseline / soft tip), 0.39 / 0.45 % (737) and 0.58 / 0.67 % (f16)
when going from N = 6 to N+1 and N+2 modes.

## 4. Coupling per body (all relative to the 1-g trim shape; feedback through `external_reactions`)

Each JSBSim frame (1/120 s) the coupler reads the FDM state and integrates the 25-DOF modal system (Newmark average
acceleration, 2 substeps, implicit in aero stiffness and damping). It then writes the **elastic aero increments** relative
to the trimmed static equilibrium: zero at t0, so JSBSim's rigid tables keep owning trim. The increments go to
`external_reactions/flexwing_F/{x,y,z}` (BODY, lbf, at AERORP) and `external_reactions/flexwing_M/{l,m,n}` (lbf·ft),
after which the real `fdm.run()` is called.

| body | loads on the structure | elastic feedback to the FDM |
|---|---|---|
| wings | Schrenk share of FDM lift, roll damping q·κ·a·c·p·y/V, aileron strips (FDM surface positions, Cl_δa calibrated), inertia −m(g·Nz − ṗ·y), in-plane: inertia m·g·Nx and half the airframe drag | ΔL, Δroll, Δpitch from Δα = cosΛ·θ − sinΛ·w′ − ẇ/V + d34·θ̇/V; elastic dihedral β·w′ |
| HT | Δα_ht = (α − α_trim)(1 − dε/dα) + q·l_h/V (+ Δδe for all-moving T38/f16), elevator strips (c172x/737), p·y/V, inertia (Nz, q̇·x, ṗ·y) | ΔL, Δroll, Δpitch (tail effectiveness and pitch damping change) |
| VT | Δβ_fin = −(β − β_trim) + r·l_v/V − p·z_v/V, rudder strips, inertia (Ny, ṙ·x, ṗ·z) | ΔY, Δroll, Δyaw |
| fuselage vertical | tail strip loads at arm x, own inertia (−g·Nz, q̇·x) | the tail rides on the fuselage tip: incidence −w′_tip and plunge velocity ẇ_tip enter the HT Δα, so pitch damping and tail effectiveness drop |
| fuselage lateral | fin loads, own inertia (−g·Ny, ṙ·x) | fin sideslip −v′_tip and lateral plunge enter the fin Δβ, so yaw damping and fin effectiveness drop |

Control-effectiveness loss is part of the dynamics: aileron, elevator and rudder strip loads deform the structure, and the
resulting elastic Δα is fed back.

**Mass feedback.** `apply_mass_v2(fdm, model)` is called before IC/trim and sets 5 point masses:
- wing R/L: v1's injected pair, at the v2 spanwise mass centroid;
- `flexbody_ht` at AERORP + l_h;
- `flexbody_vt` at AERORP + l_v, z_v;
- `flexbody_fus` at AERORP + l_h/2.

Weight, CG and inertia change accordingly (tested: Δweight = total_lb exactly; CG moves aft; Iyy grows). At the baseline genome every delta is exactly 0.0.

## 5. Outputs, constraints and units

`flexeval.evaluate(gains, struct_genome, scenarios, model, *, fidelity, root, dt=1/120, record=False, profile=None,
weights=None, root_v2=None, sim=None)` returns:

| field | type / units | meaning |
|---|---|---|
| `cost` | float | `float(np.mean(per-scenario costs))` at every fidelity (lower is better) |
| `terms` | dict, always the 24 keys of `flexeval.TERM_KEYS` | `track, effort, comfort, heading, hold` (`hold` = the Runner's v5 hold_osc, 0.0 unless the profile has w_hold > 0; means over the scenarios flown to the end, as `evolution/fidelity.py aggregate()`), `J_flutter_margin, J_div_margin, J_mass, J_bm_rms, J_bm_peak, J_tip, J_twist` (same meaning as v1 / the Runner), and the v2-only terms `J_reversal_margin, J_tail_bm_peak, J_fus_bm_peak, J_smooth`, the pre-flight limit-load sizing terms `flexbody.SIZING_TERMS` = `J_wing_bm_limit, J_wing_torque_limit, J_wing_ip_limit, J_wing_tip_bm_limit, J_tail_bm_limit, J_fus_bm_limit` (reduced and full) and the full-only flown terms `J_wing_torque_peak, J_wing_ip_peak` (§12). Inapplicable = 0.0, never NaN (a non-finite term raises). |
| `terms_available` | list | the keys that are actually computed at this fidelity |
| `status` | str | `ok`, the first non-ok scenario status (sim fail, `structural_ultimate`, `structural_ultimate_empennage`), or a margin-gate fail `flutter` / `divergence` / `reversal` / `margin_error` (not flown, cost = `fail_cost` = 2·profile.fail_base = 2000) |
| `fidelity`, `model_version` | str | §7 |
| `margins_fidelity`, `margin_gate` | str / float | `None` for rigid; `'reduced'` with gate 0.9, `'full'` with gate 1.0 |
| `margins` | dict | full: `flutter_margin`, `div_margin`, `reversal_margin` (×V_D EAS, conservative min over blocks, cap 3.0), `margin_cap`, `v_dive_keas`, `q_dive_psf`, `margin_error`, `f_modes_hz` {body: [Hz]}, `blocks` {wingR, (wingL), empennage_pitch, empennage_yaw: `flutter_margin`, `flutter_margin_qs`, `coalescence_margin`, `coalescence_margin_undamped` (diagnostic), `flutter_not_found_below_cap`, `f_flutter_hz`, `div_margin`, `div_not_found_below_cap`, `<ctrl>_reversal_margin`, `<ctrl>_reversal_not_found_below_cap`, `<ctrl>_effectiveness_at_VD` (ctrl = aileron / elevator / rudder; elastic ÷ rigid), `ht_alpha_effectiveness_at_VD` / `vt_alpha_effectiveness_at_VD` (tail and fin lift effectiveness incl. fuselage bending), `margin_error`}. reduced: v1 `flutter_margin`, `div_margin`, `reversal_margin: None`, `f_modes_hz`, `detail` |
| `mass` | dict, lb | `wingR_lb, wingL_lb, ht_lb, vt_lb, fus_lb, total_lb` (structural mass change vs baseline, fed to JSBSim), `baseline_flexible_lb`, `total_frac` |
| `loads` (full) | dict, lbf·ft | per root load `wingR_bm, wingL_bm, wingR_torque, wingL_torque, wingR_ip_bm, wingL_ip_bm, htR_bm, htL_bm, vt_bm, fusV_bm, fusL_bm`: `peak_abs_lbft` (max over scenarios), `rms_dev_1g_lbft` (RMS deviation from the 1-g value, mean over scenarios), `trim_1g_lbft` |
| `per_scenario` | list | `cost, sim_cost, status, t_end, track, effort, comfort, struct{terms}, delta_mass_lb_applied, wall_s`, full also `loads`, `tip_max_ft`, `twist_max_deg`, `tail_ratio`, `fus_ratio`, `torque_ratio`, `ip_ratio` (flown peak / sizing allowable) |
| `sizing` (reduced, full) | dict | `flexbody.sizing_v2` result: `terms`, `ratios`, `allowables_lbft`, `demand_lbft` per check (`wingR/L_bm`, `wingR/L_torque`, `wingR/L_ip`, `ht`, `vt`, `fusV`, `fusL`), §12 |
| `projection` (reduced) | dict | `project_to_reduced` result |
| `telemetry` (record=True) | list per scenario | `trajectory` (the Runner's TrajRecorder output; flex fidelities add channels `flex.<coupler key>`), `structure` (full-rate coupler histories; full fidelity also has the telemetry scalars `flexbody.TELEMETRY_KEYS_V2`) and, at full fidelity, `nodes` (FE node layout plus per-frame nodal values of every body, §11). These exports exist only with record=True. |

Root-load sign/units: bending moments lbf·ft, positive up-bending (wing/HT) or toward +side force (VT/fusL); torque nose-up
positive; fuselage moments at the wing station. The complete per-field sign table (probed from the code) is in §11.

**Constraints and cost terms (weights `flexbody.StructWeightsV2`, defaults = v1's).**
- **Hard gate before flying:** min(flutter, divergence, reversal) margin < 1.0 at full (< 0.9 at reduced, which has
  flutter/divergence only) → not flown, cost = fail_cost.
- **Margin penalties:** `J_x = w·max(0, (1.2 − m)/0.2)²` for flutter, divergence and reversal (w = 1 each). Shaping:
  penalty only below 1.2 (1.0 → 1 per margin), **exactly 0 at or above 1.2** (no reward for extra margin), hard fail below
  the gate. The 3.0 "not found below cap" value therefore neither rewards nor penalises (tested:
  `test_margin_terms_shaping_no_reward_above_target`). Genome's `fd_bridge.precheck_v2` uses the same formula.
- **Limit-load sizing (pre-flight, §12):** `J = w·max(0, demand/allowable − 1)²` for the wing root bending (w 1.0),
  torque (0.1) and in-plane (0.1) moments, the HT/VT roots (1.0) and the aft fuselage vertical/lateral (1.0). Allowables
  scale with each body's stiffness gene / local EI (§12). At the baseline every term is exactly 0. A body's own term is ~0 when *that body's* scale ≥ 1; cross-body loads can still raise a term (e.g. stiffer tail → fuselage).
- `J_mass = 0.3·total_frac` (masses min-gauge floored, §2), with total_frac = total structural Δmass / baseline flexible mass (wings + tails + aft
  fuselage; same definition at reduced and full).
- Wing response terms as v1: `J_bm_rms` (0.25·RMS(BM − BM_1g)/BM_1g), `J_bm_peak` (2·hinge² above n_limit·BM_1g·wing_ei_root),
  `J_tip`, `J_twist`.
- `J_tail_bm_peak`, `J_fus_bm_peak`: hinge² of peak/allowable. The allowable is the tail/fin lift at q_D with 5°
  incidence (plus n_limit inertia for the vertical fuselage), × the stiffness gene.
- `J_wing_torque_peak`, `J_wing_ip_peak` (full only, w = 0.1 each): hinge² of the flown peak root torque / in-plane moment
  over the §12 sizing allowable.
- **Ultimate fails:** a wing peak > 1.5·allowable → `structural_ultimate`; a tail/fuselage ratio > 1.5 →
  `structural_ultimate_empennage`.

## 6. Per-aircraft fixed values (not genes)

| aircraft | span ft | Λ° | x_ea / x_cg (chord) | wing f_b1 / f_t1 / f_ip Hz | wing mass lb | V_D KEAS | HT span ft / mass lb / f_b1 / f_t1 / control | VT height ft / mass lb / f_b1 / f_t1 | l_h ft / aft-fuselage lb / f_v1 / f_l1 |
|---|---|---|---|---|---|---|---|---|---|
| c172x | 36.0 | 0 | 0.38 / 0.42 | 7.0 / 22.0 / 17.5 | 180 | 180 | 11.3 / 26 / 14 / 40 / elevator | 5.0 / 16 / 12 / 35 | 15.7 / 120 / 9.0 / 10.0 |
| T38 | 25.25 | 24 | 0.40 / 0.42 | 10.0 / 35.0 / 25.0 | 800 | 595 | 11.7 / 150 / 20 / 50 / all-moving | 8.1 / 120 / 14 / 48 | 13.0 / 1000 / 12 / 13 |
| 737 | 94.7 | 25 | 0.36 / 0.38 | 2.7 / 11.0 / 6.75 | 10500 | 400 | 46.4 / 2400 / 6 / 18 / elevator | 23.1 / 1800 / 4.5 / 14 | 48.0 / 12000 / 3.0 / 3.5 |
| f16 | 30.0 | 32 | 0.40 / 0.43 | 8.5 / 28.0 / 21.25 | 1600 | 595 | 12.6 / 400 / 20 / 50 / all-moving | 8.4 / 250 / 15 / 40 | 16.5 / 2500 / 10 / 11 |

Elastic axis, section CG and tip mass (0) stay fixed per aircraft, exactly as v1 (INTERFACE.md §0). Wing values are v1's
`flexwing.params_for`. Empennage and fuselage values are notional (`flexbody.V2_PROFILES`). The T38 HT/VT torsion frequencies
were raised from 45/40 Hz to 50/48 Hz so that the baseline empennage meets 1.2 V_D like a certified aircraft (the fin
coalescence margin was 1.07).

Baseline margins (full v2, ×V_D):

| aircraft | flutter (block) | divergence | reversal (block) |
|---|---|---|---|
| c172x | 1.228 (wing) | 1.669 | 1.473 (aileron) |
| T38 | 1.162 (wing) | 3.0 (flag) | 1.998 (aileron) |
| 737 | 1.177 (wing) | 3.0 (flag) | 1.273 (aileron) |
| f16 | 1.307 (wing) | 3.0 (flag) | 3.0 (flag) |

v1 baseline flutter margins were 1.233 / 1.269 / 1.240 / 1.412. v2 is lower on the swept wings because of its extra modes.

## 7. Fidelity contract (agreed with the Evolution Runner; implemented in `flexeval.py`)

`evaluate(gains, struct_genome, scenarios, model, *, fidelity: Literal['rigid','reduced','full'], root, dt=1/120, record=False)`
→ `{cost, terms, status, telemetry?, model_version, fidelity, terms_available, margins_fidelity, …}` (§5).

| fidelity | structure | coupling | gate | `model_version` |
|---|---|---|---|---|
| `rigid` | none (no coupler, struct genome only validated) | `evolution/sim.py simulate()` exactly as the Runner calls it; **bit-identical** to the Phase-1 rigid eval (tested against the package `evolution.sim`) | – | `rigid:jsbsim1.3.1:<sha8>`, **byte-identical to `evolution/fidelity.py model_version(profile,'rigid')`** (tested; e.g. c172x `rigid:jsbsim1.3.1:e0a73fc9`) |
| `reduced` | v1 FlexWing, n_bend = 1 + torsion (2 modes per semi-wing), v1 coupler, v1 GA substeps (2), prepared root = `root` | wing only; v2 genome → `project_to_reduced` | 0.9 (`margins_fidelity='reduced'`) | `reduced:flexv1:<sha8>` over flexwing.py, coupled_sim.py, flexbody.py, flexeval.py, v1 + v2 parameters, gene schema, weights, substeps, gate, aircraft files |
| `full` | v2 (this document), prepared root = `root + "_v2"` | all bodies | 1.0 | `full:flexv2:<sha8>` over flexbody.py, flexwing.py, flexeval.py, v2 parameters, gene schema, weights, substeps, gate, aircraft files |

- `model_version` changes when code, aircraft data, parameters, gene ranges or weights change. All of these are tested
  except the substeps/terms list, which are hashed too. Per-genome genes are inputs, not part of the version.
- Flex fidelities use the Runner's own hook protocol `simulate(..., flex=FlexHookV2)`:
  - `attach` sets the mass before IC/trim;
  - `wrap` installs coupler + proxy after trim;
  - `state()` serves the recorders;
  - `finish(status)` returns the post-flight terms.
  With an older `sim.py` that lacks `flex=`, the wrapper falls back to swapping `_new_fdm` on a private module instance.
  Both paths are bit-identical (tested).
- **`project_to_reduced(struct_genome, model)`** is deterministic and uses weights from the baseline v2 wing's modes:
  - `s` = 1st-bending strain-energy average of the EI multiplier;
  - `s·r` = 1st-torsion strain-energy average of GJ;
  - `nsm` = 1st-bending kinetic-energy average of the NSM multiplier;
  - `zeta` passes through;
  - asymmetric genomes use the L/R mean;
  - tail and fuselage genes are dropped (listed in `_detail.dropped`);
  - no clipping: values outside the v1 ranges are flagged in `_detail.outside_v1_gene_range`.
  Uniform multipliers project exactly.
- **First-mode frequency error, reduced vs full** (32 random genomes, `v2_validation.json`):
  - bending: mean −5.3 / −4.8 / −4.8 / −4.9 %, max |·| 10.4 / 9.5 / 9.5 / 9.6 % (c172x / T38 / 737 / f16);
  - torsion: mean −3.9 / −2.2 / −3.1 / −2.3 %, max 7.6 / 6.0 / 6.4 / 4.5 %;
  - 0.00 % at the baseline.
  The bias is low because v1 ties structural mass to the uniform `s` while v2's structural mass follows the
  outboard-decreasing local EI. A kinetic-energy mass projection was tried and rejected: it fixes bending but needs
  nsm < 0.2 and moves the error to torsion.
- Reduced fidelity applies only the projected wing mass to JSBSim (v1 rule, no min-gauge floor: v1 stays bit-identical).
  `J_mass`, `J_smooth` and the five §12 sizing terms are computed from the v2 model, so they are identical at reduced and
  full.

## 8. Runtimes and rank correlation

> The runtime, rank-correlation and validation numbers in §8–§9 (`v2_benchmarks.json`, `spearman.json`,
> `v2_validation.json`) were measured **before** the §12 mass-exploit fix (min gauge + sizing terms). Runtimes are
> unaffected (sizing is a few ms per genome); the Spearman values would shift slightly because the total cost changed.

**Wall time per 90 s scenario** (`v2_compare.py bench`; mean of the 3 Phase-1 scenarios, min of 2 repeats, dt = 1/120, BLAS
1 thread; `v2_benchmarks.json`). The box is shared with other agents (load average 22–26 during the run), so the CPU time
(`process_time`) is the robust number and wall time is shown for reference. `v1 flex` is the v1 default wing (n_bend = 2 +
torsion), run through the same hook.

| aircraft | rigid cpu / wall | reduced cpu / wall | full cpu / wall | v1 flex cpu / wall | full/v1 | full/reduced | full/rigid |
|---|---|---|---|---|---|---|---|
| c172x | 0.22 / 0.45 s | 1.24 / 2.90 s | 1.76 / 4.60 s | 1.20 / 2.68 s | 1.47× | 1.42× | 8.1× |
| T38   | 0.18 / 0.48 s | 1.21 / 3.62 s | 1.70 / 5.46 s | 1.12 / 3.72 s | 1.51× | 1.41× | 9.2× |
| 737   | 0.18 / 0.44 s | 1.20 / 3.36 s | 1.66 / 4.53 s | 1.13 / 3.38 s | 1.47× | 1.39× | 9.2× |
| f16   | 0.24 / 0.47 s | 1.25 / 1.42 s | 1.75 / 2.19 s | 1.42 / 1.63 s | 1.24× | 1.40× | 7.5× |

- An earlier parallel wall-clock run (load 12–18, `v2_results/bench_parallel_pre_taper/`) gave rigid 0.3–0.65 s,
  reduced 2.1–3.4 s, full 2.4–5.3 s and v1 flex 1.3–2.6 s, which is the same ordering.
- The reduced/full times are whole `evaluate()` calls: model build and margin pre-screen (once per genome, ≈ 0.35 s CPU
  for `margins_v2`) amortised over the 3 scenarios. The v1-flex column has no margin screen.
- **Full v2 costs ≈ 1.2–1.5× v1 CPU**, well inside the 2–3× budget. No extra "reduced-mode" option is proposed. The
  `reduced` fidelity (≈ v1 cost) stays available for screening.

**Spearman rank correlation of the genome total cost** (`v2_compare.py rank`; 32 individuals per set, 3 scenarios × 90 s;
`spearman.json`).
- `genomes` = the best Phase-1 gains with random v2 struct genomes.
- `joint` = random struct genomes with gains × exp(U(±ln 1.5)).
- Ties (fail_cost) get average ranks. "both ok" excludes genomes that hard-fail at either fidelity.

| set | reduced vs full | (both ok, n) | rigid vs full | gate agreement | full fails / reduced fails |
|---|---|---|---|---|---|
| c172x genomes | **0.942** | 0.909 (27) | undefined (rigid cost constant) | 0.875 | 5 / 1 |
| c172x joint   | **0.940** | 0.920 (29) | −0.047 | 0.906 | 3 / 0 |
| 737 genomes   | **0.843** | 0.617 (23) | undefined | 0.719 | 9 / 0 |
| 737 joint     | **0.870** | 0.585 (20) | 0.158 | 0.625 | 12 / 0 |

**Reduced vs full margins** (same genomes):
- **Wing flutter margin, reduced − full (wing block):**
  - c172x mean +0.029 (max 0.073);
  - 737 mean +0.087 / +0.093 (max 0.13 / 0.51);
  - rank correlation of the margins 0.99 / 0.99 / 0.99 / 0.97.
- **Divergence margin, reduced − full:**
  - c172x mean +0.020 (max 0.063);
  - 737 is capped at 3.0 at both fidelities (swept wing, κ-washout).
- **Reading:**
  - Reduced ranks well where the gate agrees.
  - On the swept 737 it is optimistic by ~0.09 in flutter margin, so it passes genomes that full v2 fails (§10.1).
  - Rigid carries no structural information.

## 9. Validation

**Hand calcs** (`flexbody.hand_calcs`; uniform cantilever L = 10, c = 2, EI = 2e6, GJ = 1.5e6, m = 0.5, EA 0.40c, full-span
25 % flap; FE with 32 elements and the default 3b+2t truncation):

| quantity | closed form | FE | diff |
|---|---|---|---|
| 1st bending f = (1.875²/2π)√(EI/mL⁴) | 11.19182 Hz | 11.19182 Hz | +0.00001 % |
| 1st torsion f = (1/4L)√(GJ/I_α) | 86.6025 Hz | 86.6112 Hz | +0.010 % |
| strip-theory divergence q_D = (π/2L)²·GJ/(c·e·a) | 9817.5 | 9821.4 | +0.040 % |
| aileron reversal q_R (exact strip theory, torsion only) | 8643.8 | 8647.3 | +0.041 % |
| tip deflection pL⁴/8EI | 0.0062500 | 0.0062500 | −0.0003 % |

Mesh convergence (8/16/32/64 elements): the q_D error is 0.65 / 0.16 / 0.04 / 0.01 %, which is O(h²).

**OpenAeroStruct 2.12.0** (in `_oas_venv`, works). The test case is a rectangular AR-10 wing (semi-span 10 m, chord 2 m,
tube spar r = 0.1 m, t = 0.01 m, E 70 GPa, G 30 GPa, EA at 0.35c, VLM 20×2 panels per side, α = 2°). flexbody's strip
theory uses the same EI/GJ/EA and the OAS rigid CL_α (4.903/rad) spread uniformly:

| q (Pa) | q/q_D | CL diff | tip deflection diff | tip twist diff |
|---|---|---|---|---|
| 551 | 0.03 | +0.1 % | +15.2 % | +1.4 % |
| 1531 | 0.08 | +0.2 % | +15.8 % | +1.7 % |
| 3001 | 0.15 | +0.3 % | +16.5 % | +2.2 % |
| 4961 | 0.24 | +0.2 % | +17.1 % | +2.6 % |
| 7411 | 0.36 | −1.3 % | +16.0 % | +1.7 % |

- **Divergence:** flexbody q_D = 20 398 Pa (closed form 20 390, +0.04 %). OAS has no divergence solver, so OAS q_D was
  extrapolated from the twist amplification (q/θ_tip linear in q): 20 689 / 20 167 / 19 021 Pa for fit windows
  q < 0.26 / 0.37 / 0.55 q_D. That puts flexbody at **−1.4 / +1.1 / +7.2 %**.
- Above ~0.4 q_D OAS deflections exceed 20 % of the semi-span (geometrically nonlinear), which is why the widest window
  drifts.
- Strip theory over-predicts bending deflection by ~16 % here: there is no VLM tip loss, so more load sits outboard. The
  FDM-coupled model uses a Schrenk lift share for the rigid load (which has tip loss) but strip theory for the elastic
  increments.

**Guarantees with tests** (`test_flexbody.py`, `test_flexwing.py`):
- flex is opt-in;
- flex off on the v2 prepared copies is bit-identical to stock (4 aircraft);
- v2 coupler with exact-zero feedback is bit-identical to stock (4 aircraft);
- the v1 path is bit-identical to before v2 (legacy fingerprint, 4 aircraft, full float64);
- rigid fidelity is bit-identical to `evolution.sim.simulate`;
- deterministic (repeat runs `==`, all fidelities);
- finite outputs (terms, margins, histories);
- `model_version` tracks code, data, parameters, gene ranges and weights.

## 10. Caveats and open decisions

1. **The reduced margins are optimistic on swept wings.** The reduced wing flutter margin is higher than full v2's wing
   block by +0.03 (c172x) and +0.09 (737) on average; the ranking agrees (Spearman 0.99). With the 0.9 screening gate, the
   reduced fidelity passed every 737 genome while full failed 9/32 and 12/32. **Decision:** keep 0.9, or use a per-aircraft
   reduced gate (≈ 1.0 for 737/T38), or subtract a bias.
2. **About a third of uniformly random v2 genomes fail the full gate** (c172x 20/64, 737 24/64, T38 26/64, f16 16/64 with
   the final ranges and T38 tail), mostly wing flutter at low root stiffness. That is physical for a design space centred on a baseline
   with only 1.16–1.31 V_D margin, but it costs GA evaluations. Options: raise the `wing_ei_root` floor to 0.7, or seed the
   initial population near the baseline.
3. The taper ranges were narrowed from 0.5–1.05 to 0.75–1.05 at resume (tip/root EI down to 0.06 was unphysical and made
   55–80 % of random genomes flutter). The T38 empennage torsion frequencies were raised (§6). Both are encoded in
   `model_version`.
4. Rigid cost carries no structural information (Spearman rigid-vs-full ≈ 0, §8). Rigid is only useful for gains-only
   screening.
5. All empennage/fuselage numbers are notional. Effects that are not modelled:
   - T-tail and HT-on-fin;
   - fuselage torsion;
   - engine/pylon and store masses;
   - unsteady (C(k)) aero;
   - transonic effects (M clamp 0.9);
   - the trim tail load (tail loads are perturbations from trim).
6. **Telemetry mapping: who maps what** (status 2026-10-06; ER and Sim Bridge code read, not edited).
   - **ER** (`evolution/fidelity.py`: `FlexState`, `fd_to_structure_channels`, schema `evolution-flex-state/2`) maps the
     **wings** from FD's v2 modal state: 9 evenly spaced nodes per semi-wing (strip values interpolated, 0 at the root,
     FD tip value at ξ = 1), body FRD metres about the CG, with `dz = −w·0.3048`, `dy = 0`, `wingR.twist = +θ_R` and
     `wingL.twist = −θ_L`.
   - ER also passes every `coupler.last` key through as `flex.<key>`. This now includes the 6 new telemetry scalars
     (§11) without any ER change.
   - **Sim Bridge** (`sim_bridge/v2_map.py`) maps the HT, VT and aft fuselage from the tip scalars, using an assumed
     cantilever shape.
   - **FD** now also exports the exact FE nodal values of every body (wings, HT L/R, VT, fusV, fusL) with their node
     coordinates. These are in `evaluate(record=True)` → `telemetry[i]['nodes']`, and from any modal state through
     `flexbody.node_values(mdl, eta)` and `flexbody.node_layout(mdl, rp_offset_body_ft)`. ER's `FlexState` already
     carries `eta`, so it could map HT/VT/fuselage from the modal state and drop the assumed shape. That wiring is ER's /
     Sim Bridge's to do.
   - Wing in-plane deflection (`v_ft`, `wing*_tip_ip_ft`) is **chordwise (body x, + aft)**. It is not the viewer's `dy`.
7. `evaluate()`'s `fail_cost` is 2·profile.fail_base (the Runner's rule). The v1 StructWeights default is the same 2000.

## 11. Telemetry sign conventions

These signs were read from `flexbody.py` and checked with deterministic probes:
- `flexbody.static_probe`: the static response to one JSBSim input;
- `flexbody.mode_probe`: one elastic mode, fed through a two-way coupler.

The probes are pinned by `test_flexbody.py::test_sign_probe_*` on all four aircraft; the raw output is in
`v2_results/sign_probe.json`.

**Frames:**
- **JSBSim body axes:** x forward, y right, **z down**.
- **FD's `w`** is the body's own "positive load" direction: **up (body −z)** for the wings, HT and fusV, and **toward body
  +y** for the VT and fusL.
- **Deflections** are the own elastic deflection relative to the body's clamped root. They include the 1-g trim shape.
  - HT strips also ride on the fusV tip: plunge `fusV_tip_w_ft` and incidence `ht_incidence_deg`.
  - VT strips also ride on the fusL tip: lateral `fusL_tip_w_ft` and incidence `vt_sideslip_deg`.
  - The HT/VT tip values do **not** include that fuselage motion.
- **Tail and fin root loads** are perturbations from trim; the trim tail load is not modelled.

| field (`flex.<key>`) | units | axis | + direction | reference point |
|---|---|---|---|---|
| `wingR_bm`, `wingL_bm` | lbf·ft | bending about the wing chord line | up-bending: up load × arm, tip toward body −z | beam root (y0) |
| `wingR_torque`, `wingL_torque` | lbf·ft | about the elastic axis | nose-up (LE up), on both sides | EA at the beam root |
| `wingR_ip_bm`, `wingL_ip_bm` | lbf·ft | in-plane (chordwise) bending | aft load × arm, tip bends **aft** | beam root |
| `htR_bm`, `htL_bm` | lbf·ft | HT bending | up-bending | HT root (perturbation from trim) |
| `vt_bm` | lbf·ft | fin bending | load toward body +y, tip toward +y | fin root (perturbation from trim) |
| `fusV_bm` | lbf·ft | aft-fuselage vertical bending | **tail-up** bending: Σ up load × distance aft, tail deflects up | wing station (AERORP x) |
| `fusL_bm` | lbf·ft | aft-fuselage lateral bending | tail load / deflection toward body +y | wing station |
| `tip_w_ft_R`, `tip_w_ft_L` | ft | wing tip, normal to the chord plane | **up** (body −z) | beam root |
| `tip_twist_R_deg`, `tip_twist_L_deg` | deg | elastic twist about the EA | **LE up** (nose-up, raises local α) on both sides | beam root |
| `wingR_tip_ip_ft`, `wingL_tip_ip_ft` (**new**) | ft | in-plane (chordwise) tip deflection | **aft** (body −x) | beam root |
| `ht_tip_w_ft` (right HT), `htL_tip_w_ft` (**new**) | ft | HT tip | **up** (body −z) | HT root (excludes the fusV tip plunge) |
| `ht_tip_twist_deg` (right), `htL_tip_twist_deg` (**new**) | deg | HT elastic twist about its EA | **LE up** (nose-up) on both sides | HT root (excludes `ht_incidence_deg`) |
| `vt_tip_w_ft` | ft | fin tip | toward body **+y** (right) | fin root (excludes the fusL tip motion) |
| `vt_tip_twist_deg` (**new**) | deg | fin elastic twist about its EA | **LE toward +y**: right-hand about body +z (down), i.e. **left-hand** about the root→tip (upward) axis | fin root (excludes `vt_sideslip_deg`) |
| `fusV_tip_w_ft` | ft | aft-fuselage tip (tail) vertical | **up** (tail up) | wing station |
| `fusL_tip_w_ft` | ft | aft-fuselage tip lateral | toward body **+y** | wing station |
| `ht_incidence_deg` | deg | tail incidence increment from fuselage slope, = −w′_fusV,tip | **nose-up / tail LE up** (raises tail α) | tail (fusV tip) |
| `vt_sideslip_deg` | deg | fin incidence increment from fuselage slope, = −w′_fusL,tip | **fin LE toward +y**. This adds to the fin angle of attack α_v = −β + r·l_v/V − p·z_v/V, so it is the **opposite sense to aircraft β**: the equivalent local sideslip increment is −`vt_sideslip_deg` | fin (fusL tip) |
| `dL_lbf` | lbf | elastic lift increment, normal to V in the body x–z plane | **up**; written as body F_x = dL·sin α, F_z = −dL·cos α (z down) | applied at the AERORP (`flexwing_F` location = AERORP in all four prepared XMLs) |
| `dY_lbf` | lbf | body y | **right** (+y) | applied at the AERORP |
| `dRoll_lbft` | lbf·ft | body x (JSBSim `l`) | right wing down | about the AERORP |
| `dPitch_lbft` | lbf·ft | body y (JSBSim `m`) | nose up | about the AERORP |
| `dYaw_lbft` | lbf·ft | body z (JSBSim `n`) | nose right | about the AERORP |

**Moment reference.** The moments are summed with strip arms measured from the AERORP:
- the wing quarter chord is swept through the MAC quarter chord at the AERORP;
- the HT and VT sit at `metrics/lh-ft` and `metrics/lv-ft` aft;
- the fin AC is z_v above.

The probe recovers these arms (HT torsion: −dPitch/dL lies within the HT strip x range). They are pure couples on
`flexwing_M` (BODY frame). JSBSim adds r(AERORP − CG) × F for the force on `flexwing_F`.

**Node exports** (`telemetry[i]['nodes']`, schema `fd-flexbody-nodes/1`, full fidelity, record=True only):
- **`components`:** one per body (`wingR, wingL, htR, htL, vt, fusV, fusL`), each with:
  - `axis_nodes_body_ft`: undeformed elastic-axis FE nodes, body FRD ft, origin = CG at trim. `rp_offset_body_ft` is
    the AERORP relative to the CG.
  - `node_station_ft` (from the clamped root) and `node_span_frac`;
  - the `fields` sign doc and the `root` description.
  - Node counts: wings 33, tails and fuselage 13 (node 0 = clamped root = 0).
- **`values[body][field]`:** one row per coupler step, aligned with `structure`. The fields are:
  - `w_ft`: same sign as the tip fields above;
  - `theta_deg`: surfaces only, same sign as the tip twists;
  - `v_ft`: wings only, + aft.

  The last node equals the tip scalar (tested).
- The geometry follows the strip arms and is approximate: no dihedral and no HT height.

**Mapping to Sim Bridge's viewer convention** (dz + down, dy + right, twist right-hand about node i → i+1):

| FD field | Sim Bridge value |
|---|---|
| wing / HT / fusV `w` | dz = −w·0.3048 |
| VT / fusL `w` | dy = +w·0.3048 |
| wingR θ | twist = +θ |
| wingL θ | twist = −θ |
| htail θ (left → right polyline, both halves) | twist = +θ |
| vtail θ (root → tip) | twist = **−θ** |
| wing `v` (+ aft) | a body-x displacement dx = −v·0.3048, not `dy` |

## 12. Limit-load sizing, minimum gauge and margin shaping (mass-exploit fix)

**Why.** Genome Architect found that a uniform-init c172x GA drove `fuselage_stiffness_scale` / `tail_stiffness_scale`
towards the 0.6 floor. Mass fell linearly with the gene (credit `J_mass = 0.3·total_frac`), but no strength check
tightened, because the tail/fuselage peak terms only bite on flown peaks above an allowable that the benign Phase-1
scenarios never reach. The fix makes every stiffness reduction pay for itself before flight. It changes reduced/full
costs and `model_version`; rigid is unchanged, and the baseline genome's terms are all exactly 0.

**(b) Minimum gauge** (`flexbody.MIN_GAUGE = {"f_min": 0.5, "k": 1.0}`, `gauge_mass_factor(s)`): the structural part of
each body's mass is `m0·g(s)`, `g(s) = f_min + (1 − f_min)·s^k`, instead of `m0·s` (§2). It applies to the wing strips
(EI and GJ parts separately), HT/VT and the aft fuselage, and to the FE mass, `mass`, `J_mass` and the JSBSim point masses
alike. At s = 0.6, g = 0.8, so the saving is 20 %, not 40 %. Non-structural mass (`wing_nsm_*`) is not gauge-floored; instead **P2.5 raised both gene floors to 1.0** (range 1.0–1.25), so the GA can only add non-structural mass, never lighten it below baseline.

**(a, c) Binding limit-load allowables** (`flexbody.sizing_v2`, pre-flight, reduced and full; weights in
`StructWeightsV2`):

`ratio = demand(current masses) / allowable`, `allowable = (1 + MS) · design load of the BASELINE structure · stiffness gene`
(MS = `design_margin_of_safety` = 0; strength ∝ stiffness, as the v1 wing allowable n_limit·M_1g·wing_ei_root).
`J = w·max(0, ratio − 1)²`, max over left/right. A body's own term is ~0 when *that body's* stiffness scale ≥ 1 at a
matched laminate (baseline masses), and grows quadratically below 1 against a linear, floored mass credit. Cross-body
loads can still raise a term when that body's gene is ≥ 1: e.g. c172x with `tail_stiffness_scale` = 1.5 and
`fuselage_stiffness_scale` = 1.0 gives `J_fus_bm_limit` ≈ 3.4e-4 (heavier/stiffer tail raises fuselage vertical demand).

| term (w) | check | scaling gene | design load (rigid load bases at n_limit, no elastic relief) |
|---|---|---|---|
| `J_wing_bm_limit` (1.0) | wing root bending | `wing_ei_root·(1 ± asym δ)` | n·W_d·(Schrenk lift share × arm) − n·g·(m·dy × arm) |
| `J_wing_torque_limit` (0.1) | wing root torque about the EA | `wing_ei_root·wing_gj_ratio_root` (= GJ scale) | \|n·W_d·(lift × (x_ea − 0.25)·c) + n·g·(m·dy × x_θ)\| + \|q_D·5°·(aileron hinge/torque per q·δ)\|, floored at 0.05·MAC·n·W_d·Σlift |
| `J_wing_ip_limit` (0.1) | wing root in-plane (chordwise) bending | `wing_ei_root` (in-plane EI follows EI) | q_D·S_w·C_D,d (0.03)·(drag share × arm) + 0.5·g·(m·dy × arm) (n_x,d = 0.5), floored at 0.1 × bending |
| `J_wing_tip_bm_limit` (1.0) | wing outboard bending at `tip_bm_eta` = 0.875 | local EI multiplier at that station (PCHIP of the taper chain; includes `wing_ei_taper_4`) | same bending formula about the station y = η·L, integrating only loads outboard of it |
| `J_tail_bm_limit` (1.0) | HT root, VT root | `tail_stiffness_scale` | tail/fin lift at q_D and 5° (`allowables_v2`) |
| `J_fus_bm_limit` (1.0) | aft fuselage at the wing station, vertical and lateral | `fuselage_stiffness_scale` | vertical: l_h·L_HT(q_D, 5°) + n·g·(m_fus·l_h²/2 + m_tails·l_h) with the CURRENT fuselage and tail masses; lateral: fin load × l_v |

- W_d = empty weight: a notional, genome-independent design weight. Absolute allowables are not certification values;
  only the ratios matter.
- Root + one outboard station (`tip_bm_eta` = 0.875). Softening only `wing_ei_taper_4` to 0.75 fires
  `J_wing_tip_bm_limit` (~0.02 on c172x) while root `J_wing_bm_limit` stays ~0. Further outboard coverage is still via
  the flown `J_bm_peak` / tip / twist terms and `J_smooth`.
- The in-plane design moment is set by its floor (10 % of bending) on all four aircraft, and torque/bending is 0.11 /
  0.27 / 0.10 / 0.34 (c172x / T38 / 737 / f16).
- Because demand uses the current masses, a lighter wing loses inertia relief, so bending demand rises slightly (e.g.
  `wing_gj_ratio_root` 0.8 alone gives J_wing_bm_limit ≈ 2e-6).
- Flown counterparts (full only, w = 0.1): `J_wing_torque_peak` / `J_wing_ip_peak` = hinge² of the flown peak root torque /
  in-plane moment over these allowables. They are reported in `per_scenario` as `torque_ratio` / `ip_ratio`.

Baseline allowables (lbf·ft; `v2_results/sizing_baseline.json`; every ratio is exactly 1 and every term is 0 at the
baseline):

| aircraft | wing bm | wing torque | wing in-plane | HT | VT | fus V | fus L |
|---|---|---|---|---|---|---|---|
| c172x | 15081 | 1679 | 1508 | 962 | 913 | 18673 | 6474 |
| T38 | 97013 | 25849 | 9701 | 7777 | 41129 | 181291 | 160422 |
| 737 | 1355132 | 131377 | 135513 | 195493 | 382961 | 3628970 | 1793849 |
| f16 | 270034 | 91375 | 27003 | 13257 | 46062 | 487273 | 190873 |

All three stiffness genes at 0.6 (same file):
- the sizing terms total ≈ 1.47: wing bm 0.48–0.49, torque 0.04, in-plane 0.05, tail 0.44, fuselage 0.44;
- the mass credit is only J_mass = 0.3 × (−0.097 … −0.114) ≈ −0.03.

**(e) Margin shaping.** `J = w·max(0, (1.2 − m)/0.2)²` (`margin_terms_v2`, same in Genome's `fd_bridge.precheck_v2`):
- penalty only below 1.2;
- hard fail below the gate (1.0 full, 0.9 reduced);
- exactly 0 at or above 1.2.

So there is no reward for margin above 1.2, and the 3.0 cap / not-found value is harmless (tested).

**(d) GA check** (`v2_exploit_check.py`: c172x, uniform init, same seed, pop 16 × 8 gens, 45 s scenarios, 113 evals each;
`v2_results/exploit_check_{before,after}.json`):

| | before (pre-fix code) | after |
|---|---|---|
| best tail / fus / wing_ei_root | 0.60 / 0.83 / 1.07 | 0.97 / 0.99 / 1.15 |
| population median fus scale, gen 0 → 7 | 1.13 → 0.80 | 1.13 → 1.12 |
| best mass_frac / J_mass | −0.105 / −0.031 | −0.051 / −0.015 |
| best structural cost (FD; Genome formula without sizing terms) | 0.0049 / 0.0025 | 0.022 / 0.012 |

The "before" best genome re-evaluated with the new code gets J_mass −0.020 (frac −0.066 with min gauge),
J_tail_bm_limit 0.43 and J_fus_bm_limit 0.04, so it is now strongly penalised. The short run did not reach fus < 0.65 in
either case (fraction 0.0), but the "before" population was drifting down, with tail already at the floor.

### P2.5 addendum (nsm floor + outboard BM)

- `wing_nsm_root` / `wing_nsm_tip` range is now **1.0–1.25** (was 0.8–1.25). Decode rejects values below 1.0.
- New sizing term `J_wing_tip_bm_limit` (weight 1.0) in `SIZING_TERMS` / `TERM_KEYS` (24 keys). At the baseline genome the
  term is exactly 0.
- New `model_version` strings (reduced + full change; rigid unchanged): see
  `v2_results/model_versions_post_p25.json`.

## 13. P3-A1: denser full structural model (opt-in fidelity `full_a1`)

**Select it.** `import flexeval_a1 as fa; fa.evaluate(gains, struct_genome, scenarios, model, fidelity="full_a1", root=...)`.
`fa.evaluate` with `rigid` / `reduced` / `full` just calls `flexeval.evaluate` (same code, same outputs, tested), so a
consumer can switch its import and choose A1 per call. The model on its own: `flexbody_a1.FlexBodyModelA1(model, genome)`.
Version strings: `fa.model_version("full_a1", model, root)`. The existing `full` is unchanged.
The Runner's `evolution/fidelity.py` only knows `rigid|reduced|full`. Wiring `full_a1` in is ER's to do; FD did not edit it.

**Why new files.** `flexbody.py`, `flexwing.py`, `flexeval.py` and `coupled_sim.py` are hashed into the pinned `reduced`
and `full` strings, so none of them was touched (md5 unchanged). A1 lives in `flexbody_a1.py` (model, `sizing_a1`, truncation
helpers) and `flexeval_a1.py` (fidelity, `model_version`, hook). The studies are in `p3a1_study.py` and the tests in
`test_flexbody_a1.py` (17 tests).

| | `full` (flexv2) | `full_a1` (flexv2a1) |
|---|---|---|
| wing strips = elements per semi-wing | 32 | **64** |
| wing modes per side | 3 b + 2 t + 1 ip | **4 b + 3 t + 2 ip** |
| HT / VT / aft fuselage | 12 el; 2b+1t / 2b+1t / 2+2 | identical (same arrays) |
| modal DOF | 25 | 31 |
| genes, ranges (P2.5), weights, gate 1.0, substeps 2, TERM_KEYS (24), root `<root>_v2` | | unchanged |
| `J_wing_tip_bm_limit` (η 0.875, w 1.0) | strip-discrete (`sizing_v2`) | station-exact (`sizing_a1`, below) |

- Wing EI/GJ/EIv calibration (uncoupled f_b1 / f_t1 / f_ip) and the aileron Cl_δa strip calibration are redone on the
  64-strip mesh with flexbody's own rules. With full's config (32 strips, 3b+2t+1ip), `FlexBodyModelA1` is
  bit-identical to `FlexBodyModel` (tested).
- Baseline: every sizing term is exactly 0 (all 4 aircraft) and the mass deltas are 0. The baseline margins are within
  0.01 of full (flutter) and 0.01 (reversal); e.g. T38 1.161 vs 1.162, 737 reversal 1.266 vs 1.273.
- **Strip count 64** (`v2_results/p3a1_truncation.json` → `strip_convergence_pct_vs_128`). Measured against 128 strips:
  - all metrics are within 0.08 % at 64 strips, 0.15 % at 48 and 0.37 % at 32 (the worst metric is the T38 η 0.875
    moment, error O(h));
  - the exception is the aileron-reversal margin, which is non-monotone (aileron-edge quantisation): ≤ 0.51 % at 48–96
    strips and 1.05 % at 32.
  - Strips only enter the per-genome build and margin screen: 64 strips cost +0.03–0.07 s CPU per genome over 48.
- **`sizing_a1` (the one numerical change).** `sizing_v2` uses whole strips with centre ≥ 0.875 and the EI multiplier of
  the nearest strip centre (ties go inboard). That makes `J_wing_tip_bm_limit` jump with the strip count: on c172x with
  `wing_ei_taper_4` = 0.75 it is 0.0175 / 0.0155 / 0.0096 / 0.0136 / 0.0107 at 32 / 48 / 64 / 96 / 128 strips.
  `sizing_a1` instead integrates the outboard part of each strip and interpolates the EI multiplier AT η 0.875
  (PCHIP). It converges (0.01183 → 0.01189 from 32 to 256 strips) and is still exactly 0 at the baseline. Every other
  check is `sizing_v2`'s, unchanged. The strip-discrete value stays in `sizing['tip_bm_strip_discrete']`.

  Tip-soft genome, `J_wing_tip_bm_limit`:

  | aircraft | A1 | full as flown | full model, station-exact |
  |---|---|---|---|
  | c172x | 0.0119 | 0.0175 | 0.0118 |
  | T38 | 0.0106 | 0.0067 | 0.0106 |
  | 737 | 0.0114 | 0.0073 | 0.0113 |
  | f16 | 0.0107 | 0.0068 | 0.0107 |

**Modal truncation** (`p3a1_study.py truncation` → `v2_results/p3a1_truncation.json`). The variants go from N =
4b+3t+2ip to +1 and +2 per family, +1 in every family, and +1 / +2 next-lowest modes of any type. The metrics are on the
right wing:
- root and η 0.875 bending moment, static aeroelastic (0.9 V_D, 1° + 1 g) and 1-cos gust peak;
- tip deflection;
- in-plane: 1 g fore-aft tip deflection and 1-cos n_x pulse root moment;
- flutter speed (to 4 V_D) and the gate's wing-block flutter, divergence and aileron-reversal margins.

Max |Δ| in %, every case below the 2 % threshold. Cells are baseline / tip-soft, max over the +1 and +2 variants and
over all metrics:

| aircraft | bending +1, +2 | torsion +1, +2 | in-plane +1, +2 | all families +1 | next-lowest +1, +2 | flutter speed / margin | root BM | η 0.875 BM |
|---|---|---|---|---|---|---|---|---|
| c172x | 0.01 / 0.01 | 0.10 / 0.12 | 0.16 / 0.17 | 0.16 / 0.17 | 0.10 / 0.12 | 0.001 / 0.002 | 0.003 / 0.005 | 0.10 / 0.12 |
| T38 | 0.45 / 0.47 | 0.70 / 0.70 | 0.87 / 0.93 | **1.02 / 1.03** | 0.70 / 0.70 | 0.013 / 0.017 | 0.018 / 0.019 | 1.02 / 1.03 |
| 737 | 0.05 / 0.16 | 0.33 / 0.30 | 0.59 / 0.63 | 0.59 / 0.63 | 0.33 / 0.30 | 0.005 / 0.007 | 0.006 / 0.009 | 0.33 / 0.30 |
| f16 | 0.28 / 0.30 | 0.72 / 0.79 | 0.69 / 0.74 | 0.69 / 0.74 | 0.72 / 0.79 | 0.008 / 0.010 | 0.018 / 0.019 | 0.72 / 0.79 |

- The in-plane modes only affect the in-plane metrics. The out-of-plane metrics do not see them (0.000 %), and the
  static in-plane root moment is pure force summation.
- Compared with full, A1 moves the η 0.875 elastic moment by +1.7 to +5.2 % on T38 and 737 (+0.9 % f16, −0.2 %
  c172x) and the in-plane tip deflection by −1.9 to −6.2 %. So full's 3b+2t+1ip set is not converged to 2 % on those quantities. The
  flutter and root moments agree within 0.15 %. None of full's flown cost terms uses the outboard elastic moment.

**CPU** (`p3a1_study.py bench` → `v2_results/p3a1_benchmark.json`). Same method as §8: whole `evaluate()`, baseline
genome, Phase-1 best gains, 3 × 90 s scenarios, CPU per scenario, min of 5 repeats, BLAS 1 thread, load ≈ 1.

| | c172x | T38 | 737 | f16 |
|---|---|---|---|---|
| full, s CPU / scenario | 1.66 | 1.57 | 1.54 | 1.63 |
| full_a1, s CPU / scenario | 1.77 | 1.68 | 1.68 | 1.76 |
| A1 / full | 1.07× | 1.07× | 1.09× | 1.08× |

**Version strings** (`v2_results/model_versions_post_p3a1.json`; the rigid / reduced / full entries there equal
`model_versions_post_p25.json`, which was not modified):

| aircraft | full_a1 |
|---|---|
| c172x | `full_a1:flexv2a1:36fb4f5a` |
| T38 | `full_a1:flexv2a1:f248873e` |
| 737 | `full_a1:flexv2a1:522189cb` |
| f16 | `full_a1:flexv2a1:4a9e12bc` |

The hash covers `flexbody.py`, `flexwing.py`, `flexeval.py`, `flexbody_a1.py`, `flexeval_a1.py`, the A1 parameters
(`a1_params`: v2 parameters with `wing_v2` = the A1 mesh and mode set, `variant: a1`, `a1_fmt`), terms, weights, gene
schema, substeps, gate and the `<root>_v2` aircraft files. Any edit to the A1 files changes the A1 strings only.
Editing `flexbody.py`, `flexwing.py` or `flexeval.py` changes both the full and A1 strings.

**Caveats.**
1. `TERM_KEYS` (24) and the gene schema (12 + 2 asymmetric, P2.5 ranges) are unchanged.
2. A1 costs differ slightly from full at the same genome because of the denser model. The one systematic difference is
   `J_wing_tip_bm_limit` (station-exact), so A1 and full costs are not interchangeable in one cache.
   **`J_mass` (ER smoke FYI):** A1 vs full can differ slightly for the same genome (e.g. 737 ≈ 0.000231655 vs 0.000232237)
   because the denser 64-strip mesh resolves the PCHIP EI/GJ/NSM distributions (and therefore the min-gauge mass integral)
   more finely than 32 strips. Expected mesh-density effect; A1 strings were not changed for it.
3. Higher wing modes reach 150–240 Hz at the 1/240 s Newmark substep. Average-acceleration Newmark stays stable but
   period-lengthens those modes; they respond quasi-statically. The substeps were kept at 2.
   - Check: tip-soft genome, 3 s manoeuvre + gust (`_scratch/p3a1/substep_a1.py`, 2 vs 4 substeps). Wing root BM, torque,
     tip deflection and twist peaks change ≤ 1.2 %. f16 in-plane root moment changes 2.4 %, T38 HT root moment 7.8 %.
   - That is the same pattern as `full` (0.95 % / 3.0 % / 7.8 %), so it is pre-existing and not caused by A1.
4. The structural data are still notional (§10.5).


## 14. P3-B1: planform shape genes (opt-in fidelity `full_a1_b1`)

**Select it.** `import flexeval_b1 as fb1; fb1.evaluate(gains, struct_genome, scenarios, model, fidelity="full_a1_b1", root=..., shape_genome=...)`.
`fb1.evaluate` with `rigid` / `reduced` / `full` / `full_a1` delegates to `flexeval_a1` / `flexeval` (same outputs; a non-baseline
`shape_genome` is rejected at those fidelities). The model on its own: `flexbody_b1.FlexBodyModelB1(model, struct_genome, shape_genes=...)`.
Version strings: `fb1.model_version("full_a1_b1", model, root)`. The existing `full` / `full_a1` / reduced / rigid are unchanged
(no hashed A1 / v2 source was edited). Decode / geometry gate live in `planform_b1.py` (Genome / Evolution mirror that module).

**Locked B1 shape gene list (6 genes, L↔R symmetry on).** Separate block from the 12 structure genes (P2.5 ranges). Decode
order: shape → planform strips + geometry gate → wing rebuild → structure genes on the new baseline → margins + flight.

| # | name | range | scale | default | meaning |
|---|---|---|---|---|---|
| 0 | `wing_chord_taper_1` | 0.85 – 1.05 | linear | 1.0 | chord-multiplier ratio CP1/CP0 (η 1/3 / 0); then area-renormalised |
| 1 | `wing_chord_taper_2` | 0.85 – 1.05 | linear | 1.0 | ratio CP2/CP1 (η 2/3 / 1/3) |
| 2 | `wing_chord_taper_3` | 0.85 – 1.05 | linear | 1.0 | ratio CP3/CP2 (η 1 / 2/3) |
| 3 | `wing_twist_mid_deg` | −2 – +1 | linear | 0.0 | geometric twist at η 0.5 relative to the root (deg, nose-up +) |
| 4 | `wing_twist_tip_deg` | −4 – +1 | linear | 0.0 | geometric twist at the tip relative to the root (deg; negative = washout) |
| 5 | `wing_sweep_qc_delta_deg` | −5 – +5 | linear | 0.0 | additive delta on the baseline **quarter-chord** sweep (deg), not LE |

- Chord CPs at beam η = 0, 1/3, 2/3, 1 (4 stations → root + 3 ratios, no sawtooth). Interpolation to the 64-strip mesh:
  monotone PCHIP in log space (`flexbody.pchip`), then **rescaled so the semi-wing planform area = baseline** (S and span
  stay the JSBSim values: chord genes redistribute area spanwise; they never resize the wing). Root twist is fixed at 0
  (a uniform incidence is absorbed by trim on fixed tables). Twist(η) = PCHIP through (0, 0), (0.5, mid), (1, tip).
- Encodings: named dict of physical values (missing → default) or a vector in [0,1]^6 in table order. **The [0,1] map is
  LINEAR IN VALUE for all 6 genes, chord tapers included: value = lo + u·(hi − lo), encode = (value − lo)/(hi − lo); there
  is no log mapping** (`planform_b1.GENE_ENCODING`, hashed since r1). "log" in this section only refers to the spanwise
  PCHIP of the chord multipliers (a planform rule), not to the gene scale. A GA operator may mutate in log space
  internally, but it must hand FD physical values (dict) or a linear-normalised vector. (Over 0.85–1.05 the two differ by
  at most 0.006 at u = 0.5: 0.95 linear vs 0.9447 geometric.)
  `planform_b1.decode_shape_b1` / `encode_shape_b1` / `shape_schema()` / `shape_defaults()` / `is_baseline_shape` /
  `shape_cache_key`. Raises `ValueError` (never clips): unknown keys, structure-gene names in the shape block, deferred
  keys at non-baseline values, NaN/inf, out of range, wrong vector length / entries outside [0,1].
- **Deferred (B2 or later; decode rejects non-baseline values):** `wing_dihedral_delta_deg` (geometric dihedral is not
  represented by the strip model / `node_layout` / `external_reactions` path — only elastic β·w′ exists),
  `wing_thickness_scale` / `wing_camber_scale` (section shape → B2 CST), `wing_chord_root` / `wing_span_scale` /
  `wing_area_scale` / `wing_twist_root_deg` (wing size or root incidence need rescaled JSBSim tables).

**Geometry gate** (`planform_b1.geometry_gate(pw, shape_genes, n_el=64)` → `{ok, reason, details}`). Cheap (~0.1 ms).
Rejects before the model is built / flown (status `geometry_gate:<reason>`, cost = fail_cost, not a fitness credit):
negative/tiny chord, LE/TE crossover (self-intersect), tip/root taper outside [0.12, 1.25], |sweep| > 45°, LE kink >
25°. The whole B1 gene box is feasible by construction on all 4 aircraft (tested). **The gate is a safety net only**: it
never fires for in-range genes (ER scan: 64 corners + 2000 random shapes per aircraft, closest T38 tip/root taper 0.139 vs
0.12; FD test `test_b1_planform_area_preserving_and_feasible_by_construction`). It exists for future range widening /
B2 genes / hand-built shapes; ER's reject path is exercised with an injected stricter limit.

**Physics path (strip / VLM-style increments on fixed JSBSim tables; no CFD).** Host = A1 (64 strips, 4b+3t+2ip).
1. Planform strips: shaped chord c(y), absolute quarter-chord sweep = baseline + Δ; area and span fixed.
2. AC hold: wing re-positioned in x so the Schrenk-weighted quarter-chord x equals the baseline (static margin / Cm tables
   unchanged).
3. Aero: DATCOM CLa(AR fixed, new sweep), strip lw = c·dy·a, e_c / d34 / x_θ from local c, G = cosΛ θ − sinΛ w′, Schrenk
   share from the shaped chord, Theodorsen apparent mass on c², aileron Cl_δa strip calibration redone on the shaped chords.
4. Geometry-derived structural baseline: EI, GJ, EIv = the **baseline** root constants (A1 calibration) × (c / c_root0)^3;
   mass distribution ~ c^1 with the wing **total** fixed; structure genes then multiply as today. So a tip-light planform
   is softer at the tip (frequencies move).
5. Geometric twist (root = 0): the **basic** (zero-net) strip load q κ_w lw (twist − Schrenk-share mean). Loads the
   structure (static trim shape, root / outboard BM, torque); its elastic response feeds back like any elastic increment
   (relative to the trim shape). **r1: its rigid pitch moment is NOT fed back** — it is a Cm0 shift absorbed by the trim
   elevator (r0 fed back (q κ_w − trim)·Σ(−x L_basic), an unphysical q-proportional moment; see r1 addendum). Tables fixed.
6. Sizing (same 6 `SIZING_TERMS` / weights / `TERM_KEYS` = 24): wing allowables = **baseline-planform** design loads ×
   structure gene × geometric strength (c/c0)^3 at the check station; demand = current planform design loads (Schrenk +
   basic twist at q_D, counted only where twist **raises** the demand — no sizing credit for washout relief); outboard
   check station-exact (`sizing_a1` method). Tail / fuselage checks unchanged.
7. Flown terms: `flexbody.response_terms_v2` formulas; wing torque / in-plane peak allowables from `sizing_b1`. **r1:** the
   `J_bm_rms` denominator and the `J_bm_peak` allowable use m_ref = (flown 1-g root BM − trim basic-twist BM) ×
   `bm_ref_ratio` (baseline-planform / shaped 1-g root BM per g, this genome's masses) [× (c/c0)^3 root strength factor
   for the allowable], instead of the shape's own 1-g BM.

Baseline shape (all defaults) short-circuits to the A1 constructor / `margin_terms_a1` / `FlexBodyCoupler` arithmetic →
bit-identical to `full_a1` (see acceptance). Cache key on the shape: `planform_b1.shape_cache_key(shape_genome)` (rebuild
is ~+0.05–0.09 s CPU per genome; gen-0 screening can key on it).

**API for Evolution.**

```python
import flexeval_b1 as fb1
import planform_b1 as pb1

out = fb1.evaluate(
    gains, struct_genome, scenarios, model,
    fidelity="full_a1_b1",          # only fidelity that consumes shape_genome
    root=root,                      # prepared root; B1 uses <root>_v2 like full / full_a1
    shape_genome=None | {} | {..},  # dict of physical values or [0,1]^6; None/{} = baseline
)
# out keys = full_a1 keys + shape_genes, shape_cache_key, geometry_gate, planform
# geometry_gate fail -> status "geometry_gate:<reason>", cost = fail_cost, no flight
# decode fails -> ValueError (same policy as the structure genome)

mdl = fb1.build_model(struct_genome, model, shape_genome=..., root_v2=...)
# or: flexbody_b1.FlexBodyModelB1(model, struct_genome, shape_genes=...)
gate = pb1.geometry_gate(pw, shape_genome, n_el=64)   # {ok, reason, details}
```

**model_version scheme** (`v2_results/model_versions_post_p3b1.json`):

| fidelity | string | pin file |
|---|---|---|
| `rigid` / `reduced` / `full` | unchanged (= `model_versions_post_p25.json`) | listed for reference |
| `full_a1` | unchanged (= `model_versions_post_p3a1.json`) | listed for reference |
| `full_a1_b1` | `full_a1_b1:flexv2b1:<sha8>` | **new** |

Hash covers: `flexbody.py`, `flexwing.py`, `flexeval.py`, `flexbody_a1.py`, `flexeval_a1.py`, `planform_b1.py`,
`flexbody_b1.py`, `flexeval_b1.py`, A1/B1 params (incl. frozen shape schema / rules), terms, weights, struct + shape gene
schemas, substeps, gate, `<root>_v2` aircraft files. Per-genome shape / structure values are inputs, not part of the
version. Editing any of those modules changes the B1 strings; editing `flexbody.py` / `flexwing.py` / `flexeval.py` also
changes full / reduced / A1; the A1 modules alone change A1 + B1.

Pins (this box, after B1 land):

| aircraft | full_a1_b1 |
|---|---|
| c172x | `full_a1_b1:flexv2b1:3e40908a` |
| T38 | `full_a1_b1:flexv2b1:982bce54` |
| 737 | `full_a1_b1:flexv2b1:1bc748ac` |
| f16 | `full_a1_b1:flexv2b1:bfb25718` |

**Acceptance** (`v2_results/p3b1_acceptance.json`). Baseline shape (None / {} / all defaults) + structure = full_a1 bit-intent:

| check | result |
|---|---|
| A: whole `evaluate` (4 aircraft × 2 structure genomes × 3 Phase-1 90 s scenarios × 3 baseline-shape encodings), cost / 24 terms / margins / mass / sizing / per-scenario physics / loads / every coupler history channel | **exact** (max \|Δ\| = 0.0, 24 cases) |
| B: every `FlexBodyModel` matrix of `FlexBodyModelB1(shape={})` vs `FlexBodyModelA1` | **exact** (8 cases) |
| C: rebuild path at 1e-9 shape perturbation (continuity) | margins ≤ 6.25e-09, terms ≤ 6.73e-10, matrices rel ≤ 1.32e-08, 90 s cost ≤ 6.57e-10 (eps 1e-6) |

Different by design: `model_version` / `fidelity` / `margins_fidelity` strings, `per_scenario[*].wall_s`, and the B1-only keys
above. Platform note: compare A1 vs B1-baseline in the **same process** (exact); another BLAS / thread count is outside
the pin.

**CPU** (`v2_results/p3b1_benchmark.json`; process CPU, min of 3, BLAS 1 thread, Phase-1 best gains, one 90 s scenario;
shaped genome = chord tapers 0.95 / twist mid −0.5 tip −2 / sweep +2°):

| | c172x | T38 | 737 | f16 |
|---|---|---|---|---|
| A1 build, s | 0.087 | 0.094 | 0.051 | 0.095 |
| B1 shaped rebuild, s | 0.172 | 0.181 | 0.098 | 0.180 |
| rebuild overhead, s | 0.085 | 0.087 | 0.047 | 0.085 |
| eval 90 s full_a1, s | 1.875 | 1.779 | 1.690 | 1.850 |
| eval 90 s B1 baseline, s | 1.897 | 1.778 | 1.704 | 1.869 |
| eval 90 s B1 shaped, s | 1.978 | 1.947 | 1.847 | 1.995 |
| shaped / A1 | 1.055× | 1.094× | 1.093× | 1.079× |

Shape rebuild is cheap enough for gen-0 screening; key the rebuilt model / margins on `shape_cache_key` if the same shape
is re-evaluated with different controllers.

**Caveats / open for Genome & ER.**
1. `TERM_KEYS` (24) and the 12 structure gene ranges (P2.5) are unchanged. No new `J_*` terms.
2. Wing size (chord_root / area / span) is deferred: would need rescaled JSBSim tables. Chord genes only redistribute.
3. Geometric dihedral and thickness/camber deferred (see above).
4. ~~Node telemetry still lays out the baseline planform~~ — r1: `flexbody_b1.node_layout_b1(mdl, rp)` (=
   `FlexBodyModelB1.node_layout`, `flexeval_b1.node_layout`, used by `FlexHookB1.node_telemetry`) follows the shaped wing;
   see r1 addendum. `flexbody.node_layout` itself (hashed) still gives the baseline-chord layout.
5. FDM point masses carry only the gene Δmass (planform area is fixed, so wing mass total is fixed); strip mass
   redistribution does not move the FDM wing point-mass Y (centroid shift from chord reshape is not modelled — documented
   under `planform_b1.shape_params_for_hash` rules).
6. Tests: `test_flexbody_b1.py` (extend, don't break the prior 123). Files: `planform_b1.py`, `flexbody_b1.py`,
   `flexeval_b1.py`, `p3b1_study.py`, `test_flexbody_b1.py`, `v2_results/model_versions_post_p3b1.json`,
   `v2_results/p3b1_acceptance.json`, `v2_results/p3b1_benchmark.json`.

### P3-B1 r1 addendum (ER follow-up: T38 mid wash-in, node layout, encoding) — new strings, r0 pins left intact

**Q1 verdict: the T38 drift of `wing_twist_mid_deg` to +1 was a (small) model loophole, not the source of the cost drop.**
Study `v2_results/p3b1r1_twistmid_study.json` (`_scratch/p3b1/twist_mid_study.py`, checkpoints
`p3b1r1_twistmid.partial.jsonl`): T38:g4:r0 of `phase3b1-smoke-s1` re-flown (reproduces 0.15620043 bit for bit), twist_mid
swept −2…+3 (values > 1 via a study-only range bypass).
- **Where 0.2262 → 0.1562 came from: controller gains.** 0.2262 is the *phase3a1* run's T38 best (other GA run, same
  scenarios / profile). B1 gains + A1-run structure + baseline shape = 0.1626; B1 best with baseline shape = 0.1571; A1
  best + B1 shape = 0.2339 (worse). The track gain (s0 0.138 → 0.032, s2 0.156 → 0.102) is kd_pitch 0.165 → 0.0082,
  kd_alt 0.92 → 0.36, ki_hdg 0.145 → 1e-4.
- **r0 twist_mid effect on that genome:** cost 0.156798 (0) → 0.156200 (+1) → 0.155127 (+3): −0.0006 / deg (−0.38 %),
  monotone through the ceiling; s1 track −0.6 % / deg, J_bm_rms −1.6 % / deg; trim α / elevator unchanged
  (4.6188°, no free trim, no CL0/Cm0 double count at t0). Below 0 sizing penalises it (pre-flight 0.005 → 0.035 at −2).
- **Kill-switch decomposition:** rigid pitch feedback off → track flat (s1 0.093631 → 0.093615 → 0.093582 for 0/1/3);
  elastic twist loading (`twist_Q`) off → no change; rigid root loads (`twist_RB`) off → the J_bm_rms part goes. Two
  mechanisms: (a) `(q κ_w − trim)·twist_pitch` = a pitch moment ∝ Δq — on a trimmed aircraft the basic-twist moment is a
  Cm0 shift that the trim elevator cancels *at every q* (both ∝ q), so ∂M/∂q|α,δe = 0; r0's constant offset left
  ∂M/∂q = twist_pitch (pseudo speed-stability term; T38 twist_pitch −1.46 lbf·ft/psf at twist_mid +1, i.e. ΔCm0 ≈
  −0.0011, rms 3.7 lbf·ft vs ~1.5e5 lbf·ft/rad pitch stiffness — tiny but systematic). (b) `J_bm_rms = rms/m_1g` with
  m_1g including the wash-in basic-load BM (+337 lbf·ft at +1): the shape inflated its own reference (rms itself rose).
- **Hand calc** (`_scratch/p3b1/twist_handcalc.py`): Prandtl lifting line on the shaped T38 planform gives a basic-load
  ΔCm0 of −0.0007 per +1° mid (strip theory −0.0011, ~1.6× LL — strip theory over-concentrates the basic load), and a
  zero-net load cannot change trim α. An honest effect on tracking is ≈ 0; a 2× track gain is impossible.
  Aileron effectiveness: unchanged by twist (strip Cl_δa calibration depends on chord only). FDM wing point-mass Y
  centroid: unaffected by twist (mass ∝ c).
- **Fix (r1):** (a) rigid basic-twist pitch feedback removed (twist still loads the structure; the elastic response and
  absolute root loads stay); (b) flown wing-BM reference anchored as rule 7. **After:** twist_mid 0 / +1 / +3 → cost
  0.157157 / 0.157143 / 0.157115 (−1.4e-5 / deg, 43× smaller, ≈ neutral; residual = physical elastic coupling), s1 track
  0.093631 / 0.093615 / 0.093582. Same genome at r1: 0.157143 (r0 0.156200). Remaining known gap: mid wash-in has no
  induced-drag / stall-margin cost (rigid tables own drag; no new J_* allowed), so in [0, +1] it is now ≈ cost-neutral and
  may drift freely — treat its value as uninformative (or narrow the Genome range) until B2.
- Regression tests: `test_b1r1_twist_pitch_not_fed_back` (twist_pitch × 1e3 → flight bit-identical) and
  `test_b1r1_t38_mid_washin_is_near_neutral` (T38 best, ER scenario s1: |Δcost(0 → +1)| < 1e-4; r0 gave 8.8e-4); both
  fail on the r0 code (checked), pass on r1. `test_b1r1_flown_bm_reference_not_inflated_by_shape`.

**Q2: node layout follows the shaped wing (implemented, cheap, display-only).** `node_layout_b1`: EA node x = −[(y −
y_mac)·tan Λ_qc,shaped + ac_shift + (x_ea − 0.25)·c_shaped(y)] (same geometry as the strip arms / e_c; c_shaped from the
same log-PCHIP + area-norm law, at the 65 node stations), y / z unchanged; twist rotates sections about the EA, so the
axis does not move — it is exported per wing node as `geometric_twist_deg`, with `chord_ft` and the twisted
`le_nodes_body_ft` / `te_nodes_body_ft` (LE = EA + x_ea·c·(cos θ, 0, −sin θ), TE = EA − (1 − x_ea)·c·(cos θ, 0, −sin θ),
body FRD, same origin). Extra keys only for shaped wings; other bodies unchanged; baseline shape → `flexbody.node_layout`
lists exactly (tested). Sim Bridge `geometry_from_layout` reads only `axis_nodes_body_ft` / `node_span_frac`, so the new
keys are ignored unless used. **ER / SB call-site change needed:** they call `flexbody.node_layout(obj)` directly — switch
to `flexbody_b1.node_layout_b1(obj, rp)` (or `obj.node_layout(rp)`) for full_a1_b1.

**1.80° vs 1.68°:** pure geometry, not a bug (same in r0 and r1). The exported axis is the elastic axis at x_ea = 0.40
of a taper-0.2 chord, i.e. aft = y·tan Λ_qc + 0.15·c(y): on the T38 it is swept 18.70°, not 24°. Its angle is
atan(tan Λ_qc − k), k = 0.15·c_root(1 − λ)/s, so dΛ_ea/dΛ_qc = sec²Λ_qc / (1 + tan²Λ_ea) = 1.198/1.114 = 1.075:
+1.68° quarter-chord → +1.81° EA line (test `test_b1r1_node_ea_slope_vs_quarter_chord_sweep`). The uniform AC-hold shift
translates the axis and doesn't change its slope.

**Strings (`v2_results/model_versions_post_p3b1r1.json`; `planform_b1.B1_REV = 1`, hashed; tag / fidelity unchanged):**

| aircraft | full_a1_b1 r1 | (r0, `model_versions_post_p3b1.json`, left intact) |
|---|---|---|
| c172x | `full_a1_b1:flexv2b1:56ee798e` | `3e40908a` |
| T38 | `full_a1_b1:flexv2b1:7e871977` | `982bce54` |
| 737 | `full_a1_b1:flexv2b1:6523753c` | `1bc748ac` |
| f16 | `full_a1_b1:flexv2b1:617078a9` | `bfb25718` |

rigid / reduced / full / full_a1 strings unchanged (= post_p25 / post_p3a1); no hashed A1 / v2 source edited (MD5s
checked). r0 cache rows / runs are not comparable with r1 for non-baseline shapes; baseline-shape results are
identical (acceptance below).

**r1 acceptance** (`v2_results/p3b1r1_acceptance.json`): re-run on the r1 code, all pass —
A whole-`evaluate` baseline ≡ full_a1 **exact** (24 cases, max |Δ| = 0.0); B matrices **exact** (8); C continuity at 1e-9
(margins ≤ 6.25e-09, terms ≤ 7.4e-10, matrices rel ≤ 1.32e-08, 90 s cost ≤ 6.6e-10). Log `p3b1r1_acceptance_run.log`.
CPU: r1 removes one multiply-add per step from the shaped coupler; `p3b1_benchmark.json` (r0) stands.

**Tests:** `test_flexbody_b1.py` 44 cases (r0 36 + 8 r1: old-pin-intact, encoding, twist-pitch regression, T38 wash-in
regression, BM reference, node layout × 2, EA slope). Full suite (`test_flexwing.py test_flexbody.py test_flexbody_a1.py
test_flexbody_b1.py`) **167 passed** (`v2_results/pytest_p3b1r1.log`).

r1 files: `planform_b1.py` (B1_REV, GENE_ENCODING), `flexbody_b1.py` (coupler, `bm_ref_ratio`, `wing_bm_reference_b1`,
`node_layout_b1`), `flexeval_b1.py` (FlexHookB1 qk_ref / node telemetry, `node_layout`), `p3b1_study.py` (rev-aware outputs;
refuses to overwrite the r0 pin file), `test_flexbody_b1.py`, `v2_results/model_versions_post_p3b1r1.json`,
`p3b1r1_acceptance.json`, `p3b1r1_twistmid_study.json`. r0 copies: `_scratch/p3b1/*.r0.py`.
