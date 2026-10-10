# Phase 4 (ring course): FD section, control-surface model (`full_a1_b2a_cs`)

Status: IMPLEMENTED (spec 02:20 PT, implementation 02:35 PT, Oct 7 2026). Sim Bridge's `flight_sim_3d/PHASE4_RINGS_SPEC.md` skeleton did not
exist anywhere on the box when this was written, so this file is the FD section. Paste or link it verbatim as
the FD section once the skeleton exists. Implementation status and measured numbers are in §P4.9.

Code (all NEW files; nothing frozen is touched):
`ctrlsurf_p4.py` (surface tables, actuator, hinge/flex model, signal recorder), `flexeval_p4.py` (fidelity wrapper),
`p4_limits.py` (writes `v2_results/p4_aircraft_limits.json`), `test_ctrlsurf_p4.py`.
Pins: `v2_results/model_versions_post_p4cs.json`. Frozen A1 / B1 r1 / B2a files and their md5 lists are unchanged.

## P4.1 Fidelity and API

```python
import flexeval_p4 as fp4
r = fp4.evaluate(gains, struct_genome, scenarios, model, fidelity='full_a1_b2a_cs', root=ROOT,
                 shape_genome={...B1+B2a...}, cs_mode='active',      # 'active' | 'passthrough'
                 cs_overrides=None, record=False)
# any other fidelity -> flexeval_b2.evaluate unchanged (pure delegation)
fp4.surface_table(model)          # dict: P4.3 table for one aircraft (degrees, deg/s, s)
fp4.model_version('full_a1_b2a_cs', model, root)
```
* `cs_mode='passthrough'`: the actuator writes the controller's command unchanged, and the hinge/flex diagnostics are
  computed read-only. Every pre-existing output field equals `full_a1_b2a` **bit for bit** (tested). The `ctrl_surfaces`
  block is still added.
* `cs_mode='active'` (default for P4 runs): FD's actuator sits between the controller and JSBSim (P4.4). The JSBSim FCS still runs
  behind it, so the JSBSim clip/rate/kinematic elements remain the hard limits.
* The flex coupling of control loads is the existing B2a two-way coupler. Its strip loads read the JSBSim *surface positions*
  (`fcs/*-pos-*`), so the FD actuator output is automatically the load the structure sees. **No extra effectiveness
  multiplier is applied to the aero in flight.** The elastic roll/pitch/yaw increments already come back through
  `external_reactions`. Scaling the commands by η as well would double-count, and an η computed from a rigid reference
  would reopen the r1-style rigid-feedback loophole. η is exported as a diagnostic and limit (P4.5).

## P4.2 Sign conventions
* Commands: JSBSim `fcs/elevator-cmd-norm`, `fcs/aileron-cmd-norm`, `fcs/rudder-cmd-norm`, `fcs/flap-cmd-norm`,
  `fcs/speedbrake-cmd-norm`, range [-1, 1] ([0, 1] for flap and speedbrake).
* JSBSim sign (all four models, probed in p4_limits.py): elevator + cmd = trailing edge down = nose-down pitch
  (the controller writes `elev = elev_trim - u_p`). Aileron + cmd = **right roll (+p, right wing down)** on c172x, T38, 737
  and f16 (`roll_sign_p_per_plus_ail` = +1 in p4_aircraft_limits.json). Rudder + cmd = JSBSim + rudder (trailing edge left,
  nose-left yaw). Ring controllers must not assume otherwise.
* Exported deflections `delta_*_deg` use the **JSBSim sign** (same sign as the command norm), in degrees at the surface.
  Per-side ailerons: `delta_ail_L_deg` / `delta_ail_R_deg` = JSBSim left/right pos (trailing edge down +).
* Hinge moment `H_*_lbft`: + = moment tending to deflect the surface toward + δ (i.e. aiding the command when same sign).
  Body-frame loads are body FRD (x fwd, y right, z down), consistent with INTERFACE_v2 §11.

## P4.3 Surfaces and limits per aircraft
Source column: J = read from the aircraft's JSBSim XML (jsbsim_root_v2, = shipped JSBSim 1.3.1 models + FD prep);
L = literature (Stevens & Lewis 2003, F-16 model: δe ±25°, 60°/s; δa ±21.5°, 80°/s; δr ±30°, 120°/s; τ = 1/20.2 s);
N = notional (plausible engineering default, labelled). δ ranges are the deflection at command ±1 (asymmetric where JSBSim is).

| aircraft | surface | δ range at cmd -1 / +1 (deg) | src | rate limit (deg/s) | src | τ lag (s) | src | JSBSim position property |
|---|---|---|---|---|---|---|---|---|
| c172x | elevator | -28 / +23 (clipped ±19.5 by actuator) | J | 60 | N | 0.0167 (JSBSim `<lag>60`) + 0.05 FD | J/N | fcs/elevator-pos-rad |
| c172x | aileron L/R | -20 / +15 (clip -20.1/+14.9) | J | 90 (JSBSim 1.57 rad/s) | J | 0.05 | N | fcs/left/right-aileron-pos-rad |
| c172x | rudder | ±16 | J | 60 | N | 0.05 | N | fcs/rudder-pos-rad |
| c172x | flaps | 0 … 30 (kinematic 0-10 2 s, 10-20 1 s, 20-30 1 s) | J | 5–10 | J | – | – | fcs/flap-pos-deg |
| T38 | elevator | -1 … +0.583 norm × 17 deg/norm = -17 / +9.9 | J(norm)/N(deg) | 40 | N | 0.05 | N | fcs/elevator-pos-norm |
| T38 | aileron L/R | norm L -1/+0.75, R -1/+0.65 × 20 deg/norm (flexwing assumption) | J/N | 60 | N | 0.05 | N | fcs/left/right-aileron-pos-norm |
| T38 | rudder | ±1 norm × 30 deg/norm | J/N | 60 | N | 0.05 | N | fcs/rudder-pos-norm |
| T38 | flaps | 0 … 40 (24° in 5 s, 40° 3 s more) | J | ~5 | J | – | – | fcs/flap-pos-deg |
| T38 | speedbrake | 0 … 1 norm in 1 s | J | 1 /s (norm) | J | – | – | fcs/speedbrake-pos-norm |
| 737 | elevator | ±17.2 (±0.3 rad) | J | 40 | N | 0.08 | N | fcs/elevator-pos-rad |
| 737 | aileron L/R | ±20.1 (±0.35 rad) | J | 45 | N | 0.08 | N | fcs/left/right-aileron-pos-rad |
| 737 | rudder | ±20.1 (±0.35 rad) | J | 40 | N | 0.08 | N | fcs/rudder-pos-rad |
| 737 | flaps | 0 … 1 norm (kinematic, 22 s full) | J | – | J | – | – | fcs/flap-pos-norm |
| 737 | speedbrake (flight spoilers) | 0 … 1 norm in 0.6 s | J | 1.67 /s (norm) | J | – | – | fcs/speedbrake-pos-norm |
| f16 | elevator (stabilator, symmetric) | ±25 (±0.436 rad) | J | 60 | L | 0.0495 | L | fcs/elevator-pos-rad |
| f16 | aileron (flaperon + diff. tail, aero uses fcs/aileron-pos-rad) | ±21.5 (±0.375 rad) | J | 80 | L | 0.0495 | L | fcs/aileron-pos-rad |
| f16 | rudder | ±30 (±0.524 rad) | J | 120 | L | 0.0495 | L | fcs/rudder-pos-rad |
| f16 | flaps (TEF/LEF, FCS-scheduled) | FCS-owned, not commanded by FD | J | – | – | – | – | – |
| f16 | speedbrake | 0 … 43° (FCS-limited), FCS-scheduled | J | – | J | – | – | – |

The JSBSim FCS elements stay in the loop (c172x elevator lag 60/s and aileron rate 1.57 rad/s; f16 kinematics 0.3 s / 0.4 s
full travel). The FD actuator is the *binding* limit wherever it is slower. Flaps and speedbrake are pass-through by
default: the ring controller may command them, and FD exports their positions and saturation, but no FD lag is added.

## P4.4 Actuator model (per surface, discrete, deterministic, dt = sim.DT = 1/120 s)
Normalised space u ∈ [-1, 1] (JSBSim command space), degrees via the per-surface table (piecewise linear, asymmetric).
```
u_c   = clip(cmd, -1, 1)
y_lag = y + (1 - exp(-dt/τ)) (u_c - y)              # first-order lag, exact ZOH
Δmax  = rate_limit_norm_per_s * dt                  # rate_limit_deg / (deg per norm on the side moved toward)
y'    = y + clip(y_lag - y, -Δmax, +Δmax)           # rate limit
y'    = clip(y', -u_hinge_max, +u_hinge_max)        # hinge-moment (blow-down) limit, P4.5, only if |H| would exceed H_max
write fcs/<surface>-cmd-norm = y'  -> JSBSim FCS -> fcs/<surface>-pos-*
```
Initial state = trimmed command (no start-up transient). Saturation flags (per step, bool):
`pos_sat` (|u_c| ≥ 1 or y' at the hinge limit), `rate_sat` (rate clip active), `hinge_sat` (hinge limit active).

## P4.5 Hinge moments and flex coupling
**Hinge moment** (Chα/Chδ, per surface):
`H = q · S_c · c_c · (Chα · α_s + Chδ · δ)` [lbf·ft], with
* α_s: local angle of attack at the surface. Wing ailerons: α + elastic twist at the aileron mid-station
  (`tip_twist_{L,R}_deg` × η_mid of the aileron span, linear-twist approximation). HT: α(1-dε/dα) + `ht_incidence_deg`.
  VT: β + `vt_sideslip_deg`.
* Chα = -0.10 /rad, Chδ = -0.35 /rad (N; typical plain sealed surface with partial aerodynamic balance, ESDU 89009 ballpark).
* S_c, c_c: aileron S_c = 0.12·S_w·(η1-η0) per side and c_c = 0.25 c̄(η_mid); elevator 0.35 S_h, 0.35 c̄_h; rudder 0.30 S_v, 0.30 c̄_v
  (N). f16 stabilator: all-moving, so H is about the pivot with Chδ = -0.05 (N, near-balanced).
* H_max per surface = 1.25 × |H| at q_D (= q at V_D from flexwing profiles), δ = full deflection, α_s = 0. This is a fixed per-aircraft
  constant, not gene-dependent. If |H(δ_cmd)| > H_max, the deflection is limited to the δ where |H| = H_max (blow-down).
  This only binds near or above V_D, so it is rarely active in the course envelope. Exported regardless.

**Effective (flex-reduced) control power**: η_c(q) = elastic/rigid control effectiveness from the SAME B2a model and the
SAME solver as the margin screen (`flexbody.BlockAero.effectiveness`, blocks wingR/aileron, empennage_pitch/elevator,
empennage_yaw/rudder). It is tabulated once per genome on a q grid (0 … 1.2 q_D, 48 points, log-spaced) and interpolated
per step at `aero/qbar-psf`. η < 0 means reversed. Export: `eta_ail`, `eta_elev`, `eta_rud`, and the control power
`Lda_eff = η_ail · Cl_δa(rigid) · q S b` (lbf·ft/rad) etc. This is a **diagnostic**: the aero is not multiplied by it (see P4.1).
The flown elastic effect is already in the coupler feedback. Reversal-trend hand calc: with a uniform torsion wing,
η(q) ≈ (1 - q/q_R)/(1 - q/q_D) (2-D typical section), and the test checks that the tabulated η decreases monotonically in q and
crosses 0 within ±15 % of the margin-screen `q_reversal` when that is below 1.2 q_D.

**Control-surface loads into the coupler**: unchanged B2a mechanism. Aileron strip loads (`q_dail_R/L`), elevator (`q_delev`
or the all-moving `qkh_alpha_ht`) and rudder (`q_drud`) load bases are driven from the JSBSim position properties above. The FD
actuator therefore changes the structural load history only through real surface motion. Hinge moments are NOT added as
extra torsion load (they are reacted inside the surface/actuator, and the design torque already has a q_D·δ_d hinge part).

No gene-dependent denominators: all normalisations use fixed per-aircraft constants (δ_max, rate_max, H_max, q_D).

## P4.6 Per-step signals (export)
`result["ctrl_surfaces"]` (always) = `{"schema": "fd-ctrlsurf/1", "mode", "sample_hz": 120, "surfaces": [...], "summary": {...}}`;
with `record=True`, `telemetry[i]["ctrl_surfaces"]` = per-step arrays at **120 Hz** (every JSBSim step, aligned with the
`structure` histories; Sim Bridge may decimate to 30 Hz by taking every 4th sample, t_k = (k+1)·dt after the step).

Per surface `s` ∈ {elev, ail, rud, flap, sbrk} (flap/sbrk only where present), arrays:
| key | unit | meaning |
|---|---|---|
| `t_s` | s | time after the step |
| `{s}_cmd_norm` | – | controller command (before the FD actuator) |
| `{s}_act_norm` | – | FD actuator output written to JSBSim |
| `{s}_cmd_deg` | deg | command mapped to degrees (JSBSim sign) |
| `{s}_deg` | deg | actual JSBSim surface position (ail: antisymmetric (L - R)/2 for two-property models; f16: fcs/aileron-pos-rad; per side in `ail_L_deg`, `ail_R_deg`) |
| `{s}_rate_dps` | deg/s | backward-difference rate of `{s}_deg` |
| `{s}_pos_sat`, `{s}_rate_sat`, `{s}_hinge_sat` | 0/1 | saturation flags |
| `{s}_hinge_lbft` | lbf·ft | hinge moment (aileron: R side; `ail_L_hinge_lbft` too) |
| `{s}_eta` | – | flex effectiveness η (elev, ail, rud) |
| `{s}_power_eff` | lbf·ft/deg | effective moment per degree = η × rigid control derivative × qS(b or c̄) |

Summary (per scenario and mean) = suggested **scoring signals** (Evolution owns the weights):
`use_tv_deg_per_s[s]` = ∫|δ̇|dt / T (deg/s), `use_rms_deg[s]`, `sat_frac_pos[s]`, `sat_frac_rate[s]`, `sat_frac_hinge[s]`,
`hinge_peak_frac[s]` = max|H|/H_max, `eta_min[s]` (min η flown), `reversal_flown` (any η ≤ 0 while that surface moved),
`lag_rms_deg[s]` = rms(cmd_deg - deg). Recommended normalisers (fixed per aircraft): δ_max and rate_max from P4.3.

## P4.7 Aircraft limits for course scaling: `v2_results/p4_aircraft_limits.json`
Per aircraft, at the Phase-1 trim point (h0, KCAS from configs/phase1.json) on `<root>_v2` (baseline, B2 defaults), rigid:
`trim` {h_ft, kcas, ktas, vt_fps, alpha_deg, throttle, weight_lb};
`n_struct_limit` (flexwing n_limit), `n_profile` (profile nz_limits), `n_aero_max` = q S CLmax0/W (planform_b2 SECTIONS);
`n_inst = min(n_struct_limit, n_aero_max)`; `R_inst_ft = V²/(g√(n_inst²-1))`;
`n_sus` from T_max = D0 + D_i n² (T_max = thrust at throttle_max measured in JSBSim at trim, D split with an induced-drag
estimate CD_i = CL²/(π e AR), e = 0.8 N), `R_sus_ft`, `bank_sus_deg = acos(1/n_sus)`;
`bank_max_deg` = min(profile max_abs_phi (45, a sim fail limit), acos(1/n_inst));
`roll_rate_max_dps` (simulated: full aileron step from trim, active FD actuator, peak |p| in 3 s, rigid);
`pitch_rate_max_dps` = g(n_inst-1)/V (g-limited) and the simulated 1 s elevator-step peak;
`climb_fpm_max` = (T_max - D)V/W; `descent_fpm_idle` = D V/W at idle thrust, `descent_fpm_speedbrake` (where present);
`ring_hint`: min ring spacing ≈ V·(t_roll(60°)+t_settle) and min lateral offset curvature radius = 1.5 R_sus (suggested).

## P4.8 Genome Q&A (for genome/PHASE4_RINGS_GENOME.md, 29 genes)
1. **Live surfaces.** Elevator, aileron and rudder are live on all four aircraft. The JSBSim FCS maps `fcs/rudder-cmd-norm` to a
   rudder position used by Cn/Cy on c172x, T38, 737 and f16. T38 and 737 add yaw-damper summers. The f16 FCS adds its own
   rudder/aileron interconnect and pid paths, and its aileron is flaperon + differential tail lumped into `fcs/aileron-pos-rad`.
   Flaps: c172x (0–30°), T38 (0–40°), 737 (norm) are kinematic and slow (seconds), so they are not useful for ring steering. On the f16
   the TEF/LEF are FCS-scheduled and not commandable. Speedbrake: T38 (1 s), 737 flight spoilers (0.6 s), f16 (FCS-scheduled,
   43°). c172x has none. The genome drives only aileron/elevator/rudder/throttle, which is fine. Flap and speedbrake are optional
   pass-through channels.
2. **Bank and nz limits** (p4_aircraft_limits.json; the sim fails at |φ| > 45° on every Phase-1 profile, `max_abs_phi_deg`):
   | aircraft | nz profile limits | n_struct | n_inst @ trim (aero/struct) | bank for n_inst | sustained bank (est) |
   |---|---|---|---|---|---|
   | c172x | −1 / 3.8 | 3.8 | 3.56 | 73.7° | 59° |
   | T38 | −3 / 7.33 | 7.33 | 4.20 (CLmax-limited at 300 KCAS) | 76.2° | 67° |
   | 737 | −1 / 2.5 | 2.5 | 2.5 | 66.4° | 66° |
   | f16 | −3 / 9.0 | 9.0 | 8.79 | 83.5° | 78° |
   The genome's nz −0.5…4 g fits inside every profile except the **737 (cap nz_max at 2.5)** and c172x (3.56 at trim speed, so clamp
   to `n_inst`). Bank 15–75° is above the 45° sim fail limit on all four. Either Evolution raises `max_abs_phi_deg` for the
   ring profile (recommended: c172x 60, T38 70, 737 60 (pax-style 30–45 if comfort is scored), f16 80), or bank_max is
   clamped to 45. Clamp genes to `min(gene, file limit)` per aircraft and do not re-range per aircraft.
3. **Fidelity** `full_a1_b2a_cs` (flexeval_p4.evaluate, cs_mode 'active' for rings, 'passthrough' = B2a bit-identical).
   Pins: `v2_results/model_versions_post_p4cs.json` (per aircraft: active, passthrough, and the underlying b2a string, which equals
   model_versions_post_p3b2a.json). Version format `full_a1_b2a_cs:p4cs0:<mode>:<sha8>`.
4. **Rates and lag**: see the P4.3 table (c172x 60/90/60 deg/s, τ 0.05; T38 40/60/60, τ 0.05; 737 40/45/40, τ 0.08; f16 60/80/120,
   τ 0.0495, elev/ail/rud). The FD actuator already applies the physical lag and rate. `tau_cmd_s` is therefore **not redundant
   if read as a pilot/guidance prefilter**, but its lower end (0.02–0.05 s) is below or equal to the actuator τ and does nothing.
   Recommendation: keep it as a command-shaping gene and **re-range it to 0.05–0.5 s (log)** with default 0.1. Do not use it to
   emulate actuator lag. Surface-use scoring reads the FD actual deflection, so a fast prefilter cannot hide rate saturation.
5. **Gain rescaling** (unit-free genes): scale the inner-loop surface gains per aircraft by fixed constants, never genome-dependent ones:
   * roll: δa_norm per (deg/s of p error) × K_p,ref with K_p,ref = 1/(p_max_dps) using `roll_rate_max_dps` from the limits file
     (full aileron → p_max). Equivalent to normalising by Cl_δa q̄ S b / (Ixx-rate damping) but measured, so it includes JSBSim tables.
   * pitch: δe_norm per g of nz error × 1/(n_inst − 1) (or per deg/s q × 1/pitch_rate_g_limited_dps).
   * yaw/turn coordination: r_cmd = g tanφ / V already carries units, and the rudder gain is scaled by the aileron ratio
     (k_ari unit-free as is).
   * Use the trim q̄ (fixed per profile), not the instantaneous q̄ or η. Gain-scheduling on q̄ is allowed as `× q̄_trim/q̄`, a
     fixed formula, but **do not divide by the flex η**: that rewards soft wings (reversal loophole, as in r1).
   Put the constants in genome profiles.py tags from p4_aircraft_limits.json (`roll_rate_max_dps`, `n_inst`,
   `pitch_rate_g_limited_dps`, `trim.vt_fps`).

## P4.9 `fly_course` API (for ER, evolution/analysis/PHASE4_SCORING_PROPOSAL.md §8)
```python
r = fp4.fly_course(profile, gains, struct, shape, surfaces_genome, course, fidelity='full_a1_b2a_cs',
                   model='c172x', guidance=callable, cs_mode='active', record_hz=30.0)
# 'full_a1_b2a_p4' is accepted as an alias; fidelity='full_a1_b2a' = same plant, surfaces pass-through
```
* `guidance(state, gains, course, model) -> {'aileron','elevator','rudder','throttle'[, 'flap','speedbrake']}` command norms,
  called every 1/120 s step before the FD actuator. `state` = {t, pos_m [x N, y E, z up], phi, theta, psi (rad), p, q, r (rad/s),
  vc_kts, vt_fps, nz, alpha, beta (rad), agl_m, h_dot_fps}. `fp4.demo_guidance` is an FD test law only.
* `profile`: Profile, dict (`Profile.from_dict`) or None (Phase-1 default). `struct` = 12-gene struct dict. `shape` = B1+B2a dict or
  None. `surfaces_genome` = per-surface overrides of P4.3 (e.g. `{'ail': {'rate_dps': 60}}`) or None (FD defaults; recommended,
  since actuator limits are plant, not genes). `course` = {'start': {'alt_ft','kcas'}, 'duration_s', + rings (passed to guidance)}.
  Start heading is 0 (north). Rotate ring coordinates into this frame (origin = start point).
* Returns, at `record_hz`: `t`, `pos` [N,3] m, `att` [N,3] rad, `v_kcas`, `v_ms` (true), `nz`, `alpha` rad, `agl` m,
  `surfaces` {name: deg[N]}, `surfaces_120hz` (all P4.6 keys), `surface_limits` {name: (min_deg, max_deg, rate_max_dps or None)},
  `ctrl_summary` (P4.6 scoring signals), `status` ∈ {ok, ground, diverged, structural_failure, geometry_gate, margin_fail, trim_failed},
  `struct_failed`, `t_end`, `terms` (the 24 TERM_KEYS: pre-flight margins/mass/sizing + flown structural terms; track/effort = 0,
  since ER scores rings), `margins`, `energy` (fd-energy/2), `limits`, `model_version`.
  structural_failure = |nz| > 1.5 n_limit (ultimate) or the B2a structural-ultimate fail flag. There is no sim attitude/45° fail in fly_course.
* Determinism: repeat calls are identical (tested). CPU ≈ 0.7 s per 20 s course flown (≈3.2 s per 90 s).

## P4.10 Course limits for ER (in `v2_results/p4_aircraft_limits.json`)
| aircraft | α_stall (deg) | v_max (KCAS) | M_max | course bank (deg) | ER placeholder | differs |
|---|---|---|---|---|---|---|
| c172x | 15 | 163 (POH VNE; FD V_D 180) | – | 60 | 15°, 60°, 1.25 v_ref = 125 kt | v_max higher (163) |
| T38 | 18 | 565 (0.95 × FD V_D 595; published ~710 KIAS / M1.3) | 0.90 (FD strip theory limit) | 75 | 18°, 60° | bank 75; Mach cap |
| 737 | 13 | 350 (VMO, 737-200) | 0.84 (MMO) | 60 | 12°, 60° | α 13 |
| f16 | 25 (FCS AoA limiter; tables go higher) | 565 (0.95 V_D; published 800 KCAS / M2) | 0.90 | 80 | (none), 60° | bank 80 |
α_stall = planform_b2 SECTIONS (clean wing, notional handbook); f16 uses the FCS limiter. Prefer these over 1.25·v_ref:
1.25 v_ref is far above VNE for the c172x at low v_ref, and above M0.9 for the jets at high v_ref. Use `v_max = min(1.25 v_ref, v_max_kcas)`.
Bank: sustained-turn banks (P4.8) are 59/67/66/78°. The course bank above is a scoring limit, not a structural one.

## P4.11 Guidance ownership (recommendation, agreed with the parent default)
**Genome owns the guidance law and its parameterisation** (ring → attitude/nz/speed commands, mixing, prefilter). **FD owns the
plant**: JSBSim + B2a flex + actuators/rate/lag/hinge limits + aircraft limits. FD does NOT ship a guidance law. `fly_course`
takes guidance as a callable (Genome's `phase4_rings` decode → closure). **Evolution owns scoring**: ring terms + FD signals.
FD clamps nothing on the command side except the physical limits (cmd ∈ [-1, 1], throttle ≤ throttle_max).

## P4.12 ER Q&A
* *Fidelity / pins?* `full_a1_b2a_cs` (alias `full_a1_b2a_p4`). Pins in `v2_results/model_versions_post_p4cs.json`, checksums in
  `v2_results/FROZEN_P4cs.md5`. Cache key: model_version + struct + shape + surfaces_genome + course (incl. seed/stage/K) + guidance genes.
* *Do the 24 terms change?* No. TERM_KEYS are unchanged. The surface-use signals are outside TERM_KEYS (`in_cost: False`).
* *Suggested surface-use terms*: Σ_s use_tv_deg_per_s/rate_max (agility cost), sat_frac_rate + sat_frac_pos (actuator abuse),
  hinge_peak_frac > 1 (should never happen), and reversal_flown, which should be a hard flag.
* *J_g*: use n_inst and n_struct_limit from the limits file, not the genome clamp (agree with §8).
* *CPU*: active ≈ 1.1–1.3 × full_a1_b2a per 90 s scenario (P4.13).

## P4.13 Acceptance (02:35 PT)
* passthrough == full_a1_b2a bit for bit on all 4 aircraft (baseline) and on a shaped + struct genome (c172x). Only the
  model_version / fidelity labels differ, and the `ctrl_surfaces` block is added.
* determinism (active, repeat runs and fly_course), actuator rate-limit, lag (exact ZOH), hinge blow-down,
  η(q) monotone + softer-GJ ⇒ lower η + η(q_D) = margin-screen `aileron_effectiveness_at_VD` within 0.02 (reversal trend),
  limits-file sanity, frozen md5 check. `test_ctrlsurf_p4.py` 19 tests.
* CPU, one 90 s Phase-1 scenario (`v2_results/p4cs_cpu.json`, process time s, b2a / active / passthrough):
  c172x 2.54/2.72/2.90, T38 2.20/2.63/2.86, 737 2.08/2.65/2.61, f16 2.28/2.79/2.53, so ≈ +7…27 % from the per-step Python
  capture (single runs, noisy).
* Active-mode cost change vs B2a on the Phase-1 best gains: c172x +2.5 %, T38 +0.7 %, 737 +3.3 %, f16 +2.8 %. No rate saturation
  at Phase-1 gains. Flown η_min: ail 0.96/0.88/0.68/0.93 (c172x/T38/737/f16).
