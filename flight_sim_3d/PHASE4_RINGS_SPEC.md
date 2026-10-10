# Phase 4 ring course: shared spec (v0.6, Sim Bridge, 2026-10-07 ~05:20 PT; `ring_course/1.2`)

**Status:** DRAFT. Sim Bridge owns the course, gate rules, telemetry and viewer sections. FD (§8), Genome (§7) and
ER (§6) sections link their own documents. Sim Bridge did not rewrite them.

**Location:** there was no writable team copy of `flight_sim_3d/`; the staging snapshots under
`/workspace/phase*-push-staging/` are read-only. This file is therefore
`/workspace/flight-sim-team/flight_sim_3d/PHASE4_RINGS_SPEC.md`.

**Corleone's requirements (02:15 PT):**
- Rings are generated on the track, 5 at a time, at varying X/Y/Z.
- Plane-to-plane collisions are OFF.
- Control surfaces move to control the craft.
- Scoring covers control-surface use, agility, and accuracy through the rings.

**Code (Sim Bridge):**
- `sim-bridge/sim_bridge/ring_course.py` is the canonical course module. It is **stdlib only**, so ER/FD/Genome can
  import it directly, like `v2_map.py`.
- `sim-bridge/sim_bridge/rings.py` holds the viewer/trajectory helpers.
- `sim-bridge/viewer/js/rings.js` renders the rings.
- Tests are in `sim-bridge/tests/test_rings.py`.

> **v0.6 (`ring_course/1.2`):**
> - Per-ring timeout in `score_course` (§3).
> - New seed derivation `course_seed(run_seed, gen, k, aircraft, holdout)`: courses change every run AND every
>   generation (§4).
> - Trajectories carry their course provenance (§5).
>
> Ring geometry for a given (model, stage, seed) is unchanged from 1.1, but scores can differ (timeouts), and new runs
> get new seeds.
>
> **v0.5 COURSE GEOMETRY CHANGE:** from v0.5 (`ring_course` VERSION 1.1) on, all geometry uses **TAS at h0** (ER's
> Q-ER5 decision, 02:53 PT). Every course generated with v0.4.1 or earlier (VERSION 1.0, KCAS-as-m/s) differs, so the
> same (model, stage, seed) now gives different rings. **ER and Genome must re-generate courses** and re-key their
> caches with the course `version`.

## 0. Reconciliation with Architecture v1 (frozen 2026-09-13/14)

**Phase 4 amendment to Architecture v1 approved by Corleone 2026-10-07 02:45 PT: C1–C6 exactly as written; rim-hit = crash OFF by default.** Phase 4 aircraft: c172x, T38, 737, f16.

No Architecture v1 *document* exists on the box, because the v1 repo was never minted. The v1 contract below comes
from the team's lane-lock records.

| v1 item (owner) | Phase 4 | status |
|---|---|---|
| `CourseGenerator(courseSeed) -> Ring[]` + entry spawn (SB) | `ring_course.make_course(model, stage, seed)` + `ring_at(course, k)` + `start` | OK |
| Sequential rings, CG pass (FD `gatePass`) | kept. Strictly sequential, CG segment crossing (§3). FD adopts SB's `gate_check` / `score_course` as the reference and ships no gatePass/rimHit of its own (FD §P4.14); scoring = SB + ER. | OK |
| Rim hit = crash (FD `rimHit`) | **C5 (approved):** ER's accuracy term is continuous at the rim and has no rim crash. The rim rule is implemented but **off by default** (`rim_tube_m = 0`). It is a decision for Corleone/FD/ER. | APPROVED (amendment) |
| `validateCourse`, reject + resample via seed+k (FD) | **C6 (approved):** ER's stage geometry is bounded by construction (offsets are fractions of R_turn at 30° bank, spacing ≥ 10 s), so there is no reject loop. The only clamp: the vertical step is reflected to keep the altitude ≥ 2 × min AGL. FD may add a hard check later. | APPROVED (amendment) |
| `nominalTime` → `T_ref` (FD) | `nominal_time(course)` = polyline length start → ring M−1 / V_TAS (ER's definition, TAS since v0.5) | OK (SB/ER, FD confirmed) |
| Objectives `{gatesHit, timeSec, tAlive, finished, crash}` + T_ref | `score_course` returns passes, misses, M, J_ring_miss, finished, crash, timed_out, t_last, nominal_time_s, time_limit_s. tAlive comes from FD's `status`. | extension |
| Fitness gatesHit ≫ finish ≫ timeBonus ≫ crashPenalty 0.25 (ER) | **C1 (approved):** ER's Phase 4 cost has weighted terms plus a hard fail (§6) | APPROVED (amendment) |
| Envelope V_cruise = 40 m/s, n_max 2.5, h_band | **C2 (approved):** per-aircraft V_ref (100/300/250 KCAS) and FD's per-aircraft limits (`p4_aircraft_limits.json`) | APPROVED (amendment) |
| One course seed per generation, new each generation | **C3 (approved) (resolved compatibly):** K = 4 courses per generation, shared by the whole population, plus 8 fixed hold-out courses | APPROVED (amendment) |
| Open-loop u, δt = 1; genome = T0 + nLapse + 48 CPs | **C4 (approved):** closed-loop guidance chromosome (29 genes, §7) | APPROVED (amendment) |
| Finite ordered ring list + `finished` | Finite M per course (default 15), revealed through a sliding window of 5 | OK |
| Fixed dt | sim dt 1/120 s; gates are interpolated between steps | OK |
| Replay-only viewer | kept | OK |
| v1 world frame (in-browser three.js) | **NED**, origin on the ground below the start (§1); conversions to FD N/E/U, trajectory ENU and three.js | defined here |

## 1. Frames and units (canonical: NED)

- **Course / ring frame (canonical): NED metres.** x = North, y = East, z = Down. The origin is the ground point
  directly below the start position. The start is at `[0, 0, −h0]`. Heading is in degrees from north, clockwise; the
  start heading is 0 (north).
  The ground is 0 m MSL under every start (JSBSim default terrain; FD §P4.14, resolves Q-FD7).
- **FD `fly_course` positions are [N, E, U] m** (FD §P4.14): N and E are ground distances from the start, U is the altitude
  above MSL (h-sl). With ground at 0 m MSL, the origin is the ground point below the start. If terrain is ever added, U stays
  MSL (then D = −(U − ground_msl) would need the terrain height; not needed now). The mapping is **Up = −D**:
  `neu = [N, E, −D]`, `ned = [N, E, −Up]`.
  - Helpers: `ned_to_neu`, `neu_to_ned`, `fd_pos_to_ned`.
  - Every ring carries both forms (`centre_m`/`normal` in NED, plus `centre_neu_m`/`normal_neu`), so the two frames
    can't be confused.
  - A test shows that unconverted N/E/U input scores 0 passes.
- **Sim Bridge trajectory files: ENU** (x E, y N, z up), origin at the start, z relative to `frame.origin_alt_m`
  (= start altitude MSL).
  - `ned_to_traj_enu(p, origin_alt_m) = [E, N, −D + ground_msl − origin_alt_m]`.
  - Trajectory `course` blocks add `centre_enu_m`/`normal_enu` per ring.
- **Viewer three.js:** `ned_to_three(p) = [E, −D, −N]` (x east, y up, z south), before layout offsets and vertical
  exaggeration.
- **Body:** FRD. Quaternions body→world as `[w, x, y, z]`.
- **Units:** m, s, deg in JSON, kt for `kcas`, ft for `alt_ft`.

## 2. Course generator (Sim Bridge; geometry = ER `evolution/rings.py` STAGES)

API (ER's signature; `model` = aircraft id `c172x` | `T38` | `737` | `f16`):

```python
c = ring_course.make_course(model, stage, seed, n_rings=5, M=None, rim_tube_m=0.0)   # -> dict
c["rings"]      # first n_rings rings (the initial window): [{k, id, centre_m, normal, radius_m, tube_m, scored,
                #                                             centre_neu_m, normal_neu}]
c["start"]      # {pos: [0,0,-h0] NED, heading_deg: 0, alt_ft, kcas}
c["M"], c["window"], c["nominal_time_s"], c["time_limit_s"], c["params"], c["frame"]
ring_course.ring_at(c, k)     # ring k for ANY k >= 0, deterministic from (seed, k); k >= M = unscored preview
ring_course.window(c, n)      # active indices [n, min(M, n+5))
```

**D1 (resolved: ER accepts it as-is):** ER asked for `-> list[dict]` plus a start state. SB returns one dict with
`rings` and `start`, because the course also has to carry M, the limits and the seed for `ring_at`. The ring keys
are `centre_m`/`normal`/`radius_m` (with units) where ER's draft uses `centre`/`normal`/`radius`. ER to confirm or
ask for aliases.

### 2.1 Stages (ER's numbers; R_turn = V_TAS² / (g · tan 30°); "s at V_ref" = seconds at V_TAS)

| stage | spacing (s at V_ref) | lateral offset (× R_turn) | vertical offset (× spacing) | ring radius (s at V_ref) | M |
|---|---|---|---|---|---|
| easy | 20 | ±0.05 | ±0.02 | 0.60 | 15 |
| medium | 14 | ±0.15 | ±0.05 | 0.40 | 15 |
| hard | 10 | ±0.30 | ±0.08 | 0.25 | 15 |

- **Vertical offset**, defined the same way as ER's: a uniform ±(fraction × spacing) height change per ring,
  relative to the previous ring.
- The lateral offset is uniform ±(fraction × R_turn), to the right of the *current track*.
- Each ring is `spacing` ahead of the previous one along the current track; the track is the horizontal direction
  of the previous chord.

### 2.2 Per aircraft (V_ref input = KCAS; geometry = TAS at h0, ISA, since v0.5)

`ring_course.cas_to_tas(v_cas_ms, h_m)` uses the ISA troposphere and the compressible (isentropic, subsonic) formula,
stdlib only:
- qc = p0[(1 + 0.2 (Vc/a0)²)^3.5 − 1];
- M = √(5[(qc/p + 1)^(2/7) − 1]);
- TAS = M · a(h).

The course object exposes `v_tas_ms` (top level and `params`), plus `v_ref_kcas`. `aircraft_info(model)` gives the
same derived numbers, and `params.v_ref_mps` is kept as an alias of `v_tas_ms`. Spacing, radius, R_turn,
`nominal_time_s` and `time_limit_s` all use V_TAS.

| | V_ref | h0 | V_TAS | R_turn (30°) | spacing easy / medium / hard | radius easy / medium / hard |
|---|---|---|---|---|---|---|
| c172x | 100 KCAS | 4000 ft | 106.1 kt (54.57 m/s) | 526 m | 1091 / 764 / 546 m | 32.7 / 21.8 / 13.6 m |
| T38 | 300 KCAS | 10000 ft | 345.4 kt (177.67 m/s) | 5576 m | 3553 / 2487 / 1777 m | 106.6 / 71.1 / 44.4 m |
| 737 | 250 KCAS | 10000 ft | 288.7 kt (148.52 m/s) | 3896 m | 2970 / 2079 / 1485 m | 89.1 / 59.4 / 37.1 m |
| f16 | 350 KCAS | 10000 ft | 401.5 kt (206.57 m/s) | 7536 m | 4131 / 2892 / 2066 m | 123.9 / 82.6 / 51.6 m |

Checks (tests):
- The f16 matches FD: 401.5 KTAS, 206.6 m/s, R_turn 7536 m vs FD ≈ 7540 m.
- The other three match hand ISA values: c172x 106.1, T38 345.4 and 737 288.7 KTAS.
- On all 4 aircraft × 3 stages, every leg is ≥ FD's `ring_hint.min_spacing_ft`.
- The circumradius of consecutive ring centres is ≥ `ring_hint.min_path_radius_ft`.
- Every ring is ≥ 2 × min AGL.

**f16 sources:** FD, `flight-dynamics/v2_results/p4_aircraft_limits.json` (trim block), plus FD's PHASE4 doc.
- Pins: active `430793a6`, pass-through `43245063` (`v2_results/model_versions_post_p4cs.json`; FD P4 files frozen,
  `FROZEN_P4cs.md5`, read-only for SB).
- V_ref = **350 KCAS at 10 000 ft** (401.5 KTAS, 677.7 ft/s). Start altitude h0 = 10 000 ft (the trim altitude).
- FD values:
  - 30°-bank turn radius at V_ref ≈ 24 700 ft (7540 m, using TAS);
  - min path radius 4690 ft (1430 m); min ring spacing 2401 ft (732 m);
  - instantaneous turn radius 1634 ft at 8.8 g; roll rate 110 °/s; course bank 80°;
  - v_max 565 KCAS / M 0.90.
- Control surfaces: elevator, aileron, rudder, plus a flight-computer-scheduled speedbrake.
- Geometry uses TAS since v0.5, so SB's R_turn (7536 m) matches FD's ≈ 7540 m. The FD values are also stored in
  `AIRCRAFT["f16"]` as `fd_*`.
- min_agl = 500 ft: FD's limits file has no min AGL, so it is set like the other jets (flagged in Q-FD9).

- **Note:** the c172x hard radius of 13.6 m is close to its 11 m span, which is fine for a CG pass. If rim/mesh rules
  return, it becomes tight.

### 2.3 Determinism

- Ring k uses `u = sha256("p4course|seed|k|j")` → uniform [−1, 1) for j = 0 (lateral) and 1 (vertical). Ring k
  depends only on (seed, k) and ring k−1, so `ring_at(c, k)` is deterministic for any k.
- It is stdlib only: **not bit-identical to ER's numpy reference generator** (ER said SB's replaces it).
- The altitude clamp: if a vertical step would put the ring below 2 × min AGL (500 ft), the step is reflected upward.

## 3. Window, order, time limit and gate rules (`ring_course.score_course`, pure, stdlib)

- **Sliding window of 5:**
  - Active = `[n, min(M, n+5))`, where n is the first unresolved ring.
  - When ring n resolves (pass or miss), the window slides by one: ring n+5 spawns, up to ring M−1.
  - At the end the window shrinks (rings M−4 … M−1).
- **Previews:** `ring_at(c, k ≥ M)` exists for guidance look-ahead only: unscored, never displayed. So rings n and
  n+1 always exist for the controller (§3a).
- **Course end:**
  - Each course has M scored rings (default 15 on every stage). `finished` = all M resolved by flying, before the
    time limit.
  - The course object exposes `M`, `nominal_time_s` and `time_limit_s`.
- **Time limit** = 1.5 × `nominal_time_s`, where nominal time = polyline length start → ring 0 → … → ring M−1, divided
  by V_ref. Rings not resolved by the limit are `missed_time` (t = limit).
- **Crossing:** take the CG segment p_i → p_{i+1} (sim steps, 1/120 s).
  - Signed distance `s = (p − c)·n̂`; a forward crossing is `s_i < 0 ≤ s_{i+1}`.
  - Interpolate with `f = s_i / (s_i − s_{i+1})`; ρ = in-plane distance of the crossing point from the centre.
  - The time is interpolated the same way.
  - Backward crossings never count.
- **Ring n** resolves on its first forward crossing with ρ ≤ 4 r (ER's capture rule):
  - **pass** if ρ ≤ r;
  - **miss** otherwise, with `miss_m = ρ − r`.
  - Plane crossings with ρ > 4 r are ignored, so the plane can come back.
  - With `rim_tube_m > 0`: **rim** if |ρ − r| ≤ tube. That is a crash and ends the course (v1; off by default, C5).
- **Out of order:** a forward crossing *inside* (ρ ≤ r) a later window ring j (n < j < n+5) before ring n resolves
  marks rings n … j−1 as **missed_order** (at that time). Ring j is resolved as a pass, and scanning continues at
  j+1. This is consistent with v1's sequential rings: v1 had no rule for skipped rings, and this rule only adds one.
  ER's own `ring_crossings` (search after t_prev, no skip rule) scores a skipped ring as a miss too. They are
  equivalent on miss counts, differing only in the resolution time; a cross-check test is pending (Q-ER6).
- **Per-ring timeout (1.2; Genome's guidance applies the same rule):**
  - **leg time** of ring k = `|centre_k − centre_{k−1}| / v_tas_ms` (3-D distance between ring centres), where leg 0
    runs from `start.pos`. See `ring_course.leg_time(course, k)`.
  - The clock for ring n starts at the resolution time `t_res` of ring n−1 (t[0] for ring 0). For a pass, miss or
    out-of-order resolution, that is the interpolated crossing time; for a timeout, it is the sample time below.
  - If no crossing resolves ring n on the segment (t[i−1], t[i]] and `t[i] − t_res > ring_timeout_factor × leg_time(n)`,
    ring n is **timeout** at t[i] (the first sample past the deadline; `deadline_t` is recorded too). Then t_res = t[i]
    and the window slides by one. There is at most one timeout per sample.
  - `ring_timeout_factor` is a course parameter (`params.ring_timeout_factor`, `make_course(..., ring_timeout_factor=3.0)`,
    default **3.0**).
  - The overall time limit still applies, and timeouts count as misses.
  - `score_course(..., ring_timeout=False)` reproduces 1.1 scoring (for ER's phase4-smoke-s1).
- **Outputs:**
  - `gates[]`: `{k, id, result ∈ pass|miss|missed_order|missed_time|timeout|rim, t, rho_m, rho_norm, miss_m, point_m}`;
  - `passes`, `misses`, `M`, **`J_ring_miss = (M − passes)/M`** (= misses/M once all M are resolved);
  - `finished`, `crash`, `timed_out`, `t_last`, `nominal_time_s`, `time_limit_s`.
- **Frame:** `score_course` takes NED positions. Convert FD's N/E/U with `fd_pos_to_ned`.

## 3a. Guidance interface (Genome Architect's questions)

1. **Ring frames:** each ring has `id` (`"<model>:<stage>:<seed>:<k>"`), `k`, `centre_m` (NED), `normal` (unit,
   **direction of travel**), `radius_m`, plus `centre_neu_m`/`normal_neu` (FD frame) and `scored`.
2. **Spacing and look-ahead:**
   - Spacing per §2.1/§2.2.
   - **Guarantee:** rings n and n+1 always exist (`ring_at` previews past M). Rings n … n+4 are active while
     n+4 < M.
3. **Window:** slides by one. When n is passed or missed, ring n+5 spawns, so 5 are active until the end of the
   course.
4. **V_ref:** `c["v_ref_kcas"]` (KCAS, = `start.kcas`) and `c["v_tas_ms"]` (TAS at h0, which the geometry uses).
   The speed command = `v_cmd_scale × V_ref`, in KCAS (the controller reads `vc_kts`).
5. **Pass and miss:** §3 exactly.
6. **Per-step inputs:** `ring_course.guidance_view(pos_ned, q_body_to_ned, c, n, n_ahead=2)` returns the following
   for rings n and n+1:
   - `range_m`;
   - LOS unit vectors `los_ned` (**canonical**), `los_enu` and `los_body_frd`;
   - `bearing_deg = atan2(y_b, x_b)` (+ = right) and `elevation_deg = atan2(−z_b, hypot)` (+ = above), both in body
     FRD;
   - `normal_ned`, `normal_body_frd`, `radius_m`, `id`, `scored`, `centre_ned`.

   **Resolved Q-G4: the controller receives the canonical NED state.** The N/E/U → NED flip happens ONCE, at the bridge,
   so neither the genome nor FD's `fly_course` flips any sign:

   ```python
   gi = ring_course.guidance_inputs(fd_state, course, n, n_ahead=2)
   ```
   - `fd_state` = FD's `guidance(state, ...)` dict: `t, pos_m [N,E,Up], phi, theta, psi, p, q, r, vc_kts, vt_fps, nz,
     alpha, beta, agl_m, h_dot_fps`. A raw-NED caller passes `pos_ned` instead of `pos_m` (exactly one).
   - Returns `pos_ned`, `alt_m` (= −D), `gamma_rad`/`gamma_deg` (= asin(h_dot/V_true), climb > 0), `q_body_to_ned`
     (from φ, θ, ψ), `n`, the FD fields passed through unchanged, and `rings[]` for n, n+1:
     `k, id, scored, radius_m, range_m, los_body_frd, bearing_deg, elevation_deg, los_ned, centre_ned, normal_ned,
     normal_body_frd`.
   - Ring positions and normals are NED.

   **Guidance contract (Q-G5 resolved: Genome confirmed the names, no renames).** The genome reads exactly:
   - `rings[0..1]`: `los_body_frd`, `range_m`, `bearing_deg`, `elevation_deg`, `normal_body_frd`, `radius_m`;
   - `alt_m`, `gamma_deg`, `phi`, `theta`, `p`, `q`, `r`, `vc_kts`, `nz`, `alpha`, `beta`.

   Other returned fields are informational.

## 4. Seed policy (ER contract, Corleone 04:59 PT: courses change every run AND every generation)

**One helper:**

```python
ring_course.course_seed(run_seed, gen, k, aircraft, holdout=False) -> int  # unsigned 64-bit
```

**Canonical string** (UTF-8; integers in base-10 ASCII as Python `str(int(x))`, no padding; `aircraft` = the
`AIRCRAFT` id verbatim, case-sensitive: `c172x`, `T38`, `737`, `f16`):

```
train   (holdout=False): "p4course-seed/1|run=<run_seed>|gen=<gen>|k=<k>|ac=<aircraft>"         k = 0..K-1
holdout (holdout=True):  "p4course-seed/1|run=<run_seed>|gen=holdout|k=<j>|ac=<aircraft>"       j = 0..7, gen ignored
seed = int.from_bytes(sha256(string).digest()[:8], "big")
```

- Golden value: `course_seed(1, 5, 2, "T38")` = **2920704114203819991**. Reproduce it with
  `printf '%s' 'p4course-seed/1|run=1|gen=5|k=2|ac=T38' | sha256sum`, then take the first 16 hex digits as an integer.
- Hold-out golden: `course_seed(1, *, 3, "f16", holdout=True)` = 6831254776690595837.
- The stage sets difficulty only and is **not** an input to the seed. Python's `hash()` is never used.
- Train: K = 4 courses per genome per generation, shared by the population.
- Hold-out: 8 courses per run, fixed across the generations of a run, different between runs (derived from
  run_seed). **Resolved:** ER confirmed this is Corleone's ask (C3: "8 fixed hold-out courses for logging"). They are
  logged for the gen best and final elites only, never used for selection.
- Thin wrappers: `train_courses(run_seed, gen, aircraft, stage, K=4)` and
  `holdout_courses(run_seed, aircraft, stage, n=8)`. Both return courses with a `provenance` block.
- Legacy (1.1, ER's phase4-smoke-s1 only): `train_seeds`, `holdout_seeds` and `legacy_course_seed_v11` are 32-bit,
  with no aircraft. They are kept so ER's frozen smoke code still imports; new runs must not use them.
- Curriculum stage: ER's `curriculum_stage`. The cache key must include the stage, the seed and the course `version`.

## 5. Trajectory / telemetry additions (ga-flightsim-traj/2, additive, optional)

- `course`: `rings.course_block(course, origin_alt_m)`. This is the canonical NED course with all M rings, each ring
  also carrying `centre_enu_m`/`normal_enu` (trajectory frame) and `centre_neu_m`/`normal_neu`.
- `gates`: the `score_course` gates. `gate_summary`: the `score_course` summary without the gates.
- `events` gain `gate_pass` / `gate_miss` (detail text) for the HUD log.
- **Control surfaces (FD `ctrl_surfaces`, fd-ctrlsurf/1):**
  - FD records them at 120 Hz. **Sim Bridge decimates them to 30 Hz** (every 4th sample, as FD suggests) with
    `rings.ctrl_surface_channels`.
  - New channels: `elev_deg`, `ail_deg`, `rud_deg`, `ail_L_deg`, `ail_R_deg` (and `flap_deg`, `sbrk_deg`,
    `*_cmd_deg` where present).
  - The viewer page is decimated further to 5–10 Hz, which is fine for display.
  - The HUD shows a `SURF elev/ail/rud` line in degrees (FD position; **aileron + = right roll**).
  - The existing `elevator`/`aileron`/`rudder` channels stay cmd-norm.
- **Course provenance (v0.6, required for new runs):** the `course` block carries `course_seed`, `run_seed`, `gen`,
  `k`, `stage` and `version` (plus `provenance` {…, holdout, aircraft, seed_scheme}). For hold-out courses, `gen` is
  null and `k` = j.
  - `rings.course_block` writes these fields from `with_provenance`/`train_courses`.
  - **Replay and viewer:** the course is regenerated from (course_seed, stage, aircraft, version) with
    `ring_course.regenerate(block)`, then checked against the embedded rings with `check_rings` (tolerance 1 mm).
  - Versions 1.1 and 1.2 are geometry-compatible; 1.0 is rejected.
- Files without `course` stay valid; the viewer shows rings only when `course` is present.

## 6. Scoring and evolution (ER)

**Link:** [`evolution/analysis/PHASE4_SCORING_PROPOSAL.md`](../evolution/analysis/PHASE4_SCORING_PROPOSAL.md)
(ER's proposal; not rewritten here). SB-relevant points it uses:
- J_ring_miss = misses/M with M from the course;
- J_ring_acc from ρ/r;
- J_time vs T_ref = `nominal_time_s`;
- K = 4 plus 8 hold-out;
- stages easy/medium/hard;
- hard fail on ground/structural/divergence.

The surface-use terms (`J_defl_rms`, `J_rate_rms`, `J_sat`, `J_chatter`) use FD's `ctrl_surfaces` (§8).

## 7. Guidance / control chromosome (Genome Architect)

**Source:** [`genome/PHASE4_RINGS_GENOME.md`](../genome/PHASE4_RINGS_GENOME.md), quoted verbatim below without
edits. The course questions it raises for SB are answered in §2–§3a.

> # Phase 4 ring course: Genome section (Genome Architect, 2026-10-07 ~02:40 PT)
>
> Meant for the Genome section of Sim Bridge's `flight_sim_3d/PHASE4_RINGS_SPEC.md`. That file and `flight_sim_3d/`
> did not exist anywhere under /workspace/flight-sim-team at 02:20 PT, so this section lives here and gets pasted in once SB publishes the skeleton.
>
> ## 1. Chromosome: 29 genes = guidance 12 | inner_loop 11 | mixing 6
> All genes are stored as u in [0,1] and decoded by `genome_schema.GeneSpec` (log: min·(max/min)^u, linear, log0 = exact 0 for u ≤ 0.05).
> Identity = encode(default). Ranges are c172x reference values. Surfaces are JSBSim `fcs/{aileron,elevator,rudder,throttle}-cmd-norm`.
>
> | block | gene | range | default | scale | units |
> |---|---|---|---|---|---|
> | guidance | t_preview_s | 1–8 | 3 | log | s (aim point along the line of sight (LOS) to the next ring) |
> | guidance | w_next_ring | 0–0.6 | 0.25 | linear | blend toward ring n+1 |
> | guidance | k_lat | 0.1–5 | 1.0 | log | deg bank / deg LOS bearing error |
> | guidance | k_lat_rate | 1e-3–2 | 0 | log0 | deg bank/(deg/s) |
> | guidance | k_vert | 0.05–3 | 0.5 | log | deg γ / deg vertical LOS error |
> | guidance | k_vert_rate | 1e-3–1 | 0 | log0 | deg γ/(deg/s) |
> | guidance | bank_max_deg | 15–75 | 45 | linear | deg |
> | guidance | nz_max_g | 1.5–4 | 2.5 | linear | g (clamped to FD limit) |
> | guidance | nz_min_g | −0.5–1 | 0 | linear | g |
> | guidance | gamma_max_deg | 5–25 | 15 | linear | deg |
> | guidance | v_cmd_scale | 0.85–1.25 | 1.0 | linear | × course V_ref |
> | guidance | v_turn_comp | 0–0.15 | 0 | linear | × V_cmd per (nz−1) g |
> | inner_loop | kp/ki/kd_roll, kp/ki/kd_pitch, kr_yaw, kbeta_yaw, tau_washout, kp_spd, ki_spd | identical to genome_schema.ALL_GENES (test-enforced) | | | |
> | mixing | k_ari | 0–0.6 | 0 | linear | rudder per aileron |
> | mixing | k_turn_coord | 0–1.5 | 0 | linear | rudder per deg/s r_cmd = g·tanφ/V |
> | mixing | k_elev_bank | 0–0.5 | 0 | linear | elevator × (1/cosφ − 1) |
> | mixing | ail_auth | 0.4–1 | 1.0 | linear | × travel |
> | mixing | rud_auth | 0.2–1 | 1.0 | linear | × travel |
> | mixing | tau_cmd_s | 0.02–0.5 | 0.05 | log | s, surface command prefilter |
>
> With every mixing gene at identity, the airframe sees the plain inner-loop commands: no interconnect, no feed-forward, full authority.
>
> ## 2. Carry-over (recommendation: A)
> - **A (recommended, wired):** `carry_over: "none"`. Structure is held at FD baseline and the planform at B1 r1 identity. Neither is in the vector.
>   Reasons: there's no ring-course structural or energy cost yet (ER owns scoring), the moving-surface fidelity has no pins,
>   and B1/B2a gave small shape gains (≤ 0.03 cost) that would compete with 29 new controller genes for search budget.
> - **B (wired, opt-in):** `carry_over: "structure"` adds phase3_b1's 12 structure_v2 genes as a 4th whole block (41 genes).
>   Needs FD's flexeval as the structural cost (struct_v2_source fd) on the ring envelope.
> - **C (not wired):** structure plus B2a shape (9 genes, tc locked) means 50 genes. It needs block_ops shape ops plus FD's ring fidelity and pins, and the geometry gate stays a hard reject.
>
> ## 3. Operators (baseline per ER's B1 A/B)
> Elite 2, flat rank with p 0.2, whole-block crossover (`d = rng.random(3)`, guidance|inner_loop|mixing, d < 0.5 → parent A), then
> gauss mutation (`rng.random(29) < 0.15`, N(0, 0.08) on hits, clip). Gen-0 is `rng.random((pop, 29))`, the same as phase3_b1's controller block.
> Evidence (evolution/analysis/STATUS_P3B1_ab.md): tweaked (elite 4 + uniform) beat baseline in only 1 of 6 (seed, aircraft) pairs.
> The mean Δ was c172x −0.85 %, T38 +0.22 %, 737 +2.46 %, and there was no consistent shape-contribution gain. ER recommends baseline. Caveat: 2 seeds, and the effects are inside 1 seed sd.
>
> ## 4. Files
> `phase4_rings.py` (standalone loader / encode / decode / operators / evaluator stub), `presets/phase4_rings.json`,
> `p4_operator_trace.py` → `runs/p4_operator_trace.json`, `tests/test_phase4_rings.py`, and `tests/test_adapter.py` (skips the
> phase4 preset in test_all_presets_load). adapter.py, block_ops.py and every existing preset are unchanged.
>
> ## 5. Open questions
> - SB: the course spec (ring frames, radius, spacing, how many rings ahead are visible, V_ref per aircraft, pass/miss geometry, what counts as "rolling sets"); the guidance-state API (LOS to rings n and n+1).
> - FD: are rudder and differential surfaces live on every aircraft in the moving-surface model? Per-aircraft nz/bank limits for clamping; ring fidelity name and pins; surface rate limits (tau_cmd_s may be redundant); per-aircraft gain rescale (profiles.py tags).
> - ER: ring scoring terms (miss distance, time, effort, structural via FD); the A/B rerun on rings; mirror the phase4 operators and compare against runs/p4_operator_trace.json hashes.

## 8. Control-surface model and flight (FD)

**Link:** [`flight-dynamics/PHASE4_FD_CONTROL_SURFACES.md`](../flight-dynamics/PHASE4_FD_CONTROL_SURFACES.md)
(FD's section; not rewritten here). Interface points SB relies on:
- `fly_course` returns `pos` as **north/east/UP metres**. Convert it with `ring_course.fd_pos_to_ned` before
  `score_course` (§1).
- `ctrl_surfaces` arrive at 120 Hz and SB decimates them to 30 Hz (§5).
- Sign conventions, from FD:
  - elevator + = trailing edge down (nose down);
  - **aileron + = right roll (+p, right wing down)** on c172x, T38, 737 **and f16**. The "fourth" aircraft in FD's
    note is the **f16**, which is in FD's tables but not in the Phase 4 c172x/T38/737 line-up (Q-C3);
  - rudder + = JSBSim + rudder.
- Surface limits, rates and lags per aircraft: FD's P4.3 table and `v2_results/p4_aircraft_limits.json`.

## 9. Collisions between planes: OFF

- Each aircraft runs in its own independent JSBSim instance; nothing couples them.
- The viewer's formation and side-by-side layouts may overlap aircraft visually; overlap is **never** a crash or an
  event.
- Ground impact still ends that aircraft's run (FD `status` = ground). A rim crash applies only if C5 turns it on.

## 10. Viewer rendering (Sim Bridge)

- Rings are drawn as tori, using the trajectory transform (true-position layout; in formation layout the lanes
  are re-mapped, so rings are hidden there).
- Colours:
  - current target: bright amber;
  - other active rings: amber;
  - passed: green;
  - missed (miss/order/time): red;
  - rim: orange;
  - unspawned: hidden (`rings=all` shows them faint).
- The state comes from `gates[]` at playback time; the window logic matches `rings.ring_state_at` (node test).
- HUD lines:
  - `RINGS done/M · pass · miss · rim · acc · last miss`;
  - `course <model> <stage> seed … · M · T_nom · limit`;
  - `SURF` (§5).
- Synthetic files carry `course.synthetic = true`, the HUD says SYNTHETIC, and the page note banner says SYNTHETIC.
- `rings=0` hides the rings.

## 11. Open questions

**Resolved (v0.6)**
- Seed derivation: ER adopted `course_seed(run_seed, gen, k, aircraft, holdout)` (§4).
- Hold-out policy: fixed within a run, derived from run_seed; ER confirmed this is Corleone's ask.
- Per-ring timeout mismatch (ER): now implemented in `score_course` (§3), factor 3.0.

**Resolved (v0.5)**
- Q-ER5 (ER, 02:53 PT): all four aircraft use TAS at h0 for geometry and T_ref (§2.2). Courses changed from v0.4.1.
- Q-FD10 (ER): surface-use scoring uses the actual deflections and rates FD reports, so the f16's FBW commands are fine.

**Resolved (v0.3)**
- Q-FD1: FD uses SB's `gate_check`/`score_course` as the reference; there is no FD gatePass/rimHit (FD §P4.14).
- Q-FD7: the ground is 0 m MSL under every start.
- Q-FD8: `fly_course` pos = [N, E, U MSL] m, origin at the ground below the start (§1).
- nominal_time moves to SB/ER (confirmed by FD).
- Q-ER7: ER accepts one course dict with `centre_m`/`radius_m`/`normal`; no aliases (D1 closed).
- Q-G4: the controller receives canonical NED via `guidance_inputs` (§3a).
- Q-G5: Genome confirmed the `guidance_inputs` names (contract in §3a).
- Q-C1/Q-C2: closed by Corleone's approval (02:45 PT): C1–C6 as written; rim-hit = crash OFF by default.
- Q-C3: the f16 joins Phase 4 (§2.2).

**FD**
- Q-FD2: should a hard flyability check (`validateCourse`) be added on top of ER's bounded stages (C6)?
- Q-FD3: rim rule on the CG (v1) or on wing tips / mesh, if C5 turns rims on?

**Genome**
- Q-G2: is n and n+1 enough, or does the controller want n+2 … n+4 (available while < M)?
- Q-G3: carry-over A (structure/shape frozen): ER and Corleone to confirm.

**ER**
- Q-ER1: the cost ordering vs v1 (C1).
- Q-ER3: confirm M = 15 on every stage and the 1.5 × nominal time limit.
- Q-ER6: cross-check `score_course` against `evolution/rings.ring_crossings` on shared tracks.

**f16 (new)**
- Q-FD9 (FD/ER): f16 min_agl = 500 ft is SB's choice (FD's limits file has none); V_ref and h0 now come from FD's trim block.
- Q-FD11 (viewer): the f16 viewer model is procedural (cropped delta, single fin; `aircraft.js` dimensions span 9.96 m,
  length 15.06 m). The shape is ESTIMATED; there is no JSBSim mesh.
