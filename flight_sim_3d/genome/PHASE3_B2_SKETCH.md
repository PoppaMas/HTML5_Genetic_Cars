# P3-B2 genome-side spec: **WIRED for B2a** (FD INTERFACE_v2 §15.10–15.11 / `v2_results/p3b2_gene_spec.json`)

**Status (2026-10-06 ~21:33 PT):** **WIRED / signed-off.** FD B2a SIGNED OFF (~21:31 PT). Genome opt-in presets
`phase3_b2a` / `phase3_b2a_x` land under `genome/` (`shape_b2.py`, block key `shape_b2`, fidelity `full_a1_b2a`).
Pins (`model_versions_post_p3b2a.json`):
c172x `full_a1_b2a:flexv2b2a:847bed9b`, T38 `…:b635a51d`, 737 `…:5d5a8f17`, f16 `…:07b08913`.
**Ranges (B2a r0 review):** dihedral **0…+3** (0…+2 on 737); tc_root floors raised (c172x 0.875, T38 0.925, 737 0.90, f16 0.85).
`locked_genes` = `wing_tc_root_scale`, `wing_tc_tip_ratio` (FD `REQUIRES_ENERGY`) until ER wires `J_energy`. Active shape
n_s = 9 → **29 genes**. B2b size (`wing_area`/`wing_aspect`) stays unimplemented / locked. ER does not yet list
`full_a1_b2a` in `FIDELITIES` → genome evaluator uses `via='fd'` (`flexeval_b2.evaluate` with ER inputs).
Hard rules unchanged: `struct_v2_source: fd`, TERM_KEYS = 24, P2.5 structure ranges, no parallel structural formula,
gate rejects = hard fail_cost, never a credit. Legacy / v4 / v5 / phase1_flex / phase2_flex / phase3_b1 / phase3_b1_x
bit-identity preserved (`shape_b1` paths untouched).

## 1. Genes (FD's exact list and order; linear in value; per-aircraft ranges; L = R)

Vector = the 6 B1 r1 genes (unchanged, `planform_b1` order) then the rows below in this order. Lengths are **11** at
`full_a1_b2a` and **13** at `full_a1_b2`. Decode is `value = lo + u·(hi − lo)` and encode is its inverse, with **no log
scale** (JSON `decode`). **Ranges are per aircraft, so decode needs the model:** `pb2.decode_shape_b2(genes, model)` /
`encode_shape_b2(genes, model)`. These raise and never clip. In B2a, decode accepts `wing_area_scale` /
`wing_aspect_scale` only at 1.0 (`deferred_in_b2a`).

| idx | gene | stage | c172x | T38 | 737 | f16 | default | identity u (c172x / T38 / 737 / f16) |
|---|---|---|---|---|---|---|---|---|
| 6 | `wing_dihedral_delta_deg` | B2a | **0 … +3** | **0 … +3** | **0 … +2** | **0 … +3** | 0 | **0 / 0 / 0 / 0** |
| 7 | `wing_tc_root_scale` | B2a, **LOCKED** (§4) | **0.875 … 1.25** | **0.925 … 1.25** | **0.90 … 1.15** | 0.85 … 1.25 | 1 | **0.333 / 0.231 / 0.4 / 0.375** |
| 8 | `wing_tc_tip_ratio` | B2a, **LOCKED** (§4) | 0.85 … 1.15 | same | same | same | 1 | 0.5 |
| 9 | `wing_camber_root_delta_pct` | B2a | −1.0 … +2.0 | −0.5 … +1.5 | −1.0 … +1.0 | −0.5 … +1.0 | 0 | 0.333 / 0.25 / 0.5 / 0.333 |
| 10 | `wing_camber_tip_delta_pct` | B2a | as row 9 | as row 9 | as row 9 | as row 9 | 0 | as row 9 |
| 11 | `wing_area_scale` | B2b | 0.90 … 1.15 | same | same | same | 1 | 0.4 |
| 12 | `wing_aspect_scale` | B2b | 0.90 … 1.15 | same | same | same | 1 | 0.4 |

- **Meanings (§15.1):**
  - dihedral: a uniform geometric ΔΓ (tip up +) on top of the JSBSim Γ0.
  - τ(η) = τ_r·(1 + (ratio − 1)·η), multiplied onto the baseline t/c.
  - Δm(η) is linear in η, additive on the thin-airfoil-equivalent m_eq (%c). It is additive because the T38 baseline
    camber is 0.
  - size: b′ = b0·√(k_S·k_A), c′ = c·√(k_S/k_A). Span is derived and bounded by the gate.
- **Baseline sections (notional, hashed):**

  | aircraft | section | t/c root / tip | m_eq root / tip | Γ0 |
  |---|---|---|---|---|
  | c172x | NACA 2412 | 0.12 / 0.12 | 1.8 / 1.8 | 1.73° |
  | T38 | 65A004.8 | 0.048 / 0.048 | 0 / 0 | 0° |
  | 737 | BAC equivalent | 0.15 / 0.105 | 2.5 / 2.0 | 6° |
  | f16 | 64A204 | 0.040 / 0.040 | 1.6 / 1.6 | 0° |

- **Default in range, identity exact:** every one of the 28 (aircraft × gene) rows contains its default.
  `lo + encode(default)·(hi − lo) == default` holds bit for bit for all 28 (checked from the JSON). Identity u is
  **not 0.5** for the asymmetric ranges. Genome must use `encode_shape_b2(defaults, model)`, never a hard-coded 0.5 or a
  zero vector.
- **u is not portable across aircraft.** Cross-aircraft seeding goes through physical dicts and can fall out of range:
  c172x t/c 0.8 is out of range on the other three, and dihedral +3 is out of range on the 737. Decode raises. The seeder
  rejects the row and logs it; it never clips silently.
- **Drift guard:** `B2_SCHEMA_PIN` holds (name, aircraft, lo, hi, default, stage) for all 28 rows, plus the vector order
  and `decode` string. Load fails on drift against the JSON now, and against `pb2.shape_schema_b2(model)` once it exists.

## 2. Block layout, crossover, mutation, gen 0

- **One shape block**, in FD's vector order: 6 (B1) → 11 (B2a) → 13 (B2b), minus locked genes (§4). The preset block key
  is `shape_b2` (wired) instead of `shape_b1`. It is not split into `shape_b1 + shape_b2`, because FD
  decodes one vector and a fourth block would break the crossover draw parity. Chromosome = controller 8 | structure 12 |
  shape n_s. With t/c locked, B2a gives n_s = 9 and 29 genes.
- **Crossover (parity with `phase3_b1` / `phase3_b1_x`):**
  - `shape_crossover: block`: `rng.random(3) < 0.5` for controller, structure and shape, unchanged.
  - `shape_crossover: uniform`: `rng.random(2 + n_s)`, in the order controller, structure, then each **active** shape
    gene in gene order. Elite 4 as in `phase3_b1_x`.
  - With n_s = 6 both modes are exactly B1's draws.
- **Mutation:**
  - `hit = rng.random(20 + n_s) < rate`. Controller and structure hits are as today. If any shape gene was hit, draw
    `standard_normal(n_s)` and apply the clipped Gaussian to the hit shape genes only.
  - σ = 0.25 × half-range in operator space, which is **linear u for all 7 B2 genes** (FD: "no log scale"; σ_u = 0.125).
    In physical units: c172x dihedral 0.75°, camber 0.375 %c, area / aspect 0.031.
  - The B1 chord tapers keep ln-x internally (parity with `phase3_b1` and ER).
  - Note on clipping: where the identity sits near a bound (T38 camber u 0.25), about 2 % of draws clip onto `lo` (m_eq
    −0.5, which is the gate's lower edge, see §7).
- **Gen 0 at identity:**
  - B1 sequence first: `rng.random((pop, 20))`, then `standard_normal((pop, 12))`.
  - Then `standard_normal((pop, n_s))` for the shape block, centred on FD's identity u per aircraft with
    `init_sigma_half_range` 0.25 (B1 parity).
  - `init_sigma_half_range: 0` gives a pure-identity gen 0 for verify runs.
  - Optional `init.seed_runs` from B1 elites (a B1 genome plus B2 defaults is a valid B2a genome).
- **ER parity** is required, as for B1: ER's `ga.py` / batch config must implement the same n_s-generalised draws and the
  same `locked_genes` semantics. A bitwise test is planned (`test_operators_bitwise_equal_to_er_ga` extended).

## 3. Decode / evaluate path

```
29 genes -> {controller 8, structure 12, shape 9 active}
shape: active u -> physical (genome GeneSpec, = FD arithmetic, per-aircraft lo/hi)
       + locked genes = FD schema default literal  ->  full 11-key dict
       -> FD decode_shape_b2(dict, model) -> geometry_gate_b2(pw, genes, model, envelope=ER envelope)
       -> fb2.evaluate(..., fidelity="full_a1_b2a", shape_genome=dict)
          (B2 part at default -> flexeval_b1 full_a1_b1 on <root>_v2, exact; else native increments on <root>_v2b2)
```
- **Envelope:** the gate's stall / trim / clearance checks take `envelope` (the scenario min/max speeds and altitudes).
  Genome passes **ER's** envelope from the resolved ER profile, never a genome-side table. Otherwise genome and ER could
  gate the same genome differently. Cache key = `pb2.shape_cache_key_b2(genes, model)` plus an envelope hash.
- **model_version:** `full_a1_b2a:flexv2b2a:<sha8>` vs `model_versions_post_p3b2a.json`, same warn / raise policy as
  `phase3_b1`. B1 r1 strings must be unchanged (§15.6).

## 4. t/c lock (spec only): `locked_genes`, **excluded from the encoded vector**

- **Preset field:** `"locked_genes": ["wing_tc_root_scale", "wing_tc_tip_ratio"]` (top level, next to `gene_overrides`).
- **Load checks:** every name must be an FD shape gene for this fidelity. Once FD publishes a lock flag in the JSON /
  schema, load fails if an FD-locked gene is missing from the list. Unlocking needs Corleone's approval of the energy
  cost, then a preset change.
- **Semantics:**
  - Locked genes are **not in the GA vector**: no init draw, no crossover draw, no mutation hit.
  - The evaluator injects the **FD schema default literal** (1.0), not a decoded u.
  - Records store the full named 11-key dict and `locked_genes`, so a genome migrates losslessly when the lock lifts. On
    migration, missing genes take `encode(default)`.
- **Why exclusion, not pinning:**
  1. It is structurally impossible for any operator to move a locked gene, including ER `ga.py`, seeding, clip and
     uniform crossover. A pinned u needs every path to honour a mask.
  2. Locking all 5 B2 genes gives n_s = 6, so the GA stream is **exactly** `phase3_b1` / `phase3_b1_x`. A pinned vector
     would still consume hit and crossover draws.
  3. No float dependency on re-decoding an asymmetric identity u (c172x 0.444…). It is exact today, but exclusion makes
     that irrelevant.
- **B1 bit-identity holds:** locked = default literal. With dihedral and camber at identity, FD's B2 part is at default,
  so `flexeval_b2` delegates to `flexeval_b1` (§15.6), which is exact by construction.
- **B2b:** recommend `wing_area_scale` / `wing_aspect_scale` start in `locked_genes` too, for the same drag reason (§6).

## 5. Gate and reject rules (§15.5; all hard: fail_cost, not flown, never a credit)

`geometry_gate_b2` = the B1 gate plus:

| check | c172x | T38 | 737 | f16 |
|---|---|---|---|---|
| absolute t/c(η) | 0.08–0.18 | 0.033–0.070 | 0.07–0.18 | 0.028–0.060 |
| absolute m_eq(η) %c | −0.5 … 4.5 | −0.5 … 2.0 | 0.5 … 4.0 | 0 … 3.0 |
| \|Δm_tip − Δm_root\| | ≤ 3.0 %c | same | same | same |
| \|θ_eff\| = \|θ_B1 + 2Δm − 2Δm_r\| | ≤ 8° | same | same | same |
| tip clearance at bank φ_g, gear static | n/a (high wing) | 8°, ≥ 1 ft | 6°, ≥ 1.5 ft | 8°, ≥ 1 ft |
| span b′ (B2b) | ≤ 41 ft | ≤ 29 ft | ≤ 118 ft | ≤ 35 ft |

The table above is followed by these checks:
- stall: 1.2·V_s(W_max, CLmax′) ≤ min scenario speed;
- trim pre-check: |Δδe| ≤ 70 % of travel at min and max speed, and α_trim′ inside the table range;
- JSBSim `trim_failed` (throttle ≤ throttle_max) stays the authoritative final check.

FD decode errors still raise `ValueError`: out of range, unknown keys, B2b keys off default in B2a, NaN.

Genome side:
- `geometry_rejected(res)` recognises the B2 statuses.
- The gene box must be gate-feasible by construction. FD scans it; genome repeats the corner scan on c172x and T38 with
  ER's envelope.
- An in-box reject is reported to FD (FD narrows the range and never loosens the gate). It is not absorbed as a GA cost.

## 6. Drag / energy dependency: **APPROVED by Corleone (2026-10-06 8:19 PM PT)**

> **Status 8:20 PM PT:** Corleone approved the energy cost. ER owns the term; FD supplies the signal (energy export) and
> weight guidance. `wing_tc_*` unlock **only after** FD's export + `locked_until: "energy_cost"` flag are in §15/JSON,
> ER's term and weight land, and new pins are taken. FD confirmed all 8 §7 items go into the B2a build (inclusive gate
> with 1e-9 tolerance, `encode_shape_b2` helper, envelope hash in the cache key, reflex/CLmax/dihedral probes). B2a pins
> + `planform_b2` ETA ~1:30 AM PT. B2b size stays locked (not built tonight).

All B2 drag goes into JSBSim natively: camber polar, form and wave drag (B2a), induced drag and size CD0 (B2b). But
`evolution/sim.py` holds speed with a closed-loop throttle (`thr_kp` 0.05, `thr_ki` 0.01), and `effort` is elevator
total variation only. So **drag never reaches the scalar** unless the throttle saturates or trim fails. FD (message, not
yet in §15 or the JSON) is adding an **energy export outside TERM_KEYS**: drag-power ratio vs a frozen baseline, plus mean
throttle. FD recommends that ER add an energy term from it.
- Until Corleone approves that term (ER sim cost and weight; genome's preset weights must equal ER's profile, as today),
  **`wing_tc_*` stay locked**.
- Unlocking means: remove them from `locked_genes`, add the energy weight to the ER profile and the genome preset, take
  new pins, and re-run the sweeps (§8, V6).
- Dihedral and camber run unlocked in B2a (FD: "probably safe", confirming numerically). Camber's drag-polar closure is
  also throttle-absorbed. Its remaining closures are trim authority, downwash, the stall gate and one-sided design
  torque. Expect near-neutral drift (an uninformative value) rather than an exploit until the energy term exists.

## 7. Inconsistencies / GA-side risks in §15 and the JSON (for FD)

1. **The lock and the energy export are not in §15 or the JSON.** The JSON has no `locked` / `default_locked` field, and
   §15 never mentions an energy export or throttle absorption. Ask FD to publish both (a JSON flag plus export key names
   and units) so genome and ER read the lock from FD instead of inventing it.
2. **§15.4 closures that rely on native drag are not closures for the scalar.** Thick t/c ("form + wave drag"), camber
   off-design ("profile-drag polar"), AR ("free induced drag") and area (CD0) all depend on drag the autothrottle
   absorbs. The mass and strength closures are real. In B2b, a **small area** gets a signed `J_mass` credit plus a lower
   gust response, with only the stall gate pushing back. A **small AR** gets a lower root BM and less mass with invisible
   induced drag. Both are likely bound pile-ups, so B2b size needs the same lock / energy term as t/c.
3. **§15.7 test A encodings:** it lists "3 encodings (None / {} / defaults / zero-vector-of-defaults)", which is four
   items. With asymmetric ranges a zero vector decodes to all-`lo` (dihedral −3, t/c 0.8, camber −1), not to defaults. It
   should be the encoded-default vector (u = 0.5 / 0.444 / 0.333 / 0.4 …). The B2a → B2b vector padding must likewise use
   u = 0.4, not 0 or 0.5.
4. **Box corners sit exactly on gate limits.** c172x |Δm_tip − Δm_root| reaches exactly 3.0 (−1 vs +2), and T38 m_eq
   reaches exactly −0.5 (lower limit). Both are fine only if the comparisons are inclusive and use the decoded floats.
   Otherwise a legal corner gets a hard 2000. Add both corners to FD's scan test explicitly.
5. **Reflex camber:**
   - The T38 range reaches m_eq −0.5 (reflexed) on a 0 baseline. Negative camber also appears on the c172x (down to 0.8)
     and f16 (down to 1.1).
   - The one-sided rule covers **design** torque only. The flown `J_wing_torque_peak` / `J_twist` see the real nose-up
     relief, so a small flown-term credit for reflex is possible.
   - Watch for it in the camber sweep.
6. **Dihedral has no lateral scenarios:**
   - Flights have a fixed-gain wing leveller (`roll_kp` / `roll_kd` not genes) and rudder = 0. Aileron activity is not
     in `effort`, and `comfort` has no lateral channel. The heading weight is 0.01.
   - So ΔΓ is seen only through heading and the flown β-loads (`dih_Q`, ∝ |ΔΓ|, delta only).
   - §15.5 has no Dutch-roll or spiral check.
   - Expect neutral drift with a weak pull to 0. Values are uninformative until lateral scenarios exist.
   - Γ0·β loads are absent at every fidelity (737 Γ0 = 6°), so the structural minimum sits at ΔΓ = 0, not at the
     physical optimum.
   - Check for a sign asymmetry against the existing elastic β·w′ term: a one-sided `J_bm_rms` credit for one sign of ΔΓ
     would be an exploit.
7. **CLmax wording:** "CLmax only enters the stall gate" holds for t/c. For camber, §15.2 flies half of the ΔCL0 above
   the stall fade (CLmax gain ½·ΔCL0). That is fine, but camber has a flown stall effect.
8. **The gate depends on `envelope`,** but `shape_cache_key_b2` does not include it. Two profiles with different speeds
   can share a cache key with different gate results. The key or the cache must include the envelope.
9. Minor: the JSON gives `vector_index_after_b1` but no absolute index (6 + i). Gen-0 clip mass at asymmetric bounds (§2).

## 8. Verify plan (genome side, after FD lands B2a code and pins)

- **V0 schema:** the pin equals the JSON and `pb2.shape_schema_b2(model)` for all 4 aircraft. Encoding string matches.
- **V1 encoding:** genome decode/encode vs `decode_shape_b2` / `encode_shape_b2` on ≥ 20 000 vectors per aircraft (random,
  corners, identity): 0 mismatches. Identity → FD defaults exactly.
- **V2 B2a at defaults ≡ B1 r1, bit for bit.**
  - Cases: the `p3b1_verify` set on c172x via the `phase3_b2a` Task at `full_a1_b2a`, compared against `full_a1_b1` r1 in
    the same process. That is baseline 0.2419400885448951, soft tip 0.25879473336084763, washout, tip chord, sweep, the
    combo, twist_mid ±, and ER's 6 shapes (`CROSSCHECK_p3b1.md`).
  - Encodings: dict with B2 keys at defaults, missing keys, encoded-default vector, and lock-injected.
  - Pass = cost, 24 terms, margins, mass and per-scenario values with max |Δ| = 0.0. `model_version` / fidelity labels
    differ by design.
  - Also FD's forced-native-path check D, if exposed.
- **V3 GA stream:** `locked_genes` = all 5 B2 genes gives gen 0 plus 3 generations bit-identical to `phase3_b1` and
  `phase3_b1_x` (same seed, arrays and costs). With t/c locked, the first 20 gen-0 columns equal `phase2_flex`'s.
- **V4 pins:** `model_versions_post_p3b2a.json` matches live. The B1 r1 / A1 strings are unchanged.
- **V5 single-gene sweeps (exploit catch; c172x and T38 first, then 737 and f16):**
  - Genes: dihedral (9 points lo → hi), camber root, camber tip, uniform camber (root = tip) and anti-symmetric camber
    (root = −tip). Each uses the best `phase3_b1` genome and the 3 ER scenarios.
  - Report Δcost by term, trim α / elevator, the gate status and FD's energy export.
  - Flags: a monotone slide to a bound with |Δcost| > 1e-4 is a **blocker**. Flat within 1e-4 means **neutral drift**
    (log the value as uninformative; this is §15.7 item 8).
  - Specific checks: a sign asymmetry in `J_bm_rms` for ±ΔΓ, `J_wing_torque_peak` relief for reflex, and the energy
    export rising while the cost stays flat (documents the throttle gap).
- **V6 t/c diagnostic (locked genes, study only, `via="fd"`):** sweep τ_r and the tip ratio to measure what the lock
  prevents, i.e. the cost gain from thick plus lower `wing_ei_root`. This is evidence for Corleone's decision; nothing
  from it enters a run.
- **V7 gate corners:** the 2⁵ B2a corners × the B1 corners on c172x and T38 with ER's envelope give zero in-box rejects,
  including the two exact-boundary corners from §7.4.

## 9. Open questions

**FD:**
1. Will you publish the t/c lock flag and the energy-export keys and units in §15 and the JSON?
2. Will the gate comparisons be inclusive at the c172x camber-gradient and T38 m_eq corners?
3. Will you fix test A's "zero-vector-of-defaults"?
4. Should the envelope be part of the cache key?
5. Do you agree B2b size should start locked until the energy term exists?
6. Reflex flown-torque credit: OK, or one-sided in the flown terms too?
7. Dihedral: add a Dutch-roll / spiral pre-check, or accept neutral drift until lateral scenarios exist?

**Corleone:** ~~approve the ER energy term?~~ **Approved 8:19 PM PT.** `wing_tc_*` unlock once FD's export and ER's term
land (then 11 shape genes, 31 total). Until then B2a flies dihedral and camber only (9 shape genes, 29 total).
