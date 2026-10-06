# Hand-off to Evolution Runner: Phase-1 heading hold (2026-10-06)

You can apply this file on its own, without reading genome/ code. It adds one outer loop, two genes and one cost term
to the Phase-1 task you already run bit-for-bit with genome/ (evolution/runs/phase1_report.md). Nothing else changes.

## 0. Why

- **Symptom:** the c172x drifts about 25° in heading over 90 s on every seed; the jets drift 0°.
- **Root cause:** propeller torque reaction plus a P-only wing leveller.
  - At trim, JSBSim balances the prop torque (`moments/l-prop-lbsft` +235 lb·ft roll, `n-prop` +99 lb·ft yaw) with
    aileron −0.075 and rudder −0.004.
  - The controller then overwrites the aileron with `−0.05·φ − 0.02·p`, which has no trim term and no integrator. It
    settles where `−0.05·φ = −0.075`, i.e. φ ≈ +1.5°, with sideslip ≈ 0.
  - That standing bank turns the aircraft at g·tanφ/V ≈ 0.28°/s, which is 25° in 90 s.
  - The jets have no prop torque, so their trim aileron is 0 and they hold φ = 0.
- **Fix:** an outer heading loop that commands bank into the existing wing leveller.

## 1. Profile changes

### 1a. `genome/aircraft_profiles/phase1_shared.json` (the set of record): blocks added

```json
"task": {
  "heading_hold": {
    "genes": ["kp_hdg", "ki_hdg"],
    "heading_target": "initial heading (psi at t=0, 0 deg true in every scenario)",
    "error": "e_psi = wrap180(psi_target - psi), wrap180(a) = (a + 180) % 360 - 180",
    "law": "phi_cmd = clip(kp_hdg*e_psi + ki_hdg*I, +-bank_limit_deg); I += e_psi*dt only if ki_hdg > 0, clipped to +-hdg_i_limit_deg/ki_hdg",
    "inner": "aileron = clip(-kp_roll*(phi - phi_cmd) - kd_roll*p, +-0.5)",
    "hdg_i_limit_deg": 10.0,
    "cost_term": "track_heading_rms",
    "hdg_rms_ref_deg": 5.0,
    "weight": 0.01
  }
},
"aircraft": {
  "c172x": {"heading_hold": {"enabled": true, "bank_limit_deg": 16.0,
            "gain_bounds": {"kp_hdg": {"min": 0.05,  "max": 5.0,  "scale": "log"}, "ki_hdg": {"min": 1e-05,   "max": 0.1,   "scale": "log0"}}}},
  "t38":   {"heading_hold": {"enabled": true, "bank_limit_deg": 25.0,
            "gain_bounds": {"kp_hdg": {"min": 0.164, "max": 16.4, "scale": "log"}, "ki_hdg": {"min": 3.29e-05, "max": 0.329, "scale": "log0"}}}},
  "b737":  {"heading_hold": {"enabled": true, "bank_limit_deg": 25.0,
            "gain_bounds": {"kp_hdg": {"min": 0.137, "max": 13.7, "scale": "log"}, "ki_hdg": {"min": 2.74e-05, "max": 0.274, "scale": "log0"}}}}
}
```

(The file also carries `_why` notes.)

- **Bank limit:** min(standard-rate bank at the trim TAS, 25° autopilot convention), rounded down. Standard-rate bank
  is atan(V·3°/s / g): c172x 106 KTAS → 16.2° → 16; T38 349 KTAS → 43.8° → 25; 737 291 KTAS → 38.6° → 25.
- **Gain bounds:** the c172x reference range × V/V_ref (TAS at trim), the same rule as the other outer loops. Turn rate
  is g·tanφ/V, so bank per degree of heading error scales with V. Factors: T38 3.290, 737 2.742. Values are given to
  3 significant figures.
- **Weight 0.01:** chosen from a c172x sweep (seed 1, pop 48 × 40):

  | weight | final drift |
  |---|---|
  | 0 | 15° |
  | 0.003 | 5° |
  | 0.01 | 0.7° |
  | 0.03 | 0.7° |
  | 0.1 | 0.6° (track_alt 0.0599) |

  0.01 is the smallest weight that holds within 1°, and track_alt does not get worse. Once the heading is held, the
  term is about 1 % of the cost.

### 1b. `genome/exports/evolution_phase1_profiles.json` (regenerated with `$PY genome/export_shared_profiles.py`)

Changes per profile against the config of your `phase1` run:
- **New keys** (not accepted by your `Profile` yet; listed in the file's `_needs_code`):

  | profile | heading_hold | bank_limit_deg | hdg_i_limit_deg | w_heading | hdg_rms_ref_deg |
  |---|---|---|---|---|---|
  | phase1_c172x | true | 16.0 | 10.0 | 0.01 | 5.0 |
  | phase1_T38 | true | 25.0 | 10.0 | 0.01 | 5.0 |
  | phase1_737 | true | 25.0 | 10.0 | 0.01 | 5.0 |

- **`gain_bounds` additions:**

  | profile | kp_hdg | ki_hdg |
  |---|---|---|
  | phase1_c172x | [0.05, 5.0] | [1e-05, 0.1] |
  | phase1_T38 | [0.164, 16.4] | [3.29e-05, 0.329] |
  | phase1_737 | [0.137, 13.7] | [2.74e-05, 0.274] |

- **`gene_kinds` addition:** `"ki_hdg": "log0"` (kp_hdg is plain `log`).
- **Precision fix:** `thr_kp` / `thr_ki` are now written at full precision. You had already overridden them.

Checked read-only against your loader (`sim.Profile.from_dict` + `genome.make_schema`, with bytecode writing off): after
removing the heading keys and genes, every profile is accepted and equals the `profiles` block of
`evolution/runs/phase1-s1/config.json` exactly (no differing keys).

**F-16** (yours, not in the shared set): set `"heading_hold": false`. Its `aileron-cmd-norm` is a roll-rate demand to
the FBW FCS (FD INTERFACE.md §5d), so the wing-leveller and heading gains need their own derivation. If you want
placeholders anyway, the rule gives bank 25°, factor ≈ 3.84 (≈ 407 KTAS at 350 KCAS / 10 kft), so kp_hdg 0.192–19.2
and ki_hdg 3.84e-5–0.384. These are unverified.

## 2. Code changes in evolution/

Line numbers refer to your files as of 04:23–04:25 PT today.

### 2a. `sim.py`: `Profile` (dataclass at ~line 60)

Add fields:

```python
    # heading hold (Phase-1, 2026-10-06): heading error -> bank cmd -> wing leveller
    heading_hold: bool = False
    bank_limit_deg: float = 25.0
    hdg_i_limit_deg: float = 10.0
    w_heading: float = 0.0              # cost += w_heading * RMS(e_psi) / hdg_rms_ref_deg
    hdg_rms_ref_deg: float = 5.0
```

In `from_dict`, validate: `0 < bank_limit_deg < max_abs_phi_deg`, `hdg_i_limit_deg > 0`, `hdg_rms_ref_deg > 0`, and
`w_heading >= 0`.

### 2b. `sim.py`: `simulate()` (~line 598)

Set up before the loop, next to `roll_kp, roll_kd = P.roll_kp, P.roll_kd`:

```python
    hh = P.heading_hold
    if hh:
        kp_h, ki_h = gains["kp_hdg"], gains["ki_hdg"]
    i_hdg = 0.0
    c_psi: List[float] = []             # psi per step flown (deg), for the cost term
```

Inside the loop, replace the wing-leveller line (~704, `ail = min(max(-roll_kp * phi - roll_kd * p, -0.5), 0.5)`) with:

```python
        phi_cmd = 0.0
        if hh:
            psi = fdm["attitude/psi-deg"]                 # 0..360
            e_psi = (0.0 - psi + 180.0) % 360.0 - 180.0   # wrap180(psi_target - psi), psi_target = 0 (initial heading)
            if ki_h > 0:
                lim = P.hdg_i_limit_deg / ki_h
                i_hdg = min(max(i_hdg + e_psi * DT, -lim), lim)
            phi_cmd = min(max(kp_h * e_psi + ki_h * i_hdg, -P.bank_limit_deg), P.bank_limit_deg)
            c_psi.append(psi)
        ail = min(max(-roll_kp * (phi - phi_cmd) - roll_kd * p, -0.5), 0.5)
```

- **Placement:** this sits after the envelope checks and before the `fdm[...] = ...` writes, the same step as the other
  loops. ψ is read at the start of the step, like h and θ.
- **No change with the flag off:** `phi - 0.0` is exactly `phi`, so heading_hold = false stays bit-identical to today.
- **Heading target:** genome/ uses the scenario's heading target (default 0 = the IC heading `ic/psi-true-deg = 0`).
  Your scenarios have no heading steps, so the constant 0.0 is equivalent.

Cost at the end, after the comfort term (keep this order for bit-identity with genome's weighted sum
`track + 2·effort + 0.05·comfort + 0.01·heading`):

```python
        if hh and P.w_heading > 0:
            e = (0.0 - np.asarray(c_psi) + 180.0) % 360.0 - 180.0   # exactly genome's numpy form
            heading_rms = float(np.sqrt(np.mean(e ** 2))) / P.hdg_rms_ref_deg
            cost = cost + P.w_heading * heading_rms
```

Add `out["heading_rms"] = heading_rms` (and, if you like, `out["hdg_final_deg"]`).

- genome/ computes this term with numpy over the recorded ψ array of the steps flown
  (`fitness.py` objective `track_heading_rms`): `e = (psi_target - psi + 180) % 360 - 180` on the array, then
  `sqrt(mean(e**2)) / 5`, where `np.mean` uses pairwise summation. Use the numpy form above, not a running Python sum
  or the scalar in-loop `e_psi`, or the last bits can differ.
- genome's in-loop controller uses the Python scalar form shown above.
- Failed runs keep the fail cost; no heading term is added.

### 2c. `genome.py`: two optional genes

```python
HEADING_GENES: List[Gene] = [
    Gene("kp_hdg", 0.05, 5.0, units="deg bank/deg hdg", doc="outer heading P: bank cmd per deg heading error"),
    Gene("ki_hdg", 1e-5, 0.1, kind="log0", units="deg bank/(deg*s)", doc="outer heading I"),
]
```

- **Signature:** `make_schema(bounds=None, kinds=None, heading_hold=False)`. It returns the 6 pitch genes and, when
  heading_hold is set, appends `HEADING_GENES` in that order: **[kp_alt, ki_alt, kd_alt, kp_pitch, ki_pitch, kd_pitch,
  kp_hdg, ki_hdg]**. That order is genome's canonical layout, and you need it for bit-identical GA runs.
- **Bounds and kinds:** validate them against the names of the schema actually built (so `kp_hdg` in `gain_bounds`
  without heading_hold is an error).
- **Keep `GENE_NAMES` / `N_GENES` (= 6):** they are the legacy default only.

### 2d. `batch.py`

- **Line 113 and line 300:** `genome.make_schema(prof.gain_bounds, prof.gene_kinds, prof.heading_hold)`.
- **Line 315:** `ga.generation_zero(rng, pop_size, genome.N_GENES)` → use `len(st["schema"])`. With 8 genes, a
  hard-coded 6 would silently drop the heading genes.
- **Lines 409–416:** `best_gains`, `genes_at_bound`, `gain_bounds` and `gene_kinds` already iterate over the schema,
  so nothing to do there.

### 2e. Cache

Nothing to do. `cache.py` keys on the resolved profile and the sha of sim.py + genome.py, so the new keys and code
changes miss automatically.

### 2f. Optional reporting (`phase1_report.py`)

Heading drift per scenario: `wrap180(psi_end - psi_0)` and max |Δψ|. ψ is already a recorder channel.

## 3. How to check you match genome/ bit-for-bit

Re-fly these best genomes (3 scenarios, scenario_seed 1, the exported profiles) and compare costs. genome/ gets them
with `adapter.load_task("phase1_default", {"aircraft": ...})`.

| aircraft (genome/ run) | genes (kp_alt, ki_alt, kd_alt, kp_pitch, ki_pitch, kd_pitch, kp_hdg, ki_hdg) | cost (mean of 3) | per-scenario costs |
|---|---|---|---|
| c172x (`runs/hdg_after_c172x_s1`) | 0.2397326688652956, 0.05, 0.38523085001510937, 0.07049452717617341, 1.9653025820764143e-05, 0.014608444547277827, 2.242870209249964, 0.0003706638173012445 | 0.19600587132367517 | 0.10529427560176219, 0.31566235928965425, 0.167060979079609 |
| T38 (`runs/hdg_after_t38_s1`) | 0.22365803353253752, 4.1598113596235896e-07, 0.0941094102393032, 0.06574330468426912, 0.0, 0.0024262902418641465, 1.945038464385377, 0.00475971406828075 | 0.0966245736792507 | 0.07072963520626355, 0.12908011182321508, 0.0900639740082735 |
| 737 (`runs/hdg_after_b737_s1`) | 0.19092554817716106, 0.030997595118024638, 0.34324408429885966, 0.05970803545301804, 0.005608195701037316, 0.036460842911213105, 0.23541201575286405, 0.008622211397029298 | 0.10678211633774594 | 0.07842957400564782, 0.1370541422493403, 0.10486263275824974 |

Genome order: [kp_alt, ki_alt, kd_alt, kp_pitch, ki_pitch, kd_pitch, kp_hdg, ki_hdg]. The normalized genome is in each run's
`best_gains.json` (`genome`) under `flight_sim_3d/genome/runs/<run>/`, if you prefer to re-decode them with your schema.

To check the flag-off path: with `heading_hold: false` (or the keys removed), your Phase-1 rerun genomes must keep
their current costs exactly. genome/ checks the same on its v4 genomes (`tests/test_heading.py`).

## 4. Results on genome/ (for context)

Heading hold before/after (genome/heading_report.py). c172x: pop 48 × 40, 3 seeds; evolve.py sets
scenario seed = GA seed, so each before/after pair shares its scenario set. Jets: pop 32 × 20. "Before" = phase1_default
with `heading_hold: false` (= v4 task, bit-identical). Drift = ψ(90 s) − ψ(0) in calm / wind 1 / wind 2. Max |Δψ| is
dominated by the weathercock transient when the steady wind starts at t = 0 in the windy scenarios.

| run | best cost | cost w/o heading term | track_alt | final drift ° | max abs Δψ ° | max pitch ° calm / worst | nz calm / all | kp_hdg / ki_hdg | genes at/near bounds |
|---|---|---|---|---|---|---|---|---|---|
| c172x before s1 (v4) | 0.2063 | 0.2063 | 0.0610 | +25.4 / +19.4 / +27.1 | 27.1 | 5.1 / 6.1 | 0.89–1.11 / 0.52–1.47 | – | none |
| c172x before s2 | 0.2357 | 0.2357 | 0.0575 | +25.6 / +29.2 / +22.9 | 29.2 | 5.5 / 6.9 | 0.86–1.17 / 0.44–1.52 | – | kd_alt near upper |
| c172x before s3 | 0.1957 | 0.1957 | 0.0763 | +25.5 / +28.1 / +28.3 | 28.3 | 4.9 / 5.2 | 0.90–1.10 / 0.60–1.39 | – | ki_pitch zeroed |
| **c172x after s1** | 0.1960 | 0.1941 | 0.0586 | +0.7 / +0.8 / +0.7 | 9.3 | 4.7 / 5.9 | 0.89–1.10 / 0.44–1.55 | 2.24 / 3.7e-4 | ki_alt upper (0.05), ki_pitch near lower |
| **c172x after s2** | 0.1869 | 0.1856 | 0.0547 | −0.1 / +0.1 / +0.1 | 5.1 | 4.7 / 5.9 | 0.89–1.10 / 0.44–1.44 | 1.30 / 0.10 | ki_alt upper, ki_hdg upper (0.1) |
| **c172x after s3** | 0.1977 | 0.1964 | 0.0709 | −0.1 / −0.0 / +0.1 | 4.6 | 4.9 / 5.1 | 0.89–1.11 / 0.55–1.39 | 1.60 / 0.047 | ki_hdg near upper |
| T38 before s1 (v4) | 0.0918 | 0.0918 | 0.0282 | +0.0 / −1.9 / +0.4 | 2.9 | 6.1 / 6.4 | 0.85–1.13 / 0.68–1.22 | – | ki_alt zeroed |
| T38 after s1 / s2 / s3 | 0.0966 / 0.0930 / 0.0925 | 0.0963 / 0.0926 / 0.0921 | 0.033 / 0.031 / 0.028 | 0.0 | 2.7–2.8 | 6.1 / 6.3–6.4 | 0.85–1.14 / 0.68–1.22 | 1.3–1.9 | s3: ki_hdg near lower |
| 737 before s1 (v4) | 0.1102 | 0.1102 | 0.0391 | +0.0 / −2.2 / +0.4 | 2.8 | 5.2 / 5.2 | 0.85–1.14 / 0.72–1.20 | – | ki_alt zeroed |
| 737 after s1 | 0.1068 | 0.1059 | 0.0322 | +0.0 / +0.8 / −0.2 | 2.8 | 5.3 / 5.4 | 0.84–1.15 / 0.71–1.21 | 0.235 / 0.0086 | ki_alt near upper |

**c172x means over 3 seeds:**

| | cost | track_alt | final drift |
|---|---|---|---|
| before | 0.2126 | 0.0649 | 19–29° |
| after | 0.1935 (0.1920 without the heading term) | 0.0614 | ≤ 0.8° |

Holding the heading also helps altitude tracking: no standing bank, so no lift loss.

**Jets:**
- **T38:** Evolution Runner's flag-off s2/s3 on the same task gave 0.0938 / 0.0928 (scenario seed 1). Our flag-on
  s2/s3 used the same scenario seed and gave 0.0930 / 0.0925. s1 is 5 % worse (0.0966), which looks like GA search
  noise with 2 extra genes at a small budget.
- **Existing v4 genomes, flag on** (default heading gains, ki 0): T38 0.0918 → 0.0922 and 737 0.1102 → 0.1103
  (without the heading term), track_alt unchanged.

**Weight sweep** (c172x seed 1):

| weight | final drift | cost w/o heading term | track_alt |
|---|---|---|---|
| 0 | 15° (kp_hdg goes to its lower bound) | 0.1943 | 0.0632 |
| 0.003 | 5.4° | 0.1954 | 0.0657 |
| **0.01 (default)** | 0.7° | 0.1941 | 0.0586 |
| 0.03 | 0.8° | 0.1923 | 0.0542 |
| 0.1 | 0.6° | 0.1919 | 0.0599 |

The ki_hdg pin at 0.1 (s2) sits on a flat optimum: re-flying that genome with ki_hdg 0.03 / 0.1 / 0.2 / 0.4 gives
0.1875 / 0.1869 / 0.1870 / 0.1892, so the bound was not widened.

