# Flight-model quality check: bundled JSBSim 1.3.1 aircraft (jets + props)

Outputs and scripts:
- `fm_quality.csv` / `fm_quality.jsonl` hold the data; `FM_QUALITY_TABLE.md` is the per-model table built by `make_fm_report.py`.
- `fm_quality.py` is the driver and `fmq_worker.py` the per-model subprocess.
- Phase-1 aircraft deep check: `phase1_check.py` → `phase1_check.json` (see INTERFACE.md §4).
- Inputs read-only from `flight-sim-plan/` (`$FLIGHT_SIM_PLAN_DIR`, an external planning folder not included in this repo): cases.txt (speeds and altitudes), inventory_raw.json, aircraft_inventory.csv, flytest_results.jsonl.

## What is checked (41 powered models)

1. **Loads.** FlightGear-only properties a model references are auto-created at 0 and reported.
2. **Trims.** Built-in full trim (`do_simple_trim = 1`) at the cases.txt speeds and altitude.
   - Fallback ladder: mode 0, 0.85×/1.2×/0.7× speed, half altitude, then an engine-off Newton glide trim.
   - Recorded: residuals, α, throttle, pitch trim, plus a 10 s open-loop hold (Δθ, ΔKCAS, roll-off).
   - **Gear is commanded UP before trim.** All models load gear-down. The earlier gear-down run is kept as `fm_quality_geardown.*`.
3. **Flies a step cleanly.**
   - ±0.1 elevator doublet (1–3 s), flown to 25 s with the repo's wing leveler. Envelope watchdog on pitch, α and Nz; q-oscillation decay ratio.
   - Closed-loop 3° pitch step with a PID auto-scaled by the measured pitch-rate gain. Pass = final error < 0.5°, overshoot < 2°, no envelope violation.
4. **Deterministic.** The doublet run is repeated and arrays are compared bit-for-bit.
5. **RTF.** Pure stepping and with the Python control loop, measured with 6 worker processes in parallel on the shared box. Single-process numbers are higher.

Rating rules (`fm_quality.rate`):
- **avoid**: no trim, an envelope excursion in the doublet, or non-deterministic.
- **usable-with-caveats**: any of
  - glide-trim only;
  - off-nominal trim point;
  - FlightGear properties needed;
  - trim α outside −3…10°;
  - trim throttle > 0.95;
  - generic pitch step fails;
  - q-decay > 0.8;
  - roll-off > 10° in 10 s;
  - RTF < 50;
  - author marks the model ALPHA.
- **good-to-evolve**: none of the above.

## Result: 18 good-to-evolve · 18 usable-with-caveats · 5 avoid; all 41 tested models deterministic

### Shortlist (good-to-evolve; numbers at the trim point; RTF measured with 6 workers in parallel)

| model | type | trim | α° | thr | pitch-rate gain °/s/unit | 3° step: err° / ovs° / t90 s | RTF pure / loop |
|---|---|---|---|---|---|---|---|
| **c172x** (Phase 1) | piston | 100 KCAS / 4 kft | 0.79 | 0.78 | −26.3 | 0.27 / 1.03 / 0.95 | 510 / 184 |
| **T38** (Phase 1) | AB jet | 300 / 10 kft | 4.61 | 0.354 (pos 0.71 = 71 % MIL) | −39.7 | 0.08 / 0.32 / 2.33 | 1200 / 771 |
| **737** (Phase 1) | airliner | 250 / 10 kft | 3.28 | 0.586 | −15.8 | 0.10 / 0.30 / 2.42 | 571 / 324 |
| c172p | piston | 100 / 4 kft | 0.39 | 0.72 | −64.3 | 0.43 / 0.06 / 2.75 | 309 / 335 |
| c172r | piston | 100 / 4 kft | 1.48 | 0.805 | −63.3 | 0.50 / 0.08 / 2.97 | 416 / 288 |
| c182 | piston | 120 / 5 kft | −0.29 | 0.88 | −45.9 | 0.20 / 0.35 | 629 / 285 |
| DHC6 | turboprop | 140 / 5 kft | −2.16 | 0.927 | −32.2 | 0.12 / 0.28 | 920 / 515 |
| OV10 | turboprop | 150 / 3 kft | 4.95 | 0.728 | −32.1 | 0.17 / 0.30 | 1374 / 849 |
| t6texan2 | turboprop | 200 / 10 kft | −0.04 | 0.668 | −83.4 | 0.30 / 0.32 | 563 / 481 |
| T37 | jet trainer | 200 / 10 kft | 1.29 | 0.776 | −32.2 | 0.09 / 0.35 | 864 / 674 |
| F80C | jet | 250 / 10 kft | 4.58 | 0.821 | −31.2 | 0.09 / 0.31 | 1297 / 838 |
| f15 | AB jet | 350 / 10 kft | 1.73 | 0.618 | −34.9 | 0.02 / 0.34 | 810 / 366 |
| f16 | AB jet (FBW) | 350 / 10 kft | 1.05 | 0.284 | −32.0 | 0.02 / 0.39 | 318 / 211 |
| A320 | airliner | 250 / 10 kft | 3.08 | 0.611 | −20.0 | 0.21 / 0.24 | 326 / 256 |
| 787-8 | airliner | 250 / 10 kft | 3.13 | 0.518 | −16.9 | 0.07 / 0.22 | 810 / 355 |
| MD11 | airliner | 250 / 10 kft | 4.34 | 0.483 | −7.3 | 0.03 / 0.89 | 1132 / 342 |
| Concorde | AB SST | 300 / 10 kft | 4.75 | 0.652 | −11.0 | 0.06 / 0.35 | 433 / 327 |
| Boeing314 | 4-piston flying boat | 130 / 5 kft | 2.72 | 0.712 | −16.1 | 0.07 / 0.40 | 294 / 324 |

Suggested evolution set:
- c172x as the GA reference;
- T38 and 737 for Phase 1;
- next, c182 or c172r (GA props), t6texan2 or T37 (trainers), A320 or 787-8 (airliners) and f15/f16 (fighters).
  - f16 has its own FBW FCS: the GA tunes an outer loop around an already-stabilised aircraft.

Phase-1 notes:
- The c172x has the lightest pitch damping of the shortlist (q decay ratio 0.586; doublet Δθ 10.9°). It passes, but it is the hardest to stabilise of the three. It is also the repo's reference aircraft, so it stays.
- T38 throttle-cmd 0.5 = MIL; above that is afterburner.
- 737 built-in trim needs ≥ 190 KCAS at 10 kft (INTERFACE gotcha 4).

### usable-with-caveats (18)

| model | caveat |
|---|---|
| A4, B747, Camel, F4N, global5000, J3Cub | author marks model ALPHA (otherwise clean) |
| f104 | ALPHA, plus needs FG property `systems/radar/range` |
| fokker100 | needs FG-only properties (`/gear/gear/wow`, `/sim/model/pushback/*`) |
| L17 | needs FG property `fcs/flaps-pos-deg` |
| B17, C130, L410, p51d, Short_S23 | built-in powered trim fails ("udot doesn't appear to be trimmable"; magneto/advance/spin-up did not help). Engine-off Newton glide trim only. B17 is also ALPHA |
| dr1 | glide-trim only, plus FG properties |
| c310 | trim throttle 0.963 at 130 KCAS (little thrust margin) |
| pa28 | trims (80 KCAS, thr 0.91), but the generic auto-scaled pitch step leaves a 1.12° final error. It is under-tuned, not unstable |
| pogo-jsbsim | only trims at 105 KCAS with α 10.8°; pitch step α-excursion; ALPHA |

### avoid (5)

| model | reason |
|---|---|
| XB-70 | Nz excursion at 11.5 s after a small doublet; q oscillation grows ×28.6 (unstable) |
| fokker50 | glide trim only; pitch excursion at 8.2 s |
| pc7 | glide trim only; pitch excursion at 6.2 s; rolls off 28.5° in 10 s |
| wrightFlyer1903 | pitch excursion at 3.3 s (unstable by design, 26 KCAS) |
| f22 | no trim at any rung of the ladder, glide included |

### Gear-up vs gear-down (both runs kept)

Retracting the gear before trim changed these ratings:
- **F80C** (thr 0.99 → 0.82), **OV10** (0.983 → 0.728), **T37** (0.973 → 0.776) and **t6texan2** (0.953 → 0.668): caveat → good.
- L410: no trim → glide-trim caveat.
- B17/C130/p51d glide trims keep working once the gear is retracted by stepping first. Gotcha: `run_ic` advances actuators by one dt per call.
- T38 trim throttle 0.435 → 0.354, 737 0.690 → 0.586.

### Caveats of this check

- Tested at one point per model (the cases.txt speeds and altitude), with generic controllers. "Avoid" means "not without model-specific work", not "broken".
- RTF numbers come from 6 workers running in parallel on a shared box.
- The pitch-step PID is auto-scaled, not tuned, so a failure there is a soft signal.
- No lateral-directional quality metric beyond the roll-off in the 10 s hold. The wing leveler masks spiral modes during the doublet.
