# Benchmark b1

## bench_baseline.json  (run `bench_baseline-b1`)

batch wall 58.6 s on 8 workers, 2706 sims computed (46.2 sims/s); cache rerun: 1.0 s, hit rate 1.000, 0 sims computed, identical results: True; box load avg (1 min) at start/end 9.1/13.1; sequential schedule (cache off): 61.5 s, identical: True

| aircraft | profile | best fitness (gen0 -> final) | hold RMS ft (calm / mean) | hold max abs ft | overshoot ft | settle s (+-20 ft) | invalid gen0 / all gens | sims | cpu s/sim | aircraft wall s | sims/s | rerun hit rate | genes at bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c172x | baseline | 0.3192 -> **0.1923** | 3.13 / 2.95 | 7.39 | 6.9 | 7.9 | 0.250 / 0.033 | 1341 | 0.200 | 57.8 | 23.2 | 1.000 | kp_alt@max |
| f16 | baseline | **not run: trim_failed: TrimFailureError: Trim Failed** | | | | | | | | | | | |
| 737 | baseline | **not run: trim_failed: TrimFailureError: Trim Failed** | | | | | | | | | | | |
| t6texan2 | baseline | 0.2013 -> **0.1690** | 0.48 / 1.16 | 4.12 | 5.4 | 9.6 | 0.000 / 0.000 | 1365 | 0.139 | 56.8 | 24.0 | 1.000 | kp_alt@max |

## bench_ic.json  (run `bench_ic-b1`)

batch wall 116.5 s on 8 workers, 5574 sims computed (47.8 sims/s); cache rerun: 1.9 s, hit rate 1.000, 0 sims computed, identical results: True; box load avg (1 min) at start/end 13.6/14.5; sequential schedule (cache off): 128.7 s, identical: True

| aircraft | profile | best fitness (gen0 -> final) | hold RMS ft (calm / mean) | hold max abs ft | overshoot ft | settle s (+-20 ft) | invalid gen0 / all gens | sims | cpu s/sim | aircraft wall s | sims/s | rerun hit rate | genes at bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c172x | baseline | 0.3192 -> **0.1923** | 3.13 / 2.95 | 7.39 | 6.9 | 7.9 | 0.250 / 0.033 | 1341 | 0.191 | 113.0 | 11.9 | 1.000 | kp_alt@max |
| f16 | ic_f16 | 0.1804 -> **0.0727** | 0.07 / 0.88 | 4.96 | 4.7 | 5.8 | 0.219 / 0.034 | 1335 | 0.197 | 114.0 | 11.7 | 1.000 | ki_alt@min, ki_pitch@min, kd_pitch@min |
| 737 | ic_737 | 0.2387 -> **0.1317** | 0.79 / 1.35 | 5.33 | 9.4 | 8.4 | 0.062 / 0.014 | 1392 | 0.143 | 115.2 | 12.1 | 1.000 | - |
| t6texan2 | ic_t6texan2 | 0.1265 -> **0.0720** | 0.19 / 0.55 | 2.64 | 2.6 | 5.5 | 0.188 / 0.019 | 1506 | 0.131 | 112.0 | 13.4 | 1.000 | kp_alt@max |

## bench_adjusted.json  (run `bench_adjusted-b1`)

batch wall 142.9 s on 8 workers, 5604 sims computed (39.2 sims/s); cache rerun: 2.2 s, hit rate 1.000, 0 sims computed, identical results: True; box load avg (1 min) at start/end 14.8/14.5; sequential schedule (cache off): 202.6 s, identical: True

| aircraft | profile | best fitness (gen0 -> final) | hold RMS ft (calm / mean) | hold max abs ft | overshoot ft | settle s (+-20 ft) | invalid gen0 / all gens | sims | cpu s/sim | aircraft wall s | sims/s | rerun hit rate | genes at bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c172x | baseline | 0.3192 -> **0.1923** | 3.13 / 2.95 | 7.39 | 6.9 | 7.9 | 0.250 / 0.033 | 1341 | 0.228 | 135.6 | 9.9 | 1.000 | kp_alt@max |
| f16 | adj_f16 | 0.1076 -> **0.0721** | 0.05 / 0.91 | 5.16 | 4.8 | 5.7 | 0.188 / 0.025 | 1482 | 0.238 | 138.4 | 10.7 | 1.000 | - |
| 737 | adj_737 | 0.2387 -> **0.1245** | 0.73 / 1.47 | 5.87 | 4.6 | 7.5 | 0.219 / 0.030 | 1533 | 0.170 | 141.4 | 10.8 | 1.000 | - |
| t6texan2 | adj_t6texan2 | 0.1052 -> **0.0726** | 0.19 / 0.44 | 2.29 | 6.3 | 5.9 | 0.250 / 0.016 | 1248 | 0.163 | 139.6 | 8.9 | 1.000 | ki_alt@min |

