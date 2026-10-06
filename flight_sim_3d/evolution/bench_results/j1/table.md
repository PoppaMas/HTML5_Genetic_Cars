# Benchmark j1

## bench_jets.json  (run `bench_jets-j1`)

batch wall 124.3 s on 8 workers, 5589 sims computed (45.0 sims/s); cache rerun: 1.9 s, hit rate 1.000, 0 sims computed, identical results: True; box load avg (1 min) at start/end 8.3/8.5

| aircraft | profile | best fitness (gen0 -> final) | hold RMS ft (calm / mean) | hold max abs ft | overshoot ft | settle s (+-20 ft) | invalid gen0 / all gens | sims | cpu s/sim | aircraft wall s | sims/s | rerun hit rate | genes at bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c172x | baseline | 0.3192 -> **0.1923** | 3.13 / 2.95 | 7.39 | 6.9 | 7.9 | 0.250 / 0.033 | 1341 | 0.188 | 122.0 | 11.0 | 1.000 | kp_alt@max |
| T38 | jet_T38 | 0.2036 -> **0.1315** | 1.63 / 3.28 | 11.63 | 7.9 | 14.8 | 0.000 / 0.000 | 1395 | 0.162 | 120.8 | 11.5 | 1.000 | ki_alt@min |
| 737 | jet_737 | 0.3374 -> **0.1920** | 1.21 / 1.89 | 6.85 | 7.4 | 12.2 | 0.219 / 0.022 | 1425 | 0.148 | 123.1 | 11.6 | 1.000 | - |
| f16 | jet_f16 | 0.1944 -> **0.1217** | 1.61 / 3.01 | 11.61 | 8.1 | 9.6 | 0.031 / 0.002 | 1428 | 0.202 | 119.6 | 11.9 | 1.000 | ki_alt@min, kd_pitch@min |

## bench_jets_step200.json  (run `bench_jets_step200-j1`)

batch wall 260.9 s on 8 workers, 6009 sims computed (23.0 sims/s); cache rerun: 2.9 s, hit rate 1.000, 0 sims computed, identical results: True; box load avg (1 min) at start/end 8.5/23.4

| aircraft | profile | best fitness (gen0 -> final) | hold RMS ft (calm / mean) | hold max abs ft | overshoot ft | settle s (+-20 ft) | invalid gen0 / all gens | sims | cpu s/sim | aircraft wall s | sims/s | rerun hit rate | genes at bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c172x | baseline | 0.3192 -> **0.1923** | 3.13 / 2.95 | 7.39 | 6.9 | 7.9 | 0.250 / 0.033 | 1341 | 0.368 | 255.2 | 5.3 | 1.000 | kp_alt@max |
| T38 | jet_T38 | 0.1678 -> **0.0833** | 0.27 / 1.56 | 6.89 | 8.5 | 5.3 | 0.031 / 0.002 | 1548 | 0.320 | 248.5 | 6.2 | 1.000 | ki_alt@min |
| 737 | jet_737 | 0.2339 -> **0.1288** | 0.78 / 1.42 | 5.80 | 4.9 | 8.2 | 0.219 / 0.025 | 1527 | 0.282 | 258.6 | 5.9 | 1.000 | - |
| f16 | jet_f16 | 0.1814 -> **0.0744** | 0.13 / 1.50 | 6.96 | 6.8 | 5.3 | 0.062 / 0.003 | 1593 | 0.395 | 252.5 | 6.3 | 1.000 | ki_alt@min, ki_pitch@min, kd_pitch@min |

