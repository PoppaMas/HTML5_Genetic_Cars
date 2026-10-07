FD v2 ladder benchmark (c172x, 737; pop 32 x 4 gens, seed 2, cache off, 8 workers). Times PT. Speedups are vs full-only viz OFF (no viz-on case for this seed).

| case | start-end PT | batch wall s | speedup vs full-only viz on (wall / CPU) | total task CPU s | load 1-min start→end | c172x mean gen wall s | c172x best full cost (Δ vs full-only) | c172x feasible rate (full-scored) | c172x final elites = full-only? (overlap) | 737 mean gen wall s | 737 best full cost (Δ vs full-only) | 737 feasible rate (full-scored) | 737 final elites = full-only? (overlap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full-vizoff | 07:18:37-07:23:44 | 307.3 | 1.00x / 1.00x | 1102 | 6.5→22.2 | 74.0 | 0.28892 (+0.00000) | 0.73 (n=128) | yes (1.00) | 64.1 | 0.16208 (+0.00000) | 0.80 (n=128) | yes (1.00) |
| rigid-full | 07:23:44-07:25:36 | 111.3 | 2.76x / 3.18x | 346 | 22.2→23.1 | 23.1 | 0.35286 (+0.06394) | 0.78 (n=18) | no (0.00) | 26.1 | 0.19902 (+0.03694) | 0.97 (n=32) | no (0.00) |
| reduced-full | 07:25:36-07:28:28 | 172.0 | 1.79x / 1.02x | 1083 | 23.1→16.3 | 40.2 | 0.29944 (+0.01052) | 1.00 (n=19) | no (0.00) | 41.5 | 0.18034 (+0.01826) | 0.97 (n=34) | no (0.00) |
| rigid-reduced-full | 07:28:28-07:30:19 | 110.6 | 2.78x / 1.55x | 710 | 16.3→15.8 | 22.7 | 0.27926 (-0.00966) | 1.00 (n=19) | no (0.00) | 26.1 | 0.20012 (+0.03804) | 1.00 (n=34) | no (0.00) |

Counterfactual on the full-only populations (512 extra genome evaluations, 143 s, load 11.9→8.5):

| aircraft | gen | n full ok / reduced ok / rigid ok | reduced pass & full fail | Spearman reduced vs full (both ok, n) | rigid vs full (both ok, n) | elite set same as full-only: rigid→full / reduced→full / rigid→reduced→full |
|---|---|---|---|---|---|---|
| c172x | 0 | 15 / 20 / 22 | 5 | +0.861 (+0.986, 15) | +0.679 (+0.986, 15) | yes/yes/yes |
| c172x | 1 | 20 / 25 / 25 | 6 | +0.837 (+0.954, 19) | +0.640 (+0.937, 19) | yes/yes/yes |
| c172x | 2 | 28 / 27 / 28 | 0 | +0.989 (+0.986, 27) | +0.976 (+0.964, 28) | yes/yes/yes |
| c172x | 3 | 31 / 30 / 31 | 0 | +0.959 (+0.950, 30) | +0.914 (+0.906, 31) | no/no/no |
| 737 | 0 | 16 / 21 / 25 | 5 | +0.822 (+0.697, 16) | +0.707 (+0.691, 16) | yes/yes/yes |
| 737 | 1 | 31 / 32 / 32 | 1 | +0.922 (+0.920, 31) | +0.928 (+0.932, 31) | yes/yes/yes |
| 737 | 2 | 28 / 29 / 29 | 2 | +0.874 (+0.871, 27) | +0.840 (+0.824, 27) | yes/yes/yes |
| 737 | 3 | 27 / 32 / 32 | 5 | +0.557 (+0.981, 27) | +0.642 (+0.968, 27) | yes/yes/yes |

| aircraft | k_full (+ elites) | ladder | gens elite set differs (of G) | mean elite overlap | gens best same |
|---|---|---|---|---|---|
| c172x | 4 | rigid-full | 1/4 | 0.88 | 3/4 |
| c172x | 4 | reduced-full | 1/4 | 0.88 | 3/4 |
| c172x | 4 | rigid-reduced-full | 1/4 | 0.88 | 3/4 |
| 737 | 8 | rigid-full | 0/4 | 1.00 | 4/4 |
| 737 | 8 | reduced-full | 0/4 | 1.00 | 4/4 |
| 737 | 8 | rigid-reduced-full | 0/4 | 1.00 | 4/4 |

Per-generation Spearman logged by the ladder runs (screen vs full on the re-scored set):

- rigid-full c172x (n re-scored [4, 5, 4, 5]): rigid_vs_full +0.400; rigid_vs_full +0.400; rigid_vs_full +0.632; rigid_vs_full +0.600
- rigid-full 737 (n re-scored [8, 8, 8, 8]): rigid_vs_full +0.595; rigid_vs_full +0.571; rigid_vs_full +0.643; rigid_vs_full +0.548
- reduced-full c172x (n re-scored [4, 5, 5, 5]): reduced_vs_full +1.000; reduced_vs_full +1.000; reduced_vs_full +0.900; reduced_vs_full +0.000
- reduced-full 737 (n re-scored [8, 8, 9, 9]): reduced_vs_full +0.786; reduced_vs_full +0.405; reduced_vs_full +0.450; reduced_vs_full -0.400
- rigid-reduced-full c172x (n re-scored [4, 5, 5, 5]): rigid_vs_full +1.000, reduced_vs_full +1.000, rigid_vs_reduced +0.500; rigid_vs_full +1.000, reduced_vs_full +1.000, rigid_vs_reduced +0.450; rigid_vs_full +0.100, reduced_vs_full +0.800, rigid_vs_reduced +0.000; rigid_vs_full +0.800, reduced_vs_full +0.800, rigid_vs_reduced +0.700
- rigid-reduced-full 737 (n re-scored [8, 8, 9, 9]): rigid_vs_full +0.714, reduced_vs_full +0.786, rigid_vs_reduced +0.804; rigid_vs_full +0.452, reduced_vs_full +0.167, rigid_vs_reduced +0.891; rigid_vs_full -0.117, reduced_vs_full +0.183, rigid_vs_reduced +0.944; rigid_vs_full +0.800, reduced_vs_full +0.500, rigid_vs_reduced +0.588
