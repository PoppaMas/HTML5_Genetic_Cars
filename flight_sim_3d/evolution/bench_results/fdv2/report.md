FD v2 ladder benchmark (c172x, 737; pop 32 x 4 gens, seed 1, cache off, 8 workers). Times PT.

| case | start-end PT | batch wall s | speedup vs full-only viz on (wall / CPU) | total task CPU s | load 1-min start→end | c172x mean gen wall s | c172x best full cost (Δ vs full-only) | c172x feasible rate (full-scored) | c172x final elites = full-only? (overlap) | 737 mean gen wall s | 737 best full cost (Δ vs full-only) | 737 feasible rate (full-scored) | 737 final elites = full-only? (overlap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full-vizon | 06:59:31-07:04:31 | 300.1 | 1.00x / 1.00x | 1193 | 11.8→15.2 | 61.8 | 0.25680 (+0.00000) | 0.78 (n=128) | yes (1.00) | 60.2 | 0.20478 (+0.00000) | 0.77 (n=128) | yes (1.00) |
| full-vizoff | 07:04:31-07:09:55 | 323.8 | 0.93x / 1.07x | 1113 | 15.2→18.7 | 79.4 | 0.25680 (+0.00000) | 0.78 (n=128) | yes (1.00) | 67.0 | 0.20478 (+0.00000) | 0.77 (n=128) | yes (1.00) |
| rigid-full | 07:09:55-07:10:59 | 63.4 | 4.74x / 3.37x | 355 | 18.7→12.0 | 14.6 | 0.28169 (+0.02489) | 0.90 (n=20) | no (0.00) | 13.3 | 0.19142 (-0.01336) | 0.89 (n=36) | no (0.00) |
| reduced-full | 07:10:59-07:13:30 | 150.9 | 1.99x / 1.10x | 1081 | 12.0→8.3 | 35.2 | 0.30551 (+0.04871) | 1.00 (n=19) | no (0.00) | 36.6 | 0.22157 (+0.01680) | 0.94 (n=36) | no (0.00) |
| rigid-reduced-full | 07:13:30-07:15:16 | 105.8 | 2.84x / 1.78x | 672 | 8.3→11.9 | 21.4 | 0.28263 (+0.02583) | 1.00 (n=17) | no (0.00) | 24.8 | 0.26762 (+0.06285) | 0.94 (n=36) | no (0.00) |

Counterfactual on the full-only populations (512 extra genome evaluations, 152 s, load 9.9→10.3):

| aircraft | gen | n full ok / reduced ok / rigid ok | reduced pass & full fail | Spearman reduced vs full (both ok, n) | rigid vs full (both ok, n) | elite set same as full-only: rigid→full / reduced→full / rigid→reduced→full |
|---|---|---|---|---|---|---|
| c172x | 0 | 13 / 21 / 23 | 8 | +0.667 (+0.967, 13) | +0.513 (+0.934, 13) | yes/yes/yes |
| c172x | 1 | 28 / 30 / 30 | 2 | +0.943 (+0.972, 28) | +0.907 (+0.928, 28) | yes/yes/yes |
| c172x | 2 | 27 / 26 / 26 | 0 | +0.956 (+0.921, 26) | +0.944 (+0.899, 26) | yes/no/yes |
| c172x | 3 | 32 / 32 / 32 | 0 | +0.927 (+0.927, 32) | +0.798 (+0.798, 32) | yes/yes/yes |
| 737 | 0 | 16 / 23 / 26 | 7 | +0.710 (+0.918, 16) | +0.552 (+0.882, 16) | yes/yes/yes |
| 737 | 1 | 25 / 29 / 30 | 5 | +0.663 (+0.862, 24) | +0.583 (+0.827, 24) | yes/yes/yes |
| 737 | 2 | 26 / 30 / 30 | 4 | +0.446 (+0.595, 26) | +0.571 (+0.623, 26) | yes/yes/yes |
| 737 | 3 | 31 / 32 / 32 | 1 | +0.409 (+0.455, 31) | +0.606 (+0.621, 31) | no/yes/yes |

| aircraft | k_full (+ elites) | ladder | gens elite set differs (of G) | mean elite overlap | gens best same |
|---|---|---|---|---|---|
| c172x | 4 | rigid-full | 0/4 | 1.00 | 4/4 |
| c172x | 4 | reduced-full | 1/4 | 0.88 | 4/4 |
| c172x | 4 | rigid-reduced-full | 0/4 | 1.00 | 4/4 |
| 737 | 8 | rigid-full | 1/4 | 0.88 | 4/4 |
| 737 | 8 | reduced-full | 0/4 | 1.00 | 4/4 |
| 737 | 8 | rigid-reduced-full | 0/4 | 1.00 | 4/4 |

Per-generation Spearman logged by the ladder runs (screen vs full on the re-scored set):

- rigid-full c172x (n re-scored [4, 5, 5, 6]): rigid_vs_full +0.800; rigid_vs_full +0.872; rigid_vs_full +0.205; rigid_vs_full -0.600
- rigid-full 737 (n re-scored [8, 9, 10, 9]): rigid_vs_full +0.467; rigid_vs_full -0.328; rigid_vs_full +0.006; rigid_vs_full +0.250
- reduced-full c172x (n re-scored [4, 5, 5, 5]): reduced_vs_full +1.000; reduced_vs_full +0.900; reduced_vs_full -0.100; reduced_vs_full +0.000
- reduced-full 737 (n re-scored [8, 9, 9, 10]): reduced_vs_full +0.786; reduced_vs_full -0.117; reduced_vs_full -0.217; reduced_vs_full -0.200
- rigid-reduced-full c172x (n re-scored [4, 5, 4, 4]): rigid_vs_full +0.400, reduced_vs_full +1.000, rigid_vs_reduced +0.810; rigid_vs_full +0.359, reduced_vs_full +0.900, rigid_vs_reduced +0.577; rigid_vs_full +0.738, reduced_vs_full +0.400, rigid_vs_reduced +0.048; rigid_vs_full +0.200, reduced_vs_full +1.000, rigid_vs_reduced -0.042
- rigid-reduced-full 737 (n re-scored [8, 9, 10, 9]): rigid_vs_full +0.452, reduced_vs_full +0.786, rigid_vs_reduced +0.777; rigid_vs_full -0.561, reduced_vs_full -0.350, rigid_vs_reduced +0.836; rigid_vs_full -0.061, reduced_vs_full +0.079, rigid_vs_reduced +0.727; rigid_vs_full -0.226, reduced_vs_full -0.350, rigid_vs_reduced +0.809
