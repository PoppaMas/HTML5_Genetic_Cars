# STATUS E: Phase-1 v5 (verified 2026-10-06 07:3x PT, no new runs needed)

`configs/phase1_v5.json` (opt-in; c172x/T38/737; v4 phase1_hdg.json stays default). Runs `runs/phase1v5-s{1,2,3}`,
report `logs/phase1v5_report.json`. Bit-identical (tests/test_v5.py): c172x v5 s1 0.20625113824760744,
T38 0.08778028702197657, 737 0.11126020207078736, v4 c172x genome on v5 0.22748161325788208 (per-scenario + hold_osc).

| aircraft | s1 | s2 | s3 | mean | std ddof=1 (pop) |
|---|---|---|---|---|---|
| c172x | 0.22294229004024413 | 0.2059428225172206 | 0.23225242158061704 | 0.22038 | 0.01334 (0.01089) |
| T38 | 0.08778028702197657 | 0.11142965429309869 | 0.09158059727203648 | 0.09693 | 0.01270 (0.01037) |
| 737 | 0.11126020207078736 | 0.10884742151296913 | 0.11355070908836735 | 0.11122 | 0.00235 (0.00192) |

vs Genome: T38 s1 and 737 s1 equal Genome's references bit for bit; T38 ki_alt 0.05 (s1, s3), downdraft residual
-0.00 ft (s1, s3); s2 other basin (ki_alt 7.4e-7, residual +3.26 ft). c172x calm hold p-p 5.94/3.41/5.30 ft vs Genome
3.2-3.9 (only s2 inside; Genome used pop 48x40). Walls 235 / 323 / 618 s, load 5.6->12.0, 12.0->17.2, 17.2->43.3.
0 re-sim mismatches; 27 trajectories validate.

## Update (07:51 PT steering): v5-only ki_alt upper bound 0.5 (genome handoff section 6)
* configs/phase1_v5.json = updated genome export (ki_alt [min, 0.5] on all three; repo-relative aircraft_root). v4
  (phase1_hdg.json, ki_alt 0.05) unchanged and still bit-identical (test_heading). test_v5 updated: v5 = v4 + 4 keys +
  ki_alt 0.5; the 4 v5 reference genomes still re-fly bit for bit (8 passed).
* The old phase1v5-s{1,2,3} runs used the old 0.05 bound.
* New run phase1v5-ki05-s1 (seed 1, pop 32 x 20, 07:53:46-08:00:05 PT, wall 378.8 s, load 0.9 -> 21.3 from other
  agents' jobs, 0 re-sim mismatches):
  | aircraft | best | ki_alt | hold p-p calm ft | downdraft residual / max err ft | old-bound s1 |
  |---|---|---|---|---|---|
  | c172x | 0.22514600492676504 | 0.0736 | 6.18 | -0.55 / 4.98 | 0.222942 |
  | T38 | 0.0877912874821699 | 0.0656 | 0.50 | -0.00 / 4.68 | 0.087780 |
  | 737 | 0.1379114184282066 | 0.00131 (kp_pitch@max) | 0.59 | +3.02 / 4.70 | 0.111260 |
  T38 equals Genome's new reference (genome/runs/v5_kialt05_t38_s1: best_cost 0.0877912874821699, ki_alt 0.0656048)
  bit for bit. c172x/737 were not rerun by Genome. The 737 got worse with the wider bound on this seed (0.1379 vs 0.1113):
  the GA settled in a low-ki_alt basin. One seed only.
