# STATUS B: heading hold (verified 2026-10-06 07:3x PT, no new runs needed)

Config `configs/phase1_hdg.json` (v4, == genome/exports, the DEFAULT for heading work; f16 heading off).
Runs `runs/phase1hdg-s{1,2,3}`, pop 32 x 20, scenario_seed 1, 8 workers. Report `logs/phase1hdg_report.json`.

Bit-identical (tests/test_heading.py, re-run in final suite): handoff genomes re-fly to c172x 0.19600587132367517,
T38 0.0966245736792507, 737 0.10678211633774594, every per-scenario cost equal.

| aircraft | s1 | s2 | s3 | mean | std ddof=1 (pop) | max final drift deg per seed |
|---|---|---|---|---|---|---|
| c172x | 0.1974858449873731 | 0.19537268793562368 | 0.20087619133530174 | 0.19791 | 0.00278 (0.00227) | 0.78/0.74/2.22 |
| T38 | 0.0966245736792507 | 0.09300736455393906 | 0.09245137662400156 | 0.09403 | 0.00227 (0.00185) | 0.04/0.09/0.00 |
| 737 | 0.10678211633774594 | 0.110647497109246 | 0.10888489971293346 | 0.10877 | 0.00194 (0.00158) | 0.78/0.04/0.07 |
| f16 (off) | 0.0900948 | 0.0887625 | 0.0894204 | 0.08943 | 0.00067 | - |

c172x calm drift 25.4/25.5/25.3 deg (phase1-s*) -> 0.25/0.53/1.85 deg. Genome: T38 0.0966/0.0930/0.0925 matched per seed;
737 s1 0.1068 matched; c172x mean 0.1935 is pop 48x40 with scenario seed = GA seed (different budget/scenarios);
their s1 genome = 0.1960 on our code (bit-identical), ours 0.1975 at 32x20.
Walls (batch, 4 aircraft): 208.9 / 322.2 / 320.0 s, load 1-min 0.3->12.8, 12.8->15.4, 15.4->19.4. 0 re-sim mismatches.
Sockets: all FD-root models have 0 socket elements, so the 737 strip is a no-op on FD's root. Guard
(`sim.socket_policy`): explicit aircraft_root with sockets -> refuse (load_failed); stock package data -> strip a /tmp
copy + loud WARNING; override EVOLUTION_SOCKET_POLICY.
