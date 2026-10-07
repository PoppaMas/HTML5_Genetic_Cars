# STATUS A: f16 on Flight Dynamics' jsbsim_root (verified 2026-10-06 07:3x PT, no new runs needed)

Runs reused: `runs/phase1-f16fd-s{1,2,3}` (default profile `phase1_f16`, ki_alt max 0.05) and
`runs/phase1-f16fd-kialt03-s{1,2,3}` (opt-in `phase1_f16_kialt03`, ki_alt max 0.3). pop 32 x 20, 3 scenarios,
scenario_seed 1, 8 workers. Config: `phase1_f16` has aircraft_root = FD jsbsim_root, throttle_max 0.5.

Trim (test `test_fd_f16_loads_trims_like_fd_and_has_no_sockets`, re-run in the final suite):
alpha 1.04546 deg (FD 1.045), throttle-cmd 0.28358 (FD 0.2836), pos 0.5672, pitch-trim -0.06008, gear 0,
point mass [0] = 0 lb placeholder, [1] = 230 lb pilot. Nothing in evolution/ indexes point masses.

| seed | best (default) | best (kialt03) | hold RMS mean (calm) ft default | invalid all/gen0/final | wall s | load 1-min start->end |
|---|---|---|---|---|---|---|
| 1 | 0.09009480210402053 | 0.09017720598212726 | 1.085 (0.808) | 0.0094/0.125/0 | 78.1 evolve (79.5 batch) | 8.0->13.9 |
| 2 | 0.08876249360294831 | 0.08881327062170023 | 0.566 (0.116) | 0.0016/0.031/0 | 82.4 (83.4) | 13.6->14.2 |
| 3 | 0.08942040323392227 | 0.0893009028947082 | 0.553 (0.110) | 0/0/0 | 78.0 (79.5) | 14.2->19.5 |

Default: mean 0.08943, std ddof=1 0.00067 (pop 0.00054), range 0.08876-0.09009. Identical to the stock-model phase1 f16
(0.0901/0.0888/0.0894, mean 0.0894) to full precision (same best genomes).
kialt03: mean 0.08943, std 0.00069, range 0.08881-0.09018; per-seed +0.09%/+0.06%/-0.13%; walls 80.2/92.2/87.1 s
(load 12.0->17.1). 0 re-sim mismatches everywhere.

Bounds verdict: default bounds kept. `analysis/f16_bounds_probe.json` (1-D sweeps + 256 random genomes: 92.6% valid)
and `f16_fine_sweep.json`: kp_alt optimum 0.3-0.4 inside the bound; kd_pitch@min costs 2e-5; ki_alt has a second
basin above 0.05 (s2 0.09 -> 0.0853, -3.9%) but the GA with ki_alt max 0.3 (kialt03) never reached it
(best ki_alt 5e-7..3e-5) -> NOT adopted; `phase1.json`/`phase1_hdg.json` still use `phase1_f16`.
Sockets: FD f16.xml has no port= elements (meta network_io_removed 0); no stripped copy is made.
