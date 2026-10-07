# STATUS P3-B1 r1 PILOT — `phase3b1r1-pilot-s1` DONE (64×60, rigid→full_a1_b1); tweaked A/B RUNNING (2026-10-06 ~21:15 PT)

**Run:** `evolution/runs/phase3b1r1-pilot-s1`, config `configs/phase3b1_pilot.json` (r1 pins; c172x seed 1 / T38 2 / 737 3,
scenario_seed 1, 3 scenarios), launched 20:08:46 PT from the code snapshot `/workspace/er_pilot_code_b1r1` (launch.sh there)
with `EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1`, 8 workers, PID 1880480. **Exit 0** at ~21:04 PT ("done in 3312.3 s,
41436 sims computed, 2253/43689 cache hits"), one session, no resume needed. run.json `fd_dir` = `evolution/_fd_pin_p3b1r1`,
code_sha `6e8239d0e3c8634f` (= current code), pins full_a1_b1 c172x `56ee798e` / T38 `7e871977` / 737 `6523753c`.
Resim mismatches: none. Report: `analysis/p3b1r1_pilot_report.py` → `analysis/phase3b1r1_pilot_s1_check.json`.

## 1. Baseline results (best = rank 0 of g59 at full_a1_b1 = checkpoint best_per_gen[59]; not min over genomes.jsonl)
| | c172x | T38 | 737 |
|---|---|---|---|
| **best cost** (id) | **0.248063** (c172x:g59:r0) | **0.109280** (T38:g59:r0) | **0.135173** (737:g59:r0) |
| best-so-far g0/10/20/30/40/50/59 | 0.3799 / 0.2766 / 0.2575 / 0.2523 / 0.2523 / 0.2491 / 0.2481 | 0.1536 / 0.1201 / 0.1149 / 0.1114 / 0.1095 / 0.1093 / 0.1093 | 0.3096 / 0.1446 / 0.1377 / 0.1360 / 0.1356 / 0.1356 / 0.1352 |
| last improvement | g59 (still improving) | g51 | g58 |
| best-so-far monotone | yes | yes | yes |
| Σ J_* (≥ 0) | 0.034091 | 0.014608 | 0.015099 |
| J_mass | 0.002112 | 0.000444 | 0.000478 |
| J_wing_tip_bm_limit | 0 | 0 | 8.4e-6 |
| mass Δ | +0.704 % | +0.148 % | +0.159 % |
| flutter margin | 1.2191 | 1.2009 | 1.1981 (J_flutter_margin 9.3e-5) |
| stiffness genes at floor (final pop) | none | none | none |
| nsm root / tip at 1.0 (final pop) | 6 % / 78 % | 92 % / 23 % | 92 % / 98 % |
| rejects, all rows (3840) | 18: overload 17, attitude 1 (all rigid screen) | 12: diverged 9, struct_ult_empennage 2, flutter 1 | 45: overload 22, flutter 22, diverged 1 |
| rejects at full_a1_b1 | 0 | struct_ult_empennage 2, flutter 1 | flutter 22 |
| geometry_gate rejects | 0 | 0 | 0 |
| vs Phase 2 seeds 1–3 (mean ± sd) | 0.21539 ± 0.01337 → +0.0327 (+2.4 sd) | 0.09580 ± 0.00321 → +0.0135 (+4.2 sd) | 0.12348 ± 0.00284 → +0.0117 (+4.1 sd) |
| vs r1 16×5 smoke | 0.3830 → −0.1349 | 0.1571 → −0.0479 | 0.2721 → −0.1369 |
| CPU s/scenario rigid / full_a1_b1 (blended) | 0.234 / 2.008 (0.587) | 0.201 / 1.915 (0.541) | 0.187 / 1.859 (0.512) |

**Phase 2 comparison is NOT like-for-like:** Phase 2 bests were on the post-mass pins without the P2.5 floors, without
the A1 65-node mesh / station-exact tip term and without the shape genes (and a different full model). The B1 pilot is
worse than the Phase 2 seed mean on all three (+2.4 / +4.2 / +4.1 sd), one seed only.

Shape genes (best; final-pop mean; fraction of final pop within 2 % of floor / ceiling):
| gene [bounds] | c172x | T38 | 737 |
|---|---|---|---|
| chord_taper_1 [0.85, 1.05] | 0.897; 0.894; 0.02 / 0 | 0.986; 0.986; 0 / 0 | 1.004; 1.008; 0 / 0.02 |
| chord_taper_2 [0.85, 1.05] | **1.050**; 1.031; 0 / 0.14 | 0.971; 0.969; 0 / 0 | 1.008; 1.009; 0 / 0 |
| chord_taper_3 [0.85, 1.05] | 0.939; 0.947; 0 / 0 | 0.997; 0.995; 0 / 0.02 | 0.952; 0.952; 0 / 0 |
| twist_mid [−2, +1] ° | **−2.000**; −1.944; **0.84** / 0 | +0.787; +0.784; 0 / 0.14 | +0.370; +0.368; 0 / 0.02 |
| twist_tip [−4, +1] ° | −1.662; −1.596; 0 / 0 | −3.693; −3.632; 0.14 / 0 | −3.141; −3.053; 0 / 0 |
| sweep_qc_delta [−5, +5] ° | +3.547; +3.618; 0 / 0 | −2.567; −2.252; 0 / 0 | **−5.000**; −4.718; **0.67** / 0 |
- **FLAGS (≥ 25 % of final pop at a bound):** c172x twist_mid at the −2° floor (84 %, best = −2.0); 737 sweep_qc_delta at
  the −5° floor (67 %, best = −5.0). Possible bound-limited optima / exploits — for FD to check (r1 verify had baseline
  c172x twist_mid −2 at 0.345485 with tip term 0.1016; the pilot best at −2 has tip term 0, i.e. the structure / controller
  co-evolved around it).
- **T38 twist_mid is NOT at +1 this time:** best +0.787, 14 % of final pop at the ceiling (smoke: 69 % at +1, neutral).
- Wall 3312.3 s (55 s/gen, 3 aircraft concurrent, 8 workers). Box load (1-min, loadmon every 60 s): start 0.43, mean 11.9
  (8.3–16.1) during the run, end 11.8 on 8 cores — other agents (5× `p3b2_study.py` from /workspace/venv-fs) shared the
  box; CPU/wall are load-confounded.

## 2. Shape split (each best re-flown on all 3 scenarios, full_a1_b1, frozen FD dir, fresh process)
| | own shape (re-fly; bit-identical to log) | shape reset to FD baseline | shape contribution |
|---|---|---|---|
| c172x | 0.248063 ✔ | 0.255527 (ok) | −0.007464 (−2.9 %): effort −0.0026 raw, comfort −0.0287 raw, J_bm_rms −0.0005; flutter 1.2191 vs 1.2546 |
| T38 | 0.109280 ✔ | 0.117146 (ok) | −0.007866 (−6.7 %): almost all J_flutter_margin (−0.00788; baseline shape flutter margin 1.1822 vs 1.2009) |
| 737 | 0.135173 ✔ | 0.135883 (ok) | −0.000710 (−0.5 %): effort −0.0006 raw, comfort +0.0049 raw, J_bm_rms +0.0002 |
So the controller + structure carry most of the cost at the best (baseline-shape cost is within 3 % / 7 % / 0.5 % of the
best); shape is worth ~0.007 on c172x and T38 and ~0.0007 on 737. The T38 shape gain is a flutter-margin recovery, not a
flight-term gain. (Gen-0 shapes were not baseline — shape init sigma 0.25 half-range — so no clean "from g0" split.)

## 3. LADDER CONCERN: rigid screen vs full_a1_b1 (`analysis/p3b1r1_ladder_rho.py` → `phase3b1r1_ladder_rho.json`)
rho = Spearman(rigid, authoritative) on the genomes re-scored each gen (rigid top k_full = 16 + carried elites, ~17–18).
| run / aircraft | mean | median | frac < 0 | gens ≥ 30 mean (frac < 0) | rigid sd / gap sd in re-scored set (median) |
|---|---|---|---|---|---|
| **B1 pilot c172x** | −0.132 | −0.225 | 0.72 | −0.301 (0.97) | 8.3e-4 / 7.1e-3 (×8.7) |
| **B1 pilot T38** | −0.126 | −0.227 | 0.70 | −0.275 (0.90) | 1.5e-4 / 1.1e-2 (×70) |
| **B1 pilot 737** | −0.156 | −0.190 | 0.77 | −0.255 (0.93) | 5.9e-5 / 1.1e-2 (×154) |
| P2 pilot s1 c172x / T38 / 737 | −0.282* / −0.143 / −0.072 | −0.577* / −0.145 / −0.074 | 0.72 / 0.72 / 0.65 | −0.33 / −0.15 / −0.15 | ×7 / ×431 / ×112 |
| P2 pilot s2 | +0.081 / −0.062 / −0.102 | +0.035 / −0.102 / −0.076 | 0.48 / 0.65 / 0.63 | +0.01 / −0.16 / −0.21 | ×7 / ×859 / ×104 |
| P2 pilot s3 | −0.044 / +0.041 / −0.145 | −0.091 / +0.034 / −0.213 | 0.62 / 0.50 / 0.75 | −0.21 / +0.05 / −0.22 | ×24 / ×112 / ×66 |
(*P2 s1 c172x re-scored only ~4–5 genomes/gen, so its rho is very noisy.) Gen 0 (random genomes incl. infeasibles) is the
only consistently positive gen (+0.53 / +0.85 / +0.61).
- **Not new to B1:** the Phase 2 rigid→full pilots already had mostly negative rho (mean −0.28…+0.08, 48–77 % of gens < 0).
  B1 is somewhat more consistently negative late (gens ≥ 30: 90–97 % of gens < 0, mean ≈ −0.28 vs P2 −0.33…+0.05).
- **Why:** inside the promoted set the rigid costs are nearly tied (sd 6e-5…8e-4), while (full − rigid) varies 9–150× more
  (shape genes, flex structure terms, margins: all invisible to rigid). The mean gap full − rigid is +0.05 / +0.04 / +0.08.
  The ordering of the top 16 by rigid is therefore noise w.r.t. full, with a mild systematic anti-correlation (genomes
  the rigid sim likes best — presumably the most aggressive controllers — pay more at full).
- **Promotion vs random (within the promoted set):** the gen's full-best sat on average at rigid position 10.4 / 9.8 / 9.2
  of ~17 (random expectation 8.2 / 8.1 / 7.8), and was in the rigid top 4 in 27 % / 23 % / 25 % of gens (random 23 / 23 /
  24 %). So *within* the promoted 16 the rigid order is no better than random (slightly worse). Whether the 16-of-64 cut
  itself beats random **cannot be decided from the logs** (non-promoted genomes have no full cost): the rigid screen still
  removes the overload / attitude / diverged failures (17 / 1 / 9 rigid-only rejects never reached full). Probe ready:
  `analysis/p3b1r1_screen_vs_random.py GEN` re-scores the ~46 non-promoted genomes of a gen at full_a1_b1 (≈ 3 × 46 × 3
  scenarios × ~1.9 CPU s ≈ 13 CPU-min per gen) and compares rigid promotion with random 16-subsets. It was started at
  nice 19 during the A/B, starved (no aircraft finished), and stopped — **deferred until the A/B has exited.**
- **Implication (evidence only, no fix applied):** past gen 0 the rigid screen mostly acts as a feasibility filter, not a
  ranker; selection among promoted genomes is driven entirely by full_a1_b1, and ~75 % of each generation is never seen at
  full. Options to evaluate: full-only evaluation (cost ≈ 4× the full stage: ~64 vs ~17 full genomes/gen, rough estimate ≈ +100–150 % wall
  per gen at today's 55 s), or a cheaper B1-aware screen (e.g. reduced/A1 flex with shape, or rigid + pre-flight
  margins/mass terms) so the screen sees the shape and structure terms that dominate the within-top spread.

## 4. Trajectories (`runs/phase3b1r1-pilot-s1/trajectories/`, exported by the run itself at exit with the snapshot code)
- 9 files `traj_{c172x,T38,737}_phase3b1r1-pilot-s1_g{0,29,59}.json` + `index.json` (9 entries). `validate_traj`
  (--min-gens-per-aircraft 3): **OK, exit 0**. All 9: `planform` header `fd-planform/1` with genes = the row's shape,
  `structure.node_layout = "FD flexbody_b1.node_layout_b1 (P3-B1 r1)"`, per-node chord / twist / LE / TE on wingR + wingL,
  `structure.modal_twist_sign_fixed: true`, model_version = run pin, scenario cost = genomes.jsonl rank-0 row
  (`analysis/p3b1r1_pilot_traj_check.py` → `analysis/phase3b1r1_pilot_s1_traj_check.json`).
- **Fresh-process replay** of `traj_T38_phase3b1r1-pilot-s1_g59.json` (T38:g59:r0, T38:s0) through `_fd_pin_p3b1r1`
  (snapshot code, fd_dir check OK): 0.07783981426439955 = trajectory header = genomes.jsonl, **bit for bit**; model_version
  `full_a1_b1:flexv2b1:7e871977` matches.

## 5. Tweaked A/B `phase3b1r1-pilot-tweaked-s1` (RUNNING)
- Code snapshot `/workspace/er_pilot_code_b1r1_tweaked` (taken 20:58:34 PT; same method as `er_pilot_code_b1r1`: the 15
  evolution/*.py + configs/ copied read-only, symlinks runs / cache / logs / _fd_pin_p3b1r1; SHA256 list in
  `SNAPSHOT_SHA256.txt`; code differs from the baseline snapshot only in batch.py / ga.py = the opt-in `shape_crossover`, configs/ adds the tweaked configs; eval
  code_sha identical `6e8239d0e3c8634f`). Launcher `launch.sh` (+ `launch.sh resume`), load monitor `loadmon.sh` →
  `logs/phase3b1r1-pilot-tweaked-s1.load.log`.
- Config = snapshot copy of `configs/phase3b1_pilot_tweaked.json`: run.json differs from the baseline only in run_id /
  created / ga (`elite: 4`, `shape_crossover: "uniform"`); pins, seeds, scenarios identical; `--model-versions` all match;
  run.json `fd_dir` = `evolution/_fd_pin_p3b1r1`.
- Launched 21:06:08 PT, PID 1922093, log `logs/phase3b1r1-pilot-tweaked-s1.log`. **Gen 0 identical to the baseline:**
  240/240 cache hits per aircraft, all 192 gen-0 rows equal (genome, cost, per-scenario cost, status, fidelity, screen
  cost), same best / median / rescored set / rho.
- ~54–56 s/gen (gens 1–7, box load ~12 on 8 cores) → ETA ≈ 22:00–22:02 PT (+ ~1 min trajectory export at exit).
