# P3B2a determinism investigation (in progress, started 01:08 PT 2026-10-07)
Targets: c172x:g0:r9 (s0,s2), c172x:g0:r10 (s0), c172x:g1:r15 (s2) in runs/phase3b2a-smoke-s1.

## 01:15 PT — Reproduction (fresh process, snapshot code /workspace/er_smoke_code_b2a, EVOLUTION_FD_DIR=evolution/_fd_pin_p3b2a; script /workspace/b2det/rep3.py)
Whole-genome eval (ev.evaluate(genome,'c172x',None,run.json) = batch path, one FD flexeval_b2.evaluate over 3 scenarios) reproduces genomes.jsonl BIT-EXACT for all 3.
Single-scenario eval (ev.evaluate(...,k,...) = Sim Bridge's per-scenario path) differs, reproducing their numbers:
- r9: s0 3.560088358927666 vs logged 3.5528782217181023 (2.0e-3); s2 3.698967848192427 vs 3.6322964712316512 (1.8e-2)
- r10: s0 16.02962365963003 vs 16.01594600660605 (8.5e-4)
- r15: s2 2.7338794985876946 vs 2.6956604094861185 (1.4e-2)
- s1 (the overloaded scenario) matches in all. => NOT worker state; deterministic context dependence (which scenarios are flown together).

## 01:25 PT — Cache provenance
history.jsonl c172x: g0 48/48 sims computed (0 cache hits), g1 39 computed + 6 mem hits (elites), 0 disk. Cache unit = genome x ALL scenarios (batch.py ~785-842), so cached or not, the logged value is the whole-genome value, which a fresh process reproduces bit-exactly. Cache not involved.

## Q4 phase3b1r1-pilot-tweaked-s2 c172x
Best last improved at g29 (g28 0.2420510265812971 -> g29 0.24036583920850696, c172x:g29:r0). g30..g59 best rows are carried_elite=True with genome_norm identical to g29's; checkpoint best_per_gen[59].genome == [29].genome. Elitism holding, not stuck: g59 pop 64/64 unique genomes, mean per-gene std 0.0415 (g29 0.0517, g0 0.1596), mean pairwise L2 0.321 (g0 1.314), median cost 0.2186 (g29 0.2290) still improving.

## 01:45 PT — Root cause (Evolution side, NOT FD, NOT worker state)
- Warm single process evaluating all c172x g0+g1 genomes in index order (/workspace/b2det/warm.py) reproduces every logged row bit-exact -> no carried state.
- Single-scenario minus logged per-scenario cost == that scenario's J_energy+J_speed_guard exactly; FD cost (energy_terms.cost_fd) is identical both ways.
- Cause: evolution/fidelity.py:946 (pre-fix) `if energy_cost and out["status"] == "ok" and r["per_scenario"]:` gates the energy addition on the GENOME status. energy_terms() (fidelity.py:195) already zeroes non-ok SCENARIOS. So an ok scenario of an overload genome gets no energy in the batch (whole-genome) eval but does get it when flown alone. Contradicts the energy_terms docstring (fidelity.py:189-190: "a single-scenario re-fly reproduces its row cost bit for bit").
- J_energy/J_speed_guard on overload rows: computed and LOGGED in energy_terms (e.g. r9 J_speed_guard 0.01975) but NOT in cost. Order dependence of the accumulation: none (pure function of FD per-scenario energy; sums are mean over scenarios in fixed scenario order).
- FD candidates (FGFDMExec reuse, lru caches, ensure_root_v2b2, early-exit) excluded: FD cost identical in all contexts, warm process == fresh.
- Fix: drop the genome-status condition (per-scenario gate only); docstrings updated. Regression test tests/test_p3b2a.py::test_energy_per_scenario_independent_of_genome_status (fails on old code at single==whole, passes on new).

## B1 check (phase3b1r1-pilot-s1, FD _fd_pin_p3b1r1, /workspace/b2det/b1ovl*.py)
Whole and single-scenario replays == logged for 11/11 non-ok rows: rigid c172x overload r47,r48, attitude r53; 737 overload r56,r57; T38 diverged r55,r62; full T38 structural_ultimate_empennage g0 r13,r14; 737 flutter g10 r14,r15. B1 has no energy term -> not affected.
Seed-1 replay checks with fixed code: baseline T38 g59 s0 0.07783981426439955, tweaked 0.07474663523979805 (bit-identical, refly_own_bit_identical True).
- 01:55 PT full suite with fix: 179 passed (178 + regression), log /workspace/b2det/pytest_full.log. Launching phase3b2a-smoke-detfix-s1 from snapshot /workspace/er_smoke_code_b2a_detfix (same config, seed, 8 workers).
- Phase 2 replay with fixed code (FD _fd_pin_post_mass, /workspace/elitism_audit/reload_check.py): phase2-pilot-s1 T38 g30 s0 0.06540879802372773, c172x g59 s0 0.12499663997434847, both bit-identical to genomes.jsonl and checkpoint.

## 02:12 PT — Re-run + replays
- Old run replayed with fixed code (720 single-scenario evals, 8 fresh processes, /workspace/b2det/rep_oldrun_newcode.jsonl): 716/720 exact; the 4 misses are exactly Sim Bridge's (r10 s0, r9 s0, r9 s2, r15 s2) — logged values were the buggy ones.
- phase3b2a-smoke-detfix-s1 (fresh code /workspace/er_smoke_code_b2a_detfix, code_sha c72951169fbccafe vs old ce1d68dacb2fcac8; 338.5 s, 639 sims): same 240 individual ids/genomes in the same ranks as phase3b2a-smoke-s1; only costs changed: r10 976.4288036668904->976.4333628845652, r9 515.0292998214112->515.0539269928014, r15 597.9655012534253->597.9782409497924. Bests g4 unchanged (c172x 0.3878333132087896, T38 0.15516077310585433, 737 0.22351666196599074).
- Impact: no rank order change in any generation (computed from logged energy_terms), so no selection/best depended on the wrong value. B1/A1/phase2 unaffected (no energy term; gate only exists in B2 block).
- phase3b2a-smoke-detfix-s1 replayed with fixed code, 720 single-scenario evals in 8 fresh processes through _fd_pin_p3b2a: **720/720 bit-exact**, model_version all match (/workspace/b2det/rep_newrun.jsonl).

## Status: DONE (02:20 PT). Not done: nothing pushed; FD untouched; old run dirs untouched. Old smoke's 3 genomes.jsonl values stay wrong-by-definition (documented here); new run is phase3b2a-smoke-detfix-s1.
