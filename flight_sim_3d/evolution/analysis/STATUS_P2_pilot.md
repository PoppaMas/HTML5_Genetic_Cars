# STATUS P2 pilot — DONE 09:39 PT

Genome clearance 08:34 PT. Chain (`logs/phase2_pilot_chain.sh`) finished 09:38:05 PT. Everything local; no push; read-only folders untouched.

## Cost formula (pre-launch audit — no fix needed)
FD `flexeval` alone. Evolution adds nothing (mass-credit clip OFF).
```
per_scenario cost = sim_cost + Σ RESP_TERMS[full] + Σ PRE_TERMS[full]
sim_cost          = track + 2.0·effort + 0.05·comfort + 0.01·(heading_RMS_deg / 5.0)
genome cost       = mean over scenarios
```
`w_hold` = 0 (v4). Profile: ramp 600 fpm, corners 0.1 g, `alt_ref_ff` true.
Audit on the three smoke bests (`analysis/phase2_cost_audit.py`): every residual of both identities = 0.0; re-flown costs bit-identical to logged. Genome CROSSCHECK_phase2.md: bit-identical on smoke bests.

Pins (`--model-versions` before launch): all match FD `model_versions_post_mass.json`.
Seeds: c172x=1, T38=2, 737=3 (smoke kept seed 1 shared; pilot/bench use distinct seeds).

## Runs (8 workers each, sequential)

| run | wall | load start→end | sims | notes |
|---|---|---|---|---|
| `phase2-pilot-s1` | **2728.6 s** (45m29s) 08:37:38–09:23:07 | 9.18 → 5.10 | 40506 | 64×60, ladders, cache on |
| `phase2-c172x-full-s1` | **355.6 s** (5m56s) 09:23:07–09:29:02 | 5.10 → 7.73 | 1353 | 32×15 full-only, `--no-cache` |
| `phase2-bench-rf-s1` | **180.0 s** (3m0s) 09:29:02–09:32:03 | 7.73 → 8.37 | 2658 | 32×8, rigid→full, cache off |
| `phase2-bench-rrf-s1` | **362.4 s** (6m3s) 09:32:03–09:38:05 | 8.50 → 7.43 | 3684 | 32×8, rigid→reduced→full, cache off |

## Pilot per aircraft (gen 59)

Judged by **total structural cost** (sum of J_*), not the single J_mass flag. `phase2_check.py` exits 1 solely for J_mass < 0; all three total struct sums are positive.

| | c172x | T38 | 737 |
|---|---|---|---|
| **best cost** | **0.22869** | **0.09758** | **0.12594** |
| controller (weighted) | 0.20684 | 0.09434 | 0.12163 |
| structural (Σ J_*) | 0.02186 | 0.00324 | 0.00431 |
| J_mass | −0.01276 | −0.01103 | −0.01043 |
| mass Δ frac / lb | −4.25% / −14.5 lb | −3.68% / −76.1 lb | −3.48% / −928 lb |
| flutter / div / rev | 1.227 / 1.660 / 1.463 | 1.205 / 3.0 / 2.103 | 1.199 / 3.0 / 1.289 |
| invalid all / final | 0.99% / 0% | 0.44% / 0% | 0.65% / 0% |
| evolve wall / CPU | 2721 s / 5237 s | 2379 s / 7206 s | 2390 s / 6835 s |
| sims computed | 12858 | 13866 | 13782 |
| ladder | rigid→reduced→full | rigid→full | rigid→full |
| n_rescored / gen (mean) | 4.5 | 17.1 | 16.7 |
| Spearman screen(rigid)↔full mean | −0.28 | −0.14 | −0.07 |
| Spearman reduced↔full mean | −0.03 | — | — |
| model_version | full:flexv2:11df8fe4 | full:flexv2:e9535bc7 | full:flexv2:2c73464b |

Controller = `track + 2·effort + 0.05·comfort + 0.01·heading` from aggregate terms; verified `ctrl + struct = cost` bit for bit.

### Best-individual genes within 2% of a bound
- **c172x:** `wing_ei_taper_4` at ceiling (1.05).
- **T38:** `ki_alt` at floor (0), `wing_nsm_root/tip` at floor (0.8), `struct_damping_ratio` at ceiling (0.05).
- **737:** `ki_alt` at floor (0), `wing_nsm_root/tip` at floor (0.8).

### Final-population pile share (≥5% shown)
- **c172x:** `wing_ei_taper_4` ceiling **91%**; no stiffness gene at floor.
- **T38:** `wing_nsm_root/tip` floor **91%/92%**; `struct_damping_ratio` ceiling **92%**; `ki_alt` floor **97%**. No stiffness gene at floor.
- **737:** `wing_nsm_root/tip` floor **89%/91%**; `ki_alt` floor **89%**. No stiffness gene at floor.

§12 fix is doing its job: fuselage/tail stiffness stay near 1.0; the free mass lever left is non-structural mass (`wing_nsm_*`, not floored by MIN_GAUGE).

### Structural terms at the optimum (nonzero)
- c172x: J_bm_rms 0.0308, J_fus_bm_limit 0.0020, J_tail_bm_limit 0.0007, J_wing_bm_limit 0.0007, J_smooth 0.0004, J_mass −0.0128.
- T38: J_bm_rms 0.0140, J_fus_bm_limit 0.0002, J_smooth ≈0, J_mass −0.0110.
- 737: J_bm_rms 0.0138, J_fus_bm_limit 0.0005, J_wing_bm_limit 0.0004, J_mass −0.0104, J_flutter_margin ≈0.

## Full-only c172x comparison (`phase2-c172x-full-s1`)
- Best **0.26149** (gen 14), controller-dominated; struct sum 0.0295; mass −1.3%; margins 1.239 / 1.677 / 1.479.
- Wall 355.6 s, CPU 2493 s, 1353 sims, invalid 3.3% / final 0%.
- Vs pilot c172x (0.22869 at gen 59): pilot is **0.033 cheaper**, but on a much larger budget (64×60 ladder vs 32×15 full). Not an apples-to-apples quality comparison.
- Wall efficiency: full-only spent 2493 CPU-s for 15 gens; pilot c172x spent 5237 CPU-s over 60 gens with only ~4.5 full rescored/gen — the ladder cuts full evals by ~14× vs a full-only 64×60 would have.

## Ladder bench: rigid→full (rf) vs rigid→reduced→full (rrf)

| | rf best / wall | rrf best / wall | Δbest (rrf−rf) |
|---|---|---|---|
| c172x | 0.27066 / (shared 180 s) | 0.27781 / (shared 362 s) | **+0.00715** (worse) |
| T38 | 0.11280 | 0.11391 | **+0.00111** (worse) |
| 737 | 0.14937 | 0.14972 | **+0.00036** (worse) |

Wall ratio rrf/rf = **2.01×**. Sims 3684 vs 2658. Reduced↔full Spearman (rrf, mean both-ok): c172x 0.73, T38 0.36, 737 0.42.

**Recommendation:** prefer **rigid→full** for this task. At 32×8, inserting reduced costs ~2× wall and yields slightly worse bests on all three aircraft. Reduced’s ranking agreement with full among the rescored top-k is only middling on the jets. Keep c172x’s three-stage ladder in the pilot only if a longer run shows it helps; the short bench does not.

Caveat: 8 generations is short; ρ and bests are noisy. Revisit if a longer A/B is warranted.

## Trajectories for Sim Bridge (`ga-flightsim-traj/1`, validated OK)
Directory: `evolution/runs/phase2-pilot-s1/trajectories/`
- Index: `index.json` (`ga-flightsim-traj-index/1`)
- Per aircraft, gens 0 / 29 / 59:
  - `traj_c172x_phase2-pilot-s1_g{0,29,59}.json` — full / full:flexv2:11df8fe4
  - `traj_T38_phase2-pilot-s1_g{0,29,59}.json` — full / full:flexv2:e9535bc7
  - `traj_737_phase2-pilot-s1_g{0,29,59}.json` — full / full:flexv2:2c73464b

## Surprises
1. **Rigid↔full Spearman among the rescored elite is ≈0 or negative** in the pilot (−0.28 / −0.14 / −0.07). Expected: rigid ignores structure; within the top quarter already screened by rigid, full re-ranking is driven by structural terms.
2. **`wing_nsm_*` piles at the floor** on T38/737 (~90%). Free mass credit; MIN_GAUGE does not floor non-structural mass (§12). Not a stiffness-floor exploit.
3. **`struct_damping_ratio` → ceiling** on T38 (92%). Harmless for mass; higher damping.
4. **`wing_ei_taper_4` → ceiling** on c172x (91%). Tip stiffening, not a floor pile.
5. **No stiffness gene piled at its floor** — the §12 sizing + min-gauge fix holds under a 64×60 search.
6. **J_mass negative, total structural cost positive** on all three — wing (and some fuselage) mass down, sizing/response terms more than compensate.
7. **`ki_alt` → 0** on T38/737 (89–97% of final pop). Same pattern as Phase-1 jets; the 0.5 upper bound is not the binding side.
8. Distinct seeds eliminated the smoke’s identical gen-0 draws across aircraft.

## Artifacts
- Checks: `analysis/phase2_pilot_s1_check.json`, `phase2_c172x_full_s1_check.json`, `phase2_bench_compare.json`
- Logs: `logs/phase2_pilot_chain.log`, `logs/phase2_pilot_chain.sh`
- Cost audit: `analysis/phase2_cost_audit.py` (smoke residuals = 0)
- Configs: `configs/phase2_{pilot,c172x_full,bench_rf,bench_rrf}.json` (pins from FD file; smoke unchanged)

## Seed 2 (started 12:25 PT) — rigid→full on ALL three
- Config `configs/phase2_pilot_s2.json`: seeds c172x=2, T38=3, 737=4, scenario_seed=2.
- Ladder: global rigid→full, min_full_frac 0.25 (c172x no longer screens reduced; matches ladder-bench recommendation).
- Pins verified match. Run `phase2-pilot-s2`, code_sha b2982bc8250965d2, 8 workers.
- Note: s1 c172x used rigid→reduced→full; s2+ use rigid→full — c172x cross-seed compare is not identical ladders.
- Stop after s2 for Sim Bridge ping; do NOT start s3 until resumed.

## Seed 2 finished (12:25–13:14 PT) — EXIT 0, wall 2985.4 s
- Run `phase2-pilot-s2`, code_sha b2982bc8250965d2, load 0.41→7.26, 41304 sims / 2202 cache hits, 8 workers.
- Ladder: **rigid→full on all three** (c172x no reduced). Seeds c172x=2, T38=3, 737=4, scenario_seed=2.
- Trajectories: `runs/phase2-pilot-s2/trajectories/` (g0/g29/g59), schema ga-flightsim-traj/2 + FlexState/3 from this run.
- Check: `analysis/phase2_pilot_s2_check.json`.

| | c172x | T38 | 737 |
|---|---|---|---|
| **best cost** | **0.21551** | **0.09771** | **0.12413** |
| controller (weighted) | 0.19974 | 0.09264 | 0.12125 |
| structural (Σ J_*) | 0.01577 | 0.00507 | 0.00288 |
| J_mass | −0.01598 | −0.00908 | −0.01029 |
| mass Δ frac / lb | −5.33% / −18.2 lb | −3.03% / −62.7 lb | −3.43% / −916 lb |
| flutter / div / rev | 1.243 / 1.686 / 1.487 | 1.200 / 3.0 / 2.093 | 1.198 / 3.0 / 1.293 |
| invalid all / final | 0.83% / 0% | 0.21% / 0% | 0.65% / 0% |
| evolve wall / CPU | 2967 s / 7651 s | 2934 s / 7300 s | 2954 s / 7223 s |
| sims computed | 13683 | 13794 | 13827 |
| ladder | rigid→full | rigid→full | rigid→full |
| n_rescored / gen (mean) | 16.2 | 16.8 | 16.9 |
| Spearman screen(rigid)↔full mean | +0.08 | −0.06 | −0.10 |
| model_version | full:flexv2:11df8fe4 | full:flexv2:e9535bc7 | full:flexv2:2c73464b |

Controller = sim_cost mean; verified `ctrl + struct = cost` bit for bit.

### Best-individual genes within 2% of a bound
- **c172x:** `wing_gj_ratio_root` near ceiling.
- **T38:** `ki_alt` at floor (0), `wing_nsm_root/tip` at floor (0.8).
- **737:** `ki_alt` at floor (0), `wing_nsm_root/tip` at floor (0.8), `struct_damping_ratio` near ceiling.

### Final-population pile share (≥5% shown)
- **c172x:** `wing_nsm_root` floor **20%**; `wing_gj_ratio_root` ceiling **45%**. No stiffness gene at floor.
- **T38:** `wing_nsm_root/tip` floor **89%/94%**; `ki_alt` floor **89%**. No stiffness gene at floor.
- **737:** `wing_nsm_root/tip` floor **80%/92%**; `ki_alt` floor **62%**; `struct_damping_ratio` ceiling **91%**; `wing_gj_ratio_root` ceiling **22%**. No stiffness gene at floor.

### Structural terms at the optimum (nonzero)
- c172x: J_bm_rms 0.0289, J_fus_bm_limit 0.0017, J_wing_bm_limit 0.0006, J_tail_bm_limit 0.0003, J_smooth 0.0002, J_mass −0.0160.
- T38: J_bm_rms 0.0138, J_fus_bm_limit 0.0002, J_smooth 0.0001, J_mass −0.0091.
- 737: J_bm_rms 0.0131, J_flutter_margin ≈0, J_mass −0.0103.

`phase2_check` exits 1 only for J_mass < 0; total structural cost ≥ 0 on all three. Flag for FD: wing_nsm_* at 0.8 floor on T38/737 (already pinged).

## Twist-sign fix (schema /3 modal rename) — DONE
- `fd_to_structure_channels`: `name.startswith("wingR")` so `wingR_modal.twist` keeps + for nose-up (FE `wingR.twist` unchanged).
- Regression: `tests/test_v2_map.py::test_modal_twist_sign_after_rename` (passes).
- `test_pilot_ladders_gates_and_budget` updated for global rigid→full pilot (no per-aircraft mfa on c172x).

## s1 traj re-export to /2 + FlexState/3 — DONE
- Script: `analysis/reexport_traj_schema3.py`. Backup: `runs/phase2-pilot-s1/trajectories_traj1/` (old /1 copies).
- All 9 files: ga-flightsim-traj/2 + evolution-flex-state/3, comps wingR_modal/wingL_modal/wingR/wingL (+tails/fus), `reexported=schema3_v2_map`.
- Costs bit-identical (mismatches []). Did not re-run GA.

## Seed 3 finished (13:23–14:12 PT) — EXIT 0 (resume after FD P2.5 mid-run abort)
- First session aborted at gen ~40 when FD landed P2.5 (hashes drifted). Resume with `EVOLUTION_FD_DIR=evolution/_fd_pin_post_mass` (pre_p25 sources; **did not write** into `flight-dynamics/`). Pins stayed post-mass: c172x `11df8fe4`, T38 `e9535bc7`, 737 `2c73464b`.
- Wall: session0 1987.6 s (abort) + session1 898.9 s resume ≈ **2886.5 s** total; resume sims 13002 / 1290 cache hits. code_sha 78ed97ac7b22eb67.
- Ladder: rigid→full all three. Seeds c172x=3, T38=4, 737=5, scenario_seed=3.
- Check: `analysis/phase2_pilot_s3_check.json`. Trajectories: `runs/phase2-pilot-s3/trajectories/`.

| | c172x | T38 | 737 |
|---|---|---|---|
| **best cost** | **0.20196** | **0.09210** | **0.12038** |
| controller / structural | 0.19300 / 0.00896 | 0.08952 / 0.00258 | 0.11774 / 0.00264 |
| J_mass | −0.02053 | −0.01107 | −0.01069 |
| mass Δ frac / lb | −6.84% / −23.4 lb | −3.69% / −76.4 lb | −3.56% / −952 lb |
| flutter / div / rev | 1.203 / 1.636 / 1.440 | 1.204 / 3.0 / 2.116 | 1.199 / 3.0 / 1.289 |
| invalid all / final | 1.04% / 0% | 0.10% / 0% | 0.60% / 0% |
| evolve wall / CPU | 2801 / 7581 s | 2802 / 7118 s | 2804 / 7127 s |
| sims computed | 13551 | 13626 | 13722 |
| n_rescored / gen | 16.6 | 16.7 | 17.2 |
| Spearman rigid↔full | −0.04 | +0.04 | −0.15 |
| wing_nsm floor pile | **86%/94%** | **88%/94%** | **88%/89%** |
| stiffness at floor | none | none | none |
| model_version | full:flexv2:11df8fe4 | full:flexv2:e9535bc7 | full:flexv2:2c73464b |

### Best-individual genes near bounds
- **c172x:** `wing_nsm_root/tip` at floor (0.8).
- **T38:** `ki_alt`, `ki_pitch`, `wing_gj_ratio_root`, `wing_nsm_*` floors.
- **737:** `wing_nsm_*` floors, `struct_damping_ratio` near ceiling.

## Seeds 1–3 aggregate (mean ± std, ddof=1)
Artifact: `analysis/phase2_pilot_seeds.json`. **Note:** s1 c172x used rigid→reduced→full; s2/s3 rigid→full everywhere. All three on **post-mass** pins (nsm floor 0.8).

| | best cost | mass Δ frac | flutter margin |
|---|---|---|---|
| c172x | 0.21539 ± 0.01337 | −5.47% ± 1.30% | 1.224 ± 0.020 |
| T38 | 0.09580 ± 0.00321 | −3.46% ± 0.38% | 1.203 ± 0.002 |
| 737 | 0.12348 ± 0.00284 | −3.49% ± 0.07% | 1.199 ± 0.001 |

Per-seed bests: c172x 0.22869 / 0.21551 / 0.20196; T38 0.09758 / 0.09771 / 0.09210; 737 0.12594 / 0.12413 / 0.12038.

## P2.5 smoke `phase2-smoke-p25-s1` — EXIT 0, wall 153.6 s (14:14–14:16 PT)
- Pins from `model_versions_post_p25.json`: c172x `e11b8214`, T38 `8bf7a250`, 737 `eeb82fb9`. Configs via `make_phase2_configs.py --p25` (**only** smoke_p25 + pilot_p25; post-mass configs untouched).
- Evolution `TERM_KEYS` = 24 (+ `J_wing_tip_bm_limit`). code_sha b7ec79967423ec7f.
- Genome tip-verify re-fly (`analysis/p25_tip_verify_evolution.json`): **bit-identical** total / per-scenario / tip / mv vs Genome pair; soft-tip `J_wing_tip_bm_limit=0.017509…` matches. Pre-P2.5 `wing_nsm_*=0.9` **rejects** (`outside [1.0, 1.25]`).
- Bests (g4): c172x **0.33002**, T38 **0.22571**, 737 **0.25648**. Check: `analysis/phase2_smoke_p25_s1_check.json` (no flags).
- **nsm still piles** at new floor 1.0 (final pop root/tip ≈ c172x 81%/50%, T38 81%/62%, 737 56%/56%). Cannot lighten below baseline via nsm; best J_mass ≥ 0 on optima.
- **Tip BM:** controlled soft-tip (taper4=0.75) costs tip BM as Genome showed. Smoke GA (baseline±0.10) never reached taper4≤0.80; tip BM still nonzero on many evals (max ~0.28 on 737) via other soft-structure paths; optima kept tip BM ≈0.
- **No 64×60** (`phase2_pilot_p25` written but not launched).
