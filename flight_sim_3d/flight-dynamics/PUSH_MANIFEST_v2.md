# Flight Dynamics flex v2 (Phase 2): push manifest

**Status: NOT pushed.** This is a list for whoever does the Phase-2 push, after approval. Flight Dynamics made no git push,
no PR and no master change. Sizes and hashes are those of the local working copy at
`/workspace/flight-sim-team/flight-dynamics/` at the time of writing. The target is
`flight_sim_3d/flight-dynamics/` (same layout as the Phase-1 staging).

Test command (from that directory): `python -m pytest -q test_flexwing.py test_flexbody.py` → **105 passed** locally.

## 1. v2 code and tests (push)
| file | bytes | sha256[:12] |
|---|---|---|
| `flexbody.py` | 102965 | `43b46360cf14` |
| `flexeval.py` | 31092 | `1180c7583bd5` |
| `test_flexbody.py` | 37876 | `d991cc512653` |
| `legacy_fingerprint.py` | 1979 | `e35915a7cadb` |
| `v1_legacy_fingerprint.json` | 702 | `2cb2904c10b4` |

`legacy_fingerprint.py` / `v1_legacy_fingerprint.json` back the v1 bit-identity test in `test_flexbody.py`.

## 2. Prepared v2 aircraft: `jsbsim_root_v2/` (push these 26 regular files, 219243 bytes; not the symlinks)
| file | bytes | sha256[:12] |
|---|---|---|
| `jsbsim_root_v2/aircraft/737/737.xml` | 33242 | `bb46c65a1f59` |
| `jsbsim_root_v2/aircraft/737/cruise_init.xml` | 489 | `7f2ebb073c6f` |
| `jsbsim_root_v2/aircraft/737/cruise_steady_turn_init.xml` | 471 | `9cbc34823ae6` |
| `jsbsim_root_v2/aircraft/737/flexbody_meta.json` | 597 | `df0edf2fdf2d` |
| `jsbsim_root_v2/aircraft/737/flexwing_meta.json` | 134 | `0e3e82793ce4` |
| `jsbsim_root_v2/aircraft/737/reset00.xml` | 566 | `9794e588609b` |
| `jsbsim_root_v2/aircraft/737/rudder_kick_init.xml` | 372 | `79750a266657` |
| `jsbsim_root_v2/aircraft/T38/T38.xml` | 31245 | `5b1d1dcb3346` |
| `jsbsim_root_v2/aircraft/T38/flexbody_meta.json` | 585 | `4dc9ab9cdd7d` |
| `jsbsim_root_v2/aircraft/T38/flexwing_meta.json` | 133 | `800ac3c85b13` |
| `jsbsim_root_v2/aircraft/T38/reset00.xml` | 548 | `9563915170f7` |
| `jsbsim_root_v2/aircraft/c172x/c172ap.xml` | 9456 | `1e83fec37bea` |
| `jsbsim_root_v2/aircraft/c172x/c172x.xml` | 70645 | `eafee1353ade` |
| `jsbsim_root_v2/aircraft/c172x/elevator_doublet_init.xml` | 445 | `bb4ac1901302` |
| `jsbsim_root_v2/aircraft/c172x/flexbody_meta.json` | 592 | `50453a7710ec` |
| `jsbsim_root_v2/aircraft/c172x/flexwing_meta.json` | 101 | `5d12972e62c5` |
| `jsbsim_root_v2/aircraft/c172x/output.xml` | 644 | `c2364918e724` |
| `jsbsim_root_v2/aircraft/c172x/reset00.xml` | 627 | `f2520031e2a6` |
| `jsbsim_root_v2/aircraft/c172x/reset01.xml` | 332 | `26d4f4ee0f10` |
| `jsbsim_root_v2/aircraft/c172x/reset_at_rest.xml` | 564 | `89136139c132` |
| `jsbsim_root_v2/aircraft/f16/Systems/hook.xml` | 2311 | `9067ea104820` |
| `jsbsim_root_v2/aircraft/f16/Systems/pushback.xml` | 1192 | `a3c0e2fc8a49` |
| `jsbsim_root_v2/aircraft/f16/f16.xml` | 62602 | `e364589643a1` |
| `jsbsim_root_v2/aircraft/f16/flexbody_meta.json` | 634 | `d1e8bc1c2785` |
| `jsbsim_root_v2/aircraft/f16/flexwing_meta.json` | 168 | `dd56771779eb` |
| `jsbsim_root_v2/aircraft/f16/reset00.xml` | 548 | `738c7d0f3495` |

`jsbsim_root_v2/engine` and `jsbsim_root_v2/systems` are **symlinks** to the venv's installed JSBSim data. Do NOT push their
targets; recreate the links on the target the same way as Phase 1's `jsbsim_root` (`link_jsbsim_data.py` in the Phase-1
staging): `jsbsim_root_v2/engine` -> `/workspace/sandbox-run-20261006-023947/venv/lib/python3.13/site-packages/jsbsim/engine`; `jsbsim_root_v2/systems` -> `/workspace/sandbox-run-20261006-023947/venv/lib/python3.13/site-packages/jsbsim/systems`.

## 3. Tools (push)
| file | bytes | sha256[:12] |
|---|---|---|
| `v2_compare.py` | 14346 | `9ff27c83e788` |
| `v2_validate.py` | 8178 | `30b651ed6b32` |
| `_oas_check.py` | 5395 | `11eca03e8bf6` |
| `v2_exploit_check.py` | 5630 | `c9540ad3020e` |

`_oas_check.py` needs a separate OpenAeroStruct venv (`_oas_venv`, NOT pushed). It is optional: its result is saved in
`v2_results/oas_raw.json` / `v2_validation.json`.

## 4. Docs (push)
| file | bytes | sha256[:12] |
|---|---|---|
| `INTERFACE_v2.md` | 42194 | `85ca76db6aec` |
| `PUSH_MANIFEST_v2.md` | this file | |

Append-only changes to the v1 docs. The v1 prefix is bit-identical; push only the appended tail on top of the Phase-1 copy:

| file | bytes | hash |
|---|---|---|
| `INTERFACE.md` | 41795 (v1 prefix 41339 bytes unchanged + 456 bytes appended) | `456e58b985e9` (appended bytes) |
| `AEROELASTIC_DESIGN.md` | 20678 (v1 prefix 15548 bytes unchanged + 5130 bytes appended) | `ba36ea6f9682` (appended bytes) |

Note: the Phase-1 staging copy of `INTERFACE.md` has line 3 rewritten to a repo-relative genome path. Apply the
appended tail to the staged copy, not the whole local file. No v2 file contains an absolute `/workspace` or `/tmp` path
(checked with rg).

## 5. Results (push, small JSON)
| file | bytes | sha256[:12] |
|---|---|---|
| `v2_benchmarks.json` | 5968 | `c45053874484` |
| `spearman.json` | 15550 | `6b301de21562` |
| `v2_validation.json` | 13102 | `f4b4a51e0795` |
| `v2_results/oas_raw.json` | 14683 | `2af652bc9037` |
| `v2_results/sign_probe.json` | 30926 | `892b439efa0a` |
| `v2_results/exploit_check_before.json` | 10341 | `84d7e084f00f` |
| `v2_results/exploit_check_after.json` | 11277 | `e530e18df2f1` |
| `v2_results/sizing_baseline.json` | 4532 | `46075865fdb0` |
| `v2_results/model_versions_post_mass.json` | 516 | `597546bdd89f` |

Raw checkpoints behind `v2_benchmarks.json` / `spearman.json` (optional, 530875 bytes total):
| file | bytes | sha256[:12] |
|---|---|---|
| `v2_results/bench_737.jsonl` | 3716 | `c4630601a5ed` |
| `v2_results/bench_T38.jsonl` | 3708 | `edbfbb7faf39` |
| `v2_results/bench_c172x.jsonl` | 3703 | `56630d59c80e` |
| `v2_results/bench_f16.jsonl` | 3725 | `80b948f6df60` |
| `v2_results/rank_737_genomes.s0.jsonl` | 55284 | `fe31f0224e49` |
| `v2_results/rank_737_genomes.s1.jsonl` | 54430 | `8ea3f75971fd` |
| `v2_results/rank_737_joint.s0.jsonl` | 71892 | `92ec5812f04e` |
| `v2_results/rank_737_joint.s1.jsonl` | 72314 | `84644998465c` |
| `v2_results/rank_c172x_genomes.s0.jsonl` | 57197 | `6e00661402f2` |
| `v2_results/rank_c172x_genomes.s1.jsonl` | 56133 | `659154e6673b` |
| `v2_results/rank_c172x_joint.s0.jsonl` | 74560 | `cdd8a832dc20` |
| `v2_results/rank_c172x_joint.s1.jsonl` | 74213 | `3310341157da` |

## 6. Required but unchanged (v1, already in Phase 1, bit-identical; do NOT re-push)
| file | bytes | sha256[:12] |
|---|---|---|
| `flexwing.py` | 62182 | `10d723a0cf31` |
| `coupled_sim.py` | 9704 | `d1b0006f50c0` |
| `test_flexwing.py` | 20399 | `ebe35dc35fa3` |
- `jsbsim_root/` (22 files): unchanged.

## 7. Do NOT push
- `_scratch/`: backups (`*.pre_*`, `*.post_*`), probe/timing scripts, the v1 md5 snapshot.
- `_oas_venv/` (≈ 383 MB OpenAeroStruct venv).
- `__pycache__/`, `.pytest_cache/`.
- All logs: `*.log` at the top level and in `v2_results/` (`pytest_*.log`, `exploit_*.log`, `bench_seq.log`, `validate*.log`,
  `rank_*.log`).
- `v2_results/bench_parallel_pre_taper/`, `v2_results/bench_seq_wall_only/` (superseded benchmark runs).
- `/tmp/oldteam/` (the symlinked pre-fix copy used for the "before" GA run).
- Phase-1-only artefacts already pushed (fm_quality*, flex_demo*, phase1_check*, f16_check*, socket_*, verify_*,
  external_reactions_proof.json, gene_range_margins.json): unchanged by v2.

## 8. Cross-team notes for the push
- `model_version` (reduced/full) changed with the mass-exploit fix, so caches keyed on older FD versions are stale. Final
  values are in `v2_results/model_versions_post_mass.json`; rigid is unchanged.
- `flexeval.TERM_KEYS` now has 23 keys. Genome's `fitness.structural_v2` / `V2_HINGE_TERMS` should add
  `J_wing_bm_limit, J_wing_torque_limit, J_wing_ip_limit, J_tail_bm_limit, J_fus_bm_limit, J_wing_torque_peak,
  J_wing_ip_peak` (INTERFACE_v2 §12). Genome code was not edited by FD.
