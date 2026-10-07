# STATUS F: path changes ported from /workspace/phase1-push-staging/repo/flight_sim_3d/evolution/ (2026-10-06 ~07:41 PT)

Snapshot is read-only and OLDER than our tree (pre-FD-v2, pre-v5); only its path changes were ported.
Pre-port copies: /tmp/preF/{batch,bench,fidelity,sim,conftest,test_equivalence}.py.

Ported (work in both layouts; team layout resolves to exactly the same strings as before):
* batch.py: `DEFAULT_SOURCE_REPO = _default_source_repo()`: $EVOLUTION_SOURCE_REPO, else the repo-relative
  `dirname(dirname(PKG_DIR))` when it contains `flight_sim/` (push layout), else the legacy sandbox clone (team layout,
  unchanged value -> run-id hashes unchanged). Docstring `cd flight_sim_3d`.
* fidelity.py: `FD_DIR` = $EVOLUTION_FD_DIR or `<dir of evolution/>/flight-dynamics` (= /workspace/flight-sim-team/flight-dynamics here).
* sim.py: `TEAM_ROOT`, `abs_root()`; `Profile.from_dict` makes a relative aircraft_root absolute; `model_dir` and
  `FGFDMExec(...)` go through abs_root.
* tests/test_equivalence.py: ORIG = $FLIGHT_SIM_DIR or `<DEFAULT_SOURCE_REPO>/flight_sim` (snapshot: `<repo>/flight_sim`).
* tests/test_heading.py, tests/test_phase1.py: export aircraft_root passed through sim.abs_root (no-op for absolute).
* bench.py docstring, tests/conftest.py comment.
* NEW tests/test_paths.py (abs_root, relative == absolute profile, FD_DIR default, source repo in both layouts).

Side effect: sim.py + fidelity.py are in cache.code_sha, so code_sha changed: runs made before the port refuse
`--resume` (by design) and old eval-cache entries miss. Finished runs/replays are unaffected (model_version does not
include code_sha).

Differences NOT copied (listed per instructions):
* configs/phase1.json, phase1_hdg.json: snapshot uses `"aircraft_root": "flight-dynamics/jsbsim_root"`; ours keep the
  absolute path because they must equal genome/exports (absolute here). Now that abs_root exists, relativising them at
  push time needs no code change (note: the raw profile text is in the run-id hash, so default run ids change; cache keys
  use the resolved absolute path and do not).
* configs/phase1_smoke.json (snapshot only: pop 8 x 3 gens smoke config).
* README.md, batch.py/eval.py/fidelity.py/runinfo.py/trajectory.py/tests/test_eval.py: remaining differences are the
  snapshot's older code (single-screen multi-fidelity, full(v1) stand-in, older docs) that our tree supersedes.
* Only in ours: analysis/bench_fastmode*.py, bench_fdv2*.py, microbench_fidelity.py, phase1v5_report.py, tests/test_v5.py, configs/phase1_v5.json.

## Follow-ups (07:51 PT steering)
* `sim.TEAM_ROOT` override: $EVOLUTION_TEAM_ROOT (or $FLIGHT_SIM_TEAM_ROOT); relative roots resolve against it.
* fidelity.FD_DIR = $EVOLUTION_FD_DIR or `<TEAM_ROOT>/flight-dynamics`; no absolute path left in fidelity.py.
* run.json: aircraft_root, resolved_profile.aircraft_root, git.repo team-relative (`runinfo.team_rel`), `paths_relative_to`.
* configs/phase1_v5.json now uses genome's repo-relative aircraft_root (export changed); v4/phase1 exports still absolute.
* Remaining absolute default: batch._LEGACY_SOURCE_REPO (team-layout fallback only; unused in the push layout).
