# STATUS D: Sim Bridge replay interface (verified 2026-10-06 07:4x PT)

All items exist; tests listed. Changes this session: run.json gains `individual_id_format` and `scenario_id_format`
(runinfo.py); genomes.jsonl rows gain `eval_seed` (batch.py; backfill rows too); test added in
`test_evaluate_reproduces_genomes_jsonl_exactly` (id format, is_best == rank 0, scenario ids, eval_seed).

* run.json (`ga-flightsim-run/1`): run_id, schema, git_sha, jsbsim_version, code_sha, seed, eval_seed, fitness_sense "min",
  sim_dt_s, aircraft[] (resolved_profile, aircraft_root, scenario_ids, genes, model_files_sha, model_version,
  screen/ladder_model_version, reduced_gate, min_full_frac, fitness_cfg), scenarios[] (every Scenario field, absolute
  steps, ramp_plan, draft fields when present), target_semantics, fidelity, model_version, eval entry,
  individual_id_format, scenario_id_format.
* genomes.jsonl (`ga-flightsim-genomes/1`): one row per individual per generation; fsync before checkpoint; truncated
  to the checkpoint on resume (no duplicates: test_resume, test_multi_fidelity_rescoring_ranking_and_resume).
* eval.evaluate(genome, aircraft, scenario, run_cfg, recorder=None, *, fidelity=None): tests
  test_evaluate_reproduces_genomes_jsonl_exactly, test_reduced_fidelity_run_reproduces_and_logs,
  test_multi_fidelity_* (full rows replay), test_recorder_bit_identical_t0_call_and_pre_step_controls,
  test_recorder_is_read_only, test_recorder_inert_t0_flex_state_and_twist_sign.
* controls_timing "pre_step" + channel_doc: test_trajectory_doc_channel_doc_and_controls_timing, test_channel_doc_flex_raw_and_sb_channels.
* flex_state `evolution-flex-state/2` documented in README; twist: FD raw + = LE up on both wings; SB wingR = +raw,
  wingL = -raw (test_twist_convention_known_nose_up_case).
* Ids: `<aircraft>:g<generation>:r<rank>` (backfill `<aircraft>:g<generation>:best`), scenario `<aircraft>:s<index>`;
  unchanged (not broken: unique per run, stable across resume). Sim Bridge's replay now treats ids as opaque strings
  (sim-bridge/sim_bridge/replay.py docstring, sim-bridge/tests/test_ids.py).

## Follow-ups (07:51 PT steering)
1. scenario_index = position in the aircraft's scenario_ids (batch passes s_idx into st["scenarios_d"], built in
   make_scenarios order = scenario_id(ac, i)). Trajectory headers now also carry `scenario_id`.
2. ids unique per run, stable across resume (truncate + deterministic regeneration, resume tests). `:best` only in
   backfilled best-only runs (no population ranking); `:r0` = best in new runs. Intended; documented (README + run.json).
3. trajectory `fitness_sense` is now exactly "min" (+ `fitness_doc`). Tested.
4. run.json aircraft_root (and resolved_profile.aircraft_root) and git.repo are team-relative (`runinfo.team_rel`,
   `paths_relative_to`). Tested; replay via eval.evaluate still exact.
5. rigid model_version hashes every file in <root>/aircraft/<model>/ (incl. flexwing_meta.json, init XMLs). The
   c172x 01333e0c -> e0a73fc9 move is from FD editing c172x.xml (04:57 PT, flight-neutral 0 lb flex point masses);
   T38/737 at 04:48. Not narrowed: FD's flexeval.rigid_model_version computes the identical string independently;
   narrowing one side would desync, and an XML-only hash would have changed anyway.
6. Unrescored multi-fidelity rows: cost/fidelity/model_version = highest stage reached (screen or middle); new
   per-row `ladder_model_version {fid: mv}` (each fidelity has its own); run.json has aircraft[].ladder_model_version.
7. Every row now has eval_seed; one scenario seed per run, so run.json eval_seed applies to older rows without it.
8. v2_map.py: standalone (no evolution import), so importing it would not be circular. Recommendation: keep the wing
   mapping in evolution/fidelity.py (it uses FD's modal shapes at 9 nodes; v2_map defers to it for wings), and import
   v2_map read-only, optional, for htail/vtail/fuselage + struct.* scalars, rather than duplicating it (one source of
   truth per component). Not wired in this session.
