# evolution: multi-aircraft GA batch runner for the JSBSim flight-sim prototype

Scales the `flight_sim/` prototype (PoppaMas/HTML5_Genetic_Cars, branch
`flight-sim-prototype`, commit `5850721`) from one aircraft to a batch of
aircraft. It adds a shared process pool, an on-disk result cache, per-generation
checkpoints with exact resume, run logging, and 3D-viewer trajectory export.
The GA operators, gene encoding, controller, and cost function are the
prototype's own code, vendored here. **With the `baseline` profile on `c172x` it
reproduces the prototype bit-for-bit.** `evolve.py --pop-size 24
--generations 15 --seed 1` and `configs/c172x_equiv.json` give the same best
cost, `0.19833289735948487`, and the same gains (`tests/test_equivalence.py`
checks this on a smaller run).

```
evolution/
  batch.py          CLI + orchestrator (python -m evolution.batch)
  sim.py            JSBSim evaluation, generalised: Profile (IC, envelope, cost, gain bounds), trim-failure
                    handling, socket-free model loading, 30 Hz record mode; Phase-1 options (ramped reference,
                    climb-rate feed-forward, comfort term, roll gains, external JSBSim root), all off by default
  ga.py, genome.py  vendored from flight_sim/ (ga.py unchanged; genome.py adds per-profile gain-bound / kind
                    overrides and the log0 kind)
  cache.py          sqlite evaluation cache + key definition
  trajectory.py     ga-flightsim-traj/2 export + altitude-hold metrics
  validate_traj.py  stand-alone format validator (python -m evolution.validate_traj)
  bench.py          benchmark driver (fresh run + cache rerun + schedule comparison -> table)
  configs/          c172x_equiv.json, bench_baseline.json, bench_ic.json, bench_adjusted.json,
                    bench_jets.json, bench_jets_step200.json, phase1.json, seeds/ (seed-robustness variants)
  seed_table.py     stage B vs C comparison over seeds 1-3
  phase1_report.py  Phase-1 seed study tables (python -m evolution.phase1_report)
  tests/            pytest suite (equivalence, cache, determinism, resume after SIGKILL, trajectory schema, sanitising)
  runs/<run_id>/    experiment output (see below)
  cache/            sqlite caches
  bench_results/    benchmark tables (results.json, table.md)
```

## How to run

Uses the existing venv (`jsbsim 1.3.1`, numpy, pytest). Run from the parent
directory so that `evolution` is importable:

```bash
cd /workspace/flight-sim-team
PY=/workspace/sandbox-run-20261006-023947/venv/bin/python

$PY -m evolution.batch --config evolution/configs/bench_adjusted.json          # a batch of 4 aircraft
$PY -m evolution.batch --config evolution/configs/bench_adjusted.json          # same command again: resumes, or no-op if finished
$PY -m evolution.batch --resume evolution/runs/<run_id>                         # resume from the stored config.json
$PY -m evolution.batch --config X.json --run-id my-rerun                        # fresh run dir; still hits the cache
$PY -m evolution.batch --config evolution/configs/phase1.json --seed 2 --run-id phase1-s2   # GA seed override
$PY -m evolution.batch --config X.json --workers 4 --schedule sequential --no-cache   # execution-only overrides
$PY -m evolution.validate_traj evolution/runs/<run_id> --min-gens-per-aircraft 3
$PY -m pytest evolution/tests -q
$PY -m evolution.bench --tag T --schedules evolution/configs/bench_{baseline,ic,adjusted}.json   # needs a new tag

# heading hold (opt-in config), fast mode and fidelity (see the sections below)
$PY -m evolution.batch --config evolution/configs/phase1_hdg.json --seed 1 --run-id phase1hdg-s1
$PY -m evolution.batch --config X.json --viz on                                  # record every eval + live best file (slow)
$PY -m evolution.batch --config X.json --fidelity full --struct-genes             # FD flexeval flex v2 (+ FD's 12 v2 struct genes)
$PY -m evolution.batch --config X.json --fidelity full --struct-genes --multi-fidelity --screen rigid,reduced   # ladder rigid->reduced->full
$PY -m evolution.batch --config X.json --fidelity full --struct-genes --multi-fidelity --screen rigid --top-k 4
$PY -m evolution.runinfo --backfill evolution/runs/phase1-s1                      # run.json + best-of-gen rows for old runs
```

The default run id is `<config-stem>-<8-hex hash of the resolved config>`.
Running the same config again therefore resumes the same run. If that run
already finished, the command re-exports the trajectories and summary from the
checkpoints and computes nothing. Use `--run-id` to get a fresh run directory.
The exit code is 1 if any aircraft raised an error. An aircraft that fails to
trim or load is *skipped*, its reason goes into `summary.json`, and that is not
treated as an error.

### Paths (both layouts)

The code has no absolute `/workspace` defaults that a repo push would break (ported from the push snapshot):
* **Team root** `sim.TEAM_ROOT` = the directory that holds `evolution/` and `flight-dynamics/` (`flight_sim_3d/` in the
  repo, `/workspace/flight-sim-team` here); override `$EVOLUTION_TEAM_ROOT` (or Sim Bridge's `$FLIGHT_SIM_TEAM_ROOT`).
* `sim.abs_root()`: a relative `aircraft_root` (e.g. `"flight-dynamics/jsbsim_root"`, as genome/'s v5 export now
  writes it) is resolved against the team root.
  `Profile.from_dict` stores the absolute path, so cache keys are the same for relative and absolute spellings (the raw
  config text is in the run-id hash, so the default run id differs). Each config spells the root exactly as genome/'s
  export does (v4/phase1: absolute; v5: relative).
* **run.json** writes `aircraft_root` (also inside `resolved_profile`) and `git.repo` relative to the team root
  (`runinfo.team_rel`; `paths_relative_to` says so). `sim.abs_root` / Sim Bridge's `paths.resolve_model_root` resolve them.
* `fidelity.FD_DIR` = `$EVOLUTION_FD_DIR`, else `<team root>/flight-dynamics` (no absolute default).
* `batch.DEFAULT_SOURCE_REPO` = `$EVOLUTION_SOURCE_REPO`, else the repo root when it has `flight_sim/`, else the sandbox
  clone (team layout, same value as before).
* `tests/test_equivalence.py` finds the prototype at `$FLIGHT_SIM_DIR`, else `<source repo>/flight_sim`.

## Config schema (`batch.json`)

Keys starting with `_` are comments and are ignored. Unknown keys are errors.

| key | default | meaning |
|---|---|---|
| `run_id` | `null` | `null` = `<stem>-<hash>` |
| `runs_dir` | `"runs"` | relative paths are relative to the `evolution/` package dir |
| `cache` | `{"enabled": true, "path": "cache/evals.sqlite"}` | on-disk eval cache (*execution-only*) |
| `workers` | `0` | pool size; 0 = usable cores (*execution-only*) |
| `schedule` | `"concurrent"` | `concurrent`: all aircraft share the pool at once. `sequential`: one aircraft at a time, individuals in parallel (*execution-only*) |
| `seed` | `1` | GA seed. Each aircraft uses its own numpy Generator seeded with this, or with the aircraft's own `seed` |
| `scenario_seed` | `null` (= seed) | seed for the disturbance scenarios (wind, turbulence, gust) |
| `scenarios` | `3` | scenarios averaged per fitness. Scenario 0 is calm air |
| `ga` | `pop_size 24, generations 15, elite 2, selection_p 0.2, crossover uniform, blx_alpha 0.3, mutation_rate 0.15, mutation_sigma 0.08, mutation_mode gauss` | same meaning as in `flight_sim/evolve.py` |
| `ga.shape_crossover` | `"block"` | genome_kind `phase3_b1` only. `"block"` = whole-block crossover (Genome's B1 spec; dropped from the resolved config, so old run ids / resume are unchanged); `"uniform"` = opt-in **tweaked preset** (with `ga.elite: 4`; Genome's `phase3_b1_x`): per-gene uniform crossover inside the shape block, controller / structure whole, one `rng.random(8)` per child. Spec: `analysis/TWEAKED_PRESET_SPEC.md`; configs `configs/phase3b1_pilot_tweaked*.json` |
| `trajectories` | `{"generations": "auto", "scenario": 0, "sample_hz": 30}` | `auto` = `[0, (G-1)//2, G-1]`. A list adds generations; the final one is always saved |
| `metrics` | `{"band_ft": 20, "hold_after_s": 20}` | settling band, and the start of the "hold" window after each target step |
| `source_repo` | `$EVOLUTION_SOURCE_REPO`, else the repo root (push layout `<repo>/flight_sim_3d/evolution`), else the sandbox clone | its git sha / branch / dirty flag are recorded |
| `profiles` | `{"baseline": {}}` | name -> `sim.Profile` overrides |
| `aircraft` | `[{"name": "c172x", "profile": "baseline"}]` | list of `{name, profile, overrides?, seed?}`. Each JSBSim model may appear once per batch |
| `fidelity` | `"rigid"` | `rigid` (JSBSim only) / `reduced` (FD flexeval: v1 wing on the projected v2 genome, per-aircraft gate) / `full` (FD flexeval: flex v2, gate 1.0) / `full_a1` (FD flexeval_a1: P3-A1 64-strip model, gate 1.0) |
| `fidelity_per_aircraft` | `{}` | `{name: {reduced_gate, min_full_frac}}` overrides of `fidelity.DEFAULT_PER_AIRCRAFT` (c172x 0.9 / 0.0; 737, T38, f16 1.0 / 0.25) |
| `struct_genes` | `false` | append FD's `STRUCT_SCHEMA` genes (`stiffness_scale`, `torsion_bend_ratio`, `zeta`, `nonstruct_scale`) to the genome |
| `multi_fidelity` | `{"enabled": false, "screen": "reduced", "top_k": 4, "min_full_frac": null, "mid_k": null}` | ladder screen(s) → `fidelity`; `screen` = a fidelity or an ascending list; full re-scores max(top_k, ⌈min_full_frac·pop⌉) + elites (min_full_frac null = per aircraft) |
| `viz` | `"off"` | `on` = trajectory recorded on every evaluation + `runs/<id>/live/<aircraft>.json` each generation (*execution-only*) |

*Execution-only* keys can't change results. They are excluded from the run-id
hash and from the resume consistency check, so you can resume with a different
worker count or schedule.

**Profile fields** (`sim.Profile`). The defaults are exactly the prototype's
c172x constants, so `"baseline": {}` means "the original task".

| group | fields (defaults) |
|---|---|
| task | `h0_ft 4000`, `speed_kts 100`, `duration_s 90`, `steps_rel_ft [[0,0],[5,200],[50,0]]` (target steps relative to h0) |
| envelope | `min_agl_ft 500`, `min_kcas 55`, `max_abs_theta_deg 30`, `max_abs_phi_deg 45`, `nz_limits [-1, 3.8]`, `max_alt_err_ft 1000` |
| controller | `pitch_cmd_limits_deg [-8, 12]`, `alt_i_limit_deg 5`, `pitch_i_limit 0.4`, `thr_kp 0.05`, `thr_ki 0.01` |
| jets | `gear_up false` (set `gear/gear-cmd-norm = 0` before `run_ic`), `throttle_all_engines false` (false = only `fcs/throttle-cmd-norm[0]`, which is what the prototype does), `throttle_max 1.0` (e.g. 0.5 = T38 military power, no afterburner). A trim throttle above `throttle_max` counts as `trim_failed` |
| cost | `alt_err_scale_ft 100`, `itae_t0_s 10`, `itae_cap_s 30`, `w_effort 2`, `fail_base 1000` |
| GA | `gain_bounds {}`: per-gene `[min, max]` override of the log-scale ranges |
| misc | `origin_lat_deg` / `origin_lon_deg null` (JSBSim default 0/0, left untouched for bit-equivalence), `extra_props {}` (properties to create before the IC, e.g. FlightGear-only inputs some models read) |

## Parallelism model (decision and why)

**One process pool of `workers` = usable cores (8), shared by all aircraft.
Fine-grained tasks: one individual × one scenario. Aircraft run concurrently as
lightweight driver threads in the parent.** The threads only do GA bookkeeping
and sqlite. Every JSBSim step runs in the pool, so there are never more than 8
simulation processes, however many aircraft there are. The pool uses
`forkserver`, which is safe with a threaded parent, and preloads
jsbsim/numpy/sim. Workers set `PR_SET_PDEATHSIG`, so killing a batch leaves no
orphaned sims.

Why this split rather than the alternatives:

* *Across aircraft only* (one process per aircraft): 4 aircraft use 4 of 8
  cores, and 20 aircraft oversubscribe. Wall time is set by the slowest
  airframe.
* *Nested pools* (e.g. 4 aircraft × 2 workers): this oversubscribes or idles
  unless the aircraft count divides the cores. A per-aircraft pool also sits
  idle at that aircraft's generation barrier, and once that aircraft finishes.
* *Across individuals only, aircraft one after another* (`schedule:
  sequential`): no oversubscription, but every generation ends with a ragged
  tail (sims that end early on an envelope violation, slower airframes), and
  cores sit idle at each barrier. With a shared pool, the other aircraft's
  queued tasks fill that tail. The measured comparison is in the benchmark
  section.
* *Per-scenario granularity* gives 3× more, smaller tasks than
  per-individual. That helps load balance now, and matters more when one
  evaluation becomes a heavy 3D or flexible-structure run. The task function
  (`eval.task`, formerly `sim.eval_task`: dicts in, dict out) can later go to a multi-host executor
  without touching the GA.

Determinism doesn't depend on any of this. Each aircraft has its own RNG, and
results are joined by cache key, not by completion order. The tests check that
`workers=1, sequential` and `workers=8, concurrent` give identical runs.

Not done yet, and possibly needed for much heavier evals: FDM reuse between
tasks (today every sim builds a fresh `FGFDMExec` and trims, which takes a few
ms against roughly 0.1–0.2 s per 90 s sim), and early abort of hopeless
individuals.

## Cache semantics

`cache.py`: sqlite in WAL mode with one row per single-scenario simulation.

`key = sha256(canonical JSON of {aircraft, genome float64 bytes (hex), full
resolved profile (IC, envelope, cost weights, gain bounds), scenario dict (incl.
its turbulence seed), scenario_seed, jsbsim version, code_sha})`

`code_sha` is the sha256 of `sim.py` + `genome.py` + `fidelity.py` + `eval.py`,
i.e. the code that is actually executed on the evaluation path. Since the
fast-mode work the key also holds the `fidelity` and the `model_version` the
worker returned (`rigid:jsbsim1.3.1:<sha8>`, `reduced:flexv1:<sha8>`,
`full:flexv2:<sha8>`, both from FD's flexeval); the batch checks every computed result's
fidelity/model_version against the expected one before caching it. The source clone's git sha is
recorded in every run, but it isn't part of the key. The GA `seed` is
deliberately **not** in the key: it decides *which* genomes get evaluated, not
what a given genome scores, so runs with different GA seeds can share results.
The scenario seed, which does change results, is in the key.

* Identical work is never re-flown. Within a run, elites and duplicates come
  from an in-memory memo (`cache_hits_mem`). Across runs they come from sqlite
  (`cache_hits_disk`). Rerunning a finished config under a new run id gave a
  measured hit rate of 1.000 with 0 sims computed (benchmark below).
* Only the parent writes, committing once per generation. A SIGKILL loses at
  most the current generation's uncommitted results.
* Cached results equal fresh ones bit for bit (tested). Each trajectory
  re-simulation also checks that the re-flown cost equals the evaluated cost
  (`resim_mismatches` in `summary.json`, empty in every run so far).
* Caveat: editing JSBSim's aircraft XML in place without changing the package
  version would not invalidate the cache.

## Resume semantics

After each generation, each aircraft writes
`runs/<id>/checkpoints/<aircraft>.json` atomically (tmp + fsync + rename). It
holds the next population (exact float64 via JSON repr), the numpy PCG64
bit-generator state, the full per-generation history, and the best individual
of every generation. On restart (same command, or `--resume RUN_DIR`):

* the stored resolved config must match, and so must `jsbsim_version` and
  `code_sha`. Otherwise it refuses, because results would not be identical.
  The `*-b1` runs were made before the jet options were added to `sim.py`. They
  are complete, but they can no longer be resumed;
* each aircraft continues from `gen_next` with the restored RNG state, so the
  next generation is produced exactly as in an uninterrupted run;
* `history.jsonl` is rebuilt from the checkpoints. Lines written after the last
  checkpoint of a killed session are dropped, so there is exactly one line per
  (aircraft, generation). Each line carries a `session` number, and
  `sessions.jsonl` logs every start, including the generation each aircraft
  resumed at.

Result fields (populations, RNG state, fitness stats, best genomes,
trajectories) are identical to an uninterrupted run. Timing and cache-counter
fields naturally differ.

## Experiment logging: `runs/<run_id>/`

| file | contents |
|---|---|
| `config.json` | `resolved` (full resolved config incl. every aircraft's resolved profile) + `provenance` (seed, scenario_seed, `git_sha` + branch + dirty flag of the source clone, `jsbsim_version`, `code_sha`, host, cpu_count, usable cores, workers, python, numpy, platform) |
| `sessions.jsonl` | one line per start/resume (time, host, workers, schedule, resumed_at_generation, argv) |
| `history.jsonl` | one line per (aircraft, generation), fields listed below |
| `checkpoints/<aircraft>.json` | resume state (above) |
| `summary.json` | per aircraft: preflight trim (or skip reason), best fitness, gains, normalized genome, `genes_at_bound`, final per-scenario costs, altitude-hold metrics per scenario, invalid/crash rates, totals, trajectory files, `resim_mismatches`. Plus batch wall time, sims/s, cache hit rate, and box load average at start/end |
| `trajectories/` | `traj_<aircraft>_<run_id>_g<gen>.json` + `index.json` |
| `run.json` | replay manifest `ga-flightsim-run/1` (see "Replay interface"); written at run start |
| `genomes.jsonl` | one row per individual per generation (`ga-flightsim-genomes/1`), resume-safe |
| `live/<aircraft>.json` | `--viz on` only: trajectory of the current best, rewritten every generation |

`history.jsonl` fields:

* fitness: `best`, `mean`, `median`, `std`
* validity: `valid_rate`, `invalid_rate`, `crash_rate`, `status_counts`
* work: `n_individuals`, `unique_sims`, `sims_computed`, `cache_hits_mem`, `cache_hits_disk`, `cache_hit_rate`
* timing: `eval_wall_s` (generation wall time), `eval_cpu_s` (summed worker seconds)
* elite: `best_genome` (normalized), `best_gains`
* `session`

Fitness is the prototype's cost averaged over scenarios, and **lower is
better**. `invalid` means at least one scenario ended in an envelope violation
or a trim/load failure. `crash` means at least one scenario went below 500 ft
AGL.

### Model sanitising (network sockets)

Some stock JSBSim models declare socket I/O. In the benchmark set that is only
the **737**: a telnet `<input port="5137"/>` and a `QTJSBSIM` UDP input on 5139
that can *set* `fcs/elevator-cmd-norm`, throttle, and so on. The c172x's telnet
input is commented out. Every FDM load tried to bind those ports, which logged
thousands of "Could not bind" lines in a multi-process run. Whichever worker won
the bind was listening on the host, and anything that sent packets could
change a running evaluation. `sim._aircraft_root` loads such models from a
cached copy in `$TMPDIR/evolution_aircraft/<jsbsim>-<xml sha>/` with only the
top-level `<input|output ... port=...>` elements removed. An XML sanity check
asserts that nothing else was removed. Physics files are untouched, and results
are bit-identical to the stock model (`tests/test_sanitize.py`, plus re-flying
the earlier benchmark's 737 elite). This applies to the stock package only: an explicit
`aircraft_root` with socket elements is refused (see "Socket policy" below).

## Trajectory format `ga-flightsim-traj/2` (`/1` still accepted by `validate_traj`)

One file per saved best individual. By default the runner saves the best of
generation 0, of the midpoint `(G-1)//2`, and of the final generation for
every aircraft, so the viewer's compare mode has three per aircraft. Each saved
elite is **re-simulated** in record mode after evolution. The GA hot path never
records.

```jsonc
{
  "schema": "ga-flightsim-traj/2",
  "run_id": "...", "aircraft": "f16", "jsbsim_version": "1.3.1", "git_sha": "<source clone sha>",
  "seed": 1, "generation": 19, "fitness": 0.0727,          // GA cost, lower is better (see fitness_sense)
  "fitness_sense": "minimize (GA cost, mean over scenarios)", "scenario_index": 0, "scenario_cost": 0.06, "status": "ok",
  "genome": {"kp_alt": 0.26, ...},                          // decoded gains (physical units)
  "frame": {"origin_lat_deg": 0.0, "origin_lon_deg": 0.0, "origin_alt_m": 3048.0,
            "axes": "ENU metres, x=east y=north z=up", "attitude": "quat body->ENU [w,x,y,z]",
            "body_axes": "JSBSim body FRD: x forward, y right wing, z down",
            "attitude_source_of_truth": "quaternion; phi/theta/psi are HUD-only (...)"},
  "units": {"phi": "rad", "theta": "rad", "psi": "rad", "controls": "norm -1..1, throttle 0..1"},
  "dt_s": 0.0333, "sim_dt_s": 0.00833, "sample_hz": 30,
  "target": {"alt_m": 3048.0, "steps": [{"t": 0, "alt_m": ...}, ...], "speed_kcas": 350},
  "wind": {...}, "control_conventions": "...",
  "events": [{"t": 0, "type": "start", "detail": "..."}, {"t": 5, "type": "target_change", "detail": "..."},
             {"t": 31.2, "type": "gust", "detail": "..."}, {"t": 90, "type": "end", "detail": "completed"}],
             // or {"type": "terminated", "detail": "<crash|stall|attitude|overload|diverged>"}
  "channels": ["t","x","y","z","qw","qx","qy","qz","vx","vy","vz","alt_msl_m","phi","theta","psi",
               "throttle","elevator","aileron","rudder",
               "target_alt_m","kcas","nz","ub","vb","wb","lat_deg","lon_deg"],   // extras after the required 19
  "data": [[...], ...]
}
```

* **Positions** `x, y, z` are ENU metres from the origin (the trimmed start
  point), with `z = alt_msl_m - origin_alt_m`. `x`/`y` come from geodetic
  lat/lon with local WGS-84 radii. **Velocities** `vx, vy, vz` are ENU m/s
  relative to the ground. `t` is in s.
* **Attitude: the quaternion is the source of truth.** It rotates body-frame
  (FRD) vectors into ENU and is built as `q = q_ENU<-NED ⊗ q_NED<-body(ZYX
  Euler)`. Hemisphere continuity is kept between samples, so it interpolates
  cleanly. `phi, theta, psi` are **radians** and HUD-only. They are JSBSim's
  Euler angles relative to local NED, with psi in [0, 2π).
* **Controls**: `elevator`, `aileron`, `rudder` are the normalized commands
  sent to JSBSim (`fcs/*-cmd-norm`, −1..1; elevator + = trailing edge down,
  i.e. nose down). `throttle` is 0..1. Pitch trim is held by JSBSim's
  `pitch-trim-cmd`, so the trimmed elevator command is 0.
* Extra channels: `target_alt_m`, `kcas`, `nz` (g), `ub, vb, wb` (body-axis
  ground velocity, m/s, used by the validator), `lat_deg`, `lon_deg`. The viewer
  ignores unknown channels, so adding more doesn't need a version bump.
* Rows are sampled every 4th 120 Hz step (30 Hz). The last row is the end state
  (t = duration) or the termination state, so the last interval can be shorter.

**`index.json` is an object with an `entries` list** (the validator also
accepts a bare top-level list). Each entry has exactly `{generation, fitness,
aircraft, file}`, with `file` relative to `trajectories/`:

```json
{"schema": "ga-flightsim-traj-index/1", "traj_schema": "ga-flightsim-traj/2", "run_id": "...",
 "entries": [{"generation": 0, "fitness": 0.18, "aircraft": "f16", "file": "traj_f16_<run_id>_g0.json"}, ...]}
```

**Validator** (`python -m evolution.validate_traj PATH... [--min-gens-per-aircraft N]`, stdlib + numpy). It checks:

* the schema string, field types, the exact `units` and frame strings, the
  required channels, and the row widths;
* finite values on a uniform 30 Hz time base;
* control ranges, and that the Euler angles are in radians (degrees get
  rejected);
* that the quaternion is unit length and agrees with phi/theta/psi (max
  rotation difference 2e-4 rad);
* **independently of the Euler angles**: rotating the body velocity `ub,vb,wb`
  by the quaternion must reproduce the ENU velocity. This catches a wrong frame
  or quaternion convention;
* positions against the integrated velocity, and the origin;
* that every index entry has the right shape, points to a file that exists, and
  matches that file.

## Benchmark

All numbers below are measured. They come from `bench_results/b1/` and
`bench_results/j1/` (`results.json` + `table.md`), `bench_results/seeds_ic_vs_adjusted.md`,
and each run's `summary.json`. Settings: JSBSim 1.3.1, pop 32 × 20 generations,
3 scenarios (calm + 2 windy/gusty), seed 1, 8 workers, `schedule: concurrent`,
fresh cache. The **box is shared with other agents**. The 1-minute load average
at the start and end of each batch is recorded, and wall times scale with it:
cpu s/sim roughly doubled in the one run that hit load 23.

Column definitions:

* **fitness**: GA cost, lower is better.
* **hold RMS / hold max**: altitude error (ft) at least 20 s after each target
  change. "calm" is scenario 0; "mean" is the average over the 3 scenarios.
* **overshoot**: maximum over steps and scenarios.
* **settle**: time to stay within ±20 ft after a step (worst case).
* **invalid**: fraction of individuals with any envelope violation.
* **sims/s per aircraft**: measured while all aircraft share the 8 workers.
* **rerun hit rate**: same config under a new run id with the same cache.
  Every rerun computed 0 sims and gave identical results.

### Stage A: honest baseline (every aircraft on the original c172x task: 4000 ft, 100 KCAS, +200 ft)

`configs/bench_baseline.json`, run `bench_baseline-b1`. Batch wall 58.6 s, 2706 sims (46.2 sims/s), load 9.1→13.1.

| aircraft | best fitness (gen0 → final) | hold RMS ft calm / mean | hold max ft | overshoot ft | settle s | invalid gen0 / all | cpu s/sim | rerun hit rate | genes at bound |
|---|---|---|---|---|---|---|---|---|---|
| c172x | 0.3192 → **0.1923** | 3.13 / 2.95 | 7.39 | 6.9 | 7.9 | 0.250 / 0.033 | 0.200 | 1.000 | kp_alt@max |
| f16 | **not run: JSBSim trim fails at 100 KCAS / 4000 ft** (`TrimFailureError`, "wdot doesn't appear to be trimmable") | | | | | | | | |
| 737 | **not run: trim fails at 100 KCAS / 4000 ft** (same error) | | | | | | | | |
| t6texan2 | 0.2013 → **0.1690** | 0.48 / 1.16 | 4.12 | 5.4 | 9.6 | 0.000 / 0.000 | 0.139 | 1.000 | kp_alt@max |

### Stage B: per-aircraft trim IC only (gene ranges, fitness, envelope, task unchanged)

`configs/bench_ic.json`, run `bench_ic-b1`. Batch wall 116.5 s, 5574 sims (47.8 sims/s), load 13.6→14.5.
ICs are from `flight-sim-plan/flytest_results.jsonl`: f16 350 KCAS / 10 000 ft,
737 250 / 10 000, t6texan2 200 / 10 000. Gear was left down and only throttle[0]
was commanded (prototype behaviour).

| aircraft | best fitness (gen0 → final) | hold RMS ft calm / mean | hold max ft | overshoot ft | settle s | invalid gen0 / all | cpu s/sim | rerun hit rate | genes at bound |
|---|---|---|---|---|---|---|---|---|---|
| c172x | 0.3192 → **0.1923** | 3.13 / 2.95 | 7.39 | 6.9 | 7.9 | 0.250 / 0.033 | 0.191 | 1.000 | kp_alt@max |
| f16 | 0.1804 → **0.0727** | 0.07 / 0.88 | 4.96 | 4.7 | 5.8 | 0.219 / 0.034 | 0.197 | 1.000 | ki_alt@min, ki_pitch@min, kd_pitch@min |
| 737 | 0.2387 → **0.1317** | 0.79 / 1.35 | 5.33 | 9.4 | 8.4 | 0.062 / 0.014 | 0.143 | 1.000 | - |
| t6texan2 | 0.1265 → **0.0720** | 0.19 / 0.55 | 2.64 | 2.6 | 5.5 | 0.188 / 0.019 | 0.131 | 1.000 | kp_alt@max |

### Stage C: stage B + small per-aircraft changes (`configs/bench_adjusted.json`; c172x unchanged)

Run `bench_adjusted-b1`. Batch wall 142.9 s, 5604 sims (39.2 sims/s), load 14.8→14.5. Changes:

* f16: lower bounds of `ki_alt`, `ki_pitch` and `kd_pitch` widened 100×, because stage B pinned all three at the minimum;
* 737: `nz_limits [-1, 2.5]` (transport category) instead of 3.8 g;
* t6texan2: upper bound of `kp_alt` raised from 0.5 to 2.0.

| aircraft | best fitness (gen0 → final) | hold RMS ft calm / mean | hold max ft | overshoot ft | settle s | invalid gen0 / all | cpu s/sim | rerun hit rate | genes at bound |
|---|---|---|---|---|---|---|---|---|---|
| c172x | 0.3192 → **0.1923** | 3.13 / 2.95 | 7.39 | 6.9 | 7.9 | 0.250 / 0.033 | 0.228 | 1.000 | kp_alt@max |
| f16 | 0.1076 → **0.0721** | 0.05 / 0.91 | 5.16 | 4.8 | 5.7 | 0.188 / 0.025 | 0.238 | 1.000 | - |
| 737 | 0.2387 → **0.1245** | 0.73 / 1.47 | 5.87 | 4.6 | 7.5 | 0.219 / 0.030 | 0.170 | 1.000 | - |
| t6texan2 | 0.1052 → **0.0726** | 0.19 / 0.44 | 2.29 | 6.3 | 5.9 | 0.250 / 0.016 | 0.163 | 1.000 | ki_alt@min |

**Seed robustness of the stage C changes**, seeds 1–3, same scenarios within
each seed (`seed_table.py`). Best fitness change C vs B:

* f16: −0.9 %, −0.4 %, −1.3 % (mean −0.9 %);
* 737: −5.5 %, +0.2 %, −1.4 % (mean −2.2 %);
* t6texan2: +0.9 %, +1.8 %, −6.0 % (mean −1.1 %).

### Jets at Flight Dynamics' verified trim points (`configs/bench_jets.json`, run `bench_jets-j1`)

The recipe follows `flight-dynamics/INTERFACE.md` §4a: gear up before
`run_ic`, `do_simple_trim = 1` (full trim), and all engines commanded. The
measured trims match Flight Dynamics exactly:

| aircraft | trim point | trim throttle | α |
|---|---|---|---|
| T38 | 300 KCAS / 10 000 ft, throttle clamped to 0.5 (MIL, no AB) | 0.354 | 4.61° |
| 737 | 250 KCAS / 10 000 ft | 0.586 | 3.28° |
| f16 | 350 KCAS / 10 000 ft (not verified by Flight Dynamics) | 0.284 | – |

The 737 fails to trim at 185 KCAS. Envelope limits: T38 min 180 KCAS and n +7.33/−3.0;
737 min 195 KCAS (the trimmer floor is about 190) and n +2.5/−1.0; f16 min 200 KCAS and n +9/−3.

**Step scaling:** jets use a **speed-scaled altitude step: Δh = 200 ft × KCAS/100**
(T38 600 ft, 737 500 ft, f16 700 ft), with `alt_err_scale_ft = Δh/2` and
`max_alt_err_ft = 5·Δh`. Why:

* For a fixed pitch-command limit, climb rate scales with true speed. Scaling
  the step with speed keeps the required flight-path angle and capture time
  comparable to the c172x's 200 ft at 100 KCAS.
* A 200 ft step at 300+ KCAS is captured in a few seconds, so it barely
  exercises the outer loop.
* Scaling the error normalisation with the step keeps the cost dimensionless
  relative to the step, so the tracking/effort trade-off stays the same as for
  the c172x.

Gene ranges, controller, pitch-command limits, cost weights and duration (90 s)
are unchanged.

Batch wall 124.3 s, 5589 sims (45.0 sims/s), load 8.3→8.5. Cache rerun: 1.9 s, hit rate 1.000, identical.

| aircraft | profile | best fitness (gen0 → final) | hold RMS ft calm / mean | hold max ft | overshoot ft | settle s (±20 ft) | invalid gen0 / all | sims | cpu s/sim | aircraft wall s | rerun hit rate | genes at bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c172x | baseline (unchanged) | 0.3192 → **0.1923** | 3.13 / 2.95 | 7.39 | 6.9 | 7.9 | 0.250 / 0.033 | 1341 | 0.188 | 122.0 | 1.000 | kp_alt@max |
| T38 | jet_T38 (600 ft step) | 0.2036 → **0.1315** | 1.63 / 3.28 | 11.63 | 7.9 | 14.8 | 0.000 / 0.000 | 1395 | 0.162 | 120.8 | 1.000 | ki_alt@min |
| 737 | jet_737 (500 ft step) | 0.3374 → **0.1920** | 1.21 / 1.89 | 6.85 | 7.4 | 12.2 | 0.219 / 0.022 | 1425 | 0.148 | 123.1 | 1.000 | - |
| f16 | jet_f16 (700 ft step) | 0.1944 → **0.1217** | 1.61 / 3.01 | 11.61 | 8.1 | 9.6 | 0.031 / 0.002 | 1428 | 0.202 | 119.6 | 1.000 | ki_alt@min, kd_pitch@min |

`configs/bench_jets_step200.json` (run `bench_jets_step200-j1`) uses the same
recipe with the c172x's 200 ft step and 100 ft error scale, to isolate the
effect of step scaling. Best fitness: T38 0.0833, 737 0.1288, f16 0.0744,
c172x 0.1923. Hold RMS calm / mean (ft): T38 0.27 / 1.56, 737 0.78 / 1.42, f16
0.13 / 1.50. Invalid rate over all generations was ≤ 0.025, and the same genes
were pinned. Batch wall was 260.9 s, but the load rose 8.5→23.4 during that run
because of other agents, so do not compare its timing.

### Parallelism measurement (same work, cache off vs fresh cache, identical results)

| config | concurrent (shared pool) | sequential (aircraft one by one, individuals in parallel) |
|---|---|---|
| bench_baseline (2 aircraft fly) | 58.6 s | 61.5 s |
| bench_ic (4 aircraft) | 116.5 s | 128.7 s |
| bench_adjusted (4 aircraft) | 142.9 s | 202.6 s |

Concurrent was faster in all three, by 5–42 %. The load from other agents
varied between runs (load average about 9–15), so the size of the gap is noisy.

### Verdict: do jets need different gain ranges or fitness?

* **Trim / IC is the blocker, not gain ranges.** On the c172x task (100 KCAS,
  4000 ft) JSBSim cannot trim the f16 or the 737 at all. Once they start from
  their own trim point, they evolve well with the **unchanged** gene ranges and
  fitness. Final invalid rate was 0, and the calm-air hold RMS after settling
  was 0.07–0.8 ft (stage B), or 1.2–1.6 ft with the larger jet steps.
* **Gene ranges:** the 737 pins no gene. The f16 pins `ki_alt`, `ki_pitch` and
  `kd_pitch` at their *lower* bounds, i.e. it switches those terms off; its FBW
  FCS already damps and integrates pitch. The T38 pins only `ki_alt`@min.
  Widening those bounds (stage C) changed best fitness by −0.4 to −1.3 % on the
  f16 over 3 seeds: consistent but negligible. The changes for the 737 and
  t6texan2 were mixed in sign. **Recommendation: no per-jet gain ranges
  needed.** The option exists (`gain_bounds` per profile) if a future airframe
  needs it.
* **Fitness / envelope:** keep the cost function. Selection uses ranks, so the
  cost scale doesn't matter within one aircraft. But two fixed constants are
  c172x-specific and should be per-aircraft:
  * the altitude-step size and error scale. Cross-aircraft costs aren't
    comparable with a fixed 200 ft step, because fast jets capture it almost
    for free;
  * the envelope (`min_kcas`, `nz_limits`). The 737 was allowed 3.8 g, but its
    limit is 2.5 g. The jet profiles now carry these.
* **Duration:** 90 s is enough. The worst settle time was 14.8 s (T38, 600 ft
  step) within a 45 s segment.
* Engines: the prototype only commanded `throttle-cmd-norm[0]`. On twins
  (737, T38) engine 2 stayed at trim throttle, which affected stages B and C.
  The jet profiles use `throttle_all_engines: true`. The c172x is
  single-engine, so it is unaffected.

### Other measurements

* **Equivalence**: `configs/c172x_equiv.json` (pop 24 × 15, seed 1) gives best
  cost 0.19833289735948487, identical to the prototype's `evolve.py`. Batch
  wall was 57.0 s with the cache off at load 18.2; earlier, at load about 4,
  it took 24.5 s. The prototype `evolve.py` took 28.7 s at load about 4 on
  this box, including its plotting.
* **Resume at scale**: `bench_ic.json` was run as `resume-demo-b1` with the
  cache off. It was SIGKILLed as a process group with the checkpoints at
  generations c172x 9, f16 9, 737 9, t6texan2 10 of 20, then resumed with
  `--resume` (cache off, 157.3 s). The result is **identical** to the
  uninterrupted `bench_ic-b1`: populations, RNG states, per-generation stats,
  best genomes, and all 12 trajectory files. `history.jsonl` has 80 lines,
  80 unique (aircraft, generation), sessions {0, 1}.
* **Throughput**: 45–48 sims/s per batch on 8 workers at load 8–15 (one sim =
  one aircraft × one 90 s scenario at 120 Hz). Worker cpu s/sim ranged
  0.13–0.24; per aircraft: t6texan2 < 737 < T38 < c172x ≈ f16.
* Trajectory re-simulation matched the evaluated cost bit for bit in every run
  (`resim_mismatches: []`). `validate_traj --min-gens-per-aircraft 3` passed
  for every benchmark run.

## Phase-1 shared profiles (`configs/phase1.json`, runs `phase1-s{1,2,3}`)

Genome Architect's shared set (`genome/aircraft_profiles/phase1_shared.json`,
`genome/exports/evolution_phase1_profiles.json`, rationale in `genome/ALIGNMENT.md`)
adopted as a separate profile set. The `baseline` profile, the earlier configs and
the prototype-equivalence test are unchanged: every new `Profile` field defaults to
the prototype behaviour.

| ALIGNMENT item | here |
|---|---|
| A. profiles (trim points, 200 ft step, scale 100 / max err 1000, envelopes, per-aircraft clamp, thr gains, gear up, all engines, explicit gain bounds) | `configs/phase1.json`: the export's blocks verbatim, except thr_kp/thr_ki at full precision (see below) |
| B. 600 fpm ramp, 0.1 g corners, scored against the reference, ITAE clock at the command | `ramp_fpm`, `ramp_accel_g` → `Scenario.target/target_rate/ramp_plan` (same closed-form constant-acceleration phases as `sim_ext._smooth_plan`, re-implemented, not imported) |
| C. climb-rate feed-forward | `alt_ref_ff`: outer D term on `h_dot - h_ref_dot` |
| D. comfort 0.05 | `w_comfort`, `comfort_params`, `comfort_weights`; `sim.comfort_terms` = `genome/fitness.py` formulas on per-step nz/θ/q |
| E. log0 for ki_alt / ki_pitch | `gene_kinds` + `genome.Gene(kind="log0")`, zero band 0.05; a zeroed gene is reported as `@zero` |
| F. wing-leveler gains | `roll_kp`/`roll_kd` (values = genome adapter `sim_fixed`: c172x 0.05/0.02, T38 0.11802/0.04721, 737 0.61616/0.24646, f16 0.03450/0.01380) |
| G. FD `jsbsim_root` | `aircraft_root`; models load with `FGFDMExec(root)`; socket stripping applies to that root (copy in /tmp keyed by a hash of FD's files, FD's folder is never written) |
| 17. GA budget | pop 32 × 20 gens, 3 scenarios, scenario_seed 1 fixed (= genome's draws); `--seed` varies the GA seed only |

**FD `jsbsim_root` contents (updated 2026-10-06 05:25 PT):** c172x, T38, 737 and,
since ~04:57 PT, **f16** are all present (with FD's flex point masses; each flies
bit-identically to stock, tested). All four Phase-1 profiles now set `aircraft_root`
to FD's root. Asking for a model a root doesn't have is a clean `load_failed`, never a
silent fallback. **Sockets:** FD re-prepared its models at 04:48 PT (`flexwing_meta.json`
fmt 3). The 737 now has its 3 network I/O elements removed at source
(`network_io_removed: 3`), so no model in FD's root declares socket I/O and
`sim._aircraft_root` returns FD's folder unchanged for all four. The /tmp strip is
kept as a guard for any root that still has sockets. The test checks "the file we
load has no `port=` element" for each model instead of hard-coding which ones need
stripping.

**f16 is not in the shared set** (it loads from FD's root since the 05:00 PT rerun,
see "f16 on Flight Dynamics' root" below). `phase1_f16` = the Phase-1 task/fitness rules +
the trim point and envelope bench_jets-j1 used (350 KCAS / 10000 ft, trim throttle
0.284 gear up, min 200 KCAS, n +9/−3) + genome's adapter values for f16 (clamp
[−10, 15], control-power-scaled speed PI and roll gains, gene ranges unioned with the
prototype ranges as ALIGNMENT #11 does for the other jets). Genome's own derived f16
profile uses 20000 ft and is flagged "backend not checked"; we kept the 10 kft point
that trims and that j1 used.

**Cache key / resume.** New Profile fields are in the resolved profile dict, so they
are in the key automatically; the key additionally carries `model_files`, a sha256 of
every file in the `aircraft/<model>/` directory actually loaded, so FD editing
`jsbsim_root` in place is a miss, not a stale hit. sim.py/genome.py changed, so runs
made before this change refuse `--resume` (`code_sha changed`), by design.

**Hold metrics with a ramp.** Segments start at command changes
(`target_cmd_alt_m`); the hold window is ≥ 20 s after the command *and* reference at
rest (`target_rate_mps` = 0), i.e. 28.1–50 s and 73.1–90 s here (25–50 / 70–90 s for an
instant step). Settle/overshoot are measured against the command, so settle time
includes the ~23 s ramp. Trajectories gain `target_cmd_alt_m` and `target_rate_mps`
channels and `target.reference`; the validator checks that the reference slope matches
`target_rate_mps` and that it sits on the command whenever it is at rest.

### f16 on Flight Dynamics' root (runs `phase1-f16fd-s{1,2,3}`, 2026-10-06 05:00–05:04 PT)

`phase1_f16` now has `aircraft_root` = FD's `jsbsim_root` and `throttle_max` 0.5;
everything else is unchanged. FD's prepared f16 (INTERFACE.md §5) adds a 0 lb
placeholder at point-mass index 0, so the pilot is now index 1. Nothing in
`evolution/` indexes point masses (`rg -i "pointmass|pilot"` finds nothing), so the
index shift doesn't matter.

**Trim vs FD** (350 KCAS, 10000 ft, gear up; `test_fd_f16_loads_trims_like_fd_and_has_no_sockets`):

| quantity | ours | FD (INTERFACE §5) |
|---|---|---|
| α = θ | 1.04546° | 1.045° |
| throttle-cmd / throttle-pos | 0.28358 / 0.5672 | 0.2836 / 0.567 |
| pitch-trim-cmd-norm | −0.06008 | −0.0601 |
| stabilator `fcs/elevator-pos-deg` | −1.111° | −1.111° |
| Mach / qbar | 0.629 / 403.11 psf | 0.629 / 403 psf |
| weight, point mass [0] / [1] | 20630 lb, 0 / 230 lb | 20630 lb, placeholder 0 / Pilot 230 lb |
| gear pos after trim | 0 | 0 |
| 20 s hands-off hold | Δθ +0.0277°, Δα −0.0016°, −0.104 kt, +3.94 ft | +0.028°, −0.002°, −0.10 kt, +3.9 ft |

The FBW stick is a g/pitch-rate demand, so trim lands in `pitch-trim-cmd-norm` with
elevator-cmd 0. The stock f16 trims to exactly the same state (its index 0 is the
230 lb pilot). With phase1-s1's best f16 gains, FD's copy and stock give bit-identical
costs on all 3 scenarios, and the 0.5 clamp is not hit (max throttle 0.30 / 0.45 / 0.30).

**Sockets:** none. FD's f16.xml has no `port=` / network elements (only FCS
`<input>`/`<output>` and a commented-out CSV output); meta `network_io_removed: 0`.
`_aircraft_root("f16", FD)` returns None, so no stripped copy is made.

**Runs** (`python -m evolution.batch --config configs/phase1.json --aircraft f16 --seed N --run-id phase1-f16fd-sN`,
f16-only batches, 8 workers, pop 32 × 20, 3 scenarios, scenario_seed 1):

| seed | best | hold RMS mean (calm) ft | invalid all gens / gen 0 / final | genes at bound | wall s | load 1-min start → end |
|---|---|---|---|---|---|---|
| 1 | 0.09009480210402053 | 1.085 (0.81) | 0.0094 / 0.125 / 0 | kd_pitch@min | 79.5 | 8.0 → 13.9 |
| 2 | 0.08876249360294831 | 0.566 (0.12) | 0.0016 / 0.031 / 0 | ki_alt@zero | 83.4 | 13.6 → 14.2 |
| 3 | 0.08942040323392227 | 0.553 (0.11) | 0 / 0 / 0 | ki_alt@zero | 79.5 | 14.2 → 19.5 |

Mean 0.08943, std (ddof=1) 0.00067, range 0.08876–0.09009. These are identical to the
stock `phase1-s*` f16 (0.0901 / 0.0888 / 0.0894) to full precision: same best genomes
and same final populations. Only the population *mean* differs in 5 of 60 generation
records (individuals that hit the new 0.5 throttle clamp). Gen 0/9/19 trajectories are
in `runs/phase1-f16fd-sN/trajectories/` and all validate.

**Bounds verdict: unchanged.** Evidence: `analysis/f16_bounds_probe.py` →
`analysis/f16_bounds_probe.json` (1-D sweeps through each seed's best from min/30 to
max×10, plus 256 uniform-random genomes) and `analysis/f16_fine_sweep.json`.
* kp_alt sits at 0.94–0.97 of its normalized (log) range, but its optimum is 0.3–0.4,
  inside the bound (s1: 0.4 → 0.0901, 0.5 → 0.0904, 1.0 → 0.1025).
* kd_pitch@min on s1: going below 1e-4 changes the cost by 2e-5 (the FBW already
  damps q).
* Random genomes are 92.6 % valid. No quintile of any gene drops below 77 % valid, so
  most of the range is not unflyable.
* ki_alt has a second basin above its 0.05 bound (s2: 0.09 → 0.0853, −3.9 %; s3
  −2.8 %; s1 worse). Opt-in profile `phase1_f16_kialt03` (ki_alt max 0.3; use
  `--profile-for f16=phase1_f16_kialt03`) was run on the same 3 seeds
  (`phase1-f16fd-kialt03-sN`): 0.09018 / 0.08881 / 0.08930, mean 0.08943 (identical
  mean, per-seed +0.10 / +0.06 / −0.14 %). The GA never reached the upper basin, so
  it is **not adopted** and is not the default. The basin is a search-budget question,
  not a bounds error.

New batch flags: `--aircraft a,b` (run a subset of a config's aircraft) and
`--profile-for AIRCRAFT=PROFILE` (swap one aircraft's profile). Both are validated and
rejected on `--resume`.

### Cross-team agreement

* genome/'s best Phase-1 genomes (`genome/runs/phase1_{default_c172x,t38,b737}_v4`)
  re-flown here give their per-scenario cost, track, effort and comfort **bit for bit**
  (`test_reproduces_genome_phase1_costs_bitwise`). This needs thr_kp/thr_ki at full
  precision: the export rounds them to 6 digits (T38 0.036674 vs 0.0366743705…), which
  moves T38/737 costs by 1e-8–5e-7 relative. `phase1.json` uses the adapter's values;
  genome/ could export full precision.
* Whole GA runs agree too: our `phase1-s1` T38 best is 0.0917745811719954 and 737
  0.11024576983937538, identical to genome's `phase1_t38_v4` / `phase1_b737_v4`
  (same GA operators, seed 1, budget). Their c172x run used 48 × 40, so it differs.
* bench_jets-j1's best genomes re-flown under Phase-1: T38 0.1636, 737 0.1237, the
  same numbers ALIGNMENT reports from genome's simulator.

### Results (measured 2026-10-06 04:29–04:37 PT, 8 workers, concurrent schedule)

#### Per aircraft, per seed (GA seed varies; scenario set fixed, scenario_seed 1)

| aircraft | seed | best cost | gen-0 best | track / effort / comfort (mean of 3 scen.) | hold RMS mean (calm) ft | overshoot max ft | invalid rate all gens / final | genes at bound | evolve wall s |
|---|---|---|---|---|---|---|---|---|---|
| c172x | 1 | 0.2245 | 0.3316 | 0.0724 / 0.0297 / 1.853 | 4.90 (5.95) | 11.9 | 0.023 / 0.000 | - | 146.0 |
| c172x | 2 | 0.2083 | 0.3603 | 0.0674 / 0.0268 / 1.745 | 3.39 (4.54) | 6.4 | 0.016 / 0.000 | - | 150.0 |
| c172x | 3 | 0.2075 | 0.2666 | 0.0700 / 0.0269 / 1.675 | 2.24 (1.74) | 7.0 | 0.013 / 0.000 | - | 140.6 |
| T38 | 1 | 0.0918 | 0.1125 | 0.0282 / 0.0052 / 1.063 | 1.04 (0.45) | 4.7 | 0.005 / 0.000 | ki_alt@zero | 142.4 |
| T38 | 2 | 0.0938 | 0.1244 | 0.0264 / 0.0069 / 1.070 | 0.89 (0.33) | 4.3 | 0.002 / 0.000 | ki_alt@zero, ki_pitch@zero | 148.0 |
| T38 | 3 | 0.0928 | 0.0981 | 0.0254 / 0.0066 / 1.084 | 0.95 (0.37) | 4.6 | 0.003 / 0.000 | ki_alt@zero | 143.5 |
| 737 | 1 | 0.1102 | 0.1398 | 0.0391 / 0.0104 / 1.006 | 1.52 (1.04) | 4.8 | 0.031 / 0.000 | ki_alt@zero | 144.6 |
| 737 | 2 | 0.1124 | 0.1300 | 0.0373 / 0.0112 / 1.055 | 1.96 (1.73) | 6.5 | 0.017 / 0.000 | - | 144.4 |
| 737 | 3 | 0.1099 | 0.1214 | 0.0355 / 0.0118 / 1.018 | 1.32 (0.70) | 5.2 | 0.016 / 0.000 | ki_alt@zero | 141.9 |
| f16 | 1 | 0.0901 | 0.0997 | 0.0187 / 0.0071 / 1.145 | 1.08 (0.81) | 3.8 | 0.009 / 0.000 | kd_pitch@min | 143.4 |
| f16 | 2 | 0.0888 | 0.1005 | 0.0174 / 0.0107 / 0.999 | 0.57 (0.12) | 3.1 | 0.002 / 0.000 | ki_alt@zero | 146.3 |
| f16 | 3 | 0.0894 | 0.0929 | 0.0167 / 0.0110 / 1.016 | 0.55 (0.11) | 3.1 | 0.000 / 0.000 | ki_alt@zero | 139.0 |

#### Across seeds

| aircraft | best cost per seed | mean | std (ddof=1) | min-max | hold RMS mean across seeds ft | invalid rate (all gens, mean over seeds) | bench_jets-j1 best (different task) |
|---|---|---|---|---|---|---|---|
| c172x | 0.2245 / 0.2083 / 0.2075 | 0.2134 | 0.0096 | 0.2075-0.2245 | 3.51 | 0.017 | 0.1923 |
| T38 | 0.0918 / 0.0938 / 0.0928 | 0.0928 | 0.0010 | 0.0918-0.0938 | 0.96 | 0.003 | 0.1315 |
| 737 | 0.1102 / 0.1124 / 0.1099 | 0.1109 | 0.0014 | 0.1099-0.1124 | 1.60 | 0.021 | 0.1920 |
| f16 | 0.0901 / 0.0888 / 0.0894 | 0.0894 | 0.0007 | 0.0888-0.0901 | 0.73 | 0.004 | 0.1217 |

#### Wall time and box load

| run | seed | wall s | loadavg 1/5/15 start | end | sims computed | cache hits | sims/s |
|---|---|---|---|---|---|---|---|
| phase1-s1 | 1 | 147.4 | 1.7/1.8/6.4 | 7.7/4.3/6.7 | 5742 | 903 | 39.0 |
| phase1-s2 | 2 | 151.4 | 7.7/4.3/6.7 | 7.7/5.9/6.9 | 5721 | 945 | 37.8 |
| phase1-s3 | 3 | 145.0 | 7.7/5.9/6.9 | 7.8/6.8/7.1 | 5424 | 945 | 37.4 |
| total | | 443.7 | | | | | |

#### bench_jets-j1 best genomes re-flown under the Phase-1 task (3 scenarios, scenario_seed 1)

| aircraft | j1 best cost (its own task) | j1 genome under Phase-1 | Phase-1 GA best, seed 1 / mean of seeds |
|---|---|---|---|
| c172x | 0.1923 | 0.2352 (ok) | 0.2245 / 0.2134 |
| T38 | 0.1315 | 0.1636 (ok) | 0.0918 / 0.0928 |
| 737 | 0.1920 | 0.1237 (ok) | 0.1102 / 0.1109 |
| f16 | 0.1217 | 0.1482 (ok) | 0.0901 / 0.0894 |

Notes:
* Wall time is per batch (4 aircraft concurrently on one 8-worker pool), 145–151 s per
  seed, 443.7 s total, box load 1.7 → 7.8 (1-min) — mostly our own 8 workers. No
  disk-cache hits across seeds (0 for every aircraft), so each run computed all of its
  ~5.4–5.7 k sims; the in-memory hits are elites/duplicates within a run.
* bench_jets-j1 numbers are a different task (speed-scaled 600/500/700 ft instant step,
  error scale = step/2, no comfort, [−8, 12]° clamp, C172 helper gains), so the
  columns are not comparable as numbers; the re-fly table is the like-for-like view.
* The integrator is switched off by the GA (ki_alt `@zero`) on most jet runs; on the
  c172x the integral gains stay inside the range. Final-generation invalid rate is 0
  everywhere; over all generations ≤ 3.1 %.

## Heading hold (`configs/phase1_hdg.json`, runs `phase1hdg-s{1,2,3}`)

This implements `genome/HANDOFF_heading_hold.md` exactly. Without it the c172x drifts
about 25° in heading over 90 s (the prototype's roll loop only levels the wings).

* **Profile fields** (`sim.Profile`): `heading_hold` (default `false`),
  `bank_limit_deg`, `hdg_i_limit_deg`, `w_heading`, `hdg_rms_ref_deg`, all
  validated in `from_dict`.
* **Controller** (inside the 120 Hz loop, target heading = initial heading 0°):
  * `e_ψ = wrap180(0 − ψ)`.
  * The integrator is clamped at `hdg_i_limit_deg / ki_hdg`.
  * `φ_cmd = clamp(kp_hdg·e_ψ + ki_hdg·∫e_ψ, ±bank_limit_deg)`.
  * The leveller becomes `−roll_kp·(φ − φ_cmd) − roll_kd·p`.
  * With the flag off the old branch runs unchanged, so it is bit-identical
    (`test_heading.py`).
* **Cost**: `cost += w_heading · RMS(e_ψ)/hdg_rms_ref_deg`, added after comfort.
  It uses the handoff's numpy expression over the per-step ψ samples.
* **Per-scenario output**: `heading_rms`, `hdg_drift_deg` (wrap180(ψ_end − ψ_0))
  and `hdg_max_abs_err_deg`.
* **Genes**: `make_schema(bounds, kinds, heading_hold=True)` appends `kp_hdg`
  (log) and `ki_hdg` (log0) after the 6 legacy genes. The batch passes the flag,
  and gen 0 is sized from the schema (8 genes).
* **Configs**: ON for c172x/T38/737, OFF for f16, opt-in via `phase1_hdg.json`
  (it equals genome/'s export exactly). `phase1.json`, the baseline equivalence
  test and the `phase1-s*` runs are untouched.
* **Cache**: the heading fields are in the resolved profile, and sim.py/genome.py
  are in `code_sha`, so the cache key changes. The test covers 6 variants.

**Bit-identical to genome/**: the handoff's three best genomes re-fly to genome's
numbers bit for bit, both the cost and every per-scenario cost:

| aircraft | cost |
|---|---|
| c172x | 0.19600587132367517 |
| T38 | 0.0966245736792507 |
| 737 | 0.10678211633774594 |

The normalized genomes decode to the same gains.

**Results** (pop 32 × 20, scenario_seed 1, 8 workers, 2026-10-06 05:34–05:48 PT):

| aircraft | s1 | s2 | s3 | mean | std (pop.) | phase1 (no heading) mean | max final drift ° (per seed) | max \|Δψ\| ° |
|---|---|---|---|---|---|---|---|---|
| c172x | 0.197486 | 0.195373 | 0.200876 | 0.19791 | 0.00227 | 0.21341 | 0.78 / 0.75 / 2.22 | 9.38 |
| T38 | 0.096625 | 0.093007 | 0.092451 | 0.09403 | 0.00185 | 0.09277 | 0.04 / 0.09 / 0.00 | 2.78 |
| 737 | 0.106782 | 0.110647 | 0.108885 | 0.10877 | 0.00158 | 0.11086 | 0.78 / 0.04 / 0.07 | 2.76 |
| f16 (off) | 0.090095 | 0.088762 | 0.089420 | 0.08943 | 0.00054 | 0.08943 (identical) | – | – |

Notes on the results:

* **c172x drift.** Over the calm scenario, drift went from 25.4 / 25.5 / 25.3°
  (phase1-s*) to 0.25 / 0.53 / 1.85°. Over all scenarios the worst final drift is
  ≤ 0.8° on s1/s2 and 2.2° on s3, whose kp_hdg = 0.18 is low.
* **Against genome/.** T38 matches their numbers per seed (0.0966 / 0.0930 /
  0.0925), and 737 s1 matches their 0.1068. Their c172x numbers (mean 0.1935,
  drift ≤ 0.8°) come from pop 48 × 40 with scenario seed = GA seed, so they are a
  different budget and a different scenario set. Only s1 shares our scenario set:
  theirs is 0.1960 at 48 × 40, ours 0.1975 at 32 × 20.
* **Run checks.** Wall time per batch: 208.9 / 322.2 / 320.0 s. Box load 0.3 → 19
  (other agents). 0 re-sim mismatches. Gen 0/9/19 trajectories validate in all
  three runs (36 files).
* **Sockets.** All four models load from FD's root with 0 socket elements, so
  the 737 strip never runs (see "Socket policy").

### Socket policy

`sim.socket_policy(root)` decides what happens to `<input|output port=…>`:

* **Stock package data**: `strip`. The model is loaded from a sanitized copy,
  and a loud stderr WARNING is printed once per process.
* **Explicit `aircraft_root`** (e.g. FD's root): `refuse`, raising
  `SimSetupError("load_failed")`. Another team's tree must be fixed at the
  source, not silently patched.
* **Override**: `EVOLUTION_SOCKET_POLICY=strip|refuse`.

`preflight` reports `socket_io_elements` / `socket_policy`, and the batch logs a
warning when any are found. Since FD stripped the 737 at the source (04:48 PT),
every FD-root model has 0 socket elements, so the guard is a no-op that would
refuse loudly.

## Fast mode: `--viz`, `--fidelity`, multi-fidelity

### `--viz {on,off}` (config key `viz`, default off, execution-only)

* **off**: no recorder in the hot path. The best of generations 0 / mid / final
  are re-flown with the recorder after evolution (`_export`), and every re-fly
  checks that its cost equals the logged cost (`resim_mismatches`).
* **on**: every evaluation records a trajectory (the old slow path).
  `runs/<id>/live/<aircraft>.json` holds the current best of each generation.
* Results are bit-identical on/off: same genomes.jsonl costs, history and
  checkpoints (`test_viz_on_off_bit_identical`).

### `--fidelity {rigid,reduced,full,full_a1}` (config key `fidelity`, part of the run identity)

The single adapter is `evolution/fidelity.py`. Since 06:41 PT it wires `reduced` and `full` to **Flight Dynamics'
`flexeval.evaluate`** (`flight-dynamics/flexeval.py`, spec `flight-dynamics/INTERFACE_v2.md`). FD's code is
**imported, never copied**: `flexeval`, `flexbody`, `flexwing` and `coupled_sim` are loaded from FD's folder with
bytecode writing off. The earlier `full(v1)` stand-in (our own n_bend = 2 coupler) is gone. Its numbers stay below only
as a labelled historical benchmark; the code is in `/workspace/backups/evolution-pre-fdv2-0648.tgz`.

| fidelity | what (FD flexeval) | model_version (from FD's result) | margin gate |
|---|---|---|---|
| `rigid` | JSBSim only; our `sim.simulate` unchanged, bit-identical to every rigid run so far | `rigid:jsbsim1.3.1:<sha8>` (FD computes the same string) | – |
| `reduced` | v1 FlexWing (1 bending + 1 torsion mode) on the v2 genome projected by FD's `project_to_reduced` | `reduced:flexv1:<sha8>` (FD's sha includes the gate) | **per aircraft**: c172x 0.9; 737 / T38 / f16 1.0 |
| `full` | flex v2 (`flexbody.py`: 25-DOF wings + empennage + fuselage), flown on `jsbsim_root_v2` | `full:flexv2:<sha8>` | 1.0 |
| `full_a1` | P3-A1, opt-in (INTERFACE_v2 §13): FD `flexeval_a1.evaluate` / `flexbody_a1.FlexBodyModelA1` — 64 strips, 4b+3t+2ip per semi-wing, tails/fuselage as full (31 DOF); `J_wing_tip_bm_limit` station-exact at η 0.875; flown on `jsbsim_root_v2`; FE wings have 65 nodes in flex_state / trajectories | `full_a1:flexv2a1:<sha8>` (pins: `model_versions_post_p3a1.json`) | 1.0 |

* **full vs full_a1.** Costs are not interchangeable (denser model, station-exact tip term). The cache key carries
  fidelity + model_version, so the two never share an entry, and `cache.eval_key` refuses a model_version whose prefix is
  not `<fidelity>:`. `full_a1` may be a ladder top rung (`rigid->full_a1`); it cannot screen for `full`. Configs:
  `analysis/make_phase2_configs.py --p3a1` → `configs/phase3a1_smoke.json` / `phase3a1_pilot.json`.

* **FD's folder stays read-only.** FD's `evaluate` calls `ensure_root` / `ensure_root_v2`, which write only when a
  prepared root is stale. We run the same staleness tests read-only first (`fidelity.check_prepared`) and refuse if a
  root is stale. In our processes `flexwing.prepare_aircraft` and `flexbody.prepare_aircraft_v2` are replaced by
  functions that raise; FD's files themselves are untouched.
* **Reduced gate per aircraft.** Config `fidelity_per_aircraft: {name: {reduced_gate, min_full_frac}}` overrides
  `fidelity.DEFAULT_PER_AIRCRAFT` (c172x 0.9 / 0.0; 737, T38, f16 1.0 / 0.25; unknown aircraft 1.0 / 0.25). The swept
  wings get 1.0 because FD measured the reduced flutter margin about +0.09 optimistic on the 737. The adapter sets FD's
  `MARGIN_GATE["reduced"]` for the duration of each call and restores it afterwards. FD's `model_version` hashes the
  gate, so the cache and run identity follow it. `run.json` records `aircraft[].reduced_gate`, and replays use it.
* **Unit of work.** Rigid stays per scenario. Reduced and full are one `flexeval.evaluate` call **per genome over all
  scenarios**, so margins and the model build happen once per genome. Their cache key is genome × all scenarios × gate ×
  model_version. A one-scenario call (`fidelity.evaluate_scenario`, used by the export and live viz) equals that
  scenario's entry of the all-scenario call bit for bit (tested).
* **Result** (`fidelity.evaluate_genome`): `cost` (= mean of FD's per-scenario costs), `status`, FD's 23 `terms`
  (`track, effort, comfort, heading, hold, J_flutter_margin, J_div_margin, J_mass, J_bm_rms, J_bm_peak, J_tip, J_twist,
  J_reversal_margin, J_tail_bm_peak, J_fus_bm_peak, J_smooth`, plus FD's §12 sizing terms `J_wing_bm_limit,
  J_wing_torque_limit, J_wing_ip_limit, J_tail_bm_limit, J_fus_bm_limit` and full-only `J_wing_torque_peak, J_wing_ip_peak`,
  added by FD at ~07:56 PT; inapplicable = 0.0, not in `terms_available`),
  `margins` (flutter, divergence, reversal), `margins_fidelity`, `margin_gate`, `reduced_gate`, `mass_total_frac`,
  `projection` (reduced) and `per_scenario` (cost, sim_cost, status, struct terms, tip / twist maxima). A margin-gate
  fail is not flown: cost = fail_cost = 2·fail_base, with one aligned `not_flown` entry per scenario.
* **Feasibility.** `feasible` = status ok at the fidelity scored (`feasibility_fidelity`). In multi-fidelity runs
  feasibility is trusted only from full: screen-only rows get `feasible: null`.
* **`--struct-genes`** appends FD's 12 v2 genes (`flexbody.gene_schema(False)`: wing_ei_root, wing_ei_taper_1..4,
  wing_gj_ratio_root/tip, wing_nsm_root/tip, tail_stiffness_scale, fuselage_stiffness_scale, struct_damping_ratio).
  Decoded values are clamped to FD's [lo, hi], because a log decode at u = 1 can overshoot by 1 ulp and FD rejects it.
* **Agreed hook (pass-through):** `fidelity.evaluate(gains, struct_genome, scenarios, model, *, fidelity, root, dt,
  record)`.
* **Checks** (`tests/test_eval.py`):
  * our reduced/full results equal a direct `flexeval.evaluate` call bit for bit;
  * model_version formats, and the gate in the reduced sha;
  * a flutter margin of 0.95 passes reduced at gate 0.9 and fails at 1.0 (cost 2·fail_base, not flown);
  * a recorder changes nothing.

### Multi-fidelity (`--multi-fidelity --screen S[,S2] --top-k K`, config `multi_fidelity`)

`screen` is one fidelity or an ascending list. The ladder is screen(s) + `fidelity`, e.g. rigid→full, reduced→full,
rigid→reduced→full. Each generation:

1. Stage 0 scores everyone.
2. Each later stage scores the best of the previous stage (by that stage's cost) **plus every carried elite**:
   * the authoritative (full) stage scores k_full = max(top_k, ⌈min_full_frac·pop⌉) individuals. min_full_frac is per
     aircraft: **0.25 on swept wings**, so ≥ 8 of 32 plus elites; c172x uses top_k (4);
   * a middle stage scores mid_k (default 2·k_full).
3. Ranking: full-scored first (by full cost), then the rest by the highest fidelity reached and its cost. Elites and the
   best are therefore always full verdicts. Feasibility comes only from full.

Logging:
* **genomes.jsonl:** `rescored_at_full`, `screen_cost` / `screen_fidelity` (first stage), `ladder_cost`,
  `ladder_status` and `ladder_model_version` ({fidelity: …} for every stage the genome reached). A row's `cost`,
  `fidelity` and `model_version` are those of the **highest stage it reached** (full when `rescored_at_full`, else the
  middle or screen stage); each fidelity has its own model_version.
* **history.jsonl:**
  * `ladder`, `k_full`, `k_mid`, `n_rescored`, `rescored_idx`;
  * `spearman {"rigid_vs_full", "reduced_vs_full", "rigid_vs_reduced"}` over the re-scored set, plus
    `spearman_both_ok` (pairs where both statuses are ok, matching FD's "both ok" column);
  * `ladder_pairs`, `stages{fid: sims, cpu, wall, n}`;
  * `n_scored_authoritative`, `feasible_rate_authoritative`.

The rule is deterministic and resumable (kill-between-rows-and-checkpoint test).

### Benchmark: FD v2 ladders (c172x + 737)

> **FD changed v2 after this benchmark.** INTERFACE_v2 §12 (limit-load sizing, minimum gauge, margin shaping;
> flexeval/flexbody edited 07:45–07:56 PT) changes reduced/full costs and model_versions. Every number below is
> pre-§12 (runs 06:59–07:40 PT). FD notes that runtimes are unaffected and Spearman values shift slightly.

`analysis/bench_fdv2.py` (pop 32 × 4 gens, c172x + 737, struct genes, cache off, 8 workers, seeds 1–3; seed 1 06:59–07:15 PT,
seeds 2–3 07:18–07:40 PT, box load 1-min 6.5–23). Raw: `bench_results/fdv2/results{,-s2,-s3}.json`; counterfactual +
tables: `report{,-s2}.md` (`analysis/bench_fdv2_report.py`); across seeds: `seeds.md` (`analysis/bench_fdv2_seeds.py`).
Full write-up: `analysis/STATUS_C.md`. The seed-3 counterfactual was not run: Flight Dynamics edited `flexeval.py` /
`flexbody.py` at 07:45–07:46 PT and our model_version guard stopped it (`report-s3.log`).

| case | c172x best full cost s1/s2/s3 (mean, Δ vs full-only) | feasible (full-scored) | task CPU s | 737 best s1/s2/s3 (mean, Δ) | feasible | task CPU s |
|---|---|---|---|---|---|---|
| full-only (viz off) | 0.2568 / 0.2889 / 0.3926 (0.3128) | 0.78/0.73/0.70 | 577/545/556 | 0.2048 / 0.1621 / 0.1892 (0.1854) | 0.77/0.80/0.74 | 535/557/501 |
| rigid→full | 0.2817 / 0.3529 / 0.3866 (0.3404, +0.028) | 0.90/0.78/0.95 | 145/133/155 | 0.1914 / 0.1990 / 0.1967 (0.1957, +0.010) | 0.89/0.97/0.92 | 209/213/220 |
| reduced→full | 0.3055 / 0.2994 / 0.3416 (0.3155, +0.003) | 1/1/1 | 525/522/513 | 0.2216 / 0.1803 / 0.1978 (0.1999, +0.015) | 0.94/0.97/0.97 | 556/561/576 |
| rigid→reduced→full | 0.2826 / 0.2793 / 0.3402 (0.3007, −0.012) | 1/1/1 | 245/275/276 | 0.2676 / 0.2001 / 0.2231 (0.2303, +0.045) | 0.94/1/0.91 | 427/436/425 |

* **viz on vs off** (seed 1, full-only): task CPU 1193 → 1113 s (−6.7 %), gen wall c172x 61.8 → 79.4 s / 737 60.2 → 67.0 s
  (load rose 11.8 → 18.7 during the pair, so wall is load-dominated); identical costs.
* **Speedup vs full-only viz on** (seed 1, wall / CPU): full viz off 0.93× / 1.07×, rigid→full 4.74× / 3.37×, reduced→full
  1.99× / 1.10×, rigid→reduced→full 2.84× / 1.78×.
* **CPU per 90 s scenario**: rigid 0.17–0.23 s, reduced 1.04–1.18 s, full 0.89–1.65 s (FD: 0.2 / 1.2 / 1.7).
* **Spearman** over the whole 32-genome populations (counterfactual, seeds 1–2, 8 gens): reduced vs full c172x median +0.94
  (both ok +0.96; FD 0.94 / 0.91–0.92), 737 +0.69 (both ok +0.87; FD 0.84–0.87 / 0.59–0.62). Logged per generation on the
  re-scored top set: rigid vs full ≈ 0 and unstable (−0.60…+1.00), as FD found.
* **Elite sets**: the final elites of every ladder run differ from full-only (overlap 0; the GA trajectories diverge). On
  identical populations the ladder's rule picks a different elite set in 0–2 of 8 generations.
* **Recommendation**: c172x rigid→reduced→full (rigid→full for quick exploration); swept wings (737; T38/f16 by extension)
  full-only for authoritative runs and rigid→full (≥ 25 % re-scored) as the fast mode. Reduced costs about as much as full
  and ranks the 737 poorly, so it does not belong in swept-wing ladders.

### HISTORICAL benchmark against the full(v1) stand-in (c172x + T38, pop 32 × 3 gens, seed 1, cache off, 8 workers)

Kept for the record. It measured our own v1 n_bend = 2 stand-in, which FD's v2 has replaced. Code: `/workspace/backups/evolution-pre-fdv2-0648.tgz`; scripts `analysis/bench_fastmode*.py` / `microbench_fidelity.py` (marked historical).

Measured 2026-10-06 06:24–06:36 PT. Box load (1-min) stayed at 7.6–9.2 throughout. Other teams' jobs were
still on the box, so absolute times are about 2× an idle box, but every case saw the same load.

* **Setup.** `analysis/bench_fastmode.py` runs the cases one after another. Each run is `bench_results/fastmode/runs/bench-*`
  and has its own `run.json` + `genomes.jsonl`. Raw results are in `results.json`; `analysis/bench_fastmode_report.py` writes
  `report.md` / `report.json`.
* **Genes.** The flex cases use struct genes (12 genes). Rigid uses 8, so rigid gen 0 is a different population.
* **Loaded-box run.** An earlier full run at load 9–38 (05:57–06:22 PT, `results_loaded_0557-0622.json`) is kept for
  reference only. Its times are not comparable across cases.

| case | wall s | speedup vs full(v1)+viz on | sims | cpu s/sim | mean gen wall s | load 1-min start→end | c172x best | T38 best | Δbest vs full(v1) viz off (c172x / T38) |
|---|---|---|---|---|---|---|---|---|---|
| rigid-vizon | 20.1 | 4.85× | 540 | 0.275 | 5.81 | 7.6→7.8 | 0.24381 | 0.11201 | – |
| rigid-vizoff | 17.6 | 5.55× | 540 | 0.245 | 5.2 | 7.8→8.0 | 0.24381 | 0.11201 | – |
| reduced-vizoff | 86.3 | 1.13× | 552 | 1.195 | 25.65 | 8.0→8.3 | 0.35189 | 0.12860 | – |
| full-vizoff | 80.5 | 1.21× | 549 | 1.114 | 23.65 | 8.3→8.7 | 0.40717 | 0.19691 | +0.00000 / +0.00000 |
| full-vizon | 97.6 | 1.0× | 549 | 1.323 | 28.13 | 8.7→8.9 | 0.40717 | 0.19691 | +0.00000 / +0.00000 |
| mf-red-k4 | 100.7 | 0.97× | 621 | 1.249 | 32.63 | 8.9→8.9 | 0.35297 | 0.12093 | -0.05420 / -0.07598 |
| mf-red-k8 | 116.9 | 0.83× | 681 | 1.322 | 37.72 | 8.9→9.2 | 0.35297 | 0.12473 | -0.05420 / -0.07218 |
| mf-rigid-k4 | 25.6 | 3.81× | 615 | 0.29 | 7.47 | 9.2→8.2 | 0.39444 | 0.15214 | -0.01273 / -0.04477 |
| mf-rigid-k8 | 43.6 | 2.24× | 684 | 0.457 | 12.74 | 8.2→8.7 | 0.38061 | 0.16468 | -0.02656 / -0.03223 |

| aircraft | gen-0 Spearman reduced vs full(v1) | rigid vs full(v1) | (feasible-at-full only: n, reduced, rigid) |
|---|---|---|---|
| c172x | +0.794 | +0.709 | 18, +1.000, +0.990 |
| T38 | +0.944 | +0.911 | 26, +0.999, +0.977 |

| multi-fidelity case | per-gen Spearman over the re-scored set (c172x; T38) | full sims (c172x/T38) | screen sims |
|---|---|---|---|
| mf-red-k4 | +1.00, +1.00, +0.94; +1.00, +1.00, +1.00 | 36/33 | 276/276 |
| mf-red-k8 | +1.00, +0.98, +1.00; +0.98, +1.00, +0.86 | 66/63 | 276/276 |
| mf-rigid-k4 | +0.00, -0.50, +0.60; +0.80, +0.60, +0.20 | 33/30 | 276/276 |
| mf-rigid-k8 | +0.38, +0.03, +0.15; +0.21, +0.57, +0.30 | 66/66 | 276/276 |

**Per-simulation timing** (`analysis/microbench_fidelity.py`, `microbench.json`). One 90 s scenario (wind +
turbulence), with the configurations interleaved round-robin. Load was about 8. Values are min / median of 3, in
seconds:

| | rigid off / on | reduced off / on | full(v1) off / on |
|---|---|---|---|
| c172x | 0.223 / 0.289 | 1.206 / 1.446 | 1.246 / 1.302 |
| T38 | 0.194 / 0.241 | 1.206 / 1.302 | 1.207 / 1.359 |

Costs are identical with viz on and off in every configuration.

Reading:

* **Viz off** saves 4–23 % per sim (most at rigid, where recording is a larger share of the work) and 12 % (rigid) /
  18 % (full(v1)) of batch wall time. Results are identical.
* **Rigid vs full(v1)** is 5–6× faster. FD's own c172x figures on their loaded box are rigid 0.65 s and v1 n_bend=2
  2.6–3.3 s per scenario.
* **Reduced is not cheaper than full(v1).** It costs the same per sim (1.2 s; batch 86 s vs 81 s). The 2×2 path uses
  `np.linalg.inv`, while the 3×3 uses FD's closed-form `_inv`, and per-step Python overhead dominates. FD's v2 timings
  show the same thing: v2-reduced 3.44 s vs v1 n_bend=2 2.6–3.3 s per scenario.

  As a result, `--multi-fidelity --screen reduced` is slower than plain full(v1): 0.97× at k = 4 and 0.83× at k = 8.
  The screen costs as much as the full pass and then the top-k are flown again.
* **`--screen rigid`** is the fast option: 3.8× at k = 4 and 2.2× at k = 8. But rigid cannot see the structural genes,
  so the Spearman over the re-scored set is low (−0.5 to +0.8).
* **Gen-0 Spearman over the same 32 genomes:**
  * reduced vs full(v1): +0.79 (c172x) / +0.94 (T38);
  * rigid vs full(v1): +0.71 / +0.91;
  * among genomes feasible at full: reduced 1.000 / 0.999, rigid 0.990 / 0.977.

  So the rank disagreement is almost entirely feasibility. Margins between 0.9 and 1.0 pass the reduced screen
  (gate 0.9) and fail full (gate 1.0, fail_cost).
* **Best costs.** The Δbest values in the table are GA-trajectory differences after 3 generations with 1 seed (the
  ranking changes the population). They are not a fidelity bias. Every multi-fidelity best is a full(v1)-scored verdict.
* **Spearman logs** use FD's method (stable argsort, average ranks for ties, Pearson on ranks; FD `v2_compare._spearman`).
  `rescored_pairs` in history.jsonl lets anyone recompute them.


### FD v2 integration status (06:41 PT hand-off)

Done:
* `full` / `reduced` → `flexeval` (import);
* model_version from FD;
* per-aircraft reduced gate;
* swept-wing ≥ 25 % full re-score;
* the 3-level ladder;
* v2 struct genes;
* the recorder via FD's `FlexHookV2` (below).

Still open on FD's side:
* flex_state for the tail and fuselage as node geometry (their raw `flex.*` diagnostics are passed through);
* in-plane wing modes (we send dy = 0).

## Replay interface (Sim Bridge): `run.json`, `genomes.jsonl`, `eval.py`

* **`runs/<id>/run.json`** (`ga-flightsim-run/1`, written at run start). It
  holds:
  * run_id, git_sha, jsbsim_version, code_sha, seed, `eval_seed` (=
    scenario_seed), `fitness_sense: "min"`, `sim_dt_s`;
  * `fidelity` / `fidelity_label`, `multi_fidelity`, `struct_genes`, and
    `model_version {aircraft: …}`;
  * per aircraft: resolved profile, aircraft_root, `scenario_ids`, genes (name,
    min, max, kind, group), model_files_sha, model_version (+
    screen_model_version) and `fitness_cfg`;
  * **resolved scenario definitions** `scenarios[]`: id `<ac>:s<i>`, every
    `sim.Scenario` field (absolute `steps`, wind, gust, `ramp_fpm`,
    `ramp_accel_g`) plus the precomputed `ramp_plan`;
  * `target_semantics`, the aggregate rule, and the eval entry point.
* **`runs/<id>/genomes.jsonl`** (`ga-flightsim-genomes/1`): one row per
  individual per generation. Fields:
  * `individual_id` **`<aircraft>:g<generation>:r<rank>`** (string, r0 = best of the generation; backfilled best-only
    rows use `<aircraft>:g<generation>:best`). Ids are opaque and stable across resumes; match them exactly, don't parse
    them. run.json carries the format in `individual_id_format` (and `scenario_id_format`: `<aircraft>:s<index>`);
  * `eval_seed` (= run.json `eval_seed` = scenario_seed), `index`, `rank`,
    `is_best`, `is_elite`, `carried_elite`;
  * `genome {name: value}`, `genome_norm`, `gains`, `struct`;
  * `cost`, `per_scenario_cost`, `scenario_ids`, `status`, `feasible`,
    `feasibility_fidelity`, `terms`, `terms_available`, `fidelity`,
    `model_version` (+ margins and the multi-fidelity fields).

  Rows are fsync'ed **before** the generation's checkpoint. On resume the file
  is truncated back to the checkpointed generations, so there are no duplicates
  and no gaps. Older runs get `run.json` + best-of-generation rows via
  `python -m evolution.runinfo --backfill RUN_DIR…` (rows flagged
  `backfill: true`; `model_files_current_differs` if FD edited the model since).
  This was done for `phase1-s*`, `phase1-f16fd-s*` and `phase1hdg-s*`, and every
  final-generation row replays exactly.
* **`evolution/eval.py`**: `evaluate(genome, aircraft, scenario, run_cfg,
  recorder=None, *, fidelity=None)`.
  * It is the single evaluation path. The batch workers call `eval.task` (rigid, per scenario: `sim.simulate`) or
    `eval.task_genome` (reduced/full, per genome: FD's `flexeval.evaluate`). Replays go through the same
    `fidelity.evaluate_genome`; at reduced/full they use the aircraft's logged `reduced_gate`.
  * `genome` is the row's `genome` dict (or a normalized list).
  * `scenario` is a run.json entry (dict, used **as given**, never re-drawn), an
    id, an index, a list, or None (= all of the aircraft's).
  * `run_cfg` is a path or a dict.
  * It returns the aggregate plus `per_scenario_cost`, `scenario_ids`, the
    resolved `scenarios`, `model_version`, and `model_version_match` against the
    logged one.
  * `eval.scenario_object(...)` gives a `sim.Scenario` (target / target_cmd /
    target_rate) without re-drawing.
  * Tested: evaluate reproduces genomes.jsonl costs exactly at rigid, reduced and full (multi-fidelity rows). A
    single-scenario call returns `per_scenario_cost[i]`, and an edited scenario changes the cost.
  * Checked by hand at 07:19 PT on `bench_results/fdv2/runs/fdv2-rigid-full` (`737:g3:r0`, full v2) with a recorder:
    * cost 0.19141775547450582 and per-scenario costs equal the row bit for bit, `model_version_match`;
    * `telemetry_check` bit-identical in all 3 scenarios;
    * the recorder got t = 0 with a schema /2 flex_state (80 channels), and `final` at 90 s.

### Recorder protocol

* `recorder(t, fdm[, flex_state])` is called **once at t = 0** after IC + trim
  (+ wind/flex setup), before the first step. It is then called after every
  120 Hz step with t = (k+1)·DT.
* `recorder.final(t_end, fdm[, flex_state])` is called at the end if defined.
* `fdm` is a `sim.ReadOnlyFDM`: writes raise `TypeError`, so a recorder can't
  perturb the run (bit-identity tested).
* `flex_state` is passed only at reduced/full.
* Our `sim.TrajRecorder` produces exactly the old export: trajectory row 0 = the
  t = 0 call, verified bit-identical on 24 flights.

**`controls_timing: "pre_step"`** (header field + `controls_timing_doc`): the
control channels of the row at time t hold the commands applied over the step
**starting** at t. The state channels are the state at t. The final row repeats
the last commands. Unchanged from before and tested.

**`channel_doc`**: a top-level `{name: description}` in every trajectory file
for every non-core channel (no renames). `target_alt_m` = the **reference**
(ramped, what is tracked and scored); `target_cmd_alt_m` = the **commanded**
step; `target_rate_mps` = d(reference)/dt.

**Trajectory header ids**: `fitness_sense` is exactly `"min"` (`fitness_doc` explains), `scenario_index` is the position
in the aircraft's `scenario_ids` (= `sim.make_scenarios` order), and `scenario_id` = `<aircraft>:s<scenario_index>`.

**Ids and seeds.** Individual ids are unique per run (one per aircraft × generation × rank) and stable across resumes
(rows are truncated to the checkpoint and regenerated deterministically; the resume tests compare them). `:best`
appears only in backfilled runs (`runinfo --backfill`), which have best-of-generation rows only and no population
ranking; new runs log every individual as `:r<rank>`, so `:r0` is the best. The difference is intended. Every row
carries `eval_seed`; a run has one scenario seed, so a row without it (older files) uses run.json `eval_seed`.

**model_version (rigid)** = `rigid:jsbsim<ver>:sha8({"model_files": sha16 of every file in <root>/aircraft/<model>/})`.
It covers the whole model directory, non-physics files (`flexwing_meta.json`, init/reset XMLs) included. FD's
`flexeval.rigid_model_version` computes the same string with its own code, so narrowing it on one side would desync the
two. phase1-s1 (04:29–04:37 PT) logged c172x `01333e0c`; today it is `e0a73fc9` because FD edited `c172x.xml` itself at
04:57 PT (0 lb flex point masses, flight-neutral, hence bit-exact costs) and `flexwing_meta.json`. T38 and 737 changed at
04:48 PT (`T38.xml` / `737.xml` re-prepared, 737 socket I/O removed; today T38 `69ac40f7`, 737 `19463d4b`; phase1-s1 f16
used the stock model, today FD's root). These are real model-file edits, so a hash narrowed to XMLs would have
changed too. run.json flags it per aircraft (`model_files_current_differs`).

### `flex_state` (schema `evolution-flex-state/3`, `evolution/fidelity.py`)

At reduced/full a recorder rides a **second flight with FD's own `FlexHookV2`** (identical physics), wrapped by our
`SBHook`. The cost still comes from `flexeval.evaluate`, and the telemetry flight's sim cost must equal FD's
per-scenario `sim_cost` bit for bit (`result["telemetry_check"]`). Live viz (`--viz on`) instead uses FD's own
`record=True` trajectories from the same flight (raw `flex.*` channels only).

**Schema /3 (2026-10-06).** Trajectory files are `ga-flightsim-traj/2`. Branch on these strings.

* **full fidelity (source of truth = FD FE nodes via Sim Bridge `v2_map` 2.0.0):**
  * shared channel names `wingR.*` / `wingL.*` are FE nodal (dof `dz`, `dx`, `twist`; in-plane `dx = -v·0.3048`);
  * plus `htail.*`, `vtail.*`, `fuselage.*` and `struct.*` scalars (FD §11 signs);
  * the previous 9-node modal wings are kept as `wingR_modal.*` / `wingL_modal.*` (`node_span_frac` in the structure
    header; dof `dz`, `dy=0`, `twist`).
* **reduced:** modal `wingR` / `wingL` only (unchanged names); no v2_map components.
* Import: `fidelity.v2_map_mod()` loads `sim-bridge/sim_bridge/v2_map.py` read-only (stdlib only; no package import).

**Public FlexState API** (Sim Bridge can stop using `_aircraft_entry` / `_struct_obj`):

| member | meaning |
|---|---|
| `.schema` | `"evolution-flex-state/3"` |
| `.structure` | static header (components, units, `v2_map` block at full) |
| `.fidelity` / `.model_version` | as flown |
| `.eta` / `.raw` | FD modal state / `coupler.last` |
| `.fd_model` | FD `FlexBodyModel` (full) or `FlexWing` (reduced); read-only |
| `.nodes()` | current-frame FE nodal values `{body: {field: [float]}}`, or `None` at reduced |
| `.node_layout()` | FD `node_layout` (body FRD ft, origin CG), or `None` at reduced |
| `.v2_geometry` / `.v2_map_version` | v2_map geometry / `"2.0.0"` at full, else `None` |
| `.channels()` | SB channels + `flex.*` |
| `.as_dict()` | JSON-able snapshot |

Also public: `fidelity.make_fd_model(profile_d, struct, fidelity)` builds the FD model without private helpers.
Private `_struct_obj` / `_aircraft_entry` remain for the batch path until Sim Bridge switches.

**Twist / signs.** Unchanged for modal wings (`fd_to_structure_channels`). FE wings / empennage use v2_map's
`SIGN_TABLE` (= FD INTERFACE_v2 §11). Tests: `test_v2_map.py`, `test_recorder_inert_t0_flex_state_and_twist_sign`,
`test_twist_convention_known_nose_up_case`.

**`eval.evaluate(..., pin=<model_version>)`:** raises `RuntimeError` on mismatch. Batch/`pin_model_version` still
enforced separately by `batch.check_pins` and the cache guard.

## Phase-1 v5 candidate (`configs/phase1_v5.json`, runs `phase1v5-s{1,2,3}`)

This implements `genome/HANDOFF_phase1_v5.md`. The profiles are `genome/exports/evolution_phase1_v5_profiles.json`,
which is v4 plus 4 keys and, since genome's handoff section 6 (~07:30 PT), a **v5-only ki_alt upper bound of 0.5** (v4 keeps
0.05). `configs/phase1_v5.json` equals the updated export (repo-relative `aircraft_root`). The reference genomes still
re-fly bit for bit. **The `phase1v5-s{1,2,3}` runs below used the old 0.05 bound.** With the new bound, `phase1v5-ki05-s1` (seed 1, 07:54–08:00 PT,
379 s, load 0.9→21.3) gives c172x 0.225146 (ki_alt 0.074), T38 **0.0877912874821699** (ki_alt 0.0656, equal to genome's
`runs/v5_kialt05_t38_s1` bit for bit), and 737 0.137911 (ki_alt 0.0013, kp_pitch at max, downdraft residual +3.0 ft; worse
than the old-bound 0.111260 on this seed). See `analysis/STATUS_E.md`. **v4 (`phase1_hdg.json`) stays the default.** v5 is a separate, opt-in config with
c172x/T38/737 only (f16 is not in the v5 export).

* **`w_hold`, `hold_ref_ft`, `hold_settle_s`** (defaults 0 / 5 / 5).
  * `cost += w_hold · hold_osc`, added after the heading term.
  * `hold_osc` is the RMS, over hold samples, of (e − mean of e in its window), with e = h_cmd − h, divided by
    `hold_ref_ft`.
  * A hold sample is one where the reference equals the command (|ref − cmd| < 1e-6 ft) and the reference rate is 0,
    excluding the first `hold_settle_s` of each stretch.
  * `sim.hold_osc_term` is a verbatim port of genome's code. `t` is built as `k·DT`.
  * Reported, not in the cost: `hold_osc`, `hold_pp_ft` (max ptp(e) over the windows), and `draft_residual_ft` /
    `draft_max_err_ft` for the downdraft scenario.
* **`disturbance_scenario`** `{downdraft_fps, onset_t_s, onset_ramp_s, steps_rel_ft}` (default None).
  * `make_scenarios` appends one scenario after the n standard ones: a copy of the calm one that holds h0, with no
    wind or gusts, plus `draft_fps / draft_t_s / draft_ramp_s`. It consumes no RNG draws, so scenarios 0..n−1 are
    unchanged.
  * `vertical_gust_series` adds `draft·0.5·(1 − cos(π·clip((t − t0)/ramp, 0, 1)))` to wind-down (+ = down).
  * The cost is the mean of all n + 1 scenarios. Downdraft sizes (V_TAS·tan 1.5°): c172x 4.69, T38 15.43, 737
    12.86 ft/s.
* **Flags off = bit-identical.**
  * `Scenario.to_dict` omits the three draft keys when `draft_fps == 0`, so v4 scenario dicts, run.json and
    scenario cache keys are unchanged.
  * The v4 / phase1 / baseline tests still reproduce genome's costs bit for bit.
  * Cache keys change for every new profile key and for a downdraft scenario (`test_v5.py`).
* **run.json** lists the downdraft scenario as `<ac>:s3` with its draft fields, and `fitness_cfg` documents the hold
  term and the disturbance.
* **Term key `hold`** was added to the uniform term keys (0.0 unless `w_hold > 0`). FD's `flexeval.TERM_KEYS` doesn't
  have it yet.

**Bit-identical check** (`test_reproduces_genome_v5_reference_genomes_bitwise`, handoff section 3). Every
per-scenario cost (calm, wind 1, wind 2, downdraft), the mean of 4, and hold_osc per scenario (6 digits) equal
genome's:

| genome | cost (mean of 4) |
|---|---|
| c172x v5 s1 | 0.20625113824760744 |
| T38 v5 s1 | 0.08778028702197657 |
| 737 v5 s1 | 0.11126020207078736 |
| c172x v4 genome on v5 | 0.22748161325788208 |

Runs `phase1v5-s{1,2,3}` (`configs/phase1_v5.json`, GA seed 1/2/3, scenario_seed 1, 8 workers): s1 06:38–06:42, s2 06:42–06:47,
s3 06:47–06:57 PT. Walls: 235 s (load 5.6→12.0), 323 s (12.0→17.2), 618 s (17.2→43.3; other agents' jobs pushed the box to 62).
0 re-sim mismatches. All 27 trajectory files (gens 0 / 9 / 19 per aircraft) validate. Every run.json carries the 4th scenario
`<ac>:s3` (downdraft 4.69 / 15.43 / 12.86 ft/s). Report: `analysis/phase1v5_report.py` → `logs/phase1v5_report.json`.

| aircraft | seed | best (v5 task) | hold p-p calm / worst ft | downdraft residual / max err ft | ki_alt | ki_pitch | v4 genome on v5 | v5 genome on v4 (v4 best) |
|---|---|---|---|---|---|---|---|---|
| c172x | s1 | 0.222942 | 5.94 / 10.22 | -0.40 / 6.48 | 0.05 | 0.00265 | 0.2431 | 0.2004 (0.1975) |
| c172x | s2 | 0.205943 | 3.41 / 10.68 | +0.15 / 4.76 | 0.05 | 0.000103 | 0.2159 | 0.2086 (0.1954) |
| c172x | s3 | 0.232252 | 5.30 / 8.60 | +2.01 / 4.76 | 0 | 0 | 0.2598 | 0.2290 (0.2009) |
| c172x | mean ± std (pop.) | 0.22038 ± 0.01089 | | | | | | |
| T38 | s1 | 0.087780 | 0.39 / 5.94 | -0.00 / 5.73 | 0.05 | 0.00572 | 0.1401 | 0.0870 (0.0966) |
| T38 | s2 | 0.111430 | 0.13 / 5.25 | +3.26 / 3.67 | 7.36e-07 | 0 | 0.1345 | 0.1008 (0.0930) |
| T38 | s3 | 0.091581 | 0.93 / 6.17 | -0.00 / 4.92 | 0.05 | 1.26e-05 | 0.1297 | 0.0931 (0.0925) |
| T38 | mean ± std (pop.) | 0.09693 ± 0.01037 | | | | | | |
| 737 | s1 | 0.111260 | 0.93 / 6.96 | +0.01 / 6.57 | 0.0278 | 0.002 | 0.1087 | 0.1116 (0.1068) |
| 737 | s2 | 0.108847 | 1.05 / 7.40 | -0.00 / 6.86 | 0.0354 | 0.00525 | 0.1535 | 0.1069 (0.1106) |
| 737 | s3 | 0.113551 | 1.22 / 6.47 | -0.00 / 5.10 | 0.05 | 1.72e-05 | 0.1433 | 0.1187 (0.1089) |
| 737 | mean ± std (pop.) | 0.11122 ± 0.00192 | | | | | | |

Compared with Genome's reported v5 results (HANDOFF_phase1_v5.md):

* **T38 s1 and 737 s1 reproduce Genome bit for bit.** Our best costs are 0.08778028702197657 and 0.11126020207078736, the
  exact reference values, with the same genes.
  * T38: **ki_alt = 0.05** (at its upper bound) and **downdraft residual −0.00 ft**, as Genome reports (0.0 / 5.7 ft).
  * T38 s3 also lands at ki_alt 0.05 with residual −0.00 ft. T38 s2 found another basin: ki_alt 7.4e-07, residual +3.26 ft,
    cost 0.1114.
* **c172x does not match**, and this is the same budget difference as with v4. Genome's c172x v5 run (`runs/v5_sweep_w01`) used
  **pop 48 × 40 generations**; ours is the phase1 GA (pop 32 × 20).
  * Their reference genome scores 0.20625 on our v5 code (bit-identical, `test_v5`). Our seeds reach 0.2229 / 0.2059 / 0.2323.
  * Calm hold p-p: Genome 3.9 / 3.7 / 3.2 ft; ours 5.94 / 3.41 / 5.30 ft. Only s2 falls inside Genome's 3.2–3.9 ft band.
    Genome's s2/s3 also changed the scenario seed, so only s1 is directly comparable.
* **The downdraft term does what Genome describes on the jets.** Every v4 genome re-flown on v5 shows a +5.8 to +7.2 ft
  residual on T38 / 737 (except 737 s1, −0.00). The v5 optima drive it to about 0 when ki_alt is near its bound.
* **Not adopted as the default**, as agreed: v4 (`phase1_hdg.json`) stays the default.

## Phase 2 wiring (Genome `phase2_flex`, FD v2 after the §12 mass fix)

Configs, all generated by `analysis/make_phase2_configs.py` from `configs/phase1_hdg.json`:
- `configs/phase2_pilot.json`: 64 × 60, seed 1, viz off.
- `configs/phase2_smoke.json`: 16 × 5, full fidelity only.
- `configs/phase2_bench_rf.json` and `configs/phase2_bench_rrf.json`: 32 × 8, cache off, ladders rigid→full and rigid→reduced→full.

What every config sets:
- **Genome (20 genes).** The v4 controller (8 genes, from `phase1_hdg`, with v5's `ki_alt` upper bound of 0.5) plus FD's 12 v2 struct genes from `flexbody.gene_schema()`. Bounds, scale and baseline (`Gene.default`) are read from FD at load time.
- **`struct_asymmetric`.** Defaults to false. Set it to true to add FD's 2 asymmetry genes (14 struct genes).
- **`init`.** `{"mode": "baseline", "sigma": 0.10, "blocks": ["struct"]}`, where sigma must be in [0.10, 0.15] (normalized). Generation 0 is the usual uniform draw. Then the struct columns are set to `clip(encode(FD baseline) + sigma·N(0,1), 0, 1)`. This is Genome's `init_pop.generation_zero` algorithm, but Genome uses sigma 0.05. The default `"uniform"` leaves generation 0 bit-identical to earlier runs.
- **Roots.** `aircraft_root` is `"flight-dynamics/jsbsim_root"`, team-relative. `sim.abs_root` resolves it under `EVOLUTION_FD_DIR` when that is set, otherwise under the team root. Full fidelity uses `<root>_v2` (FD's `flexbody.ROOT_V2`). The configs contain no absolute paths.
- **Ladders and gates.**
  - `multi_fidelity` is global: rigid→full with `min_full_frac` 0.25, so at least 25 % of the population plus the elites are re-scored at full.
  - `multi_fidelity_per_aircraft` overrides any of `enabled`, `screen`, `top_k`, `min_full_frac` and `mid_k` for one aircraft. The pilot uses it for c172x: rigid→reduced→full.
  - `fidelity_per_aircraft` reduced gates: c172x 0.9, T38/737 1.0. The full gate is FD's 1.0.
  - Feasibility is trusted only from the authoritative stage, which is full.
- **`pin_model_version`** (`{aircraft: {fidelity: FD model_version}}`). The values come from FD's `v2_results/model_versions_post_mass.json`.
  - `Batch.run()` refuses (SystemExit) in these cases:
    - a pin still reads `PENDING-FD-NEW-MODEL-VERSION`; this happens before the run dir is touched;
    - a reduced/full fidelity in the aircraft's ladder is unpinned;
    - FD's current string differs from the pin; this is checked right after the per-aircraft describe step, before run.json, any evaluation, cache write or checkpoint.
  - Resuming re-checks the pins.
  - Reduced pins are FD's default-gate (0.9) strings. Evolution's reduced string also hashes the per-aircraft gate, so for T38/737 at gate 1.0 the run's own string differs: `reduced:flexv1:87b7e096` / `67224a27`, against FD's `52be19ae` / `870ff0b7`. `check_pins` accepts a pin that matches either FD's default-gate string or the run's own string. It then allows exactly the run's string in the cache guard.
  - The cache guard is `EvalCache.pins`. It refuses to store any reduced/full result whose model_version is not allowed.
  - `python -m evolution.batch --config C --model-versions` prints the current and pinned strings and whether they match. It is read-only.
- **Mass-credit clip** (`Profile.flex_mass_credit_clip`, a subset of `["ht", "vt", "fus"]`): Genome's interim `fd_bridge.mass_term_v2` clip, applied as `fidelity.apply_mass_credit_clip`.
  - It replaces FD's `J_mass` with the clipped value in every per-scenario cost and records `J_mass_fd` and `mass_credit_delta`.
  - It is bit-identical when no clipped body loses mass.
  - It is off in all Phase 2 configs, matching Genome's `phase2_flex` since 08:15 PT, where FD's §12 fix replaced it. `make_phase2_configs.py --clip` turns it on for an A/B comparison.
- **Cost.** The cost is FD's `flexeval` cost, not reweighted: sim cost (track, effort, comfort, heading, hold) plus every FD pre-flight and flown term at `StructWeightsV2` weights. That is 23 `TERM_KEYS`, including the 5 §12 sizing terms and the 2 full-only flown peak terms.
- **Flex rows** in `genomes.jsonl` also carry `mass_total_frac`, `mass_lb` (FD `mass_summary`, per-body Δlb), `task_cpu_s`, and the clip fields when the clip is set.

Checks (read-only):
- `$PY evolution/analysis/phase2_check.py RUN_DIR [--compare RUN_DIR_B] [--json OUT]`, on the last generation:
  - flags a negative structural cost (sum of `J_*` terms) at the optimum, and any negative term;
  - prints the fraction of the final population within 2 % (normalized) of each struct gene's floor;
  - flags a stiffness gene where ≥ 25 % of the population sits at its floor;
  - reports the invalid rate;
  - exits 1 on any flag.
- `$PY evolution/analysis/phase2_fidelity_bench.py RUN_DIR [--per-aircraft 16]`: re-scores a sample of the run's genomes at rigid, reduced and full, outside the cache. It reports CPU per scenario, Spearman reduced-vs-full and rigid-vs-full, and checks full against the logged costs bit for bit.

## Tests

`python -m pytest evolution/tests -q`: **80 passed** in 89.3 s (2026-10-06 08:03–08:05 PT, load 4.0→3.5; log `logs/pytest_final4.log`).

* `test_paths.py` (5): abs_root / team_rel inverse, relative == absolute profile, FD_DIR default, source repo in both layouts.
* `test_eval.py` replay test also checks: id formats + `individual_id_format`, `eval_seed` per row, team-relative
  run.json paths, trajectory `fitness_sense == "min"` and `scenario_id`.

* `test_equivalence.py`: `sim.simulate` is bit-identical to the prototype's,
  and a whole batch equals the prototype's `evolve.run` (best cost, gains,
  genome).
* `test_cache.py`: a rerun under a new run id computes 0 sims with a hit rate
  of 1.0 and identical results. The key changes for each of aircraft, a 1-ulp
  genome change, profile, gain bounds, scenario seed, scenario-set seed, JSBSim
  version, and code sha. A cached result equals a fresh simulation.
* `test_determinism.py`: repeat sims match, record mode doesn't perturb the
  result, and `workers=1 sequential` equals `workers=8 concurrent`.
* `test_resume.py`: SIGKILLs the whole process group mid-run (after ≥3
  checkpointed generations of the leading aircraft) and resumes with
  `--resume`. Populations, RNG state, history stats, best genomes, and
  trajectories must equal an uninterrupted no-cache reference, and
  `history.jsonl` must have one line per (aircraft, generation) with sessions
  0 and 1.
* `test_trajectory.py`: the validator accepts real runs, and the index shape
  plus gen 0/mid/final are present. Units, frame, and orientation are checked:
  at t = 0 the nose points north and body-down points to ENU-down. The
  quaternion agrees with the Euler angles and with the body velocity, a
  200-attitude sweep checks the quaternion function, and the validator rejects
  degrees, swapped quaternion components, wrong units, throttle out of range,
  a wrong schema, a missing channel, swapped x/y, and bad index entries.
* `test_jets.py`: the T38 and 737 trim points match Flight Dynamics
  (throttle 0.354 / 0.586, α 4.61° / 3.28°, gear up, 2 engines). The 737 fails
  to trim at 185 KCAS. The T38 throttle never exceeds 0.5.
* `test_sanitize.py`: the 737's socket elements are stripped, results are
  identical to the stock model, and FCS `<input>`/`<output>` are kept.
* `test_phase1.py` (15): ramp shape (600 fpm / 0.1 g limits, continuity at the
  command, exact trapezoid duration, symmetric descent, triangle for short
  moves, mid-move reversal, ITAE clock at the command); feed-forward wiring
  (D-only outer loop climbs at ~10 ft/s with FF and ~0 without; no ramp => FF
  flag is a bit-identical no-op); comfort term (cost = track + 2 effort + 0.05
  comfort, hand-computed terms) and live roll gains; log0 decode/encode and
  `make_schema` kinds; `phase1.json` content and agreement with genome's export
  (keys/genes genome lists under `_needs_code`, i.e. the heading-hold request, are
  skipped); loading from Flight Dynamics' `jsbsim_root` (all four present, c172x/T38/737
  trim, FD copy == stock bit-for-bit, the loaded XML never has socket elements, FD
  folder unchanged, a model missing from a root = clean `load_failed`); FD f16 trim ==
  FD's numbers, no sockets, point masses [0]=0/[1]=230, 20 s hold, FD == stock cost,
  throttle ≤ 0.5; eval key covers every new field + model file hash; genome/'s best
  Phase-1 genomes re-fly with bit-identical per-scenario cost/track/effort/comfort;
  ramped trajectory validates and hold window.
* `test_heading.py` (9): the handoff's three best genomes re-fly bit for bit
  (cost and per-scenario costs; normalized genomes decode equal); the flag off is
  inert; schema layout (kp_hdg log, ki_hdg log0 after the legacy 6) and name
  validation; `phase1_hdg.json` == genome's export; the eval key changes for 6
  heading variants; batch gen 0 has 8 genes; socket guard (refuse for an explicit
  root, strip loudly for package data, FD root has 0 elements).
* `test_eval.py` (15):
  * Recorder: bit-identity, the t = 0 call, pre_step controls, the read-only FDM.
  * Trajectory docs: `channel_doc` / `controls_timing` (reference vs command).
  * Fidelity (FD flexeval v2):
    * model_version formats (`full:flexv2`), FD's 23 never-NaN terms, the gate in the reduced sha, FD's MARGIN_GATE
      restored, per-aircraft defaults;
    * our reduced/full == a direct `flexeval.evaluate` call bit for bit, and one scenario == its entry in the
      all-scenario call; FD's prepare functions are disabled in-process;
    * flex recorder inert with `telemetry_check` bit-identical; flex_state shape, t = 0 probe, SB tip == FD raw tip
      (dz and twist sign);
    * the twist-convention known cases (v1 and v2);
    * per-aircraft margin gates (0.95: gate 0.9 ok, gate 1.0 fails, not flown, 2·fail_base).
  * Replay: `eval.evaluate` reproduces genomes.jsonl exactly (rigid; scenario used as given; single-scenario call; edited
    scenario changes the cost); viz on/off identical at rigid and at full (+ live file only with viz on); a reduced run
    reproduces and records `reduced_gate`.
  * Multi-fidelity (rigid → full with 20 genes): ranking, re-scoring of top-k + carried elites, feasibility only from
    full (`feasible: null` otherwise), full rows replay exactly, Spearman in history, and kill-between-rows-and-
    checkpoint + resume == an uninterrupted run. 3-level ladder rigid → reduced → full with min_full_frac 0.6 (k_full =
    3 of 5), all three Spearman pairs, `ladder_cost`, run.json ladder versions; a descending ladder is rejected.
  * Backfill of a phase1-s1 copy replays exactly.
* `test_v5.py` (8):
  * genome's four v5 reference genomes re-fly bit for bit (per-scenario costs, mean of 4, hold_osc);
  * `phase1_v5.json` == genome's export == v4 + 4 keys;
  * flags off are inert, v4 scenario dicts are unchanged, and the hold-alone / downdraft-alone compositions bisect
    exactly;
  * downdraft onset shape, and no RNG draws consumed;
  * validation, and the cache key changes for each new key;
  * run.json carries the 4th scenario and the fitness doc.

