# Hand-off to Evolution Runner: Phase-1 v5 candidate (2026-10-06)

You can apply this file on its own, without reading genome/ code. It adds **one cost term** (hold quality) and **one
extra scenario** (a sustained downdraft) to the Phase-1 task you already run bit for bit with genome/
(`configs/phase1_hdg.json` = genome's v4). No genes change. **v5 is a candidate, not the new default:** genome/ keeps
`phase1_default` = v4 (frozen as `presets/phase1_v4.json`) and adds `phase1_v5` next to it. Implement it behind flags
that default off, so `phase1_hdg.json` stays bit-identical.

## 0. Why

Sim Bridge's replay of the v4 runs showed:
- gains vary up to 60× between seeds at similar cost (a flat cost surface);
- ki_alt sits at zero on most jet seeds;
- the c172x oscillates during the hold (Sim Bridge: ~17 ft peak-to-peak on s1; genome measures 6.4 ft calm / 12.1 ft
  worst scenario on `hdg_after_c172x_s1`, after excluding the first 5 s of the hold);
- learning stalls after generation ~9.

Nothing in v4 rewards a well-damped hold (track_alt is an ITAE-like |e| integral that a small oscillation barely moves)
or an altitude integrator (no scenario has a sustained disturbance that a P-D loop can't cancel). v5 adds one of each.

## 1. Profile changes

### 1a. `genome/aircraft_profiles/phase1_shared.json` (the set of record): blocks added

```json
"task": {
  "v5": {
    "enabled_by_default": false,
    "hold_quality": {
      "cost_term": "hold_osc",
      "definition": "hold windows = stretches where the altitude reference equals the commanded altitude with zero reference rate, minus the first hold_settle_s of each stretch; hold_osc = RMS over all hold samples of (e - mean_window(e)), e = h_cmd - h, divided by hold_ref_ft",
      "hold_settle_s": 5.0,
      "hold_ref_ft": 5.0,
      "weight": 0.1
    },
    "disturbance": {
      "kind": "sustained downdraft, one extra calm scenario appended to the scenario set (same aggregation as the rest: mean)",
      "steps_rel_ft": [[0.0, 0.0]],
      "onset_t_s": 10.0,
      "onset_ramp_s": 4.0,
      "onset_shape": "w = draft * 0.5*(1 - cos(pi*clip((t - onset_t_s)/onset_ramp_s, 0, 1))), added to wind-down-fps (+ = down)",
      "scaling": "downdraft_fps = V_TAS(trim) * tan(gamma_equiv_deg)",
      "gamma_equiv_deg": 1.5
    }
  }
},
"aircraft": {
  "c172x": {"disturbance": {"downdraft_fps": 4.69}},
  "t38":   {"disturbance": {"downdraft_fps": 15.43}},
  "b737":  {"disturbance": {"downdraft_fps": 12.86}}
}
```

(The file also carries `_why` notes.) genome/ presets opt in with `"hold_quality": true` and
`"disturbance_scenario": true` (`presets/phase1_v5.json`); both need the shared set, and the legacy task never reads it.

- **Capture / hold definition.** It is defined by the *reference*, not by the aircraft, so it can't be gamed by never
  "capturing": a sample is in a hold window when the (smoothed, 0.1 g-cornered) reference equals the commanded altitude
  (|ref − cmd| < 1e-6 ft) and the reference rate is exactly 0, and at least `hold_settle_s` = 5 s have passed since that
  stretch began. On the standard 4000 → 4200 → 4000 ft profile (600 fpm, 0.1 g corners; each 200 ft move takes
  ~23.1 s) that gives two windows, 33.108–49.992 s and 78.108–89.992 s (the 0–5 s stretch before the first step is
  all settle time). In the downdraft scenario (no step) the window is 5.0–89.992 s.
- **hold_osc** = RMS over all hold samples of (e − mean of e in that window), e = h_cmd − h, / `hold_ref_ft`. It
  measures oscillation, not a standing offset (that stays track_alt's and the integrator's job). `hold_ref_ft` = 5 ft:
  the v4 best genomes have an RMS hold deviation of ~2.6 ft on the c172x (hold_osc ≈ 0.5) and ~1.2 ft on the jets.
- **Weight 0.1** from a sweep on c172x seed 1, pop 48 × 40 (genome `runs/v5_sweep_w*`, `v5_sweep_report.py`); the
  v4 yardstick is the same genome scored by the v4 task (3 scenarios):

  | hold weight | hold_osc | hold p-p calm / worst (ft) | v4-task cost | v4 track_alt |
  |---|---|---|---|---|
  | v4 genome | 0.526 | 6.4 / 12.1 | 0.1960 | 0.0586 |
  | 0 (downdraft only) | 0.518 | 7.4 / 12.6 | 0.2006 | 0.0621 |
  | 0.03 | 0.443 | 5.6 / 10.4 | 0.1971 | 0.0518 |
  | **0.1** | 0.373 | 3.9 / 9.7 | **0.1951** | 0.0493 |
  | 0.3 | 0.254 | 3.1 / 9.1 | 0.2228 | 0.0422 |

  0.1 is the largest weight with no loss on the v4 task; 0.3 buys its hold quality with more elevator activity and
  comfort cost (+14 % on the v4 task). The downdraft alone does not change hold quality, so the term is what damps it.
  At 0.1 the hold term is ~18 % of the c172x cost.
- **Downdraft size** = V_TAS(trim)·tan 1.5°: the same 1.5° air-relative flight-path offset on every aircraft, which a
  P-D altitude loop can only cancel with a standing altitude error (expensive under the ITAE weighting) and an
  integrator cancels exactly. c172x 179.1 ft/s → 4.69 ft/s; T38 589.2 → 15.43; 737 491.0 → 12.86.

### 1b. `genome/exports/evolution_phase1_v5_profiles.json` (generated with `$PY genome/export_shared_profiles.py`)

Identical to `exports/evolution_phase1_profiles.json` (= your `phase1_hdg.json`, the v4 export) plus four keys per
profile, listed in the file's `_needs_code`:

```json
"w_hold": 0.1,
"hold_ref_ft": 5.0,
"hold_settle_s": 5.0,
"disturbance_scenario": {"downdraft_fps": 4.69, "onset_t_s": 10.0, "onset_ramp_s": 4.0, "steps_rel_ft": [[0.0, 0.0]]}
```

`downdraft_fps` is 15.43 for `phase1_T38` and 12.86 for `phase1_737`. `w_hold` is already divided by w_track (= 1).
f16 is not in the v5 export (not part of this request); keep it on v4.

## 2. Code changes in evolution/

Line numbers refer to your files as of 05:33 PT today (after heading hold).

### 2a. `sim.py`: `Profile` (dataclass, heading fields at lines 88–93)

Add after `hdg_rms_ref_deg`:

```python
    # Phase-1 v5 (genome/HANDOFF_phase1_v5.md): hold-quality term + sustained-downdraft scenario
    w_hold: float = 0.0                  # cost += w_hold * hold_osc
    hold_ref_ft: float = 5.0
    hold_settle_s: float = 5.0
    disturbance_scenario: Optional[Dict] = None   # {"downdraft_fps", "onset_t_s", "onset_ramp_s", "steps_rel_ft"}
```

In `from_dict` (~line 137) validate `w_hold >= 0`, `hold_ref_ft > 0`, `hold_settle_s >= 0`, and for
`disturbance_scenario`: `downdraft_fps` finite, `onset_ramp_s > 0`, `onset_t_s >= 0`; convert `steps_rel_ft` to tuples
like the top-level one (~line 128). In `to_dict` (~line 148) emit them as lists.

### 2b. `sim.py`: `Scenario` (dataclass, line 153) and `vertical_gust_series` (line 279)

Add three fields at the end of `Scenario`:

```python
    draft_fps: float = 0.0               # sustained vertical wind (+ = down), 1-cos onset
    draft_t_s: float = 10.0
    draft_ramp_s: float = 4.0
```

and at the end of `vertical_gust_series`, before `return w`:

```python
        if self.draft_fps:
            t = np.arange(n) * DT
            x = np.clip((t - self.draft_t_s) / self.draft_ramp_s, 0.0, 1.0)
            w = w + self.draft_fps * 0.5 * (1.0 - np.cos(np.pi * x))
```

- genome/ computes exactly this (`sim_ext.ExtScenario.vertical_gust_series`) on top of the base series, which is all
  zeros in the downdraft scenario (no gusts). Keep `t = np.arange(n) * DT` and the expression order for bit identity.
- **Cache / flag-off identity:** `to_dict()` now carries three more keys on every scenario. Costs don't change with
  `draft_fps = 0` (the branch is skipped), but if your cache key hashes the scenario dicts, every v4 entry will miss
  once. If you want to keep the v4 cache warm, drop the three keys from `to_dict()` when `draft_fps == 0` (and let
  `from_dict` default them).

### 2c. `sim.py`: `make_scenarios` (line 306)

After the `if profile is not None:` block (~line 330), before `return out`:

```python
    ds = profile.disturbance_scenario if profile is not None else None
    if ds:
        calm = out[0]
        d = Scenario(**{**calm.to_dict(), "steps": [(float(t), float(profile.h0_ft + dh)) for t, dh in ds["steps_rel_ft"]]})
        d.wind_north_fps = d.wind_east_fps = 0.0
        d.gust_sigma_fps = 0.0
        d.discrete_gust_fps = 0.0
        d.draft_fps = float(ds["downdraft_fps"])
        d.draft_t_s = float(ds["onset_t_s"])
        d.draft_ramp_s = float(ds["onset_ramp_s"])
        out.append(d)
    return out
```

- The downdraft scenario is a copy of the calm scenario (same seed, duration, h0, speed, ramp settings), holding h0,
  with no wind/gusts. It is **appended last**, after the n standard scenarios, and **does not consume RNG draws**, so
  scenarios 0..n−1 are unchanged. With `"scenarios": 3` you fly 4; the per-genome cost is the **mean of all 4**.
- `discrete_gust_t_s` / `_len_s` keep the calm scenario's defaults; they're inert with `discrete_gust_fps = 0`.

### 2d. `sim.py`: `simulate()` (~line 742)

Before the loop, next to `c_psi`:

```python
    hold_on = P.w_hold > 0
    c_h: List[float] = []; c_ref: List[float] = []; c_cmd: List[float] = []; c_rate: List[float] = []
```

Inside the loop, after the envelope checks (the `if status != "ok": ... break` at ~line 838), i.e. for every step that
is flown, using the values already computed at the top of the step:

```python
        if hold_on:
            c_h.append(h)
            c_ref.append(target)
            c_cmd.append(sc.target_cmd(t)[0])      # your target_cmd returns (h_cmd, t_step)
            c_rate.append(h_ref_dot)
```

(genome/ records `h` = `position/h-sl-ft` read at the start of the step, `target` = the smoothed reference,
`target_cmd` = the commanded altitude and `target_rate` = the reference rate used for feed-forward, at t = k·DT.)

Cost at the end, **after the heading term** (genome's weighted sum is
`track + 2·effort + 0.05·comfort + 0.01·heading + 0.1·hold_osc`, accumulated in that order):

```python
        if hold_on:
            hold_osc = hold_osc_term(np.asarray([k * DT for k in range(len(c_h))]), np.asarray(c_h),
                                     np.asarray(c_ref), np.asarray(c_cmd), np.asarray(c_rate),
                                     P.hold_settle_s, P.hold_ref_ft)
            cost = cost + P.w_hold * hold_osc
```

with this module-level helper (a verbatim port of genome's `fitness.hold_mask` / `hold_windows` / `obj_hold_osc`; keep
the Python loop and numpy calls as written, `np.mean` is pairwise):

```python
def hold_osc_term(t, h, ref, cmd, rate, settle_s, ref_ft):
    done = (np.abs(ref - cmd) < 1e-6) & (rate == 0.0)
    m = np.zeros(t.size, bool)
    start = None
    for i, d in enumerate(done):
        if d and start is None:
            start = t[i]
        elif not d:
            start = None
        if d and t[i] >= start + settle_s - 1e-9:
            m[i] = True
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return 0.0
    wins = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    e = cmd - h
    dev = np.concatenate([e[w] - np.mean(e[w]) for w in wins])
    return float(np.sqrt(np.mean(dev ** 2))) / ref_ft
```

- genome's `t` channel is the Python float `k * DT` (checked equal to `np.arange(n) * DT` at DT = 1/120 for these
  runs; building it as `k * DT` is the safe form, since a last-bit difference could move a window edge by one sample).
- Failed runs keep the fail cost; no hold term is added. Add `out["hold_osc"] = hold_osc` for reporting.
- Optional diagnostics genome reports (not in the cost): `hold_pp_ft` = max over windows of ptp(e) with
  `hold_settle_s` = 5; `draft_residual_ft` = mean(e) over the last 20 s of the downdraft scenario (+ = below);
  `draft_max_err_ft` = max |e| after the onset.

### 2e. `batch.py`

Nothing structural: `make_scenarios(cfg["scenarios"], scenario_seed, prof)` (line 299) returns n + 1 scenarios when the
profile has `disturbance_scenario`, and the per-scenario loops (lines 387, 405) already iterate over
`len(st["scenarios_d"])`. Check that anything that assumes `len(scenarios) == cfg["scenarios"]` (reports, trajectory
indices, the fidelity module's per-scenario term table) uses the actual list length. genes and `genome.py`: unchanged.

### 2f. Cache

The new Profile keys are in the resolved profile and sim.py is in `code_sha`, so v5 entries miss automatically. See 2b
for the v4 cache.

## 3. How to check you match genome/ bit for bit

Re-fly these genomes on the v5 export profiles (3 standard scenarios + the downdraft, scenario_seed 1) and compare the
cost (mean of 4), every per-scenario cost and, if you report it, hold_osc per scenario. genome/ gets them with
`adapter.load_task("phase1_v5", {"aircraft": ...})`. Gene order: [kp_alt, ki_alt, kd_alt, kp_pitch, ki_pitch, kd_pitch,
kp_hdg, ki_hdg]; normalized genomes are in each run's `best_gains.json` under `flight_sim_3d/genome/runs/`.

| genome (genome/ run) | genes | cost (mean of 4) | per-scenario costs (calm, wind 1, wind 2, downdraft) | hold_osc per scenario |
|---|---|---|---|---|
| c172x (`runs/v5_sweep_w01` = v5 s1) | 0.31437738799504616, 0.05, 0.3366549881674388, 0.08049753722927444, 1.190156607327809e-05, 0.013658908124502132, 2.369220826682193, 0.00013558574893233208 | 0.20625113824760744 | 0.12785226753903256, 0.37197659717077186, 0.1995611326112344, 0.1256145556693909 | 0.256607, 0.519764, 0.366012, 0.349807 |
| T38 (`runs/v5_t38_s1`) | 0.22365803353253752, 0.05, 0.22164873364794022, 0.11246859204665012, 0.005717237540206061, 0.03970195171260285, 1.691686672038538, 0.001567717437456872 | 0.08778028702197657 | 0.05596533384050053, 0.15315702451795365, 0.10218697003758949, 0.03981181969186257 | 0.017774, 0.268917, 0.217608, 0.219374 |
| 737 (`runs/v5_b737_s1`) | 0.22211834556429566, 0.027826279920683275, 0.39030385285880465, 0.04034489490119028, 0.0019985506113642682, 0.019528305027763263, 0.29667099721658546, 0.006343030445800094 | 0.11126020207078736 | 0.08574017385551826, 0.17287390261551305, 0.1343919545022519, 0.0520347773098663 | 0.031471, 0.303025, 0.248167, 0.295660 |
| c172x v4 genome (`runs/hdg_after_c172x_s1`) flown on v5 | 0.2397326688652956, 0.05, 0.38523085001510937, 0.07049452717617341, 1.9653025820764143e-05, 0.014608444547277827, 2.242870209249964, 0.0003706638173012445 | 0.22748161325788208 | 0.1474148332864362, 0.377508656233405, 0.2141770857427959, 0.17082587776889113 | 0.421206, 0.618463, 0.471161, 0.593962 |

- **Flag-off path:** with none of the four v5 keys, `phase1_hdg.json` must keep its current costs exactly (genome checks
  the same: `presets/phase1_v4.json` reproduces `hdg_after_c172x_s1` / `hdg_after_t38_s1` / `hdg_after_b737_s1` bit for
  bit, `tests/test_v5.py`).
- **Downdraft alone** (`w_hold` 0, `disturbance_scenario` set) and **hold term alone** are separate checks if a mismatch
  needs bisecting: the downdraft scenario's per-scenario cost without the hold term is the 4th cost above minus
  0.1 × the 4th hold_osc.

## 4. Results on genome/ (for context)

v5 vs v4 on the same seeds (c172x pop 48 × 40, scenario seed = GA seed; jets pop 32 × 20, seed 1). Best costs are under
each genome's own task and are **not comparable across presets**; the cross-evaluation table below is. Hold p-p and the
downdraft residual are measured on the v5 scenario set for both; track_alt is over each task's own scenarios (v5: mean of 4, incl. the downdraft). `v5_report.py` → `runs/v5_report.json`.

| preset | aircraft s | best cost (own task) | track_alt | max pitch calm/worst | nz calm / all | max abs hdg drift | hold p-p calm/worst ft | downdraft residual / max err ft | ki_alt | ki_pitch | at bounds | plateau gen (1 % / last impr.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4 | c172x s1 | 0.1960 | 0.0586 | 4.7 / 5.9 | 0.89-1.10 / 0.44-1.55 | 0.8 | 6.4 / 12.1 | +0.01 / 7.3 | 0.05 | 1.97e-05 | ki_alt@upper(0.05), ki_pitch@near_lower(1.97e-05) | 30 / 35 |
| v4 | c172x s2 | 0.1869 | 0.0547 | 4.7 / 5.9 | 0.89-1.10 / 0.44-1.44 | 0.1 | 5.3 / 15.7 | -0.55 / 7.0 | 0.05 | 0.00332 | ki_alt@upper(0.05), ki_hdg@upper(0.1) | 14 / 26 |
| v4 | c172x s3 | 0.1977 | 0.0709 | 4.9 / 5.1 | 0.89-1.11 / 0.55-1.39 | 0.1 | 11.4 / 11.6 | +4.31 / 9.8 | 5.11e-06 | 7.78e-05 | ki_hdg@near_upper(0.0469) | 18 / 39 |
| v4 | t38 s1 | 0.0966 | 0.0329 | 6.1 / 6.3 | 0.85-1.14 / 0.68-1.22 | 0.0 | 0.3 / 6.4 | +6.85 / 7.9 | 4.16e-07 | 0 | ki_alt@near_lower(4.16e-07), ki_pitch@zeroed(0) | 13 / 19 |
| v4 | b737 s1 | 0.1068 | 0.0322 | 5.3 / 5.4 | 0.84-1.15 / 0.71-1.21 | 0.8 | 0.6 / 7.1 | -0.00 / 7.0 | 0.031 | 0.00561 | ki_alt@near_upper(0.031) | 14 / 18 |
| v5 | c172x s1 | 0.2063 | 0.0457 | 4.8 / 6.0 | 0.88-1.11 / 0.51-1.56 | 0.7 | 3.9 / 9.7 | -0.09 / 5.9 | 0.05 | 1.19e-05 | ki_alt@upper(0.05), ki_pitch@near_lower(1.19e-05) | 22 / 27 |
| v5 | c172x s2 | 0.2164 | 0.0490 | 4.5 / 6.0 | 0.88-1.11 / 0.44-1.45 | 0.4 | 3.7 / 7.6 | +0.84 / 4.3 | 0.00939 | 2.03e-05 | ki_pitch@near_lower(2.03e-05), kp_hdg@near_upper(3.71) | 11 / 25 |
| v5 | c172x s3 | 0.2204 | 0.0606 | 4.5 / 5.7 | 0.88-1.11 / 0.49-1.42 | 0.2 | 3.2 / 8.5 | +3.69 / 4.9 | 1.63e-05 | 0.000379 | ki_hdg@near_upper(0.0444) | 8 / 13 |
| v5 | t38 s1 | 0.0878 | 0.0157 | 6.1 / 6.4 | 0.85-1.14 / 0.68-1.22 | 0.0 | 0.4 / 5.9 | -0.00 / 5.7 | 0.05 | 0.00572 | ki_alt@upper(0.05) | 9 / 16 |
| v5 | b737 s1 | 0.1113 | 0.0279 | 5.4 / 5.4 | 0.84-1.15 / 0.70-1.22 | 0.5 | 0.9 / 7.0 | +0.01 / 6.6 | 0.0278 | 0.002 | ki_alt@near_upper(0.0278) | 15 / 18 |

**c172x gene spread across the 3 seeds** (max/min of nonzero values; log10 std; zeros):

| gene | v4 max/min | v4 log10-std | v4 zeros | v5 max/min | v5 log10-std | v5 zeros |
|---|---|---|---|---|---|---|
| kp_alt | 1.3 | 0.07 | 0 | 1.4 | 0.08 | 0 |
| ki_alt | 9791.5 | 2.30 | 0 | 3063.8 | 1.84 | 0 |
| kd_alt | 2.2 | 0.18 | 0 | 1.9 | 0.14 | 0 |
| kp_pitch | 2.8 | 0.25 | 0 | 1.3 | 0.06 | 0 |
| ki_pitch | 168.7 | 1.15 | 0 | 31.8 | 0.81 | 0 |
| kd_pitch | 1.3 | 0.06 | 0 | 1.7 | 0.12 | 0 |
| kp_hdg | 1.7 | 0.12 | 0 | 1.6 | 0.10 | 0 |
| ki_hdg | 269.8 | 1.32 | 0 | 327.5 | 1.29 | 0 |

**Cross-evaluation** (`v5_cross_eval.py`; same scenario seeds):

| run | v4 task: v4 genome / v5 genome | v4 track_alt: v4 / v5 genome | v5 task: v4 genome / v5 genome |
|---|---|---|---|
| c172x s1 | 0.1960 / 0.1951 | 0.0586 / 0.0493 | 0.2275 / 0.2063 |
| c172x s2 | 0.1869 / 0.2222 | 0.0547 / 0.0510 | 0.2163 / 0.2164 |
| c172x s3 | 0.1977 / 0.2076 | 0.0709 / 0.0528 | 0.2466 / 0.2204 |
| t38 s1 | 0.0966 / 0.0870 | 0.0329 / 0.0191 | 0.1401 / 0.0878 |
| b737 s1 | 0.1068 / 0.1116 | 0.0322 / 0.0336 | 0.1087 / 0.1113 |

**Reading:**
- **Hold oscillation (c172x):** calm hold p-p 6.4 / 5.3 / 11.4 → 3.9 / 3.7 / 3.2 ft, worst scenario 12.1 / 15.7 / 11.6 →
  9.7 / 7.6 / 8.5 ft. The jets were already < 1 ft calm; their worst-scenario p-p (~6–7 ft) is turbulence response.
  (Sim Bridge's ~17 ft includes the capture transient: without the 5 s settle exclusion, v4 s1 gives 10.4 / 18.9 / 14.5 ft.)
- **I-gains:** the downdraft moves ki_alt off zero where it was zero on the jets: T38 4e-7 → 0.05 (its upper bound),
  ki_pitch 0 → 0.0057, downdraft residual 6.85 → 0.00 ft, and the T38 v5 genome is also better on the v4 task (0.0870
  vs 0.0966). The 737 already had ki_alt ≈ 0.03 (near its upper bound) in v4. On the c172x ki_alt is 0.05 / 0.0094 /
  1.6e-5 (s3 still effectively zero, residual 3.7 ft); ki_pitch stays small (1e-5 – 4e-4).
- **Gene spread (c172x, 3 seeds):** ki_alt max/min 9.8e3 → 3.1e3 (log10-std 2.30 → 1.84), ki_pitch 169 → 32
  (1.15 → 0.81), kp_pitch 2.8 → 1.3; ki_hdg unchanged (270 → 328). Narrower, but the surface is still flat in the
  integrator directions.
- **Plateau:** v5 does not fix the stall; c172x reaches 1 % of its final best at gen 22 / 11 / 8 (v4: 30 / 14 / 18).
  On c172x s2 and the 737 the v5 GA ends at or above the v4 genome's v5 cost (0.2164 vs 0.2163; 0.1113 vs 0.1087), so
  part of the spread is search budget, not fitness.
- **ki_alt bound:** ki_alt sits at or near its upper bound in 3 of 5 v5 runs (c172x s1, T38, 737), as in v4 (c172x s1/s2, 737). Widening it would
  change the shared bounds and therefore v4; if the team wants it, do it as a v5-only override.

## 5. Recommendation

Keep v4 as the shared default for now; run v5 alongside. v5 does what it was asked to do (damped hold on the c172x, a
working altitude integrator on the T38, no loss on max pitch / g / heading), but it does not narrow the gene spread
much, does not fix the early plateau, pins ki_alt at its bound on 3 of 5 runs, and isn't implemented in evolution/ yet.
Suggested gate for making it the default: ki_alt bound decision (v5-only), a seeded or larger-budget rerun showing
the GA beats the v4 genomes on the v5 task, your bit-for-bit check in section 3, and team sign-off.
