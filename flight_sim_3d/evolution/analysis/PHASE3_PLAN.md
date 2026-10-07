# Phase 3 gated plan (Python / JSBSim / FD path)

**For:** Corleone. **Date:** 2026-10-06 ~13:55 PT (P2.5 approval noted ~13:54 PT). **Update ~17:40 PT: Corleone held
64×60 A1 pilot (`configs/phase3a1_pilot.json`); started P3-B (richer wing-shape genes). A1 smoke remains done; P3-B in
progress with FD + Genome (gene list / shape→mesh API); Evolution waiting on that lock.** **Update ~18:22 PT: P3-B1 wired in Evolution (`full_a1_b1`, genome kind `phase3_b1`); verify + 16x5 smoke done; B1 64×60 pilot ready, NOT launched (needs Corleone); next gate P3-C.** **Update ~19:01 PT: FD B1 r1 pinned (`post_p3b1r1`, frozen copy `evolution/_fd_pin_p3b1r1`) and smoked (`phase3b1r1-smoke-s1`); r0 superseded; the 64×60 B1 pilot (r1 pins) is held for tomorrow pending Corleone.** **Update ~21:15 PT: B1 r1 64×60 pilot `phase3b1r1-pilot-s1` DONE (launched 20:08 PT, exit 0, 3312 s): bests c172x 0.2481 / T38 0.1093 / 737 0.1352; rigid screen rho mostly negative (ladder concern, see `STATUS_P3B1_pilot.md`); tweaked A/B `phase3b1r1-pilot-tweaked-s1` (elite 4, uniform shape crossover) RUNNING, ETA ≈ 22:02 PT.**  
**Authors merge:** Evolution Runner plan + FD mesh/morph sketch + Genome `PHASE3_CHROMOSOME_SKETCH.md`.  
**Do not implement Phase 3 yet.** Phase 2 pins/cache and all `phase2-pilot-*` runs stay untouched. No GitHub push. Architecture v1 Three.js path is out of scope.

Corleone chose **all** of A–D. Staged so they are not one bang: **P2.5 → P3-A1 → P3-B → P3-C → P3-D0** (A2 / D1 later if needed).

---

## 1. Goal (plain language)

Keep evolving controllers + structure on the same JSBSim/FD stack, but let the wing **look and behave more like a real design space**: denser FE (more strips/modes), planform/shape genes beyond today’s EI/GJ/NSM multipliers, evals that force each aircraft through several wing configs, and eventually geometry that **changes during a run**. Structural cost stays **FD `flexeval`** (Phase 2 rule: `struct_v2_source: fd`); controller terms ride on top unchanged.

---

## 2. Staged gates

### P2.5 — close Phase 2 mass/deflection loopholes *(before any P3 code)*

| | |
|---|---|
| **Why** | Pilot piles on `wing_nsm_*` floors and negative `J_mass`; §12 floors stiffness-linked mass but not NSM. FD also wants tip/spanwise BM or tip-deflection allowable + a §12 wording nit. |
| **Owner** | **FD** (gene ranges / sizing / INTERFACE_v2); Genome + Evolution re-pin after. |
| **Work** | **Choice A (approved):** floor NSM at **1.0** (keep NSM genes); add tip allowable; clarify §12 text. Publish new `model_version` strings. |
| **Acceptance** | FD smoke: baseline margins unchanged intent; random genomes no longer free-ride on NSM-only mass credit; new `model_version` strings published. Evolution: new pins only — **no** rewrite of Phase 2 run dirs. |
| **CPU** | Negligible (pre-flight sizing / gene decode). |
| **model_version** | **New** full (and reduced if schema/weights hashed) — Phase 2 cache keys diverge by design. |
| **Status** | **APPROVED** (Corleone choice A @ 13:53 PT). **In progress with FD** — patching NSM floor + tip allowable + §12 wording; will publish new `model_versions`. Genome + Evolution re-pin after FD lands. |

### P3-A1 — denser FE mesh + more modes *(smoke only)*

| | |
|---|---|
| **Owner** | **FD** (mesh/modes); Evolution (pins, smoke config); Sim Bridge (node layout if FE node count changes). |
| **Work** | **A1:** 48–64 strips; modes **4b + 3t + 2ip** (vs today’s ~32 strips / 3b+2t+1ip). Truncation gate stays **2%**. Same **12** struct genes. Flag behind fidelity/version so Phase 2 path remains. **A2 fallback** (later): 64–96 strips / 5b+3t+2ip if truncation fails. |
| **Acceptance** | Truncation check ≤ 2%; baseline hand-calcs / OAS-class checks still pass at A1; Evolution smoke (small pop×gens) on c172x+T38+737 with pinned A1 `model_version`; traj export still validates (`ga-flightsim-traj/2` + flex-state). |
| **CPU** | FD estimate **~1.4–1.8×** current full (A2 **~2–3×**). **Measure after FD prototype** — do not budget GA on these ratios alone. |
| **model_version** | **New** `full:flexv2:…` (gene count unchanged; mesh/modes hashed). Reduced may stay flexv1 unless FD changes projection. |
| **Evolution Runner** | New smoke config + pins only; genome schema **unchanged**; scenarios unchanged; traj: denser `nodes` if FD emits them — Sim Bridge/ER map must tolerate longer node lists. |
| **As delivered (FD, INTERFACE_v2 §13)** | Separate opt-in fidelity **`full_a1`** (`flexeval_a1` / `FlexBodyModelA1`): 64 strips, 4b+3t+2ip per semi-wing, tails/fuselage = full, 31 DOF; `J_wing_tip_bm_limit` station-exact at η 0.875. Pins `full_a1:flexv2a1:<sha8>` in `model_versions_post_p3a1.json` (full strings unchanged = post_p25). FD CPU/scenario full → A1: c172x 1.66→1.77, T38 1.57→1.68, 737 1.54→1.68, f16 1.63→1.76 (1.07–1.09×). |
| **Status** | **WIRED (Evolution, ~15:00 PT).** Tip-verify: full bit-identical to P2.5; A1 tip-soft J_tip 0.01187, baseline 0. Smoke `phase3a1-smoke-s1` (16x5, exit 0, 168.3 s): bests c172x 0.3234 / T38 0.2262 / 737 0.2565, structural sums ≥ 0, no stiffness genes at floor, nsm piles at 1.0 as in P2.5; measured CPU/scenario 1.62 / 1.84 / 1.81 s (P2.5 full smoke 1.51 / 1.68 / 1.65). Traj `/2` valid with 65-node wings. **64×60 A1 pilot (`configs/phase3a1_pilot.json`, rigid→full_a1) Held by Corleone (~17:40 PT) — not launched.** Details: `STATUS_P3.md`. |

### P3-B — richer wing planform / shape genes

| | |
|---|---|
| **Owner** | **FD** (geometry → aero/mass increments + mesh rebuild); **Genome** (shape block, operators); Evolution (schema, init, ladder). |
| **Work** | Genome **B1** first: spanwise CPs for chord / twist / sweep-Δ / optional dihedral-Δ; root + per-segment ratios (no sawtooth); symmetry on. FD: planform as **aero + mass increments on fixed JSBSim tables**; EI/GJ remain **multipliers on a geometry-derived baseline**. Gene count: Genome ~**12–18** shape; FD ballpark **+6–10** — **PENDING** exact list after FD prototype. Decode: shape → mesh/aero → structure genes → margins + flight. Cheap geometry gate (negative chord / self-intersect) before flight. |
| **Acceptance** | Geometry gate rejects illegal shapes; baseline shape + structure=1.0 matches A1 bit-intent; smoke GA moves shape genes without mass/stiffness exploits; per-block crossover (controller \| structure \| shape). |
| **CPU** | Mesh rebuild each genome — **measure after FD prototype**. Ladder: screen on baseline-shape or rigid still OK if Spearman stays useful. |
| **model_version** | **New** (gene schema + geometry path in hash). |
| **Evolution Runner** | Genome schema adds `shape` / planform block; configs `struct_genes` + `shape_genes`; init baseline-seed shape; traj: optional planform metadata for Sim Bridge (PENDING fields). |
| **Status** | **B1 r1 PINNED + SMOKED (Evolution, ~19:01 PT); r0 superseded.** FD B1 r1 (frozen 18:47 PT) pins `post_p3b1r1` (c172x 56ee798e / T38 7e871977 / 737 6523753c / f16 617078a9); frozen FD `evolution/_fd_pin_p3b1r1` (md5 13/13 OK). Node layout = FD `node_layout_b1` (EA follows shaped chord / sweep / AC shift). Baseline = A1 bit for bit on r1. Smoke `phase3b1r1-smoke-s1` (16x5, exit 0, 297.1 s, loaded box): bests c172x 0.3830 / T38 0.1571 / 737 0.2721 (r0 0.3861 / 0.1562 / 0.2706; A1 0.3234 / 0.2262 / 0.2565); Σ J_* ≥ 0; no stiffness at floor; 0 gate rejects. T38 twist_mid still 69 % at +1° but neutral (baseline-shape re-fly 0.15714 vs 0.15714, shape +7.8e-6). Traj valid, replay bit for bit through the frozen dir. **64×60 B1 pilot `phase3b1r1-pilot-s1` DONE (~21:04 PT, 3312 s, snapshot `/workspace/er_pilot_code_b1r1`, fd_dir `_fd_pin_p3b1r1`):** bests c172x 0.248063 / T38 0.109280 / 737 0.135173 (Phase 2 seeds 0.2154 / 0.0958 / 0.1235, not like-for-like); Σ J_* ≥ 0; 0 gate rejects; shape worth −0.0075 / −0.0079 / −0.0007 at the best; piles c172x twist_mid −2° (84 %), 737 sweep −5° (67 %). Ladder concern: rigid→full_a1_b1 rho mean −0.13 / −0.13 / −0.16 (70–77 % of gens < 0). Tweaked A/B `phase3b1r1-pilot-tweaked-s1` RUNNING (21:06 PT). Details: `STATUS_P3B1_pilot.md`. Earlier r0 line: see `STATUS_P3.md`. |

### P3-C — multi-config eval scenarios

| | |
|---|---|
| **Owner** | **Evolution** (scenario set, aggregate, ladder); Genome (preset profile); FD (which knobs are first-class); Sim Bridge (per-config traj tags). |
| **Work** | Configs are **scenario tags**, not genes. FD sketch: **clean / flaps / fuel / pullup-or-roll**. Genome: mean aggregate; **one gate fail ⇒ individual fails**. Start **K=2–3**. Screen / ladder on **config-0** only; promote elites to full multi-config. |
| **Acceptance** | Same genome gets distinct costs/terms per config; logged per-config `J_*`; fail policy matches Phase 2 hard gate; smoke with K=2 then K=3. |
| **CPU** | **×K** full evals (plus margin screens). Mitigate: screen config-0; cache per `(genome, config, model_version)`. |
| **model_version** | Unchanged unless FD adds config-dependent structure; pin set may grow per-config if roots differ (**PENDING**). |
| **Evolution Runner** | New scenario set `phase3_multiconfig`; traj schema: `config_id` (and overrides) on each exported traj / index entry. |

### P3-D0 — discrete morph segments *(during a run)*

| | |
|---|---|
| **Owner** | **FD** (DOF list, rate limits, margin sampling); Genome (`morph_schedule` block); Evolution (time-varying shape hook); Sim Bridge (morph channels). |
| **Work** | **D0 first:** discrete morph segments / open-loop pose schedule (Genome D1: 2–4 waypoints, 1–2 DOFs, rate-limited). Margins = **min over sampled poses** (endpoints + mid), not averaged into reward above 1.2. Continuous morph (**D1** in FD naming) later — FD warns **~5–20×** CPU. |
| **Acceptance** | Pose changes mid-scenario visibly in traj/nodes; rate limits enforced; min-margin gate behaves; smoke with morph flag off ≡ B/C bit-identical. |
| **CPU** | D0: ×(pose samples) for margins + one continuous flight — **measure after FD prototype**. D1 continuous: defer. |
| **model_version** | **New** when morph DOFs / coupler enter the hash. |
| **Evolution Runner** | Genome +4–8 morph genes; scenario events or normalized-time waypoints; traj: morph command + active pose channels (**PENDING** exact names with Sim Bridge). |

---

## 3. Evolution Runner checklist (per gate)

| Gate | configs | genome schema | scenarios | traj / cache |
|---|---|---|---|---|
| P2.5 | re-pin only; optional `phase2_flex` follow-on after FD publishes new `model_versions` | NSM floor at 1.0 (Genome preset; keep NSM genes) | unchanged | new pins ⇒ new cache namespace; **do not mutate** old Phase 2 run dirs |
| P3-A1 | `phase3a1_smoke` ✅ ran; `phase3a1_pilot` **Held** (Corleone ~17:40 PT) | unchanged (12 struct) | Phase 2 set | 65-node wings validate; A1 pins; full/full_a1 cache disjoint |
| P3-B1 | r1: `phase3b1_smoke` ✅ ran (`phase3b1r1-smoke-s1`); `phase3b1_pilot` ✅ ran (`phase3b1r1-pilot-s1`); `phase3b1_pilot_tweaked` A/B running (`phase3b1r1-pilot-tweaked-s1`); r0 `*_r0.json` superseded | + 6-gene shape block (`phase3_b1`) | single config | optional `planform` header (`fd-planform/1`); post_p3b1 pins; shape in cache key |
| P3-C | `phase3_c_*` | unchanged vs B | multi-config set; aggregate mean; fail-on-any-gate | `config_id` on traj/index; cache key includes config |
| P3-D0 | `phase3_d0_smoke` | + morph_schedule | morph-enabled scenarios | morph channels; morph off = prior bit-identical |

---

## 4. Open questions (PENDING)

1. **P2.5 — RESOLVED:** Corleone approved choice A @ 13:53 PT (NSM floor at 1.0; tip allowable; §12 wording). FD instructed to patch and publish new `model_versions`. Genome + Evolution re-pin when FD lands.
2. **FD — RESOLVED:** A1 = 64 strips, 4b+3t+2ip; truncation ≤ 1.03 % (< 2 %), so no A2 needed for now.
3. **FD + Genome:** final planform gene list (FD 6–10 vs Genome 12–18 B1) and which params denser mesh accepts first (chord / twist / sweep / thickness?). **RESOLVED for B1:** FD's 6 genes (`SHAPE_GENES_B1`) wired ~18:22 PT. Open: Genome wants chord genes log-scale while FD's schema says linear (log applied in the operator only); T38 twist_mid ceiling pile. Pilot (~21:04 PT): T38 twist_mid no longer piled (14 % at +1); NEW piles c172x twist_mid at −2° (84 %) and 737 sweep_qc at −5° (67 %) — FD to check. NEW: rigid screen rank-anticorrelated with full_a1_b1 among promoted genomes (STATUS_P3B1_pilot.md §3).
4. **FD:** morph DOFs for D0; does stiffness follow morph or aero-only OML?
5. **FD:** first-class C knobs (flap ≠ morph) and whether margins are per-config.
6. **Evolution / Corleone:** confirm K=3 mean aggregate and one-gate-fail ⇒ individual fail for C.
7. **Sim Bridge:** traj fields for planform metadata, `config_id`, morph channels — extend `/2` vs new schema bump.

---

## 5. Non-goals

- Do **not** break Phase 2 `pin_model_version` / EvalCache for existing runs; new versions get new pins.
- Do **not** touch running or finished `phase2-pilot-s*` (or other Phase 2 run dirs).
- Do **not** implement Phase 3 code in this planning pass (P3-A1 wiring since done, opt-in only).
- Do **not** push to GitHub.
- Do **not** invent a parallel structural cost — FD `flexeval` only.
- Architecture v1 Three.js path stays separate.
- B2 CST airfoils, B3 free-vertex morph, D1 continuous morph, NSGA multi-objective: later options only.

---

## 6. Suggested sequence (owners)

```
Corleone: P2.5 APPROVED (choice A @ 13:53 PT)
    → FD P2.5 patch in progress (NSM floor + tip allowable + §12) + new model_versions
    → Genome preset + Evolution re-pin / optional Phase 2 follow-on smoke
    → FD A1 prototype (measure CPU) → Evolution A1 smoke   [DONE 2026-10-06 ~15:00 PT]
    → Corleone: 64×60 A1 pilot HELD (~17:40 PT); start P3-B
    → FD+Genome B1 gene lock + shape→mesh API on full_a1   [DONE]
    → Evolution B1 wiring + verify + smoke   [DONE ~18:22 PT]
    → FD B1 r1 frozen → Evolution re-pin + frozen FD copy + r1 smoke   [DONE ~19:01 PT; r0 superseded]
    → Corleone: B1 64×60 pilot, r1 pins   [DONE ~21:04 PT: phase3b1r1-pilot-s1]
    → tweaked A/B (elite 4, uniform shape crossover)   [RUNNING from 21:06 PT, ETA ~22:02 PT]
    → ladder review: rigid screen rho < 0 (full-only vs B1-aware screen; screen-vs-random probe after the A/B)
    → next gate: P3-C
    → Evolution+Genome C multiconfig (K=2→3)
    → FD D0 DOFs → Evolution D0 smoke
```

## P3-B2a (2026-10-06 ~22:50 PT)
Wired, not launched; see STATUS_P3B2a.md. A/B tweaked B1: c172x 0.240135 (-3.20%), T38 0.109360 (+0.07%), 737 0.140783 (+4.15%) vs baseline; screen probe g40 done (phase3b1r1_screen_vs_random_g40.json).
