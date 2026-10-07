# Phase 3 chromosome / fitness sketch (Genome Architect)

**Scope:** Corleone wants Phase 3 covering (A) denser mesh, (B) richer wing-shape genes, (C) multi-config wing scenarios, (D) morphing/mobile geometry during a run. Phase 2 pilots stay as-is. This note covers **B, C, D** for Evolution Runner to merge with FD's mesh notes (A). Nothing here is implemented yet.

**Update 2026-10-06 ~17:50 PT:** Corleone held the A1 64×60 pilot and chose **B1 next** (P3-B planform). Implementation-ready genome-side spec (decode order, FD alignment, geometry-gate reject, per-block ops, OPEN on FD): **`PHASE3_B1_SPEC.md`**. Do not treat the B1 ballpark counts below as a locked gene list — wait for FD.
**Update ~18:40 PT:** FD LOCKED B1 at 6 genes (`planform_b1.py`, INTERFACE_v2 §14); implemented as opt-in preset `phase3_b1` (26 genes, `full_a1_b1`). See `PHASE3_B1_SPEC.md` / `CROSSCHECK_p3b1.md`.

**Hard rule:** FD's flexeval structural cost (18 `J_*` terms, FD weights) stays the structural part of the scalar. Controller terms ride on top exactly as Phase 2 (`track_alt`, `effort`, `comfort`, `heading`). Do not invent a parallel structural formula.

---

## B — richer wing-shape genes (static geometry)

Shape is a **separate gene block** from the Phase 2 structure block (stiffness / NSM / damping). Decode order: shape → FD mesh/aero rebuild (or FD morph of baseline mesh) → structure genes applied on the new geometry → margins + flight.

### Option B1 — planform control points (recommended for first gate)
- Genes: spanwise stations (4–6) for **chord**, **twist (deg)**, **sweep delta**, optional **dihedral delta**. Encode as multiplicative scales or deltas about the baseline planform, not absolute CFD coords.
- Smoothness by construction: same trick as v2 `wing_ei_taper_*` — store **root value + per-segment ratios** (or cubic-Bernstein weights with fixed knot count), so the GA cannot sawtooth.
- Symmetry default on (mirror L→R). Asymmetry waits for lateral scenarios, same policy as Phase 2.
- Bounds: keep designs flyable and meshable; reject self-intersect / negative chord / extreme taper before flight (cheap geometry gate, not a fitness credit).
- Gene count ballpark (sketch only): Genome once said ~12–18; **ER/FD expect 6–10** — when FD publishes N, genome takes exactly N (see `PHASE3_B1_SPEC.md`). + existing 12 structure + 8 controller.

### Option B2 — CST / class-shape airfoil coefficients
- Fewer genes (~8–12) for thickness and camber modes at a few span stations, plus a coarse planform (AR, taper, sweep).
- Better for section shape; worse for planform story Corleone asked for. Pair with B1 later if FD's denser mesh can resolve sections.

### Option B3 — free vertex / dense CP morph
- Matches early GA Flight Sim mesh-CP idea; high dimensional, needs Laplacian/smooth mutation and repair.
- Defer until denser mesh (A) and FD say the aero/flex pipeline can consume it. Highest exploit risk.

**Recommendation:** gate **B1** first. Keep structure genes as today; shape only changes the baseline the structure genes scale. Operators: per-block crossover (controller | structure | shape); shape mutation with neighbor smoothing / ratio representation.

**Open for FD:** which parameters their denser mesh / strip model actually accepts (chord, twist, sweep, thickness?) and whether shape rebuild is cheap enough for gen-0 screening.

---

## C — multi-config wing scenarios (fitness profile, not chromosome)

"Config" here means a **scenario tag**, not a gene: e.g. cruise planform, high-lift / flaps-down, fuel/payload mass, CG shift, Mach/altitude band, optional locked morph pose if D exists.

### How it shows up without breaking flexeval
1. **Scenario set expands** (like v5's downdraft): each individual is evaluated on configs `{k=1..K}`.
2. **Per config `k`:**
   - Apply config overrides (mass, CG, flap/morph pose, atmosphere) **outside** flexeval.
   - Call FD margins + flexeval on that geometry/state → get the 18 structural `J_*` (and gate).
   - Run the flight → controller terms as today.
   - Per-config scalar: `J_controller(k) + J_structural_flexeval(k)` (structural weight 1.0, same as Phase 2 `struct_v2_source: fd`).
3. **Aggregate across configs** with an explicit policy in the preset:
   - **default:** `mean_k` (matches Phase 2 mean-over-scenarios);
   - **optional:** `mean + λ * max` or CVaR-worst-m, if Corleone wants robustness over average.
4. **Never** fold config identity into the chromosome. Never reweight FD's internal terms. If a config cannot fly (gate fail), same hard cost as Phase 2 (2000 / not flown), and decide a priori whether one failed config fails the individual or only that slot (recommend: **one gate fail ⇒ individual fails**, same as a single bad Phase 2 scenario under strict gate).

### Profile sketch
```json
"scenarios": {
  "set": "phase3_multiconfig",
  "configs": [
    {"id": "cruise", "mass_scale": 1.0, "flap": 0.0},
    {"id": "heavy", "mass_scale": 1.15, "cg_shift_mac": 0.05},
    {"id": "high_lift", "flap": 0.3, "v_approach_kcas": 70}
  ],
  "aggregate": "mean"
}
```
Weights stay on **named controller objectives**; structural stays "whatever FD returned for that config." Log per-config terms for Sim Bridge replays.

**Open for FD/Sim Bridge:** which config knobs are first-class (flap ≠ morph), and cost of K full-fidelity evals (budget gate: start K=2–3).

---

## D — morphing / mobile geometry during a run

Morph is **time-varying shape**, so the chromosome must separate **what can morph** from **when / how it commands**.

### Option D1 — pose schedule (open-loop morph) — recommended first gate
- Genes: morph amplitudes at 2–4 waypoints in normalized scenario time (or tied to scenario events: "after capture", "on climb"). Interpolate with clamped rate limits (deg/s or % chord/s) so the schedule is physically paced.
- Actuated DOFs (start tiny): e.g. outboard twist ±δ, or trailing-edge morph ±δ — **FD must define the DOF list**.
- Controller genes unchanged; morph schedule is a parallel open-loop channel (same spirit as Phase 1 open-loop throttle).
- Gene count: ~4–8 morph genes.

### Option D2 — morph as an extra control loop
- Genes: gains from altitude/airspeed/load error → morph command (plus rate limits).
- Stronger, but couples into handling qualities and needs FD's actuated-plant linearization. Defer until D1 proves the pipeline.

### Option D3 — morph waypoints as Bezier / spline CPs in time
- Like D1 with smoother schedules; more genes and more smoothness constraints. Use if D1 schedules chatter.

**Coupling rules (fitness):**
- Geometry at time `t` is baseline shape (B) ⊕ morph(t) (D). Structure genes still scale the **current** structural properties if FD says stiffness rides with morph; otherwise document "morph is aero-only."
- flexeval / margins: **minimum over morph poses sampled in the scenario** (endpoints + mid), or FD's continuous bound if they provide one. Do not average margins into a reward above 1.2 (same Phase 2 shaping).
- Structural `J_*` still come from FD for the pose(s) they define; we do not invent morph-specific structural terms.

**Recommendation:** D1 with 1–2 DOFs, rate-limited, margins = min over a small pose set. Keep Phase 2 structure block; add `morph_schedule` block behind a preset flag.

---

## Suggested gates for Corleone (merge with FD mesh plan)

| Gate | Deliverable | Genome impact | Eval cost |
|---|---|---|---|
| **P3.0** | FD denser mesh (A) behind a flag; Phase 2 genes unchanged | 0 | smoke only |
| **P3.1** | B1 planform genes + geometry gate; single config | +12–18 genes | ~same as P2 × mesh rebuild |
| **P3.2** | C multi-config K=2–3; mean aggregate; flexeval untouched | profile only | ×K |
| **P3.3** | D1 morph schedule 1–2 DOF; margin min-over-poses | +4–8 genes | × poses for margins |
| **P3.4** | Optional NSGA on controller vs structural vs morph-effort | operators | same budget as P2 recommendation |

**Non-negotiables:** Phase 2 pilots frozen; `struct_v2_source: fd` remains default for structural; legacy / v4 / v5 / phase1_flex bit-identity untouched; no push without Corleone.

**Questions back to ER/FD:**
1. FD: which shape params does the denser mesh accept first (chord/twist/sweep)?
2. FD: morph DOFs and whether stiffness follows morph or only outer mould line.
3. ER: OK with K=3 mean aggregate for C, and one gate-fail ⇒ individual fail?
4. Corleone: prioritize B before D, or a thin D1 demo earlier for the "mobile geometry" headline?
