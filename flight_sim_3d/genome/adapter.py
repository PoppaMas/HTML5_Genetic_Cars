"""Adapter: plug a TaskConfig into the *unmodified* flight_sim/evolve.py and ga.py.

evolve.py does ``import genome, sim, ga`` and uses only:
    genome.N_GENES, genome.GENE_NAMES, genome.SCHEMA[i].name/.units, genome.decode(g)
    sim.make_scenarios(n, seed), sim.evaluate(gains, scenarios) -> {"cost", "per_scenario"[...status...]},
    sim.AIRCRAFT; plot_results.py additionally uses sim.simulate(gains, sc, record=True)["trace"].
``install(task)`` registers shim modules under those names in sys.modules
*before* evolve.py is imported, so evolve.py/ga.py run byte-for-byte unchanged.
Workers inherit the shims through fork (Linux default; run_evolve.py forces it).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fitness as F  # noqa: E402
import genome_schema as GS  # noqa: E402
import profiles as P  # noqa: E402
import scenarios as SC  # noqa: E402
from flightsim_path import orig_sim  # noqa: E402

PRESET_DIR = os.path.join(HERE, "presets")

# Genes each backend actually feeds into dynamics (anything else is inert).
BACKEND_CONSUMES = {
    "legacy_sim": {"kp_alt", "ki_alt", "kd_alt", "kp_pitch", "ki_pitch", "kd_pitch"},
}


def backend_capabilities(name: str) -> Dict:
    if name == "legacy_sim":
        return {"name": name, "consumes": sorted(BACKEND_CONSUMES[name]), "channels": []}
    import sim_ext
    return sim_ext.CAPABILITIES


@dataclass
class Task:
    name: str
    spec: GS.GenomeSpec
    fitness: F.FitnessConfig
    scenarios: Dict
    profile: P.AircraftProfile
    raw: Dict
    warnings: List[str] = field(default_factory=list)
    conditions: Dict = field(default_factory=dict)   # scenarios.apply_conditions input ({} = legacy)
    sim_fixed: Dict = field(default_factory=dict)    # disabled-block genes the backend should still use
    flex_constants: Dict = field(default_factory=dict)  # fixed flex inputs in use (reporting only; FD applies its own)
    disturbance: Dict = field(default_factory=dict)     # v5: extra sustained-downdraft scenario ({} = none)
    init: Dict = field(default_factory=dict)            # Phase 2: generation-0 seeding ({} = evolve.py's uniform draw)

    def make_scenarios(self, n: int, seed: int):
        scs = SC.apply_conditions(SC.make_scenarios(n, seed, self.scenarios), self.conditions)
        if self.disturbance:
            scs = list(scs) + [make_disturbance_scenario(scs[0], self.disturbance)]
        return scs

    def evaluate(self, gains, scenarios, record=None):
        if self.sim_fixed:
            gains = {**self.sim_fixed, **gains}
        return F.evaluate(gains, scenarios, self.fitness, record=record)


def load_task(name_or_path: str, overrides: Optional[Dict] = None) -> Task:
    path = name_or_path if os.path.exists(name_or_path) else os.path.join(PRESET_DIR, f"{name_or_path}.json")
    with open(path) as f:
        raw = json.load(f)
    raw = _deep_merge(raw, overrides or {})
    return build_task(raw)


def apply_shared(raw: Dict) -> Dict:
    """Fill a preset's task definition from the shared Phase-1 set (aircraft_profiles/phase1_shared.json).

    Preset keys win; fitness.weights is replaced as a whole (so a preset can drop comfort). Adds the
    "shared_task_resolved" block that build_task turns into scenario conditions (steps, duration, error scale).
    """
    if not raw.get("shared"):
        return raw
    if raw["shared"] != "phase1":
        raise ValueError(f"unknown shared set {raw['shared']!r} (only 'phase1')")
    t = P.load_shared()["task"]
    base = {"fitness": {"weights": dict(t["fitness_weights"]), "params": dict(t["comfort_params"])},
            "scenarios": {"ramp_fpm": t["ramp_fpm"], "ramp_accel_g": t["ramp_accel_g"]},
            "controller": {"alt_ref_ff": t["alt_ref_ff"]}}
    out = _deep_merge(base, raw)
    if "weights" in raw.get("fitness", {}):
        out["fitness"]["weights"] = dict(raw["fitness"]["weights"])
    out["shared_task_resolved"] = {"steps_rel_ft": [tuple(x) for x in t["steps_rel_ft"]], "duration_s": t["duration_s"],
                           "alt_err_scale_ft": t["alt_err_scale_ft"], "max_alt_err_ft": t["max_alt_err_ft"]}
    return out


def make_disturbance_scenario(calm, dist: Dict):
    """v5: the calm scenario, holding its start altitude, with a sustained downdraft (1-cos onset). Appended last."""
    import dataclasses
    h0 = calm.steps[0][1]
    return dataclasses.replace(calm, label="downdraft", steps=[(float(t), h0 + float(dh)) for t, dh in dist["steps_rel_ft"]],
                               wind_north_fps=0.0, wind_east_fps=0.0, gust_sigma_fps=0.0, discrete_gust_fps=0.0,
                               draft_fps=float(dist["downdraft_fps"]), draft_t_s=float(dist["onset_t_s"]),
                               draft_ramp_s=float(dist["onset_ramp_s"]))


def resolve_v5(raw: Dict, prof: P.AircraftProfile):
    """(hold-quality settings or None, disturbance settings or None). Off unless the preset (or the shared set's
    enabled_by_default) turns them on; only presets on the shared Phase-1 set can."""
    shared = bool(raw.get("shared"))
    v5 = P.load_shared()["task"].get("v5", {}) if shared else {}
    out = []
    for key, sect in (("hold_quality", "hold_quality"), ("disturbance_scenario", "disturbance")):
        flag = raw.get(key)
        if flag is not None and not isinstance(flag, bool):
            raise ValueError(f"{key} must be true/false, got {flag!r}")
        on = flag if flag is not None else bool(v5.get("enabled_by_default", False))
        if on and not shared:
            raise ValueError(f"{key} needs the shared Phase-1 set (\"shared\": \"phase1\")")
        if not on:
            out.append(None)
            continue
        s = {k: v for k, v in v5[sect].items() if not k.startswith("_")}
        if sect == "disturbance":
            if "downdraft_fps" not in prof.disturbance:
                raise ValueError(f"disturbance_scenario: no downdraft_fps for {prof.name} in phase1_shared.json")
            s["downdraft_fps"] = float(prof.disturbance["downdraft_fps"])
        out.append(s)
    return tuple(out)


def resolve_heading_hold(raw: Dict, prof: P.AircraftProfile) -> Optional[Dict]:
    """Heading-hold settings if the task flies with it, else None.

    Default: on for presets that use the shared Phase-1 set when the aircraft's shared entry has
    heading_hold.enabled (all Phase-1 aircraft); never for tasks without "shared" (the legacy preset).
    A preset can force it with "heading_hold": true/false.
    """
    flag = raw.get("heading_hold")
    if flag is not None and not isinstance(flag, bool):
        raise ValueError(f"heading_hold must be true/false, got {flag!r}")
    shared = bool(raw.get("shared"))
    ac = prof.heading_hold or {}
    on = flag if flag is not None else (shared and bool(ac.get("enabled")))
    if not on:
        return None
    if not shared or "bank_limit_deg" not in ac:
        raise ValueError(f"heading_hold needs the shared Phase-1 set and a bank limit for {prof.name} (phase1_shared.json)")
    t = {k: v for k, v in P.load_shared()["task"]["heading_hold"].items() if not k.startswith("_")}
    return {**t, "bank_limit_deg": float(ac["bank_limit_deg"])}


def _deep_merge(a: Dict, b: Dict) -> Dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_task(raw: Dict) -> Task:
    if isinstance(raw.get("blocks"), dict) and raw["blocks"].get("shape_b2"):
        import shape_b2  # P3-B2a (opt-in): FD's B2a shape genes + full_a1_b2a; builds the rest through this function
        return shape_b2.build_task_b2(raw, build_task)
    if isinstance(raw.get("blocks"), dict) and raw["blocks"].get("shape_b1"):
        import shape_b1  # P3-B1 (opt-in): FD's 6 shape genes + full_a1_b1; builds the rest through this function
        return shape_b1.build_task_b1(raw, build_task)
    raw = apply_shared(raw)
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    prof = P.load_profile(raw.get("aircraft", P.REFERENCE_PROFILE))
    warnings = list(prof.check())
    # blocks: {name: true} evolves the whole block, {name: [genes]} only those genes (the rest held at defaults)
    blk = dict(raw.get("blocks", {"pitch_altitude": True}))
    hh = resolve_heading_hold(raw, prof)
    if hh and not (blk.get("roll_heading") is True):
        sub = list(blk["roll_heading"]) if isinstance(blk.get("roll_heading"), list) else []
        blk["roll_heading"] = sub + [g for g in GS.HEADING_HOLD_GENES if g not in sub]
    import fd_bridge
    # Phase 2: FD flex v2. The structure_v2 genes come from FD's flexbody.gene_schema() (the source of truth).
    flex_raw = dict(raw.get("flex", {}))
    flex_v2 = bool(flex_raw.get("enabled", False)) and int(flex_raw.get("version", 1)) == 2
    asym_v2 = bool(flex_raw.get("asymmetric", False))
    if "version" in flex_raw and int(flex_raw["version"]) not in (1, 2):
        raise ValueError(f"flex.version must be 1 or 2, got {flex_raw['version']!r}")
    if "asymmetric" in flex_raw and not flex_v2:
        raise ValueError("flex.asymmetric needs flex v2 (flex.version = 2)")
    extra_genes = None
    if blk.pop(GS.STRUCTURE_V2_BLOCK, False):
        if not flex_v2:
            raise ValueError("the structure_v2 block needs flex v2 (flex: {enabled: true, version: 2})")
        if blk.get("structure"):
            raise ValueError("structure (flex v1) and structure_v2 (flex v2) cannot be evolved together")
        extra_genes = GS.genes_from_fd_schema(fd_bridge.gene_schema_v2(asym_v2))
        clash = sorted(set(raw.get("gene_overrides", {})) & {g.name for g in extra_genes})
        if clash:
            raise ValueError(f"gene_overrides {clash}: flex v2 gene ranges come from FD's gene_schema(), not the task")
    elif flex_v2:
        warnings.append("flex v2 without the structure_v2 block: every structure gene flies at FD's baseline")
    if raw.get("fitness", {}).get("params", {}).get("struct_v2_mass_credit_clip") and not flex_v2:
        raise ValueError("fitness.params.struct_v2_mass_credit_clip needs flex v2 (flex.version = 2)")
    if "struct_v2_source" in raw.get("fitness", {}).get("params", {}) and not flex_v2:
        raise ValueError("fitness.params.struct_v2_source needs flex v2 (flex.version = 2)")
    blocks = [b for b, on in blk.items() if on]
    subsets = {b: v for b, v in blk.items() if isinstance(v, list) and v}
    fixed_chord = sorted(set(raw.get("gene_overrides", {})) & set(fd_bridge.CHORD_CONSTANTS))
    if fixed_chord:
        raise ValueError(f"{fixed_chord} are not genes in Phase 1 (fixed per aircraft, FD decision 2026-10-06); "
                         "set them with the task's flex.elastic_axis_frac / flex.section_cg_frac instead")
    overrides = {k: {"default": v} for k, v in prof.gene_defaults.items()}
    if not raw.get("legacy_pitch_ranges", False):  # shared Phase-1 gain bounds (never for the legacy task)
        for k, v in prof.shared_gain_bounds.items():
            overrides[k] = {**overrides.get(k, {}), **v}
    for k, v in {**prof.gain_overrides, **raw.get("gene_overrides", {})}.items():
        overrides[k] = {**overrides.get(k, {}), **v}
    spec = GS.build_spec(blocks, P.range_factors(prof), overrides, legacy_pitch_ranges=raw.get("legacy_pitch_ranges", False),
                         gene_subsets=subsets, extra_genes=extra_genes)
    fd = raw.get("fitness", {})
    params = {"n_limits": tuple(prof.n_limits), **fd.get("params", {})}
    weights = dict(fd.get("weights", {"track_alt": 1.0, "effort": 2.0}))
    if hh:  # heading-error cost term (a preset may set its own weight, including 0)
        weights.setdefault(hh["cost_term"], hh["weight"])
        params.setdefault("hdg_rms_ref_deg", hh["hdg_rms_ref_deg"])
    hq, dist = resolve_v5(raw, prof)
    if hq:  # v5 hold-quality term
        weights.setdefault(hq["cost_term"], hq["weight"])
        params.setdefault("hold_ref_ft", hq["hold_ref_ft"])
        params.setdefault("hold_settle_s", hq["hold_settle_s"])
    fcfg = F.FitnessConfig(mode=fd.get("mode", "scalar"),
                           weights=weights,
                           aggregate=fd.get("aggregate", {"mode": "mean"}),
                           params=params,
                           pareto_objectives=fd.get("pareto_objectives", ["track_alt", "effort", "comfort"]),
                           backend=raw.get("backend", "jsbsim_ext"))
    caps = dict(backend_capabilities(fcfg.backend))
    flex = dict(raw.get("flex", {}))
    flex_on = bool(flex.get("enabled", False))
    if flex_on:
        import fd_bridge
        if fcfg.backend != "jsbsim_ext":
            raise ValueError("flex mode needs the jsbsim_ext backend")
        if flex_v2:  # FD flexbody consumes the v2 structure genes (validated by FD's decode_genome_v2)
            fd_bridge.require_root_v2(prof.jsbsim_model)
            caps["consumes"] = list(caps["consumes"]) + list(fd_bridge.gene_names_v2(asym_v2))
        else:
            fd_bridge.require_root(prof.jsbsim_model)
            # with flex on, FD's coupled_sim consumes the structure genes -> the inert-block guard is lifted for them
            caps["consumes"] = list(caps["consumes"]) + list(fd_bridge.FLEX_GENES)
            caps["channels"] = list(caps["channels"]) + ["wing_root_bending", "tip_deflection", "tip_twist"]
    inert = [g.name for g in spec.genes if g.name not in caps["consumes"]]
    if inert:
        msg = f"genes {inert} are not consumed by backend {caps['name']!r} (no dynamics for them yet)"
        if not raw.get("allow_inert_blocks", False):
            raise ValueError(msg + "; disable those blocks or set allow_inert_blocks=true")
        warnings.append("INERT: " + msg)
    for n, why in F.availability(fcfg, caps["channels"]).items():
        warnings.append(f"SKIP objective {n}: {why} from backend {caps['name']!r}")
    if prof.jsbsim_model and prof.jsbsim_model not in caps.get("aircraft_supported", ["c172x"]):
        warnings.append(f"aircraft {prof.jsbsim_model!r}: ranges derived, but the backend has not been checked with this model")
    # Task conditions. The legacy preset sets none (exact sim.Scenario objects, sim.py constants).
    scfg = dict(raw.get("scenarios", {"set": "legacy"}))
    ctrl = dict(raw.get("controller", {}))
    conditions: Dict = {}
    if raw.get("use_profile_conditions", True) and (prof.name != P.REFERENCE_PROFILE or ctrl or "ramp_fpm" in scfg or flex_on):
        conditions.update(prof.task_conditions())
    if "ramp_fpm" in scfg:
        conditions["ramp_fpm"] = scfg.pop("ramp_fpm")
    if raw.get("shared_task_resolved") and conditions:
        conditions.update(raw["shared_task_resolved"])
    if "ramp_accel_g" in scfg:
        acc = scfg.pop("ramp_accel_g")
        if acc is not None:
            if "ramp_fpm" not in conditions:
                raise ValueError("scenarios.ramp_accel_g needs scenarios.ramp_fpm (it smooths the ramp corners)")
            if not acc > 0:
                raise ValueError(f"scenarios.ramp_accel_g must be > 0 g, got {acc}")
            conditions["ramp_accel_g"] = float(acc)
    for k in ("alt_ref_ff", "pitch_cmd_limits_deg", "min_kcas", "nz_limits", "throttle_max"):
        if k in ctrl:
            conditions[k] = ctrl[k]
    if hh:
        if not conditions:
            raise ValueError("heading_hold needs the jsbsim_ext task conditions (a shared Phase-1 preset)")
        conditions["bank_cmd_limit_deg"] = float(hh["bank_limit_deg"])
        conditions["hdg_i_limit_deg"] = float(hh["hdg_i_limit_deg"])
    flex_constants_used = {}
    if flex_on and flex_v2:
        bad = set(flex) - {"enabled", "version", "asymmetric", "substeps", "mode", "model_version_check"}
        if bad & set(fd_bridge.CHORD_CONSTANTS) or bad & {"tip_mass_frac", "x_ea", "x_cg"}:
            raise ValueError(f"flex {sorted(bad)}: elastic axis, section CG and tip mass are fixed per aircraft by FD "
                             "(INTERFACE_v2.md section 6) and cannot be overridden from a task config")
        if bad:
            raise ValueError(f"unknown flex v2 keys {sorted(bad)}")
        if flex.get("mode", "twoway") != "twoway":
            raise ValueError("flex v2 is two-way coupled only (FD FlexBodyCoupler, mode 'twoway')")
        conditions["flex_mode"] = "v2"
        conditions["flex_substeps"] = int(flex.get("substeps", 2))
        conditions["flex_asymmetric"] = asym_v2
        src = F.struct_v2_source(fcfg.params)
        if src == "fd":
            if fcfg.params.get("struct_v2_mass_credit_clip"):
                raise ValueError("struct_v2_mass_credit_clip needs struct_v2_source 'genome' (in 'fd' mode structural_v2 "
                                 "is FD's flexeval cost, J_mass included, unchanged)")
            if fcfg.mode == "scalar" and fcfg.weights.get("structural_v2", 0.0) != 1.0:
                warnings.append("struct_v2_source 'fd' with structural_v2 weight != 1: the cost is no longer flexeval's")
        clip = fcfg.params.get("struct_v2_mass_credit_clip")
        if clip:  # off by default since FD's §12 fix; A/B only
            fd_bridge.mass_term_v2({k: 0.0 for k in ("wingR_lb", "wingL_lb", "ht_lb", "vt_lb", "fus_lb")} |
                                   {"baseline_flexible_lb": 1.0}, 0.0, tuple(clip))  # validates the body names
            warnings.append(f"struct_v2_mass_credit_clip on {list(clip)} (A/B only: superseded by FD's §12 sizing terms "
                            "and minimum-gauge floor)")
        mv = fd_bridge.check_model_version(prof.jsbsim_model, "full", flex.get("model_version_check", "warn"))
        if mv["message"]:
            warnings.append("FD MODEL_VERSION: " + mv["message"])
        if asym_v2:
            warnings.append("flex.asymmetric: FD's asymmetry genes only matter once lateral/roll scenarios exist; "
                            "the current scenario sets have none (keep them off for Phase 2 runs)")
        fd_vals = fd_bridge.fd_chord_defaults(prof.jsbsim_model)
        flex_constants_used = {**fd_vals, "tip_mass_frac": 0.0, "source": "FD flexwing.AIRCRAFT_PROFILES (v2 uses v1's wing values)",
                               "fidelity": "full", "margin_gate": fd_bridge.V2_MARGIN_GATES["full"],
                               "fd_model_version": mv["current"], "fd_model_version_recorded": mv["recorded"]}
    elif flex_on:
        conditions["flex_mode"] = flex.get("mode", "twoway")
        conditions["flex_substeps"] = int(flex.get("substeps", 2))
        bad = set(flex) - {"enabled", "mode", "substeps"}
        if bad & set(fd_bridge.CHORD_CONSTANTS) or bad & {"tip_mass_frac", "x_ea", "x_cg"}:
            raise ValueError(f"flex {sorted(bad)}: elastic axis, section CG and tip mass are fixed per aircraft in Phase 1 "
                             "(Flight Dynamics decision, INTERFACE.md section 0) and cannot be overridden from a task config")
        if bad:
            raise ValueError(f"unknown flex keys {sorted(bad)}")
        fc = prof.flex_constants
        if not all(k in fc for k in fd_bridge.CHORD_CONSTANTS):
            raise ValueError(f"flex mode needs profile {prof.name} flex_constants (elastic_axis_frac, section_cg_frac)")
        fd_bridge.check_chord_constants(fc["elastic_axis_frac"], fc["section_cg_frac"], f"profile {prof.name}: ")
        fd_vals = fd_bridge.fd_chord_defaults(prof.jsbsim_model)
        mine = {k: float(fc[k]) for k in fd_bridge.CHORD_CONSTANTS}
        if fd_vals and fd_vals != mine:
            raise ValueError(f"profile {prof.name} flex_constants {mine} differ from FD's fixed values {fd_vals}; FD's are used")
        flex_constants_used = {**mine, "tip_mass_frac": 0.0}
    # Disabled roll/speed blocks: feed their (profile-scaled) defaults to the backend so jets don't fly
    # with C172-tuned helper gains. For c172x these equal sim.py's constants (tested bit-identical).
    sim_fixed = {}
    if raw.get("pass_fixed_helpers", True):
        for name in ("kp_roll", "ki_roll", "kd_roll", "kp_spd", "ki_spd", "kd_spd"):
            if name in spec.fixed:
                sim_fixed[name] = spec.fixed[name]
    if prof.gain_bounds_provisional:  # e.g. f16 (FBW): derived ranges are placeholders, untuned
        from dataclasses import replace as _replace
        spec.genes = [_replace(g, provisional=True) if g.block not in ("structure", GS.STRUCTURE_V2_BLOCK) else g
                      for g in spec.genes]
        warnings.append(f"PROVISIONAL gain bounds for {prof.name}: {prof.gain_bounds_provisional}")
    warnings += prof.warnings + spec.notes
    if dist and not conditions:
        raise ValueError("disturbance_scenario needs the jsbsim_ext task conditions (a shared Phase-1 preset)")
    init = resolve_init(raw.get("init"), spec, prof)
    return Task(raw.get("name", "task"), spec, fcfg, scfg, prof, raw, warnings, conditions, sim_fixed, flex_constants_used,
                dist or {}, init)


INIT_MODES = ("uniform", "baseline")


def resolve_init(cfg: Optional[Dict], spec: GS.GenomeSpec, prof) -> Dict:
    """Generation-0 seeding config ({} = evolve.py's own uniform draw, unchanged). Keys:
    mode            "uniform" | "baseline" (structure genes at their baseline + N(0, sigma) in normalized units)
    sigma           normalized-units std for the seeded genes (default 0.05); full ranges kept (clipped to [0, 1])
    blocks          blocks seeded near their baseline (default ["structure_v2"]); the other genes keep the uniform
                    draw, i.e. controller genes are initialised exactly as the current presets do
    seed_runs       {aircraft: [run dir, ...]}: best genomes (e.g. v4) copied into the first individuals for the genes
                    they have (controller genes); their structure genes sit exactly at the baseline
    """
    if not cfg:
        return {}
    cfg = dict(cfg)
    bad = set(cfg) - {"mode", "sigma", "blocks", "seed_runs", "_comment"}
    if bad:
        raise ValueError(f"unknown init keys {sorted(bad)}")
    mode = cfg.get("mode", "uniform")
    if mode not in INIT_MODES:
        raise ValueError(f"init.mode must be one of {INIT_MODES}, got {mode!r}")
    sigma = float(cfg.get("sigma", 0.05))
    if not (0.0 <= sigma <= 0.5):
        raise ValueError(f"init.sigma must be in [0, 0.5] (normalized units), got {sigma}")
    blocks = list(cfg.get("blocks", [GS.STRUCTURE_V2_BLOCK]))
    unknown = set(blocks) - set(spec.enabled_blocks)
    if unknown:
        raise ValueError(f"init.blocks {sorted(unknown)} are not evolved by this task ({list(spec.enabled_blocks)})")
    runs = dict(cfg.get("seed_runs", {}))
    return {"mode": mode, "sigma": sigma, "blocks": blocks, "seed_runs": list(runs.get(prof.name, []))}


# --------------------------------------------------------------------------- shims

_TASK: Optional[Task] = None


def make_genome_module(task: Task) -> types.ModuleType:
    m = types.ModuleType("genome")
    m.__doc__ = f"adapter shim for task {task.name!r}"
    m.SCHEMA = list(task.spec.genes)
    m.GENE_NAMES = list(task.spec.names)
    m.N_GENES = task.spec.n_genes
    m.decode = task.spec.decode
    m.encode = task.spec.encode
    m.Gene = GS.GeneSpec
    m.TASK = task
    return m


def make_sim_module(task: Task) -> types.ModuleType:
    S = orig_sim()
    m = types.ModuleType("sim")
    m.__dict__.update({k: v for k, v in vars(S).items() if not k.startswith("__")})
    m.__doc__ = f"adapter shim for task {task.name!r} (wraps flight_sim/sim.py)"
    import sim_ext
    m.Scenario = sim_ext.ExtScenario if (task.conditions or task.scenarios.get("set", "legacy") != "legacy") else S.Scenario
    m.AIRCRAFT = task.profile.jsbsim_model or S.AIRCRAFT
    m.make_scenarios = task.make_scenarios
    m.evaluate = task.evaluate

    def simulate(gains, sc, record=False):
        return sim_ext.simulate(gains, sc, record=record)
    def simulate_task(gains, sc, record=False):  # merge fixed helpers like Task.evaluate does
        return sim_ext.simulate({**task.sim_fixed, **gains}, sc, record=record)
    m.simulate = simulate_task if task.sim_fixed else simulate
    m.TASK = task
    return m


def install(task: Task) -> Dict[str, types.ModuleType]:
    """Register shims as ``genome`` / ``sim``. Call before importing evolve.py / plot_results.py."""
    global _TASK
    _TASK = task
    mods = {"genome": make_genome_module(task), "sim": make_sim_module(task)}
    for k in ("evolve", "plot_results"):  # make sure they bind to the shims
        sys.modules.pop(k, None)
    sys.modules.update(mods)
    return mods
