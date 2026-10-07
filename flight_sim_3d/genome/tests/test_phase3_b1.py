"""P3-B1 shape block (phase3_b1 preset, shape_b1.py, block_ops.py). Read-only on FD / evolution (no bytecode)."""
import copy
import json
import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True

import adapter  # noqa: E402
import block_ops  # noqa: E402
import init_pop  # noqa: E402
import shape_b1 as SB  # noqa: E402
from flightsim_path import orig_ga  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ER_A1_BASELINE = (0.2419400885448951, [0.12772764189700128, 0.39710530396908106, 0.2009873197686029], 0.0)


@pytest.fixture(scope="module")
def task():
    return adapter.load_task("phase3_b1")


# ------------------------------------------------------------------------------------------------ schema mirror
def test_schema_pin_equals_fd_live_module():
    pb1 = SB.planform()
    live = tuple((g.name, g.lo, g.hi, g.scale, g.default) for g in pb1.SHAPE_GENES_B1)
    assert live == SB.B1_SCHEMA_PIN
    assert pb1.N_SHAPE_GENES == 6 and pb1.B1_FIDELITY == SB.B1_FIDELITY
    SB.check_schema_pin()


def test_schema_drift_fails_loudly():
    pb1 = SB.planform()
    bad = list(pb1.shape_schema())
    bad[2] = type(bad[2])(bad[2].name, 0.8, bad[2].hi, bad[2].scale, bad[2].default, bad[2].meaning)
    with pytest.raises(ValueError, match="drifted"):
        SB.check_schema_pin(bad)


def test_shape_genes_built_from_fd_schema():
    pb1 = SB.planform()
    gs = SB.shape_genes()
    assert [(g.name, g.min, g.max, g.scale, g.default) for g in gs] == \
        [(g.name, g.lo, g.hi, g.scale, g.default) for g in pb1.shape_schema()]
    assert all(g.block == SB.SHAPE_B1_BLOCK and g.scale == "linear" for g in gs)   # FD's vector encoding


def _fd_pins(name):
    return json.load(open(os.path.join(SB.TEAM, "flight-dynamics", "v2_results", name)))


def test_b1_pins_match_fd_r1_file():
    pins = _fd_pins("model_versions_post_p3b1r1.json")
    assert {m: pins[m]["full_a1_b1"] for m in SB.FD_B1_MODEL_VERSIONS} == SB.FD_B1_MODEL_VERSIONS
    assert {m: pins[m]["full_a1"] for m in SB.FD_A1_MODEL_VERSIONS} == SB.FD_A1_MODEL_VERSIONS
    sch = pins["b1_schema"]
    assert tuple((g["name"], g["lo"], g["hi"], g["scale"], g["default"]) for g in sch["genes"]) == SB.B1_SCHEMA_PIN
    assert sch["b1_rev"] == SB.B1_REV_PIN == SB.planform().B1_REV
    assert sch["encoding"] == SB.B1_GENE_ENCODING_PIN


def test_r0_pins_recorded_as_superseded():
    r0 = _fd_pins("model_versions_post_p3b1.json")
    assert {m: r0[m]["full_a1_b1"] for m in SB.FD_B1_MODEL_VERSIONS_R0} == SB.FD_B1_MODEL_VERSIONS_R0
    assert set(SB.FD_B1_MODEL_VERSIONS.values()).isdisjoint(SB.FD_B1_MODEL_VERSIONS_R0.values())
    for m in SB.FD_B1_MODEL_VERSIONS:
        assert SB.classify_b1_model_version(m, SB.FD_B1_MODEL_VERSIONS[m]) == "match"
        assert SB.classify_b1_model_version(m, SB.FD_B1_MODEL_VERSIONS_R0[m]) == "superseded_r0"
        assert SB.classify_b1_model_version(m, "full_a1_b1:flexv2b1:deadbeef") == "unknown"


def test_gene_encoding_pin_and_drift():
    pb1 = SB.planform()
    assert pb1.GENE_ENCODING == SB.B1_GENE_ENCODING_PIN
    SB.check_encoding_pin()
    with pytest.raises(ValueError, match="GENE_ENCODING drifted"):
        SB.check_encoding_pin("log for chord tapers")


def test_genome_decode_encode_equals_fd_gene_encoding_bitwise():
    """FD r1: all 6 genes linear in value, value = lo + u (hi - lo); genome storage must be exactly that."""
    pb1, gs = SB.planform(), SB.shape_genes()
    rng = np.random.default_rng(7)
    U = np.vstack([rng.random((2000, 6)), np.zeros(6), np.ones(6), [[g.encode(g.default) for g in gs]],
                   np.eye(6), 1.0 - np.eye(6)])
    for u in U:
        fd = pb1.decode_shape_b1(list(u))
        gd = SB.shape_from_values({g.name: g.decode(x) for g, x in zip(gs, u)}, gs)
        assert gd == fd
        assert np.array_equal(pb1.encode_shape_b1(fd), np.array([g.encode(gd[g.name]) for g in gs]))


# ------------------------------------------------------------------------------------------------ preset / task
def test_preset_26_genes_blocks_init_ops(task):
    blocks = [g.block for g in task.spec.genes]
    assert task.spec.n_genes == 26
    assert blocks[:8] == ["pitch_altitude"] * 6 + ["roll_heading"] * 2
    assert blocks[8:20] == ["structure_v2"] * 12 and blocks[20:] == [SB.SHAPE_B1_BLOCK] * 6
    p2 = adapter.load_task("phase2_flex")
    assert [(g.name, g.min, g.max, g.scale) for g in task.spec.genes[:20]] == \
        [(g.name, g.min, g.max, g.scale) for g in p2.spec.genes]
    assert task.init == {"mode": "baseline", "sigma": 0.1, "blocks": ["structure_v2"], "seed_runs": []}
    assert task.operators["crossover"] == "blocks"
    sp = task.operators["shape_ops"]
    assert sp.idx == list(range(20, 26)) and sp.log == [True] * 3 + [False] * 3
    assert sp.init_sigma_frac == sp.mut_sigma_frac == 0.25 and sp.mutation_rate is None
    assert np.allclose(sp.sigma(0.25)[3:], [0.375, 0.625, 1.25])                     # 0.125 x (hi - lo)
    assert np.allclose(sp.sigma(0.25)[:3], 0.125 * (np.log(1.05) - np.log(0.85)))
    assert task.operators["blocks"] == {"controller": list(range(8)), "structure": list(range(8, 20)),
                                        "shape": list(range(20, 26))}
    assert task.flex_constants["fidelity"] == "full_a1_b1"
    assert task.b1_model_version["status"] == "match"
    assert task.sim_fixed == {}


def test_phase2_flex_untouched_by_b1():
    p2 = adapter.load_task("phase2_flex")
    assert type(p2) is adapter.Task and not hasattr(p2, "operators")
    assert p2.init == {"mode": "baseline", "sigma": 0.1, "blocks": ["structure_v2"], "seed_runs": []}


def test_preset_guards():
    raw = json.load(open(os.path.join(HERE, "presets", "phase3_b1.json")))
    bad = copy.deepcopy(raw); bad["flex"]["fidelity"] = "full_a1"
    with pytest.raises(ValueError, match="full_a1_b1"):
        adapter.build_task(bad)
    bad = copy.deepcopy(raw); bad["fitness"]["weights"]["effort"] = 3.0
    with pytest.raises(ValueError, match="cost weights"):
        adapter.build_task(bad)
    bad = copy.deepcopy(raw); bad["operators"]["crossover"] = "uniform"
    with pytest.raises(ValueError, match="crossover"):
        adapter.build_task(bad)
    bad = copy.deepcopy(raw); bad["operators"]["shape"]["log_genes"] = ["wing_ei_root"]
    with pytest.raises(ValueError, match="log_genes"):
        adapter.build_task(bad)


def test_model_version_raise_mode(monkeypatch):
    pd = SB.er_profile("evolution/configs/phase2_smoke_p25.json", "c172x")
    assert SB.check_b1_model_version("c172x", pd, "raise")["match"] is True
    monkeypatch.setitem(SB.FD_B1_MODEL_VERSIONS, "c172x", "full_a1_b1:flexv2b1:deadbeef")
    assert SB.check_b1_model_version("c172x", pd, "warn")["status"] == "unknown"
    with pytest.raises(RuntimeError):
        SB.check_b1_model_version("c172x", pd, "raise")


def test_model_version_r0_is_warn_only_never_raises(monkeypatch):
    """r0 -> r1 transition (same policy as the P2.5 post-mass window): a live r0 string warns, even in 'raise'."""
    pd = SB.er_profile("evolution/configs/phase2_smoke_p25.json", "c172x")
    monkeypatch.setattr(SB, "fd_b1_model_version", lambda profile_d: SB.FD_B1_MODEL_VERSIONS_R0["c172x"])
    for mode in ("warn", "raise"):
        r = SB.check_b1_model_version("c172x", pd, mode)
        assert r["status"] == "superseded_r0" and r["match"] is False and "r0" in r["message"]


def test_shape_from_values_is_exact_passthrough():
    """No clamp / snap between the genome decode and FD (r1: FD GENE_ENCODING exactly, = ER's shape_values)."""
    gs = SB.shape_genes()
    vals = {g.name: g.default for g in gs}
    vals["wing_chord_taper_1"] = 1.0 + 1e-15
    vals["wing_twist_tip_deg"] = -2.0
    out = SB.shape_from_values(vals, gs)
    assert out == vals and out["wing_chord_taper_1"] != 1.0
    vals["wing_chord_taper_2"] = 1.05 + 1e-14          # out of range is passed through; FD's decode raises
    with pytest.raises(ValueError, match="outside its range"):
        SB.planform().decode_shape_b1(SB.shape_from_values(vals, gs))
    u = [g.encode(g.default) for g in gs]             # identity planform round-trips exactly
    assert SB.shape_from_values({g.name: g.decode(x) for g, x in zip(gs, u)}, gs) == SB.planform().shape_defaults()


# ------------------------------------------------------------------------------------------------ reject path
@pytest.fixture(scope="module")
def er_setup():
    pd = SB.er_profile("evolution/configs/phase2_smoke_p25.json", "c172x")
    g = json.load(open(os.path.join(HERE, "runs", "p25_tip_verify_c172x.json")))["baseline"]
    return pd, SB.er_scenarios(pd, 3, 1), g["gains"], g["struct"]


@pytest.mark.parametrize("shape", [{"wing_chord_taper_3": 0.5}, {"wing_dihedral_delta_deg": 2.0},
                                   {"wing_ei_root": 1.0}, {"wing_sweep_qc_delta_deg": float("nan")}])
def test_decode_errors_raise_never_clip(er_setup, shape):
    pd, scs, gains, struct = er_setup
    with pytest.raises(ValueError):
        SB.evaluate_b1(pd, gains, struct, shape, scs)


def test_geometry_gate_reject_is_hard_not_flown(er_setup, monkeypatch):
    pd, scs, gains, struct = er_setup
    pb1 = SB.planform()
    monkeypatch.setattr(pb1, "GATE_MAX_TAPER", 0.5)   # in memory only: make an in-range shape fail FD's gate
    r = SB.evaluate_b1(pd, gains, struct, {"wing_chord_taper_3": 0.9}, scs, via="fd")
    assert r["status"] == "geometry_gate:extreme_taper" and SB.geometry_rejected(r)
    assert r["cost"] == 2000.0 and r["feasible"] is False
    assert len(r["per_scenario"]) == 3 and all(e["not_flown"] and e["cost"] == 2000.0 for e in r["per_scenario"])
    assert all(v == 0.0 for k, v in r["terms"].items())   # no structural credit, no flown terms


def test_in_range_box_corners_pass_fd_gate(er_setup):
    import itertools
    F, esim, _, _ = SB.er_modules()
    pd = er_setup[0]
    m = F.fd_modules()
    geom = m["fb"].geometry_for("c172x", F.roots(esim.Profile.from_dict(pd))[1])
    pw = m["fw"].params_for("c172x", geom.bw_ft, geom.sw_ft2, geom.empty_wt_lb)
    pb1 = SB.planform()
    for c in itertools.product(*[(g.lo, g.hi) for g in pb1.shape_schema()]):
        assert pb1.geometry_gate(pw, dict(zip(pb1.SHAPE_NAMES, c)), n_el=64).ok


# ------------------------------------------------------------------------------------------------ operators
def test_block_crossover_keeps_blocks_intact(task):
    rng = np.random.default_rng(0)
    ops = task.operators
    seen = set()
    for _ in range(200):
        a, b = rng.random(26), rng.random(26)
        c = block_ops.block_crossover(rng, a, b, ops)
        src = []
        for name, idx in ops["blocks"].items():
            if np.array_equal(c[idx], a[idx]):
                src.append("a")
            else:
                assert np.array_equal(c[idx], b[idx]), name
                src.append("b")
        seen.add(tuple(src))
    assert len(seen) == 8   # every block independently from either parent


def test_shape_mutation_encoded_space(task):
    ga = orig_ga()
    cfg = ga.GAConfig(mutation_rate=1.0, mutation_sigma=0.08)
    sp = task.operators["shape_ops"]
    rng = np.random.default_rng(1)
    g = np.concatenate([np.full(20, 0.5), sp.identity_u()])
    kids = np.array([block_ops.block_mutate(rng, g, cfg, task.operators) for _ in range(4000)])
    x = np.array(sp.lo) + kids[:, 20:] * (np.array(sp.hi) - np.array(sp.lo))
    assert abs(np.median(np.abs(np.log(x[:, :3]))) / 0.6745 - sp.sigma(0.25)[0]) < 0.002   # ln-space for chord
    assert abs(np.median(np.abs(x[:, 3] - 0.0)) / 0.6745 - 0.375) < 0.02                  # twist mid (linear)
    assert abs((kids[:, 8:20] - 0.5).std() - 0.08) < 0.004                                # structure unchanged
    assert kids.min() >= 0.0 and kids.max() <= 1.0
    # controller/structure part == ga.mutate when the shape block is never hit
    ops0 = dict(task.operators, shape_ops=block_ops.ShapeOps(**{**sp.__dict__, "mutation_rate": 0.0}))
    for seed in range(5):
        c2 = ga.GAConfig(mutation_rate=0.3, mutation_sigma=0.08)
        mine = block_ops.block_mutate(np.random.default_rng(seed), g, c2, ops0)
        ref = np.random.default_rng(seed)
        hit = ref.random(26) < np.where(np.arange(26) >= 20, 0.0, 0.3)
        exp = g.copy(); exp[hit] += ref.normal(0.0, 0.08, int(hit.sum()))
        assert np.array_equal(mine, np.clip(exp, 0, 1))


def test_next_generation_keeps_elites_and_bounds(task):
    ga = orig_ga()
    cfg = ga.GAConfig(pop_size=12, elite=2)
    ranked = np.random.default_rng(3).random((12, 26))
    new = block_ops.next_generation(ga, np.random.default_rng(4), ranked, cfg, task.operators)
    assert new.shape == (12, 26) and np.array_equal(new[:2], ranked[:2])
    assert new.min() >= 0.0 and new.max() <= 1.0


def test_generation_zero_identity_shape(task):
    pop = block_ops.generation_zero(np.random.default_rng(7), 64, 26, task.spec, task.init, task.operators)
    p2 = adapter.load_task("phase2_flex")    # first 20 genes == phase2_flex's seeded gen 0 (same seed)
    ref20 = init_pop.generation_zero(np.random.default_rng(7), 64, 20, p2.spec, p2.init)
    assert np.array_equal(pop[:, :20], ref20)
    sp = task.operators["shape_ops"]
    assert np.allclose(np.median(pop[:, 20:], axis=0), sp.identity_u(), atol=0.05)
    ident = SB.shape_from_values({g.name: g.decode(x) for g, x in zip(task.spec.genes[20:], sp.identity_u())},
                                 task.spec.genes[20:])
    assert ident == SB.planform().shape_defaults()          # centre = FD's identity planform exactly


def test_generation_zero_without_operators_unchanged():
    p2 = adapter.load_task("phase2_flex")
    pop = init_pop.generation_zero(np.random.default_rng(11), 16, p2.spec.n_genes, p2.spec, p2.init)
    rng = np.random.default_rng(11)          # the Phase 2 formula, inline
    ref = rng.random((16, p2.spec.n_genes))
    idx, base = init_pop.baseline_u(p2.spec, p2.init["blocks"])
    ref[:, idx] = np.clip(base + 0.1 * rng.standard_normal((16, len(idx))), 0.0, 1.0)
    assert np.array_equal(pop, ref)


@pytest.fixture(scope="module")
def er_ga():
    SB.er_modules()
    from evolution import batch, ga as ega
    return ega, batch


def _er_spec(ega, sp):
    return ega.ShapeSpec(idx=list(sp.idx), lo=list(sp.lo), hi=list(sp.hi), log=list(sp.log), default=list(sp.default),
                         init_sigma_frac=sp.init_sigma_frac, mut_sigma_frac=sp.mut_sigma_frac,
                         mutation_rate=0.15 if sp.mutation_rate is None else sp.mutation_rate)


def test_operators_bitwise_equal_to_er_ga(task, er_ga):
    """Genome block_ops == ER evolution/ga.py P3-B1 operators (read-only import), same seeds -> same arrays."""
    ega, batch = er_ga
    if not hasattr(ega, "next_generation_blocks"):
        pytest.skip("ER ga.py has no P3-B1 operators")
    ops, sp = task.operators, task.operators["shape_ops"]
    es = _er_spec(ega, sp)
    blocks = list(ops["blocks"].values())
    assert blocks == batch.gene_blocks(["gains"] * 8 + ["struct"] * 12 + ["shape"] * 6)
    cfg = ega.GAConfig(pop_size=20, elite=2, mutation_rate=0.15, mutation_sigma=0.08)
    for seed in range(6):
        assert np.array_equal(block_ops.shape_generation_zero(np.random.default_rng(seed), 32, sp),
                              ega.shape_generation_zero(np.random.default_rng(seed), 32, es))
        ranked = np.random.default_rng(100 + seed).random((20, 26))
        mine = block_ops.next_generation(orig_ga(), np.random.default_rng(seed), ranked, cfg, ops)
        theirs = ega.next_generation_blocks(np.random.default_rng(seed), ranked, cfg, blocks, es)
        assert np.array_equal(mine, theirs)


def test_generation_zero_bitwise_equal_to_er_batch(task, er_ga):
    ega, batch = er_ga
    if not hasattr(ega, "shape_generation_zero"):
        pytest.skip("ER ga.py has no P3-B1 operators")
    F = SB.er_modules()[0]
    sp = task.operators["shape_ops"]
    for seed in (1, 2, 3):
        mine = block_ops.generation_zero(np.random.default_rng(seed), 16, 26, task.spec, task.init, task.operators)
        rng = np.random.default_rng(seed)       # ER batch.py phase3_b1 gen-0 sequence
        pop = ega.generation_zero(rng, 16, 20)
        pop = batch.seed_generation_zero(pop, rng, [None] * 8 + F.struct_schema(), ["gains"] * 8 + ["struct"] * 12,
                                         {"mode": "baseline", "sigma": 0.1})
        theirs = np.hstack([pop, ega.shape_generation_zero(rng, 16, _er_spec(ega, sp))])
        assert np.array_equal(mine, theirs)


# ------------------------------------------------------------------------------------------------ physics (sim)
@pytest.mark.sim
def test_baseline_shape_equals_a1_bitwise(er_setup, task):
    pd, scs, gains, struct = er_setup
    r = SB.evaluate_b1(pd, gains, struct, {}, scs, check_mv=SB.FD_B1_MODEL_VERSIONS["c172x"])
    assert r["status"] == "ok"
    assert (r["cost"], [e["cost"] for e in r["per_scenario"]], r["terms"]["J_wing_tip_bm_limit"]) == ER_A1_BASELINE
    # the same genome through the phase3_b1 Task (physical values, identity shape) -> same numbers
    vals = {**gains, **struct, **SB.planform().shape_defaults()}
    rt = task.evaluate(vals, task.make_scenarios(3, 1))
    assert rt["cost"] == ER_A1_BASELINE[0] and rt["J_wing_tip_bm_limit"] == 0.0
    assert set(rt["objectives"]) == {"track_alt", "effort", "comfort", "track_heading_rms", "structural_v2"}
