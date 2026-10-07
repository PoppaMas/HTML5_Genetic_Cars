"""P3-B2a shape block (phase3_b2a preset, shape_b2.py). Read-only on FD / evolution (no bytecode)."""
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
import shape_b1 as SB1  # noqa: E402
import shape_b2 as SB  # noqa: E402
from flightsim_path import orig_ga  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCKED = list(SB.DEFAULT_LOCKED_GENES)


@pytest.fixture(scope="module")
def task():
    return adapter.load_task("phase3_b2a")


@pytest.fixture(scope="module")
def task_x():
    return adapter.load_task("phase3_b2a_x")


@pytest.fixture(scope="module")
def task_b1():
    return adapter.load_task("phase3_b1")


# ------------------------------------------------------------------------------------------------ schema
def test_schema_pin_equals_fd_live_module():
    pb2 = SB.planform()
    SB.check_schema_pin()
    SB.check_encoding_pin()
    assert pb2.B2_FIDELITY == SB.B2_FIDELITY and pb2.B2_REV == SB.B2_REV_PIN
    assert pb2.REQUIRES_ENERGY == tuple(SB.DEFAULT_LOCKED_GENES)
    live_b1 = tuple((g.name, float(g.lo), float(g.hi), g.scale, float(g.default)) for g in pb2.pb1.SHAPE_GENES_B1)
    assert live_b1 == SB.B2_B1_SCHEMA_PIN == SB1.B1_SCHEMA_PIN


def test_b2_pins_match_fd_file():
    pins = json.load(open(os.path.join(SB.TEAM, "flight-dynamics", "v2_results", "model_versions_post_p3b2a.json")))
    assert {m: pins[m]["full_a1_b2a"] for m in SB.FD_B2_MODEL_VERSIONS} == SB.FD_B2_MODEL_VERSIONS
    assert {m: pins[m]["full_a1_b1"] for m in SB.FD_B1_MODEL_VERSIONS} == SB.FD_B1_MODEL_VERSIONS
    sch = pins["b2_schema"]
    assert sch["b2_rev"] == SB.B2_REV_PIN == SB.planform().B2_REV
    assert sch["encoding"] == SB.B2_GENE_ENCODING_PIN


def test_locked_genes_excluded_from_vector_length(task):
    assert task.spec.n_genes == 29
    shape = [g for g in task.spec.genes if g.block == SB.SHAPE_B2_BLOCK]
    assert len(shape) == 9
    names = [g.name for g in shape]
    assert names[:6] == [g[0] for g in SB.B2_B1_SCHEMA_PIN]
    assert names[6:] == ["wing_dihedral_delta_deg", "wing_camber_root_delta_pct", "wing_camber_tip_delta_pct"]
    assert "wing_tc_root_scale" not in names and "wing_tc_tip_ratio" not in names
    assert task.locked_genes == LOCKED


def test_identity_u_matches_fd(task):
    idu = SB.active_identity_u("c172x", LOCKED)
    assert np.allclose(task.operators["shape_ops"].identity_u(), idu)
    full = SB.pack_shape_u(idu, LOCKED, "c172x", task.shape_gene_specs)
    assert np.allclose(full, SB.planform().identity_u("c172x"))
    assert SB.planform().is_baseline_b2(SB.planform().decode_shape_b2(full, "c172x"), "c172x")


def test_packing_injects_fd_defaults():
    active = {"wing_dihedral_delta_deg": 1.5, "wing_camber_root_delta_pct": 0.5}
    full = SB.pack_shape_dict(active, LOCKED, "c172x")
    assert full["wing_tc_root_scale"] == 1.0 and full["wing_tc_tip_ratio"] == 1.0
    assert full["wing_dihedral_delta_deg"] == 1.5
    assert full["wing_chord_taper_1"] == 1.0  # missing active -> schema default via shape_defaults_b2 base


def test_load_rejects_missing_energy_locked_genes():
    raw = json.load(open(os.path.join(HERE, "presets", "phase3_b2a.json")))
    bad = copy.deepcopy(raw)
    bad["locked_genes"] = ["wing_dihedral_delta_deg"]  # missing tc
    with pytest.raises(ValueError, match="requires-energy|REQUIRES_ENERGY|energy"):
        adapter.build_task(bad)
    bad2 = copy.deepcopy(raw)
    bad2["locked_genes"] = []
    with pytest.raises(ValueError, match="requires-energy|energy"):
        adapter.build_task(bad2)


def test_preset_29_genes_blocks_init_ops(task, task_b1):
    blocks = [g.block for g in task.spec.genes]
    assert blocks[:8] == ["pitch_altitude"] * 6 + ["roll_heading"] * 2
    assert blocks[8:20] == ["structure_v2"] * 12 and blocks[20:] == [SB.SHAPE_B2_BLOCK] * 9
    assert [(g.name, g.min, g.max, g.scale) for g in task.spec.genes[:20]] == \
        [(g.name, g.min, g.max, g.scale) for g in task_b1.spec.genes[:20]]
    assert task.init == {"mode": "baseline", "sigma": 0.1, "blocks": ["structure_v2"], "seed_runs": []}
    sp = task.operators["shape_ops"]
    assert sp.idx == list(range(20, 29)) and sp.log == [True] * 3 + [False] * 6
    assert task.operators["blocks"] == {"controller": list(range(8)), "structure": list(range(8, 20)),
                                        "shape": list(range(20, 29))}
    assert task.flex_constants["fidelity"] == "full_a1_b2a"
    assert task.b2_model_version["status"] == "match"
    assert task.fitness.params.get("struct_v2_source") == "fd" or \
        task.raw["fitness"]["params"]["struct_v2_source"] == "fd"


def test_phase3_b2a_x_mirrors_b1_x(task, task_x):
    assert task_x.spec.n_genes == 29 and task_x.ga_overrides == {"elite": 4}
    assert task_x.operators["shape_crossover"] == "uniform"
    assert task.operators["shape_crossover"] == "block"
    assert [g.name for g in task_x.spec.genes] == [g.name for g in task.spec.genes]
    assert task_x.locked_genes == task.locked_genes


def test_phase2_and_b1_untouched():
    p2 = adapter.load_task("phase2_flex")
    assert type(p2) is adapter.Task and not hasattr(p2, "operators")
    b1 = adapter.load_task("phase3_b1")
    assert b1.spec.n_genes == 26
    assert all(g.block == SB1.SHAPE_B1_BLOCK for g in b1.spec.genes[20:])


def test_preset_guards():
    raw = json.load(open(os.path.join(HERE, "presets", "phase3_b2a.json")))
    bad = copy.deepcopy(raw); bad["flex"]["fidelity"] = "full_a1_b1"
    with pytest.raises(ValueError, match="full_a1_b2a"):
        adapter.build_task(bad)
    bad = copy.deepcopy(raw); bad["fitness"]["weights"]["effort"] = 3.0
    with pytest.raises(ValueError, match="cost weights"):
        adapter.build_task(bad)


# ------------------------------------------------------------------------------------------------ operators
def test_first_6_shape_cols_match_b1_identity_treatment(task, task_b1):
    """With only tc locked: first 6 shape columns use the same B1 identity / log-gene treatment as phase3_b1."""
    sp, sp1 = task.operators["shape_ops"], task_b1.operators["shape_ops"]
    assert sp.log[:6] == sp1.log == [True] * 3 + [False] * 3
    assert np.allclose(sp.identity_u()[:6], sp1.identity_u())
    assert np.allclose(sp.sigma(0.25)[:6], sp1.sigma(0.25))
    # gen-0 first 20 cols == phase2_flex (same seed); first 6 shape cols ~ B1 identity centre
    pop = block_ops.generation_zero(np.random.default_rng(7), 64, 29, task.spec, task.init, task.operators)
    p2 = adapter.load_task("phase2_flex")
    ref20 = init_pop.generation_zero(np.random.default_rng(7), 64, 20, p2.spec, p2.init)
    assert np.array_equal(pop[:, :20], ref20)
    assert np.allclose(np.median(pop[:, 20:26], axis=0), sp1.identity_u(), atol=0.05)


def test_uniform_crossover_draws_2_plus_n_s(task_x):
    ops = task_x.operators
    n_s = len(ops["blocks"]["shape"])
    assert n_s == 9
    a, b = np.arange(29.0), -np.arange(29.0) - 1.0
    for seed in range(10):
        rec = {}
        r1 = np.random.default_rng(seed)
        c = block_ops.block_crossover(r1, a, b, ops, rec)
        r2 = np.random.default_rng(seed)
        d = r2.random(2 + n_s)
        assert len(rec["shape_mask_a"]) == n_s
        assert rec["ctrl_a"] == bool(d[0] < 0.5) and rec["struct_a"] == bool(d[1] < 0.5)
        assert rec["shape_mask_a"] == [bool(x < 0.5) for x in d[2:]]


def test_block_crossover_keeps_blocks_intact(task):
    rng = np.random.default_rng(0)
    ops = task.operators
    seen = set()
    for _ in range(200):
        a, b = rng.random(29), rng.random(29)
        c = block_ops.block_crossover(rng, a, b, ops)
        src = []
        for name, idx in ops["blocks"].items():
            if np.array_equal(c[idx], a[idx]):
                src.append("a")
            else:
                assert np.array_equal(c[idx], b[idx]), name
                src.append("b")
        seen.add(tuple(src))
    assert len(seen) == 8


def test_phase3_b1_golden_operators_unchanged(task_b1):
    """Existing B1 operator path stays bit-identical (shape_b2 mapping must not alter shape_b1)."""
    ga = orig_ga()
    cfg = ga.GAConfig(pop_size=12, elite=2, mutation_rate=0.15, mutation_sigma=0.08)
    ranked = np.random.default_rng(3).random((12, 26))
    new = block_ops.next_generation(ga, np.random.default_rng(4), ranked, cfg, task_b1.operators)
    assert new.shape == (12, 26) and np.array_equal(new[:2], ranked[:2])
    assert task_b1.operators["blocks"]["shape"] == list(range(20, 26))
    assert block_ops.gene_block(task_b1.spec.genes[20]) == "shape"
