import numpy as np
import pytest

import genome_schema as GS
from flightsim_path import orig_genome


def test_roundtrip_all_scales():
    rng = np.random.default_rng(0)
    for g in GS.ALL_GENES.values():
        for v in rng.random(50):
            x = g.decode(v)
            assert g.min - 1e-12 <= x <= g.max + 1e-12 or (g.scale == "log0" and x == 0.0)
            if g.scale == "log0" and v <= g.zero_band:
                assert x == 0.0 and g.encode(x) == 0.0
            else:
                assert np.isclose(g.encode(x), v, atol=1e-12)


def test_log0_switches_off_and_spans_range():
    g = GS.ALL_GENES["ki_alt"]
    assert g.scale == "log0"
    assert g.decode(0.0) == 0.0 and g.decode(g.zero_band) == 0.0
    assert np.isclose(g.decode(g.zero_band + 1e-12), g.min)
    assert np.isclose(g.decode(1.0), g.max)


def test_legacy_ranges_match_original_genome_py():
    og = orig_genome()
    assert [g.name for g in og.SCHEMA] == list(GS.LEGACY_PITCH_RANGES)
    for g in og.SCHEMA:
        assert GS.LEGACY_PITCH_RANGES[g.name] == (g.min, g.max, g.kind)


def test_legacy_spec_decode_bit_identical_to_original():
    og = orig_genome()
    spec = GS.build_spec(["pitch_altitude"], legacy_pitch_ranges=True)
    assert spec.names == og.GENE_NAMES and spec.n_genes == og.N_GENES
    rng = np.random.default_rng(1)
    for _ in range(200):
        g = rng.random(6)
        assert spec.decode(g) == og.decode(g)
        assert spec.encode(og.decode(g)) == og.encode(og.decode(g))


def test_block_enable_disable_layout_and_fixed():
    s1 = GS.build_spec(["pitch_altitude"])
    s3 = GS.build_spec(["speed_throttle", "pitch_altitude", "roll_heading"])  # order normalized
    sall = GS.build_spec(GS.BLOCK_ORDER)
    assert s1.n_genes == 6 and s3.n_genes == 14 and sall.n_genes == 21
    assert s3.enabled_blocks == ("pitch_altitude", "roll_heading", "speed_throttle")
    lay = s3.layout()
    assert lay["pitch_altitude"] == slice(0, 6) and lay["roll_heading"] == slice(6, 11) and lay["speed_throttle"] == slice(11, 14)
    assert "kp_roll" in s1.fixed and s1.fixed["kp_roll"] == 0.05  # disabled -> default (= legacy wing leveler)
    full = s1.decode(s1.default_genome(), include_fixed=True)
    assert set(full) == set(GS.ALL_GENES)
    with pytest.raises(ValueError):
        GS.build_spec(["nope"])
    with pytest.raises(ValueError):
        s3.decode(np.zeros(6))


def test_transfer_between_specs_keeps_physical_values():
    s1 = GS.build_spec(["pitch_altitude"])
    s3 = GS.build_spec(["pitch_altitude", "roll_heading", "speed_throttle"])
    g1 = np.random.default_rng(2).random(6)
    g3 = s3.transfer(g1, s1)
    d1, d3 = s1.decode(g1), s3.decode(g3)
    for k, v in d1.items():
        assert np.isclose(d3[k], v, rtol=1e-12, atol=0)
    assert np.isclose(d3["kp_roll"], 0.05)


def test_overrides_and_validation():
    s = GS.build_spec(["pitch_altitude"], overrides={"kp_alt": {"min": 0.01, "max": 1.0}})
    g = s.genes[0]
    assert (g.min, g.max) == (0.01, 1.0) and any("override kp_alt" in n for n in s.notes)
    with pytest.raises(ValueError):
        GS.build_spec(["pitch_altitude"], overrides={"bogus": {}})
    with pytest.raises(ValueError):
        GS.GeneSpec("x", "b", 0.0, 1.0, "log")


def test_at_bounds_reporting():
    spec = GS.build_spec(["pitch_altitude"])  # kp_alt log, ki_alt log0, ...
    best = np.array([1.0, 0.01, 0.5, 0.0, 0.5, 0.995])
    rep = {(e["gene"], e["bound"]) for e in spec.at_bounds(best)}
    assert ("kp_alt", "upper") in rep and ("kp_pitch", "lower") in rep and ("kd_pitch", "upper") in rep
    assert ("ki_alt", "zeroed") in rep and ("ki_alt", "lower") not in rep  # log0: zero band is 'switched off'
    assert not any(e[0] == "kd_alt" for e in rep)
    near = {(e["gene"], e["bound"]) for e in spec.at_bounds(np.array([0.93, 0.08, 0.5, 0.05, 0.5, 0.5]))}
    assert near == {("kp_alt", "near_upper"), ("ki_alt", "near_lower"), ("kp_pitch", "near_lower")}
    pop = np.tile(np.full(6, 0.5), (10, 1)); pop[:3, 0] = 1.0
    e = [x for x in spec.at_bounds(pop) if x["gene"] == "kp_alt"][0]
    assert np.isclose(e["fraction"], 0.3)
    lo, hi = GS.suggest_widened_range(spec.genes[0], "upper")
    assert lo == spec.genes[0].min and hi == 4 * spec.genes[0].max


def test_v2_ranges_widen_pinned_genes():
    v2 = {g.name: g for g in GS.build_spec(["pitch_altitude"]).genes}
    assert v2["kp_alt"].max > GS.LEGACY_PITCH_RANGES["kp_alt"][1]
    assert v2["ki_alt"].scale == "log0"  # can reach exactly 0 instead of pinning at 1e-5
