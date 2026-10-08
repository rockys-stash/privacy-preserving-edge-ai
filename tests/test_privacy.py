import math

import numpy as np
import pytest

from edgebench import privacy
from edgebench.features import extract
from edgebench.privacy import Calibration, PrivacyConfig

RNG = np.random.default_rng(0)
SIG = RNG.normal(0, 1, (40, 9, 128))
SEG = np.repeat([0, 1, 2], [20, 13, 7])
FEATS, SPEC = extract(SIG)
CAL = Calibration.fit(SIG, FEATS)
CFG = PrivacyConfig()


def test_blocks_never_cross_segments_and_hold_at_most_k() -> None:
    b = privacy.blocks(SEG, 8)
    for u in np.unique(b):
        members = np.flatnonzero(b == u)
        assert len(set(SEG[members])) == 1
        assert 1 <= len(members) <= 8
        assert np.all(np.diff(members) == 1)
    assert len(np.unique(b)) == math.ceil(20 / 8) + math.ceil(13 / 8) + math.ceil(7 / 8)


@pytest.mark.parametrize("level", range(5))
def test_raw_payload_shrinks_with_level(level: int) -> None:
    p = privacy.raw(SIG, SEG, level, CAL, CFG, np.random.default_rng(1))
    assert p.kind == "raw"
    expected_ch = 9 if level == 0 else 6
    assert p.data.shape[1] == expected_ch
    assert p.data.shape[2] == (128 if level < 2 else 64)
    assert len(p.units) == len(SIG)
    if level >= 4:
        assert p.n_units < len(SIG)
        assert p.bytes_per_window() < p.bytes_per_unit


def test_raw_l1_drops_exactly_the_gravity_channels() -> None:
    p = privacy.raw(SIG, SEG, 1, CAL, CFG, np.random.default_rng(1))
    keep = [c for c in range(9) if c not in privacy.GRAVITY_CHANNELS]
    np.testing.assert_array_equal(p.data, SIG[:, keep, :])


def test_feature_levels_drop_gravity_features_and_quantise() -> None:
    p0 = privacy.features(FEATS, SPEC, SEG, 0, CAL, CFG, np.random.default_rng(1))
    p1 = privacy.features(FEATS, SPEC, SEG, 1, CAL, CFG, np.random.default_rng(1))
    p2 = privacy.features(FEATS, SPEC, SEG, 2, CAL, CFG, np.random.default_rng(1))
    assert p0.data.shape[1] == 165 and p1.data.shape[1] == 110
    assert not any("total_acc" in SPEC.names[i] for i in SPEC.select(frozenset({"total_acc"})))
    assert p2.bytes_per_unit == math.ceil(110 * 4 / 8)
    # 4-bit quantisation: at most 16 distinct values per feature
    assert max(len(np.unique(p2.data[:, j])) for j in range(p2.data.shape[1])) <= 16


def test_feature_l3_noise_scale_matches_the_ldp_calibration() -> None:
    cfg = PrivacyConfig(feature_epsilon=2.0)
    big = np.repeat(FEATS, 50, axis=0)
    seg = np.repeat(np.arange(len(big) // 8 + 1), 8)[: len(big)]
    p2 = privacy.features(big, SPEC, seg, 2, CAL, cfg, np.random.default_rng(3))
    p3 = privacy.features(big, SPEC, seg, 3, CAL, cfg, np.random.default_rng(3))
    idx = SPEC.select(frozenset({"total_acc"}))
    z = (p3.data - p2.data) / (CAL.feature_hi[idx] - CAL.feature_lo[idx])
    d = len(idx)
    # Laplace(b) has mean absolute value b = d / epsilon
    assert abs(np.mean(np.abs(z)) / (d / 2.0) - 1) < 0.02


def test_randomised_response_keep_rate() -> None:
    labels = np.zeros(200_000, dtype=np.int64)
    out = privacy.randomised_response(labels, 1.0, np.random.default_rng(0))
    p_keep = math.e / (math.e + 5)
    assert abs((out == 0).mean() - p_keep) < 0.005
    # the other labels are uniform
    counts = np.bincount(out[out != 0], minlength=6)[1:]
    assert np.min(counts) / np.max(counts) > 0.95


def test_smoothing_is_causal_and_stays_inside_a_segment() -> None:
    labels = np.array([0, 1, 1, 1, 2, 2, 0, 0])
    seg = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    out = privacy.smooth(labels, seg, 4)
    assert out[0] == 0  # nothing before it
    assert out[3] == 1
    assert out[5] == 2  # first window of segment 1 only sees itself
    # changing a future label never changes a past decision
    later = labels.copy()
    later[-1] = 5
    np.testing.assert_array_equal(privacy.smooth(later, seg, 4)[:-1], out[:-1])


def test_local_l0_sends_label_and_probabilities_l1_only_the_label() -> None:
    proba = RNG.dirichlet(np.ones(6), size=len(SEG))
    p0 = privacy.decisions(proba, SEG, 0, CFG, np.random.default_rng(0))
    p1 = privacy.decisions(proba, SEG, 1, CFG, np.random.default_rng(0))
    assert p0.data.shape == (len(SEG), 7) and p0.bytes_per_unit == 25
    assert p1.data.shape == (len(SEG), 1) and p1.bytes_per_unit == 1
    np.testing.assert_array_equal(p1.data[:, 0], proba.argmax(axis=1))
