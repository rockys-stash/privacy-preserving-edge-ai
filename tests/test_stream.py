"""The streaming pipeline must reproduce the batch transformations (the E5 claim, on a subset)."""

import numpy as np
import pytest

from tests.conftest import needs_data

pytestmark = needs_data


@pytest.fixture(scope="module")
def setup():  # type: ignore[no-untyped-def]
    from edgebench import bench, data
    from edgebench.privacy import Calibration, PrivacyConfig

    c = bench.corpus()
    segs = data.segments(data.load_all())
    test_subj = 9
    tr = np.flatnonzero(c.subject != test_subj)
    te = np.flatnonzero(c.subject == test_subj)
    cal = Calibration.fit(c.signals[tr], c.feats[tr])
    dev = bench.device_model(c.feats[tr], c.activity[tr], 0)
    picked = [(k, s) for k, s in enumerate(segs) if s.subject == test_subj][:6]
    return c, picked, tr, te, cal, dev, PrivacyConfig()


def test_segments_rebuild_every_window_exactly() -> None:
    from edgebench import data

    d = data.load_all()
    segs = data.segments(d)
    assert sum(s.n_windows for s in segs) == len(d.subject)
    for s in segs[:40]:
        for j in range(s.n_windows):
            np.testing.assert_array_equal(s.samples[:, j * 64 : j * 64 + 128], d.signals[s.first_window + j])
        assert len(set(s.activity.tolist())) >= 1


@pytest.mark.parametrize("arch", ["centralized", "edge", "local"])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_streamed_decisions_equal_batch(setup, arch: str, level: int) -> None:  # type: ignore[no-untyped-def]
    from edgebench.bench import CloudModel, payload, quantised_proba, unit_labels
    from edgebench.measure import LinkProfile
    from edgebench.stream import replay

    c, picked, tr, _te, cal, dev, cfg = setup
    idx = np.concatenate([np.arange(s.first_window, s.first_window + s.n_windows) for _, s in picked])
    rng = np.random.default_rng(0)
    cloud = None
    if arch != "local":
        p_tr = payload(arch, c, tr, level, cal, cfg, rng, quantised_proba(dev, c.feats[tr]))
        cloud = CloudModel(p_tr.kind, 0, 40).fit(p_tr, unit_labels(c.activity[tr], p_tr.units))
    p = payload(arch, c, idx, level, cal, cfg, rng, quantised_proba(dev, c.feats[idx]))
    batch = (p.data[:, 0].astype(np.int64) if arch == "local" else cloud.predict(p))[p.units]  # type: ignore[union-attr]
    link = LinkProfile("wifi", 1e7, 0.01, 5e-8)
    events = replay([s for _, s in picked], [k for k, _ in picked], arch, level, cal, cfg, dev, cloud, link, 0)
    assert len(events) == len(idx)
    np.testing.assert_array_equal([e.decision for e in events], batch)
    assert all(e.arrived_at_s is not None and e.arrived_at_s > e.closed_at_s for e in events)


def test_l4_waits_for_the_block(setup) -> None:  # type: ignore[no-untyped-def]
    from edgebench.measure import LinkProfile
    from edgebench.stream import replay

    _c, picked, _tr, _te, cal, dev, cfg = setup
    seg = max((s for _, s in picked), key=lambda s: s.n_windows)
    k = next(k for k, s in picked if s is seg)
    events = replay([seg], [k], "local", 4, cal, cfg, dev, None, LinkProfile("l", 1e7, 0.01, 0.0), 0)
    first_block = events[: min(cfg.block, len(events))]
    assert len({e.arrived_at_s for e in first_block}) == 1
    assert first_block[0].arrived_at_s >= first_block[-1].closed_at_s  # type: ignore[operator]
