"""Cost of each architecture x level: operations, wall time, bytes, latency and an energy proxy.

Operation counts are analytic (they do not depend on this machine). Wall times are measured on
this machine, single-threaded, one window at a time. Device latency and energy use stated device
and link profiles (configs/cost.yaml): they are modelled, and labelled as such everywhere.
"""

from __future__ import annotations

import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from edgebench import privacy, rocket
from edgebench.bench import ARCHITECTURES, CloudModel, Corpus, device_model, payload, quantised_proba, unit_labels
from edgebench.data import FS, STEP
from edgebench.features import extract, operation_count
from edgebench.privacy import Calibration, PrivacyConfig

WINDOW_STEP_S = STEP / FS  # a new window closes every 1.28 s


@dataclass(frozen=True)
class LinkProfile:
    name: str
    rate_bps: float  # effective uplink throughput
    one_way_s: float  # propagation and protocol delay
    joule_per_byte: float  # radio energy per transmitted byte (assumed)


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    ops_per_s: float  # sustained arithmetic throughput (assumed)
    joule_per_op: float  # energy per arithmetic operation (assumed)


def device_ops(arch: str, level: int, cfg: PrivacyConfig, n_features: int, t: int = 128) -> float:
    """Arithmetic operations on the device per window (analytic)."""
    channels = 9 if level == 0 else 6
    if arch == "centralized":
        samples = channels * t
        ops = 0.0
        if level >= 2:
            ops += samples + samples / cfg.raw_downsample * 3  # averaging, then clip/scale/round
        if level >= 3:
            ops += samples / cfg.raw_downsample * 3  # noise draw (counted as ~2 ops) and add
        return ops / (cfg.block if level >= 4 else 1)
    feat = operation_count(9 if (arch == "local" or level == 0) else 6, t)  # the local model uses all channels
    if arch == "edge":
        d = n_features
        ops = float(feat)
        if level >= 4:
            ops += d  # running block mean
        tail = (3 * d if level >= 2 else 0) + (3 * d if level >= 3 else 0)
        return ops + tail / (cfg.block if level >= 4 else 1)
    ops = float(feat) + 2 * n_features * privacy.N_CLASSES + 3 * privacy.N_CLASSES  # scale, int8 matvec, softmax
    if level >= 2:
        ops += cfg.smooth * 2
    if level >= 3:
        ops += 4
    return ops


def cloud_ops(arch: str, model: Any, kernels: list[rocket.Kernel] | None, t: int) -> float:
    """Arithmetic operations in the cloud per payload (analytic)."""
    if arch == "local":
        return 0.0
    if arch == "centralized":
        assert kernels is not None
        n_feat = 2 * len(kernels)
        return float(rocket.operation_count(kernels, t) + 2 * n_feat + 2 * n_feat * privacy.N_CLASSES)
    hgb = model
    depth = np.mean([pred.get_max_depth() for it in hgb._predictors for pred in it])  # comparisons per tree
    return float(hgb.n_iter_ * len(hgb._predictors[0]) * depth)


def _median_time(fn: Callable[[], object], reps: int) -> float:
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


@dataclass
class CostRow:
    arch: str
    level: int
    bytes_per_payload: int
    bytes_per_window: float
    zlib_bytes_per_window: float
    uplink_bps: float
    device_ops: float
    cloud_ops: float
    device_ms_measured: float
    cloud_ms_measured: float
    model_bytes_on_device: int
    block_wait_s: float


def costs(
    c: Corpus, cfg: PrivacyConfig, seed: int, train_subjects: np.ndarray, n_kernels: int, reps: int
) -> list[CostRow]:
    """Bytes and operations for every cell, and single-window wall times measured here."""
    tr = np.flatnonzero(np.isin(c.subject, train_subjects))
    te = np.flatnonzero(~np.isin(c.subject, train_subjects))
    cal = Calibration.fit(c.signals[tr], c.feats[tr])
    dev = device_model(c.feats[tr], c.activity[tr], seed)
    proba_tr = quantised_proba(dev, c.feats[tr])
    one = te[:1]
    rows = []
    for arch in ARCHITECTURES:
        for level in privacy.LEVELS:
            rng = np.random.default_rng(seed)
            p_tr = payload(arch, c, tr, level, cal, cfg, rng, proba_tr)
            # bytes as serialised on the wire, and after zlib, measured on 200 consecutive test payloads
            sample = payload(arch, c, te[:1600], level, cal, cfg, rng, quantised_proba(dev, c.feats[te[:1600]]))
            wire = _serialise(sample, cfg, level)
            zbytes = len(zlib.compress(wire, 9)) / len(sample.units)
            model, kernels = None, None
            if arch != "local":
                m = CloudModel(p_tr.kind, seed, n_kernels).fit(p_tr, unit_labels(c.activity[tr], p_tr.units))
                model, kernels = (m.model[-1] if arch == "centralized" else m.model), m.kernels or None
            t = p_tr.data.shape[2] if arch == "centralized" else 128
            n_feat = p_tr.data.shape[1] if arch == "edge" else c.feats.shape[1]

            def device_stage(arch: str = arch, level: int = level, rng: np.random.Generator = rng) -> object:
                x = c.signals[one]
                if arch == "centralized":
                    return privacy.raw(x, c.segment[one], min(level, 3), cal, cfg, rng)
                f, spec = extract(x)  # wall time covers all 9 channels; operation counts follow the level
                if arch == "edge":
                    return privacy.features(f, spec, c.segment[one], min(level, 3), cal, cfg, rng)
                return quantised_proba(dev, f)

            dev_ms = _median_time(device_stage, reps) * 1e3
            cloud_ms = 0.0
            if arch != "local":
                p1 = payload(arch, c, one, min(level, 3), cal, cfg, rng, None)

                def cloud_stage(m: CloudModel = m, p1: privacy.Payload = p1) -> object:
                    return m.predict(p1)

                cloud_ms = _median_time(cloud_stage, reps) * 1e3
            model_bytes = (
                (c.feats.shape[1] + 1) * privacy.N_CLASSES + 2 * c.feats.shape[1] * 4 if arch == "local" else 0
            )
            bpw = sample.bytes_per_window()
            rows.append(CostRow(arch, level, sample.bytes_per_unit, bpw, zbytes, bpw * 8 / WINDOW_STEP_S,
                                device_ops(arch, level, cfg, n_feat), cloud_ops(arch, model, kernels, t) / (cfg.block if level >= 4 else 1),
                                dev_ms, cloud_ms, model_bytes,
                                (cfg.block - 1) / 2 * WINDOW_STEP_S if level >= 4 else 0.0))  # fmt: skip
    return rows


def _serialise(p: privacy.Payload, cfg: PrivacyConfig, level: int) -> bytes:
    """The payload as the device would put it on the wire (no framing)."""
    if p.kind == "raw":
        if level >= 2:
            return _q8(p.data).tobytes()
        return p.data.astype(np.float32).tobytes()
    if p.kind == "features":
        if level >= 3:
            return p.data.astype(np.float16).tobytes()
        if level >= 2:
            return _pack4(p.data).tobytes()
        return p.data.astype(np.float32).tobytes()
    if p.data.shape[1] > 1:
        return np.concatenate(
            [p.data[:, :1].astype(np.uint8).ravel(), p.data[:, 1:].astype(np.float32).ravel().view(np.uint8)]
        ).tobytes()
    return p.data[:, 0].astype(np.uint8).tobytes()


def _q8(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(axis=(0, 2), keepdims=True), x.max(axis=(0, 2), keepdims=True) + 1e-12
    return np.round((x - lo) / (hi - lo) * 255).astype(np.uint8)


def _pack4(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(axis=0), x.max(axis=0) + 1e-12
    q = np.round((x - lo) / (hi - lo) * 15).astype(np.uint8)
    if q.shape[1] % 2:
        q = np.pad(q, ((0, 0), (0, 1)))
    return (q[:, 0::2] << 4) | q[:, 1::2]


def latency_s(row: CostRow, device: DeviceProfile, link: LinkProfile) -> float:
    """Window closes -> decision available at the cloud (modelled device compute and link)."""
    tx = row.bytes_per_payload * 8 / link.rate_bps + link.one_way_s
    return row.device_ops / device.ops_per_s + tx + row.cloud_ms_measured / 1e3 + row.block_wait_s


def energy_j_per_window(row: CostRow, device: DeviceProfile, link: LinkProfile) -> float:
    """Device energy proxy per window: computation plus radio (stated constants, not a measurement)."""
    return row.device_ops * device.joule_per_op + row.bytes_per_window * link.joule_per_byte
