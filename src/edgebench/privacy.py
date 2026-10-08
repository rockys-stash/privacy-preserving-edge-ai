"""Privacy levels: what each architecture lets leave the device.

Levels are cumulative (docs/PLAN.md §2):

    L0 full  ->  L1 minimise  ->  L2 coarsen  ->  L3 perturb  ->  L4 aggregate

Calibration constants (channel spread, quantisation ranges, feature bounds) are public parameters
of the device firmware. They are computed from the training cohort only and passed in, never from
the person whose data is being transformed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from edgebench.features import FeatureSpec

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

LEVELS = (0, 1, 2, 3, 4)
LEVEL_NAMES = {0: "full", 1: "minimise", 2: "coarsen", 3: "perturb", 4: "aggregate"}
GRAVITY_CHANNELS = (0, 1, 2)  # total acceleration contains gravity, so device orientation
N_CLASSES = 6


@dataclass(frozen=True)
class PrivacyConfig:
    raw_downsample: int = 2  # 50 Hz -> 25 Hz
    raw_bits: int = 8
    raw_noise: float = 0.5  # Gaussian sigma as a share of each channel's standard deviation
    feature_bits: int = 4
    feature_epsilon: float = 4.0  # per payload, Laplace mechanism on the clipped feature vector
    label_epsilon: float = 4.0  # per payload, k-ary randomised response
    smooth: int = 4  # windows in the causal majority filter (local L2+)
    block: int = 8  # windows per aggregated payload (L4)


@dataclass(frozen=True)
class Calibration:
    """Public firmware constants derived from the training cohort."""

    channel_lo: Array  # per raw channel (9,)
    channel_hi: Array
    channel_std: Array
    feature_lo: Array  # per feature (d,)
    feature_hi: Array

    @staticmethod
    def fit(signals: Array, feats: Array) -> Calibration:
        return Calibration(
            channel_lo=np.quantile(signals, 0.001, axis=(0, 2)),
            channel_hi=np.quantile(signals, 0.999, axis=(0, 2)),
            channel_std=signals.std(axis=(0, 2)),
            feature_lo=np.quantile(feats, 0.01, axis=0),
            feature_hi=np.quantile(feats, 0.99, axis=0) + 1e-9,
        )


@dataclass
class Payload:
    """What the cloud receives: one row per transmitted unit, and which windows each unit covers."""

    kind: str  # raw | features | decision
    data: Array  # raw: (units, C, T); features: (units, d); decision: (units, 1 + optional 6 probabilities)
    units: IntArray  # for every window, the index of the unit that carries it
    bytes_per_unit: int
    notes: dict[str, float | str] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return int(self.data.shape[0])

    def bytes_per_window(self) -> float:
        return self.bytes_per_unit * self.n_units / len(self.units)


def blocks(segment: IntArray, k: int) -> IntArray:
    """Block index per window: consecutive groups of ``k`` windows that never cross a segment."""
    out = np.empty(len(segment), dtype=np.int64)
    b, count, prev = -1, 0, None
    for i, s in enumerate(segment):
        if s != prev or count == k:
            b, count, prev = b + 1, 0, s
        out[i] = b
        count += 1
    return out


def _quantise(x: Array, lo: Array, hi: Array, bits: int) -> Array:
    """Uniform quantisation inside [lo, hi] (broadcast), returned as the decoded value."""
    levels = 2**bits - 1
    q = np.round((np.clip(x, lo, hi) - lo) / (hi - lo) * levels)
    return lo + q / levels * (hi - lo)


def _block_reduce(x: Array, units: IntArray, how: str) -> Array:
    n = int(np.max(units)) + 1
    if how == "mean":
        out = np.zeros((n, *x.shape[1:]))
        np.add.at(out, units, x)
        return out / np.bincount(units, minlength=n).reshape(-1, *([1] * (x.ndim - 1)))
    # "middle": the middle window of each block
    first = np.searchsorted(units, np.arange(n))
    last = np.searchsorted(units, np.arange(n), side="right") - 1
    return x[(first + last) // 2]


def raw(
    signals: Array, segment: IntArray, level: int, cal: Calibration, cfg: PrivacyConfig, rng: np.random.Generator
) -> Payload:
    """Centralized architecture: the device transmits raw windows."""
    n = len(signals)
    units = np.arange(n)
    x, ch = signals, np.arange(signals.shape[1])
    if level >= 1:
        ch = np.array([c for c in ch if c not in GRAVITY_CHANNELS])
        x = x[:, ch, :]
    bytes_per_sample = 4
    if level >= 2:
        f = cfg.raw_downsample
        x = (
            x[:, :, : x.shape[2] // f * f].reshape(x.shape[0], x.shape[1], -1, f).mean(axis=3)
        )  # anti-alias by averaging
    if level >= 3:
        x = x + rng.normal(0.0, 1.0, x.shape) * (cfg.raw_noise * cal.channel_std[ch])[None, :, None]
    if level >= 2:
        x = _quantise(x, cal.channel_lo[ch][None, :, None], cal.channel_hi[ch][None, :, None], cfg.raw_bits)
        bytes_per_sample = math.ceil(cfg.raw_bits / 8)
    if level >= 4:
        units = blocks(segment, cfg.block)
        x = _block_reduce(x, units, "middle")
    return Payload("raw", x, units, int(x.shape[1] * x.shape[2] * bytes_per_sample),
                   {"channels": len(ch), "samples": int(x.shape[2])})  # fmt: skip


def features(feats: Array, spec: FeatureSpec, segment: IntArray, level: int, cal: Calibration, cfg: PrivacyConfig,
             rng: np.random.Generator) -> Payload:  # fmt: skip
    """Edge architecture: the device transmits a feature vector."""
    n = len(feats)
    units = np.arange(n)
    idx = np.arange(feats.shape[1]) if level < 1 else spec.select(frozenset({"total_acc"}))
    x, lo, hi = feats[:, idx], cal.feature_lo[idx], cal.feature_hi[idx]
    if level >= 4:  # the device averages clean features over the block, then applies L2-L3 once
        units = blocks(segment, cfg.block)
        x = _block_reduce(x, units, "mean")
    nbytes = 4 * x.shape[1]
    notes: dict[str, float | str] = {"features": int(x.shape[1])}
    if level >= 2:
        x = _quantise(x, lo, hi, cfg.feature_bits)
        nbytes = math.ceil(x.shape[1] * cfg.feature_bits / 8)
    if level >= 3:
        # Each coordinate is scaled to [0, 1]; changing the whole window moves the vector by at most
        # d in L1 norm, so Laplace noise of scale d / epsilon gives epsilon-LDP per payload.
        d = x.shape[1]
        z = (x - lo) / (hi - lo) + rng.laplace(0.0, d / cfg.feature_epsilon, x.shape)
        x = lo + z * (hi - lo)
        nbytes = 2 * d  # sent as float16
        notes["epsilon"] = cfg.feature_epsilon
    return Payload("features", x, units, int(nbytes), notes)


def randomised_response(labels: IntArray, epsilon: float, rng: np.random.Generator, k: int = N_CLASSES) -> IntArray:
    """k-ary randomised response: keep with probability e^eps / (e^eps + k - 1), else a uniform other label."""
    keep = rng.random(len(labels)) < math.exp(epsilon) / (math.exp(epsilon) + k - 1)
    other = (labels + rng.integers(1, k, len(labels))) % k
    return np.where(keep, labels, other)


def smooth(labels: IntArray, segment: IntArray, width: int, k: int = N_CLASSES) -> IntArray:
    """Causal majority over the last ``width`` windows of the same segment (ties: the newest label)."""
    out = labels.copy()
    for i in range(len(labels)):
        lo = i
        while lo > 0 and i - lo + 1 < width and segment[lo - 1] == segment[i]:
            lo -= 1
        counts = np.bincount(labels[lo : i + 1], minlength=k)
        best = np.flatnonzero(counts == np.max(counts))
        out[i] = labels[i] if labels[i] in best else int(best[0])
    return out


def decisions(proba: Array, segment: IntArray, level: int, cfg: PrivacyConfig, rng: np.random.Generator) -> Payload:
    """Fully local architecture: the device decides and transmits the decision."""
    n = len(proba)
    units = np.arange(n)
    label = proba.argmax(axis=1).astype(np.int64)
    if level == 0:
        return Payload("decision", np.column_stack([label, proba]), units, 1 + 4 * proba.shape[1])
    if level >= 2:
        label = smooth(label, segment, cfg.smooth)
    notes: dict[str, float | str] = {}
    if level >= 4:
        units = blocks(segment, cfg.block)
        nb = int(np.max(units)) + 1
        counts = np.zeros((nb, N_CLASSES), dtype=np.int64)
        np.add.at(counts, (units, label), 1)
        label = counts.argmax(axis=1).astype(np.int64)
    if level >= 3:
        label = randomised_response(label, cfg.label_epsilon, rng)
        notes["epsilon"] = cfg.label_epsilon
    return Payload("decision", label[:, None].astype(np.float64), units, 1, notes)
