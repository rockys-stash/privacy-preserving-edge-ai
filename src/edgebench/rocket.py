"""Random convolutional kernel features for raw multichannel windows (after ROCKET).

Dempster, Petitjean and Webb, "ROCKET: exceptionally fast and accurate time series
classification using random convolutional kernels", Data Mining and Knowledge Discovery, 2020.

Each kernel has a random length (7, 9 or 11), mean-centred normal weights on a random subset of
channels, a uniform bias, an exponential dilation and random padding. Each kernel gives two
features: the proportion of positive values and the maximum of its output. Kernels are drawn
from a seed and do not depend on data, so the transform has no training step and cannot leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Kernel:
    weights: Array  # (channels_used, length)
    channels: NDArray[np.int64]
    bias: float
    dilation: int
    padding: int


def make_kernels(n_kernels: int, n_channels: int, length: int, seed: int) -> list[Kernel]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_kernels):
        k = int(rng.choice([7, 9, 11]))
        n_ch = int(min(n_channels, 2 ** rng.uniform(0, np.log2(n_channels + 1))))
        n_ch = max(1, n_ch)
        ch = np.sort(rng.choice(n_channels, n_ch, replace=False))
        w = rng.normal(0, 1, (n_ch, k))
        w -= w.mean(axis=1, keepdims=True)
        dil = int(2 ** rng.uniform(0, np.log2((length - 1) / (k - 1))))
        pad = ((k - 1) * dil) // 2 if rng.integers(2) else 0
        out.append(Kernel(w, ch, float(rng.uniform(-1, 1)), dil, pad))
    return out


def transform(x: Array, kernels: list[Kernel]) -> Array:
    """(n, channels, T) -> (n, 2 * kernels) features: [ppv, max] per kernel."""
    n = x.shape[0]
    feats = np.empty((n, 2 * len(kernels)))
    for j, kn in enumerate(kernels):
        xs = x[:, kn.channels, :]
        if kn.padding:
            xs = np.pad(xs, ((0, 0), (0, 0), (kn.padding, kn.padding)))
        span = (kn.weights.shape[1] - 1) * kn.dilation
        out_len = xs.shape[2] - span
        acc = np.full((n, out_len), kn.bias)
        for c in range(kn.weights.shape[0]):
            for i in range(kn.weights.shape[1]):
                off = i * kn.dilation
                acc += kn.weights[c, i] * xs[:, c, off : off + out_len]
        feats[:, 2 * j] = (acc > 0).mean(axis=1)
        feats[:, 2 * j + 1] = acc.max(axis=1)
    return feats


def operation_count(kernels: list[Kernel], t: int) -> int:
    """Multiply-accumulates plus the two pooling passes, for one window of length ``t``."""
    total = 0
    for kn in kernels:
        out_len = t + 2 * kn.padding - (kn.weights.shape[1] - 1) * kn.dilation
        total += out_len * (2 * kn.weights.size + 2)
    return total
