"""On-device feature extraction: the "local processing" of the edge architecture.

Per window, for each signal channel and for the magnitude of each sensor triad: six time-domain
statistics and seven spectral ones, plus the three pairwise axis correlations of each triad.
Every feature is tagged with the channel it comes from, so the privacy levels can drop the
orientation-revealing ones (anything computed from total acceleration, which contains gravity).

Operation counts are analytic and documented next to each block; they do not depend on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from edgebench.data import CHANNELS

Array = NDArray[np.float64]

TRIADS = {"total_acc": (0, 1, 2), "body_acc": (3, 4, 5), "body_gyro": (6, 7, 8)}
BANDS_HZ = ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 25.0))
TIME_STATS = ("mean", "std", "min", "max", "mad", "energy")
FREQ_STATS = (*(f"band_{lo:g}_{hi:g}" for lo, hi in BANDS_HZ), "dominant_hz", "spectral_entropy")


@dataclass(frozen=True)
class FeatureSpec:
    names: tuple[str, ...]
    source: tuple[str, ...]  # triad each feature derives from: total_acc | body_acc | body_gyro

    def select(self, exclude: frozenset[str]) -> NDArray[np.int64]:
        return np.array([i for i, s in enumerate(self.source) if s not in exclude], dtype=np.int64)


def _signals(w: Array) -> tuple[list[str], list[str], Array]:
    """The 9 channels plus the 3 triad magnitudes: (names, sources, (n, 12, T))."""
    names, sources, rows = [], [], []
    for triad, idx in TRIADS.items():
        for i in idx:
            names.append(CHANNELS[i])
            sources.append(triad)
            rows.append(w[:, i, :])
        names.append(f"{triad}_mag")
        sources.append(triad)
        rows.append(np.sqrt(sum(w[:, i, :] ** 2 for i in idx)))
    return names, sources, np.stack(rows, axis=1)


def extract(windows: Array, fs: float = 50.0) -> tuple[Array, FeatureSpec]:
    """Features for a batch of windows (n, 9, T) at sampling rate ``fs``."""
    names, sources, x = _signals(windows)
    n, _, t = x.shape
    centred = x - x.mean(axis=2, keepdims=True)
    time = np.stack([x.mean(axis=2), x.std(axis=2), x.min(axis=2), x.max(axis=2),
                     np.abs(centred).mean(axis=2), (x**2).mean(axis=2)], axis=2)  # fmt: skip
    spec = np.abs(np.fft.rfft(centred, axis=2)) ** 2
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)
    bands = [spec[:, :, (freqs >= lo) & (freqs < hi)].sum(axis=2) for lo, hi in BANDS_HZ]
    total = spec[:, :, 1:].sum(axis=2) + 1e-12
    p = spec[:, :, 1:] / total[:, :, None]
    entropy = -(p * np.log(p + 1e-12)).sum(axis=2) / np.log(p.shape[2])
    dominant = freqs[1:][np.argmax(spec[:, :, 1:], axis=2)]
    freq = np.stack([*(b / total for b in bands), dominant, entropy], axis=2)
    per = np.concatenate([time, freq], axis=2)  # (n, 12, 13)
    feats = [per.reshape(n, -1)]
    fnames = [f"{ch}.{s}" for ch in names for s in (*TIME_STATS, *FREQ_STATS)]
    fsrc = [src for src in sources for _ in (*TIME_STATS, *FREQ_STATS)]
    for triad, idx in TRIADS.items():
        a, b, cc = (windows[:, i, :] - windows[:, i, :].mean(axis=1, keepdims=True) for i in idx)
        sd = [np.sqrt((v**2).mean(axis=1)) + 1e-12 for v in (a, b, cc)]
        for (u, su), (v, sv), tag in (
            ((a, sd[0]), (b, sd[1]), "xy"),
            ((a, sd[0]), (cc, sd[2]), "xz"),
            ((b, sd[1]), (cc, sd[2]), "yz"),
        ):
            feats.append(((u * v).mean(axis=1) / (su * sv))[:, None])
            fnames.append(f"{triad}.corr_{tag}")
            fsrc.append(triad)
    return np.concatenate(feats, axis=1), FeatureSpec(tuple(fnames), tuple(fsrc))


def operation_count(n_channels: int, t: int) -> int:
    """Arithmetic operations to extract the features of one window (analytic, approximate).

    magnitudes: 3 triads x T x (3 mul + 2 add + 1 sqrt); time stats: ~7 T per signal;
    real FFT of length T: ~2.5 T log2 T; power and band sums: ~3 T/2 per signal;
    entropy: ~4 (T/2) per signal; correlations: 9 x 3 T.
    """
    triads = n_channels // 3
    signals = n_channels + triads
    fft = 2.5 * t * np.log2(t)
    per_signal = 7 * t + fft + 1.5 * t + 2 * t
    return int(triads * 6 * t + signals * per_signal + triads * 3 * 3 * t)
