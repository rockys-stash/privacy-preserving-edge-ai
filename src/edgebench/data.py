"""The UCI HAR dataset: download, integrity check, loading and stream reconstruction.

The original release archive is fetched from a public mirror (the UCI host is not reachable from
the build environment) and accepted only if its SHA-256 matches the pinned value. The loader then
checks the counts the dataset's own README states. See data/DATASET.md.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ARCHIVE = "HAR_Smartphones.zip"
MIRROR_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/HAR_Smartphones.zip"
SHA256 = "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0"
FOLDER = "UCI HAR Dataset"

FS = 50  # Hz, stated in the dataset README
WINDOW = 128  # samples (2.56 s)
STEP = 64  # 50% overlap
ACTIVITIES = ("walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying")
# Channel order used everywhere: total acceleration (g), body acceleration (g), angular velocity (rad/s).
CHANNELS = ("total_acc_x", "total_acc_y", "total_acc_z", "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z")  # fmt: skip
EXPECTED = {"train": (7352, 21), "test": (2947, 9)}  # windows, subjects (README and dataset page)

Array = NDArray[np.float64]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(raw: Path = RAW, url: str = MIRROR_URL) -> Path:
    """Download (if needed), verify and extract the archive; return the dataset folder."""
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / ARCHIVE
    if not archive.exists():
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # fixed https URL; the archive is verified below
        tmp.rename(archive)
    digest = sha256(archive)
    if digest != SHA256:
        archive.rename(archive.with_suffix(".rejected"))
        raise RuntimeError(f"{archive.name} has SHA-256 {digest}, expected {SHA256}; moved aside")
    folder = raw / FOLDER
    if not (folder / "train" / "y_train.txt").exists():
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.namelist() if m.startswith(FOLDER + "/") and "/." not in m]
            z.extractall(raw, members)
    return folder


@dataclass(frozen=True)
class Split:
    signals: Array  # (windows, 9 channels, 128 samples)
    activity: NDArray[np.int64]  # 0..5
    subject: NDArray[np.int64]  # 1..30


@lru_cache(maxsize=2)
def load(split: str, raw: Path = RAW) -> Split:
    folder = raw / FOLDER / split
    if not folder.exists():
        raise FileNotFoundError(f"{folder} missing: run `edgebench fetch` (or `make data`) first")
    sig = np.stack([np.loadtxt(folder / "Inertial Signals" / f"{c}_{split}.txt") for c in CHANNELS], axis=1)
    y = np.loadtxt(folder / f"y_{split}.txt", dtype=np.int64) - 1
    s = np.loadtxt(folder / f"subject_{split}.txt", dtype=np.int64)
    n, k = EXPECTED[split]
    if sig.shape != (n, len(CHANNELS), WINDOW) or len(set(s.tolist())) != k or len(y) != n:
        raise ValueError(f"{split}: unexpected shape {sig.shape} / {len(set(s.tolist()))} subjects")
    keep = subset()
    if keep is not None:  # smoke runs and tests only; every reported result uses all 30 subjects
        m = np.isin(s, keep)
        return Split(sig[m], y[m], s[m])
    return Split(sig, y, s)


def subset() -> list[int] | None:
    """Subjects selected by EDGEBENCH_SUBJECTS (comma-separated), or None for all of them."""
    v = os.environ.get("EDGEBENCH_SUBJECTS", "").strip()
    return [int(x) for x in v.split(",") if x.strip()] or None


def load_all(raw: Path = RAW) -> Split:
    """All 10,299 windows of the 30 subjects, in the dataset's order (train then test)."""
    a, b = load("train", raw), load("test", raw)
    return Split(np.concatenate([a.signals, b.signals]), np.concatenate([a.activity, b.activity]),
                 np.concatenate([a.subject, b.subject]))  # fmt: skip


@dataclass(frozen=True)
class Segment:
    """A continuous stretch of one subject's recording, rebuilt from consecutive overlapping windows."""

    subject: int
    first_window: int  # index into load_all()
    n_windows: int
    samples: Array  # (9, n_samples)
    activity: NDArray[np.int64]  # per window


def segments(d: Split, tol: float = 1e-9) -> list[Segment]:
    """Join windows whose second half equals the next window's first half (same subject).

    The dataset was windowed with 50% overlap from continuous recordings, so consecutive windows of
    one recording share 64 samples exactly; a mismatch marks a break between recordings.
    """
    out: list[Segment] = []
    start = 0
    n = len(d.subject)
    for i in range(1, n + 1):
        cont = (
            i < n
            and d.subject[i] == d.subject[i - 1]
            and np.allclose(d.signals[i - 1, :, STEP:], d.signals[i, :, :STEP], atol=tol, rtol=0)
        )
        if not cont:
            w = d.signals[start:i]
            samples = np.concatenate([w[0], *[x[:, STEP:] for x in w[1:]]], axis=1)
            out.append(Segment(int(d.subject[start]), start, i - start, samples, d.activity[start:i]))
            start = i
    return out
