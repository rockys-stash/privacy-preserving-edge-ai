"""The benchmark: every architecture at every privacy level, evaluated on unseen people.

Protocol (docs/METRICS.md):

- Utility: subject-grouped k-fold cross-validation. Firmware calibration, the on-device model and
  the cloud model are fitted on the training subjects only; accuracy is measured per window on
  the held-out subjects. Repeated over seeds (fold assignment, noise, random kernels).
- Exposure: an adversary at the cloud who has enrolled every subject with part of their own
  payloads (alternate recording segments of each subject and activity) tries to name the subject
  of each remaining payload. Splitting by segment keeps overlapping windows out of both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from edgebench import data, privacy, rocket
from edgebench.features import FeatureSpec, extract
from edgebench.privacy import Calibration, Payload, PrivacyConfig

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
ARCHITECTURES = ("centralized", "edge", "local")


@dataclass(frozen=True)
class Corpus:
    signals: Array
    activity: IntArray
    subject: IntArray
    segment: IntArray
    feats: Array
    spec: FeatureSpec


@lru_cache(maxsize=1)
def corpus() -> Corpus:
    d = data.load_all()
    seg = np.empty(len(d.subject), dtype=np.int64)
    for k, s in enumerate(data.segments(d)):
        seg[s.first_window : s.first_window + s.n_windows] = k
    feats, spec = extract(d.signals)
    return Corpus(d.signals, d.activity, d.subject, seg, feats, spec)


def subject_folds(subjects: IntArray, k: int, seed: int) -> list[IntArray]:
    ids = np.unique(subjects)
    np.random.default_rng(seed).shuffle(ids)
    return [np.sort(f) for f in np.array_split(ids, k)]


def unit_labels(y: IntArray, units: IntArray) -> IntArray:
    """The most frequent label among the windows each unit carries."""
    n = int(np.max(units)) + 1
    counts = np.zeros((n, int(np.max(y)) + 1), dtype=np.int64)
    np.add.at(counts, (units, y), 1)
    return counts.argmax(axis=1).astype(np.int64)


def device_model(feats: Array, y: IntArray, seed: int) -> Any:
    """The on-device classifier of the fully local architecture (weights are int8 on the device)."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)).fit(feats, y)


def quantised_proba(model: Any, feats: Array) -> Array:
    """Predict with int8-quantised weights, as the device would."""
    scaler, lr = model[0], model[-1]
    z = scaler.transform(feats)
    w = lr.coef_
    scale = np.abs(w).max() / 127
    wq = np.round(w / scale) * scale
    logits = z @ wq.T + lr.intercept_
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


class CloudModel:
    """The cloud's classifier for a payload kind; for decisions the cloud uses what it receives."""

    def __init__(self, kind: str, seed: int, n_kernels: int = 500, family: str = "strong") -> None:
        # family "linear" swaps the classifier for a logistic regression on the same inputs: a
        # weaker adversary, used to check that exposure rankings do not depend on the attacker (E2).
        self.kind, self.seed, self.n_kernels, self.family = kind, seed, n_kernels, family
        self.kernels: list[rocket.Kernel] = []
        self.model: Any = None

    def _x(self, p: Payload) -> Array:
        if self.kind == "raw":
            if not self.kernels:
                self.kernels = rocket.make_kernels(self.n_kernels, p.data.shape[1], p.data.shape[2], self.seed)
            return rocket.transform(p.data, self.kernels)
        if self.kind == "features":
            return p.data
        return np.column_stack([np.eye(privacy.N_CLASSES)[p.data[:, 0].astype(int)], p.data[:, 1:]])

    def fit(self, p: Payload, y_units: IntArray) -> CloudModel:
        x = self._x(p)
        if self.family == "linear":
            self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=self.seed)).fit(
                x, y_units
            )
        elif self.kind == "raw":
            self.model = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))).fit(
                x, y_units
            )
        elif self.kind == "features":
            self.model = HistGradientBoostingClassifier(random_state=self.seed).fit(x, y_units)
        else:
            self.model = LogisticRegression(max_iter=3000, random_state=self.seed).fit(x, y_units)
        return self

    def predict(self, p: Payload) -> IntArray:
        return np.asarray(self.model.predict(self._x(p)), dtype=np.int64)


def payload(arch: str, c: Corpus, idx: IntArray, level: int, cal: Calibration, cfg: PrivacyConfig,
            rng: np.random.Generator, proba: Array | None = None) -> Payload:  # fmt: skip
    """What the device sends for windows ``idx`` (in stream order)."""
    seg = c.segment[idx]
    if arch == "centralized":
        return privacy.raw(c.signals[idx], seg, level, cal, cfg, rng)
    if arch == "edge":
        return privacy.features(c.feats[idx], c.spec, seg, level, cal, cfg, rng)
    assert proba is not None
    return privacy.decisions(proba, seg, level, cfg, rng)


@dataclass
class FoldResult:
    arch: str
    level: int
    seed: int
    fold: int
    accuracy: float
    macro_f1: float
    bytes_per_window: float
    test_windows: int
    per_subject: dict[int, float]  # accuracy per held-out subject


def evaluate_fold(c: Corpus, train_subjects: IntArray, test_subjects: IntArray, seed: int, fold: int,
                  cfg: PrivacyConfig, levels: tuple[int, ...] = privacy.LEVELS,
                  archs: tuple[str, ...] = ARCHITECTURES, n_kernels: int = 500) -> tuple[list[FoldResult], Array]:  # fmt: skip
    """Utility of every requested cell on one fold; also returns the device's test probabilities."""
    tr = np.flatnonzero(np.isin(c.subject, train_subjects))
    te = np.flatnonzero(np.isin(c.subject, test_subjects))
    cal = Calibration.fit(c.signals[tr], c.feats[tr])
    dev = device_model(c.feats[tr], c.activity[tr], seed)
    proba_tr, proba_te = quantised_proba(dev, c.feats[tr]), quantised_proba(dev, c.feats[te])
    out = []
    for arch in archs:
        for level in levels:
            rng = np.random.default_rng([seed, fold, level, ARCHITECTURES.index(arch)])
            p_tr = payload(arch, c, tr, level, cal, cfg, rng, proba_tr)
            p_te = payload(arch, c, te, level, cal, cfg, rng, proba_te)
            if arch == "local":
                pred_units = p_te.data[:, 0].astype(np.int64)  # the cloud takes the device's decision
            else:
                model = CloudModel(p_tr.kind, seed, n_kernels).fit(p_tr, unit_labels(c.activity[tr], p_tr.units))
                pred_units = model.predict(p_te)
            pred = pred_units[p_te.units]
            y = c.activity[te]
            subj = c.subject[te]
            per = {int(s): float((pred[subj == s] == y[subj == s]).mean()) for s in np.unique(subj)}
            out.append(FoldResult(arch, level, seed, fold, float((pred == y).mean()),
                                  float(f1_score(y, pred, average="macro")), p_te.bytes_per_window(), len(te), per))  # fmt: skip
    return out, proba_te


def enrolment_split(c: Corpus, share: float, seed: int) -> NDArray[np.bool_]:
    """Enrolment windows: for each subject and activity, a ``share`` of its segments, chosen at random."""
    rng = np.random.default_rng(seed)
    enrol = np.zeros(len(c.subject), dtype=bool)
    for s in np.unique(c.subject):
        for a in np.unique(c.activity):
            segs = np.unique(c.segment[(c.subject == s) & (c.activity == a)])
            if len(segs) < 2:
                continue
            k = min(len(segs) - 1, max(1, round(share * len(segs))))
            enrol |= np.isin(c.segment, rng.choice(segs, k, replace=False))
    return enrol


@dataclass
class ExposureResult:
    arch: str
    level: int
    seed: int
    share: float
    attacker: str
    payload_accuracy: float
    payload_advantage: float
    segment_accuracy: float
    n_subjects: int
    attack_payloads: int


def exposure(c: Corpus, arch: str, level: int, seed: int, cfg: PrivacyConfig, proba: Array | None, share: float = 0.5,
             family: str = "strong", n_kernels: int = 500) -> ExposureResult:  # fmt: skip
    """Re-identification by an enrolled adversary at the cloud, from this cell's payloads."""
    enrol = enrolment_split(c, share, seed)
    en, at = np.flatnonzero(enrol), np.flatnonzero(~enrol)
    cal = Calibration.fit(c.signals[en], c.feats[en])
    rng = np.random.default_rng([seed, 7919, level, ARCHITECTURES.index(arch)])
    p_en = payload(arch, c, en, level, cal, cfg, rng, None if proba is None else proba[en])
    p_at = payload(arch, c, at, level, cal, cfg, rng, None if proba is None else proba[at])
    model = CloudModel(p_en.kind, seed, n_kernels, family).fit(p_en, unit_labels(c.subject[en], p_en.units))
    pred = model.predict(p_at)
    truth = unit_labels(c.subject[at], p_at.units)
    acc = float((pred == truth).mean())
    n = len(np.unique(c.subject))
    # per recording segment: majority vote over the segment's payloads
    unit_seg = unit_labels(c.segment[at], p_at.units)
    votes = [np.argmax(np.bincount(pred[unit_seg == s])) == truth[unit_seg == s][0] for s in np.unique(unit_seg)]
    return ExposureResult(arch, level, seed, share, family, acc, (acc - 1 / n) / (1 - 1 / n), float(np.mean(votes)), n,
                          p_at.n_units)  # fmt: skip
