"""Streaming replay: sensor -> device -> link -> cloud, one sample at a time on a simulated clock.

The batch benchmark transforms all windows at once. This module runs the same transformations the
way a deployment would: the sensor emits samples at 50 Hz, the device fills a 128-sample buffer and
closes a window every 64 samples, runs its stage, and puts the payload on a modelled link; the cloud
decides when the payload arrives. E5 checks that the decisions equal the batch decisions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from edgebench import privacy
from edgebench.bench import CloudModel, quantised_proba
from edgebench.data import FS, STEP, WINDOW, Segment
from edgebench.features import FeatureSpec, extract
from edgebench.measure import LinkProfile
from edgebench.privacy import Calibration, PrivacyConfig

Array = NDArray[np.float64]


@dataclass
class WindowEvent:
    """One window's trip through the pipeline, as the console's Lab view shows it."""

    window: int  # index within the segment
    segment: int
    closed_at_s: float
    truth: int
    payload_kind: str = ""
    payload_bytes: int = 0
    sent: bool = False  # False until the payload carrying this window is sent
    arrived_at_s: float | None = None
    decision: int | None = None
    preview: list[float] = field(default_factory=list)  # the first numbers of what was sent


class Device:
    """Buffers samples into windows and runs the architecture's on-device stage."""

    def __init__(
        self, arch: str, level: int, cal: Calibration, cfg: PrivacyConfig, device_model: Any, seed: int
    ) -> None:
        self.arch, self.level, self.cal, self.cfg, self.model = arch, level, cal, cfg, device_model
        self.rng = np.random.default_rng(seed)
        self.spec: FeatureSpec | None = None
        self.buffer: deque[Array] = deque(maxlen=WINDOW)
        self.samples = 0
        self.history: list[int] = []
        self.block: list[Array] = []

    def reset(self) -> None:
        """A gap in the recording: windows never span two segments."""
        self.buffer.clear()
        self.samples = 0
        self.history = []  # labels so far in this segment (local L2+)
        self.block = []  # what an aggregated payload (L4) is built from

    def push(self, sample: Array) -> Array | None:
        """Add one 9-channel sample; return a (9, 128) window when one closes."""
        self.buffer.append(sample)
        self.samples += 1
        if self.samples >= WINDOW and (self.samples - WINDOW) % STEP == 0:
            return np.stack(self.buffer, axis=1)
        return None

    def stage(self, window: Array, seg_id: int) -> privacy.Payload | None:
        """The on-device stage for one window; None while an aggregated block is filling (L4)."""
        w, segment = window[None], np.array([seg_id])
        if self.arch == "centralized":
            if self.level >= 4:  # keep the window; the block sends its middle one, transformed
                self.block.append(window)
                return None
            return privacy.raw(w, segment, self.level, self.cal, self.cfg, self.rng)
        f, spec = extract(w)
        self.spec = spec
        if self.arch == "edge":
            if self.level >= 4:  # keep clean features; the block mean is transformed once
                self.block.append(f[0])
                return None
            return privacy.features(f, spec, segment, self.level, self.cal, self.cfg, self.rng)
        proba = quantised_proba(self.model, f)
        label = int(np.argmax(proba))
        self.history.append(label)
        if self.level == 0:
            return privacy.Payload(
                "decision", np.column_stack([[label], proba]), np.array([0]), 1 + 4 * privacy.N_CLASSES
            )
        if self.level >= 2:
            recent = np.array(self.history[-self.cfg.smooth :], dtype=np.int64)
            label = int(privacy.smooth(recent, np.zeros(len(recent), dtype=np.int64), self.cfg.smooth)[-1])
        if self.level >= 4:
            self.block.append(np.array([float(label)]))
            return None
        if self.level == 3:
            label = int(privacy.randomised_response(np.array([label]), self.cfg.label_epsilon, self.rng)[0])
        return privacy.Payload("decision", np.array([[float(label)]]), np.array([0]), 1)

    def flush(self, seg_id: int) -> privacy.Payload | None:
        """Send the current block (L4), full or cut short by the end of a segment."""
        if not self.block:
            return None
        rows, self.block = np.stack(self.block), []
        segment = np.array([seg_id])
        # A one-row input forms a single block, so the batch L4 transformation applies unchanged.
        if self.arch == "centralized":
            mid = rows[(len(rows) - 1) // 2][None]
            return privacy.raw(mid, segment, 4, self.cal, self.cfg, self.rng)
        if self.arch == "edge":
            assert self.spec is not None
            return privacy.features(
                rows.mean(axis=0, keepdims=True), self.spec, segment, 4, self.cal, self.cfg, self.rng
            )
        counts = np.bincount(rows[:, 0].astype(np.int64), minlength=privacy.N_CLASSES)
        label = privacy.randomised_response(np.array([int(np.argmax(counts))]), self.cfg.label_epsilon, self.rng)
        return privacy.Payload("decision", np.array([[float(label[0])]]), np.array([0]), 1)


def replay(segments: list[Segment], seg_ids: list[int], arch: str, level: int, cal: Calibration, cfg: PrivacyConfig,
           device_model: Any, cloud: CloudModel | None, link: LinkProfile, seed: int = 0) -> list[WindowEvent]:  # fmt: skip
    """Replay segments back to back on one simulated clock; return one event per window, in order."""
    dev = Device(arch, level, cal, cfg, device_model, seed)
    events: list[WindowEvent] = []
    clock = 0.0
    for seg, sid in zip(segments, seg_ids, strict=True):
        dev.reset()
        waiting: list[WindowEvent] = []  # windows whose payload has not been sent yet (L4)
        n_in_segment = 0
        for k in range(seg.samples.shape[1]):
            clock += 1 / FS
            window = dev.push(seg.samples[:, k])
            if window is None:
                continue
            ev = WindowEvent(n_in_segment, sid, clock, int(seg.activity[n_in_segment]))
            n_in_segment += 1
            events.append(ev)
            waiting.append(ev)
            p = dev.stage(window, sid)
            if p is None and len(dev.block) >= cfg.block:
                p = dev.flush(sid)
            if p is not None:
                _deliver(p, waiting, cloud, link, clock)
                waiting = []
        if waiting:
            _deliver(dev.flush(sid), waiting, cloud, link, clock)
    return events


def _deliver(
    p: privacy.Payload | None, evs: list[WindowEvent], cloud: CloudModel | None, link: LinkProfile, clock: float
) -> None:
    if p is None:
        return
    arrive = clock + p.bytes_per_unit * 8 / link.rate_bps + link.one_way_s
    if p.kind == "decision":
        decision: int | None = int(p.data[0, 0])
    else:
        decision = int(cloud.predict(p)[0]) if cloud is not None else None
    preview = [round(float(v), 4) for v in np.ravel(p.data)[:8]]
    for e in evs:
        e.payload_kind, e.payload_bytes, e.sent, e.arrived_at_s = p.kind, p.bytes_per_unit, True, arrive
        e.decision, e.preview = decision, preview
