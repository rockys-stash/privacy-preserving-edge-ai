"""E5: does the streaming pipeline reproduce the batch results?

Models are trained on the dataset's 21 training subjects. Every recording segment of the 9 test
subjects is then replayed sample by sample through device, link and cloud, and each window's
streamed decision is compared with the batch decision for the same window. Replays are also saved
for the console's Lab view."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from edgebench import bench, data, privacy
from edgebench.bench import CloudModel, device_model, payload, quantised_proba, unit_labels
from edgebench.experiments.common import result_dir, table
from edgebench.measure import LinkProfile
from edgebench.privacy import Calibration, PrivacyConfig
from edgebench.stream import replay

DESCRIPTION = "Streaming replay of the test subjects through device, link and cloud, compared with batch decisions."


def run(cfg: dict[str, Any], results: Path | None = None) -> Path:
    c = bench.corpus()
    d = data.load_all()
    segs = data.segments(d)
    train_subj = np.unique(data.load("train").subject)
    tr = np.flatnonzero(np.isin(c.subject, train_subj))
    te = np.flatnonzero(~np.isin(c.subject, train_subj))
    test_segs = [(k, s) for k, s in enumerate(segs) if s.subject not in set(train_subj.tolist())]
    pc = PrivacyConfig(**cfg.get("privacy", {}))
    link = LinkProfile(**cfg["link"])
    cal = Calibration.fit(c.signals[tr], c.feats[tr])
    dev = device_model(c.feats[tr], c.activity[tr], cfg["seed"])
    proba_tr, proba_te = quantised_proba(dev, c.feats[tr]), quantised_proba(dev, c.feats[te])
    with result_dir("e5_streaming", cfg, DESCRIPTION, results) as (rd, s):
        rows, lab = [], {}
        lab_subject = int(cfg["lab_subject"])
        for arch in bench.ARCHITECTURES:
            for level in cfg["levels"]:
                rng = np.random.default_rng(cfg["seed"])
                p_tr = payload(arch, c, tr, level, cal, pc, rng, proba_tr)
                cloud = (
                    None
                    if arch == "local"
                    else CloudModel(p_tr.kind, cfg["seed"], cfg["n_kernels"]).fit(
                        p_tr, unit_labels(c.activity[tr], p_tr.units)
                    )
                )
                p_te = payload(arch, c, te, level, cal, pc, rng, proba_te)
                batch = (p_te.data[:, 0].astype(np.int64) if arch == "local" else cloud.predict(p_te))[p_te.units]  # type: ignore[union-attr]
                events = replay(
                    [sg for _, sg in test_segs],
                    [k for k, _ in test_segs],
                    arch,
                    level,
                    cal,
                    pc,
                    dev,
                    cloud,
                    link,
                    cfg["seed"],
                )
                streamed = np.array([-1 if e.decision is None else e.decision for e in events])
                lat = np.array([e.arrived_at_s - e.closed_at_s for e in events if e.arrived_at_s is not None])
                deterministic = level < 3  # L3 and L4 (cumulative) draw fresh noise, so only their accuracy can match
                rows.append({"cell": f"{arch}/L{level}", "arch": arch, "level": level, "windows_batch": len(te), "windows_streamed": len(events),
                             "agreement": float((streamed == batch).mean()) if len(events) == len(te) else None,
                             "accuracy_batch": float((batch == c.activity[te]).mean()),
                             "accuracy_streamed": float((streamed == c.activity[te]).mean()) if len(events) == len(te) else None,
                             "median_latency_s": float(np.median(lat)), "p95_latency_s": float(np.quantile(lat, 0.95)),
                             "deterministic": deterministic})  # fmt: skip
                lab_events = [
                    asdict(e)
                    for e, (k, sg) in zip(events, _owners(test_segs), strict=True)
                    if sg.subject == lab_subject
                ]
                lab[f"{arch}/L{level}"] = lab_events
        rd.write_json("consistency.json", rows)
        rd.write_json("lab_replay.json", {"subject": lab_subject, "segments": _lab_segments(segs, lab_subject),
                                          "activities": list(data.ACTIVITIES), "cells": lab})  # fmt: skip
        s.title = "E5 Streaming replay"
        s.question = "Do decisions made sample by sample through device, link and cloud equal the batch decisions?"
        s.tables.append(table("consistency", "Streamed against batch decisions for the 9 test subjects",
            [("cell", "Cell", "l"), ("windows_batch", "Windows (batch)", "r"), ("windows_streamed", "Windows (streamed)", "r"),
             ("agreement", "Same decision", "r"), ("accuracy_batch", "Accuracy batch", "r"), ("accuracy_streamed", "Accuracy streamed", "r"),
             ("median_latency_s", "Median latency s", "r"), ("p95_latency_s", "p95 latency s", "r")], rows))  # fmt: skip
        det = [r for r in rows if r["deterministic"]]
        s.findings.append(
            f"{sum(1 for r in det if r['agreement'] == 1.0)}/{len(det)} deterministic cells streamed exactly the batch decisions."
        )
        s.notes.append(
            f"Latency uses the modelled link '{link.name}' and the cloud's measured compute is not included here (see E4)."
        )
    return rd.path


def _owners(test_segs: list[tuple[int, data.Segment]]) -> list[tuple[int, data.Segment]]:
    """The segment each streamed window belongs to, in stream order."""
    return [(k, sg) for k, sg in test_segs for _ in range(sg.n_windows)]


def _lab_segments(segs: list[data.Segment], subject: int) -> list[dict[str, Any]]:
    """Downsampled traces of one subject's segments for the Lab view (every 2nd sample, 3 decimals)."""
    out = []
    for k, sg in enumerate(segs):
        if sg.subject == subject:
            out.append({"segment": k, "activity": int(sg.activity[0]), "n_windows": sg.n_windows,
                        "samples": np.round(sg.samples[:, ::2], 3).tolist()})  # fmt: skip
    return out


__all__ = ["privacy", "run"]
