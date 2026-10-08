"""E3: what formal local differential privacy costs. Sweep epsilon for the two cells that carry a
guarantee (edge L3: Laplace on the clipped feature vector; local L3: randomised response)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from edgebench import bench, privacy
from edgebench.experiments import parallel
from edgebench.experiments.common import Summary, result_dir, table
from edgebench.experiments.latest import latest
from edgebench.privacy import PrivacyConfig

DESCRIPTION = "Local differential privacy: accuracy and re-identification against epsilon."


def _cv_job(eps: float, seed: int, fold: int, k: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    c = bench.corpus()
    test = bench.subject_folds(c.subject, k, seed)[fold]
    train = np.setdiff1d(np.unique(c.subject), test)
    pc = PrivacyConfig(**{**cfg, "feature_epsilon": eps, "label_epsilon": eps})
    res, _ = bench.evaluate_fold(c, train, test, seed, fold, pc, levels=(3,), archs=("edge", "local"))
    return [{**asdict(r), "epsilon": eps} for r in res]


def _exp_job(eps: float, arch: str, seed: int, cfg: dict[str, Any], proba: np.ndarray | None) -> dict[str, Any]:
    pc = PrivacyConfig(**{**cfg, "feature_epsilon": eps, "label_epsilon": eps})
    return {**asdict(bench.exposure(bench.corpus(), arch, 3, seed, pc, proba)), "epsilon": eps}


def rr_expected(acc: float, eps: float, k: int = privacy.N_CLASSES) -> float:
    """Expected accuracy after k-ary randomised response, given the accuracy of the labels it perturbs."""
    keep = math.exp(eps) / (math.exp(eps) + k - 1)
    return acc * keep + (1 - acc) * (1 - keep) / (k - 1)


def summarise(s: Summary, cv: list[dict[str, Any]], ex: list[dict[str, Any]], smoothed_acc: float | None) -> None:
    s.title = "E3 Local differential privacy"
    s.question = "How fast does accuracy fall as epsilon shrinks, for features and for labels?"
    rows = []
    for eps in sorted({r["epsilon"] for r in cv}):
        for arch in ("edge", "local"):
            a = [r["accuracy"] for r in cv if r["epsilon"] == eps and r["arch"] == arch]
            e = [r["payload_accuracy"] for r in ex if r["epsilon"] == eps and r["arch"] == arch]
            row = {"epsilon": eps, "arch": arch, "accuracy": float(np.mean(a)), "accuracy_sd": float(np.std(a)),
                   "reidentification": float(np.mean(e)) if e else None}  # fmt: skip
            if arch == "local" and smoothed_acc is not None:
                row["randomised_response_expected"] = rr_expected(smoothed_acc, eps)
            rows.append(row)
    s.tables.append(table("sweep", "Accuracy (mean and sd over folds) and re-identification per payload, by epsilon per payload",
        [("epsilon", "Epsilon", "r"), ("arch", "Architecture", "l"), ("accuracy", "Accuracy", "r"), ("accuracy_sd", "sd", "r"),
         ("randomised_response_expected", "Expected (randomised response)", "r"), ("reidentification", "Re-identification", "r")], rows))  # fmt: skip
    for arch in ("edge", "local"):
        mine = [r for r in rows if r["arch"] == arch]
        s.findings.append(f"{arch}: accuracy {mine[0]['accuracy']:.3f} at epsilon {mine[0]['epsilon']:g}, "
                          f"{mine[-1]['accuracy']:.3f} at epsilon {mine[-1]['epsilon']:g}.")  # fmt: skip
    s.notes.append(
        "Epsilon is per payload (one window, or one block at L4). A person sends one payload every 1.28 s, so over time the guarantee composes: k payloads give k x epsilon."
    )


def run(cfg: dict[str, Any], results: Path | None = None) -> Path:
    seed, k, workers = cfg["seed"], cfg["folds"], cfg.get("workers", 4)
    pcfg = cfg.get("privacy", {})
    e1 = latest("e1_frontier", results)
    proba = np.load(e1 / f"oof_proba_seed{seed}.npy")
    cells = json.loads((e1 / "cells.json").read_text())
    smoothed = next((c["accuracy"] for c in cells if c["cell"] == "local/L2"), None)
    with result_dir("e3_epsilon", cfg, DESCRIPTION, results) as (rd, s):
        cv = [
            r
            for rows in parallel.run(
                _cv_job, [(e, seed, f, k, pcfg) for e in cfg["epsilons"] for f in range(k)], workers
            )
            for r in rows
        ]
        ex = parallel.run(
            _exp_job,
            [(e, a, seed, pcfg, proba if a == "local" else None) for e in cfg["epsilons"] for a in ("edge", "local")],
            workers,
        )
        rd.write_json("cv.json", cv)
        rd.write_json("exposure.json", ex)
        summarise(s, cv, ex, smoothed)
    return rd.path
