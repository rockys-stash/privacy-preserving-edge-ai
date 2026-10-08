"""E1: the frontier grid. Every architecture at every privacy level: accuracy on unseen people,
exposure to an enrolled re-identification adversary, and bytes on the wire."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from edgebench import bench, data, privacy
from edgebench.experiments import parallel
from edgebench.experiments.common import Summary, result_dir, table
from edgebench.privacy import PrivacyConfig

DESCRIPTION = "Frontier grid: 3 architectures x 5 privacy levels; accuracy, exposure and bytes per window."


OFFICIAL_FOLD = 99  # fold id of the dataset's own 21/9 split (seeds the privacy noise)


def _fold_job(
    seed: int, fold: int, k: int, cfg: dict[str, Any], n_kernels: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    c = bench.corpus()
    folds = bench.subject_folds(c.subject, k, seed)
    test = folds[fold]
    train = np.setdiff1d(np.unique(c.subject), test)
    res, proba = bench.evaluate_fold(c, train, test, seed, fold, PrivacyConfig(**cfg), n_kernels=n_kernels)
    te = np.flatnonzero(np.isin(c.subject, test))
    majority = float((c.activity[te] == np.argmax(np.bincount(c.activity[np.isin(c.subject, train)]))).mean())
    return [asdict(r) for r in res], {"seed": seed, "fold": fold, "test_index": te.tolist(), "proba": proba.tolist(),
                                       "majority_accuracy": majority}  # fmt: skip


def _official_job(cfg: dict[str, Any], n_kernels: int) -> list[dict[str, Any]]:
    c = bench.corpus()
    train = np.unique(data.load("train").subject)
    test = np.setdiff1d(np.unique(c.subject), train)
    res, _ = bench.evaluate_fold(c, train, test, 0, OFFICIAL_FOLD, PrivacyConfig(**cfg), n_kernels=n_kernels)
    return [asdict(r) for r in res]


def _exposure_job(arch: str, level: int, seed: int, cfg: dict[str, Any], proba: np.ndarray | None, share: float,
                  family: str, n_kernels: int) -> dict[str, Any]:  # fmt: skip
    c = bench.corpus()
    return asdict(bench.exposure(c, arch, level, seed, PrivacyConfig(**cfg), proba, share, family, n_kernels))


def oof_proba(fold_meta: list[dict[str, Any]], seed: int, n: int) -> np.ndarray:
    """Out-of-fold device probabilities: every window scored by the model that never saw its subject."""
    out = np.full((n, privacy.N_CLASSES), np.nan)
    for m in fold_meta:
        if m["seed"] == seed:
            out[np.asarray(m["test_index"])] = np.asarray(m["proba"])
    assert not np.isnan(out).any()
    return out


def bootstrap_ci(values: np.ndarray, seed: int = 0, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pareto(points: list[tuple[str, float, float]]) -> list[str]:
    """Cells not dominated in (accuracy higher, exposure lower)."""
    keep = []
    for name, a, e in points:
        dominated = any((a2 >= a and e2 <= e) and (a2 > a or e2 < e) for n2, a2, e2 in points if n2 != name)
        if not dominated:
            keep.append(name)
    return keep


def summarise(s: Summary, folds: list[dict[str, Any]], official: list[dict[str, Any]], exposure: list[dict[str, Any]],
              majority: float) -> list[dict[str, Any]]:  # fmt: skip
    s.title = "E1 Frontier grid"
    s.question = "Where does each architecture and privacy level sit on accuracy, exposure and bandwidth?"
    cells = []
    base = {}
    for arch in bench.ARCHITECTURES:
        for level in privacy.LEVELS:
            fr = [f for f in folds if f["arch"] == arch and f["level"] == level]
            scores: dict[int, list[float]] = {}
            for f in fr:  # keys are ints in memory and strings after a JSON round trip
                for k, v in f["per_subject"].items():
                    scores.setdefault(int(k), []).append(v)
            subj = sorted(scores)
            per_subject = np.array([np.mean(scores[k]) for k in subj])
            acc = float(per_subject.mean())
            lo, hi = bootstrap_ci(per_subject)
            ex = [e for e in exposure if e["arch"] == arch and e["level"] == level]
            off = next(o for o in official if o["arch"] == arch and o["level"] == level)
            row = {"cell": f"{arch}/L{level}", "arch": arch, "level": level, "level_name": privacy.LEVEL_NAMES[level],
                   "accuracy": acc, "accuracy_lo": lo, "accuracy_hi": hi,
                   "macro_f1": float(np.mean([f["macro_f1"] for f in fr])),
                   "official_split_accuracy": off["accuracy"],
                   "exposure": float(np.mean([e["payload_accuracy"] for e in ex])),
                   "exposure_lo": float(np.min([e["payload_accuracy"] for e in ex])),
                   "exposure_hi": float(np.max([e["payload_accuracy"] for e in ex])),
                   "advantage": float(np.mean([e["payload_advantage"] for e in ex])),
                   "segment_reid": float(np.mean([e["segment_accuracy"] for e in ex])),
                   "bytes_per_window": float(np.mean([f["bytes_per_window"] for f in fr])),
                   "subjects": len(subj), "fold_runs": len(fr)}  # fmt: skip
            if level == 0:
                base[arch] = acc
            cells.append(row)
    for r in cells:
        # Utility loss is reported as a signed difference: negative means less accurate than the reference.
        r["delta_vs_arch_L0"] = r["accuracy"] - base[r["arch"]]
        r["delta_vs_centralized_L0"] = r["accuracy"] - base["centralized"]
    n = cells[0]["subjects"]
    # "Transmit nothing" (predict the majority class, send nothing) takes part in the Pareto comparison,
    # so a cell that is both less accurate than it and no less exposing is not on the frontier.
    # An adversary can always guess, so measured re-identification below chance is sampling noise:
    # the comparison uses max(measured, chance).
    nothing = ("nothing", majority, 1 / n)
    front = set(pareto([(r["cell"], r["accuracy"], max(r["exposure"], 1 / n)) for r in cells] + [nothing]))
    for r in cells:
        r["pareto"] = r["cell"] in front
    s.tables.append(table("grid", "Every cell (accuracy: mean over subjects and seeds, 95% bootstrap CI over subjects; exposure: re-identification accuracy per payload, mean and range over seeds)",
        [("cell", "Cell", "l"), ("level_name", "Level", "l"), ("accuracy", "Accuracy", "r"), ("accuracy_lo", "CI low", "r"),
         ("accuracy_hi", "CI high", "r"), ("delta_vs_centralized_L0", "vs centralized L0", "r"), ("exposure", "Re-identification", "r"),
         ("segment_reid", "Re-id per segment", "r"), ("bytes_per_window", "Bytes / window", "r"), ("pareto", "Pareto", "l")],
        [{**r, "pareto": "yes" if r["pareto"] else ""} for r in cells]))  # fmt: skip
    s.findings.append(f"Majority-class accuracy {majority:.3f}; chance re-identification 1/{n} = {1 / n:.3f}.")
    s.findings.append(
        "Pareto set (accuracy vs re-identification, with 'transmit nothing' as a competitor): "
        + ", ".join(r["cell"] for r in cells if r["pareto"])
        + ("; 'transmit nothing' itself is on it." if "nothing" in front else ".")
    )
    below = [r["cell"] for r in cells if r["accuracy_hi"] < majority]
    if below:
        s.findings.append("Below the majority-class floor (upper CI bound): " + ", ".join(below) + ".")
    return cells


def run(cfg: dict[str, Any], results: Path | None = None) -> Path:
    seeds, k, n_kernels, workers = cfg["seeds"], cfg["folds"], cfg["n_kernels"], cfg.get("workers", 4)
    pcfg = cfg.get("privacy", {})
    n = len(bench.corpus().subject)
    with result_dir("e1_frontier", cfg, DESCRIPTION, results) as (rd, s):
        out = parallel.run(_fold_job, [(sd, f, k, pcfg, n_kernels) for sd in seeds for f in range(k)], workers)
        folds = [r for rows, _ in out for r in rows]
        meta = [m for _, m in out]
        official = _official_job(pcfg, n_kernels)
        jobs = []
        for sd in seeds:
            proba = oof_proba(meta, sd, n)
            np.save(rd.path / f"oof_proba_seed{sd}.npy", proba)
            for arch in bench.ARCHITECTURES:
                for level in privacy.LEVELS:
                    jobs.append(
                        (
                            arch,
                            level,
                            sd,
                            pcfg,
                            proba if arch == "local" else None,
                            cfg.get("share", 0.5),
                            "strong",
                            n_kernels,
                        )
                    )
        exposure = parallel.run(_exposure_job, jobs, workers)
        majority = float(np.mean([m["majority_accuracy"] for m in meta]))
        rd.write_json("folds.json", folds)
        rd.write_json("official_split.json", official)
        rd.write_json("exposure.json", exposure)
        cells = summarise(s, folds, official, exposure, majority)
        rd.write_json("cells.json", cells)
        n = len(np.unique(bench.corpus().subject))
        rd.write_json(
            "baselines.json", {"majority_accuracy": majority, "chance_reidentification": 1 / n, "subjects": n}
        )
    return rd.path
