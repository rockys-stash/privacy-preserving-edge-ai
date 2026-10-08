"""Find the latest completed run of an experiment."""

from __future__ import annotations

from pathlib import Path

from edgebench.experiments.common import ROOT


def latest(experiment: str, results: Path | None = None) -> Path:
    base = (results or ROOT / "results") / experiment
    marker = base / "LATEST"
    if not marker.is_file():
        raise FileNotFoundError(f"{experiment} has not been run yet (make {experiment.split('_')[0]})")
    return base / marker.read_text().strip()
