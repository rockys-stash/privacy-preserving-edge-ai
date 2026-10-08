"""E2: does the exposure ranking depend on how strong the adversary is?

The re-identification attack is repeated with less and more enrolment data per person and with a
weaker (linear) model family. If the ordering of cells by exposure is stable, E1's conclusions do
not hinge on one attacker."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau

from edgebench import bench, privacy
from edgebench.experiments import parallel
from edgebench.experiments.common import Summary, result_dir, table
from edgebench.experiments.latest import latest
from edgebench.privacy import PrivacyConfig

DESCRIPTION = "Adversary strength: enrolment share and model family; stability of the exposure ranking."


def _job(arch: str, level: int, seed: int, share: float, family: str, cfg: dict[str, Any], proba: np.ndarray | None,
         n_kernels: int) -> dict[str, Any]:  # fmt: skip
    return asdict(
        bench.exposure(bench.corpus(), arch, level, seed, PrivacyConfig(**cfg), proba, share, family, n_kernels)
    )


def summarise(s: Summary, rows: list[dict[str, Any]], ref: tuple[float, str]) -> None:
    s.title = "E2 Adversary strength"
    s.question = "Does the ordering of cells by exposure hold for weaker and stronger adversaries?"
    cells = [f"{a}/L{lv}" for a in bench.ARCHITECTURES for lv in privacy.LEVELS]
    settings = sorted({(r["share"], r["attacker"]) for r in rows})

    def vec(share: float, family: str) -> list[float]:
        m = {
            f"{r['arch']}/L{r['level']}": r["payload_accuracy"]
            for r in rows
            if r["share"] == share and r["attacker"] == family
        }
        return [m[c] for c in cells]

    base = vec(*ref)
    out = []
    for share, family in settings:
        v = vec(share, family)
        tau = float(kendalltau(base, v).statistic)
        out.append({"setting": f"{family}, enrol {share:.0%}", "share": share, "attacker": family, "kendall_tau": tau,
                    **{c: x for c, x in zip(cells, v, strict=True)}})  # fmt: skip
    cols = [("setting", "Adversary", "l"), ("kendall_tau", "Kendall tau vs reference", "r")] + [
        (c, c, "r") for c in cells
    ]
    s.tables.append(table("settings", "Re-identification accuracy per payload, by adversary", cols, out))
    taus = [r["kendall_tau"] for r in out if (r["share"], r["attacker"]) != ref]
    s.findings.append(f"Kendall tau of the exposure ranking against the reference adversary ({ref[1]}, {ref[0]:.0%} enrolment): "
                      f"{min(taus):.2f} to {max(taus):.2f} across {len(taus)} other adversaries.")  # fmt: skip


def run(cfg: dict[str, Any], results: Path | None = None) -> Path:
    seed, n_kernels, workers = cfg["seed"], cfg["n_kernels"], cfg.get("workers", 4)
    pcfg = cfg.get("privacy", {})
    proba = np.load(latest("e1_frontier", results) / f"oof_proba_seed{seed}.npy")
    with result_dir("e2_adversary", cfg, DESCRIPTION, results) as (rd, s):
        jobs = [(a, lv, seed, sh, fam, pcfg, proba if a == "local" else None, n_kernels)
                for sh in cfg["shares"] for fam in cfg["families"] for a in bench.ARCHITECTURES for lv in privacy.LEVELS]  # fmt: skip
        rows = parallel.run(_job, jobs, workers)
        rd.write_json("exposure.json", rows)
        summarise(s, rows, (cfg["reference_share"], cfg["reference_family"]))
    return rd.path
