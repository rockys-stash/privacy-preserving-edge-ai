"""The written report: docs/generated/results.md and docs/figures/*.svg, generated from the latest runs only.

Nothing here computes a result. Every number and every point in a figure is read from a run
directory's JSON, and each section ends with that run's id, commit and runtime.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edgebench.experiments.common import ROOT
from edgebench.experiments.latest import latest

ORDER = ("e1_frontier", "e2_adversary", "e3_epsilon", "e4_cost", "e5_streaming")
ARCH_COLOUR = {"centralized": "#2a78d6", "edge": "#eb6834", "local": "#1baf7a"}  # validated, light theme
ARCH_LABEL = {"centralized": "Centralized", "edge": "Edge", "local": "Fully local"}
INK, MUTED, RULE = "#1f2023", "#5f6168", "#d9d9d6"

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#ececea",
    "grid.linewidth": 0.6, "svg.fonttype": "none", "svg.hashsalt": "edgebench", "lines.linewidth": 2,
})  # fmt: skip


def load_run(experiment: str, results: Path | None) -> dict[str, Any] | None:
    """A run's summary, provenance and raw JSON files, or None if it has not been run."""
    try:
        d = latest(experiment, results)
    except FileNotFoundError:
        return None
    out: dict[str, Any] = {"dir": d}
    for f in d.glob("*.json"):
        out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return out


def fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        a = abs(v)
        if a != 0 and (a >= 1e5 or a < 1e-3):
            return f"{v:.3g}"
        return f"{v:,.3f}" if a < 100 else f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return "" if v is None else str(v)


def md_table(t: dict[str, Any], max_rows: int | None = None) -> list[str]:
    cols = t["columns"]
    rows = t["rows"] if max_rows is None else t["rows"][:max_rows]
    out = [f"*{t['caption']}*", "", "| " + " | ".join(c["label"] for c in cols) + " |",
           "|" + "|".join("---:" if c["align"] == "r" else "---" for c in cols) + "|"]  # fmt: skip
    out += ["| " + " | ".join(fmt(r.get(c["key"])) for c in cols) + " |" for r in rows]
    if max_rows is not None and len(t["rows"]) > max_rows:
        out.append(f"\n{len(t['rows']) - max_rows} more rows in the run's summary.json.")
    return [*out, ""]


def section(run: dict[str, Any], figures: list[tuple[str, str]], skip_tables: tuple[str, ...] = ()) -> list[str]:
    s, p = run["summary"], run["provenance"]
    lines = [f"## {s['title']}", "", f"**Question.** {s['question']}", ""]
    lines += ["**Findings.**", ""] + [f"- {f}" for f in s["findings"]] + [""]
    for name, alt in figures:
        lines += [f"![{alt}](../figures/{name})", ""]
    for t in s["tables"]:
        if t["id"] not in skip_tables:
            lines += md_table(t)
    lines += [f"- {n}" for n in s["notes"]] + ([""] if s["notes"] else [])
    dirty = " (uncommitted changes)" if p["dirty"] else ""
    lines += [f"<sub>Run `{p['run_id']}`, commit `{p['commit'][:7]}`{dirty}, {p['runtime_seconds']:.0f} s on "
              f"{p['cpus']} CPUs, subjects: {p.get('subjects', 'all 30')}.</sub>", ""]  # fmt: skip
    return lines


def _label(ax: Any, x: float, y: float, text: str, colour: str) -> None:
    ax.annotate(text, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)


def fig_frontier(cells: list[dict[str, Any]], findings: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    n = cells[0]["subjects"]
    ax.axvline(1 / n, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.annotate("chance re-identification", (1 / n, 0.02), xytext=(4, 0), textcoords="offset points", fontsize=8,
                color=MUTED, xycoords=("data", "axes fraction"))  # fmt: skip
    front = sorted((c for c in cells if c["pareto"]), key=lambda c: c["exposure"])
    if front:
        ax.plot([c["exposure"] for c in front], [c["accuracy"] for c in front], color=MUTED, lw=1, zorder=1)
    for arch, colour in ARCH_COLOUR.items():
        mine = [c for c in cells if c["arch"] == arch]
        for c in mine:
            ax.plot([c["exposure_lo"], c["exposure_hi"]], [c["accuracy"]] * 2, color=colour, lw=1.2, zorder=2)
            ax.plot([c["exposure"]] * 2, [c["accuracy_lo"], c["accuracy_hi"]], color=colour, lw=1.2, zorder=2)
            ax.scatter(c["exposure"], c["accuracy"], s=36, color=colour, edgecolor="white", linewidth=1.5, zorder=3,
                       label=ARCH_LABEL[arch] if c["level"] == 0 else None)  # fmt: skip
            if c["pareto"] or c["level"] == 0:  # selective labels; every cell is in the table below
                _label(ax, c["exposure"], c["accuracy"], f"{ARCH_LABEL[arch]} L{c['level']}", colour)
    ax.set_xlabel("Re-identification accuracy per payload (lower is more private)")
    ax.set_ylabel("Activity accuracy, unseen subjects")
    ax.set_title("Accuracy against exposure, 15 cells (labelled: L0 and the Pareto set)")
    ax.grid(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_bandwidth(cells: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arch, colour in ARCH_COLOUR.items():
        mine = sorted((c for c in cells if c["arch"] == arch), key=lambda c: c["level"])
        for c in mine:
            ax.plot([c["bytes_per_window"]] * 2, [c["accuracy_lo"], c["accuracy_hi"]], color=colour, lw=1.2)
            ax.scatter(c["bytes_per_window"], c["accuracy"], s=36, color=colour, edgecolor="white", linewidth=1.5,
                       zorder=3, label=ARCH_LABEL[arch] if c["level"] == 0 else None)  # fmt: skip
            _label(ax, c["bytes_per_window"], c["accuracy"], f"L{c['level']}", colour)
    ax.set_xscale("log")
    ax.set_xlabel("Bytes leaving the device per window (log scale)")
    ax.set_ylabel("Activity accuracy, unseen subjects")
    ax.set_title("Accuracy against bandwidth")
    ax.grid(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_epsilon(sweep: list[dict[str, Any]], majority: float | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for arch in ("edge", "local"):
        rows = sorted((r for r in sweep if r["arch"] == arch), key=lambda r: r["epsilon"])
        xs = [r["epsilon"] for r in rows]
        ax.plot(xs, [r["accuracy"] for r in rows], color=ARCH_COLOUR[arch], marker="o", ms=5,
                label=f"{ARCH_LABEL[arch]} (measured)")  # fmt: skip
        if arch == "local":
            exp = [r["randomised_response_expected"] for r in rows]
            ax.plot(xs, exp, color=ARCH_COLOUR[arch], lw=1, ls=(0, (3, 3)), label="Fully local (expected under RR)")
    if majority is not None:
        ax.axhline(majority, color=MUTED, lw=1, ls=(0, (1, 2)))
        ax.annotate("majority class", (0.01, majority), xytext=(0, 3), textcoords="offset points", fontsize=8,
                    color=MUTED, xycoords=("axes fraction", "data"))  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget epsilon per payload (log scale; smaller is more private)")
    ax.set_ylabel("Activity accuracy")
    ax.set_title("Accuracy under local differential privacy (L3)")
    ax.grid(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_breakeven(sweep: list[dict[str, Any]], device: str, out: Path) -> None:
    links = sorted({r["link"] for r in sweep if r["device"] == device})
    fig, axes = plt.subplots(1, len(links), figsize=(2.3 * len(links) + 0.6, 3.2), sharey=True)
    for ax, link in zip(axes, links, strict=True):
        rows = sorted(
            (r for r in sweep if r["device"] == device and r["link"] == link), key=lambda r: r["radio_factor"]
        )
        xs = [r["radio_factor"] for r in rows]
        for arch, colour in ARCH_COLOUR.items():
            ax.plot(xs, [r[f"{arch}_uj"] for r in rows], color=colour, label=ARCH_LABEL[arch])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(link)
        ax.set_xlabel("Radio energy x assumed")
        ax.grid(True)
    axes[0].set_ylabel("Device energy proxy per window, uJ")
    axes[-1].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(f"L0 energy proxy on a {device}", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_report(results: Path | None = None, out: Path | None = None) -> Path:
    out = out or ROOT / "docs"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    runs = {e: load_run(e, results) for e in ORDER}
    lines = ["# Measured results", "",
             "Generated by `edgebench report` from the latest run of each experiment. Every number below was "
             "produced by the code in this repository on the UCI HAR dataset; nothing was typed in by hand.", ""]  # fmt: skip
    pending = [e for e, r in runs.items() if r is None]
    if pending:
        lines += [f"**Status: pending** for {', '.join(pending)}: not run yet.", ""]
    majority = None
    if (r := runs["e1_frontier"]) is not None:
        cells = r["cells"]
        fig_frontier(cells, r["summary"]["findings"], figs / "frontier.svg")
        fig_bandwidth(cells, figs / "bandwidth.svg")
        majority = float(r["baselines"]["majority_accuracy"])
        lines += section(r, [("frontier.svg", "Accuracy against re-identification for every cell"),
                             ("bandwidth.svg", "Accuracy against bytes per window for every cell")])  # fmt: skip
    if (r := runs["e2_adversary"]) is not None:
        lines += section(r, [])
    if (r := runs["e3_epsilon"]) is not None:
        sweep = next(t for t in r["summary"]["tables"] if t["id"] == "sweep")["rows"]
        fig_epsilon(sweep, majority, figs / "epsilon.svg")
        lines += section(r, [("epsilon.svg", "Accuracy against epsilon for edge features and local labels")])
    if (r := runs["e4_cost"]) is not None:
        be = next(t for t in r["summary"]["tables"] if t["id"] == "breakeven")["rows"]
        devices = sorted({b["device"] for b in be})
        for d in devices:
            fig_breakeven(be, d, figs / f"breakeven_{d}.svg")
        lines += section(r, [(f"breakeven_{d}.svg", f"Energy proxy against radio energy, {d}") for d in devices],
                         skip_tables=("modelled",))  # fmt: skip
        lines += ["The full modelled latency and energy table (every cell, device and link) is in the run's "
                  "`modelled.json`.", ""]  # fmt: skip
    if (r := runs["e5_streaming"]) is not None:
        lines += section(r, [])
    path = out / "generated" / "results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
