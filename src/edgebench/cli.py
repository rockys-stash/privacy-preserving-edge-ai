"""Command line: fetch the data, run experiments, build the report, serve the console."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from edgebench.experiments.common import ROOT, load_config

EXPERIMENTS = {
    "e1": "e1_frontier",
    "e2": "e2_adversary",
    "e3": "e3_epsilon",
    "e4": "e4_cost",
    "e5": "e5_streaming",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="edgebench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the UCI HAR archive and check its SHA-256")
    e = sub.add_parser("experiment", help="run one experiment (e1..e5) or all of them in order")
    e.add_argument("name", choices=[*EXPERIMENTS, "all"])
    e.add_argument(
        "--configs", type=Path, default=ROOT / "configs", help="config directory (configs/smoke for a smoke run)"
    )
    e.add_argument("--results", type=Path, help="results root (default results/)")
    e.add_argument("--workers", type=int, help="override the config's worker count")
    r = sub.add_parser("report", help="write docs/REPORT.md and docs/figures/ from the latest runs")
    r.add_argument("--results", type=Path)
    r.add_argument("--out", type=Path, default=ROOT / "docs")
    s = sub.add_parser("serve", help="serve the console and API")
    s.add_argument("--host", default=os.environ.get("EDGEBENCH_HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("EDGEBENCH_PORT", "8000")))
    s.add_argument("--results", type=Path)
    a = p.parse_args(argv)

    if a.cmd == "fetch":
        from edgebench import data

        print(data.fetch())
        return 0
    if a.cmd == "experiment":
        names = list(EXPERIMENTS) if a.name == "all" else [a.name]
        for n in names:
            module = EXPERIMENTS[n]
            cfg = load_config(a.configs / f"{module}.yaml")
            if a.workers is not None and "workers" in cfg:
                cfg["workers"] = a.workers
            exp = importlib.import_module(f"edgebench.experiments.{module}")
            print(f"{n}: running", file=sys.stderr, flush=True)
            print(exp.run(cfg, a.results))
        return 0
    if a.cmd == "report":
        from edgebench.report import write_report

        print(write_report(a.results, a.out))
        return 0
    if a.cmd == "serve":
        import uvicorn

        from edgebench.api import create_app

        uvicorn.run(create_app(a.results, host=a.host), host=a.host, port=a.port, log_level="info")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
