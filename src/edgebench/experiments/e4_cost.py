"""E4: what each cell costs. Bytes, operations and measured wall time per window; modelled latency
and a device energy proxy under stated device and link profiles, with the radio's energy per byte
swept over two orders of magnitude to find where the cheapest architecture changes."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from edgebench import bench, data
from edgebench.experiments.common import Summary, result_dir, table
from edgebench.measure import CostRow, DeviceProfile, LinkProfile, costs, energy_j_per_window, latency_s
from edgebench.privacy import PrivacyConfig

DESCRIPTION = "Cost: bytes, operations, measured compute, modelled latency and energy proxy."


def summarise(s: Summary, rows: list[CostRow], devices: list[DeviceProfile], links: list[LinkProfile],
              factors: list[float]) -> list[dict[str, Any]]:  # fmt: skip
    s.title = "E4 Cost"
    s.question = (
        "What does each cell cost in bandwidth, computation, latency and energy, and when is fully local cheapest?"
    )
    s.tables.append(table("measured", "Per window: bytes on the wire, operations (analytic) and single-window wall time (measured here)",
        [("cell", "Cell", "l"), ("bytes_per_window", "Bytes / window", "r"), ("zlib_bytes_per_window", "zlib bytes / window", "r"),
         ("uplink_bps", "Uplink bit/s", "r"), ("device_ops", "Device ops", "r"), ("cloud_ops", "Cloud ops", "r"),
         ("device_ms_measured", "Device ms (here)", "r"), ("cloud_ms_measured", "Cloud ms (here)", "r"), ("block_wait_s", "Block wait s", "r")],
        [{"cell": f"{r.arch}/L{r.level}", **asdict(r)} for r in rows]))  # fmt: skip
    modelled = []
    for d in devices:
        for ln in links:
            for r in rows:
                modelled.append({"cell": f"{r.arch}/L{r.level}", "arch": r.arch, "level": r.level, "device": d.name, "link": ln.name,
                                 "latency_s": latency_s(r, d, ln), "energy_uj": energy_j_per_window(r, d, ln) * 1e6})  # fmt: skip
    s.tables.append(table("modelled", "Modelled latency (window closes -> decision at the cloud) and device energy proxy per window",
        [("cell", "Cell", "l"), ("device", "Device", "l"), ("link", "Link", "l"), ("latency_s", "Latency s", "r"),
         ("energy_uj", "Energy proxy uJ", "r")], modelled))  # fmt: skip
    # Break-even: cheapest cell on energy among the L0 cells, as the radio's cost per byte is scaled.
    sweep = []
    for d in devices:
        for ln in links:
            for f in factors:
                scaled = LinkProfile(ln.name, ln.rate_bps, ln.one_way_s, ln.joule_per_byte * f)
                e = {r.arch: energy_j_per_window(r, d, scaled) for r in rows if r.level == 0}
                sweep.append({"device": d.name, "link": ln.name, "radio_factor": f, **{f"{a}_uj": v * 1e6 for a, v in e.items()},
                              "cheapest": min(e, key=lambda a: e[a])})  # fmt: skip
    s.tables.append(table("breakeven", "Cheapest L0 architecture on device energy as radio energy per byte is scaled",
        [("device", "Device", "l"), ("link", "Link", "l"), ("radio_factor", "Radio x", "r"), ("centralized_uj", "Centralized uJ", "r"),
         ("edge_uj", "Edge uJ", "r"), ("local_uj", "Local uJ", "r"), ("cheapest", "Cheapest", "l")], sweep))  # fmt: skip
    flips = sorted({(r["device"], r["link"]) for r in sweep if r["cheapest"] != "local"})
    s.findings.append("Fully local is the cheapest L0 architecture on energy in every swept setting." if not flips else
                      "Fully local is not cheapest on energy for: " + "; ".join(f"{d} over {ln}" for d, ln in flips) + " (at some radio factors).")  # fmt: skip
    s.notes.append(
        "Device ops/s, energy per operation and radio energy per byte are stated assumptions (configs/e4_cost.yaml), not measurements. Wall times were measured on this machine and are shown for scale only."
    )
    return modelled


def run(cfg: dict[str, Any], results: Path | None = None) -> Path:
    c = bench.corpus()
    train = np.unique(data.load("train").subject)
    devices = [DeviceProfile(**d) for d in cfg["devices"]]
    links = [LinkProfile(**ln) for ln in cfg["links"]]
    with result_dir("e4_cost", cfg, DESCRIPTION, results) as (rd, s):
        rows = costs(c, PrivacyConfig(**cfg.get("privacy", {})), cfg["seed"], train, cfg["n_kernels"], cfg["reps"])
        rd.write_json("costs.json", [asdict(r) for r in rows])
        modelled = summarise(s, rows, devices, links, cfg["radio_factors"])
        rd.write_json("modelled.json", modelled)
    return rd.path
