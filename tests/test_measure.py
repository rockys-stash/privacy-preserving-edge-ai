from edgebench.measure import CostRow, DeviceProfile, LinkProfile, energy_j_per_window, latency_s


def row(**kw: float) -> CostRow:
    base = dict(arch="edge", level=0, bytes_per_window=100.0, bytes_per_payload=100.0, zlib_bytes_per_window=90.0,
                uplink_bps=625.0, device_ops=1e6, cloud_ops=1e4, device_ms_measured=0.1, cloud_ms_measured=2.0,
                model_bytes_on_device=0,
                block_wait_s=0.0)  # fmt: skip
    base.update(kw)
    return CostRow(**base)  # type: ignore[arg-type]


DEV = DeviceProfile("d", ops_per_s=1e8, joule_per_op=1e-10)
LINK = LinkProfile("l", rate_bps=1e5, one_way_s=0.02, joule_per_byte=1e-6)


def test_latency_adds_device_link_cloud_and_block_wait() -> None:
    r = row(block_wait_s=4.48)
    expected = 1e6 / 1e8 + 100 * 8 / 1e5 + 0.02 + 0.002 + 4.48
    assert abs(latency_s(r, DEV, LINK) - expected) < 1e-12


def test_energy_is_compute_plus_radio_per_window() -> None:
    r = row(bytes_per_window=12.5, bytes_per_payload=100.0)  # an L4 payload shared by 8 windows
    assert abs(energy_j_per_window(r, DEV, LINK) - (1e6 * 1e-10 + 12.5 * 1e-6)) < 1e-15
