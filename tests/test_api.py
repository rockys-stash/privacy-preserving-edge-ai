"""API behaviour on a synthetic results tree (invented numbers, labelled as such; not results)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edgebench.api import create_app

RUN = "20260101T000000Z-abc1234"


def make_results(root: Path) -> Path:
    d = root / "e1_frontier" / RUN
    d.mkdir(parents=True)
    (root / "e1_frontier" / "LATEST").write_text(RUN + "\n")
    prov = {"experiment": "e1_frontier", "run_id": RUN, "commit": "abc1234", "dirty": False, "runtime_seconds": 1,
            "cpus": 1, "created_at": "2026-01-01T00:00:00+00:00", "subjects": "fixture"}  # fmt: skip
    (d / "provenance.json").write_text(json.dumps(prov))
    (d / "summary.json").write_text(
        json.dumps({"title": "FIXTURE", "question": "q", "findings": [], "tables": [], "notes": []})
    )
    (d / "cells.json").write_text(json.dumps([{"cell": "edge/L0", "note": "synthetic fixture"}]))
    (d / "baselines.json").write_text(
        json.dumps({"majority_accuracy": 0.0, "chance_reidentification": 0.0, "subjects": 0})
    )
    return root


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("EDGEBENCH_API_TOKEN", raising=False)
    return TestClient(create_app(make_results(tmp_path), static_dir=tmp_path / "nodist"))


def test_reads_latest_runs_and_reports_pending(client: TestClient) -> None:
    items = client.get("/api/experiments").json()["items"]
    status = {i["experiment"]: i["status"] for i in items}
    assert status["e1_frontier"] == "complete" and status["e5_streaming"] == "pending"
    fr = client.get("/api/frontier").json()
    assert fr["cells"][0]["cell"] == "edge/L0" and fr["costs"] is None
    assert client.get("/api/lab").status_code == 404
    assert client.get("/api/experiments/e2_adversary").status_code == 404


def test_unknown_experiment_and_traversal_are_rejected(client: TestClient) -> None:
    assert client.get("/api/experiments/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/experiments/e9_nothing").status_code == 404
    assert client.get("/api/nope").status_code == 404


def test_api_is_read_only(client: TestClient) -> None:
    assert client.post("/api/experiments").status_code == 405
    assert client.delete("/api/frontier").status_code == 405


def test_security_headers(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_token_required_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGEBENCH_API_TOKEN", "t0ken-for-tests")
    c = TestClient(create_app(make_results(tmp_path)))
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/frontier").status_code == 401
    assert c.get("/api/frontier", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/frontier", headers={"Authorization": "Bearer t0ken-for-tests"}).status_code == 200


def test_refuses_public_bind_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDGEBENCH_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        create_app(tmp_path, host="0.0.0.0")
    create_app(tmp_path, host="::1")


def test_rate_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGEBENCH_RATE_PER_SEC", "0.001")
    c = TestClient(create_app(make_results(tmp_path)))
    codes = [c.get("/api/health").status_code for _ in range(130)]
    assert codes.count(429) >= 10 and codes[0] == 200
