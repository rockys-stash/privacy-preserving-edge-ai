from pathlib import Path

from edgebench.experiments.common import load_config
from edgebench.report import fmt, md_table, write_report


def test_report_without_runs_says_pending(tmp_path: Path) -> None:
    path = write_report(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert "Status: pending" in text and "e1_frontier" in text
    assert not list((tmp_path / "docs" / "figures").glob("*.svg"))


def test_markdown_table_formatting() -> None:
    t = {"caption": "c", "columns": [{"key": "a", "label": "A", "align": "l"}, {"key": "b", "label": "B", "align": "r"}],
         "rows": [{"a": "x", "b": 0.12345}, {"a": "y", "b": None}]}  # fmt: skip
    lines = md_table(t)
    assert lines[2] == "| A | B |" and lines[3] == "|---|---:|"
    assert lines[4] == "| x | 0.123 |" and lines[5] == "| y |  |"
    assert fmt(1e-5) == "1e-05" and fmt(12345.6) == "12,345.6" and fmt(True) == "yes"


def test_every_config_loads_with_numbers_as_numbers() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    for f in [*root.glob("e*.yaml"), *(root / "smoke").glob("e*.yaml")]:
        cfg = load_config(f)
        assert isinstance(cfg["privacy"]["feature_epsilon"], float)
        for prof in cfg.get("devices", []) + cfg.get("links", []):
            assert all(isinstance(v, float) for k, v in prof.items() if k != "name"), (f, prof)
