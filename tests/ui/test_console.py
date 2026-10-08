"""Browser tests for the Perimeter console.

The fixture runs the smoke configs (configs/smoke, 8 subjects, tiny models) for E1, E4 and E5 into a
temporary results tree, so the console is tested on real pipeline output while E2 and E3 stay
pending. Run with ``uv run pytest -m ui`` after ``npm --prefix web run build`` and ``make data``.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from edgebench.cli import main as cli
from tests.conftest import needs_data

ROOT = Path(__file__).resolve().parents[2]
AXE = ROOT / "web" / "node_modules" / "axe-core" / "axe.min.js"
pytestmark = [pytest.mark.ui, needs_data]
playwright_api = pytest.importorskip("playwright.sync_api")

VIEWPORTS = {"desktop": {"width": 1440, "height": 900}, "mobile": {"width": 390, "height": 844}}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def base_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    if not (ROOT / "web" / "dist" / "index.html").exists():
        pytest.skip("frontend not built (npm --prefix web run build)")
    import uvicorn

    from edgebench.api import create_app

    results = tmp_path_factory.mktemp("results")
    for e in ("e1", "e4", "e5"):
        assert cli(["experiment", e, "--configs", str(ROOT / "configs" / "smoke"), "--results", str(results)]) == 0
    mp = pytest.MonkeyPatch()
    mp.delenv("EDGEBENCH_API_TOKEN", raising=False)
    app = create_app(results)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    yield url
    server.should_exit = True
    thread.join(timeout=5)
    mp.undo()


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with playwright_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


PAGES = [
    ("lab", "/lab", "Perimeter"),
    ("frontier", "/frontier", "Accuracy against exposure"),
    ("grid", "/grid", "Macro-F1"),
    ("e1", "/experiments/e1_frontier", "Pareto set"),
    ("e2-pending", "/experiments/e2_adversary", "Status: pending"),
    ("method", "/method", "Limits"),
    ("not-found", "/nowhere", "Page not found"),
]


def _open(browser: Any, base_url: str, viewport: str, path: str, marker: str, scheme: str = "light") -> Any:
    ctx = browser.new_context(viewport=VIEWPORTS[viewport], color_scheme=scheme)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "404" not in m.text else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(base_url + path)
    page.get_by_text(marker, exact=False).first.wait_for(timeout=30000)
    page.wait_for_load_state("networkidle")
    page.errors = errors
    return page


@pytest.mark.parametrize("viewport", list(VIEWPORTS))
@pytest.mark.parametrize(("name", "path", "marker"), PAGES)
def test_page_renders(browser: Any, base_url: str, viewport: str, name: str, path: str, marker: str) -> None:
    page = _open(browser, base_url, viewport, path, marker)
    assert page.locator("h1").count() == 1
    assert not page.locator(".skeleton").count(), "loading skeleton still visible"
    assert not page.locator("[role=alert]").count(), "error state rendered"
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 1, f"page scrolls horizontally by {overflow}px"
    assert not page.errors, page.errors
    page.context.close()


def test_lab_transport_keyboard(browser: Any, base_url: str) -> None:
    page = _open(browser, base_url, "desktop", "/lab", "Perimeter")
    counter = page.get_by_text("window 1/", exact=False)
    total = counter.inner_text().split("/")[1]
    page.locator("main h1").click()
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    page.get_by_text(f"window 3/{total}").wait_for()
    page.keyboard.press("ArrowLeft")
    page.get_by_text(f"window 2/{total}").wait_for()
    # while the segment picker has focus the arrow keys belong to it: the segment changes (so the
    # replay restarts at window 1) and the transport shortcut does not also step
    page.locator("select").first.focus()
    page.keyboard.press("ArrowRight")
    page.get_by_text("window 1/", exact=False).wait_for()
    page.get_by_role("button", name="Fully local").click()
    assert "decision" in page.locator(".lane.perimeter").inner_text()
    page.context.close()


def test_frontier_selection_and_table_view(browser: Any, base_url: str) -> None:
    page = _open(browser, base_url, "desktop", "/frontier", "Accuracy against exposure")
    page.locator("g.pt").first.focus()
    page.keyboard.press("Enter")
    page.get_by_text("Advantage over chance").wait_for()
    page.get_by_role("button", name="Table").click()
    rows = page.locator("table tbody tr")
    assert rows.count() == 15
    assert page.get_by_role("button", name="Energy proxy").count() == 0  # cost chart hidden in table view
    page.context.close()


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("viewport", list(VIEWPORTS))
def test_no_wcag_violations(browser: Any, base_url: str, viewport: str, scheme: str) -> None:
    """axe-core audit (WCAG 2.1 A/AA and best practices) of every view in both themes."""
    if not AXE.exists():
        pytest.skip("axe-core not installed (npm --prefix web ci)")
    ctx = browser.new_context(viewport=VIEWPORTS[viewport], color_scheme=scheme, bypass_csp=True)
    page = ctx.new_page()
    found: dict[str, list[str]] = {}
    for _, path, marker in PAGES:
        page.goto(base_url + path)
        page.get_by_text(marker, exact=False).first.wait_for(timeout=30000)
        page.wait_for_load_state("networkidle")
        page.add_script_tag(path=str(AXE))
        violations = page.evaluate(
            """async () => (await axe.run(document, {runOnly: ['wcag2a', 'wcag2aa', 'wcag21aa',
               'best-practice']})).violations.map(v => v.id + ': ' + v.nodes.map(n => n.target.join(' ')).join(', '))"""
        )
        if violations:
            found[path] = violations
    ctx.close()
    assert not found, found
