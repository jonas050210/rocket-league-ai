"""Headless smoke test for the web GUI.

The GUI is plain JavaScript served by the API, so pytest cannot exercise it
directly.  ``tests/gui/smoke.mjs`` renders every view in jsdom against canned
API responses and asserts that the new controls exist, that a resume plan is
fetched and displayed, and that unavailable statistics are labelled instead of
invented.

The test is skipped when node or jsdom is not installed::

    cd tests/gui && npm install     # once, needs network access
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

GUI_DIR = Path(__file__).resolve().parent / "gui"
REPO_ROOT = GUI_DIR.parents[1]

pytestmark = pytest.mark.gui


def _node() -> str | None:
    return shutil.which("node")


def _jsdom_ready(node: str) -> bool:
    probe = subprocess.run(
        [node, "-e", "import('jsdom').then(() => 0).catch(() => process.exit(1))"],
        cwd=GUI_DIR,
        capture_output=True,
        timeout=120,
    )
    return probe.returncode == 0


def test_gui_views_render_without_errors():
    node = _node()
    if node is None:
        pytest.skip("node is not installed; the GUI smoke test needs it")
    if not _jsdom_ready(node):
        pytest.skip("jsdom is not installed; run `cd tests/gui && npm install`")
    result = subprocess.run(
        [node, "smoke.mjs", str(REPO_ROOT)],
        cwd=GUI_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"GUI smoke test failed:\n{result.stdout}\n{result.stderr}"
    assert "all GUI smoke checks passed" in result.stdout
