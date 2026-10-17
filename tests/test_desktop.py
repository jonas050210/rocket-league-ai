"""Native Qt tests: explicitly opt in on a host with a display and native libraries."""

import os
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from test_backend_process_integration import backend_process
from test_training_integration import approved, wait_for

pytestmark = [pytest.mark.desktop, pytest.mark.skipif(
    os.environ.get("RLAI_NATIVE_TEST") != "1",
    reason="Set RLAI_NATIVE_TEST=1 on WSLg/X11 or under xvfb-run; native dependencies required",
)]


def desktop_smoke(root, url, screenshot):
    result = subprocess.run(
        [sys.executable, "-m", "rocket_league_ai", "desktop", "--backend-url", url,
         "--data-dir", str(root), "--smoke", "--screenshot", str(screenshot)],
        capture_output=True, text=True, timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "six views loaded, backend connected; closing GUI only" in result.stdout
    assert screenshot.is_file() and screenshot.stat().st_size > 1000
    assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_native_window_closes_reopens_and_does_not_own_real_training(tmp_path, small_config):
    root = tmp_path / "workspace"
    with backend_process(root) as (backend, client):
        # Launch and close once with no run. Native startup must have no training side effect.
        desktop_smoke(root, str(client.base_url).rstrip("/"), tmp_path / "native-idle.png")
        assert client.get("/api/snapshot").json()["runs"] == []
        run_id = approved(client, "start_run", {"config": small_config.model_dump(mode="json")})["run_id"]
        wait_for(lambda: client.get("/api/snapshot").json()["active_run"]["status"] == "running")
        assert client.post(f"/api/runs/{run_id}/control", json={"command": "pause"}).status_code == 200
        wait_for(lambda: client.get("/api/snapshot").json()["active_run"]["status"] == "paused")
        before = client.get("/api/snapshot").json()["active_run"]["metrics"]["steps"]
        desktop_smoke(root, str(client.base_url).rstrip("/"), tmp_path / "native-active.png")
        assert backend.poll() is None
        active = client.get("/api/snapshot").json()["active_run"]
        assert active["id"] == run_id and active["status"] == "paused"
        assert active["metrics"]["steps"] == before
        # Reopen again during the same active run, not just once after an idle window.
        desktop_smoke(root, str(client.base_url).rstrip("/"), tmp_path / "native-reopened.png")
        active = client.get("/api/snapshot").json()["active_run"]
        assert backend.poll() is None and active["id"] == run_id and active["status"] == "paused"
        assert active["metrics"]["steps"] == before
        assert client.post(f"/api/runs/{run_id}/control", json={"command": "resume"}).status_code == 200
        wait_for(lambda: client.get("/api/snapshot").json()["active_run"]["metrics"]["steps"] > before)
        assert client.post(f"/api/runs/{run_id}/control", json={"command": "stop"}).status_code == 200
        wait_for(lambda: client.get("/api/snapshot").json()["active_run"] is None)
        run = client.get("/api/snapshot").json()["runs"][0]
        assert run["status"] == "stopped"


def test_native_launcher_bootstraps_independent_backend_without_starting_training(tmp_path):
    root = tmp_path / "workspace"
    with socket.socket() as socket_probe:
        socket_probe.bind(("127.0.0.1", 0))
        port = socket_probe.getsockname()[1]
    backend_pid = None
    try:
        desktop_smoke(root, f"http://127.0.0.1:{port}", tmp_path / "native-bootstrap.png")
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=3) as client:
            client.headers["Authorization"] = "Bearer " + (root / "access-token").read_text().strip()
            assert client.get("/api/health").json()["training_auto_start"] is False
            snapshot = client.get("/api/snapshot").json()
            assert snapshot["runs"] == [] and snapshot["active_run"] is None
        # Discover only this test's detached backend; never stop an unrelated local service.
        match = re.search(r"Started server process \[(\d+)\]", (root / "backend.log").read_text())
        assert match, (root / "backend.log").read_text()
        backend_pid = int(match[1])
        assert str(root) in Path(f"/proc/{backend_pid}/cmdline").read_text()
    finally:
        if backend_pid is None and (root / "backend.log").is_file():
            match = re.search(r"Started server process \[(\d+)\]", (root / "backend.log").read_text())
            backend_pid = int(match[1]) if match else None
        if backend_pid:
            try:
                os.kill(backend_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
