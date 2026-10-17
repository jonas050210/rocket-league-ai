"""Actual service-process death/restart, EOF shutdown, ownership, and confirmed recovery."""

import os
import signal
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from test_training_integration import approved, wait_for

from rocket_league_ai.checkpoints import CheckpointStore
from rocket_league_ai.persistence import lease_available

pytestmark = pytest.mark.integration


@contextmanager
def backend_process(root):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "service-test.log").open("a") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "rocket_league_ai", "backend", "--host", "0.0.0.0",
             "--port", str(port), "--data-dir", str(root)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2)

        def ready():
            if process.poll() is not None:
                pytest.fail("Backend exited before readiness: " + (root / "service-test.log").read_text())
            try:
                return client.get("/api/health").status_code == 200
            except httpx.TransportError:
                return False

        try:
            wait_for(ready, timeout=15, description="actual HTTP backend startup")
            client.headers["Authorization"] = "Bearer " + (root / "access-token").read_text().strip()
            yield process, client
        finally:
            client.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def test_backend_crash_stops_orphan_then_restart_offers_only_explicit_recovery(tmp_path, small_config):
    root = tmp_path / "workspace"
    with backend_process(root) as (process, client):
        assert client.get("/api/snapshot").json()["runs"] == []
        # A service must never acquire a Torch/CUDA context merely by starting.
        assert "torch/lib/libtorch" not in Path(f"/proc/{process.pid}/maps").read_text()
        run_id = approved(client, "start_run", {"config": small_config.model_dump(mode="json")})["run_id"]

        def running():
            active = client.get("/api/snapshot").json()["active_run"]
            return active and active["status"] == "running" and active["metrics"].get("updates", 0) > 0

        wait_for(running, description="real HTTP worker training")
        assert client.post(f"/api/runs/{run_id}/control", json={"command": "pause"}).status_code == 200
        wait_for(lambda: client.get("/api/snapshot").json()["active_run"]["status"] == "paused")
        checkpoint_store = CheckpointStore(root / "runs" / run_id)
        checkpoint = checkpoint_store.latest_valid()
        assert checkpoint and checkpoint["step"] > 0
        assert not lease_available(root / "active-worker.lock")
        # Kill only the backend, not the independent worker's process group.
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        wait_for(lambda: lease_available(root / "active-worker.lock"), timeout=15,
                 description="stdin EOF cooperative checkpoint/worker cleanup")
        assert checkpoint_store.require_valid(checkpoint["id"])["integrity"] == "valid"
    with backend_process(root) as (_, client):
        snapshot = client.get("/api/snapshot").json()
        assert snapshot["active_run"] is None and len(snapshot["runs"]) == 1
        assert snapshot["runs"][0]["status"] == "interrupted"
        assert "No automatic restart" in snapshot["runs"][0]["error"]
        assert not snapshot["worker_health"]["orphan_lease_held"]
        assert snapshot["latest_valid"][run_id] is not None
        child_id = approved(client, "recover_run", {
            "run_id": run_id, "checkpoint_id": snapshot["latest_valid"][run_id],
        })["run_id"]
        assert child_id != run_id
        assert client.post(f"/api/runs/{child_id}/control", json={"command": "stop"}).status_code == 200
        wait_for(lambda: client.get("/api/snapshot").json()["active_run"] is None,
                 description="confirmed recovered worker shutdown")
        runs = client.get("/api/snapshot").json()["runs"]
        assert runs[0]["status"] == "stopped" and runs[0]["parent"]["run_id"] == run_id
        assert runs[1]["status"] == "interrupted"
        assert checkpoint_store.require_valid(checkpoint["id"])["integrity"] == "valid"


def test_service_sigterm_gracefully_stops_worker_and_leaves_durable_records(tmp_path, small_config):
    root = tmp_path / "workspace"
    with backend_process(root) as (process, client):
        run_id = approved(client, "start_run", {"config": small_config.model_dump(mode="json")})["run_id"]
        wait_for(lambda: client.get("/api/snapshot").json()["active_run"]["metrics"].get("updates", 0) > 0)
        process.terminate()
        process.wait(timeout=25)
        # Uvicorn 0.42 intentionally re-raises the original signal after lifespan cleanup.
        assert process.returncode == -signal.SIGTERM
        assert "Application shutdown complete" in (root / "service-test.log").read_text()
        assert lease_available(root / "backend.lock") and lease_available(root / "active-worker.lock")
    with backend_process(root) as (_, client):
        snapshot = client.get("/api/snapshot").json()
        assert snapshot["active_run"] is None
        assert snapshot["runs"][0]["status"] == "stopped"
        assert snapshot["latest_valid"][run_id]
