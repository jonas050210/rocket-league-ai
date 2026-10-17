"""Real released stack and subprocess lifecycle tests. No synthetic training backend."""

import json
import os
import signal
import time
import zipfile
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def wait_for(predicate, timeout=30, description="worker state"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.05)
    pytest.fail(f"Timed out waiting for {description}; last result: {last!r}")


def approved(client, kind, payload=None):
    response = client.post("/api/proposals", json={"kind": kind, "payload": payload or {}})
    assert response.status_code == 200, response.text
    proposal = response.json()
    response = client.post(f"/api/proposals/{proposal['id']}/confirm", json={
        "confirmed": True, "expected_digest": proposal["digest"],
    })
    assert response.status_code == 200, response.text
    return response.json()


def start_and_pause(client, manager, config):
    run_id = approved(client, "start_run", {"config": config.model_dump(mode="json")})["run_id"]
    wait_for(lambda: manager.store.run(run_id)["status"] == "running", description="real stack ready")
    assert manager.active.process.pid != os.getpid()
    response = client.post(f"/api/runs/{run_id}/control", json={"command": "pause"})
    assert response.status_code == 200, response.text
    wait_for(lambda: manager.store.run(run_id)["status"] == "paused", description="pause boundary")
    return run_id


def stop_and_wait(client, manager, run_id):
    response = client.post(f"/api/runs/{run_id}/control", json={"command": "stop"})
    assert response.status_code == 200, response.text
    wait_for(lambda: manager.active is None, description="worker shutdown and lease release")
    assert manager.store.run(run_id)["status"] == "stopped", manager.store.run(run_id)


def assert_state_equal(first, second):
    import numpy as np
    import torch

    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_state_equal(first[key], second[key])
    elif isinstance(first, list | tuple):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            assert_state_equal(left, right)
    else:
        assert first == second


def test_real_pause_resume_stop_and_gui_connection_independence(client, manager, small_config):
    import torch

    run_id = start_and_pause(client, manager, small_config)
    paused = manager.store.run(run_id)
    steps = paused["metrics"]["steps"]
    assert steps > 0 and steps % small_config.batch_size == 0
    assert paused["metrics"]["updates"] > 0
    assert paused["metrics"]["win_rate"] is None
    time.sleep(0.2)
    assert manager.store.run(run_id)["metrics"]["steps"] == steps
    assert client.post(f"/api/runs/{run_id}/control", json={"command": "pause"}).status_code == 409
    # A GUI is an HTTP client, not the owner of the app lifespan or learner process.
    gui = TestClient(client.app)
    gui.headers["Authorization"] = "Bearer " + manager.token
    assert gui.get("/api/snapshot").json()["active_run"]["id"] == run_id
    gui.close()
    assert manager.active.process.poll() is None
    started = time.monotonic()
    assert client.get("/api/health").status_code == 200
    assert time.monotonic() - started < 1
    assert client.post(f"/api/runs/{run_id}/control", json={"command": "resume"}).status_code == 200
    wait_for(lambda: manager.store.run(run_id)["metrics"]["steps"] > steps, description="new PPO updates")
    stop_and_wait(client, manager, run_id)
    run = manager.store.run(run_id)
    checkpoints = manager.checkpoints(run_id)
    latest = checkpoints.latest_valid(run["config_hash"])
    assert latest["step"] == run["metrics"]["steps"]
    initial = min(checkpoints.inventory(), key=lambda item: item["step"])
    before = torch.load(checkpoints.path(initial["id"]) / "ppo_learner/actor_critic.pt", weights_only=True)
    after = torch.load(checkpoints.path(latest["id"]) / "ppo_learner/actor_critic.pt", weights_only=True)
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert all(torch.isfinite(value).all() for value in after.values())
    assert any(event["kind"] == "checkpoint_committed" for event in manager.store.events(0, 1000)["events"])


def test_real_recovery_restores_weights_optimizers_buffer_gae_rng_and_counters(client, manager, small_config):
    import numpy as np
    import torch

    run_id = start_and_pause(client, manager, small_config)
    stop_and_wait(client, manager, run_id)
    original = manager.store.run(run_id)
    checkpoints = manager.checkpoints(run_id)
    checkpoint = checkpoints.latest_valid(original["config_hash"])
    parent = checkpoints.path(checkpoint["id"])
    assert checkpoint["step"] > 0
    target = {"run_id": run_id, "checkpoint_id": checkpoint["id"]}
    proposal_response = client.post("/api/proposals", json={"kind": "recover_run", "payload": target})
    assert proposal_response.status_code == 200, proposal_response.text
    proposal = proposal_response.json()
    assert manager.active is None  # Merely offering recovery never starts training.
    response = client.post(f"/api/proposals/{proposal['id']}/confirm", json={
        "confirmed": True, "expected_digest": proposal["digest"],
    })
    assert response.status_code == 200, response.text
    child_id = response.json()["run_id"]
    # Request stop during initialization; its initial checkpoint records the actual restored state.
    stop_and_wait(client, manager, child_id)
    child = manager.store.run(child_id)
    assert child_id != run_id and child["parent"] == target
    assert child["config_hash"] == original["config_hash"]
    assert manager.store.run(run_id) == original
    recovered = manager.checkpoints(child_id)
    initial = min(recovered.inventory(), key=lambda item: item["step"])
    restored = recovered.path(initial["id"])
    assert initial["step"] == checkpoint["step"]
    for filename in ("ppo_learner/actor_critic.pt", "ppo_learner/optimizers.pt", "rng.pt"):
        assert_state_equal(torch.load(parent / filename, weights_only=True),
                           torch.load(restored / filename, weights_only=True))
    for filename in ("ppo_learner/misc.json", "ppo_agent.json", "experience_buffer/trajectory_processor.json"):
        assert json.loads((parent / filename).read_text()) == json.loads((restored / filename).read_text())
    old_app = json.loads((parent / "application.json").read_text())
    new_app = json.loads((restored / "application.json").read_text())
    for key in ("numpy_rng", "python_rng", "buffer_rng", "episode_returns"):
        assert old_app[key] == new_app[key]
    for key in ("steps", "iterations", "updates", "episodes", "goal_episodes"):
        assert old_app["metrics"][key] == new_app["metrics"][key]
    with zipfile.ZipFile(parent / "experience_buffer/experience_buffer.zip") as old, \
            zipfile.ZipFile(restored / "experience_buffer/experience_buffer.zip") as new:
        assert set(old.namelist()) == set(new.namelist())
        for name in old.namelist():
            if name.endswith(".pt"):
                assert_state_equal(torch.load(BytesIO(old.read(name)), weights_only=True),
                                   torch.load(BytesIO(new.read(name)), weights_only=True))
            elif name.endswith(".npy"):
                np.testing.assert_array_equal(np.load(BytesIO(old.read(name))), np.load(BytesIO(new.read(name))))
            else:
                assert old.read(name) == new.read(name)
    # The parent manifests still verify after loading; recovery never rewrites the parent.
    assert checkpoints.require_valid(checkpoint["id"])["integrity"] == "valid"


def test_real_worker_kill_is_failed_not_restarted_and_checkpoint_is_preserved(client, manager, small_config):
    run_id = start_and_pause(client, manager, small_config)
    checkpoint = manager.checkpoints(run_id).latest_valid()
    os.killpg(manager.active.process.pid, signal.SIGKILL)
    wait_for(lambda: manager.active is None, description="unexpected exit cleanup")
    run = manager.store.run(run_id)
    assert run["status"] == "failed" and "without a terminal acknowledgement" in run["error"]
    assert len(manager.store.runs()) == 1
    assert manager.checkpoints(run_id).require_valid(checkpoint["id"])["integrity"] == "valid"
    assert client.get("/api/health").json()["status"] == "ready"


def test_real_frozen_worker_health_deadline_preserves_checkpoint(client, manager, small_config):
    run_id = start_and_pause(client, manager, small_config)
    checkpoint = manager.checkpoints(run_id).latest_valid()
    manager.stall_timeout = 1.5
    os.kill(manager.active.process.pid, signal.SIGSTOP)
    wait_for(lambda: manager.active is None, timeout=8, description="watchdog termination")
    run = manager.store.run(run_id)
    assert run["status"] == "failed" and "health deadline" in run["error"]
    assert manager.checkpoints(run_id).require_valid(checkpoint["id"])["integrity"] == "valid"
    assert any(event["kind"] == "watchdog_timeout" for event in manager.store.events(0, 1000)["events"])


def test_real_cuda_request_without_device_is_an_actionable_failure(client, manager, small_config):
    import torch

    if torch.cuda.is_available():
        pytest.skip("This failure path requires a host without CUDA")
    config = small_config.model_copy(update={"device": "cuda:0"})
    run_id = approved(client, "start_run", {"config": config.model_dump(mode="json")})["run_id"]
    wait_for(lambda: manager.active is None, description="explicit CUDA rejection")
    run = manager.store.run(run_id)
    assert run["status"] == "failed" and "No CPU fallback occurred" in run["error"]
    assert manager.checkpoints(run_id).inventory() == []
    assert len(manager.store.runs()) == 1


@pytest.mark.parametrize("n_proc", [1, 2])
def test_bounded_real_run_completes_exact_step_budget(client, manager, small_config, n_proc):
    config = small_config.model_copy(update={"total_timesteps": 512, "n_proc": n_proc})
    run_id = approved(client, "start_run", {"config": config.model_dump(mode="json")})["run_id"]
    wait_for(lambda: manager.active is None, description="completed bounded PPO run")
    run = manager.store.run(run_id)
    assert run["status"] == "completed", run
    assert run["metrics"]["steps"] == 512 and run["metrics"]["iterations"] == 4
    assert run["metrics"]["updates"] > 0
    assert manager.checkpoints(run_id).latest_valid()["step"] == 512
    assert all(item["integrity"] == "valid" for item in manager.checkpoints(run_id).inventory())


def test_delayed_repeated_pause_resume_remains_bounded_after_fast_collector_waves(client, manager, small_config):
    run_id = approved(client, "start_run", {"config": small_config.model_dump(mode="json")})["run_id"]
    wait_for(lambda: manager.store.run(run_id)["metrics"].get("steps", 0) >= 1024)
    for _ in range(3):
        assert client.post(f"/api/runs/{run_id}/control", json={"command": "pause"}).status_code == 200
        wait_for(lambda: manager.store.run(run_id)["status"] == "paused", timeout=3,
                 description="fenced collector pause without discarded early responses")
        before = manager.store.run(run_id)["metrics"]["steps"]
        assert client.post(f"/api/runs/{run_id}/control", json={"command": "resume"}).status_code == 200
        wait_for(lambda before=before: manager.store.run(run_id)["metrics"]["steps"] >= before + 512, timeout=3,
                 description="real post-resume collector waves")
    stop_and_wait(client, manager, run_id)
