import json
import os
import time

import pytest
from conftest import requires_training, unit_checkpoint_writer

from rocket_league_ai.api import create_app
from rocket_league_ai.config import TrainingConfig
from rocket_league_ai.domain import AppError
from rocket_league_ai.metadata import experiment_metadata
from rocket_league_ai.persistence import Lease


def confirm(client, proposal, **overrides):
    return client.post(f"/api/proposals/{proposal['id']}/confirm", json={
        "confirmed": True, "expected_digest": proposal["digest"], **overrides,
    })


def test_launch_has_no_training_side_effect(client, manager):
    assert client.get("/api/health").json()["training_auto_start"] is False
    state = client.get("/api/snapshot").json()
    assert state["runs"] == [] and state["active_run"] is None
    from rocket_league_ai.metadata import package_versions
    assert state["evaluation_available"] is all(v is not None for v in package_versions().values())
    assert state["visualization_available"] is True  # recorded-replay viewer
    assert state["benchmark"]["session"] is None and state["evaluations"] == []
    assert state["designated_best"] is None
    assert manager.active is None
    assert os.stat(manager.root / "access-token").st_mode & 0o777 == 0o600


def test_capability_and_origin_protection(client):
    assert client.get("/api/snapshot", headers={"Authorization": "wrong"}).status_code == 401
    assert client.post("/api/proposals", json={"kind": "start_run"}, headers={"Origin": "https://foreign.invalid"}).status_code == 403
    assert client.get("/api/health", headers={"Authorization": ""}).status_code == 200
    assert client.post("/api/runs/invalid/control", json={"command": "stop"}).status_code == 422


def test_configuration_errors_are_actionable_and_no_plan_change(client):
    original = client.get("/api/snapshot").json()["plan"]
    response = client.post("/api/config/validate", json={"batch_size": 256, "bogus": True})
    assert response.status_code == 422
    assert "Extra inputs" in response.text
    assert client.get("/api/snapshot").json()["plan"] == original
    response = client.post("/api/config/validate", json=TrainingConfig(device="cpu").model_dump(mode="json"))
    assert response.json()["valid"] is True
    assert client.get("/api/snapshot").json()["plan"] == original


@requires_training
def test_start_requires_persisted_single_use_explicit_confirmation(client, manager, monkeypatch):
    calls = []
    monkeypatch.setattr(manager, "_launch", lambda config: calls.append(config) or {"run_id": "unit-test-only"})
    proposal = client.post("/api/proposals", json={"kind": "start_run"}).json()
    assert calls == []
    assert proposal["rationale"] and proposal["impact"]
    assert confirm(client, proposal, confirmed=False).status_code == 422
    assert confirm(client, proposal, confirmed=1).status_code == 422
    assert confirm(client, proposal, expected_digest="0" * 64).status_code == 422
    assert calls == []
    assert confirm(client, proposal).status_code == 200
    assert len(calls) == 1
    assert confirm(client, proposal).status_code == 409
    assert len(calls) == 1
    assert manager.store.proposal(proposal["id"])["status"] == "approved"


@requires_training
def test_strategy_apply_is_confirmed_and_stales_old_proposals(client, manager):
    old = manager.propose("start_run", {})
    config = TrainingConfig(device="cpu", seed=456)
    proposal = manager.propose("apply_config", {"config": config.model_dump(mode="json")})
    assert manager.store.setting("plan")["device"] == "cuda:0"
    assert proposal["payload"]["changes"]["seed"]["after"] == 456
    assert confirm(client, proposal).status_code == 200
    assert manager.store.setting("plan")["device"] == "cpu"
    assert manager.store.setting("plan_revision") == 2
    assert confirm(client, old).status_code == 409
    assert manager.store.runs() == []


@requires_training
def test_rejected_expired_and_extra_payloads(client, manager):
    proposal = manager.propose("start_run", {})
    assert client.post(f"/api/proposals/{proposal['id']}/reject", json={}).status_code == 200
    assert confirm(client, proposal).status_code == 409
    expired = manager.propose("start_run", {})
    with manager.store.db:
        manager.store.db.execute("UPDATE proposals SET expires=? WHERE id=?", (time.time() - 1, expired["id"]))
    assert confirm(client, expired).status_code == 409
    assert manager.store.proposal(expired["id"])["status"] == "expired"
    assert client.post("/api/proposals", json={"kind": "start_run", "payload": {"arbitrary_command": "no"}}).status_code == 422


def test_missing_stack_is_not_a_fake_training_mode(client, manager, monkeypatch):
    monkeypatch.setattr("rocket_league_ai.orchestration.package_versions", lambda: {"torch": None})
    response = client.post("/api/proposals", json={"kind": "start_run"})
    assert response.status_code == 409 and "--extra training" in response.text
    assert manager.active is None


def test_consequential_model_actions_and_protections(client, manager):
    run = manager.store.create_run(TrainingConfig(), experiment_metadata())
    manager.store.update_run(run["id"], status="stopped")
    checkpoints = manager.checkpoints(run["id"])
    first = checkpoints.publish(0, run["config_hash"], unit_checkpoint_writer, {})
    second = checkpoints.publish(512, run["config_hash"], unit_checkpoint_writer, {})
    target = {"run_id": run["id"], "checkpoint_id": first["id"]}
    proposal = manager.propose("designate_checkpoint", target)
    assert "no evidence" in proposal["rationale"]
    assert manager.store.setting("designated_best") is None
    assert confirm(client, proposal).status_code == 200
    assert manager.store.setting("designated_best") == target
    with pytest.raises(AppError, match="protected"):
        manager.propose("delete_checkpoint", target)
    other = {"run_id": run["id"], "checkpoint_id": second["id"]}
    deletion = manager.propose("delete_checkpoint", other)
    assert checkpoints.path(second["id"]).exists()
    assert confirm(client, deletion, confirmed=False).status_code == 422
    assert checkpoints.path(second["id"]).exists()
    assert confirm(client, deletion).status_code == 200
    assert not checkpoints.path(second["id"]).exists()
    assert checkpoints.path(first["id"]).exists()
    assert confirm(client, deletion).status_code == 409


def test_last_recovery_checkpoint_protected(manager):
    run = manager.store.create_run(TrainingConfig(), experiment_metadata())
    manager.store.update_run(run["id"], status="stopped")
    item = manager.checkpoints(run["id"]).publish(0, run["config_hash"], unit_checkpoint_writer, {})
    with pytest.raises(AppError, match="last valid"):
        manager.propose("delete_checkpoint", {"run_id": run["id"], "checkpoint_id": item["id"]})


def test_duplicate_backend_and_orphan_worker_prevented(manager):
    with pytest.raises(AppError, match="another process"):
        Lease(manager.root / "backend.lock")
    with Lease(manager.root / "active-worker.lock"):
        with pytest.raises(AppError, match="previous worker"):
            manager.propose("start_run", {})


def test_structured_logs_cursor_reconnect_and_bounded_query(client, manager):
    start = manager.store.event("warning", "test_warning", "First warning", data={"structured": True})
    end = manager.store.event("error", "test_error", "Second error")
    result = client.get(f"/api/events?after={start}&limit=1").json()
    assert len(result["events"]) == 1 and result["events"][0]["seq"] == end
    assert result["cursor"] == end
    assert client.get(f"/api/events?after={end}").json()["events"] == []
    assert client.get("/api/events?after=-1").status_code == 422
    assert client.get("/api/events?limit=1001").status_code == 422


def test_preview_is_read_only_without_capability(tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(create_app(tmp_path, preview_readonly=True)) as preview:
        assert preview.get("/api/snapshot").status_code == 200
        assert preview.post("/api/proposals", json={"kind": "start_run"}).status_code == 401
        assert "frame-ancestors" not in preview.get("/").headers["content-security-policy"]
        assert preview.get("/assets/app.js").status_code == 200


def test_report_does_not_invent_evaluations(client):
    report = client.get("/api/report").json()
    assert report["snapshot"]["checkpoints"] == []
    # No evaluation has run, so none may be reported; availability reflects the stack.
    assert report["snapshot"]["evaluations"] == []
    assert report["snapshot"]["evaluation_comparison"] == []
    assert "benchmark/evaluation evidence" in report["scope"]
    assert "access-token" not in json.dumps(report)


@pytest.mark.parametrize("identifier", [None, [], {}, 123, "../outside"])
def test_untrusted_identifiers_are_validation_errors_not_server_crashes(client, identifier):
    response = client.post("/api/proposals", json={
        "kind": "recover_run", "payload": {"run_id": identifier, "checkpoint_id": "invalid"},
    })
    assert response.status_code == 422


@pytest.mark.parametrize(("field", "value", "message"), [
    ("packages", {"not": "the original profile"}, "Dependency versions differ"),
    ("training_source_sha256", "0" * 64, "source differs"),
    ("python", "0.0.0", "interpreter profile"),
])
def test_recovery_rejects_changed_dependencies_source_and_interpreter(manager, field, value, message):
    run = manager.store.create_run(TrainingConfig(), {**experiment_metadata(), field: value})
    manager.store.update_run(run["id"], status="stopped")
    checkpoint = manager.checkpoints(run["id"]).publish(128, run["config_hash"], unit_checkpoint_writer, {})
    with pytest.raises(AppError, match=message):
        manager.propose("recover_run", {"run_id": run["id"], "checkpoint_id": checkpoint["id"]})
    assert manager.active is None and len(manager.store.runs()) == 1


def test_recovery_rechecks_source_profile_between_review_and_confirmation(manager, monkeypatch):
    run = manager.store.create_run(TrainingConfig(), experiment_metadata())
    manager.store.update_run(run["id"], status="stopped")
    checkpoint = manager.checkpoints(run["id"]).publish(128, run["config_hash"], unit_checkpoint_writer, {})
    proposal = manager.propose("recover_run", {"run_id": run["id"], "checkpoint_id": checkpoint["id"]})
    monkeypatch.setattr("rocket_league_ai.metadata.training_source_hash", lambda: "0" * 64)
    with pytest.raises(AppError, match="source differs"):
        manager.confirm(proposal["id"], expected_digest=proposal["digest"], confirmed=True)
    assert manager.active is None and len(manager.store.runs()) == 1
    # Rejected preconditions do not execute or consume the reviewed action.
    assert manager.store.proposal(proposal["id"])["status"] == "pending"


def test_direct_start_continues_from_latest_checkpoint_and_exports_rlbot(client, manager, monkeypatch):
    monkeypatch.setattr(
        "rocket_league_ai.orchestration.package_versions",
        lambda: {"torch": "test", "numpy": "test", "rlgym-learn": "test"},
    )
    # Seed an existing run with a valid checkpoint at step 2048
    base_cfg = TrainingConfig(device="cpu", n_proc=2, batch_size=2048, timesteps_per_iteration=2048, buffer_size=6144)
    run = manager.store.create_run(base_cfg, experiment_metadata())
    manager.store.update_run(run["id"], status="completed", metrics={**run["metrics"], "steps": 2048, "steps_per_second": 2400.0})
    checkpoint = manager.checkpoints(run["id"]).publish(2048, run["config_hash"], unit_checkpoint_writer, {})

    snap = client.get("/api/snapshot").json()
    assert snap["highest_step"] == 2048
    assert snap["latest_checkpoint"]["checkpoint_id"] == checkpoint["id"]
    assert snap["average_steps_per_second"] >= 2000.0
    assert len(snap["roadmap"]) == 5
    assert snap["roadmap"][0]["current_steps"] == 2048

    # Mock _launch so we don't spawn a real subprocess without torch
    launched = {}

    def fake_launch(config, *, parent=None):
        launched["config"] = config
        launched["parent"] = parent
        return {"run_id": "newrun123"}

    monkeypatch.setattr(manager, "_launch", fake_launch)

    res = client.post(
        "/api/runs/start-direct",
        json={"milestone_id": "milestone-1", "continue_from_latest": True},
    )
    assert res.status_code == 200
    assert launched["parent"]["run_id"] == run["id"]
    assert launched["parent"]["checkpoint_id"] == checkpoint["id"]
    assert launched["parent"]["allow_config_evolution"] is True
    assert launched["config"].total_timesteps >= 1_000_000

    # Test RLBot export endpoint
    rlbot = client.get(f"/api/checkpoints/{run['id']}/{checkpoint['id']}/rlbot")
    assert rlbot.status_code == 200
    bundle = rlbot.json()
    assert bundle["checkpoint_id"] == checkpoint["id"]
    assert "[Locations]" in bundle["files"]["bot.cfg"]
    assert "RLAIBot" in bundle["files"]["agent.py"]
    assert "RLBotAgent" in bundle["files"]["agent.py"]


def test_live_match_start_pause_kickoff_reset_and_stop(client, manager):
    base_cfg = TrainingConfig(device="cpu", n_proc=1, batch_size=512, timesteps_per_iteration=512, buffer_size=1536)
    run = manager.store.create_run(base_cfg, experiment_metadata())
    manager.store.update_run(run["id"], status="completed")
    checkpoint = manager.checkpoints(run["id"]).publish(512, run["config_hash"], unit_checkpoint_writer, {})

    res = client.post(
        "/api/live-match/start",
        json={"mode": "human_vs_ai", "scenario": "mixed-scenarios-v1", "speed": 2.0, "launch_client": True},
    )
    assert res.status_code == 200
    snap = res.json()
    assert snap["status"] == "running"
    assert snap["checkpoint"]["checkpoint_id"] == checkpoint["id"]
    assert snap["export_dir"] is not None
    assert snap["current_frame"] is not None
    assert len(snap["current_frame"]["cars"]) == 2

    paused = client.post("/api/live-match/control", json={"action": "toggle_pause"}).json()
    assert paused["paused"] is True
    assert paused["status"] == "paused"

    resumed = client.post("/api/live-match/control", json={"action": "toggle_pause"}).json()
    assert resumed["paused"] is False

    reset = client.post("/api/live-match/control", json={"action": "reset_kickoff"}).json()
    assert reset["status"] == "running"

    next_g = client.post("/api/live-match/control", json={"action": "next_game"}).json()
    assert next_g["game_number"] >= 2

    stopped = client.post("/api/live-match/stop").json()
    assert stopped["status"] == "stopped"


