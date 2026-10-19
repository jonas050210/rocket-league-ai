"""HTTP API contract used by the GUI."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from rlai import checkpoints
from rlai.api import create_app
from rlai.config import TrainingConfig
from rlai.store import Store


@pytest.fixture()
def client(data_root, models_dir):
    app = create_app()
    with TestClient(app) as test_client:
        test_client.app = app
        yield test_client


def test_health_and_static(client):
    assert client.get("/api/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert "Rocket League AI" in page.text
    assert client.get("/theme/theme.css").status_code == 200
    assert client.get("/app.js").status_code == 200


def test_modes_and_defaults(client):
    modes = client.get("/api/modes").json()
    assert {mode["id"] for mode in modes} == {
        "soccer-1v1",
        "soccer-2v2",
        "soccer-3v3",
        "heatseeker-2v2",
    }
    defaults = client.get("/api/config/defaults").json()
    assert defaults["training"]["mode"] in {m["id"] for m in modes}
    # the simplified workflow offers exactly one automatic configuration:
    # 1v1 with the stable backend defaults and the machine-checked worker count
    assert "presets" not in defaults
    default_run = defaults["default_run"]
    assert default_run["mode"] == "soccer-1v1"
    config = default_run["config"]
    assert config["mode"] == "soccer-1v1"
    assert config["obs_version"] in {1, 2}
    assert config["total_steps"] >= config["rollout_steps"] * config["n_workers"]
    assert default_run["device"] in {"auto", "cpu", "cuda", "mps"}
    assert default_run["workers"]["recommended_workers"] >= 1


def test_validate_accepts_and_rejects(client):
    config = client.get("/api/config/defaults").json()["training"]
    config["total_steps"] = 2000
    config["rollout_steps"] = 256
    config["n_workers"] = 1
    config["minibatch_size"] = 256
    assert client.post("/api/config/validate", json=config).status_code == 200
    bad = dict(config, mode="hoops")
    assert client.post("/api/config/validate", json=bad).status_code == 422
    # obs v3 (own-velocity block) is a real layout now; unknown ones still fail
    assert client.post("/api/config/validate", json=dict(config, obs_version=3)).status_code == 200
    assert client.post("/api/config/validate", json=dict(config, obs_version=99)).status_code == 422
    assert client.post("/api/config/validate", json=dict(config, eval_interval=-1)).status_code == 422
    legacy = dict(config, obs_version=1, reward={**config["reward"], "demo": 0.4})
    legacy["reward"].pop("bump", None)
    assert client.post("/api/config/validate", json=legacy).status_code == 200


def _compatible_checkpoint(models_dir, config: TrainingConfig, **kwargs) -> dict:
    """A checkpoint that a resume from ``config`` accepts."""
    policy_config = config.policy_config()
    payload = {
        "model_state": policy_config.build().state_dict(),
        "policy_config": policy_config.to_dict(),
        "optimizer_state": {"state": {}, "param_groups": [{"lr": config.learning_rate}]},
        "step": kwargs.pop("step", 2048),
        "run_id": kwargs.pop("run_id", "parent-run"),
        "mode": config.mode,
        "training_config": config.to_dict(),
        "extra": {
            "obs_version": config.obs_version,
            "iteration": 4,
            "rng": {"numpy": {}, "torch": None, "python": None},
            "curriculum": {"stage": 0, "total_iterations": 4},
            "league": {"agents": []},
        },
    }
    payload.update(kwargs)
    return checkpoints.save_checkpoint(models_dir, **payload)


def _config_from(client, **overrides) -> dict:
    payload = client.get("/api/config/defaults").json()["training"]
    payload.update(
        run_name="api test",
        total_steps=4096,
        rollout_steps=256,
        n_workers=1,
        minibatch_size=256,
        mode="soccer-1v1",
        # these tests exercise explicit configuration handling; automatic
        # continuation is covered separately
        resume_mode="fresh",
        # a training run is only started after an explicit confirmation
        confirm=True,
    )
    payload.update(overrides)
    return payload


def test_run_lifecycle(client, monkeypatch):
    started = {}

    def fake_start(config, run_id, resume_from=None, **kwargs):
        started["config"] = config
        started["run_id"] = run_id
        started["resume_from"] = resume_from
        started.update(kwargs)

    monkeypatch.setattr(client.app.state.jobs, "start_training", fake_start)
    payload = client.get("/api/config/defaults").json()["training"]
    payload.update(
        run_name="api test",
        total_steps=2000,
        rollout_steps=256,
        n_workers=1,
        minibatch_size=256,
        mode="soccer-1v1",
        resume_mode="fresh",
        confirm=True,
    )
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 200, response.text
    run = response.json()
    assert started["run_id"] == run["id"]
    assert client.get(f"/api/runs/{run['id']}").json()["name"] == "api test"
    assert client.get("/api/runs").json()
    assert client.post(f"/api/runs/{run['id']}/pause").json()["control"]["pause"] is True
    assert client.post(f"/api/runs/{run['id']}/resume").json()["control"]["pause"] is False
    assert client.post(f"/api/runs/{run['id']}/stop").json()["control"]["stop"] is True
    assert client.get(f"/api/runs/{run['id']}/metrics").status_code == 200
    assert client.get(f"/api/runs/{run['id']}/logs").status_code == 200
    assert client.get("/api/runs/unknown").status_code == 404
    assert started["resume_from"] is None
    # nothing was requested, so the mode stays explicit and fresh
    assert started["resume_mode"] == "fresh"
    assert started["allow_concurrent"] is False
    bad = dict(payload, resume_from="not-a-model", resume_mode="continue")
    assert client.post("/api/runs", json=bad).status_code == 400
    assert client.post("/api/runs", json=dict(payload, resume_mode="sideways")).status_code == 400


def test_the_event_stream_is_a_streaming_response(client):
    """The SSE endpoint must return a stream, not an empty body."""
    import asyncio

    from fastapi.responses import StreamingResponse

    endpoint = next(
        route.endpoint
        for route in client.app.routes
        if getattr(route, "path", None) == "/api/events"
    )

    class _Disconnected:
        async def is_disconnected(self) -> bool:
            return True

    response = asyncio.run(endpoint(_Disconnected(), None, 0))
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"


def test_start_run_needs_an_explicit_confirmation(client, monkeypatch):
    """A training run is never started by an unconfirmed request."""
    started = {}

    def fake_start(config, run_id, resume_from=None, **kwargs):
        started["run_id"] = run_id

    monkeypatch.setattr(client.app.state.jobs, "start_training", fake_start)
    payload = _config_from(client, confirm=False)
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 400, response.text
    assert "not confirmed" in response.json()["detail"]
    # nothing was started and no run row was created
    assert started == {}
    assert client.get("/api/runs").json() == []


def test_start_run_refuses_a_second_training_run(client, monkeypatch):
    def refusing(*args, **kwargs):
        raise RuntimeError("training run abc is already active; stop it first")

    monkeypatch.setattr(client.app.state.jobs, "start_training", refusing)
    payload = _config_from(client)
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 409, response.text
    assert "already active" in response.json()["detail"]
    # the refused run is kept, marked queued, so the GUI can explain and retry
    runs = client.get("/api/runs").json()
    assert runs[0]["status"] == "queued"
    assert "already active" in runs[0]["message"]


def test_start_run_passes_an_explicit_resume(client, models_dir, monkeypatch):
    started = {}

    def fake_start(config, run_id, resume_from=None, **kwargs):
        started["resume_from"] = resume_from
        started.update(kwargs)

    config = TrainingConfig.model_validate(_config_from(client))
    manifest = _compatible_checkpoint(models_dir, config)
    monkeypatch.setattr(client.app.state.jobs, "start_training", fake_start)

    payload = _config_from(client, resume_mode="continue", resume_from=manifest["id"])
    assert client.post("/api/runs", json=payload).status_code == 200
    assert started["resume_mode"] == "continue"
    assert started["resume_from"].endswith(manifest["path"])

    # fresh ignores the checkpoint instead of silently restoring it
    payload = _config_from(client, resume_mode="fresh", resume_from=manifest["id"])
    assert client.post("/api/runs", json=payload).status_code == 200
    assert started["resume_mode"] == "fresh" and started["resume_from"] is None

    # an incompatible explicit checkpoint is refused before any run row is doomed
    base = _config_from(client)
    incompatible = _config_from(
        client,
        resume_mode="continue",
        resume_from=manifest["id"],
        policy={**base["policy"], "hidden": [64]},
    )
    response = client.post("/api/runs", json=incompatible)
    assert response.status_code == 400, response.text
    assert "architecture" in response.json()["detail"]


def test_resume_plan_endpoint_is_read_only(client, models_dir):
    config = _config_from(client)
    response = client.post("/api/resume/plan", json={"config": config, "mode": "continue"})
    assert response.status_code == 200
    plan = response.json()
    assert plan["will_resume"] is False
    assert plan["restores"] == []
    # the fixture checkpoint has a different architecture: rejected with a reason,
    # and never silently used
    assert plan["rejected"], plan
    assert any("architecture" in reason for item in plan["rejected"] for reason in item["reasons"])
    assert plan["considered"] == []

    manifest = _compatible_checkpoint(models_dir, TrainingConfig.model_validate(config))
    plan = client.post("/api/resume/plan", json={"config": config, "mode": "continue"}).json()
    assert plan["will_resume"] is True
    assert plan["model_id"] == manifest["id"]
    assert plan["restored_step"] == 2048
    assert plan["has_optimizer"] is True
    assert plan["missing"] == []
    assert any("optimiser state" in item for item in plan["restores"])
    assert any("RNG state" in item for item in plan["restores"])

    transfer = client.post("/api/resume/plan", json={"config": config, "mode": "transfer"}).json()
    assert transfer["restored_step"] == 0
    assert transfer["restores"][0].startswith("weights")

    fresh = client.post("/api/resume/plan", json={"config": config, "mode": "fresh"}).json()
    assert fresh["will_resume"] is False

    assert (
        client.post("/api/resume/plan", json={"config": config, "mode": "bogus"}).status_code == 400
    )
    assert client.post("/api/resume/plan", json={"config": {"mode": "hoops"}}).status_code == 400
    # an explicit request that cannot be honoured never falls back silently
    response = client.post(
        "/api/resume/plan", json={"config": config, "mode": "continue", "checkpoint": "missing-id"}
    )
    assert response.status_code == 200
    plan = response.json()
    assert plan["will_resume"] is False
    assert plan["explicit_unavailable"] is True
    assert any("missing-id" in warning for warning in plan["warnings"])

    # naming a real but incompatible checkpoint is equally explicit
    other = dict(config, policy={**config["policy"], "hidden": [64]})
    plan = client.post(
        "/api/resume/plan",
        json={"config": other, "mode": "continue", "checkpoint": manifest["id"]},
    ).json()
    assert plan["will_resume"] is False
    assert plan["explicit_unavailable"] is True
    assert plan["rejected"][0]["reasons"]


def test_start_plan_endpoint_is_read_only(client, models_dir):
    """The simplified form asks this what a start would do, before it starts."""
    # without a checkpoint: the plan says a fresh model will be initialised
    payload = client.post("/api/runs/start-plan", json={"total_steps": 4096}).json()
    assert payload["will_resume"] is False
    assert payload["resume_mode"] == "fresh"
    assert payload["config"]["mode"] == "soccer-1v1"
    assert payload["config"]["total_steps"] == 4096
    assert payload["device"] in {"cpu", "cuda", "mps"}
    assert payload["workers"]["recommended_workers"] >= 1
    # the run name is honoured but optional
    named = client.post(
        "/api/runs/start-plan", json={"total_steps": 4096, "run_name": "smoke test"}
    ).json()
    assert named["config"]["run_name"] == "smoke test"
    # technical values keep their backend defaults; the user never sent them
    assert payload["config"]["learning_rate"] == 3e-4
    assert payload["config"]["n_workers"] == 2
    # out-of-range budgets are rejected like every other config
    assert client.post("/api/runs/start-plan", json={"total_steps": 1}).status_code == 422

    # with a compatible checkpoint: the plan says which checkpoint would be
    # restored, using the same logic the actual start would use
    config = TrainingConfig.model_validate(_config_from(client))
    manifest = _compatible_checkpoint(models_dir, config)
    payload = client.post("/api/runs/start-plan", json={"total_steps": 8192}).json()
    assert payload["will_resume"] is True
    assert payload["resume_mode"] == "continue"
    assert payload["resume"]["model_id"] == manifest["id"]
    assert payload["resume"]["restored_step"] == 2048
    assert any("optimiser state" in item for item in payload["resume"]["restores"])


def test_simplified_start_run_payload(client, models_dir, monkeypatch):
    """The redesigned form posts only the user-controlled values.

    Everything else comes from the backend defaults; the confirmation gate
    and the checkpoint-continuation safety rules must keep applying.
    """
    started = {}

    def fake_start(config, run_id, resume_from=None, **kwargs):
        started["config"] = config
        started["resume_from"] = resume_from
        started.update(kwargs)

    monkeypatch.setattr(client.app.state.jobs, "start_training", fake_start)

    # explicit fresh start: a slim payload without any technical parameter
    response = client.post(
        "/api/runs",
        json={
            "total_steps": 4096,
            "run_name": "slim run",
            "resume_mode": "fresh",
            "confirm": True,
        },
    )
    assert response.status_code == 200, response.text
    config = started["config"]
    assert config.mode == "soccer-1v1"
    assert config.total_steps == 4096
    assert config.run_name == "slim run"
    # defaults filled in by the server, not by the user
    assert config.learning_rate == 3e-4
    assert config.minibatch_size == 1024
    assert started["resume_mode"] == "fresh"
    assert started["resume_from"] is None

    # a slim payload that was not confirmed still refuses to start
    started.clear()
    response = client.post("/api/runs", json={"total_steps": 4096, "confirm": False})
    assert response.status_code == 400
    assert "not confirmed" in response.json()["detail"]
    assert started == {}

    # continue is only honoured when a compatible checkpoint exists;
    # without one it is refused instead of silently starting from scratch
    response = client.post(
        "/api/runs",
        json={"total_steps": 4096, "resume_mode": "continue", "confirm": True},
    )
    assert response.status_code == 400, response.text

    manifest = _compatible_checkpoint(
        models_dir, TrainingConfig.model_validate(_config_from(client))
    )
    response = client.post(
        "/api/runs",
        json={"total_steps": 4096, "resume_mode": "continue", "confirm": True},
    )
    assert response.status_code == 200, response.text
    assert started["resume_mode"] == "continue"
    assert started["resume_from"].endswith(manifest["path"])


def test_omitted_resume_mode_continues_automatically(client, models_dir, monkeypatch):
    """No resume fields means auto: resume if possible, fresh only if empty."""
    started = {}

    def fake_start(config, run_id, resume_from=None, **kwargs):
        started["resume_from"] = resume_from
        started.update(kwargs)

    monkeypatch.setattr(client.app.state.jobs, "start_training", fake_start)
    config = TrainingConfig.model_validate(_config_from(client, resume_mode="fresh"))
    manifest = _compatible_checkpoint(models_dir, config)

    response = client.post(
        "/api/runs", json={"total_steps": 4096, "confirm": True}
    )
    assert response.status_code == 200, response.text
    assert started["resume_mode"] == "continue"
    assert started["resume_from"].endswith(manifest["path"])

    # an unusable checkpoint is refused, not silently replaced by a new model
    models_dir.joinpath(manifest["id"]).joinpath(
        checkpoints.MODEL_FILE
    ).write_bytes(b"broken")
    response = client.post(
        "/api/runs", json={"total_steps": 4096, "confirm": True}
    )
    assert response.status_code == 400, response.text
    assert "automatic resume" in response.json()["detail"]


def test_model_validation_endpoint(client, models_dir):
    config = _config_from(client)
    manifest = _compatible_checkpoint(models_dir, TrainingConfig.model_validate(config))
    response = client.post(f"/api/models/{manifest['id']}/validate", json={"config": config})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    assert payload["step"] == 2048
    assert payload["resume"]["continue"] is True
    assert payload["resume"]["transfer"] is True
    assert payload["resume"]["missing_state"] == []

    bare = _compatible_checkpoint(
        models_dir,
        TrainingConfig.model_validate(config),
        step=512,
        run_id="bare-run",
        extra={"obs_version": TrainingConfig.model_validate(config).obs_version},
    )
    payload = client.post(f"/api/models/{bare['id']}/validate", json={"config": config}).json()
    assert payload["ok"] is True
    # resumable, but the GUI must say which state is missing rather than promise it
    assert payload["resume"]["continue"] is True
    assert set(payload["resume"]["missing_state"]) == {"rng", "curriculum", "league"}

    # an incompatible architecture is reported with a reason, not accepted
    other = dict(config, policy={**config["policy"], "hidden": [64]})
    payload = client.post(f"/api/models/{manifest['id']}/validate", json={"config": other}).json()
    assert payload["ok"] is False
    assert any("architecture" in reason for reason in payload["reasons"])
    assert payload["resume"]["continue"] is False

    assert client.post("/api/models/nope/validate", json={}).status_code == 404


def test_report_endpoints(client, data_root):
    store = Store(data_root / "rlai.sqlite3")
    run_id = store.create_run("api report", "soccer-1v1", {"mode": "soccer-1v1"}, 2000)
    store.update_run(run_id, status="completed", step=2000)

    assert client.get(f"/api/runs/{run_id}/report").status_code == 404
    assert "rebuild" in client.get(f"/api/runs/{run_id}/report").json()["detail"]
    rebuilt = client.get(f"/api/runs/{run_id}/report?rebuild=true")
    assert rebuilt.status_code == 200
    assert rebuilt.json()["schema"] == "rlai-run-report-v1"
    assert rebuilt.json()["run_id"] == run_id
    assert client.get("/api/runs/unknown/report").status_code == 404

    report = {"run_id": run_id, "schema": "stale-marker", "status": "completed",
              "mode": "soccer-1v1", "steps": {"completed": 2000}}
    store.save_report(run_id, report)
    assert client.get(f"/api/runs/{run_id}/report").json()["schema"] == "stale-marker"
    # ?rebuild=true recomputes instead of returning the stale copy
    assert (
        client.get(f"/api/runs/{run_id}/report?rebuild=true").json()["schema"]
        == "rlai-run-report-v1"
    )
    listed = client.get("/api/reports").json()
    assert any(item["run_id"] == run_id for item in listed)

    other = store.create_run("second", "soccer-2v2", {}, 2000)
    comparison = client.post("/api/runs/compare", json={"run_ids": [run_id, other]})
    assert comparison.status_code == 200
    assert comparison.json()["comparable"] is False
    assert client.post("/api/runs/compare", json={"run_ids": ["nope"]}).status_code == 404


def test_evaluation_suite_endpoint(client, models_dir, monkeypatch):
    started = {}

    def fake_eval(**kwargs):
        started.update(kwargs)
        return "ev-suite"

    monkeypatch.setattr(client.app.state.jobs, "start_evaluation", fake_eval)
    model_id = _compatible_checkpoint(models_dir, TrainingConfig.model_validate(_config_from(client)))["id"]

    response = client.post(
        "/api/evaluations/suite",
        json={
            "model_id": model_id,
            "opponents": ["idle", "chase"],
            "games": 2,
            "match_seconds": 10,
            "sides": ["blue", "orange"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "suite" and body["sides"] == ["blue", "orange"]
    assert started["phase"] == "suite" and started["mirror"] is True
    assert started["opponents"] == ["idle", "chase"]

    # a suite needs at least two opponents; bad names/sides/budgets are rejected up front
    assert client.post(
        "/api/evaluations/suite", json={"model_id": model_id, "opponents": ["idle"]}
    ).status_code == 400
    assert client.post(
        "/api/evaluations/suite", json={"model_id": model_id, "opponents": ["idle", "telepath"]}
    ).status_code == 400
    assert client.post(
        "/api/evaluations/suite",
        json={"model_id": model_id, "opponents": ["idle", "chase"], "sides": ["purple"]},
    ).status_code == 400
    assert client.post(
        "/api/evaluations/suite",
        json={
            "model_id": model_id,
            "opponents": ["idle", "chase"],
            "games": 500,
            "match_seconds": 600,
        },
    ).status_code == 400
    assert client.post(
        "/api/evaluations/suite",
        json={"model_id": model_id, "opponents": ["idle", "chase"], "phase": "nonsense"},
    ).status_code == 400

    # the single-opponent route keeps its own phase and side settings
    response = client.post(
        f"/api/models/{model_id}/evaluate",
        json={
            "model_id": model_id,
            "opponent": "chase",
            "games": 2,
            "match_seconds": 10,
            "sides": ["blue", "orange"],
            "mirror": False,
            "phase": "manual",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["kind"] == "single"
    assert started["phase"] == "manual" and started["mirror"] is False
    assert started["sides"] == ["blue", "orange"]


def test_model_registry_endpoints(client, trained_model, monkeypatch):
    models = client.get("/api/models").json()
    assert any(m["id"] == trained_model["id"] for m in models)
    model_id = trained_model["id"]
    detail = client.get(f"/api/models/{model_id}").json()
    assert detail["id"] == model_id

    started = {}

    def fake_eval(**kwargs):
        started.update(kwargs)
        return "ev-test"

    monkeypatch.setattr(client.app.state.jobs, "start_evaluation", fake_eval)
    response = client.post(
        f"/api/models/{model_id}/evaluate",
        json={"model_id": model_id, "opponent": "idle", "games": 2, "match_seconds": 10},
    )
    assert response.status_code == 200, response.text
    assert started["model_id"] == model_id
    assert started["mode"] == "soccer-1v1"

    promoted = client.post(f"/api/models/{model_id}/promote")
    assert promoted.status_code == 200 and promoted.json()["stable"] is True
    assert (
        client.post(f"/api/models/{model_id}/evaluate", json={"model_id": model_id}).status_code
        == 200
    )
    assert client.delete(f"/api/models/{model_id}").status_code == 400  # stable protected


def test_live_status(client):
    status = client.get("/api/live/status").json()
    assert status["platform"]
    assert isinstance(status["launch_supported"], bool)
    if not status["launch_supported"]:
        response = client.post("/api/live/launch", json={"mode": "soccer-1v1"})
        assert response.status_code == 501
        assert "Windows" in response.json()["detail"]


def test_live_preflight_and_dry_run(client, trained_model):
    report = client.get("/api/live/preflight").json()
    assert {"ok", "platform", "checks", "server"} == set(report)
    names = {check["name"] for check in report["checks"]}
    assert {"model", "match config parser", "rlbot server"} <= names
    selected = client.get(
        f"/api/live/preflight?model_id={trained_model['id']}"
    ).json()
    model_check = next(c for c in selected["checks"] if c["name"] == "model")
    assert model_check["ok"] is True

    response = client.post(
        "/api/live/launch",
        json={"mode": "soccer-1v1", "model_id": trained_model["id"], "dry_run": True},
    )
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["dry_run"] is True and record["launched"] is False
    assert record["model_id"] == trained_model["id"]
    assert record["human_team"] == "blue" and record["fill"] == "psyonix"
    assert os.path.exists(record["config"])

    response = client.post(
        "/api/live/launch",
        json={
            "mode": "soccer-2v2",
            "model_id": trained_model["id"],
            "dry_run": True,
            "human_team": "orange",
            "fill": "policy",
            "human": False,
        },
    )
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["human"] is False and record["human_team"] == "orange"
    with open(record["config"], "rb") as handle:
        import tomllib

        parsed = tomllib.load(handle)
    assert len(parsed["cars"]) == 4


def test_live_launch_rejects_a_path_as_model_id(client):
    response = client.post(
        "/api/live/launch",
        json={"mode": "soccer-1v1", "model_id": "../../etc/passwd", "dry_run": True},
    )
    assert response.status_code == 400
    assert "invalid model id" in response.json()["detail"]
    assert client.get("/api/live/preflight?model_id=../..").status_code == 400
    assert (
        client.post(
            "/api/live/launch",
            json={"mode": "soccer-1v1", "human_team": "purple", "dry_run": True},
        ).status_code
        == 400
    )


def test_auth_token(monkeypatch, data_root, models_dir):
    monkeypatch.setenv("RLAI_AUTH_TOKEN", "secret")
    app = create_app()
    with TestClient(app) as client:
        # read-only routes stay available; mutations require the token
        assert client.get("/api/modes").status_code == 200
        assert client.post("/api/config/validate", json={}).status_code == 401
        invalid = {"mode": "hoops"}
        assert (
            client.post(
                "/api/config/validate", json=invalid, headers={"x-rlai-token": "secret"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/config/validate",
                json=invalid,
                headers={"authorization": "Bearer secret"},
            ).status_code
            == 422
        )
        # the event stream cannot send headers, so it accepts ?token=
        assert client.get("/api/events").status_code == 401
        assert client.get("/api/events?token=nope").status_code == 401


def test_cors_is_opt_in(monkeypatch, data_root, models_dir):
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/health", headers={"origin": "https://evil.example"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    monkeypatch.setenv("RLAI_CORS_ORIGINS", "https://gui.example")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/health", headers={"origin": "https://gui.example"})
        assert response.headers.get("access-control-allow-origin") == "https://gui.example"
        response = client.get("/api/health", headers={"origin": "https://evil.example"})
        assert "access-control-allow-origin" not in response.headers


def test_omitted_resume_mode_starts_fresh_only_without_any_checkpoint(
    client, models_dir, monkeypatch
):
    started = {}

    def fake_start(config, run_id, resume_from=None, **kwargs):
        started["resume_from"] = resume_from
        started.update(kwargs)

    monkeypatch.setattr(client.app.state.jobs, "start_training", fake_start)
    # the shared fixture contains a checkpoint; this test needs an empty store
    import shutil

    for child in models_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    response = client.post("/api/runs", json={"total_steps": 4096, "confirm": True})

    assert response.status_code == 200, response.text
    assert started["resume_mode"] == "fresh"
    assert started["resume_from"] is None
