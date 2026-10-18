"""HTTP API contract used by the GUI."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from rlai.api import create_app


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
    assert len(defaults["presets"]) >= 7
    for preset in defaults["presets"]:
        config = preset["config"]
        assert config["obs_version"] in {1, 2}
        assert config["total_steps"] >= config["rollout_steps"] * config["n_workers"]
        assert 0 <= config["eval_interval"] <= 100_000
    curriculum = next(p for p in defaults["presets"] if p["id"] == "1v1-curriculum")
    assert curriculum["config"]["curriculum"] is True
    assert curriculum["config"]["eval_interval"] > 0


def test_validate_accepts_and_rejects(client):
    config = client.get("/api/config/defaults").json()["training"]
    config["total_steps"] = 2000
    config["rollout_steps"] = 256
    config["n_workers"] = 1
    config["minibatch_size"] = 256
    assert client.post("/api/config/validate", json=config).status_code == 200
    bad = dict(config, mode="hoops")
    assert client.post("/api/config/validate", json=bad).status_code == 422
    assert client.post("/api/config/validate", json=dict(config, obs_version=3)).status_code == 422
    assert client.post("/api/config/validate", json=dict(config, eval_interval=-1)).status_code == 422
    legacy = dict(config, obs_version=1, reward={**config["reward"], "demo": 0.4})
    legacy["reward"].pop("bump", None)
    assert client.post("/api/config/validate", json=legacy).status_code == 200


def test_run_lifecycle(client, monkeypatch):
    started = {}

    def fake_start(config, run_id, resume_from=None):
        started["config"] = config
        started["run_id"] = run_id
        started["resume_from"] = resume_from

    monkeypatch.setattr(client.app.state.jobs, "start_training", fake_start)
    payload = client.get("/api/config/defaults").json()["training"]
    payload.update(
        run_name="api test",
        total_steps=2000,
        rollout_steps=256,
        n_workers=1,
        minibatch_size=256,
        mode="soccer-1v1",
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
    bad = dict(payload, resume_from="not-a-model")
    assert client.post("/api/runs", json=bad).status_code == 400


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
