"""The API behind the new GUI features.

The GUI is the central control centre, so every feature it offers needs a
documented endpoint.  The tests below check the contract: what the endpoint
answers, what it refuses, and - most importantly - that nothing starts a
simulation or a training run without an explicit confirmation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rlai.api import create_app
from rlai.store import Store


@pytest.fixture()
def client(data_root, models_dir):
    app = create_app()
    with TestClient(app) as test_client:
        test_client.app = app
        yield test_client


@pytest.fixture()
def store(data_root):
    return Store(data_root / "rlai.sqlite3")


# ------------------------------------------------------------------- stats
def test_the_stats_endpoint_answers_without_a_run(client):
    payload = client.get("/api/stats").json()
    assert payload["available"] is False
    assert "no run" in payload["reason"]


def test_the_stats_endpoint_describes_a_run(client, store, data_root):
    run_id = store.create_run("stats", "soccer-1v1", {}, 2000, run_id="run-1")
    store.update_run(run_id, status="running", step=64, total_steps=2000)
    store.log_metric(run_id, 64, "charts/episodic_return", 1.5)
    store.log_metric(run_id, 64, "time/sps", 120.0)
    session = store.start_session(run_id, started_at="2026-01-01T00:00:00+00:00")
    store.finish_session(
        session,
        ended_at="2026-01-01T00:00:12+00:00",
        duration_s=12.0,
        status="completed",
        end_step=64,
    )
    payload = client.get("/api/stats", params={"run_id": run_id}).json()
    assert payload["available"] is True
    assert payload["timing"]["accumulated_s"] == pytest.approx(12.0)
    assert payload["progress"]["step"] == 64
    assert payload["learning"]["charts/episodic_return"]["last"] == pytest.approx(1.5)
    assert payload["performance"]["steps_per_second"] == pytest.approx(120.0)
    assert len(payload["history"]) == 1


def test_the_stats_endpoint_rejects_an_unknown_run(client):
    assert client.get("/api/stats", params={"run_id": "nope"}).status_code == 404


def test_the_stats_overview_lists_the_runs(client, store):
    store.create_run("one", "soccer-1v1", {}, 2000, run_id="run-1")
    store.create_run("two", "soccer-1v1", {}, 2000, run_id="run-2")
    payload = client.get("/api/stats/overview").json()
    assert payload["count"] == 2
    assert {row["id"] for row in payload["runs"]} == {"run-1", "run-2"}


# ------------------------------------------------------------------- teams
def test_a_team_can_be_saved_read_and_deleted(client):
    team = {
        "name": "duo",
        "mode": "soccer-2v2",
        "bots": [
            {"name": "striker", "difficulty": "pro", "behaviour": "aggressive"},
            {"name": "keeper", "difficulty": "rock", "behaviour": "defensive"},
        ],
    }
    response = client.post("/api/teams", json=team)
    assert response.status_code == 200
    assert response.json()["size"] == 2

    listed = client.get("/api/teams").json()
    assert listed["count"] == 1
    assert listed["teams"][0]["name"] == "duo"

    assert client.get("/api/teams/duo").status_code == 200
    assert client.delete("/api/teams/duo").json()["deleted"] == "duo"
    assert client.get("/api/teams/duo").status_code == 404


def test_an_invalid_team_is_rejected_with_a_message(client):
    response = client.post(
        "/api/teams",
        json={"name": "broken", "bots": [{"name": "x", "difficulty": "legend"}]},
    )
    assert response.status_code == 400
    assert "legend" in response.json()["detail"]


def test_a_team_without_bots_is_rejected(client):
    response = client.post("/api/teams", json={"name": "empty", "bots": []})
    assert response.status_code == 422


# ------------------------------------------------------------- difficulties
def test_the_difficulty_catalogue_is_honest(client):
    payload = client.get("/api/difficulties").json()
    names = {profile["name"] for profile in payload["profiles"]}
    assert names == {"rock", "tiny", "medium", "pro"}
    # the percentages are targets, not measurements
    assert payload["targets_are_measured"] is False
    for profile in payload["profiles"]:
        assert profile["target_is_measured"] is False


def test_the_behaviour_profiles_are_listed(client):
    payload = client.get("/api/bot-behaviour").json()
    assert {profile["name"] for profile in payload["profiles"]} == {
        "aggressive",
        "defensive",
        "balanced",
        "adaptive",
    }


def test_the_behaviour_preview_shows_a_decision(client):
    payload = client.post(
        "/api/bot-behaviour/preview",
        json={"difficulty": "pro", "behaviour": "aggressive", "overrides": {}},
    ).json()
    assert payload["decision"]["intent"] in {
        "shoot",
        "attack",
        "rotate",
        "chase",
    }
    assert payload["brain"]["profile"] == "aggressive"


def test_the_behaviour_preview_rejects_an_unknown_difficulty(client):
    response = client.post(
        "/api/bot-behaviour/preview", json={"difficulty": "legend", "behaviour": "balanced"}
    )
    assert response.status_code == 400


# ------------------------------------------------------------ player memory
def test_the_player_memory_starts_empty(client):
    payload = client.get("/api/player-memory").json()
    assert payload["summary"]["players"] == 0
    assert "never part of a checkpoint" in payload["note"]


def test_a_note_for_an_unknown_player_is_a_404(client):
    response = client.post("/api/player-memory/note", json={"name": "nobody", "note": "hi"})
    assert response.status_code == 404


def test_the_player_memory_can_be_exported(client, store):
    from rlai.player_memory import PlayerMemory, rows_from_memory

    memory = PlayerMemory()
    memory.remember("Nitro", won=True, goals_for=2)
    store.replace_player_memory(rows_from_memory(memory))

    payload = client.get("/api/player-memory").json()
    assert payload["summary"]["players"] == 1
    exported = client.get("/api/player-memory/export").json()
    assert exported["schema"] == "rlai-player-memory-v1"
    assert exported["players"][0]["name"] == "Nitro"


def test_the_player_memory_can_be_reset(client, store):
    from rlai.player_memory import PlayerMemory, rows_from_memory

    memory = PlayerMemory()
    memory.remember("Nitro")
    store.replace_player_memory(rows_from_memory(memory))
    assert client.post("/api/player-memory/reset").json()["deleted"] == 1
    assert client.get("/api/player-memory").json()["summary"]["players"] == 0


# ------------------------------------------------------------------ rivals
def test_a_rival_can_be_registered_and_listed(client):
    response = client.post(
        "/api/rivals", json={"name": "pro-bot", "kind": "checkpoint", "reference": "m-1"}
    )
    assert response.status_code == 200
    payload = client.get("/api/rivals").json()
    assert payload["count"] == 1
    assert payload["rivals"][0]["name"] == "pro-bot"
    # a rival without a measured game has no win rate
    assert payload["rivals"][0]["win_rate"] is None
    assert payload["bottleneck"]["available"] is False


def test_a_rival_is_removed(client):
    client.post("/api/rivals", json={"name": "pro-bot"})
    assert client.delete("/api/rivals/pro-bot").status_code == 200
    assert client.delete("/api/rivals/pro-bot").status_code == 404


def test_the_rival_trend_of_an_unknown_rival_says_so(client):
    payload = client.get("/api/rivals/nobody/trend").json()
    assert payload["available"] is False
    assert "unknown rival" in payload["reason"]


def test_the_rival_registry_uses_the_measured_evaluations(client, store):
    store.add_evaluation(
        {
            "mode": "soccer-1v1",
            "opponent": "difficulty:pro",
            "games": 4,
            "wins": 3,
            "losses": 1,
            "draws": 0,
            "goals_for": 7,
            "goals_against": 2,
            "model_id": "m-1",
        }
    )
    payload = client.get("/api/rivals").json()
    rival = payload["rivals"][0]
    assert rival["name"] == "difficulty:pro"
    assert rival["win_rate"] == pytest.approx(0.75)
    assert payload["bottleneck"]["available"] is True


# ---------------------------------------------------------------- analysis
def test_the_analysis_endpoint_reports_the_limits(client, store):
    run_id = store.create_run("r", "soccer-1v1", {}, 2000, run_id="run-1")
    payload = client.get("/api/analysis", params={"run_id": run_id}).json()
    assert payload["replays"]["available"] is False
    assert payload["evaluations"]["available"] is False
    assert payload["episodes"]["available"] is False


# --------------------------------------------------------------------- lab
def test_the_lab_overview_lists_the_opponents(client):
    payload = client.get("/api/lab").json()
    assert payload["opponents"]["strength"]
    assert payload["opponents"]["diagnostic"]
    assert "confirmation" in payload["note"]


def test_a_lab_plan_never_starts_a_simulation(client, models_dir):
    response = client.post(
        "/api/lab/plan", json={"opponents": ["chase"], "games": 2, "match_seconds": 5}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["executed"] is False
    assert payload["total_games"] == 2
    assert payload["opponent_classes"]["chase"]


def test_a_lab_plan_rejects_an_unknown_opponent(client):
    response = client.post("/api/lab/plan", json={"opponents": ["not-a-bot"]})
    assert response.status_code == 400


def test_a_lab_run_needs_an_explicit_confirmation(client, models_dir):
    response = client.post(
        "/api/lab/run",
        json={"opponents": ["chase"], "games": 1, "match_seconds": 5, "execute": False},
    )
    assert response.status_code == 400
    assert "not confirmed" in response.json()["detail"]


def test_a_lab_run_without_a_checkpoint_is_refused(client):
    response = client.post(
        "/api/lab/run", json={"opponents": ["chase"], "games": 1, "execute": True}
    )
    assert response.status_code == 400
    assert "no checkpoint" in response.json()["detail"]
