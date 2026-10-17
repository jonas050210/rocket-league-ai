import json

from rocket_league_ai.config import TrainingConfig
from rocket_league_ai.metadata import experiment_metadata
from rocket_league_ai.persistence import atomic_json
from rocket_league_ai.store import Store


def test_metadata_and_immutable_config_persist_without_absolute_repo_path(tmp_path):
    store = Store(tmp_path)
    config = TrainingConfig(seed=456)
    metadata = experiment_metadata()
    run = store.create_run(config, metadata)
    assert run["config_hash"] == config.digest()
    assert str(tmp_path) not in json.dumps(metadata)
    recorded = metadata["packages"]["rlgym-learn"]
    assert recorded is None or recorded == "2.0.0"  # None when the extra is not installed
    store.update_run(run["id"], status="paused", metrics={"steps": 512})
    store.close()
    reopened = Store(tmp_path)
    reopened.interrupt_unfinished()
    result = reopened.run(run["id"])
    assert result["status"] == "interrupted" and result["metrics"]["steps"] == 512
    assert result["config"]["seed"] == 456
    assert "No automatic restart" in result["error"]
    assert json.loads((reopened.run_dir(run["id"]) / "config.json").read_text())["seed"] == 456
    reopened.close()


def test_atomic_metadata_replacement(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"version": 1})
    atomic_json(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 2}
    assert list(tmp_path.iterdir()) == [path]


def test_plan_and_revision_are_one_atomic_transaction(tmp_path):
    import sqlite3

    import pytest

    store = Store(tmp_path)
    original = store.setting("plan")
    with store.db:
        store.db.execute("""
            CREATE TRIGGER fault_revision BEFORE INSERT ON settings
            WHEN NEW.key='plan_revision' BEGIN
                SELECT RAISE(ABORT, 'injected revision write failure');
            END;
        """)
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        store.set_settings({"plan": {"not": "committed"}, "plan_revision": 2})
    assert store.setting("plan") == original and store.setting("plan_revision") == 1
    store.close()


def test_pending_reviews_expire_across_backend_restart(tmp_path):
    from rocket_league_ai.orchestration import Orchestrator

    first = Orchestrator(tmp_path)
    proposal = first.propose("apply_config", {"config": TrainingConfig(device="cpu").model_dump(mode="json")})
    assert first.store.proposal(proposal["id"])["status"] == "pending"
    first.close()
    second = Orchestrator(tmp_path)
    try:
        assert second.store.proposal(proposal["id"])["status"] == "expired"
        assert second.store.setting("plan")["device"] == "cuda:0"
        assert second.active is None
    finally:
        second.close()


def test_constructor_failure_releases_backend_ownership(tmp_path):
    import pytest

    from rocket_league_ai.domain import AppError
    from rocket_league_ai.orchestration import Orchestrator
    from rocket_league_ai.persistence import lease_available

    (tmp_path / "access-token").write_text("invalid")
    with pytest.raises(AppError, match="Invalid access-token"):
        Orchestrator(tmp_path)
    assert lease_available(tmp_path / "backend.lock")
    (tmp_path / "access-token").write_text("a" * 48)
    manager = Orchestrator(tmp_path)
    manager.close()
