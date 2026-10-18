"""SQLite store behaviour that the API and GUI depend on."""

from __future__ import annotations

from rlai.store import Store


def test_run_lifecycle(tmp_path):
    store = Store(tmp_path / "test.db")
    run_id = store.create_run("unit", "soccer-1v1", {"n_workers": 1}, 1000)
    run = store.get_run(run_id)
    assert run["status"] == "queued"
    assert run["name"] == "unit"
    store.update_run(run_id, status="completed", step=1010, message="done")
    run = store.get_run(run_id)
    assert run["status"] == "completed" and run["step"] == 1010
    assert [r["id"] for r in store.list_runs()] == [run_id]


def test_metrics_and_logs(tmp_path):
    store = Store(tmp_path / "test.db")
    run_id = store.create_run("metrics", "soccer-1v1", {}, 100)
    store.log_metric(run_id, 10, "charts/episodic_return", 1.5)
    store.log_metric(run_id, 20, "charts/episodic_return", 2.5)
    store.log_metrics(run_id, 20, {"losses/policy": 0.1, "time/sps": 900})
    metrics = store.get_metrics(run_id)
    assert len(metrics) == 4
    assert store.metric_names(run_id) == sorted({m["name"] for m in metrics})
    store.log(run_id, "WARN", "hello")
    logs = store.tail_logs(run_id)
    assert logs[-1]["message"] == "hello"
    assert logs[-1]["level"] == "WARN"


def test_episode_summary(tmp_path):
    store = Store(tmp_path / "test.db")
    run_id = store.create_run("episodes", "soccer-1v1", {}, 1000)
    for outcome, reward in (("win", 10.0), ("loss", -10.0), ("draw", 0.0)):
        store.log_episode(run_id, 100, reward, 30, 1 if outcome == "win" else 0, 0, outcome)
    summary = store.episode_summary(run_id)
    assert summary["count"] == 3
    assert summary["wins"] == 1 and summary["losses"] == 1 and summary["draws"] == 1


def test_settings(tmp_path):
    store = Store(tmp_path / "test.db")
    assert store.get_setting("missing", "fallback") == "fallback"
    store.set_setting("theme", "dark")
    assert store.get_setting("theme") == "dark"


def test_model_sync_and_scores(tmp_path):
    store = Store(tmp_path / "test.db")
    manifest = {
        "id": "m-1",
        "path": "m-1/model.pt",
        "step": 10,
        "created_at": "2026-01-01T00:00:00+00:00",
        "state": "candidate",
        "mode": "soccer-1v1",
    }
    store.sync_models([manifest])
    models = store.list_models()
    assert models[0]["id"] == "m-1"
    store.set_model_eval_score("m-1", 0.75)
    assert store.list_models()[0]["eval_score"] == 0.75
