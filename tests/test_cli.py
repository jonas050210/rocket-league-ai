"""CLI entry points: run registration, live preflight and dry runs."""

from __future__ import annotations

import json

from rlai import checkpoints, cli


def test_train_registers_a_missing_run_id(data_root, monkeypatch, capsys):
    seen = {}

    def fake_run_training(config, run_id, store, resume_from=None):
        seen["run_id"] = run_id
        seen["steps"] = config.total_steps
        seen["store_has_run"] = store.get_run(run_id) is not None
        return {"status": "completed", "step": 0, "message": "fake"}

    monkeypatch.setattr("rlai.trainer.run_training", fake_run_training)
    code = cli.main(
        [
            "train",
            "--config-file",
            "configs/smoke-1v1.json",
            "--run-id",
            "manual-run",
            "--max-steps",
            "4000",
        ]
    )
    assert code == 0
    assert seen["run_id"] == "manual-run"
    assert seen["steps"] == 4000
    # the trainer (not the CLI) is responsible for registering a missing row
    assert seen["store_has_run"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_live_preflight_exits_nonzero_without_windows(data_root, capsys):
    code = cli.main(["live", "--preflight"])
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["platform"] == "Linux"
    assert any(check["name"] == "windows" and not check["ok"] for check in report["checks"])


def test_live_dry_run_writes_files(data_root, models_dir, monkeypatch, capsys):
    model_id = checkpoints.list_models(models_dir)[0]["id"]
    code = cli.main(
        [
            "live",
            "--mode",
            "soccer-2v2",
            "--model-id",
            model_id,
            "--human-team",
            "orange",
            "--fill",
            "policy",
            "--dry-run",
        ]
    )
    assert code == 0, capsys.readouterr().err
    record = json.loads(capsys.readouterr().out)
    assert record["dry_run"] is True and record["launched"] is False
    assert record["human_team"] == "orange" and record["fill"] == "policy"
    assert record["model_id"] == model_id
    assert record["validation"]["ok"] is True

    code = cli.main(["live", "--list"])
    assert code == 0
    listed = json.loads(capsys.readouterr().out)
    assert any(item["directory"] == record["directory"] for item in listed)


def test_live_launch_fails_clearly_off_windows(data_root, models_dir, capsys):
    model_id = checkpoints.list_models(models_dir)[0]["id"]
    code = cli.main(["live", "--model-id", model_id, "--mode", "soccer-1v1"])
    assert code == 2
    captured = capsys.readouterr()
    assert "live match unavailable" in captured.err
    assert "preflight" in captured.err
