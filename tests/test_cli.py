"""CLI entry points: run registration, resume planning, reports and live dry runs."""

from __future__ import annotations

import json

import pytest

from rlai import checkpoints, cli
from rlai.config import TrainingConfig
from rlai.store import Store


def _smoke_config(path, **overrides) -> TrainingConfig:
    config = TrainingConfig.load(path)
    if overrides:
        config = config.model_copy(update=overrides)
        config.save(path)
    return config


def test_train_registers_a_missing_run_id(data_root, monkeypatch, capsys):
    seen = {}

    def fake_run_training(config, run_id, store, resume_from=None, **kwargs):
        seen["run_id"] = run_id
        seen["steps"] = config.total_steps
        seen["store_has_run"] = store.get_run(run_id) is not None
        seen["resume_from"] = resume_from
        seen["resume_mode"] = kwargs.get("resume_mode")
        seen["resume_plan"] = kwargs.get("resume_plan")
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
    # automatic continuation is the default: a plan is built and finds no
    # checkpoint, so the trainer will start a new policy from scratch
    assert seen["resume_mode"] == "continue"
    assert seen["resume_from"] is None
    assert seen["resume_plan"] is not None
    assert seen["resume_plan"].will_resume is False
    assert seen["resume_plan"].candidates_found == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_train_plan_only_never_starts_training(data_root, monkeypatch, capsys):
    def boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("plan-only must not train")

    monkeypatch.setattr("rlai.trainer.run_training", boom)
    code = cli.main(
        [
            "train",
            "--config-file",
            "configs/smoke-1v1.json",
            "--auto-resume",
            "--plan-only",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["resume_plan"]["mode"] == "continue"
    assert payload["resume_plan"]["will_resume"] is False


def test_automatic_resume_hands_the_plan_to_the_trainer(data_root, monkeypatch, capsys):
    config = TrainingConfig.load("configs/smoke-1v1.json")
    policy_config = config.policy_config()
    manifest = checkpoints.save_checkpoint(
        data_root / "models",
        model_state=policy_config.build().state_dict(),
        policy_config=policy_config.to_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        step=2048,
        run_id="parent-run",
        mode=config.mode,
        training_config=config.to_dict(),
        extra={"obs_version": config.obs_version, "iteration": 4},
        label="final",
    )
    seen = {}

    def fake_run_training(cfg, run_id, store, resume_from=None, **kwargs):
        seen.update(resume_from=resume_from, plan=kwargs.get("resume_plan"), mode=kwargs.get("resume_mode"))
        return {"status": "completed", "step": 2048, "message": "fake"}

    monkeypatch.setattr("rlai.trainer.run_training", fake_run_training)
    # no --auto-resume / --resume-mode needed: continuation is the default
    code = cli.main(
        ["train", "--config-file", "configs/smoke-1v1.json", "--run-id", "child"]
    )
    assert code == 0
    assert seen["mode"] == "continue"
    assert seen["plan"].model_id == manifest["id"]
    assert seen["resume_from"] == str(data_root / "models" / manifest["path"])
    printed = capsys.readouterr().out
    assert manifest["id"] in printed


def test_resume_plan_command_reports_the_newest_checkpoint(data_root, capsys):
    config = TrainingConfig.load("configs/smoke-1v1.json")
    policy_config = config.policy_config()
    checkpoints.save_checkpoint(
        data_root / "models",
        model_state=policy_config.build().state_dict(),
        policy_config=policy_config.to_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        step=4096,
        run_id="parent",
        mode=config.mode,
        training_config=config.to_dict(),
        extra={"obs_version": config.obs_version},
    )
    code = cli.main(["resume-plan", "--config-file", "configs/smoke-1v1.json"])
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["will_resume"] is True
    assert plan["restored_step"] == 4096
    assert plan["mode"] == "continue"

    # a fresh plan never restores anything
    code = cli.main(["resume-plan", "--config-file", "configs/smoke-1v1.json", "--mode", "fresh"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["will_resume"] is False


def test_report_command_prints_and_rebuilds(data_root, capsys):
    store = Store(data_root / "rlai.sqlite3")
    run_id = store.create_run("report-run", "soccer-1v1", {}, 2000)
    assert cli.main(["report", "--run-id", run_id]) == 3
    assert "no report stored" in capsys.readouterr().err

    store.save_report(run_id, {"run_id": run_id, "status": "completed", "steps": {"completed": 128}})
    assert cli.main(["report", "--run-id", run_id, "--summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "completed"

    # rebuilding from the database works for runs that predate the report
    store2 = Store(data_root / "rlai.sqlite3")
    other = store2.create_run("old-run", "soccer-1v1", {}, 2000)
    assert cli.main(["report", "--run-id", other, "--rebuild"]) == 0
    rebuilt = json.loads(capsys.readouterr().out)
    assert rebuilt["run_id"] == other
    assert rebuilt["schema"] == "rlai-run-report-v1"

    # --rebuild also wins over a stored (possibly stale) report
    store2.save_report(run_id, {"run_id": run_id, "schema": "stale-marker"})
    assert cli.main(["report", "--run-id", run_id]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "stale-marker"
    assert cli.main(["report", "--run-id", run_id, "--rebuild"]) == 0
    fresh = json.loads(capsys.readouterr().out)
    assert fresh["schema"] == "rlai-run-report-v1"


def test_report_compare_command(data_root, capsys):
    store = Store(data_root / "rlai.sqlite3")
    first = store.create_run("a", "soccer-1v1", {}, 2000)
    second = store.create_run("b", "soccer-2v2", {}, 2000)
    assert cli.main(["report", "--compare", f"{first},{second}"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["comparable"] is False
    assert "not comparable" in payload["warning"]


@pytest.mark.slow
def test_evaluate_cli_side_balanced_and_suite(data_root, model_path, capsys):
    code = cli.main(
        [
            "evaluate",
            "--model",
            str(model_path),
            "--mode",
            "soccer-1v1",
            "--opponents",
            "idle,chase",
            "--games",
            "1",
            "--match-seconds",
            "5",
            "--sides",
            "blue,orange",
            "--phase",
            "manual",
        ]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["kind"] == "suite"
    assert set(result["results"]) == {"idle", "chase"}
    assert result["protocol"]["sides"] == ["blue", "orange"]
    assert result["diagnostic"]["opponents"] == ["idle"]
    assert result["strength"]["opponents"] == ["chase"]
    store = Store(data_root / "rlai.sqlite3")
    rows = store.list_evaluations(phase="manual")
    assert len(rows) == 2
    assert {row["opponent"] for row in rows} == {"idle", "chase"}


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


def _checkpoint(models_dir, config, *, step, hidden=None, label=""):
    from rlai.policy import PolicyConfig

    base = config.policy_config()
    policy = PolicyConfig(
        obs_size=base.obs_size,
        hidden=tuple(hidden or base.hidden),
        activation=base.activation,
    )
    model = policy.build()
    return checkpoints.save_checkpoint(
        models_dir,
        model_state=model.state_dict(),
        policy_config=policy.to_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        step=step,
        run_id="parent",
        mode=config.mode,
        training_config=config.to_dict(),
        extra={"obs_version": config.obs_version},
        label=label,
    )


def test_train_refuses_unusable_checkpoints_before_creating_a_run(data_root, monkeypatch, capsys):
    config = TrainingConfig.load("configs/smoke-1v1.json")
    _checkpoint(data_root / "models", config, step=2048, hidden=(64,))

    def boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("an unusable checkpoint must not start training")

    monkeypatch.setattr("rlai.trainer.run_training", boom)
    code = cli.main(["train", "--config-file", "configs/smoke-1v1.json"])

    assert code == 3
    assert "Refusing to start a fresh model" in capsys.readouterr().err
    assert Store(data_root / "rlai.sqlite3").list_runs() == []


def test_explicit_fresh_overrides_existing_checkpoints(data_root, monkeypatch, capsys):
    config = TrainingConfig.load("configs/smoke-1v1.json")
    _checkpoint(data_root / "models", config, step=2048, hidden=(64,))
    seen = {}

    def fake_run_training(cfg, run_id, store, resume_from=None, **kwargs):
        seen.update(resume_from=resume_from, mode=kwargs.get("resume_mode"))
        return {"status": "completed", "step": 0, "message": "fake"}

    monkeypatch.setattr("rlai.trainer.run_training", fake_run_training)
    code = cli.main(
        ["train", "--config-file", "configs/smoke-1v1.json", "--resume-mode", "fresh"]
    )
    assert code == 0
    assert seen["mode"] == "fresh"
    assert seen["resume_from"] is None
    assert "starts a new policy" in capsys.readouterr().err


def test_explicit_resume_from_overrides_the_newest_checkpoint(data_root, capsys):
    config = TrainingConfig.load("configs/smoke-1v1.json")
    older = _checkpoint(data_root / "models", config, step=256)
    _checkpoint(data_root / "models", config, step=4096, label="final")

    code = cli.main(
        [
            "train",
            "--config-file",
            "configs/smoke-1v1.json",
            "--resume-from",
            older["id"],
            "--plan-only",
        ]
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)["resume_plan"]
    assert plan["will_resume"] is True
    assert plan["model_id"] == older["id"]
    assert plan["restored_step"] == 256
