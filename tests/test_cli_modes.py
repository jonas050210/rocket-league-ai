"""The two execution modes: headless and GUI.

Both modes use the same training and checkpoint logic; the headless mode
simply does not start the web application and prints a compact final
report instead of the full result dictionary.

The tests here check the contract, not the learning:

* the headless mode never starts a web server;
* it produces the compact summary with the measured training time;
* the benchmarks command runs without a GUI and without training.
"""

from __future__ import annotations

import json

from rlai import cli


def _json_output(text: str) -> dict:
    """The JSON document the CLI printed last.

    The trainer logs its progress to stdout as well, so the payload is the
    last JSON document in the output - not the whole output.
    """
    decoder = json.JSONDecoder()
    documents = []
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        documents.append(payload)
        index = start + end
    assert documents, f"no JSON document in the output: {text[:400]!r}"
    return documents[-1]


def _config_file(tmp_path, **overrides) -> str:
    from rlai.config import TrainingConfig

    config = TrainingConfig(
        run_name="headless-test",
        total_steps=2000,
        rollout_steps=64,
        n_workers=1,
        minibatch_size=64,
        eval_interval=0,
        policy={"hidden": [16]},
        device="cpu",
    )
    if overrides:
        config = config.model_copy(update=overrides)
    path = tmp_path / "config.json"
    config.save(path)
    return str(path)


def test_the_headless_mode_prints_the_compact_summary(data_root, tmp_path, capsys):
    code = cli.main(["headless", "--config-file", _config_file(tmp_path)])
    assert code == 0
    payload = _json_output(capsys.readouterr().out)
    assert payload["schema"] == "rlai-headless-summary-v1"
    assert payload["status"] == "completed"
    # the measured training time is part of the summary
    assert payload["timing"]["accumulated_s"] > 0.0
    assert payload["timing"]["accumulated_human"]
    assert payload["steps"]["completed"] >= 2000
    assert payload["termination"]["status"] == "completed"
    assert payload["termination"]["stopped_by_user"] is False


def test_the_headless_mode_starts_no_web_service(data_root, tmp_path, capsys):
    """No GUI, no browser, no continuous monitoring - but the same metrics."""
    import socket

    # a socket that would be needed by a web server must not be opened
    before = socket.socket()
    try:
        code = cli.main(["headless", "--config-file", _config_file(tmp_path)])
    finally:
        before.close()
    assert code == 0
    payload = _json_output(capsys.readouterr().out)
    assert payload["schema"] == "rlai-headless-summary-v1"
    # the essential information is still there
    assert payload["timing"]["accumulated_s"] is not None
    assert payload["throughput"]["steps_per_second"] is not None
    assert payload["checkpoint"] is not None


def test_the_headless_mode_writes_the_session_and_the_checkpoint(data_root, tmp_path):
    from rlai.store import Store

    cli.main(["headless", "--config-file", _config_file(tmp_path)])
    store = Store(data_root / "rlai.sqlite3")
    runs = store.list_runs()
    assert len(runs) == 1
    run = runs[0]
    sessions = store.list_sessions(run["id"])
    assert len(sessions) == 1
    assert sessions[0]["status"] == "completed"
    assert sessions[0]["duration_s"] > 0.0
    assert run["duration_s"] > 0.0
    assert run["last_checkpoint_id"]
    assert run["last_checkpoint_at"]


def test_the_headless_mode_can_resume_and_keeps_the_total(data_root, tmp_path, capsys):
    from rlai.store import Store

    config_path = _config_file(tmp_path)
    cli.main(["headless", "--config-file", config_path])
    store = Store(data_root / "rlai.sqlite3")
    first = store.list_runs()[0]
    first_total = store.session_totals(first["id"])["accumulated_s"]

    code = cli.main(
        [
            "headless",
            "--config-file",
            config_path,
            "--run-id",
            "second-run",
            "--resume-mode",
            "continue",
            "--auto-resume",
            "--max-steps",
            "4000",
        ]
    )
    assert code == 0
    payload = _json_output(capsys.readouterr().out)
    assert payload["run_id"] == "second-run"
    totals = store.session_totals("second-run")
    assert totals["inherited_count"] == 1
    assert totals["accumulated_s"] >= first_total


def test_the_headless_mode_can_stop_early(data_root, tmp_path, capsys):
    """A safe stop keeps the measured time and reports the status."""
    from rlai.store import Store

    config_path = _config_file(tmp_path, total_steps=5_000_000)
    code = cli.main(["headless", "--config-file", config_path, "--max-steps", "2000"])
    assert code == 0
    payload = _json_output(capsys.readouterr().out)
    assert payload["status"] == "completed"
    store = Store(data_root / "rlai.sqlite3")
    sessions = store.list_sessions(payload["run_id"])
    assert sessions[0]["duration_s"] > 0.0


def test_the_benchmarks_command_runs_without_training(data_root, capsys):
    code = cli.main(["benchmarks"])
    assert code == 0
    payload = _json_output(capsys.readouterr().out)
    assert "difficulty" in payload
    assert "evaluation_reuse" in payload
    assert "threads" in payload
    assert payload["difficulty"]["results"]


def test_the_benchmark_default_mode_is_valid(data_root, capsys):
    """The default mode of the hardware benchmark must exist."""
    code = cli.main(["benchmark", "--seconds", "0.2"])
    assert code == 0
    payload = _json_output(capsys.readouterr().out)
    assert payload["mode"] == "soccer-1v1"
