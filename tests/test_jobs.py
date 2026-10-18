"""JobManager control files and subprocess environment."""

from __future__ import annotations

import json

from rlai.config import ensure_dirs
from rlai.jobs import JobManager
from rlai.store import Store


def test_control_files_are_atomic_and_sticky(tmp_path, monkeypatch):
    monkeypatch.setenv("RLAI_DATA_DIR", str(tmp_path))
    store = Store(tmp_path / "test.db")
    run_id = store.create_run("jobs", "soccer-1v1", {}, 1000)
    jobs = JobManager(store, ensure_dirs(tmp_path))
    try:
        assert jobs.control(run_id, pause=True) == {"pause": True, "stop": False}
        # a stop request clears pause and persists
        assert jobs.control(run_id, stop=True) == {"pause": False, "stop": True}
        on_disk = json.loads((tmp_path / "runs" / run_id / "control.json").read_text())
        assert on_disk == {"pause": False, "stop": True}
        assert not jobs.job_alive(run_id)
    finally:
        jobs._stop.set()


def test_environment_contains_pythonpath_and_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RLAI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "/existing")
    store = Store(tmp_path / "test.db")
    jobs = JobManager(store, ensure_dirs(tmp_path))
    try:
        env = jobs._env()
        assert env["RLAI_DATA_DIR"] == str(tmp_path)
        assert env["PYTHONPATH"].split(":")[0].endswith("/src")
        assert env["PYTHONPATH"].endswith("/existing")
    finally:
        jobs._stop.set()
