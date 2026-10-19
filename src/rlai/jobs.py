"""Background job management for training and evaluation subprocesses.

Long-running work never executes inside the API process.  Each job is a
separate Python process, controlled through a small JSON control file (works
identically on Windows and Linux) and observed through the database.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .config import TrainingConfig, data_dir, ensure_dirs, project_root
from .store import Store


class JobManager:
    def __init__(self, store: Store, paths: dict[str, Path] | None = None):
        self.store = store
        self.paths = paths or ensure_dirs()
        self.processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.reconcile()
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="rlai-watchdog")
        self._watchdog.start()

    # -- helpers -----------------------------------------------------------
    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(project_root() / "src") + (
            os.pathsep + existing if existing else ""
        )
        env["RLAI_DATA_DIR"] = str(data_dir())
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _run_dir(self, run_id: str) -> Path:
        path = self.paths["runs"] / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def reconcile(self) -> None:
        """Mark runs that claim to be active but have no live process."""
        active = {"starting", "running", "paused", "stopping"}
        for run in self.store.list_runs(limit=200):
            if run["status"] in active and run["id"] not in self.processes:
                pid = run.get("pid")
                alive = False
                if pid:
                    try:
                        os.kill(int(pid), 0)
                        alive = True
                    except (OSError, ValueError):
                        alive = False
                if not alive:
                    self.store.update_run(
                        run["id"],
                        status="interrupted",
                        message="process is no longer running",
                    )
                    self.store.log(run["id"], "WARNING", "run marked interrupted after restart")

    # -- training ----------------------------------------------------------
    def active_jobs(self) -> list[str]:
        """Ids of jobs whose process is still alive."""
        with self._lock:
            items = list(self.processes.items())
        return [job_id for job_id, process in items if process.poll() is None]

    def start_training(
        self,
        config: TrainingConfig,
        run_id: str,
        resume_from: str | None = None,
        resume_mode: str = "fresh",
        allow_concurrent: bool = False,
        config_overrides: list[str] | tuple[str, ...] | None = None,
    ) -> dict:
        if resume_mode not in ("fresh", "continue", "transfer"):
            raise ValueError(f"resume_mode must be fresh, continue or transfer, got {resume_mode!r}")
        if not allow_concurrent:
            running = [job for job in self.active_jobs() if not job.startswith("ev-")]
            if not running:
                active = self.store.latest_active_run()
                if active and active["id"] != run_id:
                    running = [active["id"]]
            if running:
                raise RuntimeError(
                    f"training run {running[0]} is already active; stop it first or start this "
                    "run with allow_concurrent=true (two trainers will fight for the same CPUs)"
                )
        run_dir = self._run_dir(run_id)
        config_path = run_dir / "config.json"
        config.save(config_path)
        control_path = run_dir / "control.json"
        control_path.write_text(json.dumps({"pause": False, "stop": False}), encoding="utf-8")
        log_path = run_dir / "stdout.log"
        log_file = open(log_path, "ab")
        command = [
            sys.executable,
            "-m",
            "rlai.cli",
            "train",
            "--run-id",
            run_id,
            "--config-file",
            str(config_path),
        ]
        command += ["--resume-mode", resume_mode]
        if resume_from:
            command += ["--resume-from", str(resume_from)]
        if config_overrides:
            command += ["--override-config", ",".join(str(f) for f in config_overrides)]
        process = subprocess.Popen(
            command,
            cwd=str(project_root()),
            env=self._env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        log_file.close()
        with self._lock:
            self.processes[run_id] = process
        self.store.update_run(run_id, status="starting", pid=process.pid, message="started")
        self.store.log(
            run_id,
            "INFO",
            f"launched training process pid={process.pid} resume_mode={resume_mode}"
            + (f" resume_from={resume_from}" if resume_from else ""),
        )
        return {"run_id": run_id, "pid": process.pid, "resume_mode": resume_mode}

    def control(self, run_id: str, *, pause: bool | None = None, stop: bool | None = None) -> dict:
        run_dir = self._run_dir(run_id)
        control_path = run_dir / "control.json"
        data = {"pause": False, "stop": False}
        if control_path.exists():
            try:
                data.update(json.loads(control_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        if pause is not None:
            data["pause"] = bool(pause)
        if stop is not None:
            data["stop"] = bool(stop)
            data["pause"] = False
        tmp = control_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, control_path)
        return data

    def cancel(self, run_id: str) -> None:
        with self._lock:
            process = self.processes.get(run_id)
        if process and process.poll() is None:
            self.control(run_id, stop=True)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

    # -- evaluation --------------------------------------------------------
    def start_evaluation(
        self,
        *,
        model_id: str,
        model_path: Path,
        mode: str,
        opponent: str,
        games: int,
        seed: int,
        match_seconds: float,
        device: str = "cpu",
        opponents: list[str] | None = None,
        sides: list[str] | tuple[str, ...] = ("blue",),
        mirror: bool = True,
        stochastic: bool = False,
        phase: str = "manual",
    ) -> str:
        eval_id = f"ev-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        run_dir = self._run_dir(f"eval-{eval_id}")
        log_path = run_dir / "stdout.log"
        log_file = open(log_path, "ab")
        command = [
            sys.executable,
            "-m",
            "rlai.cli",
            "evaluate",
            "--model",
            str(model_path),
            "--model-id",
            model_id,
            "--mode",
            mode,
            "--opponent",
            opponent,
            "--games",
            str(games),
            "--seed",
            str(seed),
            "--match-seconds",
            str(match_seconds),
            "--device",
            device,
            "--store",
            str(self.store.path),
            "--sides",
            ",".join(sides) or "blue",
            "--phase",
            phase,
        ]
        if opponents and len(opponents) > 1:
            command += ["--opponents", ",".join(opponents)]
        if not mirror:
            command += ["--no-mirror"]
        if stochastic:
            command += ["--stochastic"]
        process = subprocess.Popen(
            command,
            cwd=str(project_root()),
            env=self._env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        log_file.close()
        with self._lock:
            self.processes[eval_id] = process
        self.store.log(
            None,
            "INFO",
            f"evaluation {eval_id} started for {model_id} vs "
            f"{'+'.join(opponents) if opponents and len(opponents) > 1 else opponent} "
            f"({games} games, sides={','.join(sides)}, mirror={mirror})",
        )
        return eval_id

    def job_alive(self, run_id: str) -> bool:
        with self._lock:
            process = self.processes.get(run_id)
        return bool(process and process.poll() is None)

    # -- watchdog ----------------------------------------------------------
    def _watch(self) -> None:
        terminal = {"completed", "failed", "stopped", "interrupted"}
        while not self._stop.wait(2.0):
            finished: list[str] = []
            with self._lock:
                items = list(self.processes.items())
            for job_id, process in items:
                code = process.poll()
                if code is None:
                    continue
                finished.append(job_id)
                run = self.store.get_run(job_id)
                if run and run["status"] not in terminal:
                    status = "completed" if code == 0 else "failed"
                    self.store.update_run(
                        job_id,
                        status=status,
                        message=f"process exited with code {code}",
                    )
                    self.store.log(
                        job_id, "INFO" if code == 0 else "ERROR", f"process exited with code {code}"
                    )
            if finished:
                with self._lock:
                    for job_id in finished:
                        self.processes.pop(job_id, None)

    def shutdown(self) -> None:
        self._stop.set()
