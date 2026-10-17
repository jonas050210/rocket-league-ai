"""Independent local backend: ownership, durable proposals, supervision and recovery."""

import json
import os
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .benchmark import BenchmarkSpec, aggregate_iterations, combine_repeats, recommend
from .checkpoints import CheckpointStore, file_hash, retention_candidates
from .config import TrainingConfig, changes, digest
from .domain import ACTIVE, TERMINAL, AppError, EvaluationResult, RunStatus, control_target
from .evaluation import (
    EvaluationSpec,
    GameRecord,
    LeagueSpec,
    compare_evaluations,
    elo_ratings,
    experiment_journal,
)
from .evaluation import summarize as summarize_evaluation
from .metadata import experiment_metadata, package_versions, require_recovery_compatible
from .monitoring import SystemMonitor, next_experiment, recommendations
from .persistence import Lease, atomic_json, lease_available
from .store import Store, checked_id, new_id

EVENT_PREFIX = "RLAI_EVENT "


@dataclass
class LiveWorker:
    run_id: str
    process: subprocess.Popen
    heartbeat: float = field(default_factory=time.monotonic)
    progress_age: float = 0.0
    reader: threading.Thread | None = None
    failed: bool = False
    exit_cleanup_sent: bool = False


class Orchestrator:
    def __init__(self, root: Path, *, stall_timeout: float = 120, shutdown_timeout: float = 20):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lease = Lease(self.root / "backend.lock")
        try:
            self.store = Store(self.root)
            self.store.interrupt_unfinished()
            self.lock = threading.RLock()
            self.active: LiveWorker | None = None
            self.stop_event = threading.Event()
            self.closing = False
            self.stall_timeout, self.shutdown_timeout = stall_timeout, shutdown_timeout
            self.checkpoint_stores: dict[str, CheckpointStore] = {}
            # At most one auxiliary workload (benchmark or evaluation) at a time;
            # both are forbidden while a training worker is active and vice versa.
            self.auxiliary: str | None = None
            self.benchmark_stop = threading.Event()
            self.evaluation_stop = threading.Event()
            self.league_stop = threading.Event()
            self.evaluation_process: subprocess.Popen | None = None
            self._workspace_bytes_cache: tuple[float, int] = (0.0, 0)
            self.token = self._load_token()
            self.system = SystemMonitor()
            self.watchdog = threading.Thread(target=self._watch, daemon=True, name="worker-supervisor")
            self.store.event("info", "backend_started", "Backend ready. Training starts only after explicit user confirmation.")

            self.watchdog.start()
        except BaseException:
            # Failed startup must not retain ownership or background threads.
            if hasattr(self, "stop_event"):
                self.stop_event.set()
            if hasattr(self, "system"):
                self.system.close()
            if hasattr(self, "store"):
                self.store.close()
            self.lease.close()
            raise

    def _load_token(self):
        path = self.root / "access-token"
        if path.is_file():
            token = path.read_text().strip()
            if len(token) < 32:
                raise AppError("Invalid access-token file. Stop the backend and regenerate the local capability.")
            path.chmod(0o600)
            return token
        token = secrets.token_urlsafe(32)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return token

    def checkpoints(self, run_id):
        self.store.run(run_id)
        if run_id not in self.checkpoint_stores:
            self.checkpoint_stores[run_id] = CheckpointStore(self.store.run_dir(run_id))
        return self.checkpoint_stores[run_id]

    def snapshot(self):
        with self.lock:
            # Benchmark measurements are real training runs, but they are not
            # experiments. Keeping them out of history/checkpoint inventory is
            # what makes "20 s per candidate" cheap enough to run often.
            all_runs = self.store.runs()
            runs = [run for run in all_runs if (run.get("metadata") or {}).get("purpose") != "benchmark"]
            models = [{**checkpoint, "run_id": run["id"], "run_name": run["name"]}
                      for run in runs for checkpoint in self.checkpoints(run["id"]).inventory()]
            latest = {}
            for run in runs:
                valid = [item for item in models if item["run_id"] == run["id"] and item["integrity"] == "valid"]
                item = max(valid, key=lambda value: (value["step"], value["created"]), default=None)
                latest[run["id"]] = item["id"] if item else None
            system = self.system.snapshot()
            active = self.store.run(self.active.run_id) if self.active else None
            if active and (active.get("metadata") or {}).get("purpose") == "benchmark":
                active = None
            return {
                "runs": runs, "active_run": active, "checkpoints": models,
                "latest_valid": latest, "designated_best": self.store.setting("designated_best"),
                "plan": self.store.setting("plan"), "plan_revision": self.store.setting("plan_revision"),
                "system": system, "recommendations": recommendations(
                    runs, system,
                    evaluations=self.store.evaluations(limit=10, light=True),
                    metric_history=[event["data"] for event in self.store.events_for_run(runs[0]["id"], "training_metrics")] if runs else [],
                    benchmark=self._benchmark_snapshot(), workspace_bytes=self.workspace_bytes(),
                ),
                "training_packages": package_versions(),
                "evaluation_available": all(version is not None for version in package_versions().values()),
                "visualization_available": True,
                "benchmark": self._benchmark_snapshot(),
                "benchmark_sessions": self.store.benchmarks(limit=10),
                "benchmark_adoption": self.store.setting("benchmark_adoption"),
                "evaluations": self.store.evaluations(light=True),
                "evaluation_comparison": compare_evaluations(self.store.evaluations(light=True)),
                "workspace_bytes": self.workspace_bytes(),
                "retention_candidates": retention_candidates(models, self.store.setting("designated_best"), latest),
                "recovery_candidates": [
                    {"run_id": run["id"], "name": run["name"], "status": run["status"],
                     "checkpoint_id": latest[run["id"]]}
                    for run in runs if run["status"] in {"failed", "interrupted", "stopped"} and latest.get(run["id"])
                ],
                "leagues": self.store.leagues(),
                "checkpoint_ratings": elo_ratings(self.store.evaluations(light=True)),
                "experiment_journal": experiment_journal(runs, self.store.evaluations(light=True)),
                "next_experiment": next_experiment(runs, self.store.evaluations(light=True),
                                                   self._benchmark_snapshot(), self.store.setting("plan")),
                "worker_health": {
                    "connected": self.active is not None,
                    "heartbeat_age_seconds": time.monotonic() - self.active.heartbeat if self.active else None,
                    "progress_age_seconds": self.active.progress_age if self.active else None,
                    "orphan_lease_held": self.active is None and not lease_available(self.root / "active-worker.lock"),
                },
            }

    def _require_idle(self):
        if self.closing:
            raise AppError("Backend is shutting down. Reconnect after it has restarted.")
        if self.auxiliary:
            raise AppError(f"A {self.auxiliary} is in progress. Wait for it to finish or stop it first.")
        if self.active:
            raise AppError("A worker is still active. Stop it and wait for the terminal acknowledgement.")
        if not lease_available(self.root / "active-worker.lock"):
            raise AppError("A previous worker is still releasing its lease. Wait for graceful shutdown; no duplicate run was started.")

    def _target(self, payload, *, prepared=False):
        allowed = {"run_id", "checkpoint_id"} | ({"manifest_sha256", "expected_best"} if prepared else set())
        if not payload.keys() <= allowed or not {"run_id", "checkpoint_id"} <= payload.keys():
            raise AppError("Select a run_id and checkpoint_id from the inventory; unknown fields are not accepted.", 422)
        run = self.store.run(payload["run_id"])
        checkpoints = self.checkpoints(run["id"])
        checkpoint = checkpoints.require_valid(payload["checkpoint_id"], run["config_hash"])
        manifest_hash = file_hash(checkpoints.path(checkpoint["id"]) / "manifest.json")
        if prepared and manifest_hash != payload["manifest_sha256"]:
            raise AppError("Checkpoint changed since this proposal was reviewed. Create a new proposal.")
        return run, checkpoint, manifest_hash

    def propose(self, kind: str, payload: dict) -> dict:
        with self.lock:
            rationale, impact = "", ""
            if kind in {"start_run", "apply_config"}:
                if not payload.keys() <= {"config"}:
                    raise AppError("Only a validated config is accepted for this action.", 422)
                self._require_idle()
                plan = TrainingConfig.model_validate(self.store.setting("plan"))
                config = TrainingConfig.model_validate(payload.get("config", plan.model_dump()))
                diff = changes(plan, config)
                payload = {"config": config.model_dump(mode="json"), "changes": diff}
                rationale = (
                    "Validate the headless 1v1 kickoff-only PPO foundation, not competitive strength. "
                    + (f"This changes {len(diff)} approved-plan fields; review each difference." if diff else "Uses the approved training plan without strategy changes.")
                )
                if kind == "start_run":
                    missing = [name for name, version in package_versions().items() if version is None]
                    if missing:
                        raise AppError("Training extra is missing: " + ", ".join(missing) + ". Run uv sync --frozen --extra training.")
                    impact = (f"Starts one independent worker with {config.n_proc} simulation process(es), "
                              f"up to {config.total_timesteps:,} agent steps on {config.device}. "
                              "Consumes CPU/RAM and CUDA resources if selected; creates checkpoints, deletes nothing. "
                              "CPU mode is an explicit correctness test, not target-machine throughput.")
                else:
                    impact = "Replaces the approved plan for future runs only. No existing run config or checkpoint is modified; no training starts."
            elif kind == "recover_run":
                self._require_idle()
                run, checkpoint, checksum = self._target(payload)
                if run["status"] not in TERMINAL or run["status"] == RunStatus.COMPLETED:
                    raise AppError("Recover a stopped, failed, or interrupted run, not a completed or active run.")
                if checkpoint["step"] >= run["config"]["total_timesteps"]:
                    raise AppError("This checkpoint has already reached its original step budget. Approve a separate new experiment instead.")
                require_recovery_compatible(run["metadata"])
                payload = {"run_id": run["id"], "checkpoint_id": checkpoint["id"], "manifest_sha256": checksum}
                rationale = f"Recover {run['name']} from {checkpoint['id']} with verified learner/optimizer/buffer/RNG state at step {checkpoint['step']:,}; preserve the original experiment as evidence."
                impact = "Starts a NEW linked run using the immutable original configuration. Simulator episodes reset; post-checkpoint progress is lost. No checkpoint is overwritten and no strategy is changed."
            elif kind in {"designate_checkpoint", "delete_checkpoint"}:
                self._require_idle()
                run, checkpoint, checksum = self._target(payload)
                target = {"run_id": run["id"], "checkpoint_id": checkpoint["id"]}
                best = self.store.setting("designated_best")
                payload = {**target, "manifest_sha256": checksum, "expected_best": best}
                if kind == "designate_checkpoint":
                    rationale = "Manual user designation of a checkpoint. There is no implemented evaluator and no evidence that this checkpoint is competitively strongest."
                    impact = (f"Designates {checkpoint['id']} from {run['name']} at step {checkpoint['step']:,}. "
                              "Replaces the designated-best pointer only. Does not modify weights, delete models, or assert an evaluation win rate. The previous designation is retained in the audit log.")
                else:
                    self._check_delete(target)
                    rationale = "User-requested removal of an unprotected checkpoint to reclaim disk space; no automatic retention policy is enabled."
                    impact = (f"Permanently deletes {checkpoint['id']} from {run['name']} at step {checkpoint['step']:,}: "
                              f"{sum(item['bytes'] for item in checkpoint['files'].values()):,} bytes of checkpoint data. "
                              "This cannot be undone. The designated best and last valid recovery checkpoint are protected.")
            else:
                raise AppError("Unknown action. Supported: start_run, apply_config, recover_run, designate_checkpoint, delete_checkpoint.", 422)
            now = time.time()
            proposal = {
                "id": new_id(), "kind": kind, "payload": payload, "rationale": rationale,
                "impact": impact, "plan_revision": self.store.setting("plan_revision"),
                "status": "pending", "created": now, "expires": now + 600, "result": None,
            }
            proposal["digest"] = digest(proposal)
            self.store.save_proposal(proposal)
            self.store.event("info", "action_proposed", f"Confirmation required: {kind}", data=proposal)
            return proposal

    def _check_delete(self, target):
        if self.store.setting("designated_best") == target:
            raise AppError("The designated best checkpoint is protected. Explicitly designate a replacement first.")
        valid = [item for item in self.checkpoints(target["run_id"]).inventory() if item["integrity"] == "valid"]
        if len(valid) <= 1:
            raise AppError("The last valid recovery checkpoint is protected and cannot be deleted.")

    def confirm(self, proposal_id: str, expected_digest: str, confirmed: bool) -> dict:
        with self.lock:
            proposal = self.store.proposal(proposal_id)
            if confirmed is not True:
                raise AppError("Explicit user confirmation is required.", 422)
            if proposal["status"] != "pending":
                raise AppError("This proposal is already resolved; it cannot be replayed.")
            if proposal["expires"] < time.time():
                self.store.finish_proposal(proposal_id, "expired")
                raise AppError("Proposal expired. Review a fresh proposal before confirming.")
            if not secrets.compare_digest(proposal["digest"], expected_digest):
                raise AppError("Confirmation does not match the reviewed proposal.", 422)
            if proposal["plan_revision"] != self.store.setting("plan_revision"):
                raise AppError("Approved plan changed since this proposal. Review a new proposal.")
            self._require_idle()
            kind, payload = proposal["kind"], proposal["payload"]
            # Revalidate all mutable preconditions immediately before the side effect.
            if kind in {"recover_run", "designate_checkpoint", "delete_checkpoint"}:
                run, _, _ = self._target(payload, prepared=True)
            if kind == "recover_run":
                require_recovery_compatible(run["metadata"])
            if kind in {"designate_checkpoint", "delete_checkpoint"}:
                if payload["expected_best"] != self.store.setting("designated_best"):
                    raise AppError("The model designation changed. Review a new proposal.")
            if kind == "delete_checkpoint":
                self._check_delete({key: payload[key] for key in ("run_id", "checkpoint_id")})
            self.store.finish_proposal(proposal_id, "executing")
            try:
                if kind == "apply_config":
                    config = TrainingConfig.model_validate(payload["config"])
                    self.store.set_settings({
                        "plan": config.model_dump(mode="json"),
                        "plan_revision": proposal["plan_revision"] + 1,
                    })
                    result = {"plan_revision": proposal["plan_revision"] + 1}
                elif kind == "start_run":
                    result = self._launch(TrainingConfig.model_validate(payload["config"]))
                elif kind == "recover_run":
                    result = self._launch(TrainingConfig.model_validate(run["config"]), parent={
                        "run_id": payload["run_id"], "checkpoint_id": payload["checkpoint_id"],
                    })
                elif kind == "designate_checkpoint":
                    result = {key: payload[key] for key in ("run_id", "checkpoint_id")}
                    self.store.set_setting("designated_best", result)
                else:
                    shutil.rmtree(self.checkpoints(payload["run_id"]).path(payload["checkpoint_id"]))
                    result = {"deleted": payload["checkpoint_id"]}
                self.store.finish_proposal(proposal_id, "approved", result)
                self.store.event("info", "action_confirmed", f"User confirmed: {kind}", data={
                    "proposal_id": proposal_id, "rationale": proposal["rationale"],
                    "impact": proposal["impact"], "result": result,
                })
                return result
            except Exception as error:
                self.store.finish_proposal(proposal_id, "failed", {"error": str(error)})
                self.store.event("error", "action_failed", str(error), data={"proposal_id": proposal_id})
                raise

    def reject(self, proposal_id):
        with self.lock:
            if self.store.proposal(proposal_id)["status"] != "pending":
                raise AppError("Proposal is already resolved.")
            self.store.finish_proposal(proposal_id, "rejected")
            self.store.event("info", "action_rejected", "User declined the proposal", data={"proposal_id": proposal_id})
            return {"status": "rejected"}

    def _launch(self, config, parent=None, purpose=None):
        metadata = experiment_metadata()
        if purpose:
            metadata["purpose"] = purpose
        run = self.store.create_run(config, metadata, parent)
        run_dir = self.store.run_dir(run["id"])
        if parent:
            atomic_json(run_dir / "resume.json", parent)
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "rocket_league_ai.training.worker", "--run-dir", str(run_dir)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
        except OSError as error:
            self.store.update_run(run["id"], status=RunStatus.FAILED, error=f"Cannot launch worker: {error}")
            raise AppError("Could not launch the Python worker. Inspect the backend environment and logs.", 503) from error
        worker = LiveWorker(run["id"], process)
        self.active = worker
        worker.reader = threading.Thread(target=self._read_worker, args=(worker,), daemon=True, name="worker-events")
        worker.reader.start()
        self.store.event("info", "run_starting", "Explicitly authorized training worker is starting", run["id"], {"parent": parent})
        return {"run_id": run["id"]}

    # -- Hardware benchmark (genuine bounded runs, never auto-applied) -------
    def workspace_bytes(self) -> int:
        now = time.monotonic()
        if now - self._workspace_bytes_cache[0] < 60:
            return self._workspace_bytes_cache[1]
        total = 0
        for path in self.root.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
        self._workspace_bytes_cache = (now, total)
        return total

    def _benchmark_snapshot(self):
        sessions = self.store.benchmarks(limit=1)
        if not sessions:
            return {"session": None, "results": [], "running": False}
        session = sessions[0]
        return {"session": session, "results": self.store.benchmark_results(session["id"]),
                "running": session["status"] == "running"}

    def start_benchmark(self, spec: BenchmarkSpec) -> dict:
        with self.lock:
            self._require_idle()
            # Input validation must not depend on installed packages: a bad
            # candidate list is a 422 on every host, while a missing training
            # extra is a 409 that only makes sense for an otherwise valid spec.
            plan = TrainingConfig.model_validate(self.store.setting("plan"))
            for candidate in spec.candidates():
                try:
                    TrainingConfig.model_validate(
                        spec.measurement_config(plan, candidate).model_dump(mode="json"))
                except Exception as error:
                    raise AppError(
                        f"Candidate {candidate['device']} / {candidate['n_proc']} env cannot be "
                        f"measured with this plan: {error}", 422) from None
            # Fail before a long session starts, not halfway through a candidate.
            missing = [name for name, version in package_versions().items() if version is None]
            if missing:
                raise AppError("Training extra is missing: " + ", ".join(missing) +
                               ". Run uv sync --frozen --extra training before benchmarking.")
            session = self.store.create_benchmark(spec.model_dump(mode="json"))
            self.auxiliary = "benchmark"
            self.benchmark_stop.clear()
            self.store.event("info", "benchmark_started",
                             f"Benchmark started: {len(spec.candidates())} candidate(s), {spec.repeats} repeat(s) each, "
                             f"warm-up {spec.warmup_iterations} + up to {spec.effective_measure_iterations} measured iterations "
                             f"of {spec.batch_size} agent steps. Each measurement is a real bounded training run.")
            thread = threading.Thread(target=self._run_benchmark, args=(session["id"], spec),
                                      daemon=True, name="benchmark")
            thread.start()
            return session

    def stop_benchmark(self) -> dict:
        with self.lock:
            if self.auxiliary != "benchmark":
                raise AppError("No benchmark is running.")
            self.benchmark_stop.set()
            if self.active:
                try:
                    self.control(self.active.run_id, "stop")
                except AppError:
                    pass
            self.store.event("warning", "benchmark_stop_requested", "Benchmark stop requested by user.")
            return {"status": "stopping"}

    def host_fingerprint(self) -> dict:
        """Cheap host identity for a benchmark session; no Torch import."""
        sample = self.system.snapshot()
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_model": platform.processor() or platform.machine(),
            "cpu_count": sample.get("cpu_count"),
            "gpu_name": sample.get("gpu_name"),
            "gpu_memory_total_mib": sample.get("gpu_memory_total_mib"),
        }

    def _adopt_benchmark_defaults(self, session_id: str, recommendation: dict) -> dict:
        """Adopt only execution fields (device/n_proc) after strict safety gates.

        The learning objective is untouched: batch size, network, reward,
        seed and every strategy field remain exactly the approved plan's. The
        previous plan is stored so one click can revert the change.
        """
        candidate = recommendation.get("candidate")
        if not candidate:
            return {"status": "not-applicable", "reason": "no recommendation"}
        plan = TrainingConfig.model_validate(self.store.setting("plan"))
        update = {"device": candidate["device"], "n_proc": candidate["n_proc"]}
        changed = {key: {"before": getattr(plan, key), "after": value}
                   for key, value in update.items() if getattr(plan, key) != value}
        if not changed:
            return {"status": "unchanged", "changes": {}}
        payload = {**plan.model_dump(mode="json"), **update}
        try:
            updated = TrainingConfig.model_validate(payload)
        except Exception as error:
            return {"status": "rejected", "error": str(error), "changes": changed}
        revision = self.store.setting("plan_revision")
        record = {
            "benchmark_id": session_id, "changes": changed,
            "previous_plan": plan.model_dump(mode="json"),
            "host": self.host_fingerprint(), "created": time.time(),
        }
        self.store.set_settings({
            "plan": updated.model_dump(mode="json"),
            "plan_revision": revision + 1,
            "benchmark_adoption": record,
        })
        self.store.event(
            "info", "plan_adopted_from_benchmark",
            f"Benchmark {session_id[:12]} adopted {candidate['device']} with {candidate['n_proc']} "
            "environment process(es) as the default for future runs. Strategy fields are unchanged.",
            data=record,
        )
        return {"status": "adopted", "changes": changed, "plan_revision": revision + 1}

    def _run_benchmark(self, session_id: str, spec: BenchmarkSpec):
        benchmark_run_ids = []

        def peak(resources, name):
            values = [item[name] for item in resources if item.get(name) is not None]
            return max(values) if values else None

        try:
            plan = TrainingConfig.model_validate(self.store.setting("plan"))
            for candidate in spec.candidates():
                if self.benchmark_stop.is_set():
                    break
                repeats_iterations, resources, run_ids, error = [], [], [], None
                for _ in range(spec.repeats):
                    if self.benchmark_stop.is_set():
                        break
                    config = spec.measurement_config(plan, candidate)
                    iterations, sampled, run_id, run_error = self._measure_once(config, spec)
                    run_ids.append(run_id)
                    benchmark_run_ids.append(run_id)
                    if sampled:
                        resources.append(sampled)
                    if run_error:
                        error = run_error
                        break
                    repeats_iterations.append(iterations)
                if error or len(repeats_iterations) != spec.repeats:
                    self.store.save_benchmark_result(session_id, {
                        "candidate": candidate, "status": "failed",
                        "error": error or "Benchmark stopped before all repeats completed.",
                        "run_id": run_ids[-1] if run_ids else None,
                    })
                    continue
                # Aggregate per repeat first (warm-up excluded and episode
                # counters counted within each run), then combine repeats.
                per_repeat = [aggregate_iterations(iterations, spec.warmup_iterations, spec.batch_size)
                              for iterations in repeats_iterations]
                aggregate = combine_repeats(per_repeat)
                startup_samples = [item["startup_seconds"] for item in resources
                                   if item.get("startup_seconds") is not None]
                wall_samples = [item["measurement_seconds"] for item in resources
                                if item.get("measurement_seconds")]
                merged = {
                    "startup_seconds": sorted(startup_samples)[len(startup_samples) // 2] if startup_samples else None,
                    "measurement_seconds": sorted(wall_samples)[len(wall_samples) // 2] if wall_samples else None,
                    "stop_reason": next((item.get("stop_reason") for item in resources
                                         if item.get("stop_reason")), None),
                    "cpu_percent_peak": peak(resources, "cpu_percent_peak"),
                    "memory_percent_peak": peak(resources, "memory_percent_peak"),
                    "gpu_percent_peak": peak(resources, "gpu_percent_peak"),
                    "gpu_memory_used_mib_peak": peak(resources, "gpu_memory_used_mib_peak"),
                    "gpu_memory_percent_peak": peak(resources, "gpu_memory_percent_peak"),
                    "gpu_peak_bytes": peak(resources, "gpu_peak_bytes"),
                }
                self.store.save_benchmark_result(session_id, {
                    "candidate": candidate, "status": "completed", "aggregate": aggregate,
                    "resources": merged, "run_id": run_ids[-1],
                })
            results = self.store.benchmark_results(session_id)
            recommendation = recommend(results, spec)
            recommendation["host"] = self.host_fingerprint()
            recommendation["measurement_seconds"] = spec.measurement_seconds
            if spec.auto_default and recommendation.get("auto_default"):
                recommendation["adoption"] = self._adopt_benchmark_defaults(session_id, recommendation)
            stopped = self.benchmark_stop.is_set()
            self.store.update_benchmark(session_id, status="stopped" if stopped else "completed",
                                        recommendation=recommendation)
            self.store.event("info", "benchmark_finished",
                             "Benchmark stopped by user." if stopped else
                             f"Benchmark finished. {recommendation['explanation']}")
        except Exception as error:  # keep the session honest instead of hanging in "running"
            self.store.update_benchmark(session_id, status="failed", error=str(error))
            self.store.event("error", "benchmark_failed", str(error))
        finally:
            with self.lock:
                if self.auxiliary == "benchmark":
                    self.auxiliary = None
            # Measurement runs are disposable. Keep their lightweight SQLite
            # record (purpose=benchmark, filtered from history) but remove the
            # throwaway configs/checkpoints so frequent 20 s sweeps do not grow
            # the workspace and checkpoint inventory without bound.
            for run_id in set(benchmark_run_ids):
                try:
                    shutil.rmtree(self.store.run_dir(run_id), ignore_errors=True)
                except (AppError, OSError):
                    # Cleanup is best-effort; never turn a completed measurement
                    # into a failed benchmark because a disposable directory moved.
                    pass
                self.checkpoint_stores.pop(run_id, None)

    def _measure_once(self, config: TrainingConfig, spec: BenchmarkSpec):
        """Run one genuine bounded training run and collect its real telemetry.

        Returns (iterations, resources, run_id, error). Iterations are the
        worker's own per-update measurements in order; nothing is synthesized.

        The run stops on the first of: the wall-clock measurement window
        (after warm-up, never before ``min_iterations`` measured updates),
        the configured iteration ceiling, or the user. A benchmark therefore
        lasts about ``measurement_seconds`` per candidate on this host instead
        of depending on an arbitrary step count.
        """
        with self.lock:
            launched = self._launch(config, purpose="benchmark")
        run_id = launched["run_id"]
        peaks = {"cpu_percent_peak": 0.0, "memory_percent_peak": 0.0,
                 "gpu_percent_peak": None, "gpu_memory_used_mib_peak": None,
                 "gpu_memory_percent_peak": None, "gpu_peak_bytes": None,
                 "startup_seconds": None, "measurement_seconds": 0.0,
                 "samples": 0, "stop_reason": None}
        launched_at = time.monotonic()
        hard_deadline = launched_at + spec.timeout_seconds
        started_at = None
        stop_requested = False
        stop_reason = None
        run = self.store.run(run_id)
        while time.monotonic() < hard_deadline:
            if self.benchmark_stop.is_set():
                stop_reason = "user"
                break
            run = self.store.run(run_id)
            if run["status"] not in ACTIVE:
                break
            metrics = run.get("metrics") or {}
            steps = metrics.get("steps") or 0
            if steps and started_at is None:
                started_at = time.monotonic()
                peaks["startup_seconds"] = started_at - launched_at
            sample = self.system.snapshot()
            peaks["samples"] += 1
            peaks["cpu_percent_peak"] = max(peaks["cpu_percent_peak"], sample.get("cpu_percent") or 0)
            peaks["memory_percent_peak"] = max(peaks["memory_percent_peak"], sample.get("memory_percent") or 0)
            if sample.get("gpu_percent") is not None:
                peaks["gpu_percent_peak"] = max(peaks["gpu_percent_peak"] or 0, sample["gpu_percent"])
            if sample.get("gpu_memory_used_mib") is not None:
                peaks["gpu_memory_used_mib_peak"] = max(peaks["gpu_memory_used_mib_peak"] or 0, sample["gpu_memory_used_mib"])
            if sample.get("gpu_memory_percent") is not None:
                peaks["gpu_memory_percent_peak"] = max(peaks["gpu_memory_percent_peak"] or 0, sample["gpu_memory_percent"])
            if started_at is not None and not stop_requested:
                completed = steps // max(config.timesteps_per_iteration, 1)
                measured = max(completed - spec.warmup_iterations, 0)
                elapsed = time.monotonic() - started_at
                time_reached = elapsed >= spec.measurement_seconds and measured >= spec.min_iterations
                ceiling_reached = measured >= spec.effective_measure_iterations
                if time_reached or ceiling_reached:
                    stop_reason = "time" if time_reached else "iteration ceiling"
                    try:
                        self.control(run_id, "stop")
                    except AppError:
                        pass
                    stop_requested = True
            # A 0.25 s poll let a fast CPU finish several extra updates before
            # the stop request; 0.1 s bounds the overshoot to well under one
            # second without measurably loading the host.
            time.sleep(0.1)
        else:
            stop_reason = "timeout"

        # Wait for an update-boundary stop; never overlap the next candidate.
        stop_wait_deadline = time.monotonic() + 30
        run = self.store.run(run_id)
        while run["status"] in ACTIVE and time.monotonic() < stop_wait_deadline:
            if not stop_requested:
                try:
                    self.control(run_id, "stop")
                except AppError:
                    pass
                stop_requested = True
            time.sleep(0.1)
            run = self.store.run(run_id)
        if run["status"] in ACTIVE:
            peaks["stop_reason"] = "stop acknowledgement timeout"
            # A benchmark run has a much larger budget than the 20 s window, so
            # never leave a stuck worker alive into the next candidate.
            with self.lock:
                worker = self.active
                if worker and worker.run_id == run_id:
                    self._kill_group(worker)
            deadline = time.monotonic() + 10
            while self.active is not None and time.monotonic() < deadline:
                time.sleep(0.05)
            return [], peaks, run_id, "measurement run did not stop within 30 s of the request"

        elapsed = (time.monotonic() - started_at) if started_at else 0.0
        peaks["stop_reason"] = stop_reason or ("completed" if run["status"] == RunStatus.COMPLETED else run["status"])
        iterations, cursor = [], 0
        while True:
            batch = self.store.events(cursor, 1000)
            for event in batch["events"]:
                if event["run_id"] == run_id and event["kind"] == "training_metrics":
                    metrics = event["data"]
                    if metrics.get("steps") and (not iterations or iterations[-1]["steps"] != metrics["steps"]):
                        iterations.append(metrics)
            if batch["cursor"] == cursor:
                break
            cursor = batch["cursor"]
        # The window starts when the first update lands, so subtract the
        # warm-up updates' own reported duration to report the measured window
        # rather than "time since the first update".
        warmup_seconds = sum((item.get("iteration_seconds") or 0)
                             for item in iterations[:spec.warmup_iterations])
        peaks["measurement_seconds"] = max(elapsed - warmup_seconds, 0.0)
        final = self.store.run(run_id)["metrics"]
        if final.get("gpu_peak_bytes") is not None:
            peaks["gpu_peak_bytes"] = final["gpu_peak_bytes"]
        if stop_reason == "user":
            return iterations, peaks, run_id, "Benchmark stopped by user before all repeats completed."
        if stop_reason == "timeout":
            return iterations, peaks, run_id, f"measurement exceeded {spec.timeout_seconds}s"
        if run["status"] == RunStatus.FAILED:
            return iterations, peaks, run_id, run.get("error") or "measurement run failed"
        if started_at is None:
            return iterations, peaks, run_id, "worker produced no completed PPO update before the timeout"
        if len(iterations) <= spec.warmup_iterations:
            return iterations, peaks, run_id, "worker reported too few completed iterations to measure"
        return iterations, peaks, run_id, None

    # -- Checkpoint evaluation (separate frozen-inference subprocess) --------
    def start_evaluation(self, spec: EvaluationSpec) -> dict:
        with self.lock:
            self._require_idle()
            missing = [name for name, version in package_versions().items() if version is None]
            if missing:
                raise AppError("Training extra is missing: " + ", ".join(missing) +
                               ". Run uv sync --frozen --extra training before evaluating.")
            run, checkpoint, _ = self._target({"run_id": spec.run_id, "checkpoint_id": spec.checkpoint_id})
            if spec.opponent_checkpoint_id:
                self._target({"run_id": spec.opponent_run_id, "checkpoint_id": spec.opponent_checkpoint_id})
            evaluation = self.store.create_evaluation(spec.model_dump(mode="json"))
            self.evaluation_stop.clear()
            self.auxiliary = "evaluation"
            self.store.event("info", "evaluation_started",
                             f"Evaluating {checkpoint['id']} against {spec.opponent_id} over {spec.games} "
                             f"side-balanced games (seed {spec.seed}). Opponent and candidate weights are frozen.",
                             run["id"])
            thread = threading.Thread(target=self._run_evaluation,
                                      args=(evaluation["id"], spec), daemon=True, name="evaluation")
            thread.start()
            return evaluation

    def _execute_evaluation(self, evaluation_id: str, spec: EvaluationSpec):
        """Run one evaluation to a terminal record. Raises on failure; marks
                the record stopped (and returns "stopped") when stop was requested."""
        directory = self.root / "evaluations" / evaluation_id
        directory.mkdir(parents=True, mode=0o700)
        request = {"spec": spec.model_dump(mode="json"), "root": str(self.root)}
        atomic_json(directory / "request.json", request)
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "rocket_league_ai.training.evaluate",
             "--request", str(directory / "request.json"), "--output", str(directory)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        self.evaluation_process = process
        deadline = time.monotonic() + spec.total_games * spec.max_game_seconds / max(spec.parallel_workers, 1) + 900
        while process.poll() is None and time.monotonic() < deadline and not self.evaluation_stop.is_set():
            progress_file = directory / "progress.json"
            if progress_file.is_file():
                try:
                    progress = json.loads(progress_file.read_text())
                    self.store.update_evaluation(evaluation_id, progress=progress["completed"] / max(progress["games"], 1))
                except (ValueError, KeyError, OSError):
                    pass
            time.sleep(0.5)
        if self.evaluation_stop.is_set():
            self._kill_evaluation_group(process)
            self.store.update_evaluation(evaluation_id, status="stopped",
                                         error="Evaluation stopped by user. No partial result is reported.")
            self.store.event("warning", "evaluation_stopped", "Evaluation stopped by user; no partial result reported.", spec.run_id)
            return
        if process.poll() is None:
            self._kill_evaluation_group(process)
            raise AppError("Evaluation exceeded its time budget and was stopped. No partial result is reported.")
        output = process.stdout.read() if process.stdout else ""
        if process.returncode != 0:
            raise AppError(f"Evaluation process failed: {output.strip().splitlines()[-1] if output.strip() else 'no output'}")
        payload = json.loads((directory / "result.json").read_text())
        games = [GameRecord.model_validate(game) for game in payload["games"]]
        run = self.store.run(spec.run_id)
        result = summarize_evaluation(
            spec, games, candidate_id=payload["candidate_id"],
            candidate_config_hash=run["config_hash"],
            environment_version=spec.scenario_environment or run["config"]["environment_version"],
        )
        # Cross-check counting with the shared frozen contract model.
        EvaluationResult(
            candidate_id=result["candidate_id"], opponent_id=result["opponent_id"],
            wins=result["wins"], losses=result["losses"], draws=result["draws"],
            games=result["games"], seed=result["seed"], side_balanced=True,
            environment_version=result["environment_version"],
            goals_for=result["goals_for"], goals_against=result["goals_against"],
        )
        trajectory = f"evaluations/{evaluation_id}/trajectory.json" if (directory / "trajectory.json").is_file() else None
        self.store.update_evaluation(evaluation_id, status="completed", result=result, trajectory=trajectory, progress=1.0)
        self.store.event("info", "evaluation_finished",
                         f"Evaluation completed: {result['wins']}W/{result['losses']}L/{result['draws']}D, "
                         f"goals {result['goals_for']}–{result['goals_against']}, win rate "
                         f"{result['win_rate']:.0%} (95% CI {result['win_rate_ci95'][0]:.0%}–{result['win_rate_ci95'][1]:.0%}).",
                         spec.run_id)
        return "completed"

    def _run_evaluation(self, evaluation_id: str, spec: EvaluationSpec):
        try:
            self._execute_evaluation(evaluation_id, spec)
        except Exception as error:
            self.store.update_evaluation(evaluation_id, status="failed", error=str(error))
            self.store.event("error", "evaluation_failed", str(error), spec.run_id)
        finally:
            self.evaluation_process = None
            with self.lock:
                if self.auxiliary == "evaluation":
                    self.auxiliary = None

    # -- Checkpoint league (sequential round-robin of frozen matches) ---------
    def start_league(self, spec: LeagueSpec) -> dict:
        with self.lock:
            self._require_idle()
            missing = [name for name, version in package_versions().items() if version is None]
            if missing:
                raise AppError("Training extra is missing: " + ", ".join(missing) +
                               ". Run uv sync --frozen --extra training before running a league.")
            for ref in spec.checkpoints:
                self._target({"run_id": ref["run_id"], "checkpoint_id": ref["checkpoint_id"]})
            league = self.store.create_league(spec.model_dump(mode="json"))
            self.auxiliary = "league"
            self.league_stop.clear()
            self.evaluation_stop.clear()
            self.store.event("info", "league_started",
                             f"League started: {len(spec.checkpoints)} checkpoints, "
                             f"{len(spec.pairs())} matches of {spec.games_per_pair} games. All weights frozen; "
                             "results feed Elo ratings. Starting the league was an explicit user action.")
            thread = threading.Thread(target=self._run_league, args=(league["id"], spec),
                                      daemon=True, name="league")
            thread.start()
            return league

    def _run_league(self, league_id: str, spec):
        evaluation_ids, failures = [], 0
        try:
            for pair_index, (opponent, candidate) in enumerate(spec.pairs()):
                if self.league_stop.is_set():
                    break
                ev_spec = EvaluationSpec(
                    run_id=candidate["run_id"], checkpoint_id=candidate["checkpoint_id"],
                    opponent_run_id=opponent["run_id"], opponent_checkpoint_id=opponent["checkpoint_id"],
                    games=spec.games_per_pair, seed=spec.seed + pair_index * 1000,
                    max_game_seconds=spec.max_game_seconds, parallel_workers=spec.parallel_workers,
                    record_games=1 if pair_index == 0 else 0,
                )
                evaluation = self.store.create_evaluation(ev_spec.model_dump(mode="json"))
                evaluation_ids.append(evaluation["id"])
                self.store.update_league(league_id, evaluation_ids=evaluation_ids)
                try:
                    outcome = self._execute_evaluation(evaluation["id"], ev_spec)
                except Exception as error:
                    self.store.update_evaluation(evaluation["id"], status="failed", error=str(error))
                    self.store.event("error", "evaluation_failed", str(error), ev_spec.run_id)
                    failures += 1
                    continue
                if outcome == "stopped":
                    break
            if self.league_stop.is_set():
                status = "stopped"
            elif failures and failures == len(spec.pairs()):
                status = "failed"
            elif failures:
                status = "completed_with_failures"
            else:
                status = "completed"
            self.store.update_league(league_id, status=status, evaluation_ids=evaluation_ids)
            self.store.event("info", "league_finished",
                             f"League {status}: {len(evaluation_ids)} match(es) recorded. Ratings in Models are Elo over checkpoint matches only.")
        except Exception as error:
            self.store.update_league(league_id, status="failed", error=str(error))
            self.store.event("error", "league_failed", str(error))
        finally:
            self.evaluation_process = None
            with self.lock:
                if self.auxiliary == "league":
                    self.auxiliary = None

    def stop_league(self, league_id: str) -> dict:
        with self.lock:
            league = self.store.league(league_id)
            if self.auxiliary != "league" or league["status"] != "running":
                raise AppError("This league is not running.")
            self.league_stop.set()
            self.evaluation_stop.set()
            return {"status": "stopping"}

    def _kill_evaluation_group(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def stop_evaluation(self, evaluation_id: str) -> dict:
        with self.lock:
            evaluation = self.store.evaluation(evaluation_id)
            if self.auxiliary != "evaluation" or evaluation["status"] != "running":
                raise AppError("This evaluation is not running.")
            self.evaluation_stop.set()
            return {"status": "stopping"}

    def trajectory(self, evaluation_id: str) -> dict:
        evaluation = self.store.evaluation(evaluation_id)
        if evaluation["status"] != "completed" or not evaluation.get("trajectory"):
            raise AppError("No recorded replay is available for this evaluation.", 404)
        path = self.root / evaluation["trajectory"]
        if not path.is_file():
            raise AppError("The recorded replay file is missing from the workspace.", 404)
        return json.loads(path.read_text())

    def control(self, run_id: str, command: str):
        with self.lock:
            checked_id(run_id)
            if not self.active or self.active.run_id != run_id:
                raise AppError("This run has no connected worker. Use explicit recovery for an interrupted run.")
            run = self.store.run(run_id)
            target = control_target(run["status"], command)
            self._send(self.active, command)
            self.store.update_run(run_id, status=target)
            self.store.event("info", "control_requested", f"{command.capitalize()} requested; awaiting update-boundary acknowledgement", run_id)
            return {"status": target}

    def _send(self, worker, command):
        try:
            worker.process.stdin.write(json.dumps({"command": command}) + "\n")
            worker.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            raise AppError("Worker control channel closed. Wait for failure status and review recovery.") from None

    def _read_worker(self, worker):
        try:
            while line := worker.process.stdout.readline(65536):
                if line.startswith(EVENT_PREFIX):
                    try:
                        data = json.loads(line[len(EVENT_PREFIX):])
                        self._worker_event(worker, data)
                    except (ValueError, KeyError, TypeError) as error:
                        self.store.event("error", "protocol_error", f"Invalid worker event: {error}", worker.run_id)
                elif line.strip():
                    level = "warning" if "warning" in line.lower() else "info"
                    self.store.event(level, "worker_output", line.rstrip(), worker.run_id)
        finally:
            return_code = worker.process.wait()
            with self.lock:
                status = self.store.run(worker.run_id)["status"]
                if status in ACTIVE:
                    self.store.update_run(worker.run_id, status=RunStatus.FAILED, error=(
                        f"Worker exited with code {return_code} without a terminal acknowledgement. "
                        "Last committed checkpoint is preserved. Inspect logs, then explicitly recover."
                    ))
                self.store.event("info" if return_code == 0 else "error", "worker_exited",
                                 f"Worker exited with code {return_code}; no automatic restart", worker.run_id)
                self._kill_group(worker)
                worker.process.stdin.close()
                worker.process.stdout.close()
                if self.active is worker:
                    self.active = None

    def _worker_event(self, worker, data):
        with self.lock:
            if data.get("version") != 1:
                raise ValueError("Unsupported worker protocol version")
            worker.heartbeat = time.monotonic()
            kind = data["kind"]
            if kind == "heartbeat":
                worker.progress_age = data["progress_age_seconds"]
                return
            run = self.store.run(worker.run_id)
            if kind in {"ready", "metrics"}:
                fields = {"metrics": data["metrics"]}
                if kind == "ready" and run["status"] == RunStatus.STARTING:
                    fields["status"] = RunStatus.RUNNING
                self.store.update_run(worker.run_id, **fields)
                worker.progress_age = 0
                if kind == "ready" or data.get("updated"):
                    self.store.event("info", "training_metrics", "PPO update completed" if kind == "metrics" else "Real training stack ready",
                                     worker.run_id, data["metrics"])
            elif kind == "failure":
                worker.failed = True
                self.store.update_run(worker.run_id, status=RunStatus.FAILED, error=data["message"])
                self.store.event("error", "training_failed", data["message"], worker.run_id)
            elif kind == "state":
                status = RunStatus(data["status"])
                if worker.failed or (run["status"] == RunStatus.STOPPING and status in {RunStatus.RUNNING, RunStatus.PAUSED}):
                    return
                fields = {"status": status}
                if "metrics" in data:
                    fields["metrics"] = data["metrics"]
                self.store.update_run(worker.run_id, **fields)
                self.store.event("info", "training_state", data["message"], worker.run_id, {"status": status})
            elif kind == "checkpoint":
                self.store.event("info", "checkpoint_committed", f"Verified checkpoint committed at step {data['step']:,}", worker.run_id, data)
            elif kind == "log":
                self.store.event(data.get("level", "info"), "training_log", data["message"], worker.run_id)
            else:
                raise ValueError(f"Unknown worker event: {kind}")

    @staticmethod
    def _kill_group(worker):
        try:
            os.killpg(worker.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _watch(self):
        while not self.stop_event.wait(0.5):
            with self.lock:
                worker = self.active
                if not worker:
                    continue
                if worker.process.poll() is not None:
                    # Ensure orphan descendants cannot keep inherited stdout/leases open.
                    if not worker.exit_cleanup_sent:
                        worker.exit_cleanup_sent = True
                        self._kill_group(worker)
                    continue
                status = self.store.run(worker.run_id)["status"]
                heartbeat_stale = time.monotonic() - worker.heartbeat > self.stall_timeout
                progress_stale = worker.progress_age > self.stall_timeout and status != RunStatus.PAUSED
                if heartbeat_stale or progress_stale:
                    message = "Worker health deadline exceeded. Process group stopped; last committed checkpoint preserved. Inspect logs before confirmed recovery."
                    worker.failed = True
                    self.store.update_run(worker.run_id, status=RunStatus.FAILED, error=message)
                    self.store.event("error", "watchdog_timeout", message, worker.run_id)
                    self._kill_group(worker)

    def close(self):
        with self.lock:
            self.closing = True
            self.benchmark_stop.set()
            if self.evaluation_process and self.evaluation_process.poll() is None:
                self.evaluation_process.terminate()
            worker = self.active
            if worker and worker.process.poll() is None:
                try:
                    self._send(worker, "stop")
                    if self.store.run(worker.run_id)["status"] in ACTIVE:
                        self.store.update_run(worker.run_id, status=RunStatus.STOPPING)
                except AppError:
                    pass
        if worker:
            try:
                worker.process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self.store.event("warning", "shutdown_timeout", "Graceful deadline reached; preserving previous checkpoint and terminating worker group", worker.run_id)
                self._kill_group(worker)
                worker.process.wait(timeout=5)
            worker.reader.join(timeout=5)
        self.stop_event.set()
        self.watchdog.join(timeout=3)
        self.system.close()
        self.store.event("info", "backend_stopped", "Backend shut down. Checkpoints and experiment state retained.")
        self.store.close()
        self.lease.close()
