"""FastAPI application: JSON API, SSE stream and static GUI."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import re
import time
from pathlib import Path

import psutil
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import checkpoints
from ..config import AppSettings, TrainingConfig, data_dir, ensure_dirs, project_root
from ..hardware import benchmark, diagnostics, system_info
from ..jobs import JobManager
from ..metrics_util import enrich_models
from ..modes import all_modes, get_mode
from ..store import Store
from ..version import __version__

MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class StartRunRequest(TrainingConfig):
    """Training configuration plus a resume decision.

    Omitting ``resume_mode`` means automatic continuation: the newest valid,
    compatible checkpoint is restored when one exists, and a new policy is
    started only when the models directory contains no checkpoint at all.
    Existing but unusable checkpoints are refused, never silently replaced by
    a fresh model.  Explicit values keep their old meaning: ``fresh`` always
    starts a new policy, ``continue`` requires a usable checkpoint and
    ``transfer`` keeps only the weights and restarts the step count.

    ``confirm`` must be set by the client.  It is the server-side half of the
    confirmation dialog: a training run is never started by a request that did
    not explicitly ask for it.
    """

    resume_from: str | None = None
    resume_mode: str | None = None
    #: inherited config fields to change deliberately on a continue resume
    #: (a recorded phase change; see rlai.resume.merge_config)
    config_overrides: list[str] = []
    allow_concurrent: bool = False
    #: the user confirmed the start (the GUI asks before it posts)
    confirm: bool = False


class ResumePlanRequest(BaseModel):
    """Ask what a resume *would* restore.  Never starts training."""

    config: dict
    mode: str = "continue"
    run_id: str | None = None
    checkpoint: str | None = None
    config_overrides: list[str] = []


class StartPlanRequest(BaseModel):
    """The simplified training request: the only two user-controlled values.

    Everything else (mode, PPO hyper-parameters, network, worker count,
    device) keeps its backend default, so the user never has to understand
    them.  The mode is fixed to 1v1 for this workflow; the server enforces
    it here so the GUI cannot drift.
    """

    total_steps: int = Field(default=200_000, ge=2_000, le=5_000_000_000)
    run_name: str | None = Field(default=None, min_length=1, max_length=80)


class ValidateModelRequest(BaseModel):
    """Validate a checkpoint against a config before spending time on it."""

    config: dict | None = None
    require_optimizer: bool = False


class CompareRunsRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list, min_length=1, max_length=12)


class BenchmarkRequest(BaseModel):
    mode: str = "soccer-1v1"
    seconds: float = Field(default=3.0, ge=0.5, le=10.0)
    workers: int = Field(default=1, ge=1, le=8)
    parallel: bool = True
    include_inference: bool = True


# hard ceiling on a single evaluation job so one request cannot occupy the box
MAX_EVAL_SIM_SECONDS = 4 * 3600.0


class EvaluateRequest(BaseModel):
    model_id: str
    mode: str | None = None
    opponent: str = "chase"
    opponents: list[str] = Field(default_factory=list, max_length=8)
    games: int = Field(default=20, ge=1, le=500)
    seed: int = 1000
    match_seconds: float = Field(default=120.0, ge=5, le=600)
    device: str = "cpu"
    sides: list[str] = Field(default_factory=lambda: ["blue"], max_length=2)
    mirror: bool = True
    stochastic: bool = False
    phase: str = "manual"

    def effective_opponents(self) -> list[str]:
        names = [name for name in self.opponents if name]
        return names or [self.opponent]

    def budget_check(self) -> None:
        sides = max(1, len({side for side in self.sides if side}))
        games = self.games * sides if sides > 1 else self.games
        projected = games * len(self.effective_opponents()) * self.match_seconds
        if projected > MAX_EVAL_SIM_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"evaluation would run ~{projected / 60:.0f} min of simulated matches "
                    f"({games} games x {len(self.effective_opponents())} opponents x "
                    f"{self.match_seconds:.0f} s); reduce games or match_seconds "
                    f"(cap {MAX_EVAL_SIM_SECONDS / 3600:.0f} h)"
                ),
            )
        for side in self.sides:
            if side not in ("blue", "orange"):
                raise HTTPException(status_code=400, detail=f"unknown side {side!r}")
        if self.phase not in ("before", "periodic", "after", "manual", "suite"):
            raise HTTPException(status_code=400, detail=f"unknown phase {self.phase!r}")


class LiveLaunchRequest(BaseModel):
    model_id: str | None = None
    mode: str = "soccer-1v1"
    human_team: str = "blue"
    fill: str = "psyonix"
    human: bool = True
    timeout: float = 0.0
    dry_run: bool = False


class TeamRequest(BaseModel):
    """A team of bots for the team builder."""

    name: str = Field(min_length=1, max_length=80)
    mode: str = "soccer-1v1"
    description: str = ""
    bots: list[dict] = Field(default_factory=list, min_length=1, max_length=3)
    parent: str | None = None


class BotSettingsRequest(BaseModel):
    """Per-bot behaviour settings."""

    difficulty: str = "medium"
    behaviour: str = "balanced"
    overrides: dict[str, float] = Field(default_factory=dict)


class RivalRequest(BaseModel):
    """A named rival."""

    name: str = Field(min_length=1, max_length=80)
    kind: str = "checkpoint"
    reference: str = ""
    description: str = ""


class PlayerMemoryRequest(BaseModel):
    """A note for a remembered player."""

    name: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=500)


class EvalLabRequest(BaseModel):
    """A repeatable evaluation experiment."""

    model_ids: list[str] = Field(default_factory=list, max_length=8)
    opponents: list[str] = Field(default_factory=list, max_length=8)
    games: int = Field(default=6, ge=1, le=200)
    match_seconds: float = Field(default=30.0, ge=5, le=600)
    seed: int = 1000
    sides: list[str] = Field(default_factory=lambda: ["blue"], max_length=2)
    mirror: bool = True
    mode: str = "soccer-1v1"
    #: run the experiment immediately; the GUI asks for a confirmation first
    execute: bool = False


def _settings() -> "AppSettings":
    """Validated exposure settings shared with ``rlai serve``."""
    from ..config import AppSettings

    return AppSettings.from_env()


def _process_usage(pid: int | None = None) -> dict:
    """CPU and memory of one process (default: this API process).

    ``psutil.cpu_percent(interval=None)`` compares against the *previous*
    call, so the very first call always returns 0.0.  The value is
    therefore marked as unavailable until a second measurement exists.
    """
    try:
        import psutil

        process = psutil.Process(pid) if pid else psutil.Process()
        with process.oneshot():
            info = process.cpu_percent(interval=None)
            rss = process.memory_info().rss
            threads = process.num_threads()
        return {
            "pid": process.pid,
            "cpu_percent": info,
            "rss_mb": round(rss / 1024**2, 1),
            "threads": threads,
            "note": "cpu_percent is measured against the previous call",
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": str(exc)}


def _gpu_process_usage() -> list[dict] | None:
    """Per-process GPU usage via nvidia-smi (``None`` when unavailable)."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        rows = []
        for line in result.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                rows.append({"pid": int(parts[0]), "used_memory_mb": float(parts[1])})
            except ValueError:
                continue
        return rows
    except Exception:
        return None


def create_app(store_path: Path | None = None) -> FastAPI:
    paths = ensure_dirs()
    store = Store(store_path or paths["root"] / "rlai.sqlite3")
    jobs = JobManager(store, paths)
    settings = _settings()
    token = settings.auth_token

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.weak_token:
            store.log(
                None,
                "WARNING",
                "RLAI_AUTH_TOKEN is shorter than 12 characters; prefer a long random token",
            )
        # refresh model registry from disk on startup
        try:
            store.sync_models(checkpoints.list_models(paths["models"]))
        except Exception as exc:  # pragma: no cover - defensive
            store.log(None, "WARNING", f"model registry sync failed: {exc}")
        yield
        jobs.shutdown()

    app = FastAPI(title="Rocket League AI", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.jobs = jobs
    app.state.paths = paths
    # Opt-in cross-origin access: the GUI is served by this app, so the default
    # is no CORS middleware at all (same-origin only).
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def require_auth(request: Request) -> None:
        if not token:
            return
        provided = request.headers.get("x-rlai-token") or ""
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            provided = auth[7:]
        # EventSource cannot set headers, so the SSE route may pass ?token=
        if not provided and request.url.path == "/api/events":
            provided = request.query_params.get("token", "")
        if not provided or not hmac.compare_digest(str(provided), token):
            raise HTTPException(status_code=401, detail="invalid capability token")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "time": time.time()}

    @app.get("/api/system")
    def system():
        return {
            "info": system_info(),
            "active_run": store.latest_active_run(),
            "data_dir": str(data_dir()),
        }

    @app.get("/api/system/resources")
    def resources():
        """Live resource usage.

        The numbers describe the *machine*; the training process itself is
        reported separately, because the trainer runs in its own process
        and the API process never holds the model.  Reporting the API
        process' GPU memory as "the training's GPU memory" was misleading,
        so the trainer's own usage is measured through its process id.
        """
        memory = psutil.virtual_memory()
        payload = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": os.cpu_count(),
            "memory_percent": memory.percent,
            "memory_used_gb": round((memory.total - memory.available) / 1024**3, 2),
            "memory_total_gb": round(memory.total / 1024**3, 2),
            "active_run": store.latest_active_run(),
        }
        payload["process"] = _process_usage()
        try:
            import torch

            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(idx)
                payload["gpu"] = {
                    "name": properties.name,
                    "memory_total_mb": round(properties.total_memory / 1024**2, 1),
                    "source": "the GPU of this machine",
                }
                # nvidia-smi knows the real per-process usage; torch cannot
                # see the trainer's allocation because it lives in another
                # process.
                usage = _gpu_process_usage()
                if usage is not None:
                    payload["gpu"]["processes"] = usage
        except Exception:
            pass
        return payload

    @app.get("/api/diagnostics")
    def get_diagnostics():
        return diagnostics()

    @app.post("/api/benchmark", dependencies=[Depends(require_auth)])
    def run_benchmark(request: BenchmarkRequest):
        get_mode(request.mode)
        return benchmark(
            request.mode,
            request.seconds,
            request.workers,
            parallel=request.parallel,
            include_inference=request.include_inference,
        )

    @app.get("/api/modes")
    def modes():
        return [mode.to_dict() for mode in all_modes()]

    @app.get("/api/config/defaults")
    def config_defaults():
        """Backend defaults plus the automatic 1v1 configuration.

        ``default_run`` is the single configuration the simplified training
        form uses: 1v1, the stable backend defaults for every technical
        parameter, the resolved device and the machine-checked worker count.
        Benchmark and multi-mode configurations stay available through the
        ``configs/`` files and the CLI; they are not offered as ordinary
        training choices.
        """
        from ..resources import plan_resources

        base = TrainingConfig()
        return {
            "training": base.to_dict(),
            "default_run": {
                "mode": base.mode,
                "config": base.to_dict(),
                "device": base.resolved_device(),
                "workers": plan_resources(base.n_workers).to_dict(),
                "note": (
                    "automatic 1v1 configuration; every technical parameter keeps its "
                    "backend default and the worker count is checked against this machine"
                ),
            },
        }

    @app.post("/api/config/validate", dependencies=[Depends(require_auth)])
    def validate_config(config: TrainingConfig):
        from ..resources import plan_resources

        warnings: list[str] = []
        plan = plan_resources(config.n_workers)
        warnings.extend(plan.warnings)
        if config.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    warnings.append("device=cuda was requested but no CUDA device is available.")
            except Exception:
                warnings.append("device=cuda was requested but torch could not be imported.")
        if config.self_play and config.total_steps < 500_000:
            warnings.append(
                "Self-play needs many steps to be meaningful; this budget is very small."
            )
        if config.minibatch_size >= config.rollout_steps * config.n_workers:
            warnings.append(
                "minibatch_size covers the whole rollout: the policy is updated only "
                "once per iteration.  A smaller minibatch gives the learner more "
                "gradient steps per collected step."
            )
        return {
            "valid": True,
            "config": config.to_dict(),
            "warnings": warnings,
            "resources": plan.to_dict(),
        }

    @app.get("/api/runs")
    def list_runs(limit: int = 50):
        return store.list_runs(limit=limit)

    @app.post("/api/resume/plan")
    def resume_plan(request: ResumePlanRequest):
        """Preview which checkpoint a run would resume from.  Read-only."""
        from ..resume import plan_resume

        try:
            config = TrainingConfig.model_validate(request.config or {})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid config: {exc}") from exc
        if request.mode not in ("fresh", "continue", "transfer"):
            raise HTTPException(status_code=400, detail=f"unknown resume mode {request.mode!r}")
        try:
            plan = plan_resume(
                paths["models"],
                config,
                mode=request.mode,
                run_id=request.run_id,
                checkpoint=request.checkpoint,
                explicit_overrides=request.config_overrides,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return plan.to_dict()

    @app.post("/api/runs/start-plan")
    def start_plan(request: StartPlanRequest):
        """Compute what a simplified training start *would* do.  Read-only.

        The simplified workflow exposes exactly two user-controlled values
        (``total_steps`` and optionally ``run_name``).  This endpoint builds
        the effective configuration from the stable 1v1 backend defaults,
        checks the worker count against the machine (existing resource
        logic) and previews the checkpoint continuation (existing resume
        logic), so the GUI can show the user, before the confirmation
        dialog, exactly which run will start: continue from which checkpoint
        or initialise a new model, on which device, with how many workers.

        Nothing here starts training; ``POST /api/runs`` still requires
        ``confirm=true``.
        """
        from ..resources import plan_resources
        from ..resume import plan_resume

        config = TrainingConfig(total_steps=request.total_steps)
        if request.run_name:
            config.run_name = request.run_name
        if config.mode != "soccer-1v1":
            # the simplified workflow trains 1v1 only; the server decides
            # this, not the client
            config = config.model_copy(update={"mode": "soccer-1v1"})
        resume = plan_resume(paths["models"], config, mode="continue")
        resources = plan_resources(config.n_workers)
        return {
            "config": config.to_dict(),
            "device": config.resolved_device(),
            "workers": resources.to_dict(),
            "resume": resume.to_dict(),
            "will_resume": resume.will_resume,
            "resume_mode": "continue" if resume.will_resume else "fresh",
            "warnings": [*resume.warnings, *resources.warnings],
            "notes": resources.notes,
        }

    @app.post("/api/runs", dependencies=[Depends(require_auth)])
    def start_run(request: StartRunRequest):
        if not request.confirm:
            # The GUI asks the user before it posts; the server refuses a
            # start that was not confirmed, so a stray request can never
            # begin a training run.
            raise HTTPException(
                status_code=400,
                detail=(
                    "the training start was not confirmed; set confirm=true to start "
                    "the run (the GUI asks the user first)"
                ),
            )
        stored = request.model_dump(
            exclude={"resume_from", "resume_mode", "allow_concurrent", "confirm", "config_overrides"}
        )
        config = TrainingConfig.model_validate(stored)
        resume_from = request.resume_from
        explicit_mode = request.resume_mode
        if explicit_mode is not None and explicit_mode not in (
            "fresh",
            "continue",
            "transfer",
        ):
            raise HTTPException(
                status_code=400, detail=f"unknown resume mode {explicit_mode!r}"
            )
        if explicit_mode is None:
            # Automatic default.  Prefer a named checkpoint when the caller
            # supplied one; otherwise discover the newest usable checkpoint.
            from ..resume import plan_resume

            resume_mode = "continue"
            if not resume_from:
                plan = plan_resume(
                    paths["models"],
                    config,
                    mode="continue",
                    explicit_overrides=request.config_overrides,
                )
                if plan.will_resume:
                    resume_from = str(plan.checkpoint_path)
                elif plan.candidates_found:
                    reasons = [
                        reason
                        for item in plan.rejected
                        for reason in item.get("reasons", [])
                    ] or plan.warnings
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"automatic resume found {plan.candidates_found} checkpoint(s) "
                            "but none is usable: "
                            + "; ".join(reasons)
                            + ". Start deliberately with resume_mode='fresh' for a new "
                            "policy or resume_mode='transfer' to reuse weights."
                        ),
                    )
                else:
                    resume_mode = "fresh"
        else:
            resume_mode = explicit_mode
            if resume_mode == "fresh":
                resume_from = None
        if resume_mode in ("continue", "transfer") and not resume_from:
            # Automatic continuation: resolve the newest valid, compatible
            # checkpoint with the same logic the CLI uses by default and pass
            # the explicit path to the trainer.  A run that was asked to
            # continue is never turned into a fresh model silently; when no
            # suitable checkpoint exists the start is refused with a reason.
            from ..resume import plan_resume

            plan = plan_resume(
                paths["models"], config, mode=resume_mode,
                explicit_overrides=request.config_overrides,
            )
            if not plan.will_resume:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"cannot {resume_mode} automatically: "
                        + "; ".join(plan.warnings or ["no compatible checkpoint found"])
                    ),
                )
            resume_from = str(plan.checkpoint_path)
        if resume_from:
            try:
                resume_path = checkpoints.resolve_model_path(paths["models"], resume_from)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            resume_from = str(resume_path)
            stored["resume_from"] = resume_from
            # validate before creating a run row that could only fail
            from ..resume import plan_resume

            plan = plan_resume(
                paths["models"], config, mode=resume_mode, checkpoint=resume_from,
                explicit_overrides=request.config_overrides,
            )
            if not plan.will_resume:
                reasons = [
                    reason
                    for item in plan.rejected
                    for reason in item.get("reasons", [])
                ] or plan.warnings
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"cannot {resume_mode} from {resume_from}: " + "; ".join(reasons)
                    ),
                )
        stored["resume_mode"] = resume_mode
        run_id = store.create_run(config.run_name, config.mode, stored, config.total_steps)
        try:
            jobs.start_training(
                config,
                run_id,
                resume_from=resume_from,
                resume_mode=resume_mode,
                allow_concurrent=request.allow_concurrent,
                config_overrides=request.config_overrides,
            )
        except RuntimeError as exc:
            # an explicit refusal (another run is active), not a crash
            store.update_run(run_id, status="queued", message=str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            store.update_run(run_id, status="failed", message=str(exc))
            raise HTTPException(status_code=500, detail=f"could not start training: {exc}") from exc
        return store.get_run(run_id)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="unknown run")
        run["episode_summary"] = store.episode_summary(run_id)
        run["metric_names"] = store.metric_names(run_id)
        run["alive"] = jobs.job_alive(run_id)
        return run

    @app.post("/api/runs/{run_id}/pause", dependencies=[Depends(require_auth)])
    def pause_run(run_id: str):
        _require_run(store, run_id)
        return {"control": jobs.control(run_id, pause=True)}

    @app.post("/api/runs/{run_id}/resume", dependencies=[Depends(require_auth)])
    def resume_run(run_id: str):
        _require_run(store, run_id)
        return {"control": jobs.control(run_id, pause=False)}

    @app.post("/api/runs/{run_id}/stop", dependencies=[Depends(require_auth)])
    def stop_run(run_id: str):
        _require_run(store, run_id)
        return {"control": jobs.control(run_id, stop=True)}

    @app.get("/api/runs/{run_id}/metrics")
    def get_metrics(run_id: str, names: str | None = None, since_step: int = 0, limit: int = 5000):
        _require_run(store, run_id)
        name_list = [name for name in (names or "").split(",") if name]
        return store.get_metrics(run_id, name_list or None, since_step, limit)

    @app.get("/api/reports")
    def list_reports(limit: int = 50):
        return store.list_reports(limit=limit)

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str, rebuild: bool = False):
        _require_run(store, run_id)
        from ..report import build_run_report

        if rebuild:
            # explicit rebuild wins: the stored copy may predate a report fix
            return build_run_report(
                store,
                run_id,
                models_dir=paths["models"],
                run_dir=paths["runs"] / run_id,
            )
        stored = store.get_report(run_id)
        if stored and stored.get("report"):
            return stored["report"]
        raise HTTPException(
            status_code=404,
            detail=(
                f"run {run_id} has no stored report (it finished before reporting existed "
                "or crashed before writing one); request ?rebuild=true to recompute it "
                "from the database"
            ),
        )

    @app.post("/api/runs/compare")
    def compare_runs_endpoint(request: CompareRunsRequest):
        from ..report import compare_runs

        unknown = [item for item in request.run_ids if not store.get_run(item)]
        if unknown:
            raise HTTPException(status_code=404, detail=f"unknown runs: {', '.join(unknown)}")
        return compare_runs(store, request.run_ids)

    @app.get("/api/runs/{run_id}/logs")
    def get_logs(run_id: str, after: int = 0, limit: int = 500):
        return store.tail_logs(run_id, after, limit)

    @app.get("/api/logs")
    def all_logs(after: int = 0, limit: int = 500):
        return store.tail_logs(None, after, limit)

    @app.get("/api/models")
    def list_models():
        store.sync_models(checkpoints.list_models(paths["models"]))
        return enrich_models(store, paths["models"])

    @app.get("/api/models/{model_id}")
    def get_model(model_id: str):
        _validate_model_id(model_id)
        try:
            manifest = checkpoints.get_model(paths["models"], model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return manifest

    @app.post("/api/models/{model_id}/validate")
    def validate_model(model_id: str, request: ValidateModelRequest | None = None):
        """Compatibility check used by the GUI before offering a resume."""
        _validate_model_id(model_id)
        try:
            manifest = checkpoints.get_model(paths["models"], model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        request = request or ValidateModelRequest()
        config = None
        if request.config:
            try:
                config = TrainingConfig.model_validate(request.config)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid config: {exc}") from exc
        validation = checkpoints.validate_checkpoint(
            paths["models"] / manifest["path"],
            policy_config=config.policy_config().to_dict() if config else None,
            mode=config.mode if config else None,
            obs_version=config.obs_version if config else None,
            require_optimizer=request.require_optimizer,
        )
        payload = validation.to_dict()
        payload["model_id"] = model_id
        payload["manifest"] = manifest
        payload["resume"] = {
            "continue": validation.ok and validation.has_optimizer,
            "transfer": validation.ok,
            "missing_state": [
                name
                for name, present in (
                    ("optimizer", validation.has_optimizer),
                    ("rng", validation.has_rng),
                    ("curriculum", validation.has_curriculum),
                    ("league", validation.has_league),
                )
                if not present
            ],
        }
        return payload

    @app.post("/api/models/{model_id}/promote", dependencies=[Depends(require_auth)])
    def promote(model_id: str):
        _validate_model_id(model_id)
        try:
            manifest = checkpoints.promote_model(paths["models"], model_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.sync_models(checkpoints.list_models(paths["models"]))
        store.log(None, "INFO", f"model {model_id} promoted to stable")
        return manifest

    @app.delete("/api/models/{model_id}", dependencies=[Depends(require_auth)])
    def delete_model(model_id: str):
        _validate_model_id(model_id)
        try:
            checkpoints.delete_model(paths["models"], model_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.sync_models(checkpoints.list_models(paths["models"]))
        return {"deleted": model_id}

    def _start_evaluation(model_id: str, request: EvaluateRequest, *, suite: bool) -> dict:
        _validate_model_id(model_id)
        request.budget_check()
        try:
            manifest = checkpoints.get_model(paths["models"], model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        model_path = paths["models"] / manifest["path"]
        mode = request.mode or manifest.get("mode") or "soccer-1v1"
        get_mode(mode)
        opponents = request.effective_opponents()
        if suite and len(opponents) < 2:
            raise HTTPException(
                status_code=400, detail="a suite needs at least two opponents"
            )
        for name in opponents:
            _check_opponent(name, paths["models"])
        job_id = jobs.start_evaluation(
            model_id=model_id,
            model_path=model_path,
            mode=mode,
            opponent=opponents[0],
            opponents=opponents if suite else None,
            games=request.games,
            seed=request.seed,
            match_seconds=request.match_seconds,
            device=request.device,
            sides=request.sides or ["blue"],
            mirror=request.mirror,
            stochastic=request.stochastic,
            phase="suite" if suite else request.phase,
        )
        return {
            "job_id": job_id,
            "model_id": model_id,
            "status": "started",
            "opponents": opponents,
            "sides": request.sides or ["blue"],
            "mirror": request.mirror,
            "kind": "suite" if suite else "single",
        }

    @app.post("/api/models/{model_id}/evaluate", dependencies=[Depends(require_auth)])
    def evaluate(model_id: str, request: EvaluateRequest):
        return _start_evaluation(model_id, request, suite=False)

    @app.post("/api/evaluations/suite", dependencies=[Depends(require_auth)])
    def evaluate_suite_endpoint(request: EvaluateRequest):
        """Multi-opponent, side-balanced evaluation (separates diagnostics from strength)."""
        return _start_evaluation(request.model_id, request, suite=True)

    @app.get("/api/evaluations")
    def evaluations(model_id: str | None = None, phase: str | None = None, limit: int = 100):
        items = store.list_evaluations(phase=phase, limit=limit)
        if model_id:
            items = [item for item in items if item.get("model_id") == model_id]
        return items

    @app.get("/api/live/status")
    def live_status():
        import platform

        stable = checkpoints.stable_model(paths["models"])
        return {
            "platform": platform.system(),
            "is_windows": platform.system() == "Windows",
            "stable_model": stable,
            "rlbot_installed": _rlbot_installed(),
            "launch_supported": platform.system() == "Windows" and _rlbot_installed(),
            "note": "Rocket League must be running and RLBot v5 installed. Live matches cannot be verified inside this sandbox.",
        }

    @app.get("/api/live/preflight")
    def live_preflight(model_id: str | None = None):
        from ..live import preflight

        if model_id:
            _validate_model_id(model_id)
        return preflight(paths, model_id=model_id)

    @app.post("/api/live/launch", dependencies=[Depends(require_auth)])
    def live_launch(request: LiveLaunchRequest):
        from ..live import LiveUnavailable, launch_match

        if request.model_id:
            # the id ends up in the bot command line; keep it a plain identifier
            _validate_model_id(request.model_id)
        if request.human_team not in ("blue", "orange"):
            raise HTTPException(status_code=400, detail=f"unknown side {request.human_team!r}")
        if request.fill not in ("psyonix", "policy"):
            raise HTTPException(status_code=400, detail=f"unknown fill {request.fill!r}")
        try:
            return launch_match(
                paths,
                request.model_id,
                request.mode,
                human_team=request.human_team,
                fill=request.fill,
                human=request.human,
                dry_run=request.dry_run,
                timeout=request.timeout,
            )
        except LiveUnavailable as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.get("/api/events", dependencies=[Depends(require_auth)])
    async def events(request: Request, run_id: str | None = None, after: int = 0):
        async def generator():
            last_log = int(after)
            last_metrics = 0.0
            while True:
                if await request.is_disconnected():
                    break
                try:
                    logs = store.tail_logs(run_id, last_log, 100)
                    for item in logs:
                        last_log = max(last_log, item["id"])
                        yield "data: " + json.dumps({"type": "log", **item}) + "\n\n"
                    if run_id:
                        now = time.time()
                        if now - last_metrics > 1.0:
                            last_metrics = now
                            run = store.get_run(run_id)
                            metrics = store.get_metrics(run_id, since_step=0, limit=2000)
                            latest: dict[str, float] = {}
                            for metric in metrics:
                                latest[metric["name"]] = metric["value"]
                            yield (
                                "data: "
                                + json.dumps(
                                    {
                                        "type": "metrics",
                                        "run": run,
                                        "metrics": latest,
                                        "summary": store.episode_summary(run_id),
                                    }
                                )
                                + "\n\n"
                            )
                    else:
                        yield "data: " + json.dumps({"type": "ping", "ts": time.time()}) + "\n\n"
                except Exception as exc:  # keep the stream alive
                    yield "data: " + json.dumps({"type": "error", "error": str(exc)}) + "\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    # -- statistics --------------------------------------------------------
    @app.get("/api/stats")
    def get_stats(run_id: str | None = None, since_step: int = 0):
        """Everything the Stats tab shows.

        Without a run id the most recent run is used, so the tab always has
        something to show.  Values that were never measured are reported as
        unavailable - never as a zero.
        """
        from ..stats import build_stats

        target = run_id
        if not target:
            runs = store.list_runs(limit=1)
            if not runs:
                return {
                    "available": False,
                    "reason": "no run has been recorded yet",
                }
            target = runs[0]["id"]
        _require_run(store, target)
        return build_stats(
            store,
            target,
            models_dir=paths["models"],
            resources=resources(),
            since_step=since_step,
        )

    @app.get("/api/stats/overview")
    def stats_overview(limit: int = 20):
        """A compact overview of every run, for the history table."""
        from ..stats import timing_block

        rows = []
        for run in store.list_runs(limit=limit):
            timing = timing_block(store, run)
            rows.append(
                {
                    "id": run["id"],
                    "name": run.get("name"),
                    "mode": run.get("mode"),
                    "status": run.get("status"),
                    "step": run.get("step"),
                    "total_steps": run.get("total_steps"),
                    "created_at": run.get("created_at"),
                    "started_at": run.get("started_at"),
                    "ended_at": run.get("ended_at"),
                    "accumulated_s": timing.get("accumulated_s"),
                    "accumulated_human": timing.get("accumulated_human"),
                    "session_count": timing.get("session_count"),
                    "interrupted_count": timing.get("interrupted_count"),
                    "last_checkpoint_at": run.get("last_checkpoint_at"),
                    "last_checkpoint_id": run.get("last_checkpoint_id"),
                }
            )
        return {"runs": rows, "count": len(rows)}

    # -- team builder ------------------------------------------------------
    @app.get("/api/teams")
    def list_teams_endpoint():
        from ..team_builder import describe_team, team_from_any

        teams = []
        for row in store.list_teams():
            try:
                team = team_from_any(row.get("team") or {})
            except Exception:
                continue
            described = describe_team(team)
            described["updated_at"] = row.get("updated_at")
            teams.append(described)
        return {"teams": teams, "count": len(teams)}

    @app.post("/api/teams", dependencies=[Depends(require_auth)])
    def save_team(request: TeamRequest):
        from ..team_builder import describe_team, validate_team

        try:
            team = validate_team(request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.save_team(team.name, describe_team(team), mode=team.mode)
        return describe_team(team)

    @app.get("/api/teams/{name}")
    def get_team_endpoint(name: str):
        row = store.get_team(name)
        if not row:
            raise HTTPException(status_code=404, detail=f"unknown team {name!r}")
        return row

    @app.delete("/api/teams/{name}", dependencies=[Depends(require_auth)])
    def delete_team_endpoint(name: str):
        deleted = store.delete_team(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"unknown team {name!r}")
        return {"deleted": name}

    @app.get("/api/difficulties")
    def list_difficulties():
        from ..difficulty import describe_all

        return describe_all()

    @app.get("/api/bot-behaviour")
    def list_bot_behaviour():
        from ..botbrain import available_profiles

        return {"profiles": available_profiles()}

    @app.post("/api/bot-behaviour/preview", dependencies=[Depends(require_auth)])
    def preview_bot_behaviour(request: BotSettingsRequest):
        """Show what a bot would decide in a given situation."""
        from ..botbrain import BotBrain, BrainContext
        from ..difficulty import brain_parameters

        try:
            parameters = brain_parameters(request.difficulty, request.behaviour)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.overrides:
            from ..botbrain import merge_overrides

            parameters = merge_overrides(parameters, request.overrides)
        brain = BotBrain(profile=request.behaviour, parameters=parameters)
        context = BrainContext()
        decision = brain.decide(context)
        return {
            "brain": brain.to_dict(),
            "decision": decision.to_dict(),
            "note": "the context is the default one; the live bot uses the real state",
        }

    # -- player memory -----------------------------------------------------
    @app.get("/api/player-memory")
    def get_player_memory():
        from ..player_memory import memory_from_rows

        memory = memory_from_rows(store.list_player_memory())
        return {
            "summary": memory.summary(),
            "players": memory.to_dict()["players"],
            "note": (
                "the player memory is data about past encounters; it is never part "
                "of a checkpoint and never influences the policy weights"
            ),
        }

    @app.post("/api/player-memory/note", dependencies=[Depends(require_auth)])
    def set_player_note(request: PlayerMemoryRequest):
        from ..player_memory import memory_from_rows, rows_from_memory

        memory = memory_from_rows(store.list_player_memory())
        try:
            memory.set_note(request.name, request.note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store.replace_player_memory(rows_from_memory(memory))
        return {"ok": True, "name": request.name}

    @app.post("/api/player-memory/reset", dependencies=[Depends(require_auth)])
    def reset_player_memory():
        deleted = store.delete_player_memory()
        return {"deleted": deleted}

    @app.delete("/api/player-memory/{name}", dependencies=[Depends(require_auth)])
    def delete_player(name: str):
        deleted = store.delete_player_memory(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"unknown player {name!r}")
        return {"deleted": name}

    @app.get("/api/player-memory/export")
    def export_player_memory():
        from ..player_memory import memory_from_rows

        memory = memory_from_rows(store.list_player_memory())
        return memory.export()

    # -- rivals ------------------------------------------------------------
    @app.get("/api/rivals")
    def list_rivals():
        from ..rivals import RivalRegistry

        registry = RivalRegistry()
        for row in store.list_rivals():
            data = row.get("rival") or {}
            registry.register(
                row["name"],
                kind=row.get("kind") or data.get("kind") or "checkpoint",
                reference=row.get("reference") or data.get("reference") or row["name"],
                description=row.get("description") or data.get("description") or "",
            )
        # merge the measured history from the evaluations
        registry.build_from_evaluations(store.list_evaluations(limit=500))
        for rival in registry.rivals.values():
            store.save_rival(rival.name, rival.to_dict())
        return registry.to_dict()

    @app.post("/api/rivals", dependencies=[Depends(require_auth)])
    def save_rival(request: RivalRequest):
        from ..rivals import RivalRegistry

        registry = RivalRegistry()
        registry.register(
            request.name,
            kind=request.kind,
            reference=request.reference or request.name,
            description=request.description,
        )
        rival = registry.get(request.name)
        assert rival is not None
        store.save_rival(rival.name, rival.to_dict())
        return rival.to_dict()

    @app.delete("/api/rivals/{name}", dependencies=[Depends(require_auth)])
    def delete_rival(name: str):
        deleted = store.delete_rival(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"unknown rival {name!r}")
        return {"deleted": name}

    @app.get("/api/rivals/{name}/trend")
    def rival_trend(name: str):
        from ..rivals import RivalRegistry

        registry = RivalRegistry().build_from_evaluations(store.list_evaluations(limit=500))
        return registry.trend(name)

    # -- match analysis ----------------------------------------------------
    @app.get("/api/analysis")
    def match_analysis(run_id: str | None = None):
        from ..replay import analyze_matches

        return analyze_matches(store, run_id, models_dir=paths["models"])

    # -- bot evaluation lab ------------------------------------------------
    @app.get("/api/lab")
    def lab_overview():
        """The Bot Evaluation Lab: what can be measured right now."""
        from ..evaluation import DIAGNOSTIC_OPPONENTS, STRENGTH_OPPONENTS

        models = enrich_models(store, paths["models"])
        return {
            "models": [
                {
                    "id": model.get("id"),
                    "step": model.get("step"),
                    "created_at": model.get("created_at"),
                    "eval_score": model.get("eval_score"),
                    "duration_s": model.get("duration_s"),
                    "duration_human": model.get("duration_human"),
                    "state": model.get("state"),
                }
                for model in models
            ],
            "opponents": {
                "strength": list(STRENGTH_OPPONENTS),
                "diagnostic": list(DIAGNOSTIC_OPPONENTS),
            },
            "recent_evaluations": store.list_evaluations(limit=25),
            "note": (
                "an experiment is only executed when the GUI asks for a "
                "confirmation; planning it never starts a simulation"
            ),
        }

    @app.post("/api/lab/plan", dependencies=[Depends(require_auth)])
    def lab_plan(request: EvalLabRequest):
        """Plan a repeatable experiment.  Never starts a simulation."""
        from ..evaluation import classify_opponent

        opponents = request.opponents or ["chase"]
        unknown = []
        for name in opponents:
            try:
                from ..opponents import validate_opponent_name

                validate_opponent_name(name)
            except ValueError as exc:
                unknown.append(str(exc))
        if unknown:
            raise HTTPException(status_code=400, detail="; ".join(unknown))
        model_ids = []
        for model_id in request.model_ids:
            _validate_model_id(model_id)
            try:
                checkpoints.get_model(paths["models"], model_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            model_ids.append(model_id)
        sides = [side for side in request.sides if side]
        games = request.games * max(1, len(sides))
        projected = games * len(opponents) * request.match_seconds
        return {
            "models": model_ids,
            "opponents": opponents,
            "opponent_classes": {name: classify_opponent(name) for name in opponents},
            "games_per_model": games,
            "total_games": games * len(opponents) * max(1, len(model_ids)),
            "projected_sim_seconds": projected,
            "projected_sim_minutes": round(projected / 60.0, 1),
            "mode": request.mode,
            "seed": request.seed,
            "sides": sides,
            "mirror": request.mirror,
            "executed": False,
            "note": "planning an experiment never starts a simulation",
        }

    @app.post("/api/lab/run", dependencies=[Depends(require_auth)])
    def lab_run(request: EvalLabRequest):
        """Execute a planned experiment (only after an explicit confirmation)."""
        from ..evaluation import evaluate_model

        if not request.execute:
            raise HTTPException(
                status_code=400,
                detail="the experiment was not confirmed; set execute=true to run it",
            )
        if not request.model_ids:
            raise HTTPException(status_code=400, detail="no checkpoint selected")
        if not request.opponents:
            raise HTTPException(status_code=400, detail="no opponent selected")
        results = []
        for model_id in request.model_ids:
            _validate_model_id(model_id)
            try:
                manifest = checkpoints.get_model(paths["models"], model_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            path = paths["models"] / manifest["path"]
            for opponent in request.opponents:
                result = evaluate_model(
                    path,
                    mode=request.mode,
                    opponent=opponent,
                    games=request.games,
                    seed=request.seed,
                    match_seconds=request.match_seconds,
                    device="cpu",
                    sides=tuple(request.sides),
                    mirror=request.mirror,
                )
                result["model_id"] = model_id
                result["phase"] = "lab"
                store.add_evaluation(result)
                results.append(
                    {
                        "model_id": model_id,
                        "opponent": opponent,
                        "games": result.get("games"),
                        "wins": result.get("wins"),
                        "losses": result.get("losses"),
                        "draws": result.get("draws"),
                        "win_rate": result.get("win_rate"),
                        "win_rate_ci95": result.get("win_rate_ci95"),
                    }
                )
        return {"executed": True, "results": results, "count": len(results)}

    # -- static GUI --------------------------------------------------------
    web_dir = Path(__file__).resolve().parents[1] / "web"
    theme_dir = project_root() / "ui" / "theme"
    if theme_dir.exists():
        app.mount("/theme", StaticFiles(directory=theme_dir), name="theme")
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app


def _check_opponent(name: str, models_dir: Path) -> None:
    """Reject unknown opponents here instead of letting a spawned job die late."""
    from ..opponents import build_opponent

    if not isinstance(name, str) or not name.strip():
        raise HTTPException(status_code=400, detail="opponent name must be a non-empty string")
    name = name.strip()
    if name.startswith("model:"):
        try:
            checkpoints.resolve_model_path(models_dir, name.split(":", 1)[1])
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"opponent {name!r}: {exc}") from exc
        return
    try:
        build_opponent(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _require_run(store: Store, run_id: str) -> dict:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="unknown run")
    return run


def _validate_model_id(model_id: str) -> None:
    if not MODEL_ID_RE.match(model_id) or ".." in model_id:
        raise HTTPException(status_code=400, detail="invalid model id")


def _rlbot_installed() -> bool:
    try:
        import rlbot  # noqa: F401

        return True
    except Exception:
        return False
