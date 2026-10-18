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
    """Training configuration plus an explicit resume decision.

    ``resume_mode`` is never inferred silently: "fresh" starts a new policy,
    "continue" restores weights/optimizer/step/RNG/curriculum from a compatible
    checkpoint and "transfer" keeps only the weights and restarts the step count.
    """

    resume_from: str | None = None
    resume_mode: str = "fresh"
    allow_concurrent: bool = False


class ResumePlanRequest(BaseModel):
    """Ask what a resume *would* restore.  Never starts training."""

    config: dict
    mode: str = "continue"
    run_id: str | None = None
    checkpoint: str | None = None


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


def _settings() -> "AppSettings":
    """Validated exposure settings shared with ``rlai serve``."""
    from ..config import AppSettings

    return AppSettings.from_env()


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
        memory = psutil.virtual_memory()
        payload = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": os.cpu_count(),
            "memory_percent": memory.percent,
            "memory_used_gb": round((memory.total - memory.available) / 1024**3, 2),
            "memory_total_gb": round(memory.total / 1024**3, 2),
            "active_run": store.latest_active_run(),
        }
        try:
            import torch

            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(idx)
                payload["gpu"] = {
                    "name": properties.name,
                    "memory_total_mb": round(properties.total_memory / 1024**2, 1),
                    "memory_allocated_mb": round(torch.cuda.memory_allocated(idx) / 1024**2, 1),
                    "source": "torch (training runs in a subprocess, so allocated memory is not live)",
                }
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
        return {
            "training": TrainingConfig().to_dict(),
            "presets": _presets(),
        }

    @app.post("/api/config/validate", dependencies=[Depends(require_auth)])
    def validate_config(config: TrainingConfig):
        warnings: list[str] = []
        cores = os.cpu_count() or 1
        if config.n_workers > cores:
            warnings.append(
                f"{config.n_workers} workers on {cores} logical cores will oversubscribe the CPU."
            )
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
        return {"valid": True, "config": config.to_dict(), "warnings": warnings}

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
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return plan.to_dict()

    @app.post("/api/runs", dependencies=[Depends(require_auth)])
    def start_run(request: StartRunRequest):
        resume_from = request.resume_from
        resume_mode = request.resume_mode or "fresh"
        if resume_mode not in ("fresh", "continue", "transfer"):
            raise HTTPException(status_code=400, detail=f"unknown resume mode {resume_mode!r}")
        if resume_mode == "fresh":
            resume_from = None
        stored = request.model_dump(
            exclude={"resume_from", "resume_mode", "allow_concurrent"}
        )
        config = TrainingConfig.model_validate(stored)
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
                paths["models"], config, mode=resume_mode, checkpoint=resume_from
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


def _presets() -> list[dict]:
    base = TrainingConfig()
    return [
        {
            "id": "ball-control",
            "name": "Ball control warm-up",
            "description": "Short episodes, dense touch and approach rewards. Learn to reach the ball.",
            "config": base.model_copy(
                update={
                    "run_name": "ball control",
                    "mode": "soccer-1v1",
                    "opponent": "idle",
                    "total_steps": 100_000,
                    "episode_seconds": 15,
                    "n_workers": 4,
                    "rollout_steps": 512,
                    "eval_interval": 10,
                    "eval_opponent": "idle",
                    "eval_games": 4,
                    "eval_match_seconds": 20,
                }
            ).to_dict(),
        },
        {
            "id": "1v1-chase",
            "name": "1v1 vs chaser",
            "description": "First real match baseline against a scripted opponent that drives at the ball.",
            "config": base.model_copy(
                update={
                    "run_name": "1v1 vs chaser",
                    "mode": "soccer-1v1",
                    "opponent": "chase",
                    "total_steps": 2_000_000,
                    "n_workers": 6,
                    "eval_interval": 10,
                    "eval_games": 6,
                }
            ).to_dict(),
        },
        {
            "id": "1v1-curriculum",
            "name": "1v1 curriculum",
            "description": "Staged opponents (idle → chaser → mixed → self-play); promotion only on a measured win rate.",
            "config": base.model_copy(
                update={
                    "run_name": "1v1 curriculum",
                    "mode": "soccer-1v1",
                    "opponent": "idle",
                    "curriculum": True,
                    "total_steps": 2_000_000,
                    "n_workers": 6,
                    "eval_interval": 10,
                    "eval_games": 6,
                }
            ).to_dict(),
        },
        {
            "id": "1v1-selfplay",
            "name": "1v1 self-play",
            "description": "Opponent pool of frozen snapshots. Use after the chaser baseline is stable.",
            "config": base.model_copy(
                update={
                    "run_name": "1v1 self-play",
                    "mode": "soccer-1v1",
                    "self_play": True,
                    "total_steps": 10_000_000,
                    "n_workers": 8,
                    "eval_interval": 20,
                }
            ).to_dict(),
        },
        {
            "id": "2v2",
            "name": "2v2 team play",
            "description": "Shared policy for both blue cars, 4 s decision window, self-play opponents.",
            "config": base.model_copy(
                update={
                    "run_name": "2v2 team",
                    "mode": "soccer-2v2",
                    "self_play": True,
                    "total_steps": 20_000_000,
                    "n_workers": 8,
                    "eval_interval": 20,
                }
            ).to_dict(),
        },
        {
            "id": "3v3",
            "name": "3v3 team play",
            "description": "Three cars per team. Observation is already padded; fine-tune from 2v2.",
            "config": base.model_copy(
                update={
                    "run_name": "3v3 team",
                    "mode": "soccer-3v3",
                    "self_play": True,
                    "total_steps": 30_000_000,
                    "n_workers": 8,
                    "eval_interval": 20,
                }
            ).to_dict(),
        },
        {
            "id": "heatseeker",
            "name": "Heatseeker 2v2",
            "description": "Homing ball, unlimited boost. Train separately: reward and dynamics differ.",
            "config": base.model_copy(
                update={
                    "run_name": "heatseeker 2v2",
                    "mode": "heatseeker-2v2",
                    "self_play": True,
                    "total_steps": 10_000_000,
                    "n_workers": 8,
                    "eval_interval": 20,
                }
            ).to_dict(),
        },
    ]
