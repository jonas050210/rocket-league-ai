"""FastAPI application: JSON API, SSE stream and static GUI."""

from __future__ import annotations

import asyncio
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
from ..config import TrainingConfig, data_dir, ensure_dirs, project_root
from ..hardware import benchmark, diagnostics, system_info
from ..jobs import JobManager
from ..metrics_util import enrich_models
from ..modes import all_modes, get_mode
from ..store import Store
from ..version import __version__

MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class StartRunRequest(TrainingConfig):
    """Training configuration plus an optional checkpoint to continue from."""

    resume_from: str | None = None


class BenchmarkRequest(BaseModel):
    mode: str = "soccer-1v1"
    seconds: float = Field(default=3.0, ge=0.5, le=10.0)
    workers: int = Field(default=1, ge=1, le=8)


class EvaluateRequest(BaseModel):
    model_id: str
    mode: str | None = None
    opponent: str = "chase"
    games: int = Field(default=20, ge=1, le=500)
    seed: int = 1000
    match_seconds: float = Field(default=120.0, ge=10, le=600)
    device: str = "cpu"


class LiveLaunchRequest(BaseModel):
    model_id: str | None = None
    mode: str = "soccer-1v1"
    human_team: str = "blue"
    fill: str = "psyonix"
    human: bool = True
    timeout: float = 0.0
    dry_run: bool = False


def create_app(store_path: Path | None = None) -> FastAPI:
    paths = ensure_dirs()
    store = Store(store_path or paths["root"] / "rlai.sqlite3")
    jobs = JobManager(store, paths)
    token = os.environ.get("RLAI_AUTH_TOKEN")

    app = FastAPI(title="Rocket League AI", version=__version__)
    app.state.store = store
    app.state.jobs = jobs
    app.state.paths = paths
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_auth(request: Request) -> None:
        if not token:
            return
        provided = request.headers.get("x-rlai-token") or request.query_params.get("token")
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            provided = auth[7:]
        if provided != token:
            raise HTTPException(status_code=401, detail="invalid capability token")

    # refresh model registry from disk on startup
    try:
        store.sync_models(checkpoints.list_models(paths["models"]))
    except Exception:
        pass

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
        return benchmark(request.mode, request.seconds, request.workers)

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

    @app.post("/api/runs", dependencies=[Depends(require_auth)])
    def start_run(request: StartRunRequest):
        resume_from = request.resume_from
        stored = request.model_dump(exclude={"resume_from"})
        config = TrainingConfig.model_validate(stored)
        if resume_from:
            try:
                resume_path = checkpoints.resolve_model_path(paths["models"], resume_from)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            resume_from = str(resume_path)
            stored["resume_from"] = resume_from
        run_id = store.create_run(config.run_name, config.mode, stored, config.total_steps)
        try:
            jobs.start_training(config, run_id, resume_from=resume_from)
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

    @app.post("/api/models/{model_id}/evaluate", dependencies=[Depends(require_auth)])
    def evaluate(model_id: str, request: EvaluateRequest):
        _validate_model_id(model_id)
        try:
            manifest = checkpoints.get_model(paths["models"], model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        model_path = paths["models"] / manifest["path"]
        mode = request.mode or manifest.get("mode") or "soccer-1v1"
        get_mode(mode)
        job_id = jobs.start_evaluation(
            model_id=model_id,
            model_path=model_path,
            mode=mode,
            opponent=request.opponent,
            games=request.games,
            seed=request.seed,
            match_seconds=request.match_seconds,
            device=request.device,
        )
        return {"job_id": job_id, "model_id": model_id, "status": "started"}

    @app.get("/api/evaluations")
    def evaluations(model_id: str | None = None, limit: int = 100):
        items = store.list_evaluations(limit=limit)
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
    def live_preflight():
        from ..live import preflight

        return preflight(paths)

    @app.post("/api/live/launch", dependencies=[Depends(require_auth)])
    def live_launch(request: LiveLaunchRequest):
        from ..live import LiveUnavailable, launch_match

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

    @app.get("/api/events")
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

    @app.on_event("shutdown")
    def _shutdown():
        jobs.shutdown()

    return app


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
