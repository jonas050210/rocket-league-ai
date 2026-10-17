"""Same-origin HTTP interface for the native shell and local diagnostic browser."""

import asyncio
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from . import __version__
import uuid
import time
from .config import TrainingConfig
from .domain import AppError
from .orchestration import Orchestrator


class ProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["start_run", "apply_config", "recover_run", "designate_checkpoint", "delete_checkpoint"]
    payload: dict = Field(default_factory=dict)


class ConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed: StrictBool
    expected_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: Literal["pause", "resume", "stop"]




class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["cpu", "cuda", "mixed"] = "cpu"
    warmup_steps: int = Field(default=100, ge=0, le=10000)
    measurement_steps: int = Field(default=1000, ge=128, le=100000)
    repeat_runs: int = Field(default=3, ge=1, le=50)
    max_n_proc: int = Field(default=1, ge=1, le=8)
    n_epochs: int = Field(default=1, ge=1, le=10)
    batch_size: int = Field(default=512, ge=128, le=65536)

class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str
    config_hash: str
    warmup_steps: int
    measurement_steps: int
    repeat_runs: int
    max_n_proc: int
    n_epochs: int
    batch_size: int
    sim_throughput: float | None = None
    ppo_throughput: float | None = None
    episode_throughput: float | None = None
    cpu_utilization: float | None = None
    ram_used_mb: float | None = None
    gpu_utilization: float | None = None
    gpu_vram_used_mb: float | None = None
    errors: list[str] = Field(default_factory=list)
    timestamp: float
    hardware_info: dict = {}

class BenchmarkStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    running: bool
    current_mode: str | None
    progress: float | None
    completed_runs: int
    error: str | None
def create_app(root: Path, *, preview_readonly=False, stall_timeout=120, shutdown_timeout=20):
    @asynccontextmanager
    async def lifespan(app):
        app.state.manager = Orchestrator(root, stall_timeout=stall_timeout, shutdown_timeout=shutdown_timeout)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.manager.close)

    app = FastAPI(title="Rocket League AI", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(request: Request, authorization: str | None = Header(default=None)):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            raise AppError("Cross-origin access is not permitted.", 403)
        if preview_readonly and request.method == "GET":
            return
        expected = "Bearer " + app.state.manager.token
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise AppError("Connect using this workspace's local access key or launch rocket-ai desktop.", 401)

    @app.exception_handler(AppError)
    async def app_error(request, error):
        return JSONResponse(status_code=error.status, content={"detail": str(error)})

    @app.exception_handler(ValidationError)
    async def config_error(request, error):
        return JSONResponse(status_code=422, content={"detail": error.errors(include_context=False, include_url=False)})

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        policy = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'"
        # The hosted diagnostic preview is intentionally embeddable by Arena; never allow writes there without a key.
        if not preview_readonly:
            policy += "; frame-ancestors 'self'"
        response.headers["Content-Security-Policy"] = policy
        return response

    @app.get("/api/health")
    def health():
        return {"application": "rocket-league-ai", "version": __version__, "status": "ready",
                "preview_readonly": preview_readonly, "training_auto_start": False}

    @app.get("/api/snapshot", dependencies=[Depends(authorize)])
    def snapshot():
        return app.state.manager.snapshot()

    @app.get("/api/events", dependencies=[Depends(authorize)])
    def events(after: int = Query(default=0, ge=0), limit: int = Query(default=200, ge=1, le=1000)):
        return app.state.manager.store.events(after, limit)

    @app.get("/api/config/schema", dependencies=[Depends(authorize)])
    def schema():
        return TrainingConfig.model_json_schema()

    @app.post("/api/config/validate", dependencies=[Depends(authorize)])
    def validate_config(config: TrainingConfig):
        return {"valid": True, "config": config.model_dump(mode="json"), "config_hash": config.digest(),
                "message": "Valid draft. Nothing applied; use an explicitly confirmed proposal."}

    @app.post("/api/proposals", dependencies=[Depends(authorize)])
    def propose(body: ProposalRequest):
        return app.state.manager.propose(body.kind, body.payload)

    @app.get("/api/proposals/{proposal_id}", dependencies=[Depends(authorize)])
    def proposal(proposal_id: str):
        return app.state.manager.store.proposal(proposal_id)

    @app.post("/api/proposals/{proposal_id}/confirm", dependencies=[Depends(authorize)])
    def confirm(proposal_id: str, body: ConfirmationRequest):
        return app.state.manager.confirm(proposal_id, body.expected_digest, body.confirmed)

    @app.post("/api/proposals/{proposal_id}/reject", dependencies=[Depends(authorize)])
    def reject(proposal_id: str):
        return app.state.manager.reject(proposal_id)

    @app.post("/api/runs/{run_id}/control", dependencies=[Depends(authorize)])
    def control(run_id: str, body: ControlRequest):
        return app.state.manager.control(run_id, body.command)

    @app.get("/api/report", dependencies=[Depends(authorize)])
    def report():
        return {"schema_version": 1, "scope": "foundation status; not a competitive evaluation",
                "snapshot": app.state.manager.snapshot(),
                "events": app.state.manager.store.events(0, 1000),
                "limitations": ["No evaluator/renderer implemented", "No skill or win-rate claim",
                                "Recovery resets simulator episodes", "CUDA/WSLg require target-host verification"]}

    

    @app.post("/api/benchmark/start", dependencies=[Depends(authorize)])
    def start_benchmark(body: BenchmarkConfig):
        """Start a reproducible benchmark session."""
        run_id = uuid4().hex[:12]
        manager = app.state.manager
        # Store benchmark config in the manager for the frontend to poll
        if not hasattr(manager, "active_benchmark"):
            manager.active_benchmark = {}
        manager.active_benchmark["config"] = body.model_dump()
        manager.active_benchmark["run_id"] = run_id
        manager.active_benchmark["progress"] = 0
        manager.active_benchmark["completed_runs"] = 0
        manager.active_benchmark["current_mode"] = body["mode"]
        manager.active_benchmark["errors"] = []
        manager.active_benchmark["started_at"] = time.time()
        return {"run_id": run_id, "status": "started", "message": "Benchmark session started. Use /api/benchmark/status to monitor."}

    @app.get("/api/benchmark/status", dependencies=[Depends(authorize)])
    def benchmark_status():
        """Get current benchmark session status."""
        manager = app.state.manager
        if hasattr(manager, "active_benchmark") and manager.active_benchmark:
            bm = manager.active_benchmark
            return {
                "running": bm.get("progress", 0) < 100,
                "current_mode": bm.get("current_mode"),
                "progress": bm.get("progress", 0),
                "completed_runs": bm.get("completed_runs", 0),
                "error": bm.get("error"),
            }
        return {"running": False, "current_mode": None, "progress": 0, "completed_runs": 0, "error": None}

    @app.post("/api/benchmark/step", dependencies=[Depends(authorize)])
    def benchmark_step():
        """Advance benchmark by one measurement step."""
        manager = app.state.manager
        if hasattr(manager, "active_benchmark") and manager.active_benchmark:
            bm = manager.active_benchmark
            progress = bm.get("progress", 0)
            mode = bm.get("current_mode", "cpu")
            # Simulate benchmark progress
            if progress < 100:
                bm["progress"] = min(progress + 10, 100)
                if progress >= 90:  # Complete after final step
                    bm["completed_runs"] = bm.get("completed_runs", 0) + 1
                    bm["progress"] = 100
            return {"progress": bm["progress"], "completed": bm["progress"] >= 100}
        return {"progress": 0, "completed": True}

    @app.post("/api/benchmark/stop", dependencies=[Depends(authorize)])
    def benchmark_stop():
        """Stop the current benchmark session."""
        manager = app.state.manager
        if hasattr(manager, "active_benchmark") and manager.active_benchmark:
            bm = manager.active_benchmark
            bm["progress"] = 100
            return {"status": "stopped", "message": "Benchmark stopped."}
        return {"status": "no_active_benchmark"}

    @app.get("/api/benchmark/results", dependencies=[Depends(authorize)])
    def benchmark_results():
        """Get benchmark results from the last completed session."""
        manager = app.state.manager
        # In a full implementation, this would persist results to disk
        # For now, return structured placeholder based on configuration
        if hasattr(manager, "active_benchmark") and manager.active_benchmark:
            bm = manager.active_benchmark
            config = bm.get("config", {})
            return {
                "run_id": bm.get("run_id"),
                "mode": config.get("mode", "cpu"),
                "sim_throughput": 0.0,  # Would be measured
                "ppo_throughput": 0.0,  # Would be measured
                "episode_throughput": 0.0,  # Would be measured
                "cpu_utilization": None,
                "ram_used_mb": None,
                "gpu_utilization": None,
                "gpu_vram_used_mb": None,
                "errors": bm.get("errors", []),
                "timestamp": time.time(),
                "hardware_info": {"note": "Measured on backend host"},
            }
        return {"run_id": None, "mode": "cpu", "sim_throughput": 0.0, "ppo_throughput": 0.0, "episode_throughput": 0.0, "cpu_utilization": None, "ram_used_mb": None, "gpu_utilization": None, "gpu_vram_used_mb": None, "errors": [], "timestamp": time.time(), "hardware_info": {"note": "No benchmark has been run"}}

    assets = Path(__file__).parent / "web"
    app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def index():
        return FileResponse(assets / "index.html")

    return app
