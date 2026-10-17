"""Same-origin HTTP interface for the web control room."""

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
from .benchmark import BenchmarkSpec
from .config import TrainingConfig
from .domain import AppError
from .evaluation import EvaluationSpec, LeagueSpec, compare_evaluations
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


class DirectStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preset: str | None = None
    preset_id: str | None = None
    max_steps: int | None = Field(default=None, ge=128, le=1_000_000_000)
    duration_seconds: float | None = Field(default=None, gt=0, le=604800)
    milestone_id: str | None = None
    continue_from_latest: bool = True


class LiveMatchStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["human_vs_ai", "ai_vs_ai", "sim_live"] = "human_vs_ai"
    scenario: Literal["kickoff-v1", "mixed-scenarios-v1"] = "mixed-scenarios-v1"
    team_size: Literal[1, 2] = 1
    boost_mode: Literal["standard", "unlimited", "starved"] = "standard"
    arena_map: Literal["DFHStadium", "ChampionsField", "Mannfield_Night", "NeonFields", "UtopiaColiseum"] = "DFHStadium"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    launch_client: bool = True
    run_id: str | None = None
    checkpoint_id: str | None = None


class LiveMatchControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "toggle_pause",
        "pause",
        "resume",
        "reset_kickoff",
        "next_game",
        "set_speed",
        "set_scenario",
        "set_boost",
        "set_map",
        "human_input",
    ]
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    scenario: Literal["kickoff-v1", "mixed-scenarios-v1"] | None = None
    boost_mode: Literal["standard", "unlimited", "starved"] | None = None
    arena_map: Literal["DFHStadium", "ChampionsField", "Mannfield_Night", "NeonFields", "UtopiaColiseum"] | None = None
    human_controls: dict[str, float] | None = None


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
            raise AppError("Connect using this workspace's local access key.", 401)

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

    @app.post("/api/runs/start-direct", dependencies=[Depends(authorize)])
    def start_direct(body: DirectStartRequest):
        return app.state.manager.start_direct(
            preset=body.preset or body.preset_id,
            max_steps=body.max_steps,
            duration_seconds=body.duration_seconds,
            milestone_id=body.milestone_id,
            continue_from_latest=body.continue_from_latest,
        )

    @app.post("/api/runs/{run_id}/control", dependencies=[Depends(authorize)])
    def control(run_id: str, body: ControlRequest):
        return app.state.manager.control(run_id, body.command)

    @app.get("/api/checkpoints/{run_id}/{checkpoint_id}/rlbot", dependencies=[Depends(authorize)])
    def export_rlbot(run_id: str, checkpoint_id: str):
        return app.state.manager.export_rlbot_bundle(run_id, checkpoint_id)

    @app.get("/api/report", dependencies=[Depends(authorize)])
    def report():
        return {"schema_version": 1, "scope": "measured foundation status plus benchmark/evaluation evidence",
                "snapshot": app.state.manager.snapshot(),
                "events": app.state.manager.store.events(0, 1000),
                "limitations": ["Replay viewer is a recorded top-down schematic, not a 3D renderer",
                                "Evaluation covers scripted baselines and frozen checkpoints",
                                "Recovery resets simulator episodes", "CUDA requires target-host verification"]}

    @app.post("/api/benchmarks", dependencies=[Depends(authorize)])
    def start_benchmark(body: BenchmarkSpec):
        return app.state.manager.start_benchmark(body)

    @app.post("/api/benchmarks/auto", dependencies=[Depends(authorize)])
    def start_auto_benchmark():
        return app.state.manager.start_auto_benchmark()

    @app.get("/api/benchmarks", dependencies=[Depends(authorize)])
    def benchmarks():
        manager = app.state.manager
        return {"sessions": manager.store.benchmarks(),
                "latest": manager._benchmark_snapshot()}

    @app.post("/api/benchmarks/stop", dependencies=[Depends(authorize)])
    def stop_benchmark():
        return app.state.manager.stop_benchmark()

    @app.post("/api/evaluations", dependencies=[Depends(authorize)])
    def start_evaluation(body: EvaluationSpec):
        return app.state.manager.start_evaluation(body)

    @app.get("/api/evaluations", dependencies=[Depends(authorize)])
    def evaluations():
        manager = app.state.manager
        return {"evaluations": manager.store.evaluations(),
                "comparison": compare_evaluations(manager.store.evaluations())}

    @app.post("/api/evaluations/{evaluation_id}/stop", dependencies=[Depends(authorize)])
    def stop_evaluation(evaluation_id: str):
        return app.state.manager.stop_evaluation(evaluation_id)

    @app.get("/api/evaluations/{evaluation_id}/trajectory", dependencies=[Depends(authorize)])
    def trajectory(evaluation_id: str):
        return app.state.manager.trajectory(evaluation_id)

    @app.get("/api/leagues", dependencies=[Depends(authorize)])
    def leagues():
        manager = app.state.manager
        return {"leagues": manager.store.leagues()}

    @app.post("/api/leagues", dependencies=[Depends(authorize)])
    def start_league(body: LeagueSpec):
        return app.state.manager.start_league(body)

    @app.post("/api/leagues/{league_id}/stop", dependencies=[Depends(authorize)])
    def stop_league(league_id: str):
        return app.state.manager.stop_league(league_id)

    @app.get("/api/live-match", dependencies=[Depends(authorize)])
    def get_live_match():
        return app.state.manager.live_match.snapshot()

    @app.post("/api/live-match/start", dependencies=[Depends(authorize)])
    def start_live_match(body: LiveMatchStartRequest | None = None):
        req = body or LiveMatchStartRequest()
        return app.state.manager.live_match.start(
            mode=req.mode,
            scenario=req.scenario,
            team_size=req.team_size,
            boost_mode=req.boost_mode,
            arena_map=req.arena_map,
            speed=req.speed,
            launch_client=req.launch_client,
            run_id=req.run_id,
            checkpoint_id=req.checkpoint_id,
        )

    @app.post("/api/live-match/control", dependencies=[Depends(authorize)])
    def control_live_match(body: LiveMatchControlRequest):
        return app.state.manager.live_match.control(
            body.action,
            speed=body.speed,
            scenario=body.scenario,
            boost_mode=body.boost_mode,
            arena_map=body.arena_map,
            human_controls=body.human_controls,
        )

    @app.post("/api/live-match/stop", dependencies=[Depends(authorize)])
    def stop_live_match():
        return app.state.manager.live_match.stop()

    assets = Path(__file__).parent / "web"
    app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def index():
        return FileResponse(assets / "index.html")

    return app
