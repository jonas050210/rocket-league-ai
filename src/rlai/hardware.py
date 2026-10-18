"""Hardware detection, dependency diagnostics and throughput benchmark."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
import time

import numpy as np
import psutil

from .config import data_dir, project_root

REQUIRED = [
    "torch",
    "rlgym-rocket-league",
    "rocketsim",
    "numpy",
    "fastapi",
    "uvicorn",
    "pydantic",
    "psutil",
]
OPTIONAL = ["rlbot", "rlbot_flatbuffers"]


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def system_info() -> dict:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(data_dir().anchor or "/")
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "cpu_percent": psutil.cpu_percent(interval=0.15),
        "memory_total_gb": round(memory.total / 1024**3, 2),
        "memory_available_gb": round(memory.available / 1024**3, 2),
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "data_dir": str(data_dir()),
        "project_root": str(project_root()),
        "is_windows": platform.system() == "Windows",
        "packages": {name: _version(name) for name in REQUIRED + OPTIONAL},
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "mps_available": bool(
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            ),
        }
    except Exception as exc:  # pragma: no cover - torch missing
        info["torch"] = {"error": str(exc)}
    try:
        import RocketSim  # noqa: F401

        info["rocketsim"] = {"import_ok": True}
    except Exception as exc:
        info["rocketsim"] = {"import_ok": False, "error": str(exc)}
    try:
        import rlbot  # noqa: F401

        info["rlbot"] = {"import_ok": True, "version": _version("rlbot")}
    except Exception as exc:
        info["rlbot"] = {"import_ok": False, "error": str(exc)}
    return info


def diagnostics() -> dict:
    info = system_info()
    checks = []

    def add(name: str, ok: bool, detail: str = "", required: bool = True):
        checks.append({"name": name, "ok": ok, "detail": detail, "required": required})

    for package in REQUIRED:
        version = _version(package)
        add(f"package:{package}", version is not None, version or "not installed")
    for package in OPTIONAL:
        version = _version(package)
        add(f"package:{package}", version is not None, version or "not installed", required=False)
    add(
        "gpu",
        bool(info.get("torch", {}).get("cuda_available")),
        "CUDA available" if info.get("torch", {}).get("cuda_available") else "CPU-only (supported)",
        required=False,
    )
    add(
        "simulator",
        info.get("rocketsim", {}).get("import_ok", False),
        info.get("rocketsim", {}).get("error", ""),
    )
    try:
        test = data_dir() / "tmp" / ".write-test"
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("ok", encoding="utf-8")
        test.unlink()
        add("data directory writable", True, str(data_dir()))
    except Exception as exc:
        add("data directory writable", False, str(exc))
    add(
        "rlbot",
        info.get("rlbot", {}).get("import_ok", False),
        "required for live matches",
        required=False,
    )
    return {"info": info, "checks": checks, "ok": all(c["ok"] for c in checks if c["required"])}


def _benchmark_worker(remote, mode: str, seconds: float, worker_index: int) -> None:
    """Run one bounded simulation loop in its own process."""
    try:
        import torch

        torch.set_num_threads(1)
        from .envs import MatchEnv
        from .modes import get_mode
        from .opponents import IdleOpponent

        spec = get_mode(mode)
        env = MatchEnv(spec, IdleOpponent(), seed=worker_index, episode_seconds=30)
        env.reset()
        rng = np.random.default_rng(worker_index)
        remote.send({"type": "ready", "worker": worker_index})
        message = remote.recv()
        duration = float(message.get("seconds", seconds))
        steps = 0
        ticks = 0
        started = time.perf_counter()
        while time.perf_counter() - started < duration:
            actions = {a: int(rng.integers(90)) for a in env.policy_agents}
            result = env.step(actions)
            steps += 1
            ticks += env.tick_skip
            if result.done or result.truncated:
                env.reset()
        elapsed = time.perf_counter() - started
        remote.send(
            {
                "type": "result",
                "worker": worker_index,
                "steps": steps,
                "ticks": ticks,
                "seconds": elapsed,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        try:
            import traceback

            remote.send(
                {"type": "error", "error": str(exc), "traceback": traceback.format_exc()}
            )
        except Exception:
            pass
    finally:
        try:
            remote.close()
        except Exception:
            pass


def _simulation_loop(mode: str, seconds: float, worker_index: int) -> dict:
    """In-process simulation loop (single worker or parallel fallback)."""
    import torch

    torch.set_num_threads(1)
    from .envs import MatchEnv
    from .modes import get_mode
    from .opponents import IdleOpponent

    spec = get_mode(mode)
    env = MatchEnv(spec, IdleOpponent(), seed=worker_index, episode_seconds=30)
    env.reset()
    rng = np.random.default_rng(worker_index)
    steps = 0
    ticks = 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        actions = {a: int(rng.integers(90)) for a in env.policy_agents}
        result = env.step(actions)
        steps += 1
        ticks += env.tick_skip
        if result.done or result.truncated:
            env.reset()
    return {
        "type": "result",
        "worker": worker_index,
        "steps": steps,
        "ticks": ticks,
        "seconds": time.perf_counter() - started,
    }


def benchmark_inference(
    *,
    hidden: tuple[int, ...] = (256, 256),
    obs_version: int | None = None,
    batch: int = 8,
    seconds: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Measure policy inference cost on its own.

    Simulation throughput says nothing about how expensive a decision is, and
    the two are the knobs that decide how many workers a machine can carry.
    """
    import torch

    from .obs import OBS_VERSION, obs_size
    from .policy import PolicyConfig, TorchPolicy

    version = OBS_VERSION if obs_version is None else int(obs_version)
    config = PolicyConfig(obs_size=obs_size(version), hidden=tuple(hidden))
    policy = TorchPolicy.from_config(config, device=device)
    rng = np.random.default_rng(0)
    observations = {
        f"agent-{index}": rng.normal(size=config.obs_size).astype(np.float32)
        for index in range(max(1, int(batch)))
    }
    seconds = float(max(0.1, min(seconds, 10.0)))
    # warm-up: first call allocates, and on CUDA it also loads kernels
    for _ in range(3):
        policy.act(observations, deterministic=True)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    decisions = 0
    agents = len(observations)
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        policy.act(observations, deterministic=True)
        decisions += 1
    elapsed = time.perf_counter() - started
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    parameters = sum(p.numel() for p in policy.model.parameters())
    return {
        "device": device,
        "obs_version": version,
        "obs_size": config.obs_size,
        "hidden": list(hidden),
        "parameters": int(parameters),
        "batch": agents,
        "decisions": decisions,
        "agent_decisions": decisions * agents,
        "seconds": elapsed,
        "decisions_per_s": decisions / elapsed if elapsed else 0.0,
        "agent_decisions_per_s": decisions * agents / elapsed if elapsed else 0.0,
        "milliseconds_per_decision": 1000.0 * elapsed / decisions if decisions else 0.0,
    }


def benchmark(
    mode: str = "soccar-1v1",
    seconds: float = 3.0,
    workers: int = 1,
    *,
    parallel: bool = True,
    include_inference: bool = True,
) -> dict:
    """Measure real simulation throughput (bounded, no training).

    ``workers > 1`` runs one *process* per worker and reports the wall-clock
    time of the measured phase.  The previous implementation ran the workers
    one after another in this process and then divided the summed steps by the
    longest single loop, which reported roughly ``workers`` times the
    throughput the machine actually delivers.  ``sequential_seconds`` and
    ``speedup`` make the difference visible instead of hiding it.
    """
    import multiprocessing as mp

    seconds = float(max(0.5, min(seconds, 10.0)))
    worker_count = int(max(1, min(workers, 8)))
    results: list[dict] = []
    used_parallel = False
    error: str | None = None

    phase_started = time.perf_counter()
    if parallel and worker_count > 1:
        try:
            ctx = mp.get_context("spawn")
            pipes = []
            processes = []
            for index in range(worker_count):
                parent, child = ctx.Pipe()
                process = ctx.Process(
                    target=_benchmark_worker,
                    args=(child, mode, seconds, index),
                    daemon=True,
                    name=f"rlai-bench-{index}",
                )
                process.start()
                pipes.append(parent)
                processes.append(process)
            # start-up (interpreter + torch import) is not part of the
            # measurement; only the collecting phase is timed.
            for index, parent in enumerate(pipes):
                if not parent.poll(180):
                    raise TimeoutError(f"benchmark worker {index} did not start")
                message = parent.recv()
                if message.get("type") != "ready":
                    raise RuntimeError(f"benchmark worker {index} failed: {message}")
            measured_started = time.perf_counter()
            for parent in pipes:
                parent.send({"cmd": "go", "seconds": seconds})
            for index, parent in enumerate(pipes):
                if not parent.poll(180):
                    raise TimeoutError(f"benchmark worker {index} timed out")
                message = parent.recv()
                if message.get("type") != "result":
                    raise RuntimeError(
                        f"benchmark worker {index} error: {message.get('error')}"
                    )
                results.append(message)
            wall_seconds = time.perf_counter() - measured_started
            used_parallel = True
            for parent in pipes:
                try:
                    parent.close()
                except Exception:
                    pass
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
        except Exception as exc:  # pragma: no cover - environment specific
            error = f"{type(exc).__name__}: {exc}"
            results = []
            used_parallel = False
    if not results:
        results = [_simulation_loop(mode, seconds, index) for index in range(worker_count)]
        # Sequential loops run one after another, so their own timings add up
        # to the wall clock of the measured phase.  Environment construction
        # is deliberately excluded on both paths: it is a fixed start-up cost,
        # not throughput.
        wall_seconds = sum(float(item["seconds"]) for item in results)

    total_steps = sum(int(item["steps"]) for item in results)
    total_ticks = sum(int(item["ticks"]) for item in results)
    sequential_seconds = sum(float(item["seconds"]) for item in results)
    # The reported rate must be reproducible from the reported wall time.
    measured_seconds = max(wall_seconds, 1e-9)
    single_worker_rate = (
        sum(
            float(item["steps"]) / max(float(item["seconds"]), 1e-9) for item in results
        )
        / len(results)
    )
    payload = {
        "mode": mode,
        "workers": len(results),
        "parallel": used_parallel,
        "env_steps_per_s": total_steps / measured_seconds,
        "sim_ticks_per_s": total_ticks / measured_seconds,
        "seconds": measured_seconds,
        "wall_seconds": measured_seconds,
        "sequential_seconds": sequential_seconds,
        "requested_seconds": seconds,
        "single_worker_steps_per_s": single_worker_rate,
        "speedup": (
            (total_steps / measured_seconds) / single_worker_rate if single_worker_rate else 0.0
        ),
        "total_steps": total_steps,
        "total_ticks": total_ticks,
        "per_worker": [
            {
                "worker": item.get("worker", index),
                "steps": int(item["steps"]),
                "ticks": int(item["ticks"]),
                "seconds": float(item["seconds"]),
                "steps_per_s": float(item["steps"]) / max(float(item["seconds"]), 1e-9),
            }
            for index, item in enumerate(results)
        ],
        "measurement": (
            "wall clock of the collecting phase; one process per worker"
            if used_parallel
            else "wall clock of sequential in-process loops"
        ),
    }
    if error:
        payload["parallel_error"] = error
    if include_inference:
        payload["inference"] = benchmark_inference(seconds=min(1.0, seconds))
    payload["total_seconds"] = time.perf_counter() - phase_started
    return payload
