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


def benchmark(mode: str = "soccer-1v1", seconds: float = 3.0, workers: int = 1) -> dict:
    """Measure real simulation throughput (bounded, no training)."""
    from .envs import MatchEnv
    from .modes import get_mode
    from .opponents import IdleOpponent

    spec = get_mode(mode)
    seconds = float(max(0.5, min(seconds, 10.0)))
    results = []
    for worker in range(max(1, min(workers, 8))):
        env = MatchEnv(spec, IdleOpponent(), seed=worker, episode_seconds=30)
        env.reset()
        rng = np.random.default_rng(worker)
        steps = 0
        ticks = 0
        started = time.perf_counter()
        while time.perf_counter() - started < seconds:
            actions = {a: int(rng.integers(90)) for a in env.blue_agents}
            result = env.step(actions)
            steps += 1
            ticks += env.tick_skip
            if result.done or result.truncated:
                env.reset()
        elapsed = time.perf_counter() - started
        results.append({"steps": steps, "ticks": ticks, "seconds": elapsed})
    total_steps = sum(r["steps"] for r in results)
    total_ticks = sum(r["ticks"] for r in results)
    total_seconds = max(r["seconds"] for r in results)
    return {
        "mode": mode,
        "workers": len(results),
        "env_steps_per_s": total_steps / total_seconds,
        "sim_ticks_per_s": total_ticks / total_seconds,
        "seconds": total_seconds,
        "per_worker": results,
    }
