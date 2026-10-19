"""Resource-aware limits for the training process.

More workers are not automatically faster.  A RocketSim arena costs memory
and CPU, and on a small machine extra workers make every worker slower
instead of adding throughput.  This module turns the machine's actual
resources into a concrete worker budget and explains the decision, so the
GUI can warn *before* a run starts and the trainer can refuse a clearly
impossible configuration.

The limits are advisory by default: the trainer clamps to the recommendation
and records a warning, and the API reports the same numbers in
``/api/config/validate``.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

#: Rough per-worker cost of one RocketSim arena plus the Python/torch runtime.
#: Measured on the reference machine (see docs/TESTING.md); deliberately
#: generous so the guard fires before the machine starts swapping.
ESTIMATED_WORKER_MEMORY_MB = 700.0

#: Below this much free memory a run is not started at all.
MIN_FREE_MEMORY_MB = 600.0

#: Never recommend more workers than this, even on a big server.
MAX_RECOMMENDED_WORKERS = 16


@dataclass
class ResourcePlan:
    """The worker budget for one machine, with the reasons behind it."""

    recommended_workers: int
    requested_workers: int
    cpu_count: int
    memory_available_mb: float
    memory_total_mb: float
    disk_free_gb: float
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def clamped(self) -> bool:
        return self.recommended_workers < self.requested_workers

    def to_dict(self) -> dict:
        return {
            "recommended_workers": self.recommended_workers,
            "requested_workers": self.requested_workers,
            "cpu_count": self.cpu_count,
            "memory_available_mb": round(self.memory_available_mb, 1),
            "memory_total_mb": round(self.memory_total_mb, 1),
            "disk_free_gb": round(self.disk_free_gb, 2),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "clamped": self.clamped,
        }


def _memory_mb() -> tuple[float, float]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return memory.available / 1024**2, memory.total / 1024**2
    except Exception:  # pragma: no cover - psutil missing
        try:
            page = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            return page / 1024**2, total / 1024**2
        except (ValueError, OSError, AttributeError):
            return 4096.0, 8192.0


def _disk_free_gb() -> float:
    try:
        from .config import data_dir

        usage = shutil.disk_usage(data_dir().anchor or "/")
        return usage.free / 1024**3
    except Exception:  # pragma: no cover - defensive
        return 0.0


def plan_resources(
    requested_workers: int,
    *,
    cpu_count: int | None = None,
    memory_available_mb: float | None = None,
    memory_total_mb: float | None = None,
    disk_free_gb: float | None = None,
) -> ResourcePlan:
    """Compute a safe worker budget for ``requested_workers``.

    The CPU budget is the number of *logical* cores: every worker needs a
    core for the simulation and the trainer needs one for the learner.  The
    memory budget is derived from the free memory and the estimated cost of
    one worker.  The smaller of the two wins, clamped to at least one.
    """
    cores = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    available, total = _memory_mb()
    if memory_available_mb is not None:
        available = float(memory_available_mb)
    if memory_total_mb is not None:
        total = float(memory_total_mb)
    free_disk = float(disk_free_gb if disk_free_gb is not None else _disk_free_gb())

    requested = max(1, int(requested_workers))

    # One core stays for the trainer process and the API/GUI.
    cpu_budget = max(1, cores - 1) if cores > 1 else 1
    memory_budget = int(available // ESTIMATED_WORKER_MEMORY_MB)
    memory_budget = max(1, memory_budget)

    recommended = max(1, min(requested, cpu_budget, memory_budget, MAX_RECOMMENDED_WORKERS))

    warnings: list[str] = []
    notes: list[str] = []

    if requested > cores:
        warnings.append(
            f"{requested} workers on {cores} logical cores will oversubscribe the CPU; "
            f"each worker will run slower instead of adding throughput."
        )
    if requested * ESTIMATED_WORKER_MEMORY_MB > available:
        warnings.append(
            f"{requested} workers need roughly "
            f"{requested * ESTIMATED_WORKER_MEMORY_MB / 1024:.1f} GB but only "
            f"{available / 1024:.1f} GB are free; the machine will start swapping."
        )
    if available < MIN_FREE_MEMORY_MB:
        warnings.append(
            f"only {available / 1024:.1f} GB of memory are free - training is not safe "
            "until other programs release memory."
        )
    if free_disk and free_disk < 2.0:
        warnings.append(
            f"only {free_disk:.1f} GB of disk space are free; checkpoints and the "
            "database need room to grow."
        )
    if recommended < requested:
        notes.append(
            f"recommended worker count on this machine: {recommended} "
            f"(CPU budget {cpu_budget}, memory budget {memory_budget})."
        )
    if cores <= 2:
        notes.append(
            "this machine has very few cores; run one worker and expect a low "
            "step rate - a GPU does not speed up the simulation."
        )
    if not warnings:
        notes.append(f"{recommended} workers fit into the available resources.")

    return ResourcePlan(
        recommended_workers=recommended,
        requested_workers=requested,
        cpu_count=cores,
        memory_available_mb=available,
        memory_total_mb=total,
        disk_free_gb=free_disk,
        warnings=warnings,
        notes=notes,
    )


def clamp_workers(
    requested_workers: int,
    *,
    enabled: bool = True,
    **kwargs,
) -> tuple[int, ResourcePlan]:
    """Return the worker count to actually use, plus the plan behind it."""
    plan = plan_resources(requested_workers, **kwargs)
    if not enabled:
        return max(1, int(requested_workers)), plan
    return plan.recommended_workers, plan
