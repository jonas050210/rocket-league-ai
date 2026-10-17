"""Reproducibility metadata, without credentials or machine-specific absolute paths."""

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from .domain import AppError

TRAINING_PACKAGES = (
    "rlgym-api", "rlgym-rocket-league", "rocketsim", "rlgym-learn", "rlgym-learn-algos", "torch", "numpy",
)


def package_versions(names=TRAINING_PACKAGES) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def training_source_hash() -> str:
    """Identify the actual training/config/storage implementation, including dirty source."""
    package = Path(__file__).parent
    paths = sorted((package / "training").rglob("*.py")) + [
        package / "config.py", package / "checkpoints.py", package / "persistence.py",
    ]
    checksum = hashlib.sha256()
    for path in paths:
        checksum.update(path.relative_to(package).as_posix().encode() + b"\0")
        checksum.update(path.read_bytes() + b"\0")
    return checksum.hexdigest()


def require_recovery_compatible(metadata: dict) -> None:
    if not isinstance(metadata, dict):
        raise AppError("Experiment metadata is malformed. Restore trusted experiment evidence before recovery.")
    if metadata.get("packages") != package_versions():
        raise AppError("Dependency versions differ. Restore the original locked environment before recovery.")
    if metadata.get("training_source_sha256") != training_source_hash():
        raise AppError("Training/config/storage source differs or is unrecorded. Restore the recorded implementation before recovery; changed strategies require a separately approved new experiment.")
    if str(metadata.get("python", "")).split(".")[:2] != sys.version.split()[0].split(".")[:2]:
        raise AppError("Python major/minor version differs. Restore the original interpreter profile before recovery.")


def experiment_metadata() -> dict:
    root = repository_root()
    git = {"commit": None, "dirty": None}
    try:
        git["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
        git["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, timeout=2,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    lock = root / "uv.lock"
    return {
        "schema_version": 1,
        "git": git,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        "training_source_sha256": training_source_hash(),
        "packages": package_versions(),
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "architecture": platform.machine(),
        "resume_semantics": "update-boundary learner/optimizer/buffer/RNG restore; environments reset",
        "evaluation": "frozen-checkpoint vs scripted baselines (evaluation.py); self-play training reward is not win rate",
    }
