"""Crash-safe, immutable checkpoint publication. No automatic retention/deletion."""

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt

from .domain import AppError
from .persistence import atomic_json, sync_directory

CHECKPOINT_ID = re.compile(r"^step-\d{12}-[a-f0-9]{12}$")
REQUIRED = {
    "ppo_agent.json", "application.json", "rng.pt", "ppo_learner/actor_critic.pt",
    "ppo_learner/optimizers.pt", "ppo_learner/misc.json",
    "experience_buffer/experience_buffer.zip", "experience_buffer/trajectory_processor.json",
}


class FileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: StrictInt = Field(ge=0)


class CheckpointManifest(BaseModel):
    """Reject malformed metadata before inventory sorting or any trusted-local load."""

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: StrictInt = Field(ge=1, le=1)
    kind: Literal["rlgym-learn-ppo-v1"]
    id: str = Field(pattern=CHECKPOINT_ID.pattern)
    step: StrictInt = Field(ge=0, le=999_999_999_999)
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    created: FiniteFloat = Field(gt=0)
    files: dict[str, FileRecord]
    metadata: dict


def validated_manifest(value: object) -> dict:
    manifest = CheckpointManifest.model_validate(value).model_dump()
    if int(manifest["id"].split("-")[1]) != manifest["step"]:
        raise ValueError("Checkpoint step differs from its immutable identifier")
    if not REQUIRED <= manifest["files"].keys():
        raise ValueError("Manifest is missing required PPO recovery files")
    for name in manifest["files"]:
        parts = PurePosixPath(name)
        if (parts.is_absolute() or ".." in parts.parts or not parts.parts
                or str(parts) != name or "\\" in name or name == "manifest.json"):
            raise ValueError("Unsafe checkpoint manifest file path")
    return manifest


def file_hash(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


class CheckpointStore:
    def __init__(self, run_dir: Path):
        self.root = run_dir / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[tuple, dict]] = {}

    def path(self, checkpoint_id: str) -> Path:
        if not isinstance(checkpoint_id, str) or not CHECKPOINT_ID.fullmatch(checkpoint_id):
            raise AppError("Invalid checkpoint identifier. Choose an inventory item.", 422)
        path = self.root / checkpoint_id
        if path.is_symlink():
            raise AppError("Symlink checkpoints are not trusted.", 422)
        return path

    def publish(self, step: int, config_hash: str, writer: Callable[[Path], None], metadata: dict) -> dict:
        checkpoint_id = f"step-{step:012d}-{uuid.uuid4().hex[:12]}"
        stage = self.root / f".pending-{uuid.uuid4().hex}"
        stage.mkdir()
        try:
            writer(stage)
            files = {}
            for path in sorted(stage.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Checkpoint writers must not produce symlinks")
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                    files[path.relative_to(stage).as_posix()] = {
                        "sha256": file_hash(path), "bytes": path.stat().st_size,
                    }
            if not REQUIRED <= files.keys():
                raise ValueError(f"Incomplete checkpoint: missing {sorted(REQUIRED - files.keys())}")
            manifest = {
                "schema_version": 1, "kind": "rlgym-learn-ppo-v1", "id": checkpoint_id,
                "step": step, "config_hash": config_hash, "created": time.time(),
                "files": files, "metadata": metadata,
            }
            manifest = validated_manifest(manifest)
            atomic_json(stage / "manifest.json", manifest)
            for path in sorted(stage.rglob("*"), reverse=True):
                if path.is_dir():
                    sync_directory(path)
            sync_directory(stage)
            os.replace(stage, self.path(checkpoint_id))
            sync_directory(self.root)
            # A crash before this pointer update is safe: latest_valid scans complete manifests.
            atomic_json(self.root / "latest.json", {"id": checkpoint_id})
            return manifest
        finally:
            # Only remove our unpublished temporary output, never a valuable checkpoint.
            if stage.exists():
                shutil.rmtree(stage)

    def inspect(self, checkpoint_id: str, *, fresh: bool = False) -> dict:
        try:
            path = self.path(checkpoint_id)
            if not path.is_dir():
                raise ValueError("Checkpoint directory is missing")
            paths = sorted(path.rglob("*"))
            if any(item.is_symlink() for item in paths):
                raise ValueError("Checkpoint contains a symlink")
            signature = tuple((item.relative_to(path).as_posix(), item.stat().st_size,
                               item.stat().st_mtime_ns, item.stat().st_ctime_ns, item.stat().st_ino)
                              for item in paths if item.is_file())
            cached = self._cache.get(checkpoint_id)
            if not fresh and cached and cached[0] == signature:
                return cached[1]
            manifest = validated_manifest(json.loads((path / "manifest.json").read_text()))
            if manifest["id"] != checkpoint_id:
                raise ValueError("Invalid checkpoint manifest identity")
            actual = {item.relative_to(path).as_posix() for item in paths if item.is_file()}
            if actual != set(manifest["files"]) | {"manifest.json"}:
                raise ValueError("Checkpoint files differ from manifest")
            for name, record in manifest["files"].items():
                file = path / name
                if file.resolve().parent != path.resolve() and path.resolve() not in file.resolve().parents:
                    raise ValueError("Manifest path escapes checkpoint directory")
                if file.stat().st_size != record["bytes"] or file_hash(file) != record["sha256"]:
                    raise ValueError(f"Integrity check failed: {name}")
            result = {**manifest, "integrity": "valid", "error": None, "evaluation": None}
            self._cache[checkpoint_id] = (signature, result)
            return result
        except (OSError, ValueError, KeyError, TypeError, AppError) as error:
            return {"id": checkpoint_id, "integrity": "corrupt", "error": str(error), "evaluation": None}

    def inventory(self) -> list[dict]:
        return [self.inspect(item.name) for item in sorted(self.root.iterdir(), reverse=True)
                if CHECKPOINT_ID.fullmatch(item.name)]

    def latest_valid(self, config_hash: str | None = None) -> dict | None:
        candidates = self.inventory()
        valid = [item for item in candidates if item["integrity"] == "valid"
                 and (config_hash is None or item["config_hash"] == config_hash)]
        return max(valid, key=lambda item: (item["step"], item["created"]), default=None)

    def require_valid(self, checkpoint_id: str, config_hash: str | None = None) -> dict:
        item = self.inspect(checkpoint_id, fresh=True)
        if item["integrity"] != "valid":
            raise AppError(f"Checkpoint is not recoverable: {item['error']}. Select an older valid checkpoint.")
        if config_hash is not None and item["config_hash"] != config_hash:
            raise AppError("Checkpoint configuration hash mismatch. Recover using its immutable original config.")
        return item
