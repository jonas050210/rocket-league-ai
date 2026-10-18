"""Atomic checkpoint storage and model registry.

Rules enforced here (safety requirements):
* checkpoints are written to a temporary file and atomically renamed;
* an existing checkpoint is never overwritten — a new id is generated;
* the stable model is never deleted or replaced automatically;
* promotion is an explicit operation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

CHECKPOINT_FORMAT = "rlai-checkpoint-v1"
MANIFEST = "manifest.json"
MODEL_FILE = "model.pt"
STABLE_POINTER = "stable.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")
    return text[:48] or "model"


def new_model_id(step: int, label: str = "") -> str:
    suffix = f"-{_slug(label)}" if label else ""
    return f"m-{_utc_stamp()}-{int(step):010d}{suffix}"


def checkpoint_dir(models_dir: str | Path, model_id: str) -> Path:
    return Path(models_dir) / model_id


def save_checkpoint(
    models_dir: str | Path,
    *,
    model_state: dict,
    policy_config: dict,
    step: int,
    run_id: str | None = None,
    mode: str | None = None,
    training_config: dict | None = None,
    optimizer_state: dict | None = None,
    extra: dict | None = None,
    label: str = "",
) -> dict:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_id = new_model_id(step, label)
    directory = models_dir / model_id
    counter = 1
    while directory.exists():
        model_id = f"{new_model_id(step, label)}-{counter}"
        directory = models_dir / model_id
        counter += 1
    directory.mkdir(parents=True)

    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "model": {k: v.detach().cpu() for k, v in model_state.items()},
        "optimizer": optimizer_state,
        "policy_config": policy_config,
        "training_config": training_config,
        "run_id": run_id,
        "mode": mode,
        "created_at": _utc_now(),
        "extra": extra or {},
    }
    tmp_path = directory / (MODEL_FILE + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, directory / MODEL_FILE)

    manifest = {
        "id": model_id,
        "path": f"{model_id}/{MODEL_FILE}",
        "step": int(step),
        "created_at": payload["created_at"],
        "run_id": run_id,
        "mode": mode,
        "state": "candidate",
        "policy_config": policy_config,
        "size_bytes": (directory / MODEL_FILE).stat().st_size,
        "label": label,
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_checkpoint(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a {CHECKPOINT_FORMAT} checkpoint (got {payload.get('format')!r})"
        )
    return payload


def read_manifest(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / MANIFEST).read_text(encoding="utf-8"))


def write_manifest(model_dir: str | Path, manifest: dict) -> None:
    path = Path(model_dir) / MANIFEST
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _stable_id(models_dir: str | Path) -> str | None:
    pointer = Path(models_dir) / STABLE_POINTER
    if not pointer.exists():
        return None
    try:
        return str(json.loads(pointer.read_text(encoding="utf-8")).get("id"))
    except (json.JSONDecodeError, OSError):
        return None


def list_models(models_dir: str | Path) -> list[dict]:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    stable = _stable_id(models_dir)
    models: list[dict] = []
    for directory in models_dir.iterdir():
        if not directory.is_dir():
            continue
        manifest_path = directory / MANIFEST
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        manifest["stable"] = manifest.get("id") == stable
        models.append(manifest)
    models.sort(key=lambda m: (m.get("created_at", ""), m.get("step", 0)), reverse=True)
    return models


def get_model(models_dir: str | Path, model_id: str) -> dict:
    directory = checkpoint_dir(models_dir, model_id)
    if not (directory / MANIFEST).exists():
        raise KeyError(f"unknown model {model_id!r}")
    manifest = read_manifest(directory)
    manifest["stable"] = _stable_id(models_dir) == model_id
    manifest["absolute_path"] = str(directory / MODEL_FILE)
    return manifest


def set_model_state(models_dir: str | Path, model_id: str, state: str) -> dict:
    if state not in {"candidate", "stable", "archived", "rejected"}:
        raise ValueError("state must be candidate, stable, archived or rejected")
    directory = checkpoint_dir(models_dir, model_id)
    manifest = get_model(models_dir, model_id)
    manifest.pop("absolute_path", None)
    manifest.pop("stable", None)
    manifest["state"] = state
    write_manifest(directory, manifest)
    return manifest


def promote_model(models_dir: str | Path, model_id: str) -> dict:
    models_dir = Path(models_dir)
    get_model(models_dir, model_id)  # raises for unknown ids before writing anything
    previous = stable_model(models_dir)
    pointer = {
        "id": model_id,
        "promoted_at": _utc_now(),
        "previous": previous.get("id") if previous else None,
    }
    tmp = models_dir / (STABLE_POINTER + ".tmp")
    tmp.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    os.replace(tmp, models_dir / STABLE_POINTER)
    set_model_state(models_dir, model_id, "stable")
    if previous and previous.get("id") != model_id:
        try:
            set_model_state(models_dir, previous["id"], "archived")
        except KeyError:
            pass
    return get_model(models_dir, model_id)


def stable_model(models_dir: str | Path) -> dict | None:
    pointer = Path(models_dir) / STABLE_POINTER
    if not pointer.exists():
        return None
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return get_model(models_dir, data["id"])
    except KeyError:
        return None


def delete_model(models_dir: str | Path, model_id: str) -> None:
    models_dir = Path(models_dir)
    manifest = get_model(models_dir, model_id)
    if manifest.get("stable"):
        raise ValueError("refusing to delete the stable model")
    remaining = [m for m in list_models(models_dir) if m["id"] != model_id]
    if not remaining:
        raise ValueError("refusing to delete the last remaining checkpoint")
    shutil.rmtree(checkpoint_dir(models_dir, model_id))


def prune_candidates(models_dir: str | Path, keep: int = 8, run_id: str | None = None) -> list[str]:
    """Delete oldest non-stable candidates beyond ``keep``.

    When ``run_id`` is given only checkpoints from that run are considered,
    so starting a new run can never delete another run's history.  Never
    touches the stable model, the newest checkpoint, or the last remaining
    checkpoint.  Returns deleted ids.
    """
    models = [
        m
        for m in list_models(models_dir)
        if not m.get("stable") and (run_id is None or m.get("run_id") == run_id)
    ]
    if len(models) <= keep:
        return []
    deleted: list[str] = []
    for manifest in models[keep:]:
        if len(list_models(models_dir)) <= 1:
            break
        try:
            delete_model(models_dir, manifest["id"])
            deleted.append(manifest["id"])
        except ValueError:
            continue
    return deleted


def resolve_model_path(models_dir: str | Path, reference: str) -> Path:
    path = Path(reference)
    if path.exists():
        return path
    candidate = checkpoint_dir(models_dir, reference) / MODEL_FILE
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"model {reference!r} not found")


def export_weights(models_dir: str | Path, model_id: str, output: str | Path) -> Path:
    """Export a plain TorchScript module for deployment."""
    from .policy import PolicyConfig, TorchPolicy

    manifest = get_model(models_dir, model_id)
    payload = load_checkpoint(Path(models_dir) / manifest["path"])
    config = PolicyConfig.from_dict(payload.get("policy_config"))
    policy = TorchPolicy.from_config(config)
    policy.model.load_state_dict(payload["model"])
    policy.model.eval()
    scripted = torch.jit.trace(policy.model, torch.zeros(1, config.obs_size))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    return output
