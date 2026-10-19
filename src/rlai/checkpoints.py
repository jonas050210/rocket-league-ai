"""Atomic checkpoint storage, validation and model registry.

Rules enforced here (safety requirements):
* checkpoints are written to a temporary file and atomically renamed;
* an existing checkpoint is never overwritten — a new id is generated;
* the latest completed save is recorded in ``latest.json`` (metadata, not a
  filename convention), so a resume never has to trust a timestamp in a
  directory name;
* retention deletes only complete, older checkpoints and keeps the newest
  ``keep`` ones, the latest pointer target, every best checkpoint and the
  newest final/interrupted checkpoint; before anything is deleted it checks
  that a loadable checkpoint remains;
* the stable model is never deleted or replaced automatically;
* promotion is an explicit operation;
* loading uses ``weights_only`` by default, so a hostile file cannot execute
  code inside the training process;
* model references coming from the network are confined to the models
  directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .policy import clone_state_dict

CHECKPOINT_FORMAT = "rlai-checkpoint-v1"
#: Bumped when new optional fields are added.  Older readers accept newer
#: files only if the format string matches; newer readers accept version 1.
CHECKPOINT_VERSION = 2
MANIFEST = "manifest.json"
MODEL_FILE = "model.pt"
STABLE_POINTER = "stable.json"
BEST_POINTER = "best.json"
#: Updated after every successful save.  This is the authoritative "latest
#: completed checkpoint" record; the id still contains a timestamp, but resume
#: selection uses this metadata (and the step in the manifest), never the
#: lexical order of directory names.
LATEST_POINTER = "latest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")
    return text[:48] or "model"


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync so a rename survives a power loss."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + ``os.replace``)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _timing_value(timing: Any, key: str) -> float | None:
    """Read one duration out of the checkpoint timing block.

    Returns ``None`` when the value was never measured, so an old
    checkpoint stays loadable and the GUI can report "unknown".
    """
    if not isinstance(timing, dict):
        return None
    value = timing.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        "version": CHECKPOINT_VERSION,
        "step": int(step),
        # A real copy: ``detach().cpu()`` aliases the live parameters when the
        # model is already on the CPU, and the file must never depend on what
        # the learner does between building the payload and writing it.
        "model": clone_state_dict(model_state),
        "optimizer": optimizer_state,
        "policy_config": policy_config,
        "training_config": training_config,
        "run_id": run_id,
        "mode": mode,
        "created_at": _utc_now(),
        "extra": extra or {},
    }
    manifest = {
        "id": model_id,
        "path": f"{model_id}/{MODEL_FILE}",
        "step": int(step),
        "created_at": payload["created_at"],
        "run_id": run_id,
        "mode": mode,
        "state": "candidate",
        "policy_config": policy_config,
        "size_bytes": 0,
        "label": label,
        # cheap metadata for the GUI and for resume planning; older manifests
        # simply do not have these keys.
        "checkpoint_version": CHECKPOINT_VERSION,
        "has_optimizer": bool(optimizer_state),
        "obs_version": (extra or {}).get("obs_version"),
        "has_rng": isinstance((extra or {}).get("rng"), dict),
        "parent_run": (extra or {}).get("parent_run"),
        # training time at the moment the checkpoint was written.  Absent for
        # checkpoints written before durations were tracked - the GUI then
        # shows "unknown" instead of a made-up number.
        "duration_s": _timing_value((extra or {}).get("timing"), "accumulated_s"),
        "session_duration_s": _timing_value((extra or {}).get("timing"), "session_s"),
    }
    model_path = directory / MODEL_FILE
    tmp_path = directory / (MODEL_FILE + ".tmp")
    try:
        # Write the model first, fsync it, then atomically publish it.  The
        # manifest is published only afterwards and is itself atomic, so a
        # reader never sees a half-written checkpoint and a failure in this
        # new directory cannot touch a previously valid one.
        with open(tmp_path, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, model_path)
        _fsync_dir(directory)
        manifest["size_bytes"] = model_path.stat().st_size
        write_manifest(directory, manifest)
    except BaseException:
        # This directory was created by this call and is not referenced by any
        # pointer yet.  Removing the partial attempt keeps failed saves from
        # accumulating while leaving every earlier checkpoint untouched.
        shutil.rmtree(directory, ignore_errors=True)
        raise
    try:
        _write_pointer(
            models_dir,
            LATEST_POINTER,
            {
                "id": model_id,
                "path": f"{model_id}/{MODEL_FILE}",
                "step": int(step),
                "run_id": run_id,
                "mode": mode,
                "label": label,
                "created_at": manifest["created_at"],
            },
        )
    except OSError:
        # The checkpoint itself is complete and valid; the pointer is an
        # optimization.  Resume still finds it through the manifests.
        pass
    return manifest


def load_checkpoint(
    path: str | Path,
    *,
    weights_only: bool = True,
    map_location: str = "cpu",
) -> dict:
    """Load a checkpoint payload.

    ``weights_only=True`` is the default and refuses files that would execute
    arbitrary code through ``pickle``.  Every checkpoint written by this
    project only contains tensors, numbers, strings, lists and dicts, so the
    safe loader accepts them.  ``weights_only=False`` remains available for
    hand-built legacy files, but callers must opt in explicitly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=weights_only)
    except Exception as exc:
        if weights_only:
            raise ValueError(
                f"{path} could not be loaded safely ({type(exc).__name__}: {exc}). "
                "If this is a trusted legacy checkpoint, load it with weights_only=False."
            ) from exc
        raise
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a checkpoint dictionary")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a {CHECKPOINT_FORMAT} checkpoint (got {payload.get('format')!r})"
        )
    version = int(payload.get("version", 1))
    if version > CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} was written by a newer version of rlai "
            f"(checkpoint version {version}, supported up to {CHECKPOINT_VERSION})"
        )
    return payload


@dataclass
class CheckpointValidation:
    """Result of validating one checkpoint for a specific purpose."""

    path: str
    ok: bool
    reasons: list[str] = field(default_factory=list)
    step: int | None = None
    run_id: str | None = None
    mode: str | None = None
    label: str = ""
    created_at: str | None = None
    obs_version: int | None = None
    policy_config: dict | None = None
    training_config: dict | None = None
    extra: dict | None = None
    has_model: bool = False
    has_optimizer: bool = False
    has_rng: bool = False
    has_curriculum: bool = False
    has_league: bool = False
    has_timing: bool = False
    duration_s: float | None = None
    session_duration_s: float | None = None
    size_bytes: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "ok": self.ok,
            "reasons": list(self.reasons),
            "step": self.step,
            "run_id": self.run_id,
            "mode": self.mode,
            "label": self.label,
            "created_at": self.created_at,
            "obs_version": self.obs_version,
            "policy_config": self.policy_config,
            "training_config": self.training_config,
            "extra": self.extra,
            "has_model": self.has_model,
            "has_optimizer": self.has_optimizer,
            "has_rng": self.has_rng,
            "has_curriculum": self.has_curriculum,
            "has_league": self.has_league,
            "has_timing": self.has_timing,
            "duration_s": self.duration_s,
            "session_duration_s": self.session_duration_s,
            "size_bytes": self.size_bytes,
            "error": self.error,
        }


def validate_checkpoint(
    path: str | Path,
    *,
    policy_config: dict | None = None,
    mode: str | None = None,
    obs_version: int | None = None,
    require_optimizer: bool = False,
    load: bool = True,
) -> CheckpointValidation:
    """Check integrity and compatibility *before* anything is restored.

    Verifies, in order: the file exists and is a supported checkpoint, the
    weights are finite and match the expected architecture, the optimiser
    state is present when a continuation needs it, and the mode / observation
    layout agree with the run that wants to resume.  Every failure is
    collected in ``reasons`` so the GUI can explain a rejected checkpoint.
    """
    from .policy import PolicyConfig

    path = Path(path)
    result = CheckpointValidation(path=str(path), ok=False)
    if not path.exists():
        result.error = "file does not exist"
        result.reasons.append("file does not exist")
        return result
    try:
        result.size_bytes = path.stat().st_size
    except OSError as exc:
        result.error = f"cannot stat file: {exc}"
        result.reasons.append(result.error)
        return result

    directory = path.parent
    if (directory / MANIFEST).exists():
        try:
            manifest = read_manifest(directory)
            result.step = manifest.get("step")
            result.run_id = manifest.get("run_id")
            result.mode = manifest.get("mode")
            result.label = manifest.get("label", "") or ""
            result.created_at = manifest.get("created_at")
        except (json.JSONDecodeError, OSError) as exc:
            result.reasons.append(f"manifest unreadable: {exc}")

    if not load:
        result.ok = not result.reasons
        return result

    try:
        payload = load_checkpoint(path)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.reasons.append(result.error)
        return result

    model_state = payload.get("model")
    if not isinstance(model_state, dict) or not model_state:
        result.reasons.append("checkpoint contains no model weights")
        return result
    result.has_model = True
    result.step = int(payload.get("step", result.step or 0))
    result.run_id = payload.get("run_id") or result.run_id
    result.mode = payload.get("mode") or result.mode
    extra = payload.get("extra") or {}
    result.obs_version = extra.get("obs_version")
    result.policy_config = payload.get("policy_config")
    stored_config = payload.get("training_config")
    result.training_config = stored_config if isinstance(stored_config, dict) else None
    result.extra = extra if isinstance(extra, dict) else {}
    result.has_optimizer = bool(payload.get("optimizer"))
    result.has_rng = isinstance(extra.get("rng"), dict)
    result.has_curriculum = isinstance(extra.get("curriculum"), dict)
    result.has_league = isinstance(extra.get("league"), dict)
    timing = extra.get("timing")
    result.has_timing = isinstance(timing, dict)
    result.duration_s = _timing_value(timing, "accumulated_s")
    result.session_duration_s = _timing_value(timing, "session_s")

    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor):
            result.reasons.append(f"weight {name!r} is not a tensor")
            break
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            result.reasons.append(f"weight {name!r} contains NaN or Inf")
            break

    saved_config = PolicyConfig.from_dict(result.policy_config)
    if policy_config is not None:
        expected = PolicyConfig.from_dict(policy_config)
        if saved_config.to_dict() != expected.to_dict():
            result.reasons.append(
                "architecture mismatch: "
                f"{saved_config.to_dict()} vs requested {expected.to_dict()}"
            )
    if obs_version is not None:
        from .obs import obs_size

        if saved_config.obs_size != obs_size(int(obs_version)):
            result.reasons.append(
                f"observation layout mismatch: checkpoint obs_size={saved_config.obs_size}, "
                f"requested obs_version={int(obs_version)} (size {obs_size(int(obs_version))})"
            )
    if mode is not None and result.mode not in (None, mode):
        result.reasons.append(f"mode mismatch: checkpoint {result.mode!r} vs requested {mode!r}")
    if require_optimizer and not result.has_optimizer:
        result.reasons.append("no optimizer state: cannot continue training exactly")

    if not result.reasons:
        try:
            model = saved_config.build()
            model.load_state_dict(model_state)
        except Exception as exc:
            result.reasons.append(f"weights do not fit the declared architecture: {exc}")

    result.ok = not result.reasons
    return result


def read_manifest(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / MANIFEST).read_text(encoding="utf-8"))


def write_manifest(model_dir: str | Path, manifest: dict) -> None:
    _atomic_write_text(Path(model_dir) / MANIFEST, json.dumps(manifest, indent=2))


def _write_pointer(models_dir: str | Path, name: str, payload: dict) -> None:
    """Publish a pointer atomically; a pointer is metadata, never weights."""
    _atomic_write_text(Path(models_dir) / name, json.dumps(payload, indent=2))


def _read_pointer(models_dir: str | Path, name: str) -> dict:
    pointer = Path(models_dir) / name
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def latest_record(models_dir: str | Path) -> dict:
    """The last successfully completed save, or ``{}``.

    This is advisory metadata: a damaged pointer is ignored and discovery
    falls back to the manifests, but it lets resume and retention identify
    the latest save without parsing timestamps out of directory names.
    """
    record = _read_pointer(models_dir, LATEST_POINTER)
    return record if record.get("id") else {}


def latest_model(models_dir: str | Path) -> dict | None:
    """Manifest of the latest checkpoint, if it still exists and is readable."""
    record = latest_record(models_dir)
    if not record.get("id"):
        return None
    try:
        return get_model(models_dir, str(record["id"]))
    except (KeyError, json.JSONDecodeError, OSError):
        return None


def set_label(models_dir: str | Path, model_id: str, label: str) -> dict:
    """Relabel an existing checkpoint without writing another model file.

    Used when a checkpoint at the final step already exists (for example one
    written by the final evaluation) so completion does not create a duplicate
    copy of identical weights just to attach the ``final`` label.
    """
    models_dir = Path(models_dir)
    directory = checkpoint_dir(models_dir, model_id)
    manifest = get_model(models_dir, model_id)
    for key in ("absolute_path", "stable", "best"):
        manifest.pop(key, None)
    manifest["label"] = label
    write_manifest(directory, manifest)
    record = latest_record(models_dir)
    if str(record.get("id")) == model_id:
        record = dict(record, label=label)
        try:
            _write_pointer(models_dir, LATEST_POINTER, record)
        except OSError:
            # The manifest is authoritative; a stale pointer label is not
            # worth failing a completed run for.
            pass
    return manifest


def _stable_id(models_dir: str | Path) -> str | None:
    return _pointer_id(models_dir, STABLE_POINTER)


def _pointer_id(models_dir: str | Path, name: str) -> str | None:
    value = _read_pointer(models_dir, name).get("id")
    return str(value) if value else None


def protocol_key(protocol: dict | None) -> str:
    """Stable identity of an evaluation protocol.

    Two evaluations are only comparable when every knob that can change the
    result is identical: opponents, number of games, sides, match length,
    seeds, mode, observation layout and whether actions were sampled.
    """
    if not protocol:
        return "default"
    canonical = json.dumps(protocol, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _best_pointer_path(models_dir: str | Path, key: str) -> Path:
    models_dir = Path(models_dir)
    if key in ("", "default"):
        return models_dir / BEST_POINTER
    return models_dir / f"best-{key}.json"


def best_model(models_dir: str | Path) -> dict | None:
    """The checkpoint with the best measured evaluation score.

    This is deliberately separate from the stable pointer: a checkpoint can
    be the best so far without being promoted for deployment.
    """
    model_id = _pointer_id(models_dir, BEST_POINTER)
    if not model_id:
        return None
    try:
        return get_model(models_dir, model_id)
    except KeyError:
        return None


def best_record(models_dir: str | Path, protocol: dict | None = None) -> dict:
    """Raw best-pointer record (may be empty).

    Without ``protocol`` this is the primary pointer (``best.json``); with a
    protocol it is the best measurement taken under exactly that protocol.
    """
    key = protocol_key(protocol)
    pointer = _best_pointer_path(models_dir, key)
    if key != "default" and not pointer.exists():
        primary = Path(models_dir) / BEST_POINTER
        if primary.exists():
            try:
                data = json.loads(primary.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            if isinstance(data, dict) and data.get("protocol_key") == key:
                return data
        return {}
    if not pointer.exists():
        return {}
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def best_records(models_dir: str | Path) -> list[dict]:
    """Every best pointer, one per evaluation protocol."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    records: list[dict] = []
    for pointer in sorted(models_dir.glob("best*.json")):
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and data.get("id"):
            data = dict(data)
            data["pointer"] = pointer.name
            records.append(data)
    return records


def set_best(
    models_dir: str | Path,
    model_id: str,
    score: float,
    *,
    metric: str = "win_rate",
    protocol: dict | None = None,
) -> dict:
    """Record a better evaluation score.  Never lowers an existing best.

    Scores are only compared inside one evaluation protocol.  A result
    measured with a different protocol is written to its own pointer file so
    that "best model" always means "best under a fixed, reproducible
    measurement" instead of "best under whichever settings happened to be
    easy that day".
    """
    models_dir = Path(models_dir)
    get_model(models_dir, model_id)  # validate
    key = protocol_key(protocol)
    pointer = _best_pointer_path(models_dir, key)
    previous = {}
    if pointer.exists():
        try:
            previous = json.loads(pointer.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
    if previous.get("score") is not None and float(score) <= float(previous["score"]):
        return previous
    record = {
        "id": model_id,
        "score": float(score),
        "metric": metric,
        "protocol": protocol or {},
        "protocol_key": key,
        "updated_at": _utc_now(),
        "previous": previous.get("id"),
    }
    _atomic_write_text(pointer, json.dumps(record, indent=2))
    if key != "default":
        # Keep the primary pointer in sync only when it is empty or already
        # describes the same protocol; never mix incomparable measurements.
        primary = best_record(models_dir)
        if not primary or primary.get("protocol_key") == key:
            _atomic_write_text(models_dir / BEST_POINTER, json.dumps(record, indent=2))
    return record


def list_models(models_dir: str | Path) -> list[dict]:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    stable = _stable_id(models_dir)
    best_ids = {str(record["id"]) for record in best_records(models_dir)}
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
        manifest["best"] = str(manifest.get("id")) in best_ids
        models.append(manifest)
    models.sort(key=lambda m: (m.get("created_at", ""), m.get("step", 0)), reverse=True)
    return models


def get_model(models_dir: str | Path, model_id: str) -> dict:
    directory = checkpoint_dir(models_dir, model_id)
    if not (directory / MANIFEST).exists():
        raise KeyError(f"unknown model {model_id!r}")
    manifest = read_manifest(directory)
    manifest["stable"] = _stable_id(models_dir) == model_id
    manifest["best"] = any(str(record.get("id")) == model_id for record in best_records(models_dir))
    manifest["absolute_path"] = str(directory / MODEL_FILE)
    return manifest


def set_model_state(models_dir: str | Path, model_id: str, state: str) -> dict:
    if state not in {"candidate", "stable", "archived", "rejected"}:
        raise ValueError("state must be candidate, stable, archived or rejected")
    directory = checkpoint_dir(models_dir, model_id)
    manifest = get_model(models_dir, model_id)
    manifest.pop("absolute_path", None)
    manifest.pop("stable", None)
    manifest.pop("best", None)
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
    _atomic_write_text(models_dir / STABLE_POINTER, json.dumps(pointer, indent=2))
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
    # A pointer must never dangle.  Pruning already skips the best and latest
    # checkpoints; an explicit delete removes the pointers together with it.
    if manifest.get("best"):
        for pointer in list(models_dir.glob("best*.json")):
            try:
                if json.loads(pointer.read_text(encoding="utf-8")).get("id") == model_id:
                    pointer.unlink()
            except (json.JSONDecodeError, OSError):
                continue
    if str(latest_record(models_dir).get("id")) == model_id:
        try:
            (models_dir / LATEST_POINTER).unlink()
        except OSError:
            pass
    shutil.rmtree(checkpoint_dir(models_dir, model_id))


def _loadable(models_dir: str | Path, manifest: dict) -> bool:
    """True when the checkpoint can actually be loaded again."""
    path = Path(models_dir) / str(manifest.get("path") or "")
    try:
        return validate_checkpoint(path).ok
    except Exception:
        return False


def prune_candidates(models_dir: str | Path, keep: int = 8, run_id: str | None = None) -> list[str]:
    """Delete old routine checkpoints while keeping every safety anchor.

    ``keep`` is the number of checkpoints to retain from the run's normal
    history (the newest by training step, then by creation time).  Stable,
    every best checkpoint, the latest pointer target and the newest
    final/interrupted checkpoint are *additional* anchors and are never
    removed by this function.

    When ``run_id`` is given only checkpoints from that run are candidates, so
    routine training of one run can never delete another run's history.
    Before deleting anything the function verifies that a loadable checkpoint
    remains; if every existing file is corrupt it deletes nothing.  Returns
    the deleted ids.
    """
    models_dir = Path(models_dir)
    keep = max(0, int(keep))
    all_models = list_models(models_dir)
    if not all_models:
        return []

    eligible = [
        m
        for m in all_models
        if not m.get("stable")
        and not m.get("best")
        and (run_id is None or m.get("run_id") == run_id)
    ]
    if len(eligible) <= keep:
        return []

    # Newest training progress first.  The step is the reliable ordering
    # signal; created_at only breaks ties (two saves in the same second).
    eligible.sort(
        key=lambda m: (int(m.get("step") or 0), str(m.get("created_at") or "")),
        reverse=True,
    )
    delete = eligible[keep:]

    protected: set[str] = set()
    latest_id = str(latest_record(models_dir).get("id") or "")
    if latest_id:
        protected.add(latest_id)
    for label in ("final", "interrupt"):
        labelled = [
            m
            for m in all_models
            if m.get("label") == label and (run_id is None or m.get("run_id") == run_id)
        ]
        if labelled:
            newest = max(
                labelled,
                key=lambda m: (int(m.get("step") or 0), str(m.get("created_at") or "")),
            )
            protected.add(str(newest["id"]))
    delete = [m for m in delete if str(m["id"]) not in protected]
    if not delete:
        return []

    deleting_ids = {str(item["id"]) for item in delete}
    retained = [m for m in all_models if m["id"] not in deleting_ids]
    if not any(_loadable(models_dir, m) for m in retained):
        # Deleting anything now would remove the only recoverable state.  Try
        # to salvage the newest loadable candidate instead; if nothing loads,
        # leave the directory untouched and let the caller report it.
        salvage = next((m for m in delete if _loadable(models_dir, m)), None)
        if salvage is None:
            return []
        delete = [m for m in delete if m["id"] != salvage["id"]]

    deleted: list[str] = []
    for manifest in delete:
        if len(list_models(models_dir)) <= 1:
            break
        try:
            delete_model(models_dir, manifest["id"])
            deleted.append(manifest["id"])
        except (ValueError, OSError):
            continue
    return deleted


def load_policy(
    path: str | Path,
    device: str = "cpu",
    *,
    weights_only: bool = True,
    map_location: str | None = None,
):
    """Load a checkpoint into a ready-to-use :class:`~rlai.policy.TorchPolicy`.

    Single implementation shared by evaluation, model opponents and the RLBot
    bot, so all three reject the same files and behave identically.  The safe
    loader is the default; ``weights_only=False`` is an explicit opt-in for a
    trusted legacy file.
    """
    from .policy import PolicyConfig, TorchPolicy

    payload = load_checkpoint(
        path, weights_only=weights_only, map_location=map_location or device
    )
    config = PolicyConfig.from_dict(payload.get("policy_config"))
    policy = TorchPolicy.from_config(config, device=device)
    policy.model.load_state_dict(payload["model"])
    policy.model.eval()
    return policy


def resolve_model_path(
    models_dir: str | Path,
    reference: str,
    *,
    allow_outside: bool = False,
) -> Path:
    """Resolve a model id or path to a checkpoint file.

    By default the result must live inside ``models_dir``: references that
    arrive over HTTP must never be able to point the loader at an arbitrary
    file on the machine.  Local callers (the CLI, where the user already has
    filesystem access) can opt out with ``allow_outside=True``.
    """
    models_dir = Path(models_dir).resolve()
    text = str(reference).strip()
    if not text:
        raise FileNotFoundError("empty model reference")

    path = Path(text)
    candidates: list[Path] = []
    if path.is_absolute() or path.exists():
        candidates.append(path)
    candidates.append(checkpoint_dir(models_dir, text) / MODEL_FILE)
    if not path.is_absolute():
        candidates.append(models_dir / path)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.exists():
            continue
        inside = resolved == models_dir or models_dir in resolved.parents
        if not inside and not allow_outside:
            raise ValueError(
                f"model reference {reference!r} resolves outside the models directory "
                f"({resolved}); pass allow_outside=True only for local, trusted paths"
            )
        return resolved
    raise FileNotFoundError(f"model {reference!r} not found")


def list_checkpoint_candidates(
    models_dir: str | Path,
    *,
    mode: str | None = None,
    run_id: str | None = None,
    require_optimizer: bool | None = None,
) -> list[dict]:
    """All checkpoints, newest first, with the metadata a resume needs.

    ``require_optimizer`` filters for checkpoints that can continue an
    interrupted run exactly (weights *and* optimiser state).  Sorting is by
    training step (then creation time and the latest pointer), so a run that
    was resumed from an older checkpoint still yields its newest progress
    first instead of relying on the timestamp inside a directory name.
    """
    candidates: list[dict] = []
    for manifest in list_models(models_dir):
        if mode is not None and manifest.get("mode") not in (None, mode):
            continue
        if run_id is not None and manifest.get("run_id") != run_id:
            continue
        if require_optimizer is True and manifest.get("has_optimizer") is False:
            # Only manifests written by this version carry the flag; unknown
            # entries stay in the list and are validated when they are used.
            continue
        directory = checkpoint_dir(models_dir, manifest["id"])
        model_file = directory / MODEL_FILE
        manifest = dict(manifest)
        manifest["absolute_path"] = str(model_file)
        manifest["exists"] = model_file.exists()
        manifest["size_bytes"] = model_file.stat().st_size if model_file.exists() else 0
        candidates.append(manifest)
    latest_id = str(latest_record(models_dir).get("id") or "")
    # Training progress (the step) is the reliable ordering signal.  The
    # creation timestamp is only a tie-breaker, and the latest pointer
    # disambiguates two saves at the same step.  Directory names are never
    # parsed: the old ``m-<timestamp>-<step>`` convention was not a contract.
    candidates.sort(
        key=lambda item: (
            int(item.get("step") or 0),
            str(item.get("created_at") or ""),
            str(item.get("id")) == latest_id,
        ),
        reverse=True,
    )
    return candidates


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
