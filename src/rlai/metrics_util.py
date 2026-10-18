"""Merge filesystem checkpoint manifests with evaluation scores from the DB."""

from __future__ import annotations

from pathlib import Path

from . import checkpoints
from .store import Store


def enrich_models(store: Store, models_dir: str | Path) -> list[dict]:
    models = checkpoints.list_models(models_dir)
    scores = {row["id"]: row.get("eval_score") for row in store.list_models()}
    for model in models:
        model["eval_score"] = scores.get(model["id"])
    return models
