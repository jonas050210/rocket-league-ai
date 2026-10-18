"""Checkpoint safety: atomic ids, explicit promotion, deletion guards."""

from __future__ import annotations

import pytest
import torch

from rlai import checkpoints
from rlai.policy import PolicyConfig


def _save(models_dir, step):
    config = PolicyConfig(hidden=(16,))
    model = config.build()
    return checkpoints.save_checkpoint(
        models_dir,
        model_state=model.state_dict(),
        policy_config=config.to_dict(),
        step=step,
        mode="soccer-1v1",
    )


def test_save_load_and_list(tmp_path):
    manifest = _save(tmp_path, 10)
    path = tmp_path / manifest["path"]
    assert path.exists()
    payload = checkpoints.load_checkpoint(path)
    assert payload["step"] == 10
    assert payload["policy_config"]["hidden"] == [16]
    models = checkpoints.list_models(tmp_path)
    assert len(models) == 1 and models[0]["id"] == manifest["id"]
    assert models[0]["stable"] is False


def test_no_overwrite_same_step(tmp_path):
    first = _save(tmp_path, 10)
    second = _save(tmp_path, 10)
    assert first["id"] != second["id"]
    assert len(checkpoints.list_models(tmp_path)) == 2


def test_promotion_archives_previous(tmp_path):
    first = _save(tmp_path, 10)
    second = _save(tmp_path, 20)
    checkpoints.promote_model(tmp_path, first["id"])
    promoted = checkpoints.promote_model(tmp_path, second["id"])
    assert promoted["stable"] is True
    stable = checkpoints.stable_model(tmp_path)
    assert stable["id"] == second["id"]
    first_manifest = checkpoints.get_model(tmp_path, first["id"])
    assert first_manifest["state"] == "archived"


def test_delete_rules(tmp_path):
    first = _save(tmp_path, 10)
    second = _save(tmp_path, 20)
    checkpoints.delete_model(tmp_path, first["id"])
    with pytest.raises(ValueError, match="last remaining"):
        checkpoints.delete_model(tmp_path, second["id"])
    checkpoints.promote_model(tmp_path, second["id"])
    with pytest.raises(ValueError, match="stable"):
        checkpoints.delete_model(tmp_path, second["id"])


def test_pruning_is_scoped_to_one_run(tmp_path):
    old = checkpoints.save_checkpoint(
        tmp_path,
        model_state=PolicyConfig(hidden=(16,)).build().state_dict(),
        policy_config=PolicyConfig(hidden=(16,)).to_dict(),
        step=1,
        run_id="run-a",
    )
    for step in range(2, 8):
        checkpoints.save_checkpoint(
            tmp_path,
            model_state=PolicyConfig(hidden=(16,)).build().state_dict(),
            policy_config=PolicyConfig(hidden=(16,)).to_dict(),
            step=step,
            run_id="run-b",
        )
    deleted = checkpoints.prune_candidates(tmp_path, keep=3, run_id="run-b")
    assert deleted
    remaining = {m["id"] for m in checkpoints.list_models(tmp_path)}
    assert old["id"] in remaining
    assert sum(1 for m in checkpoints.list_models(tmp_path) if m.get("run_id") == "run-b") == 3


def test_resolve_model_path(tmp_path):
    manifest = _save(tmp_path, 5)
    assert checkpoints.resolve_model_path(tmp_path, manifest["id"]).exists()
    with pytest.raises(FileNotFoundError):
        checkpoints.resolve_model_path(tmp_path, "does-not-exist")


def test_torch_load_of_production_checkpoint(trained_model):
    from pathlib import Path

    payload = checkpoints.load_checkpoint(
        Path(trained_model["_models_dir"]) / trained_model["path"]
    )
    config = PolicyConfig.from_dict(payload["policy_config"])
    model = config.build()
    model.load_state_dict(payload["model"])
    with torch.no_grad():
        logits, values = model(torch.zeros(1, config.obs_size))
    assert logits.shape == (1, 90) and values.shape == (1,)
