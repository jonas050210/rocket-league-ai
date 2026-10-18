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


def test_best_pointer_only_moves_up(tmp_path):
    first = _save(tmp_path, 10)
    second = _save(tmp_path, 20)
    assert checkpoints.best_model(tmp_path) is None
    assert checkpoints.best_record(tmp_path) == {}

    record = checkpoints.set_best(tmp_path, first["id"], 0.4)
    assert record["id"] == first["id"]
    assert checkpoints.best_model(tmp_path)["id"] == first["id"]
    assert checkpoints.best_record(tmp_path)["score"] == pytest.approx(0.4)

    # a worse or equal score never replaces the pointer
    assert checkpoints.set_best(tmp_path, second["id"], 0.39)["id"] == first["id"]
    assert checkpoints.set_best(tmp_path, second["id"], 0.4)["id"] == first["id"]
    assert checkpoints.best_model(tmp_path)["id"] == first["id"]

    record = checkpoints.set_best(tmp_path, second["id"], 0.7, metric="win_rate")
    assert record["previous"] == first["id"]
    assert checkpoints.best_model(tmp_path)["id"] == second["id"]
    # best is not the same thing as stable
    assert checkpoints.best_model(tmp_path)["stable"] is False
    assert checkpoints.stable_model(tmp_path) is None


def test_best_flag_in_listing_and_unknown_id(tmp_path):
    manifest = _save(tmp_path, 10)
    checkpoints.set_best(tmp_path, manifest["id"], 0.5)
    listed = checkpoints.list_models(tmp_path)
    assert listed[0]["best"] is True
    assert checkpoints.get_model(tmp_path, manifest["id"])["best"] is True
    with pytest.raises(KeyError):
        checkpoints.set_best(tmp_path, "does-not-exist", 0.9)


def test_pruning_never_deletes_the_best(tmp_path):
    first = _save(tmp_path, 10)
    for step in (20, 30, 40):
        _save(tmp_path, step)
    checkpoints.set_best(tmp_path, first["id"], 0.8)
    deleted = checkpoints.prune_candidates(tmp_path, keep=1)
    assert first["id"] not in deleted
    assert checkpoints.best_model(tmp_path)["id"] == first["id"]
    assert first["id"] in {m["id"] for m in checkpoints.list_models(tmp_path)}


def test_deleting_best_removes_the_pointer(tmp_path):
    first = _save(tmp_path, 10)
    _save(tmp_path, 20)
    checkpoints.set_best(tmp_path, first["id"], 0.8)
    checkpoints.delete_model(tmp_path, first["id"])
    assert checkpoints.best_model(tmp_path) is None
    assert checkpoints.best_record(tmp_path) == {}


def test_corrupt_best_pointer_is_ignored(tmp_path):
    _save(tmp_path, 10)
    (tmp_path / checkpoints.BEST_POINTER).write_text("{not json", encoding="utf-8")
    assert checkpoints.best_model(tmp_path) is None
    assert checkpoints.best_record(tmp_path) == {}
