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


def test_save_records_the_latest_pointer(tmp_path):
    """Discovery must not have to parse a timestamp out of a directory name."""
    first = _save(tmp_path, 10)
    second = _save(tmp_path, 20)
    record = checkpoints.latest_record(tmp_path)
    assert record["id"] == second["id"]
    assert record["step"] == 20
    assert checkpoints.latest_model(tmp_path)["id"] == second["id"]
    assert first["id"] != second["id"]


def test_a_failed_save_leaves_the_previous_checkpoint_and_pointer_intact(
    tmp_path, monkeypatch
):
    first = _save(tmp_path, 10)

    def broken_save(payload, handle):
        handle.write(b"partial garbage")
        raise OSError("disk full")

    monkeypatch.setattr(checkpoints.torch, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        _save(tmp_path, 20)

    models = checkpoints.list_models(tmp_path)
    assert [m["id"] for m in models] == [first["id"]]
    assert checkpoints.latest_record(tmp_path)["id"] == first["id"]
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == [first["id"]]
    assert checkpoints.load_checkpoint(tmp_path / first["path"])["step"] == 10


def test_pruning_keeps_the_best_checkpoint_when_it_is_older_than_recovery(tmp_path):
    best = _save(tmp_path, 10)
    for step in (20, 30, 40):
        _save(tmp_path, step)
    checkpoints.set_best(tmp_path, best["id"], 0.9)

    deleted = checkpoints.prune_candidates(tmp_path, keep=1)

    remaining = {m["id"] for m in checkpoints.list_models(tmp_path)}
    assert best["id"] in remaining
    assert checkpoints.latest_record(tmp_path)["id"] in remaining
    assert deleted
    assert best["id"] not in deleted


def test_pruning_keeps_a_final_checkpoint_when_it_is_older_than_recovery(tmp_path):
    final = _save(tmp_path, 10)
    checkpoints.set_label(tmp_path, final["id"], "final")
    for step in (20, 30, 40):
        _save(tmp_path, step)

    checkpoints.prune_candidates(tmp_path, keep=1)

    remaining = {m["id"] for m in checkpoints.list_models(tmp_path)}
    assert final["id"] in remaining
    assert checkpoints.get_model(tmp_path, final["id"])["label"] == "final"


def test_pruning_never_deletes_the_only_loadable_checkpoint(tmp_path):
    valid = _save(tmp_path, 10)
    corrupt = _save(tmp_path, 20)
    (tmp_path / corrupt["path"]).write_bytes(b"not a torch file")

    deleted = checkpoints.prune_candidates(tmp_path, keep=0)

    assert deleted == []
    assert checkpoints.load_checkpoint(tmp_path / valid["path"])["step"] == 10
    # the newest (corrupt) attempt is retained too; nothing was removed
    assert {m["id"] for m in checkpoints.list_models(tmp_path)} == {
        valid["id"],
        corrupt["id"],
    }


def test_pruning_does_nothing_when_every_checkpoint_is_corrupt(tmp_path):
    for step in (10, 20):
        manifest = _save(tmp_path, step)
        (tmp_path / manifest["path"]).write_bytes(b"broken")
    assert checkpoints.prune_candidates(tmp_path, keep=0) == []
    assert len(checkpoints.list_models(tmp_path)) == 2


def test_candidate_order_uses_the_training_step(tmp_path):
    _save(tmp_path, 64)
    _save(tmp_path, 1024)
    _save(tmp_path, 512)
    steps = [item["step"] for item in checkpoints.list_checkpoint_candidates(tmp_path)]
    assert steps == [1024, 512, 64]
    # the pointer still records the last *save*, not the highest step
    assert checkpoints.latest_record(tmp_path)["step"] == 512


def test_pruning_keeps_the_latest_saved_checkpoint_even_below_the_keep_window(tmp_path):
    # simulate an out-of-order save: the pointer represents a completed save
    # at a lower step than checkpoints that were written earlier
    _save(tmp_path, 20)
    _save(tmp_path, 30)
    _save(tmp_path, 40)
    last_saved = _save(tmp_path, 10)
    assert checkpoints.latest_record(tmp_path)["id"] == last_saved["id"]

    checkpoints.prune_candidates(tmp_path, keep=1)

    remaining = {m["id"] for m in checkpoints.list_models(tmp_path)}
    assert last_saved["id"] in remaining
    # the newest training progress is retained as well
    assert checkpoints.list_checkpoint_candidates(tmp_path)[0]["step"] == 40


def test_a_failed_manifest_write_leaves_no_half_published_checkpoint(tmp_path, monkeypatch):
    first = _save(tmp_path, 10)

    def broken_manifest(directory, manifest):
        raise OSError("disk full while writing manifest")

    monkeypatch.setattr(checkpoints, "write_manifest", broken_manifest)
    with pytest.raises(OSError, match="manifest"):
        _save(tmp_path, 20)

    assert [m["id"] for m in checkpoints.list_models(tmp_path)] == [first["id"]]
    assert checkpoints.latest_record(tmp_path)["id"] == first["id"]
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == [first["id"]]
