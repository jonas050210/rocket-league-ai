import json

import pytest
from conftest import unit_checkpoint_writer

from rocket_league_ai.checkpoints import CheckpointStore
from rocket_league_ai.config import TrainingConfig
from rocket_league_ai.domain import AppError
from rocket_league_ai.persistence import Lease, lease_available


@pytest.fixture
def checkpoints(tmp_path):
    return CheckpointStore(tmp_path)


def publish(checkpoints, step=128):
    return checkpoints.publish(step, TrainingConfig().digest(), unit_checkpoint_writer, {"test_fixture": True})


def test_atomic_publication_and_integrity(checkpoints):
    item = publish(checkpoints)
    assert checkpoints.inspect(item["id"])["integrity"] == "valid"
    assert checkpoints.latest_valid()["id"] == item["id"]
    assert json.loads((checkpoints.root / "latest.json").read_text())["id"] == item["id"]
    assert not list(checkpoints.root.glob(".pending-*"))


def test_failed_write_preserves_last_valid_and_pointer(checkpoints):
    previous = publish(checkpoints)

    def broken_writer(path):
        (path / "partial.pt").write_bytes(b"incomplete")
        assert checkpoints.latest_valid()["id"] == previous["id"]
        raise OSError("disk full (injected test fault)")

    with pytest.raises(OSError, match="disk full"):
        checkpoints.publish(256, TrainingConfig().digest(), broken_writer, {})
    assert checkpoints.latest_valid()["id"] == previous["id"]
    assert len(checkpoints.inventory()) == 1
    assert not list(checkpoints.root.glob(".pending-*"))


def test_crash_staging_and_corrupt_latest_pointer_are_ignored(checkpoints):
    item = publish(checkpoints)
    pending = checkpoints.root / ".pending-interrupted-process"
    pending.mkdir()
    (pending / "weights.pt").write_bytes(b"interrupted write")
    (checkpoints.root / "latest.json").write_text("broken pointer")
    assert len(checkpoints.inventory()) == 1
    assert checkpoints.latest_valid()["id"] == item["id"]


def test_corrupt_newest_falls_back_and_explicit_load_fails(checkpoints):
    previous = publish(checkpoints)
    newest = publish(checkpoints, 256)
    file = checkpoints.path(newest["id"]) / "ppo_learner/actor_critic.pt"
    file.write_bytes(b"corrupted weights")
    assert checkpoints.inspect(newest["id"])["integrity"] == "corrupt"
    assert checkpoints.latest_valid()["id"] == previous["id"]
    with pytest.raises(AppError, match="older valid"):
        checkpoints.require_valid(newest["id"])


def test_incomplete_writer_cannot_publish(checkpoints):
    with pytest.raises(ValueError, match="Incomplete"):
        checkpoints.publish(0, TrainingConfig().digest(), lambda path: None, {})
    assert checkpoints.inventory() == []


def test_config_mismatch_rejected_before_deserialization(checkpoints):
    item = publish(checkpoints)
    with pytest.raises(AppError, match="configuration hash"):
        checkpoints.require_valid(item["id"], TrainingConfig(seed=321).digest())


@pytest.mark.parametrize("value", ["../outside", "/tmp/arbitrary", "step-1", "a" * 32])
def test_no_arbitrary_checkpoint_paths(checkpoints, value):
    with pytest.raises(AppError):
        checkpoints.path(value)


def test_symlink_payload_and_unexpected_files_are_rejected(checkpoints, tmp_path):
    item = publish(checkpoints)
    target = checkpoints.path(item["id"])
    file = target / "ppo_learner/actor_critic.pt"
    file.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"unit-test-checkpoint-bytes")
    file.symlink_to(outside)
    assert checkpoints.inspect(item["id"])["integrity"] == "corrupt"


def test_lease_prevents_duplicate_ownership(tmp_path):
    path = tmp_path / "worker.lock"
    assert lease_available(path)
    with Lease(path):
        assert not lease_available(path)
        with pytest.raises(AppError, match="another process"):
            Lease(path)
    assert lease_available(path)


@pytest.mark.parametrize("mutate", [
    lambda value: [],
    lambda value: {**value, "step": "128"},
    lambda value: {**value, "step": True},
    lambda value: {**value, "step": -1},
    lambda value: {**value, "step": 256},
    lambda value: {**value, "created": float("nan")},
    lambda value: {**value, "config_hash": "invalid"},
    lambda value: {**value, "schema_version": True},
    lambda value: {**value, "files": []},
    lambda value: {**value, "files": {**value["files"], "rng.pt": None}},
    lambda value: {**value, "files": {**value["files"], "rng.pt": {"bytes": -1, "sha256": "0" * 64}}},
])
def test_malformed_manifests_are_corrupt_not_inventory_crashes(checkpoints, mutate):
    older = publish(checkpoints, 0)
    item = publish(checkpoints)
    path = checkpoints.path(item["id"]) / "manifest.json"
    path.write_text(json.dumps(mutate(json.loads(path.read_text()))))
    assert checkpoints.inspect(item["id"])["integrity"] == "corrupt"
    assert checkpoints.latest_valid()["id"] == older["id"]


def test_symlink_checkpoint_is_listed_corrupt_without_breaking_other_models(checkpoints, tmp_path):
    item = publish(checkpoints)
    linked = checkpoints.root / "step-000000000256-aaaaaaaaaaaa"
    linked.symlink_to(tmp_path, target_is_directory=True)
    inventory = checkpoints.inventory()
    assert len(inventory) == 2
    assert next(value for value in inventory if value["id"] == linked.name)["integrity"] == "corrupt"
    assert checkpoints.latest_valid()["id"] == item["id"]


def test_same_size_damage_with_preserved_mtime_invalidates_cache(checkpoints):
    import os

    item = publish(checkpoints)
    assert checkpoints.inspect(item["id"])["integrity"] == "valid"
    path = checkpoints.path(item["id"]) / "rng.pt"
    stat = path.stat()
    path.write_bytes(b"x" * stat.st_size)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert checkpoints.inspect(item["id"])["integrity"] == "corrupt"
