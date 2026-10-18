"""Mode catalogue and configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rlai.config import TrainingConfig
from rlai.modes import all_modes, get_mode, mode_ids

EXPECTED = {
    "soccer-1v1": (1, False),
    "soccer-2v2": (2, False),
    "soccer-3v3": (3, False),
    "heatseeker-2v2": (2, True),
}


def test_four_modes_are_present():
    assert set(mode_ids()) == set(EXPECTED)


@pytest.mark.parametrize("mode_id,expected", EXPECTED.items())
def test_mode_spec(mode_id, expected):
    spec = get_mode(mode_id)
    team_size, heatseeker = expected
    assert spec.team_size == team_size
    assert spec.heatseeker is heatseeker
    assert spec.players == team_size * 2
    assert spec.infinite_boost is heatseeker
    assert spec.opponent_slots == 3
    assert spec.to_dict()["players"] == spec.players


def test_unknown_mode_is_rejected():
    with pytest.raises(KeyError):
        get_mode("soccer-4v4")


def test_all_modes_are_stable_instances():
    first = all_modes()
    second = all_modes()
    assert [m.id for m in first] == [m.id for m in second]


def test_config_rejects_short_run():
    with pytest.raises(ValidationError):
        TrainingConfig(total_steps=100, rollout_steps=64, n_workers=4)


def test_config_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        TrainingConfig(mode="hoops")


def test_config_rejects_bad_device():
    with pytest.raises(ValidationError):
        TrainingConfig(device="tpu")


def test_shipped_configs_are_valid():
    from pathlib import Path

    configs = sorted(Path(__file__).resolve().parents[1].joinpath("configs").glob("*.json"))
    assert len(configs) >= 7
    modes = set()
    for path in configs:
        config = TrainingConfig.load(path)
        config.to_dict()
        modes.add(config.mode)
    assert modes == set(EXPECTED)


def test_config_roundtrip(tmp_path):
    config = TrainingConfig(run_name="roundtrip", mode="heatseeker-2v2")
    path = config.save(tmp_path / "config.json")
    loaded = TrainingConfig.load(path)
    assert loaded.to_dict() == config.to_dict()
