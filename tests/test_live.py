"""Live match configuration generation and platform guards."""

from __future__ import annotations

import sys
import tomllib

import pytest

from rlai.config import ensure_dirs
from rlai.live import LiveUnavailable, launch_match, write_match_config
from rlai.modes import all_modes, get_mode


@pytest.mark.parametrize("mode_id", [spec.id for spec in all_modes()])
def test_generated_match_config(mode_id, tmp_path):
    spec = get_mode(mode_id)
    directory = tmp_path / "live"
    path = write_match_config(
        mode=mode_id,
        model_path=tmp_path / "models" / "m" / "model.pt",
        python=sys.executable,
        directory=directory,
    )
    data = tomllib.loads(path.read_text())
    assert data["rlbot"]["launcher"] == "Epic"
    assert data["match"]["game_mode"] == ("Heatseeker" if spec.heatseeker else "Soccer")
    cars = data["cars"]
    assert len(cars) == spec.team_size + 1
    assert cars[0]["type"] == "Human" and cars[0]["team"] == "Blue"
    assert all(car["team"] == "Orange" and car["auto_start"] for car in cars[1:])
    agent_ids = []
    for index, car in enumerate(cars[1:]):
        bot_path = directory / f"bot-{index}.toml"
        assert bot_path.exists()
        bot = tomllib.loads(bot_path.read_text())["settings"]
        agent_ids.append(bot["agent_id"])
        assert bot["root_dir"]
        assert "bot.py" in bot["run_command"]
    assert len(set(agent_ids)) == len(agent_ids)
    if spec.heatseeker:
        assert data["mutators"]["boost_amount"] == "UnlimitedBoost"


def test_live_launch_is_refused_off_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("RLAI_DATA_DIR", str(tmp_path))
    paths = ensure_dirs(tmp_path)
    with pytest.raises(LiveUnavailable):
        launch_match(paths, None, "soccer-1v1")
