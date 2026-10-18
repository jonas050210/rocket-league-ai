"""Live match configuration generation and platform guards."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

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


def test_preflight_reports_every_check(data_root, monkeypatch):
    from rlai.live import preflight
    from rlai.config import ensure_dirs

    paths = ensure_dirs()
    report = preflight(paths)
    assert set(report) == {"ok", "platform", "checks"}
    names = {check["name"] for check in report["checks"]}
    assert {"windows", "rlbot", "bot entrypoint", "model"} <= names
    for check in report["checks"]:
        assert isinstance(check["ok"], bool)
        assert check["detail"]
    # without a model the required checks cannot all pass
    assert report["ok"] is False


def test_dry_run_writes_match_files(data_root, models_dir, monkeypatch):
    from rlai.config import ensure_dirs
    from rlai.live import launch_match

    from rlai import checkpoints

    paths = ensure_dirs()
    model_id = checkpoints.list_models(models_dir)[0]["id"]
    result = launch_match(paths, model_id, "soccer-1v1", dry_run=True)
    assert result["dry_run"] is True
    assert result["launched"] is False
    assert Path(result["config"]).exists()
    assert Path(result["directory"]).is_dir()
    assert Path(result["model"]).exists()
    assert "pid" not in result


def test_dry_run_rejects_unknown_mode(data_root, models_dir):
    import pytest

    from rlai.config import ensure_dirs
    from rlai.live import LiveUnavailable, launch_match

    with pytest.raises(LiveUnavailable, match="unsupported mode"):
        launch_match(ensure_dirs(), None, "hoops", dry_run=True)
