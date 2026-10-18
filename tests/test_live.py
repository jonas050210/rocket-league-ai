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
    # RLBot's enum uses Psyonix's spelling "Soccar"; "Soccer" raises ConfigParsingException.
    assert data["match"]["game_mode"] == ("Heatseeker" if spec.heatseeker else "Soccar")
    cars = data["cars"]
    # a full match: human + policy team + the human team's fillers
    assert len(cars) == 2 * spec.team_size
    assert cars[0]["type"] == "Human" and cars[0]["team"] == "Blue"
    policy_cars = [car for car in cars if "config_file" in car]
    assert len(policy_cars) == spec.team_size
    assert all(car["team"] == "Orange" and car["auto_start"] for car in policy_cars)
    if spec.team_size > 1:
        fillers = [car for car in cars if car.get("type") == "psyonix"]
        assert len(fillers) == spec.team_size - 1
        assert all(car["team"] == "Blue" for car in fillers)
    agent_ids = []
    for index in range(spec.team_size):
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
    assert {"ok", "platform", "checks", "server"} == set(report)
    assert {"available", "running"} <= set(report["server"])
    names = {check["name"] for check in report["checks"]}
    assert {
        "windows",
        "rlbot",
        "rlbot server",
        "bot entrypoint",
        "launcher entrypoint",
        "model",
        "match config parser",
    } <= names
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
    assert result["validation"]["ok"] is True
    assert result["validation"]["players"] == 2


def test_validation_catches_the_soccer_typo(tmp_path):
    from rlai.live import validate_match_config

    directory = tmp_path / "live"
    path = write_match_config(
        mode="soccer-1v1",
        model_path=tmp_path / "model.pt",
        python=sys.executable,
        directory=directory,
    )
    assert validate_match_config(path)["ok"] is True
    broken = path.read_text().replace('game_mode = "Soccar"', 'game_mode = "Soccer"')
    path.write_text(broken)
    report = validate_match_config(path)
    assert report["ok"] is False
    assert "Soccer" in report["detail"]


def test_dry_run_rejects_unknown_mode(data_root, models_dir):
    import pytest

    from rlai.config import ensure_dirs
    from rlai.live import LiveUnavailable, launch_match

    with pytest.raises(LiveUnavailable, match="unsupported mode"):
        launch_match(ensure_dirs(), None, "hoops", dry_run=True)


def test_human_team_and_fill_options(tmp_path):
    directory = tmp_path / "live"
    path = write_match_config(
        mode="soccer-2v2",
        model_path=tmp_path / "model.pt",
        python=sys.executable,
        directory=directory,
        human_team="orange",
        fill="policy",
    )
    data = tomllib.loads(path.read_text())
    assert data["cars"][0]["type"] == "Human"
    assert data["cars"][0]["team"] == "Orange"
    # policy fills both teams: 2v2 means four policy bots, no Psyonix bots
    bots = [car for car in data["cars"] if "config_file" in car]
    assert len(bots) == 3
    assert sum(1 for car in bots if car["team"] == "Blue") == 2
    assert sum(1 for car in bots if car["team"] == "Orange") == 1
    assert not any(car.get("type") == "psyonix" for car in data["cars"])

    path = write_match_config(
        mode="soccer-3v3",
        model_path=tmp_path / "model.pt",
        python=sys.executable,
        directory=directory,
        human_team="blue",
        fill="psyonix",
    )
    data = tomllib.loads(path.read_text())
    psyonix = [car for car in data["cars"] if car.get("type") == "psyonix"]
    assert len(psyonix) == 2
    assert all(car["team"] == "Blue" for car in psyonix)

    path = write_match_config(
        mode="soccer-1v1",
        model_path=tmp_path / "model.pt",
        python=sys.executable,
        directory=directory,
        human=False,
    )
    data = tomllib.loads(path.read_text())
    assert not any(car.get("type") == "Human" for car in data["cars"])
    assert len(data["cars"]) == 2


def test_invalid_human_options_are_rejected(tmp_path):
    from rlai.live import LiveUnavailable

    for kwargs in ({"human_team": "green"}, {"fill": "robots"}):
        with pytest.raises(LiveUnavailable):
            write_match_config(
                mode="soccer-1v1",
                model_path=tmp_path / "model.pt",
                python=sys.executable,
                directory=tmp_path / "live",
                **kwargs,
            )


def test_generated_configs_parse_with_rlbot(tmp_path):
    """The regression test for ``game_mode = "Soccer"``.

    RLBot's own parser is the authority here: a config that only looks right
    in our TOML reader but fails ``load_match_config`` cannot start a match.
    """
    rlbot_config = pytest.importorskip("rlbot.config")
    for spec in all_modes():
        for fill in ("psyonix", "policy"):
            directory = tmp_path / f"{spec.id}-{fill}"
            path = write_match_config(
                mode=spec.id,
                model_path=tmp_path / "model.pt",
                python=sys.executable,
                directory=directory,
                fill=fill,
            )
            parsed = rlbot_config.load_match_config(path)
            mode_name = str(parsed.game_mode).split(".")[-1]
            assert mode_name == ("Heatseeker" if spec.heatseeker else "Soccar")
            assert len(parsed.player_configurations) == 2 * spec.team_size
            assert [int(p.team) for p in parsed.player_configurations].count(0) == spec.team_size
            assert [int(p.team) for p in parsed.player_configurations].count(1) == spec.team_size
            for player in parsed.player_configurations:
                variety = player.variety
                if type(variety).__name__ == "CustomBot":
                    assert variety.run_command.strip()
                    assert Path(variety.root_dir).is_dir()
                    assert "bot.py" in variety.run_command


def test_launcher_dry_run_parses_every_mode(tmp_path):
    import subprocess

    from rlai.live import write_match_config

    for spec in all_modes():
        directory = tmp_path / spec.id
        path = write_match_config(
            mode=spec.id,
            model_path=tmp_path / "model.pt",
            python=sys.executable,
            directory=directory,
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "rlai.rlbot.run",
                "--config",
                str(path),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        summary = __import__("json").loads(completed.stdout)
        assert summary["game_mode"] == (
            "GameMode.Heatseeker" if spec.heatseeker else "GameMode.Soccar"
        )
        assert len(summary["players"]) == 2 * spec.team_size
