"""Create a live RLBot match configuration and launch it (Windows only)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from . import checkpoints
from .config import project_root
from .modes import get_mode, mode_ids


class LiveUnavailable(RuntimeError):
    pass


_LIVE: dict[str, dict] = {}


def _toml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _bot_toml(name: str, agent_id: str, python: str, bot_file: Path, repo: Path) -> str:
    run_command = f"{_toml_quote(python)} {_toml_quote(str(bot_file))}"
    return f"""[settings]
name = {_toml_quote(name)}
agent_id = {_toml_quote(agent_id)}
root_dir = {_toml_quote(str(repo))}
run_command = {_toml_quote(run_command)}
run_command_linux = {_toml_quote(run_command)}

[details]
description = "Rocket League AI policy trained with discrete-action PPO in RocketSim."
developer = "rocket-league-ai"
language = "Python"
tags = ["1v1", "teamplay", "heatseeker"]
"""


def write_match_config(
    *,
    mode: str,
    model_path: Path,
    python: str,
    directory: Path,
) -> Path:
    spec = get_mode(mode)
    repo = project_root()
    bot_file = repo / "rlbot" / "bot.py"
    directory.mkdir(parents=True, exist_ok=True)
    cars = ['[[cars]]\ntype = "Human"\nteam = "Blue"']
    for index in range(spec.team_size):
        bot_config = directory / f"bot-{index}.toml"
        bot_config.write_text(
            _bot_toml(
                f"RLAI {spec.id} {index}",
                f"rlai/{mode}-{index}",
                python,
                bot_file,
                repo,
            ),
            encoding="utf-8",
        )
        cars.append(
            "[[cars]]\n"
            f"config_file = {_toml_quote(str(bot_config))}\n"
            'team = "Orange"\n'
            "auto_start = true"
        )
    game_mode = "Heatseeker" if spec.heatseeker else "Soccer"
    mutators = 'boost_amount = "UnlimitedBoost"\n' if spec.heatseeker else ""
    match = f"""[rlbot]
launcher = "Epic"
auto_start_agents = true
wait_for_agents = true

[match]
game_mode = {_toml_quote(game_mode)}
game_map_upk = {_toml_quote("Stadium_P")}
skip_replays = true
start_without_countdown = true
existing_match_behavior = "Restart"
enable_rendering = true
enable_state_setting = true
auto_save_replay = false
freeplay = false

[mutators]
match_length = "TenMinutes"
max_score = "Unlimited"
overtime = "Unlimited"
{mutators}
{chr(10).join(cars)}
"""
    path = directory / "match.toml"
    path.write_text(match, encoding="utf-8")
    (directory / "launch.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "model": str(model_path),
                "python": python,
                "created_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def launch_match(paths: dict[str, Path], model_id: str | None, mode: str) -> dict:
    if platform.system() != "Windows":
        raise LiveUnavailable(
            "Live match launch requires Windows with Rocket League (Epic) and RLBot v5 installed. "
            "This environment can only run the headless simulator. Use the documented Windows procedure."
        )
    try:
        import rlbot  # noqa: F401
    except Exception as exc:
        raise LiveUnavailable(f"rlbot is not installed in this environment: {exc}") from exc
    if mode not in mode_ids():
        raise LiveUnavailable(f"unsupported mode {mode!r}")
    if model_id:
        manifest = checkpoints.get_model(paths["models"], model_id)
    else:
        manifest = checkpoints.stable_model(paths["models"])
    if not manifest:
        raise LiveUnavailable("No trained model available. Train or promote a checkpoint first.")
    model_path = paths["models"] / manifest["path"]
    directory = paths["root"] / "live" / f"{mode}-{int(time.time())}"
    config = write_match_config(
        mode=mode, model_path=model_path, python=sys.executable, directory=directory
    )
    log_path = directory / "launch.log"
    log_file = open(log_path, "ab")
    env = os.environ.copy()
    env["RLAI_MODEL"] = str(model_path)
    env["RLAI_MODE"] = mode
    env["PYTHONPATH"] = str(project_root() / "src")
    env["RLAI_DATA_DIR"] = str(paths["root"])
    process = subprocess.Popen(
        [sys.executable, "-m", "rlai.rlbot.run", "--config", str(config)],
        cwd=str(project_root()),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    log_file.close()
    record = {
        "pid": process.pid,
        "mode": mode,
        "model_id": manifest["id"],
        "config": str(config),
        "log": str(log_path),
    }
    _LIVE[str(process.pid)] = record
    return record
