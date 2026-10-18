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


def prepare_match(paths: dict[str, Path], model_id: str | None, mode: str) -> dict:
    """Resolve a model and write all match files without launching anything.

    Useful for a dry run on any OS and as the single source of truth for both
    the GUI and the CLI.
    """
    if mode not in mode_ids():
        raise LiveUnavailable(f"unsupported mode {mode!r}")
    if model_id:
        try:
            manifest = checkpoints.get_model(paths["models"], model_id)
        except KeyError as exc:
            raise LiveUnavailable(str(exc)) from exc
    else:
        manifest = checkpoints.stable_model(paths["models"])
    if not manifest:
        raise LiveUnavailable("No trained model available. Train or promote a checkpoint first.")
    model_path = paths["models"] / manifest["path"]
    directory = paths["root"] / "live" / f"{mode}-{int(time.time())}"
    config = write_match_config(
        mode=mode, model_path=model_path, python=sys.executable, directory=directory
    )
    return {
        "mode": mode,
        "model_id": manifest["id"],
        "model": str(model_path),
        "config": str(config),
        "directory": str(directory),
        "log": str(directory / "launch.log"),
    }


def preflight(paths: dict[str, Path]) -> dict:
    """Environment checks for a reliable first live test on Windows."""
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "required": required})

    is_windows = platform.system() == "Windows"
    add("windows", is_windows, f"platform is {platform.system()}", required=True)
    try:
        import rlbot

        add("rlbot", True, f"rlbot {getattr(rlbot, '__version__', 'unknown')} importable", True)
    except Exception as exc:
        add("rlbot", False, f"rlbot import failed: {exc}", True)
    bot_file = project_root() / "rlbot" / "bot.py"
    add("bot entrypoint", bot_file.exists(), str(bot_file), True)
    stable = checkpoints.stable_model(paths["models"])
    best = checkpoints.best_model(paths["models"])
    add(
        "model",
        bool(stable or best),
        f"stable={stable['id'] if stable else None} best={best['id'] if best else None}",
        True,
    )
    try:
        probe = paths["root"] / "live" / ".preflight"
        probe.mkdir(parents=True, exist_ok=True)
        (probe / "write.test").write_text("ok", encoding="utf-8")
        (probe / "write.test").unlink()
        add("data directory writable", True, str(paths["root"]), True)
    except OSError as exc:
        add("data directory writable", False, str(exc), True)
    try:
        import psutil

        running = any(
            proc.info.get("name", "") == "RocketLeague.exe"
            for proc in psutil.process_iter(["name"])
        )
        add("game running", running, "RocketLeague.exe", False)
    except Exception as exc:  # pragma: no cover - psutil is a dependency
        add("game running", False, str(exc), False)
    return {
        "ok": all(check["ok"] for check in checks if check["required"]),
        "platform": platform.system(),
        "checks": checks,
    }


def launch_match(
    paths: dict[str, Path], model_id: str | None, mode: str, *, dry_run: bool = False
) -> dict:
    if not dry_run and platform.system() != "Windows":
        raise LiveUnavailable(
            "Live match launch requires Windows with Rocket League (Epic) and RLBot v5 installed. "
            "This environment can only run the headless simulator. Use the documented Windows procedure."
        )
    record = prepare_match(paths, model_id, mode)
    if dry_run:
        record["launched"] = False
        record["dry_run"] = True
        return record
    try:
        import rlbot  # noqa: F401
    except Exception as exc:
        raise LiveUnavailable(f"rlbot is not installed in this environment: {exc}") from exc
    config = Path(record["config"])
    model_path = Path(record["model"])
    log_path = Path(record["log"])
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
    record.update({"pid": process.pid, "launched": True})
    _LIVE[str(process.pid)] = record
    return record
