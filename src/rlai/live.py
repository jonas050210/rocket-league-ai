"""Create an RLBot v5 match from a trained policy and launch it.

The module is deliberately split into three steps:

``prepare_match`` writes and validates the match files without starting
anything, ``preflight`` reports every prerequisite, and ``launch_match``
starts the launcher process on Windows.  A dry run therefore works on any
operating system and is the recommended first check before a live match.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import checkpoints
from .config import project_root
from .modes import get_mode, mode_ids

TEAM_IDS = {"blue": 0, "orange": 1}
TEAM_LABELS = {0: "Blue", 1: "Orange"}


class LiveUnavailable(RuntimeError):
    pass


_LIVE: dict[str, dict] = {}


def _toml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _bot_toml(
    *,
    name: str,
    agent_id: str,
    python: str,
    bot_file: Path,
    repo: Path,
    model_path: Path,
    mode: str,
) -> str:
    arguments = [
        python,
        str(bot_file),
        "--model",
        str(model_path),
        "--mode",
        mode,
    ]
    run_command = " ".join(_toml_quote(argument) for argument in arguments)
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
    human_team: str = "blue",
    fill: str = "psyonix",
    human: bool = True,
) -> Path:
    """Write ``match.toml`` and one bot TOML per policy car.

    The policy always controls a full team (``spec.team_size`` cars).  The
    human team is completed with Psyonix bots (default) or additional policy
    bots, so a 2v2 or 3v3 match really has the requested team size instead of
    a lone human against a full bot team.  Set ``human=False`` to watch a
    policy-only exhibition match.
    """
    if mode not in mode_ids():
        raise LiveUnavailable(f"unsupported mode {mode!r}")
    human_team = human_team.strip().lower()
    if human_team not in TEAM_IDS:
        raise LiveUnavailable("human_team must be 'blue' or 'orange'")
    fill = fill.strip().lower()
    if fill not in {"psyonix", "policy"}:
        raise LiveUnavailable("fill must be 'psyonix' or 'policy'")

    spec = get_mode(mode)
    repo = project_root()
    bot_file = repo / "rlbot" / "bot.py"
    directory.mkdir(parents=True, exist_ok=True)

    human_side = TEAM_IDS[human_team]
    policy_side = 1 - human_side
    cars: list[str] = []

    if human:
        cars.append(
            "[[cars]]\n"
            'type = "Human"\n'
            f"team = {_toml_quote(TEAM_LABELS[human_side])}"
        )

    policy_index = 0
    for team in ([policy_side] if human else [0, 1]):
        for _ in range(spec.team_size):
            bot_config = directory / f"bot-{policy_index}.toml"
            bot_config.write_text(
                _bot_toml(
                    name=f"RLAI {spec.id} {policy_index}",
                    agent_id=f"rlai/{mode}-{policy_index}",
                    python=python,
                    bot_file=bot_file,
                    repo=repo,
                    model_path=model_path,
                    mode=mode,
                ),
                encoding="utf-8",
            )
            cars.append(
                "[[cars]]\n"
                f"config_file = {_toml_quote(str(bot_config))}\n"
                f"team = {_toml_quote(TEAM_LABELS[team])}\n"
                "auto_start = true"
            )
            policy_index += 1

    if human and spec.team_size > 1:
        filler_count = spec.team_size - 1
        for index in range(filler_count):
            if fill == "policy":
                bot_config = directory / f"bot-{policy_index}.toml"
                bot_config.write_text(
                    _bot_toml(
                        name=f"RLAI teammate {index}",
                        agent_id=f"rlai/{mode}-{policy_index}",
                        python=python,
                        bot_file=bot_file,
                        repo=repo,
                        model_path=model_path,
                        mode=mode,
                    ),
                    encoding="utf-8",
                )
                cars.append(
                    "[[cars]]\n"
                    f"config_file = {_toml_quote(str(bot_config))}\n"
                    f"team = {_toml_quote(TEAM_LABELS[human_side])}\n"
                    "auto_start = true"
                )
                policy_index += 1
            else:
                cars.append(
                    "[[cars]]\n"
                    'type = "psyonix"\n'
                    f"team = {_toml_quote(TEAM_LABELS[human_side])}\n"
                    'skill = "AllStar"\n'
                    f'name = {_toml_quote(f"RLAI fill {index + 1}")}'
                )

    # RLBot's GameMode enum uses Psyonix's spelling "Soccar".
    game_mode = "Heatseeker" if spec.heatseeker else "Soccar"
    mutators = 'boost_amount = "UnlimitedBoost"\n' if spec.heatseeker else ""
    match = f"""[rlbot]
launcher = "Epic"
auto_start_agents = true
wait_for_agents = true

[match]
game_mode = {_toml_quote(game_mode)}
game_map_upk = {_toml_quote("Stadium_P")}
skip_replays = true
instant_start = true
existing_match_behavior = "Restart"
enable_rendering = true
enable_state_setting = true
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
                "human_team": human_team,
                "fill": fill,
                "human": human,
                "created_at": time.time(),
                "config": str(path),
                "bot_configs": sorted(str(p) for p in directory.glob("bot-*.toml")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _unique_directory(root: Path, mode: str) -> Path:
    base = f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    candidate = root / "live" / base
    suffix = 1
    while candidate.exists():
        candidate = root / "live" / f"{base}-{suffix}"
        suffix += 1
    return candidate


def prepare_match(
    paths: dict[str, Path],
    model_id: str | None,
    mode: str,
    *,
    human_team: str = "blue",
    fill: str = "psyonix",
    human: bool = True,
) -> dict:
    """Resolve a model and write all match files without launching anything."""
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
        raise LiveUnavailable(
            "No trained model available. Train or promote a checkpoint first, "
            "or select a model explicitly."
        )
    model_path = paths["models"] / manifest["path"]
    if not model_path.exists():
        raise LiveUnavailable(f"checkpoint file is missing: {model_path}")
    directory = _unique_directory(paths["root"], mode)
    config = write_match_config(
        mode=mode,
        model_path=model_path,
        python=sys.executable,
        directory=directory,
        human_team=human_team,
        fill=fill,
        human=human,
    )
    return {
        "mode": mode,
        "model_id": manifest["id"],
        "model": str(model_path),
        "config": str(config),
        "directory": str(directory),
        "log": str(directory / "launch.log"),
        "human_team": human_team.lower(),
        "fill": fill.lower(),
        "human": human,
    }


def rlbot_server_info() -> dict:
    """Locate a running or installed RLBotServer without starting it."""
    try:
        from rlbot.managers import gateway
        from rlbot.managers.match import RLBOT_SERVER_NAME
    except Exception as exc:  # pragma: no cover - depends on installation
        return {"available": False, "running": False, "error": str(exc)}
    try:
        process, port = gateway.find_server_process(RLBOT_SERVER_NAME)
    except Exception as exc:  # pragma: no cover - psutil failures
        process, port = None, None
        error = str(exc)
    else:
        error = None
    if process is not None:
        return {
            "available": True,
            "running": True,
            "path": None,
            "port": port,
            "pid": int(process.pid),
        }
    path = Path.cwd() / RLBOT_SERVER_NAME
    if not path.exists():
        found = shutil.which(RLBOT_SERVER_NAME)
        if found:
            path = Path(found)
    if not path.exists() and os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            bin_dir = Path(local) / "RLBot5" / "bin"
            path = bin_dir / "RLBotServer.exe"
            for name in ("RLBotServer.exe", "RLBotServer"):
                candidate = bin_dir / name
                if candidate.exists():
                    path = candidate
                    break
    return {
        "available": bool(path.exists()),
        "running": False,
        "path": str(path) if path.exists() else None,
        "port": port,
        "error": error,
    }


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        process = psutil.Process(int(pid))
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except Exception:
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError):
            return False


def list_matches(paths: dict[str, Path], limit: int = 20) -> list[dict]:
    """Recent live matches with launcher liveness and the last log line."""
    root = paths["root"] / "live"
    if not root.exists():
        return []
    matches: list[dict] = []
    for directory in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        record_path = directory / "launch.json"
        if not directory.is_dir() or not record_path.exists():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        record["directory"] = str(directory)
        record["config_exists"] = (directory / "match.toml").exists()
        record["alive"] = _pid_alive(record.get("pid"))
        log_path = directory / "launch.log"
        record["log"] = str(log_path)
        if log_path.exists():
            lines = [
                line.strip()
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            record["last_log"] = lines[-1] if lines else ""
        matches.append(record)
        if len(matches) >= limit:
            break
    return matches


def preflight(paths: dict[str, Path]) -> dict:
    """Environment checks for a reliable first live test on Windows."""
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "required": required})

    is_windows = platform.system() == "Windows"
    add("windows", is_windows, f"platform is {platform.system()}", required=True)
    try:
        import rlbot

        add(
            "rlbot",
            True,
            f"rlbot {getattr(rlbot, '__version__', 'unknown')} importable",
            True,
        )
    except Exception as exc:
        add("rlbot", False, f"rlbot import failed: {exc}", True)
    server = rlbot_server_info()
    add(
        "rlbot server",
        bool(server.get("available")),
        (
            f"running on port {server.get('port')}"
            if server.get("running")
            else f"not running; executable={server.get('path') or 'not found'}"
        ),
        required=is_windows,
    )
    bot_file = project_root() / "rlbot" / "bot.py"
    add("bot entrypoint", bot_file.exists(), str(bot_file), True)
    add(
        "launcher entrypoint",
        (project_root() / "src" / "rlai" / "rlbot" / "run.py").exists(),
        "python -m rlai.rlbot.run",
        True,
    )
    models = checkpoints.list_models(paths["models"])
    stable = checkpoints.stable_model(paths["models"])
    best = checkpoints.best_model(paths["models"])
    launchable = bool(stable or best)
    add(
        "model",
        launchable,
        (
            f"{len(models)} checkpoint(s); stable="
            f"{stable['id'] if stable else None} best={best['id'] if best else None}"
            + ("" if launchable else "; promote one or pass --model-id")
        ),
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
        from rlbot.config import load_match_config  # noqa: F401

        add("match config parser", True, "rlbot.config.load_match_config", True)
    except Exception as exc:  # pragma: no cover - depends on installation
        add("match config parser", False, str(exc), True)
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
        "server": server,
    }


def validate_match_config(path: Path) -> dict:
    """Parse a generated TOML with RLBot's own loader.

    This is what catches enum typos such as ``game_mode = "Soccer"`` before a
    real launch.  Returns a JSON-friendly report instead of raising so the
    GUI can display it.
    """
    try:
        from rlbot.config import load_match_config
    except Exception as exc:  # pragma: no cover - rlbot is a hard dependency
        return {"ok": None, "parser": None, "detail": f"rlbot unavailable: {exc}"}
    try:
        config = load_match_config(Path(path))
    except Exception as exc:
        return {
            "ok": False,
            "parser": "rlbot.config.load_match_config",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": True,
        "parser": "rlbot.config.load_match_config",
        "game_mode": getattr(config.game_mode, "name", str(config.game_mode)),
        "players": len(config.player_configurations),
        "detail": f"{len(config.player_configurations)} players accepted",
    }


def _update_launch_record(directory: Path, **updates) -> None:
    path = directory / "launch.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        record = {}
    record.update(updates)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def launch_match(
    paths: dict[str, Path],
    model_id: str | None,
    mode: str,
    *,
    human_team: str = "blue",
    fill: str = "psyonix",
    human: bool = True,
    dry_run: bool = False,
    timeout: float = 0.0,
) -> dict:
    if not dry_run and platform.system() != "Windows":
        raise LiveUnavailable(
            "Live match launch requires Windows with Rocket League (Epic) and RLBot v5 "
            "installed. This environment can only run the headless simulator. Use the "
            "documented Windows procedure (or --dry-run to inspect the files)."
        )
    record = prepare_match(
        paths,
        model_id,
        mode,
        human_team=human_team,
        fill=fill,
        human=human,
    )
    _update_launch_record(Path(record["directory"]), status="prepared")
    if dry_run:
        validation = validate_match_config(Path(record["config"]))
        record["validation"] = validation
        record["launched"] = False
        record["dry_run"] = True
        _update_launch_record(
            Path(record["directory"]), status="prepared", validation=validation
        )
        if validation.get("ok") is False:
            raise LiveUnavailable(
                "RLBot rejected the generated match config: "
                f"{validation.get('detail')} (files kept in {record['directory']})"
            )
        return record
    try:
        import rlbot  # noqa: F401
    except Exception as exc:
        raise LiveUnavailable(f"rlbot is not installed in this environment: {exc}") from exc
    server = rlbot_server_info()
    if not server.get("available"):
        raise LiveUnavailable(
            "RLBotServer was not found. Install RLBot v5 and run the setup check again. "
            f"Details: {server.get('error') or 'no RLBotServer executable'}"
        )
    config = Path(record["config"])
    log_path = Path(record["log"])
    env = os.environ.copy()
    env["RLAI_MODEL"] = record["model"]
    env["RLAI_MODE"] = mode
    env["PYTHONPATH"] = str(project_root() / "src")
    env["RLAI_DATA_DIR"] = str(paths["root"])
    command = [sys.executable, "-u", "-m", "rlai.rlbot.run", "--config", str(config)]
    if timeout and timeout > 0:
        command += ["--timeout", str(float(timeout))]
    try:
        with open(log_path, "ab") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(project_root()),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
    except OSError as exc:
        _update_launch_record(
            Path(record["directory"]), status="failed", error=str(exc)
        )
        raise LiveUnavailable(f"could not start RLBot launcher: {exc}") from exc
    record.update({"pid": process.pid, "launched": True, "status": "launching"})
    _update_launch_record(
        Path(record["directory"]),
        pid=process.pid,
        status="launching",
        started_at=time.time(),
        launcher_command=command,
        timeout=float(timeout),
    )
    _LIVE[str(process.pid)] = record
    return record
