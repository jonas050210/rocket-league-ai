"""Start an RLBot v5 match that loads the Rocket League AI bot.

Run this on the machine that has Rocket League installed::

    python -m rlai.rlbot.run --config data/live/.../match.toml

``--dry-run`` parses and validates the files with RLBot's own parser without
starting a server or the game.  That is the recommended check on Linux and
before the first Windows launch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def _summary(config) -> dict:
    players = []
    for player in config.player_configurations:
        variety = player.variety
        kind = type(variety).__name__
        entry = {"type": kind, "team": int(player.team)}
        name = getattr(variety, "name", None)
        if name:
            entry["name"] = str(name)
        agent_id = getattr(variety, "agent_id", None)
        if agent_id:
            entry["agent_id"] = str(agent_id)
        players.append(entry)
    return {
        "game_mode": getattr(config.game_mode, "name", str(config.game_mode)),
        "players": players,
        "auto_start_agents": bool(config.auto_start_agents),
        "wait_for_agents": bool(config.wait_for_agents),
        "launcher": getattr(config.launcher, "name", str(config.launcher)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch or validate an RLBot v5 match")
    parser.add_argument("--config", required=True, help="Path to the match TOML")
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Abort if the match does not end within N seconds (0 waits forever).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the match config and print a summary; do not start anything.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"match config not found: {config_path}", file=sys.stderr)
        return 2
    try:
        from rlbot.config import load_match_config
    except Exception as exc:
        print(f"RLBot is not installed: {exc}", file=sys.stderr)
        return 2
    try:
        parsed = load_match_config(config_path)
    except Exception as exc:
        print(f"invalid match config: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    summary = _summary(parsed)
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    try:
        from rlbot import flat
        from rlbot.managers import MatchManager
    except Exception as exc:
        print(f"RLBot manager is not available: {exc}", file=sys.stderr)
        return 2

    match_manager = MatchManager()
    started = time.monotonic()
    try:
        match_manager.start_match(config_path, wait_for_start=False)
        print("match configuration sent; waiting for the first packet", flush=True)
        deadline = started + args.timeout if args.timeout > 0 else None
        while match_manager.packet is None:
            if deadline is not None and time.monotonic() > deadline:
                print("timed out waiting for the first game packet", file=sys.stderr)
                return 3
            time.sleep(0.2)
        print("match started; press Ctrl+C to stop", flush=True)
        while True:
            packet = match_manager.packet
            if packet is not None and packet.match_info.match_phase == flat.MatchPhase.Ended:
                print("match ended", flush=True)
                break
            if deadline is not None and time.monotonic() > deadline:
                print("match timeout reached; shutting down", file=sys.stderr)
                return 3
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("interrupted; shutting down", flush=True)
        return 0
    except Exception as exc:
        print(f"could not start the match: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    finally:
        try:
            match_manager.shut_down()
        except Exception as exc:  # pragma: no cover - depends on RLBot state
            print(f"shutdown warning: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
