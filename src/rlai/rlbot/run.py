"""Start an RLBot v5 match that loads the Rocket League AI bot.

Run this on the machine that has Rocket League installed::

    python -m rlai.rlbot.run --config data/live/.../match.toml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import sleep

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch an RLBot v5 match")
    parser.add_argument("--config", required=True, help="Path to the match TOML")
    parser.add_argument(
        "--timeout", type=float, default=0.0, help="Optional abort timeout in seconds"
    )
    args = parser.parse_args()

    from rlbot import flat
    from rlbot.managers import MatchManager

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"match config not found: {config_path}", file=sys.stderr)
        return 2
    match_manager = MatchManager()
    match_manager.start_match(config_path)
    print("match started; press Ctrl+C to stop", flush=True)
    try:
        while True:
            packet = match_manager.packet
            if packet is not None and packet.match_info.match_phase == flat.MatchPhase.Ended:
                break
            sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        match_manager.shut_down()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
