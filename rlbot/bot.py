"""Entry point used by RLBot to start the bot process.

The launcher passes the checkpoint and mode explicitly, so a match does not
depend on environment variables inherited from RLBotServer.  Everything is
also written to ``data/logs/bot.log`` to make a failed live test debuggable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rocket League AI RLBot agent")
    parser.add_argument("--model", default=None, help="checkpoint path or model id")
    parser.add_argument("--mode", default=None, help="game mode id, e.g. soccer-1v1")
    parser.add_argument("--data-dir", default=None, help="override RLAI_DATA_DIR")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="file log level",
    )
    return parser.parse_args(argv)


def _log_file() -> Path:
    data = Path(os.environ.get("RLAI_DATA_DIR", ROOT / "data"))
    logs = data / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs / "bot.log"


def _setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("rlai.bot.entry")
    logger.setLevel(getattr(logging, level))
    path = _log_file()
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == path
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(stream)
    return logger


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.data_dir:
        os.environ["RLAI_DATA_DIR"] = args.data_dir
    if args.model:
        os.environ["RLAI_MODEL"] = args.model
    if args.mode:
        os.environ["RLAI_MODE"] = args.mode
    logger = _setup_logging(args.log_level)
    logger.info(
        "bot process starting pid=%s model=%s mode=%s cwd=%s",
        os.getpid(),
        os.environ.get("RLAI_MODEL"),
        os.environ.get("RLAI_MODE"),
        Path.cwd(),
    )
    if args.model and not Path(args.model).exists():
        logger.error("checkpoint does not exist: %s", args.model)
        print(f"checkpoint does not exist: {args.model}", file=sys.stderr)
        return 2
    try:
        from rlai.rlbot.bot import main as run_bot

        run_bot()
    except KeyboardInterrupt:
        logger.info("bot process interrupted")
        return 0
    except Exception:
        logger.error("bot process failed\n%s", traceback.format_exc())
        traceback.print_exc()
        return 1
    logger.info("bot process stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
