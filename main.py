#!/usr/bin/env python3
"""Convenience entry point for training and the other CLI commands.

``python3 main.py --steps 500000``
    Train 500,000 steps **in addition to** the newest checkpoint.  A
    checkpoint at step 2,000,000 continues to 2,500,000, and the learning-rate
    anneal covers the added 500,000 steps.  Without a checkpoint it is simply
    500,000 steps from scratch.

``python3 main.py --steps 500000 --plan-only``
    Show which checkpoint would be restored and what the target would be,
    without training anything.

``python3 main.py train --config-file configs/smoke-1v1.json``
    Full CLI syntax, identical to ``python -m rlai.cli``.

``python3 main.py serve``
    Start the web dashboard.

The script adds ``src/`` to ``sys.path`` when the package is not installed, so
a plain ``python3 main.py`` works from a checkout without ``pip install -e .``
(Torch and RocketSim still have to be installed).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
#: used for training commands when no --config-file was given
DEFAULT_CONFIG = ROOT / "configs" / "1v1-mastery-1v1.json"

#: sub-commands of rlai.cli; anything else is treated as an option of "train"
COMMANDS = {
    "serve",
    "train",
    "headless",
    "resume-plan",
    "report",
    "evaluate",
    "benchmark",
    "benchmarks",
    "live",
}
#: commands that need a configuration file
CONFIG_COMMANDS = {"train", "headless", "resume-plan"}


def _prepare_argv(argv: list[str]) -> list[str]:
    """Fill in the implicit sub-command and the default configuration file."""
    argv = list(argv)
    if not argv or argv[0].startswith("-"):
        # ``python3 main.py --steps 500000``: no sub-command means "train"
        argv = ["train", *argv]
    wants_help = any(item in {"-h", "--help"} for item in argv)
    if (
        argv[0] in CONFIG_COMMANDS
        and not wants_help
        and not any(item == "--config-file" or item.startswith("--config-file=") for item in argv)
    ):
        argv = [*argv, "--config-file", str(DEFAULT_CONFIG)]
        print(f"NOTE: no --config-file given; using {DEFAULT_CONFIG}", file=sys.stderr)
    return argv


def main(argv: list[str] | None = None) -> int:
    source = ROOT / "src"
    if source.is_dir():
        sys.path.insert(0, str(source))
    from rlai.cli import main as cli_main

    prepared = _prepare_argv(sys.argv[1:] if argv is None else argv)
    if prepared and prepared[0] in CONFIG_COMMANDS and not DEFAULT_CONFIG.is_file():
        print(f"ERROR: configuration file not found: {DEFAULT_CONFIG}", file=sys.stderr)
        print("Pass --config-file <path> or restore the examples in configs/", file=sys.stderr)
        return 2
    return int(cli_main(prepared))


if __name__ == "__main__":
    raise SystemExit(main())
