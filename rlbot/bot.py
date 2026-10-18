"""Entry point used by RLBot to start the bot process.

RLBot runs this file with the working directory set to this folder.  Model
selection happens through the RLAI_MODEL / RLAI_MODE environment variables,
which the GUI launcher sets for the whole match process tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlai.rlbot.bot import main  # noqa: E402

if __name__ == "__main__":
    main()
