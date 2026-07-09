from __future__ import annotations

import os
import sys
from pathlib import Path

BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))
os.environ.setdefault("BOT_STRATEGY", "beam_rl_tuned")

from my_bot import main


if __name__ == "__main__":
    main()
