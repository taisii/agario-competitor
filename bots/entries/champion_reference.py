from __future__ import annotations

import os
import sys
from pathlib import Path

BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

# Deliberately expensive reference profile for strength experiments before
# distilling improvements into the submission-safe defaults.
os.environ.setdefault("BOT_STRATEGY", "champion")
os.environ.setdefault("BOT_CHAMPION_DEPTH", "11")
os.environ.setdefault("BOT_CHAMPION_WIDTH", "32")
os.environ.setdefault("BOT_CHAMPION_ANGLES", "48")
os.environ.setdefault("BOT_METRICS_ENABLED", "0")

from my_bot import main


if __name__ == "__main__":
    main()
