from __future__ import annotations

import sys
from pathlib import Path

BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from runtime import run_configured_strategy  # noqa: E402


if __name__ == "__main__":
    run_configured_strategy("replay_dominance")
