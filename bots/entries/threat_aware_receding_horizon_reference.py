from __future__ import annotations

import sys
from pathlib import Path

BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from runtime import run_configured_strategy  # noqa: E402


if __name__ == "__main__":
    # Deliberately expensive reference profile for strength experiments before
    # distilling improvements into the submission-safe defaults.
    run_configured_strategy(
        "threat_aware_receding_horizon",
        environment_defaults={
            "BOT_RECEDING_HORIZON_DEPTH": "11",
            "BOT_RECEDING_HORIZON_WIDTH": "32",
            "BOT_RECEDING_HORIZON_ANGLES": "48",
            "BOT_METRICS_ENABLED": "0",
        },
    )
