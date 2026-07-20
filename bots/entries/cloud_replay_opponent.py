"""Run one clone from an empirically observed official seven-team cohort."""

from __future__ import annotations

import os
from pathlib import Path
import sys


BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from runtime import run_bot  # noqa: E402
from strategies.replay_opponents import create_cloud_replay_opponent  # noqa: E402


def _log_selection(strategy_name: str) -> None:
    print(
        f"cloud_replay_opponent={strategy_name}",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    target_slot = int(os.environ.get("BOT_CLOUD_TARGET_SLOT", "0"))
    run_bot(
        lambda: create_cloud_replay_opponent(
            target_slot=target_slot,
            on_selected=_log_selection,
        )
    )


if __name__ == "__main__":
    main()
