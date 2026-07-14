"""Run one randomly selected specialised official-replay clone."""

from __future__ import annotations

import sys
from pathlib import Path


BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from runtime import run_bot  # noqa: E402
from strategies.replay_opponents import create_random_replay_opponent  # noqa: E402


def _log_selection(strategy_name: str) -> None:
    print(
        f"random_replay_opponent={strategy_name}",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    run_bot(
        lambda: create_random_replay_opponent(
            on_selected=_log_selection,
        )
    )


if __name__ == "__main__":
    main()
