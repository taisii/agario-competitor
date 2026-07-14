"""Offline-only entry for measuring an archived replay clone.

The active opponent panel uses dedicated ``replay_team_<id>.py`` entries.
This adapter intentionally requires an explicit team ID and is used only by
the strength evaluator to measure candidates before they are promoted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from runtime import run_replay_candidate  # noqa: E402


def main() -> None:
    try:
        team_id = int(os.environ["BOT_REPLAY_TEAM_ID"])
    except KeyError as exc:
        raise RuntimeError("BOT_REPLAY_TEAM_ID is required for replay_candidate") from exc
    run_replay_candidate(team_id)


if __name__ == "__main__":
    main()
