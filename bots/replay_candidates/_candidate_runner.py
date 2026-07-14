"""Shared runtime bootstrap for archived replay-clone evaluation adapters."""

import sys
from pathlib import Path


BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from runtime import run_replay_candidate  # noqa: E402


def run_candidate(team_id: int) -> None:
    run_replay_candidate(team_id)
