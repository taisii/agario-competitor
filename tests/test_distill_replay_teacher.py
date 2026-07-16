from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from distill_replay_teacher import _player_id_for_team  # noqa: E402


def test_player_id_for_team_uses_team_mapping() -> None:
    started = {
        "players": [
            {"player_id": 4, "team_id": 12},
            {"player_id": 1, "team_id": 73},
        ]
    }

    assert _player_id_for_team(started, 73) == 1
