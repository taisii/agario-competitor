from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_strategy_outcomes import analyze_replay  # noqa: E402


def test_analyze_replay_tracks_middle_mass_and_player_eating(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "game.json"
    replay.write_text(
        json.dumps(
            [
                {
                    "event_type": "event_game_started",
                    "max_rounds": 3,
                    "players": [
                        {
                            "player_id": 0,
                            "alive": True,
                            "blobs": [{"radius": 1.0}],
                        }
                    ],
                },
                {
                    "event_type": "move_player",
                    "player_id": 0,
                },
                {
                    "event_type": "event_player_eaten",
                    "eater_player_id": 0,
                    "eaten_player_id": 1,
                    "eaten_radius": 2.0,
                },
                {
                    "event_type": "event_player_moved",
                    "player_id": 0,
                    "alive": True,
                    "blobs": [{"radius": 2.0}],
                },
                {
                    "event_type": "move_player",
                    "player_id": 0,
                },
                {
                    "event_type": "event_player_eaten",
                    "eater_player_id": 1,
                    "eaten_player_id": 0,
                    "eaten_radius": 1.0,
                },
                {
                    "event_type": "event_player_moved",
                    "player_id": 0,
                    "alive": True,
                    "blobs": [{"radius": 1.5}],
                },
                {
                    "event_type": "move_player",
                    "player_id": 0,
                },
                {
                    "event_type": "event_player_moved",
                    "player_id": 0,
                    "alive": False,
                    "blobs": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    metrics = analyze_replay(replay, player_id=0)

    assert metrics.captures == 1
    assert metrics.captured_mass == 4.0
    assert metrics.blobs_lost == 1
    assert metrics.lost_mass == 1.0
    assert metrics.middle_mean_mass == 2.25
    assert metrics.deaths == 1
    assert metrics.final_mass == 0.0
