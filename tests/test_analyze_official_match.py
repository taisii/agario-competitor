from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.analyze_official_match import analyze


def _player(player_id: int, team_id: int, x: float, radius: float) -> dict[str, object]:
    return {
        "player_id": player_id,
        "team_id": team_id,
        "alive": True,
        "blobs": [
            {
                "blob_id": 0,
                "pos": [x, 10.0],
                "radius": radius,
                "merge_cooldown": 0,
            }
        ],
    }


def test_analyze_compares_capture_intent_and_outcome(tmp_path: Path) -> None:
    prey = _player(0, 73, 10.0, 1.0)
    hunter = _player(1, 58, 12.0, 2.0)
    events = [
        {
            "event_type": "event_game_started",
            "arena_size": 60.0,
            "vision_size": 20.0,
            "max_rounds": 1,
            "players": [prey, hunter],
        },
        {
            "event_type": "move_player",
            "player_id": 0,
            "direction": {"x": 1.0, "y": 0.0},
            "split": False,
        },
        {
            "event_type": "move_player",
            "player_id": 1,
            "direction": {"x": -1.0, "y": 0.0},
            "split": True,
        },
        {
            "event_type": "event_food_eaten",
            "player_id": 1,
            "blob_id": 0,
            "food_ids": [1, 2],
            "new_radius": 2.0,
        },
        {
            "event_type": "event_virus_consumed",
            "player_id": 1,
            "blob_id": 0,
            "virus_id": 1,
            "virus_pos": [12.0, 10.0],
            "pieces_created": 2,
        },
        {
            "event_type": "event_player_eaten",
            "eater_player_id": 1,
            "eater_blob_id": 0,
            "eater_pos": [11.0, 10.0],
            "eaten_player_id": 0,
            "eaten_blob_id": 0,
            "eaten_pos": [10.0, 10.0],
            "eater_radius": 2.2,
            "eaten_player_alive": False,
        },
        {
            "event_type": "event_player_moved",
            "player_id": 0,
            "alive": False,
            "blobs": [],
        },
        {
            "event_type": "event_player_moved",
            "player_id": 1,
            "alive": True,
            "blobs": hunter["blobs"],
        },
        {"event_type": "event_player_won", "player_id": 1},
    ]
    replay = tmp_path / "match-1-replay.json"
    replay.write_text(json.dumps(events))

    result = analyze(replay)
    by_player = {row["player_id"]: row for row in result["players"]}

    assert result["winner_player_id"] == 1
    assert by_player[1]["prey_opportunities"] == 1
    assert by_player[1]["prey_approaches"] == 1
    assert by_player[1]["prey_split_attempts"] == 1
    assert by_player[1]["kills"] == 1
    assert by_player[1]["food"] == 2
    assert by_player[1]["viruses"] == 1
    assert by_player[0]["death_rounds"] == [0]


def test_analyze_incomplete_match_reports_formation_without_a_winner(
    tmp_path: Path,
) -> None:
    player = _player(0, 73, 10.0, 1.0)
    fragmented_blobs = [
        {
            "blob_id": blob_id,
            "pos": [x, 10.0],
            "radius": 1.0,
            "merge_cooldown": 5,
        }
        for blob_id, x in enumerate((10.0, 12.0))
    ]
    events = [
        {
            "event_type": "event_game_started",
            "arena_size": 60.0,
            "vision_size": 20.0,
            "max_rounds": 1,
            "players": [player],
        },
        {
            "event_type": "event_player_moved",
            "player_id": 0,
            "alive": True,
            "blobs": fragmented_blobs,
        },
    ]
    replay = tmp_path / "match-2-replay.json"
    replay.write_text(json.dumps(events))

    result = analyze(replay)
    metrics = result["players"][0]

    assert result["winner_player_id"] is None
    assert result["winner_team_id"] is None
    assert metrics["final_blob_count"] == 2
    assert metrics["fragmented_fraction"] == 1.0
    assert math.isclose(metrics["final_extent_ratio"], math.sqrt(2.0))
    assert metrics["final_largest_blob_mass_share"] == 0.5
