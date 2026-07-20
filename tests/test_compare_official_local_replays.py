from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.compare_official_local_replays import analyze_match, summarize


def _player(player_id: int, team_id: int, x: float, radius: float) -> dict[str, object]:
    return {
        "player_id": player_id,
        "team_id": team_id,
        "alive": radius > 0.0,
        "blobs": (
            [{"blob_id": 0, "pos": [x, 10.0], "radius": radius}]
            if radius > 0.0
            else []
        ),
    }


def _move(player_id: int, x: float, y: float = 0.0) -> dict[str, object]:
    return {
        "event_type": "move_player",
        "player_id": player_id,
        "direction": {"x": x, "y": y},
        "split": False,
    }


def _snapshot(player_id: int, x: float, radius: float) -> dict[str, object]:
    player = _player(player_id, 73 if player_id == 0 else 9, x, radius)
    return {
        "event_type": "event_player_moved",
        "player_id": player_id,
        "alive": player["alive"],
        "blobs": player["blobs"],
    }


def test_analyze_match_separates_fragment_loss_death_and_resource_mass(
    tmp_path: Path,
) -> None:
    events = [
        {
            "event_type": "event_game_started",
            "max_rounds": 3,
            "players": [_player(0, 73, 10.0, 1.0), _player(1, 9, 12.0, 1.0)],
        },
        {
            "event_type": "event_virus_spawned",
            "viruses": [{"virus_id": 4, "radius": 1.5}],
        },
        _move(0, 1.0),
        _move(1, -1.0),
        {"event_type": "event_food_eaten", "player_id": 0, "food_ids": [1, 2]},
        {"event_type": "event_virus_consumed", "player_id": 0, "virus_id": 4},
        {
            "event_type": "event_player_eaten",
            "eater_player_id": 0,
            "eaten_player_id": 1,
            "eaten_radius": 0.5,
            "eaten_player_alive": True,
        },
        _snapshot(0, 11.0, math.sqrt(3.545)),
        _snapshot(1, 12.0, 0.8),
        _move(0, 1.0),
        _move(1, -1.0),
        {
            "event_type": "event_player_eaten",
            "eater_player_id": 1,
            "eaten_player_id": 0,
            "eaten_radius": 0.5,
            "eaten_player_alive": True,
        },
        _snapshot(0, 12.0, 1.0),
        _snapshot(1, 11.0, 1.0),
        _move(0, 1.0),
        _move(1, -1.0),
        {
            "event_type": "event_player_eaten",
            "eater_player_id": 1,
            "eaten_player_id": 0,
            "eaten_radius": 1.0,
            "eaten_player_alive": False,
        },
        _snapshot(0, 0.0, 0.0),
        _snapshot(1, 10.0, math.sqrt(2.0)),
        {"event_type": "event_player_won", "player_id": 1},
    ]
    replay = tmp_path / "match-1-replay.json"
    replay.write_text(json.dumps(events))

    row = analyze_match(replay, team_id=73, terminal_horizon=2)

    assert row.captured_fragments == 1
    assert row.captured_mass == 0.25
    assert row.eliminations == 0
    assert row.lost_fragments == 2
    assert row.lost_mass == 1.25
    assert row.full_deaths == 1
    assert row.terminal_lost_fragments == 2
    assert not row.final_alive
    assert not row.terminal_stalled
    assert row.resources["food"]["count"] == 2
    assert math.isclose(float(row.resources["food"]["gross_mass"]), 0.045)
    assert row.resources["virus"] == {"count": 1, "gross_mass": 2.25}
    assert row.resources["enemy"] == {"count": 1, "gross_mass": 0.25}


def test_terminal_stall_uses_physical_motion_not_command_presence(tmp_path: Path) -> None:
    events: list[dict[str, object]] = [
        {
            "event_type": "event_game_started",
            "max_rounds": 3,
            "players": [_player(0, 73, 10.0, 1.0), _player(1, 9, 12.0, 1.0)],
        }
    ]
    for _ in range(3):
        events.extend((_move(0, 1.0), _move(1, -1.0)))
        events.extend((_snapshot(0, 10.0, 1.0), _snapshot(1, 12.0, 1.0)))
    events.append({"event_type": "event_player_won", "player_id": 0})
    replay = tmp_path / "match-2-replay.json"
    replay.write_text(json.dumps(events))

    row = analyze_match(
        replay,
        player_id=0,
        terminal_horizon=3,
        stalled_distance=0.1,
    )
    summary = summarize((row,))

    assert row.terminal_command_rounds == 3
    assert row.terminal_zero_commands == 0
    assert row.terminal_center_distance == 0.0
    assert row.terminal_stalled
    assert summary["terminal"]["stalled_match_rate"] == 1.0
