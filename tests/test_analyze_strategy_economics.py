from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_strategy_economics import (  # noqa: E402
    PursuitEpisode,
    _distribution,
    _eaten_events,
    _episode_summary,
    _player_milestones,
)


def test_eaten_events_preserve_round_and_target_mass() -> None:
    events = [
        {"event_type": "event_game_started"},
        {"event_type": "move_player", "player_id": 1},
        {"event_type": "move_player", "player_id": 2},
        {
            "event_type": "event_player_eaten",
            "eater_player_id": 1,
            "eaten_player_id": 2,
            "eaten_radius": 1.5,
        },
        {"event_type": "move_player", "player_id": 1},
        {
            "event_type": "event_player_eaten",
            "eater_player_id": 2,
            "eaten_player_id": 1,
            "eaten_radius": 3.0,
        },
    ]

    rows = _eaten_events(events, eater_player_id=1)

    assert len(rows) == 1
    assert rows[0].round_number == 0
    assert rows[0].eaten_player_id == 2
    assert rows[0].mass == 2.25


def test_episode_summary_separates_probability_and_conditional_mass() -> None:
    episodes = (
        PursuitEpisode(
            eta_if_stationary=3.0,
            edge_clearance=1.0,
            split=False,
            target_mass=2.0,
            target_mass_by_horizon={40: 4.0},
            any_enemy_mass_by_horizon={40: 5.0},
        ),
        PursuitEpisode(
            eta_if_stationary=8.0,
            edge_clearance=3.0,
            split=True,
            target_mass=6.0,
            target_mass_by_horizon={40: 0.0},
            any_enemy_mass_by_horizon={40: 1.0},
        ),
    )

    summary = _episode_summary(episodes, horizon=40)

    assert summary["episodes"] == 2
    assert summary["target_capture_rate"] == 0.5
    assert summary["expected_target_mass_per_start"] == 2.0
    assert summary["mean_target_mass_given_capture"] == 4.0
    assert summary["expected_any_enemy_mass_per_start"] == 3.0
    assert math.isclose(summary["mean_offered_target_mass"], 4.0)


def test_player_milestones_capture_first_enemy_and_terminal_mass() -> None:
    events = [
        {
            "event_type": "event_game_started",
            "players": [
                {
                    "player_id": 1,
                    "blobs": [{"radius": 1.0}],
                },
                {
                    "player_id": 2,
                    "blobs": [{"radius": 1.0}],
                },
            ],
        },
        {"event_type": "move_player", "player_id": 1},
        {"event_type": "move_player", "player_id": 2},
        {
            "event_type": "event_player_eaten",
            "eater_player_id": 1,
            "eaten_player_id": 2,
            "eaten_radius": 1.0,
        },
        {
            "event_type": "event_player_moved",
            "player_id": 1,
            "blobs": [{"radius": math.sqrt(2.0)}],
        },
        {
            "event_type": "event_player_moved",
            "player_id": 2,
            "blobs": [],
        },
        {"event_type": "move_player", "player_id": 1},
        {
            "event_type": "event_player_moved",
            "player_id": 1,
            "blobs": [{"radius": 1.2}],
        },
        {"event_type": "event_player_won", "player_id": 1},
    ]

    milestones, terminal_round = _player_milestones(events)

    assert terminal_round == 1
    assert milestones[1].won
    assert milestones[1].first_enemy_capture_round == 0
    assert math.isclose(milestones[1].mass_after_first_enemy_capture or 0.0, 2.0)
    assert milestones[1].first_captured_enemy_mass == 1.0
    assert milestones[1].final_mass == 1.44
    assert milestones[2].first_enemy_capture_round is None
    assert milestones[2].final_mass == 0.0


def test_distribution_uses_linear_quartiles() -> None:
    summary = _distribution((1, 2, 3, 4))

    assert summary == {
        "count": 4,
        "mean": 2.5,
        "median": 2.5,
        "p25": 1.75,
        "p75": 3.25,
        "min": 1.0,
        "max": 4.0,
    }
