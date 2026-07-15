from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_mass_gain_sources import _extract_gain_timeline  # noqa: E402


def test_gain_timeline_attributes_resource_mass_and_peak() -> None:
    events = [
        {
            "event_type": "event_game_started",
            "players": [
                {
                    "player_id": 1,
                    "blobs": [{"radius": 1.0}],
                }
            ],
        },
        {
            "event_type": "event_virus_spawned",
            "viruses": [{"virus_id": 4, "radius": 1.5}],
        },
        {"event_type": "move_player", "player_id": 1},
        {
            "event_type": "event_food_eaten",
            "player_id": 1,
            "food_ids": [1, 2],
        },
        {
            "event_type": "event_virus_consumed",
            "player_id": 1,
            "virus_id": 4,
        },
        {
            "event_type": "event_player_eaten",
            "eater_player_id": 1,
            "eaten_radius": 2.0,
        },
        {
            "event_type": "event_player_moved",
            "player_id": 1,
            "blobs": [{"radius": 3.0}],
        },
    ]

    gains, peaks = _extract_gain_timeline(events)

    assert math.isclose(gains[0][1]["food"].mass, 0.045)
    assert gains[0][1]["virus"].mass == 2.25
    assert gains[0][1]["enemy"].mass == 4.0
    assert peaks[1] == 9.0
