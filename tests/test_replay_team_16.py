from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.replay_imitation import (  # noqa: E402
    ImitationBlob,
    ImitationObservation,
    ImitationPoint,
)
from strategies.replay_team_16 import (  # noqa: E402
    PROFILE,
    ReplayTeam16Strategy,
    SPLIT_RATE,
)


def _observation(
    *,
    own: tuple[ImitationBlob, ...] | None = None,
    enemies: tuple[ImitationBlob, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=500,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=own or (ImitationBlob(20.0, 20.0, 3.0, player_id=1, blob_id=0),),
        visible_blobs=enemies,
        visible_food=(ImitationPoint(24.0, 20.0, entity_id=1),),
        visible_viruses=(),
    )


def test_team16_profile_covers_all_four_matches() -> None:
    assert PROFILE.source_matches == (11646, 11667, 11681, 11694, 13933, 13938, 13939)


def test_team16_direction_is_unit_length() -> None:
    decision = ReplayTeam16Strategy().choose_observation(_observation())

    assert math.isclose(math.hypot(*decision.direction), 1.0)


def test_team16_accepts_safe_capable_prey_candidate() -> None:
    prey = ImitationBlob(30.0, 20.0, 1.0, player_id=2, blob_id=4)

    candidate = ReplayTeam16Strategy._split_candidate(
        _observation(enemies=(prey,))
    )

    assert candidate is not None
    assert candidate[1] == prey


def test_team16_predator_suppresses_split_candidate() -> None:
    prey = ImitationBlob(30.0, 20.0, 1.0, player_id=2, blob_id=4)
    predator = ImitationBlob(18.0, 20.0, 4.0, player_id=3, blob_id=0)

    assert ReplayTeam16Strategy._split_candidate(
        _observation(enemies=(prey, predator))
    ) is None


def test_team16_sparse_split_roll_is_deterministic() -> None:
    kwargs = {
        "round_number": 500,
        "player_id": 1,
        "own_radius": 3.0,
        "prey_radius": 1.0,
        "prey_distance": 10.0,
    }
    first = ReplayTeam16Strategy._split_roll(**kwargs)
    second = ReplayTeam16Strategy._split_roll(**kwargs)

    assert first == second
    assert 0.0 <= first < 1.0
    assert 0.0 < SPLIT_RATE < 1.0
