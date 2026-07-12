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
from strategies.replay_team_9 import (  # noqa: E402
    PROFILE,
    ReplayTeam9Strategy,
    SPLIT_RATE,
)


def _observation(
    *,
    own: tuple[ImitationBlob, ...] | None = None,
    enemies: tuple[ImitationBlob, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=700,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=own or (ImitationBlob(20.0, 20.0, 3.0, player_id=0, blob_id=0),),
        visible_blobs=enemies,
        visible_food=(ImitationPoint(24.0, 20.0, entity_id=1),),
        visible_viruses=(),
    )


def test_team9_profile_covers_all_five_matches() -> None:
    assert PROFILE.source_matches == (11646, 11673, 11716, 11719, 11756)


def test_team9_direction_is_unit_length() -> None:
    decision = ReplayTeam9Strategy().choose_observation(_observation())

    assert math.isclose(math.hypot(*decision.direction), 1.0)


def test_team9_split_candidate_requires_one_capable_blob_and_prey() -> None:
    prey = ImitationBlob(26.0, 20.0, 1.0, player_id=2, blob_id=4)

    candidate = ReplayTeam9Strategy._split_candidate(
        _observation(enemies=(prey,))
    )

    assert candidate is not None
    assert candidate[1] == prey


def test_team9_predator_suppresses_split_candidate() -> None:
    prey = ImitationBlob(26.0, 20.0, 1.0, player_id=2, blob_id=4)
    predator = ImitationBlob(18.0, 20.0, 4.0, player_id=3, blob_id=0)

    candidate = ReplayTeam9Strategy._split_candidate(
        _observation(enemies=(prey, predator))
    )

    assert candidate is None


def test_team9_sparse_split_roll_is_deterministic() -> None:
    kwargs = {
        "round_number": 700,
        "player_id": 0,
        "own_radius": 3.0,
        "prey_radius": 1.0,
        "prey_distance": 6.0,
    }
    first = ReplayTeam9Strategy._split_roll(**kwargs)
    second = ReplayTeam9Strategy._split_roll(**kwargs)

    assert first == second
    assert 0.0 <= first < 1.0
    assert 0.0 < SPLIT_RATE < 1.0
