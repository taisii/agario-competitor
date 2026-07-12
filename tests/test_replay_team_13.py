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
from strategies.replay_team_13 import (  # noqa: E402
    PROFILE,
    ReplayTeam13Strategy,
)


def _observation(
    *,
    own_radius: float = 2.64,
    enemies: tuple[ImitationBlob, ...] = (),
    viruses: tuple[ImitationPoint, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=574,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=(ImitationBlob(20.0, 20.0, own_radius, player_id=5, blob_id=0),),
        visible_blobs=enemies,
        visible_food=(ImitationPoint(24.0, 20.0, entity_id=1),),
        visible_viruses=viruses,
    )


def test_team13_profile_covers_all_three_matches() -> None:
    assert PROFILE.source_matches == (11681, 11694, 11753)


def test_team13_fitted_direction_is_unit_length() -> None:
    decision = ReplayTeam13Strategy().choose_observation(_observation())

    assert math.isclose(math.hypot(*decision.direction), 1.0)


def test_team13_splits_exactly_toward_close_safe_prey() -> None:
    prey = ImitationBlob(23.4, 20.0, 1.3, player_id=2, blob_id=7)

    decision = ReplayTeam13Strategy().choose_observation(
        _observation(enemies=(prey,))
    )

    assert decision.split
    assert decision.direction == (1.0, 0.0)
    assert decision.target_id == "2:7"


def test_team13_does_not_split_below_observed_mass_floor() -> None:
    prey = ImitationBlob(23.0, 20.0, 1.0, player_id=2, blob_id=7)

    target = ReplayTeam13Strategy._split_target(
        _observation(own_radius=2.5, enemies=(prey,))
    )

    assert target is None


def test_team13_predator_suppresses_close_prey_split() -> None:
    prey = ImitationBlob(23.0, 20.0, 1.0, player_id=2, blob_id=7)
    predator = ImitationBlob(18.0, 20.0, 4.0, player_id=3, blob_id=0)

    target = ReplayTeam13Strategy._split_target(
        _observation(enemies=(prey, predator))
    )

    assert target is None
