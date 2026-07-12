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
from strategies.replay_team_29 import (  # noqa: E402
    SOURCE_MATCHES,
    ReplayTeam29Strategy,
)


def _observation(
    *,
    own: tuple[ImitationBlob, ...] | None = None,
    visible_blobs: tuple[ImitationBlob, ...] = (),
    food: tuple[ImitationPoint, ...] = (),
    viruses: tuple[ImitationPoint, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=500,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=own
        if own is not None
        else (ImitationBlob(10.0, 10.0, 2.0, player_id=0, blob_id=0),),
        visible_blobs=visible_blobs,
        visible_food=food,
        visible_viruses=viruses,
    )


def test_team29_source_matches_cover_every_observed_replay() -> None:
    assert SOURCE_MATCHES == (11646, 11679, 11745, 11757)


def test_team29_chases_food_nearest_to_mass_center() -> None:
    own = (
        ImitationBlob(10.0, 10.0, 3.0, player_id=0, blob_id=0),
        ImitationBlob(20.0, 10.0, 1.0, player_id=0, blob_id=1),
    )
    # The mass center is (11, 10), so the north food is nearer than the east one.
    food = (
        ImitationPoint(11.0, 12.0, entity_id=1),
        ImitationPoint(15.0, 10.0, entity_id=2),
    )

    direction = ReplayTeam29Strategy._direction(
        _observation(own=own, food=food)
    )

    assert math.isclose(direction[0], 0.0, abs_tol=1e-12)
    assert math.isclose(direction[1], 1.0)


def test_team29_keeps_food_priority_with_predator_prey_and_virus() -> None:
    predator = ImitationBlob(12.0, 10.0, 4.0, player_id=1, blob_id=0)
    prey = ImitationBlob(9.0, 10.0, 1.0, player_id=2, blob_id=0)
    virus = ImitationPoint(10.0, 9.0, radius=1.5, entity_id=3)
    food = (ImitationPoint(10.0, 12.0, entity_id=4),)

    direction = ReplayTeam29Strategy._direction(
        _observation(
            visible_blobs=(predator, prey),
            food=food,
            viruses=(virus,),
        )
    )

    assert direction == (0.0, 1.0)


def test_team29_returns_no_target_without_visible_food() -> None:
    assert ReplayTeam29Strategy._direction(_observation()) == (0.0, 0.0)


def test_team29_never_splits_even_when_large_with_visible_prey() -> None:
    own = (ImitationBlob(10.0, 10.0, 5.0, player_id=0, blob_id=0),)
    prey = ImitationBlob(11.0, 10.0, 1.0, player_id=2, blob_id=0)
    observation = _observation(own=own, visible_blobs=(prey,))

    assert not ReplayTeam29Strategy._split_decision(observation)
