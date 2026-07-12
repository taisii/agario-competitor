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
from strategies.replay_team_25 import (  # noqa: E402
    SOURCE_MATCHES,
    ReplayTeam25Strategy,
)


def _observation(
    *,
    own_radius: float = 2.0,
    visible_blobs: tuple[ImitationBlob, ...] = (),
    food: tuple[ImitationPoint, ...] = (),
    viruses: tuple[ImitationPoint, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=500,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=(
            ImitationBlob(10.0, 10.0, own_radius, player_id=0, blob_id=0),
        ),
        visible_blobs=visible_blobs,
        visible_food=food,
        visible_viruses=viruses,
    )


def test_team25_source_matches_cover_all_observed_replays() -> None:
    assert SOURCE_MATCHES == (11719, 11725, 11752, 11756)


def test_team25_chases_nearest_food_without_predator() -> None:
    food = (
        ImitationPoint(18.0, 10.0, entity_id=1),
        ImitationPoint(10.0, 12.0, entity_id=2),
    )

    direction, target_kind, reason = ReplayTeam25Strategy._direction(
        _observation(food=food)
    )

    assert direction == (0.0, 1.0)
    assert target_kind == "food"
    assert reason == "team25_nearest_food"


def test_team25_predator_field_overrides_food() -> None:
    predator = ImitationBlob(12.0, 10.0, 3.0, player_id=1, blob_id=0)
    food = (ImitationPoint(12.0, 10.0, entity_id=1),)

    direction, target_kind, reason = ReplayTeam25Strategy._direction(
        _observation(visible_blobs=(predator,), food=food)
    )

    assert direction[0] < 0.0
    assert math.isclose(direction[1], 0.0, abs_tol=1e-12)
    assert target_kind == "escape"
    assert reason == "team25_predator_field"


def test_team25_combines_all_predators_in_escape_field() -> None:
    east = ImitationBlob(12.0, 10.0, 3.0, player_id=1, blob_id=0)
    north = ImitationBlob(10.0, 14.0, 3.0, player_id=2, blob_id=0)

    direction, _, _ = ReplayTeam25Strategy._direction(
        _observation(visible_blobs=(east, north))
    )

    assert direction[0] < 0.0
    assert direction[1] < 0.0
    assert abs(direction[0]) > abs(direction[1])


def test_team25_ignores_safe_prey_and_virus_when_food_is_visible() -> None:
    prey = ImitationBlob(11.0, 10.0, 1.0, player_id=1, blob_id=0)
    food = (ImitationPoint(10.0, 12.0, entity_id=1),)
    virus = (ImitationPoint(8.0, 10.0, radius=1.5, entity_id=2),)

    direction, target_kind, _ = ReplayTeam25Strategy._direction(
        _observation(visible_blobs=(prey,), food=food, viruses=virus)
    )

    assert direction == (0.0, 1.0)
    assert target_kind == "food"


def test_team25_has_no_target_without_predator_or_food() -> None:
    direction, target_kind, reason = ReplayTeam25Strategy._direction(_observation())

    assert direction == (0.0, 0.0)
    assert target_kind == "none"
    assert reason == "team25_no_target"


def test_team25_never_splits_even_with_large_blob_and_visible_prey() -> None:
    prey = ImitationBlob(11.0, 10.0, 1.0, player_id=1, blob_id=0)
    observation = _observation(
        own_radius=5.0,
        visible_blobs=(prey,),
    )

    assert not ReplayTeam25Strategy._split_decision(observation)
