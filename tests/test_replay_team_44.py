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
from strategies.replay_team_44 import (  # noqa: E402
    ANGLE_STEP,
    MAX_TURN_STEPS,
    ReplayTeam44Strategy,
)


def _observation(
    *,
    own_radius: float = 3.0,
    visible_blobs: tuple[ImitationBlob, ...] = (),
    food: tuple[ImitationPoint, ...] = (),
    viruses: tuple[ImitationPoint, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=400,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=(ImitationBlob(10.0, 10.0, own_radius, player_id=0, blob_id=0),),
        visible_blobs=visible_blobs,
        visible_food=food,
        visible_viruses=viruses,
    )


def test_team44_direction_uses_32_bins_and_limits_inertial_turn() -> None:
    strategy = ReplayTeam44Strategy()
    strategy._previous_direction = (1.0, 0.0)

    direction = strategy._inertial_direction((-1.0, 0.0))
    angle = math.atan2(direction[1], direction[0])

    assert math.isclose(abs(angle), MAX_TURN_STEPS * ANGLE_STEP)
    assert math.isclose(angle / ANGLE_STEP, round(angle / ANGLE_STEP))


def test_team44_never_splits_with_visible_predator() -> None:
    strategy = ReplayTeam44Strategy()
    predator = ImitationBlob(13.0, 10.0, 4.0, player_id=1, blob_id=0)
    food = tuple(ImitationPoint(11.0 + index * 0.2, 10.0, entity_id=index) for index in range(8))

    split, reason = strategy._split_decision(
        _observation(visible_blobs=(predator,), food=food),
        (1.0, 0.0),
    )

    assert not split
    assert reason == "predator_visible"


def test_team44_splits_for_safe_reachable_prey() -> None:
    strategy = ReplayTeam44Strategy()
    prey = ImitationBlob(15.0, 10.0, 1.0, player_id=1, blob_id=0)

    split, reason = strategy._split_decision(
        _observation(visible_blobs=(prey,)),
        (1.0, 0.0),
    )

    assert split
    assert reason == "safe_split_capture"


def test_team44_requires_observed_mass_floor_for_split() -> None:
    strategy = ReplayTeam44Strategy()
    prey = ImitationBlob(12.0, 10.0, 0.5, player_id=1, blob_id=0)

    split, reason = strategy._split_decision(
        _observation(own_radius=1.5, visible_blobs=(prey,)),
        (1.0, 0.0),
    )

    assert not split
    assert reason == "below_observed_mass_floor"


def test_team44_splits_toward_edible_virus_when_children_remain_capable() -> None:
    strategy = ReplayTeam44Strategy()
    virus = ImitationPoint(15.0, 10.0, radius=1.5, entity_id=7)

    split, reason = strategy._split_decision(
        _observation(viruses=(virus,)),
        (1.0, 0.0),
    )

    assert split
    assert reason == "edible_virus_split"
