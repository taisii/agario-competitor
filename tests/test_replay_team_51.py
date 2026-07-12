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
from strategies.replay_team_51 import (  # noqa: E402
    MIN_SPLIT_WALL_CLEARANCE,
    ReplayTeam51Strategy,
    SAFE_ANGLE_STEP,
)


def _observation(
    *,
    round_number: int = 400,
    own_pos: tuple[float, float] = (20.0, 20.0),
    own_radius: float = 2.0,
    enemies: tuple[ImitationBlob, ...] = (),
    food: tuple[ImitationPoint, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=round_number,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=(
            ImitationBlob(
                own_pos[0],
                own_pos[1],
                own_radius,
                player_id=0,
                blob_id=0,
            ),
        ),
        visible_blobs=enemies,
        visible_food=food,
        visible_viruses=(),
    )


def test_team51_safe_direction_uses_sixteen_heading_grid() -> None:
    strategy = ReplayTeam51Strategy()
    food = (ImitationPoint(24.0, 22.0, entity_id=1),)

    decision = strategy.choose_observation(_observation(food=food))
    angle = math.atan2(decision.direction[1], decision.direction[0])

    assert math.isclose(angle / SAFE_ANGLE_STEP, round(angle / SAFE_ANGLE_STEP))
    assert decision.reason == "team51_safe_grid_inertia"


def test_team51_split_aims_exactly_at_safe_nearby_prey() -> None:
    strategy = ReplayTeam51Strategy()
    prey = ImitationBlob(25.0, 20.0, 1.0, player_id=3, blob_id=7)

    decision = strategy.choose_observation(_observation(enemies=(prey,)))

    assert decision.split
    assert decision.direction == (1.0, 0.0)
    assert decision.target_id == "3:7"
    assert decision.reason == "team51_exact_prey_split"


def test_team51_does_not_split_with_visible_predator() -> None:
    strategy = ReplayTeam51Strategy()
    prey = ImitationBlob(25.0, 20.0, 1.0, player_id=3, blob_id=7)
    predator = ImitationBlob(17.0, 20.0, 3.0, player_id=4, blob_id=1)

    decision = strategy.choose_observation(
        _observation(enemies=(prey, predator))
    )

    assert not decision.split
    assert decision.reason == "team51_predator_field"


def test_team51_does_not_split_inside_observed_wall_clearance() -> None:
    strategy = ReplayTeam51Strategy()
    own_x = MIN_SPLIT_WALL_CLEARANCE - 0.01
    prey = ImitationBlob(own_x + 4.0, 20.0, 0.9, player_id=3, blob_id=7)

    decision = strategy.choose_observation(
        _observation(own_pos=(own_x, 20.0), enemies=(prey,))
    )

    assert not decision.split


def test_team51_resets_stale_inertia_after_respawn_gap() -> None:
    strategy = ReplayTeam51Strategy()
    strategy._previous_direction = (-1.0, 0.0)
    strategy._last_round = 100
    food = (ImitationPoint(24.0, 20.0, entity_id=1),)

    decision = strategy.choose_observation(
        _observation(round_number=131, food=food)
    )

    assert decision.diagnostics["respawned"] is True
    assert decision.direction[0] > 0.0
