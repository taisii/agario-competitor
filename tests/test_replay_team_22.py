from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_22 import (  # noqa: E402
    ANGLE_STEP_DEGREES,
    DANGER_SURFACE_GAP,
    PREY_CHASE_DISTANCE,
    ReplayTeam22Strategy,
)


def _game(
    own: tuple[BlobModel, ...],
    *,
    enemies: tuple[VisibleBlobModel, ...] = (),
    food: tuple[FoodModel, ...] = (),
) -> SimpleNamespace:
    total_mass = sum(blob.radius * blob.radius for blob in own)
    center_x = sum(blob.pos[0] * blob.radius * blob.radius for blob in own) / total_mass
    center_y = sum(blob.pos[1] * blob.radius * blob.radius for blob in own) / total_mass
    me = SimpleNamespace(
        player_id=0,
        x=center_x,
        y=center_y,
        radius=math.sqrt(total_mass),
        alive=True,
        blobs={blob.blob_id: blob for blob in own},
    )
    return SimpleNamespace(
        state=SimpleNamespace(
            me=me,
            visible_blobs=list(enemies),
            visible_food=list(food),
            visible_viruses=[],
            map=SimpleNamespace(size=60.0),
            round=10,
            max_rounds=1400,
            rankings=[0, 1],
        )
    )


def _choose(
    strategy: ReplayTeam22Strategy,
    game: SimpleNamespace,
):
    return strategy.choose(
        StrategyContext(game=game, query=SimpleNamespace())
    )


def test_team_22_moves_to_nearest_food_from_nearest_own_blob() -> None:
    far_blob = BlobModel(blob_id=0, pos=(5.0, 5.0), radius=1.0)
    near_blob = BlobModel(blob_id=1, pos=(20.0, 20.0), radius=1.0)
    food = FoodModel(food_id=7, pos=(22.5, 19.0))

    decision = _choose(
        ReplayTeam22Strategy(),
        _game((far_blob, near_blob), food=(food,)),
    )

    assert decision.direction == (2.5, -1.0)
    assert decision.target_kind == "food"
    assert decision.diagnostics["origin_blob_id"] == near_blob.blob_id
    assert decision.split is False


def test_team_22_chases_nearby_edible_enemy_before_food() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=1.2)
    prey = VisibleBlobModel(
        player_id=4,
        team_id=4,
        blob_id=3,
        pos=(10.0 + PREY_CHASE_DISTANCE, 10.0),
        radius=1.0,
    )
    food = FoodModel(food_id=7, pos=(10.0, 11.0))

    decision = _choose(
        ReplayTeam22Strategy(),
        _game((own,), enemies=(prey,), food=(food,)),
    )

    assert math.isclose(decision.direction[0], PREY_CHASE_DISTANCE)
    assert decision.direction[1] == 0.0
    assert decision.target_kind == "prey"
    assert decision.target_id == "4:3"
    assert decision.split is False


def test_team_22_ignores_edible_enemy_beyond_replay_chase_distance() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=1.2)
    prey = VisibleBlobModel(
        player_id=4,
        team_id=4,
        blob_id=3,
        pos=(10.0 + PREY_CHASE_DISTANCE + 0.01, 10.0),
        radius=1.0,
    )
    food = FoodModel(food_id=7, pos=(10.0, 11.0))

    decision = _choose(
        ReplayTeam22Strategy(),
        _game((own,), enemies=(prey,), food=(food,)),
    )

    assert decision.direction == (0.0, 1.0)
    assert decision.target_kind == "food"


def test_team_22_uses_unit_fifteen_degree_grid_in_danger() -> None:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0)
    predator = VisibleBlobModel(
        player_id=5,
        team_id=5,
        blob_id=2,
        pos=(30.0 + 1.0 + 2.0 + DANGER_SURFACE_GAP, 30.0),
        radius=2.0,
    )

    decision = _choose(
        ReplayTeam22Strategy(),
        _game((own,), enemies=(predator,)),
    )

    assert math.isclose(math.hypot(*decision.direction), 1.0)
    angle = math.degrees(math.atan2(decision.direction[1], decision.direction[0])) % 360
    assert math.isclose(angle % ANGLE_STEP_DEGREES, 0.0, abs_tol=1e-9)
    assert decision.direction[0] < 0.0
    assert decision.target_kind == "escape"
    assert math.isclose(
        decision.diagnostics["surface_gap"],
        DANGER_SURFACE_GAP,
        abs_tol=1e-9,
    )
    assert decision.split is False


def test_team_22_does_not_enter_danger_mode_above_surface_gap_threshold() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=1.0)
    predator = VisibleBlobModel(
        player_id=5,
        team_id=5,
        blob_id=2,
        pos=(10.0 + 1.0 + 2.0 + DANGER_SURFACE_GAP + 0.01, 10.0),
        radius=2.0,
    )
    food = FoodModel(food_id=7, pos=(10.0, 12.0))

    decision = _choose(
        ReplayTeam22Strategy(),
        _game((own,), enemies=(predator,), food=(food,)),
    )

    assert decision.direction == (0.0, 2.0)
    assert decision.target_kind == "food"
    assert decision.split is False
