from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.retained_growth import (  # noqa: E402
    GrowthAction,
    ProjectedBlob,
    RetainedValue,
    StaticRetainedGrowthStrategy,
    _prey_growth_value,
)


def _context(
    own: tuple[BlobModel, ...],
    *,
    enemies: tuple[VisibleBlobModel, ...] = (),
) -> StrategyContext:
    total_mass = sum(blob.radius * blob.radius for blob in own)
    me = SimpleNamespace(
        player_id=0,
        radius=math.sqrt(total_mass),
        alive=True,
        blobs={blob.blob_id: blob for blob in own},
    )
    state = SimpleNamespace(
        me=me,
        visible_blobs=list(enemies),
        visible_food=[],
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=10,
        max_rounds=1400,
        rankings=[0, 1],
    )
    return StrategyContext(game=SimpleNamespace(state=state), query=SimpleNamespace())


def test_catastrophic_actions_ignore_growth_and_maximise_escape_margin() -> None:
    action = GrowthAction((1.0, 0.0), reason="prey")
    closer = RetainedValue(action, 0.0, 100.0, True, 1, -2.0)
    farther = RetainedValue(action, 0.0, 0.0, True, 1, -0.5)

    assert farther.total > closer.total


def test_corner_prey_does_not_receive_repeated_absolute_mass_value() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(0.9, 59.1),
        radius=0.9,
    )
    # A large blob is clamped by both walls and cannot improve its gap.
    path = ProjectedBlob(
        source_index=0,
        start_x=6.5,
        start_y=53.5,
        end_x=6.5,
        end_y=53.5,
        radius=6.5,
        speed=0.5,
    )

    assert _prey_growth_value(path=path, enemy=prey, arena_size=60.0) == 0.0


def test_split_future_is_not_misreported_as_immediate_capture() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(10.0, 10.0),
        radius=0.9,
    )
    path = ProjectedBlob(
        source_index=0,
        start_x=5.0,
        start_y=10.0,
        end_x=5.2,
        end_y=10.0,
        radius=2.0,
        speed=0.2,
        split_action=True,
        split_ejected=True,
    )

    assert _prey_growth_value(path=path, enemy=prey, arena_size=60.0) == 0.0


def test_safe_immediate_split_capture_is_selected() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=3.0)
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(16.0, 10.0),
        radius=1.0,
    )

    decision = StaticRetainedGrowthStrategy().choose(
        _context((own,), enemies=(prey,))
    )

    assert decision.reason == "split_prey"
    assert decision.split
