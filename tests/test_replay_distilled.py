from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.replay_distilled import (  # noqa: E402
    DEFAULT_MAX_TEACHER_CORRECTION_DEGREES,
    REPLAY_BASE_FEATURE_COUNT,
    REPLAY_DIRECTION_WEIGHTS,
    REPLAY_REGIME_DIRECTION_WEIGHTS,
    REPLAY_TEACHER_SOURCE_MATCHES,
    _replay_feature_vectors,
    _rotate_toward,
    _weighted_direction,
)
from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_imitation import (  # noqa: E402
    direction_feature_vectors,
    observation_from_context,
    observation_regime,
)


def _angle_degrees(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    dot = max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))
    return math.degrees(math.acos(dot))


def test_distilled_weight_shapes_match_runtime_features() -> None:
    expected = 1 + REPLAY_BASE_FEATURE_COUNT

    assert len(REPLAY_DIRECTION_WEIGHTS) == expected
    assert len(REPLAY_REGIME_DIRECTION_WEIGHTS) == 8
    assert all(
        len(weights) == expected
        for weights in REPLAY_REGIME_DIRECTION_WEIGHTS
    )
    assert len(REPLAY_TEACHER_SOURCE_MATCHES) == 31
    assert DEFAULT_MAX_TEACHER_CORRECTION_DEGREES == 2.5


def test_submission_safe_features_match_training_features() -> None:
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=0,
            blobs={
                0: BlobModel(
                    blob_id=0,
                    pos=(8.0, 12.0),
                    radius=2.0,
                    merge_cooldown=3,
                ),
                1: BlobModel(
                    blob_id=1,
                    pos=(10.0, 13.0),
                    radius=1.0,
                    merge_cooldown=0,
                ),
            },
        ),
        visible_blobs=[
            VisibleBlobModel(
                player_id=1,
                team_id=1,
                blob_id=0,
                pos=(13.0, 12.0),
                radius=3.0,
                merge_cooldown=0,
            ),
            VisibleBlobModel(
                player_id=2,
                team_id=2,
                blob_id=0,
                pos=(6.0, 17.0),
                radius=0.5,
                merge_cooldown=0,
            ),
        ],
        visible_food=[
            FoodModel(food_id=4, pos=(7.0, 15.0)),
            FoodModel(food_id=5, pos=(15.0, 10.0)),
        ],
        visible_viruses=[
            VirusModel(virus_id=8, pos=(9.0, 18.0), radius=1.5),
        ],
        map=SimpleNamespace(size=60.0),
        round=42,
        max_rounds=1400,
    )
    context = StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(),
    )
    previous = (0.6, -0.8)

    actual, actual_regime = _replay_feature_vectors(context, previous)
    observation = observation_from_context(context)
    expected = direction_feature_vectors(observation, previous)

    assert actual_regime == observation_regime(observation)
    assert len(actual) == len(expected)
    for actual_vector, expected_vector in zip(actual, expected):
        assert actual_vector == pytest.approx(expected_vector)


def test_weighted_direction_uses_semantic_residual_feature() -> None:
    features = ((0.0, 1.0), (1.0, 0.0))

    direction = _weighted_direction(
        (2.0, 1.0),
        features,
        fallback=(1.0, 0.0),
    )

    assert direction[0] > 0.0
    assert direction[1] > direction[0]


def test_teacher_correction_is_angle_bounded() -> None:
    source = (1.0, 0.0)
    target = (0.0, 1.0)

    corrected = _rotate_toward(source, target, math.radians(15.0))

    assert math.isclose(_angle_degrees(source, corrected), 15.0)
    assert corrected[0] > 0.0
    assert corrected[1] > 0.0


def test_zero_teacher_correction_preserves_semantic_direction() -> None:
    source = (0.6, 0.8)

    assert _rotate_toward(source, (-1.0, 0.0), 0.0) == source
