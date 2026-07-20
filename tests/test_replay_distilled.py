from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.replay_distilled import (  # noqa: E402
    DEFAULT_MAX_TEACHER_CORRECTION_DEGREES,
    REPLAY_BASE_FEATURE_COUNT,
    REPLAY_DIRECTION_WEIGHTS,
    REPLAY_REGIME_DIRECTION_WEIGHTS,
    REPLAY_TEACHER_SOURCE_MATCHES,
    ReplayDistilledStrategy,
    _rotate_toward,
    _weighted_direction,
)
from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402


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


def test_teacher_correction_can_be_suppressed_near_threats(monkeypatch) -> None:
    monkeypatch.setenv("REPLAY_DISTILLED_CORRECTION_SAFETY_MARGIN", "100")
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0)
    predator = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(39.0, 30.0),
        radius=3.0,
    )
    state = SimpleNamespace(
        me=SimpleNamespace(player_id=0, blobs={0: own}),
        visible_blobs=[predator],
        visible_food=[],
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=0,
        max_rounds=1400,
    )
    context = StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(),
    )

    decision = ReplayDistilledStrategy().choose(context)

    assert decision.diagnostics["replay_teacher_correction_suppressed"] is True
    assert decision.diagnostics["replay_teacher_correction_degrees"] == 0.0
