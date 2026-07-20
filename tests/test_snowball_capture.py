from __future__ import annotations

from types import SimpleNamespace

from lib.models.blob_model import BlobModel, VisibleBlobModel
from strategies.base import StrategyContext, StrategyDecision
from strategies.snowball_capture import SnowballCaptureLayer


def _context(
    *,
    rank_first: bool = True,
    extra_enemy: VisibleBlobModel | None = None,
) -> StrategyContext:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=5.0)
    enemy = VisibleBlobModel(
        blob_id=0,
        player_id=1,
        team_id=1,
        pos=(34.5, 30.0),
        radius=1.5,
    )
    enemies = [enemy] + ([] if extra_enemy is None else [extra_enemy])
    state = SimpleNamespace(
        me=SimpleNamespace(player_id=0, blobs={0: own}),
        visible_blobs=enemies,
        visible_food=[],
        visible_viruses=[],
        rankings=[0, 1] if rank_first else [1, 0],
        map=SimpleNamespace(size=60.0),
        round=100,
    )
    return StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(),
    )


def _decision() -> StrategyDecision:
    return StrategyDecision(
        direction=(0.985, 0.174),
        reason="continue",
        diagnostics={},
    )


def test_secure_leader_capture_overrides_base_route() -> None:
    actual = SnowballCaptureLayer().adjust(_context(), _decision())

    assert actual.split is False
    assert actual.reason == "snowball_contact_capture"
    assert actual.target_id == "1:0"
    assert actual.diagnostics["snowball_capture_intervened"] is True
    assert actual.diagnostics["secured_one_step_mass"]["enemy"] > 0.0


def test_layer_does_not_spend_capital_when_not_leading() -> None:
    original = _decision()
    actual = SnowballCaptureLayer().adjust(
        _context(rank_first=False),
        original,
    )

    assert actual.direction == original.direction
    assert actual.split is False
    assert actual.diagnostics["snowball_capture_intervened"] is False


def test_layer_rejects_post_capture_predator_envelope() -> None:
    predator = VisibleBlobModel(
        blob_id=0,
        player_id=2,
        team_id=2,
        pos=(38.0, 30.0),
        radius=6.0,
    )
    actual = SnowballCaptureLayer().adjust(
        _context(extra_enemy=predator),
        _decision(),
    )

    assert actual.split is False
    assert actual.diagnostics["snowball_capture_intervened"] is False
