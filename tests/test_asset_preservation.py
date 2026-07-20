from __future__ import annotations

import math
from types import SimpleNamespace

from lib.models.blob_model import BlobModel, VisibleBlobModel
from strategies.asset_preservation import (
    AssetPreservationLayer,
    _asset_risk,
    _rotate_toward,
)
from strategies.base import StrategyContext, StrategyDecision


def _context(
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...] = (),
    *,
    round_number: int = 0,
    player_id: int = 0,
) -> StrategyContext:
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=player_id,
            blobs={blob.blob_id: blob for blob in own},
        ),
        visible_blobs=list(enemies),
        map=SimpleNamespace(size=60.0),
        round=round_number,
    )
    return StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(),
    )


def _predator(*, x: float = 40.0) -> VisibleBlobModel:
    return VisibleBlobModel(
        blob_id=0,
        player_id=1,
        team_id=1,
        pos=(x, 30.0),
        radius=4.0,
    )


def test_override_turn_is_bounded_without_replacing_base_route() -> None:
    bounded = _rotate_toward(
        (1.0, 0.0),
        (-1.0, 0.0),
        math.radians(30.0),
    )

    assert math.isclose(bounded[0], math.cos(math.radians(30.0)))
    assert math.isclose(abs(bounded[1]), math.sin(math.radians(30.0)))


def test_layer_is_observational_only_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ASSET_PRESERVATION_ENABLED", "0")
    layer = AssetPreservationLayer()
    single = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    fragmented = (
        BlobModel(blob_id=0, pos=(30.0, 29.0), radius=2.0),
        BlobModel(blob_id=1, pos=(30.0, 31.0), radius=2.0),
    )
    original = StrategyDecision(
        direction=(1.0, 0.0),
        split=True,
        reason="keep",
    )

    layer.adjust(_context(single), StrategyDecision(direction=(1.0, 0.0)))
    actual = layer.adjust(
        _context(fragmented, (_predator(x=38.0),), round_number=1),
        original,
    )

    assert actual.direction == original.direction
    assert actual.split is original.split
    assert actual.reason == original.reason
    assert actual.diagnostics["asset_preservation_enabled"] is False
    assert actual.diagnostics["asset_preservation_trigger"] == "virus"
    assert actual.diagnostics["asset_preservation_intervened"] is False


def test_layer_intervenes_only_after_observed_fragmentation(monkeypatch) -> None:
    monkeypatch.setenv("ASSET_PRESERVATION_ENABLED", "1")
    layer = AssetPreservationLayer()
    fragmented = (
        BlobModel(blob_id=0, pos=(30.0, 29.0), radius=2.0),
        BlobModel(blob_id=1, pos=(30.0, 31.0), radius=2.0),
    )
    original = StrategyDecision(
        direction=(1.0, 0.0),
        split=True,
        reason="attack",
    )

    actual = layer.adjust(_context(fragmented, (_predator(),)), original)

    assert actual.direction == original.direction
    assert actual.split is original.split
    assert actual.reason == original.reason
    assert actual.diagnostics["asset_preservation_trigger"] is None
    assert actual.diagnostics["asset_preservation_remaining"] == 0
    assert actual.diagnostics["asset_preservation_intervened"] is False


def test_layer_never_rewrites_direction_in_post_split_state(monkeypatch) -> None:
    monkeypatch.setenv("ASSET_PRESERVATION_ENABLED", "1")
    monkeypatch.setenv("ASSET_PRESERVATION_MAX_OVERRIDE_DEGREES", "180")
    monkeypatch.setenv("ASSET_PRESERVATION_MIN_SAVED_MASS_FRACTION", "0")
    layer = AssetPreservationLayer()
    single = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    fragmented = (
        BlobModel(blob_id=0, pos=(30.0, 29.0), radius=2.0),
        BlobModel(blob_id=1, pos=(30.0, 31.0), radius=2.0),
    )
    predator = _predator(x=46.0)

    layer.adjust(
        _context(single),
        StrategyDecision(direction=(1.0, 0.0), split=True),
    )
    actual = layer.adjust(
        _context(fragmented, (predator,), round_number=1),
        StrategyDecision(direction=(1.0, 0.0), split=True, reason="attack"),
    )

    assert math.isclose(actual.direction[0], 1.0)
    assert math.isclose(actual.direction[1], 0.0, abs_tol=1.0e-12)
    assert actual.split is False
    assert actual.reason == "post_fragment_split_veto"
    assert actual.diagnostics["asset_preservation_trigger"] == "own_split"
    assert actual.diagnostics["asset_preservation_intervened"] is True
    assert (
        actual.diagnostics["asset_preservation_reduces_valuable_exposure"]
        is True
    )
    assert (
        actual.diagnostics["asset_preservation_selected_danger"]
        < actual.diagnostics["asset_preservation_base_danger"]
    )


def test_layer_vetoes_split_when_post_split_shape_creates_exposure(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ASSET_PRESERVATION_ENABLED", "1")
    monkeypatch.setenv("ASSET_PRESERVATION_MIN_SAVED_MASS_FRACTION", "0")
    layer = AssetPreservationLayer()
    single = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    fragmented = (
        BlobModel(blob_id=0, pos=(30.0, 29.0), radius=2.0),
        BlobModel(blob_id=1, pos=(30.0, 31.0), radius=2.0),
    )

    layer.adjust(
        _context(single),
        StrategyDecision(direction=(1.0, 0.0), split=True),
    )
    actual = layer.adjust(
        _context(fragmented, (_predator(x=46.0),), round_number=1),
        StrategyDecision(direction=(1.0, 0.0), split=True, reason="attack"),
    )

    assert actual.direction == (1.0, 0.0)
    assert actual.split is False
    assert actual.reason == "post_fragment_split_veto"
    assert (
        actual.diagnostics["asset_preservation_reduces_valuable_exposure"]
        is True
    )
    assert actual.diagnostics["asset_preservation_suppressed_split"] is True


def test_layer_does_not_intervene_on_non_split_routes(monkeypatch) -> None:
    monkeypatch.setenv("ASSET_PRESERVATION_ENABLED", "1")
    monkeypatch.setenv("ASSET_PRESERVATION_MAX_OVERRIDE_DEGREES", "180")
    monkeypatch.setenv("ASSET_PRESERVATION_MIN_SAVED_MASS_FRACTION", "0")
    layer = AssetPreservationLayer()
    single = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    fragmented = (
        BlobModel(blob_id=0, pos=(30.0, 29.0), radius=2.0),
        BlobModel(blob_id=1, pos=(30.0, 31.0), radius=2.0),
    )

    layer.adjust(
        _context(single),
        StrategyDecision(direction=(1.0, 0.0), split=True),
    )
    first = layer.adjust(
        _context(fragmented, (_predator(x=42.0),), round_number=1),
        StrategyDecision(direction=(1.0, 0.0), reason="first"),
    )
    second = layer.adjust(
        _context(fragmented, (_predator(x=42.0),), round_number=2),
        StrategyDecision(direction=(1.0, 0.0), reason="second"),
    )

    assert first.diagnostics["asset_preservation_intervened"] is False
    assert first.reason == "first"
    assert second.reason == "second"
    assert second.direction == (1.0, 0.0)
    assert second.diagnostics["asset_preservation_window_intervened"] is False
    assert second.diagnostics["asset_preservation_intervened"] is False


def test_layer_preserves_split_when_secured_gain_covers_saved_mass(monkeypatch) -> None:
    monkeypatch.setenv("ASSET_PRESERVATION_ENABLED", "1")
    monkeypatch.setenv("ASSET_PRESERVATION_MIN_SAVED_MASS_FRACTION", "0")
    layer = AssetPreservationLayer()
    fragmented = (
        BlobModel(blob_id=0, pos=(30.0, 29.0), radius=2.0),
        BlobModel(blob_id=1, pos=(30.0, 31.0), radius=2.0),
    )
    decision = StrategyDecision(
        direction=(1.0, 0.0),
        split=True,
        reason="secured_capture",
        diagnostics={
            "secured_one_step_mass": {
                "enemy": 8.0,
                "virus": 0.0,
                "food": 0.0,
            }
        },
    )

    actual = layer.adjust(
        _context(fragmented, (_predator(x=46.0),), round_number=1),
        decision,
    )

    assert actual.split is True
    assert actual.reason == "secured_capture"
    assert actual.diagnostics["asset_preservation_secured_mass"] == 8.0
    assert actual.diagnostics["asset_preservation_intervened"] is False


def test_asset_risk_values_large_fragments_and_enemy_split_reach() -> None:
    own = (
        BlobModel(blob_id=0, pos=(30.0, 29.0), radius=2.0),
        BlobModel(blob_id=1, pos=(30.0, 31.0), radius=2.0),
    )
    enemies = (_predator(x=38.0),)

    toward = _asset_risk(
        own,
        enemies,
        direction=(1.0, 0.0),
        arena_size=60.0,
        horizon=3.0,
    )
    away = _asset_risk(
        own,
        enemies,
        direction=(-1.0, 0.0),
        arena_size=60.0,
        horizon=3.0,
    )

    assert toward.exposed_mass == 8.0
    assert away.exposed_mass == 8.0
    assert toward.valuable_exposed_mass == 8.0
    assert away.valuable_exposed_mass == 8.0
    assert away.weighted_danger < toward.weighted_danger
    assert away.minimum_margin > toward.minimum_margin


def test_layer_does_not_reroute_to_save_only_subunit_shards(monkeypatch) -> None:
    monkeypatch.setenv("ASSET_PRESERVATION_ENABLED", "1")
    layer = AssetPreservationLayer()
    single = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.3),)
    shards = (
        BlobModel(blob_id=0, pos=(30.0, 29.0), radius=0.9),
        BlobModel(blob_id=1, pos=(30.0, 31.0), radius=0.9),
    )

    layer.adjust(
        _context(single),
        StrategyDecision(direction=(1.0, 0.0), split=True),
    )
    actual = layer.adjust(
        _context(shards, (_predator(x=42.0),), round_number=1),
        StrategyDecision(direction=(1.0, 0.0), split=True, reason="grow"),
    )

    assert actual.direction == (1.0, 0.0)
    assert actual.split is True
    assert actual.reason == "grow"
    assert actual.diagnostics["asset_preservation_intervened"] is False
    assert actual.diagnostics["asset_preservation_base_valuable_exposed_mass"] == 0.0


def test_layer_ignores_small_savings_relative_to_total_assets(monkeypatch) -> None:
    monkeypatch.setenv("ASSET_PRESERVATION_ENABLED", "1")
    monkeypatch.setenv("ASSET_PRESERVATION_MAX_OVERRIDE_DEGREES", "180")
    monkeypatch.setenv("ASSET_PRESERVATION_MIN_SAVED_MASS_FRACTION", "0.20")
    layer = AssetPreservationLayer()
    single = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    fragmented = (
        BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.8),
        BlobModel(blob_id=1, pos=(34.0, 30.0), radius=1.0),
    )

    layer.adjust(
        _context(single),
        StrategyDecision(direction=(1.0, 0.0), split=True),
    )
    actual = layer.adjust(
        _context(fragmented, (_predator(x=40.0),), round_number=1),
        StrategyDecision(direction=(1.0, 0.0), reason="keep_opportunity"),
    )

    assert actual.direction == (1.0, 0.0)
    assert actual.reason == "keep_opportunity"
    assert actual.diagnostics["asset_preservation_intervened"] is False
    assert actual.diagnostics["asset_preservation_saved_mass_fraction"] < 0.20


def test_split_risk_projects_the_post_split_shape() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    enemies = (_predator(x=42.0),)

    unsplit = _asset_risk(
        own,
        enemies,
        direction=(1.0, 0.0),
        split=False,
        arena_size=60.0,
        horizon=3.0,
    )
    split = _asset_risk(
        own,
        enemies,
        direction=(1.0, 0.0),
        split=True,
        arena_size=60.0,
        horizon=3.0,
    )

    assert split.valuable_exposed_mass > unsplit.valuable_exposed_mass


def test_new_match_reset_does_not_inherit_fragmentation_window(monkeypatch) -> None:
    monkeypatch.setenv("ASSET_PRESERVATION_ENABLED", "1")
    layer = AssetPreservationLayer()
    single = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    fragmented = (
        BlobModel(blob_id=0, pos=(30.0, 29.0), radius=2.0),
        BlobModel(blob_id=1, pos=(30.0, 31.0), radius=2.0),
    )

    layer.adjust(
        _context(single, round_number=100),
        StrategyDecision(direction=(1.0, 0.0), split=True),
    )
    actual = layer.adjust(
        _context(fragmented, (_predator(),), round_number=0),
        StrategyDecision(direction=(1.0, 0.0), split=True, reason="new_match"),
    )

    assert actual.reason == "new_match"
    assert actual.diagnostics["asset_preservation_trigger"] is None
    assert actual.diagnostics["asset_preservation_remaining"] == 0
