from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
import strategies.event_driven as event_driven  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.event_driven import (  # noqa: E402
    SAFETY_DIRECTION_INDEX,
    EventDrivenStaticSearchStrategy,
    ProjectedFragment,
    TrackedTarget,
    _capture_cascade,
    _one_step_attack_reach,
    _prepare_enemy_reachability_cache,
    _project_one_step_fragments,
    _shield_action,
    _split_attack_reach,
    _tracked_event_direction,
)
from strategies.features import player_speed  # noqa: E402


def _visible(
    *,
    player_id: int,
    pos: tuple[float, float],
    radius: float,
    blob_id: int = 0,
) -> VisibleBlobModel:
    return VisibleBlobModel(
        player_id=player_id,
        team_id=player_id,
        blob_id=blob_id,
        pos=pos,
        radius=radius,
    )


def _context(
    own: tuple[BlobModel, ...],
    *,
    round_number: int = 1,
    enemies: tuple[VisibleBlobModel, ...] = (),
    foods: tuple[FoodModel, ...] = (),
    own_player_id: int = 0,
    rankings: tuple[int, ...] | None = None,
    view_center: tuple[float, float] | None = None,
    vision_size: float = 20.0,
) -> StrategyContext:
    total_mass = sum(blob.radius * blob.radius for blob in own)
    center = (
        sum(blob.pos[0] * blob.radius * blob.radius for blob in own) / total_mass,
        sum(blob.pos[1] * blob.radius * blob.radius for blob in own) / total_mass,
    )
    enemy_player_ids = tuple(dict.fromkeys(int(enemy.player_id) for enemy in enemies))
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=own_player_id,
            blobs={blob.blob_id: blob for blob in own},
        ),
        visible_blobs=list(enemies),
        visible_food=list(foods),
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=round_number,
        rankings=list(rankings or (own_player_id, *enemy_player_ids)),
        view_center=view_center or center,
        vision_size=vision_size,
    )
    return StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(),
    )


def test_quiet_static_growth_has_no_heavy_planner() -> None:
    strategy = EventDrivenStaticSearchStrategy()
    own = (BlobModel(blob_id=0, pos=(10.0, 10.0), radius=1.0),)
    food = FoodModel(food_id=1, pos=(14.0, 10.0))

    decision = strategy.choose(_context(own, foods=(food,)))

    assert not hasattr(strategy, "_planner")
    assert decision.reason == "static_backbone"
    assert decision.direction[0] > 0.0
    assert decision.diagnostics["planner_calls"] == 0


def test_event_split_is_sent_only_when_entering_opportunity() -> None:
    strategy = EventDrivenStaticSearchStrategy()
    own = (BlobModel(blob_id=0, pos=(10.0, 10.0), radius=4.0),)
    prey = _visible(player_id=1, pos=(16.0, 10.0), radius=1.0)

    decisions = tuple(
        strategy.choose(_context(own, enemies=(prey,), round_number=round_number))
        for round_number in range(1, 6)
    )

    assert decisions[0].split
    assert not any(decision.split for decision in decisions[1:])
    assert decisions[0].reason == "split_prey"
    assert decisions[0].diagnostics["planner_calls"] == 0


def test_split_tracking_distinguishes_same_player_fragments_by_continuity() -> None:
    strategy = EventDrivenStaticSearchStrategy()
    own = (BlobModel(blob_id=0, pos=(10.0, 10.0), radius=4.0),)
    right = _visible(player_id=1, pos=(16.0, 10.0), radius=1.0)
    left = _visible(player_id=1, pos=(4.0, 10.0), radius=1.0)

    first = strategy.choose(_context(own, enemies=(right,), round_number=1))
    second = strategy.choose(_context(own, enemies=(left,), round_number=2))

    assert first.split and first.direction[0] > 0.0
    assert second.split and second.direction[0] < 0.0


def test_boundary_split_bait_is_probed_without_splitting() -> None:
    strategy = EventDrivenStaticSearchStrategy()
    own = (
        BlobModel(
            blob_id=0,
            pos=(50.383, 17.079),
            radius=math.sqrt(61.0059),
        ),
    )
    prey = _visible(
        player_id=6,
        pos=(56.275, 3.725),
        radius=3.721,
    )
    context = _context(
        own,
        enemies=(prey,),
        rankings=(0, 6, 1, 7, 3, 5, 2, 4),
        view_center=(50.0, 17.079),
    )

    decision = strategy.choose(context)

    assert not decision.split
    assert decision.reason == "prey_probe"
    assert decision.direction[1] < 0.0


def test_hidden_mass_probe_is_not_repeated_on_same_spatial_target() -> None:
    strategy = EventDrivenStaticSearchStrategy()
    own = (
        BlobModel(
            blob_id=0,
            pos=(50.383, 17.079),
            radius=math.sqrt(61.0059),
        ),
    )
    prey = _visible(player_id=6, pos=(56.275, 3.725), radius=3.721)
    first = strategy.choose(
        _context(
            own,
            enemies=(prey,),
            round_number=1,
            rankings=(0, 6),
            view_center=(50.0, 17.079),
        )
    )
    second = strategy.choose(
        _context(
            own,
            enemies=(prey,),
            round_number=10,
            rankings=(0, 6),
            view_center=(50.0, 17.079),
        )
    )

    assert first.reason == "prey_probe"
    assert second.reason != "prey_probe"
    assert not second.split


def test_split_remains_aggressive_when_child_lands_deep_in_view() -> None:
    strategy = EventDrivenStaticSearchStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=math.sqrt(61.0059)),)
    prey = _visible(player_id=6, pos=(42.0, 30.0), radius=3.721)

    decision = strategy.choose(
        _context(
            own,
            enemies=(prey,),
            rankings=(0, 6),
            view_center=(40.0, 30.0),
        )
    )

    assert decision.split
    assert decision.reason == "split_prey"


def test_visible_same_player_mass_prevents_false_hidden_bait_veto() -> None:
    strategy = EventDrivenStaticSearchStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    prey = _visible(player_id=1, pos=(36.0, 30.0), radius=1.0, blob_id=0)
    visible_main = _visible(
        player_id=1,
        pos=(20.0, 30.0),
        radius=3.8,
        blob_id=1,
    )

    decision = strategy.choose(
        _context(
            own,
            enemies=(prey, visible_main),
            rankings=(0, 1),
            view_center=(30.0, 30.0),
        )
    )

    assert decision.split
    assert decision.reason == "split_prey"


def test_wall_blocked_hunter_does_not_chase_unreachable_corner_prey() -> None:
    strategy = EventDrivenStaticSearchStrategy()
    own = tuple(
        BlobModel(blob_id=index, pos=(55.0, 55.0), radius=3.0) for index in range(16)
    )
    prey = _visible(player_id=1, pos=(59.1, 59.1), radius=0.9)

    decision = strategy.choose(_context(own, enemies=(prey,)))

    assert decision.reason == "static_backbone"


def test_safety_rejected_split_is_not_chased_as_if_it_executed() -> None:
    strategy = EventDrivenStaticSearchStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    prey = _visible(player_id=1, pos=(36.0, 30.0), radius=1.0)
    predator = _visible(player_id=2, pos=(38.0, 30.0), radius=3.2)

    rejected = strategy.choose(_context(own, enemies=(prey, predator), round_number=1))
    follow_up = strategy.choose(_context(own, enemies=(prey, predator), round_number=2))

    assert not rejected.split
    assert rejected.reason == "reachable_safety_override"
    assert follow_up.reason != "split_prey"


def test_reachable_shield_handles_symmetric_pincer() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    enemies = (
        _visible(player_id=1, pos=(27.5, 30.0), radius=1.5),
        _visible(player_id=2, pos=(32.5, 30.0), radius=1.5),
    )

    result = _shield_action(
        own=own,
        enemies=enemies,
        nominal=(1.0, 0.0),
        split=False,
        arena_size=60.0,
    )

    assert abs(result.direction[1]) > 0.5
    assert result.retained_mass == 1.0
    assert not result.catastrophe


def test_reachable_shield_checks_post_split_children() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    predator = _visible(player_id=1, pos=(38.0, 30.0), radius=3.2)

    result = _shield_action(
        own=own,
        enemies=(predator,),
        nominal=(1.0, 0.0),
        split=True,
        arena_size=60.0,
    )

    assert not result.split
    assert math.isclose(result.retained_mass, 16.0)


def test_capture_cascade_grows_predator_before_rechecking_large_fragment() -> None:
    own = (
        BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0, merge_cooldown=10),
        BlobModel(blob_id=1, pos=(26.0, 30.0), radius=1.0, merge_cooldown=10),
        BlobModel(blob_id=2, pos=(34.0, 30.0), radius=1.0, merge_cooldown=10),
    )
    enemies = (
        _visible(player_id=1, pos=(29.5, 30.0), radius=math.sqrt(10.7)),
        _visible(player_id=2, pos=(30.5, 30.0), radius=math.sqrt(10.7)),
    )
    fragments = _project_one_step_fragments(
        own=own,
        direction=(0.0, 1.0),
        split=False,
        arena_size=60.0,
    )

    captured = _capture_cascade(fragments, enemies)

    assert captured == frozenset(range(3))


def test_capture_cascade_uses_one_coherent_enemy_direction() -> None:
    fragments = (
        ProjectedFragment(source_index=0, pos=(28.2, 30.0), radius=1.0),
        ProjectedFragment(source_index=1, pos=(31.8, 30.0), radius=1.0),
    )
    enemy = _visible(player_id=1, pos=(30.0, 30.0), radius=1.5)

    captured = _capture_cascade(fragments, (enemy,))

    assert len(captured) == 1


def test_cached_enemy_projections_preserve_capture_result() -> None:
    own = tuple(
        BlobModel(
            blob_id=index,
            pos=(27.0 + index % 4, 27.0 + index // 4),
            radius=1.0,
            merge_cooldown=10,
        )
        for index in range(16)
    )
    enemies = (
        _visible(player_id=1, pos=(30.0, 27.0), radius=2.2),
        _visible(player_id=2, pos=(32.0, 29.0), radius=2.2),
    )
    cache = _prepare_enemy_reachability_cache(enemies, 60.0)

    for direction in ((1.0, 0.0), (0.0, 1.0), (-1.0, -1.0)):
        fragments = _project_one_step_fragments(
            own=own,
            direction=direction,
            split=False,
            arena_size=60.0,
        )

        assert _capture_cascade(
            fragments,
            enemies,
            arena_size=60.0,
            enemy_cache=cache,
        ) == _capture_cascade(fragments, enemies, arena_size=60.0)


def test_expanded_shield_reuses_fixed_enemy_projections(monkeypatch) -> None:
    own = tuple(
        BlobModel(
            blob_id=index,
            pos=(25.0 + index % 4 * 1.3, 25.0 + index // 4 * 1.3),
            radius=1.0,
            merge_cooldown=10,
        )
        for index in range(16)
    )
    enemies = tuple(
        _visible(
            player_id=index + 1,
            pos=(30.0 + index % 2 * 2.0, 27.0 + index // 2 * 2.0),
            radius=2.2,
        )
        for index in range(4)
    )
    fixed_calls: Counter[tuple[int, int, int, bool]] = Counter()
    original = event_driven._project_enemy_eaters

    def counted_projection(**kwargs):
        direction = kwargs["direction"]
        key = (round(direction[0] * 1000), round(direction[1] * 1000))
        if key in SAFETY_DIRECTION_INDEX:
            enemy = kwargs["enemy"]
            fixed_calls[(enemy.player_id, key[0], key[1], kwargs["split"])] += 1
        return original(**kwargs)

    monkeypatch.setattr(event_driven, "_project_enemy_eaters", counted_projection)

    result = _shield_action(
        own=own,
        enemies=enemies,
        nominal=(1.0, 0.0),
        split=False,
        arena_size=60.0,
    )

    assert result.direction != (1.0, 0.0)
    # Each fixed enemy scenario is projected once for the nominal pass and
    # once into the cache. Expanded own-direction candidates reuse the cache.
    assert max(fixed_calls.values()) <= 2


def test_split_reach_does_not_double_count_parent_speed() -> None:
    expected = max(4.0 + player_speed(4.0), _split_attack_reach(4.0))

    assert math.isclose(_one_step_attack_reach(4.0, 1.0), expected)


def test_target_tracking_rejects_reused_public_blob_id_on_opposite_side() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    reused_id = _visible(
        player_id=1,
        blob_id=0,
        pos=(20.0, 30.0),
        radius=1.0,
    )
    target = TrackedTarget("prey", (40.0, 30.0), player_id=1, radius=1.0)

    direction = _tracked_event_direction(
        context=_context(own, enemies=(reused_id,)),
        target=target,
        fallback=(1.0, 0.0),
    )

    assert direction is None
