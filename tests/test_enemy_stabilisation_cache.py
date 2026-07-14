from __future__ import annotations

import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.local_tactical_search import LocalTacticalSearchStrategy  # noqa: E402
from strategies.receding_horizon import (  # noqa: E402
    Action,
    EnemyBlob,
    OwnBlob,
    SearchNode,
)
from strategies.world_transition import (  # noqa: E402
    CompleteJointCommand,
    PlayerCommand,
)


def _random_enemy_layout(rng: random.Random) -> tuple[EnemyBlob, ...]:
    result = []
    for player_id in range(1, rng.randint(2, 5)):
        center_x = rng.uniform(8.0, 52.0)
        center_y = rng.uniform(8.0, 52.0)
        for blob_id in range(rng.randint(1, 16)):
            radius = rng.uniform(0.35, 3.0)
            result.append(
                EnemyBlob(
                    player_id=player_id,
                    blob_id=blob_id,
                    x=min(60.0 - radius, max(radius, center_x + rng.uniform(-4, 4))),
                    y=min(60.0 - radius, max(radius, center_y + rng.uniform(-4, 4))),
                    radius=radius,
                    direction=(rng.uniform(-1, 1), rng.uniform(-1, 1)),
                    stale_rounds=rng.randint(0, 2),
                    merge_cooldown=rng.choice((0, 1, 8, 30)),
                    eject_vx=rng.uniform(-0.5, 0.5),
                    eject_vy=rng.uniform(-0.5, 0.5),
                )
            )
    return tuple(result)


def test_turn_cache_preserves_exact_randomised_enemy_stabilisation() -> None:
    rng = random.Random(0x51AB1E)
    strategy = LocalTacticalSearchStrategy()

    for _ in range(250):
        enemies = _random_enemy_layout(rng)
        expected = strategy._stabilise_enemy_blobs_uncached(enemies, 60.0)
        actual = strategy._stabilise_enemy_blobs(enemies, 60.0)

        assert actual == expected
        # The second call must return the immutable memoized object, not merely
        # a numerically close reconstruction.
        assert strategy._stabilise_enemy_blobs(enemies, 60.0) is actual


def test_turn_cache_preserves_exact_randomised_joint_transitions() -> None:
    rng = random.Random(0xC0FFEE)
    cached = LocalTacticalSearchStrategy()
    reference = LocalTacticalSearchStrategy()
    reference._stabilise_enemy_blobs = (  # type: ignore[method-assign]
        reference._stabilise_enemy_blobs_uncached
    )
    cached._own_player_id = reference._own_player_id = 0

    for sample_id in range(100):
        own_radius = rng.uniform(0.8, 3.5)
        own = OwnBlob(
            blob_id=0,
            x=rng.uniform(10.0, 50.0),
            y=rng.uniform(10.0, 50.0),
            radius=own_radius,
            merge_cooldown=rng.choice((0, 5)),
        )
        enemies = tuple(
            enemy for enemy in _random_enemy_layout(rng) if enemy.blob_id < 6
        )
        node = SearchNode(
            own_blobs=(own,),
            enemies=enemies,
            score=0.0,
            first_direction=(1.0, 0.0),
            first_split=False,
            first_reason="parity",
            last_direction=(1.0, 0.0),
        )
        angle = rng.uniform(-math.pi, math.pi)
        action = Action(
            (math.cos(angle), math.sin(angle)),
            split=rng.random() < 0.2,
            reason="parity",
        )
        commands = {
            0: PlayerCommand(action.direction, split=action.split),
        }
        for player_id in {enemy.player_id for enemy in enemies}:
            enemy_angle = rng.uniform(-math.pi, math.pi)
            commands[player_id] = PlayerCommand(
                (math.cos(enemy_angle), math.sin(enemy_angle)),
                split=rng.random() < 0.15,
            )
        joint = CompleteJointCommand.build(
            live_player_ids=set(commands),
            commands=commands,
        )
        foods = tuple(
            FoodModel(
                food_id=index,
                pos=(rng.uniform(5.0, 55.0), rng.uniform(5.0, 55.0)),
            )
            for index in range(4)
        )
        viruses = (
            VirusModel(
                virus_id=sample_id,
                pos=(rng.uniform(5.0, 55.0), rng.uniform(5.0, 55.0)),
                radius=1.5,
            ),
        )
        kwargs = dict(
            node=node,
            action=action,
            foods=foods,
            viruses=viruses,
            arena_size=60.0,
            first_step=True,
            joint_command=joint,
        )

        assert cached._joint_physical_step(**kwargs) == reference._joint_physical_step(
            **kwargs
        )
