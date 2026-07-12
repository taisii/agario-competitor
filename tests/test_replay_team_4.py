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
from strategies.replay_team_4 import (  # noqa: E402
    FARM_SPLIT_RATE,
    PROFILE,
    ReplayTeam4Strategy,
)


def _observation(
    *,
    own: tuple[ImitationBlob, ...] | None = None,
    enemies: tuple[ImitationBlob, ...] = (),
    viruses: tuple[ImitationPoint, ...] = (),
    round_number: int = 1000,
) -> ImitationObservation:
    return ImitationObservation(
        round_number=round_number,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=own or (ImitationBlob(20.0, 20.0, 3.0, player_id=2, blob_id=0),),
        visible_blobs=enemies,
        visible_food=(ImitationPoint(24.0, 20.0, entity_id=1),),
        visible_viruses=viruses,
    )


def test_team4_profile_covers_all_observed_matches() -> None:
    assert PROFILE.source_matches == (11673, 11698, 11752, 13937)


def test_team4_direction_is_unit_length() -> None:
    decision = ReplayTeam4Strategy().choose_observation(_observation())

    assert math.isclose(math.hypot(*decision.direction), 1.0)


def test_team4_predator_disables_farm_split() -> None:
    predator = ImitationBlob(22.0, 20.0, 4.0, player_id=3, blob_id=0)
    strategy = ReplayTeam4Strategy()

    split, roll = strategy._split_decision(_observation(enemies=(predator,)))

    assert not split
    assert roll is None


def test_team4_requires_one_merge_ready_high_mass_blob() -> None:
    strategy = ReplayTeam4Strategy()
    low_mass = (ImitationBlob(20.0, 20.0, 2.0, player_id=2, blob_id=0),)
    fragmented = (
        ImitationBlob(20.0, 20.0, 2.0, player_id=2, blob_id=0),
        ImitationBlob(22.0, 20.0, 2.0, player_id=2, blob_id=1),
    )
    cooling = (
        ImitationBlob(20.0, 20.0, 3.0, player_id=2, blob_id=0, merge_cooldown=1),
    )

    assert not strategy._farm_split_candidate(_observation(own=low_mass))
    assert not strategy._farm_split_candidate(_observation(own=fragmented))
    assert not strategy._farm_split_candidate(_observation(own=cooling))


def test_team4_sparse_split_roll_is_deterministic() -> None:
    first = ReplayTeam4Strategy._split_roll(
        round_number=1000,
        player_id=2,
        radius=3.0,
    )
    second = ReplayTeam4Strategy._split_roll(
        round_number=1000,
        player_id=2,
        radius=3.0,
    )

    assert first == second
    assert 0.0 <= first < 1.0
    assert 0.0 < FARM_SPLIT_RATE < 1.0
