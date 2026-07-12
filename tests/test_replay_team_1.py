from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.replay_imitation import (  # noqa: E402
    ImitationBlob,
    ImitationObservation,
)
from strategies.replay_team_1 import (  # noqa: E402
    MAX_BLOB_COUNT,
    ReplayTeam1Strategy,
)


def _observation(
    *,
    own: tuple[ImitationBlob, ...] | None = None,
    visible_blobs: tuple[ImitationBlob, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=500,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=own
        if own is not None
        else (ImitationBlob(10.0, 10.0, 2.2, player_id=0, blob_id=0),),
        visible_blobs=visible_blobs,
        visible_food=(),
        visible_viruses=(),
    )


def test_team1_uses_both_observed_matches_for_direction_profile() -> None:
    strategy = ReplayTeam1Strategy()

    assert strategy.profile.source_matches == (11710, 11753, 13932, 13940)


def test_team1_splits_for_reachable_prey_inside_heading_corridor() -> None:
    prey = ImitationBlob(15.0, 10.0, 1.0, player_id=1, blob_id=0)

    split, reason = ReplayTeam1Strategy._split_decision(
        _observation(visible_blobs=(prey,)),
        (1.0, 0.0),
    )

    assert split
    assert reason == "reachable_prey_split"


def test_team1_does_not_split_for_prey_outside_heading_corridor() -> None:
    prey = ImitationBlob(10.0, 15.0, 1.0, player_id=1, blob_id=0)

    split, reason = ReplayTeam1Strategy._split_decision(
        _observation(visible_blobs=(prey,)),
        (1.0, 0.0),
    )

    assert not split
    assert reason == "no_reachable_split_prey"


def test_team1_does_not_split_when_child_cannot_eat_target() -> None:
    near_equal = ImitationBlob(13.0, 10.0, 2.0, player_id=1, blob_id=0)

    split, reason = ReplayTeam1Strategy._split_decision(
        _observation(visible_blobs=(near_equal,)),
        (1.0, 0.0),
    )

    assert not split
    assert reason == "no_reachable_split_prey"


def test_team1_does_not_split_below_engine_minimum_mass() -> None:
    own = (ImitationBlob(10.0, 10.0, 1.4, player_id=0, blob_id=0),)
    prey = ImitationBlob(12.0, 10.0, 0.5, player_id=1, blob_id=0)

    split, reason = ReplayTeam1Strategy._split_decision(
        _observation(own=own, visible_blobs=(prey,)),
        (1.0, 0.0),
    )

    assert not split
    assert reason == "no_reachable_split_prey"


def test_team1_suppresses_split_at_blob_cap() -> None:
    own = tuple(
        ImitationBlob(10.0 + index * 0.1, 10.0, 2.0, player_id=0, blob_id=index)
        for index in range(MAX_BLOB_COUNT)
    )
    prey = ImitationBlob(12.0, 10.0, 0.5, player_id=1, blob_id=0)

    split, reason = ReplayTeam1Strategy._split_decision(
        _observation(own=own, visible_blobs=(prey,)),
        (1.0, 0.0),
    )

    assert not split
    assert reason == "blob_cap"
