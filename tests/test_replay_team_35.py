from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.replay_imitation import (  # noqa: E402
    ImitationBlob,
    ImitationObservation,
)
from strategies.replay_team_35 import (  # noqa: E402
    MIN_SPLIT_RADIUS,
    ReplayTeam35Strategy,
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
        else (
            ImitationBlob(
                10.0,
                10.0,
                MIN_SPLIT_RADIUS,
                player_id=0,
                blob_id=0,
                merge_cooldown=0,
            ),
        ),
        visible_blobs=visible_blobs,
        visible_food=(),
        visible_viruses=(),
    )


def _prey() -> ImitationBlob:
    return ImitationBlob(13.0, 10.0, 1.0, player_id=1, blob_id=0)


def test_team35_profile_uses_all_observed_source_matches() -> None:
    strategy = ReplayTeam35Strategy()

    assert strategy.profile.source_matches == (11673, 11716, 11725, 11739, 11753)


def test_team35_splits_with_one_merge_ready_large_blob_and_visible_prey() -> None:
    split, reason = ReplayTeam35Strategy._split_decision(
        _observation(visible_blobs=(_prey(),))
    )

    assert split
    assert reason == "single_blob_prey_split"


def test_team35_does_not_split_below_observed_radius_floor() -> None:
    small = ImitationBlob(
        10.0,
        10.0,
        MIN_SPLIT_RADIUS - 0.01,
        player_id=0,
        blob_id=0,
    )

    split, reason = ReplayTeam35Strategy._split_decision(
        _observation(own=(small,), visible_blobs=(_prey(),))
    )

    assert not split
    assert reason == "below_observed_radius_floor"


def test_team35_does_not_split_while_fragments_are_unmerged() -> None:
    fragments = (
        ImitationBlob(10.0, 10.0, 2.2, player_id=0, blob_id=0),
        ImitationBlob(11.0, 10.0, 2.2, player_id=0, blob_id=1),
    )

    split, reason = ReplayTeam35Strategy._split_decision(
        _observation(own=fragments, visible_blobs=(_prey(),))
    )

    assert not split
    assert reason == "requires_single_blob"


def test_team35_does_not_split_during_merge_cooldown() -> None:
    cooling = ImitationBlob(
        10.0,
        10.0,
        2.2,
        player_id=0,
        blob_id=0,
        merge_cooldown=1,
    )

    split, reason = ReplayTeam35Strategy._split_decision(
        _observation(own=(cooling,), visible_blobs=(_prey(),))
    )

    assert not split
    assert reason == "merge_cooldown_active"


def test_team35_predator_suppresses_split_even_when_prey_is_visible() -> None:
    predator = ImitationBlob(14.0, 10.0, 3.0, player_id=2, blob_id=0)

    split, reason = ReplayTeam35Strategy._split_decision(
        _observation(visible_blobs=(_prey(), predator))
    )

    assert not split
    assert reason == "predator_visible"


def test_team35_requires_visible_prey_to_split() -> None:
    split, reason = ReplayTeam35Strategy._split_decision(_observation())

    assert not split
    assert reason == "no_visible_prey"
