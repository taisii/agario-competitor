from __future__ import annotations

import math
from pathlib import Path
import sys


BOTS_DIR = Path(__file__).resolve().parents[1] / "bots"
sys.path.insert(0, str(BOTS_DIR))

from strategies.replay_imitation import (
    FEATURE_NAMES,
    SPLIT_FEATURE_NAMES,
    ImitationBlob,
    ImitationObservation,
    ImitationPoint,
    ReplayProfile,
    direction_feature_vectors,
    predict_direction,
    predict_split,
)
from strategies.replay_profiles import PROFILES


def _observation() -> ImitationObservation:
    return ImitationObservation(
        round_number=100,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=(ImitationBlob(10.0, 10.0, 1.5, player_id=0),),
        visible_blobs=(
            ImitationBlob(8.0, 10.0, 2.0, player_id=1),
            ImitationBlob(13.0, 10.0, 0.5, player_id=2),
        ),
        visible_food=(ImitationPoint(10.0, 12.0, entity_id=1),),
        visible_viruses=(ImitationPoint(15.0, 10.0, 1.5, 1),),
    )


def _nearest_food_profile() -> ReplayProfile:
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("nearest_food")] = 1.0
    return ReplayProfile(
        team_id=999,
        direction_weights=tuple(weights),
        split_weights=tuple(0.0 for _ in SPLIT_FEATURE_NAMES),
        split_threshold=math.inf,
    )


def test_direction_features_reconstruct_relations() -> None:
    vectors = dict(zip(FEATURE_NAMES, direction_feature_vectors(_observation())))
    assert vectors["nearest_food"] == (0.0, 1.0)
    assert vectors["nearest_predator_escape"] == (1.0, 0.0)
    assert vectors["nearest_prey"] == (1.0, 0.0)
    assert vectors["virus_escape"] == (-1.0, 0.0)


def test_profile_predicts_direction_and_respects_no_split() -> None:
    profile = _nearest_food_profile()
    direction = predict_direction(profile, _observation())
    split, score = predict_split(profile, _observation(), (1.0, 0.0), direction)
    assert abs(direction[0]) < 1e-12
    assert abs(direction[1] - 1.0) < 1e-12
    assert split is False
    assert score == 0.0


def test_generated_profiles_cover_all_replay_opponents() -> None:
    expected = {
        1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 15, 16, 17, 21, 22, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 38, 39, 44, 48, 49, 51,
        53, 55, 56, 58, 59, 63, 68, 75, 77,
    }
    assert set(PROFILES) == expected
    for profile in PROFILES.values():
        assert len(profile.direction_weights) == len(FEATURE_NAMES)
        assert len(profile.split_weights) == len(SPLIT_FEATURE_NAMES)
        assert profile.source_matches
