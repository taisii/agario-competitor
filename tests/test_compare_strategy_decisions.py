from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_strategy_decisions import (  # noqa: E402
    ComparisonSample,
    _angle_degrees,
    summarise,
)


def _sample(**overrides) -> ComparisonSample:
    values = {
        "round_number": 10,
        "player_mass": 4.0,
        "blob_count": 1,
        "angle_degrees": 10.0,
        "split_agreement": True,
        "semantic_direction": (1.0, 0.0),
        "replay_direction": (1.0, 0.0),
        "semantic_split": False,
        "replay_split": False,
        "semantic_reason": "nearest_food",
        "replay_reason": "food_cluster",
        "semantic_elapsed_ms": 0.2,
        "replay_elapsed_ms": 0.8,
        "semantic_proxy_value": 1.0,
        "replay_proxy_value": 1.02,
        "proxy_regret": 0.02,
        "strategic_focus": "background",
        "predator_visible": False,
        "wall_clearance": 4.0,
        "current_safety_margin": None,
        "selected_safety_margin": None,
    }
    values.update(overrides)
    return ComparisonSample(**values)


def test_angle_difference_is_bounded_and_directional() -> None:
    assert _angle_degrees((1.0, 0.0), (1.0, 0.0)) == 0.0
    assert _angle_degrees((1.0, 0.0), (0.0, 1.0)) == 90.0
    assert _angle_degrees((1.0, 0.0), (-1.0, 0.0)) == 180.0


def test_recommendation_requires_difference_and_proxy_evidence() -> None:
    ordinary = _sample()
    different_but_worse = _sample(angle_degrees=100.0, proxy_regret=-0.4)
    supported = _sample(
        angle_degrees=100.0,
        proxy_regret=0.3,
        replay_proxy_value=1.3,
        strategic_focus="prey",
    )

    summary = summarise([ordinary, different_but_worse, supported])

    assert not ordinary.adoption_candidate
    assert not different_but_worse.adoption_candidate
    assert supported.adoption_candidate
    assert summary["adoption_candidate_rate"] == 1 / 3
    assert summary["recommended_differences"][0]["adoption_candidates"] == 1


def test_split_disagreement_can_be_recommended_without_large_angle() -> None:
    sample = _sample(
        split_agreement=False,
        proxy_regret=0.2,
        replay_proxy_value=1.2,
        strategic_focus="virus",
    )

    assert sample.adoption_candidate


def test_background_food_difference_is_never_an_adoption_candidate() -> None:
    sample = _sample(angle_degrees=170.0, proxy_regret=10.0)

    assert not sample.adoption_candidate
