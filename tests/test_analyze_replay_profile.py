from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze_replay_profile  # noqa: E402


def _profile(
    *,
    utility_calls: int = 1,
    utility_hits: int = 0,
    utility_misses: int = 1,
) -> dict[str, object]:
    phase_ms = {
        name: 0.0 for name in analyze_replay_profile.REQUIRED_PHASE_KEYS
    }
    phase_ms["setup_and_search_control"] = 1.0
    return {
        "schema_version": 1,
        "round": 0,
        "sample_every_n": 100,
        "phase_ms": phase_ms,
        "operation_inclusive_ms": {"choose": 1.0},
        "calls": {"choose": 1, "utility": utility_calls},
        "counts": {
            "utility_hit": utility_hits,
            "utility_miss": utility_misses,
            "replay_candidate_raw": 2,
            "replay_candidate_unique": 1,
            "replay_candidate_zero_drops": 0,
            "replay_candidate_duplicate_drops": 1,
        },
        "value_sums": {},
    }


def test_validate_profile_accepts_well_formed_sample(tmp_path) -> None:
    sample = analyze_replay_profile.ProfileSample(
        tmp_path / "metrics.jsonl",
        1,
        0,
        _profile(),
    )

    assert analyze_replay_profile.validate_profile(sample) == []


def test_validate_profile_checks_conservation_per_sample(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    too_many_calls = analyze_replay_profile.ProfileSample(
        path,
        1,
        0,
        _profile(utility_calls=1, utility_hits=0, utility_misses=0),
    )
    too_many_lookups = analyze_replay_profile.ProfileSample(
        path,
        2,
        0,
        _profile(utility_calls=0, utility_hits=0, utility_misses=1),
    )

    violations = [
        *analyze_replay_profile.validate_profile(too_many_calls),
        *analyze_replay_profile.validate_profile(too_many_lookups),
    ]

    assert len([item for item in violations if "utility calls=" in item]) == 2


def test_validate_profile_checks_shared_cache_conservation(tmp_path) -> None:
    profile = _profile()
    calls = profile["calls"]
    counts = profile["counts"]
    assert isinstance(calls, dict)
    assert isinstance(counts, dict)
    calls["cache_hazard"] = 2
    counts["hazard_hit"] = 1
    sample = analyze_replay_profile.ProfileSample(
        tmp_path / "metrics.jsonl",
        1,
        0,
        profile,
    )

    assert any(
        "hazard calls=2 != hits+misses=1" in item
        for item in analyze_replay_profile.validate_profile(sample)
    )


def test_main_rejects_metrics_without_profile(tmp_path, monkeypatch) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"round": 0, "decision_diagnostics": {}}) + "\n")
    monkeypatch.setattr(sys, "argv", ["analyze_replay_profile.py", str(path)])

    with pytest.raises(SystemExit, match="no replay_profile samples"):
        analyze_replay_profile.main()


def test_validate_profile_rejects_unknown_schema(tmp_path) -> None:
    profile = _profile()
    profile["schema_version"] = 2
    sample = analyze_replay_profile.ProfileSample(
        tmp_path / "metrics.jsonl",
        1,
        0,
        profile,
    )

    assert any(
        "unsupported schema_version=2" in item
        for item in analyze_replay_profile.validate_profile(sample)
    )


def test_validate_profile_rejects_empty_measurement_sections(tmp_path) -> None:
    sample = analyze_replay_profile.ProfileSample(
        tmp_path / "metrics.jsonl",
        1,
        0,
        {
            "schema_version": 1,
            "round": 0,
            "sample_every_n": 100,
            "phase_ms": {},
            "operation_inclusive_ms": {},
            "calls": {},
            "counts": {},
            "value_sums": {},
        },
    )

    violations = analyze_replay_profile.validate_profile(sample)

    assert any("phase_ms missing required keys=" in item for item in violations)
    assert any("exactly one choose/fallback call" in item for item in violations)


def test_validate_profile_rejects_unscheduled_round(tmp_path) -> None:
    profile = _profile()
    profile["round"] = 99
    sample = analyze_replay_profile.ProfileSample(
        tmp_path / "metrics.jsonl",
        1,
        99,
        profile,
    )

    assert any(
        "round=99 is not sampled by sample_every_n=100" in item
        for item in analyze_replay_profile.validate_profile(sample)
    )
