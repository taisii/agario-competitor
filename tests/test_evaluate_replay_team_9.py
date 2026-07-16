from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_replay_team_9 import (
    parse_args,
    partition_samples,
    resolve_replay_paths,
)


def test_resolve_replay_paths_uses_only_sources_and_root_preference(
    tmp_path: Path,
) -> None:
    preferred = tmp_path / "preferred"
    fallback = tmp_path / "fallback"
    preferred.mkdir()
    fallback.mkdir()
    preferred_match = preferred / "match-100-replay.json"
    preferred_match.write_text("[]")
    (fallback / "match-100-replay.json").write_text("[]")
    fallback_match = fallback / "match-200-replay.json"
    fallback_match.write_text("[]")
    (fallback / "match-999-replay.json").write_text("[]")

    paths = resolve_replay_paths((preferred, fallback), (100, 200))

    assert paths == (preferred_match, fallback_match)


def test_resolve_replay_paths_reports_missing_sources(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"\[100, 200\].*searched"):
        resolve_replay_paths((tmp_path,), (100, 200))


def test_parse_args_accepts_repeated_roots_and_jobs() -> None:
    args = parse_args(
        [
            "--replay-dir",
            "/first",
            "--replay-dir",
            "/second",
            "--jobs",
            "2",
        ]
    )

    assert args.replay_dirs == [Path("/first"), Path("/second")]
    assert args.jobs == 2


def test_partition_samples_is_nonempty_and_disjoint() -> None:
    before = SimpleNamespace(match_id=28_999)
    after = SimpleNamespace(match_id=29_000)

    training, held_out = partition_samples(  # type: ignore[arg-type]
        [after, before],
        29_000,
    )

    assert training == [before]
    assert held_out == [after]
    assert {sample.match_id for sample in training}.isdisjoint(
        sample.match_id for sample in held_out
    )


def test_partition_samples_rejects_empty_cohort() -> None:
    with pytest.raises(ValueError, match="non-empty, disjoint"):
        partition_samples(  # type: ignore[arg-type]
            [SimpleNamespace(match_id=29_000)],
            29_000,
        )
