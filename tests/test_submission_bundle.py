from __future__ import annotations

import ast
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bots"))

from scripts.build_submission import build_submission  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.receding_horizon import ThreatAwareRecedingHorizonStrategy, EnemyTrack, OwnBlob, _speed  # noqa: E402


def test_submission_bundle_is_single_file_without_local_imports() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output, digest = build_submission(Path(directory) / "my_bot.py")
        source = output.read_text(encoding="utf-8")
        tree = ast.parse(source)

    assert len(digest) == 64
    assert "class ReplayDominanceStrategy(ThreatAwareRecedingHorizonStrategy)" in source
    assert "strategy = ReplayDominanceStrategy()" in source
    assert 'if __name__ == "__main__":' in source
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] not in {
                "strategies",
                "telemetry",
            }


def test_virus_hunter_submission_contains_only_required_strategy_classes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output, _ = build_submission(
            Path(directory) / "virus_hunter.py",
            strategy_name="virus_hunter",
        )
        source = output.read_text(encoding="utf-8")
        tree = ast.parse(source)

    assert "class VirusHunterStrategy" in source
    assert "strategy = VirusHunterStrategy()" in source
    assert "class PotentialFieldVirusFarmerStrategy" not in source
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] not in {
                "strategies",
                "telemetry",
            }


def test_replay_dominance_submission_is_self_contained_and_uses_new_strategy() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output, _ = build_submission(
            Path(directory) / "replay_dominance.py",
            strategy_name="replay_dominance",
        )
        source = output.read_text(encoding="utf-8")
        tree = ast.parse(source)

    assert "class ThreatAwareRecedingHorizonStrategy" in source
    assert "class ReplayDominanceStrategy(ThreatAwareRecedingHorizonStrategy)" in source
    assert "strategy = ReplayDominanceStrategy()" in source
    assert source.count('if __name__ == "__main__":') == 1
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] not in {
                "strategies",
                "telemetry",
            }


def test_censored_predator_track_advances_toward_vulnerable_blob() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks[(1, 0)] = EnemyTrack(
        player_id=1,
        blob_id=0,
        x=10.0,
        y=10.0,
        radius=2.0,
        direction=(0.0, 0.0),
        last_seen_round=1,
    )
    own = OwnBlob(blob_id=0, x=20.0, y=10.0, radius=1.0)
    state = SimpleNamespace(
        round=2,
        visible_blobs=(),
        view_center=(50.0, 50.0),
        vision_size=8.0,
    )
    context = StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(update={}),
    )

    enemies = strategy._update_enemy_memory(context, (own,), arena_size=60.0)

    assert len(enemies) == 1
    assert math.isclose(enemies[0].x, 10.0 + _speed(2.0))
    assert enemies[0].direction == (1.0, 0.0)
    assert enemies[0].stale_rounds == 1
