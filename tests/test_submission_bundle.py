from __future__ import annotations

import ast
import importlib.util
from io import StringIO
import math
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bots"))

from scripts.build_submission import build_submission  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.features import player_speed  # noqa: E402
from strategies.receding_horizon import (  # noqa: E402
    EnemyTrack,
    OwnBlob,
    ThreatAwareRecedingHorizonStrategy,
)


class _ShortReader(StringIO):
    """Exercise framed reads that return less than the requested body."""

    def read(self, size: int = -1) -> str:
        return super().read(min(size, 2) if size > 1 else size)


def test_submission_bundle_is_single_file_without_local_imports() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output, digest = build_submission(Path(directory) / "my_bot.py")
        source = output.read_text(encoding="utf-8")
        tree = ast.parse(source)

    assert len(digest) == 64
    assert "class SemanticLookaheadStrategy" in source
    assert "strategy = SemanticLookaheadStrategy()" in source
    assert 'if __name__ == "__main__":' in source
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] not in {
                "strategies",
                "telemetry",
            }


def test_submission_fast_query_reader_preserves_framing_and_direct_validation() -> (
    None
):
    with tempfile.TemporaryDirectory() as directory:
        output, _ = build_submission(
            Path(directory) / "semantic_lookahead.py",
            strategy_name="semantic_lookahead",
        )
        module_name = "semantic_fast_input_test"
        spec = importlib.util.spec_from_file_location(module_name, output)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)

    payload = '{"query_type":"move_player"}'
    connection = SimpleNamespace(
        _from_engine_pipe=_ShortReader(f"{len(payload)},{payload}")
    )

    class FakeQuery:
        @classmethod
        def model_validate_json(cls, raw: str) -> str:
            return raw

    module.QueryMovePlayer = FakeQuery
    game = SimpleNamespace(connection=connection)
    module._install_fast_query_reader(game)

    assert connection.get_next_query() == payload


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

    module = ModuleType("submission_test")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, output, "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    strategy = module.ReplayDominanceStrategy()
    assert strategy._can_still_consume_virus_at_contact(
        SimpleNamespace(radius=2.0, mass=4.0, pos=(10.0, 10.0)),
        SimpleNamespace(radius=1.5, pos=(10.0, 10.0)),
    )


def test_semantic_lookahead_submission_is_self_contained() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output, _ = build_submission(
            Path(directory) / "semantic_lookahead.py",
            strategy_name="semantic_lookahead",
        )
        source = output.read_text(encoding="utf-8")
        tree = ast.parse(source)

    assert "class SemanticLookaheadStrategy" in source
    assert "strategy = SemanticLookaheadStrategy()" in source
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] not in {
                "strategies",
                "telemetry",
            }


def test_replay_dominance_submission_preserves_local_import_aliases() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output, _ = build_submission(
            Path(directory) / "replay_dominance.py",
            strategy_name="replay_dominance",
        )
        module_name = "submission_runtime_test"
        spec = importlib.util.spec_from_file_location(module_name, output)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
        namespace = vars(module)

    assert (
        namespace["_feature_can_consume_virus"]
        is namespace["_replay_can_consume_virus"]
    )
    assert namespace["_replay_can_consume_virus"] is not namespace["can_consume_virus"]
    assert namespace["_replay_decayed_radius"] is namespace["decayed_radius"]
    assert namespace["_feature_movement_speed"] is namespace["movement_speed"]
    assert namespace["_replay_can_consume_virus"](
        2.0,
        1.5,
        eat_size_ratio=1.1,
    )


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
    assert math.isclose(enemies[0].x, 10.0 + player_speed(2.0))
    assert enemies[0].direction == (1.0, 0.0)
    assert enemies[0].stale_rounds == 1
