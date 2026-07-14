from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.interface.queries.query_move import QueryMovePlayer  # noqa: E402
import runtime  # noqa: E402
from runtime import run_bot  # noqa: E402
from telemetry import _json_safe  # noqa: E402


def test_metrics_json_replaces_non_finite_diagnostics() -> None:
    assert _json_safe(
        {"margin": float("inf"), "nested": [float("-inf"), float("nan"), 1.0]}
    ) == {"margin": None, "nested": [None, None, 1.0]}


class _Decision:
    def __init__(self) -> None:
        self.player_ids: list[int] = []

    def to_move(self, player_id: int):
        self.player_ids.append(player_id)
        return ("move", player_id)


class _Strategy:
    name = "test_strategy"

    def __init__(self, decision: _Decision) -> None:
        self.decision = decision
        self.contexts = []

    def choose(self, context):
        self.contexts.append(context)
        return self.decision


class _Game:
    def __init__(self, queries) -> None:
        self._queries = iter(queries)
        self.state = SimpleNamespace(me=SimpleNamespace(player_id=7))
        self.moves = []

    def get_next_query(self):
        try:
            return next(self._queries)
        except StopIteration as exc:
            raise EOFError from exc

    def send_move(self, move) -> None:
        self.moves.append(move)


class _Metrics:
    def __init__(self) -> None:
        self.records = []
        self.closed = False

    def record(self, **record) -> None:
        self.records.append(record)

    def close(self) -> None:
        self.closed = True


def test_runtime_drives_strategy_records_metrics_and_closes_resources() -> None:
    query = QueryMovePlayer.model_construct()
    game = _Game([query])
    decision = _Decision()
    strategy = _Strategy(decision)
    metrics = _Metrics()

    run_bot(
        lambda: strategy,
        game_factory=lambda: game,
        metrics_factory=lambda: metrics,
    )

    assert len(strategy.contexts) == 1
    assert strategy.contexts[0].game is game
    assert strategy.contexts[0].query is query
    assert decision.player_ids == [7]
    assert game.moves == [("move", 7)]
    assert len(metrics.records) == 1
    assert metrics.records[0]["strategy_name"] == "test_strategy"
    assert metrics.records[0]["decision"] is decision
    assert metrics.records[0]["elapsed_ms"] >= 0.0
    assert metrics.closed


def test_runtime_rejects_unsupported_queries_and_still_closes_metrics() -> None:
    game = _Game([object()])
    metrics = _Metrics()

    try:
        run_bot(
            lambda: _Strategy(_Decision()),
            game_factory=lambda: game,
            metrics_factory=lambda: metrics,
        )
    except RuntimeError as exc:
        assert "Unsupported query type" in str(exc)
    else:
        raise AssertionError("unsupported query must fail")

    assert metrics.closed


def test_configured_strategy_preserves_environment_override(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("BOT_STRATEGY", "environment_strategy")
    monkeypatch.setattr(
        runtime,
        "run_strategy",
        lambda name, **kwargs: calls.append((name, kwargs)),
    )

    runtime.run_configured_strategy(
        "entry_default",
        environment_defaults={"BOT_OPTION": "default"},
    )

    assert calls == [
        (
            "environment_strategy",
            {"environment_defaults": {"BOT_OPTION": "default"}},
        )
    ]


def test_strategy_environment_defaults_are_scoped_to_one_run(monkeypatch) -> None:
    seen = []
    monkeypatch.delenv("BOT_SCOPED_OPTION", raising=False)
    monkeypatch.setattr(
        runtime,
        "run_bot",
        lambda strategy_factory: seen.append(
            (strategy_factory, runtime.os.environ["BOT_SCOPED_OPTION"])
        ),
    )

    runtime.run_strategy(
        "food_greedy",
        environment_defaults={"BOT_SCOPED_OPTION": "temporary"},
    )

    assert len(seen) == 1
    assert seen[0][1] == "temporary"
    assert "BOT_SCOPED_OPTION" not in runtime.os.environ


def test_replay_opponent_runtime_uses_the_opponent_catalog(monkeypatch) -> None:
    factories = []
    monkeypatch.setattr(runtime, "run_bot", lambda factory: factories.append(factory))

    runtime.run_replay_opponent(25)

    assert len(factories) == 1
    monkeypatch.setattr(runtime, "create_replay_opponent", lambda team_id: team_id)
    assert factories[0]() == 25
