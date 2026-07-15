"""Shared process runtime for local bot strategies.

Entries are intentionally responsible only for selecting/configuring a strategy.
Protocol handling, timing, telemetry and resource cleanup live here so every bot
is exercised under the same runtime semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from time import perf_counter

from helper.game import Game
from lib.interface.queries.query_move import QueryMovePlayer
from strategies.base import Strategy, StrategyContext
from strategies.registry import create_strategy
from telemetry import MetricsLogger


StrategyFactory = Callable[[], Strategy]


def run_bot(
    strategy_factory: StrategyFactory,
    *,
    game_factory: Callable[[], Game] = Game,
    metrics_factory: Callable[[], MetricsLogger] = MetricsLogger,
) -> None:
    """Run one strategy until the engine closes its query stream.

    Factories keep process-owned state (game, strategy and metrics handles) local
    to a single invocation.  The injectable factories also let contract tests
    exercise the real loop without launching the simulator.
    """

    game = game_factory()
    strategy = strategy_factory()
    metrics = metrics_factory()

    try:
        while True:
            try:
                query = game.get_next_query()
            except EOFError:
                break

            match query:
                case QueryMovePlayer():
                    started_at = perf_counter()
                    decision = strategy.choose(StrategyContext(game=game, query=query))
                    elapsed_ms = (perf_counter() - started_at) * 1000.0
                    metrics.record(
                        game=game,
                        strategy_name=strategy.name,
                        decision=decision,
                        elapsed_ms=elapsed_ms,
                    )
                    game.send_move(decision.to_move(game.state.me.player_id))
                case _:
                    raise RuntimeError(f"Unsupported query type: {type(query)}")
    finally:
        metrics.close()


def run_strategy(
    strategy_name: str,
    *,
    environment_defaults: Mapping[str, str] | None = None,
) -> None:
    """Run a catalogued strategy with optional, externally overridable defaults."""

    defaults = environment_defaults or {}
    added_keys = tuple(key for key in defaults if key not in os.environ)
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    try:
        run_bot(lambda: create_strategy(strategy_name))
    finally:
        # Entry defaults belong to this invocation, not to a later bot run in
        # the same interpreter (for example a contract test or tournament host).
        for key in added_keys:
            os.environ.pop(key, None)


def run_configured_strategy(
    default_strategy_name: str,
    *,
    environment_defaults: Mapping[str, str] | None = None,
) -> None:
    """Run ``BOT_STRATEGY`` when set, otherwise the entry's named default."""

    run_strategy(
        os.environ.get("BOT_STRATEGY", default_strategy_name),
        environment_defaults=environment_defaults,
    )
