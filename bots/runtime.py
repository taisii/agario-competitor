"""Shared process runtime for local bot strategies.

Entries are intentionally responsible only for selecting/configuring a strategy.
Protocol handling, timing, telemetry and resource cleanup live here so every bot
is exercised under the same runtime semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from helper.game import Game
from lib.interface.queries.query_move import QueryMovePlayer
from strategies.base import Strategy, StrategyContext
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
