import os
from time import perf_counter

from helper.game import Game
from lib.interface.queries.query_move import QueryMovePlayer
from strategies.base import StrategyContext
from strategies.registry import create_strategy
from telemetry import MetricsLogger


def main() -> None:
    game = Game()
    strategy = create_strategy(os.environ.get("BOT_STRATEGY", "champion"))
    metrics = MetricsLogger()

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


if __name__ == "__main__":
    main()
