from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from helper.game import Game  # noqa: E402
from lib.interface.queries.query_move import QueryMovePlayer  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_2 import ReplayTeam2Strategy  # noqa: E402
from telemetry import MetricsLogger  # noqa: E402


def main() -> None:
    game = Game()
    strategy = ReplayTeam2Strategy()
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
                    metrics.record(
                        game=game,
                        strategy_name=strategy.name,
                        decision=decision,
                        elapsed_ms=(perf_counter() - started_at) * 1000.0,
                    )
                    game.send_move(decision.to_move(game.state.me.player_id))
                case _:
                    raise RuntimeError(f"Unsupported query type: {type(query)}")
    finally:
        metrics.close()


if __name__ == "__main__":
    main()
