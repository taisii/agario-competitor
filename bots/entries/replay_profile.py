from __future__ import annotations

"""Run any generated replay profile selected by ``BOT_REPLAY_TEAM_ID``."""

import os
import sys
from pathlib import Path
from time import perf_counter


BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from helper.game import Game  # noqa: E402
from lib.interface.queries.query_move import QueryMovePlayer  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_imitation import ReplayImitationStrategy  # noqa: E402
from strategies.replay_profiles import PROFILES  # noqa: E402
from telemetry import MetricsLogger  # noqa: E402


def main() -> None:
    try:
        team_id = int(os.environ["BOT_REPLAY_TEAM_ID"])
        profile = PROFILES[team_id]
    except KeyError as exc:
        available = ", ".join(str(value) for value in sorted(PROFILES))
        raise SystemExit(
            f"Set BOT_REPLAY_TEAM_ID to an available opponent team: {available}"
        ) from exc

    game = Game()
    strategy = ReplayImitationStrategy(profile)
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
