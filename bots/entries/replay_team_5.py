from __future__ import annotations

import sys
from pathlib import Path

BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from helper.game import Game
from lib.interface.queries.query_move import QueryMovePlayer
from strategies.base import StrategyContext
from strategies.replay_team_5 import ReplayTeam5Strategy


def main() -> None:
    game = Game()
    strategy = ReplayTeam5Strategy()
    while True:
        try:
            query = game.get_next_query()
        except EOFError:
            break
        match query:
            case QueryMovePlayer():
                decision = strategy.choose(StrategyContext(game=game, query=query))
                game.send_move(decision.to_move(game.state.me.player_id))
            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()
