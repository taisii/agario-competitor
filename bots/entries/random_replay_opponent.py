from __future__ import annotations

"""Run one randomly selected specialised official-replay clone."""

import importlib
import os
import random
import sys
import time
from pathlib import Path


BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from helper.game import Game  # noqa: E402
from lib.interface.queries.query_move import QueryMovePlayer  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402


def available_replay_team_ids() -> tuple[int, ...]:
    return tuple(
        sorted(
            int(path.stem.removeprefix("replay_team_"))
            for path in (BOTS_DIR / "strategies").glob("replay_team_*.py")
        )
    )


def selected_replay_team_id(*, player_id: int) -> int:
    team_ids = available_replay_team_ids()
    if not team_ids:
        raise RuntimeError("No replay_team_* strategies are available")
    base_seed = int(
        os.environ.get("BOT_RANDOM_SEED", os.getpid() ^ time.time_ns())
    )
    trial = int(os.environ.get("BOT_BENCHMARK_TRIAL", "0"))
    seed = (
        base_seed
        ^ ((trial + 1) * 0x9E3779B1)
        ^ ((player_id + 1) * 0x85EBCA77)
    )
    team_id = random.Random(seed).choice(team_ids)
    return team_id


def create_random_replay_strategy(*, player_id: int):
    team_id = selected_replay_team_id(player_id=player_id)
    module = importlib.import_module(f"strategies.replay_team_{team_id}")
    expected_name = f"replay_team_{team_id}"
    candidates = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and value.__module__ == module.__name__
        and getattr(value, "name", None) == expected_name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one {expected_name!r} strategy class, found {len(candidates)}"
        )
    return candidates[0]()


def main() -> None:
    game = Game()
    strategy = None
    while True:
        try:
            query = game.get_next_query()
        except EOFError:
            break
        match query:
            case QueryMovePlayer():
                if strategy is None:
                    strategy = create_random_replay_strategy(
                        player_id=game.state.me.player_id,
                    )
                    print(
                        f"random_replay_opponent={strategy.name}",
                        file=sys.stderr,
                        flush=True,
                    )
                decision = strategy.choose(StrategyContext(game=game, query=query))
                game.send_move(decision.to_move(game.state.me.player_id))
            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()
