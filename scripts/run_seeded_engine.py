from __future__ import annotations

"""Start the installed engine with a reproducible arena random stream."""

import argparse
from functools import wraps
import json
import os
import random
from pathlib import Path

from engine.config.io_config import CORE_DIRECTORY
from engine.game_engine import GameEngine
from engine.interface.io import player_connection
from engine.interface.logging.event_inspector import EventInspector


OFFICIAL_CUMULATIVE_TIMEOUT_SECONDS = player_connection.CUMULATIVE_TIMEOUT_SECONDS
OFFICIAL_TURN_TIMEOUT_SECONDS = player_connection.TIMEOUT_SECONDS


class ResultsOnlyGameEngine(GameEngine):
    """Engine variant that preserves outcomes without serialising replays."""

    def finish(self) -> None:
        inspector = EventInspector(
            self.state.private_event_history,
            self.state.get_rankings(),
            self.state.get_final_masses(),
        )
        result = inspector.get_result()
        output = Path(CORE_DIRECTORY) / "output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(result.model_dump_json())
        # The official cumulative timeout measures the complete query/response
        # wall time, not only time spent inside Strategy.choose().  The engine
        # does not expose this counter publicly, so the local diagnostic runner
        # records the authoritative PlayerConnection value alongside results.
        response_seconds = {
            str(player_id): player.connection._cumulative_time
            for player_id, player in self.state.players.items()
        }
        (output / "response_timings.json").write_text(
            json.dumps(response_seconds, indent=2, sort_keys=True) + "\n"
        )
        print(f"[engine]: match complete, outcome was {{{result}}}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help="Write results.json but omit replay and visualiser recordings.",
    )
    return parser.parse_args()


def configure_local_cumulative_timeout() -> None:
    """Optionally raise the timeout for correctness-first local benchmarks.

    Official-format verification leaves this variable unset and therefore
    uses the kit's authoritative eight-second limit.  The override exists only
    so a deliberately slow reference planner can provide an optimisation
    oracle before its choices are reproduced by a submission-safe planner.
    """

    raw_timeout = os.environ.get("AGARIO_LOCAL_CUMULATIVE_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return
    timeout = float(raw_timeout)
    if timeout <= 0.0:
        raise ValueError("AGARIO_LOCAL_CUMULATIVE_TIMEOUT_SECONDS must be positive")
    player_connection.CUMULATIVE_TIMEOUT_SECONDS = timeout


def configure_local_turn_timeout() -> None:
    """Optionally raise the per-query timeout for behavior-only replays."""

    raw_timeout = os.environ.get("AGARIO_LOCAL_TURN_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return
    timeout = int(raw_timeout)
    if timeout <= 0:
        raise ValueError("AGARIO_LOCAL_TURN_TIMEOUT_SECONDS must be positive")
    player_connection.TIMEOUT_SECONDS = timeout


def configure_local_player_timeouts() -> bool:
    """Relax only designated local opponents while keeping candidates strict.

    The engine evaluates player queries sequentially, while its timeout
    decorator reads module-level limits. Wrapping the two decorated query
    methods lets each call select the appropriate limits without changing the
    installed engine or granting extra time to the candidate under test.
    """

    raw_player_ids = os.environ.get("AGARIO_LOCAL_RELAXED_PLAYER_IDS")
    if raw_player_ids is None:
        return False
    relaxed_player_ids = frozenset(
        int(value.strip())
        for value in raw_player_ids.split(",")
        if value.strip()
    )
    if not relaxed_player_ids:
        raise ValueError("AGARIO_LOCAL_RELAXED_PLAYER_IDS must not be empty")

    relaxed_cumulative = float(
        os.environ.get("AGARIO_LOCAL_CUMULATIVE_TIMEOUT_SECONDS", "60")
    )
    relaxed_turn = int(os.environ.get("AGARIO_LOCAL_TURN_TIMEOUT_SECONDS", "10"))
    strict_cumulative = float(
        os.environ.get(
            "AGARIO_STRICT_CUMULATIVE_TIMEOUT_SECONDS",
            str(OFFICIAL_CUMULATIVE_TIMEOUT_SECONDS),
        )
    )
    strict_turn = int(
        os.environ.get(
            "AGARIO_STRICT_TURN_TIMEOUT_SECONDS",
            str(OFFICIAL_TURN_TIMEOUT_SECONDS),
        )
    )
    if min(relaxed_cumulative, strict_cumulative) <= 0.0:
        raise ValueError("cumulative timeout values must be positive")
    if min(relaxed_turn, strict_turn) <= 0:
        raise ValueError("turn timeout values must be positive")

    def install(method_name: str) -> None:
        original = getattr(player_connection.PlayerConnection, method_name)

        @wraps(original)
        def with_player_timeout(self, *args, **kwargs):
            previous_turn = player_connection.TIMEOUT_SECONDS
            previous_cumulative = player_connection.CUMULATIVE_TIMEOUT_SECONDS
            is_relaxed = self.player_id in relaxed_player_ids
            player_connection.TIMEOUT_SECONDS = (
                relaxed_turn if is_relaxed else strict_turn
            )
            player_connection.CUMULATIVE_TIMEOUT_SECONDS = (
                relaxed_cumulative if is_relaxed else strict_cumulative
            )
            try:
                return original(self, *args, **kwargs)
            finally:
                player_connection.TIMEOUT_SECONDS = previous_turn
                player_connection.CUMULATIVE_TIMEOUT_SECONDS = previous_cumulative

        setattr(player_connection.PlayerConnection, method_name, with_player_timeout)

    install("_query_move")
    install("_query_move_union")
    return True


def main() -> None:
    args = parse_args()
    player_specific_timeouts = configure_local_player_timeouts()
    if not player_specific_timeouts:
        configure_local_cumulative_timeout()
        configure_local_turn_timeout()
    random.seed(int(os.environ.get("AGARIO_ENGINE_RANDOM_SEED", "0")))
    cumulative_timeout = os.environ.get(
        "AGARIO_ENGINE_CUMULATIVE_TIMEOUT_SECONDS"
    )
    if cumulative_timeout is not None and not player_specific_timeouts:
        player_connection.CUMULATIVE_TIMEOUT_SECONDS = max(
            0.001,
            float(cumulative_timeout),
        )
    query_timeout = os.environ.get("AGARIO_ENGINE_QUERY_TIMEOUT_SECONDS")
    if query_timeout is not None and not player_specific_timeouts:
        player_connection.TIMEOUT_SECONDS = max(1, int(query_timeout))
    engine_type = ResultsOnlyGameEngine if args.no_recording else GameEngine
    engine_type().start()


if __name__ == "__main__":
    main()
