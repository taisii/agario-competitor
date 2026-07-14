from __future__ import annotations

"""Start the installed engine with a reproducible arena random stream."""

import argparse
import json
import os
import random
from pathlib import Path

from engine.config.io_config import CORE_DIRECTORY
from engine.game_engine import GameEngine
from engine.interface.io import player_connection
from engine.interface.logging.event_inspector import EventInspector


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


def main() -> None:
    args = parse_args()
    configure_local_cumulative_timeout()
    configure_local_turn_timeout()
    random.seed(int(os.environ.get("AGARIO_ENGINE_RANDOM_SEED", "0")))
    cumulative_timeout = os.environ.get(
        "AGARIO_ENGINE_CUMULATIVE_TIMEOUT_SECONDS"
    )
    if cumulative_timeout is not None:
        player_connection.CUMULATIVE_TIMEOUT_SECONDS = max(
            0.001,
            float(cumulative_timeout),
        )
    query_timeout = os.environ.get("AGARIO_ENGINE_QUERY_TIMEOUT_SECONDS")
    if query_timeout is not None:
        player_connection.TIMEOUT_SECONDS = max(1, int(query_timeout))
    engine_type = ResultsOnlyGameEngine if args.no_recording else GameEngine
    engine_type().start()


if __name__ == "__main__":
    main()
