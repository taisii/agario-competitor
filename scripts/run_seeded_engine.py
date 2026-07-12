from __future__ import annotations

"""Start the installed engine with a reproducible arena random stream."""

import argparse
import os
import random
from pathlib import Path

from engine.config.io_config import CORE_DIRECTORY
from engine.game_engine import GameEngine
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
        print(f"[engine]: match complete, outcome was {{{result}}}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help="Write results.json but omit replay and visualiser recordings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(os.environ.get("AGARIO_ENGINE_RANDOM_SEED", "0")))
    engine_type = ResultsOnlyGameEngine if args.no_recording else GameEngine
    engine_type().start()


if __name__ == "__main__":
    main()
