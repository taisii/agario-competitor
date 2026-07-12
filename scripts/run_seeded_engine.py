from __future__ import annotations

"""Start the installed engine with a reproducible arena random stream."""

import os
import random
import sys

from engine.__main__ import main as engine_main


def main() -> None:
    random.seed(int(os.environ["AGARIO_ENGINE_RANDOM_SEED"]))
    engine_main()


if __name__ == "__main__":
    main()
