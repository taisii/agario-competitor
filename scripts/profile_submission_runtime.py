from __future__ import annotations

"""Run the built submission under cProfile inside a real simulator process.

This wrapper is for local diagnosis only. It preserves the submission's pipe
working directory, so blocking input, Pydantic decoding, state mutation,
strategy work, response encoding, and pipe writes all appear in one profile.
"""

import cProfile
import os
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".agario" / "profiles" / "submission-runtime.prof"


def main() -> None:
    output = Path(os.environ.get("BOT_PROFILE_OUTPUT", DEFAULT_OUTPUT)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        runpy.run_path(str(ROOT / "dist" / "my_bot.py"), run_name="__main__")
    finally:
        profiler.disable()
        profiler.dump_stats(output)


if __name__ == "__main__":
    main()
