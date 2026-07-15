"""Shared bootstrap for declarative simulator entry files."""

import sys
from pathlib import Path

BOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTS_DIR))

from runtime import run_configured_strategy, run_strategy  # noqa: E402


def run_entry(entry_file: str, *, configurable: bool = False) -> None:
    name = Path(entry_file).stem
    (run_configured_strategy if configurable else run_strategy)(name)
