from __future__ import annotations

import ast
import importlib.util
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_submission import build_submission  # noqa: E402


def _load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_submission_is_self_contained_and_importable(tmp_path: Path) -> None:
    output, digest = build_submission(tmp_path / "my_bot.py")
    source = output.read_text()
    tree = ast.parse(source)
    module = _load_submission(output)

    assert len(digest) == 64
    assert module.ReplayDistilledStrategy.name == "replay_distilled"
    assert "class AssetPreservationLayer" in source
    assert "strategy = ReplayDistilledStrategy()" in source
    assert source.count('if __name__ == "__main__":') == 1
    local_roots = {"simulation", "strategies", "telemetry"}
    assert not {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } & local_roots


class _ShortReader(StringIO):
    def read(self, size: int = -1) -> str:
        return super().read(min(size, 2) if size > 1 else size)


def test_submission_fast_reader_preserves_framing(tmp_path: Path) -> None:
    output, _ = build_submission(tmp_path / "my_bot.py")
    module = _load_submission(output)
    payload = '{"query_type":"move_player"}'
    connection = SimpleNamespace(
        _from_engine_pipe=_ShortReader(f"{len(payload)},{payload}")
    )

    class FakeQuery:
        @classmethod
        def model_validate_json(cls, raw: str) -> str:
            return raw

    module.QueryMovePlayer = FakeQuery
    module._install_fast_query_reader(SimpleNamespace(connection=connection))

    assert connection.get_next_query() == payload


def test_submission_build_is_deterministic(tmp_path: Path) -> None:
    first, first_digest = build_submission(tmp_path / "first.py")
    second, second_digest = build_submission(tmp_path / "second.py")

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
