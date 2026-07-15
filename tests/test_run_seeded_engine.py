from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_fast_simulation, run_seeded_engine


class _Result:
    def model_dump_json(self) -> str:
        return '{"result_type":"SUCCESS"}'

    def __str__(self) -> str:
        return "result_type='SUCCESS' ranking=[0] final_masses={0: 1.0}"


class _Inspector:
    def __init__(self, event_history, rankings, final_masses) -> None:
        assert event_history == ["event"]
        assert rankings == [0]
        assert final_masses == {0: 1.0}

    def get_result(self) -> _Result:
        return _Result()


def test_results_only_engine_omits_large_recording_files(
    tmp_path,
    monkeypatch,
) -> None:
    state = SimpleNamespace(
        private_event_history=["event"],
        get_rankings=lambda: [0],
        get_final_masses=lambda: {0: 1.0},
        players={
            0: SimpleNamespace(
                connection=SimpleNamespace(_cumulative_time=1.25),
            )
        },
    )
    engine = run_seeded_engine.ResultsOnlyGameEngine.__new__(
        run_seeded_engine.ResultsOnlyGameEngine
    )
    engine.state = state
    monkeypatch.setattr(run_seeded_engine, "EventInspector", _Inspector)
    monkeypatch.setattr(run_seeded_engine, "CORE_DIRECTORY", str(tmp_path))

    engine.finish()

    output = tmp_path / "output"
    assert (output / "results.json").read_text() == '{"result_type":"SUCCESS"}'
    assert (output / "response_timings.json").read_text() == '{\n  "0": 1.25\n}\n'
    assert not (output / "game.json").exists()
    assert not (output / "visualiser_forwards_differential.json").exists()


def test_local_reference_timeout_override_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("AGARIO_LOCAL_CUMULATIVE_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr(
        run_seeded_engine.player_connection,
        "CUMULATIVE_TIMEOUT_SECONDS",
        8,
    )

    run_seeded_engine.configure_local_cumulative_timeout()

    assert run_seeded_engine.player_connection.CUMULATIVE_TIMEOUT_SECONDS == 60.0


def test_fast_runner_forwards_local_timeout_override() -> None:
    env = {"PATH": "/bin"}

    run_fast_simulation.forward_engine_overrides(
        env,
        {
            "AGARIO_LOCAL_CUMULATIVE_TIMEOUT_SECONDS": "60",
            "AGARIO_LOCAL_TURN_TIMEOUT_SECONDS": "10",
            "AGARIO_LOCAL_RELAXED_PLAYER_IDS": "1,2,3,4,5,6,7",
            "AGARIO_STRICT_CUMULATIVE_TIMEOUT_SECONDS": "8",
            "AGARIO_STRICT_TURN_TIMEOUT_SECONDS": "1",
        },
    )

    assert env["AGARIO_LOCAL_CUMULATIVE_TIMEOUT_SECONDS"] == "60"
    assert env["AGARIO_LOCAL_TURN_TIMEOUT_SECONDS"] == "10"
    assert env["AGARIO_LOCAL_RELAXED_PLAYER_IDS"] == "1,2,3,4,5,6,7"
    assert env["AGARIO_STRICT_CUMULATIVE_TIMEOUT_SECONDS"] == "8"
    assert env["AGARIO_STRICT_TURN_TIMEOUT_SECONDS"] == "1"


def test_local_turn_timeout_override_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("AGARIO_LOCAL_TURN_TIMEOUT_SECONDS", "10")
    monkeypatch.setattr(run_seeded_engine.player_connection, "TIMEOUT_SECONDS", 1)

    run_seeded_engine.configure_local_turn_timeout()

    assert run_seeded_engine.player_connection.TIMEOUT_SECONDS == 10


def test_player_specific_timeout_keeps_candidate_strict(monkeypatch) -> None:
    observed = []

    def query(connection, *_args, **_kwargs):
        observed.append(
            (
                connection.player_id,
                run_seeded_engine.player_connection.TIMEOUT_SECONDS,
                run_seeded_engine.player_connection.CUMULATIVE_TIMEOUT_SECONDS,
            )
        )

    monkeypatch.setattr(
        run_seeded_engine.player_connection.PlayerConnection,
        "_query_move",
        query,
    )
    monkeypatch.setattr(
        run_seeded_engine.player_connection.PlayerConnection,
        "_query_move_union",
        query,
    )
    monkeypatch.setenv("AGARIO_LOCAL_RELAXED_PLAYER_IDS", "1,2,3,4,5,6,7")
    monkeypatch.setenv("AGARIO_LOCAL_TURN_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("AGARIO_LOCAL_CUMULATIVE_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("AGARIO_STRICT_TURN_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("AGARIO_STRICT_CUMULATIVE_TIMEOUT_SECONDS", "8")

    assert run_seeded_engine.configure_local_player_timeouts()
    candidate = run_seeded_engine.player_connection.PlayerConnection.__new__(
        run_seeded_engine.player_connection.PlayerConnection
    )
    candidate.player_id = 0
    opponent = run_seeded_engine.player_connection.PlayerConnection.__new__(
        run_seeded_engine.player_connection.PlayerConnection
    )
    opponent.player_id = 5
    candidate._query_move(None, None, None)
    opponent._query_move(None, None, None)

    assert observed == [(0, 1, 8.0), (5, 10, 60.0)]
