from __future__ import annotations

from types import SimpleNamespace

from scripts import run_seeded_engine


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
