from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from strategies.base import StrategyContext, StrategyDecision  # noqa: E402
from strategies.outcome_teacher_hybrid import (  # noqa: E402
    _semantic_capture_proposal,
)


def _context(
    *,
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
) -> StrategyContext:
    return StrategyContext(
        game=SimpleNamespace(
            state=SimpleNamespace(
                me=SimpleNamespace(
                    blobs={blob.blob_id: blob for blob in own},
                ),
                visible_blobs=list(enemies),
                round=300,
                max_rounds=1400,
            )
        ),
        query=SimpleNamespace(),
    )


def _enemy(*, player_id: int, radius: float) -> VisibleBlobModel:
    return VisibleBlobModel(
        player_id=player_id,
        team_id=player_id,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=radius,
    )


def _semantic(target_id: str) -> StrategyDecision:
    return StrategyDecision(
        direction=(1.0, 0.0),
        target_kind="prey",
        target_id=target_id,
        reason="capture_enemy",
        diagnostics={"selected_contact_turns": 3.0},
    )


def test_proposes_large_nearby_semantic_prey_to_replay_evaluator() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),)
    prey = _enemy(player_id=1, radius=1.0)

    proposal, diagnostics = _semantic_capture_proposal(
        _context(own=own, enemies=(prey,)),
        semantic=_semantic("1:0"),
    )

    assert proposal is not None
    assert proposal.reason == "semantic_prey"
    assert all(diagnostics["checks"].values())


def test_rejects_small_low_value_semantic_prey() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    prey = _enemy(player_id=1, radius=0.4)

    proposal, diagnostics = _semantic_capture_proposal(
        _context(own=own, enemies=(prey,)),
        semantic=_semantic("1:0"),
    )

    assert proposal is None
    assert not diagnostics["checks"]["target_mass"]
