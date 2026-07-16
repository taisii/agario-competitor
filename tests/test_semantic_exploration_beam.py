from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.semantic_exploration_beam import (  # noqa: E402
    SemanticExplorationBeamStrategy,
    _split_mobility_profile,
)


def _context(
    own: tuple[BlobModel, ...],
    *,
    rankings: tuple[int, ...],
    round_number: int,
    enemies: tuple[VisibleBlobModel, ...] = (),
    viruses: tuple[VirusModel, ...] = (),
) -> StrategyContext:
    state = SimpleNamespace(
        me=SimpleNamespace(player_id=0, blobs={blob.blob_id: blob for blob in own}),
        visible_food=[],
        visible_viruses=list(viruses),
        visible_blobs=list(enemies),
        rankings=list(rankings),
        round=round_number,
        max_rounds=1400,
        map=SimpleNamespace(size=60.0),
    )
    return StrategyContext(game=SimpleNamespace(state=state), query=SimpleNamespace())


def test_no_enemy_adds_one_exploration_split_to_existing_candidates() -> None:
    strategy = SemanticExplorationBeamStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=8.0),)

    decision = strategy.choose(
        _context(own, rankings=(0, 1, 2), round_number=1000)
    )

    candidate_scores = decision.diagnostics["candidate_scores"]
    assert "continue" in candidate_scores
    assert "explore_split_continue" in candidate_scores
    assert decision.diagnostics["candidate_count"] == 2


def test_recent_non_leader_mass_continuously_prices_split_exposure() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=8.0),)
    informed = SemanticExplorationBeamStrategy()
    uninformed = SemanticExplorationBeamStrategy()
    informed._unseen_enemy_turns = 20
    uninformed._unseen_enemy_turns = 20

    informed.choose(_context(own, rankings=(1, 0, 2), round_number=999))
    informed_decision = informed.choose(
        _context(own, rankings=(0, 1, 2), round_number=1000)
    )
    uninformed_decision = uninformed.choose(
        _context(own, rankings=(0, 1, 2), round_number=1000)
    )

    informed_score = informed_decision.diagnostics["candidate_scores"][
        "explore_split_continue"
    ]
    uninformed_score = uninformed_decision.diagnostics["candidate_scores"][
        "explore_split_continue"
    ]
    assert informed._rival_mass_lower_bound > 63.0
    assert informed_score < uninformed_score - 30.0


def test_blind_split_value_falls_when_no_payback_horizon_remains() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=8.0),)
    early = SemanticExplorationBeamStrategy()
    late = SemanticExplorationBeamStrategy()
    early._unseen_enemy_turns = 20
    late._unseen_enemy_turns = 20

    early_decision = early.choose(
        _context(own, rankings=(0, 1, 2), round_number=1200)
    )
    late_decision = late.choose(
        _context(own, rankings=(0, 1, 2), round_number=1399)
    )

    early_score = early_decision.diagnostics["candidate_scores"][
        "explore_split_continue"
    ]
    late_score = late_decision.diagnostics["candidate_scores"][
        "explore_split_continue"
    ]
    assert late_score < early_score


def test_consumable_virus_adds_bounded_split_harvest_candidate() -> None:
    strategy = SemanticExplorationBeamStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    viruses = (VirusModel(virus_id=7, pos=(40.0, 30.0), radius=1.5),)

    decision = strategy.choose(
        _context(
            own,
            rankings=(0, 1, 2),
            round_number=900,
            viruses=viruses,
        )
    )

    assert set(decision.diagnostics["candidate_scores"]) == {
        "continue",
        "nearest_virus",
        "split_nearest_virus",
    }


def test_split_virus_candidate_requires_post_split_consumer() -> None:
    strategy = SemanticExplorationBeamStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),)
    viruses = (VirusModel(virus_id=7, pos=(36.0, 30.0), radius=1.5),)

    decision = strategy.choose(
        _context(
            own,
            rankings=(0, 1, 2),
            round_number=900,
            viruses=viruses,
        )
    )

    assert "split_nearest_virus" not in decision.diagnostics["candidate_scores"]


def test_visible_fragments_form_player_level_rival_lower_bound() -> None:
    strategy = SemanticExplorationBeamStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=8.0),)
    enemies = (
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=0,
            pos=(20.0, 20.0),
            radius=5.0,
        ),
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=1,
            pos=(22.0, 20.0),
            radius=4.0,
        ),
    )

    strategy.choose(
        _context(
            own,
            rankings=(0, 1, 2),
            round_number=1000,
            enemies=enemies,
        )
    )

    assert strategy._rival_mass_lower_bound == 41.0


def test_split_profile_reports_post_split_largest_blob_mass() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=8.0),)

    mobility, anchor_loss, post_split_anchor = _split_mobility_profile(own)

    assert mobility > 0.0
    assert anchor_loss == 32.0
    assert post_split_anchor == 32.0
