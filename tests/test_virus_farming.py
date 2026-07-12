from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.config.arena import MAX_BLOB_COUNT  # noqa: E402
from lib.config.player import EAT_SIZE_RATIO, MASS_DECAY_RATE  # noqa: E402
from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.virus_farming import PotentialFieldVirusFarmerStrategy  # noqa: E402
from strategies.registry import create_strategy  # noqa: E402


def _game(
    own: tuple[BlobModel, ...],
    *,
    enemies: tuple[VisibleBlobModel, ...] = (),
    food: tuple[FoodModel, ...] = (),
    viruses: tuple[VirusModel, ...] = (),
) -> SimpleNamespace:
    total_mass = sum(blob.radius * blob.radius for blob in own)
    center = (
        sum(blob.pos[0] * blob.radius * blob.radius for blob in own) / total_mass,
        sum(blob.pos[1] * blob.radius * blob.radius for blob in own) / total_mass,
    )
    me = SimpleNamespace(
        player_id=0,
        blobs={blob.blob_id: blob for blob in own},
        pos=center,
        radius=math.sqrt(total_mass),
        alive=True,
    )
    state = SimpleNamespace(
        me=me,
        visible_blobs=list(enemies),
        visible_food=list(food),
        visible_viruses=list(viruses),
        map=SimpleNamespace(size=60.0),
        round=400,
        max_rounds=1400,
        rankings=[0, 1],
    )
    return SimpleNamespace(state=state)


def _context(game: SimpleNamespace) -> StrategyContext:
    return StrategyContext(game=game, query=SimpleNamespace())


def test_potential_field_virus_farmer_prioritises_viable_virus_over_prey_and_food() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    harmless_prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(13.0, 10.0),
        radius=0.4,
    )
    food = FoodModel(food_id=1, pos=(9.0, 10.0))
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = PotentialFieldVirusFarmerStrategy().choose(
        _context(
            _game(
                (own,),
                enemies=(harmless_prey,),
                food=(food,),
                viruses=(virus,),
            )
        )
    )

    assert decision.target_kind == "virus"
    assert decision.target_id == "7"
    assert decision.direction == (1.0, 0.0)
    assert not decision.split
    assert decision.reason == "reachable_virus"
    assert decision.diagnostics["potential_field_virus_farmer_mode"] == "pursue_virus"


def test_potential_field_virus_farmer_aims_from_individually_capable_fragment() -> None:
    capable = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    nearer_but_small = BlobModel(blob_id=1, pos=(20.0, 10.0), radius=1.0)
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = PotentialFieldVirusFarmerStrategy().choose(
        _context(_game((capable, nearer_but_small), viruses=(virus,)))
    )

    assert decision.target_kind == "virus"
    assert decision.direction == (1.0, 0.0)
    assert decision.diagnostics["hunter_blob_id"] == capable.blob_id


def test_potential_field_virus_farmer_does_not_use_aggregate_radius_as_eating_power() -> None:
    fragments = (
        BlobModel(
            blob_id=0,
            pos=(10.0, 10.0),
            radius=1.2,
            merge_cooldown=12,
        ),
        BlobModel(
            blob_id=1,
            pos=(12.5, 10.0),
            radius=1.2,
            merge_cooldown=12,
        ),
    )
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = PotentialFieldVirusFarmerStrategy().choose(
        _context(_game(fragments, viruses=(virus,)))
    )

    assert decision.target_kind != "virus"
    assert decision.diagnostics["currently_consumable_pairs"] == 0
    assert decision.diagnostics["virus_unavailable_reason"] == (
        "insufficient_blob_mass"
    )


def test_potential_field_virus_farmer_merge_cooldown_does_not_delay_virus_contact() -> None:
    # Since 2026.1.13 the virus center, not its edge, must enter the blob.
    virus = VirusModel(virus_id=7, pos=(12.5, 10.0), radius=1.5)
    threshold_mass = virus.radius * virus.radius * EAT_SIZE_RATIO
    one_turn_safe_mass = threshold_mass / (1.0 - MASS_DECAY_RATE) + 1e-6
    own = BlobModel(
        blob_id=0,
        pos=(10.0, 10.0),
        radius=math.sqrt(one_turn_safe_mass),
        merge_cooldown=18,
    )

    decision = PotentialFieldVirusFarmerStrategy().choose(
        _context(_game((own,), viruses=(virus,)))
    )

    assert decision.target_kind == "virus"
    assert decision.diagnostics["turns_to_contact"] == 1
    assert decision.diagnostics["projected_mass_at_contact"] > threshold_mass


def test_potential_field_virus_farmer_reports_growth_only_consumption_at_blob_cap() -> None:
    hunter = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    fragments = tuple(
        BlobModel(
            blob_id=index,
            pos=(20.0 + index, 20.0),
            radius=0.4,
            merge_cooldown=12,
        )
        for index in range(1, MAX_BLOB_COUNT)
    )
    virus = VirusModel(virus_id=7, pos=(13.6, 10.0), radius=1.5)

    decision = PotentialFieldVirusFarmerStrategy().choose(
        _context(_game((hunter, *fragments), viruses=(virus,)))
    )

    assert decision.target_kind == "virus"
    assert decision.diagnostics["projected_pieces_created"] == 1


def test_potential_field_virus_farmer_uses_reachable_prey_split_without_visible_virus() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(14.0, 10.0),
        radius=0.8,
    )

    decision = PotentialFieldVirusFarmerStrategy().choose(
        _context(_game((own,), enemies=(prey,)))
    )

    assert decision.target_kind == "prey"
    assert decision.direction[0] > 0.0
    assert decision.split
    assert decision.diagnostics["potential_field_virus_farmer_mode"] == "potential_growth"
    assert decision.diagnostics["offensive_split_allowed"] is True


def test_potential_field_virus_farmer_does_not_resplit_fragmented_player() -> None:
    own = (
        BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0),
        BlobModel(blob_id=1, pos=(8.0, 10.0), radius=2.0),
    )
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(14.0, 10.0),
        radius=0.8,
    )

    decision = PotentialFieldVirusFarmerStrategy().choose(
        _context(_game(own, enemies=(prey,)))
    )

    assert decision.target_kind == "prey"
    assert not decision.split
    assert decision.diagnostics["offensive_split_requested"] is True
    assert decision.diagnostics["split_suppressed_reason"] == "fragmented"


def test_potential_field_virus_farmer_keeps_immediate_escape_authoritative() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    predator = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(12.0, 10.0),
        radius=3.0,
    )
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = PotentialFieldVirusFarmerStrategy().choose(
        _context(_game((own,), enemies=(predator,), viruses=(virus,)))
    )

    assert decision.target_kind == "escape"
    assert decision.direction[0] < 0.0
    assert decision.diagnostics["potential_field_virus_farmer_mode"] == "emergency_escape"


def test_potential_field_virus_farmer_rejects_post_split_predator_risk() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)
    post_split_predator = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(18.5, 10.0),
        radius=1.2,
    )

    decision = PotentialFieldVirusFarmerStrategy().choose(
        _context(
            _game(
                (own,),
                enemies=(post_split_predator,),
                viruses=(virus,),
            )
        )
    )

    assert decision.target_kind != "virus"
    assert decision.diagnostics["virus_unavailable_reason"] == (
        "post_split_predator_risk"
    )
    assert decision.diagnostics["post_split_rejected_pairs"] == 1


def test_potential_field_virus_farmer_preserves_mass_target_from_virus_split() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=7.0)
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = PotentialFieldVirusFarmerStrategy().choose(
        _context(_game((own,), viruses=(virus,)))
    )

    assert decision.target_kind != "virus"
    assert decision.diagnostics["virus_unavailable_reason"] == (
        "mass_target_preservation"
    )
    assert decision.diagnostics["mass_target_rejected_pairs"] == 1


def test_potential_field_virus_farmer_keeps_mass_target_latched_after_decay() -> None:
    strategy = PotentialFieldVirusFarmerStrategy()
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)
    reached = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=math.sqrt(40.1))
    decayed = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=math.sqrt(39.9))

    strategy.choose(_context(_game((reached,), viruses=(virus,))))
    decision = strategy.choose(_context(_game((decayed,), viruses=(virus,))))

    assert decision.target_kind != "virus"
    assert decision.diagnostics["mass_target_latched"] is True
    assert decision.diagnostics["virus_unavailable_reason"] == (
        "mass_target_preservation"
    )


def test_potential_field_virus_farmer_is_registered() -> None:
    strategy = create_strategy("potential_field_virus_farmer")

    assert isinstance(strategy, PotentialFieldVirusFarmerStrategy)
