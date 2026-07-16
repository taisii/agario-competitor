from __future__ import annotations

"""ReplayDominance scoring with semantic capture proposals."""

from dataclasses import replace

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import can_eat_player_blob
from strategies.receding_horizon import Action, ReplayDominanceStrategy, SearchNode
from strategies.semantic_potential import SemanticLookaheadStrategy


SEMANTIC_PROPOSAL_MAX_ROUND_FRACTION = 0.45
SEMANTIC_PROPOSAL_MAX_OWN_MASS = 8.0
SEMANTIC_PROPOSAL_MAX_CONTACT_TURNS = 4.0
SEMANTIC_PROPOSAL_MIN_TARGET_MASS = 0.5
SEMANTIC_PROPOSAL_MIN_MASS_SHARE = 0.12


class OutcomeTeacherHybridStrategy(ReplayDominanceStrategy):
    """Let semantic find prey routes and replay_dominance decide their value."""

    name = "outcome_teacher_hybrid"

    def __init__(self) -> None:
        super().__init__()
        self._semantic = SemanticLookaheadStrategy()
        self._semantic_proposal: Action | None = None
        self._semantic_proposal_diagnostics: dict[str, object] = {}

    def choose(self, context: StrategyContext) -> StrategyDecision:
        if _semantic_proposal_scene(context):
            semantic = self._semantic.choose(context)
            (
                self._semantic_proposal,
                self._semantic_proposal_diagnostics,
            ) = _semantic_capture_proposal(context, semantic=semantic)
        else:
            self._semantic_proposal = None
            self._semantic_proposal_diagnostics = {
                "proposal_offered": False,
                "prefilter_passed": False,
            }
        decision = super().choose(context)
        diagnostics = dict(decision.diagnostics)
        diagnostics["outcome_teacher"] = {
            **self._semantic_proposal_diagnostics,
            "selected_semantic_proposal": decision.reason == "semantic_prey",
        }
        return replace(decision, diagnostics=diagnostics)

    def _additional_proxy_actions(
        self,
        *,
        node: SearchNode,
        first_step: bool,
    ) -> tuple[Action, ...]:
        if not first_step or self._semantic_proposal is None:
            return ()
        return (self._semantic_proposal,)


def _semantic_proposal_scene(context: StrategyContext) -> bool:
    """Reject ordinary turns before paying for the second policy."""

    state = context.game.state
    own = tuple(state.me.blobs.values())
    enemies = tuple(state.visible_blobs)
    round_number = int(getattr(state, "round", 0))
    max_rounds = max(1, int(getattr(state, "max_rounds", 1400)))
    if (
        len(own) != 1
        or len(enemies) != 1
        or round_number >= max_rounds * SEMANTIC_PROPOSAL_MAX_ROUND_FRACTION
    ):
        return False
    blob = own[0]
    enemy = enemies[0]
    own_mass = blob.radius * blob.radius
    target_mass = enemy.radius * enemy.radius
    return (
        own_mass <= SEMANTIC_PROPOSAL_MAX_OWN_MASS
        and can_eat_player_blob(
            blob.radius,
            enemy.radius,
            radius_margin=1.03,
        )
        and target_mass >= SEMANTIC_PROPOSAL_MIN_TARGET_MASS
        and target_mass / own_mass >= SEMANTIC_PROPOSAL_MIN_MASS_SHARE
    )


def _semantic_capture_proposal(
    context: StrategyContext,
    *,
    semantic: StrategyDecision,
) -> tuple[Action | None, dict[str, object]]:
    state = context.game.state
    own = tuple(state.me.blobs.values())
    enemies = tuple(state.visible_blobs)
    round_number = int(getattr(state, "round", 0))
    max_rounds = max(1, int(getattr(state, "max_rounds", 1400)))
    contact_turns = semantic.diagnostics.get("selected_contact_turns")
    target = next(
        (
            enemy
            for enemy in enemies
            if f"{enemy.player_id}:{enemy.blob_id}" == semantic.target_id
        ),
        None,
    )
    own_mass = sum(blob.radius * blob.radius for blob in own)
    target_mass = 0.0 if target is None else target.radius * target.radius
    checks = {
        "semantic_capture": (
            semantic.target_kind == "prey"
            and semantic.reason in {"capture_enemy", "intercept_enemy"}
        ),
        "non_split": not semantic.split,
        "single_blob": len(own) == 1,
        "low_capital": own_mass <= SEMANTIC_PROPOSAL_MAX_OWN_MASS,
        "early_or_middle": (
            round_number
            < max_rounds * SEMANTIC_PROPOSAL_MAX_ROUND_FRACTION
        ),
        "isolated_prey": len(enemies) == 1,
        "target_present": target is not None,
        "target_mass": target_mass >= SEMANTIC_PROPOSAL_MIN_TARGET_MASS,
        "target_mass_share": (
            own_mass > 0.0
            and target_mass / own_mass >= SEMANTIC_PROPOSAL_MIN_MASS_SHARE
        ),
        "short_contact": (
            isinstance(contact_turns, (int, float))
            and contact_turns <= SEMANTIC_PROPOSAL_MAX_CONTACT_TURNS
        ),
    }
    offered = all(checks.values())
    return (
        (
            Action(
                direction=semantic.direction,
                split=False,
                reason="semantic_prey",
            )
            if offered
            else None
        ),
        {
            "proposal_offered": offered,
            "prefilter_passed": True,
            "checks": checks,
            "round_fraction": round_number / max_rounds,
            "target_mass": target_mass,
            "target_mass_share": (
                target_mass / own_mass if own_mass > 0.0 else 0.0
            ),
            "contact_turns": contact_turns,
        },
    )
