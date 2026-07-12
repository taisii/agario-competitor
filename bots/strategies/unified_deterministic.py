from __future__ import annotations

"""Deterministic-cost root-balanced beam search.

The policy and value function are inherited from :mod:`unified_core`.  This
module changes only search scheduling: every retained root receives exactly one
exact continuation, and branch counts depend on simulator cost rather than on
which semantic action happens to appear before a wall-clock deadline.
"""

import math
from dataclasses import replace

from lib.config.arena import ARENA_SIZE
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import normalise
from strategies.receding_horizon import OwnBlob, SearchNode
from strategies.unified_core import UnifiedCoreStrategy


class UnifiedDeterministicStrategy(UnifiedCoreStrategy):
    name = "unified_deterministic"

    def __init__(self) -> None:
        super().__init__()
        # With deterministic two-ply search, a safe farm split receives its
        # future collection benefit explicitly.  Price the countervailing
        # concentration option as the expected recoverable share of one virus.
        self.capability_option_mass = 2.25

    def _choose(
        self,
        context: StrategyContext,
        *,
        deadline: float,
        turn_budget: float,
    ) -> StrategyDecision:
        state = context.game.state
        own_blobs = tuple(
            OwnBlob(
                blob_id=blob.blob_id,
                x=blob.pos[0],
                y=blob.pos[1],
                radius=blob.radius,
                merge_cooldown=blob.merge_cooldown,
            )
            for blob in state.me.blobs.values()
        )
        if not own_blobs:
            return StrategyDecision(direction=self.previous_direction, reason="dead_fallback")

        round_number = int(state.round)
        arena_size = float(state.map.size or ARENA_SIZE)
        self._read_public_moves(context)
        enemies = self._update_enemy_memory(context, own_blobs, arena_size)

        foods = tuple(state.visible_food)
        viruses = tuple(state.visible_viruses)
        enemies = tuple(enemies)
        food_targets = tuple(self._fragment_food_targets(own_blobs, foods))

        rankings = tuple(int(player_id) for player_id in state.rankings)
        rank_position = self._rank_position(rankings, state.me.player_id)
        self._current_rank_position = rank_position
        self._current_progress = round_number / max(1, int(state.max_rounds))
        try:
            rank_index = rankings.index(int(state.me.player_id))
        except ValueError:
            rank_index = len(rankings)
        self._rival_values = {
            player_id: 1.0 / (1.0 + abs(other_index - rank_index))
            for other_index, player_id in enumerate(rankings)
            if other_index != rank_index
        }

        start = SearchNode(
            own_blobs=own_blobs,
            enemies=enemies,
            score=0.0,
            first_direction=self.previous_direction,
            first_split=False,
            first_reason="keep",
            last_direction=self.previous_direction,
        )
        start = replace(start, score=self._state_value(start, foods, viruses, arena_size))

        root_actions = self._candidate_actions(
            node=start,
            foods=foods,
            food_targets=food_targets,
            viruses=viruses,
            arena_size=arena_size,
            first_step=True,
            angle_offset=round_number,
        )

        # Measured transition cost is dominated by blob-pair stabilisation.
        # Keep two exact root alternatives in every state; only compact states
        # receive one equally allocated continuation per root.
        fragment_count = len(own_blobs)
        exact_budget = 4 if fragment_count <= 3 else 2
        root_limit = 2
        selected_roots = self._select_actions_no_reservation(
            node=start,
            actions=root_actions,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            limit=root_limit,
            require_diversity=True,
        )

        root_nodes: list[SearchNode] = []
        rejected: list[SearchNode] = []
        for action in selected_roots:
            result = self._transition(
                node=start,
                action=action,
                foods=foods,
                viruses=viruses,
                arena_size=arena_size,
                first_step=True,
            )
            (rejected if result.fatal else root_nodes).append(result.node)

        if not root_nodes:
            best = max(rejected, key=self._node_order_key) if rejected else start
            reached_depth = 1 if selected_roots else 0
            reason = "least_bad" if rejected else "no_search_result"
            exact_transitions = len(selected_roots)
        else:
            # Give a continuation to every surviving root or to none.  This
            # prevents an operation quota from favouring whichever semantic
            # action happened to be listed first.
            balanced_roots = self._best_per_root(root_nodes)
            exact_transitions = len(selected_roots)
            continuation_budget = exact_budget - exact_transitions
            if continuation_budget >= len(balanced_roots):
                advanced: list[SearchNode] = []
                for node in balanced_roots:
                    actions = self._candidate_actions(
                        node=node,
                        foods=foods,
                        food_targets=food_targets,
                        viruses=viruses,
                        arena_size=arena_size,
                        first_step=False,
                        angle_offset=round_number + 1,
                    )
                    selected = self._select_actions_no_reservation(
                        node=node,
                        actions=actions,
                        foods=foods,
                        viruses=viruses,
                        arena_size=arena_size,
                        limit=1,
                        require_diversity=False,
                    )
                    if not selected:
                        advanced.append(node)
                        continue
                    result = self._transition(
                        node=node,
                        action=selected[0],
                        foods=foods,
                        viruses=viruses,
                        arena_size=arena_size,
                        first_step=False,
                    )
                    exact_transitions += 1
                    advanced.append(result.node)
                best = max(advanced, key=self._node_order_key)
                reached_depth = 2
            else:
                best = max(balanced_roots, key=self._node_order_key)
                reached_depth = 1
            reason = best.first_reason

        direction = normalise(best.first_direction) or self.previous_direction
        self.previous_direction = direction
        return StrategyDecision(
            direction=direction,
            split=best.first_split,
            target_kind=(
                "prey"
                if "prey" in reason or "rival" in reason
                else "resource"
                if "food" in reason or "virus" in reason
                else "beam"
            ),
            reason=reason,
            score=self._terminal_score(best),
            diagnostics={
                "depth": reached_depth,
                "root_candidates": len(root_actions),
                "root_selected": len(selected_roots),
                "exact_transitions": exact_transitions,
                "rank": rank_position,
                "progress": round(self._current_progress, 4),
                "projected_food": best.projected_food,
                "projected_captures": best.projected_captures,
                "projected_blob_count": len(best.own_blobs),
                "min_safety_margin": best.min_safety_margin,
                "deterministic_quota": True,
                "exact_budget": exact_budget,
                "turn_budget_ms": round(turn_budget * 1000.0, 3),
                "compute_spent_ms": round(self.compute_spent_seconds * 1000.0, 3),
            },
        )

    def _select_actions_no_reservation(
        self,
        *,
        node: SearchNode,
        actions,
        foods,
        viruses,
        arena_size: float,
        limit: int,
        require_diversity: bool,
    ):
        """Branch-and-bound screen without a semantic split reservation."""

        if len(actions) <= limit:
            return list(actions)
        screen = self._screening_context(node, foods, viruses, arena_size)
        scored = [
            (
                self._cheap_action_score(node, action, foods, arena_size, screen),
                self._action_order_key(action),
                action,
            )
            for action in actions
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = []
        for _, _, action in scored:
            if require_diversity and any(
                existing.split == action.split
                and abs(
                    (
                        (
                            math.atan2(existing.direction[1], existing.direction[0])
                            - math.atan2(action.direction[1], action.direction[0])
                            + math.pi
                        )
                        % (2.0 * math.pi)
                    )
                    - math.pi
                )
                < math.pi / 18.0
                for existing in selected
            ):
                continue
            selected.append(action)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _action_order_key(action):
        angle = math.atan2(action.direction[1], action.direction[0]) % (2.0 * math.pi)
        angle_bin = int(round(angle / (2.0 * math.pi) * 4096)) % 4096
        return (action.split, angle_bin, action.reason)

    def _node_order_key(self, node: SearchNode):
        angle = math.atan2(node.first_direction[1], node.first_direction[0]) % (2.0 * math.pi)
        angle_bin = int(round(angle / (2.0 * math.pi) * 4096)) % 4096
        return (self._terminal_score(node), not node.first_split, -angle_bin)
