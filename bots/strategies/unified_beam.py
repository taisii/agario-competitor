from __future__ import annotations

"""Root-balanced receding-horizon search for the official 2026.1.13 engine.

The policy has one endpoint value function.  Food, viruses, prey, fragment
coverage, walls, and predators enter that value continuously.  Candidate names
are diagnostics only; they do not select a phase or bypass the search.

Search is split into two inexpensive operations:

* score every root direction with a cheap projected state;
* run the exact public-engine transition for a small, angle-diverse set, then
  extend every surviving root equally before pruning roots.

This prevents a short per-turn deadline from turning action-list order into the
actual policy.
"""

import math
import os
from dataclasses import replace
from time import perf_counter

from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT
from lib.config.player import (
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MASS_DECAY_RATE,
    SPLIT_MIN_MASS,
    STARTING_RADIUS,
)
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import can_eat_player_blob, normalise, squared_distance
from strategies.receding_horizon import (
    EPSILON,
    SQRT2,
    TAU,
    Action,
    EnemyBlob,
    EnemyTrack,
    OwnBlob,
    ReplayDominanceStrategy,
    SearchNode,
    StepResult,
    _can_consume_virus,
    _can_split_eat,
    _clamp,
    _decayed_radius,
    _damped,
    _rotate,
    _speed,
    _split_attack_reach,
    _with_grown_radius,
)


def _sigmoid_negative_margin(margin: float, scale: float) -> float:
    """Probability-like hazard: 0.5 at zero margin, near 1 inside reach."""

    z = max(-60.0, min(60.0, margin / max(scale, 1e-6)))
    return 1.0 / (1.0 + math.exp(z))


def _angle_key(direction: tuple[float, float], bins: int = 96) -> int:
    return int(round(math.atan2(direction[1], direction[0]) / TAU * bins)) % bins


class UnifiedBeamStrategy(ReplayDominanceStrategy):
    """A single-value, root-balanced beam search."""

    name = "unified_beam"

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        super().__init__(
            depth=3 if depth is None else depth,
            width=5 if width is None else width,
            angular_samples=12 if angular_samples is None else angular_samples,
        )
        # 3 ms * 1,400 = 4.2 s.  The accounting bank leaves room for parsing,
        # validation, and occasional expensive sixteen-fragment transitions.
        self.max_turn_seconds = float(os.environ.get("BOT_UNIFIED_MAX_TURN_SECONDS", "0.003"))
        self.compute_budget_seconds = float(
            os.environ.get("BOT_UNIFIED_TOTAL_BUDGET_SECONDS", "5.1")
        )
        self.max_food = int(os.environ.get("BOT_UNIFIED_MAX_FOOD", "20"))
        self.max_enemies = int(os.environ.get("BOT_UNIFIED_MAX_ENEMIES", "10"))
        self._current_progress = 0.0
        self._current_rank_position = 8
        self._state_value_calls = 0
        self.corridor_blend = 0.4
        self.food_value_limit = 16
        self.cheap_food_value_limit = 9
        self.steering_cost = 0.01

    # ------------------------------------------------------------------
    # Observation and search
    # ------------------------------------------------------------------

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

        # Keep resources closest to any fragment.  A mass-centre sort discards
        # wall food and flank targets that are already close to an outer piece.
        def resource_gap(item: object) -> float:
            pos = item.pos  # type: ignore[attr-defined]
            return min(max(0.0, math.dist(blob.pos, pos) - blob.radius) for blob in own_blobs)

        # Preserve the authoritative map order for the physical transition.
        # Candidate generation and endpoint evaluation rank opportunities
        # independently, so mutating resource order here only corrupts ties
        # after an earlier eater grows or is clamped.
        foods = tuple(state.visible_food)
        viruses = tuple(state.visible_viruses)
        center = self._mass_center(own_blobs)
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
        start = replace(
            start,
            score=self._state_value(start, foods, viruses, arena_size),
        )

        root_actions = self._candidate_actions(
            node=start,
            foods=foods,
            food_targets=food_targets,
            viruses=viruses,
            arena_size=arena_size,
            first_step=True,
            angle_offset=round_number,
        )
        root_limit = 9 if len(own_blobs) <= 4 else (7 if len(own_blobs) <= 10 else 5)
        selected_roots = self._select_actions(
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
        root_evaluated = 0
        # Root actions are ordered by a value projection, not by semantic type.
        # A partial root therefore remains a best-first approximation rather
        # than an arbitrary prefix such as escape -> virus -> food.
        for action in selected_roots:
            if root_evaluated and perf_counter() >= deadline:
                break
            result = self._transition(
                node=start,
                action=action,
                foods=foods,
                viruses=viruses,
                arena_size=arena_size,
                first_step=True,
            )
            root_evaluated += 1
            (rejected if result.fatal else root_nodes).append(result.node)

        if not root_nodes:
            if rejected:
                best = max(rejected, key=self._terminal_score)
                reason = "least_bad"
            else:
                best = start
                reason = "no_search_result"
            reached_depth = 1 if root_evaluated else 0
            search_timed_out = perf_counter() >= deadline
        else:
            # Keep one continuation per root.  Global top-k pruning at depth 1
            # can otherwise erase all but one first action before it receives
            # the same lookahead as its competitors.
            frontier = self._best_per_root(root_nodes)
            frontier = sorted(frontier, key=self._terminal_score, reverse=True)[: max(self.width + 2, 5)]
            reached_depth = 1
            search_timed_out = root_evaluated < len(selected_roots)

            for depth_index in range(1, max(1, self.depth)):
                if perf_counter() >= deadline:
                    search_timed_out = True
                    break
                completed_layer: list[SearchNode] = []
                layer_complete = True
                # Every retained root gets a turn before any root is discarded.
                for node in frontier:
                    if perf_counter() >= deadline:
                        layer_complete = False
                        break
                    local_actions = self._candidate_actions(
                        node=node,
                        foods=foods,
                        food_targets=food_targets,
                        viruses=viruses,
                        arena_size=arena_size,
                        first_step=False,
                        angle_offset=round_number + depth_index,
                    )
                    local_limit = 3 if len(node.own_blobs) <= 8 else 2
                    selected = self._select_actions(
                        node=node,
                        actions=local_actions,
                        foods=foods,
                        viruses=viruses,
                        arena_size=arena_size,
                        limit=local_limit,
                        require_diversity=False,
                    )
                    continuations: list[SearchNode] = []
                    for action in selected:
                        # Complete a root's small local bundle once started.
                        result = self._transition(
                            node=node,
                            action=action,
                            foods=foods,
                            viruses=viruses,
                            arena_size=arena_size,
                            first_step=False,
                        )
                        if not result.fatal:
                            continuations.append(result.node)
                    if continuations:
                        completed_layer.append(max(continuations, key=self._terminal_score))
                    else:
                        completed_layer.append(node)

                if not layer_complete:
                    search_timed_out = True
                    break
                frontier = sorted(
                    self._best_per_root(completed_layer),
                    key=self._terminal_score,
                    reverse=True,
                )[: max(1, self.width)]
                reached_depth = depth_index + 1

            best = max(frontier, key=self._terminal_score)
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
                "root_evaluated": root_evaluated,
                "rank": rank_position,
                "progress": round(self._current_progress, 4),
                "projected_food": best.projected_food,
                "projected_captures": best.projected_captures,
                "projected_blob_count": len(best.own_blobs),
                "min_safety_margin": best.min_safety_margin,
                "search_timed_out": search_timed_out,
                "turn_budget_ms": round(turn_budget * 1000.0, 3),
                "compute_spent_ms": round(self.compute_spent_seconds * 1000.0, 3),
            },
        )

    def _best_per_root(self, nodes: list[SearchNode]) -> list[SearchNode]:
        best: dict[tuple[int, bool], SearchNode] = {}
        for node in nodes:
            key = (_angle_key(node.first_direction), node.first_split)
            incumbent = best.get(key)
            if incumbent is None or self._terminal_score(node) > self._terminal_score(incumbent):
                best[key] = node
        return list(best.values())

    # ------------------------------------------------------------------
    # Candidate generation and cheap root screening
    # ------------------------------------------------------------------

    def _candidate_actions(
        self,
        *,
        node: SearchNode,
        foods,
        food_targets,
        viruses,
        arena_size: float,
        first_step: bool,
        allow_split: bool = True,
        angle_offset: int = 0,
    ) -> tuple[Action, ...]:
        actions: list[Action] = []

        actions.append(Action(node.last_direction, reason="keep" if first_step else "continue"))
        steer_angles = (-math.pi / 6, -math.pi / 12, math.pi / 12, math.pi / 6)
        for angle in steer_angles:
            actions.append(Action(_rotate(node.last_direction, angle), reason="steer"))

        escape = self._escape_vector(node)
        if escape != (0.0, 0.0):
            actions.extend(
                (
                    Action(escape, reason="escape"),
                    Action(_rotate(escape, math.pi / 6), reason="escape_tangent"),
                    Action(_rotate(escape, -math.pi / 6), reason="escape_tangent"),
                )
            )

        # Each resource is approached from the fragment with the least
        # centre-clearance, not from the aggregate centre.
        available_food = [food for food in foods if food.food_id not in node.eaten_food_ids]
        food_pairs: list[tuple[float, int, OwnBlob, object]] = []
        for food in available_food:
            origin = min(node.own_blobs, key=lambda blob: max(0.0, math.dist(blob.pos, food.pos) - blob.radius))
            gap = max(0.0, math.dist(origin.pos, food.pos) - origin.radius)
            food_pairs.append((gap, food.food_id, origin, food))
        food_pairs.sort(key=lambda item: (item[0], item[1]))
        for _, _, origin, food in food_pairs[: 4 if first_step else 2]:
            actions.append(
                Action(
                    normalise((food.pos[0] - origin.x, food.pos[1] - origin.y)),
                    reason="fragment_food",
                )
            )
        for target in food_targets[: 3 if first_step else 1]:
            origin = min(node.own_blobs, key=lambda blob: squared_distance(blob.pos, target))
            actions.append(
                Action(normalise((target[0] - origin.x, target[1] - origin.y)), reason="food_density")
            )

        virus_actions: list[tuple[float, int, Action]] = []
        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            for origin in node.own_blobs:
                if not self._can_still_consume_virus_at_contact(origin, virus):
                    continue
                gap = max(0.0, math.dist(origin.pos, virus.pos) - origin.radius)
                direction = normalise((virus.pos[0] - origin.x, virus.pos[1] - origin.y))
                virus_actions.append((gap, virus.virus_id, Action(direction, reason="virus")))
        virus_actions.sort(key=lambda item: (item[0], item[1]))
        actions.extend(item[2] for item in virus_actions[: 4 if first_step else 2])

        prey_actions: list[tuple[float, float, Action]] = []
        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            rival = self._rival_values.get(enemy.player_id, 0.25)
            for origin in node.own_blobs:
                normal_possible = can_eat_player_blob(origin.radius, enemy.radius)
                split_possible = (
                    len(node.own_blobs) < MAX_BLOB_COUNT
                    and _can_split_eat(origin.radius, enemy.radius)
                )
                if not normal_possible and not split_possible:
                    continue
                direction = self._intercept_direction(origin, enemy)
                gap = max(0.0, math.dist(origin.pos, enemy.pos) - origin.radius)
                priority = rival * enemy.mass / (1.0 + gap)
                prey_actions.append((-priority, gap, Action(direction, reason="rival_prey" if rival >= 0.5 else "prey")))
                if split_possible:
                    prey_actions.append((-priority * 1.05, gap, Action(direction, split=True, reason="split_rival_prey" if rival >= 0.5 else "split_prey")))
        prey_actions.sort(key=lambda item: (item[0], item[1]))
        actions.extend(item[2] for item in prey_actions[: 6 if first_step else 3])

        # Center and wall normals are ordinary directions whose value is
        # decided by resources and risk.  They are not phase transitions.
        center = node.center
        actions.append(
            Action(
                normalise((arena_size * 0.5 - center[0], arena_size * 0.5 - center[1])),
                reason="center",
            )
        )
        wall = self._wall_vector(node.primary, arena_size)
        if wall != (0.0, 0.0):
            actions.append(Action(wall, reason="wall_normal"))

        if first_step:
            sample_count = max(8, self.angular_samples)
            for index in range(sample_count):
                angle = TAU * ((index + angle_offset) % sample_count) / sample_count
                actions.append(Action((math.cos(angle), math.sin(angle)), reason="angle"))

        # Split is another control bit.  Offer it for a compact direction set
        # whenever the engine permits at least one split; the endpoint value
        # decides whether coverage/capture outweighs vulnerability.
        can_split = (
            allow_split
            and len(node.own_blobs) < MAX_BLOB_COUNT
            and any(blob.mass >= SPLIT_MIN_MASS for blob in node.own_blobs)
        )
        if can_split:
            split_sources = [
                action
                for action in actions
                if not action.split
                and action.reason in {
                    "keep",
                    "continue",
                    "escape",
                    "escape_tangent",
                    "fragment_food",
                    "food_density",
                    "virus",
                    "prey",
                    "rival_prey",
                }
            ]
            for action in split_sources[: 7 if first_step else 3]:
                actions.append(
                    Action(action.direction, split=True, reason=f"split_{action.reason}")
                )

        return self._dedupe_actions(actions)

    def _fragment_food_targets(
        self,
        own_blobs: tuple[OwnBlob, ...],
        foods: tuple[object, ...],
    ) -> list[tuple[float, float]]:
        if not foods:
            return []
        scored: list[tuple[float, tuple[float, float]]] = []
        for food in foods:
            neighbours = [
                other
                for other in foods
                if squared_distance(food.pos, other.pos) <= 12.25
            ]
            target = (
                sum(other.pos[0] for other in neighbours) / len(neighbours),
                sum(other.pos[1] for other in neighbours) / len(neighbours),
            )
            origin = min(own_blobs, key=lambda blob: squared_distance(blob.pos, target))
            gap = max(0.0, math.dist(origin.pos, target) - origin.radius)
            scored.append(((len(neighbours) + 0.35) / (1.0 + gap), target))
        scored.sort(key=lambda item: item[0], reverse=True)
        result: list[tuple[float, float]] = []
        seen: set[tuple[int, int]] = set()
        for _, target in scored:
            key = (round(target[0] * 2), round(target[1] * 2))
            if key in seen:
                continue
            seen.add(key)
            result.append(target)
            if len(result) >= 6:
                break
        return result

    def _select_actions(
        self,
        *,
        node: SearchNode,
        actions: tuple[Action, ...],
        foods,
        viruses,
        arena_size: float,
        limit: int,
        require_diversity: bool,
    ) -> list[Action]:
        if len(actions) <= limit:
            return list(actions)

        screen = self._screening_context(node, foods, viruses, arena_size)
        scored = [
            (
                self._cheap_action_score(
                    node,
                    action,
                    foods,
                    arena_size,
                    screen,
                ),
                index,
                action,
            )
            for index, action in enumerate(actions)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))

        selected: list[Action] = []
        for _, _, action in scored:
            if require_diversity and selected:
                # Preserve split/non-split alternatives and suppress only
                # nearly identical directions of the same control type.
                if any(
                    existing.split == action.split
                    and abs(
                        ((math.atan2(existing.direction[1], existing.direction[0])
                          - math.atan2(action.direction[1], action.direction[0])
                          + math.pi) % TAU) - math.pi
                    ) < math.pi / 18
                    for existing in selected
                ):
                    continue
            selected.append(action)
            if len(selected) >= limit:
                break

        # Ensure the best legal split competes at the exact root when one was
        # screened out only by angular similarity.
        if require_diversity and selected and not any(action.split for action in selected):
            split_candidate = next((item[2] for item in scored if item[2].split), None)
            if split_candidate is not None and limit >= 2:
                selected[-1] = split_candidate
        return selected

    def _screening_context(self, node, foods, viruses, arena_size: float):
        """Return the first derivative of the endpoint value with respect to motion.

        This computation is shared by every candidate.  The previous version
        recomputed the complete endpoint value for each angle, spending about
        two thirds of the turn budget before one exact transition was run.
        """

        gx = 0.0
        gy = 0.0
        total_mass = max(node.total_mass, EPSILON)

        # Derivative of expected retained mass under the smooth hazard model.
        for own in node.own_blobs:
            for enemy in node.enemies:
                if not can_eat_player_blob(enemy.radius, own.radius):
                    continue
                distance = max(0.25, math.dist(own.pos, enemy.pos))
                uncertainty = 0.35 + 0.32 * enemy.stale_rounds
                reach = enemy.radius + _speed(enemy.radius) + uncertainty
                if _can_split_eat(enemy.radius, own.radius):
                    reach = max(reach, _split_attack_reach(enemy.radius) + uncertainty)
                margin = distance - reach
                probability = _sigmoid_negative_margin(margin, 1.15)
                derivative = probability * (1.0 - probability) / 1.15
                unit = ((own.x - enemy.x) / distance, (own.y - enemy.y) / distance)
                weight = 135.0 * own.mass / total_mass * derivative
                gx += unit[0] * weight
                gy += unit[1] * weight

        # Opportunity gradients are derivatives of m*exp(-turns/horizon).
        for food in foods:
            if food.food_id in node.eaten_food_ids:
                continue
            origin = min(
                node.own_blobs,
                key=lambda blob: max(0.0, math.dist(blob.pos, food.pos) - blob.radius),
            )
            distance = max(0.25, math.dist(origin.pos, food.pos))
            gap = max(0.0, distance - origin.radius)
            speed = max(_speed(origin.radius), 0.1)
            turns = gap / speed
            derivative = (
                FOOD_RADIUS * FOOD_RADIUS
                * math.exp(-turns / 7.0)
                / (7.0 * speed)
            )
            gx += (food.pos[0] - origin.x) / distance * derivative * 100.0
            gy += (food.pos[1] - origin.y) / distance * derivative * 100.0

        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            best = None
            for origin in node.own_blobs:
                if not self._can_still_consume_virus_at_contact(origin, virus):
                    continue
                distance = max(0.25, math.dist(origin.pos, virus.pos))
                gap = max(0.0, distance - origin.radius)
                speed = max(_speed(origin.radius), 0.1)
                turns = gap / speed
                net = self._virus_retained_net(
                    node,
                    origin,
                    virus,
                    arena_size,
                    coarse=True,
                )
                potential = net * math.exp(-turns / 8.0)
                if best is None or potential > best[0]:
                    best = (potential, distance, speed, origin)
            if best is None:
                continue
            potential, distance, speed, origin = best
            derivative = potential / (8.0 * speed)
            gx += (virus.pos[0] - origin.x) / distance * derivative * 100.0
            gy += (virus.pos[1] - origin.y) / distance * derivative * 100.0

        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            rival = 0.7 + 0.6 * self._rival_values.get(enemy.player_id, 0.25)
            best = None
            for own in node.own_blobs:
                distance = max(0.25, math.dist(own.pos, enemy.pos))
                if can_eat_player_blob(own.radius, enemy.radius):
                    closing = max(0.12, _speed(own.radius) - 0.65 * _speed(enemy.radius))
                    turns = max(0.0, distance - own.radius) / closing
                    potential = enemy.mass * rival * math.exp(-turns / 5.0)
                    if best is None or potential > best[0]:
                        best = (potential, distance, closing, own)
                if len(node.own_blobs) < MAX_BLOB_COUNT and _can_split_eat(own.radius, enemy.radius):
                    split_gap = max(0.0, distance - _split_attack_reach(own.radius))
                    turns = 0.45 + split_gap / max(_speed(own.radius), 0.1)
                    potential = enemy.mass * rival * math.exp(-turns / 5.0)
                    if best is None or potential > best[0]:
                        best = (potential, distance, max(_speed(own.radius), 0.1), own)
            if best is None:
                continue
            potential, distance, speed, own = best
            derivative = potential / (5.0 * speed)
            gx += (enemy.x - own.x) / distance * derivative * 72.0
            gy += (enemy.y - own.y) / distance * derivative * 72.0

        return gx, gy

    def _cheap_action_score(
        self,
        node: SearchNode,
        action: Action,
        foods,
        arena_size: float,
        screen: tuple[float, float],
    ) -> float:
        direction = normalise(action.direction)
        gx, gy = screen
        score = direction[0] * gx + direction[1] * gy

        # Exact clamping is inexpensive and prevents an outward wall direction
        # from winning the derivative screen despite producing no movement.
        expected = 0.0
        useful = 0.0
        for blob in node.own_blobs:
            speed = _speed(blob.radius)
            nx = _clamp(blob.x + direction[0] * speed, blob.radius, arena_size - blob.radius)
            ny = _clamp(blob.y + direction[1] * speed, blob.radius, arena_size - blob.radius)
            expected += blob.mass * speed
            useful += blob.mass * max(
                0.0,
                (nx - blob.x) * direction[0] + (ny - blob.y) * direction[1],
            )
        if expected > EPSILON:
            score -= (1.0 - useful / expected) * 10.0

        steering = 1.0 - max(
            -1.0,
            min(
                1.0,
                node.last_direction[0] * direction[0]
                + node.last_direction[1] * direction[1],
            ),
        )
        score -= steering * self.steering_cost

        if action.split:
            score += self._screen_split_delta(node, direction, foods)
        return score

    def _screen_split_delta(self, node: SearchNode, direction, foods) -> float:
        """Cheap geometric split delta used only for branch ordering."""

        delta = 0.0
        eligible = [
            blob
            for blob in node.own_blobs
            if blob.mass >= SPLIT_MIN_MASS
        ][: max(0, MAX_BLOB_COUNT - len(node.own_blobs))]
        if not eligible:
            return -math.inf

        for origin in eligible:
            child_radius = origin.radius / SQRT2
            # Immediate capture is a real discontinuity in the engine and is
            # therefore retained in the screen rather than represented by a
            # semantic split rule.
            for enemy in node.enemies:
                if not can_eat_player_blob(child_radius, enemy.radius):
                    continue
                rel_x = enemy.x - origin.x
                rel_y = enemy.y - origin.y
                forward = rel_x * direction[0] + rel_y * direction[1]
                lateral = abs(rel_x * direction[1] - rel_y * direction[0])
                reach = 3.0 * child_radius + 1.6 + _speed(child_radius)
                if -0.1 <= forward <= reach and lateral <= child_radius * 1.15:
                    rival = 0.7 + 0.6 * self._rival_values.get(enemy.player_id, 0.25)
                    delta += enemy.mass * rival * 45.0

            # Price the exposed child continuously using the same reach model.
            probe = OwnBlob(
                blob_id=origin.blob_id,
                x=origin.x,
                y=origin.y,
                radius=child_radius,
                merge_cooldown=18,
            )
            hazard, _ = self._blob_hazard(probe, node.enemies, ARENA_SIZE)
            delta -= origin.mass * hazard * 45.0

            # Splitting can collect several pellets in a lane before merging.
            lane = 0
            for food in foods:
                if food.food_id in node.eaten_food_ids:
                    continue
                rel_x = food.pos[0] - origin.x
                rel_y = food.pos[1] - origin.y
                forward = rel_x * direction[0] + rel_y * direction[1]
                lateral = abs(rel_x * direction[1] - rel_y * direction[0])
                if 0.0 <= forward <= 7.0 and lateral <= child_radius * 1.2:
                    lane += 1
            delta += lane * FOOD_RADIUS * FOOD_RADIUS * 30.0

        return delta

    # ------------------------------------------------------------------
    # Exact 2026.1.13 transition, with endpoint rather than accumulated score
    # ------------------------------------------------------------------

    def _transition(
        self,
        *,
        node: SearchNode,
        action: Action,
        foods,
        viruses,
        arena_size: float,
        first_step: bool,
    ) -> StepResult:
        direction = normalise(action.direction)
        own_blobs = list(node.own_blobs)
        if action.split:
            own_blobs = self._apply_split(own_blobs, direction, arena_size)

        before_move = own_blobs
        own_blobs = [self._move_own(blob, direction, arena_size) for blob in own_blobs]
        blocked_movement = self._blocked_movement_distance(before_move, own_blobs, direction)
        enemies = self._move_enemies(node.enemies, own_blobs, arena_size)

        eaten_food_ids = set(node.eaten_food_ids)
        captured_enemy_ids = set(node.captured_enemy_ids)
        consumed_virus_ids = set(node.consumed_virus_ids)
        projected_food = node.projected_food
        projected_captures = node.projected_captures

        own_blobs = [replace(blob, radius=_decayed_radius(blob.radius)) for blob in own_blobs]
        own_blobs = self._stabilise_own_blobs(own_blobs, arena_size)
        enemies = self._stabilise_enemy_blobs(enemies, arena_size)

        own_blobs, enemies = self._resolve_all_viruses(
            own_blobs=own_blobs,
            enemies=enemies,
            viruses=viruses,
            consumed_virus_ids=consumed_virus_ids,
            arena_size=arena_size,
        )
        own_blobs = self._stabilise_own_blobs(own_blobs, arena_size)
        enemies = self._stabilise_enemy_blobs(enemies, arena_size)

        # Food is contested globally.  The official engine awards each pellet
        # to the largest touching blob and clamps that eater after growth.
        own_blobs, enemies, own_food = self._resolve_all_food(
            own_blobs=own_blobs,
            enemies=enemies,
            foods=foods,
            eaten_food_ids=eaten_food_ids,
            arena_size=arena_size,
        )
        projected_food += own_food

        own_blobs, enemies, _, captures = self._resolve_interactions(
            own_blobs,
            enemies,
            captured_enemy_ids,
            arena_size,
        )
        projected_captures += captures
        own_blobs = self._stabilise_own_blobs(own_blobs, arena_size)
        enemies = self._stabilise_enemy_blobs(enemies, arena_size)

        if not own_blobs:
            dead = self._replace_node(
                node=node,
                own_blobs=(),
                enemies=enemies,
                score=-1_000_000.0,
                direction=direction,
                action=action,
                first_step=first_step,
                eaten_food_ids=eaten_food_ids,
                captured_enemy_ids=captured_enemy_ids,
                consumed_virus_ids=consumed_virus_ids,
                projected_food=projected_food,
                projected_captures=projected_captures,
                min_safety_margin=-math.inf,
            )
            return StepResult(dead, fatal=True)

        min_margin = self._minimum_safety_margin(own_blobs, enemies, arena_size)
        next_node = self._replace_node(
            node=node,
            own_blobs=tuple(own_blobs),
            enemies=enemies,
            score=0.0,
            direction=direction,
            action=action,
            first_step=first_step,
            eaten_food_ids=eaten_food_ids,
            captured_enemy_ids=captured_enemy_ids,
            consumed_virus_ids=consumed_virus_ids,
            projected_food=projected_food,
            projected_captures=projected_captures,
            min_safety_margin=min(node.min_safety_margin, min_margin),
        )
        steering = 1.0 - max(-1.0, min(1.0, node.last_direction[0] * direction[0] + node.last_direction[1] * direction[1]))
        control_cost = (
            node.control_cost
            + blocked_movement * 1.2
            + steering * self.steering_cost
        )
        score = self._state_value(next_node, foods, viruses, arena_size) - control_cost
        return StepResult(
            replace(next_node, score=score, control_cost=control_cost),
            fatal=False,
        )

    def _resolve_all_viruses(
        self,
        *,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        viruses,
        consumed_virus_ids: set[int],
        arena_size: float,
    ) -> tuple[list[OwnBlob], tuple[EnemyBlob, ...]]:
        """Resolve visible viruses across own and enemy blobs in engine order."""

        own_by_id = {blob.blob_id: blob for blob in own_blobs}
        enemy_by_key = {enemy.key: enemy for enemy in enemies}
        for virus in viruses:
            if virus.virus_id in consumed_virus_ids:
                continue
            candidates: list[tuple[int, int, str, OwnBlob | EnemyBlob]] = []
            for blob in own_by_id.values():
                if (
                    _can_consume_virus(blob.radius, virus.radius)
                    and squared_distance(blob.pos, virus.pos) <= blob.radius * blob.radius
                ):
                    candidates.append((self._own_player_id, blob.blob_id, "own", blob))
            for enemy in enemy_by_key.values():
                if (
                    _can_consume_virus(enemy.radius, virus.radius)
                    and squared_distance(enemy.pos, virus.pos) <= enemy.radius * enemy.radius
                ):
                    candidates.append((enemy.player_id, enemy.blob_id, "enemy", enemy))
            if not candidates:
                continue
            player_id, blob_id, kind, origin = min(
                candidates,
                key=lambda item: (-item[3].radius, item[0], item[1]),
            )
            consumed_virus_ids.add(virus.virus_id)
            if kind == "own":
                assert isinstance(origin, OwnBlob)
                piece_count = max(1, MAX_BLOB_COUNT - len(own_by_id) + 1)
                piece_radius = math.sqrt(
                    (origin.mass + virus.radius * virus.radius) / piece_count
                )
                fragments = self._virus_replacement_fragments(
                    origin=origin,
                    piece_radius=piece_radius,
                    piece_count=piece_count,
                    arena_size=arena_size,
                    occupied_ids=set(own_by_id),
                )
                del own_by_id[blob_id]
                own_by_id.update({fragment.blob_id: fragment for fragment in fragments})
                continue

            assert isinstance(origin, EnemyBlob)
            player_keys = [key for key in enemy_by_key if key[0] == player_id]
            piece_count = max(1, MAX_BLOB_COUNT - len(player_keys) + 1)
            piece_radius = math.sqrt(
                (origin.mass + virus.radius * virus.radius) / piece_count
            )
            cols = math.ceil(math.sqrt(piece_count))
            rows = math.ceil(piece_count / cols)
            spacing = piece_radius * 2.0 + 1e-6
            x_offset = (cols - 1) * spacing / 2.0
            y_offset = (rows - 1) * spacing / 2.0
            used = {key[1] for key in player_keys}
            next_id = 0
            del enemy_by_key[(player_id, blob_id)]
            for index in range(piece_count):
                row = index // cols
                col = index % cols
                if index == 0:
                    new_id = blob_id
                else:
                    while next_id in used:
                        next_id += 1
                    new_id = next_id
                    used.add(new_id)
                    next_id += 1
                fragment = EnemyBlob(
                    player_id=player_id,
                    blob_id=new_id,
                    x=_clamp(
                        origin.x + col * spacing - x_offset,
                        piece_radius,
                        arena_size - piece_radius,
                    ),
                    y=_clamp(
                        origin.y + row * spacing - y_offset,
                        piece_radius,
                        arena_size - piece_radius,
                    ),
                    radius=piece_radius,
                    direction=origin.direction,
                    stale_rounds=origin.stale_rounds,
                    merge_cooldown=18 if piece_count > 1 else origin.merge_cooldown,
                )
                enemy_by_key[fragment.key] = fragment

        return list(own_by_id.values()), tuple(enemy_by_key.values())

    def _stabilise_enemy_blobs(
        self,
        enemies: tuple[EnemyBlob, ...],
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        by_player: dict[int, list[EnemyBlob]] = {}
        for enemy in enemies:
            by_player.setdefault(enemy.player_id, []).append(enemy)
        result: list[EnemyBlob] = []
        for player_id, group in by_player.items():
            metadata = {enemy.blob_id: enemy for enemy in group}
            converted = [
                OwnBlob(
                    blob_id=enemy.blob_id,
                    x=enemy.x,
                    y=enemy.y,
                    radius=enemy.radius,
                    merge_cooldown=enemy.merge_cooldown,
                )
                for enemy in group
            ]
            stabilised = self._stabilise_own_blobs(converted, arena_size)
            for blob in stabilised:
                source = metadata.get(blob.blob_id, group[0])
                result.append(
                    EnemyBlob(
                        player_id=player_id,
                        blob_id=blob.blob_id,
                        x=blob.x,
                        y=blob.y,
                        radius=blob.radius,
                        direction=source.direction,
                        stale_rounds=source.stale_rounds,
                        merge_cooldown=blob.merge_cooldown,
                    )
                )
        return tuple(result)

    def _resolve_all_food(
        self,
        *,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        foods,
        eaten_food_ids: set[int],
        arena_size: float,
    ) -> tuple[list[OwnBlob], tuple[EnemyBlob, ...], int]:
        own_by_id = {blob.blob_id: blob for blob in own_blobs}
        enemy_by_key = {enemy.key: enemy for enemy in enemies}
        own_food = 0
        for food in foods:
            if food.food_id in eaten_food_ids:
                continue
            candidates: list[tuple[int, int, str, OwnBlob | EnemyBlob]] = []
            for blob in own_by_id.values():
                if squared_distance(blob.pos, food.pos) <= blob.radius * blob.radius:
                    candidates.append((self._own_player_id, blob.blob_id, "own", blob))
            for enemy in enemy_by_key.values():
                if squared_distance(enemy.pos, food.pos) <= enemy.radius * enemy.radius:
                    candidates.append((enemy.player_id, enemy.blob_id, "enemy", enemy))
            if not candidates:
                continue
            player_id, blob_id, kind, eater = min(
                candidates,
                key=lambda item: (-item[3].radius, item[0], item[1]),
            )
            grown = math.sqrt(eater.mass + FOOD_RADIUS * FOOD_RADIUS)
            if kind == "own":
                assert isinstance(eater, OwnBlob)
                own_by_id[blob_id] = _with_grown_radius(eater, grown, arena_size)
                own_food += 1
            else:
                assert isinstance(eater, EnemyBlob)
                enemy_by_key[(player_id, blob_id)] = _with_grown_radius(
                    eater,
                    grown,
                    arena_size,
                )
            eaten_food_ids.add(food.food_id)
        return list(own_by_id.values()), tuple(enemy_by_key.values()), own_food

    # ------------------------------------------------------------------
    # One continuous endpoint value
    # ------------------------------------------------------------------

    def _state_value(
        self,
        node: SearchNode,
        foods,
        viruses,
        arena_size: float,
        *,
        cheap: bool = False,
    ) -> float:
        self._state_value_calls += 1
        if not node.own_blobs:
            return -1_000_000.0

        total_mass = node.total_mass
        hazard_by_blob: list[tuple[OwnBlob, float, float]] = []
        safe_mass = 0.0
        minimum_margin = math.inf
        for blob in node.own_blobs:
            hazard, margin = self._blob_hazard(blob, node.enemies, arena_size)
            hazard_by_blob.append((blob, hazard, margin))
            safe_mass += blob.mass * (1.0 - hazard)
            minimum_margin = min(minimum_margin, margin)

        lost_fraction = max(0.0, 1.0 - safe_mass / max(total_mass, EPSILON))
        coverage = sum(blob.radius for blob in node.own_blobs) / max(math.sqrt(total_mass), EPSILON)
        collection_multiplier = 1.0 + 0.18 * max(0.0, min(3.0, coverage - 1.0))

        food_future = self._food_future_mass(node, foods, collection_multiplier, cheap)
        virus_future = self._virus_future_mass(node, viruses, arena_size, cheap)
        prey_future = self._prey_future_mass(node, cheap)

        # All positive opportunities are expressed as expected retained mass.
        # Log utility prevents a single large visible prey estimate from
        # overwhelming certain current mass while preserving mass ordering.
        effective_mass = max(
            0.0,
            safe_mass
            + food_future
            + virus_future
            + prey_future,
        )
        value = 100.0 * math.log1p(effective_mass)

        # The log term alone is too tolerant of sacrificing a large fraction
        # of current mass for speculative future mass.  This term is smooth in
        # the safety margin and has no rank/round threshold.
        value -= 135.0 * lost_fraction

        # Fragment coverage improves simultaneous collection and pincer reach,
        # but only while those fragments remain controllable.
        cooldown_mass = sum(
            blob.mass * min(1.0, blob.merge_cooldown / 18.0)
            for blob in node.own_blobs
        ) / max(total_mass, EPSILON)
        value -= 5.0 * cooldown_mass * lost_fraction

        # Tiny deterministic tie-breaks prefer a larger safety margin and do
        # not create a behavioural phase.
        if math.isfinite(minimum_margin):
            value += 0.08 * max(-10.0, min(10.0, minimum_margin))
        return value

    def _blob_hazard(
        self,
        blob: OwnBlob,
        enemies: tuple[EnemyBlob, ...],
        arena_size: float,
    ) -> tuple[float, float]:
        survival = 1.0
        minimum_margin = math.inf
        # A newly split fragment must remain viable until it can merge again.
        # The horizon varies continuously with remaining cooldown; it is not a
        # separate split/virus phase.
        horizon = 2.0 + 4.5 * min(1.0, blob.merge_cooldown / 18.0)
        own_speed = _speed(blob.radius)
        for enemy in enemies:
            if not can_eat_player_blob(enemy.radius, blob.radius):
                continue
            distance = math.dist(blob.pos, enemy.pos)
            uncertainty = 0.35 + 0.32 * enemy.stale_rounds
            enemy_speed = _speed(enemy.radius)
            closing = max(0.0, enemy_speed - 0.82 * own_speed)
            normal_reach = (
                enemy.radius
                + enemy_speed
                + closing * (horizon - 1.0)
                + uncertainty
            )
            reach = normal_reach
            if _can_split_eat(enemy.radius, blob.radius):
                child_speed = _speed(enemy.radius / SQRT2)
                split_closing = max(0.0, child_speed - 0.82 * own_speed)
                reach = max(
                    reach,
                    _split_attack_reach(enemy.radius)
                    + split_closing * (horizon - 1.0)
                    + uncertainty,
                )
            margin = distance - reach

            # A wall matters only when it blocks the direction that increases
            # this predator distance.  Longer cooldown makes a blocked retreat
            # matter for more than the current frame.
            away = normalise((blob.x - enemy.x, blob.y - enemy.y))
            intended = own_speed
            moved_x = _clamp(blob.x + away[0] * intended, blob.radius, arena_size - blob.radius)
            moved_y = _clamp(blob.y + away[1] * intended, blob.radius, arena_size - blob.radius)
            useful = math.dist(blob.pos, (moved_x, moved_y))
            margin -= max(0.0, intended - useful) * (1.4 + 0.2 * (horizon - 1.0))

            probability = _sigmoid_negative_margin(margin, 1.2)
            survival *= 1.0 - probability
            minimum_margin = min(minimum_margin, margin)
        return 1.0 - survival, minimum_margin

    def _minimum_safety_margin(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        arena_size: float,
    ) -> float:
        result = math.inf
        for blob in own_blobs:
            _, margin = self._blob_hazard(blob, enemies, arena_size)
            result = min(result, margin)
        return result

    def _food_future_mass(self, node: SearchNode, foods, multiplier: float, cheap: bool) -> float:
        values: list[float] = []
        direction = normalise(node.last_direction)
        for food in foods:
            if food.food_id in node.eaten_food_ids:
                continue
            best_probability = 0.0
            for blob in node.own_blobs:
                dx = food.pos[0] - blob.x
                dy = food.pos[1] - blob.y
                distance = math.hypot(dx, dy)
                gap = max(0.0, distance - blob.radius)
                speed = max(_speed(blob.radius), 0.1)
                turns = gap / speed
                radial = math.exp(-turns / 7.0)

                # Continuing along a line that actually intersects pellets is
                # more useful than moving toward the average of several radial
                # potentials and missing all of them.  This is the geometric
                # capture probability under the node's current control, not a
                # food-specific mode.
                forward = dx * direction[0] + dy * direction[1]
                lateral = abs(dx * direction[1] - dy * direction[0])
                if forward >= -blob.radius:
                    corridor_width = blob.radius + FOOD_RADIUS + 0.18
                    lateral_probability = 1.0 / (
                        1.0
                        + math.exp(
                            max(
                                -40.0,
                                min(40.0, (lateral - corridor_width) / 0.22),
                            )
                        )
                    )
                    forward_gap = max(0.0, forward - blob.radius)
                    corridor = (
                        lateral_probability
                        * math.exp(-forward_gap / (12.0 * speed))
                    )
                    radial += self.corridor_blend * max(0.0, corridor - radial)
                best_probability = max(best_probability, radial)
            values.append(FOOD_RADIUS * FOOD_RADIUS * best_probability)
        values.sort(reverse=True)
        limit = self.cheap_food_value_limit if cheap else self.food_value_limit
        # Fragment coverage may make more pellets reachable, but it cannot
        # make one pellet worth more than its transferable engine mass.
        return sum(values[:limit])

    def _virus_retained_net(
        self,
        node: SearchNode,
        origin: OwnBlob,
        virus,
        arena_size: float,
        *,
        coarse: bool,
    ) -> float:
        piece_count = max(1, MAX_BLOB_COUNT - len(node.own_blobs) + 1)
        post_mass = origin.mass + virus.radius * virus.radius
        piece_radius = math.sqrt(post_mass / piece_count)
        direction = normalise((virus.pos[0] - origin.x, virus.pos[1] - origin.y))
        contact_x = _clamp(
            virus.pos[0] - direction[0] * origin.radius,
            piece_radius,
            arena_size - piece_radius,
        )
        contact_y = _clamp(
            virus.pos[1] - direction[1] * origin.radius,
            piece_radius,
            arena_size - piece_radius,
        )
        contact_origin = replace(origin, x=contact_x, y=contact_y)

        before_hazard, _ = self._blob_hazard(origin, node.enemies, arena_size)
        retained_before = origin.mass * (1.0 - before_hazard)
        if coarse or piece_count == 1:
            probe = OwnBlob(
                blob_id=origin.blob_id,
                x=contact_x,
                y=contact_y,
                radius=piece_radius,
                merge_cooldown=18 if piece_count > 1 else origin.merge_cooldown,
            )
            hazard, _ = self._blob_hazard(probe, node.enemies, arena_size)
            retained_after = post_mass * (1.0 - hazard)
        else:
            fragments = self._virus_replacement_fragments(
                origin=contact_origin,
                piece_radius=piece_radius,
                piece_count=piece_count,
                arena_size=arena_size,
                occupied_ids={blob.blob_id for blob in node.own_blobs},
            )
            retained_after = 0.0
            for fragment in fragments:
                hazard, _ = self._blob_hazard(fragment, node.enemies, arena_size)
                retained_after += fragment.mass * (1.0 - hazard)
        return retained_after - retained_before

    def _virus_future_mass(self, node: SearchNode, viruses, arena_size: float, cheap: bool) -> float:
        values: list[tuple[float, float]] = []
        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            best = -math.inf
            best_turns = math.inf
            for origin in node.own_blobs:
                if not self._can_still_consume_virus_at_contact(origin, virus):
                    continue
                gap = max(0.0, math.dist(origin.pos, virus.pos) - origin.radius)
                turns = gap / max(_speed(origin.radius), 0.1)
                net = self._virus_retained_net(
                    node,
                    origin,
                    virus,
                    arena_size,
                    coarse=cheap,
                )
                potential = net * math.exp(-turns / 8.0)
                if potential > best:
                    best = potential
                    best_turns = turns
            if best > -math.inf:
                values.append((best, best_turns))
        if not values:
            return 0.0
        values.sort(key=lambda item: item[0], reverse=True)
        positive = sum(value for value, _ in values[: 1 if cheap else 2] if value > 0.0)
        # Only an unsafe virus close enough to be touched accidentally should
        # repel the trajectory.  A distant bad virus should not cancel a safe
        # resource elsewhere in view.
        negative = sum(
            value * math.exp(-turns / 2.0)
            for value, turns in values
            if value < 0.0
        )
        return positive + negative

    def _prey_future_mass(self, node: SearchNode, cheap: bool) -> float:
        values: list[float] = []
        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            best_turns = math.inf
            capable_origins: list[OwnBlob] = []
            for own in node.own_blobs:
                distance = math.dist(own.pos, enemy.pos)
                if can_eat_player_blob(own.radius, enemy.radius):
                    closing_speed = max(0.12, _speed(own.radius) - 0.65 * _speed(enemy.radius))
                    turns = max(0.0, distance - own.radius) / closing_speed
                    best_turns = min(best_turns, turns)
                    capable_origins.append(own)
                if (
                    len(node.own_blobs) < MAX_BLOB_COUNT
                    and _can_split_eat(own.radius, enemy.radius)
                ):
                    split_gap = max(0.0, distance - _split_attack_reach(own.radius))
                    turns = 0.45 + split_gap / max(_speed(own.radius), 0.1)
                    best_turns = min(best_turns, turns)
                    capable_origins.append(own)
            if not capable_origins or not math.isfinite(best_turns):
                continue

            enclosure = self._enclosure(node.own_blobs, enemy)
            rival = 0.7 + 0.6 * self._rival_values.get(enemy.player_id, 0.25)
            capture_probability = min(
                1.0,
                rival
                * (1.0 + 0.8 * enclosure)
                * math.exp(-best_turns / 5.0),
            )
            values.append(enemy.mass * capture_probability)
        values.sort(reverse=True)
        return sum(values[: 1 if cheap else 3]) * 0.72

    def _enclosure(self, own_blobs: tuple[OwnBlob, ...], enemy: EnemyBlob) -> float:
        angles = sorted(
            math.atan2(blob.y - enemy.y, blob.x - enemy.x) % TAU
            for blob in own_blobs
            if can_eat_player_blob(blob.radius, enemy.radius)
        )
        if len(angles) < 2:
            return 0.0
        gaps = [
            (angles[(index + 1) % len(angles)] - angles[index]) % TAU
            for index in range(len(angles))
        ]
        return max(0.0, min(1.0, 1.0 - max(gaps) / TAU))

    def _terminal_score(self, node: SearchNode) -> float:
        return node.score

    # ------------------------------------------------------------------
    # Better public-position tracking; opponent moves are not normally exposed
    # ------------------------------------------------------------------

    def _update_enemy_memory(
        self,
        context: StrategyContext,
        own_blobs: tuple[OwnBlob, ...],
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        state = context.game.state
        round_number = int(state.round)
        visible_keys: set[tuple[int, int]] = set()
        previous = self.enemy_tracks
        advanced: dict[tuple[int, int], EnemyTrack] = {}

        for key, track in previous.items():
            explicit = self.last_moves.get(track.player_id)
            direction = explicit[0] if explicit is not None else track.direction
            if direction == (0.0, 0.0):
                vulnerable = [
                    own
                    for own in own_blobs
                    if can_eat_player_blob(track.radius, own.radius)
                ]
                if vulnerable:
                    target = min(vulnerable, key=lambda own: squared_distance((track.x, track.y), own.pos))
                    direction = normalise((target.x - track.x, target.y - track.y))
            radius = _decayed_radius(track.radius)
            speed = _speed(track.radius)
            advanced[key] = replace(
                track,
                x=_clamp(track.x + direction[0] * speed, radius, arena_size - radius),
                y=_clamp(track.y + direction[1] * speed, radius, arena_size - radius),
                radius=radius,
                direction=direction,
                merge_cooldown=max(0, track.merge_cooldown - 1),
            )

        for blob in state.visible_blobs:
            key = (blob.player_id, blob.blob_id)
            visible_keys.add(key)
            explicit = self.last_moves.get(blob.player_id)
            if explicit is not None:
                direction = explicit[0]
            else:
                old = previous.get(key)
                if old is None:
                    direction = (0.0, 0.0)
                else:
                    displacement = (blob.pos[0] - old.x, blob.pos[1] - old.y)
                    measured = normalise(displacement)
                    # Blend finite-difference motion with the previous estimate
                    # to reduce separation/merge jitter without assuming access
                    # to the opponent's private MovePlayer event.
                    direction = normalise(
                        (
                            measured[0] * 0.72 + old.direction[0] * 0.28,
                            measured[1] * 0.72 + old.direction[1] * 0.28,
                        )
                    )
            advanced[key] = EnemyTrack(
                player_id=blob.player_id,
                blob_id=blob.blob_id,
                x=blob.pos[0],
                y=blob.pos[1],
                radius=blob.radius,
                direction=direction,
                last_seen_round=round_number,
                merge_cooldown=blob.merge_cooldown,
            )

        view_center = tuple(state.view_center)
        half_view = float(state.vision_size) / 2.0
        kept: dict[tuple[int, int], EnemyTrack] = {}
        enemies: list[EnemyBlob] = []
        for key, track in advanced.items():
            stale_rounds = round_number - track.last_seen_round
            if stale_rounds > 8:
                continue
            should_be_visible = (
                abs(track.x - view_center[0]) <= half_view + track.radius
                and abs(track.y - view_center[1]) <= half_view + track.radius
            )
            if key not in visible_keys and should_be_visible:
                continue
            kept[key] = track
            if stale_rounds and not any(
                can_eat_player_blob(track.radius, own.radius)
                for own in own_blobs
            ):
                continue
            enemies.append(
                EnemyBlob(
                    player_id=track.player_id,
                    blob_id=track.blob_id,
                    x=track.x,
                    y=track.y,
                    radius=track.radius,
                    direction=track.direction,
                    stale_rounds=stale_rounds,
                    merge_cooldown=track.merge_cooldown,
                )
            )
        self.enemy_tracks = kept
        return tuple(enemies)

    def _move_enemies(
        self,
        enemies: tuple[EnemyBlob, ...],
        own_blobs: list[OwnBlob],
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        """Expected hostile motion; the value function separately prices reach.

        Using a 100% worst-case chase in the transition makes every candidate
        share the same artificial collision and reduces action discrimination.
        This blend preserves measured momentum while remaining hostile when an
        enemy can consume a fragment.
        """

        moved: list[EnemyBlob] = []
        by_player: dict[int, list[EnemyBlob]] = {}
        for enemy in enemies:
            by_player.setdefault(enemy.player_id, []).append(enemy)
        for group in by_player.values():
            total_mass = sum(enemy.mass for enemy in group)
            center = (
                sum(enemy.x * enemy.mass for enemy in group) / total_mass,
                sum(enemy.y * enemy.mass for enemy in group) / total_mass,
            )
            observed = normalise(
                (
                    sum(enemy.direction[0] * enemy.mass for enemy in group),
                    sum(enemy.direction[1] * enemy.mass for enemy in group),
                )
            )
            predators = [
                own
                for own in own_blobs
                if any(can_eat_player_blob(own.radius, enemy.radius) for enemy in group)
            ]
            prey = [
                own
                for own in own_blobs
                if any(can_eat_player_blob(enemy.radius, own.radius) for enemy in group)
            ]
            if prey:
                target = min(prey, key=lambda own: squared_distance(center, own.pos))
                adversarial = normalise((target.x - center[0], target.y - center[1]))
                observed_weight = 0.45 if observed != (0.0, 0.0) else 0.0
                direction = normalise(
                    (
                        adversarial[0] * (1.0 - observed_weight) + observed[0] * observed_weight,
                        adversarial[1] * (1.0 - observed_weight) + observed[1] * observed_weight,
                    )
                )
            elif predators:
                hunter = min(predators, key=lambda own: squared_distance(center, own.pos))
                flee = normalise((center[0] - hunter.x, center[1] - hunter.y))
                observed_weight = 0.55 if observed != (0.0, 0.0) else 0.0
                direction = normalise(
                    (
                        flee[0] * (1.0 - observed_weight) + observed[0] * observed_weight,
                        flee[1] * (1.0 - observed_weight) + observed[1] * observed_weight,
                    )
                )
            else:
                direction = observed
            for enemy in group:
                radius = _decayed_radius(enemy.radius)
                speed = _speed(enemy.radius)
                moved.append(
                    replace(
                        enemy,
                        x=_clamp(enemy.x + direction[0] * speed, radius, arena_size - radius),
                        y=_clamp(enemy.y + direction[1] * speed, radius, arena_size - radius),
                        radius=radius,
                        direction=direction,
                        merge_cooldown=max(0, enemy.merge_cooldown - 1),
                    )
                )
        return tuple(moved)
