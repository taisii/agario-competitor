from __future__ import annotations

"""Fast unified beam with continuous capability and contested-virus value."""

import math

from lib.config.arena import MAX_BLOB_COUNT
from lib.config.player import EAT_SIZE_RATIO, SPLIT_MIN_MASS
from strategies.features import can_eat_player_blob, normalise
from strategies.receding_horizon import (
    EPSILON,
    SearchNode,
    _can_consume_virus,
    _can_split_eat,
    _clamp,
    _speed,
    _split_attack_reach,
)
from strategies.unified_beam import UnifiedBeamStrategy, _sigmoid_negative_margin


def _softplus(value: float) -> float:
    if value > 40.0:
        return value
    if value < -40.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(_clamp(-value, -40.0, 40.0)))


class UnifiedCoreStrategy(UnifiedBeamStrategy):
    """One endpoint utility; no growth, farming, hunting, or endgame modes."""

    name = "unified_core"

    def __init__(self) -> None:
        super().__init__()
        self._arena_size_for_option = 60.0
        self.capability_option_mass = 0.55
        self._coalition_profile_cache: dict[tuple[object, ...], tuple[tuple[float, ...], ...]] = {}
        self._view_center = (30.0, 30.0)
        self._view_half = 30.0

    def choose(self, context):
        # Enemy tuples are immutable within a decision.  Clearing here makes
        # id-based profile caching safe across turns while avoiding repeated
        # grouping in every endpoint and screening evaluation.
        self._coalition_profile_cache.clear()
        state = context.game.state
        self._view_center = tuple(state.view_center)
        self._view_half = float(state.vision_size) / 2.0
        return super().choose(context)


    def _coalition_profiles(self, enemies) -> tuple[tuple[float, ...], ...]:
        """Continuous near-merge threat profiles for each enemy player.

        The official engine exposes fragments separately, but a coherent group
        can become one predator when its cooldown expires.  The profile blends
        from the largest current fragment to the group's total mass according
        to remaining cooldown and geometric cohesion; it is not a mode switch.
        """

        key = tuple(enemies)
        cached = self._coalition_profile_cache.get(key)
        if cached is not None:
            return cached
        by_player: dict[int, list[object]] = {}
        for enemy in enemies:
            if enemy.stale_rounds > 2:
                continue
            by_player.setdefault(enemy.player_id, []).append(enemy)
        profiles: list[tuple[float, ...]] = []
        for player_id, group in by_player.items():
            if len(group) < 2:
                continue
            total_mass = sum(enemy.mass for enemy in group)
            largest_mass = max(enemy.mass for enemy in group)
            if total_mass <= largest_mass + EPSILON:
                continue
            cx = sum(enemy.x * enemy.mass for enemy in group) / total_mass
            cy = sum(enemy.y * enemy.mass for enemy in group) / total_mass
            spread = math.sqrt(
                sum(
                    enemy.mass * ((enemy.x - cx) ** 2 + (enemy.y - cy) ** 2)
                    for enemy in group
                )
                / total_mass
            )
            cooldown = sum(enemy.mass * enemy.merge_cooldown for enemy in group) / total_mass
            equivalent_radius = math.sqrt(total_mass)

            # A partially visible equal-size grid is evidence for sibling
            # fragments just outside the rectangular view.  Estimate only a
            # bounded multiple of the observed count, then fade it smoothly
            # with edge clearance and radius variation.
            mean_radius = sum(enemy.radius for enemy in group) / len(group)
            radius_variance = sum((enemy.radius - mean_radius) ** 2 for enemy in group) / len(group)
            uniformity = math.exp(-6.0 * math.sqrt(radius_variance) / max(mean_radius, 0.1))
            edge_clearance = min(
                self._view_half - abs(cx - self._view_center[0]),
                self._view_half - abs(cy - self._view_center[1]),
            )
            edge_share = 1.0 / (
                1.0 + math.exp(_clamp((edge_clearance - equivalent_radius - 2.0) / 1.2, -40.0, 40.0))
            )
            count = len(group)
            fragment_signal = max(
                1.0 - math.exp(-cooldown / 5.0),
                1.0 - math.exp(-(count - 1) / 3.0),
            )
            missing_count = min(
                MAX_BLOB_COUNT - count,
                count * (1.0 + 1.5 * fragment_signal) * edge_share * uniformity,
            )
            typical_mass = total_mass / count
            estimated_total_mass = total_mass + max(0.0, missing_count) * typical_mass

            # One turn of normal motion plus merge attraction closes roughly
            # one radius-scale gap.  Cooldown and excess spread therefore act
            # as additive delays in the same exponential survival horizon.
            cooldown_share = math.exp(-cooldown / 6.5)
            excess_spread = max(0.0, spread - equivalent_radius)
            cohesion_share = math.exp(-excess_spread / max(1.0, equivalent_radius + 1.0))
            readiness = cooldown_share * cohesion_share
            effective_mass = largest_mass + readiness * (estimated_total_mass - largest_mass)
            profiles.append(
                (
                    float(player_id),
                    cx,
                    cy,
                    math.sqrt(effective_mass),
                    readiness,
                    float(max(enemy.stale_rounds for enemy in group)),
                )
            )
        result = tuple(profiles)
        self._coalition_profile_cache[key] = result
        return result

    def _blob_hazard(self, blob, enemies, arena_size: float):
        base_hazard, minimum_margin = super()._blob_hazard(blob, enemies, arena_size)
        survival = 1.0 - base_hazard
        own_speed = _speed(blob.radius)
        horizon = 2.0 + 4.5 * min(1.0, blob.merge_cooldown / 18.0)
        for _, cx, cy, radius, readiness, stale_rounds in self._coalition_profiles(enemies):
            if readiness <= 1e-4 or not can_eat_player_blob(radius, blob.radius):
                continue
            distance = max(0.25, math.hypot(blob.x - cx, blob.y - cy))
            enemy_speed = _speed(radius)
            uncertainty = 0.45 + 0.36 * stale_rounds
            closing = max(0.0, enemy_speed - 0.82 * own_speed)
            reach = radius + enemy_speed + closing * (horizon - 1.0) + uncertainty
            if _can_split_eat(radius, blob.radius):
                child_speed = _speed(radius / math.sqrt(2.0))
                split_closing = max(0.0, child_speed - 0.82 * own_speed)
                reach = max(
                    reach,
                    _split_attack_reach(radius)
                    + split_closing * (horizon - 1.0)
                    + uncertainty,
                )
            margin = distance - reach

            away = normalise((blob.x - cx, blob.y - cy))
            nx = _clamp(blob.x + away[0] * own_speed, blob.radius, arena_size - blob.radius)
            ny = _clamp(blob.y + away[1] * own_speed, blob.radius, arena_size - blob.radius)
            useful = math.hypot(nx - blob.x, ny - blob.y)
            margin -= max(0.0, own_speed - useful) * (1.4 + 0.2 * (horizon - 1.0))

            probability = readiness * _sigmoid_negative_margin(margin, 1.45)
            survival *= 1.0 - min(0.995, probability)
            if readiness >= 0.08:
                minimum_margin = min(minimum_margin, margin)
        return 1.0 - survival, minimum_margin

    @staticmethod
    def _readiness(mass: float, threshold: float) -> float:
        smoothing = 0.16
        shortfall = smoothing * _softplus((threshold - mass) / smoothing)
        return math.exp(-shortfall / 0.55)

    def _screening_context(self, node, foods, viruses, arena_size: float):
        gx, gy = super()._screening_context(node, foods, viruses, arena_size)
        total_mass = max(node.total_mass, EPSILON)
        for own in node.own_blobs:
            own_speed = _speed(own.radius)
            for _, cx, cy, radius, readiness, stale_rounds in self._coalition_profiles(node.enemies):
                if readiness <= 1e-4 or not can_eat_player_blob(radius, own.radius):
                    continue
                distance = max(0.25, math.hypot(own.x - cx, own.y - cy))
                uncertainty = 0.45 + 0.36 * stale_rounds
                reach = radius + _speed(radius) + uncertainty
                if _can_split_eat(radius, own.radius):
                    reach = max(reach, _split_attack_reach(radius) + uncertainty)
                margin = distance - reach
                base_probability = _sigmoid_negative_margin(margin, 1.45)
                derivative = readiness * base_probability * (1.0 - base_probability) / 1.45
                weight = 135.0 * own.mass / total_mass * derivative
                gx += (own.x - cx) / distance * weight
                gy += (own.y - cy) / distance * weight
        return gx, gy

    def _escape_vector(self, node: SearchNode) -> tuple[float, float]:
        base = super()._escape_vector(node)
        x, y = base
        for own in node.own_blobs:
            for _, cx, cy, radius, readiness, stale_rounds in self._coalition_profiles(node.enemies):
                if readiness <= 1e-4 or not can_eat_player_blob(radius, own.radius):
                    continue
                danger_radius = radius
                if _can_split_eat(radius, own.radius):
                    danger_radius = max(danger_radius, _split_attack_reach(radius))
                distance = max(0.25, math.hypot(own.x - cx, own.y - cy))
                if distance > danger_radius + 10.0:
                    continue
                severity = readiness * max(0.2, danger_radius + 10.0 - distance) / distance
                x += (own.x - cx) / distance * severity * own.mass
                y += (own.y - cy) / distance * severity * own.mass
        return normalise((x, y))

    def _prey_future_mass(self, node: SearchNode, cheap: bool) -> float:
        profiles = {int(item[0]): item for item in self._coalition_profiles(node.enemies)}
        values: list[float] = []
        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            best_turns = math.inf
            capable_origins = []
            for own in node.own_blobs:
                distance = math.dist(own.pos, enemy.pos)
                if can_eat_player_blob(own.radius, enemy.radius):
                    closing_speed = max(0.12, _speed(own.radius) - 0.65 * _speed(enemy.radius))
                    turns = max(0.0, distance - own.radius) / closing_speed
                    best_turns = min(best_turns, turns)
                    capable_origins.append(own)
                if len(node.own_blobs) < MAX_BLOB_COUNT and _can_split_eat(own.radius, enemy.radius):
                    split_gap = max(0.0, distance - _split_attack_reach(own.radius))
                    turns = 0.45 + split_gap / max(_speed(own.radius), 0.1)
                    best_turns = min(best_turns, turns)
                    capable_origins.append(own)
            if not capable_origins or not math.isfinite(best_turns):
                continue

            retaliation = 0.0
            profile = profiles.get(enemy.player_id)
            if profile is not None:
                _, cx, cy, radius, readiness, _ = profile
                for own in capable_origins:
                    if not can_eat_player_blob(radius, own.radius):
                        continue
                    distance = math.hypot(own.x - cx, own.y - cy)
                    reach = radius + _speed(radius)
                    if _can_split_eat(radius, own.radius):
                        reach = max(reach, _split_attack_reach(radius))
                    retaliation = max(
                        retaliation,
                        readiness * _sigmoid_negative_margin(distance - reach, 1.8),
                    )

            enclosure = self._enclosure(node.own_blobs, enemy)
            rival = 0.7 + 0.6 * self._rival_values.get(enemy.player_id, 0.25)
            capture_probability = min(
                1.0,
                rival
                * (1.0 + 0.8 * enclosure)
                * math.exp(-best_turns / 5.0)
                * (1.0 - 0.9 * min(1.0, retaliation)),
            )
            values.append(enemy.mass * capture_probability)
        values.sort(reverse=True)
        return sum(values[: 1 if cheap else 3]) * 0.72

    def _capability_mass(self, node: SearchNode) -> float:
        if not node.own_blobs:
            return 0.0
        threshold = 1.5 * 1.5 * EAT_SIZE_RATIO
        # Capability is useful only if the fragment carrying it survives.  This
        # makes an escape split compete normally when a concentrated blob is
        # already inside a predator envelope; no emergency mode is required.
        # Readiness is monotone in mass.  Only the two largest fragments can
        # be non-dominated: the second is retained because a nearby predator
        # can make it safer than the largest.  This avoids repeating the full
        # hazard scan over sixteen nearly equal virus fragments.
        candidates = node.own_blobs
        best_expected_option = 0.0
        for blob in candidates:
            hazard, _ = self._blob_hazard(blob, node.enemies, self._arena_size_for_option)
            best_expected_option = max(
                best_expected_option,
                self._readiness(blob.mass, threshold) * (1.0 - hazard),
            )
        # The coefficient is expected recoverable virus mass, not a phase
        # bonus.  Subclasses may use a different observation/search model.
        return self.capability_option_mass * best_expected_option

    def _screen_split_delta(self, node: SearchNode, direction, foods) -> float:
        """Order split branches by the same concentration option as endpoints."""

        delta = super()._screen_split_delta(node, direction, foods)
        if not math.isfinite(delta):
            return delta
        return delta

    def _state_value(self, node, foods, viruses, arena_size: float, *, cheap: bool = False) -> float:
        self._arena_size_for_option = arena_size
        value = super()._state_value(node, foods, viruses, arena_size, cheap=cheap)
        if not node.own_blobs:
            return value
        option = (
            self._capability_mass(node)
            if any(virus.virus_id not in node.consumed_virus_ids for virus in viruses)
            else 0.0
        )
        mass = max(node.total_mass, EPSILON)
        return value + 100.0 * (math.log1p(mass + option) - math.log1p(mass))

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
                net = self._virus_retained_net(node, origin, virus, arena_size, coarse=cheap)

                enemy_eta = math.inf
                enemy_radius = 0.0
                for enemy in node.enemies:
                    if enemy.stale_rounds > 2 or not _can_consume_virus(enemy.radius, virus.radius):
                        continue
                    enemy_gap = max(0.0, math.dist(enemy.pos, virus.pos) - enemy.radius)
                    eta = enemy_gap / max(_speed(enemy.radius), 0.1)
                    if eta < enemy_eta:
                        enemy_eta = eta
                        enemy_radius = enemy.radius
                if math.isfinite(enemy_eta):
                    tie_bias = 0.12 * (origin.radius - enemy_radius)
                    contest_share = _logistic((enemy_eta - turns + tie_bias) / 1.05)
                else:
                    contest_share = 1.0

                potential = max(0.0, net) * contest_share * math.exp(-turns / 8.0)
                if net < 0.0:
                    potential += net * math.exp(-turns / 2.0)
                if potential > best:
                    best = potential
                    best_turns = turns
            if best > -math.inf:
                values.append((best, best_turns))
        if not values:
            return 0.0
        values.sort(key=lambda item: item[0], reverse=True)
        positive = sum(value for value, _ in values[: 1 if cheap else 2] if value > 0.0)
        negative = sum(value * math.exp(-turns / 2.0) for value, turns in values if value < 0.0)
        return positive + negative
