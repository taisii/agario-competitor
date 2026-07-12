# Bot architecture

The bot is split along behavior boundaries rather than entry-file boundaries.

1. **Runtime** owns the contest process lifecycle: queries, moves, telemetry,
   and per-turn timing. It does not decide where to move.
2. **State extraction** converts the visible `agario-kit` models into immutable
   planning state.
3. **Simulation** implements engine rules shared by every predictive policy.
   Collision ordering and constants are verified against the installed engine.
4. **Search** expands actions under a deadline. It does not embed a particular
   survival, farming, or hunting objective.
5. **Policy** provides candidates and evaluates projected states. A public
   strategy name identifies one policy/profile, not another copy of the runtime
   or physics implementation.
6. **Benchmark orchestration** parallelises independent matches under one
   process limit. Per-turn candidate expansion remains sequential because every
   transition shares a millisecond-scale deadline and mutable anytime-search
   frontier; process startup or thread coordination would cost more than the
   transition being evaluated.

```text
Game query
  -> runtime
  -> visible-state extraction
  -> shared engine-matched transition
  -> policy candidate/evaluation hooks
  -> StrategyDecision
  -> runtime sends MovePlayer
```

## Invariants

- Engine rules have one authoritative implementation. A policy must not carry
  a private variation of food, virus, split, merge, or visibility semantics.
- Public strategy names and legacy aliases are declared in the strategy
  catalog. Importing the catalog must not import every implementation.
- Simulator entry files are declarative adapters. The query loop exists only
  in the runtime.
- Replay opponents may share implementation and differ by immutable profiles,
  but team-specific observable behavior remains a separate public strategy.
- The generated submission is derived from modular sources and must remain a
  self-contained single Python file.

## Adding a strategy

1. Reuse the shared state and simulation layers.
2. Implement only the policy-specific candidate or evaluation behavior.
3. Add a catalog entry with its category and submission capability.
4. Add deterministic policy tests and, for predictive logic, an engine-contract
   case covering every newly used transition.
5. Compare it against the current candidate using immutable benchmark runs.
