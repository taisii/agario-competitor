# agario-competitor

Starter repo for a competition bot.

## Competition information

This repository targets the **SYNCS 2026 Agar.io bot competition**. The
official [SYNCS Game Hub](https://syncs.org.au/competition2026/game) is the
source of truth for team membership, submissions, submission history, and
competition status. The portal was last verified as reachable on 12 July 2026.

- The current game has eight players.
- The official submission is a single Python file, generated as
  `dist/my_bot.py` by this repository.
- Competition announcements and engine update notices are posted in the
  [competition Discord](https://discord.gg/24We3YWM7e).
- The advertised 40 PEP hours require a SYNCS account registered with a USyd
  email, `Needs pep hours` enabled, membership in a competition team, and at
  least five submissions by that team. Confirm eligibility and submission
  counts in the Game Hub.

The latest official match replays can also be turned into local opponent
clones. See [Official replay opponent clones](docs/replay-opponents.md) for the
per-team strategies, fidelity gates, and parallel simulation commands.

## First run
1. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

2. Run these from this folder
```bash
uv sync --frozen
uv run interactive 7:bots/my_bot.py
```

This template installs the published `agario-kit` package from PyPI. The local
interactive launcher expects `count:path` specs whose counts sum to `n - 1`.
For the current 8-player game, that means the counts must sum to `7`.
The lockfile currently resolves `agario-kit==2026.1.14`, the engine version
used by the official match reports. A virus is consumed only when its center
lies inside a blob, and food/player growth is clamped back into the arena in
the same round. Food, virus, and opponent-blob IDs are rebuilt for each public
query, so strategies must use geometry and internal IDs for tracking across
turns rather than treating public indices as stable identities.

To play manually against example bots instead, run:

```bash
uv run interactive 2:bots/my_bot.py 5:bots/entries/survival_greedy.py
```

To watch a non-interactive simulation, run:

```bash
uv run simulation 8:bots/my_bot.py
```

To run benchmark matches in parallel and aggregate results, run:

```bash
uv run python scripts/benchmark_simulations.py --trials 8 --jobs 1
```

The benchmark runner writes one isolated simulation workspace per match under
`.agario/benchmarks/`, plus `matches.csv`, `results.json`, and `run_config.json`
at the benchmark root. The default smoke benchmark runs four `food_greedy`
bots and four `survival_greedy` bots.

Replay-dominance profiling is opt-in. To sample one complete turn in every 100
without charging every turn for timers, add
`BOT_REPLAY_PROFILE_EVERY_N=100` to a benchmark variant. The structured sample
is stored in `decision_diagnostics.replay_profile` and separates additive phase
time from nested inclusive operation time. Cache and candidate counters include
conservation data (`lookups = hits + misses` and
`raw = unique + zero_drops + duplicate_drops`). Aggregate and validate it with:

```bash
uv run python scripts/analyze_replay_profile.py \
  .agario/benchmarks/<run>/<variant>/run_*/match/submission0/bot_metrics.jsonl
```

The analyzer fails closed if any input contains no profile samples, uses an
unsupported schema, or violates a per-sample timing, cache, or candidate
invariant. This prevents an uninstrumented or mismatched log from being used as
refactoring evidence accidentally.

`BOT_REPLAY_AUDIT_EVERY_N` performs additional exact transitions and is intended
only for an offline ranking audit. Its caches, counters, fatal candidates, and
elapsed time are isolated from the normal search profile.

To test the default replay-dominance receding-horizon strategy against seven
randomly selected official-replay clone strategies, run:

```bash
uv run simulation 1:bots/entries/replay_dominance.py 7:bots/entries/random_replay_opponent.py --headless
```

The random replay pool contains all 42 observed enemy `team_id` values, not
only the curated dedicated-entry panel. The same random seed, benchmark trial,
and player slot reproduce the same selection.

To screen it against every active replay opponent, with individual matches sharing one
global parallel-job limit, run:

```bash
uv run python scripts/benchmark_replay_opponents.py --trials 2 --jobs 1
```

To compare `semantic_potential` and `replay_dominance` on the exact same
observations from a local `game.json`, without adding any oracle work to the
submitted bot, run:

```bash
uv run python scripts/compare_strategy_decisions.py \
  .agario/simulation/output/game.json \
  --player 0 \
  --output .agario/comparisons/semantic-vs-replay.json
```

The report separates direction/split agreement from `proxy_regret`, which is
the advantage of replay-dominance's action when both actions are evaluated by
the same offline replay proxy. A difference is an adoption candidate only when
the actions differ materially and the replay action also clears the configured
proxy-regret threshold. Per-strategy decision timings are reported alongside
quality so a closer imitation cannot hide a production cost regression.
Background and food-only disagreements are diagnostic only: adoption is driven
by threat, enemy-capture, and virus contexts, then accepted or rejected by
paired mean final mass, which is the leaderboard objective.

To attribute a strong replay player's gross mass gains and measure whether
`semantic_potential` matched the action that preceded each gain, run:

```bash
uv run python scripts/analyze_mass_gain_sources.py \
  .agario/simulation/output/game.json \
  --player 1 \
  --output .agario/comparisons/winner-mass-gains.json
```

The attribution follows engine event order and reports enemy, virus, and food
mass separately. `semantic_potential` evaluates those sources in the same mass
unit: food contributes `FOOD_RADIUS²`, a virus contributes `radius²`, and an
enemy fragment contributes its captured `radius²`. Directional potential is a
four-turn fan of reachable mass discounted by contact time. The food part is
bounded to the nearest eight pellets because two concrete food candidates are
already scored separately; enemy and virus sources remain uncapped. Split
candidates and virus-target candidates receive a bounded visible-state
one-turn transition in engine order (decay, virus, food, player eating), so
topology changes cannot be scored using the safer pre-command blobs. When
consecutive observations reveal the selected prey's motion, its next center is
projected; other enemy centers remain fixed because their next commands are
not observable.

The submission runtime gate profiles the same 1,007 reconstructed observations
before a build. Bounded food sources, cached movement speeds, and lazy wall
projection reduced profiled strategy time from 249 ms to 186 ms per replay.
Peak mass remains a diagnostic, but acceptance is gated on paired final mass.
In the final 48-match screen against seven `replay_dominance` opponents, mean
cumulative query/response time was 0.797 s, p95 was 1.097 s, and the maximum
was 1.194 s against the engine's authoritative 8 s cumulative limit. Over the
same two independent seed sets, mean final mass improved from 36.19 to 52.29.

The standalone builder also replaces the starter kit's one-member query-union
wrapper and 1 KiB bytearray receive loop with direct `QueryMovePlayer`
validation and a length-preserving string reader. A full-process profile moved
JSON validation from 64.3 ms to 59.7 ms and framed-message assembly from 20.3
ms to 8.6 ms over 1,400 turns. Four paired simulations all improved cumulative
response time (321 ms mean to 314 ms) with byte-for-byte identical match
results. The remaining gap is principally pipe wake-up and process scheduling,
which is outside the submitted Python process's controllable CPU work.

The matrix uses the no-recording fast runner by default. Add `--official` for a
final process-layout and recording check after narrowing the candidates; it is
substantially slower and is not intended for the exhaustive screen. For a
deterministic fast-runner divergence, pass `--record` directly to
`scripts/run_fast_simulation.py` to preserve `game.json` without changing the
seeded engine path.

The active replay panel contains only empirically strong, tactically active
opponents. Re-evaluate it after importing new official replays:

```bash
uv run python scripts/rank_replay_opponents.py \
  --replay-dir .agario/replays/official/submission-4 \
  --replay-dir .agario/replays/official/submission-4-extra
```

## Building the official submission

The 2026 submission portal accepts one `.py` file. The default candidate is
`semantic_lookahead`; keep the modular strategy sources for development and
generate the upload artifact with:

```bash
uv run python scripts/build_submission.py
uv run python -m py_compile dist/my_bot.py
uv run simulation 1:dist/my_bot.py 7:bots/entries/random_opponent.py --headless
```

Build a different supported submission strategy with `--strategy`. The previous
`replay_dominance` candidate remains available for an immediate rollback:

```bash
uv run python scripts/build_submission.py --strategy replay_dominance
```

Upload `dist/my_bot.py` at `https://syncs.org.au/competition2026/game`.
`dist/` is ignored because the file is generated; the builder prints its
SHA-256 hash so the uploaded revision can be recorded separately.

For the advertised 40 PEP hours, the organisers require a SYNCS account using
a USyd email with `Needs pep hours` enabled, membership in a competition team,
and at least five submissions by that team. The portal submission history is
the authoritative count; keep the hash and successful portal status for each
meaningful iteration.

`random_opponent` shuffles a six-policy pool (`semantic_lookahead`,
`semantic_potential`, `replay_dominance`, `threat_aware_receding_horizon`,
`event_driven_static_search`, and `static_retained_growth`) per paired trial.
Opponent slots 1–7 cover every policy, with one duplicated; override the pool
with `BOT_RANDOM_STRATEGIES=a,b,c`.

Example strategy screen:

```bash
uv run python scripts/benchmark_simulations.py \
  --trials 4 \
  --jobs 2 \
  --variants \
    receding_horizon:BOT_STRATEGY=threat_aware_receding_horizon \
    potential_field:BOT_STRATEGY=potential_field_hunter \
    virus_hunter:BOT_STRATEGY=virus_hunter \
  --submission 1:bots/my_bot.py 7:bots/entries/random_opponent.py \
  --tracked-slots 0
```

Strategy implementations are grouped by their fundamental algorithm:

- `bots/strategies/greedy.py`: immediate food, prey, and escape target rules.
- `bots/strategies/potential_field.py`: weighted potential-field movement.
- `bots/strategies/receding_horizon.py`: predictive receding-horizon strategies.
- `bots/strategies/virus_farming.py`: safe virus pursuit and potential-field farming.
- `bots/strategies/replay_opponents.py`: dedicated catalog for replay-derived
  benchmark opponents. These are intentionally separate from candidate bot
  strategies.
- `bots/strategies/replay_imitation.py`: replay-fitted opponent behavior; it
  needs a validated profile before it can be registered in the opponent catalog.
- `bots/strategies/local_tactical_search.py`: shallow two-step local planning
  with rational nearby-opponent responses and reversal-only steering cost.

Available local strategy entry points:

- `bots/entries/food_greedy.py`: nearest-food baseline.
- `bots/entries/survival_greedy.py`: flee nearby predators, then chase prey/food.
- `bots/entries/virus_hunter.py`: prioritise consumable viruses, growing on prey/food until one is safely reachable.
- `bots/entries/potential_field_virus_farmer.py`: safe virus farming with potential-field prey/food growth.
- `bots/entries/potential_field_hunter.py`: potential-field hunter.
- `bots/entries/threat_aware_receding_horizon.py`: robust adversarial prediction with engine-matched split physics.
- `bots/entries/threat_aware_receding_horizon_reference.py`: deliberately expensive reference profile of the same strategy.
- `bots/entries/replay_dominance.py`: default unified search policy for survival, virus growth, wall mobility, and rival elimination.
- `bots/entries/expected_final_mass.py`: expected-final-mass search that evaluates proposals from strong official-replay policies instead of protecting ordinal rank.
- `bots/entries/local_tactical_search.py`: submission-safe local tactical search; validates the DP-ranked roots with exact engine physics.
- `bots/entries/local_tactical_search_reference.py`: correctness-first wide local planner used as an optimisation oracle; not submission-safe.
- `bots/entries/random_opponent.py`: picks one stable strategy at process startup.

Example mixed match:

```bash
uv run simulation 2:bots/entries/food_greedy.py 2:bots/entries/survival_greedy.py 2:bots/entries/potential_field_hunter.py 2:bots/entries/virus_hunter.py
```

The legacy beam/RL family, `unified_deterministic`, and the frozen
`virus_farming_receding_horizon` strategy were removed from the public catalog.
Saved competition screens showed cumulative timeouts or consistent domination
by `replay_dominance`, and their private simulators disagreed with the current
engine. Historical strategy names now fail explicitly instead of silently
selecting a different policy.

## Writing a bot

- Add or change decision policy code under `bots/strategies/`.
- Register public policies in `bots/strategies/registry.py`; the runtime loads
  implementations lazily from that catalog.
- Register official replay clones in `bots/strategies/replay_opponents.py`.
  They are benchmark opponents, not selectable candidate strategies.
- Keep engine transition rules out of policies. Shared movement, collision,
  visibility, food, and virus rules belong under `bots/simulation/`.
- Keep process I/O and telemetry in `bots/runtime.py`. Entry files should only
  select a strategy and invoke the shared runtime.

Run the complete test suite and build the single-file submission before a
benchmark or portal upload:

```bash
uv run pytest -q
uv run python scripts/build_submission.py
uv run python -m py_compile dist/my_bot.py
```

## Updating during the competition

We may make changes to the game engine during the event. Normal development
uses the committed lockfile. When a new platform version is announced, update
the engine explicitly, run the engine-contract and regression tests, and commit
the resulting lockfile:

```bash
uv lock --upgrade-package agario-kit
uv sync --frozen
uv run pytest -q
```

We will send a message on [Discord](https://discord.gg/24We3YWM7e) if this happens.
