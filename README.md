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
The lockfile currently resolves `agario-kit==2026.1.13`. In that release, a
consumable virus is hit only when its center lies inside the blob radius, and
food/player growth is clamped back into the arena in the same round.

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

To screen it against every active replay opponent, with individual matches sharing one
global parallel-job limit, run:

```bash
uv run python scripts/benchmark_replay_opponents.py --trials 2 --jobs 1
```

The matrix uses the no-recording fast runner by default. Add `--official` for a
final process-layout and recording check after narrowing the candidates; it is
substantially slower and is not intended for the exhaustive screen.

The active replay panel contains only empirically strong, tactically active
opponents. Re-evaluate it after importing new official replays:

```bash
uv run python scripts/rank_replay_opponents.py \
  --replay-dir .agario/replays/official/submission-4 \
  --replay-dir .agario/replays/official/submission-4-extra
```

## Building the official submission

The 2026 submission portal accepts one `.py` file. Keep the modular strategy
sources for development and generate the upload artifact with:

```bash
uv run python scripts/build_submission.py
uv run python -m py_compile dist/my_bot.py
uv run simulation 1:dist/my_bot.py 7:bots/entries/random_opponent.py --headless
```

Build a different supported submission strategy with `--strategy`, for example:

```bash
uv run python scripts/build_submission.py --strategy virus_hunter
```

Upload `dist/my_bot.py` at `https://syncs.org.au/competition2026/game`.
`dist/` is ignored because the file is generated; the builder prints its
SHA-256 hash so the uploaded revision can be recorded separately.

For the advertised 40 PEP hours, the organisers require a SYNCS account using
a USyd email with `Needs pep hours` enabled, membership in a competition team,
and at least five submissions by that team. The portal submission history is
the authoritative count; keep the hash and successful portal status for each
meaningful iteration.

`random_opponent` samples from `food_greedy`, `survival_greedy`,
`potential_field_hunter`, and `potential_field_virus_farmer` by default;
override it with `BOT_RANDOM_STRATEGIES=a,b,c`.

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
