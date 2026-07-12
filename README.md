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
uv sync --upgrade
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
uv run interactive 2:bots/my_bot.py 5:bots/other_bot.py
```

To watch a non-interactive simulation, run:

```bash
uv run simulation 8:bots/my_bot.py
```

To run benchmark matches in parallel and aggregate results, run:

```bash
uv run python scripts/benchmark_simulations.py --trials 8 --jobs 2
```

The benchmark runner writes one isolated simulation workspace per match under
`.agario/benchmarks/`, plus `matches.csv`, `results.json`, and `run_config.json`
at the benchmark root. The default benchmark compares `baseline` and `smooth`
beam settings with four `food_greedy` bots and four `beam_survival` bots.

To test the default replay-dominance receding-horizon strategy against seven
randomly selected official-replay clone strategies, run:

```bash
uv run simulation 1:bots/entries/replay_dominance.py 7:bots/entries/random_replay_opponent.py --headless
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
`beam_survival`, and `potential_field_hunter` by default; override it with
`BOT_RANDOM_STRATEGIES=a,b,c`.

Example strategy screen:

```bash
uv run python scripts/benchmark_simulations.py \
  --trials 4 \
  --jobs 2 \
  --variants \
    receding_horizon:BOT_STRATEGY=threat_aware_receding_horizon \
    potential_field:BOT_STRATEGY=potential_field_hunter \
    beam_survival:BOT_STRATEGY=beam_survival \
  --submission 1:bots/my_bot.py 7:bots/entries/random_opponent.py \
  --tracked-slots 0
```

Strategy implementations are grouped by their fundamental algorithm:

- `bots/strategies/greedy.py`: immediate food, prey, and escape target rules.
- `bots/strategies/potential_field.py`: weighted potential-field movement.
- `bots/strategies/beam_search.py`: all beam-search implementations and profiles.
- `bots/strategies/receding_horizon.py`: predictive receding-horizon strategies.
- `bots/strategies/virus_farming.py`: safe virus pursuit and potential-field farming.
- `bots/strategies/replay_imitation.py`: replay-fitted imitation policy; it needs a validated profile before it can be registered or ranked.
- `bots/snapshots/`: frozen comparison artifacts, not active strategy source files.

Available local strategy entry points:

- `bots/entries/food_greedy.py`: nearest-food baseline.
- `bots/entries/survival_greedy.py`: flee nearby predators, then chase prey/food.
- `bots/entries/virus_hunter.py`: prioritise consumable viruses, growing on prey/food until one is safely reachable.
- `bots/entries/potential_field_virus_farmer.py`: safe virus farming with potential-field prey/food growth.
- `bots/entries/beam_survival.py`: shallow rollout focused on survival and food/prey.
- `bots/entries/potential_field_hunter.py`: potential-field hunter.
- `bots/entries/beam_hunter.py`: beam rollout with move/split candidates, predator, wall, virus, food, and prey scoring.
- `bots/entries/threat_aware_receding_horizon.py`: robust adversarial prediction with engine-matched split physics.
- `bots/entries/threat_aware_receding_horizon_reference.py`: deliberately expensive reference profile of the same strategy.
- `bots/entries/virus_farming_receding_horizon.py`: frozen virus-farming receding-horizon comparison strategy.
- `bots/entries/replay_dominance.py`: default unified search policy for survival, virus growth, wall mobility, and rival elimination.
- `bots/entries/beam_rl_*.py`: imported beam/RL profile bots with submission-safe caps.
- `bots/entries/random_opponent.py`: picks one stable strategy at process startup.

Example mixed match:

```bash
uv run simulation 2:bots/entries/food_greedy.py 2:bots/entries/survival_greedy.py 2:bots/entries/potential_field_hunter.py 2:bots/entries/beam_hunter.py
```

## Writing a bot

- Put your bot logic in `bots/my_bot.py`.
- Import `Game` from `helper.game`.
- Read visible state from `game.state`.
- Return moves using the `lib.interface.events.moves` models.

## Updating during the competition

We may make changes to the game engine during the event. When a new platform version is published, please run this command to bring your version of the game engine up to date:

```bash
uv sync --upgrade
```

We will send a message on [Discord](https://discord.gg/24We3YWM7e) if this happens.
