# agario-competitor

Starter repo for a competition bot.

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

To test one champion strategy against seven randomly selected stable opponent
strategies, run:

```bash
uv run simulation 1:bots/entries/champion.py 7:bots/entries/random_opponent.py --headless
```

`bots/entries/champion.py` defaults to the submission-safe `champion` strategy. Override it with
`BOT_CHAMPION_STRATEGY=<strategy_name>`. `random_opponent` samples from
`food_greedy`, `survival_greedy`, `beam_survival`, and `potential_hunter` by
default; override with `BOT_RANDOM_STRATEGIES=a,b,c`.

Example champion screen:

```bash
uv run python scripts/benchmark_simulations.py \
  --trials 4 \
  --jobs 2 \
  --variants \
    potential_hunter:BOT_CHAMPION_STRATEGY=potential_hunter \
    survival_greedy:BOT_CHAMPION_STRATEGY=survival_greedy \
    beam_survival:BOT_CHAMPION_STRATEGY=beam_survival \
  --submission 1:bots/entries/champion.py 7:bots/entries/random_opponent.py \
  --tracked-slots 0
```

Available local strategy entry points:

- `bots/entries/food_greedy.py`: nearest-food baseline.
- `bots/entries/survival_greedy.py`: flee nearby predators, then chase prey/food.
- `bots/entries/beam_survival.py`: shallow rollout focused on survival and food/prey.
- `bots/entries/potential_hunter.py`: potential-field hunter inspired by the public bot style.
- `bots/entries/beam_hunter.py`: beam rollout with move/split candidates, predator, wall, virus, food, and prey scoring.
- `bots/entries/champion.py`: robust receding-horizon submission strategy using public opponent moves, adversarial predator prediction, and engine-matched split physics.
- `bots/entries/champion_reference.py`: deliberately expensive reference profile for strength experiments before optimizing the same logic for submission.
- `bots/entries/beam_rl_*.py`: imported beam/RL profile bots with submission-safe caps.
- `bots/entries/random_opponent.py`: picks one stable strategy at process startup.

Example mixed match:

```bash
uv run simulation 2:bots/entries/food_greedy.py 2:bots/entries/survival_greedy.py 2:bots/entries/potential_hunter.py 2:bots/entries/beam_hunter.py
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
