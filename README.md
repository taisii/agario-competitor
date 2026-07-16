# agario-competitor

Bot repository for the **SYNCS 2026 Agar.io competition**. The official game
has eight players and accepts one generated Python file.

## Setup

```bash
uv sync --frozen
```

The lockfile resolves `agario-kit==2026.1.14`, matching the engine used by the
saved official match reports.

## Six-policy sparring pool

Local opponents deliberately use six diverse, team-independent policies:

- `semantic_lookahead`: semantic targets with bounded second-ply refinement.
- `semantic_potential`: the fast one-ply semantic potential policy.
- `replay_dominance`: bounded receding-horizon search for growth and elimination.
- `threat_aware_receding_horizon`: conservative adversarial prediction.
- `event_driven_static_search`: static growth with tactical event handling.
- `static_retained_growth`: two-stage growth ranking with a catastrophe filter.

There are no team-ID-specific opponent clones. Each `random_opponent` process
selects one policy after its player slot is known. Selection is deterministic
for the same seed and trial, but shuffled between trials. In the standard
candidate-at-slot-zero layout, slots 1–7 always contain all six policies, with
one policy duplicated.

Run one match:

```bash
uv run simulation \
  1:bots/my_bot.py \
  7:bots/entries/random_opponent.py \
  --headless
```

Override the pool only with registered policies:

```bash
BOT_RANDOM_STRATEGIES=semantic_lookahead,replay_dominance \
uv run simulation 1:bots/my_bot.py 7:bots/entries/random_opponent.py --headless
```

## Paired benchmarks

The default benchmark is the candidate in slot 0 against the randomized
six-policy pool:

```bash
uv run python scripts/benchmark_simulations.py --trials 8 --fast --jobs 1
```

Fast benchmarks give only the seven opponent slots a relaxed 10-second
per-turn / 60-second cumulative budget. The tracked candidate remains under a
one-second per-turn gate, so a strong but expensive sparring policy cannot hide
a submission timeout.

For large statistical runs where timeout validity is intentionally irrelevant,
use throughput mode:

```bash
uv run python scripts/benchmark_simulations.py --throughput --trials 32
```

Throughput mode skips recordings, relaxes every player to 60 seconds per turn
and 600 seconds cumulatively, and selects up to four concurrent matches from
the available CPU count. Override that choice explicitly with `--jobs 8`, for
example. Re-run promising candidates with ordinary `--fast --jobs 1` before
submission so the one-second gate is tested without CPU-contention noise.

`BOT_RANDOM_SEED` and `BOT_BENCHMARK_TRIAL` make the opponent assignment
reproducible. Variants in the same trial therefore see the same policies in the
same slots, preventing opponent sampling noise from being mistaken for a
strategy improvement.

Results are written below `.agario/benchmarks/`. Candidate acceptance should
use paired mean final mass; rank, peak mass, kills, and action diagnostics are
secondary evidence.

## Strategy entry points

- `bots/entries/semantic_lookahead.py`
- `bots/entries/semantic_potential.py`
- `bots/entries/replay_dominance.py`
- `bots/entries/threat_aware_receding_horizon.py`
- `bots/entries/event_driven_static_search.py`
- `bots/entries/static_retained_growth.py`
- `bots/entries/random_opponent.py`

Entry files only select a policy. Process I/O and telemetry live in
`bots/runtime.py`, while shared physics live under `bots/simulation/`.

## Building the official submission

The default build is the current submission candidate, `semantic_lookahead`:

```bash
uv run python scripts/build_submission.py
uv run python -m py_compile dist/my_bot.py
uv run simulation 1:dist/my_bot.py 7:bots/entries/random_opponent.py --headless
```

Build another member of the pool with `--strategy`; for example, the previous
`replay_dominance` candidate remains available for an immediate rollback:

```bash
uv run python scripts/build_submission.py --strategy replay_dominance
```

The builder prints the SHA-256 hash. Record that hash with the corresponding
portal submission at the [SYNCS Game Hub](https://syncs.org.au/competition2026/game).

## Replay analysis

Saved official replays remain analysis data, not executable opponents. Useful
tools include:

```bash
uv run python scripts/analyze_official_match.py path/to/replay.json
uv run python scripts/compare_strategy_decisions.py path/to/game.json --player 0
uv run python scripts/analyze_mass_gain_sources.py path/to/game.json --player 0
```

## Verification

```bash
uv run pytest -q
uv run python scripts/build_submission.py
uv run python -m py_compile dist/my_bot.py
```

When the competition engine changes, update it explicitly and rerun the full
verification:

```bash
uv lock --upgrade-package agario-kit
uv sync --frozen
uv run pytest -q
```
