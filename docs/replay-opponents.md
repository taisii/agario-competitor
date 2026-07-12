# Official replay opponent clones

The 20 official replay files under
`.agario/replays/official/latest-20/` contain 8-player matches. Team `73`
appears exactly once in every match and is treated as this repository's team.
The remaining 140 slots collapse to 42 stable opponent `team_id` values.
Replay files do not include public team names, so clone names intentionally use
those stable IDs: `replay_team_<id>`.

Each opponent has two runnable files:

- `bots/strategies/replay_team_<id>.py`: the inferred policy.
- `bots/entries/replay_team_<id>.py`: a simulator entry point that runs that
  policy directly and does not depend on global environment configuration.

The shared fitter and visibility reconstruction live in
`scripts/replay_imitation.py` and `bots/strategies/replay_imitation.py`. They
rebuild each bot's `QueryMovePlayer` view using the official 2026.1.13 engine
rules: dynamic vision size, wall-clamped view center, square point visibility,
circle visibility, food/virus lifecycle, player blobs, and mass rankings.

## Run a local panel

Select any seven opponent entries alongside the bot under development. Counts
must sum to eight:

```bash
uv run simulation \
  1:bots/my_bot.py \
  1:bots/entries/replay_team_2.py \
  1:bots/entries/replay_team_13.py \
  1:bots/entries/replay_team_17.py \
  1:bots/entries/replay_team_22.py \
  1:bots/entries/replay_team_25.py \
  1:bots/entries/replay_team_35.py \
  1:bots/entries/replay_team_44.py
```

Smoke-test every available clone in mixed 8-player matches, with three matches
running in parallel:

```bash
uv run python scripts/simulate_replay_opponents.py --jobs 3
```

The batch report and full simulator workspaces are written below
`.agario/replay-imitation/simulations/`.

Refit the generated profiles and refresh the autonomous shadow-replay report:

```bash
uv run python scripts/replay_imitation.py
```

Multiple dated cohorts can be combined without copying or flattening them:

```bash
uv run python scripts/replay_imitation.py \
  --replay-dir .agario/replays/official/latest-20 \
  --replay-dir .agario/replays/official/submission-4
```

If cross-validation shows that a team's behavior changed between cohorts,
replace only that team's training scope with one or more repeated overrides:

```bash
uv run python scripts/replay_imitation.py \
  --replay-dir .agario/replays/official/latest-20 \
  --replay-dir .agario/replays/official/submission-4 \
  --team-replay-dir 15=.agario/replays/official/latest-20
```

This writes fitted profiles to `bots/strategies/replay_profiles.py` and the
detailed ignored report to `.agario/replay-imitation/report.json`.

## What “PASS” means

Direction vectors are normalized before comparison because the engine
normalizes commands and official bots use different raw magnitudes. A clone's
own previous prediction is fed back during evaluation; the evaluator never
uses the official bot's previous action as hidden teacher input.

The direction gate is:

- median angular error at most 15 degrees;
- 75th-percentile angular error at most 30 degrees;
- at least 70% of actions within 30 degrees; and
- at most 10% of actions over 90 degrees.

For opponents with at least five split commands, split events must reach at
least 70% precision and recall with a two-round timing tolerance. Opponents
with no split commands must keep their false-split rate at or below 0.2%.

`shadow PASS` measures the available official observations. `LOMO PASS` is the
stronger leave-one-match-out result for teams seen in multiple matches. A bot
can intentionally remain `FAIL` when the replay evidence shows hidden random
headings or sparse split decisions that cannot be inferred from observable
state. Such clones reproduce the observed movement style and event frequency,
but are not mislabeled as exact behavioral copies.

## Per-opponent verdicts

All 42 entries compile, pass their dedicated tests, and finish mixed local
simulations without a ban or timeout. Eleven pass the full-data autonomous
shadow gate. Nine also pass the stronger match-held-out gate. `Style only`
means the clone reproduces observed direction grids, inertia, target priority,
or split frequency, but not the exact hidden random sequence or split timing.

| Team | Shadow | LOMO | Reconstructed behavior |
|---:|:---:|:---:|---|
| 1 | FAIL | FAIL | Regime-fitted direction; geometric prey split gate |
| 2 | PASS | PASS | Nearest food from the nearest real fragment; no split |
| 3 | FAIL | FAIL | Stateful field mixture; close-prey split with 18-round re-arm |
| 4 | FAIL | FAIL | Field movement; sparse high-mass farming split |
| 5 | FAIL | FAIL | Inertial field mixture; sparse split |
| 6 | Style only | FAIL | Hidden-RNG 16-direction random walk; rare split |
| 9 | FAIL | FAIL | Inertial field movement; safe-prey split with 90-round re-arm |
| 10 | FAIL | FAIL | Regime-fitted direction; unstable split timing |
| 12 | FAIL | FAIL | Food/escape/inertia mixture; sparse split |
| 13 | PASS | FAIL | Food/inertia movement; uniquely identified prey split |
| 14 | FAIL | FAIL | Prey/escape mixture; close-prey split with 15-round re-arm |
| 15 | FAIL | FAIL | Strong inertia with unstable targets and split timing |
| 16 | FAIL | FAIL | Fitted direction; sparse child-safe prey split |
| 17 | PASS | PASS | Nearest food or predator field; no split |
| 21 | FAIL | FAIL | Direction PASS: food/predator regimes; split timing FAIL |
| 22 | FAIL | FAIL | Food/prey pursuit; 15-degree danger search; no split |
| 24 | FAIL | FAIL | Inertia with food, prey, and predator fields |
| 25 | PASS | PASS | Nearest food or inverse-distance predator field; no split |
| 26 | PASS | PASS | Pure nearest-food policy; no split |
| 27 | PASS | PASS | Raw vector from nearest fragment to nearest food; no split |
| 28 | PASS | PASS | Nearest food despite enemies or viruses; no split |
| 29 | PASS | PASS | Mass-center nearest-food policy; no split |
| 30 | PASS | PASS | Raw vector from nearest fragment to nearest food; no split |
| 31 | FAIL | FAIL | 16-direction inertia; burst prey splitting |
| 32 | FAIL | FAIL | Mixed food/prey/wall/predator field policy |
| 34 | FAIL | N/A | Single-trace fitted movement and split policy |
| 35 | PASS | FAIL | Regime-fitted movement; highly precise prey split gate |
| 38 | Style only | FAIL | Hidden-RNG 16-direction random walk; rare split |
| 39 | Style only | FAIL | Persistent random heading, wall reflection, predator avoidance |
| 44 | FAIL | FAIL | 32-direction inertia; tactical prey/virus/farming split |
| 48 | FAIL | FAIL | Food/prey/predator field mixture; frequent split |
| 49 | FAIL | FAIL | Food/prey/predator regimes; close-prey split with 18-round re-arm |
| 51 | FAIL | FAIL | 16-direction safe movement, continuous escape, prey split gate |
| 53 | FAIL | FAIL | Direction PASS: food/prey/predator regimes; split timing FAIL |
| 55 | Style only | FAIL | Hidden-RNG 24-direction movement; no split |
| 56 | Style only | FAIL | Hidden-RNG 24-direction movement; burst splitting |
| 58 | FAIL | FAIL | Stateful fitted field policy |
| 59 | FAIL | FAIL | Stateful direction; current close-prey split with 17-round re-arm |
| 63 | FAIL | FAIL | Inertial field movement; unstable split timing |
| 68 | FAIL | FAIL | Direction PASS: prey, predator, then food priority; split FAIL |
| 75 | PASS | PASS | Raw vector from nearest fragment to nearest food; no split |
| 77 | FAIL | FAIL | Food-led regime policy; no split |

## Top-ten reproduction target

The public leaderboard and replay payload expose different identifiers. The
leaderboard currently starts with `team`, `Banana`, `Decay Rate`, `Bot
Battle`, `Washed CS Students`, `OJ`, `imposters`, `SUNMO`, `Engorgio`, and
`PorkyPig.py`, while replay JSON deliberately contains only numeric `team_id`
values. The number shown in parentheses on the leaderboard is a match count,
not a team ID. Stale Match Report pages also remove the participant metadata,
so this repository does not claim an unverified name-to-ID mapping.

For reproducible local work, the initial top-ten target is therefore defined
by final placement across the saved 20 official matches. In order, those IDs
are `15`, `3`, `58`, `49`, `9`, `63`, `12`, `14`, `35`, and `59`. This is an
empirical replay cohort, not a claim that the IDs correspond positionally to
the ten public names above.

All ten currently fail the autonomous LOMO gate. A coarse nonlinear model
that added map-region and game-phase branches was evaluated and rejected
because it made eight of the ten direction results worse (for example, team
15's median error changed from 34.9° to 37.5°, and team 35's from 10.4° to
12.8°). Generated profiles were restored after that experiment.

A sampled nearest-observation oracle was also evaluated across held-out
matches. It was allowed to search recorded training observations using the
full reconstructed feature vector, including the teacher's previous command,
which makes it an optimistic upper bound rather than a deployable clone. Its
median direction errors remained 21.9°–31.3° for the six latter targets and
23.7°–28.3° for the four former targets; only 48.6%–62.3% of predictions were
within 30°. Consequently, exact held-out command reproduction is not supported
by the present 2–5 traces per team. More official traces or a verified mapping
to newer matches is required to distinguish hidden state/randomness from
submission changes. The runtime clones remain useful behavior models, but the
strict `validation_passed` flag must stay false until the documented gate is
actually met.

Submission #4 subsequently produced 11 fresh official matches (`13931`–
`13941`), stored separately under `.agario/replays/official/submission-4/`.
They add observations for nine of the ten target IDs and reveal three new IDs
(`78`, `79`, and `80`). Combining cohorts improves team 3's direction median
from 23.3° to 19.3° and split F1 from 0.00 to 0.44; team 14 improves from
27.5° to 24.7° and 0.12 to 0.22. Team 49 and 63 also improve direction.
Teams 15, 58, 9, 12, and 35 retain their older cohort because the combined
LOMO result regressed. Team 59 uses the combined cohort: its median direction
error rises modestly from 14.7° to 16.4°, while split F1 gains from 0.00 to
0.40. These selections are recorded in the generated report's
`team_replay_dirs` field rather than being hidden in hand-edited profiles.

Team 49's added trace exposed a missing state variable: a qualifying close
prey remains visible for several frames, but the source bot does not request a
split every frame. A team-specific 18-round re-arm interval now sits on top of
the geometric rule (prey visible within 13 arena units, radius ratio at least
1.5, no predator). Mean held-out split F1 rises from 0.22 to 0.74. Its overall
LOMO verdict remains FAIL because direction is just below the aggregate gate
(69.8% rather than 70% within 30°) and not every split fold reaches 70%
precision and recall.

Team 59 uses the same state-machine form with a slightly wider prey gate and
a 17-round re-arm interval. Its mean held-out split F1 rises from 0.40 to 0.61;
the two fresh Submission #4 folds score 0.77 and 0.70. Older matches remain
inconsistent, so this is treated as reproduction of the current behavior and
does not change the strict overall FAIL verdict.

Team 3 now uses an 18-round re-arm interval around its larger-prey gate. Mean
held-out split F1 rises from 0.44 to 0.63. The current-cohort folds with no
split produce no false events, while match 13938 reaches F1 0.60. Its movement
model remains below the direction gate, so the overall verdict stays FAIL.

Team 35's runtime and evaluator now share its previously hand-verified split
invariant. Mean held-out split F1 rises from 0.19 to 0.99. Team-specific ridge
tuning increases direction-passing folds from three to four out of five. A
shared predator-visible direction mixture then lowers the median direction
error from 10.4° to 8.6° and raises the within-30° rate from 79.4% to 81.6%.
Only match 11739 remains outside the direction gate (p75 32.3°, 11.8% over
90°). The overall verdict stays FAIL because LOMO requires every fold to pass.

Team 9 previously used a synthetic hash to match only the aggregate split
frequency. Replaying that runtime rule against the source observations gives
F1 0.065, below even the generic classifier. It has been replaced with a
deterministic safe-prey gate and 90-round re-arm interval shared by runtime and
evaluation. Mean held-out split F1 rises from 0.17 to 0.42 (full-trace F1
0.49). Match 11756 contains one unmatched split, so the overall verdict stays
FAIL rather than treating frequency matching as behavioral reproduction.

Team 14 uses a close-prey gate with a 15-round re-arm interval. Across all
seven traces the fixed rule reaches precision 65.4%, recall 53.1%, and F1
0.59, versus mean held-out F1 0.22 for the generic classifier. Two new events
in match 13939 remain unmatched, and direction is still below its gate, so the
overall verdict remains FAIL.
