# Replay candidates 1, 9, and 35

Teams `1`, `9`, and `35` are the three current candidates for the strong
opponent seen in the downloaded official replays. They are behavioral
candidates, not a confirmed identification of the public leaderboard bot.
Replay JSON does not expose the public team name, so the evidence cannot prove
that any of these IDs is the approximately 45-average top bot.

`team_id` is the stable official registration ID. It is not the per-match
`player_id`/player slot (`P0` through `P7`), which is reassigned between
matches. Set `BOT_REPLAY_TEAM_ID` to the registration ID, not a player slot.

## Run a candidate

The offline candidate entry accepts any archived or custom replay clone:

```bash
BOT_REPLAY_TEAM_ID=35 uv run simulation \
  1:bots/entries/replay_candidate.py \
  7:bots/entries/random_replay_opponent.py \
  --headless
```

Replace `35` with `1` or `9`. The entry calls
`create_replay_candidate(team_id)`; these candidates do not need to be in the
curated active-opponent panel.

## Locked chronological evaluation

Rule selection uses only the earlier match cohort. The later `298xx` matches
remain locked until the direction and split rules have been chosen. After
holdout evaluation, the deployed direction profile is refitted on all source
matches; the profile metadata continues to record the strict training-only to
holdout result, not an in-sample score.

| Team | Training cohort | Holdout cohort | Decisions (train / holdout) |
|---:|---|---|---:|
| 1 | 10 matches, `27777`–`28319` | 10 matches, `29848`–`29904` | 12,673 / 12,835 |
| 9 | 18 matches, `27012`–`28318` | 11 matches, `29844`–`29905` | 23,195 / 14,680 |
| 35 | 17 matches, `11673`–`27360` | 12 matches, `29848`–`29904` | 22,240 / 15,810 |

The adopted policies are:

- Team 1: prefer visible food or prey for direction; split when a radius-2+
  merge-ready fragment, at most eight fragments, and visible prey or an edible
  virus coincide, with a 15-round rearm.
- Team 9: use regime- and fragmentation-specific fitted direction; split from
  one merge-ready radius-2+ blob toward prey within 15 units and at least three
  times smaller, with no predator and a 15-round rearm.
- Team 35: use regime-specific fitted direction; split from one merge-ready
  radius-2.5+ blob toward prey within 16 units, at least 1.5 times smaller and
  aligned at cosine 0.9+, with no predator and an 18-round rearm.

## Strict holdout results

Direction is evaluated autonomously: the clone feeds back its own previous
prediction rather than the recorded teacher command. Split rules are selected
on training matches only and replayed statefully on the holdout.

| Team | Direction median | Within 30° | Split exact F1 | Split ±2-round F1 | Strict validation |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.000001° | 71.61% | 40.14% | 48.10% | FAIL |
| 9 | 15.175° | 69.52% | 43.65% | 50.76% | FAIL |
| 35 | 7.155° | 86.59% | 78.70% | — | PASS |

Team 1 still fails the direction gate because its 75th-percentile error is
37.73° and 11.41% of directions exceed 90°. Team 9 narrowly misses the median
and within-30° gates, and its split timing remains only partially observable.
Team 35 passes the focused chronological gate with split precision 76.76%,
recall 80.74%, and movement-transition accuracy 81.54%.

Reproduce the reports with:

```bash
uv run python scripts/analyze_replay_team_1.py --jobs 4
uv run python scripts/evaluate_replay_team_9.py
uv run python scripts/evaluate_replay_team_35.py --jobs 4
```

The implementations are in
[`replay_team_1.py`](../bots/strategies/replay_team_1.py),
[`replay_team_9.py`](../bots/strategies/replay_team_9.py), and
[`replay_team_35.py`](../bots/strategies/replay_team_35.py).

## Limits

- Numeric replay IDs cannot currently be mapped reliably to public names.
- A clone matches commands conditioned on reconstructed visibility; it cannot
  recover private memory, a hidden random stream, or a changed submission.
- Frames within one match are correlated. The chronological split is by whole
  match, and the match—not the frame—is the independent evidence unit.
- Team 1 has heavy-tailed direction errors and low split recall. Team 9 still
  misses more than half of exact split events. Team 35 is the best reproduced
  of the three, but this does not prove that it is the leaderboard leader.
- `validation_passed` measures behavioral reproduction only. It does not
  establish identity or guarantee the clone's local simulator strength equals
  the original bot's official strength.
