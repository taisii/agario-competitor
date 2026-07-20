# Knowledge index

`current/` contains evidence supporting the submitted `replay_distilled` policy.
Read these documents before changing its behavior:

- `current/teacher-selection-2026-07-16.md`: strategy selection by mean final mass.
- `current/replay-distilled.md`: residual model, runtime, and submission evidence.
- `current/official-vs-local-2026-07-17.md`: limits of local replay evaluation.

These are evidence records, not current CLI manuals. One-off analysis commands and
alternative strategy builders mentioned in them may have been removed; use the root
README for supported commands.

`history/` preserves rejected or superseded investigations so failed approaches are
not repeated. Commands in historical documents may reference tools removed during
the submission-only repository reduction.

## Structure decision — 2026-07-20

The repository was reduced from a general strategy research platform to the current
submission dependency closure. Confirmed removals include old strategy families,
replay-clone runtime code, one-off analyzers, generated calibration artifacts, and
their dedicated tests.

The tracked `.agario` replay cohort (30 files, about 347MB) was removed from Git but
left intact in the local ignored workspace. No Git-history rewrite was performed.

Measured source reduction relative to commit `135c3f2` and final verification are
recorded in the refactor handoff for this branch.
