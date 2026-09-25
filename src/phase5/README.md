# Phase 5 — Decision Layer

Owner: Teammate

Responsibilities:

- Convert pair scores into final entity-resolution decisions.
- Apply confidence thresholds.
- Apply score-margin rules.
- Handle ambiguous candidates.
- Handle singleton Source 1 entities.
- Prevent unnecessary false merges.
- Produce final competition submission.

The decision layer must not invent matches that were not present
in the candidate set.
