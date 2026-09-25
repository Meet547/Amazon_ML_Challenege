# Pipeline Contracts

The project is a sequential pipeline:

RAW DATA
   ↓
PHASE 1 — NORMALIZATION
   ↓
PHASE 2 — BLOCKING
   ↓
PHASE 3 — MATCHING
   ↓
PHASE 4 — OPTIMIZATION
   ↓
PHASE 5 — DECISION
   ↓
SUBMISSION
   ↓
EVALUATION
   ↓
ERROR ANALYSIS
   ↓
IMPROVEMENT LOOP


## Phase 1 → Phase 2

Phase 1 produces normalized representations of Source 1, Source 2
and Source 3 records.

Original IDs must always be preserved.

The entity output retains `entity_id`, `business_name`,
`business_address`, and `country`, and appends `business_name_normalized`,
`business_address_normalized`, and `country_normalized`. Phase 2 uses the
preserved `entity_id` values for its `s1_id` / `candidate_id` references.
The artifacts are Zstandard-compressed Parquet under `outputs/normalized/` to
support large datasets without duplicating text as expanded TSV output.


## Phase 2 → Phase 3

Phase 2 produces candidate pairs.

Minimum conceptual fields:

- s1_id
- candidate_id
- candidate_source
- block_method

Phase 3 must never depend on Phase 2 implementation details.
It consumes the candidate-pair contract.


## Phase 3 → Phase 4

Phase 3 produces pair-level scores.

Minimum conceptual fields:

- s1_id
- candidate_id
- candidate_source
- match_score

Additional feature columns may be included.


## Phase 4 → Phase 5

Phase 4 provides validated thresholds/configuration and
optimization results.

Phase 5 applies the decision policy.


## Phase 5 → Submission

Phase 5 produces the exact submission format required by
the competition.

The official competition README and validate_submission.py
are authoritative for the final schema.

## Competition Export Boundary

The project intentionally distinguishes between:

`outputs/`
Internal pipeline artifacts, experiments, scores and diagnostics.

`output/`
Competition-ready artifacts only.

The final competition package must contain:

output/matching_results.tsv
output/candidate_pairs.tsv

The official `validate_submission.py` script is the final formatting
authority for these files.
