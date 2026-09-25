# Instructions for Phase 3–5 Development

You are responsible for Phase 3, Phase 4 and Phase 5.

DO NOT redesign the repository.

DO NOT:

- Rename phase folders.
- Move Phase 1.
- Move Phase 2.
- Create another src directory.
- Create another pipeline architecture.
- Duplicate normalization.
- Duplicate blocking.
- Modify datasets.
- Add external business databases.
- Use external geocoding/business APIs.
- Hard-code competition test labels.
- Tune on hidden/test labels.

Your code belongs primarily inside:

src/phase3/
src/phase4/
src/phase5/

Shared evaluation code belongs inside:

src/evaluation/

Shared utilities belong inside:

src/common/

Tests belong inside:

tests/


## Integration

Phase 3 receives candidate pairs from Phase 2.

Conceptually:

Phase 2
  ↓
candidate_pairs
  ↓
Phase 3
  ↓
scored_pairs
  ↓
Phase 4
  ↓
optimized configuration
  ↓
Phase 5
  ↓
final matching.tsv


## Core principle

Blocking determines what pairs are even considered.

Matching determines how likely each candidate pair is a true match.

Decision determines whether a scored candidate is actually emitted.

Do not allow the matcher or decision layer to recover a pair
that Phase 2 never generated.


## Evaluation

Use training-derived validation/holdout data.

Track:

- Precision
- Recall
- Macro F0.5
- TP
- FP
- FN
- Candidate recall
- False merges
- Missed matches
- Singleton behavior

Because F0.5 weights precision more heavily than recall,
avoid aggressive matching when evidence is weak.


## Engineering

Prefer:

- Polars
- PyArrow
- DuckDB
- NumPy
- SciPy
- scikit-learn
- LightGBM

for large-scale processing.

Avoid:

- pandas operations requiring the entire dataset in RAM
- full Cartesian products
- unnecessary copies of huge datasets


## Reproducibility

Every experiment should have:

- configuration
- random seed
- metrics
- model configuration
- output location

Do not overwrite the previous experiment silently.


## Final rule

If a change requires modifying Phase 1 or Phase 2,
STOP and discuss it with Meet first.

The interface between phases is more important than
individual implementation details.
