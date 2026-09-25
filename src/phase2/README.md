# Phase 2 — Candidate Generation / Blocking

Phase 2 emits a high-recall set of S1→S2/S3 candidate pairs. It does not score
pairs or decide whether two entities match. Inputs are the six Phase 1 Parquet
tables under `outputs/normalized/{train,test}` with the seven ordered String
columns documented in `docs/CONTRACTS.md`. The loader keeps the three sources
separate and uses lazy Polars scans.

## Blocking methods

| Method | Key | Policy |
|---|---|---|
| `exact_name` | `business_name_normalized` | Exact non-empty values. |
| `name_country` | normalized name and country | Recorded as provenance when an exact-name pair also has equal country. Since it is a strict subset of exact-name pairs, it does not trigger a second candidate join. |
| `exact_address` | `business_address_normalized` | Exact non-empty values. |
| `rare_name_token` | unique tokens of normalized names | Tokens of at least five characters; retain a shared token only if S1 frequency × combined S2/S3 frequency is at most 1,000. |

The token policy is measured, frequency-based common-key suppression, not a
fixed stopword list. On the current normalized data, exact-name blocks have
about 904.6k shared keys, 21.76M estimated raw pairs, median size 2, P90 5,
P95 9, P99 33, maximum 1,043 records. Exact-address blocks have about 487.5k
shared keys, 0.77M estimated pairs, median size 2, P90 3, P95 4, P99 7,
maximum 32. Name-token keys without a cap imply roughly 98.8B pair products;
the per-key cap retains about 63.5k shared keys and estimates at most 7.8M raw
token pairs. These are pre-run frequency-profile measurements; the run writes
the detailed split-specific statistics and top shared keys into its metrics
JSON.

All methods are unioned then deduplicated on `(s1_id, candidate_id,
candidate_source)`. The internal Parquet schema is `s1_id`, `candidate_id`,
`candidate_source`, `block_methods` (`List[String]`). Candidate source is
carried explicitly from S2/S3 inputs. Empty normalized keys never block.

## Evaluation and outputs

Train evaluates generated pairs against the validated train-only ground truth.
Recall is computed over positive truth pairs; singleton S1 rows have no true
target and are counted separately. The report includes overall and target
source recall, full/partial/zero recovery among positive S1 entities,
candidate volume including zero-candidate S1s, per-method/source volumes,
and cumulative method contribution. Test mode does not load ground truth.

```bash
.venv/bin/python -m src.phase2.run --split train
.venv/bin/python -m src.phase2.run --split test
```

Runs write internal pairs under `outputs/candidates/{split}/`, metrics under
`outputs/metrics/phase2_{split}.json`, and test competition-format candidates
to `output/candidate_pairs.tsv`. The TSV has exactly
`source1_entity_id` and `candidate_entity_ids`, one row per S1 (including those
with no candidates). Internal Parquet retains provenance for Phase 3.

## Limitations

Exact blocks miss spelling and formatting variations that normalization does
not unify. Rare-token blocks can still add false candidates; Phase 2 makes no
match decision. The 1,000 estimated-pair cap is a reproducible initial
tradeoff; train recall must be reviewed before treating the configuration as
final. Candidate Parquet and the competition TSV can be large at this dataset
scale. Full-data resource requirements should be checked before running.
