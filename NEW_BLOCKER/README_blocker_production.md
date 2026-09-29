# Production Business Entity Blocking (S1 → S2/S3)

This package implements a disk-backed, vectorized multi-stage blocker for the Amazon ML Challenge 2026 business entity-resolution task.

The challenge data has three source files with four columns:

- `entity_id`
- `business_name`
- `business_address`
- `country`

Source 1 is the reference source. The blocker generates candidate Source-2/Source-3 IDs for every Source-1 ID.

## Why this implementation

The actual training files shown for this project are approximately:

- S1: 210 MB
- S2: 489 MB
- S3: 504 MB
- ~1.2 GB of raw TSV
- ~15 million records total

The implementation avoids Python row-by-row candidate generation. DuckDB performs TSV ingestion, vectorized normalization, grouping, joins and external sorting, with `temp_directory` used for disk spilling when RAM is constrained. DuckDB documents both explicit CSV/TSV options such as `all_varchar` and disk spilling through `temp_directory`/`memory_limit`.

## Blocking stages

1. Country-aware exact normalized name.
2. Country-aware legal-suffix-stripped name.
3. Strict countryless exact-name fallback.
4. Strict countryless core-name fallback.
5. Country-aware name + postal code.
6. Country-aware house number + first six name characters.
7. Country-aware exact normalized address.
8. Strict countryless exact-address fallback.
9. Country-aware house number + address street signature.
10. Country-aware name edge signature (first/last four characters).
11. Rarest useful name token.
12. Second rarest useful name token.
13. Order-independent pair of rare name tokens.
14. Rare character anchors (first/middle/last short character windows).
15. Optional non-ASCII → ASCII transliteration name block.
16. Optional transliteration address block.
17. Optional sorted-neighborhood rescue, **only for S1 rows with no candidates after the main passes**.

The important design rule is that common/high-frequency blocks are not expanded blindly. They are filtered by target-side document frequency or replaced by more selective composite blocks.

## Normalization

### Business names

- lowercase
- accent stripping
- Unicode letters/numbers preserved
- `&` → `and`
- punctuation/separators collapsed
- common corporate abbreviations standardized (`pvt`, `ltd`, `corp`, `inc`)
- legal suffixes removed to form a second `name_core` representation
- both space-preserving and compact representations retained

### Addresses

- lowercase
- accent stripping
- Unicode letters/numbers preserved
- punctuation/separators collapsed
- common street/unit abbreviations canonicalized
- house number extracted
- 5/6 digit postal code extracted
- multiple compact address signatures retained

For non-ASCII data, `Unidecode` is optionally applied only to build ASCII transliteration keys. ASCII rows are copied directly into the corresponding ASCII key, so cross-script matches can share a transliterated key.

## Installation

```bash
python3 -m pip install -r requirements_blocking.txt
```

Recommended hardware for the full 15M-row training run:

- fast NVMe/SSD
- 16 GB RAM or more preferred
- 8+ CPU cores preferred
- at least 25 GB free disk before the run; more may be needed if the candidate set is large

The pipeline is disk-backed, but candidate volume can dominate storage. Keep significantly more free space than the minimum when testing broader thresholds.

## Full training run with recall evaluation

```bash
python3 business_blocker_production.py \
  --source1 datasets/train/train_source1.tsv \
  --source2 datasets/train/train_source2.tsv \
  --source3 datasets/train/train_source3.tsv \
  --ground-truth datasets/train/train_ground_truth.tsv \
  --output output/candidate_pairs.tsv \
  --workdir .blocking_work \
  --memory 8GB \
  --threads 10 \
  --sorted-rescue
```

The output is exactly the challenge candidate format:

```text
source1_entity_id    candidate_entity_ids
```

There is exactly one row per Source-1 entity. Empty candidate lists are retained.

The run also creates:

```text
.blocking_work/blocking.duckdb
.blocking_work/blocking_report.json
```

The report contains per-pass pair counts/timing, final candidate volume, and ground-truth recall when supplied.

## Recall gate

For tuning, add:

```bash
--min-recall 0.90
```

The process exits with status 2 when measured pair-level blocking recall is below 90%.

For a 95% target:

```bash
--min-recall 0.95
```

Do not accept the blocker based on candidate volume alone. The measured recall on the training ground truth is the gate.

## Tuning order

Change one parameter at a time, while keeping a frozen validation subset when comparing variants.

1. `--rare-token-df`
2. `--rare-gram-df`
3. `--gram-slots`
4. `--sorted-window`
5. name/address frequency ceilings

A practical goal is to maximize pair-level recall while minimizing candidate pairs per S1 row.

## Candidate output vs. model input

For the challenge, `candidate_pairs.tsv` must be the **last candidate set immediately before the matching model**. If you add another filtering stage later, that later set is the one that must be written to the candidate file.

## Notes on country

The pipeline treats `country` as an open string feature. It does not hard-code a US/India whitelist, so unseen test-country values remain supported.

## Fair-play compliance

The blocker uses only the supplied source tables and local computation. No external business lookup, geocoding, registration database, API, or web data is used.
