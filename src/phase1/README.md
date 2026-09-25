# Phase 1 — Data Ingestion & Normalization

Phase 1 reads the competition's raw TSV entity files, validates source identity,
profiles the records, and writes deterministic normalized TSV files for Phase 2.
It never reads ground truth and never removes or merges records.

## Inputs and schema

The default inputs are `datasets/{train,test}/{train,test}_source{1,2,3}.tsv`.
Each entity file must have, in order, `entity_id`, `business_name`,
`business_address`, and `country`. IDs must be non-empty, unique within a file,
and start with the file's expected prefix (`S1-`, `S2-`, or `S3-`). Values are
read as UTF-8 strings; empty fields are missing. Country is open-set: Phase 1
normalizes labels as text and does not map, infer, or filter countries.

## Output contract

For each input, Phase 1 writes a compressed Parquet artifact under
`outputs/normalized/` (for example `train/train_source1.parquet`), with
raw columns retained and these columns appended. Parquet avoids the several
gigabytes of extra text duplication that would result from repeating raw and
normalized strings in expanded TSV files. Phase 2 should load these Parquet
artifacts using Polars or PyArrow; the Phase 2 loader is not implemented yet.

| Field | Meaning |
| --- | --- |
| `business_name_normalized` | Canonical name string |
| `business_address_normalized` | Canonical address string |
| `country_normalized` | Canonical country label |

The ID column is preserved byte-for-byte as a string. Empty/missing text
normalizes to an empty string while its raw field remains empty. Normalization
uses Unicode NFKC, case folding, punctuation-to-space replacement, and collapsed
whitespace. It retains all semantic words, including legal suffixes, and all
numeric address tokens. The same deterministic functions apply to train and test.

Phase 2 can use `entity_id` as its source record identity and the normalized name,
address, and country fields to build blocking keys. The candidate pair interface
remains the contract documented in `docs/CONTRACTS.md` (`s1_id`, `candidate_id`,
`candidate_source`, `block_method`). Phase 2 implementation is not present yet,
so there is not currently a concrete loader for an end-to-end invocation.

## Run

Install the documented dependencies, then run all six source files:

```bash
python -m src.phase1.run
```

Select a split or custom paths with:

```bash
python -m src.phase1.run --split train
python -m src.phase1.run --data-dir /path/to/datasets --output-dir /path/to/normalized
```

The command streams TSV input through Polars, validates before writing, checks
output row counts, and logs per-file row counts, field missing counts and
uniqueness, text-length summaries, duplicate ID count, and runtime. Any invalid
ID or schema fails clearly instead of silently rewriting IDs or dropping rows.
Duplicate business records with different IDs are retained.
