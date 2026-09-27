# Phase 2.6 High-Recall Final Report

## Status

The frozen Phase 2.6 blocker achieved **95.0733% pair recall on full train** and passed the separate deterministic 100K holdout at **95.0563%**. Phase 2.6 is validated at the pair-generation contract level. S3 recall is slightly below 95%; this is reported directly below.

## Frozen configuration

- Existing exact-name, exact-address, rare-name-token, name-token-pair, address-token, numeric-address, core-name, and sorted-name blocks retain their Phase 2 limits.
- Address token-pair keys exclude the configured generic address tokens. Component token frequency ceiling: **25,000 per source side**. Joint cross-source pair estimate cap: **5,000**.
- Name-address composite keys use name tokens of length at least 3 excluding legal-form tokens, and address tokens excluding generic address tokens. Component frequency ceiling: **25,000 per source side**. Joint cross-source pair estimate cap: **1,000**.
- Key-frequency profiles use full entity populations. Ground truth is loaded only for evaluation; it does not influence key construction, frequency limits, or record routing.
- The 100K development sample is frozen at SHA-256 `8b9000172e2680a37707dd9fd6de9707309d430d14475ec6181bcb3d1dc269db`. The separate 100K holdout excludes every development S1 and has SHA-256 `d85426dcefbf19650375bb63d3988e9491d07b1a786765e55863be098baa0e48`.

## Development results

| Configuration | Candidate pairs | Pair recall | S2 | S3 | Full-S1 | Mean/S1 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing baseline | 5,260,514 | 91.4838% | 91.7964% | 91.1905% | 79.6699% | 52.61 | 135 | 218 | 911 |
| Address pairs, cap 1,000 | 6,646,799 | 93.9704% | 94.3192% | 93.6431% | 84.5548% | 66.47 | 174 | 271 | 1,405 |
| Address pairs, cap 2,500 | 7,676,786 | 94.1801% | 94.5325% | 93.8494% | 84.9638% | 76.77 | 214 | 343 | 2,323 |
| Address pairs, cap 5,000 | 8,591,532 | 94.2970% | 94.6490% | 93.9666% | 85.1947% | 85.92 | 253 | 426 | 3,241 |
| Address pairs 5,000 + name-address 1,000 | 10,039,113 | **95.0765%** | 95.2267% | 94.9355% | 86.6567% | 100.39 | 281 | 467 | 3,286 |

At the point the development configuration crossed 95%, tuning stopped as required. Name-pair joint-cap, numeric-composite, adaptive, cross-script, and targeted-character experiments were not run on development after that stop condition.

Incremental recovery and candidate growth against the preceding configuration:

| Addition | Newly recovered true pairs | Added unique candidate pairs | Recovery per million added |
|---|---:|---:|---:|
| Address pairs, cap 1,000 | 8,597 | 1,386,285 | 6,203 |
| Raise address-pair cap to 2,500 | 725 | 1,029,987 | 704 |
| Raise address-pair cap to 5,000 | 404 | 914,746 | 442 |
| Name-address composite, cap 1,000 | 2,695 | 1,447,581 | 1,861 |

The composite stage supplied the final measured gain to the target with mean candidates/S1 100.39 and P99 467, within the stated engineering targets.

## Holdout result

The frozen configuration was evaluated once on the disjoint deterministic 100K S1 holdout against full train S2/S3:

- True pairs: 345,914; recovered: 328,813; pair recall: **95.0563%**.
- S2 recall: **95.2684%**; S3 recall: **94.8582%**.
- Full-S1 recall: **86.6471%** (all known target entities recovered for that S1).
- Candidate pairs: 10,145,238; mean/S1 101.45; P95 285; P99 479; max 1,549.
- Development/holdout overlap: zero.

## Full-train result

The actual 2,206,821 train S1 rows were retrieved against all 5,034,616 S2 and 5,285,603 S3 rows. No sample extrapolation was used.

- Candidate pairs: **222,529,211**.
- Pair recall: **95.0733%** (7,262,049 / 7,638,365).
- S2: **95.2448%** (3,517,980 / 3,693,619).
- S3: **94.9128%** (3,744,069 / 3,944,746).
- Full-S1 recall: **86.7012%** (1,806,483 / 2,083,574 positive S1s).
- S1 coverage: 2,206,444 / 2,206,821; 377 S1s had zero candidates.
- Candidates/S1: mean 100.85, median 73, P95 284, P99 470, max 3,396.
- Unique candidate IDs: 9,895,733.
- The 2,667,796,160-byte final Parquet artifact is `outputs/candidates/train/candidate_pairs.parquet`. SHA-256: `32a43ca8514098c84fb856a026d965b6423c37525d671967d5b635579573401f`.

## Remaining misses

There are **376,316** missed full-train pairs (4.9267%). A separate signal audit found the following overlapping evidence among missed pairs; counts are not additive:

| Residual signal | Missed pairs with signal |
|---|---:|
| At least one shared normalized address token | 284,974 (75.72%) |
| At least two shared normalized address tokens | 275,381 (73.18%) |
| Shared numeric address component | 220,400 (58.57%) |
| At least one shared normalized name token | 192,750 (51.23%) |
| Shared name and address token | 101,815 (27.06%) |
| Exact normalized address | 0 |
| Exact normalized name | 0 |

These counts indicate unresolved selectivity/admission and normalization cases remain. Cross-script and character-specific residual attribution was not performed because the frozen configuration already passed the overall recall target and the strict development stop condition prohibits further tuning on that sample. S3 remains 0.0872 percentage points below 95% on full train.

## Integrity and Phase 3 handoff

The final Parquet schema is:

| Column | Type | Meaning |
|---|---|---|
| `s1_id` | string | Phase 1 S1 ID |
| `candidate_id` | string | S2 or S3 target ID |
| `candidate_source` | string | `S2` or `S3` |
| `block_methods` | list[string] | Sorted provenance methods |

All 11 configured provenance methods occur; no row has empty provenance. The artifact is deduplicated by `(s1_id, candidate_id, candidate_source)` through 64 deterministic hash partitions and per-partition grouping.

Independent DuckDB integrity checks found zero candidate S1 IDs absent from Phase 1, zero candidate targets absent from S2/S3, zero source-prefix mismatches, zero malformed nonempty match lists, zero ground-truth S1/target IDs absent from Phase 1, and zero duplicate ground-truth pairs. Candidate rows are formed only from Phase 1 source IDs; no labels enter generation.

The contract supplies `s1_id`, `candidate_id`, `candidate_source`, and `block_methods` expected by the Phase 3 loader. Phase 3 was not modified or run.

## Resuming the development experiments

The baseline preparation and staged experiment runner are in `scripts/phase2/prepare_phase2_6_baseline.py` and `scripts/phase2/resume_phase2_6_experiments.py`. They use the frozen 100K ID file, atomically checkpoint the manifest, cache components/unions under `/private/tmp/phase2_6_cache` (override with `PHASE2_6_CACHE_DIR`), and skip completed experiments with a valid cached union on restart. The runner stops development tuning as soon as pair recall reaches 95%.

## Runtime, memory, reproducibility, and tests

- Development stage runtimes: address cap 1,000 183.87s; 2,500 225.26s; 5,000 265.10s; name-address 1,000 278.94s. Reported experiment peak RSS was 6.85 GiB.
- Holdout runtime: 480.14s; peak RSS 7.45 GiB.
- Independent full-train metrics pass: 194.18s; peak RSS 1.19 GiB. The full candidate construction completed, but its wall time/peak RSS was not captured by the first runner after its post-write metrics step exited; no estimate is substituted.
- Available disk after completion: approximately 5.8 GB. Full candidate generation used staged Parquet and streaming hash-partition deduplication.
- Reproducibility: baseline, all three address-pair tiers, and the selected name-address development result were reproduced exactly by the checkpointed runner against the earlier measurements. The holdout and full-train artifact were each run once; the artifact hash above identifies the measured full-train output.
- Tests: **40 passed**. Phase 2 CLI help succeeded, the full suite includes Phase 1/Phase 2 integration coverage, and `git diff --check` passed.
- Recovery attempts: an early checkpoint runner had a Polars DataFrame/LazyFrame join mismatch, fixed before checkpoint completion; the first full-train merge was interrupted by premature cleanup of source blocks, then rerun with staged deletion only after downstream copies completed. The final full-train artifact and metrics come from the successful rerun.

## Result

Overall full-train pair recall exceeds 95%, and the disjoint holdout also exceeds 95%. S3 recall is slightly below 95% on both holdout and full train. This report records the measured result and residual misses without claiming 95% for either source-specific S3 metric.
