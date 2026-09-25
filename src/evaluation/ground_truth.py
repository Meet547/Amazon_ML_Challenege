"""Load and validate training ground-truth labels for supervised phases.

This module is intentionally separate from Phase 1 entity normalization. It
returns one row per Source 1 entity and a list of matched S2/S3 IDs; empty lists
represent valid singleton labels.
"""

from pathlib import Path

import polars as pl

GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
ENTITY_ID_COLUMN = "entity_id"


class GroundTruthValidationError(ValueError):
    """Raised when training labels or their entity references are invalid."""


def _scan_tsv(path: str | Path) -> pl.LazyFrame:
    return pl.scan_csv(path, separator="\t", has_header=True, infer_schema=False,
                       null_values=[""], encoding="utf8-lossy")


def _count(frame: pl.LazyFrame) -> int:
    return frame.select(pl.len()).collect(engine="streaming").item()


def _validate_entity_file(path: str | Path, source: str) -> pl.LazyFrame:
    frame = _scan_tsv(path)
    columns = frame.collect_schema().names()
    if ENTITY_ID_COLUMN not in columns:
        raise GroundTruthValidationError(
            f"{path}: required {ENTITY_ID_COLUMN!r} column is missing"
        )
    ids = pl.col(ENTITY_ID_COLUMN)
    stats = frame.select(
        ids.null_count().alias("null_ids"),
        (ids.is_not_null() & (ids.str.strip_chars() == "")).sum().alias("blank_ids"),
        (ids.is_not_null() & ~ids.str.starts_with(f"{source}-")).sum().alias("wrong_prefix"),
    ).collect(engine="streaming").row(0, named=True)
    duplicate_count = _count(
        frame.group_by(ENTITY_ID_COLUMN).len()
        .filter(pl.col(ENTITY_ID_COLUMN).is_not_null() & (pl.col("len") > 1))
    )
    stats["duplicate_ids"] = duplicate_count
    problems = {key: value for key, value in stats.items() if value}
    if problems:
        raise GroundTruthValidationError(f"{path}: invalid {source} IDs: {problems}")
    return frame.select(pl.col(ENTITY_ID_COLUMN).alias("entity_id"))


def load_ground_truth(
    ground_truth_path: str | Path = "datasets/train/train_ground_truth.tsv",
    source1_path: str | Path = "datasets/train/train_source1.tsv",
    source2_path: str | Path = "datasets/train/train_source2.tsv",
    source3_path: str | Path = "datasets/train/train_source3.tsv",
) -> pl.DataFrame:
    """Return validated labels with ``matched_entity_ids`` as ``List[String]``.

    All paths refer to training data. This function is never called by Phase 1
    and must not be used to derive or alter test entity features.
    """
    ground_truth = _scan_tsv(ground_truth_path)
    columns = ground_truth.collect_schema().names()
    if columns != GROUND_TRUTH_COLUMNS:
        raise GroundTruthValidationError(
            f"{ground_truth_path}: expected columns {GROUND_TRUTH_COLUMNS}, got {columns}"
        )

    s1_ids = _validate_entity_file(source1_path, "S1")
    s2_ids = _validate_entity_file(source2_path, "S2")
    s3_ids = _validate_entity_file(source3_path, "S3")

    source1 = pl.col("source1_entity_id")
    raw_matches = pl.col("matched_entity_ids").fill_null("")
    base = ground_truth.select(source1, raw_matches.alias("_raw_matches"))
    issues: dict[str, int] = {}

    s1_nulls = _count(base.filter(source1.is_null()))
    if s1_nulls:
        issues["null_source1_entity_id"] = s1_nulls
    s1_blanks = _count(base.filter(source1.is_not_null() & (source1.str.strip_chars() == "")))
    if s1_blanks:
        issues["blank_source1_entity_id"] = s1_blanks
    s1_prefix = _count(base.filter(source1.is_not_null() & ~source1.str.starts_with("S1-")))
    if s1_prefix:
        issues["invalid_source1_prefix"] = s1_prefix
    duplicate_s1 = _count(
        base.group_by("source1_entity_id").len()
        .filter(pl.col("source1_entity_id").is_not_null() & (pl.col("len") > 1))
    )
    if duplicate_s1:
        issues["duplicate_source1_rows"] = duplicate_s1
    malformed_lists = _count(base.filter(pl.col("_raw_matches").str.contains(r"(^,|,$|,,)")))
    if malformed_lists:
        issues["malformed_target_lists"] = malformed_lists

    missing_s1 = _count(
        base.select("source1_entity_id").filter(pl.col("source1_entity_id").is_not_null())
        .join(s1_ids, left_on="source1_entity_id", right_on="entity_id", how="anti")
    )
    if missing_s1:
        issues["unknown_source1_references"] = missing_s1
    unlabeled_s1 = _count(
        s1_ids.join(
            base.select("source1_entity_id"),
            left_on="entity_id", right_on="source1_entity_id", how="anti",
        )
    )
    if unlabeled_s1:
        issues["missing_source1_labels"] = unlabeled_s1

    nonempty = base.filter(pl.col("_raw_matches") != "")
    targets = nonempty.select(
        "source1_entity_id",
        pl.col("_raw_matches").str.split(",").alias("matched_entity_id"),
    ).explode("matched_entity_id", empty_as_null=True)
    empty_tokens = _count(targets.filter(pl.col("matched_entity_id") == ""))
    if empty_tokens and "malformed_target_lists" not in issues:
        issues["empty_target_tokens"] = empty_tokens
    whitespace_targets = _count(
        targets.filter(pl.col("matched_entity_id") != pl.col("matched_entity_id").str.strip_chars())
    )
    if whitespace_targets:
        issues["target_whitespace"] = whitespace_targets
    invalid_prefix = _count(
        targets.filter(
            ~pl.col("matched_entity_id").str.starts_with("S2-")
            & ~pl.col("matched_entity_id").str.starts_with("S3-")
        )
    )
    if invalid_prefix:
        issues["invalid_target_prefix"] = invalid_prefix
    duplicate_pairs = _count(
        targets.group_by("source1_entity_id", "matched_entity_id").len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_pairs:
        issues["duplicate_source1_target_pairs"] = duplicate_pairs

    s2_targets = targets.filter(pl.col("matched_entity_id").str.starts_with("S2-"))
    s3_targets = targets.filter(pl.col("matched_entity_id").str.starts_with("S3-"))
    unknown_s2 = _count(
        s2_targets.select("matched_entity_id").join(
            s2_ids, left_on="matched_entity_id", right_on="entity_id", how="anti"
        )
    )
    unknown_s3 = _count(
        s3_targets.select("matched_entity_id").join(
            s3_ids, left_on="matched_entity_id", right_on="entity_id", how="anti"
        )
    )
    if unknown_s2:
        issues["unknown_s2_references"] = unknown_s2
    if unknown_s3:
        issues["unknown_s3_references"] = unknown_s3

    if issues:
        raise GroundTruthValidationError(f"Invalid training ground truth: {issues}")

    labels = base.select(
        "source1_entity_id",
        pl.when(pl.col("_raw_matches") == "")
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col("_raw_matches").str.split(","))
        .alias("matched_entity_ids"),
    )
    return labels.collect(engine="streaming")
