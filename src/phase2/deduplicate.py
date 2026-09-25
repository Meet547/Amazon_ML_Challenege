"""Candidate-pair deduplication and competition TSV serialization."""

from pathlib import Path

import polars as pl

from src.common.io import ensure_parent


def deduplicate_with_provenance(raw_pairs: pl.LazyFrame) -> pl.LazyFrame:
    """Return one row per entity pair with a sorted unique method list."""
    return raw_pairs.group_by("s1_id", "candidate_id", "candidate_source").agg(
        pl.col("block_methods")
        .list.explode(keep_nulls=False, empty_as_null=False)
        .unique().sort().alias("block_methods")
    ).select("s1_id", "candidate_id", "candidate_source", "block_methods")


def write_candidate_tsv(
    pairs: pl.LazyFrame,
    source1: pl.LazyFrame,
    output_path: str | Path,
) -> Path:
    """Write one submission-format candidate row per Source 1 ID."""
    path = ensure_parent(output_path)
    by_source1 = pairs.group_by("s1_id").agg(
        pl.col("candidate_id").sort().alias("_candidate_ids")
    )
    export = (
        source1.select(pl.col("entity_id").alias("source1_entity_id"))
        .join(by_source1, left_on="source1_entity_id", right_on="s1_id", how="left")
        .select(
            "source1_entity_id",
            pl.col("_candidate_ids").list.join(",").fill_null("").alias("candidate_entity_ids"),
        )
    )
    export.sink_csv(path, separator="\t", include_header=True, engine="streaming")
    return path
