"""Blocking-key construction and data-driven bucket profiling."""

import polars as pl

from .config import LARGE_BLOCK_PAIR_THRESHOLD, LARGE_BLOCK_RECORD_THRESHOLD


def source2_and_source3(tables) -> pl.LazyFrame:
    """Combine target sources while carrying explicit S2/S3 identity."""
    return pl.concat([
        tables.source2.select(
            pl.col("entity_id").alias("candidate_id"),
            pl.lit("S2").alias("candidate_source"),
            "business_name_normalized", "business_address_normalized", "country_normalized",
        ),
        tables.source3.select(
            pl.col("entity_id").alias("candidate_id"),
            pl.lit("S3").alias("candidate_source"),
            "business_name_normalized", "business_address_normalized", "country_normalized",
        ),
    ], how="vertical")


def exact_key_frame(frame: pl.LazyFrame, field: str, id_column: str, key_column: str) -> pl.LazyFrame:
    return (
        frame.select(pl.col(id_column).alias("entity_id"), pl.col(field).alias("key"))
        .filter(pl.col("key") != "")
        .select("entity_id", pl.col("key").alias(key_column))
    )


def name_country_key_frame(frame: pl.LazyFrame, id_column: str, country_column: str) -> pl.LazyFrame:
    return (
        frame.select(
            pl.col(id_column).alias("entity_id"),
            pl.col("business_name_normalized"),
            pl.col(country_column).alias("country_normalized"),
        )
        .filter((pl.col("business_name_normalized") != "") & (pl.col("country_normalized") != ""))
        .select(
            "entity_id",
            pl.concat_str(["business_name_normalized", "country_normalized"], separator="|").alias("key"),
        )
    )


def token_key_frame(
    frame: pl.LazyFrame,
    id_column: str,
    country_column: str,
    min_length: int,
    include_country: bool = False,
) -> pl.LazyFrame:
    tokens = (
        frame.select(
            pl.col(id_column).alias("entity_id"),
            pl.col("business_name_normalized"),
            pl.col(country_column).alias("country_normalized"),
        )
        .filter(pl.col("business_name_normalized") != "")
        # Unique within each normalized name before explode. Each entity ID
        # has one source row (Phase 1 contract), so this avoids a global
        # multi-million-row (entity_id, token) deduplication/sort.
        .with_columns(pl.col("business_name_normalized").str.split(" ").list.unique().alias("key"))
        .explode("key", empty_as_null=True)
        .filter(pl.col("key").str.len_chars() >= min_length)
    )
    if include_country:
        return tokens.select(
            "entity_id",
            pl.concat_str(["key", "country_normalized"], separator="|").alias("key"),
        )
    return tokens.select("entity_id", "key")


def address_token_key_frame(frame: pl.LazyFrame, id_column: str, min_length: int = 2) -> pl.LazyFrame:
    """Tokenize normalized addresses for conservative shared-token blocking."""
    return (
        frame.select(pl.col(id_column).alias("entity_id"), pl.col("business_address_normalized"))
        .filter(pl.col("business_address_normalized") != "")
        .with_columns(pl.col("business_address_normalized").str.split(" ").list.unique().alias("key"))
        .explode("key", empty_as_null=True)
        .filter(pl.col("key").str.len_chars() >= min_length)
        .select("entity_id", "key")
    )


def numeric_address_key_frame(frame: pl.LazyFrame, id_column: str, min_length: int = 2) -> pl.LazyFrame:
    """Extract distinct numeric address components as blocking keys."""
    return (
        frame.select(pl.col(id_column).alias("entity_id"), pl.col("business_address_normalized"))
        .filter(pl.col("business_address_normalized") != "")
        .with_columns(pl.col("business_address_normalized").str.extract_all(r"\d+").list.unique().alias("key"))
        .explode("key", empty_as_null=True)
        .filter(pl.col("key").str.len_chars() >= min_length)
        .select("entity_id", "key")
    )


def name_token_pair_key_frame(frame: pl.LazyFrame, id_column: str, min_length: int = 3) -> pl.LazyFrame:
    """Build order-invariant keys from each pair of distinct name tokens."""
    tokens = (
        frame.select(pl.col(id_column).alias("entity_id"), pl.col("business_name_normalized"))
        .filter(pl.col("business_name_normalized") != "")
        .with_columns(pl.col("business_name_normalized").str.split(" ").list.unique().alias("tokens"))
        .explode("tokens", empty_as_null=True)
        .filter(pl.col("tokens").str.len_chars() >= min_length)
        .select("entity_id", pl.col("tokens").alias("token"))
    )
    first = tokens.select("entity_id", pl.col("token").alias("token_a"))
    second = tokens.select("entity_id", pl.col("token").alias("token_b"))
    return (
        first.join(second, on="entity_id", how="inner")
        .filter(pl.col("token_a") < pl.col("token_b"))
        .select("entity_id", pl.concat_str(["token_a", "token_b"], separator="␟").alias("key"))
    )


def profile_key_blocks(
    method: str,
    left_keys: pl.LazyFrame,
    right_keys: pl.LazyFrame,
    pair_cap: int | None = None,
) -> tuple[pl.LazyFrame, dict]:
    """Return shared-key counts and summary stats; candidate estimate is n1 * nt."""
    # Materialize the compact frequency tables exactly once per side. Keeping
    # these counts in memory avoids re-running name tokenization/explode for
    # each requested statistic (which causes repeated huge temporary spills on
    # the full data). The row-level token tables remain lazy for candidate joins.
    left = left_keys.group_by("key").len().rename({"len": "s1_count"}).collect(engine="streaming")
    right = right_keys.group_by("key").len().rename({"len": "target_count"}).collect(engine="streaming")
    shared = left.join(right, on="key", how="inner").with_columns(
        (pl.col("s1_count") + pl.col("target_count")).alias("block_size"),
        (pl.col("s1_count") * pl.col("target_count")).alias("pair_estimate"),
    ).lazy()
    summary_exprs = [
        pl.len().alias("shared_key_count"),
        pl.col("block_size").mean().alias("mean_block_size"),
        pl.col("block_size").median().alias("median_block_size"),
        pl.col("block_size").quantile(0.90, interpolation="nearest").alias("p90_block_size"),
        pl.col("block_size").quantile(0.95, interpolation="nearest").alias("p95_block_size"),
        pl.col("block_size").quantile(0.99, interpolation="nearest").alias("p99_block_size"),
        pl.col("block_size").max().alias("max_block_size"),
        pl.col("pair_estimate").sum().alias("estimated_candidate_pairs_before_dedup"),
        (pl.col("block_size") > LARGE_BLOCK_RECORD_THRESHOLD).sum().alias("blocks_over_1000_records"),
        (pl.col("pair_estimate") > LARGE_BLOCK_PAIR_THRESHOLD).sum().alias("blocks_over_1m_candidate_pairs"),
    ]
    if pair_cap is not None:
        summary_exprs.extend([
            (pl.col("pair_estimate") <= pair_cap).sum().alias("eligible_shared_key_count"),
            pl.col("pair_estimate").filter(pl.col("pair_estimate") <= pair_cap).sum().alias("eligible_estimated_candidate_pairs"),
        ])
    summary = shared.select(*summary_exprs).collect(engine="streaming").row(0, named=True)
    shared_count = summary.pop("shared_key_count")
    summary["method"] = method
    summary["shared_key_count"] = shared_count
    summary["unique_key_count"] = left.height + right.height - shared_count
    summary["record_count"] = left["s1_count"].sum() + right["target_count"].sum()
    if pair_cap is not None:
        summary["pair_cap"] = pair_cap
    else:
        summary["pair_cap"] = None
        summary["eligible_shared_key_count"] = shared_count
        summary["eligible_estimated_candidate_pairs"] = summary["estimated_candidate_pairs_before_dedup"]
    top = (
        shared.sort(["pair_estimate", "key"], descending=[True, False])
        .select("key", "s1_count", "target_count", "block_size", "pair_estimate")
        .head(10).collect(engine="streaming").to_dicts()
    )
    summary["largest_shared_blocks"] = top
    return shared, summary
