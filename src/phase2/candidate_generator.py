"""Candidate generation from indexed Phase 1 normalized fields."""

from dataclasses import dataclass

import polars as pl

from .blocks import (
    exact_key_frame,
    name_country_key_frame,
    profile_key_blocks,
    source2_and_source3,
    token_key_frame,
)
from .config import NAME_TOKEN_MAX_PAIR_ESTIMATE, NAME_TOKEN_MIN_LENGTH
from .deduplicate import deduplicate_with_provenance
from .loader import EntityTables

PAIR_COLUMNS = ["s1_id", "candidate_id", "candidate_source", "block_methods"]
METHODS = ("exact_name", "name_country", "exact_address", "rare_name_token")


@dataclass(frozen=True)
class CandidateGeneration:
    pairs: pl.LazyFrame
    block_statistics: dict[str, dict]
    strategy_counts: dict[str, int]
    configuration: dict


def _source1(tables: EntityTables) -> pl.LazyFrame:
    return tables.source1.select(
        pl.col("entity_id").alias("s1_id"),
        "business_name_normalized", "business_address_normalized", "country_normalized",
    )


def _name_pairs(s1: pl.LazyFrame, targets: pl.LazyFrame) -> pl.LazyFrame:
    left = s1.select(
        "s1_id", pl.col("business_name_normalized").alias("key"),
        pl.col("country_normalized").alias("s1_country"),
    ).filter(pl.col("key") != "")
    right = targets.select(
        "candidate_id", "candidate_source",
        pl.col("business_name_normalized").alias("key"),
        pl.col("country_normalized").alias("candidate_country"),
    ).filter(pl.col("key") != "")
    return left.join(right, on="key", how="inner").select(
        "s1_id", "candidate_id", "candidate_source",
        pl.when(
            (pl.col("s1_country") != "")
            & (pl.col("candidate_country") != "")
            & (pl.col("s1_country") == pl.col("candidate_country"))
        )
        .then(pl.lit(["exact_name", "name_country"], dtype=pl.List(pl.String)))
        .otherwise(pl.lit(["exact_name"], dtype=pl.List(pl.String)))
        .alias("block_methods"),
    )


def _address_pairs(s1: pl.LazyFrame, targets: pl.LazyFrame) -> pl.LazyFrame:
    left = s1.select(pl.col("s1_id"), pl.col("business_address_normalized").alias("key")).filter(pl.col("key") != "")
    right = targets.select(
        "candidate_id", "candidate_source", pl.col("business_address_normalized").alias("key")
    ).filter(pl.col("key") != "")
    return left.join(right, on="key", how="inner").select(
        "s1_id", "candidate_id", "candidate_source",
        pl.lit(["exact_address"], dtype=pl.List(pl.String)).alias("block_methods"),
    )


def _rare_token_pairs(
    s1: pl.LazyFrame,
    targets: pl.LazyFrame,
    left_tokens: pl.LazyFrame,
    right_tokens: pl.LazyFrame,
    shared_tokens: pl.LazyFrame,
    pair_cap: int,
) -> pl.LazyFrame:
    eligible = shared_tokens.filter(pl.col("pair_estimate") <= pair_cap).select("key")
    left = left_tokens.select(pl.col("entity_id").alias("s1_id"), "key").join(
        eligible, on="key", how="inner"
    )
    right = right_tokens.select(
        pl.col("entity_id").alias("candidate_id"), "candidate_source", "key",
    ).join(eligible, on="key", how="inner")
    return left.join(right, on="key", how="inner").select(
        "s1_id", "candidate_id", "candidate_source",
        pl.lit(["rare_name_token"], dtype=pl.List(pl.String)).alias("block_methods"),
    )


def generate_candidates(
    tables: EntityTables,
    token_min_length: int = NAME_TOKEN_MIN_LENGTH,
    token_pair_cap: int = NAME_TOKEN_MAX_PAIR_ESTIMATE,
) -> CandidateGeneration:
    """Build unique S1-to-S2/S3 pairs and preserve every contributing method."""
    if token_min_length < 1 or token_pair_cap < 1:
        raise ValueError("token length and pair cap must be positive")
    s1 = _source1(tables)
    targets = source2_and_source3(tables).with_columns(
        pl.col("candidate_id").str.slice(0, 2).alias("candidate_source")
    )

    blocks = {}
    name_left = exact_key_frame(s1, "business_name_normalized", "s1_id", "key")
    name_right = exact_key_frame(targets, "business_name_normalized", "candidate_id", "key")
    _, blocks["exact_name"] = profile_key_blocks("exact_name", name_left, name_right)

    country_left = name_country_key_frame(s1, "s1_id", "country_normalized")
    country_right = name_country_key_frame(targets, "candidate_id", "country_normalized")
    _, blocks["name_country"] = profile_key_blocks("name_country", country_left, country_right)

    address_left = exact_key_frame(s1, "business_address_normalized", "s1_id", "key")
    address_right = exact_key_frame(targets, "business_address_normalized", "candidate_id", "key")
    _, blocks["exact_address"] = profile_key_blocks("exact_address", address_left, address_right)

    left_tokens = token_key_frame(s1, "s1_id", "country_normalized", token_min_length)
    right_tokens = token_key_frame(targets, "candidate_id", "country_normalized", token_min_length).join(
        targets.select(pl.col("candidate_id").alias("entity_id"), "candidate_source"),
        on="entity_id", how="inner",
    )
    shared_tokens, blocks["rare_name_token"] = profile_key_blocks(
        "rare_name_token", left_tokens, right_tokens, pair_cap=token_pair_cap,
    )

    # Country-exact name is a strict subset of exact-name pairs. It is recorded
    # as provenance on those pairs, avoiding a redundant second large join.
    raw_pairs = pl.concat([
        _name_pairs(s1, targets),
        _address_pairs(s1, targets),
        _rare_token_pairs(s1, targets, left_tokens, right_tokens, shared_tokens, token_pair_cap),
    ], how="vertical")
    pairs = deduplicate_with_provenance(raw_pairs).select(PAIR_COLUMNS)

    # Counts are emitted by the execution stage after the unique-pair artifact
    # is written; these are key-count diagnostics that guide the strategy.
    strategy_counts = {}
    for method in METHODS:
        strategy_counts[method] = blocks[method]["eligible_estimated_candidate_pairs"]
    configuration = {
        "name_token_min_length": token_min_length,
        "name_token_max_pair_estimate": token_pair_cap,
        "exact_name_country": "provenance subset of exact_name; no redundant candidate join",
    }
    return CandidateGeneration(pairs, blocks, strategy_counts, configuration)
