"""Candidate-volume and blocking-recall evaluation."""

import polars as pl


def candidate_volume(pairs: pl.LazyFrame, source1: pl.LazyFrame) -> dict:
    counts = source1.select(pl.col("entity_id").alias("s1_id")).join(
        pairs.group_by("s1_id").agg(pl.len().alias("candidate_count")),
        on="s1_id", how="left",
    ).with_columns(pl.col("candidate_count").fill_null(0))
    stats = counts.select(
        pl.col("candidate_count").sum().alias("total_pairs"),
        pl.col("candidate_count").mean().alias("mean_per_s1"),
        pl.col("candidate_count").median().alias("median_per_s1"),
        pl.col("candidate_count").quantile(.9, interpolation="nearest").alias("p90_per_s1"),
        pl.col("candidate_count").quantile(.95, interpolation="nearest").alias("p95_per_s1"),
        pl.col("candidate_count").quantile(.99, interpolation="nearest").alias("p99_per_s1"),
        pl.col("candidate_count").max().alias("max_per_s1"),
        (pl.col("candidate_count") == 0).sum().alias("zero_candidate_s1"),
        pl.len().alias("source1_count"),
    ).collect(engine="streaming").row(0, named=True)
    stats["by_source"] = pairs.group_by("candidate_source").len().sort("candidate_source").collect(engine="streaming").to_dicts()
    method_pairs = pairs.explode("block_methods", empty_as_null=True)
    stats["by_method"] = method_pairs.group_by("block_methods").len().sort("block_methods").collect(engine="streaming").to_dicts()
    return stats


def evaluate_recall(pairs: pl.LazyFrame, labels: pl.DataFrame) -> dict:
    """Evaluate truth-pair recovery. Singleton labels are reported separately."""
    truth = labels.lazy().select(
        pl.col("source1_entity_id").alias("s1_id"),
        pl.col("matched_entity_ids").alias("candidate_id"),
    ).explode("candidate_id", empty_as_null=True).filter(pl.col("candidate_id").is_not_null())
    predicted = pairs.select("s1_id", "candidate_id").unique()
    truth_count = truth.select(pl.len()).collect(engine="streaming").item()
    recovered = truth.join(predicted, on=["s1_id", "candidate_id"], how="inner")
    recovered_total = recovered.select(pl.len()).collect(engine="streaming").item()
    by_source = {}
    for prefix in ("S2-", "S3-"):
        sub_truth = truth.filter(pl.col("candidate_id").str.starts_with(prefix))
        total = sub_truth.select(pl.len()).collect(engine="streaming").item()
        found = sub_truth.join(predicted, on=["s1_id", "candidate_id"], how="inner").select(pl.len()).collect(engine="streaming").item()
        by_source[prefix[:2]] = {"true_pairs": total, "recovered_pairs": found, "recall": found / total if total else None}

    per_s1_truth = truth.group_by("s1_id").len().rename({"len": "truth_count"})
    per_s1_found = recovered.group_by("s1_id").len().rename({"len": "recovered_count"})
    per_s1 = per_s1_truth.join(per_s1_found, on="s1_id", how="left").with_columns(pl.col("recovered_count").fill_null(0))
    recovery_counts = per_s1.select(
        pl.len().alias("positive_s1_count"),
        (pl.col("recovered_count") == pl.col("truth_count")).sum().alias("fully_recovered_s1"),
        ((pl.col("recovered_count") > 0) & (pl.col("recovered_count") < pl.col("truth_count"))).sum().alias("partially_recovered_s1"),
        (pl.col("recovered_count") == 0).sum().alias("zero_recall_s1"),
    ).collect(engine="streaming").row(0, named=True)
    singleton_count = labels.select((pl.col("matched_entity_ids").list.len() == 0).sum()).item()
    positives = recovery_counts["positive_s1_count"]
    recovery_counts["fully_recovered_s1_pct"] = recovery_counts["fully_recovered_s1"] / positives if positives else None
    recovery_counts["partially_recovered_s1_pct"] = recovery_counts["partially_recovered_s1"] / positives if positives else None
    recovery_counts["zero_recall_s1_pct"] = recovery_counts["zero_recall_s1"] / positives if positives else None
    return {
        "true_pairs": truth_count,
        "recovered_true_pairs": recovered_total,
        "candidate_recall": recovered_total / truth_count if truth_count else None,
        "by_candidate_source": by_source,
        "positive_s1_recovery": recovery_counts,
        "singleton_s1_count_excluded_from_recall": singleton_count,
    }


def contribution_by_method(pairs: pl.LazyFrame, labels: pl.DataFrame) -> dict:
    """Compute cumulative unique-pair and truth recall as methods are added."""
    order = [
        "exact_name", "name_country", "exact_address", "rare_name_token",
        "rare_name_token_pair", "address_token", "numeric_address",
    ]
    rows = []
    active = []
    for method in order:
        active.append(method)
        subset = pairs.filter(pl.col("block_methods").list.eval(pl.element().is_in(active)).list.any())
        volume = subset.select(pl.struct("s1_id", "candidate_id").n_unique()).collect(engine="streaming").item()
        recall = evaluate_recall(subset, labels)
        rows.append({"methods": list(active), "candidate_pairs": volume, "candidate_recall": recall["candidate_recall"]})
    return {"cumulative": rows}
