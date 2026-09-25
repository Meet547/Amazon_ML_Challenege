"""Stream raw train/test TSV files into normalized Phase 1 Parquet artifacts."""

import argparse
import logging
import time
from pathlib import Path

import polars as pl

from .normalize import normalized_frame

LOGGER = logging.getLogger("phase1")
REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
SOURCES = ("source1", "source2", "source3")


def _scan(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, separator="\t", has_header=True, infer_schema=False,
                       null_values=[""], encoding="utf8-lossy")


def _profile(frame: pl.LazyFrame) -> dict:
    expressions = [pl.len().alias("row_count")]
    for column in REQUIRED_COLUMNS:
        expressions.extend([
            pl.col(column).null_count().alias(f"{column}_missing"),
            pl.col(column).n_unique().alias(f"{column}_unique"),
        ])
    for column in ("business_name", "business_address"):
        lengths = pl.col(column).str.len_chars()
        expressions.extend([
            lengths.mean().alias(f"{column}_avg_length"),
            lengths.median().alias(f"{column}_median_length"),
            lengths.min().alias(f"{column}_min_length"),
            lengths.max().alias(f"{column}_max_length"),
        ])
    return frame.select(expressions).collect(engine="streaming").row(0, named=True)


def _validate(frame: pl.LazyFrame, source_number: int) -> dict:
    expected = f"S{source_number}-"
    ids = pl.col("entity_id")
    basic = frame.select(
        ids.is_null().sum().alias("null_ids"),
        (ids.is_not_null() & (ids.str.strip_chars() == "")).sum().alias("blank_ids"),
        (ids.is_not_null() & ~ids.str.starts_with(expected)).sum().alias("wrong_prefix"),
    ).collect(engine="streaming").row(0, named=True)
    duplicates = (frame.group_by("entity_id").len()
                  .filter((pl.col("entity_id").is_not_null()) & (pl.col("len") > 1))
                  .select(pl.len().alias("duplicate_id_count"))
                  .collect(engine="streaming").item())
    basic["duplicate_id_count"] = duplicates
    return basic


def process_file(input_path: str | Path, output_path: str | Path,
                 source_number: int) -> dict:
    """Validate, profile, normalize, and write one source as TSV."""
    input_path, output_path = Path(input_path), Path(output_path)
    frame = _scan(input_path)
    columns = frame.collect_schema().names()
    if columns != REQUIRED_COLUMNS:
        raise ValueError(f"{input_path}: expected columns {REQUIRED_COLUMNS}, got {columns}")

    started = time.perf_counter()
    validation = _validate(frame, source_number)
    if any(validation[key] for key in ("null_ids", "blank_ids", "wrong_prefix", "duplicate_id_count")):
        raise ValueError(f"{input_path}: ID validation failed: {validation}")
    profile = _profile(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_frame(frame).sink_parquet(output_path, compression="zstd", engine="streaming")
    profile.update(validation)
    profile["runtime_seconds"] = round(time.perf_counter() - started, 2)
    profile["output_rows"] = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    if profile["output_rows"] != profile["row_count"]:
        raise RuntimeError(f"Unexpected row count change for {input_path}: {profile}")
    return profile


def run(data_dir: str | Path = "datasets", output_dir: str | Path = "outputs/normalized",
        splits: tuple[str, ...] = ("train", "test")) -> dict:
    results = {}
    for split in splits:
        for source_number, source in enumerate(SOURCES, start=1):
            path = Path(data_dir) / split / f"{split}_{source}.tsv"
            if not path.is_file():
                raise FileNotFoundError(path)
            output = Path(output_dir) / split / f"{split}_{source}.parquet"
            summary = process_file(path, output, source_number)
            results[f"{split}/{source}"] = summary
            LOGGER.info("%s rows=%s missing(name/address/country)=%s/%s/%s unique(id/name/address/country)=%s/%s/%s/%s name_length(avg/median/min/max)=%s/%s/%s/%s address_length=%s/%s/%s/%s duplicates=%s runtime=%.2fs output=%s",
                        f"{split}/{source}", summary["row_count"],
                        summary["business_name_missing"], summary["business_address_missing"],
                        summary["country_missing"], summary["entity_id_unique"],
                        summary["business_name_unique"], summary["business_address_unique"],
                        summary["country_unique"], summary["business_name_avg_length"],
                        summary["business_name_median_length"], summary["business_name_min_length"],
                        summary["business_name_max_length"], summary["business_address_avg_length"],
                        summary["business_address_median_length"], summary["business_address_min_length"],
                        summary["business_address_max_length"], summary["duplicate_id_count"],
                        summary["runtime_seconds"], output)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--output-dir", default="outputs/normalized")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.data_dir, args.output_dir,
        ("train", "test") if args.split == "all" else (args.split,))


if __name__ == "__main__":
    main()
