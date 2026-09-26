"""Command-line entry point for train evaluation and test candidate generation."""

import argparse
import json
import shutil
import time
from pathlib import Path

import duckdb
import polars as pl
import pyarrow.parquet as pq

from src.common.config import PHASE1_OUTPUT, PHASE2_OUTPUT, PHASE2_PROFILE
from src.evaluation.ground_truth import load_ground_truth
from .candidate_generator import generate_candidates
from .deduplicate import write_candidate_tsv
from .evaluate import candidate_volume, contribution_by_method, evaluate_recall
from .loader import load_entity_tables


def _merge_block_files(block_paths: list[Path], output_path: Path, work_dir: Path) -> None:
    """Hash-partition method rows, deduplicate each partition, and stream merge."""
    con = duckdb.connect(database=":memory:")
    writer = None
    try:
        con.execute("SET memory_limit = '2GB'")
        con.execute("SET threads = 2")
        con.execute("SET preserve_insertion_order = false")
        con.execute(f"SET temp_directory = '{(work_dir / 'duckdb_temp').as_posix()}'")
        con.execute("SET max_temp_directory_size = '2GB'")
        source_files_sql = "[" + ",".join(
            "'" + path.as_posix().replace("'", "''") + "'" for path in block_paths
        ) + "]"
        partitions_dir = work_dir / "raw_partitions"
        partitions_dir.mkdir()
        partitions_path = partitions_dir.as_posix().replace("'", "''")
        con.execute(f"""
            COPY (
                SELECT *, hash(s1_id, candidate_id, candidate_source) % 64 AS _bucket
                FROM read_parquet({source_files_sql})
            ) TO '{partitions_path}'
            (FORMAT PARQUET, PARTITION_BY (_bucket), COMPRESSION ZSTD)
        """)
        deduplicated_dir = work_dir / "deduplicated_partitions"
        deduplicated_dir.mkdir()
        partition_files = sorted(partitions_dir.rglob("*.parquet"))
        if not partition_files:
            raise ValueError("candidate generation produced no pairs")
        keys_sql = "s1_id, candidate_id, candidate_source"
        for index, partition_file in enumerate(partition_files):
            source_path = partition_file.as_posix().replace("'", "''")
            target_path = (deduplicated_dir / f"part-{index:03d}.parquet").as_posix().replace("'", "''")
            con.execute(f"""
                COPY (
                    SELECT {keys_sql},
                           list_sort(list_distinct(flatten(list(block_methods)))) AS block_methods
                    FROM read_parquet('{source_path}')
                    GROUP BY {keys_sql}
                ) TO '{target_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        for partition_file in sorted(deduplicated_dir.glob("*.parquet")):
            for batch in pq.ParquetFile(partition_file).iter_batches(batch_size=250_000):
                if writer is None:
                    writer = pq.ParquetWriter(output_path, batch.schema, compression="zstd")
                writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()
        con.close()


def run(split: str, phase1_dir: str | Path = PHASE1_OUTPUT,
        candidates_dir: str | Path = PHASE2_OUTPUT,
        metrics_path: str | Path | None = None,
        ground_truth_path: str | Path = "datasets/train/train_ground_truth.tsv",
        raw_data_dir: str | Path = "datasets/train",
        submission_path: str | Path = "output/candidate_pairs.tsv") -> dict:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    started = time.perf_counter()
    tables = load_entity_tables(split, phase1_dir)
    generation = generate_candidates(tables)
    artifact = Path(candidates_dir) / split / "candidate_pairs.parquet"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    work_dir = artifact.parent / f".{artifact.stem}_blocks"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    block_paths = []
    try:
        for method, block_pairs in generation.raw_pairs_by_method.items():
            block_path = work_dir / f"{method}.parquet"
            block_pairs.sink_parquet(block_path, compression="zstd", engine="streaming")
            block_paths.append(block_path)
        staged_artifact = work_dir / "candidate_pairs.parquet"
        _merge_block_files(block_paths, staged_artifact, work_dir)
        staged_artifact.replace(artifact)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    volume = candidate_volume(pl.scan_parquet(artifact), tables.source1)
    metrics = {
        "schema_version": 1,
        "split": split,
        "blocking_configuration": generation.configuration,
        "blocking_statistics": generation.block_statistics,
        "candidate_volume": volume,
        "candidate_pairs_parquet": str(artifact),
    }
    if split == "train":
        labels = load_ground_truth(
            Path(ground_truth_path), Path(raw_data_dir) / "train_source1.tsv",
            Path(raw_data_dir) / "train_source2.tsv", Path(raw_data_dir) / "train_source3.tsv",
        )
        metrics["recall"] = evaluate_recall(pl.scan_parquet(artifact), labels)
        metrics["contribution"] = contribution_by_method(pl.scan_parquet(artifact), labels)
    else:
        write_candidate_tsv(pl.scan_parquet(artifact), tables.source1, submission_path)
        metrics["submission_tsv"] = str(submission_path)
    metrics["runtime_seconds"] = round(time.perf_counter() - started, 3)
    metrics_path = Path(metrics_path or f"{Path(PHASE2_PROFILE).with_suffix('').as_posix()}_{split}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--phase1-dir", default=PHASE1_OUTPUT)
    parser.add_argument("--candidates-dir", default=PHASE2_OUTPUT)
    parser.add_argument("--metrics-path")
    parser.add_argument("--ground-truth", default="datasets/train/train_ground_truth.tsv")
    parser.add_argument("--raw-data-dir", default="datasets/train")
    parser.add_argument("--submission", default="output/candidate_pairs.tsv")
    args = parser.parse_args()
    result = run(args.split, args.phase1_dir, args.candidates_dir, args.metrics_path,
                 args.ground_truth, args.raw_data_dir, args.submission)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
