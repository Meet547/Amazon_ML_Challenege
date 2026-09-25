"""Command-line entry point for train evaluation and test candidate generation."""

import argparse
import json
import time
from pathlib import Path

import polars as pl

from src.common.config import PHASE1_OUTPUT, PHASE2_OUTPUT, PHASE2_PROFILE
from src.evaluation.ground_truth import load_ground_truth
from .candidate_generator import generate_candidates
from .deduplicate import write_candidate_tsv
from .evaluate import candidate_volume, contribution_by_method, evaluate_recall
from .loader import load_entity_tables


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
    generation.pairs.sink_parquet(artifact, compression="zstd", engine="streaming")
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
