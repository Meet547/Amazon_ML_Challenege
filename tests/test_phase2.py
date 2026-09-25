import polars as pl
import json

from src.phase2.candidate_generator import generate_candidates
from src.phase2.deduplicate import write_candidate_tsv
from src.phase2.evaluate import candidate_volume, contribution_by_method, evaluate_recall
from src.phase2.loader import ENTITY_COLUMNS, load_entity_tables
from src.phase2.run import run


def _entity(entity_id, name, address, country):
    return {
        "entity_id": entity_id, "business_name": name, "business_address": address,
        "country": country, "business_name_normalized": name,
        "business_address_normalized": address, "country_normalized": country,
    }


def _write_tables(root):
    (root / "train").mkdir(parents=True)
    rows = {
        "source1": [
            _entity("S1-1", "acme group", "12 main road", "us"),
            _entity("S1-2", "", "", "us"),
        ],
        "source2": [
            _entity("S2-1", "acme group", "different", "us"),
            _entity("S2-2", "other", "12 main road", "ca"),
            _entity("S2-3", "", "", "ca"),
        ],
        "source3": [_entity("S3-1", "acme group", "elsewhere", "ca")],
    }
    for source, values in rows.items():
        pl.DataFrame(values, schema=ENTITY_COLUMNS).write_parquet(root / "train" / f"train_{source}.parquet")
    return rows


def test_loader_validates_phase1_contract_and_preserves_sources(tmp_path):
    expected = _write_tables(tmp_path / "normalized")
    loaded = load_entity_tables("train", tmp_path / "normalized")
    assert loaded.source1.collect()["entity_id"].to_list() == [r["entity_id"] for r in expected["source1"]]
    assert loaded.source2.collect_schema().names() == ENTITY_COLUMNS
    assert loaded.source3.collect()["entity_id"].to_list() == ["S3-1"]


def test_blocking_deduplicates_and_preserves_provenance_without_s1_candidates(tmp_path):
    _write_tables(tmp_path / "normalized")
    tables = load_entity_tables("train", tmp_path / "normalized")
    generation = generate_candidates(tables, token_pair_cap=1)
    pairs = generation.pairs.sort("s1_id", "candidate_id").collect()
    assert pairs.select("s1_id", "candidate_id").n_unique() == pairs.height
    assert set(pairs["candidate_source"].to_list()) == {"S2", "S3"}
    assert all(not value.startswith("S1-") for value in pairs["candidate_id"].to_list())
    acme = pairs.filter(pl.col("s1_id") == "S1-1")
    assert set(acme["candidate_id"].to_list()) == {"S2-1", "S2-2", "S3-1"}
    assert acme.filter(pl.col("candidate_id") == "S2-1")["block_methods"].to_list()[0] == ["exact_name", "name_country"]
    assert "exact_address" in pairs.filter(pl.col("candidate_id") == "S2-2")["block_methods"].to_list()[0]
    assert pairs.filter(pl.col("s1_id") == "S1-2").is_empty()


def test_candidate_tsv_contract_and_volume_includes_zero_candidates(tmp_path):
    _write_tables(tmp_path / "normalized")
    tables = load_entity_tables("train", tmp_path / "normalized")
    pairs = generate_candidates(tables).pairs
    output = write_candidate_tsv(pairs, tables.source1, tmp_path / "candidate_pairs.tsv")
    result = pl.read_csv(output, separator="\t")
    assert result.columns == ["source1_entity_id", "candidate_entity_ids"]
    assert result.height == 2
    assert result.filter(pl.col("source1_entity_id") == "S1-2")["candidate_entity_ids"].item() == ""
    volume = candidate_volume(pairs, tables.source1)
    assert volume["total_pairs"] == 3
    assert volume["zero_candidate_s1"] == 1


def test_ground_truth_recall_excludes_singletons_and_reports_partial_recovery(tmp_path):
    _write_tables(tmp_path / "normalized")
    tables = load_entity_tables("train", tmp_path / "normalized")
    pairs = generate_candidates(tables).pairs
    labels = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "matched_entity_ids": [["S2-1", "S3-1", "S2-999"], []],
    })
    result = evaluate_recall(pairs, labels)
    assert result["candidate_recall"] == 2 / 3
    assert result["by_candidate_source"]["S2"]["recovered_pairs"] == 1
    assert result["by_candidate_source"]["S3"]["recovered_pairs"] == 1
    assert result["positive_s1_recovery"]["partially_recovered_s1"] == 1
    assert result["singleton_s1_count_excluded_from_recall"] == 1
    stages = contribution_by_method(pairs, labels)["cumulative"]
    assert len(stages) == 4
    assert stages[0]["candidate_recall"] == 2 / 3
    assert stages[1]["candidate_recall"] == 2 / 3


def test_test_run_never_needs_ground_truth_and_emits_metrics(tmp_path):
    rows = _write_tables(tmp_path / "normalized")
    (tmp_path / "normalized" / "test").mkdir()
    for source in rows:
        pl.read_parquet(tmp_path / "normalized" / "train" / f"train_{source}.parquet").write_parquet(
            tmp_path / "normalized" / "test" / f"test_{source}.parquet"
        )
    result = run(
        "test", phase1_dir=tmp_path / "normalized", candidates_dir=tmp_path / "candidates",
        metrics_path=tmp_path / "metrics.json", ground_truth_path=tmp_path / "does-not-exist.tsv",
        submission_path=tmp_path / "output" / "candidate_pairs.tsv",
    )
    assert "recall" not in result
    assert json.loads((tmp_path / "metrics.json").read_text())["split"] == "test"
    assert (tmp_path / "output" / "candidate_pairs.tsv").is_file()
