import polars as pl
import pytest
import json

from src.evaluation.ground_truth import GroundTruthValidationError, load_ground_truth
from src.phase1.normalize import (
    normalize_address, normalize_country, normalize_name, normalized_frame,
)
from src.phase1.run import process_file, run


def test_name_normalization_is_conservative_and_idempotent():
    value = normalize_name("ABC   Technologies, Pvt. Ltd.")
    assert value == "abc technologies pvt ltd"
    assert normalize_name(value) == value
    assert "india" in normalize_name("ABC Technologies India")


def test_missing_and_unicode_normalize_deterministically():
    assert normalize_name(None) == ""
    assert normalize_name(float("nan")) == ""
    assert normalize_name("Cafe\u0301") == normalize_name("Café") == "café"
    assert normalize_name("मॉडर्न फाइनेंस") == "मॉडर्न फाइनेंस"


def test_address_preserves_numbers_and_country_is_open_set():
    assert normalize_address("Plot 42, Sector 17") == "plot 42 sector 17"
    assert normalize_country("Hauts-de-France") == "hauts de france"


def test_frame_preserves_raw_values_and_duplicate_business_rows():
    raw = pl.DataFrame({
        "entity_id": ["S1-001", "S1-002"],
        "business_name": ["ABC Technologies", "ABC Technologies"],
        "business_address": ["Plot 42", "Plot 42"],
        "country": ["India", "India"],
    }).lazy()
    result = normalized_frame(raw).collect()
    assert result.height == 2
    assert result["entity_id"].to_list() == ["S1-001", "S1-002"]
    assert result["business_name"].to_list() == ["ABC Technologies"] * 2
    assert result["business_name_normalized"].to_list() == ["abc technologies"] * 2


def test_process_file_detects_bad_ids_and_writes_phase2_fields(tmp_path):
    source = tmp_path / "train_source1.tsv"
    output = tmp_path / "normalized.parquet"
    source.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tABC Technologies\tPlot 42, Sector 17\tIndia\n"
        "S1-002\t\t\tFrance\n", encoding="utf-8")
    stats = process_file(source, output, 1)
    result = pl.read_parquet(output)
    assert stats["row_count"] == stats["output_rows"] == 2
    assert result.columns == [
        "entity_id", "business_name", "business_address", "country",
        "business_name_normalized", "business_address_normalized", "country_normalized",
    ]
    assert result["entity_id"].to_list() == ["S1-001", "S1-002"]
    assert result["business_address_normalized"][0] == "plot 42 sector 17"
    assert result["business_name_normalized"][1] == ""
    assert result["country_normalized"][1] == "france"

    source.write_text(source.read_text(encoding="utf-8").replace("S1-002", "S2-002"), encoding="utf-8")
    with pytest.raises(ValueError, match="ID validation failed"):
        process_file(source, output, 1)


@pytest.mark.parametrize("header,invalid_row", [
    ("entity_id\tbusiness_name\tbusiness_address\n", "S1-1\tA\tB\n"),
    ("entity_id\tbusiness_name\tbusiness_address\tcountry\textra\n", "S1-1\tA\tB\tUS\tx\n"),
    ("business_name\tentity_id\tbusiness_address\tcountry\n", "A\tS1-1\tB\tUS\n"),
])
def test_process_file_rejects_missing_unexpected_or_reordered_columns(tmp_path, header, invalid_row):
    source = tmp_path / "input.tsv"
    source.write_text(header + invalid_row, encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        process_file(source, tmp_path / "out.parquet", 1)


@pytest.mark.parametrize("rows,expected", [
    (["\tAcme\t1 Main St\tUS"], "null_ids"),
    (["   \tAcme\t1 Main St\tUS"], "blank_ids"),
    (["S1-1\tAcme\t1 Main St\tUS", "S1-1\tOther\t2 Main St\tUS"], "duplicate_id_count"),
    (["S2-1\tAcme\t1 Main St\tUS"], "wrong_prefix"),
])
def test_process_file_rejects_invalid_ids_with_validation_detail(tmp_path, rows, expected):
    source = tmp_path / "input.tsv"
    source.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" +
                      "\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        process_file(source, tmp_path / "out.parquet", 1)


def test_process_file_rejects_malformed_tsv_row(tmp_path):
    source = tmp_path / "input.tsv"
    source.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                      "S1-1\tAcme\t1 Main St\tUS\textra\n", encoding="utf-8")
    with pytest.raises(pl.exceptions.ComputeError):
        process_file(source, tmp_path / "out.parquet", 1)


def _write_train_entities(root):
    root.mkdir(parents=True)
    values = {
        "source1": [("S1-1", "Acme", "1 Main St", "US"),
                    ("S1-2", "   ", "", "France")],
        "source2": [("S2-1", "Acme", "1 Main St", "US")],
        "source3": [("S3-1", "Other", "2 Main St", "India")],
    }
    for source, rows in values.items():
        text = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        text += "".join("\t".join(row) + "\n" for row in rows)
        (root / f"train_{source}.tsv").write_text(text, encoding="utf-8")


def test_run_persists_deterministic_profile_and_complete_parquet_artifacts(tmp_path):
    data = tmp_path / "datasets" / "train"
    _write_train_entities(data)
    outputs = tmp_path / "outputs" / "normalized"
    profile_path = tmp_path / "outputs" / "metrics" / "phase1_profile.json"

    run(tmp_path / "datasets", outputs, ("train",), profile_path)
    first_profile_bytes = profile_path.read_bytes()
    profile = json.loads(first_profile_bytes)
    assert profile["schema_version"] == 1
    assert [item["dataset"] for item in profile["datasets"]] == [
        "train_source1", "train_source2", "train_source3",
    ]
    source1_profile = profile["datasets"][0]
    assert source1_profile["row_count"] == source1_profile["unique_entity_id_count"] == 2
    assert source1_profile["blank_string_counts"]["business_name"] == 1
    assert source1_profile["country_distribution"] == [
        {"country": "France", "len": 1}, {"country": "US", "len": 1},
    ]
    assert source1_profile["source_prefix_distribution"] == [
        {"source_prefix": "S1-", "len": 2},
    ]
    assert source1_profile["duplicate_info"] == {"entity_id_count": 0, "full_row_count": 0}

    artifact = outputs / "train" / "train_source1.parquet"
    result = pl.read_parquet(artifact)
    assert result.columns == [
        "entity_id", "business_name", "business_address", "country",
        "business_name_normalized", "business_address_normalized", "country_normalized",
    ]
    assert result["entity_id"].to_list() == ["S1-1", "S1-2"]
    assert result["business_name"].to_list() == ["Acme", "   "]
    assert result["business_name_normalized"].to_list() == ["acme", ""]
    assert result.height == source1_profile["row_count"]

    run(tmp_path / "datasets", outputs, ("train",), profile_path)
    assert profile_path.read_bytes() == first_profile_bytes


def test_ground_truth_loader_preserves_empty_singleton_labels(tmp_path):
    train = tmp_path / "train"
    _write_train_entities(train)
    gt = train / "train_ground_truth.tsv"
    gt.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n",
                  encoding="utf-8")
    labels = load_ground_truth(gt, train / "train_source1.tsv",
                               train / "train_source2.tsv", train / "train_source3.tsv")
    assert labels.columns == ["source1_entity_id", "matched_entity_ids"]
    assert labels["matched_entity_ids"].to_list() == [["S2-1", "S3-1"], []]


@pytest.mark.parametrize("target,issue", [
    ("S4-1", "invalid_target_prefix"),
    ("S2-missing", "unknown_s2_references"),
    ("S3-missing", "unknown_s3_references"),
    ("S3-1 ", "target_whitespace"),
    ("S2-1,S2-1", "duplicate_source1_target_pairs"),
    ("S2-1,,S3-1", "malformed_target_lists"),
])
def test_ground_truth_loader_rejects_malformed_or_invalid_targets(tmp_path, target, issue):
    train = tmp_path / "train"
    _write_train_entities(train)
    gt = train / "train_ground_truth.tsv"
    gt.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\t" + target + "\n",
                  encoding="utf-8")
    with pytest.raises(GroundTruthValidationError, match=issue):
        load_ground_truth(gt, train / "train_source1.tsv",
                          train / "train_source2.tsv", train / "train_source3.tsv")


def test_ground_truth_loader_rejects_unknown_source1_and_duplicate_rows(tmp_path):
    train = tmp_path / "train"
    _write_train_entities(train)
    gt = train / "train_ground_truth.tsv"
    gt.write_text("source1_entity_id\tmatched_entity_ids\nS1-missing\t\nS1-1\t\nS1-1\t\n",
                  encoding="utf-8")
    with pytest.raises(GroundTruthValidationError, match="unknown_source1_references"):
        load_ground_truth(gt, train / "train_source1.tsv",
                          train / "train_source2.tsv", train / "train_source3.tsv")


def test_ground_truth_loader_rejects_source1_entities_missing_labels(tmp_path):
    train = tmp_path / "train"
    _write_train_entities(train)
    gt = train / "train_ground_truth.tsv"
    gt.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\t\n", encoding="utf-8")
    with pytest.raises(GroundTruthValidationError, match="missing_source1_labels"):
        load_ground_truth(gt, train / "train_source1.tsv",
                          train / "train_source2.tsv", train / "train_source3.tsv")


def test_ground_truth_loader_rejects_bad_schema_and_whitespace(tmp_path):
    train = tmp_path / "train"
    _write_train_entities(train)
    gt = train / "train_ground_truth.tsv"
    gt.write_text("source1_entity_id\tother\nS1-1\t\n", encoding="utf-8")
    with pytest.raises(GroundTruthValidationError, match="expected columns"):
        load_ground_truth(gt, train / "train_source1.tsv",
                          train / "train_source2.tsv", train / "train_source3.tsv")
    gt.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\t S2-1\n", encoding="utf-8")
    with pytest.raises(GroundTruthValidationError, match="target_whitespace"):
        load_ground_truth(gt, train / "train_source1.tsv",
                          train / "train_source2.tsv", train / "train_source3.tsv")


def test_ground_truth_loader_rejects_null_and_invalid_source1_ids(tmp_path):
    train = tmp_path / "train"
    _write_train_entities(train)
    gt = train / "train_ground_truth.tsv"
    gt.write_text("source1_entity_id\tmatched_entity_ids\n\t\nS2-1\t\n",
                  encoding="utf-8")
    with pytest.raises(GroundTruthValidationError, match="null_source1_entity_id"):
        load_ground_truth(gt, train / "train_source1.tsv",
                          train / "train_source2.tsv", train / "train_source3.tsv")
    gt.write_text("source1_entity_id\tmatched_entity_ids\nS2-1\t\n", encoding="utf-8")
    with pytest.raises(GroundTruthValidationError, match="invalid_source1_prefix"):
        load_ground_truth(gt, train / "train_source1.tsv",
                          train / "train_source2.tsv", train / "train_source3.tsv")


def test_phase2_handoff_contract_requires_entity_identity_and_canonical_fields():
    # Phase 2 is currently documentation-only; these fields cover the documented
    # candidate interface's S1/S2/S3 entity references and matching attributes.
    source = pl.DataFrame({"entity_id": ["S1-001"], "business_name": ["Acme"],
                          "business_address": ["1 Main St"], "country": ["US"]}).lazy()
    output = normalized_frame(source).collect()
    assert {"entity_id", "business_name_normalized", "business_address_normalized",
            "country_normalized"}.issubset(output.columns)
    assert output["entity_id"][0] == "S1-001"
