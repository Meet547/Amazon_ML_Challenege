import polars as pl
import pytest

from src.phase1.normalize import (
    normalize_address, normalize_country, normalize_name, normalized_frame,
)
from src.phase1.run import process_file


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


def test_phase2_handoff_contract_requires_entity_identity_and_canonical_fields():
    # Phase 2 is currently documentation-only; these fields cover the documented
    # candidate interface's S1/S2/S3 entity references and matching attributes.
    source = pl.DataFrame({"entity_id": ["S1-001"], "business_name": ["Acme"],
                          "business_address": ["1 Main St"], "country": ["US"]}).lazy()
    output = normalized_frame(source).collect()
    assert {"entity_id", "business_name_normalized", "business_address_normalized",
            "country_normalized"}.issubset(output.columns)
    assert output["entity_id"][0] == "S1-001"
