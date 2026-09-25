"""Lazy loader for Phase 1 Parquet entity artifacts."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.common.config import PHASE1_OUTPUT

ENTITY_COLUMNS = [
    "entity_id", "business_name", "business_address", "country",
    "business_name_normalized", "business_address_normalized", "country_normalized",
]
SOURCES = ("source1", "source2", "source3")


@dataclass(frozen=True)
class EntityTables:
    """The three lazy entity tables for one split, kept separate by source."""

    split: str
    source1: pl.LazyFrame
    source2: pl.LazyFrame
    source3: pl.LazyFrame


def load_entity_tables(
    split: str,
    data_dir: str | Path = PHASE1_OUTPUT,
) -> EntityTables:
    """Read and validate the Phase 1 entity schemas without materializing rows."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(data_dir) / split
    frames = {}
    for source in SOURCES:
        path = root / f"{split}_{source}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pl.scan_parquet(path)
        schema = frame.collect_schema()
        if schema.names() != ENTITY_COLUMNS:
            raise ValueError(f"{path}: expected columns {ENTITY_COLUMNS}, got {schema.names()}")
        non_string = {name: dtype for name, dtype in schema.items() if dtype != pl.String}
        if non_string:
            raise TypeError(f"{path}: expected all string columns, got {non_string}")
        frames[source] = frame
    return EntityTables(split, frames["source1"], frames["source2"], frames["source3"])
