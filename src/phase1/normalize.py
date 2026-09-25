"""Deterministic normalization of entity fields while retaining raw values."""

import unicodedata
import math
from typing import Any

import polars as pl

def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    # pandas/numpy missing values can reach this function in in-memory callers.
    if not isinstance(value, str):
        if isinstance(value, float) and math.isnan(value):
            return ""
        value = str(value)
    value = unicodedata.normalize("NFKC", value).casefold()
    # Keep Unicode combining marks (for example Devanagari vowel signs),
    # letters and numbers; regex \w excludes many combining marks.
    chars = []
    for char in value:
        category = unicodedata.category(char)
        if char.isspace():
            chars.append(" ")
        elif category[0] in {"L", "N", "M"} or char == "_":
            chars.append(char)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def normalize_name(value: Any) -> str:
    """Normalize case, Unicode, punctuation and spacing; retain all words."""
    return _normalize_text(value)


def normalize_address(value: Any) -> str:
    """Conservatively normalize an address, preserving letters and numbers."""
    return _normalize_text(value)


def normalize_country(value: Any) -> str:
    """Normalize country labels as open-set strings without semantic guessing."""
    return _normalize_text(value)


def normalized_frame(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Add canonical string fields to a frame without changing raw columns."""
    return frame.with_columns(
        pl.col("business_name").fill_null("").map_elements(normalize_name, return_dtype=pl.String)
        .alias("business_name_normalized"),
        pl.col("business_address").fill_null("").map_elements(normalize_address, return_dtype=pl.String)
        .alias("business_address_normalized"),
        pl.col("country").fill_null("").map_elements(normalize_country, return_dtype=pl.String)
        .alias("country_normalized"),
    )
