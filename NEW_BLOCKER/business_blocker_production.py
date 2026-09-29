#!/usr/bin/env python3
"""
Production-grade multi-stage blocking / candidate generation for the
Amazon ML Challenge 2026 business entity-resolution task.

Designed for ~5M rows per source (S1/S2/S3) on a local SSD.

Key properties
--------------
* DuckDB performs the heavy lifting: vectorized TSV ingestion, normalization,
  grouping, hashing/joins, external sort and disk spilling.
* No giant Python dictionaries of 15M rows.
* S2/S3 are materialized once into a single target table; all blocking passes
  then operate against that table.
* Candidate pairs are stored as compact numeric row-id pairs internally.
* Block sizes are capped by frequency, so pathological common names/tokens do
  not create quadratic candidate explosions.
* Rare-token, token-pair and character-anchor blocks catch spelling/noise
  variations that exact-name/address blocks miss.
* An optional sorted-neighborhood rescue is applied only to unresolved S1 rows.
* Ground-truth evaluation can be run in the same process, with a hard recall
  gate so a weak blocker can fail the run instead of silently shipping.

Output
------
The default output is challenge-compatible candidate_pairs.tsv:
    source1_entity_id    candidate_entity_ids

A one-row-per-S1 candidate list is written, including singleton S1 rows with an
empty candidate list. No Source-1 IDs are emitted on the candidate side.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "DuckDB is required. Install with: python -m pip install 'duckdb>=1.4,<2'"
    ) from exc

try:
    import unidecode as _unidecode_mod
except ImportError:
    _unidecode_mod = None


# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# These are deliberately conservative. We prefer decomposing a large block
# into stronger composite blocks rather than expanding a huge common-value
# block.
DEFAULTS = {
    "exact_name_df": 75,
    "exact_core_df": 75,
    "exact_address_df": 40,
    "name_postal_df": 100,
    "house_name_df": 75,
    "house_address_df": 75,
    "rare_token_df": 50,
    "rare_pair_df": 75,
    "rare_gram_df": 50,
    "rescue_df": 150,
    "sorted_window": 8,
    "max_gram_slots": 3,
    "batch_s1": 50_000,
}

LEGAL_SUFFIX_RE = r"\b(?:private limited|limited|corporation|incorporated|llc|llp|plc|pllc)$"


def now() -> float:
    return time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def physical_memory_gb() -> float:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)
        except Exception:
            pass
    return 16.0


def default_memory_limit() -> str:
    # Keep substantial RAM available for the OS/file cache and Python.
    gb = max(2.0, min(12.0, physical_memory_gb() * 0.55))
    # DuckDB accepts decimal quantities such as 8GB.
    return f"{gb:.1f}GB"


@dataclass
class PassResult:
    name: str
    pair_count: int
    elapsed_sec: float


class Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = now()

    def done(self) -> float:
        elapsed = now() - self.t0
        log(f"{self.label}: {elapsed:,.2f}s")
        return elapsed


# ---------------------------------------------------------------------------
# SQL expressions: normalization is vectorized in DuckDB.
# ---------------------------------------------------------------------------


def make_name_norm_expr(col: str) -> str:
    """Unicode-preserving canonical name normalizer."""
    x = f"coalesce({col}, '')"
    x = f"lower(strip_accents({x}))"
    x = f"replace({x}, '&', ' and ')"
    # Keep Unicode letters/numbers; remove punctuation and separators.
    x = f"regexp_replace({x}, '[^\\p{{L}}\\p{{N}}]+', ' ', 'g')"
    x = f"trim(regexp_replace({x}, '\\s+', ' ', 'g'))"
    # Normalize common legal abbreviations without globally mapping 'co'.
    for short, long in [
        ("pvt", "private"),
        ("ltd", "limited"),
        ("corp", "corporation"),
        ("inc", "incorporated"),
    ]:
        x = f"regexp_replace({x}, '\\b{short}\\b', '{long}', 'g')"
    x = f"trim(regexp_replace({x}, '\\s+', ' ', 'g'))"
    return x


def make_address_norm_expr(col: str) -> str:
    """Fast address canonicalizer; intentionally retains numeric content."""
    x = f"coalesce({col}, '')"
    x = f"lower(strip_accents({x}))"
    x = f"replace({x}, '&', ' and ')"
    x = f"regexp_replace({x}, '[^\\p{{L}}\\p{{N}}]+', ' ', 'g')"
    x = f"trim(regexp_replace({x}, '\\s+', ' ', 'g'))"
    # Word-level road/unit abbreviations.
    mappings = [
        ("street", "st"), ("st", "st"),
        ("road", "rd"), ("rd", "rd"),
        ("avenue", "ave"), ("ave", "ave"),
        ("boulevard", "blvd"), ("blvd", "blvd"),
        ("drive", "dr"), ("dr", "dr"),
        ("lane", "ln"), ("ln", "ln"),
        ("highway", "hwy"), ("hwy", "hwy"),
        ("parkway", "pkwy"), ("pkwy", "pkwy"),
        ("circle", "cir"), ("cir", "cir"),
        ("court", "ct"), ("ct", "ct"),
        ("place", "pl"), ("pl", "pl"),
        ("terrace", "ter"), ("ter", "ter"),
        ("apartment", "apt"), ("apt", "apt"),
        ("suite", "ste"), ("ste", "ste"),
        ("unit", "unit"),
    ]
    # Apply long forms first so 'street' is handled before 'st'.
    for short, canonical in mappings:
        if short != canonical:
            x = f"regexp_replace({x}, '\\b{short}\\b', '{canonical}', 'g')"
    x = f"trim(regexp_replace({x}, '\\s+', ' ', 'g'))"
    return x


def build_feature_table(con: duckdb.DuckDBPyConnection, table_name: str, raw_table: str) -> None:
    """Create a compact, reusable feature table for one source."""
    sql = f"""
    CREATE TABLE {table_name} AS
    WITH b AS (
        SELECT
            row_number() OVER () - 1 AS row_id,
            CAST(entity_id AS VARCHAR) AS entity_id,
            CAST(country AS VARCHAR) AS country_raw,
            {make_name_norm_expr('business_name')} AS name_space0,
            {make_address_norm_expr('business_address')} AS address_space
        FROM {raw_table}
    ),
    n AS (
        SELECT
            *,
            trim(regexp_replace(name_space0, '{LEGAL_SUFFIX_RE}', '', 'g')) AS name_core_space0
        FROM b
    ),
    f AS (
        SELECT
            row_id,
            entity_id,
            nullif(trim(lower(country_raw)), '') AS country,
            name_space0 AS name_space,
            CASE
                WHEN length(trim(regexp_replace(name_space0, '{LEGAL_SUFFIX_RE}', '', 'g'))) >= 3
                THEN trim(regexp_replace(name_space0, '{LEGAL_SUFFIX_RE}', '', 'g'))
                ELSE name_space0
            END AS name_core_space,
            address_space,
            nullif(regexp_extract(address_space, '^\\s*([0-9]{{1,8}}[A-Za-z]?)', 1), '') AS house_num,
            nullif(regexp_extract(address_space, '.*\\b([0-9]{{5,6}})\\b', 1), '') AS postal,
        FROM n
    )
    SELECT
        *,
        replace(name_space, ' ', '') AS name_key,
        replace(name_core_space, ' ', '') AS name_core_key,
        substr(replace(name_core_space, ' ', ''), 1, 4) AS name_first4,
        substr(replace(name_core_space, ' ', ''), 1, 6) AS name_first6,
        substr(replace(name_core_space, ' ', ''), greatest(length(replace(name_core_space, ' ', '')) - 3, 1), 4) AS name_last4,
        substr(replace(name_core_space, ' ', ''), greatest(length(replace(name_core_space, ' ', '')) - 1, 1), 2) AS name_last2,
        length(replace(name_core_space, ' ', '')) AS name_len,
        floor(length(replace(name_core_space, ' ', '')) / 3)::INTEGER AS name_len_bucket,
        replace(address_space, ' ', '') AS address_key,
        substr(replace(address_space, ' ', ''), 1, 12) AS address_first12,
        substr(
            regexp_replace(replace(address_space, ' ', ''), '^[0-9]+[A-Za-z]?', '', 'g'),
            1, 12
        ) AS address_street12,
        substr(replace(name_core_space, ' ', ''), 1, 4)
            || '|' ||
        substr(replace(name_core_space, ' ', ''), greatest(length(replace(name_core_space, ' ', '')) - 3, 1), 4)
            AS name_edge4
    FROM f;
    """
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(sql)


def maybe_add_ascii_features(con: duckdb.DuckDBPyConnection, table_name: str, enable: bool) -> bool:
    """Optional local transliteration for non-ASCII records only."""
    if not enable:
        return False
    if _unidecode_mod is None:
        log("unidecode not installed; skipping transliteration (unicode normalization remains enabled)")
        return False

    def translit(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _unidecode_mod.unidecode(value)

    try:
        con.create_function("py_unidecode", translit, [str], str, null_handling="special")
    except Exception as exc:
        log(f"Could not register transliteration UDF; continuing without it: {exc}")
        return False

    con.execute(f"ALTER TABLE {table_name} ADD COLUMN name_ascii_key VARCHAR")
    con.execute(f"ALTER TABLE {table_name} ADD COLUMN address_ascii_key VARCHAR")
    con.execute(f"ALTER TABLE {table_name} ADD COLUMN name_ascii_core_key VARCHAR")

    # Apply the relatively expensive Python UDF only to non-ASCII strings.
    con.execute(f"""
        UPDATE {table_name}
        SET
          name_ascii_key = name_key,
          name_ascii_core_key = name_core_key,
          address_ascii_key = address_key
    """)
    con.execute(f"""
        UPDATE {table_name}
        SET
          name_ascii_key = replace(py_unidecode(name_space), ' ', ''),
          name_ascii_core_key = replace(py_unidecode(name_core_space), ' ', ''),
          address_ascii_key = replace(py_unidecode(address_space), ' ', '')
        WHERE regexp_matches(name_space || address_space, '[^\\x00-\\x7F]')
    """)
    return True


# ---------------------------------------------------------------------------
# Block statistics / rare token feature tables
# ---------------------------------------------------------------------------


def create_safe_key_table(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    key_expr: str,
    key_table: str,
    max_df: int,
    include_country: bool = True,
) -> None:
    con.execute(f"DROP TABLE IF EXISTS {key_table}")
    country = "country," if include_country else ""
    country_group = "country," if include_country else ""
    con.execute(f"""
        CREATE TABLE {key_table} AS
        SELECT {country} {key_expr} AS block_key
        FROM {source_table}
        WHERE {key_expr} IS NOT NULL AND {key_expr} <> ''
        GROUP BY {country} {key_expr}
        HAVING COUNT(*) <= {int(max_df)}
    """)


def create_token_features(
    con: duckdb.DuckDBPyConnection,
    s1: str,
    target: str,
    rare_df: int,
    rare_token_slots: int = 2,
) -> None:
    """Build target and S1 rare token tables, using target-side DF."""
    log("Building token document frequencies")
    con.execute("DROP TABLE IF EXISTS target_name_tokens")
    con.execute("DROP TABLE IF EXISTS target_token_df")
    con.execute("DROP TABLE IF EXISTS s1_name_tokens")
    con.execute("DROP TABLE IF EXISTS target_rare_tokens")
    con.execute("DROP TABLE IF EXISTS s1_rare_tokens")
    con.execute("DROP TABLE IF EXISTS target_token_pair")
    con.execute("DROP TABLE IF EXISTS s1_token_pair")

    token_cte = """
    SELECT DISTINCT row_id, country, token
    FROM {table}
    CROSS JOIN UNNEST(regexp_split_to_array(name_core_space, '\\s+')) AS u(token)
    WHERE token IS NOT NULL AND token <> '' AND length(token) >= 3
    """

    con.execute(
        "CREATE TABLE target_name_tokens AS "
        + token_cte.format(table=target)
    )
    con.execute(
        "CREATE TABLE s1_name_tokens AS "
        + token_cte.format(table=s1)
    )

    con.execute("""
        CREATE TABLE target_token_df AS
        SELECT country, token, COUNT(DISTINCT row_id) AS df
        FROM target_name_tokens
        GROUP BY country, token
    """)

    # Keep only target tokens that are selective enough to be a practical block.
    con.execute(f"""
        CREATE TABLE target_rare_tokens AS
        SELECT t.row_id, t.country, t.token
        FROM target_name_tokens t
        JOIN target_token_df d USING(country, token)
        WHERE d.df <= {int(rare_df)}
    """)

    # For each S1 record choose its rarest useful tokens based on target-side DF.
    con.execute(f"""
        CREATE TABLE s1_rare_tokens AS
        WITH ranked AS (
            SELECT
                t.row_id,
                t.country,
                t.token,
                d.df,
                row_number() OVER (
                    PARTITION BY t.row_id
                    ORDER BY d.df ASC, length(t.token) DESC, t.token ASC
                ) AS rn
            FROM s1_name_tokens t
            JOIN target_token_df d USING(country, token)
            WHERE d.df <= {int(rare_df)}
        )
        SELECT row_id, country, token, rn
        FROM ranked
        WHERE rn <= {int(rare_token_slots)}
    """)

    # Two-token order-independent signature.
    con.execute("""
        CREATE TABLE target_token_pair AS
        SELECT country, row_id, string_agg(token, '|' ORDER BY token) AS token_pair
        FROM target_rare_tokens
        GROUP BY country, row_id
        HAVING COUNT(*) >= 2
    """)
    con.execute(f"""
        CREATE TABLE s1_token_pair AS
        SELECT country, row_id, string_agg(token, '|' ORDER BY token) AS token_pair
        FROM s1_rare_tokens
        WHERE rn <= {int(rare_token_slots)}
        GROUP BY country, row_id
        HAVING COUNT(*) >= 2
    """)


def create_gram_features(
    con: duckdb.DuckDBPyConnection,
    s1: str,
    target: str,
    max_slots: int,
    gram_df: int,
) -> None:
    """Create a small character-anchor index without exploding every q-gram."""
    con.execute("DROP TABLE IF EXISTS target_name_grams")
    con.execute("DROP TABLE IF EXISTS target_gram_df")
    con.execute("DROP TABLE IF EXISTS target_rare_grams")
    con.execute("DROP TABLE IF EXISTS s1_name_grams")

    slots_target = [
        "substr(name_core_key, 1, 4)",
        "substr(name_core_key, 3, 4)",
        "substr(name_core_key, greatest(length(name_core_key) - 3, 1), 4)",
    ][: max_slots]
    slots_s1 = [
        "substr(name_core_key, 1, 4)",
        "substr(name_core_key, 3, 4)",
        "substr(name_core_key, greatest(length(name_core_key) - 3, 1), 4)",
    ][: max_slots]

    t_queries = [
        f"SELECT row_id, country, {expr} AS gram FROM {target} WHERE length(name_core_key) >= 4"
        for expr in slots_target
    ]
    s_queries = [
        f"SELECT row_id, country, {expr} AS gram FROM {s1} WHERE length(name_core_key) >= 4"
        for expr in slots_s1
    ]
    con.execute("CREATE TABLE target_name_grams AS SELECT DISTINCT * FROM (" + " UNION ALL ".join(t_queries) + ")")
    con.execute("CREATE TABLE s1_name_grams AS SELECT DISTINCT * FROM (" + " UNION ALL ".join(s_queries) + ")")

    con.execute("""
        CREATE TABLE target_gram_df AS
        SELECT country, gram, COUNT(DISTINCT row_id) AS df
        FROM target_name_grams
        WHERE gram IS NOT NULL AND gram <> ''
        GROUP BY country, gram
    """)
    con.execute(f"""
        CREATE TABLE target_rare_grams AS
        SELECT g.row_id, g.country, g.gram
        FROM target_name_grams g
        JOIN target_gram_df d USING(country, gram)
        WHERE d.df <= {int(gram_df)}
    """)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def candidate_pair_key(s1_row_expr: str, target_row_expr: str, target_count: int) -> str:
    return f"(({s1_row_expr})::UBIGINT * {int(target_count)}) + ({target_row_expr})::UBIGINT"


def run_pass(
    con: duckdb.DuckDBPyConnection,
    name: str,
    select_sql: str,
    pass_results: list[PassResult],
) -> None:
    t0 = now()
    log(f"PASS {name}")
    con.execute("DROP TABLE IF EXISTS _pass_pairs")
    con.execute(f"CREATE TEMP TABLE _pass_pairs AS {select_sql}")
    count = int(con.execute("SELECT COUNT(*) FROM _pass_pairs").fetchone()[0])
    con.execute("INSERT INTO candidate_pairs SELECT pair_key FROM _pass_pairs")
    con.execute("DROP TABLE _pass_pairs")
    elapsed = now() - t0
    pass_results.append(PassResult(name=name, pair_count=count, elapsed_sec=elapsed))
    log(f"  {count:,} pairs | {elapsed:,.2f}s")


def build_candidates(
    con: duckdb.DuckDBPyConnection,
    s1: str,
    target: str,
    target_count: int,
    args: argparse.Namespace,
) -> list[PassResult]:
    pass_results: list[PassResult] = []
    con.execute("DROP TABLE IF EXISTS candidate_pairs")
    con.execute("CREATE TABLE candidate_pairs(pair_key UBIGINT)")

    pair = lambda s, t: candidate_pair_key(s, t, target_count)

    # Precompute safe composite-key tables. Large blocks are intentionally
    # excluded; they are handled by more selective composite/rescue blocks.
    log("Building selective target block dictionaries")
    safe_defs = [
        # Country-aware selective blocks.
        ("country || '|' || name_key", "safe_name", args.exact_name_df),
        ("country || '|' || name_core_key", "safe_core", args.exact_core_df),
        ("country || '|' || address_key", "safe_address", args.exact_address_df),
        ("country || '|' || name_core_key || '|' || postal", "safe_name_postal", args.name_postal_df),
        ("country || '|' || house_num || '|' || name_first6", "safe_house_name", args.house_name_df),
        ("country || '|' || house_num || '|' || address_street12", "safe_house_address", args.house_address_df),
        ("country || '|' || name_edge4", "safe_name_edge", max(args.exact_name_df, 75)),
        # Very strict countryless fallbacks for mislabeled/missing country.
        ("name_key", "safe_name_global", 25),
        ("name_core_key", "safe_core_global", 25),
        ("address_key", "safe_address_global", 10),
        ("name_core_key || '|' || postal", "safe_name_postal_global", 50),
    ]
    for expr, tab, cap in safe_defs:
        create_safe_key_table(con, target, expr, tab, cap, include_country=False)

    # 1. Exact full normalized name, country-aware.
    run_pass(
        con,
        "01_exact_name",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM {s1} s
        JOIN {target} t
          ON s.country = t.country AND s.name_key = t.name_key
        JOIN safe_name k
          ON k.block_key = s.country || '|' || s.name_key
        WHERE s.name_key <> ''
        """,
        pass_results,
    )

    # 2. Exact legal-suffix-stripped name, country-aware.
    run_pass(
        con,
        "02_exact_core_name",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM {s1} s
        JOIN {target} t
          ON s.country = t.country AND s.name_core_key = t.name_core_key
        JOIN safe_core k
          ON k.block_key = s.country || '|' || s.name_core_key
        WHERE s.name_core_key <> ''
        """,
        pass_results,
    )

    # 3. Countryless exact name fallback, strictly capped.
    run_pass(
        con,
        "03_countryless_exact_name",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM {s1} s
        JOIN {target} t ON s.name_key = t.name_key
        JOIN safe_name_global k ON k.block_key = s.name_key
        WHERE s.name_key <> ''
        """,
        pass_results,
    )

    # 4. Countryless legal-suffix-stripped name fallback.
    run_pass(
        con,
        "04_countryless_exact_core_name",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM {s1} s
        JOIN {target} t ON s.name_core_key = t.name_core_key
        JOIN safe_core_global k ON k.block_key = s.name_core_key
        WHERE s.name_core_key <> ''
        """,
        pass_results,
    )

    # 3. Name + postal. This rescues common business names.
    run_pass(
        con,
        "05_name_postal",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM {s1} s
        JOIN {target} t
          ON s.country = t.country
         AND s.name_core_key = t.name_core_key
         AND s.postal = t.postal
        JOIN safe_name_postal k
          ON k.block_key = s.country || '|' || s.name_core_key || '|' || s.postal
        WHERE s.name_core_key <> '' AND s.postal <> ''
        """,
        pass_results,
    )

    # 6. House number + name prefix. Typo resilient, still highly selective.
    run_pass(
        con,
        "06_house_name_prefix",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM {s1} s
        JOIN {target} t
          ON s.country = t.country
         AND s.house_num = t.house_num
         AND s.name_first6 = t.name_first6
        JOIN safe_house_name k
          ON k.block_key = s.country || '|' || s.house_num || '|' || s.name_first6
        WHERE s.house_num <> '' AND s.name_first6 <> ''
        """,
        pass_results,
    )

    # 7. Exact normalized address.
    run_pass(
        con,
        "07_exact_address",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM {s1} s
        JOIN {target} t
          ON s.country = t.country AND s.address_key = t.address_key
        JOIN safe_address k
          ON k.block_key = s.country || '|' || s.address_key
        WHERE s.address_key <> ''
        """,
        pass_results,
    )

    # 8. Countryless exact-address fallback, very tightly capped.
    run_pass(
        con,
        "08_countryless_exact_address",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM {s1} s
        JOIN {target} t ON s.address_key = t.address_key
        JOIN safe_address_global k ON k.block_key = s.address_key
        WHERE s.address_key <> ''
        """,
        pass_results,
    )

    # 9. House + street signature. Helps when address punctuation/components move.
    run_pass(
        con,
        "09_house_address_signature",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM {s1} s
        JOIN {target} t
          ON s.country = t.country
         AND s.house_num = t.house_num
         AND s.address_street12 = t.address_street12
        JOIN safe_house_address k
          ON k.block_key = s.country || '|' || s.house_num || '|' || s.address_street12
        WHERE s.house_num <> '' AND s.address_street12 <> ''
        """,
        pass_results,
    )

    run_pass(
        con,
        "10_name_edge4",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM {s1} s
        JOIN {target} t
          ON s.country = t.country AND s.name_edge4 = t.name_edge4
        JOIN safe_name_edge k
          ON k.block_key = s.country || '|' || s.name_edge4
        WHERE s.name_edge4 <> ''
        """,
        pass_results,
    )

    # 11. Rare token #1.
    run_pass(
        con,
        "11_rare_token_1",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM s1_rare_tokens s
        JOIN target_rare_tokens t
          ON s.country = t.country AND s.token = t.token
        WHERE s.rn = 1
        """,
        pass_results,
    )

    # 12. Rare token #2. A second independent key increases recall.
    run_pass(
        con,
        "12_rare_token_2",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM s1_rare_tokens s
        JOIN target_rare_tokens t
          ON s.country = t.country AND s.token = t.token
        WHERE s.rn = 2
        """,
        pass_results,
    )

    # 13. Order-independent pair of rare tokens.
    con.execute("DROP TABLE IF EXISTS safe_token_pair")
    con.execute(f"""
        CREATE TABLE safe_token_pair AS
        SELECT country || '|' || token_pair AS block_key
        FROM target_token_pair
        GROUP BY country || '|' || token_pair
        HAVING COUNT(*) <= {int(args.rare_pair_df)}
    """)
    run_pass(
        con,
        "13_rare_token_pair",
        f"""
        SELECT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM s1_token_pair s
        JOIN target_token_pair t
          ON s.country = t.country AND s.token_pair = t.token_pair
        JOIN safe_token_pair k
          ON k.block_key = s.country || '|' || s.token_pair
        """,
        pass_results,
    )

    # 14. Character anchors. Each record can emit up to N short anchors.
    run_pass(
        con,
        "14_character_anchor",
        f"""
        SELECT DISTINCT {pair('s.row_id', 't.row_id')} AS pair_key
        FROM s1_name_grams s
        JOIN target_rare_grams t
          ON s.country = t.country AND s.gram = t.gram
        """,
        pass_results,
    )

    # 15-16. Optional transliteration passes. These are deliberately narrower and
    # reuse exact/edge keys only on non-ASCII rows.
    if args.transliteration:
        con.execute("DROP TABLE IF EXISTS safe_ascii_name")
        con.execute("DROP TABLE IF EXISTS safe_ascii_address")
        con.execute(f"""
            CREATE TABLE safe_ascii_name AS
            SELECT country || '|' || name_ascii_core_key AS block_key
            FROM target
            WHERE name_ascii_core_key <> ''
            GROUP BY country || '|' || name_ascii_core_key
            HAVING COUNT(*) <= {int(args.exact_core_df)}
        """)
        con.execute(f"""
            CREATE TABLE safe_ascii_address AS
            SELECT country || '|' || address_ascii_key AS block_key
            FROM target
            WHERE address_ascii_key <> ''
            GROUP BY country || '|' || address_ascii_key
            HAVING COUNT(*) <= {int(args.exact_address_df)}
        """)

        run_pass(
            con,
            "15_ascii_name_exact",
            f"""
            SELECT {pair('s.row_id', 't.row_id')} AS pair_key
            FROM {s1} s
            JOIN {target} t
              ON s.country = t.country
             AND s.name_ascii_core_key <> ''
             AND s.name_ascii_core_key = t.name_ascii_core_key
            JOIN safe_ascii_name k
              ON k.block_key = s.country || '|' || s.name_ascii_core_key
            """,
            pass_results,
        )
        run_pass(
            con,
            "16_ascii_address",
            f"""
            SELECT {pair('s.row_id', 't.row_id')} AS pair_key
            FROM {s1} s
            JOIN {target} t
              ON s.country = t.country
             AND s.address_ascii_key <> ''
             AND s.address_ascii_key = t.address_ascii_key
            JOIN safe_ascii_address k
              ON k.block_key = s.country || '|' || s.address_ascii_key
            """,
            pass_results,
        )

    # De-duplicate after all broad/cheap passes.
    t0 = now()
    con.execute("CREATE TABLE candidate_pairs_distinct AS SELECT DISTINCT pair_key FROM candidate_pairs")
    con.execute("DROP TABLE candidate_pairs")
    con.execute("ALTER TABLE candidate_pairs_distinct RENAME TO candidate_pairs")
    log(f"Union/dedup candidates: {now() - t0:,.2f}s")

    # Optional sorted-neighborhood rescue for only unresolved S1 rows.
    if args.sorted_rescue:
        unresolved = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM {s1} s
                WHERE NOT EXISTS (
                    SELECT 1 FROM candidate_pairs c
                    WHERE CAST(c.pair_key / {int(target_count)} AS UBIGINT) = s.row_id
                )
                """
            ).fetchone()[0]
        )
        log(f"Unresolved S1 after main passes: {unresolved:,}")
        if unresolved > 0:
            run_sorted_rescue(con, s1, target, target_count, args.sorted_window, pass_results)
            con.execute("CREATE TABLE candidate_pairs_distinct AS SELECT DISTINCT pair_key FROM candidate_pairs")
            con.execute("DROP TABLE candidate_pairs")
            con.execute("ALTER TABLE candidate_pairs_distinct RENAME TO candidate_pairs")

    return pass_results


def run_sorted_rescue(
    con: duckdb.DuckDBPyConnection,
    s1: str,
    target: str,
    target_count: int,
    window: int,
    pass_results: list[PassResult],
) -> None:
    """Sorted-neighborhood only for unresolved S1 rows.

    Coarse partition: country + first2 chars + length bucket. Within each
    bucket, records are ordered lexically by name_core_key. Only cross-source
    neighbors within +/- window are emitted.
    """
    t0 = now()
    log("PASS 17_sorted_neighborhood_rescue")
    con.execute("DROP TABLE IF EXISTS _unresolved_s1")
    con.execute("DROP TABLE IF EXISTS _sn_target")

    con.execute(f"""
        CREATE TEMP TABLE _unresolved_s1 AS
        SELECT
            s.*,
            substr(s.name_core_key, 1, 2) AS coarse_prefix
        FROM {s1} s
        WHERE NOT EXISTS (
            SELECT 1 FROM candidate_pairs c
            WHERE CAST(c.pair_key / {int(target_count)} AS UBIGINT) = s.row_id
        )
        AND s.name_core_key <> ''
    """)
    con.execute(f"""
        CREATE TEMP TABLE _sn_target AS
        SELECT
            t.*,
            substr(t.name_core_key, 1, 2) AS coarse_prefix
        FROM {target} t
        WHERE t.name_core_key <> ''
    """)

    # Row-number windows over compact partitions. This is one global sort of
    # the target index, but it runs only once and only as a final rescue.
    con.execute("DROP TABLE IF EXISTS _sn_combined")
    con.execute("""
        CREATE TEMP TABLE _sn_combined AS
        SELECT
            'S1' AS src,
            row_id,
            country,
            coarse_prefix,
            name_len_bucket,
            name_core_key,
            row_number() OVER (
                PARTITION BY country, coarse_prefix, name_len_bucket
                ORDER BY name_core_key, row_id
            ) AS rn
        FROM _unresolved_s1
        UNION ALL
        SELECT
            'T' AS src,
            row_id,
            country,
            coarse_prefix,
            name_len_bucket,
            name_core_key,
            row_number() OVER (
                PARTITION BY country, coarse_prefix, name_len_bucket
                ORDER BY name_core_key, row_id
            ) AS rn
        FROM _sn_target
    """)

    run_sql = f"""
        SELECT DISTINCT
            {candidate_pair_key('s.row_id', 't.row_id', target_count)} AS pair_key
        FROM _sn_combined s
        JOIN _sn_combined t
          ON s.src = 'S1'
         AND t.src = 'T'
         AND s.country = t.country
         AND s.coarse_prefix = t.coarse_prefix
         AND s.name_len_bucket = t.name_len_bucket
         AND abs(s.rn - t.rn) <= {int(window)}
    """
    con.execute("CREATE TEMP TABLE _pass_pairs AS " + run_sql)
    cnt = int(con.execute("SELECT COUNT(*) FROM _pass_pairs").fetchone()[0])
    con.execute("INSERT INTO candidate_pairs SELECT pair_key FROM _pass_pairs")
    con.execute("DROP TABLE _pass_pairs")
    con.execute("DROP TABLE _sn_combined")
    con.execute("DROP TABLE _sn_target")
    con.execute("DROP TABLE _unresolved_s1")
    elapsed = now() - t0
    pass_results.append(PassResult("17_sorted_neighborhood_rescue", cnt, elapsed))
    log(f"  {cnt:,} pairs | {elapsed:,.2f}s")


# ---------------------------------------------------------------------------
# Output / evaluation
# ---------------------------------------------------------------------------


def create_sorted_candidate_parquet(
    con: duckdb.DuckDBPyConnection,
    s1: str,
    target: str,
    target_count: int,
    path: Path,
) -> None:
    """Export numeric+text candidate rows in S1 order for streaming assembly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con.execute(f"""
        COPY (
            SELECT
                CAST(c.pair_key / {int(target_count)} AS UBIGINT) AS s1_row,
                CAST(c.pair_key % {int(target_count)} AS UBIGINT) AS target_row,
                s.entity_id AS source1_entity_id,
                t.entity_id AS candidate_entity_id
            FROM candidate_pairs c
            JOIN {s1} s
              ON s.row_id = CAST(c.pair_key / {int(target_count)} AS UBIGINT)
            JOIN {target} t
              ON t.row_id = CAST(c.pair_key % {int(target_count)} AS UBIGINT)
            ORDER BY s1_row, target_row
        ) TO {sql_literal(str(path))}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


def stream_challenge_output(
    con: duckdb.DuckDBPyConnection,
    s1: str,
    sorted_parquet: Path,
    output_path: Path,
    n_s1: int,
    batch_s1: int,
    gzip_output: bool,
) -> None:
    """Generate the required one-row-per-S1 candidate list without a 5M-group
    in-memory string_agg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    mode = "wt"
    with opener(output_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for lo in range(0, n_s1, batch_s1):
            hi = min(n_s1, lo + batch_s1)
            rows = con.execute(
                f"""
                SELECT
                    s.row_id,
                    s.entity_id,
                    coalesce(string_agg(p.candidate_entity_id, ',' ORDER BY p.candidate_entity_id), '') AS candidate_entity_ids
                FROM {s1} s
                LEFT JOIN read_parquet({sql_literal(str(sorted_parquet))}) p
                  ON p.s1_row = s.row_id
                 AND p.s1_row >= {lo}
                 AND p.s1_row < {hi}
                WHERE s.row_id >= {lo} AND s.row_id < {hi}
                GROUP BY s.row_id, s.entity_id
                ORDER BY s.row_id
                """
            ).fetchall()
            for _, entity_id, candidate_ids in rows:
                writer.writerow([entity_id, candidate_ids or ""])
            if lo == 0 or hi == n_s1 or hi % (batch_s1 * 10) == 0:
                log(f"Wrote S1 rows {lo:,}-{hi:,} / {n_s1:,}")


def evaluate_ground_truth(
    con: duckdb.DuckDBPyConnection,
    gt_path: Path,
    candidate_parquet: Path,
) -> dict:
    """Pair-level blocking recall from the challenge's GT mapping file."""
    log("Evaluating blocking recall")
    con.execute("DROP TABLE IF EXISTS ground_truth")
    con.execute("DROP TABLE IF EXISTS ground_truth_pairs")
    con.execute("""
        CREATE TABLE ground_truth AS
        SELECT *
        FROM read_csv(
            {path},
            delim='\\t',
            header=true,
            all_varchar=true
        )
    """.format(path=sql_literal(str(gt_path))))

    con.execute("""
        CREATE TABLE ground_truth_pairs AS
        SELECT
            gt.source1_entity_id,
            trim(matched_id) AS matched_entity_id
        FROM ground_truth gt
        CROSS JOIN UNNEST(string_split(coalesce(gt.matched_entity_ids, ''), ',')) AS u(matched_id)
        WHERE trim(matched_id) <> ''
    """)

    total_gt = int(con.execute("SELECT COUNT(*) FROM ground_truth_pairs").fetchone()[0])
    recovered = int(con.execute(f"""
        SELECT COUNT(*)
        FROM ground_truth_pairs g
        JOIN read_parquet({sql_literal(str(candidate_parquet))}) c
          ON g.source1_entity_id = c.source1_entity_id
         AND g.matched_entity_id = c.candidate_entity_id
    """).fetchone()[0])

    s1_positive = int(con.execute(
        "SELECT COUNT(DISTINCT source1_entity_id) FROM ground_truth_pairs"
    ).fetchone()[0])
    s1_recovered = int(con.execute(f"""
        SELECT COUNT(DISTINCT g.source1_entity_id)
        FROM ground_truth_pairs g
        JOIN read_parquet({sql_literal(str(candidate_parquet))}) c
          ON g.source1_entity_id = c.source1_entity_id
         AND g.matched_entity_id = c.candidate_entity_id
    """).fetchone()[0])

    recall = recovered / total_gt if total_gt else 1.0
    entity_coverage = s1_recovered / s1_positive if s1_positive else 1.0
    result = {
        "total_ground_truth_pairs": total_gt,
        "recovered_ground_truth_pairs": recovered,
        "blocking_recall": recall,
        "positive_s1_entities": s1_positive,
        "s1_entities_with_at_least_one_recovered_match": s1_recovered,
        "positive_s1_entity_coverage": entity_coverage,
    }
    log(
        f"Blocking recall: {recall * 100:.3f}% "
        f"({recovered:,}/{total_gt:,}) | S1 positive coverage: {entity_coverage * 100:.3f}%"
    )
    return result


def compute_candidate_stats(
    con: duckdb.DuckDBPyConnection,
    s1: str,
    target_count: int,
) -> dict:
    pair_count = int(con.execute("SELECT COUNT(*) FROM candidate_pairs").fetchone()[0])
    n_s1 = int(con.execute(f"SELECT COUNT(*) FROM {s1}").fetchone()[0])

    # Candidate counts per S1 are calculated over compact row IDs. This is a
    # simple group-by and avoids materializing the full candidate strings.
    avg_candidates = float(con.execute(f"""
        SELECT coalesce(avg(cnt), 0)
        FROM (
            SELECT CAST(pair_key / {int(target_count)} AS UBIGINT) AS s1_row, COUNT(*) AS cnt
            FROM candidate_pairs
            GROUP BY 1
        ) q
    """).fetchone()[0])
    max_candidates = int(con.execute(f"""
        SELECT coalesce(max(cnt), 0)
        FROM (
            SELECT CAST(pair_key / {int(target_count)} AS UBIGINT) AS s1_row, COUNT(*) AS cnt
            FROM candidate_pairs
            GROUP BY 1
        ) q
    """).fetchone()[0])
    touched_s1 = int(con.execute(f"""
        SELECT COUNT(DISTINCT CAST(pair_key / {int(target_count)} AS UBIGINT))
        FROM candidate_pairs
    """).fetchone()[0])

    total_possible = n_s1 * target_count
    reduction_ratio = 1.0 - (pair_count / total_possible) if total_possible else 1.0

    return {
        "s1_rows": n_s1,
        "target_rows": target_count,
        "candidate_pairs": pair_count,
        "candidate_s1_rows_with_at_least_one": touched_s1,
        "candidate_s1_coverage": (touched_s1 / n_s1 if n_s1 else 1.0),
        "average_candidates_per_touched_s1": avg_candidates,
        "max_candidates_for_one_s1": max_candidates,
        "theoretical_pair_space": total_possible,
        "candidate_reduction_ratio": reduction_ratio,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Production-grade multi-stage business entity blocker")
    p.add_argument("--source1", required=True, type=Path)
    p.add_argument("--source2", required=True, type=Path)
    p.add_argument("--source3", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("candidate_pairs.tsv"))
    p.add_argument("--ground-truth", type=Path, default=None)
    p.add_argument("--workdir", type=Path, default=Path("blocking_work"))
    p.add_argument("--memory", default=None, help="DuckDB memory limit, e.g. 8GB")
    p.add_argument("--threads", type=int, default=max(2, min(12, (os.cpu_count() or 4))))
    p.add_argument("--temp-dir", type=Path, default=None)
    p.add_argument("--batch-s1", type=int, default=DEFAULTS["batch_s1"])
    p.add_argument("--gzip-output", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true")
    p.add_argument("--no-transliteration", dest="transliteration", action="store_false")
    p.set_defaults(transliteration=True)
    p.add_argument("--sorted-rescue", action="store_true", help="Run sorted-neighborhood only for unresolved S1 rows")
    p.add_argument("--sorted-window", type=int, default=DEFAULTS["sorted_window"])

    p.add_argument("--exact-name-df", type=int, default=DEFAULTS["exact_name_df"])
    p.add_argument("--exact-core-df", type=int, default=DEFAULTS["exact_core_df"])
    p.add_argument("--exact-address-df", type=int, default=DEFAULTS["exact_address_df"])
    p.add_argument("--name-postal-df", type=int, default=DEFAULTS["name_postal_df"])
    p.add_argument("--house-name-df", type=int, default=DEFAULTS["house_name_df"])
    p.add_argument("--house-address-df", type=int, default=DEFAULTS["house_address_df"])
    p.add_argument("--rare-token-df", type=int, default=DEFAULTS["rare_token_df"])
    p.add_argument("--rare-pair-df", type=int, default=DEFAULTS["rare_pair_df"])
    p.add_argument("--rare-gram-df", type=int, default=DEFAULTS["rare_gram_df"])
    p.add_argument("--rare-token-slots", type=int, default=2, choices=[1, 2])
    p.add_argument("--gram-slots", type=int, default=DEFAULTS["max_gram_slots"], choices=[1, 2, 3])
    p.add_argument("--min-recall", type=float, default=None,
                   help="Fail with exit code 2 when --ground-truth recall is below this value")
    return p.parse_args()


def validate_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"Empty input file: {path}")


def main() -> int:
    args = parse_args()
    for path in [args.source1, args.source2, args.source3]:
        validate_input(path)
    if args.ground_truth:
        validate_input(args.ground_truth)

    ensure_dir(args.workdir)
    db_path = args.workdir / "blocking.duckdb"
    temp_dir = args.temp_dir or (args.workdir / "duckdb_tmp")
    ensure_dir(temp_dir)

    if free_disk_gb(args.workdir) < 25:
        log("WARNING: less than 25 GB free in workdir filesystem; candidate generation may spill heavily.")

    mem = args.memory or default_memory_limit()
    log(f"Platform: {platform.platform()}")
    log(f"CPUs: {os.cpu_count() or 1} | RAM: {physical_memory_gb():.1f} GB")
    log(f"DuckDB threads: {args.threads} | memory_limit: {mem}")
    log(f"Workdir: {args.workdir.resolve()}")

    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"SET threads = {int(args.threads)}")
        con.execute(f"SET memory_limit = {sql_literal(mem)}")
        con.execute(f"SET temp_directory = {sql_literal(str(temp_dir.resolve()))}")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET enable_progress_bar = false")

        # Ingest raw sources only once. DuckDB's read_csv supports explicit
        # all-VARCHAR ingestion, which avoids repeated type inference.
        log("Loading source tables")
        for name, path in [
            ("raw_s1", args.source1),
            ("raw_s2", args.source2),
            ("raw_s3", args.source3),
        ]:
            con.execute(f"DROP TABLE IF EXISTS {name}")
            con.execute(f"""
                CREATE TABLE {name} AS
                SELECT *
                FROM read_csv(
                    {sql_literal(str(path.resolve()))},
                    delim='\\t',
                    header=true,
                    all_varchar=true
                )
            """)
            cols = [r[0] for r in con.execute(f"DESCRIBE {name}").fetchall()]
            missing = [c for c in REQUIRED_COLUMNS if c not in cols]
            if missing:
                raise ValueError(f"{path}: missing required columns {missing}; got {cols}")
            log(f"  {name}: {int(con.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]):,} rows")

        # Feature materialization.
        for table_name, raw in [("s1", "raw_s1"), ("s2", "raw_s2"), ("s3", "raw_s3")]:
            timer = Timer(f"Normalize {raw} -> {table_name}")
            build_feature_table(con, table_name, raw)
            timer.done()
            if args.transliteration:
                ascii_enabled = maybe_add_ascii_features(con, table_name, enable=True)
                log(f"  transliteration features: {'enabled' if ascii_enabled else 'disabled'}")

        # Combine S2 and S3 targets once.
        con.execute("DROP TABLE IF EXISTS target")
        con.execute("""
            CREATE TABLE target AS
            SELECT * FROM s2
            UNION ALL BY NAME
            SELECT * FROM s3
        """)
        target_count = int(con.execute("SELECT COUNT(*) FROM target").fetchone()[0])
        n_s1 = int(con.execute("SELECT COUNT(*) FROM s1").fetchone()[0])
        log(f"Target rows (S2+S3): {target_count:,}")
        log(f"Source-1 rows: {n_s1:,}")

        # Target-side feature indices.
        timer = Timer("Build rare-token features")
        create_token_features(
            con,
            "s1",
            "target",
            rare_df=args.rare_token_df,
            rare_token_slots=args.rare_token_slots,
        )
        timer.done()

        timer = Timer("Build character-anchor features")
        create_gram_features(
            con,
            "s1",
            "target",
            max_slots=args.gram_slots,
            gram_df=args.rare_gram_df,
        )
        timer.done()

        # Candidate generation.
        timer = Timer("Candidate generation")
        pass_results = build_candidates(con, "s1", "target", target_count, args)
        timer.done()

        stats = compute_candidate_stats(con, "s1", target_count)
        log(
            f"Final candidates: {stats['candidate_pairs']:,} | "
            f"avg/touched-S1: {stats['average_candidates_per_touched_s1']:.2f} | "
            f"max/S1: {stats['max_candidates_for_one_s1']:,} | "
            f"reduction ratio: {stats['candidate_reduction_ratio'] * 100:.6f}%"
        )

        # Sorted candidate intermediate. It makes final output streaming safe.
        sorted_parquet = args.workdir / "candidate_rows_sorted.parquet"
        timer = Timer("Materialize sorted candidate rows")
        create_sorted_candidate_parquet(con, "s1", "target", target_count, sorted_parquet)
        timer.done()

        # Challenge-compatible final file.
        output_path = args.output
        if args.gzip_output and output_path.suffix != ".gz":
            output_path = output_path.with_suffix(output_path.suffix + ".gz")
        timer = Timer("Write final candidate_pairs.tsv")
        stream_challenge_output(
            con,
            "s1",
            sorted_parquet,
            output_path,
            n_s1,
            args.batch_s1,
            args.gzip_output,
        )
        timer.done()

        evaluation = None
        if args.ground_truth:
            evaluation = evaluate_ground_truth(con, args.ground_truth, sorted_parquet)
            if args.min_recall is not None and evaluation["blocking_recall"] < args.min_recall:
                log(
                    f"RECALL GATE FAILED: {evaluation['blocking_recall']:.6f} "
                    f"< required {args.min_recall:.6f}"
                )
                exit_code = 2
            else:
                exit_code = 0
        else:
            exit_code = 0

        report = {
            "inputs": {
                "source1": str(args.source1),
                "source2": str(args.source2),
                "source3": str(args.source3),
                "ground_truth": str(args.ground_truth) if args.ground_truth else None,
            },
            "config": {
                "memory": mem,
                "threads": args.threads,
                "transliteration": bool(args.transliteration),
                "sorted_rescue": bool(args.sorted_rescue),
                "sorted_window": args.sorted_window,
                "exact_name_df": args.exact_name_df,
                "exact_core_df": args.exact_core_df,
                "exact_address_df": args.exact_address_df,
                "name_postal_df": args.name_postal_df,
                "house_name_df": args.house_name_df,
                "house_address_df": args.house_address_df,
                "rare_token_df": args.rare_token_df,
                "rare_pair_df": args.rare_pair_df,
                "rare_gram_df": args.rare_gram_df,
                "rare_token_slots": args.rare_token_slots,
                "gram_slots": args.gram_slots,
            },
            "stats": stats,
            "passes": [
                {"name": p.name, "pair_count": p.pair_count, "elapsed_sec": p.elapsed_sec}
                for p in pass_results
            ],
            "evaluation": evaluation,
            "output": str(output_path),
        }
        report_path = args.workdir / "blocking_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"Report: {report_path}")

        if not args.keep_intermediate:
            try:
                sorted_parquet.unlink()
            except FileNotFoundError:
                pass

        return exit_code
    finally:
        con.close()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
