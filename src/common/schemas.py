"""
Stable data contracts between pipeline phases.

These schemas are intentionally lightweight.
The exact competition submission schema must follow the
official challenge README and validator.
"""

from dataclasses import dataclass
from typing import Literal


CandidateSource = Literal["S2", "S3"]


@dataclass(frozen=True)
class CandidatePair:
    s1_id: str
    candidate_id: str
    candidate_source: CandidateSource
    block_method: str


@dataclass(frozen=True)
class CandidateSetPair:
    """One unique entity pair with all blocking methods that generated it."""

    s1_id: str
    candidate_id: str
    candidate_source: CandidateSource
    block_methods: tuple[str, ...]


@dataclass(frozen=True)
class ScoredPair:
    s1_id: str
    candidate_id: str
    candidate_source: CandidateSource
    match_score: float
