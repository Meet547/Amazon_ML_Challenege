"""Phase 1: validated, lossless entity normalization."""

from .normalize import normalize_address, normalize_country, normalize_name

__all__ = ["normalize_address", "normalize_country", "normalize_name"]
