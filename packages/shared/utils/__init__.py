# -*- coding: utf-8 -*-
"""
CCEA Shared Utilities.

Common utilities safe for both Cloud and Agent zones.
"""

from typing import Final

ZONE: Final[str] = "shared"

from .validation import (
    validate_symbol,
    validate_quantity,
    validate_price,
    is_valid_digest,
    is_valid_uuid,
)

from .hashing import (
    compute_sha256,
    compute_content_hash,
    verify_digest,
)

__all__ = [
    # Validation
    "validate_symbol",
    "validate_quantity",
    "validate_price",
    "is_valid_digest",
    "is_valid_uuid",
    # Hashing
    "compute_sha256",
    "compute_content_hash",
    "verify_digest",
]
