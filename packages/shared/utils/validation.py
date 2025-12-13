# -*- coding: utf-8 -*-
"""
Validation Utilities.

Common validation functions for data integrity checks.
Safe for use in both Cloud and Agent zones.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional
from uuid import UUID


# Symbol validation pattern (alphanumeric, underscores, dashes)
SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9_\-/]{1,32}$")

# SHA256 digest pattern (64 hex chars)
SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")

# SHA256 with prefix
SHA256_PREFIXED_PATTERN = re.compile(r"^sha256:[a-fA-F0-9]{64}$")


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format.

    Args:
        symbol: Trading symbol to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_symbol("BTCUSDT")
        True
        >>> validate_symbol("BTC/USD")
        True
        >>> validate_symbol("invalid symbol!")
        False
    """
    if not symbol or not isinstance(symbol, str):
        return False
    return bool(SYMBOL_PATTERN.match(symbol))


def validate_quantity(
    quantity: str | Decimal | float,
    min_value: Optional[Decimal] = None,
    max_value: Optional[Decimal] = None,
    allow_zero: bool = False,
    allow_negative: bool = False,
) -> tuple[bool, Optional[str], Optional[Decimal]]:
    """
    Validate quantity value.

    Args:
        quantity: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_zero: Whether zero is valid
        allow_negative: Whether negative values are valid

    Returns:
        Tuple of (is_valid, error_message, parsed_value)

    Examples:
        >>> validate_quantity("100.5")
        (True, None, Decimal('100.5'))
        >>> validate_quantity("-10", allow_negative=False)
        (False, 'Negative values not allowed', Decimal('-10'))
    """
    try:
        if isinstance(quantity, (int, float)):
            parsed = Decimal(str(quantity))
        elif isinstance(quantity, str):
            parsed = Decimal(quantity)
        elif isinstance(quantity, Decimal):
            parsed = quantity
        else:
            return False, f"Invalid type: {type(quantity)}", None
    except (InvalidOperation, ValueError) as e:
        return False, f"Invalid decimal: {e}", None

    # Check zero
    if parsed == Decimal("0") and not allow_zero:
        return False, "Zero not allowed", parsed

    # Check negative
    if parsed < Decimal("0") and not allow_negative:
        return False, "Negative values not allowed", parsed

    # Check min
    if min_value is not None and parsed < min_value:
        return False, f"Value {parsed} below minimum {min_value}", parsed

    # Check max
    if max_value is not None and parsed > max_value:
        return False, f"Value {parsed} above maximum {max_value}", parsed

    return True, None, parsed


def validate_price(
    price: str | Decimal | float,
    min_tick: Optional[Decimal] = None,
) -> tuple[bool, Optional[str], Optional[Decimal]]:
    """
    Validate price value.

    Args:
        price: Price to validate
        min_tick: Minimum tick size for validation

    Returns:
        Tuple of (is_valid, error_message, parsed_value)
    """
    is_valid, error, parsed = validate_quantity(
        price, min_value=Decimal("0"), allow_zero=False, allow_negative=False
    )

    if not is_valid:
        return is_valid, error, parsed

    # Check tick size
    if min_tick is not None and parsed is not None:
        remainder = parsed % min_tick
        if remainder != Decimal("0"):
            return (
                False,
                f"Price {parsed} not aligned to tick size {min_tick}",
                parsed,
            )

    return True, None, parsed


def is_valid_digest(digest: str, with_prefix: bool = False) -> bool:
    """
    Validate SHA256 digest format.

    Args:
        digest: Digest string to validate
        with_prefix: Whether to expect 'sha256:' prefix

    Returns:
        True if valid SHA256 digest format
    """
    if not digest or not isinstance(digest, str):
        return False

    if with_prefix:
        return bool(SHA256_PREFIXED_PATTERN.match(digest))
    return bool(SHA256_PATTERN.match(digest))


def is_valid_uuid(value: str) -> bool:
    """
    Validate UUID format.

    Args:
        value: String to validate

    Returns:
        True if valid UUID format
    """
    if not value or not isinstance(value, str):
        return False

    try:
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def validate_schema_version(
    version: str,
    min_version: str = "1.0.0",
    max_version: str = "99.99.99",
) -> tuple[bool, Optional[str]]:
    """
    Validate schema version string.

    Args:
        version: Version string (semver format)
        min_version: Minimum supported version
        max_version: Maximum supported version

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Simple semver pattern
    semver_pattern = re.compile(r"^\d+\.\d+\.\d+$")

    if not semver_pattern.match(version):
        return False, f"Invalid version format: {version}"

    def parse_version(v: str) -> tuple[int, int, int]:
        parts = v.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2])

    try:
        v = parse_version(version)
        min_v = parse_version(min_version)
        max_v = parse_version(max_version)

        if v < min_v:
            return False, f"Version {version} below minimum {min_version}"
        if v > max_v:
            return False, f"Version {version} above maximum {max_version}"

        return True, None
    except (ValueError, IndexError) as e:
        return False, f"Version parse error: {e}"
