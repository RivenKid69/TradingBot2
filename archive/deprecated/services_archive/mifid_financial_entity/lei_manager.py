# -*- coding: utf-8 -*-
"""
LEI (Legal Entity Identifier) Management Module.

Implements LEI validation, verification, and management per ISO 17442 standard.
Required for MiFID II/MiFIR transaction reporting (Article 26).

"No LEI, No Trade" - Without a valid LEI, transaction reports cannot be
submitted to National Competent Authorities.

LEI Format (ISO 17442):
    - 20 characters total
    - Characters 1-4: Prefix (assigned to LOU)
    - Characters 5-6: Reserved (00)
    - Characters 7-18: Entity-specific (alphanumeric)
    - Characters 19-20: Check digits (ISO 7064 Mod 97-10)

References:
    - ISO 17442: https://www.iso.org/standard/78829.html
    - GLEIF: https://www.gleif.org/en/about-lei/iso-17442-the-lei-code-structure
    - MiFIR Article 26: Transaction reporting requirements
"""

from __future__ import annotations

import re
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class LEIStatus(str, Enum):
    """
    LEI Registration Status per GLEIF taxonomy.

    Reference: https://www.gleif.org/en/about-lei/common-data-file-format
    """

    ISSUED = "ISSUED"
    LAPSED = "LAPSED"
    MERGED = "MERGED"
    RETIRED = "RETIRED"
    DUPLICATE = "DUPLICATE"
    ANNULLED = "ANNULLED"
    PENDING_TRANSFER = "PENDING_TRANSFER"
    PENDING_ARCHIVAL = "PENDING_ARCHIVAL"
    TRANSFERRED = "TRANSFERRED"

    @property
    def is_valid_for_trading(self) -> bool:
        """Check if status allows trading per MiFID II requirements."""
        return self in (
            LEIStatus.ISSUED,
            LEIStatus.PENDING_TRANSFER,
            LEIStatus.PENDING_ARCHIVAL,
        )


@dataclass(frozen=True)
class LEIRecord:
    """
    Legal Entity Identifier record with registration details.

    Attributes:
        lei: 20-character LEI code (ISO 17442 format)
        legal_name: Legal name of the entity
        country: ISO 3166-1 alpha-2 country code
        registration_date: Initial LEI registration date
        next_renewal_date: Next annual renewal due date
        status: Current LEI registration status
        managing_lou: Local Operating Unit managing this LEI
        last_update: Last record update timestamp
    """

    lei: str
    legal_name: str = ""
    country: str = ""
    registration_date: Optional[date] = None
    next_renewal_date: Optional[date] = None
    status: LEIStatus = LEIStatus.ISSUED
    managing_lou: str = ""
    last_update: Optional[datetime] = None
    entity_category: str = ""
    registration_authority_id: str = ""
    jurisdiction: str = ""

    def is_valid(self) -> bool:
        """
        Check if LEI is valid for trading.

        Returns True if:
            - Status allows trading (ISSUED, PENDING_TRANSFER, PENDING_ARCHIVAL)
            - Not expired (or no expiry date set)
        """
        return self.status.is_valid_for_trading and not self.is_expired()

    def is_expired(self) -> bool:
        """Check if LEI has passed its renewal date."""
        if self.next_renewal_date is None:
            return False
        return self.next_renewal_date < date.today()

    def days_until_renewal(self) -> Optional[int]:
        """Calculate days until renewal is required."""
        if self.next_renewal_date is None:
            return None
        delta = self.next_renewal_date - date.today()
        return delta.days

    def needs_renewal_warning(self, warning_days: int = 30) -> bool:
        """Check if renewal warning should be issued."""
        days = self.days_until_renewal()
        if days is None:
            return False
        return 0 < days <= warning_days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lei": self.lei,
            "legal_name": self.legal_name,
            "country": self.country,
            "registration_date": self.registration_date.isoformat() if self.registration_date else None,
            "next_renewal_date": self.next_renewal_date.isoformat() if self.next_renewal_date else None,
            "status": self.status.value,
            "managing_lou": self.managing_lou,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "entity_category": self.entity_category,
            "registration_authority_id": self.registration_authority_id,
            "jurisdiction": self.jurisdiction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LEIRecord":
        """Create LEIRecord from dictionary."""
        reg_date = data.get("registration_date")
        if isinstance(reg_date, str):
            reg_date = date.fromisoformat(reg_date)

        renewal_date = data.get("next_renewal_date")
        if isinstance(renewal_date, str):
            renewal_date = date.fromisoformat(renewal_date)

        last_update = data.get("last_update")
        if isinstance(last_update, str):
            last_update = datetime.fromisoformat(last_update)

        status_str = data.get("status", "ISSUED")
        try:
            status = LEIStatus(status_str)
        except ValueError:
            status = LEIStatus.ISSUED

        return cls(
            lei=data.get("lei", ""),
            legal_name=data.get("legal_name", ""),
            country=data.get("country", ""),
            registration_date=reg_date,
            next_renewal_date=renewal_date,
            status=status,
            managing_lou=data.get("managing_lou", ""),
            last_update=last_update,
            entity_category=data.get("entity_category", ""),
            registration_authority_id=data.get("registration_authority_id", ""),
            jurisdiction=data.get("jurisdiction", ""),
        )


@dataclass
class LEIValidationResult:
    """
    Result of LEI validation check.

    Used for pre-trade validation to ensure "No LEI, No Trade" compliance.
    """

    is_valid: bool
    lei: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    record: Optional[LEIRecord] = None
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def __bool__(self) -> bool:
        return self.is_valid

    @property
    def allows_trading(self) -> bool:
        """Check if validation result allows trade execution."""
        return self.is_valid and (self.record is None or self.record.is_valid())


class LEIManager:
    """
    MiFID II LEI Management and Validation.

    Provides:
        - LEI format validation (ISO 17442)
        - LEI checksum verification (Mod 97-10)
        - Integration with GLEIF API for verification
        - Caching for performance
        - Pre-trade validation

    Example:
        manager = LEIManager()

        # Validate LEI format
        if manager.validate_format("5493001KJTIIGC8Y1R12"):
            # Check before trading
            result = manager.check_before_trade("5493001KJTIIGC8Y1R12")
            if result.allows_trading:
                # Proceed with trade
                pass
    """

    # LEI pattern: 18 alphanumeric + 2 check digits
    LEI_PATTERN = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")

    # Character to numeric mapping for Mod 97-10 calculation
    CHAR_MAP = {chr(ord('A') + i): str(10 + i) for i in range(26)}

    def __init__(
        self,
        own_lei: str = "",
        gleif_client: Optional[Any] = None,
        cache_ttl_hours: int = 24,
        verify_before_trade: bool = True,
        renewal_warning_days: int = 30,
        allow_pending_status: bool = True,
    ):
        """
        Initialize LEI Manager.

        Args:
            own_lei: The firm's own LEI for proprietary trading.
            gleif_client: Optional GLEIFClient for API verification.
            cache_ttl_hours: Cache TTL for verified LEIs.
            verify_before_trade: Whether to verify LEI before each trade.
            renewal_warning_days: Days before expiry to warn.
            allow_pending_status: Allow PENDING_* statuses for trading.
        """
        self._own_lei = own_lei
        self._gleif_client = gleif_client
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._verify_before_trade = verify_before_trade
        self._renewal_warning_days = renewal_warning_days
        self._allow_pending_status = allow_pending_status

        # Thread-safe cache
        self._cache: Dict[str, Tuple[LEIRecord, datetime]] = {}
        self._cache_lock = Lock()

        # Validation statistics
        self._stats = {
            "validations_total": 0,
            "validations_passed": 0,
            "validations_failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
        }

        logger.info(
            "LEIManager initialized",
            extra={
                "own_lei": self._own_lei[:8] + "..." if self._own_lei else "not_set",
                "cache_ttl_hours": cache_ttl_hours,
                "verify_before_trade": verify_before_trade,
            },
        )

    @property
    def own_lei(self) -> str:
        """Get the firm's own LEI."""
        return self._own_lei

    def get_own_lei(self) -> str:
        """Get the firm's own LEI (for compatibility)."""
        return self._own_lei

    def set_own_lei(self, lei: str) -> None:
        """Set the firm's own LEI."""
        if lei and not self.validate_format(lei):
            raise ValueError(f"Invalid LEI format: {lei}")
        self._own_lei = lei
        logger.info(f"Own LEI updated: {lei[:8]}..." if lei else "Own LEI cleared")

    def validate_format(self, lei: str) -> bool:
        """
        Validate LEI format per ISO 17442.

        Args:
            lei: LEI string to validate.

        Returns:
            True if format is valid (18 alphanumeric + 2 digits).

        Note:
            This only validates format, not checksum or GLEIF registration.
        """
        if not lei or not isinstance(lei, str):
            return False
        return bool(self.LEI_PATTERN.match(lei.upper()))

    def validate_checksum(self, lei: str) -> bool:
        """
        Validate LEI checksum using ISO 7064 Mod 97-10 algorithm.

        The same algorithm is used for IBAN validation.

        Args:
            lei: LEI string to validate.

        Returns:
            True if checksum is valid.

        Reference:
            - ISO 7064: https://en.wikipedia.org/wiki/ISO_7064
        """
        if not self.validate_format(lei):
            return False

        lei = lei.upper()

        # Convert letters to numbers (A=10, B=11, ..., Z=35)
        numeric = ""
        for char in lei:
            if char.isalpha():
                numeric += self.CHAR_MAP[char]
            else:
                numeric += char

        # Mod 97-10 check: remainder should be 1
        try:
            remainder = int(numeric) % 97
            return remainder == 1
        except ValueError:
            return False

    def calculate_check_digits(self, lei_base: str) -> str:
        """
        Calculate check digits for LEI base (first 18 characters).

        Args:
            lei_base: First 18 characters of LEI.

        Returns:
            2-digit check code.
        """
        if len(lei_base) != 18:
            raise ValueError("LEI base must be exactly 18 characters")

        lei_base = lei_base.upper()

        # Append 00 to calculate check digits
        test_lei = lei_base + "00"

        # Convert to numeric
        numeric = ""
        for char in test_lei:
            if char.isalpha():
                numeric += self.CHAR_MAP[char]
            else:
                numeric += char

        # Calculate check digits: 98 - (base mod 97)
        remainder = int(numeric) % 97
        check = 98 - remainder

        return f"{check:02d}"

    def validate(self, lei: str) -> LEIValidationResult:
        """
        Comprehensive LEI validation including format and checksum.

        Args:
            lei: LEI string to validate.

        Returns:
            LEIValidationResult with validation status.
        """
        self._stats["validations_total"] += 1

        if not lei:
            self._stats["validations_failed"] += 1
            return LEIValidationResult(
                is_valid=False,
                lei="",
                error_code="EMPTY_LEI",
                error_message="LEI cannot be empty",
            )

        lei = lei.upper().strip()

        # Format validation
        if not self.validate_format(lei):
            self._stats["validations_failed"] += 1
            return LEIValidationResult(
                is_valid=False,
                lei=lei,
                error_code="INVALID_FORMAT",
                error_message=f"Invalid LEI format: expected 18 alphanumeric + 2 digits",
            )

        # Checksum validation
        if not self.validate_checksum(lei):
            self._stats["validations_failed"] += 1
            return LEIValidationResult(
                is_valid=False,
                lei=lei,
                error_code="INVALID_CHECKSUM",
                error_message="LEI checksum validation failed (ISO 7064 Mod 97-10)",
            )

        self._stats["validations_passed"] += 1
        return LEIValidationResult(
            is_valid=True,
            lei=lei,
        )

    def get_cached(self, lei: str) -> Optional[LEIRecord]:
        """
        Get LEI record from cache if valid.

        Args:
            lei: LEI to look up.

        Returns:
            LEIRecord if cached and not expired, None otherwise.
        """
        lei = lei.upper().strip()

        with self._cache_lock:
            if lei in self._cache:
                record, cached_at = self._cache[lei]
                if datetime.utcnow() - cached_at < self._cache_ttl:
                    self._stats["cache_hits"] += 1
                    return record
                else:
                    # Expired - remove from cache
                    del self._cache[lei]

        self._stats["cache_misses"] += 1
        return None

    def cache_record(self, record: LEIRecord) -> None:
        """
        Cache an LEI record.

        Args:
            record: LEIRecord to cache.
        """
        if not record or not record.lei:
            return

        lei = record.lei.upper().strip()

        with self._cache_lock:
            self._cache[lei] = (record, datetime.utcnow())

    def clear_cache(self) -> int:
        """
        Clear all cached LEI records.

        Returns:
            Number of entries cleared.
        """
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def verify_with_gleif(self, lei: str) -> Optional[LEIRecord]:
        """
        Verify LEI against GLEIF database.

        Args:
            lei: LEI to verify.

        Returns:
            LEIRecord if found and verified, None otherwise.

        Note:
            Requires GLEIFClient to be configured.
        """
        if not self._gleif_client:
            logger.warning("GLEIF client not configured, skipping API verification")
            return None

        # Check cache first
        cached = self.get_cached(lei)
        if cached:
            return cached

        self._stats["api_calls"] += 1

        try:
            # Call GLEIF API
            response = await self._gleif_client.get_lei(lei)
            if response and response.is_success:
                record = response.to_lei_record()
                self.cache_record(record)
                return record
        except Exception as e:
            logger.error(f"GLEIF API error for {lei}: {e}")

        return None

    def check_before_trade(
        self,
        lei: Optional[str] = None,
        allow_pending: Optional[bool] = None,
    ) -> LEIValidationResult:
        """
        Pre-trade LEI check per MiFID II "No LEI, No Trade" requirement.

        Args:
            lei: LEI to check. If None, uses own_lei.
            allow_pending: Override for allowing PENDING_* statuses.

        Returns:
            LEIValidationResult indicating if trading is allowed.

        Example:
            result = manager.check_before_trade()
            if not result.allows_trading:
                raise ComplianceError(f"Trading blocked: {result.error_message}")
        """
        lei = lei or self._own_lei

        if not lei:
            return LEIValidationResult(
                is_valid=False,
                lei="",
                error_code="NO_LEI",
                error_message="No LEI configured - trading not allowed per MiFID II",
            )

        # Basic validation
        result = self.validate(lei)
        if not result.is_valid:
            return result

        # Check cached record for status
        cached = self.get_cached(lei)
        if cached:
            allow = allow_pending if allow_pending is not None else self._allow_pending_status

            if not cached.status.is_valid_for_trading:
                return LEIValidationResult(
                    is_valid=False,
                    lei=lei,
                    error_code="INVALID_STATUS",
                    error_message=f"LEI status {cached.status.value} does not allow trading",
                    record=cached,
                )

            if cached.is_expired():
                return LEIValidationResult(
                    is_valid=False,
                    lei=lei,
                    error_code="EXPIRED",
                    error_message=f"LEI expired on {cached.next_renewal_date}",
                    record=cached,
                )

            # Check for renewal warning
            if cached.needs_renewal_warning(self._renewal_warning_days):
                logger.warning(
                    f"LEI renewal due in {cached.days_until_renewal()} days",
                    extra={"lei": lei, "renewal_date": str(cached.next_renewal_date)},
                )

            return LEIValidationResult(
                is_valid=True,
                lei=lei,
                record=cached,
            )

        # No cached record - return basic validation result
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        with self._cache_lock:
            cache_size = len(self._cache)

        return {
            **self._stats,
            "cache_size": cache_size,
            "own_lei_configured": bool(self._own_lei),
        }

    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate LEI validation report for compliance documentation.

        Returns:
            Dictionary with validation report data.
        """
        stats = self.get_statistics()
        own_lei_valid = self.validate(self._own_lei) if self._own_lei else None

        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "own_lei": {
                "lei": self._own_lei[:8] + "..." if self._own_lei else None,
                "is_valid": own_lei_valid.is_valid if own_lei_valid else False,
                "validation_error": own_lei_valid.error_message if own_lei_valid and not own_lei_valid.is_valid else None,
            },
            "statistics": stats,
            "configuration": {
                "cache_ttl_hours": self._cache_ttl.total_seconds() / 3600,
                "verify_before_trade": self._verify_before_trade,
                "renewal_warning_days": self._renewal_warning_days,
                "allow_pending_status": self._allow_pending_status,
            },
        }


def create_lei_manager(
    own_lei: str = "",
    gleif_client: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> LEIManager:
    """
    Factory function to create LEIManager instance.

    Args:
        own_lei: The firm's own LEI.
        gleif_client: Optional GLEIFClient for API verification.
        config: Optional configuration dictionary.

    Returns:
        Configured LEIManager instance.
    """
    config = config or {}

    return LEIManager(
        own_lei=own_lei or config.get("own_lei", ""),
        gleif_client=gleif_client,
        cache_ttl_hours=config.get("cache_ttl_hours", 24),
        verify_before_trade=config.get("verify_before_trade", True),
        renewal_warning_days=config.get("renewal_warning_days", 30),
        allow_pending_status=config.get("allow_pending_status", True),
    )


__all__ = [
    "LEIStatus",
    "LEIRecord",
    "LEIValidationResult",
    "LEIManager",
    "create_lei_manager",
]
