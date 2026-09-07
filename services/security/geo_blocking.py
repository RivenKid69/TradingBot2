"""
Geographic access restrictions for sanctioned jurisdictions.

Implements compliance with international sanctions by blocking access from
countries subject to comprehensive trade restrictions.

References:
    - EU Council Regulations (sanctions programs)
    - OFAC Sanctions Programs (US)
    - UK Financial Sanctions (OFSI)
    - UN Security Council Sanctions

Blocked Countries (as of 2024):
    - Cuba (CU): OFAC Comprehensive Sanctions
    - Iran (IR): OFAC Comprehensive Sanctions
    - North Korea (KP): OFAC/UN Comprehensive Sanctions
    - Syria (SY): OFAC/EU Comprehensive Sanctions
    - Russia (RU): EU/UK Comprehensive Sanctions (since 2022)
    - Belarus (BY): EU Sanctions

Note: This list should be reviewed periodically as sanctions change.
Last reviewed: December 2024

Example:
    >>> service = GeoBlockingService(geoip_provider)
    >>> result = service.check_ip("1.2.3.4")
    >>> if not result.allowed:
    ...     raise AccessDeniedError(result.block_reason.value)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set

logger = logging.getLogger("security.geo_blocking")


class BlockReason(Enum):
    """Reasons for blocking access."""

    OFAC_SANCTIONS = "US OFAC Comprehensive Sanctions"
    EU_SANCTIONS = "EU Council Sanctions"
    UK_SANCTIONS = "UK Financial Sanctions (OFSI)"
    UN_SANCTIONS = "UN Security Council Sanctions"
    PLATFORM_POLICY = "Platform Policy Restriction"
    HIGH_RISK_JURISDICTION = "High Risk Jurisdiction"


@dataclass
class Country:
    """Country information from GeoIP lookup."""

    code: str  # ISO 3166-1 alpha-2
    name: str
    continent: Optional[str] = None
    is_in_eu: bool = False


@dataclass
class GeoCheckResult:
    """Result of a geographic access check."""

    allowed: bool
    country_code: Optional[str]
    country_name: Optional[str] = None
    block_reason: Optional[BlockReason] = None
    checked_at: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed": self.allowed,
            "country_code": self.country_code,
            "country_name": self.country_name,
            "block_reason": self.block_reason.value if self.block_reason else None,
            "checked_at": self.checked_at.isoformat(),
            "metadata": self.metadata,
        }


class GeoIPProviderProtocol(Protocol):
    """Protocol for GeoIP lookup providers."""

    def lookup(self, ip_address: str) -> Country:
        """
        Look up country for an IP address.

        Args:
            ip_address: IP address to look up

        Returns:
            Country information
        """
        ...


class MockGeoIPProvider:
    """
    Mock GeoIP provider for testing.

    In production, use MaxMind GeoIP2 or IP2Location.
    """

    def __init__(self, default_country: str = "US"):
        self._default_country = default_country
        self._ip_map: Dict[str, Country] = {}

        # Add some default mappings
        self._country_data = {
            "US": Country(code="US", name="United States", continent="NA", is_in_eu=False),
            "GB": Country(code="GB", name="United Kingdom", continent="EU", is_in_eu=False),
            "DE": Country(code="DE", name="Germany", continent="EU", is_in_eu=True),
            "FR": Country(code="FR", name="France", continent="EU", is_in_eu=True),
            "NL": Country(code="NL", name="Netherlands", continent="EU", is_in_eu=True),
            "RU": Country(code="RU", name="Russia", continent="EU", is_in_eu=False),
            "IR": Country(code="IR", name="Iran", continent="AS", is_in_eu=False),
            "KP": Country(code="KP", name="North Korea", continent="AS", is_in_eu=False),
            "SY": Country(code="SY", name="Syria", continent="AS", is_in_eu=False),
            "CU": Country(code="CU", name="Cuba", continent="NA", is_in_eu=False),
            "BY": Country(code="BY", name="Belarus", continent="EU", is_in_eu=False),
        }

    def set_ip_country(self, ip_address: str, country_code: str) -> None:
        """Set country for a specific IP (for testing)."""
        if country_code in self._country_data:
            self._ip_map[ip_address] = self._country_data[country_code]
        else:
            self._ip_map[ip_address] = Country(code=country_code, name=country_code)

    def lookup(self, ip_address: str) -> Country:
        """Look up country for an IP address."""
        if ip_address in self._ip_map:
            return self._ip_map[ip_address]

        # Return default country for unknown IPs
        return self._country_data.get(
            self._default_country, Country(code=self._default_country, name=self._default_country)
        )


# ISO 3166-1 alpha-2 country codes for blocked countries
BLOCKED_COUNTRIES: Dict[str, BlockReason] = {
    # OFAC Comprehensive Sanctions
    "CU": BlockReason.OFAC_SANCTIONS,  # Cuba
    "IR": BlockReason.OFAC_SANCTIONS,  # Iran
    "KP": BlockReason.OFAC_SANCTIONS,  # North Korea
    "SY": BlockReason.OFAC_SANCTIONS,  # Syria
    # EU Comprehensive Sanctions (post-2022)
    "RU": BlockReason.EU_SANCTIONS,  # Russia
    "BY": BlockReason.EU_SANCTIONS,  # Belarus
}

# High-risk jurisdictions (not blocked, but flagged for enhanced due diligence)
HIGH_RISK_COUNTRIES: Set[str] = {
    "AF",  # Afghanistan
    "MM",  # Myanmar
    "VE",  # Venezuela
    "YE",  # Yemen
    "ZW",  # Zimbabwe
    "LY",  # Libya
    "SD",  # Sudan
    "SS",  # South Sudan
    "SO",  # Somalia
    "CD",  # Democratic Republic of Congo
}


class GeoBlockingService:
    """
    Geographic access restriction service.

    Features:
        - IP-based country detection
        - Sanctions compliance (OFAC, EU, UK, UN)
        - High-risk jurisdiction flagging
        - Logging for compliance audits
        - Configurable block lists

    Example:
        >>> geoip = MaxMindProvider(license_key="xxx")
        >>> service = GeoBlockingService(geoip)
        >>> result = service.check_ip("1.2.3.4")
        >>> if not result.allowed:
        ...     log_blocked_access(result)
        ...     raise AccessDeniedError("Access from your location is restricted")
    """

    def __init__(
        self,
        geoip_provider: GeoIPProviderProtocol,
        additional_blocked: Optional[Dict[str, BlockReason]] = None,
        additional_high_risk: Optional[Set[str]] = None,
    ):
        """
        Initialize the geo-blocking service.

        Args:
            geoip_provider: GeoIP lookup provider
            additional_blocked: Additional countries to block
            additional_high_risk: Additional high-risk countries
        """
        self._geoip = geoip_provider

        # Combine default and additional blocked countries
        self._blocked = {**BLOCKED_COUNTRIES}
        if additional_blocked:
            self._blocked.update(additional_blocked)

        # Combine default and additional high-risk countries
        self._high_risk = HIGH_RISK_COUNTRIES.copy()
        if additional_high_risk:
            self._high_risk.update(additional_high_risk)

    def check_ip(self, ip_address: str) -> GeoCheckResult:
        """
        Check if an IP address is from an allowed jurisdiction.

        Args:
            ip_address: IP address to check

        Returns:
            GeoCheckResult with allowed status and details
        """
        try:
            country = self._geoip.lookup(ip_address)

            if country.code in self._blocked:
                result = GeoCheckResult(
                    allowed=False,
                    country_code=country.code,
                    country_name=country.name,
                    block_reason=self._blocked[country.code],
                    metadata={
                        "ip_address": ip_address,
                        "is_high_risk": country.code in self._high_risk,
                    },
                )
                logger.warning(
                    f"GEO_BLOCKED | ip={ip_address} | "
                    f"country={country.code} | "
                    f"reason={self._blocked[country.code].value}"
                )
                return result

            # Allowed, but check if high-risk
            is_high_risk = country.code in self._high_risk
            if is_high_risk:
                logger.info(f"GEO_HIGH_RISK | ip={ip_address} | " f"country={country.code}")

            return GeoCheckResult(
                allowed=True,
                country_code=country.code,
                country_name=country.name,
                metadata={
                    "ip_address": ip_address,
                    "is_high_risk": is_high_risk,
                    "is_eu": country.is_in_eu,
                },
            )

        except Exception as e:
            # Fail-open for unknown IPs, but log for review
            logger.error(f"GEO_LOOKUP_FAILED | ip={ip_address} | error={str(e)}")
            return GeoCheckResult(
                allowed=True,
                country_code=None,
                metadata={
                    "ip_address": ip_address,
                    "lookup_failed": True,
                    "error": str(e),
                },
            )

    def check_registration(self, ip_address: str, declared_country: str) -> GeoCheckResult:
        """
        Check both IP and declared country during registration.

        Block if either the IP location or declared country is sanctioned.

        Args:
            ip_address: IP address of the user
            declared_country: Country code declared by user

        Returns:
            GeoCheckResult with combined check result
        """
        # First check IP
        ip_check = self.check_ip(ip_address)

        if not ip_check.allowed:
            return ip_check

        # Then check declared country
        if declared_country.upper() in self._blocked:
            result = GeoCheckResult(
                allowed=False,
                country_code=declared_country.upper(),
                block_reason=self._blocked[declared_country.upper()],
                metadata={
                    "ip_address": ip_address,
                    "ip_country": ip_check.country_code,
                    "declared_country": declared_country.upper(),
                    "mismatch": ip_check.country_code != declared_country.upper(),
                },
            )
            logger.warning(
                f"GEO_BLOCKED_DECLARED | ip={ip_address} | "
                f"declared={declared_country.upper()} | "
                f"reason={self._blocked[declared_country.upper()].value}"
            )
            return result

        # Check for country mismatch (potential evasion)
        if ip_check.country_code and ip_check.country_code != declared_country.upper():
            logger.info(
                f"GEO_MISMATCH | ip={ip_address} | "
                f"ip_country={ip_check.country_code} | "
                f"declared={declared_country.upper()}"
            )
            ip_check.metadata["declared_country"] = declared_country.upper()
            ip_check.metadata["mismatch"] = True

        return GeoCheckResult(
            allowed=True,
            country_code=declared_country.upper(),
            metadata={
                "ip_address": ip_address,
                "ip_country": ip_check.country_code,
                "declared_country": declared_country.upper(),
                "mismatch": (
                    ip_check.country_code != declared_country.upper()
                    if ip_check.country_code
                    else False
                ),
                "is_high_risk": declared_country.upper() in self._high_risk,
            },
        )

    def is_blocked_country(self, country_code: str) -> bool:
        """
        Check if a country code is blocked.

        Args:
            country_code: ISO 3166-1 alpha-2 country code

        Returns:
            True if the country is blocked
        """
        return country_code.upper() in self._blocked

    def is_high_risk_country(self, country_code: str) -> bool:
        """
        Check if a country code is high-risk.

        Args:
            country_code: ISO 3166-1 alpha-2 country code

        Returns:
            True if the country is high-risk
        """
        return country_code.upper() in self._high_risk

    def get_blocked_countries(self) -> Dict[str, str]:
        """
        Get list of blocked countries.

        Returns:
            Dictionary mapping country codes to block reasons
        """
        return {code: reason.value for code, reason in self._blocked.items()}

    def get_high_risk_countries(self) -> Set[str]:
        """
        Get list of high-risk countries.

        Returns:
            Set of high-risk country codes
        """
        return self._high_risk.copy()

    def add_blocked_country(self, country_code: str, reason: BlockReason) -> None:
        """
        Add a country to the blocked list.

        Args:
            country_code: ISO 3166-1 alpha-2 country code
            reason: Reason for blocking
        """
        self._blocked[country_code.upper()] = reason
        logger.info(f"GEO_BLOCK_ADDED | country={country_code.upper()} | " f"reason={reason.value}")

    def remove_blocked_country(self, country_code: str) -> bool:
        """
        Remove a country from the blocked list.

        Args:
            country_code: ISO 3166-1 alpha-2 country code

        Returns:
            True if country was removed, False if not found
        """
        if country_code.upper() in self._blocked:
            del self._blocked[country_code.upper()]
            logger.info(f"GEO_BLOCK_REMOVED | country={country_code.upper()}")
            return True
        return False

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate a compliance report for the geo-blocking configuration.

        Returns:
            Report dictionary
        """
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "blocked_countries": {
                code: {
                    "reason": reason.value,
                    "sanctions_programs": self._get_sanctions_programs(reason),
                }
                for code, reason in self._blocked.items()
            },
            "high_risk_countries": list(self._high_risk),
            "total_blocked": len(self._blocked),
            "total_high_risk": len(self._high_risk),
            "references": [
                "OFAC Sanctions Programs: https://ofac.treasury.gov/sanctions-programs-and-country-information",
                "EU Sanctions Map: https://www.sanctionsmap.eu/",
                "UK Financial Sanctions: https://www.gov.uk/government/publications/financial-sanctions-consolidated-list-of-targets",
            ],
        }

    def _get_sanctions_programs(self, reason: BlockReason) -> List[str]:
        """Get applicable sanctions programs for a block reason."""
        programs = {
            BlockReason.OFAC_SANCTIONS: [
                "OFAC Comprehensive Sanctions",
                "Cuban Assets Control Regulations",
                "Iranian Transactions and Sanctions Regulations",
                "North Korea Sanctions Regulations",
                "Syrian Sanctions Regulations",
            ],
            BlockReason.EU_SANCTIONS: [
                "EU Council Regulation 833/2014 (Russia)",
                "EU Council Regulation 765/2006 (Belarus)",
            ],
            BlockReason.UK_SANCTIONS: [
                "Russia (Sanctions) (EU Exit) Regulations 2019",
            ],
            BlockReason.UN_SANCTIONS: [
                "UN Security Council Resolutions",
            ],
        }
        return programs.get(reason, [])
