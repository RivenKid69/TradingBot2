# -*- coding: utf-8 -*-
"""
GLEIF (Global Legal Entity Identifier Foundation) API Client.

Provides integration with GLEIF's public API for LEI verification
and lookup as required by MiFID II transaction reporting.

API Documentation:
    - GLEIF API v1: https://www.gleif.org/en/lei-data/gleif-lei-look-up-api
    - API Docs: https://api.gleif.org/api/v1/documentation
    - Rate Limits: 60 requests/minute for public API

References:
    - GLEIF: https://www.gleif.org/
    - LEI Search: https://search.gleif.org/
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class GLEIFErrorCode(str, Enum):
    """GLEIF API error codes."""

    NOT_FOUND = "NOT_FOUND"
    INVALID_LEI = "INVALID_LEI"
    RATE_LIMITED = "RATE_LIMITED"
    API_ERROR = "API_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT = "TIMEOUT"
    PARSE_ERROR = "PARSE_ERROR"


@dataclass
class GLEIFError(Exception):
    """
    GLEIF API error with structured details.

    Attributes:
        code: Error code from GLEIFErrorCode enum.
        message: Human-readable error message.
        lei: LEI that caused the error (if applicable).
        http_status: HTTP status code (if applicable).
        response_body: Raw response body (for debugging).
    """

    code: GLEIFErrorCode
    message: str
    lei: str = ""
    http_status: Optional[int] = None
    response_body: Optional[str] = None

    def __str__(self) -> str:
        return f"GLEIFError({self.code.value}): {self.message}"


@dataclass
class GLEIFEntity:
    """
    Entity information from GLEIF API.

    Maps to GLEIF Level 1 data (Legal Entity Reference Data).
    """

    lei: str
    legal_name: str
    legal_address_city: str = ""
    legal_address_country: str = ""
    legal_address_postal_code: str = ""
    headquarters_city: str = ""
    headquarters_country: str = ""
    registration_authority_id: str = ""
    registration_authority_entity_id: str = ""
    entity_category: str = ""
    entity_sub_category: str = ""
    legal_form: str = ""
    jurisdiction: str = ""


@dataclass
class GLEIFRegistration:
    """
    Registration information from GLEIF API.

    Contains LEI lifecycle and management data.
    """

    initial_registration_date: Optional[date] = None
    last_update_date: Optional[datetime] = None
    next_renewal_date: Optional[date] = None
    status: str = "ISSUED"
    managing_lou: str = ""
    validation_sources: str = ""
    corroboration_level: str = ""


@dataclass
class GLEIFResponse:
    """
    Complete GLEIF API response for a single LEI.

    Combines entity data, registration info, and relationship data.
    """

    lei: str
    is_success: bool
    entity: Optional[GLEIFEntity] = None
    registration: Optional[GLEIFRegistration] = None
    error: Optional[GLEIFError] = None
    raw_data: Optional[Dict[str, Any]] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    def to_lei_record(self) -> Optional["LEIRecord"]:
        """
        Convert GLEIF response to LEIRecord.

        Returns:
            LEIRecord if response is successful, None otherwise.
        """
        if not self.is_success or not self.entity:
            return None

        # Import here to avoid circular dependency
        from services.compliance.lei_manager import LEIRecord, LEIStatus

        try:
            status = LEIStatus(self.registration.status if self.registration else "ISSUED")
        except ValueError:
            status = LEIStatus.ISSUED

        return LEIRecord(
            lei=self.lei,
            legal_name=self.entity.legal_name,
            country=self.entity.legal_address_country,
            registration_date=self.registration.initial_registration_date if self.registration else None,
            next_renewal_date=self.registration.next_renewal_date if self.registration else None,
            status=status,
            managing_lou=self.registration.managing_lou if self.registration else "",
            last_update=self.registration.last_update_date if self.registration else None,
            entity_category=self.entity.entity_category,
            registration_authority_id=self.entity.registration_authority_id,
            jurisdiction=self.entity.jurisdiction,
        )


class GLEIFClient:
    """
    GLEIF API Client for LEI verification.

    Provides synchronous and asynchronous methods for:
        - Single LEI lookup
        - Batch LEI lookup
        - LEI search by name
        - Rate limiting compliance

    Example:
        client = GLEIFClient()

        # Async lookup
        response = await client.get_lei("5493001KJTIIGC8Y1R12")
        if response.is_success:
            print(f"Legal name: {response.entity.legal_name}")

        # Sync lookup
        response = client.get_lei_sync("5493001KJTIIGC8Y1R12")

    Rate Limits:
        - Public API: 60 requests/minute
        - Client implements automatic rate limiting
    """

    DEFAULT_BASE_URL = "https://api.gleif.org/api/v1"
    DEFAULT_TIMEOUT = 30.0
    RATE_LIMIT_REQUESTS = 60
    RATE_LIMIT_WINDOW = 60.0  # seconds

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        enable_rate_limiting: bool = True,
    ):
        """
        Initialize GLEIF API client.

        Args:
            base_url: GLEIF API base URL.
            timeout: Request timeout in seconds.
            enable_rate_limiting: Enable automatic rate limiting.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._enable_rate_limiting = enable_rate_limiting

        # Rate limiting state
        self._request_timestamps: List[float] = []

        # Statistics
        self._stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "rate_limit_waits": 0,
        }

        logger.info(
            "GLEIFClient initialized",
            extra={"base_url": self._base_url, "rate_limiting": enable_rate_limiting},
        )

    def _check_rate_limit(self) -> float:
        """
        Check rate limit and return required wait time.

        Returns:
            Seconds to wait before next request (0 if no wait needed).
        """
        if not self._enable_rate_limiting:
            return 0.0

        now = time.time()
        window_start = now - self.RATE_LIMIT_WINDOW

        # Remove old timestamps
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > window_start
        ]

        if len(self._request_timestamps) >= self.RATE_LIMIT_REQUESTS:
            # Calculate wait time
            oldest = min(self._request_timestamps)
            wait_time = (oldest + self.RATE_LIMIT_WINDOW) - now
            return max(0.0, wait_time)

        return 0.0

    def _record_request(self) -> None:
        """Record a request for rate limiting."""
        self._request_timestamps.append(time.time())
        self._stats["requests_total"] += 1

    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string from GLEIF API."""
        if not date_str:
            return None
        try:
            # GLEIF uses ISO format: YYYY-MM-DD
            return date.fromisoformat(date_str[:10])
        except (ValueError, TypeError):
            return None

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string from GLEIF API."""
        if not dt_str:
            return None
        try:
            # Handle various ISO formats
            dt_str = dt_str.replace("Z", "+00:00")
            if "." in dt_str:
                # Truncate microseconds if too long
                base, frac = dt_str.split(".")
                frac_part = frac.split("+")[0].split("-")[0][:6]
                tz_part = "+" + dt_str.split("+")[1] if "+" in dt_str else ""
                if "-" in frac and not tz_part:
                    tz_part = "-" + dt_str.split("-")[-1]
                dt_str = f"{base}.{frac_part}{tz_part}"
            return datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            return None

    def _parse_response(self, data: Dict[str, Any]) -> GLEIFResponse:
        """Parse GLEIF API response into GLEIFResponse."""
        # GLEIF API returns array for single lookups
        records = data.get("data", [])
        if not records:
            return GLEIFResponse(
                lei="",
                is_success=False,
                error=GLEIFError(
                    code=GLEIFErrorCode.NOT_FOUND,
                    message="No LEI records found",
                ),
            )

        record = records[0] if isinstance(records, list) else records
        attributes = record.get("attributes", {})
        lei = attributes.get("lei", "")

        # Parse entity data
        entity_data = attributes.get("entity", {})
        legal_address = entity_data.get("legalAddress", {})
        headquarters = entity_data.get("headquartersAddress", {})
        registration_authority = entity_data.get("registeredAs", {})

        entity = GLEIFEntity(
            lei=lei,
            legal_name=entity_data.get("legalName", {}).get("name", ""),
            legal_address_city=legal_address.get("city", ""),
            legal_address_country=legal_address.get("country", ""),
            legal_address_postal_code=legal_address.get("postalCode", ""),
            headquarters_city=headquarters.get("city", ""),
            headquarters_country=headquarters.get("country", ""),
            registration_authority_id=registration_authority.get("registrationAuthorityID", ""),
            registration_authority_entity_id=registration_authority.get("registrationAuthorityEntityID", ""),
            entity_category=entity_data.get("category", ""),
            entity_sub_category=entity_data.get("subCategory", ""),
            legal_form=entity_data.get("legalForm", {}).get("id", ""),
            jurisdiction=entity_data.get("jurisdiction", ""),
        )

        # Parse registration data
        reg_data = attributes.get("registration", {})
        registration = GLEIFRegistration(
            initial_registration_date=self._parse_date(reg_data.get("initialRegistrationDate")),
            last_update_date=self._parse_datetime(reg_data.get("lastUpdateDate")),
            next_renewal_date=self._parse_date(reg_data.get("nextRenewalDate")),
            status=reg_data.get("status", "ISSUED"),
            managing_lou=reg_data.get("managingLou", ""),
            validation_sources=reg_data.get("validationSources", ""),
            corroboration_level=reg_data.get("corroborationLevel", ""),
        )

        return GLEIFResponse(
            lei=lei,
            is_success=True,
            entity=entity,
            registration=registration,
            raw_data=data,
        )

    async def get_lei(self, lei: str) -> GLEIFResponse:
        """
        Look up a single LEI asynchronously.

        Args:
            lei: LEI to look up (20 characters).

        Returns:
            GLEIFResponse with entity and registration data.

        Raises:
            GLEIFError: On API or network errors.
        """
        lei = lei.upper().strip()

        # Check rate limit
        wait_time = self._check_rate_limit()
        if wait_time > 0:
            self._stats["rate_limit_waits"] += 1
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
            import asyncio
            await asyncio.sleep(wait_time)

        self._record_request()

        try:
            import httpx

            url = f"{self._base_url}/lei-records/{lei}"

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    url,
                    headers={"Accept": "application/vnd.api+json"},
                )

                if response.status_code == 404:
                    self._stats["requests_failed"] += 1
                    return GLEIFResponse(
                        lei=lei,
                        is_success=False,
                        error=GLEIFError(
                            code=GLEIFErrorCode.NOT_FOUND,
                            message=f"LEI not found: {lei}",
                            lei=lei,
                            http_status=404,
                        ),
                    )

                if response.status_code == 429:
                    self._stats["requests_failed"] += 1
                    return GLEIFResponse(
                        lei=lei,
                        is_success=False,
                        error=GLEIFError(
                            code=GLEIFErrorCode.RATE_LIMITED,
                            message="GLEIF API rate limit exceeded",
                            lei=lei,
                            http_status=429,
                        ),
                    )

                if response.status_code != 200:
                    self._stats["requests_failed"] += 1
                    return GLEIFResponse(
                        lei=lei,
                        is_success=False,
                        error=GLEIFError(
                            code=GLEIFErrorCode.API_ERROR,
                            message=f"API error: HTTP {response.status_code}",
                            lei=lei,
                            http_status=response.status_code,
                            response_body=response.text,
                        ),
                    )

                data = response.json()
                result = self._parse_response({"data": [data.get("data", {})]})
                self._stats["requests_success"] += 1
                return result

        except ImportError:
            # httpx not available
            logger.warning("httpx not installed, using sync fallback")
            return self.get_lei_sync(lei)

        except Exception as e:
            self._stats["requests_failed"] += 1
            logger.error(f"GLEIF API error: {e}")
            return GLEIFResponse(
                lei=lei,
                is_success=False,
                error=GLEIFError(
                    code=GLEIFErrorCode.NETWORK_ERROR,
                    message=str(e),
                    lei=lei,
                ),
            )

    def get_lei_sync(self, lei: str) -> GLEIFResponse:
        """
        Look up a single LEI synchronously.

        Args:
            lei: LEI to look up (20 characters).

        Returns:
            GLEIFResponse with entity and registration data.
        """
        lei = lei.upper().strip()

        # Check rate limit
        wait_time = self._check_rate_limit()
        if wait_time > 0:
            self._stats["rate_limit_waits"] += 1
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
            time.sleep(wait_time)

        self._record_request()

        try:
            import urllib.request
            import json

            url = f"{self._base_url}/lei-records/{lei}"
            request = urllib.request.Request(
                url,
                headers={"Accept": "application/vnd.api+json"},
            )

            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                if response.status != 200:
                    self._stats["requests_failed"] += 1
                    return GLEIFResponse(
                        lei=lei,
                        is_success=False,
                        error=GLEIFError(
                            code=GLEIFErrorCode.API_ERROR,
                            message=f"API error: HTTP {response.status}",
                            lei=lei,
                            http_status=response.status,
                        ),
                    )

                data = json.loads(response.read().decode("utf-8"))
                result = self._parse_response({"data": [data.get("data", {})]})
                self._stats["requests_success"] += 1
                return result

        except urllib.error.HTTPError as e:
            self._stats["requests_failed"] += 1
            if e.code == 404:
                return GLEIFResponse(
                    lei=lei,
                    is_success=False,
                    error=GLEIFError(
                        code=GLEIFErrorCode.NOT_FOUND,
                        message=f"LEI not found: {lei}",
                        lei=lei,
                        http_status=404,
                    ),
                )
            return GLEIFResponse(
                lei=lei,
                is_success=False,
                error=GLEIFError(
                    code=GLEIFErrorCode.API_ERROR,
                    message=str(e),
                    lei=lei,
                    http_status=e.code,
                ),
            )

        except Exception as e:
            self._stats["requests_failed"] += 1
            logger.error(f"GLEIF API error: {e}")
            return GLEIFResponse(
                lei=lei,
                is_success=False,
                error=GLEIFError(
                    code=GLEIFErrorCode.NETWORK_ERROR,
                    message=str(e),
                    lei=lei,
                ),
            )

    async def search_by_name(
        self,
        name: str,
        country: Optional[str] = None,
        limit: int = 10,
    ) -> List[GLEIFResponse]:
        """
        Search for LEIs by entity name.

        Args:
            name: Entity name to search for.
            country: Optional ISO country code filter.
            limit: Maximum results to return.

        Returns:
            List of GLEIFResponse matching the search.
        """
        # Check rate limit
        wait_time = self._check_rate_limit()
        if wait_time > 0:
            self._stats["rate_limit_waits"] += 1
            import asyncio
            await asyncio.sleep(wait_time)

        self._record_request()

        try:
            import httpx
            from urllib.parse import urlencode

            params = {
                "filter[entity.legalName]": name,
                "page[size]": min(limit, 100),
            }
            if country:
                params["filter[entity.legalAddress.country]"] = country

            url = f"{self._base_url}/lei-records?{urlencode(params)}"

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    url,
                    headers={"Accept": "application/vnd.api+json"},
                )

                if response.status_code != 200:
                    self._stats["requests_failed"] += 1
                    return []

                data = response.json()
                records = data.get("data", [])
                self._stats["requests_success"] += 1

                results = []
                for record in records[:limit]:
                    parsed = self._parse_response({"data": [record]})
                    if parsed.is_success:
                        results.append(parsed)

                return results

        except Exception as e:
            self._stats["requests_failed"] += 1
            logger.error(f"GLEIF search error: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self._stats,
            "current_rate_limit_queue": len(self._request_timestamps),
        }


def create_gleif_client(
    base_url: str = GLEIFClient.DEFAULT_BASE_URL,
    timeout: float = GLEIFClient.DEFAULT_TIMEOUT,
    enable_rate_limiting: bool = True,
) -> GLEIFClient:
    """
    Factory function to create GLEIFClient instance.

    Args:
        base_url: GLEIF API base URL.
        timeout: Request timeout in seconds.
        enable_rate_limiting: Enable automatic rate limiting.

    Returns:
        Configured GLEIFClient instance.
    """
    return GLEIFClient(
        base_url=base_url,
        timeout=timeout,
        enable_rate_limiting=enable_rate_limiting,
    )


__all__ = [
    "GLEIFErrorCode",
    "GLEIFError",
    "GLEIFEntity",
    "GLEIFRegistration",
    "GLEIFResponse",
    "GLEIFClient",
    "create_gleif_client",
]
