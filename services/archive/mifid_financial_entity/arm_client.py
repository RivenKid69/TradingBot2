# -*- coding: utf-8 -*-
"""
ARM (Approved Reporting Mechanism) Client Module.

Provides integration with ARMs for MiFIR transaction reporting per Article 26.
ARMs are authorized entities that submit transaction reports to NCAs on behalf
of investment firms.

Supported ARM Providers:
    - Bloomberg BTRL (Transaction Reporting)
    - CME TRAX
    - LSEG UnaVista
    - Tradeweb
    - Mock ARM (for testing)

Report Submission Flow:
    1. Investment firm creates TransactionReport
    2. Report is validated locally
    3. Report is converted to ARM format (typically ISO 20022 XML)
    4. ARM submits to NCA
    5. ARM returns confirmation/rejection
    6. Status is tracked for audit trail

References:
    - MiFIR Article 26: Transaction reporting obligation
    - ESMA ARM Register: https://www.esma.europa.eu/databases-library/registers-and-data/approved-reporting-mechanisms
    - RTS 22: Format and content of transaction reports
"""

from __future__ import annotations

import logging
import time
import uuid
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Callable
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations
# ============================================================================


class ARMProvider(str, Enum):
    """Supported ARM providers."""

    BLOOMBERG_BTRL = "bloomberg_btrl"
    CME_TRAX = "cme_trax"
    LSEG_UNAVISTA = "lseg_unavista"
    TRADEWEB = "tradeweb"
    MOCK = "mock"


class ARMEnvironment(str, Enum):
    """ARM environment types."""

    PRODUCTION = "production"
    UAT = "uat"
    SANDBOX = "sandbox"


class SubmissionStatus(str, Enum):
    """Status of a submitted report."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    ERROR = "error"


class ErrorCode(str, Enum):
    """ARM error codes."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    FORMAT_ERROR = "FORMAT_ERROR"
    DUPLICATE_REPORT = "DUPLICATE_REPORT"
    NCA_REJECTION = "NCA_REJECTION"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ARMError(Exception):
    """
    ARM submission error with structured details.

    Attributes:
        code: Error code from ErrorCode enum.
        message: Human-readable error message.
        report_id: ID of the report that failed (if applicable).
        details: Additional error details.
    """

    code: ErrorCode
    message: str
    report_id: str = ""
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"ARMError({self.code.value}): {self.message}"


@dataclass
class SubmissionResult:
    """
    Result of a transaction report submission.

    Attributes:
        report_id: Original report reference number.
        confirmation_id: ARM confirmation ID.
        status: Current submission status.
        submitted_at: Submission timestamp.
        nca_received_at: When NCA received the report (if known).
        errors: List of error messages (if any).
        warnings: List of warning messages.
        arm_provider: Which ARM processed the report.
    """

    report_id: str
    confirmation_id: str = ""
    status: SubmissionStatus = SubmissionStatus.PENDING
    submitted_at: Optional[datetime] = None
    nca_received_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    arm_provider: str = ""
    raw_response: Optional[Dict[str, Any]] = None

    def is_success(self) -> bool:
        """Check if submission was successful."""
        return self.status in (SubmissionStatus.SUBMITTED, SubmissionStatus.ACCEPTED)

    def is_final(self) -> bool:
        """Check if status is final (no more updates expected)."""
        return self.status in (
            SubmissionStatus.ACCEPTED,
            SubmissionStatus.REJECTED,
            SubmissionStatus.CANCELLED,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "confirmation_id": self.confirmation_id,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "nca_received_at": self.nca_received_at.isoformat() if self.nca_received_at else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "arm_provider": self.arm_provider,
        }


@dataclass
class BatchSubmissionResult:
    """
    Result of a batch submission.

    Attributes:
        batch_id: Unique batch identifier.
        total_reports: Number of reports in batch.
        submitted: Number successfully submitted.
        failed: Number that failed.
        results: Individual results for each report.
    """

    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_reports: int = 0
    submitted: int = 0
    failed: int = 0
    results: List[SubmissionResult] = field(default_factory=list)
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_reports == 0:
            return 0.0
        return (self.submitted / self.total_reports) * 100


@dataclass
class ARMClientConfig:
    """
    Configuration for ARM client.

    Attributes:
        provider: ARM provider type.
        environment: Production, UAT, or sandbox.
        api_key: API key for authentication.
        api_secret: API secret for authentication.
        base_url: Override for API base URL.
        timeout_seconds: Request timeout.
        max_retries: Maximum retry attempts.
        batch_size: Maximum reports per batch.
        rate_limit_per_minute: Maximum requests per minute.
    """

    provider: ARMProvider = ARMProvider.MOCK
    environment: ARMEnvironment = ARMEnvironment.SANDBOX
    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 3
    batch_size: int = 100
    rate_limit_per_minute: int = 60
    nca_code: str = ""  # e.g., "FCA", "BaFin", "AMF"
    lei: str = ""  # Submitting firm's LEI


# ============================================================================
# Abstract ARM Client
# ============================================================================


class ARMClient(ABC):
    """
    Abstract base class for ARM clients.

    Defines the interface for all ARM provider implementations.
    Each provider must implement the core submission methods.

    Example:
        client = BloombergBTRLClient(config)

        # Submit single report
        result = await client.submit_report(report)

        # Submit batch
        results = await client.submit_batch(reports)

        # Check status
        status = await client.query_status(confirmation_id)
    """

    def __init__(self, config: ARMClientConfig):
        """
        Initialize ARM client.

        Args:
            config: ARM client configuration.
        """
        self._config = config
        self._rate_limit_timestamps: deque = deque()

        # Statistics
        self._stats = {
            "submissions_total": 0,
            "submissions_success": 0,
            "submissions_failed": 0,
            "rate_limit_waits": 0,
            "retries": 0,
        }

        # Handle both enum and string providers (for fallback case)
        provider_value = (
            config.provider.value
            if hasattr(config.provider, 'value')
            else config.provider
        )
        environment_value = (
            config.environment.value
            if hasattr(config.environment, 'value')
            else config.environment
        )

        logger.info(
            f"ARMClient initialized",
            extra={
                "provider": provider_value,
                "environment": environment_value,
            },
        )

    @property
    def provider(self) -> ARMProvider:
        """Get ARM provider type."""
        return self._config.provider

    @property
    def environment(self) -> ARMEnvironment:
        """Get environment type."""
        return self._config.environment

    def _check_rate_limit(self) -> float:
        """
        Check rate limit and return required wait time.

        Returns:
            Seconds to wait before next request (0 if no wait needed).
        """
        now = time.time()
        window_start = now - 60.0  # 1 minute window

        # Remove old timestamps
        while self._rate_limit_timestamps and self._rate_limit_timestamps[0] < window_start:
            self._rate_limit_timestamps.popleft()

        if len(self._rate_limit_timestamps) >= self._config.rate_limit_per_minute:
            oldest = self._rate_limit_timestamps[0]
            wait_time = (oldest + 60.0) - now
            return max(0.0, wait_time)

        return 0.0

    def _record_request(self) -> None:
        """Record a request for rate limiting."""
        self._rate_limit_timestamps.append(time.time())
        self._stats["submissions_total"] += 1

    @abstractmethod
    async def submit_report(self, report: Any) -> SubmissionResult:
        """
        Submit a single transaction report.

        Args:
            report: TransactionReport instance to submit.

        Returns:
            SubmissionResult with status and confirmation.

        Raises:
            ARMError: On submission failure.
        """
        pass

    @abstractmethod
    async def submit_batch(self, reports: List[Any]) -> BatchSubmissionResult:
        """
        Submit a batch of transaction reports.

        Args:
            reports: List of TransactionReport instances.

        Returns:
            BatchSubmissionResult with individual results.
        """
        pass

    @abstractmethod
    async def query_status(self, confirmation_id: str) -> SubmissionResult:
        """
        Query status of a previously submitted report.

        Args:
            confirmation_id: ARM confirmation ID.

        Returns:
            SubmissionResult with current status.
        """
        pass

    @abstractmethod
    async def cancel_report(self, confirmation_id: str, reason: str = "") -> bool:
        """
        Cancel a previously submitted report.

        Args:
            confirmation_id: ARM confirmation ID.
            reason: Reason for cancellation.

        Returns:
            True if cancellation was successful.
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if ARM service is available.

        Returns:
            True if service is healthy.
        """
        # Default implementation - override in subclasses
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self._stats,
            "provider": self._config.provider.value,
            "environment": self._config.environment.value,
            "rate_limit_queue": len(self._rate_limit_timestamps),
        }


# ============================================================================
# Mock ARM Client (for testing)
# ============================================================================


class MockARMClient(ARMClient):
    """
    Mock ARM client for testing and development.

    Simulates ARM behavior without making actual API calls.
    Useful for:
        - Unit testing
        - Integration testing
        - Development without ARM credentials
        - Compliance demonstrations

    Behavior:
        - Accepts all valid reports
        - Rejects reports with validation errors
        - Simulates network latency (configurable)
        - Tracks all submissions in memory
    """

    def __init__(
        self,
        config: ARMClientConfig,
        simulate_latency_ms: float = 100.0,
        failure_rate: float = 0.0,
    ):
        """
        Initialize mock ARM client.

        Args:
            config: ARM client configuration.
            simulate_latency_ms: Simulated network latency.
            failure_rate: Fraction of requests to fail (0.0-1.0).
        """
        super().__init__(config)
        self._latency_ms = simulate_latency_ms
        self._failure_rate = failure_rate
        self._submissions: Dict[str, SubmissionResult] = {}
        self._batch_id = 0

    async def submit_report(self, report: Any) -> SubmissionResult:
        """Submit a single report to mock ARM."""
        # Check rate limit
        wait_time = self._check_rate_limit()
        if wait_time > 0:
            self._stats["rate_limit_waits"] += 1
            await asyncio.sleep(wait_time)

        self._record_request()

        # Simulate latency
        await asyncio.sleep(self._latency_ms / 1000.0)

        # Check for simulated failure
        import random
        if random.random() < self._failure_rate:
            self._stats["submissions_failed"] += 1
            return SubmissionResult(
                report_id=getattr(report, "transaction_reference_number", str(uuid.uuid4())),
                status=SubmissionStatus.ERROR,
                errors=["Simulated ARM failure"],
                arm_provider=self.provider.value,
                submitted_at=datetime.now(timezone.utc),
            )

        # Validate report
        report_id = getattr(report, "transaction_reference_number", str(uuid.uuid4()))
        errors = []

        if hasattr(report, "validate"):
            errors = report.validate()

        if errors:
            self._stats["submissions_failed"] += 1
            result = SubmissionResult(
                report_id=report_id,
                confirmation_id="",
                status=SubmissionStatus.REJECTED,
                errors=errors,
                arm_provider=self.provider.value,
                submitted_at=datetime.now(timezone.utc),
            )
        else:
            self._stats["submissions_success"] += 1
            confirmation_id = f"MOCK-{uuid.uuid4().hex[:12].upper()}"
            result = SubmissionResult(
                report_id=report_id,
                confirmation_id=confirmation_id,
                status=SubmissionStatus.ACCEPTED,
                arm_provider=self.provider.value,
                submitted_at=datetime.now(timezone.utc),
                nca_received_at=datetime.now(timezone.utc),
            )

        # Store for status queries
        self._submissions[result.confirmation_id or report_id] = result

        logger.info(
            f"Mock ARM: Report submitted",
            extra={
                "report_id": report_id,
                "confirmation_id": result.confirmation_id,
                "status": result.status.value,
            },
        )

        return result

    async def submit_batch(self, reports: List[Any]) -> BatchSubmissionResult:
        """Submit a batch of reports to mock ARM."""
        self._batch_id += 1
        batch_id = f"MOCK-BATCH-{self._batch_id:06d}"

        results = []
        for report in reports:
            result = await self.submit_report(report)
            results.append(result)

        submitted = sum(1 for r in results if r.is_success())
        failed = len(results) - submitted

        return BatchSubmissionResult(
            batch_id=batch_id,
            total_reports=len(reports),
            submitted=submitted,
            failed=failed,
            results=results,
        )

    async def query_status(self, confirmation_id: str) -> SubmissionResult:
        """Query status from mock ARM."""
        await asyncio.sleep(self._latency_ms / 1000.0)

        if confirmation_id in self._submissions:
            return self._submissions[confirmation_id]

        return SubmissionResult(
            report_id="",
            confirmation_id=confirmation_id,
            status=SubmissionStatus.ERROR,
            errors=["Confirmation ID not found"],
            arm_provider=self.provider.value,
        )

    async def cancel_report(self, confirmation_id: str, reason: str = "") -> bool:
        """Cancel a report in mock ARM."""
        await asyncio.sleep(self._latency_ms / 1000.0)

        if confirmation_id in self._submissions:
            self._submissions[confirmation_id].status = SubmissionStatus.CANCELLED
            logger.info(f"Mock ARM: Report cancelled", extra={"confirmation_id": confirmation_id})
            return True

        return False

    def get_all_submissions(self) -> List[SubmissionResult]:
        """Get all submissions (mock-specific method for testing)."""
        return list(self._submissions.values())

    def clear_submissions(self) -> None:
        """Clear all submissions (mock-specific method for testing)."""
        self._submissions.clear()


# ============================================================================
# Bloomberg BTRL Client
# ============================================================================


class BloombergBTRLClient(ARMClient):
    """
    Bloomberg Transaction Reporting (BTRL) client.

    Bloomberg BTRL is one of the major ARM providers for MiFIR reporting.

    Features:
        - RESTful API with JSON/XML support
        - Real-time status updates
        - Batch submission support
        - NCA-specific routing

    API Endpoints (example):
        - POST /v1/transactions - Submit report
        - GET /v1/transactions/{id} - Query status
        - DELETE /v1/transactions/{id} - Cancel report

    Note:
        Requires Bloomberg BTRL subscription and credentials.
        See: https://www.bloomberg.com/professional/solution/regulatory-reporting/

    This implementation is a template - actual API details may differ.
    """

    DEFAULT_URLS = {
        ARMEnvironment.PRODUCTION: "https://btrl-api.bloomberg.com",
        ARMEnvironment.UAT: "https://btrl-api-uat.bloomberg.com",
        ARMEnvironment.SANDBOX: "https://btrl-api-sandbox.bloomberg.com",
    }

    def __init__(self, config: ARMClientConfig):
        """Initialize Bloomberg BTRL client."""
        super().__init__(config)
        self._base_url = config.base_url or self.DEFAULT_URLS.get(
            config.environment,
            self.DEFAULT_URLS[ARMEnvironment.SANDBOX],
        )

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make authenticated request to BTRL API."""
        try:
            import httpx

            url = f"{self._base_url}{endpoint}"
            headers = {
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
                "X-API-Key": self._config.api_key,
            }

            async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                elif method == "POST":
                    response = await client.post(url, headers=headers, json=data)
                elif method == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if response.status_code == 429:
                    raise ARMError(
                        code=ErrorCode.RATE_LIMIT_EXCEEDED,
                        message="Bloomberg BTRL rate limit exceeded",
                    )

                if response.status_code == 401:
                    raise ARMError(
                        code=ErrorCode.AUTHENTICATION_ERROR,
                        message="Bloomberg BTRL authentication failed",
                    )

                if response.status_code >= 400:
                    raise ARMError(
                        code=ErrorCode.INTERNAL_ERROR,
                        message=f"Bloomberg BTRL error: HTTP {response.status_code}",
                        details={"response": response.text},
                    )

                return response.json()

        except ImportError:
            logger.error("httpx not installed - cannot connect to Bloomberg BTRL")
            raise ARMError(
                code=ErrorCode.CONNECTION_ERROR,
                message="httpx library required for Bloomberg BTRL",
            )

    async def submit_report(self, report: Any) -> SubmissionResult:
        """Submit report to Bloomberg BTRL."""
        # Check rate limit
        wait_time = self._check_rate_limit()
        if wait_time > 0:
            self._stats["rate_limit_waits"] += 1
            await asyncio.sleep(wait_time)

        self._record_request()

        report_id = getattr(report, "transaction_reference_number", str(uuid.uuid4()))

        # Validate locally first
        if hasattr(report, "validate"):
            errors = report.validate()
            if errors:
                self._stats["submissions_failed"] += 1
                return SubmissionResult(
                    report_id=report_id,
                    status=SubmissionStatus.REJECTED,
                    errors=errors,
                    arm_provider=self.provider.value,
                    submitted_at=datetime.now(timezone.utc),
                )

        try:
            # Convert to API format
            payload = {
                "report": report.to_dict() if hasattr(report, "to_dict") else {},
                "submitting_lei": self._config.lei,
                "nca": self._config.nca_code,
            }

            response = await self._make_request("POST", "/v1/transactions", payload)

            self._stats["submissions_success"] += 1
            return SubmissionResult(
                report_id=report_id,
                confirmation_id=response.get("confirmationId", ""),
                status=SubmissionStatus.SUBMITTED,
                arm_provider=self.provider.value,
                submitted_at=datetime.now(timezone.utc),
                raw_response=response,
            )

        except ARMError:
            self._stats["submissions_failed"] += 1
            raise

        except Exception as e:
            self._stats["submissions_failed"] += 1
            logger.error(f"Bloomberg BTRL submission error: {e}")
            return SubmissionResult(
                report_id=report_id,
                status=SubmissionStatus.ERROR,
                errors=[str(e)],
                arm_provider=self.provider.value,
                submitted_at=datetime.now(timezone.utc),
            )

    async def submit_batch(self, reports: List[Any]) -> BatchSubmissionResult:
        """Submit batch to Bloomberg BTRL."""
        batch_id = str(uuid.uuid4())
        results = []

        # BTRL typically supports batch submission
        # For simplicity, we submit individually here
        for report in reports:
            try:
                result = await self.submit_report(report)
                results.append(result)
            except ARMError as e:
                results.append(SubmissionResult(
                    report_id=getattr(report, "transaction_reference_number", ""),
                    status=SubmissionStatus.ERROR,
                    errors=[str(e)],
                    arm_provider=self.provider.value,
                ))

        submitted = sum(1 for r in results if r.is_success())

        return BatchSubmissionResult(
            batch_id=batch_id,
            total_reports=len(reports),
            submitted=submitted,
            failed=len(reports) - submitted,
            results=results,
        )

    async def query_status(self, confirmation_id: str) -> SubmissionResult:
        """Query status from Bloomberg BTRL."""
        try:
            response = await self._make_request("GET", f"/v1/transactions/{confirmation_id}")

            status_map = {
                "PENDING": SubmissionStatus.PENDING,
                "SUBMITTED": SubmissionStatus.SUBMITTED,
                "ACCEPTED": SubmissionStatus.ACCEPTED,
                "REJECTED": SubmissionStatus.REJECTED,
                "CANCELLED": SubmissionStatus.CANCELLED,
            }

            return SubmissionResult(
                report_id=response.get("reportId", ""),
                confirmation_id=confirmation_id,
                status=status_map.get(response.get("status", ""), SubmissionStatus.PENDING),
                errors=response.get("errors", []),
                warnings=response.get("warnings", []),
                arm_provider=self.provider.value,
                raw_response=response,
            )

        except ARMError:
            raise

        except Exception as e:
            return SubmissionResult(
                report_id="",
                confirmation_id=confirmation_id,
                status=SubmissionStatus.ERROR,
                errors=[str(e)],
                arm_provider=self.provider.value,
            )

    async def cancel_report(self, confirmation_id: str, reason: str = "") -> bool:
        """Cancel report in Bloomberg BTRL."""
        try:
            await self._make_request("DELETE", f"/v1/transactions/{confirmation_id}")
            return True
        except ARMError:
            return False

    async def health_check(self) -> bool:
        """Check Bloomberg BTRL service health."""
        try:
            await self._make_request("GET", "/v1/health")
            return True
        except Exception:
            return False


# ============================================================================
# File-based ARM Client (for offline/batch processing)
# ============================================================================


class FileARMClient(ARMClient):
    """
    File-based ARM client for offline batch processing.

    Writes reports to files for later submission via SFTP/manual upload.
    Useful for:
        - Batch processing pipelines
        - Disaster recovery
        - Regulatory testing
        - Offline environments

    Output Formats:
        - XML (ISO 20022)
        - JSON
        - CSV (for analysis)
    """

    def __init__(
        self,
        config: ARMClientConfig,
        output_directory: str = "output/arm_reports",
        format: str = "xml",
    ):
        """
        Initialize file-based ARM client.

        Args:
            config: ARM client configuration.
            output_directory: Directory for output files.
            format: Output format (xml, json, csv).
        """
        super().__init__(config)
        self._output_dir = output_directory
        self._format = format
        self._file_counter = 0

        # Ensure directory exists
        import os
        os.makedirs(output_directory, exist_ok=True)

    async def submit_report(self, report: Any) -> SubmissionResult:
        """Write report to file."""
        report_id = getattr(report, "transaction_reference_number", str(uuid.uuid4()))

        # Validate
        if hasattr(report, "validate"):
            errors = report.validate()
            if errors:
                return SubmissionResult(
                    report_id=report_id,
                    status=SubmissionStatus.REJECTED,
                    errors=errors,
                    arm_provider="file",
                )

        self._file_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._output_dir}/report_{timestamp}_{self._file_counter:06d}.{self._format}"

        try:
            if self._format == "xml" and hasattr(report, "to_xml"):
                content = report.to_xml()
            elif self._format == "json" and hasattr(report, "to_dict"):
                content = json.dumps(report.to_dict(), indent=2)
            else:
                content = str(report)

            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)

            self._stats["submissions_success"] += 1
            return SubmissionResult(
                report_id=report_id,
                confirmation_id=filename,
                status=SubmissionStatus.SUBMITTED,
                arm_provider="file",
                submitted_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            self._stats["submissions_failed"] += 1
            return SubmissionResult(
                report_id=report_id,
                status=SubmissionStatus.ERROR,
                errors=[str(e)],
                arm_provider="file",
            )

    async def submit_batch(self, reports: List[Any]) -> BatchSubmissionResult:
        """Write batch to files."""
        results = []
        for report in reports:
            result = await self.submit_report(report)
            results.append(result)

        submitted = sum(1 for r in results if r.is_success())
        return BatchSubmissionResult(
            batch_id=str(uuid.uuid4()),
            total_reports=len(reports),
            submitted=submitted,
            failed=len(reports) - submitted,
            results=results,
        )

    async def query_status(self, confirmation_id: str) -> SubmissionResult:
        """Query file status (check if file exists)."""
        import os
        exists = os.path.exists(confirmation_id)
        return SubmissionResult(
            report_id="",
            confirmation_id=confirmation_id,
            status=SubmissionStatus.SUBMITTED if exists else SubmissionStatus.ERROR,
            arm_provider="file",
        )

    async def cancel_report(self, confirmation_id: str, reason: str = "") -> bool:
        """Delete file to cancel report."""
        import os
        try:
            if os.path.exists(confirmation_id):
                os.remove(confirmation_id)
                return True
        except Exception:
            pass
        return False


# ============================================================================
# Factory Function
# ============================================================================


def create_arm_client(
    config: ARMClientConfig,
    **kwargs,
) -> ARMClient:
    """
    Factory function to create ARM client based on configuration.

    Args:
        config: ARM client configuration.
        **kwargs: Additional provider-specific arguments.

    Returns:
        Configured ARMClient instance.

    Example:
        config = ARMClientConfig(
            provider=ARMProvider.BLOOMBERG_BTRL,
            environment=ARMEnvironment.UAT,
            api_key="your-api-key",
        )
        client = create_arm_client(config)
    """
    provider_map = {
        ARMProvider.MOCK: MockARMClient,
        ARMProvider.BLOOMBERG_BTRL: BloombergBTRLClient,
        # Add more providers as implemented
    }

    client_class = provider_map.get(config.provider)

    if not client_class:
        logger.warning(f"Unknown ARM provider {config.provider}, using mock")
        client_class = MockARMClient

    return client_class(config, **kwargs)


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    # Enums
    "ARMProvider",
    "ARMEnvironment",
    "SubmissionStatus",
    "ErrorCode",
    # Data classes
    "ARMError",
    "SubmissionResult",
    "BatchSubmissionResult",
    "ARMClientConfig",
    # Clients
    "ARMClient",
    "MockARMClient",
    "BloombergBTRLClient",
    "FileARMClient",
    # Factory
    "create_arm_client",
]
