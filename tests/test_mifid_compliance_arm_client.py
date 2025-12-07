# -*- coding: utf-8 -*-
"""
Tests for MiFID II ARM (Approved Reporting Mechanism) Client Module.

Tests cover:
    - ARMClientConfig configuration
    - MockARMClient functionality
    - FileARMClient functionality
    - SubmissionResult and BatchSubmissionResult
    - Rate limiting
    - Error handling
    - Factory function

References:
    - MiFIR Article 26: Transaction reporting via ARM
    - ESMA ARM Register
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timezone
from decimal import Decimal

from services.compliance.arm_client import (
    # Enums
    ARMProvider,
    ARMEnvironment,
    SubmissionStatus,
    ErrorCode,
    # Data classes
    ARMError,
    SubmissionResult,
    BatchSubmissionResult,
    ARMClientConfig,
    # Clients
    ARMClient,
    MockARMClient,
    BloombergBTRLClient,
    FileARMClient,
    # Factory
    create_arm_client,
)
from services.compliance.transaction_report import (
    TransactionReport,
    TransactionReportParty,
    BuySellIndicator,
    IdentifierType,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config():
    """Create a mock ARM client configuration."""
    return ARMClientConfig(
        provider=ARMProvider.MOCK,
        environment=ARMEnvironment.SANDBOX,
        api_key="test-key",
        api_secret="test-secret",
        timeout_seconds=30.0,
        max_retries=3,
        batch_size=100,
        rate_limit_per_minute=60,
        lei="5493001KJTIIGC8Y1R12",
    )


@pytest.fixture
def valid_report():
    """Create a valid transaction report for testing."""
    return TransactionReport(
        transaction_reference_number="TEST-001",
        executing_entity_lei="5493001KJTIIGC8Y1R12",
        instrument_isin="US0378331005",
        quantity=Decimal("100"),
        price=Decimal("150.25"),
        price_currency="USD",
        venue_mic="XNAS",
        trading_datetime=datetime.now(timezone.utc),
        buy_sell_indicator=BuySellIndicator.BUY,
        # RTS 22 requires buyer/seller identification
        buyer=TransactionReportParty(
            identifier="5493001KJTIIGC8Y1R12",
            identifier_type=IdentifierType.LEI,
        ),
        seller=TransactionReportParty(
            identifier="",
            identifier_type=IdentifierType.NONE,
        ),
    )


@pytest.fixture
def invalid_report():
    """Create an invalid transaction report for testing."""
    return TransactionReport(
        transaction_reference_number="INVALID-001",
        executing_entity_lei="",  # Missing required field
        instrument_isin="INVALID",  # Invalid ISIN
        quantity=Decimal("0"),  # Invalid quantity
        venue_mic="",  # Missing venue
    )


@pytest.fixture
def mock_client(mock_config):
    """Create a mock ARM client."""
    return MockARMClient(mock_config)


# ============================================================================
# ARMClientConfig Tests
# ============================================================================


class TestARMClientConfig:
    """Tests for ARMClientConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ARMClientConfig()
        assert config.provider == ARMProvider.MOCK
        assert config.environment == ARMEnvironment.SANDBOX
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3
        assert config.batch_size == 100
        assert config.rate_limit_per_minute == 60

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ARMClientConfig(
            provider=ARMProvider.BLOOMBERG_BTRL,
            environment=ARMEnvironment.PRODUCTION,
            api_key="my-key",
            timeout_seconds=60.0,
            batch_size=50,
        )
        assert config.provider == ARMProvider.BLOOMBERG_BTRL
        assert config.environment == ARMEnvironment.PRODUCTION
        assert config.api_key == "my-key"
        assert config.timeout_seconds == 60.0
        assert config.batch_size == 50


# ============================================================================
# SubmissionResult Tests
# ============================================================================


class TestSubmissionResult:
    """Tests for SubmissionResult."""

    def test_successful_result(self):
        """Test successful submission result."""
        result = SubmissionResult(
            report_id="TEST-001",
            confirmation_id="CONF-001",
            status=SubmissionStatus.ACCEPTED,
            submitted_at=datetime.now(timezone.utc),
            arm_provider="mock",
        )
        assert result.is_success()
        assert result.is_final()

    def test_pending_result(self):
        """Test pending submission result."""
        result = SubmissionResult(
            report_id="TEST-001",
            status=SubmissionStatus.PENDING,
        )
        assert not result.is_success()
        assert not result.is_final()

    def test_rejected_result(self):
        """Test rejected submission result."""
        result = SubmissionResult(
            report_id="TEST-001",
            status=SubmissionStatus.REJECTED,
            errors=["Invalid ISIN", "Missing quantity"],
        )
        assert not result.is_success()
        assert result.is_final()
        assert len(result.errors) == 2

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = SubmissionResult(
            report_id="TEST-001",
            confirmation_id="CONF-001",
            status=SubmissionStatus.SUBMITTED,
            arm_provider="mock",
        )
        data = result.to_dict()
        assert data["report_id"] == "TEST-001"
        assert data["confirmation_id"] == "CONF-001"
        assert data["status"] == "submitted"
        assert data["arm_provider"] == "mock"


# ============================================================================
# BatchSubmissionResult Tests
# ============================================================================


class TestBatchSubmissionResult:
    """Tests for BatchSubmissionResult."""

    def test_empty_batch(self):
        """Test empty batch result."""
        result = BatchSubmissionResult()
        assert result.total_reports == 0
        assert result.submitted == 0
        assert result.failed == 0
        assert result.success_rate() == 0.0

    def test_full_success_batch(self):
        """Test batch with all successful submissions."""
        results = [
            SubmissionResult(report_id=f"TEST-{i}", status=SubmissionStatus.ACCEPTED)
            for i in range(10)
        ]
        result = BatchSubmissionResult(
            total_reports=10,
            submitted=10,
            failed=0,
            results=results,
        )
        assert result.success_rate() == 100.0

    def test_partial_success_batch(self):
        """Test batch with partial success."""
        result = BatchSubmissionResult(
            total_reports=10,
            submitted=7,
            failed=3,
        )
        assert result.success_rate() == 70.0


# ============================================================================
# MockARMClient Tests
# ============================================================================


class TestMockARMClient:
    """Tests for MockARMClient."""

    @pytest.mark.asyncio
    async def test_submit_valid_report(self, mock_client, valid_report):
        """Test submitting a valid report."""
        result = await mock_client.submit_report(valid_report)

        assert result.status == SubmissionStatus.ACCEPTED
        assert result.confirmation_id.startswith("MOCK-")
        assert result.arm_provider == "mock"
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_submit_invalid_report(self, mock_client, invalid_report):
        """Test submitting an invalid report."""
        result = await mock_client.submit_report(invalid_report)

        assert result.status == SubmissionStatus.REJECTED
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_submit_batch(self, mock_client, valid_report):
        """Test batch submission."""
        reports = [valid_report for _ in range(5)]
        result = await mock_client.submit_batch(reports)

        assert result.total_reports == 5
        assert result.submitted == 5
        assert result.failed == 0
        assert result.batch_id.startswith("MOCK-BATCH-")

    @pytest.mark.asyncio
    async def test_submit_mixed_batch(self, mock_client, valid_report, invalid_report):
        """Test batch with mixed valid/invalid reports."""
        reports = [valid_report, invalid_report, valid_report]
        result = await mock_client.submit_batch(reports)

        assert result.total_reports == 3
        assert result.submitted == 2
        assert result.failed == 1

    @pytest.mark.asyncio
    async def test_query_status(self, mock_client, valid_report):
        """Test querying submission status."""
        # First submit
        submit_result = await mock_client.submit_report(valid_report)

        # Then query
        status = await mock_client.query_status(submit_result.confirmation_id)

        assert status.status == SubmissionStatus.ACCEPTED
        assert status.confirmation_id == submit_result.confirmation_id

    @pytest.mark.asyncio
    async def test_query_unknown_status(self, mock_client):
        """Test querying status of unknown report."""
        status = await mock_client.query_status("UNKNOWN-ID")

        assert status.status == SubmissionStatus.ERROR
        assert len(status.errors) > 0

    @pytest.mark.asyncio
    async def test_cancel_report(self, mock_client, valid_report):
        """Test cancelling a submitted report."""
        # First submit
        submit_result = await mock_client.submit_report(valid_report)

        # Then cancel
        success = await mock_client.cancel_report(submit_result.confirmation_id)
        assert success

        # Verify status
        status = await mock_client.query_status(submit_result.confirmation_id)
        assert status.status == SubmissionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_unknown_report(self, mock_client):
        """Test cancelling unknown report."""
        success = await mock_client.cancel_report("UNKNOWN-ID")
        assert not success

    @pytest.mark.asyncio
    async def test_simulated_failure(self, mock_config):
        """Test simulated failure rate."""
        client = MockARMClient(mock_config, failure_rate=1.0)  # 100% failure

        report = TransactionReport(
            executing_entity_lei="5493001KJTIIGC8Y1R12",
            instrument_isin="US0378331005",
            quantity=Decimal("100"),
            price=Decimal("150"),
            price_currency="USD",
            venue_mic="XNAS",
            trading_datetime=datetime.now(timezone.utc),
        )

        result = await client.submit_report(report)
        assert result.status == SubmissionStatus.ERROR

    @pytest.mark.asyncio
    async def test_simulated_latency(self, mock_config):
        """Test simulated latency."""
        latency_ms = 200
        client = MockARMClient(mock_config, simulate_latency_ms=latency_ms)

        report = TransactionReport(
            executing_entity_lei="5493001KJTIIGC8Y1R12",
            instrument_isin="US0378331005",
            quantity=Decimal("100"),
            price=Decimal("150"),
            price_currency="USD",
            venue_mic="XNAS",
            trading_datetime=datetime.now(timezone.utc),
        )

        start = datetime.now()
        await client.submit_report(report)
        elapsed = (datetime.now() - start).total_seconds() * 1000

        assert elapsed >= latency_ms * 0.9  # Allow 10% tolerance

    def test_get_statistics(self, mock_client):
        """Test getting client statistics."""
        stats = mock_client.get_statistics()

        assert "submissions_total" in stats
        assert "submissions_success" in stats
        assert "submissions_failed" in stats
        assert "provider" in stats
        assert stats["provider"] == "mock"

    def test_get_all_submissions(self, mock_client):
        """Test mock-specific get_all_submissions method."""
        submissions = mock_client.get_all_submissions()
        assert isinstance(submissions, list)

    def test_clear_submissions(self, mock_client):
        """Test mock-specific clear_submissions method."""
        mock_client.clear_submissions()
        assert len(mock_client.get_all_submissions()) == 0


# ============================================================================
# FileARMClient Tests
# ============================================================================


class TestFileARMClient:
    """Tests for FileARMClient."""

    @pytest.fixture
    def file_client(self, mock_config):
        """Create a file-based ARM client with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FileARMClient(mock_config, output_directory=tmpdir, format="json")

    @pytest.mark.asyncio
    async def test_submit_report_to_file(self, mock_config, valid_report):
        """Test submitting report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = FileARMClient(mock_config, output_directory=tmpdir, format="json")
            result = await client.submit_report(valid_report)

            assert result.is_success()
            assert result.confirmation_id.endswith(".json")
            assert os.path.exists(result.confirmation_id)

    @pytest.mark.asyncio
    async def test_submit_report_xml_format(self, mock_config, valid_report):
        """Test submitting report in XML format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = FileARMClient(mock_config, output_directory=tmpdir, format="xml")
            result = await client.submit_report(valid_report)

            assert result.is_success()
            assert result.confirmation_id.endswith(".xml")

    @pytest.mark.asyncio
    async def test_submit_batch_to_files(self, mock_config, valid_report):
        """Test batch submission to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = FileARMClient(mock_config, output_directory=tmpdir)
            reports = [valid_report for _ in range(3)]
            result = await client.submit_batch(reports)

            assert result.total_reports == 3
            assert result.submitted == 3

    @pytest.mark.asyncio
    async def test_query_file_status(self, mock_config, valid_report):
        """Test querying file status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = FileARMClient(mock_config, output_directory=tmpdir)
            submit_result = await client.submit_report(valid_report)

            status = await client.query_status(submit_result.confirmation_id)
            assert status.status == SubmissionStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_cancel_file_report(self, mock_config, valid_report):
        """Test cancelling (deleting) file report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = FileARMClient(mock_config, output_directory=tmpdir)
            submit_result = await client.submit_report(valid_report)

            success = await client.cancel_report(submit_result.confirmation_id)
            assert success
            assert not os.path.exists(submit_result.confirmation_id)


# ============================================================================
# BloombergBTRLClient Tests
# ============================================================================


class TestBloombergBTRLClient:
    """Tests for BloombergBTRLClient (without actual API calls)."""

    def test_initialization(self):
        """Test client initialization."""
        config = ARMClientConfig(
            provider=ARMProvider.BLOOMBERG_BTRL,
            environment=ARMEnvironment.SANDBOX,
            api_key="test-key",
        )
        client = BloombergBTRLClient(config)

        assert client.provider == ARMProvider.BLOOMBERG_BTRL
        assert client.environment == ARMEnvironment.SANDBOX

    def test_environment_urls(self):
        """Test different environment URLs."""
        # Production
        config = ARMClientConfig(
            provider=ARMProvider.BLOOMBERG_BTRL,
            environment=ARMEnvironment.PRODUCTION,
        )
        client = BloombergBTRLClient(config)
        assert "bloomberg" in client._base_url.lower()

        # UAT
        config.environment = ARMEnvironment.UAT
        client = BloombergBTRLClient(config)
        assert "uat" in client._base_url.lower()

    def test_custom_base_url(self):
        """Test custom base URL override."""
        config = ARMClientConfig(
            provider=ARMProvider.BLOOMBERG_BTRL,
            base_url="https://custom.api.example.com",
        )
        client = BloombergBTRLClient(config)
        assert client._base_url == "https://custom.api.example.com"


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateARMClient:
    """Tests for create_arm_client factory function."""

    def test_create_mock_client(self):
        """Test creating mock client."""
        config = ARMClientConfig(provider=ARMProvider.MOCK)
        client = create_arm_client(config)

        assert isinstance(client, MockARMClient)

    def test_create_bloomberg_client(self):
        """Test creating Bloomberg client."""
        config = ARMClientConfig(provider=ARMProvider.BLOOMBERG_BTRL)
        client = create_arm_client(config)

        assert isinstance(client, BloombergBTRLClient)

    def test_create_unknown_provider_fallback(self):
        """Test fallback to mock for unknown provider."""
        config = ARMClientConfig()
        # Manually set an invalid provider value
        config.provider = "unknown_provider"  # type: ignore

        client = create_arm_client(config)
        assert isinstance(client, MockARMClient)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestARMError:
    """Tests for ARMError exception."""

    def test_error_creation(self):
        """Test creating ARM error."""
        error = ARMError(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid ISIN format",
            report_id="TEST-001",
        )
        assert error.code == ErrorCode.VALIDATION_ERROR
        assert error.message == "Invalid ISIN format"
        assert error.report_id == "TEST-001"

    def test_error_string(self):
        """Test error string representation."""
        error = ARMError(
            code=ErrorCode.TIMEOUT_ERROR,
            message="Request timed out",
        )
        error_str = str(error)
        assert "TIMEOUT_ERROR" in error_str
        assert "Request timed out" in error_str


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for ARM enumerations."""

    def test_arm_provider_values(self):
        """Test ARMProvider enum values."""
        assert ARMProvider.MOCK.value == "mock"
        assert ARMProvider.BLOOMBERG_BTRL.value == "bloomberg_btrl"
        assert ARMProvider.CME_TRAX.value == "cme_trax"
        assert ARMProvider.LSEG_UNAVISTA.value == "lseg_unavista"

    def test_arm_environment_values(self):
        """Test ARMEnvironment enum values."""
        assert ARMEnvironment.PRODUCTION.value == "production"
        assert ARMEnvironment.UAT.value == "uat"
        assert ARMEnvironment.SANDBOX.value == "sandbox"

    def test_submission_status_values(self):
        """Test SubmissionStatus enum values."""
        assert SubmissionStatus.PENDING.value == "pending"
        assert SubmissionStatus.SUBMITTED.value == "submitted"
        assert SubmissionStatus.ACCEPTED.value == "accepted"
        assert SubmissionStatus.REJECTED.value == "rejected"
        assert SubmissionStatus.CANCELLED.value == "cancelled"
        assert SubmissionStatus.ERROR.value == "error"

    def test_error_code_values(self):
        """Test ErrorCode enum values."""
        assert ErrorCode.VALIDATION_ERROR.value == "VALIDATION_ERROR"
        assert ErrorCode.AUTHENTICATION_ERROR.value == "AUTHENTICATION_ERROR"
        assert ErrorCode.RATE_LIMIT_EXCEEDED.value == "RATE_LIMIT_EXCEEDED"
        assert ErrorCode.TIMEOUT_ERROR.value == "TIMEOUT_ERROR"


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_tracking(self, mock_client, valid_report):
        """Test that rate limit is tracked."""
        # Submit several reports
        for _ in range(5):
            await mock_client.submit_report(valid_report)

        stats = mock_client.get_statistics()
        assert stats["rate_limit_queue"] <= 5

    @pytest.mark.asyncio
    async def test_disabled_rate_limiting(self, mock_config, valid_report):
        """Test client with disabled rate limiting."""
        config = ARMClientConfig(
            provider=ARMProvider.MOCK,
            rate_limit_per_minute=1000,  # High limit
        )
        client = MockARMClient(config)

        # Should not block
        result = await client.submit_report(valid_report)
        assert result.is_success()
