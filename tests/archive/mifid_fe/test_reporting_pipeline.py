# -*- coding: utf-8 -*-
"""
Tests for MiFID II Transaction Reporting Pipeline Module.

Tests cover:
    - PipelineConfig configuration
    - TransactionReportingPipeline functionality
    - Trade event processing
    - Report queuing and batching
    - Status tracking
    - Cancellation and amendment flows
    - Pipeline metrics
    - Local caching
    - Factory function

References:
    - MiFIR Article 26: Transaction reporting
    - RTS 22: Report format requirements
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from services.archive.mifid_financial_entity.reporting_pipeline import (
    # Enums
    PipelineStatus,
    ReportQueuePriority,
    # Data classes
    PipelineConfig,
    QueuedReport,
    PipelineMetrics,
    # Main class
    TransactionReportingPipeline,
    # Factory
    create_reporting_pipeline,
)
from services.archive.mifid_financial_entity.arm_client import (
    ARMProvider,
    ARMEnvironment,
    ARMClientConfig,
    MockARMClient,
    SubmissionStatus,
)
from services.archive.mifid_financial_entity.transaction_report import (
    TransactionReport,
    TransactionReportParty,
    TradingCapacity,
    BuySellIndicator,
    ReportStatus,
    IdentifierType,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def pipeline_config():
    """Create a pipeline configuration for testing."""
    return PipelineConfig(
        arm_config=ARMClientConfig(
            provider=ARMProvider.MOCK,
            environment=ARMEnvironment.SANDBOX,
        ),
        batch_size=10,
        batch_interval_seconds=1.0,  # Fast for testing
        max_retry_attempts=3,
        retry_delay_seconds=0.1,
        t_plus_one_deadline_hours=24.0,
        enable_local_cache=False,  # Disable for most tests
        lei="5493001KJTIIGC8Y1R12",
        default_currency="EUR",
        trading_capacity=TradingCapacity.DEAL,
    )


@pytest.fixture
def pipeline(pipeline_config):
    """Create a pipeline instance for testing."""
    return TransactionReportingPipeline(pipeline_config)


@pytest.fixture
def valid_trade():
    """Create a valid trade dictionary for testing."""
    return {
        "order_id": "ORDER-001",
        "isin": "US0378331005",
        "quantity": 100,
        "price": 150.25,
        "side": "BUY",
        "venue": "XNAS",
        "currency": "USD",
    }


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


# ============================================================================
# PipelineConfig Tests
# ============================================================================


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.batch_size == 100
        assert config.batch_interval_seconds == 60.0
        assert config.max_retry_attempts == 3
        assert config.enable_local_cache is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            batch_size=50,
            batch_interval_seconds=30.0,
            lei="5493001KJTIIGC8Y1R12",
            default_currency="GBP",
        )
        assert config.batch_size == 50
        assert config.batch_interval_seconds == 30.0
        assert config.lei == "5493001KJTIIGC8Y1R12"
        assert config.default_currency == "GBP"


# ============================================================================
# QueuedReport Tests
# ============================================================================


class TestQueuedReport:
    """Tests for QueuedReport."""

    def test_queued_report_creation(self, valid_report):
        """Test creating a queued report."""
        queued = QueuedReport(
            report=valid_report,
            priority=ReportQueuePriority.NORMAL,
        )
        assert queued.report == valid_report
        assert queued.priority == ReportQueuePriority.NORMAL
        assert queued.retry_count == 0

    def test_is_expired(self, valid_report):
        """Test expiry check."""
        queued = QueuedReport(
            report=valid_report,
            queued_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        assert queued.is_expired(deadline_hours=24.0)

    def test_is_not_expired(self, valid_report):
        """Test non-expired report."""
        queued = QueuedReport(
            report=valid_report,
            queued_at=datetime.now(timezone.utc),
        )
        assert not queued.is_expired(deadline_hours=24.0)


# ============================================================================
# PipelineMetrics Tests
# ============================================================================


class TestPipelineMetrics:
    """Tests for PipelineMetrics."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = PipelineMetrics()
        assert metrics.reports_received == 0
        assert metrics.reports_submitted == 0
        assert metrics.reports_accepted == 0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = PipelineMetrics(
            reports_received=100,
            reports_submitted=90,
            reports_accepted=85,
            reports_rejected=5,
        )
        data = metrics.to_dict()

        assert data["reports_received"] == 100
        assert data["reports_submitted"] == 90
        assert data["reports_accepted"] == 85
        assert "success_rate" in data


# ============================================================================
# TransactionReportingPipeline Tests
# ============================================================================


class TestTransactionReportingPipeline:
    """Tests for TransactionReportingPipeline."""

    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.status == PipelineStatus.STOPPED
        assert pipeline.metrics.reports_received == 0

    @pytest.mark.asyncio
    async def test_start_and_stop(self, pipeline):
        """Test starting and stopping the pipeline."""
        await pipeline.start()
        assert pipeline.status == PipelineStatus.RUNNING

        await pipeline.stop()
        assert pipeline.status == PipelineStatus.STOPPED

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, pipeline):
        """Test pausing and resuming the pipeline."""
        await pipeline.start()
        assert pipeline.status == PipelineStatus.RUNNING

        await pipeline.pause()
        assert pipeline.status == PipelineStatus.PAUSED

        await pipeline.resume()
        assert pipeline.status == PipelineStatus.RUNNING

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_queue_report(self, pipeline, valid_report):
        """Test queuing a report."""
        report_id = await pipeline.queue_report(valid_report)

        assert report_id == valid_report.transaction_reference_number
        assert pipeline.metrics.reports_received == 1
        assert pipeline.metrics.reports_pending == 1

    @pytest.mark.asyncio
    async def test_queue_report_high_priority(self, pipeline, valid_report):
        """Test queuing a report with high priority."""
        report_id = await pipeline.queue_report(
            valid_report,
            priority=ReportQueuePriority.HIGH,
        )

        assert report_id == valid_report.transaction_reference_number
        # High priority reports should be at the front of the queue

    @pytest.mark.asyncio
    async def test_on_trade_executed(self, pipeline, valid_trade):
        """Test processing a trade execution event."""
        report_id = await pipeline.on_trade_executed(valid_trade)

        assert report_id == "ORDER-001"
        assert pipeline.metrics.reports_received == 1

    @pytest.mark.asyncio
    async def test_on_trade_executed_with_algorithm(self, pipeline, valid_trade):
        """Test processing trade with algorithm ID."""
        report_id = await pipeline.on_trade_executed(
            valid_trade,
            algorithm_id="ALGO-001",
        )

        assert report_id == "ORDER-001"

    @pytest.mark.asyncio
    async def test_flush_reports(self, pipeline, valid_report):
        """Test flushing all pending reports."""
        await pipeline.start()

        # Queue several reports
        for i in range(5):
            report = TransactionReport(
                transaction_reference_number=f"TEST-{i:03d}",
                executing_entity_lei="5493001KJTIIGC8Y1R12",
                instrument_isin="US0378331005",
                quantity=Decimal("100"),
                price=Decimal("150"),
                price_currency="USD",
                venue_mic="XNAS",
                trading_datetime=datetime.now(timezone.utc),
                buyer=TransactionReportParty(
                    identifier="5493001KJTIIGC8Y1R12",
                    identifier_type=IdentifierType.LEI,
                ),
                seller=TransactionReportParty(
                    identifier="",
                    identifier_type=IdentifierType.NONE,
                ),
            )
            await pipeline.queue_report(report)

        # Flush
        result = await pipeline.flush_reports()

        assert result.total_reports == 5
        assert pipeline.metrics.reports_pending == 0

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_get_report_status(self, pipeline, valid_report):
        """Test getting report status."""
        await pipeline.start()
        await pipeline.queue_report(valid_report)
        await pipeline.flush_reports()

        status = await pipeline.get_report_status(
            valid_report.transaction_reference_number
        )

        assert status is not None
        assert status.status == SubmissionStatus.ACCEPTED

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_cancel_pending_report(self, pipeline, valid_report):
        """Test cancelling a pending report."""
        await pipeline.queue_report(valid_report)

        success = await pipeline.cancel_report(
            valid_report.transaction_reference_number
        )

        assert success
        assert pipeline.metrics.reports_pending == 0

    @pytest.mark.asyncio
    async def test_submit_amendment(self, pipeline, valid_report):
        """Test submitting an amendment."""
        await pipeline.start()
        await pipeline.queue_report(valid_report)
        await pipeline.flush_reports()

        # Submit amendment
        amendment_id = await pipeline.submit_amendment(
            valid_report.transaction_reference_number,
            amendments={"quantity": Decimal("200")},
            reason="Quantity correction",
        )

        assert amendment_id != valid_report.transaction_reference_number
        assert pipeline.metrics.reports_received == 2

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_submit_cancellation(self, pipeline, valid_report):
        """Test submitting a cancellation."""
        await pipeline.start()
        await pipeline.queue_report(valid_report)
        await pipeline.flush_reports()

        # Submit cancellation
        cancellation_id = await pipeline.submit_cancellation(
            valid_report.transaction_reference_number,
            reason="Error in original report",
        )

        assert cancellation_id != valid_report.transaction_reference_number
        assert pipeline.metrics.reports_received == 2

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_submit_amendment_unknown_report(self, pipeline):
        """Test submitting amendment for unknown report."""
        with pytest.raises(ValueError) as exc_info:
            await pipeline.submit_amendment(
                "UNKNOWN-001",
                amendments={},
            )
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_submit_cancellation_unknown_report(self, pipeline):
        """Test submitting cancellation for unknown report."""
        with pytest.raises(ValueError) as exc_info:
            await pipeline.submit_cancellation("UNKNOWN-001")
        assert "not found" in str(exc_info.value).lower()

    def test_get_statistics(self, pipeline):
        """Test getting pipeline statistics."""
        stats = pipeline.get_statistics()

        assert "status" in stats
        assert "metrics" in stats
        assert "queue" in stats
        assert "config" in stats

    def test_generate_compliance_report(self, pipeline):
        """Test generating compliance report."""
        report = pipeline.generate_compliance_report()

        assert report["report_type"] == "MIFIR_TRANSACTION_REPORTING_COMPLIANCE"
        assert report["regulation"] == "MiFIR Article 26"
        assert "statistics" in report
        assert "compliance_status" in report


# ============================================================================
# Callback Tests
# ============================================================================


class TestPipelineCallbacks:
    """Tests for pipeline callbacks."""

    @pytest.mark.asyncio
    async def test_on_submitted_callback(self, pipeline_config, valid_report):
        """Test callback when report is submitted."""
        callback_called = []

        async def on_submitted(report, result):
            callback_called.append((report, result))

        pipeline = TransactionReportingPipeline(
            pipeline_config,
            on_report_submitted=on_submitted,
        )

        await pipeline.start()
        await pipeline.queue_report(valid_report)
        await pipeline.flush_reports()

        assert len(callback_called) == 1
        assert callback_called[0][0].transaction_reference_number == valid_report.transaction_reference_number

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_on_failed_callback(self, pipeline_config):
        """Test callback when report fails permanently."""
        callback_called = []

        async def on_failed(report, error):
            callback_called.append((report, error))

        config = PipelineConfig(
            arm_config=ARMClientConfig(provider=ARMProvider.MOCK),
            max_retry_attempts=0,  # No retries
            enable_local_cache=False,
            lei="5493001KJTIIGC8Y1R12",
        )

        pipeline = TransactionReportingPipeline(
            config,
            on_report_failed=on_failed,
        )

        # Create invalid report
        invalid_report = TransactionReport(
            executing_entity_lei="",  # Invalid
            instrument_isin="INVALID",
        )

        await pipeline.start()
        await pipeline.queue_report(invalid_report)
        await pipeline.flush_reports()

        # Give time for processing
        await asyncio.sleep(0.1)

        await pipeline.stop()


# ============================================================================
# Caching Tests
# ============================================================================


class TestPipelineCaching:
    """Tests for pipeline caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_enabled(self, valid_report):
        """Test caching when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                arm_config=ARMClientConfig(provider=ARMProvider.MOCK),
                enable_local_cache=True,
                cache_directory=tmpdir,
                lei="5493001KJTIIGC8Y1R12",
            )
            pipeline = TransactionReportingPipeline(config)

            await pipeline.queue_report(valid_report)

            # Check cache file exists
            cache_file = os.path.join(
                tmpdir,
                f"{valid_report.transaction_reference_number}.json"
            )
            assert os.path.exists(cache_file)

    @pytest.mark.asyncio
    async def test_cache_cleared_after_submission(self, valid_report):
        """Test cache is cleared after successful submission."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                arm_config=ARMClientConfig(provider=ARMProvider.MOCK),
                enable_local_cache=True,
                cache_directory=tmpdir,
                lei="5493001KJTIIGC8Y1R12",
            )
            pipeline = TransactionReportingPipeline(config)

            await pipeline.start()
            await pipeline.queue_report(valid_report)
            await pipeline.flush_reports()

            # Check cache file is removed
            cache_file = os.path.join(
                tmpdir,
                f"{valid_report.transaction_reference_number}.json"
            )
            assert not os.path.exists(cache_file)

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_load_cached_reports_on_startup(self):
        """Test loading cached reports on startup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First, create a cached report
            config = PipelineConfig(
                arm_config=ARMClientConfig(provider=ARMProvider.MOCK),
                enable_local_cache=True,
                cache_directory=tmpdir,
                lei="5493001KJTIIGC8Y1R12",
            )
            pipeline1 = TransactionReportingPipeline(config)

            report = TransactionReport(
                transaction_reference_number="CACHED-001",
                executing_entity_lei="5493001KJTIIGC8Y1R12",
                instrument_isin="US0378331005",
                quantity=Decimal("100"),
                price=Decimal("150"),
                price_currency="USD",
                venue_mic="XNAS",
                trading_datetime=datetime.now(timezone.utc),
                buyer=TransactionReportParty(
                    identifier="5493001KJTIIGC8Y1R12",
                    identifier_type=IdentifierType.LEI,
                ),
                seller=TransactionReportParty(
                    identifier="",
                    identifier_type=IdentifierType.NONE,
                ),
            )
            await pipeline1.queue_report(report)

            # Create new pipeline - should load cached report
            pipeline2 = TransactionReportingPipeline(config)
            await pipeline2.start()

            # Should have loaded the cached report
            assert pipeline2.metrics.reports_pending == 1

            await pipeline2.stop()


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateReportingPipeline:
    """Tests for create_reporting_pipeline factory function."""

    def test_create_pipeline_basic(self):
        """Test creating pipeline with basic parameters."""
        pipeline = create_reporting_pipeline(
            lei="5493001KJTIIGC8Y1R12",
        )

        assert pipeline._config.lei == "5493001KJTIIGC8Y1R12"
        assert pipeline._config.arm_config.provider == ARMProvider.MOCK

    def test_create_pipeline_with_provider(self):
        """Test creating pipeline with specific ARM provider."""
        pipeline = create_reporting_pipeline(
            lei="5493001KJTIIGC8Y1R12",
            arm_provider=ARMProvider.BLOOMBERG_BTRL,
            arm_environment=ARMEnvironment.UAT,
            api_key="test-key",
        )

        assert pipeline._config.arm_config.provider == ARMProvider.BLOOMBERG_BTRL
        assert pipeline._config.arm_config.environment == ARMEnvironment.UAT
        assert pipeline._config.arm_config.api_key == "test-key"

    def test_create_pipeline_with_custom_config(self):
        """Test creating pipeline with custom configuration."""
        pipeline = create_reporting_pipeline(
            lei="5493001KJTIIGC8Y1R12",
            batch_size=50,
            default_currency="GBP",
        )

        assert pipeline._config.batch_size == 50
        assert pipeline._config.default_currency == "GBP"


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for pipeline enumerations."""

    def test_pipeline_status_values(self):
        """Test PipelineStatus enum values."""
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.PAUSED.value == "paused"
        assert PipelineStatus.STOPPED.value == "stopped"
        assert PipelineStatus.ERROR.value == "error"

    def test_report_queue_priority_values(self):
        """Test ReportQueuePriority enum values."""
        assert ReportQueuePriority.HIGH.value == "high"
        assert ReportQueuePriority.NORMAL.value == "normal"
        assert ReportQueuePriority.LOW.value == "low"


# ============================================================================
# Integration Tests
# ============================================================================


class TestPipelineIntegration:
    """Integration tests for the full pipeline flow."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, pipeline_config):
        """Test complete pipeline flow from trade to submission."""
        pipeline = TransactionReportingPipeline(pipeline_config)

        await pipeline.start()

        # Process multiple trades
        trades = [
            {
                "order_id": f"ORDER-{i:03d}",
                "isin": "US0378331005",
                "quantity": 100 * (i + 1),
                "price": 150.25,
                "side": "BUY" if i % 2 == 0 else "SELL",
                "venue": "XNAS",
                "currency": "USD",
            }
            for i in range(10)
        ]

        for trade in trades:
            await pipeline.on_trade_executed(trade, algorithm_id="ALGO-001")

        # Flush all reports
        result = await pipeline.flush_reports()

        assert result.total_reports == 10
        assert result.submitted == 10
        assert pipeline.metrics.reports_received == 10

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_with_retry(self, pipeline_config):
        """Test pipeline retry mechanism."""
        # Create ARM client with high failure rate
        arm_client = MockARMClient(
            pipeline_config.arm_config,
            failure_rate=0.5,  # 50% failure
        )

        pipeline = TransactionReportingPipeline(
            pipeline_config,
            arm_client=arm_client,
        )

        await pipeline.start()

        # Queue reports
        for i in range(5):
            report = TransactionReport(
                transaction_reference_number=f"TEST-{i:03d}",
                executing_entity_lei="5493001KJTIIGC8Y1R12",
                instrument_isin="US0378331005",
                quantity=Decimal("100"),
                price=Decimal("150"),
                price_currency="USD",
                venue_mic="XNAS",
                trading_datetime=datetime.now(timezone.utc),
                buyer=TransactionReportParty(
                    identifier="5493001KJTIIGC8Y1R12",
                    identifier_type=IdentifierType.LEI,
                ),
                seller=TransactionReportParty(
                    identifier="",
                    identifier_type=IdentifierType.NONE,
                ),
            )
            await pipeline.queue_report(report)

        # Flush
        await pipeline.flush_reports()

        # Some retries should have occurred
        # (exact number depends on random failures)

        await pipeline.stop()
