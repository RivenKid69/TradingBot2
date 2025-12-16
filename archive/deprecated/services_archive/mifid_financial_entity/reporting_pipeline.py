# -*- coding: utf-8 -*-
"""
Transaction Reporting Pipeline Module.

Orchestrates the end-to-end transaction reporting process per MiFIR Article 26.
Handles trade event processing, report generation, validation, batching,
submission to ARMs, and status tracking.

Pipeline Flow:
    1. Trade Execution Event received
    2. TransactionReport created via builder
    3. Report validated locally
    4. Report enriched with reference data (LEI, ISIN)
    5. Report queued for batch submission
    6. Batch submitted to ARM (within T+1 deadline)
    7. Status tracked and audit trail maintained

Regulatory Requirements:
    - Reports must be submitted by T+1 (next working day)
    - All mandatory fields must be populated
    - Reports must be in ISO 20022 format
    - Failed reports must be corrected and resubmitted
    - Audit trail must be maintained for 5 years

References:
    - MiFIR Article 26: Transaction reporting obligation
    - RTS 22: Format and content of transaction reports
    - ESMA Guidelines on transaction reporting
"""

from __future__ import annotations

import logging
import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Callable, Awaitable
from enum import Enum
from collections import deque
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# Type imports (avoid circular imports)
# ============================================================================


# Forward references for type hints
from services.archive.mifid_financial_entity.transaction_report import (
    TransactionReport,
    TransactionReportBuilder,
    TransactionReportParty,
    BuySellIndicator,
    TradingCapacity,
    ReportStatus,
    IdentifierType,
)
from services.archive.mifid_financial_entity.arm_client import (
    ARMClient,
    ARMClientConfig,
    ARMProvider,
    ARMEnvironment,
    SubmissionResult,
    BatchSubmissionResult,
    SubmissionStatus,
    create_arm_client,
)


# ============================================================================
# Enumerations
# ============================================================================


class PipelineStatus(str, Enum):
    """Status of the reporting pipeline."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ReportQueuePriority(str, Enum):
    """Priority levels for report queue."""

    HIGH = "high"  # Corrections, amendments
    NORMAL = "normal"  # Regular trades
    LOW = "low"  # Back-dated reports


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PipelineConfig:
    """
    Configuration for transaction reporting pipeline.

    Attributes:
        arm_config: Configuration for ARM client.
        batch_size: Maximum reports per batch.
        batch_interval_seconds: Maximum wait time before sending batch.
        max_retry_attempts: Maximum retry attempts for failed reports.
        retry_delay_seconds: Delay between retry attempts.
        t_plus_one_deadline_hours: Hours before T+1 deadline to trigger flush.
        enable_local_cache: Cache reports locally for recovery.
        cache_directory: Directory for local cache.
        lei: Firm's LEI for report generation.
        default_currency: Default currency for reports.
        country_of_branch: Country code of supervising branch.
    """

    arm_config: ARMClientConfig = field(default_factory=ARMClientConfig)
    batch_size: int = 100
    batch_interval_seconds: float = 60.0
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 30.0
    t_plus_one_deadline_hours: float = 20.0  # Submit by 8pm for next day
    enable_local_cache: bool = True
    cache_directory: str = "state/compliance/report_cache"
    lei: str = ""
    default_currency: str = "EUR"
    country_of_branch: str = ""
    trading_capacity: TradingCapacity = TradingCapacity.DEAL


@dataclass
class QueuedReport:
    """
    Report in the submission queue.

    Tracks report status through the submission process.
    """

    report: TransactionReport
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: ReportQueuePriority = ReportQueuePriority.NORMAL
    retry_count: int = 0
    last_error: str = ""
    submission_result: Optional[SubmissionResult] = None

    def is_expired(self, deadline_hours: float = 24.0) -> bool:
        """Check if report has exceeded T+1 deadline."""
        elapsed = (datetime.now(timezone.utc) - self.queued_at).total_seconds() / 3600
        return elapsed > deadline_hours


@dataclass
class PipelineMetrics:
    """
    Metrics for pipeline monitoring.

    Tracks submission statistics, errors, and performance.
    """

    reports_received: int = 0
    reports_submitted: int = 0
    reports_accepted: int = 0
    reports_rejected: int = 0
    reports_failed: int = 0
    reports_pending: int = 0
    batches_submitted: int = 0
    total_retry_attempts: int = 0
    last_submission_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    last_error_message: str = ""
    avg_submission_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reports_received": self.reports_received,
            "reports_submitted": self.reports_submitted,
            "reports_accepted": self.reports_accepted,
            "reports_rejected": self.reports_rejected,
            "reports_failed": self.reports_failed,
            "reports_pending": self.reports_pending,
            "batches_submitted": self.batches_submitted,
            "total_retry_attempts": self.total_retry_attempts,
            "last_submission_at": self.last_submission_at.isoformat() if self.last_submission_at else None,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
            "last_error_message": self.last_error_message,
            "avg_submission_time_ms": round(self.avg_submission_time_ms, 2),
            "success_rate": self._calculate_success_rate(),
        }

    def _calculate_success_rate(self) -> float:
        """Calculate submission success rate."""
        total = self.reports_submitted + self.reports_failed
        if total == 0:
            return 100.0
        return (self.reports_accepted / total) * 100


# ============================================================================
# Reporting Pipeline
# ============================================================================


class TransactionReportingPipeline:
    """
    End-to-end transaction reporting pipeline for MiFIR Article 26.

    Manages the complete lifecycle of transaction reports from trade
    execution to NCA submission via ARM.

    Features:
        - Trade event processing and report generation
        - Local validation before submission
        - Batching for efficient submission
        - Automatic retry on failures
        - T+1 deadline monitoring
        - Local caching for disaster recovery
        - Audit trail integration

    Example:
        # Initialize pipeline
        config = PipelineConfig(
            arm_config=ARMClientConfig(provider=ARMProvider.MOCK),
            lei="5493001KJTIIGC8Y1R12",
        )
        pipeline = TransactionReportingPipeline(config)

        # Start pipeline
        await pipeline.start()

        # Process trade
        await pipeline.on_trade_executed({
            "order_id": "12345",
            "isin": "US0378331005",
            "quantity": 100,
            "price": 150.25,
            "side": "BUY",
            "venue": "XNAS",
        })

        # Stop pipeline (flushes pending reports)
        await pipeline.stop()
    """

    def __init__(
        self,
        config: PipelineConfig,
        arm_client: Optional[ARMClient] = None,
        on_report_submitted: Optional[Callable[[TransactionReport, SubmissionResult], Awaitable[None]]] = None,
        on_report_failed: Optional[Callable[[TransactionReport, str], Awaitable[None]]] = None,
    ):
        """
        Initialize reporting pipeline.

        Args:
            config: Pipeline configuration.
            arm_client: Optional pre-configured ARM client.
            on_report_submitted: Callback when report is submitted.
            on_report_failed: Callback when report fails permanently.
        """
        self._config = config
        self._arm_client = arm_client or create_arm_client(config.arm_config)

        # Callbacks
        self._on_submitted = on_report_submitted
        self._on_failed = on_report_failed

        # Report builder
        self._builder = TransactionReportBuilder(
            executing_entity_lei=config.lei,
            default_currency=config.default_currency,
            trading_capacity=config.trading_capacity,
            country_of_branch=config.country_of_branch,
        )

        # Queues
        self._pending_queue: deque[QueuedReport] = deque()
        self._retry_queue: deque[QueuedReport] = deque()
        self._submitted_reports: Dict[str, QueuedReport] = {}

        # State
        self._status = PipelineStatus.STOPPED
        self._metrics = PipelineMetrics()
        self._lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None

        # Local cache
        if config.enable_local_cache:
            Path(config.cache_directory).mkdir(parents=True, exist_ok=True)

        logger.info(
            "TransactionReportingPipeline initialized",
            extra={
                "lei": config.lei[:8] + "..." if config.lei else "not_set",
                "arm_provider": config.arm_config.provider.value,
                "batch_size": config.batch_size,
            },
        )

    @property
    def status(self) -> PipelineStatus:
        """Get pipeline status."""
        return self._status

    @property
    def metrics(self) -> PipelineMetrics:
        """Get pipeline metrics."""
        return self._metrics

    async def start(self) -> None:
        """
        Start the reporting pipeline.

        Begins processing queued reports and starts background tasks.
        """
        if self._status == PipelineStatus.RUNNING:
            return

        self._status = PipelineStatus.RUNNING

        # Start batch processing task
        self._batch_task = asyncio.create_task(self._batch_processing_loop())

        # Start deadline monitoring task
        self._monitor_task = asyncio.create_task(self._deadline_monitoring_loop())

        # Load cached reports
        await self._load_cached_reports()

        logger.info("Transaction reporting pipeline started")

    async def stop(self, flush: bool = True) -> None:
        """
        Stop the reporting pipeline.

        Args:
            flush: If True, submit all pending reports before stopping.
        """
        if self._status == PipelineStatus.STOPPED:
            return

        self._status = PipelineStatus.STOPPED

        # Flush pending reports
        if flush:
            await self.flush_reports()

        # Cancel background tasks
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Save remaining reports to cache
        await self._save_pending_to_cache()

        logger.info("Transaction reporting pipeline stopped")

    async def pause(self) -> None:
        """Pause report submission (continues queuing)."""
        self._status = PipelineStatus.PAUSED
        logger.info("Transaction reporting pipeline paused")

    async def resume(self) -> None:
        """Resume report submission."""
        if self._status == PipelineStatus.PAUSED:
            self._status = PipelineStatus.RUNNING
            logger.info("Transaction reporting pipeline resumed")

    async def on_trade_executed(
        self,
        trade: Dict[str, Any],
        algorithm_id: str = "",
        priority: ReportQueuePriority = ReportQueuePriority.NORMAL,
    ) -> str:
        """
        Process a trade execution event.

        Creates a TransactionReport from the trade data and queues it
        for submission to the ARM.

        Args:
            trade: Trade data dictionary with required fields.
            algorithm_id: ID of the algorithm that made the decision.
            priority: Queue priority for the report.

        Returns:
            Transaction reference number.

        Trade Data Fields:
            Required:
                - isin: Instrument ISIN
                - quantity: Trade quantity
                - price: Execution price
                - side: "BUY" or "SELL"
                - venue/mic: Execution venue MIC

            Optional:
                - order_id: Order identifier (generated if not provided)
                - timestamp: Execution timestamp (now if not provided)
                - currency: Price currency
                - counterparty_lei: Counterparty LEI
        """
        # Build report from trade data
        report = self._builder.from_trade(trade, algorithm_id)

        # Queue for submission
        return await self.queue_report(report, priority)

    async def queue_report(
        self,
        report: TransactionReport,
        priority: ReportQueuePriority = ReportQueuePriority.NORMAL,
    ) -> str:
        """
        Queue a transaction report for submission.

        Args:
            report: TransactionReport to queue.
            priority: Queue priority.

        Returns:
            Transaction reference number.
        """
        async with self._lock:
            # Validate report
            errors = report.validate()
            if errors:
                logger.warning(
                    f"Report validation errors: {errors}",
                    extra={"report_id": report.transaction_reference_number},
                )
                # Still queue but mark for review
                report.report_status = ReportStatus.PENDING

            # Create queued report
            queued = QueuedReport(
                report=report,
                priority=priority,
            )

            # Add to appropriate queue position based on priority
            if priority == ReportQueuePriority.HIGH:
                self._pending_queue.appendleft(queued)
            else:
                self._pending_queue.append(queued)

            self._metrics.reports_received += 1
            self._metrics.reports_pending = len(self._pending_queue) + len(self._retry_queue)

            # Cache locally
            if self._config.enable_local_cache:
                await self._cache_report(report)

        logger.debug(
            f"Report queued",
            extra={
                "report_id": report.transaction_reference_number,
                "priority": priority.value,
                "queue_size": len(self._pending_queue),
            },
        )

        return report.transaction_reference_number

    async def flush_reports(self) -> BatchSubmissionResult:
        """
        Immediately submit all pending reports.

        Returns:
            BatchSubmissionResult with submission details.
        """
        async with self._lock:
            reports = [q.report for q in self._pending_queue]
            self._pending_queue.clear()

            # Add retry queue
            retry_reports = [q.report for q in self._retry_queue]
            self._retry_queue.clear()
            reports.extend(retry_reports)

        if not reports:
            return BatchSubmissionResult()

        return await self._submit_batch(reports)

    async def get_report_status(self, report_id: str) -> Optional[SubmissionResult]:
        """
        Get status of a submitted report.

        Args:
            report_id: Transaction reference number.

        Returns:
            SubmissionResult if found, None otherwise.
        """
        if report_id in self._submitted_reports:
            queued = self._submitted_reports[report_id]
            return queued.submission_result

        # Check with ARM
        if self._arm_client:
            try:
                return await self._arm_client.query_status(report_id)
            except Exception:
                pass

        return None

    async def cancel_report(self, report_id: str, reason: str = "") -> bool:
        """
        Cancel a pending or submitted report.

        Args:
            report_id: Transaction reference number.
            reason: Reason for cancellation.

        Returns:
            True if cancellation was successful.
        """
        # Check pending queue
        async with self._lock:
            for i, queued in enumerate(self._pending_queue):
                if queued.report.transaction_reference_number == report_id:
                    del self._pending_queue[i]
                    self._metrics.reports_pending -= 1
                    logger.info(f"Cancelled pending report: {report_id}")
                    return True

        # Check submitted reports
        if report_id in self._submitted_reports:
            queued = self._submitted_reports[report_id]
            if queued.submission_result and queued.submission_result.confirmation_id:
                success = await self._arm_client.cancel_report(
                    queued.submission_result.confirmation_id,
                    reason,
                )
                if success:
                    logger.info(f"Cancelled submitted report: {report_id}")
                return success

        return False

    async def submit_amendment(
        self,
        original_report_id: str,
        amendments: Dict[str, Any],
        reason: str = "",
    ) -> str:
        """
        Submit an amendment to a previously submitted report.

        Args:
            original_report_id: ID of the original report.
            amendments: Dictionary of fields to amend.
            reason: Reason for amendment.

        Returns:
            New transaction reference number.
        """
        # Find original report
        original = None
        if original_report_id in self._submitted_reports:
            original = self._submitted_reports[original_report_id].report

        if not original:
            raise ValueError(f"Original report not found: {original_report_id}")

        # Create amendment
        amendment = original.create_amendment(reason)

        # Apply amendments
        for key, value in amendments.items():
            if hasattr(amendment, key):
                setattr(amendment, key, value)

        # Queue with high priority
        return await self.queue_report(amendment, ReportQueuePriority.HIGH)

    async def submit_cancellation(
        self,
        original_report_id: str,
        reason: str = "",
    ) -> str:
        """
        Submit a cancellation report.

        Args:
            original_report_id: ID of the report to cancel.
            reason: Reason for cancellation.

        Returns:
            Cancellation report transaction reference number.
        """
        # Find original report
        original = None
        if original_report_id in self._submitted_reports:
            original = self._submitted_reports[original_report_id].report

        if not original:
            raise ValueError(f"Original report not found: {original_report_id}")

        # Create cancellation
        cancellation = original.create_cancellation(reason)

        # Queue with high priority
        return await self.queue_report(cancellation, ReportQueuePriority.HIGH)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            "status": self._status.value,
            "metrics": self._metrics.to_dict(),
            "queue": {
                "pending": len(self._pending_queue),
                "retry": len(self._retry_queue),
                "submitted": len(self._submitted_reports),
            },
            "arm_client": self._arm_client.get_statistics() if self._arm_client else {},
            "config": {
                "batch_size": self._config.batch_size,
                "batch_interval_seconds": self._config.batch_interval_seconds,
                "max_retry_attempts": self._config.max_retry_attempts,
            },
        }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate compliance report for audit purposes.

        Returns:
            Dictionary with compliance report data.
        """
        return {
            "report_type": "MIFIR_TRANSACTION_REPORTING_COMPLIANCE",
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "regulation": "MiFIR Article 26",
            "arm_provider": self._config.arm_config.provider.value,
            "environment": self._config.arm_config.environment.value,
            "statistics": self.get_statistics(),
            "compliance_status": {
                "all_reports_submitted": self._metrics.reports_pending == 0,
                "rejection_rate": self._calculate_rejection_rate(),
                "t_plus_one_compliance": self._check_t_plus_one_compliance(),
            },
        }

    def _calculate_rejection_rate(self) -> float:
        """Calculate report rejection rate."""
        total = self._metrics.reports_accepted + self._metrics.reports_rejected
        if total == 0:
            return 0.0
        return (self._metrics.reports_rejected / total) * 100

    def _check_t_plus_one_compliance(self) -> bool:
        """Check if all reports are within T+1 deadline."""
        for queued in self._pending_queue:
            if queued.is_expired(self._config.t_plus_one_deadline_hours):
                return False
        return True

    # ========================================================================
    # Background Processing
    # ========================================================================

    async def _batch_processing_loop(self) -> None:
        """Background task for batch processing."""
        while self._status == PipelineStatus.RUNNING:
            try:
                await self._process_batch()
                await asyncio.sleep(self._config.batch_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                self._metrics.last_error_at = datetime.now(timezone.utc)
                self._metrics.last_error_message = str(e)
                await asyncio.sleep(5.0)  # Brief pause on error

    async def _deadline_monitoring_loop(self) -> None:
        """Background task for T+1 deadline monitoring."""
        while self._status in (PipelineStatus.RUNNING, PipelineStatus.PAUSED):
            try:
                await self._check_deadlines()
                await asyncio.sleep(60.0)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Deadline monitoring error: {e}")
                await asyncio.sleep(60.0)

    async def _process_batch(self) -> None:
        """Process a batch of reports from the queue."""
        if self._status != PipelineStatus.RUNNING:
            return

        # Collect batch
        batch: List[TransactionReport] = []

        async with self._lock:
            # First process retry queue
            while self._retry_queue and len(batch) < self._config.batch_size:
                queued = self._retry_queue.popleft()
                if queued.retry_count < self._config.max_retry_attempts:
                    batch.append(queued.report)
                else:
                    # Max retries exceeded
                    self._metrics.reports_failed += 1
                    if self._on_failed:
                        await self._on_failed(queued.report, queued.last_error)

            # Then process pending queue
            while self._pending_queue and len(batch) < self._config.batch_size:
                queued = self._pending_queue.popleft()
                batch.append(queued.report)

            self._metrics.reports_pending = len(self._pending_queue) + len(self._retry_queue)

        if not batch:
            return

        await self._submit_batch(batch)

    async def _submit_batch(self, reports: List[TransactionReport]) -> BatchSubmissionResult:
        """Submit a batch of reports to ARM."""
        if not reports:
            return BatchSubmissionResult()

        start_time = datetime.now(timezone.utc)

        try:
            result = await self._arm_client.submit_batch(reports)

            # Process results
            for i, sub_result in enumerate(result.results):
                report = reports[i]

                if sub_result.is_success():
                    self._metrics.reports_submitted += 1
                    if sub_result.status == SubmissionStatus.ACCEPTED:
                        self._metrics.reports_accepted += 1
                    report.report_status = ReportStatus.SUBMITTED
                    report.arm_confirmation_id = sub_result.confirmation_id
                    report.submitted_at = datetime.now(timezone.utc)

                    # Store for status queries
                    self._submitted_reports[report.transaction_reference_number] = QueuedReport(
                        report=report,
                        submission_result=sub_result,
                    )

                    # Remove from cache
                    if self._config.enable_local_cache:
                        await self._remove_from_cache(report.transaction_reference_number)

                    # Callback
                    if self._on_submitted:
                        await self._on_submitted(report, sub_result)

                else:
                    # Failed - queue for retry
                    self._metrics.reports_rejected += 1
                    async with self._lock:
                        queued = QueuedReport(
                            report=report,
                            retry_count=1,
                            last_error="; ".join(sub_result.errors),
                        )
                        self._retry_queue.append(queued)
                        self._metrics.total_retry_attempts += 1

            # Update metrics
            self._metrics.batches_submitted += 1
            self._metrics.last_submission_at = datetime.now(timezone.utc)

            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._metrics.avg_submission_time_ms = (
                self._metrics.avg_submission_time_ms * 0.9 + elapsed_ms * 0.1
            )

            logger.info(
                f"Batch submitted",
                extra={
                    "batch_id": result.batch_id,
                    "total": result.total_reports,
                    "submitted": result.submitted,
                    "failed": result.failed,
                },
            )

            return result

        except Exception as e:
            # Re-queue all reports
            logger.error(f"Batch submission failed: {e}")
            self._metrics.last_error_at = datetime.now(timezone.utc)
            self._metrics.last_error_message = str(e)

            async with self._lock:
                for report in reports:
                    queued = QueuedReport(
                        report=report,
                        retry_count=1,
                        last_error=str(e),
                    )
                    self._retry_queue.append(queued)

            return BatchSubmissionResult(
                total_reports=len(reports),
                failed=len(reports),
            )

    async def _check_deadlines(self) -> None:
        """Check for reports approaching T+1 deadline."""
        deadline_threshold = self._config.t_plus_one_deadline_hours - 2  # 2 hours before

        expired_reports = []

        async with self._lock:
            for queued in self._pending_queue:
                if queued.is_expired(deadline_threshold):
                    expired_reports.append(queued)

        if expired_reports:
            logger.warning(
                f"{len(expired_reports)} reports approaching T+1 deadline",
                extra={"count": len(expired_reports)},
            )

            # Force flush if we have reports approaching deadline
            if self._status == PipelineStatus.RUNNING:
                await self.flush_reports()

    # ========================================================================
    # Caching
    # ========================================================================

    async def _cache_report(self, report: TransactionReport) -> None:
        """Cache report to local storage for recovery."""
        if not self._config.enable_local_cache:
            return

        try:
            cache_path = Path(self._config.cache_directory) / f"{report.transaction_reference_number}.json"
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f)
        except Exception as e:
            logger.warning(f"Failed to cache report: {e}")

    async def _remove_from_cache(self, report_id: str) -> None:
        """Remove report from local cache."""
        if not self._config.enable_local_cache:
            return

        try:
            cache_path = Path(self._config.cache_directory) / f"{report_id}.json"
            if cache_path.exists():
                cache_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove report from cache: {e}")

    async def _load_cached_reports(self) -> None:
        """Load cached reports on startup."""
        if not self._config.enable_local_cache:
            return

        cache_dir = Path(self._config.cache_directory)
        if not cache_dir.exists():
            return

        loaded = 0
        for cache_file in cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                report = TransactionReport.from_dict(data)

                # Queue for submission
                queued = QueuedReport(
                    report=report,
                    priority=ReportQueuePriority.HIGH,  # Cached reports are priority
                )
                self._pending_queue.appendleft(queued)
                loaded += 1

            except Exception as e:
                logger.warning(f"Failed to load cached report {cache_file}: {e}")

        if loaded > 0:
            logger.info(f"Loaded {loaded} cached reports")
            self._metrics.reports_pending += loaded

    async def _save_pending_to_cache(self) -> None:
        """Save all pending reports to cache on shutdown."""
        if not self._config.enable_local_cache:
            return

        for queued in self._pending_queue:
            await self._cache_report(queued.report)

        for queued in self._retry_queue:
            await self._cache_report(queued.report)


# ============================================================================
# Factory Function
# ============================================================================


def create_reporting_pipeline(
    lei: str,
    arm_provider: ARMProvider = ARMProvider.MOCK,
    arm_environment: ARMEnvironment = ARMEnvironment.SANDBOX,
    api_key: str = "",
    **kwargs,
) -> TransactionReportingPipeline:
    """
    Factory function to create TransactionReportingPipeline.

    Args:
        lei: Firm's LEI for report generation.
        arm_provider: ARM provider to use.
        arm_environment: ARM environment (production, uat, sandbox).
        api_key: API key for ARM authentication.
        **kwargs: Additional configuration parameters.

    Returns:
        Configured TransactionReportingPipeline instance.

    Example:
        pipeline = create_reporting_pipeline(
            lei="5493001KJTIIGC8Y1R12",
            arm_provider=ARMProvider.BLOOMBERG_BTRL,
            arm_environment=ARMEnvironment.UAT,
            api_key="your-api-key",
            batch_size=50,
        )
    """
    arm_config = ARMClientConfig(
        provider=arm_provider,
        environment=arm_environment,
        api_key=api_key,
        lei=lei,
    )

    config = PipelineConfig(
        arm_config=arm_config,
        lei=lei,
        **kwargs,
    )

    return TransactionReportingPipeline(config)


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    # Enums
    "PipelineStatus",
    "ReportQueuePriority",
    # Data classes
    "PipelineConfig",
    "QueuedReport",
    "PipelineMetrics",
    # Main class
    "TransactionReportingPipeline",
    # Factory
    "create_reporting_pipeline",
]
