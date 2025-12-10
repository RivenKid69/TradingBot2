# -*- coding: utf-8 -*-
"""
MiFID II Audit Trail Writer - MiFIR Article 25 Compliance.

This module provides the main interface for writing audit trail records
with chain integrity verification per MiFIR Article 25 requirements.

Features:
    - Write-once (append-only) audit records
    - Cryptographic hash chaining for tamper detection
    - Async and sync write interfaces
    - Automatic batching for performance
    - Integration with retention policy
    - Real-time chain verification

The AuditTrailWriter is the primary entry point for all audit logging
in the trading system.

Usage:
    writer = create_audit_trail_writer(firm_lei="5493001KJTIIGC8Y1R12")
    writer.write_order_submitted(order_id, instrument, side, qty, price)
    writer.write_risk_event(AuditEventType.PRE_TRADE_CHECK_FAILED, "Price collar breach")

References:
    - MiFIR Article 25: Obligation to maintain records
    - RTS 25: Clock synchronisation for timestamps
    - ESMA Guidelines on record keeping
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Callable,
    Union,
    Tuple,
)

from services.core.risk_controls.audit_models import (
    AuditRecord,
    AuditRecordBuilder,
    AuditEventType,
    AuditRecordPriority,
    AuditRecordStatus,
    AuditChainStatus,
    AuditExportRequest,
    AuditExportResult,
    OrderSide,
    create_order_submitted_record,
    create_order_filled_record,
    create_risk_event_record,
    create_system_event_record,
)
from services.core.risk_controls.audit_storage import (
    AuditStorageBackend,
    AuditStorageConfig,
    StorageBackendType,
    StorageMetrics,
    create_audit_storage,
)
from services.core.risk_controls.retention_policy import (
    RetentionManager,
    RetentionPolicyConfig,
    create_retention_manager,
)

logger = logging.getLogger(__name__)


class WriterMode(Enum):
    """Operating mode for the audit trail writer."""

    SYNC = "sync"  # Synchronous writes
    ASYNC = "async"  # Asynchronous with background worker
    BATCH = "batch"  # Batch writes for performance


class WriterState(Enum):
    """State of the audit trail writer."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AuditTrailWriterConfig:
    """
    Configuration for the audit trail writer.

    Attributes:
        firm_lei: Firm's Legal Entity Identifier (required).
        mode: Operating mode (sync, async, batch).
        storage_config: Storage backend configuration.
        retention_config: Retention policy configuration.
        batch_size: Records per batch in batch mode.
        batch_interval_ms: Max wait before flushing batch (ms).
        queue_size: Max pending records in async mode.
        verify_on_write: Verify hash after each write.
        auto_sequence: Auto-increment sequence numbers.
        enable_retention_management: Enable background retention.
        on_write_error: Callback for write errors.
        on_chain_break: Callback for chain integrity failures.
    """

    firm_lei: str = ""
    mode: WriterMode = WriterMode.ASYNC
    storage_config: Optional[AuditStorageConfig] = None
    retention_config: Optional[RetentionPolicyConfig] = None
    batch_size: int = 100
    batch_interval_ms: int = 1000
    queue_size: int = 10000
    verify_on_write: bool = False
    auto_sequence: bool = True
    enable_retention_management: bool = True
    on_write_error: Optional[Callable[[AuditRecord, Exception], None]] = None
    on_chain_break: Optional[Callable[[AuditChainStatus], None]] = None


@dataclass
class WriterMetrics:
    """Metrics for the audit trail writer."""

    records_written: int = 0
    records_queued: int = 0
    records_failed: int = 0
    batches_written: int = 0
    chain_verifications: int = 0
    chain_failures: int = 0
    last_write_timestamp_ns: Optional[int] = None
    last_sequence_number: int = 0
    queue_depth: int = 0
    write_latency_ms: float = 0.0
    avg_batch_size: float = 0.0


class AuditTrailWriter:
    """
    Write-once audit trail with chain integrity verification.

    Per MiFIR Article 25: "Records must be tamper-proof and cannot be altered"

    This class provides:
    - Thread-safe record writing
    - Cryptographic hash chaining
    - Multiple write modes (sync, async, batch)
    - Integration with storage and retention

    Example:
        writer = AuditTrailWriter(config)
        writer.start()

        # Write order event
        writer.write_order_submitted(
            order_id="ORD-12345",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
        )

        # Write risk event
        writer.write_risk_event(
            AuditEventType.PRE_TRADE_CHECK_FAILED,
            "Price collar breach",
        )

        writer.stop()
    """

    def __init__(
        self,
        config: AuditTrailWriterConfig,
        storage: Optional[AuditStorageBackend] = None,
    ):
        """
        Initialize audit trail writer.

        Args:
            config: Writer configuration.
            storage: Optional pre-configured storage backend.
        """
        if not config.firm_lei:
            raise ValueError("firm_lei is required for audit trail writer")

        self.config = config
        self._storage = storage
        self._retention_manager: Optional[RetentionManager] = None

        self._state = WriterState.INITIALIZING
        self._lock = threading.Lock()
        self._sequence_number = 0
        self._last_hash: Optional[str] = None
        self._metrics = WriterMetrics()

        # Async/batch mode components
        self._queue: queue.Queue[AuditRecord] = queue.Queue(maxsize=config.queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._batch_buffer: List[AuditRecord] = []
        self._last_batch_time = time.time()

        # Initialize components
        self._initialize()

    def _initialize(self) -> None:
        """Initialize storage and retention components."""
        try:
            # Initialize storage
            if self._storage is None:
                storage_config = self.config.storage_config or AuditStorageConfig()
                self._storage = create_audit_storage(storage_config)

            # Get last hash from storage
            self._last_hash = self._storage.get_last_hash()

            # Get last sequence number
            latest = self._storage.get_latest_record()
            if latest:
                self._sequence_number = latest.sequence_number

            # Initialize retention manager
            if self.config.enable_retention_management:
                retention_config = self.config.retention_config or RetentionPolicyConfig()
                self._retention_manager = create_retention_manager(
                    self._storage, retention_config
                )

            self._state = WriterState.STOPPED
            logger.info(
                f"Audit trail writer initialized (mode={self.config.mode.value}, "
                f"last_seq={self._sequence_number})"
            )

        except Exception as e:
            self._state = WriterState.ERROR
            logger.error(f"Failed to initialize audit trail writer: {e}")
            raise

    def start(self) -> None:
        """Start the audit trail writer."""
        if self._state == WriterState.RUNNING:
            return

        self._state = WriterState.RUNNING

        # Start background worker for async/batch modes
        if self.config.mode in {WriterMode.ASYNC, WriterMode.BATCH}:
            self._worker_thread = threading.Thread(
                target=self._writer_worker,
                daemon=True,
                name="audit-trail-writer",
            )
            self._worker_thread.start()

        # Start retention manager
        if self._retention_manager:
            self._retention_manager.start()

        logger.info("Audit trail writer started")

    def stop(self, flush: bool = True, timeout: float = 10.0) -> None:
        """
        Stop the audit trail writer.

        Args:
            flush: Whether to flush pending records before stopping.
            timeout: Maximum wait time for flush.
        """
        if self._state != WriterState.RUNNING:
            return

        self._state = WriterState.STOPPING

        # Flush pending records
        if flush:
            self.flush(timeout=timeout)

        # Stop worker thread
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)

        # Stop retention manager
        if self._retention_manager:
            self._retention_manager.stop()

        # Close storage
        if self._storage:
            self._storage.close()

        self._state = WriterState.STOPPED
        logger.info("Audit trail writer stopped")

    def write(self, record: AuditRecord) -> bool:
        """
        Write a single audit record.

        Args:
            record: The audit record to write.

        Returns:
            True if successfully written (sync) or queued (async).

        Raises:
            RuntimeError: If writer is not running.
        """
        if self._state != WriterState.RUNNING:
            raise RuntimeError("Audit trail writer is not running")

        # Set firm LEI if not set
        if not record.firm_lei:
            record.firm_lei = self.config.firm_lei

        # Set sequence number
        if self.config.auto_sequence:
            with self._lock:
                self._sequence_number += 1
                record.sequence_number = self._sequence_number

        # Route to appropriate write method
        if self.config.mode == WriterMode.SYNC:
            return self._write_sync(record)
        else:
            return self._queue_record(record)

    def _write_sync(self, record: AuditRecord) -> bool:
        """Write record synchronously."""
        start_time = time.time()

        try:
            with self._lock:
                # Set chain hash
                record.previous_record_hash = self._last_hash

                # Compute hash
                record.record_hash = record.compute_hash()

                # Write to storage
                success = self._storage.append(record)

                if success:
                    self._last_hash = record.record_hash
                    self._metrics.records_written += 1
                    self._metrics.last_write_timestamp_ns = time.time_ns()
                    self._metrics.last_sequence_number = record.sequence_number

                    # Verify if configured
                    if self.config.verify_on_write:
                        self._verify_last_record(record)
                else:
                    self._metrics.records_failed += 1
                    if self.config.on_write_error:
                        self.config.on_write_error(record, RuntimeError("Storage write failed"))

                # Update latency
                elapsed_ms = (time.time() - start_time) * 1000
                self._metrics.write_latency_ms = (
                    self._metrics.write_latency_ms * 0.9 + elapsed_ms * 0.1
                )

                return success

        except Exception as e:
            self._metrics.records_failed += 1
            logger.error(f"Error writing audit record: {e}")
            if self.config.on_write_error:
                self.config.on_write_error(record, e)
            return False

    def _queue_record(self, record: AuditRecord) -> bool:
        """Queue record for async writing."""
        try:
            self._queue.put_nowait(record)
            self._metrics.records_queued += 1
            self._metrics.queue_depth = self._queue.qsize()
            return True
        except queue.Full:
            self._metrics.records_failed += 1
            logger.warning("Audit trail queue full, dropping record")
            if self.config.on_write_error:
                self.config.on_write_error(record, RuntimeError("Queue full"))
            return False

    def _writer_worker(self) -> None:
        """Background worker for async/batch writing."""
        while self._state in {WriterState.RUNNING, WriterState.STOPPING}:
            try:
                # Get record from queue
                try:
                    record = self._queue.get(timeout=0.1)
                except queue.Empty:
                    # Check if batch needs flushing
                    if (
                        self.config.mode == WriterMode.BATCH
                        and self._batch_buffer
                        and (time.time() - self._last_batch_time) * 1000
                        >= self.config.batch_interval_ms
                    ):
                        self._flush_batch()
                    continue

                if self.config.mode == WriterMode.ASYNC:
                    # Write immediately
                    self._write_sync(record)
                else:
                    # Add to batch
                    self._batch_buffer.append(record)
                    if len(self._batch_buffer) >= self.config.batch_size:
                        self._flush_batch()

                self._queue.task_done()

            except Exception as e:
                logger.error(f"Audit trail worker error: {e}")

        # Flush remaining on stop
        if self._batch_buffer:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Flush batch buffer to storage."""
        if not self._batch_buffer:
            return

        with self._lock:
            # Set chain hashes for all records
            for record in self._batch_buffer:
                record.previous_record_hash = self._last_hash
                record.record_hash = record.compute_hash()
                self._last_hash = record.record_hash

            # Write batch
            written = self._storage.append_batch(self._batch_buffer)

            # Update metrics
            self._metrics.records_written += written
            self._metrics.batches_written += 1
            self._metrics.records_failed += len(self._batch_buffer) - written

            if self._batch_buffer:
                self._metrics.last_write_timestamp_ns = time.time_ns()
                self._metrics.last_sequence_number = self._batch_buffer[-1].sequence_number

            # Update average batch size
            self._metrics.avg_batch_size = (
                self._metrics.avg_batch_size * 0.9 + len(self._batch_buffer) * 0.1
            )

            self._batch_buffer.clear()
            self._last_batch_time = time.time()

    def flush(self, timeout: float = 10.0) -> bool:
        """
        Flush all pending records.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            True if all records were flushed successfully.
        """
        start_time = time.time()

        # Wait for queue to empty
        while not self._queue.empty():
            if time.time() - start_time > timeout:
                logger.warning("Flush timeout reached")
                return False
            time.sleep(0.01)

        # Flush batch buffer
        if self.config.mode == WriterMode.BATCH:
            self._flush_batch()

        return True

    def _verify_last_record(self, record: AuditRecord) -> None:
        """Verify the last written record."""
        self._metrics.chain_verifications += 1

        stored = self._storage.read_by_id(record.record_id)
        if stored is None:
            self._metrics.chain_failures += 1
            self._handle_chain_break("Record not found after write")
            return

        if stored.record_hash != record.record_hash:
            self._metrics.chain_failures += 1
            self._handle_chain_break("Hash mismatch after write")

    def _handle_chain_break(self, message: str) -> None:
        """Handle chain integrity failure."""
        logger.error(f"Audit chain integrity failure: {message}")

        status = AuditChainStatus(
            is_valid=False,
            records_checked=1,
            error_message=message,
        )

        if self.config.on_chain_break:
            self.config.on_chain_break(status)

    # ===== High-Level Write Methods =====

    def write_order_submitted(
        self,
        order_id: str,
        instrument_isin: str,
        side: OrderSide,
        quantity: Union[Decimal, float, int, str],
        price: Union[Decimal, float, int, str],
        currency: str = "EUR",
        venue_mic: str = "XOFF",
        algorithm_id: Optional[str] = None,
        trader_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write ORDER_SUBMITTED audit record.

        Args:
            order_id: Unique order identifier.
            instrument_isin: Instrument ISIN.
            side: Order side (BUY/SELL).
            quantity: Order quantity.
            price: Order price.
            currency: Price currency.
            venue_mic: Trading venue MIC.
            algorithm_id: Algorithm ID (if algorithmic order).
            trader_id: Trader ID.
            client_order_id: Client-assigned order ID.
            details: Additional details.

        Returns:
            True if record was written/queued successfully.
        """
        record = create_order_submitted_record(
            firm_lei=self.config.firm_lei,
            order_id=order_id,
            instrument_isin=instrument_isin,
            side=side,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)),
            currency=currency,
            venue_mic=venue_mic,
            algorithm_id=algorithm_id,
            trader_id=trader_id,
            client_order_id=client_order_id,
            details=details,
        )
        return self.write(record)

    def write_order_filled(
        self,
        order_id: str,
        instrument_isin: str,
        side: OrderSide,
        fill_quantity: Union[Decimal, float, int, str],
        fill_price: Union[Decimal, float, int, str],
        currency: str = "EUR",
        venue_mic: str = "XOFF",
        is_partial: bool = False,
        remaining_quantity: Optional[Union[Decimal, float, int, str]] = None,
        algorithm_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write ORDER_FILLED or ORDER_PARTIALLY_FILLED audit record.

        Args:
            order_id: Order identifier.
            instrument_isin: Instrument ISIN.
            side: Order side.
            fill_quantity: Filled quantity.
            fill_price: Fill price.
            currency: Price currency.
            venue_mic: Trading venue MIC.
            is_partial: Whether this is a partial fill.
            remaining_quantity: Remaining quantity (for partial fills).
            algorithm_id: Algorithm ID.
            details: Additional details.

        Returns:
            True if record was written/queued successfully.
        """
        remaining = (
            Decimal(str(remaining_quantity))
            if remaining_quantity is not None
            else None
        )

        record = create_order_filled_record(
            firm_lei=self.config.firm_lei,
            order_id=order_id,
            instrument_isin=instrument_isin,
            side=side,
            fill_quantity=Decimal(str(fill_quantity)),
            fill_price=Decimal(str(fill_price)),
            currency=currency,
            venue_mic=venue_mic,
            is_partial=is_partial,
            remaining_quantity=remaining,
            algorithm_id=algorithm_id,
            details=details,
        )
        return self.write(record)

    def write_order_cancelled(
        self,
        order_id: str,
        reason: str = "",
        algorithm_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write ORDER_CANCELLED audit record.

        Args:
            order_id: Order identifier.
            reason: Cancellation reason.
            algorithm_id: Algorithm ID.
            details: Additional details.

        Returns:
            True if record was written/queued successfully.
        """
        event_details = details.copy() if details else {}
        event_details["reason"] = reason

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_CANCELLED)
            .firm_lei(self.config.firm_lei)
            .order_id(order_id)
            .details(event_details)
            .priority(AuditRecordPriority.NORMAL)
            .build_unsafe()
        )

        if algorithm_id:
            record.algorithm_id = algorithm_id

        return self.write(record)

    def write_order_rejected(
        self,
        order_id: str,
        reason: str,
        rejection_code: Optional[str] = None,
        algorithm_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write ORDER_REJECTED audit record.

        Args:
            order_id: Order identifier.
            reason: Rejection reason.
            rejection_code: Optional rejection code.
            algorithm_id: Algorithm ID.
            details: Additional details.

        Returns:
            True if record was written/queued successfully.
        """
        event_details = details.copy() if details else {}
        event_details["reason"] = reason
        if rejection_code:
            event_details["rejection_code"] = rejection_code

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_REJECTED)
            .firm_lei(self.config.firm_lei)
            .order_id(order_id)
            .details(event_details)
            .priority(AuditRecordPriority.HIGH)
            .build_unsafe()
        )

        if algorithm_id:
            record.algorithm_id = algorithm_id

        return self.write(record)

    def write_risk_event(
        self,
        event_type: AuditEventType,
        reason: str,
        algorithm_id: Optional[str] = None,
        order_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write risk-related audit record.

        Args:
            event_type: Risk event type.
            reason: Reason for the risk event.
            algorithm_id: Algorithm ID.
            order_id: Related order ID.
            details: Additional details.

        Returns:
            True if record was written/queued successfully.
        """
        record = create_risk_event_record(
            firm_lei=self.config.firm_lei,
            event_type=event_type,
            reason=reason,
            algorithm_id=algorithm_id,
            order_id=order_id,
            details=details,
        )
        return self.write(record)

    def write_system_event(
        self,
        event_type: AuditEventType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write system-level audit record.

        Args:
            event_type: System event type.
            message: Event message/description.
            details: Additional details.

        Returns:
            True if record was written/queued successfully.
        """
        record = create_system_event_record(
            firm_lei=self.config.firm_lei,
            event_type=event_type,
            message=message,
            details=details,
        )
        return self.write(record)

    def write_algorithm_event(
        self,
        event_type: AuditEventType,
        algorithm_id: str,
        version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write algorithm-related audit record.

        Args:
            event_type: Algorithm event type.
            algorithm_id: Algorithm identifier.
            version: Algorithm version.
            parameters: Algorithm parameters (for changes).
            details: Additional details.

        Returns:
            True if record was written/queued successfully.
        """
        event_details = details.copy() if details else {}
        if version:
            event_details["version"] = version
        if parameters:
            event_details["parameters"] = parameters

        record = (
            AuditRecordBuilder()
            .event_type(event_type)
            .firm_lei(self.config.firm_lei)
            .algorithm(algorithm_id)
            .details(event_details)
            .priority(AuditRecordPriority.HIGH)
            .build_unsafe()
        )

        return self.write(record)

    def write_kill_switch_event(
        self,
        triggered_by: str,
        reason: str,
        scope: str = "ALL",
        orders_cancelled: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write kill switch audit record.

        Args:
            triggered_by: Person or system that triggered.
            reason: Reason for triggering.
            scope: Scope of cancellation.
            orders_cancelled: Number of orders cancelled.
            details: Additional details.

        Returns:
            True if record was written/queued successfully.
        """
        event_details = details.copy() if details else {}
        event_details["triggered_by"] = triggered_by
        event_details["reason"] = reason
        event_details["scope"] = scope
        event_details["orders_cancelled"] = orders_cancelled

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.KILL_SWITCH_TRIGGERED)
            .firm_lei(self.config.firm_lei)
            .details(event_details)
            .priority(AuditRecordPriority.CRITICAL)
            .build_unsafe()
        )

        return self.write(record)

    # ===== Query Methods =====

    def verify_chain(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> AuditChainStatus:
        """
        Verify chain integrity.

        Args:
            start_time: Start of verification range.
            end_time: End of verification range.

        Returns:
            AuditChainStatus with verification results.
        """
        return self._storage.verify_chain(start_time, end_time)

    def export(self, request: AuditExportRequest) -> AuditExportResult:
        """
        Export audit records.

        Args:
            request: Export request parameters.

        Returns:
            AuditExportResult with export details.
        """
        return self._storage.export(request)

    def get_metrics(self) -> WriterMetrics:
        """Get writer metrics."""
        self._metrics.queue_depth = self._queue.qsize()
        return self._metrics

    def get_storage_metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        return self._storage.get_metrics()

    @property
    def state(self) -> WriterState:
        """Get current writer state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if writer is running."""
        return self._state == WriterState.RUNNING


def create_audit_trail_writer(
    firm_lei: str,
    mode: WriterMode = WriterMode.ASYNC,
    storage_path: Optional[str] = None,
    storage_type: StorageBackendType = StorageBackendType.SQLITE,
    enable_retention: bool = True,
    auto_start: bool = True,
) -> AuditTrailWriter:
    """
    Factory function to create audit trail writer.

    Args:
        firm_lei: Firm's Legal Entity Identifier.
        mode: Operating mode.
        storage_path: Path for storage backend.
        storage_type: Type of storage backend.
        enable_retention: Enable retention management.
        auto_start: Start writer automatically.

    Returns:
        Configured AuditTrailWriter instance.
    """
    storage_config = AuditStorageConfig(
        backend_type=storage_type,
        database_path=storage_path or "state/compliance/audit_trail.db",
    )

    config = AuditTrailWriterConfig(
        firm_lei=firm_lei,
        mode=mode,
        storage_config=storage_config,
        enable_retention_management=enable_retention,
    )

    writer = AuditTrailWriter(config)

    if auto_start:
        writer.start()

    return writer


__all__ = [
    # Enums
    "WriterMode",
    "WriterState",
    # Config
    "AuditTrailWriterConfig",
    # Metrics
    "WriterMetrics",
    # Main class
    "AuditTrailWriter",
    # Factory
    "create_audit_trail_writer",
]
