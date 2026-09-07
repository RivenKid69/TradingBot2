# -*- coding: utf-8 -*-
"""
MiFID II Audit Trail Storage Backends - MiFIR Article 25 Compliance.

This module provides storage backends for persisting audit trail records
per MiFIR Article 25 requirements:

- Write-once (append-only) storage for tamper resistance
- High-performance indexing for fast retrieval
- Support for 5-7 year retention periods
- Chain integrity verification
- Export capabilities for NCA requests

Storage Options (implemented):
    - SQLiteAuditStorage: For development and small deployments
    - FileAuditStorage: JSON Lines format for simple deployments
    - MemoryAuditStorage: For testing (not persistent)

Planned (not yet implemented):
    - PostgreSQL/TimescaleDB: Planned for enterprise multi-node deployments.
      Status: Raises NotImplementedError. See create_audit_storage() for details.

References:
    - MiFIR Article 25: Obligation to maintain records
    - RTS 24 (Regulation 2017/580): Order book data
    - ESMA Guidelines on record keeping
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Iterator,
    Generator,
    Callable,
    Union,
    Tuple,
)

from services.core.risk_controls.audit_models import (
    AuditRecord,
    AuditEventType,
    AuditRecordStatus,
    AuditChainStatus,
    AuditExportRequest,
    AuditExportResult,
)


logger = logging.getLogger(__name__)


class StorageBackendType(Enum):
    """Types of audit storage backends.

    Currently implemented backends:
        - MEMORY: In-memory storage (development/testing only, not persistent)
        - SQLITE: SQLite file-based storage (single-node deployments)
        - FILE: JSON file-based storage (simple deployments)

    Planned backends (not yet implemented):
        - POSTGRESQL: PostgreSQL storage (enterprise multi-node deployments)
          Requires: psycopg2 or asyncpg. See docs for installation.
    """

    MEMORY = "memory"
    SQLITE = "sqlite"
    FILE = "file"
    # Planned: PostgreSQL for enterprise scalable deployments.
    # See create_audit_storage() for status.
    POSTGRESQL = "postgresql"


class StorageState(Enum):
    """State of the storage backend."""

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    READONLY = "readonly"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class AuditStorageConfig:
    """
    Configuration for audit storage backend.

    Attributes:
        backend_type: Type of storage backend.
        database_path: Path to database file (SQLite/File).
        database_url: Connection URL (PostgreSQL).
        table_name: Name of the audit trail table.
        retention_years: Default retention period in years.
        max_batch_size: Maximum records per batch write.
        sync_mode: fsync mode for durability (off/normal/full).
        compression_enabled: Enable compression for archived records.
        integrity_check_on_read: Verify hash on every read.
        auto_archive_days: Days before auto-archiving (0 = disabled).
        backup_enabled: Enable automatic backups.
        backup_path: Path for backup files.
    """

    backend_type: StorageBackendType = StorageBackendType.SQLITE
    database_path: str = "state/compliance/audit_trail.db"
    database_url: str = ""
    table_name: str = "audit_trail"
    retention_years: int = 5
    max_batch_size: int = 1000
    sync_mode: str = "normal"  # off, normal, full
    compression_enabled: bool = False
    integrity_check_on_read: bool = False
    auto_archive_days: int = 365
    backup_enabled: bool = True
    backup_path: str = "state/compliance/audit_backup"


@dataclass
class StorageMetrics:
    """Metrics for storage operations."""

    records_written: int = 0
    records_read: int = 0
    write_errors: int = 0
    read_errors: int = 0
    integrity_checks: int = 0
    integrity_failures: int = 0
    last_write_timestamp: Optional[int] = None
    last_read_timestamp: Optional[int] = None
    total_storage_bytes: int = 0


class AuditStorageBackend(ABC):
    """
    Abstract base class for audit storage backends.

    All implementations must be append-only (write-once) for tamper resistance.
    Records cannot be modified or deleted except through official retention
    policy procedures.
    """

    @abstractmethod
    def append(self, record: AuditRecord) -> bool:
        """
        Append a single record to the audit trail.

        Args:
            record: AuditRecord to append.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def append_batch(self, records: List[AuditRecord]) -> int:
        """
        Append multiple records atomically.

        Args:
            records: List of AuditRecords to append.

        Returns:
            Number of records successfully written.
        """
        pass

    @abstractmethod
    def read_by_id(self, record_id: str) -> Optional[AuditRecord]:
        """
        Read a single record by ID.

        Args:
            record_id: Unique record identifier.

        Returns:
            AuditRecord if found, None otherwise.
        """
        pass

    @abstractmethod
    def read_range(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[AuditEventType]] = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> List[AuditRecord]:
        """
        Read records within a time range.

        Args:
            start_time: Start of time range.
            end_time: End of time range.
            event_types: Filter by event types (None = all).
            limit: Maximum records to return.
            offset: Number of records to skip.

        Returns:
            List of matching AuditRecords.
        """
        pass

    @abstractmethod
    def read_by_order_id(self, order_id: str) -> List[AuditRecord]:
        """
        Read all records for a specific order.

        Args:
            order_id: Order identifier.

        Returns:
            List of AuditRecords related to the order.
        """
        pass

    @abstractmethod
    def read_by_algorithm_id(
        self,
        algorithm_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AuditRecord]:
        """
        Read all records for a specific algorithm.

        Args:
            algorithm_id: Algorithm identifier.
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            List of AuditRecords for the algorithm.
        """
        pass

    @abstractmethod
    def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> int:
        """
        Count records matching criteria.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.
            event_types: Optional event type filter.

        Returns:
            Number of matching records.
        """
        pass

    @abstractmethod
    def get_latest_record(self) -> Optional[AuditRecord]:
        """
        Get the most recent record.

        Returns:
            Latest AuditRecord or None if empty.
        """
        pass

    @abstractmethod
    def get_last_hash(self) -> Optional[str]:
        """
        Get the hash of the most recent record.

        Used for chain integrity verification.

        Returns:
            Hash string or None if no records.
        """
        pass

    @abstractmethod
    def verify_chain(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 1000,
    ) -> AuditChainStatus:
        """
        Verify chain integrity for records in time range.

        Args:
            start_time: Start of verification range.
            end_time: End of verification range.
            batch_size: Records to check per batch.

        Returns:
            AuditChainStatus with verification results.
        """
        pass

    @abstractmethod
    def export(
        self,
        request: AuditExportRequest,
    ) -> AuditExportResult:
        """
        Export audit records per NCA request.

        Args:
            request: Export request parameters.

        Returns:
            AuditExportResult with export details.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage backend and release resources."""
        pass


class MemoryAuditStorage(AuditStorageBackend):
    """
    In-memory audit storage for testing.

    NOT suitable for production - all data is lost on restart.
    """

    def __init__(self, config: Optional[AuditStorageConfig] = None):
        """Initialize memory storage."""
        self.config = config or AuditStorageConfig(backend_type=StorageBackendType.MEMORY)
        self._records: List[AuditRecord] = []
        self._records_by_id: Dict[str, AuditRecord] = {}
        self._records_by_order: Dict[str, List[AuditRecord]] = {}
        self._records_by_algorithm: Dict[str, List[AuditRecord]] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._metrics = StorageMetrics()
        self._state = StorageState.READY
        self._last_hash: Optional[str] = None

    def append(self, record: AuditRecord) -> bool:
        """Append a single record."""
        with self._lock:
            try:
                # Set chain hash
                record.previous_record_hash = self._last_hash

                # Update status BEFORE computing hash (status is included in hash)
                record.status = AuditRecordStatus.WRITTEN

                # Always recompute hash to include chain link and status
                record.record_hash = record.compute_hash()

                # Store record
                self._records.append(record)
                self._records_by_id[record.record_id] = record

                # Index by order_id
                if record.order_id:
                    if record.order_id not in self._records_by_order:
                        self._records_by_order[record.order_id] = []
                    self._records_by_order[record.order_id].append(record)

                # Index by algorithm_id
                if record.algorithm_id:
                    if record.algorithm_id not in self._records_by_algorithm:
                        self._records_by_algorithm[record.algorithm_id] = []
                    self._records_by_algorithm[record.algorithm_id].append(record)

                # Update chain hash
                self._last_hash = record.record_hash

                # Update metrics
                self._metrics.records_written += 1
                self._metrics.last_write_timestamp = time.time_ns()

                return True
            except Exception as e:
                logger.error(f"Error appending record: {e}")
                self._metrics.write_errors += 1
                return False

    def append_batch(self, records: List[AuditRecord]) -> int:
        """Append multiple records."""
        count = 0
        for record in records:
            if self.append(record):
                count += 1
        return count

    def read_by_id(self, record_id: str) -> Optional[AuditRecord]:
        """Read record by ID."""
        with self._lock:
            self._metrics.records_read += 1
            self._metrics.last_read_timestamp = time.time_ns()
            return self._records_by_id.get(record_id)

    def read_range(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[AuditEventType]] = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> List[AuditRecord]:
        """Read records in time range."""
        with self._lock:
            start_ns = int(start_time.timestamp() * 1e9)
            end_ns = int(end_time.timestamp() * 1e9)

            results = []
            for record in self._records:
                if start_ns <= record.event_timestamp_ns <= end_ns:
                    if event_types is None or record.event_type in event_types:
                        results.append(record)

            self._metrics.records_read += len(results)
            self._metrics.last_read_timestamp = time.time_ns()

            return results[offset : offset + limit]

    def read_by_order_id(self, order_id: str) -> List[AuditRecord]:
        """Read records for an order."""
        with self._lock:
            results = self._records_by_order.get(order_id, [])
            self._metrics.records_read += len(results)
            self._metrics.last_read_timestamp = time.time_ns()
            return results

    def read_by_algorithm_id(
        self,
        algorithm_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AuditRecord]:
        """Read records for an algorithm."""
        with self._lock:
            records = self._records_by_algorithm.get(algorithm_id, [])

            if start_time or end_time:
                start_ns = int(start_time.timestamp() * 1e9) if start_time else 0
                end_ns = int(end_time.timestamp() * 1e9) if end_time else time.time_ns()

                records = [
                    r
                    for r in records
                    if start_ns <= r.event_timestamp_ns <= end_ns
                ]

            self._metrics.records_read += len(records)
            self._metrics.last_read_timestamp = time.time_ns()
            return records

    def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> int:
        """Count matching records."""
        with self._lock:
            if not start_time and not end_time and not event_types:
                return len(self._records)

            count = 0
            start_ns = int(start_time.timestamp() * 1e9) if start_time else 0
            end_ns = int(end_time.timestamp() * 1e9) if end_time else time.time_ns()

            for record in self._records:
                if start_ns <= record.event_timestamp_ns <= end_ns:
                    if event_types is None or record.event_type in event_types:
                        count += 1

            return count

    def get_latest_record(self) -> Optional[AuditRecord]:
        """Get latest record."""
        with self._lock:
            return self._records[-1] if self._records else None

    def get_last_hash(self) -> Optional[str]:
        """Get hash of latest record."""
        with self._lock:
            return self._last_hash

    def verify_chain(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 1000,
    ) -> AuditChainStatus:
        """Verify chain integrity."""
        with self._lock:
            records_checked = 0
            previous_hash = None

            for i, record in enumerate(self._records):
                # Apply time filters
                if start_time:
                    start_ns = int(start_time.timestamp() * 1e9)
                    if record.event_timestamp_ns < start_ns:
                        continue
                if end_time:
                    end_ns = int(end_time.timestamp() * 1e9)
                    if record.event_timestamp_ns > end_ns:
                        break

                records_checked += 1
                self._metrics.integrity_checks += 1

                # Verify hash
                computed = record.compute_hash()
                if computed != record.record_hash:
                    self._metrics.integrity_failures += 1
                    return AuditChainStatus(
                        is_valid=False,
                        records_checked=records_checked,
                        first_invalid_record_id=record.record_id,
                        first_invalid_index=i,
                        error_message=f"Hash mismatch at record {i}",
                    )

                # Verify chain
                if previous_hash is not None:
                    if record.previous_record_hash != previous_hash:
                        self._metrics.integrity_failures += 1
                        return AuditChainStatus(
                            is_valid=False,
                            records_checked=records_checked,
                            first_invalid_record_id=record.record_id,
                            first_invalid_index=i,
                            error_message=f"Chain break at record {i}",
                        )

                previous_hash = record.record_hash

            return AuditChainStatus(
                is_valid=True,
                records_checked=records_checked,
            )

    def export(self, request: AuditExportRequest) -> AuditExportResult:
        """Export records."""
        with self._lock:
            try:
                # Get matching records
                records = []
                start_ns = int(request.start_datetime.timestamp() * 1e9) if request.start_datetime else 0
                end_ns = int(request.end_datetime.timestamp() * 1e9) if request.end_datetime else time.time_ns()

                for record in self._records:
                    if start_ns <= record.event_timestamp_ns <= end_ns:
                        if request.event_types is None or record.event_type in request.event_types:
                            if request.order_ids is None or record.order_id in request.order_ids:
                                if request.algorithm_ids is None or record.algorithm_id in request.algorithm_ids:
                                    if request.instrument_isins is None or record.instrument_isin in request.instrument_isins:
                                        records.append(record)

                # Verify chain if requested
                chain_status = None
                if request.include_chain_verification:
                    chain_status = self.verify_chain(request.start_datetime, request.end_datetime)

                return AuditExportResult(
                    request_id=request.request_id,
                    success=True,
                    records_exported=len(records),
                    chain_verification=chain_status,
                )
            except Exception as e:
                return AuditExportResult(
                    request_id=request.request_id,
                    success=False,
                    records_exported=0,
                    error_message=str(e),
                )

    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        return self._metrics

    def close(self) -> None:
        """Close storage."""
        self._state = StorageState.CLOSED

    def clear(self) -> None:
        """Clear all records (for testing only)."""
        with self._lock:
            self._records.clear()
            self._records_by_id.clear()
            self._records_by_order.clear()
            self._records_by_algorithm.clear()
            self._last_hash = None
            self._metrics = StorageMetrics()


class SQLiteAuditStorage(AuditStorageBackend):
    """
    SQLite-based audit storage for development and small deployments.

    Features:
        - ACID transactions for data integrity
        - B-tree indexes for fast queries
        - WAL mode for concurrent reads
        - Automatic schema management

    For production with high volume, consider PostgreSQL with TimescaleDB.
    """

    def __init__(self, config: Optional[AuditStorageConfig] = None):
        """Initialize SQLite storage."""
        self.config = config or AuditStorageConfig(backend_type=StorageBackendType.SQLITE)
        self._db_path = self.config.database_path
        self._table_name = self.config.table_name
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._metrics = StorageMetrics()
        self._state = StorageState.UNINITIALIZED
        self._last_hash: Optional[str] = None

        # Initialize database
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")

        # Set sync mode
        if self.config.sync_mode == "off":
            conn.execute("PRAGMA synchronous=OFF")
        elif self.config.sync_mode == "full":
            conn.execute("PRAGMA synchronous=FULL")
        else:
            conn.execute("PRAGMA synchronous=NORMAL")

        return conn

    def _init_database(self) -> None:
        """Initialize database schema."""
        # Create directory if needed
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Create main audit trail table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    event_timestamp_ns INTEGER NOT NULL,
                    record_timestamp_ns INTEGER NOT NULL,
                    firm_lei TEXT NOT NULL,
                    algorithm_id TEXT,
                    trader_id TEXT,
                    order_id TEXT,
                    client_order_id TEXT,
                    instrument_isin TEXT,
                    instrument_symbol TEXT,
                    venue_mic TEXT,
                    side TEXT,
                    quantity TEXT,
                    price TEXT,
                    currency TEXT,
                    notional_value TEXT,
                    details TEXT,
                    sequence_number INTEGER,
                    priority TEXT,
                    status TEXT,
                    previous_record_hash TEXT,
                    record_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for fast retrieval
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_event_timestamp
                ON {self._table_name}(event_timestamp_ns)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_order_id
                ON {self._table_name}(order_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_algorithm_id
                ON {self._table_name}(algorithm_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_instrument_isin
                ON {self._table_name}(instrument_isin)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_event_type
                ON {self._table_name}(event_type)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_firm_lei
                ON {self._table_name}(firm_lei)
            """)

            conn.commit()

            # Get last hash
            cursor.execute(f"""
                SELECT record_hash FROM {self._table_name}
                ORDER BY id DESC LIMIT 1
            """)
            row = cursor.fetchone()
            self._last_hash = row["record_hash"] if row else None

            self._state = StorageState.READY
            logger.info(f"SQLite audit storage initialized at {self._db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._state = StorageState.ERROR
            raise
        finally:
            conn.close()

    def append(self, record: AuditRecord) -> bool:
        """Append a single record."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Set chain hash
                record.previous_record_hash = self._last_hash

                # Update status BEFORE computing hash (status is included in hash)
                record.status = AuditRecordStatus.WRITTEN

                # Always recompute hash to include chain link and status
                record.record_hash = record.compute_hash()

                # Insert record
                cursor.execute(
                    f"""
                    INSERT INTO {self._table_name} (
                        record_id, event_type, event_timestamp_ns, record_timestamp_ns,
                        firm_lei, algorithm_id, trader_id, order_id, client_order_id,
                        instrument_isin, instrument_symbol, venue_mic, side,
                        quantity, price, currency, notional_value, details,
                        sequence_number, priority, status,
                        previous_record_hash, record_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.event_type.value,
                        record.event_timestamp_ns,
                        record.record_timestamp_ns,
                        record.firm_lei,
                        record.algorithm_id,
                        record.trader_id,
                        record.order_id,
                        record.client_order_id,
                        record.instrument_isin,
                        record.instrument_symbol,
                        record.venue_mic,
                        record.side.value if record.side else None,
                        str(record.quantity) if record.quantity else None,
                        str(record.price) if record.price else None,
                        record.currency,
                        str(record.notional_value) if record.notional_value else None,
                        json.dumps(record.details),
                        record.sequence_number,
                        record.priority.value,
                        record.status.value,
                        record.previous_record_hash,
                        record.record_hash,
                    ),
                )

                conn.commit()

                # Update chain hash
                self._last_hash = record.record_hash

                # Update metrics
                self._metrics.records_written += 1
                self._metrics.last_write_timestamp = time.time_ns()

                return True

            except sqlite3.IntegrityError as e:
                logger.error(f"Duplicate record_id: {record.record_id}")
                self._metrics.write_errors += 1
                return False
            except Exception as e:
                logger.error(f"Error appending record: {e}")
                self._metrics.write_errors += 1
                return False
            finally:
                conn.close()

    def append_batch(self, records: List[AuditRecord]) -> int:
        """Append multiple records in a single transaction."""
        if not records:
            return 0

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                count = 0

                for record in records:
                    try:
                        # Set chain hash
                        record.previous_record_hash = self._last_hash

                        # Compute hash if not set
                        if not record.record_hash:
                            record.record_hash = record.compute_hash()

                        # Update status
                        record.status = AuditRecordStatus.WRITTEN

                        # Insert record
                        cursor.execute(
                            f"""
                            INSERT INTO {self._table_name} (
                                record_id, event_type, event_timestamp_ns, record_timestamp_ns,
                                firm_lei, algorithm_id, trader_id, order_id, client_order_id,
                                instrument_isin, instrument_symbol, venue_mic, side,
                                quantity, price, currency, notional_value, details,
                                sequence_number, priority, status,
                                previous_record_hash, record_hash
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                record.record_id,
                                record.event_type.value,
                                record.event_timestamp_ns,
                                record.record_timestamp_ns,
                                record.firm_lei,
                                record.algorithm_id,
                                record.trader_id,
                                record.order_id,
                                record.client_order_id,
                                record.instrument_isin,
                                record.instrument_symbol,
                                record.venue_mic,
                                record.side.value if record.side else None,
                                str(record.quantity) if record.quantity else None,
                                str(record.price) if record.price else None,
                                record.currency,
                                str(record.notional_value) if record.notional_value else None,
                                json.dumps(record.details),
                                record.sequence_number,
                                record.priority.value,
                                record.status.value,
                                record.previous_record_hash,
                                record.record_hash,
                            ),
                        )

                        # Update chain hash
                        self._last_hash = record.record_hash
                        count += 1

                    except sqlite3.IntegrityError:
                        logger.warning(f"Skipping duplicate record: {record.record_id}")
                        self._metrics.write_errors += 1

                conn.commit()

                # Update metrics
                self._metrics.records_written += count
                self._metrics.last_write_timestamp = time.time_ns()

                return count

            except Exception as e:
                logger.error(f"Error in batch append: {e}")
                conn.rollback()
                self._metrics.write_errors += len(records)
                return 0
            finally:
                conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> AuditRecord:
        """Convert database row to AuditRecord."""
        from decimal import Decimal

        from services.core.risk_controls.audit_models import (
            AuditRecordPriority,
            OrderSide,
        )

        return AuditRecord(
            record_id=row["record_id"],
            event_type=AuditEventType(row["event_type"]),
            event_timestamp_ns=row["event_timestamp_ns"],
            record_timestamp_ns=row["record_timestamp_ns"],
            firm_lei=row["firm_lei"],
            algorithm_id=row["algorithm_id"],
            trader_id=row["trader_id"],
            order_id=row["order_id"],
            client_order_id=row["client_order_id"],
            instrument_isin=row["instrument_isin"],
            instrument_symbol=row["instrument_symbol"],
            venue_mic=row["venue_mic"],
            side=OrderSide(row["side"]) if row["side"] else None,
            quantity=Decimal(row["quantity"]) if row["quantity"] else None,
            price=Decimal(row["price"]) if row["price"] else None,
            currency=row["currency"],
            notional_value=Decimal(row["notional_value"]) if row["notional_value"] else None,
            details=json.loads(row["details"]) if row["details"] else {},
            sequence_number=row["sequence_number"] or 0,
            priority=AuditRecordPriority(row["priority"]) if row["priority"] else AuditRecordPriority.NORMAL,
            status=AuditRecordStatus(row["status"]) if row["status"] else AuditRecordStatus.WRITTEN,
            previous_record_hash=row["previous_record_hash"],
            record_hash=row["record_hash"],
        )

    def read_by_id(self, record_id: str) -> Optional[AuditRecord]:
        """Read record by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM {self._table_name} WHERE record_id = ?",
                (record_id,),
            )
            row = cursor.fetchone()

            self._metrics.records_read += 1
            self._metrics.last_read_timestamp = time.time_ns()

            if row:
                record = self._row_to_record(row)

                # Verify integrity if configured
                if self.config.integrity_check_on_read:
                    computed = record.compute_hash()
                    if computed != record.record_hash:
                        self._metrics.integrity_failures += 1
                        logger.error(f"Integrity check failed for record {record_id}")
                        return None

                return record
            return None

        except Exception as e:
            logger.error(f"Error reading record {record_id}: {e}")
            self._metrics.read_errors += 1
            return None
        finally:
            conn.close()

    def read_range(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[AuditEventType]] = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> List[AuditRecord]:
        """Read records in time range."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            start_ns = int(start_time.timestamp() * 1e9)
            end_ns = int(end_time.timestamp() * 1e9)

            if event_types:
                event_type_values = [et.value for et in event_types]
                placeholders = ",".join("?" * len(event_type_values))
                query = f"""
                    SELECT * FROM {self._table_name}
                    WHERE event_timestamp_ns BETWEEN ? AND ?
                    AND event_type IN ({placeholders})
                    ORDER BY event_timestamp_ns ASC
                    LIMIT ? OFFSET ?
                """
                params = [start_ns, end_ns] + event_type_values + [limit, offset]
            else:
                query = f"""
                    SELECT * FROM {self._table_name}
                    WHERE event_timestamp_ns BETWEEN ? AND ?
                    ORDER BY event_timestamp_ns ASC
                    LIMIT ? OFFSET ?
                """
                params = [start_ns, end_ns, limit, offset]

            cursor.execute(query, params)
            rows = cursor.fetchall()

            records = [self._row_to_record(row) for row in rows]

            self._metrics.records_read += len(records)
            self._metrics.last_read_timestamp = time.time_ns()

            return records

        except Exception as e:
            logger.error(f"Error reading range: {e}")
            self._metrics.read_errors += 1
            return []
        finally:
            conn.close()

    def read_by_order_id(self, order_id: str) -> List[AuditRecord]:
        """Read records for an order."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT * FROM {self._table_name}
                WHERE order_id = ?
                ORDER BY event_timestamp_ns ASC
                """,
                (order_id,),
            )
            rows = cursor.fetchall()
            records = [self._row_to_record(row) for row in rows]

            self._metrics.records_read += len(records)
            self._metrics.last_read_timestamp = time.time_ns()

            return records

        except Exception as e:
            logger.error(f"Error reading by order_id {order_id}: {e}")
            self._metrics.read_errors += 1
            return []
        finally:
            conn.close()

    def read_by_algorithm_id(
        self,
        algorithm_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AuditRecord]:
        """Read records for an algorithm."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            if start_time and end_time:
                start_ns = int(start_time.timestamp() * 1e9)
                end_ns = int(end_time.timestamp() * 1e9)
                cursor.execute(
                    f"""
                    SELECT * FROM {self._table_name}
                    WHERE algorithm_id = ?
                    AND event_timestamp_ns BETWEEN ? AND ?
                    ORDER BY event_timestamp_ns ASC
                    """,
                    (algorithm_id, start_ns, end_ns),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT * FROM {self._table_name}
                    WHERE algorithm_id = ?
                    ORDER BY event_timestamp_ns ASC
                    """,
                    (algorithm_id,),
                )

            rows = cursor.fetchall()
            records = [self._row_to_record(row) for row in rows]

            self._metrics.records_read += len(records)
            self._metrics.last_read_timestamp = time.time_ns()

            return records

        except Exception as e:
            logger.error(f"Error reading by algorithm_id {algorithm_id}: {e}")
            self._metrics.read_errors += 1
            return []
        finally:
            conn.close()

    def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> int:
        """Count matching records."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            conditions = []
            params = []

            if start_time:
                conditions.append("event_timestamp_ns >= ?")
                params.append(int(start_time.timestamp() * 1e9))
            if end_time:
                conditions.append("event_timestamp_ns <= ?")
                params.append(int(end_time.timestamp() * 1e9))
            if event_types:
                placeholders = ",".join("?" * len(event_types))
                conditions.append(f"event_type IN ({placeholders})")
                params.extend([et.value for et in event_types])

            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            else:
                where_clause = ""

            query = f"SELECT COUNT(*) as cnt FROM {self._table_name} {where_clause}"
            cursor.execute(query, params)
            row = cursor.fetchone()

            return row["cnt"] if row else 0

        except Exception as e:
            logger.error(f"Error counting records: {e}")
            return 0
        finally:
            conn.close()

    def get_latest_record(self) -> Optional[AuditRecord]:
        """Get latest record."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM {self._table_name} ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None
        except Exception as e:
            logger.error(f"Error getting latest record: {e}")
            return None
        finally:
            conn.close()

    def get_last_hash(self) -> Optional[str]:
        """Get hash of latest record."""
        with self._lock:
            return self._last_hash

    def verify_chain(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 1000,
    ) -> AuditChainStatus:
        """Verify chain integrity."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            conditions = []
            params = []

            if start_time:
                conditions.append("event_timestamp_ns >= ?")
                params.append(int(start_time.timestamp() * 1e9))
            if end_time:
                conditions.append("event_timestamp_ns <= ?")
                params.append(int(end_time.timestamp() * 1e9))

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            query = f"""
                SELECT * FROM {self._table_name}
                {where_clause}
                ORDER BY id ASC
            """
            cursor.execute(query, params)

            records_checked = 0
            previous_hash = None

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                for i, row in enumerate(rows):
                    record = self._row_to_record(row)
                    records_checked += 1
                    self._metrics.integrity_checks += 1

                    # Verify hash
                    computed = record.compute_hash()
                    if computed != record.record_hash:
                        self._metrics.integrity_failures += 1
                        return AuditChainStatus(
                            is_valid=False,
                            records_checked=records_checked,
                            first_invalid_record_id=record.record_id,
                            first_invalid_index=records_checked - 1,
                            error_message=f"Hash mismatch at record {records_checked - 1}",
                        )

                    # Verify chain (skip for first record)
                    if previous_hash is not None:
                        if record.previous_record_hash != previous_hash:
                            self._metrics.integrity_failures += 1
                            return AuditChainStatus(
                                is_valid=False,
                                records_checked=records_checked,
                                first_invalid_record_id=record.record_id,
                                first_invalid_index=records_checked - 1,
                                error_message=f"Chain break at record {records_checked - 1}",
                            )

                    previous_hash = record.record_hash

            return AuditChainStatus(
                is_valid=True,
                records_checked=records_checked,
            )

        except Exception as e:
            logger.error(f"Error verifying chain: {e}")
            return AuditChainStatus(
                is_valid=False,
                records_checked=0,
                error_message=str(e),
            )
        finally:
            conn.close()

    def export(self, request: AuditExportRequest) -> AuditExportResult:
        """Export records for NCA request."""
        try:
            # Build query
            conditions = []
            params = []

            if request.start_datetime:
                conditions.append("event_timestamp_ns >= ?")
                params.append(int(request.start_datetime.timestamp() * 1e9))
            if request.end_datetime:
                conditions.append("event_timestamp_ns <= ?")
                params.append(int(request.end_datetime.timestamp() * 1e9))
            if request.event_types:
                placeholders = ",".join("?" * len(request.event_types))
                conditions.append(f"event_type IN ({placeholders})")
                params.extend([et.value for et in request.event_types])
            if request.order_ids:
                placeholders = ",".join("?" * len(request.order_ids))
                conditions.append(f"order_id IN ({placeholders})")
                params.extend(request.order_ids)
            if request.algorithm_ids:
                placeholders = ",".join("?" * len(request.algorithm_ids))
                conditions.append(f"algorithm_id IN ({placeholders})")
                params.extend(request.algorithm_ids)
            if request.instrument_isins:
                placeholders = ",".join("?" * len(request.instrument_isins))
                conditions.append(f"instrument_isin IN ({placeholders})")
                params.extend(request.instrument_isins)

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                query = f"""
                    SELECT * FROM {self._table_name}
                    {where_clause}
                    ORDER BY event_timestamp_ns ASC
                """
                cursor.execute(query, params)

                # Count and optionally write to file
                records_exported = 0
                export_path = None

                if request.export_format == "json":
                    export_dir = os.path.dirname(self.config.backup_path)
                    if export_dir:
                        os.makedirs(export_dir, exist_ok=True)
                    export_path = os.path.join(
                        self.config.backup_path,
                        f"export_{request.request_id}_{int(time.time())}.jsonl",
                    )

                    os.makedirs(os.path.dirname(export_path), exist_ok=True)
                    with open(export_path, "w") as f:
                        while True:
                            rows = cursor.fetchmany(1000)
                            if not rows:
                                break
                            for row in rows:
                                record = self._row_to_record(row)
                                f.write(record.to_json() + "\n")
                                records_exported += 1
                else:
                    # Just count
                    while True:
                        rows = cursor.fetchmany(1000)
                        if not rows:
                            break
                        records_exported += len(rows)

                # Verify chain if requested
                chain_status = None
                if request.include_chain_verification:
                    chain_status = self.verify_chain(
                        request.start_datetime, request.end_datetime
                    )

                return AuditExportResult(
                    request_id=request.request_id,
                    success=True,
                    records_exported=records_exported,
                    export_path=export_path,
                    chain_verification=chain_status,
                )

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Error exporting records: {e}")
            return AuditExportResult(
                request_id=request.request_id,
                success=False,
                records_exported=0,
                error_message=str(e),
            )

    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        # Update storage size
        if os.path.exists(self._db_path):
            self._metrics.total_storage_bytes = os.path.getsize(self._db_path)
        return self._metrics

    def close(self) -> None:
        """Close storage."""
        self._state = StorageState.CLOSED


class FileAuditStorage(AuditStorageBackend):
    """
    File-based audit storage using JSON Lines format.

    Simple, portable format suitable for:
        - Development
        - Small deployments
        - Backup/archive storage

    Each line in the file is a complete JSON record.
    """

    def __init__(self, config: Optional[AuditStorageConfig] = None):
        """Initialize file storage."""
        self.config = config or AuditStorageConfig(backend_type=StorageBackendType.FILE)
        self._file_path = self.config.database_path
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._metrics = StorageMetrics()
        self._state = StorageState.UNINITIALIZED
        self._last_hash: Optional[str] = None
        self._records_cache: Dict[str, AuditRecord] = {}

        # Initialize storage
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize file storage."""
        # Create directory if needed
        file_dir = os.path.dirname(self._file_path)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)

        # Create file if not exists
        if not os.path.exists(self._file_path):
            with open(self._file_path, "w") as f:
                pass  # Create empty file

        # Read last hash from existing records
        try:
            with open(self._file_path, "r") as f:
                last_line = None
                for line in f:
                    if line.strip():
                        last_line = line
                if last_line:
                    record = AuditRecord.from_json(last_line)
                    self._last_hash = record.record_hash
        except Exception as e:
            logger.warning(f"Could not read last hash: {e}")

        self._state = StorageState.READY
        logger.info(f"File audit storage initialized at {self._file_path}")

    def append(self, record: AuditRecord) -> bool:
        """Append a single record."""
        with self._lock:
            try:
                # Set chain hash
                record.previous_record_hash = self._last_hash

                # Update status BEFORE computing hash (status is included in hash)
                record.status = AuditRecordStatus.WRITTEN

                # Always recompute hash to include chain link and status
                record.record_hash = record.compute_hash()

                # Write to file
                with open(self._file_path, "a") as f:
                    f.write(record.to_json() + "\n")
                    if self.config.sync_mode in ("normal", "full"):
                        f.flush()
                        if self.config.sync_mode == "full":
                            os.fsync(f.fileno())

                # Cache record
                self._records_cache[record.record_id] = record

                # Update chain hash
                self._last_hash = record.record_hash

                # Update metrics
                self._metrics.records_written += 1
                self._metrics.last_write_timestamp = time.time_ns()

                return True

            except Exception as e:
                logger.error(f"Error appending record: {e}")
                self._metrics.write_errors += 1
                return False

    def append_batch(self, records: List[AuditRecord]) -> int:
        """Append multiple records."""
        count = 0
        with self._lock:
            try:
                with open(self._file_path, "a") as f:
                    for record in records:
                        # Set chain hash
                        record.previous_record_hash = self._last_hash

                        # Compute hash if not set
                        if not record.record_hash:
                            record.record_hash = record.compute_hash()

                        # Update status
                        record.status = AuditRecordStatus.WRITTEN

                        # Write
                        f.write(record.to_json() + "\n")

                        # Cache
                        self._records_cache[record.record_id] = record

                        # Update chain
                        self._last_hash = record.record_hash
                        count += 1

                    if self.config.sync_mode in ("normal", "full"):
                        f.flush()
                        if self.config.sync_mode == "full":
                            os.fsync(f.fileno())

                self._metrics.records_written += count
                self._metrics.last_write_timestamp = time.time_ns()

            except Exception as e:
                logger.error(f"Error in batch append: {e}")
                self._metrics.write_errors += len(records) - count

        return count

    def _iter_records(self) -> Iterator[AuditRecord]:
        """Iterate over all records in file."""
        with open(self._file_path, "r") as f:
            for line in f:
                if line.strip():
                    yield AuditRecord.from_json(line)

    def read_by_id(self, record_id: str) -> Optional[AuditRecord]:
        """Read record by ID."""
        # Check cache first
        if record_id in self._records_cache:
            self._metrics.records_read += 1
            self._metrics.last_read_timestamp = time.time_ns()
            return self._records_cache[record_id]

        # Search file
        for record in self._iter_records():
            if record.record_id == record_id:
                self._metrics.records_read += 1
                self._metrics.last_read_timestamp = time.time_ns()
                return record

        return None

    def read_range(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[AuditEventType]] = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> List[AuditRecord]:
        """Read records in time range."""
        start_ns = int(start_time.timestamp() * 1e9)
        end_ns = int(end_time.timestamp() * 1e9)

        results = []
        skipped = 0

        for record in self._iter_records():
            if start_ns <= record.event_timestamp_ns <= end_ns:
                if event_types is None or record.event_type in event_types:
                    if skipped >= offset:
                        results.append(record)
                        if len(results) >= limit:
                            break
                    else:
                        skipped += 1

        self._metrics.records_read += len(results)
        self._metrics.last_read_timestamp = time.time_ns()

        return results

    def read_by_order_id(self, order_id: str) -> List[AuditRecord]:
        """Read records for an order."""
        results = []
        for record in self._iter_records():
            if record.order_id == order_id:
                results.append(record)

        self._metrics.records_read += len(results)
        self._metrics.last_read_timestamp = time.time_ns()
        return results

    def read_by_algorithm_id(
        self,
        algorithm_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AuditRecord]:
        """Read records for an algorithm."""
        start_ns = int(start_time.timestamp() * 1e9) if start_time else 0
        end_ns = int(end_time.timestamp() * 1e9) if end_time else time.time_ns()

        results = []
        for record in self._iter_records():
            if record.algorithm_id == algorithm_id:
                if start_ns <= record.event_timestamp_ns <= end_ns:
                    results.append(record)

        self._metrics.records_read += len(results)
        self._metrics.last_read_timestamp = time.time_ns()
        return results

    def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> int:
        """Count matching records."""
        start_ns = int(start_time.timestamp() * 1e9) if start_time else 0
        end_ns = int(end_time.timestamp() * 1e9) if end_time else time.time_ns()

        count = 0
        for record in self._iter_records():
            if start_ns <= record.event_timestamp_ns <= end_ns:
                if event_types is None or record.event_type in event_types:
                    count += 1

        return count

    def get_latest_record(self) -> Optional[AuditRecord]:
        """Get latest record."""
        latest = None
        for record in self._iter_records():
            latest = record
        return latest

    def get_last_hash(self) -> Optional[str]:
        """Get hash of latest record."""
        with self._lock:
            return self._last_hash

    def verify_chain(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 1000,
    ) -> AuditChainStatus:
        """Verify chain integrity."""
        start_ns = int(start_time.timestamp() * 1e9) if start_time else 0
        end_ns = int(end_time.timestamp() * 1e9) if end_time else time.time_ns()

        records_checked = 0
        previous_hash = None

        for record in self._iter_records():
            # Apply time filters
            if record.event_timestamp_ns < start_ns:
                continue
            if record.event_timestamp_ns > end_ns:
                break

            records_checked += 1
            self._metrics.integrity_checks += 1

            # Verify hash
            computed = record.compute_hash()
            if computed != record.record_hash:
                self._metrics.integrity_failures += 1
                return AuditChainStatus(
                    is_valid=False,
                    records_checked=records_checked,
                    first_invalid_record_id=record.record_id,
                    first_invalid_index=records_checked - 1,
                    error_message=f"Hash mismatch at record {records_checked - 1}",
                )

            # Verify chain
            if previous_hash is not None:
                if record.previous_record_hash != previous_hash:
                    self._metrics.integrity_failures += 1
                    return AuditChainStatus(
                        is_valid=False,
                        records_checked=records_checked,
                        first_invalid_record_id=record.record_id,
                        first_invalid_index=records_checked - 1,
                        error_message=f"Chain break at record {records_checked - 1}",
                    )

            previous_hash = record.record_hash

        return AuditChainStatus(
            is_valid=True,
            records_checked=records_checked,
        )

    def export(self, request: AuditExportRequest) -> AuditExportResult:
        """Export records."""
        try:
            records = self.read_range(
                request.start_datetime or datetime.min,
                request.end_datetime or datetime.now(),
                request.event_types,
            )

            # Apply additional filters
            if request.order_ids:
                records = [r for r in records if r.order_id in request.order_ids]
            if request.algorithm_ids:
                records = [r for r in records if r.algorithm_id in request.algorithm_ids]
            if request.instrument_isins:
                records = [r for r in records if r.instrument_isin in request.instrument_isins]

            # Verify chain if requested
            chain_status = None
            if request.include_chain_verification:
                chain_status = self.verify_chain(request.start_datetime, request.end_datetime)

            return AuditExportResult(
                request_id=request.request_id,
                success=True,
                records_exported=len(records),
                chain_verification=chain_status,
            )

        except Exception as e:
            return AuditExportResult(
                request_id=request.request_id,
                success=False,
                records_exported=0,
                error_message=str(e),
            )

    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        if os.path.exists(self._file_path):
            self._metrics.total_storage_bytes = os.path.getsize(self._file_path)
        return self._metrics

    def close(self) -> None:
        """Close storage."""
        self._state = StorageState.CLOSED


def create_audit_storage(
    config: Optional[AuditStorageConfig] = None,
    backend_type: Optional[StorageBackendType] = None,
) -> AuditStorageBackend:
    """
    Factory function to create audit storage backend.

    Currently supported backends:
        - StorageBackendType.MEMORY: In-memory (development/testing)
        - StorageBackendType.SQLITE: SQLite file-based (production single-node)
        - StorageBackendType.FILE: JSON file-based (simple deployments)

    Planned backends (raise NotImplementedError):
        - StorageBackendType.POSTGRESQL: Enterprise multi-node deployments.
          Status: Planned for future release. Requires psycopg2 or asyncpg.
          For enterprise PostgreSQL audit storage needs, contact support or
          use SQLite with external replication.

    Args:
        config: Storage configuration.
        backend_type: Override backend type from config.

    Returns:
        Configured AuditStorageBackend instance.

    Raises:
        NotImplementedError: If POSTGRESQL backend is requested (planned feature).
        ValueError: If unknown backend type is requested.
    """
    if config is None:
        config = AuditStorageConfig()

    actual_type = backend_type or config.backend_type

    if actual_type == StorageBackendType.MEMORY:
        return MemoryAuditStorage(config)
    elif actual_type == StorageBackendType.SQLITE:
        return SQLiteAuditStorage(config)
    elif actual_type == StorageBackendType.FILE:
        return FileAuditStorage(config)
    elif actual_type == StorageBackendType.POSTGRESQL:
        # PostgreSQL backend is planned for enterprise deployments.
        # Implementation requires psycopg2 or asyncpg dependencies.
        # For current enterprise needs, use SQLite with external backup/replication.
        raise NotImplementedError(
            "PostgreSQL audit storage is a planned feature (not yet implemented). "
            "Requires psycopg2 or asyncpg. For enterprise deployments, use "
            "StorageBackendType.SQLITE with external backup/replication, or "
            "contact support for roadmap timeline."
        )
    else:
        raise ValueError(f"Unknown storage backend type: {actual_type}")


__all__ = [
    # Enums
    "StorageBackendType",
    "StorageState",
    # Config
    "AuditStorageConfig",
    "StorageMetrics",
    # Base class
    "AuditStorageBackend",
    # Implementations
    "MemoryAuditStorage",
    "SQLiteAuditStorage",
    "FileAuditStorage",
    # Factory
    "create_audit_storage",
]
