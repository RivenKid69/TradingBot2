# -*- coding: utf-8 -*-
"""
Telemetry Buffer - Durable telemetry storage.

Design Doc Phase 5:
- Local journal + telemetry buffer
- Durable очередь событий (sqlite/jsonl), восстановление после рестарта

CCEA Phase 5 Component.

Implementation Notes:
- SQLite connections are explicitly closed to avoid Windows file locking issues
- Background threads are cleanly shut down via stop_event
- Uses contextlib.closing for deterministic connection cleanup
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, Iterator, List, Optional
from uuid import uuid4


# Constants
MAX_BUFFER_SIZE: Final[int] = 100000
BATCH_SIZE: Final[int] = 100
FLUSH_INTERVAL_SECONDS: Final[int] = 30


class TelemetryEventType(Enum):
    """Types of telemetry events."""
    HEARTBEAT = auto()
    ORDER_SUBMITTED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()
    POSITION_UPDATE = auto()
    PNL_UPDATE = auto()
    ERROR = auto()
    WARNING = auto()
    KILL_SWITCH = auto()
    DEGRADED_MODE = auto()
    PREFLIGHT_RESULT = auto()
    CONFIG_CHANGE = auto()
    STATE_CHANGE = auto()
    METRIC = auto()


class TelemetryLevel(Enum):
    """Telemetry detail level."""
    AGGREGATED = "aggregated"  # Default for retail
    DETAILED_NON_SENSITIVE = "detailed"  # Opt-in
    RAW_ORDER_EVENTS = "raw"  # Enterprise-only, opt-in


@dataclass
class TelemetryEvent:
    """
    Single telemetry event.
    """
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: TelemetryEventType = TelemetryEventType.METRIC
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: TelemetryLevel = TelemetryLevel.AGGREGATED
    data: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    sent: bool = False
    sent_at: Optional[datetime] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "data": self.data,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "sent": self.sent,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TelemetryEvent:
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=TelemetryEventType[data["event_type"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=TelemetryLevel(data.get("level", "aggregated")),
            data=data.get("data", {}),
            run_id=data.get("run_id"),
            agent_id=data.get("agent_id"),
            sent=data.get("sent", False),
            sent_at=datetime.fromisoformat(data["sent_at"]) if data.get("sent_at") else None,
            retry_count=data.get("retry_count", 0),
        )

    def get_redacted_data(self) -> Dict[str, Any]:
        """
        Get data with sensitive values redacted.

        Mandatory redaction per Design Doc.
        """
        sensitive_patterns = [
            "key", "secret", "password", "token", "credential",
            "api_key", "api_secret", "access_token", "refresh_token",
            "bearer", "authorization", "auth",
        ]

        redacted = {}
        for key, value in self.data.items():
            key_lower = key.lower()
            if any(p in key_lower for p in sensitive_patterns):
                redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                # Recursively redact nested dicts
                redacted[key] = self._redact_dict(value, sensitive_patterns)
            else:
                redacted[key] = value
        return redacted

    def _redact_dict(self, d: Dict[str, Any], patterns: List[str]) -> Dict[str, Any]:
        """Recursively redact dictionary."""
        result = {}
        for key, value in d.items():
            key_lower = key.lower()
            if any(p in key_lower for p in patterns):
                result[key] = "[REDACTED]"
            elif isinstance(value, dict):
                result[key] = self._redact_dict(value, patterns)
            else:
                result[key] = value
        return result


@dataclass
class TelemetryBufferConfig:
    """
    Telemetry buffer configuration.
    """
    # Storage
    db_path: Path = field(default_factory=lambda: Path.home() / ".ccea" / "telemetry.db")
    max_buffer_size: int = MAX_BUFFER_SIZE
    max_age_days: int = 7

    # Sending
    batch_size: int = BATCH_SIZE
    flush_interval_seconds: int = FLUSH_INTERVAL_SECONDS
    max_retries: int = 3
    retry_delay_seconds: int = 60

    # Levels
    default_level: TelemetryLevel = TelemetryLevel.AGGREGATED
    allow_raw_events: bool = False  # Enterprise only

    # Redaction
    redaction_enabled: bool = True  # Cannot be disabled per Design Doc

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "db_path": str(self.db_path),
            "max_buffer_size": self.max_buffer_size,
            "max_age_days": self.max_age_days,
            "batch_size": self.batch_size,
            "flush_interval_seconds": self.flush_interval_seconds,
            "default_level": self.default_level.value,
            "allow_raw_events": self.allow_raw_events,
            "redaction_enabled": self.redaction_enabled,
        }


class TelemetryBuffer:
    """
    Durable telemetry buffer with SQLite persistence.

    Design Doc Phase 5:
    - Durable queue of events (sqlite/jsonl)
    - Recovery after restart
    - Mandatory redaction

    Usage:
        buffer = TelemetryBuffer(config)

        # Add events
        buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"status": "running"},
        ))

        # Flush to cloud (when connected)
        buffer.flush(send_fn=cloud_client.send_telemetry)

        # On restart
        pending = buffer.get_pending_count()
    """

    def __init__(
        self,
        config: Optional[TelemetryBufferConfig] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize telemetry buffer.

        Args:
            config: Configuration
            agent_id: Agent identifier
            run_id: Current run identifier
        """
        self.config = config or TelemetryBufferConfig()
        self._agent_id = agent_id
        self._run_id = run_id

        # Ensure redaction is always enabled
        self.config.redaction_enabled = True

        # Thread safety
        self._lock = threading.RLock()

        # In-memory buffer for batching
        self._memory_buffer: List[TelemetryEvent] = []

        # Initialize database
        self._init_database()

        # Background flusher
        self._flushing = False
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _init_database(self) -> None:
        """Initialize SQLite database."""
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use contextlib.closing to ensure connection is properly closed
        # sqlite3.connect context manager only handles commit/rollback, NOT close()
        with contextlib.closing(sqlite3.connect(str(self.config.db_path))) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    data TEXT NOT NULL,
                    run_id TEXT,
                    agent_id TEXT,
                    sent INTEGER DEFAULT 0,
                    sent_at TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_sent
                ON telemetry_events (sent, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_event_id
                ON telemetry_events (event_id)
            """)
            conn.commit()

    def set_run_id(self, run_id: str) -> None:
        """Set current run ID."""
        self._run_id = run_id

    def set_agent_id(self, agent_id: str) -> None:
        """Set agent ID."""
        self._agent_id = agent_id

    def add(self, event: TelemetryEvent) -> str:
        """
        Add event to buffer.

        Args:
            event: Telemetry event

        Returns:
            Event ID
        """
        # Set agent/run IDs
        if event.agent_id is None:
            event.agent_id = self._agent_id
        if event.run_id is None:
            event.run_id = self._run_id

        # Apply level restrictions
        if event.level == TelemetryLevel.RAW_ORDER_EVENTS and not self.config.allow_raw_events:
            event.level = TelemetryLevel.DETAILED_NON_SENSITIVE

        with self._lock:
            # Add to memory buffer
            self._memory_buffer.append(event)

            # Persist if batch size reached
            if len(self._memory_buffer) >= self.config.batch_size:
                self._persist_batch()

        return event.event_id

    def add_metric(
        self,
        name: str,
        value: Any,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Add a metric event.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags

        Returns:
            Event ID
        """
        event = TelemetryEvent(
            event_type=TelemetryEventType.METRIC,
            level=self.config.default_level,
            data={
                "metric_name": name,
                "value": value,
                "tags": tags or {},
            },
        )
        return self.add(event)

    def add_heartbeat(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add heartbeat event.

        Args:
            status: Status string
            details: Optional details

        Returns:
            Event ID
        """
        event = TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            level=TelemetryLevel.AGGREGATED,
            data={
                "status": status,
                **(details or {}),
            },
        )
        return self.add(event)

    def add_error(
        self,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add error event.

        Args:
            error_type: Type of error
            message: Error message
            details: Optional details

        Returns:
            Event ID
        """
        event = TelemetryEvent(
            event_type=TelemetryEventType.ERROR,
            level=TelemetryLevel.DETAILED_NON_SENSITIVE,
            data={
                "error_type": error_type,
                "message": message,
                **(details or {}),
            },
        )
        return self.add(event)

    def _persist_batch(self) -> int:
        """Persist memory buffer to database."""
        if not self._memory_buffer:
            return 0

        events = self._memory_buffer[:]
        self._memory_buffer.clear()

        with contextlib.closing(sqlite3.connect(str(self.config.db_path))) as conn:
            for event in events:
                # Apply redaction
                data = event.get_redacted_data() if self.config.redaction_enabled else event.data

                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO telemetry_events
                        (event_id, event_type, timestamp, level, data, run_id, agent_id, sent, retry_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)
                    """, (
                        event.event_id,
                        event.event_type.name,
                        event.timestamp.isoformat(),
                        event.level.value,
                        json.dumps(data),
                        event.run_id,
                        event.agent_id,
                    ))
                except sqlite3.IntegrityError:
                    pass  # Duplicate, skip

            conn.commit()

        return len(events)

    def flush(
        self,
        send_fn: Optional[Callable[[List[Dict[str, Any]]], bool]] = None,
    ) -> int:
        """
        Flush pending events to cloud.

        Args:
            send_fn: Function to send events (returns True on success)

        Returns:
            Number of events sent
        """
        # First persist memory buffer
        with self._lock:
            self._persist_batch()

        if send_fn is None:
            return 0

        total_sent = 0

        with contextlib.closing(sqlite3.connect(str(self.config.db_path))) as conn:
            # Get pending events
            cursor = conn.execute("""
                SELECT id, event_id, event_type, timestamp, level, data, run_id, agent_id, retry_count
                FROM telemetry_events
                WHERE sent = 0 AND retry_count < ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (self.config.max_retries, self.config.batch_size))

            rows = cursor.fetchall()
            if not rows:
                return 0

            # Prepare batch
            events_to_send = []
            event_ids = []

            for row in rows:
                event_dict = {
                    "event_id": row[1],
                    "event_type": row[2],
                    "timestamp": row[3],
                    "level": row[4],
                    "data": json.loads(row[5]),
                    "run_id": row[6],
                    "agent_id": row[7],
                }
                events_to_send.append(event_dict)
                event_ids.append(row[0])

            # Try to send
            try:
                success = send_fn(events_to_send)

                if success:
                    # Mark as sent
                    placeholders = ",".join("?" * len(event_ids))
                    conn.execute(f"""
                        UPDATE telemetry_events
                        SET sent = 1, sent_at = ?
                        WHERE id IN ({placeholders})
                    """, [datetime.utcnow().isoformat()] + event_ids)
                    conn.commit()
                    total_sent = len(event_ids)
                else:
                    # Increment retry count
                    placeholders = ",".join("?" * len(event_ids))
                    conn.execute(f"""
                        UPDATE telemetry_events
                        SET retry_count = retry_count + 1
                        WHERE id IN ({placeholders})
                    """, event_ids)
                    conn.commit()

            except Exception:
                # Increment retry count
                placeholders = ",".join("?" * len(event_ids))
                conn.execute(f"""
                    UPDATE telemetry_events
                    SET retry_count = retry_count + 1
                    WHERE id IN ({placeholders})
                """, event_ids)
                conn.commit()

        return total_sent

    def get_pending_count(self) -> int:
        """Get count of pending (unsent) events."""
        with self._lock:
            self._persist_batch()

        with contextlib.closing(sqlite3.connect(str(self.config.db_path))) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM telemetry_events WHERE sent = 0
            """)
            return cursor.fetchone()[0]

    def get_pending_events(self, limit: int = 100) -> List[TelemetryEvent]:
        """
        Get pending events.

        Args:
            limit: Maximum events to return

        Returns:
            List of pending events
        """
        with self._lock:
            self._persist_batch()

        events = []
        with contextlib.closing(sqlite3.connect(str(self.config.db_path))) as conn:
            cursor = conn.execute("""
                SELECT event_id, event_type, timestamp, level, data, run_id, agent_id, retry_count
                FROM telemetry_events
                WHERE sent = 0
                ORDER BY timestamp ASC
                LIMIT ?
            """, (limit,))

            for row in cursor:
                event = TelemetryEvent(
                    event_id=row[0],
                    event_type=TelemetryEventType[row[1]],
                    timestamp=datetime.fromisoformat(row[2]),
                    level=TelemetryLevel(row[3]),
                    data=json.loads(row[4]),
                    run_id=row[5],
                    agent_id=row[6],
                    retry_count=row[7],
                )
                events.append(event)

        return events

    def cleanup_old_events(self) -> int:
        """
        Remove old sent events.

        Returns:
            Number of events deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=self.config.max_age_days)

        with contextlib.closing(sqlite3.connect(str(self.config.db_path))) as conn:
            cursor = conn.execute("""
                DELETE FROM telemetry_events
                WHERE sent = 1 AND timestamp < ?
            """, (cutoff.isoformat(),))
            conn.commit()
            return cursor.rowcount

    def start_background_flush(
        self,
        send_fn: Callable[[List[Dict[str, Any]]], bool],
    ) -> None:
        """
        Start background flush thread.

        Args:
            send_fn: Function to send events
        """
        if self._flushing:
            return

        self._flushing = True
        self._stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            args=(send_fn,),
            daemon=True,
        )
        self._flush_thread.start()

    def stop_background_flush(self, timeout: float = 10.0) -> bool:
        """
        Stop background flush thread.

        Args:
            timeout: Maximum time to wait for thread to stop (seconds)

        Returns:
            True if thread stopped cleanly, False if timed out
        """
        self._flushing = False
        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=timeout)
            stopped = not self._flush_thread.is_alive()
            self._flush_thread = None
            return stopped
        return True

    def _flush_loop(self, send_fn: Callable[[List[Dict[str, Any]]], bool]) -> None:
        """Background flush loop."""
        while not self._stop_event.is_set():
            try:
                self.flush(send_fn)
                self.cleanup_old_events()
            except Exception:
                pass

            # Use shorter wait intervals to respond faster to stop_event
            # This prevents long delays when stopping the thread
            self._stop_event.wait(self.config.flush_interval_seconds)

    def close(self) -> None:
        """
        Close the telemetry buffer and release all resources.

        This method should be called when the buffer is no longer needed.
        It ensures:
        - Background flush thread is stopped
        - Pending events in memory are persisted to disk
        - No open database connections remain

        Usage:
            buffer = TelemetryBuffer(config)
            try:
                # use buffer...
            finally:
                buffer.close()

        Or use as context manager:
            with TelemetryBuffer(config) as buffer:
                # use buffer...
        """
        # Stop background flusher first
        self.stop_background_flush(timeout=10.0)

        # Persist any remaining in-memory events
        with self._lock:
            self._persist_batch()

    def __enter__(self) -> "TelemetryBuffer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics.

        Returns:
            Statistics dictionary
        """
        with contextlib.closing(sqlite3.connect(str(self.config.db_path))) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN sent = 0 THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN sent = 1 THEN 1 ELSE 0 END) as sent,
                    SUM(CASE WHEN retry_count >= ? THEN 1 ELSE 0 END) as failed
                FROM telemetry_events
            """, (self.config.max_retries,))
            row = cursor.fetchone()

            return {
                "total_events": row[0] or 0,
                "pending_events": row[1] or 0,
                "sent_events": row[2] or 0,
                "failed_events": row[3] or 0,
                "memory_buffer_size": len(self._memory_buffer),
                "db_path": str(self.config.db_path),
            }

    def export_jsonl(self, output_path: Path, sent_only: bool = False) -> int:
        """
        Export events to JSONL file.

        Args:
            output_path: Output file path
            sent_only: Only export sent events

        Returns:
            Number of events exported
        """
        with self._lock:
            self._persist_batch()

        count = 0
        with contextlib.closing(sqlite3.connect(str(self.config.db_path))) as conn:
            query = """
                SELECT event_id, event_type, timestamp, level, data, run_id, agent_id, sent, sent_at
                FROM telemetry_events
            """
            if sent_only:
                query += " WHERE sent = 1"
            query += " ORDER BY timestamp ASC"

            cursor = conn.execute(query)

            with open(output_path, "w") as f:
                for row in cursor:
                    event_dict = {
                        "event_id": row[0],
                        "event_type": row[1],
                        "timestamp": row[2],
                        "level": row[3],
                        "data": json.loads(row[4]),
                        "run_id": row[5],
                        "agent_id": row[6],
                        "sent": bool(row[7]),
                        "sent_at": row[8],
                    }
                    f.write(json.dumps(event_dict) + "\n")
                    count += 1

        return count
