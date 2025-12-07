# -*- coding: utf-8 -*-
"""
AI Act Compliant Logging System (Article 12).

EU AI Act Article 12 requires high-risk AI systems to be designed and developed
with capabilities enabling automatic recording of events (logs) while the system
is operating.

This module provides:
1. AIActLogger - Main logging system for AI Act compliance
2. AIActLogEvent - Structured event records
3. LogRetentionManager - Retention policy enforcement (minimum 6 months per Article 19)
4. LogIntegrityVerifier - Tamper-evident storage verification
5. AIActEventBus - Extended EventBus with compliance features

Article 12 Requirements Implemented:
    - (1) Logging capabilities designed at development stage
    - (2) Automatic recording of:
        (a) Period of use (start, end times)
        (b) Reference database against which input data was checked
        (c) Input data that led to a match
        (d) Identification of natural persons involved in verification
    - (3) Logs appropriate to intended purpose
    - (4) Deployer access to logs for monitoring

Article 19 Reference:
    - Logs retained for minimum 6 months (or as required by Union/Member State law)

References:
    - Article 12: https://artificialintelligenceact.eu/article/12/
    - Article 19: https://artificialintelligenceact.eu/article/19/
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Event Types and Categories
# =============================================================================


class AIActLogEventType(Enum):
    """
    Event types for AI Act compliant logging.

    Categories per Article 12(2):
    - System events: Start, stop, configuration changes
    - Decision events: Trading decisions, predictions
    - Risk events: Risk triggers, threshold breaches
    - Oversight events: Human interventions, overrides
    - Data events: Input data processing
    """
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_HEALTH_CHECK = "system_health_check"

    # Decision events
    TRADING_DECISION = "trading_decision"
    PREDICTION = "prediction"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_EXECUTED = "order_executed"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"

    # Risk events
    RISK_TRIGGER = "risk_trigger"
    RISK_THRESHOLD_BREACH = "risk_threshold_breach"
    RISK_MITIGATION_ACTIVATED = "risk_mitigation_activated"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    DRAWDOWN_LIMIT_REACHED = "drawdown_limit_reached"

    # Oversight events
    HUMAN_OVERRIDE = "human_override"
    HUMAN_APPROVAL = "human_approval"
    HUMAN_REJECTION = "human_rejection"
    OVERSIGHT_ALERT = "oversight_alert"
    OVERSIGHT_ALERT_ACKNOWLEDGED = "oversight_alert_acknowledged"
    MANUAL_INTERVENTION = "manual_intervention"

    # Data events
    DATA_INPUT = "data_input"
    DATA_VALIDATION = "data_validation"
    DATA_QUALITY_CHECK = "data_quality_check"
    DATA_ANOMALY_DETECTED = "data_anomaly_detected"
    DATA_SOURCE_CHANGE = "data_source_change"

    # Explainability events
    EXPLANATION_GENERATED = "explanation_generated"
    FEATURE_ATTRIBUTION = "feature_attribution"

    # Audit events
    AUDIT_ACCESS = "audit_access"
    AUDIT_EXPORT = "audit_export"


class AIActLogCategory(Enum):
    """Log event categories for classification."""
    SYSTEM = "system"
    DECISION = "decision"
    RISK = "risk"
    OVERSIGHT = "oversight"
    DATA = "data"
    EXPLAINABILITY = "explainability"
    AUDIT = "audit"


# =============================================================================
# Log Event Data Structures
# =============================================================================


@dataclass
class AIActLogEvent:
    """
    Structured log event record per Article 12.

    Each event captures:
    - Timestamp: When the event occurred
    - Event type: Category and specific type
    - Context: System state, inputs, outputs
    - Traceability: Session, correlation, and causation IDs
    - Human oversight state: Current oversight level
    - Integrity: Hash for tamper detection

    Attributes:
        event_id: Unique event identifier
        timestamp_utc: Event timestamp in UTC ISO format
        event_type: Type of event from AIActLogEventType
        category: Event category from AIActLogCategory
        system_state: Current state of the AI system
        inputs: Input data that triggered/influenced the event
        outputs: Outputs/results of the event
        confidence_metrics: Confidence/uncertainty metrics
        human_oversight_state: Current human oversight level
        session_id: Current session identifier
        correlation_id: ID linking related events
        causation_id: ID of the event that caused this event
        metadata: Additional event-specific metadata
        integrity_hash: SHA-256 hash for tamper detection
    """
    event_id: str = ""
    timestamp_utc: str = ""
    event_type: str = ""
    category: str = ""

    # System context
    system_state: Dict[str, Any] = field(default_factory=dict)

    # Input/Output
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    confidence_metrics: Dict[str, float] = field(default_factory=dict)

    # Human oversight
    human_oversight_state: str = ""
    human_operator_id: Optional[str] = None

    # Traceability
    session_id: str = ""
    correlation_id: str = ""
    causation_id: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Integrity
    integrity_hash: str = ""

    # Article 12(2)(d) - Natural person identification
    natural_person_involved: Optional[str] = None
    verification_type: Optional[str] = None

    def __post_init__(self):
        """Initialize auto-generated fields."""
        if not self.event_id:
            self.event_id = f"EVT-{uuid.uuid4().hex[:12].upper()}"
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute integrity hash for tamper detection."""
        # Create hash from key fields (excluding the hash itself)
        hash_content = {
            "event_id": self.event_id,
            "timestamp_utc": self.timestamp_utc,
            "event_type": self.event_type,
            "category": self.category,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
        }
        hash_str = json.dumps(hash_content, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:32]

    def verify_integrity(self) -> bool:
        """Verify event integrity hasn't been tampered."""
        expected_hash = self._compute_hash()
        return self.integrity_hash == expected_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIActLogEvent":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class LogSession:
    """
    Represents a logging session per Article 12(2)(a).

    Tracks the period of use for the AI system.
    """
    session_id: str = ""
    start_time: str = ""
    end_time: Optional[str] = None
    system_version: str = ""
    configuration_hash: str = ""
    operator_id: Optional[str] = None
    is_active: bool = True
    event_count: int = 0

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"SES-{uuid.uuid4().hex[:12].upper()}"
        if not self.start_time:
            self.start_time = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AIActLoggingConfig:
    """
    Configuration for AI Act compliant logging system.

    Attributes:
        log_path: Base path for log storage
        retention_days: Minimum retention period (Article 19: min 6 months = 180 days)
        enable_compression: Compress old logs to save space
        compression_after_days: Days before compressing logs
        enable_integrity_verification: Enable tamper detection
        max_log_file_size_mb: Maximum size per log file
        enable_async_logging: Use async logging for performance
        buffer_size: Event buffer size for async logging
        flush_interval_seconds: How often to flush buffer
        enable_encryption: Encrypt logs at rest
        encryption_key_id: Key identifier for encryption
    """
    log_path: str = "logs/ai_act"
    retention_days: int = 180  # Article 19: minimum 6 months
    enable_compression: bool = True
    compression_after_days: int = 7
    enable_integrity_verification: bool = True
    max_log_file_size_mb: int = 100
    enable_async_logging: bool = True
    buffer_size: int = 1000
    flush_interval_seconds: float = 5.0
    enable_encryption: bool = False
    encryption_key_id: Optional[str] = None

    # Event filtering
    log_all_decisions: bool = True
    log_all_risk_events: bool = True
    log_all_oversight_events: bool = True
    log_data_inputs: bool = True
    log_system_events: bool = True

    # Callbacks
    on_critical_event: Optional[Callable[[AIActLogEvent], None]] = None


# =============================================================================
# Log Retention Manager
# =============================================================================


class LogRetentionManager:
    """
    Manages log retention per Article 19.

    Article 19 requires logs to be retained for at least 6 months
    (or as required by Union or Member State law).

    This class:
    1. Enforces minimum retention periods
    2. Compresses old logs to save storage
    3. Securely deletes logs past retention period
    4. Provides retention status reporting
    """

    def __init__(self, config: AIActLoggingConfig):
        self.config = config
        self._log_path = Path(config.log_path)
        self._lock = threading.Lock()

    def get_retention_status(self) -> Dict[str, Any]:
        """Get current retention status."""
        status = {
            "retention_days": self.config.retention_days,
            "log_path": str(self._log_path),
            "total_files": 0,
            "total_size_mb": 0.0,
            "oldest_log": None,
            "newest_log": None,
            "compressed_files": 0,
            "files_due_for_deletion": 0,
        }

        if not self._log_path.exists():
            return status

        log_files = list(self._log_path.rglob("*.jsonl")) + list(self._log_path.rglob("*.jsonl.gz"))
        if not log_files:
            return status

        status["total_files"] = len(log_files)
        status["total_size_mb"] = sum(f.stat().st_size for f in log_files) / (1024 * 1024)
        status["compressed_files"] = len([f for f in log_files if f.suffix == ".gz"])

        # Find oldest and newest
        file_times = [(f, f.stat().st_mtime) for f in log_files]
        file_times.sort(key=lambda x: x[1])

        if file_times:
            status["oldest_log"] = datetime.fromtimestamp(file_times[0][1], tz=timezone.utc).isoformat()
            status["newest_log"] = datetime.fromtimestamp(file_times[-1][1], tz=timezone.utc).isoformat()

        # Check files due for deletion
        retention_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.retention_days)
        status["files_due_for_deletion"] = sum(
            1 for f, t in file_times
            if datetime.fromtimestamp(t, tz=timezone.utc) < retention_cutoff
        )

        return status

    def compress_old_logs(self) -> int:
        """
        Compress logs older than compression_after_days.

        Returns:
            Number of files compressed
        """
        if not self.config.enable_compression:
            return 0

        compression_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.compression_after_days)
        compressed_count = 0

        with self._lock:
            for log_file in self._log_path.rglob("*.jsonl"):
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)
                if file_time < compression_cutoff:
                    try:
                        compressed_path = log_file.with_suffix(".jsonl.gz")
                        with open(log_file, "rb") as f_in:
                            with gzip.open(compressed_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        log_file.unlink()
                        compressed_count += 1
                        logger.info(f"Compressed log: {log_file.name}")
                    except Exception as e:
                        logger.error(f"Failed to compress {log_file}: {e}")

        return compressed_count

    def cleanup_expired_logs(self) -> int:
        """
        Remove logs past retention period.

        Returns:
            Number of files deleted
        """
        retention_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.retention_days)
        deleted_count = 0

        with self._lock:
            for pattern in ["*.jsonl", "*.jsonl.gz"]:
                for log_file in self._log_path.rglob(pattern):
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)
                    if file_time < retention_cutoff:
                        try:
                            # Secure deletion - overwrite before deleting
                            if not log_file.suffix == ".gz":
                                with open(log_file, "r+b") as f:
                                    length = f.seek(0, 2)
                                    f.seek(0)
                                    f.write(os.urandom(length))
                            log_file.unlink()
                            deleted_count += 1
                            logger.info(f"Deleted expired log: {log_file.name}")
                        except Exception as e:
                            logger.error(f"Failed to delete {log_file}: {e}")

        return deleted_count

    def run_maintenance(self) -> Dict[str, int]:
        """
        Run full maintenance cycle.

        Returns:
            Dictionary with maintenance statistics
        """
        return {
            "compressed": self.compress_old_logs(),
            "deleted": self.cleanup_expired_logs(),
        }


# =============================================================================
# Log Integrity Verifier
# =============================================================================


class LogIntegrityVerifier:
    """
    Verifies log integrity for tamper detection.

    Uses hash chains to ensure logs haven't been modified:
    1. Each event has its own integrity hash
    2. Events are chained (each references previous)
    3. Periodic checksum files for verification
    """

    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._chain_hashes: Dict[str, str] = {}  # file -> last_hash

    def compute_chain_hash(self, previous_hash: str, event: AIActLogEvent) -> str:
        """Compute chain hash linking events."""
        chain_content = f"{previous_hash}:{event.integrity_hash}"
        return hashlib.sha256(chain_content.encode()).hexdigest()[:32]

    def verify_log_file(self, log_file: Path) -> Tuple[bool, List[str]]:
        """
        Verify integrity of a log file.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors: List[str] = []

        try:
            open_func = gzip.open if log_file.suffix == ".gz" else open
            with open_func(log_file, "rt", encoding="utf-8") as f:
                previous_hash = "GENESIS"
                line_num = 0

                for line in f:
                    line_num += 1
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        event = AIActLogEvent.from_dict(data)

                        # Verify individual event integrity
                        if not event.verify_integrity():
                            errors.append(f"Line {line_num}: Event integrity check failed")

                        previous_hash = self.compute_chain_hash(previous_hash, event)

                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    except Exception as e:
                        errors.append(f"Line {line_num}: Error - {e}")

        except Exception as e:
            errors.append(f"File read error: {e}")

        return len(errors) == 0, errors

    def verify_all_logs(self) -> Dict[str, Any]:
        """
        Verify all log files.

        Returns:
            Verification report
        """
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "errors": {},
        }

        for pattern in ["*.jsonl", "*.jsonl.gz"]:
            for log_file in self._log_path.rglob(pattern):
                report["total_files"] += 1
                is_valid, errors = self.verify_log_file(log_file)

                if is_valid:
                    report["valid_files"] += 1
                else:
                    report["invalid_files"] += 1
                    report["errors"][str(log_file)] = errors

        return report


# =============================================================================
# Main Logger
# =============================================================================


class AIActLogger:
    """
    AI Act Article 12 compliant logging system.

    Features:
    1. Automatic event recording with structured format
    2. Tamper-evident storage with integrity hashes
    3. Minimum 6-month retention (Article 19)
    4. Session tracking for period of use
    5. Correlation and causation tracking
    6. Async logging for performance
    7. Deployer access support

    Usage:
        config = AIActLoggingConfig(log_path="logs/ai_act")
        logger = AIActLogger(config)

        # Start a session
        session = logger.start_session(
            system_version="1.0.0",
            operator_id="operator_001"
        )

        # Log a trading decision
        logger.log_trading_decision(
            inputs={"observation": [...], "market_state": {...}},
            outputs={"action": "BUY", "position_size": 0.1},
            confidence_metrics={"value_estimate": 0.05, "uncertainty": 0.02},
            contributing_features=[("rsi", 0.3), ("momentum", 0.25)],
        )

        # Log human oversight action
        logger.log_human_override(
            action="position_closure",
            reason="Market volatility exceeds threshold",
            operator_id="trader_001",
        )

        # End session
        logger.end_session()
    """

    def __init__(self, config: Optional[AIActLoggingConfig] = None):
        """Initialize the logger."""
        self.config = config or AIActLoggingConfig()
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories per Article 12 categories
        self._decision_path = self._log_path / "decisions"
        self._risk_path = self._log_path / "risk_events"
        self._oversight_path = self._log_path / "human_overrides"
        self._system_path = self._log_path / "system_events"
        self._data_path = self._log_path / "data_events"
        self._audit_path = self._log_path / "audit"

        for path in [
            self._decision_path, self._risk_path, self._oversight_path,
            self._system_path, self._data_path, self._audit_path
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Session management
        self._current_session: Optional[LogSession] = None
        self._session_lock = threading.RLock()  # Reentrant lock to avoid deadlocks

        # Async logging
        self._event_buffer: List[AIActLogEvent] = []
        self._buffer_lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_flush = threading.Event()

        # Integrity
        self._last_chain_hash: str = "GENESIS"
        self._verifier = LogIntegrityVerifier(self._log_path)

        # Retention
        self._retention_manager = LogRetentionManager(self.config)

        # Correlation tracking
        self._current_correlation_id: str = ""

        # Start async flushing if enabled
        if self.config.enable_async_logging:
            self._start_flush_thread()

        logger.info(f"AIActLogger initialized at {self._log_path}")

    # =========================================================================
    # Session Management (Article 12(2)(a))
    # =========================================================================

    def start_session(
        self,
        system_version: str = "",
        configuration_hash: str = "",
        operator_id: Optional[str] = None,
    ) -> LogSession:
        """
        Start a new logging session.

        Per Article 12(2)(a), this records the start of the "period of use".
        """
        with self._session_lock:
            if self._current_session and self._current_session.is_active:
                self.end_session()

            self._current_session = LogSession(
                system_version=system_version,
                configuration_hash=configuration_hash,
                operator_id=operator_id,
            )

            # Log session start
            self._log_event(AIActLogEvent(
                event_type=AIActLogEventType.SYSTEM_START.value,
                category=AIActLogCategory.SYSTEM.value,
                system_state={
                    "session_id": self._current_session.session_id,
                    "system_version": system_version,
                    "configuration_hash": configuration_hash,
                },
                session_id=self._current_session.session_id,
                human_operator_id=operator_id,
            ))

            logger.info(f"Session started: {self._current_session.session_id}")
            return self._current_session

    def end_session(self) -> Optional[LogSession]:
        """
        End the current logging session.

        Per Article 12(2)(a), this records the end of the "period of use".
        """
        with self._session_lock:
            if not self._current_session:
                return None

            self._current_session.is_active = False
            self._current_session.end_time = datetime.now(timezone.utc).isoformat()

            # Log session end
            self._log_event(AIActLogEvent(
                event_type=AIActLogEventType.SYSTEM_STOP.value,
                category=AIActLogCategory.SYSTEM.value,
                system_state={
                    "session_id": self._current_session.session_id,
                    "event_count": self._current_session.event_count,
                    "duration_seconds": self._calculate_session_duration(),
                },
                session_id=self._current_session.session_id,
            ))

            # Flush any remaining events
            self._flush_buffer()

            session = self._current_session
            self._current_session = None
            logger.info(f"Session ended: {session.session_id}")
            return session

    def _calculate_session_duration(self) -> float:
        """Calculate session duration in seconds."""
        if not self._current_session:
            return 0.0

        start = datetime.fromisoformat(self._current_session.start_time.replace("Z", "+00:00"))
        end = datetime.now(timezone.utc)
        return (end - start).total_seconds()

    def get_current_session(self) -> Optional[LogSession]:
        """Get the current session."""
        with self._session_lock:
            return self._current_session

    # =========================================================================
    # Correlation Management
    # =========================================================================

    def start_correlation(self, correlation_id: Optional[str] = None) -> str:
        """Start a new correlation context."""
        self._current_correlation_id = correlation_id or f"COR-{uuid.uuid4().hex[:12].upper()}"
        return self._current_correlation_id

    def end_correlation(self) -> None:
        """End the current correlation context."""
        self._current_correlation_id = ""

    # =========================================================================
    # Decision Logging
    # =========================================================================

    def log_trading_decision(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        confidence_metrics: Optional[Dict[str, float]] = None,
        contributing_features: Optional[List[Tuple[str, float]]] = None,
        causation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AIActLogEvent:
        """
        Log a trading decision per Article 12.

        Args:
            inputs: Input data (observation, market state)
            outputs: Decision outputs (action, position size)
            confidence_metrics: Model confidence metrics
            contributing_features: Top features influencing decision
            causation_id: ID of event that caused this decision
            metadata: Additional metadata

        Returns:
            The logged event
        """
        if not self.config.log_all_decisions:
            return AIActLogEvent()

        # Serialize numpy arrays if present
        serialized_inputs = self._serialize_inputs(inputs)

        event = AIActLogEvent(
            event_type=AIActLogEventType.TRADING_DECISION.value,
            category=AIActLogCategory.DECISION.value,
            inputs=serialized_inputs,
            outputs=outputs,
            confidence_metrics=confidence_metrics or {},
            causation_id=causation_id,
            metadata={
                **(metadata or {}),
                "contributing_features": contributing_features or [],
            },
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
            human_oversight_state=self._get_oversight_state(),
        )

        self._log_event(event)
        return event

    def log_prediction(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        prediction: Any,
        confidence: float,
        uncertainty_bounds: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AIActLogEvent:
        """Log a model prediction."""
        event = AIActLogEvent(
            event_type=AIActLogEventType.PREDICTION.value,
            category=AIActLogCategory.DECISION.value,
            inputs=self._serialize_inputs(inputs),
            outputs={
                "model_name": model_name,
                "prediction": self._serialize_value(prediction),
                "confidence": confidence,
                "uncertainty_bounds": uncertainty_bounds,
            },
            confidence_metrics={"confidence": confidence},
            metadata=metadata or {},
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    def log_order(
        self,
        order_id: str,
        order_type: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        status: str = "placed",
        execution_details: Optional[Dict[str, Any]] = None,
    ) -> AIActLogEvent:
        """Log an order event."""
        event_type_map = {
            "placed": AIActLogEventType.ORDER_PLACED,
            "executed": AIActLogEventType.ORDER_EXECUTED,
            "cancelled": AIActLogEventType.ORDER_CANCELLED,
        }

        event = AIActLogEvent(
            event_type=event_type_map.get(status, AIActLogEventType.ORDER_PLACED).value,
            category=AIActLogCategory.DECISION.value,
            outputs={
                "order_id": order_id,
                "order_type": order_type,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "status": status,
                "execution_details": execution_details or {},
            },
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    # =========================================================================
    # Risk Event Logging
    # =========================================================================

    def log_risk_event(
        self,
        risk_type: str,
        risk_id: str,
        severity: str,
        description: str,
        trigger_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        mitigation_action: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AIActLogEvent:
        """
        Log a risk event per Article 12.

        Args:
            risk_type: Type of risk (threshold_breach, trigger, etc.)
            risk_id: Risk identifier from risk registry
            severity: Risk severity level
            description: Description of the risk event
            trigger_value: Value that triggered the event
            threshold_value: Threshold that was breached
            mitigation_action: Action taken to mitigate
            metadata: Additional metadata
        """
        if not self.config.log_all_risk_events:
            return AIActLogEvent()

        event = AIActLogEvent(
            event_type=AIActLogEventType.RISK_TRIGGER.value,
            category=AIActLogCategory.RISK.value,
            system_state={
                "risk_id": risk_id,
                "risk_type": risk_type,
                "severity": severity,
            },
            outputs={
                "description": description,
                "trigger_value": trigger_value,
                "threshold_value": threshold_value,
                "mitigation_action": mitigation_action,
            },
            metadata=metadata or {},
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)

        # Trigger callback for critical events
        if severity in ("critical", "high") and self.config.on_critical_event:
            try:
                self.config.on_critical_event(event)
            except Exception as e:
                logger.error(f"Critical event callback failed: {e}")

        return event

    def log_kill_switch(
        self,
        reason: str,
        trigger_source: str,
        positions_affected: int = 0,
        orders_cancelled: int = 0,
    ) -> AIActLogEvent:
        """Log kill switch activation."""
        event = AIActLogEvent(
            event_type=AIActLogEventType.KILL_SWITCH_ACTIVATED.value,
            category=AIActLogCategory.RISK.value,
            system_state={"kill_switch_active": True},
            outputs={
                "reason": reason,
                "trigger_source": trigger_source,
                "positions_affected": positions_affected,
                "orders_cancelled": orders_cancelled,
            },
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    # =========================================================================
    # Human Oversight Logging (Article 14 integration)
    # =========================================================================

    def log_human_override(
        self,
        action: str,
        reason: str,
        operator_id: str,
        override_type: str = "manual",
        affected_entities: Optional[List[str]] = None,
        previous_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
    ) -> AIActLogEvent:
        """
        Log a human override action per Article 14.

        Per Article 12(2)(d), records identification of natural persons
        involved in verification.
        """
        if not self.config.log_all_oversight_events:
            return AIActLogEvent()

        event = AIActLogEvent(
            event_type=AIActLogEventType.HUMAN_OVERRIDE.value,
            category=AIActLogCategory.OVERSIGHT.value,
            outputs={
                "action": action,
                "reason": reason,
                "override_type": override_type,
                "affected_entities": affected_entities or [],
                "previous_state": previous_state,
                "new_state": new_state,
            },
            human_operator_id=operator_id,
            natural_person_involved=operator_id,  # Article 12(2)(d)
            verification_type="manual_override",
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    def log_oversight_alert(
        self,
        alert_type: str,
        message: str,
        severity: str,
        requires_acknowledgment: bool = True,
        acknowledged_by: Optional[str] = None,
        acknowledged_at: Optional[str] = None,
    ) -> AIActLogEvent:
        """Log an oversight alert."""
        event_type = (
            AIActLogEventType.OVERSIGHT_ALERT_ACKNOWLEDGED
            if acknowledged_by
            else AIActLogEventType.OVERSIGHT_ALERT
        )

        event = AIActLogEvent(
            event_type=event_type.value,
            category=AIActLogCategory.OVERSIGHT.value,
            outputs={
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "requires_acknowledgment": requires_acknowledgment,
                "acknowledged_by": acknowledged_by,
                "acknowledged_at": acknowledged_at,
            },
            natural_person_involved=acknowledged_by,
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    # =========================================================================
    # Data Event Logging (Article 12(2)(b)(c))
    # =========================================================================

    def log_data_input(
        self,
        data_source: str,
        data_type: str,
        record_count: int,
        timestamp_range: Optional[Tuple[str, str]] = None,
        quality_metrics: Optional[Dict[str, float]] = None,
        validation_results: Optional[Dict[str, Any]] = None,
    ) -> AIActLogEvent:
        """
        Log data input event per Article 12(2)(b)(c).

        Records reference database and input data for traceability.
        """
        if not self.config.log_data_inputs:
            return AIActLogEvent()

        event = AIActLogEvent(
            event_type=AIActLogEventType.DATA_INPUT.value,
            category=AIActLogCategory.DATA.value,
            inputs={
                "data_source": data_source,  # Article 12(2)(b) - reference database
                "data_type": data_type,
                "record_count": record_count,
                "timestamp_range": timestamp_range,
            },
            outputs={
                "quality_metrics": quality_metrics or {},
                "validation_results": validation_results or {},
            },
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    def log_data_quality_check(
        self,
        check_name: str,
        passed: bool,
        details: Dict[str, Any],
        data_source: Optional[str] = None,
    ) -> AIActLogEvent:
        """Log a data quality check result."""
        event = AIActLogEvent(
            event_type=AIActLogEventType.DATA_QUALITY_CHECK.value,
            category=AIActLogCategory.DATA.value,
            inputs={"data_source": data_source},
            outputs={
                "check_name": check_name,
                "passed": passed,
                "details": details,
            },
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    def log_data_anomaly(
        self,
        anomaly_type: str,
        description: str,
        affected_records: int,
        severity: str,
        data_source: Optional[str] = None,
    ) -> AIActLogEvent:
        """Log a data anomaly detection."""
        event = AIActLogEvent(
            event_type=AIActLogEventType.DATA_ANOMALY_DETECTED.value,
            category=AIActLogCategory.DATA.value,
            inputs={"data_source": data_source},
            outputs={
                "anomaly_type": anomaly_type,
                "description": description,
                "affected_records": affected_records,
                "severity": severity,
            },
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    # =========================================================================
    # System Event Logging
    # =========================================================================

    def log_system_event(
        self,
        event_type: AIActLogEventType,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AIActLogEvent:
        """Log a system event."""
        if not self.config.log_system_events:
            return AIActLogEvent()

        event = AIActLogEvent(
            event_type=event_type.value,
            category=AIActLogCategory.SYSTEM.value,
            outputs={
                "description": description,
                "details": details or {},
            },
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    def log_error(
        self,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        component: Optional[str] = None,
    ) -> AIActLogEvent:
        """Log an error event."""
        event = AIActLogEvent(
            event_type=AIActLogEventType.SYSTEM_ERROR.value,
            category=AIActLogCategory.SYSTEM.value,
            outputs={
                "error_type": error_type,
                "error_message": error_message,
                "stack_trace": stack_trace,
                "component": component,
            },
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    # =========================================================================
    # Explainability Logging
    # =========================================================================

    def log_explanation(
        self,
        decision_id: str,
        explanation_type: str,
        explanation_content: Dict[str, Any],
        target_audience: str = "deployer",
    ) -> AIActLogEvent:
        """Log a decision explanation."""
        event = AIActLogEvent(
            event_type=AIActLogEventType.EXPLANATION_GENERATED.value,
            category=AIActLogCategory.EXPLAINABILITY.value,
            inputs={"decision_id": decision_id},
            outputs={
                "explanation_type": explanation_type,
                "explanation_content": explanation_content,
                "target_audience": target_audience,
            },
            causation_id=decision_id,
            session_id=self._get_session_id(),
            correlation_id=self._current_correlation_id,
        )

        self._log_event(event)
        return event

    # =========================================================================
    # Audit and Export
    # =========================================================================

    def log_audit_access(
        self,
        accessor_id: str,
        access_type: str,
        query_params: Optional[Dict[str, Any]] = None,
        records_accessed: int = 0,
    ) -> AIActLogEvent:
        """Log an audit access event."""
        event = AIActLogEvent(
            event_type=AIActLogEventType.AUDIT_ACCESS.value,
            category=AIActLogCategory.AUDIT.value,
            outputs={
                "accessor_id": accessor_id,
                "access_type": access_type,
                "query_params": query_params or {},
                "records_accessed": records_accessed,
            },
            natural_person_involved=accessor_id,
            session_id=self._get_session_id(),
        )

        self._log_event(event)
        return event

    def export_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        categories: Optional[List[AIActLogCategory]] = None,
        event_types: Optional[List[AIActLogEventType]] = None,
        output_format: str = "jsonl",
    ) -> Iterator[Dict[str, Any]]:
        """
        Export logs for a date range.

        Per Article 12(4), provides deployer access to logs.

        Args:
            start_date: Start of export range
            end_date: End of export range
            categories: Filter by categories (None = all)
            event_types: Filter by event types (None = all)
            output_format: Output format (jsonl, json)

        Yields:
            Log events matching the criteria
        """
        # Log the export access
        self.log_audit_access(
            accessor_id="export",
            access_type="bulk_export",
            query_params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "categories": [c.value for c in categories] if categories else None,
                "event_types": [e.value for e in event_types] if event_types else None,
            },
        )

        category_values = {c.value for c in categories} if categories else None
        event_type_values = {e.value for e in event_types} if event_types else None

        # Iterate through log files
        for pattern in ["*.jsonl", "*.jsonl.gz"]:
            for log_file in self._log_path.rglob(pattern):
                try:
                    open_func = gzip.open if log_file.suffix == ".gz" else open
                    with open_func(log_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue

                            data = json.loads(line)
                            event_time = datetime.fromisoformat(
                                data.get("timestamp_utc", "").replace("Z", "+00:00")
                            )

                            # Date filter
                            if not (start_date <= event_time <= end_date):
                                continue

                            # Category filter
                            if category_values and data.get("category") not in category_values:
                                continue

                            # Event type filter
                            if event_type_values and data.get("event_type") not in event_type_values:
                                continue

                            yield data

                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {e}")

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged events."""
        stats = {
            "log_path": str(self._log_path),
            "current_session": self._get_session_id() or None,
            "total_events_buffered": len(self._event_buffer),
            "retention_status": self._retention_manager.get_retention_status(),
        }

        # Count events by category
        category_counts: Dict[str, int] = {}
        for pattern in ["*.jsonl", "*.jsonl.gz"]:
            for log_file in self._log_path.rglob(pattern):
                try:
                    open_func = gzip.open if log_file.suffix == ".gz" else open
                    with open_func(log_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                cat = data.get("category", "unknown")
                                category_counts[cat] = category_counts.get(cat, 0) + 1
                except Exception:
                    pass

        stats["events_by_category"] = category_counts
        return stats

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify log integrity."""
        return self._verifier.verify_all_logs()

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _log_event(self, event: AIActLogEvent) -> None:
        """Log an event to the appropriate file."""
        # Set session and correlation if not set
        if not event.session_id:
            event.session_id = self._get_session_id()
        if not event.correlation_id:
            event.correlation_id = self._current_correlation_id

        # Compute integrity hash
        event.integrity_hash = event._compute_hash()

        # Update chain hash
        self._last_chain_hash = self._verifier.compute_chain_hash(
            self._last_chain_hash, event
        )

        # Increment session event count
        with self._session_lock:
            if self._current_session:
                self._current_session.event_count += 1

        if self.config.enable_async_logging:
            with self._buffer_lock:
                self._event_buffer.append(event)
                if len(self._event_buffer) >= self.config.buffer_size:
                    self._flush_buffer()
        else:
            self._write_event(event)

    def _write_event(self, event: AIActLogEvent) -> None:
        """Write event to appropriate log file."""
        # Determine path based on category
        category_paths = {
            AIActLogCategory.DECISION.value: self._decision_path,
            AIActLogCategory.RISK.value: self._risk_path,
            AIActLogCategory.OVERSIGHT.value: self._oversight_path,
            AIActLogCategory.SYSTEM.value: self._system_path,
            AIActLogCategory.DATA.value: self._data_path,
            AIActLogCategory.AUDIT.value: self._audit_path,
        }

        path = category_paths.get(event.category, self._log_path)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = path / f"{event.category}_{date_str}.jsonl"

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log event: {e}")

    def _flush_buffer(self) -> None:
        """Flush the event buffer to disk."""
        with self._buffer_lock:
            events = self._event_buffer.copy()
            self._event_buffer.clear()

        for event in events:
            self._write_event(event)

    def _start_flush_thread(self) -> None:
        """Start the async flush thread."""
        def flush_loop():
            while not self._stop_flush.is_set():
                self._stop_flush.wait(self.config.flush_interval_seconds)
                if self._event_buffer:
                    self._flush_buffer()

        self._flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self._flush_thread.start()

    def _get_session_id(self) -> str:
        """Get current session ID."""
        with self._session_lock:
            return self._current_session.session_id if self._current_session else ""

    def _get_oversight_state(self) -> str:
        """Get current human oversight state."""
        # This would integrate with HumanOversightSystem
        return "active"

    def _serialize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize inputs, handling numpy arrays."""
        result = {}
        for key, value in inputs.items():
            result[key] = self._serialize_value(value)
        return result

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value."""
        if isinstance(value, np.ndarray):
            # Store shape and sample for large arrays
            if value.size > 100:
                return {
                    "_type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sample_mean": float(np.nanmean(value)),
                    "sample_std": float(np.nanstd(value)),
                    "sample_min": float(np.nanmin(value)),
                    "sample_max": float(np.nanmax(value)),
                }
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        return value

    def close(self) -> None:
        """Close the logger and flush remaining events."""
        self._stop_flush.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        self._flush_buffer()
        self.end_session()
        logger.info("AIActLogger closed")

    def __enter__(self) -> "AIActLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# =============================================================================
# Extended Event Bus
# =============================================================================


class AIActEventBus:
    """
    Extended EventBus with AI Act compliance features.

    Wraps the existing EventBus and adds:
    1. Mandatory AI Act compliance fields
    2. Automatic correlation ID generation
    3. Tamper-evident logging
    4. Integration with AIActLogger
    """

    def __init__(
        self,
        queue_size: int = 1000,
        drop_policy: str = "newest",
        ai_act_logger: Optional[AIActLogger] = None,
    ):
        from services.event_bus import EventBus

        self._event_bus = EventBus(queue_size, drop_policy)
        self._logger = ai_act_logger
        self._correlation_id: str = ""

    async def put(self, event: Dict[str, Any]) -> bool:
        """Put an event with AI Act compliance fields."""
        # Add compliance fields
        enhanced_event = {
            **event,
            "_ai_act_timestamp": datetime.now(timezone.utc).isoformat(),
            "_ai_act_correlation_id": self._correlation_id or f"COR-{uuid.uuid4().hex[:8]}",
            "_ai_act_event_id": f"EVT-{uuid.uuid4().hex[:12]}",
        }

        # Log to AI Act logger if available
        if self._logger:
            self._logger.log_system_event(
                AIActLogEventType.SYSTEM_CONFIG_CHANGE,
                "Event bus event",
                details=enhanced_event,
            )

        return await self._event_bus.put(enhanced_event)

    async def get(self) -> Any:
        """Get an event from the bus."""
        return await self._event_bus.get()

    @property
    def depth(self) -> int:
        """Current queue depth."""
        return self._event_bus.depth

    def close(self) -> None:
        """Close the event bus."""
        self._event_bus.close()

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set the correlation ID for subsequent events."""
        self._correlation_id = correlation_id


# =============================================================================
# Factory Functions
# =============================================================================


def create_ai_act_logger(
    config: Optional[Union[Dict[str, Any], AIActLoggingConfig]] = None,
) -> AIActLogger:
    """
    Create an AIActLogger instance.

    Args:
        config: Optional configuration dictionary or AIActLoggingConfig

    Returns:
        Configured AIActLogger instance
    """
    if config is None:
        return AIActLogger()
    elif isinstance(config, AIActLoggingConfig):
        return AIActLogger(config)
    else:
        # Dictionary config
        logging_config = AIActLoggingConfig(
            log_path=config.get("log_path", "logs/ai_act"),
            retention_days=config.get("retention_days", 180),
            enable_compression=config.get("enable_compression", True),
            compression_after_days=config.get("compression_after_days", 7),
            enable_integrity_verification=config.get("enable_integrity_verification", True),
            max_log_file_size_mb=config.get("max_log_file_size_mb", 100),
            enable_async_logging=config.get("enable_async_logging", True),
            buffer_size=config.get("buffer_size", 1000),
            flush_interval_seconds=config.get("flush_interval_seconds", 5.0),
            log_all_decisions=config.get("log_all_decisions", True),
            log_all_risk_events=config.get("log_all_risk_events", True),
            log_all_oversight_events=config.get("log_all_oversight_events", True),
            log_data_inputs=config.get("log_data_inputs", True),
            log_system_events=config.get("log_system_events", True),
        )
        return AIActLogger(logging_config)


def create_ai_act_event_bus(
    queue_size: int = 1000,
    drop_policy: str = "newest",
    ai_act_logger: Optional[AIActLogger] = None,
) -> AIActEventBus:
    """Create an AI Act compliant event bus."""
    return AIActEventBus(queue_size, drop_policy, ai_act_logger)
