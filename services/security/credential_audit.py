"""
Detailed audit logging for broker credential access.

Provides structured logging and anomaly detection for credential access,
supporting compliance requirements and security monitoring.

References:
    - ISO 27001 A.12.4: Logging and monitoring
    - SOC 2 CC6.1: Logical access security
    - PCI DSS 10.2: Audit trail requirements
    - GDPR Art. 30: Records of processing activities

Features:
    - Structured event logging
    - Anomaly detection (high volume, repeated failures)
    - User access history (for GDPR data subject requests)
    - Retention policy support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger("credential_audit")


class CredentialAccessType(Enum):
    """Types of credential access operations."""

    CREATE = "create"
    READ = "read"
    DECRYPT = "decrypt"
    UPDATE = "update"
    DELETE = "delete"
    ROTATE = "rotate"
    VERIFY = "verify"
    FAILED_ATTEMPT = "failed_attempt"
    EXPORT = "export"


@dataclass
class CredentialAccessEvent:
    """
    Represents a credential access event for audit logging.

    Attributes:
        timestamp: When the access occurred
        user_id: User who owns the credential
        credential_id: Identifier of the accessed credential
        broker: Broker associated with the credential
        access_type: Type of access operation
        purpose: Stated reason for access
        source_ip: IP address of the request
        user_agent: User agent string (if applicable)
        success: Whether the access succeeded
        error_message: Error details if access failed
        request_id: Correlation ID for request tracing
        metadata: Additional context
    """

    timestamp: datetime
    user_id: str
    credential_id: str
    broker: str
    access_type: CredentialAccessType
    purpose: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "credential_id": self.credential_id,
            "broker": self.broker,
            "access_type": self.access_type.value,
            "purpose": self.purpose,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "success": self.success,
            "error_message": self.error_message,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CredentialAccessEvent":
        """Deserialize event from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data["user_id"],
            credential_id=data["credential_id"],
            broker=data["broker"],
            access_type=CredentialAccessType(data["access_type"]),
            purpose=data["purpose"],
            source_ip=data.get("source_ip"),
            user_agent=data.get("user_agent"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            request_id=data.get("request_id"),
            metadata=data.get("metadata", {}),
        )

    def to_log_string(self) -> str:
        """Format event as a structured log string."""
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"CREDENTIAL_ACCESS | "
            f"status={status} | "
            f"user={self.user_id} | "
            f"credential={self.credential_id} | "
            f"broker={self.broker} | "
            f"type={self.access_type.value} | "
            f"purpose={self.purpose} | "
            f"ip={self.source_ip or 'unknown'}"
        )


class AuditStorageProtocol(Protocol):
    """Protocol for audit event storage backends."""

    def append(self, event: CredentialAccessEvent) -> None:
        """Store an audit event."""
        ...

    def query(
        self,
        user_id: Optional[str] = None,
        credential_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        access_types: Optional[List[CredentialAccessType]] = None,
        success_only: bool = False,
        limit: int = 1000
    ) -> List[CredentialAccessEvent]:
        """Query stored events."""
        ...

    def delete_before(self, cutoff: datetime) -> int:
        """Delete events before cutoff date (retention policy)."""
        ...


class InMemoryAuditStorage:
    """
    In-memory implementation of audit storage.

    Suitable for testing and development. For production,
    use a persistent storage backend.
    """

    def __init__(self, max_events: int = 100_000):
        """
        Initialize in-memory storage.

        Args:
            max_events: Maximum events to retain in memory
        """
        self._events: List[CredentialAccessEvent] = []
        self._max_events = max_events

    def append(self, event: CredentialAccessEvent) -> None:
        """Store an audit event."""
        self._events.append(event)

        # Enforce max events limit (FIFO eviction)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    def query(
        self,
        user_id: Optional[str] = None,
        credential_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        access_types: Optional[List[CredentialAccessType]] = None,
        success_only: bool = False,
        limit: int = 1000
    ) -> List[CredentialAccessEvent]:
        """Query stored events with filters."""
        result = self._events

        if user_id:
            result = [e for e in result if e.user_id == user_id]

        if credential_id:
            result = [e for e in result if e.credential_id == credential_id]

        if since:
            result = [e for e in result if e.timestamp >= since]

        if until:
            result = [e for e in result if e.timestamp <= until]

        if access_types:
            result = [e for e in result if e.access_type in access_types]

        if success_only:
            result = [e for e in result if e.success]

        return result[-limit:]

    def delete_before(self, cutoff: datetime) -> int:
        """Delete events before cutoff date."""
        original_count = len(self._events)
        self._events = [e for e in self._events if e.timestamp >= cutoff]
        return original_count - len(self._events)

    def clear(self) -> int:
        """Clear all events. Returns count of deleted events."""
        count = len(self._events)
        self._events = []
        return count


@dataclass
class AnomalyAlert:
    """Represents a detected anomaly in credential access patterns."""

    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    user_id: str
    description: str
    timestamp: datetime
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert to dictionary."""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "user_id": self.user_id,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "evidence": self.evidence,
        }


class CredentialAuditLogger:
    """
    Structured audit logging for credential access with anomaly detection.

    Features:
        - Structured event logging to configurable backends
        - Real-time anomaly detection
        - User access history queries (GDPR support)
        - Retention policy enforcement

    Anomaly Detection Rules:
        - High volume: >1000 accesses per user per day
        - Repeated failures: >10 failures per hour
        - Unusual hours: Accesses outside normal patterns
        - Multiple IPs: >5 distinct IPs per hour

    Example:
        >>> storage = InMemoryAuditStorage()
        >>> audit_logger = CredentialAuditLogger(storage)
        >>> audit_logger.log_access(
        ...     user_id="user_001",
        ...     credential_id="cred_123",
        ...     broker="alpaca",
        ...     access_type=CredentialAccessType.DECRYPT,
        ...     purpose="order_execution",
        ...     source_ip="127.0.0.1",
        ...     success=True
        ... )
    """

    # Anomaly detection thresholds
    HIGH_VOLUME_THRESHOLD = 1000  # accesses per day
    FAILURE_THRESHOLD = 10  # failures per hour
    MULTI_IP_THRESHOLD = 5  # distinct IPs per hour

    def __init__(
        self,
        storage: AuditStorageProtocol,
        alert_callback: Optional[Callable[[AnomalyAlert], None]] = None
    ):
        """
        Initialize the audit logger.

        Args:
            storage: Backend for storing audit events
            alert_callback: Optional callback for anomaly alerts
        """
        self._storage = storage
        self._alert_callback = alert_callback
        self._logger = logging.getLogger("credential_audit")

    def log_access(
        self,
        user_id: str,
        credential_id: str,
        broker: str,
        access_type: CredentialAccessType,
        purpose: str,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CredentialAccessEvent:
        """
        Log a credential access event.

        Args:
            user_id: User who owns the credential
            credential_id: Identifier of the accessed credential
            broker: Broker associated with the credential
            access_type: Type of access operation
            purpose: Stated reason for access
            source_ip: IP address of the request
            user_agent: User agent string
            success: Whether the access succeeded
            error_message: Error details if access failed
            request_id: Correlation ID for request tracing
            metadata: Additional context

        Returns:
            The created audit event
        """
        event = CredentialAccessEvent(
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            credential_id=credential_id,
            broker=broker,
            access_type=access_type,
            purpose=purpose,
            source_ip=source_ip,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            request_id=request_id,
            metadata=metadata or {},
        )

        # Store event
        self._storage.append(event)

        # Log to standard logger
        log_level = logging.INFO if success else logging.WARNING
        self._logger.log(log_level, event.to_log_string())

        # Check for anomalies asynchronously
        self._check_anomalies(event)

        return event

    def _check_anomalies(self, event: CredentialAccessEvent) -> None:
        """Check for anomalous access patterns."""
        anomalies = self.detect_anomalies(event.user_id)

        for alert in anomalies:
            self._logger.warning(
                f"ANOMALY_DETECTED | type={alert.alert_type} | "
                f"severity={alert.severity} | user={alert.user_id} | "
                f"description={alert.description}"
            )

            if self._alert_callback:
                self._alert_callback(alert)

    def detect_anomalies(
        self,
        user_id: str,
        check_window_hours: int = 24
    ) -> List[AnomalyAlert]:
        """
        Detect suspicious access patterns for a user.

        Args:
            user_id: User to check
            check_window_hours: Time window for analysis

        Returns:
            List of detected anomalies
        """
        anomalies: List[AnomalyAlert] = []
        now = datetime.now(timezone.utc)

        # Get recent events for user
        since_day = now - timedelta(hours=check_window_hours)
        since_hour = now - timedelta(hours=1)

        # Use higher limit to accurately detect high volume anomalies
        day_events = self._storage.query(user_id=user_id, since=since_day, limit=self.HIGH_VOLUME_THRESHOLD + 100)
        hour_events = self._storage.query(user_id=user_id, since=since_hour, limit=self.HIGH_VOLUME_THRESHOLD + 100)

        # Check: High volume
        if len(day_events) > self.HIGH_VOLUME_THRESHOLD:
            anomalies.append(AnomalyAlert(
                alert_type="high_volume",
                severity="medium",
                user_id=user_id,
                description=f"High access volume: {len(day_events)} accesses in 24h",
                timestamp=now,
                evidence={"access_count": len(day_events), "threshold": self.HIGH_VOLUME_THRESHOLD}
            ))

        # Check: Multiple failures
        failures = [e for e in hour_events if not e.success]
        if len(failures) > self.FAILURE_THRESHOLD:
            anomalies.append(AnomalyAlert(
                alert_type="repeated_failures",
                severity="high",
                user_id=user_id,
                description=f"Multiple access failures: {len(failures)} failures in 1h",
                timestamp=now,
                evidence={"failure_count": len(failures), "threshold": self.FAILURE_THRESHOLD}
            ))

        # Check: Multiple IPs
        ips = set(e.source_ip for e in hour_events if e.source_ip)
        if len(ips) > self.MULTI_IP_THRESHOLD:
            anomalies.append(AnomalyAlert(
                alert_type="multiple_ips",
                severity="medium",
                user_id=user_id,
                description=f"Access from multiple IPs: {len(ips)} distinct IPs in 1h",
                timestamp=now,
                evidence={"ip_count": len(ips), "ips": list(ips)[:10], "threshold": self.MULTI_IP_THRESHOLD}
            ))

        return anomalies

    def get_user_access_history(
        self,
        user_id: str,
        days: int = 90,
        limit: int = 1000
    ) -> List[CredentialAccessEvent]:
        """
        Get access history for a user (supports GDPR data subject requests).

        Args:
            user_id: User to query
            days: Number of days to look back
            limit: Maximum events to return

        Returns:
            List of access events for the user
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        return self._storage.query(
            user_id=user_id,
            since=since,
            limit=limit
        )

    def get_credential_access_history(
        self,
        credential_id: str,
        days: int = 90,
        limit: int = 1000
    ) -> List[CredentialAccessEvent]:
        """
        Get access history for a specific credential.

        Args:
            credential_id: Credential to query
            days: Number of days to look back
            limit: Maximum events to return

        Returns:
            List of access events for the credential
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        return self._storage.query(
            credential_id=credential_id,
            since=since,
            limit=limit
        )

    def apply_retention_policy(self, retention_days: int = 365) -> int:
        """
        Apply retention policy by deleting old events.

        Per GDPR and other regulations, audit logs typically must
        be retained for a minimum period but also deleted after
        the retention period expires.

        Args:
            retention_days: Number of days to retain events

        Returns:
            Number of events deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        deleted_count = self._storage.delete_before(cutoff)

        self._logger.info(
            f"RETENTION_POLICY_APPLIED | "
            f"retention_days={retention_days} | "
            f"deleted_count={deleted_count}"
        )

        return deleted_count

    def generate_access_report(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate an access report for a user.

        Useful for compliance reporting and user transparency.

        Args:
            user_id: User to report on
            days: Number of days to include

        Returns:
            Summary report with statistics and event list
        """
        events = self.get_user_access_history(user_id, days=days)

        # Compute statistics
        total_accesses = len(events)
        successful = sum(1 for e in events if e.success)
        failed = total_accesses - successful

        access_by_type = {}
        for event in events:
            type_name = event.access_type.value
            access_by_type[type_name] = access_by_type.get(type_name, 0) + 1

        access_by_broker = {}
        for event in events:
            access_by_broker[event.broker] = access_by_broker.get(event.broker, 0) + 1

        unique_ips = set(e.source_ip for e in events if e.source_ip)

        return {
            "user_id": user_id,
            "report_period_days": days,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "statistics": {
                "total_accesses": total_accesses,
                "successful": successful,
                "failed": failed,
                "success_rate": round(successful / total_accesses * 100, 2) if total_accesses else 0,
                "unique_ips": len(unique_ips),
            },
            "access_by_type": access_by_type,
            "access_by_broker": access_by_broker,
            "events": [e.to_dict() for e in events[-100:]],  # Last 100 events
        }
