# -*- coding: utf-8 -*-
"""
SIEM Integration Service (Splunk/ELK Export).

DORA Phase 3 Block 3.4: SIEM integration

Provides enterprise SIEM integration capabilities:
- Splunk HTTP Event Collector (HEC) export
- Elasticsearch/ELK Stack export
- Common Event Format (CEF) support
- Real-time event streaming
- Batch export capabilities

DORA References:
    - Art. 10: Detection requirements
    - Art. 17: ICT incident management
    - Art. 13(3): ICT security awareness and training
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================


class SIEMProvider(Enum):
    """Supported SIEM providers."""

    SPLUNK = "splunk"
    ELASTICSEARCH = "elasticsearch"
    ELK = "elk"  # Elasticsearch + Logstash + Kibana
    OPENSEARCH = "opensearch"
    AZURE_SENTINEL = "azure_sentinel"
    IBM_QRADAR = "ibm_qradar"
    GENERIC_SYSLOG = "generic_syslog"
    GENERIC_CEF = "generic_cef"


class EventSeverity(Enum):
    """Security event severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class EventCategory(Enum):
    """Security event categories per DORA requirements."""

    # Authentication events
    AUTH_SUCCESS = "authentication_success"
    AUTH_FAILURE = "authentication_failure"
    AUTH_LOCKOUT = "authentication_lockout"

    # Access events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PRIVILEGE_ESCALATION = "privilege_escalation"

    # Data events
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"

    # Security events
    MALWARE_DETECTED = "malware_detected"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    VULNERABILITY_SCAN = "vulnerability_scan"
    POLICY_VIOLATION = "policy_violation"

    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR = "error"

    # Incident events
    INCIDENT_CREATED = "incident_created"
    INCIDENT_UPDATED = "incident_updated"
    INCIDENT_RESOLVED = "incident_resolved"

    # Compliance events
    COMPLIANCE_CHECK = "compliance_check"
    AUDIT_LOG = "audit_log"


class ExportStatus(Enum):
    """Event export status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SIEMConfig:
    """SIEM connection configuration."""

    provider: SIEMProvider
    endpoint: str
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    index: str = "dora-events"
    source: str = "quantitative-research-platform"
    sourcetype: str = "dora:security"
    ssl_verify: bool = True
    batch_size: int = 100
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 5

    def get_connection_string(self) -> str:
        """Get masked connection string for logging."""
        return f"{self.provider.value}://{self.endpoint}/{self.index}"


@dataclass
class SecurityEvent:
    """Security event for SIEM export."""

    event_id: str
    timestamp: datetime
    category: EventCategory
    severity: EventSeverity
    source_system: str
    source_ip: str | None
    user_id: str | None
    client_id: str | None
    action: str
    outcome: str  # success, failure, unknown
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    raw_data: str | None = None

    def to_splunk_hec(self) -> dict[str, Any]:
        """Convert to Splunk HEC format."""
        return {
            "time": self.timestamp.timestamp(),
            "event": {
                "event_id": self.event_id,
                "category": self.category.value,
                "severity": self.severity.value,
                "source_system": self.source_system,
                "source_ip": self.source_ip,
                "user_id": self.user_id,
                "client_id": self.client_id,
                "action": self.action,
                "outcome": self.outcome,
                "message": self.message,
                "details": self.details,
                "tags": self.tags,
            },
            "source": "dora-platform",
            "sourcetype": "dora:security",
        }

    def to_elasticsearch(self) -> dict[str, Any]:
        """Convert to Elasticsearch document format."""
        return {
            "@timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "event": {
                "category": self.category.value,
                "severity": self.severity.value,
                "action": self.action,
                "outcome": self.outcome,
            },
            "source": {
                "system": self.source_system,
                "ip": self.source_ip,
            },
            "user": {"id": self.user_id},
            "client": {"id": self.client_id},
            "message": self.message,
            "details": self.details,
            "tags": self.tags,
        }

    def to_cef(self) -> str:
        """Convert to Common Event Format (CEF)."""
        # CEF format: CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
        severity_map = {
            EventSeverity.CRITICAL: 10,
            EventSeverity.HIGH: 7,
            EventSeverity.MEDIUM: 5,
            EventSeverity.LOW: 3,
            EventSeverity.INFORMATIONAL: 1,
        }
        cef_severity = severity_map.get(self.severity, 5)

        extensions = [
            f"rt={int(self.timestamp.timestamp() * 1000)}",
            f"cat={self.category.value}",
            f"act={self.action}",
            f"outcome={self.outcome}",
            f"msg={self.message}",
        ]
        if self.source_ip:
            extensions.append(f"src={self.source_ip}")
        if self.user_id:
            extensions.append(f"suser={self.user_id}")
        if self.client_id:
            extensions.append(f"cs1={self.client_id}")
            extensions.append("cs1Label=ClientID")

        return (
            f"CEF:0|DORA-Platform|QuantResearch|1.0|{self.event_id}|"
            f"{self.message[:50]}|{cef_severity}|{' '.join(extensions)}"
        )

    def to_syslog(self) -> str:
        """Convert to syslog format (RFC 5424)."""
        facility = 1  # user-level
        severity_map = {
            EventSeverity.CRITICAL: 2,  # Critical
            EventSeverity.HIGH: 3,  # Error
            EventSeverity.MEDIUM: 4,  # Warning
            EventSeverity.LOW: 5,  # Notice
            EventSeverity.INFORMATIONAL: 6,  # Informational
        }
        severity = severity_map.get(self.severity, 6)
        priority = facility * 8 + severity

        timestamp = self.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        hostname = self.source_system or "dora-platform"

        return (
            f"<{priority}>1 {timestamp} {hostname} DORA-Platform - {self.event_id} "
            f"[dora category=\"{self.category.value}\" action=\"{self.action}\"] {self.message}"
        )

    def calculate_hash(self) -> str:
        """Calculate event hash for deduplication."""
        content = f"{self.timestamp.isoformat()}{self.category.value}{self.action}{self.message}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class EventBatch:
    """Batch of events for export."""

    batch_id: str
    events: list[SecurityEvent]
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ExportStatus = ExportStatus.PENDING
    target_provider: SIEMProvider | None = None
    attempt_count: int = 0
    last_error: str | None = None
    exported_at: datetime | None = None

    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.events)

    def mark_in_progress(self) -> None:
        """Mark batch as in progress."""
        self.status = ExportStatus.IN_PROGRESS
        self.attempt_count += 1

    def mark_completed(self) -> None:
        """Mark batch as completed."""
        self.status = ExportStatus.COMPLETED
        self.exported_at = datetime.utcnow()

    def mark_failed(self, error: str) -> None:
        """Mark batch as failed."""
        self.last_error = error
        self.status = ExportStatus.FAILED


@dataclass
class ExportResult:
    """Result of an export operation."""

    batch_id: str
    provider: SIEMProvider
    status: ExportStatus
    events_exported: int
    events_failed: int
    started_at: datetime
    completed_at: datetime | None
    duration_ms: int | None
    error_message: str | None = None
    response_data: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.events_exported + self.events_failed
        if total == 0:
            return 0.0
        return self.events_exported / total * 100


@dataclass
class SIEMConnection:
    """SIEM connection state."""

    connection_id: str
    config: SIEMConfig
    is_connected: bool = False
    last_connected: datetime | None = None
    last_error: str | None = None
    events_exported_total: int = 0
    batches_exported_total: int = 0


# =============================================================================
# SIEM Exporters
# =============================================================================


class BaseSIEMExporter:
    """Base class for SIEM exporters."""

    def __init__(self, config: SIEMConfig) -> None:
        """Initialize exporter."""
        self.config = config

    def export_event(self, event: SecurityEvent) -> bool:
        """Export a single event."""
        raise NotImplementedError

    def export_batch(self, batch: EventBatch) -> ExportResult:
        """Export a batch of events."""
        raise NotImplementedError

    def test_connection(self) -> bool:
        """Test connection to SIEM."""
        raise NotImplementedError


class SplunkExporter(BaseSIEMExporter):
    """Splunk HTTP Event Collector exporter."""

    def export_event(self, event: SecurityEvent) -> bool:
        """Export a single event to Splunk."""
        # In production, this would use requests library
        # Simulating export for now
        hec_data = event.to_splunk_hec()
        return self._send_to_hec([hec_data])

    def export_batch(self, batch: EventBatch) -> ExportResult:
        """Export a batch of events to Splunk."""
        started_at = datetime.utcnow()
        batch.mark_in_progress()

        try:
            hec_data = [event.to_splunk_hec() for event in batch.events]
            success = self._send_to_hec(hec_data)

            completed_at = datetime.utcnow()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            if success:
                batch.mark_completed()
                return ExportResult(
                    batch_id=batch.batch_id,
                    provider=SIEMProvider.SPLUNK,
                    status=ExportStatus.COMPLETED,
                    events_exported=len(batch.events),
                    events_failed=0,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                )
            else:
                batch.mark_failed("HEC export failed")
                return ExportResult(
                    batch_id=batch.batch_id,
                    provider=SIEMProvider.SPLUNK,
                    status=ExportStatus.FAILED,
                    events_exported=0,
                    events_failed=len(batch.events),
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                    error_message="HEC export failed",
                )
        except Exception as e:
            batch.mark_failed(str(e))
            return ExportResult(
                batch_id=batch.batch_id,
                provider=SIEMProvider.SPLUNK,
                status=ExportStatus.FAILED,
                events_exported=0,
                events_failed=len(batch.events),
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_ms=0,
                error_message=str(e),
            )

    def _send_to_hec(self, events: list[dict[str, Any]]) -> bool:
        """Send events to Splunk HEC endpoint."""
        # In production, this would use HTTP POST to HEC endpoint
        # Simulating successful export
        return True

    def test_connection(self) -> bool:
        """Test connection to Splunk."""
        # In production, this would send a test event
        return True


class ElasticsearchExporter(BaseSIEMExporter):
    """Elasticsearch/ELK Stack exporter."""

    def export_event(self, event: SecurityEvent) -> bool:
        """Export a single event to Elasticsearch."""
        doc = event.to_elasticsearch()
        return self._index_document(doc)

    def export_batch(self, batch: EventBatch) -> ExportResult:
        """Export a batch of events to Elasticsearch using bulk API."""
        started_at = datetime.utcnow()
        batch.mark_in_progress()

        try:
            docs = [event.to_elasticsearch() for event in batch.events]
            success_count, fail_count = self._bulk_index(docs)

            completed_at = datetime.utcnow()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            if fail_count == 0:
                batch.mark_completed()
                status = ExportStatus.COMPLETED
            else:
                batch.mark_failed(f"{fail_count} documents failed to index")
                status = ExportStatus.FAILED if success_count == 0 else ExportStatus.COMPLETED

            return ExportResult(
                batch_id=batch.batch_id,
                provider=SIEMProvider.ELASTICSEARCH,
                status=status,
                events_exported=success_count,
                events_failed=fail_count,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )
        except Exception as e:
            batch.mark_failed(str(e))
            return ExportResult(
                batch_id=batch.batch_id,
                provider=SIEMProvider.ELASTICSEARCH,
                status=ExportStatus.FAILED,
                events_exported=0,
                events_failed=len(batch.events),
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_ms=0,
                error_message=str(e),
            )

    def _index_document(self, doc: dict[str, Any]) -> bool:
        """Index a single document."""
        # In production, this would use Elasticsearch client
        return True

    def _bulk_index(self, docs: list[dict[str, Any]]) -> tuple[int, int]:
        """Bulk index documents."""
        # In production, this would use Elasticsearch bulk API
        return len(docs), 0

    def test_connection(self) -> bool:
        """Test connection to Elasticsearch."""
        # In production, this would ping the cluster
        return True


# =============================================================================
# Main Service Class
# =============================================================================


class SIEMExportService:
    """
    SIEM Integration Service.

    Provides enterprise SIEM integration capabilities per DORA Art. 10.
    """

    def __init__(self) -> None:
        """Initialize SIEM export service."""
        self._connections: dict[str, SIEMConnection] = {}
        self._exporters: dict[str, BaseSIEMExporter] = {}
        self._events: list[SecurityEvent] = []
        self._batches: dict[str, EventBatch] = {}
        self._export_results: list[ExportResult] = []

    def add_connection(self, config: SIEMConfig) -> SIEMConnection:
        """Add a SIEM connection."""
        connection_id = str(uuid4())
        connection = SIEMConnection(connection_id=connection_id, config=config)

        # Create appropriate exporter
        if config.provider == SIEMProvider.SPLUNK:
            self._exporters[connection_id] = SplunkExporter(config)
        elif config.provider in (SIEMProvider.ELASTICSEARCH, SIEMProvider.ELK, SIEMProvider.OPENSEARCH):
            self._exporters[connection_id] = ElasticsearchExporter(config)
        else:
            self._exporters[connection_id] = ElasticsearchExporter(config)  # Default to ES

        self._connections[connection_id] = connection
        return connection

    def remove_connection(self, connection_id: str) -> bool:
        """Remove a SIEM connection."""
        if connection_id in self._connections:
            del self._connections[connection_id]
            del self._exporters[connection_id]
            return True
        return False

    def get_connection(self, connection_id: str) -> SIEMConnection | None:
        """Get connection by ID."""
        return self._connections.get(connection_id)

    def list_connections(self) -> list[SIEMConnection]:
        """List all connections."""
        return list(self._connections.values())

    def test_connection(self, connection_id: str) -> bool:
        """Test a SIEM connection."""
        exporter = self._exporters.get(connection_id)
        if not exporter:
            return False

        connection = self._connections[connection_id]
        try:
            result = exporter.test_connection()
            if result:
                connection.is_connected = True
                connection.last_connected = datetime.utcnow()
            return result
        except Exception as e:
            connection.is_connected = False
            connection.last_error = str(e)
            return False

    def create_event(
        self,
        category: EventCategory,
        severity: EventSeverity,
        action: str,
        outcome: str,
        message: str,
        source_system: str = "dora-platform",
        source_ip: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
        details: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> SecurityEvent:
        """Create a new security event."""
        event = SecurityEvent(
            event_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            category=category,
            severity=severity,
            source_system=source_system,
            source_ip=source_ip,
            user_id=user_id,
            client_id=client_id,
            action=action,
            outcome=outcome,
            message=message,
            details=details or {},
            tags=tags or [],
        )
        self._events.append(event)
        return event

    def export_event(self, event: SecurityEvent, connection_id: str) -> bool:
        """Export a single event to a SIEM."""
        exporter = self._exporters.get(connection_id)
        if not exporter:
            return False

        result = exporter.export_event(event)
        if result:
            connection = self._connections[connection_id]
            connection.events_exported_total += 1
        return result

    def create_batch(self, events: list[SecurityEvent]) -> EventBatch:
        """Create an event batch."""
        batch = EventBatch(batch_id=str(uuid4()), events=events)
        self._batches[batch.batch_id] = batch
        return batch

    def export_batch(self, batch_id: str, connection_id: str) -> ExportResult:
        """Export a batch of events to a SIEM."""
        batch = self._batches.get(batch_id)
        if not batch:
            raise ValueError(f"Batch not found: {batch_id}")

        exporter = self._exporters.get(connection_id)
        if not exporter:
            raise ValueError(f"Connection not found: {connection_id}")

        batch.target_provider = self._connections[connection_id].config.provider
        result = exporter.export_batch(batch)

        connection = self._connections[connection_id]
        connection.events_exported_total += result.events_exported
        connection.batches_exported_total += 1

        self._export_results.append(result)
        return result

    def export_pending_events(self, connection_id: str, batch_size: int | None = None) -> ExportResult | None:
        """Export pending events to a SIEM."""
        connection = self._connections.get(connection_id)
        if not connection:
            return None

        size = batch_size or connection.config.batch_size

        # Get events not yet exported
        pending_events = self._events[-size:] if self._events else []
        if not pending_events:
            return None

        batch = self.create_batch(pending_events)
        return self.export_batch(batch.batch_id, connection_id)

    def get_export_statistics(self, connection_id: str | None = None) -> dict[str, Any]:
        """Get export statistics."""
        if connection_id:
            connection = self._connections.get(connection_id)
            if not connection:
                return {}

            results = [r for r in self._export_results if r.batch_id in self._batches and self._batches[r.batch_id].target_provider == connection.config.provider]
            return {
                "connection_id": connection_id,
                "provider": connection.config.provider.value,
                "is_connected": connection.is_connected,
                "events_exported_total": connection.events_exported_total,
                "batches_exported_total": connection.batches_exported_total,
                "last_connected": connection.last_connected.isoformat() if connection.last_connected else None,
            }

        # Global statistics
        return {
            "total_connections": len(self._connections),
            "active_connections": sum(1 for c in self._connections.values() if c.is_connected),
            "total_events_pending": len(self._events),
            "total_batches": len(self._batches),
            "total_exports": len(self._export_results),
            "successful_exports": sum(1 for r in self._export_results if r.status == ExportStatus.COMPLETED),
            "failed_exports": sum(1 for r in self._export_results if r.status == ExportStatus.FAILED),
        }

    def get_events_by_category(
        self,
        category: EventCategory,
        since: datetime | None = None,
    ) -> list[SecurityEvent]:
        """Get events by category."""
        events = [e for e in self._events if e.category == category]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events

    def get_events_by_severity(
        self,
        severity: EventSeverity,
        since: datetime | None = None,
    ) -> list[SecurityEvent]:
        """Get events by severity."""
        events = [e for e in self._events if e.severity == severity]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events

    def get_events_by_client(
        self,
        client_id: str,
        since: datetime | None = None,
    ) -> list[SecurityEvent]:
        """Get events for a specific client."""
        events = [e for e in self._events if e.client_id == client_id]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events


# =============================================================================
# Factory Functions
# =============================================================================


def create_siem_export() -> SIEMExportService:
    """Create SIEM export service instance."""
    return SIEMExportService()


def export_to_splunk(
    service: SIEMExportService,
    endpoint: str,
    api_key: str,
    events: list[SecurityEvent],
    index: str = "dora-events",
) -> ExportResult:
    """Export events to Splunk (convenience function)."""
    config = SIEMConfig(
        provider=SIEMProvider.SPLUNK,
        endpoint=endpoint,
        api_key=api_key,
        index=index,
    )
    connection = service.add_connection(config)
    batch = service.create_batch(events)
    return service.export_batch(batch.batch_id, connection.connection_id)


def export_to_elk(
    service: SIEMExportService,
    endpoint: str,
    events: list[SecurityEvent],
    index: str = "dora-events",
    username: str | None = None,
    password: str | None = None,
) -> ExportResult:
    """Export events to ELK Stack (convenience function)."""
    config = SIEMConfig(
        provider=SIEMProvider.ELK,
        endpoint=endpoint,
        username=username,
        password=password,
        index=index,
    )
    connection = service.add_connection(config)
    batch = service.create_batch(events)
    return service.export_batch(batch.batch_id, connection.connection_id)
