# -*- coding: utf-8 -*-
"""
Comprehensive tests for SIEM Integration Service.

Tests DORA Phase 3 Block 3.4: SIEM integration (Splunk/ELK export).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from services.enterprise.siem_export import (
    # Enums
    EventCategory,
    EventSeverity,
    ExportStatus,
    SIEMProvider,
    # Data structures
    EventBatch,
    ExportResult,
    SecurityEvent,
    SIEMConfig,
    SIEMConnection,
    # Exporters
    BaseSIEMExporter,
    ElasticsearchExporter,
    SplunkExporter,
    # Service
    SIEMExportService,
    # Factory
    create_siem_export,
    export_to_elk,
    export_to_splunk,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestSIEMProvider:
    """Tests for SIEMProvider enum."""

    def test_enum_values(self) -> None:
        """Test all SIEM providers exist."""
        assert SIEMProvider.SPLUNK.value == "splunk"
        assert SIEMProvider.ELASTICSEARCH.value == "elasticsearch"
        assert SIEMProvider.ELK.value == "elk"
        assert SIEMProvider.OPENSEARCH.value == "opensearch"
        assert SIEMProvider.AZURE_SENTINEL.value == "azure_sentinel"
        assert SIEMProvider.IBM_QRADAR.value == "ibm_qradar"


class TestEventSeverity:
    """Tests for EventSeverity enum."""

    def test_enum_values(self) -> None:
        """Test all severity levels exist."""
        assert EventSeverity.CRITICAL.value == "critical"
        assert EventSeverity.HIGH.value == "high"
        assert EventSeverity.MEDIUM.value == "medium"
        assert EventSeverity.LOW.value == "low"
        assert EventSeverity.INFORMATIONAL.value == "informational"


class TestEventCategory:
    """Tests for EventCategory enum."""

    def test_authentication_categories(self) -> None:
        """Test authentication event categories."""
        assert EventCategory.AUTH_SUCCESS.value == "authentication_success"
        assert EventCategory.AUTH_FAILURE.value == "authentication_failure"
        assert EventCategory.AUTH_LOCKOUT.value == "authentication_lockout"

    def test_access_categories(self) -> None:
        """Test access event categories."""
        assert EventCategory.ACCESS_GRANTED.value == "access_granted"
        assert EventCategory.ACCESS_DENIED.value == "access_denied"
        assert EventCategory.PRIVILEGE_ESCALATION.value == "privilege_escalation"

    def test_data_categories(self) -> None:
        """Test data event categories."""
        assert EventCategory.DATA_ACCESS.value == "data_access"
        assert EventCategory.DATA_MODIFICATION.value == "data_modification"
        assert EventCategory.DATA_DELETION.value == "data_deletion"


class TestExportStatus:
    """Tests for ExportStatus enum."""

    def test_enum_values(self) -> None:
        """Test all export statuses exist."""
        assert ExportStatus.PENDING.value == "pending"
        assert ExportStatus.IN_PROGRESS.value == "in_progress"
        assert ExportStatus.COMPLETED.value == "completed"
        assert ExportStatus.FAILED.value == "failed"
        assert ExportStatus.RETRYING.value == "retrying"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestSIEMConfig:
    """Tests for SIEMConfig dataclass."""

    def test_creation(self) -> None:
        """Test config creation with all fields."""
        config = SIEMConfig(
            provider=SIEMProvider.SPLUNK,
            endpoint="https://splunk.example.com:8088",
            api_key="test-api-key",
            index="security-events",
        )
        assert config.provider == SIEMProvider.SPLUNK
        assert config.endpoint == "https://splunk.example.com:8088"
        assert config.api_key == "test-api-key"
        assert config.index == "security-events"

    def test_default_values(self) -> None:
        """Test config default values."""
        config = SIEMConfig(
            provider=SIEMProvider.SPLUNK,
            endpoint="https://splunk.example.com:8088",
        )
        assert config.index == "dora-events"
        assert config.source == "quantitative-research-platform"
        assert config.sourcetype == "dora:security"
        assert config.ssl_verify is True
        assert config.batch_size == 100

    def test_get_connection_string(self) -> None:
        """Test connection string generation."""
        config = SIEMConfig(
            provider=SIEMProvider.SPLUNK,
            endpoint="splunk.example.com:8088",
            index="test-index",
        )
        conn_str = config.get_connection_string()
        assert "splunk" in conn_str
        assert "splunk.example.com" in conn_str
        assert "test-index" in conn_str


class TestSecurityEvent:
    """Tests for SecurityEvent dataclass."""

    def test_creation(self) -> None:
        """Test event creation with all fields."""
        event = SecurityEvent(
            event_id="evt-001",
            timestamp=datetime.utcnow(),
            category=EventCategory.AUTH_FAILURE,
            severity=EventSeverity.HIGH,
            source_system="auth-service",
            source_ip="192.168.1.100",
            user_id="user-123",
            client_id="client-001",
            action="login",
            outcome="failure",
            message="Failed login attempt",
        )
        assert event.event_id == "evt-001"
        assert event.category == EventCategory.AUTH_FAILURE
        assert event.severity == EventSeverity.HIGH
        assert event.source_ip == "192.168.1.100"

    def test_to_splunk_hec(self) -> None:
        """Test conversion to Splunk HEC format."""
        event = SecurityEvent(
            event_id="evt-001",
            timestamp=datetime.utcnow(),
            category=EventCategory.AUTH_FAILURE,
            severity=EventSeverity.HIGH,
            source_system="auth-service",
            source_ip="192.168.1.100",
            user_id="user-123",
            client_id="client-001",
            action="login",
            outcome="failure",
            message="Failed login attempt",
        )
        hec_data = event.to_splunk_hec()
        assert "time" in hec_data
        assert "event" in hec_data
        assert hec_data["event"]["event_id"] == "evt-001"
        assert hec_data["event"]["severity"] == "high"

    def test_to_elasticsearch(self) -> None:
        """Test conversion to Elasticsearch format."""
        event = SecurityEvent(
            event_id="evt-001",
            timestamp=datetime.utcnow(),
            category=EventCategory.AUTH_FAILURE,
            severity=EventSeverity.HIGH,
            source_system="auth-service",
            source_ip="192.168.1.100",
            user_id="user-123",
            client_id="client-001",
            action="login",
            outcome="failure",
            message="Failed login attempt",
        )
        es_data = event.to_elasticsearch()
        assert "@timestamp" in es_data
        assert "event_id" in es_data
        assert es_data["event"]["category"] == "authentication_failure"

    def test_to_cef(self) -> None:
        """Test conversion to CEF format."""
        event = SecurityEvent(
            event_id="evt-001",
            timestamp=datetime.utcnow(),
            category=EventCategory.AUTH_FAILURE,
            severity=EventSeverity.HIGH,
            source_system="auth-service",
            source_ip="192.168.1.100",
            user_id="user-123",
            client_id="client-001",
            action="login",
            outcome="failure",
            message="Failed login attempt",
        )
        cef = event.to_cef()
        assert cef.startswith("CEF:0")
        assert "DORA-Platform" in cef
        assert "src=192.168.1.100" in cef

    def test_to_syslog(self) -> None:
        """Test conversion to syslog format."""
        event = SecurityEvent(
            event_id="evt-001",
            timestamp=datetime.utcnow(),
            category=EventCategory.AUTH_FAILURE,
            severity=EventSeverity.HIGH,
            source_system="auth-service",
            source_ip=None,
            user_id=None,
            client_id=None,
            action="login",
            outcome="failure",
            message="Failed login attempt",
        )
        syslog = event.to_syslog()
        assert "<" in syslog
        assert "DORA-Platform" in syslog
        assert "evt-001" in syslog

    def test_calculate_hash(self) -> None:
        """Test event hash calculation."""
        event = SecurityEvent(
            event_id="evt-001",
            timestamp=datetime.utcnow(),
            category=EventCategory.AUTH_FAILURE,
            severity=EventSeverity.HIGH,
            source_system="auth-service",
            source_ip=None,
            user_id=None,
            client_id=None,
            action="login",
            outcome="failure",
            message="Failed login attempt",
        )
        hash1 = event.calculate_hash()
        assert len(hash1) == 16


class TestEventBatch:
    """Tests for EventBatch dataclass."""

    def test_creation(self) -> None:
        """Test batch creation."""
        events = [
            SecurityEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.utcnow(),
                category=EventCategory.AUTH_SUCCESS,
                severity=EventSeverity.INFORMATIONAL,
                source_system="auth",
                source_ip=None,
                user_id=None,
                client_id=None,
                action="login",
                outcome="success",
                message="Successful login",
            )
            for i in range(5)
        ]
        batch = EventBatch(batch_id="batch-001", events=events)
        assert batch.batch_id == "batch-001"
        assert batch.size == 5
        assert batch.status == ExportStatus.PENDING

    def test_mark_in_progress(self) -> None:
        """Test marking batch as in progress."""
        batch = EventBatch(batch_id="batch-001", events=[])
        batch.mark_in_progress()
        assert batch.status == ExportStatus.IN_PROGRESS
        assert batch.attempt_count == 1

    def test_mark_completed(self) -> None:
        """Test marking batch as completed."""
        batch = EventBatch(batch_id="batch-001", events=[])
        batch.mark_completed()
        assert batch.status == ExportStatus.COMPLETED
        assert batch.exported_at is not None

    def test_mark_failed(self) -> None:
        """Test marking batch as failed."""
        batch = EventBatch(batch_id="batch-001", events=[])
        batch.mark_failed("Connection timeout")
        assert batch.status == ExportStatus.FAILED
        assert batch.last_error == "Connection timeout"


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_creation(self) -> None:
        """Test result creation."""
        result = ExportResult(
            batch_id="batch-001",
            provider=SIEMProvider.SPLUNK,
            status=ExportStatus.COMPLETED,
            events_exported=100,
            events_failed=0,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=150,
        )
        assert result.batch_id == "batch-001"
        assert result.events_exported == 100
        assert result.success_rate == 100.0

    def test_success_rate_partial(self) -> None:
        """Test success rate with failures."""
        result = ExportResult(
            batch_id="batch-001",
            provider=SIEMProvider.SPLUNK,
            status=ExportStatus.COMPLETED,
            events_exported=80,
            events_failed=20,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=150,
        )
        assert result.success_rate == 80.0

    def test_success_rate_empty(self) -> None:
        """Test success rate with no events."""
        result = ExportResult(
            batch_id="batch-001",
            provider=SIEMProvider.SPLUNK,
            status=ExportStatus.COMPLETED,
            events_exported=0,
            events_failed=0,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=0,
        )
        assert result.success_rate == 0.0


class TestSIEMConnection:
    """Tests for SIEMConnection dataclass."""

    def test_creation(self) -> None:
        """Test connection creation."""
        config = SIEMConfig(
            provider=SIEMProvider.SPLUNK,
            endpoint="https://splunk.example.com:8088",
        )
        connection = SIEMConnection(
            connection_id="conn-001",
            config=config,
        )
        assert connection.connection_id == "conn-001"
        assert connection.is_connected is False
        assert connection.events_exported_total == 0


# =============================================================================
# Service Tests
# =============================================================================


class TestSIEMExportService:
    """Tests for SIEMExportService."""

    @pytest.fixture
    def service(self) -> SIEMExportService:
        """Create service instance for testing."""
        return SIEMExportService()

    @pytest.fixture
    def splunk_config(self) -> SIEMConfig:
        """Create Splunk configuration."""
        return SIEMConfig(
            provider=SIEMProvider.SPLUNK,
            endpoint="https://splunk.example.com:8088",
            api_key="test-key",
        )

    @pytest.fixture
    def elk_config(self) -> SIEMConfig:
        """Create ELK configuration."""
        return SIEMConfig(
            provider=SIEMProvider.ELK,
            endpoint="https://elk.example.com:9200",
            username="elastic",
            password="secret",
        )

    def test_initialization(self, service: SIEMExportService) -> None:
        """Test service initialization."""
        assert len(service._connections) == 0
        assert len(service._events) == 0

    def test_add_connection(
        self, service: SIEMExportService, splunk_config: SIEMConfig
    ) -> None:
        """Test adding a SIEM connection."""
        connection = service.add_connection(splunk_config)
        assert connection.connection_id is not None
        assert connection.config == splunk_config
        assert len(service._connections) == 1

    def test_add_multiple_connections(
        self,
        service: SIEMExportService,
        splunk_config: SIEMConfig,
        elk_config: SIEMConfig,
    ) -> None:
        """Test adding multiple SIEM connections."""
        conn1 = service.add_connection(splunk_config)
        conn2 = service.add_connection(elk_config)
        assert len(service._connections) == 2
        assert conn1.connection_id != conn2.connection_id

    def test_remove_connection(
        self, service: SIEMExportService, splunk_config: SIEMConfig
    ) -> None:
        """Test removing a SIEM connection."""
        connection = service.add_connection(splunk_config)
        result = service.remove_connection(connection.connection_id)
        assert result is True
        assert len(service._connections) == 0

    def test_remove_connection_not_found(self, service: SIEMExportService) -> None:
        """Test removing non-existent connection."""
        result = service.remove_connection("non-existent")
        assert result is False

    def test_get_connection(
        self, service: SIEMExportService, splunk_config: SIEMConfig
    ) -> None:
        """Test getting connection by ID."""
        connection = service.add_connection(splunk_config)
        retrieved = service.get_connection(connection.connection_id)
        assert retrieved is not None
        assert retrieved.connection_id == connection.connection_id

    def test_list_connections(
        self,
        service: SIEMExportService,
        splunk_config: SIEMConfig,
        elk_config: SIEMConfig,
    ) -> None:
        """Test listing all connections."""
        service.add_connection(splunk_config)
        service.add_connection(elk_config)
        connections = service.list_connections()
        assert len(connections) == 2

    def test_test_connection(
        self, service: SIEMExportService, splunk_config: SIEMConfig
    ) -> None:
        """Test testing a SIEM connection."""
        connection = service.add_connection(splunk_config)
        result = service.test_connection(connection.connection_id)
        assert result is True
        assert connection.is_connected is True
        assert connection.last_connected is not None

    def test_create_event(self, service: SIEMExportService) -> None:
        """Test creating a security event."""
        event = service.create_event(
            category=EventCategory.AUTH_FAILURE,
            severity=EventSeverity.HIGH,
            action="login",
            outcome="failure",
            message="Failed login attempt",
            source_system="auth-service",
            user_id="user-123",
        )
        assert event.event_id is not None
        assert event.category == EventCategory.AUTH_FAILURE
        assert len(service._events) == 1

    def test_export_event(
        self, service: SIEMExportService, splunk_config: SIEMConfig
    ) -> None:
        """Test exporting a single event."""
        connection = service.add_connection(splunk_config)
        event = service.create_event(
            category=EventCategory.AUTH_SUCCESS,
            severity=EventSeverity.LOW,
            action="login",
            outcome="success",
            message="Successful login",
        )
        result = service.export_event(event, connection.connection_id)
        assert result is True
        assert connection.events_exported_total == 1

    def test_create_batch(self, service: SIEMExportService) -> None:
        """Test creating an event batch."""
        events = [
            service.create_event(
                category=EventCategory.AUTH_SUCCESS,
                severity=EventSeverity.INFORMATIONAL,
                action="login",
                outcome="success",
                message=f"Event {i}",
            )
            for i in range(5)
        ]
        batch = service.create_batch(events)
        assert batch.batch_id is not None
        assert batch.size == 5

    def test_export_batch(
        self, service: SIEMExportService, splunk_config: SIEMConfig
    ) -> None:
        """Test exporting an event batch."""
        connection = service.add_connection(splunk_config)
        events = [
            service.create_event(
                category=EventCategory.AUTH_SUCCESS,
                severity=EventSeverity.INFORMATIONAL,
                action="login",
                outcome="success",
                message=f"Event {i}",
            )
            for i in range(5)
        ]
        batch = service.create_batch(events)
        result = service.export_batch(batch.batch_id, connection.connection_id)
        assert result.status == ExportStatus.COMPLETED
        assert result.events_exported == 5
        assert connection.batches_exported_total == 1

    def test_export_pending_events(
        self, service: SIEMExportService, splunk_config: SIEMConfig
    ) -> None:
        """Test exporting pending events."""
        connection = service.add_connection(splunk_config)
        for i in range(10):
            service.create_event(
                category=EventCategory.AUTH_SUCCESS,
                severity=EventSeverity.INFORMATIONAL,
                action="login",
                outcome="success",
                message=f"Event {i}",
            )
        result = service.export_pending_events(
            connection.connection_id, batch_size=5
        )
        assert result is not None
        assert result.events_exported == 5

    def test_get_export_statistics(
        self, service: SIEMExportService, splunk_config: SIEMConfig
    ) -> None:
        """Test getting export statistics."""
        connection = service.add_connection(splunk_config)
        service.test_connection(connection.connection_id)

        stats = service.get_export_statistics(connection.connection_id)
        assert stats["connection_id"] == connection.connection_id
        assert stats["provider"] == "splunk"
        assert stats["is_connected"] is True

    def test_get_export_statistics_global(
        self, service: SIEMExportService, splunk_config: SIEMConfig
    ) -> None:
        """Test getting global export statistics."""
        service.add_connection(splunk_config)
        stats = service.get_export_statistics()
        assert stats["total_connections"] == 1
        assert "total_events_pending" in stats

    def test_get_events_by_category(self, service: SIEMExportService) -> None:
        """Test getting events by category."""
        service.create_event(
            category=EventCategory.AUTH_FAILURE,
            severity=EventSeverity.HIGH,
            action="login",
            outcome="failure",
            message="Failed login",
        )
        service.create_event(
            category=EventCategory.AUTH_SUCCESS,
            severity=EventSeverity.LOW,
            action="login",
            outcome="success",
            message="Successful login",
        )

        failures = service.get_events_by_category(EventCategory.AUTH_FAILURE)
        assert len(failures) == 1
        assert failures[0].category == EventCategory.AUTH_FAILURE

    def test_get_events_by_severity(self, service: SIEMExportService) -> None:
        """Test getting events by severity."""
        service.create_event(
            category=EventCategory.AUTH_FAILURE,
            severity=EventSeverity.HIGH,
            action="login",
            outcome="failure",
            message="Failed login",
        )
        service.create_event(
            category=EventCategory.AUTH_SUCCESS,
            severity=EventSeverity.LOW,
            action="login",
            outcome="success",
            message="Successful login",
        )

        high_severity = service.get_events_by_severity(EventSeverity.HIGH)
        assert len(high_severity) == 1
        assert high_severity[0].severity == EventSeverity.HIGH

    def test_get_events_by_client(self, service: SIEMExportService) -> None:
        """Test getting events by client."""
        service.create_event(
            category=EventCategory.AUTH_SUCCESS,
            severity=EventSeverity.LOW,
            action="login",
            outcome="success",
            message="Login",
            client_id="client-001",
        )
        service.create_event(
            category=EventCategory.AUTH_SUCCESS,
            severity=EventSeverity.LOW,
            action="login",
            outcome="success",
            message="Login",
            client_id="client-002",
        )

        events = service.get_events_by_client("client-001")
        assert len(events) == 1
        assert events[0].client_id == "client-001"


# =============================================================================
# Exporter Tests
# =============================================================================


class TestSplunkExporter:
    """Tests for SplunkExporter."""

    @pytest.fixture
    def exporter(self) -> SplunkExporter:
        """Create exporter instance."""
        config = SIEMConfig(
            provider=SIEMProvider.SPLUNK,
            endpoint="https://splunk.example.com:8088",
            api_key="test-key",
        )
        return SplunkExporter(config)

    def test_export_event(self, exporter: SplunkExporter) -> None:
        """Test exporting single event."""
        event = SecurityEvent(
            event_id="evt-001",
            timestamp=datetime.utcnow(),
            category=EventCategory.AUTH_SUCCESS,
            severity=EventSeverity.LOW,
            source_system="auth",
            source_ip=None,
            user_id=None,
            client_id=None,
            action="login",
            outcome="success",
            message="Successful login",
        )
        result = exporter.export_event(event)
        assert result is True

    def test_test_connection(self, exporter: SplunkExporter) -> None:
        """Test connection test."""
        result = exporter.test_connection()
        assert result is True


class TestElasticsearchExporter:
    """Tests for ElasticsearchExporter."""

    @pytest.fixture
    def exporter(self) -> ElasticsearchExporter:
        """Create exporter instance."""
        config = SIEMConfig(
            provider=SIEMProvider.ELASTICSEARCH,
            endpoint="https://elastic.example.com:9200",
        )
        return ElasticsearchExporter(config)

    def test_export_event(self, exporter: ElasticsearchExporter) -> None:
        """Test exporting single event."""
        event = SecurityEvent(
            event_id="evt-001",
            timestamp=datetime.utcnow(),
            category=EventCategory.AUTH_SUCCESS,
            severity=EventSeverity.LOW,
            source_system="auth",
            source_ip=None,
            user_id=None,
            client_id=None,
            action="login",
            outcome="success",
            message="Successful login",
        )
        result = exporter.export_event(event)
        assert result is True

    def test_test_connection(self, exporter: ElasticsearchExporter) -> None:
        """Test connection test."""
        result = exporter.test_connection()
        assert result is True


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_siem_export(self) -> None:
        """Test creating SIEM export service."""
        service = create_siem_export()
        assert service is not None
        assert isinstance(service, SIEMExportService)

    def test_export_to_splunk(self) -> None:
        """Test convenience function for Splunk export."""
        service = create_siem_export()
        events = [
            SecurityEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.utcnow(),
                category=EventCategory.AUTH_SUCCESS,
                severity=EventSeverity.LOW,
                source_system="auth",
                source_ip=None,
                user_id=None,
                client_id=None,
                action="login",
                outcome="success",
                message="Successful login",
            )
            for i in range(3)
        ]
        result = export_to_splunk(
            service=service,
            endpoint="https://splunk.example.com:8088",
            api_key="test-key",
            events=events,
        )
        assert result.status == ExportStatus.COMPLETED
        assert result.events_exported == 3

    def test_export_to_elk(self) -> None:
        """Test convenience function for ELK export."""
        service = create_siem_export()
        events = [
            SecurityEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.utcnow(),
                category=EventCategory.AUTH_SUCCESS,
                severity=EventSeverity.LOW,
                source_system="auth",
                source_ip=None,
                user_id=None,
                client_id=None,
                action="login",
                outcome="success",
                message="Successful login",
            )
            for i in range(3)
        ]
        result = export_to_elk(
            service=service,
            endpoint="https://elastic.example.com:9200",
            events=events,
        )
        assert result.status == ExportStatus.COMPLETED
