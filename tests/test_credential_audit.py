"""
Tests for CredentialAuditLogger - Audit logging for credential access.

Comprehensive tests covering:
    - Access event logging
    - Anomaly detection
    - User history queries
    - Retention policy

References:
    - ISO 27001 A.12.4: Logging and monitoring
    - SOC 2 CC6.1: Logical access security
    - PCI DSS 10.2: Audit trail
"""

import pytest
from datetime import datetime, timedelta, timezone

from services.security.credential_audit import (
    CredentialAuditLogger,
    CredentialAccessType,
    CredentialAccessEvent,
    InMemoryAuditStorage,
    AnomalyAlert,
)


class TestCredentialAccessEvent:
    """Tests for CredentialAccessEvent data class."""

    @pytest.fixture
    def sample_event(self):
        """Create a sample event."""
        return CredentialAccessEvent(
            timestamp=datetime.now(timezone.utc),
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="order_execution",
            source_ip="127.0.0.1",
            user_agent="Mozilla/5.0",
            success=True,
            request_id="req_001",
            metadata={"extra": "data"},
        )

    def test_event_to_dict(self, sample_event):
        """Verify event serialization."""
        data = sample_event.to_dict()

        assert data["user_id"] == "user_001"
        assert data["credential_id"] == "cred_123"
        assert data["broker"] == "alpaca"
        assert data["access_type"] == "decrypt"
        assert data["success"] is True

    def test_event_from_dict(self, sample_event):
        """Verify event deserialization."""
        data = sample_event.to_dict()
        restored = CredentialAccessEvent.from_dict(data)

        assert restored.user_id == sample_event.user_id
        assert restored.credential_id == sample_event.credential_id
        assert restored.access_type == sample_event.access_type
        assert restored.success == sample_event.success

    def test_event_to_log_string(self, sample_event):
        """Verify log string format."""
        log_str = sample_event.to_log_string()

        assert "CREDENTIAL_ACCESS" in log_str
        assert "SUCCESS" in log_str
        assert "user_001" in log_str
        assert "alpaca" in log_str
        assert "decrypt" in log_str

    def test_failed_event_log_string(self):
        """Verify failed event log string shows FAILED."""
        event = CredentialAccessEvent(
            timestamp=datetime.now(timezone.utc),
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="test",
            success=False,
            error_message="Auth failed",
        )
        log_str = event.to_log_string()
        assert "FAILED" in log_str


class TestInMemoryAuditStorage:
    """Tests for InMemoryAuditStorage."""

    @pytest.fixture
    def storage(self):
        """Create storage instance."""
        return InMemoryAuditStorage()

    @pytest.fixture
    def sample_event(self):
        """Create a sample event."""
        return CredentialAccessEvent(
            timestamp=datetime.now(timezone.utc),
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="test",
            success=True,
        )

    def test_append_and_query(self, storage, sample_event):
        """Verify event can be appended and queried."""
        storage.append(sample_event)
        events = storage.query(user_id="user_001")
        assert len(events) == 1
        assert events[0].user_id == "user_001"

    def test_query_by_credential_id(self, storage, sample_event):
        """Verify query by credential_id works."""
        storage.append(sample_event)
        events = storage.query(credential_id="cred_123")
        assert len(events) == 1

    def test_query_by_since(self, storage):
        """Verify query by since timestamp works."""
        old_event = CredentialAccessEvent(
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="old",
            success=True,
        )
        new_event = CredentialAccessEvent(
            timestamp=datetime.now(timezone.utc),
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="new",
            success=True,
        )

        storage.append(old_event)
        storage.append(new_event)

        events = storage.query(since=datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert len(events) == 1
        assert events[0].purpose == "new"

    def test_query_by_access_types(self, storage):
        """Verify query by access types works."""
        for access_type in [CredentialAccessType.CREATE, CredentialAccessType.DECRYPT]:
            event = CredentialAccessEvent(
                timestamp=datetime.now(timezone.utc),
                user_id="user_001",
                credential_id="cred_123",
                broker="alpaca",
                access_type=access_type,
                purpose="test",
                success=True,
            )
            storage.append(event)

        events = storage.query(access_types=[CredentialAccessType.CREATE])
        assert len(events) == 1
        assert events[0].access_type == CredentialAccessType.CREATE

    def test_query_success_only(self, storage):
        """Verify success_only filter works."""
        for success in [True, False]:
            event = CredentialAccessEvent(
                timestamp=datetime.now(timezone.utc),
                user_id="user_001",
                credential_id="cred_123",
                broker="alpaca",
                access_type=CredentialAccessType.DECRYPT,
                purpose="test",
                success=success,
            )
            storage.append(event)

        events = storage.query(success_only=True)
        assert len(events) == 1
        assert events[0].success is True

    def test_query_limit(self, storage):
        """Verify query limit works."""
        for i in range(10):
            event = CredentialAccessEvent(
                timestamp=datetime.now(timezone.utc),
                user_id="user_001",
                credential_id="cred_123",
                broker="alpaca",
                access_type=CredentialAccessType.DECRYPT,
                purpose=f"test_{i}",
                success=True,
            )
            storage.append(event)

        events = storage.query(limit=5)
        assert len(events) == 5

    def test_delete_before(self, storage):
        """Verify delete_before removes old events."""
        old_event = CredentialAccessEvent(
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="old",
            success=True,
        )
        new_event = CredentialAccessEvent(
            timestamp=datetime.now(timezone.utc),
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="new",
            success=True,
        )

        storage.append(old_event)
        storage.append(new_event)

        deleted = storage.delete_before(datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert deleted == 1

        events = storage.query()
        assert len(events) == 1
        assert events[0].purpose == "new"

    def test_max_events_limit(self):
        """Verify max_events limit is enforced."""
        storage = InMemoryAuditStorage(max_events=100)

        for i in range(150):
            event = CredentialAccessEvent(
                timestamp=datetime.now(timezone.utc),
                user_id="user_001",
                credential_id="cred_123",
                broker="alpaca",
                access_type=CredentialAccessType.DECRYPT,
                purpose=f"test_{i}",
                success=True,
            )
            storage.append(event)

        events = storage.query()
        assert len(events) == 100


class TestCredentialAuditLogger:
    """Tests for CredentialAuditLogger."""

    @pytest.fixture
    def storage(self):
        """Create storage instance."""
        return InMemoryAuditStorage()

    @pytest.fixture
    def logger(self, storage):
        """Create logger instance."""
        return CredentialAuditLogger(storage)

    def test_log_access(self, logger, storage):
        """Verify access is logged."""
        event = logger.log_access(
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="order_execution",
            source_ip="127.0.0.1",
            success=True,
        )

        assert event is not None
        assert event.user_id == "user_001"

        events = storage.query(user_id="user_001")
        assert len(events) == 1

    def test_log_access_with_all_fields(self, logger):
        """Verify all fields are captured."""
        event = logger.log_access(
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="order_execution",
            source_ip="127.0.0.1",
            user_agent="Mozilla/5.0",
            success=True,
            request_id="req_123",
            metadata={"order_id": "ord_456"},
        )

        assert event.source_ip == "127.0.0.1"
        assert event.user_agent == "Mozilla/5.0"
        assert event.request_id == "req_123"
        assert event.metadata["order_id"] == "ord_456"

    def test_log_failed_access(self, logger, storage):
        """Verify failed access is logged."""
        event = logger.log_access(
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="test",
            success=False,
            error_message="Invalid key",
        )

        assert event.success is False
        assert event.error_message == "Invalid key"


class TestAnomalyDetection:
    """Tests for anomaly detection."""

    @pytest.fixture
    def storage(self):
        """Create storage instance."""
        return InMemoryAuditStorage()

    @pytest.fixture
    def logger(self, storage):
        """Create logger instance."""
        return CredentialAuditLogger(storage)

    def test_detect_high_volume(self, logger, storage):
        """Verify high volume anomaly is detected."""
        # Create more than threshold events
        for i in range(1100):
            event = CredentialAccessEvent(
                timestamp=datetime.now(timezone.utc),
                user_id="user_001",
                credential_id="cred_123",
                broker="alpaca",
                access_type=CredentialAccessType.DECRYPT,
                purpose=f"test_{i}",
                success=True,
            )
            storage.append(event)

        anomalies = logger.detect_anomalies("user_001")
        high_volume = [a for a in anomalies if a.alert_type == "high_volume"]
        assert len(high_volume) == 1
        assert high_volume[0].severity == "medium"

    def test_detect_repeated_failures(self, logger, storage):
        """Verify repeated failures anomaly is detected."""
        # Create more than threshold failures in last hour
        for i in range(15):
            event = CredentialAccessEvent(
                timestamp=datetime.now(timezone.utc),
                user_id="user_001",
                credential_id="cred_123",
                broker="alpaca",
                access_type=CredentialAccessType.DECRYPT,
                purpose="test",
                success=False,
                error_message="Auth failed",
            )
            storage.append(event)

        anomalies = logger.detect_anomalies("user_001")
        failures = [a for a in anomalies if a.alert_type == "repeated_failures"]
        assert len(failures) == 1
        assert failures[0].severity == "high"

    def test_detect_multiple_ips(self, logger, storage):
        """Verify multiple IPs anomaly is detected."""
        # Access from more than threshold IPs in last hour
        for i in range(10):
            event = CredentialAccessEvent(
                timestamp=datetime.now(timezone.utc),
                user_id="user_001",
                credential_id="cred_123",
                broker="alpaca",
                access_type=CredentialAccessType.DECRYPT,
                purpose="test",
                source_ip=f"192.168.1.{i}",
                success=True,
            )
            storage.append(event)

        anomalies = logger.detect_anomalies("user_001")
        multi_ip = [a for a in anomalies if a.alert_type == "multiple_ips"]
        assert len(multi_ip) == 1
        assert multi_ip[0].severity == "medium"

    def test_no_anomalies_normal_usage(self, logger, storage):
        """Verify no anomalies for normal usage."""
        # Create a few successful events
        for i in range(5):
            event = CredentialAccessEvent(
                timestamp=datetime.now(timezone.utc),
                user_id="user_001",
                credential_id="cred_123",
                broker="alpaca",
                access_type=CredentialAccessType.DECRYPT,
                purpose="test",
                source_ip="127.0.0.1",
                success=True,
            )
            storage.append(event)

        anomalies = logger.detect_anomalies("user_001")
        assert len(anomalies) == 0


class TestUserAccessHistory:
    """Tests for user access history queries."""

    @pytest.fixture
    def storage(self):
        """Create storage instance."""
        return InMemoryAuditStorage()

    @pytest.fixture
    def logger(self, storage):
        """Create logger instance."""
        return CredentialAuditLogger(storage)

    def test_get_user_access_history(self, logger, storage):
        """Verify user access history is returned."""
        for i in range(5):
            logger.log_access(
                user_id="user_001",
                credential_id="cred_123",
                broker="alpaca",
                access_type=CredentialAccessType.DECRYPT,
                purpose=f"test_{i}",
                success=True,
            )

        history = logger.get_user_access_history("user_001", days=30)
        assert len(history) == 5

    def test_get_credential_access_history(self, logger, storage):
        """Verify credential access history is returned."""
        for user in ["user_001", "user_002"]:
            logger.log_access(
                user_id=user,
                credential_id="shared_cred",
                broker="alpaca",
                access_type=CredentialAccessType.DECRYPT,
                purpose="test",
                success=True,
            )

        history = logger.get_credential_access_history("shared_cred", days=30)
        assert len(history) == 2


class TestRetentionPolicy:
    """Tests for retention policy."""

    @pytest.fixture
    def storage(self):
        """Create storage instance."""
        return InMemoryAuditStorage()

    @pytest.fixture
    def logger(self, storage):
        """Create logger instance."""
        return CredentialAuditLogger(storage)

    def test_apply_retention_policy(self, logger, storage):
        """Verify retention policy deletes old events."""
        # Add old event
        old_event = CredentialAccessEvent(
            timestamp=datetime.now(timezone.utc) - timedelta(days=400),
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="old",
            success=True,
        )
        storage.append(old_event)

        # Add recent event
        logger.log_access(
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="new",
            success=True,
        )

        deleted = logger.apply_retention_policy(retention_days=365)
        assert deleted == 1

        events = storage.query()
        assert len(events) == 1
        assert events[0].purpose == "new"


class TestAccessReport:
    """Tests for access report generation."""

    @pytest.fixture
    def storage(self):
        """Create storage instance."""
        return InMemoryAuditStorage()

    @pytest.fixture
    def logger(self, storage):
        """Create logger instance."""
        return CredentialAuditLogger(storage)

    def test_generate_access_report(self, logger, storage):
        """Verify access report is generated correctly."""
        # Add various events
        for success in [True, True, False]:
            for broker in ["alpaca", "binance"]:
                logger.log_access(
                    user_id="user_001",
                    credential_id=f"cred_{broker}",
                    broker=broker,
                    access_type=CredentialAccessType.DECRYPT,
                    purpose="test",
                    source_ip="127.0.0.1",
                    success=success,
                )

        report = logger.generate_access_report("user_001", days=30)

        assert report["user_id"] == "user_001"
        assert "statistics" in report
        assert report["statistics"]["total_accesses"] == 6
        assert report["statistics"]["successful"] == 4
        assert report["statistics"]["failed"] == 2
        assert "access_by_type" in report
        assert "access_by_broker" in report
        assert "events" in report


class TestAnomalyAlert:
    """Tests for AnomalyAlert data class."""

    def test_alert_to_dict(self):
        """Verify alert serialization."""
        alert = AnomalyAlert(
            alert_type="high_volume",
            severity="medium",
            user_id="user_001",
            description="High access volume detected",
            timestamp=datetime.now(timezone.utc),
            evidence={"access_count": 1500},
        )

        data = alert.to_dict()

        assert data["alert_type"] == "high_volume"
        assert data["severity"] == "medium"
        assert data["user_id"] == "user_001"
        assert "evidence" in data


class TestAlertCallback:
    """Tests for alert callback functionality."""

    def test_alert_callback_called(self):
        """Verify alert callback is invoked on anomaly."""
        storage = InMemoryAuditStorage()
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        logger = CredentialAuditLogger(storage, alert_callback=callback)

        # Create conditions for anomaly (many failures)
        for i in range(15):
            logger.log_access(
                user_id="user_001",
                credential_id="cred_123",
                broker="alpaca",
                access_type=CredentialAccessType.DECRYPT,
                purpose="test",
                success=False,
                error_message="Auth failed",
            )

        # Callback should have been invoked
        assert len(alerts_received) > 0
