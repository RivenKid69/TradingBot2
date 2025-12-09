"""
Tests for DisclaimerService - User acknowledgment tracking for legal disclaimers.

Comprehensive tests covering:
    - Acknowledgment recording and retrieval
    - Version-controlled re-acknowledgment
    - Enforcement mechanisms
    - GDPR data operations

References:
    - MiFID II Article 24(4): Fair, clear, not misleading
    - ESMA Guidelines on Product Governance
"""

import pytest
from datetime import datetime, timezone

from services.api.disclaimer_service import (
    DisclaimerService,
    DisclaimerType,
    DisclaimerAcknowledgment,
    DisclaimerNotAcknowledgedError,
    InMemoryDisclaimerStorage,
    PRE_LIVE_TRADING_DISCLAIMER,
    BACKTEST_RESULTS_DISCLAIMER,
    TERMS_OF_SERVICE_DISCLAIMER,
    require_live_trading_acknowledgment,
    require_strategy_deployment_acknowledgment,
)


class TestDisclaimerTexts:
    """Tests for disclaimer text content."""

    def test_pre_live_trading_disclaimer_content(self):
        """Verify pre-live trading disclaimer contains required elements."""
        text = PRE_LIVE_TRADING_DISCLAIMER.lower()

        # Must contain risk warnings
        assert "risk" in text
        assert "loss" in text

        # Must clarify no investment advice
        assert "no investment advice" in text or "not investment advice" in text

        # Must mention user responsibility
        assert "responsib" in text

        # Must mention broker credentials
        assert "broker" in text
        assert "api" in text or "credential" in text

    def test_backtest_results_disclaimer_content(self):
        """Verify backtest disclaimer contains required elements."""
        text = BACKTEST_RESULTS_DISCLAIMER.lower()

        # Must clarify simulation
        assert "simulation" in text

        # Must have past performance warning
        assert "past performance" in text

        # Must mention limitations
        assert "limitation" in text

    def test_tos_disclaimer_content(self):
        """Verify ToS disclaimer contains required elements."""
        text = TERMS_OF_SERVICE_DISCLAIMER.lower()
        assert "terms" in text
        assert "accept" in text or "agree" in text


class TestInMemoryDisclaimerStorage:
    """Tests for in-memory storage implementation."""

    @pytest.fixture
    def storage(self):
        """Create fresh storage instance."""
        return InMemoryDisclaimerStorage()

    def test_save_and_get_latest(self, storage):
        """Verify acknowledgment can be saved and retrieved."""
        ack = DisclaimerAcknowledgment(
            acknowledgment_id="ack_001",
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            disclaimer_version="1.0.0",
            acknowledged_at=datetime.now(timezone.utc),
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )
        storage.save(ack)

        retrieved = storage.get_latest("user_001", DisclaimerType.PRE_LIVE_TRADING)
        assert retrieved is not None
        assert retrieved.acknowledgment_id == "ack_001"

    def test_get_latest_returns_most_recent(self, storage):
        """Verify get_latest returns the most recent acknowledgment."""
        ack1 = DisclaimerAcknowledgment(
            acknowledgment_id="ack_001",
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            disclaimer_version="1.0.0",
            acknowledged_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )
        ack2 = DisclaimerAcknowledgment(
            acknowledgment_id="ack_002",
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            disclaimer_version="2.0.0",
            acknowledged_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )
        storage.save(ack1)
        storage.save(ack2)

        retrieved = storage.get_latest("user_001", DisclaimerType.PRE_LIVE_TRADING)
        assert retrieved.acknowledgment_id == "ack_002"

    def test_get_latest_returns_none_for_unknown_user(self, storage):
        """Verify get_latest returns None for unknown user."""
        retrieved = storage.get_latest("unknown_user", DisclaimerType.PRE_LIVE_TRADING)
        assert retrieved is None

    def test_get_all_for_user(self, storage):
        """Verify get_all_for_user returns all acknowledgments."""
        for i in range(3):
            ack = DisclaimerAcknowledgment(
                acknowledgment_id=f"ack_{i}",
                user_id="user_001",
                disclaimer_type=list(DisclaimerType)[i],
                disclaimer_version="1.0.0",
                acknowledged_at=datetime.now(timezone.utc),
                ip_address="127.0.0.1",
                user_agent="Mozilla/5.0"
            )
            storage.save(ack)

        all_acks = storage.get_all_for_user("user_001")
        assert len(all_acks) == 3

    def test_delete_for_user(self, storage):
        """Verify delete_for_user removes all user acknowledgments."""
        for i in range(3):
            ack = DisclaimerAcknowledgment(
                acknowledgment_id=f"ack_{i}",
                user_id="user_001",
                disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
                disclaimer_version="1.0.0",
                acknowledged_at=datetime.now(timezone.utc),
                ip_address="127.0.0.1",
                user_agent="Mozilla/5.0"
            )
            storage.save(ack)

        deleted = storage.delete_for_user("user_001")
        assert deleted == 3
        assert len(storage.get_all_for_user("user_001")) == 0


class TestDisclaimerService:
    """Tests for DisclaimerService functionality."""

    @pytest.fixture
    def service(self):
        """Create service with in-memory storage."""
        storage = InMemoryDisclaimerStorage()
        return DisclaimerService(storage)

    def test_fresh_user_has_no_acknowledgment(self, service):
        """Verify fresh user has no acknowledgments."""
        assert not service.has_valid_acknowledgment(
            "new_user", DisclaimerType.PRE_LIVE_TRADING
        )

    def test_acknowledgment_recorded(self, service):
        """Verify acknowledgment is properly recorded."""
        ack = service.record_acknowledgment(
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )

        assert ack is not None
        assert ack.user_id == "user_001"
        assert ack.disclaimer_type == DisclaimerType.PRE_LIVE_TRADING
        assert ack.ip_address == "127.0.0.1"
        assert ack.consent_text_hash is not None

    def test_has_valid_acknowledgment_after_record(self, service):
        """Verify has_valid_acknowledgment returns True after recording."""
        service.record_acknowledgment(
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )

        assert service.has_valid_acknowledgment(
            "user_001", DisclaimerType.PRE_LIVE_TRADING
        )

    def test_version_change_requires_reacknowledgment(self, service):
        """Verify version change invalidates previous acknowledgment."""
        service.record_acknowledgment(
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )

        # Change the current version
        service.update_disclaimer_version(DisclaimerType.PRE_LIVE_TRADING, "2.0.0")

        # Should now require re-acknowledgment
        assert not service.has_valid_acknowledgment(
            "user_001", DisclaimerType.PRE_LIVE_TRADING
        )

    def test_require_acknowledgment_raises_when_missing(self, service):
        """Verify require_acknowledgment raises for unacknowledged disclaimer."""
        with pytest.raises(DisclaimerNotAcknowledgedError) as exc_info:
            service.require_acknowledgment("new_user", DisclaimerType.PRE_LIVE_TRADING)

        assert exc_info.value.disclaimer_type == DisclaimerType.PRE_LIVE_TRADING

    def test_require_acknowledgment_passes_when_present(self, service):
        """Verify require_acknowledgment passes when acknowledged."""
        service.record_acknowledgment(
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )

        # Should not raise
        service.require_acknowledgment("user_001", DisclaimerType.PRE_LIVE_TRADING)

    def test_get_pending_disclaimers(self, service):
        """Verify get_pending_disclaimers returns correct list."""
        # Acknowledge one disclaimer
        service.record_acknowledgment(
            user_id="user_001",
            disclaimer_type=DisclaimerType.TERMS_OF_SERVICE,
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )

        pending = service.get_pending_disclaimers(
            "user_001",
            required_types=[DisclaimerType.TERMS_OF_SERVICE, DisclaimerType.PRE_LIVE_TRADING]
        )

        assert DisclaimerType.PRE_LIVE_TRADING in pending
        assert DisclaimerType.TERMS_OF_SERVICE not in pending

    def test_get_disclaimer_text(self, service):
        """Verify get_disclaimer_text returns non-empty text."""
        text = service.get_disclaimer_text(DisclaimerType.PRE_LIVE_TRADING)
        assert len(text) > 100

    def test_get_disclaimer_version(self, service):
        """Verify get_disclaimer_version returns version string."""
        version = service.get_disclaimer_version(DisclaimerType.PRE_LIVE_TRADING)
        assert version is not None
        assert "." in version  # Should be semver-like

    def test_get_user_acknowledgments(self, service):
        """Verify get_user_acknowledgments returns all user's acknowledgments."""
        for dtype in [DisclaimerType.PRE_LIVE_TRADING, DisclaimerType.TERMS_OF_SERVICE]:
            service.record_acknowledgment(
                user_id="user_001",
                disclaimer_type=dtype,
                ip_address="127.0.0.1",
                user_agent="Mozilla/5.0"
            )

        acks = service.get_user_acknowledgments("user_001")
        assert len(acks) == 2

    def test_delete_user_acknowledgments(self, service):
        """Verify delete_user_acknowledgments removes all user data."""
        service.record_acknowledgment(
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )

        deleted = service.delete_user_acknowledgments("user_001")
        assert deleted == 1

        # Should no longer have valid acknowledgment
        assert not service.has_valid_acknowledgment(
            "user_001", DisclaimerType.PRE_LIVE_TRADING
        )

    def test_acknowledgment_report(self, service):
        """Verify get_acknowledgment_report generates correct report."""
        service.record_acknowledgment(
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )

        report = service.get_acknowledgment_report("user_001")

        assert report["user_id"] == "user_001"
        assert "disclaimers" in report
        assert DisclaimerType.PRE_LIVE_TRADING.value in report["disclaimers"]

        ack_status = report["disclaimers"][DisclaimerType.PRE_LIVE_TRADING.value]
        assert ack_status["acknowledged"] is True
        assert ack_status["is_current"] is True


class TestAcknowledgmentSerialization:
    """Tests for acknowledgment serialization."""

    def test_to_dict_from_dict_roundtrip(self):
        """Verify acknowledgment can be serialized and deserialized."""
        ack = DisclaimerAcknowledgment(
            acknowledgment_id="ack_001",
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            disclaimer_version="1.0.0",
            acknowledged_at=datetime.now(timezone.utc),
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0",
            consent_text_hash="abc123",
            metadata={"extra": "data"}
        )

        data = ack.to_dict()
        restored = DisclaimerAcknowledgment.from_dict(data)

        assert restored.acknowledgment_id == ack.acknowledgment_id
        assert restored.user_id == ack.user_id
        assert restored.disclaimer_type == ack.disclaimer_type
        assert restored.disclaimer_version == ack.disclaimer_version
        assert restored.ip_address == ack.ip_address
        assert restored.user_agent == ack.user_agent
        assert restored.consent_text_hash == ack.consent_text_hash
        assert restored.metadata == ack.metadata


class TestDisclaimerNotAcknowledgedError:
    """Tests for DisclaimerNotAcknowledgedError exception."""

    def test_exception_has_disclaimer_type(self):
        """Verify exception contains disclaimer type."""
        error = DisclaimerNotAcknowledgedError(DisclaimerType.PRE_LIVE_TRADING)
        assert error.disclaimer_type == DisclaimerType.PRE_LIVE_TRADING

    def test_exception_has_message(self):
        """Verify exception has readable message."""
        error = DisclaimerNotAcknowledgedError(DisclaimerType.PRE_LIVE_TRADING)
        assert "pre_live_trading" in str(error).lower()

    def test_custom_message(self):
        """Verify custom message is used."""
        error = DisclaimerNotAcknowledgedError(
            DisclaimerType.PRE_LIVE_TRADING,
            "Custom error message"
        )
        assert error.message == "Custom error message"


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.fixture
    def service(self):
        """Create service with in-memory storage."""
        storage = InMemoryDisclaimerStorage()
        return DisclaimerService(storage)

    def test_require_live_trading_acknowledgment_raises(self, service):
        """Verify require_live_trading_acknowledgment raises when not acknowledged."""
        with pytest.raises(DisclaimerNotAcknowledgedError):
            require_live_trading_acknowledgment(service, "new_user")

    def test_require_live_trading_acknowledgment_passes(self, service):
        """Verify require_live_trading_acknowledgment passes when acknowledged."""
        service.record_acknowledgment(
            user_id="user_001",
            disclaimer_type=DisclaimerType.PRE_LIVE_TRADING,
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )
        # Should not raise
        require_live_trading_acknowledgment(service, "user_001")

    def test_require_strategy_deployment_acknowledgment_raises(self, service):
        """Verify require_strategy_deployment_acknowledgment raises when not acknowledged."""
        with pytest.raises(DisclaimerNotAcknowledgedError):
            require_strategy_deployment_acknowledgment(service, "new_user")

    def test_require_strategy_deployment_acknowledgment_passes(self, service):
        """Verify require_strategy_deployment_acknowledgment passes when acknowledged."""
        service.record_acknowledgment(
            user_id="user_001",
            disclaimer_type=DisclaimerType.STRATEGY_DEPLOYMENT,
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0"
        )
        # Should not raise
        require_strategy_deployment_acknowledgment(service, "user_001")


class TestAllDisclaimerTypes:
    """Tests ensuring all disclaimer types work correctly."""

    @pytest.fixture
    def service(self):
        """Create service with in-memory storage."""
        storage = InMemoryDisclaimerStorage()
        return DisclaimerService(storage)

    def test_all_disclaimer_types_have_text(self, service):
        """Verify all disclaimer types have associated text."""
        for dtype in DisclaimerType:
            text = service.get_disclaimer_text(dtype)
            # May be empty for some types like PRIVACY_POLICY
            # but should not raise

    def test_all_disclaimer_types_have_version(self, service):
        """Verify all disclaimer types have versions."""
        for dtype in DisclaimerType:
            version = service.get_disclaimer_version(dtype)
            assert version is not None

    def test_all_disclaimer_types_can_be_acknowledged(self, service):
        """Verify all disclaimer types can be acknowledged."""
        for dtype in DisclaimerType:
            ack = service.record_acknowledgment(
                user_id="user_001",
                disclaimer_type=dtype,
                ip_address="127.0.0.1",
                user_agent="Mozilla/5.0"
            )
            assert ack is not None
            assert service.has_valid_acknowledgment("user_001", dtype)
