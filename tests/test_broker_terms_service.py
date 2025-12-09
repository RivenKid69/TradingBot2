"""
Tests for BrokerTermsService - Broker API terms compliance tracking.

Comprehensive tests covering:
    - Broker-specific warnings
    - Acknowledgment tracking
    - Version control
    - Enforcement mechanisms
    - Rate limit information

References:
    - Interactive Brokers API Agreement
    - Alpaca Platform Agreement
    - Binance API Terms of Use
"""

import pytest
from datetime import datetime, timezone

from services.broker.terms_compliance import (
    BrokerTermsService,
    SupportedBroker,
    BrokerTermsAcknowledgment,
    BrokerTermsNotAcknowledgedError,
    InMemoryBrokerTermsStorage,
    BROKER_TERMS_WARNINGS,
    BROKER_RATE_LIMITS,
    BrokerRateLimitInfo,
)


class TestSupportedBroker:
    """Tests for SupportedBroker enum."""

    def test_all_brokers_defined(self):
        """Verify expected brokers are defined."""
        expected_brokers = [
            "interactive_brokers",
            "alpaca",
            "binance",
            "coinbase",
            "kraken",
            "oanda",
            "deribit",
        ]

        broker_values = [b.value for b in SupportedBroker]
        for expected in expected_brokers:
            assert expected in broker_values, f"Missing broker: {expected}"

    def test_broker_enum_values(self):
        """Verify enum values are correct."""
        assert SupportedBroker.ALPACA.value == "alpaca"
        assert SupportedBroker.BINANCE.value == "binance"


class TestBrokerTermsWarnings:
    """Tests for broker-specific warning texts."""

    def test_all_brokers_have_warnings(self):
        """Verify all supported brokers have warning text."""
        for broker in SupportedBroker:
            warning = BROKER_TERMS_WARNINGS.get(broker, "")
            assert len(warning) > 100, f"Missing warning for {broker.value}"

    def test_interactive_brokers_warning_content(self):
        """Verify IB warning contains required elements."""
        warning = BROKER_TERMS_WARNINGS[SupportedBroker.INTERACTIVE_BROKERS].lower()

        assert "interactive brokers" in warning
        assert "api" in warning
        assert "third" in warning or "third-party" in warning
        assert "rate limit" in warning or "50" in warning

    def test_alpaca_warning_content(self):
        """Verify Alpaca warning contains required elements."""
        warning = BROKER_TERMS_WARNINGS[SupportedBroker.ALPACA].lower()

        assert "alpaca" in warning
        assert "api" in warning
        assert "paper trading" in warning or "paper" in warning
        assert "rate limit" in warning or "200" in warning

    def test_binance_warning_content(self):
        """Verify Binance warning contains required elements."""
        warning = BROKER_TERMS_WARNINGS[SupportedBroker.BINANCE].lower()

        assert "binance" in warning
        assert "api" in warning
        assert "ip" in warning  # IP whitelisting
        assert "rate limit" in warning

    def test_coinbase_warning_content(self):
        """Verify Coinbase warning contains required elements."""
        warning = BROKER_TERMS_WARNINGS[SupportedBroker.COINBASE].lower()

        assert "coinbase" in warning
        assert "api" in warning

    def test_warnings_mention_no_withdrawal(self):
        """Verify warnings mention no withdrawal access needed."""
        for broker in SupportedBroker:
            warning = BROKER_TERMS_WARNINGS.get(broker, "").lower()
            # Should mention withdrawal restrictions
            assert "withdraw" in warning or "transfer" in warning, \
                f"Warning for {broker.value} should mention withdrawal restrictions"


class TestBrokerRateLimits:
    """Tests for broker rate limit information."""

    def test_all_brokers_have_rate_limits(self):
        """Verify all supported brokers have rate limit info."""
        for broker in SupportedBroker:
            limits = BROKER_RATE_LIMITS.get(broker)
            assert limits is not None, f"Missing rate limits for {broker.value}"

    def test_rate_limit_info_structure(self):
        """Verify rate limit info has correct structure."""
        for broker in SupportedBroker:
            limits = BROKER_RATE_LIMITS.get(broker)
            if limits:
                assert hasattr(limits, "orders_per_second")
                assert hasattr(limits, "api_calls_per_minute")
                assert hasattr(limits, "description")
                assert hasattr(limits, "documentation_url")

    def test_alpaca_rate_limits(self):
        """Verify Alpaca rate limits are reasonable."""
        limits = BROKER_RATE_LIMITS[SupportedBroker.ALPACA]
        assert limits.api_calls_per_minute == 200
        assert limits.orders_per_second > 0
        assert "alpaca" in limits.documentation_url.lower()

    def test_binance_rate_limits(self):
        """Verify Binance rate limits are reasonable."""
        limits = BROKER_RATE_LIMITS[SupportedBroker.BINANCE]
        assert limits.orders_per_second == 10
        assert limits.api_calls_per_minute > 0


class TestInMemoryBrokerTermsStorage:
    """Tests for in-memory storage implementation."""

    @pytest.fixture
    def storage(self):
        """Create fresh storage instance."""
        return InMemoryBrokerTermsStorage()

    def test_save_and_get_latest(self, storage):
        """Verify acknowledgment can be saved and retrieved."""
        ack = BrokerTermsAcknowledgment(
            acknowledgment_id="ack_001",
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            terms_version="2024.1",
            acknowledged_at=datetime.now(timezone.utc),
            ip_address="127.0.0.1"
        )
        storage.save(ack)

        retrieved = storage.get_latest("user_001", SupportedBroker.ALPACA)
        assert retrieved is not None
        assert retrieved.acknowledgment_id == "ack_001"

    def test_get_latest_returns_most_recent(self, storage):
        """Verify get_latest returns most recent acknowledgment."""
        ack1 = BrokerTermsAcknowledgment(
            acknowledgment_id="ack_001",
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            terms_version="2023.1",
            acknowledged_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ip_address="127.0.0.1"
        )
        ack2 = BrokerTermsAcknowledgment(
            acknowledgment_id="ack_002",
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            terms_version="2024.1",
            acknowledged_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ip_address="127.0.0.1"
        )
        storage.save(ack1)
        storage.save(ack2)

        retrieved = storage.get_latest("user_001", SupportedBroker.ALPACA)
        assert retrieved.acknowledgment_id == "ack_002"

    def test_get_latest_returns_none_for_unknown(self, storage):
        """Verify get_latest returns None for unknown user/broker."""
        retrieved = storage.get_latest("unknown", SupportedBroker.ALPACA)
        assert retrieved is None

    def test_get_all_for_user(self, storage):
        """Verify get_all_for_user returns all acknowledgments."""
        for broker in [SupportedBroker.ALPACA, SupportedBroker.BINANCE]:
            ack = BrokerTermsAcknowledgment(
                acknowledgment_id=f"ack_{broker.value}",
                user_id="user_001",
                broker=broker,
                terms_version="2024.1",
                acknowledged_at=datetime.now(timezone.utc),
                ip_address="127.0.0.1"
            )
            storage.save(ack)

        all_acks = storage.get_all_for_user("user_001")
        assert len(all_acks) == 2

    def test_delete_for_user(self, storage):
        """Verify delete_for_user removes all user acknowledgments."""
        ack = BrokerTermsAcknowledgment(
            acknowledgment_id="ack_001",
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            terms_version="2024.1",
            acknowledged_at=datetime.now(timezone.utc),
            ip_address="127.0.0.1"
        )
        storage.save(ack)

        deleted = storage.delete_for_user("user_001")
        assert deleted == 1
        assert len(storage.get_all_for_user("user_001")) == 0


class TestBrokerTermsService:
    """Tests for BrokerTermsService functionality."""

    @pytest.fixture
    def service(self):
        """Create service with in-memory storage."""
        storage = InMemoryBrokerTermsStorage()
        return BrokerTermsService(storage)

    def test_get_supported_brokers(self, service):
        """Verify get_supported_brokers returns all brokers."""
        brokers = service.get_supported_brokers()
        assert len(brokers) == len(SupportedBroker)

    def test_get_broker_warning(self, service):
        """Verify get_broker_warning returns warning text."""
        warning = service.get_broker_warning(SupportedBroker.ALPACA)
        assert len(warning) > 100
        assert "alpaca" in warning.lower()

    def test_get_broker_rate_limits(self, service):
        """Verify get_broker_rate_limits returns rate limit info."""
        limits = service.get_broker_rate_limits(SupportedBroker.ALPACA)
        assert limits is not None
        assert limits.api_calls_per_minute == 200

    def test_get_current_version(self, service):
        """Verify get_current_version returns version string."""
        version = service.get_current_version(SupportedBroker.ALPACA)
        assert version is not None
        assert "." in version or len(version) > 0

    def test_record_acknowledgment(self, service):
        """Verify acknowledgment is recorded correctly."""
        ack = service.record_acknowledgment(
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            ip_address="127.0.0.1"
        )

        assert ack is not None
        assert ack.user_id == "user_001"
        assert ack.broker == SupportedBroker.ALPACA
        assert ack.ip_address == "127.0.0.1"

    def test_has_valid_acknowledgment_after_record(self, service):
        """Verify has_valid_acknowledgment returns True after recording."""
        service.record_acknowledgment(
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            ip_address="127.0.0.1"
        )

        assert service.has_valid_acknowledgment("user_001", SupportedBroker.ALPACA)

    def test_fresh_user_has_no_acknowledgment(self, service):
        """Verify fresh user has no acknowledgments."""
        assert not service.has_valid_acknowledgment("new_user", SupportedBroker.ALPACA)

    def test_version_change_requires_reacknowledgment(self, service):
        """Verify version change invalidates previous acknowledgment."""
        service.record_acknowledgment(
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            ip_address="127.0.0.1"
        )

        # Change the version
        service.update_terms_version(SupportedBroker.ALPACA, "2025.1")

        # Should now require re-acknowledgment
        assert not service.has_valid_acknowledgment("user_001", SupportedBroker.ALPACA)

    def test_require_acknowledgment_before_key_submission_raises(self, service):
        """Verify require_acknowledgment raises when not acknowledged."""
        with pytest.raises(BrokerTermsNotAcknowledgedError) as exc_info:
            service.require_acknowledgment_before_key_submission(
                "new_user", SupportedBroker.ALPACA
            )

        assert exc_info.value.broker == SupportedBroker.ALPACA

    def test_require_acknowledgment_before_key_submission_passes(self, service):
        """Verify require_acknowledgment passes when acknowledged."""
        service.record_acknowledgment(
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            ip_address="127.0.0.1"
        )

        # Should not raise
        service.require_acknowledgment_before_key_submission(
            "user_001", SupportedBroker.ALPACA
        )

    def test_get_pending_acknowledgments(self, service):
        """Verify get_pending_acknowledgments returns correct list."""
        service.record_acknowledgment(
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            ip_address="127.0.0.1"
        )

        pending = service.get_pending_acknowledgments(
            "user_001",
            brokers=[SupportedBroker.ALPACA, SupportedBroker.BINANCE]
        )

        assert SupportedBroker.BINANCE in pending
        assert SupportedBroker.ALPACA not in pending

    def test_get_user_acknowledgments(self, service):
        """Verify get_user_acknowledgments returns all user's acknowledgments."""
        for broker in [SupportedBroker.ALPACA, SupportedBroker.BINANCE]:
            service.record_acknowledgment(
                user_id="user_001",
                broker=broker,
                ip_address="127.0.0.1"
            )

        acks = service.get_user_acknowledgments("user_001")
        assert len(acks) == 2

    def test_delete_user_acknowledgments(self, service):
        """Verify delete_user_acknowledgments removes all user data."""
        service.record_acknowledgment(
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            ip_address="127.0.0.1"
        )

        deleted = service.delete_user_acknowledgments("user_001")
        assert deleted == 1
        assert not service.has_valid_acknowledgment("user_001", SupportedBroker.ALPACA)

    def test_get_acknowledgment_status(self, service):
        """Verify get_acknowledgment_status generates correct report."""
        service.record_acknowledgment(
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            ip_address="127.0.0.1"
        )

        status = service.get_acknowledgment_status("user_001")

        assert status["user_id"] == "user_001"
        assert "brokers" in status
        assert SupportedBroker.ALPACA.value in status["brokers"]

        alpaca_status = status["brokers"][SupportedBroker.ALPACA.value]
        assert alpaca_status["acknowledged"] is True
        assert alpaca_status["is_current"] is True


class TestAcknowledgmentSerialization:
    """Tests for acknowledgment serialization."""

    def test_to_dict_from_dict_roundtrip(self):
        """Verify acknowledgment can be serialized and deserialized."""
        ack = BrokerTermsAcknowledgment(
            acknowledgment_id="ack_001",
            user_id="user_001",
            broker=SupportedBroker.ALPACA,
            terms_version="2024.1",
            acknowledged_at=datetime.now(timezone.utc),
            ip_address="127.0.0.1",
            metadata={"extra": "data"}
        )

        data = ack.to_dict()
        restored = BrokerTermsAcknowledgment.from_dict(data)

        assert restored.acknowledgment_id == ack.acknowledgment_id
        assert restored.user_id == ack.user_id
        assert restored.broker == ack.broker
        assert restored.terms_version == ack.terms_version
        assert restored.ip_address == ack.ip_address
        assert restored.metadata == ack.metadata


class TestBrokerTermsNotAcknowledgedError:
    """Tests for BrokerTermsNotAcknowledgedError exception."""

    def test_exception_has_broker(self):
        """Verify exception contains broker."""
        error = BrokerTermsNotAcknowledgedError(SupportedBroker.ALPACA)
        assert error.broker == SupportedBroker.ALPACA

    def test_exception_has_message(self):
        """Verify exception has readable message."""
        error = BrokerTermsNotAcknowledgedError(SupportedBroker.ALPACA)
        assert "alpaca" in str(error).lower()

    def test_custom_message(self):
        """Verify custom message is used."""
        error = BrokerTermsNotAcknowledgedError(
            SupportedBroker.ALPACA,
            "Custom error message"
        )
        assert error.message == "Custom error message"


class TestAllBrokersIntegration:
    """Integration tests for all brokers."""

    @pytest.fixture
    def service(self):
        """Create service with in-memory storage."""
        storage = InMemoryBrokerTermsStorage()
        return BrokerTermsService(storage)

    def test_all_brokers_can_be_acknowledged(self, service):
        """Verify all brokers can be acknowledged."""
        for broker in SupportedBroker:
            ack = service.record_acknowledgment(
                user_id="user_001",
                broker=broker,
                ip_address="127.0.0.1"
            )
            assert ack is not None
            assert service.has_valid_acknowledgment("user_001", broker)

    def test_all_brokers_have_warnings_and_limits(self, service):
        """Verify all brokers have both warnings and rate limits."""
        for broker in SupportedBroker:
            warning = service.get_broker_warning(broker)
            assert len(warning) > 0, f"No warning for {broker.value}"

            limits = service.get_broker_rate_limits(broker)
            assert limits is not None, f"No rate limits for {broker.value}"
