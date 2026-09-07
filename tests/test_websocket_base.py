# -*- coding: utf-8 -*-
"""
tests/test_websocket_base.py
Comprehensive tests for the async WebSocket wrapper.

Tests cover:
- WebSocketConfig validation
- ConnectionStats tracking
- WebSocketMessage creation
- AsyncWebSocket state management
- Handler implementations (Alpaca, Polygon)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from adapters.websocket_base import (
    WebSocketConfig,
    ConnectionStats,
    WebSocketMessage,
    MessageType,
    AlpacaWebSocketHandlers,
    PolygonWebSocketHandlers,
)


# =============================================================================
# WebSocketConfig Tests
# =============================================================================


class TestWebSocketConfig:
    """Tests for WebSocketConfig validation and defaults."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WebSocketConfig(url="wss://example.com")

        assert config.url == "wss://example.com"
        assert config.reconnect_enabled is True
        assert config.reconnect_max_attempts == 0  # unlimited
        assert config.reconnect_delay_initial == 1.0
        assert config.reconnect_delay_max == 60.0
        assert config.ping_interval == 30.0
        assert config.ping_timeout == 10.0
        assert config.message_queue_size == 10000
        assert config.rate_limit_per_second == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WebSocketConfig(
            url="wss://custom.com",
            api_key="test_key",
            api_secret="test_secret",
            reconnect_enabled=False,
            reconnect_max_attempts=5,
            reconnect_delay_initial=2.0,
            reconnect_delay_max=120.0,
            ping_interval=15.0,
            ping_timeout=5.0,
            message_queue_size=5000,
            rate_limit_per_second=100.0,
        )

        assert config.url == "wss://custom.com"
        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.reconnect_enabled is False
        assert config.reconnect_max_attempts == 5
        assert config.reconnect_delay_initial == 2.0
        assert config.reconnect_delay_max == 120.0
        assert config.ping_interval == 15.0
        assert config.ping_timeout == 5.0
        assert config.message_queue_size == 5000
        assert config.rate_limit_per_second == 100.0

    def test_url_required(self):
        """Test that URL is required."""
        with pytest.raises(ValueError):
            WebSocketConfig(url="")


# =============================================================================
# ConnectionStats Tests
# =============================================================================


class TestConnectionStats:
    """Tests for ConnectionStats tracking."""

    def test_initial_values(self):
        """Test initial stats values."""
        stats = ConnectionStats()

        assert stats.connected_at is None
        assert stats.disconnected_at is None
        assert stats.messages_received == 0
        assert stats.messages_sent == 0
        assert stats.bytes_received == 0
        assert stats.bytes_sent == 0
        assert stats.reconnect_count == 0
        assert stats.errors == 0
        assert stats.last_message_at is None
        assert stats.last_ping_at is None
        assert stats.last_pong_at is None
        assert stats.latency_ms is None

    def test_reset(self):
        """Test stats reset functionality."""
        stats = ConnectionStats()
        stats.messages_received = 100
        stats.messages_sent = 50
        stats.bytes_received = 10000
        stats.errors = 5

        stats.reset()

        assert stats.messages_received == 0
        assert stats.messages_sent == 0
        assert stats.bytes_received == 0
        assert stats.errors == 0

    def test_tracking_connection_times(self):
        """Test connection timestamp tracking."""
        stats = ConnectionStats()

        now = datetime.now(timezone.utc)
        stats.connected_at = now

        assert stats.connected_at == now
        assert stats.disconnected_at is None

        later = datetime.now(timezone.utc)
        stats.disconnected_at = later

        assert stats.disconnected_at == later


# =============================================================================
# WebSocketMessage Tests
# =============================================================================


class TestWebSocketMessage:
    """Tests for WebSocketMessage."""

    def test_message_creation(self):
        """Test message creation."""
        data = {"type": "trade", "symbol": "AAPL"}
        msg = WebSocketMessage(
            data=data,
            raw='{"type": "trade", "symbol": "AAPL"}',
            message_type=MessageType.TEXT,
        )

        assert msg.data == data
        assert msg.message_type == MessageType.TEXT
        assert msg.received_at is not None

    def test_message_types(self):
        """Test different message types."""
        assert MessageType.TEXT.value == "text"
        assert MessageType.BINARY.value == "binary"
        assert MessageType.PING.value == "ping"
        assert MessageType.PONG.value == "pong"
        assert MessageType.CLOSE.value == "close"
        assert MessageType.ERROR.value == "error"

    def test_message_with_sequence(self):
        """Test message with sequence number."""
        msg = WebSocketMessage(
            data={"test": "data"},
            raw='{"test": "data"}',
            sequence=42,
        )

        assert msg.sequence == 42


# =============================================================================
# AlpacaWebSocketHandlers Tests
# =============================================================================


class TestAlpacaWebSocketHandlers:
    """Tests for Alpaca WebSocket handlers."""

    def test_initialization(self):
        """Test handler initialization with callbacks."""
        on_bar_callback = MagicMock()
        on_trade_callback = MagicMock()

        handlers = AlpacaWebSocketHandlers(
            on_bar=on_bar_callback,
            on_trade=on_trade_callback,
        )

        # Callbacks should be stored
        assert handlers._on_bar == on_bar_callback
        assert handlers._on_trade == on_trade_callback

    def test_initialization_no_callbacks(self):
        """Test handler initialization without callbacks."""
        handlers = AlpacaWebSocketHandlers()

        # Should work without callbacks
        assert handlers._on_bar is None
        assert handlers._on_trade is None


# =============================================================================
# PolygonWebSocketHandlers Tests
# =============================================================================


class TestPolygonWebSocketHandlers:
    """Tests for Polygon WebSocket handlers."""

    def test_initialization(self):
        """Test handler initialization with callbacks."""
        on_bar_callback = MagicMock()
        on_trade_callback = MagicMock()

        handlers = PolygonWebSocketHandlers(
            on_bar=on_bar_callback,
            on_trade=on_trade_callback,
        )

        # Callbacks should be stored
        assert handlers._on_bar == on_bar_callback
        assert handlers._on_trade == on_trade_callback

    def test_initialization_no_callbacks(self):
        """Test handler initialization without callbacks."""
        handlers = PolygonWebSocketHandlers()

        # Should work without callbacks
        assert handlers._on_bar is None
        assert handlers._on_trade is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestWebSocketIntegration:
    """Integration tests for WebSocket components."""

    def test_alpaca_handlers_callbacks(self):
        """Test Alpaca handlers with callback flow."""
        received_bars = []

        def on_bar(bar_data):
            received_bars.append(bar_data)

        handlers = AlpacaWebSocketHandlers(on_bar=on_bar)

        # Verify handler is ready
        assert handlers._on_bar is not None
        assert callable(handlers._on_bar)

    def test_polygon_handlers_callbacks(self):
        """Test Polygon handlers with callback flow."""
        received_bars = []

        def on_bar(bar_data):
            received_bars.append(bar_data)

        handlers = PolygonWebSocketHandlers(on_bar=on_bar)

        # Verify handler is ready
        assert handlers._on_bar is not None
        assert callable(handlers._on_bar)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = WebSocketConfig(
            url="wss://stream.example.com",
            reconnect_delay_max=30.0,
            ping_interval=15.0,
        )

        assert config.reconnect_delay_max == 30.0
        assert config.ping_interval == 15.0

        # Invalid config (empty URL)
        with pytest.raises(ValueError):
            WebSocketConfig(url="")

    def test_stats_lifecycle(self):
        """Test connection stats lifecycle."""
        stats = ConnectionStats()

        # Simulate connection
        stats.connected_at = datetime.now(timezone.utc)
        stats.messages_received = 100
        stats.bytes_received = 50000

        assert stats.messages_received == 100

        # Simulate disconnect
        stats.disconnected_at = datetime.now(timezone.utc)
        stats.errors = 1

        assert stats.errors == 1

        # Reset counters
        stats.reset()
        assert stats.messages_received == 0
        assert stats.errors == 0
        # Timestamps preserved
        assert stats.connected_at is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
