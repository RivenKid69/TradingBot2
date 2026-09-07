# -*- coding: utf-8 -*-
"""
adapters/deribit/websocket.py

Deribit WebSocket Client for Real-Time Streaming.

This module provides WebSocket connectivity for Deribit options,
supporting both public and private channels with automatic
reconnection and heartbeat management.

Features:
    - Real-time ticker/quote updates
    - Order book streaming
    - Trade feed
    - User order updates (authenticated)
    - Position updates (authenticated)
    - DVOL streaming

Deribit WebSocket Protocol:
    - JSON-RPC 2.0 over WebSocket
    - Heartbeat every 10 seconds (configurable)
    - Automatic ping/pong for connection health
    - Subscription-based channels

Channel Naming:
    - ticker.{instrument_name}.{interval}  (e.g., "ticker.BTC-28MAR25-100000-C.100ms")
    - book.{instrument_name}.{group}.{depth}.{interval}
    - trades.{instrument_name}.{interval}
    - deribit_price_index.{index_name}
    - deribit_volatility_index.{index_name}

References:
    - Deribit WebSocket API: https://docs.deribit.com/#websocket-api
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

# Optional async WebSocket library
try:
    import websockets
    from websockets.client import WebSocketClientProtocol

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    websockets = None
    WebSocketClientProtocol = None


# =============================================================================
# Constants
# =============================================================================

DERIBIT_WS_URL = "wss://www.deribit.com/ws/api/v2"
DERIBIT_TESTNET_WS_URL = "wss://test.deribit.com/ws/api/v2"

# Heartbeat interval (seconds)
HEARTBEAT_INTERVAL = 10

# Reconnection settings
RECONNECT_DELAY_INITIAL = 1.0
RECONNECT_DELAY_MAX = 60.0
RECONNECT_BACKOFF_MULTIPLIER = 2.0

# Default subscription intervals
DEFAULT_TICKER_INTERVAL = "100ms"
DEFAULT_BOOK_INTERVAL = "100ms"
DEFAULT_TRADES_INTERVAL = "100ms"


# =============================================================================
# Enums
# =============================================================================


class DeribitChannelType(str, Enum):
    """Types of Deribit subscription channels."""

    TICKER = "ticker"
    BOOK = "book"
    TRADES = "trades"
    PRICE_INDEX = "deribit_price_index"
    VOLATILITY_INDEX = "deribit_volatility_index"
    USER_ORDERS = "user.orders"
    USER_TRADES = "user.trades"
    USER_PORTFOLIO = "user.portfolio"
    USER_CHANGES = "user.changes"


class ConnectionState(str, Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DeribitStreamConfig:
    """
    Configuration for Deribit WebSocket streaming.
    """

    testnet: bool = True
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    heartbeat_interval: float = HEARTBEAT_INTERVAL
    reconnect_enabled: bool = True
    reconnect_delay_initial: float = RECONNECT_DELAY_INITIAL
    reconnect_delay_max: float = RECONNECT_DELAY_MAX
    auto_subscribe_on_reconnect: bool = True

    @property
    def ws_url(self) -> str:
        """Get WebSocket URL based on configuration."""
        return DERIBIT_TESTNET_WS_URL if self.testnet else DERIBIT_WS_URL

    @property
    def requires_auth(self) -> bool:
        """Check if authentication is configured."""
        return bool(self.client_id and self.client_secret)


@dataclass
class DeribitSubscription:
    """
    Represents a subscription to a Deribit channel.
    """

    channel: str  # Full channel name
    channel_type: DeribitChannelType
    instrument_name: Optional[str] = None
    interval: str = "100ms"
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
    is_private: bool = False

    @classmethod
    def ticker(
        cls,
        instrument_name: str,
        interval: str = DEFAULT_TICKER_INTERVAL,
        callback: Optional[Callable] = None,
    ) -> "DeribitSubscription":
        """Create ticker subscription."""
        return cls(
            channel=f"ticker.{instrument_name}.{interval}",
            channel_type=DeribitChannelType.TICKER,
            instrument_name=instrument_name,
            interval=interval,
            callback=callback,
        )

    @classmethod
    def book(
        cls,
        instrument_name: str,
        group: str = "none",
        depth: int = 10,
        interval: str = DEFAULT_BOOK_INTERVAL,
        callback: Optional[Callable] = None,
    ) -> "DeribitSubscription":
        """Create order book subscription."""
        return cls(
            channel=f"book.{instrument_name}.{group}.{depth}.{interval}",
            channel_type=DeribitChannelType.BOOK,
            instrument_name=instrument_name,
            interval=interval,
            callback=callback,
        )

    @classmethod
    def trades(
        cls,
        instrument_name: str,
        interval: str = DEFAULT_TRADES_INTERVAL,
        callback: Optional[Callable] = None,
    ) -> "DeribitSubscription":
        """Create trades subscription."""
        return cls(
            channel=f"trades.{instrument_name}.{interval}",
            channel_type=DeribitChannelType.TRADES,
            instrument_name=instrument_name,
            interval=interval,
            callback=callback,
        )

    @classmethod
    def price_index(
        cls,
        currency: str,
        callback: Optional[Callable] = None,
    ) -> "DeribitSubscription":
        """Create price index subscription."""
        index_name = f"{currency.lower()}_usd"
        return cls(
            channel=f"deribit_price_index.{index_name}",
            channel_type=DeribitChannelType.PRICE_INDEX,
            callback=callback,
        )

    @classmethod
    def volatility_index(
        cls,
        currency: str,
        callback: Optional[Callable] = None,
    ) -> "DeribitSubscription":
        """Create volatility index (DVOL) subscription."""
        index_name = f"{currency.lower()}_usd"
        return cls(
            channel=f"deribit_volatility_index.{index_name}",
            channel_type=DeribitChannelType.VOLATILITY_INDEX,
            callback=callback,
        )

    @classmethod
    def user_orders(
        cls,
        currency: str,
        callback: Optional[Callable] = None,
    ) -> "DeribitSubscription":
        """Create user orders subscription (requires auth)."""
        return cls(
            channel=f"user.orders.{currency.upper()}.raw",
            channel_type=DeribitChannelType.USER_ORDERS,
            callback=callback,
            is_private=True,
        )

    @classmethod
    def user_portfolio(
        cls,
        currency: str,
        callback: Optional[Callable] = None,
    ) -> "DeribitSubscription":
        """Create user portfolio subscription (requires auth)."""
        return cls(
            channel=f"user.portfolio.{currency.upper()}",
            channel_type=DeribitChannelType.USER_PORTFOLIO,
            callback=callback,
            is_private=True,
        )


@dataclass
class DeribitMessage:
    """
    Parsed message from Deribit WebSocket.
    """

    channel: Optional[str]
    channel_type: Optional[DeribitChannelType]
    data: Dict[str, Any]
    timestamp_ms: int
    raw: Dict[str, Any]

    @classmethod
    def from_notification(cls, msg: Dict[str, Any]) -> "DeribitMessage":
        """Parse a subscription notification message."""
        params = msg.get("params", {})
        channel = params.get("channel", "")
        data = params.get("data", {})

        # Determine channel type from channel name
        channel_type = None
        for ct in DeribitChannelType:
            if channel.startswith(ct.value):
                channel_type = ct
                break

        return cls(
            channel=channel,
            channel_type=channel_type,
            data=data,
            timestamp_ms=data.get("timestamp", int(time.time() * 1000)),
            raw=msg,
        )

    @classmethod
    def from_response(cls, msg: Dict[str, Any]) -> "DeribitMessage":
        """Parse a JSON-RPC response message."""
        return cls(
            channel=None,
            channel_type=None,
            data=msg.get("result", {}),
            timestamp_ms=int(time.time() * 1000),
            raw=msg,
        )


# =============================================================================
# WebSocket Client
# =============================================================================


class DeribitWebSocketClient:
    """
    Production-grade WebSocket client for Deribit.

    Features:
        - Automatic connection management
        - Authentication for private channels
        - Heartbeat for connection health
        - Automatic reconnection with backoff
        - Subscription management
        - Message callbacks

    Example:
        >>> async def on_ticker(msg):
        ...     print(f"Ticker: {msg['data']}")
        ...
        >>> client = DeribitWebSocketClient(DeribitStreamConfig(testnet=True))
        >>> await client.connect()
        >>> await client.subscribe(DeribitSubscription.ticker("BTC-28MAR25-100000-C", callback=on_ticker))
        >>> await client.run_forever()
    """

    def __init__(self, config: DeribitStreamConfig):
        """
        Initialize WebSocket client.

        Args:
            config: Stream configuration
        """
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets library required. Install with: pip install websockets")

        self._config = config
        self._ws: Optional[WebSocketClientProtocol] = None
        self._state = ConnectionState.DISCONNECTED
        self._subscriptions: Dict[str, DeribitSubscription] = {}
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._request_id = 0
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._reconnect_delay = config.reconnect_delay_initial
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks
        self._on_connect: Optional[Callable[[], None]] = None
        self._on_disconnect: Optional[Callable[[Optional[Exception]], None]] = None
        self._on_message: Optional[Callable[[DeribitMessage], None]] = None
        self._on_error: Optional[Callable[[Exception], None]] = None

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state in (ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED)

    @property
    def is_authenticated(self) -> bool:
        """Check if authenticated."""
        return self._state == ConnectionState.AUTHENTICATED

    def set_on_connect(self, callback: Callable[[], None]) -> None:
        """Set callback for connection established."""
        self._on_connect = callback

    def set_on_disconnect(self, callback: Callable[[Optional[Exception]], None]) -> None:
        """Set callback for disconnection."""
        self._on_disconnect = callback

    def set_on_message(self, callback: Callable[[DeribitMessage], None]) -> None:
        """Set callback for all messages."""
        self._on_message = callback

    def set_on_error(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for errors."""
        self._on_error = callback

    async def connect(self) -> bool:
        """
        Connect to Deribit WebSocket.

        Returns:
            True if connection successful
        """
        if self.is_connected:
            return True

        try:
            self._state = ConnectionState.CONNECTING
            logger.info(f"Connecting to Deribit WebSocket: {self._config.ws_url}")

            self._ws = await websockets.connect(
                self._config.ws_url,
                ping_interval=None,  # We handle heartbeats ourselves
                ping_timeout=30,
            )

            self._state = ConnectionState.CONNECTED
            self._reconnect_delay = self._config.reconnect_delay_initial
            logger.info("Connected to Deribit WebSocket")

            # Authenticate if credentials provided
            if self._config.requires_auth:
                success = await self._authenticate()
                if not success:
                    logger.warning("Authentication failed, continuing with public access only")
                else:
                    self._state = ConnectionState.AUTHENTICATED

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Trigger callback
            if self._on_connect:
                try:
                    self._on_connect()
                except Exception as e:
                    logger.error(f"Error in on_connect callback: {e}")

            return True

        except Exception as e:
            self._state = ConnectionState.DISCONNECTED
            logger.error(f"Failed to connect: {e}")
            if self._on_error:
                self._on_error(e)
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        self._state = ConnectionState.CLOSED

        # Cancel tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self._ws:
            await self._ws.close()
            self._ws = None

        logger.info("Disconnected from Deribit WebSocket")

    async def _authenticate(self) -> bool:
        """Authenticate with Deribit."""
        try:
            response = await self._send_request(
                "public/auth",
                {
                    "grant_type": "client_credentials",
                    "client_id": self._config.client_id,
                    "client_secret": self._config.client_secret,
                },
            )

            if response and "access_token" in response:
                self._access_token = response["access_token"]
                self._refresh_token = response.get("refresh_token")
                self._token_expiry = time.time() + response.get("expires_in", 900) - 60
                logger.info("WebSocket authentication successful")
                return True

            logger.error("Authentication failed: no access token")
            return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def subscribe(self, subscription: DeribitSubscription) -> bool:
        """
        Subscribe to a channel.

        Args:
            subscription: Subscription details

        Returns:
            True if subscription successful
        """
        if not self.is_connected:
            logger.error("Cannot subscribe: not connected")
            return False

        if subscription.is_private and not self.is_authenticated:
            logger.error(
                f"Cannot subscribe to private channel {subscription.channel}: not authenticated"
            )
            return False

        try:
            method = "private/subscribe" if subscription.is_private else "public/subscribe"
            response = await self._send_request(method, {"channels": [subscription.channel]})

            if response:
                self._subscriptions[subscription.channel] = subscription
                logger.info(f"Subscribed to {subscription.channel}")
                return True

            return False

        except Exception as e:
            logger.error(f"Subscription error for {subscription.channel}: {e}")
            return False

    async def unsubscribe(self, channel: str) -> bool:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel name to unsubscribe

        Returns:
            True if unsubscription successful
        """
        if channel not in self._subscriptions:
            return True

        try:
            sub = self._subscriptions[channel]
            method = "private/unsubscribe" if sub.is_private else "public/unsubscribe"
            response = await self._send_request(method, {"channels": [channel]})

            if response:
                del self._subscriptions[channel]
                logger.info(f"Unsubscribed from {channel}")
                return True

            return False

        except Exception as e:
            logger.error(f"Unsubscription error for {channel}: {e}")
            return False

    async def _send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """Send JSON-RPC request and wait for response."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        self._request_id += 1
        request_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            message["params"] = params

        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_id] = future

        try:
            await self._ws.send(json.dumps(message))
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {method}")
            return None
        finally:
            self._pending_requests.pop(request_id, None)

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats to keep connection alive."""
        while self.is_connected:
            try:
                await asyncio.sleep(self._config.heartbeat_interval)
                if self._ws and self.is_connected:
                    # Use Deribit's test endpoint as heartbeat
                    await self._send_request("public/test", timeout=5.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

    async def _receive_loop(self) -> None:
        """Receive and process messages."""
        while self._running and self._ws:
            try:
                raw_msg = await self._ws.recv()
                msg = json.loads(raw_msg)

                # Handle response to request
                if "id" in msg and msg["id"] in self._pending_requests:
                    request_id = msg["id"]
                    future = self._pending_requests.get(request_id)
                    if future and not future.done():
                        if "error" in msg:
                            future.set_exception(Exception(msg["error"]))
                        else:
                            future.set_result(msg.get("result"))

                # Handle subscription notification
                elif "method" in msg and msg["method"] == "subscription":
                    parsed = DeribitMessage.from_notification(msg)

                    # Call subscription callback
                    if parsed.channel and parsed.channel in self._subscriptions:
                        sub = self._subscriptions[parsed.channel]
                        if sub.callback:
                            try:
                                sub.callback(parsed.data)
                            except Exception as e:
                                logger.error(f"Error in subscription callback: {e}")

                    # Call global message callback
                    if self._on_message:
                        try:
                            self._on_message(parsed)
                        except Exception as e:
                            logger.error(f"Error in on_message callback: {e}")

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                await self._handle_disconnect(e)
                break

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                if self._on_error:
                    self._on_error(e)

    async def _handle_disconnect(self, error: Optional[Exception] = None) -> None:
        """Handle disconnection and potentially reconnect."""
        prev_state = self._state
        self._state = ConnectionState.DISCONNECTED

        # Trigger callback
        if self._on_disconnect:
            try:
                self._on_disconnect(error)
            except Exception as e:
                logger.error(f"Error in on_disconnect callback: {e}")

        # Attempt reconnection if enabled
        if self._config.reconnect_enabled and self._running:
            await self._reconnect()

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        while self._running:
            self._state = ConnectionState.RECONNECTING
            logger.info(f"Attempting reconnection in {self._reconnect_delay}s...")

            await asyncio.sleep(self._reconnect_delay)

            if await self.connect():
                # Resubscribe to channels
                if self._config.auto_subscribe_on_reconnect:
                    channels = list(self._subscriptions.values())
                    self._subscriptions.clear()
                    for sub in channels:
                        await self.subscribe(sub)
                return

            # Increase delay for next attempt
            self._reconnect_delay = min(
                self._reconnect_delay * RECONNECT_BACKOFF_MULTIPLIER,
                self._config.reconnect_delay_max,
            )

    async def run_forever(self) -> None:
        """
        Run the WebSocket client indefinitely.

        Handles message receiving and reconnection.
        """
        if not self.is_connected:
            if not await self.connect():
                raise RuntimeError("Failed to connect")

        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())

        try:
            await self._receive_task
        except asyncio.CancelledError:
            pass
        finally:
            await self.disconnect()

    async def run_for(self, duration: float) -> None:
        """
        Run the WebSocket client for a specified duration.

        Args:
            duration: Duration in seconds
        """
        if not self.is_connected:
            if not await self.connect():
                raise RuntimeError("Failed to connect")

        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())

        try:
            await asyncio.sleep(duration)
        finally:
            await self.disconnect()


# =============================================================================
# Factory Function
# =============================================================================


def create_deribit_websocket_client(
    testnet: bool = True,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    reconnect: bool = True,
    **kwargs,
) -> DeribitWebSocketClient:
    """
    Create a Deribit WebSocket client.

    Args:
        testnet: If True, use testnet
        client_id: API client ID (for private channels)
        client_secret: API client secret (for private channels)
        reconnect: Enable automatic reconnection
        **kwargs: Additional configuration

    Returns:
        Configured WebSocket client

    Example:
        >>> # Public channels only
        >>> client = create_deribit_websocket_client(testnet=True)
        >>>
        >>> # With authentication for private channels
        >>> client = create_deribit_websocket_client(
        ...     testnet=True,
        ...     client_id="your_client_id",
        ...     client_secret="your_client_secret",
        ... )
    """
    config = DeribitStreamConfig(
        testnet=testnet,
        client_id=client_id,
        client_secret=client_secret,
        reconnect_enabled=reconnect,
        **kwargs,
    )
    return DeribitWebSocketClient(config)
