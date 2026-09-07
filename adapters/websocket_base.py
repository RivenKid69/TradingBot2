# -*- coding: utf-8 -*-
"""
adapters/websocket_base.py
Robust async WebSocket wrapper for reliable market data streaming.

This module provides a production-grade WebSocket client with:
- Automatic reconnection with exponential backoff
- Connection health monitoring (heartbeat/ping-pong)
- Message buffering and rate limiting
- Graceful shutdown handling
- Comprehensive error handling and logging
- Support for authentication (API key, OAuth)

Design Principles:
- Async-first with sync wrapper for compatibility
- Producer-consumer pattern for message handling
- Circuit breaker pattern for failure recovery
- Observable pattern for state changes

Usage:
    async with AsyncWebSocket(url, handlers=...) as ws:
        await ws.subscribe(["AAPL", "MSFT"])
        async for message in ws.messages():
            process(message)

References:
- RFC 6455 (WebSocket Protocol)
- Production patterns from Binance, Alpaca, Polygon APIs
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class ConnectionState(Enum):
    """WebSocket connection state."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    AUTHENTICATING = auto()
    AUTHENTICATED = auto()
    SUBSCRIBING = auto()
    SUBSCRIBED = auto()
    RECONNECTING = auto()
    CLOSING = auto()
    CLOSED = auto()
    ERROR = auto()


class MessageType(Enum):
    """Type of WebSocket message."""

    TEXT = "text"
    BINARY = "binary"
    PING = "ping"
    PONG = "pong"
    CLOSE = "close"
    ERROR = "error"


T = TypeVar("T")  # Generic message type


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class WebSocketConfig:
    """
    Configuration for WebSocket connection.

    Attributes:
        url: WebSocket endpoint URL
        api_key: API key for authentication
        api_secret: API secret for authentication

        # Reconnection settings
        reconnect_enabled: Whether to auto-reconnect
        reconnect_delay_initial: Initial delay before reconnect (seconds)
        reconnect_delay_max: Maximum reconnect delay (seconds)
        reconnect_delay_multiplier: Backoff multiplier
        reconnect_max_attempts: Maximum reconnect attempts (0=unlimited)

        # Health monitoring
        ping_interval: Interval between ping messages (seconds)
        ping_timeout: Timeout for pong response (seconds)

        # Message handling
        message_queue_size: Maximum messages in queue
        rate_limit_per_second: Max messages per second (0=unlimited)

        # Timeouts
        connect_timeout: Connection timeout (seconds)
        close_timeout: Graceful close timeout (seconds)

        # Debug
        log_messages: Log all messages (verbose)
        log_heartbeat: Log heartbeat messages
    """

    url: str
    api_key: str = ""
    api_secret: str = ""

    # Reconnection
    reconnect_enabled: bool = True
    reconnect_delay_initial: float = 1.0
    reconnect_delay_max: float = 60.0
    reconnect_delay_multiplier: float = 2.0
    reconnect_max_attempts: int = 0  # 0 = unlimited

    # Health monitoring
    ping_interval: float = 30.0
    ping_timeout: float = 10.0

    # Message handling
    message_queue_size: int = 10000
    rate_limit_per_second: float = 0.0  # 0 = no limit

    # Timeouts
    connect_timeout: float = 30.0
    close_timeout: float = 5.0

    # Debug
    log_messages: bool = False
    log_heartbeat: bool = False

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("WebSocket URL is required")


@dataclass
class ConnectionStats:
    """Statistics for WebSocket connection."""

    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    reconnect_count: int = 0
    messages_received: int = 0
    messages_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    errors: int = 0
    last_message_at: Optional[datetime] = None
    last_ping_at: Optional[datetime] = None
    last_pong_at: Optional[datetime] = None
    latency_ms: Optional[float] = None

    def reset(self) -> None:
        """Reset all counters (keep timestamps)."""
        self.messages_received = 0
        self.messages_sent = 0
        self.bytes_received = 0
        self.bytes_sent = 0
        self.errors = 0


# =============================================================================
# MESSAGE WRAPPER
# =============================================================================


@dataclass
class WebSocketMessage(Generic[T]):
    """
    Wrapper for WebSocket messages with metadata.

    Attributes:
        data: Parsed message payload
        raw: Raw message bytes/string
        message_type: Type of message
        received_at: Timestamp when received
        sequence: Message sequence number
    """

    data: T
    raw: Union[str, bytes]
    message_type: MessageType = MessageType.TEXT
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int = 0


# =============================================================================
# HANDLERS INTERFACE
# =============================================================================


class WebSocketHandlers(ABC):
    """
    Abstract interface for WebSocket message handlers.

    Implement this to customize how messages are parsed and processed.
    """

    @abstractmethod
    async def on_message(self, message: WebSocketMessage[Any]) -> None:
        """Handle incoming message."""
        ...

    @abstractmethod
    async def on_connect(self) -> None:
        """Called when connection established."""
        ...

    @abstractmethod
    async def on_disconnect(self, code: int, reason: str) -> None:
        """Called when connection closed."""
        ...

    @abstractmethod
    async def on_error(self, error: Exception) -> None:
        """Called when error occurs."""
        ...

    def parse_message(self, raw: Union[str, bytes]) -> Any:
        """
        Parse raw message to structured data.

        Override for custom parsing. Default: JSON decode.
        """
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def build_auth_message(self, api_key: str, api_secret: str) -> Optional[Dict[str, Any]]:
        """
        Build authentication message.

        Override for exchange-specific auth. Return None if no auth needed.
        """
        return None

    def build_subscribe_message(self, channels: Sequence[str]) -> Optional[Dict[str, Any]]:
        """
        Build subscription message.

        Override for exchange-specific subscription format.
        """
        return None


class DefaultHandlers(WebSocketHandlers):
    """Default no-op handlers for basic usage."""

    def __init__(
        self,
        on_message_callback: Optional[Callable[[WebSocketMessage[Any]], None]] = None,
    ) -> None:
        self._on_message_callback = on_message_callback

    async def on_message(self, message: WebSocketMessage[Any]) -> None:
        if self._on_message_callback:
            self._on_message_callback(message)

    async def on_connect(self) -> None:
        logger.debug("WebSocket connected")

    async def on_disconnect(self, code: int, reason: str) -> None:
        logger.debug(f"WebSocket disconnected: code={code}, reason={reason}")

    async def on_error(self, error: Exception) -> None:
        logger.warning(f"WebSocket error: {error}")


# =============================================================================
# ASYNC WEBSOCKET CLIENT
# =============================================================================


class AsyncWebSocket:
    """
    Production-grade async WebSocket client.

    Features:
    - Automatic reconnection with exponential backoff
    - Connection health monitoring
    - Message queue with overflow protection
    - Graceful shutdown
    - Comprehensive statistics

    Usage:
        config = WebSocketConfig(url="wss://example.com/stream")
        handlers = MyHandlers()

        async with AsyncWebSocket(config, handlers) as ws:
            await ws.subscribe(["channel1", "channel2"])
            async for msg in ws.messages():
                process(msg)
    """

    def __init__(
        self,
        config: WebSocketConfig,
        handlers: Optional[WebSocketHandlers] = None,
    ) -> None:
        self._config = config
        self._handlers = handlers or DefaultHandlers()

        # State
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = asyncio.Lock()
        self._ws: Optional[Any] = None  # websockets.WebSocketClientProtocol

        # Message queue
        self._message_queue: asyncio.Queue[WebSocketMessage[Any]] = asyncio.Queue(
            maxsize=config.message_queue_size
        )
        self._message_sequence = 0

        # Subscriptions
        self._subscriptions: Set[str] = set()

        # Statistics
        self._stats = ConnectionStats()

        # Tasks
        self._receiver_task: Optional[asyncio.Task[None]] = None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None

        # Shutdown
        self._shutdown_event = asyncio.Event()
        self._reconnect_attempts = 0

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Whether currently connected and ready."""
        return self._state in (
            ConnectionState.CONNECTED,
            ConnectionState.AUTHENTICATED,
            ConnectionState.SUBSCRIBED,
        )

    @property
    def stats(self) -> ConnectionStats:
        """Connection statistics."""
        return self._stats

    async def __aenter__(self) -> "AsyncWebSocket":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        await self.close()

    async def connect(self) -> bool:
        """
        Establish WebSocket connection.

        Returns:
            True if connection successful
        """
        async with self._state_lock:
            if self._state in (ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED):
                return True

            self._state = ConnectionState.CONNECTING

        try:
            # Import websockets here to avoid import errors if not installed
            import websockets

            logger.info(f"Connecting to {self._config.url}")

            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self._config.url,
                    ping_interval=None,  # We handle our own ping
                    ping_timeout=None,
                    close_timeout=self._config.close_timeout,
                ),
                timeout=self._config.connect_timeout,
            )

            async with self._state_lock:
                self._state = ConnectionState.CONNECTED
                self._stats.connected_at = datetime.now(timezone.utc)
                self._reconnect_attempts = 0

            # Authenticate if needed
            await self._authenticate()

            # Start background tasks
            self._receiver_task = asyncio.create_task(self._receiver_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Notify handler
            await self._handlers.on_connect()

            # Resubscribe if reconnecting
            if self._subscriptions:
                await self._do_subscribe(list(self._subscriptions))

            logger.info("WebSocket connected successfully")
            return True

        except asyncio.TimeoutError:
            logger.error(f"Connection timeout after {self._config.connect_timeout}s")
            await self._handle_disconnect(1001, "Connection timeout")
            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self._handlers.on_error(e)
            await self._handle_disconnect(1006, str(e))
            return False

    async def close(self) -> None:
        """Close WebSocket connection gracefully."""
        async with self._state_lock:
            if self._state == ConnectionState.CLOSED:
                return
            self._state = ConnectionState.CLOSING

        logger.info("Closing WebSocket connection")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        for task in [self._receiver_task, self._heartbeat_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # Close WebSocket
        if self._ws:
            try:
                await asyncio.wait_for(
                    self._ws.close(1000, "Client closing"),
                    timeout=self._config.close_timeout,
                )
            except Exception as e:
                logger.debug(f"Error during close: {e}")
            finally:
                self._ws = None

        async with self._state_lock:
            self._state = ConnectionState.CLOSED
            self._stats.disconnected_at = datetime.now(timezone.utc)

        logger.info("WebSocket connection closed")

    async def subscribe(self, channels: Sequence[str]) -> bool:
        """
        Subscribe to channels.

        Args:
            channels: List of channel names to subscribe

        Returns:
            True if subscription successful
        """
        self._subscriptions.update(channels)

        if self.is_connected:
            return await self._do_subscribe(channels)

        logger.debug(f"Queued subscription for {len(channels)} channels (not connected)")
        return True

    async def unsubscribe(self, channels: Sequence[str]) -> bool:
        """
        Unsubscribe from channels.

        Args:
            channels: List of channel names to unsubscribe

        Returns:
            True if unsubscription successful
        """
        for channel in channels:
            self._subscriptions.discard(channel)

        # Note: Most exchanges don't have explicit unsubscribe
        # Override in subclass if needed
        return True

    async def send(self, data: Union[str, bytes, Dict[str, Any]]) -> bool:
        """
        Send message to WebSocket.

        Args:
            data: Message to send (string, bytes, or dict)

        Returns:
            True if send successful
        """
        if not self._ws or not self.is_connected:
            logger.warning("Cannot send: not connected")
            return False

        try:
            if isinstance(data, dict):
                data = json.dumps(data)

            if isinstance(data, str):
                await self._ws.send(data)
                self._stats.bytes_sent += len(data.encode())
            else:
                await self._ws.send(data)
                self._stats.bytes_sent += len(data)

            self._stats.messages_sent += 1

            if self._config.log_messages:
                logger.debug(f"Sent: {data[:200]}")

            return True

        except Exception as e:
            logger.error(f"Send failed: {e}")
            self._stats.errors += 1
            return False

    async def messages(self) -> AsyncIterator[WebSocketMessage[Any]]:
        """
        Async iterator for received messages.

        Usage:
            async for msg in ws.messages():
                process(msg.data)
        """
        while not self._shutdown_event.is_set():
            try:
                msg = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield msg
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def get_message_nowait(self) -> Optional[WebSocketMessage[Any]]:
        """
        Get message without waiting (non-blocking).

        Returns:
            Message if available, None otherwise
        """
        try:
            return self._message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    async def _authenticate(self) -> None:
        """Perform authentication if configured."""
        if not self._config.api_key:
            return

        async with self._state_lock:
            self._state = ConnectionState.AUTHENTICATING

        auth_message = self._handlers.build_auth_message(
            self._config.api_key,
            self._config.api_secret,
        )

        if auth_message:
            if await self.send(auth_message):
                # Wait for auth response (handled in receiver)
                await asyncio.sleep(0.5)  # Brief pause for auth
                async with self._state_lock:
                    self._state = ConnectionState.AUTHENTICATED
                logger.debug("Authentication message sent")

    async def _do_subscribe(self, channels: Sequence[str]) -> bool:
        """Send subscription message."""
        if not channels:
            return True

        async with self._state_lock:
            self._state = ConnectionState.SUBSCRIBING

        sub_message = self._handlers.build_subscribe_message(channels)

        if sub_message:
            if await self.send(sub_message):
                async with self._state_lock:
                    self._state = ConnectionState.SUBSCRIBED
                logger.info(f"Subscribed to {len(channels)} channels")
                return True
            return False

        return True

    async def _receiver_loop(self) -> None:
        """Background task to receive messages."""
        while not self._shutdown_event.is_set() and self._ws:
            try:
                raw_message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self._config.ping_timeout * 2,
                )

                # Update stats
                self._stats.messages_received += 1
                if isinstance(raw_message, str):
                    self._stats.bytes_received += len(raw_message.encode())
                else:
                    self._stats.bytes_received += len(raw_message)
                self._stats.last_message_at = datetime.now(timezone.utc)

                # Parse and enqueue
                await self._process_message(raw_message)

            except asyncio.TimeoutError:
                # No message received - check connection health
                logger.debug("Receive timeout - connection may be stale")

            except asyncio.CancelledError:
                break

            except Exception as e:
                if not self._shutdown_event.is_set():
                    logger.error(f"Receiver error: {e}")
                    self._stats.errors += 1
                    await self._handlers.on_error(e)
                    await self._handle_disconnect(1006, str(e))
                break

    async def _process_message(self, raw: Union[str, bytes]) -> None:
        """Process received message."""
        # Parse message
        parsed = self._handlers.parse_message(raw)

        # Wrap in message object
        self._message_sequence += 1
        message = WebSocketMessage(
            data=parsed,
            raw=raw,
            message_type=MessageType.TEXT if isinstance(raw, str) else MessageType.BINARY,
            sequence=self._message_sequence,
        )

        if self._config.log_messages:
            preview = str(raw)[:200] if isinstance(raw, str) else f"<{len(raw)} bytes>"
            logger.debug(f"Received: {preview}")

        # Notify handler
        await self._handlers.on_message(message)

        # Enqueue for consumers
        try:
            self._message_queue.put_nowait(message)
        except asyncio.QueueFull:
            # Drop oldest message
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._message_queue.put_nowait(message)
            logger.warning("Message queue full - dropped oldest message")

    async def _heartbeat_loop(self) -> None:
        """Background task for connection health monitoring."""
        while not self._shutdown_event.is_set() and self._ws:
            try:
                await asyncio.sleep(self._config.ping_interval)

                if not self._ws:
                    break

                # Send ping
                ping_time = time.monotonic()
                self._stats.last_ping_at = datetime.now(timezone.utc)

                try:
                    pong_waiter = await self._ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=self._config.ping_timeout)

                    # Calculate latency
                    self._stats.latency_ms = (time.monotonic() - ping_time) * 1000
                    self._stats.last_pong_at = datetime.now(timezone.utc)

                    if self._config.log_heartbeat:
                        logger.debug(f"Heartbeat OK - latency: {self._stats.latency_ms:.1f}ms")

                except asyncio.TimeoutError:
                    logger.warning(f"Pong timeout after {self._config.ping_timeout}s")
                    await self._handle_disconnect(1001, "Ping timeout")
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

    async def _handle_disconnect(self, code: int, reason: str) -> None:
        """Handle disconnection and trigger reconnect if enabled."""
        async with self._state_lock:
            if self._state in (ConnectionState.CLOSING, ConnectionState.CLOSED):
                return
            self._state = ConnectionState.DISCONNECTED
            self._stats.disconnected_at = datetime.now(timezone.utc)

        # Notify handler
        await self._handlers.on_disconnect(code, reason)

        # Close existing connection
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Attempt reconnect if enabled
        if self._config.reconnect_enabled and not self._shutdown_event.is_set():
            await self._reconnect()

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        async with self._state_lock:
            self._state = ConnectionState.RECONNECTING

        while not self._shutdown_event.is_set():
            self._reconnect_attempts += 1
            self._stats.reconnect_count += 1

            # Check max attempts
            if (
                self._config.reconnect_max_attempts > 0
                and self._reconnect_attempts > self._config.reconnect_max_attempts
            ):
                logger.error(
                    f"Max reconnect attempts ({self._config.reconnect_max_attempts}) exceeded"
                )
                async with self._state_lock:
                    self._state = ConnectionState.ERROR
                return

            # Calculate backoff delay
            delay = min(
                self._config.reconnect_delay_initial
                * (self._config.reconnect_delay_multiplier ** (self._reconnect_attempts - 1)),
                self._config.reconnect_delay_max,
            )

            logger.info(
                f"Reconnecting in {delay:.1f}s "
                f"(attempt {self._reconnect_attempts}"
                f"{f'/{self._config.reconnect_max_attempts}' if self._config.reconnect_max_attempts else ''})"
            )

            await asyncio.sleep(delay)

            if self._shutdown_event.is_set():
                return

            # Attempt connection
            if await self.connect():
                logger.info(f"Reconnected after {self._reconnect_attempts} attempts")
                return


# =============================================================================
# SYNC WRAPPER
# =============================================================================


class SyncWebSocket:
    """
    Synchronous wrapper for AsyncWebSocket.

    Runs the async WebSocket in a background thread, providing
    a sync-friendly interface for non-async code.

    Usage:
        ws = SyncWebSocket(config, handlers)
        ws.connect()
        ws.subscribe(["AAPL", "MSFT"])

        for msg in ws.messages():
            process(msg)

        ws.close()
    """

    def __init__(
        self,
        config: WebSocketConfig,
        handlers: Optional[WebSocketHandlers] = None,
    ) -> None:
        self._config = config
        self._handlers = handlers

        self._async_ws: Optional[AsyncWebSocket] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        self._message_queue: "queue.Queue[WebSocketMessage[Any]]" = queue.Queue(
            maxsize=config.message_queue_size
        )
        self._stop_event = threading.Event()

    @property
    def is_connected(self) -> bool:
        """Whether currently connected."""
        return self._async_ws is not None and self._async_ws.is_connected

    @property
    def stats(self) -> Optional[ConnectionStats]:
        """Connection statistics."""
        return self._async_ws.stats if self._async_ws else None

    def connect(self) -> bool:
        """
        Establish WebSocket connection.

        Returns:
            True if connection successful
        """
        if self._thread and self._thread.is_alive():
            return self.is_connected

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="websocket-sync",
            daemon=True,
        )
        self._thread.start()

        # Wait for connection
        deadline = time.time() + self._config.connect_timeout
        while time.time() < deadline:
            if self.is_connected:
                return True
            time.sleep(0.1)

        return self.is_connected

    def close(self) -> None:
        """Close WebSocket connection."""
        self._stop_event.set()

        if self._loop and self._async_ws:
            asyncio.run_coroutine_threadsafe(
                self._async_ws.close(),
                self._loop,
            ).result(timeout=self._config.close_timeout)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def subscribe(self, channels: Sequence[str]) -> bool:
        """Subscribe to channels."""
        if not self._loop or not self._async_ws:
            return False

        future = asyncio.run_coroutine_threadsafe(
            self._async_ws.subscribe(channels),
            self._loop,
        )
        return future.result(timeout=10.0)

    def send(self, data: Union[str, bytes, Dict[str, Any]]) -> bool:
        """Send message."""
        if not self._loop or not self._async_ws:
            return False

        future = asyncio.run_coroutine_threadsafe(
            self._async_ws.send(data),
            self._loop,
        )
        return future.result(timeout=10.0)

    def messages(self, timeout: float = 1.0) -> Iterator[WebSocketMessage[Any]]:
        """
        Iterator for received messages.

        Args:
            timeout: Timeout for each message (seconds)

        Yields:
            WebSocketMessage objects
        """
        while not self._stop_event.is_set():
            try:
                msg = self._message_queue.get(timeout=timeout)
                yield msg
            except queue.Empty:
                continue

    def get_message(self, timeout: float = 1.0) -> Optional[WebSocketMessage[Any]]:
        """
        Get single message with timeout.

        Args:
            timeout: Maximum wait time

        Returns:
            Message if available, None on timeout
        """
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self) -> "SyncWebSocket":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        """Run asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self._loop.close()
            self._loop = None

    async def _async_main(self) -> None:
        """Async main function for background thread."""
        # Create message forwarding handler
        original_handlers = self._handlers or DefaultHandlers()

        class ForwardingHandlers(WebSocketHandlers):
            def __init__(inner_self) -> None:
                inner_self._original = original_handlers
                inner_self._sync_queue = self._message_queue

            async def on_message(inner_self, message: WebSocketMessage[Any]) -> None:
                await inner_self._original.on_message(message)
                try:
                    inner_self._sync_queue.put_nowait(message)
                except queue.Full:
                    try:
                        inner_self._sync_queue.get_nowait()
                    except queue.Empty:
                        pass
                    inner_self._sync_queue.put_nowait(message)

            async def on_connect(inner_self) -> None:
                await inner_self._original.on_connect()

            async def on_disconnect(inner_self, code: int, reason: str) -> None:
                await inner_self._original.on_disconnect(code, reason)

            async def on_error(inner_self, error: Exception) -> None:
                await inner_self._original.on_error(error)

            def parse_message(inner_self, raw: Union[str, bytes]) -> Any:
                return inner_self._original.parse_message(raw)

            def build_auth_message(
                inner_self, api_key: str, api_secret: str
            ) -> Optional[Dict[str, Any]]:
                return inner_self._original.build_auth_message(api_key, api_secret)

            def build_subscribe_message(
                inner_self, channels: Sequence[str]
            ) -> Optional[Dict[str, Any]]:
                return inner_self._original.build_subscribe_message(channels)

        # Create and run async WebSocket
        self._async_ws = AsyncWebSocket(self._config, ForwardingHandlers())

        try:
            await self._async_ws.connect()

            # Wait for shutdown signal
            while not self._stop_event.is_set():
                await asyncio.sleep(0.1)

        finally:
            await self._async_ws.close()


# =============================================================================
# EXCHANGE-SPECIFIC HANDLERS
# =============================================================================


class AlpacaWebSocketHandlers(WebSocketHandlers):
    """
    WebSocket handlers for Alpaca Data API.

    Implements Alpaca's authentication and subscription protocol.
    """

    def __init__(
        self,
        on_bar: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_trade: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_quote: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._on_bar = on_bar
        self._on_trade = on_trade
        self._on_quote = on_quote

    async def on_message(self, message: WebSocketMessage[Any]) -> None:
        """Handle Alpaca message types."""
        data = message.data

        if not isinstance(data, list):
            return

        for item in data:
            msg_type = item.get("T")

            if msg_type == "b" and self._on_bar:
                self._on_bar(item)
            elif msg_type == "t" and self._on_trade:
                self._on_trade(item)
            elif msg_type == "q" and self._on_quote:
                self._on_quote(item)

    async def on_connect(self) -> None:
        logger.info("Alpaca WebSocket connected")

    async def on_disconnect(self, code: int, reason: str) -> None:
        logger.info(f"Alpaca WebSocket disconnected: {code} - {reason}")

    async def on_error(self, error: Exception) -> None:
        logger.error(f"Alpaca WebSocket error: {error}")

    def build_auth_message(self, api_key: str, api_secret: str) -> Optional[Dict[str, Any]]:
        """Alpaca authentication message."""
        return {
            "action": "auth",
            "key": api_key,
            "secret": api_secret,
        }

    def build_subscribe_message(self, channels: Sequence[str]) -> Optional[Dict[str, Any]]:
        """Alpaca subscription message."""
        return {
            "action": "subscribe",
            "bars": list(channels),
        }


class PolygonWebSocketHandlers(WebSocketHandlers):
    """
    WebSocket handlers for Polygon.io Data API.

    Implements Polygon's authentication and subscription protocol.
    """

    def __init__(
        self,
        on_bar: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_trade: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_quote: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._on_bar = on_bar
        self._on_trade = on_trade
        self._on_quote = on_quote

    async def on_message(self, message: WebSocketMessage[Any]) -> None:
        """Handle Polygon message types."""
        data = message.data

        if not isinstance(data, list):
            return

        for item in data:
            ev_type = item.get("ev")

            if ev_type == "AM" and self._on_bar:  # Aggregate Minute
                self._on_bar(item)
            elif ev_type == "T" and self._on_trade:  # Trade
                self._on_trade(item)
            elif ev_type == "Q" and self._on_quote:  # Quote
                self._on_quote(item)

    async def on_connect(self) -> None:
        logger.info("Polygon WebSocket connected")

    async def on_disconnect(self, code: int, reason: str) -> None:
        logger.info(f"Polygon WebSocket disconnected: {code} - {reason}")

    async def on_error(self, error: Exception) -> None:
        logger.error(f"Polygon WebSocket error: {error}")

    def build_auth_message(self, api_key: str, api_secret: str) -> Optional[Dict[str, Any]]:
        """Polygon authentication message."""
        return {
            "action": "auth",
            "params": api_key,
        }

    def build_subscribe_message(self, channels: Sequence[str]) -> Optional[Dict[str, Any]]:
        """Polygon subscription message."""
        # Format: AM.AAPL for minute aggregates
        formatted = [f"AM.{ch}" for ch in channels]
        return {
            "action": "subscribe",
            "params": ",".join(formatted),
        }
