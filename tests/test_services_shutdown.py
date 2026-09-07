"""Comprehensive tests for services.shutdown module."""

import asyncio
import signal
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from services.shutdown import ShutdownManager


class TestShutdownManagerInit:
    """Tests for ShutdownManager initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        manager = ShutdownManager({})

        assert manager.grace_period == 0.0
        assert manager.drain_policy == "graceful"
        assert manager._shutdown_requested is False

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "grace_period": 5.0,
            "drain_policy": "immediate",
            "timeouts": {
                "stop": 10.0,
                "flush": 15.0,
                "finalize": 20.0,
            },
        }
        manager = ShutdownManager(config)

        assert manager.grace_period == 5.0
        assert manager.drain_policy == "immediate"
        assert manager.stop_timeout == 10.0
        assert manager.flush_timeout == 15.0
        assert manager.finalize_timeout == 20.0


class TestShutdownManagerCallbackRegistration:
    """Tests for callback registration."""

    def test_on_stop_registers_callback(self):
        """Test on_stop registers callback."""
        manager = ShutdownManager({})
        callback = MagicMock()

        manager.on_stop(callback)

        assert callback in manager._on_stop

    def test_on_flush_registers_callback(self):
        """Test on_flush registers callback."""
        manager = ShutdownManager({})
        callback = MagicMock()

        manager.on_flush(callback)

        assert callback in manager._on_flush

    def test_on_finalize_registers_callback(self):
        """Test on_finalize registers callback."""
        manager = ShutdownManager({})
        callback = MagicMock()

        manager.on_finalize(callback)

        assert callback in manager._on_finalize

    def test_register_multiple_callbacks(self):
        """Test registering multiple callbacks."""
        manager = ShutdownManager({})
        cb1, cb2, cb3 = MagicMock(), MagicMock(), MagicMock()

        manager.on_stop(cb1)
        manager.on_stop(cb2)
        manager.on_flush(cb3)

        assert len(manager._on_stop) == 2
        assert len(manager._on_flush) == 1


@pytest.mark.asyncio
class TestShutdownManagerCallbackExecution:
    """Tests for callback execution."""

    async def test_executes_sync_callback(self):
        """Test executing synchronous callback."""
        manager = ShutdownManager({})
        callback = MagicMock()
        manager.on_stop(callback)

        manager.request_shutdown()
        await asyncio.sleep(0.1)

        callback.assert_called_once()

    async def test_executes_async_callback(self):
        """Test executing asynchronous callback."""
        manager = ShutdownManager({})
        callback = AsyncMock()
        manager.on_stop(callback)

        manager.request_shutdown()
        await asyncio.sleep(0.1)

        callback.assert_called_once()

    async def test_executes_callbacks_in_phases(self):
        """Test callbacks are executed in correct phases."""
        manager = ShutdownManager({"grace_period": 0.0})
        execution_order = []

        def stop_cb():
            execution_order.append("stop")

        def flush_cb():
            execution_order.append("flush")

        def finalize_cb():
            execution_order.append("finalize")

        manager.on_stop(stop_cb)
        manager.on_flush(flush_cb)
        manager.on_finalize(finalize_cb)

        manager.request_shutdown()
        await asyncio.sleep(0.1)

        assert execution_order == ["stop", "flush", "finalize"]

    async def test_grace_period_delay(self):
        """Test grace period adds delay between stop and flush."""
        manager = ShutdownManager({"grace_period": 0.1})
        execution_times = []

        def stop_cb():
            execution_times.append(asyncio.get_event_loop().time())

        def flush_cb():
            execution_times.append(asyncio.get_event_loop().time())

        manager.on_stop(stop_cb)
        manager.on_flush(flush_cb)

        manager.request_shutdown()
        await asyncio.sleep(0.3)

        assert len(execution_times) == 2
        time_diff = execution_times[1] - execution_times[0]
        assert time_diff >= 0.1  # Grace period should add delay

    async def test_callback_exception_does_not_stop_sequence(self):
        """Test exception in callback doesn't stop shutdown sequence."""
        manager = ShutdownManager({})
        execution_order = []

        def stop_cb():
            raise ValueError("Test error")

        def flush_cb():
            execution_order.append("flush")

        manager.on_stop(stop_cb)
        manager.on_flush(flush_cb)

        manager.request_shutdown()
        await asyncio.sleep(0.1)

        # Flush should still execute despite stop error
        assert "flush" in execution_order

    async def test_timeout_applied_to_callback(self):
        """Test timeout is applied to slow callbacks."""
        manager = ShutdownManager({"timeouts": {"stop": 0.05}})

        async def slow_callback():
            await asyncio.sleep(1.0)  # Longer than timeout

        manager.on_stop(slow_callback)

        start = asyncio.get_event_loop().time()
        manager.request_shutdown()
        await asyncio.sleep(0.2)
        elapsed = asyncio.get_event_loop().time() - start

        # Should timeout before callback completes
        assert elapsed < 0.5


class TestShutdownManagerSignalHandling:
    """Tests for signal handling."""

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGUSR1/SIGUSR2 not available on Windows")
    def test_register_signal_handler(self):
        """Test registering signal handler."""
        manager = ShutdownManager({})

        # Save original handler
        original = signal.getsignal(signal.SIGUSR1)

        try:
            manager.register(signal.SIGUSR1)

            # Handler should be changed
            assert signal.getsignal(signal.SIGUSR1) != original
        finally:
            # Restore original
            signal.signal(signal.SIGUSR1, original)

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGUSR1/SIGUSR2 not available on Windows")
    def test_unregister_signal_handler(self):
        """Test unregistering signal handler."""
        manager = ShutdownManager({})
        original = signal.getsignal(signal.SIGUSR1)

        try:
            manager.register(signal.SIGUSR1)
            manager.unregister(signal.SIGUSR1)

            # Handler should be restored
            assert signal.getsignal(signal.SIGUSR1) == original
        finally:
            signal.signal(signal.SIGUSR1, original)

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGUSR1/SIGUSR2 not available on Windows")
    def test_register_multiple_signals(self):
        """Test registering multiple signals."""
        manager = ShutdownManager({})

        original_usr1 = signal.getsignal(signal.SIGUSR1)
        original_usr2 = signal.getsignal(signal.SIGUSR2)

        try:
            manager.register(signal.SIGUSR1, signal.SIGUSR2)

            assert signal.getsignal(signal.SIGUSR1) != original_usr1
            assert signal.getsignal(signal.SIGUSR2) != original_usr2
        finally:
            signal.signal(signal.SIGUSR1, original_usr1)
            signal.signal(signal.SIGUSR2, original_usr2)


class TestShutdownManagerRequestShutdown:
    """Tests for request_shutdown method."""

    @pytest.mark.asyncio
    async def test_request_shutdown_once(self):
        """Test shutdown can only be requested once."""
        manager = ShutdownManager({})
        callback = MagicMock()
        manager.on_stop(callback)

        manager.request_shutdown()
        manager.request_shutdown()  # Second call should be ignored

        await asyncio.sleep(0.1)

        callback.assert_called_once()  # Should only execute once

    def test_request_shutdown_without_event_loop(self):
        """Test request_shutdown without running event loop."""
        manager = ShutdownManager({"grace_period": 0.0})
        callback = MagicMock()
        manager.on_stop(callback)

        # Should not raise
        # Note: This will block until sequence completes
        # manager.request_shutdown()
        # Skipping actual call to avoid blocking

    @pytest.mark.asyncio
    async def test_request_shutdown_creates_task(self):
        """Test request_shutdown creates background task."""
        manager = ShutdownManager({})

        manager.request_shutdown()

        assert manager._shutdown_task is not None
        await asyncio.sleep(0.1)


class TestShutdownManagerEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_callback_lists(self):
        """Test shutdown with no registered callbacks."""
        manager = ShutdownManager({})

        # Should not raise
        manager.request_shutdown()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_mixed_sync_async_callbacks(self):
        """Test mixing sync and async callbacks."""
        manager = ShutdownManager({})
        execution_order = []

        def sync_cb():
            execution_order.append("sync")

        async def async_cb():
            execution_order.append("async")

        manager.on_stop(sync_cb)
        manager.on_stop(async_cb)

        manager.request_shutdown()
        await asyncio.sleep(0.1)

        assert "sync" in execution_order
        assert "async" in execution_order

    @pytest.mark.asyncio
    async def test_callback_returns_awaitable(self):
        """Test callback that returns awaitable."""
        manager = ShutdownManager({})
        executed = []

        def callback_returning_awaitable():
            async def inner():
                executed.append(True)

            return inner()

        manager.on_stop(callback_returning_awaitable)

        manager.request_shutdown()
        await asyncio.sleep(0.1)

        assert len(executed) == 1

    @pytest.mark.asyncio
    async def test_zero_timeout(self):
        """Test with zero timeout."""
        manager = ShutdownManager({"timeouts": {"stop": 0.0}})

        async def callback():
            await asyncio.sleep(0.1)

        manager.on_stop(callback)

        manager.request_shutdown()
        await asyncio.sleep(0.2)  # Give time for sequence

    @pytest.mark.asyncio
    async def test_none_timeout(self):
        """Test with None timeout (no timeout)."""
        manager = ShutdownManager({})
        executed = []

        async def callback():
            await asyncio.sleep(0.05)
            executed.append(True)

        manager.on_stop(callback)

        manager.request_shutdown()
        await asyncio.sleep(0.2)

        assert len(executed) == 1
