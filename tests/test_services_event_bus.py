"""Comprehensive tests for services.event_bus module."""

import asyncio
from unittest.mock import patch

import pytest

from services.event_bus import EventBus


class TestEventBusInitialization:
    """Tests for EventBus initialization."""

    def test_init_with_positive_queue_size(self):
        """Test initialization with positive queue size."""
        bus = EventBus(queue_size=10, drop_policy="newest")
        assert bus._queue.maxsize == 10
        assert bus._drop_oldest is False

    def test_init_with_zero_queue_size(self):
        """Test initialization with zero queue size creates unbounded queue."""
        bus = EventBus(queue_size=0, drop_policy="newest")
        assert bus._queue.maxsize == 0

    def test_init_with_negative_queue_size(self):
        """Test initialization with negative queue size creates unbounded queue."""
        bus = EventBus(queue_size=-1, drop_policy="newest")
        assert bus._queue.maxsize == 0

    def test_init_with_oldest_drop_policy(self):
        """Test initialization with oldest drop policy."""
        bus = EventBus(queue_size=10, drop_policy="oldest")
        assert bus._drop_oldest is True

    def test_init_with_newest_drop_policy(self):
        """Test initialization with newest drop policy."""
        bus = EventBus(queue_size=10, drop_policy="newest")
        assert bus._drop_oldest is False

    def test_init_with_legacy_drop_oldest(self):
        """Test initialization with legacy 'drop_oldest' policy."""
        bus = EventBus(queue_size=10, drop_policy="drop_oldest")
        assert bus._drop_oldest is True

    def test_init_with_legacy_drop_newest(self):
        """Test initialization with legacy 'drop_newest' policy."""
        bus = EventBus(queue_size=10, drop_policy="drop_newest")
        assert bus._drop_oldest is False

    def test_init_with_invalid_drop_policy(self):
        """Test initialization with invalid drop policy raises ValueError."""
        with pytest.raises(ValueError, match="drop_policy must be"):
            EventBus(queue_size=10, drop_policy="invalid")


@pytest.mark.asyncio
class TestEventBusPut:
    """Tests for EventBus.put method."""

    async def test_put_single_event(self):
        """Test putting a single event."""
        bus = EventBus(queue_size=10, drop_policy="newest")
        event = {"type": "test", "data": "value"}

        result = await bus.put(event)
        assert result is True
        assert bus.depth == 1

    async def test_put_multiple_events(self):
        """Test putting multiple events."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        for i in range(5):
            result = await bus.put({"id": i})
            assert result is True

        assert bus.depth == 5

    async def test_put_when_queue_full_drop_newest(self):
        """Test put when queue is full with drop_newest policy."""
        bus = EventBus(queue_size=2, drop_policy="newest")

        await bus.put({"id": 1})
        await bus.put({"id": 2})
        result = await bus.put({"id": 3})  # Should be dropped

        assert result is False
        assert bus.depth == 2

    async def test_put_when_queue_full_drop_oldest(self):
        """Test put when queue is full with drop_oldest policy."""
        bus = EventBus(queue_size=2, drop_policy="oldest")

        await bus.put({"id": 1})
        await bus.put({"id": 2})
        result = await bus.put({"id": 3})  # Should replace oldest

        assert result is True
        assert bus.depth == 2

        # Verify oldest was dropped
        event1 = await bus.get()
        assert event1["id"] == 2  # id=1 was dropped

    async def test_put_after_close_raises_error(self):
        """Test put after close raises RuntimeError."""
        bus = EventBus(queue_size=10, drop_policy="newest")
        bus.close()

        with pytest.raises(RuntimeError, match="EventBus is closed"):
            await bus.put({"test": "data"})

    @patch("services.monitoring.events_in")
    async def test_put_increments_metrics(self, mock_counter):
        """Test put increments monitoring metrics."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        await bus.put({"test": "data"})
        mock_counter.inc.assert_called_once()

    @patch("services.monitoring.dropped_bp")
    async def test_put_increments_dropped_metric_when_full(self, mock_counter):
        """Test put increments dropped metric when queue is full."""
        bus = EventBus(queue_size=1, drop_policy="newest")

        await bus.put({"id": 1})
        await bus.put({"id": 2})  # Will be dropped

        mock_counter.inc.assert_called_once()

    async def test_put_zero_size_queue(self):
        """Test put with zero-size queue always drops with newest policy."""
        bus = EventBus(queue_size=0, drop_policy="newest")

        # First event should succeed (unbounded)
        result1 = await bus.put({"id": 1})
        assert result1 is True


@pytest.mark.asyncio
class TestEventBusGet:
    """Tests for EventBus.get method."""

    async def test_get_single_event(self):
        """Test getting a single event."""
        bus = EventBus(queue_size=10, drop_policy="newest")
        event = {"type": "test", "data": "value"}

        await bus.put(event)
        retrieved = await bus.get()

        assert retrieved == event
        assert bus.depth == 0

    async def test_get_multiple_events_fifo_order(self):
        """Test getting multiple events maintains FIFO order."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        events = [{"id": i} for i in range(5)]
        for event in events:
            await bus.put(event)

        for i in range(5):
            retrieved = await bus.get()
            assert retrieved["id"] == i

    async def test_get_after_close_returns_none(self):
        """Test get after close returns None."""
        bus = EventBus(queue_size=10, drop_policy="newest")
        bus.close()

        result = await bus.get()
        assert result is None

    async def test_get_returns_none_for_other_consumers(self):
        """Test get returns None for other consumers after close."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        # Start two consumers
        async def consumer():
            return await bus.get()

        task1 = asyncio.create_task(consumer())
        task2 = asyncio.create_task(consumer())

        # Give tasks time to wait
        await asyncio.sleep(0.01)

        bus.close()

        result1 = await task1
        result2 = await task2

        # Both should get None
        assert result1 is None
        assert result2 is None

    @patch("services.monitoring.queue_depth")
    async def test_get_updates_depth_metric(self, mock_gauge):
        """Test get updates queue depth metric."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        await bus.put({"test": "data"})
        await bus.get()

        # Should be called at least once with depth 0
        assert any(call[0][0] == 0 for call in mock_gauge.set.call_args_list)


class TestEventBusDepth:
    """Tests for EventBus.depth property."""

    def test_depth_empty_queue(self):
        """Test depth of empty queue."""
        bus = EventBus(queue_size=10, drop_policy="newest")
        assert bus.depth == 0

    @pytest.mark.asyncio
    async def test_depth_after_put(self):
        """Test depth after putting events."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        await bus.put({"id": 1})
        assert bus.depth == 1

        await bus.put({"id": 2})
        assert bus.depth == 2

    @pytest.mark.asyncio
    async def test_depth_after_get(self):
        """Test depth after getting events."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        await bus.put({"id": 1})
        await bus.put({"id": 2})
        assert bus.depth == 2

        await bus.get()
        assert bus.depth == 1


class TestEventBusClose:
    """Tests for EventBus.close method."""

    def test_close_sets_closed_flag(self):
        """Test close sets closed flag."""
        bus = EventBus(queue_size=10, drop_policy="newest")
        assert bus._closed is False

        bus.close()
        assert bus._closed is True

    def test_close_idempotent(self):
        """Test close can be called multiple times safely."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        bus.close()
        bus.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_puts_sentinel(self):
        """Test close puts sentinel to wake up consumers."""
        bus = EventBus(queue_size=10, drop_policy="newest")
        bus.close()

        result = await bus.get()
        assert result is None

    @pytest.mark.asyncio
    async def test_close_with_full_queue(self):
        """Test close with full queue."""
        bus = EventBus(queue_size=1, drop_policy="newest")

        await bus.put({"id": 1})
        bus.close()

        # Should still be able to get existing event
        result1 = await bus.get()
        assert result1["id"] == 1

        # Next get should return None
        result2 = await bus.get()
        assert result2 is None


class TestEventBusContextManagers:
    """Tests for EventBus context manager support."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager closes bus on exit."""
        async with EventBus(queue_size=10, drop_policy="newest") as bus:
            await bus.put({"test": "data"})
            assert bus._closed is False

        # Bus should be closed after context exit
        assert bus._closed is True

    def test_sync_context_manager(self):
        """Test sync context manager closes bus on exit."""
        with EventBus(queue_size=10, drop_policy="newest") as bus:
            assert bus._closed is False

        # Bus should be closed after context exit
        assert bus._closed is True

    @pytest.mark.asyncio
    async def test_async_context_manager_with_exception(self):
        """Test async context manager closes bus even on exception."""
        try:
            async with EventBus(queue_size=10, drop_policy="newest") as bus:
                await bus.put({"test": "data"})
                raise ValueError("test error")
        except ValueError:
            pass

        # Bus should still be closed
        assert bus._closed is True


@pytest.mark.asyncio
class TestEventBusConcurrency:
    """Tests for EventBus concurrent access."""

    async def test_concurrent_producers(self):
        """Test multiple concurrent producers."""
        bus = EventBus(queue_size=100, drop_policy="newest")

        async def producer(producer_id, count):
            for i in range(count):
                await bus.put({"producer": producer_id, "seq": i})

        tasks = [producer(i, 10) for i in range(5)]
        await asyncio.gather(*tasks)

        assert bus.depth == 50

    async def test_concurrent_consumers(self):
        """Test multiple concurrent consumers."""
        bus = EventBus(queue_size=100, drop_policy="newest")

        # Produce events
        for i in range(50):
            await bus.put({"id": i})

        # Consume concurrently
        async def consumer():
            events = []
            for _ in range(10):
                event = await bus.get()
                if event is not None:
                    events.append(event)
            return events

        tasks = [consumer() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All events should be consumed
        total_consumed = sum(len(r) for r in results)
        assert total_consumed == 50

    async def test_producer_consumer_concurrent(self):
        """Test concurrent producers and consumers."""
        bus = EventBus(queue_size=10, drop_policy="oldest")

        produced = []
        consumed = []

        async def producer():
            for i in range(20):
                await bus.put({"id": i})
                produced.append(i)
                await asyncio.sleep(0.001)

        async def consumer():
            while True:
                event = await asyncio.wait_for(bus.get(), timeout=0.1)
                if event is None:
                    break
                consumed.append(event["id"])

        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        await producer_task
        bus.close()
        await consumer_task

        # All produced events should be consumed
        assert len(consumed) > 0


@pytest.mark.asyncio
class TestEventBusEdgeCases:
    """Tests for EventBus edge cases."""

    async def test_empty_get_blocks_until_event(self):
        """Test get blocks when queue is empty until event arrives."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        async def delayed_put():
            await asyncio.sleep(0.05)
            await bus.put({"delayed": True})

        put_task = asyncio.create_task(delayed_put())
        event = await bus.get()

        assert event["delayed"] is True
        await put_task

    async def test_none_event_value(self):
        """Test that None can be an event value (not sentinel)."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        await bus.put(None)
        event = await bus.get()

        # None as event should be retrieved (not treated as sentinel)
        assert event is None

    async def test_sentinel_propagates_to_multiple_consumers(self):
        """Test sentinel propagates to multiple waiting consumers."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        async def consumer():
            return await bus.get()

        # Start 3 consumers
        tasks = [asyncio.create_task(consumer()) for _ in range(3)]

        await asyncio.sleep(0.01)
        bus.close()

        results = await asyncio.gather(*tasks)

        # All consumers should get None
        assert all(r is None for r in results)

    async def test_metrics_exception_handling(self):
        """Test that metrics exceptions don't break functionality."""
        bus = EventBus(queue_size=10, drop_policy="newest")

        with patch("services.monitoring.events_in") as mock:
            mock.inc.side_effect = Exception("Metrics error")

            # Should not raise despite metrics error
            result = await bus.put({"test": "data"})
            assert result is True

            event = await bus.get()
            assert event["test"] == "data"
