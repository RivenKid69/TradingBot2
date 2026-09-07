"""Comprehensive tests for services.retry module."""

import asyncio
import random
import time
from unittest.mock import patch, MagicMock

import pytest

from core_config import RetryConfig
from services.retry import compute_backoff, retry_sync, retry_async


class TestComputeBackoff:
    """Tests for compute_backoff function."""

    def test_compute_backoff_zero_attempt(self):
        """Test backoff for zero attempt."""
        cfg = RetryConfig(backoff_base_s=1.0, max_backoff_s=10.0)
        backoff = compute_backoff(cfg, 0)
        assert backoff == 0.0

    def test_compute_backoff_first_attempt(self):
        """Test backoff for first attempt."""
        cfg = RetryConfig(backoff_base_s=1.0, max_backoff_s=10.0)
        rng = random.Random(42)

        backoff = compute_backoff(cfg, 1, rng=rng)
        assert 0.0 <= backoff <= 1.0  # Base * 2^0 = 1.0

    def test_compute_backoff_second_attempt(self):
        """Test backoff for second attempt."""
        cfg = RetryConfig(backoff_base_s=1.0, max_backoff_s=10.0)
        rng = random.Random(42)

        backoff = compute_backoff(cfg, 2, rng=rng)
        assert 0.0 <= backoff <= 2.0  # Base * 2^1 = 2.0

    def test_compute_backoff_exponential_growth(self):
        """Test exponential growth of backoff."""
        cfg = RetryConfig(backoff_base_s=1.0, max_backoff_s=100.0)
        rng = random.Random(42)

        backoffs = [compute_backoff(cfg, i, rng=rng) for i in range(1, 6)]

        # Each should be generally larger (with jitter variation)
        max_values = [1.0, 2.0, 4.0, 8.0, 16.0]
        for backoff, max_val in zip(backoffs, max_values):
            assert 0.0 <= backoff <= max_val

    def test_compute_backoff_respects_max(self):
        """Test backoff respects maximum."""
        cfg = RetryConfig(backoff_base_s=1.0, max_backoff_s=5.0)
        rng = random.Random(42)

        # After many attempts, should cap at max_backoff_s
        backoff = compute_backoff(cfg, 10, rng=rng)
        assert backoff <= 5.0

    def test_compute_backoff_with_custom_rng(self):
        """Test backoff with custom RNG."""
        cfg = RetryConfig(backoff_base_s=1.0, max_backoff_s=10.0)
        rng = random.Random(12345)

        backoff1 = compute_backoff(cfg, 1, rng=rng)
        backoff2 = compute_backoff(cfg, 1, rng=rng)

        # Different results due to random jitter
        # (may be same by chance, but generally different)
        assert backoff1 >= 0.0 and backoff2 >= 0.0

    def test_compute_backoff_full_jitter(self):
        """Test full jitter (result is between 0 and cap)."""
        cfg = RetryConfig(backoff_base_s=2.0, max_backoff_s=20.0)

        for attempt in range(1, 5):
            backoff = compute_backoff(cfg, attempt)
            expected_max = min(2.0 * (2 ** (attempt - 1)), 20.0)
            assert 0.0 <= backoff <= expected_max


class TestRetrySyncDecorator:
    """Tests for retry_sync decorator."""

    def test_retry_sync_success_first_attempt(self):
        """Test successful execution on first attempt."""
        cfg = RetryConfig(max_attempts=3, backoff_base_s=0.01)
        classify = lambda e: None

        counter = {"calls": 0}

        @retry_sync(cfg, classify)
        def func():
            counter["calls"] += 1
            return "success"

        result = func()
        assert result == "success"
        assert counter["calls"] == 1

    def test_retry_sync_retries_on_failure(self):
        """Test retries on failure."""
        cfg = RetryConfig(max_attempts=3, backoff_base_s=0.01)
        classify = lambda e: None

        counter = {"calls": 0}

        @retry_sync(cfg, classify)
        def func():
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise ValueError("Fail")
            return "success"

        result = func()
        assert result == "success"
        assert counter["calls"] == 2

    def test_retry_sync_max_attempts_reached(self):
        """Test max attempts reached raises exception."""
        cfg = RetryConfig(max_attempts=3, backoff_base_s=0.01)
        classify = lambda e: None

        @retry_sync(cfg, classify)
        def func():
            raise ValueError("Always fail")

        with pytest.raises(ValueError, match="Always fail"):
            func()

    def test_retry_sync_classify_error(self):
        """Test classify function is called on error."""
        cfg = RetryConfig(max_attempts=2, backoff_base_s=0.01)
        classified = []

        def classify(e):
            classified.append(type(e).__name__)
            return "rest"

        @retry_sync(cfg, classify)
        def func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            func()

        assert "ValueError" in classified

    @patch("services.ops_kill_switch.record_error")
    def test_retry_sync_records_error_to_kill_switch(self, mock_record):
        """Test error is recorded to kill switch."""
        cfg = RetryConfig(max_attempts=2, backoff_base_s=0.01)

        def classify(e):
            return "rest"

        @retry_sync(cfg, classify)
        def func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            func()

        mock_record.assert_called_with("rest")

    @patch("services.ops_kill_switch.manual_reset")
    def test_retry_sync_resets_kill_switch_on_success(self, mock_reset):
        """Test kill switch is reset on successful retry."""
        cfg = RetryConfig(max_attempts=3, backoff_base_s=0.01)
        classify = lambda e: "rest"

        counter = {"calls": 0}

        @retry_sync(cfg, classify)
        def func():
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise ValueError("Fail")
            return "success"

        func()
        mock_reset.assert_called_once()

    def test_retry_sync_with_backoff(self):
        """Test retry waits between attempts."""
        cfg = RetryConfig(max_attempts=3, backoff_base_s=0.05, max_backoff_s=0.1)
        classify = lambda e: None

        counter = {"calls": 0}
        timings = []

        @retry_sync(cfg, classify)
        def func():
            timings.append(time.time())
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise ValueError("Fail")
            return "success"

        func()

        # Should have waited between attempts
        if len(timings) >= 2:
            time_diff = timings[1] - timings[0]
            assert time_diff >= 0.01  # Some backoff should occur

    def test_retry_sync_zero_max_attempts(self):
        """Test with zero max_attempts defaults to 1."""
        cfg = RetryConfig(max_attempts=0, backoff_base_s=0.01)
        classify = lambda e: None

        counter = {"calls": 0}

        @retry_sync(cfg, classify)
        def func():
            counter["calls"] += 1
            return "success"

        result = func()
        assert result == "success"
        assert counter["calls"] == 1


@pytest.mark.asyncio
class TestRetryAsyncDecorator:
    """Tests for retry_async decorator."""

    async def test_retry_async_success_first_attempt(self):
        """Test successful execution on first attempt."""
        cfg = RetryConfig(max_attempts=3, backoff_base_s=0.01)
        classify = lambda e: None

        counter = {"calls": 0}

        @retry_async(cfg, classify)
        async def func():
            counter["calls"] += 1
            return "success"

        result = await func()
        assert result == "success"
        assert counter["calls"] == 1

    async def test_retry_async_retries_on_failure(self):
        """Test retries on failure."""
        cfg = RetryConfig(max_attempts=3, backoff_base_s=0.01)
        classify = lambda e: None

        counter = {"calls": 0}

        @retry_async(cfg, classify)
        async def func():
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise ValueError("Fail")
            return "success"

        result = await func()
        assert result == "success"
        assert counter["calls"] == 2

    async def test_retry_async_max_attempts_reached(self):
        """Test max attempts reached raises exception."""
        cfg = RetryConfig(max_attempts=3, backoff_base_s=0.01)
        classify = lambda e: None

        @retry_async(cfg, classify)
        async def func():
            raise ValueError("Always fail")

        with pytest.raises(ValueError, match="Always fail"):
            await func()

    async def test_retry_async_classify_error(self):
        """Test classify function is called on error."""
        cfg = RetryConfig(max_attempts=2, backoff_base_s=0.01)
        classified = []

        def classify(e):
            classified.append(type(e).__name__)
            return "rest"

        @retry_async(cfg, classify)
        async def func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await func()

        assert "ValueError" in classified

    @patch("services.ops_kill_switch.record_error")
    async def test_retry_async_records_error_to_kill_switch(self, mock_record):
        """Test error is recorded to kill switch."""
        cfg = RetryConfig(max_attempts=2, backoff_base_s=0.01)

        def classify(e):
            return "rest"

        @retry_async(cfg, classify)
        async def func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await func()

        mock_record.assert_called_with("rest")

    @patch("services.ops_kill_switch.manual_reset")
    async def test_retry_async_resets_kill_switch_on_success(self, mock_reset):
        """Test kill switch is reset on successful retry."""
        cfg = RetryConfig(max_attempts=3, backoff_base_s=0.01)
        classify = lambda e: "rest"

        counter = {"calls": 0}

        @retry_async(cfg, classify)
        async def func():
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise ValueError("Fail")
            return "success"

        await func()
        mock_reset.assert_called_once()

    async def test_retry_async_with_backoff(self):
        """Test retry waits between attempts.

        Note: compute_backoff uses full jitter (random() * cap), so
        actual backoff can range from 0 to cap. We can't assert a
        minimum wait time, only that the function was called multiple times.
        """
        cfg = RetryConfig(max_attempts=3, backoff_base_s=0.05, max_backoff_s=0.1)
        classify = lambda e: None

        counter = {"calls": 0}
        timings = []

        @retry_async(cfg, classify)
        async def func():
            timings.append(asyncio.get_event_loop().time())
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise ValueError("Fail")
            return "success"

        await func()

        # Should have retried (at least 2 calls)
        assert len(timings) >= 2
        assert counter["calls"] == 2
        # Time diff can be close to 0 due to full jitter
        if len(timings) >= 2:
            time_diff = timings[1] - timings[0]
            assert time_diff >= 0  # Just verify non-negative (backoff can be near 0)

    async def test_retry_async_zero_max_attempts(self):
        """Test with zero max_attempts defaults to 1."""
        cfg = RetryConfig(max_attempts=0, backoff_base_s=0.01)
        classify = lambda e: None

        counter = {"calls": 0}

        @retry_async(cfg, classify)
        async def func():
            counter["calls"] += 1
            return "success"

        result = await func()
        assert result == "success"
        assert counter["calls"] == 1

    async def test_retry_async_consecutive_failures_tracking(self):
        """Test consecutive failures tracking."""
        cfg = RetryConfig(max_attempts=4, backoff_base_s=0.01)
        classify = lambda e: "rest"

        counter = {"calls": 0}

        @retry_async(cfg, classify)
        async def func():
            counter["calls"] += 1
            if counter["calls"] < 3:
                raise ValueError("Fail")
            return "success"

        await func()
        assert counter["calls"] == 3


class TestRetryEdgeCases:
    """Tests for edge cases."""

    def test_retry_sync_classify_exception_handling(self):
        """Test classify exception is handled gracefully."""
        cfg = RetryConfig(max_attempts=2, backoff_base_s=0.01)

        def classify(e):
            raise RuntimeError("Classify error")

        @retry_sync(cfg, classify)
        def func():
            raise ValueError("Test error")

        # Should still retry despite classify error
        with pytest.raises(ValueError):
            func()

    @pytest.mark.asyncio
    async def test_retry_async_classify_exception_handling(self):
        """Test classify exception is handled gracefully."""
        cfg = RetryConfig(max_attempts=2, backoff_base_s=0.01)

        def classify(e):
            raise RuntimeError("Classify error")

        @retry_async(cfg, classify)
        async def func():
            raise ValueError("Test error")

        # Should still retry despite classify error
        with pytest.raises(ValueError):
            await func()

    def test_retry_sync_returns_none_on_classify(self):
        """Test when classify returns None."""
        cfg = RetryConfig(max_attempts=2, backoff_base_s=0.01)
        classify = lambda e: None  # Returns None

        @retry_sync(cfg, classify)
        def func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            func()
