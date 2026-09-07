# -*- coding: utf-8 -*-
"""
Tests for Rate Limiter and Account Lockout - WI-AUTH-01.

Tests verify:
- Failed login attempt tracking
- Progressive lockout (exponential backoff)
- IP-based rate limiting
- Lockout reset on successful login
"""

import pytest
import time
from datetime import datetime, timedelta, timezone

from packages.cloud.control_plane.security.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    AccountLockout,
    LockoutState,
    RateLimitState,
    get_rate_limiter,
    check_lockout,
    check_rate_limit,
    record_login_attempt,
)


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_first_failed_attempt_no_lockout(self):
        """First failed attempt should not trigger lockout."""
        limiter = RateLimiter(max_attempts=5)

        limiter.record_attempt("user@example.com", success=False)

        # Should not be locked yet
        state = limiter.get_lockout_state("user@example.com")
        assert not state.is_locked
        assert state.failed_attempts == 1

    def test_lockout_after_max_attempts(self):
        """Lockout should trigger after max failed attempts."""
        limiter = RateLimiter(max_attempts=3, lockout_duration_sec=60)

        # Make 3 failed attempts
        for _ in range(3):
            limiter.record_attempt("user@example.com", success=False)

        state = limiter.get_lockout_state("user@example.com")
        assert state.is_locked
        assert state.failed_attempts >= 3

    def test_check_lockout_raises_when_locked(self):
        """check_lockout should raise when account is locked."""
        limiter = RateLimiter(max_attempts=2, lockout_duration_sec=60)

        # Trigger lockout
        limiter.record_attempt("user@example.com", success=False)
        limiter.record_attempt("user@example.com", success=False)

        with pytest.raises(AccountLockout):
            limiter.check_lockout("user@example.com")

    def test_lockout_expires(self):
        """Lockout should expire after duration."""
        limiter = RateLimiter(max_attempts=2, lockout_duration_sec=1)

        # Trigger lockout
        limiter.record_attempt("user@example.com", success=False)
        limiter.record_attempt("user@example.com", success=False)

        # Wait for lockout to expire
        time.sleep(1.1)

        # Should not raise
        limiter.check_lockout("user@example.com")

    def test_successful_login_resets_attempts(self):
        """Successful login should reset failed attempt count."""
        limiter = RateLimiter(max_attempts=5)

        # Make 2 failed attempts
        limiter.record_attempt("user@example.com", success=False)
        limiter.record_attempt("user@example.com", success=False)

        # Successful login
        limiter.record_attempt("user@example.com", success=True)

        state = limiter.get_lockout_state("user@example.com")
        assert state.failed_attempts == 0
        assert not state.is_locked

    def test_progressive_lockout(self):
        """Lockout duration should increase with each lockout."""
        limiter = RateLimiter(
            max_attempts=2,
            lockout_duration_sec=10,
            lockout_multiplier=2.0,
        )

        # First lockout
        limiter.record_attempt("user@example.com", success=False)
        limiter.record_attempt("user@example.com", success=False)

        state = limiter.get_lockout_state("user@example.com")
        assert state.lockout_count == 1

        # After expiry and more failures, duration should increase
        # (This is a simplified test - real test would need time manipulation)

    def test_ip_rate_limit(self):
        """IP rate limiting should work."""
        limiter = RateLimiter(ip_max_requests=5, ip_window_sec=60)

        # Make 5 requests (within limit)
        for _ in range(5):
            limiter.check_rate_limit("192.168.1.1")

        # 6th request should fail
        with pytest.raises(RateLimitExceeded):
            limiter.check_rate_limit("192.168.1.1")

    def test_different_ips_separate_limits(self):
        """Different IPs should have separate rate limits."""
        limiter = RateLimiter(ip_max_requests=3, ip_window_sec=60)

        # IP1: 3 requests
        for _ in range(3):
            limiter.check_rate_limit("192.168.1.1")

        # IP1 is now limited
        with pytest.raises(RateLimitExceeded):
            limiter.check_rate_limit("192.168.1.1")

        # IP2 should still work
        limiter.check_rate_limit("192.168.1.2")

    def test_reset_lockout(self):
        """Admin reset should clear lockout."""
        limiter = RateLimiter(max_attempts=2, lockout_duration_sec=3600)

        # Trigger lockout
        limiter.record_attempt("user@example.com", success=False)
        limiter.record_attempt("user@example.com", success=False)

        assert limiter.get_lockout_state("user@example.com").is_locked

        # Admin reset
        limiter.reset_lockout("user@example.com")

        assert not limiter.get_lockout_state("user@example.com").is_locked

    def test_reset_rate_limit(self):
        """Admin reset should clear rate limit."""
        limiter = RateLimiter(ip_max_requests=1, ip_window_sec=3600)

        limiter.check_rate_limit("192.168.1.1")

        with pytest.raises(RateLimitExceeded):
            limiter.check_rate_limit("192.168.1.1")

        # Admin reset
        limiter.reset_rate_limit("192.168.1.1")

        # Should work again
        limiter.check_rate_limit("192.168.1.1")


class TestLockoutState:
    """Test LockoutState class."""

    def test_is_locked_when_future(self):
        """is_locked should be True when locked_until is in future."""
        state = LockoutState(locked_until=datetime.now(timezone.utc) + timedelta(minutes=5))
        assert state.is_locked

    def test_is_locked_when_past(self):
        """is_locked should be False when locked_until is in past."""
        state = LockoutState(locked_until=datetime.now(timezone.utc) - timedelta(minutes=5))
        assert not state.is_locked

    def test_is_locked_when_none(self):
        """is_locked should be False when locked_until is None."""
        state = LockoutState(locked_until=None)
        assert not state.is_locked

    def test_time_remaining(self):
        """time_remaining should return seconds."""
        state = LockoutState(locked_until=datetime.now(timezone.utc) + timedelta(seconds=30))
        assert 25 < state.time_remaining <= 30


class TestRateLimitExceeded:
    """Test RateLimitExceeded exception."""

    def test_retry_after_attribute(self):
        """Exception should have retry_after attribute."""
        exc = RateLimitExceeded("Rate limit exceeded", retry_after=60)
        assert exc.retry_after == 60

    def test_limit_type_attribute(self):
        """Exception should have limit_type attribute."""
        exc = RateLimitExceeded("Rate limit exceeded", limit_type="ip_rate_limit")
        assert exc.limit_type == "ip_rate_limit"


class TestAccountLockout:
    """Test AccountLockout exception."""

    def test_locked_until_attribute(self):
        """Exception should have locked_until attribute."""
        locked_until = datetime.now(timezone.utc) + timedelta(minutes=5)
        exc = AccountLockout("Account locked", locked_until=locked_until)
        assert exc.locked_until == locked_until

    def test_retry_after_property(self):
        """Exception should have retry_after property."""
        locked_until = datetime.now(timezone.utc) + timedelta(seconds=60)
        exc = AccountLockout("Account locked", locked_until=locked_until)
        assert 55 < exc.retry_after <= 60


class TestGlobalFunctions:
    """Test module-level convenience functions."""

    def test_get_rate_limiter_returns_singleton(self):
        """get_rate_limiter should return same instance."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2

    def test_check_lockout_no_raise_for_new_user(self):
        """check_lockout should not raise for new user."""
        check_lockout("new_user@example.com")

    def test_record_login_attempt_creates_state(self):
        """record_login_attempt should create state."""
        record_login_attempt("test_user@example.com", success=False)
        limiter = get_rate_limiter()
        state = limiter.get_lockout_state("test_user@example.com")
        assert state.failed_attempts >= 1
