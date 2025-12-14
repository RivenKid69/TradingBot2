# -*- coding: utf-8 -*-
"""
Rate Limiter and Account Lockout - WI-AUTH-01.

CLOUD ZONE ONLY.

Implements rate limiting and account lockout for authentication:
- Failed login attempt tracking
- Progressive lockout (exponential backoff)
- IP-based rate limiting
- User-based rate limiting

References:
- OWASP Authentication Cheat Sheet
- NIST 800-63B (account lockout)
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Any, Dict, Final, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

# Default rate limit settings
DEFAULT_MAX_ATTEMPTS: Final[int] = 5  # Max failed attempts before lockout
DEFAULT_LOCKOUT_DURATION_SEC: Final[int] = 300  # 5 minutes initial lockout
DEFAULT_LOCKOUT_MULTIPLIER: Final[float] = 2.0  # Exponential backoff
DEFAULT_MAX_LOCKOUT_DURATION_SEC: Final[int] = 3600  # Max 1 hour lockout
DEFAULT_ATTEMPT_WINDOW_SEC: Final[int] = 900  # 15 minute window for counting attempts

# IP rate limiting
DEFAULT_IP_MAX_REQUESTS: Final[int] = 100  # Max requests per IP per window
DEFAULT_IP_WINDOW_SEC: Final[int] = 60  # 1 minute window


# ============================================================================
# Exceptions
# ============================================================================

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: int = 0,
        limit_type: str = "rate_limit",
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_type = limit_type


class AccountLockout(Exception):
    """Raised when account is locked out."""

    def __init__(
        self,
        message: str,
        locked_until: datetime,
        failed_attempts: int = 0,
    ):
        super().__init__(message)
        self.locked_until = locked_until
        self.failed_attempts = failed_attempts

    @property
    def retry_after(self) -> int:
        """Seconds until lockout expires."""
        now = datetime.now(timezone.utc)
        if self.locked_until.tzinfo is None:
            locked_until = self.locked_until.replace(tzinfo=timezone.utc)
        else:
            locked_until = self.locked_until
        delta = locked_until - now
        return max(0, int(delta.total_seconds()))


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AttemptRecord:
    """Record of a login attempt."""
    timestamp: datetime
    success: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class LockoutState:
    """State of account lockout."""
    locked_until: Optional[datetime] = None
    failed_attempts: int = 0
    lockout_count: int = 0  # Number of times locked out
    last_attempt: Optional[datetime] = None
    attempts: List[AttemptRecord] = field(default_factory=list)

    @property
    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if self.locked_until is None:
            return False
        now = datetime.now(timezone.utc)
        locked = self.locked_until
        if locked.tzinfo is None:
            locked = locked.replace(tzinfo=timezone.utc)
        return now < locked

    @property
    def time_remaining(self) -> int:
        """Seconds remaining in lockout."""
        if not self.is_locked:
            return 0
        now = datetime.now(timezone.utc)
        locked = self.locked_until
        if locked.tzinfo is None:
            locked = locked.replace(tzinfo=timezone.utc)
        return max(0, int((locked - now).total_seconds()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_locked": self.is_locked,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None,
            "time_remaining": self.time_remaining,
            "failed_attempts": self.failed_attempts,
            "lockout_count": self.lockout_count,
        }


@dataclass
class RateLimitState:
    """State of rate limiting for an IP/user."""
    request_count: int = 0
    window_start: Optional[datetime] = None
    blocked_until: Optional[datetime] = None

    @property
    def is_blocked(self) -> bool:
        """Check if currently blocked."""
        if self.blocked_until is None:
            return False
        now = datetime.now(timezone.utc)
        blocked = self.blocked_until
        if blocked.tzinfo is None:
            blocked = blocked.replace(tzinfo=timezone.utc)
        return now < blocked

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_count": self.request_count,
            "is_blocked": self.is_blocked,
            "blocked_until": self.blocked_until.isoformat() if self.blocked_until else None,
        }


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """
    In-memory rate limiter with account lockout support.

    For production, consider using Redis for distributed rate limiting.

    Usage:
        limiter = RateLimiter()

        # Check before authentication
        limiter.check_rate_limit(ip_address="1.2.3.4")
        limiter.check_lockout(user_id="user123")

        # Record attempt result
        limiter.record_attempt(
            user_id="user123",
            success=False,
            ip_address="1.2.3.4",
        )
    """

    def __init__(
        self,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        lockout_duration_sec: int = DEFAULT_LOCKOUT_DURATION_SEC,
        lockout_multiplier: float = DEFAULT_LOCKOUT_MULTIPLIER,
        max_lockout_duration_sec: int = DEFAULT_MAX_LOCKOUT_DURATION_SEC,
        attempt_window_sec: int = DEFAULT_ATTEMPT_WINDOW_SEC,
        ip_max_requests: int = DEFAULT_IP_MAX_REQUESTS,
        ip_window_sec: int = DEFAULT_IP_WINDOW_SEC,
    ):
        """
        Initialize rate limiter.

        Args:
            max_attempts: Max failed attempts before lockout
            lockout_duration_sec: Initial lockout duration in seconds
            lockout_multiplier: Multiplier for progressive lockout
            max_lockout_duration_sec: Maximum lockout duration
            attempt_window_sec: Window for counting attempts
            ip_max_requests: Max requests per IP per window
            ip_window_sec: IP rate limit window in seconds
        """
        self._max_attempts = max_attempts
        self._lockout_duration_sec = lockout_duration_sec
        self._lockout_multiplier = lockout_multiplier
        self._max_lockout_duration_sec = max_lockout_duration_sec
        self._attempt_window_sec = attempt_window_sec
        self._ip_max_requests = ip_max_requests
        self._ip_window_sec = ip_window_sec

        # State storage (in-memory, use Redis for production)
        self._user_lockouts: Dict[str, LockoutState] = defaultdict(LockoutState)
        self._ip_rates: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = Lock()

    def check_lockout(self, user_id: str) -> None:
        """
        Check if user is locked out.

        Args:
            user_id: User identifier (email or ID)

        Raises:
            AccountLockout: If account is locked
        """
        with self._lock:
            state = self._user_lockouts.get(user_id)
            if state and state.is_locked:
                raise AccountLockout(
                    message=f"Account locked. Try again in {state.time_remaining} seconds.",
                    locked_until=state.locked_until,
                    failed_attempts=state.failed_attempts,
                )

    def check_rate_limit(self, ip_address: str) -> None:
        """
        Check IP rate limit.

        Args:
            ip_address: Client IP address

        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        with self._lock:
            state = self._ip_rates.get(ip_address)
            if state and state.is_blocked:
                retry_after = int((state.blocked_until - datetime.now(timezone.utc)).total_seconds())
                raise RateLimitExceeded(
                    message="Too many requests. Please try again later.",
                    retry_after=max(0, retry_after),
                    limit_type="ip_rate_limit",
                )

            # Check if in window
            now = datetime.now(timezone.utc)
            if state is None:
                state = RateLimitState()
                self._ip_rates[ip_address] = state

            if state.window_start is None:
                state.window_start = now
                state.request_count = 1
            else:
                window_start = state.window_start
                if window_start.tzinfo is None:
                    window_start = window_start.replace(tzinfo=timezone.utc)

                # Check if window expired
                if (now - window_start).total_seconds() > self._ip_window_sec:
                    state.window_start = now
                    state.request_count = 1
                else:
                    state.request_count += 1

                    if state.request_count > self._ip_max_requests:
                        # Block for window duration
                        state.blocked_until = now + timedelta(seconds=self._ip_window_sec)
                        raise RateLimitExceeded(
                            message="Too many requests. Please try again later.",
                            retry_after=self._ip_window_sec,
                            limit_type="ip_rate_limit",
                        )

    def record_attempt(
        self,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """
        Record a login attempt.

        Args:
            user_id: User identifier
            success: Whether attempt was successful
            ip_address: Client IP address
            user_agent: Client user agent
        """
        with self._lock:
            state = self._user_lockouts[user_id]
            now = datetime.now(timezone.utc)

            # Record attempt
            attempt = AttemptRecord(
                timestamp=now,
                success=success,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            state.attempts.append(attempt)
            state.last_attempt = now

            # Prune old attempts
            cutoff = now - timedelta(seconds=self._attempt_window_sec)
            state.attempts = [
                a for a in state.attempts
                if (a.timestamp.replace(tzinfo=timezone.utc) if a.timestamp.tzinfo is None else a.timestamp) > cutoff
            ]

            if success:
                # Reset on successful login
                state.failed_attempts = 0
                state.locked_until = None
            else:
                # Increment failed attempts
                state.failed_attempts += 1

                # Check if lockout threshold reached
                if state.failed_attempts >= self._max_attempts:
                    # Calculate lockout duration with exponential backoff
                    duration = self._lockout_duration_sec * (
                        self._lockout_multiplier ** state.lockout_count
                    )
                    duration = min(duration, self._max_lockout_duration_sec)

                    state.locked_until = now + timedelta(seconds=duration)
                    state.lockout_count += 1

    def get_lockout_state(self, user_id: str) -> LockoutState:
        """Get lockout state for a user."""
        with self._lock:
            return self._user_lockouts.get(user_id, LockoutState())

    def get_rate_limit_state(self, ip_address: str) -> RateLimitState:
        """Get rate limit state for an IP."""
        with self._lock:
            return self._ip_rates.get(ip_address, RateLimitState())

    def reset_lockout(self, user_id: str) -> None:
        """Reset lockout for a user (admin action)."""
        with self._lock:
            if user_id in self._user_lockouts:
                state = self._user_lockouts[user_id]
                state.locked_until = None
                state.failed_attempts = 0

    def reset_rate_limit(self, ip_address: str) -> None:
        """Reset rate limit for an IP (admin action)."""
        with self._lock:
            if ip_address in self._ip_rates:
                del self._ip_rates[ip_address]

    def cleanup_expired(self) -> int:
        """
        Clean up expired entries.

        Returns:
            Number of entries cleaned up
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            cleaned = 0

            # Clean up lockouts
            expired_users = []
            for user_id, state in self._user_lockouts.items():
                if state.locked_until:
                    locked = state.locked_until
                    if locked.tzinfo is None:
                        locked = locked.replace(tzinfo=timezone.utc)
                    if now > locked:
                        # Keep state but clear lockout
                        state.locked_until = None
                        cleaned += 1

            # Clean up IP rates
            expired_ips = []
            for ip, state in self._ip_rates.items():
                if state.window_start:
                    window = state.window_start
                    if window.tzinfo is None:
                        window = window.replace(tzinfo=timezone.utc)
                    if (now - window).total_seconds() > self._ip_window_sec * 2:
                        expired_ips.append(ip)

            for ip in expired_ips:
                del self._ip_rates[ip]
                cleaned += 1

            return cleaned


# ============================================================================
# Global Instance
# ============================================================================

# Singleton instance
_default_limiter: Optional[RateLimiter] = None
_limiter_lock = Lock()


def get_rate_limiter() -> RateLimiter:
    """Get the default rate limiter instance."""
    global _default_limiter
    with _limiter_lock:
        if _default_limiter is None:
            _default_limiter = RateLimiter()
        return _default_limiter


def check_lockout(user_id: str) -> None:
    """Check if user is locked out."""
    get_rate_limiter().check_lockout(user_id)


def check_rate_limit(ip_address: str) -> None:
    """Check IP rate limit."""
    get_rate_limiter().check_rate_limit(ip_address)


def record_login_attempt(
    user_id: str,
    success: bool,
    ip_address: Optional[str] = None,
) -> None:
    """Record a login attempt."""
    get_rate_limiter().record_attempt(user_id, success, ip_address)
