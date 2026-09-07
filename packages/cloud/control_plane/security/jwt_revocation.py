# -*- coding: utf-8 -*-
"""
JWT Revocation - WI-AUTH-01.

CLOUD ZONE ONLY.

Implements JWT token revocation via JTI (JWT ID) blocklist.
This allows immediate token invalidation on logout.

For production, use Redis or database storage.

References:
- RFC 7519 (JWT)
- OWASP JWT Cheat Sheet
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, Final, List, Optional, Set


# ============================================================================
# Constants
# ============================================================================

# Maximum entries in blocklist (LRU eviction)
DEFAULT_MAX_BLOCKLIST_SIZE: Final[int] = 100000

# Default token TTL (24 hours)
DEFAULT_TOKEN_TTL_HOURS: Final[int] = 24


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class RevokedToken:
    """Record of a revoked token."""

    jti: str
    revoked_at: datetime
    expires_at: datetime
    reason: str = "logout"
    user_id: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if token has naturally expired (can be cleaned up)."""
        now = datetime.now(timezone.utc)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return now > expires

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "jti": self.jti,
            "revoked_at": self.revoked_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "reason": self.reason,
            "user_id": self.user_id,
        }


@dataclass
class BlocklistStats:
    """Statistics about the blocklist."""

    total_entries: int = 0
    active_entries: int = 0
    expired_entries: int = 0
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_entries": self.total_entries,
            "active_entries": self.active_entries,
            "expired_entries": self.expired_entries,
            "oldest_entry": self.oldest_entry.isoformat() if self.oldest_entry else None,
            "newest_entry": self.newest_entry.isoformat() if self.newest_entry else None,
        }


# ============================================================================
# JTI Blocklist
# ============================================================================


class JTIBlocklist:
    """
    In-memory JTI blocklist with LRU eviction.

    DEPLOYMENT CONSIDERATIONS:
        Single-instance: In-memory storage is acceptable.
        Multi-instance: Use Redis or database for consistent revocation.

    PRODUCTION REQUIREMENTS (multi-instance deployments):
        1. Replace in-memory storage with Redis/database backend
        2. Configure replication for revocation propagation
        3. Monitor revocation distribution latency

    CONTROL ARTIFACTS:
        - docs/security/JWT_REVOCATION_REQUIREMENTS.md
        - Metrics: jwt_revoked_count, jwt_revocation_check_count

    Usage:
        blocklist = JTIBlocklist()

        # Revoke a token
        blocklist.revoke(jti="abc123", expires_at=token_expiry)

        # Check if revoked
        if blocklist.is_revoked(jti="abc123"):
            raise AuthError("Token revoked")
    """

    def __init__(
        self,
        *,
        max_size: int = DEFAULT_MAX_BLOCKLIST_SIZE,
        token_ttl_hours: int = DEFAULT_TOKEN_TTL_HOURS,
    ):
        """
        Initialize blocklist.

        Args:
            max_size: Maximum entries (LRU eviction when exceeded)
            token_ttl_hours: Default token TTL for expiry calculation
        """
        self._max_size = max_size
        self._token_ttl_hours = token_ttl_hours

        # Use OrderedDict for LRU eviction
        self._blocklist: OrderedDict[str, RevokedToken] = OrderedDict()
        self._lock = Lock()

        # User -> JTIs mapping for bulk revocation
        self._user_tokens: Dict[str, Set[str]] = {}

    def revoke(
        self,
        jti: str,
        *,
        expires_at: Optional[datetime] = None,
        reason: str = "logout",
        user_id: Optional[str] = None,
    ) -> RevokedToken:
        """
        Revoke a token by JTI.

        Args:
            jti: JWT ID to revoke
            expires_at: When the token naturally expires (for cleanup)
            reason: Reason for revocation
            user_id: User ID who owned the token

        Returns:
            RevokedToken record
        """
        now = datetime.now(timezone.utc)

        # Default expiry if not provided
        if expires_at is None:
            expires_at = now + timedelta(hours=self._token_ttl_hours)
        elif expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        token = RevokedToken(
            jti=jti,
            revoked_at=now,
            expires_at=expires_at,
            reason=reason,
            user_id=user_id,
        )

        with self._lock:
            # Add to blocklist
            self._blocklist[jti] = token

            # Move to end (most recently used)
            self._blocklist.move_to_end(jti)

            # Track user's tokens
            if user_id:
                if user_id not in self._user_tokens:
                    self._user_tokens[user_id] = set()
                self._user_tokens[user_id].add(jti)

            # LRU eviction if needed
            while len(self._blocklist) > self._max_size:
                oldest_jti, oldest_token = self._blocklist.popitem(last=False)
                # Remove from user mapping
                if oldest_token.user_id and oldest_token.user_id in self._user_tokens:
                    self._user_tokens[oldest_token.user_id].discard(oldest_jti)

        return token

    def is_revoked(self, jti: str) -> bool:
        """
        Check if a token is revoked.

        Args:
            jti: JWT ID to check

        Returns:
            True if token is revoked and not yet expired
        """
        with self._lock:
            token = self._blocklist.get(jti)
            if token is None:
                return False

            # Check if token has naturally expired (can be removed)
            if token.is_expired:
                return False

            return True

    def get_revocation(self, jti: str) -> Optional[RevokedToken]:
        """
        Get revocation record for a JTI.

        Args:
            jti: JWT ID to look up

        Returns:
            RevokedToken if found, None otherwise
        """
        with self._lock:
            return self._blocklist.get(jti)

    def revoke_all_user_tokens(
        self,
        user_id: str,
        *,
        reason: str = "logout_all",
        expires_at: Optional[datetime] = None,
    ) -> int:
        """
        Revoke all tokens for a user.

        This is useful when user changes password or for security events.

        Note: This only revokes tokens we know about. For true logout-all,
        you need to store all issued tokens or use short-lived tokens.

        Args:
            user_id: User ID to revoke tokens for
            reason: Reason for revocation
            expires_at: When tokens expire (for cleanup)

        Returns:
            Number of tokens revoked
        """
        with self._lock:
            if user_id not in self._user_tokens:
                return 0

            jtis = list(self._user_tokens[user_id])
            for jti in jtis:
                if jti in self._blocklist:
                    self._blocklist[jti].reason = reason

            return len(jtis)

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from blocklist.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired = []
            for jti, token in self._blocklist.items():
                if token.is_expired:
                    expired.append(jti)

            for jti in expired:
                token = self._blocklist.pop(jti)
                # Remove from user mapping
                if token.user_id and token.user_id in self._user_tokens:
                    self._user_tokens[token.user_id].discard(jti)

            return len(expired)

    def get_stats(self) -> BlocklistStats:
        """Get blocklist statistics."""
        with self._lock:
            active = 0
            expired = 0
            oldest = None
            newest = None

            for token in self._blocklist.values():
                if token.is_expired:
                    expired += 1
                else:
                    active += 1

                if oldest is None or token.revoked_at < oldest:
                    oldest = token.revoked_at
                if newest is None or token.revoked_at > newest:
                    newest = token.revoked_at

            return BlocklistStats(
                total_entries=len(self._blocklist),
                active_entries=active,
                expired_entries=expired,
                oldest_entry=oldest,
                newest_entry=newest,
            )

    def clear(self) -> int:
        """
        Clear all entries (admin action).

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._blocklist)
            self._blocklist.clear()
            self._user_tokens.clear()
            return count


# ============================================================================
# Global Instance
# ============================================================================

# Singleton instance
_default_blocklist: Optional[JTIBlocklist] = None
_blocklist_lock = Lock()


def get_blocklist() -> JTIBlocklist:
    """Get the default blocklist instance."""
    global _default_blocklist
    with _blocklist_lock:
        if _default_blocklist is None:
            _default_blocklist = JTIBlocklist()
        return _default_blocklist


def revoke_token(
    jti: str,
    *,
    expires_at: Optional[datetime] = None,
    reason: str = "logout",
    user_id: Optional[str] = None,
) -> RevokedToken:
    """
    Revoke a token by JTI.

    Args:
        jti: JWT ID to revoke
        expires_at: When the token naturally expires
        reason: Reason for revocation
        user_id: User ID who owned the token

    Returns:
        RevokedToken record
    """
    return get_blocklist().revoke(
        jti,
        expires_at=expires_at,
        reason=reason,
        user_id=user_id,
    )


def is_token_revoked(jti: str) -> bool:
    """
    Check if a token is revoked.

    Args:
        jti: JWT ID to check

    Returns:
        True if token is revoked
    """
    return get_blocklist().is_revoked(jti)


def revoke_all_user_tokens(
    user_id: str,
    *,
    reason: str = "logout_all",
) -> int:
    """
    Revoke all tokens for a user.

    Args:
        user_id: User ID
        reason: Reason for revocation

    Returns:
        Number of tokens revoked
    """
    return get_blocklist().revoke_all_user_tokens(user_id, reason=reason)
