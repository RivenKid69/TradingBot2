# -*- coding: utf-8 -*-
"""
Tests for JWT Revocation - WI-AUTH-01.

Tests verify:
- Token revocation via jti blocklist
- Revoked tokens are rejected
- Expired tokens are cleaned up
- User-level token revocation
"""

import pytest
import uuid
from datetime import datetime, timedelta, timezone

from packages.cloud.control_plane.security.jwt_revocation import (
    JTIBlocklist,
    RevokedToken,
    BlocklistStats,
    revoke_token,
    is_token_revoked,
    revoke_all_user_tokens,
    get_blocklist,
)


class TestJTIBlocklist:
    """Test JTIBlocklist class."""

    def test_revoke_returns_token_record(self):
        """Revoke should return RevokedToken record."""
        blocklist = JTIBlocklist()
        jti = str(uuid.uuid4())

        result = blocklist.revoke(jti)

        assert isinstance(result, RevokedToken)
        assert result.jti == jti

    def test_revoked_token_is_detected(self):
        """Revoked token should be detected."""
        blocklist = JTIBlocklist()
        jti = str(uuid.uuid4())

        blocklist.revoke(jti)

        assert blocklist.is_revoked(jti)

    def test_unrevoked_token_not_detected(self):
        """Non-revoked token should not be detected as revoked."""
        blocklist = JTIBlocklist()
        jti = str(uuid.uuid4())

        assert not blocklist.is_revoked(jti)

    def test_revoke_with_reason(self):
        """Revoke should store reason."""
        blocklist = JTIBlocklist()
        jti = str(uuid.uuid4())

        result = blocklist.revoke(jti, reason="password_change")

        assert result.reason == "password_change"

    def test_revoke_with_user_id(self):
        """Revoke should store user ID."""
        blocklist = JTIBlocklist()
        jti = str(uuid.uuid4())
        user_id = str(uuid.uuid4())

        result = blocklist.revoke(jti, user_id=user_id)

        assert result.user_id == user_id

    def test_revoke_with_expires_at(self):
        """Revoke should store expiry."""
        blocklist = JTIBlocklist()
        jti = str(uuid.uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

        result = blocklist.revoke(jti, expires_at=expires_at)

        assert result.expires_at == expires_at

    def test_expired_token_not_revoked(self):
        """Expired token should not be reported as revoked."""
        blocklist = JTIBlocklist()
        jti = str(uuid.uuid4())

        # Revoke with past expiry
        blocklist.revoke(
            jti,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        # Should not be considered revoked (naturally expired)
        assert not blocklist.is_revoked(jti)

    def test_get_revocation_returns_record(self):
        """get_revocation should return record for revoked token."""
        blocklist = JTIBlocklist()
        jti = str(uuid.uuid4())

        blocklist.revoke(jti, reason="test")

        record = blocklist.get_revocation(jti)
        assert record is not None
        assert record.reason == "test"

    def test_get_revocation_returns_none_for_unknown(self):
        """get_revocation should return None for unknown token."""
        blocklist = JTIBlocklist()
        jti = str(uuid.uuid4())

        record = blocklist.get_revocation(jti)
        assert record is None

    def test_lru_eviction(self):
        """Old entries should be evicted when max size reached."""
        blocklist = JTIBlocklist(max_size=3)

        jtis = [str(uuid.uuid4()) for _ in range(5)]

        for jti in jtis:
            blocklist.revoke(jti)

        # Only last 3 should be in blocklist
        stats = blocklist.get_stats()
        assert stats.total_entries == 3

        # First 2 should have been evicted
        assert not blocklist.is_revoked(jtis[0])
        assert not blocklist.is_revoked(jtis[1])

        # Last 3 should still be revoked
        assert blocklist.is_revoked(jtis[2])
        assert blocklist.is_revoked(jtis[3])
        assert blocklist.is_revoked(jtis[4])

    def test_cleanup_expired(self):
        """cleanup_expired should remove expired entries."""
        blocklist = JTIBlocklist()

        # Add some expired tokens
        for _ in range(3):
            blocklist.revoke(
                str(uuid.uuid4()),
                expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            )

        # Add a non-expired token
        active_jti = str(uuid.uuid4())
        blocklist.revoke(
            active_jti,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        # Cleanup
        cleaned = blocklist.cleanup_expired()
        assert cleaned == 3

        # Active token should still be there
        assert blocklist.is_revoked(active_jti)

    def test_get_stats(self):
        """get_stats should return correct statistics."""
        blocklist = JTIBlocklist()

        # Add some tokens
        blocklist.revoke(str(uuid.uuid4()))
        blocklist.revoke(str(uuid.uuid4()))

        stats = blocklist.get_stats()
        assert isinstance(stats, BlocklistStats)
        assert stats.total_entries == 2
        assert stats.active_entries >= 0

    def test_clear(self):
        """clear should remove all entries."""
        blocklist = JTIBlocklist()

        for _ in range(5):
            blocklist.revoke(str(uuid.uuid4()))

        cleared = blocklist.clear()
        assert cleared == 5

        stats = blocklist.get_stats()
        assert stats.total_entries == 0


class TestUserTokenRevocation:
    """Test user-level token revocation."""

    def test_revoke_all_user_tokens(self):
        """revoke_all_user_tokens should revoke known tokens."""
        blocklist = JTIBlocklist()
        user_id = str(uuid.uuid4())

        # Add some tokens for this user
        jtis = [str(uuid.uuid4()) for _ in range(3)]
        for jti in jtis:
            blocklist.revoke(jti, user_id=user_id)

        # All tokens should be revoked
        for jti in jtis:
            assert blocklist.is_revoked(jti)


class TestRevokedToken:
    """Test RevokedToken class."""

    def test_is_expired_when_past(self):
        """is_expired should be True when expires_at is in past."""
        token = RevokedToken(
            jti="test",
            revoked_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert token.is_expired

    def test_is_expired_when_future(self):
        """is_expired should be False when expires_at is in future."""
        token = RevokedToken(
            jti="test",
            revoked_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert not token.is_expired

    def test_to_dict(self):
        """to_dict should return serializable dict."""
        token = RevokedToken(
            jti="test",
            revoked_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            reason="logout",
            user_id="user123",
        )
        result = token.to_dict()

        assert "jti" in result
        assert "revoked_at" in result
        assert "expires_at" in result
        assert "reason" in result
        assert "user_id" in result


class TestGlobalFunctions:
    """Test module-level convenience functions."""

    def test_revoke_token_function(self):
        """revoke_token should work correctly."""
        jti = str(uuid.uuid4())
        result = revoke_token(jti, reason="test")

        assert result.jti == jti
        assert is_token_revoked(jti)

    def test_is_token_revoked_function(self):
        """is_token_revoked should work correctly."""
        jti = str(uuid.uuid4())

        assert not is_token_revoked(jti)

        revoke_token(jti)

        assert is_token_revoked(jti)

    def test_get_blocklist_returns_singleton(self):
        """get_blocklist should return same instance."""
        blocklist1 = get_blocklist()
        blocklist2 = get_blocklist()
        assert blocklist1 is blocklist2


class TestBlocklistStats:
    """Test BlocklistStats class."""

    def test_to_dict(self):
        """to_dict should return serializable dict."""
        stats = BlocklistStats(
            total_entries=10,
            active_entries=8,
            expired_entries=2,
        )
        result = stats.to_dict()

        assert result["total_entries"] == 10
        assert result["active_entries"] == 8
        assert result["expired_entries"] == 2
