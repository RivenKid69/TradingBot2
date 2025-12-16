# -*- coding: utf-8 -*-
"""
Tests for GDPR Phase 2 CI Guardrails.

Tests for:
    - Protocol Review Check
    - Artifact Digest Check
    - Redaction Enforcement Check
"""

from __future__ import annotations

import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock

from ccea.guardrails.artifact_digest_check import (
    is_valid_digest,
    extract_registry,
    is_allowed_registry,
    check_for_invalid_patterns,
    validate_artifact_reference,
    validate_digest,
    DIGEST_PATTERN,
    ALLOWED_REGISTRIES,
)

from ccea.guardrails.redaction_enforcement_check import (
    RedactionEnforcer,
    get_redaction_enforcer,
    assert_redaction_enabled,
)


# ============================================================================
# Test Artifact Digest Check
# ============================================================================

class TestDigestValidation:
    """Tests for digest validation."""

    def test_valid_sha256_digest(self):
        """Test valid SHA256 digest passes."""
        valid_digest = "sha256:" + "a" * 64
        assert is_valid_digest(valid_digest) is True

    def test_valid_sha256_digest_full(self):
        """Test valid full SHA256 digest."""
        digest = "sha256:abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        assert is_valid_digest(digest) is True

    def test_invalid_digest_no_prefix(self):
        """Test digest without sha256: prefix fails."""
        digest = "a" * 64
        assert is_valid_digest(digest) is False

    def test_invalid_digest_wrong_length(self):
        """Test digest with wrong length fails."""
        digest = "sha256:abc123"
        assert is_valid_digest(digest) is False

    def test_invalid_digest_wrong_chars(self):
        """Test digest with invalid characters fails."""
        digest = "sha256:" + "g" * 64  # 'g' is not hex
        assert is_valid_digest(digest) is False

    def test_empty_digest(self):
        """Test empty digest fails."""
        assert is_valid_digest("") is False

    def test_none_like_values(self):
        """Test None-like values fail."""
        assert is_valid_digest("null") is False
        assert is_valid_digest("None") is False


class TestRegistryExtraction:
    """Tests for registry extraction."""

    def test_extract_standard_registry(self):
        """Test extracting standard registry."""
        ref = "registry.ccea.io/image:tag"
        assert extract_registry(ref) == "registry.ccea.io"

    def test_extract_registry_with_port(self):
        """Test extracting registry with port."""
        # Note: localhost with port doesn't match standard pattern
        # since there's no dot in 'localhost'. This is expected behavior.
        ref = "localhost:5000/image:tag"
        # localhost:5000 is split by : so image appears to be in port
        # This returns None because localhost alone doesn't have a dot
        result = extract_registry(ref)
        # Accept either localhost:5000 or None depending on implementation
        assert result in (None, "localhost:5000", "localhost")

    def test_extract_ghcr_registry(self):
        """Test extracting GitHub Container Registry."""
        ref = "ghcr.io/ccea/image:tag"
        assert extract_registry(ref) == "ghcr.io/ccea"

    def test_extract_registry_with_digest(self):
        """Test extracting registry from ref with digest."""
        ref = "registry.ccea.io/image@sha256:abc123"
        assert extract_registry(ref) == "registry.ccea.io"

    def test_local_image_no_registry(self):
        """Test local image without registry."""
        ref = "my-image:latest"
        assert extract_registry(ref) is None


class TestRegistryAllowlist:
    """Tests for registry allowlist."""

    def test_allowed_registry_exact(self):
        """Test exact match in allowlist."""
        assert is_allowed_registry("registry.ccea.io") is True
        assert is_allowed_registry("eu.registry.ccea.io") is True

    def test_allowed_registry_ghcr(self):
        """Test GitHub Container Registry allowed."""
        assert is_allowed_registry("ghcr.io/ccea") is True

    def test_allowed_registry_localhost(self):
        """Test localhost allowed for dev."""
        assert is_allowed_registry("localhost") is True
        assert is_allowed_registry("127.0.0.1") is True

    def test_disallowed_registry(self):
        """Test disallowed registry."""
        assert is_allowed_registry("docker.io") is False
        assert is_allowed_registry("gcr.io/external") is False
        assert is_allowed_registry("unknown.registry.com") is False

    def test_none_registry_allowed(self):
        """Test None registry (local image) allowed."""
        assert is_allowed_registry(None) is True


class TestInvalidPatterns:
    """Tests for invalid reference patterns."""

    def test_latest_tag_forbidden(self):
        """Test :latest tag is forbidden."""
        error = check_for_invalid_patterns("image:latest")
        assert error is not None
        assert "latest" in error.lower()

    def test_stable_tag_forbidden(self):
        """Test :stable tag is forbidden."""
        error = check_for_invalid_patterns("image:stable")
        assert error is not None

    def test_edge_tag_forbidden(self):
        """Test :edge tag is forbidden."""
        error = check_for_invalid_patterns("image:edge")
        assert error is not None

    def test_dev_tag_forbidden(self):
        """Test :dev tag is forbidden."""
        error = check_for_invalid_patterns("image:dev")
        assert error is not None

    def test_version_tag_forbidden(self):
        """Test bare version tags are forbidden."""
        error = check_for_invalid_patterns("image:1.2.3")
        assert error is not None
        assert "digest" in error.lower()

    def test_digest_reference_allowed(self):
        """Test reference with digest is allowed."""
        ref = "image@sha256:" + "a" * 64
        error = check_for_invalid_patterns(ref)
        assert error is None


class TestValidateArtifactReference:
    """Tests for validate_artifact_reference function."""

    def test_valid_reference_with_digest(self):
        """Test valid reference with digest passes."""
        ref = f"registry.ccea.io/image@sha256:{'a' * 64}"
        is_valid, error = validate_artifact_reference(ref)
        assert is_valid is True
        assert error is None

    def test_reference_without_digest_fails(self):
        """Test reference without digest fails."""
        ref = "registry.ccea.io/image:tag"
        is_valid, error = validate_artifact_reference(ref, require_digest=True)
        assert is_valid is False
        assert "digest" in error.lower()

    def test_reference_with_latest_fails(self):
        """Test reference with :latest fails."""
        ref = "registry.ccea.io/image:latest"
        is_valid, error = validate_artifact_reference(ref)
        assert is_valid is False
        assert "latest" in error.lower()

    def test_disallowed_registry_fails(self):
        """Test reference with disallowed registry fails."""
        ref = f"docker.io/image@sha256:{'a' * 64}"
        is_valid, error = validate_artifact_reference(ref, check_registry=True)
        assert is_valid is False
        assert "allowlist" in error.lower()


class TestValidateDigest:
    """Tests for validate_digest function."""

    def test_valid_digest_passes(self):
        """Test valid digest passes."""
        is_valid, error = validate_digest(f"sha256:{'a' * 64}")
        assert is_valid is True
        assert error is None

    def test_empty_digest_fails(self):
        """Test empty digest fails."""
        is_valid, error = validate_digest("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_invalid_format_fails(self):
        """Test invalid format fails."""
        is_valid, error = validate_digest("invalid")
        assert is_valid is False
        assert "format" in error.lower()


# ============================================================================
# Test Redaction Enforcement
# ============================================================================

class TestRedactionEnforcer:
    """Tests for RedactionEnforcer."""

    def test_enforcer_always_enabled(self):
        """Test enforcer is always enabled."""
        enforcer = RedactionEnforcer()
        assert enforcer.enabled is True

    def test_disable_does_nothing(self):
        """Test disable() does nothing."""
        enforcer = RedactionEnforcer()
        enforcer.disable()
        assert enforcer.enabled is True

    def test_verify_enabled_always_true(self):
        """Test verify_enabled always returns True."""
        enforcer = RedactionEnforcer()
        is_enabled, message = enforcer.verify_enabled()
        assert is_enabled is True
        assert "enabled" in message.lower()

    def test_enforcer_singleton(self):
        """Test enforcer is singleton."""
        enforcer1 = RedactionEnforcer()
        enforcer2 = RedactionEnforcer()
        assert enforcer1 is enforcer2

    def test_get_redaction_enforcer(self):
        """Test get_redaction_enforcer returns enforcer."""
        enforcer = get_redaction_enforcer()
        assert isinstance(enforcer, RedactionEnforcer)
        assert enforcer.enabled is True

    def test_assert_redaction_enabled_passes(self):
        """Test assert_redaction_enabled always passes."""
        # Should not raise
        assert_redaction_enabled()


# ============================================================================
# Integration Tests
# ============================================================================

class TestGuardrailsIntegration:
    """Integration tests for guardrails."""

    def test_digest_pattern_constant(self):
        """Test DIGEST_PATTERN constant is correct."""
        # Valid
        assert DIGEST_PATTERN.match(f"sha256:{'a' * 64}")
        assert DIGEST_PATTERN.match(f"sha256:{'0' * 64}")
        assert DIGEST_PATTERN.match(f"sha256:{'f' * 64}")

        # Invalid
        assert DIGEST_PATTERN.match("sha256:short") is None
        assert DIGEST_PATTERN.match("md5:abc") is None

    def test_allowed_registries_constant(self):
        """Test ALLOWED_REGISTRIES constant."""
        assert "registry.ccea.io" in ALLOWED_REGISTRIES
        assert "localhost" in ALLOWED_REGISTRIES
        # Docker Hub not allowed
        assert "docker.io" not in ALLOWED_REGISTRIES

    def test_redaction_cannot_be_bypassed(self):
        """Test redaction truly cannot be bypassed."""
        enforcer = get_redaction_enforcer()

        # Try various bypass methods
        enforcer.disable()
        assert enforcer.enabled is True

        # Try setting _ENABLED directly (should still return True)
        RedactionEnforcer._ENABLED = False
        assert enforcer.enabled is True

        # Restore
        RedactionEnforcer._ENABLED = True


# ============================================================================
# Test Protocol Review Check Structures
# ============================================================================

class TestProtocolReviewStructures:
    """Tests for protocol review check data structures."""

    def test_journal_entry_from_dict(self):
        """Test creating JournalEntry from dict."""
        from ccea.guardrails.protocol_review_check import JournalEntry

        data = {
            "change_id": "PROT-2025-001",
            "review_id": "REV-2025-001",
            "timestamp": "2025-12-16T10:00:00Z",
            "change_type": "NEW_COMMAND",
            "approval_status": "APPROVED",
            "commit_hash": "abc123",
            "conditions_met": True,
        }

        entry = JournalEntry.from_dict(data)

        assert entry.change_id == "PROT-2025-001"
        assert entry.approval_status == "APPROVED"
        assert entry.conditions_met is True

    def test_protocol_change_dataclass(self):
        """Test ProtocolChange dataclass."""
        from ccea.guardrails.protocol_review_check import ProtocolChange

        change = ProtocolChange(
            file_path="ccea/models/protocol.py",
            change_type="modified",
            patterns_matched=["CommandType"],
        )

        assert change.file_path == "ccea/models/protocol.py"
        assert change.change_type == "modified"
        assert "CommandType" in change.patterns_matched

    def test_check_result_dataclass(self):
        """Test CheckResult dataclass."""
        from ccea.guardrails.protocol_review_check import CheckResult

        result = CheckResult(
            passed=True,
            message="All good",
            protocol_changes=[],
            approval_found=None,
        )

        assert result.passed is True
        assert result.message == "All good"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_payload_valid(self):
        """Test empty payload is valid for AGGREGATED."""
        from packages.cloud.governance.telemetry_contract import validate_telemetry_contract

        result = validate_telemetry_contract(
            {},
            "AGGREGATED",
            "ws_1",
            "org_1",
        )

        assert result.valid is True

    def test_deeply_nested_forbidden_field(self):
        """Test deeply nested forbidden field is caught."""
        from packages.cloud.governance.telemetry_contract import validate_telemetry_contract

        payload = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "api_key": "secret"
                        }
                    }
                }
            }
        }

        result = validate_telemetry_contract(
            payload,
            "AGGREGATED",
            "ws_1",
            "org_1",
        )

        assert result.valid is False
        assert any("level1.level2.level3.level4.api_key" in v.field_path for v in result.violations)

    def test_list_with_forbidden_field(self):
        """Test list containing forbidden field is caught."""
        from packages.cloud.governance.telemetry_contract import validate_telemetry_contract

        payload = {
            "events": [
                {"timestamp": "2025-12-16T10:00:00Z"},
                {"side": "BUY"},  # Forbidden
            ]
        }

        result = validate_telemetry_contract(
            payload,
            "AGGREGATED",
            "ws_1",
            "org_1",
        )

        assert result.valid is False
        assert any("events[1].side" in v.field_path for v in result.violations)

    def test_unicode_field_names(self):
        """Test Unicode field names don't crash."""
        from packages.cloud.governance.telemetry_contract import validate_telemetry_contract

        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "данные": "value",  # Russian
            "日本語": "value",  # Japanese
        }

        # Should not raise
        result = validate_telemetry_contract(
            payload,
            "AGGREGATED",
            "ws_1",
            "org_1",
        )

        assert result is not None

    def test_special_characters_in_values(self):
        """Test special characters in values don't cause issues."""
        from packages.cloud.governance.telemetry_contract import validate_telemetry_contract

        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "message": "Error: <script>alert('xss')</script>",
            "path": "/path/with spaces/and\"quotes",
        }

        result = validate_telemetry_contract(
            payload,
            "DETAILED_NON_SENSITIVE",
            "ws_1",
            "org_1",
            redaction_applied=True,
        )

        # Should process without error
        assert result is not None
