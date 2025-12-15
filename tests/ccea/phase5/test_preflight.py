# -*- coding: utf-8 -*-
"""
Tests for Pre-flight Checker.

Design Doc D1: Pre-flight checks before start/upgrade.
"""

import pytest
import hashlib
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from unittest.mock import MagicMock, patch

from packages.agent.daemon.preflight import (
    PreflightChecker,
    PreflightConfig,
    PreflightResult,
    PreflightCheck,
    PreflightCheckType,
    PreflightCheckResult,
)


class TestPreflightCheck:
    """Tests for PreflightCheck dataclass."""

    def test_create_preflight_check(self):
        """Test creating a preflight check."""
        check = PreflightCheck(
            check_type=PreflightCheckType.VAULT_UNLOCKED,
            result=PreflightCheckResult.PASSED,
            message="Vault is unlocked",
            duration_ms=10.5,
        )

        assert check.check_type == PreflightCheckType.VAULT_UNLOCKED
        assert check.result == PreflightCheckResult.PASSED
        assert check.required is True

    def test_check_to_dict(self):
        """Test serialization."""
        check = PreflightCheck(
            check_type=PreflightCheckType.TIME_SYNC,
            result=PreflightCheckResult.WARNING,
            message="Slight drift detected",
            details={"drift_ms": 500},
        )

        d = check.to_dict()
        assert d["check_type"] == "TIME_SYNC"
        assert d["result"] == "warning"


class TestPreflightResult:
    """Tests for PreflightResult dataclass."""

    def test_create_result(self):
        """Test creating preflight result."""
        result = PreflightResult(run_id="test-123")

        assert result.passed is False
        assert result.errors == []
        assert result.warnings == []

    def test_counts(self):
        """Test count properties."""
        result = PreflightResult()
        result.checks = [
            PreflightCheck(
                check_type=PreflightCheckType.VAULT_UNLOCKED,
                result=PreflightCheckResult.PASSED,
                message="OK",
            ),
            PreflightCheck(
                check_type=PreflightCheckType.TIME_SYNC,
                result=PreflightCheckResult.FAILED,
                message="Failed",
            ),
            PreflightCheck(
                check_type=PreflightCheckType.NETWORK_CONNECTIVITY,
                result=PreflightCheckResult.WARNING,
                message="Warning",
            ),
        ]

        assert result.passed_count == 1
        assert result.failed_count == 1
        assert result.warning_count == 1

    def test_evidence_hash(self):
        """Test evidence hash computation."""
        result = PreflightResult()
        result.checks = [
            PreflightCheck(
                check_type=PreflightCheckType.VAULT_UNLOCKED,
                result=PreflightCheckResult.PASSED,
                message="OK",
            ),
        ]
        result.passed = True

        hash1 = result.get_evidence_hash()
        assert len(hash1) == 64  # SHA256 hex

        # Different result should have different hash
        result2 = PreflightResult()
        result2.passed = False
        hash2 = result2.get_evidence_hash()
        assert hash1 != hash2


class TestPreflightConfig:
    """Tests for PreflightConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PreflightConfig()

        assert config.max_time_drift_seconds == 5.0
        assert config.require_broker_connectivity is True
        assert config.min_schema_version == "1.0.0"

    def test_custom_config(self):
        """Test custom configuration."""
        config = PreflightConfig(
            max_time_drift_seconds=10.0,
            skip_broker_check=True,
        )

        assert config.max_time_drift_seconds == 10.0
        assert config.skip_broker_check is True


class TestPreflightChecker:
    """Tests for PreflightChecker."""

    @pytest.fixture
    def checker(self):
        """Create PreflightChecker."""
        config = PreflightConfig(
            skip_broker_check=True,
            skip_time_sync=True,
            skip_network_check=True,
        )
        return PreflightChecker(config=config)

    def test_run_preflight_minimal(self, checker):
        """Test running preflight with minimal config."""
        result = checker.run_preflight()

        assert isinstance(result, PreflightResult)
        assert result.preflight_id is not None
        assert result.duration_ms > 0

    def test_vault_check_no_vault(self, checker):
        """Test vault check when vault is not set."""
        # With vault required
        checker.config.require_vault_unlocked = True
        result = checker.run_preflight()

        # Should fail
        vault_check = next(
            (c for c in result.checks if c.check_type == PreflightCheckType.VAULT_UNLOCKED),
            None
        )
        assert vault_check is not None
        assert vault_check.result == PreflightCheckResult.FAILED

    def test_vault_check_locked(self, checker):
        """Test vault check when vault is locked."""
        mock_vault = MagicMock()
        mock_vault.is_locked = True
        checker._vault = mock_vault

        result = checker.run_preflight()

        vault_check = next(
            (c for c in result.checks if c.check_type == PreflightCheckType.VAULT_UNLOCKED),
            None
        )
        assert vault_check.result == PreflightCheckResult.FAILED

    def test_vault_check_unlocked(self, checker):
        """Test vault check when vault is unlocked."""
        mock_vault = MagicMock()
        mock_vault.is_locked = False
        checker._vault = mock_vault

        result = checker.run_preflight()

        vault_check = next(
            (c for c in result.checks if c.check_type == PreflightCheckType.VAULT_UNLOCKED),
            None
        )
        assert vault_check.result == PreflightCheckResult.PASSED

    def test_credentials_check(self, checker):
        """Test credentials check."""
        mock_vault = MagicMock()
        mock_vault.is_locked = False
        mock_vault.list_credentials.return_value = [
            {"credential_id": "binance:api_key"},
            {"credential_id": "binance:api_secret"},
        ]
        checker._vault = mock_vault

        result = checker.run_preflight(broker_name="binance")

        cred_check = next(
            (c for c in result.checks if c.check_type == PreflightCheckType.CREDENTIALS_AVAILABLE),
            None
        )
        assert cred_check.result == PreflightCheckResult.PASSED

    def test_credentials_missing(self, checker):
        """Test credentials check when missing."""
        mock_vault = MagicMock()
        mock_vault.is_locked = False
        mock_vault.list_credentials.return_value = []
        checker._vault = mock_vault

        result = checker.run_preflight(broker_name="binance")

        cred_check = next(
            (c for c in result.checks if c.check_type == PreflightCheckType.CREDENTIALS_AVAILABLE),
            None
        )
        assert cred_check.result == PreflightCheckResult.FAILED

    def test_schema_version_check_valid(self, checker):
        """Test schema version within range."""
        manifest = {"schema_version": "1.5.0", "entrypoint": "main.py"}

        result = checker.run_preflight(manifest=manifest)

        schema_check = next(
            (c for c in result.checks if c.check_type == PreflightCheckType.SCHEMA_VERSION),
            None
        )
        assert schema_check.result == PreflightCheckResult.PASSED

    def test_schema_version_check_missing(self, checker):
        """Test schema version missing."""
        manifest = {"entrypoint": "main.py"}  # No schema_version

        result = checker.run_preflight(manifest=manifest)

        schema_check = next(
            (c for c in result.checks if c.check_type == PreflightCheckType.SCHEMA_VERSION),
            None
        )
        assert schema_check.result == PreflightCheckResult.FAILED

    def test_manifest_validation(self, checker):
        """Test manifest validation."""
        # Valid manifest
        manifest = {"schema_version": "1.0.0", "entrypoint": "main.py"}
        result = checker.run_preflight(manifest=manifest)

        manifest_check = next(
            (c for c in result.checks if c.check_type == PreflightCheckType.MANIFEST_VALID),
            None
        )
        assert manifest_check.result == PreflightCheckResult.PASSED

        # Invalid manifest (missing entrypoint)
        manifest = {"schema_version": "1.0.0"}
        result = checker.run_preflight(manifest=manifest)

        manifest_check = next(
            (c for c in result.checks if c.check_type == PreflightCheckType.MANIFEST_VALID),
            None
        )
        assert manifest_check.result == PreflightCheckResult.FAILED

    def test_digest_verification(self, checker):
        """Test artifact digest verification."""
        with NamedTemporaryFile(delete=False) as f:
            f.write(b"test artifact content")
            artifact_path = Path(f.name)

        try:
            # Compute correct digest
            sha256 = hashlib.sha256()
            sha256.update(b"test artifact content")
            correct_digest = sha256.hexdigest()

            result = checker.run_preflight(
                artifact_path=artifact_path,
                artifact_digest=correct_digest,
            )

            digest_check = next(
                (c for c in result.checks if c.check_type == PreflightCheckType.DIGEST_VERIFICATION),
                None
            )
            assert digest_check.result == PreflightCheckResult.PASSED

            # Wrong digest
            result = checker.run_preflight(
                artifact_path=artifact_path,
                artifact_digest="wrongdigest",
            )

            digest_check = next(
                (c for c in result.checks if c.check_type == PreflightCheckType.DIGEST_VERIFICATION),
                None
            )
            assert digest_check.result == PreflightCheckResult.FAILED

        finally:
            artifact_path.unlink()

    def test_signature_verification(self, checker):
        """Test signature verification.

        Design Doc compliance note:
        - Without ArtifactVerifier configured, signature presence returns WARNING
          (crypto verification skipped, not recommended for production)
        - With ArtifactVerifier, real crypto verification returns PASSED/FAILED
        - Empty signature always returns FAILED
        """
        with NamedTemporaryFile(delete=False) as f:
            f.write(b"test artifact")
            artifact_path = Path(f.name)

        try:
            # Valid signature (non-empty) - without ArtifactVerifier returns WARNING
            # (Design Doc: presence check alone is not sufficient for production)
            result = checker.run_preflight(
                artifact_path=artifact_path,
                signature=b"valid_signature_bytes",
            )

            sig_check = next(
                (c for c in result.checks if c.check_type == PreflightCheckType.SIGNATURE_VERIFICATION),
                None
            )
            # Without ArtifactVerifier: WARNING (crypto skipped)
            # With ArtifactVerifier: PASSED (crypto verified)
            assert sig_check.result in (
                PreflightCheckResult.PASSED,
                PreflightCheckResult.WARNING,
            )

            # Empty signature - always FAILED
            result = checker.run_preflight(
                artifact_path=artifact_path,
                signature=b"",
            )

            sig_check = next(
                (c for c in result.checks if c.check_type == PreflightCheckType.SIGNATURE_VERIFICATION),
                None
            )
            assert sig_check.result == PreflightCheckResult.FAILED

        finally:
            artifact_path.unlink()

    def test_resources_check(self, checker):
        """Test resource availability check."""
        result = checker.run_preflight()

        resource_check = next(
            (c for c in result.checks if c.check_type == PreflightCheckType.RESOURCES_AVAILABLE),
            None
        )
        # Should pass on most systems
        assert resource_check is not None

    def test_overall_result(self, checker):
        """Test overall result computation."""
        mock_vault = MagicMock()
        mock_vault.is_locked = False
        checker._vault = mock_vault

        manifest = {"schema_version": "1.0.0", "entrypoint": "main.py"}
        result = checker.run_preflight(manifest=manifest)

        # With skipped checks and valid manifest, should pass
        assert len(result.errors) == 0
        # result.passed depends on no errors

    def test_last_result_stored(self, checker):
        """Test that last result is stored."""
        result1 = checker.run_preflight()
        assert checker.last_result == result1

        result2 = checker.run_preflight()
        assert checker.last_result == result2
