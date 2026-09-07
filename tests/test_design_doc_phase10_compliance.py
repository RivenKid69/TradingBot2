# -*- coding: utf-8 -*-
"""
Design Doc Phase 10 Compliance Tests.

Tests for all medium and low priority Design Doc compliance implementations:
- Protocol versioning negotiation (10.3/10.4)
- Research sandbox execution (15.3)
- Governance DB integration
- Enterprise/airgapped config (Phase 9)
- Preflight manifest permissions (9.2, 2.1)
- Degraded mode LIVE policy (9.6, 13.2)
- ChangeClass vocabulary unification
- Guardrails path fixes

100% test coverage target.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4

import pytest


# ============================================================================
# Test: Protocol Versioning Negotiation (Design Doc 10.3/10.4)
# ============================================================================

class TestProtocolVersioning:
    """Tests for protocol version negotiation."""

    def test_version_negotiate_request_validation(self):
        """Test version negotiation request validation."""
        from packages.cloud.control_plane.routers.agent_lifecycle import (
            VersionNegotiateRequest,
        )

        # Valid request
        request = VersionNegotiateRequest(
            min_supported="1.0.0",
            max_supported="2.0.0",
            preferred="1.5.0",
        )
        assert request.min_supported == "1.0.0"
        assert request.max_supported == "2.0.0"

    def test_version_negotiate_request_invalid_range(self):
        """Test version negotiation rejects invalid range."""
        from packages.cloud.control_plane.routers.agent_lifecycle import (
            VersionNegotiateRequest,
        )
        from pydantic import ValidationError

        # max < min should fail validation
        with pytest.raises(ValidationError):
            VersionNegotiateRequest(
                min_supported="2.0.0",
                max_supported="1.0.0",
            )

    def test_version_negotiator_success(self):
        """Test successful version negotiation."""
        from ccea.protocol.schema_versioning import (
            SchemaVersionNegotiator,
            VersionNegotiationRequest,
            NegotiationStatus,
        )

        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )

        request = VersionNegotiationRequest(
            agent_id="agent_testagent12345678901234",
            min_supported="1.5.0",
            max_supported="1.9.0",
        )

        result = negotiator.negotiate(request)
        assert result.status == NegotiationStatus.SUCCESS
        assert result.selected_version is not None

    def test_version_negotiator_no_overlap(self):
        """Test version negotiation failure with no overlap."""
        from ccea.protocol.schema_versioning import (
            SchemaVersionNegotiator,
            VersionNegotiationRequest,
            NegotiationStatus,
        )

        negotiator = SchemaVersionNegotiator(
            min_supported="2.0.0",
            max_supported="3.0.0",
        )

        request = VersionNegotiationRequest(
            agent_id="agent_testagent12345678901234",
            min_supported="1.0.0",
            max_supported="1.5.0",
        )

        result = negotiator.negotiate(request)
        assert result.status == NegotiationStatus.FAILED


# ============================================================================
# Test: Research Execution Service (Design Doc 15.3)
# ============================================================================

class TestResearchExecutionService:
    """Tests for research execution service."""

    def test_execution_service_config(self):
        """Test execution service configuration."""
        from packages.cloud.control_plane.services.research_execution_service import (
            ExecutionServiceConfig,
            ExecutionMode,
        )

        config = ExecutionServiceConfig(
            default_mode=ExecutionMode.CONTAINER,
            max_concurrent_jobs=5,
            max_timeout_seconds=7200,
        )

        assert config.default_mode == ExecutionMode.CONTAINER
        assert config.max_concurrent_jobs == 5
        assert config.max_timeout_seconds == 7200

    def test_job_submit_request_validation(self):
        """Test job submission request validation."""
        from packages.cloud.control_plane.services.research_execution_service import (
            JobSubmitRequest,
            JobPriority,
        )

        request = JobSubmitRequest(
            tenant_id="tenant_123",
            user_id="user_456",
            workspace_id="workspace_789",
            name="Test Job",
            code="print('hello')",
            cpu=2.0,
            memory_mb=2048,
            timeout_seconds=600,
        )

        assert request.tenant_id == "tenant_123"
        assert request.cpu == 2.0
        assert request.priority == JobPriority.NORMAL

    def test_audit_event_creation(self):
        """Test audit event creation."""
        from packages.cloud.control_plane.services.research_execution_service import (
            AuditEvent,
        )

        event = AuditEvent(
            event_type="job.submit",
            tenant_id="tenant_123",
            job_id="job_456",
            action="submit",
        )

        assert event.event_type == "job.submit"
        assert event.success is True

        event_dict = event.to_dict()
        assert "timestamp" in event_dict
        assert "event_id" in event_dict


# ============================================================================
# Test: Governance DB Service
# ============================================================================

class TestGovernanceDBService:
    """Tests for governance DB integration."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_create_residency_policy(self, mock_session):
        """Test creating residency policy."""
        from packages.cloud.control_plane.services.governance_db_service import (
            GovernanceDBService,
        )

        service = GovernanceDBService(session=mock_session)

        policy = await service.create_residency_policy(
            workspace_id=uuid4(),
            country_code="DE",
        )

        assert policy is not None
        assert policy.gdpr_compliant is True

    @pytest.mark.asyncio
    async def test_create_retention_policy(self, mock_session):
        """Test creating retention policy."""
        from packages.cloud.control_plane.services.governance_db_service import (
            GovernanceDBService,
        )
        from packages.cloud.governance.retention import RetentionAction

        service = GovernanceDBService(session=mock_session)

        policy = await service.create_retention_policy(
            workspace_id=uuid4(),
            data_type="telemetry_events",
            retention_days=90,
            action=RetentionAction.DELETE,
        )

        assert policy is not None
        assert policy.retention_days == 90


# ============================================================================
# Test: Enterprise Config (Design Doc Phase 9)
# ============================================================================

class TestEnterpriseConfig:
    """Tests for enterprise/airgapped configuration."""

    def test_airgapped_config_from_env(self):
        """Test airgapped config loading from environment."""
        from packages.cloud.control_plane.config.enterprise_config import (
            AirGappedConfig,
            EnvVar,
        )

        with patch.dict(os.environ, {
            EnvVar.AIR_GAPPED_MODE.value: "true",
            EnvVar.OFFLINE_VERIFICATION.value: "true",
        }):
            config = AirGappedConfig.from_env()
            assert config.enabled is True
            assert config.offline_verification is True

    def test_mtls_config(self):
        """Test mTLS configuration."""
        from packages.cloud.control_plane.config.enterprise_config import (
            MTLSConfig,
        )

        config = MTLSConfig(
            enabled=True,
            cert_path=Path("/certs/server.crt"),
            key_path=Path("/certs/server.key"),
        )

        assert config.enabled is True
        assert config.cert_path is not None

    def test_enterprise_config_validation(self):
        """Test enterprise config validation."""
        from packages.cloud.control_plane.config.enterprise_config import (
            EnterpriseConfig,
            AirGappedConfig,
            SigningConfig,
        )

        config = EnterpriseConfig(
            environment="production",
            enterprise_mode=True,
            airgapped=AirGappedConfig(enabled=True),
            # Note: __post_init__ auto-corrects keyless=False for airgapped
            # But offline signing without key_path should still trigger validation error
        )

        errors = config.validate()
        # Airgapped mode auto-sets offline_mode=True but no key_path provided
        # Should have error about offline signing requiring key path
        assert any("signing" in e.lower() and "key" in e.lower() for e in errors)

    def test_enterprise_config_to_dict(self):
        """Test enterprise config serialization."""
        from packages.cloud.control_plane.config.enterprise_config import (
            EnterpriseConfig,
        )

        config = EnterpriseConfig(
            environment="production",
            enterprise_mode=True,
        )

        data = config.to_dict()
        assert data["environment"] == "production"
        assert data["enterprise_mode"] is True


# ============================================================================
# Test: Preflight Manifest Permissions (Design Doc 9.2, 2.1)
# ============================================================================

class TestPreflightManifestPermissions:
    """Tests for preflight manifest permission checks."""

    def test_check_manifest_permissions_valid(self):
        """Test valid manifest permissions pass."""
        from packages.agent.daemon.preflight import (
            PreflightChecker,
            PreflightConfig,
            PreflightCheckResult,
        )

        checker = PreflightChecker(config=PreflightConfig())

        manifest = {
            "schema_version": "1.0.0",
            "entrypoint": "main.py",
            "permissions": {
                "network_allowed": False,
                "filesystem_readonly": True,
                "max_memory_mb": 512,
            },
        }

        result = checker._check_manifest_permissions(manifest)
        assert result.result == PreflightCheckResult.PASSED

    def test_check_manifest_permissions_unbounded_network(self):
        """Test unbounded network access is rejected."""
        from packages.agent.daemon.preflight import (
            PreflightChecker,
            PreflightConfig,
            PreflightCheckResult,
        )

        checker = PreflightChecker(config=PreflightConfig())

        manifest = {
            "schema_version": "1.0.0",
            "entrypoint": "main.py",
            "permissions": {
                "network_allowed": True,
                "egress_allowlist": [],  # No allowlist!
            },
        }

        result = checker._check_manifest_permissions(manifest)
        assert result.result == PreflightCheckResult.FAILED

    def test_check_egress_policy_wildcard_rejected(self):
        """Test wildcard egress is rejected."""
        from packages.agent.daemon.preflight import (
            PreflightChecker,
            PreflightConfig,
            PreflightCheckResult,
        )

        checker = PreflightChecker(config=PreflightConfig())

        manifest = {
            "permissions": {
                "network_allowed": True,
                "egress_allowlist": ["*"],  # Wildcard!
            },
        }

        result = checker._check_egress_policy(manifest)
        assert result.result == PreflightCheckResult.FAILED

    def test_check_filesystem_policy_dangerous_paths(self):
        """Test dangerous filesystem paths are rejected."""
        from packages.agent.daemon.preflight import (
            PreflightChecker,
            PreflightConfig,
            PreflightCheckResult,
        )

        checker = PreflightChecker(config=PreflightConfig())

        manifest = {
            "permissions": {
                "filesystem_readonly": False,
                "allowed_paths": ["/etc/passwd"],  # Dangerous!
            },
        }

        result = checker._check_filesystem_policy(manifest)
        assert result.result == PreflightCheckResult.FAILED

    def test_check_manifest_no_manifest_fail_closed(self):
        """Test no manifest triggers fail-closed behavior."""
        from packages.agent.daemon.preflight import (
            PreflightChecker,
            PreflightConfig,
            PreflightCheckResult,
        )

        checker = PreflightChecker(config=PreflightConfig())

        result = checker._check_manifest_permissions(None)
        assert result.result == PreflightCheckResult.FAILED
        assert "fail-closed" in result.message.lower()


# ============================================================================
# Test: Degraded Mode LIVE Policy (Design Doc 9.6, 13.2)
# ============================================================================

class TestDegradedModeLivePolicy:
    """Tests for degraded mode LIVE-specific policies."""

    def test_trading_mode_enum(self):
        """Test trading mode enum values."""
        from packages.agent.daemon.degraded_mode import TradingMode

        assert TradingMode.BACKTEST.value == "backtest"
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.LIVE.value == "live"

    def test_live_cloud_unreachable_action(self):
        """Test LIVE mode cloud unreachable action."""
        from packages.agent.daemon.degraded_mode import (
            DegradedModeConfig,
            DegradedModeAction,
            TradingMode,
        )

        config = DegradedModeConfig(
            trading_mode=TradingMode.LIVE,
            live_cloud_unreachable_action=DegradedModeAction.RESTRICT,
        )

        action = config.get_cloud_unreachable_action()
        assert action == DegradedModeAction.RESTRICT

    def test_live_force_halt_on_cloud_loss(self):
        """Test LIVE mode force halt on cloud loss."""
        from packages.agent.daemon.degraded_mode import (
            DegradedModeConfig,
            DegradedModeAction,
            TradingMode,
        )

        config = DegradedModeConfig(
            trading_mode=TradingMode.LIVE,
            live_force_halt_on_cloud_loss=True,
        )

        action = config.get_cloud_unreachable_action()
        assert action == DegradedModeAction.HALT

    def test_local_telemetry_config(self):
        """Test local telemetry configuration."""
        from packages.agent.daemon.degraded_mode import DegradedModeConfig

        config = DegradedModeConfig(
            local_telemetry_enabled=True,
            local_telemetry_buffer_size=5000,
            local_telemetry_path="/custom/path",
        )

        assert config.local_telemetry_enabled is True
        assert config.local_telemetry_buffer_size == 5000
        assert config.local_telemetry_path == "/custom/path"

    def test_degraded_mode_manager_cloud_status(self):
        """Test degraded mode manager cloud status reporting."""
        from packages.agent.daemon.degraded_mode import (
            DegradedModeManager,
            DegradedModeConfig,
            DegradedMode,
        )

        manager = DegradedModeManager(config=DegradedModeConfig())

        # Report connected
        manager.report_cloud_status(connected=True)
        assert DegradedMode.CLOUD_UNREACHABLE not in manager.active_modes

        # Report disconnected
        manager.report_cloud_status(connected=False)
        # Won't immediately enter degraded mode (timeout check)


# ============================================================================
# Test: ChangeClass Vocabulary Unification
# ============================================================================

class TestChangeClassVocabulary:
    """Tests for unified ChangeClass vocabulary."""

    def test_change_class_values(self):
        """Test ChangeClass enum has all required values."""
        from packages.shared.contracts.config import ChangeClass

        assert ChangeClass.TRADING_IMPACTING.value == "trading_impacting"
        assert ChangeClass.OPERATIONAL.value == "operational"
        assert ChangeClass.INFORMATIONAL.value == "informational"
        assert ChangeClass.DATA_SENSITIVE.value == "data_sensitive"
        assert ChangeClass.SAFETY_OPERATION.value == "safety_operation"

    def test_requires_local_approval(self):
        """Test requires_local_approval method."""
        from packages.shared.contracts.config import ChangeClass

        assert ChangeClass.requires_local_approval(ChangeClass.TRADING_IMPACTING) is True
        assert ChangeClass.requires_local_approval(ChangeClass.DATA_SENSITIVE) is True
        assert ChangeClass.requires_local_approval(ChangeClass.OPERATIONAL) is False
        assert ChangeClass.requires_local_approval(ChangeClass.INFORMATIONAL) is False

    def test_is_safety_operation(self):
        """Test is_safety_operation method."""
        from packages.shared.contracts.config import ChangeClass

        assert ChangeClass.is_safety_operation(ChangeClass.SAFETY_OPERATION) is True
        assert ChangeClass.is_safety_operation(ChangeClass.TRADING_IMPACTING) is False

    def test_can_bypass_approval(self):
        """Test can_bypass_approval method."""
        from packages.shared.contracts.config import ChangeClass

        assert ChangeClass.can_bypass_approval(ChangeClass.SAFETY_OPERATION) is True
        assert ChangeClass.can_bypass_approval(ChangeClass.INFORMATIONAL) is True
        assert ChangeClass.can_bypass_approval(ChangeClass.TRADING_IMPACTING) is False


# ============================================================================
# Test: Guardrails Path Fixes
# ============================================================================

class TestGuardrailsPaths:
    """Tests for guardrails path fixes."""

    def test_find_design_doc_path(self):
        """Test design doc path finder."""
        from ccea.guardrails.design_doc_check import (
            find_design_doc_path,
            PRIMARY_SNAPSHOT_PATH,
        )

        # Should return a path (may be None if doc doesn't exist in test env)
        result = find_design_doc_path()
        # Just test it doesn't crash
        assert result is None or isinstance(result, Path)

    def test_compute_sha256(self):
        """Test SHA256 computation."""
        from ccea.guardrails.design_doc_check import compute_sha256

        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            sha = compute_sha256(temp_path)
            assert len(sha) == 64  # SHA256 hex length
            assert all(c in "0123456789abcdef" for c in sha)
        finally:
            temp_path.unlink()


# ============================================================================
# Test: Deprecated Module CI Guardrail
# ============================================================================

class TestDeprecatedModuleGuardrail:
    """Tests for deprecated module CI guardrail."""

    def test_deprecated_import_blocked_in_ci(self):
        """Test deprecated import is blocked in CI."""
        # Save original env
        original_ci = os.environ.get("CI")
        original_allow = os.environ.get("CCEA_ALLOW_DEPRECATED")

        try:
            # Set CI mode
            os.environ["CI"] = "true"
            os.environ.pop("CCEA_ALLOW_DEPRECATED", None)

            # Clear any cached imports
            import sys
            if "ccea.agent" in sys.modules:
                del sys.modules["ccea.agent"]

            # Import should fail
            with pytest.raises(ImportError) as exc_info:
                import ccea.agent  # noqa

            assert "DEPRECATED" in str(exc_info.value)

        finally:
            # Restore original env
            if original_ci is None:
                os.environ.pop("CI", None)
            else:
                os.environ["CI"] = original_ci

            if original_allow is not None:
                os.environ["CCEA_ALLOW_DEPRECATED"] = original_allow


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for combined functionality."""

    def test_preflight_runs_all_checks(self):
        """Test preflight runs all check types including new ones."""
        from packages.agent.daemon.preflight import (
            PreflightChecker,
            PreflightConfig,
            PreflightCheckType,
        )

        checker = PreflightChecker(config=PreflightConfig(
            skip_broker_check=True,
            skip_network_check=True,
            skip_time_sync=True,
            require_vault_unlocked=False,
        ))

        manifest = {
            "schema_version": "1.0.0",
            "entrypoint": "main.py",
            "permissions": {
                "network_allowed": False,
                "filesystem_readonly": True,
            },
        }

        result = checker.run_preflight(manifest=manifest)

        # Check that all new check types are included
        check_types = [c.check_type for c in result.checks]
        assert PreflightCheckType.MANIFEST_PERMISSIONS in check_types
        assert PreflightCheckType.EGRESS_POLICY in check_types
        assert PreflightCheckType.FILESYSTEM_POLICY in check_types

    def test_enterprise_config_airgapped_overrides(self):
        """Test enterprise config applies airgapped overrides."""
        from packages.cloud.control_plane.config.enterprise_config import (
            EnterpriseConfig,
            AirGappedConfig,
        )

        # Create config with airgapped enabled
        config = EnterpriseConfig(
            environment="production",
            enterprise_mode=True,
            airgapped=AirGappedConfig(enabled=True),
        )

        # Airgapped should override telemetry
        assert config.telemetry.local_only is True
        assert config.telemetry.external_export is False

        # Airgapped should override signing
        assert config.signing.offline_mode is True
        assert config.signing.keyless is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
