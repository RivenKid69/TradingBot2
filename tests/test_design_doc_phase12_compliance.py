# -*- coding: utf-8 -*-
"""
Tests for Design Doc Phase 12 Compliance.

Tests for critical gaps fixed:
1. Telemetry Agent→Cloud end-to-end pipeline
2. Manifest permissions block
3. Governance DB persistence
4. Research jobs workspace dependency
5. SBOM enforcement by default
6. SignatureAlgorithm terminology (ed25519)
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4


# =============================================================================
# 1. Telemetry Agent→Cloud Pipeline Tests
# =============================================================================

class TestTelemetryPipeline:
    """Tests for end-to-end telemetry pipeline."""

    def test_cloud_client_send_telemetry_method_exists(self):
        """CloudClient should have send_telemetry method."""
        from packages.agent.cloud.client import CloudClient, CloudClientConfig

        config = CloudClientConfig(base_url="https://test.example.com")
        client = CloudClient(config)
        assert hasattr(client, "send_telemetry"), "CloudClient must have send_telemetry method"
        assert callable(client.send_telemetry), "send_telemetry must be callable"

    def test_cloud_client_send_telemetry_batch_method_exists(self):
        """CloudClient should have send_telemetry_batch method."""
        from packages.agent.cloud.client import CloudClient, CloudClientConfig

        config = CloudClientConfig(base_url="https://test.example.com")
        client = CloudClient(config)
        assert hasattr(client, "send_telemetry_batch"), "CloudClient must have send_telemetry_batch method"
        assert callable(client.send_telemetry_batch), "send_telemetry_batch must be callable"

    def test_send_telemetry_empty_events_returns_true(self):
        """send_telemetry with empty events should return True."""
        from packages.agent.cloud.client import CloudClient, CloudClientConfig

        config = CloudClientConfig(base_url="https://test.example.com")
        client = CloudClient(config)
        result = client.send_telemetry([])
        assert result is True, "Empty events should return True"

    def test_agentd_has_flush_telemetry_method(self):
        """AgentDaemon should have _flush_telemetry_to_cloud method."""
        from packages.agent.daemon.agentd import AgentDaemon

        assert hasattr(AgentDaemon, "_flush_telemetry_to_cloud"), \
            "AgentDaemon must have _flush_telemetry_to_cloud method"

    def test_agent_telemetry_endpoint_models(self):
        """Agent telemetry endpoint models should exist."""
        from packages.cloud.control_plane.routers.agent_lifecycle import (
            AgentTelemetryRequest,
            AgentTelemetryResponse,
        )

        # Test request model
        request = AgentTelemetryRequest(
            events=[{"event_type": "TEST", "timestamp": datetime.now(timezone.utc).isoformat()}],
            batch_size=1,
        )
        assert len(request.events) == 1

        # Test response model
        response = AgentTelemetryResponse(
            accepted=True,
            events_received=1,
            events_stored=1,
            errors=[],
        )
        assert response.accepted is True

    def test_telemetry_redaction(self):
        """Telemetry events should have sensitive data redacted."""
        from packages.cloud.control_plane.routers.agent_lifecycle import _redact_telemetry_event

        event = {
            "event_type": "TEST",
            "api_key": "secret123",
            "password": "pass456",
            "data": {
                "secret_token": "token789",
                "value": 42,
            }
        }

        redacted = _redact_telemetry_event(event)

        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["password"] == "[REDACTED]"
        assert redacted["data"]["secret_token"] == "[REDACTED]"
        assert redacted["data"]["value"] == 42


# =============================================================================
# 2. Manifest Permissions Block Tests
# =============================================================================

class TestManifestPermissions:
    """Tests for manifest permissions block."""

    def test_network_permissions_dataclass_exists(self):
        """NetworkPermissions dataclass should exist."""
        from packages.shared.contracts.manifest import NetworkPermissions

        perms = NetworkPermissions(
            enabled=True,
            egress_allowlist=["api.example.com"],
            max_connections=5,
        )
        assert perms.enabled is True
        assert "api.example.com" in perms.egress_allowlist

    def test_filesystem_permissions_dataclass_exists(self):
        """FilesystemPermissions dataclass should exist."""
        from packages.shared.contracts.manifest import FilesystemPermissions

        perms = FilesystemPermissions(
            readonly=True,
            allowed_paths=["/tmp"],
            temp_dir_allowed=True,
        )
        assert perms.readonly is True
        assert "/tmp" in perms.allowed_paths

    def test_resource_permissions_dataclass_exists(self):
        """ResourcePermissions dataclass should exist."""
        from packages.shared.contracts.manifest import ResourcePermissions

        perms = ResourcePermissions(
            max_memory_mb=1024,
            max_cpu_percent=80.0,
            max_execution_time_seconds=3600,
        )
        assert perms.max_memory_mb == 1024
        assert perms.max_cpu_percent == 80.0

    def test_permissions_dataclass_exists(self):
        """Permissions dataclass should exist with all sub-permissions."""
        from packages.shared.contracts.manifest import (
            Permissions,
            NetworkPermissions,
            FilesystemPermissions,
            ResourcePermissions,
        )

        perms = Permissions()
        assert isinstance(perms.network, NetworkPermissions)
        assert isinstance(perms.filesystem, FilesystemPermissions)
        assert isinstance(perms.resources, ResourcePermissions)

    def test_permissions_default_restrictive(self):
        """Permissions.default_restrictive should create restrictive defaults."""
        from packages.shared.contracts.manifest import Permissions

        perms = Permissions.default_restrictive()

        assert perms.network.enabled is False
        assert perms.network.egress_allowlist == []
        assert perms.filesystem.readonly is True
        assert perms.filesystem.allowed_paths == []

    def test_artifact_manifest_includes_permissions(self):
        """ArtifactManifest should include permissions field."""
        from packages.shared.contracts.manifest import ArtifactManifest, Permissions

        manifest = ArtifactManifest(strategy_name="test")
        assert hasattr(manifest, "permissions")
        assert isinstance(manifest.permissions, Permissions)

    def test_manifest_to_dict_includes_permissions(self):
        """ArtifactManifest.to_dict should include permissions."""
        from packages.shared.contracts.manifest import ArtifactManifest

        manifest = ArtifactManifest(strategy_name="test")
        d = manifest.to_dict()

        assert "permissions" in d
        assert "network" in d["permissions"]
        assert "filesystem" in d["permissions"]
        assert "resources" in d["permissions"]

    def test_manifest_from_dict_parses_permissions(self):
        """ArtifactManifest.from_dict should parse permissions."""
        from packages.shared.contracts.manifest import ArtifactManifest

        data = {
            "strategy_name": "test",
            "permissions": {
                "network": {"enabled": True, "egress_allowlist": ["api.test.com"]},
                "filesystem": {"readonly": False},
                "resources": {"max_memory_mb": 2048},
            }
        }

        manifest = ArtifactManifest.from_dict(data)

        assert manifest.permissions.network.enabled is True
        assert "api.test.com" in manifest.permissions.network.egress_allowlist
        assert manifest.permissions.filesystem.readonly is False
        assert manifest.permissions.resources.max_memory_mb == 2048


# =============================================================================
# 3. Governance DB Persistence Tests
# =============================================================================

class TestGovernanceDBPersistence:
    """Tests for DB-backed governance persistence."""

    def test_governance_policy_model_exists(self):
        """GovernancePolicy model should exist."""
        from packages.cloud.control_plane.models import GovernancePolicy

        assert GovernancePolicy.__tablename__ == "governance_policies"
        assert hasattr(GovernancePolicy, "policy_type")
        assert hasattr(GovernancePolicy, "config")

    def test_dsar_request_model_exists(self):
        """DSARRequest model should exist."""
        from packages.cloud.control_plane.models import DSARRequest

        assert DSARRequest.__tablename__ == "dsar_requests"
        assert hasattr(DSARRequest, "request_type")
        assert hasattr(DSARRequest, "status")
        assert hasattr(DSARRequest, "deadline")

    def test_legal_hold_model_exists(self):
        """LegalHold model should exist."""
        from packages.cloud.control_plane.models import LegalHold

        assert LegalHold.__tablename__ == "legal_holds"
        assert hasattr(LegalHold, "data_type")
        assert hasattr(LegalHold, "reason")

    def test_break_glass_request_model_exists(self):
        """BreakGlassRequest model should exist."""
        from packages.cloud.control_plane.models import BreakGlassRequest

        assert BreakGlassRequest.__tablename__ == "break_glass_requests"
        assert hasattr(BreakGlassRequest, "reason")
        assert hasattr(BreakGlassRequest, "approved")

    def test_governance_audit_log_model_exists(self):
        """GovernanceAuditLog model should exist."""
        from packages.cloud.control_plane.models import GovernanceAuditLog

        assert GovernanceAuditLog.__tablename__ == "governance_audit_logs"
        assert hasattr(GovernanceAuditLog, "action")
        assert hasattr(GovernanceAuditLog, "details")

    def test_dsar_request_status_enum(self):
        """DSARRequestStatus enum should have all required statuses."""
        from packages.cloud.control_plane.models import DSARRequestStatus

        assert DSARRequestStatus.PENDING.value == "pending"
        assert DSARRequestStatus.VERIFIED.value == "verified"
        assert DSARRequestStatus.IN_PROGRESS.value == "in_progress"
        assert DSARRequestStatus.COMPLETED.value == "completed"

    def test_residency_policy_type_enum(self):
        """ResidencyPolicyType enum should exist."""
        from packages.cloud.control_plane.models import ResidencyPolicyType

        assert ResidencyPolicyType.CLOUD.value == "cloud"
        assert ResidencyPolicyType.ENTERPRISE_LOCAL.value == "enterprise_local"
        assert ResidencyPolicyType.HYBRID.value == "hybrid"


# =============================================================================
# 4. Research Jobs Workspace Dependency Tests
# =============================================================================

class TestResearchJobsWorkspace:
    """Tests for research jobs workspace dependency."""

    def test_current_workspace_model_exists(self):
        """CurrentWorkspace model should exist."""
        from packages.cloud.control_plane.dependencies import CurrentWorkspace

        workspace = CurrentWorkspace(
            id=uuid4(),
            name="test",
            org_id=uuid4(),
            settings={},
        )
        assert workspace.name == "test"

    def test_get_current_workspace_function_exists(self):
        """get_current_workspace function should exist."""
        from packages.cloud.control_plane.dependencies import get_current_workspace

        assert callable(get_current_workspace)

    def test_workspace_dep_type_alias_exists(self):
        """WorkspaceDep type alias should exist."""
        from packages.cloud.control_plane.dependencies import WorkspaceDep

        assert WorkspaceDep is not None

    def test_research_jobs_imports_workspace_dep(self):
        """Research jobs router should import WorkspaceDep."""
        from packages.cloud.control_plane.routers.research_jobs import WorkspaceDep

        assert WorkspaceDep is not None


# =============================================================================
# 5. SBOM Enforcement Tests
# =============================================================================

class TestSBOMEnforcement:
    """Tests for SBOM enforcement by default."""

    def test_artifact_verifier_default_require_sbom_true(self):
        """ArtifactVerifier should have require_sbom=True by default."""
        from ccea.artifact.verifier import ArtifactVerifier

        verifier = ArtifactVerifier()
        assert verifier._require_sbom is True, \
            "ArtifactVerifier should require SBOM by default"

    def test_artifact_verifier_can_disable_sbom_requirement(self):
        """ArtifactVerifier should allow disabling SBOM requirement."""
        from ccea.artifact.verifier import ArtifactVerifier

        verifier = ArtifactVerifier(require_sbom=False)
        assert verifier._require_sbom is False

    def test_create_verifier_from_key_manager_default_require_sbom_true(self):
        """create_verifier_from_key_manager should have require_sbom=True by default."""
        import inspect
        from ccea.artifact.verifier import create_verifier_from_key_manager

        sig = inspect.signature(create_verifier_from_key_manager)
        require_sbom_param = sig.parameters.get("require_sbom")

        assert require_sbom_param is not None
        assert require_sbom_param.default is True, \
            "create_verifier_from_key_manager should default require_sbom=True"

    def test_missing_sbom_rejection_reason_exists(self):
        """MISSING_SBOM rejection reason should exist."""
        from ccea.artifact.verifier import RejectionReason

        assert RejectionReason.MISSING_SBOM.value == "missing_sbom"


# =============================================================================
# 6. SignatureAlgorithm Terminology Tests
# =============================================================================

class TestSignatureAlgorithmTerminology:
    """Tests for SignatureAlgorithm terminology fix."""

    def test_ed25519_is_default_algorithm(self):
        """ED25519 should be the default signature algorithm."""
        from packages.shared.contracts.manifest import Signature, SignatureAlgorithm

        sig = Signature()
        assert sig.algorithm == SignatureAlgorithm.ED25519

    def test_ed25519_enum_value_exists(self):
        """ED25519 enum value should exist."""
        from packages.shared.contracts.manifest import SignatureAlgorithm

        assert SignatureAlgorithm.ED25519.value == "ed25519"

    def test_sigstore_enum_value_exists_for_future(self):
        """SIGSTORE enum value should exist for future integration."""
        from packages.shared.contracts.manifest import SignatureAlgorithm

        assert SignatureAlgorithm.SIGSTORE.value == "sigstore"

    def test_signature_from_dict_defaults_to_ed25519(self):
        """Signature.from_dict should default to ED25519."""
        from packages.shared.contracts.manifest import Signature, SignatureAlgorithm

        # Missing algorithm should default to ed25519
        data = {"signature_value": "abc123"}
        sig = Signature.from_dict(data)
        assert sig.algorithm == SignatureAlgorithm.ED25519

    def test_signature_from_dict_handles_invalid_algorithm(self):
        """Signature.from_dict should handle invalid algorithm gracefully."""
        from packages.shared.contracts.manifest import Signature, SignatureAlgorithm

        data = {"algorithm": "unknown_algo", "signature_value": "abc123"}
        sig = Signature.from_dict(data)
        assert sig.algorithm == SignatureAlgorithm.ED25519

    def test_signature_to_dict_includes_ed25519(self):
        """Signature.to_dict should include ed25519 algorithm."""
        from packages.shared.contracts.manifest import Signature, SignatureAlgorithm

        sig = Signature(algorithm=SignatureAlgorithm.ED25519, signature_value="test")
        d = sig.to_dict()
        assert d["algorithm"] == "ed25519"

    def test_artifact_model_default_signature_algorithm_is_ed25519(self):
        """Artifact model should default signature_algorithm to ed25519."""
        from packages.cloud.control_plane.models import Artifact

        # Check the default value in the column definition
        sig_algo_column = Artifact.__table__.c.signature_algorithm
        assert sig_algo_column.default.arg == "ed25519"


# =============================================================================
# 7. Telemetry Service Tests
# =============================================================================

class TestTelemetryService:
    """Tests for TelemetryService."""

    def test_telemetry_service_class_exists(self):
        """TelemetryService class should exist."""
        from packages.cloud.control_plane.services.telemetry_service import TelemetryService

        assert TelemetryService is not None

    def test_telemetry_service_has_store_agent_event(self):
        """TelemetryService should have store_agent_event method."""
        from packages.cloud.control_plane.services.telemetry_service import TelemetryService

        assert hasattr(TelemetryService, "store_agent_event")
        assert callable(getattr(TelemetryService, "store_agent_event"))

    def test_telemetry_service_has_query_events(self):
        """TelemetryService should have query_events method."""
        from packages.cloud.control_plane.services.telemetry_service import TelemetryService

        assert hasattr(TelemetryService, "query_events")
        assert callable(getattr(TelemetryService, "query_events"))


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegrationManifestPermissions:
    """Integration tests for manifest with permissions."""

    def test_manifest_round_trip_with_permissions(self):
        """Manifest should round-trip with permissions intact."""
        from packages.shared.contracts.manifest import (
            ArtifactManifest,
            Permissions,
            NetworkPermissions,
        )

        original = ArtifactManifest(
            strategy_name="test_strategy",
            permissions=Permissions(
                network=NetworkPermissions(
                    enabled=True,
                    egress_allowlist=["api.binance.com", "pypi.org"],
                ),
            ),
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = ArtifactManifest.from_dict(data)

        assert restored.strategy_name == original.strategy_name
        assert restored.permissions.network.enabled == original.permissions.network.enabled
        assert restored.permissions.network.egress_allowlist == original.permissions.network.egress_allowlist

    def test_manifest_json_serialization_with_permissions(self):
        """Manifest should serialize to JSON with permissions."""
        from packages.shared.contracts.manifest import ArtifactManifest

        manifest = ArtifactManifest(strategy_name="json_test")
        data = manifest.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert "permissions" in parsed
        assert parsed["permissions"]["network"]["enabled"] is False  # default


class TestIntegrationSignatureAlgorithm:
    """Integration tests for signature algorithm."""

    def test_signature_algorithm_backwards_compatibility(self):
        """Old manifests with sigstore should still work."""
        from packages.shared.contracts.manifest import Signature, SignatureAlgorithm

        # Old format with sigstore
        old_data = {
            "algorithm": "sigstore",
            "signature_value": "old_signature",
        }

        sig = Signature.from_dict(old_data)
        assert sig.signature_value == "old_signature"
        # sigstore is still a valid enum value
        assert sig.algorithm == SignatureAlgorithm.SIGSTORE

    def test_new_manifests_use_ed25519(self):
        """New manifests should use ed25519 by default."""
        from packages.shared.contracts.manifest import ArtifactManifest, SignatureAlgorithm

        manifest = ArtifactManifest(strategy_name="new_manifest")
        assert manifest.signature.algorithm == SignatureAlgorithm.ED25519


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
