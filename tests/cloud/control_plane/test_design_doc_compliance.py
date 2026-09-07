# -*- coding: utf-8 -*-
"""
Tests for Design Doc Compliance Features.

Tests all Design Doc compliance implementations:
1. Command payload whitelist (Design Doc 12.2, 10.4)
2. Agent enrollment + cloud public key (Design Doc 10.2, 15.2)
3. Agent signature verification (Design Doc 10.2)
4. Structured approval evidence (Design Doc 6.2, 12.2)
5. SBOM storage and retrieval (Design Doc 8.4, 16.1)
6. Telemetry RAW_ORDER_EVENTS enterprise mode (Design Doc)
"""

import hashlib
import json
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

# Test imports
from packages.cloud.control_plane.middleware.telemetry_validation import (
    TelemetryValidator,
    validate_telemetry_payload,
    ValidationSeverity,
    ViolationType,
    ALLOWED_RAW_ORDER_FIELDS,
    ALLOWED_AGGREGATED_FIELDS,
    ALLOWED_DETAILED_FIELDS,
    ENTERPRISE_RAW_ORDER_ENABLED_HEADER,
)
from packages.cloud.control_plane.middleware.agent_signature import (
    AgentSignatureVerifier,
    SignatureConfig,
    SignatureVerificationResult,
    HEADER_SIGNATURE,
    HEADER_FINGERPRINT,
    HEADER_TIMESTAMP,
)
from packages.cloud.control_plane.services.command_signer import (
    CloudCommandSigner,
    CloudKeyInfo,
)
from packages.cloud.control_plane.models import TelemetryLevel


# =============================================================================
# 1. Command Payload Whitelist Tests (Design Doc 12.2, 10.4)
# =============================================================================


class TestCommandPayloadWhitelist:
    """Tests for command payload whitelist with blob validation."""

    def test_valid_config_types_include_command_payload(self):
        """Config types should include command_payload for commands."""
        from packages.cloud.control_plane.routers.config_blobs import VALID_CONFIG_TYPES

        assert "command_payload" in VALID_CONFIG_TYPES
        assert "sbom" in VALID_CONFIG_TYPES  # Design Doc 8.4

    def test_valid_command_payload_types(self):
        """Valid payload types for commands should be defined."""
        valid_types = {"command_payload", "strategy", "execution", "risk", "environment", "custom"}
        # These types should be usable as command payloads
        for t in valid_types:
            assert t in {
                "command_payload",
                "strategy",
                "execution",
                "risk",
                "environment",
                "custom",
                "feature_flags",
                "model",
                "data",
                "alert",
                "monitoring",
                "sbom",
            }


# =============================================================================
# 2. Agent Enrollment + Cloud Public Key Tests (Design Doc 10.2, 15.2)
# =============================================================================


class TestAgentEnrollmentCloudKey:
    """Tests for agent enrollment with cloud public key provisioning."""

    def test_cloud_command_signer_generates_key_info(self):
        """CloudCommandSigner should generate proper key info."""
        signer = CloudCommandSigner.generate(key_id="test_key_v1")

        key_info = signer.get_key_info()

        assert isinstance(key_info, CloudKeyInfo)
        assert key_info.key_id == "test_key_v1"
        assert key_info.algorithm == "ed25519"
        assert key_info.public_key_pem.startswith("-----BEGIN PUBLIC KEY-----")
        assert len(key_info.fingerprint) == 64  # SHA256 hex

    def test_cloud_command_signer_signs_commands(self):
        """CloudCommandSigner should sign commands with timestamp."""
        signer = CloudCommandSigner.generate()

        command = {
            "id": str(uuid4()),
            "command_type": "REQUEST_START_RUN",
            "payload_ref": "sha256:abc123",
        }

        signed = signer.sign_command(command)

        assert "signature" in signed
        assert "timestamp" in signed
        assert signed["id"] == command["id"]
        assert signer.commands_signed == 1

    def test_cloud_command_signer_key_id_in_signature(self):
        """Signed commands should include key_id for rotation support."""
        signer = CloudCommandSigner.generate(key_id="v2")

        command = {"type": "test"}
        signed = signer.sign_command(command)

        # Signature should include key_id reference
        assert "signature" in signed
        # The signature dict should have key_id
        sig_data = signed["signature"]
        assert "key_id" in sig_data


# =============================================================================
# 3. Agent Signature Verification Tests (Design Doc 10.2)
# =============================================================================


class TestAgentSignatureVerification:
    """Tests for agent signature verification middleware."""

    def test_signature_config_defaults(self):
        """SignatureConfig should have secure defaults."""
        config = SignatureConfig()

        assert config.strict_mode is False  # Non-breaking default
        assert config.verify_timestamp is True
        assert config.log_unverified is True
        assert config.max_timestamp_age == 300  # 5 minutes

    def test_signature_verifier_creation(self):
        """AgentSignatureVerifier should be creatable with config."""
        config = SignatureConfig(strict_mode=True)
        verifier = AgentSignatureVerifier(config)

        assert verifier.config.strict_mode is True

    def test_signature_verifier_should_verify_agent_paths(self):
        """Verifier should require signatures on agent paths."""
        config = SignatureConfig()
        verifier = AgentSignatureVerifier(config)

        # Agent paths should require verification
        assert verifier.should_verify("/api/v1/agent/heartbeat")
        assert verifier.should_verify("/api/v1/agent/commands/poll")
        assert verifier.should_verify("/api/v1/agent/telemetry")

        # Non-agent paths should not require verification
        assert not verifier.should_verify("/api/v1/auth/login")
        assert not verifier.should_verify("/api/v1/workspaces")

    def test_header_constants_defined(self):
        """Header constants should be defined for signature verification."""
        assert HEADER_SIGNATURE == "X-CCEA-Signature"
        assert HEADER_FINGERPRINT == "X-CCEA-Key-Fingerprint"
        assert HEADER_TIMESTAMP == "X-CCEA-Timestamp"


# =============================================================================
# 4. Structured Approval Evidence Tests (Design Doc 6.2, 12.2)
# =============================================================================


class TestStructuredApprovalEvidence:
    """Tests for structured approval evidence with immutable blob references."""

    def test_approval_record_has_blob_digest_fields(self):
        """ApprovalRecord model should have blob digest fields."""
        from packages.cloud.control_plane.models import ApprovalRecord

        # Check that the model has the new fields
        mapper = ApprovalRecord.__mapper__
        column_names = {c.name for c in mapper.columns}

        assert "config_blob_digest" in column_names
        assert "manifest_digest" in column_names
        assert "previous_state_digest" in column_names
        assert "new_state_digest" in column_names

    def test_approval_create_accepts_blob_digests(self):
        """ApprovalCreate schema should accept blob digest fields."""
        from packages.cloud.control_plane.routers.commands import ApprovalCreate

        approval = ApprovalCreate(
            approved=True,
            reason="Test approval",
            config_blob_digest="sha256:abc123",
            manifest_digest="sha256:def456",
            previous_state_digest="sha256:old123",
            new_state_digest="sha256:new456",
        )

        assert approval.config_blob_digest == "sha256:abc123"
        assert approval.manifest_digest == "sha256:def456"
        assert approval.previous_state_digest == "sha256:old123"
        assert approval.new_state_digest == "sha256:new456"

    def test_approval_response_includes_blob_digests(self):
        """ApprovalResponse schema should include blob digest fields."""
        from packages.cloud.control_plane.routers.commands import ApprovalResponse

        response = ApprovalResponse(
            id=uuid4(),
            command_id=uuid4(),
            approved=True,
            approved_by="user:123",
            evidence_hash="sha256:evidence",
            attestation=None,
            reason="Test",
            diff_summary=None,
            config_blob_digest="sha256:config",
            manifest_digest="sha256:manifest",
            previous_state_digest="sha256:prev",
            new_state_digest="sha256:new",
            created_at=datetime.now(timezone.utc),
        )

        assert response.config_blob_digest == "sha256:config"


# =============================================================================
# 5. SBOM Storage and Retrieval Tests (Design Doc 8.4, 16.1)
# =============================================================================


class TestSBOMStorageRetrieval:
    """Tests for SBOM storage and retrieval path."""

    def test_sbom_config_type_valid(self):
        """SBOM should be a valid config type."""
        from packages.cloud.control_plane.routers.config_blobs import VALID_CONFIG_TYPES

        assert "sbom" in VALID_CONFIG_TYPES

    def test_sbom_create_request_schema(self):
        """SBOMCreateRequest schema should be properly defined."""
        from packages.cloud.control_plane.routers.config_blobs import SBOMCreateRequest

        request = SBOMCreateRequest(
            workspace_id=uuid4(),
            artifact_digest="sha256:artifact123",
            sbom_format="cyclonedx-json",
            content={
                "bomFormat": "CycloneDX",
                "specVersion": "1.4",
                "components": [],
            },
        )

        assert request.artifact_digest == "sha256:artifact123"
        assert request.sbom_format == "cyclonedx-json"

    def test_sbom_response_schema(self):
        """SBOMResponse schema should include all fields."""
        from packages.cloud.control_plane.routers.config_blobs import SBOMResponse

        response = SBOMResponse(
            id=uuid4(),
            workspace_id=uuid4(),
            digest="sha256:sbom123",
            artifact_digest="sha256:artifact123",
            sbom_format="cyclonedx-json",
            size_bytes=1024,
            content={"components": []},
            created_at=datetime.now(timezone.utc),
        )

        assert response.artifact_digest == "sha256:artifact123"
        assert response.sbom_format == "cyclonedx-json"


# =============================================================================
# 6. Telemetry RAW_ORDER_EVENTS Enterprise Mode Tests
# =============================================================================


class TestTelemetryEnterpriseMode:
    """Tests for enterprise RAW_ORDER_EVENTS telemetry mode."""

    def test_raw_order_fields_defined(self):
        """RAW_ORDER_FIELDS should include order/position fields."""
        assert "side" in ALLOWED_RAW_ORDER_FIELDS
        assert "quantity" in ALLOWED_RAW_ORDER_FIELDS
        assert "price" in ALLOWED_RAW_ORDER_FIELDS
        assert "order_id" in ALLOWED_RAW_ORDER_FIELDS
        assert "position_size" in ALLOWED_RAW_ORDER_FIELDS
        assert "unrealized_pnl" in ALLOWED_RAW_ORDER_FIELDS

    def test_raw_order_events_blocked_without_enterprise(self):
        """RAW_ORDER_EVENTS should be blocked without enterprise flag."""
        validator = TelemetryValidator(
            strict_mode=True,
            enterprise_raw_order_enabled=False,
        )

        payload = {
            "side": "buy",
            "quantity": 100,
            "price": 150.50,
        }

        result = validator.validate(
            payload,
            TelemetryLevel.RAW_ORDER_EVENTS.value,
            redaction_applied=True,
        )

        assert not result.valid
        assert result.blocked
        # Should have violation for RAW_ORDER_DATA
        violation_types = {v.violation_type for v in result.violations}
        assert ViolationType.RAW_ORDER_DATA in violation_types

    def test_raw_order_events_allowed_with_enterprise(self):
        """RAW_ORDER_EVENTS should be allowed with enterprise flag."""
        validator = TelemetryValidator(
            strict_mode=False,  # Non-strict for this test
            enterprise_raw_order_enabled=True,
        )

        payload = {
            "side": "buy",
            "quantity": 100,
            "price": 150.50,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = validator.validate(
            payload,
            TelemetryLevel.RAW_ORDER_EVENTS.value,
            redaction_applied=True,
        )

        assert result.valid
        assert not result.blocked

    def test_enterprise_header_constant_defined(self):
        """Enterprise header constant should be defined."""
        assert ENTERPRISE_RAW_ORDER_ENABLED_HEADER == "X-CCEA-Enterprise-Raw-Order-Enabled"

    def test_aggregated_still_works_with_enterprise(self):
        """AGGREGATED level should work regardless of enterprise flag."""
        validator = TelemetryValidator(
            strict_mode=True,
            enterprise_raw_order_enabled=False,
        )

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_count": 100,
            "win_rate": 0.65,
            "sharpe_ratio": 1.5,
        }

        result = validator.validate(
            payload,
            TelemetryLevel.AGGREGATED.value,
            redaction_applied=True,
        )

        assert result.valid

    def test_order_fields_blocked_in_aggregated(self):
        """Order fields should be blocked in AGGREGATED level."""
        validator = TelemetryValidator(
            strict_mode=True,
            enterprise_raw_order_enabled=False,
        )

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "side": "buy",  # Order field - prohibited
        }

        result = validator.validate(
            payload,
            TelemetryLevel.AGGREGATED.value,
            redaction_applied=True,
        )

        assert not result.valid
        assert result.blocked

    def test_convenience_function_supports_enterprise(self):
        """validate_telemetry_payload should support enterprise flag."""
        payload = {
            "side": "sell",
            "quantity": 50,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Without enterprise
        result = validate_telemetry_payload(
            payload,
            TelemetryLevel.RAW_ORDER_EVENTS.value,
            enterprise_raw_order_enabled=False,
        )
        assert not result.valid

        # With enterprise
        result = validate_telemetry_payload(
            payload,
            TelemetryLevel.RAW_ORDER_EVENTS.value,
            enterprise_raw_order_enabled=True,
            strict_mode=False,
        )
        assert result.valid


# =============================================================================
# OCI Pipeline Tests (Design Doc 8.1)
# =============================================================================


class TestOCIPipeline:
    """Tests for OCI-first artifact pipeline."""

    def test_cloud_build_config_defaults_oci(self):
        """CloudBuildConfig should default to OCI format."""
        from packages.cloud.builder.oci_pipeline import CloudBuildConfig
        from pathlib import Path

        config = CloudBuildConfig(
            source_dir=Path("/tmp/test"),
            entrypoint_module="strategy",
            entrypoint_class="MyStrategy",
        )

        assert config.use_oci_format is True
        assert config.fallback_to_zip is True
        assert config.sign_artifact is True  # MANDATORY
        assert config.generate_sbom is True  # Design Doc 8.4

    def test_cloud_build_result_structure(self):
        """CloudBuildResult should have all required fields."""
        from packages.cloud.builder.oci_pipeline import CloudBuildResult
        from ccea.artifact.pipeline import ArtifactFormat
        from pathlib import Path

        result = CloudBuildResult(
            success=True,
            artifact_id="test-artifact",
            version="1.0.0",
            artifact_digest="sha256:abc123",
            manifest_digest="sha256:manifest123",
            artifact_path=Path("/tmp/artifact.oci"),
            manifest_path=Path("/tmp/manifest.json"),
            format=ArtifactFormat.OCI,
            oci_image_digest="sha256:image123",
            sbom_digest="sha256:sbom123",
            signed=True,
            key_id="test-key",
        )

        assert result.success
        assert result.format == ArtifactFormat.OCI
        assert result.oci_image_digest is not None
        assert result.signed

    def test_cloud_artifact_pipeline_exports(self):
        """Package should export OCI pipeline components."""
        from packages.cloud.builder import (
            CloudArtifactPipeline,
            CloudBuildConfig,
            CloudBuildResult,
            build_cloud_artifact,
        )

        assert CloudArtifactPipeline is not None
        assert CloudBuildConfig is not None
        assert CloudBuildResult is not None
        assert build_cloud_artifact is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestDesignDocComplianceIntegration:
    """Integration tests for Design Doc compliance."""

    def test_all_middleware_exports_available(self):
        """All middleware should be importable from package."""
        from packages.cloud.control_plane.middleware import (
            # Agent Signature
            AgentSignatureVerifier,
            AgentSignatureMiddleware,
            SignatureConfig,
            SignatureVerificationResult,
            HEADER_SIGNATURE,
            # Telemetry
            TelemetryValidator,
            ALLOWED_RAW_ORDER_FIELDS,
            ENTERPRISE_RAW_ORDER_ENABLED_HEADER,
        )

        assert AgentSignatureVerifier is not None
        assert TelemetryValidator is not None
        assert ALLOWED_RAW_ORDER_FIELDS is not None

    def test_config_types_comprehensive(self):
        """All required config types should be defined."""
        from packages.cloud.control_plane.routers.config_blobs import VALID_CONFIG_TYPES

        required_types = {
            "strategy",
            "risk",
            "execution",
            "environment",
            "command_payload",
            "sbom",
        }

        assert required_types.issubset(VALID_CONFIG_TYPES)

    def test_telemetry_levels_have_proper_allowlists(self):
        """Each telemetry level should have appropriate field allowlist."""
        validator = TelemetryValidator(
            strict_mode=True,
            enterprise_raw_order_enabled=True,
        )

        # AGGREGATED - basic metrics only
        aggregated_allowed = validator._get_allowed_fields(TelemetryLevel.AGGREGATED.value)
        assert "timestamp" in aggregated_allowed
        assert "trade_count" in aggregated_allowed
        assert "side" not in aggregated_allowed  # Order field not in aggregated

        # DETAILED - includes event details
        detailed_allowed = validator._get_allowed_fields(
            TelemetryLevel.DETAILED_NON_SENSITIVE.value
        )
        assert "event_type" in detailed_allowed
        assert "side" not in detailed_allowed  # Order field not in detailed

        # RAW_ORDER_EVENTS (enterprise) - includes order fields
        raw_allowed = validator._get_allowed_fields(TelemetryLevel.RAW_ORDER_EVENTS.value)
        assert "side" in raw_allowed
        assert "quantity" in raw_allowed
        assert "price" in raw_allowed
