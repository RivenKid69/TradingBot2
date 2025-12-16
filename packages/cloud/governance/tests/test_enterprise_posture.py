# -*- coding: utf-8 -*-
"""
Tests for Enterprise Posture Service - Phase 9 GDPR Implementation.

Comprehensive test coverage for:
    - EnterprisePostureConfig
    - TelemetryLocalModeService
    - EnterpriseEvidencePackExporter
    - EnterprisePostureValidator
    - EnterprisePostureService
    - Factory functions
"""

import json
import os
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from packages.cloud.governance.enterprise_posture import (
    # Enums
    EnterpriseDeploymentMode,
    TelemetryDestination,
    KeyManagementType,
    UpdateSource,
    PostureCheckResult,
    AuditAction,
    # Data classes
    EnterprisePostureConfig,
    PostureCheck,
    PostureValidationReport,
    TelemetryLocalModeConfig,
    EnterpriseEvidencePack,
    PostureAuditEvent,
    # Services
    TelemetryLocalModeService,
    EnterpriseEvidencePackExporter,
    EnterprisePostureValidator,
    EnterprisePostureService,
    # Factory functions
    create_on_prem_posture,
    create_air_gapped_posture,
    # Constants
    EU_REGIONS,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def workspace_id() -> str:
    """Test workspace ID."""
    return "ws-test-123"


@pytest.fixture
def organization_id() -> str:
    """Test organization ID."""
    return "org-test-456"


@pytest.fixture
def principal_id() -> str:
    """Test principal ID."""
    return "user@example.com"


@pytest.fixture
def audit_events() -> List[PostureAuditEvent]:
    """Collector for audit events."""
    return []


@pytest.fixture
def on_event(audit_events: List[PostureAuditEvent]):
    """Event callback that collects events."""
    def callback(event: PostureAuditEvent):
        audit_events.append(event)
    return callback


@pytest.fixture
def posture_service(on_event) -> EnterprisePostureService:
    """Create posture service for testing."""
    return EnterprisePostureService(on_event=on_event)


@pytest.fixture
def telemetry_local_service(on_event) -> TelemetryLocalModeService:
    """Create telemetry local mode service for testing."""
    return TelemetryLocalModeService(on_event=on_event)


@pytest.fixture
def validator(on_event) -> EnterprisePostureValidator:
    """Create posture validator for testing."""
    return EnterprisePostureValidator(on_event=on_event)


@pytest.fixture
def evidence_exporter(on_event) -> EnterpriseEvidencePackExporter:
    """Create evidence exporter for testing."""
    return EnterpriseEvidencePackExporter(on_event=on_event)


# =============================================================================
# Test EnterprisePostureConfig
# =============================================================================

class TestEnterprisePostureConfig:
    """Tests for EnterprisePostureConfig data class."""

    def test_default_config_is_saas(self):
        """Test default config is SaaS mode."""
        config = EnterprisePostureConfig()
        assert config.deployment_mode == EnterpriseDeploymentMode.SAAS
        assert config.telemetry_destination == TelemetryDestination.CLOUD
        assert config.telemetry_local_only is False
        assert config.air_gapped is False

    def test_config_to_dict(self, workspace_id, organization_id):
        """Test config serialization."""
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            telemetry_local_only=True,
        )
        data = config.to_dict()

        assert data["workspace_id"] == workspace_id
        assert data["organization_id"] == organization_id
        assert data["deployment_mode"] == "on_prem_full"
        assert data["telemetry_local_only"] is True

    def test_config_with_cmk(self):
        """Test config with customer-managed keys."""
        config = EnterprisePostureConfig(
            key_management=KeyManagementType.CUSTOMER_MANAGED,
            cmk_provider="aws_kms",
            cmk_key_id="arn:aws:kms:eu-west-1:123:key/abc",
        )

        assert config.key_management == KeyManagementType.CUSTOMER_MANAGED
        assert config.cmk_provider == "aws_kms"
        assert config.cmk_key_id is not None

    def test_config_with_raw_telemetry_opt_in(self):
        """Test config with RAW telemetry opt-in."""
        now = datetime.now(timezone.utc)
        config = EnterprisePostureConfig(
            raw_telemetry_enabled=True,
            raw_telemetry_opt_in_date=now,
            raw_telemetry_opt_in_by="admin@company.com",
        )

        assert config.raw_telemetry_enabled is True
        assert config.raw_telemetry_opt_in_date == now
        assert config.raw_telemetry_opt_in_by == "admin@company.com"


# =============================================================================
# Test TelemetryLocalModeService
# =============================================================================

class TestTelemetryLocalModeService:
    """Tests for TelemetryLocalModeService."""

    def test_enable_local_mode(self, telemetry_local_service, workspace_id, principal_id, audit_events):
        """Test enabling telemetry local mode."""
        config = TelemetryLocalModeConfig(
            storage_path="/data/telemetry",
            retention_days=90,
        )

        result = telemetry_local_service.enable_local_mode(
            workspace_id=workspace_id,
            config=config,
            enabled_by=principal_id,
        )

        assert result.enabled is True
        assert result.cloud_export_blocked is True
        assert result.storage_path == "/data/telemetry"

        # Check audit event
        assert len(audit_events) == 1
        assert audit_events[0].action == AuditAction.TELEMETRY_MODE_CHANGED
        assert audit_events[0].details["mode"] == "local"

    def test_disable_local_mode(self, telemetry_local_service, workspace_id, principal_id):
        """Test disabling telemetry local mode."""
        # First enable
        config = TelemetryLocalModeConfig()
        telemetry_local_service.enable_local_mode(workspace_id, config, principal_id)

        # Then disable
        result = telemetry_local_service.disable_local_mode(
            workspace_id=workspace_id,
            disabled_by=principal_id,
            reason="Migrating to cloud",
        )

        assert result.enabled is False
        assert result.cloud_export_blocked is False

    def test_is_local_mode_enabled(self, telemetry_local_service, workspace_id, principal_id):
        """Test checking if local mode is enabled."""
        assert telemetry_local_service.is_local_mode_enabled(workspace_id) is False

        config = TelemetryLocalModeConfig()
        telemetry_local_service.enable_local_mode(workspace_id, config, principal_id)

        assert telemetry_local_service.is_local_mode_enabled(workspace_id) is True

    def test_validate_telemetry_destination_blocked(
        self, telemetry_local_service, workspace_id, principal_id
    ):
        """Test that cloud destination is blocked when local mode enabled."""
        config = TelemetryLocalModeConfig()
        telemetry_local_service.enable_local_mode(workspace_id, config, principal_id)

        allowed, error = telemetry_local_service.validate_telemetry_destination(
            workspace_id, "cloud"
        )

        assert allowed is False
        assert "local mode is enabled" in error.lower()

    def test_validate_telemetry_destination_allowed(self, telemetry_local_service, workspace_id):
        """Test that destination is allowed when local mode not enabled."""
        allowed, error = telemetry_local_service.validate_telemetry_destination(
            workspace_id, "cloud"
        )

        assert allowed is True
        assert error is None


# =============================================================================
# Test EnterprisePostureValidator
# =============================================================================

class TestEnterprisePostureValidator:
    """Tests for EnterprisePostureValidator."""

    def test_validate_eu_residency_pass(self, validator, workspace_id, principal_id):
        """Test EU residency validation passes for EU regions."""
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            data_residency="eu",
            infrastructure_regions={"eu-west-1", "eu-central-1"},
        )

        report = validator.validate(config, principal_id)

        assert report.overall_result == PostureCheckResult.PASS
        assert report.failed_count == 0

    def test_validate_eu_residency_fail(self, validator, workspace_id, principal_id):
        """Test EU residency validation fails for non-EU regions."""
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            data_residency="us",
            infrastructure_regions={"us-east-1"},
        )

        report = validator.validate(config, principal_id)

        assert report.overall_result == PostureCheckResult.FAIL
        assert report.failed_count > 0

        # Check for residency violations
        residency_failures = [
            c for c in report.checks
            if c.category == "residency" and c.result == PostureCheckResult.FAIL
        ]
        assert len(residency_failures) > 0

    def test_validate_on_prem_telemetry_local(self, validator, workspace_id, principal_id):
        """Test on-prem mode requires local telemetry."""
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            telemetry_destination=TelemetryDestination.LOCAL,
            telemetry_local_only=True,
            telemetry_cloud_export_enabled=False,
            data_residency="on-prem",
            infrastructure_regions={"on-prem"},
        )

        report = validator.validate(config, principal_id)

        assert report.overall_result in [PostureCheckResult.PASS, PostureCheckResult.WARNING]
        assert report.is_compliant is True

    def test_validate_on_prem_cloud_telemetry_fails(self, validator, workspace_id, principal_id):
        """Test on-prem mode with cloud telemetry fails."""
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            telemetry_destination=TelemetryDestination.CLOUD,
            telemetry_local_only=False,
            telemetry_cloud_export_enabled=True,
            data_residency="on-prem",
            infrastructure_regions={"on-prem"},
        )

        report = validator.validate(config, principal_id)

        assert report.failed_count > 0

    def test_validate_air_gapped(self, validator, workspace_id, principal_id):
        """Test air-gapped mode validation."""
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            deployment_mode=EnterpriseDeploymentMode.AIR_GAPPED,
            telemetry_destination=TelemetryDestination.LOCAL,
            telemetry_local_only=True,
            telemetry_cloud_export_enabled=False,
            air_gapped=True,
            offline_verification_enabled=True,
            update_source=UpdateSource.OFFLINE,
            data_residency="on-prem",
            infrastructure_regions={"on-prem"},
        )

        report = validator.validate(config, principal_id)

        assert report.is_compliant is True

    def test_validate_air_gapped_missing_requirements(self, validator, workspace_id, principal_id):
        """Test air-gapped mode fails without required settings."""
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            deployment_mode=EnterpriseDeploymentMode.AIR_GAPPED,
            air_gapped=False,  # Should be True
            offline_verification_enabled=False,  # Should be True
            update_source=UpdateSource.CLOUD,  # Should be OFFLINE
            data_residency="on-prem",
            infrastructure_regions={"on-prem"},
        )

        report = validator.validate(config, principal_id)

        assert report.failed_count > 0
        assert report.is_compliant is False

    def test_validate_raw_telemetry_enterprise_only(self, validator, workspace_id, principal_id):
        """Test RAW telemetry requires enterprise deployment."""
        # SaaS with RAW should fail
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            deployment_mode=EnterpriseDeploymentMode.SAAS,
            raw_telemetry_enabled=True,
            data_residency="eu",
            infrastructure_regions={"eu-west-1"},
        )

        report = validator.validate(config, principal_id)

        # Find RAW telemetry check
        raw_checks = [c for c in report.checks if "raw" in c.name.lower()]
        assert any(c.result == PostureCheckResult.FAIL for c in raw_checks)

    def test_validate_raw_telemetry_opt_in_required(self, validator, workspace_id, principal_id):
        """Test RAW telemetry requires opt-in record."""
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            deployment_mode=EnterpriseDeploymentMode.ENTERPRISE_CLOUD,
            raw_telemetry_enabled=True,
            raw_telemetry_opt_in_date=None,  # Missing opt-in
            data_residency="eu",
            infrastructure_regions={"eu-west-1"},
        )

        report = validator.validate(config, principal_id)

        opt_in_checks = [c for c in report.checks if "opt_in" in c.name.lower()]
        assert any(c.result == PostureCheckResult.FAIL for c in opt_in_checks)

    def test_validate_reports_stored(self, validator, workspace_id, principal_id):
        """Test validation reports are stored and retrievable."""
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            data_residency="eu",
            infrastructure_regions={"eu-west-1"},
        )

        report = validator.validate(config, principal_id)

        # Retrieve report
        retrieved = validator.get_report(report.report_id)
        assert retrieved is not None
        assert retrieved.report_id == report.report_id

        # List reports
        reports = validator.list_reports(workspace_id=workspace_id)
        assert len(reports) >= 1


# =============================================================================
# Test EnterpriseEvidencePackExporter
# =============================================================================

class TestEnterpriseEvidencePackExporter:
    """Tests for EnterpriseEvidencePackExporter."""

    def test_export_offline(self, evidence_exporter, workspace_id, principal_id, audit_events):
        """Test offline evidence pack export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "evidence-pack.zip")

            pack = evidence_exporter.export_offline(
                workspace_id=workspace_id,
                exported_by=principal_id,
                output_path=output_path,
                sign=True,
            )

            assert pack.workspace_id == workspace_id
            assert pack.export_mode == "offline"
            assert pack.file_path == output_path
            assert os.path.exists(output_path)
            assert pack.file_size_bytes > 0
            assert pack.integrity_hash.startswith("sha256:")
            assert pack.signature is not None

            # Check audit event
            export_events = [e for e in audit_events if e.action == AuditAction.EVIDENCE_EXPORTED]
            assert len(export_events) == 1

    def test_export_offline_categories(self, evidence_exporter, workspace_id, principal_id):
        """Test offline export with specific categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "evidence-pack.zip")

            categories = {"posture_config", "residency_evidence"}
            pack = evidence_exporter.export_offline(
                workspace_id=workspace_id,
                exported_by=principal_id,
                output_path=output_path,
                categories=categories,
                sign=False,
            )

            assert pack.categories_included == categories

    def test_verify_pack_valid(self, evidence_exporter, workspace_id, principal_id):
        """Test verification of valid evidence pack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "evidence-pack.zip")

            # Export pack
            evidence_exporter.export_offline(
                workspace_id=workspace_id,
                exported_by=principal_id,
                output_path=output_path,
            )

            # Verify pack
            valid, details = evidence_exporter.verify_pack(output_path)

            assert valid is True
            assert "manifest" in details
            assert details.get("all_artifacts_valid", False) is True

    def test_verify_pack_not_found(self, evidence_exporter):
        """Test verification of non-existent pack."""
        valid, details = evidence_exporter.verify_pack("/nonexistent/path.zip")

        assert valid is False
        assert "error" in details

    def test_verify_pack_corrupted(self, evidence_exporter, workspace_id, principal_id):
        """Test verification of corrupted pack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "evidence-pack.zip")

            # Export pack
            evidence_exporter.export_offline(
                workspace_id=workspace_id,
                exported_by=principal_id,
                output_path=output_path,
            )

            # Corrupt the pack (overwrite with garbage)
            with open(output_path, "wb") as f:
                f.write(b"corrupted data")

            # Verify pack
            valid, details = evidence_exporter.verify_pack(output_path)

            assert valid is False

    def test_list_packs(self, evidence_exporter, workspace_id, principal_id):
        """Test listing exported packs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export multiple packs
            for i in range(3):
                output_path = os.path.join(tmpdir, f"pack-{i}.zip")
                evidence_exporter.export_offline(
                    workspace_id=workspace_id,
                    exported_by=principal_id,
                    output_path=output_path,
                )

            packs = evidence_exporter.list_packs(workspace_id=workspace_id)
            assert len(packs) == 3


# =============================================================================
# Test EnterprisePostureService
# =============================================================================

class TestEnterprisePostureService:
    """Tests for EnterprisePostureService."""

    def test_configure_on_prem(self, posture_service, workspace_id, organization_id, principal_id, audit_events):
        """Test configuring on-prem deployment."""
        config = posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            configured_by=principal_id,
        )

        assert config.deployment_mode == EnterpriseDeploymentMode.ON_PREM_FULL
        assert config.telemetry_destination == TelemetryDestination.LOCAL
        assert config.telemetry_local_only is True
        assert config.telemetry_cloud_export_enabled is False
        assert config.data_residency == "on-prem"

        # Check audit event
        assert len(audit_events) > 0
        mode_events = [e for e in audit_events if e.action == AuditAction.MODE_CONFIGURED]
        assert len(mode_events) >= 1

    def test_configure_air_gapped(self, posture_service, workspace_id, organization_id, principal_id):
        """Test configuring air-gapped deployment."""
        config = posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.AIR_GAPPED,
            configured_by=principal_id,
        )

        assert config.deployment_mode == EnterpriseDeploymentMode.AIR_GAPPED
        assert config.air_gapped is True
        assert config.offline_verification_enabled is True
        assert config.update_source == UpdateSource.OFFLINE

    def test_configure_enterprise_cloud(self, posture_service, workspace_id, organization_id, principal_id):
        """Test configuring enterprise cloud deployment."""
        config = posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ENTERPRISE_CLOUD,
            configured_by=principal_id,
        )

        assert config.deployment_mode == EnterpriseDeploymentMode.ENTERPRISE_CLOUD
        assert config.key_management == KeyManagementType.CUSTOMER_MANAGED

    def test_configure_vpc_managed(self, posture_service, workspace_id, organization_id, principal_id):
        """Test configuring VPC managed deployment."""
        config = posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.VPC_MANAGED,
            configured_by=principal_id,
        )

        assert config.deployment_mode == EnterpriseDeploymentMode.VPC_MANAGED
        assert config.telemetry_destination == TelemetryDestination.HYBRID

    def test_get_config(self, posture_service, workspace_id, organization_id, principal_id):
        """Test retrieving configuration."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            configured_by=principal_id,
        )

        config = posture_service.get_config(workspace_id)
        assert config is not None
        assert config.workspace_id == workspace_id

    def test_update_config(self, posture_service, workspace_id, organization_id, principal_id):
        """Test updating configuration."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            configured_by=principal_id,
        )

        updated = posture_service.update_config(
            workspace_id=workspace_id,
            updated_by=principal_id,
            evidence_export_path="/new/path",
        )

        assert updated.evidence_export_path == "/new/path"

    def test_enable_raw_telemetry(self, posture_service, workspace_id, organization_id, principal_id, audit_events):
        """Test enabling RAW telemetry for enterprise."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ENTERPRISE_CLOUD,
            configured_by=principal_id,
        )

        success, error = posture_service.enable_raw_telemetry(
            workspace_id=workspace_id,
            organization_id=organization_id,
            enabled_by=principal_id,
            acknowledgment_text="I acknowledge the risks of RAW telemetry...",
        )

        assert success is True
        assert error is None

        config = posture_service.get_config(workspace_id)
        assert config.raw_telemetry_enabled is True
        assert config.raw_telemetry_opt_in_date is not None

        # Check audit event
        raw_events = [e for e in audit_events if e.action == AuditAction.RAW_TELEMETRY_ENABLED]
        assert len(raw_events) == 1

    def test_enable_raw_telemetry_saas_fails(self, posture_service, workspace_id, organization_id, principal_id):
        """Test RAW telemetry fails for SaaS deployment."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.SAAS,
            configured_by=principal_id,
        )

        success, error = posture_service.enable_raw_telemetry(
            workspace_id=workspace_id,
            organization_id=organization_id,
            enabled_by=principal_id,
            acknowledgment_text="...",
        )

        assert success is False
        assert "not available for SaaS" in error

    def test_disable_raw_telemetry(self, posture_service, workspace_id, organization_id, principal_id, audit_events):
        """Test disabling RAW telemetry."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ENTERPRISE_CLOUD,
            configured_by=principal_id,
        )

        posture_service.enable_raw_telemetry(
            workspace_id=workspace_id,
            organization_id=organization_id,
            enabled_by=principal_id,
            acknowledgment_text="...",
        )

        result = posture_service.disable_raw_telemetry(
            workspace_id=workspace_id,
            disabled_by=principal_id,
            reason="No longer needed",
        )

        assert result is True

        config = posture_service.get_config(workspace_id)
        assert config.raw_telemetry_enabled is False

    def test_is_telemetry_local_mode(self, posture_service, workspace_id, organization_id, principal_id):
        """Test checking telemetry local mode."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            configured_by=principal_id,
        )

        assert posture_service.is_telemetry_local_mode(workspace_id) is True

    def test_validate_telemetry_destination(self, posture_service, workspace_id, organization_id, principal_id):
        """Test validating telemetry destination."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            configured_by=principal_id,
        )

        allowed, error = posture_service.validate_telemetry_destination(
            workspace_id, "cloud"
        )

        assert allowed is False
        assert error is not None

    def test_export_evidence_offline(self, posture_service, workspace_id, organization_id, principal_id):
        """Test offline evidence export through service."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            configured_by=principal_id,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "evidence.zip")

            pack = posture_service.export_evidence_offline(
                workspace_id=workspace_id,
                exported_by=principal_id,
                output_path=output_path,
            )

            assert pack is not None
            assert os.path.exists(output_path)

    def test_validate_posture(self, posture_service, workspace_id, organization_id, principal_id):
        """Test posture validation through service."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            configured_by=principal_id,
        )

        report = posture_service.validate_posture(
            workspace_id=workspace_id,
            validated_by=principal_id,
        )

        assert report is not None
        assert report.workspace_id == workspace_id

    def test_validate_posture_no_config(self, posture_service, principal_id):
        """Test validation fails gracefully without config."""
        report = posture_service.validate_posture(
            workspace_id="nonexistent",
            validated_by=principal_id,
        )

        assert report.overall_result == PostureCheckResult.FAIL

    def test_export_posture_evidence(self, posture_service, workspace_id, organization_id, principal_id):
        """Test exporting posture evidence for evidence pack."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            configured_by=principal_id,
        )

        posture_service.validate_posture(workspace_id, principal_id)

        evidence = posture_service.export_posture_evidence(workspace_id)

        assert evidence["workspace_id"] == workspace_id
        assert evidence["posture_config"] is not None
        assert "latest_validation" in evidence

    def test_get_events(self, posture_service, workspace_id, organization_id, principal_id):
        """Test retrieving audit events."""
        posture_service.configure(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
            configured_by=principal_id,
        )

        events = posture_service.get_events(workspace_id=workspace_id)
        assert len(events) > 0


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_on_prem_posture(self, workspace_id, organization_id, principal_id):
        """Test on-prem posture factory."""
        config = create_on_prem_posture(
            workspace_id=workspace_id,
            organization_id=organization_id,
            configured_by=principal_id,
            telemetry_storage_path="/custom/telemetry",
            evidence_export_path="/custom/evidence",
        )

        assert config.deployment_mode == EnterpriseDeploymentMode.ON_PREM_FULL
        assert config.telemetry_local_only is True
        assert config.data_residency == "on-prem"

    def test_create_air_gapped_posture(self, workspace_id, organization_id, principal_id):
        """Test air-gapped posture factory."""
        config = create_air_gapped_posture(
            workspace_id=workspace_id,
            organization_id=organization_id,
            configured_by=principal_id,
            local_registry_url="registry.internal:5000",
        )

        assert config.deployment_mode == EnterpriseDeploymentMode.AIR_GAPPED
        assert config.air_gapped is True
        assert config.offline_verification_enabled is True


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_eu_regions_contains_expected(self):
        """Test EU_REGIONS contains expected regions."""
        expected = [
            "eu-west-1", "eu-west-2", "eu-central-1",
            "europe-west1", "westeurope", "northeurope",
        ]
        for region in expected:
            assert region in EU_REGIONS

    def test_eu_regions_no_us_regions(self):
        """Test EU_REGIONS does not contain US regions."""
        us_regions = ["us-east-1", "us-west-2", "eastus", "westus"]
        for region in us_regions:
            assert region not in EU_REGIONS


# =============================================================================
# Test Data Class Serialization
# =============================================================================

class TestDataClassSerialization:
    """Tests for data class to_dict methods."""

    def test_posture_check_to_dict(self):
        """Test PostureCheck serialization."""
        check = PostureCheck(
            name="test_check",
            category="security",
            result=PostureCheckResult.PASS,
            message="Check passed",
        )

        data = check.to_dict()
        assert data["name"] == "test_check"
        assert data["result"] == "pass"

    def test_validation_report_to_dict(self, workspace_id):
        """Test PostureValidationReport serialization."""
        report = PostureValidationReport(
            workspace_id=workspace_id,
            overall_result=PostureCheckResult.PASS,
            passed_count=5,
            failed_count=0,
        )

        data = report.to_dict()
        assert data["workspace_id"] == workspace_id
        assert data["overall_result"] == "pass"
        assert data["is_compliant"] is True

    def test_telemetry_local_config_to_dict(self):
        """Test TelemetryLocalModeConfig serialization."""
        config = TelemetryLocalModeConfig(
            storage_path="/data",
            retention_days=90,
        )

        data = config.to_dict()
        assert data["storage_path"] == "/data"
        assert data["retention_days"] == 90

    def test_evidence_pack_to_dict(self, workspace_id):
        """Test EnterpriseEvidencePack serialization."""
        pack = EnterpriseEvidencePack(
            workspace_id=workspace_id,
            export_mode="offline",
        )

        data = pack.to_dict()
        assert data["workspace_id"] == workspace_id
        assert data["export_mode"] == "offline"

    def test_audit_event_to_dict(self, workspace_id, principal_id):
        """Test PostureAuditEvent serialization."""
        event = PostureAuditEvent(
            workspace_id=workspace_id,
            action=AuditAction.MODE_CONFIGURED,
            actor_id=principal_id,
        )

        data = event.to_dict()
        assert data["workspace_id"] == workspace_id
        assert data["action"] == "mode_configured"
        assert data["integrity_hash"].startswith("sha256:")


# =============================================================================
# Test Integrity Hashing
# =============================================================================

class TestIntegrityHashing:
    """Tests for integrity hash computation."""

    def test_validation_report_hash_deterministic(self, workspace_id):
        """Test validation report hash is deterministic."""
        report1 = PostureValidationReport(
            report_id="fixed-id",
            workspace_id=workspace_id,
            overall_result=PostureCheckResult.PASS,
            validated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        report2 = PostureValidationReport(
            report_id="fixed-id",
            workspace_id=workspace_id,
            overall_result=PostureCheckResult.PASS,
            validated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        assert report1.integrity_hash == report2.integrity_hash

    def test_audit_event_hash_changes_with_content(self, workspace_id):
        """Test audit event hash changes with different content."""
        event1 = PostureAuditEvent(
            event_id="fixed-id",
            workspace_id=workspace_id,
            action=AuditAction.MODE_CONFIGURED,
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        event2 = PostureAuditEvent(
            event_id="fixed-id",
            workspace_id=workspace_id,
            action=AuditAction.RAW_TELEMETRY_ENABLED,
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        assert event1.integrity_hash != event2.integrity_hash
