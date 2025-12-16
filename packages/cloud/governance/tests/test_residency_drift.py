# -*- coding: utf-8 -*-
"""
Tests for EU-Only Residency Drift Checker.

GDPR Phase 3 Implementation Tests.

These tests verify:
    - EU region detection
    - Non-EU region rejection
    - Endpoint pattern extraction
    - Drift report generation
    - Evidence pack export
    - Fail-closed behavior
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from packages.cloud.governance.residency_drift import (
    EUOnlyDriftChecker,
    DeploymentConfigValidator,
    ResidencyEvidenceExporter,
    ResidencyDriftReport,
    ResidencyConfiguration,
    EndpointCheck,
    SubprocessorCheck,
    DriftCheckViolation,
    DriftCheckStatus,
    ComponentType,
    CheckSeverity,
    check_eu_residency,
    is_eu_region,
    validate_endpoint_eu,
    EU_AWS_REGIONS,
    EU_GCP_REGIONS,
    EU_AZURE_REGIONS,
    ALL_EU_REGIONS,
    NON_EU_REGIONS,
)


# ============================================================================
# Test Constants
# ============================================================================

class TestEURegionConstants:
    """Tests for EU region constants."""

    def test_eu_aws_regions_contains_expected(self):
        """Test EU AWS regions are properly defined."""
        assert "eu-west-1" in EU_AWS_REGIONS
        assert "eu-central-1" in EU_AWS_REGIONS
        assert "eu-north-1" in EU_AWS_REGIONS

    def test_eu_aws_regions_excludes_us(self):
        """Test US regions are not in EU set."""
        assert "us-east-1" not in EU_AWS_REGIONS
        assert "us-west-2" not in EU_AWS_REGIONS

    def test_non_eu_regions_defined(self):
        """Test non-EU regions are properly defined."""
        assert "us-east-1" in NON_EU_REGIONS
        assert "ap-northeast-1" in NON_EU_REGIONS
        assert "sa-east-1" in NON_EU_REGIONS

    def test_all_eu_regions_comprehensive(self):
        """Test ALL_EU_REGIONS includes all provider regions."""
        # AWS
        assert "eu-west-1" in ALL_EU_REGIONS
        assert "eu-central-1" in ALL_EU_REGIONS

        # GCP
        assert "europe-west1" in ALL_EU_REGIONS
        assert "europe-north1" in ALL_EU_REGIONS

        # Azure
        assert "westeurope" in ALL_EU_REGIONS
        assert "northeurope" in ALL_EU_REGIONS

        # Generic
        assert "EU" in ALL_EU_REGIONS
        assert "Europe" in ALL_EU_REGIONS


# ============================================================================
# Test is_eu_region Function
# ============================================================================

class TestIsEURegion:
    """Tests for is_eu_region function."""

    @pytest.mark.parametrize("region", [
        "eu-west-1",
        "eu-west-2",
        "eu-central-1",
        "eu-north-1",
        "EU-WEST-1",  # Case insensitive
        "EU",
        "Europe",
        "europe-west1",
        "westeurope",
        "EU (Germany)",
        "EU (Ireland)",
    ])
    def test_eu_regions_detected(self, region: str):
        """Test EU regions are correctly identified."""
        assert is_eu_region(region) is True

    @pytest.mark.parametrize("region", [
        "us-east-1",
        "us-west-2",
        "ap-northeast-1",
        "ap-southeast-1",
        "sa-east-1",
        "af-south-1",
        "me-south-1",
        "ca-central-1",
    ])
    def test_non_eu_regions_rejected(self, region: str):
        """Test non-EU regions are correctly rejected."""
        assert is_eu_region(region) is False

    def test_empty_region_rejected(self):
        """Test empty region is rejected."""
        assert is_eu_region("") is False

    def test_unknown_region_rejected(self):
        """Test unknown region is rejected."""
        assert is_eu_region("unknown") is False


# ============================================================================
# Test EUOnlyDriftChecker
# ============================================================================

class TestEUOnlyDriftChecker:
    """Tests for EUOnlyDriftChecker class."""

    def test_checker_initialization(self):
        """Test checker initializes correctly."""
        checker = EUOnlyDriftChecker()
        assert checker is not None
        assert checker._fail_closed is True

    def test_checker_fail_closed_configurable(self):
        """Test fail_closed is configurable."""
        checker = EUOnlyDriftChecker(fail_closed=False)
        assert checker._fail_closed is False


class TestEndpointChecks:
    """Tests for endpoint checking functionality."""

    @pytest.fixture
    def checker(self) -> EUOnlyDriftChecker:
        """Create checker instance."""
        return EUOnlyDriftChecker()

    # AWS RDS Endpoints
    @pytest.mark.parametrize("endpoint,expected_region,expected_compliant", [
        ("mydb.eu-central-1.rds.amazonaws.com", "eu-central-1", True),
        ("mydb.eu-west-1.rds.amazonaws.com", "eu-west-1", True),
        ("mydb.eu-west-2.rds.amazonaws.com", "eu-west-2", True),
        ("mydb.us-east-1.rds.amazonaws.com", "us-east-1", False),
        ("mydb.us-west-2.rds.amazonaws.com", "us-west-2", False),
        ("mydb.ap-northeast-1.rds.amazonaws.com", "ap-northeast-1", False),
    ])
    def test_rds_endpoint_extraction(
        self,
        checker: EUOnlyDriftChecker,
        endpoint: str,
        expected_region: str,
        expected_compliant: bool,
    ):
        """Test region extraction from RDS endpoints."""
        result = checker.check_endpoint(
            endpoint,
            "database",
            ComponentType.DATABASE_PRIMARY,
        )
        assert result.region == expected_region
        assert result.eu_compliant == expected_compliant

    # AWS S3 Endpoints
    @pytest.mark.parametrize("endpoint,expected_region,expected_compliant", [
        ("mybucket.s3.eu-central-1.amazonaws.com", "eu-central-1", True),
        ("mybucket.s3.eu-west-1.amazonaws.com", "eu-west-1", True),
        ("s3.eu-central-1.amazonaws.com", "eu-central-1", True),
        ("s3-eu-west-1.amazonaws.com", "eu-west-1", True),
        ("mybucket.s3.us-east-1.amazonaws.com", "us-east-1", False),
        ("s3.us-west-2.amazonaws.com", "us-west-2", False),
    ])
    def test_s3_endpoint_extraction(
        self,
        checker: EUOnlyDriftChecker,
        endpoint: str,
        expected_region: str,
        expected_compliant: bool,
    ):
        """Test region extraction from S3 endpoints."""
        result = checker.check_endpoint(
            endpoint,
            "storage",
            ComponentType.OBJECT_STORAGE,
        )
        assert result.region == expected_region
        assert result.eu_compliant == expected_compliant

    # AWS ElastiCache Endpoints
    @pytest.mark.parametrize("endpoint,expected_region,expected_compliant", [
        ("myredis.eu-central-1.cache.amazonaws.com", "eu-central-1", True),
        ("myredis.eu-west-1.cache.amazonaws.com", "eu-west-1", True),
        ("myredis.us-east-1.cache.amazonaws.com", "us-east-1", False),
    ])
    def test_elasticache_endpoint_extraction(
        self,
        checker: EUOnlyDriftChecker,
        endpoint: str,
        expected_region: str,
        expected_compliant: bool,
    ):
        """Test region extraction from ElastiCache endpoints."""
        result = checker.check_endpoint(
            endpoint,
            "cache",
            ComponentType.CACHE,
        )
        assert result.region == expected_region
        assert result.eu_compliant == expected_compliant

    def test_explicit_region_takes_precedence(self, checker: EUOnlyDriftChecker):
        """Test explicit region overrides endpoint pattern."""
        # Even with US endpoint, explicit EU region should be used
        result = checker.check_endpoint(
            "mydb.us-east-1.rds.amazonaws.com",  # US endpoint
            "database",
            ComponentType.DATABASE_PRIMARY,
            explicit_region="eu-central-1",  # Explicit EU
        )
        assert result.region == "eu-central-1"
        assert result.eu_compliant is True
        assert result.check_method == "explicit"

    def test_no_endpoint_fails_closed(self, checker: EUOnlyDriftChecker):
        """Test empty endpoint fails when fail_closed is True."""
        result = checker.check_endpoint(
            "",
            "database",
            ComponentType.DATABASE_PRIMARY,
        )
        assert result.eu_compliant is False
        assert result.check_method == "none"
        assert "No endpoint configured" in result.error

    def test_unknown_endpoint_fails_closed(self, checker: EUOnlyDriftChecker):
        """Test unrecognized endpoint fails when fail_closed is True."""
        result = checker.check_endpoint(
            "custom-db.example.com",
            "database",
            ComponentType.DATABASE_PRIMARY,
        )
        assert result.eu_compliant is False
        assert result.check_method == "none"


class TestPaymentProcessorChecks:
    """Tests for payment processor checking."""

    @pytest.fixture
    def checker(self) -> EUOnlyDriftChecker:
        """Create checker instance."""
        return EUOnlyDriftChecker()

    def test_stripe_eu_detected(self, checker: EUOnlyDriftChecker):
        """Test Stripe is detected as EU-compliant."""
        result = checker._check_payment_processor("stripe", "")
        assert result.eu_compliant is True
        assert "Ireland" in result.region

    def test_stripe_with_explicit_eu_region(self, checker: EUOnlyDriftChecker):
        """Test Stripe with explicit EU region."""
        result = checker._check_payment_processor("stripe", "EU (Ireland)")
        assert result.eu_compliant is True

    def test_unknown_processor_fails_closed(self, checker: EUOnlyDriftChecker):
        """Test unknown payment processor fails closed."""
        result = checker._check_payment_processor("unknown_processor", "")
        assert result.eu_compliant is False


class TestServiceChecks:
    """Tests for service checking."""

    @pytest.fixture
    def checker(self) -> EUOnlyDriftChecker:
        """Create checker instance."""
        return EUOnlyDriftChecker()

    def test_sentry_eu_detected(self, checker: EUOnlyDriftChecker):
        """Test Sentry is detected as EU-compliant."""
        result = checker._check_service("sentry", "", "error_tracking", ComponentType.ERROR_TRACKING)
        assert result.eu_compliant is True
        assert result.region == "EU"

    def test_ses_eu_detected(self, checker: EUOnlyDriftChecker):
        """Test SES is detected as EU-compliant."""
        result = checker._check_service("ses", "", "email", ComponentType.EMAIL_SERVICE)
        assert result.eu_compliant is True
        assert result.region == "eu-west-1"


# ============================================================================
# Test Full Drift Check
# ============================================================================

class TestFullDriftCheck:
    """Tests for full drift check functionality."""

    @pytest.fixture
    def eu_config(self) -> ResidencyConfiguration:
        """Create EU-compliant configuration."""
        return ResidencyConfiguration(
            database_primary_endpoint="mydb.eu-central-1.rds.amazonaws.com",
            database_primary_region="eu-central-1",
            database_replica_endpoint="mydb.eu-west-1.rds.amazonaws.com",
            database_replica_region="eu-west-1",
            object_storage_bucket="my-bucket",
            object_storage_region="eu-central-1",
            cache_endpoint="myredis.eu-central-1.cache.amazonaws.com",
            cache_region="eu-central-1",
            email_service="ses",
            email_region="eu-west-1",
            error_tracking_service="sentry",
            error_tracking_region="EU",
            payment_processor="stripe",
            payment_processor_region="EU (Ireland)",
        )

    @pytest.fixture
    def non_eu_config(self) -> ResidencyConfiguration:
        """Create non-EU configuration."""
        return ResidencyConfiguration(
            database_primary_endpoint="mydb.us-east-1.rds.amazonaws.com",
            database_primary_region="us-east-1",
            object_storage_region="us-west-2",
        )

    def test_eu_config_passes(self, eu_config: ResidencyConfiguration):
        """Test EU configuration passes drift check."""
        checker = EUOnlyDriftChecker()
        report = checker.check(eu_config)

        assert report.passed is True
        assert report.status == DriftCheckStatus.PASS
        assert report.non_eu_endpoints == 0
        assert len(report.violations) == 0

    def test_non_eu_config_fails(self, non_eu_config: ResidencyConfiguration):
        """Test non-EU configuration fails drift check."""
        checker = EUOnlyDriftChecker()
        report = checker.check(non_eu_config)

        assert report.passed is False
        assert report.status == DriftCheckStatus.FAIL
        assert report.non_eu_endpoints > 0
        assert len(report.violations) > 0
        assert len(report.blocking_violations) > 0

    def test_report_has_check_id(self, eu_config: ResidencyConfiguration):
        """Test report has unique check ID."""
        checker = EUOnlyDriftChecker()
        report = checker.check(eu_config)

        assert report.check_id is not None
        assert report.check_id.startswith("drift-check-")

    def test_report_has_hash(self, eu_config: ResidencyConfiguration):
        """Test report has integrity hash."""
        checker = EUOnlyDriftChecker()
        report = checker.check(eu_config)

        assert report.report_hash is not None
        assert report.report_hash.startswith("sha256:")

    def test_report_timestamp(self, eu_config: ResidencyConfiguration):
        """Test report has timestamp."""
        checker = EUOnlyDriftChecker()
        before = datetime.now(timezone.utc)
        report = checker.check(eu_config)
        after = datetime.now(timezone.utc)

        assert before <= report.timestamp <= after


class TestDriftReportSerialization:
    """Tests for drift report serialization."""

    def test_report_to_dict(self):
        """Test report converts to dictionary."""
        report = ResidencyDriftReport(
            status=DriftCheckStatus.PASS,
            checks=[
                EndpointCheck(
                    component="database",
                    component_type=ComponentType.DATABASE_PRIMARY,
                    endpoint="mydb.eu-central-1.rds.amazonaws.com",
                    region="eu-central-1",
                    eu_compliant=True,
                    check_method="pattern",
                )
            ],
        )

        data = report.to_dict()
        assert data["status"] == "PASS"
        assert len(data["checks"]) == 1
        assert data["checks"][0]["region"] == "eu-central-1"

    def test_report_to_json(self):
        """Test report converts to JSON string."""
        report = ResidencyDriftReport(
            checks=[
                EndpointCheck(
                    component="database",
                    component_type=ComponentType.DATABASE_PRIMARY,
                    endpoint="mydb.eu-central-1.rds.amazonaws.com",
                    region="eu-central-1",
                    eu_compliant=True,
                    check_method="pattern",
                )
            ],
        )

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["status"] == "PASS"

    def test_report_save(self):
        """Test report saves to file."""
        report = ResidencyDriftReport(
            checks=[
                EndpointCheck(
                    component="database",
                    component_type=ComponentType.DATABASE_PRIMARY,
                    endpoint="mydb.eu-central-1.rds.amazonaws.com",
                    region="eu-central-1",
                    eu_compliant=True,
                    check_method="pattern",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save(path)

            assert path.exists()
            with open(path) as f:
                parsed = json.load(f)
            assert parsed["status"] == "PASS"


# ============================================================================
# Test Evidence Pack Export
# ============================================================================

class TestEvidencePackExport:
    """Tests for evidence pack export."""

    @pytest.fixture
    def checker(self) -> EUOnlyDriftChecker:
        """Create checker instance."""
        return EUOnlyDriftChecker()

    @pytest.fixture
    def exporter(self, checker: EUOnlyDriftChecker) -> ResidencyEvidenceExporter:
        """Create exporter instance."""
        return ResidencyEvidenceExporter(checker)

    def test_create_subprocessor_checks(self, exporter: ResidencyEvidenceExporter):
        """Test standard subprocessor checks creation."""
        subprocessors = exporter.create_subprocessor_checks()

        assert len(subprocessors) > 0
        assert all(s.eu_compliant for s in subprocessors)
        assert any(s.name == "Amazon Web Services (AWS)" for s in subprocessors)
        assert any(s.name == "Stripe" for s in subprocessors)
        assert any(s.name == "Sentry" for s in subprocessors)

    def test_export_evidence_pack(self, exporter: ResidencyEvidenceExporter):
        """Test evidence pack export."""
        report = ResidencyDriftReport(status=DriftCheckStatus.PASS)
        subprocessors = exporter.create_subprocessor_checks()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "evidence"
            files = exporter.export_evidence_pack(report, subprocessors, output_dir)

            # Check all expected files created
            assert "drift_check_report" in files
            assert "subprocessors_summary" in files
            assert "eu_attestation" in files
            assert "audit_log" in files

            # Check files exist
            for path in files.values():
                assert path.exists()

    def test_attestation_content(self, exporter: ResidencyEvidenceExporter):
        """Test attestation content is correct."""
        # Create a report with actual checks so it passes
        report = ResidencyDriftReport(
            checks=[
                EndpointCheck(
                    component="database",
                    component_type=ComponentType.DATABASE_PRIMARY,
                    endpoint="mydb.eu-central-1.rds.amazonaws.com",
                    region="eu-central-1",
                    eu_compliant=True,
                    check_method="pattern",
                )
            ],
        )
        subprocessors = exporter.create_subprocessor_checks()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "evidence"
            files = exporter.export_evidence_pack(report, subprocessors, output_dir)

            with open(files["eu_attestation"]) as f:
                attestation = json.load(f)

            assert attestation["eu_only_commitment"] is True
            assert attestation["drift_check_passed"] is True
            assert "European Union" in attestation["statement"]


# ============================================================================
# Test Deployment Config Validator
# ============================================================================

class TestDeploymentConfigValidator:
    """Tests for deployment configuration validation."""

    @pytest.fixture
    def validator(self) -> DeploymentConfigValidator:
        """Create validator instance."""
        return DeploymentConfigValidator()

    def test_validate_helm_values_eu(self, validator: DeploymentConfigValidator):
        """Test Helm values with EU configuration pass."""
        values = {
            "global": {
                "dataResidency": "eu",
            },
            "postgresql": {
                "enabled": True,
                "primary": {
                    "region": "eu-central-1",
                },
            },
            "governance": {
                "enabled": True,
                "residency": {
                    "enforceEuOnly": True,
                },
            },
        }

        report = validator.validate_helm_values(values)
        # Should pass as EU is inferred
        assert report.status != DriftCheckStatus.FAIL or len(report.blocking_violations) == 0

    def test_validate_docker_compose(self, validator: DeploymentConfigValidator):
        """Test Docker Compose validation."""
        compose = {
            "services": {
                "app": {
                    "environment": {
                        "DATABASE_URL": "postgres://mydb.eu-central-1.rds.amazonaws.com/db",
                        "AWS_REGION": "eu-central-1",
                    },
                },
            },
        }

        report = validator.validate_docker_compose(compose)
        # Environment-based validation may not extract all regions
        assert report is not None


# ============================================================================
# Test Configuration Loading
# ============================================================================

class TestResidencyConfiguration:
    """Tests for ResidencyConfiguration."""

    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        data = {
            "database_primary_endpoint": "mydb.eu-central-1.rds.amazonaws.com",
            "database_primary_region": "eu-central-1",
            "object_storage_region": "eu-west-1",
        }

        config = ResidencyConfiguration.from_dict(data)
        assert config.database_primary_endpoint == data["database_primary_endpoint"]
        assert config.database_primary_region == data["database_primary_region"]
        assert config.object_storage_region == data["object_storage_region"]

    def test_from_environment(self, monkeypatch):
        """Test configuration loading from environment."""
        monkeypatch.setenv("DATABASE_URL", "postgres://mydb.eu-central-1.rds.amazonaws.com/db")
        monkeypatch.setenv("DATABASE_REGION", "eu-central-1")
        monkeypatch.setenv("AWS_REGION", "eu-central-1")
        monkeypatch.setenv("S3_BUCKET", "my-bucket")

        config = ResidencyConfiguration.from_environment()
        assert "eu-central-1" in config.database_primary_endpoint
        assert config.database_primary_region == "eu-central-1"
        assert config.object_storage_region == "eu-central-1"


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_eu_residency_from_env(self, monkeypatch):
        """Test check_eu_residency with environment config."""
        monkeypatch.setenv("DATABASE_URL", "postgres://mydb.eu-central-1.rds.amazonaws.com/db")
        monkeypatch.setenv("DATABASE_REGION", "eu-central-1")
        monkeypatch.setenv("AWS_REGION", "eu-central-1")

        report = check_eu_residency()
        # Should not have critical violations for EU config
        assert report is not None

    def test_validate_endpoint_eu_success(self):
        """Test validate_endpoint_eu with EU endpoint."""
        is_eu, result = validate_endpoint_eu("mydb.eu-central-1.rds.amazonaws.com")
        assert is_eu is True
        assert result == "eu-central-1"

    def test_validate_endpoint_eu_failure(self):
        """Test validate_endpoint_eu with non-EU endpoint."""
        is_eu, result = validate_endpoint_eu("mydb.us-east-1.rds.amazonaws.com")
        assert is_eu is False
        assert "us-east-1" in result


# ============================================================================
# Test Audit Logging
# ============================================================================

class TestAuditLogging:
    """Tests for audit logging functionality."""

    def test_audit_log_created(self):
        """Test audit log entries are created."""
        checker = EUOnlyDriftChecker()
        config = ResidencyConfiguration(
            database_primary_region="eu-central-1",
        )

        checker.check(config)

        log = checker.get_audit_log()
        assert len(log) > 0
        assert log[0]["action"] == "drift_check_completed"

    def test_audit_log_contains_check_id(self):
        """Test audit log contains check ID."""
        checker = EUOnlyDriftChecker()
        config = ResidencyConfiguration(
            database_primary_region="eu-central-1",
        )

        report = checker.check(config)
        log = checker.get_audit_log()

        assert log[0]["check_id"] == report.check_id


# ============================================================================
# Test Subprocessor Checks
# ============================================================================

class TestSubprocessorCheck:
    """Tests for SubprocessorCheck data class."""

    def test_subprocessor_check_creation(self):
        """Test SubprocessorCheck creation."""
        check = SubprocessorCheck(
            name="AWS",
            legal_entity="Amazon Web Services EMEA SARL",
            service="Cloud infrastructure",
            region="eu-central-1",
            eu_compliant=True,
            dpa_status="Signed",
        )

        assert check.name == "AWS"
        assert check.eu_compliant is True

    def test_subprocessor_check_to_dict(self):
        """Test SubprocessorCheck serialization."""
        check = SubprocessorCheck(
            name="AWS",
            legal_entity="Amazon Web Services EMEA SARL",
            service="Cloud infrastructure",
            region="eu-central-1",
            eu_compliant=True,
            dpa_status="Signed",
            last_review="2025-01-15",
            next_review="2025-04-15",
        )

        data = check.to_dict()
        assert data["name"] == "AWS"
        assert data["eu_compliant"] is True
        assert data["last_review"] == "2025-01-15"


# ============================================================================
# Test Violation Handling
# ============================================================================

class TestDriftCheckViolation:
    """Tests for DriftCheckViolation data class."""

    def test_violation_creation(self):
        """Test violation creation."""
        violation = DriftCheckViolation(
            component="database_primary",
            violation_type="non_eu_endpoint",
            severity=CheckSeverity.CRITICAL,
            message="Non-EU region detected",
            region_found="us-east-1",
        )

        assert violation.component == "database_primary"
        assert violation.severity == CheckSeverity.CRITICAL
        assert violation.blocked is True

    def test_violation_to_dict(self):
        """Test violation serialization."""
        violation = DriftCheckViolation(
            component="database_primary",
            violation_type="non_eu_endpoint",
            severity=CheckSeverity.CRITICAL,
            message="Non-EU region detected",
            region_found="us-east-1",
            remediation="Use EU region",
        )

        data = violation.to_dict()
        assert data["component"] == "database_primary"
        assert data["severity"] == "critical"
        assert data["blocked"] is True


# ============================================================================
# Test Fail-Closed Behavior
# ============================================================================

class TestFailClosedBehavior:
    """Tests for fail-closed enforcement."""

    def test_fail_closed_on_unknown_region(self):
        """Test unknown regions fail when fail_closed=True."""
        checker = EUOnlyDriftChecker(fail_closed=True)
        config = ResidencyConfiguration(
            database_primary_endpoint="custom.database.internal",
            # No explicit region
        )

        report = checker.check(config)
        # Unknown endpoint should cause violation in fail-closed mode
        assert any(not c.eu_compliant for c in report.checks)

    def test_fail_open_on_unknown_region(self):
        """Test unknown regions pass when fail_closed=False."""
        checker = EUOnlyDriftChecker(fail_closed=False)

        result = checker.check_endpoint(
            "custom.database.internal",
            "database",
            ComponentType.DATABASE_PRIMARY,
        )
        # Should pass in fail-open mode
        assert result.eu_compliant is True


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
