# -*- coding: utf-8 -*-
"""
Tests for Enterprise Posture CI Guardrail - Phase 9 GDPR Implementation.

Comprehensive test coverage for:
    - EU residency detection
    - Telemetry configuration validation
    - Air-gapped mode requirements
    - On-prem mode requirements
    - External URL detection
    - Mode auto-detection
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import yaml

from ccea.guardrails.enterprise_posture_check import (
    EnterprisePostureCheck,
    DeploymentMode,
    ViolationSeverity,
    PostureViolation,
    PostureCheckReport,
    EU_REGIONS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def check() -> EnterprisePostureCheck:
    """Create posture check instance."""
    return EnterprisePostureCheck(mode=DeploymentMode.AUTO)


@pytest.fixture
def on_prem_check() -> EnterprisePostureCheck:
    """Create posture check for on-prem mode."""
    return EnterprisePostureCheck(mode=DeploymentMode.ON_PREM)


@pytest.fixture
def air_gapped_check() -> EnterprisePostureCheck:
    """Create posture check for air-gapped mode."""
    return EnterprisePostureCheck(mode=DeploymentMode.AIR_GAPPED)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_yaml_file(path: Path, content: Dict) -> Path:
    """Helper to create YAML file."""
    with open(path, "w") as f:
        yaml.dump(content, f)
    return path


def create_env_file(path: Path, content: str) -> Path:
    """Helper to create env file."""
    with open(path, "w") as f:
        f.write(content)
    return path


# =============================================================================
# Test EU Residency Detection
# =============================================================================


class TestEUResidencyDetection:
    """Tests for EU residency validation."""

    def test_eu_region_passes(self, check, temp_dir):
        """Test EU regions pass validation."""
        config = {
            "global": {
                "dataResidency": "eu",
                "region": "eu-west-1",
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        residency_violations = [v for v in report.violations if "NON_EU" in v.rule_id]
        assert len(residency_violations) == 0

    def test_us_region_fails(self, check, temp_dir):
        """Test US regions fail validation."""
        config = {
            "global": {
                "dataResidency": "us",
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        residency_violations = [
            v for v in report.violations if v.rule_id == "NON_EU_DATA_RESIDENCY"
        ]
        assert len(residency_violations) == 1
        assert residency_violations[0].severity == ViolationSeverity.CRITICAL

    def test_non_eu_infrastructure_region(self, check, temp_dir):
        """Test non-EU infrastructure region is flagged."""
        config = {
            "database": {
                "region": "us-east-1",
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        region_violations = [v for v in report.violations if v.rule_id == "NON_EU_REGION_VALUE"]
        assert len(region_violations) >= 1

    def test_on_prem_residency_passes(self, check, temp_dir):
        """Test on-prem residency passes."""
        config = {
            "global": {
                "dataResidency": "on-prem",
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        residency_violations = [
            v for v in report.violations if v.rule_id == "NON_EU_DATA_RESIDENCY"
        ]
        assert len(residency_violations) == 0

    def test_env_file_aws_region(self, check, temp_dir):
        """Test env file with non-EU AWS region."""
        env_content = """
CCEA_DATA_RESIDENCY=eu
AWS_REGION=us-west-2
"""
        create_env_file(temp_dir / ".env", env_content)

        report = check.run([temp_dir])

        aws_violations = [v for v in report.violations if v.rule_id == "NON_EU_AWS_REGION"]
        assert len(aws_violations) == 1

    def test_env_file_eu_region_passes(self, check, temp_dir):
        """Test env file with EU region passes."""
        env_content = """
CCEA_DATA_RESIDENCY=eu
AWS_REGION=eu-west-1
"""
        create_env_file(temp_dir / ".env", env_content)

        report = check.run([temp_dir])

        aws_violations = [v for v in report.violations if v.rule_id == "NON_EU_AWS_REGION"]
        assert len(aws_violations) == 0


# =============================================================================
# Test Telemetry Configuration
# =============================================================================


class TestTelemetryConfiguration:
    """Tests for telemetry configuration validation."""

    def test_redaction_mandatory_true_passes(self, check, temp_dir):
        """Test redaction mandatory = true passes."""
        config = {
            "controlPlane": {
                "config": {
                    "CCEA_TELEMETRY_REDACTION_MANDATORY": "true",
                },
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        redaction_violations = [
            v for v in report.violations if v.rule_id == "REDACTION_NOT_MANDATORY"
        ]
        assert len(redaction_violations) == 0

    def test_redaction_mandatory_false_fails(self, check, temp_dir):
        """Test redaction mandatory = false fails."""
        config = {
            "controlPlane": {
                "config": {
                    "CCEA_TELEMETRY_REDACTION_MANDATORY": "false",
                },
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        redaction_violations = [
            v for v in report.violations if v.rule_id == "REDACTION_NOT_MANDATORY"
        ]
        assert len(redaction_violations) == 1
        assert redaction_violations[0].severity == ViolationSeverity.CRITICAL

    def test_on_prem_telemetry_local_required(self, on_prem_check, temp_dir):
        """Test on-prem requires telemetry local mode."""
        config = {
            "global": {
                "dataResidency": "on-prem",
            },
            "telemetryIngester": {
                "config": {
                    "CCEA_TELEMETRY_REDACTION_MANDATORY": "true",
                    # Missing CCEA_TELEMETRY_LOCAL_ONLY
                },
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = on_prem_check.run([temp_dir])

        local_violations = [v for v in report.violations if v.rule_id == "TELEMETRY_NOT_LOCAL"]
        assert len(local_violations) >= 1

    def test_on_prem_telemetry_local_enabled_passes(self, on_prem_check, temp_dir):
        """Test on-prem with telemetry local mode enabled passes."""
        config = {
            "global": {
                "dataResidency": "on-prem",
            },
            "telemetryIngester": {
                "config": {
                    "CCEA_TELEMETRY_REDACTION_MANDATORY": "true",
                    "CCEA_TELEMETRY_LOCAL_ONLY": "true",
                },
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = on_prem_check.run([temp_dir])

        local_violations = [v for v in report.violations if v.rule_id == "TELEMETRY_NOT_LOCAL"]
        assert len(local_violations) == 0


# =============================================================================
# Test Air-Gapped Mode
# =============================================================================


class TestAirGappedMode:
    """Tests for air-gapped mode validation."""

    def test_air_gapped_flag_required(self, air_gapped_check, temp_dir):
        """Test air-gapped flag is required."""
        config = {
            "global": {
                "dataResidency": "on-prem",
                "airgapped": False,  # Should be True
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = air_gapped_check.run([temp_dir])

        air_gap_violations = [v for v in report.violations if v.rule_id == "AIR_GAP_NOT_ENABLED"]
        assert len(air_gap_violations) >= 1

    def test_external_egress_not_allowed(self, air_gapped_check, temp_dir):
        """Test external egress must be disabled for air-gapped."""
        config = {
            "global": {
                "airgapped": True,
                "dataResidency": "on-prem",
            },
            "networkPolicy": {
                "egress": {
                    "allowExternal": True,  # Should be False
                },
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = air_gapped_check.run([temp_dir])

        egress_violations = [v for v in report.violations if v.rule_id == "EXTERNAL_EGRESS_ALLOWED"]
        assert len(egress_violations) >= 1

    def test_external_url_detected(self, air_gapped_check, temp_dir):
        """Test external URLs are detected in air-gapped mode."""
        config = {
            "global": {
                "airgapped": True,
                "dataResidency": "on-prem",
            },
            "registry": {
                "url": "https://registry.docker.io/v2",
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = air_gapped_check.run([temp_dir])

        url_violations = [v for v in report.violations if v.rule_id == "EXTERNAL_URL_IN_AIR_GAP"]
        assert len(url_violations) >= 1

    def test_local_url_allowed(self, air_gapped_check, temp_dir):
        """Test local URLs are allowed in air-gapped mode."""
        config = {
            "global": {
                "airgapped": True,
                "dataResidency": "on-prem",
            },
            "registry": {
                "url": "https://registry.internal.local:5000",
            },
            "networkPolicy": {
                "egress": {
                    "allowExternal": False,
                },
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = air_gapped_check.run([temp_dir])

        url_violations = [v for v in report.violations if v.rule_id == "EXTERNAL_URL_IN_AIR_GAP"]
        assert len(url_violations) == 0

    def test_env_file_external_url(self, air_gapped_check, temp_dir):
        """Test env file with external URL in air-gapped mode."""
        env_content = """
CCEA_AIR_GAPPED_MODE=true
CCEA_DATA_RESIDENCY=on-prem
UPSTREAM_REGISTRY_URL=https://registry-1.docker.io
"""
        create_env_file(temp_dir / ".env", env_content)

        report = air_gapped_check.run([temp_dir])

        url_violations = [v for v in report.violations if v.rule_id == "EXTERNAL_URL_IN_AIR_GAP"]
        assert len(url_violations) >= 1


# =============================================================================
# Test On-Prem Mode
# =============================================================================


class TestOnPremMode:
    """Tests for on-prem mode validation."""

    def test_evidence_export_local_recommended(self, on_prem_check, temp_dir):
        """Test evidence export local is recommended for on-prem."""
        config = {
            "global": {
                "dataResidency": "on-prem",
            },
            "governance": {
                "config": {
                    # Missing CCEA_EVIDENCE_EXPORT_LOCAL_ONLY
                },
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = on_prem_check.run([temp_dir])

        evidence_violations = [
            v for v in report.violations if v.rule_id == "EVIDENCE_EXPORT_NOT_LOCAL"
        ]
        # This is a MEDIUM severity (warning)
        assert len(evidence_violations) >= 1

    def test_complete_on_prem_config_passes(self, on_prem_check, temp_dir):
        """Test complete on-prem configuration passes."""
        config = {
            "global": {
                "dataResidency": "on-prem",
                "enterpriseMode": True,
            },
            "controlPlane": {
                "config": {
                    "CCEA_TELEMETRY_REDACTION_MANDATORY": "true",
                    "CCEA_TELEMETRY_LOCAL_ONLY": "true",
                },
            },
            "telemetryIngester": {
                "config": {
                    "CCEA_TELEMETRY_REDACTION_MANDATORY": "true",
                    "CCEA_TELEMETRY_LOCAL_ONLY": "true",
                },
            },
            "governance": {
                "config": {
                    "CCEA_EVIDENCE_EXPORT_LOCAL_ONLY": "true",
                },
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = on_prem_check.run([temp_dir])

        # Should pass with no critical/high violations
        assert report.critical_count == 0
        assert report.passed is True


# =============================================================================
# Test Mode Auto-Detection
# =============================================================================


class TestModeAutoDetection:
    """Tests for deployment mode auto-detection."""

    def test_detect_air_gapped(self, check, temp_dir):
        """Test auto-detection of air-gapped mode."""
        config = {
            "global": {
                "airgapped": True,
                "dataResidency": "on-prem",
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        assert report.mode_detected == "air_gapped"

    def test_detect_on_prem(self, check, temp_dir):
        """Test auto-detection of on-prem mode."""
        config = {
            "global": {
                "dataResidency": "on-prem",
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        assert report.mode_detected == "on_prem"

    def test_detect_enterprise_cloud(self, check, temp_dir):
        """Test auto-detection of enterprise cloud mode."""
        config = {
            "global": {
                "enterpriseMode": True,
                "dataResidency": "eu",
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        assert report.mode_detected == "enterprise_cloud"

    def test_detect_from_env_file(self, check, temp_dir):
        """Test auto-detection from env file."""
        env_content = """
CCEA_AIR_GAPPED_MODE=true
CCEA_DATA_RESIDENCY=on-prem
"""
        create_env_file(temp_dir / ".env", env_content)

        report = check.run([temp_dir])

        assert report.mode_detected == "air_gapped"


# =============================================================================
# Test Report Generation
# =============================================================================


class TestReportGeneration:
    """Tests for report generation."""

    def test_report_counts_violations(self, check, temp_dir):
        """Test report correctly counts violations by severity."""
        config = {
            "global": {
                "dataResidency": "us",  # Critical
            },
            "controlPlane": {
                "config": {
                    "CCEA_TELEMETRY_REDACTION_MANDATORY": "false",  # Critical
                },
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        assert report.critical_count >= 2
        assert report.should_fail_ci is True
        assert report.passed is False

    def test_report_to_dict(self, check, temp_dir):
        """Test report serialization."""
        config = {
            "global": {
                "dataResidency": "eu",
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])
        data = report.to_dict()

        assert "checked_at" in data
        assert "files_checked" in data
        assert "violations" in data
        assert "should_fail_ci" in data

    def test_report_files_checked(self, check, temp_dir):
        """Test report tracks files checked."""
        config = {"global": {"dataResidency": "eu"}}
        create_yaml_file(temp_dir / "values.yaml", config)
        create_yaml_file(temp_dir / "values-prod.yaml", config)

        report = check.run([temp_dir])

        assert report.files_checked >= 2


# =============================================================================
# Test Violation Details
# =============================================================================


class TestViolationDetails:
    """Tests for violation detail capture."""

    def test_violation_includes_file_path(self, check, temp_dir):
        """Test violations include file path."""
        config = {
            "global": {
                "dataResidency": "us",
            },
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        assert len(report.violations) > 0
        assert str(file_path) in report.violations[0].file_path

    def test_violation_includes_key_path(self, check, temp_dir):
        """Test violations include key path."""
        config = {
            "global": {
                "dataResidency": "us",
            },
        }
        create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        residency_violations = [
            v for v in report.violations if v.rule_id == "NON_EU_DATA_RESIDENCY"
        ]
        assert len(residency_violations) > 0
        assert residency_violations[0].key_path == "global.dataResidency"

    def test_violation_includes_actual_value(self, check, temp_dir):
        """Test violations include actual value."""
        config = {
            "global": {
                "dataResidency": "us-west-2",
            },
        }
        create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        residency_violations = [
            v for v in report.violations if v.rule_id == "NON_EU_DATA_RESIDENCY"
        ]
        assert len(residency_violations) > 0
        assert residency_violations[0].actual_value == "us-west-2"

    def test_env_violation_includes_line_number(self, check, temp_dir):
        """Test env file violations include line number."""
        env_content = """
# Comment
CCEA_DATA_RESIDENCY=eu
AWS_REGION=us-east-1
"""
        create_env_file(temp_dir / ".env", env_content)

        report = check.run([temp_dir])

        aws_violations = [v for v in report.violations if v.rule_id == "NON_EU_AWS_REGION"]
        assert len(aws_violations) > 0
        assert aws_violations[0].line_number is not None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_yaml_file(self, check, temp_dir):
        """Test handling of empty YAML file."""
        file_path = temp_dir / "empty.yaml"
        file_path.write_text("")

        report = check.run([temp_dir])
        # Should not crash
        assert report is not None

    def test_invalid_yaml_file(self, check, temp_dir):
        """Test handling of invalid YAML file."""
        file_path = temp_dir / "invalid.yaml"
        file_path.write_text("invalid: yaml: content: [")

        report = check.run([temp_dir])

        # Should log error but not crash
        parse_errors = [v for v in report.violations if v.rule_id == "YAML_PARSE_ERROR"]
        assert len(parse_errors) >= 1

    def test_nonexistent_path(self, check):
        """Test handling of nonexistent path."""
        report = check.run([Path("/nonexistent/path")])
        # Should not crash
        assert report.files_checked == 0

    def test_multiple_config_files(self, check, temp_dir):
        """Test handling multiple config files."""
        config1 = {
            "global": {"dataResidency": "eu"},
        }
        config2 = {
            "global": {"dataResidency": "us"},  # Violation
        }

        create_yaml_file(temp_dir / "values.yaml", config1)
        create_yaml_file(temp_dir / "values-bad.yaml", config2)

        report = check.run([temp_dir])

        # Should detect violation in second file
        assert report.critical_count >= 1

    def test_specific_config_file(self, check, temp_dir):
        """Test checking specific config file."""
        config = {
            "global": {"dataResidency": "eu"},
        }
        file_path = create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([], config_files=[file_path])

        assert report.files_checked == 1


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_eu_regions_contains_expected(self):
        """Test EU_REGIONS contains expected values."""
        expected = ["eu-west-1", "europe-west1", "westeurope", "on-prem"]
        for region in expected:
            assert region in EU_REGIONS

    def test_eu_regions_no_us(self):
        """Test EU_REGIONS excludes US regions."""
        us_regions = ["us-east-1", "us-west-2", "eastus"]
        for region in us_regions:
            assert region not in EU_REGIONS


# =============================================================================
# Test Violation Severity
# =============================================================================


class TestViolationSeverity:
    """Tests for violation severity handling."""

    def test_critical_fails_ci(self, check, temp_dir):
        """Test critical violations fail CI."""
        config = {
            "global": {"dataResidency": "us"},
        }
        create_yaml_file(temp_dir / "values.yaml", config)

        report = check.run([temp_dir])

        assert report.critical_count >= 1
        assert report.should_fail_ci is True

    def test_high_fails_ci(self, air_gapped_check, temp_dir):
        """Test high severity violations fail CI."""
        config = {
            "global": {
                "airgapped": True,
                "dataResidency": "on-prem",
            },
            "networkPolicy": {
                "egress": {
                    "allowExternal": True,
                },
            },
        }
        create_yaml_file(temp_dir / "values.yaml", config)

        report = air_gapped_check.run([temp_dir])

        assert report.high_count >= 1
        assert report.should_fail_ci is True

    def test_medium_does_not_fail_ci(self, on_prem_check, temp_dir):
        """Test medium severity violations don't fail CI alone."""
        config = {
            "global": {
                "dataResidency": "on-prem",
            },
            "controlPlane": {
                "config": {
                    "CCEA_TELEMETRY_REDACTION_MANDATORY": "true",
                    "CCEA_TELEMETRY_LOCAL_ONLY": "true",
                },
            },
            "telemetryIngester": {
                "config": {
                    "CCEA_TELEMETRY_REDACTION_MANDATORY": "true",
                    "CCEA_TELEMETRY_LOCAL_ONLY": "true",
                },
            },
            # Missing governance config (medium severity)
        }
        create_yaml_file(temp_dir / "values.yaml", config)

        report = on_prem_check.run([temp_dir])

        # Should pass even with medium warnings
        assert report.passed is True
