# -*- coding: utf-8 -*-
"""
Tests for EU-Only Residency Guardrail Check.

GDPR Phase 3 Implementation Tests.

These tests verify:
    - Region detection in config files
    - YAML file scanning
    - Environment file scanning
    - Helm values validation
    - Docker Compose validation
    - Report generation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from ccea.guardrails.residency_check import (
    check_residency_compliance,
    check_helm_values,
    check_docker_compose,
    check_yaml_file,
    check_env_file,
    detect_region_in_string,
    is_eu_region,
    is_non_eu_region,
    generate_report,
    find_config_files,
    CheckResult,
    CheckSeverity,
    ResidencyViolation,
    ResidencyCheckResult,
    EU_AWS_REGIONS,
    NON_EU_AWS_REGIONS,
)


# ============================================================================
# Test Region Detection
# ============================================================================

class TestRegionDetection:
    """Tests for region detection functions."""

    @pytest.mark.parametrize("region", [
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "eu-central-1",
        "eu-north-1",
        "EU-WEST-1",  # Case insensitive
        "europe-west1",  # GCP
        "westeurope",  # Azure
    ])
    def test_is_eu_region_true(self, region: str):
        """Test EU regions are correctly identified."""
        assert is_eu_region(region) is True

    @pytest.mark.parametrize("region", [
        "us-east-1",
        "us-west-2",
        "ap-northeast-1",
        "sa-east-1",
        "ca-central-1",
    ])
    def test_is_eu_region_false(self, region: str):
        """Test non-EU regions are rejected."""
        assert is_eu_region(region) is False

    @pytest.mark.parametrize("region", [
        "us-east-1",
        "us-west-2",
        "us-east-2",
        "ap-northeast-1",
        "ap-southeast-1",
        "sa-east-1",
        "me-south-1",
    ])
    def test_is_non_eu_region_true(self, region: str):
        """Test non-EU regions are correctly identified."""
        assert is_non_eu_region(region) is True

    @pytest.mark.parametrize("region", [
        "eu-west-1",
        "eu-central-1",
        "eu-north-1",
    ])
    def test_is_non_eu_region_false(self, region: str):
        """Test EU regions are not flagged as non-EU."""
        assert is_non_eu_region(region) is False


class TestDetectRegionInString:
    """Tests for region detection in strings."""

    def test_detect_eu_region_in_url(self):
        """Test EU region detection in URL."""
        content = "https://mydb.eu-central-1.rds.amazonaws.com:5432/db"
        eu_regions, non_eu_regions = detect_region_in_string(content)

        assert "eu-central-1" in eu_regions
        assert len(non_eu_regions) == 0

    def test_detect_non_eu_region_in_url(self):
        """Test non-EU region detection in URL."""
        content = "https://mydb.us-east-1.rds.amazonaws.com:5432/db"
        eu_regions, non_eu_regions = detect_region_in_string(content)

        assert "us-east-1" in non_eu_regions

    def test_detect_multiple_regions(self):
        """Test detection of multiple regions."""
        content = "primary: eu-central-1, replica: eu-west-1, backup: us-east-1"
        eu_regions, non_eu_regions = detect_region_in_string(content)

        assert "eu-central-1" in eu_regions
        assert "eu-west-1" in eu_regions
        assert "us-east-1" in non_eu_regions


# ============================================================================
# Test YAML File Checking
# ============================================================================

class TestYAMLFileChecking:
    """Tests for YAML file checking."""

    def test_check_yaml_file_eu_only(self):
        """Test YAML file with only EU regions passes."""
        content = """
database:
  primary:
    region: eu-central-1
    endpoint: mydb.eu-central-1.rds.amazonaws.com
  replica:
    region: eu-west-1
    endpoint: mydb.eu-west-1.rds.amazonaws.com
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()

            violations, eu_regions, non_eu_regions = check_yaml_file(Path(f.name))

            critical = [v for v in violations if v.severity == CheckSeverity.CRITICAL]
            assert len(critical) == 0
            assert len(eu_regions) > 0
            assert len(non_eu_regions) == 0

    def test_check_yaml_file_non_eu_region(self):
        """Test YAML file with non-EU region fails."""
        content = """
database:
  primary:
    region: us-east-1
    endpoint: mydb.us-east-1.rds.amazonaws.com
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()

            violations, eu_regions, non_eu_regions = check_yaml_file(Path(f.name))

            critical = [v for v in violations if v.severity == CheckSeverity.CRITICAL]
            assert len(critical) > 0
            assert len(non_eu_regions) > 0

    def test_check_yaml_file_mixed_regions(self):
        """Test YAML file with mixed regions detects non-EU."""
        content = """
services:
  eu_service:
    region: eu-central-1
  us_service:
    region: us-west-2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()

            violations, eu_regions, non_eu_regions = check_yaml_file(Path(f.name))

            assert len(non_eu_regions) > 0
            assert len(eu_regions) > 0
            assert any(v.severity == CheckSeverity.CRITICAL for v in violations)


# ============================================================================
# Test Environment File Checking
# ============================================================================

class TestEnvFileChecking:
    """Tests for environment file checking."""

    def test_check_env_file_eu_only(self):
        """Test env file with only EU regions passes."""
        content = """
DATABASE_REGION=eu-central-1
AWS_REGION=eu-central-1
S3_REGION=eu-west-1
REDIS_REGION=eu-central-1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(content)
            f.flush()

            violations, eu_regions, non_eu_regions = check_env_file(Path(f.name))

            critical = [v for v in violations if v.severity == CheckSeverity.CRITICAL]
            assert len(critical) == 0
            assert len(eu_regions) > 0

    def test_check_env_file_non_eu_region(self):
        """Test env file with non-EU region fails."""
        content = """
DATABASE_REGION=us-east-1
AWS_REGION=us-east-1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(content)
            f.flush()

            violations, eu_regions, non_eu_regions = check_env_file(Path(f.name))

            critical = [v for v in violations if v.severity == CheckSeverity.CRITICAL]
            assert len(critical) > 0
            assert len(non_eu_regions) > 0

    def test_check_env_file_non_eu_in_url(self):
        """Test env file with non-EU region in URL fails."""
        content = """
DATABASE_URL=postgres://mydb.us-east-1.rds.amazonaws.com:5432/db
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(content)
            f.flush()

            violations, eu_regions, non_eu_regions = check_env_file(Path(f.name))

            assert len(non_eu_regions) > 0


# ============================================================================
# Test Helm Values Checking
# ============================================================================

class TestHelmValuesChecking:
    """Tests for Helm values file checking."""

    def test_check_helm_values_eu_only(self):
        """Test Helm values with EU configuration passes."""
        content = """
global:
  dataResidency: eu

postgresql:
  enabled: true
  primary:
    region: eu-central-1

redis:
  enabled: true
  region: eu-central-1

governance:
  enabled: true
  residency:
    enforceEuOnly: true
    allowedRegions:
      - eu-west-1
      - eu-central-1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()

            result = check_helm_values(Path(f.name))

            assert result.result != CheckResult.FAIL or result.non_eu_regions_found == 0

    def test_check_helm_values_non_eu_region(self):
        """Test Helm values with non-EU region fails."""
        content = """
global:
  dataResidency: us

postgresql:
  enabled: true
  primary:
    region: us-east-1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()

            result = check_helm_values(Path(f.name))

            assert result.non_eu_regions_found > 0

    def test_check_helm_values_eu_enforcement_disabled(self):
        """Test Helm values with EU enforcement disabled fails."""
        content = """
governance:
  residency:
    enforceEuOnly: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()

            result = check_helm_values(Path(f.name))

            # Should have violation for disabled enforcement
            assert any(
                "enforceEuOnly" in v.message.lower() or "eu_enforcement" in v.violation_type
                for v in result.violations
            )


# ============================================================================
# Test Docker Compose Checking
# ============================================================================

class TestDockerComposeChecking:
    """Tests for Docker Compose file checking."""

    def test_check_docker_compose_eu_only(self):
        """Test Docker Compose with EU configuration passes."""
        content = """
version: "3.8"
services:
  app:
    image: myapp:latest
    environment:
      - DATABASE_REGION=eu-central-1
      - AWS_REGION=eu-central-1
      - S3_REGION=eu-west-1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()

            result = check_docker_compose(Path(f.name))

            critical = [v for v in result.violations if v.severity == CheckSeverity.CRITICAL]
            assert len(critical) == 0

    def test_check_docker_compose_non_eu_region(self):
        """Test Docker Compose with non-EU region fails."""
        content = """
version: "3.8"
services:
  app:
    image: myapp:latest
    environment:
      - DATABASE_REGION=us-east-1
      - AWS_REGION=us-west-2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()

            result = check_docker_compose(Path(f.name))

            assert result.non_eu_regions_found > 0


# ============================================================================
# Test Full Compliance Check
# ============================================================================

class TestFullComplianceCheck:
    """Tests for full residency compliance check."""

    def test_check_residency_compliance_empty_dir(self):
        """Test compliance check on empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_residency_compliance(Path(tmpdir))

            # Empty directory should pass (no violations)
            assert result.files_checked == 0

    def test_check_residency_compliance_eu_config(self):
        """Test compliance check with EU-only config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create EU-only config file
            config_path = Path(tmpdir) / "values.yaml"
            config_path.write_text("""
global:
  region: eu-central-1
database:
  region: eu-central-1
""")

            result = check_residency_compliance(Path(tmpdir))

            assert result.files_checked > 0
            assert result.non_eu_regions_found == 0

    def test_check_residency_compliance_non_eu_config(self):
        """Test compliance check with non-EU config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create non-EU config file
            config_path = Path(tmpdir) / "values.yaml"
            config_path.write_text("""
global:
  region: us-east-1
database:
  region: us-west-2
""")

            result = check_residency_compliance(Path(tmpdir))

            assert result.files_checked > 0
            assert result.non_eu_regions_found > 0
            assert result.passed is False


# ============================================================================
# Test Report Generation
# ============================================================================

class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report_json(self):
        """Test JSON report generation."""
        result = ResidencyCheckResult(
            passed=True,
            result=CheckResult.PASS,
            message="All checks passed",
            files_checked=5,
            eu_regions_found=3,
            non_eu_regions_found=0,
        )

        json_str = generate_report(result)
        parsed = json.loads(json_str)

        assert parsed["passed"] is True
        assert parsed["result"] == "pass"
        assert parsed["files_checked"] == 5

    def test_generate_report_with_violations(self):
        """Test report generation with violations."""
        result = ResidencyCheckResult(
            passed=False,
            result=CheckResult.FAIL,
            message="Non-EU regions found",
            violations=[
                ResidencyViolation(
                    file_path="/path/to/file.yaml",
                    line_number=10,
                    violation_type="non_eu_region",
                    message="Non-EU region us-east-1 detected",
                    region_found="us-east-1",
                    severity=CheckSeverity.CRITICAL,
                )
            ],
            files_checked=1,
            non_eu_regions_found=1,
        )

        json_str = generate_report(result)
        parsed = json.loads(json_str)

        assert parsed["passed"] is False
        assert len(parsed["violations"]) == 1
        assert parsed["violations"][0]["region_found"] == "us-east-1"

    def test_generate_report_to_file(self):
        """Test report generation to file."""
        result = ResidencyCheckResult(
            passed=True,
            result=CheckResult.PASS,
            message="All checks passed",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_report(result, output_path)

            assert output_path.exists()
            with open(output_path) as f:
                parsed = json.load(f)
            assert parsed["passed"] is True


# ============================================================================
# Test File Discovery
# ============================================================================

class TestFileDiscovery:
    """Tests for config file discovery."""

    def test_find_config_files_yaml(self):
        """Test finding YAML config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some config files
            (Path(tmpdir) / "values.yaml").touch()
            (Path(tmpdir) / "values-production.yaml").touch()
            (Path(tmpdir) / "docker-compose.yml").touch()

            files = find_config_files(Path(tmpdir))

            assert len(files) >= 3

    def test_find_config_files_excludes_test_dirs(self):
        """Test that test directories are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test directory
            test_dir = Path(tmpdir) / "tests"
            test_dir.mkdir()
            (test_dir / "test_values.yaml").touch()

            # Create real config
            (Path(tmpdir) / "values.yaml").touch()

            files = find_config_files(Path(tmpdir))

            # Test files should be excluded
            assert not any("tests" in str(f) for f in files)


# ============================================================================
# Test Violation Data Class
# ============================================================================

class TestResidencyViolation:
    """Tests for ResidencyViolation data class."""

    def test_violation_creation(self):
        """Test violation creation."""
        violation = ResidencyViolation(
            file_path="/path/to/file.yaml",
            line_number=10,
            violation_type="non_eu_region",
            message="Non-EU region detected",
            region_found="us-east-1",
            severity=CheckSeverity.CRITICAL,
        )

        assert violation.file_path == "/path/to/file.yaml"
        assert violation.line_number == 10
        assert violation.severity == CheckSeverity.CRITICAL

    def test_violation_to_dict(self):
        """Test violation serialization."""
        violation = ResidencyViolation(
            file_path="/path/to/file.yaml",
            line_number=10,
            violation_type="non_eu_region",
            message="Non-EU region detected",
            region_found="us-east-1",
            severity=CheckSeverity.CRITICAL,
            code_snippet="region: us-east-1",
        )

        data = violation.to_dict()

        assert data["file_path"] == "/path/to/file.yaml"
        assert data["severity"] == "critical"
        assert data["code_snippet"] == "region: us-east-1"


# ============================================================================
# Test Result Data Class
# ============================================================================

class TestResidencyCheckResult:
    """Tests for ResidencyCheckResult data class."""

    def test_result_creation(self):
        """Test result creation."""
        result = ResidencyCheckResult(
            passed=True,
            result=CheckResult.PASS,
            message="All checks passed",
            files_checked=5,
            eu_regions_found=3,
            non_eu_regions_found=0,
        )

        assert result.passed is True
        assert result.result == CheckResult.PASS
        assert result.files_checked == 5

    def test_result_to_dict(self):
        """Test result serialization."""
        result = ResidencyCheckResult(
            passed=False,
            result=CheckResult.FAIL,
            message="Non-EU regions found",
            violations=[],
            files_checked=1,
            non_eu_regions_found=1,
        )

        data = result.to_dict()

        assert data["passed"] is False
        assert data["result"] == "fail"
        assert "check_timestamp" in data


# ============================================================================
# Test Constants
# ============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_eu_aws_regions_defined(self):
        """Test EU AWS regions are defined."""
        assert len(EU_AWS_REGIONS) > 0
        assert "eu-west-1" in EU_AWS_REGIONS
        assert "eu-central-1" in EU_AWS_REGIONS

    def test_non_eu_aws_regions_defined(self):
        """Test non-EU AWS regions are defined."""
        assert len(NON_EU_AWS_REGIONS) > 0
        assert "us-east-1" in NON_EU_AWS_REGIONS
        assert "ap-northeast-1" in NON_EU_AWS_REGIONS

    def test_no_overlap_between_eu_and_non_eu(self):
        """Test no overlap between EU and non-EU regions."""
        overlap = EU_AWS_REGIONS & NON_EU_AWS_REGIONS
        assert len(overlap) == 0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
