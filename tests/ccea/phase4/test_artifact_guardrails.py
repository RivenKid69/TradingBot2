# -*- coding: utf-8 -*-
"""
Tests for Artifact CI Guardrails.

Per Design Doc Phase 4:
- artifact-signature-required: Artifact must be signed
- no-secrets-in-artifact: No secrets in artifact content
- valid-manifest-schema: Manifest follows schema
- sbom-present: SBOM is generated
- no-order-payloads: No order-like content in manifest
"""

import json
import pytest
import zipfile
from pathlib import Path

from ccea.guardrails.artifact_check import (
    ArtifactGuardrails,
    GuardrailCheck,
    GuardrailReport,
    CheckSeverity,
    CheckResult,
    run_artifact_guardrails,
)


class TestGuardrailCheck:
    """Tests for GuardrailCheck."""

    def test_check_creation(self):
        """Test check creation."""
        check = GuardrailCheck(
            name="test-check",
            result=CheckResult.PASSED,
            severity=CheckSeverity.ERROR,
            message="Test passed",
        )

        assert check.name == "test-check"
        assert check.result == CheckResult.PASSED
        assert check.severity == CheckSeverity.ERROR

    def test_check_to_dict(self):
        """Test check serialization."""
        check = GuardrailCheck(
            name="test-check",
            result=CheckResult.FAILED,
            severity=CheckSeverity.ERROR,
            message="Test failed",
            details="Missing field",
        )

        data = check.to_dict()

        assert data["name"] == "test-check"
        assert data["result"] == "failed"
        assert data["severity"] == "error"


class TestGuardrailReport:
    """Tests for GuardrailReport."""

    def test_report_creation(self):
        """Test report creation."""
        report = GuardrailReport()

        assert report.passed is True
        assert report.total_errors == 0
        assert report.total_warnings == 0

    def test_add_passing_check(self):
        """Test adding passing check."""
        report = GuardrailReport()

        check = GuardrailCheck(
            name="test",
            result=CheckResult.PASSED,
            severity=CheckSeverity.ERROR,
            message="OK",
        )
        report.add(check)

        assert report.passed is True
        assert len(report.checks) == 1

    def test_add_failing_error_check(self):
        """Test adding failing error check."""
        report = GuardrailReport()

        check = GuardrailCheck(
            name="test",
            result=CheckResult.FAILED,
            severity=CheckSeverity.ERROR,
            message="Failed",
        )
        report.add(check)

        assert report.passed is False
        assert report.total_errors == 1

    def test_add_failing_warning_check(self):
        """Test adding failing warning check."""
        report = GuardrailReport()

        check = GuardrailCheck(
            name="test",
            result=CheckResult.FAILED,
            severity=CheckSeverity.WARNING,
            message="Warning",
        )
        report.add(check)

        assert report.passed is True  # Warnings don't fail
        assert report.total_warnings == 1

    def test_report_summary(self):
        """Test report summary."""
        report = GuardrailReport()

        report.add(
            GuardrailCheck(
                name="err",
                result=CheckResult.FAILED,
                severity=CheckSeverity.ERROR,
                message="Error",
            )
        )
        report.add(
            GuardrailCheck(
                name="warn",
                result=CheckResult.FAILED,
                severity=CheckSeverity.WARNING,
                message="Warning",
            )
        )

        summary = report.summary()

        assert "FAILED" in summary
        assert "1 errors" in summary
        assert "1 warnings" in summary


class TestArtifactGuardrails:
    """Tests for ArtifactGuardrails."""

    @pytest.fixture
    def valid_manifest(self, tmp_path):
        """Create valid manifest."""
        manifest = {
            "schema_version": "1.0.0",
            "artifact_id": "test-artifact",
            "entrypoint": {
                "module": "strategy",
                "class": "TestStrategy",
            },
            "runtime": {
                "python_version": "3.11",
            },
            "deps_lock_digest": "sha256:" + "a" * 64,
            "signature": {
                "algorithm": "ed25519",
                "signature_value": "test_signature",
            },
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    @pytest.fixture
    def artifact_zip(self, tmp_path):
        """Create test artifact zip."""
        artifact_path = tmp_path / "artifact.zip"
        with zipfile.ZipFile(artifact_path, "w") as zf:
            zf.writestr("strategy.py", "class Strategy: pass\n")
            zf.writestr("config.json", '{"setting": "value"}')
        return artifact_path

    def test_guardrails_initialization(self):
        """Test guardrails initialization."""
        guardrails = ArtifactGuardrails()

        assert guardrails.require_signature is True
        assert guardrails.require_sbom is True
        assert guardrails.check_secrets is True

    def test_check_signature_required_present(self, valid_manifest):
        """Test signature check when present."""
        guardrails = ArtifactGuardrails()
        check = guardrails._check_signature_required(valid_manifest)

        assert check.result == CheckResult.PASSED

    def test_check_signature_required_missing(self, tmp_path):
        """Test signature check when missing."""
        manifest = {
            "schema_version": "1.0.0",
            "artifact_id": "test",
            # No signature
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        guardrails = ArtifactGuardrails()
        check = guardrails._check_signature_required(path)

        assert check.result == CheckResult.FAILED
        assert check.severity == CheckSeverity.ERROR

    def test_check_manifest_schema_valid(self, valid_manifest):
        """Test manifest schema check with valid manifest."""
        guardrails = ArtifactGuardrails()
        check = guardrails._check_manifest_schema(valid_manifest)

        assert check.result == CheckResult.PASSED

    def test_check_manifest_schema_invalid(self, tmp_path):
        """Test manifest schema check with invalid manifest."""
        manifest = {
            # Missing required fields
            "artifact_id": "test",
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        guardrails = ArtifactGuardrails()
        check = guardrails._check_manifest_schema(path)

        assert check.result == CheckResult.FAILED

    def test_check_manifest_schema_old_version(self, tmp_path):
        """Test manifest schema check with old version."""
        manifest = {
            "schema_version": "0.1.0",  # Too old
            "artifact_id": "test",
            "entrypoint": {"module": "m", "class": "C"},
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        guardrails = ArtifactGuardrails(min_schema_version="1.0.0")
        check = guardrails._check_manifest_schema(path)

        assert check.result == CheckResult.FAILED

    def test_check_no_order_payloads_clean(self, valid_manifest):
        """Test order payload check with clean manifest."""
        guardrails = ArtifactGuardrails()
        check = guardrails._check_no_order_payloads(valid_manifest)

        assert check.result == CheckResult.PASSED

    def test_check_no_order_payloads_found(self, tmp_path):
        """Test order payload check with prohibited fields."""
        manifest = {
            "schema_version": "1.0.0",
            "artifact_id": "test",
            "order_config": {
                "side": "BUY",
                "quantity": 100,
                "price": 50.0,
            },
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        guardrails = ArtifactGuardrails()
        check = guardrails._check_no_order_payloads(path)

        assert check.result == CheckResult.FAILED
        assert "side" in check.details or "quantity" in check.details

    def test_check_no_prohibited_commands(self, valid_manifest):
        """Test prohibited commands check."""
        guardrails = ArtifactGuardrails()
        check = guardrails._check_no_prohibited_commands(valid_manifest)

        assert check.result == CheckResult.PASSED

    def test_check_no_prohibited_commands_found(self, tmp_path):
        """Test prohibited commands check when found."""
        manifest = {
            "command": "PLACE_ORDER",
            "payload": {},
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        guardrails = ArtifactGuardrails()
        check = guardrails._check_no_prohibited_commands(path)

        assert check.result == CheckResult.FAILED
        assert "PLACE_ORDER" in check.details

    def test_check_no_secrets_clean(self, artifact_zip):
        """Test secrets check with clean artifact."""
        guardrails = ArtifactGuardrails()
        check = guardrails._check_no_secrets(artifact_zip)

        assert check.result == CheckResult.PASSED

    def test_check_no_secrets_found(self, tmp_path):
        """Test secrets check when secrets found."""
        artifact_path = tmp_path / "artifact.zip"
        with zipfile.ZipFile(artifact_path, "w") as zf:
            zf.writestr(
                "config.py",
                """
api_key = "AKIAIOSFODNN7EXAMPLE"
password = "secret123456789"
""",
            )

        guardrails = ArtifactGuardrails()
        check = guardrails._check_no_secrets(artifact_path)

        assert check.result == CheckResult.FAILED
        assert check.severity == CheckSeverity.ERROR

    def test_check_sbom_present_with_ref(self, tmp_path):
        """Test SBOM check with sbom_ref."""
        manifest = {
            "schema_version": "1.0.0",
            "sbom_ref": "sha256:" + "a" * 64,
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        guardrails = ArtifactGuardrails()
        check = guardrails._check_sbom_present(path, None)

        assert check.result == CheckResult.PASSED

    def test_check_sbom_present_missing(self, tmp_path):
        """Test SBOM check when missing."""
        manifest = {
            "schema_version": "1.0.0",
            # No sbom_ref
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        guardrails = ArtifactGuardrails()
        check = guardrails._check_sbom_present(path, None)

        assert check.result == CheckResult.FAILED
        assert check.severity == CheckSeverity.WARNING

    def test_check_artifact_full(self, artifact_zip, valid_manifest):
        """Test full artifact check."""
        guardrails = ArtifactGuardrails()
        report = guardrails.check_artifact(artifact_zip, valid_manifest)

        # Should have multiple checks
        assert len(report.checks) >= 5

    def test_check_manifest_only(self, valid_manifest):
        """Test manifest-only check."""
        guardrails = ArtifactGuardrails()
        report = guardrails.check_manifest_only(valid_manifest)

        # Should have schema-related checks
        check_names = {c.name for c in report.checks}
        assert "valid-manifest-schema" in check_names
        assert "no-order-payloads" in check_names


class TestRunArtifactGuardrails:
    """Tests for run_artifact_guardrails function."""

    def test_run_guardrails_passing(self, tmp_path):
        """Test running guardrails with passing artifact."""
        # Create valid artifact
        artifact_path = tmp_path / "artifact.zip"
        with zipfile.ZipFile(artifact_path, "w") as zf:
            zf.writestr("strategy.py", "class Strategy: pass")

        # Create valid manifest
        manifest = {
            "schema_version": "1.0.0",
            "artifact_id": "test",
            "entrypoint": {"module": "strategy", "class": "Strategy"},
            "runtime": {"python_version": "3.11"},
            "deps_lock_digest": "sha256:" + "a" * 64,
            "signature": {
                "algorithm": "ed25519",
                "signature_value": "sig",
            },
            "sbom_ref": "sha256:" + "b" * 64,
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        passed, report = run_artifact_guardrails(artifact_path, manifest_path, strict=True)

        assert passed is True

    def test_run_guardrails_failing(self, tmp_path):
        """Test running guardrails with failing artifact."""
        artifact_path = tmp_path / "artifact.zip"
        artifact_path.write_bytes(b"test")

        manifest = {
            "artifact_id": "test",
            # Missing required fields
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        passed, report = run_artifact_guardrails(artifact_path, manifest_path, strict=True)

        assert passed is False
        assert report.total_errors > 0
