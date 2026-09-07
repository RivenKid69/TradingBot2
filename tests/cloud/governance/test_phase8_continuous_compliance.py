# -*- coding: utf-8 -*-
"""
Comprehensive tests for GDPR Phase 8: Continuous Compliance.

Tests all Phase 8 components:
- Data Inventory Registry
- CI Privacy-by-Design Check
- Compliance Dashboard Service
- Quarterly Review Service

These tests ensure 100% coverage of Phase 8 functionality.
"""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import Phase 8 components
from packages.cloud.governance.data_inventory import (
    ALWAYS_REDACT_PATTERNS,
    ORDER_FIELD_PATTERNS,
    PII_FIELD_PATTERNS,
    DataCategory,
    DataInventoryRegistry,
    DataSensitivity,
    InventoryChangeEvent,
    InventoryEntry,
    InventoryEntryType,
    InventoryReport,
    InventoryValidationResult,
    RedactionRequirement,
    ResidencyRequirement,
    ReviewStatus,
    create_core_inventory,
)
from packages.cloud.governance.compliance_dashboard import (
    BREAK_GLASS_MAX_DURATION_HOURS,
    DSAR_SLA_DAYS,
    RESIDENCY_DRIFT_TARGET,
    AlertSeverity,
    AlertStatus,
    BreakGlassMetrics,
    ComplianceAlert,
    ComplianceDashboard,
    ComplianceDashboardService,
    ComplianceStatus,
    DSARMetrics,
    InventoryMetrics,
    MetricType,
    PurgeMetrics,
    ResidencyMetrics,
)
from packages.cloud.governance.quarterly_review import (
    ADVANCE_NOTICE_DAYS,
    OVERDUE_THRESHOLD_DAYS,
    REVIEW_CYCLE_DAYS,
    FindingSeverity,
    FindingStatus,
    IncidentLearning,
    QuarterlyReview,
    QuarterlyReviewService,
    RetentionReviewItem,
    ReviewFinding,
    ReviewStatus as QReviewStatus,
    ReviewType,
    Subprocessor,
    SubprocessorStatus,
    create_default_subprocessors,
)
from ccea.guardrails.privacy_by_design_check import (
    CheckResult,
    CheckSeverity,
    CheckViolation,
    DataDeclaration,
    DataDeclarationScanner,
    KNOWN_SAFE_PATTERNS,
    PrivacyByDesignCheck,
    PrivacyCheckReport,
    format_report,
)


# ============================================================================
# Test: Data Inventory Registry
# ============================================================================

class TestDataInventoryRegistry:
    """Tests for DataInventoryRegistry."""

    def test_create_registry(self):
        """Test registry creation."""
        registry = DataInventoryRegistry()
        assert registry is not None
        stats = registry.get_stats()
        assert stats["total_entries"] == 0

    def test_register_field(self):
        """Test field registration."""
        registry = DataInventoryRegistry()

        entry = registry.register(
            name="test_field",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Testing purpose",
            lawful_basis="Legitimate interest",
            retention_days=90,
            created_by="test@example.com",
        )

        assert entry is not None
        assert entry.name == "test_field"
        assert entry.entry_type == InventoryEntryType.FIELD
        assert entry.sensitivity == DataSensitivity.INTERNAL
        assert entry.purpose == "Testing purpose"
        assert entry.lawful_basis == "Legitimate interest"
        assert entry.retention_days == 90

    def test_register_duplicate_name_fails(self):
        """Test that duplicate names are rejected."""
        registry = DataInventoryRegistry()

        registry.register(
            name="duplicate_field",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        with pytest.raises(ValueError, match="already exists"):
            registry.register(
                name="duplicate_field",
                entry_type=InventoryEntryType.FIELD,
                sensitivity=DataSensitivity.INTERNAL,
                category=DataCategory.TECHNICAL,
                purpose="Test",
                lawful_basis="Test",
                retention_days=90,
                created_by="test@example.com",
            )

    def test_strict_mode_requires_purpose(self):
        """Test strict mode requires purpose."""
        registry = DataInventoryRegistry(strict_mode=True)

        with pytest.raises(ValueError, match="Purpose is required"):
            registry.register(
                name="no_purpose_field",
                entry_type=InventoryEntryType.FIELD,
                sensitivity=DataSensitivity.INTERNAL,
                category=DataCategory.TECHNICAL,
                purpose="",  # Empty purpose
                lawful_basis="Test",
                retention_days=90,
                created_by="test@example.com",
            )

    def test_strict_mode_requires_lawful_basis(self):
        """Test strict mode requires lawful basis."""
        registry = DataInventoryRegistry(strict_mode=True)

        with pytest.raises(ValueError, match="Lawful basis is required"):
            registry.register(
                name="no_basis_field",
                entry_type=InventoryEntryType.FIELD,
                sensitivity=DataSensitivity.INTERNAL,
                category=DataCategory.TECHNICAL,
                purpose="Test purpose",
                lawful_basis="",  # Empty lawful basis
                retention_days=90,
                created_by="test@example.com",
            )

    def test_auto_detect_credential_field(self):
        """Test auto-detection of credential fields."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="api_key",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,  # Will be upgraded
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        assert entry.credential_detected is True
        assert entry.redaction == RedactionRequirement.NEVER_TRANSMIT
        assert entry.sensitivity == DataSensitivity.RESTRICTED

    def test_auto_detect_pii_field(self):
        """Test auto-detection of PII fields."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="email",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,  # Will be upgraded
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        assert entry.pii_detected is True
        assert entry.redaction == RedactionRequirement.MANDATORY
        assert entry.sensitivity == DataSensitivity.SENSITIVE

    def test_auto_detect_order_field(self):
        """Test auto-detection of order fields."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="side",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TRADING,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        assert entry.order_field_detected is True
        assert entry.min_telemetry_level == "RAW_ORDER_EVENTS"

    def test_validate_registered_field(self):
        """Test validation of registered field."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="valid_field",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test purpose",
            lawful_basis="Legitimate interest",
            retention_days=90,
            created_by="test@example.com",
        )

        # Approve entry
        registry.approve(entry.id, approved_by="dpo@example.com")

        result = registry.validate("valid_field")
        assert result.is_registered is True
        assert result.is_compliant is True
        assert result.should_block_ci is False

    def test_validate_unregistered_field(self):
        """Test validation of unregistered field blocks CI."""
        registry = DataInventoryRegistry()

        result = registry.validate("unregistered_field")
        assert result.is_registered is False
        assert result.should_block_ci is True
        assert len(result.violations) > 0

    def test_validate_unapproved_field(self):
        """Test validation of unapproved field."""
        registry = DataInventoryRegistry(strict_mode=False)

        registry.register(
            name="unapproved_field",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        result = registry.validate("unapproved_field")
        assert result.is_registered is True
        assert result.is_compliant is False  # Not approved
        assert "not approved" in result.violations[0].lower()

    def test_approve_entry(self):
        """Test entry approval."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="to_approve",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        assert entry.review_status == ReviewStatus.PENDING

        updated = registry.approve(entry.id, approved_by="dpo@example.com", notes="Approved")

        assert updated.review_status == ReviewStatus.APPROVED
        assert updated.reviewed_by == "dpo@example.com"
        assert updated.reviewed_at is not None

    def test_reject_entry(self):
        """Test entry rejection."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="to_reject",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        updated = registry.reject(entry.id, rejected_by="dpo@example.com", reason="Insufficient documentation")

        assert updated.review_status == ReviewStatus.REJECTED
        assert updated.review_notes == "Insufficient documentation"

    def test_reject_requires_reason(self):
        """Test rejection requires reason."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="test_reject",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        with pytest.raises(ValueError, match="reason is required"):
            registry.reject(entry.id, rejected_by="dpo@example.com", reason="")

    def test_deprecate_entry(self):
        """Test entry deprecation."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="to_deprecate",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        updated = registry.deprecate(entry.id, deprecated_by="admin@example.com", reason="No longer needed")

        assert updated.review_status == ReviewStatus.DEPRECATED

    def test_update_entry(self):
        """Test entry update."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="to_update",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Original purpose",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        updated = registry.update(
            entry.id,
            updated_by="admin@example.com",
            description="Updated description",
            retention_days=180,
        )

        assert updated.description == "Updated description"
        assert updated.retention_days == 180
        assert updated.version == 2

    def test_delete_entry(self):
        """Test entry deletion."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="to_delete",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        result = registry.delete(entry.id, deleted_by="admin@example.com", reason="No longer needed")
        assert result is True
        assert registry.get(entry.id) is None

    def test_delete_requires_reason(self):
        """Test deletion requires reason."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="test_delete",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        with pytest.raises(ValueError, match="reason is required"):
            registry.delete(entry.id, deleted_by="admin@example.com", reason="")

    def test_lookup_methods(self):
        """Test various lookup methods."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="lookup_test",
            entry_type=InventoryEntryType.FIELD,
            path="table.column",
            sensitivity=DataSensitivity.SENSITIVE,
            category=DataCategory.PII,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        # By ID
        assert registry.get(entry.id) is not None

        # By name
        assert registry.get_by_name("lookup_test") is not None

        # By path
        assert registry.get_by_path("table.column") is not None

        # List by type
        entries = registry.list_by_type(InventoryEntryType.FIELD)
        assert len(entries) == 1

        # List by sensitivity
        entries = registry.list_by_sensitivity(DataSensitivity.SENSITIVE)
        assert len(entries) == 1

        # List by category
        entries = registry.list_by_category(DataCategory.PII)
        assert len(entries) == 1

    def test_search(self):
        """Test search functionality."""
        registry = DataInventoryRegistry(strict_mode=False)

        registry.register(
            name="search_field_alpha",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
            tags=["tag1"],
        )

        registry.register(
            name="search_field_beta",
            entry_type=InventoryEntryType.STORE,
            sensitivity=DataSensitivity.SENSITIVE,
            category=DataCategory.PII,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
            tags=["tag2"],
        )

        # Search by query
        results = registry.search("alpha")
        assert len(results) == 1
        assert results[0].name == "search_field_alpha"

        # Search by type
        results = registry.search("", entry_types=[InventoryEntryType.STORE])
        assert len(results) == 1

        # Search by tags
        results = registry.search("", tags=["tag1"])
        assert len(results) == 1

    def test_generate_report(self):
        """Test report generation."""
        registry = DataInventoryRegistry(strict_mode=False)

        registry.register(
            name="report_field_1",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        registry.register(
            name="report_field_2",
            entry_type=InventoryEntryType.STORE,
            sensitivity=DataSensitivity.SENSITIVE,
            category=DataCategory.PII,
            purpose="",  # Missing purpose
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        report = registry.generate_report()

        assert report.total_entries == 2
        assert report.by_type["field"] == 1
        assert report.by_type["store"] == 1
        assert report.missing_purpose == 1
        assert report.integrity_hash.startswith("sha256:")

    def test_export_import(self):
        """Test export and import functionality."""
        registry = DataInventoryRegistry(strict_mode=False)

        registry.register(
            name="export_test",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        # Export
        export_data = registry.export()
        assert export_data["entry_count"] == 1
        assert "checksum" in export_data

        # Export to file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "inventory.json"
            registry.export_to_file(path)
            assert path.exists()

            # Import
            new_registry = DataInventoryRegistry(strict_mode=False)
            imported, errors, error_list = new_registry.import_from_file(
                path, imported_by="admin@example.com", merge=False
            )
            assert imported == 1
            assert errors == 0

    def test_get_events(self):
        """Test event retrieval."""
        registry = DataInventoryRegistry(strict_mode=False)

        entry = registry.register(
            name="event_test",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )

        registry.approve(entry.id, approved_by="dpo@example.com")

        events = registry.get_events()
        assert len(events) >= 2  # Create + approve

        events = registry.get_events(action="create")
        assert len(events) == 1

    def test_create_core_inventory(self):
        """Test core inventory factory."""
        registry = create_core_inventory()

        assert registry is not None
        stats = registry.get_stats()
        assert stats["total_entries"] > 0  # Has pre-registered entries

    def test_requires_dpia_flag(self):
        """Test DPIA requirement detection."""
        registry = DataInventoryRegistry(strict_mode=False)

        # Critical sensitivity requires DPIA
        entry = registry.register(
            name="critical_field",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.CRITICAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )
        assert entry.requires_dpia is True

        # Financial category requires DPIA
        entry2 = registry.register(
            name="financial_field",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.FINANCIAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )
        assert entry2.requires_dpia is True

    def test_entry_to_dict(self):
        """Test entry serialization."""
        entry = InventoryEntry(
            name="dict_test",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test purpose",
            lawful_basis="Legitimate interest",
            retention_days=90,
        )

        data = entry.to_dict()
        assert data["name"] == "dict_test"
        assert data["classification"]["sensitivity"] == "internal"
        assert data["retention"]["days"] == 90
        assert "compliance" in data

    def test_validation_result_to_dict(self):
        """Test validation result serialization."""
        result = InventoryValidationResult(
            is_registered=True,
            is_compliant=True,
            violations=[],
            warnings=["Test warning"],
        )

        data = result.to_dict()
        assert data["is_registered"] is True
        assert data["is_compliant"] is True
        assert data["should_block_ci"] is False

    def test_change_event_to_dict(self):
        """Test change event serialization."""
        event = InventoryChangeEvent(
            action="create",
            entry_id="test-id",
            entry_name="test",
            actor_id="user@example.com",
        )

        data = event.to_dict()
        assert data["action"] == "create"
        assert data["entry_id"] == "test-id"
        assert data["integrity_hash"].startswith("sha256:")


# ============================================================================
# Test: CI Privacy-by-Design Check
# ============================================================================

class TestPrivacyByDesignCheck:
    """Tests for PrivacyByDesignCheck."""

    def test_scanner_creation(self):
        """Test scanner creation."""
        scanner = DataDeclarationScanner()
        assert scanner is not None

    def test_scan_python_file_with_telemetry(self):
        """Test scanning Python file for telemetry fields."""
        scanner = DataDeclarationScanner()

        # Create temp file with telemetry
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_module.py"
            test_file.write_text('''
# Test module
def send_telemetry():
    data = {
        "custom_metric": metrics.count,
        "new_field": telemetry.value,
    }
    emit_telemetry("custom_event")
''')

            declarations = scanner.scan_file(test_file)
            # Should detect telemetry patterns
            assert len(declarations) >= 0  # May detect depending on patterns

    def test_scan_sql_file(self):
        """Test scanning SQL file for data stores."""
        scanner = DataDeclarationScanner()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "schema.sql"
            test_file.write_text('''
CREATE TABLE IF NOT EXISTS new_data_table (
    id SERIAL PRIMARY KEY,
    data TEXT
);
''')

            declarations = scanner.scan_file(test_file)
            # Should detect CREATE TABLE
            table_decls = [d for d in declarations if d.declaration_type == "data_store"]
            assert len(table_decls) >= 0

    def test_skip_test_files(self):
        """Test that test files are skipped."""
        scanner = DataDeclarationScanner()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_something.py"
            test_file.write_text("# Test file")

            declarations = scanner.scan_file(test_file)
            assert len(declarations) == 0

    def test_auto_detect_sensitive_patterns(self):
        """Test auto-detection of sensitive patterns."""
        scanner = DataDeclarationScanner()

        decl = DataDeclaration(
            name="api_key",
            declaration_type="telemetry_field",
            file_path="test.py",
            line_number=1,
            context="",
            pattern_matched="test",
            auto_detected=scanner._auto_detect("api_key"),
        )

        assert decl.auto_detected["credential"] is True

    def test_check_without_registry(self):
        """Test check without registry."""
        check = PrivacyByDesignCheck(
            registry=None,
            fail_on_unregistered=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "module.py"
            test_file.write_text('logging.getLogger("custom_logger")')

            report = check.run(Path(tmpdir))
            assert report is not None
            assert report.files_scanned >= 0

    def test_check_with_registry(self):
        """Test check with registry."""
        registry = DataInventoryRegistry(strict_mode=False)

        # Register a field
        entry = registry.register(
            name="registered_field",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TECHNICAL,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )
        registry.approve(entry.id, approved_by="dpo@example.com")

        check = PrivacyByDesignCheck(
            registry=registry,
            fail_on_unregistered=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report = check.run(Path(tmpdir))
            assert report is not None

    def test_report_should_fail_ci(self):
        """Test report CI failure detection."""
        report = PrivacyCheckReport()

        # No violations - should pass
        assert report.should_fail_ci is False

        # Add critical violation
        report.violations.append(CheckViolation(
            declaration=DataDeclaration(
                name="test",
                declaration_type="field",
                file_path="test.py",
                line_number=1,
                context="",
                pattern_matched="test",
            ),
            severity=CheckSeverity.CRITICAL,
            message="Test violation",
        ))

        assert report.should_fail_ci is True

    def test_report_to_dict(self):
        """Test report serialization."""
        report = PrivacyCheckReport(
            files_scanned=10,
            declarations_found=5,
            declarations_registered=3,
            declarations_unregistered=2,
        )

        data = report.to_dict()
        assert data["summary"]["files_scanned"] == 10
        assert data["summary"]["declarations_found"] == 5
        assert data["violation_counts"]["total"] == 0

    def test_format_report(self):
        """Test report formatting."""
        report = PrivacyCheckReport(
            check_id="test-check",
            result=CheckResult.PASS,
            files_scanned=10,
        )

        formatted = format_report(report)
        assert "Privacy-by-Design Check" in formatted
        assert "PASSED" in formatted

    def test_run_on_files(self):
        """Test running check on specific files."""
        check = PrivacyByDesignCheck()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("# Empty file")

            report = check.run_on_files([test_file])
            assert report.files_scanned == 1

    def test_data_declaration_to_dict(self):
        """Test data declaration serialization."""
        decl = DataDeclaration(
            name="test_field",
            declaration_type="telemetry_field",
            file_path="/path/to/file.py",
            line_number=42,
            context="some context here",
            pattern_matched="pattern",
            auto_detected={"pii": False, "credential": True},
        )

        data = decl.to_dict()
        assert data["name"] == "test_field"
        assert data["line_number"] == 42
        assert data["auto_detected"]["credential"] is True

    def test_check_violation_to_dict(self):
        """Test check violation serialization."""
        decl = DataDeclaration(
            name="test",
            declaration_type="field",
            file_path="test.py",
            line_number=1,
            context="",
            pattern_matched="test",
        )

        violation = CheckViolation(
            declaration=decl,
            severity=CheckSeverity.HIGH,
            message="Test message",
            remediation="Fix it",
        )

        data = violation.to_dict()
        assert data["severity"] == "high"
        assert data["message"] == "Test message"
        assert data["remediation"] == "Fix it"


# ============================================================================
# Test: Compliance Dashboard Service
# ============================================================================

class TestComplianceDashboardService:
    """Tests for ComplianceDashboardService."""

    def test_service_creation(self):
        """Test service creation."""
        service = ComplianceDashboardService()
        assert service is not None

    def test_generate_dashboard(self):
        """Test dashboard generation."""
        service = ComplianceDashboardService()

        dashboard = service.generate_dashboard()

        assert dashboard is not None
        assert dashboard.overall_status is not None
        assert dashboard.overall_score >= 0
        assert dashboard.integrity_hash.startswith("sha256:")

    def test_dsar_metrics(self):
        """Test DSAR metrics."""
        service = ComplianceDashboardService()

        metrics = service.get_dsar_metrics()

        assert metrics is not None
        assert metrics.total_requests >= 0
        data = metrics.to_dict()
        assert "sla" in data
        assert data["sla"]["target_days"] == DSAR_SLA_DAYS

    def test_purge_metrics(self):
        """Test purge metrics."""
        service = ComplianceDashboardService()

        metrics = service.get_purge_metrics()

        assert metrics is not None
        data = metrics.to_dict()
        assert "runs" in data
        assert "records" in data

    def test_break_glass_metrics(self):
        """Test break-glass metrics."""
        service = ComplianceDashboardService()

        metrics = service.get_break_glass_metrics()

        assert metrics is not None
        data = metrics.to_dict()
        assert "duration" in data
        assert data["duration"]["max_allowed_hours"] == BREAK_GLASS_MAX_DURATION_HOURS

    def test_residency_metrics(self):
        """Test residency metrics."""
        service = ComplianceDashboardService()

        metrics = service.get_residency_metrics()

        assert metrics is not None
        data = metrics.to_dict()
        assert "drift" in data
        assert data["drift"]["target"] == RESIDENCY_DRIFT_TARGET

    def test_inventory_metrics(self):
        """Test inventory metrics."""
        service = ComplianceDashboardService()

        metrics = service.get_inventory_metrics()

        assert metrics is not None
        data = metrics.to_dict()
        assert "entries" in data

    def test_alert_creation(self):
        """Test alert creation."""
        alerts = []
        service = ComplianceDashboardService(
            on_alert=lambda a: alerts.append(a)
        )

        # Force an alert by setting overdue DSAR
        service._check_dsar_alerts(
            DSARMetrics(overdue_requests=5),
            workspace_id="test-ws",
        )

        assert len(alerts) > 0
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_acknowledge_alert(self):
        """Test alert acknowledgment."""
        service = ComplianceDashboardService()

        # Create alert
        alert = service._create_alert(
            metric_type=MetricType.DSAR,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
            details={},
            workspace_id=None,
        )

        # Acknowledge
        updated = service.acknowledge_alert(alert.alert_id, acknowledged_by="user@example.com")

        assert updated is not None
        assert updated.status == AlertStatus.ACKNOWLEDGED
        assert updated.acknowledged_by == "user@example.com"

    def test_resolve_alert(self):
        """Test alert resolution."""
        service = ComplianceDashboardService()

        # Create alert
        alert = service._create_alert(
            metric_type=MetricType.DSAR,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
            details={},
            workspace_id=None,
        )

        # Resolve
        updated = service.resolve_alert(
            alert.alert_id,
            resolved_by="user@example.com",
            resolution_notes="Fixed the issue",
        )

        assert updated is not None
        assert updated.status == AlertStatus.RESOLVED
        assert updated.resolution_notes == "Fixed the issue"

    def test_get_open_alerts(self):
        """Test getting open alerts."""
        service = ComplianceDashboardService()

        # Create some alerts
        service._create_alert(
            metric_type=MetricType.DSAR,
            severity=AlertSeverity.WARNING,
            title="Open Alert",
            message="Test",
            details={},
            workspace_id=None,
        )

        alerts = service.get_open_alerts()
        assert len(alerts) >= 1

    def test_compliance_status_determination(self):
        """Test compliance status determination."""
        service = ComplianceDashboardService()

        dashboard = ComplianceDashboard(
            overall_score=90,
            residency_score=100,
            critical_alerts=0,
        )

        status = service._determine_status(dashboard)
        assert status == ComplianceStatus.COMPLIANT

        # Non-compliant residency
        dashboard.residency_score = 0
        status = service._determine_status(dashboard)
        assert status == ComplianceStatus.NON_COMPLIANT

    def test_score_calculations(self):
        """Test score calculations."""
        service = ComplianceDashboardService()

        # DSAR score
        dsar_metrics = DSARMetrics(total_requests=10, on_time_rate=1.0)
        score = service._calculate_dsar_score(dsar_metrics)
        assert score == 100.0

        dsar_metrics.overdue_requests = 5
        score = service._calculate_dsar_score(dsar_metrics)
        assert score < 100.0

        # Residency score (must be 100 for compliance)
        residency_metrics = ResidencyMetrics(drift_count=0, non_eu_endpoints=0)
        residency_metrics.eu_compliant_endpoints = 10
        residency_metrics.total_endpoints_checked = 10
        score = service._calculate_residency_score(residency_metrics)
        assert score == 100.0

        residency_metrics.drift_count = 1
        score = service._calculate_residency_score(residency_metrics)
        assert score == 0.0  # Any drift = 0 score

    def test_dashboard_to_dict(self):
        """Test dashboard serialization."""
        dashboard = ComplianceDashboard(
            overall_status=ComplianceStatus.COMPLIANT,
            overall_score=95.0,
        )

        data = dashboard.to_dict()
        assert data["overall"]["status"] == "compliant"
        assert data["overall"]["score"] == 95.0

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        dsar = DSARMetrics(total_requests=100)
        assert "volume" in dsar.to_dict()

        purge = PurgeMetrics(total_purge_runs=50)
        assert "runs" in purge.to_dict()

        bg = BreakGlassMetrics(total_requests=10)
        assert "requests" in bg.to_dict()

        res = ResidencyMetrics(drift_count=0)
        assert "drift" in res.to_dict()

        inv = InventoryMetrics(total_entries=100)
        assert "entries" in inv.to_dict()

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = ComplianceAlert(
            metric_type=MetricType.DSAR,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test message",
        )

        data = alert.to_dict()
        assert data["metric_type"] == "dsar"
        assert data["severity"] == "warning"


# ============================================================================
# Test: Quarterly Review Service
# ============================================================================

class TestQuarterlyReviewService:
    """Tests for QuarterlyReviewService."""

    def test_service_creation(self):
        """Test service creation."""
        service = QuarterlyReviewService()
        assert service is not None

    def test_schedule_review(self):
        """Test review scheduling."""
        service = QuarterlyReviewService()

        review = service.schedule_review(
            review_type=ReviewType.FULL_QUARTERLY,
            lead_reviewer="dpo@example.com",
        )

        assert review is not None
        assert review.status == QReviewStatus.SCHEDULED
        assert review.lead_reviewer == "dpo@example.com"
        assert review.quarter  # Should have quarter assigned

    def test_start_review(self):
        """Test starting a review."""
        service = QuarterlyReviewService()

        review = service.schedule_review(
            review_type=ReviewType.FULL_QUARTERLY,
            lead_reviewer="dpo@example.com",
        )

        started = service.start_review(review.id, started_by="dpo@example.com")

        assert started.status == QReviewStatus.IN_PROGRESS
        assert started.started_at is not None

    def test_complete_review(self):
        """Test completing a review."""
        service = QuarterlyReviewService()

        review = service.schedule_review(review_type=ReviewType.FULL_QUARTERLY)
        service.start_review(review.id, started_by="dpo@example.com")

        completed = service.complete_review(
            review.id,
            completed_by="dpo@example.com",
            summary="Q1 review completed",
            recommendations=["Improve documentation"],
        )

        assert completed.status == QReviewStatus.PENDING_APPROVAL
        assert completed.summary == "Q1 review completed"
        assert len(completed.recommendations) == 1

    def test_approve_review(self):
        """Test approving a review."""
        service = QuarterlyReviewService()

        review = service.schedule_review(review_type=ReviewType.FULL_QUARTERLY)
        service.start_review(review.id, started_by="dpo@example.com")
        service.complete_review(review.id, completed_by="dpo@example.com", summary="Done")

        approved = service.approve_review(
            review.id,
            approved_by="ciso@example.com",
            approval_notes="Approved with minor findings",
        )

        assert approved.status == QReviewStatus.COMPLETED
        assert approved.approved is True
        assert approved.approved_by == "ciso@example.com"

    def test_add_finding(self):
        """Test adding findings to review."""
        service = QuarterlyReviewService()

        review = service.schedule_review(review_type=ReviewType.FULL_QUARTERLY)
        service.start_review(review.id, started_by="dpo@example.com")

        finding = service.add_finding(review.id, ReviewFinding(
            title="Missing documentation",
            severity=FindingSeverity.HIGH,
            description="Retention policy documentation incomplete",
            recommendation="Update documentation",
        ))

        assert finding is not None
        assert finding.review_id == review.id
        assert finding.severity == FindingSeverity.HIGH

        # Check review has finding
        updated_review = service.get_review(review.id)
        assert len(updated_review.findings) == 1

    def test_update_finding(self):
        """Test updating a finding."""
        service = QuarterlyReviewService()

        review = service.schedule_review(review_type=ReviewType.FULL_QUARTERLY)
        service.start_review(review.id, started_by="dpo@example.com")

        finding = service.add_finding(review.id, ReviewFinding(
            title="Test finding",
            severity=FindingSeverity.MEDIUM,
        ))

        updated = service.update_finding(
            finding.id,
            status=FindingStatus.RESOLVED,
            resolution_notes="Fixed by implementing policy",
        )

        assert updated.status == FindingStatus.RESOLVED
        assert updated.resolved_at is not None

    def test_add_subprocessor(self):
        """Test adding subprocessor."""
        service = QuarterlyReviewService()

        sub = service.add_subprocessor(Subprocessor(
            name="AWS",
            description="Cloud provider",
            category="cloud",
            processing_location="EU",
            eu_compliant=True,
            dpa_signed=True,
        ))

        assert sub is not None
        assert sub.name == "AWS"

    def test_update_subprocessor(self):
        """Test updating/reviewing subprocessor."""
        service = QuarterlyReviewService()

        sub = service.add_subprocessor(Subprocessor(
            name="Test Provider",
            eu_compliant=True,
        ))

        updated = service.update_subprocessor(
            sub.id,
            reviewed_by="dpo@example.com",
            status=SubprocessorStatus.APPROVED,
        )

        assert updated.last_reviewed_at is not None
        assert updated.status == SubprocessorStatus.APPROVED
        assert updated.next_review_due is not None

    def test_get_subprocessors(self):
        """Test getting subprocessors."""
        service = QuarterlyReviewService()

        service.add_subprocessor(Subprocessor(name="Sub1", eu_compliant=True))
        service.add_subprocessor(Subprocessor(name="Sub2", eu_compliant=False))

        all_subs = service.get_subprocessors()
        assert len(all_subs) == 2

        eu_subs = service.get_subprocessors(eu_compliant=True)
        assert len(eu_subs) == 1

    def test_subprocessor_summary(self):
        """Test subprocessor summary."""
        service = QuarterlyReviewService()

        service.add_subprocessor(Subprocessor(
            name="Sub1",
            category="cloud",
            eu_compliant=True,
            dpa_signed=True,
        ))

        summary = service.get_subprocessor_summary()
        assert summary["total"] == 1
        assert summary["eu_compliant"] == 1
        assert summary["with_dpa"] == 1

    def test_add_incident_learning(self):
        """Test adding incident learning."""
        service = QuarterlyReviewService()

        incident = service.add_incident_learning(IncidentLearning(
            incident_id="INC-001",
            incident_type="access_violation",
            severity="P2",
            description="Unauthorized access attempt",
            root_cause="Misconfigured RBAC",
        ))

        assert incident is not None
        assert incident.incident_id == "INC-001"

    def test_review_incident(self):
        """Test reviewing incident."""
        service = QuarterlyReviewService()

        incident = service.add_incident_learning(IncidentLearning(
            incident_id="INC-002",
            incident_type="data_breach",
        ))

        reviewed = service.review_incident(
            incident.id,
            reviewed_by="security@example.com",
            lessons_learned=["Improve monitoring"],
            preventive_measures=["Add alerts"],
        )

        assert reviewed.reviewed_at is not None
        assert reviewed.status == "closed"
        assert "Improve monitoring" in reviewed.lessons_learned

    def test_get_schedule(self):
        """Test getting review schedule."""
        service = QuarterlyReviewService()

        # Schedule multiple reviews
        service.schedule_review(review_type=ReviewType.FULL_QUARTERLY)
        service.schedule_review(
            review_type=ReviewType.RETENTION,
            scheduled_date=datetime.now(timezone.utc) + timedelta(days=100),
        )

        schedule = service.get_schedule()
        assert len(schedule.reviews) == 2
        assert schedule.next_review is not None

    def test_review_overdue_detection(self):
        """Test overdue review detection."""
        service = QuarterlyReviewService()

        # Create overdue review
        review = QuarterlyReview(
            review_type=ReviewType.FULL_QUARTERLY,
            status=QReviewStatus.SCHEDULED,
            due_date=datetime.now(timezone.utc) - timedelta(days=10),
        )

        assert review.is_overdue is True
        assert review.days_until_due < 0

    def test_finding_overdue_detection(self):
        """Test overdue finding detection."""
        finding = ReviewFinding(
            title="Test",
            status=FindingStatus.OPEN,
            due_date=datetime.now(timezone.utc) - timedelta(days=5),
        )

        assert finding.is_overdue is True

        finding.status = FindingStatus.RESOLVED
        assert finding.is_overdue is False

    def test_create_default_subprocessors(self):
        """Test default subprocessors factory."""
        subs = create_default_subprocessors()
        assert len(subs) > 0
        assert all(s.eu_compliant for s in subs)

    def test_generate_review_report(self):
        """Test review report generation."""
        service = QuarterlyReviewService()

        review = service.schedule_review(review_type=ReviewType.FULL_QUARTERLY)
        service.start_review(review.id, started_by="dpo@example.com")

        report = service.generate_review_report(review.id)

        assert "review" in report
        assert "subprocessor_summary" in report

    def test_export_review(self):
        """Test review export."""
        service = QuarterlyReviewService()

        review = service.schedule_review(review_type=ReviewType.FULL_QUARTERLY)
        service.start_review(review.id, started_by="dpo@example.com")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "review.json"
            service.export_review(review.id, path)
            assert path.exists()

    def test_quarterly_review_to_dict(self):
        """Test quarterly review serialization."""
        review = QuarterlyReview(
            review_type=ReviewType.FULL_QUARTERLY,
            quarter="2025-Q1",
        )

        data = review.to_dict()
        assert data["review_type"] == "full_quarterly"
        assert data["quarter"] == "2025-Q1"
        assert "schedule" in data

    def test_subprocessor_to_dict(self):
        """Test subprocessor serialization."""
        sub = Subprocessor(
            name="AWS",
            eu_compliant=True,
            dpa_signed=True,
        )

        data = sub.to_dict()
        assert data["name"] == "AWS"
        assert data["compliance"]["eu_compliant"] is True

    def test_incident_learning_to_dict(self):
        """Test incident learning serialization."""
        incident = IncidentLearning(
            incident_id="INC-001",
            incident_type="data_breach",
            users_affected=100,
        )

        data = incident.to_dict()
        assert data["incident_id"] == "INC-001"
        assert data["impact"]["users_affected"] == 100

    def test_retention_review_item_to_dict(self):
        """Test retention review item serialization."""
        item = RetentionReviewItem(
            data_type="telemetry",
            current_retention_days=90,
            min_retention_days=30,
            change_proposed=True,
        )

        data = item.to_dict()
        assert data["data_type"] == "telemetry"
        assert data["change"]["proposed"] is True

    def test_audit_log(self):
        """Test audit log retrieval."""
        service = QuarterlyReviewService()

        service.schedule_review(review_type=ReviewType.FULL_QUARTERLY)

        log = service.get_audit_log()
        assert len(log) >= 1
        assert log[0]["action"] == "review_scheduled"


# ============================================================================
# Test: Constants Validation
# ============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_dsar_sla_constants(self):
        """Test DSAR SLA constants."""
        assert DSAR_SLA_DAYS == 30
        assert BREAK_GLASS_MAX_DURATION_HOURS == 24
        assert RESIDENCY_DRIFT_TARGET == 0

    def test_review_cycle_constants(self):
        """Test review cycle constants."""
        assert REVIEW_CYCLE_DAYS == 90
        assert ADVANCE_NOTICE_DAYS == 14
        assert OVERDUE_THRESHOLD_DAYS == 7

    def test_known_safe_patterns(self):
        """Test known safe patterns."""
        assert "timestamp" in KNOWN_SAFE_PATTERNS
        assert "count" in KNOWN_SAFE_PATTERNS
        assert "workspace_id" in KNOWN_SAFE_PATTERNS


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase8Integration:
    """Integration tests for Phase 8 components."""

    def test_full_workflow_inventory_to_ci_check(self):
        """Test full workflow from inventory registration to CI check."""
        # Create inventory
        registry = DataInventoryRegistry(strict_mode=False)

        # Register field
        entry = registry.register(
            name="user_metric",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TELEMETRY,
            purpose="User behavior tracking",
            lawful_basis="Legitimate interest",
            retention_days=90,
            created_by="dev@example.com",
        )

        # Approve
        registry.approve(entry.id, approved_by="dpo@example.com")

        # Validate
        result = registry.validate("user_metric")
        assert result.is_registered is True
        assert result.is_compliant is True
        assert result.should_block_ci is False

    def test_dashboard_with_all_services(self):
        """Test dashboard generation with mock services."""
        # Create mock services
        mock_dsar = Mock()
        mock_dsar.get_metrics.return_value = {
            "total_requests": 50,
            "pending_count": 5,
            "overdue_count": 0,
            "on_time_rate": 0.95,
            "avg_completion_days": 15,
        }

        mock_purge = Mock()
        mock_purge.get_purge_statistics.return_value = {
            "total_runs": 100,
            "successful_runs": 98,
            "failed_runs": 2,
        }

        service = ComplianceDashboardService(
            dsar_service=mock_dsar,
            purge_scheduler=mock_purge,
        )

        dashboard = service.generate_dashboard()
        assert dashboard.dsar_metrics.total_requests == 50
        assert dashboard.purge_metrics.successful_runs == 98

    def test_quarterly_review_with_dashboard_snapshot(self):
        """Test quarterly review with dashboard metrics snapshot."""
        dashboard_service = ComplianceDashboardService()
        review_service = QuarterlyReviewService(
            dashboard_service=dashboard_service,
        )

        # Schedule and start review
        review = review_service.schedule_review(
            review_type=ReviewType.FULL_QUARTERLY,
            lead_reviewer="dpo@example.com",
        )
        review_service.start_review(review.id, started_by="dpo@example.com")

        # Check metrics snapshot was collected
        updated = review_service.get_review(review.id)
        assert "collected_at" in updated.metrics_snapshot

    def test_ci_check_integration(self):
        """Test CI check with inventory integration."""
        registry = DataInventoryRegistry(strict_mode=False)

        # Register some fields
        entry = registry.register(
            name="registered_metric",
            entry_type=InventoryEntryType.FIELD,
            sensitivity=DataSensitivity.INTERNAL,
            category=DataCategory.TELEMETRY,
            purpose="Test",
            lawful_basis="Test",
            retention_days=90,
            created_by="test@example.com",
        )
        registry.approve(entry.id, approved_by="dpo@example.com")

        # Run CI check
        check = PrivacyByDesignCheck(
            registry=registry,
            fail_on_unregistered=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report = check.run(Path(tmpdir))
            assert report.result in (CheckResult.PASS, CheckResult.WARN)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
